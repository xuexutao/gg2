# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from scipy.spatial import cKDTree

# ============================================================================
# Anisotropic Affinity utilities (Breakthrough 1)
# ----------------------------------------------------------------------------
# Existing 3DGS grouping methods (Gaussian Grouping, SAGA, Click-Gaussian, ...)
# treat each Gaussian as a point and use Euclidean KNN on xyz for the 3D
# regularization loss. This throws away the *anisotropy* encoded in the
# covariance matrix Σ = R diag(s^2) R^T which is the core expressive advantage
# of 3DGS. We rescue that information here.
# ============================================================================


def _quat_to_rotmat(q):
    """Convert normalized quaternion (N,4) in (w, x, y, z) order to (N,3,3)."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = q.shape[0]
    R = torch.empty(N, 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def gaussian_normals(scaling, rotation):
    """
    Extract per-Gaussian surface normal as the eigenvector of Σ with smallest
    eigenvalue. For Σ = R diag(s^2) R^T the eigenvectors are the columns of R
    and the eigenvalues are s^2 (elementwise). So the normal is the column of
    R corresponding to argmin(s).

    Args:
        scaling:  (N, 3) already activated scales (output of gaussians.get_scaling)
        rotation: (N, 4) raw quaternion (gaussians._rotation); will be normalized.
    Returns:
        normals:  (N, 3) unit vectors.
    """
    R = _quat_to_rotmat(rotation)  # (N, 3, 3)
    min_axis = torch.argmin(scaling, dim=-1)  # (N,)
    idx = min_axis.view(-1, 1, 1).expand(-1, 3, 1)  # (N, 3, 1)
    normals = torch.gather(R, 2, idx).squeeze(-1)  # (N, 3)
    normals = F.normalize(normals, dim=-1)
    return normals


def bhattacharyya_distance(
    mu_i, mu_j, scaling_i, scaling_j, rotation_i, rotation_j, eps=1e-6
):
    """
    Bhattacharyya distance between two 3D Gaussians.
    D_B = 1/8 (μ_i - μ_j)^T Σ^-1 (μ_i - μ_j) + 1/2 log(|Σ| / sqrt(|Σ_i||Σ_j|))
    where Σ = (Σ_i + Σ_j) / 2.

    For efficiency we only compute the Mahalanobis component w.r.t. the
    averaged covariance, which is the dominant term for grouping purposes and
    is symmetric, differentiable, and cheap.

    All inputs are expected to have matching leading dim M, e.g.
        mu_i: (M, 3)          the sample anchor means
        mu_j: (M, K, 3)       the K candidate neighbors for each anchor
    Returns (M, K) Bhattacharyya distances.
    """
    # This helper is not actually used on full pairs (too expensive); kept as
    # reference. The production path uses `mahalanobis_dist_fast` below.
    raise NotImplementedError("Use mahalanobis_dist_fast for the Top-K search.")


def _build_covariance(scaling, rotation):
    """Build (N,3,3) covariance from per-Gaussian scaling (N,3) and quaternion (N,4)."""
    R = _quat_to_rotmat(rotation)  # (N, 3, 3)
    S = torch.diag_embed(scaling * scaling)  # (N, 3, 3)  diag(s^2)
    cov = R @ S @ R.transpose(-1, -2)
    return cov


def mahalanobis_dist_fast(mu_a, cov_a, mu_b, cov_b, eps=1e-6):
    """
    Symmetric Mahalanobis-like distance between two *sets* of Gaussians.
    d(i,j) = sqrt( 0.5 * (Δ^T Σ_i^{-1} Δ + Δ^T Σ_j^{-1} Δ) ), Δ = μ_i - μ_j.

    This is symmetric and accounts for both covariances, which is what we want
    for "do these two Gaussians belong to the same surface?" rather than the
    asymmetric M(i,j) ≠ M(j,i) case.

    Args:
        mu_a:  (M, 3)      anchor means
        cov_a: (M, 3, 3)   anchor covariances
        mu_b:  (N, 3)      full-set means
        cov_b: (N, 3, 3)   full-set covariances
    Returns:
        (M, N) distance matrix.
    """
    M = mu_a.shape[0]
    N = mu_b.shape[0]
    # Δ : (M, N, 3)
    delta = mu_a.unsqueeze(1) - mu_b.unsqueeze(0)

    # Σ^-1 with small regularization
    I3 = torch.eye(3, device=mu_a.device, dtype=mu_a.dtype).expand_as(cov_a)
    cov_a_inv = torch.linalg.inv(cov_a + eps * I3)  # (M, 3, 3)
    I3b = torch.eye(3, device=mu_a.device, dtype=mu_a.dtype).expand_as(cov_b)
    cov_b_inv = torch.linalg.inv(cov_b + eps * I3b)  # (N, 3, 3)

    # term_a[m,n] = Δ[m,n]^T Σ_a_inv[m] Δ[m,n]    -> (M, N)
    # Use einsum to avoid building (M,N,3,3).
    term_a = torch.einsum("mnd,mde,mne->mn", delta, cov_a_inv, delta)
    term_b = torch.einsum("mnd,nde,mne->mn", delta, cov_b_inv, delta)

    d2 = 0.5 * (term_a + term_b)
    d2 = torch.clamp(d2, min=0.0)
    return torch.sqrt(d2 + eps)


def loss_cls_3d_aniso(
    xyz,
    scaling,
    rotation,
    predictions,
    k=5,
    lambda_val=2.0,
    max_points=200000,
    sample_size=800,
    coarse_k=64,
    normal_weight=0.0,
    normal_only_same_group=True,
    eps=1e-6,
):
    """
    Anisotropic Affinity 3D regularization loss — our Breakthrough-1 replacement
    of `loss_cls_3d` in Gaussian Grouping.

    Pipeline:
      1. Sub-sample `max_points` Gaussians to keep compute tractable.
      2. Randomly draw `sample_size` anchor Gaussians.
      3. Two-stage neighbor search:
           (a) coarse_k nearest neighbors by Euclidean distance on μ (cheap),
           (b) re-rank them by symmetric Mahalanobis distance on full Σ,
               take the top-k closest as the final neighbor set.
         This gives O(sample_size * coarse_k) Mahalanobis evaluations instead
         of O(sample_size * N), keeping the loss cheap.
      4. KL divergence between anchor identity distribution and neighbor
         identity distributions (same as original).
      5. (optional) Normal Consistency loss on the resulting neighborhood:
         anchor and neighbor normals should agree up to sign.

    This is a drop-in replacement for the original `loss_cls_3d`; the only
    new inputs are `scaling`, `rotation`, and the hyper-params.
    """
    device = xyz.device
    N_total = xyz.shape[0]

    # --- Step 1: optional down-sample over all Gaussians ---
    if N_total > max_points:
        perm = torch.randperm(N_total, device=device)[:max_points]
        xyz = xyz[perm]
        scaling = scaling[perm]
        rotation = rotation[perm]
        predictions = predictions[perm]

    N = xyz.shape[0]

    # --- Step 2: anchors ---
    idx_anchor = torch.randperm(N, device=device)[:sample_size]
    xyz_a = xyz[idx_anchor]
    scl_a = scaling[idx_anchor]
    rot_a = rotation[idx_anchor]
    pred_a = predictions[idx_anchor]

    # --- Step 3a: cheap coarse Euclidean KNN to prune candidate set ---
    with torch.no_grad():
        dists_eu = torch.cdist(xyz_a, xyz)
        coarse_k_eff = min(coarse_k, N)
        _, coarse_idx = dists_eu.topk(coarse_k_eff, largest=False)  # (sample, coarse_k)

    # --- Step 3b: Mahalanobis re-ranking among the coarse candidates ---
    # Build covariances only for anchors and the coarse candidate union.
    cov_a = _build_covariance(scl_a, rot_a)  # (sample, 3, 3)

    # Gather per-anchor candidate covariances:
    # shape (sample, coarse_k, ...). We flatten to compute once per unique
    # candidate to save memory for the simplest implementation we pay per-pair.
    scl_cand = scaling[coarse_idx]  # (sample, coarse_k, 3)
    rot_cand = rotation[coarse_idx]  # (sample, coarse_k, 4)
    xyz_cand = xyz[coarse_idx]  # (sample, coarse_k, 3)

    M = sample_size if sample_size <= N else N
    Ck = coarse_k_eff

    R_cand = _quat_to_rotmat(rot_cand.reshape(-1, 4)).reshape(M, Ck, 3, 3)
    S_cand = torch.diag_embed((scl_cand * scl_cand))
    cov_cand = R_cand @ S_cand @ R_cand.transpose(-1, -2)  # (M, Ck, 3, 3)

    # Δ: (M, Ck, 3)
    delta = xyz_a.unsqueeze(1) - xyz_cand
    I3 = torch.eye(3, device=device, dtype=xyz_a.dtype)
    cov_a_inv = torch.linalg.inv(cov_a + eps * I3)  # (M, 3, 3)
    cov_cand_inv = torch.linalg.inv(cov_cand + eps * I3)  # (M, Ck, 3, 3)

    term_a = torch.einsum("mcd,mde,mce->mc", delta, cov_a_inv, delta)
    term_b = torch.einsum("mcd,mcde,mce->mc", delta, cov_cand_inv, delta)
    d_maha = torch.sqrt(torch.clamp(0.5 * (term_a + term_b), min=0.0) + eps)  # (M, Ck)

    k_eff = min(k, Ck)
    _, topk_in_coarse = d_maha.topk(k_eff, largest=False)  # (M, k)
    # Map back to global indices in the down-sampled space:
    row_idx = torch.arange(M, device=device).unsqueeze(1).expand(-1, k_eff)
    neighbor_indices = coarse_idx[row_idx, topk_in_coarse]  # (M, k)

    # --- Step 4: KL divergence (same as original) ---
    neighbor_preds = predictions[neighbor_indices]  # (M, k, C)
    kl = pred_a.unsqueeze(1) * (
        torch.log(pred_a.unsqueeze(1) + 1e-10) - torch.log(neighbor_preds + 1e-10)
    )
    kl_loss = kl.sum(dim=-1).mean()
    num_classes = predictions.size(1)
    kl_loss = kl_loss / num_classes

    total = lambda_val * kl_loss

    # --- Step 5: Normal Consistency Loss (optional) ---
    if normal_weight > 0.0:
        normals = gaussian_normals(scaling, rotation)  # (N, 3)
        n_a = normals[idx_anchor]  # (M, 3)
        n_nb = normals[neighbor_indices]  # (M, k, 3)
        cos_abs = torch.abs((n_a.unsqueeze(1) * n_nb).sum(dim=-1))  # (M, k) in [0,1]

        if normal_only_same_group:
            # Weight by soft same-group probability based on class distributions:
            # w = sum_c p_anchor(c) * p_neighbor(c)  in [0, 1]
            same_prob = (pred_a.unsqueeze(1) * neighbor_preds).sum(dim=-1)  # (M, k)
            normal_loss = ((1.0 - cos_abs) * same_prob).sum() / (same_prob.sum() + 1e-6)
        else:
            normal_loss = (1.0 - cos_abs).mean()

        total = total + normal_weight * normal_loss

    return total


# ============================================================================
# Uncertainty-Aware Grouping (Breakthrough F)
# ----------------------------------------------------------------------------
# Motivation:
#   The anisotropy of a Gaussian's covariance Σ carries a natural measure of
#   "how confident should we be about this Gaussian's identity?":
#     * A very flat Gaussian (λ_min ≪ λ_max) sits on a surface → its normal is
#       well-defined → nearby Gaussians on the same surface should share its
#       identity. High confidence.
#     * A nearly-round Gaussian (λ_min ≈ λ_max) has no preferred direction →
#       it either sits in a volume (rare in a converged 3DGS) or is a
#       under-optimized/floating Gaussian. Low confidence.
#
# We use this signal in two ways:
#   (1) downweighting uncertain Gaussians in the 3D neighbor KL term
#       (implemented below in `loss_cls_3d_aniso_uncertain`);
#   (2) downweighting pixels whose rendered identity distribution is near-
#       uniform in the 2D CE loss (implemented in `pixel_entropy_weight`).
#
# Both use a *multiplicative* weight in [0, 1], so setting all weights to 1
# exactly recovers the original loss.
# ============================================================================


def compute_anisotropy(scaling, mode="ratio", eps=1e-8):
    """
    Per-Gaussian anisotropy score in [0, 1].

    A flat Gaussian (λ_min ≪ λ_max) → score ~ 1  → confident identity.
    A spherical Gaussian (λ_min ≈ λ_max) → score ~ 0 → uncertain identity.

    Args:
        scaling: (N, 3) already-activated scales (pc.get_scaling).
        mode:
            "ratio": 1 - λ_min / λ_max  (simple, bounded, most stable)
            "fa":    fractional anisotropy, 3D generalization of FA used in
                     diffusion MRI. Sensitive to full spectrum shape.
    Returns:
        (N,) tensor with values in [0, 1].
    """
    # Note: Gaussian Grouping uses scaling to represent *standard deviations*,
    # so eigenvalues of Σ are scaling^2. We operate on scaling directly — the
    # ratio is invariant under squaring.
    s_max = scaling.max(dim=-1).values
    s_min = scaling.min(dim=-1).values

    if mode == "ratio":
        return 1.0 - s_min / (s_max + eps)
    elif mode == "fa":
        # Fractional anisotropy: √(3/2) * std(λ) / ||λ||
        lam = scaling * scaling  # eigenvalues of Σ
        mean = lam.mean(dim=-1, keepdim=True)
        var = ((lam - mean) ** 2).mean(dim=-1)
        norm = (lam * lam).sum(dim=-1)
        fa = torch.sqrt(1.5 * var / (norm + eps))
        return torch.clamp(fa, 0.0, 1.0)
    else:
        raise ValueError(f"unknown anisotropy mode: {mode}")


def pixel_entropy_weight(logits, temperature=1.0, mode="entropy"):
    """
    Per-pixel confidence weight computed from the classifier logits over the
    rendered 2D identity feature. Low entropy (peaky softmax) = confident.

    Args:
        logits: (C, H, W) classifier output on the rendered identity feature.
        temperature: softmax temperature; >1 softens, <1 sharpens.
        mode:
            "entropy": w = 1 - H(softmax(logits)) / log(C)     ∈ [0, 1]
            "max":     w = softmax(logits).max(dim=0)          ∈ [1/C, 1]
    Returns:
        (H, W) weight tensor, detached from the classifier to avoid degenerate
        gradients that would push the classifier toward always-confident
        distributions.
    """
    C = logits.shape[0]
    probs = torch.softmax(logits / max(temperature, 1e-6), dim=0)  # (C, H, W)
    if mode == "max":
        w = probs.max(dim=0).values
    else:  # entropy-based (default)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=0)  # (H, W)
        w = 1.0 - entropy / math.log(C)
        w = torch.clamp(w, 0.0, 1.0)
    return w.detach()


def weighted_2d_ce_loss(
    cls_criterion, logits, gt_obj, weight_map, num_classes, min_weight=0.5
):
    """
    Uncertainty-weighted 2D cross-entropy for the identity classifier.

    NOTE on `min_weight`: with a large number of classes (e.g. C=256 in LERF-
    Mask), the entropy of an uninformative softmax is close to log(C) ≈ 5.54,
    so early in training almost every pixel gets a very low weight and the
    classifier never learns. We therefore default to `min_weight=0.5` — this
    still doubles the influence of confident pixels relative to uncertain
    ones, but every pixel receives at least half the original gradient so
    training does not starve. If you need a stronger reweighting schedule,
    raise it gradually from 1.0 to e.g. 0.3 as training progresses.

    Args:
        cls_criterion: torch.nn.CrossEntropyLoss(reduction='none')
        logits: (C, H, W) or (N, C, H, W); the classifier output
        gt_obj: (H, W) long  or (N, H, W) long; GT identity IDs
        weight_map: (H, W) or (N, H, W) in [0, 1]; per-pixel confidence
        num_classes: normalization constant (same as original)
        min_weight: floor on the weight so that *every* pixel still contributes
                    a little, avoiding total gradient starvation on uncertain
                    regions.

    Returns:
        Scalar loss, normalized the same way as the original.
    """
    if logits.dim() == 3:
        logits_b = logits.unsqueeze(0)
        gt_b = gt_obj.unsqueeze(0)
    else:
        logits_b = logits
        gt_b = gt_obj
    ce = cls_criterion(logits_b, gt_b)  # (N, H, W) if logits_b was (N, C, H, W)
    ce = ce.squeeze()  # (H, W) or (N, H, W)

    w = weight_map.squeeze()
    w = torch.clamp(w, min=min_weight, max=1.0)

    num = (ce * w).sum()
    den = w.sum().clamp(min=1e-6)
    loss = num / den
    return loss / torch.log(torch.tensor(float(num_classes), device=loss.device))


def loss_cls_3d_aniso_uncertain(
    xyz,
    scaling,
    rotation,
    predictions,
    k=5,
    lambda_val=2.0,
    max_points=200000,
    sample_size=800,
    coarse_k=64,
    normal_weight=0.0,
    normal_only_same_group=True,
    anisotropy_mode="ratio",
    use_anchor_weight=True,
    use_neighbor_weight=True,
    eps=1e-6,
):
    """
    Uncertainty-aware extension of `loss_cls_3d_aniso`.

    In addition to Anisotropic Affinity (C1) and Normal Consistency (part of
    C1), this function weights each (anchor, neighbor) KL contribution by the
    *anisotropy-derived confidence* of the involved Gaussians. The intuition:

        - An anchor that is poorly anisotropic is likely a floater; we should
          not strongly enforce its neighbors to match its identity distribution.
        - A neighbor that is poorly anisotropic carries a noisy identity; it
          should not drag an otherwise-coherent anchor.

    Setting use_anchor_weight=False and use_neighbor_weight=False recovers
    `loss_cls_3d_aniso` exactly (a useful baseline).

    Extra args vs. `loss_cls_3d_aniso`:
        anisotropy_mode: "ratio" | "fa"
        use_anchor_weight:   whether to multiply by anisotropy(anchor)
        use_neighbor_weight: whether to multiply by anisotropy(neighbor)
    """
    device = xyz.device
    N_total = xyz.shape[0]

    # --- Step 1: optional down-sample over all Gaussians ---
    if N_total > max_points:
        perm = torch.randperm(N_total, device=device)[:max_points]
        xyz = xyz[perm]
        scaling = scaling[perm]
        rotation = rotation[perm]
        predictions = predictions[perm]

    N = xyz.shape[0]

    # --- Pre-compute per-Gaussian anisotropy weight ---
    aniso_w = compute_anisotropy(scaling, mode=anisotropy_mode)  # (N,) in [0,1]

    # --- Step 2: anchors ---
    idx_anchor = torch.randperm(N, device=device)[:sample_size]
    xyz_a = xyz[idx_anchor]
    scl_a = scaling[idx_anchor]
    rot_a = rotation[idx_anchor]
    pred_a = predictions[idx_anchor]
    w_a = aniso_w[idx_anchor]  # (M,)

    # --- Step 3a: cheap coarse Euclidean KNN ---
    with torch.no_grad():
        dists_eu = torch.cdist(xyz_a, xyz)
        coarse_k_eff = min(coarse_k, N)
        _, coarse_idx = dists_eu.topk(coarse_k_eff, largest=False)

    # --- Step 3b: Mahalanobis re-ranking among the coarse candidates ---
    cov_a = _build_covariance(scl_a, rot_a)  # (M, 3, 3)

    scl_cand = scaling[coarse_idx]  # (M, Ck, 3)
    rot_cand = rotation[coarse_idx]
    xyz_cand = xyz[coarse_idx]

    M = sample_size if sample_size <= N else N
    Ck = coarse_k_eff

    R_cand = _quat_to_rotmat(rot_cand.reshape(-1, 4)).reshape(M, Ck, 3, 3)
    S_cand = torch.diag_embed((scl_cand * scl_cand))
    cov_cand = R_cand @ S_cand @ R_cand.transpose(-1, -2)

    delta = xyz_a.unsqueeze(1) - xyz_cand
    I3 = torch.eye(3, device=device, dtype=xyz_a.dtype)
    cov_a_inv = torch.linalg.inv(cov_a + eps * I3)
    cov_cand_inv = torch.linalg.inv(cov_cand + eps * I3)

    term_a = torch.einsum("mcd,mde,mce->mc", delta, cov_a_inv, delta)
    term_b = torch.einsum("mcd,mcde,mce->mc", delta, cov_cand_inv, delta)
    d_maha = torch.sqrt(torch.clamp(0.5 * (term_a + term_b), min=0.0) + eps)

    k_eff = min(k, Ck)
    _, topk_in_coarse = d_maha.topk(k_eff, largest=False)
    row_idx = torch.arange(M, device=device).unsqueeze(1).expand(-1, k_eff)
    neighbor_indices = coarse_idx[row_idx, topk_in_coarse]

    neighbor_preds = predictions[neighbor_indices]  # (M, k, C)
    w_nb = aniso_w[neighbor_indices]  # (M, k)

    # --- Step 4: weighted KL divergence ---
    # KL(p_anchor || p_neighbor), reshape to (M, k)
    kl_per_pair = (
        pred_a.unsqueeze(1)
        * (torch.log(pred_a.unsqueeze(1) + 1e-10) - torch.log(neighbor_preds + 1e-10))
    ).sum(dim=-1)  # (M, k)

    # Build per-pair confidence weight, floored so no pair has zero weight.
    pair_w = torch.ones_like(kl_per_pair)
    if use_anchor_weight:
        pair_w = pair_w * w_a.unsqueeze(1)
    if use_neighbor_weight:
        pair_w = pair_w * w_nb
    pair_w = torch.clamp(pair_w, min=0.05, max=1.0).detach()

    num = (kl_per_pair * pair_w).sum()
    den = pair_w.sum().clamp(min=1e-6)
    kl_loss = num / den
    num_classes = predictions.size(1)
    kl_loss = kl_loss / num_classes

    total = lambda_val * kl_loss

    # --- Step 5: Normal Consistency (weighted) ---
    if normal_weight > 0.0:
        normals = gaussian_normals(scaling, rotation)
        n_a = normals[idx_anchor]
        n_nb = normals[neighbor_indices]
        cos_abs = torch.abs((n_a.unsqueeze(1) * n_nb).sum(dim=-1))  # (M, k)

        if normal_only_same_group:
            same_prob = (pred_a.unsqueeze(1) * neighbor_preds).sum(dim=-1)
            w_combo = same_prob * pair_w
            normal_loss = ((1.0 - cos_abs) * w_combo).sum() / (w_combo.sum() + 1e-6)
        else:
            normal_loss = ((1.0 - cos_abs) * pair_w).sum() / (pair_w.sum() + 1e-6)

        total = total + normal_weight * normal_loss

    return total


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def masked_l1_loss(network_output, gt, mask):
    mask = mask.float()[None, :, :].repeat(gt.shape[0], 1, 1)
    loss = torch.abs((network_output - gt)) * mask
    loss = loss.sum() / mask.sum()
    return loss


def weighted_l1_loss(network_output, gt, weight):
    loss = torch.abs((network_output - gt)) * weight
    return loss.mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def loss_cls_3d(
    features, predictions, k=5, lambda_val=2.0, max_points=200000, sample_size=800
):
    """
    Compute the neighborhood consistency loss for a 3D point cloud using Top-k neighbors
    and the KL divergence.

    :param features: Tensor of shape (N, D), where N is the number of points and D is the dimensionality of the feature.
    :param predictions: Tensor of shape (N, C), where C is the number of classes.
    :param k: Number of neighbors to consider.
    :param lambda_val: Weighting factor for the loss.
    :param max_points: Maximum number of points for downsampling. If the number of points exceeds this, they are randomly downsampled.
    :param sample_size: Number of points to randomly sample for computing the loss.

    :return: Computed loss value.
    """
    # Conditionally downsample if points exceed max_points
    if features.size(0) > max_points:
        indices = torch.randperm(features.size(0))[:max_points]
        features = features[indices]
        predictions = predictions[indices]

    # Randomly sample points for which we'll compute the loss
    indices = torch.randperm(features.size(0))[:sample_size]
    sample_features = features[indices]
    sample_preds = predictions[indices]

    # Compute top-k nearest neighbors directly in PyTorch
    dists = torch.cdist(sample_features, features)  # Compute pairwise distances
    _, neighbor_indices_tensor = dists.topk(
        k, largest=False
    )  # Get top-k smallest distances

    # Fetch neighbor predictions using indexing
    neighbor_preds = predictions[neighbor_indices_tensor]

    # Compute KL divergence
    kl = sample_preds.unsqueeze(1) * (
        torch.log(sample_preds.unsqueeze(1) + 1e-10) - torch.log(neighbor_preds + 1e-10)
    )
    loss = kl.sum(dim=-1).mean()

    # Normalize loss into [0, 1]
    num_classes = predictions.size(1)
    normalized_loss = loss / num_classes

    return lambda_val * normalized_loss
