"""
Unit tests for the Uncertainty-Aware Grouping additions (Breakthrough F):
    * compute_anisotropy
    * pixel_entropy_weight
    * weighted_2d_ce_loss
    * loss_cls_3d_aniso_uncertain

Run:
    cd /Users/bytedance/demo/shili_51/paper/gaussian-grouping
    python -m tests.test_uncertainty
"""

import math
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.loss_utils import (
    compute_anisotropy,
    pixel_entropy_weight,
    weighted_2d_ce_loss,
    loss_cls_3d_aniso,
    loss_cls_3d_aniso_uncertain,
)


def _identity_quat(n, device):
    q = torch.zeros(n, 4, device=device)
    q[:, 0] = 1.0
    return q


def test_anisotropy_range():
    """anisotropy ∈ [0, 1]; spherical ≈ 0, flat ≈ 1."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaling = torch.tensor(
        [
            [1.0, 1.0, 1.0],  # sphere → 0
            [1.0, 1.0, 0.001],  # flat disk → ~1
            [0.5, 0.5, 0.5],  # sphere (different size) → 0
            [1.0, 0.5, 0.1],  # elongated → ~0.9
        ],
        device=device,
    )
    ratio = compute_anisotropy(scaling, mode="ratio")
    fa = compute_anisotropy(scaling, mode="fa")

    assert abs(ratio[0].item()) < 1e-4, f"Sphere anisotropy should be 0, got {ratio[0]}"
    assert ratio[1].item() > 0.99, f"Flat anisotropy should be ~1, got {ratio[1]}"
    assert abs(ratio[2].item()) < 1e-4, f"Sphere anisotropy should be 0, got {ratio[2]}"
    assert ratio[3].item() > 0.8, f"Elongated anisotropy should be high, got {ratio[3]}"
    assert (fa >= 0).all() and (fa <= 1).all()
    print(f"[OK] test_anisotropy_range  ratio={ratio.tolist()}  fa={fa.tolist()}")


def test_pixel_entropy_weight():
    """Peaky softmax → weight close to 1; uniform softmax → weight close to 0."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    C, H, W = 8, 4, 4
    # Peaky logits at pixel (0,0): class 0 strongly preferred
    logits = torch.zeros(C, H, W, device=device)
    logits[0, 0, 0] = 50.0
    # Uniform logits at pixel (1,1)
    # Mixed at others
    w = pixel_entropy_weight(logits, temperature=1.0, mode="entropy")
    assert w.shape == (H, W)
    assert w[0, 0].item() > 0.99, f"Peaky pixel should be confident, got {w[0, 0]}"
    assert abs(w[1, 1].item()) < 1e-3, f"Uniform pixel should be 0, got {w[1, 1]}"
    assert not w.requires_grad, "weight map should be detached"
    print(f"[OK] test_pixel_entropy_weight  peaky={w[0, 0]:.4f}  uniform={w[1, 1]:.4f}")


def test_weighted_2d_ce_loss_recovers_mean_when_uniform_weights():
    """When all weights are 1, the weighted CE should equal the standard one
    (up to the same normalization)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    C, H, W = 4, 5, 5
    logits = torch.randn(C, H, W, device=device)
    gt = torch.randint(0, C, (H, W), device=device).long()
    weight_map = torch.ones(H, W, device=device)

    cls_criterion = torch.nn.CrossEntropyLoss(reduction="none")

    # Weighted (uniform) version
    w_loss = weighted_2d_ce_loss(
        cls_criterion, logits, gt, weight_map, num_classes=C, min_weight=0.1
    )

    # Original version
    ce = cls_criterion(logits.unsqueeze(0), gt.unsqueeze(0)).squeeze().mean()
    orig_loss = ce / torch.log(torch.tensor(float(C), device=device))

    diff = abs(w_loss.item() - orig_loss.item())
    assert diff < 1e-5, f"Uniform-weighted should match original, diff={diff}"
    print(f"[OK] test_weighted_2d_ce_loss_recovers_mean_when_uniform_weights")


def test_weighted_2d_ce_loss_downweights_uncertain_pixels():
    """If we intentionally put high weight only on pixels where the prediction
    is correct, the loss should drop. Sanity-check that weights actually
    change the loss in the expected direction."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    C, H, W = 4, 8, 8
    logits = torch.randn(C, H, W, device=device) * 0.1
    gt = torch.randint(0, C, (H, W), device=device).long()
    # Make half the pixels trivially correct: boost the GT class by 50
    half = H // 2
    for y in range(half):
        for x in range(W):
            logits[gt[y, x], y, x] += 50.0

    # Uniform weights
    w_uniform = torch.ones(H, W, device=device)
    # Weights that emphasize the "correct" half
    w_correct = torch.zeros(H, W, device=device)
    w_correct[:half, :] = 1.0

    cls_criterion = torch.nn.CrossEntropyLoss(reduction="none")
    loss_uniform = weighted_2d_ce_loss(
        cls_criterion, logits, gt, w_uniform, num_classes=C, min_weight=0.0
    )
    loss_correct = weighted_2d_ce_loss(
        cls_criterion, logits, gt, w_correct, num_classes=C, min_weight=0.0
    )
    assert loss_correct.item() < loss_uniform.item(), (
        f"Correct-weighted loss should be lower; got {loss_correct} vs {loss_uniform}"
    )
    print(
        f"[OK] test_weighted_2d_ce_loss_downweights_uncertain_pixels  "
        f"correct={loss_correct.item():.4f} < uniform={loss_uniform.item():.4f}"
    )


def test_uncertain_loss_recovers_aniso_when_weights_off():
    """Setting use_anchor_weight=False and use_neighbor_weight=False should
    make `loss_cls_3d_aniso_uncertain` numerically close to
    `loss_cls_3d_aniso` (not exactly equal because of denominator change from
    `.mean()` → `sum/sum(w)` — but with all-uniform weights they are equal)."""
    torch.manual_seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N, C = 400, 8
    xyz = torch.randn(N, 3, device=device)
    scaling = torch.full((N, 3), 0.05, device=device)  # isotropic
    rotation = _identity_quat(N, device)
    preds = torch.softmax(torch.randn(N, C, device=device), dim=-1)

    common = dict(
        k=5,
        lambda_val=2.0,
        max_points=N,
        sample_size=128,
        coarse_k=64,
        normal_weight=0.0,
    )
    torch.manual_seed(7)
    ref = loss_cls_3d_aniso(xyz, scaling, rotation, preds, **common)
    torch.manual_seed(7)
    new = loss_cls_3d_aniso_uncertain(
        xyz,
        scaling,
        rotation,
        preds,
        use_anchor_weight=False,
        use_neighbor_weight=False,
        **common,
    )
    ratio = new.item() / max(ref.item(), 1e-8)
    assert 0.9 < ratio < 1.1, f"Should recover aniso loss; ratio={ratio}"
    print(
        f"[OK] test_uncertain_loss_recovers_aniso_when_weights_off  "
        f"ref={ref.item():.4f}  new={new.item():.4f}  ratio={ratio:.3f}"
    )


def test_uncertain_loss_differs_with_mixed_anisotropy():
    """When half the Gaussians are flat and half spherical, enabling anchor +
    neighbor weights should change the loss (sanity check that weights matter)."""
    torch.manual_seed(8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N, C = 400, 8
    xyz = torch.randn(N, 3, device=device)
    scaling = torch.empty(N, 3, device=device)
    scaling[: N // 2] = torch.tensor([0.1, 0.1, 0.005], device=device)  # flat
    scaling[N // 2 :] = torch.tensor([0.05, 0.05, 0.05], device=device)  # sphere
    rotation = _identity_quat(N, device)
    preds = torch.softmax(torch.randn(N, C, device=device), dim=-1)

    common = dict(
        k=5,
        lambda_val=2.0,
        max_points=N,
        sample_size=256,
        coarse_k=64,
        normal_weight=0.0,
    )
    torch.manual_seed(8)
    loss_off = loss_cls_3d_aniso_uncertain(
        xyz,
        scaling,
        rotation,
        preds,
        use_anchor_weight=False,
        use_neighbor_weight=False,
        **common,
    )
    torch.manual_seed(8)
    loss_on = loss_cls_3d_aniso_uncertain(
        xyz,
        scaling,
        rotation,
        preds,
        use_anchor_weight=True,
        use_neighbor_weight=True,
        **common,
    )
    diff = abs(loss_on.item() - loss_off.item())
    assert diff > 1e-6, (
        f"Weights should change the loss in mixed-anisotropy case; diff={diff}"
    )
    print(
        f"[OK] test_uncertain_loss_differs_with_mixed_anisotropy  "
        f"off={loss_off.item():.4f}  on={loss_on.item():.4f}  diff={diff:.4f}"
    )


def test_uncertain_loss_gradient_flows():
    """Gradient must flow to predictions when uncertainty weights are on."""
    torch.manual_seed(9)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N, C = 200, 16
    xyz = torch.randn(N, 3, device=device)
    scaling = torch.rand(N, 3, device=device) * 0.1 + 0.01
    rotation = _identity_quat(N, device)
    preds = torch.softmax(torch.randn(N, C, device=device), dim=-1)
    preds.requires_grad_(True)

    loss = loss_cls_3d_aniso_uncertain(
        xyz,
        scaling,
        rotation,
        preds,
        k=5,
        lambda_val=2.0,
        max_points=N,
        sample_size=64,
        coarse_k=32,
        normal_weight=0.1,
        use_anchor_weight=True,
        use_neighbor_weight=True,
    )
    loss.backward()
    assert preds.grad is not None
    assert torch.isfinite(preds.grad).all()
    print(f"[OK] test_uncertain_loss_gradient_flows  loss={loss.item():.4f}")


if __name__ == "__main__":
    print("=" * 70)
    print("Uncertainty-Aware Grouping — unit tests (Breakthrough F)")
    print("=" * 70)
    test_anisotropy_range()
    test_pixel_entropy_weight()
    test_weighted_2d_ce_loss_recovers_mean_when_uniform_weights()
    test_weighted_2d_ce_loss_downweights_uncertain_pixels()
    test_uncertain_loss_recovers_aniso_when_weights_off()
    test_uncertain_loss_differs_with_mixed_anisotropy()
    test_uncertain_loss_gradient_flows()
    print("=" * 70)
    print("All F tests passed.")
    print("=" * 70)
