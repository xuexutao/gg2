"""
Visualization utilities for Breakthrough 1 (Anisotropic Affinity).

Produces the key figures you'll put in the paper:

  vis_thin_wall.png
      Figure 1 (teaser): synthetic thin-wall toy. Euclidean KNN connects the
      two faces into one blob; Anisotropic KNN separates them cleanly.

  vis_compare_<scene>_<view_idx>.png
      Figure 2/3 (real scenes): side-by-side
          [ RGB | GT colored | Baseline pred | Ours pred ]
      using the predicted-object visualization that render.py already emits.
      Also crops a zoom-in panel to highlight boundary differences.

  vis_ablation_<scene>_<view_idx>.png
      Figure 4 (ablation): five columns
          [ RGB | Baseline | +Aniso | +Aniso+Normal | GT ]

  vis_identity_pca_<scene>.png
      Figure 5: PCA of the 16-d Identity Encoding over the 3D Gaussian
      point cloud, projected to RGB and rendered on a camera view.

Usage:
    # 1) Teaser / thin-wall toy (no data required, runs anywhere)
    python -m tests.visualize thin_wall

    # 2) Real-scene comparison between two trained runs
    python -m tests.visualize compare \
        --scene figurines \
        --baseline_model output/verify_figurines_baseline \
        --ours_model     output/verify_figurines_aniso \
        --iteration 30000 \
        --view_indices 0 1 2

    # 3) Ablation (requires 3 trained runs)
    python -m tests.visualize ablation \
        --scene figurines \
        --baseline_model output/verify_figurines_baseline \
        --aniso_only_model output/verify_figurines_aniso_only \
        --ours_model       output/verify_figurines_aniso \
        --iteration 30000 \
        --view_indices 0 1

    # 4) 3D Identity PCA scatter
    python -m tests.visualize pca \
        --baseline_model output/verify_figurines_baseline \
        --ours_model     output/verify_figurines_aniso
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# =============================================================================
# 1. Teaser figure: synthetic thin-wall scene.
#    Builds a 3D scatter, computes Euclidean-KNN vs Aniso-KNN groupings, and
#    colors points accordingly.  Saves a side-by-side PNG.
# =============================================================================


def make_thin_wall_scene(grid=10, eps=0.02, device="cpu"):
    import torch
    from utils.loss_utils import _build_covariance, _quat_to_rotmat

    grid_lin = torch.linspace(0.0, 1.0, grid, device=device)
    xx, yy = torch.meshgrid(grid_lin, grid_lin, indexing="xy")
    xy_flat = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    n_per = xy_flat.shape[0]

    front = torch.cat([xy_flat, torch.full((n_per, 1), +eps, device=device)], dim=-1)
    back = torch.cat([xy_flat, torch.full((n_per, 1), -eps, device=device)], dim=-1)
    xyz = torch.cat([front, back], dim=0)

    # Front: flat in xy, normal = +z   -> scaling (0.10, 0.10, 0.005)
    # Back : flat in yz, normal = +x   -> scaling (0.005, 0.10, 0.10)
    # i.e. structurally different Σ
    scaling = torch.empty(2 * n_per, 3, device=device)
    scaling[:n_per] = torch.tensor([0.10, 0.10, 0.005], device=device)
    scaling[n_per:] = torch.tensor([0.005, 0.10, 0.10], device=device)
    quat = torch.zeros(2 * n_per, 4, device=device)
    quat[:, 0] = 1.0
    return xyz, scaling, quat, n_per


def knn_groupings(xyz, scaling, quat, k=3, coarse_k=32):
    """Return (eu_labels, aniso_labels, eu_neighbor_purity, an_neighbor_purity):
      * labels: per-point connected-component id on each graph
      * purity: for each point, fraction of its k neighbors on the same face
    k is kept small so connected components stay distinguishable on the toy."""
    import torch
    from utils.loss_utils import _build_covariance

    N = xyz.shape[0]
    # Euclidean KNN
    dists_eu = torch.cdist(xyz, xyz)
    _, eu_nb = dists_eu.topk(k + 1, largest=False)
    eu_nb = eu_nb[:, 1:]

    # Anisotropic re-ranked KNN
    _, coarse = dists_eu.topk(coarse_k + 1, largest=False)
    coarse = coarse[:, 1:]
    cov = _build_covariance(scaling, quat)
    I3 = torch.eye(3)
    cov_inv = torch.linalg.inv(cov + 1e-6 * I3)
    cov_cand = cov[coarse]
    cov_cand_inv = cov_inv[coarse]
    delta = xyz.unsqueeze(1) - xyz[coarse]
    term_a = torch.einsum("mcd,mde,mce->mc", delta, cov_inv, delta)
    term_b = torch.einsum("mcd,mcde,mce->mc", delta, cov_cand_inv, delta)
    d_m = torch.sqrt(torch.clamp(0.5 * (term_a + term_b), min=0.0) + 1e-6)
    _, topk_idx = d_m.topk(k, largest=False)
    aniso_nb = coarse.gather(1, topk_idx)

    # Build undirected graphs and run union-find for connected components
    def union_find(n, edges):
        parent = list(range(n))

        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        for a, b in edges:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        return np.array([find(i) for i in range(n)])

    eu_edges = [(i, int(j)) for i in range(N) for j in eu_nb[i].tolist()]
    an_edges = [(i, int(j)) for i in range(N) for j in aniso_nb[i].tolist()]
    eu_lbl = union_find(N, eu_edges)
    an_lbl = union_find(N, an_edges)
    return eu_lbl, an_lbl, eu_nb, aniso_nb


def draw_thin_wall(out_path="output/vis/vis_thin_wall.png"):
    import torch

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[skip] matplotlib not available; thin-wall teaser not produced.")
        return

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    xyz, scaling, quat, n_per = make_thin_wall_scene(grid=12, eps=0.02)
    eu_lbl, an_lbl, eu_nb, an_nb = knn_groupings(xyz, scaling, quat, k=3, coarse_k=32)

    def remap(lbl):
        uniq = {v: i for i, v in enumerate(sorted(set(lbl.tolist())))}
        return np.array([uniq[v] for v in lbl])

    eu_lbl = remap(eu_lbl)
    an_lbl = remap(an_lbl)

    # Neighbor-level purity (the number you cite in the paper)
    import numpy as _np

    label = _np.array([0] * n_per + [1] * n_per)

    def neighbor_purity(nb):
        nb_np = nb.cpu().numpy() if hasattr(nb, "cpu") else _np.asarray(nb)
        return float(_np.mean(label[nb_np] == label[:, None]))

    eu_purity = neighbor_purity(eu_nb)
    an_purity = neighbor_purity(an_nb)

    fig = plt.figure(figsize=(12, 5.5))
    for i, (title, lbl, purity) in enumerate(
        [
            (
                f"Euclidean KNN (point view)\nsame-face neighbor ratio = {eu_purity:.1%}",
                eu_lbl,
                eu_purity,
            ),
            (
                f"Anisotropic KNN (ellipsoid view, ours)\nsame-face neighbor ratio = {an_purity:.1%}",
                an_lbl,
                an_purity,
            ),
        ]
    ):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2], c=lbl, s=30, cmap="tab20", depthshade=False
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=12, azim=-62)
    fig.suptitle(
        "Thin-wall toy: two faces at z = ±0.02 with opposite anisotropy\n"
        "(colors = connected components in each neighbor graph)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[vis] thin-wall teaser -> {out_path}")
    print(f"       Euclidean same-face neighbor ratio: {eu_purity:.3f}")
    print(f"       Aniso     same-face neighbor ratio: {an_purity:.3f}")


# =============================================================================
# 2. Real-scene comparison. Uses the outputs render.py already produced:
#      <MODEL>/test/ours_<iter>/objects_pred/<view>.png   (colored pred mask)
#      <MODEL>/test/ours_<iter>/gt_objects_color/<view>.png
#      <MODEL>/test/ours_<iter>/renders/<view>.png
# =============================================================================


def _load_png(p):
    return np.array(Image.open(p))


def _zoom_inset(img, x, y, w, h, factor=3):
    """Crop img[y:y+h, x:x+w] and upscale by `factor`."""
    crop = img[y : y + h, x : x + w]
    return np.array(
        Image.fromarray(crop).resize((w * factor, h * factor), Image.NEAREST)
    )


def _hstack_with_titles(panels, titles, pad=8, title_h=28, bg=(255, 255, 255)):
    """Assemble a horizontally-stacked figure with column titles."""
    try:
        from PIL import ImageDraw, ImageFont
    except ImportError:
        print("[skip] Pillow missing ImageDraw; returning raw hstack.")
        return np.concatenate(panels, axis=1)

    h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] != h:
            p = np.array(Image.fromarray(p).resize((p.shape[1], h)))
        if p.ndim == 2:
            p = np.stack([p] * 3, axis=-1)
        if p.shape[-1] == 4:
            p = p[..., :3]
        padded.append(p)
    W = sum(p.shape[1] for p in padded) + pad * (len(padded) - 1)
    canvas = np.full((h + title_h, W, 3), bg, dtype=np.uint8)
    x = 0
    for p, t in zip(padded, titles):
        canvas[title_h : title_h + p.shape[0], x : x + p.shape[1]] = p
        x += p.shape[1] + pad

    img = Image.fromarray(canvas)
    drw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    x = 0
    for p, t in zip(padded, titles):
        drw.text((x + 4, 4), t, fill=(0, 0, 0), font=font)
        x += p.shape[1] + pad
    return np.array(img)


def vis_compare(
    scene, baseline_model, ours_model, iteration, view_indices, out_dir="output/vis"
):
    os.makedirs(out_dir, exist_ok=True)
    for vi in view_indices:
        name = f"{vi:05d}.png"
        paths = {
            "rgb": Path(baseline_model)
            / "test"
            / f"ours_{iteration}"
            / "renders"
            / name,
            "gt": Path(baseline_model)
            / "test"
            / f"ours_{iteration}"
            / "gt_objects_color"
            / name,
            "baseline": Path(baseline_model)
            / "test"
            / f"ours_{iteration}"
            / "objects_pred"
            / name,
            "ours": Path(ours_model)
            / "test"
            / f"ours_{iteration}"
            / "objects_pred"
            / name,
        }
        missing = [k for k, p in paths.items() if not p.exists()]
        if missing:
            print(f"[skip] view {vi}: missing {missing}")
            continue
        rgb = _load_png(paths["rgb"])
        gt = _load_png(paths["gt"])
        base = _load_png(paths["baseline"])
        ours = _load_png(paths["ours"])
        canvas = _hstack_with_titles(
            [rgb, gt, base, ours],
            ["RGB", "GT identity", "Baseline (GG)", "Ours (Aniso)"],
        )
        out = os.path.join(out_dir, f"vis_compare_{scene}_{vi:05d}.png")
        Image.fromarray(canvas).save(out)
        print(f"[vis] compare view {vi} -> {out}")


# =============================================================================
# 3. Ablation: Baseline | +Aniso | +Aniso+Normal | GT
# =============================================================================


def vis_ablation(
    scene,
    baseline_model,
    aniso_only_model,
    ours_model,
    iteration,
    view_indices,
    out_dir="output/vis",
):
    os.makedirs(out_dir, exist_ok=True)
    for vi in view_indices:
        name = f"{vi:05d}.png"

        def pred(path):
            return Path(path) / "test" / f"ours_{iteration}" / "objects_pred" / name

        paths = {
            "rgb": Path(baseline_model)
            / "test"
            / f"ours_{iteration}"
            / "renders"
            / name,
            "baseline": pred(baseline_model),
            "aniso_only": pred(aniso_only_model),
            "ours": pred(ours_model),
            "gt": Path(baseline_model)
            / "test"
            / f"ours_{iteration}"
            / "gt_objects_color"
            / name,
        }
        missing = [k for k, p in paths.items() if not p.exists()]
        if missing:
            print(f"[skip] view {vi}: missing {missing}")
            continue
        imgs = [
            _load_png(paths[k]) for k in ["rgb", "baseline", "aniso_only", "ours", "gt"]
        ]
        titles = ["RGB", "Baseline", "+Aniso Neighbor", "+Normal Loss (full)", "GT"]
        canvas = _hstack_with_titles(imgs, titles)
        out = os.path.join(out_dir, f"vis_ablation_{scene}_{vi:05d}.png")
        Image.fromarray(canvas).save(out)
        print(f"[vis] ablation view {vi} -> {out}")


# =============================================================================
# 4. Identity-Encoding PCA scatter on the 3D point cloud.
#    Reads <model>/point_cloud/iteration_*/point_cloud.ply and the saved
#    16-d identity embedding (stored as extra cols in the ply per
#    gaussian_model save_ply convention).
# =============================================================================


def _read_ply_points_and_features(ply_path):
    """Return (xyz:[N,3], feat:[N,16]) from a Gaussian Grouping saved ply.
    The identity encoding is stored under fields 'f_obj_0' .. 'f_obj_15'."""
    from plyfile import PlyData

    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    feat_names = [n for n in v.data.dtype.names if n.startswith("f_obj_")]
    if not feat_names:
        feat_names = [n for n in v.data.dtype.names if n.startswith("obj_dc_")]
    feat_names = sorted(feat_names, key=lambda s: int(s.split("_")[-1]))
    feat = np.stack([v[n] for n in feat_names], axis=-1) if feat_names else None
    return xyz, feat


def _pca_to_rgb(feat):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=3)
    out = pca.fit_transform(feat)
    out = (out - out.min(0)) / (out.max(0) - out.min(0) + 1e-8)
    return (out * 255).astype(np.uint8)


def vis_pca(
    baseline_model, ours_model, iteration, out_dir="output/vis", max_points=100000
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[skip] matplotlib missing")
        return

    os.makedirs(out_dir, exist_ok=True)
    for tag, model_dir in [("baseline", baseline_model), ("ours", ours_model)]:
        ply_path = (
            Path(model_dir)
            / "point_cloud"
            / f"iteration_{iteration}"
            / "point_cloud.ply"
        )
        if not ply_path.exists():
            print(f"[skip] {ply_path} missing")
            continue
        xyz, feat = _read_ply_points_and_features(ply_path)
        if feat is None:
            print(f"[skip] no identity features in {ply_path}")
            continue
        if xyz.shape[0] > max_points:
            idx = np.random.choice(xyz.shape[0], max_points, replace=False)
            xyz, feat = xyz[idx], feat[idx]
        rgb = _pca_to_rgb(feat)

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb / 255.0, s=1, depthshade=False
        )
        ax.set_title(f"3D Identity PCA — {tag}")
        ax.view_init(elev=15, azim=-70)
        out = os.path.join(out_dir, f"vis_identity_pca_{tag}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[vis] PCA {tag} -> {out}")


# =============================================================================
# CLI
# =============================================================================


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("thin_wall")

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--scene", required=True)
    p_cmp.add_argument("--baseline_model", required=True)
    p_cmp.add_argument("--ours_model", required=True)
    p_cmp.add_argument("--iteration", type=int, default=30000)
    p_cmp.add_argument("--view_indices", type=int, nargs="+", default=[0, 1, 2])

    p_abl = sub.add_parser("ablation")
    p_abl.add_argument("--scene", required=True)
    p_abl.add_argument("--baseline_model", required=True)
    p_abl.add_argument("--aniso_only_model", required=True)
    p_abl.add_argument("--ours_model", required=True)
    p_abl.add_argument("--iteration", type=int, default=30000)
    p_abl.add_argument("--view_indices", type=int, nargs="+", default=[0, 1])

    p_pca = sub.add_parser("pca")
    p_pca.add_argument("--baseline_model", required=True)
    p_pca.add_argument("--ours_model", required=True)
    p_pca.add_argument("--iteration", type=int, default=30000)

    args = p.parse_args()

    if args.cmd == "thin_wall":
        draw_thin_wall()
    elif args.cmd == "compare":
        vis_compare(
            args.scene,
            args.baseline_model,
            args.ours_model,
            args.iteration,
            args.view_indices,
        )
    elif args.cmd == "ablation":
        vis_ablation(
            args.scene,
            args.baseline_model,
            args.aniso_only_model,
            args.ours_model,
            args.iteration,
            args.view_indices,
        )
    elif args.cmd == "pca":
        vis_pca(args.baseline_model, args.ours_model, args.iteration)


if __name__ == "__main__":
    main()
