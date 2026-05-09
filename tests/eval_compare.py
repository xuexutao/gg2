"""
Side-by-side evaluation: BASELINE (Gaussian Grouping) vs OURS (Anisotropic).

Reuses the IoU / Boundary-IoU definitions from script/eval_lerf_mask.py but:
  - accepts two model directories
  - handles the render output layout produced by render_lerf_mask.py
  - prints a comparison table and dumps a JSON file
  - uses the SAME GT masks, the SAME dilation ratio, the SAME file naming as
    the original eval to ensure the numbers are directly comparable to what
    Gaussian Grouping reports in the paper.

The rendered test-mask layout created by `render_lerf_mask.py` is:
    <MODEL_DIR>/test/ours_<ITER>_text/test_mask/<view_idx>/<text_prompt>.png

Whereas the GT layout is:
    data/lerf_mask/<scene>/test_mask/<view_image_name>/<text_prompt>.png

The rendered-view-index → view-image-name mapping requires that we iterate
the test cameras in the SAME order render_lerf_mask.py used (alphabetical).
We resolve this by:
    1. reading the sorted list of test-view subdirs from the GT folder;
    2. matching by positional index (0-based) with the pred subdirs.

This matches what the official eval script assumes; we make it explicit here.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


# ---------- boundary utilities (identical to script/eval_lerf_mask.py) ----------


def mask_to_boundary(mask, dilation_ratio=0.02):
    h, w = mask.shape
    img_diag = np.sqrt(h**2 + w**2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    return mask - mask_erode


def boundary_iou(gt, dt, dilation_ratio=0.02):
    dt = (dt > 128).astype("uint8")
    gt = (gt > 128).astype("uint8")
    gt_b = mask_to_boundary(gt, dilation_ratio)
    dt_b = mask_to_boundary(dt, dilation_ratio)
    inter = ((gt_b * dt_b) > 0).sum()
    union = ((gt_b + dt_b) > 0).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def iou_score(gt, dt):
    gt_bool = gt > 128
    dt_bool = dt > 128
    inter = np.logical_and(gt_bool, dt_bool).sum()
    union = np.logical_or(gt_bool, dt_bool).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def resize_to(mask, target):
    return np.array(
        Image.fromarray(mask).resize(
            (target.shape[1], target.shape[0]), resample=Image.NEAREST
        )
    )


# ---------- evaluation of a single model ----------


def evaluate_model(scene, model_dir, iteration):
    """
    Returns dict {"mIoU": float, "mBIoU": float, "per_class_iou": {...}, "per_class_biou": {...}}.
    If the rendered output is missing, returns None.
    """
    # Be robust to CWD: resolve paths relative to repo root.
    repo_root = Path(__file__).resolve().parents[1]

    # Official LERF-Mask layout is: data/lerf_mask/<scene>/test_mask
    # Some local datasets may follow: data/<scene>/test_mask
    gt_candidates = [
        repo_root / "data" / "lerf_mask" / scene / "test_mask",
        repo_root / "data" / scene / "test_mask",
    ]
    gt_root = next((p for p in gt_candidates if p.is_dir()), gt_candidates[0])
    pred_root = Path(model_dir) / "test" / f"ours_{iteration}_text" / "test_mask"

    if not gt_root.is_dir():
        print(f"[ERROR] GT dir missing: {gt_root}")
        return None
    if not pred_root.is_dir():
        print(f"[ERROR] Pred dir missing: {pred_root}")
        return None

    gt_view_dirs = sorted([d for d in gt_root.iterdir() if d.is_dir()])
    pred_view_dirs = sorted([d for d in pred_root.iterdir() if d.is_dir()])
    pred_by_name = {d.name: d for d in pred_view_dirs}

    # Primary strategy: match by directory name (works for the official LERF-Mask layout
    # where GT view dirs are numeric indices).
    # Fallback: match by positional index in sorted order (handles layouts where GT
    # dirs are view-image names but preds are numeric indices).
    def match_pred_dir(gt_idx: int, gt_view_dir: Path) -> Optional[Path]:
        if gt_view_dir.name in pred_by_name:
            return pred_by_name[gt_view_dir.name]
        if gt_idx < len(pred_view_dirs):
            return pred_view_dirs[gt_idx]
        return None

    print(f"[DEBUG] GT root: {gt_root}")
    print(f"[DEBUG] Pred root: {pred_root}")
    print(f"[DEBUG] GT views: {[d.name for d in gt_view_dirs]}")
    print(f"[DEBUG] Pred views available: {[d.name for d in pred_view_dirs[:10]]}...")

    per_class_iou = {}
    per_class_biou = {}

    for gi, gt_view_dir in enumerate(gt_view_dirs):
        pred_view_dir = match_pred_dir(gi, gt_view_dir)
        if pred_view_dir is None:
            print(
                f"[WARN] GT view {gt_view_dir.name} has no corresponding pred view (gi={gi}), skipping"
            )
            continue

        print(
            f"[DEBUG] Matching GT view {gt_view_dir.name} → Pred view {pred_view_dir.name}"
        )

        for cat_file in os.listdir(gt_view_dir):
            if not cat_file.endswith(".png"):
                continue
            cat_id = cat_file.rsplit(".", 1)[0]
            gt_path = gt_view_dir / cat_file
            pred_path = pred_view_dir / cat_file

            if not pred_path.exists():
                per_class_iou.setdefault(cat_id, []).append(0.0)
                per_class_biou.setdefault(cat_id, []).append(0.0)
                continue

            gt_mask = np.array(Image.open(gt_path).convert("L"))
            pred_mask = np.array(Image.open(pred_path).convert("L"))
            if pred_mask.shape != gt_mask.shape:
                pred_mask = resize_to(pred_mask, gt_mask)

            per_class_iou.setdefault(cat_id, []).append(iou_score(gt_mask, pred_mask))
            per_class_biou.setdefault(cat_id, []).append(
                boundary_iou(gt_mask, pred_mask)
            )

    mean_iou = {c: float(np.mean(v)) for c, v in per_class_iou.items()}
    mean_biou = {c: float(np.mean(v)) for c, v in per_class_biou.items()}
    result = {
        "mIoU": float(np.mean(list(mean_iou.values()))) if mean_iou else 0.0,
        "mBIoU": float(np.mean(list(mean_biou.values()))) if mean_biou else 0.0,
        "per_class_iou": mean_iou,
        "per_class_biou": mean_biou,
    }
    return result


# ---------- pretty printing ----------


def print_comparison(scene, base, ours):
    if base is None and ours is None:
        print("No results from either model. Did rendering complete?")
        return
    print()
    print("=" * 78)
    print(f"  Side-by-side comparison on scene: {scene}")
    print("=" * 78)

    row_fmt = "  {:<38s} {:>12s} {:>12s} {:>12s}"
    print(row_fmt.format("metric", "baseline", "ours", "Δ"))
    print("-" * 78)

    def fnum(x):
        return "N/A" if x is None else f"{x:.4f}"

    def delta(a, b):
        if a is None or b is None:
            return "-"
        return f"{(b - a):+.4f}"

    b_miou = base["mIoU"] if base else None
    o_miou = ours["mIoU"] if ours else None
    b_bi = base["mBIoU"] if base else None
    o_bi = ours["mBIoU"] if ours else None

    print(
        row_fmt.format(
            "mIoU           (region overlap)",
            fnum(b_miou),
            fnum(o_miou),
            delta(b_miou, o_miou),
        )
    )
    print(
        row_fmt.format(
            "mBIoU          (boundary IoU, our claim)",
            fnum(b_bi),
            fnum(o_bi),
            delta(b_bi, o_bi),
        )
    )

    # Per-class breakdown (union of keys)
    cats = sorted(
        set(
            list((base or {}).get("per_class_iou", {}).keys())
            + list((ours or {}).get("per_class_iou", {}).keys())
        )
    )
    if cats:
        print()
        print("  Per-class IoU / BIoU")
        print("-" * 78)
        print(f"  {'class':<38s} {'base IoU/BIoU':>18s}   {'ours IoU/BIoU':>18s}")
        for c in cats:
            bi = (base or {}).get("per_class_iou", {}).get(c)
            bb = (base or {}).get("per_class_biou", {}).get(c)
            oi = (ours or {}).get("per_class_iou", {}).get(c)
            ob = (ours or {}).get("per_class_biou", {}).get(c)
            left = "-" if bi is None else f"{bi:.3f}/{bb:.3f}"
            right = "-" if oi is None else f"{oi:.3f}/{ob:.3f}"
            print(f"  {c:<38s} {left:>18s}   {right:>18s}")

    print("=" * 78)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--baseline_model", required=True)
    parser.add_argument("--ours_model", required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    print(f"[eval] Baseline : {args.baseline_model}")
    print(f"[eval] Ours     : {args.ours_model}")
    print(f"[eval] iteration: {args.iteration}")

    base = evaluate_model(args.scene, args.baseline_model, args.iteration)
    ours = evaluate_model(args.scene, args.ours_model, args.iteration)

    print_comparison(args.scene, base, ours)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(
                {
                    "scene": args.scene,
                    "iteration": args.iteration,
                    "baseline": base,
                    "ours": ours,
                },
                f,
                indent=2,
            )
        print(f"[eval] wrote {args.out_json}")


if __name__ == "__main__":
    sys.exit(main())
