"""
Diagnose why `render_lerf_mask.py` produces poor final `test_mask` even when
Grounded-SAM and `objects_feature16` look reasonable.

This script reads an already-rendered model output and compares:
  1. GT masks in data/lerf_mask/<scene>/test_mask
  2. Pred masks in <model_dir>/test/ours_<iter>_text/test_mask
  3. Grounded-SAM visualization images saved by render_lerf_mask.py

It is intentionally lightweight and does NOT rerun rendering.

Usage:
    python tests/diagnose_render_lerf_mask.py \
        --scene room \
        --model_dir output/room_full

Optional:
    python tests/diagnose_render_lerf_mask.py \
        --scene room \
        --model_dir output/room_full \
        --iteration 30000
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


PROMPTS_BY_SCENE = {
    "figurines": [
        "green apple",
        "green toy chair",
        "old camera",
        "porcelain hand",
        "red apple",
        "red toy chair",
        "rubber duck with red hat",
    ],
    "ramen": [
        "chopsticks",
        "egg",
        "glass of water",
        "pork belly",
        "wavy noodles in bowl",
        "yellow bowl",
    ],
    "teatime": [
        "apple",
        "bag of cookies",
        "coffee mug",
        "cookies on a plate",
        "paper napkin",
        "plate",
        "sheep",
        "spoon handle",
        "stuffed bear",
        "tea in a glass",
    ],
    "room": [
        "sofa",
        "TV",
        "keyboard",
        "mouse",
        "drinking glass",
        "armchair",
    ],
}


def mask_to_bool(mask):
    return np.array(mask.convert("L")) > 128


def iou_score(gt, pred):
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def infer_iteration(model_dir: Path):
    test_dir = model_dir / "test"
    if not test_dir.is_dir():
        return None
    cands = []
    for d in test_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if name.startswith("ours_") and name.endswith("_text"):
            middle = name[len("ours_") : -len("_text")]
            if middle.isdigit():
                cands.append(int(middle))
    if not cands:
        return None
    return max(cands)


def summarize_prompt(scene, prompt, gt_root, pred_root, render_root):
    print("-" * 90)
    print(f"Prompt: {prompt}")

    grounded_sam_img = render_root / f"grounded-sam---{prompt}.png"
    if grounded_sam_img.exists():
        print(f"  Grounded-SAM visualization exists: {grounded_sam_img}")
    else:
        print(f"  [WARN] Missing Grounded-SAM visualization: {grounded_sam_img.name}")

    gt_views = sorted([d for d in gt_root.iterdir() if d.is_dir()])
    pred_views = {d.name: d for d in pred_root.iterdir() if d.is_dir()}

    total_gt_px = 0
    total_pred_px = 0
    ious = []
    missing_views = []
    zero_pred_views = []

    for gt_view in gt_views:
        view_name = gt_view.name
        gt_file = gt_view / f"{prompt}.png"
        if not gt_file.exists():
            continue
        if view_name not in pred_views:
            missing_views.append(view_name)
            continue
        pred_file = pred_views[view_name] / f"{prompt}.png"
        if not pred_file.exists():
            missing_views.append(view_name)
            continue

        gt_mask = mask_to_bool(Image.open(gt_file))
        pred_mask = mask_to_bool(Image.open(pred_file))
        if gt_mask.shape != pred_mask.shape:
            pred_mask = (
                np.array(
                    Image.fromarray((pred_mask.astype(np.uint8) * 255)).resize(
                        (gt_mask.shape[1], gt_mask.shape[0]), resample=Image.NEAREST
                    )
                )
                > 128
            )

        gt_px = int(gt_mask.sum())
        pred_px = int(pred_mask.sum())
        total_gt_px += gt_px
        total_pred_px += pred_px
        ious.append(iou_score(gt_mask, pred_mask))
        if pred_px == 0:
            zero_pred_views.append(view_name)

        print(
            f"  view={view_name:<4} GT_px={gt_px:<8} Pred_px={pred_px:<8} IoU={ious[-1]:.4f}"
        )

    mean_iou = float(np.mean(ious)) if ious else 0.0
    print(
        f"  Summary: mean_IoU={mean_iou:.4f}, total_GT_px={total_gt_px}, total_Pred_px={total_pred_px}"
    )
    if missing_views:
        print(f"  [WARN] Missing pred views/files: {missing_views}")
    if zero_pred_views:
        print(f"  [WARN] Pred all-zero on views: {zero_pred_views}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--iteration", type=int, default=None)
    args = parser.parse_args()

    scene = args.scene
    model_dir = Path(args.model_dir)
    iteration = args.iteration

    if scene not in PROMPTS_BY_SCENE:
        raise ValueError(
            f"Unknown scene '{scene}'. Known scenes: {sorted(PROMPTS_BY_SCENE.keys())}"
        )

    if iteration is None:
        iteration = infer_iteration(model_dir)
    if iteration is None:
        raise RuntimeError(
            f"Could not infer iteration from {model_dir}/test. Please pass --iteration."
        )

    gt_root = Path("data/lerf_mask") / scene / "test_mask"
    render_root = model_dir / "test" / f"ours_{iteration}_text"
    pred_root = render_root / "test_mask"

    print("=" * 90)
    print(f"Diagnosing render_lerf_mask output")
    print(f"scene      : {scene}")
    print(f"model_dir  : {model_dir}")
    print(f"iteration  : {iteration}")
    print(f"gt_root    : {gt_root}")
    print(f"render_root: {render_root}")
    print(f"pred_root  : {pred_root}")
    print("=" * 90)

    if not gt_root.is_dir():
        raise FileNotFoundError(f"GT root not found: {gt_root}")
    if not render_root.is_dir():
        raise FileNotFoundError(f"Render root not found: {render_root}")
    if not pred_root.is_dir():
        raise FileNotFoundError(f"Pred root not found: {pred_root}")

    prompts = PROMPTS_BY_SCENE[scene]
    print(f"Prompts: {prompts}")
    print()

    for prompt in prompts:
        summarize_prompt(scene, prompt, gt_root, pred_root, render_root)

    print()
    print("Next check suggestions:")
    print(
        "1. Open grounded-sam---<prompt>.png and compare with prompt-level IoU above."
    )
    print(
        "2. If grounded-sam looks good but Pred_px is often 0, the issue is likely selected_obj_ids / IOA filtering."
    )
    print(
        "3. If Pred_px is huge for many prompts, the issue is likely probability-threshold leakage."
    )
    print(
        "4. Also compare with render.py outputs in objects_pred/ to see whether the classifier itself is already fragmented."
    )


if __name__ == "__main__":
    main()
