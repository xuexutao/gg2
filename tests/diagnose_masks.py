"""
Diagnose mask files to understand why most IoUs are 0.

Usage:
    python tests/diagnose_masks.py --scene teatime \
        --gt_dir data/lerf_mask/teatime/test_mask \
        --pred_dir output/teatime_baseline/test/ours_30000_text/test_mask
"""

import argparse
import numpy as np
from PIL import Image
from pathlib import Path


def diagnose_masks(gt_dir, pred_dir, scene):
    gt_root = Path(gt_dir)
    pred_root = Path(pred_dir)

    if not gt_root.is_dir():
        print(f"[ERROR] GT dir not found: {gt_root}")
        return
    if not pred_root.is_dir():
        print(f"[ERROR] Pred dir not found: {pred_root}")
        return

    # Get GT views (like ['0', '2'])
    gt_views = sorted([d.name for d in gt_root.iterdir() if d.is_dir()])
    pred_views = {d.name: d for d in pred_root.iterdir() if d.is_dir()}

    print("=" * 80)
    print(f"  Diagnosis for scene: {scene}")
    print("=" * 80)
    print(f"GT views: {gt_views}")
    print(f"Pred views available (first 10): {sorted(pred_views.keys())[:10]}...")
    print()

    # Check each GT view
    for gt_view in gt_views:
        if gt_view not in pred_views:
            print(f"[WARN] GT view {gt_view} not found in pred, skipping")
            continue

        gt_view_dir = gt_root / gt_view
        pred_view_dir = pred_views[gt_view]

        print("-" * 80)
        print(f"  View: {gt_view}")
        print("-" * 80)

        # Get GT category files
        gt_files = sorted([f for f in gt_view_dir.iterdir() if f.suffix == ".png"])
        print(f"GT categories in view {gt_view}: {[f.stem for f in gt_files]}")
        print()

        # Check each category
        for gt_file in gt_files:
            cat_name = gt_file.stem

            # GT mask
            gt_mask = np.array(Image.open(gt_file).convert("L"))

            # Pred mask (may not exist or may be all zeros)
            pred_file = pred_view_dir / f"{cat_name}.png"

            if not pred_file.exists():
                print(f"  [MISSING] {cat_name}: Pred file not found")
                continue

            pred_mask = np.array(Image.open(pred_file).convert("L"))

            # Check if masks are all black
            gt_nonzero = np.count_nonzero(gt_mask > 128)
            pred_nonzero = np.count_nonzero(pred_mask > 128)

            # Simple IoU
            gt_bool = gt_mask > 128
            pred_bool = pred_mask > 128
            intersection = np.logical_and(gt_bool, pred_bool).sum()
            union = np.logical_or(gt_bool, pred_bool).sum()
            iou = intersection / union if union > 0 else 0.0

            status = "OK" if iou > 0 else "ZERO"
            print(
                f"  [{status}] {cat_name}: GT_nonzero={gt_nonzero}, Pred_nonzero={pred_nonzero}, IoU={iou:.4f}"
            )

            # If pred is all zeros, check if there's ANY object detected
            if pred_nonzero == 0:
                # Check what files are in this pred view
                pred_files = sorted(
                    [f.name for f in pred_view_dir.iterdir() if f.suffix == ".png"]
                )
                print(f"           Available pred files: {pred_files[:5]}...")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    args = parser.parse_args()

    diagnose_masks(args.gt_dir, args.pred_dir, args.scene)


if __name__ == "__main__":
    main()
