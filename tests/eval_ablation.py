"""
Ablation evaluation: BASELINE vs ANISO-ONLY vs FULL (3 configurations).

Automatically runs render_lerf_mask.py for each model, then evaluates
all three and prints a comparison table.

Usage:
    python tests/eval_ablation.py \
        --scene teatime \
        --baseline_model output/teatime_baseline \
        --aniso_model output/teatime_aniso_only \
        --full_model output/teatime_full \
        --iteration 30000
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

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


# ---------- render a single model ----------


def render_model(model_dir, skip_train=True):
    """
    Runs render_lerf_mask.py for the given model directory.
    Assumes render_lerf_mask.py is in the current working directory.
    """
    cmd = [sys.executable, "render_lerf_mask.py", "-m", str(model_dir)]
    if skip_train:
        cmd.append("--skip_train")

    print(f"\n[render] Running: {' '.join(cmd)}")
    print("-" * 60)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("[stderr]", result.stderr)

    if result.returncode != 0:
        print(f"[ERROR] render_lerf_mask.py failed with code {result.returncode}")
        return False

    print("[render] Done")
    return True


# ---------- evaluation of a single model ----------


def evaluate_model(scene, model_dir, iteration):
    """
    Returns dict {"mIoU": float, "mBIoU": float, "per_class_iou": {...}, "per_class_biou": {...}}.
    If the rendered output is missing, returns None.
    """
    gt_root = Path("data/lerf_mask") / scene / "test_mask"
    pred_root = Path(model_dir) / "test" / f"ours_{iteration}_text" / "test_mask"

    if not gt_root.is_dir():
        print(f"[ERROR] GT dir missing: {gt_root}")
        return None
    if not pred_root.is_dir():
        print(f"[ERROR] Pred dir missing: {pred_root}")
        print(f"        Did render_lerf_mask.py complete successfully?")
        return None

    gt_view_dirs = sorted([d for d in gt_root.iterdir() if d.is_dir()])
    pred_view_dirs = sorted([d for d in pred_root.iterdir() if d.is_dir()])

    if len(gt_view_dirs) != len(pred_view_dirs):
        print(
            f"[WARN] Number of test views differ — gt={len(gt_view_dirs)} "
            f"pred={len(pred_view_dirs)}. Using min overlap positionally."
        )

    per_class_iou = {}
    per_class_biou = {}

    for gt_view_dir, pred_view_dir in zip(gt_view_dirs, pred_view_dirs):
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


# ---------- pretty printing for 3-way comparison ----------


def print_comparison_3way(scene, base, aniso, full):
    results = [("Baseline", base), ("+Aniso", aniso), ("+Normal", full)]
    valid_results = [(name, res) for name, res in results if res is not None]

    if not valid_results:
        print("No results from any model. Did rendering complete?")
        return

    print()
    print("=" * 90)
    print(f"  Ablation Comparison on scene: {scene}")
    print("=" * 90)

    row_fmt = "  {:<32s} {:>14s} {:>14s} {:>14s}"
    print(row_fmt.format("metric", "Baseline", "+Aniso", "+Normal"))
    print("-" * 90)

    def fnum(x):
        return "N/A" if x is None else f"{x:.4f}"

    b_miou = base["mIoU"] if base else None
    a_miou = aniso["mIoU"] if aniso else None
    f_miou = full["mIoU"] if full else None

    b_biou = base["mBIoU"] if base else None
    a_biou = aniso["mBIoU"] if aniso else None
    f_biou = full["mBIoU"] if full else None

    print(
        row_fmt.format(
            "mIoU           (region overlap)",
            fnum(b_miou),
            fnum(a_miou),
            fnum(f_miou),
        )
    )
    print(
        row_fmt.format(
            "mBIoU          (boundary IoU)",
            fnum(b_biou),
            fnum(a_biou),
            fnum(f_biou),
        )
    )

    cats = sorted(
        set(
            list((base or {}).get("per_class_iou", {}).keys())
            + list((aniso or {}).get("per_class_iou", {}).keys())
            + list((full or {}).get("per_class_iou", {}).keys())
        )
    )
    if cats:
        print()
        print("  Per-class IoU / BIoU")
        print("-" * 90)
        print(
            f"  {'class':<32s} {'Baseline':>16s}   {'+Aniso':>16s}   {'+Normal':>16s}"
        )
        for c in cats:
            bi = (base or {}).get("per_class_iou", {}).get(c)
            bb = (base or {}).get("per_class_biou", {}).get(c)
            ai = (aniso or {}).get("per_class_iou", {}).get(c)
            ab = (aniso or {}).get("per_class_biou", {}).get(c)
            fi = (full or {}).get("per_class_iou", {}).get(c)
            fb = (full or {}).get("per_class_biou", {}).get(c)

            b_str = "-" if bi is None else f"{bi:.3f}/{bb:.3f}"
            a_str = "-" if ai is None else f"{ai:.3f}/{ab:.3f}"
            f_str = "-" if fi is None else f"{fi:.3f}/{fb:.3f}"
            print(f"  {c:<32s} {b_str:>16s}   {a_str:>16s}   {f_str:>16s}")

    print("=" * 90)
    print()


# ---------- main ----------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate 3-way ablation: baseline vs aniso-only vs full"
    )
    parser.add_argument(
        "--scene", required=True, help="Scene name (e.g., teatime, figurines, ramen)"
    )
    parser.add_argument(
        "--baseline_model", required=True, help="Baseline model directory"
    )
    parser.add_argument(
        "--aniso_model", required=True, help="Aniso-only model directory"
    )
    parser.add_argument(
        "--full_model", required=True, help="Full (aniso+normal) model directory"
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=30000,
        help="Training iteration for rendering (default: 30000)",
    )
    parser.add_argument(
        "--skip_render", action="store_true", help="Skip rendering, only evaluate"
    )
    parser.add_argument("--out_json", default=None, help="Output JSON file path")
    args = parser.parse_args()

    models = [
        ("Baseline", args.baseline_model),
        ("+Aniso", args.aniso_model),
        ("+Normal", args.full_model),
    ]

    print("=" * 90)
    print("  3-Way Ablation Evaluation")
    print("=" * 90)
    print(f"  Scene     : {args.scene}")
    print(f"  Iteration : {args.iteration}")
    for name, path in models:
        print(f"  {name:<10s}: {path}")
    print("=" * 90)

    if not args.skip_render:
        print("\n" + "=" * 90)
        print("  Step 1: Rendering masks for each model")
        print("=" * 90)

        for name, model_dir in models:
            print(f"\n{'=' * 60}")
            print(f"  Rendering {name}: {model_dir}")
            print(f"{'=' * 60}")
            render_model(model_dir, skip_train=True)

    print("\n" + "=" * 90)
    print("  Step 2: Evaluating all models")
    print("=" * 90)

    results = {}
    for name, model_dir in models:
        print(f"\n[eval] Evaluating {name}: {model_dir}")
        res = evaluate_model(args.scene, model_dir, args.iteration)
        results[name] = res
        if res:
            print(f"       mIoU: {res['mIoU']:.4f}, mBIoU: {res['mBIoU']:.4f}")
        else:
            print(f"       [ERROR] Evaluation failed")

    print_comparison_3way(
        args.scene,
        results["Baseline"],
        results["+Aniso"],
        results["+Normal"],
    )

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "scene": args.scene,
                    "iteration": args.iteration,
                    "baseline": results["Baseline"],
                    "aniso_only": results["+Aniso"],
                    "full": results["+Normal"],
                },
                f,
                indent=2,
            )
        print(f"[eval] wrote results to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
