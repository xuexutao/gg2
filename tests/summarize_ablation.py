"""
Pretty-print the ablation matrix summary JSON produced by run_ablation_matrix.sh.

Usage:
    python tests/summarize_ablation.py --summary output/verify_logs/ablation_summary_XXX.json
"""

import argparse
import json
import os
import sys


def fmt_pct(x):
    return f"{x * 100:.2f}"


def print_main_table(data):
    """One row per scene, columns = configs. Show mIoU / mBIoU."""
    scenes = list(data.keys())
    if not scenes:
        print("[summary] No scenes found.")
        return

    # Dynamically discover the set of config tags actually present (preserving
    # a preferred order if the user used our standard names).
    preferred_order = ["baseline", "aniso", "normal", "uncertain", "full"]
    present_tags = set()
    for scene in scenes:
        present_tags.update(data[scene].keys())
    tags = [t for t in preferred_order if t in present_tags]
    # Append any unknown tags in insertion order.
    for t in present_tags:
        if t not in tags:
            tags.append(t)

    print()
    print("=" * (20 + 28 * len(tags)))
    print("  Main Table: Average IoU / Boundary-IoU across all text prompts")
    print("=" * (20 + 28 * len(tags)))
    header = f"  {'scene':<14s}"
    for tag in tags:
        header += f" | {tag + ' mIoU':>12s} {tag + ' mBIoU':>12s}"
    print(header)
    print("-" * (20 + 28 * len(tags)))

    miou_sums = {t: 0.0 for t in tags}
    bi_sums = {t: 0.0 for t in tags}
    n = {t: 0 for t in tags}

    for scene in scenes:
        row = f"  {scene:<14s}"
        for tag in tags:
            if tag in data[scene]:
                m = data[scene][tag]["mIoU"]
                b = data[scene][tag]["mBIoU"]
                miou_sums[tag] += m
                bi_sums[tag] += b
                n[tag] += 1
                row += f" | {fmt_pct(m):>12s} {fmt_pct(b):>12s}"
            else:
                row += f" | {'—':>12s} {'—':>12s}"
        print(row)

    print("-" * (20 + 28 * len(tags)))
    row = f"  {'AVERAGE':<14s}"
    for tag in tags:
        if n[tag] > 0:
            row += f" | {fmt_pct(miou_sums[tag] / n[tag]):>12s} {fmt_pct(bi_sums[tag] / n[tag]):>12s}"
        else:
            row += f" | {'—':>12s} {'—':>12s}"
    print(row)
    print("=" * (20 + 28 * len(tags)))

    # Highlight the gains
    if "baseline" in tags and n["baseline"] > 0:
        print()
        print("  Gains over Baseline (averaged across scenes):")
        for tag in tags:
            if tag == "baseline" or n[tag] == 0:
                continue
            d_miou = miou_sums[tag] / n[tag] - miou_sums["baseline"] / n["baseline"]
            d_biou = bi_sums[tag] / n[tag] - bi_sums["baseline"] / n["baseline"]
            print(
                f"    {tag:<10s}: mIoU {fmt_pct(d_miou):>7s} pp | mBIoU {fmt_pct(d_biou):>7s} pp"
            )
        print()


def print_ablation_rows(data):
    """Per-scene ablation rows, latex-friendly."""
    print()
    print("=" * 84)
    print("  Ablation rows (latex-friendly, values = pct)")
    print("=" * 84)
    preferred_order = ["baseline", "aniso", "normal", "uncertain", "full"]
    for scene in data:
        tags_in_scene = [t for t in preferred_order if t in data[scene]]
        for t in data[scene]:
            if t not in tags_in_scene:
                tags_in_scene.append(t)
        for tag in tags_in_scene:
            d = data[scene][tag]
            print(
                f"  {scene:<12s} & {tag:<10s} & {fmt_pct(d['mIoU'])} & {fmt_pct(d['mBIoU'])} \\\\"
            )
    print("=" * 84)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True)
    args = p.parse_args()

    if not os.path.exists(args.summary):
        print(f"[summary] file not found: {args.summary}")
        sys.exit(1)
    with open(args.summary) as f:
        data = json.load(f)

    print_main_table(data)
    print_ablation_rows(data)


if __name__ == "__main__":
    main()
