"""Batch evaluation + ablation table for LERF-Mask.

This script scans:
  - GT:   data/lerf_mask/<scene>/test_mask
  - Pred: output/<model_dir>/test/ours_<iter>_text/test_mask

Then computes the same metrics as tests/eval_compare.py (mIoU, mBIoU) for each
available model variant (baseline/aniso_only/full/uncertain/other) and prints a
single Markdown table. It also writes a JSON + CSV snapshot under output/verify_logs.

Typical usage:
  python tests/run_ablation_table.py --iteration 30000

Optional:
  python tests/run_ablation_table.py --scenes figurines room teatime
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


def _abs_in_repo(p: Path) -> Path:
    """Resolve a path, treating relative paths as relative to REPO_ROOT."""
    if p.is_absolute():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


def _rel_to_repo(p: Path) -> str:
    """Best-effort path pretty-print relative to repo root."""
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(p)


def _load_eval_compare_module():
    """Load tests/eval_compare.py as a module without requiring package imports."""
    import importlib.util

    p = REPO_ROOT / "tests" / "eval_compare.py"
    spec = importlib.util.spec_from_file_location("_eval_compare", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _detect_scenes(gt_root: Path, requested: Optional[List[str]]) -> List[str]:
    scenes = []
    if not gt_root.is_dir():
        raise FileNotFoundError(f"GT root not found: {gt_root}")
    for d in sorted([p for p in gt_root.iterdir() if p.is_dir()]):
        if (d / "test_mask").is_dir():
            scenes.append(d.name)

    if requested:
        req = []
        for s in requested:
            if s not in scenes:
                # Still allow running a requested scene even if test_mask is missing;
                # evaluation will return None and show in the table.
                req.append(s)
            else:
                req.append(s)
        # keep input order, de-dup
        out = []
        seen = set()
        for s in req:
            if s not in seen:
                out.append(s)
                seen.add(s)
        return out

    return scenes


@dataclass
class ModelInfo:
    name: str
    path: Path
    scene: Optional[str]
    variant: str


def _infer_scene(model_name: str, scenes: List[str]) -> Optional[str]:
    # Prefer exact token match by splitting on underscores.
    toks = re.split(r"[_\-]+", model_name)
    for s in scenes:
        if s in toks:
            return s
    # Fallback: substring match
    for s in scenes:
        if s in model_name:
            return s
    return None


def _infer_variant(model_name: str) -> str:
    n = model_name.lower()
    if "baseline" in n:
        return "baseline"
    if "aniso" in n and "only" in n:
        return "aniso_only"
    if "aniso" in n:
        return "aniso"
    if "full" in n:
        return "full"
    if "uncertain" in n:
        return "uncertain"
    return "other"


def _has_pred_masks(model_dir: Path, iteration: int) -> bool:
    return (model_dir / "test" / f"ours_{iteration}_text" / "test_mask").is_dir()


def _scan_models(output_root: Path, scenes: List[str], iteration: int) -> List[ModelInfo]:
    models: List[ModelInfo] = []
    if not output_root.is_dir():
        raise FileNotFoundError(f"Output root not found: {output_root}")
    for p in sorted([d for d in output_root.iterdir() if d.is_dir()]):
        if not _has_pred_masks(p, iteration):
            continue
        name = p.name
        models.append(
            ModelInfo(
                name=name,
                path=p,
                scene=_infer_scene(name, scenes),
                variant=_infer_variant(name),
            )
        )
    return models


def _pick_baseline(models: List[ModelInfo]) -> Optional[ModelInfo]:
    baselines = [m for m in models if m.variant == "baseline"]
    if not baselines:
        return None
    # Prefer canonical naming.
    for key in ["_baseline", "verify_", "baseline"]:
        cands = [m for m in baselines if key in m.name]
        if cands:
            return sorted(cands, key=lambda x: len(x.name))[0]
    return sorted(baselines, key=lambda x: len(x.name))[0]


def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x:.4f}"


def _delta(a: Optional[float], b: Optional[float]) -> str:
    if a is None or b is None:
        return "-"
    return f"{(b - a):+.4f}"


def _print_markdown_table(rows: List[Dict[str, object]]):
    headers = [
        "scene",
        "variant",
        "model_dir",
        "mIoU",
        "mBIoU",
        "ΔmIoU(vs baseline)",
        "ΔmBIoU(vs baseline)",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        print(
            "| "
            + " | ".join(
                [
                    str(r.get("scene", "-")),
                    str(r.get("variant", "-")),
                    str(r.get("model_dir", "-")),
                    str(r.get("mIoU", "-")),
                    str(r.get("mBIoU", "-")),
                    str(r.get("delta_mIoU", "-")),
                    str(r.get("delta_mBIoU", "-")),
                ]
            )
            + " |"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument(
        "--output_root",
        default=str(REPO_ROOT / "output"),
        help="Root folder containing trained model outputs (default: output/)",
    )
    ap.add_argument(
        "--gt_root",
        default=str(REPO_ROOT / "data" / "lerf_mask"),
        help="Root folder containing LERF-Mask GT (default: data/lerf_mask/)",
    )
    ap.add_argument(
        "--out_dir",
        default=str(REPO_ROOT / "output" / "verify_logs"),
        help="Where to write table snapshots (default: output/verify_logs/)",
    )
    args = ap.parse_args()

    iteration = int(args.iteration)
    gt_root = _abs_in_repo(Path(args.gt_root))
    output_root = _abs_in_repo(Path(args.output_root))
    out_dir = _abs_in_repo(Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = _detect_scenes(gt_root, args.scenes)
    models_all = _scan_models(output_root, scenes, iteration)

    eval_mod = _load_eval_compare_module()
    evaluate_model = eval_mod.evaluate_model

    # Group models by scene
    by_scene: Dict[str, List[ModelInfo]] = {s: [] for s in scenes}
    unknown: List[ModelInfo] = []
    for m in models_all:
        if m.scene is None:
            unknown.append(m)
        else:
            by_scene.setdefault(m.scene, []).append(m)

    rows: List[Dict[str, object]] = []

    for scene in scenes:
        ms = by_scene.get(scene, [])
        baseline = _pick_baseline(ms)
        base_res = None
        if baseline is not None:
            base_res = evaluate_model(scene, str(baseline.path), iteration)
        # Sort variants for stable table output
        variant_rank = {"baseline": 0, "aniso_only": 1, "aniso": 2, "full": 3, "uncertain": 4, "other": 9}
        ms_sorted = sorted(ms, key=lambda x: (variant_rank.get(x.variant, 99), x.name))
        for m in ms_sorted:
            res = evaluate_model(scene, str(m.path), iteration)
            miou = res["mIoU"] if res else None
            mbiou = res["mBIoU"] if res else None
            b_miou = base_res["mIoU"] if base_res else None
            b_mbiou = base_res["mBIoU"] if base_res else None
            rows.append(
                {
                    "scene": scene,
                    "variant": m.variant,
                    "model_dir": _rel_to_repo(m.path),
                    "mIoU": _fmt(miou),
                    "mBIoU": _fmt(mbiou),
                    "delta_mIoU": _delta(b_miou, miou),
                    "delta_mBIoU": _delta(b_mbiou, mbiou),
                    "raw": res,
                    "baseline_model": _rel_to_repo(baseline.path) if baseline else None,
                }
            )

        if not ms:
            # Still record a placeholder row so the table shows missing scenes.
            rows.append(
                {
                    "scene": scene,
                    "variant": "(no models found)",
                    "model_dir": "-",
                    "mIoU": "-",
                    "mBIoU": "-",
                    "delta_mIoU": "-",
                    "delta_mBIoU": "-",
                    "raw": None,
                    "baseline_model": None,
                }
            )

    # Write snapshot files
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"ablation_table_iter{iteration}_{ts}.json"
    out_csv = out_dir / f"ablation_table_iter{iteration}_{ts}.csv"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "iteration": iteration,
                "scenes": scenes,
                "rows": rows,
                "unknown_models": [_rel_to_repo(m.path) for m in unknown],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "scene",
                "variant",
                "model_dir",
                "mIoU",
                "mBIoU",
                "delta_mIoU_vs_baseline",
                "delta_mBIoU_vs_baseline",
                "baseline_model",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.get("scene"),
                    r.get("variant"),
                    r.get("model_dir"),
                    r.get("mIoU"),
                    r.get("mBIoU"),
                    r.get("delta_mIoU"),
                    r.get("delta_mBIoU"),
                    r.get("baseline_model"),
                ]
            )

    print(f"# Ablation table (iter={iteration})")
    print(f"- snapshot_json: {_rel_to_repo(out_json)}")
    print(f"- snapshot_csv : {_rel_to_repo(out_csv)}")
    print()
    _print_markdown_table(rows)


if __name__ == "__main__":
    main()
