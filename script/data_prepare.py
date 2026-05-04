#!/usr/bin/env python3
"""
Data preparation script for Gaussian Grouping.

Input: A directory with only `images/` and `sparse/0/`
Output: A structure like `figurines/` with:
  - images/        (original, kept as-is)
  - images_train/  (training images, excludes test views)
  - object_mask/   (per-image instance segmentation from SAM Automatic Mask Generator)
  - test_mask/     (test views with text-prompt class masks from Grounded-SAM)
  - distorted/     (copy of sparse/, for MVS compatibility)

Test view selection (choose ONE):
  --num_test_views N     (default) take LAST N images
  --test_indices 0 5 10  use specific indices (0-based in sorted order)
  --test_files f1.jpg f2.jpg  use specific filenames

Usage:
  # Only directory structure
  python script/data_prepare.py --source_path data/my_scene \
    --num_test_views 4 --skip_sam

  # Pick specific views by index (e.g., first, middle, last)
  python script/data_prepare.py -s data/my_scene \
    --test_indices 0 75 150 299 \
    --sam_checkpoint path/to/sam_vit_h_4b8939.pth

  # Pick specific views by filename
  python script/data_prepare.py -s data/my_scene \
    --test_files 00001.jpg 00080.jpg 00150.jpg 00302.jpg \
    --sam_checkpoint path/to/sam_vit_h_4b8939.pth

  # Full: object_mask + test_mask (Grounded-SAM with text prompts)
  python script/data_prepare.py -s data/my_scene \
    --test_indices 0 75 150 299 \
    --sam_checkpoint path/to/sam_vit_h_4b8939.pth \
    --groundingdino_config path/to/GroundingDINO_SwinT_OGC.py \
    --groundingdino_checkpoint path/to/groundingdino_swint_ogc.pth \
    --text_prompts "green apple. red apple. toy chair. duck."
"""

import os
import sys
import argparse
import shutil
from os import makedirs, path
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Data preparation for Gaussian Grouping"
    )
    parser.add_argument(
        "--source_path",
        "-s",
        required=True,
        help="Path to source directory containing images/ and sparse/",
    )

    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "--num_test_views",
        "-n",
        type=int,
        default=4,
        help="Number of test views from the END (default: 4)",
    )
    test_group.add_argument(
        "--test_indices",
        type=int,
        nargs="+",
        default=None,
        help="Specific 0-based indices for test views, e.g., --test_indices 0 50 100 199",
    )
    test_group.add_argument(
        "--test_files",
        type=str,
        nargs="+",
        default=None,
        help="Specific filenames for test views, e.g., --test_files 00001.jpg 00100.jpg",
    )

    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default=None,
        help="Path to SAM checkpoint (e.g., sam_vit_h_4b8939.pth)",
    )
    parser.add_argument(
        "--sam_model_type",
        type=str,
        default="vit_h",
        help="SAM model type: vit_h, vit_l, vit_b (default: vit_h)",
    )
    parser.add_argument(
        "--groundingdino_config",
        type=str,
        default=None,
        help="Path to GroundingDINO config (e.g., GroundingDINO_SwinT_OGC.py)",
    )
    parser.add_argument(
        "--groundingdino_checkpoint",
        type=str,
        default=None,
        help="Path to GroundingDINO checkpoint (e.g., groundingdino_swint_ogc.pth)",
    )
    parser.add_argument(
        "--text_prompts",
        type=str,
        nargs="+",
        default=[],
        help='Text prompts for Grounded-SAM, e.g., "green apple. red apple. duck"',
    )
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.3,
        help="Box threshold for GroundingDINO (default: 0.3)",
    )
    parser.add_argument(
        "--text_threshold",
        type=float,
        default=0.45,
        help="Text threshold for GroundingDINO (default: 0.45)",
    )
    parser.add_argument(
        "--skip_object_mask",
        action="store_true",
        help="Skip object_mask/ generation (SAM Automatic)",
    )
    parser.add_argument(
        "--skip_test_mask",
        action="store_true",
        help="Skip test_mask/ generation (Grounded-SAM)",
    )
    parser.add_argument(
        "--skip_sam",
        action="store_true",
        help="Skip ALL SAM segmentation (equivalent to --skip_object_mask --skip_test_mask)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for SAM: cuda or cpu (default: cuda)",
    )
    return parser.parse_args()


def parse_text_prompts(prompts_list: List[str]) -> List[str]:
    all_prompts = []
    for item in prompts_list:
        for part in item.split("."):
            part = part.strip()
            if part:
                all_prompts.append(part)
    return all_prompts


def get_image_files(images_dir: str) -> List[str]:
    """Get sorted list of image files in images/ directory."""
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    files = []
    for f in sorted(os.listdir(images_dir)):
        if Path(f).suffix in exts:
            files.append(f)
    return files


def select_test_views(
    image_files: List[str],
    num_test: int = 4,
    test_indices: Optional[List[int]] = None,
    test_files: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Select test views based on user choice:
      - test_indices: specific 0-based indices in sorted order
      - test_files: specific filenames
      - num_test: take last N (default)

    Returns: (train_files, test_files)
    """
    if test_indices is not None:
        test_set = set()
        max_idx = len(image_files) - 1
        for idx in test_indices:
            if idx < 0 or idx > max_idx:
                print(
                    f"Warning: test index {idx} out of range [0, {max_idx}], ignoring"
                )
                continue
            test_set.add(image_files[idx])
        test_files_selected = [f for f in image_files if f in test_set]
        train_files = [f for f in image_files if f not in test_set]
        return train_files, test_files_selected

    if test_files is not None:
        available = set(image_files)
        requested = set(test_files)
        missing = requested - available
        if missing:
            print(f"Warning: test files not found: {sorted(missing)}")
        test_set = requested & available
        test_files_selected = [f for f in image_files if f in test_set]
        train_files = [f for f in image_files if f not in test_set]
        return train_files, test_files_selected

    if num_test <= 0:
        return image_files, []
    if num_test >= len(image_files):
        print(f"Warning: num_test_views={num_test} >= total images={len(image_files)}")
        return [], image_files
    train_files = image_files[:-num_test]
    test_files_selected = image_files[-num_test:]
    return train_files, test_files_selected


def create_images_train(
    source_path: str, train_files: List[str], test_files: List[str]
):
    """
    Create images_train/ directory (copy training images).
    Also: copy test images with test_0.jpg naming if not already present.
    """
    images_dir = path.join(source_path, "images")
    images_train_dir = path.join(source_path, "images_train")
    makedirs(images_train_dir, exist_ok=True)

    for fname in train_files:
        src = path.join(images_dir, fname)
        dst = path.join(images_train_dir, fname)
        if not path.exists(dst):
            shutil.copy2(src, dst)

    for i, fname in enumerate(test_files):
        src = path.join(images_dir, fname)
        dst_test_name = path.join(images_dir, f"test_{i}.jpg")
        if not path.exists(dst_test_name):
            img = Image.open(src).convert("RGB")
            img.save(dst_test_name, quality=95)


def create_distorted(source_path: str):
    """Create distorted/ directory by copying sparse/ contents."""
    sparse_dir = path.join(source_path, "sparse")
    distorted_dir = path.join(source_path, "distorted")
    distorted_sparse_dir = path.join(distorted_dir, "sparse")

    if not path.exists(sparse_dir):
        print(f"Warning: sparse/ not found at {sparse_dir}, skipping distorted/")
        return

    if path.exists(distorted_sparse_dir):
        print(f"distorted/sparse/ already exists, skipping copy")
        return

    makedirs(distorted_dir, exist_ok=True)
    shutil.copytree(sparse_dir, distorted_sparse_dir)
    print(f"Copied sparse/ -> distorted/sparse/")


def create_test_mask_dirs(source_path: str, test_files: List[str]):
    test_mask_dir = path.join(source_path, "test_mask")
    makedirs(test_mask_dir, exist_ok=True)

    for i in range(len(test_files)):
        view_dir = path.join(test_mask_dir, str(i))
        makedirs(view_dir, exist_ok=True)


def run_sam_segmentation(
    source_path: str,
    image_files: List[str],
    sam_checkpoint: str,
    sam_model_type: str,
    device: str = "cuda",
):
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    images_dir = path.join(source_path, "images")
    object_mask_dir = path.join(source_path, "object_mask")
    makedirs(object_mask_dir, exist_ok=True)

    print(f"Loading SAM {sam_model_type} from {sam_checkpoint} ...")
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )

    print(f"Generating object_mask for {len(image_files)} images ...")
    for fname in tqdm(image_files):
        img_path = path.join(images_dir, fname)
        base_name = Path(fname).stem
        mask_path = path.join(object_mask_dir, f"{base_name}.png")

        if path.exists(mask_path):
            continue

        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        masks = mask_generator.generate(img_np)

        if len(masks) == 0:
            seg = np.zeros(img_np.shape[:2], dtype=np.uint8)
        else:
            sorted_masks = sorted(masks, key=lambda x: x["area"], reverse=True)
            seg = np.zeros(img_np.shape[:2], dtype=np.int32)
            for idx, m in enumerate(sorted_masks):
                seg[m["segmentation"]] = idx + 1
            seg = seg.astype(np.uint8)

        Image.fromarray(seg).save(mask_path)

    print(f"SAM segmentation done. object_mask/ at {object_mask_dir}")


def run_grounded_sam_test_mask(
    source_path: str,
    test_files: List[str],
    text_prompts: List[str],
    sam_checkpoint: str,
    sam_model_type: str,
    groundingdino_config: str,
    groundingdino_checkpoint: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.45,
    device: str = "cuda",
):
    from segment_anything import sam_model_registry, SamPredictor
    from ext.grounded_sam import load_model_local, grouned_sam_output

    images_dir = path.join(source_path, "images")
    test_mask_dir = path.join(source_path, "test_mask")
    makedirs(test_mask_dir, exist_ok=True)
    for i in range(len(test_files)):
        makedirs(path.join(test_mask_dir, str(i)), exist_ok=True)

    print("=" * 70)
    print("Loading models for Grounded-SAM...")
    print(f"  SAM: {sam_checkpoint}")
    print(f"  GroundingDINO config: {groundingdino_config}")
    print(f"  GroundingDINO checkpoint: {groundingdino_checkpoint}")
    print(f"  Text prompts: {text_prompts}")
    print("=" * 70)

    print("Loading SAM...")
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    print("Loading GroundingDINO...")
    groundingdino_model = load_model_local(
        groundingdino_config, groundingdino_checkpoint, device=device
    )

    print(
        f"Generating test_mask for {len(test_files)} views x {len(text_prompts)} classes ..."
    )

    for view_idx, fname in enumerate(tqdm(test_files, desc="Test views")):
        img_path = path.join(images_dir, fname)
        view_dir = path.join(test_mask_dir, str(view_idx))

        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        for prompt in text_prompts:
            safe_name = prompt.strip()
            mask_path = path.join(view_dir, f"{safe_name}.png")

            if path.exists(mask_path):
                continue

            try:
                mask_bool, _ = grouned_sam_output(
                    groundingdino_model,
                    sam_predictor,
                    prompt,
                    img_np,
                    BOX_TRESHOLD=box_threshold,
                    TEXT_TRESHOLD=text_threshold,
                    device=device,
                )

                mask_uint8 = (mask_bool.cpu().numpy().astype(np.uint8)) * 255
                Image.fromarray(mask_uint8).save(mask_path)
            except Exception as e:
                print(f"  Warning: failed for view={view_idx}, prompt='{prompt}': {e}")
                empty_mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
                Image.fromarray(empty_mask).save(mask_path)

    print(f"Grounded-SAM done. test_mask/ at {test_mask_dir}")


def main():
    args = parse_args()
    source_path = args.source_path

    images_dir = path.join(source_path, "images")
    sparse_dir = path.join(source_path, "sparse")

    if not path.exists(images_dir) or not path.exists(sparse_dir):
        print(f"Error: expected images/ and sparse/ in {source_path}")
        sys.exit(1)

    image_files = get_image_files(images_dir)
    if len(image_files) == 0:
        print(f"Error: no images found in {images_dir}")
        sys.exit(1)

    print(f"Found {len(image_files)} images in {images_dir}")

    train_files, test_files = select_test_views(
        image_files,
        num_test=args.num_test_views,
        test_indices=args.test_indices,
        test_files=args.test_files,
    )
    print(f"Train: {len(train_files)}, Test: {len(test_files)}")
    if len(test_files) > 0:
        if args.test_indices is not None:
            print(f"  Test selection: by --test_indices")
        elif args.test_files is not None:
            print(f"  Test selection: by --test_files")
        else:
            print(f"  Test selection: last {args.num_test_views} images")
        print(f"  Test views: {test_files}")

    text_prompts = parse_text_prompts(args.text_prompts)
    if text_prompts:
        print(f"Text prompts: {text_prompts}")

    create_images_train(source_path, train_files, test_files)
    print("images_train/ created")

    create_distorted(source_path)

    create_test_mask_dirs(source_path, test_files)

    skip_all_sam = args.skip_sam
    skip_object_mask = skip_all_sam or args.skip_object_mask
    skip_test_mask = skip_all_sam or args.skip_test_mask

    sam_ready = bool(args.sam_checkpoint and path.exists(args.sam_checkpoint))
    grounded_sam_ready = bool(
        sam_ready
        and text_prompts
        and args.groundingdino_config
        and path.exists(args.groundingdino_config)
        and args.groundingdino_checkpoint
        and path.exists(args.groundingdino_checkpoint)
    )

    object_mask_dir = path.join(source_path, "object_mask")
    makedirs(object_mask_dir, exist_ok=True)

    if skip_all_sam:
        print("--skip_sam set, skipping ALL SAM segmentation")
    else:
        if not skip_object_mask:
            if sam_ready:
                all_images_for_sam = image_files.copy()
                for i in range(len(test_files)):
                    extra_test_name = f"test_{i}.jpg"
                    if path.exists(path.join(images_dir, extra_test_name)):
                        all_images_for_sam.append(extra_test_name)
                run_sam_segmentation(
                    source_path,
                    all_images_for_sam,
                    args.sam_checkpoint,
                    args.sam_model_type,
                    args.device,
                )
            else:
                print("")
                print("=" * 70)
                print("SAM checkpoint not provided or not found.")
                print("Skipping object_mask/ generation.")
                print("")
                print("To generate object_mask/, provide:")
                print("  --sam_checkpoint /path/to/sam_vit_h_4b8939.pth")
                print("=" * 70)
                print("")

        if not skip_test_mask and text_prompts:
            if grounded_sam_ready:
                run_grounded_sam_test_mask(
                    source_path,
                    test_files,
                    text_prompts,
                    args.sam_checkpoint,
                    args.sam_model_type,
                    args.groundingdino_config,
                    args.groundingdino_checkpoint,
                    args.box_threshold,
                    args.text_threshold,
                    args.device,
                )
            else:
                print("")
                print("=" * 70)
                print("Text prompts provided but Grounded-SAM not fully ready.")
                print("Skipping test_mask/ auto-generation.")
                print("")
                print("To auto-generate test_mask/, ALL of these are needed:")
                print("  --sam_checkpoint /path/to/sam_vit_h_4b8939.pth")
                print("  --groundingdino_config /path/to/GroundingDINO_SwinT_OGC.py")
                print(
                    "  --groundingdino_checkpoint /path/to/groundingdino_swint_ogc.pth"
                )
                print(f'  --text_prompts "{". ".join(text_prompts)}"')
                print("=" * 70)
                print("")

    print("")
    print("=" * 70)
    print("Data preparation done.")
    print("")
    print("Created/verified:")
    print(f"  images/          -> {len(image_files)} original images")
    print(f"  images_train/    -> {len(train_files)} training images")
    if skip_object_mask:
        print(f"  object_mask/     -> skipped (--skip_object_mask or --skip_sam)")
    elif sam_ready:
        print(f"  object_mask/     -> per-image instance segmentation (auto-generated)")
    else:
        print(f"  object_mask/     -> placeholder (need SAM checkpoint to generate)")
    if grounded_sam_ready and not skip_test_mask:
        print(
            f"  test_mask/       -> {len(test_files)} views x {len(text_prompts)} classes (auto-generated)"
        )
    else:
        if text_prompts and not skip_test_mask:
            print(
                f"  test_mask/       -> {len(test_files)} view dirs (needs GroundingDINO to auto-generate)"
            )
        else:
            print(f"  test_mask/       -> {len(test_files)} view dirs (placeholders)")
    if len(test_files) > 0:
        print(f"  Test files:      -> {test_files}")
    print(f"  distorted/       -> copy of sparse/ for MVS compatibility")
    print("")
    print("To change test views later:")
    print("  Use --test_indices or --test_files to re-select")
    print("  But NOTE: images_train/ and test_*.jpg naming depend on selection")
    print("=" * 70)


if __name__ == "__main__":
    main()
