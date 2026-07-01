"""渲染高斯场景的俯视 RGB 与俯视分割 mask。

默认行为：
1. 读取训练输出目录中的 point_cloud.ply 与 classifier.pth；
2. 构造一个俯视相机（默认按 ScanNet 的 y 轴向上，从上往下看）；
3. 渲染俯视 RGB 图；
4. 渲染俯视分割 mask 图；
5. 另外保存一张左右拼接图，左边 RGB，右边 mask。

示例：
    python vis_seg.py --scene scene0000_00
    python vis_seg.py --model-path output4/scene0000_00_iter30000
"""

from __future__ import annotations

import argparse
import colorsys
import math
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from gaussian_renderer import render
from scene.cameras import MiniCam
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import getProjectionMatrix


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "output4"


def id2rgb(idx: int, max_num_obj: int = 1024) -> np.ndarray:
    if not 0 <= idx <= max_num_obj:
        raise ValueError(f"ID {idx} 超出范围 [0, {max_num_obj}]")
    if idx == 0:
        return np.zeros((3,), dtype=np.uint8)

    golden_ratio = 1.6180339887
    h = (idx * golden_ratio) % 1.0
    s = 0.5 + (idx % 2) * 0.5
    l = 0.5
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)


def visualize_obj(labels: np.ndarray) -> np.ndarray:
    rgb_mask = np.zeros((*labels.shape[-2:], 3), dtype=np.uint8)
    for obj_id in np.unique(labels):
        rgb_mask[labels == obj_id] = id2rgb(int(obj_id))
    return rgb_mask


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    if tensor.ndim == 4:
        tensor = tensor[0]
    tensor = tensor.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return (tensor * 255.0).round().astype(np.uint8)


def predict_mask_with_background(
    rendering_obj: torch.Tensor,
    classifier: torch.nn.Module,
    bg_threshold: float,
) -> np.ndarray:
    logits = classifier(rendering_obj)
    pred_obj = torch.argmax(logits, dim=0)
    feat_norm = torch.norm(rendering_obj, dim=0)
    pred_obj = pred_obj.clone()
    pred_obj[feat_norm <= bg_threshold] = 0
    return pred_obj.detach().cpu().numpy().astype(np.uint8)


def save_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def resolve_model_dir(output_root: Path, scene: str, iteration: int, model_path: str | None) -> Path:
    if model_path is not None:
        model_dir = Path(model_path).expanduser().resolve()
    else:
        model_dir = (output_root / f"{scene}_iter{iteration}").resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    return model_dir


def infer_scene_name(model_dir: Path) -> str:
    name = model_dir.name
    return name.split("_iter")[0] if "_iter" in name else name


def infer_iteration(model_dir: Path, fallback: int) -> int:
    name = model_dir.name
    if "_iter" in name:
        suffix = name.rsplit("_iter", 1)[1]
        try:
            return int(suffix)
        except ValueError:
            pass
    return fallback


def load_cfg_args(model_dir: Path) -> Namespace:
    cfg_path = model_dir / "cfg_args"
    if not cfg_path.exists():
        raise FileNotFoundError(f"缺少配置文件: {cfg_path}")
    cfg_text = cfg_path.read_text()
    return eval(cfg_text, {"Namespace": Namespace})


def make_pipeline() -> SimpleNamespace:
    return SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
    )


def build_classifier(num_objects: int, num_classes: int, classifier_path: Path) -> torch.nn.Module:
    classifier = torch.nn.Conv2d(num_objects, num_classes, kernel_size=1).cuda()
    try:
        state_dict = torch.load(classifier_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(classifier_path, map_location="cpu")
    classifier.load_state_dict(state_dict)
    classifier.eval()
    return classifier


def choose_axes(up_axis: str) -> Tuple[int, Tuple[int, int]]:
    axis_map = {"x": 0, "y": 1, "z": 2}
    up_idx = axis_map[up_axis]
    plane_axes = tuple(i for i in range(3) if i != up_idx)
    return up_idx, plane_axes


def robust_bounds(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = np.percentile(xyz, 1.0, axis=0)
    hi = np.percentile(xyz, 99.0, axis=0)
    center = 0.5 * (lo + hi)
    return lo, hi, center


def build_topdown_camera(
    xyz: np.ndarray,
    width: int,
    height: int,
    up_axis: str,
    fov_deg: float,
    margin: float,
    znear: float,
    zfar: float,
) -> MiniCam:
    up_idx, plane_axes = choose_axes(up_axis)
    lo, hi, center = robust_bounds(xyz)
    span = np.maximum(hi - lo, 1e-6)

    plane_span_x = float(span[plane_axes[0]])
    plane_span_y = float(span[plane_axes[1]])
    depth_span = float(span[up_idx])

    aspect = width / max(height, 1)
    fovy = math.radians(fov_deg)
    fovx = 2.0 * math.atan(math.tan(fovy * 0.5) * aspect)

    dist_x = 0.5 * plane_span_x * margin / max(math.tan(fovx * 0.5), 1e-6)
    dist_y = 0.5 * plane_span_y * margin / max(math.tan(fovy * 0.5), 1e-6)
    cam_distance = max(dist_x, dist_y) + 0.6 * depth_span + 0.5

    axis_vecs = np.eye(3, dtype=np.float32)
    world_up_dir = axis_vecs[up_idx]
    image_up_hint = axis_vecs[plane_axes[1]]

    cam_center = center + world_up_dir * cam_distance
    target = center

    forward = target - cam_center
    forward = forward / np.linalg.norm(forward)
    right = np.cross(image_up_hint, forward)
    right = right / np.linalg.norm(right)
    true_up = np.cross(forward, right)
    true_up = true_up / np.linalg.norm(true_up)

    rotation_c2w = np.stack([right, true_up, forward], axis=1).astype(np.float32)
    translation_w2c = -(rotation_c2w.T @ cam_center.astype(np.float32))

    world_view_transform = torch.tensor(
        np.vstack(
            [
                np.hstack([rotation_c2w.T, translation_w2c[:, None]]),
                np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            ]
        ),
        dtype=torch.float32,
        device="cuda",
    ).transpose(0, 1)
    projection_matrix = getProjectionMatrix(
        znear=znear,
        zfar=zfar,
        fovX=fovx,
        fovY=fovy,
    ).transpose(0, 1).cuda()
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)

    return MiniCam(
        width=width,
        height=height,
        fovy=fovy,
        fovx=fovx,
        znear=znear,
        zfar=zfar,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
    )


def derive_output_paths(model_dir: Path, out: str | None) -> Tuple[Path, Path, Path]:
    if out is None:
        concat_path = model_dir / "topdown_rgb_mask.png"
    else:
        concat_path = Path(out).expanduser().resolve()
    stem = concat_path.stem
    suffix = concat_path.suffix or ".png"
    rgb_path = concat_path.with_name(f"{stem}_rgb{suffix}")
    mask_path = concat_path.with_name(f"{stem}_mask{suffix}")
    return rgb_path, mask_path, concat_path


def render_topdown(
    model_dir: Path,
    scene_name: str,
    iteration: int,
    num_classes: int,
    white_background: bool,
    width: int,
    height: int,
    up_axis: str,
    fov_deg: float,
    margin: float,
    znear: float,
    zfar: float,
    bg_threshold: float,
    out: str | None,
) -> Tuple[Path, Path, Path]:
    gaussians = GaussianModel(3)
    ply_path = model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    classifier_path = model_dir / "point_cloud" / f"iteration_{iteration}" / "classifier.pth"

    if not ply_path.exists():
        raise FileNotFoundError(f"点云文件不存在: {ply_path}")
    if not classifier_path.exists():
        raise FileNotFoundError(f"分类器文件不存在: {classifier_path}")

    with torch.no_grad():
        gaussians.load_ply(str(ply_path))
        classifier = build_classifier(gaussians.num_objects, num_classes, classifier_path)
        pipeline = make_pipeline()
        bg_color = [1, 1, 1] if white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        xyz = gaussians.get_xyz.detach().cpu().numpy()
        top_cam = build_topdown_camera(
            xyz=xyz,
            width=width,
            height=height,
            up_axis=up_axis,
            fov_deg=fov_deg,
            margin=margin,
            znear=znear,
            zfar=zfar,
        )

        results = render(top_cam, gaussians, pipeline, background)
        rendering = results["render"]
        rendering_obj = results["render_object"]

        rgb = tensor_to_uint8_image(rendering)
        pred_obj = predict_mask_with_background(rendering_obj, classifier, bg_threshold)
        mask = visualize_obj(pred_obj)
        concat = np.hstack([rgb, mask])

        rgb_path, mask_path, concat_path = derive_output_paths(model_dir, out)
        save_image(rgb, rgb_path)
        save_image(mask, mask_path)
        save_image(concat, concat_path)

    print(f"[vis_seg] scene          : {scene_name}")
    print(f"[vis_seg] model_dir      : {model_dir}")
    print(f"[vis_seg] rgb image      : {rgb_path}")
    print(f"[vis_seg] mask image     : {mask_path}")
    print(f"[vis_seg] concat image   : {concat_path}")
    return rgb_path, mask_path, concat_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="渲染高斯场景俯视 RGB 与俯视 mask")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="训练输出根目录")
    parser.add_argument("--scene", type=str, default="scene0000_00", help="场景名，例如 scene0000_00")
    parser.add_argument("--iteration", type=int, default=30000, help="迭代轮数")
    parser.add_argument("--model-path", type=str, default=None, help="直接指定模型目录")
    parser.add_argument("--out", type=str, default=None, help="拼接图输出路径；会同时保存 rgb 与 mask")
    parser.add_argument("--up-axis", type=str, default="y", choices=["x", "y", "z"], help="从哪一轴的正方向往下看；ScanNet 默认 y")
    parser.add_argument("--width", type=int, default=1024, help="渲染宽度")
    parser.add_argument("--height", type=int, default=1024, help="渲染高度")
    parser.add_argument("--fov-deg", type=float, default=60.0, help="俯视相机竖直视场角")
    parser.add_argument("--margin", type=float, default=1.15, help="场景覆盖边缘冗余")
    parser.add_argument("--znear", type=float, default=0.01, help="近裁剪面")
    parser.add_argument("--zfar", type=float, default=100.0, help="远裁剪面")
    parser.add_argument("--bg-threshold", type=float, default=1e-6, help="render_object 特征范数小于该值的像素视为背景")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    model_dir = resolve_model_dir(output_root, args.scene, args.iteration, args.model_path)
    scene_name = infer_scene_name(model_dir)
    iteration = infer_iteration(model_dir, args.iteration)
    cfg = load_cfg_args(model_dir)

    num_classes = getattr(cfg, "num_classes", 256)
    white_background = bool(getattr(cfg, "white_background", False))

    render_topdown(
        model_dir=model_dir,
        scene_name=scene_name,
        iteration=iteration,
        num_classes=num_classes,
        white_background=white_background,
        width=args.width,
        height=args.height,
        up_axis=args.up_axis,
        fov_deg=args.fov_deg,
        margin=args.margin,
        znear=args.znear,
        zfar=args.zfar,
        bg_threshold=args.bg_threshold,
        out=args.out,
    )


if __name__ == "__main__":
    main()
