"""从俯视角直接渲染高斯场景，并输出分割结果。

输出结构尽量对齐 `render.py`：
    <model_path>/bev/ours_<iter>/
        renders/00000.png
        objects_pred/00000.png
        objects_feature16/00000.png
        concat/00000.png

其中 `concat/00000.png` 为横向拼接结果：
    [ RGB Render | Pred Seg | Feature PCA ]
"""

from __future__ import annotations

import argparse
import json
import math
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from gaussian_renderer import render
from scene import Scene, GaussianModel
from scene.cameras import MiniCam
from utils.graphics_utils import getProjectionMatrix


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "output4"


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
    if "_iter" in name:
        return name.split("_iter")[0]
    return name


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
        raise FileNotFoundError(f"未找到 cfg_args: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    cfg = eval(content, {"Namespace": Namespace}, {})
    cfg.model_path = str(model_dir)
    return cfg


def choose_up_axis(xyz: np.ndarray, up_axis: str) -> int:
    axis_map = {"x": 0, "y": 1, "z": 2}
    if up_axis in axis_map:
        return axis_map[up_axis]

    robust_min = np.percentile(xyz, 1.0, axis=0)
    robust_max = np.percentile(xyz, 99.0, axis=0)
    span = robust_max - robust_min
    return int(np.argmin(span))


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        raise ValueError("向量长度过小，无法归一化。")
    return vec / norm


def axis_name(axis_idx: int) -> str:
    return ["x", "y", "z"][axis_idx]


def render_tensor_to_rgb(rendering: torch.Tensor) -> np.ndarray:
    image = torch.clamp(rendering, 0.0, 1.0)
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    return (image * 255.0).astype(np.uint8)


def feature_to_rgb(features: torch.Tensor) -> np.ndarray:
    """将 feature map 用 PCA 前三主成分映射为 RGB，不依赖 sklearn。"""
    c, h, w = features.shape
    feat = features.detach().float().cpu().reshape(c, -1).transpose(0, 1).numpy()
    feat = feat - feat.mean(axis=0, keepdims=True)
    u, s, _vh = np.linalg.svd(feat, full_matrices=False)
    pca = u[:, :3] * s[:3]
    pca = pca.reshape(h, w, 3)
    pca_min = pca.min(axis=(0, 1), keepdims=True)
    pca_max = pca.max(axis=(0, 1), keepdims=True)
    pca = (pca - pca_min) / np.clip(pca_max - pca_min, 1e-8, None)
    return (pca * 255.0).astype(np.uint8)


def id2rgb(obj_id: int, max_num_obj: int = 256) -> np.ndarray:
    if not 0 <= obj_id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")
    if obj_id == 0:
        return np.zeros((3,), dtype=np.uint8)

    golden_ratio = 1.6180339887
    h = (obj_id * golden_ratio) % 1.0
    s = 0.5 + (obj_id % 2) * 0.5
    l = 0.5

    import colorsys

    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)


def visualize_obj(objects: np.ndarray) -> np.ndarray:
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    for obj_id in np.unique(objects):
        rgb_mask[objects == obj_id] = id2rgb(int(obj_id))
    return rgb_mask


def build_bev_camera(
    gaussians: GaussianModel,
    image_width: int,
    image_height: int,
    up_axis: str,
    fov_deg: float,
    margin_ratio: float,
) -> tuple[MiniCam, dict]:
    xyz = gaussians.get_xyz.detach().cpu().numpy()

    robust_min = np.percentile(xyz, 1.0, axis=0)
    robust_max = np.percentile(xyz, 99.0, axis=0)
    center = 0.5 * (robust_min + robust_max)
    span = np.maximum(robust_max - robust_min, 1e-4)

    up_idx = choose_up_axis(xyz, up_axis)
    planar_axes = [axis for axis in range(3) if axis != up_idx]

    half_w = 0.5 * span[planar_axes[0]] * margin_ratio
    half_h = 0.5 * span[planar_axes[1]] * margin_ratio
    half_max = max(half_w, half_h)
    fov = math.radians(fov_deg)
    distance = half_max / math.tan(0.5 * fov)
    distance += 0.75 * span[up_idx] + 1e-3

    eye = center.copy()
    eye[up_idx] = robust_max[up_idx] + distance
    target = center.copy()

    image_up_candidates = {
        0: np.array([0.0, 0.0, 1.0], dtype=np.float32),
        1: np.array([0.0, 0.0, -1.0], dtype=np.float32),
        2: np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }
    image_up_world = image_up_candidates[up_idx]

    forward = normalize(target - eye)
    if abs(float(np.dot(forward, image_up_world))) > 0.98:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(forward, fallback))) > 0.98:
            fallback = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        image_up_world = fallback

    right = normalize(np.cross(image_up_world, forward))
    cam_up = normalize(np.cross(forward, right))

    rot_w2c = np.stack([right, cam_up, forward], axis=0).astype(np.float32)
    trans_w2c = (-rot_w2c @ eye.astype(np.float32)).astype(np.float32)

    world_view = np.eye(4, dtype=np.float32)
    world_view[:3, :3] = rot_w2c
    world_view[:3, 3] = trans_w2c
    world_view_t = torch.tensor(world_view, dtype=torch.float32, device="cuda").transpose(0, 1)

    znear = 0.01
    zfar = max(100.0, float(distance + 3.0 * span[up_idx] + 10.0))
    fovy = fov
    aspect = float(image_width) / float(image_height)
    fovx = 2.0 * math.atan(math.tan(fovy * 0.5) * aspect)
    projection = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
    full_proj = world_view_t.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)

    camera = MiniCam(
        image_width,
        image_height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_t,
        full_proj,
    )
    meta = {
        "eye": eye.tolist(),
        "target": target.tolist(),
        "up_axis": axis_name(up_idx),
        "planar_axes": [axis_name(planar_axes[0]), axis_name(planar_axes[1])],
        "fov_deg": fov_deg,
        "margin_ratio": margin_ratio,
        "znear": znear,
        "zfar": zfar,
    }
    return camera, meta


def build_output_dirs(model_dir: Path, iteration: int) -> dict[str, Path]:
    base = model_dir / "bev" / f"ours_{iteration}"
    paths = {
        "base": base,
        "renders": base / "renders",
        "objects_pred": base / "objects_pred",
        "objects_feature16": base / "objects_feature16",
        "concat": base / "concat",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_scene_and_classifier(model_dir: Path, iteration: int) -> tuple[Namespace, Scene, GaussianModel, torch.nn.Module]:
    cfg = load_cfg_args(model_dir)
    gaussians = GaussianModel(cfg.sh_degree)
    scene = Scene(cfg, gaussians, load_iteration=iteration, shuffle=False)

    classifier = torch.nn.Conv2d(gaussians.num_objects, cfg.num_classes, kernel_size=1)
    classifier.cuda()
    classifier_path = model_dir / "point_cloud" / f"iteration_{scene.loaded_iter}" / "classifier.pth"
    classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))
    classifier.eval()
    return cfg, scene, gaussians, classifier


def save_bev_outputs(
    out_dirs: dict[str, Path],
    iteration: int,
    scene_name: str,
    render_rgb: np.ndarray,
    pred_rgb: np.ndarray,
    feature_rgb: np.ndarray,
    camera_meta: dict,
) -> Path:
    image_name = "00000.png"
    Image.fromarray(render_rgb).save(out_dirs["renders"] / image_name)
    Image.fromarray(pred_rgb).save(out_dirs["objects_pred"] / image_name)
    Image.fromarray(feature_rgb).save(out_dirs["objects_feature16"] / image_name)

    concat = np.hstack([render_rgb, pred_rgb, feature_rgb]).astype(np.uint8)
    concat_path = out_dirs["concat"] / image_name
    Image.fromarray(concat).save(concat_path)

    meta_path = out_dirs["base"] / "bev_camera.json"
    payload = {
        "scene_name": scene_name,
        "iteration": iteration,
        "concat_layout": ["render", "pred_seg", "feature16_pca"],
        "camera": camera_meta,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return concat_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="俯视角高斯分割渲染")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="训练输出根目录")
    parser.add_argument("--scene", type=str, default="scene0000_00", help="场景名，例如 scene0000_00")
    parser.add_argument("--iteration", type=int, default=30000, help="训练迭代轮数")
    parser.add_argument("--model-path", type=str, default=None, help="直接指定模型目录，例如 output4/scene0000_00_iter30000")
    parser.add_argument("--up-axis", type=str, default="y", choices=["auto", "x", "y", "z"], help="俯视向上轴，ScanNet 通常是 y")
    parser.add_argument("--image-width", type=int, default=1024, help="输出图宽")
    parser.add_argument("--image-height", type=int, default=1024, help="输出图高")
    parser.add_argument("--fov-deg", type=float, default=60.0, help="俯视相机垂直视场角")
    parser.add_argument("--margin-ratio", type=float, default=1.10, help="包围场景时的视野冗余比例")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = resolve_model_dir(Path(args.output_root).expanduser().resolve(), args.scene, args.iteration, args.model_path)
    iteration = infer_iteration(model_dir, args.iteration)
    scene_name = infer_scene_name(model_dir)

    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)

    with torch.no_grad():
        cfg, scene, gaussians, classifier = load_scene_and_classifier(model_dir, iteration)
        background = torch.tensor(
            [1.0, 1.0, 1.0] if cfg.white_background else [0.0, 0.0, 0.0],
            dtype=torch.float32,
            device="cuda",
        )
        bev_camera, camera_meta = build_bev_camera(
            gaussians=gaussians,
            image_width=args.image_width,
            image_height=args.image_height,
            up_axis=args.up_axis,
            fov_deg=args.fov_deg,
            margin_ratio=args.margin_ratio,
        )
        outputs = render(bev_camera, gaussians, pipe, background)
        rendering = outputs["render"]
        rendering_obj = outputs["render_object"]

        logits = classifier(rendering_obj)
        pred_obj = torch.argmax(logits, dim=0)

        render_rgb = render_tensor_to_rgb(rendering)
        pred_rgb = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))
        feature_rgb = feature_to_rgb(rendering_obj)

        out_dirs = build_output_dirs(model_dir, scene.loaded_iter)
        concat_path = save_bev_outputs(
            out_dirs=out_dirs,
            iteration=scene.loaded_iter,
            scene_name=scene_name,
            render_rgb=render_rgb,
            pred_rgb=pred_rgb,
            feature_rgb=feature_rgb,
            camera_meta=camera_meta,
        )

    print(f"[vis_seg2] scene          : {scene_name}")
    print(f"[vis_seg2] model_dir      : {model_dir}")
    print(f"[vis_seg2] output base    : {out_dirs['base']}")
    print(f"[vis_seg2] concat image   : {concat_path}")
    print(f"[vis_seg2] up axis        : {camera_meta['up_axis']}")
    print(f"[vis_seg2] planar axes    : {camera_meta['planar_axes']}")


if __name__ == "__main__":
    main()
