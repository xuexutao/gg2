#!/usr/bin/env python3

import argparse
import io
import os
import shutil
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData, PlyElement


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.colmap_loader import rotmat2qvec


COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a raw ScanNet scene (.sens + mesh) into Gaussian Grouping COLMAP-style input"
    )
    parser.add_argument("--scene_dir", required=True, help="Path to raw ScanNet scene directory")
    parser.add_argument("--output_path", required=True, help="Output directory with images/ and sparse/0/")
    parser.add_argument(
        "--frame_skip",
        type=int,
        default=80,
        help="Keep one frame every N frames from .sens (default: 80)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=72,
        help="Maximum number of RGB frames to export (default: 72)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Exported RGB width; height is scaled automatically if --height not set (default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Exported RGB height; 0 means auto-scale to preserve aspect ratio",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing output_path before writing",
    )
    return parser.parse_args()


def read_matrix(f):
    return np.asarray(struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)


def read_header(f):
    version = struct.unpack("I", f.read(4))[0]
    if version != 4:
        raise ValueError(f"Unsupported .sens version: {version}")
    strlen = struct.unpack("Q", f.read(8))[0]
    sensor_name = f.read(strlen).decode("utf-8", errors="ignore")
    intrinsic_color = read_matrix(f)
    extrinsic_color = read_matrix(f)
    intrinsic_depth = read_matrix(f)
    extrinsic_depth = read_matrix(f)
    color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack("i", f.read(4))[0]]
    _depth_compression_type = struct.unpack("i", f.read(4))[0]
    color_width = struct.unpack("I", f.read(4))[0]
    color_height = struct.unpack("I", f.read(4))[0]
    _depth_width = struct.unpack("I", f.read(4))[0]
    _depth_height = struct.unpack("I", f.read(4))[0]
    _depth_shift = struct.unpack("f", f.read(4))[0]
    num_frames = struct.unpack("Q", f.read(8))[0]
    return {
        "sensor_name": sensor_name,
        "intrinsic_color": intrinsic_color,
        "extrinsic_color": extrinsic_color,
        "intrinsic_depth": intrinsic_depth,
        "extrinsic_depth": extrinsic_depth,
        "color_compression_type": color_compression_type,
        "color_width": color_width,
        "color_height": color_height,
        "num_frames": num_frames,
    }


def iter_frames(f, num_frames):
    for frame_idx in range(num_frames):
        camera_to_world = read_matrix(f)
        _timestamp_color = struct.unpack("Q", f.read(8))[0]
        _timestamp_depth = struct.unpack("Q", f.read(8))[0]
        color_size_bytes = struct.unpack("Q", f.read(8))[0]
        depth_size_bytes = struct.unpack("Q", f.read(8))[0]
        color_data = f.read(color_size_bytes)
        f.seek(depth_size_bytes, os.SEEK_CUR)
        yield frame_idx, camera_to_world, color_data


def decode_color(color_data, compression_type):
    if compression_type != "jpeg":
        raise ValueError(f"Unsupported color compression type: {compression_type}")
    encoded = np.frombuffer(color_data, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Failed to decode JPEG frame from .sens")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def pose_to_colmap(camera_to_world):
    if not np.isfinite(camera_to_world).all():
        return None, None
    rotation_c2w = camera_to_world[:3, :3]
    translation_c2w = camera_to_world[:3, 3]
    rotation_w2c = rotation_c2w.T
    translation_w2c = -rotation_w2c @ translation_c2w
    qvec = rotmat2qvec(rotation_w2c)
    return qvec, translation_w2c


def pick_mesh_file(scene_dir: Path):
    scene_name = scene_dir.name
    candidates = [
        scene_dir / f"{scene_name}_vh_clean_2.ply",
        scene_dir / f"{scene_name}_vh_clean.ply",
        scene_dir / f"{scene_name}.ply",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No ScanNet mesh/point cloud .ply found in {scene_dir}")


def convert_mesh_to_pointcloud_ply(src_mesh: Path, dst_ply: Path):
    ply = PlyData.read(str(src_mesh))
    vertex = ply["vertex"].data
    names = vertex.dtype.names
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)

    if all(c in names for c in ("red", "green", "blue")):
        rgb = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.uint8)
    else:
        rgb = np.full((xyz.shape[0], 3), 127, dtype=np.uint8)

    normals = np.zeros_like(xyz, dtype=np.float32)
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements["x"] = xyz[:, 0]
    elements["y"] = xyz[:, 1]
    elements["z"] = xyz[:, 2]
    elements["nx"] = normals[:, 0]
    elements["ny"] = normals[:, 1]
    elements["nz"] = normals[:, 2]
    elements["red"] = rgb[:, 0]
    elements["green"] = rgb[:, 1]
    elements["blue"] = rgb[:, 2]
    PlyData([PlyElement.describe(elements, "vertex")], text=False).write(str(dst_ply))


def ensure_clean_dir(dst: Path, overwrite: bool):
    if dst.exists() and overwrite:
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    scene_dir = Path(args.scene_dir).resolve()
    output_path = Path(args.output_path).resolve()
    images_dir = output_path / "images"
    sparse_dir = output_path / "sparse" / "0"

    if not scene_dir.exists():
        raise FileNotFoundError(f"scene_dir not found: {scene_dir}")

    sens_path = scene_dir / f"{scene_dir.name}.sens"
    if not sens_path.exists():
        raise FileNotFoundError(f"ScanNet .sens file not found: {sens_path}")

    ensure_clean_dir(output_path, args.overwrite)
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = pick_mesh_file(scene_dir)
    convert_mesh_to_pointcloud_ply(mesh_path, sparse_dir / "points3D.ply")

    with open(sens_path, "rb") as f:
        header = read_header(f)
        src_w = header["color_width"]
        src_h = header["color_height"]
        dst_w = args.width if args.width > 0 else src_w
        dst_h = args.height if args.height > 0 else int(round(src_h * (dst_w / src_w)))
        scale_x = dst_w / float(src_w)
        scale_y = dst_h / float(src_h)

        fx = float(header["intrinsic_color"][0, 0] * scale_x)
        fy = float(header["intrinsic_color"][1, 1] * scale_y)
        cx = float(header["intrinsic_color"][0, 2] * scale_x)
        cy = float(header["intrinsic_color"][1, 2] * scale_y)

        kept = []
        for frame_idx, camera_to_world, color_data in iter_frames(f, header["num_frames"]):
            if frame_idx % max(args.frame_skip, 1) != 0:
                continue
            qvec, tvec = pose_to_colmap(camera_to_world)
            if qvec is None:
                continue
            image_rgb = decode_color(color_data, header["color_compression_type"])
            if (dst_w, dst_h) != (src_w, src_h):
                image_rgb = cv2.resize(image_rgb, (dst_w, dst_h), interpolation=cv2.INTER_AREA)

            image_name = f"{len(kept):06d}.jpg"
            image_path = images_dir / image_name
            ok = cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError(f"Failed to write image: {image_path}")

            kept.append(
                {
                    "image_id": len(kept) + 1,
                    "source_frame_id": frame_idx,
                    "qvec": qvec,
                    "tvec": tvec,
                    "camera_id": 1,
                    "image_name": image_name,
                }
            )
            if len(kept) >= args.max_frames:
                break

    if len(kept) < 8:
        raise RuntimeError(f"Only kept {len(kept)} valid frames; this is too few for training")

    cameras_txt = sparse_dir / "cameras.txt"
    images_txt = sparse_dir / "images.txt"
    points3d_txt = sparse_dir / "points3D.txt"
    manifest_tsv = output_path / "frame_manifest.tsv"

    with open(cameras_txt, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {dst_w} {dst_h} {fx:.8f} {fy:.8f} {cx:.8f} {cy:.8f}\n")

    with open(images_txt, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for item in kept:
            q = item["qvec"]
            t = item["tvec"]
            f.write(
                f"{item['image_id']} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} "
                f"{t[0]:.10f} {t[1]:.10f} {t[2]:.10f} {item['camera_id']} {item['image_name']}\n\n"
            )

    with open(manifest_tsv, "w") as f:
        f.write("image_name\tsource_frame_id\n")
        for item in kept:
            f.write(f"{item['image_name']}\t{item['source_frame_id']}\n")

    with open(points3d_txt, "w") as f:
        f.write("# Empty placeholder; loader will use points3D.ply directly.\n")

    print("=" * 80)
    print(f"ScanNet scene converted: {scene_dir.name}")
    print(f"Sensor: {header['sensor_name']}")
    print(f"Frames exported: {len(kept)} / {header['num_frames']}")
    print(f"Export size: {dst_w}x{dst_h}")
    print(f"Images dir: {images_dir}")
    print(f"Frame manifest: {manifest_tsv}")
    print(f"Sparse dir: {sparse_dir}")
    print(f"Point cloud converted from: {mesh_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
