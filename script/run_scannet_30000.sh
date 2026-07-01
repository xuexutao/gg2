#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/51/new_workspace/gg2"
PYTHON_BIN="${PYTHON_BIN:-/home/tiger/miniconda3/envs/group/bin/python}"
WORKER_ID="${WORKER_ID:-3842835}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output4}"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/config/gaussian_dataset/train.json}"
ITERATIONS="${ITERATIONS:-30000}"
FRAME_SKIP="${FRAME_SKIP:-80}"
MAX_FRAMES="${MAX_FRAMES:-120}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-0}"
NUM_TEST_VIEWS="${NUM_TEST_VIEWS:-6}"
RESOLUTION="${RESOLUTION:-1}"
PORT="${PORT:-6013}"
FORCE_PREP=0
FORCE_TRAIN=0
SCENE_DIR=""
SCENE_NAME=""

usage() {
  cat <<EOF
用法：$(basename "$0") --scene_dir /path/to/scannet/sceneXXXX_YY [可选参数]

必选参数：
  --scene_dir PATH         ScanNet 原始场景目录，目录内应包含 .sens / _2d-instance-filt.zip / vh_clean_2.ply

可选参数：
  --scene_name NAME        输出实验名，默认使用场景目录名
  --output_root PATH       输出根目录，默认: ${OUTPUT_ROOT}
  --iterations N           训练迭代数，默认: ${ITERATIONS}
  --frame_skip N           从 .sens 抽帧步长，默认: ${FRAME_SKIP}
  --max_frames N           最多导出多少帧，默认: ${MAX_FRAMES}
  --width N                导出 RGB 宽度，默认: ${WIDTH}
  --height N               导出 RGB 高度，0 表示按比例缩放，默认: ${HEIGHT}
  --num_test_views N       留作测试集的视角数，默认: ${NUM_TEST_VIEWS}
  --resolution N           训练时 -r 参数，默认: ${RESOLUTION}
  --worker_id ID           mlx worker id，默认: ${WORKER_ID}
  --python_bin PATH        Python 解释器，默认: ${PYTHON_BIN}
  --config_file PATH       训练配置 JSON，默认: ${CONFIG_FILE}
  --port PORT              train.py GUI 端口，默认: ${PORT}
  --force_prepare          强制重新预处理数据
  --force_train            强制重新训练并重新渲染
  --help                   显示本帮助

示例：
  $(basename "$0") \
    --scene_dir /mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/51/gaussian-splatting_test/data/scannet/scans/scene0000_00 \
    --scene_name scannet_scene0000_00 \
    --iterations 30000 \
    --frame_skip 80 \
    --max_frames 120 \
    --width 640
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene_dir)
      SCENE_DIR="$2"
      shift 2
      ;;
    --scene_name)
      SCENE_NAME="$2"
      shift 2
      ;;
    --output_root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --frame_skip)
      FRAME_SKIP="$2"
      shift 2
      ;;
    --max_frames)
      MAX_FRAMES="$2"
      shift 2
      ;;
    --width)
      WIDTH="$2"
      shift 2
      ;;
    --height)
      HEIGHT="$2"
      shift 2
      ;;
    --num_test_views)
      NUM_TEST_VIEWS="$2"
      shift 2
      ;;
    --resolution)
      RESOLUTION="$2"
      shift 2
      ;;
    --worker_id)
      WORKER_ID="$2"
      shift 2
      ;;
    --python_bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --config_file)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --force_prepare)
      FORCE_PREP=1
      shift
      ;;
    --force_train)
      FORCE_TRAIN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] 未知参数: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$SCENE_DIR" ]]; then
  echo "[ERROR] 必须提供 --scene_dir" >&2
  usage
  exit 1
fi

SCENE_DIR="$($PYTHON_BIN -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$SCENE_DIR")"
RAW_SCENE_ID="$(basename "$SCENE_DIR")"
if [[ -z "$SCENE_NAME" ]]; then
  SCENE_NAME="$RAW_SCENE_ID"
fi

DATA_DIR="${OUTPUT_ROOT}/${SCENE_NAME}_data"
MODEL_DIR="${OUTPUT_ROOT}/${SCENE_NAME}_iter${ITERATIONS}"
LOG_DIR="${OUTPUT_ROOT}/logs"
PREP_LOG="${LOG_DIR}/${SCENE_NAME}_prepare.log"
TRAIN_LOG="${LOG_DIR}/${SCENE_NAME}_train_iter${ITERATIONS}.log"
RENDER_LOG="${LOG_DIR}/${SCENE_NAME}_render_iter${ITERATIONS}.log"
MASK_ZIP="${SCENE_DIR}/${RAW_SCENE_ID}_2d-instance-filt.zip"
MANIFEST_TSV="${DATA_DIR}/frame_manifest.tsv"

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

echo "============================================================"
echo "[ScanNet Pipeline]"
echo "scene_dir    = $SCENE_DIR"
echo "scene_name   = $SCENE_NAME"
echo "data_dir     = $DATA_DIR"
echo "model_dir    = $MODEL_DIR"
echo "worker_id    = $WORKER_ID"
echo "iterations   = $ITERATIONS"
echo "frame_skip   = $FRAME_SKIP"
echo "max_frames   = $MAX_FRAMES"
echo "image_size   = ${WIDTH}x${HEIGHT}"
echo "test_views   = $NUM_TEST_VIEWS"
echo "resolution   = $RESOLUTION"
echo "config_file  = $CONFIG_FILE"
echo "python_bin   = $PYTHON_BIN"
echo "============================================================"

[[ -f "${SCENE_DIR}/${RAW_SCENE_ID}.sens" ]] || { echo "[ERROR] 缺少 .sens 文件" >&2; exit 2; }
[[ -f "$MASK_ZIP" ]] || { echo "[ERROR] 缺少 2D instance zip: $MASK_ZIP" >&2; exit 2; }
[[ -f "$CONFIG_FILE" ]] || { echo "[ERROR] 缺少 config 文件: $CONFIG_FILE" >&2; exit 2; }

run_prepare() {
  echo "[1/4] 生成 COLMAP 风格 ScanNet 数据..."
  "$PYTHON_BIN" "$REPO_ROOT/script/prepare_scannet_scene.py" \
    --scene_dir "$SCENE_DIR" \
    --output_path "$DATA_DIR" \
    --frame_skip "$FRAME_SKIP" \
    --max_frames "$MAX_FRAMES" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --overwrite \
    2>&1 | tee "$PREP_LOG"
}

extract_masks() {
  echo "[2/4] 提取 ScanNet 2D instance masks -> object_mask/..."
  "$PYTHON_BIN" - "$MASK_ZIP" "$MANIFEST_TSV" "$DATA_DIR/object_mask" "$DATA_DIR/images/000000.jpg" <<'PY'
import csv
import io
import os
import sys
import zipfile
from pathlib import Path

from PIL import Image

mask_zip = Path(sys.argv[1])
manifest_tsv = Path(sys.argv[2])
object_mask_dir = Path(sys.argv[3])
sample_image = Path(sys.argv[4])

object_mask_dir.mkdir(parents=True, exist_ok=True)
with Image.open(sample_image) as img:
    target_size = img.size

with manifest_tsv.open("r", newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

with zipfile.ZipFile(mask_zip) as zf:
    for row in rows:
        image_name = row["image_name"]
        frame_id = row["source_frame_id"]
        src_name = f"instance-filt/{frame_id}.png"
        if src_name not in zf.namelist():
            raise FileNotFoundError(f"Mask not found in zip: {src_name}")
        out_path = object_mask_dir / (Path(image_name).stem + ".png")
        with zf.open(src_name) as fp:
            img = Image.open(io.BytesIO(fp.read()))
            img.load()
            if img.size != target_size:
                img = img.resize(target_size, Image.NEAREST)
            img.save(out_path)

print(f"mask_count={len(rows)} output_dir={object_mask_dir}")
PY
}

run_split_prepare() {
  echo "[3/4] 生成 images_train / test_mask 占位 / distorted ..."
  "$PYTHON_BIN" "$REPO_ROOT/script/data_prepare.py" \
    -s "$DATA_DIR" \
    --num_test_views "$NUM_TEST_VIEWS" \
    --skip_sam \
    2>&1 | tee -a "$PREP_LOG"
}

build_iter_list() {
  local target="$1"
  local items=()
  local last=""
  for v in 1000 7000 "$target"; do
    if (( v <= target )) && [[ "$v" != "$last" ]]; then
      items+=("$v")
      last="$v"
    fi
  done
  printf '%s\n' "${items[@]}"
}

train_on_worker() {
  echo "[4/4] 在 worker ${WORKER_ID} 上训练 ${ITERATIONS} iterations ..."
  mapfile -t iter_list < <(build_iter_list "$ITERATIONS")
  mlx worker login "$WORKER_ID" -- \
    "$PYTHON_BIN" "$REPO_ROOT/train.py" \
      -s "$DATA_DIR" \
      -m "$MODEL_DIR" \
      -r "$RESOLUTION" \
      --train_split \
      --eval \
      --iterations "$ITERATIONS" \
      --test_iterations "${iter_list[@]}" \
      --save_iterations "${iter_list[@]}" \
      --config_file "$CONFIG_FILE" \
      --num_classes 256 \
      --images images \
      --object_path object_mask \
      --port "$PORT" \
      2>&1 | tee "$TRAIN_LOG"
}

render_on_worker() {
  echo "[5/5] 在 worker ${WORKER_ID} 上渲染测试集 ..."
  mlx worker login "$WORKER_ID" -- \
    "$PYTHON_BIN" "$REPO_ROOT/render.py" \
      -m "$MODEL_DIR" \
      --num_classes 256 \
      --skip_train \
      2>&1 | tee "$RENDER_LOG"
}

if [[ "$FORCE_PREP" -eq 1 || ! -f "$MANIFEST_TSV" ]]; then
  run_prepare
else
  echo "[SKIP] 预处理已存在: $MANIFEST_TSV"
fi

extract_masks
run_split_prepare

if [[ "$FORCE_TRAIN" -eq 1 ]]; then
  rm -rf "$MODEL_DIR"
fi

if [[ -d "$MODEL_DIR/point_cloud/iteration_${ITERATIONS}" ]]; then
  echo "[SKIP] 已存在训练结果: $MODEL_DIR/point_cloud/iteration_${ITERATIONS}"
else
  train_on_worker
fi

if [[ -d "$MODEL_DIR/test/ours_${ITERATIONS}" ]]; then
  echo "[SKIP] 已存在渲染结果: $MODEL_DIR/test/ours_${ITERATIONS}"
else
  render_on_worker
fi

echo "============================================================"
echo "[DONE] ScanNet 运行脚本执行完成"
echo "data_dir  = $DATA_DIR"
echo "model_dir = $MODEL_DIR"
echo "prep_log  = $PREP_LOG"
echo "train_log = $TRAIN_LOG"
echo "render_log= $RENDER_LOG"
echo "============================================================"
