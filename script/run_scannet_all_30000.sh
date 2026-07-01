#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/51/new_workspace/gg2"
SCANS_ROOT="${SCANS_ROOT:-/mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/51/gaussian-splatting_test/data/scannet/scans}"
SINGLE_SCENE_SCRIPT="${SINGLE_SCENE_SCRIPT:-${REPO_ROOT}/script/run_scannet_30000.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output4}"
ITERATIONS="${ITERATIONS:-30000}"
FRAME_SKIP="${FRAME_SKIP:-80}"
MAX_FRAMES="${MAX_FRAMES:-120}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-0}"
NUM_TEST_VIEWS="${NUM_TEST_VIEWS:-6}"
RESOLUTION="${RESOLUTION:-1}"
WORKER_ID="${WORKER_ID:-3842835}"
PYTHON_BIN="${PYTHON_BIN:-/home/tiger/miniconda3/envs/group/bin/python}"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/config/gaussian_dataset/train.json}"
BASE_PORT="${BASE_PORT:-6013}"
MAX_SCENES="${MAX_SCENES:-0}"
START_INDEX="${START_INDEX:-0}"
SCENE_GLOB="${SCENE_GLOB:-scene*}"
FORCE_PREP=0
FORCE_TRAIN=0
DRY_RUN=0

usage() {
  cat <<EOF
用法：$(basename "$0") [可选参数]

功能：顺序遍历 ${SCANS_ROOT} 下所有 ScanNet 场景，并逐个调用单场景脚本 run_scannet_30000.sh。

可选参数：
  --scans_root PATH       ScanNet 场景根目录，默认: ${SCANS_ROOT}
  --output_root PATH      输出根目录，默认: ${OUTPUT_ROOT}
  --iterations N          每个场景训练迭代数，默认: ${ITERATIONS}
  --frame_skip N          .sens 抽帧步长，默认: ${FRAME_SKIP}
  --max_frames N          每个场景最多导出帧数，默认: ${MAX_FRAMES}
  --width N               导出 RGB 宽度，默认: ${WIDTH}
  --height N              导出 RGB 高度，0 表示等比例，默认: ${HEIGHT}
  --num_test_views N      每个场景测试视角数，默认: ${NUM_TEST_VIEWS}
  --resolution N          训练 -r 参数，默认: ${RESOLUTION}
  --worker_id ID          使用的 mlx worker，默认: ${WORKER_ID}
  --python_bin PATH       Python 解释器，默认: ${PYTHON_BIN}
  --config_file PATH      训练配置，默认: ${CONFIG_FILE}
  --base_port PORT        首个场景 train.py 使用的端口，默认: ${BASE_PORT}
  --scene_glob PATTERN    只跑匹配的场景名，默认: ${SCENE_GLOB}
  --start_index N         从第 N 个匹配场景开始，默认: ${START_INDEX}
  --max_scenes N          最多跑多少个场景，0 表示不限制，默认: ${MAX_SCENES}
  --force_prepare         强制每个场景重新预处理
  --force_train           强制每个场景重新训练/渲染
  --dry_run               只打印将要执行的场景列表，不真正运行
  --help                  显示本帮助

示例：
  $(basename "$0")

  $(basename "$0") --scene_glob 'scene000*' --max_scenes 3 --dry_run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scans_root)
      SCANS_ROOT="$2"; shift 2 ;;
    --output_root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    --iterations)
      ITERATIONS="$2"; shift 2 ;;
    --frame_skip)
      FRAME_SKIP="$2"; shift 2 ;;
    --max_frames)
      MAX_FRAMES="$2"; shift 2 ;;
    --width)
      WIDTH="$2"; shift 2 ;;
    --height)
      HEIGHT="$2"; shift 2 ;;
    --num_test_views)
      NUM_TEST_VIEWS="$2"; shift 2 ;;
    --resolution)
      RESOLUTION="$2"; shift 2 ;;
    --worker_id)
      WORKER_ID="$2"; shift 2 ;;
    --python_bin)
      PYTHON_BIN="$2"; shift 2 ;;
    --config_file)
      CONFIG_FILE="$2"; shift 2 ;;
    --base_port)
      BASE_PORT="$2"; shift 2 ;;
    --scene_glob)
      SCENE_GLOB="$2"; shift 2 ;;
    --start_index)
      START_INDEX="$2"; shift 2 ;;
    --max_scenes)
      MAX_SCENES="$2"; shift 2 ;;
    --force_prepare)
      FORCE_PREP=1; shift ;;
    --force_train)
      FORCE_TRAIN=1; shift ;;
    --dry_run)
      DRY_RUN=1; shift ;;
    --help|-h)
      usage; exit 0 ;;
    *)
      echo "[ERROR] 未知参数: $1" >&2
      usage
      exit 1
      ;;
  esac
done

[[ -d "$SCANS_ROOT" ]] || { echo "[ERROR] scans_root 不存在: $SCANS_ROOT" >&2; exit 2; }
[[ -x "$SINGLE_SCENE_SCRIPT" ]] || { echo "[ERROR] 单场景脚本不可执行: $SINGLE_SCENE_SCRIPT" >&2; exit 2; }

mapfile -t all_scenes < <(python - <<'PY' "$SCANS_ROOT" "$SCENE_GLOB"
from pathlib import Path
import fnmatch
import sys
root = Path(sys.argv[1])
pat = sys.argv[2]
scenes = sorted([p.name for p in root.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, pat)])
for s in scenes:
    print(s)
PY
)

if (( START_INDEX > 0 )); then
  all_scenes=("${all_scenes[@]:START_INDEX}")
fi
if (( MAX_SCENES > 0 )) && (( ${#all_scenes[@]} > MAX_SCENES )); then
  all_scenes=("${all_scenes[@]:0:MAX_SCENES}")
fi

if (( ${#all_scenes[@]} == 0 )); then
  echo "[ERROR] 没有匹配到任何场景。scene_glob=${SCENE_GLOB}" >&2
  exit 3
fi

LOG_DIR="${OUTPUT_ROOT}/logs"
MASTER_LOG="${LOG_DIR}/run_all_scannet_iter${ITERATIONS}.log"
SUMMARY_TSV="${LOG_DIR}/run_all_scannet_iter${ITERATIONS}_summary.tsv"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "[Batch ScanNet Pipeline]"
echo "scans_root   = $SCANS_ROOT"
echo "scene_glob   = $SCENE_GLOB"
echo "scene_count  = ${#all_scenes[@]}"
echo "output_root  = $OUTPUT_ROOT"
echo "iterations   = $ITERATIONS"
echo "worker_id    = $WORKER_ID"
echo "base_port    = $BASE_PORT"
echo "master_log   = $MASTER_LOG"
echo "summary_tsv  = $SUMMARY_TSV"
echo "============================================================"

printf 'scene\tstatus\tport\n' > "$SUMMARY_TSV"

for idx in "${!all_scenes[@]}"; do
  scene="${all_scenes[$idx]}"
  port=$((BASE_PORT + idx))
  scene_dir="${SCANS_ROOT}/${scene}"

  echo "[$((idx + 1))/${#all_scenes[@]}] $scene (port=$port)" | tee -a "$MASTER_LOG"

  if (( DRY_RUN == 1 )); then
    printf '%s\tDRY_RUN\t%s\n' "$scene" "$port" >> "$SUMMARY_TSV"
    continue
  fi

  cmd=(
    "$SINGLE_SCENE_SCRIPT"
    --scene_dir "$scene_dir"
    --scene_name "$scene"
    --output_root "$OUTPUT_ROOT"
    --iterations "$ITERATIONS"
    --frame_skip "$FRAME_SKIP"
    --max_frames "$MAX_FRAMES"
    --width "$WIDTH"
    --height "$HEIGHT"
    --num_test_views "$NUM_TEST_VIEWS"
    --resolution "$RESOLUTION"
    --worker_id "$WORKER_ID"
    --python_bin "$PYTHON_BIN"
    --config_file "$CONFIG_FILE"
    --port "$port"
  )

  if (( FORCE_PREP == 1 )); then
    cmd+=(--force_prepare)
  fi
  if (( FORCE_TRAIN == 1 )); then
    cmd+=(--force_train)
  fi

  if "${cmd[@]}" >> "$MASTER_LOG" 2>&1; then
    printf '%s\tOK\t%s\n' "$scene" "$port" >> "$SUMMARY_TSV"
  else
    printf '%s\tFAILED\t%s\n' "$scene" "$port" >> "$SUMMARY_TSV"
    echo "[WARN] scene failed: $scene ，继续下一个。" | tee -a "$MASTER_LOG"
  fi
done

echo "============================================================"
echo "[DONE] 批量脚本执行完成"
echo "master_log  = $MASTER_LOG"
echo "summary_tsv = $SUMMARY_TSV"
echo "============================================================"
