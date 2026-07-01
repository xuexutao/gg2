#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/51/new_workspace/gg2"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output4}"
ITERATIONS="${ITERATIONS:-30000}"
WORKER_ID="${WORKER_ID:-3842835}"
PYTHON_BIN="${PYTHON_BIN:-/home/tiger/miniconda3/envs/group/bin/python}"
SCENE_GLOB="${SCENE_GLOB:-scene*}"
START_INDEX="${START_INDEX:-0}"
MAX_SCENES="${MAX_SCENES:-0}"
FORCE_RENDER=0
DRY_RUN=0
SKIP_TRAIN=0
SKIP_TEST=0

usage() {
  cat <<EOF
用法：$(basename "$0") [可选参数]

功能：
  只对已经训练完成的 gg2 ScanNet 模型执行渲染，不做预处理、不做训练。
  默认同时渲染 train/test 两个 split，从而补上 opensplat3d 能看到、但旧 gg2 批处理脚本
  因为 --skip_train 没有导出的训练视角 PCA 图。

可选参数：
  --output_root PATH      输出根目录，默认: ${OUTPUT_ROOT}
  --iterations N          渲染的模型迭代数，默认: ${ITERATIONS}
  --worker_id ID          使用的 mlx worker，默认: ${WORKER_ID}
  --python_bin PATH       Python 解释器，默认: ${PYTHON_BIN}
  --scene_glob PATTERN    只渲染匹配的场景名，默认: ${SCENE_GLOB}
  --start_index N         从第 N 个匹配场景开始，默认: ${START_INDEX}
  --max_scenes N          最多渲染多少个场景，0 表示不限制，默认: ${MAX_SCENES}
  --skip_train            不渲染 train split
  --skip_test             不渲染 test split
  --force_render          即使输出已存在也重新渲染
  --dry_run               只打印将要执行的场景列表，不真正运行
  --help                  显示本帮助

说明：
  - 默认行为是同时渲染 train/test。
  - gg2 的 PCA 输出目录为：
      <model_dir>/train/ours_<iter>/objects_feature16/
      <model_dir>/test/ours_<iter>/objects_feature16/
  - 如果你想和 opensplat3d 对齐视角，重点看 train/test 中对应 split 的结果。

示例：
  $(basename "$0")
  $(basename "$0") --scene_glob 'scene0000_*' --max_scenes 2 --dry_run
  $(basename "$0") --scene_glob 'scene0000_00' --force_render
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    --iterations)
      ITERATIONS="$2"; shift 2 ;;
    --worker_id)
      WORKER_ID="$2"; shift 2 ;;
    --python_bin)
      PYTHON_BIN="$2"; shift 2 ;;
    --scene_glob)
      SCENE_GLOB="$2"; shift 2 ;;
    --start_index)
      START_INDEX="$2"; shift 2 ;;
    --max_scenes)
      MAX_SCENES="$2"; shift 2 ;;
    --skip_train)
      SKIP_TRAIN=1; shift ;;
    --skip_test)
      SKIP_TEST=1; shift ;;
    --force_render)
      FORCE_RENDER=1; shift ;;
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

[[ -d "$OUTPUT_ROOT" ]] || { echo "[ERROR] output_root 不存在: $OUTPUT_ROOT" >&2; exit 2; }

if [[ -x "$PYTHON_BIN" ]]; then
  LOCAL_PYTHON="$PYTHON_BIN"
elif command -v python3 >/dev/null 2>&1; then
  LOCAL_PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  LOCAL_PYTHON="$(command -v python)"
else
  echo "[ERROR] 本机无法找到可执行的 python/python3，用于列举场景。" >&2
  exit 2
fi

if (( SKIP_TRAIN == 1 && SKIP_TEST == 1 )); then
  echo "[ERROR] --skip_train 和 --skip_test 不能同时指定，否则没有任何内容可渲染。" >&2
  exit 2
fi

mapfile -t all_scenes < <("$LOCAL_PYTHON" - <<'PY' "$OUTPUT_ROOT" "$SCENE_GLOB" "$ITERATIONS"
from pathlib import Path
import fnmatch
import sys

output_root = Path(sys.argv[1])
scene_glob = sys.argv[2]
iterations = sys.argv[3]

items = []
suffix = f"_iter{iterations}"
for path in sorted(output_root.iterdir()):
    if not path.is_dir():
        continue
    name = path.name
    if not name.endswith(suffix):
        continue
    scene = name[:-len(suffix)]
    if fnmatch.fnmatch(scene, scene_glob):
        items.append(scene)

for scene in items:
    print(scene)
PY
)

if (( START_INDEX > 0 )); then
  all_scenes=("${all_scenes[@]:START_INDEX}")
fi
if (( MAX_SCENES > 0 )) && (( ${#all_scenes[@]} > MAX_SCENES )); then
  all_scenes=("${all_scenes[@]:0:MAX_SCENES}")
fi

if (( ${#all_scenes[@]} == 0 )); then
  echo "[ERROR] 没有匹配到任何已训练模型。scene_glob=${SCENE_GLOB}, iterations=${ITERATIONS}" >&2
  exit 3
fi

LOG_DIR="${OUTPUT_ROOT}/logs"
MASTER_LOG="${LOG_DIR}/run_all_scannet_render_iter${ITERATIONS}.log"
SUMMARY_TSV="${LOG_DIR}/run_all_scannet_render_iter${ITERATIONS}_summary.tsv"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "[Batch ScanNet Render Only]"
echo "output_root  = $OUTPUT_ROOT"
echo "scene_glob   = $SCENE_GLOB"
echo "scene_count  = ${#all_scenes[@]}"
echo "iterations   = $ITERATIONS"
echo "worker_id    = $WORKER_ID"
echo "python_bin   = $PYTHON_BIN"
echo "skip_train   = $SKIP_TRAIN"
echo "skip_test    = $SKIP_TEST"
echo "force_render = $FORCE_RENDER"
echo "master_log   = $MASTER_LOG"
echo "summary_tsv  = $SUMMARY_TSV"
echo "============================================================"

printf 'scene\tstatus\tmodel_dir\n' > "$SUMMARY_TSV"

render_outputs_ready() {
  local model_dir="$1"
  local train_ready=1
  local test_ready=1

  if (( SKIP_TRAIN == 0 )); then
    [[ -d "$model_dir/train/ours_${ITERATIONS}/objects_feature16" ]] || train_ready=0
  fi
  if (( SKIP_TEST == 0 )); then
    [[ -d "$model_dir/test/ours_${ITERATIONS}/objects_feature16" ]] || test_ready=0
  fi

  (( train_ready == 1 && test_ready == 1 ))
}

for idx in "${!all_scenes[@]}"; do
  scene="${all_scenes[$idx]}"
  model_dir="${OUTPUT_ROOT}/${scene}_iter${ITERATIONS}"
  render_log="${LOG_DIR}/${scene}_render_only_iter${ITERATIONS}.log"

  echo "[$((idx + 1))/${#all_scenes[@]}] $scene" | tee -a "$MASTER_LOG"

  if [[ ! -f "$model_dir/cfg_args" ]]; then
    printf '%s\tMISSING_CFG\t%s\n' "$scene" "$model_dir" >> "$SUMMARY_TSV"
    echo "[WARN] 跳过：缺少 cfg_args -> $model_dir" | tee -a "$MASTER_LOG"
    continue
  fi

  if [[ ! -d "$model_dir/point_cloud/iteration_${ITERATIONS}" ]]; then
    printf '%s\tMISSING_MODEL\t%s\n' "$scene" "$model_dir" >> "$SUMMARY_TSV"
    echo "[WARN] 跳过：缺少训练结果 -> $model_dir/point_cloud/iteration_${ITERATIONS}" | tee -a "$MASTER_LOG"
    continue
  fi

  if (( DRY_RUN == 1 )); then
    printf '%s\tDRY_RUN\t%s\n' "$scene" "$model_dir" >> "$SUMMARY_TSV"
    continue
  fi

  if (( FORCE_RENDER == 0 )) && render_outputs_ready "$model_dir"; then
    printf '%s\tSKIP_EXISTS\t%s\n' "$scene" "$model_dir" >> "$SUMMARY_TSV"
    echo "[SKIP] 已存在目标 PCA 输出，跳过：$scene" | tee -a "$MASTER_LOG"
    continue
  fi

  cmd=(
    mlx worker login "$WORKER_ID" --
    "$PYTHON_BIN" "$REPO_ROOT/render.py"
    -m "$model_dir"
    --iteration "$ITERATIONS"
    --num_classes 256
  )

  if (( SKIP_TRAIN == 1 )); then
    cmd+=(--skip_train)
  fi
  if (( SKIP_TEST == 1 )); then
    cmd+=(--skip_test)
  fi

  if "${cmd[@]}" 2>&1 | tee "$render_log" >> "$MASTER_LOG"; then
    printf '%s\tOK\t%s\n' "$scene" "$model_dir" >> "$SUMMARY_TSV"
  else
    printf '%s\tFAILED\t%s\n' "$scene" "$model_dir" >> "$SUMMARY_TSV"
    echo "[WARN] scene failed: $scene ，继续下一个。" | tee -a "$MASTER_LOG"
  fi
done

echo "============================================================"
echo "[DONE] 批量渲染脚本执行完成"
echo "master_log  = $MASTER_LOG"
echo "summary_tsv = $SUMMARY_TSV"
echo "============================================================"
