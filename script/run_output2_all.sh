#!/usr/bin/env bash
set -euo pipefail

# Full rerun: train.py -> render_lerf_mask.py -> batch table
# Outputs are written to: output2/

REPO_ROOT="/mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/51/new_workspace/gg2"
ITERATION="${ITERATION:-30000}"
LOG_DIR_REL="output2/verify_logs"
MAIN_LOG_REL="${LOG_DIR_REL}/pipeline_main_iter${ITERATION}.log"
STATUS_REL="${LOG_DIR_REL}/pipeline_status_iter${ITERATION}.json"
DONE_MARK_REL="${LOG_DIR_REL}/PIPELINE_DONE_iter${ITERATION}"

source ~/miniconda3/bin/activate gg3
cd "$REPO_ROOT"

mkdir -p output2 "$LOG_DIR_REL"

# Make the pipeline restartable + observable.
touch "$MAIN_LOG_REL"

now_ts() { date -Iseconds; }

write_status() {
  local stage="$1"; shift
  local scene="${1:-}"; shift || true
  local variant="${1:-}"; shift || true
  printf '{"time":"%s","iteration":%s,"stage":"%s","scene":"%s","variant":"%s"}\n' \
    "$(now_ts)" "$ITERATION" "$stage" "$scene" "$variant" > "$STATUS_REL"
}

on_exit() {
  local code=$?
  if [[ $code -eq 0 ]]; then
    write_status "done" "" ""
    echo "$(now_ts) [DONE] pipeline finished" >> "$MAIN_LOG_REL"
    touch "$DONE_MARK_REL"
  else
    write_status "failed" "" ""
    echo "$(now_ts) [FAIL] pipeline exited with code=$code" >> "$MAIN_LOG_REL"
  fi
  exit $code
}
trap on_exit EXIT

scenes=(figurines ramen teatime room)

# Variants (ablation) to rerun.
variants=(baseline aniso_only full uncertain)

cfg_for_variant() {
  local v="$1"
  case "$v" in
    baseline) echo "config/gaussian_dataset/train.json" ;;
    aniso_only) echo "config/gaussian_dataset/train_aniso_only.json" ;;
    full) echo "config/gaussian_dataset/train_full.json" ;;
    uncertain) echo "config/gaussian_dataset/train_aniso_uncertain.json" ;;
    *) echo "[FATAL] unknown variant=$v" 1>&2; exit 2 ;;
  esac
}

set_scene_params() {
  local scene="$1"
  # Defaults match existing cfg_args for LERF-Mask scenes in this repo.
  case "$scene" in
    figurines)
      DATASET="data/lerf_mask/figurines"; RES=1; TRAIN_SPLIT_FLAG="--train_split" ;;
    ramen)
      DATASET="data/lerf_mask/ramen"; RES=1; TRAIN_SPLIT_FLAG="--train_split" ;;
    teatime)
      DATASET="data/lerf_mask/teatime"; RES=-1; TRAIN_SPLIT_FLAG="" ;;
    room)
      DATASET="data/lerf_mask/room"; RES=4; TRAIN_SPLIT_FLAG="--train_split" ;;
    *)
      echo "[FATAL] unknown scene=$scene" 1>&2
      exit 2
      ;;
  esac
}

run_one() {
  local scene="$1"
  local variant="$2"

  set_scene_params "$scene"
  CFG_FILE="$(cfg_for_variant "$variant")"
  MODEL_DIR="output2/${scene}_${variant}"
  LOG_DIR="output2/verify_logs"

  echo "============================================================"
  echo "[RUN] scene=$scene variant=$variant"
  echo "      dataset=$DATASET res=$RES train_split='${TRAIN_SPLIT_FLAG:-}'"
  echo "      config=$CFG_FILE"
  echo "      model_dir=$MODEL_DIR"
  echo "============================================================"
  echo "$(now_ts) [RUN] scene=$scene variant=$variant" >> "$MAIN_LOG_REL"

  # Train (skip if already has iteration_30000 to support resume).
  if [[ -d "${MODEL_DIR}/point_cloud/iteration_${ITERATION}" ]]; then
    echo "[SKIP] train: found ${MODEL_DIR}/point_cloud/iteration_${ITERATION}"
  else
    write_status "train" "$scene" "$variant"
    python train.py \
      -s "$DATASET" \
      -r "$RES" \
      -m "$MODEL_DIR" \
      --config_file "$CFG_FILE" \
      --num_classes 256 \
      --images images \
      --object_path object_mask \
      ${TRAIN_SPLIT_FLAG:-} \
      2>&1 | tee "${LOG_DIR}/train_${scene}_${variant}_iter${ITERATION}.log"
  fi

  # Render LERF-Mask test masks.
  if [[ -d "${MODEL_DIR}/test/ours_${ITERATION}_text/test_mask" ]]; then
    echo "[SKIP] render: found ${MODEL_DIR}/test/ours_${ITERATION}_text/test_mask"
  else
    write_status "render" "$scene" "$variant"
    python render_lerf_mask.py \
      -s "$DATASET" \
      -r "$RES" \
      -m "$MODEL_DIR" \
      --iteration "$ITERATION" \
      --skip_train \
      --num_classes 256 \
      --images images \
      --object_path object_mask \
      ${TRAIN_SPLIT_FLAG:-} \
      2>&1 | tee "${LOG_DIR}/render_${scene}_${variant}_iter${ITERATION}.log"
  fi
}

for scene in "${scenes[@]}"; do
  for variant in "${variants[@]}"; do
    run_one "$scene" "$variant"
  done
done

# Final table (evaluates whatever has been produced under output2/)
write_status "table" "" ""
python tests/run_ablation_table.py \
  --iteration "$ITERATION" \
  --output_root output2 \
  --out_dir output2/verify_logs \
  2>&1 | tee "output2/verify_logs/ablation_table_iter${ITERATION}.log"

echo "[DONE] output2 pipeline finished."
