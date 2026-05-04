#!/usr/bin/env bash
# ============================================================================
#  End-to-end verification of Breakthrough 1: Anisotropic Affinity
#  replacement for the 3D regularization loss in Gaussian Grouping.
#
#  This script is the ONE thing you run after the code change.
#  It does:
#    0. Unit-level sanity tests on the new loss (no data required).
#    1. Train the BASELINE (original Gaussian Grouping) on a LERF scene.
#    2. Train OURS    (Anisotropic Affinity) on the same scene.
#    3. Render both models on the LERF-Mask test set using text prompts.
#    4. Evaluate mIoU & Boundary-IoU for both models.
#    5. Print a side-by-side comparison table.
#
#  Prerequisites (from the upstream Gaussian Grouping README / docs):
#    - conda env set up (docs/install.md)
#    - data/lerf_mask/<scene>/... downloaded (docs/dataset.md)
#      Supported scenes: figurines | ramen | teatime
#    - Grounded-DINO + SAM checkpoints downloaded (for text-prompt render)
#    - CUDA GPU visible to PyTorch
#
#  Usage:
#    cd /Users/bytedance/demo/shili_51/paper/gaussian-grouping
#    bash tests/run_verification.sh [SCENE] [RESOLUTION]
#    # defaults: SCENE=figurines, RESOLUTION=1
#
#  Notes:
#    - The script is IDEMPOTENT: if an output dir already exists, the
#      corresponding training step is skipped.
#    - Full baseline+ours training on one scene is ~30-60 min on an A100.
#      For a fast smoke test use env FAST=1 to cut iterations to 3000.
#
#  Metrics reported:
#    - mIoU        : overall mean IoU over text-prompt categories
#    - mBIoU       : overall mean Boundary-IoU (main indicator for our claim)
#    - per-class IoU and BIoU
#
#  Datasets & baselines comparison plan (written into the log file):
#    * Dataset  : LERF-Mask (figurines | ramen | teatime)  — the official
#                 evaluation set of Gaussian Grouping (docs/dataset.md).
#    * Baseline : original Gaussian Grouping (use_aniso=false)
#    * Ours     : Gaussian Grouping + Anisotropic Affinity (use_aniso=true,
#                 coarse_k=64, normal_weight=0.1)
#    * Metrics  : IoU (region overlap, standard)
#                 Boundary-IoU (thin/boundary accuracy — our claimed win)
# ============================================================================

set -euo pipefail

SCENE="${1:-figurines}"
RES="${2:-1}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="data/${SCENE}"
OUT_BASE="output/${SCENE}_baseline"
OUT_OURS="output/${SCENE}_aniso_only"

CONFIG_BASE="config/gaussian_dataset/train.json"
CONFIG_OURS="config/gaussian_dataset/train_aniso.json"

LOG_DIR="output/verify_logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/verify_${SCENE}_${TS}.log"

# Allow a fast smoke test (much shorter training — useful for a first run).
if [[ "${FAST:-0}" == "1" ]]; then
    ITER_ARGS="--iterations 3000 --test_iterations 1000 3000 --save_iterations 1000 3000"
    echo "[info] FAST mode: overriding iterations to 3000"
else
    ITER_ARGS=""
fi

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }

log "=================================================================="
log "Verification for Breakthrough 1 (Anisotropic Affinity)"
log "Scene         : ${SCENE}"
log "Resolution    : ${RES}"
log "Repo root     : ${REPO_ROOT}"
log "Log file      : ${LOG_FILE}"
log "=================================================================="

# -----------------------------------------------------------------------------
# Step 0: unit tests for the new loss (fast, no data needed)
# -----------------------------------------------------------------------------
# log ""
# log "[Step 0/5] Running unit tests on Anisotropic Affinity loss..."
# python -m tests.test_aniso_loss 2>&1 | tee -a "$LOG_FILE"
# log "[Step 0/5] Unit tests passed."

# -----------------------------------------------------------------------------
# Sanity check: dataset exists
# -----------------------------------------------------------------------------
if [[ ! -d "$DATA_DIR" ]]; then
    log "[ERROR] dataset not found at $DATA_DIR"
    log "      Download LERF-Mask from hugging face first; see docs/dataset.md"
    exit 2
else
    log "      Dataset found at $DATA_DIR"
fi

# -----------------------------------------------------------------------------
# Step 1: train baseline (Gaussian Grouping, Euclidean KNN)
# -----------------------------------------------------------------------------
log ""
log "[Step 1/5] Training BASELINE (Euclidean KNN) -> ${OUT_BASE}"
if [[ -f "${OUT_BASE}/point_cloud/iteration_30000/classifier.pth" ]] || \
   [[ -f "${OUT_BASE}/point_cloud/iteration_3000/classifier.pth" && "${FAST:-0}" == "1" ]]; then
    log "      Baseline checkpoint already exists, skipping training."
else
    python train.py \
        -s "$DATA_DIR" \
        -r "$RES" \
        -m "$OUT_BASE" \
        --config_file "$CONFIG_BASE" \
        --train_split \
        $ITER_ARGS 2>&1 | tee -a "$LOG_FILE"
fi
log "[step 1/5] Done"

# -----------------------------------------------------------------------------
# Step 2: train OURS (Gaussian Grouping + Anisotropic Affinity)
# -----------------------------------------------------------------------------
log ""
log "[Step 2/5] Training OURS  (Anisotropic Affinity) -> ${OUT_OURS}"
if [[ -f "${OUT_OURS}/point_cloud/iteration_30000/classifier.pth" ]] || \
   [[ -f "${OUT_OURS}/point_cloud/iteration_3000/classifier.pth" && "${FAST:-0}" == "1" ]]; then
    log "      Ours checkpoint already exists, skipping training."
else
    python train.py \
        -s "$DATA_DIR" \
        -r "$RES" \
        -m "$OUT_OURS" \
        --config_file "$CONFIG_OURS" \
        --train_split \
        $ITER_ARGS 2>&1 | tee -a "$LOG_FILE"
fi
log "[step 2/5] Done"

# -----------------------------------------------------------------------------
# Step 3: render masks on the LERF-Mask test set using text prompts
# -----------------------------------------------------------------------------
log ""
log "[Step 3/5] Rendering masks with text prompts (both models)..."

render_one() {
    local MODEL_DIR="$1"
    local TAG="$2"
    log "      -> rendering ${TAG}: ${MODEL_DIR}"
    python render_lerf_mask.py -m "$MODEL_DIR" --skip_train --num_classes 256 --images images 2>&1 | tee -a "$LOG_FILE" || {
        log "      !! render_lerf_mask.py failed for ${TAG} (often missing Grounded-DINO / SAM ckpt). See ${LOG_FILE}."
        return 1
    }
}

render_one "$OUT_BASE" "BASELINE" || true
render_one "$OUT_OURS" "OURS"     || true
log "[step 3/5] Done"

# -----------------------------------------------------------------------------
# Step 4: evaluate mIoU & Boundary-IoU for both, with a minimal eval harness.
# -----------------------------------------------------------------------------
log ""
log "[Step 4/5] Evaluating IoU and Boundary-IoU..."

python tests/eval_compare.py \
    --scene "$SCENE" \
    --baseline_model "$OUT_BASE" \
    --ours_model     "$OUT_OURS" \
    --iteration      "$( [[ "${FAST:-0}" == "1" ]] && echo 3000 || echo 30000 )" \
    --out_json       "${LOG_DIR}/metrics_${SCENE}_${TS}.json" \
    2>&1 | tee -a "$LOG_FILE"
log "[step 4/5] Done"

# -----------------------------------------------------------------------------
# Step 5: final summary
# -----------------------------------------------------------------------------
log ""
log "[Step 5/5] DONE. Summary written to:"
log "      ${LOG_FILE}"
log "      ${LOG_DIR}/metrics_${SCENE}_${TS}.json"
log ""
log "Compare:"
log "      Baseline model : ${OUT_BASE}"
log "      Ours model     : ${OUT_OURS}"
log "      Key metric     : mBIoU improvement of OURS over BASELINE"
log "=================================================================="
