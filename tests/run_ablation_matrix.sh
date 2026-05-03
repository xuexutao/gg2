#!/usr/bin/env bash
# ============================================================================
#  Full ablation matrix runner: 3 scenes x 3 configurations
#
#  Scenes: figurines, ramen, teatime
#  Configs:
#    1) Baseline      (train.json,           use_aniso=false)
#    2) Aniso only    (train_aniso_only.json, use_aniso=true, normal=0.0)
#    3) Full (ours)   (train_aniso.json,      use_aniso=true, normal=0.1)
#
#  Total: 9 trained models. Already-existing ones are SKIPPED.
#  Produces a final summary JSON with all results.
#
#  Usage:
#    cd /Users/bytedance/demo/shili_51/paper/gaussian-grouping
#    bash tests/run_ablation_matrix.sh
#
#    # Only a subset of scenes:
#    SCENES="figurines" bash tests/run_ablation_matrix.sh
#
#    # Fast smoke (3k iter instead of 30k):
#    FAST=1 bash tests/run_ablation_matrix.sh
# ============================================================================

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# SCENES="${SCENES:-figurines ramen teatime}"
SCENES="${!SCENES:-figurines}"
RES="${RES:-1}"

LOG_DIR="output/verify_logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/ablation_${TS}.log"
SUMMARY_JSON="${LOG_DIR}/ablation_summary_${TS}.json"

if [[ "${FAST:-0}" == "1" ]]; then
    ITER_ARGS="--iterations 3000 --test_iterations 1000 3000 --save_iterations 1000 3000"
    ITER_NUM=3000
else
    ITER_ARGS=""
    ITER_NUM=30000
fi

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }

log "============================================================="
log "Ablation matrix: ${#SCENES[@]} scenes x 5 configs (+F additions)"
log "Scenes   : $SCENES"
log "Log file : $LOG_FILE"
log "============================================================="

# Configs keyed by short tag. The order here defines the ablation columns.
#   baseline   : original Gaussian Grouping (Euclidean KNN, no extras)
#   aniso      : + Anisotropic neighbors (C1.a)
#   normal     : + Normal consistency loss (C1.b) [a.k.a. old "full"]
#   uncertain  : + Uncertainty-Aware grouping (F)     [no normal loss]
#   full       : + Aniso + Normal + Uncertainty       [final]
declare -A CFG
CFG["baseline"]="config/gaussian_dataset/train.json"
CFG["aniso"]="config/gaussian_dataset/train_aniso_only.json"
CFG["normal"]="config/gaussian_dataset/train_aniso.json"
CFG["uncertain"]="config/gaussian_dataset/train_aniso_uncertain.json"
CFG["full"]="config/gaussian_dataset/train_full.json"

# The tags to iterate over (allow user override via TAGS env var).
# TAGS="${TAGS:-baseline aniso normal uncertain full}"
TAGS="${!TAGS:-baseline uncertain}"

train_one () {
    local SCENE="$1"
    local TAG="$2"
    local CONFIG="${CFG[$TAG]}"
    local OUT_DIR="output/verify_${SCENE}_${TAG}"

    # Backward-compat: the old 'baseline' dir was named _baseline; reuse it.
    if [[ "$TAG" == "baseline" && -d "output/verify_${SCENE}_baseline" ]]; then
        OUT_DIR="output/verify_${SCENE}_baseline"
    fi
    # Backward-compat: before F existed, "full" was Aniso+Normal and was saved
    # under _aniso/. We now keep that run as the "normal" ablation row.
    if [[ "$TAG" == "normal" && -d "output/verify_${SCENE}_aniso" ]]; then
        OUT_DIR="output/verify_${SCENE}_aniso"
    fi

    local CKPT="${OUT_DIR}/point_cloud/iteration_${ITER_NUM}/classifier.pth"
    if [[ -f "$CKPT" ]]; then
        log "  [skip] ${SCENE}/${TAG}: checkpoint already exists at ${CKPT}" >&2
        echo "$OUT_DIR"
        return 0
    fi

    local DATA="data/lerf_mask/${SCENE}"
    if [[ ! -d "$DATA" ]]; then
        log "  [ERROR] dataset missing: $DATA" >&2
        return 2
    fi

    log "  [train] ${SCENE} / ${TAG} -> ${OUT_DIR}" >&2
    if [[ -d "$OUT_DIR/point_cloud/iteration_${ITER_NUM}/classifier.pth" ]]; then
        log "  [skip] ${SCENE}/${TAG}: output dir already exists" >&2
        return "$OUT_DIR"
    fi
    python train.py \
        -s "$DATA" -r "$RES" \
        -m "$OUT_DIR" \
        --config_file "$CONFIG" \
        --train_split \
        $ITER_ARGS 2>&1 | tee -a "$LOG_FILE" >/dev/null
    echo "$OUT_DIR"
}

render_one () {
    local OUT_DIR="$1"
    # Skip if already rendered
    if [[ -d "${OUT_DIR}/test/ours_${ITER_NUM}_text/test_mask" ]]; then
        return 0
    fi
    log "  [render] ${OUT_DIR}"
    python render_lerf_mask.py \
        -m "$OUT_DIR" \
        --skip_train \
        --num_classes 256 \
        --images images 2>&1 | tee -a "$LOG_FILE" >/dev/null || {
        log "  !! render failed for ${OUT_DIR}"
        return 1
    }
}

# ---- main loop ----

echo "{" > "$SUMMARY_JSON"
FIRST=1
for SCENE in $SCENES; do
    log ""
    log "=== Scene: $SCENE ==="
    declare -A MODEL_DIRS
    for TAG in $TAGS; do
        MODEL_DIRS[$TAG]=$(train_one "$SCENE" "$TAG") || true
        render_one "${MODEL_DIRS[$TAG]}" || true
    done

    log "  [eval] ${SCENE} (comparing ${TAGS} configs)"
    # Evaluate each model individually against the baseline
    for TAG in $TAGS; do
        python tests/eval_compare.py \
            --scene "$SCENE" \
            --baseline_model "${MODEL_DIRS[baseline]}" \
            --ours_model     "${MODEL_DIRS[$TAG]}" \
            --iteration      "$ITER_NUM" \
            --out_json       "${LOG_DIR}/metrics_${SCENE}_${TAG}_${TS}.json" \
            2>&1 | tee -a "$LOG_FILE" >/dev/null
    done

    # Append to summary JSON
    [[ $FIRST -eq 0 ]] && echo "," >> "$SUMMARY_JSON"
    FIRST=0
    {
        echo "  \"${SCENE}\": {"
        FIRST_TAG=1
        for TAG in $TAGS; do
            [[ $FIRST_TAG -eq 0 ]] && echo ","
            FIRST_TAG=0
            printf "    \"%s\": " "$TAG"
            cat "${LOG_DIR}/metrics_${SCENE}_${TAG}_${TS}.json" | \
                python -c "import sys,json; d=json.load(sys.stdin); print(json.dumps({'mIoU': d['ours']['mIoU'], 'mBIoU': d['ours']['mBIoU'], 'per_class_iou': d['ours']['per_class_iou'], 'per_class_biou': d['ours']['per_class_biou']}))"
        done
        echo ""
        echo "  }"
    } >> "$SUMMARY_JSON"
done
echo "}" >> "$SUMMARY_JSON"

log ""
log "============================================================="
log "ALL DONE"
log "Summary JSON : $SUMMARY_JSON"
log "Full log     : $LOG_FILE"
log "============================================================="

# Pretty-print final table
python tests/summarize_ablation.py --summary "$SUMMARY_JSON" 2>&1 | tee -a "$LOG_FILE" || true
