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
log "Ablation matrix: scenes x configs (+F additions)"
log "Scenes   : $SCENES"
log "Log file : $LOG_FILE"
log "============================================================="

config_for_tag () {
    # Bash 3.2 on macOS doesn't support associative arrays.
    # Keep this mapping in a portable case statement.
    local TAG="$1"
    case "$TAG" in
        baseline)   echo "config/gaussian_dataset/train.json" ;;
        aniso)      echo "config/gaussian_dataset/train_aniso_only.json" ;;
        normal)     echo "config/gaussian_dataset/train_aniso.json" ;;
        uncertain)  echo "config/gaussian_dataset/train_aniso_uncertain.json" ;;
        uncertain2d) echo "config/gaussian_dataset/train_aniso_uncertain2d.json" ;;
        uncertain3d) echo "config/gaussian_dataset/train_aniso_uncertain3d.json" ;;
        full)       echo "config/gaussian_dataset/train_full.json" ;;
        *)
            echo ""
            return 1
            ;;
    esac
}

# The tags to iterate over (allow user override via TAGS env var).
# TAGS="${TAGS:-baseline aniso normal uncertain full}"
TAGS="${!TAGS:-baseline uncertain}"

train_one () {
    local SCENE="$1"
    local TAG="$2"
    local CONFIG
    CONFIG="$(config_for_tag "$TAG")" || true
    if [[ -z "$CONFIG" ]]; then
        log "  [ERROR] unknown TAG '${TAG}'. Known: baseline aniso normal uncertain uncertain2d uncertain3d full"
        return 2
    fi
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
    # Store per-tag output dirs in variables: MODEL_DIR_<tag>
    for TAG in $TAGS; do
        OUT_DIR=$(train_one "$SCENE" "$TAG") || true
        eval "MODEL_DIR_${TAG}=\"${OUT_DIR}\""
        render_one "${OUT_DIR}" || true
    done

    log "  [eval] ${SCENE} (comparing ${TAGS} configs)"
    # Evaluate each model individually against the baseline
    for TAG in $TAGS; do
        BASELINE_DIR=$(eval echo "\$MODEL_DIR_baseline")
        OURS_DIR=$(eval echo "\$MODEL_DIR_${TAG}")
        python tests/eval_compare.py \
            --scene "$SCENE" \
            --baseline_model "${BASELINE_DIR}" \
            --ours_model     "${OURS_DIR}" \
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
