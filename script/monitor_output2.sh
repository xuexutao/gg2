#!/usr/bin/env bash
set -euo pipefail

# Watchdog: checks output2/verify_logs every 30 minutes.
# If the pipeline is not running (or appears stalled) and not marked done, it restarts it.
#
# Default: run locally (on a GPU worker) and detect processes via pgrep.
# If you want to run this from master/devbox and control a worker, set:
#   WORKER_ID=<mlx_worker_id>

REPO_ROOT="/mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/51/new_workspace/gg2"
ITERATION="${ITERATION:-30000}"
WORKER_ID="${WORKER_ID:-}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-1800}"
LOG_DIR="${REPO_ROOT}/output2/verify_logs"

MAIN_LOG="${LOG_DIR}/watchdog_iter${ITERATION}.log"
DONE_MARK="${LOG_DIR}/PIPELINE_DONE_iter${ITERATION}"
STATUS_JSON="${LOG_DIR}/pipeline_status_iter${ITERATION}.json"
TABLE_LOG="${LOG_DIR}/ablation_table_iter${ITERATION}.log"
WATCHDOG_STATUS_JSON="${LOG_DIR}/watchdog_status_iter${ITERATION}.json"

mkdir -p "$LOG_DIR"
touch "$MAIN_LOG"

now_ts() { date -Iseconds; }

is_done() {
  if [[ -f "$DONE_MARK" ]]; then
    return 0
  fi
  # Backward-compatible: older pipeline runs may not create the done-mark.
  # If we already have a finished ablation table log, consider it done.
  if [[ -f "$TABLE_LOG" ]]; then
    if grep -q "^# Ablation table" "$TABLE_LOG"; then
      touch "$DONE_MARK" || true
      return 0
    fi
  fi
  return 1
}

pipeline_running() {
  # Any of these processes indicates work is ongoing.
  if [[ -n "$WORKER_ID" ]]; then
    mlx worker login "$WORKER_ID" -- pgrep -f "script/run_output2_all.sh" >/dev/null 2>&1 && return 0 || true
    mlx worker login "$WORKER_ID" -- pgrep -f "python.*train\.py" >/dev/null 2>&1 && return 0 || true
    mlx worker login "$WORKER_ID" -- pgrep -f "python.*render_lerf_mask\.py" >/dev/null 2>&1 && return 0 || true
    return 1
  fi
  pgrep -f "script/run_output2_all.sh" >/dev/null 2>&1 && return 0
  pgrep -f "python.*train\.py" >/dev/null 2>&1 && return 0
  pgrep -f "python.*render_lerf_mask\.py" >/dev/null 2>&1 && return 0
  return 1
}

latest_log_mtime() {
  # Print epoch seconds of the newest file in verify_logs, or 0 if none.
  python - "$LOG_DIR" <<'PY'
import os
from pathlib import Path
import sys

log_dir = Path(sys.argv[1])
if not log_dir.is_dir():
    print(0)
    raise SystemExit

latest = 0.0
for p in log_dir.glob('**/*'):
    try:
        if p.is_file():
            latest = max(latest, p.stat().st_mtime)
    except OSError:
        pass
print(int(latest))
PY
}

# Ensure only one watchdog instance per iteration.
LOCK_FILE="${LOG_DIR}/watchdog_iter${ITERATION}.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "$(now_ts) [WATCHDOG] another instance is running (lock: $LOCK_FILE); exiting" >> "$MAIN_LOG"
  exit 0
fi

restart_pipeline() {
  echo "$(now_ts) [WATCHDOG] restarting pipeline (iter=${ITERATION})" >> "$MAIN_LOG"
  if [[ -n "$WORKER_ID" ]]; then
    # Start on worker in background so the watchdog can continue.
    mlx worker login "$WORKER_ID" -- bash -c "nohup bash '${REPO_ROOT}/script/run_output2_all.sh' >> '${LOG_DIR}/pipeline_nohup_iter${ITERATION}.log' 2>&1 &" >/dev/null 2>&1 || true
    echo "$(now_ts) [WATCHDOG] restart issued on worker_id=${WORKER_ID}" >> "$MAIN_LOG"
  else
    # Local (worker) mode.
    nohup bash "${REPO_ROOT}/script/run_output2_all.sh" >> "${LOG_DIR}/pipeline_nohup_iter${ITERATION}.log" 2>&1 &
    echo "$(now_ts) [WATCHDOG] restart issued, pid=$!" >> "$MAIN_LOG"
  fi
}

echo "$(now_ts) [WATCHDOG] started (iter=${ITERATION})" >> "$MAIN_LOG"
echo "$(now_ts) [WATCHDOG] mode: WORKER_ID='${WORKER_ID}' CHECK_INTERVAL_SEC=${CHECK_INTERVAL_SEC}" >> "$MAIN_LOG"

prev_mtime=0
while true; do
  if is_done; then
    echo "$(now_ts) [WATCHDOG] done-mark found, exiting" >> "$MAIN_LOG"
    exit 0
  fi

  cur_mtime=$(latest_log_mtime)
  if [[ "$cur_mtime" -gt 0 ]]; then
    echo "$(now_ts) [WATCHDOG] latest_log_mtime=${cur_mtime} status_json=${STATUS_JSON}" >> "$MAIN_LOG"
  else
    echo "$(now_ts) [WATCHDOG] no logs yet under ${LOG_DIR}" >> "$MAIN_LOG"
  fi

  now_epoch=$(date +%s)
  stalled=0
  if [[ "$cur_mtime" -gt 0 ]]; then
    # If no log file has been updated for > 1 hour, consider it stalled.
    if (( now_epoch - cur_mtime > 3600 )); then
      stalled=1
    fi
  fi

  running=0
  if pipeline_running; then
    running=1
    if (( stalled == 1 )); then
      echo "$(now_ts) [WATCHDOG] pipeline running but appears stalled (>3600s no log update)" >> "$MAIN_LOG"
      restart_pipeline
    else
      echo "$(now_ts) [WATCHDOG] pipeline running" >> "$MAIN_LOG"
    fi
  else
    echo "$(now_ts) [WATCHDOG] pipeline NOT running" >> "$MAIN_LOG"
    restart_pipeline
  fi

  # Heartbeat status for external checking.
  printf '{"time":"%s","iteration":%s,"worker_id":"%s","running":%s,"stalled":%s,"latest_log_mtime":%s}\n' \
    "$(now_ts)" "$ITERATION" "${WORKER_ID}" "$running" "$stalled" "$cur_mtime" > "$WATCHDOG_STATUS_JSON" || true

  prev_mtime="$cur_mtime"
  sleep "$CHECK_INTERVAL_SEC"
done
