#!/usr/bin/env bash
# One health check for the run-2 training job, for humans and the cron watchdog.
#
# The failure this exists to prevent: `tail` on a log path that does not exist
# prints nothing and exits 0.  Chain that with a live process count and you get
# a health report that reads HEALTHY from silence -- indefinitely, unattended.
# That is the sixth instance of this class in reports/decorative_guards.md, so
# every check below must FAIL LOUD rather than return empty.
#
# Exit codes are the contract:
#   0  healthy and progressing
#   1  the log is missing, unreadable, or stale (the instrument is broken)
#   2  training is dead (no process) or has a traceback
set -uo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/run2/pilot-families.yaml}"
LOG="${LOG:-reports/training/run002.log}"
STALE_S="${STALE_S:-900}"   # 15 min without a new line is stale; a step is ~3 s

fail() { echo "UNHEALTHY($1): ${*:2}"; exit "$1"; }

# 1. The instrument itself.  Never let an absent log look like a quiet one.
[ -f "$LOG" ] || fail 1 "log not found: $LOG (checked from $PWD)"
[ -r "$LOG" ] || fail 1 "log not readable: $LOG"
[ -s "$LOG" ] || fail 1 "log is empty: $LOG"

age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
[ "$age" -lt "$STALE_S" ] || fail 1 "log stale: no write for ${age}s (limit ${STALE_S}s)"

# 2. The job.
procs=$(pgrep -cf 'scwbd\.foundation\.train' || true)
[ "${procs:-0}" -gt 0 ] || fail 2 "no training process matches scwbd.foundation.train"

tb=$(grep -c 'Traceback' "$LOG" || true)
[ "${tb:-0}" -eq 0 ] || fail 2 "$tb traceback(s) in $LOG"

# 3. Progress.  A step number that does not move is a hang, not a slow step.
last=$(grep -oE 'global_step=[0-9]+' "$LOG" | tail -1 | cut -d= -f2)
[ -n "${last:-}" ] || fail 1 "no global_step line in $LOG -- wrong log, or the format changed"

stage=$(grep -oE 'stage=[A-Za-z0-9_]+' "$LOG" | tail -1 | cut -d= -f2)
rej=$(grep -oE 'npe_rejected=[0-9]+' "$LOG" | tail -1 | cut -d= -f2)
nll=$(grep -oE 'sim_forecast_nll=[0-9.]+' "$LOG" | tail -1 | cut -d= -f2)

echo "HEALTHY stage=$stage global_step=$last nll=${nll:-?} npe_rejected=${rej:-?} procs=$procs log_age=${age}s"
