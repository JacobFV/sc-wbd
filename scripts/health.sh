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
CKPT="${CKPT:-checkpoints/scwbd-002-pilot}"
STALE_S="${STALE_S:-900}"   # 15 min without a new line is stale; a step is ~3 s

fail() { echo "UNHEALTHY($1): ${*:2}"; exit "$1"; }

# 1. The instrument itself.  Never let an absent log look like a quiet one.
[ -f "$LOG" ] || fail 1 "log not found: $LOG (checked from $PWD)"
[ -r "$LOG" ] || fail 1 "log not readable: $LOG"
[ -s "$LOG" ] || fail 1 "log is empty: $LOG"

age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
# Staleness is checked LATER, only for a job that is supposed to be running.
# A finished run's log is stale by definition and stays stale forever, so
# testing it first reported a completed run as a broken instrument.

# 2. Did it FINISH?  Checked before liveness, because "no process" is the same
# observation for a completed run and a dead one -- and the caller's response to
# those is opposite: one advances the pipeline, the other relaunches training.
# This check did not exist, and the first thing it would have done on a
# successful run was tell a 10-minute watchdog to relaunch a finished job on top
# of its own evaluation.
TARGET="${TARGET:-8700}"
# Two log formats are live. Run 2's stdout capture writes `global_step=123`;
# the JsonlLogger writes `"global_step": 123`. Reading only the first reported a
# healthy run-3 job as "wrong log, or the format changed" -- an instrument
# failing on its own input again, which is the class ISSUE-004 is about.
last=$(grep -oE '"?global_step"?[=:][[:space:]]*[0-9]+' "$LOG" | tail -1 | grep -oE '[0-9]+$')

# The CHECKPOINT is the authority on how far training got; the log is a
# transcript of it and can be shorter than the run for reasons that have nothing
# to do with training.  It happened: `git checkout -- reports/training/` reverted
# run002.log to HEAD, which ends at 4686, and this script then reported a
# COMPLETE 8700-step run as "a death, not a completion" -- an instrument
# reporting on its own input rather than on its subject.  The weights say 8700.
CKPT_STEP=""
if [ -f "$CKPT/last.pt" ]; then
    CKPT_STEP=$(.venv/bin/python - "$CKPT/last.pt" 2>/dev/null <<'PYEOF'
import sys
try:
    import torch
    print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("step") or 0))
except Exception:
    pass
PYEOF
)
fi
# Take the FURTHEST evidence of progress, not the most recent instrument. Both
# are lower bounds on what ran; neither can overstate it.
if [ -n "${CKPT_STEP:-}" ] && [ "${CKPT_STEP:-0}" -gt "${last:-0}" ]; then
    [ -n "${last:-}" ] && echo "note: log ends at global_step=$last but the checkpoint records $CKPT_STEP; trusting the checkpoint"
    last="$CKPT_STEP"
fi
# Count only the real python invocation.  `pgrep -f` matches ANY command line
# containing the pattern -- including the shell running this very check, because
# make echoes the recipe and the recipe names the module.  That self-match made
# a finished run report procs=1 and skip the completion branch below.  Sixth
# pgrep misfire on this project; match the interpreter, and drop shell wrappers.
procs=$(pgrep -af 'scwbd\.foundation\.train' 2>/dev/null \
        | grep -E '(^|/)[0-9]+ .*(python[0-9.]*) ' \
        | grep -vE 'bash -c|sh -c|make ' | wc -l)
if [ "${procs:-0}" -eq 0 ] && [ -n "${last:-}" ] && [ "$last" -ge "$TARGET" ]; then
    echo "COMPLETE global_step=$last (target $TARGET), no process — training is DONE, do not relaunch"
    exit 0
fi

# 3. The job is supposed to be running, so absence now means death.
[ "${procs:-0}" -gt 0 ] || fail 2 "no training process, and global_step=${last:-none} < target $TARGET — this is a death, not a completion"

# 3b. Only now does staleness mean something: a LIVE process whose log has gone
# quiet is a hang. The same silence from a finished job is just a finished job.
[ "$age" -lt "$STALE_S" ] || fail 1 "log stale: process alive but no write for ${age}s (limit ${STALE_S}s) — this is a hang"

tb=$(grep -c 'Traceback' "$LOG" || true)
[ "${tb:-0}" -eq 0 ] || fail 2 "$tb traceback(s) in $LOG"

# 4. Progress.  A step number that does not move is a hang, not a slow step.
# Two log formats are live. Run 2's stdout capture writes `global_step=123`;
# the JsonlLogger writes `"global_step": 123`. Reading only the first reported a
# healthy run-3 job as "wrong log, or the format changed" -- an instrument
# failing on its own input again, which is the class ISSUE-004 is about.
last=$(grep -oE '"?global_step"?[=:][[:space:]]*[0-9]+' "$LOG" | tail -1 | grep -oE '[0-9]+$')
[ -n "${last:-}" ] || fail 1 "no global_step line in $LOG -- wrong log, or the format changed"

# Same two formats as `global_step` above: `k=v` from the stdout capture,
# `"k": v` from the JsonlLogger. `_field <name>` returns whichever is present,
# so the report says what it found instead of printing `?` beside a healthy run.
_field() {
    grep -oE "\"?$1\"?[=:][[:space:]]*\"?[A-Za-z0-9_.+-]+" "$LOG" \
        | tail -1 | sed -E 's/.*[=:][[:space:]]*"?//'
}
stage=$(_field stage)
rej=$(_field npe_rejected)
# Run 3's founding stage admits no simulated source, so `sim_forecast_nll` is
# absent by design there. Fall back to the measured EEG likelihood rather than
# reporting `?` for a run that is reporting a number.
nll=$(_field sim_forecast_nll)
[ -n "${nll:-}" ] || nll=$(_field eegmmidb_real_eeg_nll)

echo "HEALTHY stage=$stage global_step=$last nll=${nll:-?} npe_rejected=${rej:-?} procs=$procs log_age=${age}s"
