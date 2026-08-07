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
last=$(grep -oE 'global_step=[0-9]+' "$LOG" | tail -1 | cut -d= -f2)
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
last=$(grep -oE 'global_step=[0-9]+' "$LOG" | tail -1 | cut -d= -f2)
[ -n "${last:-}" ] || fail 1 "no global_step line in $LOG -- wrong log, or the format changed"

stage=$(grep -oE 'stage=[A-Za-z0-9_]+' "$LOG" | tail -1 | cut -d= -f2)
rej=$(grep -oE 'npe_rejected=[0-9]+' "$LOG" | tail -1 | cut -d= -f2)
nll=$(grep -oE 'sim_forecast_nll=[0-9.]+' "$LOG" | tail -1 | cut -d= -f2)

echo "HEALTHY stage=$stage global_step=$last nll=${nll:-?} npe_rejected=${rej:-?} procs=$procs log_age=${age}s"
