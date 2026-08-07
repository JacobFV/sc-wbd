#!/usr/bin/env bash
# Apply the stage-admission fix and launch run 3.
#
# Run 2 trained on simulation alone: six gates in `train.py` select behaviour by
# matching the RUN-1 stage names, run 2 renamed every stage, and five of the six
# therefore returned the wrong answer. See reports/RUN2.md §2b.
#
# The remedy was written before run 2 started and never applied. This script
# applies it, proves it worked, and only then launches — because a control arm
# trained through the same gates measures the same wrong thing, and a second
# nine-hour run spent on that is the expensive version of this mistake.
#
# It refuses rather than guessing at every step where guessing would be worse
# than stopping.
set -uo pipefail
cd "$(dirname "$0")/.."

PATCH="configs/run2/patches/0001-run_stage-config-driven-admission.patch"
CONFIG="${CONFIG:-configs/run2/pilot-families.yaml}"
LOG="${LOG:-reports/training/run003.log}"
PY="${PY:-.venv/bin/python}"

fail() { echo "REFUSED: $*"; exit 1; }
step() { printf '\n=== %s\n' "$*"; }

# 1. Never run two trainings on one GPU pool -----------------------------------
step "1/5  check nothing is training"
if pgrep -f 'scwbd\.foundation\.train' >/dev/null; then
    fail "training is already running. The GB10's ~121 GB pool is unified and
    shared; a second job does not get its own memory. Wait for it, or stop it
    deliberately."
fi

# 2. Apply the patch ------------------------------------------------------------
step "2/5  apply $PATCH"
[ -f "$PATCH" ] || fail "$PATCH not found"
if git apply --check "$PATCH" 2>/dev/null; then
    git apply "$PATCH" || fail "patch check passed but apply failed"
    echo "  applied"
elif git apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "  already applied (reverse-check succeeds) — continuing"
else
    fail "$PATCH neither applies nor is already applied. train.py has diverged;
    resolve by hand rather than forcing."
fi

# 3. Prove the fix took ---------------------------------------------------------
# These went 7-failed/4-passed before the patch and 11-passed after, measured and
# recorded in the test file's own header. Anything else means the patch landed
# but did not do what it claims.
step "3/5  admission tests must now pass"
CUDA_VISIBLE_DEVICES="" $PY -m pytest tests/foundation/test_curriculum_admission.py \
    -q -p no:warnings || fail "the admission tests still fail after the patch.
    Do NOT launch: the gates are not fixed and run 3 would repeat run 2."

# 4. The xfail markers must now be wrong ---------------------------------------
# tests/foundation/test_stage_names_reach_the_trainer.py is xfail(strict=True).
# With the gates fixed those tests XPASS, which pytest reports as a FAILURE.
# That is the designed behaviour: it forces the marker to be removed by hand so
# a fix cannot be silently absorbed. So here, failure is success.
step "4/5  stage-name tests must now XPASS (reported as failure — that is correct)"
if CUDA_VISIBLE_DEVICES="" $PY -m pytest tests/foundation/test_stage_names_reach_the_trainer.py \
    -q -p no:warnings >/dev/null 2>&1; then
    echo "  WARNING: these still xfail. The gates may be only partly fixed."
    echo "  Read the failures before launching; do not delete the marker to silence them."
else
    echo "  they XPASS, as intended."
    echo "  ACTION REQUIRED: remove the pytestmark xfail block from"
    echo "    tests/foundation/test_stage_names_reach_the_trainer.py"
    echo "  and commit it, so the suite is green for the right reason."
fi

# 5. Launch ---------------------------------------------------------------------
step "5/5  launch"
echo "  config : $CONFIG"
echo "  log    : $LOG"
echo "  NOTE: cuda_reserve_gb is the only real cap on this box. systemd MemoryMax"
echo "        does NOT bound CUDA — the ~121 GB pool is unified."
read -r -p "  launch now? [y/N] " ans
[ "${ans:-N}" = "y" ] || fail "not launched (nothing was reverted; the patch is applied)"

systemd-run --user --scope -q -p MemoryMax=96G -p MemorySwapMax=8G -- \
    env PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -m scwbd.foundation.train --config "$CONFIG" >> "$LOG" 2>&1 &
echo "  launched; watch with: LOG=$LOG make health"
