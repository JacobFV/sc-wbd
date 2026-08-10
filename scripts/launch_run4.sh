#!/usr/bin/env bash
# Launch SC-WBD-004, after the gates that can still say no.
#
# 14,600 steps at a measured ~9.30 s/step is ~38 h, and `max_wall_seconds` is
# 165,600 (46 h). A defect found at hour 14 costs 14 hours, so everything
# checkable before step 1 is checked before step 1.
#
# The gates run here are the four that were GREEN AGAINST RUN 3 while run 4 was
# what was about to be launched. They are parameterised over the run under test
# now (tests/foundation/_runs.py); this script is what makes them a launch
# precondition rather than a suite that happens to pass.
#
# It refuses rather than guessing at every step where guessing would be worse
# than stopping. Non-interactive on purpose: this runs unattended. Pass
# LAUNCH=no to do every check and stop short of starting the job.
set -uo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/run4/scwbd-004.yaml}"
LOG="${LOG:-reports/training/run004.log}"
PY="${PY:-.venv/bin/python}"
LAUNCH="${LAUNCH:-yes}"

fail() { echo "REFUSED: $*" >&2; exit 1; }
step() { printf '\n=== %s\n' "$*"; }

# 1. Never run two trainings on one unified pool --------------------------------
# Explicit PIDs, not `pkill -f`/`pgrep -f` on a bare pattern: a -f match includes
# the shell running it, which killed this project's turn five separate times.
# The [s] bracket keeps grep from matching its own argv.
step "1/6  nothing else is training"
RUNNING=$(ps -eo pid,args | grep "[s]cwbd\.foundation\.train" | awk '{print $1}')
if [ -n "$RUNNING" ]; then
    fail "training is already running (PIDs: $RUNNING). The GB10's ~121 GB pool is
    UNIFIED -- a second job does not get its own memory, and misreporting that
    OOM'd this box once already. Wait for it, or stop it deliberately."
fi

# 2. The config must declare what ISSUE-012 requires ----------------------------
# Checked here and not only in a test because it is the one class of defect that
# a green suite cannot catch: `configs/run4` inherits `base: ../run3`, and run 3
# pins `nuisance_dim: 2`. The default moved to 0; the override did not. That
# would reach T4 and die on ConstantTargetDimension ~5,400 steps and ~14 h in.
step "2/6  the run's own posterior block, not the default"
$PY - "$CONFIG" <<'PYEOF' || fail "the config does not declare ISSUE-012's remedies"
import sys
from scwbd.foundation.config import load_config
from scwbd.foundation.posterior import COND_NORMS

p = load_config(sys.argv[1]).posterior
bad = []
if p.nuisance_dim != 0:
    bad.append(
        f"nuisance_dim={p.nuisance_dim}: the trainer passes torch.zeros for these, so "
        "the flow is asked to put finite density on a point mass. On run 3 two such "
        "columns took 12.99 of the 15.9 nats npe_loss fell by."
    )
if p.cond_norm not in COND_NORMS:
    bad.append(f"cond_norm={p.cond_norm!r} is not one of {COND_NORMS}")
if not (0 < p.lr_scale <= 20):
    bad.append(f"lr_scale={p.lr_scale} is outside (0, 20]")
for b in bad:
    print("  REFUSED:", b)
print(f"  nuisance_dim={p.nuisance_dim}  cond_norm={p.cond_norm}  lr_scale={p.lr_scale}")
sys.exit(1 if bad else 0)
PYEOF

# 3. The four launch gates ------------------------------------------------------
step "3/6  launch gates, against THIS run"
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 $PY -m pytest \
    tests/foundation/test_stage_permissions_reach_the_model.py \
    tests/foundation/test_regional_tensors_moved.py \
    tests/foundation/test_balloon_parameters_receive_gradient.py \
    tests/foundation/test_card_patterns_reach_the_model.py \
    -q -p no:randomly -p no:warnings
GATE_RC=$?
[ $GATE_RC -eq 0 ] || fail "launch gates failed (rc=$GATE_RC). Do NOT launch."

# Collected-something check. An empty parametrisation passes vacuously, and that
# is how 88.8% of run 2's parameters went untrained while the loss fell.
step "3b/6  the gates actually collected run 4"
N4=$(CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD" $PY -m pytest \
    tests/foundation/test_stage_permissions_reach_the_model.py \
    tests/foundation/test_regional_tensors_moved.py \
    tests/foundation/test_balloon_parameters_receive_gradient.py \
    tests/foundation/test_card_patterns_reach_the_model.py \
    --collect-only -q -p no:randomly --co -v 2>/dev/null | grep -c '\[run4\]')
echo "  run4 cases collected: $N4"
[ "$N4" -ge 4 ] || fail "only $N4 run-4 cases collected; the gates are not aimed at run 4."

# 4. The build itself. This is what found run 4's BindingDriftError -------------
step "4/6  the trainer constructs for this config"
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD" $PY - "$CONFIG" <<'PYEOF' || fail "the trainer does not construct"
import sys
from scwbd.foundation.anatomy import load_anatomy
from scwbd.foundation.config import load_config
from scwbd.foundation.model import SCWBD

cfg = load_config(sys.argv[1])
m = SCWBD(cfg.model, load_anatomy(device="cpu"))
n = sum(p.numel() for p in m.parameters())
print(f"  built: {n:,} parameters, {len(list(m.named_parameters()))} tensors")
PYEOF

# 5. A smoke that exercises every loss path -------------------------------------
# HANDOFF-004 step e. `--quick` is the CI-sized run; it is the only check here
# that touches the GPU, and it is the only one that can catch a loss term that
# raises on its first real batch.
step "5/6  smoke: every loss path, one short pass"
if [ "${SKIP_SMOKE:-no}" = "yes" ]; then
    echo "  SKIPPED by SKIP_SMOKE=yes -- say so in the run's report."
else
    SMOKE_LOG="reports/training/run004_smoke.log"
    PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        $PY -m scwbd.foundation.train --config "$CONFIG" --quick \
        --out checkpoints/scwbd-004-smoke > "$SMOKE_LOG" 2>&1
    SMOKE_RC=$?
    if [ $SMOKE_RC -ne 0 ]; then
        echo "--- last 40 lines of $SMOKE_LOG ---" >&2
        tail -40 "$SMOKE_LOG" >&2
        fail "the smoke run exited $SMOKE_RC. Read the traceback above; do NOT launch."
    fi
    echo "  smoke passed (log: $SMOKE_LOG)"
fi

# 6. Launch ---------------------------------------------------------------------
step "6/6  launch"
echo "  config    : $CONFIG"
echo "  log       : $LOG"
echo "  steps     : 14,600 over 6 stages   (~38 h at the measured 9.30 s/step)"
echo "  max_wall  : 165,600 s = 46 h"
echo "  NOTE: cuda_reserve_gb (56.0) is the only real cap on this box."
echo "        systemd MemoryMax does NOT bound CUDA -- the ~121 GB pool is unified."

if [ "$LAUNCH" != "yes" ]; then
    echo "  LAUNCH=$LAUNCH -- every check passed, nothing started."
    exit 0
fi

mkdir -p "$(dirname "$LOG")"
systemd-run --user --scope -q -p MemoryMax=96G -p MemorySwapMax=8G -- \
    env PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -m scwbd.foundation.train --config "$CONFIG" >> "$LOG" 2>&1 &
LAUNCHED=$!
echo "  launched (pid $LAUNCHED); watch with: make health-run4"
