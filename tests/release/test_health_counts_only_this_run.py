"""`make health-run<N>` must count only the process belonging to THAT run.

`scripts/health.sh` counted any command line containing `scwbd.foundation.train`.
The comment above that line already recorded SIX `pgrep` misfires on this
project; on 2026-08-10 it produced a seventh in a new shape.

Run 4 had been stopped deliberately at step 400 (a diverging `real_bold_nll`) and
a diagnostic probe was training beside it. `make health-run4` paired run 4's
correctly-stale log with the PROBE's live process and reported

    UNHEALTHY(1): log stale: process alive but no write for 2910s — this is a hang

which is false twice over: run 4 was not hung, it was stopped, and the process it
counted was not run 4. The reading misled two consecutive checks, and a watchdog
acting on it would have relaunched a deliberately-stopped run on top of a probe.

The fix excludes only what is demonstrably a DIFFERENT run: a command line that
names some other `--config`. The first attempt scoped the count to `$CONFIG`
instead, and that was too narrow -- a run started on the DEFAULT config names no
config at all, so it would have read as a death while training, and a watchdog
acts on a death by relaunching on top of the live job. Two existing tests in
`test_health_reports_the_current_row.py` failed on exactly that, correctly.

A false hang wastes a reader's attention; a false death destroys a run. The
asymmetry is why the exclusion has to prove a process belongs elsewhere rather
than prove it belongs here. Foreign processes are printed, not silently dropped:
"nothing is running" and "someone else is running" are different facts.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HEALTH = REPO / "scripts/health.sh"

pytestmark = pytest.mark.skipif(not HEALTH.is_file(), reason="health.sh absent")


def test_the_process_count_is_scoped_to_this_runs_config() -> None:
    src = HEALTH.read_text()
    assert 'grep -v -F -- "$CONFIG"' in src, (
        "health.sh no longer excludes other runs' processes. Counting every "
        "`scwbd.foundation.train` means another run -- an ablation arm, a smoke, a "
        "diagnostic probe -- is attributed to this one, and a stopped run reads as "
        "a hang."
    )
    assert "--config[= ]" in src, (
        "the exclusion no longer requires the foreign process to NAME a config. "
        "Excluding a process that names none would call a run started on the "
        "default config a DEATH while it is running -- a false death, which a "
        "watchdog acts on by relaunching on top of a live job. Only a "
        "demonstrably different --config may be excluded."
    )


def test_a_foreign_training_process_is_named_not_swallowed() -> None:
    """Reporting `procs=0` silently would be its own defect.

    "No process for this run" and "no training at all" are different facts and
    the operator needs both. The script prints the foreign command lines so the
    reader can see what it declined to count.
    """
    src = HEALTH.read_text()
    assert "belong to ANOTHER run and are not counted here" in src, (
        "health.sh no longer reports foreign training processes. Excluding them "
        "silently turns 'someone else is training' into 'nothing is running', "
        "which is how a watchdog relaunches on top of a live job."
    )
    assert re.search(r"printf '%s\\n' \"\$foreign\" \| sed", src), (
        "the foreign process command lines are no longer printed; the note would "
        "state a count without saying what it excluded."
    )


def test_it_still_distinguishes_death_from_completion() -> None:
    """The scoping must not weaken the exit codes this script exists for.

    0 healthy, 1 the instrument is broken (you know nothing), 2 the job died.
    Narrowing the count could easily have turned every death into a completion.
    """
    src = HEALTH.read_text()
    assert 'fail 2 "no training process' in src, "the death branch (exit 2) is gone"
    assert "COMPLETE global_step=" in src, "the completion branch is gone"
    assert 'this is a death, not a completion' in src


def test_health_runs_and_returns_a_documented_exit_code() -> None:
    """Executed, not just read. An instrument that cannot run reports nothing.

    Any of 0/1/2 is acceptable here -- which one depends on whether a run happens
    to be training when the suite executes. What is NOT acceptable is a shell
    error, which would mean the watchdog's first act is a broken command.
    """
    r = subprocess.run(
        ["bash", str(HEALTH)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
        env={
            "PATH": "/usr/bin:/bin",
            "CONFIG": "configs/run4/scwbd-004.yaml",
            "LOG": "reports/training/scwbd-004_train.jsonl",
            "CKPT": "checkpoints/scwbd-004",
            "TARGET": "14600",
        },
    )
    assert r.returncode in (0, 1, 2), (
        f"health.sh exited {r.returncode}, which is not one of its documented codes "
        f"(0 healthy / 1 instrument broken / 2 died). stderr: {r.stderr[-400:]}"
    )
