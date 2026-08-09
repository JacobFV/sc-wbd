"""`scripts/health.sh` must report the CURRENT step, not the last match anywhere.

The defect this pins: `_field` grepped the whole log and took the last match,
so a key written by an earlier stage kept being displayed after that stage
ended. `sim_forecast_nll` is written only by stages admitting the simulator, so
once T4 had written one, run 3's health line showed T4's final value for the
whole of T5 -- identical to sixteen decimal places on every check, which is
indistinguishable from a hung loss.

The fallback that was supposed to prevent exactly this (`|| nll=$(_field
eegmmidb_real_eeg_nll)`) could never fire, because the stale match was never
empty. A guard whose trigger condition cannot occur is the shape
`reports/decorative_guards.md` catalogues.

Both properties matter and they fail in opposite directions, so both are
asserted: the reading must be live, and an absent field must say `n/a` rather
than borrow a number from a stage that has ended.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HEALTH = REPO / "scripts/health.sh"

#: A value no live row carries, so seeing it in the output is proof of staleness
#: rather than a coincidence of formatting.
STALE_SENTINEL = "0.5875664949417114"
LIVE_SENTINEL = "1.2345678901234567"


def _write_log(path: Path) -> None:
    """An old simulator stage, then a measured stage that reports no sim loss."""
    rows = [
        {
            "global_step": 100,
            "stage": "T4_simulator",
            "sim_forecast_nll": float(STALE_SENTINEL),
            "npe_rejected": 0,
        },
        # The current stage: no `sim_forecast_nll`, no `npe_rejected`.
        {
            "global_step": 200,
            "stage": "T5_measured_return",
            "eegmmidb_real_eeg_nll": float(LIVE_SENTINEL),
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


class _FakeTrainingProcess:
    """A real process whose command line health.sh's detector matches.

    The reporting line is only reached when a training process is LIVE: with
    `procs=0` the script exits 2 ("a death, not a completion") long before it.
    These tests first passed only because run 3's ablation happened to be
    running at the time -- an accidental pass, which is the exact class this
    repo catalogues.

    A real process rather than an env override, so the test exercises the actual
    `pgrep -af` path including its interpreter filter and its bash/make
    exclusions. Those filters are themselves a documented trap (six misfires),
    and a hook that bypassed them would stop testing them.
    """

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)", "scwbd.foundation.train"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.4)  # let it appear in the process table

    def __enter__(self) -> "_FakeTrainingProcess":
        return self

    def __exit__(self, *exc: object) -> None:
        self.proc.kill()
        self.proc.wait(timeout=10)


def _run_health(tmp_path: Path) -> str:
    log = tmp_path / "train.jsonl"
    _write_log(log)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()

    env = {
        **os.environ,
        "LOG": str(log),
        "CKPT": str(ckpt),
        "CONFIG": str(REPO / "configs/run3/scwbd-003.yaml"),
        # Below the last step, so the script takes the "still running" path and
        # reaches the reporting line rather than the COMPLETE branch.
        "TARGET": "999999",
        "STALE_S": "999999",
    }
    proc = subprocess.run(
        ["bash", str(HEALTH)], capture_output=True, text=True, env=env, cwd=str(REPO)
    )
    return proc.stdout + proc.stderr


@pytest.mark.skipif(not HEALTH.is_file(), reason="health.sh absent")
def test_a_field_from_an_ended_stage_is_not_reported_as_current(tmp_path: Path) -> None:
    with _FakeTrainingProcess():
        out = _run_health(tmp_path)
    assert STALE_SENTINEL not in out, (
        f"health.sh reported a value only present in an earlier stage's rows:\n{out}\n"
        "This is the whole-file `tail -1` read. A number from a stage that ended "
        "hours ago, displayed beside a live step count, reads as a frozen loss."
    )


@pytest.mark.skipif(not HEALTH.is_file(), reason="health.sh absent")
def test_the_live_row_is_what_gets_reported(tmp_path: Path) -> None:
    """Not reporting the stale value is only half of it -- it must report the real one."""
    with _FakeTrainingProcess():
        out = _run_health(tmp_path)
    assert LIVE_SENTINEL in out, (
        f"health.sh reported neither the stale value nor the live one:\n{out}\n"
        "Suppressing the stale read without falling through to the measured "
        "likelihood would trade a wrong number for no number."
    )


@pytest.mark.skipif(not HEALTH.is_file(), reason="health.sh absent")
def test_a_field_absent_from_the_current_stage_says_so(tmp_path: Path) -> None:
    """`npe_rejected=0` claims a clean count; T5 runs no NPE and measured nothing."""
    with _FakeTrainingProcess():
        out = _run_health(tmp_path)
    assert "npe_rejected=n/a" in out, (
        f"expected `npe_rejected=n/a` for a stage that runs no NPE:\n{out}\n"
        "Reporting 0 asserts a rejection count that nothing computed."
    )


@pytest.mark.skipif(not HEALTH.is_file(), reason="health.sh absent")
def test_the_exit_code_contract_still_holds(tmp_path: Path) -> None:
    """0 = healthy. Read on its own line, which is the trap CLAUDE.md documents."""
    log = tmp_path / "train.jsonl"
    _write_log(log)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    proc = subprocess.run(
        ["bash", str(HEALTH)],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        env={
            **os.environ,
            "LOG": str(log),
            "CKPT": str(ckpt),
            "CONFIG": str(REPO / "configs/run3/scwbd-003.yaml"),
            "TARGET": "999999",
            "STALE_S": "999999",
        },
    )
    # `procs` is 0 in a test environment unless a real run is live, so accept
    # either the healthy line or the documented death exit -- what must NOT
    # happen is a crash from the field-reading change itself.
    assert proc.returncode in (0, 2), f"unexpected exit {proc.returncode}:\n{proc.stdout}{proc.stderr}"
