"""A smoke run must not share a ``run_name`` with the run it is smoking.

``--out`` moves the CHECKPOINTS. The JSONL training log is keyed by
``train.run_name``, so a scratch run launched as
``train --config <the run> --quick --out <scratch>`` still appends to the
production log, and afterwards the two are indistinguishable.

This is ISSUE-010, which took five attempts to close because each fix redirected
one more output (``ckpt_every``, then ``log_every``, then a logger redirect, then
``out_dir``) instead of redirecting the DIRECTORIES. It destroyed a checkpoint
and a published report on the way.

It came back on 2026-08-10 in ``scripts/launch_run4.sh`` step 5, which ran the
run's own config with ``--quick --out checkpoints/scwbd-004-smoke``. The smoke
created ``reports/training/scwbd-004_train.jsonl`` -- the production log of a run
that had not started. Caught at **0 bytes** by ``make health-run4`` reporting
"log is EMPTY" where it had reported "log NOT FOUND" an hour earlier, so the
watchdog is what noticed. ``configs/run3/smoke.yaml`` already carried the warning
in a comment; a comment is not a guard.

These tests need no GPU, no data and no checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scwbd.foundation.config import load_config

REPO = Path(__file__).resolve().parents[2]
LAUNCH = REPO / "scripts/launch_run4.sh"

#: Every (run config, smoke config) pair that must differ. Discovered rather
#: than listed, so run 5's pair is covered the day it is written.
def _pairs() -> list[tuple[Path, Path]]:
    out = []
    for smoke in sorted((REPO / "configs").glob("run*/smoke.yaml")):
        run = next(
            (p for p in sorted(smoke.parent.glob("*.yaml")) if p.name != "smoke.yaml"),
            None,
        )
        if run is not None:
            out.append((run, smoke))
    return out


@pytest.mark.parametrize("run,smoke", _pairs(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_the_smoke_config_is_keyed_differently(run: Path, smoke: Path) -> None:
    r = load_config(str(run)).train
    s = load_config(str(smoke)).train
    assert s.run_name != r.run_name, (
        f"{smoke.relative_to(REPO)} shares run_name {s.run_name!r} with "
        f"{run.relative_to(REPO)}. The JSONL log is keyed by run_name, so the smoke "
        "would append to the production log and nothing afterwards could tell the two "
        "apart. --out does not help: it moves the checkpoints only."
    )
    assert s.out_dir != r.out_dir, (
        f"{smoke.relative_to(REPO)} shares out_dir {s.out_dir!r} with "
        f"{run.relative_to(REPO)}; the smoke would overwrite the run's checkpoints."
    )


def test_there_is_a_pair_to_check() -> None:
    """An empty parametrisation passes vacuously; this repo has lost work to that."""
    assert _pairs(), "no run/smoke config pairs discovered under configs/run*/"


@pytest.mark.skipif(not LAUNCH.is_file(), reason="launch script absent")
def test_the_launch_script_smokes_through_a_smoke_config() -> None:
    """Step 5 must not point at the run's own config with --quick.

    The distinction is invisible in a log until it is too late: both spellings
    run, both produce checkpoints in the scratch directory, and only one of them
    also writes the production JSONL.
    """
    src = LAUNCH.read_text()
    assert "SMOKE_CONFIG" in src, (
        "the launch script no longer uses a dedicated smoke config. Running the run's "
        "own config with --quick leaves the log keyed by the run's run_name."
    )
    assert "--config \"$SMOKE_CONFIG\" --quick" in src, (
        "the smoke no longer invokes $SMOKE_CONFIG. Check step 5 has not reverted to "
        '`--config "$CONFIG" --quick --out ...`, which is the defect.'
    )
    assert '--config "$CONFIG" --quick' not in src, (
        "the launch script still smokes through the RUN's config. --out moves the "
        "checkpoints; the log follows train.run_name."
    )
    # And it must refuse at runtime too, not merely be written correctly today.
    assert 'if [ "$SMOKE_NAME" = "$RUN_NAME" ]' in src, (
        "the launch script no longer compares the smoke's run_name to the run's at "
        "runtime. A config edit could re-introduce the collision without touching "
        "this script, and the static check above would still pass."
    )


# ======================================================================
# a smoke must REACH every stage, and a run must never be capped
# ======================================================================
def test_the_smoke_caps_steps_so_it_reaches_every_stage() -> None:
    """`--quick` shrinks rosters, not step counts.

    The first working run-4 smoke ran T1's 4,000 steps at ~9.3 s/step, hit
    `max_wall_seconds`, and stopped -- never entering T4 or T6, the two stages
    carrying this run's new code (the posterior's rewritten conditioning, and
    the first fitted individualiser). It exercised only the paths that already
    worked, which is not the check HANDOFF-004 step e asks for.
    """
    for run, smoke in _pairs():
        cap = getattr(load_config(str(smoke)).train, "max_steps_per_stage", None)
        assert cap is not None and cap > 0, (
            f"{smoke.relative_to(REPO)} does not cap max_steps_per_stage, so it runs "
            "the run's full step counts and stops at the wall clock partway through "
            "the first stage. It would never reach the later stages."
        )
        # ONE_CYCLE_MIN_STEPS is 3, and one step cannot catch a term that raises
        # on its SECOND batch -- a stale cached shape, an exhausted iterator.
        assert cap >= 4, f"{smoke.relative_to(REPO)} caps at {cap}; use at least 4"


@pytest.mark.parametrize("run,smoke", _pairs(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_the_run_config_never_caps_steps(run: Path, smoke: Path) -> None:
    """`max_steps_per_stage` in a RUN config silently truncates training.

    Every stage would finish early and look normal; the log would show each
    stage completing, and only the step counts would betray it. It belongs in a
    smoke config and nowhere else.
    """
    cap = getattr(load_config(str(run)).train, "max_steps_per_stage", None)
    assert cap is None, (
        f"{run.relative_to(REPO)} sets train.max_steps_per_stage={cap}. That caps "
        "EVERY stage and exists only so a smoke can reach all of them. In a run it "
        "truncates training silently."
    )


# ======================================================================
# the REPORT directory, not just the log key
# ======================================================================
@pytest.mark.parametrize("run,smoke", _pairs(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_the_smoke_writes_reports_somewhere_else_entirely(run: Path, smoke: Path) -> None:
    """`report_dir` must differ, or the smoke overwrites the run's reports.

    A distinct `run_name` moves the JSONL and a distinct `out_dir` moves the
    checkpoints -- and neither moves the mixture reports. `run_stage` writes each
    one TWICE: run-scoped under `report_dir/<run_name>/`, and to a legacy FLAT
    `report_dir/mixture_<stage>.json` that every run with matching stage names
    overwrites by design.

    Run 4 reuses run 3's stage names, so its smoke replaced 22,113 lines of
    `mixture_T1_measured_founding.json` with 81 lines of smoke output, plus four
    siblings. They are tracked, so git restored them -- but run 3's run-scoped
    directory holds only `T4`, because that write landed mid-run, so for
    T1/T2/T3/T5 the flat files are run 3's ONLY record.

    Redirecting the DIRECTORY is the fix rather than enumerating the outputs.
    ISSUE-010 needed five attempts because each one redirected one more output
    and something was still writing to a production path every time.
    """
    r = load_config(str(run)).train
    s = load_config(str(smoke)).train
    assert s.report_dir != r.report_dir, (
        f"{smoke.relative_to(REPO)} shares report_dir {s.report_dir!r} with "
        f"{run.relative_to(REPO)}. run_name moves the log and out_dir moves the "
        "checkpoints; NEITHER moves the mixture reports, and the flat "
        "`mixture_<stage>.json` is overwritten by any run sharing a stage name."
    )
    assert not str(s.report_dir).rstrip("/") == str(r.report_dir).rstrip("/"), (
        "report_dir differs only by a trailing slash"
    )
