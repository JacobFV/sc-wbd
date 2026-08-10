"""A stage of a few steps must build its LR schedule.

`OneCycleLR` divides by the length of each phase. `pct_start = min(0.3, warmup /
steps)` can round the warmup and anneal boundaries onto the same step, and then
`get_lr` raises

    ZeroDivisionError: float division by zero

before the first batch. Measured on a 6-step smoke of run 4's `T6_individual`:
`min(0.3, 1/6)` put both boundaries on step 1.

No production stage reaches it -- 1,200 steps with a 60-step warmup is 0.05 --
which is the reason it is worth a guard rather than a reason it is not. CLAUDE.md
and HANDOFF-004 both ask for a SHORT bounded run before a launch and before any
cost claim; a scheduler that crashes only on short runs makes the precondition
harder to run than the thing it is a precondition for, and the way that gets
resolved is by skipping the precondition.
"""

from __future__ import annotations

import pytest
import torch

from scwbd.foundation.train import ONE_CYCLE_MIN_STEPS, _one_cycle_pct_start


@pytest.mark.parametrize("steps", [1, 2, 3, 4, 5, 6, 7, 9, 12, 23, 50, 600, 1200, 5000])
@pytest.mark.parametrize("warmup", [0, 1, 2, 60, 250])
def test_the_schedule_builds_and_steps(steps: int, warmup: int) -> None:
    """A real scheduler, built and stepped.

    Asserting the arithmetic instead would have passed the first version of
    this guard, whose bounds were off by one: `1/n` still threw at every step
    count including 600, because OneCycle's first phase ends at
    `pct_start * n - 1` and needs that to be at least 1.
    """
    p = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.AdamW([p], lr=1e-3)
    total = max(steps, ONE_CYCLE_MIN_STEPS)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=[1e-3],
        total_steps=total,
        pct_start=_one_cycle_pct_start(warmup, steps),
    )
    for _ in range(total):
        opt.step()
        sched.step()
    assert all(g["lr"] > 0 for g in opt.param_groups)


def test_the_minimum_is_the_smallest_workable_one() -> None:
    """`ONE_CYCLE_MIN_STEPS` must be exactly where torch starts working.

    Too low and `run_stage` still throws on a 2-step smoke; too high and it
    silently runs more steps than the stage asked for. Enumerated, so a torch
    upgrade that moves the boundary fails here rather than at launch.
    """
    def builds(n: int, pct: float) -> bool:
        p = torch.nn.Parameter(torch.zeros(2))
        o = torch.optim.AdamW([p], lr=1e-3)
        try:
            s = torch.optim.lr_scheduler.OneCycleLR(o, max_lr=[1e-3], total_steps=n, pct_start=pct)
            for _ in range(n):
                o.step()
                s.step()
            return True
        except Exception:  # noqa: BLE001 - any failure means this n is unusable
            return False

    n = ONE_CYCLE_MIN_STEPS
    assert any(builds(n, k / n) for k in range(1, n)), (
        f"no pct_start works at total_steps={n}; ONE_CYCLE_MIN_STEPS is too low"
    )
    assert not any(builds(n - 1, k / (n - 1)) for k in range(1, n - 1)), (
        f"total_steps={n - 1} works, so ONE_CYCLE_MIN_STEPS is higher than it "
        "needs to be and a short stage runs more steps than it asked for"
    )


def test_a_production_warmup_is_left_alone() -> None:
    """The clamp must not silently rewrite a schedule anyone chose.

    Run 4's stages are (steps, warmup) = (4000, 200), (600, 60), (800, 80),
    (5000, 250), (3000, 150), (1200, 60). All are far from the boundary and
    must come out exactly as `warmup / steps`.
    """
    for steps, warmup in ((4000, 200), (600, 60), (800, 80), (5000, 250), (3000, 150), (1200, 60)):
        assert _one_cycle_pct_start(warmup, steps) == pytest.approx(warmup / steps)


def test_the_cap_at_three_tenths_survives() -> None:
    """A warmup longer than 30% of the run is still capped, as before."""
    assert _one_cycle_pct_start(900, 1000) == pytest.approx(0.3)


def test_run_stage_uses_the_helper() -> None:
    """Guards the guard: the helper can be correct and never called.

    That is the shape of the defect it replaced -- `pct_start` was computed
    inline, so nothing could assert anything about it without a stage-length
    run on the GPU. Asserted against the source for the same reason.
    """
    import inspect

    from scwbd.foundation.train import FoundationTrainer

    src = inspect.getsource(FoundationTrainer.run_stage)
    assert "_one_cycle_pct_start(stage.warmup, stage.steps)" in src, (
        "`run_stage` computes `pct_start` itself again. A short bounded stage "
        "then raises ZeroDivisionError from inside OneCycleLR before the first "
        "batch, and the pre-launch smoke this repo requires cannot be run."
    )
    assert "total_steps=max(stage.steps, ONE_CYCLE_MIN_STEPS)" in src, (
        "`run_stage` allows fewer total steps than OneCycleLR has any valid "
        "`pct_start` for"
    )
