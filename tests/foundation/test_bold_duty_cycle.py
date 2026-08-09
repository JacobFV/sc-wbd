"""`model.bold_every` must gate the measured BOLD term.

The field landed with ISSUE-008's fix and was read by nothing: `config.py`
declared it, `real_bold_losses` mentioned it in a comment, and `run_stage` ran
the term on every step regardless. A config key describing a schedule the
trainer does not run is worse than no key -- the model card would quote a duty
cycle the run never had, and the fMRI likelihood's evidence count would be wrong
by exactly the factor the key claims.

The schedule is `FoundationTrainer.bold_due` rather than an inline modulo so it
can be asserted without a 500-step rollout.
"""

from __future__ import annotations

from types import SimpleNamespace

from scwbd.foundation.config import FoundationConfig
from scwbd.foundation.train import FoundationTrainer


def _tr(bold_every: int) -> SimpleNamespace:
    cfg = FoundationConfig()
    cfg.model.bold_every = bold_every
    return SimpleNamespace(cfg=cfg)


def test_every_step_at_one() -> None:
    tr = _tr(1)
    assert [FoundationTrainer.bold_due(tr, s) for s in range(1, 9)] == [True] * 8


def test_one_step_in_four_at_four() -> None:
    tr = _tr(4)
    due = [s for s in range(1, 21) if FoundationTrainer.bold_due(tr, s)]
    assert due == [4, 8, 12, 16, 20], (
        "the BOLD term did not run on the schedule the config declares, so the "
        "number of windows behind every fMRI number is not what the model card "
        "would say it is"
    )


def test_a_zero_or_negative_duty_cycle_does_not_disable_the_term() -> None:
    """`bold_every: 0` is a config error, not a way to turn the term off.

    Silently disabling the measured fMRI likelihood on a typo would leave
    `ds002336_real` admitted, contributing nothing, and reading as trained.
    Turning it off is done by not admitting the source.
    """
    for bad in (0, -3):
        tr = _tr(bad)
        assert FoundationTrainer.bold_due(tr, 1) is True
        assert FoundationTrainer.bold_due(tr, 2) is True


def test_the_run_stage_loop_consults_it() -> None:
    """Guards the guard: `bold_due` could be correct and never called.

    That was the defect -- the field existed, its meaning was written down, and
    the loop ignored it. Asserted against the source because the alternative is
    a stage-length run on the GPU.
    """
    import inspect

    src = inspect.getsource(FoundationTrainer.run_stage)
    assert "self.bold_due(step)" in src, (
        "`run_stage` does not consult the duty cycle. `model.bold_every` would "
        "then be a config key that describes a schedule the trainer does not run."
    )
