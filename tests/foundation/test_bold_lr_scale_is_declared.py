"""`bold_lr_scale` must not reach a run config by accident.

ISSUE-016: the measured BOLD likelihood degrades during training because the
SHARED TRUNK moves under it -- `ds002336_real` is 4.13% of the mixture and is
outvoted 23.2:1 by the EEG-like sources. Measured on matched LR schedules with
the same seed:

    arm A  as launched                       real_bold_nll 3.21 @160, 12.96 @400
    arm B  five Balloon ODE constants frozen               3.70 @160 -- no better
    arm C  shared trunk frozen                             1.92 @160, falling

`train.bold_lr_scale` puts `bold.*` in its own optimiser group so the head can
TRACK a trunk that is moving rather than be dragged by it. It defaults to **1.0**,
which leaves `bold.*` in the main group and the run bit-identical to before.

That default is the point. The field was added to run a probe (arm D at 5.0), and
a probe value silently inherited into a run config would change what the run is
while every report still described the old one. The remedy is pre-registered in
`reports/RUN4_LAUNCH_PLAN.md` §6 and adopted only if arm D's number clears the
bar written before the data existed.

These tests need no GPU, no data and no checkpoint.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from scwbd.foundation.config import TrainConfig, load_config
from scwbd.foundation.train import FoundationTrainer

REPO = Path(__file__).resolve().parents[2]


def test_the_default_leaves_the_run_unchanged() -> None:
    """1.0 must mean 'no separate group', not 'a separate group at the same rate'.

    A separate group at an equal rate is NOT a no-op: AdamW keeps per-group
    state, and `OneCycleLR` is built with one `max_lr` per group, so splitting
    the parameters changes the schedule's shape even when the rates match. The
    default has to be indistinguishable from the code before this field existed.
    """
    assert TrainConfig().bold_lr_scale == 1.0, (
        "the default is no longer 1.0. Any other default silently changes every "
        "run that does not mention the field."
    )
    src = inspect.getsource(FoundationTrainer.run_stage)
    assert "self.bold_lr_scale != 1.0" in src, (
        "`run_stage` no longer special-cases 1.0. At 1.0 `bold.*` must stay in the "
        "main parameter group; a separate group at an equal rate still changes the "
        "OneCycleLR max_lr list and AdamW's per-group state."
    )
    assert "model_params + bold_params" in src, (
        "the 1.0 branch no longer folds `bold.*` back into the main group, so the "
        "default is not the pre-existing behaviour."
    )


def test_the_run_config_does_not_carry_a_probe_value() -> None:
    """configs/run4/scwbd-004.yaml must not inherit 5.0 from a probe.

    The probes set it deliberately and say so in their own headers. The run adopts
    it only under §6's rule, and when it does, this test is the place that has to
    be updated -- deliberately, with the arm-D number written down beside it.
    """
    cfg = load_config(str(REPO / "configs/run4/scwbd-004.yaml"))
    got = cfg.train.bold_lr_scale
    assert got == 1.0, (
        f"configs/run4/scwbd-004.yaml sets bold_lr_scale={got}. If arm D earned it "
        "under RUN4_LAUNCH_PLAN.md §6 (real_bold_nll < 2.5 at step 340), update this "
        "test in the same commit and record the number that justified it. If it "
        "arrived by inheriting a probe config, remove it: the probes are diagnostics, "
        "not run settings."
    )


#: The ISSUE-016 arms, in their own directory. Globbed rather than listed so a
#: fifth arm is covered the day it is written.
def _probes() -> list[Path]:
    return sorted(p for p in (REPO / "configs/run4/probes").glob("*.yaml"))


def test_there_are_probes_to_check() -> None:
    """An empty glob passes a parametrized test vacuously.

    The probes were moved out of `configs/run4/` after their arms were recorded,
    which left the original `configs/run4/probe_*.yaml` glob matching NOTHING —
    the test below would have reported success while checking no files at all.
    This repo lost 88.8% of run 2's parameters to an unmatched glob.
    """
    assert _probes(), (
        "no probe configs found under configs/run4/probes/. If they were deleted, "
        "delete the checks below with them rather than leaving a glob that passes "
        "by matching nothing."
    )


def test_no_probe_config_sits_beside_the_run_config() -> None:
    """`configs/run4/` holds the run and its smoke, and nothing that looks like either.

    Three diagnostic arms lived there during ISSUE-016. Someone reaching for a
    config in a hurry could have taken one, and each writes to a different
    run_name, so the mistake would surface as a missing run rather than an error.
    """
    stray = sorted(
        p.name for p in (REPO / "configs/run4").glob("*.yaml")
        if p.name not in {"scwbd-004.yaml", "smoke.yaml"}
    )
    assert not stray, (
        f"configs/run4/ contains {stray}. Only the run config and its smoke belong "
        "there; diagnostics go in configs/run4/probes/ so nobody launches one."
    )


@pytest.mark.parametrize("probe", _probes(), ids=lambda p: p.name)
def test_every_probe_config_is_marked_as_a_diagnostic(probe: Path) -> None:
    """A probe must be unmistakable in the file, not only in a commit message.

    Each says what it is on its first lines and writes to its own run_name,
    out_dir and report_dir, so running one can neither be mistaken for the run
    nor overwrite the run's artifacts.
    """
    head = probe.read_text()[:400].upper()
    assert "PROBE" in head and ("DIAGNOSTIC" in head or "NOT A RUN" in head), (
        f"{probe.name} does not declare itself a diagnostic in its opening lines. "
        "It sits in the same directory as the run config."
    )
    t = load_config(str(probe)).train
    run = load_config(str(REPO / "configs/run4/scwbd-004.yaml")).train
    assert t.run_name != run.run_name, f"{probe.name} shares the run's run_name"
    assert t.out_dir != run.out_dir, f"{probe.name} shares the run's out_dir"
    assert t.report_dir != run.report_dir, (
        f"{probe.name} shares the run's report_dir, so its mixture reports would "
        "overwrite the run's. See the smoke's version of this same defect."
    )


def test_bold_params_are_selected_by_prefix_not_by_the_optimiser_order() -> None:
    """The group is built from `named_parameters()`, so it cannot drift silently.

    Selecting by index or by insertion order would break the first time a module
    was added to `BOLDHead`, and it would break by training the wrong tensors
    rather than by raising.
    """
    src = inspect.getsource(FoundationTrainer.run_stage)
    assert 'n.startswith("bold.")' in src, (
        "the BOLD parameter group is no longer selected by name prefix. Anything "
        "positional silently trains the wrong tensors when BOLDHead changes."
    )
