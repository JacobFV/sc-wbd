"""Every configured stage name must be one the trainer actually knows.

This is the test that would have caught run 2's largest defect on day one, and
it did not exist.

Run 2 renamed all six training stages. Three mechanisms in ``train.py`` are keyed
to the **run-1** names, and every one of them failed *toward permissive*:

===========================================  ==========================================
mechanism                                    what an unmatched name does
===========================================  ==========================================
``STAGE_PERMISSIONS.get(name, ("*",))``      full gradient permission, no restriction
``name in ("III_sliced", ...)`` real loss    no measured-data gradient, ever
``name == "V_individual"``                   no individualizer is built
===========================================  ==========================================

So the run trained for nine hours on simulated data alone, with the per-stage
gradient restrictions inert and no individualizer, under stage names like
``T1_measured_founding`` and ``T1_individualisation``. Nothing crashed. The loss
fell. ``npe_rejected`` stayed at zero.

> A dictionary lookup with a permissive default is a configuration system that
> cannot report a typo. ``.get(name, ("*",))`` and ``name in (...)`` are
> unfalsifiable by construction: there is no stage name they reject.

These tests make the *name* checkable. They deliberately do **not** assert that
any particular stage uses real data — that is a research decision. They assert
that the trainer has an opinion about every stage the config declares, so a
rename cannot silently turn a mechanism off.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: The names ``train.py`` gates a measured-data loss on. Kept here rather than
#: imported so that the test breaks loudly if the tuple in ``train.py`` is
#: edited without anyone thinking about the configs that depend on it.
REAL_DATA_STAGES = {"III_sliced", "IV_assembly", "V_individual"}
INDIVIDUALIZER_STAGE = "V_individual"

#: **These tests fail today, on purpose, and that is the point.**
#:
#: All four fail for all three run-2 configs -- the defect is in the config
#: family, not in one file. They are marked ``xfail(strict=True)`` rather than
#: deleted or skipped, which does three things a skip would not:
#:
#:   * the suite stays green, so a real defect does not block shipping 002;
#:   * the failure stays *recorded and runnable*, instead of living only in a
#:     report nobody executes;
#:   * ``strict=True`` means that when run 3 fixes the gates, these XPASS --
#:     which pytest reports as a **failure** -- forcing whoever fixed it to come
#:     back here and remove the marker. A guard that quietly starts passing is
#:     how a fix goes unrecorded.
#:
#: Do not "fix" these by widening REAL_DATA_STAGES to include the run-2 names.
#: That makes the tests pass without making the trainer use measured data, which
#: is the decorative-guard move this repository exists to catalogue. The fix is
#: in ``train.py``: key the mechanisms on a declared stage property so an
#: undeclared stage is a refusal at config load, not a silent wildcard.
#: See ``reports/RUN2.md`` §2b.
pytestmark = pytest.mark.xfail(
    strict=True,
    reason=(
        "run-2 configs renamed every stage; STAGE_PERMISSIONS, the real-data "
        "loss gate and the individualizer gate are all keyed to run-1 names. "
        "See reports/RUN2.md section 2b. Fix belongs in train.py, not here."
    ),
)

CONFIGS = [
    "configs/run2/pilot-families.yaml",
    "configs/run2/scwbd-001-families.yaml",
    "configs/run2/scwbd-001.yaml",
]


def _stages(rel: str):
    from scwbd.foundation.config import load_config

    return list(load_config(str(ROOT / rel)).train.stages)


@pytest.mark.parametrize("rel", CONFIGS)
def test_every_stage_name_is_known_to_the_permission_table(rel: str):
    """An unknown stage silently gets ``("*",)`` — unrestricted.

    This is the check that turns a typo into a failure instead of into a
    permission grant.
    """
    from scwbd.foundation.train import STAGE_PERMISSIONS

    if not (ROOT / rel).is_file():
        pytest.skip(f"{rel} not present")
    unknown = [s.name for s in _stages(rel) if s.name not in STAGE_PERMISSIONS]
    assert not unknown, (
        f"{rel}: {unknown} have no STAGE_PERMISSIONS entry, so each trains with "
        f'full gradient permission ("*"). Known: {sorted(STAGE_PERMISSIONS)}'
    )


@pytest.mark.parametrize("rel", CONFIGS)
def test_at_least_one_stage_takes_a_gradient_on_measured_data(rel: str):
    """A config whose stages never touch real data trains on simulation alone.

    Run 2 satisfied this vacuously for nine hours: the real-EEG loader was
    built and its split fingerprinted, and no stage was ever in the tuple that
    would have used it. The only loss field the log ever emitted was
    ``sim_forecast_nll``.
    """
    if not (ROOT / rel).is_file():
        pytest.skip(f"{rel} not present")
    names = {s.name for s in _stages(rel) if getattr(s, "steps", 0) > 0}
    assert names & REAL_DATA_STAGES, (
        f"{rel}: no stage in {sorted(REAL_DATA_STAGES)}, so `real_losses` is never "
        f"called and the model trains on simulation alone. Declared stages: "
        f"{sorted(names)}"
    )


@pytest.mark.parametrize("rel", CONFIGS)
def test_a_stage_named_for_individualisation_actually_individualises(rel: str):
    """``T1_individualisation`` built no individualizer and ran ordinary steps.

    Naming is not the defect on its own — the defect is a name that *claims* a
    mechanism the trainer will not run. So this checks the pairing in both
    directions rather than banning any particular word.
    """
    if not (ROOT / rel).is_file():
        pytest.skip(f"{rel} not present")
    for s in _stages(rel):
        claims = "individual" in s.name.lower()
        runs = s.name == INDIVIDUALIZER_STAGE
        if claims and not runs:
            pytest.fail(
                f"{rel}: stage {s.name!r} is named for individualisation but the "
                f"trainer only builds an individualizer for {INDIVIDUALIZER_STAGE!r}, "
                "so it runs ordinary training under a name that says otherwise"
            )


@pytest.mark.parametrize("rel", CONFIGS)
def test_a_stage_named_for_measurement_uses_measurement(rel: str):
    """``T1_measured_founding`` was not founded on measurements."""
    if not (ROOT / rel).is_file():
        pytest.skip(f"{rel} not present")
    for s in _stages(rel):
        if "measured" in s.name.lower() and s.name not in REAL_DATA_STAGES:
            pytest.fail(
                f"{rel}: stage {s.name!r} is named for measured data but is not in "
                f"{sorted(REAL_DATA_STAGES)}, so it takes no gradient on any "
                "measurement"
            )
