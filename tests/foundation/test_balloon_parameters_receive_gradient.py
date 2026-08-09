"""Gate #6: the Balloon-Windkessel parameters, measured on the weights.

The executable form of ISSUE-008. `real_bold_nll` climbing five orders of
magnitude is a symptom that needs interpretation; five physical parameters
bit-identical to their initialisation is the cause, stated as a fact about
bytes.

    log_kappa   log_gamma   log_tau   alpha   neural_gain

`BOLDHead.step` is the Balloon integrator and the only consumer of them, so if
those tensors never move, the ODE was never called -- whatever the BOLD loss
reported.

**The stronger fact, measured at the T4 -> T5 boundary.** They are frozen in
`T4_simulator` too. The simulator stage does not integrate the ODE either, for
three independent reasons, any one of which is sufficient:

* `StageAdmission.with_hemo` defaults to False and no stage of
  `configs/run3/scwbd-003.yaml` sets it, so `rollout()` is never asked for
  haemodynamics. The `stage.name in ("IV_assembly",)` fallback in
  `sim_losses` cannot fire either -- run 3's stages are named T1..T5.
* Even under `with_hemo=True`, `sim_losses` reads `roll.activity` and never
  `roll.hemo`. The compartments would be integrated and discarded.
* The one term that does touch four of the five, `BOLDHead.prior_penalty()`,
  reaches them through two routes and neither is open: `sim_losses` scales it
  by `lambda_cal`, which is set only in T2 (a stage admitting no simulator, so
  `sim_losses` does not run there); and `anat_losses` adds it unscaled, but
  `configs/curriculum/source_cards/anatomical_prior.yaml` freezes `bold.*`
  deliberately -- *"Training bold.* would have made a prior penalty its sole
  author."*

So the ODE was never integrated at any point in run 3, and the correct summary
of ISSUE-008 is not "the measured path does not call the physics" but "nothing
does".

**What this file refuses.** An earlier version of this gate scoped itself to
stages admitting no simulator, on the theory that T4 would move the five and a
gradient arriving there would be a false pass. That theory was wrong -- checked
against `stage_T4_simulator.pt`, they are frozen there too. The scoping is gone
because it was protecting against something that does not happen, and a guard
carrying a false rationale teaches the next reader the wrong model. What
survives from it is the real lesson: `bold.*` is granted by all five stages, so
permission was never what froze these, and the run-2 reflex ("some card's glob
names nothing") is the wrong diagnosis here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUN3 = REPO / "checkpoints/scwbd-003"
CONFIG = REPO / "configs/run3/scwbd-003.yaml"

#: The five parameters of the Balloon-Windkessel model, and nothing else in
#: `BOLDHead`. `log_noise`, `logvar_gain` and `rho` are observation-noise and
#: readout terms -- they take gradient from the signal equation whether or not
#: the ODE ran, and all three DID move, so including them would mask exactly
#: what is being measured.
BALLOON = ("log_kappa", "log_gamma", "log_tau", "alpha", "neural_gain")


def _load(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _stage_checkpoints() -> list[Path]:
    return sorted(RUN3.glob("stage_*.pt"))


def _frozen_balloon(ck: dict) -> list[str]:
    """Which of the five are bit-identical to their initialisation."""
    rep = (ck.get("extra") or {}).get("moved_since_init")
    if not rep:
        pytest.skip("checkpoint predates `moved_since_init`")
    frozen = rep.get("frozen_tensors") or []
    # `frozen_tensors` is truncated at 200 entries. Under truncation, absence
    # from the list is not evidence of movement, and a gate that read it as
    # such would pass by overflowing rather than by training.
    if len(frozen) >= 200 and rep.get("n_frozen", 0) > len(frozen):
        pytest.skip(
            f"frozen_tensors is truncated ({len(frozen)} of {rep['n_frozen']}); "
            "absence from it is not evidence"
        )
    return [n for n in BALLOON if f"bold.{n}" in frozen]


def _balloon_hash(ck: dict) -> str:
    sd = ck.get("model") or {}
    h = hashlib.sha256()
    for n in BALLOON:
        t = sd.get(f"bold.{n}")
        if t is None:
            pytest.skip(f"checkpoint has no bold.{n}")
        h.update(n.encode())
        h.update(t.detach().to("cpu").float().numpy().tobytes())
    return h.hexdigest()


@pytest.mark.parametrize(
    "ckpt", _stage_checkpoints(), ids=lambda p: p.name[len("stage_") : -len(".pt")]
)
def test_the_balloon_parameters_are_frozen_in_every_stage(ckpt: Path) -> None:
    """ISSUE-008, asserted as the measured fact it is -- in EVERY stage.

    This test is GREEN while the defect is open, and that is deliberate: it
    pins the diagnosis to bytes so the day the BOLD path changes -- in either
    direction, repaired or further broken -- something goes red and names the
    issue.

    **When this goes red because the path was repaired**, the repair is done:
    close ISSUE-008 and invert this test to ``assert not frozen``. Do not
    delete it. That inversion is gate #6 proper, and until the fix lands there
    is nothing for it to assert that would not be wishful.
    """
    frozen = _frozen_balloon(_load(ckpt))
    assert frozen == list(BALLOON), (
        f"{ckpt.name}: expected all five Balloon parameters frozen (ISSUE-008), "
        f"got frozen={frozen}. If the BOLD path now integrates the ODE, this is "
        "the good failure: close ISSUE-008 in reports/known_issues.md and invert "
        "this assertion to `assert not frozen`. If it is not, then some other "
        "path is writing to the haemodynamic parameters and the BOLD likelihood "
        "is no longer the only thing they answer to."
    )


def test_the_simulator_stage_does_not_integrate_the_ode_either() -> None:
    """The correction to this gate's first draft, kept as a check.

    It is tempting to assume the simulator exercises the physics -- it is the
    stage named for it, and `sim_losses` is the only caller that can pass
    `with_hemo`. It does not. Asserted here so the assumption cannot quietly
    return, and so that turning `with_hemo` on without also consuming
    `roll.hemo` in a loss does not read as a fix.
    """
    if not CONFIG.is_file():
        pytest.skip("run-3 config absent")
    import yaml

    cfg = yaml.safe_load(CONFIG.read_text())
    hemo_on = [
        s["name"]
        for s in cfg["train"]["stages"]
        if ((s.get("extra") or {}).get("curriculum") or {}).get("with_hemo")
    ]
    assert not hemo_on, (
        f"{hemo_on} now request with_hemo. That alone does not discharge "
        "ISSUE-008: `sim_losses` reads roll.activity and never roll.hemo, so "
        "the compartments would be integrated and thrown away. Check that a "
        "loss actually consumes them before treating this as repaired."
    )

    t4 = RUN3 / "stage_T4_simulator.pt"
    if not t4.is_file():
        pytest.skip("run 3 has not written the simulator stage yet")
    assert _frozen_balloon(_load(t4)) == list(BALLOON), (
        "the simulator stage moved the Balloon parameters. If that is real, the "
        "ODE is being integrated somewhere this file does not know about, and "
        "the T4 -> T5 comparison below stops meaning what it says."
    )


def test_the_measured_return_stage_leaves_them_untouched() -> None:
    """T5 admits tiers 1-2 and is a whole stage of measured data.

    Byte comparison across the T4 -> T5 boundary, independent of
    `moved_since_init`: two different mechanisms agreeing is worth more than
    one mechanism asserted twice.
    """
    t4 = RUN3 / "stage_T4_simulator.pt"
    t5 = RUN3 / "stage_T5_measured_return.pt"
    if not (t4.is_file() and t5.is_file()):
        pytest.skip("run 3 has not reached the end of T5 yet")

    before, after = _balloon_hash(_load(t4)), _balloon_hash(_load(t5))
    if before == after:
        pytest.xfail(
            "ISSUE-008 confirmed across the T4 -> T5 boundary: a whole stage of "
            "measured BOLD left the five Balloon parameters bit-identical."
        )
    # Reached only once the measured path integrates. Keep the assertion so the
    # xfail above cannot be the permanent resting state of this test.
    assert before != after


def test_there_is_something_to_check() -> None:
    """Guards the guard: an empty parametrisation is a green that measured nothing.

    `reports/decorative_guards.md` names this shape. With no run-3 checkpoint on
    disk every parametrised test above collects zero cases and the file reports
    success while asserting nothing at all.
    """
    if not RUN3.is_dir():
        pytest.skip("no run-3 checkpoint directory yet")
    assert _stage_checkpoints(), (
        "no run-3 stage checkpoint on disk, so the gate above is vacuous."
    )
