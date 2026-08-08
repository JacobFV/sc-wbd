"""Gate #6: the Balloon-Windkessel parameters, measured on the weights.

The executable form of ISSUE-008. `real_bold_nll` climbing five orders of
magnitude is a symptom that needs interpretation; five physical parameters
bit-identical to their initialisation is the cause, stated as a fact about
bytes.

`BOLDHead.step` is the Balloon integrator and the **only** consumer of

    log_kappa   log_gamma   log_tau   alpha   neural_gain

so if those tensors never move, the ODE was never called, whatever the BOLD
loss reported. `SCWBD.rollout(with_hemo=True)` is the only caller of `step`,
and only `sim_losses` passes it -- `real_bold_losses` reads the `hemo`
component straight out of the learned regional state and hands two
unconstrained latents to a Balloon signal equation as blood volume and
deoxyhaemoglobin.

**The trap this file exists to avoid.** The obvious gate is "the five moved by
the end of the run". It is a false pass. `bold.*` is granted by all five stages
of `configs/run3/scwbd-003.yaml`, so permission is not what freezes them; and
T4_simulator *does* run the ODE, on synthetic data. A gradient arriving there
says nothing about whether measured fMRI reaches the physics, which is the
entire question. So the measurement below is scoped to stages that admit no
simulator, and a T4 gradient is explicitly refused as evidence.

This is the same distinction run 3 got wrong at a larger scale: `moved_since_init`
answers "did a gradient arrive", not "did it carry information".
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
#: readout terms -- they receive gradient from the signal equation whether or
#: not the ODE ran, so including them would mask exactly what is being measured.
BALLOON = ("log_kappa", "log_gamma", "log_tau", "alpha", "neural_gain")

#: Integrity tier of the simulator. A stage admitting it can move the Balloon
#: parameters from synthetic dynamics alone.
SIMULATOR_TIER = 4


def _load(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _stage_admits(stage: str) -> set[int] | None:
    """Tiers a stage admits, read from the run config rather than its name."""
    if not CONFIG.is_file():
        return None
    import yaml

    cfg = yaml.safe_load(CONFIG.read_text())
    for s in cfg["train"]["stages"]:
        if s["name"] == stage:
            cur = (s.get("extra") or {}).get("curriculum") or {}
            return {int(t) for t in (cur.get("admits") or [])}
    return None


def _measured_only_checkpoints() -> list[Path]:
    """Stage checkpoints whose stage admits no simulator tier.

    Derived from the config, not from the stage name, so renaming a stage or
    admitting tier 4 into it cannot quietly move a checkpoint out of scope.
    """
    out: list[Path] = []
    for p in sorted(RUN3.glob("stage_*.pt")):
        stage = p.name[len("stage_") : -len(".pt")]
        admits = _stage_admits(stage)
        if admits is not None and SIMULATOR_TIER not in admits:
            out.append(p)
    return out


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
    "ckpt", _measured_only_checkpoints(), ids=lambda p: p.name[len("stage_") : -len(".pt")]
)
def test_the_balloon_parameters_are_frozen_in_every_measured_only_stage(ckpt: Path) -> None:
    """ISSUE-008, asserted as the measured fact it is.

    This test is GREEN while the defect is open, and that is deliberate: it
    pins the diagnosis to bytes so the day the measured BOLD path changes --
    in either direction, fixed or further broken -- something goes red and
    names the issue.

    **When this goes red because the path was repaired**, the repair is done:
    close ISSUE-008 and invert this test to
    ``assert not frozen`` under the same parametrisation. Do not delete it.
    That inversion is gate #6 proper, and until the fix lands there is nothing
    for it to assert that would not be wishful.
    """
    frozen = _frozen_balloon(_load(ckpt))
    assert frozen == list(BALLOON), (
        f"{ckpt.name}: expected all five Balloon parameters frozen (ISSUE-008), "
        f"got frozen={frozen}. If the measured BOLD path now integrates the ODE, "
        "this is the good failure: close ISSUE-008 in reports/known_issues.md and "
        "invert this assertion to `assert not frozen`. If it is not, then some "
        "other path is writing to the haemodynamic parameters and the BOLD "
        "likelihood is no longer the only thing they answer to."
    )


def test_a_simulator_gradient_is_not_evidence_the_measured_path_integrates() -> None:
    """The false pass this file was written to refuse.

    T4_simulator calls `rollout(with_hemo=True)`, so the five parameters can
    move there on synthetic dynamics while measured fMRI still never reaches
    them. A gate reading `moved_since_init` at the end of the run would report
    them moved and discharge nothing.

    T5_measured_return admits tiers 1-2 only -- no simulator -- so the bytes
    across the T4 -> T5 boundary are the discriminator: if `real_bold_losses`
    integrated the ODE, a stage of measured BOLD would change them.
    """
    t4 = RUN3 / "stage_T4_simulator.pt"
    t5 = RUN3 / "stage_T5_measured_return.pt"
    if not (t4.is_file() and t5.is_file()):
        pytest.skip("run 3 has not reached the T4 -> T5 boundary yet")

    admits5 = _stage_admits("T5_measured_return") or set()
    assert SIMULATOR_TIER not in admits5, (
        "T5_measured_return now admits the simulator, so it can no longer "
        "discriminate a measured gradient from a synthetic one. Pick another "
        "measured-only stage or this test has quietly stopped measuring."
    )

    before, after = _balloon_hash(_load(t4)), _balloon_hash(_load(t5))
    if before == after:
        pytest.xfail(
            "ISSUE-008 confirmed across the T4 -> T5 boundary: a whole stage of "
            "measured BOLD left the five Balloon parameters bit-identical. The "
            "simulator moved them in T4; measured fMRI did not touch them."
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
    if not RUN3.is_dir() or not sorted(RUN3.glob("stage_*.pt")):
        pytest.skip("no run-3 stage checkpoint on disk yet")
    assert _measured_only_checkpoints(), (
        "run-3 stage checkpoints exist but none is measured-only, so the gate "
        "above is vacuous. Check `admits` in configs/run3/scwbd-003.yaml."
    )
