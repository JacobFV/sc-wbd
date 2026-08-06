"""Gradient permissions ``A_k`` must actually block gradients.

ARCHITECTURE.md §7 rule 2: "A source updates only the modules its
``GradientPermission`` names."  The strong form of that statement is that a
non-permitted parameter's ``.grad`` is ``None`` -- not zero, not "we zero it
later", ``None``.  Zeroing after the fact is undone by the next optimiser step
with momentum; taking the gradient only w.r.t. permitted parameters is not.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from scwbd.foundation.mixture import (
    ROLE_AUTHORITY,
    ConflictLog,
    ConflictPolicy,
    GradientGate,
    LossNormalizer,
    MixtureTrainer,
    MixtureWeights,
    RoleViolation,
    SourceSpec,
    assert_not_row_count,
)

REPO = Path(__file__).resolve().parents[2]
CARDS = REPO / "configs" / "source_cards"


class ToyModel(nn.Module):
    """Named modules that mirror the foundation model's permission vocabulary."""

    def __init__(self) -> None:
        super().__init__()
        self.local = nn.Linear(4, 4)
        self.coupling = nn.Linear(4, 4)
        self.eeg = nn.Linear(4, 4)
        self.bold = nn.Linear(4, 4)

    def forward(self, x):
        return self.bold(self.eeg(self.coupling(self.local(x))))


def _spec(**kw) -> SourceSpec:
    base = dict(id="s", role="likelihood", losses=("likelihood",), n_eff=10.0, measurement_variance=1.0, bias_halfwidth=0.1)
    base.update(kw)
    return SourceSpec(**base)


# ----------------------------------------------------------------------
def test_non_permitted_parameters_have_grad_none():
    m = ToyModel()
    spec = _spec(id="eeg_only", gradient_permission=("eeg.*",))
    gate = GradientGate(m, {"eeg_only": spec})
    loss = m(torch.randn(8, 4)).pow(2).mean()
    grads = gate.grads(loss, "eeg_only", retain_graph=False)
    gate.accumulate(grads)
    for name, p in m.named_parameters():
        if name.startswith("eeg."):
            assert p.grad is not None, f"{name} should have been updated"
            assert torch.isfinite(p.grad).all()
        else:
            assert p.grad is None, f"{name} received a gradient it was not permitted"


def test_frozen_beats_grant():
    m = ToyModel()
    spec = _spec(id="s", gradient_permission=("*",), frozen=("coupling.*",))
    gate = GradientGate(m, {"s": spec})
    gate.accumulate(gate.grads(m(torch.randn(4, 4)).sum(), "s", retain_graph=False))
    assert m.coupling.weight.grad is None
    assert m.local.weight.grad is not None


def test_evaluation_only_and_negative_control_get_zero_weight():
    for role in ("evaluation_only", "negative_control"):
        s = SourceSpec(id=role, role=role, losses=())
        assert s.reliability() == 0.0
        assert ROLE_AUTHORITY[role] == 0.0


def test_simulated_source_cannot_be_a_likelihood():
    with pytest.raises(RoleViolation, match="simulator-conditioned"):
        SourceSpec(id="sim", role="likelihood", is_simulated=True, losses=("likelihood",))


def test_teacher_must_be_distillation_and_is_off_by_default():
    with pytest.raises(RoleViolation, match="never a subject likelihood"):
        SourceSpec(id="t", role="likelihood", is_teacher=True, losses=("likelihood",))
    cards = SourceSpec.load_dir(CARDS)
    assert "tribe_v2_teacher" in cards
    assert cards["tribe_v2_teacher"].enabled is False, "TRIBE v2 distillation must be OFF by default"
    assert cards["tribe_v2_teacher"].role == "distillation"


def test_role_cannot_declare_a_loss_it_does_not_license():
    with pytest.raises(RoleViolation, match="licenses"):
        SourceSpec(id="cal", role="calibration", losses=("likelihood",))


# ----------------------------------------------------------------------
# w_e is a reliability term, not a row count
# ----------------------------------------------------------------------
def test_weight_is_not_proportional_to_row_count():
    cards = list(SourceSpec.load_dir(CARDS).values())
    assert cards
    assert_not_row_count(cards)


def test_weight_responds_to_bias_and_discrepancy():
    clean = _spec(id="a", n_eff=100, measurement_variance=0.1, bias_halfwidth=0.0, model_discrepancy=0.0)
    biased = _spec(id="b", n_eff=100, measurement_variance=0.1, bias_halfwidth=1.0, model_discrepancy=0.0)
    noisy_sim = _spec(id="c", n_eff=100, measurement_variance=0.1, bias_halfwidth=0.0, model_discrepancy=1.0)
    assert biased.reliability() < clean.reliability()
    assert noisy_sim.reliability() < clean.reliability()
    narrow = _spec(id="d", n_eff=100, measurement_variance=0.1, bias_halfwidth=0.0, validity_overlap=0.2)
    assert narrow.reliability() < clean.reliability()


def test_unknown_bias_is_expensive_not_free():
    known = _spec(id="k", bias_halfwidth=0.0, measurement_variance=0.1)
    unknown = _spec(id="u", bias_halfwidth=None, measurement_variance=0.1)
    assert unknown.reliability() < known.reliability()


def test_calibration_cannot_outrank_a_likelihood_on_precision_alone():
    """A precise but non-biological source must not dominate (Appendix D)."""
    cal = _spec(id="cal", role="calibration", losses=("calibration",), n_eff=64, measurement_variance=0.001, bias_halfwidth=0.001)
    like = _spec(id="like", role="likelihood", losses=("likelihood",), n_eff=109, measurement_variance=0.35, bias_halfwidth=0.2)
    w = MixtureWeights({"cal": cal, "like": like})
    assert w["like"] > w["cal"] * 0.5, "role authority must temper pure inverse-variance weighting"


# ----------------------------------------------------------------------
# normalisation
# ----------------------------------------------------------------------
def test_losses_are_normalised_within_participant_and_source():
    n = LossNormalizer(warmup=0, momentum=0.5)
    small = torch.tensor(1.0)
    big = torch.tensor(1000.0)
    for _ in range(30):
        a = n(small.clone(), source="A", participant="p1")
        b = n(big.clone(), source="B", participant="p2")
    assert abs(float(a) - float(b)) < 0.5 * max(float(a), float(b)), (
        "a source whose raw loss is 1000x larger must not stay 1000x larger after normalisation"
    )


# ----------------------------------------------------------------------
# conflict
# ----------------------------------------------------------------------
def test_gradient_conflict_is_measured_by_module_and_source():
    log = ConflictLog(window=10)
    g = torch.randn(6)
    out = log.update({"a": {"local.w": g, "eeg.w": g}, "b": {"local.w": -g, "eeg.w": g}})
    assert out["conflict/a|b/local"] == pytest.approx(-1.0, abs=1e-4)
    assert out["conflict/a|b/eeg"] == pytest.approx(1.0, abs=1e-4)
    summary = log.summary()
    mods = {r["module"] for r in summary["pairs"]}
    assert mods == {"local", "eeg"}


def test_conflict_policy_escalates_rather_than_averaging():
    log = ConflictLog(window=50)
    g = torch.randn(8)
    for _ in range(25):
        log.update({"a": {"local.w": g}, "b": {"local.w": -g}})
    specs = {
        "a": _spec(id="a", n_eff=100, measurement_variance=0.1),
        "b": _spec(id="b", n_eff=2, measurement_variance=2.0),
    }
    pol = ConflictPolicy(min_observations=10)
    acts = pol.evaluate(log, MixtureWeights(specs))
    assert acts, "a persistently opposed pair must trigger an action"
    assert acts[0]["action"] == "adapter"
    assert acts[0]["yielding_source"] == "b", "the less reliable source yields"


def test_mixture_trainer_applies_per_source_masks():
    m = ToyModel()
    specs = {
        "eeg_src": _spec(id="eeg_src", gradient_permission=("eeg.*",), n_eff=50, measurement_variance=0.5),
        "bold_src": _spec(id="bold_src", gradient_permission=("bold.*",), n_eff=50, measurement_variance=0.5),
    }
    mt = MixtureTrainer(m, specs)
    x = torch.randn(8, 4)
    losses = {"eeg_src": m(x).pow(2).mean(), "bold_src": m(x + 1).abs().mean()}
    diag = mt.step(losses)
    assert m.eeg.weight.grad is not None
    assert m.bold.weight.grad is not None
    assert m.local.weight.grad is None, "no source permitted `local`, so it must stay untouched"
    assert m.coupling.weight.grad is None
    assert set(diag["weights"]) == {"eeg_src", "bold_src"}
    rep = mt.report()
    assert set(rep["per_source_contribution"]) == {"eeg_src", "bold_src"}
    assert rep["gradient_permission_audit"]["eeg_src"]["n_blocked"] > 0


def test_shipped_source_cards_load_and_validate():
    cards = SourceSpec.load_dir(CARDS)
    assert {"sim_wholebrain", "eegmmidb_real", "anatomical_prior"} <= set(cards)
    assert cards["sim_wholebrain"].role == "prior", "a simulator is never a subject likelihood"
    assert cards["sim_wholebrain"].is_simulated is True
    assert cards["eegmmidb_real"].role == "likelihood"
    # the simulator's n_eff must be the number of parameter sets, and its rows
    # must dwarf it without buying authority
    assert cards["sim_wholebrain"].n_rows > cards["sim_wholebrain"].n_eff
    assert cards["negative_control_shuffled"].reliability() == 0.0
