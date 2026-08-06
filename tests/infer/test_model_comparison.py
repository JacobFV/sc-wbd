"""Posterior branching: an observationally equivalent pair with different
intervention consequences must return ``UnresolvedCausalAmbiguity``.

The constructed pair is exact, not approximate.  Two 2-node linear--Gaussian
systems differ only in the *direction* of the coupling::

    M_forward:  1 -> 2        M_reverse:  2 -> 1

with a symmetric latent noise ``Q = q I``, equal time constants, and a read
head ``L = [1, 1]`` that sums the two nodes.  Swapping the node labels is a
permutation ``P`` with ``L P = L``, so the two models generate *identically
distributed* passive data -- the log-likelihood difference is zero to machine
precision.  Writing into node 1 breaks the symmetry: the forward model
propagates the input to node 2 and the reverse model does not.

This is refusal **R04** made concrete: an effective/causal operator cannot be
estimated from passive correlation alone.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from scwbd.infer.filters import (
    LinearGaussianSSM,
    ObservationChannel,
    deterministic_response,
    kalman_filter,
    simulate_lgssm,
)
from scwbd.infer.model_comparison import (
    ModelEvidence,
    bayes_factor,
    compare_models,
    laplace_log_evidence,
    psis_loo,
    stacking_weights,
    waic,
)
from scwbd.infer.types import CalibrationClaimError, UnresolvedCausalAmbiguity

torch.set_default_dtype(torch.float64)

T = 60
Q0 = 0.4
DECAY = 0.85
GAIN = 0.35


def _model(direction: str, drive_node: int | None = None):
    F = torch.tensor([[DECAY, 0.0], [0.0, DECAY]])
    if direction == "forward":
        F[1, 0] = GAIN                     # 1 -> 2
    else:
        F[0, 1] = GAIN                     # 2 -> 1
    F = F.unsqueeze(0)
    Q = (Q0 * torch.eye(2)).unsqueeze(0)
    P0 = (Q0 / (1 - DECAY**2) * torch.eye(2)).unsqueeze(0)
    m0 = torch.zeros(1, 2)
    H = torch.tensor([[1.0, 1.0]]).unsqueeze(0)     # symmetric read: L P = L
    R = (0.05 * torch.eye(1)).unsqueeze(0)
    inp = torch.zeros(1, T, 2)
    if drive_node is not None:
        inp[0, T // 3, drive_node] = 6.0            # one calibrated impulse
    return LinearGaussianSSM(
        F, Q, m0, P0, [ObservationChannel("y", H, R, torch.arange(T))], T, inp
    )


def test_passive_data_cannot_distinguish_the_direction():
    """Exact observational equivalence, asserted to machine precision."""
    fwd, rev = _model("forward"), _model("reverse")
    data, _ = simulate_lgssm(fwd, seed=3, batch=1)
    a = float(kalman_filter(fwd, data).log_likelihood[0])
    b = float(kalman_filter(rev, data).log_likelihood[0])
    assert abs(a - b) < 1e-9, f"models are not observationally equivalent: {a} vs {b}"


def test_intervention_does_distinguish_the_direction():
    fwd, rev = _model("forward", drive_node=0), _model("reverse", drive_node=0)
    mu_f = deterministic_response(fwd)["y"][0, :, 0]
    mu_r = deterministic_response(rev)["y"][0, :, 0]
    assert float((mu_f - mu_r).abs().max()) > 0.5
    data, _ = simulate_lgssm(fwd, seed=11, batch=1)
    a = float(kalman_filter(fwd, data).log_likelihood[0])
    b = float(kalman_filter(rev, data).log_likelihood[0])
    assert a - b > 2.3, "the impulse failed to separate the two models"


def test_unresolved_causal_ambiguity_is_returned_not_averaged():
    fwd, rev = _model("forward"), _model("reverse")
    data, _ = simulate_lgssm(fwd, seed=3, batch=1)
    ev = [
        ModelEvidence("coupling_1_to_2",
                      float(kalman_filter(fwd, data).log_likelihood[0]), "exact_kalman"),
        ModelEvidence("coupling_2_to_1",
                      float(kalman_filter(rev, data).log_likelihood[0]), "exact_kalman"),
    ]
    # what each model predicts for the *same* write into node 1
    pred = {
        "coupling_1_to_2": {
            "impulse_node1_peak_at_node2": float(
                deterministic_response(_model("forward", 0))["y"].abs().max()
            )
        },
        "coupling_2_to_1": {
            "impulse_node1_peak_at_node2": float(
                deterministic_response(_model("reverse", 0))["y"].abs().max()
            )
        },
    }
    out = compare_models(ev, intervention_predictions=pred,
                         intervention_uncertainty={"impulse_node1_peak_at_node2": 0.2})
    assert isinstance(out, UnresolvedCausalAmbiguity), out
    assert set(out.candidate_models) == {"coupling_1_to_2", "coupling_2_to_1"}
    assert out.max_log_evidence_gap < 1e-6
    assert out.intervention_divergence > out.divergence_threshold
    assert "R04" in out.resolution_experiment or "passive" in out.resolution_experiment
    with pytest.raises(CalibrationClaimError):
        out.averaged_recommendation()
    d = out.to_dict()
    assert d["type"] == "UnresolvedCausalAmbiguity"
    assert "mean" not in d and "best" not in d


def test_interventional_data_resolve_the_ambiguity():
    fwd = _model("forward", drive_node=0)
    rev = _model("reverse", drive_node=0)
    data, _ = simulate_lgssm(fwd, seed=11, batch=1)
    ev = [
        ModelEvidence("coupling_1_to_2", float(kalman_filter(fwd, data).log_likelihood[0]), "kf"),
        ModelEvidence("coupling_2_to_1", float(kalman_filter(rev, data).log_likelihood[0]), "kf"),
    ]
    out = compare_models(
        ev,
        intervention_predictions={"coupling_1_to_2": {"x": 1.0},
                                  "coupling_2_to_1": {"x": 3.0}},
        intervention_uncertainty={"x": 0.2},
    )
    assert not isinstance(out, UnresolvedCausalAmbiguity)
    assert out["best"] == "coupling_1_to_2"
    assert out["resolved"] is True


def test_no_ambiguity_when_interventions_agree():
    ev = [ModelEvidence("a", 10.0, "kf"), ModelEvidence("b", 10.2, "kf")]
    out = compare_models(ev,
                         intervention_predictions={"a": {"x": 1.0}, "b": {"x": 1.02}},
                         intervention_uncertainty={"x": 0.5})
    assert not isinstance(out, UnresolvedCausalAmbiguity)


def test_waic_and_loo_on_a_known_problem():
    rng = np.random.default_rng(0)
    n, S = 60, 400
    y = rng.normal(0.4, 1.0, n)
    draws_good = rng.normal(0.4, 1 / math.sqrt(n), S)
    draws_bad = rng.normal(2.5, 1 / math.sqrt(n), S)

    def ll(mu):
        return -0.5 * np.log(2 * np.pi) - 0.5 * (y[None, :] - mu[:, None]) ** 2

    g, b = waic(ll(draws_good)), waic(ll(draws_bad))
    assert g["elpd_waic"] > b["elpd_waic"]
    lg, lb = psis_loo(ll(draws_good)), psis_loo(ll(draws_bad))
    assert lg["elpd_loo"] > lb["elpd_loo"]
    assert abs(lg["elpd_loo"] - g["elpd_waic"]) < 3.0
    assert "pareto_k_max" in lg
    w = stacking_weights({"good": ll(draws_good).mean(0), "bad": ll(draws_bad).mean(0)})
    assert w["good"] > w["bad"]


def test_bayes_factor_labels():
    a = ModelEvidence("a", 10.0, "x")
    b = ModelEvidence("b", 0.0, "x")
    assert "decisive for a" in bayes_factor(a, b)["interpretation"]
    assert bayes_factor(a, a)["interpretation"].startswith("not worth")


def test_laplace_evidence_matches_a_gaussian_integral():
    # log int exp(-(x-m)^2/(2 s^2)) dx = log(sqrt(2 pi) s)
    s = 1.7
    H = np.array([[1 / s**2]])
    got = laplace_log_evidence(0.0, H)
    assert abs(got - math.log(math.sqrt(2 * math.pi) * s)) < 1e-12


def test_compare_models_needs_two_models():
    with pytest.raises(ValueError):
        compare_models([ModelEvidence("a", 1.0, "x")])
