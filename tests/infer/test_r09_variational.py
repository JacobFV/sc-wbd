"""Refusal **R09**: a pseudo-loss-derived posterior may never be reported as a
calibrated Bayesian posterior."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scwbd.infer.types import CalibrationClaimError, PosteriorSummary
from scwbd.infer.variational import (
    GaussianFactor,
    LossFactor,
    SlicedObjective,
    gaussian_kl,
    generalized_posterior_from,
)


def _clean() -> SlicedObjective:
    o = SlicedObjective()
    o.add_likelihood("eeg_native", torch.tensor(12.0), normalized=True, evaluable=True)
    o.add_kl("boundary_state", torch.tensor(3.0))
    return o


def test_clean_objective_is_a_negative_elbo():
    v = _clean().evaluate()
    assert float(v.negative_elbo) == 15.0
    assert v.posterior_class == "bayesian"
    assert v.report()["is_negative_elbo"] is True


def test_pseudo_loss_blocks_the_elbo_claim():
    o = _clean()
    o.add_pseudo("scale_compatibility", torch.tensor(2.0), "scale")
    v = o.evaluate()
    assert v.posterior_class == "generalized"
    assert float(v.total) == 17.0            # still optimisable
    assert float(v.elbo_part) == 15.0        # still reportable, separately
    with pytest.raises(CalibrationClaimError, match=r"\[R09\]"):
        _ = v.negative_elbo
    # the pseudo terms are reported separately, not merged
    r = v.report()
    assert set(r["pseudo_terms"]) == {"scale_compatibility"}
    assert r["pseudo_roles"]["scale_compatibility"] == "scale"


def test_unnormalised_likelihood_blocks_the_elbo_claim():
    o = SlicedObjective()
    o.add_likelihood("distilled_teacher", torch.tensor(4.0),
                     normalized=False, evaluable=False)
    o.add_kl("prior", torch.tensor(1.0))
    v = o.evaluate()
    assert "distilled_teacher" in v.non_generative_factors
    with pytest.raises(CalibrationClaimError, match="normalized and evaluable"):
        _ = v.negative_elbo


def test_generalized_posterior_refuses_calibration_semantics():
    o = _clean()
    o.add_pseudo("boundary_agreement", torch.tensor(1.0), "boundary_target")
    v = o.evaluate()
    post = generalized_posterior_from(v, ["a21"], np.array([30.0]), np.array([[4.0]]))
    assert post.kind == "generalized_posterior"
    assert post.provenance["posterior_class"] == "generalized"
    with pytest.raises(CalibrationClaimError, match=r"\[R09\]"):
        post.assert_calibrated_bayesian("interval coverage report")
    # the clean one does not raise
    clean = generalized_posterior_from(_clean().evaluate(), ["a21"],
                                       np.array([30.0]), np.array([[4.0]]))
    assert clean.kind == "bayesian"
    clean.assert_calibrated_bayesian()


def test_posterior_summary_rejects_mislabelled_construction():
    with pytest.raises(CalibrationClaimError):
        PosteriorSummary(["a"], np.zeros(1), np.eye(1), kind="bayesian",
                         pseudo_loss_terms=["compatibility"])


def test_pseudo_role_must_be_declared():
    o = SlicedObjective()
    with pytest.raises(ValueError, match="not one of the auxiliary roles"):
        o.add_pseudo("sneaky", torch.tensor(1.0), "likelihood")


def test_objective_requires_both_terms():
    o = SlicedObjective()
    o.add_likelihood("only", torch.tensor(1.0))
    with pytest.raises(ValueError, match="likelihood term and a KL term"):
        o.evaluate()


def test_gaussian_kl_matches_closed_form():
    q = GaussianFactor(torch.tensor([0.5, -1.0]), torch.log(torch.tensor([2.0, 0.5])))
    p = GaussianFactor(torch.tensor([0.0, 0.0]), torch.log(torch.tensor([1.0, 1.0])))
    got = float(gaussian_kl(q, p))
    want = 0.0
    for m, s in ((0.5, 2.0), (-1.0, 0.5)):
        want += 0.5 * (s**2 + m**2 - 1 - 2 * np.log(s))
    assert abs(got - want) < 1e-10
