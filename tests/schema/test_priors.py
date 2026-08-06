"""Priors: deterministic sampling and analytic log densities.

ARCHITECTURE.md sec. 3: "All stochastic entry points take an explicit seed.
Determinism is a test, not an aspiration."
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scwbd.schema import (
    BetaPrior,
    DiracPrior,
    GammaPrior,
    LogNormalPrior,
    NormalPrior,
    UniformPrior,
    Unit,
    as_prior,
)

ALL = [
    NormalPrior(loc=0.5, scale=2.0),
    LogNormalPrior(mu=-1.0, sigma=0.5),
    UniformPrior(low=-1.0, high=3.0),
    BetaPrior(alpha=2.0, beta=5.0),
    GammaPrior(shape=3.0, rate=2.0),
    DiracPrior(value=0.25),
]


@pytest.mark.parametrize("prior", ALL, ids=lambda p: p.kind)
def test_sampling_is_deterministic_in_the_seed(prior):
    a = np.asarray(prior.sample(1234, 16))
    b = np.asarray(prior.sample(1234, 16))
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("prior", ALL, ids=lambda p: p.kind)
def test_different_seeds_differ(prior):
    if prior.kind == "dirac":
        pytest.skip("a Dirac prior is the same everywhere by construction")
    a = np.asarray(prior.sample(1, 32))
    b = np.asarray(prior.sample(2, 32))
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("prior", ALL, ids=lambda p: p.kind)
def test_seed_must_be_an_int(prior):
    with pytest.raises(TypeError, match="seed"):
        prior.sample("not a seed")  # type: ignore[arg-type]


@pytest.mark.parametrize("prior", ALL, ids=lambda p: p.kind)
def test_samples_lie_in_support(prior):
    lo, hi = prior.support()
    xs = np.atleast_1d(np.asarray(prior.sample(7, 256), dtype=float))
    assert np.all(xs >= lo - 1e-12) and np.all(xs <= hi + 1e-12)


def test_normal_logpdf_matches_closed_form():
    p = NormalPrior(loc=1.0, scale=2.0)
    x = 1.5
    expected = -0.5 * ((x - 1.0) / 2.0) ** 2 - math.log(2.0) - 0.5 * math.log(2 * math.pi)
    assert p.logpdf(x) == pytest.approx(expected)


def test_logpdf_integrates_to_one():
    """Numerically integrate each density over its support."""
    cases = [
        (NormalPrior(loc=0.0, scale=1.0), -8.0, 8.0),
        (LogNormalPrior(mu=0.0, sigma=0.6), 1e-6, 40.0),
        (UniformPrior(low=-2.0, high=5.0), -3.0, 6.0),
        (BetaPrior(alpha=2.0, beta=3.0), 1e-9, 1 - 1e-9),
        (GammaPrior(shape=2.5, rate=1.3), 1e-9, 40.0),
    ]
    for prior, lo, hi in cases:
        xs = np.linspace(lo, hi, 200_001)
        density = np.exp(prior.logpdf(xs))
        mass = np.trapezoid(density, xs)
        assert mass == pytest.approx(1.0, abs=2e-3), prior.kind


def test_logpdf_is_minus_inf_outside_support():
    assert LogNormalPrior(mu=0.0, sigma=1.0).logpdf(-1.0) == -math.inf
    assert UniformPrior(low=0.0, high=1.0).logpdf(2.0) == -math.inf
    assert BetaPrior(alpha=2.0, beta=2.0).logpdf(1.5) == -math.inf
    assert GammaPrior(shape=1.0, rate=1.0).logpdf(-0.5) == -math.inf


def test_means_are_right():
    assert NormalPrior(loc=3.0, scale=1.0).mean() == 3.0
    assert UniformPrior(low=0.0, high=4.0).mean() == 2.0
    assert BetaPrior(alpha=2.0, beta=6.0).mean() == pytest.approx(0.25)
    assert GammaPrior(shape=4.0, rate=2.0).mean() == pytest.approx(2.0)
    assert LogNormalPrior(mu=0.0, sigma=1.0).mean() == pytest.approx(math.exp(0.5))
    assert DiracPrior(value=7.0).mean() == 7.0


def test_dirac_is_a_declaration_not_an_estimate():
    p = DiracPrior(value=2.0)
    assert p.sample(0) == 2.0
    assert p.logpdf(2.0) == 0.0
    assert p.logpdf(2.0001) == -math.inf
    assert p.support() == (2.0, 2.0)


def test_priors_carry_units():
    p = LogNormalPrior(mu=-4.6, sigma=0.5, units=Unit("s"))
    assert p.units.same_dimension("s")
    assert p.model_dump(mode="json")["units"] == "s"


def test_invalid_parameters_are_refused():
    with pytest.raises(ValueError):
        NormalPrior(loc=0.0, scale=0.0)
    with pytest.raises(ValueError):
        UniformPrior(low=1.0, high=1.0)
    with pytest.raises(ValueError):
        BetaPrior(alpha=-1.0, beta=1.0)
    with pytest.raises(ValueError):
        GammaPrior(shape=1.0, rate=0.0)


def test_as_prior_round_trips_through_dicts():
    original = GammaPrior(shape=2.0, rate=3.0, units=Unit("Hz"))
    revived = as_prior(original.model_dump(mode="json"))
    assert revived == original
    assert revived.content_hash() == original.content_hash()
