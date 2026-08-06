"""The anatomy-prior adapter: ``from_prior`` / ``theta_from_prior``.

These tests pin down the two mappings that are *not* identities, because both
are the kind of mistake that produces plausible-looking output:

* the backend parameter spelled ``ei_ratio`` is an **inhibitory gain**, so it
  must run opposite to the prior's excitation/inhibition ratio;
* the prior's intrinsic-autocorrelation timescale is not a synaptic constant, so
  it is applied as a modulation and bounded by each backend's calibrated support.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from scwbd.dynamics import LinearGaussian, ReducedWongWang, WilsonCowan
from scwbd.schema.priors import LogNormalPrior

BACKENDS_WITH_EI = (WilsonCowan, ReducedWongWang)
ALL_BACKENDS = (WilsonCowan, ReducedWongWang, LinearGaussian)

#: parcel 0 is the least excitable, parcel 3 the most; parcels 4-5 stand in for
#: the no-receptor-coverage branch (centred on 1.0, double width).
_EI_MU = (-0.6, -0.2, 0.2, 0.6)
_TAU_S = (0.05, 0.09, 0.15, 0.25, 0.35, 0.15)


class FakeBrainPrior:
    """Duck-typed stand-in for ``scwbd.anatomy.BrainPrior``."""

    n_parcels = 6

    def ei_ratio_prior(self):
        covered = [
            LogNormalPrior(mu=m, sigma=0.3, provenance="receptor-derived E/I proxy")
            for m in _EI_MU
        ]
        uncovered = [
            LogNormalPrior(mu=0.0, sigma=0.6, provenance="NO RECEPTOR COVERAGE")
            for _ in range(2)
        ]
        return covered + uncovered

    def timescale_prior(self):
        return [
            LogNormalPrior(mu=math.log(t), sigma=0.5, units="s", provenance="hierarchy rank")
            for t in _TAU_S
        ]

    def velocity_prior(self):
        return LogNormalPrior(mu=math.log(5.0), sigma=0.2, units="m/s", provenance="delay model")


@pytest.fixture
def prior():
    return FakeBrainPrior()


def _timescale_key(be):
    return next(k for k in be.timescale_params if k in be.defaults)


# -- shape, wiring, determinism ------------------------------------------


@pytest.mark.parametrize("cls", ALL_BACKENDS)
def test_theta_from_prior_shapes_and_regions(cls, prior):
    be = cls.from_prior(prior)
    theta = be.theta_from_prior(prior, batch=8, seed=0, device="cpu")
    assert theta.batch == 8
    assert theta.n_regions == prior.n_parcels
    assert theta.get(_timescale_key(be)).shape == (8, prior.n_parcels, 1)


@pytest.mark.parametrize("cls", ALL_BACKENDS)
def test_determinism_same_seed_same_draw(cls, prior):
    be = cls.from_prior(prior)
    a = be.theta_from_prior(prior, batch=8, seed=3, device="cpu")
    b = be.theta_from_prior(prior, batch=8, seed=3, device="cpu")
    key = _timescale_key(be)
    assert torch.equal(a.get(key), b.get(key))
    assert torch.equal(a.get("velocity"), b.get("velocity"))


@pytest.mark.parametrize("cls", ALL_BACKENDS)
def test_different_seeds_give_different_draws(cls, prior):
    be = cls.from_prior(prior)
    a = be.theta_from_prior(prior, batch=8, seed=3, device="cpu")
    b = be.theta_from_prior(prior, batch=8, seed=4, device="cpu")
    assert not torch.equal(a.get(_timescale_key(be)), b.get(_timescale_key(be)))


@pytest.mark.parametrize("cls", ALL_BACKENDS)
def test_each_batch_element_is_its_own_parameter_set(cls, prior):
    """The batch axis must carry prior spread, not one value broadcast B times.

    This is the whole point of sampling distributions rather than reading point
    estimates: a broadcast prior would make every 'parameter set' identical.
    """
    be = cls.from_prior(prior)
    theta = be.theta_from_prior(prior, batch=32, seed=0, device="cpu")
    tau = theta.get(_timescale_key(be))[:, :, 0]
    assert tau.std(dim=0).min() > 0.0, "parcels must vary across the batch"
    vel = theta.get("velocity").reshape(32, -1)[:, 0]
    assert float(vel.std()) > 0.0


# -- the E/I inversion ----------------------------------------------------


@pytest.mark.parametrize("cls", BACKENDS_WITH_EI)
def test_ei_ratio_is_inverted_relative_to_the_prior(cls, prior):
    """More excitable parcel -> LOWER backend ``ei_ratio`` (an inhibitory gain).

    Mapping the two directly because they share a name would invert the cortical
    E/I gradient end to end while still looking entirely plausible.
    """
    be = cls.from_prior(prior)
    theta = be.theta_from_prior(prior, batch=256, seed=0, device="cpu")
    gain = theta.get("ei_ratio")[:, :, 0].median(dim=0).values.numpy()
    covered = gain[: len(_EI_MU)]
    # prior mu increases with excitability, so the inhibitory gain must decrease
    assert np.all(np.diff(covered) < 0), f"expected decreasing inhibitory gain, got {covered}"
    rho = np.corrcoef(np.asarray(_EI_MU), covered)[0, 1]
    assert rho < -0.9


@pytest.mark.parametrize("cls", BACKENDS_WITH_EI)
def test_ei_ratio_centred_near_one_at_the_prior_centre(cls, prior):
    be = cls.from_prior(prior)
    theta = be.theta_from_prior(prior, batch=512, seed=0, device="cpu")
    gain = theta.get("ei_ratio")[:, :, 0]
    # geometric mean over all parcels should sit near 1.0 by construction
    gm = float(torch.exp(torch.log(gain).mean()))
    assert 0.8 < gm < 1.25, gm


# -- the timescale mapping ------------------------------------------------


@pytest.mark.parametrize("cls", ALL_BACKENDS)
def test_timescale_preserves_the_hierarchy_ranking(cls, prior):
    be = cls.from_prior(prior)
    theta = be.theta_from_prior(prior, batch=256, seed=0, device="cpu")
    tau = theta.get(_timescale_key(be))[:, :, 0].median(dim=0).values.numpy()
    # parcels 0..4 have strictly increasing prior centres
    assert np.all(np.diff(tau[:5]) > 0), tau


@pytest.mark.parametrize("cls", ALL_BACKENDS)
def test_timescale_stays_inside_the_calibrated_support(cls, prior):
    """The prior is far wider than a synaptic constant; the backend must bound it."""
    be = cls.from_prior(prior)
    key = _timescale_key(be)
    theta = be.theta_from_prior(prior, batch=256, seed=0, device="cpu")
    tau = theta.get(key)[:, :, 0]
    lo, hi = be.support_of(key)
    # tolerance is relative: the clamp happens in float32, so a bound of 0.2 is
    # representable only as 0.20000000298
    if lo is not None:
        assert float(tau.min()) >= lo * (1 - 1e-6)
    if hi is not None:
        assert float(tau.max()) <= hi * (1 + 1e-6)


@pytest.mark.parametrize("cls", ALL_BACKENDS)
def test_timescale_is_a_modulation_not_an_absolute_copy(cls, prior):
    """Prior seconds must not be written straight into a synaptic constant."""
    be = cls.from_prior(prior)
    key = _timescale_key(be)
    theta = be.theta_from_prior(prior, batch=256, seed=0, device="cpu")
    tau = theta.get(key)[:, :, 0].median(dim=0).values.numpy()
    centre = math.exp(float(np.mean(np.log(_TAU_S))))
    # the median parcel should land near the backend default, not near the prior's
    # own seconds, unless the two happen to coincide
    if abs(be.defaults[key] - centre) / centre > 0.25:
        assert abs(np.median(tau) - centre) / centre > 0.15


@pytest.mark.parametrize("cls", ALL_BACKENDS)
def test_clamping_is_reported_not_silent(cls, prior):
    be = cls.from_prior(prior)
    key = _timescale_key(be)
    theta = be.theta_from_prior(prior, batch=64, seed=0, device="cpu")
    rec = theta.provenance[key]
    assert "n_clamped" in rec and "fraction_clamped" in rec
    assert 0.0 <= rec["fraction_clamped"] <= 1.0
    assert rec["clamped_to_support"] == list(be.support_of(key))


# -- provenance -----------------------------------------------------------


@pytest.mark.parametrize("cls", BACKENDS_WITH_EI)
def test_provenance_records_source_and_transform(cls, prior):
    be = cls.from_prior(prior)
    theta = be.theta_from_prior(prior, batch=4, seed=0, device="cpu")
    rec = theta.provenance["ei_ratio"]
    assert "ei_ratio_prior" in rec["source"]
    assert "inhibitory gain" in rec["transform"]
    assert rec["sampled_per_batch_element"] is True


@pytest.mark.parametrize("cls", BACKENDS_WITH_EI)
def test_no_coverage_parcels_stay_distinguishable(cls, prior):
    """Parcels without receptor coverage must remain identifiable downstream.

    Cajal states ignorance with a wider prior rather than an imputed value; that
    distinction is worthless if the adapter flattens it, so the per-parcel
    citation index is carried through.
    """
    be = cls.from_prior(prior)
    theta = be.theta_from_prior(prior, batch=4, seed=0, device="cpu")
    rec = theta.provenance["ei_ratio"]
    idx = np.asarray(rec["parcel_provenance_index"])
    assert len(rec["distinct_provenance"]) == 2
    assert len(idx) == prior.n_parcels
    uncovered = [i for i, t in enumerate(rec["distinct_provenance"]) if "NO RECEPTOR COVERAGE" in t]
    assert uncovered, "the ignorance branch must survive as its own provenance entry"
    assert (idx == uncovered[0]).sum() == 2


@pytest.mark.parametrize("cls", ALL_BACKENDS)
def test_provenance_survives_detach_and_device_moves(cls, prior):
    be = cls.from_prior(prior)
    theta = be.theta_from_prior(prior, batch=4, seed=0, device="cpu")
    assert theta.provenance
    assert theta.detach().provenance == theta.provenance
    assert theta.to("cpu").provenance == theta.provenance
    assert theta.with_(sigma=0.01).provenance == theta.provenance


# -- graceful degradation -------------------------------------------------


def test_missing_fields_are_skipped_not_faked(prior):
    """A prior we do not have must not become a number we pretend to have."""

    class Sparse:
        n_parcels = 6

    be = WilsonCowan.from_prior(Sparse())
    theta = be.theta_from_prior(Sparse(), batch=4, seed=0, device="cpu")
    # nothing was taken from the prior, so nothing claims to have been
    assert theta.provenance == {}
    # and the parameter falls back to the backend's *own* prior (sample_theta
    # draws tau_e from Prior(0.010, 0.002)), not to an invented regional pattern
    # non-regional parameters stay (B, 1, 1) for broadcast efficiency
    tau = theta.get("tau_e").expand(4, Sparse.n_parcels, 1)[:, :, 0]
    assert float((tau - tau[:, :1]).abs().max()) == 0.0, "should be flat across regions"
    assert 0.002 < float(tau.mean()) < 0.020


def test_region_count_can_come_from_n_regions_or_weights(prior):
    class ByRegions:
        n_regions = 5

    class ByWeights:
        weights = np.zeros((7, 7))

    assert WilsonCowan._prior_n_regions(ByRegions()) == 5
    assert WilsonCowan._prior_n_regions(ByWeights()) == 7
    with pytest.raises(ValueError, match="cannot determine region count"):
        WilsonCowan._prior_n_regions(object())


def test_velocity_feeds_the_delay_model(prior):
    """``velocity`` is what ``DelayedConnectome`` needs; the adapter must supply it."""
    from scwbd.dynamics import DelayedConnectome, EdgeSet

    be = WilsonCowan.from_prior(prior)
    theta = be.theta_from_prior(prior, batch=4, seed=0, device="cpu")
    edges = EdgeSet.random(prior.n_parcels, density=0.3, seed=0, device="cpu")
    con = DelayedConnectome(edges, mode=be.coupling_kind, n_channels=be.n_coupling_channels)
    steps = con.delay_steps(theta, 1e-3)
    assert torch.isfinite(steps).all()
    assert float(steps.max()) > 0.0
