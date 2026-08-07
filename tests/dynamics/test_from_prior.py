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

from scwbd.dynamics import (
    JansenRit,
    Kuramoto,
    LinearGaussian,
    ReducedWongWang,
    StuartLandau,
    WilsonCowan,
)
from scwbd.dynamics.base import map_fragility
from scwbd.schema.priors import LogNormalPrior

#: Mirrors agent C's ei_proxy ledger: the two markers the E/I contrast is built
#: from are the two least route-stable maps in the receptor panel.
_ROUTE_FRAGILE = {"NMDA": 0.590, "GABAa": 0.685}
_FORBIDDEN = (
    "Not a measurement of excitation/inhibition balance. NMDA (route r=0.59), "
    "GABAa (route r=0.69) disagree between the surface-sampling and volumetric-join "
    "routes, so the sign of this contrast in any given parcel is not robust."
)

BACKENDS_WITH_EI = (WilsonCowan, ReducedWongWang)
ALL_BACKENDS = (WilsonCowan, ReducedWongWang, LinearGaussian)

#: parcel 0 is the least excitable, parcel 3 the most; parcels 4-5 stand in for
#: the no-receptor-coverage branch (centred on 1.0, double width).
_EI_MU = (-0.6, -0.2, 0.2, 0.6)
_TAU_S = (0.05, 0.09, 0.15, 0.25, 0.35, 0.15)


class _Ledger:
    forbidden_inference = _FORBIDDEN
    validity_domain = {
        "route_fragile_ingredients": _ROUTE_FRAGILE,
        "interpretation": "relative, rank-meaningful only; zero is the cortical mean",
    }


class _Map:
    ledger = _Ledger()


class FakeBrainPrior:
    """Duck-typed stand-in for ``scwbd.anatomy.BrainPrior``."""

    n_parcels = 6
    maps = {"ei_proxy": _Map()}

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


@pytest.mark.parametrize("cls", BACKENDS_WITH_EI)
def test_route_fragility_of_the_ei_maps_is_propagated(cls, prior):
    """NMDA/GABA-A are route-fragile and are exactly what the E/I contrast uses.

    The ledger says the per-parcel *sign* of the contrast is not robust to a
    defensible change in how the PET volume is read.  A regional E/I pattern that
    might flip sign must not reach the training corpus looking like a
    measurement, so the disclosure travels with the parameter.
    """
    be = cls.from_prior(prior)
    theta = be.theta_from_prior(prior, batch=4, seed=0, device="cpu")
    frag = theta.provenance["ei_ratio"]["route_fragility"]
    assert frag["route_fragile_ingredients"] == _ROUTE_FRAGILE
    assert "not robust" in frag["forbidden_inference"]


class _PlainMap:
    """A map with a ledger that carries no route-fragility entry.

    Native surface annotations are like this: the route check is defined for
    PET volumes only, so there is no such number to carry.
    """

    class ledger:  # noqa: N801
        forbidden_inference = "T1w/T2w is a myelin proxy, not a myelin concentration."
        validity_domain = {"interpretation": "relative, rank-meaningful only"}


class FakeBrainPriorWithOrdering(FakeBrainPrior):
    """A prior that exposes ``ei_ordering`` -- i.e. the shape shipped since
    2026-08-06. The suite previously only had ``FakeBrainPrior``, which does
    not, so every fragility test ran down the legacy branch and the new one was
    unexercised (``reports/decorative_guards.md``: verifying through a different
    path than production uses).
    """

    maps = {"ei_proxy": _Map(), "myelin_t1t2": _PlainMap(), "cortical_thickness": _PlainMap()}

    def ei_ordering(self, source=None):
        rec = {
            "ordering": "hcp_hierarchy",
            "maps_used": [
                {"map": "myelin_t1t2", "source_key": "hcps1200_maps"},
                {"map": "cortical_thickness", "source_key": "hcps1200_maps"},
            ],
            "licence_keys": ["hcps1200_maps"],
            "degraded": False,
            "missing": {},
        }
        return None, rec


@pytest.mark.parametrize("cls", BACKENDS_WITH_EI)
def test_disclosure_follows_the_chosen_ordering_not_a_hardcoded_map(cls):
    """The fragility disclosed must be that of the maps actually read.

    Watched fail by restoring ``map_fragility(brain_prior, "ei_proxy")``: the
    record then reports NMDA/GABA-A route fragility for a prior built from
    myelin and thickness -- true of a map, and of no map this prior read.
    """
    bp = FakeBrainPriorWithOrdering()
    be = cls.from_prior(bp)
    theta = be.theta_from_prior(bp, batch=4, seed=0, device="cpu")
    rec = theta.provenance["ei_ratio"]
    assert rec["ei_ordering"]["disclosed"] is True
    assert rec["ei_ordering"]["ordering"] == "hcp_hierarchy"
    assert rec["ei_ordering"]["maps_used"] == ["myelin_t1t2", "cortical_thickness"]
    assert rec["ei_ordering"]["licence_keys"] == ["hcps1200_maps"]
    frag = rec["route_fragility"]
    assert "route_fragile_ingredients" not in frag, (
        "the receptor contrast's fragility must not be attributed to maps that "
        "were not read"
    )
    assert frag["disclosed"] is False
    assert "unmeasured, not clean" in frag["note"]
    # the maps' own forbidden_inference strings are kept, just not filed as
    # route fragility
    assert "myelin_t1t2" in rec["ei_ordering"]["forbidden_inference"]


def test_a_degraded_ordering_is_visible_in_the_parameter_record():
    """A prior built from two of three declared maps must not look complete."""

    class Degraded(FakeBrainPriorWithOrdering):
        def ei_ordering(self, source=None):
            _, rec = FakeBrainPriorWithOrdering.ei_ordering(self, source)
            rec["degraded"] = True
            rec["missing"] = {"intrinsic_timescale_meg": "removed by a test"}
            return None, rec

    bp = Degraded()
    theta = WilsonCowan.from_prior(bp).theta_from_prior(bp, batch=2, seed=0, device="cpu")
    rec = theta.provenance["ei_ratio"]["ei_ordering"]
    assert rec["degraded"] is True
    assert "intrinsic_timescale_meg" in rec["missing"]


def test_absent_ledger_is_reported_as_undisclosed_not_as_safe(prior):
    """Silence about fragility must not read as a clean bill of health."""

    class NoLedger(FakeBrainPrior):
        maps: dict = {}

    be = WilsonCowan.from_prior(NoLedger())
    theta = be.theta_from_prior(NoLedger(), batch=4, seed=0, device="cpu")
    frag = theta.provenance["ei_ratio"]["route_fragility"]
    assert frag["disclosed"] is False
    assert "no ledger" in frag["note"]


def test_map_fragility_is_defensive_about_shapes():
    assert map_fragility(object(), "ei_proxy") == {}

    class Weird:
        maps = {"ei_proxy": object()}

    assert map_fragility(Weird(), "ei_proxy") == {}
    assert map_fragility(FakeBrainPrior(), "no_such_map") == {}


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
    # Nothing was taken from the prior, so nothing claims to have been -- but the
    # absence is *stated*, not left as an empty dict.  An empty provenance is
    # exactly the failure this module had: indistinguishable from "applied and
    # fitted perfectly".
    assert all(v["applied"] is False for v in theta.provenance.values())
    assert theta.provenance["ei_ratio"]["reason"] == "prior exposes no ei_ratio_prior"
    assert theta.provenance["timescale"]["reason"] == "prior exposes no timescale_prior"
    assert "velocity" not in theta.provenance
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


# -- the prior must never be skipped silently -----------------------------
#
# Regression tests for a real defect: `timescale_params` listed only
# ("tau_E", "tau_e", "tau_s", "tau"), so Jansen-Rit (rate constant `a`) and
# Stuart-Landau / Kuramoto (frequency `f`) matched nothing and the whole block
# was skipped **writing no provenance at all**.  Their 0% clamp rate read as
# "the prior fitted" when it actually meant "the prior never arrived", and
# ~21.6% of the training corpus was anatomically flat without disclosing it.

EVERY_BACKEND = (WilsonCowan, ReducedWongWang, JansenRit, StuartLandau, Kuramoto, LinearGaussian)
#: backends that genuinely have no parameter the timescale prior maps onto
NO_TIMESCALE = (StuartLandau, Kuramoto)


@pytest.mark.parametrize("cls", EVERY_BACKEND)
def test_every_backend_records_a_timescale_verdict(cls, prior):
    """Applied or not, there must be a record. Absence of evidence is the bug."""
    be = cls.from_prior(prior)
    theta = be.theta_from_prior(prior, batch=8, seed=0, device="cpu")
    ts = [v for k, v in theta.provenance.items()
          if isinstance(v, dict) and k not in ("ei_ratio", "velocity")]
    assert ts, f"{cls.__name__} wrote no timescale provenance at all"
    rec = ts[0]
    assert "applied" in rec
    if not rec["applied"]:
        assert rec["reason"], "a refusal must carry a reason"
        assert "consequence" in rec


@pytest.mark.parametrize("cls", EVERY_BACKEND)
def test_every_backend_records_an_ei_verdict(cls, prior):
    be = cls.from_prior(prior)
    theta = be.theta_from_prior(prior, batch=8, seed=0, device="cpu")
    rec = theta.provenance["ei_ratio"]
    assert "applied" in rec
    if not rec["applied"]:
        assert rec["reason"]


@pytest.mark.parametrize("cls", NO_TIMESCALE)
def test_unmapped_backends_say_why_rather_than_going_quiet(cls, prior):
    be = cls.from_prior(prior)
    theta = be.theta_from_prior(prior, batch=8, seed=0, device="cpu")
    rec = theta.provenance["timescale"]
    assert rec["applied"] is False
    # the reason must be the backend's own considered one, not the generic fallback
    assert rec["reason"] == cls.timescale_not_mapped_reason
    assert "never arrived" in rec["consequence"]


def test_jansen_rit_maps_its_rate_constant_reciprocally(prior):
    """`a` is a rate: 1/a is the PSP time constant, so slower parcels get SMALLER a.

    Applying the direct factor to a 1/s parameter would install the anatomical
    hierarchy backwards while looking perfectly healthy.
    """
    be = JansenRit.from_prior(prior)
    theta = be.theta_from_prior(prior, batch=64, seed=0, device="cpu")
    rec = theta.provenance["a"]
    assert rec["applied"] is True and rec["inverse_timescale"] is True
    a = theta.get("a")[:, :, 0].median(dim=0).values.numpy()
    # parcels 0..4 have increasing prior timescales -> a must DECREASE
    assert np.all(np.diff(a[:5]) < 0), a
    # and 1/a must land in the same band tau_e occupies
    lo, hi = be.support_of("a")
    assert lo is not None and hi is not None
    assert float(a.min()) >= lo * (1 - 1e-6) and float(a.max()) <= hi * (1 + 1e-6)


def test_stuart_landau_bifurcation_parameter_is_never_scaled(prior):
    """`a` changes sign; scaling it would move parcels across the Hopf bifurcation."""
    be = StuartLandau.from_prior(prior)
    base = be.sample_theta(16, prior.n_parcels, seed=0, device="cpu")
    theta = be.theta_from_prior(prior, batch=16, seed=0, device="cpu")
    assert torch.equal(base.get("a"), theta.get("a"))
    assert torch.equal(base.get("f"), theta.get("f"))
