"""``BrainPrior``: assembly, regional heterogeneity, and refusals."""

from __future__ import annotations

import numpy as np
import pytest

from scwbd.anatomy.priors import EI_LOG_RANGE, TIMESCALE_RANGE_MS, BrainPrior


def test_loads_with_cortex_and_subcortex(brain_prior, parc_small):
    assert brain_prior.n_parcels == parc_small.n_parcels + 14
    assert brain_prior.n_cortex == parc_small.n_parcels
    assert int((brain_prior.structure == "subcortex").sum()) == 14
    assert brain_prior.labels.shape == (brain_prior.n_parcels,)


def test_all_per_parcel_arrays_agree_in_length(brain_prior):
    n = brain_prior.n_parcels
    for name in ("labels", "hemi", "network", "structure", "areas_mm2", "volumes_mm3"):
        assert getattr(brain_prior, name).shape == (n,), name
    assert brain_prior.centroids_mni.shape == (n, 3)
    assert np.isfinite(brain_prior.centroids_mni).all()


def test_coupling_mask_is_symmetric_and_sparse(brain_prior):
    m = brain_prior.coupling_mask("soft")
    np.testing.assert_array_equal(m, m.T)
    assert not np.diag(m).any()
    assert 0 < m.mean() < 0.5


def test_coupling_mask_is_monotone_in_class(brain_prior):
    assert (
        brain_prior.coupling_mask("hard").sum()
        <= brain_prior.coupling_mask("soft").sum()
        <= brain_prior.coupling_mask("proposed").sum()
    )


# ---------------------------------------------------------------------------
# regional heterogeneity -- the thesis's named failure mode
# ---------------------------------------------------------------------------
def test_ei_prior_is_one_distribution_per_parcel(brain_prior):
    priors = brain_prior.ei_ratio_prior()
    assert len(priors) == brain_prior.n_parcels
    for p in priors:
        assert p.kind == "lognormal"
        assert p.units == "dimensionless"
        assert p.provenance


def test_ei_priors_actually_differ_across_parcels(brain_prior):
    """One identical neural mass per parcel is the failure mode to avoid."""
    mus = np.array([p.mu for p in brain_prior.ei_ratio_prior()])
    ctx = brain_prior.structure == "cortex"
    assert mus[ctx].std() > 0.05, "cortical E/I priors are effectively identical"
    assert len(set(np.round(mus[ctx], 6).tolist())) > 0.9 * ctx.sum()


def test_ei_prior_span_is_the_declared_modelling_range(brain_prior):
    mus = np.array([p.mu for p in brain_prior.ei_ratio_prior()])
    ctx = brain_prior.structure == "cortex"
    assert np.abs(mus[ctx]).max() <= 3.0 * EI_LOG_RANGE + 1e-9
    ratios = np.exp(mus[ctx])
    assert 1.2 < ratios.max() / ratios.min() < 12.0


def test_parcels_without_ordering_coverage_get_a_wider_prior_not_a_made_up_value(brain_prior):
    """Renamed from ...without_receptor_coverage... on 2026-08-06.

    The E/I prior no longer defaults to a receptor map, so the wording is now
    about the ordering; the invariant is unchanged and so is the branch it
    guards. See ``reports/ei_ordering_substitution.md``.
    """
    priors = brain_prior.ei_ratio_prior()
    sub = np.flatnonzero(brain_prior.structure == "subcortex")
    ctx = np.flatnonzero(brain_prior.structure == "cortex")
    assert priors[sub[0]].sigma > priors[ctx[0]].sigma
    assert priors[sub[0]].mu == 0.0
    assert "NO ORDERING COVERAGE" in priors[sub[0]].provenance


def test_timescale_prior_is_one_distribution_per_parcel(brain_prior):
    priors = brain_prior.timescale_prior()
    assert len(priors) == brain_prior.n_parcels
    for p in priors:
        assert p.kind == "lognormal"
        assert p.units == "s"


def test_timescale_priors_span_the_literature_range(brain_prior):
    priors = brain_prior.timescale_prior()
    ctx = brain_prior.structure == "cortex"
    med_ms = np.array([np.exp(p.mu) for p in priors])[ctx] * 1e3
    lo, hi = TIMESCALE_RANGE_MS
    assert med_ms.min() == pytest.approx(lo, rel=0.05)
    assert med_ms.max() == pytest.approx(hi, rel=0.05)
    assert med_ms.std() > 0.1 * med_ms.mean()


def test_timescale_priors_are_positive_and_sampleable(brain_prior):
    priors = brain_prior.timescale_prior()
    s = np.array([p.sample(0) for p in priors])
    assert (s > 0).all()
    assert (s < 10.0).all(), "a cortical intrinsic timescale of >10 s is not plausible"


def test_timescale_prior_follows_the_hierarchy(brain_prior):
    """Transmodal parcels should get slower priors than unimodal ones."""
    from scipy import stats

    rank = brain_prior.hierarchy_rank()
    mus = np.array([p.mu for p in brain_prior.timescale_prior()])
    m = np.isfinite(rank)
    rho = stats.spearmanr(rank[m], mus[m]).statistic
    assert abs(rho) > 0.5


def test_receptor_profile_shape_and_nan_policy(brain_prior):
    v, names = brain_prior.receptor_profile()
    assert v.shape == (brain_prior.n_parcels, len(names))
    ctx = brain_prior.structure == "cortex"
    assert np.isfinite(v[ctx]).all()
    assert np.isnan(v[~ctx]).all(), "subcortical receptor values are nan, not imputed"


# ---------------------------------------------------------------------------
# delays
# ---------------------------------------------------------------------------
def test_median_delays_are_physiological(brain_prior):
    d = brain_prior.median_delay_ms()
    m = brain_prior.coupling_mask("soft")
    v = d[m]
    assert (v > 0).all()
    assert v.min() < 5.0 and v.max() < 50.0


def test_velocity_prior_is_exposed_as_a_distribution(brain_prior):
    p = brain_prior.velocity_prior()
    assert p.kind == "lognormal"
    assert p.units == "m/s"
    s = np.asarray(p.sample(0, 1000))
    assert s.std() / s.mean() > 0.2


# ---------------------------------------------------------------------------
# G2
# ---------------------------------------------------------------------------
def test_controls_are_reachable_from_the_prior(brain_prior):
    c = brain_prior.controls(seed=0)
    assert sorted(c) == ["dense", "distance_matched", "graph_only", "local_only", "randomized"]
    for k, v in c.items():
        assert v.control_kind == k


# ---------------------------------------------------------------------------
# honesty
# ---------------------------------------------------------------------------
def test_declares_what_it_cannot_support(brain_prior):
    w = brain_prior.what_this_cannot_support()
    for key in ("direction", "laminar_termination", "subject_specificity",
                "receptor_density", "zero_edge_semantics", "weight_scale"):
        assert key in w and len(w[key]) > 40
    assert "Appendix A" in w["receptor_density"]
    assert "undirected" in w["direction"]


def test_ledger_summary_covers_every_object(brain_prior):
    s = brain_prior.ledger_summary()
    assert "structural_connectome" in s
    assert "geometry" in s
    assert sum(k.startswith("map.") for k in s) >= 10
    for k, led in s.items():
        assert led["bias_status"] in (
            "design_estimable", "externally_bounded", "prior_specified_sensitivity"
        )


def test_summary_is_machine_readable(brain_prior):
    s = brain_prior.summary()
    assert s["n_parcels"] == brain_prior.n_parcels
    assert s["direction_known"] is False
    assert s["edge_classes"]["hard"] > 0
    assert 0 < s["median_delay_ms"]["median"] < 50


# ---------------------------------------------------------------------------
# cerebellum: present as parcels, absent as connectivity
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def prior_with_cerebellum():
    return BrainPrior.load(
        "Schaefer100x7", include_subcortex=True, include_cerebellum=True,
        cerebellar_atlas="Buckner17",
    )


def test_cerebellum_adds_parcels(prior_with_cerebellum, brain_prior):
    assert prior_with_cerebellum.n_parcels == brain_prior.n_parcels + 17
    assert int((prior_with_cerebellum.structure == "cerebellum").sum()) == 17


def test_cerebellar_edges_are_absent_and_the_reason_is_recorded(prior_with_cerebellum):
    p = prior_with_cerebellum
    cb = p.structure == "cerebellum"
    assert not p.structural.weights[cb].any(), "we do not invent a cerebellar connectome"
    assert (p.structural.evidence[cb] == 0).all()
    r = p.unresolved["cerebellar_structural_connectivity"]
    assert "polysynaptic" in r and "'no evidence'" in r


def test_cerebellar_distances_are_still_finite(prior_with_cerebellum):
    """No connectivity does not mean no geometry."""
    p = prior_with_cerebellum
    assert np.isfinite(p.structural.distance_mm).all()
    cb = p.structure == "cerebellum"
    assert (p.structural.distance_mm[cb][:, ~cb] > 0).all()


def test_excluding_subcortex_is_recorded_as_a_limitation():
    p = BrainPrior.load("Schaefer100x7", include_subcortex=False)
    assert p.n_parcels == 100
    assert "subcortex_excluded" in p.unresolved
    assert "thalamic" in p.unresolved["subcortex_excluded"].lower()
