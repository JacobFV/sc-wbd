"""Structural prior: symmetry, evidence classes, delays, provenance."""

from __future__ import annotations

import numpy as np
import pytest

from scwbd.anatomy._compat import EVIDENCE_ORDER, evidence_rank
from scwbd.anatomy.connectome import (
    CONDUCTION_VELOCITY_PRIOR,
    EDR_LAMBDA_PRIOR,
    TORTUOSITY_PRIOR,
    StructuralPrior,
    load_structural_prior,
)


# ---------------------------------------------------------------------------
# shape and symmetry
# ---------------------------------------------------------------------------
def test_weights_are_symmetric_because_dmri_is_undirected(sc_small):
    """Human diffusion MRI gives no direction, so the matrix must be symmetric."""
    np.testing.assert_allclose(sc_small.weights, sc_small.weights.T, atol=1e-12)
    assert sc_small.direction_known is False


def test_hierarchy_prior_is_antisymmetric(sc_small):
    """The *direction* prior is the one object that must NOT be symmetric."""
    h = sc_small.hierarchy_prior
    np.testing.assert_allclose(h, -h.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(h), 0.0, atol=1e-12)
    assert np.abs(h).max() > 0, "hierarchy prior should not be identically zero"
    assert np.abs(h).max() <= 1.0 + 1e-9


def test_evidence_and_distance_are_symmetric(sc_small):
    np.testing.assert_array_equal(sc_small.evidence, sc_small.evidence.T)
    np.testing.assert_allclose(sc_small.distance_mm, sc_small.distance_mm.T, atol=1e-9)
    np.testing.assert_allclose(sc_small.functional, sc_small.functional.T, atol=1e-9)


def test_no_self_loops(sc_small):
    assert not np.diag(sc_small.weights).any()
    assert not np.diag(sc_small.evidence).any()


def test_no_nans_anywhere(sc_small):
    for name in ("weights", "distance_mm", "euclidean_mm", "consistency",
                 "functional", "cross_species_support", "hierarchy_prior"):
        a = getattr(sc_small, name)
        assert np.isfinite(a).all(), f"{name} contains non-finite values"


def test_weights_are_non_negative(sc_small):
    """The upstream log transform makes a few weights negative; we floor them."""
    assert (sc_small.weights >= 0).all()
    assert sc_small.provenance["n_negative_weights_floored"] >= 0


def test_labels_match_the_parcellation_plus_subcortex(sc_small, parc_small):
    assert sc_small.n_parcels == parc_small.n_parcels + 14
    np.testing.assert_array_equal(
        sc_small.labels[: parc_small.n_parcels], parc_small.labels
    )
    assert sc_small.weights.shape == (sc_small.n_parcels,) * 2


def test_density_is_sparse_not_dense(sc_small):
    d = sc_small.density()
    assert 0.01 < d < 0.5, f"implausible connectome density {d:.3f}"


# ---------------------------------------------------------------------------
# evidence classification
# ---------------------------------------------------------------------------
def test_every_class_is_a_declared_class(sc_small):
    assert set(np.unique(sc_small.evidence).tolist()) <= set(range(len(EVIDENCE_ORDER)))


def test_all_three_classes_are_populated(sc_small):
    c = sc_small.class_counts()
    assert c["hard"] > 0, "no hard edges: classification is not doing any work"
    assert c["soft"] > 0
    assert c["absent"] > 0
    assert sum(c.values()) == sc_small.n_parcels * (sc_small.n_parcels - 1) // 2


def test_hard_edges_are_a_strict_subset_of_present_edges(sc_small):
    hard = sc_small.evidence == evidence_rank("hard")
    assert (sc_small.weights[hard] > 0).all()
    assert hard.sum() < (sc_small.weights > 0).sum()


def test_hard_edges_are_more_consistent_than_soft(sc_small):
    hard = sc_small.evidence == evidence_rank("hard")
    soft = sc_small.evidence == evidence_rank("soft")
    if hard.any() and soft.any():
        assert sc_small.consistency[hard].mean() > sc_small.consistency[soft].mean()


def test_hard_edges_are_shorter_or_stronger_than_soft(sc_small):
    hard = sc_small.evidence == evidence_rank("hard")
    soft = sc_small.evidence == evidence_rank("soft")
    if hard.any() and soft.any():
        assert (
            np.median(sc_small.distance_mm[hard]) < np.median(sc_small.distance_mm[soft])
            or np.median(sc_small.weights[hard]) > np.median(sc_small.weights[soft])
        )


def test_proposed_edges_have_no_tractography_support(sc_small):
    prop = sc_small.evidence == evidence_rank("proposed")
    assert (sc_small.weights[prop] == 0).all(), (
        "a 'proposed' edge is by definition one tractography did not find"
    )


def test_subcortical_edges_cannot_be_hard(sc_small, parc_small):
    """We have exactly one observation of them, so `hard` is unreachable."""
    n_ctx = parc_small.n_parcels
    sub = np.zeros(sc_small.n_parcels, dtype=bool)
    sub[n_ctx:] = True
    block = np.outer(sub, np.ones(sc_small.n_parcels, dtype=bool))
    assert (sc_small.evidence[block] != evidence_rank("hard")).all()


def test_mask_is_monotone_in_evidence_class(sc_small):
    hard = sc_small.mask("hard")
    soft = sc_small.mask("soft")
    prop = sc_small.mask("proposed")
    assert hard.sum() <= soft.sum() <= prop.sum()
    assert (soft | hard == soft).all()


def test_edge_evidence_explains_itself(sc_small):
    iu = np.triu_indices(sc_small.n_parcels, 1)
    hard = np.flatnonzero(sc_small.evidence[iu] == evidence_rank("hard"))
    i, j = iu[0][hard[0]], iu[1][hard[0]]
    e = sc_small.edge_evidence(int(i), int(j))
    assert e.evidence_class == "hard"
    assert e.weight > 0
    assert len(e.reasons) >= 3
    joined = " ".join(e.reasons).lower()
    assert "direction is unknown" in joined
    assert "cross-species" in joined
    assert "functional correlation is not structural" in joined


def test_edge_evidence_for_an_absent_edge_says_absence_is_not_independence(sc_small):
    iu = np.triu_indices(sc_small.n_parcels, 1)
    absent = np.flatnonzero(sc_small.evidence[iu] == evidence_rank("absent"))
    i, j = iu[0][absent[0]], iu[1][absent[0]]
    e = sc_small.edge_evidence(int(i), int(j))
    assert e.evidence_class == "absent"
    assert any("conditional independence" in r for r in e.reasons)


def test_edge_evidence_rejects_self_edges(sc_small):
    with pytest.raises(ValueError):
        sc_small.edge_evidence(0, 0)


# ---------------------------------------------------------------------------
# delays
# ---------------------------------------------------------------------------
def test_velocity_prior_is_a_distribution_not_a_point():
    assert CONDUCTION_VELOCITY_PRIOR.kind == "lognormal"
    assert CONDUCTION_VELOCITY_PRIOR.sigma > 0.1
    assert CONDUCTION_VELOCITY_PRIOR.units == "m/s"
    assert "Caminiti" in CONDUCTION_VELOCITY_PRIOR.provenance
    lo, hi = CONDUCTION_VELOCITY_PRIOR.support()
    assert lo == 0.0


def test_velocity_prior_spans_the_literature_range():
    s = np.asarray(CONDUCTION_VELOCITY_PRIOR.sample(0, 200_000))
    q = np.quantile(s, [0.025, 0.5, 0.975])
    assert 1.5 < q[0] < 3.0
    assert 5.0 < q[1] < 7.0
    assert 14.0 < q[2] < 22.0


def test_tortuosity_and_edr_priors_are_cited():
    for p in (TORTUOSITY_PRIOR, EDR_LAMBDA_PRIOR):
        assert p.provenance
        assert p.sigma > 0
    assert "CROSS-SPECIES" in EDR_LAMBDA_PRIOR.provenance


def test_median_delays_are_positive_and_physiological(sc_small):
    d_ms = sc_small.delays.median_delay_s() * 1e3
    m = sc_small.mask("soft") & ~np.eye(sc_small.n_parcels, dtype=bool)
    v = d_ms[m]
    assert (v > 0).all(), "a conduction delay is strictly positive"
    assert v.min() < 5.0, "the shortest connections should be a few milliseconds"
    assert v.max() < 50.0, f"longest median delay {v.max():.1f} ms is out of range"
    assert 3.0 < np.median(v) < 30.0


def test_delay_matrix_is_symmetric_and_zero_on_the_diagonal(sc_small):
    d = sc_small.delays.median_delay_s()
    np.testing.assert_allclose(d, d.T, atol=1e-15)
    np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-15)


def test_delay_sampling_is_deterministic_given_a_seed(sc_small):
    a = sc_small.delays.sample_delay_s(7, 3)
    b = sc_small.delays.sample_delay_s(7, 3)
    np.testing.assert_array_equal(a, b)
    c = sc_small.delays.sample_delay_s(8, 3)
    assert not np.allclose(a, c)


def test_sampled_delays_vary_and_stay_positive(sc_small):
    s = sc_small.delays.sample_delay_s(0, 64)
    assert s.shape == (64,) + sc_small.distance_mm.shape
    assert (s >= 0).all()
    m = sc_small.mask("soft")
    spread = s[:, m].std(axis=0) / np.maximum(s[:, m].mean(axis=0), 1e-12)
    assert spread.mean() > 0.2, "delays must carry their uncertainty, not a point value"


def test_per_edge_sampling_keeps_the_delay_matrix_symmetric(sc_small):
    s = sc_small.delays.sample_delay_s(3, 2, per_edge=True)
    np.testing.assert_allclose(s, np.swapaxes(s, 1, 2), atol=1e-12)


# ---------------------------------------------------------------------------
# provenance and ledger
# ---------------------------------------------------------------------------
def test_connectome_ledger_satisfies_r08_and_names_the_forbidden_inference(sc_small):
    led = sc_small.ledger
    assert led.has_estimator()
    assert led.bias_status == "prior_specified_sensitivity"
    assert led.bias_interval[0] < led.bias_interval[1], "a swept range, not a point"
    fi = led.validity_domain["forbidden_inference"]
    assert "direction" in fi.lower()
    assert led.validity_domain["direction_known"] is False
    assert led.validity_domain["laminar_resolved"] is False


def test_provenance_names_cohort_pipeline_and_thresholding(sc_small):
    src = sc_small.provenance["source"]
    assert "MRtrix3" in src["pipeline"]
    assert "SIFT2" in src["pipeline"]
    assert "streamline" in src["bias"].lower()
    assert src["license"]
    assert sc_small.provenance["classification"]["long_mm"] > 0


def test_weights_units_admit_they_are_arbitrary(sc_small):
    assert "arbitrary" in sc_small.weights_units.lower()


def test_independent_streams_are_recorded_with_their_independence(sc_dk):
    """Desikan-Killiany has a genuinely different cohort available."""
    assert sc_dk.n_streams >= 1
    names = " ".join(sc_dk.stream_names)
    assert "lausanne" in names.lower()
    for st in sc_dk.provenance["streams"]:
        assert 0 < st["independence"] <= 1
        assert st["note"]


def test_refuses_a_parcellation_with_no_measured_connectome():
    with pytest.raises(ValueError, match="Refusing to synthesise"):
        load_structural_prior("Destrieux")
    with pytest.raises(ValueError, match="Refusing to synthesise"):
        load_structural_prior("Buckner17")


def test_roundtrip(tmp_path, sc_small):
    p = tmp_path / "sc.npz"
    sc_small.save(p)
    q = StructuralPrior.load(p)
    np.testing.assert_allclose(q.weights, sc_small.weights)
    np.testing.assert_array_equal(q.evidence, sc_small.evidence)
    assert q.stream_names == sc_small.stream_names
    assert q.delays.velocity_prior.mu == sc_small.delays.velocity_prior.mu
