"""G2 controls: each null must preserve exactly what it claims to preserve.

Gate G2 asks whether anatomy improves inference.  The answer is only meaningful
if the baselines are honest, which means each control must destroy the thing it
is meant to destroy and preserve the thing it is meant to preserve.  These
tests are therefore as load-bearing as the connectome itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from scwbd.anatomy._compat import evidence_rank

CONTROL_NAMES = ["randomized", "distance_matched", "dense", "local_only", "graph_only"]


# ---------------------------------------------------------------------------
# shared invariants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", CONTROL_NAMES)
def test_controls_are_symmetric_loopless_and_finite(controls, name):
    c = controls[name]
    np.testing.assert_allclose(c.weights, c.weights.T, atol=1e-12)
    assert not np.diag(c.weights).any()
    assert np.isfinite(c.weights).all()
    assert (c.weights >= 0).all()


@pytest.mark.parametrize("name", CONTROL_NAMES)
def test_controls_declare_themselves_and_carry_no_anatomical_evidence(controls, name):
    c = controls[name]
    assert c.control_kind == name
    present = c.weights > 0
    off = ~np.eye(c.n_parcels, dtype=bool)
    # A null graph has no anatomical support: every edge is `proposed`.
    assert (c.evidence[present & off] == evidence_rank("proposed")).all()
    assert (c.evidence == evidence_rank("hard")).sum() == 0
    assert c.n_streams == 0
    assert c.provenance["control"]["note"]


@pytest.mark.parametrize("name", CONTROL_NAMES)
def test_controls_keep_the_same_nodes(controls, sc_small, name):
    c = controls[name]
    assert c.n_parcels == sc_small.n_parcels
    np.testing.assert_array_equal(c.labels, sc_small.labels)


@pytest.mark.parametrize("name", CONTROL_NAMES)
def test_control_edge_evidence_says_it_is_a_control(controls, name):
    c = controls[name]
    e = c.edge_evidence(0, 1)
    assert any("control graph" in r for r in e.reasons)
    assert e.mechanistic_status == "surrogate"


# ---------------------------------------------------------------------------
# randomized: preserves degree (exactly) and strength (in rank)
# ---------------------------------------------------------------------------
def test_randomized_preserves_the_degree_sequence_exactly(controls, sc_small):
    r = controls["randomized"]
    np.testing.assert_array_equal(np.sort(r.degree()), np.sort(sc_small.degree()))
    np.testing.assert_array_equal(r.degree(), sc_small.degree())


def test_randomized_preserves_the_edge_count_and_the_weight_multiset(controls, sc_small):
    r = controls["randomized"]
    iu = np.triu_indices(r.n_parcels, 1)
    a = np.sort(r.weights[iu][r.weights[iu] > 0])
    b = np.sort(sc_small.weights[iu][sc_small.weights[iu] > 0])
    assert a.size == b.size
    np.testing.assert_allclose(a, b, rtol=1e-9)


def test_randomized_preserves_the_strength_sequence_in_rank(controls, sc_small):
    r = controls["randomized"]
    from scipy import stats

    rho = stats.spearmanr(r.strength(), sc_small.strength()).statistic
    assert rho > 0.8, f"strength rank not preserved (rho={rho:.2f})"


def test_randomized_destroys_the_topology(controls, sc_small):
    r = controls["randomized"]
    same = ((r.weights > 0) == (sc_small.weights > 0)).mean()
    overlap = ((r.weights > 0) & (sc_small.weights > 0)).sum() / max(
        (sc_small.weights > 0).sum(), 1
    )
    assert overlap < 0.6, f"rewiring kept {overlap:.0%} of the original edges"


def test_randomized_destroys_the_distance_dependence(controls, sc_small):
    """This is exactly why `distance_matched` also has to exist."""
    r = controls["randomized"]
    d_new = r.distance_mm[r.weights > 0].mean()
    d_old = sc_small.distance_mm[sc_small.weights > 0].mean()
    assert d_new > d_old * 1.1, "a degree-preserving rewiring should lengthen edges"


def test_randomized_is_deterministic_given_a_seed(sc_small):
    a = sc_small.randomized(42)
    b = sc_small.randomized(42)
    np.testing.assert_array_equal(a.weights, b.weights)
    c = sc_small.randomized(43)
    assert not np.allclose(a.weights, c.weights)


# ---------------------------------------------------------------------------
# distance_matched: preserves degree AND the edge-length distribution
# ---------------------------------------------------------------------------
def test_distance_matched_preserves_the_degree_sequence(controls, sc_small):
    np.testing.assert_array_equal(controls["distance_matched"].degree(), sc_small.degree())


def test_distance_matched_preserves_the_edge_length_distribution(controls, sc_small):
    dm = controls["distance_matched"]
    d_new = np.sort(dm.distance_mm[np.triu(dm.weights > 0, 1)])
    d_old = np.sort(sc_small.distance_mm[np.triu(sc_small.weights > 0, 1)])
    assert d_new.size == d_old.size
    # quantile-by-quantile agreement, because the swaps are binned by length
    q = np.linspace(0.05, 0.95, 19)
    np.testing.assert_allclose(
        np.quantile(d_new, q), np.quantile(d_old, q), rtol=0.06
    )
    assert abs(d_new.mean() - d_old.mean()) < 0.05 * d_old.mean()


def test_distance_matched_is_more_distance_faithful_than_randomized(controls, sc_small):
    dm, r = controls["distance_matched"], controls["randomized"]
    ref = sc_small.distance_mm[sc_small.weights > 0].mean()
    assert abs(dm.distance_mm[dm.weights > 0].mean() - ref) < abs(
        r.distance_mm[r.weights > 0].mean() - ref
    )


def test_distance_matched_still_rewires(controls, sc_small):
    dm = controls["distance_matched"]
    overlap = ((dm.weights > 0) & (sc_small.weights > 0)).sum() / max(
        (sc_small.weights > 0).sum(), 1
    )
    assert overlap < 0.95, "a control that changes nothing is not a control"


def test_distance_matched_preserves_the_weight_distance_relationship(controls, sc_small):
    from scipy import stats

    dm = controls["distance_matched"]
    m_new = np.triu(dm.weights > 0, 1)
    m_old = np.triu(sc_small.weights > 0, 1)
    rho_new = stats.spearmanr(dm.weights[m_new], dm.distance_mm[m_new]).statistic
    rho_old = stats.spearmanr(sc_small.weights[m_old], sc_small.distance_mm[m_old]).statistic
    assert np.sign(rho_new) == np.sign(rho_old)
    assert abs(rho_new - rho_old) < 0.25


# ---------------------------------------------------------------------------
# dense / local_only / graph_only
# ---------------------------------------------------------------------------
def test_dense_is_complete_and_uniform(controls, sc_small):
    d = controls["dense"]
    off = ~np.eye(d.n_parcels, dtype=bool)
    assert (d.weights[off] > 0).all()
    assert np.allclose(d.weights[off], d.weights[off][0])
    assert d.density() == pytest.approx(1.0)


def test_dense_matches_the_total_strength(controls, sc_small):
    assert controls["dense"].weights.sum() == pytest.approx(sc_small.weights.sum(), rel=1e-9)


def test_local_only_deletes_every_long_edge(controls, sc_small):
    lo = controls["local_only"]
    present = lo.weights > 0
    assert lo.distance_mm[present].max() <= 40.0 + 1e-9
    assert present.sum() < (sc_small.weights > 0).sum()


def test_local_only_keeps_short_edges_untouched(controls, sc_small):
    lo = controls["local_only"]
    short = sc_small.distance_mm <= 40.0
    np.testing.assert_allclose(lo.weights[short], sc_small.weights[short])


def test_graph_only_keeps_topology_and_flattens_weights(controls, sc_small):
    g = controls["graph_only"]
    np.testing.assert_array_equal(g.weights > 0, sc_small.weights > 0)
    vals = g.weights[g.weights > 0]
    assert np.allclose(vals, vals[0]), "graph_only must remove weight information"
    assert g.weights.sum() == pytest.approx(sc_small.weights.sum(), rel=1e-9)


def test_graph_only_preserves_degree_but_not_strength_ordering(controls, sc_small):
    g = controls["graph_only"]
    np.testing.assert_array_equal(g.degree(), sc_small.degree())
    # strength is now proportional to degree by construction
    s = g.strength()
    assert np.allclose(s / np.maximum(g.degree(), 1), (s / np.maximum(g.degree(), 1))[0])


# ---------------------------------------------------------------------------
# the set as a whole
# ---------------------------------------------------------------------------
def test_all_five_controls_are_produced(controls):
    assert sorted(controls) == sorted(CONTROL_NAMES)


def test_no_control_reproduces_the_empirical_graph(controls, sc_small):
    for name, c in controls.items():
        assert not np.allclose(c.weights, sc_small.weights), f"{name} is a no-op"


def test_controls_survive_a_roundtrip(tmp_path, controls):
    from scwbd.anatomy.connectome import StructuralPrior

    for name, c in controls.items():
        p = tmp_path / f"{name}.npz"
        c.save(p)
        q = StructuralPrior.load(p)
        assert q.control_kind == name
        np.testing.assert_allclose(q.weights, c.weights)
