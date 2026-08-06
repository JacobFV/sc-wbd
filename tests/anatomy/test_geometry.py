"""Geometry: metric properties, mesh operators, adjacency."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from scwbd.anatomy.geometry import (
    ParcelGeometry,
    SurfaceGeometry,
    geodesic_from_sources,
    load_surface,
    parcel_geometry,
)


# ---------------------------------------------------------------------------
# distances
# ---------------------------------------------------------------------------
def test_distance_matrices_are_symmetric_with_zero_diagonal(geom_small):
    for m in (geom_small.euclidean_mm, geom_small.geodesic_mm):
        np.testing.assert_allclose(m, m.T, atol=1e-9)
        np.testing.assert_allclose(np.diag(m), 0.0, atol=1e-12)


def test_no_nans_in_distances(geom_small):
    assert np.isfinite(geom_small.euclidean_mm).all()
    assert np.isfinite(geom_small.geodesic_mm).all()


def test_euclidean_obeys_the_triangle_inequality(geom_small):
    d = geom_small.euclidean_mm
    n = d.shape[0]
    i, j, k = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    viol = d[i, k] - (d[i, j] + d[j, k])
    assert viol.max() <= 1e-6, f"triangle inequality violated by {viol.max():.3g} mm"


def test_geodesic_obeys_triangle_inequality_within_a_hemisphere(geom_small, parc_small):
    """The spliced full matrix is not a metric; each hemisphere block is."""
    d = geom_small.geodesic_mm
    for h in ("L", "R"):
        idx = np.flatnonzero(parc_small.hemi == h)
        b = d[np.ix_(idx, idx)]
        m = b.shape[0]
        i, j, k = np.meshgrid(np.arange(m), np.arange(m), np.arange(m), indexing="ij")
        viol = (b[i, k] - (b[i, j] + b[j, k])).max()
        # the heat method is an approximation, so allow a small slack
        assert viol <= 0.02 * b.max(), f"{h} geodesic block badly non-metric ({viol:.2f} mm)"


def test_geodesic_is_at_least_euclidean_within_a_hemisphere(geom_small):
    """You cannot get there faster along a curved surface than through space."""
    m = geom_small.method_matrix == 0
    off = m & ~np.eye(m.shape[0], dtype=bool)
    ratio = geom_small.geodesic_mm[off] / np.maximum(geom_small.euclidean_mm[off], 1e-9)
    # heat-method numerical slack near the diagonal; require the bulk to hold
    assert np.percentile(ratio, 2) > 0.9
    assert np.median(ratio) > 1.0


def test_cross_hemisphere_entries_are_flagged_as_euclidean(geom_small, parc_small):
    L = parc_small.hemi == "L"
    cross = np.outer(L, ~L)
    assert (geom_small.method_matrix[cross] == 1).all()
    np.testing.assert_allclose(
        geom_small.geodesic_mm[cross], geom_small.euclidean_mm[cross], atol=1e-9
    )


def test_roughly_half_the_pairs_are_within_hemisphere(geom_small):
    assert 0.4 < geom_small.fraction_geodesic() < 0.6


def test_distance_magnitudes_are_anatomically_plausible(geom_small):
    e = geom_small.euclidean_mm
    assert e.max() < 220.0, "no two human brain parcels are 22 cm apart"
    assert e[e > 0].min() > 1.0


# ---------------------------------------------------------------------------
# adjacency
# ---------------------------------------------------------------------------
def test_parcel_adjacency_is_symmetric_and_loopless(geom_small):
    a = geom_small.adjacency
    np.testing.assert_array_equal(a, a.T)
    assert not np.diag(a).any()


def test_adjacent_parcels_are_closer_than_non_adjacent(geom_small):
    a = geom_small.adjacency
    off = ~np.eye(a.shape[0], dtype=bool)
    near = geom_small.geodesic_mm[a & off]
    far = geom_small.geodesic_mm[(~a) & off]
    assert near.mean() < far.mean()


def test_no_cross_hemispheric_surface_adjacency(geom_small, parc_small):
    L = parc_small.hemi == "L"
    assert not geom_small.adjacency[np.outer(L, ~L)].any()


# ---------------------------------------------------------------------------
# mesh operators
# ---------------------------------------------------------------------------
def test_cotangent_laplacian_annihilates_constants():
    s = load_surface("fsLR", "32k", "midthickness", "L")
    lap, mass = s.cotangent_laplacian()
    ones = np.ones(s.n_vertices)
    r = lap @ ones
    assert np.abs(r).max() < 1e-6, "Laplacian must have zero row sums"
    assert (mass > 0).all()
    np.testing.assert_allclose(mass.sum(), s.vertex_areas().sum(), rtol=1e-9)


def test_laplacian_is_symmetric():
    s = load_surface("fsLR", "32k", "midthickness", "R")
    lap, _ = s.cotangent_laplacian()
    asym = abs(lap - lap.T)
    assert asym.max() < 1e-9


def test_vertex_adjacency_is_symmetric_binary_loopless():
    s = load_surface("fsaverage5", "10k", "midthickness", "L")
    a = s.adjacency()
    assert (a != a.T).nnz == 0
    assert set(np.unique(a.data).tolist()) <= {1.0}
    assert a.diagonal().sum() == 0
    deg = np.asarray(a.sum(axis=1)).ravel()
    # a closed triangulated surface has mean vertex degree very near 6
    assert 5.0 < deg.mean() < 7.0


def test_local_kernel_rows_sum_to_one_and_stay_local():
    s = load_surface("fsaverage5", "10k", "midthickness", "L")
    src = np.arange(0, s.n_vertices, 500)
    k = s.local_kernel(8.0, sources=src)
    assert k.shape == (src.size, s.n_vertices)
    np.testing.assert_allclose(np.asarray(k.sum(axis=1)).ravel(), 1.0, atol=1e-9)
    density = k.nnz / (k.shape[0] * k.shape[1])
    assert density < 0.05, "a local kernel that touches everything is not local"


def test_surface_geometry_bundle_builds():
    g = SurfaceGeometry.build("fsaverage5", "10k", "L")
    assert g.laplacian.shape == (g.surface.n_vertices,) * 2
    assert g.mass.shape == (g.surface.n_vertices,)
    assert sp.issparse(g.adjacency)


def test_geodesic_from_a_source_is_zero_at_the_source():
    s = load_surface("fsaverage5", "10k", "midthickness", "L")
    d = geodesic_from_sources(s, np.array([0, 100, 1000]))
    assert d.shape == (3, s.n_vertices)
    assert np.isfinite(d).all()
    assert abs(d[0, 0]) < 1e-6
    assert abs(d[1, 100]) < 1e-6
    assert (d >= -1e-6).all()


# ---------------------------------------------------------------------------
# ledger and io
# ---------------------------------------------------------------------------
def test_geometry_ledger_satisfies_r08(geom_small):
    led = geom_small.ledger
    assert led.has_estimator(), "R08 would refuse this ledger"
    assert led.bias_status == "externally_bounded"
    assert led.external_bound_source
    assert "forbidden_inference" in led.validity_domain


def test_geometry_roundtrip(tmp_path, geom_small):
    p = tmp_path / "g.npz"
    geom_small.save(p)
    q = ParcelGeometry.load(p)
    np.testing.assert_allclose(q.geodesic_mm, geom_small.geodesic_mm)
    np.testing.assert_array_equal(q.method_matrix, geom_small.method_matrix)
    assert q.ledger.bias_status == geom_small.ledger.bias_status
