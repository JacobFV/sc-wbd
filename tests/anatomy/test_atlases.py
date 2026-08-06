"""Parcellations: label counts, supports, centroids, areas, provenance."""

from __future__ import annotations

import numpy as np
import pytest

from scwbd.anatomy.atlases import (
    ATLAS_SPECS,
    Parcellation,
    available_parcellations,
    crosswalk,
    load_parcellation,
)

SURFACE_CASES = [
    ("Schaefer100x7", 100),
    ("Schaefer200x7", 200),
    ("Schaefer300x7", 300),
    ("Schaefer400x7", 400),
    ("DesikanKilliany", 68),
    ("Glasser360", 360),
]


@pytest.mark.parametrize("name,n", SURFACE_CASES)
def test_parcel_counts_match_labels(name, n):
    p = load_parcellation(name, "fsLR", "32k")
    assert p.n_parcels == n
    assert p.labels.shape == (n,)
    assert len(set(p.labels.tolist())) == n


@pytest.mark.parametrize("name,_n", SURFACE_CASES)
def test_vertex_labels_cover_exactly_the_declared_parcels(name, _n):
    p = load_parcellation(name, "fsLR", "32k")
    seen = set()
    for h, vl in p.vertex_labels.items():
        u = np.unique(vl)
        assert u.min() >= -1, "only -1 encodes unassigned"
        assert u.max() < p.n_parcels
        seen |= set(u[u >= 0].tolist())
    assert seen == set(range(p.n_parcels)), "every parcel must own at least one vertex"


@pytest.mark.parametrize("name,_n", SURFACE_CASES)
def test_no_nans_in_centroids_or_areas(name, _n):
    p = load_parcellation(name, "fsLR", "32k")
    assert np.isfinite(p.centroids_mni).all()
    assert np.isfinite(p.areas_mm2).all()
    assert (p.areas_mm2 > 0).all()


@pytest.mark.parametrize("name,_n", SURFACE_CASES)
def test_hemisphere_labels_agree_with_geometry(name, _n):
    """A parcel declared left must sit on the left, once leakage is removed."""
    p = load_parcellation(name, "fsLR", "32k")
    x = p.centroids_mni[:, 0]
    assert (x[p.hemi == "L"] < 0).all()
    assert (x[p.hemi == "R"] > 0).all()


def test_total_cortical_area_is_physiological(parc_main):
    """Both hemispheres of a group-average midthickness surface, mm^2."""
    total = parc_main.areas_mm2.sum()
    assert 6e4 < total < 2.5e5, f"implausible total cortical area {total:.0f} mm^2"


def test_volumetric_parcellation_has_volumes_not_areas():
    p = load_parcellation("TianS2", "MNI152", "1mm")
    assert np.isnan(p.areas_mm2).all(), "a volumetric atlas has no surface area"
    assert (p.volumes_mm3 > 0).all()
    assert p.voxel_labels is not None and p.affine is not None
    assert p.vertex_labels is None


def test_surface_parcellation_has_areas_not_volumes(parc_small):
    assert np.isnan(parc_small.volumes_mm3).all()
    assert (parc_small.areas_mm2 > 0).all()


@pytest.mark.parametrize(
    "name,space,density,n",
    [("TianS1", "MNI152", "1mm", 16), ("TianS2", "MNI152", "1mm", 32),
     ("Aseg14", "MNI152", "1mm", 14), ("Buckner7", "MNI152", "1mm", 7),
     ("Buckner17", "MNI152", "1mm", 17)],
)
def test_volumetric_counts(name, space, density, n):
    p = load_parcellation(name, space, density)
    assert p.n_parcels == n
    assert np.isfinite(p.centroids_mni).all()


def test_every_parcellation_carries_full_provenance():
    for name in ATLAS_SPECS:
        spec = ATLAS_SPECS[name]
        if not spec["spaces"]:
            continue
        space = next(iter(spec["spaces"]))
        p = load_parcellation(name, space, spec["spaces"][space][0])
        pr = p.provenance
        for fieldname in ("atlas_id", "version", "template_id", "software",
                          "source_url", "license", "citation"):
            v = getattr(pr, fieldname)
            assert isinstance(v, str) and v, f"{name}.provenance.{fieldname} is empty"
        assert "http" in pr.source_url or "surfer" in pr.source_url


def test_refuses_unreleased_space():
    """We do not resample an atlas into a space its authors never released."""
    with pytest.raises(ValueError, match="not distributed in space"):
        load_parcellation("Destrieux", "fsLR", "32k")


def test_refuses_unknown_atlas():
    with pytest.raises(KeyError):
        load_parcellation("NotAnAtlas")


def test_schaefer1000_is_not_offered_on_fslr():
    """One parcel is empty in the upstream fsLR-1000 vertex map, so we refuse it."""
    assert "fsLR" not in ATLAS_SPECS["Schaefer1000x7"]["spaces"]
    assert "MNI152" in ATLAS_SPECS["Schaefer1000x7"]["spaces"]


def test_crosswalk_is_a_distribution_not_an_identity(parc_main, parc_dk):
    c = crosswalk(parc_main, parc_dk)
    assert c.shape == (parc_main.n_parcels, parc_dk.n_parcels)
    assert (c >= 0).all()
    # Most Schaefer parcels straddle more than one DK parcel; that overlap is
    # the honest content of an atlas crosswalk.
    n_multi = int((np.count_nonzero(c, axis=1) > 1).sum())
    assert n_multi > 0.5 * parc_main.n_parcels
    # No parcel can be assigned more area than it has.
    assert (c.sum(axis=1) <= parc_main.areas_mm2 * 1.001 + 1e-6).all()


def test_roundtrip_save_load(tmp_path, parc_small):
    p = tmp_path / "p.npz"
    parc_small.save(p)
    q = Parcellation.load(p)
    assert q.name == parc_small.name
    np.testing.assert_array_equal(q.labels, parc_small.labels)
    np.testing.assert_allclose(q.centroids_mni, parc_small.centroids_mni)
    for h in parc_small.vertex_labels:
        np.testing.assert_array_equal(q.vertex_labels[h], parc_small.vertex_labels[h])
    q.validate()


def test_registry_is_self_consistent():
    reg = available_parcellations()
    assert reg and set(reg) == set(ATLAS_SPECS)
    for name, spec in reg.items():
        assert spec["structure"] in ("cortex", "subcortex", "cerebellum")
        assert isinstance(spec["spaces"], dict)
