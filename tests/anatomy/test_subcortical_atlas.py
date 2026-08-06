"""The subcortical atlas: the default must be permissive, the NC one opt-in.

Same shape as ``test_ei_ordering.py``, for the same reason and against the same
class of defect. Every assertion here was watched fail; the mutation is named in
each docstring.
"""

from __future__ import annotations

import numpy as np
import pytest

from scwbd.anatomy import sources as S
from scwbd.anatomy.atlases import _ASEG14, _ASEG14_FROM_TIAN, load_parcellation
from scwbd.anatomy.connectome import (
    DEFAULT_SUBCORTICAL_ATLAS,
    SUBCORTICAL_ATLASES,
    load_structural_prior,
)
from scwbd.anatomy.priors import BrainPrior
from scwbd.release.licence import is_noncommercial_text, is_share_alike_text

ASEG14_NAMES = [short for short, _ in _ASEG14]


# ---------------------------------------------------------------------------
# the default path
# ---------------------------------------------------------------------------
def test_default_subcortical_atlas_is_not_non_commercial():
    """The point of the substitution, computed from source terms.

    Watched fail by setting ``DEFAULT_SUBCORTICAL_ATLAS = "Aseg14"``.
    """
    key = SUBCORTICAL_ATLASES[DEFAULT_SUBCORTICAL_ATLAS]["licence_key"]
    text = S.SRC[key]["license"]
    assert is_noncommercial_text(text) is False, (key, text)
    assert is_share_alike_text(text) is False, (key, text)


def test_default_brain_prior_does_not_load_harvard_oxford(brain_prior):
    """``harvardoxford`` must not appear among the sources of a default build.

    Watched fail by flipping the default.
    """
    prov = brain_prior.provenance
    assert "harvardoxford" not in prov["sources"]
    assert prov["subcortical_atlas"]["name"] == DEFAULT_SUBCORTICAL_ATLAS
    assert prov["subcortical_atlas"]["licence_key"] != "harvardoxford"
    assert prov["subcortical_atlas"]["is_default"] is True


def test_the_licence_choice_is_recorded_verbatim_from_the_registry(brain_prior):
    """A choice that leaves no trace is a default, not a choice.

    Watched fail by dropping the ``licence`` key from the provenance block.
    """
    rec = brain_prior.provenance["subcortical_atlas"]
    assert rec["licence"] == S.SRC[rec["licence_key"]]["license"]
    assert "non-commercial" not in rec["licence"].lower()


# ---------------------------------------------------------------------------
# the substitution must not move the connectome's rows
# ---------------------------------------------------------------------------
def test_both_atlases_carry_the_same_14_labels_in_the_same_order():
    """The connectome's row order is fixed by strucLabels_sctx and is not a choice.

    A join by position rather than by structure identity is how the pallidum and
    the putamen get swapped, silently, with every downstream number still
    looking plausible.

    Watched fail by reversing ``_ASEG14_FROM_TIAN``'s thalamus entry into a
    different structure's slot.
    """
    for name in SUBCORTICAL_ATLASES:
        p = load_parcellation(name, "MNI152", "1mm")
        assert [str(x) for x in p.labels] == ASEG14_NAMES, name
        assert [str(x) for x in p.hemi] == [n[0] for n in ASEG14_NAMES], name
        assert set(map(str, p.structure)) == {"subcortex"}, name


def test_each_structure_lands_on_its_own_structure_not_a_neighbour():
    """The guard the label test only *claimed* to be.

    Checking that both atlases carry the same 14 label strings says nothing
    about whether the geometry behind each label is the right structure.
    Mutation S4 -- swapping ``pal`` and ``put`` in the merge map, which is
    precisely the positional-join failure the label test's docstring names --
    passed every test in this file. Two adjacent structures a few millimetres
    apart survive any aggregate displacement bound.

    So: for each structure, the *nearest* Harvard-Oxford structure to the new
    centroid must be itself. That fires on any swap, and it is the assertion the
    docstring was writing a cheque for.

    Watched fail by swapping ``pal``/``put``, and again by swapping
    ``caud``/``put``.
    """
    new_p = load_parcellation("Aseg14T", "MNI152", "1mm")
    old_p = load_parcellation("Aseg14", "MNI152", "1mm")
    d = np.linalg.norm(
        new_p.centroids_mni[:, None, :] - old_p.centroids_mni[None, :, :], axis=-1
    )
    nearest = np.argmin(d, axis=1)
    wrong = [
        (ASEG14_NAMES[i], ASEG14_NAMES[j])
        for i, j in enumerate(nearest)
        if i != j
    ]
    assert not wrong, f"structure(s) nearest a different structure: {wrong}"


def test_the_tian_merge_map_covers_every_structure_exactly_once():
    """Every aseg structure must be built, and no Tian parcel used twice."""
    stems = sorted({n[1:] for n in ASEG14_NAMES})
    assert sorted(_ASEG14_FROM_TIAN) == stems
    used = [p for parts in _ASEG14_FROM_TIAN.values() for p in parts]
    assert len(used) == len(set(used)), f"a Tian parcel is used twice: {used}"
    assert set(_ASEG14_FROM_TIAN["thal"]) == {"aTHA", "pTHA"}, (
        "the connectome resolves one thalamus; Tian S1 splits it, and both "
        "halves must be recombined rather than one being picked"
    )


def test_the_permissive_atlas_is_volumetric_like_the_one_it_replaces():
    """Orphaning check: it must be a drop-in, not a different kind of object.

    Watched fail by building it from a surface mesh instead (the ENIGMA
    candidate, which has no ``voxel_labels`` -- see the report).
    """
    p = load_parcellation(DEFAULT_SUBCORTICAL_ATLAS, "MNI152", "1mm")
    assert p.voxel_labels is not None and p.affine is not None
    assert p.vertex_labels is None
    assert (p.volumes_mm3 > 0).all()
    assert np.isnan(p.areas_mm2).all()


def test_the_connectome_still_loads_and_its_label_order_check_still_passes():
    """`load_structural_prior` order-checks labels against the ENIGMA matrix."""
    sp = load_structural_prior("Schaefer100x7", include_subcortex=True)
    assert sp.n_parcels == 114
    assert [str(x) for x in sp.labels[-14:]] == ASEG14_NAMES


def test_the_subcortical_atlas_is_in_the_connectome_cache_key():
    """Two atlases must not share a cache entry.

    They produce different subcortical centroids and therefore different delays;
    a shared key would serve one atlas's delays for the other, silently.

    Watched fail by dropping ``sub_atlas`` from the ``tag`` expression -- the two
    priors then come back with identical distance matrices.
    """
    a = load_structural_prior("Schaefer100x7", include_subcortex=True,
                              subcortical_atlas="Aseg14T")
    b = load_structural_prior("Schaefer100x7", include_subcortex=True,
                              subcortical_atlas="Aseg14")
    assert not np.allclose(a.distance_mm[-14:, :], b.distance_mm[-14:, :]), (
        "the two subcortical atlases produced identical distances, which means "
        "one cache entry was served for both"
    )


# ---------------------------------------------------------------------------
# the opt-in path
# ---------------------------------------------------------------------------
def test_harvard_oxford_is_still_available_and_declares_itself_non_commercial():
    """Retained as an explicit, self-recording choice -- not deleted.

    Watched fail by removing the ``Aseg14`` entry from ``SUBCORTICAL_ATLASES``.
    """
    assert "Aseg14" in SUBCORTICAL_ATLASES
    entry = SUBCORTICAL_ATLASES["Aseg14"]
    assert entry["licence_key"] == "harvardoxford"
    assert "NON-COMMERCIAL" in entry["description"]
    assert is_noncommercial_text(S.SRC["harvardoxford"]["license"]) is True

    bp = BrainPrior.load("Schaefer100x7", subcortical_atlas="Aseg14")
    rec = bp.provenance["subcortical_atlas"]
    assert rec["name"] == "Aseg14"
    assert rec["is_default"] is False
    assert "harvardoxford" in bp.provenance["sources"]


def test_an_unknown_subcortical_atlas_raises_rather_than_falling_back():
    """A typo must not silently select the default.

    Both entry points are guarded and the guards are redundant by design --
    ``load_structural_prior`` is public API in its own right. Mutation S6
    (deleting the ``BrainPrior`` guard) originally killed nothing, because the
    connectome guard caught it and the test could not tell which fired. The
    messages now name their own module and the test asserts which one it got.
    """
    with pytest.raises(KeyError, match="BrainPrior.load"):
        BrainPrior.load("Schaefer100x7", subcortical_atlas="Aseg14Q")
    with pytest.raises(KeyError, match="load_structural_prior"):
        load_structural_prior("Schaefer100x7", subcortical_atlas="Aseg14Q")


# ---------------------------------------------------------------------------
# what the substitution cost
# ---------------------------------------------------------------------------
def test_the_substitution_changes_geometry_and_the_change_is_bounded():
    """Regression guard on the measured anatomical cost.

    Measured 2026-08-06: centroid displacement median 1.58 mm, max 4.23 mm
    (right accumbens); volume ratio 0.74-1.13 for twelve structures and
    2.18-2.47 for the accumbens. The bounds are loose on purpose -- the point is
    that this is a real change of delineation, neither a no-op nor a wholesale
    relabelling, and that a future change to either atlas moves it.
    """
    new = load_parcellation("Aseg14T", "MNI152", "1mm")
    old = load_parcellation("Aseg14", "MNI152", "1mm")
    disp = np.linalg.norm(new.centroids_mni - old.centroids_mni, axis=1)
    assert 0.5 < np.median(disp) < 3.0, f"median displacement {np.median(disp):.2f} mm"
    assert disp.max() < 8.0, f"max displacement {disp.max():.2f} mm"
    ratio = new.volumes_mm3 / old.volumes_mm3
    acc = [i for i, n in enumerate(ASEG14_NAMES) if n.endswith("accumb")]
    others = [i for i in range(14) if i not in acc]
    assert (ratio[others] > 0.5).all() and (ratio[others] < 1.5).all()
    assert (ratio[acc] > 1.5).all(), (
        "the accumbens disagreement is the substitution's largest anatomical "
        "cost and is asserted so it cannot quietly disappear from the report"
    )


def test_the_delay_change_is_small_against_the_velocity_prior(brain_prior):
    """The centroids exist to produce delays; that is where the cost is judged.

    Measured: the subcortical block of the median delay matrix moves by +0.13 %
    on signed mean, against a conduction-velocity prior whose 95 % interval
    spans a factor of nine. Signed, not absolute: for a treatment/control
    difference the sign is the quantity of interest.
    """
    new = brain_prior
    old = BrainPrior.load(new.atlas, subcortical_atlas="Aseg14")
    n_ctx = new.n_cortex
    sub = np.zeros((new.n_parcels, new.n_parcels), bool)
    sub[n_ctx:, :] = True
    sub[:, n_ctx:] = True
    m = new.structural.mask("soft") & sub
    a, b = new.median_delay_ms(), old.median_delay_ms()
    ok = m & np.isfinite(a) & np.isfinite(b) & (b > 0)
    rel = ((a - b) / b)[ok]
    assert abs(float(rel.mean())) < 0.05, f"signed mean delay change {rel.mean():+.4f}"
    assert float(np.abs(rel).max()) < 0.5
