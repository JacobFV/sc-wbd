"""Frame handling for BOLD -> parcel registration.

The mutation each guard is tested against is named in its docstring
(``reports/decorative_guards.md`` rec. 1).

The guard that matters most here is
:func:`test_hemisphere_guard_fires_on_a_mirrored_atlas`. The Schaefer volume is
stored with ``affine[0,0] = -1`` and the matching template with ``+1``; indexing
one by the other's voxel indices mirrors the brain by 181 mm. Because the atlas
is bilateral, every parcel-count, coverage and mean-signal summary stays
completely plausible under that error -- only a lateralisation claim exposes it,
and by then it is a published result rather than a bug report.
"""

from __future__ import annotations

import numpy as np
import pytest

from scwbd.anatomy.registration import (
    ATLAS_TEMPLATE,
    TransformChain,
    assert_hemisphere_convention,
    labels_to_epi_grid,
    load_atlas_volume,
    parcel_coverage,
)


@pytest.fixture(scope="module")
def atlas():
    return load_atlas_volume("Schaefer400x7")


def test_shipped_atlas_is_on_the_template_this_module_registers_to(atlas):
    """The Schaefer volume must be NLin6Asym, not NLin2009cAsym.

    They have different grids (182x218x182 vs 193x229x193) and differ by several
    mm in ventral and temporal cortex. Registering to one and labelling with the
    other is a silent systematic error.
    """
    _, _, _, _, meta = atlas
    assert meta["provenance"]["template_id"] == ATLAS_TEMPLATE


def test_hemisphere_convention_holds_on_the_shipped_atlas(atlas):
    vl, aff, _, hemi, _ = atlas
    out = assert_hemisphere_convention(vl, aff, hemi)
    assert out["mean_world_x_L"] < 0 < out["mean_world_x_R"]


def test_hemisphere_guard_fires_on_a_mirrored_atlas(atlas):
    """Mutation: replace the atlas affine with the template's (x sign flipped).

    Same shape, same y/z origin, opposite x direction -- exactly what happens if
    someone reuses the template affine for the label volume.
    """
    vl, _, _, hemi, _ = atlas
    mirrored = np.array(
        [[1.0, 0, 0, -91.0], [0, 1.0, 0, -126.0], [0, 0, 1.0, -72.0], [0, 0, 0, 1.0]]
    )
    with pytest.raises(ValueError, match="hemisphere convention violated|mirrored"):
        assert_hemisphere_convention(vl, mirrored, hemi)


def test_identity_chain_returns_the_atlas_unchanged(atlas):
    """A no-op transform must reproduce the atlas on its own grid.

    If the affine bookkeeping in ``labels_to_epi_grid`` is inverted or composed
    in the wrong order, this is where it shows: the round trip stops being the
    identity.
    """
    vl, aff, _, hemi, _ = atlas
    chain = TransformChain(
        subject="identity",
        epi_from_t1w=np.eye(4),
        t1w_from_template=np.eye(4),
        engine="identity",
        lineage="test",
    )
    out = labels_to_epi_grid(chain, vl.shape, aff, vl, aff, hemi=hemi)
    assert out.shape == vl.shape
    agree = float((out == vl).mean())
    assert agree > 0.999, f"identity chain did not reproduce the atlas ({agree:.4f})"


def test_a_translated_chain_actually_moves_the_labels(atlas):
    """Mutation: 10 mm translation must change which parcel a voxel gets.

    If this did not change the output, ``perturbed`` would be a no-op and the
    perturbation control would pass vacuously -- a decorative control, which is
    the specific failure the control exists to detect in others.
    """
    vl, aff, _, hemi, _ = atlas
    base = TransformChain("s", np.eye(4), np.eye(4), "identity", "test")
    moved = base.perturbed((10.0, 0.0, 0.0))
    assert np.allclose(moved.epi_from_t1w[:3, 3], [10.0, 0.0, 0.0])

    a = labels_to_epi_grid(base, vl.shape, aff, vl, aff, hemi=hemi)
    b = labels_to_epi_grid(moved, vl.shape, aff, vl, aff, hemi=hemi)
    inside = (a >= 0) & (b >= 0)
    changed = float((a[inside] != b[inside]).mean())
    assert changed > 0.15, (
        f"a 10 mm shift changed only {changed:.1%} of in-brain parcel assignments; "
        "the perturbation control would be measuring nothing"
    )


def test_atlas_loader_refuses_a_template_mismatch(monkeypatch):
    """Mutation: declare the module registers to a different MNI variant."""
    import scwbd.anatomy.registration as R

    monkeypatch.setattr(R, "ATLAS_TEMPLATE", "MNI152NLin2009cAsym")
    with pytest.raises(ValueError, match="Mixing MNI variants|defined on"):
        R.load_atlas_volume("Schaefer400x7")


def test_coverage_counts_zero_as_unobserved(atlas):
    """A parcel with no EPI voxels is unobserved, not zero-signal."""
    _, aff, names, _, _ = atlas
    lab = np.full((4, 4, 4), -1, dtype=np.int32)
    lab[0, 0, 0] = 7
    cov = parcel_coverage(lab, 400, names, aff, subject="s", run="r")
    assert cov.covered.sum() == 1
    assert cov.covered[7]
    assert cov.n_voxels[7] == 1
    assert cov.summary()["n_uncovered"] == 399
    assert "impute" in cov.notes.lower()


def test_anatomy_prior_declares_the_frame_it_is_actually_in():
    """The declared frame must come from the atlas, not a hardcoded default.

    Mutation this catches: restoring ``frame=str(getattr(obj, "frame",
    "MNI152NLin2009cAsym_RAS"))``. BrainPrior exposes no ``frame`` attribute, so
    that default won on every load and asserted a template the data is not in --
    a *declared* wrong frame, which downstream registration would trust, rather
    than an unknown one that R01 would refuse.
    """
    from scwbd.foundation.anatomy import load_anatomy

    a = load_anatomy(device="cpu")
    assert "fsLR" in a.frame and "conte69" in a.frame
    assert "NLin2009cAsym" not in a.frame, (
        "the prior is declaring MNI152NLin2009cAsym again; its support is the "
        "fsLR-32k/conte69 surface and its volumetric atlas is MNI152NLin6Asym"
    )


def test_coverage_records_whether_a_brain_mask_was_used(atlas):
    """An FOV count must not be quotable as a signal count.

    Mutation this catches: dropping the ``basis`` field so both paths return
    the same-looking object. The EPI array is a box; without a mask every
    parcel inside the box reads as 'covered', which on a real subject gave
    400/400 and would have been reported as full observation.
    """
    _, aff, names, _, _ = atlas
    lab = np.zeros((6, 6, 6), dtype=np.int32)  # whole box is parcel 0
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[0, 0, 0] = True

    fov = parcel_coverage(lab, 400, names, aff, subject="s", run="r")
    masked = parcel_coverage(lab, 400, names, aff, subject="s", run="r", brain_mask=mask)

    assert fov.basis == "field_of_view"
    assert masked.basis == "brain_mask"
    assert fov.n_voxels[0] == 216 and masked.n_voxels[0] == 1
    assert "NOT A BRAIN MASK" in fov.notes
    assert "NOT A BRAIN MASK" not in masked.notes
    assert fov.summary()["basis"] == "field_of_view"
