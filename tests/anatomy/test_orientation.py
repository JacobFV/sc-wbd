"""Per-parcel net dipole orientation and the folding coherence that scales it.

Every assertion here was watched fail before it was trusted
(``reports/decorative_guards.md`` rec. 1); the mutation used is named in each
docstring.

The substantive test is :func:`test_coherence_rises_as_parcels_shrink`, which is
the one that can embarrass the whole field: if within-parcel cancellation were
not real, coherence would not depend on parcel size, and ``coherence`` would be
a decorative multiplier of 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from scwbd.anatomy import BrainPrior, load_parcellation
from scwbd.anatomy.geometry import parcel_orientation

MAIN_ATLAS = "Schaefer400x7"


@pytest.fixture(scope="module")
def parc():
    return load_parcellation(MAIN_ATLAS, "fsLR", "32k")


@pytest.fixture(scope="module")
def ori(parc):
    return parcel_orientation(parc)


# ------------------------------------------------------------------ the object
def test_every_cortical_parcel_gets_an_orientation(ori):
    assert ori.n_parcels == 400
    assert ori.n_covered() == 400
    assert ori.normal.shape == (400, 3)


def test_normals_are_unit_vectors_where_covered(ori):
    m = np.linalg.norm(ori.normal[ori.covered], axis=1)
    assert np.abs(m - 1.0).max() < 1e-9


def test_coherence_is_a_fraction(ori):
    c = ori.coherence[ori.covered]
    assert (c >= 0).all() and (c <= 1).all()
    # And it is not degenerate: a constant 1.0 would mean no folding was
    # measured at all, i.e. the vector sum was never actually taken.
    assert c.min() < 0.5, "no parcel shows cancellation -- coherence is not being computed"
    assert c.std() > 0.05


def test_effective_area_is_coherence_times_area(ori):
    c, a, e = ori.coherence[ori.covered], ori.area_mm2[ori.covered], ori.effective_area_mm2[ori.covered]
    assert np.allclose(e, c * a, rtol=1e-9)
    assert (e <= a + 1e-6).all(), "folding cannot increase the sensor-visible moment"


def test_sign_convention_is_resolved_not_assumed(ori):
    """Agreement near 0.5 would mean inconsistent winding and a meaningless sign.

    It is well below 1 because deep sulcal faces genuinely point inward, so a
    value near 1 would be as suspicious as one near 0.5.
    """
    assert 0.6 < ori.sign_agreement < 0.98
    assert ori.handedness and ori.sign_convention
    assert ori.frame.startswith("fsLR")


def test_orientation_refuses_a_volumetric_parcellation():
    """Mutation: ask for a dipole direction on a volume atlas.

    A volume parcel has no cortical sheet; returning nan for all of them would
    look like a covered=False result rather than a category error.
    """
    vol = load_parcellation("Aseg14T", "MNI152", "1mm")
    with pytest.raises(ValueError, match="volumetric|surface parcellation"):
        parcel_orientation(vol)


# ----------------------------------------------------------- padding onto 414
def test_brain_prior_pads_orientation_and_marks_subcortex_uncovered():
    """The 14 subcortical parcels must be nan/covered=False, never zero.

    A zero vector is a direction of zero length, which a downstream lead field
    would happily multiply by and silently get nothing from. nan and an explicit
    covered mask cannot be used by accident.
    """
    p = BrainPrior.load(MAIN_ATLAS, include_subcortex=True)
    o = p.dipole_orientation()
    assert p.n_parcels == 414
    assert o.n_parcels == 414
    assert o.n_covered() == 400
    assert o.covered[:400].all()
    assert not o.covered[400:].any()
    assert np.isnan(o.normal[400:]).all(), "subcortical normals must be nan, not zero"
    assert np.isnan(o.coherence[400:]).all()


# ------------------------------------------------------- the substantive claim
def test_coherence_rises_as_parcels_shrink():
    """Within-parcel cancellation is real, and finer parcels recover some of it.

    This is the geometric mechanism behind 🧭 Gauss's lead-field result. If
    coherence did not depend on parcel size, cancellation would not be being
    measured and the multiplier would be decorative.

    It also bounds the payoff from subdividing: total cortical area is fixed, so
    the most any parcellation can recover is the gap between the current
    surviving fraction and 1.
    """
    surv = {}
    for atlas in ("Schaefer100x7", "Schaefer200x7", "Schaefer400x7"):
        o = parcel_orientation(load_parcellation(atlas, "fsLR", "32k"))
        a = o.area_mm2[o.covered]
        e = o.effective_area_mm2[o.covered]
        surv[atlas] = float(e.sum() / a.sum())

    assert surv["Schaefer100x7"] < surv["Schaefer200x7"] < surv["Schaefer400x7"], (
        f"coherence does not increase as parcels shrink: {surv}. Either "
        "cancellation is not being computed or the areas are wrong."
    )
    # Total cortical area must NOT change with the parcellation -- if it does,
    # faces are being dropped or double counted at parcel boundaries.
    areas = [
        float(np.nansum(parcel_orientation(load_parcellation(a, "fsLR", "32k")).area_mm2))
        for a in ("Schaefer100x7", "Schaefer400x7")
    ]
    assert abs(areas[0] - areas[1]) / areas[0] < 0.01, (
        f"total cortical area depends on the parcellation ({areas}); face-to-parcel "
        "assignment is losing or duplicating faces"
    )


def test_orientation_reaches_the_foundation_prior():
    from scwbd.foundation.anatomy import load_anatomy

    a = load_anatomy(device="cpu")
    assert a.normal is not None and a.normal.shape == (a.n_regions, 3)
    assert a.normal_coherence is not None and a.normal_covered is not None
    assert int(a.normal_covered.sum()) == 400
    s = a.summary()["orientation"]
    assert s["n_covered"] == 400 and 0.0 < s["coherence_min"] < s["coherence_median"] <= 1.0
