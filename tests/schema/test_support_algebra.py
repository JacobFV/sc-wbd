"""O-2's algebra is defined by what it refuses, so the refusals are the tests.

An algebra that always returns a map is worse than no algebra: a fabricated
correspondence between two parcellations is indistinguishable at the type level
from a derived one.  Most of this file therefore watches a refusal fire, and
each one also checks the message *names the missing declaration* -- a refusal
nobody can act on is a wall, not a gate.
"""

from __future__ import annotations

import math

import pytest

from scwbd.schema.supports import PSF, Support, TemporalSupport
from scwbd.schema.support_algebra import (
    ElementType,
    SupportIncompatible,
    common_temporal_refinement,
    compose_psf,
    relate,
)
from scwbd.schema.units import Unit


def _parcels(n: int, frame: str = "fsLR32k", psf: PSF | None = None) -> Support:
    return Support(kind="parcel", frame=frame, units=Unit("A*m"), n_elements=n, psf=psf)


# --------------------------------------------------------------------- elements


def test_a_vector_element_without_a_frame_is_refused():
    """Three numbers with no frame are three numbers, not a vector."""
    with pytest.raises(ValueError, match="component_frame"):
        ElementType(rank=1, dim=3)


def test_a_scalar_element_cannot_have_width():
    with pytest.raises(ValueError, match="scalars"):
        ElementType(rank=0, dim=3)


def test_width_is_numbers_per_element():
    assert ElementType().width == 1
    assert ElementType(rank=1, dim=3, component_frame="fsLR32k").width == 3


# ---------------------------------------------------------------------- refusals


def test_differing_frames_are_refused_and_the_message_names_registration():
    with pytest.raises(SupportIncompatible, match="registration"):
        relate(_parcels(400), _parcels(400, frame="MNI152"))


def test_differing_units_are_refused():
    a = _parcels(400)
    b = Support(kind="parcel", frame="fsLR32k", units=Unit("V"), n_elements=400)
    with pytest.raises(SupportIncompatible, match="units differ"):
        relate(a, b)


def test_a_rank_change_without_an_orientation_is_refused():
    """The measured refusal: 32.1% vs 83.4% of the whitened lead field.

    On Schaefer400x7, the model's own 400 cortical parcels.  Stated as 5.6% vs
    51.7% until 2026-08-09, which is the same pair on Desikan-Killiany
    (ISSUE-015).

    Collapsing three numbers per parcel to one is a projection onto an
    orientation -- a physical fact about the cortex -- not a reshape.
    """
    vec = ElementType(rank=1, dim=3, component_frame="fsLR32k")
    with pytest.raises(SupportIncompatible, match="orientation_ref"):
        relate(_parcels(400), _parcels(400), src_elements=vec, dst_elements=ElementType())


def test_an_unknown_size_cannot_be_related():
    a = Support(kind="parcel", frame="fsLR32k", units=Unit("A*m"))
    with pytest.raises(SupportIncompatible, match="n_elements"):
        relate(a, _parcels(400))


def test_vector_components_in_different_frames_are_refused():
    a = ElementType(rank=1, dim=3, component_frame="fsLR32k")
    b = ElementType(rank=1, dim=3, component_frame="MNI152")
    with pytest.raises(SupportIncompatible, match="rotate"):
        relate(_parcels(400), _parcels(400), src_elements=a, dst_elements=b)


# ----------------------------------------------------------------- derived maps


def test_identical_supports_give_identity_and_identity_is_the_only_free_map():
    m = relate(_parcels(400), _parcels(400))
    assert m.kind == "identity"
    assert not m.lossy and not m.invents
    assert m.uncertainty_note == ""


def test_fine_to_coarse_is_a_lossy_restriction():
    m = relate(_parcels(32492), _parcels(400))
    assert m.kind == "restriction"
    assert m.lossy and not m.invents
    assert "discarded" in m.uncertainty_note


def test_coarse_to_fine_invents_and_must_say_so():
    m = relate(_parcels(400), _parcels(32492))
    assert m.kind == "prolongation"
    assert m.invents
    assert "not measured" in m.uncertainty_note


def test_a_prolongation_that_claims_to_be_free_cannot_be_constructed():
    """The guard, watched firing.  Without it ``invents`` is decoration."""
    from scwbd.schema.support_algebra import SupportMap

    with pytest.raises(ValueError, match="upsampling is free"):
        SupportMap(kind="prolongation", n_in=1, n_out=4, invents=False, uncertainty_note="x")
    with pytest.raises(ValueError, match="false claim"):
        SupportMap(kind="restriction", n_in=4, n_out=1, lossy=False, uncertainty_note="x")
    with pytest.raises(ValueError, match="uncertainty"):
        SupportMap(kind="restriction", n_in=4, n_out=1, lossy=True)


def test_element_width_reaches_the_map_counts():
    """400 parcels x 3 components is 1200 numbers, and the map must say 1200."""
    vec = ElementType(rank=1, dim=3, component_frame="fsLR32k")
    m = relate(_parcels(400), _parcels(400), src_elements=vec, dst_elements=vec)
    assert m.n_in == 1200 and m.n_out == 1200


def test_an_orientation_map_collapsing_rank_is_lossy_not_inventive():
    vec = ElementType(rank=1, dim=3, component_frame="fsLR32k")
    m = relate(
        _parcels(400),
        _parcels(400),
        src_elements=vec,
        dst_elements=ElementType(),
        orientation_ref="cortical_normals",
    )
    assert m.kind == "orientation"
    assert m.lossy and not m.invents
    assert m.n_in == 1200 and m.n_out == 400
    assert "cortical_normals" in m.uncertainty_note


# ------------------------------------------------------------------------ psf


def test_gaussians_compose_in_quadrature():
    a = PSF(kind="gaussian", fwhm=(3.0, 3.0))
    b = PSF(kind="gaussian", fwhm=(4.0, 4.0))
    c = compose_psf(a, b)
    assert c is not None and c.fwhm is not None
    assert math.isclose(c.fwhm[0], 5.0)


def test_opaque_kernels_are_refused_rather_than_approximated():
    a = PSF(kind="lead_field", kernel_ref="asset://lf")
    b = PSF(kind="hemodynamic", kernel_ref="asset://hrf")
    with pytest.raises(SupportIncompatible, match="kernel assets"):
        compose_psf(a, b)


def test_a_point_psf_composes_as_the_identity():
    b = PSF(kind="gaussian", fwhm=(3.0,))
    assert compose_psf(PSF(kind="point"), b) is b
    assert compose_psf(b, PSF(kind="point")) is b


def test_nominality_is_contagious():
    """A composed kernel is no more real than its least real factor."""
    a = PSF(kind="gaussian", fwhm=(3.0,), nominal=True)
    b = PSF(kind="gaussian", fwhm=(4.0,))
    c = compose_psf(a, b)
    assert c is not None and c.nominal


# -------------------------------------------------------------------- temporal


def test_eeg_against_bold_refines_onto_the_faster_clock():
    """The case O-2 was written for: 5000 Hz EEG against 0.5 Hz BOLD."""
    eeg = TemporalSupport(clock="scanner", dt=1.0 / 5000.0)
    bold = TemporalSupport(clock="scanner", dt=2.0, integration_window=2.0)
    fine, m_eeg, m_bold = common_temporal_refinement(eeg, bold)
    assert fine.dt == eeg.dt
    assert m_eeg.kind == "identity"
    assert m_bold.kind == "prolongation" and m_bold.invents
    assert "never measured" in m_bold.uncertainty_note


def test_unrelated_clocks_are_refused_and_the_message_names_the_fix():
    a = TemporalSupport(clock="eeg_amp", dt=1.0 / 5000.0)
    b = TemporalSupport(clock="scanner", dt=2.0)
    with pytest.raises(SupportIncompatible, match="synchronisation"):
        common_temporal_refinement(a, b)


def test_the_refinement_is_symmetric_in_its_arguments():
    eeg = TemporalSupport(clock="scanner", dt=1.0 / 5000.0)
    bold = TemporalSupport(clock="scanner", dt=2.0)
    f1, _, _ = common_temporal_refinement(eeg, bold)
    f2, _, _ = common_temporal_refinement(bold, eeg)
    assert f1.dt == f2.dt
