"""O-5 (vector-valued regional state) and O-6 (segment layout).

O-6 retires narrowing `padded-family-state` by making its failure mode unrepresentable rather than
merely guarded: there is no pad, so there is nothing to write into.  The tests
that made the span guard fire are therefore **not** ported here — they have no
referent.  What survives is the type discipline: a family still cannot read a
component it does not declare, because there is no offset arithmetic that could
reach one.
"""

from __future__ import annotations

import pytest
import torch

from scwbd.foundation.anatomy import load_anatomy
from scwbd.foundation.families import FamilyStateLayout, PortMismatch, SpanViolation, derive_families
from scwbd.foundation.orientation import DipoleProjection
from scwbd.foundation.ragged import FamilyState


@pytest.fixture(scope="module")
def anat():
    return load_anatomy(device="cpu")


@pytest.fixture(scope="module")
def layout(anat):
    return FamilyStateLayout(derive_families(anat))


# ======================================================================
# O-6: the segment layout
# ======================================================================
def test_the_pad_no_longer_exists(layout):
    st = FamilyState.zeros(layout, 2, 3)
    d = st.describe()
    assert d["cells_ragged"] < d["cells_if_padded"]
    assert d["cells_ragged"] == sum(f.n_regions * f.dim for f in layout)
    # the number `padded-family-state` was costing
    assert d["padding_fraction_avoided"] > 0.4


def test_blocks_are_dense_and_family_shaped(layout):
    st = FamilyState.zeros(layout, 4)
    for f in layout:
        assert st[f.name].shape == (4, f.n_regions, f.dim)


def test_a_wrong_shaped_block_is_refused_at_construction(layout):
    bad = {f.name: torch.zeros(2, f.n_regions, f.dim) for f in layout}
    name = layout.families[0].name
    bad[name] = torch.zeros(2, layout.family(name).n_regions, layout.family(name).dim + 1)
    with pytest.raises(SpanViolation, match="trailing shape"):
        FamilyState(layout, bad)


def test_a_missing_family_is_refused(layout):
    blocks = {f.name: torch.zeros(2, f.n_regions, f.dim) for f in layout}
    blocks.pop(layout.families[0].name)
    with pytest.raises(SpanViolation, match="missing block"):
        FamilyState(layout, blocks)


def test_reading_another_familys_component_is_still_refused(layout):
    """The one guarantee the span guard gave that still has a referent."""
    st = FamilyState.zeros(layout, 2)
    assert st.get("subcortex_hippo", "k").shape[-1] == 16
    with pytest.raises(SpanViolation, match="does not declare component 'k'"):
        st.get("cortex_unimodal", "k")


def test_there_is_no_whole_brain_view_of_a_family_private_component(layout):
    st = FamilyState.zeros(layout, 2)
    st.interface("rate_e")  # shared prefix: fine
    with pytest.raises(SpanViolation, match="no whole-brain view"):
        st.interface("k")


def test_padded_round_trip_is_exact(layout):
    torch.manual_seed(0)
    st = FamilyState(layout, {f.name: torch.randn(2, f.n_regions, f.dim) for f in layout})
    back = FamilyState.from_padded(layout, st.to_padded())
    for f in layout:
        assert torch.equal(st[f.name], back[f.name])


def test_out_ports_still_typed(layout):
    st = FamilyState.zeros(layout, 2)
    assert st.port("subcortex_hippo", "recall").shape[-1] == 17  # v(16)+rho(1)
    with pytest.raises(PortMismatch, match="is an in-port"):
        st.port("subcortex_hippo", "cue")


def test_segment_layout_compiles_fullgraph(layout):
    """The design claim, measured rather than asserted.

    ``nested_tensor`` fails ``fullgraph=True`` outright ("torch.compile does not
    support strided NestedTensor"); per-family dense blocks have static shapes,
    so dynamo sees ordinary dense ops and unrolls the small family loop.
    """
    torch.manual_seed(0)
    W = {f.name: torch.randn(f.dim, f.dim) * 0.05 for f in layout}

    def step(blocks):
        return {n: b + torch.tanh(b @ W[n]) for n, b in blocks.items()}

    blocks = {f.name: torch.randn(2, f.n_regions, f.dim, requires_grad=True) for f in layout}
    eager = step(blocks)
    out = torch.compile(step, fullgraph=True)(blocks)
    for k in out:
        assert torch.allclose(out[k], eager[k], atol=1e-5)
    sum(v.sum() for v in out.values()).backward()
    assert all(b.grad is not None for b in blocks.values())


# ======================================================================
# O-5: orientation
# ======================================================================
def test_every_family_declares_a_dipole_and_only_cortex_has_an_orientation(layout):
    """Renamed and inverted by O-5b, deliberately.

    This asserted that ONLY cortical families declare a dipole, which was true
    and was the defect: a component declared per-family lives in the ``private``
    block, and ``SCWBD.build_layout`` forbids an observation head from addressing
    that. So ``EEGHead.source_moment()`` returned ``None`` for the whole of run 2
    while the ``(64, 414, 3)`` lead field sat ready for it.

    ``dipole`` is now in the shared interface at a fixed offset, so **every**
    family declares it and subcortex carries the zero vector. The physical claim
    the old test was protecting has not gone away — it moved to where it belongs,
    the orientation:

    * absent **orientation** must stay ``NaN``; a direction of zero length is a
      lie and would tilt the field;
    * absent **moment** genuinely *is* zero, and contributes exactly zero
      through ``L_vec``.

    Asserting both is strictly stronger than the original, which only checked
    where the component was declared.
    """
    import numpy as np

    for f in layout:
        assert "dipole" in f.layout, (
            f"{f.name} does not declare a dipole. Since O-5b it is part of the "
            "shared interface, so a family missing it means the shared prefix is "
            "not shared and the heads read different quantities per region."
        )

    an = load_anatomy()
    normal = np.asarray(an.normal)
    sub = [i for i, d in enumerate(an.division) if d != "cortex"]
    assert np.all(np.isnan(normal[sub])), (
        "a subcortical region gained a cortical normal. The zero-moment argument "
        "depends on there being no orientation to have; filling these NaNs is the "
        "imputation ARCHITECTURE.md §7 rule 1 forbids."
    )
    # and it is three numbers, in Hz*m -- a moment, not a rate
    c = layout.family("cortex_unimodal").layout.spec("dipole")
    assert c.dim == 3 and c.units == "Hz*m"


def test_the_dipole_port_is_a_moment_and_cannot_be_wired_to_a_rate(layout):
    f = layout.family("cortex_unimodal")
    assert f.port("dipole_out").units == "Hz*m"
    table = {r["port"]: r["units"] for r in layout.check_ports()}
    assert table["dipole_out"] == "Hz*m"
    assert table["activity"] == "Hz"
    # units-matched routing must not offer a rate source to the moment sink
    rows = {(r["sink_family"], r["in_port"]): r for r in layout.routing_table()}
    induced = rows[("cortex_unimodal", "induced_field")]
    assert induced["units"] == "Hz*m"
    assert all("Hz*m" == table[p] for p in ["dipole_out"])


def test_uncovered_parcels_contribute_exactly_zero(anat):
    """14 subcortical parcels carry NaN in `normal`; none of it may reach a gradient."""
    proj = DipoleProjection(anat.normal, anat.normal_coherence, anat.normal_covered)
    assert torch.isfinite(proj.normal).all() and torch.isfinite(proj.coherence).all()
    m = torch.randn(3, anat.n_regions, 3)
    s = proj(m)
    assert torch.isfinite(s).all()
    assert float(s.detach()[..., ~anat.normal_covered].abs().max()) == 0.0


def test_a_nan_normal_cannot_survive_construction(anat):
    """The hazard: one unguarded multiply propagates NaN to every channel."""
    n = anat.normal.clone()
    assert torch.isnan(n).any(), "the real prior is expected to carry NaN on uncovered parcels"
    proj = DipoleProjection(n, anat.normal_coherence, anat.normal_covered)
    assert not torch.isnan(proj.normal).any()


def test_coherence_is_load_bearing_not_decorative(anat):
    """A bare unit normal makes every parcel look equally observable.  It is not."""
    proj = DipoleProjection(anat.normal, anat.normal_coherence, anat.normal_covered)
    cov = anat.normal_covered
    # every covered normal has length exactly 1 -- so the normal alone carries
    # no information about how much of the parcel survives cancellation
    assert torch.allclose(proj.normal[cov].norm(dim=-1), torch.ones(int(cov.sum())), atol=1e-5)
    c = proj.coherence[cov]
    assert float(c.min()) < 0.3 and float(c.max()) > 0.99
    assert int((c < 0.5).sum()) > 0, "if no parcel is below 0.5, coherence is not doing work here"
    # and dropping it changes the answer materially
    m = torch.randn(2, anat.n_regions, 3)
    with_c = proj(m)
    without_c = (m * proj.normal).sum(-1) * proj.log_gain.exp()
    rel = (with_c - without_c).abs().sum() / without_c.abs().sum().clamp_min(1e-12)
    assert float(rel) > 0.1, f"ignoring coherence changes the projection by only {float(rel):.3f}"


def test_coherence_is_a_fraction_not_a_gain(anat):
    with pytest.raises(ValueError, match=r"must lie in \[0,1\]"):
        DipoleProjection(anat.normal, anat.normal_coherence * 3.0, anat.normal_covered)


def test_lift_then_project_is_coherence_squared(anat):
    """Attenuated going in and coming out.  Physics, and why they must not compose casually."""
    proj = DipoleProjection(anat.normal, anat.normal_coherence, anat.normal_covered, learn_gain=False)
    s = torch.randn(2, anat.n_regions)
    assert torch.allclose(proj(proj.lift(s)), s * proj.coherence**2, atol=1e-5)


def test_normals_and_coherence_are_buffers_not_parameters(anat):
    """Fitting them would let the model explain a bad forward solution by rotating cortex."""
    proj = DipoleProjection(anat.normal, anat.normal_coherence, anat.normal_covered)
    names = {n for n, _ in proj.named_parameters()}
    assert names == {"log_gain"}
    assert "normal" in dict(proj.named_buffers()) and "coherence" in dict(proj.named_buffers())


def test_orientation_energy_is_reported(anat):
    from scwbd.foundation.heads import build_lead_field

    proj = DipoleProjection(anat.normal, anat.normal_coherence, anat.normal_covered)
    rep = proj.expressible_fraction(build_lead_field(anat, device="cpu").matrix)
    assert rep["n_covered"] == 400 and rep["n_regions"] == 414
    assert 0.0 < rep["energy_retained_by_coherence"] < 1.0
    assert rep["below_half"] == 23


# ======================================================================
# O-5, second half: the observation operator must consume the vector
# ======================================================================
def test_the_vector_lead_field_has_a_third_axis(anat):
    from scwbd.foundation.orientation import build_vector_lead_field

    V = build_vector_lead_field(anat)
    assert V.L.shape == (64, anat.n_regions, 3)
    m = torch.randn(2, 3, anat.n_regions, 3)
    assert V(m).shape == (2, 3, 64)


def test_a_scalar_matrix_is_refused_as_a_vector_lead_field(anat):
    from scwbd.foundation.heads import build_lead_field
    from scwbd.foundation.orientation import VectorLeadField

    scalar = build_lead_field(anat, device="cpu").matrix
    with pytest.raises(ValueError, match="n_channels, n_regions, 3"):
        VectorLeadField(scalar, ("a",) * 64)


def test_contraction_reproduces_the_scalar_operator_and_loses_energy(anat):
    """The loss becomes a named step with a measured size."""
    from scwbd.foundation.orientation import build_vector_lead_field

    V = build_vector_lead_field(anat)
    s = V.contract(anat.normal, anat.normal_coherence)
    assert s.shape == (64, anat.n_regions)
    hr = V.orientation_headroom(anat.normal, anat.normal_coherence)
    assert hr["fraction_retained_by_contraction"] < 1.0
    assert hr["headroom_multiple"] > 1.5
    assert hr["dof_vector"] == 3 * hr["dof_scalar"]


def test_rank_is_capped_by_the_montage_not_by_orientation(anat):
    """Bounds what O-5 can deliver, and is the thing most likely to be overclaimed.

    64 electrodes cap the observable rank at 64 whether the operator consumes
    one number per parcel or three.  Orientation does not buy degrees of
    freedom here; it buys a better-aligned and better-conditioned 64-dimensional
    subspace.  Any claim that it multiplies resolvable sources is false.
    """
    from scwbd.foundation.orientation import build_vector_lead_field

    V = build_vector_lead_field(anat)
    hr = V.orientation_headroom(anat.normal, anat.normal_coherence)
    assert hr["rank_vector"] == hr["rank_scalar"] == 64
