"""O-5 (vector-valued regional state) and O-6 (segment layout).

O-6 retires narrowing N-1 by making its failure mode unrepresentable rather than
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
    # the number N-1 was costing
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
def test_only_cortical_families_declare_a_dipole(layout):
    for f in layout:
        has = "dipole" in f.layout
        assert has == f.name.startswith("cortex_"), f.name
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
