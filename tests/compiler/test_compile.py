"""The compiled artifact: memory map, masks, dispatch, schedule, provenance."""

from __future__ import annotations

import pytest
import torch

from scwbd.compiler import EVIDENCE_CLASSES, compile, resolve_operator_clock
from scwbd.schema.examples import build_three_region_claim, build_three_region_schema
from scwbd.schema.examples.three_region import DELTA_BOLD, DELTA_EEG, REGION_IDS


@pytest.fixture(scope="module")
def model():
    return compile(build_three_region_schema(), claim=build_three_region_claim())


# ---------------------------------------------------------------------------
# state layout / ABI
# ---------------------------------------------------------------------------
def test_layout_covers_every_component_exactly_once(model):
    layout = model.state_layout
    expected = sum(len(r.state.components) for r in model.schema.regions)
    assert len(layout) == expected
    assert layout.total_elements == model.schema.numel()
    assert layout.total_bytes >= model.schema.nbytes()


def test_slices_are_contiguous_and_non_overlapping(model):
    layout = model.state_layout
    cursor = 0
    for entry in layout.entries:
        assert entry.elem_offset == cursor
        cursor += entry.numel
    assert cursor == layout.total_elements


def test_slice_of_and_offset_of(model):
    layout = model.state_layout
    s = layout.slice_of("region_a", "latent")
    assert isinstance(s, slice)
    assert s.stop - s.start == 1
    elem, byte = layout.offset_of("region_a", "latent")
    assert elem == s.start
    assert byte % 4 == 0  # float32 alignment
    with pytest.raises(KeyError):
        layout.slice_of("region_a", "no_such_component")


def test_regions_are_contiguous_blocks(model):
    layout = model.state_layout
    for rid in REGION_IDS:
        block = layout.region_slice(rid)
        entries = layout.of_region(rid)
        assert block.start == entries[0].elem_offset
        assert block.stop == entries[-1].elem_offset + entries[-1].numel


def test_abi_digest_is_stable_and_sensitive(model):
    other = compile(build_three_region_schema(), claim=build_three_region_claim())
    assert model.state_layout.abi_digest() == other.state_layout.abi_digest()
    assert model.state_layout.boundary_abi_digest() != model.state_layout.abi_digest()
    # Boundary contract is a strict subset of the full state.
    assert 0 < len(model.state_layout.boundary_entries()) < len(model.state_layout)


def test_boundary_state_is_the_exchange_contract(model):
    boundary = {e.qualified for e in model.state_layout.boundary_entries()}
    assert "region_a.latent" in boundary
    assert "region_a.uncertainty" not in boundary  # private


def test_compiled_model_content_hash_is_stable(model):
    other = compile(build_three_region_schema(), claim=build_three_region_claim())
    assert model.content_hash() == other.content_hash()


# ---------------------------------------------------------------------------
# adjacency
# ---------------------------------------------------------------------------
def test_adjacency_masks_are_separate_per_evidence_class(model):
    adj = model.adjacency
    assert adj.region_ids == REGION_IDS
    hard = adj.dense("hard")
    soft = adj.dense("soft")
    proposed = adj.dense("proposed")
    assert hard.shape == (3, 3)
    assert hard.dtype == torch.bool
    # a->b and b->c are hard; c->a is soft; b->a residual is proposed.
    assert hard[adj.index_of("region_a"), adj.index_of("region_b")]
    assert soft[adj.index_of("region_c"), adj.index_of("region_a")]
    assert proposed[adj.index_of("region_b"), adj.index_of("region_a")]
    assert not hard[adj.index_of("region_c"), adj.index_of("region_a")]


def test_adjacency_indices_align_with_edge_keys(model):
    adj = model.adjacency
    for cls in EVIDENCE_CLASSES:
        assert adj.indices[cls].shape[1] == len(adj.edge_keys[cls])
        assert adj.indices[cls].dtype == torch.int64


def test_proposed_edges_carry_a_penalty(model):
    adj = model.adjacency
    assert set(adj.penalties) == set(adj.edge_keys["proposed"])
    assert adj.total_penalty() > 0.0
    assert not adj.penalties.keys() & set(adj.edge_keys["hard"])


def test_absent_edges_are_exact_zeros(model):
    """A zero mask is exact independence across that direct edge."""
    adj = model.adjacency
    combined = adj.combined_dense()
    assert not combined[adj.index_of("region_a"), adj.index_of("region_c")]


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def test_dispatch_instantiates_every_operator(model):
    assert len(model.dispatch) == len(model.schema.operators)
    assert set(model.dispatch.keys()) == {op.key for op in model.schema.operators}


def test_dispatch_resolves_slices_against_the_layout(model):
    d = model.dispatch["couple_a_b"]
    assert d.src == "region_a" and d.dst == "region_b"
    assert d.src_slices["latent"] == model.state_layout.slice_of("region_a", "latent")
    assert d.family == "delayed_ssm"
    assert d.clock == "sim"


def test_unknown_delay_is_carried_as_a_prior_not_a_number(model):
    d = model.dispatch["couple_a_b"]
    assert d.delay_prior.kind == "lognormal"
    assert d.delay_prior.units.same_dimension("s")
    assert d.delay_seconds() > 0.0
    assert model.dispatch.max_delay() >= d.delay_seconds()


def test_operator_clock_defaults_to_the_destination_fastest_clock():
    schema = build_three_region_schema()
    op = schema.operator("couple_b_c").model_copy(update={"clock": None})
    assert resolve_operator_clock(schema, op) == "sim"


def test_learned_operators_are_identifiable(model):
    learned = model.dispatch.learned()
    assert [d.key for d in learned] == ["residual_b_a"]


# ---------------------------------------------------------------------------
# multirate schedule
# ---------------------------------------------------------------------------
def test_every_field_gets_a_period(model):
    sched = model.schedule
    assert len(sched.policies) == len(model.state_layout)
    assert sched.policy("region_a", "latent").period == pytest.approx(DELTA_EEG)
    assert sched.policy("region_a", "hemodynamic").period == pytest.approx(DELTA_BOLD)


def test_hyperperiod_and_sync_points_are_exact(model):
    sched = model.schedule
    assert sched.hyperperiod == pytest.approx(1.0)
    assert sched.base_dt == pytest.approx(1e-3)
    # 1 ms and 1 s coincide once per hyperperiod.
    assert sched.sync_points == (1.0,)


def test_step_plan_is_ordered_and_covers_the_fast_clock(model):
    events = model.schedule.step_plan(0.0, 0.005)
    times = [e.t_ns for e in events]
    assert times == sorted(times)
    assert len(events) == 5  # 5 ticks of 1 ms in (0, 5 ms]
    assert all("region_a.latent" in e.fields for e in events)
    # the slow field does not appear at all inside a 5 ms window
    assert not any("region_a.hemodynamic" in e.fields for e in events)
    assert not any(e.is_sync for e in events)


def test_step_plan_marks_the_tick_where_clocks_coincide(model):
    events = model.schedule.step_plan(0.999, 1.001)
    sync = [e for e in events if e.is_sync]
    assert [e.t for e in sync] == [1.0]
    assert set(sync[0].clocks) == {"sim", "scanner_volume"}


def test_step_plan_does_not_drift_over_a_long_window(model):
    events = model.schedule.step_plan(0.0, 2.0)
    slow = [e for e in events if "region_a.hemodynamic" in e.fields]
    assert [e.t for e in slow] == [1.0, 2.0]  # exact, no float accumulation
    assert len(events) == 2000


def test_interpolation_contracts_exist_for_crossed_clocks(model):
    contract = model.schedule.contract("sim", "scanner_volume")
    assert contract is not None
    assert contract.period_ratio == pytest.approx(1000.0)
    assert contract.is_downsampling
    assert contract.method in ("zero_order_hold", "linear", "band_limited", "event_exact")


def test_event_driven_fields_are_listed(model):
    # This example is purely periodic; the field must exist and be empty rather
    # than absent, so agent E can rely on it.
    assert model.schedule.event_driven_fields == ()


# ---------------------------------------------------------------------------
# gradient masks
# ---------------------------------------------------------------------------
def test_masks_exist_per_source_card(model):
    masks = model.gradient_masks
    assert masks.keys() == ("eeg_sim_v1", "fmri_sim_v1", "impulse_sim_v1")
    assert masks.matrix().shape == (3, len(masks.group_names))
    assert masks.matrix().dtype == torch.bool


def test_a_source_updates_only_what_it_names(model):
    """Rule 2: a source updates only the modules its permission names."""
    eeg = model.gradient_masks["eeg_sim_v1"]
    fmri = model.gradient_masks["fmri_sim_v1"]
    assert eeg.allows("operator:couple_a_b:params")
    assert eeg.allows("region:region_a:state:latent")
    assert not fmri.allows("operator:couple_a_b:params")
    assert fmri.allows("region:region_a:state:hemodynamic")
    assert not eeg.allows("region:region_a:state:hemodynamic")


def test_frozen_paths_beat_grants(model):
    eeg = model.gradient_masks["eeg_sim_v1"]
    frozen = [g for g in eeg.group_names if g.startswith("frame_edge:")]
    assert frozen
    assert all(not eeg.allows(g) for g in frozen)


def test_calibration_and_transform_groups_are_not_silently_trained(model):
    """No source may update the frame graph in this example."""
    for group in model.gradient_masks.group_names:
        if group.startswith("frame_edge:"):
            assert model.gradient_masks.sources_updating(group) == ()


def test_unknown_group_raises_rather_than_defaulting(model):
    with pytest.raises(KeyError):
        model.gradient_masks["eeg_sim_v1"].allows("operator:ghost:params")


# ---------------------------------------------------------------------------
# frame / clock graphs
# ---------------------------------------------------------------------------
def test_frame_paths_are_edge_lists_not_composed_transforms(model):
    path = model.frame_graph.path("scanner_RAS", "eeg_cap")
    assert path is not None
    assert len(path) == 3
    assert all(hasattr(edge, "calibration") for edge, _ in path)
    assert model.frame_graph.path_is_valid("scanner_RAS", "eeg_cap")


def test_frame_paths_use_inverses_only_where_declared(model):
    path = model.frame_graph.path("eeg_cap", "scanner_RAS")
    assert path is not None
    assert all(inverted for _, inverted in path)


def test_unreachable_frame_returns_none(model):
    assert model.frame_graph.path("scanner_RAS", "nonexistent") is None


def test_clock_graph_maps_between_clocks(model):
    cg = model.clock_graph
    assert cg.master == "sim"
    assert cg.unverified() == ()
    assert cg.orphans() == ()
    assert cg.chain_to_master("eeg_amp") == ("eeg_amp", "sim")
    assert cg.between("eeg_amp", "scanner_volume", 2.0) == pytest.approx(2.0)


def test_clock_offset_and_drift_round_trip():
    from scwbd.schema import ClockSpec

    c = ClockSpec(id="drifty", dt=1e-3, reference="sim", offset=0.5, drift=1e-4)
    assert c.from_reference(c.to_reference(3.0)) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# ledger and provenance
# ---------------------------------------------------------------------------
def test_ledger_keeps_bias_and_variance_apart(model):
    ledger = model.ledger
    assert len(ledger) > 0
    assert set(ledger.bias_status_counts) <= {
        "design_estimable",
        "externally_bounded",
        "prior_specified_sensitivity",
    }
    assert "measurement" in ledger.variance_totals
    assert ledger.unbacked_bias == ()  # else R08 would have fired
    assert ledger.sensitivity_terms()  # the example does carry swept terms
    assert "variance_totals" in ledger.as_dict()
    assert "bias_status_counts" in ledger.as_dict()


def test_provenance_records_what_was_checked(model):
    p = model.provenance
    assert p.schema_hash == model.schema.content_hash()
    assert p.requested_claim_class == "effective"
    assert p.effective_claim_class == "effective"
    assert not p.claim_was_demoted
    assert len(p.checks_passed) == 11
    assert p.extra["abi_digest"] == model.state_layout.abi_digest()
    assert p.disabling_evidence  # ARCHITECTURE sec. 4 requires this statement


def test_provenance_serializes(model):
    text = model.provenance.to_json()
    assert '"effective_claim_class":"effective"' in text


def test_summary_is_informative(model):
    text = model.summary()
    for fragment in ("StateLayout", "Adjacency", "Dispatch", "Multirate", "Gradient"):
        assert fragment in text


def test_compile_rejects_wrong_argument_types():
    with pytest.raises(TypeError):
        compile("not a schema", claim=build_three_region_claim())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        compile(build_three_region_schema(), claim="not a claim")  # type: ignore[arg-type]
