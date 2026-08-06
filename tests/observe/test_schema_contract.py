"""Every observation head must emit contract objects the compiler accepts.

Agent A operationalised refusal R08 as a machine-checkable constraint on
``scwbd.schema.UncertaintyLedger``:

* ``design_estimable``            -> must name a ``bias_estimator``;
* ``externally_bounded``          -> must name an ``external_bound_source``;
* ``prior_specified_sensitivity`` -> must carry a non-degenerate interval.

``UncertaintyLedger.has_estimator()`` is the predicate ``check_r08`` uses, so
these tests assert exactly that predicate on every ledger this package emits,
for every unit group it emits one in.  They also assert that the supports and
clocks round-trip into the schema's types, so a head can be wired into a
``SourceCard`` without an adapter shim written at the call site.
"""

from __future__ import annotations

import pytest
import torch

from scwbd.observe.base import (
    BiasTerm,
    Provenance,
    RefusalR08,
    UncertaintyLedger,
    VarianceDecomposition,
    to_schema_ledger,
    to_schema_ledgers,
    to_schema_support,
)

from .test_ledger import _all_reads

schema = pytest.importorskip("scwbd.schema")
torch.set_default_dtype(torch.float64)


def test_every_emitted_ledger_satisfies_r08(
    four_layer_head, sensor_positions, source_positions, latent_temporal
):
    reads = _all_reads(four_layer_head, sensor_positions, source_positions, latent_temporal)
    for name, r in reads.items():
        groups = r.ledger.to_schema_all()
        assert groups, f"{name}: emitted no schema ledger at all"
        for units, led in groups.items():
            assert isinstance(led, schema.UncertaintyLedger)
            assert led.has_estimator(), (
                f"{name}[{units}]: bias_status={led.bias_status!r} with "
                f"interval {led.bias_interval}, estimator={led.bias_estimator!r}, "
                f"bound={led.external_bound_source!r} -- the compiler would "
                "refuse this ledger with R08"
            )
            if led.bias_status == "design_estimable":
                assert led.bias_estimator
            elif led.bias_status == "externally_bounded":
                assert led.external_bound_source
            else:
                assert not led.is_bias_point_estimate


def test_r08_is_enforced_by_the_compilers_own_check(
    four_layer_head, sensor_positions, source_positions, latent_temporal
):
    """Run agent A's predicate, not a re-implementation of it."""
    reads = _all_reads(four_layer_head, sensor_positions, source_positions, latent_temporal)
    for name, r in reads.items():
        for units, led in r.ledger.to_schema_all().items():
            status = led.bias_status
            backed = (
                bool(led.bias_estimator)
                if status == "design_estimable"
                else bool(led.external_bound_source)
                if status == "externally_bounded"
                else not led.is_bias_point_estimate
            )
            assert backed is led.has_estimator() is True, f"{name}[{units}]"


def test_unknown_variance_components_are_omitted_never_zeroed(
    four_layer_head, sensor_positions, source_positions, latent_temporal
):
    """A component nobody estimated must not appear as a confident 0.0."""
    reads = _all_reads(four_layer_head, sensor_positions, source_positions, latent_temporal)
    for name, r in reads.items():
        led = r.ledger.to_schema()
        for comp in r.ledger.variance.unknown_components:
            key = "parameter" if comp == "parameter_posterior" else comp
            assert key not in led.variance, (
                f"{name}: unidentified component {key!r} was written into the "
                "schema variance dict"
            )
            assert key in led.notes
        assert set(led.variance) <= set(schema.VARIANCE_COMPONENTS)
        assert all(v >= 0.0 for v in led.variance.values())


def test_bias_terms_of_different_units_are_never_summed_into_one_interval():
    """Metres, seconds and volts are different claims."""
    led = UncertaintyLedger(
        variance=VarianceDecomposition(units="V^2"),
        bias=(
            BiasTerm("coreg", (-0.005, 0.005), "design_estimable", units="m",
                     estimator="ICP residual"),
            BiasTerm("timing", (-0.01, 0.01), "design_estimable", units="s",
                     estimator="TTL sync log"),
            BiasTerm("gain", (-1e-6, 1e-6), "design_estimable", units="V",
                     estimator="phantom injection"),
        ),
        provenance=Provenance(operator="t"),
        validity_domain={"x": 1},
    )
    groups = to_schema_ledgers(led)
    assert set(groups) == {"m", "s", "V"}
    assert groups["m"].bias_interval == (-0.005, 0.005)
    assert groups["s"].bias_interval == (-0.01, 0.01)
    assert groups["V"].bias_interval == (-1e-6, 1e-6)
    for g in groups.values():
        assert g.has_estimator()


def test_supports_and_clocks_round_trip_into_the_schema(
    four_layer_head, sensor_positions, source_positions, latent_temporal
):
    reads = _all_reads(four_layer_head, sensor_positions, source_positions, latent_temporal)
    for name, r in reads.items():
        s = r.to_schema_support()
        assert isinstance(s, schema.Support)
        assert str(s.units) == r.units
        assert str(s.frame) == r.support.frame
        if r.support.psf is not None:
            assert s.psf is not None, f"{name}: support lost its point-spread"
            assert not s.is_nominal, (
                f"{name}: schema support is nominal -- a label, not a physical "
                "support (thesis Sec. 2.8)"
            )
        t = r.to_schema_temporal()
        assert isinstance(t, schema.TemporalSupport)
        assert t.dt == pytest.approx(r.temporal.dt)
        assert str(t.clock) == r.temporal.clock


def test_eeg_support_arrives_as_a_lead_field_psf(
    four_layer_head, sensor_positions, source_positions, latent_temporal
):
    from scwbd.observe.eeg import EEGObservationOperator

    normals = source_positions / source_positions.norm(dim=-1, keepdim=True)
    lf = four_layer_head.lead_field(source_positions, sensor_positions).project(normals)
    op = EEGObservationOperator(lf, dt=1e-3, dtype=torch.float64)
    s = to_schema_support(op.support)
    assert s.psf is not None
    assert s.psf.kind == "lead_field"
    assert s.psf.kernel_ref, "the lead field must be referenced, not dropped"
    assert str(s.psf.units) == "V/(A*m)"
    assert s.psf.nominal is False


def test_adapter_refuses_to_emit_a_degenerate_swept_interval():
    """R08 at the adapter boundary, not only at BiasTerm construction."""
    led = UncertaintyLedger(
        variance=VarianceDecomposition(units="V^2"),
        bias=(
            BiasTerm("coreg", (-0.005, 0.005), "design_estimable", units="m",
                     estimator="ICP residual"),
        ),
        provenance=Provenance(operator="t"),
        validity_domain={"x": 1},
    )
    # asking for volts, where there is no bias term at all, must not silently
    # produce a zero-width "prior specified" interval
    with pytest.raises(RefusalR08):
        to_schema_ledger(led, units="V")


def test_lead_field_ledgers_are_compliant(
    four_layer_head, sensor_positions, source_positions
):
    lf = four_layer_head.lead_field(
        source_positions, sensor_positions, n_conductivity_draws=4, seed=1
    )
    assert lf.ledger is not None
    for units, led in lf.ledger.to_schema_all().items():
        assert led.has_estimator(), f"lead field ledger[{units}] fails R08"


def test_inverse_solution_ledger_is_compliant(four_layer_head, sensor_positions):
    from scwbd.observe.inverse import solve_inverse

    g = torch.Generator().manual_seed(4)
    pos = torch.randn((60, 3), generator=g, dtype=torch.float64)
    pos = pos / pos.norm(dim=-1, keepdim=True) * 0.05
    lf = four_layer_head.lead_field(pos, sensor_positions, dtype=torch.float64).project(
        pos / pos.norm(dim=-1, keepdim=True)
    )
    y = lf.as_matrix().to(torch.float64) @ (1e-8 * torch.randn((60, 5), dtype=torch.float64))
    nc = (1e-6**2) * torch.eye(lf.n_sensors, dtype=torch.float64)
    sol = solve_inverse(y, lf, nc, method="dSPM")
    for units, led in sol.ledger.to_schema_all().items():
        assert led.has_estimator(), f"inverse ledger[{units}] fails R08"
