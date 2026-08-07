"""Ledger completeness and refusal R08.

thesis Sec. 2.7: "Each read returns a prediction, variance decomposition,
estimated bias range, model-discrepancy flag, provenance, and validity domain."
The six variance components must each be a number or an explicit ``"unknown"``;
a silent zero for an unestimated component is the failure mode these tests
exist to prevent.

thesis_contract.tex Sec. 0.1 / ARCHITECTURE.md R08: "Bias assigned a point
estimate without an estimator or external bound" is a hard error.
"""

from __future__ import annotations

import math

import pytest
import torch

from scwbd.observe.base import (
    UNKNOWN,
    BiasTerm,
    ObservationRefusal,
    Provenance,
    RefusalR08,
    UncertaintyLedger,
    VarianceDecomposition,
)

# The module-level `torch.set_default_dtype(float64)` that stood here ran at
# COLLECTION time and changed the default for the entire process. Owned by the
# autouse fixture in conftest.py instead; set DEFAULT_DTYPE here to override.

REQUIRED_COMPONENTS = (
    "measurement",
    "within_session",
    "between_session",
    "parameter_posterior",
    "model_class",
    "numerical",
)


# --------------------------------------------------------------------------
# every head's read
# --------------------------------------------------------------------------


def _all_reads(four_layer_head, sensor_positions, source_positions, latent_temporal):
    from scwbd.observe.behavior import (
        BehaviorObservationOperator,
        MotorStage,
        PerceptionStage,
        PolicyStage,
        ReportingBias,
    )
    from scwbd.observe.bold import (
        BOLDObservationOperator,
        BalloonWindkesselReadout,
        CanonicalHRF,
        DriftModel,
        MotionModel,
        PartialVolume,
        PhysiologicalNoise,
        SliceTiming,
        reference_balloon_windkessel,
    )
    from scwbd.observe.eeg import ArtifactModel, EEGObservationOperator
    from scwbd.observe.fnirs import FNIRSObservationOperator
    from scwbd.observe.leadfield import (
        ElectrodeImpedance,
        ReferenceOperator,
        meg_lead_field,
    )

    reads = {}
    normals = source_positions / source_positions.norm(dim=-1, keepdim=True)
    lf = four_layer_head.lead_field(source_positions, sensor_positions).project(normals)
    n_src = source_positions.shape[0]

    eeg = EEGObservationOperator(
        lf,
        dt=1e-3,
        reference=ReferenceOperator.average(lf.n_sensors, dtype=torch.float64),
        impedance=ElectrodeImpedance(
            z_electrode=torch.linspace(2e3, 2e4, lf.n_sensors, dtype=torch.float64)
        ),
        artifacts=ArtifactModel(),
        dtype=torch.float64,
    )
    q = 1e-9 * torch.randn((n_src, 3000), dtype=torch.float64)
    reads["eeg"] = eeg.observe(q, latent_temporal, seed=0)

    g = torch.Generator().manual_seed(2)
    mpos = torch.randn(16, 3, generator=g, dtype=torch.float64)
    mpos = mpos / mpos.norm(dim=1, keepdim=True) * 0.115
    meg_lf = meg_lead_field(
        source_positions, mpos, mpos / mpos.norm(dim=1, keepdim=True), dtype=torch.float64
    ).project(normals)
    meg = EEGObservationOperator(meg_lf, dt=1e-3, clock="meg_amp", dtype=torch.float64)
    reads["meg"] = meg.observe(q, latent_temporal, seed=0)

    latent = torch.randn((n_src, 30000), dtype=torch.float64)
    bold = BOLDObservationOperator(
        n_elements=n_src,
        tr=1.0,
        hrf=CanonicalHRF(),
        slice_timing=SliceTiming.interleaved(n_src, 1.0),
        physio=PhysiologicalNoise(),
        motion=MotionModel(),
        drift=DriftModel(),
        partial_volume=PartialVolume(
            gm_fraction=torch.full((n_src,), 0.6, dtype=torch.float64)
        ),
        dtype=torch.float64,
    )
    reads["bold"] = bold.observe(latent, latent_temporal, seed=0, n_rho_draws=4)

    drive = torch.zeros((3, 20000), dtype=torch.float64)
    drive[:, 2000:5000] = 1.0
    bw = BOLDObservationOperator(
        n_elements=3, tr=1.0, hrf=None, balloon=BalloonWindkesselReadout(),
        dtype=torch.float64,
    )
    reads["bold_balloon"] = bw.observe_hemodynamic_state(
        reference_balloon_windkessel(drive, dt=1e-3), latent_temporal, seed=0
    )

    nirs = FNIRSObservationOperator(
        torch.tensor([8.0, 30.0, 30.0, 35.0], dtype=torch.float64),
        dt=0.1,
        dtype=torch.float64,
    )
    hb = {
        "HbO": torch.randn((4, 30000), dtype=torch.float64) * 0.5,
        "HbR": torch.randn((4, 30000), dtype=torch.float64) * 0.2,
    }
    reads["fnirs"] = nirs.observe(hb, latent_temporal, seed=0)

    beh = BehaviorObservationOperator(
        perception=PerceptionStage(),
        policy=PolicyStage(),
        motor=MotorStage(),
        reporting=ReportingBias(
            response_bias=0.1,
            estimator="payoff-matrix manipulation block with matched sensitivity",
        ),
        dtype=torch.float64,
    )
    ev = torch.linspace(-1.0, 1.0, 60, dtype=torch.float64)
    reads["behavior"] = beh.observe(ev, seed=0)
    return reads


def test_every_read_carries_all_six_variance_components(
    four_layer_head, sensor_positions, source_positions, latent_temporal
):
    reads = _all_reads(four_layer_head, sensor_positions, source_positions, latent_temporal)
    for name, r in reads.items():
        v = r.ledger.variance
        d = v.as_dict()
        assert set(d) == set(REQUIRED_COMPONENTS), f"{name}: wrong component set"
        for comp, val in d.items():
            assert isinstance(val, (int, float)) or val == UNKNOWN, (
                f"{name}.{comp} is neither a number nor an explicit 'unknown': {val!r}"
            )
            if isinstance(val, (int, float)):
                assert math.isfinite(float(val)) and float(val) >= 0.0
        assert v.is_complete(), f"{name}: incomplete variance decomposition"


def test_every_read_carries_bias_provenance_and_validity_domain(
    four_layer_head, sensor_positions, source_positions, latent_temporal
):
    reads = _all_reads(four_layer_head, sensor_positions, source_positions, latent_temporal)
    for name, r in reads.items():
        led = r.ledger
        assert led.provenance is not None, f"{name}: no provenance"
        assert led.provenance.operator, f"{name}: unnamed operator"
        assert led.provenance.references, f"{name}: no references"
        assert led.validity_domain, f"{name}: empty validity domain"
        assert "claim_boundary" in led.validity_domain, (
            f"{name}: no claim boundary -- the read does not say what it is NOT"
        )
        assert led.bias, f"{name}: no bias terms at all"
        assert led.bias_status in (
            "design_estimable",
            "externally_bounded",
            "prior_specified_sensitivity",
        )
        lo, hi = led.bias_interval
        assert lo <= hi
        assert led.is_complete(), f"{name}: ledger fails the completeness gate"


def test_unknown_variance_components_are_reported_not_hidden(
    four_layer_head, sensor_positions, source_positions, latent_temporal
):
    reads = _all_reads(four_layer_head, sensor_positions, source_positions, latent_temporal)
    for name, r in reads.items():
        v = r.ledger.variance
        # single-session reads cannot estimate between-session variance
        assert "between_session" in v.unknown_components, (
            f"{name} claims a between-session variance from one session"
        )
        if not v.is_fully_quantified():
            with pytest.raises(ObservationRefusal) as exc:
                _ = v.total
            assert exc.value.code == "R08"
            assert v.known_total >= 0.0


# --------------------------------------------------------------------------
# R08
# --------------------------------------------------------------------------


def test_r08_bias_point_estimate_without_backing_raises():
    with pytest.raises(RefusalR08) as exc:
        BiasTerm(name="atlas_mismatch", interval=(0.003, 0.003), status="design_estimable")
    assert exc.value.code == "R08"

    with pytest.raises(RefusalR08):
        BiasTerm(name="atlas_mismatch", interval=(0.003, 0.003), status="externally_bounded")

    with pytest.raises(RefusalR08):
        BiasTerm(
            name="atlas_mismatch",
            interval=(0.003, 0.003),
            status="prior_specified_sensitivity",
        )


def test_r08_admits_a_point_only_with_an_estimator_or_an_external_bound():
    ok = BiasTerm(
        name="device_gain",
        interval=(0.02, 0.02),
        status="design_estimable",
        estimator="repeated phantom measurement within session, n=20",
    )
    assert ok.half_width == 0.0
    ok2 = BiasTerm(
        name="conductivity",
        interval=(-0.1, -0.1),
        status="externally_bounded",
        external_bound="skull phantom with known conductivity",
    )
    assert ok2.midpoint == pytest.approx(-0.1)


def test_r08_prior_specified_sensitivity_must_be_a_range():
    swept = BiasTerm(
        name="neurovascular_coupling",
        interval=(-0.3, 0.3),
        status="prior_specified_sensitivity",
        sensitivity_grid=(-0.3, 0.0, 0.3),
    )
    assert swept.half_width > 0.0
    with pytest.raises(RefusalR08):
        BiasTerm(
            name="neurovascular_coupling",
            interval=(0.0, 0.0),
            status="prior_specified_sensitivity",
        )


def test_r08_behaviour_head_refuses_to_omit_the_reporting_stage():
    from scwbd.observe.behavior import BehaviorObservationOperator

    with pytest.raises(RefusalR08) as exc:
        BehaviorObservationOperator(reporting=None)
    assert "report == latent state" in exc.value.message


def test_r08_fnirs_refuses_short_separation_regression_without_a_short_channel():
    from scwbd.observe.fnirs import FNIRSObservationOperator

    op = FNIRSObservationOperator(
        torch.tensor([30.0, 30.0, 35.0], dtype=torch.float64), dtype=torch.float64
    )
    od = torch.randn((3, 2, 100), dtype=torch.float64) * 1e-3
    with pytest.raises(ObservationRefusal) as exc:
        op.recover_hb(od, short_separation_regression=True)
    assert exc.value.code == "R08"


def test_bias_status_is_the_weakest_of_its_terms():
    led = UncertaintyLedger(
        variance=VarianceDecomposition(),
        bias=(
            BiasTerm("a", (0.0, 0.0), "design_estimable", estimator="replication"),
            BiasTerm(
                "b", (-1.0, 1.0), "prior_specified_sensitivity", sensitivity_grid=(-1.0, 1.0)
            ),
        ),
        provenance=Provenance(operator="t"),
        validity_domain={"x": 1},
    )
    assert led.bias_status == "prior_specified_sensitivity"
    assert led.bias_interval == (-1.0, 1.0)


def test_variance_addition_keeps_unknowns_unknown():
    a = VarianceDecomposition(measurement=1.0, numerical=0.5, units="V^2")
    b = VarianceDecomposition(measurement=2.0, numerical=0.25, units="V^2")
    c = a + b
    assert c.measurement == pytest.approx(3.0)
    assert c.numerical == pytest.approx(0.75)
    assert c.between_session == UNKNOWN
    assert c.model_class == UNKNOWN


def test_variance_rejects_negative_and_non_unknown_strings():
    with pytest.raises(ValueError):
        VarianceDecomposition(measurement=-1.0)
    with pytest.raises(ValueError):
        VarianceDecomposition(measurement="probably_small")


def test_read_without_provenance_is_refused():
    from scwbd.observe.base import ObservationRead, Support, TemporalSupport

    led = UncertaintyLedger(variance=VarianceDecomposition(), provenance=None)
    with pytest.raises(ObservationRefusal) as exc:
        ObservationRead(
            prediction=torch.zeros(3, 4),
            units="V",
            support=Support(kind="sensor", frame="f", units="V"),
            temporal=TemporalSupport(clock="c", dt=1e-3),
            ledger=led,
        )
    assert exc.value.code == "R01"


def test_psf_none_without_a_reason_is_refused():
    from scwbd.observe.base import PSF

    with pytest.raises(ObservationRefusal):
        PSF(kind="none", frame="f")
    ok = PSF(kind="none", frame="f", reason="event marker has no spatial extent")
    assert ok.kind == "none"


def test_leadfield_psf_without_a_matrix_is_refused():
    from scwbd.observe.base import PSF

    with pytest.raises(ObservationRefusal) as exc:
        PSF(kind="leadfield", frame="head")
    assert "without a lead-field matrix" in exc.value.message
