"""Sec. 2.4 SDE, the impulse limit, and the dose/effect type separation.

SIMULATION ONLY.
"""

from __future__ import annotations

import math

import pytest
import torch

from scwbd.intervene.base import (
    SIMULATION_ONLY_NOTICE,
    BurstSequence,
    ClinicalUtility,
    ExposureWindow,
    InterventionRefusal,
    Ledger,
    LinearFieldIntervention,
    MechanisticUncertainty,
    NetworkEffect,
    PhysicalDose,
    TargetEngagement,
    ThermalHistory,
    monophasic_waveform,
)

_DT = torch.float64


def _drift():
    A = -torch.diag(torch.tensor([50.0, 80.0, 120.0, 200.0], dtype=_DT))
    return lambda x, t: A @ x


def _op(period: float):
    return LinearFieldIntervention(
        pattern=torch.tensor([1.0, 0.5, 0.0, 0.0], dtype=_DT),
        waveform=monophasic_waveform(period=period, amplitude=1.0),
    )


# ---------------------------------------------------------------------------
# the five distinct fields
# ---------------------------------------------------------------------------


def test_operator_keeps_geometry_waveform_thermal_coupling_uncertainty_distinct():
    op = _op(1e-4)
    d = op.describe()
    for key in (
        "geometry",
        "waveform",
        "coupling",
        "mechanism_candidates",
        "thermal_cem43_min",
    ):
        assert key in d
    # none of them is derivable from another
    assert op.geometry is not op.waveform
    assert op.coupling is not op.mechanistic_uncertainty
    assert op.thermal_history.cem43_s == 0.0
    assert SIMULATION_ONLY_NOTICE in d["notice"]


def test_burst_sequence_pulse_onsets():
    b = BurstSequence(
        pulse_period=0.02, n_pulses_per_burst=3, burst_period=0.2, n_bursts=4
    )
    onsets = b.pulse_onsets()
    assert b.total_pulses == 12
    assert onsets.numel() == 12
    assert float(onsets[1] - onsets[0]) == pytest.approx(0.02)
    assert float(onsets[3] - onsets[0]) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# the four levels that must not be equated
# ---------------------------------------------------------------------------


def test_physical_dose_refuses_to_become_a_neural_effect():
    dose = PhysicalDose(
        modality="tms",
        quantity="E_field",
        units="V/m",
        value=torch.ones(5, 3, dtype=_DT),
        support="test",
    )
    with pytest.raises(InterventionRefusal) as e:
        dose.as_neural_effect()
    assert e.value.code == "R04"


def test_clinical_utility_cannot_be_constructed():
    with pytest.raises(InterventionRefusal) as e:
        ClinicalUtility()
    assert e.value.code == "R11"


def test_the_four_levels_are_four_distinct_types():
    assert PhysicalDose is not TargetEngagement
    assert TargetEngagement is not NetworkEffect
    assert not issubclass(NetworkEffect, PhysicalDose)


# ---------------------------------------------------------------------------
# SDE integration
# ---------------------------------------------------------------------------


def test_deterministic_integration_matches_matrix_exponential_without_drive():
    A = -torch.diag(torch.tensor([50.0, 80.0, 120.0, 200.0], dtype=_DT))
    op = LinearFieldIntervention(
        pattern=torch.zeros(4, dtype=_DT), waveform=monophasic_waveform(1e-9, 0.0)
    )
    x0 = torch.tensor([1.0, -0.5, 0.25, 0.1], dtype=_DT)
    w = ExposureWindow(0.0, 0.02)
    res = op.integrate(x0, lambda x, t: A @ x, w, dt=1e-6)
    exact = torch.matrix_exp(A * w.duration) @ x0
    assert torch.allclose(res.final_state, exact, rtol=2e-3, atol=1e-9)


def test_stochastic_integration_requires_explicit_seed_and_is_reproducible():
    op = _op(1e-4)
    w = ExposureWindow(0.0, 0.01)
    Q = 0.01 * torch.eye(4, dtype=_DT)
    x0 = torch.zeros(4, dtype=_DT)
    with pytest.raises(ValueError):
        op.integrate(x0, _drift(), w, dt=1e-5, Q_sqrt=Q)
    a = op.integrate(x0, _drift(), w, dt=1e-5, Q_sqrt=Q, seed=7).final_state
    b = op.integrate(x0, _drift(), w, dt=1e-5, Q_sqrt=Q, seed=7).final_state
    c = op.integrate(x0, _drift(), w, dt=1e-5, Q_sqrt=Q, seed=8).final_state
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


# ---------------------------------------------------------------------------
# the impulse limit is a tested claim, not a flag
# ---------------------------------------------------------------------------


def test_impulse_limit_admitted_for_a_brief_pulse_and_error_is_quantified():
    op = _op(1e-4)  # 100 us pulse; slowest state timescale is 20 ms
    w = ExposureWindow(0.0, 0.05)
    rep = op.check_impulse_limit(torch.zeros(4, dtype=_DT), _drift(), w, dt=1e-5)
    assert rep.admitted
    assert rep.rel_error < 1e-2
    assert rep.exposure_duration_s / rep.system_timescale_s > 1.0  # window, not pulse
    assert "ADMITTED" in rep.summary()
    # the reported error is a real number, not a placeholder
    assert 0.0 < rep.rel_error < 1.0


def test_impulse_limit_error_shrinks_as_the_pulse_shortens():
    # dt is held FIXED so the finite-duration reference has the same accuracy
    # for every pulse: only the pulse duration varies.
    errs = []
    for period in (2e-3, 4e-4, 8e-5):
        rep = _op(period).check_impulse_limit(
            torch.zeros(4, dtype=_DT),
            _drift(),
            ExposureWindow(0.0, 0.02),
            dt=5e-6,
        )
        errs.append(rep.rel_error)
    assert errs[0] > errs[1] > errs[2]
    assert errs[0] > 1e-2 and errs[2] < 1e-3


def test_impulse_limit_refuses_an_underresolved_finite_duration_reference():
    # a step that steps straight over the pulse would manufacture agreement
    with pytest.raises(InterventionRefusal) as e:
        _op(1e-5).check_impulse_limit(
            torch.zeros(4, dtype=_DT), _drift(), ExposureWindow(0.0, 0.05), dt=1e-5
        )
    assert e.value.code == "R06"
    assert "steps per pulse" in str(e.value)


def test_impulse_limit_flag_is_refused_when_the_error_is_too_large():
    op = _op(0.04)  # 40 ms pulse against a 20 ms system timescale
    w = ExposureWindow(0.0, 0.05)
    rep = op.check_impulse_limit(torch.zeros(4, dtype=_DT), _drift(), w, dt=1e-4)
    assert not rep.admitted
    assert rep.rel_error > 1e-2
    with pytest.raises(InterventionRefusal) as e:
        op.integrate(
            torch.zeros(4, dtype=_DT), _drift(), w, dt=1e-4, impulse_limit=True
        )
    assert e.value.code == "R06"
    assert "impulse limit REFUSED" in str(e.value)


def test_impulse_limit_run_attaches_its_justifying_report():
    op = _op(1e-4)
    res = op.integrate(
        torch.zeros(4, dtype=_DT),
        _drift(),
        ExposureWindow(0.0, 0.05),
        dt=1e-5,
        impulse_limit=True,
    )
    assert res.impulse_report is not None
    assert res.impulse_report.admitted
    assert res.impulse_report.finite_duration_state.shape == (4,)


# ---------------------------------------------------------------------------
# thermal history
# ---------------------------------------------------------------------------


def test_cem43_accumulates_to_the_sapareto_dewey_definition():
    h = ThermalHistory()
    h = h.accumulate(43.0, 60 * 60.0)  # one hour at 43 C
    assert h.cem43_minutes == pytest.approx(60.0)

    h2 = ThermalHistory().accumulate(44.0, 30 * 60.0)  # 30 min at 44 C
    assert h2.cem43_minutes == pytest.approx(60.0)  # R=0.5 -> doubling per degree

    h3 = ThermalHistory().accumulate(42.0, 60 * 60.0)  # 1 h at 42 C, R=0.25
    assert h3.cem43_minutes == pytest.approx(15.0)


def test_cem43_is_monotone_and_never_reset_between_bursts():
    h = ThermalHistory()
    prev = 0.0
    for _ in range(5):
        h = h.accumulate(40.0, 10.0)
        assert h.cem43_s > prev
        prev = h.cem43_s
    assert h.elapsed_s == pytest.approx(50.0)
    assert h.peak_temp_c == pytest.approx(40.0)


def test_thermal_history_tracks_peak_over_a_field():
    h = ThermalHistory().accumulate(torch.tensor([37.0, 38.5, 41.0]), 1.0)
    assert h.peak_temp_c == pytest.approx(41.0)


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------


def test_ledger_merge_keeps_bias_and_variance_separate():
    a = Ledger(variance={"numerical": 1.0}, bias_interval=(-0.1, 0.2),
               bias_status="design_estimable")
    b = Ledger(variance={"numerical": 2.0, "pose": 3.0}, bias_interval=(0.0, 0.1),
               bias_status="prior_specified_sensitivity")
    m = a.merged(b)
    assert m.variance["numerical"] == 3.0
    assert m.variance["pose"] == 3.0
    assert m.bias_interval == (-0.1, pytest.approx(0.3))
    # the weaker bias status wins; it is never upgraded by merging
    assert m.bias_status == "prior_specified_sensitivity"


def test_mechanistic_uncertainty_reports_entropy_and_disagreement():
    mu = MechanisticUncertainty(
        candidates=("a", "b", "c"), log_weights=torch.zeros(3, dtype=_DT)
    )
    assert mu.entropy_nats() == pytest.approx(math.log(3.0))
    preds = torch.stack(
        [torch.zeros(10, dtype=_DT), torch.ones(10, dtype=_DT), 2 * torch.ones(10, dtype=_DT)]
    )
    assert mu.disagreement(preds) > 0.5
    assert not mu.resolved
