"""Appendix C layer 4: clock identity, drift, jitter, drops, multirate events."""

from __future__ import annotations

from fractions import Fraction

import pytest
import torch

from scwbd.transforms.calibration import CalibrationRecord, ExpiryPolicy
from scwbd.transforms.clock_graph import (
    ADMISSIBLE_CLOCK_EVIDENCE,
    Boundary,
    ClockGraph,
    ClockMap,
    ClockSpec,
    Cooldown,
    DropSpec,
    MaxSilence,
    OnEvent,
    Periodic,
    ScheduleContext,
    detect_dropped_samples,
    fit_clock_map,
    hyperperiod,
    interpolation_contract,
    schedule_multirate,
)
from scwbd.transforms.errors import (
    CalibrationExpiredError,
    ClockRelationUnknownError,
    NonInvertibleTransformError,
    TransformError,
)
from scwbd.transforms.se3 import ValidityInterval

EEG = "eeg_amp"
SCANNER = "scanner_volume"
EYE = "eyetracker"


def eeg_clock(**kw) -> ClockSpec:
    base = dict(
        id=EEG,
        rate_hz=1000,
        epoch=0.0,
        trigger_path=("amplifier_din",),
        group_delay_s=0.004,  # a 100-tap FIR at 1 kHz
        jitter_sd_s=2e-5,
        interpolation_policy="linear",
    )
    base.update(kw)
    return ClockSpec(**base)


def scanner_clock(**kw) -> ClockSpec:
    base = dict(
        id=SCANNER,
        rate_hz=Fraction(1, 720) * 1000,  # TR = 0.72 s
        epoch=0.013,
        trigger_path=("scanner_ttl", "parallel_port"),
        integration_window_s=0.72,
        jitter_sd_s=1e-4,
        interpolation_policy="none",  # a volume is not resampleable in time
    )
    base.update(kw)
    return ClockSpec(**base)


# --------------------------------------------------------------------------
# specification
# --------------------------------------------------------------------------


def test_clock_rate_is_exact_rational() -> None:
    c = scanner_clock()
    assert c.period == Fraction(72, 100)
    assert c.dt == pytest.approx(0.72)


def test_event_clock_has_no_dt() -> None:
    c = ClockSpec(id="tms_pulse", rate_hz=None, domain="event", interpolation_policy="none")
    with pytest.raises(TransformError):
        c.dt
    with pytest.raises(TransformError):
        c.index_to_time(3)


def test_clock_must_be_in_seconds() -> None:
    with pytest.raises(TransformError):
        ClockSpec(id="bad", rate_hz=1000, units="ms")


def test_index_to_time_requires_an_epoch() -> None:
    c = ClockSpec(id="free_running", rate_hz=1000, epoch=None)
    with pytest.raises(TransformError) as exc:
        c.index_to_time(10)
    assert "no declared epoch" in str(exc.value)


# --------------------------------------------------------------------------
# dropped samples
# --------------------------------------------------------------------------


def test_dropped_samples_shift_every_later_timestamp() -> None:
    """``i / fs`` is wrong the moment a packet is lost, and stays wrong."""
    c = eeg_clock(dropped=(DropSpec(start_index=500, count=3, detected_by="usb_underrun"),))
    assert c.index_to_time(499) == pytest.approx(0.499)
    # stored sample 500 is acquisition tick 503
    assert c.index_to_time(500) == pytest.approx(0.503)
    assert c.index_to_time(1000) == pytest.approx(1.003)
    naive = 1000 / 1000.0
    assert abs(c.index_to_time(1000) - naive) == pytest.approx(0.003)


def test_group_delay_is_applied_only_when_asked() -> None:
    c = eeg_clock()
    assert c.index_to_time(1000) == pytest.approx(1.0)
    assert c.index_to_time(1000, compensate_group_delay=True) == pytest.approx(1.0 - 0.004)


def test_time_inside_a_gap_is_refused_not_imputed() -> None:
    c = eeg_clock(dropped=(DropSpec(500, 3),))
    assert c.time_to_index(0.499) == 499
    assert c.time_to_index(0.503) == 500
    with pytest.raises(TransformError) as exc:
        c.time_to_index(0.5005)
    assert "dropped-sample gap" in str(exc.value)
    assert "never imputed" in exc.value.remedy


def test_sample_times_skip_the_gap() -> None:
    c = eeg_clock(dropped=(DropSpec(500, 3),))
    ts = c.sample_times(0.498, 0.505)
    assert all(abs(t - 0.5) > 1e-9 and abs(t - 0.502) > 1e-9 for t in ts)
    assert any(abs(t - 0.503) < 1e-9 for t in ts)


def test_drops_are_recovered_from_timestamps() -> None:
    dt = 0.001
    ticks = [i for i in range(20) if i not in (7, 8)]
    stamps = [i * dt for i in ticks]
    drops = detect_dropped_samples(stamps, dt)
    assert len(drops) == 1
    assert drops[0].start_index == 7 and drops[0].count == 2
    c = ClockSpec(id="x", rate_hz=1000, epoch=0.0, dropped=drops)
    for stored, tick in enumerate(ticks):
        assert c.index_to_time(stored) == pytest.approx(tick * dt)


def test_a_non_integer_gap_is_refused_rather_than_rounded() -> None:
    with pytest.raises(TransformError) as exc:
        detect_dropped_samples([0.0, 0.001, 0.0025, 0.0035], 0.001)
    assert "not an integer number of ticks" in str(exc.value)


def test_overlapping_drop_records_are_refused() -> None:
    with pytest.raises(TransformError):
        ClockSpec(id="x", rate_hz=1000, epoch=0.0, dropped=(DropSpec(10, 5), DropSpec(12, 2)))


# --------------------------------------------------------------------------
# clock maps and piecewise drift
# --------------------------------------------------------------------------


def test_affine_map_evaluates_offset_and_drift() -> None:
    m = ClockMap.affine(0.01, 30e-6, t_ref=0.0)
    t_out, slope, var = m.evaluate(100.0)
    assert t_out == pytest.approx(100.0 + 0.01 + 30e-6 * 100.0)
    assert slope == pytest.approx(1 + 30e-6)
    assert var == 0.0  # no declared parameter covariance -> no invented variance
    assert m.invert(t_out) == pytest.approx(100.0, abs=1e-9)


def test_a_backwards_running_clock_map_has_no_inverse() -> None:
    m = ClockMap.affine(0.0, -2.0)  # slope = -1
    with pytest.raises(NonInvertibleTransformError) as exc:
        m.invert(5.0)
    assert "not strictly increasing" in str(exc.value)


def test_piecewise_affine_drift_is_recovered(capsys) -> None:
    """Appendix C: "piecewise drift when one affine clock map is insufficient"."""
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(17)
    breakpoint_true = 287.0
    offset_true, drift_1, drift_2 = 0.0130, 20e-6, 80e-6

    def truth(t: float) -> float:
        d = drift_1 * t
        if t > breakpoint_true:
            d += (drift_2 - drift_1) * (t - breakpoint_true)
        return t + offset_true + d

    ta = [float(i) for i in range(0, 601, 2)]
    noise = torch.randn(len(ta), dtype=torch.float64, generator=g) * 200e-6
    tb = [truth(t) + float(n) for t, n in zip(ta, noise)]

    fitted = fit_clock_map(ta, tb)
    affine_only = fit_clock_map(ta, tb, breakpoints=[])

    print(
        "\npiecewise-affine clock drift recovery:\n"
        f"  breakpoint: true {breakpoint_true:.1f} s, fitted {fitted.breakpoints}\n"
        f"  drift before: true {drift_1 * 1e6:.1f} ppm, fitted "
        f"{fitted.drift_at(100.0) * 1e6:.1f} ppm\n"
        f"  drift after : true {drift_2 * 1e6:.1f} ppm, fitted "
        f"{fitted.drift_at(500.0) * 1e6:.1f} ppm\n"
        f"  residual sd : piecewise {fitted.residual_sd:.3e} s vs "
        f"affine-only {affine_only.residual_sd:.3e} s"
    )

    assert 1 <= len(fitted.breakpoints) <= 2  # the true change, at grid resolution
    assert min(abs(b - breakpoint_true) for b in fitted.breakpoints) < 30.0
    assert fitted.drift_at(100.0) == pytest.approx(drift_1, abs=8e-6)
    assert fitted.drift_at(500.0) == pytest.approx(drift_2, abs=8e-6)
    assert fitted.offset_at(0.0) == pytest.approx(offset_true, abs=1e-3)
    # one affine map is demonstrably insufficient here
    assert affine_only.residual_sd > 4 * fitted.residual_sd
    assert fitted.residual_sd == pytest.approx(200e-6, rel=0.5)


def test_a_genuinely_affine_relation_stays_affine() -> None:
    """Breakpoint selection must not manufacture drift changes from noise."""
    g = torch.Generator().manual_seed(3)
    ta = [float(i) for i in range(0, 601, 2)]
    noise = torch.randn(len(ta), dtype=torch.float64, generator=g) * 200e-6
    tb = [t + 0.01 + 25e-6 * t + float(n) for t, n in zip(ta, noise)]
    fitted = fit_clock_map(ta, tb)
    assert fitted.breakpoints == ()
    assert fitted.drift_at(300.0) == pytest.approx(25e-6, abs=3e-6)


def test_two_sync_points_are_not_enough_for_a_ledger() -> None:
    with pytest.raises(TransformError) as exc:
        fit_clock_map([0.0, 1.0], [0.0, 1.0])
    assert "timing uncertainty would be reported as zero" in exc.value.remedy


# --------------------------------------------------------------------------
# the graph and align()
# --------------------------------------------------------------------------


def build_graph(**kw) -> ClockGraph:
    g = ClockGraph(**kw)
    g.add_clock(eeg_clock())
    g.add_clock(scanner_clock())
    g.add_clock(ClockSpec(id=EYE, rate_hz=500, epoch=0.0, jitter_sd_s=1e-3))
    return g


def relate_by_triggers(g: ClockGraph, **kw) -> None:
    """Fit the EEG<-scanner relation from recorded TTL pulses."""
    gen = torch.Generator().manual_seed(23)
    t_scan = [0.72 * i for i in range(1, 400)]
    noise = torch.randn(len(t_scan), dtype=torch.float64, generator=gen) * 5e-4
    t_eeg = [t + 0.013 + 40e-6 * t + float(n) for t, n in zip(t_scan, noise)]
    m = fit_clock_map(t_scan, t_eeg, breakpoints=[])
    g.relate(SCANNER, EEG, m, evidence="physical_trigger", **kw)


def test_align_returns_time_and_timing_uncertainty() -> None:
    g = build_graph()
    relate_by_triggers(g)
    res = g.align(SCANNER, EEG, 100.0)
    t, sd = res
    assert t == pytest.approx(100.0 + 0.013 + 40e-6 * 100.0, abs=2e-3)
    assert sd > 0.0, "an alignment without uncertainty is a claim, not a measurement"
    assert "jitter:source" in res.variance_terms
    assert res.path == (SCANNER, EEG)
    assert res.as_record()["edges"] == [f"{SCANNER}->{EEG}"]


def test_align_is_invertible_and_consistent() -> None:
    g = build_graph()
    relate_by_triggers(g)
    forward = g.align(SCANNER, EEG, 250.0)
    back = g.align(EEG, SCANNER, forward.time)
    assert back.time == pytest.approx(250.0, abs=1e-9)
    assert back.sd > 0


def test_unknown_clock_relation_is_refused_R01() -> None:
    """Refusal R01: two streams with no evidenced relation get no timeline."""
    g = build_graph()
    relate_by_triggers(g)
    with pytest.raises(ClockRelationUnknownError) as exc:
        g.align(EYE, EEG, 10.0)
    assert exc.value.code == "R01"
    assert "no evidenced relation" in str(exc.value)
    assert "cross-correlation" in exc.value.remedy


def test_undeclared_clock_is_refused() -> None:
    g = build_graph()
    with pytest.raises(ClockRelationUnknownError):
        g.align("audio", EEG, 1.0)


def test_a_clock_relation_needs_admissible_evidence() -> None:
    """"Recorded in the same session" is not a synchronization event."""
    g = build_graph()
    with pytest.raises(ClockRelationUnknownError) as exc:
        g.relate(SCANNER, EEG, ClockMap.affine(0.013), evidence="same_session")
    assert "not an admissible synchronization basis" in str(exc.value)
    assert set(ADMISSIBLE_CLOCK_EVIDENCE) == {
        "physical_trigger",
        "shared_hardware_clock",
        "cross_correlation",
        "declared_identity",
    }


def test_timing_uncertainty_grows_with_a_noisier_calibration() -> None:
    sds = []
    for noise_scale in (1e-4, 1e-3):
        g = build_graph()
        gen = torch.Generator().manual_seed(5)
        t_scan = [0.72 * i for i in range(1, 200)]
        n = torch.randn(len(t_scan), dtype=torch.float64, generator=gen) * noise_scale
        t_eeg = [t + 0.013 + float(v) for t, v in zip(t_scan, n)]
        g.relate(
            SCANNER, EEG, fit_clock_map(t_scan, t_eeg, breakpoints=[]),
            evidence="physical_trigger",
        )
        sds.append(g.align(SCANNER, EEG, 100.0).sd)
    assert sds[1] > sds[0] * 5


def test_group_delay_difference_is_reported_not_silently_applied() -> None:
    g = build_graph()
    relate_by_triggers(g)
    res = g.align(SCANNER, EEG, 100.0)
    assert res.variance_terms["group_delay_difference_s"] == pytest.approx(-0.004)
    assert any("group delays differ" in w for w in res.warnings)


def test_expired_clock_calibration_refuses_or_inflates() -> None:
    g = build_graph()
    relate_by_triggers(
        g,
        calibration=CalibrationRecord(
            method="ttl_regression",
            validity=ValidityInterval(0.0, 300.0),
            inflation_time_constant=300.0,
        ),
    )
    inside = g.align(SCANNER, EEG, 100.0)
    with pytest.raises(CalibrationExpiredError):
        g.align(SCANNER, EEG, 900.0)
    outside = g.align(SCANNER, EEG, 900.0, expiry_policy=ExpiryPolicy.INFLATE)
    assert outside.sd > inside.sd
    assert outside.warnings and any("validity interval" in w for w in outside.warnings)
    assert not outside.validity_checks[0].inside


# --------------------------------------------------------------------------
# multirate scheduling
# --------------------------------------------------------------------------


def test_hyperperiod_is_exact() -> None:
    assert hyperperiod([1000, Fraction(1000, 720)]) == pytest.approx(0.72)
    assert hyperperiod([1000, None]) is None


def test_multirate_schedule_finds_synchronization_points() -> None:
    g = build_graph()
    g.relate(
        SCANNER, EEG, ClockMap.affine(0.0), evidence="shared_hardware_clock",
    )
    sched = schedule_multirate(
        g, [EEG, SCANNER], reference=EEG, t0=0.0, t1=3.0, tolerance=1e-6
    )
    assert sched.hyperperiod_s == pytest.approx(0.72)
    assert len(sched.events[EEG]) == 3001
    # scanner volumes at 0.013 + k*0.72 land exactly on EEG sample times
    assert len(sched.sync_points) == 5
    assert sched.sync_points[0].time == pytest.approx(0.013)
    assert sched.sync_points[1].time == pytest.approx(0.733)
    assert all(set(s.clocks) == {EEG, SCANNER} for s in sched.sync_points)
    rec = sched.as_record()
    assert rec["reference"] == EEG
    assert rec["n_events"][SCANNER] == 5


def test_interpolation_contract_refuses_to_resample_an_unresampleable_clock() -> None:
    """A BOLD volume declares policy 'none': it is not a point in time."""
    g = build_graph()
    g.relate(SCANNER, EEG, ClockMap.affine(0.0), evidence="shared_hardware_clock")
    on_grid = interpolation_contract(g, SCANNER, 0.013, reference=EEG)
    assert on_grid.admissible and on_grid.gap_s == pytest.approx(0.0, abs=1e-9)
    assert on_grid.integration_window_s == pytest.approx(0.72)
    off_grid = interpolation_contract(g, SCANNER, 0.4, reference=EEG)
    assert not off_grid.admissible
    assert "would invent samples" in off_grid.reason
    with pytest.raises(TransformError) as exc:
        off_grid.require()
    assert "Unresolved" in exc.value.remedy


def test_interpolation_contract_enforces_the_maximum_gap() -> None:
    g = ClockGraph()
    g.add_clock(
        ClockSpec(
            id="wearable",
            rate_hz=4,
            epoch=0.0,
            interpolation_policy="linear",
            max_interpolation_gap_s=0.05,
        )
    )
    c = interpolation_contract(g, "wearable", 0.125, reference="wearable")
    assert not c.admissible and "beyond the declared maximum" in c.reason
    ok = interpolation_contract(g, "wearable", 0.26, reference="wearable")
    assert ok.admissible


def test_a_dropped_window_makes_the_read_inadmissible() -> None:
    g = ClockGraph()
    g.add_clock(eeg_clock(dropped=(DropSpec(500, 20),)))
    c = interpolation_contract(g, EEG, 0.510, reference=EEG)
    assert not c.admissible
    assert "dropped" in c.reason and "never imputed" in c.reason


def test_sync_points_carry_per_clock_contracts_and_admissibility() -> None:
    g = build_graph()
    g.relate(SCANNER, EEG, ClockMap.affine(0.0), evidence="shared_hardware_clock")
    sched = schedule_multirate(
        g, [EEG, SCANNER], reference=EEG, t0=0.0, t1=1.5, tolerance=1e-6
    )
    for s in sched.sync_points:
        assert {c.clock for c in s.contracts} == {EEG, SCANNER}
        assert s.timing_sd_s >= 0.0
    assert len(sched.admissible_sync_points()) == len(sched.sync_points)


# --------------------------------------------------------------------------
# firing rules
# --------------------------------------------------------------------------


def test_periodic_firing_rule_resolves_in_physical_time() -> None:
    g = build_graph()
    g.relate(SCANNER, EEG, ClockMap.affine(0.0), evidence="shared_hardware_clock")
    ctx = ScheduleContext(g, EEG, 0.0, 2.0)
    times = Periodic(period=0.5, clock=SCANNER, phase=0.0).fire_times(ctx)
    assert times == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])


def test_event_rule_refuses_to_synthesize_events() -> None:
    g = build_graph()
    ctx = ScheduleContext(g, EEG, 0.0, 2.0)
    with pytest.raises(TransformError) as exc:
        OnEvent("tms_pulse", clock=EEG).fire_times(ctx)
    assert "do not synthesize them" in exc.value.remedy


def test_rules_compose_with_cooldown_and_max_silence() -> None:
    g = build_graph()
    ctx = ScheduleContext(
        g, EEG, 0.0, 2.0, events={"stim": [0.10, 0.11, 0.12, 1.90]}
    )
    raw = OnEvent("stim", clock=EEG)
    assert raw.fire_times(ctx) == pytest.approx([0.10, 0.11, 0.12, 1.90])
    assert Cooldown(raw, 0.05).fire_times(ctx) == pytest.approx([0.10, 1.90])
    forced = MaxSilence(Cooldown(raw, 0.05), 0.5).fire_times(ctx)
    assert len(forced) > 2 and max(
        b - a for a, b in zip(forced, forced[1:])
    ) <= 0.5 + 1e-9
    combined = (raw | Periodic(1.0, EEG)).fire_times(ctx)
    assert combined == pytest.approx([0.0, 0.10, 0.11, 0.12, 1.0, 1.90, 2.0])


def test_rules_join_the_schedule_event_table() -> None:
    g = build_graph()
    g.relate(SCANNER, EEG, ClockMap.affine(0.0), evidence="shared_hardware_clock")
    sched = schedule_multirate(
        g,
        [EEG, SCANNER],
        reference=EEG,
        t0=0.0,
        t1=1.5,
        tolerance=1e-6,
        rules={"probe": Periodic(0.25, EEG)},
        context=ScheduleContext(g, EEG, 0.0, 1.5, 1e-6),
    )
    assert sched.events["rule:probe"] == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    assert "probe" in sched.provenance["rules"]


def test_rule_serialization_round_trips_to_a_dict() -> None:
    expr = Cooldown(Periodic(0.5, EEG) | OnEvent("stim", EEG), 0.1)
    d = expr.to_dict()
    assert d["type"] == "cooldown"
    assert d["child"]["type"] == "or"
    assert d["child"]["left"]["clock"] == EEG
    assert expr.clocks_used() == {EEG}


def test_boundary_rule_fires_on_recorded_lifecycle_events() -> None:
    g = build_graph()
    g.relate(SCANNER, EEG, ClockMap.affine(0.0), evidence="shared_hardware_clock")
    ctx = ScheduleContext(
        g, EEG, 0.0, 10.0, boundaries={"block_start": [0.5, 4.5], "run_end": [20.0]}
    )
    assert Boundary("block_start", clock=SCANNER).fire_times(ctx) == pytest.approx([0.5, 4.5])
    # outside the window, nothing fires -- the boundary is not extrapolated
    assert Boundary("run_end", clock=SCANNER).fire_times(ctx) == []
    with pytest.raises(TransformError) as exc:
        Boundary("never_declared", clock=EEG).fire_times(ctx)
    assert "not declared in the schedule context" in str(exc.value)
