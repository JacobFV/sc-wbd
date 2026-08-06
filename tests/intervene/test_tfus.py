"""tFUS: propagator validation, exposure metrics, thermal dose, candidates.

SIMULATION ONLY.  The acoustic propagator is validated against the closed-form
on-axis field of a baffled circular piston,
:math:`|p(z)| = 2\\rho c u_0 |\\sin[\\tfrac{k}{2}(\\sqrt{z^2+a^2}-z)]|`.
"""

from __future__ import annotations

import math

import pytest
import torch

from scwbd.intervene.base import PhysicalDose, TargetEngagement, ThermalHistory
from scwbd.intervene.tfus.acoustics import (
    BRAIN,
    WATER,
    AcousticMedium,
    SkullLayer,
    SplitStepPropagator,
    angular_spectrum_propagate,
    hounsfield_to_acoustic,
    kwave_available,
    kwave_status,
    on_axis_piston_pressure,
    pressure_dose,
    rayleigh_sommerfeld_pressure,
)
from scwbd.intervene.tfus.exposure import (
    PlannedFocus,
    RealizedExposure,
    accumulate_thermal_dose,
    bioheat_temperature,
    exposure_metrics,
    focal_divergence,
)
from scwbd.intervene.tfus.response import (
    AuditoryConfoundNullResponse,
    IntramembraneCavitationResponse,
    MechanosensitiveChannelResponse,
    RadiationForceResponse,
    TFUSResponseModelSet,
    ThermalResponse,
    TissueContext,
    default_tfus_candidate_set,
)
from scwbd.intervene.tfus.transducer import (
    AnnularArray,
    PlanarGridArray,
    PulseSequence,
    SingleElementBowl,
    TransducerArray,
)

_DT = torch.float64
LOSSLESS = AcousticMedium("lossless_water", 1000.0, 1500.0, 0.0)
F0 = 500e3
APERTURE = 0.015
U0 = 0.05


def _flat(n: int = 256, extent: float = 0.12) -> TransducerArray:
    return TransducerArray(
        device_id="flat_piston",
        frame="transducer",
        frequency_hz=F0,
        aperture_radius_m=APERTURE,
        surface_velocity_m_per_s=U0,
        grid_n=n,
        grid_extent_m=extent,
    )


def _pulse() -> PulseSequence:
    return PulseSequence(
        frequency_hz=F0,
        burst_duration_s=20e-3,
        pulse_repetition_frequency_hz=10.0,
        n_bursts=30,
    )


# ---------------------------------------------------------------------------
# analytic validation
# ---------------------------------------------------------------------------


def test_angular_spectrum_matches_the_closed_form_piston_field():
    tr = _flat()
    _, _, dx = tr.source_grid()
    u = tr.source_velocity(focused=False)
    z = torch.linspace(0.02, 0.12, 25, dtype=_DT)
    got = angular_spectrum_propagate(u, dx, z, frequency_hz=F0, medium=LOSSLESS)[
        :, tr.grid_n // 2, tr.grid_n // 2
    ].abs()
    ref = on_axis_piston_pressure(
        z, radius_m=APERTURE, frequency_hz=F0, medium=LOSSLESS,
        surface_velocity_m_per_s=U0,
    )
    rel = float((got - ref).norm() / ref.norm())
    assert rel < 0.01, rel


def test_angular_spectrum_error_falls_with_grid_refinement():
    z = torch.linspace(0.02, 0.12, 25, dtype=_DT)
    ref = on_axis_piston_pressure(
        z, radius_m=APERTURE, frequency_hz=F0, medium=LOSSLESS,
        surface_velocity_m_per_s=U0,
    )
    errs = []
    for n, extent in ((96, 0.08), (256, 0.12)):
        tr = _flat(n, extent)
        _, _, dx = tr.source_grid()
        got = angular_spectrum_propagate(
            tr.source_velocity(focused=False), dx, z, frequency_hz=F0, medium=LOSSLESS
        )[:, n // 2, n // 2].abs()
        errs.append(float((got - ref).norm() / ref.norm()))
    assert errs[0] > errs[1], errs
    assert errs[-1] < 0.01


def test_rayleigh_integral_reproduces_the_closed_form_to_sub_percent():
    """The direct surface integral is the independent semi-analytic reference."""
    tr = _flat(256, 0.12)
    X, Y, dx = tr.source_grid()
    m = tr.aperture_mask()
    src = torch.stack([X[m], Y[m], torch.zeros_like(X[m])], dim=-1)
    u = tr.source_velocity(focused=False)[m]
    z = torch.linspace(0.02, 0.12, 13, dtype=_DT)
    obs = torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], dim=-1)
    got = rayleigh_sommerfeld_pressure(
        obs, src, u, dx * dx, frequency_hz=F0, medium=LOSSLESS
    ).abs()
    ref = on_axis_piston_pressure(
        z, radius_m=APERTURE, frequency_hz=F0, medium=LOSSLESS,
        surface_velocity_m_per_s=U0,
    )
    assert float((got - ref).norm() / ref.norm()) < 0.005


def test_spectral_and_spatial_kernels_agree_to_a_few_percent():
    """Two analytically equivalent forms; the discrepancy is numerical, and
    it is reported rather than hidden."""
    tr = _flat(256, 0.12)
    _, _, dx = tr.source_grid()
    u = tr.source_velocity(focused=False)
    z = torch.linspace(0.04, 0.12, 9, dtype=_DT)
    a = angular_spectrum_propagate(u, dx, z, frequency_hz=F0, medium=LOSSLESS,
                                   kernel="spatial")[:, 128, 128].abs()
    b = angular_spectrum_propagate(u, dx, z, frequency_hz=F0, medium=LOSSLESS,
                                   kernel="spectral")[:, 128, 128].abs()
    rel = float((a - b).norm() / a.norm())
    assert 0.0 < rel < 0.15, rel


def test_absorption_attenuates_monotonically():
    tr = _flat()
    _, _, dx = tr.source_grid()
    u = tr.source_velocity(focused=False)
    z = torch.tensor([0.06], dtype=_DT)
    p_lossless = angular_spectrum_propagate(u, dx, z, frequency_hz=F0, medium=LOSSLESS)
    p_brain = angular_spectrum_propagate(u, dx, z, frequency_hz=F0, medium=BRAIN)
    assert float(p_brain.abs().max()) < float(p_lossless.abs().max())


def test_focusing_phase_produces_gain_and_a_focus_between_source_and_geometric_focus():
    tr = TransducerArray(
        device_id="bowl", frame="transducer", frequency_hz=F0,
        aperture_radius_m=APERTURE, focal_length_m=0.06,
        surface_velocity_m_per_s=U0, grid_n=256, grid_extent_m=0.12,
    )
    _, _, dx = tr.source_grid()
    z = torch.linspace(0.02, 0.10, 41, dtype=_DT)
    unfocused = angular_spectrum_propagate(
        tr.source_velocity(focused=False), dx, z, frequency_hz=F0, medium=LOSSLESS
    )[:, 128, 128].abs()
    focused = angular_spectrum_propagate(
        tr.source_velocity(focused=True), dx, z, frequency_hz=F0, medium=LOSSLESS
    )[:, 128, 128].abs()
    assert float(focused.max()) > 2.0 * float(unfocused.max())
    # weakly focused source (gain k a^2 / 2F ~ 3.9): the on-axis maximum sits
    # proximal to the geometric focus. That shift is physics, not an error.
    z_peak = float(z[int(focused.argmax())])
    assert 0.03 < z_peak < 0.06


def test_steering_moves_the_focus_to_the_commanded_point():
    tr = _flat(256, 0.12)
    tr = TransducerArray(
        device_id="grid", frame="transducer", frequency_hz=F0,
        aperture_radius_m=APERTURE, focal_length_m=0.06,
        surface_velocity_m_per_s=U0, grid_n=256, grid_extent_m=0.12,
    )
    X, Y, dx = tr.source_grid()
    ph = tr.steering_phase([0.006, 0.0, 0.06])
    p = angular_spectrum_propagate(
        tr.source_velocity(phase=ph), dx, 0.06, frequency_hz=F0, medium=LOSSLESS
    )
    i = int(p.abs().argmax())
    assert float(X.reshape(-1)[i]) == pytest.approx(0.006, abs=1.5 * dx)
    assert float(Y.reshape(-1)[i]) == pytest.approx(0.0, abs=1.5 * dx)


def test_split_step_reduces_exactly_to_the_angular_spectrum_without_screens():
    tr = _flat(128, 0.08)
    _, _, dx = tr.source_grid()
    u = tr.source_velocity(focused=False)
    dz = 0.005
    prop = SplitStepPropagator(dx, dz, frequency_hz=F0, background=LOSSLESS)
    marched = prop.run(u, 12)[-1]
    direct = angular_spectrum_propagate(
        u, dx, 12 * dz, frequency_hz=F0, medium=LOSSLESS, kernel="spectral"
    )
    assert float((marched - direct).abs().max()) / float(direct.abs().max()) < 1e-9


# ---------------------------------------------------------------------------
# skull
# ---------------------------------------------------------------------------


def test_hounsfield_conversion_is_monotone_and_bounded():
    hu = torch.tensor([0.0, 500.0, 1000.0, 1500.0, 2000.0], dtype=_DT)
    p = hounsfield_to_acoustic(hu, frequency_hz=F0)
    assert torch.all(p["porosity"][1:] <= p["porosity"][:-1])
    assert torch.all(p["density_kg_per_m3"][1:] >= p["density_kg_per_m3"][:-1])
    assert torch.all(p["sound_speed_m_per_s"][1:] >= p["sound_speed_m_per_s"][:-1])
    # absorption falls as porosity falls
    assert float(p["alpha_np_per_m"][0]) > float(p["alpha_np_per_m"][-1])
    assert float(p["density_kg_per_m3"][0]) == pytest.approx(1000.0)
    assert float(p["sound_speed_m_per_s"][-1]) == pytest.approx(3100.0)


def test_a_skull_screen_attenuates_and_aberrates_the_focus():
    tr = _flat(128, 0.08)
    _, _, dx = tr.source_grid()
    u = tr.source_velocity(focused=False)
    n = tr.grid_n
    hu = torch.full((n, n), 1400.0, dtype=_DT)
    props = hounsfield_to_acoustic(hu, frequency_hz=F0)
    thick = torch.full((n, n), 0.006, dtype=_DT)
    # a transversely varying skull: a wedge, so it aberrates as well as absorbs
    ramp = torch.linspace(0.8, 1.2, n, dtype=_DT)[:, None].expand(n, n)
    skull = SkullLayer(
        thickness_m=thick * ramp,
        sound_speed_m_per_s=props["sound_speed_m_per_s"],
        density_kg_per_m3=props["density_kg_per_m3"],
        alpha_np_per_m=props["alpha_np_per_m"],
    )
    prop = SplitStepPropagator(dx, 0.005, frequency_hz=F0, background=BRAIN)
    free = prop.run(u, 12)[-1]
    through = prop.run(u, 12, skull=skull, skull_step=2)[-1]
    assert float(through.abs().max()) < float(free.abs().max())
    # phase aberration is present, not just amplitude loss
    assert float(skull.phase_aberration(F0, BRAIN).std()) > 0.1


def test_transmission_coefficient_is_between_zero_and_one():
    n = 8
    props = hounsfield_to_acoustic(torch.full((n, n), 1400.0, dtype=_DT))
    sk = SkullLayer(
        thickness_m=torch.full((n, n), 0.006, dtype=_DT),
        sound_speed_m_per_s=props["sound_speed_m_per_s"],
        density_kg_per_m3=props["density_kg_per_m3"],
        alpha_np_per_m=props["alpha_np_per_m"],
    )
    t = sk.transmission_coefficient(BRAIN)
    assert float(t.min()) > 0.0 and float(t.max()) < 1.0


# ---------------------------------------------------------------------------
# transducer constraints
# ---------------------------------------------------------------------------


def test_a_single_element_bowl_refuses_electronic_steering():
    with pytest.raises(ValueError, match="cannot be electronically steered"):
        SingleElementBowl().steering_phase([0.005, 0.0, 0.06])


def test_an_annular_array_refuses_lateral_steering():
    with pytest.raises(ValueError, match="only along the acoustic axis"):
        AnnularArray().steering_phase([0.005, 0.0, 0.06])
    AnnularArray().steering_phase([0.0, 0.0, 0.05])  # axial is fine


def test_a_grid_array_refuses_steering_past_the_grating_lobe_limit():
    arr = PlanarGridArray(n_elements_side=8, grid_n=64, grid_extent_m=0.08)
    lim = arr.grating_lobe_free_angle_deg()
    assert 0.0 <= lim <= 90.0
    big = 0.06 * math.tan(math.radians(lim + 20.0))
    with pytest.raises(ValueError, match="grating-lobe"):
        arr.steering_phase([big, 0.0, 0.06])


def test_duty_cycle_above_one_is_refused():
    with pytest.raises(ValueError, match="duty cycle"):
        PulseSequence(
            frequency_hz=F0,
            burst_duration_s=0.2,
            pulse_repetition_frequency_hz=10.0,
            n_bursts=5,
        )


# ---------------------------------------------------------------------------
# exposure
# ---------------------------------------------------------------------------


def test_exposure_metrics_follow_their_definitions():
    p = torch.tensor([1.0e5, 4.0e5, 2.0e5], dtype=_DT)  # Pa
    pulse = _pulse()
    m = exposure_metrics(p, medium=BRAIN, pulse=pulse, voxel_volume_m3=1e-9)
    assert m.mechanical_index == pytest.approx(0.4 / math.sqrt(0.5), rel=1e-9)
    assert m.isppa_w_per_cm2 == pytest.approx(
        (4.0e5) ** 2 / (2 * BRAIN.impedance) / 1e4, rel=1e-9
    )
    assert m.ispta_mw_per_cm2 == pytest.approx(
        m.isppa_w_per_cm2 * pulse.duty_cycle * 1e3, rel=1e-9
    )
    assert m.duty_cycle == pytest.approx(0.2)


def test_derating_is_explicit_and_changes_the_quantity():
    p = torch.tensor([4.0e5], dtype=_DT)
    full = exposure_metrics(p, pulse=_pulse())
    der = exposure_metrics(p, pulse=_pulse(), derating=0.7)
    assert der.mechanical_index < full.mechanical_index
    assert der.ledger.validity_domain["derating"] == 0.7


def test_exposure_metrics_map_onto_declared_safety_axes():
    m = exposure_metrics(torch.tensor([3.0e5], dtype=_DT), pulse=_pulse())
    axes = m.as_safety_axes()
    assert set(axes) == {
        "tfus.mechanical_index",
        "tfus.isppa_w_per_cm2",
        "tfus.ispta_mw_per_cm2",
        "tfus.duty_cycle",
    }


def test_planned_focus_cannot_masquerade_as_realized_exposure():
    planned = PlannedFocus(
        target_head_frame_m=torch.tensor([0.0, 0.02, 0.05], dtype=_DT),
        frame="head",
        steering_target_transducer_frame_m=torch.tensor([0.0, 0.0, 0.06], dtype=_DT),
        sound_speed_assumed_m_per_s=1500.0,
        pulse=_pulse(),
    )
    with pytest.raises(TypeError, match="not a realized exposure"):
        planned.as_realized()


def test_planned_and_realized_focus_diverge_and_the_sources_stay_separate():
    target = torch.tensor([0.0, 0.02, 0.05], dtype=_DT)
    planned = PlannedFocus(
        target_head_frame_m=target,
        frame="head",
        steering_target_transducer_frame_m=torch.tensor([0.0, 0.0, 0.06], dtype=_DT),
        sound_speed_assumed_m_per_s=1500.0,
        pulse=_pulse(),
    )
    grid = target + torch.tensor(
        [[0.0, 0.0, 0.0], [0.003, 0.0, -0.004], [0.0, 0.006, 0.0]], dtype=_DT
    )
    realized = RealizedExposure.from_field(
        grid,
        torch.tensor([2.0e5, 3.5e5, 1.0e5], dtype=_DT),
        medium=BRAIN,
        pulse=_pulse(),
        solver="split_step",
    )
    div = focal_divergence(
        planned,
        realized,
        planned_peak_pa=4.0e5,
        per_source_mm={
            "tracking": 0.8, "registration": 1.2, "steering": 0.5,
            "skull_model": 2.0, "coupling": 0.4,
        },
        displacement_sd_mm=1.1,
    )
    assert div.displacement_mm == pytest.approx(5.0, rel=1e-6)
    assert div.pressure_ratio == pytest.approx(0.875)
    assert len(div.per_source_mm) == 5  # never summed into one number
    assert div.ledger.bias_status == "externally_bounded"
    assert "distinct random variables" in div.ledger.validity_domain["note"]
    assert "5.00 mm" in div.summary()


def test_realized_exposure_becomes_a_physical_dose_not_an_effect():
    r = RealizedExposure.from_field(
        torch.zeros(3, 3, dtype=_DT),
        torch.tensor([1.0e5, 2.0e5, 3.0e5], dtype=_DT),
        medium=BRAIN, pulse=_pulse(), solver="split_step",
    )
    d = r.as_dose()
    assert isinstance(d, PhysicalDose)
    assert d.units == "Pa"
    assert "SIMULATION ONLY" in d.notice


# ---------------------------------------------------------------------------
# thermal
# ---------------------------------------------------------------------------


def test_bioheat_rises_then_saturates_at_the_perfusion_limit():
    t, T = bioheat_temperature(
        3.0e4, medium=BRAIN, frequency_hz=F0, duration_s=400.0, dt_s=0.05,
        duty_cycle=0.2,
    )
    assert float(T[-1]) > float(T[0])
    assert torch.all(T[1:] >= T[:-1])  # monotone rise under constant drive
    a = BRAIN.alpha_np_per_m(F0)
    steady = 2 * a * 3.0e4 * 0.2 * BRAIN.perfusion_time_constant_s / (
        BRAIN.density_kg_per_m3 * BRAIN.heat_capacity_j_per_kg_k
    )
    assert float(T[-1] - 37.0) == pytest.approx(steady, rel=0.05)


def test_thermal_dose_accumulates_over_a_temperature_trace():
    t, T = bioheat_temperature(
        3.0e4, medium=BRAIN, frequency_hz=F0, duration_s=120.0, dt_s=0.5,
        duty_cycle=0.5,
    )
    h = accumulate_thermal_dose(ThermalHistory(), t, T)
    assert h.cem43_s > 0.0
    assert h.elapsed_s == pytest.approx(120.0, rel=1e-6)
    assert h.peak_temp_c >= 37.0
    # a second burst adds to the first; it never resets
    h2 = accumulate_thermal_dose(h, t, T)
    assert h2.cem43_s > h.cem43_s
    assert h2.elapsed_s == pytest.approx(240.0, rel=1e-6)


# ---------------------------------------------------------------------------
# candidate tissue operators
# ---------------------------------------------------------------------------


def _dose() -> PhysicalDose:
    return pressure_dose(
        torch.linspace(5e4, 6e5, 64, dtype=_DT),
        support="focal_line",
        solver="split_step",
    )


def test_candidate_set_is_plural_and_includes_an_auditory_confound_null():
    cands = default_tfus_candidate_set()
    assert len(cands) >= 4
    assert any(isinstance(c, AuditoryConfoundNullResponse) for c in cands)
    assert all(c.mechanistic_status != "mechanistic" for c in cands)


def test_a_candidate_set_without_the_null_is_refused():
    with pytest.raises(ValueError, match="null/auditory-confound"):
        TFUSResponseModelSet(
            [IntramembraneCavitationResponse(), MechanosensitiveChannelResponse()]
        )


def test_a_single_candidate_is_refused():
    with pytest.raises(ValueError, match="unresolved"):
        TFUSResponseModelSet([AuditoryConfoundNullResponse()])


def test_candidates_produce_target_engagement_with_named_models():
    ctx = TissueContext(medium=BRAIN, frequency_hz=F0, duty_cycle=0.2)
    for op in default_tfus_candidate_set():
        te = op.engage(_dose(), ctx, target="sim_thalamic_tile")
        assert isinstance(te, TargetEngagement)
        assert te.response_model == op.name
        assert te.ledger.validity_domain["mechanism_resolved"] is False


def test_the_auditory_null_does_not_follow_the_acoustic_focus():
    ctx = TissueContext(duty_cycle=0.3, audible=True)
    v = AuditoryConfoundNullResponse().engage(_dose(), ctx).value
    assert float(v.std()) == 0.0  # spatially flat
    silent = TissueContext(duty_cycle=0.3, audible=False)
    assert float(AuditoryConfoundNullResponse().engage(_dose(), silent).value.abs().max()) == 0.0


def test_candidates_are_distinguishable_and_model_comparison_recovers_truth():
    ctx = TissueContext(medium=BRAIN, frequency_hz=F0, duty_cycle=0.2,
                        temperature_rise_c=0.3)
    mset = TFUSResponseModelSet()
    preds = mset.predict(_dose(), ctx)
    assert mset.disagreement(preds) > 0.1

    g = torch.Generator().manual_seed(17)
    truth = preds[1]  # mechanosensitive channel
    obs = truth + 0.02 * truth.std() * torch.randn(
        truth.numel(), generator=g, dtype=_DT
    )
    lw = mset.compare(preds, obs)
    assert mset.names[int(lw.argmax())] == "mechanosensitive_channel"
    mu = mset.to_mechanistic_uncertainty(lw)
    assert "auditory-confound null" in mu.note


def test_the_null_wins_when_the_response_does_not_follow_the_field():
    ctx = TissueContext(medium=BRAIN, frequency_hz=F0, duty_cycle=0.2)
    mset = TFUSResponseModelSet()
    preds = mset.predict(_dose(), ctx)
    g = torch.Generator().manual_seed(23)
    flat = torch.ones(preds.shape[1], dtype=_DT) + 0.01 * torch.randn(
        preds.shape[1], generator=g, dtype=_DT
    )
    lw = mset.compare(preds, flat)
    assert mset.names[int(lw.argmax())] in (
        "auditory_confound_null", "thermal"
    ), mset.names[int(lw.argmax())]


# ---------------------------------------------------------------------------
# external solver honesty
# ---------------------------------------------------------------------------


def test_kwave_status_is_honest_about_aarch64():
    s = kwave_status()
    assert s["usable"] is kwave_available()
    assert s["equivalent_to_full_wave"] is False
    assert "x86-64" in str(s["platform_note"])
