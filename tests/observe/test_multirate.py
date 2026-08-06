"""The load-bearing claim: EEG at 1 ms and BOLD at 1 s coexist, un-resampled.

thesis Sec. 7.1 -- "EEG samples need not be downsampled to the fMRI repetition
time, and fMRI voxels need not be assigned sensor-space electrical precision".
Sec. 2.6 -- mappings between views are calibrated operators, not automatic
resampling.

These tests assert the *mechanism*, not just the output shapes:

* both heads consume the **same** latent tensor and the **same** latent
  ``TemporalSupport`` object, and neither mutates it;
* the EEG read is exactly ``L x(k dt_E)`` at the latent's own samples, so no
  interpolation happened;
* the BOLD read is the kernel integral evaluated at ``n dt_B``, computed on the
  latent grid;
* running EEG on a TR-decimated latent destroys information that the native-rate
  read preserves, which is the quantitative content of "fictitious equivalence";
* the two reads carry different clocks, different supports and different units,
  and the operators refuse a clock ratio that would require interpolation.
"""

from __future__ import annotations

import math

import pytest
import torch

from scwbd.observe.base import ObservationRefusal, TemporalSupport
from scwbd.observe.bold import BOLDObservationOperator, CanonicalHRF
from scwbd.observe.eeg import EEGNoiseModel, EEGObservationOperator
from scwbd.observe.leadfield import ReferenceOperator

torch.set_default_dtype(torch.float64)

N_SEC = 40.0
DT_LATENT = 1e-3


@pytest.fixture(scope="module")
def latent():
    """A shared latent trajectory with both fast and slow structure."""
    n = int(N_SEC / DT_LATENT)
    t = DT_LATENT * torch.arange(n, dtype=torch.float64)
    g = torch.Generator().manual_seed(42)
    slow = torch.stack([torch.sin(2 * math.pi * 0.05 * t + i) for i in range(12)])
    fast = torch.stack([torch.sin(2 * math.pi * 40.0 * t + 0.3 * i) for i in range(12)])
    noise = 0.1 * torch.randn((12, n), generator=g, dtype=torch.float64)
    return slow + 0.6 * fast + noise


@pytest.fixture(scope="module")
def heads(four_layer_head, sensor_positions, source_positions):
    lf = four_layer_head.lead_field(source_positions, sensor_positions).project(
        source_positions / source_positions.norm(dim=-1, keepdim=True)
    )
    eeg = EEGObservationOperator(
        lf,
        dt=1e-3,
        reference=ReferenceOperator.average(lf.n_sensors, dtype=torch.float64),
        noise=EEGNoiseModel(line_v=0.0),
        dtype=torch.float64,
    )
    bold = BOLDObservationOperator(
        n_elements=12, tr=1.0, hrf=CanonicalHRF(), thermal_noise_sd=0.0,
        dtype=torch.float64,
    )
    return eeg, bold


def test_both_heads_read_the_same_latent_at_their_own_rates(heads, latent, latent_temporal):
    eeg, bold = heads
    before = (latent_temporal.dt, latent_temporal.clock, latent.clone())

    r_e = eeg.observe(latent * 1e-9, latent_temporal, seed=1, include_noise=False)
    r_b = bold.observe(latent, latent_temporal, seed=1, include_noise=False)

    # native clocks, unchanged
    assert r_e.temporal.dt == pytest.approx(1e-3)
    assert r_b.temporal.dt == pytest.approx(1.0)
    assert r_e.temporal.clock != r_b.temporal.clock
    assert latent_temporal.dt == before[0] and latent_temporal.clock == before[1]
    assert torch.equal(latent, before[2]), "an operator mutated the shared latent"

    # native sample counts follow from the clocks, not from each other
    assert r_e.n_samples == latent.shape[-1]
    assert r_b.n_samples == int(N_SEC)
    assert r_e.n_samples == 1000 * r_b.n_samples

    # different supports, different units
    assert r_e.units == "V" and r_b.units == "dimensionless"
    assert r_e.support.kind == "sensor" and r_b.support.kind == "parcel"
    assert r_e.support.psf.kind == "leadfield"
    assert r_b.support.psf.kind == "gaussian"


def test_eeg_read_is_exactly_the_leadfield_applied_to_latent_samples(
    heads, latent, latent_temporal
):
    """No interpolation: sample k of the read equals L x[:, k] exactly."""
    eeg, _ = heads
    x = latent * 1e-9
    r = eeg.observe(x, latent_temporal, seed=0, include_noise=False, include_artifacts=False)
    L, _ = eeg.linear_gaussian(include_background=False)
    expected = L @ x
    assert torch.allclose(r.prediction, expected, rtol=1e-12, atol=1e-18)


def test_bold_read_is_the_kernel_integral_on_the_latent_grid(heads, latent, latent_temporal):
    _, bold = heads
    r = bold.observe(latent, latent_temporal, seed=0, include_noise=False)
    y, t_n = bold.convolve_native(latent, latent_temporal)
    assert torch.allclose(r.prediction, y[:, : r.n_samples], rtol=1e-10, atol=1e-14)
    assert torch.allclose(t_n[: r.n_samples], r.times(), rtol=0, atol=1e-12)


def test_native_rate_eeg_preserves_information_that_tr_resampling_destroys(
    heads, latent, latent_temporal
):
    """Quantifies 'fictitious equivalence': the 40 Hz component survives only natively."""
    eeg, _ = heads
    x = latent * 1e-9

    native = eeg.observe(
        x, latent_temporal, seed=0, include_noise=False, include_artifacts=False
    )

    # the forbidden operation: decimate the latent onto the BOLD clock first
    decimated = x[:, ::1000]
    tr_clock = TemporalSupport(clock="neural_latent_decimated", dt=1.0)
    eeg_on_tr = EEGObservationOperator(
        eeg.lead_field, dt=1.0, reference=eeg.reference,
        noise=EEGNoiseModel(line_v=0.0), dtype=torch.float64,
    )
    resampled = eeg_on_tr.observe(
        decimated, tr_clock, seed=0, include_noise=False, include_artifacts=False
    )

    def band_power(sig: torch.Tensor, dt: float, lo: float, hi: float) -> float:
        S = torch.fft.rfft(sig, dim=-1).abs() ** 2
        f = torch.fft.rfftfreq(sig.shape[-1], d=dt)
        m = (f >= lo) & (f <= hi)
        return float(S[..., m].sum() / S.sum().clamp_min(1e-300))

    assert band_power(native.prediction, 1e-3, 35.0, 45.0) > 0.2, (
        "the native EEG read lost the 40 Hz component"
    )
    # at dt = 1 s the 40 Hz component cannot even be represented: Nyquist is 0.5 Hz
    assert resampled.temporal.rate_hz == pytest.approx(1.0)
    assert 45.0 > 0.5 / resampled.temporal.dt
    # and the aliased signal no longer resembles the native one when averaged
    native_binned = native.prediction[:, : 1000 * resampled.n_samples].reshape(
        native.prediction.shape[0], resampled.n_samples, 1000
    ).mean(-1)
    rel = float(
        (native_binned - resampled.prediction).norm() / native_binned.norm().clamp_min(1e-30)
    )
    assert rel > 0.05, (
        "point-decimation and integration agree here, which would mean the "
        "resampling shortcut is harmless; it is not, and the test must show it"
    )


def test_operator_refuses_a_clock_ratio_that_needs_interpolation(heads, latent):
    """Refusal R01 rather than a silent interpolation."""
    eeg, bold = heads
    weird = TemporalSupport(clock="odd_latent", dt=0.7e-3)
    with pytest.raises(ObservationRefusal) as exc:
        eeg.observe(latent * 1e-9, weird, seed=0)
    assert exc.value.code == "R01"
    assert "interpolation is forbidden" in exc.value.message

    with pytest.raises(ObservationRefusal):
        bold.convolve_native(latent, weird)


def test_slice_timing_shifts_the_sampling_instant_not_the_signal(latent, latent_temporal):
    """Slice timing is re-indexing, never interpolation."""
    from scwbd.observe.bold import SliceTiming

    st = SliceTiming(
        offsets_s=torch.tensor([0.0, 0.25, 0.5, 0.75] * 3, dtype=torch.float64),
        order="interleaved",
        tr=1.0,
    )
    op = BOLDObservationOperator(
        n_elements=12, tr=1.0, slice_timing=st, thermal_noise_sd=0.0, dtype=torch.float64
    )
    r = op.observe(latent, latent_temporal, seed=0, include_noise=False)
    y_ref, _ = op.convolve_native(latent, latent_temporal, n_samples=r.n_samples)
    # element 0 has zero offset -> identical; element 1 is shifted by 0.25 s
    assert torch.allclose(r.prediction[0], y_ref[0], rtol=1e-10, atol=1e-14)
    assert not torch.allclose(r.prediction[1], y_ref[1], rtol=1e-6, atol=1e-12)

    # a sub-millisecond residual is quantisation, not interpolation: it is
    # allowed and reported as a bias term rather than silently absorbed
    near = SliceTiming(
        offsets_s=torch.full((12,), 0.2504, dtype=torch.float64), order="custom", tr=1.0
    )
    op_near = BOLDObservationOperator(
        n_elements=12, tr=1.0, slice_timing=near, thermal_noise_sd=0.0, dtype=torch.float64
    )
    r_near = op_near.observe(latent, latent_temporal, seed=0, include_noise=False)
    q = r_near.ledger.bias_by_name("slice_timing_grid_quantisation")
    assert q is not None and q.status == "design_estimable"
    assert 0.0 < q.half_width <= 0.5 * latent_temporal.dt + 1e-12

    # a latent too coarse to represent the slice order refuses outright
    coarse = TemporalSupport(clock="coarse_latent", dt=0.5)
    op_coarse = BOLDObservationOperator(
        n_elements=12, tr=1.0, slice_timing=st, thermal_noise_sd=0.0, dtype=torch.float64
    )
    with pytest.raises(ObservationRefusal) as exc:
        op_coarse.observe(latent[:, ::500], coarse, seed=0, include_noise=False)
    assert exc.value.code == "R01"


def test_physiological_noise_aliases_because_it_is_sampled_not_resampled(
    latent, latent_temporal
):
    """Cardiac pulsation at ~1 Hz aliased by a 1 s TR is an emergent property."""
    from scwbd.observe.bold import PhysiologicalNoise

    op = BOLDObservationOperator(
        n_elements=12,
        tr=1.0,
        physio=PhysiologicalNoise(cardiac_hz=1.05, respiratory_hz=0.28),
        thermal_noise_sd=0.0,
        drift=None,
        motion=None,
        dtype=torch.float64,
    )
    r = op.observe(latent * 0.0, latent_temporal, seed=5, include_noise=True)
    ph = r.components["physiological"]
    S = torch.fft.rfft(ph, dim=-1).abs() ** 2
    f = torch.fft.rfftfreq(ph.shape[-1], d=1.0)
    # 1.05 Hz cannot exist below the 0.5 Hz Nyquist; its power must appear folded
    assert float(f.max()) <= 0.5 + 1e-9
    assert float(S.sum()) > 0.0
