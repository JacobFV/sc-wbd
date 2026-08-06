"""The haemodynamic kernel and its convolution, against scipy.

``h(s; rho)`` is checked against ``scipy.stats.gamma.pdf`` and the T3 integral
against ``scipy.signal.convolve``, so that neither the kernel shape nor the
discretisation is self-certified.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from scwbd.observe.base import TemporalSupport
from scwbd.observe.bold import BOLDObservationOperator, CanonicalHRF, HRFParameters

scipy_signal = pytest.importorskip("scipy.signal")
scipy_stats = pytest.importorskip("scipy.stats")

torch.set_default_dtype(torch.float64)


def test_canonical_kernel_matches_scipy_gamma_difference():
    p = HRFParameters()
    hrf = CanonicalHRF(p, normalise="none")
    dt = 0.01
    h = hrf.kernel(dt).numpy()
    s = dt * np.arange(h.size)
    ref = scipy_stats.gamma.pdf(s, p.peak_shape, scale=p.peak_scale) - (
        p.undershoot_ratio
        * scipy_stats.gamma.pdf(s, p.undershoot_shape, scale=p.undershoot_scale)
    )
    np.testing.assert_allclose(h, ref, rtol=1e-10, atol=1e-14)


def test_canonical_kernel_shape_is_physiological():
    hrf = CanonicalHRF(HRFParameters(), normalise="unit_peak")
    dt = 0.01
    h = hrf.kernel(dt)
    s = dt * torch.arange(h.numel(), dtype=torch.float64)
    peak_t = float(s[int(h.argmax())])
    assert 4.0 < peak_t < 7.0, f"HRF peaks at {peak_t:.2f}s, not 5-6s"
    trough_t = float(s[int(h.argmin())])
    assert 10.0 < trough_t < 22.0, f"undershoot at {trough_t:.2f}s"
    assert float(h.min()) < 0.0, "canonical HRF must have a post-stimulus undershoot"


def test_unit_area_normalisation_makes_the_integral_one():
    hrf = CanonicalHRF(HRFParameters(), normalise="unit_area")
    for dt in (0.001, 0.01, 0.1):
        h = hrf.kernel(dt)
        assert float((h * dt).sum()) == pytest.approx(1.0, abs=1e-9)


def test_t3_integral_matches_scipy_convolution(latent_temporal):
    """``int_0^{Th} h(s) x(t-s) ds`` evaluated at ``n dt_B``."""
    n_lat = 30000
    g = torch.Generator().manual_seed(4)
    x = torch.randn((3, n_lat), generator=g, dtype=torch.float64)
    hrf = CanonicalHRF(HRFParameters())
    op = BOLDObservationOperator(
        n_elements=3, tr=1.0, hrf=hrf, thermal_noise_sd=0.0, dtype=torch.float64
    )
    y, t_n = op.convolve_native(x, latent_temporal)

    dt = latent_temporal.dt
    h = hrf.kernel(dt).numpy()
    for i in range(3):
        full = scipy_signal.convolve(x[i].numpy(), h, mode="full")[:n_lat] * dt
        ref = full[:: int(round(1.0 / dt))][: y.shape[1]]
        np.testing.assert_allclose(y[i].numpy(), ref, rtol=1e-9, atol=1e-12)

    assert np.allclose(t_n.numpy(), np.arange(y.shape[1]) * 1.0)


def test_convolution_is_causal():
    """``h(s)`` has support on ``s >= 0`` only; the future must not leak back."""
    n = 20000
    x = torch.zeros((1, n), dtype=torch.float64)
    x[0, 10000] = 1.0
    lat = TemporalSupport(clock="neural_latent", dt=1e-3)
    op = BOLDObservationOperator(
        n_elements=1, tr=1.0, hrf=CanonicalHRF(), thermal_noise_sd=0.0, dtype=torch.float64
    )
    y, t_n = op.convolve_native(x, lat)
    before = y[0, t_n < 10.0]
    after = y[0, t_n >= 10.0]
    # FFT evaluation leaves rounding dust rather than exact zeros; require it to
    # be 12 orders of magnitude below the response instead of pretending it is 0.
    assert float(before.abs().max()) < 1e-12 * float(after.abs().max()), (
        "the response starts before the impulse"
    )


def test_impulse_response_of_the_operator_is_the_kernel_itself():
    n = 40000
    x = torch.zeros((1, n), dtype=torch.float64)
    x[0, 0] = 1.0 / 1e-3  # unit-area impulse on the latent grid
    lat = TemporalSupport(clock="neural_latent", dt=1e-3)
    hrf = CanonicalHRF(HRFParameters())
    op = BOLDObservationOperator(
        n_elements=1, tr=0.1, hrf=hrf, thermal_noise_sd=0.0, dtype=torch.float64
    )
    y, t_n = op.convolve_native(x, lat)
    h = hrf.kernel(1e-3)
    ref = h[:: 100][: y.shape[1]]
    np.testing.assert_allclose(y[0, : ref.numel()].numpy(), ref.numpy(), rtol=1e-9, atol=1e-12)


def test_rho_is_a_prior_and_its_spread_reaches_the_ledger(latent_temporal):
    priors = HRFParameters.priors()
    assert set(priors) >= {"peak_shape", "undershoot_ratio"}
    for p in priors.values():
        assert p.sd > 0, "an HRF parameter was declared as a constant"

    g = torch.Generator().manual_seed(6)
    x = torch.randn((2, 20000), generator=g, dtype=torch.float64)
    op = BOLDObservationOperator(
        n_elements=2, tr=1.0, hrf=CanonicalHRF(), thermal_noise_sd=0.0, dtype=torch.float64
    )
    r = op.observe(x, latent_temporal, seed=1, include_noise=False, n_rho_draws=8)
    v = r.ledger.variance.parameter_posterior
    assert v != "unknown" and float(v) > 0.0, (
        "sweeping the HRF prior produced no parameter-posterior variance"
    )


def test_balloon_and_canonical_routes_cannot_be_silently_mixed():
    from scwbd.observe.base import ObservationRefusal
    from scwbd.observe.bold import BalloonWindkesselReadout

    with pytest.raises(ObservationRefusal) as exc:
        BOLDObservationOperator(
            n_elements=2, hrf=CanonicalHRF(), balloon=BalloonWindkesselReadout()
        )
    assert exc.value.code == "R01"


def test_balloon_windkessel_readout_reproduces_the_signal_equation():
    from scwbd.observe.bold import BalloonWindkesselParameters, BalloonWindkesselReadout

    p = BalloonWindkesselParameters()
    v = torch.tensor([[1.05, 1.10]], dtype=torch.float64)
    q = torch.tensor([[0.95, 0.90]], dtype=torch.float64)

    class S:
        pass

    s = S()
    s.v, s.q = v, q
    got = BalloonWindkesselReadout(p).signal(s)
    ref = p.V0 * (p.k1 * (1 - q) + p.k2 * (1 - q / v) + p.k3 * (1 - v))
    assert torch.allclose(got, ref, rtol=0, atol=1e-15)
    # rest state must give exactly zero signal change
    s.v = torch.ones_like(v)
    s.q = torch.ones_like(q)
    assert float(BalloonWindkesselReadout(p).signal(s).abs().max()) < 1e-18
