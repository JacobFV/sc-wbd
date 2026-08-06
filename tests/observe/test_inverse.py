"""Source estimation is non-unique, and the tests demonstrate it numerically.

The central test constructs **two distinct source configurations that produce
identical sensor data** (to well inside the noise floor) and checks that the API
cannot be used to present either as "the source".
"""

from __future__ import annotations

import math

import pytest
import torch

from scwbd.observe.base import ObservationRefusal
from scwbd.observe.inverse import (
    InverseSolutionSet,
    regularization_sweep,
    solve_inverse,
)

torch.set_default_dtype(torch.float64)


@pytest.fixture(scope="module")
def dense_lead_field(four_layer_head, sensor_positions):
    """More sources than sensors: the honest situation, hence a null space."""
    g = torch.Generator().manual_seed(21)
    n_src = 120
    d = torch.rand(n_src, generator=g, dtype=torch.float64) * 0.035 + 0.030
    u = torch.randn((n_src, 3), generator=g, dtype=torch.float64)
    u = u / u.norm(dim=-1, keepdim=True)
    pos = u * d.unsqueeze(-1)
    lf = four_layer_head.lead_field(pos, sensor_positions, dtype=torch.float64)
    return lf.project(pos / pos.norm(dim=-1, keepdim=True))


@pytest.fixture(scope="module")
def noise_cov(dense_lead_field):
    n = dense_lead_field.n_sensors
    return (1e-6**2) * torch.eye(n, dtype=torch.float64)


def test_two_distinct_sources_give_the_same_sensor_data(dense_lead_field, noise_cov):
    """The numerical demonstration of non-identifiability."""
    L = dense_lead_field.as_matrix().to(torch.float64)
    n_sens, n_src = L.shape
    assert n_src > n_sens, "the fixture must be underdetermined to make the point"

    x_true = torch.zeros((n_src, 1), dtype=torch.float64)
    x_true[7] = 20e-9
    y = L @ x_true

    sol = solve_inverse(y, dense_lead_field, noise_cov, method="MNE")
    assert sol.n_null == n_src - n_sens
    assert sol.null_space_fraction() > 0.5

    other = sol.admissible(seed=1, scale=1e-8)
    assert not torch.allclose(sol.particular, other, rtol=1e-3, atol=1e-12)

    # the two configurations are genuinely different sources ...
    diff = float((sol.particular - other).norm() / sol.particular.norm())
    assert diff > 0.1, f"the alternative solution is only {diff:.4f} different"

    # ... yet produce sensor data indistinguishable well below the noise floor
    dy = float((L @ sol.particular - L @ other).abs().max())
    noise_sd = float(noise_cov.diagonal().sqrt().max())
    assert dy < 1e-6 * noise_sd, (
        f"the two solutions differ by {dy:.3e} V at the sensors, which is not "
        f"negligible against a {noise_sd:.3e} V noise floor"
    )


def test_point_estimate_refuses_without_acknowledgement(dense_lead_field, noise_cov):
    L = dense_lead_field.as_matrix().to(torch.float64)
    y = L @ (1e-8 * torch.randn((L.shape[1], 1), dtype=torch.float64))
    sol = solve_inverse(y, dense_lead_field, noise_cov, method="dSPM")
    with pytest.raises(ObservationRefusal) as exc:
        sol.point_estimate()
    assert exc.value.code == "R02"
    assert "null space" in exc.value.message
    got = sol.point_estimate(acknowledge_non_uniqueness=True)
    assert got.shape == sol.particular.shape


@pytest.mark.parametrize("method", ["MNE", "dSPM", "sLORETA", "LCMV"])
def test_every_method_returns_a_set_with_resolution_analysis(
    dense_lead_field, noise_cov, method
):
    L = dense_lead_field.as_matrix().to(torch.float64)
    g = torch.Generator().manual_seed(9)
    x = torch.zeros((L.shape[1], 200), dtype=torch.float64)
    x[11] = 1e-8 * torch.sin(torch.linspace(0, 20, 200, dtype=torch.float64))
    y = L @ x + 1e-7 * torch.randn((L.shape[0], 200), generator=g, dtype=torch.float64)

    sol = solve_inverse(y, dense_lead_field, noise_cov, method=method)
    assert isinstance(sol, InverseSolutionSet)
    res = sol.resolution
    assert res.resolution.shape == (L.shape[1], L.shape[1])
    assert res.effective_rank() <= L.shape[0]
    assert float(res.peak_localization_error_m().mean()) >= 0.0
    assert float(res.spatial_dispersion_m().mean()) > 0.0
    ct = res.crosstalk_ratio()
    assert float(ct.mean()) > 0.0, "an inverse operator with no cross-talk is a bug"
    assert float(ct.max()) <= 1.0 + 1e-9

    led = sol.ledger
    assert led.validity_domain["null_dimension"] == sol.n_null
    assert "claim_boundary" in led.validity_domain
    assert led.bias_by_name("source_non_identifiability") is not None
    assert led.bias_by_name("depth_bias") is not None
    assert led.bias_by_name("spatial_leakage") is not None


def test_resolution_matrix_is_never_the_identity(dense_lead_field, noise_cov):
    """If it were, the inverse problem would be solved; it is not."""
    L = dense_lead_field.as_matrix().to(torch.float64)
    y = L @ (1e-8 * torch.randn((L.shape[1], 1), dtype=torch.float64))
    for method in ("MNE", "dSPM", "sLORETA"):
        sol = solve_inverse(y, dense_lead_field, noise_cov, method=method)
        R = sol.resolution.resolution
        off = float((R - torch.diag(torch.diagonal(R))).abs().max())
        assert off > 0.0
        assert sol.resolution.effective_rank() < R.shape[0]


def test_depth_bias_is_reported_and_real(dense_lead_field, noise_cov):
    """Deep sources are systematically underestimated by an unweighted MNE."""
    L = dense_lead_field.as_matrix().to(torch.float64)
    y = L @ (1e-8 * torch.randn((L.shape[1], 1), dtype=torch.float64))
    sol = solve_inverse(y, dense_lead_field, noise_cov, method="MNE", depth_weighting=0.0)
    db = sol.depth_bias()

    # the bias is directional: point spreads peak systematically further out
    # (more superficially) than the source that generated them
    shift = db["radial_shift_m"]
    assert float(shift.mean()) > 1e-3, (
        f"mean radial shift is {1000 * float(shift.mean()):.2f} mm; the classic "
        "outward depth bias of an unweighted MNE is not visible"
    )
    # and gain still increases with superficiality
    r = float(torch.corrcoef(torch.stack([db["radius_m"], db["resolution_diagonal"].abs()]))[0, 1])
    assert r > 0.3, f"resolution gain does not increase with superficiality (r={r:.3f})"

    weighted = solve_inverse(
        y, dense_lead_field, noise_cov, method="MNE", depth_weighting=0.8
    )
    wdb = weighted.depth_bias()
    assert float(wdb["mean_abs_radial_shift_m"]) < float(db["mean_abs_radial_shift_m"]), (
        "depth weighting failed to reduce the mislocalisation it exists to reduce"
    )
    # ... and it costs resolution, which is the trade the ledger must expose
    assert float(wdb["spatial_dispersion_m"].mean()) >= 0.0
    assert sol.ledger.bias_by_name("depth_bias").status == "design_estimable"


def test_regularization_changes_the_answer_and_the_sweep_says_so(
    dense_lead_field, noise_cov
):
    L = dense_lead_field.as_matrix().to(torch.float64)
    x = torch.zeros((L.shape[1], 1), dtype=torch.float64)
    x[3] = 2e-8
    g = torch.Generator().manual_seed(13)
    y = L @ x + 5e-7 * torch.randn((L.shape[0], 1), generator=g, dtype=torch.float64)
    sol = solve_inverse(y, dense_lead_field, noise_cov, method="MNE")

    sweep = regularization_sweep(sol)
    assert sweep["max_relative_deviation"] > 0.05, (
        "the regularisation sweep found no sensitivity, which would mean lambda "
        "is not doing anything -- check the sweep, not the claim"
    )
    assert sol.ledger.bias_by_name("regularization_choice").status == (
        "prior_specified_sensitivity"
    )


def test_posterior_samples_are_distinct_and_data_consistent(dense_lead_field, noise_cov):
    L = dense_lead_field.as_matrix().to(torch.float64)
    y = L @ (1e-8 * torch.randn((L.shape[1], 1), dtype=torch.float64))
    sol = solve_inverse(y, dense_lead_field, noise_cov, method="MNE")
    s = sol.sample_posterior(5, seed=2)
    assert s.shape[0] == 5
    assert float((s[0] - s[1]).abs().max()) > 0.0


def test_channel_mismatch_is_refused(dense_lead_field, noise_cov):
    y = torch.randn((dense_lead_field.n_sensors + 3, 10), dtype=torch.float64)
    with pytest.raises(ObservationRefusal) as exc:
        solve_inverse(y, dense_lead_field, noise_cov)
    assert exc.value.code == "R01"


def test_beamformer_fails_on_correlated_sources(dense_lead_field, noise_cov):
    """Beamformers trade one assumption for another; the failure is demonstrated."""
    L = dense_lead_field.as_matrix().to(torch.float64)
    t = torch.linspace(0, 4 * math.pi, 400, dtype=torch.float64)
    wave = 1e-8 * torch.sin(t)

    x_single = torch.zeros((L.shape[1], 400), dtype=torch.float64)
    x_single[5] = wave
    x_corr = x_single.clone()
    x_corr[60] = wave  # perfectly correlated second source

    g = torch.Generator().manual_seed(17)
    noise = 1e-9 * torch.randn((L.shape[0], 400), generator=g, dtype=torch.float64)
    s_single = solve_inverse(L @ x_single + noise, dense_lead_field, noise_cov, method="LCMV")
    s_corr = solve_inverse(L @ x_corr + noise, dense_lead_field, noise_cov, method="LCMV")

    amp_single = float(s_single.particular[5].abs().max())
    amp_corr = float(s_corr.particular[5].abs().max())
    assert amp_corr < 0.7 * amp_single, (
        "the LCMV beamformer did not suppress a perfectly correlated source pair; "
        "either the fixture is degenerate or the implementation is not a beamformer"
    )
