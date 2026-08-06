"""Device parity: the numbers certified on CPU must be the numbers the CUDA run
produces.

Motivation (a real failure pattern in this project, twice): a guard that is
structurally incapable of firing while reporting green, because **the check ran
somewhere the work did not**.  The statistical version of that bug is a coverage
or Fisher number validated on a code path the benchmark never takes.  This
module's whole job is to remove that asymmetry.

The rest of ``tests/infer`` runs on CPU (fast, contention-free, deterministic).
The benchmark runs on CUDA.  These tests therefore run the *same* computations
on both devices in float64 and require agreement to floating-point tolerance --
and they must be skipped loudly, never silently, when no GPU is present.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from scwbd.infer.filters import kalman_filter, multiepoch_kalman_filter, simulate_lgssm
from scwbd.infer.fisher import expected_fisher, mean_jacobian
from scwbd.infer.linear_gaussian import (
    SystemConfig,
    build_protocol,
    calibrate_observation_noise,
    calibrate_stimulus_amplitude,
    default_eta,
    make_model,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="no CUDA device: device-parity cannot be checked here. The benchmark "
           "must not be run on a machine where this test was skipped.",
)


def _setup(device: str):
    cfg = SystemConfig(
        device=device, dtype="float64", epoch_seconds=2.0, n_epochs=2,
        n_delay_taps=14, hrf_stages=6, hrf_peak_stage=3, hrf_under_stage=6,
    )
    u0 = default_eta()
    proto = build_protocol(cfg, seed=7)
    amp = calibrate_stimulus_amplitude(cfg, u0, proto, evoked_ratio=1.0)
    proto = build_protocol(cfg, seed=7, amplitude=amp, impulse_amplitude=8.0 * amp)
    cfg = calibrate_observation_noise(cfg, u0, proto)
    return cfg, proto, u0


def test_noise_calibration_agrees_across_devices():
    """If sigma_E / sigma_B differ by device, every downstream number differs."""
    c_cpu, _, _ = _setup("cpu")
    c_gpu, _, _ = _setup("cuda")
    assert abs(c_cpu.sigma_eeg - c_gpu.sigma_eeg) / c_cpu.sigma_eeg < 1e-12
    assert abs(c_cpu.sigma_bold - c_gpu.sigma_bold) / c_cpu.sigma_bold < 1e-12


def test_random_draws_are_bitwise_identical_across_devices():
    """The *stochastic* input must be bitwise device-independent.

    Every stochastic entry point draws from an explicitly seeded CPU generator
    and moves the numbers to the device, so a seed means exactly one thing no
    matter where the run happens.  (The arithmetic that follows is then allowed
    to differ in the last bits; that is checked separately.)
    """
    draws = []
    for dev in ("cpu", "cuda"):
        g = torch.Generator(device="cpu")
        g.manual_seed(1234)
        draws.append(torch.randn(5, 7, generator=g, dtype=torch.float64).to(dev).cpu())
    assert torch.equal(draws[0], draws[1])


def test_simulated_data_agree_across_devices_to_roundoff():
    """Same seed, same device-independent draws, same trajectories.

    Not asserted bitwise: the propagation is a chain of ~2000 matmuls whose
    summation order differs between BLAS backends.  A float64 relative agreement
    of 1e-11 is the honest statement, and it is far tighter than anything the
    claim report depends on.
    """
    out = {}
    for dev in ("cpu", "cuda"):
        cfg, proto, u0 = _setup(dev)
        ssm = make_model(u0, cfg, proto).ssm(epoch=0)
        data, _ = simulate_lgssm(ssm, seed=1234, batch=3)
        out[dev] = {k: v.double().cpu() for k, v in data.items()}
    for k in out["cpu"]:
        scale = float(out["cpu"][k].abs().max())
        assert float((out["cpu"][k] - out["cuda"][k]).abs().max()) / scale < 1e-11, k


def test_log_likelihood_agrees_across_devices():
    ll = {}
    for dev in ("cpu", "cuda"):
        cfg, proto, u0 = _setup(dev)
        mdl = make_model(u0, cfg, proto)
        ssm = mdl.ssm(epoch=0)
        data, _ = simulate_lgssm(ssm, seed=99, batch=1)
        ll[dev] = float(kalman_filter(ssm, data).log_likelihood[0])
    assert abs(ll["cpu"] - ll["cuda"]) / abs(ll["cpu"]) < 1e-11, ll


def test_multiepoch_shared_riccati_agrees_across_devices():
    """The shared-covariance filter is the one the benchmark actually runs."""
    out = {}
    for dev in ("cpu", "cuda"):
        cfg, proto, u0 = _setup(dev)
        mdl = make_model(np.stack([u0, u0 * 1.01]), cfg, proto)
        ssm = mdl.multiepoch_ssm()
        g = torch.Generator().manual_seed(5)
        data = {
            "eeg": torch.randn(1, cfg.n_epochs, len(cfg.eeg_steps()), 4,
                               generator=g, dtype=torch.float64).to(mdl.F.device),
            "bold": torch.randn(1, cfg.n_epochs, len(cfg.bold_steps()), 3,
                                generator=g, dtype=torch.float64).to(mdl.F.device),
        }
        out[dev] = multiepoch_kalman_filter(
            ssm, data, n_epochs=cfg.n_epochs
        )["log_likelihood"].double().cpu().numpy()
    rel = np.abs(out["cpu"] - out["cuda"]) / np.abs(out["cpu"])
    assert rel.max() < 1e-10, out


def test_structured_transition_agrees_across_devices():
    from scwbd.infer.linear_gaussian import structured_left_mul

    vals = {}
    for dev in ("cpu", "cuda"):
        cfg, proto, u0 = _setup(dev)
        mdl = make_model(u0, cfg, proto)
        g = torch.Generator().manual_seed(3)
        X = torch.randn(1, cfg.n_state, 4, generator=g, dtype=torch.float64)
        Xd = X.to(mdl.F.device)
        fast = structured_left_mul(mdl.F, cfg)(Xd)
        assert float((fast - mdl.F @ Xd).abs().max()) < 1e-12, dev
        vals[dev] = fast.double().cpu()
    assert float((vals["cpu"] - vals["cuda"]).abs().max()) < 1e-12


def test_expected_fisher_agrees_across_devices():
    """The headline claim statistic itself, on both devices."""
    reps = {}
    for dev in ("cpu", "cuda"):
        cfg, proto, u0 = _setup(dev)
        reps[dev] = expected_fisher(u0, cfg, proto, design="joint_native")
    a, b = reps["cpu"], reps["cuda"]
    scale = np.abs(a.I_likelihood).max()
    assert np.abs(a.I_likelihood - b.I_likelihood).max() / scale < 1e-9
    for key in ("rank_likelihood", "theta_profile_min_eigenvalue_nonprior",
                "min_eigenvalue_nonprior", "condition_number_total"):
        x, y = a.metrics[key], b.metrics[key]
        if isinstance(x, int):
            assert x == y, key
        else:
            assert abs(x - y) / max(abs(x), 1e-30) < 1e-7, (key, x, y)


def test_analytic_jacobian_agrees_across_devices():
    js = {}
    for dev in ("cpu", "cuda"):
        cfg, proto, u0 = _setup(dev)
        js[dev] = {k: v.double().cpu()
                   for k, v in mean_jacobian(u0, cfg, proto, method="analytic").items()}
    for k in js["cpu"]:
        s = float(js["cpu"][k].abs().max())
        assert float((js["cpu"][k] - js["cuda"][k]).abs().max()) / s < 1e-11, k


def test_gpu_cap_is_allocator_enforced_not_cgroup_reported():
    """The cap must be the kind that actually fires.

    A host cgroup limit does not bound CUDA allocations on unified memory, so a
    run that only has a cgroup cap is unbounded while reporting green.  Assert
    that we set an allocator-level fraction and that it is what it claims.
    """
    from scwbd.infer.types import cap_gpu_memory, gpu_memory_report

    info = cap_gpu_memory(4.0)
    assert info["applied"] is True
    assert "set_per_process_memory_fraction" in info["mechanism"]
    assert 0 < info["fraction"] < 1
    with pytest.raises(torch.cuda.OutOfMemoryError):
        # 4 GiB cap: this allocation must be refused by the allocator itself
        torch.empty(int(6 * 1024**3 // 8), dtype=torch.float64, device="cuda")
    rep = gpu_memory_report()
    assert rep["cuda"] is True and rep["reserved_gib"] >= rep["allocated_gib"]
    cap_gpu_memory(20.0)
