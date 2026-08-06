"""Shared fixtures.

Tests run on CPU in float64 by default so they are deterministic and do not
compete with a benchmark run for the GPU.  Set ``SCWBD_TEST_DEVICE=cuda`` to
exercise the CUDA path.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from scwbd.infer.linear_gaussian import (
    SystemConfig,
    build_protocol,
    calibrate_observation_noise,
    calibrate_stimulus_amplitude,
    default_eta,
)

DEVICE = os.environ.get("SCWBD_TEST_DEVICE", "cpu")


@pytest.fixture(autouse=True)
def _default_dtype(request):
    """Pin the global default dtype per test, and restore it afterwards.

    Module-level ``torch.set_default_dtype`` runs at *collection* time, so the
    last module collected silently decides the dtype for every test that
    executes later.  That is global state leaking across the suite: each module
    passed in isolation while three failed in the full run.  Owning the setting
    here makes it order-independent.
    """
    prev = torch.get_default_dtype()
    want = getattr(request.module, "DEFAULT_DTYPE", torch.float64)
    torch.set_default_dtype(want)
    yield
    torch.set_default_dtype(prev)


def pytest_configure(config):
    # registered locally: pyproject.toml is owned by the architect
    config.addinivalue_line(
        "markers",
        "slow: statistical tests with enough Monte-Carlo replicates for the "
        "coverage estimate to have a meaningful error bar",
    )


@pytest.fixture(scope="session")
def tiny_cfg() -> SystemConfig:
    """A deliberately small but *structurally complete* configuration.

    It keeps the 1 ms EEG clock, the 1 s BOLD clock, a real delay line and a
    real hemodynamic cascade -- shrinking any of those would test a different
    model -- and only shortens the record.
    """
    return SystemConfig(
        device=DEVICE, dtype="float64",
        epoch_seconds=2.0, n_epochs=2,
        # n_delay_taps must satisfy tau/dt + 3*sinc_sigma = 12 + 6 = 18
        # (linear_gaussian.assert_delay_line_adequate).  The fixture previously
        # used 14, which truncated the fractional-delay kernel: the cross-
        # validation tests stayed valid because every path used the same kernel,
        # but the delay information itself was distorted.  The guard caught it
        # the moment it was adopted.
        n_delay_taps=22, hrf_stages=6, hrf_peak_stage=3, hrf_under_stage=6,
    )


@pytest.fixture(scope="session")
def tiny_setup(tiny_cfg):
    u0 = default_eta()
    proto = build_protocol(tiny_cfg, seed=7)
    amp = calibrate_stimulus_amplitude(tiny_cfg, u0, proto, evoked_ratio=1.0)
    proto = build_protocol(tiny_cfg, seed=7, amplitude=amp,
                           impulse_amplitude=8.0 * amp)
    cfg = calibrate_observation_noise(tiny_cfg, u0, proto)
    return cfg, proto, u0
