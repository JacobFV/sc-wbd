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
        n_delay_taps=14, hrf_stages=6, hrf_peak_stage=3, hrf_under_stage=6,
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
