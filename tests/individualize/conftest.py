"""Fixtures for ``scwbd.individualize``.

Everything here runs on **CPU**: a 12 h training run owns the GPU, and a test
suite that competes with it is a test suite that gets killed.

Two configurations are used and the distinction matters:

``bench_cfg``
    the configuration of the *committed* identifiability benchmark
    (``reports/identifiability/manifest.json``: ``epoch_seconds=3.0``,
    ``n_epochs=30``).  It is the only configuration whose numbers may be
    compared against ``reports/identifiability/results.json``, and it costs
    ~20-60 s per design, so tests that use it are marked ``slow``.

``small_cfg``
    structurally complete -- 1 ms EEG clock, 1 s BOLD clock, a real delay line,
    a real haemodynamic cascade -- but a short record.  Shrinking any of the
    structure would test a different model.  Absolute information values differ
    from the benchmark's; **orderings and zeros do not**, and only those are
    asserted against it.
"""

from __future__ import annotations

import os

import pytest
import torch

from scwbd.individualize.profile import benchmark_config

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: reproduces the committed benchmark configuration (tens of "
        "seconds per design); the fast suite asserts orderings instead",
    )


@pytest.fixture(autouse=True)
def _float64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


@pytest.fixture(scope="session")
def bench_cfg():
    """The committed benchmark configuration, on CPU."""
    return benchmark_config()


@pytest.fixture(scope="session")
def small_cfg():
    """Structurally complete, short record.

    ``n_delay_taps`` stays at the default 26.  It is **not** a free knob: the
    windowed-sinc delay kernel needs ``tau/dt + 3*sinc_sigma`` taps, and a
    configuration below that returns inflated Fisher information without
    raising -- see ``scwbd.individualize.profile.assert_delay_line_adequate``,
    which refuses it, and ``test_delay_line_guard.py``, which watches it fire.
    """
    return benchmark_config(
        epoch_seconds=1.5,
        n_epochs=2,
        hrf_stages=6,
        hrf_peak_stage=3,
        hrf_under_stage=6,
        dt_bold=0.5,
    )
