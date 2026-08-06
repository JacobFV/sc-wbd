"""Shared fixtures for the intervention tests. SIMULATION ONLY."""

from __future__ import annotations

import math

import pytest
import torch

from scwbd.intervene.tms import (
    FigureEightCoil,
    SphericalHeadModel,
    biphasic,
    coil_pose_on_sphere,
)

torch.manual_seed(0)


def pytest_configure(config):
    """Register the ``slow`` marker.

    These are the field solves and Monte Carlos that cost seconds each and
    cannot be made cheap without weakening what they check.  The whole suite
    runs in about two minutes; ``-m "not slow"`` gives CI a sub-30-second lane
    that still covers every refusal, invariant and closed-form comparison.
    """
    config.addinivalue_line(
        "markers",
        "slow: heavy field solve or Monte Carlo; deselect with -m 'not slow'",
    )


@pytest.fixture(scope="session")
def head() -> SphericalHeadModel:
    return SphericalHeadModel()


@pytest.fixture(scope="session")
def coil() -> FigureEightCoil:
    return FigureEightCoil()


@pytest.fixture(scope="session")
def pulse():
    return biphasic()


@pytest.fixture(scope="session")
def pose(head):
    """A left-DLPFC-*like* simulated scalp contact. Not a target for a person."""
    return coil_pose_on_sphere(
        head,
        [-0.55, 0.68, 0.48],
        standoff_m=0.004,
        handle_azimuth_rad=math.radians(45.0),
        target_label="simulated left-dorsolateral scalp contact",
    )


@pytest.fixture(scope="session")
def cortex(head):
    return head.cortical_shell(2562)
