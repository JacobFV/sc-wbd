"""Shared fixtures for the observation-operator tests."""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch

from scwbd.observe.base import Prior, TemporalSupport
from scwbd.observe.leadfield import (
    SphericalHeadModel,
    TissueConductivityPriors,
)

torch.manual_seed(0)

HEAD_RADIUS = 0.09
MNE_SPHERE_SIGMAS = (0.33, 1.0, 0.042, 0.33)
MNE_SPHERE_RELATIVE_RADII = (0.90, 0.92, 0.97, 1.0)


def _delta_conductivity(sigmas) -> TissueConductivityPriors:
    names = ("brain", "csf", "skull", "scalp")
    kw = {
        n: Prior(f"sigma_{n}", "delta", (float(s),), units="S/m", source="test fixture")
        for n, s in zip(names, sigmas)
    }
    return TissueConductivityPriors(reference="test fixture", **kw)


@pytest.fixture(scope="session")
def sphere_radii() -> tuple[float, ...]:
    return tuple(HEAD_RADIUS * r for r in MNE_SPHERE_RELATIVE_RADII)


@pytest.fixture(scope="session")
def four_layer_head(sphere_radii) -> SphericalHeadModel:
    return SphericalHeadModel(
        radii=sphere_radii, conductivity=_delta_conductivity(MNE_SPHERE_SIGMAS)
    )


@pytest.fixture(scope="session")
def homogeneous_head() -> SphericalHeadModel:
    return SphericalHeadModel(
        radii=(HEAD_RADIUS,), conductivity=TissueConductivityPriors.homogeneous(0.33)
    )


@pytest.fixture(scope="session")
def sensor_positions() -> torch.Tensor:
    """20 quasi-uniform scalp electrodes (Fibonacci upper hemisphere)."""
    n = 20
    i = torch.arange(n, dtype=torch.float64) + 0.5
    phi = torch.acos(1.0 - i / n)  # upper hemisphere
    theta = math.pi * (1.0 + 5.0**0.5) * i
    p = torch.stack(
        [torch.sin(phi) * torch.cos(theta), torch.sin(phi) * torch.sin(theta), torch.cos(phi)],
        dim=-1,
    )
    return p * HEAD_RADIUS


@pytest.fixture(scope="session")
def source_positions() -> torch.Tensor:
    g = torch.Generator().manual_seed(7)
    p = torch.randn(12, 3, generator=g, dtype=torch.float64)
    p = p / p.norm(dim=-1, keepdim=True)
    depth = torch.linspace(0.030, 0.070, 12, dtype=torch.float64).unsqueeze(-1)
    return p * depth


@pytest.fixture(scope="session")
def latent_temporal() -> TemporalSupport:
    """The shared latent clock: 1 ms, i.e. finer than every observation head."""
    return TemporalSupport(clock="neural_latent", dt=1e-3, integration_window=0.0)


def mne_sample_path() -> Path | None:
    """Locate agent B's MNE sample dataset without triggering a download."""
    candidates = []
    env = os.environ.get("MNE_DATASETS_SAMPLE_PATH") or os.environ.get("MNE_DATA")
    if env:
        candidates.append(Path(env))
    candidates += [
        Path("/data/scwbd/mne"),
        Path.home() / "mne_data",
    ]
    for root in candidates:
        if not root.exists():
            continue
        for p in (root, root / "MNE-sample-data"):
            if (p / "MEG" / "sample").is_dir():
                return p
    return None


requires_mne_sample = pytest.mark.skipif(
    mne_sample_path() is None,
    reason="MNE sample dataset not present yet (agent B is downloading it)",
)
