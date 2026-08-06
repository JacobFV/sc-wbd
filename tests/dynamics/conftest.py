"""Shared fixtures for the dynamics tests.

Tests run on CUDA when available (that is the target hardware) and fall back to
CPU otherwise; convergence-order tests force ``float64`` because a discretisation
order cannot be measured through fp32 round-off.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def small_edges(device):
    from scwbd.dynamics import EdgeSet

    return EdgeSet.random(24, density=0.2, seed=0, device=device)


def order_estimate(errors, factors):
    """Least-squares slope of log(err) vs log(dt) — the observed convergence order."""
    import math

    xs = [math.log(f) for f in factors]
    ys = [math.log(e) for e in errors]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den
