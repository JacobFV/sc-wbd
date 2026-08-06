"""Shared fixtures for the bench tests (agent J).

These tests exist to answer one question: **can the gates fail?**  A gate that
always passes is worthless, and a gate that always fails is equally worthless,
so every gate gets a null-true world (must FAIL), an effect-present world
(must PASS), and a missing-dependency world (must COULD_NOT_RUN).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scwbd.bench.gates import Thresholds  # noqa: E402
from scwbd.bench.harness import evaluate  # noqa: E402
from scwbd.bench.synthetic import RidgeGaussian  # noqa: E402


#: thresholds used across the fixture worlds.  They are preregistered per run
#: and deliberately explicit here so no test can quietly move one.
FIXTURE_THRESHOLDS = Thresholds(
    max_coverage_error=0.15,
    max_overconfidence_increase=0.05,
    boundary_rel_tol=0.60,
    n_boot=400,
)


def g1_arms():
    """Typed fusion candidate plus its two mandatory baselines, capacity-matched.

    The single-modality arm is given ``extra_parameters`` so that it carries
    the same budget as the fusion arm: the gate refuses unmatched comparisons,
    and rightly so.
    """
    return (
        RidgeGaussian(name="typed_fusion", blocks=["eeg", "bold"]),
        {
            "naive_resampling": RidgeGaussian(name="naive_resampling",
                                              blocks=["naive_resampled"]),
            "single_modality_eeg": RidgeGaussian(name="single_modality_eeg", blocks=["eeg"],
                                                 extra_parameters=3),
        },
    )


def g5_arms():
    """Individualized candidate plus the three mandatory baselines at equal capacity."""
    return (
        RidgeGaussian(name="individualized", blocks=["x", "subject_id"]),
        {
            "population": RidgeGaussian(name="population", blocks=["x"],
                                        extra_parameters=16),
            "anatomy_only": RidgeGaussian(name="anatomy_only", blocks=["x", "anat"],
                                          extra_parameters=12),
            "session_adapted": RidgeGaussian(name="session_adapted",
                                             blocks=["x", "session_id"],
                                             extra_parameters=13),
        },
    )


def decision_problem(train, test, candidate, baselines, *, n_options: int = 3, seed: int = 0):
    """Build a G5 decision problem from the same arms the gate will run.

    Each decision is a choice among ``n_options`` held-out trials; utility is
    the realised outcome and each arm picks the trial it predicts is best.
    Regret is then measured against the oracle choice.
    """
    n = (test.n // n_options) * n_options
    utility = test.targets.ravel()[:n].reshape(-1, n_options)
    chosen: dict[str, np.ndarray] = {}
    for name, model in [("candidate", candidate), *baselines.items()]:
        r = evaluate(model, train, test, seed=seed, refuse_group_overlap=False)
        mu = r.prediction.mean.ravel()[:n].reshape(-1, n_options)
        chosen[name] = mu.argmax(axis=1)
    return {"utility": utility, "chosen": chosen}


@pytest.fixture(scope="session")
def thresholds() -> Thresholds:
    return FIXTURE_THRESHOLDS
