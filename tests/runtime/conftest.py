"""Shared fixtures for the runtime suite.

Everything here is an analytic phantom.  No fixture in this directory is
subject data, and none of them authorises anything.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from scwbd.runtime import (
    DeclaredEdge,
    FrameChain,
    HeadModel,
    PoseRequest,
    ServedModel,
    TargetingService,
    spherical_phantom,
)
from scwbd.runtime.serving import _default_warmup_pose
from scwbd.transforms.se3 import Pose, exp_se3
from scwbd.transforms.uncertainty import PoseUncertainty

_DT = torch.float64


@pytest.fixture(scope="session")
def head() -> HeadModel:
    return spherical_phantom()


@pytest.fixture(scope="session")
def as_if_trained():
    """Factory marking a service as backed by a trained artifact.

    A fixture rather than an importable helper: ``from conftest import ...``
    resolves to whichever ``conftest`` is first on ``sys.path``, which is a
    different file when the suite is run across several test directories at
    once. That failure only appears in the wide run, which is exactly when
    nobody is looking at collection errors.

    ``_decide`` defers whenever ``weights_status != "trained"``: a benefit
    computed from prior-specified surrogates is a statement about the
    surrogates, not about a brain. That branch is correct and is fired by
    ``test_prediction_path.py``, but it would otherwise make ``Recommend``
    unreachable in every test -- and an unreachable ``Recommend`` cannot show
    that the *other* refusals discriminate.

    So tests that are about the uncertainty arithmetic say so explicitly here,
    rather than the production rule being weakened to keep them green.
    """
    from dataclasses import replace as _replace

    def _mark(service: TargetingService) -> TargetingService:
        service.provenance = _replace(
            service.provenance, weights_status="trained"
        )
        return service

    return _mark


@pytest.fixture(scope="session")
def service() -> TargetingService:
    return TargetingService()


@pytest.fixture(scope="session")
def served(tmp_path_factory: pytest.TempPathFactory) -> ServedModel:
    """Loaded with a checkpoint root that is guaranteed to be empty."""
    empty = tmp_path_factory.mktemp("no_checkpoints")
    return ServedModel.load(device="cpu", checkpoint_root=empty)


@pytest.fixture()
def nominal_pose(head: HeadModel) -> PoseRequest:
    """A well-specified pose over the declared target region."""
    return replace(
        _default_warmup_pose(head),
        label="nominal",
        uncertainty=PoseUncertainty.isotropic(0.0015, 0.015),
    )


def offset_pose(base: PoseRequest, head: HeadModel, twist, label: str) -> PoseRequest:
    """``base`` displaced by a right twist, still fully specified."""
    xi = torch.tensor(twist, dtype=_DT)
    matrix = base.pose.matrix @ exp_se3(xi)
    pose = Pose(
        matrix,
        head.frame,
        "coil",
        provenance={"method": "preregistered_offset"},
    )
    return replace(base, pose=pose, label=label)


@pytest.fixture(scope="session")
def consumer_chain() -> FrameChain:
    """One declared, measured edge from a foreign head frame into ours."""
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=_DT
    )
    edge = DeclaredEdge(
        pose=Pose.from_Rt(
            rotation,
            [0.001, 0.002, 0.003],
            "phantom_head_RAS",
            "consumer_head_ALS",
        ),
        provenance="measured",
        method="landmark_axes_fit_with_residual",
        uncertainty=PoseUncertainty.isotropic(0.0015, 0.010),
        session_scope="between_session",
    )
    return FrameChain([edge])


@pytest.fixture(scope="session")
def head_with_chain(head: HeadModel, consumer_chain: FrameChain) -> HeadModel:
    """The same phantom, reachable from a foreign frame through one edge."""
    return spherical_phantom(frames=consumer_chain)
