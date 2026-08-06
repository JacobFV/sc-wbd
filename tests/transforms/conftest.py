"""Shared fixtures: the running TMS-targeting frame graph of thesis §2.8.

Building the graph once, here, keeps the individual test modules about the
property being tested rather than about scaffolding.
"""

from __future__ import annotations

import pytest
import torch

from scwbd.transforms.calibration import CalibrationRecord
from scwbd.transforms.frame_graph import Frame, FrameGraph
from scwbd.transforms.se3 import DTYPE, Pose, ValidityInterval, exp_se3
from scwbd.transforms.uncertainty import PoseUncertainty
from scwbd.transforms.units import Handedness

MM = "mm"

#: Session identity shared by every session-specific edge in the fixture.
EPOCH = "sub-01_ses-01"

#: The shared optical-tracker calibration: one physical measurement that enters
#: both head<-tracker and tracker<-device.  This is the object that makes T5's
#: cross terms nonzero.
TRACKER_CAL = "optical_tracker_session_cal"


def rigid(twist, parent, child, *, epoch=EPOCH, validity=None, units=MM):
    return Pose(
        exp_se3(torch.tensor(twist, dtype=DTYPE)),
        parent,
        child,
        units=units,
        handedness=Handedness.RIGHT,
        validity=validity or ValidityInterval.unbounded(),
        epoch=epoch,
    )


@pytest.fixture
def tracker_cal() -> str:
    """Name of the shared optical-tracker calibration in :func:`chain_graph`."""
    return TRACKER_CAL


@pytest.fixture
def frames() -> list[Frame]:
    """Appendix C layer 1 nodes for the device-to-atlas chain of equation (3)."""
    return [
        Frame(
            "atlas",
            object="MNI152 template",
            origin="anterior commissure",
            axes="RAS",
            units=MM,
        ),
        Frame(
            "image",
            object="participant T1w volume",
            origin="image centre",
            axes="RAS",
            units=MM,
        ),
        Frame(
            "head",
            object="participant head",
            origin="nasion",
            axes="nasion-inion-preauricular RAS",
            units=MM,
        ),
        Frame(
            "tracker",
            object="optical tracker base",
            origin="camera optical centre",
            axes="camera RAS",
            units=MM,
        ),
        Frame(
            "device",
            object="TMS coil",
            origin="coil centre",
            axes="coil RAS (+z along the normal)",
            units=MM,
        ),
    ]


@pytest.fixture
def chain_graph(frames) -> FrameGraph:
    """The equation-(3) chain with per-edge ledgers and a shared calibration.

    ``T_atlas<-device = T_atlas<-image . T_image<-head . T_head<-tracker
    . T_tracker<-device``
    """
    g = FrameGraph()
    for f in frames:
        g.add_frame(f)

    # the shared optical-tracker calibration: 3 parameters (two rotations of the
    # camera mount and one range scale error), Sigma_c in (rad, rad, mm)
    g.declare_shared_calibration(
        TRACKER_CAL,
        torch.diag(torch.tensor([2.0e-3**2, 2.0e-3**2, 0.4**2], dtype=DTYPE)),
    )

    def sens(rows):
        return torch.tensor(rows, dtype=DTYPE)

    # atlas <- image : nonlinear in reality; here the affine part of the
    # normalization, fitted once per participant, epoch-invariant
    g.add_rigid(
        "atlas",
        "image",
        rigid([2.0, -1.0, 3.0, 0.01, 0.02, -0.005], "atlas", "image", epoch=None),
        uncertainty=PoseUncertainty.isotropic(1.5, 6e-3),
        calibration=CalibrationRecord(
            method="affine_normalization_lsq",
            n_observations=64,
            residual_rms=1.4,
            validity=ValidityInterval.unbounded(),
        ),
    )
    # image <- head : fiducial coregistration, session specific
    g.add_rigid(
        "image",
        "head",
        rigid([-4.0, 2.0, 1.0, -0.02, 0.01, 0.03], "image", "head"),
        uncertainty=PoseUncertainty.isotropic(1.0, 4e-3),
        calibration=CalibrationRecord(
            method="fiducial_lsq",
            n_observations=5,
            residual_rms=1.8,
            validity=ValidityInterval(0.0, 7200.0),
            inflation_time_constant=1800.0,
            recalibration_triggers=("participant_repositioned",),
        ),
    )
    # head <- tracker : depends on the optical tracker calibration
    g.add_rigid(
        "head",
        "tracker",
        rigid([120.0, -30.0, 250.0, 0.5, -0.2, 0.1], "head", "tracker"),
        uncertainty=PoseUncertainty(
            cov=torch.diag(torch.tensor([0.25, 0.25, 0.25, 1e-6, 1e-6, 1e-6], dtype=DTYPE)),
            calibration_source=TRACKER_CAL,
            sensitivity=sens(
                [
                    [0.0, 250.0, 1.0],
                    [-250.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
        ),
        calibration=CalibrationRecord(
            method="tracker_fiducial_registration",
            n_observations=4,
            residual_rms=1.1,
            validity=ValidityInterval(0.0, 3600.0),
            inflation_time_constant=900.0,
            recalibration_triggers=("tracker_moved",),
        ),
    )
    # tracker <- device : same tracker calibration again -> correlated error
    g.add_rigid(
        "tracker",
        "device",
        rigid([-40.0, 15.0, -180.0, -0.3, 0.15, 0.05], "tracker", "device"),
        uncertainty=PoseUncertainty(
            cov=torch.diag(torch.tensor([0.16, 0.16, 0.16, 4e-6, 4e-6, 4e-6], dtype=DTYPE)),
            calibration_source=TRACKER_CAL,
            sensitivity=sens(
                [
                    [0.0, 180.0, 1.0],
                    [-180.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
        ),
        calibration=CalibrationRecord(
            method="coil_tracker_geometry",
            n_observations=12,
            residual_rms=0.6,
            validity=ValidityInterval(0.0, 3600.0),
            inflation_time_constant=900.0,
        ),
    )
    return g
