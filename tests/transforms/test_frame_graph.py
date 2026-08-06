"""The frame graph: equation (3), stored edges, path agreement, expiry, warps."""

from __future__ import annotations

import pytest
import torch

from scwbd.transforms.calibration import CalibrationRecord, ExpiryPolicy
from scwbd.transforms.errors import (
    CalibrationExpiredError,
    HandednessError,
    NoPathError,
    NonInvertibleTransformError,
    TransformError,
    UnitMismatchError,
    UnknownFrameError,
)
from scwbd.transforms.frame_graph import (
    DeformableTransform,
    Frame,
    FrameGraph,
    TransformEdge,
    device_to_atlas_chain,
)
from scwbd.transforms.se3 import DTYPE, Pose, ValidityInterval, exp_se3
from scwbd.transforms.uncertainty import PoseUncertainty
from scwbd.transforms.units import Handedness


def t(x) -> torch.Tensor:
    return torch.as_tensor(x, dtype=DTYPE)


# --------------------------------------------------------------------------
# equation (3)
# --------------------------------------------------------------------------


def test_equation_3_chain(chain_graph) -> None:
    """``T_atlas<-device = T_atlas<-image T_image<-head T_head<-tracker T_tracker<-device``."""
    path = device_to_atlas_chain(chain_graph, at=100.0)
    assert [(e.parent, e.child) for e in path.edges] == [
        ("atlas", "image"),
        ("image", "head"),
        ("head", "tracker"),
        ("tracker", "device"),
    ]
    manual = torch.eye(4, dtype=DTYPE)
    for e in path.edges:
        manual = manual @ e.pose.matrix
    assert torch.allclose(path.pose.matrix, manual, atol=1e-12)
    assert path.pose.label == "atlas<-device"
    assert path.units == "mm"


def test_edges_are_stored_separately_not_multiplied_and_forgotten(chain_graph) -> None:
    """§2.8's explicit requirement, as an executable check."""
    before = len(chain_graph.edges)
    path = chain_graph.path("atlas", "device", at=100.0).best
    assert len(chain_graph.edges) == before, "composing must not mutate the graph"
    # the composed result still points back at its four calibration records
    assert len(path.edges) == 4
    assert [e.calibration.method for e in path.edges] == [
        "affine_normalization_lsq",
        "fiducial_lsq",
        "tracker_fiducial_registration",
        "coil_tracker_geometry",
    ]
    rec = path.as_record()
    assert rec["edges"] == [
        "atlas<-image",
        "image<-head",
        "head<-tracker",
        "tracker<-device",
    ]
    assert path.provenance["edges_stored_separately"] is True
    # every individual edge is still independently retrievable and unchanged
    assert chain_graph.path("head", "tracker", at=100.0).best.pose is not None


def test_composed_pose_carries_units_handedness_and_epoch(chain_graph) -> None:
    p = chain_graph.path("atlas", "device", at=100.0).best.pose
    assert p.units == "mm"
    assert p.handedness is Handedness.RIGHT
    assert p.epoch == "sub-01_ses-01"


def test_unknown_frame_is_refused(chain_graph) -> None:
    with pytest.raises(UnknownFrameError) as exc:
        chain_graph.path("atlas", "hippocampus")
    assert exc.value.code == "R01"


def test_disconnected_frame_yields_no_path(chain_graph) -> None:
    chain_graph.add_frame(Frame("eye_camera", object="eye tracker camera", units="mm"))
    with pytest.raises(NoPathError) as exc:
        chain_graph.path("atlas", "eye_camera")
    assert "keep the source disconnected" in exc.value.remedy


# --------------------------------------------------------------------------
# units, handedness, determinant
# --------------------------------------------------------------------------


def test_rigid_edge_between_frames_with_different_units_is_refused(chain_graph) -> None:
    chain_graph.add_frame(Frame("coil_m", object="TMS coil (SI)", units="m"))
    with pytest.raises(UnitMismatchError) as exc:
        chain_graph.add_rigid(
            "tracker",
            "coil_m",
            Pose.identity("tracker", "coil_m", units="mm"),
        )
    assert exc.value.code == "R01"


def test_voxel_indices_reach_millimetres_only_through_a_declared_affine(chain_graph) -> None:
    chain_graph.add_frame(
        Frame("voxel", object="participant T1w volume", units="voxel", axes="i,j,k")
    )
    vox2mm = torch.eye(4, dtype=DTYPE)
    vox2mm[0, 0] = vox2mm[1, 1] = vox2mm[2, 2] = 0.8  # 0.8 mm isotropic
    vox2mm[:3, 3] = t([-90.0, -126.0, -72.0])
    chain_graph.add_edge(
        TransformEdge(
            "image",
            "voxel",
            "affine",
            matrix=vox2mm,
            calibration=CalibrationRecord(method="nifti_sform", n_observations=None),
        )
    )
    path = chain_graph.path("atlas", "voxel", at=100.0).best
    assert path.units == "mm"
    assert not path.linear or path.matrix is not None
    # a voxel index maps to millimetres only along that edge
    out = path.apply(t([[10.0, 20.0, 30.0]]))
    assert out.shape == (1, 3)
    with pytest.raises(UnitMismatchError):
        # and the scalar shortcut is still refused
        from scwbd.transforms.units import convert

        convert(10.0, "voxel", "mm")


def test_reflection_in_an_affine_edge_must_be_declared(chain_graph) -> None:
    chain_graph.add_frame(
        Frame("atlas_flipped", object="MNI152 template", units="mm", handedness="right")
    )
    M = torch.eye(4, dtype=DTYPE)
    M[0, 0] = -1.0
    with pytest.raises(HandednessError) as exc:
        chain_graph.add_edge(TransformEdge("atlas_flipped", "atlas", "affine", matrix=M))
    assert "mirror" in str(exc.value) or "det" in str(exc.value)
    assert exc.value.code == "R01"


def test_a_declared_mirror_must_change_the_frame_handedness(chain_graph) -> None:
    chain_graph.add_frame(
        Frame("atlas_lh", object="MNI152 template", units="mm", handedness="left")
    )
    M = torch.eye(4, dtype=DTYPE)
    M[0, 0] = -1.0
    edge = chain_graph.add_edge(
        TransformEdge(
            "atlas_lh", "atlas", "affine", matrix=M, reflection_declared=True,
            notes="left/right flipped template",
        )
    )
    assert edge.reflection_declared
    # ... and a rigid edge may not join frames of different handedness
    with pytest.raises(HandednessError):
        chain_graph.add_rigid(
            "atlas_lh", "image", Pose.identity("atlas_lh", "image", units="mm")
        )


def test_rigid_pose_units_must_match_the_frames(chain_graph) -> None:
    with pytest.raises(UnitMismatchError):
        chain_graph.add_rigid(
            "head", "device", Pose.identity("head", "device", units="m")
        )


# --------------------------------------------------------------------------
# invertibility
# --------------------------------------------------------------------------


def _projection_graph() -> FrameGraph:
    g = FrameGraph()
    g.add_frame(Frame("volume", object="participant T1w volume", units="mm"))
    g.add_frame(Frame("slice", object="single acquired slice", units="mm"))
    P = torch.eye(4, dtype=DTYPE)
    P[2, 2] = 0.0  # collapse z: a projection, not a transform
    g.add_edge(
        TransformEdge(
            "slice",
            "volume",
            "affine",
            matrix=P,
            invertible=False,
            notes="slice selection; no inverse exists",
        )
    )
    return g


def test_non_invertible_transform_is_refused(_=None) -> None:
    g = _projection_graph()
    forward = g.path("slice", "volume")
    assert len(forward) == 1
    edge = g.edges[0]
    with pytest.raises(NonInvertibleTransformError) as exc:
        edge.reversed_edge()
    assert "invertible=False" in str(exc.value)
    # the reverse direction has no admissible path, and says why
    reverse = g.path("volume", "slice", raise_if_empty=False)
    assert len(reverse) == 0
    assert reverse.rejected and "invertible=False" in reverse.rejected[0][1]
    # asking for the best path re-raises the *specific* refusal, not a generic
    # "no path": the caller needs to know an inverse was assumed where absent
    with pytest.raises(NonInvertibleTransformError):
        reverse.best


def test_singular_affine_claiming_invertibility_is_refused() -> None:
    P = torch.eye(4, dtype=DTYPE)
    P[2, 2] = 0.0
    with pytest.raises(NonInvertibleTransformError):
        TransformEdge("a", "b", "affine", matrix=P, invertible=True)


def test_deformable_edge_without_an_inverse_is_refused() -> None:
    warp = DeformableTransform(forward=lambda p: p * 1.01, method="test")
    with pytest.raises(NonInvertibleTransformError) as exc:
        TransformEdge("atlas", "image", "deformable", warp=warp, invertible=True)
    assert "No inverse assumed where absent" in exc.value.remedy


# --------------------------------------------------------------------------
# a registration-like warp, and the round trip that is measured, not assumed
# --------------------------------------------------------------------------


def _sinusoidal_warp(amplitude: float = 3.0, k: float = 0.03, n_iter: int = 4):
    """A nonlinear image->atlas warp with a fixed-point approximate inverse.

    This is what a real nonlinear registration looks like from the runtime's
    point of view: the inverse is *computed*, converges to a tolerance, and
    therefore leaves a small but nonzero round-trip residual.
    """

    def d(p: torch.Tensor) -> torch.Tensor:
        return amplitude * torch.sin(k * p.roll(1, dims=-1))

    def fwd(p: torch.Tensor) -> torch.Tensor:
        return p + d(p)

    def inv(q: torch.Tensor) -> torch.Tensor:
        p = q.clone()
        for _ in range(n_iter):
            p = q - d(p)
        return p

    return DeformableTransform(
        forward=fwd, inverse=inv, method="sinusoidal_demons_like", residual_rms=0.9
    )


def test_round_trip_residual_is_measured_on_a_registration_like_warp() -> None:
    g = FrameGraph()
    g.add_frame(Frame("atlas", object="MNI152 template", units="mm"))
    g.add_frame(Frame("image", object="participant T1w volume", units="mm"))
    g.add_edge(
        TransformEdge(
            "atlas",
            "image",
            "deformable",
            warp=_sinusoidal_warp(),
            calibration=CalibrationRecord(method="nonlinear_registration", residual_rms=0.9),
        )
    )
    gen = torch.Generator().manual_seed(4)
    pts = (torch.rand((256, 3), dtype=DTYPE, generator=gen) - 0.5) * 160.0
    rec = g.round_trip("atlas", "image", points=pts)
    assert rec["n_points"] == 256
    # nonzero -- the fixed-point inverse is approximate -- but bounded
    assert 0.0 < rec["point_residual_rms"] < 1e-3
    assert rec["point_residual_max"] >= rec["point_residual_rms"]
    assert "translation_residual" not in rec  # a warp has no single pose residual


def test_a_worse_inverse_gives_a_visibly_worse_round_trip() -> None:
    """The residual is a measurement: a sloppier inverse must show up in it."""
    residuals = []
    for n_iter in (1, 2, 4):
        g = FrameGraph()
        g.add_frame(Frame("atlas", object="MNI152 template", units="mm"))
        g.add_frame(Frame("image", object="participant T1w volume", units="mm"))
        g.add_edge(
            TransformEdge("atlas", "image", "deformable", warp=_sinusoidal_warp(n_iter=n_iter))
        )
        gen = torch.Generator().manual_seed(4)
        pts = (torch.rand((128, 3), dtype=DTYPE, generator=gen) - 0.5) * 160.0
        residuals.append(g.round_trip("atlas", "image", points=pts)["point_residual_rms"])
    assert residuals[0] > residuals[1] > residuals[2] > 0


def test_a_deformable_path_refuses_to_become_a_pose() -> None:
    g = FrameGraph()
    g.add_frame(Frame("atlas", object="MNI152 template", units="mm"))
    g.add_frame(Frame("image", object="participant T1w volume", units="mm"))
    g.add_edge(TransformEdge("atlas", "image", "deformable", warp=_sinusoidal_warp()))
    ps = g.path("atlas", "image")
    assert ps.paths[0].pose is None and not ps.paths[0].linear
    strict = g.path("atlas", "image", require_linear=True, raise_if_empty=False)
    assert len(strict) == 0
    assert "disguised as rigid pose error" in strict.rejected[0][1]


def test_rigid_round_trip_residual_is_reported_even_when_tiny(chain_graph) -> None:
    rec = chain_graph.round_trip("atlas", "device", at=100.0)
    assert rec["translation_residual"] < 1e-9
    assert "rotation_residual_rad" in rec
    assert rec["independent_paths"] is False  # same edges, inverted


# --------------------------------------------------------------------------
# multiple paths
# --------------------------------------------------------------------------


def _add_shortcut(g: FrameGraph, jitter=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)) -> None:
    """A second, independently measured atlas<-head route (e.g. scalp digitizer)."""
    direct = (
        g.path("atlas", "head", at=100.0).best.pose.matrix
        @ exp_se3(t(jitter))
    )
    g.add_rigid(
        "atlas",
        "head",
        Pose(direct, "atlas", "head", units="mm", epoch="sub-01_ses-01"),
        uncertainty=PoseUncertainty.isotropic(2.0, 8e-3),
        calibration=CalibrationRecord(
            method="scalp_digitizer_direct",
            n_observations=200,
            residual_rms=2.2,
            validity=ValidityInterval(0.0, 7200.0),
        ),
    )


def test_multiple_paths_are_all_returned_and_compared(chain_graph) -> None:
    _add_shortcut(chain_graph, jitter=(0.4, -0.3, 0.2, 1e-4, 0.0, 0.0))
    ps = chain_graph.path("atlas", "device", at=100.0)
    assert len(ps) >= 2, "both the 4-edge chain and the shortcut must be found"
    assert ps.disagreements
    d = ps.disagreements[0]
    assert d.translation > 0.0  # measured, not assumed
    assert d.mahalanobis is not None
    assert ps.agree  # within the stated ledger
    ps.require_agreement()
    assert ps.best.uncertainty is not None


def test_paths_that_disagree_beyond_their_ledger_are_refused(chain_graph) -> None:
    _add_shortcut(chain_graph, jitter=(25.0, -18.0, 12.0, 0.05, 0.0, 0.0))
    ps = chain_graph.path("atlas", "device", at=100.0)
    assert not ps.agree
    with pytest.raises(TransformError) as exc:
        ps.require_agreement()
    assert "do not average them" in exc.value.remedy
    assert "disagree beyond" in str(exc.value)


# --------------------------------------------------------------------------
# calibration validity (Appendix C layer 9)
# --------------------------------------------------------------------------


def test_expired_calibration_refuses_by_default(chain_graph) -> None:
    """head<-tracker is valid to t=3600; at t=5000 the path must refuse."""
    with pytest.raises(CalibrationExpiredError) as exc:
        chain_graph.path("head", "tracker", at=5000.0)
    assert exc.value.code == "R01"
    assert "outside its validity interval" in str(exc.value)
    assert "Recalibrate" in exc.value.remedy


def test_expired_calibration_can_inflate_instead_but_never_silently(chain_graph) -> None:
    inside = chain_graph.path("atlas", "device", at=100.0).best
    outside = chain_graph.path(
        "atlas", "device", at=5000.0, expiry_policy=ExpiryPolicy.INFLATE
    ).best
    assert outside.inflated and not inside.inflated
    assert outside.warnings, "an inflated path must say so"
    assert any("outside its validity interval" in w for w in outside.warnings)
    # 1400 s past a 900 s time constant: each expired edge's variance is
    # scaled by 2 ** (1400/900) ~ 2.94, and the whole-path variance grows
    expired = [c for c in outside.validity_checks if not c.inside]
    assert len(expired) == 2  # head<-tracker and tracker<-device
    for c in expired:
        assert c.extrapolation_distance == pytest.approx(1400.0)
        assert c.inflation_factor == pytest.approx(2.0 ** (1400.0 / 900.0))
    ratio = float(torch.trace(outside.uncertainty.cov)) / float(
        torch.trace(inside.uncertainty.cov)
    )
    assert ratio > 1.3
    rec = outside.as_record()
    assert rec["inflated"] is True
    assert any(not c["inside_validity"] for c in rec["validity_checks"])


def test_a_fired_recalibration_trigger_refuses_under_any_policy(chain_graph) -> None:
    for policy in (ExpiryPolicy.REFUSE, ExpiryPolicy.INFLATE):
        with pytest.raises(CalibrationExpiredError) as exc:
            chain_graph.path(
                "head",
                "tracker",
                at=100.0,
                expiry_policy=policy,
                triggers_fired=("tracker_moved",),
            )
        assert "recalibration trigger" in str(exc.value)


def test_frame_validity_interval_is_enforced() -> None:
    g = FrameGraph()
    g.add_frame(Frame("head", object="participant head", units="mm"))
    g.add_frame(
        Frame(
            "cap",
            object="EEG cap",
            units="mm",
            validity=ValidityInterval(0.0, 600.0),
            notes="cap removed after 10 minutes",
        )
    )
    g.add_rigid("head", "cap", Pose.identity("head", "cap", units="mm"))
    assert g.path("head", "cap", at=300.0).best is not None
    with pytest.raises(TransformError):
        g.path("head", "cap", at=900.0)


def test_frame_without_a_declared_physical_object_is_refused() -> None:
    with pytest.raises(UnknownFrameError) as exc:
        Frame("mystery", object="", units="mm")
    assert "physical object" in str(exc.value)


def test_expiry_policy_has_no_silent_option() -> None:
    with pytest.raises(TransformError) as exc:
        ExpiryPolicy.coerce("ignore")
    assert "not an option" in exc.value.remedy
