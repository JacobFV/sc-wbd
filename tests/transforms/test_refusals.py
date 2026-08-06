"""The refusal inventory, the schema adapter, and the determinism contract.

Each entry in :data:`REFUSALS` is one row of the definition of done for
build-order item 3, wired to the call that actually raises.  The point of this
module is that the refusals are *enumerable*: a compiler wiring
``scwbd.transforms`` into ``tab:compiler-refusals`` can read this list rather
than grep the source.
"""

from __future__ import annotations

import pytest
import torch

from scwbd.transforms import errors as E
from scwbd.transforms.calibration import CalibrationRecord, ExpiryPolicy
from scwbd.transforms.clock_graph import ClockGraph, ClockMap, ClockSpec, DropSpec
from scwbd.transforms.frame_graph import (
    DeformableTransform,
    Frame,
    FrameGraph,
    TransformEdge,
)
from scwbd.transforms.se3 import DTYPE, Pose, ValidityInterval
from scwbd.transforms.sheaf import (
    CoverageReport,
    Prolongation,
    Restriction,
    Section,
    Site,
    SupportObject,
    Cover,
    glue,
)
from scwbd.transforms.uncertainty import (
    PoseUncertainty,
    monte_carlo_propagate,
    propagate_first_order,
)


def t(x) -> torch.Tensor:
    return torch.as_tensor(x, dtype=DTYPE)


# --------------------------------------------------------------------------
# the callables that must refuse
# --------------------------------------------------------------------------


def _unit_mismatch():
    Pose.identity("a", "b", units="mm").compose(Pose.identity("b", "c", units="m"))


def _handedness_mismatch():
    Pose.identity("a", "b", units="mm", handedness="right").compose(
        Pose.identity("b", "c", units="mm", handedness="left")
    )


def _reflection():
    M = torch.eye(4, dtype=DTYPE)
    M[1, 1] = -1.0
    Pose(M, "a", "b", units="mm")


def _non_invertible():
    P = torch.eye(4, dtype=DTYPE)
    P[2, 2] = 0.0
    TransformEdge("a", "b", "affine", matrix=P, invertible=False).reversed_edge()


def _deformable_without_inverse():
    TransformEdge(
        "atlas", "image", "deformable",
        warp=DeformableTransform(forward=lambda p: p), invertible=True,
    )


def _epoch_mismatch():
    Pose.identity("a", "b", units="mm", epoch="ses-01").compose(
        Pose.identity("b", "c", units="mm", epoch="ses-02")
    )


def _expired_calibration():
    g = FrameGraph()
    g.add_frame(Frame("head", object="participant head", units="mm"))
    g.add_frame(Frame("tracker", object="optical tracker", units="mm"))
    g.add_rigid(
        "head", "tracker", Pose.identity("head", "tracker", units="mm"),
        calibration=CalibrationRecord(
            method="fiducial_lsq", validity=ValidityInterval(0.0, 100.0)
        ),
    )
    g.path("head", "tracker", at=1e4)


def _unknown_frame():
    FrameGraph().frame("nowhere")


def _no_path():
    g = FrameGraph()
    g.add_frame(Frame("a", object="thing a", units="mm"))
    g.add_frame(Frame("b", object="thing b", units="mm"))
    g.path("a", "b")


def _unknown_clock_relation():
    g = ClockGraph()
    g.add_clock(ClockSpec(id="eeg", rate_hz=1000, epoch=0.0))
    g.add_clock(ClockSpec(id="eye", rate_hz=500, epoch=0.0))
    g.align("eeg", "eye", 1.0)


def _unevidenced_clock_relation():
    g = ClockGraph()
    g.add_clock(ClockSpec(id="eeg", rate_hz=1000, epoch=0.0))
    g.add_clock(ClockSpec(id="eye", rate_hz=500, epoch=0.0))
    g.relate("eeg", "eye", ClockMap.affine(0.1), evidence="they_looked_aligned")


def _dropped_sample_read():
    c = ClockSpec(id="eeg", rate_hz=1000, epoch=0.0, dropped=(DropSpec(10, 5),))
    c.time_to_index(0.012)


def _linearization_invalid():
    from scwbd.transforms.uncertainty import linearization_error

    linearization_error(
        lambda x, c: torch.sigmoid(60.0 * (x * c - 1.0)),
        t([1.0]), t([1.0]), Sx=t([[0.3]]), Sc=t([[0.05]]),
        seed=1, n=2000, raise_on_invalid=True,
    )


def _inconsistent_cross_covariance():
    propagate_first_order(
        lambda x, c: x + c, t([0.0]), t([0.0]),
        Sx=t([[1.0]]), Sc=t([[1.0]]), Sxc=t([[3.0]]),
    )


def _prolongation_without_partner():
    Prolongation(
        restriction=None, matrix=None,
        coverage=CoverageReport(10, 0.1, 0.0, 0.9), prior_sd_unresolved=1.0,
    )


def _prolongation_without_coverage():
    R = Restriction("fine", "coarse", torch.ones((1, 3), dtype=DTYPE) / 3)
    Prolongation(restriction=R, matrix=None, coverage=None, prior_sd_unresolved=1.0)


def _cocycle_obstruction():
    s = Site()
    for oid, elems in [("U", (0, 1, 2)), ("Ua", (0, 1)), ("Ub", (1, 2)), ("Uc", (1,))]:
        s.add_object(SupportObject(oid, elems, units="V", tolerance=1e-6))
    def sel(sub, parent):
        M = torch.zeros((len(sub), len(parent)), dtype=DTYPE)
        for r, e in enumerate(sub):
            M[r, parent.index(e)] = 1.0
        return M
    s.add_restriction(Restriction("U", "Ua", sel((0, 1), (0, 1, 2))))
    s.add_restriction(Restriction("U", "Ub", sel((1, 2), (0, 1, 2))))
    s.add_restriction(Restriction("Ua", "Uc", sel((1,), (0, 1))))
    s.add_restriction(Restriction("Ub", "Uc", sel((1,), (1, 2))))
    s.add_cover(Cover("U", ("Ua", "Ub"), id="c"))
    res = glue(s, [Section("Ua", t([1.0, 2.0])), Section("Ub", t([9.0, 3.0]))], "c")
    res.global_section  # refuses to materialize a raster


def _missing_section_imputed():
    s = Site()
    for oid, elems in [("U", (0, 1)), ("Ua", (0,)), ("Ub", (1,))]:
        s.add_object(SupportObject(oid, elems, units="V"))
    s.add_restriction(Restriction("U", "Ua", t([[1.0, 0.0]])))
    s.add_restriction(Restriction("U", "Ub", t([[0.0, 1.0]])))
    s.add_cover(Cover("U", ("Ua", "Ub"), id="c"))
    glue(s, [Section("Ua", t([1.0]))], "c")


def _silent_expiry_policy():
    ExpiryPolicy.coerce("ignore")


#: (label, callable, expected exception, expected refusal code)
REFUSALS = [
    ("unit mismatch", _unit_mismatch, E.UnitMismatchError, "R01"),
    ("handedness mismatch", _handedness_mismatch, E.HandednessError, "R01"),
    ("reflection / det(R) < 0", _reflection, E.HandednessError, "R01"),
    ("non-invertible transform", _non_invertible, E.NonInvertibleTransformError, "R01"),
    (
        "deformable warp claiming an inverse",
        _deformable_without_inverse,
        E.NonInvertibleTransformError,
        "R01",
    ),
    ("epoch mismatch", _epoch_mismatch, E.EpochMismatchError, "R01"),
    ("expired calibration validity", _expired_calibration, E.CalibrationExpiredError, "R01"),
    ("unknown frame", _unknown_frame, E.UnknownFrameError, "R01"),
    ("no transform path", _no_path, E.NoPathError, "R01"),
    ("unknown clock relation", _unknown_clock_relation, E.ClockRelationUnknownError, "R01"),
    (
        "unevidenced clock relation",
        _unevidenced_clock_relation,
        E.ClockRelationUnknownError,
        "R01",
    ),
    ("read inside a dropped-sample gap", _dropped_sample_read, E.TransformError, None),
    ("first-order propagation on a threshold", _linearization_invalid, E.LinearizationInvalidError, None),
    ("inconsistent cross-covariance", _inconsistent_cross_covariance, E.CovarianceError, None),
    (
        "prolongation without restriction partner",
        _prolongation_without_partner,
        E.ProlongationWithoutRestrictionError,
        "R02",
    ),
    (
        "prolongation without tested coverage",
        _prolongation_without_coverage,
        E.ProlongationWithoutRestrictionError,
        "R02",
    ),
    ("cocycle/overlap obstruction", _cocycle_obstruction, E.CocycleObstructionError, "R03"),
    ("missing section on a cover", _missing_section_imputed, E.SiteError, "R01"),
    ("silent calibration reuse", _silent_expiry_policy, E.TransformError, None),
]


@pytest.mark.parametrize(
    "label,fn,exc,code", REFUSALS, ids=[r[0] for r in REFUSALS]
)
def test_refusal_raises_with_a_real_reason(label, fn, exc, code) -> None:
    with pytest.raises(exc) as got:
        fn()
    err = got.value
    assert isinstance(err, E.TransformError)
    assert err.reason and len(err.reason) > 20, "a refusal needs a real reason string"
    assert err.remedy, "a refusal must say what would make the request admissible"
    if code is not None:
        assert err.code == code
    rec = err.as_record()
    assert set(rec) == {"code", "error", "reason", "remedy", "offending_object"}


def test_every_refusal_is_machine_readable() -> None:
    """A compiler must be able to turn any of these into a CompilerRefusal."""
    for label, fn, exc, code in REFUSALS:
        try:
            fn()
        except E.TransformError as err:
            rec = err.as_record()
            assert rec["error"] == type(err).__name__
            assert isinstance(rec["reason"], str)
        else:  # pragma: no cover
            pytest.fail(f"{label} did not refuse")


def test_a_refusal_cannot_be_raised_without_a_reason() -> None:
    with pytest.raises(ValueError):
        E.TransformError("")


# --------------------------------------------------------------------------
# determinism (ARCHITECTURE.md §3: "Determinism is a test, not an aspiration")
# --------------------------------------------------------------------------


def test_monte_carlo_is_seed_deterministic() -> None:
    kw = dict(Sx=t([[1.0]]), Sc=t([[0.5]]), Sxc=t([[0.3]]), n=512)
    f = lambda x, c: (x * c).reshape(1)
    a = monte_carlo_propagate(f, t([1.0]), t([2.0]), seed=42, **kw)
    b = monte_carlo_propagate(f, t([1.0]), t([2.0]), seed=42, **kw)
    c = monte_carlo_propagate(f, t([1.0]), t([2.0]), seed=43, **kw)
    assert torch.equal(a.value, b.value) and torch.equal(a.cov, b.cov)
    assert not torch.equal(a.value, c.value)
    assert a.provenance["seed"] == 42


def test_chain_sampling_is_seed_deterministic() -> None:
    from scwbd.transforms.uncertainty import sample_chain

    p = [Pose.from_twist(t([1.0, 0, 0, 0, 0, 0.1]), "a", "b", units="mm")]
    u = [PoseUncertainty(torch.eye(6, dtype=DTYPE) * 1e-3)]
    assert torch.equal(sample_chain(p, u, n=64, seed=7), sample_chain(p, u, n=64, seed=7))


def test_covariance_math_never_runs_in_reduced_precision() -> None:
    """ARCHITECTURE.md §3 forbids bf16/f32 in covariance propagation."""
    r = propagate_first_order(
        lambda x, c: x + c, t([1.0]), t([1.0]), Sx=t([[1.0]]), Sc=t([[1.0]])
    )
    assert r.cov.dtype is torch.float64
    assert r.bias.dtype is torch.float64
    assert DTYPE is torch.float64


# --------------------------------------------------------------------------
# the schema adapter (agent A's types)
# --------------------------------------------------------------------------


def test_schema_adapter_round_trips_frames_and_edges() -> None:
    from scwbd.transforms import schema_adapter as A

    g = FrameGraph()
    g.add_frame(
        Frame("head", object="participant head", origin="nasion", axes="R,A,S", units="mm")
    )
    g.add_frame(Frame("tracker", object="optical tracker", axes="X,Y,Z", units="mm"))
    g.add_rigid(
        "head",
        "tracker",
        Pose.from_twist(t([1.0, 2.0, 3.0, 0.01, 0.0, 0.0]), "head", "tracker", units="mm"),
        uncertainty=PoseUncertainty.isotropic(0.5, 1e-3),
        calibration=CalibrationRecord(method="fiducial_lsq", n_observations=5),
    )
    doc = A.frame_graph_to_schema(g, root="head")
    assert {n["id"] for n in doc["nodes"]} == {"head", "tracker"}
    # the schema edge points src=child -> dst=parent
    assert (doc["edges"][0]["src"], doc["edges"][0]["dst"]) == ("tracker", "head")

    rebuilt = A.frame_graph_from_schema(doc)
    assert set(rebuilt.frames) == set(g.frames)
    a = g.path("head", "tracker").best.pose.matrix
    b = rebuilt.path("head", "tracker").best.pose.matrix
    assert torch.allclose(a, b)


@pytest.mark.skipif(
    not __import__("scwbd.transforms.schema_adapter", fromlist=["x"]).SCHEMA_AVAILABLE,
    reason="scwbd.schema contract types not available yet",
)
def test_adapter_consumes_real_pydantic_schema_objects() -> None:
    """The declaration in scwbd.schema becomes a runnable frame graph."""
    from scwbd.schema.frames import (
        CalibrationManifest,
        FrameEdge,
        FrameGraphSpec,
        FrameNode,
    )
    from scwbd.schema.priors import DiracPrior
    from scwbd.transforms import schema_adapter as A

    nodes = (
        FrameNode(
            id="head",
            object="participant head",
            origin="nasion",
            axes=("R", "A", "S"),
            handedness="right",
            units="mm",
            validity_interval=(0.0, 3600.0),
        ),
        FrameNode(
            id="tracker",
            object="optical tracker base",
            origin="camera optical centre",
            axes=("X", "Y", "Z"),
            handedness="right",
            units="mm",
        ),
    )
    twist = [12.0, -4.0, 250.0, 0.1, 0.0, 0.0]
    edge = FrameEdge(
        src="tracker",
        dst="head",
        transform="rigid",
        lineage="neuronavigation fiducial registration, Polaris Vega",
        parameters={f"twist_{i}": DiracPrior(value=v) for i, v in enumerate(twist)},
        calibration=CalibrationManifest(
            id="tracker_cal",
            fitting_method="fiducial_lsq",
            n_observations=4,
            residual=1.1,
            validity_interval=(0.0, 3600.0),
            recalibration_triggers=("tracker_moved",),
        ),
    )
    spec = FrameGraphSpec(root="head", nodes=nodes, edges=(edge,))

    g = A.frame_graph_from_schema(spec)
    assert set(g.frames) == {"head", "tracker"}
    path = g.path("head", "tracker", at=100.0).best
    assert path.pose.units == "mm"
    assert torch.allclose(path.pose.log(), t(twist), atol=1e-10)
    assert path.edges[0].calibration.method == "fiducial_lsq"
    assert path.edges[0].calibration.recalibration_triggers == ("tracker_moved",)
    # both declared validity intervals are enforced, and they are different
    # refusals: the *frame definition* expires first (a head frame defined by
    # fiducials stops existing), and it has no inflate-the-ledger reading.
    with pytest.raises(E.ValidityIntervalError) as exc:
        g.path("head", "tracker", at=1e4)
    assert "no uncertainty-inflation reading" in exc.value.remedy

    # with the frame itself unbounded, it is the calibration that expires
    open_nodes = (nodes[0].model_copy(update={"validity_interval": None}), nodes[1])
    g2 = A.frame_graph_from_schema(
        FrameGraphSpec(root="head", nodes=open_nodes, edges=(edge,))
    )
    with pytest.raises(E.CalibrationExpiredError):
        g2.path("head", "tracker", at=1e4)


@pytest.mark.skipif(
    not __import__("scwbd.transforms.schema_adapter", fromlist=["x"]).SCHEMA_AVAILABLE,
    reason="scwbd.schema contract types not available yet",
)
def test_an_uncertain_schema_edge_is_not_turned_into_a_measurement() -> None:
    """A prior over a transform is a declaration, not a fitted pose."""
    from scwbd.schema.frames import FrameEdge
    from scwbd.schema.priors import NormalPrior
    from scwbd.transforms import schema_adapter as A

    edge = FrameEdge(
        src="tracker",
        dst="head",
        transform="rigid",
        lineage="tracker read",
        parameters={"twist_0": NormalPrior(loc=1.0, scale=0.5)},
    )
    with pytest.raises(E.TransformError) as exc:
        A.edge_from_schema(edge, units="mm")
    assert "declaration, not a measurement" in exc.value.remedy
    assert "prior mean" in exc.value.remedy
    # ... but supplying the fitted transform works
    e = A.edge_from_schema(edge, matrix=torch.eye(4, dtype=DTYPE), units="mm")
    assert e.kind == "rigid" and e.pose.label == "head<-tracker"


@pytest.mark.skipif(
    not __import__("scwbd.transforms.schema_adapter", fromlist=["x"]).SCHEMA_AVAILABLE,
    reason="scwbd.schema contract types not available yet",
)
def test_an_edge_without_lineage_is_refused_R01() -> None:
    from scwbd.schema.frames import FrameEdge
    from scwbd.transforms import schema_adapter as A

    edge = FrameEdge(src="tracker", dst="head", transform="identity", lineage="")
    with pytest.raises(E.TransformError) as exc:
        A.edge_from_schema(edge, units="mm")
    assert "declares no lineage" in str(exc.value)
    assert "R01" in exc.value.remedy


@pytest.mark.skipif(
    not __import__("scwbd.transforms.schema_adapter", fromlist=["x"]).SCHEMA_AVAILABLE,
    reason="scwbd.schema contract types not available yet",
)
def test_schema_clock_specs_become_a_runtime_clock_graph() -> None:
    from scwbd.schema.clocks import ClockEdge as SClockEdge
    from scwbd.schema.clocks import ClockSpec as SClockSpec
    from scwbd.transforms import schema_adapter as A

    eeg = SClockSpec(id="eeg_amp", dt=0.001, jitter_sd=2e-5, group_delay=0.004,
                     interpolation="linear")
    vol = SClockSpec(
        id="scanner_volume",
        dt=0.72,
        reference="eeg_amp",
        offset=0.013,
        drift=4e-5,
        integration_window=0.72,
        interpolation="event_exact",
        sync_evidence="physical_trigger",
    )
    rel = SClockEdge(
        src="scanner_volume", dst="eeg_amp", offset=0.013, drift=4e-5,
        sync_evidence="physical_trigger", n_observations=400, residual=5e-4,
    )
    g = A.clock_graph_from_schema([eeg, vol], [rel])
    assert g.clock("eeg_amp").dt == pytest.approx(0.001)
    # "event_exact" means the volume is not resampleable in time
    assert g.clock("scanner_volume").interpolation_policy == "none"
    aligned = g.align("scanner_volume", "eeg_amp", 100.0)
    assert aligned.time == pytest.approx(100.0 + 0.013 + 4e-5 * 100.0)
    assert aligned.sd > 0.0


@pytest.mark.skipif(
    not __import__("scwbd.transforms.schema_adapter", fromlist=["x"]).SCHEMA_AVAILABLE,
    reason="scwbd.schema contract types not available yet",
)
def test_unverified_sync_evidence_is_refused_R01() -> None:
    """The schema's own UNVERIFIED_SYNC set must not be launderable."""
    from scwbd.schema.clocks import UNVERIFIED_SYNC
    from scwbd.transforms import schema_adapter as A

    for bad in sorted(UNVERIFIED_SYNC):
        with pytest.raises(E.ClockRelationUnknownError) as exc:
            A.clock_edge_from_schema(
                {"src": "a", "dst": "b", "offset": 0.1, "sync_evidence": bad}
            )
        assert exc.value.code == "R01"
        assert "UNVERIFIED_SYNC" in exc.value.remedy


@pytest.mark.skipif(
    not __import__("scwbd.transforms.schema_adapter", fromlist=["x"]).SCHEMA_AVAILABLE,
    reason="scwbd.schema contract types not available yet",
)
def test_a_handedness_free_frame_cannot_enter_an_se3_path() -> None:
    from scwbd.transforms import schema_adapter as A

    with pytest.raises(E.HandednessError) as exc:
        A.frame_from_schema(
            {
                "id": "alpha_band",
                "object": "spectral band",
                "handedness": "not_applicable",
                "units": "Hz",
            }
        )
    assert "resolution poset" in exc.value.remedy


def test_index_coordinate_frames_keep_their_unit_distinction() -> None:
    from scwbd.transforms import schema_adapter as A

    f = A.frame_from_schema(
        {
            "id": "voxel",
            "object": "participant T1w volume",
            "handedness": "right",
            "units": "dimensionless",
            "coordinate_type": "index",
        }
    )
    assert f.units == "index"  # not a length; needs an affine edge to become mm


def test_schema_adapter_round_trips_clocks() -> None:
    from scwbd.transforms import schema_adapter as A

    g = ClockGraph()
    g.add_clock(
        ClockSpec(
            id="eeg",
            rate_hz=1000,
            epoch=0.0,
            group_delay_s=0.004,
            jitter_sd_s=2e-5,
            dropped=(DropSpec(10, 2),),
        )
    )
    g.add_clock(ClockSpec(id="scanner", rate_hz=1.25, epoch=0.01))
    g.relate("scanner", "eeg", ClockMap.affine(0.01), evidence="physical_trigger")
    doc = A.clock_graph_to_schema(g)
    assert doc["relations"][0]["sync_evidence"] == "physical_trigger"
    rebuilt = A.clock_graph_from_schema(doc["clocks"], doc["relations"])
    assert rebuilt.clock("eeg").group_delay_s == 0.004
    assert rebuilt.clock("eeg").dropped[0].count == 2
    assert rebuilt.align("scanner", "eeg", 5.0).time == pytest.approx(5.01)


def test_schema_adapter_refuses_a_silently_defaulted_field() -> None:
    from scwbd.transforms import schema_adapter as A

    with pytest.raises(E.TransformError) as exc:
        A.frame_from_schema({"id": "head", "object": "participant head"})  # no handedness
    assert "declares none of" in str(exc.value)
    assert "silently defaulted" in exc.value.remedy


def test_schema_adapter_reports_whether_the_schema_package_exists() -> None:
    from scwbd.transforms import schema_adapter as A

    assert isinstance(A.SCHEMA_AVAILABLE, bool)
    # an empty scwbd/schema/ directory still imports as a namespace package, so
    # availability means "the contract types are there", not "the import worked"
    if not A.SCHEMA_AVAILABLE:
        with pytest.raises(E.TransformError) as exc:
            A._require_schema()
        assert "agent A's schema kernel" in exc.value.remedy
    else:
        for name in A._REQUIRED_SCHEMA_NAMES:
            assert hasattr(A._schema, name)


def test_uncertainty_ledger_adapter_keeps_bias_visible() -> None:
    """An UncertaintyLedger's bias interval collapses to a midpoint -- loudly."""
    from scwbd.transforms import schema_adapter as A

    u = A.uncertainty_from_schema(
        {"variance": {"measurement": 0.25, "session": 0.25}, "bias_interval": (-1.0, 3.0)}
    )
    assert float(u.cov[0, 0]) == pytest.approx(0.5)
    assert float(u.bias[0]) == pytest.approx(1.0)  # interval midpoint
    assert A.uncertainty_from_schema(None) is None
    # and the round trip back to a ledger keeps bias and variance separate
    doc = A.uncertainty_to_schema(u)
    assert set(doc["variance"]) == {"translation", "rotation"}
    assert doc["bias_interval"] == (1.0, 1.0)
    assert doc["bias_status"] == "externally_bounded"
