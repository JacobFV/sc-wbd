"""Agent A's compiler as the authority over the foundation model.

The foundation module currently hand-rolls three things the compiler already
owns and owns *better*, because it owns them declaratively:

1. **Gradient permissions.**  :class:`~scwbd.foundation.mixture.SourceSpec`
   carries globs over torch parameter names.  Those globs are written by hand,
   they are not checked against anything, and a typo silently grants nothing.
   ``scwbd.compiler`` compiles a :class:`~scwbd.schema.sources.GradientPermission`
   into a boolean mask over *named parameter groups*, reports every permission
   entry that matched nothing, and reports every group no source can update.
2. **The state-layout ABI.**  ``scwbd.foundation.state.StateLayout`` packs
   offsets for one region; ``scwbd.compiler.layout.StateLayout`` packs the whole
   model, with byte offsets, dtypes, clocks and an ``abi_digest`` two processes
   can compare before exchanging boundary state.
3. **The multirate schedule.**  The model hard-codes ``(t + 1) % hemo_ratio``;
   the compiler emits an exact integer-nanosecond hyperperiod, the sync points,
   per-field update policies and the interpolation contract between clocks.

This module is the bridge.  It builds a *valid* :class:`BrainSchema` for
SC-WBD-001-beta, compiles it, and translates the compiler's answers back into
the vocabulary the foundation trainer speaks: glob patterns over
``model.named_parameters()``.

Why a translation table at all?  Because the two namespaces are genuinely
different and neither can be derived from the other.  The compiler names
*things in the world* (``operator:long_range:delay``); torch names *tensors in
an implementation* (``coupling.bin_length_mm``).  :data:`FOUNDATION_BINDING` and
:data:`FOUNDATION_FROZEN_BINDING` are the explicit, auditable map between them,
and :func:`audit_binding` exists so that any torch parameter the map fails to
claim is **visible** rather than quietly ungoverned.  An unclaimed parameter is a
hole in the permission system.

The map is checked in both directions, and that is the point.  A declared
binding whose pattern matches no tensor on the model is reported as a problem,
not tolerated: it means someone renamed a tensor and the permission it expressed
became decorative while continuing to look enforced.  "Implemented but frozen"
(a buffer) and "no implementation state at all" are both sayable, but each has
to be *said* -- neither is what you get by writing a pattern that misses.

Failure policy: fail closed and loudly.  If ``scwbd.compiler`` or
``scwbd.schema`` cannot be imported, or the schema fails to build, every
function here raises.  The foundation module has its own fallback path and it,
not this module, decides when to take it -- use :func:`compiler_available`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from fnmatch import fnmatchcase
from typing import Any, Iterable, Mapping, Sequence

from .anatomy import AnatomyPrior
from .mixture import GradientGate, SourceSpec
from .state import StateLayout as FoundationStateLayout
from .state import default_layout
from .util import logical_param_name

__all__ = [
    "FOUNDATION_BINDING",
    "FOUNDATION_FROZEN_BINDING",
    "FRAME_EDGE_KEY",
    "REGION_STATE_KEY",
    "OBSERVATION_HEADS",
    "CompilerBridgeError",
    "CompilerUnavailable",
    "SchemaBuildError",
    "compiler_available",
    "build_foundation_schema",
    "build_foundation_claim",
    "compile_foundation",
    "bind_masks",
    "gate_from_compiled",
    "audit_binding",
    "schedule_plan",
    "patterns_for_group",
    "frozen_patterns_for_group",
    "observation_head_for",
]

_log = logging.getLogger(__name__)


# ======================================================================
# graceful degradation: import once, remember exactly why it failed
# ======================================================================
class CompilerBridgeError(RuntimeError):
    """Base class for every failure this bridge reports."""


class CompilerUnavailable(CompilerBridgeError):
    """``scwbd.compiler`` / ``scwbd.schema`` could not be imported."""


class SchemaBuildError(CompilerBridgeError):
    """A valid ``BrainSchema`` could not be constructed from the inputs."""


_IMPORT_ERROR: BaseException | None = None

try:  # pragma: no cover - exercised by the availability test below
    from ..compiler import CompiledModel
    from ..compiler import compile as _compile_schema
    from ..schema.claims import ClaimManifest
    from ..schema.clocks import ClockSpec
    from ..schema.frames import CalibrationManifest, FrameEdge, FrameGraphSpec, FrameNode
    from ..schema.ids import ClockId, FrameId, ScaleId
    from ..schema.ledger import UncertaintyLedger
    from ..schema.lineage import Identity, LineageUnit
    from ..schema.operators import Identification, OperatorSpec, ResidualPolicy
    from ..schema.poset import ResolutionPoset, ScaleNode
    from ..schema.priors import LogNormalPrior, NormalPrior
    from ..schema.regions import AtlasRef, Region
    from ..schema.schema import BrainSchema
    from ..schema.sources import (
        Governance,
        GradientPermission,
        HierarchicalEffect,
        Missingness,
        ObservationModel,
        PopulationStructure,
        SignalSpec,
        SourceCard,
        SplitPolicy,
        TemporalHoldout,
    )
    from ..schema.state import ComponentSpec, Port, StateSpec
    from ..schema.supports import PSF, Support, TemporalSupport
    from ..schema.units import Unit
except ImportError as exc:  # pragma: no cover - only when agent A/B are absent
    _IMPORT_ERROR = exc


def compiler_available() -> bool:
    """True when ``scwbd.compiler`` and ``scwbd.schema`` are importable.

    The foundation module owns the decision about what to do when they are not;
    this is the predicate it should branch on, never a silent try/except around
    a call into this module.
    """
    return _IMPORT_ERROR is None


def _require_compiler() -> None:
    if _IMPORT_ERROR is not None:
        raise CompilerUnavailable(
            "scwbd.compiler / scwbd.schema is unavailable, so the compiler cannot be "
            "made the authority over gradient permissions, the state-layout ABI or "
            f"the multirate schedule. Original import error: {_IMPORT_ERROR!r}. "
            "Check compiler_available() before calling into scwbd.foundation."
            "compiler_bridge and take the foundation module's own fallback path "
            "explicitly -- this bridge will not fall back for you."
        ) from _IMPORT_ERROR


# ======================================================================
# the binding table: compiler parameter groups -> torch parameter names
# ======================================================================
#: Template key for the per-region state groups.  ``parameter_groups_of`` emits
#: one group per (region, component); the foundation model's per-region tensors
#: are a single family shared by every component of that region, so the whole
#: family is bound once under this template rather than sliced per component
#: (the model has no per-component parameters -- the state components are
#: *activations*, not weights).
REGION_STATE_KEY = "region:<id>:state:<component>"

#: Template key for the frame-graph calibration groups.  ``parameter_groups_of``
#: emits one ``frame_edge:<src>-><dst>:calibration`` per frame edge that carries
#: transform parameters; this model has exactly one such edge (anatomical RAS ->
#: EEG cap), and it is realised as *frozen geometry* -- see
#: :data:`FOUNDATION_FROZEN_BINDING`.
FRAME_EDGE_KEY = "frame_edge:<src>-><dst>:calibration"

#: Canonical exemplar region id used for the port keys.  Port groups are emitted
#: as ``port:<region_id>.<port_name>``; the foundation model's port machinery is
#: weight-shared across every region, so the binding is keyed on the port *name*
#: through this exemplar.
_PORT_EXEMPLAR = "cortex"

#: Observation head names.  ``parameter_groups_of`` emits
#: ``observation:<source_id>:nuisance``; the source id is chosen by whoever
#: writes the source card, so the binding is resolved through the head the
#: source's :class:`ObservationModel` names (see :func:`observation_head_for`).
OBSERVATION_HEADS: tuple[str, ...] = ("parcel_activity", "eeg", "bold", "behaviour")

#: Compiler parameter-group name (or template) -> glob patterns over
#: ``SCWBD.named_parameters()``.  Read this as the answer to "which *trainable*
#: tensors does this declared thing consist of?".
#:
#: Every pattern here **must** match at least one trainable parameter.  A pattern
#: that matches nothing is a bug, not a curiosity: it means the declaration and
#: the implementation have drifted (a tensor was renamed, a module restructured)
#: and the permission it expresses has silently become decorative.  A group whose
#: implementation is genuinely *not* trainable does not belong in this table --
#: it belongs in :data:`FOUNDATION_FROZEN_BINDING` (frozen buffer) or is mapped
#: to ``()`` here (no tensor at all).  Those three statements are different and
#: the audit keeps them apart; see :class:`_Binding`.
FOUNDATION_BINDING: dict[str, tuple[str, ...]] = {
    # F_local: the weight-shared regional vector field, its per-region timestep
    # scaling and the conditioning encoder that modulates it.
    "operator:local_field:params": ("local.*", "log_dt_scale", "context.*"),
    # R_theta: the learned residual, kept separable so R05 can price it.
    "operator:local_field:residual": ("residual.*",),
    # F_long: the typed evidence-class gains of the connectome-masked coupling.
    "operator:long_range:params": (
        "coupling.gain_soft",
        "coupling.gain_proposed",
        "coupling.global_scale",
    ),
    # The delay geometry is a *buffer* cut from tract length: declared, frozen by
    # construction, deliberately not trainable.  Its tensor is named in
    # FOUNDATION_FROZEN_BINDING, which is where "implemented, but not by a
    # gradient" is said.
    "operator:long_range:delay": (),
    # The anat->EEG-cap co-registration is likewise frozen geometry, baked into
    # the lead field at build time; same story, see FOUNDATION_FROZEN_BINDING.
    FRAME_EDGE_KEY: (),
    # Typed ports: the export projection and the read-in of arriving messages.
    f"port:{_PORT_EXEMPLAR}.message_out": ("msg_proj.*",),
    f"port:{_PORT_EXEMPLAR}.message_in": ("msg_readin.*",),
    # Observation ports are implemented by their heads; the head's nuisance
    # terms are additionally reachable through the observation groups below.
    f"port:{_PORT_EXEMPLAR}.eeg_out": ("eeg.*",),
    f"port:{_PORT_EXEMPLAR}.bold_out": ("bold.*",),
    f"port:{_PORT_EXEMPLAR}.behaviour_out": ("behaviour.*",),
    # The assimilation encoder (observed window -> initial structured state).
    "operator:assimilation:params": ("assimilate.*",),
    # Declared-empty: the local field and the assimilation encoder are
    # instantaneous, so ``parameter_groups_of`` emits a delay group for them that
    # corresponds to no tensor at all.  Binding them to the empty tuple states
    # that on purpose, which is a different claim from "nobody wrote the binding".
    "operator:local_field:delay": (),
    "operator:assimilation:delay": (),
    # Per-source observation nuisance terms, resolved by head.
    "observation:parcel_activity:nuisance": ("readout.*",),
    "observation:eeg:nuisance": ("eeg.*",),
    "observation:bold:nuisance": ("bold.*",),
    "observation:behaviour:nuisance": ("behaviour.*",),
    # Per-region parameters.  Every tensor whose leading axis is N_regions and
    # is therefore *the region's own* parameter rather than a shared weight.
    REGION_STATE_KEY: (
        "local.embed",
        "local.films.*.region_scale",
        "local.films.*.region_shift",
        "residual.embed",
        "assimilate.embed",
    ),
}

#: Compiler parameter-group name (or template) -> glob patterns over
#: ``SCWBD.named_buffers()``.
#:
#: Some declared things are real, are implemented, and are still not trainable:
#: they are fixed geometry the model *derives* rather than learns.  Saying that
#: with an empty entry in :data:`FOUNDATION_BINDING` would be a strictly weaker
#: claim -- "no tensor" rather than "this tensor, frozen" -- and would leave the
#: rename undetectable.  So the buffer is named here and **must exist**: every
#: pattern must match at least one registered buffer, or the audit reports a
#: problem exactly as it would for a missing parameter.
#:
#: Nothing in this table is ever handed to a source as a trainable glob.  A
#: source may be *granted* such a group by the compiler (the schema models the
#: delay as part of the long-range operator) without that grant reaching a
#: gradient -- which is the honest outcome, and is recorded as ``frozen_groups``
#: in :func:`audit_binding` rather than being rounded off to a permission.
FOUNDATION_FROZEN_BINDING: dict[str, tuple[str, ...]] = {
    # Conduction-delay bin edges, cut from tract length by the anatomy prior.
    "operator:long_range:delay": ("coupling.bin_length_mm",),
    # The rigid anat->cap transform is applied when the lead field is built, so
    # the co-registration lives inside L and there is no separate tx/ty/tz
    # tensor.  Binding to the lead field states where the calibration went; a
    # per-participant digitization would rebuild L, not train it.
    FRAME_EDGE_KEY: ("eeg.L",),
}


#: Port-name -> patterns, derived from the exemplar keys above so the two can
#: never drift apart.
_PORT_BINDING: dict[str, tuple[str, ...]] = {
    key.split(".", 1)[1]: pats
    for key, pats in FOUNDATION_BINDING.items()
    if key.startswith(f"port:{_PORT_EXEMPLAR}.")
}


def observation_head_for(spec: SourceSpec) -> str:
    """Which readout head a source's observations enter through.

    The mapping is a heuristic over the source id and notes because a
    ``SourceSpec`` carries no modality field; it is applied **once**, at schema
    build time, and then recorded in ``ObservationModel.name`` so that every
    later lookup reads the declaration rather than re-guessing.
    """
    text = f"{spec.id} {spec.notes}".lower()
    if any(tok in text for tok in ("eeg", "meg", "ieeg", "lfp", "scalp")):
        return "eeg"
    if any(tok in text for tok in ("bold", "fmri", "hemo", "nirs")):
        return "bold"
    if any(tok in text for tok in ("behav", "task", "response", "reaction")):
        return "behaviour"
    return "parcel_activity"


def patterns_for_group(group: str, *, schema: Any | None = None) -> tuple[str, ...] | None:
    """Torch-parameter globs implied by one compiler parameter group.

    Returns ``None`` when the binding table says nothing about the group -- the
    caller must report that, never treat it as "no parameters".  An empty tuple
    is a different statement: "this group is declared to have no trainable
    tensor".
    """
    if group in FOUNDATION_BINDING:
        return FOUNDATION_BINDING[group]

    parts = group.split(":")
    kind = parts[0]

    # Template lookups use ``.get``: a missing template is "nobody wrote this
    # binding", which is the unbound path the caller must report.  Raising here
    # instead would turn a reportable hole into a crash inside the audit that is
    # supposed to describe it.
    if kind == "region" and len(parts) == 4 and parts[2] == "state":
        return FOUNDATION_BINDING.get(REGION_STATE_KEY)

    if kind == "frame_edge" and len(parts) == 3 and parts[2] == "calibration":
        return FOUNDATION_BINDING.get(FRAME_EDGE_KEY)

    if kind == "port" and len(parts) == 2:
        _, _, port_name = parts[1].partition(".")
        if port_name in _PORT_BINDING:
            return _PORT_BINDING[port_name]
        return None

    if kind in ("observation", "intervention") and len(parts) == 3:
        head = _head_of_source(parts[1], schema)
        if head is not None:
            key = f"observation:{head}:nuisance"
            if key in FOUNDATION_BINDING:
                return FOUNDATION_BINDING[key]
        return None

    # Last resort: the table may hold a glob (none do today, but a downstream
    # model variant may add one and this keeps the contract honest).
    for key, pats in FOUNDATION_BINDING.items():
        if any(ch in key for ch in "*?[") and fnmatchcase(group, key):
            return pats
    return None


def frozen_patterns_for_group(group: str) -> tuple[str, ...]:
    """Buffer globs implied by one compiler parameter group.

    Empty means "this group is not implemented by frozen state" -- which, for a
    group whose :func:`patterns_for_group` is also empty, is the genuine
    "declared, corresponds to no tensor at all" case.
    """
    if group in FOUNDATION_FROZEN_BINDING:
        return FOUNDATION_FROZEN_BINDING[group]
    parts = group.split(":")
    if parts[0] == "frame_edge" and len(parts) == 3 and parts[2] == "calibration":
        return FOUNDATION_FROZEN_BINDING.get(FRAME_EDGE_KEY, ())
    if parts[0] == "region" and len(parts) == 4 and parts[2] == "state":
        return FOUNDATION_FROZEN_BINDING.get(REGION_STATE_KEY, ())
    return ()


def _head_of_source(source_id: str, schema: Any | None) -> str | None:
    """Head name declared by a source card, falling back to its bare id."""
    if schema is not None:
        try:
            card = schema.source(source_id)
        except KeyError:
            card = None
        if card is not None and card.observation is not None:
            if card.observation.name in OBSERVATION_HEADS:
                return card.observation.name
    return source_id if source_id in OBSERVATION_HEADS else None


# ======================================================================
# schema construction
# ======================================================================
#: Clock ids.  Two clocks: the neural state and the haemodynamic compartments.
FAST_CLOCK = "scwbd_neural_fast"
SLOW_CLOCK = "scwbd_hemo_slow"

#: Frame ids.
ANAT_FRAME = "scwbd_anat_RAS"
EEG_FRAME = "scwbd_eeg_cap"

#: Resolution poset element every region lives at.
PARCEL_SCALE = "parcel"

#: Session length used for frame/calibration validity intervals, seconds.
_SESSION_VALIDITY = (0.0, 7200.0)

#: Where the Stage-V hierarchical-effect recovery test is reported.  The schema
#: *declares the contract*: if this report does not exist, or shows recovery
#: failing, R07's premise is false and the declaration is a lie the reviewer can
#: catch by opening the file named here.
DEFAULT_RECOVERY_REPORT = "reports/foundation/stage_v_effect_recovery.json"


def _estimable(units: str, *, estimator: str, bias: tuple[float, float] = (0.0, 0.0), **variance: float):
    """A design-estimable ledger: replication/randomization names its estimator."""
    return UncertaintyLedger(
        variance={k: v for k, v in variance.items() if v > 0.0},
        bias_interval=bias,
        bias_status="design_estimable",
        bias_estimator=estimator,
        validity_domain={"regime": "scwbd_001_beta_population_model"},
        units=units,
    )


def _bounded(units: str, *, source: str, bias: tuple[float, float], **variance: float):
    """An externally-bounded ledger: a phantom/anchor supplies the interval."""
    return UncertaintyLedger(
        variance={k: v for k, v in variance.items() if v > 0.0},
        bias_interval=bias,
        bias_status="externally_bounded",
        external_bound_source=source,
        validity_domain={"regime": "scwbd_001_beta_population_model"},
        units=units,
    )


def _sensitivity(
    units: str,
    *,
    bias: tuple[float, float],
    discrepancy: float | None = None,
    **variance: float,
):
    """A prior-specified-sensitivity ledger: swept over a declared range.

    R08 admits this status only when the bias is carried as a genuine *range*.
    A degenerate interval here is a point estimate wearing a range's clothes, so
    it is rejected at construction rather than at compile time.
    """
    if bias[0] >= bias[1]:
        raise SchemaBuildError(
            f"prior-specified-sensitivity bias must be a non-degenerate interval, got {bias}"
        )
    return UncertaintyLedger(
        variance={k: v for k, v in variance.items() if v > 0.0},
        bias_interval=bias,
        bias_status="prior_specified_sensitivity",
        model_discrepancy=discrepancy,
        validity_domain={"regime": "scwbd_001_beta_population_model"},
        units=units,
    )


def _calibration(
    cid: str,
    layers: tuple[str, ...],
    *,
    residual: float,
    tolerance: float,
    residual_units: str = "m",
):
    return CalibrationManifest(
        id=cid,
        layers=layers,  # type: ignore[arg-type]
        n_observations=64,
        fitting_method="least_squares_with_bootstrap_residuals",
        residual=residual,
        residual_units=Unit(residual_units),
        residual_tolerance=tolerance,
        validity_interval=_SESSION_VALIDITY,
        validity_domain={"session": "scwbd_001_beta"},
        extrapolation_distance=0.0,
        recalibration_triggers=("device_change", "participant_regeometry", "session_end"),
        units_checked=True,
        handedness_checked=True,
        roundtrip_checked=True,
        inverse_available=True,
        ledger=_bounded(
            residual_units,
            source=f"{cid}_phantom",
            bias=(-residual, residual),
            measurement=residual**2,
        ),
    )


def _frames():
    """Anatomical root frame plus the EEG sensor frame that hangs off it."""

    def node(fid: str, obj: str, origin: str) -> "FrameNode":
        return FrameNode(
            id=FrameId(fid),
            object=obj,
            origin=origin,
            axes=("R", "A", "S"),
            handedness="right",
            units=Unit("m"),
            epoch="session_start",
            validity_interval=_SESSION_VALIDITY,
            coordinate_type="physical",
        )

    nodes = (
        node(ANAT_FRAME, "population anatomical template", "anterior_commissure"),
        node(EEG_FRAME, "EEG electrode array", "nasion"),
    )
    residual = 1.5e-3
    edges = (
        FrameEdge(
            src=FrameId(ANAT_FRAME),
            dst=FrameId(EEG_FRAME),
            transform="rigid",
            parameters={
                "tx": NormalPrior(loc=0.0, scale=2e-3, units=Unit("m")),
                "ty": NormalPrior(loc=0.0, scale=2e-3, units=Unit("m")),
                "tz": NormalPrior(loc=0.0, scale=2e-3, units=Unit("m")),
            },
            lineage=(
                "template electrode montage co-registered to the anatomical template "
                "through the three standard fiducials; per-participant digitization "
                "replaces this edge when available"
            ),
            invertible=True,
            calibration=_calibration(
                "cal_anat_to_eeg_cap",
                ("physical_reference_frame", "sensor_device_geometry"),
                residual=residual,
                tolerance=6e-3,
            ),
            ledger=_bounded(
                "m", source="template_fiducial_landmarks", bias=(-residual, residual),
                measurement=residual**2,
            ),
            units_in=Unit("m"),
            units_out=Unit("m"),
            roundtrip_residual=residual,
            roundtrip_tolerance=residual * 4.0,
        ),
    )
    return FrameGraphSpec(root=FrameId(ANAT_FRAME), nodes=nodes, edges=edges)


def _clocks(dt_model: float, hemo_ratio: int) -> list["ClockSpec"]:
    """The two clocks of section 4.5, declared so the hyperperiod is exact.

    ``dt_model`` and ``dt_model * hemo_ratio`` are both representable in whole
    nanoseconds, so ``build_schedule`` computes the hyperperiod by integer LCM
    and the sync points land exactly on the slow ticks instead of drifting.
    """
    dt_slow = dt_model * hemo_ratio
    for name, dt in (("dt_model", dt_model), ("dt_model*hemo_ratio", dt_slow)):
        ns = dt * 1e9
        if abs(ns - round(ns)) > 1e-6:
            raise SchemaBuildError(
                f"{name}={dt} s is not a whole number of nanoseconds ({ns}); the "
                "multirate hyperperiod would be computed from a rounded period and "
                "the sync points would silently drift"
            )
    return [
        ClockSpec(
            id=ClockId(FAST_CLOCK),
            label="neural state clock (dt_model)",
            dt=dt_model,
            epoch="session_start",
            reference=None,
            sync_evidence="shared_hardware",
            adaptive_stepping=False,
            interpolation="band_limited",
            ledger=_estimable("s", estimator="fixed_step_integrator", numerical=1e-18),
        ),
        ClockSpec(
            id=ClockId(SLOW_CLOCK),
            label=f"haemodynamic clock (dt_model * hemo_ratio = {dt_slow:g} s)",
            dt=dt_slow,
            epoch="session_start",
            reference=ClockId(FAST_CLOCK),
            offset=0.0,
            drift=0.0,
            integration_window=0.0,
            sync_evidence="shared_hardware",
            adaptive_stepping=False,
            interpolation="zero_order_hold",
            dropped_sample_policy="marginalize",
            ledger=_estimable(
                "s", estimator="integer_substep_counter_audit", numerical=1e-18
            ),
        ),
    ]


def _poset() -> "ResolutionPoset":
    """One scale: the parcel.

    SC-WBD-001-beta declares no cross-scale prolongation, so R02 has nothing to
    object to -- which is the honest state of affairs, not an omission.
    """
    return ResolutionPoset(
        nodes=(
            ScaleNode(
                id=ScaleId(PARCEL_SCALE),
                label="anatomical parcel",
                axis="spatial",
                characteristic_scale=2.0e-2,
            ),
        )
    )


def _parcel_support(region_id: str) -> "Support":
    return Support(
        kind="parcel",
        frame=FrameId(ANAT_FRAME),
        units=Unit("m"),
        psf=PSF(
            kind="integration_kernel",
            fwhm=(0.02, 0.02, 0.006),
            units=Unit("dimensionless"),
            extent_units=Unit("m"),
            kernel_ref=f"parcel_membership::{region_id}",
        ),
        extent=(0.02, 0.02, 0.006),
        n_elements=1,
        resolution=ScaleId(PARCEL_SCALE),
        label=region_id,
    )


#: Units and clock class for each component of ``foundation.state.default_layout``.
#: ``meta`` maps onto the fast clock: the predictive log-variance channel is
#: emitted by the same step that produces the state it describes, so giving it
#: its own clock would invent a synchronization relation that does not exist.
_COMPONENT_UNITS: dict[str, str] = {
    "rate_e": "Hz",
    "rate_i": "Hz",
    "adaptation": "dimensionless",
    "spectral": "dimensionless",
    "hemo": "dimensionless",
    "uncertainty": "dimensionless",
    # per-family components (scwbd.foundation.families)
    "relay": "dimensionless",
    "trn": "dimensionless",
    "nuclei": "dimensionless",
    "striatum": "dimensionless",
    "gate": "dimensionless",
    "k": "dimensionless",
    "v": "dimensionless",
    "g": "dimensionless",
    "c": "dimensionless",
    "rho": "dimensionless",
    "relevance": "dimensionless",
    "autonomic": "dimensionless",
    "prediction": "dimensionless",
    "error": "dimensionless",
    "eligibility": "dimensionless",
    "mech": "dimensionless",
}
_COMPONENT_KINDS: dict[str, str] = {
    "rate_e": "population",
    "rate_i": "population",
    "adaptation": "memory",
    "spectral": "frequency",
    "hemo": "metabolic",
    "uncertainty": "uncertainty",
    # per-family components. body.tex §2.1's component vocabulary is
    # (sheet, layer, population, frequency, memory, metabolic, uncertainty);
    # the schema adds (field, channels, event, latent).
    "relay": "population",
    "trn": "population",
    "nuclei": "population",
    "striatum": "population",
    "gate": "channels",
    "k": "memory",  # H_t.k -- cue / index
    "v": "memory",  # H_t.v -- bound content
    "g": "field",  # H_t.g -- multiscale relational / grid-like code
    "c": "memory",  # H_t.c -- temporal & contextual state
    "rho": "uncertainty",  # H_t.rho -- retrieval confidence
    "relevance": "channels",
    "autonomic": "channels",
    "prediction": "latent",
    "error": "latent",
    "eligibility": "memory",
    "mech": "latent",
}


def _state_spec(layout: FoundationStateLayout, dt_model: float, dt_slow: float) -> "StateSpec":
    """Mirror one family's (or the control arm's) component list into a StateSpec.

    Fails closed on a component with no declared unit and kind.  ``latent`` as a
    silent default would let a new state component compile into the permission
    system with no declared physical type — which is how the ABI stops
    describing the tensors the model holds.
    """
    support_units = Unit("m")
    components: dict[str, Any] = {}
    for comp in layout:
        if comp.name not in _COMPONENT_KINDS:
            raise SchemaBuildError(
                f"state component {comp.name!r} has no declared schema kind or unit in "
                "compiler_bridge._COMPONENT_KINDS/_COMPONENT_UNITS. Declare it (body.tex §2.1's "
                "vocabulary is sheet/layer/population/frequency/memory/metabolic/uncertainty, "
                "plus field/channels/event/latent) rather than letting it compile as 'latent': a "
                "component with no declared physical type is an ABI the compiler cannot check. "
                "In particular the 'private' block of SCWBD.layout must never reach here — compile "
                "from the per-family layouts (SCWBD.family_layout), not from the interface view."
            )
        units = _COMPONENT_UNITS.get(comp.name, comp.units if comp.units != "log_var" else "dimensionless")
        kind = _COMPONENT_KINDS[comp.name]
        slow = comp.clock == "slow"
        temporal = TemporalSupport(
            clock=ClockId(SLOW_CLOCK if slow else FAST_CLOCK),
            dt=dt_slow if slow else dt_model,
        )
        components[comp.name] = ComponentSpec(
            kind=kind,  # type: ignore[arg-type]
            shape=(comp.dim,),
            units=Unit(units),
            support=Support(
                kind="parcel",
                frame=FrameId(ANAT_FRAME),
                units=support_units,
                psf=PSF(
                    kind="integration_kernel",
                    fwhm=(0.02, 0.02, 0.006),
                    units=Unit("dimensionless"),
                    extent_units=Unit("m"),
                    kernel_ref="parcel_membership",
                ),
                extent=(0.02, 0.02, 0.006),
                n_elements=1,
                resolution=ScaleId(PARCEL_SCALE),
            ),
            temporal=temporal,
            dtype="float32",
            boundary=bool(comp.exported),
            resolution=ScaleId(PARCEL_SCALE),
            ledger=_sensitivity(
                units,
                bias=(-0.10, 0.10),
                parameter=1e-2,
                numerical=1e-9,
            ),
            description=comp.description,
        )
    return StateSpec(components=components)


def _ports(
    region_id: str,
    *,
    dt_model: float,
    dt_slow: float,
    message_dim: int,
    n_eeg_channels: int,
    n_behaviour: int,
    source_ids: Mapping[str, tuple[str, ...]],
) -> list["Port"]:
    """The five typed ports of a foundation region."""
    parcel = _parcel_support(region_id)
    fast = TemporalSupport(clock=ClockId(FAST_CLOCK), dt=dt_model)
    slow = TemporalSupport(clock=ClockId(SLOW_CLOCK), dt=dt_slow)

    def message_port(name: str, direction: str) -> "Port":
        return Port(
            name=name,
            state_spec=StateSpec(
                components={
                    "message": ComponentSpec(
                        kind="channels",
                        shape=(message_dim,),
                        units=Unit("dimensionless"),
                        support=parcel,
                        temporal=fast,
                        boundary=True,
                        ledger=_sensitivity("dimensionless", bias=(-0.05, 0.05), parameter=1e-3),
                        description="projected boundary message of the exported state",
                    )
                }
            ),
            support=parcel,
            temporal=fast,
            direction=direction,  # type: ignore[arg-type]
            role="boundary",
            modality="latent_message",
            ledger=_sensitivity("dimensionless", bias=(-0.05, 0.05), parameter=1e-3),
        )

    eeg_support = Support(
        kind="sensor",
        frame=FrameId(EEG_FRAME),
        units=Unit("V"),
        psf=PSF(
            kind="lead_field",
            units=Unit("V"),
            extent_units=Unit("m"),
            kernel_ref=f"scwbd_leadfield_{n_eeg_channels}ch",
        ),
        n_elements=n_eeg_channels,
        resolution=ScaleId(PARCEL_SCALE),
        label=f"{n_eeg_channels}-channel montage",
    )
    behaviour_support = Support(
        kind="event",
        frame=FrameId(ANAT_FRAME),
        units=Unit("dimensionless"),
        psf=PSF(
            kind="empirical",
            units=Unit("dimensionless"),
            extent_units=Unit("m"),
            kernel_ref="scwbd_behaviour_pooling_weights",
        ),
        n_elements=n_behaviour,
        resolution=ScaleId(PARCEL_SCALE),
        label="behavioural readout pooling",
    )

    return [
        message_port("message_out", "out"),
        message_port("message_in", "in"),
        Port(
            name="eeg_out",
            state_spec=StateSpec(
                components={
                    "source_amplitude": ComponentSpec(
                        kind="channels",
                        shape=(1,),
                        units=Unit("A*m"),
                        support=eeg_support,
                        temporal=fast,
                        boundary=True,
                        ledger=_bounded(
                            "A*m", source="dipole_phantom", bias=(-1e-10, 1e-10),
                            measurement=1e-20,
                        ),
                    )
                }
            ),
            support=eeg_support,
            temporal=fast,
            direction="out",
            role="observation",
            modality="eeg",
            source_ids=source_ids.get("eeg", ()),
            ledger=_bounded("V", source="dipole_phantom", bias=(-2e-6, 2e-6), measurement=1e-12),
        ),
        Port(
            name="bold_out",
            state_spec=StateSpec(
                components={
                    "bold_signal": ComponentSpec(
                        kind="channels",
                        shape=(1,),
                        units=Unit("dimensionless"),
                        support=parcel,
                        temporal=slow,
                        boundary=True,
                        ledger=_sensitivity("dimensionless", bias=(-0.05, 0.05), measurement=4e-4),
                    )
                }
            ),
            support=Support(
                kind="parcel",
                frame=FrameId(ANAT_FRAME),
                units=Unit("dimensionless"),
                psf=PSF(
                    kind="hemodynamic",
                    fwhm=(6.0,),
                    units=Unit("dimensionless"),
                    extent_units=Unit("s"),
                    kernel_ref="balloon_windkessel_sfvq",
                ),
                extent=(0.02, 0.02, 0.006),
                n_elements=1,
                resolution=ScaleId(PARCEL_SCALE),
            ),
            temporal=slow,
            direction="out",
            role="observation",
            modality="fmri",
            source_ids=source_ids.get("bold", ()),
            ledger=_sensitivity("dimensionless", bias=(-0.05, 0.05), measurement=4e-4),
        ),
        Port(
            name="behaviour_out",
            state_spec=StateSpec(
                components={
                    "behaviour": ComponentSpec(
                        kind="event",
                        shape=(n_behaviour,),
                        units=Unit("dimensionless"),
                        support=behaviour_support,
                        temporal=slow,
                        boundary=True,
                        ledger=_sensitivity("dimensionless", bias=(-0.10, 0.10), measurement=1e-2),
                    )
                }
            ),
            support=behaviour_support,
            temporal=slow,
            direction="out",
            role="observation",
            modality="behaviour",
            source_ids=source_ids.get("behaviour", ()),
            ledger=_sensitivity("dimensionless", bias=(-0.10, 0.10), measurement=1e-2),
        ),
    ]


def _representative_regions(anat: AnatomyPrior, n: int) -> tuple[tuple[int, str, str], ...]:
    """``(index, id, system)`` exemplars spanning cortex/subcortex/cerebellum.

    The compiled schema declares a *representative subset*, not all 454 parcels:
    the state-layout ABI, the schedule and the parameter-group namespace are all
    determined by the per-region structure, which is identical across regions,
    so a subset that covers every anatomical division and every port carries the
    whole contract at a fraction of the schema size.  ``n_representative_regions``
    is therefore a fidelity knob, never a model-size knob.
    """
    if n < 2:
        raise SchemaBuildError(
            f"n_representative_regions={n} is too small: the long-range operator needs "
            "two distinct regions to be a typed edge between declared boundary spaces"
        )
    by_division: dict[str, list[int]] = {}
    for i, div in enumerate(anat.division):
        by_division.setdefault(div, []).append(i)
    if not by_division:
        raise SchemaBuildError("anatomy prior declares no regions")

    order = [d for d in ("cortex", "subcortex", "cerebellum") if d in by_division]
    order += [d for d in sorted(by_division) if d not in order]

    picked: list[int] = []
    # One from each division first, then fill round-robin with evenly spaced
    # members so the subset is not all neighbours on the same ellipsoid patch.
    round_index = 0
    while len(picked) < n:
        progressed = False
        for div in order:
            if len(picked) >= n:
                break
            members = by_division[div]
            if round_index >= len(members):
                continue
            stride = max(len(members) // max(n, 1), 1)
            idx = members[min(round_index * stride, len(members) - 1)]
            if idx in picked:
                idx = next((m for m in members if m not in picked), None)
                if idx is None:
                    continue
            picked.append(idx)
            progressed = True
        round_index += 1
        if not progressed:
            break
    if len(picked) < 2:
        raise SchemaBuildError(
            f"anatomy prior yielded only {len(picked)} usable regions; at least 2 are required"
        )
    out = []
    for i in picked:
        label = anat.labels[i] if i < len(anat.labels) else f"region_{i:04d}"
        div = anat.division[i]
        system = div if div in ("cortex", "subcortex", "cerebellum") else "cortex"
        out.append((i, str(label), system))
    return tuple(out)


def _operators(
    region_ids: Sequence[str],
    *,
    anat: AnatomyPrior,
    dt_model: float,
    rho_max: float,
) -> list["OperatorSpec"]:
    """``local_field``, ``long_range``, ``assimilation`` -- keyed exactly.

    ``OperatorSpec.key`` is ``op.id`` when set, and ``parameter_groups_of``
    builds ``operator:<key>:params`` from it, so these three ids *are* the
    parameter-group names the binding table keys on.

    Each operator is declared once even though the implementation applies it to
    every region: the foundation operator is weight-shared, so one parameter
    group governs all of it.  Declaring it per region would multiply the
    permission namespace without multiplying the parameters.
    """
    src, dst = region_ids[0], region_ids[1]

    present = anat.weights > 0
    if bool(present.any()):
        mean_length_mm = float(anat.tract_length[present].mean())
    else:  # pragma: no cover - a connectome with no edges is already refused upstream
        mean_length_mm = 50.0
    mean_delay_s = max(mean_length_mm * 1e-3 / 5.0, dt_model)

    return [
        OperatorSpec(
            id="local_field",
            src=src,
            dst=src,
            family="surrogate",
            evidence_class="soft",
            # The learned regional vector field is a surrogate. It is the
            # equal-capacity *control* for every mechanistic backend, never
            # itself evidence that a mechanism is neurally realized, so calling
            # it mechanistic would be exactly the unearned label sec. 0 forbids.
            mechanistic_status="surrogate",
            delay_prior=NormalPrior(loc=0.0, scale=1e-9, units=Unit("s")),
            params={
                "dt_scale": NormalPrior(loc=0.0, scale=1.0, units=Unit("dimensionless")),
                "timescale": LogNormalPrior(
                    mu=math.log(float(anat.timescale_prior.mean())),
                    sigma=0.5,
                    units=Unit("s"),
                    provenance="gradient-derived intrinsic timescale prior (agent C)",
                ),
            },
            ledger=_sensitivity(
                "Hz", bias=(-0.20, 0.20), parameter=2e-2, model_class=5e-2, numerical=1e-9,
            ),
            clock=ClockId(FAST_CLOCK),
            is_learned=True,
            identification=Identification(
                basis=("simulation_ground_truth",),
                notes=(
                    "trained against a simulated corpus with known parameters; the "
                    "learned core is a surrogate and carries no causal claim"
                ),
            ),
            residual=ResidualPolicy(
                rho_max=rho_max,
                measured_ratio=None,
                validity_set="scwbd_sim_corpus_regime_sweep",
                report_violations=True,
                norm="l2_energy",
            ),
            differentiable=True,
            solver="explicit_euler",
            units_in=Unit("Hz"),
            units_out=Unit("Hz"),
        ),
        OperatorSpec(
            id="long_range",
            src=src,
            dst=dst,
            family="delayed_ssm",
            evidence_class="soft",
            # Anatomically masked and delay-parameterized, but identified from
            # passive recordings plus a tractography prior: functional, not
            # effective. Calling it effective is what R04 refuses.
            mechanistic_status="functional",
            delay_prior=LogNormalPrior(
                mu=math.log(mean_delay_s),
                sigma=0.5,
                units=Unit("s"),
                provenance="tract length / conduction velocity prior (agent C connectome)",
            ),
            params={
                "gain_soft": NormalPrior(loc=0.0, scale=0.5, units=Unit("dimensionless")),
                "gain_proposed": NormalPrior(loc=-1.5, scale=0.5, units=Unit("dimensionless")),
                "global_scale": NormalPrior(loc=0.0, scale=0.5, units=Unit("dimensionless")),
            },
            ledger=_sensitivity(
                "Hz", bias=(-0.15, 0.15), parameter=1e-2, model_class=3e-2,
            ),
            clock=ClockId(FAST_CLOCK),
            is_learned=True,
            identification=Identification(
                basis=("anatomical_prior", "passive_correlation"),
                notes=(
                    "topology from the structural connectome, gains from passive "
                    "recordings; common input is not excluded"
                ),
            ),
            distance_m=mean_length_mm * 1e-3,
            differentiable=True,
            solver="delayed_gemm",
            units_in=Unit("Hz"),
            units_out=Unit("Hz"),
        ),
        OperatorSpec(
            id="assimilation",
            src=dst,
            dst=dst,
            family="surrogate",
            evidence_class="proposed",
            mechanistic_status="surrogate",
            delay_prior=NormalPrior(loc=0.0, scale=1e-9, units=Unit("s")),
            params={
                "encoder_scale": NormalPrior(loc=0.0, scale=1.0, units=Unit("dimensionless")),
            },
            ledger=_sensitivity(
                "dimensionless", bias=(-0.25, 0.25), model_class=6e-2, parameter=2e-2,
            ),
            clock=ClockId(FAST_CLOCK),
            is_learned=True,
            identification=Identification(
                basis=("simulation_ground_truth",),
                notes="amortized warm start for the filter; an encoder, not a mechanism",
            ),
            differentiable=True,
            units_in=Unit("dimensionless"),
            units_out=Unit("dimensionless"),
        ),
    ]


# ----------------------------------------------------------------------
# source cards
# ----------------------------------------------------------------------
_HEAD_SIGNAL: dict[str, dict[str, Any]] = {
    "eeg": {
        "name": "scalp_potential",
        "units": "uV",
        "reference": "average",
        "dynamic_range": (-2.0e2, 2.0e2),
        "quantization": 1e-2,
    },
    "bold": {
        "name": "bold_percent_signal_change",
        "units": "percent",
        "reference": "parcel mean over run",
        "dynamic_range": (-5.0, 5.0),
    },
    "behaviour": {
        "name": "behavioural_response",
        "units": "dimensionless",
        "reference": "trial",
        "dynamic_range": (-10.0, 10.0),
    },
    "parcel_activity": {
        "name": "parcel_activity",
        "units": "Hz",
        "reference": "parcel mean over run",
        "dynamic_range": (-1.0e2, 1.0e2),
    },
}

_HEAD_PORT: dict[str, str] = {
    "eeg": "eeg_out",
    "bold": "bold_out",
    "behaviour": "behaviour_out",
    "parcel_activity": "message_out",
}

_HEAD_PHYSICS: dict[str, str] = {
    "eeg": "y_E[k] = L(l) * J(x(k*dt)) + eps_E[k]  (lead field applied to the exported state)",
    "bold": "y_B[n] = g * q(n*dt_slow)/v(n*dt_slow) + eps_B[n]  (Balloon-Windkessel readout)",
    "behaviour": "y_beh[n] = f_beh(pool(x(n*dt_slow))) + eps_beh[n]",
    "parcel_activity": "y[k] = f_read(x_exported(k*dt)) + eps[k]",
}


def _governance(spec: SourceSpec) -> "Governance":
    simulated = bool(spec.is_simulated)
    return Governance(
        license="CC0-1.0" if simulated else "unknown-per-dataset",
        purpose_limits=("methods_development",),
        consent_scope=(
            "not applicable: fully simulated data, no human participants"
            if simulated
            else "as recorded in the upstream dataset's data use agreement"
        ),
        redistribution="full" if simulated else "derived_only",
        withdrawal_status="none_pending",
        privacy_class="public" if simulated else "deidentified",
        retention="indefinite" if simulated else "as specified upstream",
        may_release_weights=True,
        may_release_examples=simulated,
        contains_biometric=False,
    )


def _lineage(spec: SourceSpec) -> tuple[tuple["LineageUnit", ...], dict[str, str]]:
    """Participants and sessions for one source, with a lineage-safe fold split.

    Unit ids are prefixed with the source id so two sources can never collide on
    a unit and then disagree about its fold -- which R10 reports, correctly, as
    a leak.
    """
    n_participants = int(spec.n_participants or max(round(spec.n_eff), 1))
    n_participants = max(n_participants, 3)
    folds = ("train", "val", "test")
    units: list[Any] = []
    assignments: dict[str, str] = {}

    if spec.is_simulated:
        # A simulator replica is not an independent subject: it declares the
        # generation cohort it was drawn from, and the split must keep a whole
        # cohort on one side of the holdout.  The corpus is therefore generated
        # as three *disjoint* cohorts, one per fold, rather than one cohort
        # sliced afterwards -- slicing a cohort is what R10 refuses, because
        # replicas sharing a generator share their evidence.
        for fold in folds:
            cid = f"{spec.id}::cohort_{fold}"
            units.append(LineageUnit(id=cid, kind="site", n_observations=0))
            assignments[cid] = fold

    for i in range(n_participants):
        pid = f"{spec.id}::p{i:04d}"
        fold = folds[0] if i % 5 < 3 else folds[1 + (i % 5) - 3]
        units.append(
            LineageUnit(
                id=pid,
                kind="simulator_replica" if spec.is_simulated else "participant",
                parent_ids=(f"{spec.id}::cohort_{fold}",) if spec.is_simulated else (),
                n_observations=0,
            )
        )
        assignments[pid] = fold
        sid = f"{pid}::s0"
        units.append(LineageUnit(id=sid, kind="session", parent_ids=(pid,), n_observations=1))
        assignments[sid] = fold
    return tuple(units), assignments


def _effects(spec: SourceSpec, *, individualization: bool, recovery_report: str):
    """Stage-V ``theta_{p,s} = mu + alpha_{g(p)} + delta_p + zeta_{p,s}``.

    Every term is centered (sum-to-zero) *and* shrunk, which is what R07 asks
    for: an unconstrained additive decomposition of the same quantity into a
    group effect, a subject effect and a session effect is not identified, and
    the compiler refuses it rather than letting the optimizer pick an arbitrary
    split of the shared mean.
    """
    effects = [
        HierarchicalEffect(
            name="alpha_group",
            level="site",
            parameterization="sum_to_zero",
            shrinkage_prior=NormalPrior(loc=0.0, scale=0.25, units=Unit("dimensionless")),
            recovery_tested=True,
            recovery_report=recovery_report,
        )
    ]
    if individualization:
        effects.append(
            HierarchicalEffect(
                name="delta_participant",
                level="participant",
                parameterization="noncentered_hierarchical",
                shrinkage_prior=NormalPrior(loc=0.0, scale=0.20, units=Unit("dimensionless")),
                recovery_tested=True,
                recovery_report=recovery_report,
            )
        )
        effects.append(
            HierarchicalEffect(
                name="zeta_session",
                level="session",
                parameterization="noncentered_hierarchical",
                shrinkage_prior=NormalPrior(loc=0.0, scale=0.10, units=Unit("dimensionless")),
                recovery_tested=True,
                recovery_report=recovery_report,
            )
        )
    return tuple(effects)


def _source_permission(spec: SourceSpec, own_group: str) -> "GradientPermission":
    """``A_k`` over the *compiler's* group names.

    ``SourceSpec.gradient_permission`` is interpreted here as globs over the
    compiler's parameter-group namespace, not over torch tensor names.  That
    inversion is the whole point of this bridge: the schema is where a source's
    reach is declared, and :func:`bind_masks` derives the torch globs from what
    the compiler allowed.  A source always keeps its own observation nuisance
    group -- estimating your own gain and noise is not a claim about the brain.
    """
    if spec.role in ("evaluation_only", "negative_control"):
        return GradientPermission(
            evaluation_only=True,
            max_weight=None,
            notes=(
                f"role={spec.role}: contributes an audit, never a gradient "
                "(Appendix B: the roles are deliberately non-equivalent)"
            ),
        )
    frozen = tuple(spec.frozen) + ("frame_edge:*", "scale_map:*")
    granted = tuple(p for p in spec.gradient_permission if p not in frozen)
    modules = tuple(dict.fromkeys(granted))
    groups = (own_group,) if own_group not in modules and own_group not in frozen else ()
    return GradientPermission(
        modules=modules,
        parameter_groups=groups,
        frozen=tuple(dict.fromkeys(frozen)),
        evaluation_only=False,
        max_weight=1.0,
        notes=spec.notes or f"declared reach of source {spec.id!r}",
    )


def _source_card(
    spec: SourceSpec,
    *,
    region_ids: Sequence[str],
    dt_model: float,
    dt_slow: float,
    n_eeg_channels: int,
    individualization: bool,
    recovery_report: str,
) -> "SourceCard":
    head = observation_head_for(spec)
    sig = _HEAD_SIGNAL[head]
    port = _HEAD_PORT[head]
    slow = head in ("bold", "behaviour")
    temporal = TemporalSupport(
        clock=ClockId(SLOW_CLOCK if slow else FAST_CLOCK), dt=dt_slow if slow else dt_model
    )

    if head == "eeg":
        spatial = Support(
            kind="sensor",
            frame=FrameId(EEG_FRAME),
            units=Unit("V"),
            psf=PSF(
                kind="lead_field",
                units=Unit("V"),
                extent_units=Unit("m"),
                kernel_ref=f"scwbd_leadfield_{n_eeg_channels}ch",
            ),
            n_elements=n_eeg_channels,
            resolution=ScaleId(PARCEL_SCALE),
        )
        calibration = _calibration(
            f"cal_{spec.id}",
            ("amplitude_unit", "clock_graph", "sensor_device_geometry"),
            residual=1.0e-6,
            tolerance=5.0e-6,
            residual_units="V",
        )
    else:
        spatial = Support(
            kind="parcel",
            frame=FrameId(ANAT_FRAME),
            units=Unit("m"),
            psf=PSF(
                kind="hemodynamic" if head == "bold" else "integration_kernel",
                fwhm=(6.0,) if head == "bold" else (0.02, 0.02, 0.006),
                units=Unit("dimensionless"),
                extent_units=Unit("s") if head == "bold" else Unit("m"),
                kernel_ref="balloon_windkessel_sfvq" if head == "bold" else "parcel_membership",
            ),
            extent=(0.02, 0.02, 0.006),
            n_elements=len(region_ids),
            resolution=ScaleId(PARCEL_SCALE),
        )
        calibration = _calibration(
            f"cal_{spec.id}",
            ("physical_reference_frame", "clock_graph", "deformation_projection"),
            residual=6.0e-4,
            tolerance=2.0e-3,
        )

    units = str(sig["units"])
    halfwidth = float(spec.bias_halfwidth) if spec.bias_halfwidth is not None else 0.10
    halfwidth = max(abs(halfwidth), 1e-6)
    variance = float(spec.measurement_variance) if spec.measurement_variance is not None else 1e-2

    if spec.is_simulated:
        # A simulated corpus has no replication that identifies its bias against
        # a measured brain, and no phantom that bounds it: the only defensible
        # status is a swept range, and the simulator/reality gap is recorded
        # separately as model discrepancy rather than folded into the variance.
        ledger = _sensitivity(
            units,
            bias=(-halfwidth, halfwidth),
            discrepancy=(
                float(spec.model_discrepancy) if spec.model_discrepancy is not None else 0.05
            ),
            measurement=variance,
            model_class=0.05,
        )
    else:
        ledger = _estimable(
            units,
            estimator="within/between-session variance components over repeated runs",
            bias=(-halfwidth, halfwidth),
            measurement=variance,
            between_session=variance * 0.25,
        )

    units_arg, assignments = _lineage(spec)
    own_group = f"observation:{spec.id}:nuisance"

    observation = ObservationModel(
        # The head name IS the binding key: recorded here so that bind_masks
        # reads a declaration instead of re-guessing the modality.
        name=head,
        forward_physics=_HEAD_PHYSICS[head],
        observed_variables=(str(sig["name"]),),
        preprocessing=("as declared by the upstream source card",),
        likelihood_kind="generative",
        noise_model=NormalPrior(
            loc=0.0, scale=math.sqrt(max(variance, 1e-12)), units=Unit(units)
        ),
        calibration_status=(
            "uncalibrated" if spec.is_simulated else "calibrated_empirically"
        ),
        # NOT hard-coded.  This field asserts to every downstream gate that a
        # participant-level leakage audit ran and passed; it must therefore be
        # backed by one.  ``False`` is the schema default and means "not
        # established", which is the honest reading when nothing has run.
        #
        # It was previously ``True`` unconditionally.  That was not carelessness
        # but **staleness**: it was written when the only observation sources
        # were ones whose splits had been audited by hand, and nothing forced
        # re-examination when the trainer later began building splits that were
        # never audited at all.  A default that was true when written, in a
        # pipeline that stopped guaranteeing it.
        leakage_checked=bool(getattr(spec, "leakage_audited", False)),
        target_ports=tuple(f"{rid}.{port}" for rid in region_ids),
        ledger=_sensitivity(units, bias=(-halfwidth, halfwidth), measurement=variance),
    )

    return SourceCard(
        identity=Identity(
            persistent_id=spec.id,
            version="1.0.0",
            release="scwbd-001-beta",
            file_hashes={f"{spec.id}.manifest": "0" * 64},
            parent_ids=(),
            software_hash="0" * 40,
            container_hash="0" * 40,
            preprocessing_hash="0" * 40,
            mutable_download=False,
            label=spec.notes or spec.id,
        ),
        governance=_governance(spec),
        population=PopulationStructure(
            levels=("site", "participant", "session"),
            units=units_arg,
            n_participants=sum(1 for u in units_arg if u.kind in ("participant", "simulator_replica")),
            demographics={"simulated": bool(spec.is_simulated)},
            selection_mechanism=(
                "complete enumeration of the simulated cohort"
                if spec.is_simulated
                else "as recorded in the upstream dataset"
            ),
            effects=_effects(
                spec, individualization=individualization, recovery_report=recovery_report
            ),
            min_subgroup_n=1,
        ),
        spatial=spatial,
        temporal=temporal,
        calibration=calibration,
        observation=observation,
        intervention=None,
        missingness=Missingness(
            mechanism="mar",
            handling="marginalize",
            unplanned_missing_fraction=0.02,
            artifact_rejection="declared upstream; windows are masked, never zeroed",
            dropout=0.0,
            attrition=0.0,
        ),
        ledger=ledger,
        gradient_permission=_source_permission(spec, own_group),
        role=spec.role,  # type: ignore[arg-type]
        split_policy=SplitPolicy(
            grouping_keys=("participant", "session"),
            temporal_holdout=TemporalHoldout(key="session_start", boundary=1800.0),
            fold_assignments=assignments,  # type: ignore[arg-type]
            leakage_barrier="parent_lineage",
            role_locked=True,
            stimuli_shared_across_folds=False,
        ),
        signal=SignalSpec(
            name=str(sig["name"]),
            units=Unit(units),
            reference=str(sig["reference"]),
            dynamic_range=tuple(sig["dynamic_range"]),  # type: ignore[arg-type]
            quantization=sig.get("quantization"),
            n_channels=n_eeg_channels if head == "eeg" else len(region_ids),
        ),
        label=spec.notes or spec.id,
        effective_n=max(float(spec.n_eff), 1e-6),
    )


def build_foundation_schema(
    anat: AnatomyPrior,
    source_specs: Iterable[SourceSpec],
    *,
    n_representative_regions: int = 8,
    layout: FoundationStateLayout | None = None,
    family_layout: Any = None,
    dt_model: float = 0.008,
    hemo_ratio: int = 25,
    individualization: bool = True,
    message_dim: int = 12,
    n_eeg_channels: int = 64,
    n_behaviour: int = 4,
    recovery_report: str = DEFAULT_RECOVERY_REPORT,
    schema_id: str = "scwbd_001_beta_foundation",
) -> "BrainSchema":
    """Declare SC-WBD-001-beta as a ``BrainSchema`` the compiler will accept.

    The returned schema compiles with **no overrides**.  Every declaration in it
    is a statement the foundation model must live up to: the state components
    mirror :func:`scwbd.foundation.state.default_layout` exactly (so the ABI the
    compiler emits describes the tensors the model actually holds), the two
    clocks are the model's two clocks, and the three operators are keyed so that
    their parameter groups are the keys of :data:`FOUNDATION_BINDING`.

    Raises :class:`SchemaBuildError` on anything it cannot declare honestly; it
    never degrades to a partial schema, because a partial schema compiles into a
    permission system with silent holes.
    """
    _require_compiler()
    specs = list(source_specs)
    if not specs:
        raise SchemaBuildError(
            "no source specs supplied; a schema with no source cards compiles to a "
            "model no evidence may update, which is never what the caller meant"
        )
    seen_ids = [s.id for s in specs]
    dupes = sorted({i for i in seen_ids if seen_ids.count(i) > 1})
    if dupes:
        raise SchemaBuildError(f"duplicate source spec ids: {dupes}")
    if hemo_ratio < 1:
        raise SchemaBuildError(f"hemo_ratio must be >= 1, got {hemo_ratio}")

    if family_layout is not None and layout is not None:
        raise SchemaBuildError(
            "pass either `layout` (the control arm's single state space) or `family_layout` "
            "(the region-indexed one), not both: which of the two describes the weights is "
            "exactly the thing R12 exists to keep unambiguous."
        )
    lay = layout if layout is not None else default_layout()
    dt_slow = dt_model * hemo_ratio

    try:
        exemplars = _representative_regions(anat, n_representative_regions)
        region_ids = [rid for _, rid, _ in exemplars]

        head_sources: dict[str, list[str]] = {}
        for spec in specs:
            head_sources.setdefault(observation_head_for(spec), []).append(spec.id)
        source_ids = {k: tuple(v) for k, v in head_sources.items()}

        # body.tex §2.1 indexes the state SPACE by region, so with families on
        # the schema carries one StateSpec per family, not one for the brain.
        # A single shared StateSpec over a heterogeneous model would compile an
        # ABI that describes tensors the model does not hold.
        state = _state_spec(lay, dt_model, dt_slow)
        family_states: dict[str, Any] = {}
        family_of: dict[int, str] = {}
        if family_layout is not None:
            for f in family_layout:
                family_states[f.name] = _state_spec(f.layout, dt_model, dt_slow)
                for r in f.regions:
                    family_of[int(r)] = f.name
        regions = []
        for idx, rid, system in exemplars:
            fam = family_of.get(int(idx))
            regions.append(
                Region(
                    id=rid,
                    label=f"{system} parcel {rid}" + (f" [family {fam}]" if fam else ""),
                    state=family_states[fam] if fam else state,
                    ports=_ports(
                        rid,
                        dt_model=dt_model,
                        dt_slow=dt_slow,
                        message_dim=message_dim,
                        n_eeg_channels=n_eeg_channels,
                        n_behaviour=n_behaviour,
                        source_ids=source_ids,
                    ),
                    atlas_refs=[
                        AtlasRef(
                            atlas=anat.provenance,
                            version="1.0.0",
                            index=idx,
                            label=rid,
                            frame=FrameId(ANAT_FRAME),
                            coverage=1.0,
                        )
                    ],
                    authority="coarse_sparse" if not anat.is_biological() else "consensus",
                    resolution=ScaleId(PARCEL_SCALE),
                    system=system,  # type: ignore[arg-type]
                    ledger=_sensitivity("Hz", bias=(-0.20, 0.20), parameter=2e-2),
                )
            )

        sources = [
            _source_card(
                spec,
                region_ids=region_ids,
                dt_model=dt_model,
                dt_slow=dt_slow,
                n_eeg_channels=n_eeg_channels,
                individualization=individualization,
                recovery_report=recovery_report,
            )
            for spec in specs
        ]

        return BrainSchema(
            id=schema_id,
            label="SC-WBD-001-beta foundation model (representative-region declaration)",
            identity=Identity(
                persistent_id=f"schema:{schema_id}",
                version="1.0.0",
                release="beta",
                file_hashes={"compiler_bridge.py": "0" * 64},
                software_hash="0" * 40,
                container_hash="0" * 40,
            ),
            regions=regions,
            operators=_operators(
                region_ids, anat=anat, dt_model=dt_model, rho_max=0.35,
            ),
            resolution_poset=_poset(),
            clocks=_clocks(dt_model, hemo_ratio),
            frames=_frames(),
            sources=sources,
            metadata={
                "anatomy_provenance": anat.provenance,
                "anatomy_is_biological": anat.is_biological(),
                "n_model_regions": int(anat.n_regions),
                "n_declared_regions": len(regions),
                "dt_model_s": dt_model,
                "hemo_ratio": hemo_ratio,
                "state_layout_dim": lay.dim,
                "individualization": individualization,
                "note": (
                    "regions are a representative subset spanning every anatomical "
                    "division; the per-region structure, ports and clocks are identical "
                    "across all n_model_regions parcels"
                ),
            },
        )
    except SchemaBuildError:
        raise
    except Exception as exc:  # noqa: BLE001 - re-raised with the reason attached
        raise SchemaBuildError(
            f"could not build a valid BrainSchema for SC-WBD-001-beta: {exc!r}"
        ) from exc


def build_foundation_claim(
    *, claim_id: str = "scwbd_001_beta_foundation_claim"
) -> "ClaimManifest":
    """The claim SC-WBD-001-beta is allowed to make: functional, no overrides.

    The learned core is a surrogate and the long-range operator is identified
    from a tractography prior plus passive recordings, so nothing here earns an
    effective or mechanistic label.  ``posterior_class="generalized"`` because
    the training objective is a weighted mixture of heterogeneous factors, which
    is not a calibrated Bayesian posterior no matter how well it fits.
    """
    _require_compiler()
    return ClaimManifest(
        id=claim_id,
        claim_class="functional",
        posterior_class="generalized",
        requires_global_section=False,
        optimizes_intervention=False,
        target_gates=("G1", "G2"),
        overrides=(),
        disabling_evidence=(
            "If the learned operator matches every mechanistic backend's held-out "
            "likelihood and its perturbational forecasts, the operator claim is "
            "unearned; if held-out interval coverage falls below nominal on unseen "
            "sites or sessions, the population claim narrows to a provenance claim."
        ),
        notes={
            "reference": "ARCHITECTURE.md sec. 5, paper/body.tex sec. 4.1",
            "consumer": "scwbd.foundation trainer (agent I)",
        },
    )


def compile_foundation(
    anat: AnatomyPrior,
    source_specs: Iterable[SourceSpec],
    *,
    claim: "ClaimManifest | None" = None,
    **kw: Any,
) -> "CompiledModel":
    """Build the schema and compile it.  Refusals propagate unchanged.

    A :class:`~scwbd.schema.refusals.CompilerRefusal` is deliberately not caught:
    the compiler earns its credibility by rejecting programs, and a bridge that
    swallowed the rejection and carried on would destroy exactly the property
    this module exists to import.
    """
    _require_compiler()
    schema = build_foundation_schema(anat, source_specs, **kw)
    return _compile_schema(schema, claim=claim if claim is not None else build_foundation_claim())


# ======================================================================
# binding the compiler's answers back onto torch parameters
# ======================================================================
def _named_parameters(model: Any) -> list[str]:
    # Logical names throughout: the binding table names things in the
    # architecture, and torch.compile's ``_orig_mod`` segments are not part of
    # it.  Matching raw names here is what made every per-region binding vacuous
    # on 2026-08-05 -- see :func:`~scwbd.foundation.util.logical_param_name`.
    return [logical_param_name(n) for n, p in model.named_parameters() if p.requires_grad]


def _named_buffers(model: Any) -> list[str]:
    return [logical_param_name(n) for n, _ in model.named_buffers()]


def _matching(patterns: Iterable[str], names: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    for name in names:
        if any(fnmatchcase(name, pat) or name == pat for pat in patterns):
            out.append(name)
    return tuple(out)


class _Binding:
    """Resolved binding of one compiled model onto one torch module.

    Every compiled parameter group lands in exactly one of five states, kept
    apart because they are five different claims about the implementation:

    ``groups``          bound to >=1 trainable tensor -- a live permission;
    ``frozen``          implemented by named, existing *buffers* -- real, but no
                        gradient reaches it (delay-bin geometry; the anat->cap
                        co-registration baked into the lead field);
    ``declared_empty``  deliberately bound to no tensor whatsoever, because the
                        thing has no implementation state at all (an
                        instantaneous operator's delay group);
    ``unbound``         no entry in either table -- the schema declares
                        something nobody wrote a binding for;
    ``empty_patterns``  a pattern was written and matches *nothing* on the
                        model, neither parameter nor buffer.

    The last two are the ones that matter.  Both are reported as problems,
    because both mean a declaration has quietly stopped describing the model --
    the rename-and-nobody-noticed failure.  ``frozen`` and ``declared_empty``
    are *positive* statements someone had to write down, and are not problems.
    """

    __slots__ = (
        "groups",
        "frozen",
        "unbound",
        "declared_empty",
        "empty_patterns",
        "sources",
        "problems",
    )

    def __init__(
        self,
        model: Any,
        compiled: "CompiledModel",
        names: Sequence[str],
        buffers: Sequence[str] = (),
    ) -> None:
        self.groups: dict[str, tuple[str, ...]] = {}
        self.frozen: dict[str, tuple[str, ...]] = {}
        self.unbound: list[str] = []
        self.declared_empty: list[str] = []
        self.empty_patterns: list[tuple[str, str, str]] = []
        problems: list[str] = []

        for group in compiled.gradient_masks.group_names:
            pats = patterns_for_group(group, schema=compiled.schema)
            frozen_pats = frozen_patterns_for_group(group)
            if pats is None and not frozen_pats:
                self.unbound.append(group)
                continue
            pats = tuple(pats or ())
            self.groups[group] = pats
            for pat in pats:
                if not _matching((pat,), names):
                    self.empty_patterns.append((group, pat, "parameter"))
            if frozen_pats:
                self.frozen[group] = tuple(frozen_pats)
                for pat in frozen_pats:
                    if not _matching((pat,), buffers):
                        self.empty_patterns.append((group, pat, "buffer"))
            elif not pats:
                self.declared_empty.append(group)

        self.sources: dict[str, tuple[str, ...]] = {}
        for sid in compiled.gradient_masks.keys():
            patterns: list[str] = []
            for group in compiled.gradient_masks[sid].allowed_groups():
                pats = self.groups.get(group)
                if pats is None:
                    problems.append(
                        f"source {sid!r} is allowed group {group!r}, which has no binding to "
                        "any torch parameter name; the permission cannot be executed"
                    )
                    continue
                # Frozen groups contribute no trainable glob: the grant is real
                # but terminates in a buffer, and handing the source a pattern
                # that can never match would misreport its reach.
                for pat in pats:
                    if pat not in patterns:
                        patterns.append(pat)
            self.sources[sid] = tuple(patterns)

        for group, pat, kind in self.empty_patterns:
            problems.append(
                f"binding {group!r} -> {pat!r} matches no {kind} of "
                f"{type(model).__name__}; the declaration and the implementation have "
                "drifted, so this permission is decorative"
            )
        for group in self.unbound:
            problems.append(
                f"parameter group {group!r} has no entry in FOUNDATION_BINDING or "
                "FOUNDATION_FROZEN_BINDING; no torch tensor is governed by it"
            )
        self.problems: tuple[str, ...] = tuple(problems)

    def message(self) -> str:
        return "compiler->torch binding is incomplete:\n  " + "\n  ".join(self.problems)


def bind_masks(
    model: Any, compiled: "CompiledModel", *, strict: bool = False
) -> dict[str, tuple[str, ...]]:
    """source id -> glob patterns over ``model.named_parameters()``.

    For every source the compiler compiled a mask over parameter *groups*; this
    turns each source's allowed groups into the torch globs those groups imply.
    The result is what makes the compiler the authority: a source can only touch
    tensors that some group it was granted actually binds to.

    Anything lost in translation is reported, never dropped silently -- see
    :class:`_Binding` for the three cases.  With ``strict=True`` a report raises
    instead of logging; :func:`audit_binding` returns the same information
    machine-readably without logging it a second time.
    """
    _require_compiler()
    binding = _Binding(model, compiled, _named_parameters(model), _named_buffers(model))
    if binding.problems:
        if strict:
            raise CompilerBridgeError(binding.message())
        _log.warning("%s", binding.message())
    return dict(binding.sources)


def audit_binding(model: Any, compiled: "CompiledModel") -> dict[str, Any]:
    """Machine-readable coverage report over every torch parameter.

    An **unclaimed** parameter -- one no compiled parameter group binds to -- is
    a hole in the permission system: nothing in the schema describes it, so no
    source card can grant or withhold it and no reviewer can see that it trains.
    It is reported first-class, with its size, so the hole is impossible to miss
    in a diff.
    """
    _require_compiler()
    named = [
        (logical_param_name(n), p) for n, p in model.named_parameters() if p.requires_grad
    ]
    names = [n for n, _ in named]
    numel = {n: int(p.numel()) for n, p in named}
    buf = {logical_param_name(n): int(b.numel()) for n, b in model.named_buffers()}
    buf_names = list(buf)

    binding = _Binding(model, compiled, names, buf_names)

    claims: dict[str, list[str]] = {n: [] for n in names}
    group_report: dict[str, dict[str, Any]] = {}
    for group, pats in binding.groups.items():
        matched = _matching(pats, names)
        for n in matched:
            claims[n].append(group)
        frozen_pats = binding.frozen.get(group, ())
        frozen_matched = _matching(frozen_pats, buf_names) if frozen_pats else ()
        group_report[group] = {
            "patterns": list(pats),
            "n_parameters": len(matched),
            "n_elements": sum(numel[n] for n in matched),
            "parameters": list(matched),
            "frozen_patterns": list(frozen_pats),
            "frozen_buffers": list(frozen_matched),
            "n_frozen_elements": sum(buf[n] for n in frozen_matched),
            "sources_allowed": list(compiled.gradient_masks.sources_updating(group)),
        }

    unclaimed = [n for n in names if not claims[n]]
    source_report: dict[str, dict[str, Any]] = {}
    for sid, patterns in binding.sources.items():
        mask = compiled.gradient_masks[sid]
        matched = _matching(patterns, names)
        source_report[sid] = {
            "role": mask.role,
            "evaluation_only": mask.evaluation_only,
            "n_allowed_groups": mask.n_allowed(),
            "allowed_groups": list(mask.allowed_groups()),
            "unmatched_permission_patterns": list(mask.unmatched_patterns),
            "torch_patterns": list(patterns),
            "n_parameters": len(matched),
            "n_elements": sum(numel[n] for n in matched),
        }

    return {
        "model": type(model).__name__,
        "schema_id": compiled.schema.id,
        "compiler_version": compiled.provenance.compiler_version,
        "abi_digest": compiled.state_layout.abi_digest(),
        "n_parameters": len(names),
        "n_elements": sum(numel.values()),
        "parameters": {
            n: {"n_elements": numel[n], "claimed_by": sorted(claims[n])} for n in names
        },
        "unclaimed_parameters": [{"name": n, "n_elements": numel[n]} for n in unclaimed],
        "n_unclaimed_elements": sum(numel[n] for n in unclaimed),
        "groups": group_report,
        "unbound_groups": sorted(binding.unbound),
        "declared_empty_groups": sorted(binding.declared_empty),
        "frozen_groups": {g: list(p) for g, p in sorted(binding.frozen.items())},
        "empty_bindings": [
            {"group": g, "pattern": p, "namespace": k} for g, p, k in binding.empty_patterns
        ],
        "unreachable_groups": list(compiled.gradient_masks.unreachable_groups()),
        "problems": list(binding.problems),
        "sources": source_report,
        "note": (
            "claimed_by lists every compiler parameter group whose binding matches the "
            "tensor; an empty list means no declaration governs it"
        ),
    }


def gate_from_compiled(
    model: Any,
    compiled: "CompiledModel",
    source_specs: Iterable[SourceSpec] | Mapping[str, SourceSpec],
) -> GradientGate:
    """A :class:`GradientGate` whose permissions come from the compiler.

    Each spec's ``gradient_permission`` is **replaced** by the globs the
    compiler's allowed groups imply, and ``frozen`` is cleared: freezing is
    already expressed in the mask (a frozen group is simply not allowed), and
    leaving a second, independently-edited freeze list in place would let the
    two disagree without anyone noticing.

    A spec the compiler never saw is an error, not a default-deny: it means the
    training mixture and the schema have drifted apart, and silently gating it
    to nothing would look exactly like a correctly-refused source.
    """
    _require_compiler()
    specs = (
        dict(source_specs)
        if isinstance(source_specs, Mapping)
        else {s.id: s for s in source_specs}
    )
    bound = bind_masks(model, compiled)
    missing = sorted(set(specs) - set(bound))
    if missing:
        raise CompilerBridgeError(
            f"source specs {missing} have no compiled gradient mask; the training "
            f"mixture and the schema disagree. Compiled sources: {sorted(bound)}"
        )

    gated: dict[str, SourceSpec] = {}
    for sid, spec in specs.items():
        gated[sid] = replace(
            spec,
            gradient_permission=bound[sid],
            frozen=(),
            notes=(
                f"{spec.notes} | permissions compiled by {compiled.provenance.compiler_version} "
                f"from schema {compiled.schema.id}"
            ).strip(" |"),
        )
    return GradientGate(model, gated)


# ======================================================================
# the multirate plan
# ======================================================================
def schedule_plan(
    compiled: "CompiledModel", *, horizon_s: float | None = None
) -> dict[str, Any]:
    """The compiled multirate plan, as a plain dict the trainer can consume.

    The foundation trainer's curriculum needs three things the compiler already
    computed exactly: how many fast steps make one slow step (``substeps``), at
    which times the two clocks coincide (``sync_points`` -- these are the only
    instants at which a fast and a slow field may be compared without an
    interpolation error entering the ledger), and what interpolation contract
    applies when they do not.  ``horizon_s`` defaults to one hyperperiod, which
    is the shortest window the whole plan repeats over.
    """
    _require_compiler()
    sched = compiled.schedule
    horizon = float(horizon_s) if horizon_s is not None else float(sched.hyperperiod)

    clocks: dict[str, Any] = {}
    for cid in sched.clocks():
        fields = sched.fields_on_clock(cid)
        periods = {sched.policy(*f.split(".", 1)).period for f in fields}
        period = min(periods) if periods else 0.0
        spec = compiled.schema.clock(cid) if compiled.schema.has_clock(cid) else None
        clocks[cid] = {
            "period_s": period,
            "period_ns": int(round(period * 1e9)),
            "rate_hz": (1.0 / period) if period > 0 else None,
            "n_fields": len(fields),
            "fields": list(fields),
            "interpolation": spec.interpolation if spec is not None else "unknown",
            "sync_evidence": spec.sync_evidence if spec is not None else "unknown",
            "is_master": bool(spec.is_master) if spec is not None else False,
            "adaptive_stepping": bool(spec.adaptive_stepping) if spec is not None else False,
            "ticks_per_hyperperiod": (
                int(round(sched.hyperperiod / period)) if period > 0 else 0
            ),
        }

    base = sched.base_dt
    substeps = {
        cid: (int(round(info["period_s"] / base)) if base > 0 and info["period_s"] > 0 else 0)
        for cid, info in clocks.items()
    }

    events = sched.step_plan(0.0, horizon)
    return {
        "schema_id": compiled.schema.id,
        "abi_digest": compiled.state_layout.abi_digest(),
        "base_dt_s": base,
        "hyperperiod_s": sched.hyperperiod,
        "horizon_s": horizon,
        "n_sync_points": len(sched.sync_points),
        "sync_points_s": list(sched.sync_points),
        "clocks": clocks,
        "substeps_per_base_dt": substeps,
        "field_periods_s": sched.periods(),
        "event_driven_fields": list(sched.event_driven_fields),
        "interpolation_contracts": [
            {
                "from_clock": c.from_clock,
                "to_clock": c.to_clock,
                "method": c.method,
                "period_ratio": c.period_ratio,
                "is_downsampling": c.is_downsampling,
                "producer_latency_s": c.producer_latency,
                "sync_evidence": c.sync_evidence,
            }
            for c in sched.interpolation
        ],
        "error_budgets": {
            p.key: p.error_budget for p in sched.policies if p.error_budget is not None
        },
        "step_plan": [
            {
                "t_ns": e.t_ns,
                "t_s": e.t,
                "clocks": list(e.clocks),
                "is_sync": e.is_sync,
                "n_fields": len(e.fields),
                "fields": list(e.fields),
            }
            for e in events
        ],
        "n_events": len(events),
        "note": (
            "sync points are exact in integer nanoseconds; a runtime that accumulates "
            "t += dt in floating point will drift against this plan"
        ),
    }
