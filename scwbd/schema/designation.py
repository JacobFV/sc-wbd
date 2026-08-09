"""R12: the SC-WBD designation is a structural claim, not a filename.

Why this exists
---------------
SC-WBD-001-beta was trained, evaluated, and released as *the model*.  It has one
operator for all 454 regions (``ModelConfig.local_core`` is a single string,
resolved once and applied everywhere) and a resolution poset with one node and
no maps (``compiler_bridge._poset``).  ``body.tex`` sec. 0.2 names exactly two
differentiators -- heterogeneous operator-valued regional state, and non-nested
source-native resolutions -- and that artifact has neither.  It is the
equal-capacity generic **control arm** of sec. 11.4's first required ablation.

Nothing in the compiler could tell the difference, so it was emitted under the
model's name and its FAIL was read as a result about the thesis.  See
``reports/scope_gap.md``.

Ownership
---------
The refusal *definition* lives here, with R01-R11, because a refusal defined in
the module it polices is a self-assessment rather than a refusal -- and the
whole reason R12 exists is that ``foundation`` emitted a control-arm artifact
under the model's name with nothing outside it able to object.  The *enforcement
point* is checkpoint emission, in ``scwbd.foundation``.  One definition, one
call site.  :func:`~scwbd.schema.refusals.r12_predicate` is the seam
``ClaimManifest.refuse_r12`` looks up.

What R12 refuses
----------------
Emitting under the SC-WBD designation an artifact whose

1. regional operator assignment is **constant across all regions**, and
2. resolution poset **declares no prolongation**,

unless the run declares itself a control -- in which case it is emitted, and
remains fully runnable, under a *control* designation no downstream reader can
confuse with the model.  A control run is legitimate.  A control run wearing the
model's name is not.

Both conditions are required.  One backend with a real multiresolution lattice,
or per-family backends at a single scale, is a partial implementation and
somebody else's refusal to write.  The sec. 11.4 control is specifically the arm
with neither differentiator.

Two further things are refused, because each is the same defect wearing a
disguise:

* a claim whose **prose** asserts the sec. 2.1 differentiator on a control-arm
  artifact.  The scope gap was not a wrong number; it was a correct artifact
  described in the words of a different one.
* an artifact whose own ``family_report()`` says ``ablation_arm="treatment"``
  while every **populated** family runs the same backend.  A partition can
  declare eleven families and still be one operator for every parcel, either
  because the families all resolved to the same backend or because the only
  differently-typed family has **no regions** -- on the real 414-parcel prior
  ``cerebellum`` is declared and empty.  Counting declared backends rather than
  backends that reach a region is precisely how a guard becomes decorative
  (``reports/decorative_guards.md``), so R12 counts populated families only.

What it reads
-------------
Whichever of these the caller has:

* the **config** -- ``model.family_state`` (the arm switch), ``model.family_cores``
  (family -> backend), ``model.local_core`` (one backend for everything),
  ``model.scale_prolongations``, and the top-level ``arm`` declaration;
* the **artifact's** ``regional_state`` -- ``SCWBD.family_report()`` as recorded
  in the checkpoint and the ``ClaimManifest``;
* the compiled :class:`~scwbd.schema.poset.ResolutionPoset`, when one exists.

The config alone settles the **control** direction: ``family_state=False`` is one
operator for every parcel by construction.  It cannot settle the **conformant**
direction, because only the partition knows which declared families actually
received regions -- so a config claiming ``family_state=True`` with no artifact
report to corroborate it is refused rather than believed.
"""

from __future__ import annotations

import re
from typing import Any, Iterator, Literal, Mapping, Sequence

from pydantic import Field, model_validator

from .base import SchemaModel
from .poset import ResolutionPoset
from .refusals import CompilerRefusal

__all__ = [
    "MODEL_DESIGNATION",
    "ArmRole",
    "ArmDeclaration",
    "OperatorAssignment",
    "ProlongationDeclaration",
    "FAMILY_STATE_PHRASES",
    "FAMILY_STATE_KEY",
    "FAMILY_CORES_KEY",
    "LOCAL_CORE_KEY",
    "PROLONGATION_KEY",
    "ARM_KEY",
    "DeclarationError",
    "read_arm",
    "read_operator_assignment",
    "read_prolongations",
    "differentiator_claims",
    "check_r12",
    "check_manifest_r12",
    "designation_for",
    "assert_designation",
]

#: The name a conformant SC-WBD checkpoint carries.
MODEL_DESIGNATION = "SC-WBD-001-beta"

#: Config keys R12 reads.  These are the contract (ARCHITECTURE.md sec. 5); a
#: config that spells the assignment some other way is unreadable, and an
#: unreadable assignment is refused rather than assumed conformant.
FAMILY_STATE_KEY = "family_state"  # model.family_state: bool, the arm switch
FAMILY_CORES_KEY = "family_cores"  # model.family_cores: {family: backend}
LOCAL_CORE_KEY = "local_core"  # model.local_core: one backend for everything
PROLONGATION_KEY = "scale_prolongations"  # model.scale_prolongations: ["fine<=coarse"]
ARM_KEY = "arm"  # top-level arm declaration

#: Phrases that assert body.tex sec. 2.1's differentiator -- heterogeneous,
#: region-indexed, operator-valued state.  A control-arm artifact may not carry
#: them.  Deliberately about *what the artifact is*, not how good it is: R12 is a
#: conformance refusal, not a performance one.
FAMILY_STATE_PHRASES: tuple[str, ...] = (
    r"heterogene\w*\s+(regional\s+|region[-\s]indexed\s+)?state",
    r"region[-\s]indexed\s+state",
    r"operator[-\s]valued\s+(regional\s+)?state",
    r"per[-\s]famil\w+\s+(operator|backend|core)",
    r"per[-\s]region\s+operator",
    r"structured\s+regional\s+state",
)


class DeclarationError(ValueError):
    """A declaration R12 must read is present but malformed."""


# ----------------------------------------------------------------------
# the declaration that makes a control run legitimate
# ----------------------------------------------------------------------
ArmRole = Literal["model", "control"]


class ArmDeclaration(SchemaModel):
    """Which arm of a comparison this run is.

    Defaults to ``"model"``, which is the fail-closed direction: a run that says
    nothing is claiming to be the model and is held to the model's structure.
    Being the control requires saying so, naming the comparison, and writing
    down why -- three deliberate acts.  That is what makes it **impossible to be
    the control by accident**, which is precisely what happened to
    SC-WBD-001-beta.

    This is *declared intent*.  ``FoundationConfig.ablation_arm()`` and
    ``SCWBD.family_report()["ablation_arm"]`` are *derived structure*.  R12 is
    the rule that the two must agree, or the artifact does not get the name.
    """

    role: ArmRole = "model"
    #: The comparison this run is the control arm of, e.g.
    #: ``"11.4:structured_regional_state"``.  Required for a control.
    controls_for: str = ""
    #: Why this arm is the control and what it holds fixed.  Required for a
    #: control, and required to be a sentence rather than a word.
    justification: str = ""

    @model_validator(mode="after")
    def _check(self) -> "ArmDeclaration":
        if self.role == "control":
            if not self.controls_for.strip():
                raise ValueError(
                    "a control run must name the comparison it is the control "
                    "arm of in arm.controls_for (e.g. "
                    "'11.4:structured_regional_state'); an unattributed control "
                    "is indistinguishable from a mislabelled model"
                )
            if len(self.justification.strip()) < 24:
                raise ValueError(
                    "a control run must state in arm.justification what it holds "
                    "fixed and what it is the control for; "
                    f"got {self.justification.strip()!r}"
                )
        else:
            if self.controls_for.strip() or self.justification.strip():
                raise ValueError(
                    f"arm.role={self.role!r} but arm.controls_for/justification "
                    "are set; a half-edited declaration is how a control becomes "
                    "a model by accident. Set role='control' or clear the fields"
                )
        return self

    @property
    def is_control(self) -> bool:
        return self.role == "control"

    def designation(self, base: str = MODEL_DESIGNATION) -> str:
        """The name an artifact from this run may carry.

        A control never carries the model's designation.  The name travels in
        the checkpoint payload, ``provenance.json`` and the ``ClaimManifest``,
        so a control arm cannot be quoted as the model downstream without
        someone editing all three.
        """
        if self.is_control:
            return f"{base}-CONTROL[{self.controls_for}]"
        return base


def read_arm(config: "Mapping[str, Any] | Any | None") -> ArmDeclaration:
    """The run's arm declaration; the default is ``role='model'``.

    Accepts a mapping **or** a config object. The annotation said ``Mapping`` and
    the body called ``config.get(ARM_KEY)``, so passing a ``FoundationConfig`` --
    which is what the checkpoint-emission path passes -- raised
    ``AttributeError: 'FoundationConfig' object has no attribute 'get'``.

    That is worse than a missing feature: the arm could not be read from a config
    object at all, so a run had no way to declare itself a control through the
    object that describes it. R12 then refuses for want of a declaration the
    caller had no means to make -- the same unactionable-remedy shape recorded in
    ARCHITECTURE.md O-7 on the manifest side.

    Duck-typed rather than converted, because a config that is neither a mapping
    nor carries an ``arm`` attribute must still yield the ``role='model'``
    default rather than raising: an absent declaration is a real state.
    """
    if config is None:
        return ArmDeclaration()
    if isinstance(config, Mapping):
        if not config:
            return ArmDeclaration()
        raw = config.get(ARM_KEY)
    else:
        raw = getattr(config, ARM_KEY, None)
    if raw is None:
        return ArmDeclaration()
    if isinstance(raw, ArmDeclaration):
        return raw
    # `FoundationConfig.arm` is an `ArmConfig` dataclass carrying exactly
    # role/controls_for/justification -- the same three fields as
    # `ArmDeclaration`, under a fourth name for one concept (O-7 again). Read it
    # structurally rather than by type, so the config layer does not have to
    # import the schema layer to be legible to it.
    if not isinstance(raw, Mapping) and all(
        hasattr(raw, f) for f in ("role", "controls_for", "justification")
    ):
        raw = {
            "role": getattr(raw, "role", "model"),
            "controls_for": getattr(raw, "controls_for", ""),
            "justification": getattr(raw, "justification", ""),
        }
    if not isinstance(raw, Mapping):
        raise DeclarationError(
            f"config.{ARM_KEY} must be a mapping with role/controls_for/"
            f"justification, got {type(raw).__name__}"
        )
    return ArmDeclaration(**{str(k): v for k, v in raw.items()})


# ----------------------------------------------------------------------
# condition 1: the regional operator assignment
# ----------------------------------------------------------------------
class OperatorAssignment(SchemaModel):
    """What operator each region actually runs.

    ``backends`` holds only families that **have regions**.  A family declared
    with a distinct backend and zero parcels changes nothing about what any
    region computes, and counting it would make R12 report heterogeneity that
    does not reach a single parcel.
    """

    #: populated family -> backend.  ``{"*": backend}`` for a global assignment.
    backends: dict[str, str] = Field(default_factory=dict)
    #: regions per populated family, same keys as ``backends``.
    region_counts: dict[str, int] = Field(default_factory=dict)
    #: families declared with no regions.  Never counted, always reported.
    unpopulated: tuple[str, ...] = ()
    #: Where this was read from, for the refusal message.
    source: str = ""
    #: True when nothing readable said what operator any region runs.
    unreadable: bool = False
    #: True when the config claims per-family operators but no artifact report
    #: exists to say which families actually received regions.
    unverified_claim: bool = False
    #: The arm the artifact reports about itself, when it reports one.
    self_reported_arm: str = ""

    @property
    def distinct(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.backends.values())))

    @property
    def n_regions(self) -> int:
        return sum(self.region_counts.values())

    @property
    def is_constant(self) -> bool:
        """One operator for every region -- the sec. 11.4 control's condition 1."""
        return len(self.distinct) <= 1

    def dominant_share(self) -> float:
        """Fraction of regions on the single most-used backend.

        Not a threshold -- R12 never refuses on it.  It is carried in the
        refusal/permit evidence so a reader can see that "heterogeneous" can
        still mean 97% of parcels on one operator.
        """
        total = self.n_regions
        if total <= 0:
            return 1.0 if self.backends else 0.0
        per: dict[str, int] = {}
        for fam, backend in self.backends.items():
            per[backend] = per.get(backend, 0) + self.region_counts.get(fam, 0)
        return max(per.values()) / total


def _families_from_report(report: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
    partition = report.get("partition")
    if isinstance(partition, Mapping) and isinstance(partition.get("families"), Sequence):
        return [f for f in partition["families"] if isinstance(f, Mapping)]
    if isinstance(report.get("families"), Sequence):
        return [f for f in report["families"] if isinstance(f, Mapping)]
    return None


def read_operator_assignment(
    model_cfg: Mapping[str, Any] | None = None,
    *,
    regional_state: Mapping[str, Any] | None = None,
) -> OperatorAssignment:
    """What operator each region runs, from the artifact if possible.

    Resolution order:

    1. ``regional_state`` (``SCWBD.family_report()``) with a partition -- the
       authoritative source, because it is the only one that knows how many
       regions each family got.  Unpopulated families are recorded and excluded.
    2. ``regional_state`` reporting ``family_state=False`` -- one backend for
       every parcel by construction.
    3. ``model.family_state=True`` with no artifact report -- a claim of
       per-family operators nothing can corroborate.  Flagged ``unverified``.
    4. ``model.family_state=False`` or absent, with ``model.local_core`` -- one
       backend for everything.
    5. nothing readable -- refused, not assumed conformant.
    """
    if regional_state:
        families = _families_from_report(regional_state)
        arm = str(regional_state.get("ablation_arm", "") or "")
        if families is not None:
            populated = {}
            counts = {}
            empty: list[str] = []
            for f in families:
                name = str(f.get("name", ""))
                n = int(f.get("n_regions", 0) or 0)
                backend = str(f.get("backend", "") or "")
                if n > 0:
                    populated[name] = backend
                    counts[name] = n
                else:
                    empty.append(name)
            declared_empty = regional_state.get("partition", {})
            if isinstance(declared_empty, Mapping):
                for name in declared_empty.get("unpopulated", ()) or ():
                    if str(name) not in empty:
                        empty.append(str(name))
            return OperatorAssignment(
                backends=populated,
                region_counts=counts,
                unpopulated=tuple(sorted(empty)),
                source="artifact:family_report.partition",
                self_reported_arm=arm,
            )
        if regional_state.get(FAMILY_STATE_KEY) is False:
            core = str(regional_state.get(LOCAL_CORE_KEY, "") or "")
            n = int(regional_state.get("n_regions", 0) or 0)
            return OperatorAssignment(
                backends={"*": core} if core else {},
                region_counts={"*": n} if core else {},
                source="artifact:family_report.local_core",
                unreadable=not core,
                self_reported_arm=arm,
            )

    cfg = model_cfg or {}
    if cfg.get(FAMILY_STATE_KEY) is True:
        cores = cfg.get(FAMILY_CORES_KEY)
        declared = (
            {str(k): str(v) for k, v in cores.items()}
            if isinstance(cores, Mapping)
            else {}
        )
        return OperatorAssignment(
            backends=declared,
            source=f"model.{FAMILY_CORES_KEY}",
            unverified_claim=True,
        )
    core = cfg.get(LOCAL_CORE_KEY)
    if isinstance(core, str) and core.strip():
        return OperatorAssignment(
            backends={"*": core.strip()},
            region_counts={"*": int(cfg.get("n_regions", 0) or 0)},
            source=f"model.{LOCAL_CORE_KEY}",
        )
    return OperatorAssignment(unreadable=True)


# ----------------------------------------------------------------------
# condition 2: the resolution poset's prolongations
# ----------------------------------------------------------------------
class ProlongationDeclaration(SchemaModel):
    """Which ``fine <= coarse`` pairs this run declares a prolongation for."""

    pairs: tuple[tuple[str, str], ...] = ()
    #: ``"model.scale_prolongations"`` or ``"poset"``.
    source: str = ""

    @property
    def declares_prolongation(self) -> bool:
        return bool(self.pairs)


def _parse_pair(entry: Any) -> tuple[str, str]:
    if isinstance(entry, Mapping):
        fine, coarse = entry.get("fine"), entry.get("coarse")
    elif isinstance(entry, str):
        if "<=" not in entry:
            raise DeclarationError(
                f"prolongation entry {entry!r} must be written 'fine<=coarse' "
                "naming the two scales it maps between"
            )
        fine, coarse = entry.split("<=", 1)
    elif isinstance(entry, Sequence) and len(entry) == 2:
        fine, coarse = entry
    else:
        raise DeclarationError(f"unreadable prolongation entry {entry!r}")
    fine, coarse = str(fine or "").strip(), str(coarse or "").strip()
    if not fine or not coarse:
        raise DeclarationError(f"prolongation entry {entry!r} names an empty scale")
    if fine == coarse:
        raise DeclarationError(
            f"prolongation entry {entry!r} is degenerate: a map from a scale to "
            "itself is not a prolongation"
        )
    return fine, coarse


def _as_mapping(obj: Any) -> Any:
    """A Mapping view of a config object, or ``obj`` unchanged.

    ISSUE-009. ``check_r12`` annotates ``config`` as ``Mapping | None`` and its
    own docstring names ``save_checkpoint`` as a call site — but save_checkpoint
    holds a ``FoundationConfig`` dataclass and passes it straight through, so
    ``config.get("model")`` raised ``AttributeError`` and **no manifest could be
    attached to a checkpoint at all**. R12 had therefore never run against a real
    training run.

    Coerced here, at the boundary, rather than at each of the three call sites:
    the schema layer must not import ``scwbd.foundation``, so it cannot name the
    type, but it can accept the shape. Dataclasses and simple attribute objects
    both work; anything already a Mapping passes through untouched.
    """
    if obj is None or isinstance(obj, Mapping):
        return obj
    import dataclasses

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # `asdict` recurses and would turn nested dataclasses into dicts too,
        # which is what the readers below want.
        return dataclasses.asdict(obj)
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    return obj


def read_prolongations(
    model_cfg: Mapping[str, Any] | None = None,
    *,
    poset: ResolutionPoset | None = None,
) -> ProlongationDeclaration:
    """Prolongations this run declares.

    A compiled :class:`~scwbd.schema.poset.ResolutionPoset` is authoritative --
    it is the object R02 validates, so reading it here means the two refusals
    cannot disagree about what was declared.  Otherwise the config's
    ``model.scale_prolongations`` is read.

    The config entry is not a free pass: whatever it names must appear in the
    compiled poset (checked when the poset is supplied), and a prolongation in
    the poset without a declared restriction partner and tested coverage is
    refused by **R02** at compile time.  R12 asks whether one was declared; R02
    asks whether it is any good.
    """
    if poset is not None:
        return ProlongationDeclaration(
            pairs=tuple((str(p.fine), str(p.coarse)) for p in poset.prolongations()),
            source="poset",
        )
    raw = (model_cfg or {}).get(PROLONGATION_KEY) or ()
    if isinstance(raw, (str, Mapping)):
        raw = [raw]
    if not isinstance(raw, Sequence):
        raise DeclarationError(
            f"model.{PROLONGATION_KEY} must be a list of 'fine<=coarse' entries, "
            f"got {type(raw).__name__}"
        )
    return ProlongationDeclaration(
        pairs=tuple(_parse_pair(e) for e in raw),
        source=f"model.{PROLONGATION_KEY}",
    )


# ----------------------------------------------------------------------
# the prose half: claims that assert the differentiator
# ----------------------------------------------------------------------
def differentiator_claims(claims: Sequence[Any]) -> tuple[tuple[str, str], ...]:
    """``(claim_id, why)`` for every claim asserting sec. 2.1's differentiator.

    Two ways to trip it, because run 1 tripped the second:

    1. ``Claim.requires_family_state=True``;
    2. prose asserting heterogeneous / region-indexed / operator-valued state or
       per-family operators.
    """
    out: list[tuple[str, str]] = []
    for c in claims:
        cid = str(getattr(c, "id", "") or getattr(c, "name", "") or "?")
        if getattr(c, "requires_family_state", False):
            out.append((cid, "claim.requires_family_state=True"))
            continue
        caveats = getattr(c, "caveats", ()) or ()
        text = f"{getattr(c, 'statement', '')} {' '.join(str(x) for x in caveats)}"
        for pat in FAMILY_STATE_PHRASES:
            if re.search(pat, text, flags=re.I):
                out.append((cid, f"statement matches /{pat}/"))
                break
    return tuple(out)


# ----------------------------------------------------------------------
# the refusal
# ----------------------------------------------------------------------
def check_r12(
    *,
    config: Mapping[str, Any] | None = None,
    regional_state: Mapping[str, Any] | None = None,
    claims: Sequence[Any] = (),
    poset: ResolutionPoset | None = None,
    base_designation: str = MODEL_DESIGNATION,
    require_arm_evidence: bool = True,
) -> Iterator[CompilerRefusal]:
    """Yield R12 for everything wrong with this artifact's designation.

    Every argument is optional because the three call sites hold different
    things: the compiler has a schema, ``save_checkpoint`` has a config and a
    model, and ``ClaimManifest.validate`` has claims and a ``regional_state``.
    R12 checks what it can see and refuses what it cannot -- but only when the
    caller supplies enough to be talking about an artifact at all.

    ``require_arm_evidence=False`` suppresses the "this artifact does not say
    which arm it is" refusal, for callers validating a manifest that is not yet
    attached to a checkpoint.
    """

    def refuse(obj: Any, detail: str, **evidence: Any) -> CompilerRefusal:
        return CompilerRefusal(
            "R12", offending_object=obj, detail=detail, evidence=evidence
        )

    try:
        arm = read_arm(config)
    except (DeclarationError, ValueError) as exc:
        yield refuse(base_designation, f"arm declaration is not valid: {exc}")
        return

    # An artifact may declare its arm through `regional_state.ablation_arm` as
    # well as through `config.arm.role`, and BOTH count.
    #
    # ARCHITECTURE.md O-7: R12's remedy says "declare arm.role='control'", and
    # `foundation.ClaimManifest` has no `arm` field to put it in -- the only
    # channel it has is `regional_state`, which `declare_regional_state`
    # populates from the model. So a manifest validated without a config was
    # refused for want of a declaration it had no means to make, while carrying
    # that exact declaration in the field it does have. A refusal whose remedy
    # is unactionable reads as help and wastes the reader's time looking for a
    # field that does not exist.
    #
    # Config wins where both are present and disagree: the config describes the
    # run, the report describes the weights, and a run that says "model" while
    # its weights say "control" is refused below on the operator condition
    # anyway.
    if not arm.is_control and regional_state:
        declared_arm = str((regional_state or {}).get("ablation_arm") or "").strip().lower()
        if declared_arm == "control":
            arm = ArmDeclaration(
                role="control",
                controls_for=str(
                    (regional_state or {}).get("controls_for")
                    or "11.4:structured_regional_state"
                ),
                justification=str(
                    (regional_state or {}).get("arm_justification")
                    or "declared by the artifact's own regional-state report "
                    "(regional_state.ablation_arm='control')"
                ),
            )

    config = _as_mapping(config)
    model_cfg = _as_mapping(config.get("model")) if config else None
    if config is not None and not isinstance(model_cfg, Mapping):
        yield refuse(
            base_designation,
            "config has no 'model' section, so the regional operator assignment "
            "cannot be read; an artifact whose operator assignment is unknown "
            "cannot be certified as anything",
        )
        return

    offenders = differentiator_claims(claims)

    if config is None and not regional_state:
        # Nothing but claims.  A manifest not yet attached to a checkpoint may
        # legitimately say nothing about its arm -- but it may not assert the
        # differentiator with no evidence of having it.
        if offenders:
            yield _prose_refusal(refuse, base_designation, offenders, arm_note=None)
        return

    assignment = read_operator_assignment(model_cfg, regional_state=regional_state)

    if assignment.unreadable:
        yield refuse(
            base_designation,
            f"neither the artifact's family report nor model.{FAMILY_STATE_KEY} / "
            f"model.{LOCAL_CORE_KEY} says what operator any region runs, so the "
            "regional operator assignment cannot be read from what will ship "
            "with these weights",
            model_keys=sorted(str(k) for k in (model_cfg or {})),
            regional_state_keys=sorted(str(k) for k in (regional_state or {})),
        )
        return

    if assignment.unverified_claim:
        yield refuse(
            base_designation,
            f"model.{FAMILY_STATE_KEY} is true and model.{FAMILY_CORES_KEY} names "
            f"{sorted(assignment.backends)}, but no artifact family report "
            "accompanies it. Only the partition knows which declared families "
            "actually received regions -- a family can be declared with its own "
            "backend and hold zero parcels -- so this claim of heterogeneity "
            "cannot be checked. Supply SCWBD.family_report() as regional_state",
            declared=dict(assignment.backends),
        )
        return

    # -- the artifact's self-report must survive contact with its own partition
    if assignment.self_reported_arm == "treatment" and assignment.is_constant:
        yield refuse(
            base_designation,
            "the artifact reports ablation_arm='treatment' but every populated "
            f"family runs the same backend ({assignment.distinct[0]!r} over "
            f"{assignment.n_regions} regions in "
            f"{len(assignment.backends)} families"
            + (
                f"; the only differently-typed families are unpopulated: "
                f"{list(assignment.unpopulated)}"
                if assignment.unpopulated
                else ""
            )
            + "). A family with no regions changes what no parcel computes, so "
            "this partition is one operator for every region and the "
            "self-reported arm is wrong",
            backends=list(assignment.distinct),
            unpopulated=list(assignment.unpopulated),
            region_counts=dict(assignment.region_counts),
        )
        return

    try:
        prolongation = read_prolongations(model_cfg, poset=poset)
        declared = (
            read_prolongations(model_cfg) if poset is not None else prolongation
        )
    except DeclarationError as exc:
        yield refuse(base_designation, f"prolongation declaration is not valid: {exc}")
        return

    if poset is not None and set(declared.pairs) - set(prolongation.pairs):
        missing = sorted(set(declared.pairs) - set(prolongation.pairs))
        yield refuse(
            base_designation,
            f"config declares prolongations {missing} that the compiled "
            "resolution poset does not carry; the declaration would be the only "
            "evidence they exist",
            config_pairs=[list(p) for p in declared.pairs],
            poset_pairs=[list(p) for p in prolongation.pairs],
        )
        return

    # THE CONFIG FIELD IS NOT EVIDENCE. Condition 2 is satisfied only by a
    # prolongation carried in the COMPILED poset, which is the object R02
    # validates. `model.scale_prolongations` states an intention; nothing checks
    # it, so honouring it here would let an edit to a YAML key switch off the
    # refusal that polices overclaiming.
    #
    # `tests/foundation/test_resolution_pair_r02.py` had to pin that field EMPTY
    # to keep R12 firing, and said why: "A config key that switches a refusal off
    # is not a declaration, it is an exemption." That pin is the workaround; this
    # is the fix. With the field no longer load-bearing, a run may declare its
    # prolongations honestly and R12 still fires unless the poset carries them.
    poset_backed = prolongation.declares_prolongation and prolongation.source == "poset"
    is_control_shaped = assignment.is_constant and not poset_backed

    # An unverifiable declaration is not refused on its own -- when the operators
    # are already heterogeneous it changes no verdict, and refusing there would be
    # noise. It is refused where it WOULD have mattered, through the prose refusal
    # below, whose note says why the declaration did not count.
    unverifiable = prolongation.declares_prolongation and not poset_backed

    if not is_control_shaped:
        return  # conformant, or partial in a way R12 does not police

    # From here the artifact IS the sec. 11.4 control arm.
    if offenders:
        yield _prose_refusal(
            refuse,
            base_designation,
            offenders,
            arm_note=(
                f"this artifact is the equal-capacity generic-operator CONTROL: "
                f"one operator ({assignment.distinct[0]!r}) for every region, "
                f"read from {assignment.source}, and "
                + (
                    f"a prolongation declared only in model.{PROLONGATION_KEY} "
                    f"({sorted(prolongation.pairs)}) with no compiled resolution "
                    "poset to carry it -- R12 reads the poset R02 validates, so a "
                    "config key is a statement of intent and cannot discharge this "
                    "refusal"
                    if unverifiable
                    else "no declared prolongation"
                )
            ),
        )
        return

    if arm.is_control:
        return  # legitimate, declared, and renamed by ArmDeclaration.designation

    yield refuse(
        base_designation,
        f"{base_designation} would be emitted with one operator "
        f"({assignment.distinct[0]!r}) for every region, read from "
        f"{assignment.source}, and a resolution poset declaring no prolongation "
        f"(read from {prolongation.source}). That is the equal-capacity generic "
        "control arm of body.tex sec. 11.4's first ablation, not the model. "
        "Declare arm.role='control' to emit it under a control designation, or "
        "give it per-family operators over populated families and a declared "
        "restriction/prolongation pair",
        assignment_source=assignment.source,
        backends=list(assignment.distinct),
        n_populated_families=len(assignment.backends),
        unpopulated=list(assignment.unpopulated),
        prolongation_source=prolongation.source,
        n_prolongations=len(prolongation.pairs),
        arm_role=arm.role,
    )


def _prose_refusal(refuse, base_designation, offenders, *, arm_note: str | None):
    what = arm_note or "this artifact declares no regional-state arm at all"
    return refuse(
        base_designation,
        ", ".join(f"claim {i!r} ({why})" for i, why in offenders)
        + f" asserts body.tex sec. 2.1's differentiator, but {what}. Section "
        "11.4's first required ablation is 'structured regional state versus one "
        "scalar or pooled vector per region'; this artifact is the second of "
        "those two, and its results are a measurement of the control rather than "
        "a test of the thesis. Restate the claim as a control-arm result, or "
        "give the artifact per-family operators over populated families",
        offending_claims=[list(o) for o in offenders],
    )


def check_manifest_r12(manifest: Any, config: Mapping[str, Any] | None = None) -> None:
    """The seam ``ClaimManifest.refuse_r12`` calls.  Raises, or returns None.

    ``manifest`` is duck-typed on purpose -- the schema layer must not import
    ``scwbd.foundation`` -- and needs only ``.regional_state`` and ``.claims``.
    Passing ``config`` as well is what lets the prolongation half of R12 run;
    without it only the operator half and the prose half are checked, and the
    refusal message says so.
    """
    regional_state = getattr(manifest, "regional_state", None) or None
    claims = list(getattr(manifest, "claims", ()) or ())
    for refusal in check_r12(
        config=config,
        regional_state=regional_state,
        claims=claims,
        base_designation=str(getattr(manifest, "model_id", MODEL_DESIGNATION)),
    ):
        raise refusal


def designation_for(
    config: Mapping[str, Any] | None, *, base: str = MODEL_DESIGNATION
) -> str:
    """The name an artifact built from ``config`` may carry."""
    return read_arm(config).designation(base)


def assert_designation(
    config: Mapping[str, Any] | None = None,
    *,
    regional_state: Mapping[str, Any] | None = None,
    claims: Sequence[Any] = (),
    poset: ResolutionPoset | None = None,
    base_designation: str = MODEL_DESIGNATION,
) -> str:
    """Return the designation this artifact may carry, or raise R12.

    Fail-closed: the first refusal is raised, and R12 is not overridable
    (:data:`~scwbd.schema.refusals.NON_OVERRIDABLE_CODES`), because an override
    demotes the claim class and a demoted claim class does not rename anything.
    """
    for refusal in check_r12(
        config=config,
        regional_state=regional_state,
        claims=claims,
        poset=poset,
        base_designation=base_designation,
    ):
        raise refusal
    return designation_for(config, base=base_designation)
