"""The eleven mandatory modeling refusals, plus this repository's local ones.

R01-R11 are verbatim from ``paper/thesis_contract.tex`` Table
``tab:compiler-refusals``, in table order as fixed by ARCHITECTURE.md sec. 2.
"These are compiler and runtime errors, not optional governance prose."

Every thesis refusal carries the *exact* remedy text from the thesis so that a
user who hits one is told what the thesis requires, not what an implementer
paraphrased.

**Local refusals** (currently R12) are *not* in the thesis table.  They are
refusals this repository added because it made a specific mistake and does not
intend to make it twice.  They are marked ``origin="local"`` so that no reader
and no report can mistake them for thesis authority, and they are excluded from
:data:`REFUSAL_CODES` -- the compiler's table-order check list -- because the
compiler's contract is "fail closed on all 11 refusals in the thesis table".
Use :data:`ALL_REFUSAL_CODES` when you mean every refusal this codebase raises.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, JsonValue

from .base import SchemaModel

__all__ = [
    "RefusalCode",
    "RefusalOrigin",
    "RefusalSpec",
    "REFUSALS",
    "REFUSAL_CODES",
    "THESIS_REFUSAL_CODES",
    "LOCAL_REFUSAL_CODES",
    "ALL_REFUSAL_CODES",
    "NON_OVERRIDABLE_CODES",
    "CompilerRefusal",
    "RefusalRecord",
    "remedy_for",
    "r12_predicate",
]

RefusalCode = Literal[
    "R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08", "R09", "R10", "R11",
    "R12",
]

#: Where a refusal's authority comes from.  ``"thesis"`` means the row exists in
#: Table tab:compiler-refusals and its remedy text is quoted, not written.
RefusalOrigin = Literal["thesis", "local"]

#: The thesis table, in table order.  This is the list the compiler driver walks
#: and the list ``tests/schema/refusals`` requires a failing fixture for.
REFUSAL_CODES: tuple[str, ...] = (
    "R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08", "R09", "R10", "R11",
)
THESIS_REFUSAL_CODES: tuple[str, ...] = REFUSAL_CODES

#: Refusals this repository added.  Not thesis rows; see the module docstring.
LOCAL_REFUSAL_CODES: tuple[str, ...] = ("R12",)

#: Every refusal any part of this codebase may raise.
ALL_REFUSAL_CODES: tuple[str, ...] = REFUSAL_CODES + LOCAL_REFUSAL_CODES


class RefusalSpec(SchemaModel):
    """One row of Table tab:compiler-refusals, or one local refusal."""

    code: RefusalCode
    rejected: str
    why: str
    remedy: str
    origin: RefusalOrigin = "thesis"
    #: Whether ``ClaimManifest.overrides`` may buy past this refusal in exchange
    #: for a demoted claim class.  A refusal about *what an artifact is called*
    #: is not overridable: demoting the claim class does not change the name,
    #: so an override would leave the mislabelling in place.
    overridable: bool = True


REFUSALS: dict[str, RefusalSpec] = {
    r.code: r
    for r in (
        RefusalSpec(
            code="R01",
            rejected=(
                "Unknown units, clock, physical support, frame, handedness, or "
                "transform lineage"
            ),
            why=(
                "Numerical compatibility would be mistaken for measurement "
                "compatibility"
            ),
            remedy=(
                "Supply a calibrated manifest with validity interval and "
                "uncertainty, or keep the source disconnected from the requested "
                "path"
            ),
        ),
        RefusalSpec(
            code="R02",
            rejected="Prolongation without a declared restriction partner and tested coverage",
            why="A coarse observation would be converted into unsupported fine structure",
            remedy=(
                "Provide paired maps, round-trip and held-out landmark tests, and "
                "an out-of-support uncertainty policy; otherwise return a "
                "distribution or refuse"
            ),
        ),
        RefusalSpec(
            code="R03",
            rejected="Global cross-scale state when overlap/cocycle residual exceeds tolerance",
            why="Conflicting local views would be hidden by one raster",
            remedy=(
                "Preserve separate sections, branch or marginalize the posterior, "
                "and emit an obstruction certificate identifying the failed path"
            ),
        ),
        RefusalSpec(
            code="R04",
            rejected="Effective or causal operator estimated from passive correlation alone",
            why="Common input, observation mixing, and omitted state remain alternatives",
            remedy=(
                "Label the object functional/statistical, or add an identified "
                "intervention or assumption set sufficient for the causal claim"
            ),
        ),
        RefusalSpec(
            code="R05",
            rejected="Learned residual allowed to dominate a mechanistic term silently",
            why="The named mechanism becomes unfalsifiable",
            remedy=(
                "Enforce a preregistered energy/gain ratio "
                "\\|R_theta\\|/\\|F_mech\\| <= rho_max on the declared validity "
                "set, report violations, or reclassify the entire module as a "
                "surrogate"
            ),
        ),
        RefusalSpec(
            code="R06",
            rejected="Adaptive step sizes for a learned propagator without semigroup testing",
            why="Coarse and composed fine transitions need not describe the same dynamics",
            remedy=(
                "Measure the semigroup residual at every permitted step pair; "
                "restrict the scheduler or include the discrepancy in prediction "
                "intervals"
            ),
        ),
        RefusalSpec(
            code="R07",
            rejected="Population, subject, and session effects without centering or shrinkage",
            why="The additive decomposition is not identified",
            remedy=(
                "Use centered/sum-to-zero constraints or proper hierarchical "
                "priors and test recovery in simulation"
            ),
        ),
        RefusalSpec(
            code="R08",
            rejected="Bias assigned a point estimate without an estimator or external bound",
            why="Calibration and model discrepancy can trade off non-identifiably",
            remedy=(
                "Classify the term as design-estimable, externally bounded, or "
                "prior-specified sensitivity; propagate the last category as a "
                "range"
            ),
        ),
        RefusalSpec(
            code="R09",
            rejected="Pseudo-likelihood compatibility treated as a calibrated posterior likelihood",
            why="Agreement penalties do not define a generative noise model",
            remedy=(
                "Report a generalized/pseudo-posterior and calibrate it "
                "empirically, or validate and promote the factor to a generative "
                "likelihood"
            ),
        ),
        RefusalSpec(
            code="R10",
            rejected=(
                "Derived scans, sessions, relatives, stimuli, or simulator replicas "
                "crossing a parent-level holdout"
            ),
            why="Duplicated evidence produces invalid confidence and shortcut prediction",
            remedy=(
                "Group by immutable lineage identifiers before splitting and fail "
                "the run when parentage is unresolved"
            ),
        ),
        RefusalSpec(
            code="R11",
            rejected="Intervention optimization outside an independently validated feasible set",
            why="A learned objective cannot guarantee device, exposure, or protocol safety",
            remedy=(
                "Require u in A_safe, independent safety checks, deferral, and the "
                "applicable ethics and regulatory review"
            ),
        ),
        RefusalSpec(
            code="R12",
            origin="local",
            overridable=False,
            rejected=(
                "A checkpoint emitted under the SC-WBD designation whose regional "
                "operator assignment is constant across all regions and whose "
                "resolution poset declares no prolongation"
            ),
            why=(
                "That artifact is the equal-capacity generic control arm of "
                "body.tex sec. 11.4's first ablation, and under the model's name "
                "its measurements read as measurements of the model"
            ),
            remedy=(
                "Assign operators per regional family and declare a validated "
                "restriction/prolongation pair; or declare the run a control "
                "(arm.role='control' naming the ablation it is the control for), "
                "which keeps it runnable under a control designation instead of "
                "the model's"
            ),
        ),
    )
}

#: Codes an override may never buy past.
NON_OVERRIDABLE_CODES: frozenset[str] = frozenset(
    code for code, spec in REFUSALS.items() if not spec.overridable
)

assert (
    tuple(c for c, s in REFUSALS.items() if s.origin == "thesis") == REFUSAL_CODES
), "refusal table must stay in thesis order"
assert (
    tuple(c for c, s in REFUSALS.items() if s.origin == "local") == LOCAL_REFUSAL_CODES
), "local refusals must be declared after the thesis table, in code order"
assert tuple(REFUSALS) == ALL_REFUSAL_CODES


def remedy_for(code: str) -> str:
    """Exact remedy text from Table tab:compiler-refusals (or a local refusal)."""
    try:
        return REFUSALS[code].remedy
    except KeyError:  # pragma: no cover - programming error
        raise KeyError(
            f"unknown refusal code {code!r}; expected one of {ALL_REFUSAL_CODES}"
        )


def r12_predicate(manifest: Any, config: Any = None) -> None:
    """The canonical R12 predicate.  Raises ``CompilerRefusal(code="R12")``.

    **This is the ownership seam.**  ``scwbd.foundation.manifest.ClaimManifest``
    ``.refuse_r12`` looks this name up on this module and delegates to it; the
    enforcement point (checkpoint emission) stays in ``foundation``, and the
    definition stays here with R01-R11.  A refusal defined inside the module it
    polices is a self-assessment, and self-assessment is what let run 1 emit the
    control arm under the model's name.

    ``config`` is optional and should be passed whenever the caller has one --
    ``save_checkpoint`` does.  Without it R12 can check the operator assignment
    and the claim prose but not the prolongation half, because nothing in the
    manifest records the resolution poset.

    Imported lazily: :mod:`scwbd.schema.designation` imports this module.
    """
    from .designation import check_manifest_r12

    check_manifest_r12(manifest, config)


def _describe(obj: Any) -> str:
    """A short, stable description of the offending object."""
    for attr in ("key", "id", "name", "persistent_id"):
        value = getattr(obj, attr, None)
        if isinstance(value, str) and value:
            return f"{type(obj).__name__}({value})"
        if isinstance(value, tuple):
            return f"{type(obj).__name__}{value}"
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return repr(obj)
    if isinstance(obj, (list, tuple)) and obj:
        return f"[{_describe(obj[0])}, ... x{len(obj)}]"
    return type(obj).__name__


class RefusalRecord(SchemaModel):
    """Serializable form of a refusal, stored in a compiled artifact."""

    code: RefusalCode
    remedy: str
    offending_object: str
    detail: str = ""
    #: Free-form evidence the check collected (residuals, ids, folds).
    evidence: dict[str, JsonValue] = Field(default_factory=dict)
    #: True when a ClaimManifest override let compilation proceed anyway.
    overridden: bool = False
    #: Claim class the artifact now carries because of the override.
    resulting_claim_class: str | None = None


class CompilerRefusal(Exception):
    """Fail-closed compiler error carrying the thesis remedy.

    Signature is fixed by ARCHITECTURE.md sec. 2:
    ``CompilerRefusal(code=..., remedy=..., offending_object=...)``.
    ``remedy`` defaults to the exact text from Table tab:compiler-refusals, so
    a check never has to retype it (and never gets it subtly wrong).
    """

    def __init__(
        self,
        code: str,
        remedy: str | None = None,
        offending_object: Any = None,
        *,
        detail: str = "",
        evidence: dict[str, Any] | None = None,
    ) -> None:
        if code not in REFUSALS:
            raise KeyError(f"unknown refusal code {code!r}")
        self.code: str = code
        self.remedy: str = remedy if remedy is not None else REFUSALS[code].remedy
        self.offending_object: Any = offending_object
        self.detail: str = detail
        self.evidence: dict[str, Any] = dict(evidence or {})
        self.spec: RefusalSpec = REFUSALS[code]
        super().__init__(str(self))

    def __str__(self) -> str:
        where = _describe(self.offending_object)
        head = f"{self.code}: {self.spec.rejected}"
        parts = [head, f"  offending object: {where}"]
        if self.detail:
            parts.append(f"  detail: {self.detail}")
        parts.append(f"  why: {self.spec.why}")
        parts.append(f"  remedy: {self.remedy}")
        return "\n".join(parts)

    def __repr__(self) -> str:
        return (
            f"CompilerRefusal(code={self.code!r}, "
            f"offending_object={_describe(self.offending_object)!r})"
        )

    def record(
        self, *, overridden: bool = False, resulting_claim_class: str | None = None
    ) -> RefusalRecord:
        return RefusalRecord(
            code=self.code,  # type: ignore[arg-type]
            remedy=self.remedy,
            offending_object=_describe(self.offending_object),
            detail=self.detail,
            evidence={k: _jsonable(v) for k, v in self.evidence.items()},
            overridden=overridden,
            resulting_claim_class=resulting_claim_class,
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    return _describe(value)
