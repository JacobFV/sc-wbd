"""``ClaimManifest`` -- what a checkpoint's evidence does and does not support.

ARCHITECTURE.md §5 requires a claim manifest alongside every checkpoint, and §0
fixes what SC-WBD-001-beta is not.  This module makes that machine-readable and,
more importantly, **makes over-claiming raise**.

Design rules encoded here:

* A claim carries the evidence that supports it, the sources that evidence came
  from, and the finding that would disable it.  A claim without a falsifier is
  refused.
* A claim whose evidence is entirely ``simulator_conditioned`` may not be given
  ``biological`` status (body.tex §6.3).
* The strings "digital twin", "clinically validated", "treatment" and friends
  are refused outright in any claim text.
* ``cannot_do`` is a required, non-empty field.  A manifest without an explicit
  statement of limits does not validate.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

# The refusal DEFINITION lives in the schema layer; this module is the
# enforcement point. Imported at module scope rather than inside the method
# because `R12Violation` derives from it -- see that class's docstring.
from ..schema.refusals import CompilerRefusal

__all__ = ["Claim", "ClaimManifest", "OverclaimError", "R12Violation", "FAMILY_STATE_PHRASES"]

EvidenceStatus = Literal["simulator_conditioned", "measured", "prior", "mixed", "none"]
ClaimStatus = Literal["supported", "partial", "unsupported", "untested", "negative"]

#: phrases that may not appear in a claim, no matter what the numbers say
FORBIDDEN = (
    r"digital\s+twin",
    r"clinical(ly)?\s+(validated|approved|ready)",
    r"\btreatment\b",
    r"\btherapy\b",
    r"\bdiagnos(is|tic|e)\b",
    r"\bconsciousness\s+(measure|detector|ground\s*truth)",
    r"\bphi\b\s*(estimate|value)",
    r"medical\s+device",
    r"cure",
)


#: Phrases that assert the §2.1 differentiator — heterogeneous, region-indexed,
#: operator-valued state.  A checkpoint whose config is the equal-capacity
#: generic control may not carry them.  This list is deliberately about *what the
#: artifact is*, not about how good it is: R12 is a conformance refusal, not a
#: performance one.
FAMILY_STATE_PHRASES = (
    r"heterogene\w*\s+(regional\s+|region[-\s]indexed\s+)?state",
    r"region[-\s]indexed\s+state",
    r"operator[-\s]valued\s+(regional\s+)?state",
    r"per[-\s]famil\w+\s+(operator|backend|core)",
    r"per[-\s]region\s+operator",
    r"structured\s+regional\s+state",
)


class OverclaimError(ValueError):
    """A claim exceeded what the artifact's evidence can support."""


class R12Violation(CompilerRefusal, OverclaimError):
    """Refusal **R12** — the artifact claims §2.1's differentiator but is the control.

    **One rule, one enforcer, one exception.** R12 used to exist twice: this
    class (an ``OverclaimError``, i.e. a ``ValueError``) and
    ``scwbd.schema.refusals.CompilerRefusal``, in unrelated hierarchies, with
    ``validate()`` reaching the schema one first — so this class's message was
    unreachable and four tests matched a live implementation that a second one
    pre-empted. It was deferred as "a design call" for three runs.

    The call was in fact already made and written down, in
    ``_r12_predicate``'s docstring: *the definition belongs with R01–R11 in the
    schema refusal set; the enforcement point is checkpoint emission and stays
    here; when the canonical predicate lands, delete the local fallback.* It
    landed and the fallback was not deleted. It is deleted now.

    This class survives as the single exception raised at the enforcement point,
    and derives from **both** vocabularies so neither side's callers break:
    ``except CompilerRefusal`` (schema) and ``except OverclaimError``
    (foundation) both catch it, and it carries the canonical predicate's
    ``code``, ``remedy``, ``detail`` and ``evidence`` unchanged.

    ``ARCHITECTURE.md`` §5: "each family declares its own backend. A single global
    ``local_core`` string is **not** conformant — that is the equal-capacity
    generic control of ``body.tex`` §11.4, not the model. Refusal **R12**
    enforces this at checkpoint emission."

    This is the refusal that would have caught run 1.  ``reports/scope_gap.md``:
    "We built the control arm of §11.4's first required ablation and shipped it
    under the name of the treatment arm."  Nothing in the pipeline could see that,
    because the arm was never a machine-readable property of the artifact.  It is
    now: :meth:`ClaimManifest.declare_regional_state`, populated from
    ``SCWBD.family_report()`` by ``save_checkpoint``.
    """

    code = "R12"


@dataclass
class Claim:
    """One statement, its evidence, and the observation that would kill it."""

    id: str
    statement: str
    status: ClaimStatus
    evidence_status: EvidenceStatus
    evidence: dict[str, Any] = field(default_factory=dict)
    sources: tuple[str, ...] = ()
    falsifier: str = ""
    caveats: tuple[str, ...] = ()
    gate: str = ""  # G1..G5 when the claim corresponds to a claim gate
    #: True when this claim depends on heterogeneous region-indexed state
    #: (body.tex §2.1).  Refusal R12 rejects it on a control-arm checkpoint.
    requires_family_state: bool = False

    def validate(self) -> None:
        text = f"{self.statement} {' '.join(self.caveats)}"
        for pat in FORBIDDEN:
            if re.search(pat, text, flags=re.I):
                raise OverclaimError(
                    f"claim {self.id!r} contains forbidden phrasing matching /{pat}/: SC-WBD-001-beta "
                    "is not a validated digital twin, not a clinical device, and makes no medical claim."
                )
        if not self.falsifier.strip():
            raise OverclaimError(
                f"claim {self.id!r} has no falsifier. A module is done only when it ships a written "
                "statement of what empirical finding would disable it (ARCHITECTURE.md §4)."
            )
        if self.evidence_status == "simulator_conditioned" and self.status == "supported":
            raise OverclaimError(
                f"claim {self.id!r} is supported only by simulator-conditioned evidence and therefore "
                "cannot have status 'supported'. Simulated data exercise physics and rare regimes but "
                "cannot establish biological validity (body.tex §6.3). Use status='partial' with an "
                "explicit caveat, or supply measured evidence."
            )


from ..schema.designation import MODEL_DESIGNATION


@dataclass
class ClaimManifest:
    """The artifact's epistemic contract.  Written next to the weights."""

    #: Derived by ``save_checkpoint`` from the config; this default exists only
    #: for manifests built standalone and must never be a second literal.
    model_id: str = MODEL_DESIGNATION
    schema_version: str = "scwbd-schema/1.0.0"
    created_utc: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    git_sha: str = ""
    config_hash: str = ""
    weights_hash: str = ""
    environment: dict[str, Any] = field(default_factory=dict)

    is_a: tuple[str, ...] = (
        "A population/general adult conditional multirate whole-brain neural operator with an "
        "amortized posterior over global coupling, conduction velocity, regional E/I balance and "
        "observation nuisance.",
        "A research artifact for hypothesis ranking and simulation.",
    )
    is_not: tuple[str, ...] = (
        "NOT a validated model of any specific person.",
        "NOT a clinical device and not evidence for any medical decision.",
        "NOT evidence that any admitted operator is neurally realized.",
        "NOT a source of stimulation protocols, joint commands, trajectories or actuation authority.",
        "NOT a consciousness measure; no Phi is estimated and no consciousness ground truth exists.",
    )
    training_sources: list[dict[str, Any]] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    negative_results: list[dict[str, Any]] = field(default_factory=list)
    cannot_do: tuple[str, ...] = ()
    overrides: list[dict[str, Any]] = field(default_factory=list)
    anatomy: dict[str, Any] = field(default_factory=dict)
    #: ``SCWBD.family_report()`` — which arm of body.tex §11.4 this artifact is,
    #: the family partition, the backend each family got, and where the partition
    #: came from.  Empty means "not declared", which R12 treats as not conformant.
    regional_state: dict[str, Any] = field(default_factory=dict)
    corpus: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    # -- construction -----------------------------------------------------
    def add_claim(self, claim: Claim) -> "ClaimManifest":
        claim.validate()
        if any(c.id == claim.id for c in self.claims):
            raise ValueError(f"duplicate claim id {claim.id!r}")
        self.claims.append(claim)
        return self

    def add_negative(self, name: str, detail: str, numbers: dict[str, Any] | None = None) -> "ClaimManifest":
        """A gate that fails is a result, not a bug (ARCHITECTURE.md §4)."""
        self.negative_results.append({"name": name, "detail": detail, "numbers": numbers or {}})
        return self

    def declare_regional_state(self, report: Mapping[str, Any]) -> "ClaimManifest":
        """Record ``SCWBD.family_report()``.  Called by ``save_checkpoint``."""
        self.regional_state = dict(report)
        return self

    # -- refusal R12 ------------------------------------------------------
    def _r12_predicate(self):
        """The canonical R12 predicate, if the refusal set defines one.

        **Ownership seam.** The refusal *definition* belongs with R01-R11 in the
        schema/compiler refusal set (📜 Noether), not in ``foundation``; the
        *enforcement point* — checkpoint emission — is here and stays here. One
        definition, one enforcement point. When Noether's canonical predicate
        lands, this lookup finds it and the local fallback below stops running;
        delete the fallback at that point rather than leaving two.
        """
        for mod, attr in (
            ("scwbd.schema.refusals", "r12_predicate"),
            ("scwbd.compiler.refusals", "r12_predicate"),
        ):
            try:
                import importlib

                return getattr(importlib.import_module(mod), attr)
            except (ImportError, AttributeError):
                continue
        return None

    def refuse_r12(self, config: Any = None, poset: Any = None) -> None:
        """Refuse a §2.1 differentiator claim on a §11.4 control-arm artifact.

        Two ways to trip it, because run 1 tripped the second one:

        1. a claim with ``requires_family_state=True``;
        2. a claim whose *prose* asserts heterogeneous / region-indexed /
           operator-valued state or per-family operators.

        Prose counts.  The scope gap was not a wrong number, it was a correct
        artifact described in the words of a different one.
        """
        canonical = self._r12_predicate()
        if canonical is None:
            raise RuntimeError(
                "no canonical R12 predicate is importable from "
                "scwbd.schema.refusals or scwbd.compiler.refusals. This used to "
                "fall back to a second implementation living here, which is how "
                "R12 came to exist twice in unrelated exception hierarchies. The "
                "fallback is deleted: one definition, one enforcement point. "
                "Restore the schema predicate rather than reinstating a local copy."
            )
        # D8: pass the CONFIG through. The predicate is
        # `r12_predicate(manifest, config)` and its prolongation half reads the
        # poset, which a manifest does not record -- calling it with the manifest
        # alone silently ran only the operator and prose halves.
        try:
            canonical(self, config, poset) if poset is not None else canonical(self, config)
        except CompilerRefusal as exc:
            # ONE exception at the enforcement point. `R12Violation` subclasses
            # both `CompilerRefusal` and `OverclaimError`, so schema-side callers
            # (`except CompilerRefusal`) and foundation-side callers
            # (`except OverclaimError`) both still work, and the two R12
            # vocabularies stop being a fork.
            raise R12Violation(
                exc.code,
                getattr(exc, "remedy", None),
                getattr(exc, "offending_object", None),
                detail=getattr(exc, "detail", "") or "",
                evidence=getattr(exc, "evidence", None),
            ) from exc

    # -- validation -------------------------------------------------------
    def validate(self, config: Any = None, poset: Any = None) -> "ClaimManifest":
        self.refuse_r12(config, poset)
        if not self.cannot_do:
            raise OverclaimError(
                "ClaimManifest.cannot_do is empty. An explicit statement of what the model cannot do "
                "is a required deliverable, not an optional courtesy."
            )
        for c in self.claims:
            c.validate()
        if self.training_sources and not any(
            s.get("evidence_status") == "measured" for s in self.training_sources
        ):
            for c in self.claims:
                if c.status == "supported" and c.evidence_status != "prior":
                    raise OverclaimError(
                        f"claim {c.id!r} is 'supported' but no measured source appears in the training "
                        "mixture. Nothing trained purely on simulator output is biologically supported."
                    )
        return self

    # -- io ---------------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["claims"] = [asdict(c) for c in self.claims]
        return d

    def save(self, path: str | Path, *, config: Any = None) -> Path:
        self.validate(config)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), indent=2, default=str))
        return p

    @classmethod
    def load(cls, path: str | Path) -> "ClaimManifest":
        d = json.loads(Path(path).read_text())
        claims = [Claim(**c) for c in d.pop("claims", [])]
        for k in ("is_a", "is_not", "cannot_do"):
            if k in d and isinstance(d[k], list):
                d[k] = tuple(d[k])
        m = cls(**d)
        m.claims = claims
        return m

    def to_markdown(self) -> str:
        lines = [
            f"# {self.model_id} - Claim Manifest",
            "",
            f"- created: `{self.created_utc}`  git: `{self.git_sha}`",
            f"- config hash: `{self.config_hash}`  weights hash: `{self.weights_hash}`",
            "",
            "## What this artifact is",
            *[f"- {s}" for s in self.is_a],
            "",
            "## What this artifact is NOT",
            *[f"- {s}" for s in self.is_not],
            "",
            "## What it cannot do",
            *[f"- {s}" for s in self.cannot_do],
            "",
            "## Claims",
        ]
        for c in self.claims:
            lines += [
                f"### `{c.id}` - **{c.status}** ({c.evidence_status})" + (f" [gate {c.gate}]" if c.gate else ""),
                f"{c.statement}",
                "",
                f"- evidence: `{json.dumps(c.evidence, default=str)}`",
                f"- sources: {', '.join(c.sources) or 'none'}",
                f"- falsifier: {c.falsifier}",
                *[f"- caveat: {x}" for x in c.caveats],
                "",
            ]
        if self.negative_results:
            lines += ["## Negative results (reported, not deleted)"]
            for n in self.negative_results:
                lines += [f"- **{n['name']}**: {n['detail']}  `{json.dumps(n['numbers'], default=str)}`"]
        return "\n".join(lines)


def hash_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
