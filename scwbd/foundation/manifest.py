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


class R12Violation(OverclaimError):
    """Refusal **R12** — the artifact claims §2.1's differentiator but is the control.

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

    def refuse_r12(self, config: Any = None) -> None:
        """Refuse a §2.1 differentiator claim on a §11.4 control-arm artifact.

        Two ways to trip it, because run 1 tripped the second one:

        1. a claim with ``requires_family_state=True``;
        2. a claim whose *prose* asserts heterogeneous / region-indexed /
           operator-valued state or per-family operators.

        Prose counts.  The scope gap was not a wrong number, it was a correct
        artifact described in the words of a different one.
        """
        canonical = self._r12_predicate()
        if canonical is not None:
            # D8: pass the CONFIG through. Noether's predicate is
            # `r12_predicate(manifest, config)` and its prolongation half reads
            # the poset, which a manifest does not record -- calling it with the
            # manifest alone silently ran only the operator and prose halves.
            canonical(self, config)
            return
        arm = str(self.regional_state.get("ablation_arm", "")) if self.regional_state else ""
        if arm == "treatment":
            return
        offenders: list[tuple[str, str]] = []
        for c in self.claims:
            if c.requires_family_state:
                offenders.append((c.id, "claim.requires_family_state=True"))
                continue
            text = f"{c.statement} {' '.join(c.caveats)}"
            for pat in FAMILY_STATE_PHRASES:
                if re.search(pat, text, flags=re.I):
                    offenders.append((c.id, f"statement matches /{pat}/"))
                    break
        if not offenders:
            return
        what = (
            "this checkpoint declares no regional-state arm at all"
            if not arm
            else "this checkpoint is the equal-capacity generic-operator CONTROL "
            "(ModelConfig.family_state=False: one local_core string and one state dimension "
            "for every parcel)"
        )
        raise R12Violation(
            "[R12] "
            + ", ".join(f"claim {i!r} ({why})" for i, why in offenders)
            + f" asserts body.tex §2.1's differentiator, but {what}. "
            "body.tex §11.4's first required ablation is 'structured regional state versus one "
            "scalar or pooled vector per region'; this artifact is the second of those two and "
            "its results are a measurement of the control, not a test of the thesis. "
            "Remedy: set model.family_state=true and assign per-family backends, or restate the "
            "claim as a control-arm result. See reports/scope_gap.md and reports/dynamics/family_state.md."
        )

    # -- validation -------------------------------------------------------
    def validate(self, config: Any = None) -> "ClaimManifest":
        self.refuse_r12(config)
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
