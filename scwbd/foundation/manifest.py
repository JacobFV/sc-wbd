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
from typing import Any, Iterable, Literal, Sequence

__all__ = ["Claim", "ClaimManifest", "OverclaimError"]

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


class OverclaimError(ValueError):
    """A claim exceeded what the artifact's evidence can support."""


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


@dataclass
class ClaimManifest:
    """The artifact's epistemic contract.  Written next to the weights."""

    model_id: str = "SC-WBD-001-beta"
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

    # -- validation -------------------------------------------------------
    def validate(self) -> "ClaimManifest":
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

    def save(self, path: str | Path) -> Path:
        self.validate()
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
