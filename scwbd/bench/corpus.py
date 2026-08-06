"""What the training corpus can and cannot support (agent J).

A gate measures a model.  A model can only carry what its training signal
contained.  When the two are confused, a model that was never taught something
gets marked FAIL for not knowing it — or, far worse, gets marked PASS on a
check its corpus could not have exercised.

This module holds the *measured* limitations of the SC-WBD-001-beta training
corpus (agent Turing's audit, ``reports/training/corpus_composition.md``) and
maps each to the claims it bounds.  A gate run against a registered artifact
consults this first, and reports ``COULD_NOT_RUN`` **naming the corpus** rather
than failing the model for a gap in its evidence.

The distinction is Appendix D's: correlation fitting versus held-out
perturbational prediction.  A trained whole-brain model sitting beside a
passing field stack looks like an end-to-end intervention path.  It is not one
unless the corpus contained interventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = ["CorpusLimitation", "CORPUS_LIMITATIONS", "REGISTERED_ARTIFACTS",
           "limitations_for", "blocking_limitation"]


@dataclass(frozen=True)
class CorpusLimitation:
    """A measured property of the training signal that bounds a claim."""

    id: str
    artifact: str
    measured: str
    consequence: str
    #: claim ids this limitation makes unevaluable (COULD_NOT_RUN, not FAIL)
    blocks: tuple[str, ...] = ()
    #: claim ids it does not block but must be disclosed on
    discloses_on: tuple[str, ...] = ()
    source: str = "reports/training/corpus_composition.md"
    found_by: str = "agent Turing"
    mitigating_fact: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "artifact": self.artifact, "measured": self.measured,
            "consequence": self.consequence, "blocks": list(self.blocks),
            "discloses_on": list(self.discloses_on), "source": self.source,
            "found_by": self.found_by, "mitigating_fact": self.mitigating_fact,
        }


#: Artifacts whose corpus has been audited.  A gate handed one of these consults
#: the register before reporting anything.
REGISTERED_ARTIFACTS: tuple[str, ...] = ("scwbd-001-beta",)


CORPUS_LIMITATIONS: tuple[CorpusLimitation, ...] = (
    CorpusLimitation(
        id="no_interventional_structure",
        artifact="scwbd-001-beta",
        measured=(
            "35 of 37 corpus shards carry control_graph: none; the remaining 2 are "
            "local_only"
        ),
        consequence=(
            "The corpus supports no claim about response to intervention or perturbation "
            "beyond the local case. Anything stronger is extrapolation from observational "
            "simulation. G4 therefore CANNOT BE SATISFIED by this artifact — not because "
            "the model failed, but because its training signal contains essentially no "
            "interventional structure. This reads COULD_NOT_RUN with the corpus named; it "
            "is not a model FAIL and it is certainly not a pass."
        ),
        blocks=("G4", "A10_correlation_vs_perturbation", "A5_typed_operators",
                "D08_operator_mechanism_claim"),
        discloses_on=("G1", "G2"),
    ),
    CorpusLimitation(
        id="timescales_clamped_to_support_boundary",
        artifact="scwbd-001-beta",
        measured=(
            "~18% of the corpus has regional timescales pinned to a support boundary rather "
            "than drawn from the prior (backend mix wilson_cowan 40.5%, wong_wang 32.4%, "
            "stuart_landau 13.5%, jansen_rit 8.1%, linear_gaussian 5.4%; measured clamp "
            "rates 47.5% RWW, 6.1% WC)"
        ),
        consequence=(
            "Where this model appears to have learned that regions are homogeneous in "
            "timescale, roughly a fifth of its training signal could have taught it that "
            "regardless of the brain, because the sampler could not express the prior. Any "
            "gate or ablation touching regional heterogeneity must disclose this rather "
            "than read homogeneity as a finding."
        ),
        discloses_on=("A1_structured_state", "G2", "G3"),
        mitigating_fact=(
            "RWW does not dominate the mixture at 32.4% and the corpus is genuinely "
            "backend-diverse; agent Turing recorded this without using it to wave the "
            "limitation away, and neither does this register"
        ),
    ),
    CorpusLimitation(
        id="slow_tier_never_built",
        artifact="scwbd-001-beta",
        measured="the slow tier was never built; the model sees only fast-tier dynamics",
        consequence=(
            "No claim about slow dynamics, and this compounds the timescale-clamping "
            "limitation above: the corpus is narrow in exactly the axis a multirate claim "
            "would need to be broad in."
        ),
        blocks=(),
        discloses_on=("A1_structured_state", "G3", "A3_resolution"),
    ),
    CorpusLimitation(
        id="ei_inversion_did_not_contaminate",
        artifact="scwbd-001-beta",
        measured=(
            "the corpus was generated through agent Hodgkin's backends as shipped, NOT via "
            "a direct name-match on ei_ratio"
        ),
        consequence=(
            "Recorded as a NEGATIVE RESULT rather than an assumption: the E/I inversion "
            "agent Hodgkin caught did not contaminate this corpus. Registered so that the "
            "check is known to have been made, not merely believed."
        ),
    ),
)


def limitations_for(claim_id: str, artifact: str | None = None
                    ) -> tuple[list[CorpusLimitation], list[CorpusLimitation]]:
    """``(blocking, disclosure_only)`` limitations for a claim on an artifact."""
    rel = [l for l in CORPUS_LIMITATIONS
           if artifact is None or l.artifact == artifact]
    return ([l for l in rel if claim_id in l.blocks],
            [l for l in rel if claim_id in l.discloses_on])


def blocking_limitation(claim_id: str, artifact: str | None) -> CorpusLimitation | None:
    """The first limitation that makes ``claim_id`` unevaluable on ``artifact``."""
    if artifact is None:
        return None
    blocking, _ = limitations_for(claim_id, artifact)
    return blocking[0] if blocking else None
