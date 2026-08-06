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
            "AS GENERATED (the bytes 001-beta actually trained on): mechanism A, timescale "
            "clamped to the support boundary, 19.07%; mechanism B, timescale prior never "
            "arrives at all, 21.62% (Stuart-Landau + Jansen-Rit). Agent Hodgkin's "
            "silent-skip fix landed AFTER these bytes were written, so the post-fix figures "
            "(A 22.32%, B 13.51%) describe a corpus that does not exist yet. Note the "
            "direction: the fix moves trajectories from 'no prior' into 'prior, clamped', "
            "so mechanism A RISES. A reader skimming for the smaller number takes 13.51% "
            "and misses that the other figure went up."
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
        id="no_ei_prior_at_all",
        artifact="scwbd-001-beta",
        measured=(
            "mechanism C: 27.03% of the corpus carries no E/I prior at all (verified "
            "independently from the shard index, not inferred from the backend mix)"
        ),
        consequence=(
            "Better than a quarter of the training signal contains no excitation/inhibition "
            "prior whatsoever. Any claim that this model has learned E/I structure must "
            "state that the corpus could not have taught it over that fraction, and no "
            "gate may read an E/I-shaped result as evidence without excluding those shards."
        ),
        discloses_on=("A1_structured_state", "G2", "A5_typed_operators"),
    ),
    CorpusLimitation(
        id="elevated_loss_rate_not_periodicity",
        artifact="scwbd-001-beta",
        measured=(
            "roughly a third of logged steps show sim_forecast_nll above 3x the running "
            "floor (bench independently measured 23% and 25% on the two committed Stage I "
            "series). RETRACTED, and recorded as retracted: this was first relayed as a "
            "PERIODICITY (steps 80/180/220/320/380/440/500, 'last four exactly 60 apart'). "
            "It was tested forward -- period 60 from step 320 predicts a spike at 560 -- and "
            "failed at the first opportunity: a spike at 540, none at 560. Every gap is a "
            "multiple of 20 by construction because that is the logging grid, and Stage I's "
            "sim set is ~560 batches/epoch, nowhere near 60. A period was fitted to a run of "
            "three."
        ),
        consequence=(
            "The elevated-loss rate is real and carries genuine training cost, but it is a "
            "RATE, not a schedule. Bench independently confirms the driver is batch "
            "composition rather than optimisation: the spikes occur at the SAME steps with "
            "the SAME magnitudes across a 1.73x learning-rate difference (step 80: 11.62x at "
            "lr 6.0e-4 versus 10.54x at 3.46e-4; step 180: 4.74x versus 4.68x; step 220: "
            "4.54x versus 4.47x). Whether this becomes a corpus mechanism turns on batch "
            "composition, and no timing claim may be made from a grid-limited series."
        ),
        discloses_on=("G1", "A1_structured_state"),
        found_by="agent Turing (claimed, then falsified and withdrawn by its author)",
        mitigating_fact=(
            "the withdrawal came four minutes after the claim, from a forward test the "
            "author designed to be able to fail"
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
