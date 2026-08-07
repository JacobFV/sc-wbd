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
            "UPDATED after agent Galvani verified ds004024 from bytes (962 pulses, 12 runs, "
            "20 kHz, 69 channels, hashes matching). G4's refusal is now PRECISE rather than "
            "blanket: direction and delay are supported at N=2 subjects; gain, dose and "
            "state_dependence are STRUCTURALLY UNAVAILABLE because pybv 0.6.0 wrote empty "
            "date fields, so meas_date is None and the pre/post design's labels are not "
            "distributed. Fetching the remaining 11 subjects does not fix it. This is a "
            "sharper refusal, NOT a pass and NOT movement toward one -- 2 of 5 named "
            "quantities with the other 3 unavailable is still a gate that cannot conclude. "
            "BENCH SIGNS OFF on the ControlGraph shape: recovery_report() returning exactly "
            "G4's five quantity names is the right interface, and declining to overload the "
            "simulated corpus's control_graph: str field is correct -- that field names "
            "anatomical control connectivity, a different object, and merging them would "
            "have made a structural absence look like a populated one. "
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
        id="single_site_corpus",
        artifact="scwbd-001-beta",
        measured=(
            "EEGMMIDB is single-site: 109 participants (71/11/27), 290,673 windows, all from "
            "one recording setup. Agent Ada's auditor returned ok=True with 0 violations and "
            "pairwise disjointness recomputed independently -- the split is clean -- alongside "
            "the warning that 'this split cannot falsify a site/device shortcut'. THIS IS A "
            "PROPERTY OF THE CORPUS, NOT THE SPLIT, and no splitting strategy can repair it: "
            "there is no second site to hold out."
        ),
        consequence=(
            "D03 (Site/device shortcuts) is COULD_NOT_RUN with the corpus named, in the same "
            "form as G4's control_graph refusal: none of its three mandatory controls is "
            "CONSTRUCTIBLE here. Leave-site-out needs a second site; a nuisance-only "
            "classifier needs site/device labels that vary; within-site label permutation "
            "exists but alone cannot falsify a site shortcut. "
            "BENCH'S RULING ON G5, requested explicitly: G5 MAY RUN, and its verdict is "
            "meaningful, but its CLAIM IS NARROWER THAN G5 AS WRITTEN and a PASS may not be "
            "reported unqualified. The reasoning: site is CONSTANT across every arm, so it "
            "cannot confound the contrast between an individualised model and its "
            "population/anatomy-only/session-adapted baselines -- a constant explains no "
            "difference. What is unsupported is (a) that any measured advantage would "
            "replicate at another site, and (b) that the individualisation is not exploiting "
            "signal characteristics specific to this setup which happen to individuate here. "
            "So the licensed claim is 'individualization improves future prediction WITHIN "
            "THIS RECORDING SETUP', and any gate reading a participant-disjoint split as "
            "licensing broader generalisation is blocked from doing so before it measures "
            "anything."
        ),
        blocks=("D03_site_device_shortcuts",),
        discloses_on=("G5", "G1", "G2", "A7_individualization"),
        source="agent Ada's leakage_audit warning; verified by agent Turing's hard gate",
        found_by="agent Ada (auditor warning), relayed with the split verified clean",
    ),
    CorpusLimitation(
        id="anatomical_prior_and_simulator_not_independent_on_bold",
        artifact="scwbd-001-beta",
        measured=(
            "gradient cosine between anatomical_prior and sim_wholebrain on the `bold` "
            "module: mean 0.99999998, minimum 0.99999988, on 50 of 50 observations "
            "(gradient_conflict_stage3.md, verified by bench at line 44). `bold` holds 3,183 "
            "parameters across 8 tensors, so this is NOT a one-parameter degeneracy where "
            "cosine is trivially +/-1: two vectors parallel to float32 precision in 3,183 "
            "dimensions are the same vector up to scale."
        ),
        consequence=(
            "THE ANATOMICAL PRIOR AND THE SIMULATOR ARE ONE PIECE OF EVIDENCE ENTERING "
            "TWICE on `bold`, not two. This is Appendix D's derived-data-duplication row "
            "appearing BETWEEN SOURCE FAMILIES rather than between two scans of one "
            "participant -- the same defect at a level the table did not anticipate. No "
            "ablation may count their agreement as corroboration, and no gate may read them "
            "as two independent votes. Encoded here so the constraint binds before anything "
            "is measured rather than being remembered at write-up."
        ),
        discloses_on=("G2", "G3", "A5_typed_operators", "D12_dataset_family_breadth"),
        source="reports/training/gradient_conflict_stage3.md",
        found_by="agent Turing",
    ),
    CorpusLimitation(
        id="coupling_operator_trained_by_real_eeg_alone",
        artifact="scwbd-001-beta",
        measured=(
            "real EEG and the simulator are orthogonal on 7 of 8 shared modules "
            "(|mean cosine| <= 0.054, fraction-negative 0.42-0.46, indistinguishable from "
            "chance). The exception is `coupling`: mean cosine -0.259, minimum -0.999, 64% "
            "of observations negative, n=22. On escalation the policy adds `coupling.*` to "
            "the yielding source's frozen patterns and REBUILDS the GradientGate -- verified "
            "enforced, not decorative."
        ),
        consequence=(
            "'Trained on a mixture' is a materially inaccurate description of this artifact's "
            "largest module. For most of Stage III the SIMULATED CORPUS WAS FROZEN OUT OF "
            "THE COUPLING OPERATOR and real EEG trained it alone -- 4,946,799 parameters, "
            "the module carrying the connectome, from 189,765 real windows. This cuts BOTH "
            "WAYS and both belong in the record: FOR the artifact, the coupling operator was "
            "not shaped by the simulator's idiosyncrasies where the two disagreed; AGAINST "
            "it, a 4.9M-parameter module was then trained by real data alone at that scale."
        ),
        discloses_on=("G1", "G2", "A2_coupling_family", "A5_typed_operators"),
        source="reports/training/gradient_conflict_stage3.md",
        found_by="agent Turing",
    ),
    CorpusLimitation(
        id="no_source_ablation_so_D12_is_unsupported",
        artifact="scwbd-001-beta",
        measured=(
            "per_source_contribution is each source's share of the NORMALISED loss, and the "
            "simulated source's loss includes the NPE term, which is genuinely negative "
            "(npe_loss = -12.19 in the live log). sim_wholebrain's -0.185 is therefore "
            "ARITHMETIC, NOT HARM -- a negative loss share is not negative transfer. Agent "
            "Turing corrected their own first reading before filing."
        ),
        consequence=(
            "No claim in EITHER DIRECTION about whether a source family earns its place is "
            "currently supported. Real negative transfer requires source_ablation, which has "
            "NOT been run. D12 (dataset-family breadth) stays COULD_NOT_RUN, and a negative "
            "contribution number must not be quoted as evidence that a family hurts."
        ),
        blocks=("D12_dataset_family_breadth",),
        source="reports/training/gradient_conflict_stage3.md",
        found_by="agent Turing (self-corrected)",
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
