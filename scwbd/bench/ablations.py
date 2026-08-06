"""Required ablations from ``body.tex`` §11.4 (agent J).

Every bullet of §11.4 is a registry entry here.  Each ablation runs its arms
at **matched capacity and compute**, and reports, for every arm:

* **variance** — held-out calibrated log score with a bootstrap interval;
* **plausible systematic error** — worst-stratum bias across session, device,
  site, anatomy, demographic stratum and task context, plus any external
  bound, classified with the §2.7 ledger status;
* the **smoothing check** — §11.4's warning, executable: *"a lower variance
  model is not preferred when it achieves stability by smoothing away the
  effect of interest."*  If the arm that wins on raw score is the arm that
  destroyed the effect, the ablation reports ``FAIL`` and refuses the
  preference.

Two further §11.4 rules are enforced rather than described:

* "Topology is supported only if it improves data efficiency, calibration,
  out-of-distribution behaviour, or causal prediction — not merely because it
  reduces parameters."  Parameter reduction alone never satisfies an ablation
  here; it is recorded as a capacity fact, not as evidence.
* "A mechanistic module is supported only if removing or replacing it worsens
  a prediction uniquely associated with its mechanism."  Ablations that carry
  a mechanistic claim require a *mechanism-specific* prediction target and
  report ``COULD_NOT_RUN`` without one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .harness import Dataset, EvalResult, evaluate
from .matching import budget_of, check_matched, matched_subcheck
from .report import (
    BaselineResult,
    ClaimManifest,
    ClaimReport,
    Interval,
    Metric,
    SubCheck,
    could_not_run,
)
from .statistics import (
    SmoothingVerdict,
    bootstrap_ci,
    paired_bootstrap,
    smoothing_check,
    stratified_bias,
    systematic_error,
)
from .corpus import limitations_for
from .gates import Thresholds, _NON_GOALS, _corpus_subchecks

__all__ = [
    "AblationSpec",
    "ABLATIONS",
    "run_ablation",
    "run_all_ablations",
    "default_effect",
    "A1_EFFECT",
    "A1_RUN2_PREREGISTRATION",
]


def default_effect(y: np.ndarray) -> float:
    """Default 'effect of interest': the dynamic range of the signal.

    A model that buys stability by predicting the mean loses this, which is
    exactly the failure §11.4 warns about.  Ablations with a sharper effect
    (a condition contrast, a dose slope, a high-frequency band) should pass
    their own ``effect`` callable.
    """
    y = np.asarray(y, dtype=float)
    return float(np.std(y.reshape(y.shape[0], -1), axis=0).mean())


def A1_EFFECT(y: np.ndarray) -> float:
    """A1's effect of interest: **between-region differentiation of dynamics**.

    Preregistered in ``reports/ablations/PREREG_A1_run2.md`` §5.2 *before* any
    heterogeneous model existed.

    :func:`default_effect` measures *global dynamic range*, and a model that
    collapses every region onto one shared dynamic **preserves global dynamic
    range exactly** while destroying precisely the effect A1 is about
    ("structured regional state versus one scalar or pooled vector per region").
    Running A1 on the default therefore yields a smoothing check that is green
    and structurally incapable of reading A1's own failure mode -- the
    decorative-guard pattern, inside the machinery built to catch it.

    For ``y`` of shape ``(windows, T, channels)`` (or any ``(n, T, C)``):

    1. temporal std over ``T``, per window and per channel;
    2. std of that **across channels** -- how differentiated regions are;
    3. mean over windows.

    Scale-equivariant, label-free, and computed from arrays the harness already
    holds, so it cannot be dropped for cost.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 2:  # (n, T) -- a single channel cannot express differentiation
        return 0.0
    if y.ndim > 3:
        y = y.reshape(y.shape[0], y.shape[1], -1)
    per_channel = y.std(axis=1)  # (n, C): temporal std per channel
    return float(per_channel.std(axis=1).mean())


#: Machine-readable summary of ``reports/ablations/PREREG_A1_run2.md``, fixed on
#: ``wt/popper`` while ``A1_structured_state`` is COULD_NOT_RUN and no
#: heterogeneous arm exists.  Prose in a report can be edited; this is imported
#: by the scoring path and asserted by ``tests/bench``.
A1_RUN2_PREREGISTRATION = (
    "TWO_CAPACITY_MATCHED_POOLED_CONTROLS_param_matched_AND_state_matched_"
    "BOTH_MUST_BE_BEATEN; PERMUTED_FAMILY_ARM_MANDATORY_FOR_ATTRIBUTION; "
    "PRIMARY_PAIRED_PARTICIPANT_CLUSTERED_NLL_PLUS_COPRIMARY_MSE_BOTH_INTERVALLED; "
    "NLL_WIN_WITHOUT_MSE_WIN_GRANTS_NO_MECHANISTIC_CLAIM; "
    "EFFECT_IS_A1_EFFECT_BETWEEN_REGION_DISPERSION_NOT_default_effect; "
    "SCORED_AS_EMITTED_AND_CALIBRATION_MATCHED_DISAGREEMENT_CLAIMS_NEITHER; "
    "SYSTEMATIC_ENVELOPE_GE_DELTA_OR_SEED_RANGE_GE_DELTA_IS_INCONCLUSIVE; "
    "V_ABLATION_AND_V_CLAIM_ARE_SEPARATE_11_2_FLOOR_BOUNDS_ONLY_V_CLAIM; "
    "RUN1_IS_A_CONTROL_CLASS_ARTIFACT_NOT_RUN2S_CONTROL_ARM"
)


@dataclass(frozen=True)
class AblationSpec:
    """One §11.4 bullet, as an executable comparison."""

    id: str
    thesis_clause: str
    question: str
    #: arm names that MUST be supplied; a missing arm is COULD_NOT_RUN
    required_arms: tuple[str, ...]
    #: the arm whose feature is under test (the "candidate")
    candidate_arm: str
    #: what happens if the candidate does not beat its controls
    consequence: str
    #: does this ablation assert a mechanism (§11.4 last paragraph)?
    mechanistic_claim: bool = False
    optional_arms: tuple[str, ...] = ()
    #: quarantined ablations are off by default (ARCHITECTURE.md rule 5)
    quarantined: bool = False
    note: str = ""
    #: An ablation whose failure mode is invisible to :func:`default_effect`
    #: names its own effect here.  :func:`run_ablation` then REFUSES to run with
    #: any other callable rather than reporting a smoothing check that cannot
    #: read the failure it exists to catch.
    required_effect: Callable[[np.ndarray], float] | None = None


ABLATIONS: dict[str, AblationSpec] = {
    "A1_structured_state": AblationSpec(
        id="A1_structured_state",
        thesis_clause="structured regional state versus one scalar or pooled vector per region",
        question="Does structured regional state predict anything a pooled scalar cannot?",
        required_arms=("structured_state", "scalar_per_region", "pooled_vector_per_region"),
        candidate_arm="structured_state",
        consequence=(
            "Collapse regional state to the supported dimensionality and stop describing "
            "regions as structured state spaces (body.tex §2.1) for the affected systems."
        ),
        mechanistic_claim=True,
        # Run 2 adds two capacity-matched pooled controls (parameter-matched and
        # state-matched -- they cannot both be satisfied at once, so BOTH are
        # required and the choice is not made after the fact) plus a
        # permuted-family arm that holds heterogeneity fixed and destroys the
        # anatomical assignment.  See reports/ablations/PREREG_A1_run2.md §1, §3.
        optional_arms=(
            "pooled_vector_per_region@param_matched",
            "pooled_vector_per_region@state_matched",
            "permuted_family_state",
        ),
        required_effect=A1_EFFECT,
        note=A1_RUN2_PREREGISTRATION,
    ),
    "A2_coupling_family": AblationSpec(
        id="A2_coupling_family",
        thesis_clause=(
            "hybrid local field plus sparse long-range graph versus fully dense, graph-only, "
            "and uniformly convolutional models"
        ),
        question="Does the local-field + sparse-graph hybrid beat its three alternatives?",
        required_arms=("hybrid_field_plus_sparse_graph", "dense", "graph_only",
                       "uniformly_convolutional"),
        candidate_arm="hybrid_field_plus_sparse_graph",
        consequence=(
            "Drop the hybrid claim and use whichever single coupling family actually wins; "
            "the hybrid's extra machinery is then unjustified complexity."
        ),
        mechanistic_claim=True,
    ),
    "A3_resolution": AblationSpec(
        id="A3_resolution",
        thesis_clause=(
            "single-resolution versus simultaneous fine/coarse cortical pyramids, arbitrary "
            "source-native resolution lattices, and sparse adaptive refinement, including "
            "scale- and parameter-matched controls"
        ),
        question="Does multiresolution machinery beat scale- and parameter-matched controls?",
        required_arms=("simultaneous_pyramid", "single_resolution_fine",
                       "single_resolution_coarse", "sparse_adaptive_refinement"),
        candidate_arm="simultaneous_pyramid",
        optional_arms=("source_native_lattice",),
        consequence=(
            "Disable the scale relation and serve source-specific views without gluing "
            "(same consequence as gate G3)."
        ),
    ),
    "A4_topology": AblationSpec(
        id="A4_topology",
        thesis_clause="hard, soft, learned, randomized, and distance-matched topology",
        question="Which topology treatment is actually supported by held-out behaviour?",
        required_arms=("hard", "soft", "learned", "randomized", "distance_matched"),
        candidate_arm="hard",
        optional_arms=("dense", "local_only"),
        consequence=(
            "Demote anatomy from compiled constraint to weak prior for the affected scale, "
            "and adopt whichever treatment (soft/learned) the data supports."
        ),
    ),
    "A5_typed_operators": AblationSpec(
        id="A5_typed_operators",
        thesis_clause="anatomically typed operators versus an equal-parameter generic operator",
        question="Does operator typing earn its mechanistic label?",
        required_arms=("typed_operators", "generic_equal_parameter"),
        candidate_arm="typed_operators",
        consequence=(
            "Reclassify the typed operators as effective/functional or surrogate "
            "(OperatorSpec.mechanistic_status) — a mechanistic label is earned only by "
            "predictions a generic surrogate misses."
        ),
        mechanistic_claim=True,
    ),
    "A6_pretraining": AblationSpec(
        id="A6_pretraining",
        thesis_clause="region-specific pretraining versus end-to-end blank-slate training",
        question="Does regional phenotype pretraining help beyond blank-slate training?",
        required_arms=("region_specific_pretraining", "end_to_end_blank_slate"),
        candidate_arm="region_specific_pretraining",
        consequence=(
            "Remove the staged regional pretraining claim (body.tex §6.1) and report the "
            "model as end-to-end trained."
        ),
    ),
    "A7_individualization": AblationSpec(
        id="A7_individualization",
        thesis_clause="population, session-adapted, and longitudinal subject models",
        question="Which level of adaptation is supported on future data?",
        required_arms=("longitudinal_subject", "population", "session_adapted"),
        candidate_arm="longitudinal_subject",
        optional_arms=("anatomy_only",),
        consequence=(
            "Retain only the supported level of adaptation and never use the phrase "
            "'individual digital twin' (same consequence as gate G5)."
        ),
    ),
    "A8_language_coupling": AblationSpec(
        id="A8_language_coupling",
        thesis_clause=(
            "language-only behavioural imitation versus a language process coupled to neural, "
            "bodily, memory, and action predictions"
        ),
        question="Does coupling language to neural/bodily/memory/action predictions help?",
        required_arms=("coupled_language_process", "language_only_imitation"),
        candidate_arm="coupled_language_process",
        consequence=(
            "Report the language channel as stylistic imitation only; it is not evidence of "
            "causal fidelity, consciousness, or personal identity (Appendix D)."
        ),
    ),
    "A9_teacher_quarantined": AblationSpec(
        id="A9_teacher_quarantined",
        thesis_clause=(
            "when the quarantined report/teacher experiment is enabled: no teacher, matched "
            "generic features and smoothness, shuffled/mismatched report, and "
            "perception-versus-imagery domain-shift controls"
        ),
        question="Does the teacher/distillation term improve *measured* held-out prediction?",
        required_arms=("with_teacher", "no_teacher", "matched_generic_features",
                       "shuffled_report", "domain_shift_perception_vs_imagery"),
        candidate_arm="with_teacher",
        consequence=(
            "Keep TRIBE v2 distillation off (ARCHITECTURE.md rule 5) and remove any claim "
            "that the teacher contributes to empirical prediction."
        ),
        quarantined=True,
        note=(
            "Quarantined: off by default and never a subject likelihood. Teacher agreement "
            "alone is never the metric (Appendix D, 'Teacher/simulator domination')."
        ),
    ),
    "A10_correlation_vs_perturbation": AblationSpec(
        id="A10_correlation_vs_perturbation",
        thesis_clause="correlation fitting versus held-out perturbational prediction",
        question="Does a model fitted to passive correlation predict held-out perturbations?",
        required_arms=("perturbation_aware", "correlation_fitted"),
        candidate_arm="perturbation_aware",
        consequence=(
            "Label the affected operators functional/statistical rather than effective or "
            "causal (compiler refusal R04), and stop reporting causal estimates from them."
        ),
        mechanistic_claim=True,
    ),
}


# --------------------------------------------------------------------------
def _arm_result(name: str, factory: Any, train: Dataset, test: Dataset, *, seed: int,
                refuse_overlap: bool) -> EvalResult:
    model = factory() if callable(factory) and not hasattr(factory, "predict") else factory
    res = evaluate(model, train, test, seed=seed, refuse_group_overlap=refuse_overlap)
    out = EvalResult(**{**res.__dict__, "arm": name})
    out.extras["_model"] = model
    return out


def run_ablation(
    spec: AblationSpec | str,
    *,
    train: Dataset | None = None,
    test: Dataset | None = None,
    arms: Mapping[str, Any] | None = None,
    effect: Callable[[np.ndarray], float] | None = None,
    mechanism_holdout: Mapping[str, Dataset] | None = None,
    external_bias_bounds: Mapping[str, tuple[float, float]] | None = None,
    retention_floor: float = 0.5,
    artifact: str | None = None,
    thresholds: Thresholds = Thresholds(),
    seed: int = 0,
    refuse_group_overlap: bool = True,
    enable_quarantined: bool = False,
) -> ClaimReport:
    """Run one §11.4 ablation and emit a :class:`ClaimReport`."""
    if isinstance(spec, str):
        spec = ABLATIONS[spec]
    thr = thresholds
    arms = dict(arms or {})

    man = ClaimManifest(
        claim_id=spec.id,
        claim_text=f"{spec.question} (§11.4: {spec.thesis_clause})",
        falsified_by=(
            "an equal-capacity, equal-compute control matches or exceeds the candidate, or "
            "the candidate wins only by smoothing away the effect of interest"
        ),
        consequence_if_failed=spec.consequence,
        thesis_reference="body.tex §11.4",
        baselines=[a for a in spec.required_arms if a != spec.candidate_arm],
        acceptance_thresholds={
            **thr.as_dict(),
            "effect_retention_floor": retention_floor,
            "mechanistic_claim": spec.mechanistic_claim,
        },
        non_goals=list(_NON_GOALS),
        seed=seed,
    )
    subs: list[SubCheck] = []
    artifacts: dict[str, Any] = {"thesis_clause": spec.thesis_clause}
    subs.extend(_corpus_subchecks(spec.id, artifact, artifacts))

    if spec.quarantined and not enable_quarantined:
        subs.append(
            could_not_run(
                "quarantine",
                "Quarantined experiment gate.",
                "this ablation belongs to a quarantined experiment which is OFF by default "
                "(ARCHITECTURE.md rule 5: TRIBE v2 distillation stays off by default and is "
                "never a subject likelihood); pass enable_quarantined=True only under an "
                "explicit claim-manifest override",
                falsified_by=man.falsified_by,
            )
        )
        return ClaimReport(manifest=man, subchecks=subs, kind="ablation",
                           notes=[spec.note]).finalize()

    # -- the effect of interest must be the one this ablation names -----
    # A smoothing check reading the WRONG effect is worse than none: it is
    # green by construction.  A1's failure mode -- every region collapsed onto
    # one shared dynamic -- preserves `default_effect` EXACTLY, so A1 declares
    # `required_effect` and an explicitly-supplied different callable is
    # refused.  Supplying nothing takes the spec's own effect, so the wrong one
    # cannot be reached by omission either.
    wrong_effect = (
        spec.required_effect is not None
        and effect is not None
        and effect is not spec.required_effect
    )
    if effect is None:
        effect = spec.required_effect or default_effect

    missing = [a for a in spec.required_arms if a not in arms]
    if train is None or test is None:
        missing.append("train/test datasets")
    if missing or wrong_effect:
        # BOTH are reported.  Returning on the effect alone would hide which
        # arms are absent, which is the more actionable fact while A1 has none.
        for i, m in enumerate(missing):
            subs.append(
                could_not_run(
                    f"arms[{i}]", "Required ablation arm or dataset",
                    f"missing: {m}; §11.4 names it explicitly, so the comparison cannot be "
                    "declared complete without it",
                    falsified_by=man.falsified_by,
                )
            )
        if wrong_effect:
            assert spec.required_effect is not None
            subs.append(
                could_not_run(
                    "effect_of_interest",
                    "The §11.4 smoothing check must read THIS ablation's effect.",
                    f"{spec.id} requires effect={spec.required_effect.__name__}; got "
                    f"{getattr(effect, '__name__', repr(effect))}. "
                    f"{spec.required_effect.__name__} measures the quantity whose loss IS "
                    "the failure mode; default_effect is preserved exactly by that failure, "
                    "so the smoothing check would be structurally incapable of firing",
                    falsified_by="the winning arm won by attenuating the effect of interest",
                )
            )
        return ClaimReport(manifest=man, subchecks=subs, kind="ablation",
                           notes=[spec.note] if spec.note else []).finalize()

    assert train is not None and test is not None
    run_arms = {k: v for k, v in arms.items()
                if k in spec.required_arms or k in spec.optional_arms}
    results = {k: _arm_result(k, v, train, test, seed=seed, refuse_overlap=refuse_group_overlap)
               for k, v in run_arms.items()}
    cand = results[spec.candidate_arm]
    controls = {k: v for k, v in results.items() if k != spec.candidate_arm}

    # -- capacity / compute matching -----------------------------------
    verdict = check_matched(
        cand.extras.get("_model"), {k: v.extras.get("_model") for k, v in controls.items()},
        tol=thr.capacity_tol, candidate_name=spec.candidate_arm,
    )
    subs.append(matched_subcheck(verdict))
    artifacts["capacity"] = {
        "n_parameters": {k: r.n_parameters for k, r in results.items()},
        "ratios": verdict.ratios,
    }

    # -- variance: held-out score per arm ------------------------------
    per_arm: dict[str, dict[str, Any]] = {}
    score_metrics: list[Metric] = []
    for k, r in results.items():
        pt, iv = bootstrap_ci(r.log_score, seed=seed, n_boot=thr.n_boot)
        per_arm[k] = {
            "log_score": pt,
            "ci": [iv.lo, iv.hi],
            "coverage_error": r.calibration.coverage_error,
            "overconfidence": r.calibration.overconfidence,
            "sharpness": r.calibration.sharpness,
            "n_parameters": r.n_parameters,
            "rmse": r.rmse,
        }
        score_metrics.append(
            Metric(name=f"{k}.heldout_log_score", value=pt, units="nats/obs",
                   kind="accuracy", interval=iv, direction="greater_is_better")
        )
        score_metrics.append(
            Metric(name=f"{k}.coverage_error", value=r.calibration.coverage_error,
                   kind="calibration", interval=r.calibration.coverage_error_interval,
                   threshold=thr.max_coverage_error, direction="less_is_better")
        )
    artifacts["per_arm"] = per_arm
    subs.append(
        SubCheck(
            name="variance_reported",
            description="Held-out calibrated log score and coverage per arm, with intervals.",
            metrics=score_metrics,
            mandatory=True,
            falsified_by="an arm's calibration is outside the preregistered tolerance",
        )
    )

    # -- candidate must beat every control -----------------------------
    delta_metrics: list[Metric] = []
    deltas: dict[str, Any] = {}
    for k, r in controls.items():
        d = paired_bootstrap(cand.log_score, r.log_score,
                             name=f"delta_log_score.{spec.candidate_arm}_vs_{k}",
                             n_boot=thr.n_boot, seed=seed)
        delta_metrics.append(d.metric(threshold=thr.min_delta_log_score))
        deltas[k] = {"mean": d.mean, "ci": [d.interval.lo, d.interval.hi],
                     "p": d.p_two_sided, "indistinguishable": d.indistinguishable}
    delta_metrics += cand.calibration.metrics(prefix=spec.candidate_arm)[:1]
    artifacts["deltas"] = deltas
    subs.append(
        SubCheck(
            name="candidate_beats_controls",
            description=(
                f"{spec.candidate_arm} versus every control at matched capacity. Parameter "
                "reduction alone is not evidence (§11.4)."
            ),
            metrics=delta_metrics,
            mandatory=True,
            falsified_by="an equal-capacity control matches or exceeds the candidate",
        )
    )

    # -- plausible systematic error ------------------------------------
    if not test.strata and not external_bias_bounds:
        subs.append(
            could_not_run(
                "systematic_error_reported",
                "Plausible systematic error alongside the variance (§11.4).",
                "the evaluation set declares no strata (session/device/site/anatomy/"
                "demographic/task) and no external bound was supplied, so systematic error "
                "is prior-specified sensitivity only and cannot be advertised as estimated",
                falsified_by="systematic error dominates the reported difference",
            )
        )
    else:
        bias_metrics: list[Metric] = []
        sys_table: dict[str, Any] = {}
        for k, r in results.items():
            # strata are per observation; multivariate targets are reduced to
            # one residual per observation so the two always align
            y_obs = r.targets.reshape(test.n, -1).mean(axis=1)
            p_obs = r.prediction.mean.reshape(test.n, -1).mean(axis=1)
            ba = stratified_bias(y_obs, p_obs, test.strata, seed=seed,
                                 n_boot=min(thr.n_boot, 400))
            mag, status, detail = systematic_error(
                ba, external_bounds=external_bias_bounds,
                model_discrepancy=r.extras.get("model_discrepancy"),
            )
            sys_table[k] = {"magnitude": mag, "status": status, **detail,
                            "table": ba.table()}
            bias_metrics.append(
                Metric(
                    name=f"{k}.systematic_error",
                    value=mag,
                    kind="systematic",
                    exact=True,
                    direction="less_is_better",
                    note=f"ledger status: {status}",
                )
            )
        artifacts["systematic_error"] = sys_table
        worst_arm = max(sys_table, key=lambda k: sys_table[k]["magnitude"])
        estimable = [k for k, v in sys_table.items()
                     if v["status"] in ("design_estimable", "externally_bounded")]
        bias_metrics.append(
            Metric(
                name="systematic_error.estimable_fraction",
                value=float(len(estimable)) / max(len(sys_table), 1),
                kind="systematic",
                exact=True,
                threshold=0.999,
                direction="greater_is_better",
                note=(
                    f"largest worst-stratum bias is {worst_arm}; systematic error and "
                    "log-score differences are in different units and are reported side by "
                    "side, never combined into one score (thesis §2.7)"
                ),
            )
        )
        # Does the candidate's advantage survive in its own worst stratum?  An
        # advantage that exists only outside the worst subgroup is a stratum
        # artefact, not a modelling result.
        cand_worst = None
        tbl = sys_table[spec.candidate_arm]["table"]
        if tbl:
            row = max(tbl, key=lambda r: abs(r["bias"]))
            lab = np.asarray(test.strata[row["factor"]])
            mask = lab == row["level"]
            best_ctrl = max(controls, key=lambda k: results[k].mean_log_score) if controls else None
            if best_ctrl is not None and mask.sum() >= 10:
                dw = paired_bootstrap(
                    cand.log_score[mask], results[best_ctrl].log_score[mask],
                    name="systematic_error.worst_stratum_delta_vs_best_control",
                    n_boot=min(thr.n_boot, 400), seed=seed,
                )
                cand_worst = {"factor": row["factor"], "level": row["level"],
                              "delta": dw.mean, "ci": [dw.interval.lo, dw.interval.hi]}
                bias_metrics.append(
                    Metric(
                        name="systematic_error.worst_stratum_delta_vs_best_control",
                        value=dw.mean, units="nats/obs", kind="systematic",
                        interval=dw.interval, threshold=-0.05,
                        direction="greater_is_better",
                        note=(f"worst stratum for the candidate: {row['factor']}="
                              f"{row['level']} (n={row['n']}); an advantage that vanishes "
                              "here is a stratum artefact"),
                    )
                )
        artifacts["candidate_worst_stratum"] = cand_worst
        subs.append(
            SubCheck(
                name="systematic_error_reported",
                description="Worst-stratum bias per arm with its uncertainty-ledger status.",
                metrics=bias_metrics,
                mandatory=True,
                falsified_by="the candidate's systematic error is larger than its advantage",
            )
        )

    # -- the smoothing check -------------------------------------------
    ranked = sorted(results.items(), key=lambda kv: -kv[1].mean_log_score)
    top_name, top_res = ranked[0]
    verdicts: dict[str, Any] = {}
    smoothing_metrics: list[Metric] = []
    top_verdict: SmoothingVerdict | None = None
    for k, r in results.items():
        # each arm is compared against the *other* best-scoring arm, so
        # "this one is more stable" is measured against a real alternative
        ref_name = next((n for n, _ in ranked if n != k), k)
        ref = results[ref_name]
        sv = smoothing_check(
            arm_name=k,
            reference_name=ref.arm,
            y_true=test.targets,
            pred_arm=r.prediction.mean.reshape(test.targets.shape),
            pred_reference=ref.prediction.mean.reshape(test.targets.shape),
            effect=effect,
            retention_floor=retention_floor,
            seed=seed,
            n_boot=min(thr.n_boot, 400),
        )
        verdicts[k] = {
            "effect_retention": sv.effect_retention,
            "variance_ratio": sv.variance_ratio,
            "lower_variance": sv.lower_variance,
            "smoothed_away": sv.smoothed_away,
            "verdict": sv.verdict,
        }
        if k == top_name:
            smoothing_metrics = sv.metrics(prefix=f"smoothing.{k}")
            top_verdict = sv
    artifacts["smoothing"] = verdicts
    subs.append(
        SubCheck(
            name="smoothing_not_preferred",
            description=(
                "§11.4: a lower-variance model is not preferred when it achieves stability by "
                f"smoothing away the effect of interest. Checked on the top-scoring arm "
                f"({top_name})."
            ),
            metrics=smoothing_metrics,
            mandatory=True,
            reason=top_verdict.verdict if top_verdict is not None else "",
            falsified_by="the winning arm won by attenuating the effect of interest",
        )
    )

    # -- mechanism claim ------------------------------------------------
    if spec.mechanistic_claim:
        if not mechanism_holdout or "train" not in mechanism_holdout or \
                "test" not in mechanism_holdout:
            subs.append(
                could_not_run(
                    "mechanism_uniquely_supported",
                    "Prediction uniquely associated with the claimed mechanism.",
                    "no mechanism-specific holdout supplied (§11.4: 'a mechanistic module is "
                    "supported only if removing or replacing it worsens a prediction uniquely "
                    "associated with its mechanism'); typically a held-out perturbation with "
                    "direction/timing/dose structure",
                    falsified_by="a generic equal-capacity surrogate matches the mechanism",
                )
            )
        else:
            mres = {k: _arm_result(k, v, mechanism_holdout["train"], mechanism_holdout["test"],
                                   seed=seed, refuse_overlap=refuse_group_overlap)
                    for k, v in run_arms.items()}
            mm: list[Metric] = []
            for k, r in mres.items():
                if k == spec.candidate_arm:
                    continue
                d = paired_bootstrap(mres[spec.candidate_arm].log_score, r.log_score,
                                     name=f"mechanism.delta_vs_{k}", n_boot=thr.n_boot, seed=seed)
                mm.append(d.metric(threshold=thr.min_delta_log_score))
            mm += mres[spec.candidate_arm].calibration.metrics(prefix="mechanism")[:1]
            subs.append(
                SubCheck(
                    name="mechanism_uniquely_supported",
                    description=(
                        "On the mechanism-specific holdout, removing/replacing the module must "
                        "cost prediction that no equal-capacity control recovers."
                    ),
                    metrics=mm,
                    mandatory=True,
                    falsified_by="a generic equal-capacity surrogate matches the mechanism",
                )
            )
            artifacts["mechanism_holdout"] = {
                k: float(np.mean(r.log_score)) for k, r in mres.items()
            }

    rows = [
        BaselineResult(
            name=k, role="§11.4 control",
            n_parameters=r.n_parameters,
            compute_flops=budget_of(r.extras.get("_model")).flops,
            metrics=[Metric(name=f"{k}.heldout_log_score", value=per_arm[k]["log_score"],
                            units="nats/obs", kind="diagnostic", exact=True)],
        )
        for k, r in controls.items()
    ]
    notes = [
        "Variance and plausible systematic error are reported side by side and never "
        "combined into one score (thesis §2.7).",
        "Parameter reduction alone is never counted as evidence for a structural choice.",
    ]
    if spec.note:
        notes.append(spec.note)
    return ClaimReport(manifest=man, subchecks=subs, baselines_run=rows, artifacts=artifacts,
                       kind="ablation", notes=notes).finalize()


def run_all_ablations(config: Mapping[str, Mapping[str, Any]] | None = None,
                      *, seed: int = 0) -> list[ClaimReport]:
    """Run every §11.4 ablation with whatever arms are available."""
    cfg = dict(config or {})
    out: list[ClaimReport] = []
    for key, spec in ABLATIONS.items():
        out.append(run_ablation(spec, seed=seed, **dict(cfg.get(key, {}))))
    return out
