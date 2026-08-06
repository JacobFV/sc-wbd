"""Claim gates G1--G5 as executable checks (agent J).

Specification: ``ARCHITECTURE.md`` §4 and ``thesis_contract.tex`` Table
``tab:claim-gates``.  Each gate implements **the falsification column**, not
the support column: the sub-checks are written so that the ordinary outcome of
a model that does not work is ``FAIL``, and the ordinary outcome of a missing
dependency is ``COULD_NOT_RUN``.

Standing rules, enforced in code and not merely documented:

* every gate runs its baselines; a gate with no baselines cannot pass
  (:meth:`ClaimReport.finalize`);
* every comparison is capacity-matched; an unmatched win is not a win
  (:mod:`scwbd.bench.matching`);
* every thesis-named falsifier is a **mandatory** sub-check.  If the evidence
  needed to try to falsify the claim is absent, the gate reports
  ``COULD_NOT_RUN`` — the claim is then unsupported, not supported;
* a failing gate carries the implementation consequence from the thesis table
  verbatim.

A gate that fails is a result. Do not tune until it passes; do not delete it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import adapters
from .harness import Arm, Dataset, EvalResult, as_factory, evaluate
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
    bootstrap_ci,
    calibration,
    data_efficiency_curve,
    decision_regret,
    paired_bootstrap,
    selection_optimism,
    stratified_bias,
)

__all__ = [
    "Thresholds",
    "CLAIMS",
    "run_g1",
    "run_g2",
    "run_g3",
    "run_g4",
    "run_g5",
    "run_all_gates",
]


# --------------------------------------------------------------------------
# preregistered thresholds
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Thresholds:
    """Preregistered acceptance thresholds.  Changing these changes the claim."""

    #: paired delta log score must have its whole interval above this
    min_delta_log_score: float = 0.0
    #: mean |empirical - nominal| coverage, in the deployment population
    max_coverage_error: float = 0.05
    #: candidate may not be more overconfident than the best baseline by more
    max_overconfidence_increase: float = 0.02
    #: relative delay recovery error
    max_delay_rel_error: float = 0.15
    #: boundary agreement between fine and coarse backends (relative)
    boundary_rel_tol: float = 0.05
    #: how much more fine-scale energy a model may emit, with the fine evidence
    #: withheld, than a coarse-only model given the same evidence, before it is
    #: called hallucination (1.0 = exactly as much as the evidence supports)
    max_hallucination_index: float = 1.25
    #: predictive sd must inflate by at least this factor when evidence is withheld
    min_uncertainty_inflation: float = 1.05
    #: Fisher min-eigenvalue (theta block, nuisance profiled out) gain ratio
    min_fisher_eig_gain: float = 1.10
    #: parameter-count matching tolerance
    capacity_tol: float = 0.10
    #: minimum |log evidence| separation added by intervention between models
    min_model_discrimination: float = 0.05
    n_boot: int = 1000

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# --------------------------------------------------------------------------
# claim texts, falsifiers and consequences -- verbatim from tab:claim-gates
# --------------------------------------------------------------------------
CLAIMS: dict[str, dict[str, str]] = {
    "G1": {
        "claim": "Typed, source-native fusion is preferable to naive resampling.",
        "support": (
            "Better held-out likelihood, calibration, delay recovery, and intervention "
            "forecast at matched compute and parameter count."
        ),
        "falsified_by": (
            "No reproducible gain over single-modality or carefully tuned resampling "
            "baselines; increased overconfidence or negative transfer."
        ),
        "consequence": (
            "Retain only the provenance/type system; narrow or remove the shared latent "
            "fusion claim."
        ),
        "reference": "thesis_contract.tex tab:claim-gates row 1; ARCHITECTURE.md §4 G1",
    },
    "G2": {
        "claim": "Anatomical topology improves inference.",
        "support": (
            "Data efficiency, out-of-distribution calibration, and causal forecast beyond "
            "dense, randomized, and distance-matched graphs."
        ),
        "falsified_by": (
            "Equal-capacity controls match or exceed performance, or topology errors are "
            "absorbed by residuals."
        ),
        "consequence": (
            "Demote anatomy from compiled constraint to weak prior for the affected scale."
        ),
        "reference": "thesis_contract.tex tab:claim-gates row 2; ARCHITECTURE.md §4 G2",
    },
    "G3": {
        "claim": "Multiresolution state adds information rather than decoration.",
        "support": (
            "Native-scale prediction, calibrated refinement, boundary agreement, and compute "
            "savings without hidden high-frequency hallucination."
        ),
        "falsified_by": (
            "Fine views cannot improve supported observables, fail round-trip tests, or "
            "become overconfident outside measured tiles."
        ),
        "consequence": (
            "Disable the scale relation or use source-specific views without gluing."
        ),
        "reference": "thesis_contract.tex tab:claim-gates row 3; ARCHITECTURE.md §4 G3",
    },
    "G4": {
        "claim": "Perturbation reduces non-identifiability.",
        "support": (
            "Rank/eigenvalue improvement in Fisher information and prospective recovery of "
            "direction, delay, gain, dose, and state dependence."
        ),
        "falsified_by": (
            "Intervention fails to distinguish posterior models or adds only field-model "
            "uncertainty."
        ),
        "consequence": (
            "Narrow the identifiable parameter set and redesign the perturbation rather than "
            "reporting a causal estimate."
        ),
        "reference": "thesis_contract.tex tab:claim-gates row 4; ARCHITECTURE.md §4 G4",
    },
    "G5": {
        "claim": "Individualization improves future prediction.",
        "support": (
            "Incremental calibrated log score and decision utility on new sessions and unseen "
            "tasks or interventions."
        ),
        "falsified_by": (
            "Anatomy-only, population, or session-adapted baselines perform equivalently."
        ),
        "consequence": (
            "Do not label the model an individual twin; retain only the supported level of "
            "adaptation."
        ),
        "reference": "thesis_contract.tex tab:claim-gates row 5; ARCHITECTURE.md §4 G5",
    },
}

_NON_GOALS = [
    "This gate does not claim a validated digital twin of any specific person.",
    "This gate does not claim that any admitted operator is neurally realized.",
    "No prospective human TMS/tFUS protocol is implemented or implied (build order stops at "
    "item 5; item 6 is out of scope: no IRB, no consent, no participants).",
]


def _manifest(gate: str, *, seed: int, thresholds: Thresholds,
              baselines: Sequence[str], source_cards: Sequence[str] = (),
              refusal_fixtures: Sequence[str] = ()) -> ClaimManifest:
    c = CLAIMS[gate]
    return ClaimManifest(
        claim_id=gate,
        claim_text=c["claim"],
        falsified_by=c["falsified_by"],
        consequence_if_failed=c["consequence"],
        thesis_reference=c["reference"],
        permitted_source_cards=list(source_cards),
        baselines=list(baselines),
        acceptance_thresholds=thresholds.as_dict(),
        refusal_fixtures=list(refusal_fixtures),
        non_goals=list(_NON_GOALS),
        seed=seed,
    )


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def _fit_eval(arm: Any, train: Dataset, test: Dataset, *, seed: int,
              refuse_overlap: bool = True) -> EvalResult:
    return evaluate(arm, train, test, seed=seed, refuse_group_overlap=refuse_overlap)


def _eval_arms(arms: Mapping[str, Any], train: Dataset, test: Dataset, *, seed: int,
               refuse_overlap: bool = True) -> dict[str, EvalResult]:
    out: dict[str, EvalResult] = {}
    for name, a in arms.items():
        model = a() if callable(a) and not hasattr(a, "predict") else a
        if not hasattr(model, "name"):
            try:
                model.name = name  # type: ignore[attr-defined]
            except Exception:
                pass
        res = _fit_eval(model, train, test, seed=seed, refuse_overlap=refuse_overlap)
        out[name] = EvalResult(**{**res.__dict__, "arm": name})
        out[name].extras.setdefault("_model", model)
    return out


def _baseline_rows(results: Mapping[str, EvalResult], roles: Mapping[str, str],
                   *, seed: int) -> list[BaselineResult]:
    rows: list[BaselineResult] = []
    for name, r in results.items():
        b = budget_of(r.extras.get("_model"))
        rows.append(
            BaselineResult(
                name=name,
                role=roles.get(name, "baseline"),
                n_parameters=r.n_parameters if r.n_parameters is not None else b.n_parameters,
                compute_flops=b.flops,
                train_steps=b.train_steps,
                metrics=[r.log_score_metric(seed=seed)] + r.calibration.metrics(prefix=name)[:1],
            )
        )
    return rows


def _delta_subcheck(
    name: str,
    description: str,
    candidate: EvalResult,
    baselines: Mapping[str, EvalResult],
    thr: Thresholds,
    *,
    seed: int,
    falsified_by: str,
    label: str = "delta_log_score",
) -> tuple[SubCheck, dict[str, Any]]:
    """Paired log-score deltas against every baseline; all must be wins."""
    metrics: list[Metric] = []
    detail: dict[str, Any] = {}
    for bname, b in baselines.items():
        if b.log_score.shape != candidate.log_score.shape:
            return (
                could_not_run(
                    name,
                    description,
                    f"baseline {bname!r} scored {b.log_score.shape} points, candidate "
                    f"{candidate.log_score.shape}; the comparison is not paired",
                    falsified_by=falsified_by,
                ),
                detail,
            )
        d = paired_bootstrap(
            candidate.log_score, b.log_score,
            name=f"{label}_vs_{bname}", n_boot=thr.n_boot, seed=seed,
        )
        metrics.append(d.metric(threshold=thr.min_delta_log_score))
        detail[bname] = {"mean": d.mean, "ci": [d.interval.lo, d.interval.hi],
                         "p": d.p_two_sided}
    # the accuracy metrics above need a calibration companion in the report
    metrics += candidate.calibration.metrics(prefix=f"{candidate.arm}")
    return (
        SubCheck(
            name=name,
            description=description,
            metrics=metrics,
            mandatory=True,
            falsified_by=falsified_by,
        ),
        detail,
    )


def _calibration_subcheck(
    candidate: EvalResult,
    baselines: Mapping[str, EvalResult],
    thr: Thresholds,
    *,
    name: str = "calibration_not_degraded",
) -> SubCheck:
    best_base_over = min((b.calibration.overconfidence for b in baselines.values()),
                         default=float("inf"))
    over = candidate.calibration.overconfidence
    metrics = [
        Metric(
            name="calibration.coverage_error",
            value=candidate.calibration.coverage_error,
            kind="calibration",
            interval=candidate.calibration.coverage_error_interval,
            threshold=thr.max_coverage_error,
            direction="less_is_better",
        ),
        Metric(
            name="calibration.overconfidence_increase_vs_best_baseline",
            value=float(over - best_base_over) if math.isfinite(best_base_over) else float("nan"),
            kind="calibration",
            interval=Interval(
                candidate.calibration.overconfidence_interval.lo - best_base_over,
                candidate.calibration.overconfidence_interval.hi - best_base_over,
            ) if math.isfinite(best_base_over) else None,
            exact=not math.isfinite(best_base_over),
            threshold=thr.max_overconfidence_increase,
            direction="less_is_better",
            note="thesis falsifier: 'increased overconfidence'",
        ),
    ]
    return SubCheck(
        name=name,
        description=(
            "Calibration in the intended deployment population; aggregate accuracy may not "
            "be reported without it (§11.2)."
        ),
        metrics=metrics,
        mandatory=True,
        falsified_by="increased overconfidence relative to the baselines",
    )


def _negative_transfer_subcheck(
    candidate: EvalResult,
    reference: EvalResult,
    test: Dataset,
    thr: Thresholds,
    *,
    seed: int,
) -> SubCheck:
    """Per-stratum check that fusion never *hurts* relative to the best single view."""
    if not test.strata:
        return could_not_run(
            "no_negative_transfer",
            "Per-stratum check that fusion never degrades a subgroup.",
            "the evaluation set declares no strata (site/device/session/task); negative "
            "transfer is not detectable without them (§11.2 requires the bias analysis)",
            falsified_by="negative transfer in any stratum",
        )
    metrics: list[Metric] = []
    worst = ("", 0.0, None)
    for factor, labels in test.strata.items():
        for level in sorted(set(np.asarray(labels).tolist()), key=str):
            m = np.asarray(labels) == level
            if m.sum() < 20:
                continue
            d = paired_bootstrap(
                candidate.log_score[m], reference.log_score[m],
                name=f"negative_transfer.{factor}={level}", n_boot=thr.n_boot, seed=seed,
            )
            if worst[2] is None or d.mean < worst[1]:
                worst = (f"{factor}={level}", d.mean, d.interval)
    if worst[2] is None:
        return could_not_run(
            "no_negative_transfer",
            "Per-stratum check that fusion never degrades a subgroup.",
            "no stratum has >=20 held-out observations; the subgroup comparison is not "
            "evaluable at this sample size",
            falsified_by="negative transfer in any stratum",
        )
    metrics.append(
        Metric(
            name="negative_transfer.worst_stratum_delta",
            value=float(worst[1]),
            units="nats/obs",
            kind="systematic",
            interval=worst[2],
            threshold=-abs(thr.min_delta_log_score) - 0.05,
            direction="greater_is_better",
            note=f"worst subgroup: {worst[0]} (vs best single-modality baseline)",
        )
    )
    return SubCheck(
        name="no_negative_transfer",
        description="Fusion may not degrade any declared subgroup relative to a single view.",
        metrics=metrics,
        mandatory=True,
        falsified_by="negative transfer: any stratum significantly worse than a single view",
    )


def _as_interval(values: Sequence[float] | float, *, seed: int) -> tuple[float, Interval | None]:
    if np.isscalar(values):
        return float(values), None  # type: ignore[arg-type]
    arr = np.asarray(list(values), dtype=float)
    if arr.size < 2:
        return float(arr.ravel()[0]), None
    return bootstrap_ci(arr, seed=seed)


# ==========================================================================
# G1 -- typed fusion > naive resampling
# ==========================================================================
def run_g1(
    *,
    train: Dataset | None = None,
    test: Dataset | None = None,
    candidate: Any = None,
    baselines: Mapping[str, Any] | None = None,
    delay_true: float | None = None,
    delay_estimates: Mapping[str, Sequence[float] | float] | None = None,
    intervention: Mapping[str, Dataset] | None = None,
    thresholds: Thresholds = Thresholds(),
    seed: int = 0,
    source_cards: Sequence[str] = (),
) -> ClaimReport:
    """G1: typed, source-native fusion versus naive resampling.

    Required baselines (thesis): a carefully tuned **naive resampling** model
    and at least one **single-modality** model, at matched compute and
    parameter count.  Required evidence: held-out likelihood, calibration,
    delay recovery, intervention forecast.  Any of these missing makes the
    gate ``COULD_NOT_RUN``.
    """
    thr = thresholds
    baselines = dict(baselines or {})
    man = _manifest(
        "G1", seed=seed, thresholds=thr,
        baselines=sorted(baselines) or ["<none supplied>"], source_cards=source_cards,
    )
    subs: list[SubCheck] = []
    rows: list[BaselineResult] = []
    artifacts: dict[str, Any] = {}

    missing: list[str] = []
    if candidate is None:
        missing.append("typed fusion candidate (agent I / agent E)")
    if train is None or test is None:
        missing.append("held-out train/test datasets (agent B source cards)")
    single = [k for k in baselines if k.startswith("single_modality")]
    if "naive_resampling" not in baselines:
        missing.append("baseline 'naive_resampling' (the thing the claim is against)")
    if not single:
        missing.append("at least one baseline named 'single_modality_*'")
    if missing:
        for i, m in enumerate(missing):
            subs.append(
                could_not_run(
                    f"inputs[{i}]", "Required gate input", f"missing: {m}",
                    falsified_by=CLAIMS["G1"]["falsified_by"],
                )
            )
        return ClaimReport(manifest=man, subchecks=subs, kind="gate").finalize()

    assert train is not None and test is not None
    results = _eval_arms({"candidate": candidate, **baselines}, train, test, seed=seed)
    cand = results["candidate"]
    base = {k: v for k, v in results.items() if k != "candidate"}

    # 1. matched capacity (mandatory: an unmatched win is not a win)
    verdict = check_matched(
        cand.extras.get("_model"),
        {k: v.extras.get("_model") for k, v in base.items()},
        tol=thr.capacity_tol,
        candidate_name="typed_fusion",
    )
    subs.append(matched_subcheck(verdict))
    artifacts["capacity"] = {
        "candidate": verdict.candidate_budget.as_dict(),
        "baselines": {k: b.as_dict() for k, b in verdict.baseline_budgets.items()},
        "ratios": verdict.ratios,
    }

    # 2. held-out likelihood versus every baseline
    sc, detail = _delta_subcheck(
        "heldout_log_score",
        "Held-out calibrated log score against naive resampling and single-modality views.",
        cand, base, thr, seed=seed,
        falsified_by="no reproducible gain over resampling or single-modality baselines",
    )
    subs.append(sc)
    artifacts["delta_log_score"] = detail

    # 3. calibration / overconfidence
    subs.append(_calibration_subcheck(cand, base, thr))

    # 4. negative transfer versus the best single-modality view
    best_single = max(single, key=lambda k: base[k].mean_log_score)
    subs.append(_negative_transfer_subcheck(cand, base[best_single], test, thr, seed=seed))

    # 5. delay recovery
    if delay_true is None or not delay_estimates or "candidate" not in delay_estimates:
        subs.append(
            could_not_run(
                "delay_recovery",
                "Recovery of the conduction/onset delay against the known value.",
                "no delay ground truth or no per-arm delay estimates were supplied; "
                "'delay recovery' is a named component of this claim and cannot be skipped",
                falsified_by="delay recovered no better than by naive resampling",
            )
        )
    else:
        cval, civ = _as_interval(delay_estimates["candidate"], seed=seed)
        err = abs(cval - delay_true) / max(abs(delay_true), 1e-9)
        base_errs = {
            k: abs(_as_interval(v, seed=seed)[0] - delay_true) / max(abs(delay_true), 1e-9)
            for k, v in delay_estimates.items() if k != "candidate"
        }
        best_base_err = min(base_errs.values()) if base_errs else float("inf")
        subs.append(
            SubCheck(
                name="delay_recovery",
                description="Relative error in recovering the true delay, versus baselines.",
                metrics=[
                    Metric(
                        name="delay.relative_error",
                        value=float(err),
                        kind="identifiability",
                        interval=(Interval(
                            abs(civ.lo - delay_true) / max(abs(delay_true), 1e-9),
                            abs(civ.hi - delay_true) / max(abs(delay_true), 1e-9),
                        ) if civ else None),
                        exact=civ is None,
                        threshold=thr.max_delay_rel_error,
                        direction="less_is_better",
                    ),
                    Metric(
                        name="delay.error_advantage_over_best_baseline",
                        value=float(best_base_err - err),
                        kind="identifiability",
                        exact=True,
                        threshold=0.0,
                        direction="greater_is_better",
                        note=f"baseline errors: {base_errs}",
                    ),
                ],
                mandatory=True,
                falsified_by="delay recovered no better than by naive resampling",
            )
        )
        artifacts["delay"] = {"true": delay_true, "candidate": cval, "baseline_errors": base_errs}

    # 6. intervention forecast on held-out perturbations
    if not intervention or "train" not in intervention or "test" not in intervention:
        subs.append(
            could_not_run(
                "intervention_forecast",
                "Forecast of held-out interventions (thesis column 2 for this claim).",
                "no intervention holdout supplied (agent G intervention operators / an "
                "identified perturbation dataset); correlation fit alone cannot support the "
                "fusion claim's intervention-forecast component",
                falsified_by="fusion gives no advantage on held-out interventions",
            )
        )
    else:
        ires = _eval_arms(
            {"candidate": candidate, **baselines}, intervention["train"], intervention["test"],
            seed=seed,
        )
        icand = ires["candidate"]
        ibase = {k: v for k, v in ires.items() if k != "candidate"}
        isc, idetail = _delta_subcheck(
            "intervention_forecast",
            "Held-out intervention forecast log score against every baseline.",
            icand, ibase, thr, seed=seed,
            falsified_by="fusion gives no advantage on held-out interventions",
            label="intervention_delta_log_score",
        )
        subs.append(isc)
        artifacts["intervention_delta"] = idetail

    # diagnostics: estimated optimism from selecting the best of the arms (§11.2)
    n_folds = 5
    if cand.log_score.size >= 3 * n_folds and len(results) >= 2:
        means = np.array([[float(np.mean(c)) for c in np.array_split(r.log_score, n_folds)]
                          for r in results.values()])
        so = selection_optimism(means, seed=seed, n_boot=400)
        subs.append(
            SubCheck(
                name="selection_optimism",
                description="Estimated optimism from selecting the best of the compared arms.",
                metrics=so.metrics(),
                mandatory=False,
            )
        )
        artifacts["selection_optimism"] = {
            "optimism": so.optimism, "n_models": so.n_models, "n_folds": so.n_folds,
        }

    rows = _baseline_rows(base, {k: ("naive-resampling control" if k == "naive_resampling"
                                     else "single-modality control") for k in base}, seed=seed)
    return ClaimReport(
        manifest=man, subchecks=subs, baselines_run=rows, artifacts=artifacts, kind="gate",
        notes=[
            "Matching is mandatory: an unmatched win is not a win.",
            "Held-out likelihood is a calibrated log score, so accuracy cannot be bought "
            "with overconfidence.",
            "This gate compares models by HELD-OUT PREDICTION, which is falsifiable: a "
            "fusion model with more inputs can and does lose out of sample (see the "
            "negative control in tests/bench). It must not be confused with the expected "
            "Fisher information comparison, where under the modality-block-diagonal form "
            "of T4 joint = sum of modalities identically. The falsifiable information-side "
            "comparisons for this claim are native-versus-naively-resampled and the "
            "non-additive joint information under joint_whitening=True; see G4's "
            "modality_additivity_declaration.",
        ],
    ).finalize()


# ==========================================================================
# G2 -- anatomy improves inference
# ==========================================================================
def run_g2(
    *,
    train: Dataset | None = None,
    test: Dataset | None = None,
    ood: Dataset | None = None,
    model_for_graph: Callable[[np.ndarray], Any] | None = None,
    anatomy: np.ndarray | None = None,
    controls: Mapping[str, np.ndarray] | None = None,
    causal_holdout: Mapping[str, Dataset] | None = None,
    data_efficiency_sizes: Sequence[int] | None = None,
    n_efficiency_seeds: int = 3,
    corrupt_fraction: float = 0.5,
    thresholds: Thresholds = Thresholds(),
    seed: int = 0,
    source_cards: Sequence[str] = (),
) -> ClaimReport:
    """G2: anatomical topology versus dense, randomized and distance-matched graphs.

    The controls are agent C's; this gate refuses to invent them, because the
    control *is* the experiment.  The gate additionally tests the second
    falsifier explicitly: **topology errors absorbed by residuals**.  A model
    whose learned residual repairs a corrupted topology has not demonstrated
    that the topology is load-bearing (compare compiler refusal R05).
    """
    thr = thresholds
    required = ("dense", "randomized", "distance_matched")
    man = _manifest(
        "G2", seed=seed, thresholds=thr,
        baselines=list(required), source_cards=source_cards,
        refusal_fixtures=["R05 (learned residual dominating a mechanistic term)"],
    )
    subs: list[SubCheck] = []
    artifacts: dict[str, Any] = {}

    if controls is None:
        dep = adapters.anatomy_controls()
        if not dep.available:
            subs.append(
                could_not_run(
                    "graph_controls", "Dense / randomized / distance-matched controls.",
                    dep.blocker,
                    falsified_by=CLAIMS["G2"]["falsified_by"],
                )
            )
    missing: list[str] = []
    if model_for_graph is None:
        missing.append("model_for_graph(adjacency) factory (agent E / agent I)")
    if anatomy is None:
        missing.append("anatomical adjacency (agent C)")
    if controls is None:
        missing.append("graph controls (agent C): " + ", ".join(required))
    elif any(k not in controls for k in required):
        missing.append(
            "graph controls missing: " + ", ".join(k for k in required if k not in controls)
        )
    if train is None or test is None:
        missing.append("train/test datasets")
    if missing:
        for i, m in enumerate(missing):
            subs.append(could_not_run(f"inputs[{i}]", "Required gate input", f"missing: {m}",
                                      falsified_by=CLAIMS["G2"]["falsified_by"]))
        return ClaimReport(manifest=man, subchecks=subs, kind="gate").finalize()

    assert model_for_graph is not None and anatomy is not None and controls is not None
    assert train is not None and test is not None

    arms: dict[str, Any] = {"anatomy": lambda: model_for_graph(anatomy)}
    for k in required:
        arms[k] = (lambda kk=k: model_for_graph(np.asarray(controls[kk])))
    results = _eval_arms(arms, train, test, seed=seed)
    cand = results["anatomy"]
    base = {k: v for k, v in results.items() if k != "anatomy"}

    # 1. capacity matching (dense will be larger; that is fine and is recorded)
    verdict = check_matched(
        cand.extras.get("_model"), {k: v.extras.get("_model") for k, v in base.items()},
        tol=thr.capacity_tol, candidate_name="anatomy",
    )
    subs.append(matched_subcheck(verdict))
    artifacts["capacity"] = {"ratios": verdict.ratios}

    # 2. equal-capacity controls must not match or exceed the anatomical model
    sc, detail = _delta_subcheck(
        "beats_equal_capacity_controls",
        "Held-out log score against dense, randomized and distance-matched graphs.",
        cand, base, thr, seed=seed,
        falsified_by="equal-capacity controls match or exceed performance",
    )
    subs.append(sc)
    artifacts["delta_log_score"] = detail

    # 3. data efficiency (topology is supported only if it helps when data is scarce)
    sizes = list(data_efficiency_sizes or [])
    if not sizes:
        sizes = [max(8, train.n // 8), max(16, train.n // 4), max(32, train.n // 2), train.n]
        sizes = sorted(set(min(s, train.n) for s in sizes))
    if len(sizes) < 2:
        subs.append(
            could_not_run(
                "data_efficiency", "Score-versus-training-size curve.",
                f"training set of {train.n} rows cannot be subsampled into >=2 sizes",
                falsified_by="no data-efficiency advantage over the controls",
            )
        )
    else:
        curves: dict[str, dict[str, Any]] = {}
        rng = np.random.default_rng(seed)
        per_arm_small: dict[str, list[float]] = {}
        for arm_name, factory in arms.items():
            scores_by_size: list[list[float]] = []
            for s in sizes:
                sc_list: list[float] = []
                for rep in range(n_efficiency_seeds):
                    idx = rng.choice(train.n, size=int(s), replace=False)
                    sub = train.subset(idx, name=f"{train.name}[{s}]")
                    r = _fit_eval(factory(), sub, test, seed=seed + rep)
                    sc_list.append(r.mean_log_score)
                scores_by_size.append(sc_list)
            curves[arm_name] = data_efficiency_curve(sizes, scores_by_size, seed=seed)
            per_arm_small[arm_name] = scores_by_size[0]
        artifacts["data_efficiency"] = curves
        best_control_auc = max(curves[k]["auc_log_size"] for k in base)
        small_delta = paired_bootstrap(
            np.array(per_arm_small["anatomy"]),
            np.array([max(per_arm_small[k][i] for k in base)
                      for i in range(n_efficiency_seeds)]),
            name="data_efficiency.smallest_size_delta", n_boot=thr.n_boot, seed=seed,
        )
        subs.append(
            SubCheck(
                name="data_efficiency",
                description=(
                    "Anatomy must win where data is scarce, not only at the largest "
                    "training size (§11.4: topology is supported only if it improves data "
                    "efficiency, calibration, OOD behaviour or causal prediction)."
                ),
                metrics=[
                    Metric(
                        name="data_efficiency.auc_advantage_over_best_control",
                        value=float(curves["anatomy"]["auc_log_size"] - best_control_auc),
                        kind="efficiency",
                        exact=True,
                        threshold=0.0,
                        direction="greater_is_better",
                    ),
                    small_delta.metric(kind="efficiency", threshold=0.0),
                ] + cand.calibration.metrics(prefix="anatomy")[:1],
                mandatory=True,
                falsified_by="controls are as data-efficient as the anatomical topology",
            )
        )

    # 4. out-of-distribution calibration
    if ood is None:
        subs.append(
            could_not_run(
                "ood_calibration", "Calibration under distribution shift.",
                "no out-of-distribution evaluation set supplied; the claim explicitly "
                "includes out-of-distribution calibration",
                falsified_by="anatomy gives no OOD calibration advantage",
            )
        )
    else:
        ores = _eval_arms(arms, train, ood, seed=seed)
        best_ctrl_err = min(ores[k].calibration.coverage_error for k in base)
        subs.append(
            SubCheck(
                name="ood_calibration",
                description="Coverage error under distribution shift versus the controls.",
                metrics=[
                    Metric(
                        name="ood.coverage_error",
                        value=ores["anatomy"].calibration.coverage_error,
                        kind="calibration",
                        interval=ores["anatomy"].calibration.coverage_error_interval,
                        threshold=thr.max_coverage_error,
                        direction="less_is_better",
                    ),
                    Metric(
                        name="ood.coverage_error_advantage_over_best_control",
                        value=float(best_ctrl_err - ores["anatomy"].calibration.coverage_error),
                        kind="calibration",
                        exact=True,
                        threshold=0.0,
                        direction="greater_is_better",
                    ),
                ],
                mandatory=True,
                falsified_by="a control is at least as well calibrated out of distribution",
            )
        )
        artifacts["ood"] = {k: ores[k].calibration.coverage_error for k in ores}

    # 5. causal forecast
    if not causal_holdout or "train" not in causal_holdout or "test" not in causal_holdout:
        subs.append(
            could_not_run(
                "causal_forecast", "Held-out interventional forecast versus the controls.",
                "no identified intervention holdout supplied (agent G / an identified "
                "perturbation dataset); passive correlation cannot support a causal-forecast "
                "advantage (compare refusal R04)",
                falsified_by="anatomy gives no causal-forecast advantage",
            )
        )
    else:
        cres = _eval_arms(arms, causal_holdout["train"], causal_holdout["test"], seed=seed)
        csc, cdetail = _delta_subcheck(
            "causal_forecast",
            "Held-out interventional forecast versus dense/randomized/distance-matched.",
            cres["anatomy"], {k: v for k, v in cres.items() if k != "anatomy"}, thr, seed=seed,
            falsified_by="anatomy gives no causal-forecast advantage",
            label="causal_delta_log_score",
        )
        subs.append(csc)
        artifacts["causal_forecast"] = cdetail

    # 6. residual absorption of topology errors -- the second thesis falsifier
    rng = np.random.default_rng(seed + 991)
    A = np.asarray(anatomy, dtype=float)
    flat = A.ravel().copy()
    on = np.where(flat > 0)[0]
    off = np.where(flat == 0)[0]
    k = int(round(corrupt_fraction * on.size))
    if k == 0 or off.size == 0:
        subs.append(
            could_not_run(
                "residual_absorption", "Corrupted-topology control.",
                "the anatomical adjacency has no edges to corrupt (or is fully dense); "
                "absorption of topology error is not testable on this graph",
                falsified_by="topology errors are absorbed by residuals",
            )
        )
    else:
        drop = rng.choice(on, size=k, replace=False)
        add = rng.choice(off, size=min(k, off.size), replace=False)
        flat[drop] = 0.0
        flat[add] = 1.0
        A_corrupt = flat.reshape(A.shape)
        cres = _fit_eval(model_for_graph(A_corrupt), train, test, seed=seed)
        d = paired_bootstrap(
            cres.log_score, cand.log_score,
            name="absorption.corrupted_minus_correct", n_boot=thr.n_boot, seed=seed,
        )
        metrics = [
            Metric(
                name="absorption.corrupted_minus_correct_log_score",
                value=d.mean,
                units="nats/obs",
                kind="identifiability",
                interval=d.interval,
                threshold=-1e-3,
                direction="less_is_better",
                require_interval_beats_threshold=True,
                note=(
                    "a corrupted topology must cost held-out likelihood; if it does not, the "
                    "residual absorbed the error and the topology is not load-bearing"
                ),
            )
        ]
        rho = getattr(cres.extras.get("_model"), "residual_energy", None)
        rho_c = getattr(cand.extras.get("_model"), "residual_energy", None)
        if callable(rho) and callable(rho_c):
            r_bad, m_bad = rho()
            r_good, m_good = rho_c()
            ratio_bad = r_bad / max(m_bad, 1e-12)
            ratio_good = r_good / max(m_good, 1e-12)
            metrics.append(
                Metric(
                    name="absorption.residual_gain_ratio_increase",
                    value=float(ratio_bad - ratio_good),
                    kind="identifiability",
                    exact=True,
                    note=(
                        f"||R||/||F_mech||: correct={ratio_good:.4g}, corrupted={ratio_bad:.4g}; "
                        "a large increase with unchanged accuracy is refusal R05 behaviour"
                    ),
                )
            )
        else:
            metrics.append(
                Metric(
                    name="absorption.residual_energy_reported",
                    value=0.0,
                    kind="diagnostic",
                    exact=True,
                    note=(
                        "arm does not expose residual_energy(); the R05 energy-ratio evidence "
                        "is unavailable and only the accuracy-cost evidence was used"
                    ),
                )
            )
        subs.append(
            SubCheck(
                name="residual_absorption",
                description=(
                    "Corrupt the topology and require the model to get worse. "
                    "Thesis falsifier: 'topology errors are absorbed by residuals'."
                ),
                metrics=metrics,
                mandatory=True,
                falsified_by="a corrupted topology costs nothing -> the residual absorbed it",
            )
        )
        artifacts["absorption"] = {
            "corrupt_fraction": corrupt_fraction,
            "delta": d.mean,
            "ci": [d.interval.lo, d.interval.hi],
        }

    rows = _baseline_rows(base, {k: "topology control" for k in base}, seed=seed)
    return ClaimReport(
        manifest=man, subchecks=subs, baselines_run=rows, artifacts=artifacts, kind="gate",
        notes=[
            "Sparsity or plausibility alone is not evidence for the declared topology "
            "(Appendix D, 'Connectome prior value').",
            "Controls are agent C's; this gate refuses to synthesise them.",
        ],
    ).finalize()


# ==========================================================================
# G3 -- multiresolution adds information
# ==========================================================================
def _hf_component(y: np.ndarray, restriction: np.ndarray) -> np.ndarray:
    """The part of ``y`` annihilated by the restriction map.

    This is the fine-scale content proper: whatever survives subtracting the
    least-squares prolongation of the coarse view. Detail living here is
    exactly the detail a coarse observation cannot see.
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    R = np.asarray(restriction, dtype=float)
    P = R.T @ np.linalg.pinv(R @ R.T)      # least-squares prolongation
    return y - (P @ (R @ y.T)).T


def _hf_energy(y: np.ndarray, restriction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(_hf_component(y, restriction) ** 2)))


def run_g3(
    *,
    fine_train: Dataset | None = None,
    fine_test: Dataset | None = None,
    coarse_train: Dataset | None = None,
    coarse_test: Dataset | None = None,
    restriction: np.ndarray | None = None,
    multires_model: Any = None,
    coarse_only_model: Any = None,
    fine_evidence_block: str = "fine_evidence",
    compute_full_fine: float | None = None,
    compute_adaptive: float | None = None,
    thresholds: Thresholds = Thresholds(),
    seed: int = 0,
    source_cards: Sequence[str] = (),
) -> ClaimReport:
    """G3: multiresolution state versus decoration.

    The high-frequency hallucination check is mandatory and is run the way
    Appendix D specifies: *withhold fine-scale evidence while retaining coarse
    data; compare uncertainty and reconstruction to a coarse-only model.*
    Fine detail is valid only where source support justifies it.
    """
    thr = thresholds
    man = _manifest(
        "G3", seed=seed, thresholds=thr,
        baselines=["coarse_only", "coarse_only(+withheld fine evidence)"],
        source_cards=source_cards,
        refusal_fixtures=["R02 (prolongation without tested restriction partner)",
                          "R03 (global cross-scale state above cocycle tolerance)"],
    )
    subs: list[SubCheck] = []
    artifacts: dict[str, Any] = {}

    missing: list[str] = []
    if multires_model is None:
        missing.append("multiresolution candidate (agent E/I)")
    if coarse_only_model is None:
        missing.append("coarse-only baseline")
    if fine_train is None or fine_test is None:
        missing.append("fine-scale train/test datasets")
    if coarse_test is None:
        missing.append("coarse-scale evaluation set (boundary observable)")
    if restriction is None:
        missing.append("restriction map R (agent D transforms / agent C parcellation)")
    if missing:
        for i, m in enumerate(missing):
            subs.append(could_not_run(f"inputs[{i}]", "Required gate input", f"missing: {m}",
                                      falsified_by=CLAIMS["G3"]["falsified_by"]))
        return ClaimReport(manifest=man, subchecks=subs, kind="gate").finalize()

    assert fine_train is not None and fine_test is not None and restriction is not None
    assert coarse_test is not None

    mk = as_factory(multires_model)
    mk_coarse = as_factory(coarse_only_model)
    fine_res = _fit_eval(mk(), fine_train, fine_test, seed=seed)
    fine_res = EvalResult(**{**fine_res.__dict__, "arm": "multiresolution"})
    coarse_res = _fit_eval(mk_coarse(), fine_train, fine_test, seed=seed)
    coarse_res = EvalResult(**{**coarse_res.__dict__, "arm": "coarse_only"})

    # 1. native-scale prediction
    sc, detail = _delta_subcheck(
        "native_scale_prediction",
        "Fine-scale held-out log score of the multiresolution model versus coarse-only.",
        fine_res, {"coarse_only": coarse_res}, thr, seed=seed,
        falsified_by="fine views cannot improve supported observables",
    )
    subs.append(sc)
    artifacts["native_scale"] = detail

    # 2. calibrated refinement
    subs.append(_calibration_subcheck(fine_res, {"coarse_only": coarse_res}, thr,
                                      name="calibrated_refinement"))

    # 3. boundary agreement (round trip through the restriction map)
    R = np.asarray(restriction, dtype=float)
    pred_fine = fine_res.prediction.mean.reshape(fine_test.n, -1)
    if pred_fine.shape[1] != R.shape[1]:
        subs.append(
            could_not_run(
                "boundary_agreement", "Fine/coarse agreement on the boundary observable.",
                f"restriction map is {R.shape} but fine predictions are {pred_fine.shape}; "
                "the restriction partner is not declared for this view (compare refusal R02)",
                falsified_by="round-trip / boundary tests fail",
            )
        )
    else:
        restricted = pred_fine @ R.T
        truth_coarse = coarse_test.targets.reshape(coarse_test.n, -1)
        n = min(restricted.shape[0], truth_coarse.shape[0])
        rel = np.abs(restricted[:n] - truth_coarse[:n]) / (
            np.abs(truth_coarse[:n]).mean() + 1e-12
        )
        pt, iv = bootstrap_ci(rel.ravel(), seed=seed, n_boot=thr.n_boot)
        subs.append(
            SubCheck(
                name="boundary_agreement",
                description=(
                    "Restricting the fine prediction must reproduce the coarse observable "
                    "within the declared tolerance before the scale relation may be used."
                ),
                metrics=[
                    Metric(
                        name="boundary.relative_disagreement",
                        value=pt,
                        kind="numerical",
                        interval=iv,
                        threshold=thr.boundary_rel_tol,
                        direction="less_is_better",
                        require_interval_beats_threshold=True,
                    )
                ],
                mandatory=True,
                falsified_by="restricted fine view disagrees with the coarse observable",
            )
        )
        artifacts["boundary"] = {"relative_disagreement": pt, "ci": [iv.lo, iv.hi]}

    # 4. high-frequency hallucination (mandatory)
    if fine_evidence_block not in fine_train.inputs:
        subs.append(
            could_not_run(
                "high_frequency_hallucination",
                "Withhold fine-scale evidence and compare reconstruction and uncertainty.",
                f"input block {fine_evidence_block!r} not present, so fine evidence cannot be "
                "withheld; the hallucination control is not runnable",
                falsified_by="fine detail emitted without source support",
            )
        )
    else:
        tr_w = fine_train.without(fine_evidence_block, name="fine.train-no-fine-evidence")
        te_w = fine_test.without(fine_evidence_block, name="fine.test-no-fine-evidence")
        try:
            wres = _fit_eval(mk(), tr_w, te_w, seed=seed)
        except Exception as exc:
            wres = None
            reason = f"model refused to run without fine evidence: {type(exc).__name__}: {exc}"
        if wres is None:
            subs.append(
                could_not_run(
                    "high_frequency_hallucination",
                    "Withhold fine-scale evidence and compare reconstruction and uncertainty.",
                    reason,
                    falsified_by="fine detail emitted without source support",
                )
            )
        else:
            # The comparison Appendix D mandates is against a *coarse-only
            # model given the same evidence*, not against the truth: fine-scale
            # structure that coarse evidence genuinely predicts is supported,
            # and only the excess over that is hallucination.
            pm = wres.prediction.mean.reshape(fine_test.n, -1)
            cm = coarse_res.prediction.mean.reshape(fine_test.n, -1)
            ty = fine_test.targets.reshape(fine_test.n, -1)
            hf_truth = _hf_energy(ty, R)
            hf_pred = _hf_energy(pm, R)
            hf_coarse = _hf_energy(cm, R)
            idx = float(hf_pred / max(hf_coarse, 1e-12))
            hf_err_pred = float(np.sqrt(np.mean(_hf_component(ty - pm, R) ** 2)))
            hf_err_coarse = float(np.sqrt(np.mean(_hf_component(ty - cm, R) ** 2)))
            err_ratio = float(hf_err_pred / max(hf_err_coarse, 1e-12))
            infl = float(np.mean(wres.prediction.sd) / max(np.mean(fine_res.prediction.sd), 1e-12))
            rng = np.random.default_rng(seed)
            boots = []
            for _ in range(200):
                b = rng.integers(0, fine_test.n, size=fine_test.n)
                boots.append(_hf_energy(pm[b], R) / max(_hf_energy(cm[b], R), 1e-12))
            lo, hi = np.quantile(boots, [0.025, 0.975])
            subs.append(
                SubCheck(
                    name="high_frequency_hallucination",
                    description=(
                        "With fine evidence withheld, the model may not emit fine detail it "
                        "cannot support, and its uncertainty must increase."
                    ),
                    metrics=[
                        Metric(
                            name="hallucination.hf_energy_index",
                            value=idx,
                            kind="systematic",
                            interval=Interval(float(lo), float(hi)),
                            threshold=thr.max_hallucination_index,
                            direction="less_is_better",
                            note=(
                                "emitted fine-scale energy / coarse-only model's fine-scale "
                                f"energy, both with the fine evidence withheld (truth's "
                                f"fine-scale energy = {hf_truth:.4g}, coarse-only = "
                                f"{hf_coarse:.4g}); >1 means detail with no source support"
                            ),
                        ),
                        Metric(
                            name="hallucination.hf_error_ratio_vs_coarse_only",
                            value=err_ratio,
                            kind="accuracy",
                            exact=True,
                            threshold=1.10,
                            direction="less_is_better",
                            note="fine-subspace reconstruction error relative to coarse-only; "
                                 "fabricated detail makes this worse, not better",
                        ),
                        Metric(
                            name="hallucination.uncertainty_inflation",
                            value=infl,
                            kind="calibration",
                            exact=True,
                            threshold=thr.min_uncertainty_inflation,
                            direction="greater_is_better",
                            note="mean predictive sd without fine evidence / with it; a model "
                                 "that stays equally confident outside measured tiles fails",
                        ),
                        Metric(
                            name="hallucination.coverage_error_without_fine_evidence",
                            value=wres.calibration.coverage_error,
                            kind="calibration",
                            interval=wres.calibration.coverage_error_interval,
                            threshold=thr.max_coverage_error * 2,
                            direction="less_is_better",
                        ),
                    ],
                    mandatory=True,
                    falsified_by=(
                        "fine detail emitted where no source supports it, or unchanged "
                        "confidence outside measured tiles"
                    ),
                )
            )
            artifacts["hallucination"] = {
                "hf_energy_truth": hf_truth,
                "hf_energy_pred_no_evidence": hf_pred,
                "hf_energy_coarse_only": hf_coarse,
                "index": idx,
                "hf_error_ratio_vs_coarse_only": err_ratio,
                "uncertainty_inflation": infl,
            }

    # 5. compute savings (reported, not claim-bearing on its own)
    if compute_full_fine is None or compute_adaptive is None:
        subs.append(
            SubCheck(
                name="compute_savings",
                description="Adaptive refinement compute versus full fine resolution.",
                metrics=[],
                mandatory=False,
                forced_status="COULD_NOT_RUN",
                reason="no compute accounting supplied for the adaptive and full-fine runs",
            )
        )
    else:
        subs.append(
            SubCheck(
                name="compute_savings",
                description="Adaptive refinement compute versus full fine resolution.",
                metrics=[
                    Metric(
                        name="compute.adaptive_over_full_fine",
                        value=float(compute_adaptive) / max(float(compute_full_fine), 1e-12),
                        kind="efficiency",
                        exact=True,
                        threshold=1.0,
                        direction="less_is_better",
                    )
                ],
                mandatory=False,
            )
        )

    rows = _baseline_rows({"coarse_only": coarse_res}, {"coarse_only": "coarse-only control"},
                          seed=seed)
    return ClaimReport(
        manifest=man, subchecks=subs, baselines_run=rows, artifacts=artifacts, kind="gate",
        notes=[
            "Boundary agreement is a precondition: fine and coarse backends must agree within "
            "the declared tolerance *before* adaptive resolution is used for inference "
            "(§11.1); see scwbd.bench.numerics.permit_adaptive_resolution.",
        ],
    ).finalize()


# ==========================================================================
# G4 -- perturbation reduces non-identifiability
# ==========================================================================
def _fisher_pair(obj: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """Accept a bare information matrix, a ``FisherReport``, or a ``DesignInformation``.

    When the backend already separates likelihood from prior the prior is
    returned separately so it is never silently counted as evidence — thesis
    §0.3 requires the prior contribution to be shown apart from the likelihood.
    """
    for like_attr, prior_attr in (("I_likelihood", "I_prior"),
                                  ("information_likelihood", "information_prior")):
        like = getattr(obj, like_attr, None)
        if like is None:
            continue
        prior = getattr(obj, prior_attr, None)
        like_arr = np.asarray(like, dtype=float)
        prior_arr = (np.asarray(prior, dtype=float) if prior is not None
                     else np.zeros_like(like_arr))
        return like_arr, prior_arr
    return np.asarray(obj, dtype=float), None


def _modality_information(obj: Any) -> dict[str, np.ndarray]:
    """Per-modality information blocks, when the backend exposes them."""
    for attr in ("information_by_modality", "I_by_modality"):
        blocks = getattr(obj, attr, None)
        if blocks:
            return {k: np.asarray(v, dtype=float) for k, v in dict(blocks).items()}
    return {}


def _rel_frobenius(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(b)) or 1.0
    return float(np.linalg.norm(a - b) / denom)


def _schur_theta(I: np.ndarray, theta_idx: np.ndarray, nuis_idx: np.ndarray) -> np.ndarray:
    """Information about theta after profiling out the nuisance block.

    This is the quantity that separates "the perturbation informed the science"
    from "the perturbation informed the field model".
    """
    I = np.asarray(I, dtype=float)
    Itt = I[np.ix_(theta_idx, theta_idx)]
    if nuis_idx.size == 0:
        return Itt
    Inn = I[np.ix_(nuis_idx, nuis_idx)]
    Itn = I[np.ix_(theta_idx, nuis_idx)]
    Inn_inv = np.linalg.pinv(Inn + 1e-9 * np.eye(Inn.shape[0]))
    return Itt - Itn @ Inn_inv @ Itn.T


def run_g4(
    *,
    fisher: Callable[[str], np.ndarray] | None = None,
    theta_index: Sequence[int] | None = None,
    nuisance_index: Sequence[int] | None = None,
    baseline_design: str = "joint_native",
    intervention_design: str = "joint_plus_impulse",
    prior_design: str | None = "prior",
    recovery: Mapping[str, Mapping[str, float]] | None = None,
    model_evidence: Mapping[str, Mapping[str, float]] | None = None,
    fisher_whitened: Callable[[str], Any] | None = None,
    single_modality_designs: Sequence[str] = ("eeg", "fmri"),
    basis: str = "prior_standardised",
    thresholds: Thresholds = Thresholds(),
    seed: int = 0,
    source_cards: Sequence[str] = (),
) -> ClaimReport:
    """G4: does the perturbation actually reduce non-identifiability?

    ``fisher`` is agent H's machinery (probed automatically when not passed).
    ``recovery`` maps each of ``direction``, ``delay``, ``gain``, ``dose``,
    ``state_dependence`` to ``{"true":..., "estimate":..., "lo":..., "hi":...}``
    from a **prospective** held-out perturbation.  ``model_evidence`` maps
    design -> {model_name: log evidence per observation} and is used to test
    the first falsifier ("intervention fails to distinguish posterior models").

    **What this gate does and does not test.**  Under the modality-block-
    diagonal form of T4, ``I_{EEG+BOLD} = I_EEG + I_BOLD`` *identically*.
    "Joint beats single-modality" is therefore an algebraic identity in that
    form, not a hypothesis, and this gate refuses to report it as a result.
    The comparison G4 actually tests is **intervention versus baseline design**
    (in the theta block, with the observation nuisances profiled out).  The
    falsifiable part of the *fusion* claim lives elsewhere: native versus
    naively resampled (agent H's benchmark, feeding G1), and the non-additive
    joint information that only appears under ``joint_whitening=True``, which
    this gate reports when ``fisher_whitened`` is supplied.

    ``basis`` is recorded in the manifest and on every eigenvalue metric.  A
    condition number is meaningless without it, so it is stated rather than
    assumed: the default ``"prior_standardised"`` basis makes ``I_prior`` the
    identity and makes parameters with different units comparable.
    """
    thr = thresholds
    man = _manifest(
        "G4", seed=seed, thresholds=thr,
        baselines=[baseline_design, "eeg-only", "fmri-only", "naive-resampled joint"],
        source_cards=source_cards,
        refusal_fixtures=["R04 (causal operator from passive correlation alone)"],
    )
    man.acceptance_thresholds["basis"] = basis
    man.acceptance_thresholds["baseline_design"] = baseline_design
    man.acceptance_thresholds["intervention_design"] = intervention_design
    subs: list[SubCheck] = []
    artifacts: dict[str, Any] = {}

    auto_probed = False
    probed_name = ""
    if fisher is None:
        dep = adapters.fisher_backend()
        if dep.available:
            fisher = dep.obj  # type: ignore[assignment]
            auto_probed = True
            probed_name = dep.name
        else:
            subs.append(
                could_not_run(
                    "fisher_information",
                    "Rank / eigenvalue improvement in expected Fisher information.",
                    dep.blocker + "; G4 consumes agent H's Fisher machinery and will not "
                    "reimplement it (a gate that computes the quantity it audits is not an audit)",
                    falsified_by=CLAIMS["G4"]["falsified_by"],
                )
            )
    if (theta_index is None or nuisance_index is None) and auto_probed:
        # agent H may declare the partition itself; consume it rather than guess
        dep = adapters.theta_partition()
        if dep.available:
            theta_names, all_names = dep.obj
            theta_index = [i for i, n in enumerate(all_names) if n in set(theta_names)]
            nuisance_index = [i for i, n in enumerate(all_names) if n not in set(theta_names)]
    if theta_index is None or nuisance_index is None:
        subs.append(
            could_not_run(
                "parameter_partition",
                "Partition of the parameter vector into science and nuisance blocks.",
                "theta_index / nuisance_index not supplied; without them, information added "
                "to the field/observation model cannot be distinguished from information "
                "added about the scientific parameters — which is exactly this claim's "
                "falsifier",
                falsified_by="intervention adds only field-model uncertainty",
            )
        )

    if fisher is not None and theta_index is not None and nuisance_index is not None:
        ti = np.asarray(list(theta_index), dtype=int)
        ni = np.asarray(list(nuisance_index), dtype=int)
        try:
            _probe = _fisher_pair(fisher(baseline_design))[0]
            p = _probe.shape[0]
            if ti.size and (ti.max() >= p or (ni.size and ni.max() >= p)):
                raise IndexError(
                    f"the declared parameter partition indexes {max(ti.max(), ni.max()) + 1} "
                    f"parameters but the information matrix is {p}x{p}; the partition and "
                    "the backend describe different parameter vectors"
                )
            b_mat, b_prior = _fisher_pair(fisher(baseline_design))
            i_mat, i_prior = _fisher_pair(fisher(intervention_design))
            if b_prior is not None:
                # the backend already separates likelihood from prior
                I_base, I_int, I_prior = b_mat, i_mat, np.zeros_like(b_mat)
                prior_separated = True
            else:
                I_base, I_int = b_mat, i_mat
                I_prior = (_fisher_pair(fisher(prior_design))[0]
                           if prior_design is not None else np.zeros_like(I_base))
                prior_separated = prior_design is not None
        except Exception as exc:
            if auto_probed and isinstance(exc, TypeError):
                reason = (
                    f"agent H's {probed_name} is present but is not a design -> information "
                    f"map (it raised {type(exc).__name__}: {exc}). G4 consumes agent H's "
                    "Fisher machinery and will not reimplement it, so pass "
                    "fisher=lambda design: expected_fisher(u, cfg, proto, design=design) "
                    "bound to the system and protocol under test."
                )
            else:
                reason = f"the Fisher backend raised {type(exc).__name__}: {exc}"
            subs.append(
                could_not_run(
                    "fisher_information", "Fisher information across designs.", reason,
                    falsified_by=CLAIMS["G4"]["falsified_by"],
                )
            )
        else:
            # prior contribution is shown separately so a full-rank posterior
            # cannot disguise a prior-dominated likelihood (thesis §0.3)
            L_base = I_base - I_prior
            L_int = I_int - I_prior
            r_base = int(np.linalg.matrix_rank(L_base, tol=1e-8))
            r_int = int(np.linalg.matrix_rank(L_int, tol=1e-8))
            ev_base = np.linalg.eigvalsh(_schur_theta(L_base, ti, ni))
            ev_int = np.linalg.eigvalsh(_schur_theta(L_int, ti, ni))
            # clip at zero: subtracting the prior can leave tiny negative
            # eigenvalues, and a negative "information" is a numerical artefact,
            # not evidence of identifiability
            min_base = float(max(np.min(ev_base), 0.0))
            min_int = float(max(np.min(ev_int), 0.0))
            gain = (min_int / min_base) if min_base > 1e-12 else (
                float("inf") if min_int > 1e-12 else 1.0
            )
            cond_base = float(np.max(ev_base) / max(np.min(ev_base), 1e-12))
            cond_int = float(np.max(ev_int) / max(np.min(ev_int), 1e-12))
            nuis_gain = float(
                np.min(np.linalg.eigvalsh(L_int[np.ix_(ni, ni)]))
                - np.min(np.linalg.eigvalsh(L_base[np.ix_(ni, ni)]))
            )
            subs.append(
                SubCheck(
                    name="fisher_rank_and_eigenvalue",
                    description=(
                        f"Likelihood-only (prior removed) rank and minimum eigenvalue of the "
                        f"theta block with nuisance profiled out, in the {basis} basis. "
                        f"The comparison is {intervention_design} versus {baseline_design} "
                        "— a design contrast that can fail, not a modality identity."
                    ),
                    metrics=[
                        Metric(
                            name="fisher.rank_increase", value=float(r_int - r_base),
                            kind="identifiability", exact=True, threshold=-0.5,
                            direction="greater_is_better",
                            note=f"rank {r_base} -> {r_int} (likelihood only)",
                        ),
                        Metric(
                            name="fisher.theta_min_eigenvalue_gain", value=float(gain),
                            kind="identifiability", exact=True,
                            threshold=thr.min_fisher_eig_gain,
                            direction="greater_is_better",
                            note=(
                                f"basis={basis}; min eig of the theta Schur complement "
                                f"{min_base:.4g} -> {min_int:.4g}; nuisance block gained "
                                f"{nuis_gain:.4g} (reported separately so 'adds only "
                                "field-model information' is visible)"
                            ),
                        ),
                        Metric(
                            name="fisher.theta_condition_number_ratio",
                            value=float(cond_int / max(cond_base, 1e-12)),
                            kind="identifiability", exact=True, threshold=1.0,
                            direction="less_is_better",
                            note=f"basis={basis}; a condition number without a declared "
                                 "basis is meaningless, so the basis travels with the number",
                        ),
                    ],
                    mandatory=True,
                    falsified_by=(
                        "no improvement in the theta block once the nuisance/field model is "
                        "profiled out -> the intervention only added field-model uncertainty"
                    ),
                )
            )
            artifacts["fisher"] = {
                "basis": basis,
                "baseline_design": baseline_design,
                "intervention_design": intervention_design,
                "rank_base": r_base, "rank_intervention": r_int,
                "theta_min_eig_base": min_base, "theta_min_eig_intervention": min_int,
                "theta_eigs_base": ev_base.tolist(), "theta_eigs_int": ev_int.tolist(),
                "nuisance_min_eig_gain": nuis_gain,
                "prior_removed": bool(prior_separated),
            }

    # ------------------------------------------------------------------
    # Modality additivity: a declaration, not a claim.
    #
    # Under T4's modality-block-diagonal form the joint information is the sum
    # of the per-modality informations *identically*.  A gate that reported
    # "joint >= single-modality" from this would be reporting arithmetic.  We
    # therefore measure the residual (to confirm the backend really is in that
    # form) and, when a whitened map is supplied, measure the part of the joint
    # information that is NOT additive — which is the only part of the fusion
    # story that can fail.
    # ------------------------------------------------------------------
    if fisher is not None:
        add_metrics: list[Metric] = []
        add_art: dict[str, Any] = {"basis": basis}
        try:
            joint_obj = fisher(baseline_design)
            joint_mat = _fisher_pair(joint_obj)[0]
            blocks = _modality_information(joint_obj)
            if not blocks:
                parts = []
                for d in single_modality_designs:
                    try:
                        parts.append(_fisher_pair(fisher(d))[0])
                    except Exception:
                        parts = []
                        break
                blocks = {d: m for d, m in zip(single_modality_designs, parts)}
            if blocks and len(blocks) >= 2:
                summed = sum(blocks.values())
                resid = _rel_frobenius(joint_mat, summed)
                add_art["block_diagonal_residual"] = resid
                add_art["modalities"] = sorted(blocks)
                add_metrics.append(
                    Metric(
                        name="additivity.block_diagonal_residual",
                        value=resid, kind="identifiability", exact=True,
                        threshold=1e-6, direction="less_is_better",
                        note=(
                            "||I_joint - sum_m I_m|| / ||I_joint|| under the "
                            "modality-block-diagonal form of T4. Near zero confirms the "
                            "IDENTITY I_{EEG+BOLD} = I_EEG + I_BOLD. 'Joint beats "
                            "single-modality' is therefore arithmetic in this form and is "
                            "NOT reported by this gate as evidence for anything."
                        ),
                    )
                )
        except Exception as exc:  # pragma: no cover - backend-specific
            add_art["error"] = f"{type(exc).__name__}: {exc}"

        if fisher_whitened is not None and theta_index is not None \
                and nuisance_index is not None:
            try:
                wj = fisher_whitened(baseline_design)
                w_mat = _fisher_pair(wj)[0]
                bd_mat = _fisher_pair(fisher(baseline_design))[0]
                ti2 = np.asarray(list(theta_index), dtype=int)
                ni2 = np.asarray(list(nuisance_index), dtype=int)
                excess = _rel_frobenius(w_mat, bd_mat)
                ev_bd = float(max(np.min(np.linalg.eigvalsh(
                    _schur_theta(bd_mat, ti2, ni2))), 0.0))
                ev_w = float(max(np.min(np.linalg.eigvalsh(
                    _schur_theta(w_mat, ti2, ni2))), 0.0))
                add_art["whitened_excess_frobenius"] = excess
                add_art["theta_min_eig_block_diagonal"] = ev_bd
                add_art["theta_min_eig_whitened"] = ev_w
                add_metrics.append(
                    Metric(
                        name="additivity.joint_content_beyond_sum",
                        value=excess, kind="identifiability", exact=True,
                        threshold=0.0, direction="greater_is_better",
                        note=(
                            "||I_joint^whitened - I_joint^block-diagonal|| / ||I_joint||: "
                            "the information carried by the EEG/BOLD cross-covariance from "
                            "shared process noise. THIS is the falsifiable part of the "
                            "fusion claim; zero here means typed fusion adds nothing beyond "
                            "adding up the modalities."
                        ),
                    )
                )
                add_metrics.append(
                    Metric(
                        name="additivity.theta_min_eig_whitened_over_block_diagonal",
                        value=float(ev_w / ev_bd) if ev_bd > 1e-12 else float("nan"),
                        kind="identifiability", exact=True,
                        note=f"basis={basis}; {ev_bd:.6g} -> {ev_w:.6g} on the theta "
                             "Schur complement",
                    )
                )
            except Exception as exc:  # pragma: no cover - backend-specific
                add_art["whitened_error"] = f"{type(exc).__name__}: {exc}"

        artifacts["additivity"] = add_art
        if add_metrics:
            subs.append(
                SubCheck(
                    name="modality_additivity_declaration",
                    description=(
                        "Declares which comparisons in this report are identities and which "
                        "can fail. Under block-diagonal T4, joint = sum of modalities "
                        "identically; only the whitened excess and the design contrasts are "
                        "falsifiable."
                    ),
                    metrics=add_metrics,
                    mandatory=False,
                    reason=(
                        "reported so that no reader mistakes I_joint >= I_single for a "
                        "result; it is arithmetic"
                    ),
                    falsified_by=(
                        "the backend is not in the declared form, or whitened joint "
                        "information adds nothing beyond the modality sum"
                    ),
                )
            )
        else:
            subs.append(
                SubCheck(
                    name="modality_additivity_declaration",
                    description="Identity-versus-hypothesis declaration for this report.",
                    metrics=[], mandatory=False, forced_status="COULD_NOT_RUN",
                    reason=(
                        "single-modality designs were not available, so the additivity "
                        "identity could not be confirmed numerically. It still holds "
                        "algebraically under block-diagonal T4: no joint-versus-single "
                        "comparison in this report is evidence for fusion."
                    ),
                )
            )

    # prospective recovery of direction / delay / gain / dose / state dependence
    needed = ("direction", "delay", "gain", "dose", "state_dependence")
    if not recovery or any(k not in recovery for k in needed):
        have = sorted(recovery or {})
        subs.append(
            could_not_run(
                "prospective_recovery",
                "Prospective recovery of direction, delay, gain, dose and state dependence.",
                f"recovery results missing for {[k for k in needed if k not in (recovery or {})]} "
                f"(have {have}); this claim's support column names all five, and a prospective "
                "perturbation dataset is required (build-order item 6 is out of scope, so this "
                "is expected to remain COULD_NOT_RUN in SC-WBD-001-beta)",
                falsified_by="parameters not recovered prospectively",
            )
        )
    else:
        metrics: list[Metric] = []
        for k in needed:
            r = recovery[k]
            true = float(r["true"])
            est = float(r["estimate"])
            lo, hi = float(r.get("lo", est)), float(r.get("hi", est))
            covered = bool(lo <= true <= hi)
            denom = max(abs(true), 1e-9)
            metrics.append(
                Metric(
                    name=f"recovery.{k}.relative_error",
                    value=abs(est - true) / denom,
                    kind="identifiability",
                    interval=Interval(abs(lo - true) / denom, abs(hi - true) / denom)
                    if hi > lo else None,
                    exact=not (hi > lo),
                    threshold=thr.max_delay_rel_error if k == "delay" else 0.30,
                    direction="less_is_better",
                )
            )
            metrics.append(
                Metric(
                    name=f"recovery.{k}.interval_covers_truth",
                    value=float(covered), kind="calibration", exact=True,
                    threshold=0.5, direction="greater_is_better",
                )
            )
        subs.append(
            SubCheck(
                name="prospective_recovery",
                description="Held-out perturbation recovery of all five named quantities.",
                metrics=metrics,
                mandatory=True,
                falsified_by="parameters not recovered prospectively, or intervals miss truth",
            )
        )

    # model discrimination under intervention
    if not model_evidence or baseline_design not in model_evidence or \
            intervention_design not in model_evidence:
        subs.append(
            could_not_run(
                "model_discrimination",
                "Does the intervention separate competing posterior model classes?",
                "no per-design model evidence supplied; the thesis falsifier 'intervention "
                "fails to distinguish posterior models' cannot be evaluated",
                falsified_by="intervention fails to distinguish posterior models",
            )
        )
    else:
        def _sep(d: str) -> float:
            v = np.array(sorted(model_evidence[d].values(), reverse=True), dtype=float)
            return float(v[0] - v[1]) if v.size >= 2 else float("nan")

        s_base, s_int = _sep(baseline_design), _sep(intervention_design)
        subs.append(
            SubCheck(
                name="model_discrimination",
                description="Log-evidence separation between competing model classes.",
                metrics=[
                    Metric(
                        name="discrimination.separation_increase",
                        value=float(s_int - s_base),
                        units="nats/obs", kind="identifiability", exact=True,
                        threshold=thr.min_model_discrimination,
                        direction="greater_is_better",
                        note=f"separation {s_base:.4g} -> {s_int:.4g}",
                    )
                ],
                mandatory=True,
                falsified_by="intervention fails to distinguish posterior models",
            )
        )
        artifacts["model_discrimination"] = {"baseline": s_base, "intervention": s_int}

    rows = [
        BaselineResult(name=baseline_design, role="passive joint design"),
        BaselineResult(name="prior-only", role="prior contribution, reported separately"),
    ]
    return ClaimReport(
        manifest=man, subchecks=subs, baselines_run=rows, artifacts=artifacts, kind="gate",
        notes=[
            f"All information matrices are reported in the {basis} basis. A condition number "
            "or an eigenvalue without a declared basis is not interpretable, so the basis is "
            "stated on every such metric.",
            "Prior contribution is removed before rank/eigenvalue comparison so a full-rank "
            "posterior cannot disguise a prior-dominated likelihood (thesis §0.3).",
            "The theta block is evaluated with the nuisance/field parameters profiled out, "
            "which is how 'adds only field-model uncertainty' is detected rather than assumed.",
            "IDENTITY, NOT RESULT: under the modality-block-diagonal form of T4, "
            "I_{EEG+BOLD} = I_EEG + I_BOLD exactly. Any 'joint beats single-modality' "
            "statement in that form is arithmetic and is not evidence for typed fusion. "
            "This gate's falsifiable comparison is "
            f"{intervention_design} versus {baseline_design}; the fusion claim's falsifiable "
            "comparisons are native-versus-resampled (G1) and the non-additive joint "
            "information that appears only under joint_whitening=True.",
            "No human stimulation protocol is implemented; prospective recovery inputs must "
            "come from an approved protocol or from simulation, and are labelled as such.",
        ],
    ).finalize()


# ==========================================================================
# G5 -- individualization improves future prediction
# ==========================================================================
def run_g5(
    *,
    train: Dataset | None = None,
    new_session: Dataset | None = None,
    unseen_task: Dataset | None = None,
    candidate: Any = None,
    baselines: Mapping[str, Any] | None = None,
    utility: Mapping[str, Any] | None = None,
    thresholds: Thresholds = Thresholds(),
    seed: int = 0,
    source_cards: Sequence[str] = (),
) -> ClaimReport:
    """G5: individualization versus anatomy-only / population / session-adapted.

    "Including the person's scan is not personalization": the ``anatomy_only``
    baseline is *mandatory* and is given the person's anatomy.  The candidate
    must beat it on the person's **future** data, otherwise the claim is that
    anatomy is informative, which is a different (and weaker) claim.

    Group overlap between ``train`` and the holdouts is expected here — the
    holdout is a new *session* or a new *task*, not a new person — so the
    harness's group-overlap refusal is disabled for this gate only, and the
    fact is recorded in the report.
    """
    thr = thresholds
    baselines = dict(baselines or {})
    required = ("population", "anatomy_only", "session_adapted")
    man = _manifest(
        "G5", seed=seed, thresholds=thr, baselines=list(required) + ["longitudinal_subject"],
        source_cards=source_cards,
        refusal_fixtures=["R07 (population/subject/session effects without centering)"],
    )
    subs: list[SubCheck] = []
    artifacts: dict[str, Any] = {}

    missing: list[str] = []
    if candidate is None:
        missing.append("individualized candidate model")
    if train is None:
        missing.append("training set")
    if new_session is None:
        missing.append("new-session holdout (the claim is about future prediction)")
    if unseen_task is None:
        missing.append("unseen-task/intervention holdout")
    for r in required:
        if r not in baselines:
            missing.append(f"mandatory baseline {r!r}")
    if missing:
        for i, m in enumerate(missing):
            subs.append(could_not_run(f"inputs[{i}]", "Required gate input", f"missing: {m}",
                                      falsified_by=CLAIMS["G5"]["falsified_by"]))
        return ClaimReport(manifest=man, subchecks=subs, kind="gate").finalize()

    assert train is not None and new_session is not None and unseen_task is not None

    arms = {"candidate": candidate, **baselines}
    res_sess = _eval_arms(arms, train, new_session, seed=seed, refuse_overlap=False)
    res_task = _eval_arms(arms, train, unseen_task, seed=seed, refuse_overlap=False)
    cand_s = res_sess["candidate"]
    base_s = {k: v for k, v in res_sess.items() if k != "candidate"}
    cand_t = res_task["candidate"]
    base_t = {k: v for k, v in res_task.items() if k != "candidate"}

    verdict = check_matched(
        cand_s.extras.get("_model"), {k: v.extras.get("_model") for k, v in base_s.items()},
        tol=thr.capacity_tol, candidate_name="individualized",
    )
    subs.append(matched_subcheck(verdict))

    sc, detail = _delta_subcheck(
        "incremental_log_score_new_session",
        "Incremental calibrated log score on a NEW SESSION of the same people.",
        cand_s, base_s, thr, seed=seed,
        falsified_by="population / anatomy-only / session-adapted baselines are equivalent",
    )
    subs.append(sc)
    artifacts["new_session_delta"] = detail

    sc2, detail2 = _delta_subcheck(
        "incremental_log_score_unseen_task",
        "Incremental calibrated log score on an unseen task or intervention.",
        cand_t, base_t, thr, seed=seed,
        falsified_by="no advantage on unseen tasks/interventions",
        label="unseen_task_delta_log_score",
    )
    subs.append(sc2)
    artifacts["unseen_task_delta"] = detail2

    subs.append(_calibration_subcheck(cand_s, base_s, thr,
                                      name="calibration_on_new_session"))

    # explicit, separately named: the scan is not the personalization
    d_anat = paired_bootstrap(
        cand_s.log_score, base_s["anatomy_only"].log_score,
        name="scan_is_not_personalization.delta_vs_anatomy_only",
        n_boot=thr.n_boot, seed=seed,
    )
    subs.append(
        SubCheck(
            name="scan_is_not_personalization",
            description=(
                "The anatomy-only baseline already contains the person's scan. The "
                "individualized model must beat it on the person's future data."
            ),
            metrics=[d_anat.metric(threshold=thr.min_delta_log_score)]
            + cand_s.calibration.metrics(prefix="candidate")[:1],
            mandatory=True,
            falsified_by="anatomy-only performs equivalently — 'including the person's scan "
                         "is not itself individualization' (§11.2)",
        )
    )

    # decision utility
    if not utility or "utility" not in utility or "chosen" not in utility:
        subs.append(
            could_not_run(
                "decision_utility",
                "Decision utility / regret against the baselines on the holdout.",
                "no decision problem supplied (utility matrix + per-arm choices); the claim's "
                "support column names decision utility explicitly, and predictive log score "
                "alone does not establish it",
                falsified_by="model-guided choices do not reduce regret",
            )
        )
    else:
        u = np.asarray(utility["utility"], dtype=float)
        chosen = utility["chosen"]
        reg_c, iv_c = decision_regret(u, np.asarray(chosen["candidate"]), seed=seed)
        base_regrets = {
            k: decision_regret(u, np.asarray(v), seed=seed)[0]
            for k, v in chosen.items() if k != "candidate"
        }
        best_base = min(base_regrets.values()) if base_regrets else float("inf")
        subs.append(
            SubCheck(
                name="decision_utility",
                description="Regret of model-guided selection versus the baselines.",
                metrics=[
                    Metric(
                        name="decision.regret", value=reg_c, kind="accuracy",
                        interval=iv_c, direction="less_is_better",
                    ),
                    Metric(
                        name="decision.regret_advantage_over_best_baseline",
                        value=float(best_base - reg_c), kind="accuracy", exact=True,
                        threshold=0.0, direction="greater_is_better",
                        note=f"baseline regrets: {base_regrets}",
                    ),
                ] + cand_s.calibration.metrics(prefix="candidate")[:1],
                mandatory=True,
                falsified_by="model-guided choices do not reduce regret",
            )
        )
        artifacts["decision"] = {"candidate_regret": reg_c, "baseline_regrets": base_regrets}

    # drift diagnostic (non-blocking, but reported)
    bias = stratified_bias(
        new_session.targets.reshape(new_session.n, -1).mean(axis=1),
        cand_s.prediction.mean.reshape(new_session.n, -1).mean(axis=1),
        new_session.strata, seed=seed,
    )
    subs.append(
        SubCheck(
            name="drift_and_subgroup_bias",
            description="Per-session/task/site bias of the individualized model (§11.2).",
            metrics=bias.metrics(prefix="individualization.bias"),
            mandatory=False,
        )
    )
    artifacts["bias_table"] = bias.table()

    rows = _baseline_rows(base_s, {k: "individualization control" for k in base_s}, seed=seed)
    return ClaimReport(
        manifest=man, subchecks=subs, baselines_run=rows, artifacts=artifacts, kind="gate",
        notes=[
            "Group overlap between train and holdout is intentional for this gate only "
            "(new session / new task, same person). Person-level generalization is tested by "
            "the participant-leakage audit in scwbd.bench.leakage, not here.",
            "A win here licenses 'the supported level of adaptation', never the phrase "
            "'individual digital twin'.",
        ],
    ).finalize()


# ==========================================================================
# runner
# ==========================================================================
def run_all_gates(config: Mapping[str, Mapping[str, Any]] | None = None,
                  *, seed: int = 0) -> list[ClaimReport]:
    """Run every gate with whatever inputs are available.

    With no configuration, every gate reports ``COULD_NOT_RUN`` naming its
    missing dependency. That is the correct output, not a placeholder.
    """
    cfg = dict(config or {})
    out = [
        run_g1(seed=seed, **dict(cfg.get("G1", {}))),
        run_g2(seed=seed, **dict(cfg.get("G2", {}))),
        run_g3(seed=seed, **dict(cfg.get("G3", {}))),
        run_g4(seed=seed, **dict(cfg.get("G4", {}))),
        run_g5(seed=seed, **dict(cfg.get("G5", {}))),
    ]
    return out
