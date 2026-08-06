"""Appendix D Table ``tab:mixture-evaluation`` as executable audits (agent J).

All twelve rows are implemented, each with the table's *mandatory
split/control*, its *primary metric*, and its *minimum interpretation* carried
into the report as the consequence of failure.

Splitting is **agent B's** (:mod:`scwbd.sources.splits`): this module consumes
:class:`~scwbd.sources.splits.GroupedSplitter` and
:func:`~scwbd.sources.splits.leakage_audit` and does not reimplement grouping.
Duplicating a splitter would mean auditing a copy of the thing under audit.

Row 10 (TMS/tFUS decision claim) is a **standing refusal**, not a check that
happens to be unimplemented: prospective human stimulation is out of scope for
SC-WBD-001-beta (no IRB, no consent, no participants), so the audit reports
``COULD_NOT_RUN`` unconditionally and says why.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from . import adapters
from .harness import Dataset, EvalResult, evaluate
from .report import (
    BaselineResult,
    ClaimManifest,
    ClaimReport,
    Interval,
    Metric,
    SubCheck,
    could_not_run,
)
from .statistics import bootstrap_ci, paired_bootstrap
from .gates import Thresholds, _NON_GOALS, run_g2, run_g3, run_g5

__all__ = [
    "APPENDIX_D_ROWS",
    "audit_participant_family_leakage",
    "audit_stimulus_memorization",
    "audit_site_device_shortcuts",
    "audit_derived_data_duplication",
    "audit_scale_hallucination",
    "audit_teacher_simulator_domination",
    "audit_connectome_prior_value",
    "audit_operator_mechanism_claim",
    "audit_individualization_claim",
    "audit_tms_tfus_decision_claim",
    "audit_language_person_model_claim",
    "audit_dataset_family_breadth",
    "run_all_audits",
]


# --------------------------------------------------------------------------
# Appendix D rows, verbatim
# --------------------------------------------------------------------------
APPENDIX_D_ROWS: dict[str, dict[str, str]] = {
    "D01_participant_family_leakage": {
        "failure_mode": "Participant or family leakage",
        "control": ("Group all sessions, derivatives, relatives and duplicate archive records "
                    "before splitting"),
        "metric": "Held-out-person likelihood, calibration and retrieval/leakage audit",
        "interpretation": ("Within-session prediction cannot support individual "
                           "generalization"),
    },
    "D02_stimulus_memorization": {
        "failure_mode": "Stimulus memorization",
        "control": ("Hold out stimuli, semantic families and temporal continuations separately "
                    "from participant holdout"),
        "metric": "Cross-stimulus neural/behavioral forecast and matched-feature baseline",
        "interpretation": ("Image/audio recognition gain is not evidence for brain dynamics "
                           "unless it predicts new measured responses"),
    },
    "D03_site_device_shortcuts": {
        "failure_mode": "Site/device shortcuts",
        "control": ("Leave-site/device/protocol-out evaluation; nuisance-only classifier and "
                    "label permutation within site"),
        "metric": "Domain calibration, worst-site error and residual site predictability",
        "interpretation": "High pooled accuracy with poor new-device calibration is failure",
    },
    "D04_derived_data_duplication": {
        "failure_mode": "Derived-data duplication",
        "control": ("Keep raw scan and every tractogram, parcellation, preprocessing "
                    "derivative or augmentation in one split"),
        "metric": "Hash/lineage audit and performance after deduplication",
        "interpretation": ("Different algorithms over one scan are not independent "
                           "participants"),
    },
    "D05_scale_hallucination": {
        "failure_mode": "Scale hallucination",
        "control": ("Withhold fine-scale evidence while retaining coarse data; compare "
                    "uncertainty and reconstruction to a coarse-only model"),
        "metric": "Coverage and error at each native scale; high-frequency energy calibration",
        "interpretation": ("Fine detail is valid only where source support or tested prior "
                           "justifies it"),
    },
    "D06_teacher_simulator_domination": {
        "failure_mode": "Teacher/simulator domination",
        "control": ("No-teacher/no-simulator, generic-feature, shuffled, parameter-perturbed "
                    "and empirical-only ablations"),
        "metric": ("Measured held-out data likelihood and calibration, never teacher agreement "
                   "alone"),
        "interpretation": ("Distillation is retained only when it improves empirical "
                           "prediction beyond matched computation"),
    },
    "D07_connectome_prior_value": {
        "failure_mode": "Connectome prior value",
        "control": ("Randomized, distance-matched, dense, graph-only, local-only and soft-edge "
                    "controls at matched parameter/compute budgets"),
        "metric": "Data efficiency, causal forecast, calibration and out-of-domain behavior",
        "interpretation": ("Sparsity or plausibility alone is not evidence for the declared "
                           "topology"),
    },
    "D08_operator_mechanism_claim": {
        "failure_mode": "Operator / mechanism claim",
        "control": ("Equal-capacity generic operator, alternate mechanism and learned-residual "
                    "controls; hold out discriminating perturbations"),
        "metric": "Timing, direction, dose/state dependence and unique intervention forecast",
        "interpretation": ("A mechanistic label is earned only by predictions a generic "
                           "surrogate misses"),
    },
    "D09_individualization_claim": {
        "failure_mode": "Individualization claim",
        "control": ("Population, session-adapted, anatomy-only and longitudinal-person models; "
                    "new session and novel task/intervention holdouts"),
        "metric": "Incremental log score, calibration, decision utility and drift",
        "interpretation": ("Including a person's scan is not personalization unless it "
                           "improves their future predictions beyond population and session "
                           "baselines"),
    },
    "D10_tms_tfus_decision_claim": {
        "failure_mode": "TMS/tFUS decision claim",
        "control": ("Prospective randomized or otherwise causally identified target/protocol "
                    "comparison with field, pose, state and sham records"),
        "metric": "Directional response, dose--response, benefit/risk and decision regret",
        "interpretation": ("Offline reconstruction supports target hypotheses, not wellness or "
                           "treatment efficacy"),
    },
    "D11_language_person_model_claim": {
        "failure_mode": "Language/person-model claim",
        "control": ("Hold out unfamiliar situations, future time windows and private facts; "
                    "compare generic LLM, language-history-only and multimodal person models"),
        "metric": "Prospective choice/report/action calibration and counterfactual consistency",
        "interpretation": ("Stylistic imitation or fact recall is not causal fidelity, "
                           "consciousness or personal identity"),
    },
    "D12_dataset_family_breadth": {
        "failure_mode": "Dataset-family breadth",
        "control": ("Report performance by empirical, boundary-only, calibration, synthetic and "
                    "evaluation-only source roles; remove each family in turn"),
        "metric": ("Per-family contribution, negative transfer, subgroup worst case and "
                   "uncertainty coverage"),
        "interpretation": ("A longer source list is useful only when each family improves a "
                           "specified port or exposes a failure"),
    },
}


def _manifest(row: str, *, seed: int, thresholds: Thresholds,
              baselines: Sequence[str] = (), source_cards: Sequence[str] = (),
              refusal_fixtures: Sequence[str] = ()) -> ClaimManifest:
    r = APPENDIX_D_ROWS[row]
    return ClaimManifest(
        claim_id=row,
        claim_text=f"{r['failure_mode']} is controlled for. Primary metric: {r['metric']}.",
        falsified_by=(
            f"The mandatory control ({r['control']}) shows the result survives only without it."
        ),
        consequence_if_failed=f"Minimum interpretation enforced: {r['interpretation']}",
        thesis_reference="appendix.tex Appendix D, tab:mixture-evaluation",
        baselines=list(baselines),
        acceptance_thresholds=thresholds.as_dict(),
        permitted_source_cards=list(source_cards),
        refusal_fixtures=list(refusal_fixtures),
        non_goals=list(_NON_GOALS),
        seed=seed,
    )


def _splitter():
    """Agent B's splitter, or a loud absence."""
    dep = adapters.probe("scwbd.sources.splits")
    return dep


def _delegate(row: str, gate_report: ClaimReport, *, gate_name: str,
              thresholds: Thresholds, seed: int) -> ClaimReport:
    """Wrap a gate's result as the Appendix D row that it implements."""
    man = _manifest(row, seed=seed, thresholds=thresholds,
                    baselines=list(gate_report.manifest.baselines))
    return ClaimReport(
        manifest=man,
        subchecks=list(gate_report.subchecks),
        baselines_run=list(gate_report.baselines_run),
        artifacts=dict(gate_report.artifacts),
        kind="leakage",
        notes=[f"Delegated to gate {gate_name}; this Appendix D row and that gate are the "
               f"same experiment and must not be double-counted as two pieces of evidence."]
        + list(gate_report.notes),
    ).finalize()


# ==========================================================================
# D01 participant / family leakage
# ==========================================================================
def audit_participant_family_leakage(
    *,
    records: Sequence[Any] | None = None,
    n_folds: int = 5,
    train: Dataset | None = None,
    test: Dataset | None = None,
    model: Any = None,
    train_embeddings: np.ndarray | None = None,
    test_embeddings: np.ndarray | None = None,
    duplicate_similarity: float = 0.999,
    max_duplicate_rate: float = 0.01,
    thresholds: Thresholds = Thresholds(),
    seed: int = 0,
) -> ClaimReport:
    """Group-before-split, then re-audit the split, then score held-out *people*."""
    thr = thresholds
    man = _manifest("D01_participant_family_leakage", seed=seed, thresholds=thr,
                    baselines=["within-session prediction (the thing that must not be claimed)"],
                    refusal_fixtures=["R10 (derived records crossing a parent-level holdout)"])
    subs: list[SubCheck] = []
    artifacts: dict[str, Any] = {}

    dep = _splitter()
    if not dep.available or records is None:
        subs.append(
            could_not_run(
                "grouped_split",
                "Group all sessions, derivatives, relatives and duplicates before splitting.",
                (dep.blocker if not dep.available else
                 "no lineage records supplied; grouping cannot be verified and R10 forbids "
                 "splitting with unresolved parentage"),
                falsified_by="a group appears on both sides of the holdout",
            )
        )
    else:
        splits = dep.obj
        try:
            split = splits.GroupedSplitter(mode="participant", n_folds=n_folds,
                                           seed=seed).split(records)
            rep = splits.leakage_audit(split, records)
        except Exception as exc:
            subs.append(
                could_not_run(
                    "grouped_split",
                    "Group-before-split via agent B's GroupedSplitter.",
                    f"splitter raised {type(exc).__name__}: {exc}",
                    falsified_by="a group appears on both sides of the holdout",
                )
            )
        else:
            artifacts["split"] = {
                "mode": split.mode, "level": split.level, "n_folds": len(split),
                "violations": [str(v) for v in rep.violations],
                "warnings": list(rep.warnings),
                "stats": {k: str(v) for k, v in rep.stats.items()},
            }
            subs.append(
                SubCheck(
                    name="grouped_split",
                    description="Agent B's grouped split plus its own leakage audit.",
                    metrics=[
                        Metric(name="leakage.violations", value=float(len(rep.violations)),
                               kind="audit", exact=True, threshold=0.5,
                               direction="less_is_better",
                               note="; ".join(str(v) for v in rep.violations) or "none"),
                        Metric(name="leakage.warnings", value=float(len(rep.warnings)),
                               kind="diagnostic", exact=True,
                               note="; ".join(rep.warnings) or "none"),
                    ],
                    mandatory=True,
                    falsified_by="any grouping violation (R10)",
                )
            )

    # held-out-person likelihood and calibration
    if train is None or test is None or model is None:
        subs.append(
            could_not_run(
                "held_out_person_likelihood",
                "Likelihood and calibration on people never seen in training.",
                "no model/train/test supplied; within-session prediction cannot substitute "
                "for held-out-person generalization",
                falsified_by="held-out-person likelihood collapses to the marginal",
            )
        )
    else:
        res = evaluate(model, train, test, seed=seed, refuse_group_overlap=True)
        pt, iv = bootstrap_ci(res.log_score, seed=seed, n_boot=thr.n_boot)
        subs.append(
            SubCheck(
                name="held_out_person_likelihood",
                description="Held-out-person log score and calibration.",
                metrics=[
                    Metric(name="heldout_person.log_score", value=pt, units="nats/obs",
                           kind="accuracy", interval=iv),
                    *res.calibration.metrics(prefix="heldout_person"),
                ],
                mandatory=True,
                falsified_by="held-out-person calibration outside tolerance",
            )
        )
        artifacts["heldout_person"] = {"log_score": pt, "ci": [iv.lo, iv.hi]}

    # retrieval / near-duplicate audit
    if train_embeddings is None or test_embeddings is None:
        subs.append(
            could_not_run(
                "retrieval_audit",
                "Nearest-neighbour retrieval of training records from held-out records.",
                "no embeddings supplied; near-duplicate archive records (the same scan under "
                "two accession numbers) cannot be detected by lineage alone",
                falsified_by="held-out records retrieve near-identical training records",
            )
        )
    else:
        A = np.asarray(train_embeddings, dtype=float)
        B = np.asarray(test_embeddings, dtype=float)
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        sim = B @ A.T
        nn = sim.max(axis=1)
        dup = (nn >= duplicate_similarity).astype(float)
        pt, iv = bootstrap_ci(dup, seed=seed, n_boot=thr.n_boot)
        subs.append(
            SubCheck(
                name="retrieval_audit",
                description="Fraction of held-out records with a near-identical training record.",
                metrics=[
                    Metric(name="retrieval.near_duplicate_rate", value=pt, kind="audit",
                           interval=iv, threshold=max_duplicate_rate,
                           direction="less_is_better",
                           note=f"cosine >= {duplicate_similarity}"),
                    Metric(name="retrieval.max_similarity", value=float(nn.max()),
                           kind="diagnostic", exact=True),
                ],
                mandatory=True,
                falsified_by="held-out records retrieve near-identical training records",
            )
        )

    return ClaimReport(manifest=man, subchecks=subs, artifacts=artifacts, kind="leakage",
                       notes=["Splitting is agent B's; this audit consumes it rather than "
                              "reimplementing it."]).finalize()


# ==========================================================================
# D02 stimulus memorization
# ==========================================================================
def audit_stimulus_memorization(
    *,
    records: Sequence[Any] | None = None,
    train: Dataset | None = None,
    cross_stimulus_test: Dataset | None = None,
    model: Any = None,
    matched_feature_baseline: Any = None,
    thresholds: Thresholds = Thresholds(),
    seed: int = 0,
) -> ClaimReport:
    """Hold out stimuli separately from participants; beat a matched-feature baseline."""
    thr = thresholds
    man = _manifest("D02_stimulus_memorization", seed=seed, thresholds=thr,
                    baselines=["matched_feature_baseline"])
    subs: list[SubCheck] = []
    artifacts: dict[str, Any] = {}

    dep = _splitter()
    if not dep.available or records is None:
        subs.append(
            could_not_run(
                "stimulus_holdout",
                "Stimulus / semantic-family / temporal-continuation holdout.",
                (dep.blocker if not dep.available else
                 "no records with stimulus_ids supplied; stimulus holdout is not verifiable"),
                falsified_by="stimuli cross the holdout",
            )
        )
    else:
        try:
            split = dep.obj.GroupedSplitter(mode="stimulus", n_folds=3, seed=seed).split(records)
            rep = dep.obj.leakage_audit(split, records)
        except Exception as exc:
            subs.append(
                could_not_run("stimulus_holdout", "Stimulus holdout via agent B's splitter.",
                              f"splitter raised {type(exc).__name__}: {exc}",
                              falsified_by="stimuli cross the holdout")
            )
        else:
            artifacts["stimulus_split"] = {
                "n_folds": len(split), "requires_trial_masking": split.requires_trial_masking,
                "violations": [str(v) for v in rep.violations],
            }
            subs.append(
                SubCheck(
                    name="stimulus_holdout",
                    description="Stimuli, semantic families and continuations held out.",
                    metrics=[Metric(name="stimulus.violations",
                                    value=float(len(rep.violations)), kind="audit", exact=True,
                                    threshold=0.5, direction="less_is_better",
                                    note="; ".join(str(v) for v in rep.violations) or "none")],
                    mandatory=True,
                    falsified_by="stimuli cross the holdout",
                )
            )

    if train is None or cross_stimulus_test is None or model is None or \
            matched_feature_baseline is None:
        subs.append(
            could_not_run(
                "cross_stimulus_forecast",
                "Cross-stimulus forecast against a matched-feature baseline.",
                "model and/or matched-feature baseline not supplied; recognition gain on seen "
                "stimuli is not evidence for brain dynamics",
                falsified_by="a matched-feature baseline predicts equally well",
            )
        )
    else:
        r_model = evaluate(model, train, cross_stimulus_test, seed=seed,
                           refuse_group_overlap=False)
        r_base = evaluate(matched_feature_baseline, train, cross_stimulus_test, seed=seed,
                          refuse_group_overlap=False)
        d = paired_bootstrap(r_model.log_score, r_base.log_score,
                             name="cross_stimulus.delta_vs_matched_features",
                             n_boot=thr.n_boot, seed=seed)
        subs.append(
            SubCheck(
                name="cross_stimulus_forecast",
                description="Held-out-stimulus forecast beyond a matched-feature baseline.",
                metrics=[d.metric(threshold=thr.min_delta_log_score)]
                + r_model.calibration.metrics(prefix="cross_stimulus")[:1],
                mandatory=True,
                falsified_by="a matched-feature baseline predicts equally well",
            )
        )
        artifacts["cross_stimulus_delta"] = {"mean": d.mean,
                                             "ci": [d.interval.lo, d.interval.hi]}

    return ClaimReport(manifest=man, subchecks=subs, artifacts=artifacts,
                       kind="leakage").finalize()


# ==========================================================================
# D03 site / device shortcuts
# ==========================================================================
def _nuisance_classifier_auc(nuisance: np.ndarray, labels: np.ndarray, *, seed: int) -> float:
    """Can a classifier predict the label from *nuisance only*?  0.5 = cannot."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict

    X = np.asarray(nuisance, dtype=float).reshape(len(labels), -1)
    y = np.asarray(labels)
    classes = np.unique(y)
    if classes.size < 2:
        return float("nan")
    yb = (y == classes[0]).astype(int)
    clf = LogisticRegression(max_iter=500)
    try:
        p = cross_val_predict(clf, X, yb, cv=min(5, int(min(np.bincount(yb)))), method="predict_proba")[:, 1]
    except Exception:
        return float("nan")
    return float(roc_auc_score(yb, p))


def audit_site_device_shortcuts(
    *,
    records: Sequence[Any] | None = None,
    per_site: Mapping[str, Mapping[str, Dataset]] | None = None,
    model_factory: Callable[[], Any] | None = None,
    nuisance_features: np.ndarray | None = None,
    nuisance_labels: np.ndarray | None = None,
    permutation_scores: Mapping[str, Sequence[float]] | None = None,
    max_nuisance_auc: float = 0.60,
    thresholds: Thresholds = Thresholds(),
    seed: int = 0,
) -> ClaimReport:
    """Leave-site-out + nuisance-only classifier + within-site label permutation.

    ``per_site`` maps a site name to ``{"train": Dataset, "test": Dataset}``
    where ``train`` excludes that site.  Worst-site error and per-site
    calibration are the primary metrics: high pooled accuracy with poor
    new-device calibration is a failure, not a caveat.
    """
    thr = thresholds
    man = _manifest("D03_site_device_shortcuts", seed=seed, thresholds=thr,
                    baselines=["nuisance-only classifier", "within-site label permutation"])
    subs: list[SubCheck] = []
    artifacts: dict[str, Any] = {}

    if not per_site or model_factory is None:
        subs.append(
            could_not_run(
                "leave_site_out",
                "Leave-site/device/protocol-out evaluation.",
                "no per-site datasets or model factory supplied; pooled accuracy alone cannot "
                "detect a site shortcut",
                falsified_by="worst-site calibration collapses on a new device",
            )
        )
    else:
        per: dict[str, dict[str, float]] = {}
        for site, dd in per_site.items():
            r = evaluate(model_factory(), dd["train"], dd["test"], seed=seed,
                         refuse_group_overlap=False)
            per[site] = {"log_score": r.mean_log_score,
                         "coverage_error": r.calibration.coverage_error,
                         "overconfidence": r.calibration.overconfidence}
        artifacts["per_site"] = per
        worst_site = max(per, key=lambda s: per[s]["coverage_error"])
        scores = np.array([per[s]["log_score"] for s in per])
        cov = np.array([per[s]["coverage_error"] for s in per])
        _, iv_s = bootstrap_ci(scores, seed=seed, n_boot=thr.n_boot)
        _, iv_c = bootstrap_ci(cov, seed=seed, n_boot=thr.n_boot)
        subs.append(
            SubCheck(
                name="leave_site_out",
                description="Worst-site error and per-site calibration under device transfer.",
                metrics=[
                    Metric(name="site.worst_coverage_error",
                           value=float(cov.max()), kind="calibration", interval=iv_c,
                           threshold=thr.max_coverage_error * 2, direction="less_is_better",
                           note=f"worst site: {worst_site}"),
                    Metric(name="site.worst_log_score", value=float(scores.min()),
                           units="nats/obs", kind="accuracy", interval=iv_s),
                    Metric(name="site.log_score_spread",
                           value=float(scores.max() - scores.min()), kind="systematic",
                           exact=True, direction="less_is_better"),
                ],
                mandatory=True,
                falsified_by="high pooled accuracy with poor new-device calibration",
            )
        )

    if nuisance_features is None or nuisance_labels is None:
        subs.append(
            could_not_run(
                "nuisance_only_classifier",
                "Can the label be predicted from nuisance (site/device) features alone?",
                "no nuisance features/labels supplied; residual site predictability is "
                "unmeasured",
                falsified_by="nuisance alone predicts the label above chance",
            )
        )
    else:
        auc = _nuisance_classifier_auc(nuisance_features, nuisance_labels, seed=seed)
        subs.append(
            SubCheck(
                name="nuisance_only_classifier",
                description="AUC of a classifier given only site/device nuisance variables.",
                metrics=[
                    Metric(name="nuisance.auc", value=auc, kind="audit", exact=True,
                           threshold=max_nuisance_auc, direction="less_is_better",
                           note="0.5 = the label is not predictable from nuisance alone")
                ],
                mandatory=True,
                falsified_by="nuisance alone predicts the label above chance",
            )
        )
        artifacts["nuisance_auc"] = auc

    if not permutation_scores or "observed" not in permutation_scores or \
            "permuted" not in permutation_scores:
        subs.append(
            could_not_run(
                "within_site_label_permutation",
                "Within-site label permutation control.",
                "no permutation scores supplied; without them, an apparent effect cannot be "
                "distinguished from site structure",
                falsified_by="permuted-within-site labels are still predicted above chance",
            )
        )
    else:
        obs = float(np.mean(permutation_scores["observed"]))
        perm = np.asarray(list(permutation_scores["permuted"]), dtype=float)
        pt, iv = bootstrap_ci(perm, seed=seed, n_boot=thr.n_boot)
        pval = float(np.mean(perm >= obs))
        subs.append(
            SubCheck(
                name="within_site_label_permutation",
                description="Observed score versus the within-site permutation null.",
                metrics=[
                    Metric(name="permutation.p_value", value=pval, kind="audit", exact=True,
                           threshold=0.05, direction="less_is_better"),
                    Metric(name="permutation.null_mean", value=pt, kind="diagnostic",
                           interval=iv),
                ],
                mandatory=True,
                falsified_by="permuted-within-site labels are still predicted above chance",
            )
        )
        artifacts["permutation"] = {"observed": obs, "null_mean": pt, "p": pval}

    return ClaimReport(manifest=man, subchecks=subs, artifacts=artifacts,
                       kind="leakage").finalize()


# ==========================================================================
# D04 derived-data duplication
# ==========================================================================
def audit_derived_data_duplication(
    *,
    records: Sequence[Any] | None = None,
    n_folds: int = 5,
    performance_with_duplicates: Sequence[float] | None = None,
    performance_after_dedup: Sequence[float] | None = None,
    thresholds: Thresholds = Thresholds(),
    seed: int = 0,
) -> ClaimReport:
    """Hash/lineage audit, plus the performance change after deduplication."""
    thr = thresholds
    man = _manifest("D04_derived_data_duplication", seed=seed, thresholds=thr,
                    refusal_fixtures=["R10"])
    subs: list[SubCheck] = []
    artifacts: dict[str, Any] = {}

    dep = _splitter()
    if not dep.available or records is None:
        subs.append(
            could_not_run(
                "hash_lineage_audit",
                "Every derivative of a scan stays in one split.",
                (dep.blocker if not dep.available else "no lineage records supplied"),
                falsified_by="a derivative crosses its parent's holdout",
            )
        )
    else:
        try:
            split = dep.obj.GroupedSplitter(mode="participant", n_folds=n_folds,
                                            seed=seed).split(records)
            rep = dep.obj.leakage_audit(split, records)
        except Exception as exc:
            subs.append(
                could_not_run("hash_lineage_audit", "Lineage/hash audit via agent B.",
                              f"splitter raised {type(exc).__name__}: {exc}",
                              falsified_by="a derivative crosses its parent's holdout")
            )
        else:
            derived = [v for v in rep.violations
                       if "deriv" in v.kind.lower() or "hash" in v.kind.lower()
                       or "duplicate" in v.kind.lower()]
            n_derived = sum(1 for r in records
                            if getattr(getattr(r, "lineage", None), "derived_from", None))
            hashes = [getattr(getattr(r, "lineage", None), "content_hash", None)
                      for r in records]
            hashes = [h for h in hashes if h]
            dup_hash = len(hashes) - len(set(hashes))
            artifacts["lineage"] = {
                "n_records": len(records), "n_derived": n_derived,
                "duplicate_content_hashes": dup_hash,
                "violations": [str(v) for v in rep.violations],
            }
            subs.append(
                SubCheck(
                    name="hash_lineage_audit",
                    description="Derivatives and duplicate-hash records grouped with parents.",
                    metrics=[
                        Metric(name="lineage.derived_violations",
                               value=float(len(derived)), kind="audit", exact=True,
                               threshold=0.5, direction="less_is_better",
                               note="; ".join(str(v) for v in derived) or "none"),
                        Metric(name="lineage.duplicate_content_hashes",
                               value=float(dup_hash), kind="audit", exact=True,
                               threshold=0.5, direction="less_is_better",
                               note="records with identical content hashes are one datum"),
                    ],
                    mandatory=True,
                    falsified_by="a derivative or archive duplicate crosses the holdout",
                )
            )

    if performance_with_duplicates is None or performance_after_dedup is None:
        subs.append(
            could_not_run(
                "performance_after_dedup",
                "Performance change once duplicates are removed.",
                "no with/without-duplicate scores supplied; the size of the inflation is "
                "unknown",
                falsified_by="performance drops materially after deduplication",
            )
        )
    else:
        a = np.asarray(list(performance_after_dedup), dtype=float)
        b = np.asarray(list(performance_with_duplicates), dtype=float)
        n = min(a.size, b.size)
        d = paired_bootstrap(a[:n], b[:n], name="dedup.delta", n_boot=thr.n_boot, seed=seed)
        subs.append(
            SubCheck(
                name="performance_after_dedup",
                description="Deduplicated performance versus the duplicated-data number.",
                metrics=[
                    Metric(name="dedup.performance_change", value=d.mean, units="nats/obs",
                           kind="audit", interval=d.interval, threshold=-0.05,
                           direction="greater_is_better",
                           note="a large drop means the reported number was inflated by "
                                "duplicated evidence"),
                ],
                mandatory=True,
                falsified_by="performance drops materially after deduplication",
            )
        )

    return ClaimReport(manifest=man, subchecks=subs, artifacts=artifacts,
                       kind="leakage").finalize()


# ==========================================================================
# D05 scale hallucination  (delegates to G3's mandatory control)
# ==========================================================================
def audit_scale_hallucination(*, thresholds: Thresholds = Thresholds(), seed: int = 0,
                              **g3_kwargs: Any) -> ClaimReport:
    rep = run_g3(thresholds=thresholds, seed=seed, **g3_kwargs)
    keep = [s for s in rep.subchecks
            if s.name in ("high_frequency_hallucination", "calibrated_refinement",
                          "boundary_agreement", "native_scale_prediction")]
    if not keep:
        keep = list(rep.subchecks)
    sub = ClaimReport(manifest=rep.manifest, subchecks=keep,
                      baselines_run=rep.baselines_run, artifacts=rep.artifacts, kind="gate")
    return _delegate("D05_scale_hallucination", sub, gate_name="G3",
                     thresholds=thresholds, seed=seed)


# ==========================================================================
# D06 teacher / simulator domination  (quarantined by default)
# ==========================================================================
def audit_teacher_simulator_domination(
    *,
    enable_quarantined: bool = False,
    arms: Mapping[str, Any] | None = None,
    train: Dataset | None = None,
    test: Dataset | None = None,
    thresholds: Thresholds = Thresholds(),
    seed: int = 0,
) -> ClaimReport:
    from .ablations import ABLATIONS, run_ablation

    man = _manifest("D06_teacher_simulator_domination", seed=seed, thresholds=thresholds,
                    baselines=list(ABLATIONS["A9_teacher_quarantined"].required_arms))
    if not enable_quarantined:
        return ClaimReport(
            manifest=man,
            subchecks=[
                could_not_run(
                    "quarantine",
                    "Teacher/simulator ablation set.",
                    "TRIBE v2 distillation stays OFF by default and is never a subject "
                    "likelihood (ARCHITECTURE.md rule 5). With the teacher disabled there is "
                    "no distillation contribution to audit, and none may be claimed.",
                    falsified_by="the teacher improves nothing measured, or dominates the loss",
                )
            ],
            kind="leakage",
            notes=["Teacher agreement is never the metric; only measured held-out data "
                   "likelihood and calibration count (Appendix D)."],
        ).finalize()
    rep = run_ablation(ABLATIONS["A9_teacher_quarantined"], train=train, test=test, arms=arms,
                       thresholds=thresholds, seed=seed, enable_quarantined=True)
    return _delegate("D06_teacher_simulator_domination", rep, gate_name="A9 ablation",
                     thresholds=thresholds, seed=seed)


# ==========================================================================
# D07 connectome prior value  (delegates to G2)
# ==========================================================================
def audit_connectome_prior_value(*, thresholds: Thresholds = Thresholds(), seed: int = 0,
                                 **g2_kwargs: Any) -> ClaimReport:
    rep = run_g2(thresholds=thresholds, seed=seed, **g2_kwargs)
    return _delegate("D07_connectome_prior_value", rep, gate_name="G2",
                     thresholds=thresholds, seed=seed)


# ==========================================================================
# D08 operator / mechanism claim
# ==========================================================================
def audit_operator_mechanism_claim(
    *,
    train: Dataset | None = None,
    test: Dataset | None = None,
    arms: Mapping[str, Any] | None = None,
    mechanism_holdout: Mapping[str, Dataset] | None = None,
    thresholds: Thresholds = Thresholds(),
    seed: int = 0,
    **kw: Any,
) -> ClaimReport:
    from .ablations import ABLATIONS, run_ablation

    rep = run_ablation(ABLATIONS["A5_typed_operators"], train=train, test=test, arms=arms,
                       mechanism_holdout=mechanism_holdout, thresholds=thresholds, seed=seed,
                       **kw)
    return _delegate("D08_operator_mechanism_claim", rep, gate_name="A5 ablation",
                     thresholds=thresholds, seed=seed)


# ==========================================================================
# D09 individualization claim  (delegates to G5)
# ==========================================================================
def audit_individualization_claim(*, thresholds: Thresholds = Thresholds(), seed: int = 0,
                                  **g5_kwargs: Any) -> ClaimReport:
    rep = run_g5(thresholds=thresholds, seed=seed, **g5_kwargs)
    return _delegate("D09_individualization_claim", rep, gate_name="G5",
                     thresholds=thresholds, seed=seed)


# ==========================================================================
# D10 TMS/tFUS decision claim  -- STANDING REFUSAL
# ==========================================================================
def audit_tms_tfus_decision_claim(*, thresholds: Thresholds = Thresholds(),
                                  seed: int = 0, **_ignored: Any) -> ClaimReport:
    """Always ``COULD_NOT_RUN``.  This is a refusal, not a missing feature.

    The control this row mandates is a *prospective, causally identified
    target/protocol comparison in humans*.  SC-WBD-001-beta's build order stops
    at item 5; item 6 is out of scope (no IRB, no consent, no participants) and
    no agent may implement a human stimulation protocol.  The audit therefore
    reports that the decision claim is unsupported and unsupportable in this
    release, whatever inputs are passed.
    """
    man = _manifest("D10_tms_tfus_decision_claim", seed=seed, thresholds=thresholds,
                    refusal_fixtures=["R11 (intervention optimization outside a validated "
                                      "A_safe)"])
    sub = could_not_run(
        "prospective_decision_comparison",
        "Prospective randomized or otherwise causally identified target/protocol comparison.",
        "OUT OF SCOPE BY CONSTRUCTION: the build order stops at item 5 (empirical subsystem); "
        "item 6 (prospective human TMS/tFUS) has no IRB, no consent and no participants, and "
        "no agent may implement a human stimulation protocol (ARCHITECTURE.md §0). No inputs "
        "can make this audit run in SC-WBD-001-beta.",
        falsified_by="any claimed wellness or treatment efficacy from offline reconstruction",
    )
    return ClaimReport(
        manifest=man, subchecks=[sub], kind="leakage",
        notes=[
            "Offline reconstruction supports target hypotheses, not wellness or treatment "
            "efficacy. Any downstream consumer (tms-robotics) must treat SC-WBD output as a "
            "prediction plus a refusal, never as a protocol.",
        ],
    ).finalize()


# ==========================================================================
# D11 language / person-model claim
# ==========================================================================
def audit_language_person_model_claim(
    *,
    train: Dataset | None = None,
    test: Dataset | None = None,
    arms: Mapping[str, Any] | None = None,
    thresholds: Thresholds = Thresholds(),
    seed: int = 0,
    **kw: Any,
) -> ClaimReport:
    from .ablations import ABLATIONS, run_ablation

    rep = run_ablation(ABLATIONS["A8_language_coupling"], train=train, test=test, arms=arms,
                       thresholds=thresholds, seed=seed, **kw)
    out = _delegate("D11_language_person_model_claim", rep, gate_name="A8 ablation",
                    thresholds=thresholds, seed=seed)
    out.notes.append(
        "No Phi estimate and no consciousness ground truth exist here (ARCHITECTURE.md rule 4); "
        "stylistic imitation or fact recall is not causal fidelity or personal identity."
    )
    return out


# ==========================================================================
# D12 dataset-family breadth
# ==========================================================================
def audit_dataset_family_breadth(
    *,
    train: Dataset | None = None,
    test: Dataset | None = None,
    model_factory: Callable[[Sequence[str]], Any] | None = None,
    families: Mapping[str, Sequence[str]] | None = None,
    roles: Mapping[str, str] | None = None,
    thresholds: Thresholds = Thresholds(),
    seed: int = 0,
) -> ClaimReport:
    """Remove each source family in turn; report contribution and negative transfer.

    ``families`` maps a family name (its source role bucket: empirical,
    boundary-only, calibration, synthetic, evaluation-only) to the input
    blocks it contributes.  ``model_factory(blocks)`` builds a model using
    exactly those blocks.
    """
    thr = thresholds
    man = _manifest("D12_dataset_family_breadth", seed=seed, thresholds=thr,
                    baselines=["leave-one-family-out"])
    subs: list[SubCheck] = []
    artifacts: dict[str, Any] = {}

    if train is None or test is None or model_factory is None or not families:
        subs.append(
            could_not_run(
                "per_family_contribution",
                "Remove each source family in turn.",
                "no families / model factory / datasets supplied; a longer source list is not "
                "evidence, so nothing may be claimed about breadth",
                falsified_by="a family contributes nothing or transfers negatively",
            )
        )
        return ClaimReport(manifest=man, subchecks=subs, kind="leakage").finalize()

    all_blocks = [b for blocks in families.values() for b in blocks]
    full = evaluate(model_factory(all_blocks), train, test, seed=seed,
                    refuse_group_overlap=False)
    contributions: dict[str, Any] = {}
    metrics: list[Metric] = []
    negative: list[str] = []
    for fam, blocks in families.items():
        kept = [b for b in all_blocks if b not in blocks]
        if not kept:
            contributions[fam] = {"skipped": "removing this family leaves no inputs"}
            continue
        r = evaluate(model_factory(kept), train, test, seed=seed, refuse_group_overlap=False)
        d = paired_bootstrap(full.log_score, r.log_score,
                             name=f"family.{fam}.contribution", n_boot=thr.n_boot, seed=seed)
        contributions[fam] = {
            "role": (roles or {}).get(fam, "unspecified"),
            "contribution": d.mean, "ci": [d.interval.lo, d.interval.hi],
            "negative_transfer": bool(d.significant_loss),
        }
        if d.significant_loss:
            negative.append(fam)
        metrics.append(
            Metric(name=f"family.{fam}.contribution", value=d.mean, units="nats/obs",
                   kind="accuracy", interval=d.interval,
                   note=f"role={contributions[fam]['role']}; negative values mean the family "
                        "hurts (negative transfer)")
        )
    metrics += full.calibration.metrics(prefix="all_families")[:1]
    metrics.append(
        Metric(name="family.negative_transfer_count", value=float(len(negative)),
               kind="audit", exact=True, threshold=0.5, direction="less_is_better",
               note=("families that hurt: " + ", ".join(negative)) if negative else "none")
    )
    artifacts["contributions"] = contributions
    subs.append(
        SubCheck(
            name="per_family_contribution",
            description="Per-family contribution and negative transfer, by source role.",
            metrics=metrics,
            mandatory=True,
            falsified_by="a family contributes nothing measurable, or transfers negatively",
        )
    )

    # subgroup worst case
    if test.strata:
        worst_metrics: list[Metric] = []
        for factor, labels in test.strata.items():
            lab = np.asarray(labels)
            per_level = {}
            for lv in sorted(set(lab.tolist()), key=str):
                m = lab == lv
                if m.sum() < 10:
                    continue
                per_level[str(lv)] = float(np.mean(full.log_score[m]))
            if per_level:
                worst = min(per_level, key=lambda k: per_level[k])
                worst_metrics.append(
                    Metric(name=f"subgroup.worst.{factor}", value=per_level[worst],
                           units="nats/obs", kind="systematic", exact=True,
                           note=f"worst level: {worst}")
                )
        subs.append(
            SubCheck(name="subgroup_worst_case",
                     description="Worst subgroup log score with all families present.",
                     metrics=worst_metrics, mandatory=False)
        )

    return ClaimReport(manifest=man, subchecks=subs, artifacts=artifacts,
                       kind="leakage",
                       notes=["A longer source list is useful only when each family improves "
                              "a specified port or exposes a failure."]).finalize()


# ==========================================================================
def run_all_audits(config: Mapping[str, Mapping[str, Any]] | None = None,
                   *, seed: int = 0) -> list[ClaimReport]:
    """Run every Appendix D row with whatever inputs are available."""
    cfg = dict(config or {})
    fns: dict[str, Callable[..., ClaimReport]] = {
        "D01_participant_family_leakage": audit_participant_family_leakage,
        "D02_stimulus_memorization": audit_stimulus_memorization,
        "D03_site_device_shortcuts": audit_site_device_shortcuts,
        "D04_derived_data_duplication": audit_derived_data_duplication,
        "D05_scale_hallucination": audit_scale_hallucination,
        "D06_teacher_simulator_domination": audit_teacher_simulator_domination,
        "D07_connectome_prior_value": audit_connectome_prior_value,
        "D08_operator_mechanism_claim": audit_operator_mechanism_claim,
        "D09_individualization_claim": audit_individualization_claim,
        "D10_tms_tfus_decision_claim": audit_tms_tfus_decision_claim,
        "D11_language_person_model_claim": audit_language_person_model_claim,
        "D12_dataset_family_breadth": audit_dataset_family_breadth,
    }
    return [fn(seed=seed, **dict(cfg.get(k, {}))) for k, fn in fns.items()]
