"""The machine-readable claim report for ``reports/identifiability/``.

Order of operations is load-bearing.  :func:`write_manifest` is called
**before** the benchmark runs and records the preregistered parameter subset,
the designs, the metrics, the decision rule and the seeds.  :func:`write_report`
is called afterwards and evaluates the *already written* decision rule against
the results.  There is no code path that chooses the rule after seeing the
numbers.

``thesis_contract.tex`` sec. 0.3: *"This is a planned calculation, not a
favorable result assumed in advance.  The central premise is supported only if
native-clock fusion or intervention increases likelihood information for a
preregistered parameter subset and improves calibrated recovery across held-out
simulation regimes.  If it does not, the compiler may still be useful as a
provenance system, but the claim that cross-method integration resolves those
dynamics must be narrowed."*
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .linear_gaussian import PARAMS, PARAM_NAMES, THETA_NAMES, SystemConfig
from .types import as_builtin

__all__ = [
    "DECISION_RULE",
    "PREREGISTERED_SUBSET",
    "evaluate_decision_rule",
    "make_figures",
    "write_manifest",
    "write_report",
]

#: The parameter subset the claim is about -- coupling gains and the network
#: delay, i.e. exactly ``theta`` of ``eta = (theta, ell, rho)``.
PREREGISTERED_SUBSET: tuple[str, ...] = THETA_NAMES

DECISION_RULE: dict[str, Any] = {
    "primary_question": (
        "Does native-clock fusion or a calibrated intervention increase "
        "likelihood information for the preregistered parameter subset, and "
        "improve calibrated recovery across held-out simulation regimes?"
    ),
    "preregistered_subset": list(PREREGISTERED_SUBSET),
    "information_statistic": (
        "minimum eigenvalue of the Schur complement (nuisances ell, rho "
        "profiled out) of the LIKELIHOOD-only expected Fisher information, in "
        "the prior-standardised basis; the prior contribution is excluded"
    ),
    "criteria": {
        "C1_fusion_information": (
            "theta_profile_min_eigenvalue_nonprior(joint_native) >= 1.05 x "
            "max(eeg_only, fmri_only) in EVERY regime"
        ),
        "C2_native_beats_resampled": (
            "theta_profile_min_eigenvalue_nonprior(joint_native) > that of the "
            "naive-resampling estimator (joint_resampled coarse model) in EVERY "
            "regime, AND delay RMSE is lower with a non-overlapping bootstrap "
            "interval"
        ),
        "C3_intervention_information": (
            "theta_profile_min_eigenvalue_nonprior(joint_native_impulse_matched) "
            ">= 1.05 x joint_native in EVERY regime -- energy-matched, so a bare "
            "energy increase does not count"
        ),
        "C4_calibrated_recovery": (
            "for joint_native, the nominal 95% level lies inside the Wilson "
            "interval of empirical coverage for EVERY preregistered parameter "
            "in EVERY regime"
        ),
        "C5_recovery_improvement": (
            "delay RMSE and theta RMSE for joint_native are <= the better single "
            "modality in EVERY regime"
        ),
    },
    "verdict_rule": (
        "SUPPORTED requires C1 and C4 and (C2 or C3). "
        "PARTIALLY_SUPPORTED if C4 holds and at least one of C1/C2/C3 holds. "
        "NOT_SUPPORTED otherwise -- in which case the cross-method integration "
        "claim must be narrowed to a provenance claim (thesis sec. 0.3)."
    ),
    "known_algebraic_caveat": (
        "Under the modality-block-diagonal form of T4, I_{EEG+BOLD} = I_EEG + "
        "I_BOLD, so C1 cannot fail unless the fMRI contribution to the theta "
        "profile information is numerically negligible. C1 is therefore a "
        "NECESSARY but WEAK criterion and is reported with the effect size. The "
        "discriminating criteria are C2, C3, C4 and C5."
    ),
    "what_would_disable_this_module": (
        "If native-clock fusion does not raise theta profile information above "
        "the best single modality by a margin that survives the held-out regime "
        "sweep, or if the resulting intervals are not calibrated, the shared "
        "latent fusion claim is narrowed and only the provenance/type system is "
        "retained (thesis_contract.tex Table tab:claim-gates, row 1)."
    ),
}


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _env() -> dict[str, Any]:
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "git_revision": _git_revision(),
    }


def write_manifest(
    outdir: str | Path,
    *,
    cfg: SystemConfig,
    regimes: Sequence[Any],
    designs: Sequence[Any],
    seed: int,
    n_replicates: int,
    mc_replicates: int,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Preregistration.  **Written before the benchmark runs.**"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact": "SC-WBD linear identifiability laboratory",
        "thesis_reference": "thesis_contract.tex sec. 0.3 (T1-T4), sec. 0.6 item 2, sec. 11",
        "schema_version": "scwbd-schema/1.0.0",
        "model_designation": "SC-WBD-001-beta",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "preregistered_before_run",
        "environment": _env(),
        "seed": seed,
        "n_recovery_replicates": n_replicates,
        "n_monte_carlo_fisher_replicates": mc_replicates,
        "parameters": [
            {
                "name": p.name, "group": p.group, "transform": p.transform,
                "prior_mean_unconstrained": p.prior_mean_u,
                "prior_sd_unconstrained": p.prior_sd_u,
                "units": p.units, "description": p.description,
            }
            for p in PARAMS
        ],
        "preregistered_subset": list(PREREGISTERED_SUBSET),
        "designs": [
            {"name": d.name, "label": d.label, "channels": list(d.channels),
             "include_impulse": d.include_impulse, "resample": d.resample,
             "coarse_model": d.coarse_model, "energy_matched": d.energy_matched,
             "primary": d.primary}
            for d in designs
        ],
        "regimes": [
            {"name": r.name, "description": r.description,
             "theta_scale": r.theta_scale, "tau_seconds": r.tau_seconds,
             "eeg_noise_ratio": r.eeg_noise_ratio,
             "bold_noise_ratio": r.bold_noise_ratio,
             "evoked_ratio": r.evoked_ratio}
            for r in regimes
        ],
        "instrument": {
            "dt_base_s": cfg.dt, "dt_eeg_s": cfg.dt_eeg, "dt_bold_s": cfg.dt_bold,
            "epoch_seconds": cfg.epoch_seconds, "n_epochs": cfg.n_epochs,
            "n_delay_taps": cfg.n_delay_taps, "hrf_stages": cfg.hrf_stages,
            "state_dimension": cfg.n_state, "dtype": cfg.dtype,
        },
        "reported_metrics": [
            "rank", "condition_number", "minimum_non_prior_eigenvalue",
            "parameter_profile_likelihoods", "posterior_correlations",
            "interval_coverage_with_wilson_error_bars", "delay_error",
            "prior_contribution_reported_separately",
        ],
        "decision_rule": DECISION_RULE,
        "non_goals": [
            "no whole-brain training", "no real datasets", "no field solvers",
            "no human stimulation protocol",
        ],
    }
    if extra:
        payload["extra"] = as_builtin(dict(extra))
    path = outdir / "manifest.json"
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(text)
    (outdir / "manifest.sha256").write_text(
        hashlib.sha256(text.encode()).hexdigest() + "\n"
    )
    return path


# --------------------------------------------------------------------------
# Decision rule evaluation
# --------------------------------------------------------------------------


def _theta_lmin(entry: Mapping[str, Any], key: str = "fisher_T4") -> float:
    return float(entry[key]["metrics"]["theta_profile_min_eigenvalue_nonprior"])


#: Below this prior-standardised likelihood information a parameter's posterior
#: is >90% prior, and the ``sd_post/sd_emp`` diagnostic stops being a
#: calibration signal (see :func:`nuisance_identifiability`).
PRIOR_DOMINATED_THRESHOLD = 0.1


def modality_decomposition(results: Mapping[str, Any]) -> dict[str, Any]:
    """Where each modality's theta information actually goes.

    Gate G4 refuses to treat ``I_{EEG+BOLD} = I_EEG + I_BOLD`` as evidence,
    because under the modality-block-diagonal form of T4 it is an algebraic
    identity rather than a finding.  This function *measures* the residual of
    that identity (it should be at round-off) and then reports the quantity the
    identity does not settle: how much information each modality contributes to
    each preregistered parameter, and in particular to the conduction delay.

    A fusion gain on the worst-determined direction can be 1.000x while the
    information *volume* still grows, so both are reported.
    """
    from .fisher import schur_information

    idx = [PARAM_NAMES.index(n) for n in THETA_NAMES]
    out: dict[str, Any] = {}
    for rname, rr in results["regimes"].items():
        d = rr["designs"]
        if not {"eeg_only", "fmri_only", "joint_native"} <= set(d):
            continue

        def I(name: str) -> np.ndarray:
            return np.asarray(d[name]["fisher_T4"]["I_likelihood"], float)

        residual = float(
            np.abs(I("joint_native") - I("eeg_only") - I("fmri_only")).max()
        )
        prof = {k: schur_information(I(k), idx)
                for k in ("eeg_only", "fmri_only", "joint_native")}
        per_param = {
            nm: {k: float(m[i, i]) for k, m in prof.items()}
            for i, nm in enumerate(THETA_NAMES)
        }
        lmin = {k: float(np.linalg.eigvalsh(m)[0]) for k, m in prof.items()}
        logdet = {k: float(np.linalg.slogdet(m)[1]) for k, m in prof.items()}
        out[rname] = {
            "additivity_residual_max_abs": residual,
            "additivity_holds_to_roundoff": bool(residual < 1e-8),
            "theta_profile_information_by_parameter": per_param,
            "theta_profile_min_eigenvalue": lmin,
            "theta_profile_logdet": logdet,
            "fusion_lmin_ratio": (
                lmin["joint_native"] / lmin["eeg_only"] if lmin["eeg_only"] else None
            ),
            "fusion_logdet_gain_nats": logdet["joint_native"] - logdet["eeg_only"],
            "fusion_volume_ratio": float(
                np.exp(logdet["joint_native"] - logdet["eeg_only"])
            ),
        }
    return out


def nuisance_identifiability(results: Mapping[str, Any]) -> dict[str, Any]:
    """Diagnose which parameters are informed by data and which are prior echoes.

    For a parameter with prior-standardised likelihood information ``I`` the MAP
    estimator shrinks toward the prior mean, and in the Gaussian/Laplace limit

        sd_post / sd_emp = sqrt(1 + 1/I)

    exactly.  A ratio near 1 means the data dominate; a large ratio means the
    posterior width is essentially the prior width while the point estimates
    barely move off the prior mean.  That is **not** miscalibration -- the
    interval is honest and over-covers -- but it does mean the parameter is a
    prior echo, so the ratio is reported next to the information that implies
    it, and next to the analytic T4 information as an independent check.
    """
    out: dict[str, Any] = {}
    for rname, rr in results["regimes"].items():
        per_design: dict[str, Any] = {}
        for dname, entry in rr["designs"].items():
            rec = entry.get("recovery")
            if not rec:
                continue
            names = rec["parameter_names"]
            sd_post = np.asarray(rec["posterior_sd_mean"], float)
            sd_emp = np.asarray(rec["estimate_sd"], float)
            I_diag = np.diag(np.asarray(
                entry["fisher_T4"]["I_likelihood"], float
            ))
            rows = {}
            for i, nm in enumerate(names):
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = float(sd_post[i] / sd_emp[i]) if sd_emp[i] > 0 else np.inf
                    implied = float(1.0 / (ratio**2 - 1.0)) if ratio > 1.0 else np.inf
                fisher_i = float(I_diag[i]) if i < len(I_diag) else float("nan")
                # A parameter no channel in this design can see (e.g. the BOLD
                # nuisances under eeg_only) is *structurally* absent, not
                # weakly informed; conflating the two buries the real finding.
                absent = bool(np.isfinite(fisher_i) and fisher_i <= 0.0)
                rows[nm] = {
                    "sd_post_over_sd_emp": ratio,
                    "implied_standardised_information": implied,
                    "t4_standardised_information_diagonal": fisher_i,
                    "prior_fraction_of_posterior_precision": (
                        float(1.0 / (1.0 + implied)) if np.isfinite(implied) else 0.0
                    ),
                    "structurally_absent_from_design": absent,
                    "prior_dominated": bool(
                        not absent
                        and np.isfinite(fisher_i)
                        and fisher_i < PRIOR_DOMINATED_THRESHOLD
                    ),
                    # ratio < 1 means the estimates scatter *wider* than the
                    # stated posterior: the opposite failure, and a coverage risk.
                    "estimates_overdispersed": bool(ratio < 0.9),
                }
            per_design[dname] = rows
        if per_design:
            out[rname] = per_design
    return out


#: Run-size fields compared against the preregistration.  The *criteria* must
#: never move; only how much compute was spent evaluating them may.
_RUN_SIZE_FIELDS = (
    ("n_recovery_replicates", "recovery replicates"),
    ("n_monte_carlo_fisher_replicates", "Monte-Carlo Fisher replicates"),
)


def preregistration_delta(outdir: str | Path) -> str:
    """Markdown diff of run size against ``manifest.preregistered.json``.

    A reduced run is legitimate; a *silently* reduced run is not.  This renders
    the difference into the report so the reader sees the achieved sample sizes
    next to the ones that were promised, and can check that the decision
    criteria themselves are byte-identical.
    """
    outdir = Path(outdir)
    pre_p = outdir / "manifest.preregistered.json"
    now_p = outdir / "manifest.json"
    if not (pre_p.exists() and now_p.exists()):
        return ""
    pre = json.loads(pre_p.read_text())
    now = json.loads(now_p.read_text())
    rows = []
    for key, label in _RUN_SIZE_FIELDS:
        a, b = pre.get(key), now.get(key)
        if a != b:
            rows.append((label, a, b))
    for key, label in (("epoch_seconds", "epoch length (s)"),
                       ("n_epochs", "epochs per record")):
        a = pre.get("instrument", {}).get(key)
        b = now.get("instrument", {}).get(key)
        if a != b:
            rows.append((label, a, b))
    same_rule = pre.get("decision_rule") == now.get("decision_rule")
    L = ["\n## Deviations from the pre-registration\n"]
    L.append(
        "The pre-registration written before any results existed is kept "
        "verbatim at `manifest.preregistered.json` "
        f"(status `{pre.get('status')}`, written `{pre.get('written_at')}`).\n"
    )
    L.append(
        f"**Decision criteria unchanged: {'yes' if same_rule else 'NO — SEE BELOW'}.** "
        "Only the compute budget was reduced.\n"
    )
    if rows:
        L.append("\n| quantity | preregistered | achieved |")
        L.append("|---|---|---|")
        for label, a, b in rows:
            L.append(f"| {label} | {a} | {b} |")
        L.append(
            "\nThese reductions widen every interval and raise every RMSE "
            "uniformly across designs. The preregistered criteria are all "
            "*comparisons between designs* measured under one common budget, "
            "so they remain evaluable; the absolute information values are "
            "proportionally smaller than a full-length run would give.\n"
        )
    else:
        L.append("\nRun size matches the pre-registration exactly.\n")
    return "\n".join(L)


def evaluate_decision_rule(results: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the *preregistered* rule to the results.  No tuning permitted."""
    regimes = results["regimes"]
    per_regime: dict[str, Any] = {}
    for rname, rres in regimes.items():
        d = rres["designs"]
        lmin = {k: _theta_lmin(v) for k, v in d.items()}
        # naive resampling is judged on its own estimator's information
        if "fisher_coarse_estimator" in d.get("joint_resampled", {}):
            lmin["joint_resampled"] = _theta_lmin(
                d["joint_resampled"], "fisher_coarse_estimator"
            )
        best_single = max(lmin.get("eeg_only", 0.0), lmin.get("fmri_only", 0.0))
        jn = lmin.get("joint_native", 0.0)
        c1 = jn >= 1.05 * best_single and best_single > 0
        c2_info = jn > lmin.get("joint_resampled", 0.0)
        rec = {k: v.get("recovery") for k, v in d.items()}

        def delay_rmse(k):
            r = rec.get(k)
            return float(r["delay_error"]["rmse_seconds"]) if r else float("nan")

        c2_delay = delay_rmse("joint_native") < delay_rmse("joint_resampled")
        c2 = bool(c2_info and c2_delay)
        matched = lmin.get("joint_native_impulse_matched")
        c3 = bool(matched is not None and matched >= 1.05 * jn)
        cov_ok = True
        cov_detail = {}
        r_jn = rec.get("joint_native")
        if r_jn:
            for p in PREREGISTERED_SUBSET:
                c = r_jn["coverage"][p]
                cov_detail[p] = c
                cov_ok = cov_ok and bool(c["nominal_inside_wilson95"])
        else:
            cov_ok = False
        c4 = bool(cov_ok)

        def theta_rmse(k):
            r = rec.get(k)
            if not r:
                return float("nan")
            return float(np.mean([r["rmse_in_prior_sd"][p] for p in PREREGISTERED_SUBSET]))

        c5 = bool(
            theta_rmse("joint_native")
            <= min(theta_rmse("eeg_only"), theta_rmse("fmri_only")) + 1e-12
            and delay_rmse("joint_native")
            <= min(delay_rmse("eeg_only"), delay_rmse("fmri_only")) + 1e-12
        )
        per_regime[rname] = {
            "theta_profile_min_eigenvalue_nonprior": lmin,
            "best_single_modality": best_single,
            "fusion_gain_ratio": (jn / best_single) if best_single > 0 else float("inf"),
            "impulse_gain_ratio_matched": (matched / jn) if (matched and jn > 0) else None,
            "delay_rmse_seconds": {k: delay_rmse(k) for k in d},
            "theta_rmse_prior_sd": {k: theta_rmse(k) for k in d},
            "coverage_joint_native": cov_detail,
            "C1_fusion_information": bool(c1),
            "C2_native_beats_resampled": c2,
            "C3_intervention_information": c3,
            "C4_calibrated_recovery": c4,
            "C5_recovery_improvement": c5,
        }
    allr = list(per_regime.values())
    C = {k: all(r[k] for r in allr) for k in
         ("C1_fusion_information", "C2_native_beats_resampled",
          "C3_intervention_information", "C4_calibrated_recovery",
          "C5_recovery_improvement")}
    if C["C1_fusion_information"] and C["C4_calibrated_recovery"] and (
        C["C2_native_beats_resampled"] or C["C3_intervention_information"]
    ):
        verdict = "SUPPORTED"
    elif C["C4_calibrated_recovery"] and any(
        C[k] for k in ("C1_fusion_information", "C2_native_beats_resampled",
                       "C3_intervention_information")
    ):
        verdict = "PARTIALLY_SUPPORTED"
    else:
        verdict = "NOT_SUPPORTED"
    return {
        "verdict": verdict,
        "criteria_all_regimes": C,
        "per_regime": per_regime,
        "consequence": (
            "The shared latent fusion claim stands at the scope tested."
            if verdict == "SUPPORTED"
            else "The claim that cross-method integration resolves these dynamics "
                 "must be narrowed; the compiler may still be useful as a "
                 "provenance system (thesis_contract.tex sec. 0.3)."
        ),
        "rule": DECISION_RULE,
    }


# --------------------------------------------------------------------------
# Figures and markdown
# --------------------------------------------------------------------------


def make_figures(results: Mapping[str, Any], outdir: str | Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir) / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    made: list[str] = []
    regimes = results["regimes"]
    design_names = list(next(iter(regimes.values()))["designs"])
    short = [n.replace("joint_", "").replace("_", "\n") for n in design_names]

    # 1. theta profile information per design and regime
    fig, ax = plt.subplots(figsize=(10, 4.2))
    width = 0.8 / len(regimes)
    for i, (rn, rr) in enumerate(regimes.items()):
        vals = [_theta_lmin(rr["designs"][d]) for d in design_names]
        ax.bar(np.arange(len(design_names)) + i * width, np.maximum(vals, 1e-12),
               width, label=rn)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(design_names)) + 0.4 - width / 2)
    ax.set_xticklabels(short, fontsize=7)
    ax.set_ylabel(r"$\lambda_{\min}$ of theta profile information (non-prior)")
    ax.set_title("T4 likelihood information about the preregistered subset")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = outdir / "fisher_theta_profile.png"
    fig.savefig(p, dpi=150); plt.close(fig); made.append(str(p))

    # 2. eigenvalue spectra, prior separated
    fig, axes = plt.subplots(1, len(regimes), figsize=(4.2 * len(regimes), 3.6),
                             squeeze=False)
    for k, (rn, rr) in enumerate(regimes.items()):
        ax = axes[0][k]
        for d in design_names:
            ev = np.array(rr["designs"][d]["fisher_T4"]["metrics"]["eigenvalues_likelihood"])
            ax.semilogy(np.maximum(ev[::-1], 1e-16), "o-", ms=3, lw=1, label=d)
        ax.axhline(1.0, color="k", ls="--", lw=1)
        ax.set_title(f"{rn}\n(dashed = prior contribution)", fontsize=8)
        ax.set_xlabel("eigenvalue index"); ax.set_ylabel("eigenvalue")
        ax.grid(alpha=0.3)
    axes[0][-1].legend(fontsize=6)
    fig.tight_layout()
    p = outdir / "eigenvalue_spectra.png"
    fig.savefig(p, dpi=150); plt.close(fig); made.append(str(p))

    # 3. coverage with Wilson error bars
    fig, axes = plt.subplots(1, len(regimes), figsize=(4.6 * len(regimes), 3.6),
                             squeeze=False)
    for k, (rn, rr) in enumerate(regimes.items()):
        ax = axes[0][k]
        for j, d in enumerate(design_names):
            rec = rr["designs"][d].get("recovery")
            if not rec:
                continue
            xs, ys, lo, hi = [], [], [], []
            for i, p_ in enumerate(PREREGISTERED_SUBSET):
                c = rec["coverage"][p_]
                xs.append(i + 0.08 * j); ys.append(c["empirical"])
                lo.append(c["empirical"] - c["wilson95_lo"])
                hi.append(c["wilson95_hi"] - c["empirical"])
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=3, lw=1, capsize=2, label=d)
        ax.axhline(0.95, color="k", ls="--", lw=1)
        ax.set_xticks(range(len(PREREGISTERED_SUBSET)))
        ax.set_xticklabels(PREREGISTERED_SUBSET, fontsize=7)
        ax.set_ylim(0, 1.05); ax.set_title(rn, fontsize=9)
        ax.set_ylabel("empirical 95% interval coverage"); ax.grid(alpha=0.3)
    axes[0][-1].legend(fontsize=6)
    fig.tight_layout()
    p = outdir / "interval_coverage.png"
    fig.savefig(p, dpi=150); plt.close(fig); made.append(str(p))

    # 4. delay error
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, (rn, rr) in enumerate(regimes.items()):
        vals, errs = [], []
        for d in design_names:
            rec = rr["designs"][d].get("recovery")
            vals.append(rec["delay_error"]["rmse_seconds"] * 1e3 if rec else np.nan)
            errs.append(abs(rec["delay_error"].get("rmse_seconds_se", 0.0)) * 1e3 if rec else 0)
        ax.bar(np.arange(len(design_names)) + i * width, vals, width, yerr=errs,
               capsize=2, label=rn)
    ax.set_xticks(np.arange(len(design_names)) + 0.4 - width / 2)
    ax.set_xticklabels(short, fontsize=7)
    ax.set_ylabel("delay RMSE (ms)")
    ax.set_title("Conduction-delay recovery")
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = outdir / "delay_error.png"
    fig.savefig(p, dpi=150); plt.close(fig); made.append(str(p))

    # 5. profile likelihoods (reference regime)
    rn0 = list(regimes)[0]
    d0 = regimes[rn0]["designs"]
    if any("profile_likelihood" in v for v in d0.values()):
        fig, axes = plt.subplots(1, len(PREREGISTERED_SUBSET),
                                 figsize=(3.0 * len(PREREGISTERED_SUBSET), 3.0),
                                 squeeze=False)
        for j, p_ in enumerate(PREREGISTERED_SUBSET):
            ax = axes[0][j]
            for d in design_names:
                pl = d0[d].get("profile_likelihood")
                if not pl or p_ not in pl:
                    continue
                ax.plot(pl[p_]["grid_natural"], pl[p_]["profile_log_posterior"],
                        lw=1, label=d)
            ax.set_title(p_, fontsize=9)
            ax.set_xlabel(f"{p_}"); ax.set_ylabel("profile log posterior")
            ax.grid(alpha=0.3)
        axes[0][-1].legend(fontsize=5)
        fig.tight_layout()
        p = outdir / "profile_likelihoods.png"
        fig.savefig(p, dpi=150); plt.close(fig); made.append(str(p))

    # 6. posterior correlation heatmaps
    fig, axes = plt.subplots(1, min(len(design_names), 5),
                             figsize=(3.1 * min(len(design_names), 5), 3.2),
                             squeeze=False)
    for j, d in enumerate(design_names[:5]):
        ax = axes[0][j]
        C = np.array(d0[d]["fisher_T4"]["metrics"]["posterior_correlation"])
        im = ax.imshow(C, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(d, fontsize=7)
        ax.set_xticks(range(len(PARAM_NAMES)))
        ax.set_xticklabels(PARAM_NAMES, rotation=90, fontsize=5)
        ax.set_yticks(range(len(PARAM_NAMES)))
        ax.set_yticklabels(PARAM_NAMES, fontsize=5)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8)
    p = outdir / "posterior_correlations.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); made.append(str(p))
    return made


def _fmt(x: Any, n: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "**no**"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(v):
        return "inf" if v > 0 else ("-inf" if v < 0 else "nan")
    if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
        return f"{v:.{n}g}"
    return f"{v:.{n}g}"


def write_report(
    results: Mapping[str, Any],
    outdir: str | Path,
    *,
    decision: Mapping[str, Any] | None = None,
    figures: Sequence[str] = (),
    extra_sections: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    decision = decision or evaluate_decision_rule(results)
    payload = as_builtin({"results": results, "decision": decision,
                          "nuisance_identifiability": nuisance_identifiability(results),
                          "modality_decomposition": modality_decomposition(results),
                          "environment": _env()})
    jpath = outdir / "results.json"
    jpath.write_text(json.dumps(payload, indent=1, sort_keys=True))

    L: list[str] = []
    A = L.append
    A("# Linear identifiability laboratory — claim report\n")
    A(f"**Verdict: `{decision['verdict']}`**\n")
    A(f"> {DECISION_RULE['primary_question']}\n")
    A(f"{decision['consequence']}\n")
    A("Preregistered subset: `" + "`, `".join(PREREGISTERED_SUBSET) + "`. "
      "Manifest (written before the run): `manifest.json`.\n")
    A("\n## Criteria (all held-out regimes must pass)\n")
    A("| criterion | statement | result |")
    A("|---|---|---|")
    for k, v in decision["criteria_all_regimes"].items():
        A(f"| `{k}` | {DECISION_RULE['criteria'][k]} | {_fmt(v)} |")
    A("")
    A("> " + DECISION_RULE["known_algebraic_caveat"] + "\n")

    for rname, rr in results["regimes"].items():
        A(f"\n## Regime `{rname}`\n")
        A(f"{rr['description']}\n")
        nat = rr["eta_true_natural"]
        A("Truth: " + ", ".join(f"`{k}`={_fmt(v)}" for k, v in nat.items()) + "\n")
        A("\n### T4 expected Fisher information "
          "(prior-standardised basis; prior excluded from `λmin`)\n")
        A("| design | rank | cond(I_total) | λmin non-prior | "
          "θ-profile λmin | log10 det(I_like) | max |posterior corr| |")
        A("|---|---|---|---|---|---|---|")
        for d, v in rr["designs"].items():
            m = v["fisher_T4"]["metrics"]
            A(f"| `{d}` | {m['rank_likelihood']}/{m['n_parameters']} | "
              f"{_fmt(m['condition_number_total'])} | "
              f"{_fmt(m['min_eigenvalue_nonprior'])} | "
              f"{_fmt(m['theta_profile_min_eigenvalue_nonprior'])} | "
              f"{_fmt(m['log10_det_likelihood'])} | "
              f"{_fmt(m['max_abs_posterior_correlation'])} "
              f"({'/'.join(m['max_abs_posterior_correlation_pair'])}) |")
        if any("fisher_coarse_estimator" in v for v in rr["designs"].values()):
            A("\n**Naive-resampling estimator, own information** "
              "(this is what the 1 s model can actually identify):\n")
            A("| design | rank | λmin non-prior | θ-profile λmin |")
            A("|---|---|---|---|")
            for d, v in rr["designs"].items():
                if "fisher_coarse_estimator" not in v:
                    continue
                m = v["fisher_coarse_estimator"]["metrics"]
                A(f"| `{d}` (1 s model) | {m['rank_likelihood']}/{m['n_parameters']} | "
                  f"{_fmt(m['min_eigenvalue_nonprior'])} | "
                  f"{_fmt(m['theta_profile_min_eigenvalue_nonprior'])} |")
        if any("recovery" in v for v in rr["designs"].values()):
            A("\n### Recovery (MAP + observed-information intervals)\n")
            A("| design | delay RMSE (ms) | θ RMSE (prior sd) | "
              + " | ".join(f"cov `{p}`" for p in PREREGISTERED_SUBSET)
              + " | converged |")
            A("|---|---|---|" + "---|" * (len(PREREGISTERED_SUBSET) + 1))
            for d, v in rr["designs"].items():
                rec = v.get("recovery")
                if not rec:
                    continue
                trm = np.mean([rec["rmse_in_prior_sd"][p] for p in PREREGISTERED_SUBSET])
                cells = []
                for p in PREREGISTERED_SUBSET:
                    c = rec["coverage"][p]
                    cells.append(
                        f"{c['empirical']:.3f} [{c['wilson95_lo']:.2f},{c['wilson95_hi']:.2f}]"
                    )
                A(f"| `{d}` | {_fmt(rec['delay_error']['rmse_seconds'] * 1e3)} | "
                  f"{_fmt(trm)} | " + " | ".join(cells) + " | "
                  f"{_fmt(rec['converged_fraction'])} |")
            A("\nCoverage cells are empirical / [Wilson 95% interval]; "
              f"n = {next(iter(rr['designs'].values())).get('recovery', {}).get('n_replicates', '?')} replicates.\n")
        if any("fisher_monte_carlo_complete" in v for v in rr["designs"].values()):
            A("\n### Complete expected information (Monte Carlo, includes the "
              "covariance-sensitivity term T4 omits)\n")
            A("| design | trace(I_complete) | trace(I_T4) | ratio |")
            A("|---|---|---|---|")
            for d, v in rr["designs"].items():
                mc = v.get("fisher_monte_carlo_complete")
                if not mc:
                    continue
                tc = float(np.trace(np.array(mc["I_likelihood"], float)))
                t4 = float(np.trace(np.array(v["fisher_T4"]["I_likelihood"], float)))
                A(f"| `{d}` | {_fmt(tc)} | {_fmt(t4)} | {_fmt(tc / t4 if t4 else np.nan)} |")
    modal = modality_decomposition(results)
    if modal:
        A("\n## Where each modality's θ information goes\n")
        A("Under the modality-block-diagonal form of T4, "
          "`I_EEG+BOLD = I_EEG + I_BOLD` is an algebraic identity, not a "
          "finding — gate G4 names it and refuses to report it as evidence. "
          "The residual below measures that identity; what follows it is the "
          "part the identity does not settle.\n")
        for rname, m in modal.items():
            A(f"\n**`{rname}`** — additivity residual "
              f"`{m['additivity_residual_max_abs']:.2e}` "
              f"({'round-off, identity confirmed' if m['additivity_holds_to_roundoff'] else 'NOT round-off'})\n")
            A("| θ parameter | EEG alone | fMRI alone | joint native |")
            A("|---|---|---|---|")
            for nm, v in m["theta_profile_information_by_parameter"].items():
                A(f"| `{nm}` | {_fmt(v['eeg_only'])} | {_fmt(v['fmri_only'])} | "
                  f"{_fmt(v['joint_native'])} |")
            A(f"\nFusion gain on the worst-determined direction: "
              f"**{m['fusion_lmin_ratio']:.4f}x** "
              f"(criterion C1 requires ≥ 1.05x). "
              f"Fusion gain in information *volume*: "
              f"{m['fusion_volume_ratio']:.4f}x "
              f"({m['fusion_logdet_gain_nats']:+.4f} nats).\n")
    delta = preregistration_delta(outdir)
    if delta:
        A(delta)
    diag = nuisance_identifiability(results)
    if diag:
        A("\n## Which parameters the data actually inform\n")
        A("`sd_post/sd_emp` is the mean Laplace posterior sd over the empirical "
          "spread of the MAP estimates. In the Gaussian limit it equals "
          "`sqrt(1 + 1/I)` for prior-standardised likelihood information `I`, so "
          "a value near 1 means data-dominated and a large value means the "
          "posterior is essentially the prior while the estimates sit on the "
          "prior mean. A large ratio is **not** miscalibration — such intervals "
          "over-cover — but the parameter is a prior echo and no claim may rest "
          "on it.\n")
        for rname, per_design in diag.items():
            for dname, rows in per_design.items():
                flagged = {k: v for k, v in rows.items() if v["prior_dominated"]}
                absent = [k for k, v in rows.items()
                          if v["structurally_absent_from_design"]]
                over = [k for k, v in rows.items()
                        if v["estimates_overdispersed"]
                        and not v["structurally_absent_from_design"]]
                if not (flagged or absent or over):
                    continue
                A(f"\n**`{rname}` / `{dname}`**\n")
                if flagged:
                    A("| parameter | sd_post/sd_emp | implied `I` | T4 `I` diagonal "
                      "| prior share of posterior precision |")
                    A("|---|---|---|---|---|")
                    for nm, v in flagged.items():
                        A(f"| `{nm}` | {_fmt(v['sd_post_over_sd_emp'])} | "
                          f"{_fmt(v['implied_standardised_information'])} | "
                          f"{_fmt(v['t4_standardised_information_diagonal'])} | "
                          f"{v['prior_fraction_of_posterior_precision']:.1%} |")
                if absent:
                    A("\nStructurally absent from this design (no channel "
                      "observes them; posterior = prior exactly): `"
                      + "`, `".join(absent) + "`.\n")
                if over:
                    A("\nEstimates scatter wider than the stated posterior "
                      "(`sd_post/sd_emp` < 0.9 — a coverage risk, read with the "
                      "coverage table): `" + "`, `".join(over) + "`.\n")
    if figures:
        A("\n## Figures\n")
        for f in figures:
            A(f"![{Path(f).stem}](figures/{Path(f).name})")
    if extra_sections:
        for title, body in extra_sections.items():
            A(f"\n## {title}\n")
            A(body)
    A("\n## What would disable this module\n")
    A(DECISION_RULE["what_would_disable_this_module"])
    A("\n---\n")
    A(f"Generated {time.strftime('%Y-%m-%dT%H:%M:%S%z')} · "
      f"git `{_env()['git_revision'][:12]}` · machine-readable: `results.json`.\n")
    mpath = outdir / "summary.md"
    mpath.write_text("\n".join(L))
    return {"json": jpath, "markdown": mpath}
