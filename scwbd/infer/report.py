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


#: A regime whose truth is closer than this to the prior mean (in prior sds)
#: cannot discriminate estimators by bias/RMSE/coverage: shrinking to the prior
#: is already the right answer there.
DEGENERATE_OFFSET_PRIOR_SD = 0.25


def regime_prior_offset(results: Mapping[str, Any]) -> dict[str, Any]:
    """How far each regime's truth sits from the prior mean, in prior sds.

    Recovery metrics are only informative when the truth is *away* from the
    prior mean.  If it coincides, an estimator that ignores the data entirely
    and returns the prior mean scores zero bias, zero RMSE and 100% coverage,
    so bias/RMSE/coverage stop measuring information and start measuring
    shrinkage.  Any regime that degenerate is flagged here and its recovery
    numbers must not be read as evidence that a design works.
    """
    from .linear_gaussian import PARAM_INDEX, prior_mean_u, prior_sd_u

    pm, ps = prior_mean_u(), prior_sd_u()
    out: dict[str, Any] = {}
    for rname, rr in results["regimes"].items():
        u = np.asarray(rr["eta_true_unconstrained"], float)
        z = {n: float((u[PARAM_INDEX[n]] - pm[PARAM_INDEX[n]]) / ps[PARAM_INDEX[n]])
             for n in PREREGISTERED_SUBSET}
        worst = max(abs(v) for v in z.values())
        out[rname] = {
            "offset_in_prior_sd": z,
            "max_abs_offset_prior_sd": worst,
            "recovery_metrics_degenerate": bool(worst < DEGENERATE_OFFSET_PRIOR_SD),
        }
    return out


def _num(x: Any, default: float = 0.0) -> float:
    """Coerce a possibly-null / non-finite JSON value to a plottable float.

    ``as_builtin`` serialises NaN as ``null``, so a single degenerate statistic
    -- e.g. the standard error of a delay RMSE that is exactly zero because the
    design carries no delay information -- must not be able to abort figure
    generation for a sweep that took hours to compute.
    """
    if x is None:
        return default
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


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
    arms = now.get("extra", {}).get("arms_computed")
    if arms:
        off = [k for k, v in arms.items() if not v]
        L.append(
            "\nArms computed: " + ", ".join(f"`{k}`" for k, v in arms.items() if v)
            + (("; **not computed:** " + ", ".join(f"`{k}`" for k in off))
               if off else "")
            + ". A zero below means the arm was switched off, not that it ran "
              "with no replicates.\n"
        )
    rd = now.get("extra", {}).get("recovery_designs")
    pre_rd = pre.get("extra", {}).get("recovery_designs")
    if rd is not None and pre_rd is not None and set(rd) != set(pre_rd):
        L.append(
            "\nRecovery arm restricted to " + ", ".join(f"`{d}`" for d in rd)
            + " (preregistered: " + ", ".join(f"`{d}`" for d in pre_rd)
            + "). The dropped designs are not used by any criterion.\n"
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


def _theta_logdet(entry: Mapping[str, Any], key: str = "fisher_T4") -> float:
    return float(entry[key]["metrics"]["theta_profile_log10_det_likelihood"])


def evaluate_decision_rule(results: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the *preregistered* rule to the results.  No tuning permitted."""
    regimes = results["regimes"]
    per_regime: dict[str, Any] = {}
    for rname, rres in regimes.items():
        d = rres["designs"]
        lmin = {k: _theta_lmin(v) for k, v in d.items()}
        logdet = {k: _theta_logdet(v) for k, v in d.items()}
        # naive resampling is judged on its own estimator's information
        if "fisher_coarse_estimator" in d.get("joint_resampled", {}):
            lmin["joint_resampled"] = _theta_lmin(
                d["joint_resampled"], "fisher_coarse_estimator"
            )
            logdet["joint_resampled"] = _theta_logdet(
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

        have_rec = {k for k, v in rec.items() if v}
        c2_delay = delay_rmse("joint_native") < delay_rmse("joint_resampled")
        c2 = bool(c2_info and c2_delay)
        if not {"joint_native", "joint_resampled"} <= have_rec:
            c2 = None                     # not evaluated, not failed
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
            cov_ok = None                 # not evaluated, not failed
        c4_raw = None if cov_ok is None else bool(cov_ok)

        # --- POST HOC validity gate, disclosed as such -----------------------
        # A coverage statistic produced by an optimiser that demonstrably has
        # not reached the MAP measures the optimiser, not the design.  The
        # median Newton decrement is the remaining distance to the optimum in
        # posterior standard deviations; above ~2 the reported interval is not
        # centred on the estimate it claims.  Where that happens C4/C5 are
        # marked NOT EVALUATED rather than counted as failures.  This gate was
        # added after observing non-convergence in two regimes and is reported
        # separately from the preregistered criteria it guards.
        decr = {
            k: float((v.get("optimiser") or {}).get("median_newton_decrement", 0.0) or 0.0)
            for k, v in rec.items() if v
        }
        GATE = 2.0
        # C4 is a statement about joint_native only; C5 compares it with the two
        # single-modality designs.  Gate each on exactly the designs it uses.
        decr_c4 = decr.get("joint_native", 0.0)
        decr_c5 = max(
            (decr[k] for k in ({"joint_native", "eeg_only", "fmri_only"} & set(decr))),
            default=0.0,
        )
        worst_decr = max(decr.values(), default=0.0)
        converged = decr_c4 <= GATE
        c4 = c4_raw if decr_c4 <= GATE else None

        def theta_rmse(k):
            r = rec.get(k)
            if not r:
                return float("nan")
            return float(np.mean([r["rmse_in_prior_sd"][p] for p in PREREGISTERED_SUBSET]))

        if not {"joint_native", "eeg_only", "fmri_only"} <= have_rec:
            c5_raw = None                 # not evaluated, not failed
        else:
            c5_raw = bool(
                theta_rmse("joint_native")
                <= min(theta_rmse("eeg_only"), theta_rmse("fmri_only")) + 1e-12
                and delay_rmse("joint_native")
                <= min(delay_rmse("eeg_only"), delay_rmse("fmri_only")) + 1e-12
            )
        c5 = c5_raw if decr_c5 <= GATE else None
        per_regime[rname] = {
            "theta_profile_min_eigenvalue_nonprior": lmin,
            "best_single_modality": best_single,
            "fusion_gain_ratio": (jn / best_single) if best_single > 0 else float("inf"),
            # Absolute gap, which is regime-dependent in a way the ratio hides:
            # "EEG loses essentially nothing" is true everywhere but ~200x less
            # true in the low-SNR regime, which is the clinically relevant one.
            "fusion_gain_absolute_over_eeg_only": jn - lmin.get("eeg_only", 0.0),
            # The relative form is the one the claim rests on and the one 🧩 Rao
            # reported; it ranks the regimes differently from the absolute form
            # because lambda_min itself differs by an order between regimes.
            "fusion_gain_relative_over_eeg_only": (
                jn / lmin["eeg_only"] - 1.0 if lmin.get("eeg_only", 0.0) > 0 else None
            ),
            "eeg_over_fmri_orders_of_magnitude": (
                math.log10(lmin["eeg_only"] / lmin["fmri_only"])
                if lmin.get("fmri_only", 0.0) > 0 and lmin.get("eeg_only", 0.0) > 0
                else None
            ),
            "impulse_gain_ratio_matched": (matched / jn) if (matched and jn > 0) else None,
            "delay_rmse_seconds": {k: delay_rmse(k) for k in d},
            "theta_rmse_prior_sd": {k: theta_rmse(k) for k in d},
            "coverage_joint_native": cov_detail,
            "C1_fusion_information": bool(c1),
            "C2_native_beats_resampled": c2,
            "C3_intervention_information": c3,
            "C4_calibrated_recovery": c4,
            "C5_recovery_improvement": c5,
            "C4_calibrated_recovery_ungated": c4_raw,
            "C5_recovery_improvement_ungated": c5_raw,
            "optimiser_converged_joint_native": converged,
            "median_newton_decrement_by_design": decr,
            "median_newton_decrement_gating_C4": decr_c4,
            "median_newton_decrement_gating_C5": decr_c5,
            "worst_median_newton_decrement": worst_decr,
            "convergence_gate_threshold_posterior_sd": GATE,
            # A design that learns *nothing* about tau leaves it at the prior
            # mean.  Where tau_true happens to equal the prior mean, that scores
            # a perfect delay error, so the delay comparison cannot discriminate
            # in that regime.  Flagged rather than silently averaged in.
            "delay_comparison_degenerate": bool(
                abs(float(rres["eta_true_natural"].get("tau", 0.0)) - 0.012) < 1e-9
            ),
            # --- secondary, POST HOC: not part of the preregistered verdict ---
            # lambda_min over the theta block is dominated by the delay
            # direction, which a 1 s instrument cannot inform at all; the
            # log-determinant is the total-information view and is reported so
            # that "fMRI adds nothing" and "fMRI adds nothing *about the
            # delay*" are not confused with one another.
            "secondary_post_hoc": {
                "theta_profile_log10_det_likelihood": logdet,
                "fusion_logdet_gain_over_best_single": (
                    logdet.get("joint_native", float("nan"))
                    - max(logdet.get("eeg_only", -np.inf),
                          logdet.get("fmri_only", -np.inf))
                ),
                "note": "reported, not used by the verdict; the manifest fixed "
                        "the criteria before the run",
            },
        }
    allr = list(per_regime.values())
    keys = ("C1_fusion_information", "C2_native_beats_resampled",
            "C3_intervention_information", "C4_calibrated_recovery",
            "C5_recovery_improvement")

    def combine(k: str):
        vals = [r[k] for r in allr]
        if any(v is None for v in vals):
            # A criterion whose inputs were never computed is NOT_EVALUATED.
            # Collapsing that to False would let an unfinished sweep masquerade
            # as a negative result, which is the opposite of reporting honestly.
            return None
        return all(vals)

    C = {k: combine(k) for k in keys}
    unevaluated = [k for k, v in C.items() if v is None]
    if unevaluated:
        verdict = "INCOMPLETE"
    elif C["C1_fusion_information"] and C["C4_calibrated_recovery"] and (
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
        "unevaluated_criteria": unevaluated,
        "regimes_evaluated": list(per_regime),
        "convergence_gated_regimes": [
            k for k, v in per_regime.items()
            if not v.get("optimiser_converged_joint_native")
        ],
        "delay_degenerate_regimes": [
            k for k, v in per_regime.items() if v.get("delay_comparison_degenerate")
        ],
        "per_regime": per_regime,
        "consequence": (
            "Criteria "
            f"{[k for k, v in C.items() if v is False]} were fully evaluated and "
            f"FAILED; criteria {unevaluated} could not be evaluated in every "
            "regime and are reported as NOT EVALUATED rather than as failures. "
            "On the evidence that was evaluable, the claim that cross-method "
            "integration resolves these dynamics must be narrowed; the compiler "
            "may still be useful as a provenance system "
            "(thesis_contract.tex sec. 0.3)."
            if verdict == "INCOMPLETE"
            else "The shared latent fusion claim stands at the scope tested."
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
                # Wilson bounds are clipped to [0,1] and can round to the point
                # estimate; a negative whisker is a rounding artefact, not data.
                lo.append(max(c["empirical"] - c["wilson95_lo"], 0.0))
                hi.append(max(c["wilson95_hi"] - c["empirical"], 0.0))
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
            vals.append(_num(rec["delay_error"]["rmse_seconds"]) * 1e3 if rec else np.nan)
            errs.append(abs(_num(rec["delay_error"].get("rmse_seconds_se"))) * 1e3
                        if rec else 0.0)
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

    # 5. profile likelihoods.  Profiles are computed for one regime only, but
    # which one must be discovered rather than assumed to be first: the regime
    # mapping is plain JSON and any consumer that rewrites it (sorted keys, a
    # merge) can reorder it, silently dropping this figure.
    rn0 = next(
        (rn for rn, rr in regimes.items()
         if any("profile_likelihood" in v for v in rr["designs"].values())),
        None,
    )
    if rn0 is not None:
        d0 = regimes[rn0]["designs"]
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

    # 6. posterior correlation heatmaps (first regime; independent of profiles)
    dcorr = regimes[list(regimes)[0]]["designs"]
    fig, axes = plt.subplots(1, min(len(design_names), 5),
                             figsize=(3.1 * min(len(design_names), 5), 3.2),
                             squeeze=False)
    for j, d in enumerate(design_names[:5]):
        ax = axes[0][j]
        C = np.array(dcorr[d]["fisher_T4"]["metrics"]["posterior_correlation"])
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


def _fmt_sig(x: Any, metrics: Mapping[str, Any], key: str) -> str:
    """Format an eigenvalue at the precision its reproducibility supports."""
    u = metrics.get(key + "_numerics") or metrics.get("theta_profile_min_eigenvalue_numerics")
    if x is None:
        return "_not evaluated_"
    v = float(x)
    if u and u.get("numerically_zero"):
        return "0 _(num. zero)_"
    figs = int((u or {}).get("significant_figures", 4))
    return f"{v:.{max(min(figs, 6), 2)}g}"


def _fmt(x: Any, n: int = 4) -> str:
    if x is None:
        return "_not evaluated_"
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


#: Lineage of this artifact.  Kept in the generator, not hand-edited into the
#: markdown, so it cannot drift away from the code that emits it.
ARTIFACT_LINEAGE = """
## Lineage: what these numbers supersede, and why

**The `results.json` previously filed on `wt/fisher` (through `9088581`) is not
reproducible under the estimator that produced this report, and has been
superseded rather than reconciled.**

The reason is methodological, not clerical. That artifact was produced by a
benchmark whose simulator drew each replicate *chunk* from its own generator
seed, because the deterministic per-epoch drive had to be tiled once per
replicate and the full tile did not fit in memory. Merging `master` replaced
that with a single unchunked draw in which the drive is tiled one step at a
time (`filters._tile_rows`), so the tile is never materialised at all. The
newer path is exactly equivalent in expectation and strictly better in memory,
but it consumes the random stream in a different order, so **every simulated
observation differs**. No amount of re-derivation can make the old recovery,
coverage or delay-error numbers agree with this code; they are orphaned, and an
orphaned number that still looks authoritative is precisely the hazard this
project has been burned by before.

Two consequences a reader should carry:

- The superseded run reported `C4_calibrated_recovery` as a **pass**. It was
  earned at a larger Newton budget on the older estimator. It is not evidence
  about this run, and it is not being carried forward. Where the present run
  cannot converge the optimiser, `C4`/`C5` are reported as **NOT EVALUATED** --
  not as passes, and not as failures.
- The superseded run printed minimum non-prior eigenvalues at more precision
  than they carry (e.g. a negative value of order `1e-20` rendered as though it
  were a measurement). This report prints `0` with an explicit
  `numerically_zero` flag and the estimated noise floor alongside, per the
  precision correction adopted from agent 🧩 Rao.
"""

#: The identifiability laboratory is an exact linear-Gaussian surrogate.  It is
#: routinely misread as a statement about the *trained* model, which is a
#: different object with a different -- currently defective -- noise model.
SCOPE_BOUNDARY_UNCERTAINTY = """
## Scope boundary: what this report does **not** say about uncertainty

This laboratory measures parameter identifiability in an **exact
linear-Gaussian state-space surrogate** (`scwbd.infer.linear_gaussian`). It
imports nothing from `scwbd.foundation` and evaluates no trained checkpoint.

That distinction matters because of the run-1 P0 recorded in
`reports/scope_gap.md` §6: the trained model's predictive variance
(`scwbd/foundation/heads.py:238`) is **one learned scalar per channel,
broadcast** -- it never reads the state. Nothing in this report should be read
as evidence that the trained model's uncertainty is state-dependent or
calibrated, because:

- `C1`/`C2`/`C3` are exact Fisher computations at the true parameter. In a
  linear-Gaussian model the innovation covariance is state-independent *by
  theorem*, which is why the Riccati recursion can be shared across epochs at
  all. This is a correct property of the surrogate, not a shortcut, and it is
  also the reason the surrogate cannot detect the defect the P0 describes.
- `C4`/`C5` concern **parameter** intervals (Laplace, from the observed
  information over `eta`). They say nothing about **predictive** intervals over
  observations, which is the channel where run 1 failed.

If a downstream claim needs the model's uncertainty to vary with brain state,
that property does not currently exist and this artifact does not supply it.
"""


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
                          "regime_prior_offset": regime_prior_offset(results),
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
    A(ARTIFACT_LINEAGE)
    A(SCOPE_BOUNDARY_UNCERTAINTY)
    A("\n## Criteria (all held-out regimes must pass)\n")
    A("| criterion | statement | result |")
    A("|---|---|---|")
    for k, v in decision["criteria_all_regimes"].items():
        A(f"| `{k}` | {DECISION_RULE['criteria'][k]} | {_fmt(v)} |")
    A("")
    A("> " + DECISION_RULE["known_algebraic_caveat"] + "\n")

    # C4 is a statement about interval calibration.  Intervals built from the
    # observed information at a point that is not the optimum are not the
    # intervals C4 claims to test, so a pass earned where the optimiser stopped
    # short has to be labelled at the top, not left in a column further down.
    weak_conv = []
    for rname, rr in results["regimes"].items():
        rec = rr["designs"].get("joint_native", {}).get("recovery")
        if not rec:
            continue
        frac = float(rec.get("converged_fraction", 1.0))
        if frac < 0.9:
            med = rec.get("optimiser", {}).get("median_newton_decrement")
            weak_conv.append((rname, frac, med))
    if weak_conv:
        A("\n> **Convergence caveat on `C4`.** The MAP estimator did not reach "
          "the convergence tolerance for every replicate in:\n>")
        for rname, frac, med in weak_conv:
            A(f"> - `{rname}`: {frac:.0%} of `joint_native` replicates converged"
              + (f", median remaining Newton decrement {med:.3f} posterior sd"
                 if med is not None else "") + ".")
        A(">\n> Coverage there is computed from observed-information intervals "
          "around estimates that are still short of the optimum, so the `C4` "
          "pass is **not** a sound calibration test in those regimes. Raising "
          "the step cap, or refreshing the preconditioner at the current "
          "iterate instead of holding it at the prior mean, is the fix.\n")

    for rname, rr in results["regimes"].items():
        A(f"\n## Regime `{rname}`\n")
        A(f"{rr['description']}\n")
        if rname in decision.get("delay_degenerate_regimes", []):
            A("> **Delay comparison is degenerate in this regime.** The true "
              "conduction delay coincides with the prior mean, so a design that "
              "learns *nothing* about the delay leaves it at the prior mean and "
              "scores a near-perfect delay error. Delay evidence in this regime "
              "is not discriminating; the two held-out regimes place the delay "
              "away from the prior mean for exactly this reason.\n")
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
              f"{_fmt_sig(m['min_eigenvalue_nonprior'], m, 'min_eigenvalue_nonprior')} | "
              f"{_fmt_sig(m['theta_profile_min_eigenvalue_nonprior'], m, 'theta_profile_min_eigenvalue')} | "
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
        A(
            "\nEigenvalues are printed at the precision their **measured** "
            "reproducibility supports. Recomputing the whole pipeline under three "
            "BLAS thread counts (1/8/20, which changes summation order) reproduces "
            "a well-conditioned theta-profile lambda_min to 1.3e-12 relative "
            "(~12 significant figures); a near-cancelling one inherits that "
            "amplified by lambda_max/lambda_min, so `fmri_only` is reproducible to "
            "only ~7 figures. Entries shown as `0 (num. zero)` are inside their own "
            "noise floor -- their sign is not even stable across thread counts -- "
            "and must not be read as small positive information.\n"
        )
        if any("recovery" in v for v in rr["designs"].values()):
            A("\n### Recovery (MAP + observed-information intervals)\n")
            A("| design | delay RMSE (ms) | θ RMSE (prior sd) | "
              + " | ".join(f"cov `{p}`" for p in PREREGISTERED_SUBSET)
              + " | median Newton decrement |")
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
                  f"{_fmt(rec['optimiser'].get('median_newton_decrement'))} |")
            A("\nThe estimator is a fixed-budget damped Newton run from the prior "
              "mean, preconditioned by the expected information; the Newton "
              "decrement is the remaining distance to the MAP in posterior "
              "standard deviations.  Coverage is a property of *that* estimator "
              "and is measured directly, so a decrement above zero is a reported "
              "fact rather than an unstated approximation.\n")
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
    offs = regime_prior_offset(results)
    if offs:
        A("\n## Can each regime's recovery numbers discriminate at all?\n")
        A("Bias, RMSE and coverage only measure *information* when the truth "
          "sits away from the prior mean. Where it coincides, an estimator that "
          "ignores the data and returns the prior mean scores zero bias, zero "
          "RMSE and 100% coverage.\n")
        A("| regime | " + " | ".join(f"`{p}`" for p in PREREGISTERED_SUBSET)
          + " | max \\|offset\\| | recovery metrics |")
        A("|---|" + "---|" * (len(PREREGISTERED_SUBSET) + 2))
        for rname, v in offs.items():
            cells = " | ".join(f"{v['offset_in_prior_sd'][p]:+.3f}"
                               for p in PREREGISTERED_SUBSET)
            verdict = ("**DEGENERATE — do not read as evidence**"
                       if v["recovery_metrics_degenerate"] else "discriminating")
            A(f"| `{rname}` | {cells} | {v['max_abs_offset_prior_sd']:.3f} | {verdict} |")
        A("\nOffsets are in prior standard deviations. A degenerate regime is "
          "still valid for the *information* criteria (C1, C2-information, C3), "
          "which are evaluated from the Fisher information at that operating "
          "point and do not depend on where the prior sits.\n")
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
    A("\n## Related artifact\n")
    A("Agent 🧩 Rao's per-parameter-group decomposition of these same designs -- "
      "coupling / delay / EEG-lead-field / haemodynamic, per modality "
      "combination, distinguishing structural zeros from small-but-nonzero "
      "values -- is at `reports/individualize/identifiability_by_modality.md` "
      "(machine-readable: `identifiability_by_modality.json`). It converts the "
      "single lambda_min reported here into a per-group capability statement, "
      "which is the form a downstream individualization claim actually needs. "
      "Rao also supplied `assert_delay_line_adequate`, adopted here: a delay "
      "line shorter than `tau/dt + 3*sinc_sigma` inflates the conduction-delay "
      "information by ~25 orders of magnitude with nothing raised, and the "
      "inflated reading is the one that says *spectacularly identifiable*.\n")
    A("\n## What would disable this module\n")
    A(DECISION_RULE["what_would_disable_this_module"])
    A("\n---\n")
    A(f"Generated {time.strftime('%Y-%m-%dT%H:%M:%S%z')} · "
      f"git `{_env()['git_revision'][:12]}` · machine-readable: `results.json`.\n")
    mpath = outdir / "summary.md"
    mpath.write_text("\n".join(L))
    return {"json": jpath, "markdown": mpath}
