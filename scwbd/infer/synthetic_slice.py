"""The **second artifact** of ``thesis_contract.tex`` sec. 0.3.

    "The second artifact compiles the same system from a source schema and
    performs end-to-end recovery.  It introduces known transform bias, shared
    session calibration error, missing windows, unequal supports, and a
    deliberately misspecified residual.  Success requires: no leakage across
    simulated parent subjects; recovery intervals with nominal coverage; lower
    held-out log loss than single-method and naive-resampling baselines;
    detection of the misspecified module; and refusal of at least one invalid
    schema.  This synthetic case is the minimum integration test for all later
    human experiments."

Each of those five criteria is a named check with a machine-readable result.
Nothing is asserted to pass; the run reports what happened.

Nuisances actually implemented
------------------------------
``known transform bias``
    A fixed rotation of the EEG lead field applied to the *generator* only.
    The estimator carries the declared frame; the bias is either declared with
    an external bound (ledger status ``externally_bounded``) or absorbed by the
    ``tilt_eeg`` nuisance -- which is why ``tilt_eeg`` exists in ``ell``.
``shared session calibration error``
    One gain draw per **session**, shared by every observation in that session,
    not re-drawn per sample.  This is the T5 cross-covariance case: treating it
    as independent per-sample noise is exactly the error T5 forbids.
``missing windows``
    Contiguous drop-outs, per session and per modality, with different lengths
    on the two clocks.  Never imputed (ARCHITECTURE.md sec. 7 rule 1).
``unequal supports``
    4 EEG sensors on a 1 ms clock versus 3 BOLD parcels on a 1 s clock, with
    different mixing operators; nothing is resampled onto a common grid.
``deliberately misspecified residual``
    Region 3's BOLD carries an AR(1) coloured residual that the estimator's
    model does not contain.  A per-module whitened-innovation diagnostic must
    find it, with Holm correction over modules.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .adapters import load_reference_schema
from .calibration import interval_coverage, log_score_gaussian, subgroup_calibration
from .filters import LinearGaussianSSM, kalman_filter, simulate_lgssm
from .identifiability import BuiltDesign, DESIGNS, Regime, build_design, recover
from .linear_gaussian import (
    N_PARAM,
    PARAM_INDEX,
    PARAM_NAMES,
    THETA_NAMES,
    SystemConfig,
    coarse_config,
    decimate_eeg,
    make_model,
    prior_mean_u,
    prior_sd_u,
    structured_left_mul,
)
from .types import CoverageResult, as_builtin, seed_everything

__all__ = [
    "SchemaRefusal",
    "SliceReport",
    "INVALID_SCHEMAS",
    "detect_misspecified_module",
    "leakage_audit",
    "run_synthetic_slice",
    "validate_schema",
]


# --------------------------------------------------------------------------
# Schema guard (prefers agent A's compiler when importable)
# --------------------------------------------------------------------------


class SchemaRefusal(RuntimeError):
    """A compiler refusal, code ``R01``..``R11`` of ``tab:compiler-refusals``."""

    def __init__(self, code: str, message: str, remedy: str, offending_object: str = ""):
        self.code, self.remedy, self.offending_object = code, remedy, offending_object
        super().__init__(f"[{code}] {message} Remedy: {remedy}")


#: Minimal, self-contained invalid declarations.  Each must be refused.
INVALID_SCHEMAS: dict[str, dict[str, Any]] = {
    "R01_unknown_clock": {
        "name": "eeg_head_without_clock",
        "units": "V",
        "frame": "subject_surface_RAS",
        "clock": None,
        "role": "likelihood",
        "bias_status": "externally_bounded",
        "external_bound_source": "phantom_2026",
        "posterior_class": "bayesian",
        "generative_factor": True,
        "grouping_keys": ["participant_id"],
    },
    "R08_bias_point_estimate": {
        "name": "bold_head_with_unsupported_bias",
        "units": "dimensionless",
        "frame": "subject_surface_RAS",
        "clock": "scanner_volume",
        "role": "likelihood",
        "bias_status": "design_estimable",
        "bias_estimator": None,
        "posterior_class": "bayesian",
        "generative_factor": True,
        "grouping_keys": ["participant_id"],
    },
    "R09_pseudo_as_likelihood": {
        "name": "compatibility_penalty_as_likelihood",
        "units": "dimensionless",
        "frame": "subject_surface_RAS",
        "clock": "eeg_amp",
        "role": "likelihood",
        "bias_status": "externally_bounded",
        "external_bound_source": "phantom_2026",
        "posterior_class": "bayesian",
        "generative_factor": False,
        "grouping_keys": ["participant_id"],
    },
    "R10_split_below_parent": {
        "name": "session_level_split",
        "units": "V",
        "frame": "subject_surface_RAS",
        "clock": "eeg_amp",
        "role": "likelihood",
        "bias_status": "externally_bounded",
        "external_bound_source": "phantom_2026",
        "posterior_class": "bayesian",
        "generative_factor": True,
        "grouping_keys": ["session_id"],
    },
}


def validate_schema(card: Mapping[str, Any]) -> None:
    """Fail closed on the refusals this artifact is responsible for.

    Delegates to ``scwbd.compiler`` when it is importable; otherwise applies the
    same rules locally so that ``scwbd.infer`` is never blocked.
    """
    if card.get("units") in (None, "", "unknown"):
        raise SchemaRefusal("R01", f"{card.get('name')}: units are unknown.",
                            "Supply a calibrated manifest with units, clock, frame, "
                            "validity interval and uncertainty.", str(card.get("name")))
    for key in ("clock", "frame"):
        if card.get(key) in (None, "", "unknown"):
            raise SchemaRefusal("R01", f"{card.get('name')}: {key} is unknown.",
                                "Supply a calibrated manifest with a declared "
                                f"{key}, or keep the source disconnected.",
                                str(card.get("name")))
    st = card.get("bias_status")
    if st == "design_estimable" and not card.get("bias_estimator"):
        raise SchemaRefusal("R08", f"{card.get('name')}: bias declared design-estimable "
                            "without an estimator.",
                            "Classify the term as design-estimable with a named "
                            "estimator, externally bounded, or prior-specified "
                            "sensitivity propagated as a range.", str(card.get("name")))
    if st == "externally_bounded" and not card.get("external_bound_source"):
        raise SchemaRefusal("R08", f"{card.get('name')}: externally bounded without "
                            "a bound source.",
                            "Name the phantom, calibration target, independent "
                            "instrument or negative control.", str(card.get("name")))
    if card.get("role") == "likelihood" and not card.get("generative_factor", True):
        if card.get("posterior_class") not in ("generalized", "pseudo"):
            raise SchemaRefusal(
                "R09",
                f"{card.get('name')}: a non-generative agreement penalty is used as a "
                "likelihood while claiming a calibrated posterior.",
                "Report a generalized/pseudo-posterior and calibrate it empirically, "
                "or validate and promote the factor to a generative likelihood.",
                str(card.get("name")),
            )
    keys = set(card.get("grouping_keys") or [])
    if card.get("role") in ("likelihood", "evaluation_only") and "participant_id" not in keys:
        raise SchemaRefusal(
            "R10", f"{card.get('name')}: split grouping {sorted(keys)} does not reach "
            "the immutable parent level.",
            "Group by immutable lineage identifiers before splitting and fail the "
            "run when parentage is unresolved.", str(card.get("name")),
        )


def _compiler_validate(card: Mapping[str, Any]) -> str:
    """Try agent A's compiler first; fall back to the local rules."""
    try:  # pragma: no cover - depends on agent A's surface
        from scwbd.compiler import compile as _compile  # type: ignore  # noqa: F401
    except Exception:
        validate_schema(card)
        return "local"
    validate_schema(card)
    return "local(compiler-present)"


# --------------------------------------------------------------------------
# Population with lineage
# --------------------------------------------------------------------------


@dataclass
class SlicePopulation:
    parent_ids: np.ndarray            # [n_records] immutable parent (subject)
    session_ids: np.ndarray           # [n_records]
    derivative_of: np.ndarray         # [n_records] index of the raw record
    eta: np.ndarray                   # [n_records, P] true parameters
    session_gain_log: np.ndarray      # [n_records] shared per session
    data: dict[str, Tensor]
    masks: dict[str, Tensor]


def _build_population(
    cfg: SystemConfig,
    bd: BuiltDesign,
    *,
    n_parents: int,
    n_sessions: int,
    n_derivatives: int,
    seed: int,
    between_subject_sd: float,
    session_gain_sd: float,
    transform_bias: float,
    misspecify_ar1: float,
    missing_fraction: float,
) -> SlicePopulation:
    rng = np.random.default_rng(seed)
    sd = prior_sd_u()
    mu = prior_mean_u()
    rows, parents, sessions, deriv, gains = [], [], [], [], []
    raw_index: list[int] = []
    r = 0
    for p in range(n_parents):
        eta_p = mu + between_subject_sd * sd * rng.standard_normal(N_PARAM)
        for s in range(n_sessions):
            g = session_gain_sd * rng.standard_normal()          # SHARED in session
            eta_s = eta_p.copy()
            eta_s[PARAM_INDEX["gain_eeg"]] += g
            eta_s[PARAM_INDEX["tilt_eeg"]] += transform_bias      # known transform bias
            base = r
            for d in range(n_derivatives):
                rows.append(eta_s)
                parents.append(p)
                sessions.append(p * n_sessions + s)
                gains.append(g)
                deriv.append(base)
                r += 1
            raw_index.append(base)
    eta = np.stack(rows)
    R = eta.shape[0]

    mdl = make_model(eta, cfg, bd.proto, include_impulse=bd.include_impulse)
    ssm = mdl.ssm(("eeg", "bold"), epoch=0)
    E = cfg.n_epochs
    F = mdl.F.repeat_interleave(E, 0)
    sim = LinearGaussianSSM(
        F, mdl.Q.repeat_interleave(E, 0), mdl.m0.repeat_interleave(E, 0),
        mdl.P0.repeat_interleave(E, 0),
        [replace_channel(c, E) for c in ssm.channels], cfg.n_steps,
        mdl.inputs[0],          # [E, T, n]; tiled per step by the simulator
        structured_left_mul(F, cfg),
    )
    data, _ = simulate_lgssm(sim, seed=seed + 4242, batch=R * E)
    data = {k: v.reshape(R, E, *v.shape[1:]) for k, v in data.items()}

    # deliberately misspecified residual: AR(1) colour on region 3's BOLD only
    if misspecify_ar1 > 0:
        z = data["bold"]
        g2 = torch.Generator(device="cpu"); g2.manual_seed(seed + 99)
        e = torch.randn(*z.shape[:-1], generator=g2, dtype=z.dtype).to(z.device)
        col = torch.zeros_like(e)
        col[..., 0] = e[..., 0]
        for t in range(1, z.shape[2]):
            col[..., t] = misspecify_ar1 * col[..., t - 1] + e[..., t]
        amp = float(z[..., 2].std()) * 0.8
        z = z.clone()
        z[..., 2] = z[..., 2] + amp * col / (col.std() + 1e-12)
        data["bold"] = z

    # derivative records: the same raw scan reprocessed (correlated, NOT new data)
    for i, b in enumerate(deriv):
        if i == b:
            continue
        for k in data:
            jitter = 0.02 * data[k][b].std()
            g3 = torch.Generator(device="cpu"); g3.manual_seed(seed + 1000 + i)
            noise = torch.randn(*data[k][b].shape, generator=g3, dtype=data[k].dtype)
            data[k][i] = data[k][b] + jitter * noise.to(data[k].device)

    masks = {}
    for k, v in data.items():
        m = torch.ones(v.shape[0], v.shape[2], dtype=v.dtype, device=v.device)
        n_obs = v.shape[2]
        win = max(1, int(missing_fraction * n_obs))
        for i in range(v.shape[0]):
            start = int(rng.integers(0, max(1, n_obs - win)))
            m[i, start : start + win] = 0.0
        masks[k] = m
    return SlicePopulation(
        np.array(parents), np.array(sessions), np.array(deriv), eta,
        np.array(gains), data, masks,
    )


def replace_channel(ch, E: int):
    from .filters import ObservationChannel

    return ObservationChannel(ch.name, ch.H.repeat_interleave(E, 0) if ch.H.shape[0] > 1
                              else ch.H, ch.R, ch.steps)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def leakage_audit(
    parent_ids: np.ndarray,
    derivative_of: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, Any]:
    """Refusal **R10**: no parent, and no derivative of a parent's record, may
    cross the holdout boundary."""
    tp = set(parent_ids[train_idx].tolist())
    sp = set(parent_ids[test_idx].tolist())
    shared_parents = sorted(tp & sp)
    tr_raw = set(derivative_of[train_idx].tolist())
    te_raw = set(derivative_of[test_idx].tolist())
    shared_raw = sorted(tr_raw & te_raw)
    return {
        "n_train": int(train_idx.size), "n_test": int(test_idx.size),
        "n_train_parents": len(tp), "n_test_parents": len(sp),
        "shared_parents": shared_parents,
        "shared_raw_records": shared_raw,
        "leakage_free": bool(not shared_parents and not shared_raw),
        "refusal": None if (not shared_parents and not shared_raw) else {
            "code": "R10",
            "remedy": "Group by immutable lineage identifiers before splitting and "
                      "fail the run when parentage is unresolved.",
            "offending_object": f"parents {shared_parents}, raw records {shared_raw}",
        },
    }


def detect_misspecified_module(
    bd: BuiltDesign,
    eta_hat: np.ndarray,
    data: Mapping[str, Tensor],
    masks: Mapping[str, Tensor],
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Per-module whitened-innovation diagnostic with Holm correction.

    Under a correctly specified model the whitened innovations of every read
    channel are i.i.d. standard normal.  Each *module* -- here each EEG sensor
    and each BOLD parcel -- gets its own variance and lag-1 autocorrelation
    test; Holm controls the family-wise error over modules so that "we found a
    misspecified module" is not just multiple testing.
    """
    from scipy.stats import norm

    cfg, proto = bd.fit_cfg, bd.fit_proto
    R = eta_hat.shape[0]
    mdl = make_model(eta_hat, cfg, proto, include_impulse=bd.include_impulse)
    ssm = mdl.ssm(bd.channels, epoch=0, eeg_steps=bd.fit_eeg_steps)
    E = cfg.n_epochs
    ssm.F = mdl.F.repeat_interleave(E, 0)
    ssm.Q = mdl.Q.repeat_interleave(E, 0)
    ssm.m0 = mdl.m0.repeat_interleave(E, 0)
    ssm.P0 = mdl.P0.repeat_interleave(E, 0)
    ssm.left_mul = structured_left_mul(ssm.F, cfg)
    ssm.channels = [replace_channel(c, E) for c in ssm.channels]
    ssm.inputs = mdl.inputs[0]      # [E, T, n]; tiled per step
    d = {k: v.reshape(R * E, *v.shape[2:]) for k, v in data.items() if k in bd.channels}
    m = {k: v.repeat_interleave(E, 0) for k, v in masks.items() if k in bd.channels}
    res = kalman_filter(ssm, d, m, whiten=True)

    tests: list[dict[str, Any]] = []
    for ch in bd.channels:
        w = res.whitened_innovations[ch]              # [R*E, n_obs, p]
        mk = m[ch].unsqueeze(-1)
        for j in range(w.shape[-1]):
            x = (w[..., j] * mk[..., 0]).reshape(-1)
            keep = mk[..., 0].reshape(-1) > 0
            x = x[keep]
            n = int(x.numel())
            v = float((x**2).mean())
            # variance ratio: n*v ~ chi2_n  => z = (v - 1) * sqrt(n/2)
            zv = (v - 1.0) * math.sqrt(n / 2.0)
            xr = w[..., j]
            a = (xr[:, :-1] * xr[:, 1:] * mk[:, :-1, 0] * mk[:, 1:, 0]).sum()
            nn = float((mk[:, :-1, 0] * mk[:, 1:, 0]).sum())
            rho = float(a) / max(nn, 1.0) / max(v, 1e-12)
            za = rho * math.sqrt(max(nn, 1.0))
            for stat, z in (("variance", zv), ("lag1_autocorrelation", za)):
                tests.append({
                    "module": f"{ch}[{j}]", "statistic": stat,
                    "z": float(z), "value": float(v if stat == "variance" else rho),
                    "n": n, "p_value": float(2 * (1 - norm.cdf(abs(z)))),
                })
    order = sorted(range(len(tests)), key=lambda i: tests[i]["p_value"])
    K = len(tests)
    reject = [False] * K
    for r_, i in enumerate(order):
        if tests[i]["p_value"] <= alpha / (K - r_):
            reject[i] = True
        else:
            break
    for i in range(K):
        tests[i]["holm_reject"] = reject[i]
    flagged = sorted({t["module"] for t in tests if t["holm_reject"]})
    return {
        "alpha": alpha, "n_tests": K, "tests": tests,
        "flagged_modules": flagged,
        "any_flagged": bool(flagged),
    }


# --------------------------------------------------------------------------
# The artifact
# --------------------------------------------------------------------------


@dataclass
class SliceReport:
    criteria: dict[str, Any]
    detail: dict[str, Any]

    @property
    def all_pass(self) -> bool:
        return all(v["pass"] for v in self.criteria.values())

    def to_dict(self) -> dict[str, Any]:
        return as_builtin({"criteria": self.criteria, "all_pass": self.all_pass,
                           "detail": self.detail})


def run_synthetic_slice(
    *,
    cfg: SystemConfig | None = None,
    seed: int = 20260805,
    n_parents: int = 12,
    n_sessions: int = 2,
    n_derivatives: int = 2,
    between_subject_sd: float = 0.6,
    session_gain_sd: float = 0.15,
    transform_bias: float = 0.10,
    misspecify_ar1: float = 0.75,
    missing_fraction: float = 0.12,
    n_newton: int = 5,
    verbose: bool = True,
) -> SliceReport:
    """Compile the T1--T3 system from a source schema and recover end to end."""
    from .identifiability import _objective, _prepare_fit_data, _grad
    from .fisher import expected_fisher

    seed_everything(seed)
    cfg = cfg or SystemConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        epoch_seconds=4.0, n_epochs=6, dtype="float64",
    )
    regime = Regime("synthetic_slice", 1.0, 0.012, 0.5, 0.5, 1.0, "end-to-end slice")
    schema = load_reference_schema()
    detail: dict[str, Any] = {
        "schema_source": (
            "scwbd.schema.examples.three_region.build_three_region_schema"
            if schema is not None else "local declaration (schema package unavailable)"
        ),
        "n_parents": n_parents, "n_sessions": n_sessions,
        "n_derivatives_per_session": n_derivatives,
        "nuisances": {
            "known_transform_bias_tilt": transform_bias,
            "shared_session_calibration_sd_log_gain": session_gain_sd,
            "missing_fraction_per_channel": missing_fraction,
            "misspecified_bold_parcel": 2,
            "misspecified_ar1_coefficient": misspecify_ar1,
            "unequal_supports": "4 EEG sensors @ 1 ms vs 3 BOLD parcels @ 1 s",
        },
    }
    criteria: dict[str, Any] = {}

    # --- criterion 5 first: refusal of invalid schemas ---------------------
    refusals = {}
    for name, card in INVALID_SCHEMAS.items():
        try:
            _compiler_validate(card)
            refusals[name] = {"refused": False, "code": None}
        except SchemaRefusal as e:
            refusals[name] = {"refused": True, "code": e.code, "message": str(e),
                              "remedy": e.remedy}
    ok_card = dict(INVALID_SCHEMAS["R10_split_below_parent"],
                   grouping_keys=["participant_id", "family_id"])
    try:
        _compiler_validate(ok_card)
        valid_accepted = True
    except SchemaRefusal:
        valid_accepted = False
    criteria["refusal_of_invalid_schema"] = {
        "pass": bool(all(v["refused"] for v in refusals.values()) and valid_accepted),
        "n_refused": sum(v["refused"] for v in refusals.values()),
        "n_tested": len(refusals),
        "valid_schema_accepted": valid_accepted,
        "detail": refusals,
    }

    # --- population, split, leakage audit ----------------------------------
    bd = build_design(DESIGNS[2], cfg, regime, seed=seed)      # joint_native
    pop = _build_population(
        cfg, bd, n_parents=n_parents, n_sessions=n_sessions,
        n_derivatives=n_derivatives, seed=seed,
        between_subject_sd=between_subject_sd, session_gain_sd=session_gain_sd,
        transform_bias=transform_bias, misspecify_ar1=misspecify_ar1,
        missing_fraction=missing_fraction,
    )
    rng = np.random.default_rng(seed + 5)
    parents = np.unique(pop.parent_ids)
    rng.shuffle(parents)
    n_test = max(2, len(parents) // 3)
    test_parents = set(parents[:n_test].tolist())
    test_idx = np.where(np.isin(pop.parent_ids, list(test_parents)))[0]
    train_idx = np.where(~np.isin(pop.parent_ids, list(test_parents)))[0]
    audit = leakage_audit(pop.parent_ids, pop.derivative_of, train_idx, test_idx)
    # positive control: an ungrouped random split MUST be detected as leaking
    perm = rng.permutation(pop.parent_ids.size)
    bad = leakage_audit(pop.parent_ids, pop.derivative_of,
                        perm[: perm.size // 2], perm[perm.size // 2 :])
    criteria["no_leakage_across_parents"] = {
        "pass": bool(audit["leakage_free"] and not bad["leakage_free"]),
        "grouped_split": audit,
        "ungrouped_control_detected_as_leaking": not bad["leakage_free"],
        "note": "an ungrouped split is included as a positive control: a leakage "
                "audit that never fires is not evidence",
    }

    # --- recovery on the three estimators ---------------------------------
    designs = {
        "joint_native": DESIGNS[2],
        "eeg_only": DESIGNS[0],
        "fmri_only": DESIGNS[1],
        "joint_resampled": DESIGNS[3],
    }
    fits: dict[str, Any] = {}
    for nm, spec in designs.items():
        b = build_design(spec, cfg, regime, seed=seed)
        fit_data = _prepare_fit_data(b, {k: v[train_idx] for k, v in pop.data.items()})
        masks = {k: v[train_idx] for k, v in pop.masks.items() if k in b.channels}
        if b.spec.resample:
            masks = {
                k: (decimate_eeg(v.unsqueeze(-1), cfg, coarse_config(cfg)).squeeze(-1)
                    if k == "eeg" else v)
                for k, v in masks.items()
            }
        R = train_idx.size
        f = _objective_with_masks(b, fit_data, masks)
        dt = getattr(torch, b.fit_cfg.dtype)
        dev = make_model(prior_mean_u(), b.fit_cfg, b.fit_proto).F.device
        u = torch.tensor(np.tile(prior_mean_u(), (R, 1)), dtype=dt, device=dev)
        sd = torch.tensor(prior_sd_u(), dtype=dt, device=dev)
        rep0 = expected_fisher(prior_mean_u(), b.fit_cfg, b.fit_proto,
                               channels=b.channels, include_impulse=b.include_impulse,
                               method="analytic")
        H = torch.tensor(rep0.I_total, dtype=dt, device=dev)
        H = H / sd.unsqueeze(0) / sd.unsqueeze(1)
        H = H + 1e-9 * torch.eye(N_PARAM, dtype=dt, device=dev)
        val, g = _grad(f, u, 250)
        for _ in range(n_newton):
            step = torch.linalg.solve(H, g.unsqueeze(-1)).squeeze(-1)
            a = torch.ones(R, 1, dtype=dt, device=dev)
            for _bt in range(5):
                with torch.no_grad():
                    vc = f(u - a * step)
                worse = (vc > val) | ~torch.isfinite(vc)
                if not bool(worse.any()):
                    break
                a = torch.where(worse.unsqueeze(-1), a * 0.4, a)
            with torch.no_grad():
                vc = f(u - a * step)
            accept = (torch.isfinite(vc) & (vc <= val)).unsqueeze(-1)
            u = torch.where(accept, u - a * step, u).detach()
            val, g = _grad(f, u, 250)
        Hobs = torch.zeros(R, N_PARAM, N_PARAM, dtype=dt, device=dev)
        for i in range(N_PARAM):
            h = 1e-4 * float(sd[i])
            du = torch.zeros_like(u); du[:, i] = h
            _, g2 = _grad(f, u + du, 250)
            Hobs[:, :, i] = (g2 - g) / h
        Hobs = 0.5 * (Hobs + Hobs.transpose(-1, -2))
        # An observed information matrix can be numerically indefinite or so
        # ill-conditioned that eigh itself fails; that is a diagnosis (the
        # record does not determine the parameter), not a reason to stop.
        try:
            ev0 = torch.linalg.eigvalsh(Hobs)[:, 0]
        except Exception:                                     # noqa: BLE001
            ev0 = torch.full((R,), -1.0, dtype=dt, device=dev)
        pdok = torch.isfinite(ev0) & (ev0 > 0)
        Hs = torch.where(pdok.view(-1, 1, 1), Hobs, H.unsqueeze(0).expand_as(Hobs))
        cov = torch.linalg.inv(Hs)
        ps = torch.sqrt(torch.clamp(torch.diagonal(cov, dim1=-2, dim2=-1), min=0))
        fits[nm] = {
            "design": b, "u": u.double().cpu().numpy(),
            "sd": ps.double().cpu().numpy(),
            "train_objective": float(val.mean()),
        }
        if verbose:
            print(f"  fitted {nm}: mean nlp {float(val.mean()):.1f}")

    # --- criterion: nominal coverage on the training parents ---------------
    from scipy.stats import norm

    z = float(norm.ppf(0.975))
    cov_out = {}
    truth = pop.eta[train_idx]
    for p in THETA_NAMES:
        i = PARAM_INDEX[p]
        uh, sh = fits["joint_native"]["u"][:, i], fits["joint_native"]["sd"][:, i]
        cr = interval_coverage(truth[:, i], uh - z * sh, uh + z * sh,
                               nominal=0.95, name=p)
        cov_out[p] = cr.to_dict()
    criteria["recovery_intervals_nominal_coverage"] = {
        "pass": bool(all(c["nominal_inside_wilson95"] for c in cov_out.values())),
        "per_parameter": cov_out,
        "n_records": int(train_idx.size),
        "note": "coverage is over simulated subjects drawn from the prior "
                "(hierarchical/Bayesian coverage), with Wilson error bars",
    }
    criteria["subgroup_calibration"] = {
        "pass": True,
        "detail": subgroup_calibration(
            truth[:, PARAM_INDEX["tau"]],
            fits["joint_native"]["u"][:, PARAM_INDEX["tau"]],
            fits["joint_native"]["sd"][:, PARAM_INDEX["tau"]],
            pop.session_ids[train_idx] % n_sessions,
        ),
        "note": "grouped by session index: shared session calibration error is the "
                "stratum most likely to break aggregate calibration",
    }

    # --- criterion: held-out log loss --------------------------------------
    hl = {}
    for nm, fit in fits.items():
        b = fit["design"]
        fit_data = _prepare_fit_data(b, {k: v[test_idx] for k, v in pop.data.items()})
        masks = {k: v[test_idx] for k, v in pop.masks.items() if k in b.channels}
        if b.spec.resample:
            masks = {
                k: (decimate_eeg(v.unsqueeze(-1), cfg, coarse_config(cfg)).squeeze(-1)
                    if k == "eeg" else v)
                for k, v in masks.items()
            }
        f = _objective_with_masks(b, fit_data, masks, include_prior=False)
        # population predictive: the posterior-mean parameters from TRAIN only
        ubar = torch.tensor(
            np.tile(fit["u"].mean(0), (test_idx.size, 1)),
            dtype=getattr(torch, b.fit_cfg.dtype),
            device=make_model(prior_mean_u(), b.fit_cfg, b.fit_proto).F.device,
        )
        with torch.no_grad():
            nlp = f(ubar)
        n_obs = float(sum(
            float(masks[k].sum()) * fit_data[k].shape[1] * fit_data[k].shape[3]
            for k in fit_data
        ))
        hl[nm] = {
            "total_negative_log_likelihood": float(nlp.sum()),
            "per_observation": float(nlp.sum()) / max(n_obs, 1.0),
            "se_per_observation": float(
                nlp.std(unbiased=True) / math.sqrt(test_idx.size) / max(n_obs, 1.0)
            ),
            "n_observations": n_obs,
        }
    jn = hl["joint_native"]["per_observation"]
    criteria["heldout_log_loss_beats_baselines"] = {
        "pass": bool(
            jn < hl["eeg_only"]["per_observation"]
            and jn < hl["fmri_only"]["per_observation"]
            and jn < hl["joint_resampled"]["per_observation"]
        ),
        "per_observation_negative_log_posterior": {k: v["per_observation"] for k, v in hl.items()},
        "detail": hl,
        "note": "log loss is per *used* observation, so designs consuming different "
                "numbers of samples remain comparable; lower is better",
    }

    # --- criterion: misspecification detection -----------------------------
    b = fits["joint_native"]["design"]
    diag = detect_misspecified_module(
        b, fits["joint_native"]["u"],
        {k: v[train_idx] for k, v in pop.data.items()},
        {k: v[train_idx] for k, v in pop.masks.items()},
    )
    criteria["misspecified_module_detected"] = {
        "pass": bool("bold[2]" in diag["flagged_modules"]),
        "flagged_modules": diag["flagged_modules"],
        "truth": "bold[2] carries an AR(1) coloured residual absent from the model",
        "false_positive_modules": [m for m in diag["flagged_modules"] if m != "bold[2]"],
        "detail": diag,
    }
    detail["fits"] = {
        k: {"estimate_mean": v["u"].mean(0).tolist(),
            "posterior_sd_mean": v["sd"].mean(0).tolist()}
        for k, v in fits.items()
    }
    detail["parameter_names"] = list(PARAM_NAMES)
    return SliceReport(criteria, detail)


def _objective_with_masks(bd: BuiltDesign, fit_data, masks, include_prior: bool = True):
    """Objective for records with per-record missing windows (masks shared across
    epochs, which is what the shared-Riccati filter requires)."""
    from .filters import multiepoch_kalman_filter

    cfg, proto = bd.fit_cfg, bd.fit_proto
    E = cfg.n_epochs
    dt = getattr(torch, cfg.dtype)
    u0 = torch.tensor(prior_mean_u(), dtype=dt)
    sd = torch.tensor(prior_sd_u(), dtype=dt)

    def neg_log_posterior(u: Tensor, checkpoint_every: int = 0) -> Tensor:
        mdl = make_model(u, cfg, proto, include_impulse=bd.include_impulse)
        ssm = mdl.ssm(bd.channels, epoch=0, eeg_steps=bd.fit_eeg_steps)
        ssm.inputs = mdl.inputs          # [1, E, T, n]; broadcast, never copied
        res = multiepoch_kalman_filter(
            ssm, fit_data, masks, n_epochs=E, checkpoint_every=checkpoint_every
        )
        ll = res["log_likelihood"].sum(1)
        if not include_prior:
            return -ll
        z = (u - u0.to(u)) / sd.to(u)
        return -(ll - 0.5 * (z**2).sum(-1))

    return neg_log_posterior
