"""``individualize(population_model, availability, data)`` -- fit only what is identified.

The contract, in one sentence: **a parameter group that the identifiability
profile does not admit is never touched by the optimiser, and the value returned
for it is bit-identical to the population value it started from, carrying the
label** ``population_prior``.

That is stronger than "we regularised it heavily".  A heavily regularised fit
still moves; a moved value invites a reader to interpret the movement.  Here the
masked coordinates are excluded from the parameterisation entirely, the
optimiser's step is projected onto the fitted subspace, and the result is
reassembled as ``population_value`` with the fitted coordinates substituted --
so bit-identity is a property of the construction, and
``tests/individualize/test_individualize_labelling.py`` asserts it with
``==`` rather than ``allclose``.

**Absence writes something.**  Every group that was not individualized produces a
:class:`GroupOutcome` with ``status="population_prior"``, the reason, and the
prior's own uncertainty in the ledger.  There is no code path in which a group
is silently skipped: :meth:`IndividualizationResult.assert_complete` fails if
any declared group is missing from the outcome table.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from scwbd.infer.linear_gaussian import (
    N_PARAM,
    PARAM_INDEX,
    PARAM_NAMES,
    SystemConfig,
    natural_from_unconstrained,
)

from .availability import ModalityAvailability
from .groups import GROUPS, ParameterGroup
from .hierarchy import Decomposition, PopulationModel, decompose_sessions
from .profile import (
    IdentifiabilityProfile,
    IdentifiabilityThresholds,
    benchmark_config,
    profile_identifiability,
)

__all__ = [
    "ConsistencyCheck",
    "GroupOutcome",
    "IndividualizationResult",
    "PatientData",
    "data_consistency",
    "individualize",
    "simulate_patient",
]

#: Status labels.  ``population_prior`` is the load-bearing one.
INDIVIDUALIZED = "individualized"
WEAKLY_INDIVIDUALIZED = "weakly_individualized"
POPULATION_PRIOR = "population_prior"


@dataclass
class PatientData:
    """One or more sessions of native-clock records for one patient.

    ``sessions[session_id][channel]`` is ``[n_epochs, n_obs, p]``.  A channel
    the patient does not have is **absent from the mapping**, never a zero
    tensor: see ``availability.refuse_zero_imputation``.
    """

    sessions: Mapping[str, Mapping[str, torch.Tensor]]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def session_ids(self) -> tuple[str, ...]:
        return tuple(self.sessions)

    def channels(self) -> tuple[str, ...]:
        chans: set[str] = set()
        for s in self.sessions.values():
            chans |= set(s)
        return tuple(sorted(chans))


#: Declared tolerance for the negative control: the whitened innovations of a
#: fitted model must have mean square within this relative distance of 1.  A
#: model whose one-step predictions are 25% wrong in variance is not the model
#: that produced the data, and no parameter fitted through it is that patient's.
CONSISTENCY_REL_TOLERANCE: float = 0.25


@dataclass(frozen=True)
class ConsistencyCheck:
    """Is the patient's record consistent with the forward model at all?

    Individualization asks *which parameter values* best explain the record.  It
    presupposes that the record is the kind of thing the model can explain.  A
    patient whose "data" are noise -- a disconnected electrode, a corrupted
    export, a phantom scan filed under a person -- has an answer to the first
    question and no business having one, so the second is checked separately.

    Under the fitted model the whitened innovations are standard normal, so
    ``mean(z^2)`` is 1 with standard error ``sqrt(2/n)``.  The gate is on the
    *relative* deviation rather than the z-score: with 10^5 samples a z-score
    fails on any model error at all, which would make the check unable to
    discriminate a corrupted record from an imperfect model -- exactly the
    decorative-guard failure.
    """

    statistic: float
    n_samples: int
    z_score: float
    rel_tolerance: float
    passed: bool
    reason: str
    per_channel: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "whitened_innovation_mean_square": self.statistic,
            "n_samples": self.n_samples,
            "z_score": self.z_score,
            "rel_tolerance": self.rel_tolerance,
            "passed": self.passed,
            "reason": self.reason,
            "per_channel": dict(self.per_channel),
        }


def data_consistency(
    bd,
    fit_data: Mapping[str, torch.Tensor],
    u: np.ndarray,
    *,
    n_epochs: int,
    rel_tolerance: float = CONSISTENCY_REL_TOLERANCE,
) -> ConsistencyCheck:
    """Whitened-innovation check of the record against the model at ``u``."""
    from scwbd.infer.filters import multiepoch_kalman_filter
    from scwbd.infer.linear_gaussian import make_model

    dt = getattr(torch, bd.fit_cfg.dtype)
    ut = torch.tensor(np.asarray(u, float), dtype=dt)
    mdl = make_model(ut, bd.fit_cfg, bd.fit_proto, include_impulse=bd.include_impulse)
    ssm = mdl.ssm(bd.channels, epoch=0, eeg_steps=bd.fit_eeg_steps)
    ssm.inputs = mdl.inputs
    with torch.no_grad():
        res = multiepoch_kalman_filter(
            ssm, dict(fit_data), n_epochs=n_epochs, whiten=True
        )
    per: dict[str, float] = {}
    tot_sq = 0.0
    tot_n = 0
    for c in bd.channels:
        z = res.get(f"whitened/{c}")
        if z is None:
            continue
        zz = z.reshape(-1)
        per[c] = float((zz**2).mean())
        tot_sq += float((zz**2).sum())
        tot_n += int(zz.numel())
    stat = tot_sq / max(tot_n, 1)
    zsc = (stat - 1.0) * np.sqrt(max(tot_n, 1) / 2.0)
    ok = abs(stat - 1.0) <= rel_tolerance
    reason = (
        f"whitened innovations have mean square {stat:.4f} over {tot_n} samples "
        f"(z = {zsc:.3g}); "
        + (
            f"within the declared {rel_tolerance:.0%} tolerance of 1."
            if ok
            else (
                f"OUTSIDE the declared {rel_tolerance:.0%} tolerance of 1: the "
                "record is not consistent with the forward model at any fitted "
                "parameter value, so nothing fitted through it is this "
                "patient's. The individualization is REJECTED and every group "
                "reverts to the population prior."
            )
        )
    )
    return ConsistencyCheck(
        statistic=stat,
        n_samples=tot_n,
        z_score=float(zsc),
        rel_tolerance=rel_tolerance,
        passed=bool(ok),
        reason=reason,
        per_channel=per,
    )


@dataclass(frozen=True)
class GroupOutcome:
    """What happened to one parameter group, and why."""

    group: str
    status: str
    parameters: tuple[str, ...]
    source: str
    value_unconstrained: Mapping[str, float]
    value_natural: Mapping[str, float]
    population_value_unconstrained: Mapping[str, float]
    posterior_sd_unconstrained: Mapping[str, float]
    identifiability_status: str
    lambda_min: float | None
    reason: str
    ledger: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_population_prior(self) -> bool:
        return self.status == POPULATION_PRIOR

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "status": self.status,
            "parameters": list(self.parameters),
            "source": self.source,
            "value_unconstrained": dict(self.value_unconstrained),
            "value_natural": dict(self.value_natural),
            "population_value_unconstrained": dict(
                self.population_value_unconstrained
            ),
            "posterior_sd_unconstrained": dict(self.posterior_sd_unconstrained),
            "identifiability_status": self.identifiability_status,
            "lambda_min": self.lambda_min,
            "reason": self.reason,
            "uncertainty_ledger": dict(self.ledger),
        }


@dataclass
class IndividualizationResult:
    """The individualized patient, with every group labelled by its provenance."""

    patient_id: str
    group: str
    availability: ModalityAvailability
    profile: IdentifiabilityProfile
    population: PopulationModel
    decomposition: Decomposition
    outcomes: Mapping[str, GroupOutcome]
    fit_mask: np.ndarray
    fit_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    consistency: ConsistencyCheck | None = None
    notes: tuple[str, ...] = ()

    # -- accessors --------------------------------------------------------
    @property
    def theta_trait(self) -> np.ndarray:
        return self.decomposition.theta_trait

    def value(self, parameter: str) -> float:
        return float(self.theta_trait[PARAM_INDEX[parameter]])

    def status_of(self, group: str) -> str:
        return self.outcomes[group].status

    @property
    def individualized_groups(self) -> tuple[str, ...]:
        return tuple(
            k
            for k, v in self.outcomes.items()
            if v.status in (INDIVIDUALIZED, WEAKLY_INDIVIDUALIZED)
        )

    @property
    def population_prior_groups(self) -> tuple[str, ...]:
        return tuple(k for k, v in self.outcomes.items() if v.is_population_prior)

    def assert_complete(self, groups: Sequence[ParameterGroup] = GROUPS) -> None:
        """Absence must write something: every declared group has an outcome."""
        missing = [g.name for g in groups if g.name not in self.outcomes]
        if missing:
            raise AssertionError(
                f"individualization result is silent about {missing}; a group "
                "that was not individualized must be RECORDED as such, not "
                "inferred from its absence from the table."
            )

    def assert_population_prior_exact(self) -> None:
        """Non-individualized coordinates equal the population value exactly."""
        pop = self.population.population_value(self.group)
        for name, out in self.outcomes.items():
            if not out.is_population_prior:
                continue
            for p in out.parameters:
                i = PARAM_INDEX.get(p)
                if i is None:
                    continue
                got = float(self.theta_trait[i])
                want = float(pop[i])
                if got != want:
                    raise AssertionError(
                        f"group {name!r} is labelled {POPULATION_PRIOR} but its "
                        f"parameter {p!r} is {got!r}, not the population value "
                        f"{want!r}; a labelled prior that has moved is a "
                        "posterior wearing the wrong label."
                    )
                for s in range(self.decomposition.zeta.shape[0]):
                    if float(self.decomposition.zeta[s, i]) != 0.0:
                        raise AssertionError(
                            f"group {name!r} is labelled {POPULATION_PRIOR} but "
                            f"has a non-zero session effect for {p!r}"
                        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "group": self.group,
            "availability": self.availability.to_dict(),
            "identifiability_profile": self.profile.to_dict(),
            "population_model": self.population.to_dict(),
            "decomposition": self.decomposition.to_dict(),
            "outcomes": {k: v.to_dict() for k, v in self.outcomes.items()},
            "individualized_groups": list(self.individualized_groups),
            "population_prior_groups": list(self.population_prior_groups),
            "fit_mask": {p: bool(self.fit_mask[i]) for i, p in enumerate(PARAM_NAMES)},
            "fit_diagnostics": dict(self.fit_diagnostics),
            "data_consistency": (
                self.consistency.to_dict() if self.consistency is not None else None
            ),
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------
# the fit
# --------------------------------------------------------------------------
def _built_design(availability: ModalityAvailability, cfg: SystemConfig, regime, seed):
    from scwbd.infer.identifiability import DESIGNS, build_design

    spec = next((d for d in DESIGNS if d.name == availability.design), None)
    if spec is None:
        raise KeyError(f"no design for {availability.design!r}")
    return build_design(spec, cfg, regime, seed=seed)


def _neg_log_posterior(
    bd,
    fit_data: dict[str, torch.Tensor],
    n_epochs: int,
    u_prior_mean: np.ndarray,
    prior_sd: np.ndarray,
):
    """Negative log posterior of the reference slice for one session.

    The prior is **this patient's**: centred on ``mu + alpha_{g(p)}`` with the
    between-person/between-session spread, not on the global population mean
    with the population prior's width.  Getting that wrong would shrink a
    patient in a shifted group back toward the grand mean, which is precisely
    the identity drift sec. 6.5 is about.
    """
    from scwbd.infer.filters import multiepoch_kalman_filter
    from scwbd.infer.linear_gaussian import make_model

    cfg, proto = bd.fit_cfg, bd.fit_proto
    dt = getattr(torch, cfg.dtype)
    u0 = torch.tensor(np.asarray(u_prior_mean, float), dtype=dt)
    sd = torch.tensor(np.asarray(prior_sd, float), dtype=dt)

    def f(u: torch.Tensor) -> torch.Tensor:
        mdl = make_model(u, cfg, proto, include_impulse=bd.include_impulse)
        ssm = mdl.ssm(bd.channels, epoch=0, eeg_steps=bd.fit_eeg_steps)
        ssm.inputs = mdl.inputs
        res = multiepoch_kalman_filter(ssm, fit_data, n_epochs=n_epochs)
        ll = res["log_likelihood"].sum(1)
        z = (u - u0.to(u)) / sd.to(u)
        return -(ll - 0.5 * (z**2).sum(-1))

    return f


def _masked_map_fit(
    bd,
    fit_data: dict[str, torch.Tensor],
    u_start: np.ndarray,
    mask: np.ndarray,
    *,
    n_newton: int,
    n_epochs: int,
    individual_sd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Newton MAP restricted to ``mask``.

    Returns ``(u_hat, posterior_sd, likelihood_only_sd, diag)``.  The prior is
    ``N(u_start, diag(individual_sd)^2)`` -- this patient's, centred on their
    group's population value.  ``likelihood_only_sd`` is reported separately
    because it, not the posterior sd, is the right input to the hierarchical
    shrinkage downstream: shrinking an already-shrunk estimate shrinks twice.

    Coordinates outside ``mask`` are held at ``u_start`` **exactly** -- the step
    is formed only in the masked subspace and written back only there, so no
    floating-point dust lands on the frozen coordinates.
    """
    from scwbd.infer.fisher import expected_fisher

    idx = np.flatnonzero(mask)
    dt = getattr(torch, bd.fit_cfg.dtype)
    u = torch.tensor(np.asarray(u_start, float), dtype=dt).reshape(1, -1)
    ind_sd = np.asarray(individual_sd, dtype=float)
    f = _neg_log_posterior(bd, fit_data, n_epochs, u_start, ind_sd)

    rep0 = expected_fisher(
        np.asarray(u_start, float),
        bd.fit_cfg,
        bd.fit_proto,
        channels=bd.channels,
        include_impulse=bd.include_impulse,
        eeg_steps=bd.fit_eeg_steps,
        method="analytic",
        standardised=False,
        device="cpu",
    )
    # Preconditioner: likelihood information + THIS patient's prior precision.
    # ``rep0.I_total`` carries the *population* prior, which is not the prior
    # this objective uses.
    H_full = np.asarray(rep0.I_likelihood, dtype=float) + np.diag(1.0 / ind_sd**2)
    H = H_full[np.ix_(idx, idx)] + 1e-9 * np.eye(len(idx))
    Ht = torch.tensor(H, dtype=dt)

    history: list[float] = []
    val = None
    for _ in range(int(n_newton)):
        uu = u.detach().requires_grad_(True)
        val = f(uu)
        (g,) = torch.autograd.grad(val.sum(), uu)
        history.append(float(val.detach().sum()))
        gm = g.detach().reshape(-1)[idx]
        step = torch.linalg.solve(Ht, gm.reshape(-1, 1)).reshape(-1)
        alpha = 1.0
        accepted = False
        base = float(val.detach().sum())
        for _bt in range(8):
            cand = u.clone()
            cand[0, idx] = u[0, idx] - alpha * step
            with torch.no_grad():
                vc = float(f(cand).sum())
            if np.isfinite(vc) and vc <= base:
                u = cand
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
    with torch.no_grad():
        final = float(f(u).sum())
    history.append(final)

    # Laplace sd from the expected information at the estimate, masked block.
    rep1 = expected_fisher(
        u.detach().reshape(-1).numpy(),
        bd.fit_cfg,
        bd.fit_proto,
        channels=bd.channels,
        include_impulse=bd.include_impulse,
        eeg_steps=bd.fit_eeg_steps,
        method="analytic",
        standardised=False,
        device="cpu",
    )
    I_like = np.asarray(rep1.I_likelihood, float)
    sd = ind_sd.copy()
    like_sd = ind_sd.copy()
    Hm = I_like[np.ix_(idx, idx)] + np.diag(1.0 / ind_sd[idx] ** 2)
    sd[idx] = np.sqrt(np.clip(np.diag(np.linalg.inv(Hm)), 0.0, None))
    # Likelihood-only standard errors, with a pseudo-inverse so a direction the
    # data do not constrain reads as "infinite se" rather than as a LAPACK
    # accident.  Capped at the prior sd because an unconstrained direction
    # cannot be *less* informative than knowing nothing.
    Lm = I_like[np.ix_(idx, idx)]
    cov_like = np.linalg.pinv(Lm, rcond=1e-12, hermitian=True)
    like_sd[idx] = np.minimum(
        np.sqrt(np.clip(np.diag(cov_like), 0.0, None)) + (np.diag(Lm) <= 0) * 1e300,
        ind_sd[idx] * 1e6,
    )

    u_hat = u.detach().reshape(-1).numpy().copy()
    # belt and braces: the frozen coordinates are the starting values, exactly
    frozen = np.setdiff1d(np.arange(N_PARAM), idx)
    u_hat[frozen] = np.asarray(u_start, float)[frozen]
    diag = {
        "objective_history": history,
        "n_newton": int(n_newton),
        "fitted_indices": idx.tolist(),
        "frozen_indices": frozen.tolist(),
        "objective_decrease": history[0] - history[-1] if len(history) > 1 else 0.0,
        "prior_used": "individual: N(mu + alpha_g, diag(person_sd^2 + session_sd^2))",
    }
    return u_hat, sd, like_sd, diag


def individualize(
    population: PopulationModel,
    availability: ModalityAvailability,
    data: PatientData | Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    *,
    profile: IdentifiabilityProfile | None = None,
    cfg: SystemConfig | None = None,
    regime: Any = None,
    thresholds: IdentifiabilityThresholds | None = None,
    n_newton: int = 4,
    seed: int = 20260805,
    counterfactual_modalities: Sequence[str] = (),
) -> IndividualizationResult:
    """Individualise ``population`` on whatever this patient actually has.

    Only the parameter groups the identifiability profile admits are fitted.
    Every other group is returned at the population value, labelled
    ``population_prior``, with the reason recorded.
    """
    from scwbd.infer.identifiability import REGIMES

    cfg = cfg or benchmark_config()
    regime = regime if regime is not None else REGIMES[0]
    if profile is None:
        profile = profile_identifiability(
            availability,
            cfg=cfg,
            regime=regime,
            thresholds=thresholds,
            seed=seed,
            counterfactual_modalities=counterfactual_modalities,  # type: ignore[arg-type]
        )
    if profile.availability_digest != availability.digest():
        raise ValueError(
            "the supplied identifiability profile was computed for a different "
            "availability; refusing to fit against a profile that does not "
            "describe this patient's data"
        )

    population.assert_centered()
    pop_value = population.population_value(availability.group)

    if isinstance(data, PatientData):
        pdata = data
    elif data is None:
        pdata = PatientData(sessions={})
    else:
        pdata = PatientData(sessions=data)

    mask = profile.fit_mask()
    notes: list[str] = []
    fit_diag: dict[str, Any] = {"per_session": {}}

    # -- decide whether an optimiser runs at all --------------------------
    no_data_reason = None
    if not pdata.sessions:
        no_data_reason = "no patient records were supplied"
    elif not mask.any():
        no_data_reason = (
            "the identifiability profile admits no parameter group, so no "
            "optimiser was run; every value below is the population prior, "
            "unmodified"
        )
    if no_data_reason is not None:
        mask = np.zeros(N_PARAM, dtype=bool)
        notes.append(no_data_reason)

    session_ids = pdata.session_ids or (
        availability.session_ids if availability.session_ids else ("s0",)
    )
    offsets = np.zeros((len(session_ids), N_PARAM))
    obs_sd = np.asarray(population.prior_sd, dtype=float).copy()
    like_sd = obs_sd.copy()
    #: marginal prior on one session's offset from the group value:
    #: ``delta_p + zeta_{p,s} ~ N(0, Sigma_person + Sigma_session)``
    individual_sd = np.sqrt(
        np.asarray(population.person_sd, float) ** 2
        + np.asarray(population.session_sd, float) ** 2
    )
    fitted_with_prior = False

    consistency: ConsistencyCheck | None = None
    if mask.any():
        bd = _built_design(availability, cfg, regime, seed)
        t0 = time.time()
        sds = []
        like_sds = []
        checks: list[ConsistencyCheck] = []
        for k, sid in enumerate(session_ids):
            rec = pdata.sessions[sid]
            missing = [c for c in bd.channels if c not in rec]
            if missing:
                raise KeyError(
                    f"session {sid!r} is missing channels {missing} that the "
                    f"design {availability.design!r} requires; a missing channel "
                    "must change the availability (and therefore the profile), "
                    "not be filled with zeros"
                )
            fit_data = {c: rec[c] for c in bd.channels}
            n_ep = int(next(iter(fit_data.values())).shape[-3])
            u_hat, sd_hat, like_sd_hat, diag = _masked_map_fit(
                bd,
                fit_data,
                pop_value,
                mask,
                n_newton=n_newton,
                n_epochs=n_ep,
                individual_sd=individual_sd,
            )
            offsets[k] = u_hat - pop_value
            sds.append(sd_hat)
            like_sds.append(like_sd_hat)
            fit_diag["per_session"][sid] = diag
            checks.append(
                data_consistency(bd, fit_data, u_hat, n_epochs=n_ep)
            )
        obs_sd = np.mean(np.stack(sds, 0), axis=0)
        like_sd = np.mean(np.stack(like_sds, 0), axis=0)
        fitted_with_prior = True
        fit_diag["seconds"] = time.time() - t0
        fit_diag["design"] = availability.design
        fit_diag["channels"] = list(bd.channels)

        # -- negative control: is the record the kind of thing this model
        # explains at all?  If not, the fitted numbers are not this patient's
        # and every group reverts to the population prior.
        worst = max(checks, key=lambda c: abs(c.statistic - 1.0))
        consistency = worst
        fit_diag["data_consistency_per_session"] = {
            sid: c.to_dict() for sid, c in zip(session_ids, checks)
        }
        if not worst.passed:
            mask = np.zeros(N_PARAM, dtype=bool)
            offsets = np.zeros_like(offsets)
            obs_sd = np.asarray(population.prior_sd, dtype=float).copy()
            like_sd = obs_sd.copy()
            fitted_with_prior = False
            notes.append(
                "NEGATIVE CONTROL FIRED: " + worst.reason
            )

    decomposition = decompose_sessions(
        population,
        availability.group,
        offsets,
        session_ids,
        observation_sd=like_sd,
        mask=mask,
        already_shrunk=fitted_with_prior,
    )
    if decomposition.centering_residual > 1e-10:
        from .hierarchy import R07Violation

        raise R07Violation(
            "[R07] session effects are not centered within the patient "
            f"(max |sum_s zeta_{{p,s}}| = {decomposition.centering_residual:.3g})"
        )

    theta = decomposition.theta_trait
    nat = natural_from_unconstrained(theta)

    outcomes: dict[str, GroupOutcome] = {}
    for g in GROUPS:
        gi = profile.groups[g.name]
        if g.kind == "anatomical":
            individualized = gi.status == "identifiable"
            outcomes[g.name] = GroupOutcome(
                group=g.name,
                status=INDIVIDUALIZED if individualized else POPULATION_PRIOR,
                parameters=g.parameters,
                source=(
                    "patient anatomy (presence-determined, no Fisher number)"
                    if individualized
                    else "population/template"
                ),
                value_unconstrained={},
                value_natural={},
                population_value_unconstrained={},
                posterior_sd_unconstrained={},
                identifiability_status=gi.status,
                lambda_min=None,
                reason=gi.reason,
                ledger={
                    "variance": {},
                    "bias_status": "prior_specified_sensitivity",
                    "note": (
                        "anatomical groups are individualised outside the "
                        "dynamical likelihood; their uncertainty is owned by "
                        "the anatomy/observe modules, not measured here"
                    ),
                },
            )
            continue

        fitted = bool(mask[[PARAM_INDEX[p] for p in g.parameters]].all())
        if not fitted and consistency is not None and not consistency.passed:
            status = POPULATION_PRIOR
            source = "population prior (NOT this patient)"
            reason = (
                "NOT individualized: the negative control rejected this "
                "patient's record. " + consistency.reason
            )
            outcomes[g.name] = GroupOutcome(
                group=g.name,
                status=status,
                parameters=g.parameters,
                source=source,
                value_unconstrained={
                    p: float(theta[PARAM_INDEX[p]]) for p in g.parameters
                },
                value_natural={p: float(nat[p]) for p in g.parameters},
                population_value_unconstrained={
                    p: float(pop_value[PARAM_INDEX[p]]) for p in g.parameters
                },
                posterior_sd_unconstrained={
                    p: float(population.prior_sd[PARAM_INDEX[p]]) for p in g.parameters
                },
                identifiability_status=gi.status,
                lambda_min=gi.lambda_min,
                reason=reason,
                ledger={
                    "variance": {
                        p: float(population.prior_sd[PARAM_INDEX[p]] ** 2)
                        for p in g.parameters
                    },
                    "variance_source": (
                        "population prior variance -- the record failed the "
                        "model-consistency check, so no patient information "
                        "was admitted"
                    ),
                    "bias_status": "prior_specified_sensitivity",
                    "validity_domain": {
                        "modalities": list(availability.present),
                        "design": availability.design,
                        "regime": profile.regime,
                        "data_consistency": consistency.to_dict(),
                    },
                    "units": "unconstrained coordinate of scwbd.infer.linear_gaussian",
                },
            )
            continue
        if fitted:
            status = (
                INDIVIDUALIZED
                if gi.status == "identifiable"
                else WEAKLY_INDIVIDUALIZED
            )
            source = "patient data"
            reason = (
                f"fitted from {list(availability.channels)} on the patient's own "
                f"records; {gi.reason}"
            )
        else:
            status = POPULATION_PRIOR
            source = "population prior (NOT this patient)"
            reason = (
                "NOT individualized. " + gi.reason + " The value below is the "
                "population value, returned unchanged and labelled as such."
            )
        outcomes[g.name] = GroupOutcome(
            group=g.name,
            status=status,
            parameters=g.parameters,
            source=source,
            value_unconstrained={p: float(theta[PARAM_INDEX[p]]) for p in g.parameters},
            value_natural={p: float(nat[p]) for p in g.parameters},
            population_value_unconstrained={
                p: float(pop_value[PARAM_INDEX[p]]) for p in g.parameters
            },
            posterior_sd_unconstrained={
                p: float(obs_sd[PARAM_INDEX[p]] if fitted else population.prior_sd[PARAM_INDEX[p]])
                for p in g.parameters
            },
            identifiability_status=gi.status,
            lambda_min=gi.lambda_min,
            reason=reason,
            ledger={
                "variance": {
                    p: float(
                        (obs_sd[PARAM_INDEX[p]] if fitted else population.prior_sd[PARAM_INDEX[p]])
                        ** 2
                    )
                    for p in g.parameters
                },
                "variance_source": (
                    "Laplace (expected information at the estimate)"
                    if fitted
                    else "population prior variance -- no patient information entered"
                ),
                "bias_status": "prior_specified_sensitivity",
                "prior_fraction": {
                    p: float(1.0 / (1.0 + (gi.per_parameter_information.get(p, 0.0))))
                    for p in g.parameters
                },
                "validity_domain": {
                    "modalities": list(availability.present),
                    "design": availability.design,
                    "regime": profile.regime,
                },
                "units": "unconstrained coordinate of scwbd.infer.linear_gaussian",
            },
        )

    if not decomposition.separable:
        notes.append(decomposition.separability_reason)
    if profile.notes:
        notes.extend(profile.notes)

    res = IndividualizationResult(
        patient_id=availability.patient_id,
        group=availability.group,
        availability=availability,
        profile=profile,
        population=population,
        decomposition=decomposition,
        outcomes=outcomes,
        fit_mask=mask,
        fit_diagnostics=fit_diag,
        consistency=consistency,
        notes=tuple(notes),
    )
    res.assert_complete()
    res.assert_population_prior_exact()
    return res


# --------------------------------------------------------------------------
# synthetic patients (for tests, demos and the negative control)
# --------------------------------------------------------------------------
def simulate_patient(
    availability: ModalityAvailability,
    *,
    eta_true: np.ndarray | None = None,
    cfg: SystemConfig | None = None,
    regime: Any = None,
    seed: int = 0,
    n_sessions: int = 1,
    pure_noise: bool = False,
) -> PatientData:
    """Simulate native-clock records for the channels this patient has.

    ``pure_noise=True`` is the **negative control**: records with the same
    shape, dtype and marginal scale as real ones, drawn from white noise with no
    dynamical model behind them.  A patient whose data are this must not come
    out looking individualized.
    """
    from scwbd.infer.filters import LinearGaussianSSM, simulate_lgssm
    from scwbd.infer.identifiability import REGIMES
    from scwbd.infer.linear_gaussian import make_model, structured_left_mul

    cfg = cfg or benchmark_config()
    regime = regime if regime is not None else REGIMES[0]
    bd = _built_design(availability, cfg, regime, seed)
    u_true = np.asarray(
        eta_true if eta_true is not None else regime.eta_true(), dtype=float
    )
    mdl = make_model(u_true, bd.cfg, bd.proto, include_impulse=bd.include_impulse)
    ssm = mdl.ssm(("eeg", "bold"), epoch=0)
    sim = LinearGaussianSSM(
        mdl.F, mdl.Q, mdl.m0, mdl.P0, ssm.channels, bd.cfg.n_steps,
        mdl.inputs[0], structured_left_mul(mdl.F, bd.cfg),
    )
    E = bd.cfg.n_epochs
    sessions: dict[str, dict[str, torch.Tensor]] = {}
    for s in range(n_sessions):
        data, _ = simulate_lgssm(sim, seed=seed + 1000 * s, batch=E)
        rec: dict[str, torch.Tensor] = {}
        for c in bd.channels:
            # [1, E, n_obs, p]: batch 1 (one patient), E epochs -- the shape
            # multiepoch_kalman_filter expects.
            y = data[c].reshape(1, E, *data[c].shape[1:])
            if pure_noise:
                g = torch.Generator(device="cpu").manual_seed(seed + 7919 * s)
                scale = float(y.std())
                y = torch.randn(y.shape, generator=g, dtype=y.dtype) * scale
            rec[c] = y
        sessions[f"s{s}"] = rec
    return PatientData(
        sessions=sessions,
        provenance={
            "simulated": True,
            "design": availability.design,
            "regime": getattr(regime, "name", str(regime)),
            "seed": seed,
            "pure_noise": pure_noise,
            "eta_true_unconstrained": u_true.tolist(),
        },
    )
