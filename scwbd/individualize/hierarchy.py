"""``theta_{p,s} = mu + alpha_{g(p)} + delta_p + zeta_{p,s}`` (body.tex sec. 6.5).

Refusal **R07** -- *population/subject/session effects without centering or
shrinkage* -- is what makes this decomposition legal to write down.  It is
enforced three ways here, none of them decorative:

1. :class:`PopulationModel` stores group effects in an unconstrained array and
   **projects out the weighted mean on every read**, so ``sum_g n_g alpha_g = 0``
   cannot be violated by any caller, optimiser or checkpoint.
   :meth:`PopulationModel.assert_centered` is the executable check and raises
   ``scwbd.foundation.individual.R07Violation`` -- the project's existing
   refusal type, not a new one.
2. :func:`decompose_sessions` centres the session effects within a patient
   (``sum_s zeta_{p,s} = 0``) and applies **normal--normal shrinkage** to
   ``delta_p``, whose factor is reported so a reader can see it is < 1.
3. :func:`hierarchical_effect_declarations` emits the
   :class:`~scwbd.schema.sources.HierarchicalEffect` records the compiler's
   ``check_r07`` reads.  ``tests/individualize/test_r07.py`` shows the compiler
   refusal FIRING on a deliberately broken declaration, so the declaration is
   known to be load-bearing rather than assumed to be.

The single-session case is the honest one.  With one session, ``delta_p`` and
``zeta_{p,s}`` are **not separately identified** -- their sum is all the data
constrain.  :func:`decompose_sessions` therefore returns the sum in ``delta``,
``zeta = 0``, and ``separable=False``, and every downstream report says the
split was not identified rather than printing a confident zero session effect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from scwbd.foundation.individual import Individualizer, R07Violation
from scwbd.infer.linear_gaussian import (
    PARAM_NAMES,
    natural_from_unconstrained,
    prior_mean_u,
    prior_sd_u,
)

__all__ = [
    "Decomposition",
    "PopulationModel",
    "R07Violation",
    "decompose_sessions",
    "hierarchical_effect_declarations",
    "recover_decomposition",
]

#: Prior sd of the person effect ``delta_p``, in prior-sd units of each
#: parameter.  Declared, not fitted, when a single patient is individualised:
#: one patient cannot estimate the population's between-person spread.
DEFAULT_PERSON_SD: float = 0.5
#: Prior sd of the session effect ``zeta_{p,s}``, same units.  Strictly smaller
#: than the person effect: identity is the slower quantity (sec. 6.5).
DEFAULT_SESSION_SD: float = 0.2


@dataclass(frozen=True)
class PopulationModel:
    """``mu`` and centered group effects ``alpha_g`` of the population fit.

    Everything is in the **unconstrained** coordinate ``u`` of
    ``scwbd.infer.linear_gaussian`` (log/logit-transformed where the natural
    parameter is positive or bounded), which is the coordinate the prior is
    diagonal Gaussian in and the coordinate the Fisher information is reported
    in.  :meth:`natural` converts for human consumption.
    """

    parameter_names: tuple[str, ...] = PARAM_NAMES
    mu: np.ndarray = field(default_factory=prior_mean_u)
    prior_sd: np.ndarray = field(default_factory=prior_sd_u)
    group_names: tuple[str, ...] = ("population",)
    group_counts: np.ndarray = field(default_factory=lambda: np.ones(1))
    #: Raw (unconstrained) group effects; the centered ones are :attr:`alpha`.
    alpha_raw: np.ndarray | None = None
    person_sd: np.ndarray | None = None
    session_sd: np.ndarray | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        P = len(self.parameter_names)
        G = len(self.group_names)
        object.__setattr__(self, "mu", np.asarray(self.mu, dtype=float).reshape(P))
        object.__setattr__(
            self, "prior_sd", np.asarray(self.prior_sd, dtype=float).reshape(P)
        )
        object.__setattr__(
            self, "group_counts", np.asarray(self.group_counts, dtype=float).reshape(G)
        )
        if self.alpha_raw is None:
            object.__setattr__(self, "alpha_raw", np.zeros((G, P)))
        else:
            object.__setattr__(
                self, "alpha_raw", np.asarray(self.alpha_raw, dtype=float).reshape(G, P)
            )
        if self.person_sd is None:
            object.__setattr__(self, "person_sd", DEFAULT_PERSON_SD * self.prior_sd)
        if self.session_sd is None:
            object.__setattr__(self, "session_sd", DEFAULT_SESSION_SD * self.prior_sd)
        if float(self.group_counts.sum()) <= 0:
            raise ValueError("group_counts must sum to a positive number")
        if np.any(np.asarray(self.session_sd) > np.asarray(self.person_sd)):
            raise R07Violation(
                "[R07] session spread exceeds person spread: a session effect "
                "that out-varies the person effect makes the identity the "
                "faster quantity and the decomposition uninterpretable "
                "(body.tex sec. 6.5)."
            )

    # -- centered effects, projected on every read ------------------------
    @property
    def alpha(self) -> np.ndarray:
        """``alpha_g`` with ``sum_g n_g alpha_g = 0`` **by construction**."""
        w = (self.group_counts / self.group_counts.sum()).reshape(-1, 1)
        return self.alpha_raw - (w * self.alpha_raw).sum(0, keepdims=True)

    def group_index(self, group: str) -> int:
        if group not in self.group_names:
            raise KeyError(
                f"unknown group {group!r}; known: {list(self.group_names)}"
            )
        return self.group_names.index(group)

    def population_value(self, group: str = "population") -> np.ndarray:
        """``mu + alpha_{g}`` -- what an un-individualised patient gets."""
        return self.mu + self.alpha[self.group_index(group)]

    def natural(self, u: np.ndarray) -> dict[str, float]:
        nat = natural_from_unconstrained(np.asarray(u, dtype=float))
        return {k: float(v) for k, v in nat.items()}

    # -- R07 --------------------------------------------------------------
    def assert_centered(self, tol: float = 1e-10) -> None:
        w = (self.group_counts / self.group_counts.sum()).reshape(-1, 1)
        resid = float(np.abs((w * self.alpha).sum(0)).max())
        if resid > tol:
            raise R07Violation(
                f"[R07] weighted group effects do not sum to zero "
                f"(max |sum_g n_g alpha_g| = {resid:.3g} > {tol})"
            )
        for name, v in (("person_sd", self.person_sd), ("session_sd", self.session_sd)):
            arr = np.asarray(v, dtype=float)
            if not np.isfinite(arr).all() or np.any(arr <= 0):
                raise R07Violation(
                    f"[R07] {name} is not a positive finite scale: shrinkage is "
                    "not defined and the decomposition is unidentified"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_names": list(self.parameter_names),
            "mu": self.mu.tolist(),
            "prior_sd": self.prior_sd.tolist(),
            "group_names": list(self.group_names),
            "group_counts": self.group_counts.tolist(),
            "alpha_centered": self.alpha.tolist(),
            "person_sd": np.asarray(self.person_sd).tolist(),
            "session_sd": np.asarray(self.session_sd).tolist(),
            "centering_residual": float(
                np.abs(
                    (
                        (self.group_counts / self.group_counts.sum()).reshape(-1, 1)
                        * self.alpha
                    ).sum(0)
                ).max()
            ),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def reference(cls, **kw: Any) -> "PopulationModel":
        """The population prior of the reference slice: ``mu = prior mean``."""
        return cls(
            provenance={
                "source": "scwbd.infer.linear_gaussian.prior_mean_u / prior_sd_u",
                "note": (
                    "a single population group with alpha = 0; this stands in "
                    "for a fitted population posterior and is labelled so that "
                    "nothing downstream can mistake it for one"
                ),
            },
            **kw,
        )


@dataclass(frozen=True)
class Decomposition:
    """``mu + alpha + delta + zeta`` for one patient, with the split's status."""

    parameter_names: tuple[str, ...]
    mu: np.ndarray
    alpha: np.ndarray
    delta: np.ndarray
    zeta: np.ndarray  # [n_sessions, P]
    session_ids: tuple[str, ...]
    shrinkage_factor: np.ndarray
    separable: bool
    separability_reason: str
    centering_residual: float
    #: True when the offsets were already posterior means under the individual
    #: prior, so :attr:`shrinkage_factor` is a diagnostic rather than a factor
    #: this function applied.
    shrinkage_applied_in_fit: bool = False

    @property
    def theta(self) -> np.ndarray:
        """``theta_{p,s}`` per session, ``[n_sessions, P]``."""
        return (self.mu + self.alpha + self.delta).reshape(1, -1) + self.zeta

    @property
    def theta_trait(self) -> np.ndarray:
        """``mu + alpha + delta`` -- the part claimed to persist across sessions."""
        return self.mu + self.alpha + self.delta

    def to_dict(self) -> dict[str, Any]:
        names = list(self.parameter_names)
        return {
            "parameter_names": names,
            "mu": self.mu.tolist(),
            "alpha": self.alpha.tolist(),
            "delta": self.delta.tolist(),
            "zeta": self.zeta.tolist(),
            "session_ids": list(self.session_ids),
            "shrinkage_factor": np.asarray(self.shrinkage_factor).tolist(),
            "shrinkage_applied_in_fit": self.shrinkage_applied_in_fit,
            "delta_zeta_separable": self.separable,
            "separability_reason": self.separability_reason,
            "zeta_centering_residual": self.centering_residual,
            "theta_trait": self.theta_trait.tolist(),
        }


def decompose_sessions(
    population: PopulationModel,
    group: str,
    session_offsets: np.ndarray,
    session_ids: Sequence[str],
    *,
    observation_sd: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    already_shrunk: bool = False,
) -> Decomposition:
    """Split per-session offsets into a shrunk ``delta_p`` and centered ``zeta``.

    ``session_offsets[s] = theta_hat_{p,s} - (mu + alpha_g)`` in the
    unconstrained coordinate.  ``observation_sd`` is the **likelihood-only**
    standard error of the per-session estimate; it sets the normal--normal
    shrinkage weight ``n / (n + (sigma/sigma_person)^2)``.

    ``already_shrunk=True`` says the offsets are posterior means from a fit that
    *already* carried the individual prior ``N(0, Sigma_person + Sigma_session)``
    -- which is what :func:`scwbd.individualize.fit.individualize` produces.
    Applying the normal--normal weight on top of that would shrink twice and
    quietly bias every individualized patient toward the population.  The factor
    is still computed and reported, as the diagnostic it is.

    ``mask`` restricts the whole decomposition to coordinates that were actually
    fitted: masked-out coordinates get exactly zero ``delta`` and ``zeta``, so
    ``theta`` is **bit-identical** to ``mu + alpha_g`` there.
    """
    P = len(population.parameter_names)
    offs = np.asarray(session_offsets, dtype=float).reshape(-1, P)
    n = offs.shape[0]
    if n == 0:
        raise ValueError("decompose_sessions needs at least one session")
    if mask is None:
        mask = np.ones(P, dtype=bool)
    mask = np.asarray(mask, dtype=bool).reshape(P)
    offs = np.where(mask.reshape(1, -1), offs, 0.0)

    person_sd = np.asarray(population.person_sd, dtype=float)
    if observation_sd is None:
        observation_sd = np.asarray(population.prior_sd, dtype=float)
    obs_sd = np.asarray(observation_sd, dtype=float).reshape(P)

    mean_off = offs.mean(0)
    with np.errstate(over="ignore", invalid="ignore"):
        ratio = np.nan_to_num(
            (obs_sd / np.maximum(person_sd, 1e-300)) ** 2, nan=np.inf, posinf=np.inf
        )
    shrink = n / (n + ratio)
    shrink = np.where(mask, shrink, 0.0)
    delta = mean_off if already_shrunk else shrink * mean_off
    delta = np.where(mask, delta, 0.0)

    if n == 1:
        # delta and zeta are not separately identified from one session.  The
        # honest encoding is: put the whole (shrunk) offset in delta, leave zeta
        # exactly zero, and say the split was NOT identified.
        zeta = np.zeros((1, P))
        separable = False
        reason = (
            "one session: delta_p (trait) and zeta_{p,s} (state) enter the "
            "likelihood only through their sum, so the split is NOT identified. "
            "The combined offset is reported as delta and must not be read as "
            "evidence that this patient's session effect is zero."
        )
    else:
        resid = offs - mean_off.reshape(1, -1)
        zeta = resid - resid.mean(0, keepdims=True)  # sum_s zeta = 0, exactly
        zeta = np.where(mask.reshape(1, -1), zeta, 0.0)
        separable = True
        reason = (
            f"{n} sessions: the within-patient session spread identifies zeta "
            "up to the sum-to-zero constraint, which is imposed exactly."
        )
    centering = float(np.abs(zeta.sum(0)).max())
    return Decomposition(
        parameter_names=tuple(population.parameter_names),
        mu=population.mu.copy(),
        alpha=population.alpha[population.group_index(group)].copy(),
        delta=delta,
        zeta=zeta,
        session_ids=tuple(session_ids),
        shrinkage_factor=shrink,
        separable=separable,
        separability_reason=reason,
        centering_residual=centering,
        shrinkage_applied_in_fit=bool(already_shrunk),
    )


# --------------------------------------------------------------------------
# compiler-facing declarations (R07)
# --------------------------------------------------------------------------
def hierarchical_effect_declarations(
    *,
    recovery_report: str = "reports/individualize/recovery.json",
    recovery_tested: bool = True,
    parameterization_group: str = "sum_to_zero",
    parameterization_person: str = "noncentered_hierarchical",
    parameterization_session: str = "noncentered_hierarchical",
    with_shrinkage: bool = True,
) -> tuple[Any, ...]:
    """The :class:`HierarchicalEffect` records the compiler's ``check_r07`` reads.

    The keyword arguments exist so a test can construct a **deliberately
    broken** declaration and watch ``check_r07`` fire.  A refusal nobody has
    seen fire is indistinguishable from one that cannot.
    """
    from scwbd.schema.priors import NormalPrior
    from scwbd.schema.sources import HierarchicalEffect

    def shrink(sd: float) -> Any:
        if not with_shrinkage:
            return None
        return NormalPrior(loc=0.0, scale=sd, provenance="body.tex sec. 6.5")

    return (
        HierarchicalEffect(
            name="alpha_group",
            level="site",
            parameterization=parameterization_group,  # type: ignore[arg-type]
            shrinkage_prior=shrink(0.3),
            recovery_tested=recovery_tested,
            recovery_report=recovery_report,
        ),
        HierarchicalEffect(
            name="delta_person",
            level="participant",
            parameterization=parameterization_person,  # type: ignore[arg-type]
            shrinkage_prior=shrink(DEFAULT_PERSON_SD),
            recovery_tested=recovery_tested,
            recovery_report=recovery_report,
        ),
        HierarchicalEffect(
            name="zeta_session",
            level="session",
            parameterization=parameterization_session,  # type: ignore[arg-type]
            shrinkage_prior=shrink(DEFAULT_SESSION_SD),
            recovery_tested=recovery_tested,
            recovery_report=recovery_report,
        ),
    )


# --------------------------------------------------------------------------
# simulated recovery of the decomposition
# --------------------------------------------------------------------------
def recover_decomposition(
    *,
    theta_dim: int = 4,
    n_groups: int = 3,
    n_participants: int = 48,
    n_sessions_per_participant: int = 4,
    noise_sd: float = 0.05,
    person_sd: float = 0.20,
    session_sd: float = 0.08,
    steps: int = 4000,
    lr: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit synthetic subjects and check ``delta_p`` is recovered and centering holds.

    Uses ``scwbd.foundation.individual.Individualizer`` -- agent Turing's module,
    consumed, not copied -- so the R07 enforcement under test is the one that
    ships.  Reports the correlation of the *parts*, not only of their sum: a
    decomposition that fits the data without being identified shows a high sum
    correlation and near-zero part correlations, which is precisely the failure
    R07 exists to catch.

    ``achievable_delta_corr`` is reported alongside the measured one.  It is the
    correlation a *perfect* shrinkage estimator would reach given the simulated
    noise, ``sqrt(sigma_p^2 / (sigma_p^2 + sigma_e^2))`` with
    ``sigma_e^2 = (sigma_zeta^2 + sigma_noise^2)/n_sessions``.  Without it the
    only available bar is a hand-chosen constant, and at ``steps=700`` the
    measured correlation ranged 0.71-0.93 **across seeds** -- a test whose
    verdict was decided by the random stream rather than by the estimator.
    ``converged`` is therefore reported and gates ``identified``.
    """
    import torch

    g = torch.Generator(device="cpu").manual_seed(int(seed))
    n_sess = n_participants * n_sessions_per_participant

    group_of = torch.randint(0, n_groups, (n_participants,), generator=g)
    counts = torch.bincount(group_of, minlength=n_groups).float()
    mu_t = torch.randn(theta_dim, generator=g) * 0.5
    a_raw = torch.randn(n_groups, theta_dim, generator=g) * 0.3
    w = (counts / counts.sum()).reshape(-1, 1)
    alpha_t = a_raw - (w * a_raw).sum(0, keepdim=True)
    delta_t = torch.randn(n_participants, theta_dim, generator=g) * person_sd
    zeta_t = torch.randn(n_sess, theta_dim, generator=g) * session_sd

    sess_participant = torch.arange(n_sess) // n_sessions_per_participant
    sess_group = group_of[sess_participant]
    y = (
        mu_t
        + alpha_t[sess_group]
        + delta_t[sess_participant]
        + zeta_t
        + torch.randn(n_sess, theta_dim, generator=g) * noise_sd
    )

    m = Individualizer(
        theta_dim,
        n_groups=n_groups,
        n_participants=n_participants,
        n_sessions=n_sess,
        group_counts=counts.tolist(),
        person_sd_init=person_sd,
        session_sd_init=session_sd,
    )
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    gi, pi, si = sess_group, sess_participant, torch.arange(n_sess)
    history: list[float] = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = m(gi, pi, si)
        loss = ((pred - y) ** 2).mean() / (2 * noise_sd**2) + m.prior_penalty() / n_sess
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    m.assert_centered(tol=1e-4)

    # convergence: the last decile must not still be improving materially
    tail = max(1, steps // 10)
    span = abs(history[0] - history[-1]) or 1.0
    last_decile_gain = (history[-tail] - history[-1]) / span
    converged = bool(last_decile_gain < 0.01)

    def corr(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.reshape(-1).double()
        b = b.reshape(-1).double()
        a = a - a.mean()
        b = b - b.mean()
        d = (a.norm() * b.norm()).clamp_min(1e-12)
        return float((a @ b) / d)

    with torch.no_grad():
        alpha_hat, delta_hat, zeta_hat = m.alpha, m.delta, m.zeta
        centering = float((w * alpha_hat).sum(0).abs().max())
        # sum-to-zero of the recovered session effects within each participant
        z_by_p = zeta_hat.reshape(n_participants, n_sessions_per_participant, -1)
        out = {
            "alpha_corr": corr(alpha_hat, alpha_t),
            "delta_corr": corr(delta_hat, delta_t),
            "zeta_corr": corr(zeta_hat, zeta_t),
            "sum_corr": corr(m(gi, pi, si), y),
            "delta_rmse": float((delta_hat - delta_t).pow(2).mean().sqrt()),
            "delta_true_sd": float(delta_t.std()),
            "group_centering_residual": centering,
            "within_person_session_mean_abs": float(
                z_by_p.mean(1).abs().mean()
            ),
            "n_participants": n_participants,
            "n_sessions": n_sess,
            "noise_sd": noise_sd,
            "person_sd": person_sd,
            "session_sd": session_sd,
            "seed": seed,
            "steps": steps,
            "final_loss": history[-1],
            "last_decile_gain": last_decile_gain,
            "converged": converged,
        }
    sigma_e2 = (session_sd**2 + noise_sd**2) / n_sessions_per_participant
    achievable = math.sqrt(person_sd**2 / (person_sd**2 + sigma_e2))
    out["achievable_delta_corr"] = achievable
    out["delta_corr_efficiency"] = out["delta_corr"] / achievable
    # Two bars, and both are needed.  The efficiency bar asks "did the
    # estimator extract what was extractable"; the absolute bar asks "was
    # anything extractable at all".  Efficiency alone passes in the
    # noise-swamped regime, where a perfect estimator also recovers nothing --
    # the estimator is then blameless and the decomposition is still not
    # identified, which is what the caller needs to be told.
    out["identified"] = bool(
        converged
        and out["alpha_corr"] > 0.8
        and out["delta_corr_efficiency"] > 0.9
        and out["delta_corr"] > 0.5
        and out["zeta_corr"] > 0.5
        and out["group_centering_residual"] < 1e-4
    )
    out["interpretation"] = (
        "sum_corr high with delta_corr/zeta_corr low would mean the "
        "decomposition fits without being identified; both are reported so "
        "that outcome cannot hide. delta_corr is judged against the correlation "
        "a perfect shrinkage estimator could reach at this noise level "
        "(achievable_delta_corr), not against a hand-chosen constant."
    )
    return out
