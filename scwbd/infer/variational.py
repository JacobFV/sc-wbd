"""The sliced-trajectory objective of ``body.tex`` equation (4), with the
honest caveat **implemented structurally** rather than written in prose.

Equation (4)::

    L_slice = sum_e { -E_q log p(Y_e | X_Se, theta_e, O_e, C_e)
                      + KL[ q_e(Z_e) || p(Z_e | U_e, C_e) ]
                      + R_ports + R_dynamics + R_cal }

and the thesis immediately qualifies it:

    "The first two terms are a negative evidence lower bound **only when the
    stated probability models are normalized and evaluable**.  Boundary, scale,
    distillation, and compatibility penalties remain auxiliary pseudo-losses;
    they are reported separately and do not inherit posterior-calibration
    claims from the generative likelihood."

That is refusal **R09**.  Here it is enforced by construction:

* :class:`SlicedObjective` keeps the two ELBO terms and the auxiliary penalties
  in *different containers*.  There is no code path that sums them into a
  number called an ELBO.
* :attr:`SlicedObjectiveValue.negative_elbo` raises
  :class:`~scwbd.infer.types.CalibrationClaimError` unless every contributing
  factor was declared ``normalized=True`` **and** ``evaluable=True``.
* A posterior produced from an objective that carried any pseudo-loss is
  emitted with ``posterior_class='generalized'`` (the field name agent A's
  compiler checks) and ``PosteriorSummary.kind='generalized_posterior'``;
  asking such an object for calibrated Bayesian semantics raises.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .types import CalibrationClaimError, PosteriorSummary, as_builtin

__all__ = [
    "GaussianFactor",
    "LossFactor",
    "PseudoLossFactor",
    "SlicedObjective",
    "SlicedObjectiveValue",
    "generalized_posterior_from",
    "gaussian_kl",
]

#: Roles whose factors are *never* generative likelihoods (Appendix D / T6).
PSEUDO_ROLES = frozenset(
    {"boundary_target", "distillation", "calibration", "scale", "compatibility"}
)
#: The compiler-visible tag.  Agent A's R09 check reads ``posterior_class``.
PosteriorClass = Literal["bayesian", "generalized", "pseudo"]


@dataclass
class LossFactor:
    """A term of equation (4).

    ``normalized``
        the factor is a properly normalised probability density in the data;
    ``evaluable``
        its normalising constant can actually be evaluated (not merely assumed).

    A factor that is not both is a pseudo-loss regardless of what it is called.
    """

    name: str
    value: Tensor
    role: str
    weight: float = 1.0
    normalized: bool = False
    evaluable: bool = False
    kind: Literal["likelihood", "kl", "penalty"] = "penalty"
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def is_generative(self) -> bool:
        return (
            self.normalized
            and self.evaluable
            and self.kind in ("likelihood", "kl")
            and self.role not in PSEUDO_ROLES
        )

    @property
    def weighted(self) -> Tensor:
        return self.weight * self.value


def PseudoLossFactor(name: str, value: Tensor, role: str, weight: float = 1.0, **kw):
    """A penalty that is explicitly *not* a normalised likelihood."""
    if role not in PSEUDO_ROLES:
        raise ValueError(
            f"role {role!r} is not one of the auxiliary roles {sorted(PSEUDO_ROLES)}; "
            "use LossFactor and declare normalized/evaluable honestly"
        )
    return LossFactor(name, value, role, weight, normalized=False, evaluable=False,
                      kind="penalty", **kw)


@dataclass
class GaussianFactor:
    """Diagonal Gaussian ``q_e(Z_e)`` over boundary/missing state and ``theta_e``."""

    mean: Tensor
    log_sd: Tensor

    def sample(self, n: int, generator: torch.Generator | None = None) -> Tensor:
        eps = torch.randn(
            n, *self.mean.shape, dtype=self.mean.dtype, device=self.mean.device,
            generator=generator,
        )
        return self.mean + torch.exp(self.log_sd) * eps

    @property
    def sd(self) -> Tensor:
        return torch.exp(self.log_sd)


def gaussian_kl(q: GaussianFactor, p: GaussianFactor) -> Tensor:
    """``KL[q || p]`` for diagonal Gaussians; normalised and evaluable."""
    vq, vp = q.sd**2, p.sd**2
    return 0.5 * (
        (vq / vp).sum(-1)
        + (((p.mean - q.mean) ** 2) / vp).sum(-1)
        - q.mean.shape[-1]
        + (p.log_sd - q.log_sd).sum(-1) * 2
    )


@dataclass
class SlicedObjectiveValue:
    """The evaluated objective, with generative and auxiliary parts separated."""

    expected_negative_log_likelihood: Tensor
    kl: Tensor
    pseudo_terms: dict[str, Tensor]
    pseudo_roles: dict[str, str]
    generative_ok: bool
    non_generative_factors: list[str]
    episode_weights: Tensor | None = None

    @property
    def total(self) -> Tensor:
        """The quantity actually optimised.  **Not** an ELBO in general."""
        return (
            self.expected_negative_log_likelihood
            + self.kl
            + sum(self.pseudo_terms.values(), torch.zeros_like(self.kl))
        )

    @property
    def elbo_part(self) -> Tensor:
        """The two terms that *may* be a negative ELBO -- unvalidated."""
        return self.expected_negative_log_likelihood + self.kl

    @property
    def negative_elbo(self) -> Tensor:
        """Refuses (R09) unless every contributing factor is normalised and
        evaluable, and no pseudo-loss was folded in."""
        if not self.generative_ok:
            raise CalibrationClaimError(
                "the first two terms of body.tex eq. (4) are a negative ELBO only "
                "when the stated probability models are normalized and evaluable; "
                f"these factors are not: {sorted(self.non_generative_factors)}.",
                offending_object="SlicedObjectiveValue.negative_elbo",
            )
        if self.pseudo_terms:
            raise CalibrationClaimError(
                "auxiliary pseudo-losses "
                f"{sorted(self.pseudo_terms)} are present; they are reported "
                "separately and never enter an ELBO.",
                offending_object="SlicedObjectiveValue.negative_elbo",
            )
        return self.elbo_part

    @property
    def posterior_class(self) -> PosteriorClass:
        return "bayesian" if (self.generative_ok and not self.pseudo_terms) else "generalized"

    def report(self) -> dict[str, Any]:
        return as_builtin(
            {
                "expected_negative_log_likelihood": self.expected_negative_log_likelihood,
                "kl": self.kl,
                "elbo_part": self.elbo_part,
                "pseudo_terms": {k: v for k, v in self.pseudo_terms.items()},
                "pseudo_roles": self.pseudo_roles,
                "total_objective": self.total,
                "posterior_class": self.posterior_class,
                "is_negative_elbo": self.posterior_class == "bayesian",
                "non_generative_factors": sorted(self.non_generative_factors),
                "caveat": (
                    "body.tex eq. (4): the first two terms are a negative ELBO only "
                    "when the stated probability models are normalized and evaluable; "
                    "boundary/scale/distillation/compatibility penalties are auxiliary "
                    "pseudo-losses reported separately (refusal R09)."
                ),
            }
        )


class SlicedObjective:
    """Assemble equation (4) for a set of episodes.

    ``episode_weights`` implements ``w_e`` of T6/Appendix D: *"a learned or
    preregistered reliability term constrained by effective sample size ...  It
    is not proportional to row count alone."*  Passing raw row counts is
    rejected.
    """

    def __init__(
        self,
        *,
        episode_weights: Mapping[str, float] | None = None,
        effective_sample_size: Mapping[str, float] | None = None,
    ) -> None:
        self.factors: list[LossFactor] = []
        self.episode_weights = dict(episode_weights or {})
        self.effective_sample_size = dict(effective_sample_size or {})

    def add(self, factor: LossFactor) -> "SlicedObjective":
        self.factors.append(factor)
        return self

    def add_likelihood(
        self, name: str, value: Tensor, *, normalized: bool = True,
        evaluable: bool = True, weight: float = 1.0, **prov: Any
    ) -> "SlicedObjective":
        self.factors.append(
            LossFactor(name, value, "likelihood", weight, normalized, evaluable,
                       "likelihood", prov)
        )
        return self

    def add_kl(self, name: str, value: Tensor, weight: float = 1.0) -> "SlicedObjective":
        self.factors.append(
            LossFactor(name, value, "prior", weight, True, True, "kl", {})
        )
        return self

    def add_pseudo(
        self, name: str, value: Tensor, role: str, weight: float = 1.0, **prov: Any
    ) -> "SlicedObjective":
        self.factors.append(PseudoLossFactor(name, value, role, weight, provenance=prov))
        return self

    def evaluate(self) -> SlicedObjectiveValue:
        nll = None
        kl = None
        pseudo: dict[str, Tensor] = {}
        roles: dict[str, str] = {}
        bad: list[str] = []
        for f in self.factors:
            if f.kind == "likelihood":
                if not f.is_generative:
                    bad.append(f.name)
                nll = f.weighted if nll is None else nll + f.weighted
            elif f.kind == "kl":
                if not f.is_generative:
                    bad.append(f.name)
                kl = f.weighted if kl is None else kl + f.weighted
            else:
                pseudo[f.name] = f.weighted
                roles[f.name] = f.role
        if nll is None or kl is None:
            raise ValueError(
                "equation (4) needs both a likelihood term and a KL term; a slice "
                "with neither is not an inference objective"
            )
        return SlicedObjectiveValue(
            expected_negative_log_likelihood=nll,
            kl=kl,
            pseudo_terms=pseudo,
            pseudo_roles=roles,
            generative_ok=not bad,
            non_generative_factors=bad,
        )


def generalized_posterior_from(
    value: SlicedObjectiveValue,
    names: Sequence[str],
    mean: np.ndarray,
    cov: np.ndarray,
    *,
    provenance: dict[str, Any] | None = None,
) -> PosteriorSummary:
    """Wrap an optimised slice into a posterior with the correct claim class.

    If any pseudo-loss contributed, the result is tagged
    ``posterior_class='generalized'`` / ``kind='generalized_posterior'`` and
    :meth:`PosteriorSummary.assert_calibrated_bayesian` will raise.  This is the
    same tag agent A's compiler checks for R09.
    """
    prov = dict(provenance or {})
    prov["posterior_class"] = value.posterior_class
    prov["objective"] = value.report()
    if value.posterior_class == "bayesian":
        return PosteriorSummary(list(names), mean, cov, kind="bayesian", provenance=prov)
    return PosteriorSummary(
        list(names), mean, cov,
        kind="generalized_posterior",
        pseudo_loss_terms=sorted(set(value.pseudo_terms) | set(value.non_generative_factors)),
        provenance=prov,
    )
