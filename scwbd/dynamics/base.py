"""The common dynamical-backend interface.

Every mechanistic backend and the learned surrogate implement *exactly* this
interface, so that a mechanistic claim can be tested against an equal-capacity
generic operator (thesis §4.3/§4.4, Appendix D "Operator / mechanism claim").
A backend earns a mechanistic label only by predictions the surrogate misses;
the interface is what makes that comparison mechanical rather than rhetorical.

    state_dim                        int, per region
    init_state(batch, n_regions, seed)  -> (B, N, D)
    drift(x, coupling_input, theta, u)  -> (B, N, D)
    diffusion(x, theta)                 -> (B, N, D)   diagonal noise amplitude
    observables(x)                      -> dict[str, Tensor]
    coupling_variable(x, theta)         -> (B, N)      what travels along edges

All methods are batched over parameter sets and vectorised over regions: there
are no Python loops over regions anywhere in a backend.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Sequence

import torch
from torch import Tensor, nn

from .types import DTYPE, ParamPack, Prior, assert_solver_dtype, default_device, make_generator

__all__ = ["DynamicsBackend", "BackendInfo", "CouplingKind", "register_backend", "get_backend", "list_backends"]

CouplingKind = str  # "additive" | "phase_difference"


@dataclass(frozen=True)
class BackendInfo:
    """Provenance and epistemic status of a backend (mirrors OperatorSpec)."""

    name: str
    family: str  # flow_ode | flow_pde | delayed_ssm | surrogate | ...
    mechanistic_status: str  # mechanistic | effective | functional | surrogate
    state_names: tuple[str, ...]
    units: tuple[str, ...]
    reference: str = ""
    #: what empirical finding would disable this backend (claim gate §4)
    falsifier: str = ""


_REGISTRY: dict[str, type["DynamicsBackend"]] = {}


def register_backend(cls: type["DynamicsBackend"]) -> type["DynamicsBackend"]:
    _REGISTRY[cls.info.name] = cls
    return cls


def get_backend(name: str) -> type["DynamicsBackend"]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_backends() -> list[str]:
    return sorted(_REGISTRY)


class DynamicsBackend(nn.Module, abc.ABC):
    """Batched, GPU-resident, differentiable regional dynamics.

    Subclasses are ``nn.Module`` so learned backends and learned residuals slot
    into the same container, but a mechanistic backend holds **no** learnable
    parameters: its parameters live in the batched :class:`ParamPack` so that
    thousands of parameter sets integrate in parallel.
    """

    info: ClassVar[BackendInfo]
    #: number of coupling channels the backend consumes
    n_coupling_channels: ClassVar[int] = 1
    #: how the coupling module must combine source and destination
    coupling_kind: ClassVar[CouplingKind] = "additive"
    #: default parameter values (biophysical literature values)
    defaults: ClassVar[Mapping[str, float]] = {}
    #: priors used by :meth:`sample_theta` for simulation-based training
    param_priors: ClassVar[Mapping[str, Prior]] = {}
    #: parameters that are naturally heterogeneous across regions
    regional_params: ClassVar[tuple[str, ...]] = ()
    #: True for learned propagators.  A learned propagator is not assumed to
    #: form a semigroup: the scheduler must hold a semigroup certificate before
    #: it may adapt the step or substitute a coarse step for fine ones (R06).
    learned: ClassVar[bool] = False

    # -- shape / identity --------------------------------------------------
    @property
    @abc.abstractmethod
    def state_dim(self) -> int: ...

    @property
    def state_names(self) -> tuple[str, ...]:
        return self.info.state_names

    # -- required interface ------------------------------------------------
    @abc.abstractmethod
    def init_state(
        self,
        batch: int,
        n_regions: int,
        *,
        seed: int,
        device: str | torch.device | None = None,
        dtype: torch.dtype = DTYPE,
        theta: ParamPack | None = None,
    ) -> Tensor:
        """Return ``(B, N, D)`` initial state.  Deterministic given ``seed``."""

    @abc.abstractmethod
    def drift(
        self,
        x: Tensor,
        coupling_input: Tensor,
        theta: ParamPack,
        u: Tensor | None = None,
        t: float = 0.0,
    ) -> Tensor:
        """Deterministic vector field ``f(x, c, theta, u)`` of shape ``(B, N, D)``."""

    @abc.abstractmethod
    def diffusion(self, x: Tensor, theta: ParamPack) -> Tensor:
        """Diagonal noise amplitude ``g(x, theta)``, shape ``(B, N, D)``.

        Diagonal noise keeps Euler–Maruyama strong order 0.5 / Milstein
        implementable without Lévy areas.  Backends with non-diagonal noise must
        say so by overriding :attr:`diagonal_noise`.
        """

    diagonal_noise: ClassVar[bool] = True

    @abc.abstractmethod
    def observables(self, x: Tensor) -> dict[str, Tensor]:
        """Named observables.  Must contain ``'activity'`` (B, N).

        ``activity`` is the quantity handed to the hemodynamic and (agent F)
        electromagnetic heads; it is *not* automatically the coupling variable.
        """

    def coupling_variable(self, x: Tensor, theta: ParamPack | None = None) -> Tensor:
        """What propagates along long-range edges, shape ``(B, N, C)``.

        ``C == n_coupling_channels``.  Defaults to ``observables(x)['activity']``
        with a channel axis.  Kept separate from ``activity`` because the
        variable a tract transmits (firing rate, phase) is not in general the
        variable an instrument observes (PSP, BOLD-driving synaptic activity).
        """
        return self.observables(x)["activity"].unsqueeze(-1)

    # -- conveniences ------------------------------------------------------
    def make_theta(
        self,
        batch: int,
        n_regions: int,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = DTYPE,
        **overrides: Tensor | float,
    ) -> ParamPack:
        return ParamPack(
            values=dict(overrides),
            batch=batch,
            n_regions=n_regions,
            device=default_device(device),
            dtype=dtype,
            defaults=dict(self.defaults),
        )

    def sample_theta(
        self,
        batch: int,
        n_regions: int,
        *,
        seed: int,
        device: str | torch.device | None = None,
        dtype: torch.dtype = DTYPE,
        extra_priors: Mapping[str, Prior] | None = None,
    ) -> ParamPack:
        priors = dict(self.param_priors)
        if extra_priors:
            priors.update(extra_priors)
        pack = ParamPack.sample(
            priors,
            batch=batch,
            n_regions=n_regions,
            seed=seed,
            per_region=self.regional_params,
            device=device,
            dtype=dtype,
            defaults=dict(self.defaults),
        )
        return pack

    def zero_coupling(self, x: Tensor) -> Tensor:
        return x.new_zeros(x.shape[0], x.shape[1], self.n_coupling_channels)

    def check_state(self, x: Tensor) -> None:
        assert_solver_dtype(x, f"{self.info.name} state")
        if x.ndim != 3 or x.shape[-1] != self.state_dim:
            raise ValueError(
                f"{self.info.name} expects state of shape (B, N, {self.state_dim}); got {tuple(x.shape)}"
            )

    def capacity(self) -> int:
        """Learnable parameter count — the currency of equal-capacity controls."""
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.info.name,
            "family": self.info.family,
            "mechanistic_status": self.info.mechanistic_status,
            "state_dim": self.state_dim,
            "state_names": list(self.state_names),
            "units": list(self.info.units),
            "coupling_kind": self.coupling_kind,
            "n_coupling_channels": self.n_coupling_channels,
            "capacity": self.capacity(),
            "reference": self.info.reference,
            "falsifier": self.info.falsifier,
        }

    # -- adapters to agent C / agent A ------------------------------------
    @classmethod
    def from_prior(cls, brain_prior: Any, **kw: Any) -> "DynamicsBackend":
        """Instantiate using ``scwbd.anatomy.BrainPrior`` regional priors.

        Duck-typed: any object exposing ``ei_prior`` (regional E/I ratio) and/or
        ``timescale_prior`` (intrinsic timescale) is accepted.  Subclasses may
        override to map those onto their own parameters; the base version just
        constructs the backend and stashes the prior for
        :meth:`theta_from_prior`.
        """
        obj = cls(**kw)
        obj._brain_prior = brain_prior  # type: ignore[attr-defined]
        return obj

    def theta_from_prior(
        self,
        brain_prior: Any,
        batch: int,
        *,
        seed: int,
        device: str | torch.device | None = None,
    ) -> ParamPack:
        """Sample regional heterogeneity from agent C's priors.

        Recognised attributes on ``brain_prior``: ``n_regions``, ``ei_prior``
        (per-region excitation/inhibition ratio, ``(N,)`` tensor or Prior list),
        ``timescale_prior`` (per-region intrinsic timescale in seconds).
        Unknown attributes are ignored rather than silently faked.
        """
        n_regions = int(getattr(brain_prior, "n_regions", 0)) or int(
            getattr(brain_prior, "weights").shape[-1]
        )
        theta = self.sample_theta(batch, n_regions, seed=seed, device=device)
        dev = theta.device
        ei = getattr(brain_prior, "ei_prior", None)
        if ei is not None and "ei_ratio" in self.defaults:
            ei_t = torch.as_tensor(ei, device=dev, dtype=theta.dtype).reshape(1, -1)
            theta.set("ei_ratio", ei_t.expand(batch, -1))
        tau = getattr(brain_prior, "timescale_prior", None)
        if tau is not None:
            tau_t = torch.as_tensor(tau, device=dev, dtype=theta.dtype).reshape(1, -1)
            for key in ("tau", "tau_e", "tau_E"):
                if key in self.defaults:
                    theta.set(key, tau_t.expand(batch, -1) * float(self.defaults[key]) / float(tau_t.mean()))
                    break
        return theta


def sigmoid_offset(x: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Zero-baseline logistic used by Wilson–Cowan: ``S(x) - S(0)`` normalised."""
    s = torch.sigmoid(a * (x - b))
    s0 = torch.sigmoid(-a * b)
    return (s - s0) / (1.0 - s0).clamp_min(1e-12)


def random_like(
    shape: Sequence[int],
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype = DTYPE,
    kind: str = "randn",
) -> Tensor:
    g = make_generator(seed, device)
    if kind == "randn":
        return torch.randn(tuple(shape), generator=g, device=device, dtype=dtype)
    return torch.rand(tuple(shape), generator=g, device=device, dtype=dtype)
