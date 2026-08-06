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
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Sequence

import torch
from torch import Tensor, nn

from .types import DTYPE, ParamPack, Prior, assert_solver_dtype, default_device, make_generator

__all__ = [
    "DynamicsBackend",
    "BackendInfo",
    "CouplingKind",
    "register_backend",
    "get_backend",
    "list_backends",
    "resolve_prior_field",
    "sample_prior_list",
]

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

    # -- adapters to the anatomy prior (agent C) ---------------------------
    #: Backend parameter names that carry the *excitatory* population timescale,
    #: in order of preference.  Only the first one present is modulated.
    timescale_params: ClassVar[tuple[str, ...]] = ("tau_E", "tau_e", "tau_s", "tau")

    #: ``name -> (low, high)`` in the parameter's own units: the range over which
    #: the backend is calibrated and numerically well-behaved.  Distinct from
    #: ``param_priors`` on purpose — everything in ``param_priors`` is *sampled*
    #: by :meth:`sample_theta`, whereas this only bounds values pushed in from
    #: outside (see :meth:`theta_from_prior`).  ``None`` means unbounded on that
    #: side.
    param_support: ClassVar[Mapping[str, tuple[float | None, float | None]]] = {}

    def support_of(self, name: str) -> tuple[float | None, float | None]:
        """Calibrated bounds for a parameter: ``param_support``, else its prior."""
        if name in self.param_support:
            lo, hi = self.param_support[name]
            return (None if lo is None else float(lo), None if hi is None else float(hi))
        p = self.param_priors.get(name)
        if p is None:
            return (None, None)
        lo, hi = getattr(p, "low", None), getattr(p, "high", None)
        return (None if lo is None else float(lo), None if hi is None else float(hi))

    @classmethod
    def from_prior(cls, brain_prior: Any, **kw: Any) -> "DynamicsBackend":
        """Instantiate a backend bound to a :class:`scwbd.anatomy.BrainPrior`.

        The prior is *stashed, not consumed*: its per-parcel distributions are
        read at sampling time by :meth:`theta_from_prior`, never snapshotted
        here.  That is deliberate — the anatomical maps behind the E/I proxy are
        still being revised upstream, and a cached copy would silently freeze a
        stale version of them into every trajectory we generate.
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
        apply: Sequence[str] = ("ei_ratio", "timescale", "velocity"),
    ) -> ParamPack:
        """Sample a batch of parameter sets carrying the anatomy prior's regional structure.

        ``brain_prior`` is duck-typed against :class:`scwbd.anatomy.BrainPrior`:

        ``ei_ratio_prior()``   per-parcel excitation/inhibition ratio (dimensionless)
        ``timescale_prior()``  per-parcel intrinsic timescale (seconds)
        ``velocity_prior()``   scalar conduction velocity (m/s)

        Each is a list of :class:`~scwbd.schema.priors.PriorBase` *distributions*,
        and each batch element gets its **own draw**.  The batch axis therefore
        carries genuine prior spread rather than one point estimate broadcast B
        times — which is the whole reason the batch axis exists.

        Two mappings here are not identities, and both are recorded in
        ``theta.provenance``:

        **E/I is inverted.**  The prior's ratio is *excitation over inhibition*
        (higher = more excitable).  The backend parameter spelled ``ei_ratio`` is
        a gain on the **inhibitory** term (``c_ei * ei_ratio`` in Wilson-Cowan,
        ``-J_i * ei_ratio * S_I`` in Wong-Wang), so it runs the other way.  We
        therefore set ``ei_ratio = centre / draw``.  Mapping the two directly
        because they share a name would invert the cortical E/I gradient
        end-to-end and still look entirely plausible.

        **Timescale is relative, not absolute.**  The prior describes intrinsic
        *autocorrelation* timescales (tens to hundreds of ms); a backend's
        ``tau_E``/``tau_e`` is a synaptic or membrane constant, a different
        physical quantity that happens to share the symbol.  Writing the prior's
        seconds straight into ``tau_e`` would be a category error and would push
        Wilson-Cowan far outside its calibrated regime, so the prior is applied
        as a dimensionless modulation about its own centre.  The absolute
        intrinsic timescale remains an emergent network property — which is what
        makes it a prediction we can be wrong about rather than an input.

        Both normalisations use the prior's own geometric centre, so an upstream
        recalibration of the underlying maps shifts the regional *pattern*
        without silently rescaling every parameter.
        """
        n_regions = self._prior_n_regions(brain_prior)
        theta = self.sample_theta(batch, n_regions, seed=seed, device=device)
        dev = theta.device

        if "ei_ratio" in apply and "ei_ratio" in self.defaults:
            name, priors = resolve_prior_field(brain_prior, "ei_ratio_prior", "ei_prior")
            if priors is not None:
                draw = sample_prior_list(priors, batch, seed=seed + 101, n_regions=n_regions)
                centre = _geometric_centre(priors)
                # inverted on purpose; see the docstring
                gain = centre / torch.as_tensor(draw, device=dev, dtype=theta.dtype).clamp_min(1e-6)
                theta.set("ei_ratio", gain)
                theta.provenance["ei_ratio"] = {
                    "source": f"{type(brain_prior).__name__}.{name}",
                    "transform": (
                        "backend ei_ratio (an inhibitory gain) = geometric_centre(prior) / draw; "
                        "the prior is excitation/inhibition and runs in the opposite direction"
                    ),
                    "centre": centre,
                    "sampled_per_batch_element": True,
                    **_provenance_index(priors),
                }

        if "timescale" in apply:
            key = next((k for k in self.timescale_params if k in self.defaults), None)
            name, priors = resolve_prior_field(brain_prior, "timescale_prior")
            if key is not None and priors is not None:
                draw = sample_prior_list(priors, batch, seed=seed + 202, n_regions=n_regions)
                centre = _geometric_centre(priors)
                scale = torch.as_tensor(draw, device=dev, dtype=theta.dtype) / centre
                tau = scale * float(self.defaults[key])
                # The hierarchy prior is much wider than the backend's calibrated
                # range, and its lower tail can drive tau below the step size,
                # where an explicit solver simply rings.  Clamp to the backend's
                # own declared support and *report* how much was clamped rather
                # than emitting quietly unstable trajectories.
                lo, hi, n_clamped = _clamp_to_support(tau, *self.support_of(key))
                theta.set(key, tau)
                theta.provenance[key] = {
                    "source": f"{type(brain_prior).__name__}.{name}",
                    "transform": (
                        f"{key} = {float(self.defaults[key]):g} s * draw / geometric_centre(prior); "
                        "applied as a dimensionless modulation because the prior is an intrinsic "
                        "autocorrelation timescale and this parameter is a synaptic constant"
                    ),
                    "centre_s": centre,
                    "backend_default_s": float(self.defaults[key]),
                    "clamped_to_support": [lo, hi],
                    "n_clamped": n_clamped,
                    "fraction_clamped": n_clamped / max(1, batch * n_regions),
                    "sampled_per_batch_element": True,
                    **_provenance_index(priors),
                }

        if "velocity" in apply:
            name, vp = resolve_prior_field(brain_prior, "velocity_prior")
            if vp is not None and hasattr(vp, "sample"):
                v = sample_prior_list([vp], batch, seed=seed + 303, n_regions=1)
                theta.set("velocity", torch.as_tensor(v, device=dev, dtype=theta.dtype).reshape(batch, 1))
                theta.provenance["velocity"] = {
                    "source": f"{type(brain_prior).__name__}.{name}",
                    "transform": "conduction velocity in m/s, one draw per parameter set",
                    "sampled_per_batch_element": True,
                    **_provenance_index([vp]),
                }

        return theta

    @staticmethod
    def _prior_n_regions(brain_prior: Any) -> int:
        for attr in ("n_parcels", "n_regions"):
            v = getattr(brain_prior, attr, None)
            if v:
                return int(v)
        w = getattr(brain_prior, "weights", None)
        if w is not None:
            return int(w.shape[-1])
        raise ValueError(
            "cannot determine region count from the prior: expected n_parcels, n_regions or weights"
        )


def resolve_prior_field(brain_prior: Any, *names: str) -> tuple[str | None, Any]:
    """Fetch a prior field that may be a zero-arg method, a property, or an array.

    Returns ``(attribute_name, value)``, or ``(None, None)`` if none of ``names``
    is present.  Absent fields yield ``None`` rather than a fabricated default:
    a prior we do not have must not become a number we pretend to have.
    """
    for name in names:
        obj = getattr(brain_prior, name, None)
        if obj is None:
            continue
        if callable(obj):
            try:
                obj = obj()
            except TypeError:
                continue
        if obj is not None:
            return name, obj
    return None, None


def _derive_seed(seed: int, i: int) -> int:
    """Per-parcel seed: independent across parcels, reproducible across runs."""
    return int((int(seed) * 1_000_003 + 7 * i + 11) % (2**63 - 1))


def sample_prior_list(priors: Any, batch: int, *, seed: int, n_regions: int | None = None):
    """Draw ``(batch, N)`` from a per-parcel list of prior distributions.

    Accepts either a sequence of :class:`~scwbd.schema.priors.PriorBase` (each
    parcel sampled independently from its own distribution, with a derived seed
    per parcel) or a plain per-parcel array of point values, which is broadcast
    across the batch.  The distribution path is the one that matters: it is what
    makes each batch element a distinct parameter set drawn from the prior.
    """
    import numpy as np

    seq = list(priors) if not isinstance(priors, (str, bytes)) else []
    if seq and hasattr(seq[0], "sample"):
        cols = [
            np.asarray(p.sample(_derive_seed(seed, i), batch), dtype=float).reshape(batch)
            for i, p in enumerate(seq)
        ]
        return np.stack(cols, axis=1)
    arr = np.asarray(priors, dtype=float).reshape(1, -1)
    if n_regions is not None and arr.shape[1] != n_regions:
        raise ValueError(f"prior array has {arr.shape[1]} entries but the model has {n_regions} regions")
    return np.broadcast_to(arr, (batch, arr.shape[1])).copy()


def _geometric_centre(priors: Any) -> float:
    """Geometric mean of a prior list's own central values.

    Used as the normalising constant so that a parcel at the prior's centre maps
    to a modulation of exactly 1.0.  Anchoring to the prior rather than to the
    sampled batch keeps the mapping deterministic and independent of batch size.
    """
    import numpy as np

    seq = list(priors)
    vals = []
    for p in seq:
        m = float(p.mean()) if hasattr(p, "mean") else float(p)
        if math.isfinite(m) and m > 0.0:
            vals.append(m)
    if not vals:
        return 1.0
    return float(np.exp(np.mean(np.log(vals))))


def _clamp_to_support(
    x: Tensor, lo: float | None, hi: float | None
) -> tuple[float | None, float | None, int]:
    """Clamp ``x`` in place to a declared support; report how much moved.

    Returns ``(low, high, n_clamped)``.  A parameter with no declared bounds is
    left alone — we clamp to stated calibration, never to a guess.
    """
    if lo is None and hi is None:
        return None, None, 0
    outside = torch.zeros_like(x, dtype=torch.bool)
    if lo is not None:
        outside |= x < lo
    if hi is not None:
        outside |= x > hi
    n = int(outside.sum())
    x.clamp_(min=lo, max=hi)
    return lo, hi, n


def _provenance_index(priors: Any) -> dict[str, Any]:
    """Compress per-parcel citation strings to a distinct list plus an index.

    Keeps every parcel's citation recoverable without storing N copies of the
    same paragraph, and makes the "no receptor coverage" parcels visible as a
    distinct provenance entry rather than blending them into the covered ones.
    """
    texts: list[str] = []
    idx: list[int] = []
    for p in list(priors):
        t = str(getattr(p, "provenance", "") or "")
        if t not in texts:
            texts.append(t)
        idx.append(texts.index(t))
    return {"distinct_provenance": texts, "parcel_provenance_index": idx}


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
