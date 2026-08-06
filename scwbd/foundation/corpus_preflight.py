"""Pre-generation checks for the simulated corpus.

The corpus is generated *by* the model, and it carries a ``theta`` label array
alongside every trajectory.  If a theta dimension does not reach the simulator,
the corpus teaches the amortised posterior to "recover" a parameter that did
nothing, and nothing downstream says so -- it surfaces much later as a
strangely wide, or strangely confident, posterior over that dimension with no
entry in any log explaining it.  ``simulate.ParameterMappingError`` already
refuses the case where a mapping writes a key the backend never reads.  It
cannot see the case this module exists for: a theta component whose effect is
multiplied by zero.

That is not hypothetical.  ``anat.gradient`` was all zeros on the real
414-parcel prior for exactly as long as nobody perturbed it, which made
``ei_gradient`` algebraically inert on every backend
(``ei = ei_global * ei_prior * (1 + ei_gradient * gradient)``).

The check is a **mechanism, not an instruction**: perturb each dimension over
its own prior range and require that something downstream moves.  It costs
milliseconds and it runs before the first shard is written.

Two levels, because one is not enough:

``check_theta_parameter_sensitivity``
    Does the dimension move any *backend parameter*?  Cheap and exact.  It is
    **not sufficient**: ``log_velocity`` never passes through
    ``_regional_theta`` at all -- it enters through the delay matrix -- so this
    level reports it as unmoved and is right to.

``check_theta_trajectory_sensitivity``
    Does the dimension move the *integrated trajectory*, more than re-drawing
    the noise seed does?  This is the one that answers the question, and it is
    the only one that can see ``log_velocity`` or a parameter whose motion a
    clamp eats.

Neither level is a claim that a dimension is *identifiable* -- that is a
question about the corpus and the posterior, not about one batch.  A dimension
that passes here can still be under-sampled by the generator; see
``check_theta_sampling_diversity``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import torch

from .anatomy import AnatomyPrior, load_anatomy
from .backends import resolve_backend
from .simulate import THETA_NAMES, CorpusSpec, ThetaPrior, _regional_theta, simulate_batch

__all__ = [
    "PARAMETER_LEVEL_EXEMPT",
    "InertThetaDimension",
    "SensitivityReport",
    "check_theta_parameter_sensitivity",
    "check_theta_trajectory_sensitivity",
    "check_theta_sampling_diversity",
    "preflight",
]

#: Dimensions that cannot move a backend parameter *by construction*, and why.
#:
#: Named explicitly rather than special-cased silently, because "the check does
#: not apply here" is exactly the shape that hides a real inertness.  A
#: dimension listed here is not excused from the trajectory level -- it is
#: excused only from the parameter level, and :func:`preflight` refuses to
#: clear it on the parameter level alone.
PARAMETER_LEVEL_EXEMPT: dict[str, str] = {
    "log_velocity": (
        "enters the integration through anat.delay_matrix(velocity) -> "
        "DelayedCoupling, never through _regional_theta. The parameter level is "
        "structurally blind to it; only the trajectory level can see it."
    ),
}


class InertThetaDimension(ValueError):
    """A theta dimension the simulator does not respond to.

    Raised rather than warned.  A corpus generated in this state is not
    salvageable after the fact: the labels are written into every shard.
    """


class DegenerateAnatomyPrior(ValueError):
    """An anatomy field carries no regional structure.

    Distinct from :class:`InertThetaDimension`, and the distinction is the whole
    point.  A gradient of *zeros* makes ``ei_gradient`` inert -- theta cancels
    algebraically.  A gradient that is *constant at a non-zero value* leaves
    ``ei_gradient`` perfectly identifiable (``1 + theta*c`` still moves with
    theta) while destroying every regional claim the corpus exists to support.
    The first is a labelling defect; the second is a science defect.  Collapsing
    them into one check would let the wrong repair look sufficient.

    The canonical instance: with the receptor volumes absent, every parcel falls
    back to the ignorance prior LogNormal(0, 0.7), whose mean is exactly
    exp(0.7**2/2) = 1.2776.  Nothing about theta is broken; the corpus is simply
    not about anatomy any more.
    """


@dataclass
class SensitivityReport:
    """What moved, per theta dimension.  Serialised into the corpus index."""

    level: str
    #: dim -> {backend -> scalar sensitivity}
    per_dim: dict[str, dict[str, float]] = field(default_factory=dict)
    #: dim -> list of backends on which it moved something
    movers: dict[str, list[str]] = field(default_factory=dict)
    inert: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "per_dim": self.per_dim,
            "movers": self.movers,
            "inert": self.inert,
            "notes": self.notes,
        }


def check_theta_parameter_sensitivity(
    anat: AnatomyPrior,
    *,
    backends: Sequence[str],
    prior: ThetaPrior | None = None,
    batch: int = 256,
    seed: int = 20260805,
) -> SensitivityReport:
    """Move each theta dim across its prior range; record which parameters move.

    Returns a report rather than raising: ``log_velocity`` legitimately moves
    nothing here, so "inert at this level" is a finding for the caller to
    interpret, not an error.  :func:`preflight` is what decides.
    """
    prior = prior or ThetaPrior()
    b = prior.bounds()
    theta0 = prior.sample(batch, seed=seed)
    rep = SensitivityReport(level="parameter")
    for k, name in enumerate(THETA_NAMES):
        per_backend: dict[str, float] = {}
        movers: list[str] = []
        for bk in backends:
            defaults = dict(resolve_backend(bk).defaults)
            lo, hi = theta0.clone(), theta0.clone()
            lo[:, k], hi[:, k] = b[k, 0], b[k, 1]
            p_lo = _regional_theta(lo, anat, bk, defaults=defaults)
            p_hi = _regional_theta(hi, anat, bk, defaults=defaults)
            worst = 0.0
            for pname in p_lo:
                a_, c_ = torch.broadcast_tensors(p_lo[pname], p_hi[pname])
                d = (c_ - a_).abs()
                scale = 0.5 * (a_.abs() + c_.abs()).mean().clamp_min(1e-12)
                worst = max(worst, float((d.mean() / scale).item()))
            per_backend[bk] = worst
            if worst > 0.0:
                movers.append(bk)
        rep.per_dim[name] = per_backend
        rep.movers[name] = movers
        if not movers:
            rep.inert.append(name)
    return rep


def check_theta_trajectory_sensitivity(
    anat: AnatomyPrior,
    *,
    spec: CorpusSpec,
    backends: Sequence[str] | None = None,
    prior: ThetaPrior | None = None,
    batch: int = 32,
    seed: int = 20260805,
    device: torch.device | None = None,
) -> SensitivityReport:
    """Move each theta dim across its prior range; record trajectory motion.

    Paired: ``simulate_batch`` derives its noise generator and its initial state
    from ``seed``, so two calls at the same seed differ **only** by theta.  The
    reference scale is a re-draw of the noise seed at fixed theta, so the
    reported number is "how far theta moves the simulator, in units of how far
    the noise does".
    """
    prior = prior or ThetaPrior()
    backends = list(backends if backends is not None else spec.backends)
    dev = device or torch.device(spec.device)
    anat = anat.to(dev)
    b = prior.bounds().to(dev)
    sub = replace(spec, batch=batch)
    theta0 = prior.sample(batch, seed=seed, device=dev)
    theta0[:, 1] = theta0[0, 1]  # generator invariant: one velocity per batch

    def run(th, s, bk):
        a, _ = simulate_batch(anat=anat, backend_name=bk, theta=th, spec=sub, seed=s, device=dev)
        return a

    rep = SensitivityReport(level="trajectory")
    floors: dict[str, float] = {}
    for bk in backends:
        A = run(theta0, seed, bk)
        A2 = run(theta0, seed + 1, bk)
        n = A.flatten().norm().clamp_min(1e-30)
        floors[bk] = float(((A2 - A).flatten().norm() / n).item())
    rep.notes["noise_floor"] = floors

    for k, name in enumerate(THETA_NAMES):
        per_backend: dict[str, float] = {}
        movers: list[str] = []
        for bk in backends:
            lo, hi = theta0.clone(), theta0.clone()
            lo[:, k], hi[:, k] = b[k, 0], b[k, 1]
            A_lo, A_hi = run(lo, seed, bk), run(hi, seed, bk)
            n = A_lo.flatten().norm().clamp_min(1e-30)
            d = float(((A_hi - A_lo).flatten().norm() / n).item())
            snr = d / max(floors[bk], 1e-30)
            per_backend[bk] = snr
            if snr > 0.0:
                movers.append(bk)
        rep.per_dim[name] = per_backend
        rep.movers[name] = movers
        if not movers:
            rep.inert.append(name)
    return rep


def check_theta_sampling_diversity(spec: CorpusSpec, *, n_shards: int) -> dict[str, Any]:
    """How many *distinct* values of each theta dim the corpus will contain.

    ``generate_corpus`` sets ``theta[:, 1] = theta[0, 1]`` -- one conduction
    velocity per batch, because a per-row ``(B,N,N)`` delay tensor costs tens of
    gigabytes.  One batch is one shard, so ``log_velocity`` gets **one draw per
    shard** while every other dimension gets one per trajectory.  Run 1 shipped
    37 distinct velocities against 37,843 values of ``log_G``.

    That is not inertness and this function does not refuse it.  It is an
    effective sample size, and it belongs in the index so that a wide
    ``log_velocity`` posterior is read as under-sampling rather than as a
    property of the brain.
    """
    n_traj = n_shards * spec.batch
    eff = {name: n_traj for name in THETA_NAMES}
    eff["log_velocity"] = n_shards
    return {
        "n_shards": n_shards,
        "n_trajectories": n_traj,
        "effective_distinct_values": eff,
        "velocity_deficit_factor": spec.batch,
        "why": (
            "generate_corpus shares one conduction velocity across a batch "
            "(simulate.py: theta[:, 1] = theta[0, 1]); one batch is one shard."
        ),
    }


def preflight(
    *,
    spec: CorpusSpec,
    anat: AnatomyPrior | None = None,
    trajectory_level: bool = True,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run every pre-generation check.  Raise on anything that would poison labels.

    A dimension is inert only if it moves **neither** a backend parameter
    **nor** a trajectory.  Passing the parameter level alone is not enough and
    failing it alone is not fatal -- ``log_velocity`` fails it by construction.
    """
    anat = anat if anat is not None else load_anatomy()
    out: dict[str, Any] = {}

    if not anat.is_biological():
        raise InertThetaDimension(
            f"anatomy provenance is {anat.provenance!r}: refusing to generate a corpus "
            "on the synthetic fallback."
        )
    out["anatomy"] = {
        "n_regions": anat.n_regions,
        "is_biological": anat.is_biological(),
        "gradient_std": float(anat.gradient.std().item()),
        "ei_prior_std": float(anat.ei_prior.std().item()),
        "timescale_prior_std": float(anat.timescale_prior.std().item()),
    }
    # Two different failures, deliberately not merged -- see DegenerateAnatomyPrior.
    for fld in ("gradient", "ei_prior", "timescale_prior"):
        v = getattr(anat, fld)
        if float(v.abs().max().item()) < 1e-8:
            raise InertThetaDimension(
                f"anat.{fld} is identically zero. Every theta dimension that "
                f"multiplies it cancels algebraically and is inert by construction."
            )
    for fld in ("gradient", "ei_prior", "timescale_prior"):
        v = getattr(anat, fld)
        # relative, not absolute: an exact `std == 0` test is a float-equality
        # test and a practically-constant field walks straight through it.
        rel = float((v.std() / (v.abs().mean() + 1e-12)).item())
        if rel < 1e-6:
            raise DegenerateAnatomyPrior(
                f"anat.{fld} has no regional structure (relative sd {rel:.3e}, "
                f"constant at {float(v.mean().item()):.4f}). The source maps did not "
                f"load; the corpus would carry anatomy in name only."
            )

    par = check_theta_parameter_sensitivity(anat, backends=spec.backends)
    out["parameter_level"] = par.as_dict()

    traj = None
    if trajectory_level:
        traj = check_theta_trajectory_sensitivity(anat, spec=spec, device=device)
        out["trajectory_level"] = traj.as_dict()

    # A dimension is condemned only when every level that CAN see it says it is
    # inert.  The parameter level is structurally blind to PARAMETER_LEVEL_EXEMPT
    # dimensions, so its silence about them is not evidence.
    if traj is not None:
        inert = [d for d in par.inert if d in traj.inert]
        unverified: list[str] = []
    else:
        inert = [d for d in par.inert if d not in PARAMETER_LEVEL_EXEMPT]
        unverified = [d for d in par.inert if d in PARAMETER_LEVEL_EXEMPT]
    out["inert"] = inert
    out["unverified_without_trajectory_level"] = unverified
    out["parameter_level_exempt"] = dict(PARAMETER_LEVEL_EXEMPT)
    if inert:
        raise InertThetaDimension(
            f"theta dimensions {inert} move neither a backend parameter nor a "
            "trajectory. Generating would label every trajectory with a parameter "
            "the simulator ignored. Fix the mapping or remove the dimension from "
            "THETA_NAMES; do not generate."
        )
    return out
