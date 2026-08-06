"""``theta = (state, gain, synapse, structure)`` — four parameter classes, four clocks.

Thesis §3.2: the macro-connectome is a slowly changing *scaffold*, not an
immutable graph and not a dense matrix to descend on.  The four classes have
progressively **stronger priors** and **slower update clocks**:

===========  =========  ==================  =========================================
class        clock      prior strength      what it is
===========  =========  ==================  =========================================
state        ~1 ms      none                the dynamical state itself
gain         ~0.1–10 s  weak               excitability / neuromodulatory gain
synapse      ~min–h     moderate           efficacy on *existing* pathways
structure    ~days      very strong        the existence of a mesoscale pathway
===========  =========  ==================  =========================================

Structural edits are **constrained graph rewriting**, not gradient descent on a
dense matrix.  A candidate edit is proposed only when a persistent residual
cannot be explained by state, measurement, or effective-coupling uncertainty,
and it is then evaluated during replay against prior competencies, energetic
cost, anatomical plausibility, and stability.  An accepted edit enters the graph
as a **proposed** edge (evidence class), never as hard-supported: adding an edge
because it improved a fit is not anatomical evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Sequence

import torch
from torch import Tensor, nn

from .coupling import EDGE_CLASS_CODES, DelayedConnectome, EdgeSet
from .scheduler import FieldPolicy
from .types import DTYPE, ParamPack, default_device

__all__ = [
    "ThetaClass",
    "THETA_CLOCKS",
    "theta_field_policies",
    "GainController",
    "SynapticPlasticity",
    "ResidualEvidence",
    "EditProposal",
    "EditDecision",
    "StructuralRewriter",
]

ThetaClass = Literal["state", "gain", "synapse", "structure"]

#: nominal update intervals in seconds (the "different clocks" of §3.2)
THETA_CLOCKS: dict[str, float] = {
    "state": 1e-3,
    "gain": 1.0,
    "synapse": 60.0,
    "structure": 86400.0,
}

#: prior precision multipliers — stronger priors for slower classes
THETA_PRIOR_PRECISION: dict[str, float] = {
    "state": 0.0,
    "gain": 1.0,
    "synapse": 10.0,
    "structure": 1000.0,
}


def theta_field_policies(scale: float = 1.0) -> list[FieldPolicy]:
    """Multirate policies for the four parameter classes.

    ``scale`` compresses the clocks for simulation studies (a 24 h structural
    clock is not runnable in a unit test); the *ratios* are preserved, and the
    compression factor is reported so it enters the ledger rather than being
    forgotten.
    """
    return [
        FieldPolicy(
            name="theta_gain",
            dt=THETA_CLOCKS["gain"] * scale,
            kind="continuous",
            interpolation="zoh",
            units="dimensionless",
            description="homeostatic/neuromodulatory gain; weak prior",
        ),
        FieldPolicy(
            name="theta_synapse",
            dt=THETA_CLOCKS["synapse"] * scale,
            kind="lazy",
            interpolation="zoh",
            inputs=("theta_gain",),
            materiality=0.05,
            units="dimensionless",
            description="synaptic efficacy on existing pathways; moderate prior",
        ),
        FieldPolicy(
            name="theta_structure",
            dt=THETA_CLOCKS["structure"] * scale,
            kind="event_driven",
            interpolation="zoh",
            units="dimensionless",
            description="constrained graph rewriting; very strong prior, event-driven only",
        ),
    ]


# ---------------------------------------------------------------------------
# theta^gain — homeostatic excitability control
# ---------------------------------------------------------------------------


class GainController(nn.Module):
    """Slow integral control of regional gain towards a target activity level.

    ``dg/dt = -(a_bar - a_target) / tau_gain``, with ``a_bar`` a leaky running
    mean of activity.  This is the mechanism that keeps a heterogeneous network
    inside its dynamic range without hand-tuning each region, and it is the
    honest place for "regional E/I balance" to live: it is a *parameter with a
    clock*, not a fitted constant.
    """

    def __init__(
        self,
        n_regions: int,
        batch: int = 1,
        *,
        target: float | Tensor = 0.1,
        tau_gain: float = 10.0,
        tau_avg: float = 1.0,
        g_min: float = 0.2,
        g_max: float = 5.0,
        device: str | torch.device | None = None,
    ):
        super().__init__()
        dev = default_device(device)
        self.register_buffer("gain", torch.ones(batch, n_regions, device=dev, dtype=DTYPE))
        self.register_buffer("avg", torch.zeros(batch, n_regions, device=dev, dtype=DTYPE))
        self.register_buffer(
            "target",
            (torch.as_tensor(target, device=dev, dtype=DTYPE).expand(batch, n_regions).clone()),
        )
        self.tau_gain, self.tau_avg = float(tau_gain), float(tau_avg)
        self.g_min, self.g_max = float(g_min), float(g_max)
        self.initialized = False

    @torch.no_grad()
    def update(self, activity: Tensor, dt: float) -> Tensor:
        """``activity``: ``(B, N)``.  Returns the updated gain ``(B, N)``."""
        a = activity.detach()
        if not self.initialized:
            self.avg.copy_(a)
            self.initialized = True
        else:
            alpha = min(dt / self.tau_avg, 1.0)
            self.avg.mul_(1 - alpha).add_(alpha * a)
        err = self.avg - self.target
        self.gain.add_(-(dt / self.tau_gain) * err).clamp_(self.g_min, self.g_max)
        return self.gain

    def apply_to(self, theta: ParamPack, name: str = "ei_ratio") -> ParamPack:
        return theta.with_(**{name: self.gain})

    def error(self) -> Tensor:
        return self.avg - self.target


# ---------------------------------------------------------------------------
# theta^synapse — efficacy on existing pathways
# ---------------------------------------------------------------------------


class SynapticPlasticity(nn.Module):
    """Hebbian/BCM efficacy updates on **existing** edges, with shrinkage.

    Operates on the edge list, so it can never create a pathway — that is the
    structural class's job and it has a much stronger prior.  ``hard`` edges are
    excluded from the learnable set (§2.2: a hard-supported pathway's existence
    is fixed; only soft/proposed efficacy moves), and every update is shrunk
    towards the anatomical prior weight.

    ``rule``:
      ``hebb``  dw = eta * (pre * post) - decay * (w - w_prior)
      ``bcm``   dw = eta * pre * post * (post - theta_m) - decay * (w - w_prior)
                with a sliding threshold ``theta_m = <post^2>``
      ``oja``   dw = eta * post * (pre - post * w)
    """

    def __init__(
        self,
        edges: EdgeSet,
        *,
        rule: Literal["hebb", "bcm", "oja"] = "bcm",
        eta: float = 1e-3,
        decay: float = 1e-2,
        w_min: float = 0.0,
        w_max: float = 5.0,
        tau_theta: float = 10.0,
        batch: int = 1,
        freeze_hard: bool = True,
    ):
        super().__init__()
        if rule not in ("hebb", "bcm", "oja"):
            raise ValueError(f"unknown plasticity rule {rule!r}")
        self.edges = edges
        self.rule = rule
        self.eta, self.decay = float(eta), float(decay)
        self.w_min, self.w_max = float(w_min), float(w_max)
        self.tau_theta = float(tau_theta)
        w0 = edges.weight if edges.weight.ndim == 2 else edges.weight.unsqueeze(0).expand(batch, -1)
        self.register_buffer("w", w0.clone())
        self.register_buffer("w_prior", w0.clone())
        self.register_buffer(
            "plastic",
            (edges.evidence != EDGE_CLASS_CODES["hard"]).to(DTYPE) if freeze_hard
            else torch.ones_like(edges.distance_mm),
        )
        self.register_buffer("theta_m", torch.zeros(batch, edges.n_regions, device=w0.device, dtype=DTYPE))
        self.n_updates = 0

    @torch.no_grad()
    def update(self, activity: Tensor, dt: float) -> Tensor:
        """``activity``: ``(B, N)`` -> updated per-edge weights ``(B, E)``."""
        a = activity.detach()
        pre = a.index_select(1, self.edges.src)
        post = a.index_select(1, self.edges.dst)
        alpha = min(dt / self.tau_theta, 1.0)
        self.theta_m.mul_(1 - alpha).add_(alpha * a.pow(2))
        if self.rule == "hebb":
            dw = pre * post
        elif self.rule == "bcm":
            thr = self.theta_m.index_select(1, self.edges.dst)
            dw = pre * post * (post - thr)
        else:  # oja
            dw = post * (pre - post * self.w)
        dw = self.eta * dw - self.decay * (self.w - self.w_prior)
        self.w.add_(dt * dw * self.plastic.unsqueeze(0)).clamp_(self.w_min, self.w_max)
        self.n_updates += 1
        return self.w

    def install(self, connectome: DelayedConnectome) -> None:
        """Write the current efficacies back into the coupling operator."""
        connectome.w_base = self.w

    def deviation(self) -> Tensor:
        return self.w - self.w_prior


# ---------------------------------------------------------------------------
# theta^structure — constrained graph rewriting
# ---------------------------------------------------------------------------


@dataclass
class ResidualEvidence:
    """Accumulated evidence that a persistent residual is *not* explained away.

    A residual only counts as structural evidence once it exceeds the combined
    state + measurement + effective-coupling uncertainty for a sustained
    fraction of a window.  This is the gate that keeps "the fit improved" from
    becoming "the anatomy changed".
    """

    n_windows: int = 0
    n_exceeding: int = 0
    residual_sum: Tensor | None = None
    last_residual: Tensor | None = None
    persistence: Tensor | None = None  # (N,) or (N,N) fraction of windows exceeding

    def summary(self) -> dict[str, float]:
        return {
            "n_windows": float(self.n_windows),
            "n_exceeding": float(self.n_exceeding),
            "max_persistence": float(self.persistence.max()) if self.persistence is not None else 0.0,
        }


@dataclass
class EditProposal:
    """A candidate structural edit.  Nothing is applied at proposal time."""

    kind: Literal["add", "remove", "retarget"]
    src: int
    dst: int
    distance_mm: float
    evidence: float  # persistence-weighted residual magnitude
    persistence: float
    reason: str = ""

    def key(self) -> tuple[str, int, int]:
        return (self.kind, self.src, self.dst)


@dataclass
class EditDecision:
    """The outcome of replay evaluation.  Every criterion is recorded separately."""

    proposal: EditProposal
    accepted: bool
    competency_delta: float  # change in prior-competency replay loss (< 0 is better)
    energetic_cost: float
    anatomical_plausibility: float  # log-prior of the edge under the anatomy prior
    stability_margin: float  # spectral margin after the edit; must stay > 0
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.proposal.kind,
            "src": self.proposal.src,
            "dst": self.proposal.dst,
            "accepted": self.accepted,
            "competency_delta": self.competency_delta,
            "energetic_cost": self.energetic_cost,
            "anatomical_plausibility": self.anatomical_plausibility,
            "stability_margin": self.stability_margin,
            "evidence": self.proposal.evidence,
            "persistence": self.proposal.persistence,
            "reasons": list(self.reasons),
        }


class StructuralRewriter:
    """Propose -> evaluate-during-replay -> (maybe) apply.  Never gradient descent.

    ``propose`` runs the uncertainty gate; ``evaluate`` runs the four
    acceptance criteria against a replay callback; ``apply`` returns a **new**
    :class:`EdgeSet` with accepted additions marked ``proposed``.
    """

    def __init__(
        self,
        *,
        persistence_threshold: float = 0.6,
        evidence_threshold: float = 1.0,
        max_edits_per_round: int = 4,
        max_distance_mm: float = 120.0,
        competency_tolerance: float = 0.0,
        energetic_cost_per_mm: float = 0.01,
        energetic_budget: float = 1.0,
        min_stability_margin: float = 1e-3,
        min_anatomical_plausibility: float = -6.0,
    ):
        self.persistence_threshold = float(persistence_threshold)
        self.evidence_threshold = float(evidence_threshold)
        self.max_edits_per_round = int(max_edits_per_round)
        self.max_distance_mm = float(max_distance_mm)
        self.competency_tolerance = float(competency_tolerance)
        self.energetic_cost_per_mm = float(energetic_cost_per_mm)
        self.energetic_budget = float(energetic_budget)
        self.min_stability_margin = float(min_stability_margin)
        self.min_anatomical_plausibility = float(min_anatomical_plausibility)
        self.history: list[EditDecision] = []

    # -- gate --------------------------------------------------------------
    @staticmethod
    def accumulate_evidence(
        residual: Tensor,
        uncertainty: Tensor,
        evidence: ResidualEvidence | None = None,
    ) -> ResidualEvidence:
        """Add one window of evidence.

        ``residual``: ``(N, N)`` (or ``(N,)``) unexplained structure; typically
        the residual cross-covariance between regions after the current model.
        ``uncertainty``: the *combined* state + measurement + effective-coupling
        uncertainty at the same shape.  Only ``|residual| > uncertainty``
        counts.
        """
        ev = evidence or ResidualEvidence()
        exceed = (residual.abs() > uncertainty).to(residual.dtype)
        ev.n_windows += 1
        ev.n_exceeding += int(exceed.sum())
        ev.residual_sum = exceed * residual.abs() if ev.residual_sum is None else ev.residual_sum + exceed * residual.abs()
        ev.last_residual = residual
        prev = ev.persistence if ev.persistence is not None else torch.zeros_like(exceed)
        ev.persistence = (prev * (ev.n_windows - 1) + exceed) / ev.n_windows
        return ev

    def propose(
        self,
        evidence: ResidualEvidence,
        existing: EdgeSet,
        distances_mm: Tensor,
    ) -> list[EditProposal]:
        """Propose edits where the residual persisted beyond the uncertainty gate."""
        if evidence.persistence is None or evidence.persistence.ndim != 2:
            raise ValueError("structural proposals need (N, N) residual persistence")
        pers = evidence.persistence.clone()
        mag = (evidence.residual_sum / max(evidence.n_windows, 1)).clone()
        N = pers.shape[0]
        # never propose an edge that already exists, nor a self-edge
        exists = torch.zeros(N, N, dtype=torch.bool, device=pers.device)
        exists[existing.dst, existing.src] = True
        exists.fill_diagonal_(True)
        pers = pers.masked_fill(exists, 0.0)
        pers = pers.masked_fill(distances_mm > self.max_distance_mm, 0.0)
        cand = (pers >= self.persistence_threshold) & (mag >= self.evidence_threshold)
        dst, src = cand.nonzero(as_tuple=True)
        if dst.numel() == 0:
            return []
        score = mag[dst, src] * pers[dst, src]
        order = torch.argsort(score, descending=True)[: self.max_edits_per_round]
        out: list[EditProposal] = []
        for i in order.tolist():
            out.append(
                EditProposal(
                    kind="add",
                    src=int(src[i]),
                    dst=int(dst[i]),
                    distance_mm=float(distances_mm[dst[i], src[i]]),
                    evidence=float(mag[dst[i], src[i]]),
                    persistence=float(pers[dst[i], src[i]]),
                    reason=(
                        "persistent residual not explained by state/measurement/effective-coupling "
                        "uncertainty"
                    ),
                )
            )
        return out

    # -- evaluation --------------------------------------------------------
    def evaluate(
        self,
        proposals: Sequence[EditProposal],
        *,
        replay: Callable[[EditProposal], float],
        anatomical_logprob: Callable[[EditProposal], float],
        stability: Callable[[EditProposal], float],
    ) -> list[EditDecision]:
        """Evaluate each proposal during replay against all four criteria.

        ``replay(proposal) -> delta_loss`` must run the *prior competencies*,
        not the new data: an edit that improves the new fit while degrading
        previously acquired behaviour is rejected.
        """
        decisions: list[EditDecision] = []
        for p in proposals:
            d_comp = float(replay(p))
            cost = self.energetic_cost_per_mm * p.distance_mm
            plaus = float(anatomical_logprob(p))
            margin = float(stability(p))
            reasons: list[str] = []
            if d_comp > self.competency_tolerance:
                reasons.append(f"degrades prior competencies (delta={d_comp:+.4g})")
            if cost > self.energetic_budget:
                reasons.append(f"energetic cost {cost:.4g} exceeds budget {self.energetic_budget:.4g}")
            if plaus < self.min_anatomical_plausibility:
                reasons.append(f"anatomically implausible (log p={plaus:.4g})")
            if margin < self.min_stability_margin:
                reasons.append(f"destabilises the network (margin={margin:.4g})")
            accepted = not reasons
            if accepted:
                reasons.append("accepted: improves replay, affordable, plausible, stable")
            dec = EditDecision(p, accepted, d_comp, cost, plaus, margin, reasons)
            decisions.append(dec)
            self.history.append(dec)
        return decisions

    # -- application -------------------------------------------------------
    @staticmethod
    def apply(edges: EdgeSet, decisions: Sequence[EditDecision], *, weight: float = 0.05) -> EdgeSet:
        """Return a new EdgeSet with accepted additions as **proposed** edges.

        A learned edge is never promoted to ``hard`` or even ``soft``: it entered
        because a residual persisted, which is evidence about the model, not
        about the anatomy.  It therefore keeps paying the proposed-edge penalty
        and stays inside an explicit model comparison.
        """
        acc = [d for d in decisions if d.accepted and d.proposal.kind == "add"]
        if not acc:
            return edges
        dev = edges.device
        new_src = torch.tensor([d.proposal.src for d in acc], device=dev, dtype=edges.src.dtype)
        new_dst = torch.tensor([d.proposal.dst for d in acc], device=dev, dtype=edges.dst.dtype)
        new_d = torch.tensor([d.proposal.distance_mm for d in acc], device=dev, dtype=edges.distance_mm.dtype)
        new_ev = torch.full((len(acc),), EDGE_CLASS_CODES["proposed"], device=dev, dtype=edges.evidence.dtype)
        w = edges.weight
        if w.ndim == 1:
            new_w = torch.cat([w, torch.full((len(acc),), weight, device=dev, dtype=w.dtype)])
        else:
            new_w = torch.cat([w, torch.full((w.shape[0], len(acc)), weight, device=dev, dtype=w.dtype)], dim=1)
        return EdgeSet(
            torch.cat([edges.src, new_src]),
            torch.cat([edges.dst, new_dst]),
            new_w,
            torch.cat([edges.distance_mm, new_d]),
            torch.cat([edges.evidence, new_ev]),
            edges.n_regions,
            edges.provenance + ("structural_rewrite",),
        )

    def report(self) -> dict[str, Any]:
        return {
            "n_proposed": len(self.history),
            "n_accepted": sum(1 for d in self.history if d.accepted),
            "decisions": [d.as_dict() for d in self.history[-32:]],
        }
