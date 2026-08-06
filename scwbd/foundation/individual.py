"""Stage V individualization without catastrophic identity drift (body.tex §6.5).

.. math::  \\theta_{p,s} = \\mu + \\alpha_{g(p)} + \\delta_p + \\zeta_{p,s}

with **centered / sum-to-zero population effects** :math:`\\sum_g n_g\\alpha_g=0`,
hierarchical shrinkage :math:`\\delta_p\\sim\\mathcal N(0,\\Sigma_{\\rm person})`
and :math:`\\zeta_{p,s}\\sim\\mathcal N(0,\\Sigma_{\\rm session})`.

The centering is not decoration.  Refusal **R07** ("population/subject/session
effects without centering or shrinkage") is enforced *by construction*: the
group effects are stored in an unconstrained tensor and the weighted mean is
projected out on **every** read, so no optimiser trajectory and no checkpoint
can leave the constraint violated.  :meth:`Individualizer.assert_centered` is
the executable check, and :func:`recovery_test` verifies on simulated data that
the decomposition is actually identified rather than merely written down.

A session effect must not become a permanent trait: sleep, fatigue, stress,
medication, electrode impedance and head pose live in ``zeta``, and
:meth:`Individualizer.consolidate` refuses to move variance from ``zeta`` into
``delta`` unless it has been observed across sessions and passes an
uncertainty-triggered deferral test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

__all__ = ["Individualizer", "R07Violation", "recovery_test"]


class R07Violation(RuntimeError):
    """Population/subject/session effects without centering or shrinkage."""

    code = "R07"


class Individualizer(nn.Module):
    """Hierarchical decomposition of the individualized parameter vector.

    Non-centered parameterisation (``delta_p = L_person z_p``) is used because it
    conditions the optimisation better, **and** the identifying constraint is
    kept anyway: a non-centered parameterisation improves inference, it does not
    remove the need for identifiability (§6.5).
    """

    def __init__(
        self,
        theta_dim: int,
        *,
        n_groups: int = 1,
        n_participants: int = 0,
        n_sessions: int = 0,
        group_counts: Sequence[float] | None = None,
        person_sd_init: float = 0.15,
        session_sd_init: float = 0.08,
    ) -> None:
        super().__init__()
        self.theta_dim = int(theta_dim)
        self.n_groups = int(n_groups)
        self.mu = nn.Parameter(torch.zeros(theta_dim))
        self._alpha_raw = nn.Parameter(torch.zeros(max(n_groups, 1), theta_dim))
        counts = torch.ones(max(n_groups, 1)) if group_counts is None else torch.as_tensor(group_counts, dtype=torch.float32)
        self.register_buffer("group_counts", counts / counts.sum().clamp_min(1e-8))
        # non-centered person / session effects
        self.z_person = nn.Parameter(torch.zeros(max(n_participants, 1), theta_dim))
        self.z_session = nn.Parameter(torch.zeros(max(n_sessions, 1), theta_dim))
        self.log_sd_person = nn.Parameter(torch.full((theta_dim,), math.log(person_sd_init)))
        self.log_sd_session = nn.Parameter(torch.full((theta_dim,), math.log(session_sd_init)))
        self.register_buffer("_person_seen_sessions", torch.zeros(max(n_participants, 1)))

    # -- centered effects -------------------------------------------------
    @property
    def alpha(self) -> Tensor:
        """Sum-to-zero group effects: ``sum_g n_g alpha_g = 0`` **by construction**."""
        w = self.group_counts.reshape(-1, 1).to(self._alpha_raw.dtype)
        return self._alpha_raw - (w * self._alpha_raw).sum(0, keepdim=True)

    @property
    def delta(self) -> Tensor:
        """Person effects, shrunk: ``delta_p = sd_person * z_p``, ``z ~ N(0,I)``."""
        return self.z_person * self.log_sd_person.exp()

    @property
    def zeta(self) -> Tensor:
        return self.z_session * self.log_sd_session.exp()

    # -- assembly ---------------------------------------------------------
    def forward(
        self,
        group: Tensor | None = None,
        participant: Tensor | None = None,
        session: Tensor | None = None,
        *,
        base: Tensor | None = None,
    ) -> Tensor:
        """``theta_{p,s}`` for a batch of (group, participant, session) indices."""
        n = 1
        for t in (group, participant, session):
            if t is not None:
                n = max(n, int(t.reshape(-1).shape[0]))
        if base is not None:
            base = base.reshape(-1, self.theta_dim)
            n = max(n, base.shape[0])
        out = self.mu.reshape(1, -1).expand(n, -1)
        if base is not None:
            out = out + (base if base.shape[0] == n else base.expand(n, -1))
        if group is not None:
            out = out + self.alpha.index_select(0, group.clamp(0, self.alpha.shape[0] - 1))
        if participant is not None:
            out = out + self.delta.index_select(0, participant.clamp(0, self.delta.shape[0] - 1))
        if session is not None:
            out = out + self.zeta.index_select(0, session.clamp(0, self.zeta.shape[0] - 1))
        return out

    # -- shrinkage penalty ------------------------------------------------
    def prior_penalty(self, *, participant: Tensor | None = None, session: Tensor | None = None) -> Tensor:
        """Negative log prior of the random effects (the shrinkage R07 demands)."""
        zp = self.z_person if participant is None else self.z_person.index_select(0, participant)
        zs = self.z_session if session is None else self.z_session.index_select(0, session)
        # z ~ N(0, I) plus half-normal priors keeping the scales from running away
        pen = 0.5 * (zp**2).sum(-1).mean() + 0.5 * (zs**2).sum(-1).mean()
        pen = pen + 0.5 * ((self.log_sd_person.exp() / 0.3) ** 2).sum() + 0.5 * ((self.log_sd_session.exp() / 0.2) ** 2).sum()
        # sessions must not out-vary persons: identity is the slower quantity
        pen = pen + torch.relu(self.log_sd_session - self.log_sd_person).pow(2).sum()
        return pen

    # -- R07 enforcement --------------------------------------------------
    def assert_centered(self, tol: float = 1e-5) -> None:
        with torch.no_grad():
            resid = (self.group_counts.reshape(-1, 1) * self.alpha).sum(0).abs().max()
        if float(resid) > tol:
            raise R07Violation(
                f"[R07] weighted group effects do not sum to zero (max |sum_g n_g alpha_g| = {float(resid):.3g} > {tol}); "
                "population/subject/session effects require centering or shrinkage."
            )
        if not torch.isfinite(self.log_sd_person).all() or not torch.isfinite(self.log_sd_session).all():
            raise R07Violation("[R07] non-finite hierarchical scales: shrinkage is not defined")

    # -- continual learning ----------------------------------------------
    def observe_session(self, participant: Tensor) -> None:
        idx = participant.reshape(-1).clamp(0, self._person_seen_sessions.shape[0] - 1)
        self._person_seen_sessions.index_add_(0, idx, torch.ones_like(idx, dtype=self._person_seen_sessions.dtype))

    @torch.no_grad()
    def consolidate(
        self,
        participant: int,
        *,
        min_sessions: int = 3,
        max_uncertainty: float = 0.5,
        rate: float = 0.25,
    ) -> dict[str, Any]:
        """Move stable session variance into the person effect -- or defer.

        "Sleep, fatigue, stress, medication, electrode impedance and head pose
        should not be consolidated as permanent traits" (§6.5).  Consolidation is
        therefore gated on repeated observation and on the person effect's own
        uncertainty; when the gate fails the method **defers** and says so.
        """
        seen = float(self._person_seen_sessions[participant])
        unc = float(self.log_sd_person.exp().mean())
        if seen < min_sessions:
            return {"action": "defer", "reason": f"only {seen:.0f} sessions observed (< {min_sessions})"}
        if unc > max_uncertainty:
            return {"action": "defer", "reason": f"person-effect uncertainty {unc:.3f} > {max_uncertainty}"}
        mean_z = self.z_session.mean(0)
        self.z_person[participant] += rate * mean_z * (self.log_sd_session.exp() / self.log_sd_person.exp().clamp_min(1e-6))
        return {"action": "consolidate", "sessions": seen, "rate": rate}

    # -- reporting --------------------------------------------------------
    def variance_decomposition(self) -> dict[str, Any]:
        a = self.alpha.detach()
        d = self.delta.detach()
        z = self.zeta.detach()
        v = {
            "group_var": a.var(0).mean().item() if a.shape[0] > 1 else 0.0,
            "person_var": d.var(0).mean().item() if d.shape[0] > 1 else 0.0,
            "session_var": z.var(0).mean().item() if z.shape[0] > 1 else 0.0,
            "sd_person": self.log_sd_person.exp().detach().cpu().tolist(),
            "sd_session": self.log_sd_session.exp().detach().cpu().tolist(),
            "centering_residual": float((self.group_counts.reshape(-1, 1) * a).sum(0).abs().max()),
        }
        tot = v["group_var"] + v["person_var"] + v["session_var"]
        v["fraction"] = {
            k: (v[k + "_var"] / tot if tot > 0 else 0.0) for k in ("group", "person", "session")
        }
        return v


# ======================================================================
# identifiability
# ======================================================================
def recovery_test(
    *,
    theta_dim: int = 4,
    n_groups: int = 3,
    n_participants: int = 60,
    n_sessions_per_participant: int = 4,
    noise_sd: float = 0.05,
    steps: int = 900,
    lr: float = 0.05,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Simulated recovery: is ``mu + alpha + delta + zeta`` actually identified?

    Generates data from a known decomposition, refits it through
    :class:`Individualizer`, and reports the correlation between true and
    recovered components.  A decomposition that is written down but not
    identified will show high correlation on the *sum* and near-zero correlation
    on the *parts* -- which is exactly the failure R07 is meant to catch, so the
    test reports both.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device(device)
    n_sess = n_participants * n_sessions_per_participant

    group_of = torch.randint(0, n_groups, (n_participants,), generator=g)
    counts = torch.bincount(group_of, minlength=n_groups).float()
    mu_t = torch.randn(theta_dim, generator=g) * 0.5
    a_raw = torch.randn(n_groups, theta_dim, generator=g) * 0.3
    alpha_t = a_raw - ((counts / counts.sum()).reshape(-1, 1) * a_raw).sum(0, keepdim=True)
    delta_t = torch.randn(n_participants, theta_dim, generator=g) * 0.20
    zeta_t = torch.randn(n_sess, theta_dim, generator=g) * 0.08

    sess_participant = torch.arange(n_sess) // n_sessions_per_participant
    sess_group = group_of[sess_participant]
    y = (
        mu_t
        + alpha_t[sess_group]
        + delta_t[sess_participant]
        + zeta_t
        + torch.randn(n_sess, theta_dim, generator=g) * noise_sd
    ).to(dev)

    m = Individualizer(
        theta_dim,
        n_groups=n_groups,
        n_participants=n_participants,
        n_sessions=n_sess,
        group_counts=counts.tolist(),
    ).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    gi, pi, si = sess_group.to(dev), sess_participant.to(dev), torch.arange(n_sess, device=dev)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = m(gi, pi, si)
        loss = ((pred - y) ** 2).mean() / (2 * noise_sd**2) + m.prior_penalty() / n_sess
        loss.backward()
        opt.step()
    m.assert_centered(tol=1e-4)

    def corr(a: Tensor, b: Tensor) -> float:
        a = a.reshape(-1).double() - a.reshape(-1).double().mean()
        b = b.reshape(-1).double() - b.reshape(-1).double().mean()
        d = (a.norm() * b.norm()).clamp_min(1e-12)
        return float((a @ b) / d)

    with torch.no_grad():
        rec = {
            "mu_error": float((m.mu.cpu() - mu_t).abs().mean()),
            "alpha_corr": corr(m.alpha.cpu(), alpha_t),
            "delta_corr": corr(m.delta.cpu(), delta_t),
            "zeta_corr": corr(m.zeta.cpu(), zeta_t),
            "sum_corr": corr(m(gi, pi, si).cpu(), y.cpu()),
            "centering_residual": float((m.group_counts.cpu().reshape(-1, 1) * m.alpha.cpu()).sum(0).abs().max()),
            "variance_decomposition": m.variance_decomposition(),
            "n_participants": n_participants,
            "n_sessions": n_sess,
            "noise_sd": noise_sd,
        }
    rec["identified"] = bool(
        rec["alpha_corr"] > 0.8 and rec["delta_corr"] > 0.8 and rec["zeta_corr"] > 0.5 and rec["centering_residual"] < 1e-4
    )
    rec["interpretation"] = (
        "A high sum_corr with low delta_corr/zeta_corr would mean the decomposition fits the data "
        "without being identified; both are reported so that cannot be hidden."
    )
    return rec
