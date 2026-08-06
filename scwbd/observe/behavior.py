"""Behavioural and report heads: perception -> policy -> motor -> **reporting bias**.

body.tex Sec. 2.4: "Behavior and report are generated through perception,
policy, motor execution, language, and reporting bias."  The four stages are
separate objects here for one reason: a report is an *observation of a report*,
never a read-out of latent state.  body.tex Sec. 8.4 and Sec. 9 make the same
point for conscious access and language; Sec. 13 forbids treating a verbal
report as ground truth about the latent process.

The module supplies:

``PerceptionStage``
    latent evidence -> noisy percept (sensory gain, internal noise, adaptation).
``PolicyStage``
    percept -> drift rate and boundary separation (speed/accuracy, prior bias,
    urgency).
``MotorStage``
    decision -> executed response (non-decision time, motor noise, lapses).
``ReportingBias``
    executed response -> reported response (criterion shift, response bias,
    demand characteristics, metacognitive distortion).  This stage is
    **mandatory**: constructing the operator without it raises R08 unless the
    caller supplies an estimator or an external bound for the bias.
``drift_diffusion_pdf``
    Navarro & Fuss (2009) first-passage density with the small/large-time switch,
    validated in tests against the analytic choice probability and the analytic
    mean decision time.
``psychometric`` / ``chronometric``
    Likelihood-bearing summaries with lapse and guess rates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import torch

from .base import (
    DIMENSIONLESS,
    SECOND,
    UNKNOWN,
    BiasTerm,
    ObservationOperator,
    ObservationRead,
    Prior,
    Provenance,
    RefusalR08,
    Support,
    TemporalSupport,
    UncertaintyLedger,
    Unresolved,
    VarianceDecomposition,
)

__all__ = [
    "PerceptionStage",
    "PolicyStage",
    "MotorStage",
    "ReportingBias",
    "drift_diffusion_pdf",
    "ddm_choice_probability",
    "ddm_mean_decision_time",
    "psychometric",
    "chronometric",
    "BehaviorObservationOperator",
]


# ==========================================================================
# drift-diffusion likelihood
# ==========================================================================


def _f0_large_time(tau: torch.Tensor, w: torch.Tensor, k_max: int = 60) -> torch.Tensor:
    k = torch.arange(1, k_max + 1, dtype=torch.float64).reshape(-1, *([1] * tau.dim()))
    return math.pi * (
        k * torch.exp(-(k**2) * math.pi**2 * tau.unsqueeze(0) / 2.0)
        * torch.sin(k * math.pi * w.unsqueeze(0))
    ).sum(0)


def _f0_small_time(tau: torch.Tensor, w: torch.Tensor, k_max: int = 20) -> torch.Tensor:
    k = torch.arange(-k_max, k_max + 1, dtype=torch.float64).reshape(-1, *([1] * tau.dim()))
    wk = w.unsqueeze(0) + 2.0 * k
    t = tau.unsqueeze(0).clamp_min(1e-12)
    return (1.0 / torch.sqrt(2.0 * math.pi * t**3) * wk * torch.exp(-(wk**2) / (2.0 * t))).sum(0)


def drift_diffusion_pdf(
    rt: torch.Tensor,
    *,
    drift: torch.Tensor | float,
    boundary: torch.Tensor | float,
    start_rel: torch.Tensor | float = 0.5,
    non_decision: torch.Tensor | float = 0.0,
    boundary_hit: Literal["lower", "upper"] = "lower",
) -> torch.Tensor:
    """First-passage density of a Wiener diffusion (Navarro & Fuss 2009).

    Convention: ``dX = drift dt + dW`` with unit diffusion, absorbing boundaries
    at ``0`` and ``boundary``, start at ``start_rel * boundary``.  Returns the
    *defective* density of hitting ``boundary_hit`` at time ``rt``; integrating
    it over ``rt`` gives the choice probability for that boundary.

    Both series expansions are evaluated and the numerically better one is
    selected per element, which is what makes the density usable as a likelihood
    across the whole RT range rather than only near the mode.
    """
    rt = torch.as_tensor(rt, dtype=torch.float64)
    v = torch.as_tensor(drift, dtype=torch.float64).expand_as(rt)
    a = torch.as_tensor(boundary, dtype=torch.float64).expand_as(rt)
    w = torch.as_tensor(start_rel, dtype=torch.float64).expand_as(rt)
    t0 = torch.as_tensor(non_decision, dtype=torch.float64).expand_as(rt)

    if boundary_hit == "upper":
        v, w = -v, 1.0 - w

    t = (rt - t0).clamp_min(0.0)
    tau = t / (a**2)
    valid = t > 0

    small = _f0_small_time(tau, w)
    large = _f0_large_time(tau, w)
    # small-time expansion is accurate for tau below ~0.4; large-time above
    f0 = torch.where(tau < 0.4, small, large)
    dens = (1.0 / a**2) * torch.exp(-v * a * w - (v**2) * t / 2.0) * f0
    return torch.where(valid, dens.clamp_min(0.0), torch.zeros_like(dens))


def ddm_choice_probability(
    *, drift: float, boundary: float, start_rel: float = 0.5,
    boundary_hit: Literal["lower", "upper"] = "lower",
) -> float:
    """Analytic probability of hitting a boundary first (gambler's ruin)."""
    v, a, w = float(drift), float(boundary), float(start_rel)
    z = w * a
    if abs(v) < 1e-9:
        p_upper = z / a
    else:
        p_upper = (1.0 - math.exp(-2.0 * v * z)) / (1.0 - math.exp(-2.0 * v * a))
    return p_upper if boundary_hit == "upper" else 1.0 - p_upper


def ddm_mean_decision_time(*, drift: float, boundary: float, start_rel: float = 0.5) -> float:
    """Analytic mean decision time (both boundaries pooled).

    ``E[T] = (a/v) P_upper - z/v``, which reduces to ``(a/2v) tanh(av/2)`` for an
    unbiased start -- the closed form used by :func:`chronometric`.
    """
    v, a, w = float(drift), float(boundary), float(start_rel)
    z = w * a
    if abs(v) < 1e-9:
        return z * (a - z)
    p_upper = (1.0 - math.exp(-2.0 * v * z)) / (1.0 - math.exp(-2.0 * v * a))
    return (a / v) * p_upper - z / v


# ==========================================================================
# psychometric / chronometric
# ==========================================================================


def psychometric(
    stimulus: torch.Tensor,
    *,
    threshold: float = 0.0,
    slope: float = 1.0,
    lapse: float = 0.02,
    guess: float = 0.0,
    criterion: float = 0.0,
) -> torch.Tensor:
    """``P(respond "1" | stimulus)`` with lapse, guess, **and a criterion shift**.

    ``criterion`` is the reporting-bias handle: it moves the reported decision
    boundary without touching sensitivity, which is exactly the confound that
    makes "the participant reported X" a different quantity from "the latent
    state was X".
    """
    s = torch.as_tensor(stimulus, dtype=torch.float64)
    z = slope * (s - threshold) - criterion
    core = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return guess + (1.0 - guess - lapse) * core


def chronometric(
    stimulus: torch.Tensor,
    *,
    drift_gain: float = 1.0,
    boundary: float = 1.0,
    non_decision: float = 0.3,
) -> torch.Tensor:
    """Mean RT as a function of stimulus strength (unbiased-start DDM)."""
    s = torch.as_tensor(stimulus, dtype=torch.float64)
    v = (drift_gain * s).abs().clamp_min(1e-6)
    return non_decision + (boundary / (2.0 * v)) * torch.tanh(boundary * v / 2.0)


# ==========================================================================
# the four stages
# ==========================================================================


@dataclass(frozen=True)
class PerceptionStage:
    """Latent evidence -> percept.  Gain, internal noise, adaptation."""

    sensory_gain: float = 1.0
    internal_noise_sd: float = 0.3
    adaptation_tau_s: float = 2.0
    adaptation_gain: float = 0.2

    def apply(
        self, evidence: torch.Tensor, dt: float, *, seed: int
    ) -> torch.Tensor:
        e = evidence.to(torch.float64)
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        if self.adaptation_gain > 0 and e.dim() >= 1 and e.shape[-1] > 1:
            alpha = math.exp(-dt / max(self.adaptation_tau_s, 1e-6))
            adapt = torch.zeros_like(e)
            run = torch.zeros(e.shape[:-1], dtype=torch.float64)
            for i in range(e.shape[-1]):
                run = alpha * run + (1 - alpha) * e[..., i]
                adapt[..., i] = run
            e = e - self.adaptation_gain * adapt
        return self.sensory_gain * e + self.internal_noise_sd * torch.randn(
            e.shape, generator=g, dtype=torch.float64
        )

    @staticmethod
    def priors() -> dict[str, Prior]:
        return {
            "sensory_gain": Prior("sensory_gain", "lognormal", (0.0, 0.3)),
            "internal_noise_sd": Prior("internal_noise_sd", "lognormal", (math.log(0.3), 0.4)),
        }


@dataclass(frozen=True)
class PolicyStage:
    """Percept -> decision parameters.  Speed/accuracy, prior bias, urgency."""

    drift_gain: float = 1.0
    boundary: float = 1.0
    start_bias: float = 0.5
    urgency_rate_per_s: float = 0.0

    def drift(self, percept: torch.Tensor) -> torch.Tensor:
        return self.drift_gain * percept.to(torch.float64)

    def boundary_at(self, t: torch.Tensor) -> torch.Tensor:
        return (self.boundary - self.urgency_rate_per_s * t.to(torch.float64)).clamp_min(0.05)

    @staticmethod
    def priors() -> dict[str, Prior]:
        return {
            "drift_gain": Prior("drift_gain", "lognormal", (0.0, 0.4)),
            "boundary": Prior("boundary", "lognormal", (0.0, 0.3), units=DIMENSIONLESS),
            "start_bias": Prior("start_bias", "normal", (0.5, 0.06)),
        }


@dataclass(frozen=True)
class MotorStage:
    """Decision -> executed response.  Non-decision time, motor noise, slips."""

    non_decision_s: float = 0.30
    non_decision_sd_s: float = 0.05
    motor_slip_rate: float = 0.01
    key_asymmetry_s: float = 0.0

    def apply(
        self, choice: torch.Tensor, decision_time: torch.Tensor, *, seed: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        n = choice.numel()
        t0 = self.non_decision_s + self.non_decision_sd_s * torch.randn(
            n, generator=g, dtype=torch.float64
        )
        t0 = t0 + self.key_asymmetry_s * choice.to(torch.float64)
        rt = (decision_time.to(torch.float64) + t0).clamp_min(1e-3)
        slip = torch.rand(n, generator=g, dtype=torch.float64) < self.motor_slip_rate
        executed = torch.where(slip, 1.0 - choice.to(torch.float64), choice.to(torch.float64))
        return executed, rt

    @staticmethod
    def priors() -> dict[str, Prior]:
        return {
            "non_decision_s": Prior(
                "non_decision_s", "lognormal", (math.log(0.30), 0.25), units=SECOND
            ),
            "motor_slip_rate": Prior("motor_slip_rate", "uniform", (0.0, 0.05)),
        }


@dataclass(frozen=True)
class ReportingBias:
    """Executed response -> **reported** response.  Never optional.

    ``estimator`` and ``external_bound`` mirror the R08 contract: a reporting
    bias may carry a point value only when the design identifies it (e.g. a
    payoff-matrix manipulation, a confidence-rating calibration block, or a
    forced-report control condition) or when an external instrument bounds it.
    """

    response_bias: float = 0.0
    """Additive shift of the reported criterion (positive favours response 1)."""
    demand_characteristic: float = 0.0
    """Pull of the reported response toward the experimenter-expected answer."""
    metacognitive_noise_sd: float = 0.2
    """Noise added between the decision variable and the reported confidence."""
    report_lapse: float = 0.01
    expected_response: float = 1.0
    estimator: str | None = None
    external_bound: str | None = None
    sensitivity_range: tuple[float, float] = (-0.3, 0.3)

    def apply(
        self,
        executed: torch.Tensor,
        decision_variable: torch.Tensor,
        *,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(reported_choice, reported_confidence)``."""
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        ex = executed.to(torch.float64)
        n = ex.numel()

        # the evidence the participant has, *signed by what they actually did*
        signed = (2.0 * ex - 1.0) * decision_variable.to(torch.float64).abs()
        # the report criterion is shifted, not the evidence: this is the
        # signal-detection form of a response bias, and it is directional --
        # a symmetric flip would leave the response proportion unchanged and
        # would therefore not be a bias at all.
        shifted = (
            signed
            + self.response_bias
            + self.demand_characteristic * (2.0 * self.expected_response - 1.0)
            + self.metacognitive_noise_sd
            * torch.randn(n, generator=g, dtype=torch.float64)
        )
        rep = (shifted > 0.0).to(torch.float64)

        lapse = torch.rand(n, generator=g, dtype=torch.float64) < self.report_lapse
        rep = torch.where(
            lapse,
            (torch.rand(n, generator=g, dtype=torch.float64) < 0.5).to(torch.float64),
            rep,
        )
        conf = torch.sigmoid(shifted.abs())
        return rep, conf

    def bias_term(self) -> BiasTerm:
        """Refuses (R08) when a non-zero bias is asserted with no backing."""
        if self.estimator:
            return BiasTerm(
                name="reporting_bias",
                interval=(self.response_bias, self.response_bias),
                status="design_estimable",
                units=DIMENSIONLESS,
                estimator=self.estimator,
                note="report is not latent state; this term is the measured gap",
            )
        if self.external_bound:
            return BiasTerm(
                name="reporting_bias",
                interval=self.sensitivity_range,
                status="externally_bounded",
                units=DIMENSIONLESS,
                external_bound=self.external_bound,
            )
        lo, hi = self.sensitivity_range
        if hi <= lo:
            raise RefusalR08(
                "reporting bias has neither an estimator nor an external bound "
                "and no non-degenerate sensitivity range to sweep",
                offending_object=self,
            )
        return BiasTerm(
            name="reporting_bias",
            interval=(lo, hi),
            status="prior_specified_sensitivity",
            units=DIMENSIONLESS,
            sensitivity_grid=tuple(
                float(x) for x in torch.linspace(lo, hi, 5, dtype=torch.float64)
            ),
            note="no payoff manipulation, confidence-calibration block, or "
            "forced-report control in this design: the report-to-state gap is "
            "swept, not estimated",
        )


# ==========================================================================
# the operator
# ==========================================================================


class BehaviorObservationOperator(ObservationOperator):
    """``O_behavior``: latent evidence -> reported choice and RT, with the gap named."""

    name = "behavior_observation_operator"
    version = "0.1.0"

    def __init__(
        self,
        *,
        perception: PerceptionStage | None = None,
        policy: PolicyStage | None = None,
        motor: MotorStage | None = None,
        reporting: ReportingBias | None = None,
        trial_isi_s: float = 2.0,
        clock: str = "behavior_trial",
        frame: str = "task_response_frame",
        response_labels: Sequence[str] = ("left", "right"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if reporting is None:
            raise RefusalR08(
                "BehaviorObservationOperator constructed without a ReportingBias "
                "stage: that silently asserts report == latent state with a point "
                "bias of exactly zero and no estimator",
                offending_object="reporting=None",
            )
        self.perception = perception or PerceptionStage()
        self.policy = policy or PolicyStage()
        self.motor = motor or MotorStage()
        self.reporting = reporting
        self.response_labels = tuple(response_labels)
        self.frame = frame
        self.dtype = dtype
        self._temporal = TemporalSupport(
            clock=clock,
            dt=float(trial_isi_s),
            integration_window=float(trial_isi_s),
            group_delay=0.0,
            jitter_sd=0.2 * float(trial_isi_s),
        )

    # -- descriptors --------------------------------------------------------
    @property
    def support(self) -> Support:
        return Support(
            kind="trial",
            frame=self.frame,
            units=DIMENSIONLESS,
            psf=None,
            n_elements=None,
            labels=("reported_choice", "reaction_time_s", "reported_confidence"),
        )

    @property
    def temporal(self) -> TemporalSupport:
        return self._temporal

    @property
    def units(self) -> str:
        return DIMENSIONLESS

    @property
    def nuisance_priors(self) -> dict[str, Prior]:
        p: dict[str, Prior] = {}
        p.update(PerceptionStage.priors())
        p.update(PolicyStage.priors())
        p.update(MotorStage.priors())
        p["reporting_response_bias"] = Prior(
            "reporting_response_bias",
            "normal",
            (0.0, 0.15),
            source="signal-detection criterion variability across sessions and "
            "instructions; not identified without a payoff manipulation",
        )
        return p

    # -- likelihood ---------------------------------------------------------
    def log_likelihood(
        self,
        choice: torch.Tensor,
        rt: torch.Tensor,
        evidence: torch.Tensor,
        *,
        include_reporting_bias: bool = True,
    ) -> torch.Tensor:
        """Per-trial log density of the observed (choice, RT) pair.

        The reporting bias enters as a shift of the *start point*, which is how
        a response bias actually manifests in a diffusion model.  Setting
        ``include_reporting_bias=False`` gives the likelihood a naive analysis
        would use, and the difference between the two is the model-class
        variance the ledger records.
        """
        v = self.policy.drift(evidence)
        a = self.policy.boundary
        w = self.policy.start_bias
        if include_reporting_bias:
            w = float(
                min(max(w + 0.25 * self.reporting.response_bias, 0.02), 0.98)
            )
        upper = choice.to(torch.float64) > 0.5
        dens_u = drift_diffusion_pdf(
            rt, drift=v, boundary=a, start_rel=w,
            non_decision=self.motor.non_decision_s, boundary_hit="upper",
        )
        dens_l = drift_diffusion_pdf(
            rt, drift=v, boundary=a, start_rel=w,
            non_decision=self.motor.non_decision_s, boundary_hit="lower",
        )
        dens = torch.where(upper, dens_u, dens_l)
        lam = self.motor.motor_slip_rate + self.reporting.report_lapse
        dens = (1.0 - lam) * dens + lam * 0.5 / max(float(rt.max()), 1e-3)
        return torch.log(dens.clamp_min(1e-300))

    # -- the read -----------------------------------------------------------
    def observe(
        self,
        evidence: torch.Tensor,
        latent_temporal: TemporalSupport | None = None,
        *,
        seed: int,
        n_trials: int | None = None,
        max_rt_s: float = 5.0,
        sim_dt: float = 1e-3,
    ) -> ObservationRead | Unresolved:
        """Simulate the full chain for a sequence of per-trial evidence values.

        ``evidence`` is ``(n_trials,)`` stimulus strength, or ``(n_trials, n_t)``
        time-resolved evidence on ``latent_temporal`` (then averaged over the
        decision window by the perception stage).
        """
        e = evidence.to(torch.float64)
        dt_lat = latent_temporal.dt if latent_temporal is not None else sim_dt
        percept_full = self.perception.apply(e, dt_lat, seed=seed + 1)
        percept = percept_full.mean(-1) if percept_full.dim() > 1 else percept_full
        n = int(percept.numel()) if n_trials is None else min(n_trials, int(percept.numel()))
        percept = percept[:n]

        v = self.policy.drift(percept)
        a = self.policy.boundary
        z0 = self.policy.start_bias * a

        g = torch.Generator(device="cpu").manual_seed(int(seed) + 2)
        n_steps = int(max_rt_s / sim_dt)
        x = torch.full((n,), float(z0), dtype=torch.float64)
        done = torch.zeros(n, dtype=torch.bool)
        choice = torch.zeros(n, dtype=torch.float64)
        dtime = torch.full((n,), float(max_rt_s), dtype=torch.float64)
        sq = math.sqrt(sim_dt)
        for k in range(n_steps):
            t = (k + 1) * sim_dt
            bound = float(self.policy.boundary_at(torch.tensor(t)))
            x = torch.where(
                done, x, x + v * sim_dt + sq * torch.randn(n, generator=g, dtype=torch.float64)
            )
            hit_up = (~done) & (x >= bound)
            hit_lo = (~done) & (x <= 0.0)
            choice = torch.where(hit_up, torch.ones_like(choice), choice)
            dtime = torch.where(hit_up | hit_lo, torch.full_like(dtime, t), dtime)
            done = done | hit_up | hit_lo
            if bool(done.all()):
                break

        executed, rt = self.motor.apply(choice, dtime, seed=seed + 3)
        decision_variable = v * dtime
        reported, confidence = self.reporting.apply(
            executed, decision_variable, seed=seed + 4
        )

        pred = torch.stack([reported, rt, confidence])
        components = {
            "percept": percept.to(self.dtype),
            "drift": v.to(self.dtype),
            "decision_time": dtime.to(self.dtype),
            "executed_choice": executed.to(self.dtype),
            "reported_choice": reported.to(self.dtype),
            "report_flipped": (reported != executed).to(self.dtype),
        }

        flip_rate = float((reported != executed).to(torch.float64).mean())
        ledger = self._ledger(
            seed=seed, n=n, flip_rate=flip_rate, rt=rt, choice=reported
        )
        return ObservationRead(
            prediction=pred.to(self.dtype),
            units=DIMENSIONLESS,
            support=self.support,
            temporal=self._temporal,
            ledger=ledger,
            components=components,
            residual_channels={
                "executed_minus_reported": (executed - reported),
            },
        )

    # -- ledger -------------------------------------------------------------
    def _ledger(
        self, *, seed: int, n: int, flip_rate: float, rt: torch.Tensor, choice: torch.Tensor
    ) -> UncertaintyLedger:
        p = float(choice.mean())
        bias = (
            self.reporting.bias_term(),
            BiasTerm(
                name="motor_execution_slip",
                interval=(-self.motor.motor_slip_rate, self.motor.motor_slip_rate),
                status="design_estimable",
                units=DIMENSIONLESS,
                estimator="catch trials with unambiguous stimuli measure the "
                "slip rate directly",
            ),
            BiasTerm(
                name="non_decision_time_attribution",
                interval=(-0.05, 0.05),
                status="externally_bounded",
                units=SECOND,
                external_bound="EMG onset or simple-RT baseline blocks bound the "
                "motor component of t0; without them, t0 absorbs any latency the "
                "decision model cannot fit",
            ),
            BiasTerm(
                name="task_instruction_and_strategy_drift",
                interval=(-0.15, 0.15),
                status="prior_specified_sensitivity",
                units=DIMENSIONLESS,
                sensitivity_grid=(-0.15, -0.075, 0.0, 0.075, 0.15),
                note="participants change speed/accuracy policy within a session; "
                "unmodelled drift is absorbed by boundary and drift estimates",
            ),
        )
        return UncertaintyLedger(
            variance=VarianceDecomposition(
                measurement=float(p * (1 - p) / max(n, 1)),
                within_session=float(rt.var(unbiased=True)) if n > 1 else UNKNOWN,
                between_session=UNKNOWN,
                parameter_posterior=UNKNOWN,
                model_class=UNKNOWN,
                numerical=0.0,
                units="dimensionless",
            ),
            bias=bias,
            model_discrepancy=UNKNOWN,
            model_discrepancy_flag=True,
            validity_domain={
                "units": "choice in {0,1}; reaction time in s; confidence in [0,1]",
                "response_labels": self.response_labels,
                "clock": self._temporal.clock,
                "n_trials": n,
                "report_flip_rate": flip_rate,
                "claim_boundary": "this read is a model of the participant's "
                "REPORT. It is not a measurement of the latent decision variable, "
                "of awareness, or of any conscious state (body.tex Sec. 8.4).",
            },
            provenance=Provenance(
                operator=self.name,
                version=self.version,
                frames=(self.frame,),
                clocks=(self._temporal.clock,),
                inputs=("latent_evidence",),
                references=(
                    "body.tex Sec. 2.4 (perception, policy, motor, language, "
                    "reporting bias)",
                    "Ratcliff & McKoon 2008; Navarro & Fuss 2009",
                ),
                seed=seed,
            ),
            notes=(
                "reported_choice and executed_choice are returned separately so "
                "that the report-to-execution gap is measurable rather than "
                "assumed zero.",
            ),
        )
