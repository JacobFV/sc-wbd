"""Sensory, cognitive and neurofeedback inputs as first-class interventions.

**SIMULATION ONLY** -- see :mod:`scwbd.intervene.base`.

Thesis Sec. 7.3: *"Images, sounds, language, social interaction, and tasks
enter through perceptual and action models. Their effects depend on
autobiographical memory, speaker and agent models, current affect, body state,
expectation, and context."*

The design consequence is structural, and this module enforces it: a sensory
input is routed through a **declared perceptual port** with its own support,
clock and units.  There is no generic ``stimulus`` vector.  Writing an image
into a "stimulus" slot would reintroduce exactly the modality-blind resampling
the architecture exists to prevent, so :class:`PortRegistry` refuses an
undeclared port name.

A presented sentence is a physical dose (characters delivered to a language
port at a time), not a neural effect -- the same separation the TMS and tFUS
stacks enforce.  :class:`SensoryIntervention` therefore produces a
:class:`~scwbd.intervene.base.PhysicalDose`, and only a named
:class:`PerceptualResponseOperator` turns it into target engagement.

Neurofeedback is a *closed loop*: :class:`NeurofeedbackLoop` couples a decoded
read channel to a presented write channel through a declared contingency, and
reports the neural evidence separately from any generative completion
(Sec. 7.3: language priors "can also conceal how little information came from
the neural signal").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence

import torch
from torch import Tensor

from .base import (
    SIMULATION_ONLY_NOTICE,
    InterventionOperator,
    InterventionRefusal,
    Ledger,
    MechanisticUncertainty,
    PhysicalDose,
    TargetEngagement,
    TissueCoupling,
    WaveformSpec,
    DeviceGeometry,
)

__all__ = [
    "PerceptualPort",
    "PortRegistry",
    "DEFAULT_PORTS",
    "SensoryContent",
    "SensoryIntervention",
    "PerceptualResponseOperator",
    "ContingencySpec",
    "NeurofeedbackLoop",
    "NeurofeedbackReport",
]

_DT = torch.float64


# ---------------------------------------------------------------------------
# ports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerceptualPort:
    """A declared write channel into a perceptual system.

    Mirrors ``scwbd.schema.Port``: a name is not enough; support, clock, units
    and direction are part of the declaration.
    """

    name: str
    modality: Literal["visual", "auditory", "somatosensory", "language", "task", "social"]
    support_kind: str  # "field" | "band" | "event" | "sensor"
    frame: str
    units: str
    dt_s: float
    integration_window_s: float = 0.0
    n_elements: int | None = None
    notice: str = SIMULATION_ONLY_NOTICE


DEFAULT_PORTS: tuple[PerceptualPort, ...] = (
    PerceptualPort("visual.retina", "visual", "field", "retinotopic", "cd/m^2", 1 / 120),
    PerceptualPort("auditory.cochlea", "auditory", "band", "tonotopic", "dB SPL", 1 / 44100),
    PerceptualPort("somatosensory.skin", "somatosensory", "field", "somatotopic", "Pa", 1 / 1000),
    PerceptualPort("language.lexical", "language", "event", "token_stream", "token", 0.25),
    PerceptualPort("task.instruction", "task", "event", "trial_stream", "event", 1.0),
    PerceptualPort("social.agent_model", "social", "event", "agent_stream", "event", 1.0),
)


class PortRegistry:
    """The declared set of perceptual ports. Refuses undeclared names."""

    def __init__(self, ports: Sequence[PerceptualPort] = DEFAULT_PORTS) -> None:
        self._ports = {p.name: p for p in ports}

    def get(self, name: str) -> PerceptualPort:
        try:
            return self._ports[name]
        except KeyError as exc:
            raise InterventionRefusal(
                "R01",
                f"undeclared perceptual port {name!r}. A sensory or cognitive "
                "input is routed through a declared port with its own support, "
                "clock and units; there is no generic 'stimulus' vector "
                "(thesis Sec. 7.3).",
                remedy=f"declare the port, or use one of {sorted(self._ports)}",
                offending_object=name,
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._ports))

    def add(self, port: PerceptualPort) -> "PortRegistry":
        return PortRegistry(tuple(self._ports.values()) + (port,))


# ---------------------------------------------------------------------------
# content and intervention
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensoryContent:
    """What is actually presented, with the context that gives it meaning.

    ``context`` is not decoration.  Thesis Sec. 7.3: the effect of a sentence
    depends on autobiographical memory, speaker and agent models, affect, body
    state and expectation, so a generic prompt-to-outcome table is
    insufficient.  ``context`` carries the declared conditioning variables and
    is required to be non-empty for the language and social ports.
    """

    port: str
    payload: Tensor  # values in the port's declared units
    onset_s: float
    duration_s: float
    description: str = ""
    context: Mapping[str, Any] = field(default_factory=dict)
    affective_valence: float = 0.0  # [-1, 1]
    arousal: float = 0.0  # [0, 1]
    notice: str = SIMULATION_ONLY_NOTICE


@dataclass(frozen=True)
class _PresentationGeometry(DeviceGeometry):
    port_name: str = ""

    def element_positions(self) -> Tensor:
        return torch.zeros(1, 3, dtype=_DT)


class SensoryIntervention(InterventionOperator):
    """A presented input, routed through a declared perceptual port.

    Satisfies the same Sec. 2.4 SDE interface as the TMS and tFUS operators:
    device geometry (the presentation apparatus), waveform (the presentation
    schedule), thermal history (trivially none), tissue coupling (the port
    transfer), and mechanistic uncertainty (which perceptual response operator)
    remain distinct fields.
    """

    def __init__(
        self,
        content: SensoryContent,
        *,
        registry: PortRegistry | None = None,
        write_pattern: Tensor | None = None,
        response_candidates: Sequence[str] = ("linear_port_transfer", "saturating_port_transfer"),
    ) -> None:
        self.registry = registry or PortRegistry()
        self.port = self.registry.get(content.port)
        if self.port.modality in ("language", "social", "task") and not content.context:
            raise InterventionRefusal(
                "R01",
                f"content on port {self.port.name!r} has no declared context. "
                "The effect of a sentence, an instruction or a social input is "
                "not a property of the input alone (thesis Sec. 7.3).",
                remedy="declare speaker/agent model, expectation, task set, "
                "affect and body-state conditioning variables",
                offending_object=content.description,
            )
        self.content = content
        self._pattern = (
            write_pattern.to(_DT).reshape(-1, 1)
            if write_pattern is not None
            else torch.ones(1, 1, dtype=_DT)
        )
        super().__init__(
            name=f"sensory:{self.port.name}",
            geometry=_PresentationGeometry(
                device_id="presentation_apparatus",
                frame=self.port.frame,
                port_name=self.port.name,
            ),
            waveform=WaveformSpec(
                name=f"presentation_{self.port.name}",
                units=self.port.units,
                period=content.duration_s,
                sample_fn=lambda t: torch.ones_like(torch.as_tensor(t, dtype=_DT)),
            ),
            coupling=TissueCoupling(
                name=f"port_transfer_{self.port.modality}",
                parameters={"dt_s": self.port.dt_s},
                mechanistic_status="functional",
                citation="thesis Sec. 7.3 (perceptual and action models)",
            ),
            mechanistic_uncertainty=MechanisticUncertainty(
                candidates=tuple(response_candidates),
                log_weights=torch.zeros(len(response_candidates), dtype=_DT),
                resolved=False,
                note="perceptual transfer is context dependent and unresolved",
            ),
        )

    # -- Sec. 2.4 interface --------------------------------------------------

    def gain(self, x, t, *, anatomy=None, context=None):
        return self._pattern.to(x.dtype)

    def drive(self, t):
        c = self.content
        on = c.onset_s <= t <= c.onset_s + c.duration_s
        amp = float(c.payload.to(_DT).abs().mean()) if c.payload.numel() else 0.0
        return torch.tensor([amp if on else 0.0], dtype=_DT)

    # -- dose ----------------------------------------------------------------

    def dose(self) -> PhysicalDose:
        """The **presented** quantity. A sentence delivered is not a percept."""
        return PhysicalDose(
            modality="sensory",
            quantity=f"{self.port.modality}_presentation",
            units=self.port.units,
            value=self.content.payload.to(_DT),
            support=f"{self.port.name}/{self.port.support_kind}@{self.port.frame}",
            ledger=Ledger(
                variance={},
                bias_status="prior_specified_sensitivity",
                validity_domain={
                    "port": self.port.name,
                    "clock_dt_s": self.port.dt_s,
                    "context_declared": sorted(self.content.context),
                    "note": "presented input, not a percept and not a neural effect",
                },
            ),
        )

    def safety_axes(self) -> dict[str, float]:
        """Map the presentation onto declared ``A_safe`` axes."""
        axes: dict[str, float] = {
            "sensory.affective_valence_abs": abs(self.content.affective_valence)
        }
        if self.port.modality == "auditory":
            axes["sensory.spl_db"] = float(self.content.payload.to(_DT).abs().max())
        if self.port.modality == "visual" and self.content.duration_s > 0:
            axes["sensory.luminance_flash_hz"] = float(
                self.content.context.get("flash_rate_hz", 0.0)
            )
        return axes


# ---------------------------------------------------------------------------
# perceptual response
# ---------------------------------------------------------------------------


@dataclass
class PerceptualResponseOperator:
    """Named candidate mapping a presented input to target engagement."""

    name: str = "linear_port_transfer"
    gain: float = 1.0
    saturation: float | None = None
    context_sensitivity: float = 0.5
    mechanistic_status: str = "functional"
    rationale: str = "declared port transfer with context-dependent gain"
    disabling_evidence: str = (
        "an effect that does not vary with declared expectation/affect context "
        "at matched physical input disables the context term"
    )
    notice: str = SIMULATION_ONLY_NOTICE

    def engage(
        self,
        dose: PhysicalDose,
        *,
        expectation: float = 0.0,
        affect: float = 0.0,
        target: str = "unnamed_target",
    ) -> TargetEngagement:
        if dose.modality != "sensory":
            raise ValueError(f"{self.name} consumes a sensory dose, got {dose.modality}")
        x = dose.value.to(_DT) * self.gain
        x = x * (1.0 + self.context_sensitivity * (expectation + affect))
        if self.saturation is not None:
            x = self.saturation * torch.tanh(x / self.saturation)
        return TargetEngagement(
            target=target,
            response_model=self.name,
            mechanistic_status=self.mechanistic_status,
            units="dimensionless",
            value=x,
            ledger=dose.ledger.merged(
                Ledger(
                    variance={},
                    bias_status="prior_specified_sensitivity",
                    model_discrepancy=1.0,
                    validity_domain={"response_model": self.name},
                )
            ),
        )


# ---------------------------------------------------------------------------
# neurofeedback
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContingencySpec:
    """The declared rule linking a decoded read channel to a presented write.

    Neurofeedback is an intervention, not an observation, precisely because
    this rule exists: the participant's state changes *because* the display
    depends on it.
    """

    read_channel: str
    write_port: str
    transform: Callable[[Tensor], Tensor]
    latency_s: float
    update_rate_hz: float
    reinforcement: Literal["positive", "negative", "bidirectional"] = "positive"
    description: str = ""
    notice: str = SIMULATION_ONLY_NOTICE


@dataclass(frozen=True)
class NeurofeedbackReport:
    """Neural evidence reported **separately** from generative completion.

    Sec. 7.3: language priors can increase decoder fluency while concealing how
    little information came from the neural signal, so the loop reports the
    neural-only contribution and the prior-assisted contribution as two
    numbers, never one accuracy.
    """

    n_updates: int
    neural_information_bits: float
    prior_assisted_information_bits: float
    closed_loop_gain: float
    control_achieved: bool
    ledger: Ledger = field(default_factory=Ledger)
    notice: str = SIMULATION_ONLY_NOTICE

    @property
    def neural_fraction(self) -> float:
        tot = self.neural_information_bits + self.prior_assisted_information_bits
        return self.neural_information_bits / tot if tot > 0 else 0.0


class NeurofeedbackLoop:
    """A closed sensory loop driven by a decoded neural read channel."""

    def __init__(
        self,
        contingency: ContingencySpec,
        *,
        registry: PortRegistry | None = None,
    ) -> None:
        self.registry = registry or PortRegistry()
        self.port = self.registry.get(contingency.write_port)
        self.contingency = contingency

    def run(
        self,
        decoded: Tensor,
        *,
        prior_completion: Tensor | None = None,
        seed: int = 0,
    ) -> tuple[list[SensoryIntervention], NeurofeedbackReport]:
        """Play the loop over a decoded trace, returning the presented inputs."""
        decoded = decoded.to(_DT).reshape(-1)
        dt = 1.0 / self.contingency.update_rate_hz
        interventions: list[SensoryIntervention] = []
        for i, v in enumerate(decoded):
            payload = self.contingency.transform(v.reshape(1))
            interventions.append(
                SensoryIntervention(
                    SensoryContent(
                        port=self.port.name,
                        payload=payload,
                        onset_s=self.contingency.latency_s + i * dt,
                        duration_s=dt,
                        description=f"neurofeedback update {i}",
                        context={
                            "contingency": self.contingency.description,
                            "read_channel": self.contingency.read_channel,
                            "closed_loop": True,
                        },
                    ),
                    registry=self.registry,
                )
            )

        def _bits(x: Tensor) -> float:
            v = float(x.var())
            return 0.5 * math.log2(1.0 + v) if v > 0 else 0.0

        neural = _bits(decoded)
        prior = _bits(prior_completion.to(_DT)) if prior_completion is not None else 0.0
        gain = float(decoded[1:].abs().mean() / decoded[:1].abs().mean().clamp_min(1e-12)) if decoded.numel() > 1 else 1.0
        return interventions, NeurofeedbackReport(
            n_updates=int(decoded.numel()),
            neural_information_bits=neural,
            prior_assisted_information_bits=prior,
            closed_loop_gain=gain,
            control_achieved=gain > 1.0 and neural > 0.0,
            ledger=Ledger(
                variance={"measurement": float(decoded.var())},
                bias_status="design_estimable",
                validity_domain={
                    "note": "neural evidence and prior-assisted completion are "
                    "reported separately (thesis Sec. 7.3)",
                    "closed_loop": True,
                },
            ),
        )
