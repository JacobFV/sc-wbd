"""Sensory / cognitive / neurofeedback interventions. SIMULATION ONLY.

Thesis Sec. 7.3: inputs enter through perceptual and action models, not a
generic stimulus vector, and neural evidence is reported separately from
language-model completion.
"""

from __future__ import annotations


import pytest
import torch

from scwbd.intervene.base import (
    ExposureWindow,
    InterventionOperator,
    InterventionRefusal,
    PhysicalDose,
    TargetEngagement,
)
from scwbd.intervene.safety import FeasibleSet, ProposedIntervention
from scwbd.intervene.sensory import (
    DEFAULT_PORTS,
    ContingencySpec,
    NeurofeedbackLoop,
    PerceptualPort,
    PerceptualResponseOperator,
    PortRegistry,
    SensoryContent,
    SensoryIntervention,
)

_DT = torch.float64


def _visual(**kw) -> SensoryContent:
    d = dict(
        port="visual.retina",
        payload=torch.full((16,), 30.0, dtype=_DT),
        onset_s=0.0,
        duration_s=0.5,
        description="simulated grating",
        context={"flash_rate_hz": 1.0, "expectation": "neutral"},
    )
    d.update(kw)
    return SensoryContent(**d)


# ---------------------------------------------------------------------------
# ports
# ---------------------------------------------------------------------------


def test_there_is_no_generic_stimulus_port():
    reg = PortRegistry()
    with pytest.raises(InterventionRefusal) as e:
        reg.get("stimulus")
    assert e.value.code == "R01"
    assert "generic 'stimulus' vector" in str(e.value)


def test_declared_ports_carry_support_clock_and_units():
    for p in DEFAULT_PORTS:
        assert p.units and p.frame and p.support_kind
        assert p.dt_s > 0.0


def test_a_new_port_can_be_declared_and_then_used():
    reg = PortRegistry().add(
        PerceptualPort("vestibular.canal", "somatosensory", "field", "head", "rad/s^2", 0.001)
    )
    assert "vestibular.canal" in reg.names()
    assert reg.get("vestibular.canal").units == "rad/s^2"


# ---------------------------------------------------------------------------
# sensory intervention
# ---------------------------------------------------------------------------


def test_a_sensory_intervention_is_an_intervention_operator():
    si = SensoryIntervention(_visual())
    assert isinstance(si, InterventionOperator)
    d = si.describe()
    assert d["coupling"] == "port_transfer_visual"
    assert d["mechanism_resolved"] is False


def test_a_presented_input_is_a_physical_dose_not_a_percept():
    dose = SensoryIntervention(_visual()).dose()
    assert isinstance(dose, PhysicalDose)
    assert dose.modality == "sensory"
    assert dose.units == "cd/m^2"
    assert "not a neural effect" in dose.ledger.validity_domain["note"]
    with pytest.raises(InterventionRefusal):
        dose.as_neural_effect()


def test_language_and_social_input_without_declared_context_is_refused():
    for port in ("language.lexical", "social.agent_model", "task.instruction"):
        c = SensoryContent(
            port=port,
            payload=torch.ones(4, dtype=_DT),
            onset_s=0.0,
            duration_s=1.0,
            description="a sentence",
            context={},
        )
        with pytest.raises(InterventionRefusal, match="no declared context"):
            SensoryIntervention(c)


def test_language_input_with_context_is_accepted():
    c = SensoryContent(
        port="language.lexical",
        payload=torch.ones(8, dtype=_DT),
        onset_s=0.0,
        duration_s=2.0,
        description="a simulated sentence",
        context={
            "speaker_model": "unfamiliar",
            "expectation": "neutral",
            "affect": 0.0,
            "body_state": "rested",
        },
    )
    si = SensoryIntervention(c)
    assert sorted(si.dose().ledger.validity_domain["context_declared"])[0] == "affect"


def test_the_drive_is_gated_by_the_presentation_window():
    si = SensoryIntervention(_visual(onset_s=1.0, duration_s=0.5))
    assert float(si.drive(0.5)) == 0.0
    assert float(si.drive(1.2)) > 0.0
    assert float(si.drive(2.0)) == 0.0


def test_a_sensory_intervention_integrates_through_the_same_sde():
    si = SensoryIntervention(
        _visual(onset_s=0.0, duration_s=0.2),
        write_pattern=torch.tensor([1.0, 0.3, 0.0], dtype=_DT),
    )
    A = -torch.diag(torch.tensor([5.0, 8.0, 12.0], dtype=_DT))
    res = si.integrate(
        torch.zeros(3, dtype=_DT), lambda x, t: A @ x, ExposureWindow(0.0, 0.4), dt=1e-3
    )
    assert float(res.final_state[0]) > 0.0
    assert float(res.final_state[2]) == 0.0  # the pattern writes nowhere there


def test_sensory_exposure_maps_onto_declared_safety_axes_and_can_be_refused():
    fs = FeasibleSet(require_pose_certification=False)
    loud = SensoryIntervention(
        SensoryContent(
            port="auditory.cochlea",
            payload=torch.full((8,), 110.0, dtype=_DT),  # dB SPL
            onset_s=0.0,
            duration_s=1.0,
            description="simulated tone",
            context={},
        )
    )
    p = ProposedIntervention(
        label="loud_tone", modality="sensory", exposure=loud.safety_axes(),
        pose_certified=True,
    )
    v = fs.contains(p)
    assert not v.feasible
    assert any("spl_db" in str(x) for x in v.violations)


def test_a_flashing_visual_input_in_the_photosensitive_band_is_refused():
    fs = FeasibleSet(require_pose_certification=False)
    si = SensoryIntervention(_visual(context={"flash_rate_hz": 12.0}))
    p = ProposedIntervention(
        label="flicker", modality="sensory", exposure=si.safety_axes(),
        pose_certified=True,
    )
    assert not fs.contains(p).feasible


def test_the_optimizer_cannot_search_over_maximally_distressing_content():
    fs = FeasibleSet(require_pose_certification=False)
    si = SensoryIntervention(
        SensoryContent(
            port="language.lexical",
            payload=torch.ones(4, dtype=_DT),
            onset_s=0.0,
            duration_s=1.0,
            description="simulated sentence",
            context={"speaker_model": "unknown"},
            affective_valence=-0.98,
        )
    )
    p = ProposedIntervention(
        label="distressing", modality="sensory", exposure=si.safety_axes(),
        pose_certified=True,
    )
    assert not fs.contains(p).feasible


# ---------------------------------------------------------------------------
# perceptual response
# ---------------------------------------------------------------------------


def test_the_perceptual_operator_produces_target_engagement_and_is_context_sensitive():
    dose = SensoryIntervention(_visual()).dose()
    op = PerceptualResponseOperator(gain=0.01, saturation=1.0)
    neutral = op.engage(dose, expectation=0.0, affect=0.0)
    primed = op.engage(dose, expectation=0.8, affect=0.3)
    assert isinstance(neutral, TargetEngagement)
    assert not isinstance(neutral, PhysicalDose)
    assert float(primed.value.max()) > float(neutral.value.max())
    assert neutral.ledger.model_discrepancy is not None


def test_a_perceptual_operator_refuses_a_non_sensory_dose():
    d = PhysicalDose(
        modality="tms", quantity="E_field", units="V/m",
        value=torch.ones(3, 3, dtype=_DT), support="x",
    )
    with pytest.raises(ValueError, match="sensory dose"):
        PerceptualResponseOperator().engage(d)


# ---------------------------------------------------------------------------
# neurofeedback
# ---------------------------------------------------------------------------


def _loop() -> NeurofeedbackLoop:
    return NeurofeedbackLoop(
        ContingencySpec(
            read_channel="eeg.alpha_power",
            write_port="visual.retina",
            transform=lambda v: 40.0 * torch.sigmoid(v),
            latency_s=0.15,
            update_rate_hz=4.0,
            description="alpha up-regulation, simulated",
        )
    )


def test_neurofeedback_produces_one_declared_presentation_per_update():
    decoded = torch.linspace(-1.0, 2.0, 12, dtype=_DT)
    ivs, rep = _loop().run(decoded)
    assert len(ivs) == 12
    assert rep.n_updates == 12
    assert all(i.port.name == "visual.retina" for i in ivs)
    assert all(i.content.context["closed_loop"] is True for i in ivs)
    # onsets respect the declared latency and update rate
    assert ivs[0].content.onset_s == pytest.approx(0.15)
    assert ivs[1].content.onset_s - ivs[0].content.onset_s == pytest.approx(0.25)


def test_neural_evidence_is_reported_separately_from_prior_assisted_completion():
    decoded = 0.1 * torch.randn(64, generator=torch.Generator().manual_seed(2), dtype=_DT)
    prior = 5.0 * torch.randn(64, generator=torch.Generator().manual_seed(3), dtype=_DT)
    _, rep = _loop().run(decoded, prior_completion=prior)
    assert rep.neural_information_bits > 0.0
    assert rep.prior_assisted_information_bits > rep.neural_information_bits
    # the honest headline: most of the apparent information is not neural
    assert rep.neural_fraction < 0.2
    assert "reported separately" in rep.ledger.validity_domain["note"]


def test_neurofeedback_without_a_language_prior_reports_all_evidence_as_neural():
    decoded = torch.linspace(-1.0, 1.0, 32, dtype=_DT)
    _, rep = _loop().run(decoded)
    assert rep.prior_assisted_information_bits == 0.0
    assert rep.neural_fraction == pytest.approx(1.0)


def test_neurofeedback_write_port_must_be_declared():
    with pytest.raises(InterventionRefusal, match="undeclared perceptual port"):
        NeurofeedbackLoop(
            ContingencySpec(
                read_channel="eeg.alpha_power",
                write_port="stimulus",
                transform=lambda v: v,
                latency_s=0.1,
                update_rate_hz=4.0,
            )
        )
