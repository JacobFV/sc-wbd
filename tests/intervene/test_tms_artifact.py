"""TMS--EEG artifact separation. SIMULATION ONLY.

Appendix A requires physical dose, instrument saturation and peripheral
co-stimulation to be separable from any candidate cortical response, and
requires artifact-injection recovery plus negative controls before a cortical
claim.
"""

from __future__ import annotations

import pytest
import torch

from scwbd.intervene.base import InterventionRefusal
from scwbd.intervene.tms.artifact import (
    AmplifierSpec,
    EvokedTemplate,
    TMSEEGArtifactModel,
)

_DT = torch.float64


def _model() -> TMSEEGArtifactModel:
    return TMSEEGArtifactModel()


def _injected(t: torch.Tensor, amp: float = 3.0) -> torch.Tensor:
    """A plausible-looking 'cortical' waveform used only as a known injection."""
    return amp * torch.exp(-0.5 * ((t - 0.060) / 0.015) ** 2) - 0.6 * amp * torch.exp(
        -0.5 * ((t - 0.030) / 0.008) ** 2
    )


def test_components_are_named_and_separable():
    m = _model()
    t = m.times()
    y, parts = m.simulate(t, e_field_v_per_m=90.0, decay_uv=5.0)
    comp = m.separate(t, y, e_field_v_per_m=90.0)
    energies = comp.component_energy()
    assert set(energies) == {
        "pulse", "saturation", "auditory", "somatosensory", "decay", "residual"
    }
    # with no injected signal, the residual is tiny next to the artifact stack
    assert energies["residual"] < 1e-3 * (
        energies["auditory"] + energies["somatosensory"] + energies["saturation"]
    )


def test_blanking_window_is_excluded_from_the_fit():
    m = _model()
    t = m.times()
    mask = m.valid_mask(t)
    blanked = t[~mask]
    assert float(blanked.min()) >= 0.0
    assert float(blanked.max()) < m.amplifier.blanking_s
    assert not bool(mask[(t >= 0) & (t < m.amplifier.blanking_s)].any())


def test_amplifier_saturation_scales_with_physical_dose_not_with_biology():
    m = _model()
    t = m.times()
    lo = m.separate(*_run(m, t, 40.0), e_field_v_per_m=40.0)
    hi = m.separate(*_run(m, t, 120.0), e_field_v_per_m=120.0)
    e_lo = lo.component_energy()["saturation"]
    e_hi = hi.component_energy()["saturation"]
    assert e_hi > 4 * e_lo  # saturation energy grows ~ (E-field)^2


def _run(m, t, ef):
    y, _ = m.simulate(t, e_field_v_per_m=ef, decay_uv=5.0)
    return t, y


def test_peripheral_and_instrumental_components_are_separately_addressable():
    m = _model()
    t = m.times()
    y, _ = m.simulate(t, e_field_v_per_m=90.0)
    c = m.separate(t, y, e_field_v_per_m=90.0)
    assert float(c.peripheral().abs().max()) > 0.0
    assert float(c.instrumental().abs().max()) > 0.0
    # they are different signals, not two views of one
    assert not torch.allclose(c.peripheral(), c.instrumental())


def test_the_residual_refuses_to_be_called_a_cortical_response():
    m = _model()
    t = m.times()
    y, _ = m.simulate(t, e_field_v_per_m=90.0)
    c = m.separate(t, y, e_field_v_per_m=90.0)
    with pytest.raises(InterventionRefusal) as e:
        c.as_cortical_response()
    assert e.value.code == "R04"
    assert c.ledger.validity_domain["residual_is_not_cortical"] is True


def test_artifact_injection_recovery():
    m = _model()
    t = m.times()
    inj = _injected(t, amp=3.0)
    rep = m.injection_recovery(t, inj, e_field_v_per_m=90.0, noise_uv=0.2, seed=1)
    assert rep["recovery_correlation"] > 0.8
    # gain < 1: part of a smooth injection is absorbed by the artifact basis.
    # That loss is the point of running the test, not a defect to tune away.
    assert 0.6 < rep["recovery_gain"] < 1.2


def test_injection_recovery_degrades_when_the_injected_signal_is_buried():
    m = _model()
    t = m.times()
    strong = m.injection_recovery(t, _injected(t, 5.0), e_field_v_per_m=90.0,
                                  noise_uv=0.2, seed=2)
    weak = m.injection_recovery(t, _injected(t, 0.05), e_field_v_per_m=90.0,
                                noise_uv=2.0, seed=2)
    assert strong["recovery_correlation"] > weak["recovery_correlation"]
    assert weak["recovery_correlation"] < 0.3


def test_the_direct_pulse_component_is_reported_as_unidentifiable():
    """It lives inside the blanking window, so nothing constrains it.

    Fitting an unconstrained column would amplify noise and then subtract it
    from the residual as if it were a known artifact.
    """
    m = _model()
    t = m.times()
    y, _ = m.simulate(t, e_field_v_per_m=90.0, noise_uv=0.2, seed=1)
    c = m.separate(t, y, e_field_v_per_m=90.0)
    assert "pulse" in c.ledger.validity_domain["unidentifiable_components"]
    assert float(c.pulse.abs().max()) == 0.0
    # and with the rank deficiency handled, the residual is just the noise
    assert float(c.residual[c.valid_mask].std()) == pytest.approx(0.2, rel=0.25)


def test_negative_control_no_injection_gives_no_recovered_signal():
    m = _model()
    t = m.times()
    zero = torch.zeros_like(t)
    rep = m.injection_recovery(t, zero, e_field_v_per_m=90.0, noise_uv=0.5, seed=3)
    assert abs(rep["recovery_gain"]) < 1e-6 or rep["injected_rms_uv"] == 0.0


def test_auditory_and_somatosensory_templates_have_the_expected_shape():
    aep = EvokedTemplate.auditory_click()
    t = torch.linspace(0.0, 0.4, 2001, dtype=_DT)
    v = aep.evaluate(t)
    n100 = float(t[int(v.argmin())])
    p200 = float(t[int(v.argmax())])
    assert 0.08 < n100 < 0.12
    assert 0.15 < p200 < 0.22

    sep = EvokedTemplate.somatosensory()
    w = sep.evaluate(t)
    assert float(t[int(w.argmax())]) < 0.03  # early muscle burst


def test_amplifier_clipping_bounds_the_recorded_trace():
    amp = AmplifierSpec(range_uv=500.0)
    m = TMSEEGArtifactModel(amplifier=amp)
    t = m.times()
    y, _ = m.simulate(t, e_field_v_per_m=200.0)
    assert float(y.abs().max()) <= 500.0 + 1e-9
