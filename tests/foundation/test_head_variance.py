"""The EEG/BOLD predictive variance must actually vary, and must be measurably dead when off.

`reports/training/p0_variance_channel.md`: run 1's heads emitted `lv =
log_noise.expand_as(y)`, a per-channel constant with no state and no horizon
axis, while the module docstring claimed a heteroscedastic noise model. Excess
NLL over the Gaussian entropy floor was +0.4467 nats -- 1.62x the entire deficit
to persistence -- and the whole of it was one scalar being 3.0x too small.

These tests exist so that defect cannot return silently. The un-repaired
behaviour is asserted to be *exactly* constant rather than described as such,
which is what makes the repaired case evidence instead of assertion.
"""

from __future__ import annotations

import math

import pytest
import torch

from scwbd.foundation.config import ModelConfig


def _model(**over):
    from scwbd.foundation.anatomy import load_anatomy
    from scwbd.foundation.model import SCWBD
    from scwbd.foundation.util import set_determinism

    set_determinism(0)
    anat = load_anatomy(device="cpu", n_cortex=8, n_subcortex=3, n_cerebellum=1, density=0.3, seed=7)
    cfg = ModelConfig(
        n_regions=12, hidden=32, n_local_layers=1, region_embed=8, context_dim=16,
        message_dim=6, n_delay_bins=2, n_spectral_modes=2, n_adaptation=2,
        n_uncertainty=3, encoder_channels=16, encoder_layers=1, n_eeg_channels=8,
        compile=False, use_bf16=False, **over,
    )
    return SCWBD(cfg, anat)


def _state(model, b=2, t=5):
    L = model.layout
    return torch.randn(b, t, model.n_regions, L.dim) * 0.5


# ---------------------------------------------------------------------------
# the un-repaired arm: constant, and MEASURED to be constant
# ---------------------------------------------------------------------------
def test_eeg_logvar_is_exactly_constant_without_observation():
    m = _model(state_dependent_variance=False)
    assert m.observation is None
    assert m.eeg.state_dependent_variance is False
    _, lv = m.eeg(_state(m))
    lv = lv.detach()
    # Not "small spread" -- exactly zero. This is run 1's defect, pinned.
    assert float(lv.std(dim=(0, 1)).max()) == 0.0, "logvar varies over time/batch but must not"
    assert float(lv.amax() - lv.amin()) == float(
        lv[0, 0].amax() - lv[0, 0].amin()
    ), "all variation is across channels only"


def test_bold_logvar_is_exactly_constant_without_observation():
    m = _model(state_dependent_variance=False)
    hemo = m.bold.initial(2, m.n_regions, torch.device("cpu"))
    _, lv = m.bold.signal(hemo)
    lv = lv.detach()
    assert float(lv.std(dim=0).max()) == 0.0


# ---------------------------------------------------------------------------
# the repaired arm: varies with state, at INITIALISATION
# ---------------------------------------------------------------------------
def test_eeg_logvar_varies_with_state_at_initialisation():
    """The point of the non-zero init: this must fire before any training.

    Zero-initialising the mixing map would make this pass vacuously -- the
    variance would be constant at step 0 and heteroscedastic only if training
    happened to find it, which is a shape rather than a mechanism.
    """
    m = _model(state_dependent_variance=True)
    assert m.observation is not None
    assert m.eeg.state_dependent_variance is True
    x = _state(m)
    _, lv = m.eeg(x)
    assert float(lv.std(dim=(0, 1)).max()) > 1e-6, "logvar does not vary with state at init"


def test_eeg_logvar_responds_to_the_uncertainty_channel_only():
    """Raising X^uncertainty must raise predicted variance; nothing else may."""
    from scwbd.foundation.uncertainty import UNCERTAINTY_COMPONENT

    m = _model(state_dependent_variance=True)
    x = _state(m)
    _, lv0 = m.eeg(x)
    sl = m.layout.slice(UNCERTAINTY_COMPONENT)
    x_hi = x.clone()
    x_hi[..., sl] = x_hi[..., sl] + 2.0
    _, lv1 = m.eeg(x_hi)
    # softplus weights make the map monotone: more uncertainty, never less variance.
    assert float((lv1 - lv0).min()) >= -1e-5
    assert float((lv1 - lv0).mean()) > 1e-4, "uncertainty channel does not drive the variance"


def test_instrument_floor_is_separately_parameterised():
    """RL-2: the floor must not be able to absorb the state term."""
    m = _model(state_dependent_variance=True)
    assert m.eeg.log_noise is not m.eeg.logvar_mix
    x = _state(m)
    _, lv0 = m.eeg(x)
    with torch.no_grad():
        m.eeg.log_noise += 1.0
    _, lv1 = m.eeg(x)
    # Shifting the floor shifts every channel by exactly 1.0 and changes no spread.
    assert torch.allclose(lv1 - lv0, torch.ones_like(lv0), atol=1e-5)


def test_observation_is_not_double_counted_in_parameters():
    """`set_observation` must not register the interface as a head submodule."""
    m = _model(state_dependent_variance=True)
    owned = {id(p) for p in m.observation.parameters()}
    assert owned, "interface has no parameters; this test would be vacuous"
    assert not ({id(p) for p in m.eeg.parameters()} & owned)
    assert not ({id(p) for p in m.bold.parameters()} & owned)
    assert len({id(p) for p in m.parameters()}) == len(list(m.parameters()))


# ---------------------------------------------------------------------------
# the closed form, and the never-trained detector
# ---------------------------------------------------------------------------
def test_calibrate_noise_floor_hits_the_closed_form():
    """Run 1 asked SGD for this over 900 steps and got 20% of the way."""
    m = _model(state_dependent_variance=False)
    torch.manual_seed(0)
    n_ch = m.eeg.log_noise.numel()
    true_sd = torch.linspace(0.25, 4.0, n_ch)
    resid = torch.randn(4096, 6, n_ch) * true_sd
    m.eeg.calibrate_noise_floor(resid)
    got = m.eeg.log_noise.detach().exp()
    assert torch.allclose(got, true_sd.pow(2), rtol=0.08), f"{got} vs {true_sd.pow(2)}"
    # And it is the NLL optimum, not merely close: perturbing it must not help.
    def nll(lv):
        return 0.5 * (math.log(2 * math.pi) + lv + resid.pow(2) * torch.exp(-lv)).mean()

    base = nll(m.eeg.log_noise.detach())
    assert base <= nll(m.eeg.log_noise.detach() + 0.1)
    assert base <= nll(m.eeg.log_noise.detach() - 0.1)


@pytest.mark.parametrize("head", ["eeg", "bold"])
def test_never_trained_noise_floor_is_detectable(head):
    """Run 1 shipped `bold.log_noise` at exactly -4.0 across 454 regions, silently."""
    m = _model(state_dependent_variance=False)
    h = getattr(m, head)
    r = h.noise_floor_report()
    assert r["at_initialisation"] is True
    assert r["warning"] is not None and "never received a gradient" in r["warning"]
    with torch.no_grad():
        h.log_noise += torch.randn_like(h.log_noise) * 0.1
    r2 = h.noise_floor_report()
    assert r2["at_initialisation"] is False
    assert r2["warning"] is None
