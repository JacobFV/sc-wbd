"""The measured BOLD path integrates the Balloon ODE across a real TR.

ISSUE-008's acceptance criterion, tested on the gradient rather than on a
checkpoint, so it does not need a 25-hour run to answer.

What was wrong. `real_bold_losses` read a four-channel component NAMED `hemo`
out of the learned regional state and handed it to a Balloon signal equation.
`BOLDHead.step` -- the actual integrator, and the only consumer of the five
physical parameters -- was never called on measured data, and in run 3 was never
called anywhere: `with_hemo` defaults False and no stage set it. So `signal()`
read two unconstrained latents as blood volume and deoxyhaemoglobin, `q/v`
wandered, `real_bold_nll` went 21.7 -> 4.4e6, and the five parameters stayed
bit-identical to their initialisation through all five stages.

And the clocks differed by 250x: 8 rollout steps at 8 ms were indexed against 8
BOLD frames at TR = 2 s.

The three tests below are the three things that have to be true together. Any
one alone can be satisfied by something that is not a fix:

* turning `with_hemo` on integrates the compartments and throws them away
  unless a loss consumes `roll.hemo`;
* consuming `roll.hemo` over 8 fast steps is still 64 ms of haemodynamics;
* and a softplus on `v`,`q` makes the number finite without making it physics,
  which is worse than an obviously broken one because nobody looks again.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from scwbd.foundation.anatomy import load_anatomy
from scwbd.foundation.config import load_config
from scwbd.foundation.model import SCWBD
from scwbd.foundation.util import set_determinism

BALLOON = ("log_kappa", "log_gamma", "log_tau", "alpha", "neural_gain")
CONFIG = "configs/run3/scwbd-003.yaml"
TR = 2.0  # ds002336's repetition time; the path reads it from the batch


@pytest.fixture(scope="module")
def anat():
    return load_anatomy(device="cpu")


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


def _small(model_cfg):
    return dataclasses.replace(
        model_cfg, hidden=64, n_local_layers=1, encoder_channels=8,
        encoder_layers=1, region_embed=8, context_dim=16,
    )


def _rollout_bold(m, model_cfg, anat):
    """Exactly the arithmetic `real_bold_losses` performs."""
    dt_slow = model_cfg.dt_model * model_cfg.hemo_ratio
    slow_per_frame = max(1, round(TR / dt_slow))
    frames = model_cfg.bold_predict_frames
    n_neural = frames * slow_per_frame * model_cfg.hemo_ratio

    th = torch.zeros(1, 6)
    m.set_mechanistic_theta(th, anat)
    ctx = torch.randn(1, 4, m.n_regions) * 0.1
    roll = m.rollout(
        y_context=ctx, theta=th, n_steps=n_neural, with_hemo=True, enforce_r05=False
    )
    hemo = roll.hemo[:, slow_per_frame - 1 :: slow_per_frame][:, :frames]
    steps = list(roll.hemo_steps)[slow_per_frame - 1 :: slow_per_frame][:frames]
    return roll, hemo, steps, slow_per_frame, frames, n_neural


def test_the_neural_clock_is_rolled_for_the_duration_a_bold_frame_covers(cfg, anat):
    """The 250x mismatch, closed by arithmetic rather than by assertion."""
    mc = _small(cfg.model)
    dt_slow = mc.dt_model * mc.hemo_ratio
    slow_per_frame = round(TR / dt_slow)
    n_neural = mc.bold_predict_frames * slow_per_frame * mc.hemo_ratio

    assert slow_per_frame == 10, (
        f"TR {TR}s / dt_slow {dt_slow}s should be 10 slow steps per frame, got "
        f"{slow_per_frame}. hemo_ratio=25 gives dt_slow=0.2s; the run-3 constant "
        "was also wrong for a real TR (2/0.008 = 250, not 25)."
    )
    simulated_seconds = n_neural * mc.dt_model
    assert simulated_seconds == pytest.approx(mc.bold_predict_frames * TR), (
        f"the rollout spans {simulated_seconds}s but the target spans "
        f"{mc.bold_predict_frames * TR}s. That is the ISSUE-008 defect: the slow "
        "modality put on the fast clock."
    )


def test_the_balloon_parameters_receive_gradient_from_measured_bold(cfg, anat):
    """The inversion gate #6's docstring names as the acceptance criterion."""
    set_determinism(0)
    mc = _small(cfg.model)
    m = SCWBD(mc, anat)
    roll, hemo, steps, _, frames, _ = _rollout_bold(m, mc, anat)

    # The log-variance is read from the fast-clock state at the step each
    # retained sample was taken, which is what `hemo_steps` records.
    mu, lv = m.bold.signal(hemo, roll.state[:, steps])
    (mu.pow(2).mean() + lv.pow(2).mean()).backward()

    frozen = [
        n for n in BALLOON
        if (g := getattr(m.bold, n).grad) is None or float(g.abs().sum()) == 0.0
    ]
    assert not frozen, (
        f"{frozen} received no gradient from a measured-BOLD loss. The Balloon "
        "ODE is still not on this path -- check that a loss consumes roll.hemo "
        "rather than a `hemo`-named slice of the regional state."
    )


def test_one_sample_is_emitted_per_repetition_time(cfg, anat):
    """Not one per fast step, and not one per slow step."""
    set_determinism(0)
    mc = _small(cfg.model)
    m = SCWBD(mc, anat)
    roll, hemo, steps, slow_per_frame, frames, n_neural = _rollout_bold(m, mc, anat)

    assert roll.hemo is not None, "with_hemo produced nothing; the ODE did not run"
    assert roll.hemo.shape[1] == n_neural // mc.hemo_ratio == frames * slow_per_frame
    assert hemo.shape[1] == frames, (
        f"expected {frames} BOLD samples, got {hemo.shape[1]}: the slow clock is "
        "not being decimated to the repetition time"
    )
    assert len(steps) == frames
    # Each retained sample must sit at the END of its frame, not the start.
    assert steps[0] == slow_per_frame * mc.hemo_ratio - 1, (
        f"first sample taken at fast step {steps[0]}, expected "
        f"{slow_per_frame * mc.hemo_ratio - 1} -- the sample should be the state "
        "at the end of the interval it represents"
    )


def test_the_compartments_stay_physiological_over_a_full_frame(cfg, anat):
    """`q/v` wandering without bound is what produced 4.4e6.

    Integrated by the ODE from equilibrium, `v` and `q` stay near 1 -- which is
    the property that made `signal()` meaningful in the simulated path and was
    absent in the measured one.
    """
    set_determinism(0)
    mc = _small(cfg.model)
    m = SCWBD(mc, anat)
    with torch.no_grad():
        roll, hemo, *_ = _rollout_bold(m, mc, anat)
    v, q = hemo[..., 2], hemo[..., 3]
    assert float(v.min()) > 0.0, "blood volume went non-positive"
    assert float(v.max()) < 10.0 and float(q.abs().max()) < 10.0, (
        f"compartments left a physiological range (v max {float(v.max()):.3g}, "
        f"|q| max {float(q.abs().max()):.3g}); the ODE is running but its inputs "
        "or parameters are not"
    )
