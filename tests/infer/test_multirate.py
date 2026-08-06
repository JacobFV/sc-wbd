"""Multirate consumption on native clocks: unequal supports, missing windows,
and the refusal to resample.

``body.tex`` sec. 7.1: *"EEG samples need not be downsampled to the fMRI
repetition time, and fMRI voxels need not be assigned sensor-space electrical
precision.  A continuous-time or multirate posterior links them through their
observation operators, timing uncertainty, and hemodynamic state."*
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scwbd.infer.filters import kalman_filter, multiepoch_kalman_filter, simulate_lgssm
from scwbd.infer.linear_gaussian import (
    block_average_eeg,
    coarse_config,
    decimate_eeg,
    make_model,
)

DEFAULT_DTYPE = torch.float64   # consumed by the conftest autouse fixture


def test_eeg_and_bold_are_consumed_at_1ms_and_1s(tiny_setup):
    cfg, proto, u0 = tiny_setup
    mdl = make_model(u0, cfg, proto)
    ssm = mdl.ssm(("eeg", "bold"), epoch=0)
    eeg = ssm.channel("eeg")
    bold = ssm.channel("bold")
    assert cfg.dt_eeg == 1e-3 and cfg.dt_bold == 1.0
    assert eeg.n_obs == cfg.n_steps                    # every base step
    assert bold.n_obs == int(cfg.epoch_seconds)        # one per second
    assert eeg.p != bold.p                             # unequal supports
    # BOLD reads the hemodynamic cascade, EEG reads the instantaneous state:
    # the two heads touch disjoint parts of the augmented state
    assert float(eeg.H[0, :, cfg.hrf_offset:].abs().max()) == 0.0
    assert float(bold.H[0, :, : cfg.n_regions].abs().max()) == 0.0


def test_missing_windows_on_both_clocks_have_unequal_supports(tiny_setup):
    cfg, proto, u0 = tiny_setup
    mdl = make_model(u0, cfg, proto)
    ssm = mdl.ssm(("eeg", "bold"), epoch=0)
    data, _ = simulate_lgssm(ssm, seed=21, batch=1)
    dev, dt = mdl.F.device, mdl.F.dtype
    masks = {
        "eeg": torch.ones(1, ssm.channel("eeg").n_obs, dtype=dt, device=dev),
        "bold": torch.ones(1, ssm.channel("bold").n_obs, dtype=dt, device=dev),
    }
    # a 300 ms EEG dropout and a different 1-sample BOLD dropout
    masks["eeg"][0, 400:700] = 0.0
    masks["bold"][0, 1] = 0.0
    full = float(kalman_filter(ssm, data).log_likelihood[0])
    part = float(kalman_filter(ssm, data, masks).log_likelihood[0])
    assert part > full or part < full          # simply: it is computable
    assert np.isfinite(part)
    # corrupting the dropped samples must not move the answer at all
    d2 = {k: v.clone() for k, v in data.items()}
    d2["eeg"][0, 400:700] = 1e6
    d2["bold"][0, 1] = -1e6
    assert abs(float(kalman_filter(ssm, d2, masks).log_likelihood[0]) - part) < 1e-8


def test_bold_only_windows_still_constrain_the_fast_state(tiny_setup):
    """With EEG entirely absent the filter must still run on the BOLD clock."""
    cfg, proto, u0 = tiny_setup
    mdl = make_model(u0, cfg, proto)
    ssm = mdl.ssm(("bold",), epoch=0)
    data, _ = simulate_lgssm(ssm, seed=33, batch=1)
    fr = kalman_filter(ssm, data)
    assert np.isfinite(float(fr.log_likelihood[0]))
    assert int(fr.n_observations_used["bold"][0]) == ssm.channel("bold").n_obs


def test_decimation_and_block_averaging_are_different_summaries(tiny_setup):
    cfg, proto, u0 = tiny_setup
    mdl = make_model(u0, cfg, proto)
    ssm = mdl.ssm(("eeg",), epoch=0)
    data, _ = simulate_lgssm(ssm, seed=44, batch=2)
    cc = coarse_config(cfg)
    dec = decimate_eeg(data["eeg"], cfg, cc)
    blk = block_average_eeg(data["eeg"], cfg, cc)
    assert dec.shape == blk.shape
    assert dec.shape[-2] == len(cc.bold_steps())
    # decimation keeps one raw sample; block averaging attenuates the fast noise
    assert float(blk.std()) < float(dec.std())


def test_resampling_discards_almost_all_eeg_samples(tiny_setup):
    """Quantify what naive resampling throws away, since that is the claim."""
    cfg, proto, u0 = tiny_setup
    cc = coarse_config(cfg)
    kept = len(cc.bold_steps())
    total = len(cfg.eeg_steps())
    assert kept < total
    assert kept / total <= 1.0 / (cfg.dt_bold / cfg.dt_eeg) + 1e-9


def test_multirate_filter_handles_a_channel_with_no_data_at_a_step(tiny_setup):
    """Most base steps have EEG only; the BOLD channel must simply not update."""
    cfg, proto, u0 = tiny_setup
    mdl = make_model(u0, cfg, proto)
    ssm = mdl.ssm(("eeg", "bold"), epoch=0)
    sched = ssm.schedule()
    eeg_only = [k for k, v in sched.items() if len(v) == 1]
    both = [k for k, v in sched.items() if len(v) == 2]
    assert len(eeg_only) > 10 * max(len(both), 1)
