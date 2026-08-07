"""Demo 1 - the multirate whole-brain forward run.

ONE latent trajectory -> regional activity (fast clock) + EEG (fast clock)
+ BOLD (slow clock), with no resampling anywhere.

Also runs the CHECKPOINT LOAD TRAP as an explicit negative control: the same
rollout with the 29 `_orig_mod.`-prefixed keys dropped (what a naive
strict=False load on CPU silently does).

Run from the main repo with PYTHONPATH=<main repo>.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import torch

from scwbd.foundation.anatomy import load_anatomy
from scwbd.foundation.config import load_config
from scwbd.foundation.model import SCWBD
from scwbd.foundation.simulate import THETA_NAMES, CorpusSpec, ThetaPrior, normalise_window, simulate_batch

OUT = pathlib.Path(sys.argv[1])
CKPT = pathlib.Path(
    "/home/brandonin/Documents/scwbd-wt/turing/checkpoints/scwbd-001-beta/stage_V_individual.pt"
)
CFG_YAML = CKPT.parent / "config.yaml"
DEV = torch.device("cuda")
SEED = 20260806
N_STEPS = 2500  # 20.0 s of simulated time at 125 Hz; 100 BOLD frames at 5 Hz

torch.cuda.set_per_process_memory_fraction(0.25, 0)
t0 = time.time()

# ---------------------------------------------------------------- checkpoint
ckpt_sha = hashlib.sha256(CKPT.read_bytes()).hexdigest()
payload = torch.load(CKPT, map_location="cpu", weights_only=False)
assert payload["format"] == "scwbd-foundation-checkpoint/1", payload["format"]
raw_sd = payload["model"]
n_ckpt_keys = len(raw_sd)
orig_keys = [k for k in raw_sd if "._orig_mod." in k]
n_orig = len(orig_keys)

# ---------------------------------------------------------------- construct
cfg = load_config(str(CFG_YAML))
assert cfg.model.n_regions == 454 and cfg.model.hidden == 288
anat = load_anatomy(device=DEV, n_cortex=400, force_fallback=True)
assert anat.n_regions == 454, anat.n_regions


def build() -> SCWBD:
    torch.manual_seed(SEED)
    m = SCWBD(cfg.model, anat).to(DEV)
    m.eval()
    return m


model = build()
n_model_keys = len(model.state_dict())

# fingerprint a `local` tensor BEFORE loading, so "loaded" is proven by a value
# change and not merely by the absence of an exception
probe_name = "local.inp.weight"
before = model.state_dict()[probe_name].detach().clone()

# ---- reconcile the torch.compile `_orig_mod.` prefix EXPLICITLY -------------
stripped = {k.replace("._orig_mod.", "."): v for k, v in raw_sd.items()}
assert len(stripped) == n_ckpt_keys, "prefix strip collided two keys"
missing, unexpected = model.load_state_dict(stripped, strict=True)
assert list(missing) == [] and list(unexpected) == [], (list(missing), list(unexpected))
n_loaded = n_ckpt_keys
after = model.state_dict()[probe_name].detach().clone()
probe_changed = not torch.equal(before, after)
assert probe_changed, "load_state_dict reported success but `local` weights did not change"

print(f"LOADED KEYS: {n_loaded} / {n_model_keys} model keys  (strict=True, 0 missing, 0 unexpected)")
print(f"  of which carried torch.compile's `_orig_mod.` prefix: {n_orig}")
print(f"  probe {probe_name} changed on load: {probe_changed}")

# what a naive strict=False load would have silently dropped
dropped_params = sum(raw_sd[k].numel() for k in orig_keys if hasattr(raw_sd[k], "numel"))
all_params = sum(v.numel() for v in raw_sd.values() if hasattr(v, "numel"))
trainable_report = payload["extra"]["parameter_report"]
trainable_total = trainable_report["TOTAL"]
trainable_dropped = trainable_report["local"] + trainable_report["residual"]
print(
    f"  naive strict=False would drop {n_orig} keys = {dropped_params:,} tensor elements "
    f"({100*dropped_params/all_params:.2f}% of all state-dict elements, "
    f"{100*trainable_dropped/trainable_total:.2f}% of TRAINABLE parameters)"
)

# ---------------------------------------------------------------- theta + context
theta = ThetaPrior().sample(1, seed=SEED, device=DEV)
theta_named = {n: float(theta[0, i]) for i, n in enumerate(THETA_NAMES)}
print("theta:", json.dumps({k: round(v, 4) for k, v in theta_named.items()}))

# in-distribution initial condition: a window from the SAME mechanistic generator
# that produced the training corpus, normalised the way training normalised it.
spec = CorpusSpec(dt=1e-3, duration_s=4.0, warmup_s=2.0, store_every=8, n_delay_bins=16)
ctx_act, ctx_meta = simulate_batch(
    anat=anat, backend_name="wilson_cowan", theta=theta, spec=spec, seed=SEED, device=DEV
)
ctx_np = normalise_window(ctx_act[0].float().cpu().numpy())  # (T, N)
n_ctx = cfg.data.context
y_ctx = torch.from_numpy(ctx_np[-n_ctx:]).to(DEV).unsqueeze(0)  # (1, 24, 454)
print(f"context: {tuple(y_ctx.shape)} from wilson_cowan, |z|max={float(np.abs(ctx_np).max()):.3f}")

# ---------------------------------------------------------------- the rollout
t_roll = time.time()
with torch.no_grad():
    roll = model.rollout(
        y_context=y_ctx, theta=theta, n_steps=N_STEPS, with_hemo=True, enforce_r05=False
    )
    eeg_mu, eeg_lv = model.eeg(roll.state)
    bold_mu, bold_lv = model.bold.signal(roll.hemo)
roll_s = time.time() - t_roll

dt_fast = cfg.model.dt_model
dt_slow = dt_fast * cfg.model.hemo_ratio
act = roll.activity[0].float().cpu().numpy()          # (T, 454)  125 Hz
eeg = eeg_mu[0].float().cpu().numpy()                 # (T, 64)   125 Hz
bold = bold_mu[0].float().cpu().numpy()               # (T_slow, 454) 5 Hz
print(
    f"rollout {roll_s:.1f}s -> state {tuple(roll.state.shape)} "
    f"activity {act.shape} @ {1/dt_fast:.1f} Hz, eeg {eeg.shape} @ {1/dt_fast:.1f} Hz, "
    f"bold {bold.shape} @ {1/dt_slow:.1f} Hz"
)
assert act.shape[0] == N_STEPS
assert bold.shape[0] == N_STEPS // cfg.model.hemo_ratio
assert eeg.shape[0] == N_STEPS

# ---------------------------------------------------------------- negative control
model_bad = build()
miss_b, unexp_b = model_bad.load_state_dict(raw_sd, strict=False)
n_dropped_keys = len(miss_b)
print(f"NEGATIVE CONTROL: naive load dropped {n_dropped_keys} keys ({len(unexp_b)} unexpected)")
with torch.no_grad():
    roll_b = model_bad.rollout(
        y_context=y_ctx, theta=theta, n_steps=N_STEPS, with_hemo=True, enforce_r05=False
    )
    eeg_b, _ = model_bad.eeg(roll_b.state)
act_b = roll_b.activity[0].float().cpu().numpy()
eeg_bn = eeg_b[0].float().cpu().numpy()


def _corr(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


control = {
    "naive_load_dropped_keys": int(n_dropped_keys),
    "naive_load_unexpected_keys": int(len(unexp_b)),
    "activity_correlation_correct_vs_naive": _corr(act, act_b),
    "eeg_correlation_correct_vs_naive": _corr(eeg, eeg_bn),
    "activity_rms_correct": float(np.sqrt((act**2).mean())),
    "activity_rms_naive": float(np.sqrt((act_b**2).mean())),
    "eeg_rms_correct": float(np.sqrt((eeg**2).mean())),
    "eeg_rms_naive": float(np.sqrt((eeg_bn**2).mean())),
}
print("control:", json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in control.items()}))
del model_bad, roll_b, eeg_b
torch.cuda.empty_cache()

# ---------------------------------------------------------------- spectra
def band_powers(x, fs):
    """x: (T, C). Returns (freqs, mean PSD across columns, band fractions, peak f)."""
    xd = x - x.mean(0, keepdims=True)
    win = np.hanning(xd.shape[0])[:, None]
    P = np.abs(np.fft.rfft(xd * win, axis=0)) ** 2
    f = np.fft.rfftfreq(xd.shape[0], 1.0 / fs)
    Pm = P.mean(1)
    tot = Pm[1:].sum()
    bands = {
        "sub_delta_0.02_0.5": (0.02, 0.5),
        "delta_0.5_4": (0.5, 4.0),
        "theta_4_8": (4.0, 8.0),
        "alpha_8_13": (8.0, 13.0),
        "beta_13_30": (13.0, 30.0),
        "gamma_30_62.5": (30.0, 62.5),
    }
    frac = {k: float(Pm[(f >= lo) & (f < hi)].sum() / tot) for k, (lo, hi) in bands.items()}
    return f, Pm, frac, float(f[1:][Pm[1:].argmax()])


f_act, P_act, frac_act, peak_f_act = band_powers(act, 1 / dt_fast)
f_eeg, P_eeg, frac_eeg, peak_f_eeg = band_powers(eeg, 1 / dt_fast)
spectral = {
    "note": (
        "Honest characterisation of what the free-running trajectory actually looks "
        "like. It is dominated by slow, largely non-oscillatory drift; it does not "
        "exhibit a physiological EEG spectrum, and no claim that it should is made."
    ),
    "activity_band_fraction": frac_act,
    "activity_peak_frequency_hz": peak_f_act,
    "eeg_band_fraction": frac_eeg,
    "eeg_peak_frequency_hz": peak_f_eeg,
}
print("activity bands:", json.dumps({k: round(v, 4) for k, v in frac_act.items()}))
print("eeg bands:", json.dumps({k: round(v, 4) for k, v in frac_eeg.items()}))

# ---------------------------------------------------------------- downsample for plotting
R = 5  # significant digits


def rnd(x):
    return [float(f"%.{R}g" % v) for v in np.asarray(x).ravel()]


def rnd2(x):
    return [rnd(row) for row in np.asarray(x)]


# regions spread across the three divisions (400 cortex, 32 subcortex, 22 cerebellum)
trace_regions = [0, 37, 91, 145, 199, 253, 307, 361, 399, 405, 420, 440]
trace_channels = list(range(0, 64, 8))
ch_names = list(model.eeg.channel_names)

t_fast = np.arange(N_STEPS) * dt_fast
t_slow = (np.arange(bold.shape[0]) + 1) * dt_slow

carpet_dec = cfg.model.hemo_ratio  # 25 -> 5 Hz, aligns the neural carpet with BOLD frames
act_carpet = act[carpet_dec - 1 :: carpet_dec]  # (100, 454)
eeg_dec = 10  # 12.5 Hz
eeg_carpet = eeg[::eeg_dec]  # (250, 64)

series = {
    "time_fast_s": {"units": "s", "rate_hz": 1 / dt_fast, "values": rnd(t_fast)},
    "time_slow_s": {"units": "s", "rate_hz": 1 / dt_slow, "values": rnd(t_slow)},
    "neural_activity_traces": {
        "units": "dimensionless (normalised regional activity, readout mean)",
        "rate_hz": 1 / dt_fast,
        "clock": "fast",
        "region_index": trace_regions,
        "shape": [len(trace_regions), N_STEPS],
        "values": rnd2(act[:, trace_regions].T),
    },
    "neural_activity_carpet": {
        "units": "dimensionless",
        "rate_hz": 1 / dt_slow,
        "clock": "fast (decimated 25x for display only)",
        "shape": list(act_carpet.T.shape),
        "note": "all 454 regions x 100 frames; decimation is for FILE SIZE, the model ran at 125 Hz",
        "values": rnd2(act_carpet.T),
    },
    "neural_activity_logvar_mean": {
        "units": "log variance",
        "values": rnd(roll.activity_logvar[0].float().cpu().numpy().mean(0)),
        "note": "per-region time-mean of the heteroscedastic predictive log-variance",
    },
    "eeg_traces": {
        "units": "arbitrary (lead field is analytic single-sphere; physical_scale_volts_per_unit recorded in provenance)",
        "rate_hz": 1 / dt_fast,
        "clock": "fast",
        "channel_index": trace_channels,
        "channel_names": [ch_names[i] for i in trace_channels],
        "shape": [len(trace_channels), N_STEPS],
        "values": rnd2(eeg[:, trace_channels].T),
    },
    "eeg_carpet": {
        "units": "arbitrary",
        "rate_hz": 1 / dt_fast / eeg_dec,
        "clock": "fast (decimated 10x for display only)",
        "channel_names": ch_names,
        "shape": list(eeg_carpet.T.shape),
        "values": rnd2(eeg_carpet.T),
    },
    "bold_traces": {
        "units": "fractional BOLD signal change (Buxton output equation, v0=0.04)",
        "rate_hz": 1 / dt_slow,
        "clock": "slow",
        "region_index": trace_regions,
        "shape": [len(trace_regions), bold.shape[0]],
        "values": rnd2(bold[:, trace_regions].T),
    },
    "psd_frequency_hz": {"units": "Hz", "values": rnd(f_act)},
    "psd_activity_mean_over_regions": {
        "units": "power (arbitrary), Hann-windowed periodogram averaged over 454 regions",
        "values": rnd(P_act),
    },
    "psd_eeg_mean_over_channels": {
        "units": "power (arbitrary), Hann-windowed periodogram averaged over 64 channels",
        "values": rnd(P_eeg),
    },
    "bold_carpet": {
        "units": "fractional BOLD signal change",
        "rate_hz": 1 / dt_slow,
        "clock": "slow",
        "shape": list(bold.T.shape),
        "note": "all 454 regions x 100 frames at the NATIVE slow-clock rate - no decimation",
        "values": rnd2(bold.T),
    },
}

summary = {
    "n_regions": int(act.shape[1]),
    "n_eeg_channels": int(eeg.shape[1]),
    "simulated_seconds": N_STEPS * dt_fast,
    "fast_clock_dt_s": dt_fast,
    "fast_clock_hz": 1 / dt_fast,
    "slow_clock_dt_s": dt_slow,
    "slow_clock_hz": 1 / dt_slow,
    "hemo_ratio": cfg.model.hemo_ratio,
    "n_fast_frames": int(act.shape[0]),
    "n_slow_frames": int(bold.shape[0]),
    "activity_min": float(act.min()),
    "activity_max": float(act.max()),
    "activity_rms": float(np.sqrt((act**2).mean())),
    "activity_region_sd_median": float(np.median(act.std(0))),
    "eeg_min": float(eeg.min()),
    "eeg_max": float(eeg.max()),
    "eeg_rms": float(np.sqrt((eeg**2).mean())),
    "bold_min": float(bold.min()),
    "bold_max": float(bold.max()),
    "bold_rms": float(np.sqrt((bold**2).mean())),
    "rollout_diagnostics": {
        k: (float(v) if isinstance(v, (int, float)) else v) for k, v in roll.diagnostics.items()
    },
    "rho": float(roll.rho),
    "spectral": spectral,
}

block = {
    "title": "Multirate whole-brain forward run: one latent trajectory, two instruments, two native clocks",
    "status": "ran",
    "what_it_is": (
        "A free-running (autoregressive, no teacher forcing) rollout of the trained "
        "SC-WBD-001-beta operator over 454 regions. The regional activity, the EEG "
        "channels and the BOLD signal are all read off the SAME state tensor: EEG "
        "through the lead-field head on the fast clock, BOLD through the "
        "Balloon-Windkessel head on the slow clock. Nothing is resampled."
    ),
    "clock_correction": (
        "The model has exactly TWO clocks, not three. The EEG head is a memoryless "
        "per-timestep map over the state, so it emits at the FAST clock (8 ms, 125 Hz) "
        "- there is no millisecond-rate EEG clock in scwbd.foundation. BOLD is stepped "
        "every 25th fast step (200 ms, 5 Hz). The multirate claim demonstrated here is "
        "125 Hz vs 5 Hz off one trajectory, not 1 kHz vs 5 Hz."
    ),
    "summary": summary,
    "series": series,
    "checkpoint_load_trap": {
        "description": (
            "29 of the 85 model keys carry torch.compile's `_orig_mod.` infix because the "
            "trainer compiles `local` and `residual` only on CUDA. A strict=False load "
            "against an uncompiled module drops all 29 and leaves those submodules at "
            "random initialisation while reporting success."
        ),
        "checkpoint_model_keys": n_ckpt_keys,
        "keys_with_orig_mod_prefix": n_orig,
        "affected_submodules": ["local", "residual"],
        "keys_loaded_this_run": n_loaded,
        "load_mode": "explicit `._orig_mod.` -> `.` rewrite, then load_state_dict(strict=True)",
        "missing_keys": [],
        "unexpected_keys": [],
        "probe_tensor": probe_name,
        "probe_value_changed_on_load": bool(probe_changed),
        "dropped_state_dict_elements_if_naive": int(dropped_params),
        "total_state_dict_elements": int(all_params),
        "dropped_fraction_of_state_dict_elements": dropped_params / all_params,
        "dropped_fraction_of_trainable_parameters": trainable_dropped / trainable_total,
        "trainable_parameters_total": int(trainable_total),
        "trainable_parameters_dropped": int(trainable_dropped),
        "note_on_the_80_2_percent_figure": (
            "80.2% is correct against TRAINABLE parameters (1,410,297 of 1,757,613 in the "
            "checkpoint's own parameter_report). Against all state-dict tensor elements "
            "(6,733,924, which include the anatomy-derived coupling buffers) the same 29 "
            "keys are 20.94%. Both denominators are reported so neither figure misleads."
        ),
        "negative_control": control,
    },
    "provenance": {
        "produced_by": "scratchpad/demo1_multirate.py",
        "device": str(DEV),
        "device_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "checkpoint": str(CKPT),
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_step": payload["step"],
        "checkpoint_stage": payload["stage"],
        "checkpoint_git_sha": payload["git_sha"],
        "loaded_key_count": n_loaded,
        "model_key_count": n_model_keys,
        "code_lineage": (
            "run from the main repo working tree, whose scwbd/foundation/{model,heads,"
            "anatomy,config,simulate,checkpoint}.py are BYTE-IDENTICAL to the checkpoint's "
            "training commit 00a61f98a8ff22a1e0fa44a01ad3a9b002233e26 (verified by cmp)"
        ),
        "repo_git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "seed": SEED,
        "theta_names": list(THETA_NAMES),
        "theta_values": theta_named,
        "theta_source": "ThetaPrior().sample(1, seed=%d) - sampled from the prior, NOT inferred by the amortised posterior" % SEED,
        "context_source": (
            "scwbd.foundation.simulate.simulate_batch(backend='wilson_cowan', "
            "duration_s=4.0, warmup_s=2.0) then normalise_window; last 24 frames used as "
            "y_context. Synthetic, from the same generator family as the training corpus."
        ),
        "anatomy": payload["extra"]["anatomy"],
        "lead_field": payload["extra"]["lead_field"],
        "wall_seconds": round(time.time() - t0, 2),
        "rollout_seconds": round(roll_s, 2),
    },
}

OUT.write_text(json.dumps(block, indent=1))
print("wrote", OUT, f"({OUT.stat().st_size/1e6:.2f} MB, {time.time()-t0:.1f}s total)")
