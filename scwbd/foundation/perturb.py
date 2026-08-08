"""Measured perturbation: TMS-EEG epochs, and the drive that stands in for the pulse.

This is the only source in the mixture where something was **done to** the brain
rather than recorded from it. ds004024 delivered single-pulse TMS over one
primary motor cortex with 64-channel EEG running, and
``scwbd.sources.perturbation.ds004024`` already loads those runs at their native
20 kHz with the pulse onsets exact on the amplifier clock. This module is the
training-facing half: it resamples epochs onto the model's clock, caches them,
and supplies the latent drive that represents the pulse.

What this closes, and what it does not
--------------------------------------
It closes "no model here has been trained on stimulation data". 003 is trained
on measured TMS-EEG and predicts a measured evoked response.

It does **not** close the field-to-response map, and the reason is a property of
the release rather than of the model. Computing a drive from a coil requires the
coil's position and orientation; ds004024 was MRI-navigated but **no per-pulse
pose log is distributed** (``ds004024.py`` records this as
``coil_pose: Provenance.UNKNOWN``, and notes it "disables the E-field operator
path outright"). With no pose there is no E-field, and with no E-field there is
nothing to validate the map *from*. So the spatial profile of the drive here is
**learned**, anchored to the atlas' motor parcels of the hemisphere the MEP says
was stimulated -- not computed from physics. ``possibilities/`` keeps "the
forward model cannot predict a perturbation it has not seen" as a live falsifier
and 003 does not close it: one target site, one intensity, two participants.

Two resolution facts, stated because they bound every number read off this path
-------------------------------------------------------------------------------
The model's fast clock is ``dt_model`` = 8 ms. TEP components earlier than about
one step -- N15, P30 -- are **below the model's temporal resolution** and no
claim about them is supported here. N45 onward is resolvable, N100 and P180
comfortably so.

The first ``artifact_exclusion_s`` after each pulse is excluded from the
likelihood by a mask rather than by deletion, so the rollout still passes
through the excluded interval (the state evolves) while the excluded samples
contribute no gradient. Deleting them would splice two segments together and
ask the operator to integrate across a discontinuity it did not produce.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

__all__ = [
    "TMSEpochConfig",
    "TMSEpochDataset",
    "TMSDrive",
    "motor_parcels",
]

LOGGER = logging.getLogger(__name__)

#: Bumped when the cached epoch content changes meaning.
CACHE_VERSION = 1


def motor_parcels(anat, hemisphere: str) -> np.ndarray:
    """Indices of the atlas' somatomotor parcels in one hemisphere.

    The anchor for the drive's spatial support. ds004024 targeted the first
    dorsal interosseous representation of M1; the Schaefer 7-network parcellation
    names its somatomotor parcels ``*_LH_SomMot_*`` / ``*_RH_SomMot_*`` and does
    not resolve the hand knob specifically, so the support is the network and the
    *profile within it* is learned. Claiming a hand-knob parcel the atlas does
    not define would be a precision the parcellation cannot carry.
    """
    if hemisphere not in ("left", "right"):
        raise ValueError(f"hemisphere must be 'left' or 'right', got {hemisphere!r}")
    tag = "_LH_" if hemisphere == "left" else "_RH_"
    idx = [i for i, lab in enumerate(anat.labels) if tag in str(lab) and "SomMot" in str(lab)]
    if not idx:
        raise ValueError(
            f"the anatomy prior has no {hemisphere} somatomotor parcels; the drive "
            "has no declared support and refuses to fall back to whole-brain"
        )
    return np.asarray(idx, dtype=np.int64)


# ======================================================================
# epochs on the model's clock
# ======================================================================
@dataclass
class TMSEpochConfig:
    """Where the spTMS runs are, and how they are brought onto the model clock."""

    root: str = "data/ds004024/1.0.0"
    cache_dir: str = "data/foundation_cache/tms_epochs"
    source: str = "ds004024_perturb"
    #: The model's fast clock, in Hz. Epochs are resampled to it.
    fs_target: float = 125.0
    #: Epoch bounds relative to pulse onset, seconds.
    tmin: float = -0.200
    tmax: float = 0.400
    #: Post-pulse interval the source card declares artefact-dominated. Masked
    #: out of the likelihood, not deleted.
    artifact_exclusion_s: float = 0.010
    #: Also blanked *before* the pulse. The trigger's own edge precedes the
    #: recorded onset sample by an undocumented amount (the TMS device's clock
    #: relative to the amplifier is not distributed), so a symmetric margin is
    #: cheaper than assuming the artefact begins exactly at sample zero.
    blank_pre_s: float = 0.002
    #: Low-pass applied before resampling. 45 Hz is the same ceiling the
    #: observational sources use, so a TEP and a resting window are commensurable.
    #: There is deliberately **no high-pass**: see ``_prepare``. A 0.5 Hz
    #: transition needs ~6.6 s of filter and an epoch is 0.6 s, so the filter
    #: silently does nothing; the DC offset is removed by pre-pulse baseline
    #: correction instead, which is the operation actually wanted here.
    h_freq: float | None = 45.0
    max_subjects: int | None = None
    max_runs_per_subject: int | None = None
    #: Cap on epochs kept per run, thinned evenly.
    max_epochs_per_run: int | None = 64
    #: Reject an epoch whose pre-pulse baseline exceeds this many robust sigma.
    #: Applied to the PRE-pulse interval only -- the post-pulse interval contains
    #: the response, and rejecting on it would select for small responses.
    z_max: float = 12.0


class TMSEpochDataset(Dataset):
    """spTMS epochs at the model's rate, with the pulse at a known step.

    One item is one pulse::

        {"eeg": (T, C) resampled epoch, robust-scaled,
         "onset_step": int   index of the pulse on the epoch's time axis,
         "loss_mask": (T,)   False across the artefact window and pre-pulse,
         "hemisphere": "left"|"right",
         "subject", "run", "source"}

    ``hemisphere`` is the level ``derive_stimulated_hemisphere`` recovered from
    the lateralised MEP. A run whose laterality falls below that function's
    stated effect-size floor is **skipped**, not assigned the sign of a noisy
    mean: the drive's spatial support is chosen by this label, so guessing it
    would put the pulse in the wrong hemisphere and train against it.
    """

    def __init__(self, cfg: TMSEpochConfig, *, build: bool = True) -> None:
        self.cfg = cfg
        self.source = cfg.source
        self.cache_root = Path(cfg.cache_dir)
        self.runs: list[dict[str, Any]] = []
        self.index: list[tuple[int, int]] = []
        self.skipped: list[dict[str, str]] = []
        self._memmaps: dict[int, np.ndarray] = {}
        self._pid = -1
        if build:
            self.build()

    # -- discovery / cache ------------------------------------------------
    def _discover(self) -> list[tuple[str, str]]:
        from scwbd.sources.perturbation.ds004024 import FETCHED_RUNS, available_subjects

        subs = list(available_subjects(self.cfg.root))
        if self.cfg.max_subjects is not None:
            subs = subs[: self.cfg.max_subjects]
        runs = list(FETCHED_RUNS)
        if self.cfg.max_runs_per_subject is not None:
            runs = runs[: self.cfg.max_runs_per_subject]
        return [(s, r) for s in subs for r in runs]

    def build(self) -> None:
        self.cache_root.mkdir(parents=True, exist_ok=True)
        kept: list[dict[str, Any]] = []
        for sub, run in self._discover():
            meta_path = self.cache_root / f"{sub}_run-{run}.json"
            meta = None
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if meta.get("cache_version") != CACHE_VERSION:
                        meta = None
                except (OSError, ValueError):
                    meta = None
            if meta is None:
                meta = self._prepare(sub, run)
                if meta is not None:
                    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            if meta is None or meta.get("status") != "ok":
                self.skipped.append(
                    {
                        "run": f"{sub}/run-{run}",
                        "reason": (meta or {}).get("reason", "unreadable"),
                    }
                )
                continue
            if not (self.cache_root / meta["shard"]).exists():
                continue
            kept.append(meta)
        self.runs = kept
        self.index = [
            (i, j) for i, m in enumerate(kept) for j in range(int(m["n_epochs"]))
        ]

    def _prepare(self, sub: str, run: str) -> dict[str, Any] | None:
        """Load one run at 20 kHz, resample the epochs, write a shard."""
        from scwbd.sources.perturbation.ds004024 import DatasetNotAvailable, load_run

        base = {"cache_version": CACHE_VERSION, "subject": sub, "run": f"run-{run}"}
        try:
            ep = load_run(
                self.cfg.root,
                sub,
                run,
                tmin=self.cfg.tmin,
                tmax=self.cfg.tmax,
                artifact_exclusion_s=self.cfg.artifact_exclusion_s,
            )
        except DatasetNotAvailable as exc:
            return {**base, "status": "absent", "reason": str(exc)[:200]}
        except Exception as exc:  # noqa: BLE001 - a bad run is a fact, not a crash
            return {**base, "status": "unreadable", "reason": f"{type(exc).__name__}: {exc}"[:200]}

        hemi = None
        for cv in ep.control_graph.manipulated:
            if cv.name == "stimulated_hemisphere":
                hemi = cv.value
        if hemi not in ("left", "right"):
            # Not a guess. The drive's support is chosen by this label.
            return {
                **base,
                "status": "hemisphere_unrecovered",
                "reason": (
                    "the lateralised MEP did not decide the stimulated hemisphere "
                    "above the stated effect-size floor; the drive's spatial "
                    "support cannot be placed and this run is skipped rather than "
                    "assigned the sign of a noisy mean"
                ),
            }

        eeg = ep.picks(["eeg"])
        data = np.array(eeg.data, dtype=np.float64)  # (n_trials, C, T) at 20 kHz, volts
        names = list(eeg.channel_names)

        # --- blank the pulse before filtering, or the filter spreads it ------
        # Measured on sub-CON001 run-01: the post-pulse peak is 5.26e-1 V
        # against a 4.4e-2 V pre-pulse baseline. A zero-phase 0.5 Hz high-pass
        # has a long, symmetric impulse response, so feeding it that step
        # smears the artefact BACKWARDS in time -- the pre-pulse robust z went
        # from 9.1 unfiltered to 12.5-13.6 filtered, and every epoch was then
        # rejected by a baseline test for contamination the preprocessing had
        # introduced.
        #
        # The blanked samples are replaced by each channel's own pre-pulse
        # median, a constant, and they are masked out of the likelihood as
        # well, so nothing here is ever a target. This substitutes samples that
        # are excluded from every loss in order to stop them corrupting the
        # samples that are not. It is not an interpolation of the response and
        # no value inside the window is used as evidence.
        t_native = eeg.times()
        blank = (t_native >= -self.cfg.blank_pre_s) & (t_native <= self.cfg.artifact_exclusion_s)
        baseline = t_native < -self.cfg.blank_pre_s
        if blank.any() and baseline.any():
            fill = np.median(data[:, :, baseline], axis=2, keepdims=True)
            data[:, :, blank] = fill
        n_blanked = int(blank.sum())

        # --- baseline correction, NOT a high-pass ----------------------------
        # ds004024's amplifier ran in DC mode (`dataset_description.json`: "EEG
        # was collected from 64 channels in DC mode"), so each channel carries a
        # large static offset. The obvious fix -- a 0.5 Hz high-pass, which is
        # what every other source in this mixture uses -- CANNOT work here: a
        # 0.5 Hz transition needs roughly 6.6 s of filter and an epoch is 0.6 s,
        # so MNE shortens the filter and the offset survives. Measured before
        # this was fixed: the trial-averaged global field power sat flat at ~57
        # scaled units from -200 ms to +400 ms, a static spatial pattern with no
        # time course, and the post/pre ratio was 0.97 -- an evoked response
        # would have been invisible underneath it.
        #
        # Per-trial per-channel pre-pulse mean subtraction is the standard
        # evoked-response baseline and removes exactly the offset the high-pass
        # was there to remove. The low-pass is kept: 45 Hz is achievable in
        # 0.6 s and is the same ceiling the observational sources use, so a TEP
        # and a resting window stay commensurable.
        if baseline.any():
            data = data - data[:, :, baseline].mean(axis=2, keepdims=True)

        import mne

        info = mne.create_info(names, ep.sfreq, ch_types="eeg", verbose="ERROR")
        epochs = mne.EpochsArray(data, info, tmin=ep.tmin, verbose="ERROR")
        if self.cfg.h_freq is not None:
            epochs.filter(None, self.cfg.h_freq, verbose="ERROR", n_jobs=1)
        epochs.resample(self.cfg.fs_target, verbose="ERROR")
        arr = epochs.get_data(copy=True)  # (n_trials, C, T') volts
        times = epochs.times

        # Average reference, matching DS004024RestDataset so the two share a head.
        arr = arr - arr.mean(axis=1, keepdims=True)

        onset_step = int(np.argmin(np.abs(times)))
        pre = arr[:, :, :onset_step]
        if pre.shape[2] < 2:
            return {**base, "status": "no_baseline", "reason": "tmin leaves <2 pre-pulse samples"}

        # One robust scale for the whole run, from the PRE-pulse baseline only.
        # Scaling by a window that contains the evoked response would divide the
        # response out of itself and make its amplitude unmeasurable.
        med = np.median(pre, axis=(0, 2), keepdims=True)
        mad = np.median(np.abs(pre - med), axis=(0, 2), keepdims=True)
        finite = mad[np.isfinite(mad) & (mad > 0)]
        scale = float(np.median(finite)) if finite.size else 0.0
        if not np.isfinite(scale) or scale <= 0:
            return {**base, "status": "flat", "reason": "pre-pulse robust scale is zero"}
        arr = arr / scale
        z_den = 1.4826 * np.maximum(mad / scale, 1e-12)
        z_med = med / scale

        keep = []
        for i in range(arr.shape[0]):
            p = arr[i, :, :onset_step]
            if not np.isfinite(arr[i]).all():
                continue
            if float(np.abs((p - z_med[0]) / z_den[0]).max()) > self.cfg.z_max:
                continue
            keep.append(i)
        cap = self.cfg.max_epochs_per_run
        if cap is not None and len(keep) > cap > 0:
            sel = np.linspace(0, len(keep) - 1, cap).round().astype(int)
            keep = [keep[i] for i in np.unique(sel)]
        if not keep:
            return {**base, "status": "no_clean_epochs", "reason": "every epoch failed the pre-pulse z test"}

        out = np.ascontiguousarray(arr[keep].transpose(0, 2, 1), dtype=np.float32)  # (n, T, C)
        shard = f"{sub}_run-{run}.npy"
        tmp = self.cache_root / (shard + ".tmp")
        # Through a handle: `np.save` appends `.npy` to any path that does not
        # already end in it, so `np.save(x.npy.tmp)` silently writes
        # `x.npy.tmp.npy` and the rename then fails on a file that is not there.
        with open(tmp, "wb") as fh:
            np.save(fh, out)
        tmp.replace(self.cache_root / shard)

        # The likelihood is scored on post-pulse samples outside the artefact
        # window. Pre-pulse samples are the assimilation context, not a target.
        mask = (times > self.cfg.artifact_exclusion_s)
        return {
            **base,
            "status": "ok",
            "shard": shard,
            "n_epochs": int(out.shape[0]),
            "n_epochs_available": int(arr.shape[0]),
            "n_times": int(out.shape[1]),
            "channels": names,
            "fs": float(self.cfg.fs_target),
            "fs_native": float(ep.sfreq),
            "onset_step": onset_step,
            "times_s": [round(float(t), 6) for t in times],
            "loss_mask": mask.tolist(),
            "n_scored_steps": int(mask.sum()),
            "n_blanked_native_samples": n_blanked,
            "blank_window_s": [-self.cfg.blank_pre_s, self.cfg.artifact_exclusion_s],
            "blank_note": (
                "Samples in the blank window were replaced by each channel's "
                "pre-pulse median at the native rate, BEFORE filtering, because "
                "a zero-phase 0.5 Hz high-pass smears a 0.5 V step backwards "
                "across the whole epoch. They are also masked out of the "
                "likelihood, so no blanked value is ever a target."
            ),
            "hemisphere": hemi,
            "scale_volts": scale,
            "tmin": float(self.cfg.tmin),
            "artifact_exclusion_s": float(self.cfg.artifact_exclusion_s),
            "record_id": ep.record_id,
            "resolution_note": (
                f"resampled {ep.sfreq:.0f} -> {self.cfg.fs_target:.0f} Hz. TEP "
                "components earlier than one model step (8 ms) are below the "
                "model's temporal resolution and are not claimed."
            ),
        }

    # -- Dataset ----------------------------------------------------------
    def _shard(self, i: int) -> np.ndarray:
        import os

        if os.getpid() != self._pid:
            self._memmaps = {}
            self._pid = os.getpid()
        mm = self._memmaps.get(i)
        if mm is None:
            mm = np.load(self.cache_root / self.runs[i]["shard"], mmap_mode="r")
            self._memmaps[i] = mm
        return mm

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ri, ei = self.index[int(idx)]
        r = self.runs[ri]
        ep = np.array(self._shard(ri)[ei], dtype=np.float32)
        return {
            "eeg": torch.from_numpy(ep),  # (T, C)
            "loss_mask": torch.tensor(r["loss_mask"], dtype=torch.bool),
            "onset_step": int(r["onset_step"]),
            "hemisphere": r["hemisphere"],
            "subject": r["subject"],
            "run": r["run"],
            "source": self.source,
            "fs": float(r["fs"]),
        }

    @property
    def window_subjects(self) -> list[str]:
        return [self.runs[i]["subject"] for i, _ in self.index]

    @property
    def subjects(self) -> list[str]:
        return sorted({r["subject"] for r in self.runs})

    def summary(self) -> dict[str, Any]:
        by_hemi: dict[str, int] = {}
        for i, _ in self.index:
            h = self.runs[i]["hemisphere"]
            by_hemi[h] = by_hemi.get(h, 0) + 1
        return {
            "source": self.source,
            "epochs": len(self),
            "runs": len(self.runs),
            "participants": len(self.subjects),
            "by_hemisphere": by_hemi,
            "skipped": self.skipped,
            "fs": self.cfg.fs_target,
            "scored_steps_per_epoch": (
                int(self.runs[0]["n_scored_steps"]) if self.runs else 0
            ),
            "artifact_exclusion_s": self.cfg.artifact_exclusion_s,
        }


# ======================================================================
# the drive
# ======================================================================
class TMSDrive(nn.Module):
    """The latent drive standing in for one TMS pulse.

    **Learned, not computed, and the distinction is the whole point.** A drive
    computed from physics would take the coil's pose, solve for the induced
    E-field, project it onto each parcel's cortical normal
    (:func:`scwbd.intervene.impulse_response.parcel_drive`) and inject the
    result. That path exists in this repository and is not usable here: no
    per-pulse coil pose is distributed with ds004024, so there is no field to
    project. Training a learned profile instead is what lets the model see
    measured stimulation at all; it is also why nothing here validates the
    field-to-response map, and the source card says so in those words.

    What *is* anchored rather than learned:

    * the hemisphere -- recovered from the lateralised MEP, an independent
      channel, per run;
    * the spatial support -- the atlas' somatomotor parcels of that hemisphere.
      The profile within the support is a softmax over learned logits, so the
      drive is a distribution over M1 and cannot leak into occipital cortex to
      fit a late component;
    * the timing -- an impulse at the pulse step, area-normalised by
      :func:`pulse_time_course` so the delivered impulse does not depend on the
      integration step.

    ``log_gain`` is per hemisphere. Both M1s were stimulated at 100% rMT of that
    participant's own threshold and the rMT is not distributed, so a shared gain
    would assert an equality of delivered dose that nothing measured.
    """

    def __init__(self, anat, *, component: str = "rate_e", init_log_gain: float = 0.0) -> None:
        super().__init__()
        self.component = component
        self._hemis = ("left", "right")
        for h in self._hemis:
            idx = torch.from_numpy(motor_parcels(anat, h))
            self.register_buffer(f"support_{h}", idx, persistent=False)
        self.logits = nn.ParameterDict(
            {h: nn.Parameter(torch.zeros(int(getattr(self, f"support_{h}").numel()))) for h in self._hemis}
        )
        self.log_gain = nn.ParameterDict(
            {h: nn.Parameter(torch.tensor(float(init_log_gain))) for h in self._hemis}
        )
        self.n_regions = int(anat.positions.shape[0])

    def profile(self, hemisphere: str) -> Tensor:
        """``(N,)`` non-negative drive per parcel, summing to ``exp(log_gain)``."""
        if hemisphere not in self._hemis:
            raise ValueError(f"unknown hemisphere {hemisphere!r}")
        idx = getattr(self, f"support_{hemisphere}")
        w = torch.softmax(self.logits[hemisphere], dim=0)
        out = torch.zeros(self.n_regions, dtype=w.dtype, device=w.device)
        return out.index_copy(0, idx.to(w.device), w * self.log_gain[hemisphere].exp())

    def forward(
        self,
        model,
        hemispheres: Sequence[str],
        *,
        n_steps: int,
        onset_step: int,
        dt_s: float,
        duration_s: float = 3e-4,
    ) -> Tensor:
        """``(B,T,N,D)`` drive for ``SCWBD.rollout(u=...)``.

        Built per row: a batch mixes runs whose stimulated hemisphere differs,
        and one shared profile would drive the wrong M1 for half of them.
        """
        from scwbd.intervene.impulse_response import build_latent_drive, pulse_time_course

        env = pulse_time_course(n_steps, dt_s=dt_s, onset_step=onset_step, duration_s=duration_s)
        rows = []
        for h in hemispheres:
            rows.append(
                build_latent_drive(
                    model,
                    self.profile(h),
                    n_steps=n_steps,
                    batch=1,
                    component=self.component,
                    time_course=env,
                    dt_s=dt_s,
                )
            )
        return torch.cat(rows, dim=0)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "component": self.component,
            "spatial_support": "atlas somatomotor parcels of the stimulated hemisphere",
            "profile": "softmax over learned logits within the support",
            "computed_from_efield": False,
            "why_not": (
                "ds004024 distributes no per-pulse coil pose, so no E-field can be "
                "computed and the field-to-response map is not exercised"
            ),
        }
        for h in self._hemis:
            p = self.profile(h).detach()
            out[h] = {
                "n_support_parcels": int(getattr(self, f"support_{h}").numel()),
                "gain": float(self.log_gain[h].detach().exp()),
                "peak_parcel": int(torch.argmax(p)),
                "peak_weight": float(p.max()),
            }
        return out
