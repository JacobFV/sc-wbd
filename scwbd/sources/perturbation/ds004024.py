"""Loader for the ds004024 spTMS runs: TMS-EEG epochs plus their control graph.

Native support is preserved exactly (``ARCHITECTURE.md`` / thesis §2.6): the
20 kHz sampling rate is never changed, no filter is applied, no artefact
rejection is performed, units stay in the SI volts MNE converts the BrainVision
microvolts to, and every epoch carries the absolute onset **sample** of its
pulse on the amplifier clock so the stimulation event clock survives epoching.

What this module will *not* do, and why it matters
--------------------------------------------------
The provider's ``dataset_description.json`` describes a pre/post design
(spTMS-Before / spTMS-After10 / spTMS-After60, crossed with left and right M1).
That design was run.  But **no per-run label for it is distributed**:

* the per-run ``*_eeg.json`` sidecars carry ``TaskName: "spTMS"`` and nothing
  that distinguishes one run from another;
* ``*_scans.tsv`` carries an ``acq_time`` column whose value is ``n/a`` for
  every row;
* the BrainVision ``.vmrk`` files were rewritten by ``pybv 0.6.0``, which
  emitted markers with an empty date field, so ``raw.info["meas_date"]`` is
  ``None`` and the wall clock is gone;
* ``events.tsv`` contains only ``Stimulus/A`` / ``Out/A`` markers, identical in
  structure across all six runs.

So the timepoint factor is :attr:`Provenance.DECLARED` — real as a study fact,
absent as a record label — and this loader refuses to synthesise it from the
run index.  It would be easy to assume ``run-01..06`` maps onto
``(Before-L, Before-R, After10-L, After10-R, After60-L, After60-R)``.  That
assumption is *not* supported here for two reasons.  First, it is an assumption
about file naming, not a measurement.  Second, it is measurably imperfect: the
hemisphere factor, which *is* recoverable from the signal, alternates
left/right exactly as that convention predicts in 11 of 12 loaded runs and
contradicts it in one (``sub-CON006`` ``run-06``; see
:func:`derive_stimulated_hemisphere`).  A convention that is wrong once in
twelve cannot be relied on to label a contrast that nothing else can check.

The hemisphere factor, by contrast, *is* recovered — from an independent
channel.  TMS of one M1 evokes a motor evoked potential in the contralateral
first dorsal interosseous, and the release ships ``EMG Left`` / ``EMG Right``.
The recovered level therefore carries :attr:`Provenance.DERIVED` together with
the paired per-trial statistic that supports it, so a consumer can tell it
apart from a label the provider actually shipped.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..lineage import Lineage, Record
from .control_graph import (
    ControlGraph,
    ControlVariable,
    ExcludedWindow,
    ExposureInterval,
    Provenance,
)

SOURCE_ID = "ds004024"
DATASET_VERSION = "1.0.0"
CLOCK_ID = "ds004024.brainvision_amp"
FRAME_ID = "CapTrak_m"
NATIVE_SFREQ_HZ = 20000.0

#: Session whose signal binaries were fetched.  Others are metadata-only.
FETCHED_SESSION = "ses-async14ms"
#: spTMS runs at 100% rMT.  Runs 07-12 (110% rMT) were not fetched.
FETCHED_RUNS: tuple[str, ...] = ("01", "02", "03", "04", "05", "06")

#: Declared spTMS intensity for runs 01-06, per ``dataset_description.json``.
#: Relative to each participant's resting motor threshold; the rMT itself, in
#: stimulator output units, is not distributed, so no absolute dose exists.
DECLARED_INTENSITY_PCT_RMT = 100.0


class DatasetNotAvailable(FileNotFoundError):
    """Raised when a requested run has no signal binary on disk.

    Distinguishes "we did not fetch this" from "upstream does not have it";
    ``ds004024.yaml`` records that upstream ships binaries for all 13
    participants and that the 2-participant subset is our fetch choice.
    """


def _read_tsv(path: str) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, encoding="utf-8-sig") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        rows = [dict(zip(header, l.rstrip("\n").split("\t"))) for l in fh]
    return header, rows


@dataclass
class TMSEpochs:
    """TMS-EEG epochs at native rate, with the stimulation clock preserved.

    ``data`` is ``(n_trials, n_channels, n_times)`` in ``units`` (SI volts for
    the electrophysiological channels).  It is never resampled and never
    filtered.  ``onset_sample`` gives each trial's pulse onset as an absolute
    index on :data:`CLOCK_ID`, so a consumer can always recover where in the
    continuous recording an epoch came from.
    """

    source_id: str
    record_id: str
    data: np.ndarray
    units: tuple[str, ...]
    sfreq: float
    channel_names: tuple[str, ...]
    channel_types: tuple[str, ...]
    channel_positions: Mapping[str, tuple[float, float, float]]
    frame_id: str
    clock_id: str
    tmin: float
    onset_sample: np.ndarray
    lineage: Lineage
    control_graph: ControlGraph
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError("data must be (n_trials, n_channels, n_times)")
        if self.data.shape[1] != len(self.channel_names):
            raise ValueError("channel_names length does not match data")
        if len(self.onset_sample) != self.data.shape[0]:
            raise ValueError("onset_sample length does not match n_trials")

    @property
    def n_trials(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[1])

    @property
    def n_times(self) -> int:
        return int(self.data.shape[2])

    def times(self) -> np.ndarray:
        """Epoch time axis in seconds relative to pulse onset."""
        return self.tmin + np.arange(self.n_times) / self.sfreq

    def resample(self, sfreq: float) -> "TMSEpochs":  # noqa: ARG002
        raise NotImplementedError(
            "ds004024 epochs are not resampleable: the 20 kHz rate is what makes "
            "the sub-millisecond TMS artefact separable from the response, and "
            "the pulse onset is exact on the amplifier sample clock. Resampling "
            "would destroy both. Work at the native rate."
        )

    def picks(self, types: Sequence[str]) -> "TMSEpochs":
        idx = [i for i, t in enumerate(self.channel_types) if t in types]
        return TMSEpochs(
            source_id=self.source_id,
            record_id=self.record_id,
            data=self.data[:, idx],
            units=tuple(self.units[i] for i in idx),
            sfreq=self.sfreq,
            channel_names=tuple(self.channel_names[i] for i in idx),
            channel_types=tuple(self.channel_types[i] for i in idx),
            channel_positions={
                k: v
                for k, v in self.channel_positions.items()
                if k in {self.channel_names[i] for i in idx}
            },
            frame_id=self.frame_id,
            clock_id=self.clock_id,
            tmin=self.tmin,
            onset_sample=self.onset_sample,
            lineage=self.lineage,
            control_graph=self.control_graph,
            meta=dict(self.meta),
        )

    def exclusion_mask(self) -> np.ndarray:
        """Boolean mask over the time axis: True where samples are excluded.

        Built from :attr:`ControlGraph.excluded_windows` so the artefact
        decision is visible to whoever consumes the epochs, rather than applied
        silently upstream of them.
        """
        t = self.times()
        mask = np.zeros(t.shape, dtype=bool)
        for w in self.control_graph.excluded_windows:
            mask |= (t >= w.t_start_s) & (t < w.t_stop_s)
        return mask


def run_paths(root: str, sub: str, run: str, session: str = FETCHED_SESSION) -> dict[str, str]:
    eegdir = os.path.join(root, sub, session, "eeg")
    stem = f"{sub}_{session}_task-spTMS_run-{run}"
    return {
        "vhdr": os.path.join(eegdir, f"{stem}_eeg.vhdr"),
        "eeg": os.path.join(eegdir, f"{stem}_eeg.eeg"),
        "json": os.path.join(eegdir, f"{stem}_eeg.json"),
        "events": os.path.join(eegdir, f"{stem}_events.tsv"),
        "channels": os.path.join(eegdir, f"{stem}_channels.tsv"),
        "electrodes": os.path.join(eegdir, f"{sub}_{session}_electrodes.tsv"),
        "coordsystem": os.path.join(eegdir, f"{sub}_{session}_coordsystem.json"),
    }


def available_subjects(root: str, session: str = FETCHED_SESSION) -> tuple[str, ...]:
    """Subjects with a *loadable* spTMS signal binary, verified by file presence.

    Directory listing is not enough on this source: ``reports/known_issues.md``
    ISSUE-003 records that interrupted ``aws s3 sync`` runs leave multipart temp
    files that make a metadata-only subject look populated.  This checks for the
    ``.eeg`` binary specifically and requires it to be non-empty.
    """
    if not os.path.isdir(root):
        return ()
    subs = []
    for sub in sorted(os.listdir(root)):
        if not sub.startswith("sub-"):
            continue
        ok = all(
            os.path.exists(run_paths(root, sub, r, session)["eeg"])
            and os.path.getsize(run_paths(root, sub, r, session)["eeg"]) > 0
            for r in FETCHED_RUNS
        )
        if ok:
            subs.append(sub)
    return tuple(subs)


def derive_stimulated_hemisphere(
    emg: np.ndarray,
    emg_names: Sequence[str],
    sfreq: float,
    tmin: float,
    *,
    mep_window_s: tuple[float, float] = (0.015, 0.045),
    min_abs_mean_log_ratio: float = 0.25,
    min_abs_t: float = 3.0,
) -> dict[str, Any]:
    """Recover which M1 was stimulated from the lateralised MEP.

    TMS of the left M1 evokes an MEP in the *right* FDI and vice versa, so the
    paired per-trial log ratio ``log(p2p_right / p2p_left)`` is positive when
    the left hemisphere was stimulated.

    Returns a dict with the decision and the statistic behind it.  When the
    laterality is weak the decision is ``None`` rather than the sign of a noisy
    mean: a factor that cannot be recovered is reported as unrecovered.
    ``min_abs_mean_log_ratio`` is a *stated* effect-size floor, not a p-value
    threshold — with 80 paired trials a negligible ratio is still "significant".
    """
    idx = {n: i for i, n in enumerate(emg_names)}
    if "EMG Right" not in idx or "EMG Left" not in idx:
        return {"hemisphere": None, "reason": "EMG Left/Right channels absent"}
    a0 = int(round((mep_window_s[0] - tmin) * sfreq))
    a1 = int(round((mep_window_s[1] - tmin) * sfreq))
    right = emg[:, idx["EMG Right"], a0:a1]
    left = emg[:, idx["EMG Left"], a0:a1]
    p2p_r = right.max(axis=1) - right.min(axis=1)
    p2p_l = left.max(axis=1) - left.min(axis=1)
    good = (p2p_r > 0) & (p2p_l > 0)
    lr = np.log(p2p_r[good] / p2p_l[good])
    n = int(len(lr))
    if n < 2:
        return {"hemisphere": None, "reason": "fewer than 2 usable trials"}
    mean = float(lr.mean())
    sd = float(lr.std(ddof=1))
    t = float(mean / (sd / np.sqrt(n))) if sd > 0 else float("inf")
    decided = abs(mean) >= min_abs_mean_log_ratio and abs(t) >= min_abs_t
    hemi = None
    if decided:
        hemi = "left" if mean > 0 else "right"
    return {
        "hemisphere": hemi,
        "statistic": "paired per-trial log(MEP_p2p_right / MEP_p2p_left)",
        "mean_log_ratio": mean,
        "sd_log_ratio": sd,
        "t": t,
        "n_trials": n,
        "frac_trials_positive": float((lr > 0).mean()),
        "mep_window_s": list(mep_window_s),
        "thresholds": {
            "min_abs_mean_log_ratio": min_abs_mean_log_ratio,
            "min_abs_t": min_abs_t,
        },
        "reason": (
            "" if decided
            else f"laterality below stated floor (|mean|={abs(mean):.3f} < "
                 f"{min_abs_mean_log_ratio}, |t|={abs(t):.1f})"
        ),
    }


def measure_saturation(
    eeg: np.ndarray, sfreq: float, tmin: float, *, rel_tol: float = 1e-3
) -> dict[str, Any]:
    """Measure how long the amplifier is pinned at its rail after each pulse.

    The rail is not declared anywhere in the release (the samples are IEEE
    float32, so the amplifier range is not recoverable from the file format),
    so it is measured: the per-trial maximum absolute value, and the last
    post-onset sample within ``rel_tol`` of it.
    """
    n_tr = eeg.shape[0]
    zero = int(round(-tmin * sfreq))
    rails, last = [], []
    for i in range(n_tr):
        seg = eeg[i]
        mx = float(np.abs(seg).max())
        rails.append(mx)
        at_rail = np.abs(np.abs(seg) - mx) < (rel_tol * mx)
        post = at_rail[:, zero:].any(axis=0)
        w = np.where(post)[0]
        last.append(float(w[-1] / sfreq) if len(w) else 0.0)
    return {
        "rail_abs_V_max": float(np.max(rails)),
        "rail_abs_V_min": float(np.min(rails)),
        "last_rail_contact_s_median": float(np.median(last)),
        "last_rail_contact_s_max": float(np.max(last)),
        "n_trials": n_tr,
        "rel_tol": rel_tol,
    }


def load_run(
    root: str,
    sub: str,
    run: str,
    *,
    session: str = FETCHED_SESSION,
    tmin: float = -0.100,
    tmax: float = 0.300,
    artifact_exclusion_s: float = 0.010,
) -> TMSEpochs:
    """Load one spTMS run as epochs with its control graph.

    ``artifact_exclusion_s`` defaults to the 10 ms the source card's
    ``ledger.validity_domain.post_pulse_window_s`` declares artefact-dominated.
    The *measured* hard-saturation extent is recorded alongside it rather than
    replacing it: saturation ending at ~0.3 ms does not mean the decay and
    recharge artefacts are over at 0.3 ms, and this loader does not claim a
    bound it did not establish.
    """
    import mne  # imported lazily: the rest of this module is importable without it

    p = run_paths(root, sub, run, session)
    if not os.path.exists(p["eeg"]):
        raise DatasetNotAvailable(
            f"{sub} {session} spTMS run-{run}: no signal binary at {p['eeg']}. "
            "Upstream ds004024 ships binaries for all 13 participants; this "
            "snapshot pinned a 2-participant subset (see ds004024.yaml "
            "identity.subset_note). Absent by our fetch choice, not upstream."
        )
    raw = mne.io.read_raw_brainvision(p["vhdr"], preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    if sfreq != NATIVE_SFREQ_HZ:
        raise ValueError(f"expected {NATIVE_SFREQ_HZ} Hz, header says {sfreq}")

    # channel types come from channels.tsv: the BrainVision header does not
    # carry them, so MNE types all 69 channels as EEG without this.
    _, chan_rows = _read_tsv(p["channels"])
    type_of = {r["name"]: r["type"].lower() for r in chan_rows}
    unit_of = {r["name"]: r["units"] for r in chan_rows}
    names = tuple(raw.ch_names)
    types = tuple(type_of.get(n, "misc") for n in names)

    # electrode positions, native CapTrak metres; non-numeric rows stay absent
    positions: dict[str, tuple[float, float, float]] = {}
    _, e_rows = _read_tsv(p["electrodes"])
    for r in e_rows:
        try:
            positions[r["name"]] = (float(r["x"]), float(r["y"]), float(r["z"]))
        except (ValueError, KeyError):
            continue
    with open(p["coordsystem"]) as fh:
        coordsystem = json.load(fh)

    _, ev_rows = _read_tsv(p["events"])
    onsets = np.array(
        [int(r["sample"]) for r in ev_rows if r.get("trial_type") == "Stimulus/A"],
        dtype=np.int64,
    )
    if onsets.size == 0:
        raise ValueError(f"{p['events']}: no Stimulus/A markers")

    a = int(round(tmin * sfreq))
    b = int(round(tmax * sfreq))
    n_times = b - a
    keep = [s for s in onsets if s + a >= 0 and s + b < raw.n_times]
    data = np.empty((len(keep), len(names), n_times), dtype=np.float64)
    for i, s in enumerate(keep):
        data[i] = raw.get_data(start=s + a, stop=s + b)
    kept = np.array(keep, dtype=np.int64)

    emg_idx = [i for i, t in enumerate(types) if t == "emg"]
    eeg_idx = [i for i, t in enumerate(types) if t == "eeg"]
    hemi = derive_stimulated_hemisphere(
        data[:, emg_idx], [names[i] for i in emg_idx], sfreq, tmin
    )
    sat = measure_saturation(data[:, eeg_idx], sfreq, tmin)

    lineage = Lineage(
        participant=sub,
        family=f"singleton:{sub}",
        site="openneuro:ds004024:shirley_ryan_abilitylab",
        device="BrainProducts_64ch_DC + MRI-navigated TMS",
        session=session,
        run=f"spTMS_run-{run}",
        extra={"task": "spTMS", "dataset_version": DATASET_VERSION},
    )
    record_id = f"{sub}/{session}/spTMS_run-{run}"

    manipulated = (
        ControlVariable(
            name="pulse_onset",
            provenance=Provenance.RECORDED,
            units="sample",
            levels_present=tuple(range(len(kept))),
            note=(
                "events.tsv carries an explicit integer 'sample' column on the "
                "amplifier clock, so pulse onsets map to samples exactly. The TMS "
                "device's own trigger clock relative to the amplifier is not "
                "documented, so residual trigger jitter is unknown and bounds any "
                "latency claim finer than one sample (50 us)."
            ),
        ),
        ControlVariable(
            name="stimulated_hemisphere",
            provenance=(
                Provenance.DERIVED if hemi["hemisphere"] else Provenance.UNKNOWN
            ),
            value=hemi["hemisphere"],
            levels_present=((hemi["hemisphere"],) if hemi["hemisphere"] else ()),
            method=(
                "lateralised MEP: paired per-trial log ratio of first dorsal "
                "interosseous MEP peak-to-peak amplitude, right EMG over left EMG, "
                "in a 15-45 ms post-pulse window; positive => left M1 stimulated"
            ),
            evidence=hemi,
            note=(
                "Recovered from an independent channel, not from a provider label: "
                "no run-level hemisphere label is distributed. Where the laterality "
                "falls below the stated effect-size floor the level is UNKNOWN "
                "rather than the sign of a noisy mean."
            ),
        ),
        ControlVariable(
            name="block_timepoint",
            provenance=Provenance.DECLARED,
            value=None,
            levels_present=(),
            note=(
                "dataset_description.json states the study ran spTMS-Before, "
                "spTMS-After10 and spTMS-After60 blocks. No per-run label for this "
                "factor is distributed: the eeg.json sidecars carry only "
                "TaskName='spTMS'; scans.tsv acq_time is 'n/a' for every row; pybv "
                "0.6.0 wrote the .vmrk markers with an empty date field so "
                "meas_date is None; and events.tsv is structurally identical across "
                "runs. The factor is therefore DECLARED, not RECORDED, and is not "
                "inferred from the run index."
            ),
        ),
        ControlVariable(
            name="intensity_pct_rmt",
            provenance=Provenance.DECLARED,
            value=DECLARED_INTENSITY_PCT_RMT,
            units="percent_of_resting_motor_threshold",
            levels_present=(DECLARED_INTENSITY_PCT_RMT,),
            note=(
                "Runs 01-06 are declared 100% rMT in dataset_description.json. The "
                "110% rMT runs 07-12 exist upstream but were not fetched, so exactly "
                "one intensity level is present and no intensity slope is estimable."
            ),
        ),
        ControlVariable(
            name="realised_dose",
            provenance=Provenance.UNKNOWN,
            note=(
                "Intensity is relative to each participant's resting motor "
                "threshold and the per-participant rMT in stimulator output units "
                "is not distributed, so the absolute dose in A/m^2 or V/m at the "
                "cortex is unknown. Not imputed."
            ),
        ),
        ControlVariable(
            name="coil_pose",
            provenance=Provenance.UNKNOWN,
            note=(
                "The coil was MRI-navigated during acquisition but no per-pulse "
                "coil position/orientation log is distributed. An intervention with "
                "an unknown pose is still an intervention; it constrains different "
                "claims than one with a measured pose, and it disables the E-field "
                "operator path outright."
            ),
        ),
        ControlVariable(
            name="pulse_width",
            provenance=Provenance.UNKNOWN,
            note=(
                "Biphasic single pulse; the waveform width is not distributed "
                "per record, so the exposure interval carries nan duration rather "
                "than a zero-width event."
            ),
        ),
    )

    exposures = ExposureInterval(
        onset_sample=kept,
        onset_s=kept / sfreq,
        duration_s=np.full(len(kept), np.nan),
        clock_id=CLOCK_ID,
        duration_provenance=Provenance.UNKNOWN,
        note=(
            "Onsets are exact on the amplifier sample clock. The physical "
            "exposure width is not distributed, so duration is nan: a zero-width "
            "exposure would assert the input was never on."
        ),
    )

    excluded = (
        ExcludedWindow(
            name="tms_artifact",
            t_start_s=0.0,
            t_stop_s=artifact_exclusion_s,
            reason=(
                "TMS-evoked artefact: amplifier saturation, decay and recharge. "
                "The saturation extent is measured per run (see 'measured'); the "
                "exclusion window is the source card's declared artefact-dominated "
                "window, which is wider than measured saturation because this "
                "loader did not establish a bound on the slower decay."
            ),
            measured=sat,
        ),
    )

    cg = ControlGraph(
        source_id=SOURCE_ID,
        record_id=record_id,
        lineage=lineage,
        manipulated=manipulated,
        exposures=exposures,
        observed_only=("eeg", "eog", "emg", "ecg"),
        excluded_windows=excluded,
        target_site=(
            f"M1_{hemi['hemisphere']}" if hemi["hemisphere"] else None
        ),
        target_site_provenance=(
            Provenance.DERIVED if hemi["hemisphere"] else Provenance.UNKNOWN
        ),
        note=(
            "Single-pulse TMS over the first dorsal interosseous representation "
            "of one primary motor cortex, MRI-navigated. Offline analysis of "
            "already-collected, consented, published data (CC0; Northwestern IRB "
            "STU00204239; NCT03723434). Supports target hypotheses only; per "
            "Appendix D, offline reconstruction licenses no efficacy or decision "
            "claim."
        ),
    )

    del raw
    return TMSEpochs(
        source_id=SOURCE_ID,
        record_id=record_id,
        data=data,
        units=tuple("V" if type_of.get(n, "") != "misc" else unit_of.get(n, "n/a") for n in names),
        sfreq=sfreq,
        channel_names=names,
        channel_types=types,
        channel_positions=positions,
        frame_id=FRAME_ID,
        clock_id=CLOCK_ID,
        tmin=tmin,
        onset_sample=kept,
        lineage=lineage,
        control_graph=cg,
        meta={
            "session": session,
            "run": run,
            "coordsystem": coordsystem,
            "n_pulses_in_events_tsv": int(onsets.size),
            "n_epochs_kept": int(len(kept)),
            "dropped_for_edge": int(onsets.size - len(kept)),
            "positions_missing_for": tuple(n for n in names if n not in positions),
        },
    )


def records(root: str, session: str = FETCHED_SESSION) -> tuple[Record, ...]:
    """Split records for every loadable spTMS run, grouped by participant.

    One :class:`Record` per run.  ``GroupedSplitter`` groups on ``family``
    (declared ``singleton:<participant>`` so the grouping level is uniform), so
    all six runs of a participant land in the same fold (R10).
    """
    out = []
    for sub in available_subjects(root, session):
        for run in FETCHED_RUNS:
            p = run_paths(root, sub, run, session)
            out.append(
                Record(
                    id=f"{sub}/{session}/spTMS_run-{run}",
                    source_id=SOURCE_ID,
                    lineage=Lineage(
                        participant=sub,
                        family=f"singleton:{sub}",
                        site="openneuro:ds004024:shirley_ryan_abilitylab",
                        device="BrainProducts_64ch_DC + MRI-navigated TMS",
                        session=session,
                        run=f"spTMS_run-{run}",
                        extra={"task": "spTMS"},
                    ),
                    path=p["eeg"],
                    n_bytes=os.path.getsize(p["eeg"]) if os.path.exists(p["eeg"]) else None,
                )
            )
    return tuple(out)
