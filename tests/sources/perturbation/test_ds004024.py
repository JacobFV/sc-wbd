"""ds004024 loader: native support, artefact accounting, and the N=2 bound.

Tests that need the 29 GB signal binaries are skipped when the data root is
absent, so the suite still runs on a metadata-only checkout.  The skip is
deliberately narrow: everything that can be tested without the binaries
(refusals, the hemisphere discriminator, the split bound) is tested
unconditionally against synthetic input.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from scwbd.sources.lineage import Lineage, LineageError, Record
from scwbd.sources.perturbation import ds004024 as ds
from scwbd.sources.perturbation.splits import (
    MIN_PARTICIPANTS_FOR_VARIANCE,
    participant_split,
    split_bound,
)

DATA_ROOT = "/data/scwbd/ds004024/1.0.0"
has_data = pytest.mark.skipif(
    not os.path.isdir(DATA_ROOT), reason="ds004024 signal binaries not on this host"
)


# -- the hemisphere discriminator must read differently on different input --
def _emg(sfreq=20000.0, tmin=-0.05, n=40, right_gain=1.0, left_gain=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n_t = int(round((0.06 - tmin) * sfreq))
    data = rng.normal(0, 1e-6, (n, 2, n_t))
    a = int(round((0.020 - tmin) * sfreq))
    b = int(round((0.030 - tmin) * sfreq))
    data[:, 0, a:b] += right_gain * 1e-4 * np.sin(np.linspace(0, np.pi, b - a))
    data[:, 1, a:b] += left_gain * 1e-4 * np.sin(np.linspace(0, np.pi, b - a))
    return data


def test_hemisphere_discriminator_reads_left_right_and_undecided():
    names = ["EMG Right", "EMG Left"]
    left = ds.derive_stimulated_hemisphere(
        _emg(right_gain=10.0, left_gain=0.1), names, 20000.0, -0.05
    )
    assert left["hemisphere"] == "left"  # right FDI responds => left M1

    right = ds.derive_stimulated_hemisphere(
        _emg(right_gain=0.1, left_gain=10.0), names, 20000.0, -0.05
    )
    assert right["hemisphere"] == "right"

    # symmetric response is NOT resolved to the sign of a noisy mean
    tie = ds.derive_stimulated_hemisphere(
        _emg(right_gain=1.0, left_gain=1.0), names, 20000.0, -0.05
    )
    assert tie["hemisphere"] is None
    assert "below stated floor" in tie["reason"]

    # every decision carries the statistic that produced it
    assert left["statistic"].startswith("paired per-trial")
    assert left["n_trials"] == 40


def test_hemisphere_discriminator_reports_missing_channels():
    out = ds.derive_stimulated_hemisphere(_emg(), ["EMG A", "EMG B"], 20000.0, -0.05)
    assert out["hemisphere"] is None and "absent" in out["reason"]


# -- native support is refused, loudly ---------------------------------
@has_data
def test_epochs_refuse_to_resample():
    ep = ds.load_run(DATA_ROOT, "sub-CON001", "01", tmin=-0.01, tmax=0.02)
    with pytest.raises(NotImplementedError, match="not resampleable"):
        ep.resample(1000.0)


@has_data
def test_native_rate_units_positions_and_event_clock_are_preserved():
    ep = ds.load_run(DATA_ROOT, "sub-CON001", "01", tmin=-0.01, tmax=0.02)
    assert ep.sfreq == ds.NATIVE_SFREQ_HZ == 20000.0
    assert ep.frame_id == "CapTrak_m"
    assert ep.clock_id == ds.CLOCK_ID
    # 64 EEG + 2 EOG + 2 EMG + 1 ECG
    counts = {t: ep.channel_types.count(t) for t in set(ep.channel_types)}
    assert counts == {"eeg": 64, "eog": 2, "emg": 2, "ecg": 1}
    # positions exist for the 64 scalp electrodes and are absent -- not zero --
    # for the peripheral channels
    assert len(ep.channel_positions) == 64
    assert set(ep.meta["positions_missing_for"]) == {
        "EMG Right", "EMG Left", "ECG", "VEOG", "HEOG",
    }
    assert all(v is not None for v in ep.channel_positions.values())
    # the stimulation clock survives epoching: absolute onsets, strictly increasing
    assert ep.onset_sample.dtype == np.int64
    assert np.all(np.diff(ep.onset_sample) > 0)
    assert ep.control_graph.exposures.clock_id == ds.CLOCK_ID


@has_data
def test_exposure_duration_is_nan_not_zero():
    ep = ds.load_run(DATA_ROOT, "sub-CON001", "01", tmin=-0.01, tmax=0.02)
    d = ep.control_graph.exposures.duration_s
    assert np.all(np.isnan(d)), "unknown pulse width must not be recorded as 0"


@has_data
def test_artifact_window_is_excluded_and_the_measurement_is_recorded():
    ep = ds.load_run(DATA_ROOT, "sub-CON001", "01", tmin=-0.01, tmax=0.02)
    w = next(w for w in ep.control_graph.excluded_windows if w.name == "tms_artifact")
    assert w.t_start_s == 0.0 and w.t_stop_s == 0.010
    # the measured saturation is recorded alongside the declared window, and is
    # much shorter than it -- the loader must not claim the tighter bound
    assert w.measured["last_rail_contact_s_max"] < w.t_stop_s
    assert w.measured["rail_abs_V_max"] > 0.1  # amplifier is genuinely railed
    mask = ep.exclusion_mask()
    t = ep.times()
    assert mask[(t >= 0) & (t < 0.010)].all()
    assert not mask[t < 0].any()


@has_data
def test_unfetched_subject_raises_and_names_it_as_our_choice():
    with pytest.raises(ds.DatasetNotAvailable) as exc:
        ds.load_run(DATA_ROOT, "sub-CON008", "01")
    msg = str(exc.value)
    assert "all 13 participants" in msg  # upstream has it
    assert "fetch choice" in msg  # we did not fetch it


@has_data
def test_available_subjects_counts_binaries_not_directories():
    subs = ds.available_subjects(DATA_ROOT)
    # 13 participant directories exist; only 2 hold signal binaries
    n_dirs = len([d for d in os.listdir(DATA_ROOT) if d.startswith("sub-")])
    assert n_dirs == 13
    assert subs == ("sub-CON001", "sub-CON006")


@has_data
def test_control_graph_refuses_dose_and_state_dependence_on_this_snapshot():
    graphs = [
        ds.load_run(DATA_ROOT, "sub-CON001", r, tmin=-0.01, tmax=0.06).control_graph
        for r in ds.FETCHED_RUNS
    ]
    from scwbd.sources.perturbation.control_graph import ControlGraph

    design = ControlGraph.combine(graphs, record_id="sub-CON001/design")
    rep = design.recovery_report()
    assert rep["direction"]["supported"] is True
    assert rep["delay"]["supported"] is True
    assert rep["gain"]["supported"] is False
    assert rep["dose"]["supported"] is False
    assert rep["state_dependence"]["supported"] is False
    # the unknowns are enumerated, not merely absent
    assert set(design.unknowns) >= {"coil_pose", "realised_dose"}


# -- splits ------------------------------------------------------------
def _records(participants, runs=("01", "02")):
    return [
        Record(
            id=f"{p}/ses-async14ms/spTMS_run-{r}",
            source_id="ds004024",
            lineage=Lineage(
                participant=p, family=f"singleton:{p}", session="ses-async14ms",
                run=f"spTMS_run-{r}",
            ),
        )
        for p in participants
        for r in runs
    ]


def test_participant_split_keeps_a_persons_runs_together():
    recs = _records(["sub-A", "sub-B"], runs=("01", "02", "03"))
    sp = participant_split(recs)
    assert len(sp.folds) == 2
    for fold in sp.folds:
        subs = {i.split("/")[0] for i in fold.test_ids}
        assert len(subs) == 1, "a participant's runs straddled a fold (R10)"
        assert not (set(fold.train_ids) & set(fold.test_ids))


def test_single_participant_split_is_refused_not_downgraded():
    with pytest.raises(LineageError, match="not evaluable"):
        participant_split(_records(["sub-A"]))


def test_split_bound_states_that_n2_cannot_estimate_variance():
    two = split_bound(_records(["sub-A", "sub-B"]))
    assert two["participant_holdout_constructible"] is True
    assert two["between_participant_variance_estimable"] is False
    assert two["pooling_permitted"] is False
    assert "never pool" in two["bound"]

    # the same function reads differently once enough participants exist
    many = split_bound(_records([f"sub-{i}" for i in range(MIN_PARTICIPANTS_FOR_VARIANCE)]))
    assert many["between_participant_variance_estimable"] is True


@has_data
def test_real_records_are_participant_grouped_and_leakage_free():
    from scwbd.sources.perturbation.splits import audit

    recs = ds.records(DATA_ROOT)
    assert len(recs) == 12  # 2 participants x 6 runs
    sp = participant_split(recs)
    rep = audit(sp, recs)
    assert rep["ok"], rep["violations"]
    assert rep["bound"]["n_participants"] == 2
    assert rep["bound"]["between_participant_variance_estimable"] is False
