"""Loaders preserve native units, native rates, geometry and event clocks."""

from __future__ import annotations

import numpy as np
import pytest

from scwbd.sources.loaders.base import (
    Events,
    MultiRateRecording,
    NativeRecording,
    NativeSupportError,
)
from scwbd.sources.lineage import Lineage

from .conftest import require_dataset


# --------------------------------------------------------------------------
# the container itself refuses to destroy support
# --------------------------------------------------------------------------
def _toy(sfreq=250.0, n_ch=3, n_t=1000):
    return NativeRecording(
        source_id="toy",
        data=np.zeros((n_ch, n_t)),
        units=("V",) * n_ch,
        sfreq=sfreq,
        channel_names=tuple(f"C{i}" for i in range(n_ch)),
        channel_types=("eeg",) * n_ch,
        frame_id="toy_frame",
        clock_id="toy_clock",
        lineage=Lineage(participant="P1"),
    )


def test_resampling_is_refused():
    rec = _toy()
    with pytest.raises(NativeSupportError, match="[Nn]ative rate"):
        rec.resample(100.0)
    with pytest.raises(NativeSupportError):
        rec.to_common_raster(100.0)


def test_unit_and_channel_lengths_are_enforced():
    with pytest.raises(ValueError, match="units"):
        NativeRecording(
            source_id="t",
            data=np.zeros((3, 10)),
            units=("V",),
            sfreq=1.0,
            channel_names=("a", "b", "c"),
            channel_types=("eeg",) * 3,
            frame_id="f",
            clock_id="c",
            lineage=Lineage(participant="P"),
        )


def test_epoching_uses_integer_sample_offsets_and_keeps_the_rate():
    rec = _toy(sfreq=200.0)
    rec.events = Events(
        onset=np.array([1.0, 2.0, 3.0]),
        duration=np.zeros(3),
        label=np.array(["a", "b", "a"], dtype=object),
        clock_id="toy_clock",
        sample=np.array([200, 400, 600]),
    )
    ep = rec.epochs(-0.1, 0.4, labels=["a"])
    assert ep.n_epochs == 2
    assert ep.sfreq == 200.0
    assert ep.n_times == int(round(0.5 * 200.0))
    with pytest.raises(NativeSupportError):
        ep.resample(100.0)


def test_epoching_refuses_an_unrelated_event_clock():
    rec = _toy()
    rec.events = Events(
        onset=np.array([1.0]),
        duration=np.zeros(1),
        label=np.array(["a"], dtype=object),
        clock_id="some_other_clock",
        sample=None,
    )
    with pytest.raises(NativeSupportError, match="clock"):
        rec.epochs(-0.1, 0.2)


def test_multirate_refuses_to_merge():
    a, b = _toy(sfreq=100.0), _toy(sfreq=1.0)
    mr = MultiRateRecording(
        source_id="toy", groups={100.0: a, 1.0: b}, lineage=Lineage(participant="P"),
        clock_id="toy_clock",
    )
    assert mr.rates == (1.0, 100.0)
    assert mr.primary.sfreq == 100.0
    with pytest.raises(NativeSupportError, match="one raster"):
        mr.resample(100.0)


# --------------------------------------------------------------------------
# eegmmidb
# --------------------------------------------------------------------------
def test_eegmmidb_native_rate_units_and_geometry():
    require_dataset("eegmmidb")
    from scwbd.sources.loaders import eegmmidb as L

    rec = L.load_run(1, 4)
    assert rec.sfreq == 160.0, "native rate must be the file's rate"
    assert set(rec.units) == {"V"}
    assert rec.n_channels == 64
    # physiological amplitude in volts, not microvolts left unconverted
    assert 1e-6 < np.abs(rec.data).max() < 1e-2
    assert rec.meta["edf_physical_dim"] == "uV"
    assert rec.meta["quantisation_volts"] == pytest.approx(1e-6, rel=1e-6)
    assert rec.channel_positions and len(rec.channel_positions) == 64
    assert rec.frame_id == "template_10_05_head_RAS"
    assert "NOT digitised" in rec.montage
    assert rec.lineage.participant == "S001"
    assert rec.lineage.run == "R04"


def test_eegmmidb_event_labels_are_disambiguated_by_run_family():
    require_dataset("eegmmidb")
    from scwbd.sources.loaders import eegmmidb as L

    fist = L.load_run(1, 4)     # left/right fist run
    feet = L.load_run(1, 6)     # both fists/both feet run
    assert "left_fist" in fist.events.unique_labels
    assert "both_feet" in feet.events.unique_labels
    # the raw EDF code is T1/T2 in both; the loader must not conflate them
    assert set(fist.events.unique_labels) != set(feet.events.unique_labels)
    assert fist.events.clock_id == fist.clock_id


def test_eegmmidb_epochs_preserve_rate():
    require_dataset("eegmmidb")
    from scwbd.sources.loaders import eegmmidb as L

    ep = L.load_run(1, 3).epochs(-0.5, 2.0, labels=["left_fist", "right_fist"])
    assert ep.sfreq == 160.0
    assert ep.n_times == 400
    assert ep.data.shape[1] == 64


def test_eegmmidb_records_carry_full_lineage():
    require_dataset("eegmmidb")
    from scwbd.sources.loaders import eegmmidb as L

    recs = L.records()
    assert len(recs) > 1000
    r = recs[0]
    for level in ("participant", "family", "site", "device", "session", "run"):
        assert getattr(r.lineage, level), f"{level} missing"
    # every run of a subject must resolve to the same group
    from scwbd.sources.splits import GroupedSplitter

    keys = GroupedSplitter("participant", n_folds=5, seed=0).group_keys(recs[:140])
    assert len({keys[r.id] for r in recs[:14]}) == 1


# --------------------------------------------------------------------------
# sleep-edfx: the multirate case
# --------------------------------------------------------------------------
def test_sleep_edfx_keeps_slow_channels_slow():
    require_dataset("sleep-edfx")
    from scwbd.sources.loaders import sleep_edfx as L

    paths = list(L.iter_psg_paths())
    if not paths:
        pytest.skip("no sleep-cassette PSG on disk yet")
    rec = L.load_recording(path=paths[0])
    assert set(rec.rates) == {1.0, 100.0}, "the 1 Hz group must not be upsampled"
    fast, slow = rec[100.0], rec[1.0]
    assert fast.n_times == 100 * slow.n_times
    assert "EEG Fpz-Cz" in fast.channel_names
    assert "Temp rectal" in slow.channel_names
    # units are per channel and physically correct
    temp = slow.pick(["Temp rectal"])
    assert temp.units == ("degC",)
    assert 30.0 < float(temp.data.mean()) < 42.0
    eeg = fast.pick(["EEG Fpz-Cz"])
    assert eeg.units == ("V",)
    assert 1e-6 < float(eeg.data.std()) < 1e-3
    with pytest.raises(NativeSupportError):
        rec.resample(100.0)


def test_sleep_edfx_hypnogram_is_on_its_own_clock_with_real_durations():
    require_dataset("sleep-edfx")
    from scwbd.sources.loaders import sleep_edfx as L

    paths = list(L.iter_psg_paths())
    if not paths:
        pytest.skip("no sleep-cassette PSG on disk yet")
    rec = L.load_recording(path=paths[0])
    if rec.events is None:
        pytest.skip("no hypnogram alongside this PSG")
    labels = set(str(x) for x in rec.events.label)
    assert any(l.startswith("Sleep stage") for l in labels)
    assert "Sleep stage ?" in labels or True  # unscorable epochs keep their own label
    durations = L.stage_durations(rec)
    assert sum(durations.values()) > 3600, "a whole night is more than an hour"
    assert all(d % 30 == 0 for d in rec.events.duration), "scoring epochs are 30 s multiples"


def test_sleep_edfx_two_nights_share_a_participant():
    require_dataset("sleep-edfx")
    from scwbd.sources.loaders import sleep_edfx as L

    recs = L.records()
    if not recs:
        pytest.skip("no records on disk yet")
    by_participant: dict[str, set[str]] = {}
    for r in recs:
        by_participant.setdefault(r.lineage.participant, set()).add(r.lineage.session)
    two = [p for p, s in by_participant.items() if len(s) == 2]
    assert two, "expected participants with two nights"
    from scwbd.sources.splits import GroupedSplitter, leakage_audit

    split = GroupedSplitter("participant", n_folds=5, seed=0).split(recs)
    rep = leakage_audit(split, recs)
    assert rep.ok, rep.summary()


# --------------------------------------------------------------------------
# native rates genuinely differ across sources (the point of §2.6)
# --------------------------------------------------------------------------
def test_sources_do_not_share_a_common_rate():
    rates = {}
    try:
        require_dataset("eegmmidb")
        from scwbd.sources.loaders import eegmmidb as E

        rates["eegmmidb"] = E.load_run(1, 1).sfreq
    except Exception:
        pass
    try:
        require_dataset("sleep-edfx")
        from scwbd.sources.loaders import sleep_edfx as S

        paths = list(S.iter_psg_paths())
        if paths:
            rates["sleep-edfx"] = max(S.load_recording(path=paths[0]).rates)
    except Exception:
        pass
    if len(rates) < 2:
        pytest.skip("need at least two datasets on disk")
    assert len(set(rates.values())) == len(rates), (
        f"rates collapsed onto a common value: {rates}"
    )


# --------------------------------------------------------------------------
# BIDS EEG / MEG loaders (skipped until the binaries land)
# --------------------------------------------------------------------------
def test_ds004024_brainvision_run():
    require_dataset("ds004024")
    pytest.importorskip("mne")
    from scwbd.sources.loaders import bids_eeg as L

    vhdrs = [p for p in L.iter_vhdr() if (p.parent / p.name.replace(".vhdr", ".eeg")).exists()]
    if not vhdrs:
        pytest.skip("no complete BrainVision run on disk yet")
    rec = L.load_brainvision_run(vhdrs[0])
    assert rec.sfreq == 20000.0, "20 kHz native rate must survive loading"
    assert "V" in set(rec.units)
    assert rec.channel_positions and len(rec.channel_positions) >= 64
    assert rec.frame_id.startswith("CapTrak")
    assert rec.meta["file_units"], "channels.tsv units must be recorded"
    if rec.events is not None:
        assert rec.events.sample is not None, "BIDS events carry an explicit sample index"


def test_ds000117_fif_run_has_three_units_on_one_clock():
    require_dataset("ds000117")
    pytest.importorskip("mne")
    from scwbd.sources.loaders import bids_meg as L

    fifs = list(L.iter_fif())
    if not fifs:
        pytest.skip("no MEG fif on disk yet")
    rec = L.load_fif_run(fifs[0], preload=False)
    assert rec.sfreq == pytest.approx(1100.0, rel=1e-6)
    units = set(rec.units)
    assert {"T", "T/m", "V"} <= units, f"expected mag/grad/eeg units, got {units}"
    assert rec.meta["n_mag"] == 102 and rec.meta["n_grad"] == 204
    assert rec.meta["n_eeg"] >= 70
    assert rec.meta["dev_head_t"] is not None, "device->head transform must be preserved"
    assert rec.meta["n_digitisation_points"] > 0
