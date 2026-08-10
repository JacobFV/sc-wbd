"""A source's montage gets its own operator, not a padded row in someone else's.

``sleepedf_real`` was disabled for run 2 with the reason recorded on the card:
"two bipolar derivations cannot constrain a 64-channel observation head". That
was true of the thing being attempted -- forcing Sleep-EDF through the eegmmidb
head -- and it is not a property of the data. A bipolar channel measures
``V(anode) - V(cathode)``; the forward operator is linear in the source
amplitudes; so the correct gain row is exactly ``L[anode] - L[cathode]``.

These tests assert that identity holds numerically, that the rank of the
resulting operator is the number of derivations (so no claim finer than a
2-dimensional projection can be read off it), and that the two ways of getting
an operator wrong both refuse rather than producing a plausible number:

* a derivation label that does not name two electrodes;
* a montage that declares no ``kind``, where guessing ``monopolar`` would
  observe a difference-of-two-electrodes through a single-electrode row.
"""

from __future__ import annotations

import pytest
import torch

from scwbd.foundation.anatomy import load_anatomy
from scwbd.foundation.config import ModelConfig
from scwbd.foundation.heads import (
    build_bipolar_lead_field,
    build_lead_field,
    parse_bipolar_derivations,
)
from scwbd.foundation.model import SCWBD
from scwbd.foundation.realdata import SLEEP_EDF_EEG_CHANNELS

SLEEP_DERIVATIONS = (("Fpz", "Cz"), ("Pz", "Oz"))


@pytest.fixture(scope="module")
def anat():
    return load_anatomy(device="cpu")


def test_sleep_edf_labels_parse_to_electrode_pairs() -> None:
    assert parse_bipolar_derivations(SLEEP_EDF_EEG_CHANNELS) == SLEEP_DERIVATIONS


@pytest.mark.parametrize("bad", ["EEG Cz", "Fpz-Cz-Pz", "EEG "])
def test_a_label_that_is_not_a_derivation_is_refused(bad: str) -> None:
    """Not silently passed through as monopolar -- that is the wrong operator."""
    with pytest.raises(ValueError, match="bipolar derivation"):
        parse_bipolar_derivations([bad])


def test_a_bipolar_row_is_exactly_the_difference_of_two_monopolar_rows(anat) -> None:
    """The identity the adapter rests on, checked against the same solve."""
    bip = build_bipolar_lead_field(anat, derivations=SLEEP_DERIVATIONS)
    mono = build_lead_field(anat, channel_names=["Fpz", "Cz", "Pz", "Oz"])
    assert torch.allclose(bip.matrix[0], mono.matrix[0] - mono.matrix[1])
    assert torch.allclose(bip.matrix[1], mono.matrix[2] - mono.matrix[3])
    assert bip.matrix_vec is not None, (
        "the free-orientation gain must survive the derivation: a 3-vector "
        "regional moment observed through the scalar contraction is back in the "
        "scalar regime, which is the 2.6x this project measured"
    )
    assert torch.allclose(bip.matrix_vec[0], mono.matrix_vec[0] - mono.matrix_vec[1])


def test_the_operator_rank_is_the_number_of_derivations(anat) -> None:
    """Two derivations constrain a 2-dimensional projection and no more.

    This is the honest version of the card's original objection, and it is why
    the note on the lead field states the rank rather than leaving a reader to
    infer that 414 parcels are identified from two channels.
    """
    bip = build_bipolar_lead_field(anat, derivations=SLEEP_DERIVATIONS)
    assert bip.matrix.shape == (2, anat.positions.shape[0])
    assert int(torch.linalg.matrix_rank(bip.matrix.float())) == 2
    assert "Rank is at most 2" in bip.note


def test_an_electrode_with_no_forward_row_is_refused(anat) -> None:
    with pytest.raises(ValueError, match="no row for electrode"):
        build_bipolar_lead_field(anat, derivations=[("Fpz", "NotAnElectrode")])


def test_the_lead_field_is_built_on_real_electrode_geometry(anat) -> None:
    """ISSUE-006: the real-geometry path must be the one that runs.

    ``_montage_positions`` raised on the first electrode it found -- ``ndarray or
    ...`` evaluates an array's truth value -- so ``build_lead_field`` fell into
    its Fibonacci-spiral fallback for every montage this project ever built,
    while the note on the result said "real 10-10 montage positions".

    Asserted two ways, because either alone is weak: the positions come back at
    all, and the operator says which geometry produced it. The second is what a
    reader of a checkpoint actually sees.
    """
    from scwbd.foundation.heads import EEGMMIDB_CHANNELS, _montage_positions

    xyz, kept = _montage_positions(EEGMMIDB_CHANNELS)
    assert kept == EEGMMIDB_CHANNELS, "standard_1005 has all 64; none may be dropped"
    assert xyz.shape == (64, 3)
    # Real electrodes sit on a head, not on a unit-radius construction: the
    # spread of radii is what distinguishes measured geometry from the spiral,
    # which places every point at exactly 95 mm.
    radii = (xyz**2).sum(axis=1) ** 0.5
    assert radii.std() > 1.0, "every electrode at one radius -- this is the spiral"

    lf = build_lead_field(anat, channel_names=EEGMMIDB_CHANNELS)
    assert "real 10-10 montage positions" in lf.note
    assert "Fibonacci" not in lf.note


def test_a_montage_that_falls_back_says_so_in_its_note(anat, monkeypatch) -> None:
    """The repair is not "it works now" -- it is that the artifact records which branch ran."""
    import scwbd.foundation.heads as H

    def boom(names):
        raise RuntimeError("mne unavailable")

    monkeypatch.setattr(H, "_montage_positions", boom)
    lf = H.build_lead_field(anat, channel_names=["Cz", "Pz"])
    assert "Fibonacci spiral" in lf.note
    assert "NOT electrode geometry" in lf.note
    assert "real 10-10 montage positions" not in lf.note


def _small(**kw) -> ModelConfig:
    return ModelConfig(hidden=64, region_embed=16, encoder_channels=16, context_dim=16, **kw)


def test_each_declared_montage_gets_its_own_head(anat) -> None:
    m = SCWBD(
        _small(
            montages={
                "sleepedf_real": {"kind": "bipolar", "channels": list(SLEEP_EDF_EEG_CHANNELS)},
            }
        ),
        anat,
    )
    head = m.eeg_head_for("sleepedf_real")
    assert head is not m.eeg
    assert head.channel_names == ("Fpz-Cz", "Pz-Oz")
    assert head.L.shape[0] == 2
    # A source with no declared montage observes through the founding head.
    assert m.eeg_head_for("eegmmidb_real") is m.eeg


def test_montage_heads_are_named_parameters_a_card_can_grant(anat) -> None:
    """The whole point of run 3: a module no card can name trains at its init.

    ``eeg_montages.*`` must appear in ``named_parameters`` so a grant pattern can
    reach it and ``test_card_patterns_reach_the_model`` can see it.
    """
    m = SCWBD(
        _small(montages={"sleepedf_real": {"kind": "bipolar", "channels": list(SLEEP_EDF_EEG_CHANNELS)}}),
        anat,
    )
    names = [n for n, _ in m.named_parameters() if n.startswith("eeg_montages.")]
    assert names, "the montage heads registered no parameters"
    assert any(n.endswith("log_noise") for n in names)


def test_a_montage_with_no_kind_is_refused(anat) -> None:
    """No default. Guessing monopolar gives a plausible number for a wrong operator."""
    with pytest.raises(ValueError, match="kind"):
        SCWBD(_small(montages={"s": {"channels": ["EEG Fpz-Cz"]}}), anat)


def test_a_montage_with_no_channels_is_refused(anat) -> None:
    with pytest.raises(ValueError, match="declares no channels"):
        SCWBD(_small(montages={"s": {"kind": "bipolar", "channels": []}}), anat)
