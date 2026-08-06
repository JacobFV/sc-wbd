"""Rule 1 (missing is missing) and rule 2 (a present modality is declared).

Every refusal in ``availability.py`` gets a test that breaks something on
purpose.  A guard nobody has watched fire is indistinguishable from one that
cannot fire.
"""

from __future__ import annotations

import pytest

from scwbd.individualize.availability import (
    MODALITIES,
    MissingModalityError,
    ModalityAvailability,
    ModalityRecord,
    UndeclaredModalityError,
    ZeroImputationRefused,
    channel_contribution,
    refuse_zero_imputation,
)


def _rec(modality="eeg", **kw):
    base = dict(
        source_card="card:x",
        support="support:x",
        clock="clock:x",
        calibration="calib:x",
    )
    base.update(kw)
    return ModalityRecord(modality=modality, **base)


# ---------------------------------------------------------------- rule 1
def test_require_absent_modality_raises_and_never_defaults():
    av = ModalityAvailability("p", (_rec("structural_mri"),))
    with pytest.raises(MissingModalityError) as e:
        av.require("eeg")
    assert "missing" in str(e.value).lower()
    # and the API offers no way to get a default instead
    assert not hasattr(av, "get")


def test_refuse_zero_imputation_fires_for_absent_and_not_for_present():
    av = ModalityAvailability("p", (_rec("structural_mri"),))
    with pytest.raises(ZeroImputationRefused):
        refuse_zero_imputation(av, "fmri")
    # discrimination: the same guard must read differently on a present one
    refuse_zero_imputation(av, "structural_mri")


def test_mri_only_maps_to_the_prior_design_not_a_zero_filled_fmri():
    av = ModalityAvailability("p", (_rec("structural_mri"), _rec("dmri")))
    assert av.channels == ()
    assert av.design == "prior"


# ---------------------------------------------------------------- rule 2 (R01)
@pytest.mark.parametrize("slot", ["source_card", "support", "clock", "calibration"])
def test_missing_declaration_refused_per_slot(slot):
    with pytest.raises(UndeclaredModalityError) as e:
        _rec(**{slot: None})
    assert "R01" in str(e.value)


@pytest.mark.parametrize("slot", ["source_card", "support", "clock", "calibration"])
def test_blank_declaration_refused_per_slot(slot):
    with pytest.raises(UndeclaredModalityError):
        _rec(**{slot: "   "})


def test_declaration_kind_distinguishes_a_name_from_an_object():
    class _Thing:
        id = "real-object"

    named = _rec()
    assert named.declaration_kind["calibration"] == "name"
    assert named.fully_objectified is False

    objd = _rec(calibration=_Thing())
    assert objd.declaration_kind["calibration"] == "object"
    assert objd.fully_objectified is False  # the other three are still names

    allobj = ModalityRecord(
        modality="eeg",
        source_card=_Thing(),
        support=_Thing(),
        clock=_Thing(),
        calibration=_Thing(),
    )
    assert allobj.fully_objectified is True


# ---------------------------------------------------------------- design map
@pytest.mark.parametrize(
    "mods,design,channels",
    [
        (["structural_mri"], "prior", ()),
        (["dmri", "behavior"], "prior", ()),
        (["structural_mri", "eeg"], "eeg_only", ("eeg",)),
        (["structural_mri", "fmri"], "fmri_only", ("bold",)),
        (["eeg", "fmri"], "joint_native", ("eeg", "bold")),
        (["meg"], "eeg_only", ("eeg",)),
        ([], "prior", ()),
    ],
)
def test_design_resolution(mods, design, channels):
    av = ModalityAvailability.from_modalities("p", mods)
    assert av.design == design
    assert av.channels == channels


def test_meg_is_flagged_as_an_eeg_proxy_rather_than_silently_substituted():
    av = ModalityAvailability.from_modalities("p", ["meg"])
    assert av.uses_meg_as_eeg_proxy is True
    chan, why = channel_contribution("meg")
    assert chan == "eeg"
    assert "PROXY" in why
    # and it is NOT flagged when real EEG is present
    both = ModalityAvailability.from_modalities("p", ["meg", "eeg"])
    assert both.uses_meg_as_eeg_proxy is False


def test_every_modality_has_a_declared_channel_contribution():
    """No modality may fall through the table silently."""
    for m in MODALITIES:
        chan, why = channel_contribution(m)
        assert chan in (None, "eeg", "bold")
        assert why.strip(), f"{m} has no stated reason"


def test_digest_discriminates():
    """The digest must read differently for different availabilities."""
    a = ModalityAvailability.from_modalities("p", ["eeg"])
    b = ModalityAvailability.from_modalities("p", ["eeg", "fmri"])
    c = ModalityAvailability.from_modalities("q", ["eeg"])
    assert len({a.digest(), b.digest(), c.digest()}) == 3
    assert a.digest() == ModalityAvailability.from_modalities("p", ["eeg"]).digest()


def test_duplicate_records_refused():
    with pytest.raises(ValueError):
        ModalityAvailability("p", (_rec("eeg"), _rec("eeg")))


def test_absent_is_reported_not_inferred_from_silence():
    av = ModalityAvailability.from_modalities("p", ["eeg"])
    d = av.to_dict()
    assert set(d["absent"]) == set(MODALITIES) - {"eeg"}
