"""Attachment is orthogonal to integrity, and it does not default.

The tier system ranks a source by how far it can be trusted. It cannot say
whether a channel is a stimulus, an observation, or something the subject
produced -- and the datasets being added carry all three in one file.
"""

from __future__ import annotations

import pytest

from pydantic import ValidationError

from scwbd.schema.attachment import (
    ATTACHMENTS,
    AttachmentError,
    ChannelSpec,
    attachment_of,
)

# Pydantic wraps anything a field validator raises into ValidationError, so the
# constructor tests assert on THAT and on the message. The bare AttachmentError
# is still what `attachment_of` raises, since it is a plain function.


def test_an_observation_must_name_its_operator():
    """Without one it asserts the carrier's state IS the measurement."""
    with pytest.raises(ValidationError, match="declares no operator"):
        ChannelSpec(name="eeg", attachment="observation", n_channels=64)


def test_a_stimulus_may_not_have_a_forward_operator():
    """A played waveform does not pass through a model of neural activity."""
    with pytest.raises(ValidationError, match="Only observations"):
        ChannelSpec(name="audio", attachment="stimulus", n_channels=1, operator="lead_field")


def test_the_three_channels_of_one_multimodal_source_coexist():
    """ds003768 ships EEG, eye tracking and stimulus video in one dataset."""
    eeg = ChannelSpec(name="eeg", attachment="observation", n_channels=64, operator="lead_field")
    gaze = ChannelSpec(name="gaze", attachment="boundary_output", n_channels=2, units="deg")
    video = ChannelSpec(name="video", attachment="stimulus", n_channels=1)
    assert {c.attachment for c in (eeg, gaze, video)} == {
        "observation", "boundary_output", "stimulus"
    }


class _Card:
    def __init__(self, channels):
        self.channels = channels
        self.role = "likelihood"


def test_attachment_does_not_default_from_role():
    """A likelihood card's channels are not all observations.

    This is the whole point: guessing 'observation' because the role is
    'likelihood' is how a stimulus gets trained as a measurement of the brain.
    """
    card = _Card({"audio": ChannelSpec(name="audio", attachment="stimulus", n_channels=1)})
    assert attachment_of(card, "audio") == "stimulus"


def test_an_undeclared_channel_is_refused_not_assumed():
    card = _Card({"eeg": ChannelSpec(name="eeg", attachment="observation",
                                     n_channels=64, operator="lead_field")})
    with pytest.raises(AttachmentError, match="declares no channel"):
        attachment_of(card, "gaze")


def test_an_unknown_attachment_is_refused():
    class _Bad:
        attachment = "whatever"

    card = _Card({"x": _Bad()})
    with pytest.raises(AttachmentError, match="not one of"):
        attachment_of(card, "x")


def test_the_taxonomy_is_the_documented_one():
    assert ATTACHMENTS == ("stimulus", "observation", "boundary_output", "context")
