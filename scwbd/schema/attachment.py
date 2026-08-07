"""Where a channel attaches to the carrier — a second axis the tier system lacks.

The integrity tiers rank a source by *how far it can be trusted*: measured, then
calibration, then prior, then simulator, then teacher. That axis is real and it
is not the only one. It cannot express the difference between

    the audio waveform a participant heard,
    the eye position they produced,
    the EEG recorded while both happened,

and today all three would have to be declared ``role: likelihood`` or dropped.
Forcing a stimulus into ``likelihood`` trains it as though it were a measurement
*of the brain*, which is not a smaller claim than the truth — it is a different
one, and it is silent.

This matters now because the sources being added carry exactly these channels.
`MEG-MASC` ships aligned audio with phonetic and word annotation; `ds003768`
ships eye tracking and ECG alongside concurrent EEG-fMRI; `LibriBrain` ships
speech. Each is one dataset with channels that attach to the model in three
different places.

**Attachment is orthogonal to integrity.** A stimulus channel can be measured
exactly (the waveform that was played is known to the sample) while telling you
nothing about the brain; an observation channel can be noisy and still be the
only evidence about latent state. Tier answers *how much do I believe this*.
Attachment answers *what does it do in the graph*.

Nothing here defaults. ``role`` does not determine attachment — a `likelihood`
card may carry an EEG channel and an eye-tracking channel — so a channel that
declares no attachment is **refused** rather than assumed to be an observation.
That is `ARCHITECTURE.md` RL-14 applied at the point the same mistake would
otherwise be made a second time.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .base import SchemaModel

__all__ = ["Attachment", "ChannelSpec", "AttachmentError", "attachment_of"]


class AttachmentError(ValueError):
    """A channel does not say where it attaches, or says something impossible."""


#: Where a channel meets the model.
#:
#: ``stimulus``         enters the carrier as a drive; the world acting on the
#:                      subject. Known independently of the subject, so it is
#:                      never evidence about latent state on its own.
#: ``observation``      a measurement *of* the carrier through a declared
#:                      operator: EEG through a lead field, BOLD through the
#:                      balloon model. The only kind that carries a likelihood
#:                      over latent state.
#: ``boundary_output``  produced *by* the subject and measured outside the
#:                      skull: eye position, motor, speech, autonomic. Evidence
#:                      about the carrier, but through the body rather than
#:                      through a physical forward model of neural activity.
#: ``context``          slowly-varying conditioning that is neither driven nor
#:                      driving on the modelled timescale: time of day, session
#:                      index, drug state.
Attachment = Literal["stimulus", "observation", "boundary_output", "context"]

ATTACHMENTS: tuple[str, ...] = ("stimulus", "observation", "boundary_output", "context")


class ChannelSpec(SchemaModel):
    """One channel group of a source, and where it attaches.

    A *group*, not a single channel: 64 EEG electrodes are one observation
    channel group sharing one operator, and splitting them would multiply the
    declaration without adding information.
    """

    name: str
    attachment: Attachment
    #: How many scalar streams this group carries.
    n_channels: int = Field(gt=0)
    units: str = ""
    #: The named clock this group is sampled on. Two groups on different clocks
    #: must be related through a declared synchronisation, not resampled.
    clock: str = ""
    #: Required for ``observation``: the operator mapping carrier to channel.
    #: An observation with no declared operator is an assertion that the model's
    #: state *is* the measurement, which is the error the lead field exists to
    #: prevent.
    operator: str | None = None
    note: str = ""

    @model_validator(mode="after")
    def _check(self) -> "ChannelSpec":
        if self.attachment == "observation" and not self.operator:
            raise AttachmentError(
                f"channel {self.name!r} attaches as an observation but declares no "
                "operator. An observation without one asserts that the carrier's state "
                "IS the measurement -- no lead field, no haemodynamic model, no "
                "projection. Name the operator, or attach it as boundary_output if it "
                "is measured outside the skull."
            )
        if self.attachment != "observation" and self.operator:
            raise AttachmentError(
                f"channel {self.name!r} attaches as {self.attachment!r} but declares "
                f"operator {self.operator!r}. Only observations pass through a forward "
                "model of neural activity; a stimulus or a boundary output does not."
            )
        return self


def attachment_of(card: object, channel: str) -> Attachment:
    """The declared attachment of one channel, or refuse.

    Deliberately has no default. ``role`` does not determine attachment: a
    ``likelihood`` card may carry an EEG channel (observation), the audio that
    was played (stimulus) and the participant's gaze (boundary_output) in one
    file. Guessing ``observation`` because the role is ``likelihood`` is exactly
    how a stimulus would get trained as a measurement of the brain.
    """
    chans = getattr(card, "channels", None) or {}
    if channel not in chans:
        raise AttachmentError(
            f"card declares no channel {channel!r}; it has {sorted(chans)}. A channel "
            "the card does not declare has no attachment, and inventing one here is "
            "the same error one level down."
        )
    spec = chans[channel]
    att = getattr(spec, "attachment", None)
    if att not in ATTACHMENTS:
        raise AttachmentError(
            f"channel {channel!r} declares attachment {att!r}, which is not one of "
            f"{ATTACHMENTS}. There is no default: see ARCHITECTURE.md RL-14 -- a "
            "lookup whose default grants is a configuration system that cannot report "
            "a typo."
        )
    return att  # type: ignore[return-value]
