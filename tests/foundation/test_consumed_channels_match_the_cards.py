"""Which channels feed a loss is hand-maintained; it must still match the cards.

`attachment_report.DEFAULT_CHANNELS_CONSUMED` maps a source id to the channel
names that actually reach a loss. It is written by hand beside the trainer's
loss methods and NOT derived from the cards, on purpose: a card that declared
its own exercise could never be caught failing to deliver it.

The cost of that choice is drift. A channel renamed on a card, or a source
added to the mixture and forgotten here, silently changes a published claim —
`attachment_report` marks the channel `declared_only` and the kind stops
counting as reached.

This is not hypothetical. `eegmmidb_real`'s card predated the attachment axis
and carried no `channels:` block at all, so the per-attachment-kind report
omitted **the founding source** from its `observation` row: the 109-participant
corpus that founds the representation was missing from the table that says which
kinds of signal reached the model. Nothing in training depended on it, which is
why it survived — the gradient path never consults that block.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scwbd.foundation.attachment_report import ATTACHMENT_KINDS, DEFAULT_CHANNELS_CONSUMED

REPO = Path(__file__).resolve().parents[2]
CARDS = REPO / "configs/curriculum/source_cards"


def _cards() -> dict[str, dict]:
    out = {}
    for f in sorted(CARDS.glob("*.yaml")):
        c = yaml.safe_load(f.read_text()) or {}
        out[str(c.get("id", f.stem))] = c
    return out


def test_every_consumed_channel_is_declared_by_its_card() -> None:
    cards = _cards()
    bad: list[str] = []
    for sid, chans in DEFAULT_CHANNELS_CONSUMED.items():
        card = cards.get(sid)
        if card is None:
            bad.append(f"{sid}: no card with this id")
            continue
        declared = set(card.get("channels") or {})
        for ch in chans:
            if ch not in declared:
                bad.append(f"{sid}.{ch}: not declared; card declares {sorted(declared)}")
    assert not bad, (
        "the consumed-channel map names channels no card declares: "
        f"{bad}. attachment_report will mark them `declared_only`, and the "
        "attachment kind will stop counting as reached — silently, because "
        "nothing in training reads the card's `channels` block."
    )


def test_every_enabled_likelihood_source_appears_in_the_map() -> None:
    """A source added to the mixture and forgotten here vanishes from the report."""
    cards = _cards()
    enabled = {
        sid
        for sid, c in cards.items()
        if c.get("enabled", True) and c.get("role") == "likelihood"
    }
    missing = sorted(enabled - set(DEFAULT_CHANNELS_CONSUMED))
    assert not missing, (
        f"enabled likelihood sources absent from DEFAULT_CHANNELS_CONSUMED: "
        f"{missing}. Each would report as contributing no channel to any "
        "attachment kind."
    )


def test_every_declared_channel_has_a_known_attachment() -> None:
    """`attachment` is the axis; a typo in it is not a smaller claim, it is a different one."""
    bad: list[str] = []
    for sid, card in _cards().items():
        for name, spec in (card.get("channels") or {}).items():
            att = (spec or {}).get("attachment")
            if att not in ATTACHMENT_KINDS:
                bad.append(f"{sid}.{name}: attachment={att!r}")
    assert not bad, f"channels with an unrecognised attachment: {bad}"


def test_an_observation_declares_an_operator_and_nothing_else_does() -> None:
    """The rule `ChannelSpec` enforces, checked on the cards as written.

    An observation with no operator asserts that the carrier's state IS the
    measurement. A stimulus or boundary output WITH one asserts it passes
    through a forward model of neural activity. Both are refused at load time;
    this catches them in the file, where the fix is obvious.
    """
    missing_op: list[str] = []
    spurious_op: list[str] = []
    for sid, card in _cards().items():
        for name, spec in (card.get("channels") or {}).items():
            att, op = (spec or {}).get("attachment"), (spec or {}).get("operator")
            if att == "observation" and not op:
                missing_op.append(f"{sid}.{name}")
            if att != "observation" and op:
                spurious_op.append(f"{sid}.{name} -> {op}")
    assert not missing_op, f"observations with no declared operator: {missing_op}"
    assert not spurious_op, f"non-observations declaring an operator: {spurious_op}"
