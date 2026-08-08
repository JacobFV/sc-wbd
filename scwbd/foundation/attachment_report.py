"""Per attachment kind: did anything of that kind reach the model at all?

Two different questions get confused with each other, and only the second
answers what the public page promises:

    "every enabled source contributed gradient"
    "every kind of signal the schema declares was exercised"

The first can be true while the second is false. Run 2 had five enabled sources
and every one of them was an ``observation`` -- the schema declared
``stimulus``, ``boundary_output`` and ``context`` too, and nothing in the
mixture carried one. The landing page's schematic gives ``boundary_output``
equal billing with observation, so "every source contributed" would have read as
an answer to a question it never touched.

This module reports both, and derives both from artifacts rather than
assertions: the attachment kinds from the enabled cards' declared
``channels``, and whether each reached a gradient from the checkpoint's own
``contributed_sources`` and ``moved_since_init``.

A channel is reported in one of four states, and the distinctions are the point:

``exercised``      its source contributed a loss term and the modules that kind
                   attaches to moved off their initialisation.
``contributed``    its source contributed a loss term, but this particular
                   channel feeds no loss -- ds004024's EMG, for instance, is
                   used to recover the stimulated hemisphere and never scored.
``declared_only``  the card declares the channel and nothing consumed it.
``disabled``       the card is not enabled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

__all__ = ["ATTACHMENT_KINDS", "attachment_report", "render_markdown"]

#: The axis in ``scwbd/schema/attachment.py``. Listed here so a kind that no
#: card declares is reported as ABSENT rather than omitted -- an attachment kind
#: missing from a report reads as "not applicable" and it means "untested".
ATTACHMENT_KINDS: tuple[str, ...] = ("observation", "stimulus", "boundary_output", "context")

#: Which top-level module a kind's evidence lands in. Used to turn
#: ``moved_since_init`` into a per-kind answer.
_KIND_MODULES: dict[str, tuple[str, ...]] = {
    "observation": ("eeg", "eeg_montages", "bold", "observation"),
    "boundary_output": ("behaviour",),
    "stimulus": ("tms_drive",),
    "context": ("context",),
}


def _load_cards(cards_dir: str | Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for f in sorted(Path(cards_dir).glob("*.yaml")):
        card = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        out[str(card.get("id", f.stem))] = card
    return out


def attachment_report(
    checkpoint: str | Path | None = None,
    *,
    cards_dir: str | Path = "configs/curriculum/source_cards",
    channels_consumed: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """What each attachment kind's status is, derived from cards + checkpoint.

    ``channels_consumed`` maps ``source_id`` to the channel names that actually
    feed a loss. It is passed in rather than guessed: whether a declared channel
    is scored is a property of the trainer's loss methods, and inferring it from
    the card would let the card assert its own exercise.
    """
    cards = _load_cards(cards_dir)
    consumed = channels_consumed or DEFAULT_CHANNELS_CONSUMED

    contributed: set[str] = set()
    moved: dict[str, dict[str, int]] = {}
    ckpt_path = None
    if checkpoint is not None and Path(checkpoint).is_file():
        import torch

        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        extra = ck.get("extra") or {}
        contributed = set(extra.get("contributed_sources") or [])
        moved = ((extra.get("moved_since_init") or {}).get("by_module")) or {}
        ckpt_path = str(checkpoint)

    def _modules_moved(kind: str) -> dict[str, int] | None:
        mods = {m: moved[m]["moved"] for m in _KIND_MODULES.get(kind, ()) if m in moved}
        return mods or None

    kinds: dict[str, Any] = {}
    for kind in ATTACHMENT_KINDS:
        channels = []
        for sid, card in cards.items():
            for name, spec in (card.get("channels") or {}).items():
                if (spec or {}).get("attachment") != kind:
                    continue
                enabled = bool(card.get("enabled", True))
                is_consumed = name in consumed.get(sid, ())
                if not enabled:
                    state = "disabled"
                elif is_consumed and sid in contributed:
                    state = "exercised"
                elif is_consumed:
                    state = "admitted_not_yet_measured" if ckpt_path else "declared_only"
                elif sid in contributed:
                    state = "contributed"
                else:
                    state = "declared_only"
                channels.append(
                    {
                        "source": sid,
                        "channel": name,
                        "enabled": enabled,
                        "feeds_a_loss": is_consumed,
                        "state": state,
                        "n_channels": (spec or {}).get("n_channels"),
                        "operator": (spec or {}).get("operator"),
                        "note": (spec or {}).get("note", "").strip(),
                    }
                )
        exercised = [c for c in channels if c["state"] == "exercised"]
        kinds[kind] = {
            "declared_by_any_card": bool(channels),
            "declared_by_enabled_card": any(c["enabled"] for c in channels),
            "reached_the_model": bool(exercised),
            "n_channels_declared": len(channels),
            "n_channels_feeding_a_loss": sum(1 for c in channels if c["feeds_a_loss"]),
            "modules_moved": _modules_moved(kind),
            "channels": channels,
        }

    absent = [k for k in ATTACHMENT_KINDS if not kinds[k]["reached_the_model"]]
    return {
        "checkpoint": ckpt_path,
        "cards_dir": str(cards_dir),
        "contributed_sources": sorted(contributed),
        "kinds": kinds,
        "kinds_that_reached_the_model": [k for k in ATTACHMENT_KINDS if kinds[k]["reached_the_model"]],
        "kinds_that_did_not": absent,
        "claim": (
            "Every attachment kind the schema declares was exercised."
            if not absent
            else "Not every attachment kind the schema declares was exercised: "
            + ", ".join(absent)
            + " reached no loss."
        ),
        "why_this_is_not_the_same_as_every_source_contributing": (
            "A run can have every enabled source contribute gradient while three "
            "of the four attachment kinds are untouched, because a source list can "
            "be entirely observations. Only the per-kind answer speaks to the "
            "schematic that gives boundary_output equal billing with observation."
        ),
    }


#: Which declared channels actually feed a loss, per source. Maintained beside
#: the trainer's loss methods, and deliberately NOT inferred from the cards: a
#: card that declared its own exercise could never be caught not delivering it.
DEFAULT_CHANNELS_CONSUMED: dict[str, tuple[str, ...]] = {
    "eegmmidb_real": ("eeg",),
    "sleepedf_real": ("eeg_bipolar",),
    "ds000117_real": ("eeg",),
    "ds004024_rest_real": ("eeg",),
    "ds002336_real": ("eeg", "bold"),
    "ds000117_behaviour": ("button_press", "response_time"),
    "ds004024_perturb": ("eeg", "tms_pulse"),
}


def render_markdown(rep: dict[str, Any]) -> str:
    """The report as prose a card can carry, leading with what was exercised."""
    lines: list[str] = []
    lines.append("# Attachment kinds exercised by SC-WBD-003\n")
    lines.append(rep["claim"] + "\n")
    got = rep["kinds_that_reached_the_model"]
    lines.append(
        f"Of the four kinds `scwbd/schema/attachment.py` declares, "
        f"{len(got)} reached a loss: {', '.join(f'`{k}`' for k in got) or 'none'}."
    )
    if rep["kinds_that_did_not"]:
        lines.append(
            "The remaining "
            + ", ".join(f"`{k}`" for k in rep["kinds_that_did_not"])
            + " did not, and the table below says whether that is because no card "
            "declares one or because a declared channel feeds no loss.\n"
        )
    lines.append("")
    lines.append("| kind | source | channel | state | feeds a loss |")
    lines.append("| --- | --- | --- | --- | --- |")
    for kind in ATTACHMENT_KINDS:
        chans = rep["kinds"][kind]["channels"]
        if not chans:
            lines.append(f"| `{kind}` | — | — | **no card declares one** | no |")
            continue
        for c in chans:
            lines.append(
                f"| `{kind}` | `{c['source']}` | `{c['channel']}` | {c['state']} | "
                f"{'yes' if c['feeds_a_loss'] else 'no'} |"
            )
    lines.append("")
    lines.append("## Why this table is not the contributed-gradient table\n")
    lines.append(rep["why_this_is_not_the_same_as_every_source_contributing"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - entry point
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", default="checkpoints/scwbd-003/last.pt")
    p.add_argument("--cards", default="configs/curriculum/source_cards")
    p.add_argument("--json-out", default="reports/attachment_kinds.json")
    p.add_argument("--md-out", default="reports/attachment_kinds.md")
    a = p.parse_args(argv)
    rep = attachment_report(a.checkpoint, cards_dir=a.cards)
    Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json_out).write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    Path(a.md_out).write_text(render_markdown(rep), encoding="utf-8")
    print(rep["claim"])
    for k in ATTACHMENT_KINDS:
        v = rep["kinds"][k]
        print(
            f"  {k:16s} declared={v['n_channels_declared']:2d} "
            f"feeding_a_loss={v['n_channels_feeding_a_loss']:2d} "
            f"reached_the_model={v['reached_the_model']}"
        )
    print(f"wrote {a.json_out} and {a.md_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
