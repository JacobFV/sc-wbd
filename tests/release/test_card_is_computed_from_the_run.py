"""Anything the card says about the weights must come from the run's configuration.

The near-miss this exists for, 2026-08-07. ``_unreachable_parameters`` read the
**live** source-card directory, so it characterised a finished run using current
configuration. The published card says the run could not train 88.8% of its
parameters; regenerating after the card patterns were repaired produced **0.9%**.

Republishing would have silently replaced the artifact's headline finding with a
number describing a run that never happened — and every other check passed. Card
live, site live, repo clean, tests green. It was found only by diffing the
published bytes against freshly generated ones, which nothing had ever done.

Fixing configuration does not retrain weights. The checkpoint records ``git_sha``;
the cards at that commit are the ones that governed it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "checkpoints/scwbd-002-pilot"


def _plan():
    from scwbd.release.publish import plan_run2_pilot

    return plan_run2_pilot(checkpoint_dir=str(CKPT))


@pytest.mark.skipif(not CKPT.is_dir(), reason="run-2 checkpoint not on disk")
def test_the_gradient_reach_figure_states_which_commit_it_came_from() -> None:
    """A number about the weights with no provenance is a number about today."""
    warnings = [w for w in _plan().warnings if "receive a gradient" in w]
    assert warnings, "the gradient-reach disclosure is gone from the card"
    text = warnings[0]

    m = re.search(r"Computed from the source cards at `([0-9a-f]{7,40})`", text)
    assert m, (
        "the disclosure does not say which commit's cards it was computed from. "
        "Without that it may have been computed from the cards currently on disk, "
        "which describe a run that never happened."
    )

    import torch

    sha = ""
    for f in sorted(CKPT.glob("stage_*.pt")):
        sha = str(torch.load(f, map_location="cpu", weights_only=False).get("git_sha") or "")
        if sha:
            break
    assert sha, "the checkpoint records no git_sha; the figure has no anchor"
    assert sha.startswith(m.group(1)), (
        f"the card cites commit {m.group(1)} but the checkpoint records {sha[:12]}. "
        "The figure is anchored to the wrong run."
    )


@pytest.mark.skipif(not CKPT.is_dir(), reason="run-2 checkpoint not on disk")
def test_a_dirty_training_tree_is_disclosed_rather_than_ignored() -> None:
    """``-dirty`` means the cards that trained may differ from the cards at the commit.

    Run 2's sha carries it. Stripping the suffix to get a usable rev is correct;
    doing so silently is not, because the suffix is precisely the warning that the
    tree and the commit disagreed.
    """
    import torch

    sha = ""
    for f in sorted(CKPT.glob("stage_*.pt")):
        sha = str(torch.load(f, map_location="cpu", weights_only=False).get("git_sha") or "")
        if sha:
            break
    if not sha.endswith("-dirty"):
        pytest.skip("this checkpoint's tree was clean; nothing to disclose")

    text = " ".join(w for w in _plan().warnings if "receive a gradient" in w)
    assert "-dirty" in text and "uncommitted" in text, (
        "the checkpoint records a dirty tree and the card does not say so. The "
        "figure was computed from the commit's cards, which may not be the cards "
        "that trained."
    )


@pytest.mark.skipif(not CKPT.is_dir(), reason="run-2 checkpoint not on disk")
def test_the_figure_is_not_recomputed_from_whatever_is_on_disk_now() -> None:
    """The discriminating check: editing a live card must not move the number.

    This is what would have caught the original defect. The live cards have since
    been repaired to grant `family_local.*` and friends; if the figure were read
    from them it would collapse from ~88% to ~1%.
    """
    warnings = [w for w in _plan().warnings if "receive a gradient" in w]
    assert warnings
    m = re.search(r"\(([0-9.]+)%\)", warnings[0])
    assert m, f"no percentage in the disclosure: {warnings[0][:120]}"
    pct = float(m.group(1))
    assert pct > 50.0, (
        f"the gradient-reach figure is {pct}%. The live cards now grant the "
        "renamed regional modules, so a figure this low means it is being computed "
        "from today's cards rather than from the run's -- which is the defect this "
        "file exists for. It was 88.8% when computed at the recorded commit."
    )
