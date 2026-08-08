"""The learned pulse must survive save/load, and its absence must be reported.

`TMSDrive` is not part of `model`: it is built beside the perturbation corpus
and only when that corpus is on disk. So it needs its own slot in the checkpoint
payload, and it did not have one — its four tensors were trained and then
dropped on every save. Those four are the whole learned content of "what a pulse
does": the amplitude and the profile over motor parcels.

Found by launch check #1, which reported `tms_drive.*` as a grant pattern naming
nothing. It named nothing because the checkpoint contained nothing; the card was
right and the checkpoint was wrong.

This matters twice over. A finished run would have published a model whose
perturbation response was re-initialised noise, and a 35-hour run interrupted
and resumed would have silently restarted the pulse from scratch — with the loss
curve continuing smoothly, because six other sources are still training.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from scwbd.foundation.anatomy import load_anatomy
from scwbd.foundation.checkpoint import load_checkpoint, save_checkpoint
from scwbd.foundation.config import ModelConfig, FoundationConfig
from scwbd.foundation.model import SCWBD
from scwbd.foundation.perturb import TMSDrive


@pytest.fixture(scope="module")
def anat():
    return load_anatomy(device="cpu")


@pytest.fixture(scope="module")
def tiny(anat):
    cfg = FoundationConfig()
    cfg.model = ModelConfig(hidden=64, region_embed=16, encoder_channels=16, context_dim=16)
    return cfg, SCWBD(cfg.model, anat)


def test_a_trained_drive_round_trips(anat, tiny) -> None:
    """Values written into the drive come back out of the checkpoint."""
    cfg, model = tiny
    src = TMSDrive(anat)
    with torch.no_grad():
        src.logits["left"].add_(0.37)
        src.log_gain["right"].add_(-1.25)
    want_logits = src.logits["left"].detach().clone()
    want_gain = src.log_gain["right"].detach().clone()

    d = Path(tempfile.mkdtemp())
    save_checkpoint(d / "rt.pt", model=model, config=cfg, step=7, stage="roundtrip", tms_drive=src)

    dst = TMSDrive(anat)
    assert float(dst.logits["left"].abs().sum()) == 0.0, "logits should start at zero"
    load_checkpoint(d / "rt.pt", tms_drive=dst, strict=False)

    assert torch.equal(dst.logits["left"], want_logits)
    assert torch.equal(dst.log_gain["right"], want_gain)


def test_a_missing_drive_is_reported_not_skipped(anat, tiny) -> None:
    """Absence is a fact the load report must carry.

    A caller that gates on ``load_report`` otherwise reads silence as success --
    the silent-load failure, one level up. A resumed run would then continue with
    a re-initialised pulse and nothing would say so.
    """
    cfg, model = tiny
    d = Path(tempfile.mkdtemp())
    save_checkpoint(d / "none.pt", model=model, config=cfg, step=0, stage="s", tms_drive=None)

    dst = TMSDrive(anat)
    payload = load_checkpoint(d / "none.pt", tms_drive=dst, strict=False)
    assert (payload.get("load_report") or {}).get("tms_drive_absent") is True


def test_the_drive_is_actually_in_the_payload(anat, tiny) -> None:
    """Guards the guard: both tests above pass if `tms_drive` silently no-ops."""
    cfg, model = tiny
    d = Path(tempfile.mkdtemp())
    save_checkpoint(d / "p.pt", model=model, config=cfg, step=0, stage="s", tms_drive=TMSDrive(anat))
    payload = torch.load(d / "p.pt", map_location="cpu", weights_only=False)
    assert "tms_drive" in payload, "save_checkpoint has no slot for the drive"
    assert sorted(payload["tms_drive"]) == [
        "log_gain.left",
        "log_gain.right",
        "logits.left",
        "logits.right",
    ]
