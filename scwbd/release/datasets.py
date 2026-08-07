"""Linking training sources to dataset cards: modality and licence.

The repository has **two** card systems and they do not reference each other:

* ``configs/source_cards/*.yaml`` — the *training mixture* cards. They carry
  role, ``enabled``, gradient permission and reliability inputs. They carry
  **no licence field and no modality field at all.**
* ``scwbd/sources/cards/*.yaml`` — the *dataset* cards. They carry
  ``governance.license``, ``signal.modalities``, availability status and the
  file manifest.

A release manifest needs both halves: which sources moved weights (mixture
cards) and what those sources are and permit (dataset cards). This module
builds the link, and it builds it **from the run's own configuration** — the
data roots in ``configs/*.yaml`` name the dataset directories — rather than
from a hard-coded table, so a config change moves the link instead of silently
invalidating it.

Where no link can be established the source is recorded with modality
``unknown`` and :data:`~scwbd.release.licence.UNKNOWN_TERM`. That is the
house rule: absence writes something.

**Modality is not integrity.** A ``-raw`` checkpoint is one trained only on
measured human observation, whatever the instrument: EEG, MEG, fMRI,
structural MRI, dMRI, EOG/EMG/ECG and physiological channels are all tier-1
measurements. fMRI is not lower-integrity than EEG. What separates tiers is
provenance — measurement vs population prior vs simulation vs teacher
prediction — and that is :data:`FAMILY_TIER` in
:mod:`scwbd.release.families`, not anything here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .licence import LicenceTerm, term_from_dataset_card

__all__ = [
    "DATASET_CARD_DIR",
    "DatasetInfo",
    "load_dataset_cards",
    "link_sources_to_datasets",
    "MEASURED_MODALITIES",
]

DATASET_CARD_DIR = Path("scwbd/sources/cards")

#: Modalities that constitute measured human observation. Verified against
#: ``scwbd/sources/cards/*.yaml`` ``signal.modalities`` on 2026-08-06; the set
#: is the union actually present on disk, not an aspirational list.
MEASURED_MODALITIES: frozenset[str] = frozenset(
    {
        "eeg", "meg", "ieeg",           # electrophysiology
        "fmri", "mri", "dwi", "dti", "swi", "pet",  # imaging
        "eog", "emg", "ecg", "resp", "temperature", # physiological
        "hypnogram", "stimulation",     # derived annotation / delivered perturbation
        "bem", "forward",               # subject head models derived from that subject's MRI
    }
)


@dataclass(frozen=True)
class DatasetInfo:
    """The release-relevant projection of one dataset card."""

    dataset_id: str
    version: str
    status: str
    modalities: tuple[str, ...]
    licence: LicenceTerm
    card_path: str
    local_path: str | None = None

    @property
    def is_available(self) -> bool:
        """``live`` or ``partial``. An ``unavailable`` source trained nothing."""
        return self.status in ("live", "partial")

    @property
    def measured_modalities(self) -> tuple[str, ...]:
        return tuple(m for m in self.modalities if m in MEASURED_MODALITIES)

    @property
    def unknown_modalities(self) -> tuple[str, ...]:
        """Modalities present on the card that this module does not classify.

        Recorded rather than dropped: a modality nobody recognised is a gap in
        this table, and it should be visible as one.
        """
        return tuple(m for m in self.modalities if m not in MEASURED_MODALITIES)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "status": self.status,
            "modalities": list(self.modalities),
            "measured_modalities": list(self.measured_modalities),
            "unknown_modalities": list(self.unknown_modalities),
            "available": self.is_available,
            "card_path": self.card_path,
            "licence": self.licence.as_dict(),
        }


def load_dataset_cards(card_dir: str | Path = DATASET_CARD_DIR) -> dict[str, DatasetInfo]:
    """Load every dataset card, keyed by dataset id."""
    d = Path(card_dir)
    if not d.is_dir():
        return {}
    out: dict[str, DatasetInfo] = {}
    for p in sorted(d.glob("*.yaml")):
        data = yaml.safe_load(p.read_text()) or {}
        ident = data.get("identity", {}) or {}
        gov = data.get("governance", {}) or {}
        sig = data.get("signal", {}) or {}
        did = str(ident.get("id") or p.stem)
        out[did] = DatasetInfo(
            dataset_id=did,
            version=str(ident.get("version") or "unknown"),
            status=str(gov.get("status") or "unknown"),
            modalities=tuple(sig.get("modalities") or ()),
            licence=term_from_dataset_card(p),
            card_path=str(p),
            local_path=ident.get("local_path"),
        )
    return out


def _dataset_ids_in_config(config: Mapping[str, Any]) -> set[str]:
    """Dataset ids implied by the run config's data roots.

    ``data.real_eeg_root: /data/scwbd/eegmmidb/1.0.0`` names the dataset
    directory, so the id is recoverable without a hard-coded table. This is
    the "regenerate, don't audit the table" rule applied to the link itself.
    """
    found: set[str] = set()
    data = config.get("data", {}) or {}
    for key, value in data.items():
        if not isinstance(value, str) or "/" not in value:
            continue
        parts = [p for p in Path(value).parts if p not in ("/", "data", "scwbd")]
        for p in parts:
            found.add(p)
    return found


def link_sources_to_datasets(
    source_ids: Sequence[str],
    *,
    config: Mapping[str, Any] | None = None,
    cards: Mapping[str, DatasetInfo] | None = None,
) -> dict[str, DatasetInfo | None]:
    """Best-effort map from mixture source id to dataset card.

    Matching is by (a) exact id, then (b) the config's data roots intersected
    with a normalised form of the source id (``eegmmidb_real`` -> ``eegmmidb``,
    ``sleepedf_real`` -> ``sleep-edfx``), then (c) ``None``.

    ``None`` is a legitimate, recorded answer: ``sim_wholebrain``,
    ``anatomical_prior``, ``montage_calibration``,
    ``negative_control_shuffled`` and ``tribe_v2_teacher`` are not datasets in
    ``scwbd/sources/cards/`` and never will be. The caller must not treat a
    ``None`` as "no constraints"; :mod:`scwbd.release.manifest` attaches an
    explicit term for each of those instead.
    """
    cards = dict(cards if cards is not None else load_dataset_cards())
    cfg_ids = _dataset_ids_in_config(config or {})
    out: dict[str, DatasetInfo | None] = {}
    for sid in source_ids:
        if sid in cards:
            out[sid] = cards[sid]
            continue
        stem = re.sub(r"_(real|sim|data)$", "", sid)
        squash = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())  # noqa: E731
        hit = None
        for did, info in cards.items():
            if squash(did) == squash(stem):
                hit = info
                break
        if hit is None:
            # fall back to the config's data roots: they name real directories
            for cid in cfg_ids:
                if cid in cards and squash(cid).startswith(squash(stem)[:6] or "\0"):
                    hit = cards[cid]
                    break
        out[sid] = hit
    return out
