"""Dataset registry: id -> (version pin, local path, fetcher, loader, card, status).

The registry is the single place that knows *where* a dataset lives and *which
release* is pinned.  It carries no opinions about training: what a source may
update lives in its card (``gradient_permission``), and what it may be split by
lives in ``split_policy``.

Availability is computed, never asserted: a dataset is ``live`` only when the
card says so **and** its manifest exists **and** its root is on disk.  Anything
else reports ``unavailable`` with a reason.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from .cards import CARD_DIR, SourceCardDoc, load_card
from .download import (
    CompositeFetcher,
    Fetcher,
    HttpFetcher,
    MneDatasetFetcher,
    OpenNeuroSnapshotFetcher,
    S3PrefixFetcher,
    UnavailableFetcher,
)
from .manifest import MANIFEST_DIR, Manifest, load_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBSET_DIR = Path(__file__).parent / "subsets"


def _read_subset(name: str) -> tuple[str, ...]:
    """Load a pinned file subset (``relpath<TAB>size``) shipped with the repo.

    The subset list is part of the reproducibility record: it says exactly
    which files of a large release are on disk, so ``python -m
    scwbd.sources.download <id>`` reproduces the same tree rather than
    "whatever fitted".
    """
    p = SUBSET_DIR / name
    if not p.exists():
        return ()
    return tuple(
        line.split("\t")[0] for line in p.read_text().splitlines() if line.strip()
    )


def data_root() -> Path:
    """Runtime data root: ``$SCWBD_DATA_ROOT``, else ``<repo>/data``, else /data/scwbd."""
    env = os.environ.get("SCWBD_DATA_ROOT")
    if env:
        return Path(env)
    repo_data = REPO_ROOT / "data"
    if repo_data.exists():
        return repo_data
    return Path("/data/scwbd")


@dataclass
class DatasetEntry:
    dataset_id: str
    version: str
    name: str
    role: str
    modalities: tuple[str, ...]
    card_name: str
    fetcher: Fetcher | None = None
    loader_ref: str | None = None  # "module:function" under scwbd.sources.loaders
    subset: str = "full"
    notes: str = ""

    # -- paths ----------------------------------------------------------
    @property
    def local_path(self) -> Path:
        return data_root() / self.dataset_id / self.version

    @property
    def card_path(self) -> Path:
        return CARD_DIR / self.card_name

    @property
    def manifest_path(self) -> Path:
        return MANIFEST_DIR / f"{self.dataset_id}__{self.version}.json"

    # -- state ----------------------------------------------------------
    def card(self) -> SourceCardDoc:
        return load_card(self.card_path)

    def manifest(self) -> Manifest:
        return load_manifest(self.dataset_id, self.version)

    def has_manifest(self) -> bool:
        return self.manifest_path.exists()

    def on_disk_bytes(self) -> int:
        if not self.local_path.exists():
            return 0
        total = 0
        for p in self.local_path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total

    def status(self) -> tuple[str, str]:
        """Return ``(status, reason)`` with status in live/partial/unavailable."""
        try:
            card = self.card()
        except Exception as exc:
            return "unavailable", f"card does not load: {exc}"
        declared = card.status
        if declared == "unavailable":
            return "unavailable", str(card.data["governance"].get("unavailable_reason", ""))
        if not self.local_path.exists():
            return "unavailable", f"declared {declared!r} but {self.local_path} is absent"
        if not self.has_manifest():
            return "partial", f"bytes present but no manifest at {self.manifest_path}"
        return declared, ""

    def loader(self) -> Callable[..., Any] | None:
        if self.loader_ref is None:
            return None
        mod_name, _, fn = self.loader_ref.partition(":")
        mod = importlib.import_module(f"scwbd.sources.loaders.{mod_name}")
        return getattr(mod, fn)

    def describe(self) -> str:
        status, reason = self.status()
        gb = self.on_disk_bytes() / 1e9
        tail = f" - {reason}" if reason else ""
        return (
            f"{self.dataset_id:<22} v{self.version:<8} {status:<12} "
            f"{gb:8.3f} GB  role={self.role:<16} {'/'.join(self.modalities)}{tail}"
        )


# --------------------------------------------------------------------------
# the register
# --------------------------------------------------------------------------
_ENTRIES: list[DatasetEntry] = [
    # ---- electrophysiology, PhysioNet open mirror ----------------------
    DatasetEntry(
        dataset_id="eegmmidb",
        version="1.0.0",
        name="EEG Motor Movement/Imagery Dataset",
        role="likelihood",
        modalities=("eeg",),
        card_name="eegmmidb.yaml",
        fetcher=S3PrefixFetcher(
            dataset_id="eegmmidb", bucket="physionet-open", prefix="eegmmidb/1.0.0/"
        ),
        loader_ref="eegmmidb:load_run",
        notes="BCI2000, 64-ch, 160 Hz, executed and imagined movement.",
    ),
    DatasetEntry(
        dataset_id="sleep-edfx",
        version="1.0.0",
        name="Sleep-EDF Database Expanded (sleep-cassette)",
        role="likelihood",
        modalities=("eeg", "eog", "emg", "hypnogram"),
        card_name="sleep-edfx.yaml",
        fetcher=S3PrefixFetcher(
            dataset_id="sleep-edfx",
            bucket="physionet-open",
            prefix="sleep-edfx/1.0.0/",
            includes=("sleep-cassette/*", "*.xls", "RECORDS*", "SHA256SUMS.txt"),
        ),
        subset="sleep-cassette only (sleep-telemetry not fetched)",
        loader_ref="sleep_edfx:load_recording",
        notes="Two-night PSG, mixed native rates (100 Hz EEG/EOG, 1 Hz oro-nasal/temp).",
    ),
    # ---- MNE sample data: MEG/EEG with subject MRI + BEM + forward -----
    DatasetEntry(
        dataset_id="mne-sample",
        version="processed-v6",
        name="MNE-Python sample dataset (Neuromag Vectorview, audiovisual)",
        role="calibration",
        modalities=("meg", "eeg", "mri", "bem", "forward"),
        card_name="mne-sample.yaml",
        fetcher=MneDatasetFetcher(dataset_id="mne-sample", mne_name="sample"),
        loader_ref="mne_datasets:load_sample",
        notes="Ships BEM surfaces, -trans.fif and a precomputed forward solution.",
    ),
    DatasetEntry(
        dataset_id="mne-somato",
        version="bids-v0.10",
        name="MNE-Python somato dataset (median nerve stimulation)",
        role="likelihood",
        modalities=("meg", "mri", "bem", "forward"),
        card_name="mne-somato.yaml",
        fetcher=MneDatasetFetcher(dataset_id="mne-somato", mne_name="somato"),
        loader_ref="mne_datasets:load_somato",
        notes="BIDS-formatted; somatosensory evoked fields with subject FreeSurfer dir.",
    ),
    DatasetEntry(
        dataset_id="mne-spm-face",
        version="v1",
        name="MNE-Python SPM faces dataset (CTF 275, faces vs scrambled)",
        role="likelihood",
        modalities=("meg", "mri", "bem"),
        card_name="mne-spm-face.yaml",
        fetcher=MneDatasetFetcher(dataset_id="mne-spm-face", mne_name="spm_face"),
        loader_ref="mne_datasets:load_spm_face",
        notes="CTF axial gradiometers - a second MEG sensor geometry for lead-field checks.",
    ),
    # ---- OpenNeuro ----------------------------------------------------
    DatasetEntry(
        dataset_id="ds004024",
        version="1.0.0",
        name="ccPAS TMS-EEG with MRI/fMRI/dMRI (Shirley Ryan AbilityLab)",
        role="likelihood",
        modalities=("eeg", "tms", "mri", "fmri"),
        card_name="ds004024.yaml",
        fetcher=S3PrefixFetcher(
            dataset_id="ds004024",
            bucket="openneuro.org",
            prefix="ds004024/",
            includes=(
                "*.json",
                "*.tsv",
                "README",
                "CHANGES",
                "sub-CON001/ses-mri/anat/*",
                "sub-CON001/ses-async14ms/eeg/*task-spTMS_run-0[1-6]*",
                "sub-CON001/ses-async14ms/eeg/*task-rest_run-01*",
                "sub-CON006/ses-mri/anat/*",
                "sub-CON006/ses-async14ms/eeg/*task-spTMS_run-0[1-6]*",
                "sub-CON006/ses-async14ms/eeg/*task-rest_run-01*",
            ),
            excludes=(".datalad/*",),
        ),
        subset=(
            "all BIDS metadata (13 subjects); binaries for sub-CON001 and sub-CON006 only, "
            "and within those only ses-async14ms spTMS runs 01-06 (the complete 100% rMT "
            "pre/post-ccPAS probe design) + resting run-01 + the T1w. The ccPAS induction "
            "run, spTMS 07-12, resting 02-04 and the 4/9 ms sessions exist upstream but "
            "were not fetched (1021 GiB full release)"
        ),
        loader_ref="bids_eeg:load_brainvision_run",
        notes="Perturbation benchmark: single-pulse TMS with EEG and digitised electrodes.",
    ),
    DatasetEntry(
        dataset_id="ds000117",
        version="1.1.0",
        name="Wakeman-Henson multimodal face-perception (MEG+EEG+fMRI+MRI)",
        role="likelihood",
        modalities=("meg", "eeg", "fmri", "mri", "dwi"),
        card_name="ds000117.yaml",
        # Everything comes from the *pinned snapshot*, never from the
        # s3://openneuro.org mirror: that mirror serves the mutable draft,
        # whose dataset_description.json and README match no published
        # snapshot (checked 2026-08-05).  Appendix B rejects an unversioned
        # mutable download, so the mirror is not used as a source of record.
        fetcher=OpenNeuroSnapshotFetcher(
            dataset_id="ds000117",
            accession="ds000117",
            tag="1.1.0",
            relpaths=_read_subset("ds000117__1.1.0.tsv"),
        ),
        subset=(
            "all 16 subjects' MRI/fMRI/dMRI/events/headshape from the S3 mirror; "
            "MEG+EEG raw .fif for sub-01 and sub-02 (runs 01-06) only"
        ),
        loader_ref="bids_meg:load_fif_run",
        notes="Cross-modal benchmark: simultaneous MEG+EEG on the same subjects and task.",
    ),
    # ---- optional -----------------------------------------------------
    DatasetEntry(
        dataset_id="things-eeg2",
        version="unknown",
        name="THINGS-EEG2 rapid serial visual presentation EEG",
        role="likelihood",
        modalities=("eeg",),
        card_name="things-eeg2.yaml",
        fetcher=None,
        notes="See card: OSF availability check result recorded in governance.",
    ),
    # ---- deliberately not downloaded (DUA / credentials) --------------
    DatasetEntry(
        dataset_id="ukbiobank-brain-imaging",
        version="unknown",
        name="UK Biobank brain imaging",
        role="prior",
        modalities=("mri", "fmri", "dmri"),
        card_name="ukbiobank-brain-imaging.yaml",
        fetcher=UnavailableFetcher(
            dataset_id="ukbiobank-brain-imaging",
            reason="Material Transfer Agreement + approved application required",
        ),
    ),
    DatasetEntry(
        dataset_id="hcp-young-adult",
        version="unknown",
        name="Human Connectome Project Young Adult (1200 subjects)",
        role="prior",
        modalities=("mri", "dmri", "fmri", "meg"),
        card_name="hcp-young-adult.yaml",
        fetcher=UnavailableFetcher(
            dataset_id="hcp-young-adult",
            reason="Open Access Data Use Terms must be accepted per user; restricted tier needs a DUA",
        ),
    ),
    DatasetEntry(
        dataset_id="tuh-eeg",
        version="unknown",
        name="Temple University Hospital EEG Corpus",
        role="likelihood",
        modalities=("eeg",),
        card_name="tuh-eeg.yaml",
        fetcher=UnavailableFetcher(
            dataset_id="tuh-eeg",
            reason="signed data use agreement and issued credentials required",
        ),
    ),
    DatasetEntry(
        dataset_id="adni",
        version="unknown",
        name="Alzheimer's Disease Neuroimaging Initiative",
        role="prior",
        modalities=("mri", "pet"),
        card_name="adni.yaml",
        fetcher=UnavailableFetcher(
            dataset_id="adni", reason="application review and DUA required via LONI/IDA"
        ),
    ),
    DatasetEntry(
        dataset_id="ram-intracranial",
        version="unknown",
        name="DARPA Restoring Active Memory intracranial EEG",
        role="likelihood",
        modalities=("ieeg", "stimulation"),
        card_name="ram-intracranial.yaml",
        fetcher=UnavailableFetcher(
            dataset_id="ram-intracranial",
            reason="registration and data use agreement required; intracranial human data",
        ),
    ),
]

REGISTRY: Mapping[str, DatasetEntry] = {e.dataset_id: e for e in _ENTRIES}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def get(dataset_id: str) -> DatasetEntry:
    if dataset_id not in REGISTRY:
        raise KeyError(f"unknown dataset {dataset_id!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[dataset_id]


def iter_entries(*, role: str | None = None, modality: str | None = None) -> Iterator[DatasetEntry]:
    for e in REGISTRY.values():
        if role is not None and e.role != role:
            continue
        if modality is not None and modality not in e.modalities:
            continue
        yield e


def live_datasets() -> list[DatasetEntry]:
    return [e for e in REGISTRY.values() if e.status()[0] in ("live", "partial")]


def unavailable_datasets() -> list[tuple[DatasetEntry, str]]:
    out = []
    for e in REGISTRY.values():
        st, reason = e.status()
        if st == "unavailable":
            out.append((e, reason))
    return out


def inventory() -> list[dict[str, Any]]:
    """Machine-readable inventory used by ``reports/data_inventory.md``."""
    rows: list[dict[str, Any]] = []
    for e in REGISTRY.values():
        st, reason = e.status()
        row: dict[str, Any] = {
            "dataset_id": e.dataset_id,
            "version": e.version,
            "name": e.name,
            "role": e.role,
            "modalities": list(e.modalities),
            "status": st,
            "reason": reason,
            "subset": e.subset,
            "bytes_on_disk": e.on_disk_bytes(),
            "local_path": str(e.local_path),
            "card": str(e.card_path),
        }
        try:
            card = e.card()
            row["license"] = card.data["governance"]["license"]
            row["n_participants"] = card.data["population"]["n_participants"]
            row["gradient"] = card.effective_gradient_permission()
        except Exception as exc:  # pragma: no cover - card errors surface in tests
            row["card_error"] = str(exc)
        rows.append(row)
    return rows


__all__ = (
    "DatasetEntry",
    "REGISTRY",
    "data_root",
    "get",
    "inventory",
    "iter_entries",
    "live_datasets",
    "unavailable_datasets",
)
