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
    #: Additional readers for a dataset that holds more than one modality in
    #: more than one container.  A simultaneous EEG+fMRI release needs two, and
    #: a single ``loader_ref`` silently makes the second modality unreachable —
    #: which is how ds000117's 18 BOLD runs sat on disk unread.
    extra_loader_refs: tuple[str, ...] = ()
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
        return self._resolve(self.loader_ref)

    @staticmethod
    def _resolve(ref: str) -> Callable[..., Any]:
        mod_name, _, fn = ref.partition(":")
        mod = importlib.import_module(f"scwbd.sources.loaders.{mod_name}")
        return getattr(mod, fn)

    def loaders(self) -> dict[str, Callable[..., Any]]:
        """Every reader registered for this dataset, keyed by its ref."""
        refs = ([self.loader_ref] if self.loader_ref else []) + list(self.extra_loader_refs)
        return {r: self._resolve(r) for r in refs}

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
        # CORRECTED 2026-08-06 (Ada): "fmri" was declared here and on the card
        # while the fetched subset holds two T1w volumes and no BOLD at all.
        # The upstream release has fMRI and diffusion; we did not fetch them,
        # and `signal.modality_evidence` in the card now has to point at a real
        # file for every entry (scwbd.sources.audit A4).
        modalities=("eeg", "eog", "emg", "ecg", "tms", "mri"),
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
        # CORRECTED 2026-08-06 (Ada). This string used to read "all 16
        # subjects' MRI/fMRI/dMRI/events/headshape from the S3 mirror".  Only
        # sub-01 and sub-02 exist under the dataset root -- re-derived by
        # walking it, not read off a report -- and the card's own subset_note
        # said so all along.  The registry and the card disagreed, and the
        # registry was the one claiming coverage.
        subset=(
            "sub-01 and sub-02 only, both complete: MEG+EEG raw .fif runs 01-06, "
            "9 BOLD runs, T1w MPRAGE, 2x7-echo FLASH, fieldmaps, dMRI, events, "
            "headshape; plus every top-level metadata file and the Maxfilter "
            "calibration pair. The remaining 14 released participants were not fetched"
        ),
        loader_ref="bids_meg:load_fif_run",
        # The 18 BOLD runs and the diffusion volumes were on disk from the
        # first fetch and had no reader; `bids_bold` is that reader.
        extra_loader_refs=("bids_bold:load_bold_run",),
        notes="Cross-modal benchmark: simultaneous MEG+EEG on the same subjects and task.",
    ),
    # ---- simultaneous EEG + fMRI: the fully paired episode (§6.4) ------
    DatasetEntry(
        dataset_id="ds002336",
        version="2.0.2",
        name="XP1 simultaneous EEG-fMRI motor-imagery neurofeedback (Lioi et al.)",
        role="likelihood",
        modalities=("eeg", "fmri", "mri"),
        card_name="ds002336.yaml",
        fetcher=OpenNeuroSnapshotFetcher(
            dataset_id="ds002336",
            accession="ds002336",
            tag="2.0.2",
            relpaths=_read_subset("ds002336__2.0.2.tsv"),
        ),
        subset=(
            "all 10 participants, complete raw arm: T1w, four raw BrainVision EEG runs "
            "(motorloc, fmriNF, eegfmriNF, eegNF) and the four matching BOLD runs, plus "
            "the gradient-artefact-corrected eeg_pp derivatives and the head-motion "
            "tsv. The NF-score .mat derivatives and the QA .tiff figures were not fetched"
        ),
        loader_ref="bids_eeg:load_brainvision_run",
        extra_loader_refs=("bids_bold:load_bold_run",),
        notes=(
            "The only source in the register where electrophysiology and haemodynamics "
            "are measured on the SAME subject at the SAME time: §6.4's fully paired "
            "episode, and the two clocks (250 Hz amplifier, 1 Hz scanner) are a real "
            "instance of §2.6 temporal non-nesting rather than a declared one."
        ),
    ),
    # ---- naturalistic / retinotopic / auditory: §6.1 regional families --
    DatasetEntry(
        dataset_id="ds000113",
        version="1.3.0",
        name="StudyForrest: audio-movie, retinotopy, object categories, movie 3T",
        role="likelihood",
        modalities=("fmri", "mri", "cardiac", "resp", "eyetrack"),
        card_name="ds000113.yaml",
        fetcher=OpenNeuroSnapshotFetcher(
            dataset_id="ds000113",
            accession="ds000113",
            tag="1.3.0",
            relpaths=_read_subset("ds000113__1.3.0.tsv"),
        ),
        subset=(
            "ses-localizer (4 retinotopic mapping runs: ccw/clw/exp/con, 4 object-category "
            "runs, 1 movie localizer) and ses-movie (8 runs of the audio-visual movie with "
            "eye-gaze physio) for the 15 participants who have them, plus T1w/T2w for all 20. "
            "ses-auditoryperception (8 runs) for sub-01..sub-04 only. NOT fetched: the 7T "
            "ses-forrestgump functional runs (~250 GB), the derivatives tree (192 GB), "
            "sourcedata, dwi, angio and the veno/SWI volumes"
        ),
        loader_ref="bids_bold:load_bold_run",
        extra_loader_refs=("bids_bold:load_physio",),
        notes=(
            "Serves four §6.1 families at once: early visual (retinotopy), auditory "
            "(speech/scene), naturalistic vision (movie + gaze), and the interoceptive "
            "interfaces (cardiac + respiratory physio recorded during every run)."
        ),
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
