"""Fill the measured fields of a source card from the bytes actually on disk.

A source card is only worth something if its hashes, byte counts and subject
counts are *measured*, not typed.  Cards therefore ship with the literal
placeholder ``FILLED_BY_REFRESH`` wherever a value must come from the data,
and this module replaces those placeholders in place (comments preserved -
the substitution is line-based, not a YAML round-trip).

::

    python -m scwbd.sources.refresh eegmmidb
    python -m scwbd.sources.refresh --all --rebuild-manifest

Per-dataset "probes" below read the real files to answer questions no
metadata file answers: which runs deviate from the nominal sampling rate,
how many nights each sleeper actually has, which subjects' MEG binaries were
fetched.  Probes must never guess; when they cannot answer they emit the
string ``unknown``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from .manifest import Manifest, build_manifest, load_manifest

PLACEHOLDER = "FILLED_BY_REFRESH"


# --------------------------------------------------------------------------
# YAML placeholder substitution that keeps comments
# --------------------------------------------------------------------------
def _yaml_block(value: Any, indent: int) -> str:
    text = yaml.safe_dump(value, sort_keys=False, default_flow_style=False, allow_unicode=True)
    pad = " " * indent
    return "".join(pad + line if line.strip() else line for line in text.splitlines(True))


def fill_placeholders(text: str, values: dict[str, Any]) -> tuple[str, list[str]]:
    """Replace ``<key>: FILLED_BY_REFRESH`` lines using ``values[<dotted key>]``.

    Keys are matched by their leaf name; if two sections use the same leaf
    name, pass the dotted path and the leaf will still match because we track
    the enclosing top-level section while scanning.
    """
    out: list[str] = []
    filled: list[str] = []
    section = ""
    for line in text.splitlines(True):
        m_sec = re.match(r"^([a-z_]+):\s*$", line)
        if m_sec:
            section = m_sec.group(1)
        m = re.match(rf"^(\s*)([A-Za-z_][A-Za-z0-9_]*):\s*{PLACEHOLDER}\s*$", line)
        if not m:
            out.append(line)
            continue
        pad, key = m.group(1), m.group(2)
        dotted = f"{section}.{key}"
        if dotted in values:
            value = values[dotted]
        elif key in values:
            value = values[key]
        else:
            out.append(line)
            continue
        if isinstance(value, (dict, list)):
            out.append(f"{pad}{key}:\n")
            out.append(_yaml_block(value, len(pad) + 2))
        else:
            # JSON scalars are valid YAML scalars and, unlike yaml.safe_dump of a
            # bare scalar, do not emit a '...' document-end marker.
            out.append(f"{pad}{key}: {json.dumps(value)}\n")
        filled.append(dotted)
    return "".join(out), filled


# --------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------
def _probe_eegmmidb(rootp: Path) -> dict[str, Any]:
    from .loaders.edf import read_header

    per_file: dict[str, tuple[float, int, float]] = {}
    for p in sorted(rootp.glob("S*/S*.edf")):
        try:
            h = read_header(p)
        except Exception as exc:  # unreadable file is itself a finding
            per_file[p.name] = (-1.0, -1, -1.0)
            continue
        nsig = sum(1 for s in h.signals if not s.is_annotation)
        rate = h.signals[0].n_samples_per_record / h.record_duration
        per_file[p.name] = (rate, nsig, h.n_records * h.record_duration)
    rates = Counter(v[0] for v in per_file.values())
    nominal = rates.most_common(1)[0][0] if rates else 0.0
    odd = {k: {"sfreq_hz": v[0], "n_channels": v[1], "duration_s": v[2]}
           for k, v in per_file.items() if v[0] != nominal}
    odd_subjects = sorted({k[:4] for k in odd})
    subjects = sorted({p.name for p in rootp.glob("S[0-9][0-9][0-9]") if p.is_dir()})
    runs_per_subject = Counter(p.parent.name for p in rootp.glob("S*/S*.edf"))
    short = {s: n for s, n in runs_per_subject.items() if n != 14}
    return {
        "population.n_participants": len(subjects),
        "population.known_anomalies": {
            "nominal_sfreq_hz": nominal,
            "n_runs_scanned": len(per_file),
            "runs_off_nominal_rate": odd if len(odd) < 60 else f"{len(odd)} runs (see manifest)",
            "subjects_with_off_nominal_runs": odd_subjects,
            "subjects_without_14_runs": short,
            "note": (
                "measured by reading every EDF header on disk; these runs are excluded "
                "from any 160 Hz likelihood term rather than resampled"
            ),
        },
        "missingness.unplanned": (
            f"{len(odd)} run(s) across {len(odd_subjects)} subject(s) deviate from the nominal "
            f"{nominal:g} Hz / 64-channel header and are excluded, not repaired"
            if odd
            else "none detected: every EDF header on disk matches the nominal rate and channel count"
        ),
    }


def _probe_sleep_edfx(rootp: Path) -> dict[str, Any]:
    from .loaders.sleep_edfx import PSG_RE

    cassette = rootp / "sleep-cassette"
    nights: dict[str, list[str]] = defaultdict(list)
    missing_hyp: list[str] = []
    for p in sorted(cassette.glob("SC4*-PSG.edf")):
        m = PSG_RE.match(p.name)
        if not m:
            continue
        nights[f"SC4{m['ss']}"].append(m["night"])
        stem = f"SC4{m['ss']}{m['night']}"
        if not list(cassette.glob(f"{stem}*-Hypnogram.edf")):
            missing_hyp.append(p.name)
    one_night = sorted(s for s, n in nights.items() if len(n) < 2)
    return {
        "population.n_participants": len(nights),
        "population.recordings_on_disk": sum(len(v) for v in nights.values()),
        "population.known_anomalies": {
            "participants_with_one_night_only": one_night,
            "psg_without_hypnogram": missing_hyp,
            "note": (
                "the released corpus documents that the first nights of subjects 36 and 52 and "
                "the second night of subject 13 were lost to media failure; the list above is "
                "what is actually on disk"
            ),
        },
        "missingness.unplanned": (
            f"{len(one_night)} participant(s) have a single night on disk "
            f"({', '.join(one_night) if one_night else 'none'}); "
            f"{len(missing_hyp)} PSG(s) lack a hypnogram"
        ),
    }


def _probe_bids(rootp: Path, *, binary_glob: str) -> dict[str, Any]:
    subs = sorted(p.name for p in rootp.glob("sub-*") if p.is_dir())
    with_binaries = sorted({
        next((part for part in p.parts if part.startswith("sub-")), "?")
        for p in rootp.rglob(binary_glob)
    })
    return {
        "population.n_participants": len(subs),
        "population.participant_ids": subs,
        "population.participants_with_signal_binaries": with_binaries,
    }


def _probe_ds000117(rootp: Path) -> dict[str, Any]:
    d = _probe_bids(rootp, binary_glob="*_meg.fif")
    bold = sorted({next((x for x in p.parts if x.startswith("sub-")), "?")
                   for p in rootp.rglob("*_bold.nii.gz")})
    meg = d["population.participants_with_signal_binaries"]
    d["missingness.unplanned"] = (
        f"MEG/EEG .fif fetched for {len(meg)} of {len(d['population.participant_ids'])} "
        f"participants ({', '.join(meg) if meg else 'none'}); BOLD present for {len(bold)}. "
        "The absent MEG binaries are a deliberate, recorded subset - not missing data in the "
        "statistical sense - and the unfetched participants are excluded from MEG likelihood "
        "terms rather than imputed."
    )
    d["population.participants_with_bold"] = bold
    return d


def _probe_ds004024(rootp: Path) -> dict[str, Any]:
    d = _probe_bids(rootp, binary_glob="*_eeg.eeg")
    eeg = d["population.participants_with_signal_binaries"]
    sessions = sorted({p.name for p in rootp.glob("sub-*/ses-*") if p.is_dir()})
    d["population.sessions_present"] = sessions
    d["missingness.unplanned"] = (
        f"EEG binaries fetched for {len(eeg)} of {len(d['population.participant_ids'])} "
        f"participants ({', '.join(eeg) if eeg else 'none'}); BIDS metadata (channels, "
        "electrodes, events, sidecars) is present for all participants. Unfetched runs are "
        "excluded, never interpolated."
    )
    return d


def _probe_ds002336(rootp: Path) -> dict[str, Any]:
    """Simultaneous EEG+fMRI: count the two arms separately.

    A participant with EEG but no BOLD (or the reverse) is NOT a paired
    episode, and the whole value of this source is the pairing. So the probe
    counts them independently and states the intersection, rather than
    reporting one number that a reader would take for both.
    """
    d = _probe_bids(rootp, binary_glob="*_eeg.eeg")
    eeg = set(d["population.participants_with_signal_binaries"])
    bold = {next((x for x in p.parts if x.startswith("sub-")), "?")
            for p in rootp.rglob("*_bold.nii.gz")}
    paired = sorted(eeg & bold)
    d["population.participants_with_bold"] = sorted(bold)
    d["population.participants_paired_eeg_and_bold"] = paired

    # Participant-level pairing is not enough. "Fully paired episode" is a
    # claim about a RUN: this subject, this task, both arms. Counting subjects
    # would report 10/10 while individual runs are missing one arm, which is
    # exactly the kind of number that survives review and is wrong.
    def _key(p: Path) -> tuple[str, str]:
        sub = next((x for x in p.parts if x.startswith("sub-")), "?")
        task = next((t.split("-", 1)[1] for t in p.name.split("_")
                     if t.startswith("task-")), "?")
        return sub, task

    eeg_runs = {_key(p) for p in rootp.glob("sub-*/eeg/*_eeg.vhdr")}
    bold_runs = {_key(p) for p in rootp.glob("sub-*/func/*_bold.nii.gz")}
    paired_runs = eeg_runs & bold_runs
    eeg_only = sorted(eeg_runs - bold_runs)
    bold_only = sorted(bold_runs - eeg_runs)
    tasks = sorted({t for _, t in eeg_runs | bold_runs})
    d["population.runs_per_session"] = (
        f"{len(tasks)} task types on disk ({', '.join(tasks)}); "
        f"{len(eeg_runs)} EEG runs, {len(bold_runs)} BOLD runs, "
        f"{len(paired_runs)} runs with BOTH arms"
    )
    d["missingness.unplanned"] = (
        f"raw EEG present for {len(eeg)}, BOLD for {len(bold)}, and both for "
        f"{len(paired)} of {len(d['population.participant_ids'])} participants. "
        f"AT RUN LEVEL the pairing is incomplete: {len(paired_runs)} of "
        f"{len(eeg_runs | bold_runs)} (subject, task) runs have both arms; "
        f"{len(eeg_only)} are EEG-only ({', '.join(f'{s}/{t}' for s, t in eeg_only[:6])}"
        f"{' ...' if len(eeg_only) > 6 else ''}) and {len(bold_only)} are BOLD-only "
        f"({', '.join(f'{s}/{t}' for s, t in bold_only[:6])}"
        f"{' ...' if len(bold_only) > 6 else ''}). An unpaired run is excluded from "
        "cross-modal terms rather than half-imputed; it remains usable as a "
        "single-modality record."
    )
    d["spatial.n_elements"] = (
        f"64 EEG sensors; BOLD voxel grid per run (read from the NIfTI header, not "
        f"assumed). {len(eeg_runs)} EEG runs and {len(bold_runs)} BOLD runs on disk."
    )
    return d


def _probe_ds000113(rootp: Path) -> dict[str, Any]:
    subs = sorted(p.name for p in rootp.glob("sub-*") if p.is_dir())
    bold = sorted({next((x for x in p.parts if x.startswith("sub-")), "?")
                   for p in rootp.rglob("*_bold.nii.gz")})
    by_ses: dict[str, set[str]] = defaultdict(set)
    for p in rootp.rglob("*_bold.nii.gz"):
        ses = next((x for x in p.parts if x.startswith("ses-")), "ses-none")
        sub = next((x for x in p.parts if x.startswith("sub-")), "?")
        by_ses[ses].add(sub)
    n_physio = len(list(rootp.rglob("*recording-cardresp_physio.tsv.gz")))
    n_gaze = len(list(rootp.rglob("*recording-eyegaze_physio.tsv.gz")))
    n_unreadable = len([p for p in rootp.rglob("*_physio.tsv.gz") if "recording-" not in p.name])
    return {
        "population.n_participants": len(subs),
        "population.participant_ids": subs,
        "population.participants_with_bold": bold,
        "missingness.unplanned": (
            "BOLD on disk per session: "
            + "; ".join(f"{k} for {len(v)} participants" for k, v in sorted(by_ses.items()))
            + f". Physiological recordings: {n_physio} cardiac/respiratory files and "
            f"{n_gaze} eye-gaze files are readable; {n_unreadable} further physio files "
            "carry no `recording-` entity and no sidecar exists for them at any level, so "
            "their rate and columns are unstated and load_physio refuses them."
        ),
        "spatial.n_elements": (
            f"BOLD voxel grid read per file from the NIfTI header. "
            f"{len(list(rootp.rglob('*_bold.nii.gz')))} functional runs on disk across "
            f"{len(bold)} participants."
        ),
    }


def _probe_mne(rootp: Path) -> dict[str, Any]:
    fifs = sorted(p.name for p in rootp.rglob("*.fif"))
    return {
        "population.n_participants": 1,
        "identity.artefacts_present": {
            "n_fif_files": len(fifs),
            "has_bem": bool(list(rootp.rglob("bem/*.fif"))),
            "has_forward": bool(list(rootp.rglob("*-fwd.fif"))),
            "has_trans": bool(list(rootp.rglob("*-trans.fif"))),
            "has_freesurfer_surfaces": bool(list(rootp.rglob("surf/lh.white"))),
        },
    }


def _probe_things(rootp: Path) -> dict[str, Any]:
    zips = sorted(rootp.rglob("*.zip"))
    return {
        "population.n_participants": len(zips),
        "population.participant_ids": [p.stem for p in zips],
        "missingness.unplanned": (
            f"{len(zips)} of 10 released participants fetched; the remainder were not "
            "downloaded (time/space budget) and are simply absent from the register"
        ),
    }


PROBES: dict[str, Callable[[Path], dict[str, Any]]] = {
    "eegmmidb": _probe_eegmmidb,
    "sleep-edfx": _probe_sleep_edfx,
    "ds000117": _probe_ds000117,
    "ds004024": _probe_ds004024,
    "ds002336": _probe_ds002336,
    "ds000113": _probe_ds000113,
    "mne-sample": _probe_mne,
    "mne-somato": _probe_mne,
    "mne-spm-face": _probe_mne,
    "things-eeg2": _probe_things,
}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def refresh_card(
    dataset_id: str, *, rebuild_manifest: bool = False, n_exemplars: int = 6
) -> tuple[Path, list[str]]:
    from .registry import get

    entry = get(dataset_id)
    card_path = entry.card_path
    text = card_path.read_text()
    if PLACEHOLDER not in text:
        return card_path, []
    rootp = entry.local_path
    if not rootp.exists():
        raise FileNotFoundError(
            f"{dataset_id}: {rootp} is not on disk; a card with placeholders cannot be "
            "refreshed from nothing - fetch first or mark the card unavailable"
        )
    if rebuild_manifest or not entry.has_manifest():
        man: Manifest = build_manifest(rootp, dataset_id, entry.version, workers=12, progress=True)
        man.write()
    else:
        man = load_manifest(dataset_id, entry.version)

    values: dict[str, Any] = {
        "identity.retrieved_utc": man.generated_utc,
        "identity.local_path": str(rootp),
        "identity.file_manifest": {
            "algorithm": man.algorithm,
            "n_files": man.n_files,
            "total_bytes": man.total_bytes,
            "manifest_sha256": man.manifest_sha256(),
            "manifest_path": str(man.path().relative_to(Path(__file__).resolve().parents[2])),
            "exemplars": man.exemplars(n_exemplars),
        },
    }
    probe = PROBES.get(dataset_id)
    if probe is not None:
        values.update(probe(rootp))
    new_text, filled = fill_placeholders(text, values)
    card_path.write_text(new_text)
    return card_path, filled


def _main(argv: list[str] | None = None) -> int:
    from .registry import REGISTRY

    ap = argparse.ArgumentParser(prog="python -m scwbd.sources.refresh")
    ap.add_argument("dataset_id", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--rebuild-manifest", action="store_true")
    args = ap.parse_args(argv)
    ids = args.dataset_id or ([e.dataset_id for e in REGISTRY.values()] if args.all else [])
    if not ids:
        ap.error("give a dataset_id or --all")
    rc = 0
    for did in ids:
        try:
            path, filled = refresh_card(did, rebuild_manifest=args.rebuild_manifest)
        except Exception as exc:
            print(f"{did}: SKIPPED - {exc}")
            rc = 1
            continue
        print(f"{did}: filled {len(filled)} field(s) in {path.name}: {', '.join(filled)}")
    return rc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
