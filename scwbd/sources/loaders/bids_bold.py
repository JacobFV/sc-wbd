"""BIDS haemodynamic loader: 4-D BOLD volumes and the physiological recordings
acquired alongside them.

Why this file exists
--------------------
``scwbd.sources.loaders.base.VolumeSeries`` — the container for a 4-D image
series at its native voxel grid and TR — has been defined and exported since
the loader package was written, and **nothing ever constructed one**
(``grep -rn VolumeSeries`` returns its own definition and the ``__init__``
re-export, and nothing else).  Meanwhile ``ds000117`` has 18 BOLD runs on
disk.  So the repository held real haemodynamic bytes, held a container for
them, and had no path between the two: every source that could reach the
training mixture was electrophysiological.  That is the concrete mechanism
behind "``-raw`` is EEG-only in practice" — not a policy, an absent reader.

What this loader refuses
------------------------
Three refusals, each because the silent alternative is easy:

1. **No repetition time, no recording.**  A BOLD series whose TR cannot be
   established is refused rather than defaulted to 2 s.  A wrong TR is a wrong
   clock, and a wrong clock is invisible downstream — the array still has the
   right shape.
2. **The sidecar and the header must agree.**  ``RepetitionTime`` in the JSON
   and ``pixdim[4]`` in the NIfTI header are two independent statements of the
   same number.  When they disagree by more than a millisecond the loader
   raises instead of silently preferring one.  (They do disagree in the wild;
   see the tests.)
3. **No resampling, ever.**  Inherited from :class:`VolumeSeries` — thesis
   §2.6.  A 0.5 Hz BOLD series and a 250 Hz EEG recording from the *same*
   simultaneous session (ds002336) stay on their own clocks and are combined,
   if at all, through the multirate machinery.  ``clock_id`` is therefore
   scanner-specific and never equal to an amplifier clock, so a naive
   concatenation cannot typecheck.

Memory
------
A 4-D run is large: ds000113's movie runs are ~65 MB gzipped and ~450 MB as
float32.  ``load_bold_run`` therefore defaults to ``volumes=None`` meaning
*header only* — ``data`` is an empty array and ``meta["shape"]`` carries the
real shape.  Pass ``volumes=slice(...)`` or ``volumes="all"`` to materialise
voxels, and be explicit about it.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from ..lineage import Lineage
from .base import NativeRecording, NativeSupportError, VolumeSeries

__all__ = [
    "BoldReadError",
    "bold_sidecar",
    "frame_from_nifti",
    "iter_bold",
    "load_bold_run",
    "load_physio",
]

#: NIfTI ``sform_code``/``qform_code`` -> the coordinate frame it names.
#: Code 0 means "no frame stated"; it is mapped to ``"unknown"`` and never to a
#: plausible-sounding default, because a wrong frame silently misplaces every
#: voxel a lead field or a parcellation would look up.
NIFTI_XFORM_FRAMES: dict[int, str] = {
    0: "unknown",
    1: "scanner_anat_RAS",
    2: "aligned_anat_RAS",
    3: "talairach_RAS",
    4: "mni152_RAS",
    5: "template_other_RAS",
}

#: Tolerance for the sidecar/header TR agreement check, seconds.
TR_TOLERANCE_S = 1e-3


class BoldReadError(RuntimeError):
    """Raised when a BOLD run cannot be read without inventing a fact."""


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------
def iter_bold(data_root: str | Path, *, include_derivatives: bool = False) -> Iterator[Path]:
    """Every raw ``*_bold.nii.gz`` under ``data_root``, in sorted order."""
    base = Path(data_root)
    if not base.exists():
        return
    for p in sorted(base.rglob("*_bold.nii.gz")):
        if not include_derivatives and "derivatives" in p.parts:
            continue
        yield p


def _entities(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in path.name.split("_"):
        if "-" in token:
            k, _, v = token.partition("-")
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# BIDS sidecar resolution
# ---------------------------------------------------------------------------
def bold_sidecar(path: str | Path) -> dict[str, Any]:
    """Resolve a BIDS JSON sidecar by the inheritance principle.

    BIDS lets a field be stated once at the dataset root and inherited by every
    run below it (ds000117 does exactly this: one
    ``task-facerecognition_bold.json`` for 18 runs).  Walking up is therefore
    not an optimisation, it is the only way to find the TR at all.  Nearer
    files win, which is what the standard specifies.
    """
    p = Path(path)
    stem = p.name
    for suf in (".nii.gz", ".nii", ".tsv.gz", ".tsv"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    # Progressively drop leading entities: sub-01_ses-mri_task-x_run-01_bold
    # matches sub-01_ses-mri_task-x_run-01_bold.json, then task-x_bold.json ...
    tokens = stem.split("_")
    candidates: list[str] = []
    for i in range(len(tokens)):
        candidates.append("_".join(tokens[i:]) + ".json")
    merged: dict[str, Any] = {}
    # Root-most directory first so that nearer files overwrite it.
    for d in list(p.parents)[::-1]:
        for cand in candidates:
            f = d / cand
            if f.is_file():
                try:
                    merged.update(json.loads(f.read_text()))
                except (OSError, json.JSONDecodeError) as exc:
                    raise BoldReadError(f"{f}: unreadable BIDS sidecar ({exc})") from exc
                merged.setdefault("_sidecars", [])
                merged["_sidecars"] = list(merged.get("_sidecars", [])) + [str(f)]
    return merged


def frame_from_nifti(header: Any) -> tuple[str, str]:
    """``(frame_id, which_form)`` from the NIfTI transform codes.

    sform is preferred when both are set, matching nibabel's ``get_best_affine``.
    """
    try:
        s_code = int(header["sform_code"])
        q_code = int(header["qform_code"])
    except Exception:  # pragma: no cover - exotic headers
        return "unknown", "none"
    if s_code:
        return NIFTI_XFORM_FRAMES.get(s_code, "unknown"), "sform"
    if q_code:
        return NIFTI_XFORM_FRAMES.get(q_code, "unknown"), "qform"
    return "unknown", "none"


# ---------------------------------------------------------------------------
# the loader
# ---------------------------------------------------------------------------
def load_bold_run(
    path: str | Path,
    *,
    source_id: str,
    volumes: slice | str | None = None,
) -> VolumeSeries:
    """Load one BOLD run at its native voxel grid and TR.

    Parameters
    ----------
    volumes
        ``None`` (default) reads the header only and leaves ``data`` empty —
        the honest default for a 450 MB array.  ``"all"`` materialises every
        volume.  A ``slice`` materialises that slice of the time axis.
    """
    import nibabel as nib

    p = Path(path)
    if not p.is_file():
        raise BoldReadError(f"{p}: no such BOLD file")
    img = nib.load(p)
    hdr = img.header
    shape = tuple(int(x) for x in img.shape)
    if len(shape) != 4:
        raise BoldReadError(
            f"{p}: expected a 4-D series, got shape {shape}. A 3-D volume is "
            f"anatomy, not a time series, and must not be given a TR."
        )

    side = bold_sidecar(p)
    tr_json = side.get("RepetitionTime")
    tr_hdr: float | None = None
    try:
        zooms = hdr.get_zooms()
        if len(zooms) >= 4 and float(zooms[3]) > 0:
            unit = hdr.get_xyzt_units()[1] if hasattr(hdr, "get_xyzt_units") else "sec"
            scale = {"sec": 1.0, "msec": 1e-3, "usec": 1e-6}.get(str(unit), None)
            if scale is not None:
                tr_hdr = float(zooms[3]) * scale
    except Exception:  # pragma: no cover - exotic headers
        tr_hdr = None

    if tr_json is None and tr_hdr is None:
        raise BoldReadError(
            f"{p}: no RepetitionTime in any BIDS sidecar and no usable pixdim[4] "
            f"in the NIfTI header. Refusing to assume a TR: a wrong clock is "
            f"invisible downstream because the array shape stays correct."
        )
    if tr_json is not None and tr_hdr is not None:
        if abs(float(tr_json) - tr_hdr) > TR_TOLERANCE_S:
            raise BoldReadError(
                f"{p}: sidecar RepetitionTime={float(tr_json)} s disagrees with the "
                f"NIfTI header pixdim[4]={tr_hdr} s by more than {TR_TOLERANCE_S} s. "
                f"Two independent statements of the same number disagree; pick one "
                f"deliberately rather than letting the loader choose."
            )
    tr = float(tr_json) if tr_json is not None else float(tr_hdr)  # type: ignore[arg-type]

    frame_id, which_form = frame_from_nifti(hdr)
    ent = _entities(p)
    participant = ent.get("sub", "unknown")
    session = ent.get("ses")
    run = ent.get("run")

    if volumes is None:
        data = np.empty((0, 0, 0, 0), dtype=np.float32)
        n_loaded = 0
    else:
        sl = slice(None) if volumes == "all" else volumes
        if not isinstance(sl, slice):
            raise TypeError("volumes must be None, 'all' or a slice")
        data = np.asanyarray(img.dataobj[..., sl], dtype=np.float32)
        n_loaded = int(data.shape[-1])

    lineage = Lineage(
        participant=participant,
        family=f"singleton:{participant}",
        site=f"{source_id}:site-1",
        device=str(side.get("ManufacturersModelName") or side.get("Manufacturer") or "unknown"),
        session=session or "unknown",
        run=run,
        extra={"record_id": p.name, "source_id": source_id},
    )
    return VolumeSeries(
        source_id=source_id,
        data=data,
        # Scanner-native BOLD intensity. NOT calibrated, NOT %-signal-change,
        # and not comparable across runs without a normalisation that this
        # loader deliberately does not apply.
        units="arbitrary BOLD units (scanner-native, uncalibrated)",
        tr=tr,
        affine=np.asarray(img.affine, dtype=np.float64),
        frame_id=frame_id,
        # A scanner clock, never an amplifier clock. ds002336 records EEG and
        # BOLD simultaneously; giving them the same clock_id would let a
        # downstream concat look legal when it is a resample in disguise.
        clock_id=f"mri_scanner:{source_id}:{participant}:{session or 'nosession'}",
        lineage=lineage,
        meta={
            "path": str(p),
            "shape": shape,
            "n_volumes": shape[3],
            "n_volumes_loaded": n_loaded,
            "duration_s": shape[3] * tr,
            "sfreq_hz": 1.0 / tr,
            "tr_source": "sidecar" if tr_json is not None else "nifti_header",
            "tr_sidecar": None if tr_json is None else float(tr_json),
            "tr_header": tr_hdr,
            "frame_from": which_form,
            "sidecars": side.get("_sidecars", []),
            "task": ent.get("task"),
            "echo_time": side.get("EchoTime"),
            "flip_angle": side.get("FlipAngle"),
            "slice_timing_present": "SliceTiming" in side,
            "entities": ent,
        },
    )


# ---------------------------------------------------------------------------
# physiological recordings acquired with the scan
# ---------------------------------------------------------------------------
def load_physio(path: str | Path, *, source_id: str) -> NativeRecording:
    """Load a BIDS ``*_physio.tsv.gz`` at its own native rate.

    These are the interoceptive channels §6.1 asks for — cardiac and
    respiratory traces recorded during the scan — and they are on a *third*
    clock, neither the scanner's nor an amplifier's, sampled far faster than
    the TR.  ds000113 ships them for every functional run.

    The sidecar is mandatory: a physio TSV has no header row, so ``Columns``
    and ``SamplingFrequency`` are the only statement of what the columns are
    and how fast they were sampled.  Without it the file is an anonymous
    matrix, and this loader says so rather than numbering the columns.
    """
    p = Path(path)
    if not p.is_file():
        raise BoldReadError(f"{p}: no such physio file")
    side = bold_sidecar(p)
    cols = side.get("Columns")
    sfreq = side.get("SamplingFrequency")
    if not cols or not sfreq:
        raise BoldReadError(
            f"{p}: BIDS physio sidecar missing 'Columns' and/or 'SamplingFrequency' "
            f"(found keys: {sorted(k for k in side if not k.startswith('_'))}). A "
            f"physio TSV carries no header row, so without these the columns are "
            f"unnamed and the rate is unknown."
        )
    opener = gzip.open if p.name.endswith(".gz") else open
    with opener(p, "rt") as fh:  # type: ignore[operator]
        arr = np.loadtxt(fh, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[1] != len(cols):
        raise BoldReadError(
            f"{p}: sidecar names {len(cols)} columns {list(cols)} but the file has "
            f"{arr.shape[1]}"
        )
    ent = _entities(p)
    participant = ent.get("sub", "unknown")
    names = tuple(str(c) for c in cols)
    types = tuple(_physio_type(c) for c in names)
    return NativeRecording(
        source_id=source_id,
        data=arr.T.astype(np.float64),
        # Physio traces are recorded in amplifier counts; BIDS states no unit
        # and inventing "V" would be a calibration claim nobody made.
        units=tuple("arbitrary units (physio amplifier, uncalibrated)" for _ in names),
        sfreq=float(sfreq),
        channel_names=names,
        channel_types=types,
        frame_id="not_applicable",
        clock_id=f"physio_amp:{source_id}:{participant}",
        lineage=Lineage(
            participant=participant,
            family=f"singleton:{participant}",
            site=f"{source_id}:site-1",
            device="physio_amplifier",
            session=ent.get("ses") or "unknown",
            run=ent.get("run"),
            extra={"record_id": p.name, "source_id": source_id},
        ),
        t0=float(side.get("StartTime", 0.0)),
        meta={"path": str(p), "sidecars": side.get("_sidecars", []),
              "recording": ent.get("recording"), "task": ent.get("task")},
    )


def _physio_type(column: str) -> str:
    c = column.strip().lower()
    if "cardiac" in c or "pulse" in c or "ppg" in c:
        return "cardiac"
    if "respirat" in c or "resp" in c:
        return "resp"
    if c.startswith("x") or c.startswith("y") or "pupil" in c or "gaze" in c:
        return "eyetrack"
    if "trigger" in c:
        return "stim"
    return "misc"


def resample(*_a: Any, **_k: Any) -> None:
    """Module-level guard: there is no resampling entry point here either."""
    raise NativeSupportError(
        "BOLD is kept at its native TR (thesis §2.6). Combine with faster "
        "modalities through the multirate machinery, not by resampling."
    )
