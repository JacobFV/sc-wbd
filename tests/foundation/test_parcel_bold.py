"""The coverage mask and the NaN pattern must agree, or an imputation is loose.

This is the one invariant that keeps a fabricated observation out of the BOLD
path. A parcel outside the acquisition's field of view has no measurement; it is
``NaN`` in the timeseries and ``False`` in the mask, and the two statements have
to be the same statement. If they ever drift apart:

* a parcel ``NaN`` but marked covered enters the likelihood as ``nan_to_num``'s
  zero and is scored against the model — a **fabricated measurement of zero**;
* a parcel finite but marked uncovered is silently discarded — real signal
  thrown away, which is merely wasteful.

The first is the one that matters. Once it is in the array nothing downstream
can distinguish it from a real zero, which is why this is asserted about the
cached artifacts themselves rather than about the function that writes them.

Coverage is genuinely per-subject: measured 399/400 for sub-xp101 and 379/400
for sub-xp102 on the same task, from field-of-view differences alone. A global
mask would be wrong for one of them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/foundation_cache/parcel_bold"


def _cached() -> list[tuple[Path, Path]]:
    if not CACHE.is_dir():
        return []
    out = []
    for npy in sorted(CACHE.glob("*.npy")):
        js = npy.with_suffix(".json")
        if js.exists():
            out.append((npy, js))
    return out


def test_nan_rows_equal_uncovered_parcels():
    """The invariant. Asserted on every cached run, not on a synthetic one."""
    pairs = _cached()
    if not pairs:
        pytest.skip("no parcellated BOLD cached yet")
    for npy, js in pairs:
        ts = np.load(npy, mmap_mode="r")
        meta = json.loads(js.read_text())
        covered = np.asarray(meta["covered"], dtype=bool)
        assert ts.shape[0] == covered.size, f"{npy.name}: {ts.shape[0]} rows, {covered.size} parcels"
        all_nan = np.isnan(np.asarray(ts)).all(axis=1)
        assert int(all_nan.sum()) == int((~covered).sum()), (
            f"{npy.name}: {int(all_nan.sum())} all-NaN rows but "
            f"{int((~covered).sum())} uncovered parcels -- the mask and the data disagree"
        )
        assert np.array_equal(all_nan, ~covered), (
            f"{npy.name}: the same COUNT of NaN rows and uncovered parcels, but not "
            "the same parcels. A count check would have passed this."
        )


def test_covered_parcels_carry_finite_signal():
    """A covered parcel with no finite value is a mask that lies the other way."""
    pairs = _cached()
    if not pairs:
        pytest.skip("no parcellated BOLD cached yet")
    for npy, js in pairs:
        ts = np.asarray(np.load(npy, mmap_mode="r"))
        covered = np.asarray(json.loads(js.read_text())["covered"], dtype=bool)
        finite_any = np.isfinite(ts).any(axis=1)
        bad = np.where(covered & ~finite_any)[0]
        assert bad.size == 0, f"{npy.name}: parcels {bad[:5].tolist()} marked covered but all non-finite"


def test_coverage_is_per_run_not_global():
    """Two subjects, same task, different coverage -- so a global mask is wrong.

    Skips rather than passing vacuously when only one run is cached: with a
    single run this property is untestable, and a test that cannot fail on one
    input should say so instead of going green.
    """
    pairs = _cached()
    if len(pairs) < 2:
        pytest.skip("need at least two cached runs to compare coverage")
    fracs = {
        js.stem: float(np.asarray(json.loads(js.read_text())["covered"], dtype=bool).mean())
        for _, js in pairs
    }
    assert len(set(round(v, 6) for v in fracs.values())) > 1 or len(fracs) == 1, (
        "every cached run has identical coverage; either the field of view really is "
        f"identical or the mask is not per-run: {fracs}"
    )


def test_the_metadata_records_unobserved_are_nan():
    """The artifact must say so itself, not rely on this file to remember."""
    pairs = _cached()
    if not pairs:
        pytest.skip("no parcellated BOLD cached yet")
    for _, js in pairs:
        meta = json.loads(js.read_text())
        assert meta.get("unobserved_are_nan") is True, f"{js.name} does not declare the convention"
        assert "coverage_note" in meta.get("provenance", {}), f"{js.name} carries no coverage note"
