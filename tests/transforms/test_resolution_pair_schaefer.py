"""The declared pair describes the parcellation the model runs on.

For two runs it did not.  ``reports/transforms/resolution_pair.json`` recorded a
real, validated pair -- ``R P = I`` to 4.4e-16, 94% landmark coverage -- against
the 68-parcel Desikan-Killiany atlas, while ``scwbd.foundation`` keeps regional
state on 414: Schaefer400x7 plus 14 Aseg14T subcortical volumes.  Nothing was
wrong with the measurement.  It was about a different brain parcellation than
the model's, and the quantity it exists to establish moves by a factor of six
between them (0.056 of the whitened lead field against 0.321).

Three things are guarded here, and each of them, if it broke, would leave a
plausible-looking number in place rather than an error:

1. **The declared artefact is the model's parcellation.**  A pair measured on
   some other atlas is evidence about that atlas.
2. **The fsaverage5 nesting the label transfer rests on.**  The Schaefer annot
   is an fsaverage5 object and is read as a label on ``fsaverage``'s first 10242
   sphere vertices.  That identity is exact -- FreeSurfer's icosahedra are
   nested -- but nothing in the file format says so.  Take the wrong 10242 and
   the parcellation becomes spatial noise, and noise still restricts, still
   satisfies ``R P = I``, and still reports an energy fraction.
3. **The two artefacts stay separable.**  They differ by 6x, they are generated
   by the same script, and ISSUE-010 was four rounds of one output landing on
   another's path.

Made to fail on purpose
-----------------------
Five mutations, all under ``PYTHONDONTWRITEBYTECODE=1``, in a window with no
other pytest process in this repository (three were running, all in
``~/Documents/formal-language-cirriculum``; checked via ``/proc/<pid>/cwd``):

===============================================  ==================================
mutation                                         caught by
===============================================  ==================================
``fsaverage5_vertices``: ``[:N]`` -> ``[-N:]``   contiguity, both hemispheres
``FSAVERAGE5_N_VERTICES``: 10242 -> 15000        uniformity *and* contiguity, both
``DECLARED_PARCELLATION``: Schaefer400x7 -> aparc  covers-the-model, DK-is-distinct
generator's ``PARCELLATIONS``: drop Schaefer      registries-agree
artefact's ``lead_field_energy_retained`` := DK's  retains-more-than-DK
===============================================  ==================================

The first row is the one worth reading.  ``[-N:]`` takes the *last* 10242
vertices of the ico-7 sphere -- the ones added by the final subdivision -- and
those are themselves near-uniformly spaced, so the uniformity test passed on it.
Only the contiguity test caught it.  Uniformity guards the ico *level*;
contiguity guards which vertices.  Neither alone is sufficient and the sweep is
how that was discovered rather than assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scwbd.transforms import resolution_pair as rp

REPO = Path(__file__).resolve().parents[2]


def _generator():
    """The measurement script, imported as a module.

    It is a script, not a package member, and the guards below have to exercise
    the code it really runs -- the slice, the registry -- rather than a
    restatement of them.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_bench_resolution_pair", REPO / "benchmarks" / "transforms" / "resolution_pair.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: 400 Schaefer cortical parcels + 14 Aseg14T subcortical volumes.  `n_regions`
#: in `configs/run3/scwbd-003.yaml`; `ModelConfig.n_regions`'s 454 default is
#: inert and is overridden there.
MODEL_N_REGIONS = 414
MODEL_N_SUBCORTICAL = 14


# --------------------------------------------------------------------------
# 1. the declared pair is the model's
# --------------------------------------------------------------------------
def test_the_declared_measurement_covers_the_models_cortical_parcels() -> None:
    m = rp.load_measurement()
    assert m is not None, f"{rp.MEASUREMENT_RELPATH} is absent or stale"
    assert m.n_coarse == MODEL_N_REGIONS - MODEL_N_SUBCORTICAL, (
        f"the declared pair has {m.n_coarse} coarse elements; the model runs "
        f"{MODEL_N_REGIONS} regions of which {MODEL_N_SUBCORTICAL} are "
        "subcortical volumes with no cortical-surface support, so the pair "
        f"should describe {MODEL_N_REGIONS - MODEL_N_SUBCORTICAL}. A pair "
        "measured on another parcellation is evidence about that parcellation."
    )
    assert m.roundtrip_ok and m.coverage_ok and m.prolongation_calibrated


def test_the_desikan_killiany_measurement_is_still_on_disk_and_is_not_the_declared_one() -> None:
    """The 68-parcel result is kept, and kept distinct.

    It is the reference the Schaefer number is read against, and overwriting it
    would destroy the comparison that gives 0.321 its meaning.
    """
    assert rp.MEASUREMENT_RELPATHS["aparc"] != rp.MEASUREMENT_RELPATHS["Schaefer400x7"]
    dk = rp.load_measurement(rp.measurement_path(parc="aparc"))
    assert dk is not None, "the Desikan-Killiany measurement has gone"
    assert dk.n_coarse == 68
    assert dk.membership_digest != rp.load_measurement().membership_digest


def test_the_generator_and_the_loader_agree_on_which_parcellations_exist() -> None:
    """Two registries, one fact.  They drift silently otherwise."""
    bench = _generator()
    assert set(bench.PARCELLATIONS) == set(rp.MEASUREMENT_RELPATHS), (
        "the generator can build a parcellation the loader has no path for, or "
        "the loader names one nothing can produce"
    )
    for name, (n_coarse, _note) in bench.PARCELLATIONS.items():
        m = rp.load_measurement(rp.measurement_path(parc=name))
        assert m is not None, f"{name}: no artefact at {rp.MEASUREMENT_RELPATHS[name]}"
        assert m.n_coarse == n_coarse, (
            f"{name}: the artefact records n_coarse={m.n_coarse} but the "
            f"generator declares {n_coarse}"
        )


# --------------------------------------------------------------------------
# 2. the nesting the label transfer rests on
# --------------------------------------------------------------------------
SUBJECTS = (
    REPO / "data/mne-sample/processed-v6/MNE-sample-data/subjects"
)
ANNOT = REPO / "assets/src/enigma/enigmatoolbox/permutation_testing/annot"

_have_data = (SUBJECTS / "fsaverage" / "surf" / "lh.sphere").is_file() and (
    ANNOT / "fsa5_lh_schaefer_400.annot"
).is_file()
needs_data = pytest.mark.skipif(
    not _have_data, reason="fsaverage surfaces or the ENIGMA annot are not on disk"
)


@needs_data
@pytest.mark.parametrize("hemi", ["lh", "rh"])
def test_fsaverages_first_10242_vertices_are_an_icosahedron(hemi: str) -> None:
    """fsaverage5 is ico-5 and ico-5 is nested in ico-7, in order.

    The evidence is geometric and does not depend on believing the convention:
    an icosahedral subset has near-uniform nearest-neighbour spacing.  A subset
    of the same size that is *not* the nested one does not.
    """
    import nibabel as nib
    from scipy.spatial import cKDTree

    rr, _ = nib.freesurfer.read_geometry(SUBJECTS / "fsaverage" / "surf" / f"{hemi}.sphere")
    nested = _generator().fsaverage5_vertices(rr)
    d, _ = cKDTree(nested).query(nested, k=2)
    spread = float(d[:, 1].max() / d[:, 1].min())
    assert spread < 2.0, (
        f"the {len(nested)} fsaverage sphere vertices the label transfer reads "
        f"the annot onto have nearest-neighbour spacing spread {spread:.1f}; an "
        "icosahedron's is ~1.2. This subset is not fsaverage5, and a "
        "parcellation transferred through it is spatial noise that still "
        "restricts and still reports an energy fraction."
    )
    # the control: a random subset of the same size, on the same sphere
    rng = np.random.default_rng(0)
    other = rr[rng.choice(len(rr), len(nested), replace=False)]
    d2, _ = cKDTree(other).query(other, k=2)
    assert float(d2[:, 1].max() / d2[:, 1].min()) > 3.0, (
        "a random subset is as uniform as the nested one, so this test "
        "distinguishes nothing"
    )


@needs_data
@pytest.mark.parametrize("hemi", ["lh", "rh"])
def test_the_schaefer_annot_is_contiguous_on_the_nested_subset(hemi: str) -> None:
    """A parcellation read onto the right vertices is made of blobs.

    Uniform spacing alone would be satisfied by any rotation of the icosahedron.
    This is the test that the *labels* land where they belong: neighbouring
    vertices mostly share a parcel.  Under permutation they share one at chance,
    1/200, which is what the control asserts.
    """
    import nibabel as nib
    from scipy.spatial import cKDTree

    rr, _ = nib.freesurfer.read_geometry(SUBJECTS / "fsaverage" / "surf" / f"{hemi}.sphere")
    nested = _generator().fsaverage5_vertices(rr)
    lab, _, _ = nib.freesurfer.read_annot(ANNOT / f"fsa5_{hemi}_schaefer_400.annot")
    assert lab.size == len(nested), (
        f"the annot has {lab.size} vertices and the transfer targets "
        f"{len(nested)}; they are not the same surface"
    )
    _, nb = cKDTree(nested).query(nested, k=7)
    same = float((lab[nb[:, 1:]] == lab[:, None]).mean())
    assert same > 0.6, (
        f"only {same:.3f} of 6-neighbour pairs share a Schaefer parcel; a "
        "contiguous 200-per-hemisphere parcellation gives ~0.82. The annot is "
        "not aligned with these vertices."
    )
    shuffled = lab[np.random.default_rng(1).permutation(lab.size)]
    same_bad = float((shuffled[nb[:, 1:]] == shuffled[:, None]).mean())
    assert same_bad < 0.1, (
        "permuted labels look as contiguous as the real ones, so this test "
        "distinguishes nothing"
    )


# --------------------------------------------------------------------------
# 3. what the measurement says
# --------------------------------------------------------------------------
def test_the_schaefer_pair_retains_more_lead_field_than_the_dk_pair() -> None:
    """The finding, pinned so a regeneration that loses it is loud.

    Not a tolerance on a number -- an ordering, and a wide margin.  0.321
    against 0.056 is the whole reason the parcellation the pair describes
    matters, and a rebuild that quietly returned the DK figure for Schaefer
    would mean the membership never reached the restriction.
    """
    s = rp.load_measurement(rp.measurement_path(parc="Schaefer400x7"))
    dk = rp.load_measurement(rp.measurement_path(parc="aparc"))
    assert s.lead_field_energy_retained > 3.0 * dk.lead_field_energy_retained, (
        f"Schaefer400x7 retains {s.lead_field_energy_retained:.4f} against "
        f"DK-68's {dk.lead_field_energy_retained:.4f}; the measured ratio is "
        "5.7x and a value near 1x means the two runs used the same membership"
    )
    # Both are measured on the same head, so they are comparable at all.
    assert s.n_fine == dk.n_fine
    assert s.provenance["subject"] == dk.provenance["subject"]
    # The Schaefer artefact carries the DK figure as its own control row -- same
    # process, same lead field, same held-out states -- so the comparison above
    # does not span two runs of the generator.  `specificity` is written beside
    # the dataclass's fields, not into it, so it is read from the record.
    rec = json.loads(rp.measurement_path(parc="Schaefer400x7").read_text())
    assert rec["specificity"]["aparc"]["lead_field_energy_retained"] == pytest.approx(
        dk.lead_field_energy_retained, rel=1e-9
    ), "the in-run DK control disagrees with the committed DK artefact"
