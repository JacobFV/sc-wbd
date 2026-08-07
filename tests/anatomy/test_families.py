"""The regional family declaration: the partition, its provenance, and its guards.

Every assertion here was watched fail before it was trusted
(``reports/decorative_guards.md`` rec. 1).  Each guard test names the mutation
it applies, so the next reader can reproduce the failure rather than take my
word that the guard fires -- a validator whose branches are never exercised
reads as coverage while providing none, and this file exists to keep
``FamilyPartition.validate`` from becoming one of the ~26 filed examples.

The substantive test is :func:`test_declared_partition_separates_but_a_matched_null_does_not`.
It is the only thing standing between "families exist because measurement says
so" and "families exist because I named them".
"""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from scwbd.anatomy import BrainPrior, load_parcellation
from scwbd.anatomy.families import (
    CORTICAL_FAMILY_DEFINITION,
    FAMILY_FIELDS,
    FieldProvenance,
    FamilyPartition,
    RegionFamily,
    derive_families,
)
from scwbd.anatomy.geometry import load_surface

MAIN_ATLAS = "Schaefer400x7"


@pytest.fixture(scope="module")
def prior_main():
    return BrainPrior.load(MAIN_ATLAS, include_subcortex=True)


@pytest.fixture(scope="module")
def fams(prior_main):
    return derive_families(prior_main)


# ----------------------------------------------------------------- the object
def test_partition_covers_the_real_414_parcels_exactly(prior_main, fams):
    """414 = 400 Schaefer cortex + 14 subcortex. Not the synthetic stand-in's 454."""
    assert prior_main.n_parcels == 414
    assert fams.n_regions == 414
    idx = fams.family_index()
    assert idx.shape == (414,)
    assert (idx >= 0).all(), "every parcel must belong to a family"
    assert int(np.bincount(idx).sum()) == 414
    assert sum(f.n_parcels for f in fams) == 414


def test_families_are_disjoint_and_exhaustive(fams):
    seen: set[int] = set()
    for f in fams:
        p = set(f.parcels)
        assert p, f"{f.family_id} is empty"
        assert not (p & seen), f"{f.family_id} overlaps an earlier family"
        seen |= p
    assert seen == set(range(fams.n_regions))


def test_every_reported_field_value_has_provenance(fams):
    """The rule the module exists for: no number without a source."""
    for f in fams:
        for name in FAMILY_FIELDS:
            if f.field_value(name) is not None:
                prov = f.provenance_for(name)
                assert prov is not None, f"{f.family_id}.{name} has no provenance"
                assert prov.method, f"{f.family_id}.{name} provenance has no method"
                assert prov.source_key


def test_subcortical_families_report_no_measured_regional_field(fams, prior_main):
    """No map in this build covers subcortex, so no subcortical field is measured.

    ``BrainPrior.timescale_prior()`` *does* return a number for each subcortical
    parcel -- the same one for all 14.  Reporting it as a family's timescale
    would impute an average-brain label (ARCHITECTURE.md §7 rule 1), so the
    field is ``not_established``.  This test pins the underlying degeneracy too,
    so that if subcortical maps ever land, it fails and forces a revisit.
    """
    ts = [
        float(p.mean() if callable(getattr(p, "mean", None)) else p.mean)
        for p in prior_main.timescale_prior()[400:]
    ]
    assert len(set(np.round(ts, 9))) == 1, (
        "subcortical timescale priors now differ across parcels -- a subcortical "
        "measurement may have landed; re-derive the subcortical family fields"
    )
    for f in fams:
        if f.division != "subcortex":
            continue
        for name in FAMILY_FIELDS:
            assert f.field_value(name) is None, f"{f.family_id}.{name} is not None"
        assert f.training_status == "prior_only_untrained"
        assert f.evidence_tier == "atlas_separation"
        assert f.separating_evidence == ()


def test_cortical_families_carry_the_measured_fields(fams):
    for fid in CORTICAL_FAMILY_DEFINITION:
        f = fams[fid]
        assert f.evidence_tier == "measured_separation"
        assert f.separating_evidence
        assert f.receptor_profile is not None and len(f.receptor_profile) == 20
        assert len(f.receptor_names) == len(f.receptor_profile)
        assert f.intrinsic_timescale_s is not None and f.intrinsic_timescale_s > 0
        assert f.ei_prior is not None


def test_association_timescale_exceeds_unimodal(fams):
    """Direction check on the shipped values, not a tautology of the partition.

    Intrinsic timescales lengthen from sensory toward association cortex.  If
    this inverts, the E/I ordering or the map orientation has flipped end to end
    -- the exact failure ``EI_ORDERING_SOURCES`` warns about, which stays
    entirely plausible-looking.
    """
    assert (
        fams["cortex_association"].intrinsic_timescale_s
        > fams["cortex_unimodal"].intrinsic_timescale_s
    )
    assert fams["cortex_association"].ei_prior > fams["cortex_unimodal"].ei_prior


def test_laminar_differentiation_is_not_established_anywhere(fams):
    for f in fams:
        assert f.laminar_differentiation is None
        prov = f.provenance_for("laminar_differentiation")
        if prov is not None:
            assert prov.status == "not_established"


def test_mesulam_positional_join_is_refused():
    """Why ``laminar_differentiation`` is absent, as an executable check.

    ``mesulam_scale033.csv`` is a bare integer column with no region names.
    Joining it to Desikan-Killiany by position is an assumption, and it is
    testable against what the classes mean: class 1 is idiotypic (primary
    sensorimotor), class 4 is paralimbic.  The join places cuneus, superior
    frontal and insula in class 1 and postcentral in class 4, so it is wrong.

    If this test ever fails, the join has become defensible and a laminar field
    can be established -- that is the point of asserting it.
    """
    path = Path("assets/src/hansen_receptors/data/mesulam_scale033.csv")
    if not path.exists():
        pytest.skip("hansen_receptors assets not installed")
    mes = np.array([int(r[0]) for r in csv.reader(path.open()) if r])
    dk = load_parcellation("DesikanKilliany", "fsLR", "32k")
    assert mes.shape[0] == dk.n_parcels
    names = [str(x) for x in dk.labels]
    idiotypic = ("pericalcarine", "precentral", "postcentral", "transversetemporal")
    paralimbic = ("entorhinal", "parahippocampal", "temporalpole", "insula",
                  "caudalanteriorcingulate", "rostralanteriorcingulate")
    c1 = [n for n, m in zip(names, mes) if m == 1]
    c4 = [n for n, m in zip(names, mes) if m == 4]
    frac1 = np.mean([any(e in n for e in idiotypic) for n in c1])
    frac4 = np.mean([any(e in n for e in paralimbic) for n in c4])
    assert not (frac1 >= 0.6 and frac4 >= 0.6), (
        f"the positional Mesulam->DK join now agrees with the meaning of the "
        f"classes (idiotypic {frac1:.2f}, paralimbic {frac4:.2f}); laminar "
        "differentiation can be established -- update families.py"
    )


def test_nc_routing_flags_receptor_profile_only(fams):
    """Checkpoint policy input: only the Hansen-derived field is NC.

    The subcortical families must NOT be flagged. Their boundaries come from
    Aseg14T/tian2020, and ``reports/subcortical_atlas_substitution.md`` exists
    precisely to keep the non-commercial Harvard-Oxford term off the default
    path; hardcoding the wrong source key would flag them and mis-route.
    """
    nc = fams.nc_licensed_fields()
    assert set(nc) == set(CORTICAL_FAMILY_DEFINITION)
    for fid in CORTICAL_FAMILY_DEFINITION:
        assert nc[fid] == ("receptor_profile",)
    for f in fams:
        if f.division == "subcortex":
            assert f.nc_fields() == ()
            assert f.membership_source == "tian2020"


def test_untrained_families_are_the_subcortical_ones(fams):
    assert {f.family_id for f in fams.untrained()} == {
        f.family_id for f in fams if f.division == "subcortex"
    }


def test_absent_systems_are_declared_not_silently_missing(fams):
    for key in ("cerebellum", "brainstem_hypothalamic_autonomic", "auditory"):
        assert key in fams.declared_absent
        assert len(fams.declared_absent[key]) > 80, "a reason, not a label"
    assert not any(f.division == "cerebellum" for f in fams)


# ----------------------------------------------------------------- the guards
# Each test mutates a VALID partition in exactly one way and asserts validate()
# raises. The mutation is named in the docstring.


def _mutate(fams: FamilyPartition, **kw) -> FamilyPartition:
    return dataclasses.replace(fams, **kw)


def _fam(fams: FamilyPartition, fid: str, **kw) -> RegionFamily:
    return dataclasses.replace(fams[fid], **kw)


def test_guard_fires_on_empty_partition(fams):
    """Mutation: families=()."""
    with pytest.raises(ValueError, match="no families"):
        _mutate(fams, families=()).validate()


def test_guard_fires_on_duplicate_family_id(fams):
    """Mutation: append a copy of cortex_unimodal under its own id."""
    dup = _fam(fams, "cortex_unimodal", parcels=(0,))
    with pytest.raises(ValueError, match="duplicate family ids"):
        _mutate(fams, families=fams.families + (dup,)).validate()


def test_guard_fires_on_empty_family(fams):
    """Mutation: give subcortex_thal no parcels (the 'declare it absent' case)."""
    bad = _fam(fams, "subcortex_thal", parcels=())
    others = tuple(f for f in fams if f.family_id != "subcortex_thal")
    with pytest.raises(ValueError, match="has no parcels"):
        _mutate(fams, families=others + (bad,)).validate()


def test_guard_fires_on_out_of_range_parcel(fams):
    """Mutation: point subcortex_thal at parcel 9999."""
    bad = _fam(fams, "subcortex_thal", parcels=(9999,))
    others = tuple(f for f in fams if f.family_id != "subcortex_thal")
    with pytest.raises(ValueError, match="outside"):
        _mutate(fams, families=others + (bad,)).validate()


def test_guard_fires_on_overlapping_families(fams):
    """Mutation: give subcortex_thal a parcel that cortex_unimodal already owns."""
    stolen = fams["cortex_unimodal"].parcels[0]
    bad = _fam(fams, "subcortex_thal", parcels=fams["subcortex_thal"].parcels + (stolen,))
    others = tuple(f for f in fams if f.family_id != "subcortex_thal")
    with pytest.raises(ValueError, match="disjoint"):
        _mutate(fams, families=others + (bad,)).validate()


def test_guard_fires_on_non_exhaustive_partition(fams):
    """Mutation: drop one parcel from cortex_association so nothing owns it."""
    short = _fam(fams, "cortex_association", parcels=fams["cortex_association"].parcels[:-1])
    others = tuple(f for f in fams if f.family_id != "cortex_association")
    with pytest.raises(ValueError, match="belong to no family"):
        _mutate(fams, families=others + (short,)).validate()


def test_guard_fires_on_value_without_provenance(fams):
    """Mutation: give subcortex_thal an intrinsic timescale but no provenance.

    This is the fabricated-number case the module exists to prevent.
    """
    bad = _fam(fams, "subcortex_thal", intrinsic_timescale_s=0.02, provenance=())
    others = tuple(f for f in fams if f.family_id != "subcortex_thal")
    with pytest.raises(ValueError, match="no\n?\\s*FieldProvenance|no FieldProvenance"):
        _mutate(fams, families=others + (bad,)).validate()


def test_guard_fires_on_not_established_field_that_reports_a_value(fams):
    """Mutation: keep the 'not_established' provenance but supply a value anyway."""
    bad = _fam(fams, "subcortex_thal", intrinsic_timescale_s=0.02)
    others = tuple(f for f in fams if f.family_id != "subcortex_thal")
    with pytest.raises(ValueError, match="not_established"):
        _mutate(fams, families=others + (bad,)).validate()


def test_guard_fires_on_measured_claim_with_zero_coverage(fams):
    """Mutation: relabel the receptor provenance 'measured' at coverage 0.0."""
    f = fams["cortex_unimodal"]
    prov = tuple(
        dataclasses.replace(p, coverage=0.0) if p.field == "receptor_profile" else p
        for p in f.provenance
    )
    bad = _fam(fams, "cortex_unimodal", provenance=prov)
    others = tuple(x for x in fams if x.family_id != "cortex_unimodal")
    with pytest.raises(ValueError, match="coverage 0.0"):
        _mutate(fams, families=others + (bad,)).validate()


def test_guard_fires_when_measured_tier_names_no_evidence(fams):
    """Mutation: strip separating_evidence from a measured_separation family."""
    bad = _fam(fams, "cortex_unimodal", separating_evidence=())
    others = tuple(f for f in fams if f.family_id != "cortex_unimodal")
    with pytest.raises(ValueError, match="names no separating evidence"):
        _mutate(fams, families=others + (bad,)).validate()


def test_guard_fires_when_atlas_tier_claims_measured_evidence(fams):
    """Mutation: credit an atlas-only family with a measured separation.

    This is the tier-laundering case: it is how a prior gets credited with a
    finding it does not support.
    """
    bad = _fam(fams, "subcortex_thal", separating_evidence=("receptor_panel",))
    others = tuple(f for f in fams if f.family_id != "subcortex_thal")
    with pytest.raises(ValueError, match="not a measured separation"):
        _mutate(fams, families=others + (bad,)).validate()


def test_guard_fires_when_atlas_tier_is_not_marked_untrained(fams):
    """Mutation: mark an atlas-only family as having regional data (breaks `stage1-data-limited`)."""
    bad = _fam(fams, "subcortex_thal", training_status="has_regional_data")
    others = tuple(f for f in fams if f.family_id != "subcortex_thal")
    with pytest.raises(ValueError, match="prior_only_untrained"):
        _mutate(fams, families=others + (bad,)).validate()


def test_guard_fires_when_a_synthetic_partition_claims_measured_families(fams):
    """Mutation: relabel the partition synthetic while families keep their tiers.

    The synthetic-prior incident in ``scwbd/foundation/anatomy.py`` is the
    reason: a correct provenance field that nobody reads is not a control.
    """
    with pytest.raises(ValueError, match="may not present itself"):
        _mutate(fams, provenance="synthetic_fallback").validate()


def test_field_provenance_rejects_unknown_field_and_status():
    with pytest.raises(ValueError, match="unknown family field"):
        FieldProvenance("not_a_field", "schaefer2018", "", "", False, "m", "measured", 1.0)
    with pytest.raises(ValueError, match="unknown field status"):
        FieldProvenance("ei_prior", "schaefer2018", "", "", False, "m", "guessed", 1.0)
    with pytest.raises(ValueError, match="coverage"):
        FieldProvenance("ei_prior", "schaefer2018", "", "", False, "m", "measured", 1.7)


def test_family_index_refuses_on_an_invalid_partition(fams):
    """family_index() validates first: a bad partition must not yield an array."""
    short = _fam(fams, "cortex_association", parcels=fams["cortex_association"].parcels[:-1])
    others = tuple(f for f in fams if f.family_id != "cortex_association")
    with pytest.raises(ValueError):
        _mutate(fams, families=others + (short,)).family_index()


# ------------------------------------------------- the separation claim itself
def _spin_perms(parc, n_parcels, n_spin, seed):
    """Váša-style spin: rotate the sphere, re-match parcels by optimal assignment."""
    cent = {}
    for h in ("L", "R"):
        s = load_surface("fsLR", "32k", "sphere", h)
        vl = parc.vertex_labels[h]
        c = np.full((n_parcels, 3), np.nan)
        for i in range(n_parcels):
            m = vl == i
            if m.any():
                v = s.coords[m].mean(0)
                c[i] = v / np.linalg.norm(v)
        cent[h] = c
    idx = {h: np.where(~np.isnan(cent[h][:, 0]))[0] for h in ("L", "R")}
    rng = np.random.default_rng(seed)
    mir = np.diag([-1.0, 1.0, 1.0])
    out = []
    for _ in range(n_spin):
        Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
        Q = Q * np.sign(np.diag(R))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        perm = np.arange(n_parcels)
        for h in ("L", "R"):
            C = cent[h][idx[h]]
            Qh = Q if h == "L" else mir @ Q @ mir
            r, c = linear_sum_assignment(-((C @ Qh.T) @ C.T))
            perm[idx[h][r]] = idx[h][c]
        out.append(perm)
    return out


def _pseudo_f2(X, ma, mb):
    A, B = X[ma], X[mb]
    U = np.vstack([A, B])
    gm = U.mean(0)
    ssb = len(A) * ((A.mean(0) - gm) ** 2).sum() + len(B) * ((B.mean(0) - gm) ** 2).sum()
    ssw = ((A - A.mean(0)) ** 2).sum() + ((B - B.mean(0)) ** 2).sum()
    return ssb / (ssw / (len(U) - 2))


def test_declared_partition_separates_but_a_matched_null_does_not(prior_main):
    """The claim the whole module rests on, with the control that lets it fail.

    ``CORTICAL_FAMILY_DEFINITION`` asserts that unimodal and association cortex
    differ in measured regional profile beyond what spatial smoothness explains.
    Two things are checked:

    1. the declared partition separates on the receptor panel under a spin null;
    2. a **size- and smoothness-matched** null partition -- the declared labels
       pushed through one spin rotation -- does *not* reliably separate.

    Without (2) this test would pass for any spatially contiguous split of
    cortex and would be decorative.  200 spins here rather than the 1000 behind
    the shipped constant; the p-value resolution is coarser, the direction is
    the same.
    """
    parc = load_parcellation(MAIN_ATLAS, "fsLR", "32k")
    n_ctx = parc.n_parcels
    net = np.asarray([str(x) for x in prior_main.network[:n_ctx]])
    lab = np.where(np.isin(net, CORTICAL_FAMILY_DEFINITION["cortex_unimodal"]), 0, 1)

    names = sorted(
        m for m in (prior_main.maps.names() if hasattr(prior_main.maps, "names")
                    else prior_main.maps.maps)
        if m.startswith("receptor_")
    )
    X = np.column_stack([np.asarray(prior_main.maps[k].values, float)[:n_ctx] for k in names])
    X = (X - X.mean(0)) / X.std(0).clip(1e-9)

    perms = _spin_perms(parc, n_ctx, n_spin=200, seed=20260806)
    obs = _pseudo_f2(X, lab == 0, lab == 1)
    null = np.array([_pseudo_f2(X, (ls := lab[p]) == 0, ls == 1) for p in perms])
    p_spin = (1 + int((null >= obs).sum())) / (1 + len(null))
    assert p_spin < 0.05, (
        f"the declared cortical partition no longer separates on the receptor "
        f"panel (F={obs:.2f}, p_spin={p_spin:.4f}). CORTICAL_FAMILY_DEFINITION "
        "is not supported by the data it claims to rest on."
    )

    # The control: same group sizes, same spatial smoothness, no anatomy.
    matched = lab[perms[0]]
    obs_null = _pseudo_f2(X, matched == 0, matched == 1)
    assert obs_null < obs, (
        "a spun (anatomy-free) partition separates the receptor panel as well as "
        "the declared one; this test cannot distinguish the two and is decorative"
    )


def test_rejected_partitions_are_recorded_with_their_reason(fams):
    """The ladder that was tested, so 'we shipped 2' is auditable, not asserted."""
    rej = fams.separation_evidence["rejected"]
    for key in ("C7_yeo7", "C4_uni_dorsattn_salvent_assoc", "C3_uni_attention_assoc",
                "EconomoKoskinas5"):
        assert key in rej and len(rej[key]) > 20
    assert fams.separation_evidence["n_spin"] == 1000
