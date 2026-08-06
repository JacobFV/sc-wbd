"""The resolution poset: overlaps, cocycles, obstruction certificates, R02/R03."""

from __future__ import annotations

import math

import pytest
import torch

from scwbd.transforms.errors import (
    CocycleObstructionError,
    ProlongationWithoutRestrictionError,
    SiteError,
)
from scwbd.transforms.se3 import DTYPE
from scwbd.transforms.sheaf import (
    CoverageReport,
    Cover,
    ObstructionCertificate,
    Prolongation,
    Restriction,
    ScalePair,
    Section,
    Site,
    SupportObject,
    glue,
    glue_or_raise,
    measure_coverage,
)


def t(x) -> torch.Tensor:
    return torch.as_tensor(x, dtype=DTYPE)


def selection(elements, parent_elements) -> torch.Tensor:
    """The plain restriction: pick the named elements out of the parent."""
    idx = {e: i for i, e in enumerate(parent_elements)}
    M = torch.zeros((len(elements), len(parent_elements)), dtype=DTYPE)
    for r, e in enumerate(elements):
        M[r, idx[e]] = 1.0
    return M


# --------------------------------------------------------------------------
# a small site: a cortical strip covered by two overlapping tiles
# --------------------------------------------------------------------------


def build_site(tolerance: float = 1e-6) -> Site:
    s = Site()
    U = tuple(range(8))
    Ua = tuple(range(0, 5))  # left tile
    Ub = tuple(range(3, 8))  # right tile
    Uc = (3, 4)  # their overlap
    for oid, elems in [("U", U), ("Ua", Ua), ("Ub", Ub), ("Uc", Uc)]:
        s.add_object(
            SupportObject(oid, elems, kind="surface_vertex", units="V", tolerance=tolerance)
        )
    s.add_restriction(Restriction("U", "Ua", selection(Ua, U), method="tile_select"))
    s.add_restriction(Restriction("U", "Ub", selection(Ub, U), method="tile_select"))
    s.add_restriction(Restriction("Ua", "Uc", selection(Uc, Ua), method="overlap_select"))
    s.add_restriction(Restriction("Ub", "Uc", selection(Uc, Ub), method="overlap_select"))
    s.add_cover(Cover("U", ("Ua", "Ub"), id="two_tiles"))
    return s


TRUTH = t([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])


def sections_from(truth: torch.Tensor, site: Site, *, perturb_b: float = 0.0):
    xa = site.restriction("U", "Ua").apply(truth)
    xb = site.restriction("U", "Ub").apply(truth).clone()
    if perturb_b:
        xb = xb + perturb_b
    return [Section("Ua", xa), Section("Ub", xb)]


# --------------------------------------------------------------------------
# site hygiene
# --------------------------------------------------------------------------


def test_undeclared_restriction_is_refused() -> None:
    s = build_site()
    with pytest.raises(SiteError) as exc:
        s.restriction("Ua", "Ub")
    assert "no defensible map" in exc.value.remedy


def test_a_cover_must_actually_cover() -> None:
    s = build_site()
    s.add_object(SupportObject("Ud", (0, 1), units="V"))
    with pytest.raises(SiteError) as exc:
        s.add_cover(Cover("U", ("Ua", "Ud"), id="bad"))
    assert "does not cover" in str(exc.value)
    assert "never filled by averaging" in exc.value.remedy


def test_restrictions_are_versioned_and_cannot_be_swapped_silently() -> None:
    s = build_site()
    with pytest.raises(SiteError) as exc:
        s.add_restriction(
            Restriction("U", "Ua", selection((0, 1, 2, 3, 4), tuple(range(8))), version="2.0.0")
        )
    assert "silently change every past residual" in str(exc.value)


def test_restriction_shape_is_checked() -> None:
    s = build_site()
    with pytest.raises(SiteError):
        s.add_restriction(Restriction("U", "Uc", torch.zeros((3, 8), dtype=DTYPE)))


# --------------------------------------------------------------------------
# gluing
# --------------------------------------------------------------------------


def test_compatible_sections_glue_and_the_residual_is_recorded() -> None:
    s = build_site()
    res = glue(s, sections_from(TRUTH, s), "two_tiles")
    assert res.ok
    g = res.global_section
    assert torch.allclose(g.values, TRUTH, atol=1e-6)
    assert g.gluing_residual < 1e-6
    # recorded, not assumed
    assert "gluing_residual" in g.as_record()
    assert g.provenance["restriction_versions"] == {"Ua": "1.0.0", "Ub": "1.0.0"}
    assert all(o.within_tolerance for o in g.overlaps)


def test_missing_section_is_refused_not_imputed() -> None:
    s = build_site()
    with pytest.raises(SiteError) as exc:
        glue(s, [Section("Ua", t([1.0, 2, 3, 4, 5]))], "two_tiles")
    assert "never imputed as zero" in exc.value.remedy


def test_overlap_disagreement_produces_an_obstruction_certificate() -> None:
    """R03: conflicting local views must not be hidden by one raster."""
    s = build_site(tolerance=1e-3)
    res = glue(s, sections_from(TRUTH, s, perturb_b=0.5), "two_tiles")
    assert not res.ok
    cert = res.certificate
    assert isinstance(cert, ObstructionCertificate)
    assert cert.failed_overlaps
    o = cert.failed_overlaps[0]
    assert o.overlap == "Uc"
    assert o.norm == pytest.approx(math.sqrt(2) * 0.5, rel=1e-9)
    assert "rho^Ua_Uc@1.0.0" in o.path_a and "rho^Ub_Uc@1.0.0" in o.path_b
    # and the runtime refuses to materialize a global raster
    with pytest.raises(CocycleObstructionError) as exc:
        res.global_section
    assert exc.value.code == "R03"
    assert exc.value.certificate is cert
    assert "Do not materialize a global raster" in exc.value.remedy
    rec = cert.as_record()
    assert rec["refusal"] == "R03"
    assert rec["branches"] == ["Ua", "Ub"]
    assert "separate views preserved" in rec["action"]


def test_the_branches_are_preserved_for_the_posterior() -> None:
    s = build_site(tolerance=1e-3)
    res = glue(s, sections_from(TRUTH, s, perturb_b=0.5), "two_tiles")
    assert [b.support for b in res.certificate.branches] == ["Ua", "Ub"]
    # the individual sections are untouched and still usable separately
    assert torch.allclose(res.certificate.branches[0].values, t([1.0, 2, 3, 4, 5]))


def test_disagreement_inside_tolerance_still_glues_but_reports_it() -> None:
    s = build_site(tolerance=1e-1)
    res = glue(s, sections_from(TRUTH, s, perturb_b=0.02), "two_tiles")
    assert res.ok
    g = res.global_section
    assert g.gluing_residual > 0.0  # the disagreement did not vanish
    assert max(o.norm for o in g.overlaps) > 0.0


def test_a_cover_that_does_not_determine_the_global_state_is_refused() -> None:
    s = Site()
    s.add_object(SupportObject("U", tuple(range(4)), units="V"))
    s.add_object(SupportObject("Ua", (0, 1), units="V"))
    s.add_object(SupportObject("Ub", (0, 1, 2, 3), units="V"))
    s.add_restriction(Restriction("U", "Ua", selection((0, 1), tuple(range(4)))))
    # Ub's restriction only sees element 0..1 too: elements 2,3 are unobserved
    M = torch.zeros((4, 4), dtype=DTYPE)
    M[0, 0] = M[1, 1] = 1.0
    s.add_restriction(Restriction("U", "Ub", M))
    s.add_cover(Cover("U", ("Ua", "Ub"), id="deficient"))
    with pytest.raises(SiteError) as exc:
        glue(s, [Section("Ua", t([1.0, 2.0])), Section("Ub", t([1.0, 2.0, 0.0, 0.0]))], "deficient")
    assert "does not determine a global section" in str(exc.value)


# --------------------------------------------------------------------------
# the cocycle condition
# --------------------------------------------------------------------------


def build_nested_site(smoothing: float = 0.0, tolerance: float = 1e-6) -> Site:
    """``Uc ⊆ Ud ⊆ Ua`` with a direct and a composite restriction path.

    ``Ua -> Ud`` optionally smooths (a parcel-averaging step), while
    ``Ua -> Uc`` selects directly.  With ``smoothing > 0`` the two routes to
    ``Uc`` no longer commute -- which is exactly the situation the cocycle
    residual exists to detect, and the reason "fine -> parcel -> coarse" and
    "fine -> coarse" must not be assumed interchangeable.
    """
    s = build_site(tolerance=tolerance)
    Ua = tuple(range(0, 5))
    Ud = (2, 3, 4)
    Uc = (3, 4)
    s.add_object(SupportObject("Ud", Ud, kind="parcel", units="V", tolerance=tolerance))
    R_ad = selection(Ud, Ua)
    if smoothing:
        # each parcel value picks up a fraction of its left neighbour
        for r, e in enumerate(Ud):
            if e - 1 in Ua:
                R_ad[r, Ua.index(e - 1)] += smoothing
                R_ad[r, Ua.index(e)] -= smoothing
    s.add_restriction(
        Restriction("Ua", "Ud", R_ad, version="1.0.0", method="parcel_average")
    )
    s.add_restriction(
        Restriction("Ud", "Uc", selection(Uc, Ud), version="1.0.0", method="select")
    )
    return s


def test_cocycle_residual_is_zero_when_the_paths_commute() -> None:
    s = build_nested_site(smoothing=0.0)
    x = s.restriction("U", "Ua").apply(TRUTH)
    om = s.cocycle_residual(x, "Ua", "Ud", "Uc")
    assert om.norm == pytest.approx(0.0, abs=1e-12)
    assert om.within_tolerance


def test_cocycle_residual_matches_the_written_formula() -> None:
    """``Omega_abc = ||rho^Ua_Uc x - rho^Ub_Uc rho^Ua_Ub x||_Wc``."""
    s = build_nested_site(smoothing=0.25)
    x = s.restriction("U", "Ua").apply(TRUTH)
    om = s.cocycle_residual(x, "Ua", "Ud", "Uc")
    direct = s.restriction("Ua", "Uc").apply(x)
    composed = s.restriction("Ud", "Uc").apply(s.restriction("Ua", "Ud").apply(x))
    assert torch.allclose(om.residual, direct - composed)
    assert om.norm == pytest.approx(float(torch.linalg.norm(direct - composed)))
    assert om.norm > 0


def test_cocycle_violation_produces_a_certificate_naming_the_failed_path() -> None:
    s = build_nested_site(smoothing=0.25, tolerance=1e-3)
    res = glue(s, sections_from(TRUTH, s), "two_tiles")
    assert not res.ok
    cert = res.certificate
    assert cert.failed_cocycles, "the cocycle check must be what fails here"
    c = cert.failed_cocycles[0]
    assert (c.a, c.b, c.c) == ("Ua", "Ud", "Uc")
    # the certificate names the exact restriction path that did not commute
    assert c.failed_path == (
        "Omega_{Ua,Ud,Uc}: direct [rho^Ua_Uc@1.0.0] vs composed "
        "[rho^Ud_Uc@1.0.0 o rho^Ua_Ud@1.0.0]"
    )
    assert any("Omega_{Ua,Ud,Uc}" in p for p in cert.failed_paths)
    rec = cert.as_record()
    assert rec["refusal"] == "R03"
    assert rec["cocycles"][0]["direct_path"] == "rho^Ua_Uc@1.0.0"
    assert rec["cocycles"][0]["composed_path"] == "rho^Ud_Uc@1.0.0 o rho^Ua_Ud@1.0.0"
    assert rec["max_residual_W"] > rec["cocycles"][0]["tolerance"]
    with pytest.raises(CocycleObstructionError):
        glue_or_raise(s, sections_from(TRUTH, s), "two_tiles")


def test_a_larger_violation_gives_a_larger_residual() -> None:
    norms = []
    for sm in (0.05, 0.25, 0.5):
        s = build_nested_site(smoothing=sm, tolerance=1e-9)
        x = s.restriction("U", "Ua").apply(TRUTH)
        norms.append(s.cocycle_residual(x, "Ua", "Ud", "Uc").norm)
    assert norms[0] < norms[1] < norms[2]


def test_weight_matrix_defines_the_norm_the_tolerance_is_stated_in() -> None:
    s = build_nested_site(smoothing=0.25, tolerance=1e-9)
    x = s.restriction("U", "Ua").apply(TRUTH)
    plain = s.cocycle_residual(x, "Ua", "Ud", "Uc").norm
    s2 = build_nested_site(smoothing=0.25, tolerance=1e-9)
    # re-declare Uc with a precision weighting: disagreement matters 100x more
    s2._objects["Uc"] = SupportObject(
        "Uc", (3, 4), units="V", weight=torch.eye(2, dtype=DTYPE) * 100.0, tolerance=1e-9
    )
    weighted = s2.cocycle_residual(x, "Ua", "Ud", "Uc").norm
    assert weighted == pytest.approx(10.0 * plain)


# --------------------------------------------------------------------------
# restriction / prolongation pairs (R02)
# --------------------------------------------------------------------------


def averaging_pair(n_fine: int = 6, block: int = 3):
    """R: average blocks of ``block`` fine elements into one coarse element."""
    n_coarse = n_fine // block
    R = torch.zeros((n_coarse, n_fine), dtype=DTYPE)
    for i in range(n_coarse):
        R[i, i * block : (i + 1) * block] = 1.0 / block
    return Restriction("fine", "coarse", R, method="block_average")


def fine_landmarks(n: int = 64, n_fine: int = 6, seed: int = 9) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn((n, n_fine), dtype=DTYPE, generator=g)


def test_prolongation_without_a_restriction_partner_is_refused_R02() -> None:
    with pytest.raises(ProlongationWithoutRestrictionError) as exc:
        Prolongation(
            restriction=None,
            matrix=torch.ones((6, 2), dtype=DTYPE),
            coverage=CoverageReport(64, 0.1, 0.0, 0.9),
            prior_sd_unresolved=1.0,
        )
    assert exc.value.code == "R02"
    assert "paired maps" in exc.value.remedy


def test_prolongation_without_tested_coverage_is_refused_R02() -> None:
    R = averaging_pair()
    with pytest.raises(ProlongationWithoutRestrictionError) as exc:
        Prolongation(restriction=R, matrix=None, coverage=None, prior_sd_unresolved=1.0)
    assert exc.value.code == "R02"
    assert "held-out landmark tests" in exc.value.remedy
    # an empty "test" does not count either
    with pytest.raises(ProlongationWithoutRestrictionError):
        Prolongation(
            restriction=R,
            matrix=None,
            coverage=CoverageReport(0, 0.0, 0.0, 1.0),
            prior_sd_unresolved=1.0,
        )


def test_prolongation_failing_its_coverage_threshold_is_refused() -> None:
    R = averaging_pair()
    P = torch.linalg.pinv(R.matrix)
    cov = measure_coverage(R, P, fine_landmarks())
    with pytest.raises(ProlongationWithoutRestrictionError) as exc:
        Prolongation(
            restriction=R, matrix=P, coverage=cov, prior_sd_unresolved=1.0,
            max_heldout_error=1e-6,
        )
    assert "coverage test failed" in str(exc.value)


def test_prolongation_must_declare_variance_on_the_unresolved_subspace() -> None:
    R = averaging_pair()
    P = torch.linalg.pinv(R.matrix)
    cov = measure_coverage(R, P, fine_landmarks())
    with pytest.raises(ProlongationWithoutRestrictionError) as exc:
        Prolongation(restriction=R, matrix=P, coverage=cov, prior_sd_unresolved=0.0)
    assert "infinite precision" in exc.value.remedy


def _admissible_pair(prior_sd: float = 1.0) -> ScalePair:
    R = averaging_pair()
    P = torch.linalg.pinv(R.matrix)
    cov = measure_coverage(R, P, fine_landmarks(), domain={"region": "dlpfc_tile"})
    return ScalePair(
        R,
        Prolongation(
            restriction=R,
            matrix=P,
            coverage=cov,
            prior_sd_unresolved=prior_sd,
            max_heldout_error=10.0,
        ),
    )


def test_prolongation_returns_a_distribution_not_a_point() -> None:
    pair = _admissible_pair()
    dist = pair.prolongation.prolong(t([1.0, -2.0]))
    assert dist.mean.numel() == 6
    assert dist.resolved_rank == 2
    assert dist.unresolved_rank == 4  # 4 fine directions no coarse datum can see
    assert float(dist.sd.min()) > 0.0
    with pytest.raises(ProlongationWithoutRestrictionError) as exc:
        dist.as_point()
    assert "distribution over fine states, not a point" in str(exc.value)
    assert "never hand the mean onward" in exc.value.remedy.lower()


def test_the_unresolved_subspace_carries_the_prior_variance() -> None:
    pair = _admissible_pair(prior_sd=2.0)
    dist = pair.prolongation.prolong(t([1.0, -2.0]))
    # variance along an unresolved direction is the declared prior
    u = dist.unresolved_basis[:, 0]
    assert float(u @ dist.cov @ u) == pytest.approx(4.0)
    # and zero along a resolved direction (nothing invented, nothing hidden)
    R = pair.restriction.matrix
    v = R[0] / torch.linalg.norm(R[0])
    assert float(v @ dist.cov @ v) == pytest.approx(0.0, abs=1e-12)
    # more prior uncertainty -> wider samples
    wide = _admissible_pair(prior_sd=8.0).prolongation.prolong(t([1.0, -2.0]))
    assert float(wide.sd.mean()) > float(dist.sd.mean())


def test_prolongation_samples_are_consistent_with_the_coarse_datum() -> None:
    """Every fine sample must restrict back to the coarse state it came from."""
    pair = _admissible_pair(prior_sd=3.0)
    coarse = t([1.0, -2.0])
    samples = pair.prolongation.prolong(coarse).sample(64, seed=2)
    back = samples @ pair.restriction.matrix.T
    assert torch.allclose(back, coarse.expand_as(back), atol=1e-9)


def test_round_trip_and_held_out_landmark_reports() -> None:
    pair = _admissible_pair()
    rep = pair.round_trip_report(fine_landmarks(n=128))
    # coarse -> fine -> coarse is exact for a proper pair
    assert rep["coarse_fine_coarse_rms"] < 1e-12
    # fine -> coarse -> fine is NOT, and pretending otherwise is refusal R02
    assert rep["fine_coarse_fine_rms"] > 0.1
    assert rep["unresolved_rank"] == 4
    cov = pair.prolongation.coverage
    assert cov.tested and cov.n_landmarks == 64
    assert 0.0 <= cov.fraction_of_fine_variance_explained <= 1.0
    assert cov.coarse_roundtrip_error < 1e-12
    assert cov.domain == {"region": "dlpfc_tile"}


def test_prolongation_provenance_names_its_partner_and_policy() -> None:
    pair = _admissible_pair()
    prov = pair.prolongation.prolong(t([1.0, -2.0])).provenance
    assert prov["restriction_partner"] == "rho^fine_coarse@1.0.0"
    assert prov["coverage"]["tested"] is True
    assert "not reconstructed structure" in prov["note"]


def test_prolongation_shape_must_match_its_partner() -> None:
    R = averaging_pair()
    cov = measure_coverage(R, torch.linalg.pinv(R.matrix), fine_landmarks())
    with pytest.raises(ProlongationWithoutRestrictionError):
        Prolongation(
            restriction=R,
            matrix=torch.ones((3, 2), dtype=DTYPE),
            coverage=cov,
            prior_sd_unresolved=1.0,
        )
