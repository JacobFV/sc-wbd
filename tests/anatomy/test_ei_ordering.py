"""The E/I ordering: the default must not read Hansen, the opt-in must work.

Every assertion here was watched fail before it was trusted
(``reports/decorative_guards.md`` rec. 1). The mutations used are named in each
test's docstring so the next reader can reproduce the failure rather than take
my word that it fires.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from scwbd.anatomy import sources as S
from scwbd.anatomy.maps import RECEPTOR_GROUPS
from scwbd.anatomy.priors import (
    DEFAULT_EI_ORDERING,
    EI_LOG_RANGE,
    EI_ORDERING_SOURCES,
)
from scwbd.release.licence import is_noncommercial_text, is_share_alike_text

#: Every map name that is derived from the Hansen PET atlas.
HANSEN_MAPS = frozenset(
    {"ei_proxy"}
    | {f"receptor_{r}" for g in RECEPTOR_GROUPS.values() for r in g}
)


# ---------------------------------------------------------------------------
# the default path
# ---------------------------------------------------------------------------
def test_default_ei_ordering_reads_no_hansen_map(brain_prior):
    """The whole point of the substitution.

    Watched fail by setting ``DEFAULT_EI_ORDERING = "hansen_receptors"``:
    fails on ``licence_keys`` and on ``map`` in one run.
    """
    z, rec = brain_prior.ei_ordering()
    assert rec["ordering"] == DEFAULT_EI_ORDERING
    assert rec["available"], "the default ordering built nothing"
    assert "hansen_receptors" not in rec["licence_keys"]
    for u in rec["maps_used"]:
        assert u["source_key"] != "hansen_receptors", u
        assert u["map"] not in HANSEN_MAPS, u
    assert np.isfinite(z).sum() >= brain_prior.n_cortex


def test_default_ei_ordering_carries_no_share_alike_and_no_nc(brain_prior):
    """Licence check computed from source terms, not from a hardcoded list.

    Watched fail by adding ``("ei_proxy", +1)`` to the ``hcp_hierarchy``
    ingredients.
    """
    _, rec = brain_prior.ei_ordering()
    for key in rec["licence_keys"]:
        text = S.SRC[key]["license"]
        assert is_share_alike_text(text) is not True, (key, text)
        assert is_noncommercial_text(text) is not True, (key, text)


def test_default_ei_prior_provenance_never_names_hansen(brain_prior):
    """A parcel's provenance string is what a downstream reader actually sees.

    Watched fail by flipping the default; every parcel's string then contains
    the CC-BY-NC-SA notice.

    It also asserts the *positive* form -- that the licence text of each
    ingredient is interpolated verbatim from the registry. Mutation M4 (delete
    the ``lic`` interpolation) originally passed every test in this file,
    because the ordering's *name* happened to carry the disclosure for the
    Hansen case. A disclosure that survives only by coincidence of naming is
    not a disclosure.
    """
    _, rec = brain_prior.ei_ordering()
    for p in brain_prior.ei_ratio_prior():
        low = p.provenance.lower()
        assert "hansen" not in low, p.provenance[:200]
        assert "cc-by-nc-sa" not in low, p.provenance[:200]
        for u in rec["maps_used"]:
            assert f"{u['source_key']} ({u['licence']})" in p.provenance


def test_the_object_still_carries_hansen_even_though_the_prior_does_not(brain_prior):
    """The distinction the licence audit turns on, asserted so it cannot rot.

    ``load_maps`` builds every map whose data is on disk, so the assembled
    ``BrainPrior`` still *contains* receptor maps. Dropping Hansen from the E/I
    prior therefore does not make the object Hansen-free, and a release path
    that reads ``provenance["sources"]`` will still see it. Read
    ``provenance["ei_ordering"]["licence_keys"]`` instead.
    """
    if "ei_proxy" not in brain_prior.maps:
        pytest.skip("no Hansen PET volumes on this machine")
    assert "hansen_receptors" in brain_prior.provenance["sources"]
    assert "hansen_receptors" not in brain_prior.provenance["ei_ordering"]["licence_keys"]


# ---------------------------------------------------------------------------
# the opt-in path
# ---------------------------------------------------------------------------
def test_hansen_ordering_is_still_available(brain_prior):
    """Opt-in, not deleted. Thesis S5 needs receptor identity.

    Watched fail by deleting the ``"hansen_receptors"`` entry from
    ``EI_ORDERING_SOURCES``.
    """
    if "ei_proxy" not in brain_prior.maps:
        pytest.skip("no Hansen PET volumes on this machine")
    z, rec = brain_prior.ei_ordering("hansen_receptors")
    assert rec["available"]
    assert rec["licence_keys"] == ["hansen_receptors"]
    assert rec["is_default"] is False
    assert [u["map"] for u in rec["maps_used"]] == ["ei_proxy"]
    assert np.isfinite(z).sum() >= brain_prior.n_cortex


def test_choosing_hansen_records_itself_in_every_parcel(brain_prior):
    """A licence choice that leaves no trace is not a choice, it is a default.

    Watched fail by dropping the ``lic`` interpolation from the citation.
    """
    if "ei_proxy" not in brain_prior.maps:
        pytest.skip("no Hansen PET volumes on this machine")
    priors = brain_prior.ei_ratio_prior("hansen_receptors")
    assert len(priors) == brain_prior.n_parcels
    for p in priors:
        assert "hansen_receptors" in p.provenance
        assert "CC-BY-NC-SA-4.0" in p.provenance
        assert "NON-COMMERCIAL AND SHARE-ALIKE" in p.provenance


def test_the_two_orderings_are_not_the_same_prior(brain_prior):
    """Regression guard on the substitution's actual cost.

    Measured 2026-08-06 on one cached Schaefer400x7 build: Spearman
    ``rho = +0.358`` over 400 cortical parcels (``+0.444`` over the 100
    parcels of the Schaefer100x7 build this fixture actually uses). The bounds are deliberately
    loose -- the point is that this is neither a reproduction (``rho -> 1``)
    nor an inversion (``rho < 0``), and that a future change to either side
    moves it.
    """
    if "ei_proxy" not in brain_prior.maps:
        pytest.skip("no Hansen PET volumes on this machine")
    zd, _ = brain_prior.ei_ordering()
    zh, _ = brain_prior.ei_ordering("hansen_receptors")
    m = np.isfinite(zd) & np.isfinite(zh)
    rho = float(spearmanr(zd[m], zh[m]).statistic)
    assert 0.15 < rho < 0.75, (
        f"rho={rho:+.4f}: the substitution's agreement with the receptor "
        "ordering has moved outside the measured band; re-measure and update "
        "reports/ei_ordering_substitution.md rather than widening this bound"
    )


# ---------------------------------------------------------------------------
# orientation, absence, and misuse
# ---------------------------------------------------------------------------
def test_declared_orientations_match_the_hierarchy_they_claim(brain_prior):
    """A silent polarity flip upstream inverts the cortical E/I gradient.

    The orientation constants are declared a priori, so this checks them
    against the sensorimotor-association axis rather than deriving them from
    it. Watched fail by flipping ``myelin_t1t2``'s sign to ``+1``.
    """
    if "sa_axis" not in brain_prior.maps:
        pytest.skip("sa_axis unavailable")
    sa = brain_prior.maps["sa_axis"].values
    for spec in EI_ORDERING_SOURCES.values():
        for map_name, orient in spec["ingredients"]:
            if map_name not in brain_prior.maps or map_name == "ei_proxy":
                continue
            v = brain_prior.maps[map_name].values
            k = min(sa.size, v.size)
            m = np.isfinite(sa[:k]) & np.isfinite(v[:k])
            rho = float(spearmanr(sa[:k][m], v[:k][m]).statistic)
            assert np.sign(rho) == orient, (
                f"{map_name}: declared orientation {orient:+d} but it correlates "
                f"{rho:+.3f} with the S-A axis"
            )


def test_a_missing_ingredient_is_recorded_not_silently_dropped(brain_prior):
    """Absence must write something (decorative_guards, the absence variant).

    Watched fail by returning early from the ``missing`` branch.
    """
    import copy

    obj = copy.copy(brain_prior)
    obj.maps = _without(brain_prior.maps, "cortical_thickness")
    z, rec = obj.ei_ordering()
    assert rec["degraded"] is True
    assert "cortical_thickness" in rec["missing"]
    assert rec["missing"]["cortical_thickness"]
    assert len(rec["maps_used"]) == 2
    assert "DEGRADED" in obj.ei_ratio_prior()[0].provenance


def test_an_ordering_with_no_ingredients_states_it_rather_than_flattening(brain_prior):
    """All-nan is the honest answer; a flat cortex is not.

    Watched fail by making the empty-stack branch fall through to zeros.
    """
    import copy

    obj = copy.copy(brain_prior)
    obj.maps = _without(
        brain_prior.maps,
        *[m for m, _ in EI_ORDERING_SOURCES[DEFAULT_EI_ORDERING]["ingredients"]],
    )
    z, rec = obj.ei_ordering()
    assert rec["available"] is False
    assert rec["licence_keys"] == []
    assert "consequence" in rec
    assert not np.isfinite(z).any()
    priors = obj.ei_ratio_prior()
    assert all(p.sigma > 0.35 for p in priors), "a missing ordering must widen, not centre"


def test_a_constant_ingredient_states_it_rather_than_centring_every_parcel(brain_prior):
    """The branch mutation M7 walked through untested.

    A map with no between-parcel variance orders nothing. The old code centred
    every covered parcel on ``z = 0`` with the *narrow* sigma, which is a
    confident claim of uniform E/I -- thesis S6.1's failure mode presented as
    coverage. It must fall to the wide branch and say why.

    Watched fail by restoring ``z[fin] = 0.0`` in that branch.
    """
    import copy
    import dataclasses

    obj = copy.copy(brain_prior)
    flat = {}
    for name, _ in EI_ORDERING_SOURCES[DEFAULT_EI_ORDERING]["ingredients"]:
        if name in brain_prior.maps:
            m = brain_prior.maps[name]
            flat[name] = dataclasses.replace(m, values=np.ones_like(m.values))
    if not flat:
        pytest.skip("no ingredients to flatten")
    obj.maps = dataclasses.replace(brain_prior.maps, maps={**brain_prior.maps.maps, **flat})
    z, rec = obj.ei_ordering()
    assert rec["degenerate"], "a variance-free composite must say so"
    assert rec["available"] is False
    assert "consequence" in rec
    assert not np.isfinite(z).any()
    assert all(p.sigma > EI_LOG_RANGE for p in obj.ei_ratio_prior())


def test_unknown_ordering_raises_rather_than_falling_back(brain_prior):
    """A typo'd source name must not silently select the default."""
    with pytest.raises(KeyError):
        brain_prior.ei_ordering("recepters")


def _without(mapset, *names):
    """A copy of ``mapset`` with ``names`` removed, keeping the reason visible."""
    import dataclasses

    return dataclasses.replace(
        mapset,
        maps={k: v for k, v in mapset.maps.items() if k not in names},
        unavailable={**mapset.unavailable, **{n: "removed by a test" for n in names}},
    )
