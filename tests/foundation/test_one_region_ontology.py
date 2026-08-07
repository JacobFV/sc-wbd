"""O-7: the region vocabularies must agree, and today they only happen to.

``ARCHITECTURE.md`` O-7 records three region vocabularies — ``schema.Region``,
``anatomy.RegionFamily`` and ``foundation.RegionFamily`` — with no enforced
relationship. The last two share exactly one field name out of 17 and 9, and
disagree on what to call a family's identity (``family_id`` vs ``name``) and its
members (``parcels`` vs ``regions``).

O-7 also predicts why its own count was wrong:

> A defect register populated by incidents will always underestimate a defect of
> this shape, because the members that have not yet collided are invisible to it.

**A fourth collision, 2026-08-07.** Assembling whole-brain haemodynamic state
meant iterating the layout's families and reading each one's identity.
``getattr(fam, "family_id", fam)`` — the anatomy vocabulary's field — silently
fell through to the *object* on the layout vocabulary, which uses ``name``. The
resulting ``KeyError`` interpolated an entire ``RegionFamily`` dataclass,
receptor panel and provenance and all, into the message. The failure was loud;
it just spent its loudness on the wrong thing.

The real fix is the rewrite O-3 describes. This file is what is enforceable
before it: **the two partitions must be the same partition.** Nothing in the
codebase checks that today. They agree — 9 families, identical membership, 414
parcels each — and there is no mechanism by which they must, so the agreement is
a coincidence maintained by hand.
"""

from __future__ import annotations

import pytest

from scwbd.foundation.anatomy import load_anatomy
from scwbd.foundation.config import load_config
from scwbd.foundation.model import SCWBD

CONFIG = "configs/run2/pilot-families.yaml"


@pytest.fixture(scope="module")
def partitions() -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    an = load_anatomy()
    model = SCWBD(load_config(CONFIG).model, an)
    anatomy = {f.family_id: set(f.parcels) for f in an.families}
    layout = {f.name: set(f.regions) for f in model.family_layout.families}
    return anatomy, layout


def test_the_two_vocabularies_name_the_same_families(partitions) -> None:
    anatomy, layout = partitions
    assert set(anatomy) == set(layout), (
        "the anatomy prior and the state layout disagree about which families "
        f"exist.\n  anatomy only: {sorted(set(anatomy) - set(layout))}\n"
        f"  layout only:  {sorted(set(layout) - set(anatomy))}\n"
        "These are two spellings of one concept (O-7); a family present in one "
        "and not the other means some region has state with no prior, or a prior "
        "with no state."
    )


def test_the_two_vocabularies_agree_on_membership(partitions) -> None:
    """Same names is not the same partition. This is the assertion with content."""
    anatomy, layout = partitions
    disagree = {k: (sorted(anatomy[k] ^ layout[k])[:6]) for k in anatomy if anatomy[k] != layout[k]}
    assert not disagree, (
        f"families disagree about their members: {disagree}. A region assigned to "
        "different families by the prior and by the state layout gets one "
        "family's dynamics and another's receptor profile, and no shape check "
        "anywhere will notice -- both are still 414 regions wide."
    )


def test_the_partition_is_total_and_disjoint(partitions) -> None:
    """Every region in exactly one family, in both vocabularies."""
    an = load_anatomy()
    for label, part in zip(("anatomy", "layout"), partitions):
        members = [r for v in part.values() for r in v]
        assert len(members) == len(set(members)), (
            f"{label}: a region belongs to more than one family"
        )
        assert set(members) == set(range(an.n_regions)), (
            f"{label}: the families do not cover exactly 0..{an.n_regions - 1}; "
            f"got {len(set(members))} distinct regions"
        )


def test_the_identity_field_names_still_differ(partitions) -> None:
    """An inverse-category guard, deliberately, with an expiry condition.

    ``reports/decorative_guards.md`` warns that a test demanding a defect still
    exist goes stale the moment it is fixed. This one is written to fail *loudly
    and usefully* at that moment: if both vocabularies grow a common identity
    field, O-7 has been closed and this file should be replaced by whatever the
    unified ontology asserts — not amended to keep passing.
    """
    an = load_anatomy()
    model = SCWBD(load_config(CONFIG).model, an)
    a = an.families[0]
    b = model.family_layout.families[0]
    assert hasattr(a, "family_id") and not hasattr(a, "name"), (
        "anatomy.RegionFamily gained a `name`; if the vocabularies have converged, "
        "close O-7 and delete this test rather than relaxing it"
    )
    assert hasattr(b, "name") and not hasattr(b, "family_id"), (
        "the layout's RegionFamily gained a `family_id`; same as above"
    )
