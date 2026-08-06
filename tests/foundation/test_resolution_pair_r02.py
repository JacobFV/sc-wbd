"""R02 now has something to check, and these tests are the proof it fires.

``scwbd/foundation/compiler_bridge.py`` used to declare a one-element resolution
poset and say so:

    SC-WBD-001-beta declares no cross-scale prolongation, so R02 has nothing to
    object to -- which is the honest state of affairs, not an omission.

An empty poset does not make R02 honest, it makes it inert: eleven refusals on
the tin, ten of them able to fail.  ``reports/decorative_guards.md`` is a list
of ~26 checks in this project that had that shape, and coverage that cannot fail
is worse than no coverage because it is counted.

So the pair is declared now (``reports/transforms/resolution_pair.md``), and
every way of breaking it is exercised below against the *production* schema, not
a synthetic fixture.  Six distinct breakages, six refusals.  The last test is
the control: unbroken, the same path yields no R02 at all, so the fixtures above
are firing on what they claim to be firing on.

Note what is deliberately *not* wired to R02: the measured boundary result
(the parcel support carries 5.6% of the whitened lead field, so the coarse view
does not preserve the EEG observable).  That is a modelling fact about the
artifact, not a violation of the map contract, and dressing it as a refusal
would let a measurement masquerade as a rule.  It is recorded in the report and
in ARCHITECTURE.md Sec. 5b instead.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

cb = pytest.importorskip("scwbd.foundation.compiler_bridge")
checks = pytest.importorskip("scwbd.compiler.checks")

from scwbd.schema.poset import ResolutionPoset  # noqa: E402
from scwbd.transforms import resolution_pair as rp  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not cb.compiler_available(), reason="scwbd.compiler unavailable"
)


def r02(poset: ResolutionPoset) -> list:
    """Every R02 refusal the compiler raises for this poset."""
    schema = SimpleNamespace(resolution_poset=poset)
    return list(checks.check_r02(schema, SimpleNamespace()))


@pytest.fixture(scope="module")
def poset() -> ResolutionPoset:
    return cb._poset()


def _swap_pair(poset: ResolutionPoset, **update) -> ResolutionPoset:
    pair = poset.maps[0].model_copy(update=update)
    return poset.model_copy(update={"maps": (pair,)})


# --------------------------------------------------------------------------
# the declaration itself
# --------------------------------------------------------------------------
def test_the_poset_is_no_longer_trivial(poset):
    assert set(poset.ids()) == {rp.SCALE_FINE, rp.SCALE_COARSE}
    assert poset.lt(rp.SCALE_FINE, rp.SCALE_COARSE)
    assert not poset.leq(rp.SCALE_COARSE, rp.SCALE_FINE)
    assert len(poset.prolongations()) == 1


def test_the_declared_pair_carries_measured_numbers_not_defaults(poset):
    pair = poset.maps[0]
    assert pair.restriction is not None and pair.prolongation is not None
    assert pair.prolongation.returns_distribution, (
        "P must return a distribution: 7430 of the 7498 fine directions are in "
        "the null space of R"
    )
    for slot in (pair.restriction, pair.prolongation):
        assert slot.roundtrip_residual is not None
        assert slot.roundtrip_tolerance is not None
        assert slot.asset_ref == rp.MEASUREMENT_RELPATH
    assert pair.out_of_support_policy == "inflate_uncertainty"
    assert pair.landmark_coverage is not None


# --------------------------------------------------------------------------
# six ways to break it; six refusals
# --------------------------------------------------------------------------
def test_R02_fires_when_the_measured_artefact_is_missing(monkeypatch):
    """The fail-closed path: no measurement, no licence."""
    monkeypatch.setattr(rp, "load_measurement", lambda *a, **k: None)
    fired = r02(cb._poset())
    assert len(fired) == 1
    detail = fired[0].detail
    for expected in (
        "no restriction partner",
        "no round-trip test",
        "no held-out landmark test",
        "no landmark coverage",
        "no out-of-support uncertainty policy",
    ):
        assert expected in detail, detail


def test_R02_fires_without_a_restriction_partner(poset):
    fired = r02(_swap_pair(poset, restriction=None))
    assert [f.code for f in fired] == ["R02"]
    assert "no restriction partner" in fired[0].detail


def test_R02_fires_when_coverage_is_below_the_declared_requirement(poset):
    fired = r02(_swap_pair(poset, landmark_coverage=0.5))
    assert [f.code for f in fired] == ["R02"]
    assert "landmark coverage 0.5" in fired[0].detail


def test_R02_fires_without_an_out_of_support_policy(poset):
    fired = r02(_swap_pair(poset, out_of_support_policy=None))
    assert [f.code for f in fired] == ["R02"]
    assert "no out-of-support uncertainty policy" in fired[0].detail


def test_R02_fires_when_the_prolongation_understates_its_uncertainty(poset):
    """The substantive check: the tests were run *and* they came back bad.

    Before this, ``roundtrip_tested=True`` alongside any residual whatsoever
    satisfied R02 -- the guard reported that somebody ran a test, not that the
    test succeeded.
    """
    p = poset.maps[0]
    broken = p.prolongation.model_copy(
        update={"roundtrip_residual": p.prolongation.roundtrip_tolerance * 10.0}
    )
    fired = r02(_swap_pair(poset, prolongation=broken))
    assert [f.code for f in fired] == ["R02"]
    assert "round-trip residual" in fired[0].detail
    assert "exceeds its tolerance" in fired[0].detail


def test_R02_fires_when_R_and_P_stop_being_a_pair(poset):
    """``R P != I`` is what "not paired" means numerically."""
    p = poset.maps[0]
    broken = p.restriction.model_copy(update={"roundtrip_residual": 0.4})
    fired = r02(_swap_pair(poset, restriction=broken))
    assert [f.code for f in fired] == ["R02"]
    assert "exceeds its tolerance" in fired[0].detail


def test_R02_fires_when_the_map_runs_the_wrong_way_up_the_poset(poset):
    """§2.6: "Some pairs have no defensible map." Nor is one direction the other."""
    reversed_poset = poset.model_copy(update={"relations": ()})
    fired = r02(reversed_poset)
    codes = [f.code for f in fired]
    assert codes.count("R02") >= 1
    assert any("not\nordered" in f.detail or "not ordered" in f.detail for f in fired), [
        f.detail for f in fired
    ]


# --------------------------------------------------------------------------
# the control
# --------------------------------------------------------------------------
def test_the_production_pair_raises_no_R02(poset):
    """Unbroken, the same path is silent -- so the six above are real."""
    assert r02(poset) == []
    assert poset.orphan_prolongations() == ()
    assert poset.unordered_maps() == ()


def test_the_committed_measurement_licenses_the_production_declaration():
    m = rp.load_measurement()
    assert m is not None, (
        f"{rp.MEASUREMENT_RELPATH} is missing. Regenerate it with "
        "benchmarks/transforms/resolution_pair.py; the schema will refuse to "
        "compile until it exists, which is the intended behaviour."
    )
    assert m.authority_policy == rp.AUTHORITY_POLICY == "fine_authoritative"


# --------------------------------------------------------------------------
# end to end, through the real compiler
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def production_schema():
    """The real ``BrainSchema``.  Slow (~2 min) and deliberately not mocked.

    Everything above tests ``check_r02`` against a hand-built poset, which
    proves the check works but not that the production schema reaches it.  This
    fixture closes that gap: the same object ``FoundationTrainer`` compiles.
    """
    from scwbd.foundation.anatomy import load_anatomy
    from scwbd.foundation.config import load_config
    from scwbd.foundation.mixture import SourceSpec

    anat = load_anatomy(
        device="cpu", n_cortex=40, n_subcortex=12, n_cerebellum=8, density=0.15, seed=7
    )
    cfg = load_config(REPO / "configs" / "scwbd_001_beta.yaml")
    specs = SourceSpec.load_dir(cfg.mixture_cards)
    probe = [
        SourceSpec(**{**s.as_dict(), "gradient_permission": s.compiler_permission})
        for s in specs.values()
    ]
    return cb.build_foundation_schema(anat, probe)


def test_the_production_schema_carries_the_pair_and_compiles(production_schema):
    from scwbd.compiler import compile as compile_schema

    assert len(production_schema.resolution_poset.maps) == 1
    compile_schema(production_schema, claim=cb.build_foundation_claim())


def test_breaking_the_pair_refuses_the_production_compile(production_schema):
    """The claim this whole task exists to make good on."""
    from scwbd.compiler import compile as compile_schema
    from scwbd.schema.refusals import REFUSALS, CompilerRefusal

    poset = production_schema.resolution_poset
    pair = poset.maps[0].model_copy(update={"restriction": None})
    broken = production_schema.model_copy(
        update={"resolution_poset": poset.model_copy(update={"maps": (pair,)})}
    )
    with pytest.raises(CompilerRefusal) as exc:
        compile_schema(broken, claim=cb.build_foundation_claim())
    assert exc.value.code == "R02"
    assert exc.value.remedy == REFUSALS["R02"].remedy
    assert "no restriction partner" in exc.value.detail


def test_the_binding_table_still_names_the_declared_scale_map():
    """A rename in ``resolution_pair`` must not leave the binding vacuous."""
    for slot, key in (
        ("restriction", cb.SCALE_MAP_RESTRICTION_KEY),
        ("prolongation", cb.SCALE_MAP_PROLONGATION_KEY),
    ):
        assert key == f"scale_map:{rp.SCALE_FINE}->{rp.SCALE_COARSE}:{slot}"
        assert key in cb.FOUNDATION_BINDING
        assert cb.FOUNDATION_BINDING[key] == (), (
            "R and P carry no trainable tensor; if that changes the entry must "
            "name the tensor, not stay empty"
        )
