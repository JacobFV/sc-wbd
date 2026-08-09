"""R12 must fire on the artifact that caused it, and on nothing else.

``reports/decorative_guards.md``: a guard whose reading is constant with respect
to the question is worse than no guard.  So these tests are built as matched
sets -- the offending shape, the conformant shape, the declared control -- and
where a real config or a real partition exists it is used rather than a stub.

Provenance of the family fixtures below: they are the output of
``scwbd.foundation.families.derive_families(load_anatomy())`` on `wt/hodgkin`
at c896d16, executed on 2026-08-06 against the real 414-parcel prior, not
invented.  Eleven families, 414 regions, ``unpopulated == ['cerebellum']``.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from scwbd.schema import (
    ALL_REFUSAL_CODES,
    LOCAL_REFUSAL_CODES,
    MODEL_DESIGNATION,
    NON_OVERRIDABLE_CODES,
    REFUSAL_CODES,
    REFUSALS,
    ArmDeclaration,
    ClaimOverride,
    CompilerRefusal,
    assert_designation,
    check_r12,
    designation_for,
    r12_predicate,
    read_operator_assignment,
    read_prolongations,
)
from scwbd.schema.ids import ScaleId
from scwbd.schema.poset import MapSpec, ResolutionPoset, ScaleMapPair, ScaleNode

REPO = Path(__file__).resolve().parents[2]
BASE = MODEL_DESIGNATION


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
def _released_config() -> dict:
    """The released run-1 config, read from the file that produced the artifact."""
    return yaml.safe_load((REPO / "configs" / "scwbd_001_beta.yaml").read_text())


def _run1_shape() -> dict:
    """The released config *as it was emitted*: no arm declaration at all.

    This is the historical artifact, reproduced from the real file rather than
    described.  If R12 does not fire on this, it does not fire on the thing it
    was written for.
    """
    cfg = _released_config()
    cfg.pop("arm", None)
    return cfg


#: The real partition, measured on `wt/hodgkin`@c896d16 (see module docstring).
#: 4 distinct backends over 414 populated regions; cerebellum declared, empty.
REAL_FAMILIES: tuple[tuple[str, str, int], ...] = (
    ("amygdala", "learned", 2),
    ("basal_ganglia", "basal_ganglia_gate", 8),
    ("cerebellum", "cerebellar_forward_model", 0),
    ("cortex_cont", "learned", 52),
    ("cortex_default", "learned", 91),
    ("cortex_dorsattn", "learned", 46),
    ("cortex_limbic", "learned", 26),
    ("cortex_salventattn", "learned", 47),
    ("cortex_sommot", "learned", 77),
    ("cortex_vis", "learned", 61),
    ("hippocampus", "hippocampal_code", 2),
    ("thalamus", "thalamic_relay", 2),
)


def _family_report(families=REAL_FAMILIES, *, arm: str = "treatment") -> dict:
    """The shape of ``SCWBD.family_report()`` for the treatment arm."""
    return {
        "layout": "family_padded",
        "narrowing": "padded-family-state",
        "family_state": True,
        "ablation_arm": arm,
        "n_regions": sum(n for _, _, n in families),
        "partition": {
            "n_families": len(families),
            "n_regions": sum(n for _, _, n in families),
            "unpopulated": [name for name, _, n in families if n == 0],
            "partition_source": "anatomy_prior",
            "families": [
                {"name": name, "backend": backend, "n_regions": n}
                for name, backend, n in families
            ],
        },
    }


def _control_report() -> dict:
    """``SCWBD.family_report()`` for the control arm."""
    return {
        "family_state": False,
        "ablation_arm": "control",
        "local_core": "learned",
        "n_regions": 414,
        "state_dim": 28,
    }


class _Claim:
    """Duck-typed stand-in for ``foundation.manifest.Claim``."""

    def __init__(self, id, statement="", caveats=(), requires_family_state=False):
        self.id = id
        self.statement = statement
        self.caveats = tuple(caveats)
        self.requires_family_state = requires_family_state


class _Manifest:
    """Duck-typed stand-in for ``foundation.manifest.ClaimManifest``."""

    def __init__(self, regional_state=None, claims=(), model_id=BASE):
        self.regional_state = dict(regional_state or {})
        self.claims = list(claims)
        self.model_id = model_id


# ======================================================================
# it fires
# ======================================================================
def test_r12_fires_on_the_released_run1_config():
    """The exact shape that shipped as SC-WBD-001-beta is refused."""
    cfg = _run1_shape()
    assert read_operator_assignment(cfg["model"]).is_constant
    assert not read_prolongations(cfg["model"]).declares_prolongation

    with pytest.raises(CompilerRefusal) as excinfo:
        assert_designation(cfg)
    r = excinfo.value
    assert r.code == "R12"
    assert r.remedy == REFUSALS["R12"].remedy
    assert "11.4" in r.detail
    assert r.evidence["backends"] == ["learned"]
    assert r.evidence["n_prolongations"] == 0
    assert r.evidence["arm_role"] == "model"


def test_r12_fires_on_every_undeclared_single_backend_config_in_the_repo():
    """Sweep the real configs: any of them without a declaration must refuse."""
    seen = 0
    for path in sorted(REPO.glob("configs/**/*.yaml")):
        raw = yaml.safe_load(path.read_text()) or {}
        if "base" in raw or not isinstance(raw.get("model"), dict):
            continue  # fragments and source cards are not run configs
        if "local_core" not in raw["model"]:
            continue
        stripped = copy.deepcopy(raw)
        stripped.pop("arm", None)
        seen += 1
        with pytest.raises(CompilerRefusal, match="R12"):
            assert_designation(stripped)
    assert seen >= 2, f"expected to sweep several run configs, swept {seen}"


def test_r12_fires_when_only_an_unpopulated_family_carries_a_second_backend():
    """The decisive case: heterogeneity that reaches no parcel.

    On the real 414-parcel prior ``cerebellum`` is declared and holds **zero**
    regions.  A partition whose only differently-typed family is that one is
    one operator for every region, however many families it declares -- and the
    artifact's own ``ablation_arm`` still says ``treatment``.  Counting declared
    backends instead of backends that reach a region is exactly how a guard
    becomes decorative.
    """
    families = tuple(
        (name, "learned" if n > 0 else backend, n)
        for name, backend, n in REAL_FAMILIES
    )
    report = _family_report(families)
    assert report["ablation_arm"] == "treatment"
    assert report["partition"]["unpopulated"] == ["cerebellum"]
    # a naive count over *declared* backends would see two and permit this
    assert len({b for _, b, _ in families}) == 2

    assignment = read_operator_assignment(regional_state=report)
    assert assignment.is_constant, "populated families all run one backend"
    assert assignment.unpopulated == ("cerebellum",)
    assert assignment.n_regions == 414

    with pytest.raises(CompilerRefusal) as excinfo:
        assert_designation({"model": {}}, regional_state=report)
    assert "self-reported arm is wrong" in excinfo.value.detail
    assert excinfo.value.evidence["unpopulated"] == ["cerebellum"]


def test_r12_fires_when_a_treatment_partition_collapses_to_one_backend():
    """Eleven populated families, all on the same backend, is still the control."""
    families = tuple((name, "learned", max(n, 1)) for name, _, n in REAL_FAMILIES)
    with pytest.raises(CompilerRefusal, match="self-reported arm is wrong"):
        assert_designation({"model": {}}, regional_state=_family_report(families))


def test_r12_fires_on_a_family_state_claim_with_no_artifact_to_corroborate_it():
    """A config may not claim per-family operators nothing can check."""
    cfg = _run1_shape()
    cfg["model"]["family_state"] = True
    cfg["model"]["family_cores"] = {"hippocampus": "hippocampal_code"}
    with pytest.raises(CompilerRefusal, match="no artifact family report"):
        assert_designation(cfg)


def test_r12_fires_when_the_operator_assignment_cannot_be_read():
    cfg = _run1_shape()
    cfg["model"].pop("local_core")
    with pytest.raises(CompilerRefusal, match="what operator any region runs"):
        assert_designation(cfg)


def test_r12_fires_when_the_config_declares_a_prolongation_the_poset_lacks():
    """The config declaration is not a free pass; the poset is authoritative."""
    cfg = _run1_shape()
    cfg["model"]["scale_prolongations"] = ["voxel<=parcel"]
    empty = ResolutionPoset(
        nodes=(ScaleNode(id=ScaleId("parcel"), label="anatomical parcel"),)
    )
    with pytest.raises(CompilerRefusal, match="does not carry"):
        assert_designation(cfg, poset=empty)


@pytest.mark.parametrize(
    "arm",
    [
        {"role": "control"},  # no ablation named
        {"role": "control", "controls_for": "11.4:x"},  # no justification
        {"role": "control", "controls_for": "11.4:x", "justification": "control"},
        {"role": "model", "controls_for": "11.4:x"},  # half-edited
    ],
)
def test_r12_fires_on_a_half_made_control_declaration(arm):
    cfg = _run1_shape()
    cfg["arm"] = arm
    with pytest.raises(CompilerRefusal, match="arm declaration is not valid"):
        assert_designation(cfg)


# -- the prose half, taken from Hodgkin's predicate --------------------------
@pytest.mark.parametrize(
    "statement",
    [
        "SC-WBD-001-beta learns heterogeneous regional state across 414 parcels.",
        "The model carries region-indexed state with per-family operators.",
        "Structured regional state improves held-out forecast likelihood.",
        "An operator-valued regional state is fit per family.",
    ],
)
def test_r12_fires_on_control_arm_prose_that_asserts_the_differentiator(statement):
    """A correct artifact described in the words of a different one."""
    m = _Manifest(
        regional_state=_control_report(),
        claims=[_Claim("c1", statement=statement)],
    )
    with pytest.raises(CompilerRefusal, match="sec. 2.1's differentiator"):
        r12_predicate(m)


def test_r12_fires_on_a_requires_family_state_claim_on_the_control_arm():
    m = _Manifest(
        regional_state=_control_report(),
        claims=[_Claim("c1", statement="a neutral sentence", requires_family_state=True)],
    )
    with pytest.raises(CompilerRefusal, match="requires_family_state"):
        r12_predicate(m)


def test_r12_fires_on_a_differentiator_claim_with_no_arm_evidence_at_all():
    """A manifest asserting the differentiator while saying nothing about its arm."""
    m = _Manifest(claims=[_Claim("c1", statement="per-family backends are fit")])
    with pytest.raises(CompilerRefusal, match="declares no regional-state arm"):
        r12_predicate(m)


# ======================================================================
# it does not over-fire
# ======================================================================
def test_r12_does_not_fire_on_the_real_conformant_partition():
    """The 11-family, 414-parcel partition Hodgkin actually produces."""
    report = _family_report()
    assignment = read_operator_assignment(regional_state=report)
    assert not assignment.is_constant
    assert assignment.distinct == (
        "basal_ganglia_gate",
        "hippocampal_code",
        "learned",
        "thalamic_relay",
    )
    assert assignment.n_regions == 414
    assert assignment.unpopulated == ("cerebellum",)
    # honest about how thin the heterogeneity is, without refusing on it
    assert assignment.dominant_share() == pytest.approx(402 / 414, abs=1e-6)

    cfg = _run1_shape()
    cfg["model"]["family_state"] = True
    cfg["model"]["scale_prolongations"] = ["voxel<=parcel"]
    assert not list(check_r12(config=cfg, regional_state=report))
    assert assert_designation(cfg, regional_state=report) == BASE


def test_r12_does_not_fire_on_a_conformant_artifact_making_the_claim():
    """The prose is only refused when the artifact does not have the property."""
    m = _Manifest(
        regional_state=_family_report(),
        claims=[_Claim("c1", statement="heterogeneous regional state is fit per family")],
    )
    r12_predicate(m)  # must not raise


def test_r12_does_not_fire_on_a_properly_declared_control():
    cfg = _run1_shape()
    cfg["arm"] = {
        "role": "control",
        "controls_for": "11.4:structured_regional_state",
        "justification": (
            "one operator for all regions and no prolongation; the "
            "equal-capacity generic control arm"
        ),
    }
    assert not list(check_r12(config=cfg))
    assert (
        assert_designation(cfg)
        == "SC-WBD-001-beta-CONTROL[11.4:structured_regional_state]"
    )


def test_the_released_config_as_it_now_stands_is_permitted_only_as_a_control():
    """The file on disk passes -- and passes as the control, not the model."""
    cfg = _released_config()
    assert cfg["arm"]["role"] == "control"
    assert (
        assert_designation(cfg)
        == "SC-WBD-001-beta-CONTROL[11.4:structured_regional_state]"
    )


def _two_scale_poset() -> ResolutionPoset:
    """A compiled, R02-shaped voxel<=parcel pair -- the evidence R12 accepts."""
    return ResolutionPoset(
        nodes=(
            ScaleNode(id=ScaleId("voxel"), label="voxel"),
            ScaleNode(id=ScaleId("parcel"), label="parcel"),
        ),
        relations=((ScaleId("voxel"), ScaleId("parcel")),),
        maps=(
            ScaleMapPair(
                fine=ScaleId("voxel"),
                coarse=ScaleId("parcel"),
                restriction=MapSpec(name="parcel_average", kind="restriction"),
                prolongation=MapSpec(
                    name="parcel_to_voxel", kind="prolongation", returns_distribution=True
                ),
                landmark_coverage=0.9,
                roundtrip_tested=True,
                landmark_tested=True,
                out_of_support_policy="return_distribution",
            ),
        ),
    )


def test_one_condition_alone_is_not_the_control_arm():
    """R12 needs *both* conditions. Neither half is the sec. 11.4 control.

    Condition 2 is satisfied by a COMPILED POSET, not by
    ``model.scale_prolongations``. This test used to grant it from the config
    field, which is exactly the exemption
    ``tests/foundation/test_resolution_pair_r02.py`` had to pin the field empty
    to prevent: "a config key that switches a refusal off is not a declaration,
    it is an exemption". The intent of this test is unchanged; the evidence it
    offers for the second half is now evidence.
    """
    report = _family_report()
    het_only = _run1_shape()
    het_only["model"]["family_state"] = True
    assert not list(check_r12(config=het_only, regional_state=report))

    prolongation_only = _run1_shape()
    assert not list(check_r12(config=prolongation_only, poset=_two_scale_poset()))


def test_the_config_field_alone_cannot_discharge_condition_two():
    """The exemption, asserted closed.

    Constant operators plus a prolongation named only in the config is the
    control arm, and saying so in a YAML key does not change that. Before this
    was fixed, the same input returned no refusal at all.
    """
    cfg = _run1_shape()
    cfg["model"]["scale_prolongations"] = ["voxel<=parcel"]
    refusals = list(check_r12(config=cfg))
    assert refusals, (
        "model.scale_prolongations alone switched R12 off -- the exemption is back"
    )

    # And with a differentiator claim it is refused as the control arm, with a
    # message that says why the declaration did not count rather than the older
    # and now-false 'no declared prolongation'.
    claimed = list(
        check_r12(
            config=cfg,
            claims=[_Claim("c1", statement="Heterogeneous regional state per parcel.")],
        )
    )
    assert claimed
    detail = " ".join(str(getattr(r, "detail", "")) for r in claimed)
    assert "cannot discharge this refusal" in detail, detail


def test_a_bare_manifest_with_no_arm_and_no_offending_claim_is_left_alone():
    """Manifests not yet attached to a checkpoint must keep validating."""
    r12_predicate(_Manifest())
    r12_predicate(_Manifest(claims=[_Claim("c", statement="EEG forecasts are reported")]))


def test_a_real_poset_with_a_prolongation_satisfies_condition_two():
    cfg = _run1_shape()
    poset = ResolutionPoset(
        nodes=(
            ScaleNode(id=ScaleId("voxel"), label="voxel"),
            ScaleNode(id=ScaleId("parcel"), label="parcel"),
        ),
        relations=((ScaleId("voxel"), ScaleId("parcel")),),
        maps=(
            ScaleMapPair(
                fine=ScaleId("voxel"),
                coarse=ScaleId("parcel"),
                restriction=MapSpec(name="parcel_average", kind="restriction"),
                prolongation=MapSpec(
                    name="parcel_to_voxel", kind="prolongation", returns_distribution=True
                ),
                landmark_coverage=0.9,
                roundtrip_tested=True,
                landmark_tested=True,
                out_of_support_policy="return_distribution",
            ),
        ),
    )
    assert not list(check_r12(config=cfg, poset=poset))


# ======================================================================
# the ownership seam and R12's standing in the refusal table
# ======================================================================
def test_the_seam_hodgkin_looks_up_exists_and_has_his_signature():
    """``ClaimManifest.refuse_r12`` resolves ``scwbd.schema.refusals.r12_predicate``.

    It calls ``canonical(self)`` with the manifest alone.  If this import path
    or arity ever changes, his lookup silently falls back to a second predicate
    -- two definitions, which is the thing the ownership ruling forbids.
    """
    import importlib
    import inspect

    mod = importlib.import_module("scwbd.schema.refusals")
    fn = getattr(mod, "r12_predicate")
    assert callable(fn)
    params = list(inspect.signature(fn).parameters.values())
    assert params[0].name == "manifest"
    assert params[0].default is inspect.Parameter.empty
    assert all(p.default is not inspect.Parameter.empty for p in params[1:]), (
        "every parameter after `manifest` must be optional; refuse_r12 calls "
        "canonical(self) with one argument"
    )
    fn(_Manifest())  # one-argument call, as his call site makes it


def test_r12_is_local_not_thesis():
    """R12 must never be quotable as a thesis requirement."""
    assert "R12" not in REFUSAL_CODES
    assert LOCAL_REFUSAL_CODES == ("R12",)
    assert ALL_REFUSAL_CODES == REFUSAL_CODES + ("R12",)
    assert REFUSALS["R12"].origin == "local"
    assert all(REFUSALS[c].origin == "thesis" for c in REFUSAL_CODES)


def test_r12_cannot_be_overridden():
    """An override demotes the claim class; it does not rename the artifact."""
    assert NON_OVERRIDABLE_CODES == frozenset({"R12"})
    with pytest.raises(ValueError, match="not overridable"):
        ClaimOverride(
            code="R12",
            justification="we would like to ship it under the model's name",
            resulting_claim_class="surrogate",
            approved_by="nobody",
        )
    ClaimOverride(
        code="R02",
        justification="documented, temporary",
        resulting_claim_class="surrogate",
        approved_by="architect",
    )


def test_designation_travels_with_the_arm():
    assert ArmDeclaration().designation() == BASE
    ctl = ArmDeclaration(
        role="control",
        controls_for="11.4:structured_regional_state",
        justification="the equal-capacity generic control arm, everything else matched",
    )
    assert ctl.designation() == f"{BASE}-CONTROL[11.4:structured_regional_state]"
    assert designation_for({"arm": ctl.model_dump()}) == ctl.designation()
