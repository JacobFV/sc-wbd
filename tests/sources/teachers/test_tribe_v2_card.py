"""The TRIBE v2 teacher card, and proof that its guard can fire.

Half of this file asserts the card says what it should.  The other half breaks
the card on purpose and demands the alarm sound, because a guard nobody has
watched fire is indistinguishable from one that cannot fire
(reports/decorative_guards.md).  Every ``CardError`` branch in
``scwbd.sources.teachers.check_teacher_quarantine`` has a matching failure test
below; ``test_every_refusal_branch_is_exercised`` fails if a new branch is added
without one.
"""

from __future__ import annotations

import copy
import re
import textwrap
from pathlib import Path

import pytest
import yaml

from scwbd.sources.cards import CARD_DIR, CardError, SourceCardDoc
from scwbd.sources.teachers import (
    PERMANENTLY_FORBIDDEN,
    REQUIRED_ABLATION_ARMS,
    TEACHER_DIR,
    check_teacher_quarantine,
    iter_teacher_card_paths,
    load_teacher_card,
)

CARD_PATH = TEACHER_DIR / "tribe-v2.yaml"


@pytest.fixture(scope="module")
def doc() -> SourceCardDoc:
    return load_teacher_card(CARD_PATH)


@pytest.fixture
def raw() -> dict:
    return yaml.safe_load(CARD_PATH.read_text())


def _mutated(raw: dict, **_unused) -> SourceCardDoc:
    """A card doc built from a mutated mapping, keeping the original path."""
    return SourceCardDoc(path=CARD_PATH, data=raw)


# ---------------------------------------------------------------------------
# 1. The card says what it should
# ---------------------------------------------------------------------------


def test_card_exists_and_validates(doc):
    assert doc.id == "tribe-v2"
    assert doc.path.stem == doc.id
    assert doc.typed is not None, f"typed projection did not bind: {doc.typed_error}"


def test_role_is_distillation_and_locked(doc):
    assert doc.role == "distillation"
    assert doc.data["split_policy"]["role_locked"] is True


def test_teacher_is_not_in_the_dataset_card_directory():
    """A glob over CARD_DIR must not be able to find a teacher."""
    assert CARD_PATH.resolve().parent != CARD_DIR.resolve()
    assert CARD_PATH.name not in {p.name for p in CARD_DIR.glob("*.yaml")}


def test_discrepancy_is_unknown_not_zero(doc):
    """Absence must write something -- and what it writes is not 0."""
    led = doc.data["ledger"]
    assert led["model_discrepancy"] == "unknown"
    assert led["model_discrepancy_status"] == "NOT_MEASURED"
    assert led["model_discrepancy_reason"].strip(), "an unmeasured field must say why"
    assert led["bias_interval"] == "unknown"
    assert led["bias_status"] == "unknown"


def test_no_gradient_is_permitted_while_the_discrepancy_is_unmeasured(doc):
    gp = doc.data["gradient_permission"]
    assert gp["allow"] == []
    assert gp["forbid"] == ["*"]
    assert gp["enabled"] is False


def test_permanently_forbidden_targets_are_absent_from_every_allow_list(doc):
    gp = doc.data["gradient_permission"]
    prereg = {e["target"] for e in gp["preregistered_allow_if_validated"]}
    named = {e["target"] for e in gp["preregistered_forbid_permanently"]}
    assert prereg.isdisjoint(PERMANENTLY_FORBIDDEN)
    for target in PERMANENTLY_FORBIDDEN:
        assert target in named, f"{target} must be named as permanently forbidden"


def test_preregistered_allow_list_only_names_observation_and_perceptual_ports(doc):
    """The permitted interfaces, per body.tex sec. 6.3, and nothing else."""
    prereg = {e["target"] for e in
              doc.data["gradient_permission"]["preregistered_allow_if_validated"]}
    assert prereg == {
        "observe.bold.hemodynamic_head.readout",
        "encode.perception.visual_port",
        "encode.perception.auditory_port",
        "encode.language.context_encoder",
    }
    for entry in doc.data["gradient_permission"]["preregistered_allow_if_validated"]:
        assert "ledger.model_discrepancy" in entry["requires"], (
            f"{entry['target']} must be gated on a measured discrepancy"
        )


def test_every_preregistered_ablation_arm_is_present(doc):
    arms = {a["id"] for a in doc.data["preregistered_ablation_branch"]["arms"]}
    assert set(REQUIRED_ABLATION_ARMS) <= arms
    assert doc.data["preregistered_ablation_branch"]["status"] == "NOT_RUN"


def test_teacher_adds_no_participants(doc):
    assert doc.data["population"]["n_participants"] == 0
    assert doc.data["observation"]["observed_variables"] == []
    assert doc.data["observation"]["likelihood_kind"] == "none"
    assert doc.data["observation"]["forward_physics"].lstrip().startswith("NONE")


def test_noncommercial_licence_constraint_is_recorded(doc):
    gov = doc.data["governance"]
    assert gov["license_is_noncommercial"] is True
    assert "NonCommercial" in gov["license_text_excerpt"]
    assert "noncommercial_only" in gov["purpose_limits_list"]
    assert gov["may_release_weights"] is False
    assert "commercial" in gov["noncommercial_constraint"].lower()


def test_credentials_are_reported_honestly(doc):
    """Weights are ungated; the text extractor is not.  Both must be stated."""
    gov = doc.data["governance"]
    assert gov["credentials_required"] is True
    detail = gov["credentials_detail"]
    assert "FALSE" in detail and "TRUE" in detail
    extractors = doc.data["observation"]["required_feature_extractors"]
    gated = {e["model"]: e["hf_gated"] for e in extractors}
    assert gated["meta-llama/Llama-3.2-3B"] == "manual"
    assert gated["facebook/vjepa2-vitg-fpc64-256"] is False
    assert gated["facebook/w2v-bert-2.0"] is False
    assert all(e["obtained"] is False for e in extractors)


def test_measured_probes_are_present_and_are_measurements(doc):
    """Numbers that came from running the thing, not from the model card."""
    p = doc.data["ledger"]["measured_probes"]
    assert p["runs_on_cpu"] is True
    assert p["strict_state_dict_load"] is True
    assert p["output_shape"] == [1, 20484, 100]
    assert p["released_predictor_shape"] == [1, 2048, 20484]
    assert 0.0 < p["modality_ablation_pearson_r"] < 1.0
    assert p["cpu_peak_rss_gib"] < 12.0, "the probe ran under MemoryMax=12G"
    assert p["output_sd_video_only"] < p["output_sd_trimodal"]


def test_manifest_root_hash_matches_the_recorded_file_hashes(doc):
    """The manifest digest must be recomputable from the card's own hashes."""
    import hashlib
    import json

    fm = doc.data["identity"]["file_manifest"]
    man = json.loads((TEACHER_DIR / "tribe-v2__f894e783.json").read_text())
    payload = {k: v["sha256"] for k, v in man["files"].items()}
    root = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert root == fm["manifest_sha256"] == man["manifest_sha256"]
    assert fm["total_bytes"] == man["total_bytes"] == sum(
        v["bytes"] for v in man["files"].values()
    )
    for name, digest in fm["exemplars"].items():
        assert man["files"][name]["sha256"] == digest
        assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_every_teacher_card_in_the_package_passes_the_guard():
    paths = list(iter_teacher_card_paths())
    assert paths, "no teacher cards found"
    for p in paths:
        load_teacher_card(p)


# ---------------------------------------------------------------------------
# 2. The guard fires.  Break it on purpose; demand the alarm.
# ---------------------------------------------------------------------------


def test_guard_fires_on_likelihood_role(raw):
    raw["role"] = "likelihood"
    with pytest.raises(CardError, match="teacher role must be"):
        check_teacher_quarantine(_mutated(raw))


def test_guard_fires_on_a_generative_likelihood_kind(raw):
    raw["observation"]["likelihood_kind"] = "generative"
    with pytest.raises(CardError, match="likelihood_kind must be 'none'"):
        check_teacher_quarantine(_mutated(raw))


def test_guard_fires_when_a_teacher_claims_participants(raw):
    raw["population"]["n_participants"] = 25
    with pytest.raises(CardError, match="n_participants must be 0"):
        check_teacher_quarantine(_mutated(raw))


def test_guard_fires_on_a_zero_model_discrepancy(raw):
    """The failure this project has already named: unmeasured recorded as zero."""
    raw["ledger"]["model_discrepancy"] = 0.0
    with pytest.raises(CardError, match="zero teacher variance"):
        check_teacher_quarantine(_mutated(raw))


def test_guard_fires_when_gradients_open_without_a_measured_discrepancy(raw):
    raw["gradient_permission"]["allow"] = [
        {"target": "observe.bold.hemodynamic_head.readout", "scales": ["parcel"]}
    ]
    with pytest.raises(CardError, match="disables the gradient path"):
        check_teacher_quarantine(_mutated(raw))


def test_guard_fires_when_the_teacher_is_enabled_without_a_measured_discrepancy(raw):
    raw["gradient_permission"]["enabled"] = True
    with pytest.raises(CardError, match="off by default"):
        check_teacher_quarantine(_mutated(raw))


@pytest.mark.parametrize("target", PERMANENTLY_FORBIDDEN)
def test_guard_fires_on_each_permanently_forbidden_target(raw, target):
    """Every one of them, individually -- not just the first in the list."""
    raw["ledger"]["model_discrepancy"] = 0.031          # pretend it was measured
    raw["gradient_permission"]["preregistered_allow_if_validated"].append(
        {"target": target, "scales": ["parcel"], "requires": ["ledger.model_discrepancy"]}
    )
    with pytest.raises(CardError, match="permanently forbidden"):
        check_teacher_quarantine(_mutated(raw))


def test_guard_fires_on_a_forbidden_target_reached_by_glob(raw):
    """`anatomy.subcortical.*` must catch a concrete child, not only itself."""
    raw["ledger"]["model_discrepancy"] = 0.031
    raw["gradient_permission"]["preregistered_allow_if_validated"].append(
        {"target": "anatomy.subcortical.thalamus_gain", "requires": []}
    )
    with pytest.raises(CardError, match="permanently forbidden"):
        check_teacher_quarantine(_mutated(raw))


@pytest.mark.parametrize("arm", REQUIRED_ABLATION_ARMS)
def test_guard_fires_when_any_control_arm_is_dropped(raw, arm):
    branch = raw["preregistered_ablation_branch"]
    branch["arms"] = [a for a in branch["arms"] if a["id"] != arm]
    with pytest.raises(CardError, match="missing required control arm"):
        check_teacher_quarantine(_mutated(raw))


def test_guard_fires_when_a_teacher_card_sits_in_the_dataset_directory(raw, tmp_path):
    """The glob-sweep failure: a teacher card placed where datasets are found."""
    doc = SourceCardDoc(path=CARD_DIR / "tribe-v2.yaml", data=raw)
    with pytest.raises(CardError, match="must not live in"):
        check_teacher_quarantine(doc)


def test_the_unmutated_card_passes_all_of_the_above(raw):
    """Control: the alarms above are not simply always-on."""
    check_teacher_quarantine(_mutated(copy.deepcopy(raw)))


def test_every_refusal_branch_is_exercised():
    """A new ``raise CardError`` in the guard without a test here fails this test.

    Counting tests would be wrong -- one branch may deserve two tests, as the
    permanently-forbidden branch does.  So instead: pull the literal message of
    every ``raise`` site out of the guard's source, pull every ``match=`` regex
    out of this file, and demand that each site is claimed by at least one
    regex.  An unclaimed site is a refusal nobody has watched fire.
    """
    import ast
    import inspect

    from scwbd.sources import teachers

    guard_src = textwrap.dedent(inspect.getsource(teachers.check_teacher_quarantine))
    tree = ast.parse(guard_src)
    messages: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)):
            continue
        if getattr(node.exc.func, "id", None) != "CardError":
            continue
        # second positional arg is the message; join implicit concatenation
        parts = [
            n.value for n in ast.walk(node.exc.args[1])
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        ]
        messages.append(" ".join(parts))
    assert messages, "no refusal branches found -- did the guard lose its raises?"

    this_src = Path(__file__).read_text()
    patterns = re.findall(r'pytest\.raises\(CardError,\s*match="([^"]+)"\)', this_src)
    assert patterns, "no match= patterns found in this file"

    unclaimed = [
        m for m in messages
        if not any(re.search(p, m) for p in patterns)
    ]
    assert not unclaimed, (
        "refusal branch(es) with no test watching them fire: "
        + "; ".join(repr(m[:80]) for m in unclaimed)
    )
