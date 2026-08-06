"""The validator, watched refusing.

The load-bearing test is :func:`test_refuses_the_shipped_config`: the
configuration that produced SC-WBD-001-beta must fail, naming the stage.  A
validator nobody has seen refuse anything is indistinguishable from one that
cannot.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scwbd.curriculum.spec import Curriculum, load_tier_policy
from scwbd.curriculum.tiers import load_mixture_cards
from scwbd.curriculum.validate import expand, parameter_universe, validate, validate_config

REPO = Path(__file__).resolve().parents[2]
BETA = REPO / "configs/scwbd_001_beta.yaml"
ORDERED = REPO / "configs/curriculum/scwbd_001_integrity_ordered.yaml"
TIERS = REPO / "configs/curriculum/tiers.yaml"

#: X01-X04 are the ordering class.  X06/X07/X09 are integration and provenance
#: findings, which a correctly *ordered* config can still carry.
ORDERING_CODES = {
    "X01_inverted_admission",
    "X02_founding_stage_not_pure",
    "X03_unfounded_parameter",
    "X04_permission_widens_with_tier",
}


@pytest.fixture(scope="module")
def universe() -> tuple[tuple[str, ...], str]:
    return parameter_universe(BETA)


# ======================================================================
# the headline
# ======================================================================
def test_refuses_the_shipped_config() -> None:
    v = validate_config(BETA, tiers_path=TIERS)
    assert not v.ok
    codes = set(v.codes())
    assert ORDERING_CODES <= codes, f"missing ordering refusals: {ORDERING_CODES - codes}"

    inversion = [r for r in v.refusals if r.code == "X01_inverted_admission"]
    assert len(inversion) == 1
    assert inversion[0].stage == "I_regional"
    assert inversion[0].evidence["lower_integrity_tier"] == 4
    assert inversion[0].evidence["higher_integrity_tier"] == 1
    assert inversion[0].evidence["first_admitted_order"] == 0

    founding = [r for r in v.refusals if r.code == "X02_founding_stage_not_pure"]
    assert [r.stage for r in founding] == ["I_regional"]

    unfounded = {r.stage: r.evidence["n_unfounded"] for r in v.refusals if r.code == "X03_unfounded_parameter"}
    assert unfounded["I_regional"] == 49
    assert set(unfounded) == {"I_regional", "II_interface", "III_sliced", "IV_assembly"}

    bold = [
        r
        for r in v.refusals
        if r.code == "X04_permission_widens_with_tier" and r.stage == "III_sliced"
    ]
    assert bold and all(n.startswith("bold.") for n in bold[0].evidence["widened"])


def test_shipped_config_admission_is_read_from_the_trainer_not_guessed() -> None:
    """001-beta declares no curriculum, so admission is reconstructed."""
    cur = Curriculum.from_config(BETA)
    assert cur.fully_declared is False
    by_name = {s.name: s for s in cur.stages}
    assert by_name["I_regional"].admits == (4,)
    assert 1 in by_name["III_sliced"].admits
    assert all("reconstructed" in s.provenance for s in cur.stages)


def test_corrected_config_has_no_ordering_refusal() -> None:
    v = validate_config(ORDERED, tiers_path=TIERS)
    offending = [r for r in v.refusals if r.code in ORDERING_CODES]
    assert offending == [], "\n".join(str(r) for r in offending)


def test_corrected_config_still_refuses_the_handover_items() -> None:
    """It is refused, and for reasons that are not ordering errors.

    Both are handover items for other owners: the trainer's hard-coded admission
    gates (X06) and the anatomy adapter that silently substitutes a synthetic
    prior (X09).  Recorded here so that "the corrected config passes" is never
    said without them.
    """
    v = validate_config(ORDERED, tiers_path=TIERS)
    assert not v.ok
    assert set(v.codes()) == {
        "X06_trainer_gate_contradicts_config",
        "X09_declared_provenance_contradicted",
    }


def test_corrected_config_totals_match_the_beta_budget() -> None:
    cur = Curriculum.from_config(ORDERED)
    beta = Curriculum.from_config(BETA)
    assert cur.total_steps() == beta.total_steps() == 8700
    assert [s.steps for s in cur.stages] == [2966, 500, 1000, 3334, 0, 900]


def test_admission_order_is_monotone_in_the_corrected_config() -> None:
    cur = Curriculum.from_config(ORDERED)
    orders = {t: cur.first_admission(t).order for t in cur.admitted_tiers()}
    assert orders == {1: 0, 2: 1, 3: 2, 4: 3}


# ======================================================================
# each refusal, fired on purpose
# ======================================================================
def _synthetic(tmp_path: Path, mutate) -> Path:
    payload = yaml.safe_load(ORDERED.read_text())
    mutate(payload)
    p = tmp_path / "mutated.yaml"
    p.write_text(yaml.safe_dump(payload, sort_keys=False))
    return p


def _run(path: Path):
    cur = Curriculum.from_config(path)
    cards = load_mixture_cards(REPO / "configs/curriculum/source_cards")
    policy = load_tier_policy(TIERS)
    uni, prov = parameter_universe(BETA)
    return validate(cur, cards, policy, universe=uni, universe_provenance=prov)


def test_x01_fires_when_the_order_is_swapped(tmp_path: Path) -> None:
    def mutate(p):
        st = p["train"]["stages"]
        st[0]["extra"]["curriculum"]["admits"] = [4]
        st[0]["extra"]["curriculum"]["tier_permissions"] = {4: ["local.*"]}

    v = _run(_synthetic(tmp_path, mutate))
    assert "X01_inverted_admission" in v.codes()


def test_a_stage_cannot_grant_what_a_card_withheld(tmp_path: Path) -> None:
    """Restrict-only, asserted rather than assumed.

    Handing tier 4 ``bold.*`` in the stage's ``tier_permissions`` must change
    nothing, because ``sim_wholebrain``'s card freezes it.  If this ever starts
    producing an X03 the intersection has become a union.
    """

    def mutate(p):
        p["train"]["stages"][3]["extra"]["curriculum"]["tier_permissions"][4] = [
            "local.*",
            "bold.*",
        ]

    v = _run(_synthetic(tmp_path, mutate))
    assert "X03_unfounded_parameter" not in v.codes()


def test_x03_fires_when_a_card_lets_a_lower_tier_reach_further(tmp_path: Path) -> None:
    """Withdraw the simulator's ``bold.*`` freeze and the refusal appears.

    This is the exact permission ``configs/source_cards/sim_wholebrain.yaml``
    grants today, and the reason it is withdrawn in the corrected card set: no
    measured source in this corpus can found the haemodynamic head.
    """
    src = REPO / "configs/curriculum/source_cards"
    d = tmp_path / "source_cards"
    d.mkdir()
    for f in src.glob("*.yaml"):
        (d / f.name).write_text(f.read_text())
    sim = yaml.safe_load((d / "sim_wholebrain.yaml").read_text())
    sim["frozen"] = [x for x in sim["frozen"] if x != "bold.*"]
    sim["gradient_permission"] = list(sim["gradient_permission"]) + ["bold.*"]
    (d / "sim_wholebrain.yaml").write_text(yaml.safe_dump(sim, sort_keys=False))
    (tmp_path / "card_metadata.yaml").write_text(
        (REPO / "configs/curriculum/card_metadata.yaml").read_text()
    )

    payload = yaml.safe_load(ORDERED.read_text())
    payload["train"]["stages"][3]["extra"]["curriculum"]["tier_permissions"][4].append("bold.*")
    cfg = tmp_path / "mutated.yaml"
    cfg.write_text(yaml.safe_dump(payload, sort_keys=False))

    uni, prov = parameter_universe(BETA)
    v = validate(
        Curriculum.from_config(cfg),
        load_mixture_cards(d),
        load_tier_policy(TIERS),
        universe=uni,
        universe_provenance=prov,
    )
    unfounded = [r for r in v.refusals if r.code == "X03_unfounded_parameter"]
    assert unfounded
    assert all(n.startswith("bold.") for n in unfounded[0].evidence["unfounded"])


def test_x08_fires_on_an_unrecorded_absence(tmp_path: Path) -> None:
    """A tier admitted with nothing behind it must say so."""

    def mutate(p):
        cur = p["train"]["stages"][4]  # the tier-5 stage
        cur["enabled"] = True
        cur["steps"] = 10
        cur["extra"]["curriculum"]["admits"] = [1, 5]
        cur["extra"]["curriculum"]["tier_permissions"] = {1: ["readout.*"], 5: ["readout.*"]}
        cur["extra"]["curriculum"]["absence"] = []

    v = _run(_synthetic(tmp_path, mutate))
    assert "X08_unrecorded_absence" in v.codes()


def test_x08_is_silenced_by_recording_the_absence(tmp_path: Path) -> None:
    """...and reads differently once the record is present."""

    def mutate(p):
        cur = p["train"]["stages"][4]
        cur["enabled"] = True
        cur["steps"] = 10
        cur["extra"]["curriculum"]["admits"] = [1, 5]
        cur["extra"]["curriculum"]["tier_permissions"] = {1: ["readout.*"], 5: ["readout.*"]}
        cur["extra"]["curriculum"]["absence"] = [
            {"tier": 5, "detail": "TRIBE v2 is not on disk", "consequence": "no distillation"}
        ]

    v = _run(_synthetic(tmp_path, mutate))
    assert "X08_unrecorded_absence" not in v.codes()


def test_x05_quantifier_is_all_observed_modalities(tmp_path: Path) -> None:
    """An EEG source keeps ``eeg.*``; a BOLD-only source does not.

    The rule is "every modality this source observes is blind to the parameter",
    not "some modality is".  Getting the quantifier wrong refuses an EEG source
    for BOLD's blindness, which reads plausible and is backwards.
    """
    src = REPO / "configs/curriculum/source_cards"
    d = tmp_path / "source_cards"
    d.mkdir()
    for f in src.glob("*.yaml"):
        (d / f.name).write_text(f.read_text())
    meta = yaml.safe_load((REPO / "configs/curriculum/card_metadata.yaml").read_text())

    # a prospective BOLD-only likelihood source: ds000117 has 663 MB of real BOLD
    # on disk (2 subjects, 18 runs) and no loader. When one lands, this fires.
    (d / "ds000117_bold.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "ds000117_bold",
                "role": "likelihood",
                "is_simulated": False,
                "losses": ["likelihood", "forecast"],
                "gradient_permission": ["bold.*", "coupling.*", "eeg.*"],
                "n_eff": 2.0,
            }
        )
    )
    meta["sources"]["ds000117_bold"] = {"observes": ["bold"]}
    (tmp_path / "card_metadata.yaml").write_text(yaml.safe_dump(meta))

    cards = load_mixture_cards(d)
    uni, prov = parameter_universe(BETA)
    v = validate(
        Curriculum.from_config(ORDERED),
        cards,
        load_tier_policy(TIERS),
        universe=uni,
        universe_provenance=prov,
        blind_rules=_rules(),
    )
    hits = [r for r in v.refusals if r.code == "X05_information_blind_update"]
    offenders = {r.evidence["source"] for r in hits}
    assert offenders == {"ds000117_bold"}
    tensors = {t for r in hits for t in r.evidence["tensors"]}
    assert all(t.startswith("eeg.") for t in tensors)
    # and it is NOT refused coupling, which BOLD does carry information about
    assert not any(t.startswith("coupling.") for t in tensors)


def _rules():
    from scwbd.curriculum.information import derive_blind_rules, load_modality_information

    return derive_blind_rules(
        load_modality_information(
            REPO / "reports/identifiability/results.json",
            REPO / "reports/identifiability/manifest.json",
        )
    )


# ======================================================================
# the glob/name-space trap (decorative_guards defect 1)
# ======================================================================
def test_universe_is_the_model_that_runs(universe) -> None:
    names, prov = universe
    assert len(names) == 152
    assert "posterior.summary.conv.0.weight" in names
    assert "individualizer.mu" in names
    assert "coupling.gain_soft" in names
    assert "_orig_mod" not in " ".join(names)
    assert "_CombinedModule" in prov


def test_expand_resolves_globs_against_real_tensors(universe) -> None:
    names, _ = universe
    assert expand(["bold.*"], names) == {
        "bold.log_kappa",
        "bold.log_gamma",
        "bold.log_tau",
        "bold.alpha",
        "bold.rho",
        "bold.neural_gain",
        "bold.log_noise",
    }
    assert expand(["coupling.*"], names, frozen=["coupling.global_scale"]) == {
        "coupling.gain_soft",
        "coupling.gain_proposed",
    }
    # a glob that names nothing expands to nothing, and is therefore visible
    assert expand(["nonexistent_module.*"], names) == set()
