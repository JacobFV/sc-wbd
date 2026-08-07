"""The measured-information rules, and the readings that would not produce them."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from scwbd.curriculum.information import (
    NEGLIGIBLE_RATIO,
    derive_blind_rules,
    load_modality_information,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "reports/identifiability/results.json"
MANIFEST = REPO / "reports/identifiability/manifest.json"


def _info():
    return load_modality_information(RESULTS, MANIFEST)


def test_provenance_says_no_real_datasets() -> None:
    """These numbers are not measurements of real fMRI, and the record says so.

    The brief that commissioned this design described the figures as "real fMRI".
    The manifest lists ``no real datasets`` as a non-goal, so any permission
    derived from them is a refusal grounded in a simulated linear geometry --
    conservative in that direction and not in the other.
    """
    prov = _info().provenance
    assert "no real datasets" in prov["non_goals"]
    assert prov["monte_carlo_disabled"] is True
    assert prov["n_regimes"] == 3


def test_eeg_is_structurally_blind_to_the_haemodynamic_parameters() -> None:
    info = _info()
    for regime in info.regimes:
        for p in ("beta_hrf", "c_under", "gain_bold"):
            assert info.diagonal[regime]["eeg"][p] == 0.0


def test_the_bar_discriminates_delay_from_coupling() -> None:
    """The same call marks ``tau`` negligible and leaves coupling alone.

    A bar that marked both would be measuring its own definition.  Measured
    BOLD/EEG ratios: tau 4.0e-07 … 1.0e-06; coupling 1.3e-03 … 9.5e-03.
    """
    rules = {(r.modality, r.param): r for r in derive_blind_rules(_info())}
    assert ("bold", "tau") in rules
    assert rules[("bold", "tau")].kind == "negligible"
    for p in ("a21", "a32", "a13"):
        assert ("bold", p) not in rules

    info = _info()
    for regime in info.regimes:
        assert info.ratio("bold", "tau", regime) < NEGLIGIBLE_RATIO
        for p in ("a21", "a32", "a13"):
            assert info.ratio("bold", p, regime) > NEGLIGIBLE_RATIO


def test_the_delay_rule_binds_nothing_and_says_so() -> None:
    """The sharpest information result constrains no gradient in this model.

    Conduction delay is a buffer cut from tract length, not a parameter.  A rule
    that quietly disappeared here would leave the impression that the delay
    result was doing work in the curriculum.  It is not, and ``binds`` records
    that rather than the rule being dropped.
    """
    rules = {(r.modality, r.param): r for r in derive_blind_rules(_info())}
    tau = rules[("bold", "tau")]
    assert tau.globs == () and tau.binds is False
    assert "not trainable in this model" in tau.note


def test_a_world_where_bold_sees_delay_produces_no_rule(tmp_path: Path) -> None:
    """The guard reads differently when the hypothesis is false.

    Raise BOLD's information about ``tau`` to EEG's level in every regime and the
    negligible rule must disappear -- otherwise it is a tripwire, not a test.
    """
    payload = json.loads(RESULTS.read_text())
    params = [p["name"] for p in json.loads(MANIFEST.read_text())["parameters"]]
    i_tau = params.index("tau")
    doctored = copy.deepcopy(payload)
    for rd in doctored["results"]["regimes"].values():
        blocks = rd["designs"]["joint_native"]["fisher_T4"]["I_by_modality"]
        blocks["bold"][i_tau][i_tau] = blocks["eeg"][i_tau][i_tau]
    p = tmp_path / "results.json"
    p.write_text(json.dumps(doctored))

    rules = {(r.modality, r.param) for r in derive_blind_rules(load_modality_information(p, MANIFEST))}
    assert ("bold", "tau") not in rules
    # the structural zeros are untouched, so the doctoring was targeted
    assert ("eeg", "gain_bold") in rules


def test_one_informative_regime_defeats_the_refusal(tmp_path: Path) -> None:
    """Blindness must hold in EVERY regime, not on average."""
    payload = json.loads(RESULTS.read_text())
    params = [p["name"] for p in json.loads(MANIFEST.read_text())["parameters"]]
    i = params.index("gain_bold")
    doctored = copy.deepcopy(payload)
    one = next(iter(doctored["results"]["regimes"]))
    blocks = doctored["results"]["regimes"][one]["designs"]["joint_native"]["fisher_T4"][
        "I_by_modality"
    ]
    blocks["eeg"][i][i] = blocks["bold"][i][i]
    p = tmp_path / "results.json"
    p.write_text(json.dumps(doctored))

    rules = {(r.modality, r.param) for r in derive_blind_rules(load_modality_information(p, MANIFEST))}
    assert ("eeg", "gain_bold") not in rules
    assert ("eeg", "beta_hrf") in rules


def test_misaligned_fisher_block_is_refused(tmp_path: Path) -> None:
    """Positional alignment between the block and the manifest is checked, not assumed."""
    payload = json.loads(RESULTS.read_text())
    for rd in payload["results"]["regimes"].values():
        b = rd["designs"]["joint_native"]["fisher_T4"]["I_by_modality"]
        b["eeg"] = [row[:-1] for row in b["eeg"][:-1]]
    p = tmp_path / "results.json"
    p.write_text(json.dumps(payload))
    try:
        load_modality_information(p, MANIFEST)
    except ValueError as exc:
        assert "refusing to align them by position" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a misaligned Fisher block was accepted")
