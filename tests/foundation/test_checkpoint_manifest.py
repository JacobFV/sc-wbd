"""Checkpoint round-trip, R07 centering, and the claim manifest's refusal to over-claim."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scwbd.foundation.checkpoint import CheckpointError, load_checkpoint, save_checkpoint
from scwbd.foundation.config import FoundationConfig, load_config
from scwbd.foundation.individual import Individualizer, R07Violation, recovery_test
from scwbd.foundation.manifest import Claim, ClaimManifest, OverclaimError
from scwbd.foundation.model import SCWBD
from scwbd.foundation.anatomy import load_anatomy
from scwbd.foundation.simulate import ThetaPrior
from scwbd.foundation.util import set_determinism

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def anat():
    return load_anatomy(device="cpu", n_cortex=30, n_subcortex=8, n_cerebellum=6, density=0.2, seed=3)


@pytest.fixture(scope="module")
def cfg():
    return load_config(REPO / "configs" / "scwbd_ci_smoke.yaml")


# ----------------------------------------------------------------------
# checkpoints
# ----------------------------------------------------------------------
def test_checkpoint_round_trip_is_exact(tmp_path, cfg, anat):
    set_determinism(0)
    m = SCWBD(cfg.model, anat)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    y = torch.randn(2, 4, anat.n_regions)
    th = ThetaPrior().sample(2, seed=0)
    m.rollout(y_context=y, theta=th, n_steps=3).activity.sum().backward()
    opt.step()

    p = save_checkpoint(tmp_path / "ck.pt", model=m, config=cfg, step=7, stage="III_sliced", optimizer=opt)
    assert p.exists()
    assert (tmp_path / "config.yaml").exists()
    prov = json.loads((tmp_path / "provenance.json").read_text())
    assert prov["step"] == 7 and prov["stage"] == "III_sliced"
    assert len(prov["weights_sha256"]) == 64
    assert "git_sha" in prov and "environment" in prov

    set_determinism(999)
    m2 = SCWBD(cfg.model, anat)
    assert not torch.equal(m2.local.inp.weight, m.local.inp.weight)
    payload = load_checkpoint(p, model=m2)
    assert payload["step"] == 7
    for (n1, p1), (n2, p2) in zip(m.state_dict().items(), m2.state_dict().items()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"{n1} did not round-trip"

    # the same rollout must reproduce exactly after reload
    set_determinism(5)
    a = m.rollout(y_context=y, theta=th, n_steps=3).activity
    set_determinism(5)
    b = m2.rollout(y_context=y, theta=th, n_steps=3).activity
    assert torch.equal(a, b)


def test_checkpoint_carries_the_state_layout_and_config(tmp_path, cfg, anat):
    m = SCWBD(cfg.model, anat)
    p = save_checkpoint(tmp_path / "c.pt", model=m, config=cfg, step=0, stage="I_regional")
    payload = torch.load(p, map_location="cpu", weights_only=False)
    assert payload["state_layout"]["dim"] == m.layout.dim
    assert payload["config"]["model"]["dt_model"] == cfg.model.dt_model
    assert payload["git_sha"]


def test_loading_a_missing_or_foreign_checkpoint_fails_loudly(tmp_path, cfg, anat):
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / "nope.pt")
    bad = tmp_path / "bad.pt"
    torch.save({"format": "somebody-elses/1", "model": {}}, bad)
    with pytest.raises(CheckpointError, match="unrecognised"):
        load_checkpoint(bad)


# ----------------------------------------------------------------------
# R07
# ----------------------------------------------------------------------
def test_population_effects_are_centered_by_construction():
    ind = Individualizer(4, n_groups=3, n_participants=10, n_sessions=20, group_counts=[10.0, 3.0, 1.0])
    with torch.no_grad():
        ind._alpha_raw.normal_(0, 5.0)  # adversarial: raw effects far from centered
    resid = (ind.group_counts.reshape(-1, 1) * ind.alpha).sum(0).abs().max().detach()
    assert float(resid) < 1e-5
    ind.assert_centered()


def test_r07_is_raised_when_shrinkage_is_undefined():
    ind = Individualizer(3, n_groups=2, n_participants=4, n_sessions=8)
    with torch.no_grad():
        ind.log_sd_person.fill_(float("nan"))
    with pytest.raises(R07Violation):
        ind.assert_centered()


def test_hierarchical_decomposition_is_identified():
    """Writing down mu+alpha+delta+zeta is not the same as identifying it."""
    r = recovery_test(steps=800, seed=1)
    assert r["centering_residual"] < 1e-4
    assert r["alpha_corr"] > 0.8, r
    assert r["delta_corr"] > 0.8, r
    assert r["zeta_corr"] > 0.4, r
    assert r["identified"] is True


def test_session_state_is_not_consolidated_into_a_trait_too_eagerly():
    ind = Individualizer(3, n_groups=1, n_participants=4, n_sessions=8)
    out = ind.consolidate(0, min_sessions=3)
    assert out["action"] == "defer", "one session must never become a permanent trait"
    ind.observe_session(torch.tensor([0, 0, 0, 0]))
    out = ind.consolidate(0, min_sessions=3, max_uncertainty=1.0)
    assert out["action"] == "consolidate"


# ----------------------------------------------------------------------
# claim manifest
# ----------------------------------------------------------------------
def _min_manifest() -> ClaimManifest:
    return ClaimManifest(cannot_do=("cannot predict any individual's future brain state",))


def test_manifest_refuses_digital_twin_language():
    m = _min_manifest()
    with pytest.raises(OverclaimError, match="digital twin"):
        m.add_claim(
            Claim(
                id="c1",
                statement="SC-WBD-001-beta is a digital twin of the adult brain.",
                status="supported",
                evidence_status="measured",
                falsifier="anything",
            )
        )


@pytest.mark.parametrize("text", ["clinically validated for treatment", "supports diagnosis of epilepsy", "estimates Phi value"])
def test_manifest_refuses_medical_and_consciousness_claims(text):
    m = _min_manifest()
    with pytest.raises(OverclaimError):
        m.add_claim(Claim(id="x", statement=text, status="partial", evidence_status="measured", falsifier="f"))


def test_manifest_requires_a_falsifier():
    m = _min_manifest()
    with pytest.raises(OverclaimError, match="falsifier"):
        m.add_claim(Claim(id="c", statement="the model forecasts EEG", status="partial", evidence_status="measured"))


def test_simulator_evidence_cannot_support_a_claim():
    m = _min_manifest()
    with pytest.raises(OverclaimError, match="simulator-conditioned"):
        m.add_claim(
            Claim(
                id="sim",
                statement="the amortized posterior recovers global coupling",
                status="supported",
                evidence_status="simulator_conditioned",
                falsifier="SBC ranks depart from uniformity",
            )
        )
    # the same claim with an honest status is allowed
    m.add_claim(
        Claim(
            id="sim",
            statement="the amortized posterior recovers global coupling under the simulator",
            status="partial",
            evidence_status="simulator_conditioned",
            falsifier="SBC ranks depart from uniformity",
            caveats=("simulator-conditioned only; no biological validity implied",),
        )
    )
    assert len(m.claims) == 1


def test_manifest_requires_an_explicit_cannot_do():
    m = ClaimManifest()
    with pytest.raises(OverclaimError, match="cannot_do"):
        m.validate()


def test_manifest_round_trips(tmp_path):
    m = _min_manifest()
    m.add_claim(
        Claim(
            id="c",
            statement="held-out sensor-space EEG forecast likelihood is reported with intervals",
            status="partial",
            evidence_status="measured",
            falsifier="a matched-capacity baseline beats it with non-overlapping intervals",
        )
    )
    m.add_negative("g5_individualization", "individual effects did not improve held-out score", {"delta": 0.01})
    p = m.save(tmp_path / "cm.json")
    again = ClaimManifest.load(p)
    assert again.claims[0].id == "c"
    assert again.negative_results[0]["name"] == "g5_individualization"
    assert "cannot do" in again.to_markdown().lower()
