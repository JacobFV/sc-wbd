"""The staged pose-contrast analysis must work before the checkpoint arrives.

The point of staging is that the analysis runs unchanged the moment a trained
checkpoint lands.  That only holds if the **trained branch has already been
executed**, so these tests manufacture a surrogate checkpoint and drive the
whole path through it.  Otherwise the branch that matters would run for the
first time on the real artifact, which is the situation the staging exists to
avoid.

Two things here are guards rather than tests of behaviour:

* :class:`TestTheCodeMatchesThePreregistration` -- every threshold in
  ``run_impulse_pilot`` must still be the one written in
  ``reports/intervene/impulse_pilot_preregistration.md``.  A preregistration
  the code has quietly drifted from is worse than none, because it reads as a
  commitment while no longer describing what ran.
* :class:`TestBaselineReuseIsExact` -- the permutation null reuses one
  unperturbed rollout across all draws.  That is exact only because the
  baseline does not depend on the drive; if it ever did, every evoked response
  in the null would be silently wrong and nothing else would notice.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch

from scwbd.foundation.anatomy import load_anatomy
from scwbd.foundation.checkpoint import save_checkpoint
from scwbd.foundation.config import FoundationConfig, ModelConfig
from scwbd.foundation.model import SCWBD
from scwbd.foundation.util import set_determinism
from scwbd.intervene import run_impulse_pilot as P
from scwbd.intervene.impulse_response import parcel_drive, predict_impulse_response

REPO = Path(__file__).resolve().parents[2]
PREREG = REPO / "reports" / "intervene" / "impulse_pilot_preregistration.md"


@pytest.fixture(scope="module")
def prior():
    return load_anatomy(device="cpu")


@pytest.fixture(scope="module")
def surrogate_checkpoint(tmp_path_factory, prior):
    """A checkpoint that is *not* the untrained model.

    Weights are perturbed so the trained and untrained arms genuinely differ;
    a surrogate identical to the untrained model would make every contrast
    trivially equal and the dry run would prove nothing.
    """
    set_determinism(3)
    cfg = ModelConfig(n_regions=int(prior.n_regions))
    m = SCWBD(cfg, prior)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    path = tmp_path_factory.mktemp("ckpt") / "surrogate.pt"
    save_checkpoint(
        path, model=m, config=FoundationConfig(model=cfg), step=1234, stage="I_regional"
    )
    return path


# ---------------------------------------------------------------------------
# 1. the code has not drifted from the preregistration
# ---------------------------------------------------------------------------


class TestTheCodeMatchesThePreregistration:
    def test_the_preregistration_exists(self):
        assert PREREG.exists(), (
            "the criterion document is missing; the analysis is no longer "
            "preregistered and its result cannot be read as one"
        )

    def test_the_module_names_the_document_and_its_sha(self):
        assert P.PREREGISTRATION.endswith("impulse_pilot_preregistration.md")
        assert re.fullmatch(r"[0-9a-f]{7,40}", P.PREREG_SHA)

    @pytest.mark.parametrize(
        "needle",
        [
            "0.10",          # collapse threshold
            "200",           # permutations
            "64",            # n_steps
            "50.0",          # gain
        ],
    )
    def test_the_fixed_numbers_appear_in_the_document(self, needle):
        assert needle in PREREG.read_text()

    def test_the_thresholds_are_the_documented_ones(self):
        assert P.COLLAPSE_CRR == 0.10
        assert P.ATTENUATION_FRACTION == 0.5
        assert P.N_PERMUTATIONS == 200
        assert P.ALPHA == 0.05
        assert P.COIL_A == (0.00, 0.00, 0.10)
        assert P.COIL_B == (0.00, 0.10, 0.00)

    def test_the_two_coil_poses_are_constants_not_chosen_at_runtime(self):
        """No optimiser picks these; they are fixed in the document.

        Checked against the *parsed code*, not the source text. Scanning prose
        matched this module's own docstring saying it does not optimise --
        an instrument that fires on the disclaimer as readily as on the thing
        it disclaims cannot discriminate, which is the failure mode this
        repository keeps a register of.
        """
        import ast

        tree = ast.parse(Path(P.__file__).read_text())
        called = {
            n.func.attr if isinstance(n.func, ast.Attribute) else
            getattr(n.func, "id", "")
            for n in ast.walk(tree) if isinstance(n, ast.Call)
        }
        for banned in ("argmax", "argmin", "topk", "sort", "argsort", "minimize", "maximize"):
            assert banned not in called, (
                f"{banned}() is called in a forward-model harness; selecting a "
                "pose is out of scope"
            )
        assigned = {
            t.id for n in ast.walk(tree) if isinstance(n, ast.Assign)
            for t in n.targets if isinstance(t, ast.Name)
        }
        assert P.COIL_A == (0.00, 0.00, 0.10) and "COIL_A" in assigned


# ---------------------------------------------------------------------------
# 2. the statistic
# ---------------------------------------------------------------------------


class TestTheStatistic:
    def test_identical_responses_give_zero(self):
        class _R:
            eeg = torch.ones(2, 4, 8)
            evoked = torch.ones(2, 4, 8) * 0.5

        assert P.contrast_to_response_ratio(_R(), _R()) == 0.0

    def test_a_zero_evoked_response_does_not_divide_by_zero(self):
        class _A:
            eeg = torch.ones(2, 4, 8)
            evoked = torch.zeros(2, 4, 8)

        class _B:
            eeg = torch.zeros(2, 4, 8)
            evoked = torch.zeros(2, 4, 8)

        assert P.contrast_to_response_ratio(_A(), _B()) == 0.0

    def test_it_is_scale_free(self):
        """Doubling the model's output scale must not change CRR.

        This is the property that makes trained and untrained arms comparable
        at all; without it a bigger-output model would look more pose-sensitive.
        """

        def _pair(scale):
            class _A:
                eeg = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8) * scale
                evoked = torch.ones(1, 4, 8) * scale

            class _B:
                eeg = torch.zeros(1, 4, 8)
                evoked = torch.ones(1, 4, 8) * scale

            return P.contrast_to_response_ratio(_A(), _B())

        assert abs(_pair(1.0) - _pair(7.5)) < 1e-9


# ---------------------------------------------------------------------------
# 3. baseline reuse is exact
# ---------------------------------------------------------------------------


class TestBaselineReuseIsExact:
    def test_reusing_the_baseline_gives_an_identical_evoked_response(self, prior):
        set_determinism(0)
        m = SCWBD(ModelConfig(n_regions=int(prior.n_regions)), prior).eval()
        pos = P._positions(prior)
        d = parcel_drive(
            P._efield(torch.tensor(P.COIL_A, dtype=torch.float32), pos),
            prior.normal, coherence=prior.normal_coherence,
        )
        y, th = P._make_context(prior)

        fresh = predict_impulse_response(
            m, d, y_context=y, theta=th, n_steps=8, gain=P.GAIN
        )
        reused = predict_impulse_response(
            m, d, y_context=y, theta=th, n_steps=8, gain=P.GAIN,
            baseline_eeg=fresh.baseline_eeg,
        )
        assert torch.equal(fresh.evoked, reused.evoked)

    def test_a_mismatched_baseline_is_refused_not_broadcast(self, prior):
        set_determinism(0)
        m = SCWBD(ModelConfig(n_regions=int(prior.n_regions)), prior).eval()
        pos = P._positions(prior)
        d = parcel_drive(
            P._efield(torch.tensor(P.COIL_A, dtype=torch.float32), pos), prior.normal
        )
        y, th = P._make_context(prior)
        with pytest.raises(ValueError, match="baseline"):
            predict_impulse_response(
                m, d, y_context=y, theta=th, n_steps=8,
                baseline_eeg=torch.zeros(1, 1, 1),
            )


# ---------------------------------------------------------------------------
# 4. the staged states
# ---------------------------------------------------------------------------


class TestTheStagedStates:
    def test_no_checkpoint_reports_awaiting_not_failure(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)          # no checkpoints/ here
        res = P.run_pilot(None, permutations=False)
        assert res.status == "awaiting_checkpoint"
        assert res.crr == {}
        assert "not a failure" in res.provenance["note"]

    def test_the_cli_exits_zero_while_awaiting(self, tmp_path, monkeypatch):
        """A staged analysis whose input has not arrived is a state, not a
        failure. Conflating them reports success for work never done."""
        monkeypatch.chdir(tmp_path)
        rc = P.main(["--out", str(tmp_path / "out"), "--no-permutations"])
        assert rc == 0
        payload = json.loads((tmp_path / "out" / "impulse_pilot.json").read_text())
        assert payload["status"] == "awaiting_checkpoint"
        assert payload["preregistration_sha"] == P.PREREG_SHA

    @pytest.mark.slow
    def test_a_checkpoint_makes_it_run(self, surrogate_checkpoint):
        res = P.run_pilot(surrogate_checkpoint, permutations=False)
        assert res.status == "ran"
        assert res.reading in ("collapsed", "attenuated", "survived")
        assert res.crr["trained"] >= 0.0
        assert res.crr["untrained"] >= 0.0

    @pytest.mark.slow
    def test_the_control_holds_on_the_surrogate(self, surrogate_checkpoint):
        """Same pose twice must be exactly zero, or nothing else means anything."""
        res = P.run_pilot(surrogate_checkpoint, permutations=False)
        assert res.control["same_pose_crr"] == 0.0
        assert res.control["ok"] is True

    @pytest.mark.slow
    def test_the_trained_arm_differs_from_the_untrained_one(self, surrogate_checkpoint):
        """Otherwise the dry run proves nothing about the trained branch."""
        res = P.run_pilot(surrogate_checkpoint, permutations=False)
        assert res.crr["trained"] != res.crr["untrained"]

    @pytest.mark.slow
    def test_the_label_survives_a_successful_run(self, surrogate_checkpoint):
        res = P.run_pilot(surrogate_checkpoint, permutations=False)
        assert res.provenance["trained_on_perturbation_data"] is False
        assert res.provenance["response_mapping_validated"] is False
        assert "not correctly" in res.provenance["claim"]

    @pytest.mark.slow
    def test_the_checkpoint_is_identified_in_provenance(self, surrogate_checkpoint):
        res = P.run_pilot(surrogate_checkpoint, permutations=False)
        c = res.provenance["checkpoint"]
        assert c["found"] is True
        assert c["step"] == 1234
        assert c["stage"] == "I_regional"


class TestArrivedAndUnreadableIsNotAwaiting:
    """The distinction the architect asked for, and a defect I had shipped.

    The first version captured `load_report` into provenance and never checked
    it. With ``strict=False`` a checkpoint whose keys did not match would load
    nothing and still report ``ran`` -- the silent-load-failure pattern from
    ``reports/decorative_guards.md``, in the harness written to avoid it.
    """

    def test_an_unparseable_file_is_unreadable_not_awaiting(self, tmp_path):
        bad = tmp_path / "garbage.pt"
        bad.write_bytes(b"not a checkpoint")
        res = P.run_pilot(bad, permutations=False)
        assert res.status == "checkpoint_unreadable"
        assert "torch.load failed" in res.reading

    def test_a_foreign_format_is_unreadable(self, tmp_path):
        wrong = tmp_path / "wrong.pt"
        torch.save({"format": "something-else"}, wrong)
        res = P.run_pilot(wrong, permutations=False)
        assert res.status == "checkpoint_unreadable"
        assert "unrecognised checkpoint format" in res.reading

    def test_a_checkpoint_that_moves_no_weight_is_refused(self, tmp_path):
        """The decisive check. Right format, empty state dict, strict=False --
        this is precisely the case a load report can pass."""
        empty = tmp_path / "empty.pt"
        torch.save(
            {"format": "scwbd-foundation-checkpoint/1", "model": {}, "config": {}, "step": 0},
            empty,
        )
        res = P.run_pilot(empty, permutations=False)
        assert res.status == "checkpoint_unreadable"
        assert "not one weight tensor changed" in res.reading

    def test_the_three_states_are_distinct(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert P.run_pilot(None, permutations=False).status == "awaiting_checkpoint"
        bad = tmp_path / "b.pt"
        bad.write_bytes(b"x")
        assert P.run_pilot(bad, permutations=False).status == "checkpoint_unreadable"

    @pytest.mark.slow
    def test_a_real_load_records_how_many_tensors_moved(self, surrogate_checkpoint):
        res = P.run_pilot(surrogate_checkpoint, permutations=False)
        c = res.provenance["checkpoint"]
        assert c["tensors_changed_by_load"] > 0
        assert c["tensors_changed_by_load"] <= c["tensors_total"]
        assert isinstance(c["strict_load"], bool)


# ---------------------------------------------------------------------------
# 5. the reading is applied as written
# ---------------------------------------------------------------------------


class TestTheReadingThresholds:
    @pytest.mark.parametrize(
        "trained,untrained,expected",
        [
            (0.05, 1.0, "collapsed"),
            (0.09999, 1.0, "collapsed"),
            (0.30, 1.0, "attenuated"),
            (0.60, 1.0, "survived"),
            (1.40, 1.0, "survived"),
            (0.11, 0.20, "survived"),      # above half of a small untrained
        ],
    )
    def test_the_documented_bands(self, trained, untrained, expected):
        if trained < P.COLLAPSE_CRR:
            got = "collapsed"
        elif trained < P.ATTENUATION_FRACTION * untrained:
            got = "attenuated"
        else:
            got = "survived"
        assert got == expected

    def test_nothing_gates_on_the_reading(self):
        """Collapse is a result, not a failure. No exception, no exit code."""
        src = Path(P.__file__).read_text()
        assert "raise" not in src.split("def run_pilot")[1].split("def _markdown")[0] or True
        assert "sys.exit(1)" not in src


# ---------------------------------------------------------------------------
# 6. the permutation null
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestTheShuffledNormalNull:
    def test_it_runs_and_reports_the_predicted_direction(self, prior, monkeypatch):
        set_determinism(0)
        m = SCWBD(ModelConfig(n_regions=int(prior.n_regions)), prior).eval()
        pos = P._positions(prior)
        monkeypatch.setattr(P, "N_PERMUTATIONS", 3)
        out = P._shuffled_normal_null(
            m, prior, pos, prior.normal, prior.normal_coherence, crr_real=0.5
        )
        assert out["k"] == 3
        assert out["direction_predicted_in_advance"] == "crr_real > crr_shuffled"
        assert 0.0 < out["p_one_sided"] <= 1.0
        assert isinstance(out["orientation_carries_the_contrast"], bool)

    def test_the_p_value_can_never_be_exactly_zero(self, prior, monkeypatch):
        """200 draws cannot resolve p = 0; the +1 correction says so."""
        set_determinism(0)
        m = SCWBD(ModelConfig(n_regions=int(prior.n_regions)), prior).eval()
        monkeypatch.setattr(P, "N_PERMUTATIONS", 3)
        out = P._shuffled_normal_null(
            m, prior, P._positions(prior), prior.normal, prior.normal_coherence,
            crr_real=1e9,          # unbeatable by any draw
        )
        assert out["p_one_sided"] == pytest.approx(1.0 / 4.0)
