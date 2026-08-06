"""The test that was failing implicitly the whole time.

> **Load two different checkpoints and assert the outputs differ.**

Before 2026-08-06 this could not have passed, because ``scwbd.runtime``
contained no ``torch.load``: three loads against three different backings
produced byte-identical numbers (``reports/runtime/consumer_contract.md`` F3).
A runtime that is green because it never loaded a model is the same defect as
a guard that is green because its input never arrives, one level up.

Everything here runs against **real artifacts on disk** and skips honestly when
they are absent. A synthetic stand-in would defeat the purpose: the whole point
is that the numbers come from weights someone trained.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scwbd.runtime._compat import Unresolved
from scwbd.runtime.predict import (
    CheckpointLoadError,
    CheckpointLoadReport,
    LoadedModel,
    rebuild_anatomy,
)
from scwbd.runtime.ports import RawStateAccessRefused

CKPT_ROOT = Path("/home/brandonin/Documents/scwbd-wt/turing/checkpoints")
BETA = CKPT_ROOT / "scwbd-001-beta"
CONTROL = CKPT_ROOT / "scwbd-001-beta-g5control"

#: Distinct training stages of the run-1 artifact -- genuinely different weights
#: produced by the same pipeline, which is exactly the comparison that matters.
STAGES = ("stage_I_regional.pt", "stage_II_interface.pt", "stage_V_individual.pt")

requires_checkpoints = pytest.mark.skipif(
    not (BETA / "last.pt").is_file(),
    reason="the real run-1 checkpoints are not on this machine",
)


@pytest.fixture(scope="module")
def context() -> torch.Tensor:
    """One fixed context window, shared by every model under comparison.

    Fixed and shared on purpose: if the input varied, differing outputs would
    prove nothing about the weights.
    """
    g = torch.Generator().manual_seed(20260806)
    return torch.randn(1, 24, 454, generator=g) * 0.1


@pytest.fixture(scope="module")
def loaded():
    return LoadedModel.from_checkpoint(BETA / "last.pt")


# ---------------------------------------------------------------------------
# the headline
# ---------------------------------------------------------------------------

@requires_checkpoints
class TestDifferentCheckpointsProduceDifferentNumbers:
    """The sharpest possible statement of the problem this closes."""

    def test_three_training_stages_give_three_different_predictions(self, context):
        means, eegs = [], []
        for name in STAGES:
            model = LoadedModel.from_checkpoint(BETA / name)
            pred = model.predict(context, n_steps=30)
            means.append(float(pred.activity.mean()))
            eegs.append(float(pred.eeg.abs().mean()))

        assert len(set(round(m, 9) for m in means)) == len(STAGES), (
            f"activity means collapsed across stages: {means} -- the weights "
            "are not reaching the prediction"
        )
        assert len(set(round(e, 12) for e in eegs)) == len(STAGES), (
            f"EEG predictions collapsed across stages: {eegs}"
        )

    def test_the_control_arm_differs_from_the_treatment_checkpoint(self, context):
        if not (CONTROL / "last.pt").is_file():
            pytest.skip("the g5 control checkpoint is not on this machine")
        a = LoadedModel.from_checkpoint(BETA / "last.pt").predict(context, n_steps=30)
        b = LoadedModel.from_checkpoint(CONTROL / "last.pt").predict(context, n_steps=30)
        assert float(a.eeg.abs().mean()) != float(b.eeg.abs().mean())

    def test_the_same_checkpoint_twice_gives_identical_numbers(self, context):
        """Paired with the above: the difference is the weights, not noise."""
        a = LoadedModel.from_checkpoint(BETA / "last.pt").predict(context, n_steps=30)
        b = LoadedModel.from_checkpoint(BETA / "last.pt").predict(context, n_steps=30)
        assert torch.equal(a.activity, b.activity)
        assert torch.equal(a.eeg, b.eeg)

    def test_a_different_context_moves_the_prediction(self, context, loaded):
        """And the inputs reach it too, not only the weights."""
        other = context * 3.0
        a = loaded.predict(context, n_steps=30)
        b = loaded.predict(other, n_steps=30)
        assert not torch.allclose(a.activity, b.activity)

    def test_theta_conditioning_moves_the_prediction(self, context, loaded):
        base = loaded.default_theta()
        a = loaded.predict(context, theta=base, n_steps=30)
        b = loaded.predict(context, theta=base + 0.5, n_steps=30)
        assert not torch.allclose(a.activity, b.activity)


# ---------------------------------------------------------------------------
# the load, honestly reported
# ---------------------------------------------------------------------------

@requires_checkpoints
class TestTheLoadIsReportedNotAssumed:
    def test_the_torch_compile_prefix_is_stripped_and_counted(self, loaded):
        """29 tensors that `strict=False` would have silently left random."""
        r = loaded.load_report
        assert len(r.renamed) == 29
        assert all("._orig_mod." in k for k in r.renamed)
        assert r.ignored == (), f"unmatched checkpoint keys: {r.ignored}"

    def test_config_fields_absent_from_the_checkpoint_are_inferred_not_defaulted(
        self, loaded
    ):
        """`state_dependent_variance` defaults True today; run 1 predates it."""
        inferred = loaded.load_report.inferred_config
        assert inferred["state_dependent_variance"] is False
        assert inferred["family_state"] is False

    def test_taking_todays_default_instead_would_leave_parameters_random(self):
        """The counterfactual, run, so the inference above is not on trust."""
        import torch as _t

        from scwbd.foundation.config import ModelConfig
        from scwbd.foundation.model import SCWBD

        payload = _t.load(str(BETA / "last.pt"), map_location="cpu",
                          weights_only=False)
        cfg = ModelConfig(**payload["config"]["model"])  # no inference
        assert cfg.state_dependent_variance is True      # today's default
        anat = rebuild_anatomy(payload["extra"]["anatomy"])
        model = SCWBD(cfg, anat)
        state = {k.replace("._orig_mod.", "."): v for k, v in payload["model"].items()}
        missing, _ = model.load_state_dict(state, strict=False)
        assert any(k.startswith("observation.") for k in missing)
        assert any(k.startswith("uncertainty_propagator.") for k in missing)

    def test_inert_parameters_are_named_rather_than_hidden_or_alarmed_about(
        self, loaded
    ):
        r = loaded.load_report
        assert set(r.inert) == {"bold.logvar_gain", "eeg.logvar_mix"}
        assert set(r.uninitialised) == set(r.inert)
        assert r.consequential == ()
        assert r.clean is True

    def test_the_report_survives_into_the_prediction(self, loaded, context):
        pred = loaded.predict(context, n_steps=30)
        assert pred.load_report is loaded.load_report
        assert pred.load_report.as_dict()["restored"] == 85

    def test_a_checkpoint_that_is_not_one_is_refused(self, tmp_path):
        p = tmp_path / "nope.pt"
        torch.save({"format": "something-else"}, p)
        with pytest.raises(CheckpointLoadError) as exc:
            LoadedModel.from_checkpoint(p)
        assert "unrecognised checkpoint format" in str(exc.value)

    def test_a_missing_file_is_refused(self, tmp_path):
        with pytest.raises(CheckpointLoadError):
            LoadedModel.from_checkpoint(tmp_path / "absent.pt")


class TestUnresolvedRatherThanRandom:
    """An output whose parameters are random is not a number."""

    def _report(self, **kw) -> CheckpointLoadReport:
        base = dict(restored=10, uninitialised=("observation.head.w",))
        base.update(kw)
        return CheckpointLoadReport(**base)

    def test_an_uninitialised_variance_head_unresolves_the_variance(self):
        r = self._report()
        assert r.consequential == ("observation.head.w",)
        assert "activity_logvar" in r.unresolved_outputs()

    def test_it_does_not_unresolve_the_mean(self):
        """The heads parameterise mean and variance separately; so do we."""
        assert "eeg" not in self._report().unresolved_outputs()

    def test_marking_it_inert_clears_it(self):
        r = self._report(inert=("observation.head.w",))
        assert r.consequential == ()
        assert r.unresolved_outputs() == ()
        assert r.clean is True

    def test_an_uninitialised_eeg_projection_unresolves_the_eeg_mean(self):
        r = self._report(uninitialised=("eeg.source_proj.0.weight",))
        assert "eeg" in r.unresolved_outputs()

    def test_unmatched_checkpoint_keys_make_the_report_dirty(self):
        assert self._report(uninitialised=(), ignored=("stray.weight",)).clean is False


@requires_checkpoints
class TestPredictionsAreTypedAndPorted:
    def test_state_is_reachable_only_through_declared_ports(self, loaded, context):
        pred = loaded.predict(context, n_steps=30)
        with pytest.raises(RawStateAccessRefused):
            pred.state[..., 0]
        rate_e = pred.state.read("all_regions", "rate_e")
        assert rate_e.units == "Hz"
        assert rate_e.values.shape == (1, 30, 454, 1)

    def test_the_contract_comes_from_the_live_model_not_a_sidecar(self, loaded):
        """So it cannot disagree with what predict() actually returns."""
        contract = loaded.port_contract()
        assert contract.is_uniform
        assert contract.width_of("all_regions") == 28
        assert "loaded_model:" in contract.source

    def test_every_output_is_a_tensor_or_unresolved_never_a_zero(self, loaded, context):
        pred = loaded.predict(context, n_steps=30)
        for name in ("activity", "activity_logvar", "eeg", "eeg_logvar",
                     "hemodynamic"):
            value = getattr(pred, name)
            assert isinstance(value, (torch.Tensor, Unresolved)), name

    def test_the_run_one_artifact_resolves_every_output(self, loaded, context):
        pred = loaded.predict(context, n_steps=30)
        assert pred.unresolved() == {}
        assert set(pred.resolved()) == {
            "activity", "activity_logvar", "eeg", "eeg_logvar", "hemodynamic"
        }

    def test_a_short_rollout_unresolves_the_haemodynamics_and_says_why(
        self, loaded, context
    ):
        """The slow clock does not tick in 16 fast steps; that is not a zero."""
        pred = loaded.predict(context, n_steps=16)
        assert isinstance(pred.hemodynamic, Unresolved)
        assert "hemo_ratio=25" in pred.hemodynamic.reason
        # ...and the fast-clock outputs are unaffected
        assert isinstance(pred.activity, torch.Tensor)

    def test_predictions_carry_no_autograd_graph(self, loaded, context):
        pred = loaded.predict(context, n_steps=30)
        assert not pred.activity.requires_grad
        assert not pred.eeg.requires_grad

    def test_a_context_with_the_wrong_region_count_is_refused(self, loaded):
        with pytest.raises(ValueError) as exc:
            loaded.predict(torch.zeros(1, 24, 200), n_steps=8)
        assert "a different brain" in str(exc.value)

    def test_a_context_of_the_wrong_rank_is_refused(self, loaded):
        with pytest.raises(ValueError) as exc:
            loaded.predict(torch.zeros(24, 454), n_steps=8)
        assert "(B, T, N)" in str(exc.value)


@requires_checkpoints
class TestTheAnatomyMustBeTheOneTheCheckpointWasTrainedOn:
    def test_the_prior_in_this_checkout_no_longer_matches_by_default(self):
        """Recorded because it is the trap: today's prior is a different size."""
        from scwbd.foundation.anatomy import load_anatomy

        live = load_anatomy(device="cpu", n_cortex=400)
        assert live.n_regions != 454, (
            "if these ever match again, this test has stopped proving anything"
        )

    def test_rebuild_reproduces_the_recorded_prior(self, loaded):
        rec = loaded.load_report.anatomy
        assert rec["n_regions"] == 454
        assert rec["provenance"] == "synthetic_fallback"
        assert rec["is_biological"] is False

    def test_a_prior_that_cannot_be_reproduced_is_refused(self):
        with pytest.raises(CheckpointLoadError) as exc:
            rebuild_anatomy({"n_regions": 999, "provenance": "synthetic_fallback"})
        assert "anatomy has moved underneath the artifact" in str(exc.value)


@requires_checkpoints
class TestTheServingSurfaceExposesTheModel:
    """`ServedModel` used to hash a checkpoint and never open it."""

    @pytest.fixture
    def served(self, tmp_path):
        import json

        from scwbd.runtime.admission import sidecar_from_checkpoint
        from scwbd.runtime.serving import ServedModel

        d = tmp_path / "scwbd-001-beta"
        d.mkdir(parents=True)
        (d / "last.pt").symlink_to(BETA / "last.pt")
        (d / "claim_manifest.json").write_text(
            json.dumps(
                sidecar_from_checkpoint(BETA / "last.pt", trust_checkpoint_pickle=True)
            )
        )
        return ServedModel.load(
            "scwbd-001-beta", device="cpu", checkpoint_root=tmp_path,
            purpose="research_offline",
        )

    def test_the_served_model_can_produce_the_loaded_model(self, served, context):
        model = served.predictor()
        pred = model.predict(context, n_steps=30)
        assert pred.activity.shape == (1, 30, 454)
        assert served.provenance.weights_status == "trained"

    def test_it_is_cached_rather_than_reloaded(self, served):
        assert served.predictor() is served.predictor()

    def test_a_service_with_no_checkpoint_refuses_rather_than_substituting(self):
        from scwbd.runtime.serving import ServedModel

        bare = ServedModel.load("nothing-here", device="cpu",
                                checkpoint_root="/nonexistent", purpose="simulation")
        with pytest.raises(CheckpointLoadError) as exc:
            bare.predictor()
        assert "no model to run" in str(exc.value)
        # ...and it did not quietly hand back the analytic backend instead
        assert bare.provenance.weights_status == "analytic_backend"

    def test_the_untrained_service_defers_instead_of_recommending(self):
        """F4: warm_up() returned Recommend while its docstring said Defer."""
        from scwbd.runtime import Defer
        from scwbd.runtime.serving import ServedModel

        bare = ServedModel.load("nothing-here", device="cpu",
                                checkpoint_root="/nonexistent", purpose="simulation")
        evaluation = bare.warm_up()
        assert isinstance(evaluation.decision, Defer), (
            "a service with no trained artifact must not recommend a coil pose"
        )
        assert "no trained artifact behind it" in evaluation.decision.reason
        assert evaluation.decision.suggested_action == "load_a_trained_checkpoint"
