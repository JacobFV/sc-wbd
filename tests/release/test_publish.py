"""Negative controls for the publish path.

Every test here is written so that it fails if the thing it guards stops
working.  ``reports/decorative_guards.md`` catalogues ~26 controls in this
repository that looked green and could not fire; the tests below therefore
*break* the input and assert the refusal, rather than asserting that a healthy
input passes.

Two properties get the most attention, because they are the two whose failure
is silent:

* a dry run must reach no network — a publish that "worked" against the wrong
  account leaves no error behind, and
* the attribution gate must actually refuse, because citation is a licence
  condition for at least one input.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from scwbd.release.publish import (
    NAMESPACE_ENV,
    ArtifactPlan,
    NamespaceError,
    PublishBlocked,
    _inputs_from_artifact,
    plan_anatomy_prior,
    publish,
    resolve_namespace,
)
from scwbd.sources.attribution import AttributionBlock, AttributionError


# ---------------------------------------------------------------------------
# the namespace is never guessed
# ---------------------------------------------------------------------------
def test_no_namespace_anywhere_is_a_refusal_not_a_default():
    with pytest.raises(NamespaceError) as exc:
        resolve_namespace(None, env={})
    assert NAMESPACE_ENV in str(exc.value)


def test_namespace_comes_from_the_environment_when_no_flag_is_given():
    assert resolve_namespace(None, env={NAMESPACE_ENV: "some-org"}) == "some-org"


def test_explicit_namespace_wins_over_the_environment():
    assert resolve_namespace("flag-org", env={NAMESPACE_ENV: "env-org"}) == "flag-org"


def test_an_empty_namespace_is_not_a_namespace():
    # The failure this prevents: SCWBD_HF_NAMESPACE="" exported by a wrapper
    # script reads as "set" to a naive check and publishes to "/<name>".
    with pytest.raises(NamespaceError):
        resolve_namespace(None, env={NAMESPACE_ENV: "   "})


def test_a_full_repo_id_is_rejected_as_a_namespace():
    with pytest.raises(NamespaceError):
        resolve_namespace("org/some-model", env={})


def test_resolve_namespace_never_consults_the_hub(monkeypatch):
    """whoami() resolving to an account is not consent to publish under it."""
    called: list[str] = []

    class Boom:
        def __init__(self, *a, **k):
            called.append("HfApi")

        def whoami(self, *a, **k):
            called.append("whoami")
            return {"name": "brandonin"}

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", type(sys)("huggingface_hub")
    )
    sys.modules["huggingface_hub"].HfApi = Boom
    with pytest.raises(NamespaceError):
        resolve_namespace(None, env={})
    assert called == []


# ---------------------------------------------------------------------------
# the gate refuses when attribution is broken
# ---------------------------------------------------------------------------
def _plan_with(unattributable):
    return ArtifactPlan(
        name="x",
        repo_type="dataset",
        attribution=AttributionBlock(entries=(), unattributable=unattributable),
    )


def test_gate_refuses_an_unattributable_artifact():
    """Mutation: an input nothing can cite must stop the publish."""
    plan = _plan_with((("mystery_atlas", "not in any registry"),))
    with pytest.raises(AttributionError) as exc:
        publish(plan, namespace="ns", dry_run=True)
    assert "mystery_atlas" in str(exc.value)


def test_the_gate_fires_on_the_dry_run_path_too():
    """A gate that only fires on --push is a gate nobody has watched fire."""
    plan = _plan_with((("mystery_atlas", "not in any registry"),))
    with pytest.raises(AttributionError):
        publish(plan, namespace="ns", dry_run=True)
    with pytest.raises(AttributionError):
        publish(plan, namespace="ns", dry_run=False)


def test_blockers_refuse_even_when_attribution_is_clean():
    plan = ArtifactPlan(
        name="x",
        repo_type="model",
        attribution=AttributionBlock(entries=(), unattributable=()),
        blockers=("weights file is missing",),
    )
    with pytest.raises(PublishBlocked) as exc:
        publish(plan, namespace="ns", dry_run=True)
    assert "weights file is missing" in str(exc.value)


# ---------------------------------------------------------------------------
# a dry run creates no remote state
# ---------------------------------------------------------------------------
def test_dry_run_never_touches_the_hub(monkeypatch, tmp_path):
    """Mutation: make any Hub call explode, then run a dry run."""
    exploded: list[str] = []

    class Detonate:
        def __init__(self, *a, **k):
            exploded.append("HfApi()")
            raise AssertionError("a dry run reached the Hub")

    fake = type(sys)("huggingface_hub")
    fake.HfApi = Detonate
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)

    plan = ArtifactPlan(
        name="x",
        repo_type="dataset",
        attribution=AttributionBlock(entries=(), unattributable=()),
    )
    res = publish(plan, namespace="ns", dry_run=True, stage_dir=tmp_path)
    assert res["dry_run"] is True
    assert exploded == []
    assert "no remote state" in res["action"]


def test_the_detonator_actually_detonates(monkeypatch, tmp_path):
    """Control for the test above: prove the explosive is live.

    Without this, ``test_dry_run_never_touches_the_hub`` would pass just as
    happily if ``publish`` never reached the Hub under *any* setting, which
    would make it decorative.
    """

    class Detonate:
        def __init__(self, *a, **k):
            raise AssertionError("reached the Hub")

    fake = type(sys)("huggingface_hub")
    fake.HfApi = Detonate
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)

    plan = ArtifactPlan(
        name="x",
        repo_type="dataset",
        attribution=AttributionBlock(entries=(), unattributable=()),
    )
    with pytest.raises(AssertionError, match="reached the Hub"):
        publish(plan, namespace="ns", dry_run=False, stage_dir=tmp_path)


def test_dry_run_is_the_DEFAULT_not_merely_available(monkeypatch, tmp_path):
    """Calling publish() with no dry_run argument must not push.

    This is separate from ``test_dry_run_never_touches_the_hub`` on purpose:
    that test passes ``dry_run=True`` explicitly, so it stays green even if the
    default is flipped to False. A caller who forgets the keyword is exactly
    the accident this guards.
    """

    class Detonate:
        def __init__(self, *a, **k):
            raise AssertionError("the default reached the Hub")

    fake = type(sys)("huggingface_hub")
    fake.HfApi = Detonate
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)

    plan = ArtifactPlan(
        name="x",
        repo_type="dataset",
        attribution=AttributionBlock(entries=(), unattributable=()),
    )
    res = publish(plan, namespace="ns", stage_dir=tmp_path)  # no dry_run kwarg
    assert res["dry_run"] is True


def test_the_cli_defaults_to_dry_run():
    """--push must be required; its absence must mean dry run."""
    import inspect

    sig = inspect.signature(publish)
    assert sig.parameters["dry_run"].default is True
    src = inspect.getsource(sys.modules[publish.__module__]._main)
    assert "dry_run=not args.push" in src


def test_dry_run_stages_the_card_locally(tmp_path):
    plan = ArtifactPlan(
        name="x",
        repo_type="dataset",
        card="# hello\n",
        attribution=AttributionBlock(entries=(), unattributable=()),
    )
    publish(plan, namespace="ns", dry_run=True, stage_dir=tmp_path)
    assert (tmp_path / "README.md").read_text() == "# hello\n"
    assert json.loads((tmp_path / "publish_plan.json").read_text())["name"] == "x"


# ---------------------------------------------------------------------------
# provenance is derived, and its absence is loud
# ---------------------------------------------------------------------------
def test_an_asset_with_no_derivable_provenance_is_unattributable(tmp_path):
    """Mutation: strip a file's provenance and watch it become unpublishable.

    Empty inputs must not read as "derives from nothing", which a licence
    union cannot distinguish from "unrestricted".
    """
    from scwbd.release.publish import _spec_for_asset

    p = tmp_path / "mystery.npz"
    np.savez(p, labels=np.arange(3))
    spec = _spec_for_asset("mystery.npz", {}, tmp_path)
    assert spec.inputs == ()
    assert spec.unattributable is not None
    assert "MANIFEST" in spec.unattributable


def test_inputs_are_read_out_of_the_artifacts_own_meta(tmp_path):
    """The prior's own provenance is the authority, not a typed list."""
    p = tmp_path / "a.npz"
    meta = {
        "provenance": {
            "streams": [{"name": "s", "source_key": "enigma_hcp_sc"}],
            "source_url": "https://github.com/yetianmed/subcortex",
        }
    }
    np.savez(p, _meta=np.array(json.dumps(meta), dtype=object))
    keys = _inputs_from_artifact(p)
    assert "enigma_hcp_sc" in keys  # from the explicit source_key
    assert "tian2020" in keys  # resolved from the URL against SRC


def test_an_unrecognised_source_url_yields_no_key(tmp_path):
    """A URL nobody recognises must not be silently mapped to something."""
    p = tmp_path / "a.npz"
    meta = {"provenance": {"source_url": "https://example.invalid/nope"}}
    np.savez(p, _meta=np.array(json.dumps(meta), dtype=object))
    assert _inputs_from_artifact(p) == ()


# ---------------------------------------------------------------------------
# the Hansen route, checked on the artifact rather than believed
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (__import__("pathlib").Path("/data/scwbd/assets/MANIFEST.json").exists()),
    reason="asset root not attached",
)
class TestHansenRoute:
    def test_the_production_prior_carries_no_hansen(self):
        plan = plan_anatomy_prior(include_maps=False)
        inputs = {i for f in plan.files for i in f.inputs}
        assert not any("hansen" in i for i in inputs), inputs
        assert plan.licence.share_alike_effective is not True

    def test_including_the_regional_maps_flips_it_to_nc_sa(self):
        """The third Hansen route. Faraday's check covered theta and the
        connectome; the maps file is neither and it is the one that bites."""
        plan = plan_anatomy_prior(include_maps=True)
        inputs = {i for f in plan.files for i in f.inputs}
        assert "hansen_receptors" in inputs
        assert plan.licence.noncommercial_effective is True
        assert plan.licence.share_alike_effective is True
        assert "hansen_receptors" in plan.licence.share_alike_sources

    def test_the_tian_citation_is_present_because_it_is_a_licence_condition(self):
        plan = plan_anatomy_prior(include_maps=False)
        assert any("Tian" in c for c in plan.attribution.citations())

    def test_the_production_prior_is_414_parcels(self):
        plan = plan_anatomy_prior(include_maps=False)
        assert "**414 parcels**" in plan.card
