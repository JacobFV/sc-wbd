"""The publish path must refuse before it uploads, and say why.

Publishing is the one irreversible step in this project: a wrong artifact under
the wrong name on a public hub cannot be recalled, only superseded.  Everything
here tests a *refusal*, because the failure mode that matters is publishing
something rather than declining to.

Three refusals earned their place by firing on real mistakes during run 2:

  identity   an ``HF_TOKEN`` in the environment silently overrode the stored
             CLI token, so ``whoami`` resolved to a different account than the
             one the operator had just logged into
  evaluation the card reads every score from ``evaluation_run2.json``; without
             it there is no honest card, so the artifact is not publishable
  collision  ``create_repo(exist_ok=True)`` uploads *into* an existing repo
             rather than failing, which turns a name collision into a silent
             merge
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


def _publish(*args: str, env_extra: dict[str, str] | None = None):
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env.pop("HF_TOKEN", None)
    env.pop("HUGGING_FACE_HUB_TOKEN", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [PY, "-m", "scwbd.release.publish", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_dry_run_is_the_default_with_no_flag():
    """A publish that uploads unless told not to is one typo from a mistake.

    This is the mutant that survived the first time it was written: every test
    passed ``dry_run=True`` explicitly, so flipping the *default* to push went
    undetected.  A suite that never lets a default choose cannot catch a change
    to it.
    """
    r = _publish("anatomy-prior", "--namespace", "definitely-not-a-real-namespace")
    out = r.stdout + r.stderr
    # Assert the POSITIVE marker, not the absence of a word.  The first version
    # of this test checked `"created" not in out` and failed on the string
    # "no remote state created" -- an instrument that fires on the disclaimer as
    # readily as on the thing disclaimed, which is the exact defect
    # reports/decorative_guards.md exists to catalogue.
    assert "DRY RUN" in out, out[-400:]
    assert "PUSHED" not in out and "UPLOADED" not in out


def test_namespace_is_required_and_never_inferred():
    """Inferring it from a token is how an artifact lands in the wrong account."""
    import os

    env = dict(os.environ)
    env.pop("SCWBD_HF_NAMESPACE", None)
    r = subprocess.run(
        [PY, "-m", "scwbd.release.publish", "anatomy-prior"],
        cwd=ROOT,
        env={**env, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode != 0 or "namespace" in (r.stdout + r.stderr).lower()


def test_run2_without_an_evaluation_is_not_publishable():
    """The card's every score is read from the evaluation; no file, no card."""
    r = _publish(
        "run2-pilot",
        "--namespace",
        "jacob-valdez",
        "--checkpoint-dir",
        "checkpoints/scwbd-002-pilot",
    )
    out = r.stdout + r.stderr
    ev = ROOT / "reports/training/evaluation_run2.json"
    if ev.exists():
        pytest.skip("an evaluation exists; this refusal is no longer reachable")
    assert "NOT PUBLISHABLE" in out, out[-600:]
    assert "evaluation_run2.json" in out


def test_the_refusal_names_what_is_missing():
    """A refusal nobody can act on is a wall, not a gate."""
    r = _publish(
        "run2-pilot",
        "--namespace",
        "jacob-valdez",
        "--checkpoint-dir",
        "checkpoints/scwbd-002-pilot",
    )
    out = r.stdout + r.stderr
    if "NOT PUBLISHABLE" not in out:
        pytest.skip("artifact is publishable; nothing to name")
    # the message must identify the artifact AND the missing thing, not just fail
    assert "scwbd-002-pilot" in out
    assert "not found" in out.lower() or "missing" in out.lower()


def test_whoami_creates_no_remote_state():
    """The identity check must be safe to run before anything is decided."""
    r = _publish("anatomy-prior", "--namespace", "jacob-valdez", "--whoami")
    assert r.returncode in (0, 3), (r.stdout + r.stderr)[-400:]
    assert "PUSHED" not in r.stdout and "UPLOADED" not in r.stdout


def test_a_stale_designation_cannot_reach_a_card():
    """``evaluate.py`` hardcoded ``SC-WBD-001-beta`` and stamped it on run-2
    results — into the very file the card reads.  The designation must come
    from the config."""
    import scwbd.foundation.evaluate as E
    from scwbd.foundation.config import load_config

    cfg = load_config("configs/run2/pilot-families.yaml")
    got = E._designation(cfg)
    assert got and "001" not in got, got
    assert got == "scwbd-002-pilot", got


def test_an_evaluation_on_disk_carries_the_right_model_id():
    """If a stale evaluation is present it must not be readable as 002's."""
    for name in ("evaluation_run2.json", "evaluation_run2_full_smoke.json"):
        f = ROOT / "reports/training" / name
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        assert "001" not in str(d.get("model_id", "")), f"{name}: {d.get('model_id')}"
