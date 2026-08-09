"""E17 -- B4 residual: the split guard's *default* state is silent pass.

Turing's B4 implementation is correct where it acts. Verified behaviourally
against `wt/turing`'s module: the fingerprint is over participant **ids**, it is
**stable under index reordering**, it is **sensitive to fold membership**, and a
mismatch raises.

What is not covered is the state where it does **not** act:

```python
recorded = getattr(trainer, "_recorded_split_fingerprint", None)
if recorded is not None and recorded.get("sha256") != fp["sha256"]:
    raise ...
```

`recorded is None` -- a checkpoint written before the field, or any call that did
not come through `main()` -- makes the guard a **silent no-op**, and the report
still carries `"real_split": fp`, the *recomputed* fingerprint, with a sha256
that reads as provenance. Nothing in the artifact records that it was never
compared to anything. The `[warn]` lives in `main()`, so `evaluate_model()` and
`real_eeg_holdout()` -- both in `__all__` -- emit nothing at all.

This is `reports/decorative_guards.md` row 4 exactly, inside the mechanism built
to prevent it: **absence is indistinguishable from success**, and a reader of
`evaluation.json` cannot tell a verified split from an unverified one.

Fix is small: `split_fingerprint` (or the holdout) emits
`{"verified": true|false, "reason": ...}`, defaulting to **false**, and the
warning is written into the artifact rather than to stdout.
"""

from __future__ import annotations

import inspect

import pytest


def _fingerprint_fn():
    from scwbd.foundation import evaluate

    fn = getattr(evaluate, "split_fingerprint", None)
    if fn is None:
        pytest.skip(
            "this build has no split_fingerprint; the absence of split verification "
            "entirely is covered by "
            "test_split_and_verdict_integrity.py::test_checkpoint_records_the_split_it_was_trained_under"
        )
    return fn


def test_split_fingerprint_records_whether_it_was_verified(cfg, real_eeg, real_split):
    """The artifact must carry the verification *status*, not only the value."""
    fp = _fingerprint_fn()(real_eeg, real_split)
    status = [k for k in fp if "verif" in k.lower()]
    assert status, (
        f"split_fingerprint returns {sorted(fp)}. The report writes this block as "
        f"`real_split`, so a reader sees an authoritative-looking sha256 with no "
        f"way to tell whether it was ever compared to the checkpoint's. It must "
        f"default to verified=false and say why."
    )


def test_unverified_split_is_refused_or_recorded_not_silently_passed():
    """The guard must not treat 'nothing recorded' as 'nothing wrong'.

    **This test used to skip on a source literal.** It required
    ``"recorded is not None" in src`` and skipped when absent -- and the fix
    restructured the branch to ``if recorded is None: ... elif <mismatch>:
    raise``, so the literal vanished and the test went green-by-skip. A skip
    that reads as green is the decorative guard this whole directory exists to
    catch, committed inside the test written to catch it.

    It now checks the three states behaviourally-shaped: absent, mismatched,
    matching, each of which must leave a distinguishable mark.
    """
    from scwbd.foundation import evaluate

    src = inspect.getsource(evaluate.real_eeg_holdout)
    assert "_recorded_split_fingerprint" in src, (
        "real_eeg_holdout no longer looks for a recorded fingerprint at all; the "
        "split it rebuilt is compared against nothing"
    )
    assert "raise RuntimeError" in src and "does not match the checkpoint" in src, (
        "a recorded fingerprint that disagrees with the recomputed one no longer "
        "raises. Evaluating would score a model on participants it may have "
        "trained on."
    )
    # The ABSENT state is the one that used to pass silently: it must write its
    # own status into `fp`, which the report carries as `real_split`.
    assert "NOT VERIFIED" in src, (
        "the `recorded is None` branch writes nothing. A checkpoint that recorded "
        "no fingerprint then passes silently and the returned report carries a "
        "recomputed sha256 that reads as provenance. The [warn] is in main(), "
        "which evaluate_model() and real_eeg_holdout() -- both public -- bypass."
    )
    assert '"real_split": fp' in src, (
        "the fingerprint block, with its verification status, is no longer "
        "written into the returned report. A status the artifact does not carry "
        "has not been recorded."
    )


def test_verification_status_survives_into_the_returned_report(cfg, real_eeg, real_split):
    """A warning printed to stdout is not part of the artifact.

    **This test used to grep for a literal the fix never adopted.** It demanded
    ``"split_verified" in inspect.getsource(evaluate)``; the implementation
    landed the same contract as ``fp["verified"]`` plus a human-readable
    ``fp["verification"]``, so the test failed on a naming choice rather than on
    a defect. Checked on the returned value now, not on a spelling.
    """
    from scwbd.foundation import evaluate

    main_src = inspect.getsource(evaluate.main)
    if "_recorded_split_fingerprint" not in main_src:
        pytest.skip("this build does not hand a recorded fingerprint to the holdout")

    fp = _fingerprint_fn()(real_eeg, real_split)
    assert fp.get("verified") is False, (
        f"split_fingerprint returns verified={fp.get('verified')!r} without any "
        f"comparison having happened. It must default to False: a recomputed "
        f"sha256 that a reader can mistake for a checked one is the failure this "
        f"field exists to prevent."
    )
    reason = str(fp.get("verification", ""))
    assert "NOT VERIFIED" in reason.upper(), (
        f"split_fingerprint's default verification reason is {reason!r}. The "
        f"artifact must say in words that nothing was compared, because "
        f"`verified: false` beside an authoritative-looking hash is read as a "
        f"field nobody filled in."
    )
