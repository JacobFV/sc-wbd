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
    """The guard must not treat 'nothing recorded' as 'nothing wrong'."""
    from scwbd.foundation import evaluate

    src = inspect.getsource(evaluate.real_eeg_holdout)
    guarded = "recorded is not None" in src
    if not guarded:
        pytest.skip("this build has no recorded-fingerprint guard to audit")
    records_absence = "split_verified" in src or "verified" in src
    assert records_absence, (
        "the guard is `if recorded is not None and <mismatch>: raise`, so a "
        "checkpoint that recorded no fingerprint passes silently and the returned "
        "report says nothing about it. The [warn] is in main(), which "
        "evaluate_model() and real_eeg_holdout() -- both public -- bypass."
    )


def test_verification_status_survives_into_the_returned_report():
    """A warning printed to stdout is not part of the artifact."""
    from scwbd.foundation import evaluate

    main_src = inspect.getsource(evaluate.main)
    if "_recorded_split_fingerprint" not in main_src:
        pytest.skip("this build does not hand a recorded fingerprint to the holdout")
    warns_to_stdout = "[warn]" in main_src and "records no real_split" in main_src
    assert not warns_to_stdout or "split_verified" in inspect.getsource(evaluate), (
        "the 'records no real_split fingerprint' warning goes to stdout only. "
        "evaluation.json is the artifact a reader receives; a provenance warning "
        "that is not in it has not been recorded."
    )
