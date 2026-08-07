"""E3 -- does the evaluation load the weights it says it loaded?

``evaluate.main`` calls ``load_checkpoint(..., strict=False)`` and discards the
return value, then prints ``loaded {ckpt}``.  ``FoundationTrainer`` applies
``torch.compile`` only when ``device.type == 'cuda'``, and ``torch.compile``
renames a wrapped submodule's parameters to ``<name>._orig_mod.<param>``.  So a
checkpoint written by the CUDA run does not key-match a CPU-built model, and
``strict=False`` drops the difference in silence.

``load_checkpoint`` already records what it dropped, in
``payload["load_report"]``.  The information exists; the caller throws it away.
"""

from __future__ import annotations

import pytest
import torch


def test_compiled_checkpoint_keys_match_a_cpu_built_model(cfg, compiled_checkpoint):
    """E3 verdict test.  Fails while the prefix is not reconciled at load time."""
    from scwbd.foundation.anatomy import load_anatomy
    from scwbd.foundation.model import SCWBD

    path, payload = compiled_checkpoint
    from scwbd.runtime.predict import rebuild_anatomy
    anat = rebuild_anatomy((payload.get("extra") or {}).get("anatomy") or {}, device="cpu")
    model = SCWBD(cfg.model, anat)

    before = {k: v.clone() for k, v in model.state_dict().items()}
    param_names = {n for n, _ in model.named_parameters()}
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    total = sum(p.numel() for p in model.parameters())
    dropped = sum(before[k].numel() for k in missing if k in param_names)
    modules = sorted({k.split(".")[0] for k in missing})

    assert not missing and not unexpected, (
        f"loading {path.name} into a CPU-built model leaves {len(missing)} of "
        f"{len(before)} tensors at random initialisation -- "
        f"{dropped:,} of {total:,} *parameters*, {100 * dropped / total:.1f}% of "
        f"the model -- in modules {modules}: the regional operator and the "
        f"residual coupling, i.e. the entire learned dynamics core. Every "
        f"held-out number computed after such a load describes an essentially "
        f"untrained operator, and the run prints 'loaded {path}'."
    )


def test_load_checkpoint_reports_what_it_dropped(cfg, compiled_checkpoint):
    """The information the caller needs already exists -- this must keep passing."""
    from scwbd.foundation.anatomy import load_anatomy
    from scwbd.foundation.checkpoint import load_checkpoint
    from scwbd.foundation.model import SCWBD

    path, peek = compiled_checkpoint
    from scwbd.runtime.predict import rebuild_anatomy

    anat = rebuild_anatomy((peek.get("extra") or {}).get("anatomy") or {}, device="cpu")
    model = SCWBD(cfg.model, anat)
    payload = load_checkpoint(
        str(path), model=model, map_location="cpu", strict=False, restore_rng=False
    )
    assert "load_report" in payload, "load_checkpoint no longer reports partial loads"
    assert payload["load_report"]["missing"], (
        "load_report exists but is empty on a checkpoint that demonstrably does "
        "not key-match; the report itself would then be decorative"
    )


def test_evaluate_asserts_on_what_it_loaded():
    """A load path with no assertion has no bug count, only a lower bound of one.

    Read at the source level deliberately: the defect is that the *caller* drops
    the report, and no runtime observation of ``main()`` distinguishes "loaded
    everything" from "loaded 66% and printed success".
    """
    import inspect

    from scwbd.foundation import evaluate

    src = inspect.getsource(evaluate.main)
    assert "load_report" in src or "strict=True" in src, (
        "evaluate.main calls load_checkpoint(..., strict=False) and discards the "
        "return value, then prints 'loaded {ckpt}' unconditionally. It must "
        "either load strictly or inspect payload['load_report'] and refuse a "
        "partial load; printing success is not a check."
    )
