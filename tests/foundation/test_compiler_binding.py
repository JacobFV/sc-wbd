"""The compiler->torch binding must actually bind.

The failure mode these tests exist for is *silent non-binding*: a declared
parameter group whose glob matches no tensor on the model.  Nothing crashes, the
gradient mask is still built, every source still gets a permission -- and the
permission governs nothing.  The audit reports it, but a report only helps if
something reads it, so it is read here.

This is not hypothetical.  The production run of 2026-08-05 opened with a wall
of ``compiler->torch binding is incomplete`` covering every ``region:*:state:*``
group, trained anyway, and would have produced a checkpoint whose gradient masks
were decorative.  A test that fails on an unmatched pattern is the thing that
was missing.

The cause is worth stating because it is invisible from a CPU test run:
``cfg.model.compile`` is true on CUDA, so ``FoundationTrainer`` wraps
``model.local`` and ``model.residual`` in ``torch.compile``, which renames their
parameters to ``local._orig_mod.*``.  Prefix globs (``local.*``) kept matching;
exact names (``local.embed``) did not.  The permission system therefore
half-applied while continuing to look enforced -- which is why several tests
below bind against a *compiled* model rather than a bare one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scwbd.foundation.anatomy import load_anatomy
from scwbd.foundation.config import load_config
from scwbd.foundation.mixture import SourceSpec
from scwbd.foundation.model import SCWBD
from scwbd.foundation.util import set_determinism

cb = pytest.importorskip("scwbd.foundation.compiler_bridge")

REPO = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not cb.compiler_available(), reason="scwbd.compiler unavailable"
)


@pytest.fixture(scope="module")
def anat():
    return load_anatomy(
        device="cpu", n_cortex=40, n_subcortex=12, n_cerebellum=8, density=0.15, seed=7
    )


@pytest.fixture(scope="module")
def cfg():
    c = load_config(REPO / "configs" / "scwbd_001_beta.yaml")
    c.model.n_regions = 60
    return c


@pytest.fixture(scope="module")
def model(cfg, anat):
    set_determinism(0)
    return SCWBD(cfg.model, anat)


@pytest.fixture(scope="module")
def compiled(cfg, anat):
    """Compile the production source cards, as ``FoundationTrainer`` does."""
    specs = SourceSpec.load_dir(cfg.mixture_cards)
    assert specs, "no source cards found; the production config ships them"
    probe = [
        SourceSpec(**{**s.as_dict(), "gradient_permission": s.compiler_permission})
        for s in specs.values()
    ]
    return cb.compile_foundation(anat, probe)


def _params(model) -> list[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]


def _buffers(model) -> list[str]:
    return [n for n, _ in model.named_buffers()]


def _matches(pattern: str, names) -> list[str]:
    from fnmatch import fnmatchcase

    return [n for n in names if fnmatchcase(n, pattern) or n == pattern]


# ----------------------------------------------------------------------
# the tables themselves describe the model that exists
# ----------------------------------------------------------------------
def test_every_declared_binding_matches_a_trainable_parameter(model):
    """No entry in FOUNDATION_BINDING may be a pattern that matches nothing.

    An empty tuple is allowed -- that is the explicit "this group has no
    trainable tensor" statement.  A *written* pattern that misses is not.
    """
    names = _params(model)
    dead = [
        (group, pat)
        for group, pats in cb.FOUNDATION_BINDING.items()
        for pat in pats
        if not _matches(pat, names)
    ]
    assert not dead, (
        "FOUNDATION_BINDING patterns match no trainable parameter of SCWBD -- the "
        f"declaration has drifted from the implementation: {dead}"
    )


def test_every_frozen_binding_matches_a_real_buffer(model):
    """Frozen groups name buffers, and those buffers must exist.

    "Implemented, but by frozen state" is a stronger and more falsifiable claim
    than "no tensor", which is exactly why it is worth checking: if the lead
    field or the delay-bin buffer is renamed, this fails instead of quietly
    downgrading to "declared empty".
    """
    bufs = _buffers(model)
    dead = [
        (group, pat)
        for group, pats in cb.FOUNDATION_FROZEN_BINDING.items()
        for pat in pats
        if not _matches(pat, bufs)
    ]
    assert not dead, f"FOUNDATION_FROZEN_BINDING names buffers SCWBD does not have: {dead}"


def test_frozen_and_trainable_tables_do_not_overlap(model):
    """A group is trainable or frozen, never quietly both."""
    for group, pats in cb.FOUNDATION_FROZEN_BINDING.items():
        trainable = cb.FOUNDATION_BINDING.get(group, ())
        assert not trainable, (
            f"{group!r} is declared frozen but also bound to trainable patterns "
            f"{trainable}; the audit cannot report both truthfully"
        )
        assert pats, f"{group!r} is in the frozen table with no buffer named"


# ----------------------------------------------------------------------
# the compiled schema binds end to end
# ----------------------------------------------------------------------
def test_production_binding_reports_no_problems(model, compiled):
    """The gate on restarting production training (ARCHITECTURE.md §7 rule 2)."""
    audit = cb.audit_binding(model, compiled)
    assert audit["problems"] == [], "\n".join(audit["problems"])
    assert audit["unbound_groups"] == []
    assert audit["empty_bindings"] == []


def test_no_trainable_parameter_is_ungoverned(model, compiled):
    """Every trainable tensor is claimed by some declared group.

    An unclaimed parameter trains under no source card's permission, so no
    reviewer can see that it trains at all.
    """
    audit = cb.audit_binding(model, compiled)
    assert audit["unclaimed_parameters"] == []
    assert audit["n_unclaimed_elements"] == 0


def test_strict_binding_is_clean_for_the_production_config(model, compiled):
    cb.bind_masks(model, compiled, strict=True)  # must not raise


def test_frame_edge_calibration_is_bound_to_the_lead_field(model, compiled):
    """The gap that blocked the restart: the anat->cap edge had no entry.

    It is bound to frozen state rather than a gradient, and to the lead field
    specifically, because that is where the co-registration is applied.
    """
    audit = cb.audit_binding(model, compiled)
    edges = [g for g in audit["groups"] if g.startswith("frame_edge:")]
    assert edges, "the compiled schema declares no frame-edge calibration group"
    for g in edges:
        rep = audit["groups"][g]
        assert rep["n_parameters"] == 0, "the frame transform must not be trainable"
        assert rep["frozen_buffers"] == ["eeg.L"]
        assert compiled.gradient_masks.sources_updating(g) == ()


def test_long_range_delay_is_frozen_not_missing(model, compiled):
    audit = cb.audit_binding(model, compiled)
    rep = audit["groups"]["operator:long_range:delay"]
    assert rep["n_parameters"] == 0
    assert rep["frozen_buffers"] == ["coupling.bin_length_mm"]
    assert "operator:long_range:delay" in audit["frozen_groups"]
    assert "operator:long_range:delay" not in audit["declared_empty_groups"]


def test_frozen_group_grants_no_trainable_glob(model, compiled):
    """A source may hold a frozen group; it must not gain a trainable pattern.

    ``sim_wholebrain`` is allowed the long-range operator, delay included.  The
    delay is a buffer, so the honest translation grants zero torch globs for it
    -- not a glob that happens to match nothing.
    """
    binds = cb.bind_masks(model, compiled)
    holders = compiled.gradient_masks.sources_updating("operator:long_range:delay")
    assert holders, "expected some source to hold the long-range delay group"
    for sid in holders:
        assert "coupling.bin_length_mm" not in binds[sid]


# ----------------------------------------------------------------------
# the guard actually fires
# ----------------------------------------------------------------------
def test_a_renamed_parameter_breaks_the_audit(model, compiled, monkeypatch):
    """Rename a tensor out from under a binding: the audit must say so.

    Without this, the whole file is unfalsifiable -- it would pass just as well
    against a binding layer that reported nothing.
    """
    monkeypatch.setitem(
        cb.FOUNDATION_BINDING,
        "operator:local_field:residual",
        ("residual.embed", "residual.renamed_in_a_refactor"),
    )
    audit = cb.audit_binding(model, compiled)
    assert audit["empty_bindings"] == [
        {
            "group": "operator:local_field:residual",
            "pattern": "residual.renamed_in_a_refactor",
            "namespace": "parameter",
        }
    ]
    assert any("renamed_in_a_refactor" in p for p in audit["problems"])
    with pytest.raises(cb.CompilerBridgeError):
        cb.bind_masks(model, compiled, strict=True)


def test_a_renamed_buffer_breaks_the_audit(model, compiled, monkeypatch):
    """The frozen table is held to the same standard as the trainable one."""
    monkeypatch.setitem(
        cb.FOUNDATION_FROZEN_BINDING, "operator:long_range:delay", ("coupling.gone",)
    )
    audit = cb.audit_binding(model, compiled)
    assert any(
        e["pattern"] == "coupling.gone" and e["namespace"] == "buffer"
        for e in audit["empty_bindings"]
    )
    with pytest.raises(cb.CompilerBridgeError):
        cb.bind_masks(model, compiled, strict=True)


def test_an_undeclared_group_breaks_the_audit(model, compiled, monkeypatch):
    """Deleting a binding must surface as 'unbound', not as 'no parameters'."""
    monkeypatch.delitem(cb.FOUNDATION_BINDING, cb.FRAME_EDGE_KEY)
    monkeypatch.delitem(cb.FOUNDATION_FROZEN_BINDING, cb.FRAME_EDGE_KEY)
    audit = cb.audit_binding(model, compiled)
    assert any(g.startswith("frame_edge:") for g in audit["unbound_groups"])
    assert any("FOUNDATION_BINDING" in p for p in audit["problems"])


# ----------------------------------------------------------------------
# torch.compile must not silently unbind the model
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def compiled_model(cfg, anat):
    """A model wrapped exactly as ``FoundationTrainer`` wraps it on CUDA.

    ``cfg.model.compile`` is true in the production config, so this -- not the
    bare module -- is the object the real run binds against.  ``torch.compile``
    returns an ``OptimizedModule`` whose parameters are renamed
    ``local._orig_mod.*``; no CUDA and no graph execution is needed to reproduce
    that, which is why this test is cheap and was never written.
    """
    import torch

    set_determinism(0)
    m = SCWBD(cfg.model, anat)
    m.local = torch.compile(m.local, dynamic=False)
    m.residual = torch.compile(m.residual, dynamic=False)
    return m


def test_compiling_a_submodule_really_does_rename_parameters(compiled_model):
    """Guard the premise: if torch stops renaming, the tests below go vacuous."""
    raw = [n for n, _ in compiled_model.named_parameters()]
    assert any("_orig_mod" in n for n in raw), (
        "torch.compile no longer inserts _orig_mod; the normalisation in "
        "util.logical_param_name and the tests below need revisiting"
    )


def test_binding_survives_torch_compile(compiled_model, compiled):
    """The 2026-08-05 failure, reproduced and fixed.

    Every per-region binding missed on the real run because the names it
    referred to had been rewritten.  Binding the compiled model must produce
    exactly the same audit as binding the eager one.
    """
    audit = cb.audit_binding(compiled_model, compiled)
    assert audit["problems"] == [], "\n".join(audit["problems"])
    assert audit["unclaimed_parameters"] == []
    region_groups = [g for g in audit["groups"] if g.startswith("region:")]
    assert region_groups
    for g in region_groups:
        assert audit["groups"][g]["n_parameters"] > 0, (
            f"{g} binds no tensor on a compiled model -- the exact-name patterns "
            "were lost to torch.compile's _orig_mod segments"
        )


def test_compiled_and_eager_models_bind_identically(model, compiled_model, compiled):
    eager = cb.audit_binding(model, compiled)
    opt = cb.audit_binding(compiled_model, compiled)
    assert opt["parameters"] == eager["parameters"]
    assert opt["n_elements"] == eager["n_elements"]


def test_source_permissions_survive_torch_compile(compiled_model):
    """A card's exact-name globs must still reach the tensors they name.

    ``eeg.log_gain`` is not affected by compilation, but ``local.embed`` is, and
    a permission set that half-applies is the dangerous case: it looks enforced.
    """
    spec = SourceSpec(
        id="probe",
        role="prior",
        losses=("prior",),
        gradient_permission=("local.embed", "residual.embed"),
    )
    hits = [n for n, _ in compiled_model.named_parameters() if spec.permits(n)]
    assert sorted(hits) == ["local._orig_mod.embed", "residual._orig_mod.embed"]


def test_gradient_gate_reaches_compiled_parameters(compiled_model):
    from scwbd.foundation.mixture import GradientGate

    spec = SourceSpec(
        id="probe", role="prior", losses=("prior",), gradient_permission=("local.embed",)
    )
    gate = GradientGate(compiled_model, {"probe": spec})
    assert gate.names("probe") == ["local._orig_mod.embed"]


def test_trainer_refuses_to_start_on_a_drifted_binding(cfg, anat, model, monkeypatch):
    """The blocker itself: a drifted binding must stop training, not warn.

    ``_bind_compiler_masks`` wraps everything in a broad ``except`` that falls
    back to the source cards' own globs when the compiler is missing.  Binding
    drift must escape that handler -- otherwise the run degrades to
    "compiler unavailable", which is the shape the 2026-08-05 run had.
    """
    from types import SimpleNamespace

    from scwbd.foundation.train import BindingDriftError, FoundationTrainer

    monkeypatch.setitem(
        cb.FOUNDATION_BINDING,
        "operator:local_field:residual",
        ("residual.renamed_in_a_refactor",),
    )
    stub = SimpleNamespace(
        sources=SourceSpec.load_dir(cfg.mixture_cards), anat=anat, model=model
    )
    with pytest.raises(BindingDriftError, match="renamed_in_a_refactor"):
        FoundationTrainer._bind_compiler_masks(stub)


def test_trainer_still_falls_back_when_the_compiler_is_absent(cfg, anat, model, monkeypatch):
    """The other half: an *absent* compiler is still a recoverable, honest mode.

    Fail-closed on drift must not turn into fail-closed on everything, or the
    foundation module loses its declared fallback path.
    """
    from types import SimpleNamespace

    from scwbd.foundation.train import FoundationTrainer

    monkeypatch.setattr(cb, "compiler_available", lambda: False)
    stub = SimpleNamespace(
        sources=SourceSpec.load_dir(cfg.mixture_cards), anat=anat, model=model
    )
    rep = FoundationTrainer._bind_compiler_masks(stub)
    assert rep["used"] is False
    assert "unavailable" in rep["reason"]


def test_declared_empty_is_not_the_same_as_unbound(model, compiled):
    """The two ways of having no trainable tensor stay distinguishable."""
    audit = cb.audit_binding(model, compiled)
    assert "operator:local_field:delay" in audit["declared_empty_groups"]
    assert "operator:local_field:delay" not in audit["unbound_groups"]
    assert "operator:local_field:delay" not in audit["frozen_groups"]
