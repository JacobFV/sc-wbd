"""Heterogeneous region-indexed state: the partition, the ports, and the span guard.

Every guard in ``scwbd.foundation.families`` has a test here that **makes it
fire**.  ``reports/decorative_guards.md`` catalogues ~26 checks that looked green
and were incapable of failing; narrowing N-1 in ``ARCHITECTURE.md`` §5b is only a
narrowing rather than a defect because the span guard is enforceable, so a
non-firing guard here would invalidate the narrowing, not just the test.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from scwbd.foundation.anatomy import load_anatomy
from scwbd.foundation.backends import resolve_backend
from scwbd.foundation.config import FoundationConfig, ModelConfig
from scwbd.foundation.families import (
    FamilyStateLayout,
    Port,
    PortMismatch,
    RegionFamily,
    SpanViolation,
    derive_families,
)
from scwbd.foundation.manifest import Claim, ClaimManifest, R12Violation
from scwbd.foundation.model import SCWBD
from scwbd.foundation.state import ComponentSpec, StateLayout


@pytest.fixture(scope="module")
def anat():
    return load_anatomy(device="cpu")


@pytest.fixture(scope="module")
def synthetic_anat():
    return load_anatomy(device="cpu", force_fallback=True)


@pytest.fixture(scope="module")
def flayout(anat):
    return FamilyStateLayout(derive_families(anat))


def _small_cfg(**kw) -> ModelConfig:
    base = dict(
        family_state=True,
        hidden=64,
        n_local_layers=2,
        region_embed=16,
        context_dim=32,
        encoder_channels=16,
        encoder_layers=2,
    )
    base.update(kw)
    return ModelConfig(**base)


# ======================================================================
# the partition comes from the prior, not from this file
# ======================================================================
def test_partition_covers_every_parcel_exactly_once(anat, flayout):
    seen: list[int] = []
    for f in flayout:
        seen.extend(f.regions)
    assert sorted(seen) == list(range(anat.n_regions))
    assert len(seen) == len(set(seen))


def test_family_count_is_whatever_the_prior_distinguishes(anat, synthetic_anat):
    """Not a fixed list: the two priors legitimately give different counts."""
    real = derive_families(anat)
    synth = derive_families(synthetic_anat)
    assert len(real) != len(synth), (
        "the real and synthetic priors distinguish different numbers of families; if these "
        "agree, the partition is being hardcoded somewhere rather than derived"
    )
    # every family name traces to a discriminator the prior actually carries
    for f in real:
        assert f.discriminator and "hardcoded" not in f.discriminator.lower()


def test_unpopulated_taxonomy_families_are_reported_not_invented(anat):
    """The real prior has no cerebellar parcels.  That must be visible."""
    part = derive_families(anat)
    assert "cerebellum" in part.unpopulated
    assert all(f.name != "cerebellum" for f in part), "an empty family must not be fabricated"
    assert any("ZERO regions" in n for n in part.notes)


def test_unparsed_subcortical_parcels_are_not_folded_into_a_neighbour(synthetic_anat):
    part = derive_families(synthetic_anat)
    names = {f.name for f in part}
    assert "subcortex_unassigned" in names
    assert "hippocampus" not in names  # the synthetic labels name no structures


def test_declared_partition_wins_over_derivation(anat):
    """Cajal's ``AnatomyPrior`` family declaration is used verbatim when present."""
    a = copy.copy(anat)
    a.family = tuple(
        "hippocampus_declared" if d == "subcortex" else f"cortex_block{i % 3}"
        for i, d in enumerate(anat.division)
    )
    part = derive_families(a)
    assert part.source == "anatomy_declared"
    assert {f.name for f in part} == {"hippocampus_declared", "cortex_block0", "cortex_block1", "cortex_block2"}
    # the declared name is matched onto an engineered backend by token
    assert part.by_name("hippocampus_declared").backend == "hippocampal_code"


def test_declared_partition_of_arbitrary_size(anat):
    """Build for an arbitrary N declared at load time, not for a fixed list."""
    for n_fam in (2, 3, 17, 41):
        a = copy.copy(anat)
        a.family = tuple(f"declared_{i % n_fam:03d}" for i in range(anat.n_regions))
        part = derive_families(a)
        assert len(part) == n_fam
        FamilyStateLayout(part)  # must be constructible at any N


def test_partial_family_declaration_raises(anat):
    a = copy.copy(anat)
    a.family_id = torch.zeros(anat.n_regions, dtype=torch.long)
    with pytest.raises(ValueError, match="family_names"):
        derive_families(a)

    b = copy.copy(anat)
    b.family = ("only", "three", "labels")
    with pytest.raises(ValueError, match="partial family declaration|family labels for"):
        derive_families(b)


def test_backend_assignment_to_a_nonexistent_family_raises(anat):
    with pytest.raises(KeyError, match="does not produce"):
        derive_families(anat, cores={"cortex_auditory": "wilson_cowan"})


# ======================================================================
# span enforcement — narrowing N-1.  Each of these MUST fire.
# ======================================================================
def test_reading_another_familys_component_raises(flayout):
    """A hippocampal component does not *exist* in a cortical family."""
    x = torch.zeros(2, flayout.n_regions, flayout.dim)
    # sanity: the component exists where it should
    assert flayout.get(x, "hippocampus", "k").shape[-1] == 16
    with pytest.raises(SpanViolation, match="does not declare component 'k'"):
        flayout.get(x, "cortex_vis", "k")


def test_raw_channel_range_outside_the_span_raises(flayout):
    d = flayout.family("thalamus").dim
    assert flayout.channels("thalamus", 0, d)  # in span
    with pytest.raises(SpanViolation, match=r"span \[0, \d+\) but asked for channels"):
        flayout.channels("thalamus", 0, flayout.dim)


def test_scatter_wider_than_the_span_raises(flayout):
    x = torch.zeros(2, flayout.n_regions, flayout.dim)
    f = flayout.family("thalamus")
    ok = torch.ones(2, f.n_regions, f.dim)
    flayout.scatter(x, "thalamus", ok)  # fine
    too_wide = torch.ones(2, f.n_regions, flayout.dim)
    with pytest.raises(SpanViolation, match="channels wide"):
        flayout.scatter(x, "thalamus", too_wide)


def test_pad_write_is_detected_and_the_offender_is_named(flayout):
    x = torch.zeros(2, flayout.n_regions, flayout.dim)
    flayout.assert_clean(x)  # clean to start
    region = int(flayout.index("thalamus")[0])
    chan = flayout.family("thalamus").dim  # first pad channel of that region
    x[0, region, chan] = 1e-6
    with pytest.raises(SpanViolation) as exc:
        flayout.assert_clean(x, where="unit test")
    msg = str(exc.value)
    assert "thalamus" in msg and f"region {region}" in msg and "N-1" in msg


def test_a_full_width_operator_fires_the_guard(anat):
    """The realistic violation: a dense ``(N, D)`` operator over the padded state.

    ``LearnedResidual`` is exactly that — it is the run-1 residual, and it is what
    the family arm replaces.  Applying it to a family-layout state writes into
    every channel of every region, including the 52% of the plane that is pad.
    If this test stops raising, the padded layout has silently become
    unenforceable and N-1 must be withdrawn in favour of the ragged layout.
    """
    from scwbd.foundation.model import LearnedResidual

    cfg = _small_cfg()
    model = SCWBD(cfg, anat)
    fl = model.family_layout
    x = torch.zeros(2, fl.n_regions, fl.dim)
    flat = LearnedResidual(model.layout, fl.n_regions, cfg, in_extra=cfg.hidden // 2)
    with torch.no_grad():
        torch.nn.init.normal_(flat.net[-1].weight, std=0.1)
        torch.nn.init.normal_(flat.net[-1].bias, std=0.1)
        dx = flat(x, torch.zeros(2, fl.n_regions, cfg.hidden // 2))
    assert dx.shape[-1] == fl.dim
    with pytest.raises(SpanViolation, match="out-of-span write detected"):
        fl.assert_clean(x + dx, where="flat LearnedResidual")


def test_conformant_rollout_leaves_the_pad_clean(anat):
    """The guard is not vacuously red: the model's own operators pass it."""
    torch.manual_seed(0)
    model = SCWBD(_small_cfg(), anat)
    theta = torch.randn(2, 6) * 0.2
    model.set_mechanistic_theta(theta, anat)
    res = model.rollout(
        y_context=torch.randn(2, 8, anat.n_regions) * 0.1, theta=theta, n_steps=6
    )
    model.family_layout.assert_clean(res.state, where="test")
    assert torch.isfinite(res.state).all()


def test_padding_fraction_is_reported(flayout):
    frac = flayout.padding_fraction()
    assert 0.0 < frac < 1.0
    # the price of N-1 must be in the artifact's own description of itself
    assert flayout.describe()["padding_fraction"] == round(frac, 4)


def test_overlapping_or_orphan_regions_raise(anat):
    part = derive_families(anat)
    fams = list(part.families)
    # drop one family: its parcels become orphans
    orphaned = replace(part, families=tuple(fams[1:]))
    with pytest.raises(SpanViolation, match="belong to no family"):
        FamilyStateLayout(orphaned)
    # duplicate a family's regions onto another: overlap
    clash = replace(
        part, families=(fams[0], replace(fams[1], regions=fams[0].regions + fams[1].regions)) + tuple(fams[2:])
    )
    with pytest.raises(SpanViolation, match="already owned by another family"):
        FamilyStateLayout(clash)


# ======================================================================
# ports
# ======================================================================
def test_a_port_over_an_undeclared_component_raises():
    lay = StateLayout((ComponentSpec("rate_e", 1, "Hz", "fast"),))
    with pytest.raises(SpanViolation, match="does not have"):
        RegionFamily(
            name="bad",
            layout=lay,
            backend="learned",
            ports=(Port("recall", ("v",), "dimensionless", "out"),),
            division="cortex",
            discriminator="test",
            rationale="test",
        )


def test_same_port_name_with_conflicting_units_raises(anat):
    part = derive_families(anat)
    fams = list(part.families)
    i = next(k for k, f in enumerate(fams) if f.name.startswith("cortex_"))
    bad = replace(
        fams[i],
        ports=tuple(
            Port(p.name, p.components, "furlongs" if p.name == "activity" else p.units, p.direction)
            for p in fams[i].ports
        ),
    )
    fams[i] = bad
    lay = FamilyStateLayout(replace(part, families=tuple(fams)))
    with pytest.raises(PortMismatch, match="conflicting units"):
        lay.check_ports()


def test_reading_an_in_port_raises(flayout):
    x = torch.zeros(2, flayout.n_regions, flayout.dim)
    assert flayout.port(x, "hippocampus", "recall").shape[-1] == 17  # v(16) + rho(1)
    with pytest.raises(PortMismatch, match="is an in-port"):
        flayout.port(x, "hippocampus", "cue")


def test_no_dangling_in_ports(flayout):
    assert flayout.dangling_ports() == []


# ======================================================================
# the engineered subsystems are actually instantiated and actually run
# ======================================================================
def test_subsystem_families_carry_their_engineered_backends(anat):
    model = SCWBD(_small_cfg(), anat)
    assigned = {n: c.backend_name for n, c in model.family_local.mech.items()}
    assert assigned["hippocampus"] == "hippocampal_code"
    assert assigned["thalamus"] == "thalamic_relay"
    assert assigned["basal_ganglia"] == "basal_ganglia_gate"
    # the backends resolve to agent E's registry, not a local stand-in
    for name, core in model.family_local.mech.items():
        assert core.backend.origin.startswith("scwbd.dynamics"), (name, core.backend.origin)
        assert core.backend.capacity() == 0, "a mechanistic backend holds no learnable parameters"


def test_hippocampal_family_declares_H_t_exactly(flayout):
    f = flayout.family("hippocampus")
    names = [c.name for c in f.layout.components]
    for comp in ("k", "v", "g", "c", "rho"):
        assert comp in names, f"body.tex §5.1 H_t = {{k,v,g,c,rho}} is missing {comp!r}"
    assert f.backend_components == ("k", "v", "g", "c", "rho")


def test_engineered_backend_state_must_fit_the_declared_components(anat):
    part = derive_families(anat)
    f = part.by_name("thalamus")
    f.check_backend(resolve_backend(f.backend))  # passes
    truncated = replace(f, backend_components=("relay",))  # 2 channels for a 3-dim backend
    with pytest.raises(SpanViolation, match="must fit its declared components"):
        truncated.check_backend(resolve_backend(f.backend))


def test_engineered_backends_contribute_a_nonzero_drift(anat):
    """§5's argument has to move numbers, not just appear in a config."""
    torch.manual_seed(0)
    model = SCWBD(_small_cfg(), anat)
    theta = torch.randn(2, 6) * 0.2
    model.set_mechanistic_theta(theta, anat)
    fl = model.family_layout
    x = fl.zero_pad(torch.randn(2, fl.n_regions, fl.dim) * 0.1)
    cpl = torch.randn(2, fl.n_regions, model.cfg.message_dim) * 0.1
    films = model.family_local.prepare(model.make_context(theta), x.dtype)
    dx = model.family_local(x, cpl, films, packs=model._family_packs)
    for name in ("hippocampus", "thalamus", "basal_ganglia"):
        block = fl.gather(dx, name).detach()
        assert float(block.abs().sum()) > 0, f"{name} backend produced an all-zero drift"
    fl.assert_clean(dx, where="family drift")


def test_mechanistic_family_without_a_param_pack_raises(anat):
    model = SCWBD(_small_cfg(), anat)
    fl = model.family_layout
    x = torch.zeros(2, fl.n_regions, fl.dim)
    cpl = torch.zeros(2, fl.n_regions, model.cfg.message_dim)
    films = model.family_local.prepare(torch.zeros(2, model.cfg.context_dim), x.dtype)
    with pytest.raises(SpanViolation, match="no ParamPack was bound"):
        model.family_local(x, cpl, films, packs={})


# ======================================================================
# the control arm still works — Popper has to be able to run it
# ======================================================================
def test_single_string_local_core_still_builds_and_rolls(anat):
    """``local_core`` as one string is the §11.4 equal-capacity control arm."""
    cfg = _small_cfg(family_state=False)
    model = SCWBD(cfg, anat)
    assert model.family_layout is None
    assert model.layout.dim == 28
    theta = torch.randn(2, 6) * 0.2
    model.set_mechanistic_theta(theta, anat)
    res = model.rollout(y_context=torch.randn(2, 8, anat.n_regions) * 0.1, theta=theta, n_steps=6)
    assert torch.isfinite(res.state).all()
    assert model.family_report()["ablation_arm"] == "control"


def test_control_arm_with_a_mechanistic_local_core(anat):
    cfg = _small_cfg(family_state=False, local_core="wilson_cowan")
    model = SCWBD(cfg, anat)
    assert model.mechanistic is not None and "mech" in model.layout


def test_per_family_backend_assignment_beats_the_global_default(anat):
    cfg = _small_cfg(local_core="learned", family_cores={"cortex_vis": "wilson_cowan"})
    model = SCWBD(cfg, anat)
    assert model.family_local.mech["cortex_vis"].backend_name == "wilson_cowan"
    assert "cortex_default" not in model.family_local.mech  # still learned


def test_ablation_arm_is_a_property_of_the_config():
    assert FoundationConfig(model=ModelConfig(family_state=True)).ablation_arm() == "treatment"
    assert FoundationConfig(model=ModelConfig(family_state=False)).ablation_arm() == "control"


# ======================================================================
# refusal R12
# ======================================================================
def _manifest(**kw) -> ClaimManifest:
    return ClaimManifest(cannot_do=("not a twin",), **kw)


def _claim(statement: str, **kw) -> Claim:
    return Claim(
        id="c1",
        statement=statement,
        status="partial",
        evidence_status="mixed",
        falsifier="stated",
        **kw,
    )


def test_r12_refuses_a_family_state_claim_on_a_control_checkpoint():
    m = _manifest(regional_state={"ablation_arm": "control", "family_state": False})
    m.add_claim(_claim("The model maintains heterogeneous regional state per parcel."))
    with pytest.raises(R12Violation, match=r"\[R12\]"):
        m.validate()


def test_r12_catches_the_flag_as_well_as_the_prose():
    m = _manifest(regional_state={"ablation_arm": "control", "family_state": False})
    m.add_claim(_claim("Regions are modelled.", requires_family_state=True))
    with pytest.raises(R12Violation):
        m.validate()


def test_r12_refuses_when_no_arm_is_declared_at_all():
    m = _manifest()
    m.add_claim(_claim("Per-family operators are assigned from the anatomy prior."))
    with pytest.raises(R12Violation, match="declares no regional-state arm"):
        m.validate()


def test_r12_admits_the_claim_on_a_treatment_checkpoint():
    m = _manifest(regional_state={"ablation_arm": "treatment", "family_state": True})
    m.add_claim(_claim("The model maintains heterogeneous regional state.", requires_family_state=True))
    m.validate()  # must not raise


def test_r12_lets_an_honest_control_arm_manifest_through():
    m = _manifest(regional_state={"ablation_arm": "control", "family_state": False})
    m.add_claim(_claim("One pooled vector per region; this is the equal-capacity control arm."))
    m.validate()


def test_checkpoint_emission_declares_the_arm(anat, tmp_path):
    from scwbd.foundation.checkpoint import save_checkpoint

    cfg = FoundationConfig(model=_small_cfg())
    model = SCWBD(cfg.model, anat)
    man = _manifest()
    man.add_claim(_claim("Heterogeneous region-indexed state.", requires_family_state=True))
    save_checkpoint(tmp_path / "ck.pt", model=model, config=cfg, step=0, stage="test", manifest=man)
    assert man.regional_state["ablation_arm"] == "treatment"
    assert man.regional_state["partition"]["n_families"] == len(model.family_layout)

    # the same manifest on a control-arm model must be refused at emission
    control = SCWBD(_small_cfg(family_state=False), anat)
    man2 = _manifest()
    man2.add_claim(_claim("Heterogeneous region-indexed state.", requires_family_state=True))
    with pytest.raises(R12Violation):
        save_checkpoint(
            tmp_path / "ck2.pt",
            model=control,
            config=FoundationConfig(model=_small_cfg(family_state=False)),
            step=0,
            stage="test",
            manifest=man2,
        )
