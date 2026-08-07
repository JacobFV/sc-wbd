"""Heterogeneous region-indexed state: the partition, the ports, and the span guard.

Every guard in ``scwbd.foundation.families`` has a test here that **makes it
fire**.  ``reports/decorative_guards.md`` catalogues ~26 checks that looked green
and were incapable of failing; narrowing `padded-family-state` in ``ARCHITECTURE.md`` §5b is only a
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
    return FamilyStateLayout(derive_families(anat, allow_derived=True))


def _small_cfg(**kw) -> ModelConfig:
    base = dict(
        family_state=True,
        # This branch has no anatomy-declared partition, so every model built
        # here uses the fallback -- which now REFUSES unless opted into. Stating
        # it in the helper is the point: the opt-in is visible at every call.
        family_allow_derived_partition=True,
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
    real = derive_families(anat, allow_derived=True)
    synth = derive_families(synthetic_anat, allow_derived=True)
    assert len(real) != len(synth), (
        "the real and synthetic priors distinguish different numbers of families; if these "
        "agree, the partition is being hardcoded somewhere rather than derived"
    )
    # every family name traces to a discriminator the prior actually carries
    for f in real:
        assert f.discriminator and "hardcoded" not in f.discriminator.lower()


def test_unpopulated_taxonomy_families_are_reported_not_invented(anat):
    """The real prior has no cerebellar parcels.  That must be visible."""
    part = derive_families(anat, allow_derived=True)
    assert "cerebellum" in part.unpopulated or "cerebellum" in " ".join(part.notes)
    assert all(f.name != "cerebellum" for f in part), "an empty family must not be fabricated"
    assert any("ZERO regions" in n for n in part.notes)


def test_unparsed_subcortical_parcels_are_not_folded_into_a_neighbour(synthetic_anat):
    part = derive_families(synthetic_anat, allow_derived=True)
    names = {f.name for f in part}
    assert "subcortex_unassigned" in names
    assert "subcortex_hippo" not in names  # the synthetic labels name no structures


def test_declared_partition_wins_over_derivation(anat):
    """Cajal's ``AnatomyPrior`` family declaration is used verbatim when present."""
    a = copy.copy(anat)
    # Declare the partition ONCE. `anat` is the real prior, which already
    # declares nine families in `families`; `copy.copy` carries that over, so
    # setting a per-parcel declaration on top of it produces an anatomy that
    # declares its families twice and differently. `_declared_families` refuses
    # that outright now -- it used to let the structured one win silently while
    # still reporting `source="anatomy_declared"`. Clearing it is what makes
    # this fixture express the thing the test is actually about.
    a.families = None
    a.family = tuple(
        "hippocampus_declared" if d == "subcortex" else f"cortex_block{i % 3}"
        for i, d in enumerate(anat.division)
    )
    part = derive_families(a, allow_derived=True)
    assert part.source == "anatomy_declared"
    assert {f.name for f in part} == {"hippocampus_declared", "cortex_block0", "cortex_block1", "cortex_block2"}
    # the declared name is matched onto an engineered backend by token
    assert part.by_name("hippocampus_declared").backend == "hippocampal_code"


def test_declared_partition_of_arbitrary_size(anat):
    """Build for an arbitrary N declared at load time, not for a fixed list."""
    for n_fam in (2, 3, 17, 41):
        a = copy.copy(anat)
        # Declare the partition ONCE. `anat` is the real prior, which already
        # declares nine families in `families`; `copy.copy` carries that over, so
        # setting a per-parcel declaration on top of it produces an anatomy that
        # declares its families twice and differently. `_declared_families` refuses
        # that outright now -- it used to let the structured one win silently while
        # still reporting `source="anatomy_declared"`. Clearing it is what makes
        # this fixture express the thing the test is actually about.
        a.families = None
        a.family = tuple(f"declared_{i % n_fam:03d}" for i in range(anat.n_regions))
        part = derive_families(a, allow_derived=True)
        assert len(part) == n_fam
        FamilyStateLayout(part)  # must be constructible at any N


def test_partial_family_declaration_raises(anat):
    a = copy.copy(anat)
    # Declare the partition ONCE. `anat` is the real prior, which already
    # declares nine families in `families`; `copy.copy` carries that over, so
    # setting a per-parcel declaration on top of it produces an anatomy that
    # declares its families twice and differently. `_declared_families` refuses
    # that outright now -- it used to let the structured one win silently while
    # still reporting `source="anatomy_declared"`. Clearing it is what makes
    # this fixture express the thing the test is actually about.
    a.families = None
    a.family_id = torch.zeros(anat.n_regions, dtype=torch.long)
    with pytest.raises(ValueError, match="family_names"):
        derive_families(a, allow_derived=True)

    b = copy.copy(anat)
    # Declare the partition ONCE. `anat` is the real prior, which already
    # declares nine families in `families`; `copy.copy` carries that over, so
    # setting a per-parcel declaration on top of it produces an anatomy that
    # declares its families twice and differently. `_declared_families` refuses
    # that outright now -- it used to let the structured one win silently while
    # still reporting `source="anatomy_declared"`. Clearing it is what makes
    # this fixture express the thing the test is actually about.
    b.families = None
    b.family = ("only", "three", "labels")
    with pytest.raises(ValueError, match="partial family declaration|family labels for"):
        derive_families(b, allow_derived=True)


def test_the_derived_fallback_refuses_unless_explicitly_opted_into(anat):
    """A fallback that produces an evidence-rejected partition must not run silently.

    Agent C tested the Yeo-7 cortical split under a Vasa spin null: it separates
    6 of 21 pairs, so it is not a partition. Every other call in this file passes
    ``allow_derived=True`` precisely because the default refuses.
    """
    # The derived fallback is only REACHED by an anatomy that declares no
    # partition. The real prior declares nine families, and a declaration
    # correctly needs no opt-in (see
    # test_an_anatomy_declared_partition_is_consumed_verbatim), so passing `anat`
    # here exercises the declared path and never reaches the refusal this test is
    # about. Strip both declaration channels so the fallback is what runs.
    undeclared = copy.copy(anat)
    undeclared.families = None
    undeclared.family_partition = None
    with pytest.raises(ValueError, match="REFUSED by default"):
        derive_families(undeclared)
    part = derive_families(undeclared, allow_derived=True)
    assert any("REJECTS" in n for n in part.notes), (
        "the opted-in path must still record that the partition is evidence-rejected"
    )


def test_a_training_config_cannot_reach_the_derived_partition_by_default(anat):
    from scwbd.foundation.model import build_family_layout

    cfg = ModelConfig(family_state=True)  # NOT _small_cfg: that opts in
    assert cfg.family_allow_derived_partition is False, "the default must refuse"
    # The derived fallback is only REACHED by an anatomy that declares no
    # partition. The real prior declares nine families, and a declaration
    # correctly needs no opt-in (see
    # test_an_anatomy_declared_partition_is_consumed_verbatim), so passing `anat`
    # here exercises the declared path and never reaches the refusal this test is
    # about. Strip both declaration channels so the fallback is what runs.
    undeclared = copy.copy(anat)
    undeclared.families = None
    undeclared.family_partition = None
    with pytest.raises(ValueError, match="REFUSED by default"):
        build_family_layout(cfg, undeclared)


def test_an_anatomy_declared_partition_is_consumed_verbatim(anat):
    """The handoff from ``scwbd.anatomy.FamilyPartition`` (agent C).

    Reproduces the shape agent C actually shipped — ``family_id`` + ``parcels``
    per family, not the flat per-parcel labelling this module first specified —
    and asserts it takes precedence with no opt-in.
    """
    from dataclasses import dataclass, field as dfield

    @dataclass
    class _F:
        family_id: str
        parcels: tuple
        evidence_tier: str = "measured_separation"
        training_status: str = "has_regional_data"
        separating_evidence: tuple = ()

    @dataclass
    class _P:
        families: tuple
        declared_absent: dict = dfield(default_factory=dict)
        separation_evidence: dict = dfield(default_factory=dict)

    n = anat.n_regions
    ctx = [i for i, d in enumerate(anat.division) if d == "cortex"]
    sub = [i for i, d in enumerate(anat.division) if d != "cortex"]
    a = copy.copy(anat)
    # Declare the partition ONCE: `families` and `family_partition` are two
    # spellings of one declaration, and `_from_anatomy_partition` used to take
    # the first and discard the second in silence. It refuses a disagreement
    # now, so a fixture supplying `family_partition` must not also carry the
    # real prior's `families` inherited through `copy.copy`.
    a.families = None
    a.family_partition = _P(
        families=(
            _F("cortex_unimodal", tuple(ctx[: len(ctx) // 3])),
            _F("cortex_association", tuple(ctx[len(ctx) // 3 :])),
            _F("subcortex_hippo", tuple(sub), "atlas_separation", "prior_only_untrained"),
        ),
        declared_absent={"cerebellum": "no cerebellar parcels in this atlas"},
    )
    part = derive_families(a)  # NO allow_derived -- a declaration needs no opt-in
    assert part.source == "anatomy_declared"
    assert {f.name for f in part} == {"cortex_unimodal", "cortex_association", "subcortex_hippo"}
    assert sum(f.n_regions for f in part) == n
    assert part.by_name("subcortex_hippo").backend == "hippocampal_code"
    assert any("declared_absent" in n or "cerebellum" in n for n in part.notes)


def test_agent_c_family_ids_map_onto_the_engineered_backends():
    """Pins the ids agent C actually shipped against the backend assignment.

    Regression: an earlier ``_KIND_TOKENS`` carried only the long forms
    (``hippocamp``, ``thalam``, ``putamen``, …) and matched **1 of these 7**.
    The other six fell through to the generic learned core with nothing raised —
    every engineered backend body.tex §5 argues for silently unassigned, while
    the config still said ``family_state: true``.
    """
    from scwbd.foundation.families import DEFAULT_FAMILY_CORES, _kind_from_declared_name

    # `_kind_from_declared_name` returns a BACKEND KIND, not a family id. The
    # expectations here read `subcortex_put` / `subcortex_hippo` -- family ids --
    # which is the O-7 vocabulary confusion in a test: the four keys of
    # DEFAULT_FAMILY_CORES are `basal_ganglia`, `cerebellum`, `hippocampus`,
    # `thalamus`, and a kind is a mechanism shared by several families rather
    # than a name for one of them. Corrected to the kind vocabulary.
    expected = {
        "cortex_unimodal": None,
        "cortex_association": None,
        "subcortex_accumb": "basal_ganglia",
        "subcortex_amyg": "amygdala",
        "subcortex_caud": "basal_ganglia",
        "subcortex_hippo": "hippocampus",
        "subcortex_pal": "basal_ganglia",
        "subcortex_put": "basal_ganglia",
        "subcortex_thal": "thalamus",
    }
    for fid, kind in expected.items():
        assert _kind_from_declared_name(fid) == kind, f"{fid} mapped to {_kind_from_declared_name(fid)!r}"
    # and the ones with an engineered backend must actually resolve to one
    for fid, kind in expected.items():
        if kind in DEFAULT_FAMILY_CORES:
            assert resolve_backend(DEFAULT_FAMILY_CORES[kind]) is not None

    # A kind with no core is the defect this test was written for -- the
    # docstring above describes six families falling through to the generic
    # learned core with nothing raised. One case survives: `subcortex_amyg` is
    # given the kind `amygdala`, and `amygdala` is not a key of
    # DEFAULT_FAMILY_CORES, so it resolves to nothing and takes the generic core
    # silently.
    #
    # Asserted as a KNOWN set rather than as "every kind has a core", so that
    # closing it is what makes this line change, and a NEW kind without a core
    # fails here immediately.
    kinds = {k for k in (_kind_from_declared_name(f) for f in expected) if k}
    coreless = sorted(k for k in kinds if k not in DEFAULT_FAMILY_CORES)
    assert coreless == ["amygdala"], (
        f"backend kinds with no engineered core: {coreless}. A family mapped to a "
        "kind that DEFAULT_FAMILY_CORES does not define falls through to the "
        "generic learned core and nothing says so -- which is the failure this "
        "test exists to catch. Add the core, or record the gap here."
    )


def test_hypothalamus_does_not_inherit_the_thalamic_backend():
    """``hypothal`` contains ``thal``.  It is not a thalamic nucleus."""
    from scwbd.foundation.families import _kind_from_declared_name

    assert _kind_from_declared_name("subcortex_hypothal") is None
    assert _kind_from_declared_name("subcortex_thal") == "subcortex_thal"


def test_an_unrecognised_non_cortical_family_is_reported_loudly(anat):
    """§5's engineered backends must not go unassigned quietly."""
    from dataclasses import dataclass

    @dataclass
    class _F:
        family_id: str
        parcels: tuple

    @dataclass
    class _P:
        families: tuple

    ctx = [i for i, d in enumerate(anat.division) if d == "cortex"]
    sub = [i for i, d in enumerate(anat.division) if d != "cortex"]
    a = copy.copy(anat)
    # Declare the partition ONCE: `families` and `family_partition` are two
    # spellings of one declaration, and `_from_anatomy_partition` used to take
    # the first and discard the second in silence. It refuses a disagreement
    # now, so a fixture supplying `family_partition` must not also carry the
    # real prior's `families` inherited through `copy.copy`.
    a.families = None
    a.family_partition = _P(families=(_F("cortex_all", tuple(ctx)), _F("subcortex_zzz", tuple(sub))))
    part = derive_families(a)
    assert any("NON-CORTICAL FAMILIES WITH NO RECOGNISED BACKEND" in n for n in part.notes)
    assert part.by_name("subcortex_zzz").backend == "learned"


def test_an_incomplete_anatomy_partition_raises(anat):
    from dataclasses import dataclass

    @dataclass
    class _F:
        family_id: str
        parcels: tuple

    @dataclass
    class _P:
        families: tuple

    a = copy.copy(anat)
    # Declare the partition ONCE: `families` and `family_partition` are two
    # spellings of one declaration, and `_from_anatomy_partition` used to take
    # the first and discard the second in silence. It refuses a disagreement
    # now, so a fixture supplying `family_partition` must not also carry the
    # real prior's `families` inherited through `copy.copy`.
    a.families = None
    a.family_partition = _P(families=(_F("only_some", tuple(range(10))),))
    with pytest.raises(ValueError, match="unassigned"):
        derive_families(a)


def test_backend_assignment_to_a_nonexistent_family_raises(anat):
    with pytest.raises(KeyError, match="does not produce"):
        derive_families(anat, cores={"cortex_auditory": "wilson_cowan"}, allow_derived=True)


# ======================================================================
# span enforcement — narrowing `padded-family-state`.  Each of these MUST fire.
# ======================================================================
def test_reading_another_familys_component_raises(flayout):
    """A hippocampal component does not *exist* in a cortical family."""
    x = torch.zeros(2, flayout.n_regions, flayout.dim)
    # sanity: the component exists where it should
    assert flayout.get(x, "subcortex_hippo", "k").shape[-1] == 16
    with pytest.raises(SpanViolation, match="does not declare component 'k'"):
        flayout.get(x, "cortex_unimodal", "k")


def test_raw_channel_range_outside_the_span_raises(flayout):
    d = flayout.family("subcortex_thal").dim
    assert flayout.channels("subcortex_thal", 0, d)  # in span
    with pytest.raises(SpanViolation, match=r"span \[0, \d+\) but asked for channels"):
        flayout.channels("subcortex_thal", 0, flayout.dim)


def test_scatter_wider_than_the_span_raises(flayout):
    x = torch.zeros(2, flayout.n_regions, flayout.dim)
    f = flayout.family("subcortex_thal")
    ok = torch.ones(2, f.n_regions, f.dim)
    flayout.scatter(x, "subcortex_thal", ok)  # fine
    too_wide = torch.ones(2, f.n_regions, flayout.dim)
    with pytest.raises(SpanViolation, match="channels wide"):
        flayout.scatter(x, "subcortex_thal", too_wide)


def test_pad_write_is_detected_and_the_offender_is_named(flayout):
    x = torch.zeros(2, flayout.n_regions, flayout.dim)
    flayout.assert_clean(x)  # clean to start
    region = int(flayout.index("subcortex_thal")[0])
    chan = flayout.family("subcortex_thal").dim  # first pad channel of that region
    x[0, region, chan] = 1e-6
    with pytest.raises(SpanViolation) as exc:
        flayout.assert_clean(x, where="unit test")
    msg = str(exc.value)
    assert "subcortex_thal" in msg and f"region {region}" in msg and "padded-family-state" in msg


def test_a_full_width_operator_fires_the_guard(anat):
    """The realistic violation: a dense ``(N, D)`` operator over the padded state.

    ``LearnedResidual`` is exactly that — it is the run-1 residual, and it is what
    the family arm replaces.  Applying it to a family-layout state writes into
    every channel of every region, including the 47% of the plane that is pad.
    If this test stops raising, the padded layout has silently become
    unenforceable and `padded-family-state` must be withdrawn in favour of the ragged layout.
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
    # the price of `padded-family-state` must be in the artifact's own description of itself
    assert flayout.describe()["padding_fraction"] == round(frac, 4)


def test_overlapping_or_orphan_regions_raise(anat):
    part = derive_families(anat, allow_derived=True)
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
    part = derive_families(anat, allow_derived=True)
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
    assert flayout.port(x, "subcortex_hippo", "recall").shape[-1] == 17  # v(16) + rho(1)
    with pytest.raises(PortMismatch, match="is an in-port"):
        flayout.port(x, "subcortex_hippo", "cue")


def test_no_dangling_in_ports(flayout):
    assert flayout.dangling_ports() == []


# ======================================================================
# the engineered subsystems are actually instantiated and actually run
# ======================================================================
def test_subsystem_families_carry_their_engineered_backends(anat):
    model = SCWBD(_small_cfg(), anat)
    assigned = {n: c.backend_name for n, c in model.family_local.mech.items()}
    assert assigned["subcortex_hippo"] == "hippocampal_code"
    assert assigned["subcortex_thal"] == "thalamic_relay"
    assert assigned["subcortex_put"] == "basal_ganglia_gate"
    # the backends resolve to agent E's registry, not a local stand-in
    for name, core in model.family_local.mech.items():
        assert core.backend.origin.startswith("scwbd.dynamics"), (name, core.backend.origin)
        assert core.backend.capacity() == 0, "a mechanistic backend holds no learnable parameters"


def test_hippocampal_family_declares_H_t_exactly(flayout):
    f = flayout.family("subcortex_hippo")
    names = [c.name for c in f.layout.components]
    for comp in ("k", "v", "g", "c", "rho"):
        assert comp in names, f"body.tex §5.1 H_t = {{k,v,g,c,rho}} is missing {comp!r}"
    assert f.backend_components == ("k", "v", "g", "c", "rho")


def test_engineered_backend_state_must_fit_the_declared_components(anat):
    part = derive_families(anat, allow_derived=True)
    f = part.by_name("subcortex_thal")
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
    for name in ("subcortex_hippo", "subcortex_thal", "subcortex_put"):
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
    cfg = _small_cfg(local_core="learned", family_cores={"cortex_unimodal": "wilson_cowan"})
    model = SCWBD(cfg, anat)
    assert model.family_local.mech["cortex_unimodal"].backend_name == "wilson_cowan"
    assert "cortex_association" not in model.family_local.mech  # still learned


def test_ablation_arm_is_a_property_of_the_config():
    assert FoundationConfig(model=ModelConfig(family_state=True)).ablation_arm() == "treatment"
    assert FoundationConfig(model=ModelConfig(family_state=False)).ablation_arm() == "control"


# ======================================================================
# the compiled schema must describe the state the model actually holds
# ======================================================================
@pytest.fixture(scope="module")
def source_specs():
    from scwbd.foundation.config import load_config
    from scwbd.foundation.mixture import SourceSpec

    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    cfg = load_config(repo / "configs" / "scwbd_001_beta.yaml")
    return list(SourceSpec.load_dir(repo / cfg.mixture_cards).values())


def test_schema_carries_one_state_spec_per_family(anat, flayout, source_specs):
    """§2.1 indexes the state *space* by region, so the ABI must too."""
    import scwbd.foundation.compiler_bridge as cb

    schema = cb.build_foundation_schema(anat, source_specs, family_layout=flayout)
    by_family = {}
    for r in schema.regions:
        assert "[family " in r.label, r.label
        fam = r.label.split("[family ")[1].rstrip("]")
        by_family[fam] = set(r.state.components)
    assert len(set(map(frozenset, by_family.values()))) > 1, (
        "every region compiled to the same StateSpec; the schema is describing a "
        "homogeneous model regardless of the weights"
    )
    if "subcortex_hippo" in by_family:
        assert {"k", "v", "g", "c", "rho"} <= by_family["subcortex_hippo"]


def test_schema_refuses_the_opaque_private_block(anat, source_specs):
    """The interface view is not a state space and must not compile as one."""
    import scwbd.foundation.compiler_bridge as cb

    model = SCWBD(_small_cfg(), anat)
    assert "private" in model.layout
    with pytest.raises(cb.SchemaBuildError, match="no declared schema kind"):
        cb.build_foundation_schema(anat, source_specs, layout=model.layout)


def test_schema_refuses_both_layouts_at_once(anat, flayout, source_specs):
    import scwbd.foundation.compiler_bridge as cb

    from scwbd.foundation.state import default_layout

    with pytest.raises(cb.SchemaBuildError, match="not both"):
        cb.build_foundation_schema(anat, source_specs, layout=default_layout(), family_layout=flayout)


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
