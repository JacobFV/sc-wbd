"""Region **families**: heterogeneous, region-indexed structured state (§2.1, §5).

``paper/body.tex`` §2.1 writes ``X_i(t) in 𝒳_i`` — the state *space* carries the
region index, and "the components need not have equal shape".  SC-WBD-001-beta
implemented ``X`` with no ``_i``: one 28-component vector for all 454 parcels and
one ``local_core`` string for the whole brain (``reports/scope_gap.md`` G-1).
This module is the fix.

The unit of heterogeneity is the **family**, not the parcel (declared narrowing
N-2: a 414-way operator assignment has no evidence to fit it).  A family
declares

* its own **component list** and therefore its own dimension,
* its own **backend** (which operator supplies ``F_local`` for its regions),
* its **ports** — the typed quantities it exposes to, and accepts from, other
  families.  Coupling crosses a family boundary through a declared port, never
  through a raw state slice.

Families are **derived from the anatomy prior**, not hardcoded: see
:func:`derive_families`.  Whatever ``load_anatomy()`` distinguishes is what you
get.  If the prior distinguishes fewer families than the taxonomy names, the
missing ones are reported as *declared but unpopulated* rather than invented —
:attr:`FamilyPartition.unpopulated`.

Layout (declared narrowing **`padded-family-state`**)
-----------------------------------
State is stored padded to ``D = max_f dim(f)`` with per-family spans ``[0, d_f)``
on the last axis, because a ragged layout breaks the batched trainer.  That
padding is observationally equivalent to a ragged layout **only if out-of-span
reads and writes are impossible**, so this module enforces the span:

* every read goes through ``(family, component)`` names — asking a family for a
  component it does not declare raises :class:`SpanViolation`, it does not
  return zeros;
* :meth:`FamilyStateLayout.channels` refuses a raw channel range that leaves the
  family's span;
* :meth:`FamilyStateLayout.scatter` refuses a value wider than the span;
* :meth:`FamilyStateLayout.assert_clean` re-derives the pad region from the
  tensor and raises if anything ever wrote there.

That last check is the one that catches code which bypasses the API — which is
the only way a violation can actually happen, and is exactly how it happens in
practice (a full-``D`` residual or readout applied to every region).  See
``tests/foundation/test_family_state.py`` for the tests that make each of these
fire; a guard with no firing test is worse than no guard
(``reports/decorative_guards.md``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterator, Literal, Mapping, Sequence

import torch
from torch import Tensor

from .state import ComponentSpec, StateLayout

__all__ = [
    "SpanViolation",
    "PortMismatch",
    "Port",
    "RegionFamily",
    "FamilyPartition",
    "FamilyStateLayout",
    "FAMILY_TAXONOMY",
    "DEFAULT_FAMILY_CORES",
    "shared_components",
    "derive_families",
]


class SpanViolation(RuntimeError):
    """A family touched state outside its declared span (narrowing **`padded-family-state`**).

    `padded-family-state` permits padded storage *only* because this is enforceable.  If this
    exception cannot be made to fire, the padded layout is not a narrowing, it is
    a defect, and the ragged/segment layout must be used instead.
    """

    code = "N1"


class PortMismatch(TypeError):
    """A cross-family message was routed between incompatible declared ports."""

    code = "N1-port"


# ======================================================================
# ports
# ======================================================================
@dataclass(frozen=True)
class Port:
    """A typed quantity a family exposes or accepts.

    A port is *not* a state slice.  It names the components it is built from, so
    a family may reorganise its private state without changing its interface,
    and a reader can never reach past the interface into the private state.
    """

    name: str
    #: components of the owning family this port is built from (out) / writes to (in)
    components: tuple[str, ...]
    units: str
    direction: Literal["out", "in"]
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "components": list(self.components),
            "units": self.units,
            "direction": self.direction,
            "description": self.description,
        }


# ======================================================================
# a family
# ======================================================================
@dataclass(frozen=True)
class RegionFamily:
    """One block of the partition: its own state space, backend and ports."""

    name: str
    layout: StateLayout
    backend: str
    ports: tuple[Port, ...]
    #: anatomical division the regions came from ("cortex"/"subcortex"/"cerebellum")
    division: str
    #: how :func:`derive_families` told these regions apart — provenance, not decoration
    discriminator: str
    #: why this backend, in one sentence, citing the section that argues for it
    rationale: str
    #: parcel indices into the anatomy prior's region axis
    regions: tuple[int, ...] = ()
    #: components, **in order**, that carry the backend's native state.  Their
    #: dimensions must sum to ``backend.state_dim``; :meth:`check_backend`
    #: verifies that against the resolved backend and raises otherwise, so a
    #: family cannot claim a backend whose state does not fit its declared
    #: components.  Empty for families on the generic learned core.
    backend_components: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        declared = {c.name for c in self.layout.components}
        for c in self.backend_components:
            if c not in declared:
                raise SpanViolation(
                    f"family {self.name!r} maps backend {self.backend!r} onto component {c!r} "
                    f"which it does not declare; it has {sorted(declared)}"
                )
        for p in self.ports:
            missing = [c for c in p.components if c not in declared]
            if missing:
                raise SpanViolation(
                    f"family {self.name!r} declares port {p.name!r} over component(s) {missing} "
                    f"that the family does not have; declared components are {sorted(declared)}. "
                    "A port over a component outside the family's state space is an out-of-span "
                    "read by construction."
                )

    @property
    def dim(self) -> int:
        return self.layout.dim

    @property
    def n_regions(self) -> int:
        return len(self.regions)

    def port(self, name: str) -> Port:
        for p in self.ports:
            if p.name == name:
                return p
        raise KeyError(f"family {self.name!r} has no port {name!r}; has {[p.name for p in self.ports]}")

    def out_ports(self) -> tuple[Port, ...]:
        return tuple(p for p in self.ports if p.direction == "out")

    def in_ports(self) -> tuple[Port, ...]:
        return tuple(p for p in self.ports if p.direction == "in")

    def port_dim(self, name: str) -> int:
        p = self.port(name)
        return sum(self.layout.spec(c).dim for c in p.components)

    @property
    def backend_dim(self) -> int:
        return sum(self.layout.spec(c).dim for c in self.backend_components)

    def backend_slices(self) -> tuple[slice, ...]:
        return tuple(self.layout.slice(c) for c in self.backend_components)

    def check_backend(self, backend) -> None:
        """Verify the resolved backend's state fits the declared components.

        A family that names an engineered backend but whose components do not
        cover its state would silently integrate a truncated vector field.
        """
        want = int(backend.state_dim)
        got = self.backend_dim
        if got != want:
            raise SpanViolation(
                f"family {self.name!r} maps backend {self.backend!r} (state_dim={want}) onto "
                f"components {list(self.backend_components)} totalling {got} channels. The "
                "backend's native state must fit its declared components exactly — a mismatch "
                "means part of the vector field is being integrated into the wrong channel or "
                "dropped."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "division": self.division,
            "discriminator": self.discriminator,
            "rationale": self.rationale,
            "dim": self.dim,
            "n_regions": self.n_regions,
            "backend_components": list(self.backend_components),
            "backend_dim": self.backend_dim,
            "components": self.layout.describe(),
            "ports": [p.as_dict() for p in self.ports],
        }


# ======================================================================
# component vocabularies
# ======================================================================
#: Whether ``dipole`` is part of the shared interface (O-5b).
#:
#: A module-level switch rather than a config field, because it must be settable
#: from a CHECKPOINT: run 2's weights were trained against the pre-O-5b layout
#: (D=59) and cannot be loaded into the post-O-5b one (D=62). The published
#: artifact has to stay evaluable from this tree, so `layout_for_checkpoint()`
#: reads the layout the checkpoint recorded and restores it.
#:
#: Default True: new runs get the dipole. Only a loader reading an old
#: checkpoint should ever set it False, and it says so when it does.
SHARED_DIPOLE: bool = True


from contextlib import contextmanager as _contextmanager


@_contextmanager
def layout_of_checkpoint(path):
    """Build models in the state layout a checkpoint was TRAINED in.

    O-5b widened the shared interface from D=59 to D=62. That is exactly the
    breakage ``ARCHITECTURE.md`` predicted when it deferred the change, and the
    prediction came true the moment it landed: ``scwbd-002-pilot``'s weights
    stopped loading, so the published artifact could no longer be evaluated from
    the tree that describes it.

    The checkpoint carries its own ``state_layout``, so it does not have to be
    guessed. This reads it and restores the matching interface for the duration
    of the block:

        with layout_of_checkpoint("checkpoints/scwbd-002-pilot/last.pt"):
            model = SCWBD(cfg.model, anat)
            model.load_state_dict(ck["model"])   # strict, and it passes

    Scoped rather than global so a process can hold both eras at once, and
    silent about nothing: it prints when it selects the legacy layout, because a
    model quietly built in an old interface is worse than one that fails to load.
    """
    import torch

    global SHARED_DIPOLE
    previous = SHARED_DIPOLE
    try:
        rec = torch.load(path, map_location="cpu", weights_only=False).get("state_layout") or {}
        names = [c.get("name") for c in (rec.get("components") or [])]
    except Exception:
        names = []
    if names and "dipole" not in names:
        # Pre-O-5b: dipole lived inside `private`, cortex-only.
        SHARED_DIPOLE = False
        print(
            f"[layout] {path} records components {names} (dim {rec.get('dim')}); "
            "rebuilding the pre-O-5b interface so these weights load",
            flush=True,
        )
    try:
        yield SHARED_DIPOLE
    finally:
        SHARED_DIPOLE = previous


def shared_components(*, n_uncertainty: int = 4) -> tuple[ComponentSpec, ...]:
    """The interface prefix every family carries, at **identical offsets**.

    Every family owns ``rate_e``/``rate_i`` (what the EEG and behaviour heads
    read), ``hemo`` (Balloon-Windkessel, slow clock, what the BOLD head reads)
    and ``uncertainty`` (§2.7: variance never collapses into bias).  Fixing them
    at offset 0 is what lets the observation heads stay family-agnostic while the
    *private* state below them differs per family.

    This is an interface commitment, not a claim that the systems are alike: the
    heads observe all of them through the same instruments, so they must expose
    the same instrument-facing quantities.
    """
    return (
        ComponentSpec("rate_e", 1, "Hz", "fast", True, True, "excitatory mean-field rate"),
        ComponentSpec("rate_i", 1, "Hz", "fast", True, True, "inhibitory mean-field rate"),
        ComponentSpec("hemo", 4, "dimensionless", "slow", False, False, "Balloon-Windkessel s,f,v,q"),
        ComponentSpec(
            "uncertainty", n_uncertainty, "log_var", "meta", False, False, "per-region predictive log-variance"
        ),
        # O-5b, closed 2026-08-07. The dipole moment is the quantity a lead
        # field integrates against, and it was declared per *cortical family*,
        # which put it in the `private` block that `SCWBD.build_layout`
        # deliberately forbids an observation head from addressing. So
        # `EEGHead.source_moment()` and the `(64, 414, 3)` `matrix_vec` were both
        # built, both correct, and both unreachable -- the same "the work exists
        # and the half that would use it is pointed somewhere else" shape this
        # project keeps finding, in the one place it costs the most: on
        # Schaefer400x7 a per-parcel scalar carries 32.1% of the whitened lead
        # field, a 3-vector moment 83.4%.
        #
        # ARCHITECTURE.md deferred this to run 3 because changing the shared
        # interface changes every offset and would invalidate the checkpoints of
        # the run then training. That run is finished, evaluated and published,
        # so the reason has expired.
        #
        # SUBCORTICAL FAMILIES WRITE ZERO, and this is the one place in this
        # codebase where a zero fill is correct rather than an imputation: a
        # parcel with no cortical sheet contributes no current dipole, and a zero
        # moment contributes exactly zero through `L_vec`. The distinction that
        # makes it correct is that absent *orientation* must stay `NaN` -- a
        # direction of zero length is a lie -- while an absent *moment* genuinely
        # is the zero vector. `AnatomyPrior.normal` keeps its NaN for those 14
        # regions; only the moment is zeroed.
    ) + (
        (ComponentSpec("dipole", 3, "Hz*m", "fast", True, True, "net current-dipole moment, anatomical frame"),)
        if SHARED_DIPOLE
        else ()
    )


_SHARED_PORTS: tuple[Port, ...] = (
    Port(
        "activity",
        ("rate_e", "rate_i"),
        "Hz",
        "out",
        "mean-field rates — the quantity the observation heads and the hemodynamic "
        "compartments consume from every family",
    ),
)


def _cortical(*, n_spectral_modes: int, n_adaptation: int, n_uncertainty: int) -> tuple[
    tuple[ComponentSpec, ...], tuple[Port, ...]
]:
    comps = shared_components(n_uncertainty=n_uncertainty) + (
        ComponentSpec("adaptation", n_adaptation, "dimensionless", "fast", False, False, "adaptation currents"),
        ComponentSpec(
            "spectral", 2 * n_spectral_modes, "dimensionless", "fast", True, True, "quadrature spectral modes"
        ),
        # O-5: the parcel's net current-dipole MOMENT, three numbers in the
        # anatomical frame. Units are Hz*m, not Hz: this is a moment, not a
        # rate, and declaring it as such is what stops it being wired to a port
        # that carries activity (PortMismatch refuses cross-unit wiring).
        #
        # Why it is worth three channels. The fraction of the whitened lead
        # field a regional state can express, measured on the model's own 400
        # cortical parcels (`reports/transforms/resolution_pair_schaefer400.md`
        # §3.1): a per-parcel SCALAR reaches eta = 0.321; a net dipole moment at
        # 3/parcel reaches 0.834. Subdividing those 400 parcels does buy
        # something -- 0.415 at 800 elements, 0.708 at 3154 -- so the case for
        # three channels is a rate, not an absolute: 1200 oriented numbers carry
        # more than 3154 scalars do. Every design decision before this one spent
        # on resolution, which is the more expensive of the two.
        #
    ) + (
        # `dipole` is normally part of `shared_components()` now, at a fixed
        # offset the observation heads can address (O-5b). It is re-declared
        # here ONLY when the shared switch is off, which happens when a loader is
        # rebuilding the pre-O-5b layout to read run 2's checkpoint -- there it
        # was cortex-only and inside `private`. Declaring it in both places at
        # once would give the cortical families two dipole spans, which is worse
        # than either arrangement because the head would read one and the
        # dynamics write the other.
        ()
        if SHARED_DIPOLE
        else (
            ComponentSpec(
                "dipole", 3, "Hz*m", "fast", True, True, "net current-dipole moment, anatomical frame"
            ),
        )
    )
    ports = _SHARED_PORTS + (
        Port("oscillatory", ("spectral",), "dimensionless", "out", "quadrature modes carried by long-range edges"),
        Port(
            "dipole_out", ("dipole",), "Hz*m", "out",
            "net current-dipole moment -- what an EEG/MEG lead field integrates against. A MOMENT: "
            "a port carrying Hz may not be wired to it.",
        ),
        Port("afferent", ("rate_e",), "Hz", "in", "excitatory drive arriving from other families"),
        Port("modulatory", ("adaptation",), "dimensionless", "in", "gain/adaptation control from subcortical families"),
        Port(
            "induced_field", ("dipole",), "Hz*m", "in",
            "exogenous drive resolved along the cortical normal (TMS/tES E-field). Faraday's "
            "impulse_response computes E.n scaled by coherence; it lands here as a VECTOR rather "
            "than being collapsed to a signed scalar.",
        ),
    )
    return comps, ports


def _thalamic(*, n_spectral_modes: int, n_uncertainty: int) -> tuple[tuple[ComponentSpec, ...], tuple[Port, ...]]:
    comps = shared_components(n_uncertainty=n_uncertainty) + (
        ComponentSpec("relay", 2, "dimensionless", "fast", False, True, "ThalamicRelay (m, h): membrane, T-current de-inactivation"),
        ComponentSpec("trn", 1, "dimensionless", "fast", False, False, "reticular (TRN) inhibitory pool"),
        ComponentSpec(
            "spectral", 2 * n_spectral_modes, "dimensionless", "fast", True, True, "thalamocortical quadrature modes"
        ),
    )
    ports = _SHARED_PORTS + (
        Port("relay_out", ("relay",), "dimensionless", "out", "tonic+burst relay output, routed to cortical families"),
        Port("oscillatory", ("spectral",), "dimensionless", "out", "thalamocortical rhythm"),
        Port("cortical_drive", ("rate_e",), "Hz", "in", "corticothalamic feedback"),
        Port("reticular", ("trn",), "dimensionless", "in", "TRN inhibition"),
    )
    return comps, ports


def _basal_ganglia(*, n_spectral_modes: int, n_uncertainty: int) -> tuple[tuple[ComponentSpec, ...], tuple[Port, ...]]:
    comps = shared_components(n_uncertainty=n_uncertainty) + (
        ComponentSpec("nuclei", 3, "dimensionless", "fast", False, True, "BasalGangliaGate (GPe, STN, GPi)"),
        ComponentSpec("striatum", 2, "dimensionless", "fast", False, False, "D1 / D2 striatal projection rates"),
        ComponentSpec("gate", 1, "dimensionless", "fast", True, False, "disinhibition gate in [0,1]"),
        ComponentSpec("spectral", 2 * n_spectral_modes, "dimensionless", "fast", True, True, "beta-band quadrature modes"),
    )
    ports = _SHARED_PORTS + (
        Port("gate_out", ("gate",), "dimensionless", "out", "disinhibition applied to thalamic routing / cortical gain"),
        Port("oscillatory", ("spectral",), "dimensionless", "out", "beta-band output"),
        Port("cortical_drive", ("rate_e",), "Hz", "in", "corticostriatal drive"),
        Port("dopaminergic", ("striatum",), "dimensionless", "in", "receptor-typed D1/D2 gain — NOT a reward signal (§5)"),
    )
    return comps, ports


def _hippocampal(*, d_key: int, d_value: int, d_grid: int, d_context: int, n_uncertainty: int) -> tuple[
    tuple[ComponentSpec, ...], tuple[Port, ...]
]:
    """``H_t = {k, v, g, c, rho}`` — body.tex §5.1, verbatim in shape."""
    comps = shared_components(n_uncertainty=n_uncertainty) + (
        ComponentSpec("k", d_key, "dimensionless", "fast", False, True, "H_t.k — cue / index"),
        ComponentSpec("v", d_value, "dimensionless", "fast", True, True, "H_t.v — bound content"),
        ComponentSpec("g", d_grid, "dimensionless", "fast", False, False, "H_t.g — multiscale relational / grid-like code"),
        ComponentSpec("c", d_context, "dimensionless", "fast", False, True, "H_t.c — temporal & contextual state"),
        ComponentSpec("rho", 1, "probability", "fast", True, False, "H_t.rho — retrieval confidence"),
    )
    ports = _SHARED_PORTS + (
        Port("recall", ("v", "rho"), "dimensionless", "out", "retrieved content with its confidence — the replay-to-cortex path"),
        Port("cue", ("k",), "dimensionless", "in", "cortical cue driving retrieval"),
        Port("context", ("c",), "dimensionless", "in", "contextual/temporal drive"),
    )
    return comps, ports


def _amygdalar(*, n_adaptation: int, n_uncertainty: int) -> tuple[tuple[ComponentSpec, ...], tuple[Port, ...]]:
    comps = shared_components(n_uncertainty=n_uncertainty) + (
        ComponentSpec("relevance", 1, "dimensionless", "fast", True, True, "learned biological relevance of the current input"),
        ComponentSpec("autonomic", 2, "dimensionless", "slow", False, False, "autonomic / defensive preparation channels"),
        ComponentSpec("adaptation", n_adaptation, "dimensionless", "fast", False, False, "adaptation currents"),
    )
    ports = _SHARED_PORTS + (
        Port("relevance_out", ("relevance",), "dimensionless", "out", "relevance signal configuring sensory/mnemonic/action families"),
        Port("sensory", ("rate_e",), "Hz", "in", "sensory and mnemonic afferents"),
    )
    return comps, ports


def _cerebellar(*, d_prediction: int, n_uncertainty: int) -> tuple[tuple[ComponentSpec, ...], tuple[Port, ...]]:
    comps = shared_components(n_uncertainty=n_uncertainty) + (
        ComponentSpec("prediction", d_prediction, "dimensionless", "fast", True, True, "forward-model prediction (Purkinje readout)"),
        ComponentSpec("error", d_prediction, "dimensionless", "fast", False, False, "climbing-fibre error trace"),
        ComponentSpec("eligibility", d_prediction, "dimensionless", "fast", False, False, "delayed granule eligibility trace"),
    )
    ports = _SHARED_PORTS + (
        Port("correction", ("prediction",), "dimensionless", "out", "residual correction returned to cortex via thalamus"),
        Port("mossy", ("rate_e",), "Hz", "in", "mossy-fibre input (cortical + subcortical)"),
        Port("climbing", ("error",), "dimensionless", "in", "climbing-fibre teaching signal"),
    )
    return comps, ports


#: The taxonomy the derivation may populate.  A name here is *not* a promise that
#: the prior contains it — :func:`derive_families` reports which ones came back
#: empty.  ``builder`` takes the dimension knobs and returns (components, ports).
FAMILY_TAXONOMY: dict[str, dict[str, Any]] = {
    "cortex": {"builder": _cortical, "division": "cortex"},
    "thalamus": {"builder": _thalamic, "division": "subcortex"},
    "basal_ganglia": {"builder": _basal_ganglia, "division": "subcortex"},
    "hippocampus": {"builder": _hippocampal, "division": "subcortex"},
    "amygdala": {"builder": _amygdalar, "division": "subcortex"},
    "cerebellum": {"builder": _cerebellar, "division": "cerebellum"},
}


#: Default backend per family *kind*.  This is where body.tex §5's argument —
#: "these systems warrant more engineered regional backends than a generic
#: transformer block" — becomes an artifact rather than a sentence.  Cortical
#: families fall through to the config's ``local_core``: neither body.tex nor the
#: anatomy prior types the seven Yeo networks differently, and inventing a
#: per-network operator assignment would be exactly the unearned differentiation
#: N-2 refuses.
#: Which declared components carry each engineered backend's native state, in
#: order.  :meth:`RegionFamily.check_backend` asserts the widths agree.
BACKEND_COMPONENTS: dict[str, tuple[str, ...]] = {
    "thalamus": ("relay", "trn"),  # ThalamicRelayBackend (m, h, trn) = 3
    "basal_ganglia": ("nuclei",),  # BasalGangliaBackend (gpe, stn, gpi) = 3
    "hippocampus": ("k", "v", "g", "c", "rho"),  # HippocampalCodeBackend, §5.1
    "cerebellum": ("prediction", "error", "eligibility"),  # CerebellarForwardBackend
}


DEFAULT_FAMILY_CORES: dict[str, str] = {
    "thalamus": "thalamic_relay",
    "basal_ganglia": "basal_ganglia_gate",
    "hippocampus": "hippocampal_code",
    "cerebellum": "cerebellar_forward_model",
    # amygdala: §5 says it is "not a scalar fear or valence node", but we have no
    # engineered amygdalar backend. It therefore inherits the generic core and is
    # declared untyped in the report rather than given a backend it has not earned.
}

_RATIONALE: dict[str, str] = {
    "cortex": (
        "Generic core: body.tex types operators per *region family*, and neither §2.1 nor the "
        "anatomy prior distinguishes the seven Yeo networks by operator class. Assigning them "
        "different mechanisms would be unearned (N-2)."
    ),
    "thalamus": (
        "§5: 'Thalamic nuclei contribute to routing, gain, synchrony, state control'. The "
        "relay/burst switch is a *state* (T-current de-inactivation), which a generic block "
        "cannot express — scwbd.dynamics.subcortical.ThalamicRelay does."
    ),
    "basal_ganglia": (
        "§5: 'Basal-ganglia loops participate in action selection, vigor, working-memory gating'. "
        "The direct/indirect/hyperdirect motif with opposite-sign D1/D2 dopaminergic gain is a "
        "constrained circuit — scwbd.dynamics.subcortical.BasalGangliaGate."
    ),
    "hippocampus": (
        "§5.1: 'the reference architecture therefore permits a high-dimensional sparse state "
        "H_t = {k,v,g,c,rho}'. This is the one family whose *state space* body.tex specifies "
        "explicitly, and it is the reason a single D for all parcels is non-conformant."
    ),
    "amygdala": (
        "§5: 'Amygdalar systems learn biological relevance ... they are not a scalar fear or "
        "valence node.' We have no engineered amygdalar backend, so it carries a relevance and "
        "autonomic state but the generic core. DECLARED UNTYPED — see reports/dynamics/family_state.md."
    ),
    "cerebellum": (
        "§5: 'Cerebellar loops learn fast forward predictions and calibrated residual corrections'. "
        "scwbd.dynamics.subcortical.Cerebellum is a granule expansion + delta-rule Purkinje readout; "
        "its own falsifier says it earns 'mechanistic' only against a capacity-matched regressor."
    ),
}


# ======================================================================
# derivation from the anatomy prior
# ======================================================================
#: Tian/aseg structure tokens -> family.  Grouping basal-ganglia nuclei together
#: is the coarsest thing the prior supports without splitting a family across
#: parcels that share no motif; caudate/putamen/pallidum/accumbens are the
#: striatopallidal complex the direct/indirect motif is defined over.
_SUBCORTICAL_TOKENS: dict[str, str] = {
    "hippo": "hippocampus",
    "hippocampus": "hippocampus",
    "amyg": "amygdala",
    "amygdala": "amygdala",
    "thal": "thalamus",
    "thalamus": "thalamus",
    "caud": "basal_ganglia",
    "caudate": "basal_ganglia",
    "put": "basal_ganglia",
    "putamen": "basal_ganglia",
    "pal": "basal_ganglia",
    "pallidum": "basal_ganglia",
    "accumb": "basal_ganglia",
    "nacc": "basal_ganglia",
}

#: Tokens that map an **anatomy-declared** family name onto a taxonomy kind, and
#: therefore onto a component list and an engineered backend.  Order matters:
#: the first token found in the (lower-cased) declared name wins.
#: Order matters — the first token found wins, so more specific entries come
#: first.  ``hypothal`` is listed ahead of ``thal`` on purpose: the hypothalamus
#: is not a thalamic nucleus and must not inherit the thalamic relay backend.
#:
#: The SHORT forms (``hippo``, ``amyg``, ``thal``, ``caud``, ``put``, ``pal``)
#: are here because agent C's shipped family ids use them —
#: ``subcortex_hippo``, ``subcortex_thal``, …  An earlier version of this table
#: had only the long forms (``hippocamp``, ``thalam``, ``putamen``, …) and
#: matched **1 of agent C's 7** subcortical families; the other six fell through
#: to the generic learned core with nothing raised, so every engineered backend
#: §5 argues for would have been silently unassigned.  That is why
#: :func:`derive_families` now reports untyped non-cortical families loudly
#: rather than only in a note.
_KIND_TOKENS: tuple[tuple[str, str], ...] = (
    ("hippocamp", "hippocampus"),
    ("subiculum", "hippocampus"),
    ("entorhinal", "hippocampus"),
    ("hippo", "hippocampus"),
    ("amygdal", "amygdala"),
    ("amyg", "amygdala"),
    ("hypothal", None),  # NOT the thalamus; no engineered backend exists for it
    ("thalam", "thalamus"),
    ("pulvinar", "thalamus"),
    ("thal", "thalamus"),
    ("basal_ganglia", "basal_ganglia"),
    ("striat", "basal_ganglia"),
    ("pallid", "basal_ganglia"),
    ("caudate", "basal_ganglia"),
    ("putamen", "basal_ganglia"),
    ("accumb", "basal_ganglia"),
    ("nacc", "basal_ganglia"),
    ("caud", "basal_ganglia"),
    ("put", "basal_ganglia"),
    ("pal", "basal_ganglia"),
    ("cerebell", "cerebellum"),
)


def _kind_from_declared_name(name: str) -> str | None:
    """Map a declared family name onto a taxonomy kind, or ``None`` if unknown.

    ``None`` means "no engineered backend in this repository", which is a fact to
    report, not a default to apply quietly.
    """
    low = name.lower()
    for tok, kind in _KIND_TOKENS:
        if tok in low:
            return kind
    return None


def _from_anatomy_partition(anat) -> tuple[tuple[str, ...], dict[str, Any]] | None:
    """Consume ``scwbd.anatomy.FamilyPartition`` (agent C) — the authoritative form.

    Agent C shipped a richer object than the flat per-parcel labelling this
    module originally specified: ``FamilyPartition`` with ``RegionFamily``
    entries carrying ``family_id``, ``parcels``, an ``evidence_tier``, a
    ``training_status`` and per-field provenance, plus ``declared_absent`` for
    systems the atlas has no parcels for.  That is a better object and it wins;
    the flat form below is kept only because tests and older priors use it.

    Duck-typed on purpose — this module must not import ``scwbd.anatomy``.
    """
    # `families` and `family_partition` are the SAME declaration in two spellings
    # (ARCHITECTURE.md O-7). This was `families or family_partition`, so whenever
    # both were present the first won and the second was discarded without a
    # word -- the third instance in this file of one declaration channel silently
    # overriding another.
    #
    # Refuse when both are present and describe different partitions. When they
    # agree, or only one exists, behave exactly as before.
    direct = getattr(anat, "families", None)
    named = getattr(anat, "family_partition", None)
    if direct and named:
        d_groups = _partition_of_families(direct)
        n_groups = _partition_of_families(getattr(named, "families", None) or named)
        if d_groups and n_groups and d_groups != n_groups:
            raise ValueError(
                "the anatomy carries BOTH `families` and `family_partition`, and they "
                f"describe different partitions ({len(d_groups)} vs {len(n_groups)} "
                "groups). These are two spellings of one declaration; picking either "
                "silently is how a caller's partition gets replaced by somebody "
                "else's under a provenance that still says `anatomy_declared`. "
                "Remove whichever is stale (ARCHITECTURE.md O-7)."
            )
    part = direct or named
    if part is None:
        return None
    fams = getattr(part, "families", None)
    if not fams:
        return None
    n = int(anat.n_regions)
    names: list[str | None] = [None] * n
    meta: dict[str, Any] = {}
    for f in fams:
        fid = str(getattr(f, "family_id", "") or getattr(f, "name", ""))
        parcels = getattr(f, "parcels", None)
        if not fid or parcels is None:
            raise ValueError(
                "scwbd.anatomy.FamilyPartition entry has no family_id/parcels; the foundation "
                "cannot bind operators to a partition it cannot read"
            )
        for p in parcels:
            p = int(p)
            if not 0 <= p < n:
                raise ValueError(f"family {fid!r} references parcel {p} outside 0..{n - 1}")
            if names[p] is not None:
                raise ValueError(f"parcel {p} is in both {names[p]!r} and {fid!r}")
            names[p] = fid
        meta[fid] = {
            "evidence_tier": getattr(f, "evidence_tier", None),
            "training_status": getattr(f, "training_status", None),
            "membership_source": getattr(f, "membership_source", None),
            "separating_evidence": list(getattr(f, "separating_evidence", ()) or ()),
        }
    missing = [i for i, v in enumerate(names) if v is None]
    if missing:
        raise ValueError(
            f"scwbd.anatomy.FamilyPartition leaves {len(missing)} parcel(s) unassigned, e.g. "
            f"{missing[:5]}. Every parcel must belong to exactly one family or nothing downstream "
            "can be enforced about its span."
        )
    return tuple(str(v) for v in names), {
        "source": "scwbd.anatomy.FamilyPartition (agent C)",
        "per_family": meta,
        "declared_absent": dict(getattr(part, "declared_absent", {}) or {}),
        "separation_evidence": dict(getattr(part, "separation_evidence", {}) or {}),
    }


def _partition_of_families(obj: Any) -> set[frozenset[int]]:
    """Group membership of a family container, names ignored, or empty if unreadable."""
    fams = getattr(obj, "families", None)
    if fams is None:
        fams = obj
    out: set[frozenset[int]] = set()
    try:
        for f in fams:
            members = getattr(f, "regions", None) or getattr(f, "parcels", None) or ()
            out.add(frozenset(int(i) for i in members))
    except TypeError:
        return set()
    return {g for g in out if g}


def _per_parcel_family_labels(anat: Any) -> list[str] | None:
    """The per-parcel family declaration as a list of names, or ``None``.

    Only the *name-sequence* forms; the ``family_id`` + ``family_names`` form is
    validated by the main body, which must keep owning its refusals.
    """
    for attr in ("family", "families", "family_name"):
        v = getattr(anat, attr, None)
        if v is None:
            continue
        try:
            seq = list(v)
        except TypeError:
            continue
        if len(seq) == int(anat.n_regions) and all(isinstance(x, str) for x in seq):
            return [str(x) for x in seq]
    return None


def _partition_of(labels: "Sequence[str]") -> set[frozenset[int]]:
    """Group membership as a set of index sets -- names ignored, structure kept."""
    groups: dict[str, set[int]] = {}
    for i, name in enumerate(labels):
        groups.setdefault(str(name), set()).add(i)
    return {frozenset(v) for v in groups.values()}


def _region_family_name(part: Any, region: int) -> str:
    for f in part:
        if region in set(getattr(f, "regions", ()) or getattr(f, "parcels", ()) or ()):
            return str(getattr(f, "name", getattr(f, "family_id", "?")))
    return "<unassigned>"


def _declared_families(anat) -> tuple[tuple[str, ...], dict[str, Any]] | None:
    """Read a family partition **declared by the anatomy prior**, if it has one.

    Prefers :func:`_from_anatomy_partition`.  The flat forms below are the
    interface this module originally specified, kept for tests and for any prior
    that supplies labels without agent C's object:

    * a per-parcel family name — ``family`` / ``families`` / ``family_name`` as a
      length-``N`` sequence of strings, **or** ``family_id`` (length ``N``
      integers) together with ``family_names`` (the id -> name table);
    * optionally ``family_provenance``: ``{family_name: {...}}`` recording what
      evidence separated it.

    Whatever the prior declares is what is used, however many families that is.
    :func:`derive_families` only falls back to deriving a partition from
    ``division`` / labels / ``system`` when **no** declaration is present, and it
    records which of the two happened.

    A *partial* declaration — ``family_id`` with no ``family_names``, or a name
    list of the wrong length — raises rather than being ignored: silently
    falling through to the derived partition is how a declared partition stops
    being the one in force.
    """
    from_c = _from_anatomy_partition(anat)
    if from_c is not None:
        # Two declaration channels exist: the structured `anat.families` that
        # `_from_anatomy_partition` reads, and the per-parcel labels
        # (`family` / `families` / `family_name`, or `family_id` + `family_names`)
        # parsed below. This short-circuit made the first win UNCONDITIONALLY,
        # so the second was unreachable for any prior carrying `families` -- and
        # the result was still stamped `source="anatomy_declared"`, reporting a
        # different declaration than the one supplied under a provenance that
        # says "declared".
        #
        # Measured: setting a two-family per-parcel declaration on the real prior
        # returned the prior's own nine families, labelled anatomy_declared, with
        # nothing raised. The partial-declaration refusal below could not fire
        # either, because control never reached it.
        #
        # Precedence is not the fix -- agreement is. If both channels are present
        # they must describe the same partition, and if they do not, neither is
        # safe to pick.
        labels = _per_parcel_family_labels(anat)
        if labels is not None:
            declared = _partition_of(labels)
            structured = _partition_of(
                [_region_family_name(from_c, i) for i in range(int(anat.n_regions))]
            )
            if declared != structured:
                raise ValueError(
                    "the anatomy declares its families TWICE and the two disagree.\n"
                    f"  anat.families    -> {len(structured)} groups\n"
                    f"  per-parcel labels -> {len(declared)} groups\n"
                    "Refusing to pick one. The structured declaration used to win "
                    "silently while the result was still reported as "
                    "`anatomy_declared`, so a caller supplying a per-parcel "
                    "partition got somebody else's and was told it was theirs. "
                    "Remove whichever declaration is stale (ARCHITECTURE.md O-7: "
                    "one region ontology)."
                )
        return from_c

    n = int(anat.n_regions)
    prov = dict(getattr(anat, "family_provenance", None) or {})

    # `scwbd.anatomy.families.FamilyPartition` -- the real prior's declaration.
    # It is a partition OBJECT, not a per-parcel label array: iterating it yields
    # one `RegionFamily` per family, so the length check below sees 9 (or however
    # many families the atlas has) against 414 parcels and reads a complete
    # declaration as a partial one. It carries exactly what is needed --
    # `family_index()` is `(n_regions,)` int and validates exhaustive-and-disjoint
    # before returning -- so it is adapted here rather than reformatted upstream.
    for attr in ("families", "family_partition", "family"):
        p = getattr(anat, attr, None)
        if p is None or not (hasattr(p, "family_index") and hasattr(p, "family_ids")):
            continue
        ids = tuple(str(x) for x in p.family_ids)
        idx = p.family_index()  # validates; raises if not a partition of n
        if len(idx) != n:
            raise ValueError(
                f"AnatomyPrior.{attr}.family_index() has length {len(idx)} for {n} parcels"
            )
        names = tuple(ids[int(i)] for i in idx)
        meta = {
            "source": f"AnatomyPrior.{attr} (FamilyPartition)",
            "per_family": prov or dict(getattr(p, "separation_evidence", {}) or {}),
            "atlas": getattr(p, "atlas", None),
            "partition_provenance": getattr(p, "provenance", None),
            "declared_absent": dict(getattr(p, "declared_absent", {}) or {}),
            "is_biological": bool(p.is_biological()) if hasattr(p, "is_biological") else None,
        }
        return names, meta

    for attr in ("family", "families", "family_name", "family_names_per_parcel"):
        v = getattr(anat, attr, None)
        if v is None:
            continue
        try:
            names = tuple(str(x) for x in v)
        except TypeError:
            continue
        if len(names) != n:
            raise ValueError(
                f"AnatomyPrior.{attr} declares {len(names)} family labels for {n} parcels. A "
                "partial family declaration is not usable and will not be silently replaced by "
                "the derived partition."
            )
        return names, {"source": f"AnatomyPrior.{attr}", "per_family": prov}

    fid = getattr(anat, "family_id", None)
    if fid is not None:
        table = getattr(anat, "family_names", None)
        if table is None:
            raise ValueError(
                "AnatomyPrior declares family_id but no family_names table; the integer ids are "
                "not interpretable and would be turned into fabricated names."
            )
        ids = fid.tolist() if isinstance(fid, Tensor) else list(fid)
        table = [str(x) for x in table]
        if len(ids) != n:
            raise ValueError(f"AnatomyPrior.family_id has length {len(ids)} for {n} parcels")
        bad = [i for i in ids if not (0 <= int(i) < len(table))]
        if bad:
            raise ValueError(f"AnatomyPrior.family_id contains ids outside family_names: {sorted(set(bad))[:5]}")
        return tuple(table[int(i)] for i in ids), {
            "source": "AnatomyPrior.family_id + family_names",
            "per_family": prov,
        }
    return None


def _cortical_key(label: str, system: int) -> tuple[str, str]:
    """(family name, discriminator) for a cortical parcel: **one** cortical family.

    This used to split the cortex by Yeo-7 network.  🧠 Cajal tested exactly that
    under a Váša spin null on a measured regional profile: it separates only 6 of
    21 pairs (``SomMot vs Vis`` at q=0.49/0.78), so it is not a partition.  The
    evidence supports unimodal/association and nothing finer, and that split is a
    *measurement* which belongs in the anatomy prior, not a string match on a
    parcel label which belongs nowhere.

    Emitting the rejected split here would have been the worse of the two
    failures: nothing downstream can tell a fabricated partition from a measured
    one, so every per-family operator would have bound to it silently.  So the
    fallback now emits the coarsest thing that is certainly true — the cortex is
    the cortex — and labels it.  It produces an artifact rather than refusing to
    (ARCHITECTURE.md §7a), and the artifact does not claim a structure nobody
    measured.
    """
    return "cortex", "AnatomyPrior.division == 'cortex' (single undifferentiated cortical family)"


def _subcortical_key(label: str) -> tuple[str, str]:
    """(family name, discriminator) for a subcortical parcel."""
    tok = re.sub(r"[^a-z]", "", label.lower())
    # strip a leading hemisphere letter: "Lhippo" -> "hippo"
    for cand in (tok, tok[1:] if tok[:1] in ("l", "r") else tok):
        for key, fam in _SUBCORTICAL_TOKENS.items():
            if cand.startswith(key) or cand.endswith(key):
                return fam, "structure token in the parcel label (Tian/aseg)"
    # NOT merged into a neighbour: an unparsed parcel is made visible.
    return "subcortex_unassigned", "UNPARSED label — no structure token matched"


def _provenance_key(anat) -> str:
    """A short provenance string for the partition record.

    ``AnatomyPrior.provenance`` is annotated ``str`` but the real
    ``scwbd.anatomy.BrainPrior`` supplies a nested dict of atlas/source/licence
    records (several kB).  Stringifying it wholesale would put the whole licence
    table into every family record, so the atlas identity is extracted and the
    full object is left where it already lives, in ``AnatomyPrior.summary()``.
    """
    p = getattr(anat, "provenance", "unknown")
    if isinstance(p, Mapping):
        atlas = p.get("atlas") or p.get("name") or "unknown_atlas"
        space = p.get("space")
        return f"{atlas}" + (f"/{space}" if space else "")
    return str(p)


@dataclass
class FamilyPartition:
    """The derived partition plus everything needed to attack it."""

    families: tuple[RegionFamily, ...]
    n_regions: int
    #: taxonomy names that the prior contained **zero** regions for
    unpopulated: tuple[str, ...]
    provenance: str
    #: ``"anatomy_declared"`` when the prior declared the partition itself,
    #: ``"derived_by_foundation"`` when this module had to infer it.  A model
    #: card that does not say which of the two happened is not saying where its
    #: operator assignment came from.
    source: str = "derived_by_foundation"
    notes: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.families)

    def __iter__(self) -> Iterator[RegionFamily]:
        return iter(self.families)

    def by_name(self, name: str) -> RegionFamily:
        for f in self.families:
            if f.name == name:
                return f
        raise KeyError(f"no family {name!r}; have {[f.name for f in self.families]}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_families": len(self.families),
            "n_regions": self.n_regions,
            "unpopulated": list(self.unpopulated),
            "partition_source": self.source,
            "provenance": self.provenance,
            "notes": list(self.notes),
            "families": [f.as_dict() for f in self.families],
        }


def derive_families(
    anat,
    *,
    cores: Mapping[str, str] | None = None,
    default_core: str = "learned",
    n_spectral_modes: int = 8,
    n_adaptation: int = 2,
    n_uncertainty: int = 4,
    d_key: int = 16,
    d_value: int = 16,
    d_grid: int = 12,
    d_context: int = 4,
    d_prediction: int = 8,
    # DEFAULT FALSE. Deriving a partition means using the Yeo-7 cortical split,
    # which separates 6 of 21 pairs under a Vasa spin null -- rejected on the
    # evidence, not merely approximate. A caller who wants it must say so.
    #
    # Checked before flipping: the only production caller of THIS function is
    # `model.py`, which already passes
    # `allow_derived=cfg.family_allow_derived_partition` (False by default).
    # `anatomy.py`'s bare `derive_families(obj)` is a different function of the
    # same name from `scwbd.anatomy.families` -- itself an instance of O-7, and
    # the reason this looked like a risky change until the import was read.
    allow_derived: bool = False,
) -> FamilyPartition:
    """Partition the prior's regions into families **using only what it declares or distinguishes**.

    **Preferred path — the prior declares the partition.**  If ``anat`` exposes
    a family declaration (see :func:`_declared_families`), that partition is
    used verbatim, for however many families it contains.  Nothing here caps,
    merges or renames it.  A declared family whose name matches no engineered
    backend in this repository gets the cortical component list and the generic
    core, and is listed in :attr:`FamilyPartition.notes` as untyped.

    **Fallback — derive it.**  When the prior declares nothing, the partition is
    inferred from what ``load_anatomy()`` does distinguish, in this order:

    1. ``AnatomyPrior.division`` — cortex / subcortex / cerebellum.
    2. cortical parcels: the Yeo network token in the label (``7Networks_LH_Vis_1``
       -> ``cortex_vis``), falling back to ``AnatomyPrior.system``.
    3. subcortical parcels: the structure token in the label (``Lhippo`` ->
       ``hippocampus``).  A label that does not parse lands in
       ``subcortex_unassigned`` — it is **not** folded into a neighbour.

    Nothing here invents a family.  A taxonomy entry the prior has no regions for
    is returned in :attr:`FamilyPartition.unpopulated`, and which of the two
    paths ran is recorded in :attr:`FamilyPartition.source`.
    """
    cores = dict(cores or {})
    labels = tuple(anat.labels)
    division = tuple(anat.division)
    system = anat.system.tolist() if isinstance(anat.system, Tensor) else list(anat.system)
    n = int(anat.n_regions)
    if not (len(labels) == len(division) == len(system) == n):
        raise ValueError(
            f"anatomy prior is internally inconsistent: n_regions={n} but "
            f"len(labels)={len(labels)}, len(division)={len(division)}, len(system)={len(system)}"
        )

    members: dict[str, list[int]] = {}
    disc: dict[str, str] = {}
    kind: dict[str, str] = {}
    div_of: dict[str, str] = {}
    notes: list[str] = []

    declared = _declared_families(anat)

    # `allow_derived` was DECORATIVE: it appeared once, in this function's
    # signature, and nothing ever read it. `ModelConfig.family_allow_derived_
    # partition` defaults to False and `model.py` threads it here faithfully --
    # into a parameter that was discarded. The config option, this function's
    # docstring, and the test that asserts "the default must refuse" all
    # described a refusal that did not exist.
    #
    # What ran instead: agent C tested the Yeo-7 cortical split under a Vasa spin
    # null and it separates 6 of 21 pairs, so it is NOT a partition on the
    # evidence. That derived split was being used silently whenever an anatomy
    # declared nothing.
    #
    # Refusing here rather than at the call sites, because the guard belongs
    # where the fallback is taken -- putting it in the callers is what let
    # `anatomy.py`'s bare `derive_families(obj)` bypass it.
    if declared is None and not allow_derived:
        raise ValueError(
            "this anatomy declares no family partition, and the derived fallback is "
            "REFUSED by default.\n"
            "  The fallback splits cortex by the Yeo-7 networks, which separate 6 of "
            "21 pairs under a Vasa spin null -- an evidence-REJECTED partition, not a "
            "cheaper one.\n"
            "  Pass allow_derived=True (or set model.family_allow_derived_partition) "
            "to state that a rejected partition is acceptable for this run, and expect "
            "FamilyPartition.notes to say REJECTS."
        )

    if declared is not None:
        names, dprov = declared
        source = str(dprov["source"])
        untyped: set[str] = set()
        untyped_noncortical: set[str] = set()
        for i in range(n):
            name = names[i]
            k = _kind_from_declared_name(name)
            if k is None:
                k = "cortex"
                untyped.add(name)
                if division[i] != "cortex":
                    untyped_noncortical.add(f"{name} (division={division[i]})")
            members.setdefault(name, []).append(i)
            disc.setdefault(name, f"declared by the anatomy prior ({source})")
            kind.setdefault(name, k)
            div_of.setdefault(name, division[i])
        notes.append(
            f"family partition was DECLARED by the anatomy prior via {source}: "
            f"{len(members)} families over {n} parcels. Not derived here."
        )
        if untyped:
            notes.append(
                "declared families with no engineered backend in this repository (generic core, "
                "component list = cortical): " + ", ".join(sorted(untyped))
            )
        if untyped_noncortical:
            # Loud, because this is how §5's engineered backends go silently
            # unassigned: a subcortical or cerebellar family whose declared name
            # this module does not recognise gets the cortical component list and
            # the generic core, and every downstream report says "per-family
            # operators" while the cortex-shaped default is what actually ran.
            notes.append(
                "*** NON-CORTICAL FAMILIES WITH NO RECOGNISED BACKEND: "
                + ", ".join(sorted(untyped_noncortical))
                + ". These got the CORTICAL component list and the generic core. body.tex §5 "
                "argues these systems warrant engineered backends; none was assigned. Either add "
                "the name token to families._KIND_TOKENS or state in the manifest that the "
                "subsystem argument has no expression for them. ***"
            )
        if dprov.get("per_family"):
            notes.append(f"per-family provenance supplied by the prior for {len(dprov['per_family'])} families")
        if dprov.get("declared_absent"):
            notes.append(
                "systems the prior declares ABSENT (no parcels in this atlas, reason given by "
                "agent C): " + "; ".join(f"{k}: {v}" for k, v in dprov["declared_absent"].items())
            )
        untrained = sorted(
            k for k, v in (dprov.get("per_family") or {}).items()
            if str(v.get("training_status", "")).startswith("prior_only")
        )
        if untrained:
            notes.append(
                "families the prior marks as carrying NO regional data (narrowing `stage1-data-limited` -- "
                "initialised from the prior and declared untrained): " + ", ".join(untrained)
            )
    else:
        notes.append(
            "PARTITION DERIVED HERE, NOT DECLARED by the anatomy prior. The cortex is ONE "
            "undifferentiated family: the Yeo-7 split this fallback used to emit is rejected by "
            "agent C's Vasa spin null (6 of 21 pairs separate) and has been removed. Any claim "
            "of cortical regional heterogeneity from this partition is unsupported -- it is the "
            "§11.4 pooled-vector control for the cortex, with the subcortex separated by atlas "
            "identity. Install scwbd.anatomy for the measured partition."
        )
    for i in (range(n) if declared is None else ()):
        d = division[i]
        if d == "cortex":
            name, why = _cortical_key(labels[i], system[i])
            k = "cortex"
        elif d == "subcortex":
            name, why = _subcortical_key(labels[i])
            k = name if name in FAMILY_TAXONOMY else "cortex"
        elif d == "cerebellum":
            name, why, k = "cerebellum", "AnatomyPrior.division == 'cerebellum'", "cerebellum"
        else:
            name, why, k = f"division_{d}", f"AnatomyPrior.division == {d!r} (outside the taxonomy)", "cortex"
        members.setdefault(name, []).append(i)
        disc.setdefault(name, why)
        kind.setdefault(name, k)
        div_of.setdefault(name, d)

    builders = {
        "cortex": lambda: _cortical(
            n_spectral_modes=n_spectral_modes, n_adaptation=n_adaptation, n_uncertainty=n_uncertainty
        ),
        "thalamus": lambda: _thalamic(n_spectral_modes=n_spectral_modes, n_uncertainty=n_uncertainty),
        "basal_ganglia": lambda: _basal_ganglia(n_spectral_modes=n_spectral_modes, n_uncertainty=n_uncertainty),
        "hippocampus": lambda: _hippocampal(
            d_key=d_key, d_value=d_value, d_grid=d_grid, d_context=d_context, n_uncertainty=n_uncertainty
        ),
        "amygdala": lambda: _amygdalar(n_adaptation=n_adaptation, n_uncertainty=n_uncertainty),
        "cerebellum": lambda: _cerebellar(d_prediction=d_prediction, n_uncertainty=n_uncertainty),
    }

    fams: list[RegionFamily] = []
    for name in sorted(members):
        k = kind[name]
        comps, ports = builders[k]()
        backend = cores.get(name) or DEFAULT_FAMILY_CORES.get(k, default_core)
        bcomps: tuple[str, ...] = ()
        if backend == "learned":
            pass
        elif backend == DEFAULT_FAMILY_CORES.get(k):
            bcomps = BACKEND_COMPONENTS.get(k, ())
        else:
            # A family assigned some *other* mechanistic backend (the per-family
            # generalisation of the old single ``local_core`` string) gets that
            # backend's native state as an explicit ``mech`` component, exactly
            # as SCWBD.build_layout did globally.
            from .backends import resolve_backend

            d = int(resolve_backend(backend).state_dim)
            comps = comps + (
                ComponentSpec("mech", d, "backend_native", "fast", False, True, f"{backend} native state"),
            )
            bcomps = ("mech",)
        if name == "subcortex_unassigned":
            notes.append(
                f"{len(members[name])} subcortical parcel(s) did not match any structure token and are "
                f"in 'subcortex_unassigned' with the generic core: "
                f"{[labels[i] for i in members[name]][:8]}"
            )
        fams.append(
            RegionFamily(
                name=name,
                layout=StateLayout(comps),
                backend=backend,
                ports=ports,
                division=div_of[name],
                discriminator=disc[name],
                rationale=_RATIONALE.get(k, _RATIONALE["cortex"]),
                regions=tuple(members[name]),
                backend_components=bcomps,
            )
        )

    present_kinds = {kind[f.name] for f in fams}
    unpopulated = tuple(sorted(set(FAMILY_TAXONOMY) - present_kinds))
    if unpopulated:
        notes.append(
            "taxonomy families with ZERO regions in this prior (declared, not fabricated): "
            + ", ".join(unpopulated)
        )
    unknown = set(cores) - {f.name for f in fams}
    if unknown:
        raise KeyError(
            f"config assigns backends to families that this anatomy prior does not produce: "
            f"{sorted(unknown)}; derived families are {sorted(f.name for f in fams)}. "
            "A backend assignment to a nonexistent family is a silent no-op, which is how a "
            "declared operator assignment stops being true."
        )
    return FamilyPartition(
        families=tuple(fams),
        n_regions=n,
        unpopulated=unpopulated,
        provenance=_provenance_key(anat),
        source="anatomy_declared" if declared is not None else "derived_by_foundation",
        notes=tuple(
            notes
            + (
                []
                if declared is not None
                else [
                    "REJECTS: this partition was DERIVED, not declared. The cortical "
                    "split is Yeo-7, which separates 6 of 21 pairs under a Vasa spin "
                    "null -- rejected on the evidence. The run opted in via "
                    "allow_derived=True; recorded here so the artifact carries the "
                    "evidence status rather than the caller having to remember it."
                ]
            )
        ),
    )


# ======================================================================
# the padded layout, with the span enforced (narrowing `padded-family-state`)
# ======================================================================
class FamilyStateLayout:
    """Padded ``(..., N, D)`` state with per-family spans ``[0, d_f)``, enforced.

    ``D = max_f d_f``.  Region ``i`` belongs to exactly one family, and only
    channels ``[0, d_f)`` of that region are state; the rest is **pad** and must
    remain identically zero for the padded layout to be observationally
    equivalent to the ragged one `padded-family-state` gave up.
    """

    def __init__(self, partition: FamilyPartition, *, device: str | torch.device = "cpu") -> None:
        self.partition = partition
        self.families = partition.families
        self.n_regions = partition.n_regions
        if not self.families:
            raise ValueError("empty partition: a FamilyStateLayout needs at least one family")
        self.dim = max(f.dim for f in self.families)
        dev = torch.device(device)

        owner = torch.full((self.n_regions,), -1, dtype=torch.long, device=dev)
        span_end = torch.zeros(self.n_regions, dtype=torch.long, device=dev)
        index: dict[str, Tensor] = {}
        for fi, f in enumerate(self.families):
            idx = torch.as_tensor(f.regions, dtype=torch.long, device=dev)
            if idx.numel():
                if int((owner[idx] >= 0).sum()) > 0:
                    clash = idx[owner[idx] >= 0][:5].tolist()
                    raise SpanViolation(
                        f"family {f.name!r} claims region(s) {clash} already owned by another family; "
                        "families must partition the regions, not overlap."
                    )
                owner[idx] = fi
                span_end[idx] = f.dim
            index[f.name] = idx
        if int((owner < 0).sum()) > 0:
            orphan = (owner < 0).nonzero().reshape(-1)[:5].tolist()
            raise SpanViolation(
                f"{int((owner < 0).sum())} region(s) belong to no family, e.g. {orphan}. "
                "An unowned region has no declared span, so nothing can be enforced about it."
            )
        self._owner = owner
        self._span_end = span_end
        self._index = index
        chan = torch.arange(self.dim, device=dev).reshape(1, -1)
        self._in_span = chan < span_end.reshape(-1, 1)  # (N, D) bool
        self.device = dev
        # family-contiguous ordering, so a per-family result can be reassembled
        # with ONE concatenation + ONE gather instead of N_families scatters.
        perm = torch.cat([index[f.name] for f in self.families]) if self.families else torch.empty(0, dtype=torch.long)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel(), device=dev)
        self._perm, self._inv = perm, inv

    # -- geometry ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.families)

    def __iter__(self) -> Iterator[RegionFamily]:
        return iter(self.families)

    def __contains__(self, name: str) -> bool:
        return name in self._index

    def family(self, name: str) -> RegionFamily:
        return self.partition.by_name(name)

    def index(self, name: str, device: "torch.device | str | None" = None) -> Tensor:
        """Parcel indices for ``name``, on ``device`` if given.

        ``FamilyStateLayout`` is a plain object, not an ``nn.Module``, so its
        index tensors are **not** moved by ``model.cuda()``.  Every consumer
        that indexes a CUDA tensor must therefore ask for the device, and the
        result is cached per device so a rollout does not re-upload the same
        small index on every step.
        """
        if name not in self._index:
            raise KeyError(f"no family {name!r}; have {sorted(self._index)}")
        idx = self._index[name]
        if device is None or torch.device(device) == idx.device:
            return idx
        key = (name, str(torch.device(device)))
        cache = self.__dict__.setdefault("_index_device_cache", {})
        if key not in cache:
            cache[key] = idx.to(device)
        return cache[key]

    def span(self, name: str) -> tuple[int, int]:
        """The family's declared channel span ``[0, d_f)``."""
        return 0, self.family(name).dim

    def in_span_mask(self, *, dtype=torch.bool) -> Tensor:
        """``(N, D)`` — True/1 where a channel is real state."""
        return self._in_span.to(dtype)

    def pad_mask(self, *, dtype=torch.bool) -> Tensor:
        """``(N, D)`` — True/1 where a channel is pad and must stay zero."""
        return (~self._in_span).to(dtype)

    def to(self, device) -> "FamilyStateLayout":
        return FamilyStateLayout(self.partition, device=device)

    def padding_fraction(self) -> float:
        """Share of the ``(N, D)`` state plane that is pad.

        The price of `padded-family-state`.  Reported, not hidden: if it is large, the ragged
        layout is the better engineering answer and `padded-family-state` should be revisited.
        """
        real = float(self._in_span.sum())
        return 1.0 - real / float(self.n_regions * self.dim)

    # -- enforced access --------------------------------------------------
    def channels(self, name: str, lo: int, hi: int) -> slice:
        """A raw channel range **checked against the family's span**."""
        d = self.family(name).dim
        if lo < 0 or hi > d or lo >= hi:
            raise SpanViolation(
                f"family {name!r} declares span [0, {d}) but asked for channels [{lo}, {hi}). "
                "Out-of-span channels are pad, not zeros with meaning: reading them would make "
                "the padded layout (narrowing `padded-family-state`) observationally different from the ragged one."
            )
        return slice(lo, hi)

    def component_slice(self, name: str, component: str) -> slice:
        f = self.family(name)
        if component not in f.layout:
            raise SpanViolation(
                f"family {name!r} does not declare component {component!r}; it has "
                f"{[c.name for c in f.layout.components]}. body.tex §2.1 indexes the state "
                "*space* by region — a component another family owns is not zero here, it does "
                "not exist here."
            )
        a, b = f.layout.span(component)
        return self.channels(name, a, b)

    def gather(self, x: Tensor, name: str) -> Tensor:
        """``(..., N, D) -> (..., n_f, d_f)`` — the family's own state, span-clipped."""
        idx = self.index(name, x.device)
        d = self.family(name).dim
        return x.index_select(-2, idx)[..., :d]

    def get(self, x: Tensor, name: str, component: str) -> Tensor:
        """``(..., N, D) -> (..., n_f, dim(component))`` — the only sanctioned read."""
        s = self.component_slice(name, component)
        return x.index_select(-2, self.index(name, x.device))[..., s]

    def port(self, x: Tensor, name: str, port: str) -> Tensor:
        """Read a family's declared **out**-port.  Private components are unreachable."""
        f = self.family(name)
        p = f.port(port)
        if p.direction != "out":
            raise PortMismatch(f"port {port!r} of family {name!r} is an in-port; it cannot be read from")
        return torch.cat([self.get(x, name, c) for c in p.components], dim=-1)

    def scatter(self, x: Tensor, name: str, value: Tensor) -> Tensor:
        """Write ``(..., n_f, d_f)`` back into the family's span.  Out-of-place."""
        f = self.family(name)
        idx = self.index(name, x.device)
        if value.shape[-1] != f.dim:
            raise SpanViolation(
                f"family {name!r} declares span [0, {f.dim}) but the value to write is "
                f"{value.shape[-1]} channels wide. Writing past the span corrupts a channel that "
                "another region's family may declare with different units."
            )
        if value.shape[-2] != idx.numel():
            raise SpanViolation(
                f"family {name!r} owns {idx.numel()} region(s) but the value covers {value.shape[-2]}"
            )
        out = x.clone()
        pad = x.shape[-1] - f.dim
        if pad:
            value = torch.cat([value, value.new_zeros(*value.shape[:-1], pad)], dim=-1)
        out.index_copy_(-2, idx, value.to(out.dtype))
        return out

    def assemble(self, chunks: Sequence[Tensor]) -> Tensor:
        """Reassemble per-family blocks (in :attr:`families` order) into ``(..., N, C)``.

        Every conformant producer of a full-brain tensor goes through here.  The
        chunks are concatenated in family-contiguous order and gathered back to
        the prior's region order, so a family that emits ``d_f`` channels cannot
        land a value in another family's channels: there is no index arithmetic
        for it to get wrong.
        """
        if len(chunks) != len(self.families):
            raise SpanViolation(
                f"assemble() got {len(chunks)} chunk(s) for {len(self.families)} families; every "
                "family must contribute, including the ones whose operator produced zeros"
            )
        for f, c in zip(self.families, chunks):
            if c.shape[-2] != f.n_regions:
                raise SpanViolation(
                    f"family {f.name!r} owns {f.n_regions} region(s) but its chunk covers {c.shape[-2]}"
                )
        return torch.cat(list(chunks), dim=-2).index_select(-2, self._inv.to(chunks[0].device))

    def split(self, x: Tensor) -> list[Tensor]:
        """Inverse of :meth:`assemble` (span-clipped): ``(..., N, D) -> [(..., n_f, d_f)]``."""
        return [self.gather(x, f.name) for f in self.families]

    # -- the guard that justifies `padded-family-state` -------------------------------------
    def assert_clean(self, x: Tensor, *, where: str = "", atol: float = 0.0) -> None:
        """Raise if anything ever wrote outside a declared span.

        This is the check that makes the padded layout a *narrowing* rather than
        a defect.  It fires on any code path that treats the state as a dense
        ``(N, D)`` block — a full-width residual, a full-width readout, an
        ``x + dx`` where ``dx`` was produced without the family mask.

        ``atol=0.0`` on purpose: the pad is never written by a conformant path,
        so it is *exactly* zero, and a tolerance would be a place for a real
        violation to hide.
        """
        if x.shape[-1] != self.dim or x.shape[-2] != self.n_regions:
            raise SpanViolation(
                f"state has shape {tuple(x.shape)}; this layout is (..., {self.n_regions}, {self.dim})"
            )
        pad = (~self._in_span).to(x.device)
        if not bool(pad.any()):
            return  # every family is at max dim: there is no pad to violate
        mag = x.detach().masked_fill(~pad, 0).abs()
        if float(mag.max()) <= atol:
            return
        # name the offender: which family, which channel, how big
        flat = mag.reshape(-1, self.n_regions, self.dim).amax(0)  # (N, D)
        region = int(flat.amax(-1).argmax())
        chan = int(flat[region].argmax())
        fam = self.families[int(self._owner[region])]
        raise SpanViolation(
            f"out-of-span write detected{(' at ' + where) if where else ''}: region {region} belongs to "
            f"family {fam.name!r} with span [0, {fam.dim}) but channel {chan} holds "
            f"{float(flat[region, chan]):.3e}. {int((mag > atol).sum())} pad element(s) are non-zero. "
            "Narrowing `padded-family-state` permits padded storage ONLY while this cannot happen: a family that can "
            "write past its span is writing into a channel another family declares with different "
            "units. Fix the operator to emit d_f channels, or drop `padded-family-state` for the ragged layout."
        )

    def zero_pad(self, x: Tensor) -> Tensor:
        """Re-zero the pad.  **Construction only.**

        Deliberately not called inside the step loop: masking every step would
        make :meth:`assert_clean` incapable of firing, which is the failure mode
        ``reports/decorative_guards.md`` catalogues.
        """
        return x * self._in_span.to(device=x.device, dtype=x.dtype)

    # -- ports across families --------------------------------------------
    def check_ports(self) -> list[dict[str, Any]]:
        """Match every out-port to the in-ports that accept it, by name and units.

        Returns the routing table.  Raises :class:`PortMismatch` when two
        families use the same port name with different units — the type error a
        raw state slice cannot even express.
        """
        by_name: dict[str, list[tuple[str, Port]]] = {}
        for f in self.families:
            for p in f.ports:
                by_name.setdefault(p.name, []).append((f.name, p))
        table: list[dict[str, Any]] = []
        for pname, entries in sorted(by_name.items()):
            units = {p.units for _, p in entries}
            if len(units) > 1:
                who = {fam: p.units for fam, p in entries}
                raise PortMismatch(
                    f"port {pname!r} is declared with conflicting units across families: {who}. "
                    "A port is a typed interface; the same name with two units is the semantic "
                    "collapse §2.1 forbids."
                )
            table.append(
                {
                    "port": pname,
                    "units": entries[0][1].units,
                    "sources": sorted(f for f, p in entries if p.direction == "out"),
                    "sinks": sorted(f for f, p in entries if p.direction == "in"),
                }
            )
        return table

    def routing_table(self) -> list[dict[str, Any]]:
        """Which out-ports may feed which in-ports, matched on **units**.

        This is the object that replaces "read region j's state vector": a
        message leaves a family through a declared out-port and enters another
        through a declared in-port of the same physical type.  An in-port with no
        compatible source is reported with ``sources: []`` rather than quietly
        receiving zeros — a dangling interface is a finding, not a default.
        """
        outs = [(f.name, p) for f in self.families for p in f.out_ports()]
        rows: list[dict[str, Any]] = []
        for f in self.families:
            for p in f.in_ports():
                src = sorted({fn for fn, op in outs if op.units == p.units and fn != f.name})
                rows.append(
                    {
                        "sink_family": f.name,
                        "in_port": p.name,
                        "units": p.units,
                        "width": f.port_dim(p.name),
                        "sources": src,
                        "dangling": not src,
                    }
                )
        return rows

    def dangling_ports(self) -> list[str]:
        return [f"{r['sink_family']}.{r['in_port']}" for r in self.routing_table() if r["dangling"]]

    # -- provenance -------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {
            "layout": "family_padded",
            "narrowing": "padded-family-state",
            "dim": self.dim,
            "n_regions": self.n_regions,
            "padding_fraction": round(self.padding_fraction(), 4),
            "span_enforced": True,
            "partition": self.partition.as_dict(),
            "port_table": self.check_ports(),
        }

    def summary_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "family": f.name,
                "n_regions": f.n_regions,
                "dim": f.dim,
                "backend": f.backend,
                "components": [c.name for c in f.layout.components],
                "out_ports": [p.name for p in f.out_ports()],
                "in_ports": [p.name for p in f.in_ports()],
                "discriminator": f.discriminator,
            }
            for f in self.families
        ]
