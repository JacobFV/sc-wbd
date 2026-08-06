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

Layout (declared narrowing **N-1**)
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
    """A family touched state outside its declared span (narrowing **N-1**).

    N-1 permits padded storage *only* because this is enforceable.  If this
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
    )
    ports = _SHARED_PORTS + (
        Port("oscillatory", ("spectral",), "dimensionless", "out", "quadrature modes carried by long-range edges"),
        Port("afferent", ("rate_e",), "Hz", "in", "excitatory drive arriving from other families"),
        Port("modulatory", ("adaptation",), "dimensionless", "in", "gain/adaptation control from subcortical families"),
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
_KIND_TOKENS: tuple[tuple[str, str], ...] = (
    ("hippocamp", "hippocampus"),
    ("subiculum", "hippocampus"),
    ("entorhinal", "hippocampus"),
    ("amygdal", "amygdala"),
    ("thalam", "thalamus"),
    ("pulvinar", "thalamus"),
    ("basal_ganglia", "basal_ganglia"),
    ("striat", "basal_ganglia"),
    ("pallid", "basal_ganglia"),
    ("caudate", "basal_ganglia"),
    ("putamen", "basal_ganglia"),
    ("accumb", "basal_ganglia"),
    ("cerebell", "cerebellum"),
)


def _kind_from_declared_name(name: str) -> str | None:
    low = name.lower()
    for tok, kind in _KIND_TOKENS:
        if tok in low:
            return kind
    return None


def _declared_families(anat) -> tuple[tuple[str, ...], dict[str, Any]] | None:
    """Read a family partition **declared by the anatomy prior**, if it has one.

    This is the interface ``scwbd.foundation`` needs from ``scwbd.anatomy``:

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
    n = int(anat.n_regions)
    prov = dict(getattr(anat, "family_provenance", None) or {})

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


_YEO = re.compile(r"^\d*Networks?_(LH|RH)_([A-Za-z]+)")


def _cortical_key(label: str, system: int) -> tuple[str, str]:
    """(family name, discriminator) for a cortical parcel."""
    m = _YEO.match(label)
    if m:
        return f"cortex_{m.group(2).lower()}", "Yeo network token in the parcel label"
    return f"cortex_system{int(system):02d}", "AnatomyPrior.system integer (labels carry no network token)"


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
    if declared is not None:
        names, dprov = declared
        source = str(dprov["source"])
        untyped: set[str] = set()
        for i in range(n):
            name = names[i]
            k = _kind_from_declared_name(name)
            if k is None:
                k = "cortex"
                untyped.add(name)
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
        if dprov.get("per_family"):
            notes.append(f"per-family provenance supplied by the prior for {len(dprov['per_family'])} families")
    else:
        notes.append(
            "anatomy prior declares NO family partition (no `family`/`family_id` attribute); the "
            "partition below was derived here from division + parcel labels + system id. Replace "
            "this path as soon as the prior declares one."
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
        notes=tuple(notes),
    )


# ======================================================================
# the padded layout, with the span enforced (narrowing N-1)
# ======================================================================
class FamilyStateLayout:
    """Padded ``(..., N, D)`` state with per-family spans ``[0, d_f)``, enforced.

    ``D = max_f d_f``.  Region ``i`` belongs to exactly one family, and only
    channels ``[0, d_f)`` of that region are state; the rest is **pad** and must
    remain identically zero for the padded layout to be observationally
    equivalent to the ragged one N-1 gave up.
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

    def index(self, name: str) -> Tensor:
        if name not in self._index:
            raise KeyError(f"no family {name!r}; have {sorted(self._index)}")
        return self._index[name]

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

        The price of N-1.  Reported, not hidden: if it is large, the ragged
        layout is the better engineering answer and N-1 should be revisited.
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
                "the padded layout (narrowing N-1) observationally different from the ragged one."
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
        idx = self.index(name)
        d = self.family(name).dim
        return x.index_select(-2, idx)[..., :d]

    def get(self, x: Tensor, name: str, component: str) -> Tensor:
        """``(..., N, D) -> (..., n_f, dim(component))`` — the only sanctioned read."""
        s = self.component_slice(name, component)
        return x.index_select(-2, self.index(name))[..., s]

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
        idx = self.index(name)
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

    # -- the guard that justifies N-1 -------------------------------------
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
            "Narrowing N-1 permits padded storage ONLY while this cannot happen: a family that can "
            "write past its span is writing into a channel another family declares with different "
            "units. Fix the operator to emit d_f channels, or drop N-1 for the ragged layout."
        )

    def zero_pad(self, x: Tensor) -> Tensor:
        """Re-zero the pad.  **Construction only.**

        Deliberately not called inside the step loop: masking every step would
        make :meth:`assert_clean` incapable of firing, which is the failure mode
        ``reports/decorative_guards.md`` catalogues.
        """
        return x * self._in_span.to(x.dtype)

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
            "narrowing": "N-1",
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
