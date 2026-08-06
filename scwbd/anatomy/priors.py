"""``BrainPrior`` -- the assembled adult anatomical prior.

This is what agent E (dynamics) and agent I (foundation model) consume::

    prior = BrainPrior.load("Schaefer400x7", include_subcortex=True)
    prior.structural.mask("soft")        # which operators may exist
    prior.delay_prior_ms()               # delay per edge, as a distribution
    prior.ei_ratio_prior()               # per-parcel E/I, as distributions
    prior.timescale_prior()              # per-parcel tau, as distributions

Regional heterogeneity is the point
-----------------------------------
ARCHITECTURE.md §5 and thesis §6.1 name the failure mode explicitly: *one
neural mass per parcel, identical everywhere, erasing regional phenotype*.  So
every per-parcel quantity here is returned as a list of
:class:`~scwbd.schema.Prior` objects -- one per parcel, differing across
parcels, each carrying its own uncertainty -- rather than a single global
number that the fitting procedure would then have to unlearn.

What this object refuses to do
------------------------------
* It will not give a direction for a human cortico-cortical edge.
* It will not give a structural connectome for a parcellation nobody measured
  one in (cerebellum in particular).
* It will not report a per-parcel receptor density as a subject's value.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

from . import sources as S
from ._compat import (
    EVIDENCE_ORDER,
    DiracPrior,
    LogNormalPrior,
    NormalPrior,
    PriorBase,
    UncertaintyLedger,
    group_average_ledger,
)
from .atlases import Parcellation, load_parcellation
from .connectome import (
    CONDUCTION_VELOCITY_PRIOR,
    TORTUOSITY_PRIOR,
    ConductionDelayModel,
    StructuralPrior,
    load_structural_prior,
)
from .geometry import ParcelGeometry, parcel_geometry
from .maps import MapSet, RECEPTOR_GROUPS, load_maps
from .paths import derived_dir

__all__ = [
    "BrainPrior",
    "TIMESCALE_RANGE_MS",
    "EI_LOG_RANGE",
    "EI_ORDERING_SOURCES",
    "DEFAULT_EI_ORDERING",
]

#: Named sources for the **cortical ordering** the E/I prior is built from.
#:
#: Each entry is ``(map_name, orientation)`` pairs plus how they are combined.
#: ``orientation`` is ``+1`` when the map increases from sensorimotor toward
#: association cortex and ``-1`` when it runs the other way; the convention is
#: fixed a priori from Demirtas et al. (2019) Neuron 101:1181 and Wang (2020)
#: Nat Rev Neurosci 21:169 -- the same two papers :data:`EI_LOG_RANGE` is
#: calibrated against -- and **not** from any correlation measured here.  Getting
#: it wrong inverts the cortical E/I gradient end to end and still looks
#: entirely plausible, which is why it is a declared constant with a test
#: (``tests/anatomy/test_ei_ordering.py``) rather than a runtime `sign()`.
#:
#: ``licence_key`` is the ``scwbd.anatomy.sources.SRC`` entry every ingredient
#: comes from.  It is here so the choice of ordering carries its own licence
#: consequence instead of that being discoverable only by reading the maps.
#:
#: ``combine`` is ``"zscore"`` everywhere, and that is a **corrected** choice.
#: The composite was first built as a mean of rank-normalised maps, which is
#: the more robust statistic and which *ties parcels*: ranks are multiples of
#: ``1/(n-1)``, so averaging three of them collides.  68 of 400 Schaefer-400
#: parcels shared a value with another parcel (332 distinct), and 16 of 100 at
#: Schaefer-100 -- which is thesis S6.1's named failure mode, parcels made
#: identical, arriving through the fix rather than the thing being fixed.  It
#: was caught by ``test_ei_priors_actually_differ_across_parcels``, an
#: invariant that predates this substitution.  Averaging z-scores is tie-free
#: (400/400 distinct), agrees with the receptor ordering indistinguishably
#: (Spearman +0.358 vs +0.369, inside the criterion's 0.05 band) and is
#: marginally *more* cross-scale stable (+0.9613 vs +0.9592).
EI_ORDERING_SOURCES: dict[str, dict[str, Any]] = {
    "hcp_hierarchy": {
        "ingredients": (
            ("myelin_t1t2", -1),
            ("cortical_thickness", +1),
            ("intrinsic_timescale_meg", +1),
        ),
        "combine": "zscore",
        "licence_keys": ("hcps1200_maps",),
        "description": (
            "Mean of three z-scored HCP S1200 group maps, oriented so the "
            "composite increases toward association cortex. Selected by the "
            "pre-committed criterion in reports/ei_ordering_substitution.md S1."
        ),
    },
    "myelin_t1t2": {
        "ingredients": (("myelin_t1t2", -1),),
        "combine": "zscore",
        "licence_keys": ("hcps1200_maps",),
        "description": "Inverted T1w/T2w myelin z-score alone.",
    },
    "sa_axis": {
        "ingredients": (("sa_axis", +1),),
        "combine": "zscore",
        "licence_keys": ("sydnor2021",),
        "description": (
            "Sensorimotor-association axis z-score alone. NOTE: sydnor2021's licence "
            "field states 'As distributed via neuromaps', which names no terms; "
            "this source is offered but is not the default for that reason."
        ),
    },
    "hansen_receptors": {
        "ingredients": (("ei_proxy", +1),),
        "combine": "zscore",
        "licence_keys": ("hansen_receptors",),
        "description": (
            "The receptor-derived E/I contrast (NMDA + mGluR5 versus GABA-A) from "
            "the Hansen PET atlas. CC-BY-NC-SA-4.0: NON-COMMERCIAL AND SHARE-ALIKE. "
            "Opt-in only. Nothing else in this package substitutes for receptor "
            "identity, which is why it is retained rather than deleted (thesis S5 "
            "neuromodulator-specific control fields)."
        ),
    },
}

#: The ordering used when the caller does not choose one.  Permissive by
#: construction: every ingredient is an HCP S1200 group map under HCP
#: open-access data-use terms, which is the *same* regime the structural
#: connectome (``enigma_hcp_sc``) already carries, so the default path adds no
#: licence surface at all.
DEFAULT_EI_ORDERING: str = "hcp_hierarchy"

#: Plausible span of cortical intrinsic timescales, in milliseconds.
#:
#: Anchors: Murray et al. (2014) Nat Neurosci 17:1661 report a hierarchy of
#: single-unit autocorrelation timescales from ~50 ms in sensory to ~350 ms in
#: prefrontal macaque cortex; Gao et al. (2020) eLife 9:e61277 estimate 20-200
#: ms from human ECoG.  We map the cortical hierarchy rank onto this span in
#: log space and carry a wide per-parcel sigma, because the mapping itself is a
#: hypothesis.
TIMESCALE_RANGE_MS: tuple[float, float] = (20.0, 250.0)

#: Half-width, in natural-log units, of the per-parcel E/I ratio prior's
#: displacement from the cortical mean.  0.35 gives roughly a factor 2.0 spread
#: between the most excitatory and most inhibitory parcel, which is the order
#: of the regional variation Wang (2020, Nat Rev Neurosci 21:169) and Demirtas
#: et al. (2019, Neuron 101:1181) use when they let E/I vary along the
#: hierarchy.  It is a *modelling span*, not a measurement.
EI_LOG_RANGE: float = 0.35


@dataclass
class BrainPrior:
    """Everything the dynamics and foundation modules need about adult anatomy."""

    atlas: str
    parcellation: Parcellation
    labels: np.ndarray
    hemi: np.ndarray
    network: np.ndarray
    structure: np.ndarray
    centroids_mni: np.ndarray
    areas_mm2: np.ndarray
    volumes_mm3: np.ndarray
    geometry: ParcelGeometry | None
    structural: StructuralPrior
    maps: MapSet
    include_subcortex: bool
    include_cerebellum: bool
    cerebellum: Parcellation | None = None
    unresolved: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    # -- basics ----------------------------------------------------------
    @property
    def n_parcels(self) -> int:
        return int(self.labels.shape[0])

    @property
    def n_cortex(self) -> int:
        return int((self.structure == "cortex").sum())

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BrainPrior({self.atlas}, n={self.n_parcels} "
            f"[cortex {self.n_cortex}, subcortex {int((self.structure=='subcortex').sum())}, "
            f"cerebellum {int((self.structure=='cerebellum').sum())}], "
            f"unresolved={sorted(self.unresolved)})"
        )

    # -- loader ----------------------------------------------------------
    @classmethod
    def load(
        cls,
        atlas: str = "Schaefer400x7",
        *,
        include_subcortex: bool = True,
        include_cerebellum: bool = False,
        cerebellar_atlas: str = "Buckner17",
        space: str = "fsLR",
        density: str = "32k",
        length: str = "euclidean",
        rebuild: bool = False,
    ) -> "BrainPrior":
        """Assemble the prior for one parcellation.

        Parameters
        ----------
        include_cerebellum
            Append cerebellar parcels.  **They arrive without any structural
            connectivity**: the group connectome shipped here covers cortex and
            14 subcortical structures only, and cortico-cerebellar traffic is
            polysynaptic through pons and thalamus, which tractography does not
            resolve.  Every cerebellar edge is therefore ``absent`` and the
            reason is recorded in :attr:`unresolved`.  This is deliberate: an
            invented cerebellar connectome would be worse than none.
        """
        parc = load_parcellation(atlas, space, density)
        sp = load_structural_prior(
            atlas,
            include_subcortex=include_subcortex,
            space=space,
            density=density,
            length=length,
            rebuild=rebuild,
        )
        geom = parcel_geometry(parc)
        ms = load_maps(parc, rebuild=rebuild)

        labels = sp.labels
        n_ctx = parc.n_parcels
        if include_subcortex:
            sub = load_parcellation("Aseg14", "MNI152", "1mm")
            hemi = np.concatenate([parc.hemi, sub.hemi])
            network = np.concatenate([parc.network, sub.network])
            structure = np.concatenate([parc.structure, sub.structure])
            centroids = np.vstack([parc.centroids_mni, sub.centroids_mni])
            areas = np.concatenate([parc.areas_mm2, np.full(sub.n_parcels, np.nan)])
            volumes = np.concatenate([np.full(n_ctx, np.nan), sub.volumes_mm3])
        else:
            hemi, network = parc.hemi, parc.network
            structure, centroids = parc.structure, parc.centroids_mni
            areas, volumes = parc.areas_mm2, np.full(n_ctx, np.nan)

        unresolved: dict[str, str] = {}
        cerebellum = None
        if include_cerebellum:
            cerebellum = load_parcellation(cerebellar_atlas, "MNI152", "1mm")
            labels = np.concatenate([labels, cerebellum.labels])
            hemi = np.concatenate([hemi, cerebellum.hemi])
            network = np.concatenate([network, cerebellum.network])
            structure = np.concatenate([structure, cerebellum.structure])
            centroids = np.vstack([centroids, cerebellum.centroids_mni])
            areas = np.concatenate([areas, np.full(cerebellum.n_parcels, np.nan)])
            volumes = np.concatenate([volumes, cerebellum.volumes_mm3])
            sp = _pad_structural(sp, cerebellum, centroids)
            unresolved["cerebellar_structural_connectivity"] = (
                "No structural connectome covers the cerebellum in the sources "
                "shipped here. Cortico-cerebellar communication is polysynaptic "
                "(cortex -> pontine nuclei -> cerebellar cortex; dentate -> "
                "thalamus -> cortex) and diffusion tractography does not resolve "
                "the synaptic relays. Every cerebellar edge is `absent`, which "
                "means 'no evidence', not 'not connected'. Supplying these edges "
                "requires either a cerebellar-specific tractography protocol or "
                "an explicit `proposed`-class functional-connectivity prior, and "
                "neither is asserted here."
            )
            unresolved["cerebellar_regional_maps"] = (
                "The cortical surface maps (gradients, myelin, thickness, "
                "sensorimotor-association axis) are undefined in the cerebellum "
                "and are returned as nan rather than imputed."
            )

        if not include_subcortex:
            unresolved["subcortex_excluded"] = (
                "Subcortex was excluded by request. Thalamic and basal-ganglia "
                "relays gate cortico-cortical effective connectivity; a "
                "cortex-only model attributes their effect to direct cortical edges."
            )
        unresolved["direction"] = (
            "Direction of human cortico-cortical edges is not available. Human "
            "diffusion MRI is undirected. `StructuralPrior.hierarchy_prior` "
            "supplies an antisymmetric *tendency* derived from a cortical "
            "hierarchy proxy and informed by macaque tracer work; it is "
            "`proposed`-class, cross-species, and is not a measurement."
        )
        unresolved["laminar_termination"] = (
            "Laminar origin and termination of projections are not resolved. "
            "BigBrain gives laminar geometry in one post-mortem brain; it does "
            "not give the laminar profile of any particular projection."
        )
        if cerebellar_atlas and not include_cerebellum:
            pass

        # What the *object* carries and what the *default priors consume* are
        # different sets, and conflating them is how a licence audit goes wrong
        # in either direction. ``sources`` is the first (a superset: load_maps
        # builds every map it finds on disk, receptor maps included);
        # ``ei_ordering`` is the second.
        map_source_keys = sorted({m.source_key for m in ms.maps.values()})
        obj = cls(
            atlas=atlas,
            parcellation=parc,
            labels=labels,
            hemi=hemi,
            network=network,
            structure=structure,
            centroids_mni=centroids,
            areas_mm2=areas,
            volumes_mm3=volumes,
            geometry=geom,
            structural=sp,
            maps=ms,
            include_subcortex=include_subcortex,
            include_cerebellum=include_cerebellum,
            cerebellum=cerebellum,
            unresolved=unresolved,
            provenance={
                "atlas": atlas,
                "space": space,
                "density": density,
                "length": length,
                "sources": {
                    k: {kk: S.SRC[k][kk] for kk in ("name", "url", "license", "citation")}
                    for k in dict.fromkeys(
                        [
                            "schaefer2018" if atlas.startswith("Schaefer") else "desikan2006",
                            "enigma_hcp_sc",
                            "neuromaps",
                        ]
                        # the parcellation used for subcortex is a real input and
                        # was previously unlisted -- see reports/licence_audit.md
                        + (["harvardoxford"] if include_subcortex else [])
                        + (["buckner2011"] if include_cerebellum else [])
                        + [k for k in map_source_keys if k in S.SRC]
                    )
                },
                "sources_note": (
                    "This is what the assembled object CARRIES, which is a superset "
                    "of what any one prior CONSUMES. load_maps builds every map "
                    "present on disk, so hansen_receptors appears here whenever the "
                    "PET volumes are installed -- including when no prior reads "
                    "them. For the E/I prior specifically, read 'ei_ordering'."
                ),
            },
        )
        # Recorded on the object, from the object, so a consumer never has to
        # re-derive which sources the default E/I prior actually reads.
        obj.provenance["ei_ordering"] = obj.ei_ordering()[1]
        return obj

    # -- coupling --------------------------------------------------------
    def coupling_mask(self, min_class: str = "soft") -> np.ndarray:
        """Boolean ``(n, n)`` of edges the compiler may instantiate."""
        return self.structural.mask(min_class)

    def delay_prior_ms(self) -> ConductionDelayModel:
        """The delay model.  Sample it; do not collapse it silently."""
        return self.structural.delays

    def median_delay_ms(self) -> np.ndarray:
        """Delay matrix in milliseconds at the median velocity and tortuosity."""
        return self.structural.delays.median_delay_s() * 1e3

    def velocity_prior(self) -> PriorBase:
        return self.structural.delays.velocity_prior

    # -- regional heterogeneity ------------------------------------------
    def _cortical_slice(self) -> np.ndarray:
        return self.structure == "cortex"

    def ei_ordering(
        self, source: str | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """The cortical ordering the E/I prior is built from, and its record.

        Returns ``(z, record)``.  ``z`` is ``(n_parcels,)``, roughly unit
        variance over the parcels it covers and ``nan`` elsewhere.  ``record``
        is machine-readable provenance: which maps were consulted, which
        registry sources they came from, their licences, and — this is the part
        that matters — **what was missing**.

        The absent case writes something.  A missing ingredient produces an
        entry in ``record["missing"]`` and a ``record["degraded"] = True``, so
        an ordering built from two of three maps cannot be mistaken for one
        built from three (``reports/decorative_guards.md``, the absence
        variant).  If *nothing* is available, ``z`` is all-``nan`` and
        ``record["available"]`` is ``False`` — which the prior then reports as
        ignorance rather than as a flat cortex.

        ``record["licence_keys"]`` is the load-bearing field for
        ``scwbd.release``: it names exactly the ``sources.SRC`` entries this
        object's E/I prior depends on, for this call, and is the honest answer
        to "does the default path touch Hansen?".
        """
        name = source or DEFAULT_EI_ORDERING
        if name not in EI_ORDERING_SOURCES:
            raise KeyError(
                f"unknown E/I ordering {name!r}; have "
                f"{sorted(EI_ORDERING_SOURCES)}"
            )
        spec = EI_ORDERING_SOURCES[name]
        n = self.n_parcels
        z = np.full(n, np.nan)
        used: list[dict[str, Any]] = []
        missing: dict[str, str] = {}
        degenerate: str | None = None
        stack: list[np.ndarray] = []
        for map_name, orient in spec["ingredients"]:
            if map_name not in self.maps:
                missing[map_name] = (
                    self.maps.unavailable.get(map_name)
                    or f"{map_name!r} is not in this MapSet"
                )
                continue
            m = self.maps[map_name]
            v = m.rank() if spec["combine"] == "mean_rank" else m.zscored()
            g = np.full(n, np.nan)
            k = min(v.size, n)
            g[:k] = orient * v[:k]
            stack.append(g)
            used.append(
                {
                    "map": map_name,
                    "orientation": int(orient),
                    "source_key": m.source_key,
                    "licence": S.SRC.get(m.source_key, {}).get("license", "unknown"),
                    "n_covered": int(m.n_covered),
                }
            )
        if stack:
            import warnings as _w

            with _w.catch_warnings():
                # an all-nan parcel is expected (subcortex); nanmean says so
                # loudly and the coverage mask already carries the fact
                _w.simplefilter("ignore", RuntimeWarning)
                comb = np.nanmean(np.stack(stack), axis=0)
            fin = np.isfinite(comb)
            if int(fin.sum()) >= 2 and float(comb[fin].std(ddof=1)) > 0:
                z[fin] = (comb[fin] - comb[fin].mean()) / comb[fin].std(ddof=1)
            else:
                # A composite with no between-parcel variance cannot order
                # anything. Centring it on zero would hand the caller a
                # confidently narrow prior that is identical in every parcel --
                # thesis S6.1's failure mode wearing a coverage mask. Leave it
                # nan so every parcel falls to the wide ignorance branch, and
                # say why. (Found by mutation M7; the branch had no test.)
                degenerate = (
                    f"the '{name}' composite has no between-parcel variance "
                    f"({int(fin.sum())} finite parcels, sd="
                    f"{float(comb[fin].std(ddof=1)) if int(fin.sum()) > 1 else float('nan'):.3g}), "
                    "so it orders nothing and was not used"
                )
        record: dict[str, Any] = {
            "ordering": name,
            "is_default": name == DEFAULT_EI_ORDERING,
            "description": spec["description"],
            "combine": spec["combine"],
            "maps_used": used,
            # keys are what a licence union must consume; empty means nothing
            # was read, not that nothing applies
            "licence_keys": sorted({u["source_key"] for u in used}),
            "declared_licence_keys": list(spec["licence_keys"]),
            "missing": missing,
            "degenerate": degenerate,
            "degraded": bool(missing) and bool(used),
            "available": bool(used) and degenerate is None,
            "n_covered": int(np.isfinite(z).sum()),
            "n_parcels": n,
            "selection": (
                "reports/ei_ordering_substitution.md -- criterion pre-committed "
                "at 97086e7, measurement at the following commit"
            ),
        }
        if not record["available"]:
            record["consequence"] = (
                (degenerate + ". ") if degenerate else ""
            ) + (
                "No usable ingredient of this ordering is present, so the E/I prior "
                "carries NO regional heterogeneity: every parcel falls to the "
                "no-coverage branch. This is stated rather than silently "
                "producing a flat cortex."
            )
        return z, record

    def ei_ratio_prior(self, source: str | None = None) -> list[PriorBase]:
        """Per-parcel excitation/inhibition ratio prior.

        Cortical parcels get a log-normal centred on ``exp(EI_LOG_RANGE * z)``
        where ``z`` is a cortical **ordering** scaled to roughly unit variance,
        so the ratio is 1.0 at the cortical mean and spans about a factor of two
        across cortex.  Sigma is deliberately as large as the between-parcel
        spread: the ordering orders parcels far better than it scales them.

        Parcels the ordering does not cover -- subcortical and cerebellar --
        receive the *same* log-normal centred on 1.0 with a wider sigma.  That
        is not an imputed value: it is an explicit statement that we do not know
        their E/I balance, and it is visible as a wider prior rather than hidden
        as a filled-in number.

        Which ordering
        --------------
        ``source`` selects an entry of :data:`EI_ORDERING_SOURCES`; the default
        is :data:`DEFAULT_EI_ORDERING` (``"hcp_hierarchy"``), a composite of
        three HCP S1200 group maps under HCP open-access data-use terms.

        ``source="hansen_receptors"`` restores the receptor-derived contrast
        that used to be the only option.  It is **CC-BY-NC-SA-4.0**: choosing it
        makes the artifact non-commercial *and* share-alike, and the choice
        records itself in every parcel's provenance string and in
        :meth:`ei_ordering`'s record.  It is retained because nothing else here
        substitutes for receptor identity -- thesis S5's neuromodulator-specific
        control fields need the tracers, not a hierarchy rank.

        What the substitution cost, measured
        ------------------------------------
        The default and the Hansen ordering agree at Spearman ``rho = +0.358``
        over 400 cortical parcels (Schaefer400x7, one cached build,
        2026-08-06; ``+0.444`` over 100 parcels at Schaefer100x7).  **That is
        not a reproduction of the old ordering**; it is
        a different one, and anything previously conditioned on the receptor E/I
        pattern changes.  See ``reports/ei_ordering_substitution.md`` S2 for the
        full comparison, including the finding that ``ei_proxy`` itself is only
        weakly aligned (``rho = +0.230``) to the cortical hierarchy the
        permissive maps carry, which is why no permissive map reproduces it.
        """
        n = self.n_parcels
        out: list[PriorBase] = []
        z, rec = self.ei_ordering(source)
        lic = ", ".join(
            f"{u['source_key']} ({u['licence']})" for u in rec["maps_used"]
        ) or "none -- no ingredient available"
        cite = (
            f"E/I ordering '{rec['ordering']}': {rec['description']} "
            f"Inputs: {lic}. Span calibrated to the hierarchical E/I gradients "
            "used by Demirtas M. et al. (2019) Neuron 101:1181-1194 and "
            "Wang X.-J. (2020) Nat Rev Neurosci 21:169-178. PROXY, NOT A "
            "MEASUREMENT: none of these maps measures synaptic conductance, and "
            "a subject's E/I balance is not inferable from an atlas value."
        )
        if rec["missing"]:
            cite += (
                " DEGRADED: the following declared ingredients were absent and "
                "the ordering was built without them -- "
                + "; ".join(f"{k}: {v}" for k, v in sorted(rec["missing"].items()))
            )
        for i in range(n):
            if np.isfinite(z[i]):
                out.append(
                    LogNormalPrior(
                        mu=float(EI_LOG_RANGE * np.clip(z[i], -3.0, 3.0)),
                        sigma=EI_LOG_RANGE,
                        units="dimensionless",
                        provenance=cite,
                    )
                )
            else:
                out.append(
                    LogNormalPrior(
                        mu=0.0,
                        sigma=2.0 * EI_LOG_RANGE,
                        units="dimensionless",
                        provenance=(
                            cite
                            + " NO ORDERING COVERAGE for this parcel (subcortical or "
                            "cerebellar -- every ingredient of every available "
                            "ordering is a cortical map): the prior is centred on "
                            "the cortical mean with double the width, which states "
                            "ignorance rather than imputing a value."
                        ),
                    )
                )
        return out

    def timescale_prior(self) -> list[PriorBase]:
        """Per-parcel intrinsic timescale prior, in seconds.

        The hierarchy of intrinsic timescales is one of the better-replicated
        regional phenotypes.  We rank parcels on the best available hierarchy
        proxy -- the MEG intrinsic-timescale map where it exists, else the
        sensorimotor-association axis, else the principal FC gradient -- and map
        that rank log-linearly onto :data:`TIMESCALE_RANGE_MS`.

        The width is large (sigma 0.5 in log space, a factor ~2.7 at one sigma)
        because the rank-to-timescale mapping is an assumption, and because the
        MEG estimate itself is smoothed by an ill-posed inverse problem.
        """
        n = self.n_parcels
        rank = np.full(n, np.nan)
        used = "none"
        for name in ("intrinsic_timescale_meg", "sa_axis", "fc_gradient1"):
            if name in self.maps:
                r = self.maps[name].rank()
                k = min(r.size, n)
                rank[:k] = r[:k]
                used = name
                break
        lo, hi = TIMESCALE_RANGE_MS
        cite = (
            f"Rank on the '{used}' map mapped log-linearly onto "
            f"{lo:g}-{hi:g} ms. Anchors: Murray J.D. et al. (2014) Nat Neurosci "
            "17:1661-1663; Gao R. et al. (2020) eLife 9:e61277; Shafiei G. et al. "
            "(2022) PLoS Biol 20:e3001735. The rank-to-timescale mapping is an "
            "assumption, not a measurement."
        )
        out: list[PriorBase] = []
        for i in range(n):
            if np.isfinite(rank[i]):
                mu = math.log(lo * 1e-3) + rank[i] * (math.log(hi) - math.log(lo))
                out.append(
                    LogNormalPrior(mu=float(mu), sigma=0.5, units="s", provenance=cite)
                )
            else:
                out.append(
                    LogNormalPrior(
                        mu=float(math.log(math.sqrt(lo * hi) * 1e-3)),
                        sigma=0.9,
                        units="s",
                        provenance=(
                            cite
                            + " NO HIERARCHY MAP for this parcel (subcortical or "
                            "cerebellar): geometric-mean centre, near-doubled width."
                        ),
                    )
                )
        return out

    def receptor_profile(self) -> tuple[np.ndarray, tuple[str, ...]]:
        """``(n_parcels, n_receptors)`` z-scored densities and their names.

        ``nan`` where a parcel has no PET coverage.  Do not fill it.
        """
        names = tuple(f"receptor_{r}" for r in self.maps.receptor_names)
        if not names:
            return np.zeros((self.n_parcels, 0)), ()
        v, _ = self.maps.matrix(names)
        out = np.full((self.n_parcels, v.shape[1]), np.nan)
        out[: v.shape[0]] = v
        return out, self.maps.receptor_names

    def hierarchy_rank(self) -> np.ndarray:
        """Rank on the sensorimotor-association axis; ``nan`` off cortex."""
        for name in ("sa_axis", "fc_gradient1", "myelin_t1t2"):
            if name in self.maps:
                r = self.maps[name].rank()
                out = np.full(self.n_parcels, np.nan)
                out[: r.size] = r
                return out
        return np.full(self.n_parcels, np.nan)

    # -- G2 ---------------------------------------------------------------
    def controls(self, seed: int = 0) -> dict[str, StructuralPrior]:
        """The five G2 null connectomes.

        A claim that "anatomy helps" is only meaningful against these.  Per
        ARCHITECTURE.md §4, a gate that fails is a result: if the model matches
        ``distance_matched`` it has learned geometry, and that must be reported
        rather than tuned away.
        """
        return self.structural.controls(seed)

    # -- honesty -----------------------------------------------------------
    def what_this_cannot_support(self) -> dict[str, str]:
        """Machine-readable list of the inferences this prior does not license."""
        out = dict(self.unresolved)
        out["subject_specificity"] = (
            "Every quantity here is a group average. None of it is this "
            "subject's anatomy, connectome, receptor density or timescale."
        )
        out["receptor_density"] = (
            "A subject's receptor density is NOT inferable from an atlas value "
            "(thesis Appendix A). Values are z-scored per tracer, so even the "
            "group-level absolute density is not recoverable."
        )
        out["zero_edge_semantics"] = (
            "A zero in the coupling mask creates exact independence inside the "
            "compiled model. It does not establish biological independence: "
            "communication may occur through another path, a shared input, "
            "volume conduction, neuromodulation, or an omitted population (§2.2)."
        )
        out["weight_scale"] = (
            f"Edge weights are on the scale '{self.structural.weights_units}'. "
            "They are not synaptic strengths and not in physical units."
        )
        return out

    def ledger_summary(self) -> dict[str, dict[str, Any]]:
        """Every uncertainty ledger this prior carries, keyed by object."""
        out: dict[str, dict[str, Any]] = {
            "structural_connectome": self.structural.ledger.model_dump(mode="json"),
        }
        if self.geometry is not None:
            out["geometry"] = self.geometry.ledger.model_dump(mode="json")
        for k, m in self.maps.maps.items():
            out[f"map.{k}"] = m.ledger.model_dump(mode="json")
        return out

    def summary(self) -> dict[str, Any]:
        c = self.structural.class_counts()
        d = self.median_delay_ms()
        m = self.structural.mask("soft")
        return {
            "atlas": self.atlas,
            "n_parcels": self.n_parcels,
            "n_cortex": self.n_cortex,
            "n_subcortex": int((self.structure == "subcortex").sum()),
            "n_cerebellum": int((self.structure == "cerebellum").sum()),
            "edge_classes": c,
            "density": self.structural.density(),
            "n_maps": len(self.maps.maps),
            "n_receptors": len(self.maps.receptor_names),
            "median_delay_ms": {
                "min": float(d[m].min()) if m.any() else None,
                "median": float(np.median(d[m])) if m.any() else None,
                "max": float(d[m].max()) if m.any() else None,
            },
            "direction_known": self.structural.direction_known,
            "unresolved": sorted(self.unresolved),
        }


def _pad_structural(
    sp: StructuralPrior, cerebellum: Parcellation, centroids: np.ndarray
) -> StructuralPrior:
    """Extend a structural prior with cerebellar parcels that have no edges."""
    n_old = sp.n_parcels
    n_new = n_old + cerebellum.n_parcels

    def grow(a: np.ndarray, fill: float = 0.0) -> np.ndarray:
        out = np.full((n_new, n_new), fill, dtype=a.dtype)
        out[:n_old, :n_old] = a
        return out

    euclid = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
    np.fill_diagonal(euclid, 0.0)
    dist = grow(sp.distance_mm)
    dist[n_old:, :] = euclid[n_old:, :]
    dist[:, n_old:] = euclid[:, n_old:]
    ev = grow(sp.evidence.astype(np.uint8), 0).astype(np.uint8)
    return StructuralPrior(
        parcellation=sp.parcellation,
        labels=np.concatenate([sp.labels, cerebellum.labels]),
        weights=grow(sp.weights),
        distance_mm=dist,
        euclidean_mm=euclid,
        evidence=ev,
        consistency=grow(sp.consistency),
        functional=grow(sp.functional),
        cross_species_support=grow(sp.cross_species_support),
        hierarchy_prior=grow(sp.hierarchy_prior),
        delays=ConductionDelayModel(
            length_mm=dist,
            velocity_prior=sp.delays.velocity_prior,
            tortuosity_prior=sp.delays.tortuosity_prior,
            length_source=sp.delays.length_source,
        ),
        ledger=sp.ledger,
        provenance={
            **sp.provenance,
            "cerebellum": {
                "atlas": cerebellum.name,
                "n": cerebellum.n_parcels,
                "structural_connectivity": "absent -- no source covers it",
            },
        },
        control_kind=sp.control_kind,
        weights_units=sp.weights_units,
        direction_known=False,
        n_streams=sp.n_streams,
        stream_names=sp.stream_names,
        seed=sp.seed,
    )
