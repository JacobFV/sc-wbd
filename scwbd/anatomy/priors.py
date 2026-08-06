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

__all__ = ["BrainPrior", "TIMESCALE_RANGE_MS", "EI_LOG_RANGE"]

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

        return cls(
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
                    for k in (
                        "schaefer2018" if atlas.startswith("Schaefer") else "desikan2006",
                        "enigma_hcp_sc",
                        "hansen_receptors",
                        "neuromaps",
                    )
                },
            },
        )

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

    def ei_ratio_prior(self) -> list[PriorBase]:
        """Per-parcel excitation/inhibition ratio prior.

        Cortical parcels get a log-normal centred on
        ``exp(EI_LOG_RANGE * z_ei)`` where ``z_ei`` is the receptor-derived E/I
        proxy scaled to roughly unit variance, so the ratio is 1.0 at the
        cortical mean and spans about a factor of two across cortex.  Sigma is
        deliberately as large as the between-parcel spread: the proxy orders
        parcels far better than it scales them.

        Parcels without receptor coverage -- subcortical and cerebellar --
        receive the *same* log-normal centred on 1.0 with a wider sigma.  That
        is not an imputed value: it is an explicit statement that we do not know
        their E/I balance, and it is visible as a wider prior rather than hidden
        as a filled-in number.
        """
        n = self.n_parcels
        out: list[PriorBase] = []
        z = np.full(n, np.nan)
        if "ei_proxy" in self.maps:
            m = self.maps["ei_proxy"]
            k = min(m.values.size, n)
            z[:k] = m.zscored()[:k]
        cite = (
            "Receptor-derived E/I proxy (NMDA + mGluR5 versus GABA-A) from the "
            "Hansen PET atlas; span calibrated to the hierarchical E/I gradients "
            "used by Demirtas M. et al. (2019) Neuron 101:1181-1194 and "
            "Wang X.-J. (2020) Nat Rev Neurosci 21:169-178. PROXY, NOT A "
            "MEASUREMENT: PET binding potential is not synaptic conductance, and "
            "a subject's E/I balance is not inferable from an atlas value."
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
                            + " NO RECEPTOR COVERAGE for this parcel (subcortical or "
                            "cerebellar): the prior is centred on the cortical mean "
                            "with double the width, which states ignorance rather "
                            "than imputing a value."
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
