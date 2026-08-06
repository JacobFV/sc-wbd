"""Structural connectivity prior with hard/soft/proposed edge classification.

This is the *probabilistic interaction grammar* of thesis §2.2: a connectome is
not a scalar matrix but a typed topology prior over which operators may exist,
each edge carrying its own evidence class, delay prior and uncertainty ledger.

Three commitments, taken from §2.2 and §3.1
-------------------------------------------
1. **A streamline count is not an attention weight.**  Weights here are on the
   arbitrary monotone scale the upstream pipeline produced (log streamline
   density), and are exposed as such.
2. **Absence of a tract is not conditional independence.**  ``absent`` means
   "no evidence in these sources", never "no pathway".  The class
   ``proposed`` exists precisely so that an edge with non-tractography support
   can be admitted under model comparison with a complexity penalty.
3. **Human diffusion MRI supplies no direction.**  ``weights`` is symmetric and
   ``direction_known`` is ``False``.  A separate, explicitly cross-species and
   explicitly ``proposed`` hierarchy prior supplies an *antisymmetric*
   feedforward/feedback tendency; it is never mistaken for a measurement.

Gate G2 controls
----------------
``StructuralPrior`` builds the null graphs that make "anatomy helps" testable:
:meth:`~StructuralPrior.randomized`, :meth:`~StructuralPrior.distance_matched`,
:meth:`~StructuralPrior.dense`, :meth:`~StructuralPrior.local_only` and
:meth:`~StructuralPrior.graph_only`.  Without them G2 has no denominator.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from . import sources as S
from ._compat import (
    EVIDENCE_ORDER,
    DiracPrior,
    LogNormalPrior,
    PriorBase,
    UncertaintyLedger,
    as_prior,
    evidence_rank,
    group_average_ledger,
)
from .atlases import Parcellation, load_parcellation
from .geometry import ParcelGeometry, parcel_geometry
from .paths import cache_dir, derived_dir, src_dir

__all__ = [
    "StructuralPrior",
    "EdgeEvidence",
    "ConductionDelayModel",
    "CONDUCTION_VELOCITY_PRIOR",
    "TORTUOSITY_PRIOR",
    "EDR_LAMBDA_PRIOR",
    "load_structural_prior",
    "ControlKind",
]

ControlKind = Literal[
    "empirical", "randomized", "distance_matched", "dense", "local_only", "graph_only"
]

# ---------------------------------------------------------------------------
# physiological priors  (agent E samples these; they are NOT point estimates)
# ---------------------------------------------------------------------------
#: Corticocortical conduction velocity of myelinated association fibres.
#:
#: Log-normal with median exp(1.792) = 6.0 m/s and sigma 0.56, giving a central
#: 95% interval of roughly 2.0-18 m/s.  Anchors: Caminiti et al. (2013) J
#: Neurosci 33:14501 measure human callosal conduction velocities spanning
#: ~3-15 m/s with a mode near 6; Drakesmith et al. (2019) NeuroImage 203:116186
#: derive 4-12 m/s from axon-diameter distributions; whole-brain modelling
#: conventionally fixes 5-10 m/s (Deco et al. 2009 PNAS; Sanz-Leon et al. 2015).
#: The spread is wide on purpose -- fixing a velocity is the single most common
#: hidden assumption in delayed whole-brain models.
CONDUCTION_VELOCITY_PRIOR = LogNormalPrior(
    mu=math.log(6.0),
    sigma=0.56,
    units="m/s",
    provenance=(
        "Caminiti R. et al. (2013) J Neurosci 33:14501-14511; "
        "Drakesmith M. et al. (2019) NeuroImage 203:116186; "
        "Deco G. et al. (2009) PNAS 106:10302-10307. "
        "Median 6.0 m/s, 95% interval approx. 2.0-18 m/s."
    ),
)

#: Ratio of true white-matter fibre length to the length proxy we can compute
#: from a template (Euclidean centroid separation).  Fibres are not straight.
#:
#: Log-normal, median 1.25, sigma 0.20 -> 95% interval approx. 0.85-1.85.  The
#: lower tail is below 1 deliberately: for long cortico-cortical connections a
#: surface geodesic *overestimates* the white-matter path, which cuts under the
#: cortex.  Anchor: streamline-length versus centroid-distance comparisons in
#: Lemarechal et al. (2022) Brain 145:1653 and Betzel & Bassett (2018) PNAS
#: 115:E4880 supplementary distance analyses.
TORTUOSITY_PRIOR = LogNormalPrior(
    mu=math.log(1.25),
    sigma=0.20,
    units="dimensionless",
    provenance=(
        "Ratio of tractography streamline length to inter-centroid distance; "
        "Lemarechal J.-D. et al. (2022) Brain 145:1653-1667; "
        "Betzel R.F., Bassett D.S. (2018) PNAS 115:E4880-E4889."
    ),
)

#: Decay constant of the exponential distance rule p(edge) ~ exp(-lambda d).
#:
#: CROSS-SPECIES. Ercsey-Ravasz et al. (2013) Neuron 80:184 fit lambda = 0.19
#: mm^-1 to macaque retrograde tracer data; Horvat et al. (2016) PLoS Biol
#: 14:e1002512 show the rule holds in mouse with a species-specific constant
#: that scales with brain size.  The human value is not measured.  We carry a
#: log-normal centred well below the macaque value (median 0.10 mm^-1) with a
#: wide sigma, because a larger brain must have a shallower decay for the
#: connectome to stay connected -- and we mark every use of it as cross-species
#: transfer.
EDR_LAMBDA_PRIOR = LogNormalPrior(
    mu=math.log(0.10),
    sigma=0.45,
    units="1/mm",
    provenance=(
        "CROSS-SPECIES TRANSFER. Ercsey-Ravasz M. et al. (2013) Neuron "
        "80:184-197 (macaque, lambda = 0.19 /mm); Horvat S. et al. (2016) PLoS "
        "Biol 14:e1002512 (mouse). Rescaled for human brain size; no direct "
        "human tracer measurement exists."
    ),
)


# ---------------------------------------------------------------------------
# per-edge evidence
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EdgeEvidence:
    """Why one edge received the class it did.

    Attributes
    ----------
    evidence_class
        ``"hard"``, ``"soft"``, ``"proposed"`` or ``"absent"``.
    weight
        Edge weight on the source pipeline's scale (log streamline density for
        the ENIGMA/HCP connectome).  ``0.0`` when absent.
    distance_mm
        Inter-parcel distance used for the delay and the distance penalty.
    consistency
        Fraction of the independent evidence streams available at this
        parcellation in which the edge is present, in ``[0, 1]``.
    n_streams
        How many independent streams were actually available.  A consistency of
        1.0 out of one stream is weak; the number is reported so that cannot be
        hidden.
    cross_species_support
        Probability of the edge under the macaque-derived exponential distance
        rule.  A prior, not an observation; see :data:`EDR_LAMBDA_PRIOR`.
    functional_corroboration
        Group resting-state functional connectivity for the pair.  Recorded but
        **never** used to promote an edge to ``hard``: functional correlation is
        not structural connectivity (ARCHITECTURE.md §7.3).
    reasons
        Human-readable list of the criteria that fired.
    """

    i: int
    j: int
    label_i: str
    label_j: str
    evidence_class: str
    weight: float
    distance_mm: float
    consistency: float
    n_streams: int
    cross_species_support: float
    functional_corroboration: float
    reasons: tuple[str, ...]
    mechanistic_status: str = "effective"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"EdgeEvidence({self.label_i} <-> {self.label_j}: {self.evidence_class}, "
            f"w={self.weight:.3g}, d={self.distance_mm:.0f}mm, "
            f"consistency={self.consistency:.2f}/{self.n_streams})"
        )


# ---------------------------------------------------------------------------
# delays
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConductionDelayModel:
    """Delay = (length x tortuosity) / velocity, with both factors uncertain.

    Agent E samples this; it must not be collapsed to a matrix of numbers
    without an explicit call to :meth:`median_delay_s`, which is deliberately
    verbose about what it is doing.
    """

    length_mm: np.ndarray
    velocity_prior: PriorBase = field(default_factory=lambda: CONDUCTION_VELOCITY_PRIOR)
    tortuosity_prior: PriorBase = field(default_factory=lambda: TORTUOSITY_PRIOR)
    length_source: str = "euclidean_centroid"

    def median_delay_s(self) -> np.ndarray:
        """Delay matrix at the median of both priors.

        This *collapses* two uncertain quantities to a point.  Use it for
        plotting, sanity checks and tests -- not for inference.
        """
        from ._compat import prior_quantile

        v = prior_quantile(self.velocity_prior, 0.5)
        t = prior_quantile(self.tortuosity_prior, 0.5)
        return (self.length_mm * t * 1e-3) / v

    def sample_delay_s(self, seed: int, n: int = 1, *, per_edge: bool = False) -> np.ndarray:
        """Draw ``n`` delay matrices, shape ``(n, N, N)``.

        Parameters
        ----------
        per_edge
            If ``False`` (default) one velocity and one tortuosity are drawn per
            sample and applied to the whole brain: this represents the
            *global* uncertainty that a subject's conduction speed is unknown.
            If ``True``, tortuosity is drawn independently per edge as well,
            representing bundle-to-bundle variation in path straightness.
            Velocity stays global either way; a per-edge velocity would imply
            edge-specific axon calibre knowledge we do not have.
        """
        rng_v = np.asarray(self.velocity_prior.sample(seed, n)).reshape(n)
        m = self.length_mm.shape[0]
        if per_edge:
            t = np.asarray(self.tortuosity_prior.sample(seed + 1, (n, m, m)))
            t = 0.5 * (t + np.swapaxes(t, 1, 2))  # keep the delay matrix symmetric
        else:
            t = np.asarray(self.tortuosity_prior.sample(seed + 1, n)).reshape(n, 1, 1)
        return (self.length_mm[None] * t * 1e-3) / rng_v.reshape(n, 1, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "velocity_prior": self.velocity_prior.model_dump(mode="json"),
            "tortuosity_prior": self.tortuosity_prior.model_dump(mode="json"),
            "length_source": self.length_source,
        }


# ---------------------------------------------------------------------------
# evidence-stream loading
# ---------------------------------------------------------------------------
def _enigma_matrix(kind: str, atlas_key: str) -> tuple[np.ndarray, np.ndarray]:
    """Load an ENIGMA/HCP matrix and its labels.

    ``kind`` is ``"struc"`` or ``"func"``; ``atlas_key`` is one of
    ``"", "_schaefer_100", ..., "_glasser_360"`` and selects the parcellation.
    """
    import pandas as pd

    from .atlases import _enigma_dir

    d = _enigma_dir() / "matrices" / "hcp_connectivity"
    # Upstream naming is inconsistent: the combined cortex+subcortex functional
    # matrices are filed under "with_ctx" for every parcellation except
    # Desikan-Killiany, which uses "with_sctx" like the structural ones do.
    for infix in ("with_sctx", "with_ctx"):
        f = d / f"{kind}Matrix_{infix}{atlas_key}.csv"
        if f.exists():
            break
    else:  # pragma: no cover
        raise FileNotFoundError(f"no {kind} matrix for atlas key {atlas_key!r} in {d}")
    mat = pd.read_csv(f, header=None).values
    lab = pd.read_csv(d / f"{kind}Labels_with_sctx{atlas_key}.csv", header=None).values.ravel()
    return np.asarray(mat, dtype=np.float64), np.asarray([str(x) for x in lab])


_ENIGMA_KEYS = {
    "DesikanKilliany": "",
    "Glasser360": "_glasser_360",
    "Schaefer100x7": "_schaefer_100",
    "Schaefer200x7": "_schaefer_200",
    "Schaefer300x7": "_schaefer_300",
    "Schaefer400x7": "_schaefer_400",
}


def _independent_streams(atlas: str) -> list[dict[str, Any]]:
    """Independent structural-connectivity observations of the same anatomy.

    Each stream is a binary presence matrix over the *cortical* parcels of
    ``atlas``, together with how independent it actually is.  Agreement between
    two pipelines applied to the same HCP scans is weaker evidence than
    agreement between HCP and the Lausanne cohort, and the ``independence``
    field records that so the classifier can weight it.
    """
    streams: list[dict[str, Any]] = []

    # (a) Hansen et al. 2022: HCP scans, independent processing pipeline.
    if atlas == "Schaefer100x7":
        p = src_dir() / "hansen_receptors" / "data" / "schaefer" / "sc_binary.npy"
        if p.exists():
            streams.append(
                {
                    "name": "hansen2022_schaefer100",
                    "present": np.load(p).astype(bool),
                    "independence": 0.5,
                    "source_key": "hansen_schaefer_sc",
                    "note": "same cohort family (HCP), different pipeline",
                }
            )
    if atlas == "DesikanKilliany":
        p = src_dir() / "hansen_receptors" / "data" / "lausanne" / "sc_weighted.npy"
        if p.exists():
            streams.append(
                {
                    "name": "hansen2022_lausanne033",
                    "present": (np.load(p) > 0),
                    "independence": 1.0,
                    "source_key": "hansen_lausanne_sc",
                    "note": "Lausanne cohort, deterministic tractography",
                }
            )
        try:
            import netneurotools.datasets as ntd

            g = ntd.fetch_famous_gmat(
                "human_struct_scale033", data_dir=str(cache_dir() / "netneurotools")
            )
            conn = np.asarray(g["conn"], dtype=float)
            if conn.shape[0] == 68:
                streams.append(
                    {
                        "name": "netneurolab_lausanne033",
                        "present": conn > 0,
                        "independence": 1.0,
                        "source_key": "netneuro_lausanne_sc",
                        "note": "different cohort, scanner and tractography algorithm",
                    }
                )
        except Exception:  # noqa: BLE001 - stream simply unavailable
            pass
    return streams


def _multiscale_replicates(atlas: str, parc: Parcellation) -> list[dict[str, Any]]:
    """Coarser-parcellation replicates of the same connectome.

    An edge that survives being re-derived at a different parcel granularity is
    less likely to be a boundary artifact.  This is *not* an independent
    cohort: it is the same scans re-binned, so it is weighted accordingly.
    """
    order = ["Schaefer100x7", "Schaefer200x7", "Schaefer300x7", "Schaefer400x7"]
    if atlas not in order:
        return []
    out: list[dict[str, Any]] = []
    fine = load_parcellation(atlas, "fsLR", "32k")
    from .atlases import crosswalk

    for other in order:
        if other == atlas:
            continue
        try:
            coarse = load_parcellation(other, "fsLR", "32k")
            mat, lab = _enigma_matrix("struc", _ENIGMA_KEYS[other])
        except Exception:  # noqa: BLE001
            continue
        nc = coarse.n_parcels
        mat = mat[:nc, :nc]
        ov = crosswalk(fine, coarse)  # (n_fine, n_coarse)
        assign = np.argmax(ov, axis=1)
        pres_c = mat > 0
        out.append(
            {
                "name": f"enigma_regrid_{other}",
                "present": pres_c[np.ix_(assign, assign)],
                "independence": 0.25,
                "source_key": "enigma_hcp_sc",
                "note": f"same HCP scans re-binned at {other}",
            }
        )
    return out


# ---------------------------------------------------------------------------
# the prior
# ---------------------------------------------------------------------------
@dataclass
class StructuralPrior:
    """Group-average structural connectivity as a typed topology prior.

    Attributes
    ----------
    weights
        ``(n, n)`` symmetric, non-negative, zero-diagonal edge weights on the
        upstream pipeline's scale.  ``weights_units`` names that scale.
    distance_mm
        ``(n, n)`` inter-parcel distance actually used for delays.
    evidence
        ``(n, n)`` uint8 codes into :data:`EVIDENCE_ORDER`
        (0 absent, 1 proposed, 2 soft, 3 hard).
    hierarchy_prior
        ``(n, n)`` **antisymmetric** feedforward tendency in ``[-1, 1]``:
        positive at ``[i, j]`` means "if this edge is directed, the modal
        direction is i -> j (feedforward, low to high in the cortical
        hierarchy)".  This is a *proposed*, cross-species-informed prior, not a
        measurement, and ``direction_known`` stays ``False``.
    control_kind
        ``"empirical"`` for the real connectome; otherwise the name of the G2
        null this object represents.
    """

    parcellation: Parcellation
    labels: np.ndarray
    weights: np.ndarray
    distance_mm: np.ndarray
    euclidean_mm: np.ndarray
    evidence: np.ndarray
    consistency: np.ndarray
    functional: np.ndarray
    cross_species_support: np.ndarray
    hierarchy_prior: np.ndarray
    delays: ConductionDelayModel
    ledger: UncertaintyLedger
    provenance: dict[str, Any]
    control_kind: str = "empirical"
    weights_units: str = "log10_streamline_density (arbitrary monotone scale)"
    direction_known: bool = False
    n_streams: int = 0
    stream_names: tuple[str, ...] = ()
    seed: int | None = None

    # -- basics ----------------------------------------------------------
    @property
    def n_parcels(self) -> int:
        return int(self.labels.shape[0])

    def __repr__(self) -> str:  # pragma: no cover
        c = self.class_counts()
        return (
            f"StructuralPrior({self.parcellation.name}, n={self.n_parcels}, "
            f"kind={self.control_kind}, density={self.density():.3f}, "
            f"hard={c['hard']}, soft={c['soft']}, proposed={c['proposed']})"
        )

    def density(self) -> float:
        n = self.n_parcels
        return float((self.weights > 0).sum() / (n * (n - 1)))

    def class_counts(self) -> dict[str, int]:
        """Number of *undirected* edges in each evidence class."""
        iu = np.triu_indices(self.n_parcels, 1)
        v = self.evidence[iu]
        return {c: int((v == k).sum()) for k, c in enumerate(EVIDENCE_ORDER)}

    def mask(self, min_class: str = "soft") -> np.ndarray:
        """Boolean adjacency mask retaining edges at or above ``min_class``.

        This is what the compiler consumes: a zero here creates *exact*
        independence across the direct edge inside the compiled model, and per
        §2.2 that is a modelling statement, not a biological one.
        """
        return self.evidence >= evidence_rank(min_class)

    def strength(self) -> np.ndarray:
        return self.weights.sum(axis=1)

    def degree(self) -> np.ndarray:
        return (self.weights > 0).sum(axis=1)

    # -- per-edge query --------------------------------------------------
    def edge_evidence(self, i: int, j: int) -> EdgeEvidence:
        """Evidence class for one edge, with the reasoning that produced it."""
        if i == j:
            raise ValueError("self-edges are not part of the long-range graph")
        cls = EVIDENCE_ORDER[int(self.evidence[i, j])]
        w = float(self.weights[i, j])
        d = float(self.distance_mm[i, j])
        cons = float(self.consistency[i, j])
        reasons: list[str] = []
        if self.control_kind != "empirical":
            reasons.append(
                f"control graph ({self.control_kind}): this edge is a null-model "
                "artifact and carries no anatomical evidence"
            )
        elif w <= 0:
            if cls == "proposed":
                reasons.append(
                    "no tractography support, but admitted for model comparison: "
                    f"short template distance ({d:.0f} mm) and strong group "
                    f"functional correlation ({self.functional[i, j]:.2f}); "
                    "tractography is known to miss short U-fibres and pathways "
                    "that run tangentially to the cortical sheet"
                )
            else:
                reasons.append(
                    "no support in any available stream; absence of an observed "
                    "tract is not evidence of conditional independence (§2.2)"
                )
        else:
            reasons.append(
                f"present in the group-average tractogram (weight {w:.3g} on the "
                f"{self.weights_units} scale)"
            )
            reasons.append(
                f"reproduced in {cons * 100:.0f}% of {self.n_streams} evidence "
                f"stream(s): {', '.join(self.stream_names) or 'none beyond the primary'}"
            )
            if d > _LONG_MM:
                reasons.append(
                    f"long-range ({d:.0f} mm): tractography false-positive rate "
                    "rises with tract length (Maier-Hein et al. 2017), so the "
                    "edge cannot be hard on tractography alone"
                )
            if cls == "hard":
                reasons.append(
                    "removal would contradict the chosen anatomical model: "
                    "reproduced across every available stream at a distance where "
                    "tractography is reliable"
                )
        reasons.append(
            f"exponential-distance-rule support {self.cross_species_support[i, j]:.3f} "
            "(CROSS-SPECIES transfer from macaque tracer statistics; informs the "
            "prior over edge existence, never asserts a human pathway)"
        )
        reasons.append(
            f"group functional connectivity {self.functional[i, j]:.2f} recorded "
            "but never used to promote an edge: functional correlation is not "
            "structural connectivity"
        )
        reasons.append(
            "direction is unknown: human diffusion MRI is undirected, so this "
            "edge is symmetric by construction"
        )
        return EdgeEvidence(
            i=i,
            j=j,
            label_i=str(self.labels[i]),
            label_j=str(self.labels[j]),
            evidence_class=cls,
            weight=w,
            distance_mm=d,
            consistency=cons,
            n_streams=self.n_streams,
            cross_species_support=float(self.cross_species_support[i, j]),
            functional_corroboration=float(self.functional[i, j]),
            reasons=tuple(reasons),
            mechanistic_status="effective" if self.control_kind == "empirical" else "surrogate",
        )

    # -- G2 controls -----------------------------------------------------
    def _control(
        self,
        *,
        kind: str,
        weights: np.ndarray,
        seed: int | None,
        note: str,
        distance_mm: np.ndarray | None = None,
    ) -> "StructuralPrior":
        w = np.asarray(weights, dtype=np.float64)
        w = 0.5 * (w + w.T)
        np.fill_diagonal(w, 0.0)
        d = self.distance_mm if distance_mm is None else distance_mm
        # Every control edge is `proposed`: a null graph has, by construction,
        # no anatomical evidence.  Downgrading is the point -- a control that
        # inherited `hard` labels would smuggle anatomy back into the baseline.
        ev = np.where(w > 0, evidence_rank("proposed"), evidence_rank("absent")).astype(np.uint8)
        np.fill_diagonal(ev, 0)
        prov = dict(self.provenance)
        prov["control"] = {"kind": kind, "seed": seed, "note": note}
        return StructuralPrior(
            parcellation=self.parcellation,
            labels=self.labels,
            weights=w,
            distance_mm=d,
            euclidean_mm=self.euclidean_mm,
            evidence=ev,
            consistency=np.zeros_like(self.consistency),
            functional=self.functional,
            cross_species_support=self.cross_species_support,
            hierarchy_prior=np.zeros_like(self.hierarchy_prior),
            delays=ConductionDelayModel(
                length_mm=d,
                velocity_prior=self.delays.velocity_prior,
                tortuosity_prior=self.delays.tortuosity_prior,
                length_source=self.delays.length_source,
            ),
            ledger=self.ledger,
            provenance=prov,
            control_kind=kind,
            weights_units=self.weights_units,
            direction_known=False,
            n_streams=0,
            stream_names=(),
            seed=seed,
        )

    def randomized(self, seed: int = 0, *, n_swaps_per_edge: int = 10) -> "StructuralPrior":
        """Degree- and strength-preserving rewiring (Maslov-Sneppen + weight sort).

        Binary topology is randomised by double-edge swaps, which preserve every
        node's degree exactly.  Weights are then reassigned so that the rank
        order of node strength is preserved: edges of the rewired graph are
        sorted by the product of their endpoints' original strengths and given
        the original weights in the same rank order (Rubinov & Sporns 2011).

        This is the control that asks: *does the model need the specific
        topology, or only its degree sequence?*
        """
        rng = np.random.default_rng(seed)
        n = self.n_parcels
        a = (self.weights > 0)
        iu = np.triu_indices(n, 1)
        edges = np.array(list(zip(*[x[a[iu]] for x in iu])), dtype=np.int64)
        if edges.size == 0:
            return self._control(kind="randomized", weights=self.weights * 0, seed=seed,
                                 note="empty graph")
        adj = a.copy()
        n_swap = int(n_swaps_per_edge * edges.shape[0])
        m = edges.shape[0]
        for _ in range(n_swap):
            e1, e2 = rng.integers(0, m, 2)
            if e1 == e2:
                continue
            a1, b1 = edges[e1]
            a2, b2 = edges[e2]
            if rng.random() < 0.5:
                a2, b2 = b2, a2
            if len({a1, b1, a2, b2}) < 4:
                continue
            if adj[a1, b2] or adj[a2, b1]:
                continue
            adj[a1, b1] = adj[b1, a1] = False
            adj[a2, b2] = adj[b2, a2] = False
            adj[a1, b2] = adj[b2, a1] = True
            adj[a2, b1] = adj[b1, a2] = True
            edges[e1] = (a1, b2)
            edges[e2] = (a2, b1)
        # weight reassignment preserving the strength sequence in rank
        s = self.strength()
        new_i, new_j = np.triu_indices(n, 1)
        keep = adj[new_i, new_j]
        new_i, new_j = new_i[keep], new_j[keep]
        expect = s[new_i] * s[new_j]
        order = np.argsort(np.argsort(expect))
        wvals = np.sort(self.weights[iu][a[iu]])
        w = np.zeros((n, n))
        w[new_i, new_j] = wvals[order]
        w = w + w.T
        return self._control(
            kind="randomized",
            weights=w,
            seed=seed,
            note=(
                "Maslov-Sneppen double-edge swaps preserve the degree sequence "
                "exactly; weights reassigned by strength-product rank (Rubinov & "
                "Sporns 2011) so the strength sequence is preserved in rank."
            ),
        )

    def distance_matched(self, seed: int = 0, *, n_bins: int = 20,
                         n_swaps_per_edge: int = 10) -> "StructuralPrior":
        """Rewiring that preserves the edge-length distribution *and* degree.

        Edges are binned by length and double-edge swaps are permitted only
        between edges in the same bin, so the connectome's characteristic
        distance decay survives while its specific topology does not.

        This is the control that matters most for G2: a large part of a
        connectome's predictive value is simply that nearby regions are
        connected.  A model that beats a *random* graph but not a
        *distance-matched* graph has learned geometry, not anatomy.
        """
        rng = np.random.default_rng(seed)
        n = self.n_parcels
        a = self.weights > 0
        iu = np.triu_indices(n, 1)
        ei, ej = iu[0][a[iu]], iu[1][a[iu]]
        if ei.size == 0:
            return self._control(kind="distance_matched", weights=self.weights * 0,
                                 seed=seed, note="empty graph")
        d = self.distance_mm[ei, ej]
        qs = np.quantile(d, np.linspace(0, 1, n_bins + 1))
        qs[0] -= 1e-9
        qs[-1] += 1e-9
        binid = np.clip(np.searchsorted(qs, d, side="right") - 1, 0, n_bins - 1)
        adj = a.copy()
        edges = np.stack([ei, ej], axis=1)
        for b in range(n_bins):
            idx = np.flatnonzero(binid == b)
            if idx.size < 4:
                continue
            for _ in range(int(n_swaps_per_edge * idx.size)):
                k1, k2 = rng.choice(idx, 2, replace=False)
                a1, b1 = edges[k1]
                a2, b2 = edges[k2]
                if rng.random() < 0.5:
                    a2, b2 = b2, a2
                if len({a1, b1, a2, b2}) < 4:
                    continue
                if adj[a1, b2] or adj[a2, b1]:
                    continue
                # the swap is only admissible if the new edges stay in this bin
                d1 = self.distance_mm[a1, b2]
                d2 = self.distance_mm[a2, b1]
                if not (qs[b] <= d1 < qs[b + 1] and qs[b] <= d2 < qs[b + 1]):
                    continue
                adj[a1, b1] = adj[b1, a1] = False
                adj[a2, b2] = adj[b2, a2] = False
                adj[a1, b2] = adj[b2, a1] = True
                adj[a2, b1] = adj[b1, a2] = True
                edges[k1] = (a1, b2)
                edges[k2] = (a2, b1)
        new_i, new_j = np.triu_indices(n, 1)
        keep = adj[new_i, new_j]
        new_i, new_j = new_i[keep], new_j[keep]
        # Preserve the weight-distance relationship, not merely the weight set:
        # original weights sorted by original edge length are handed to the new
        # edges in their own length order.
        order = np.argsort(np.argsort(self.distance_mm[new_i, new_j]))
        wvals = self.weights[ei, ej][np.argsort(d)]
        w = np.zeros((n, n))
        w[new_i, new_j] = wvals[order]
        w = w + w.T
        return self._control(
            kind="distance_matched",
            weights=w,
            seed=seed,
            note=(
                "Double-edge swaps restricted to within-length-bin pairs "
                f"({n_bins} quantile bins), preserving both the degree sequence "
                "and the edge-length distribution; weights reassigned in "
                "distance rank order so the weight-distance relationship also "
                "survives."
            ),
        )

    def dense(self) -> "StructuralPrior":
        """Fully connected graph with uniform weights, total strength preserved.

        The 'no anatomy at all' baseline: every region may talk to every other
        region equally.  Whatever the model gains over this is what the topology
        prior bought.
        """
        n = self.n_parcels
        total = float(self.weights.sum())
        w = np.full((n, n), total / max(n * (n - 1), 1))
        np.fill_diagonal(w, 0.0)
        return self._control(
            kind="dense",
            weights=w,
            seed=None,
            note="all-to-all, uniform weight, total strength matched to the empirical graph",
        )

    def local_only(self, radius_mm: float = 40.0) -> "StructuralPrior":
        """Keep only short-range edges; delete every long-range projection.

        Isolates the contribution of the sparse long-range operator ``F_long``
        of §4.1 from the local field ``F_local``.  ``radius_mm`` defaults to
        40 mm, roughly the reach of dense horizontal cortico-cortical
        connectivity.
        """
        w = np.where(self.distance_mm <= radius_mm, self.weights, 0.0)
        np.fill_diagonal(w, 0.0)
        return self._control(
            kind="local_only",
            weights=w,
            seed=None,
            note=f"edges longer than {radius_mm:g} mm deleted; local topology untouched",
        )

    def graph_only(self) -> "StructuralPrior":
        """Keep the topology, discard the weights (all present edges equal).

        Separates 'which regions are connected' from 'how strongly'.  Since the
        upstream weights are log streamline density on an arbitrary monotone
        scale, a model that depends on their exact values is depending on a
        pipeline artifact.
        """
        a = self.weights > 0
        mean_w = float(self.weights[a].mean()) if a.any() else 0.0
        w = np.where(a, mean_w, 0.0)
        np.fill_diagonal(w, 0.0)
        return self._control(
            kind="graph_only",
            weights=w,
            seed=None,
            note="binary topology preserved, every present edge given the mean weight",
        )

    def controls(self, seed: int = 0) -> dict[str, "StructuralPrior"]:
        """All five G2 controls in one call."""
        return {
            "randomized": self.randomized(seed),
            "distance_matched": self.distance_matched(seed),
            "dense": self.dense(),
            "local_only": self.local_only(),
            "graph_only": self.graph_only(),
        }

    # -- io --------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            labels=self.labels,
            weights=self.weights,
            distance_mm=self.distance_mm,
            euclidean_mm=self.euclidean_mm,
            evidence=self.evidence,
            consistency=self.consistency,
            functional=self.functional,
            cross_species_support=self.cross_species_support,
            hierarchy_prior=self.hierarchy_prior,
            _meta=np.array(
                json.dumps(
                    {
                        "parcellation": {
                            "name": self.parcellation.name,
                            "space": self.parcellation.space,
                            "density": self.parcellation.density,
                        },
                        "delays": self.delays.to_dict(),
                        "ledger": self.ledger.model_dump(mode="json"),
                        "provenance": self.provenance,
                        "control_kind": self.control_kind,
                        "weights_units": self.weights_units,
                        "direction_known": self.direction_known,
                        "n_streams": self.n_streams,
                        "stream_names": list(self.stream_names),
                        "seed": self.seed,
                    }
                )
            ),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "StructuralPrior":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["_meta"]))
        pmeta = meta["parcellation"]
        parc = load_parcellation(pmeta["name"], pmeta["space"], pmeta["density"])
        return cls(
            parcellation=parc,
            labels=z["labels"],
            weights=z["weights"],
            distance_mm=z["distance_mm"],
            euclidean_mm=z["euclidean_mm"],
            evidence=z["evidence"],
            consistency=z["consistency"],
            functional=z["functional"],
            cross_species_support=z["cross_species_support"],
            hierarchy_prior=z["hierarchy_prior"],
            delays=ConductionDelayModel(
                length_mm=z["distance_mm"],
                velocity_prior=as_prior(meta["delays"]["velocity_prior"]),
                tortuosity_prior=as_prior(meta["delays"]["tortuosity_prior"]),
                length_source=meta["delays"]["length_source"],
            ),
            ledger=UncertaintyLedger.model_validate(meta["ledger"]),
            provenance=meta["provenance"],
            control_kind=meta["control_kind"],
            weights_units=meta["weights_units"],
            direction_known=meta["direction_known"],
            n_streams=meta["n_streams"],
            stream_names=tuple(meta["stream_names"]),
            seed=meta["seed"],
        )


# ---------------------------------------------------------------------------
# classification thresholds
# ---------------------------------------------------------------------------
#: Above this centroid separation, tractography false-positive rates rise
#: sharply (Maier-Hein et al. 2017 Nat Commun 8:1349; Thomas et al. 2014 PNAS
#: 111:16574) and an edge cannot be `hard` on tractography alone.
_LONG_MM = 90.0
#: Consistency (independence-weighted fraction of streams) required for `hard`.
_HARD_CONSISTENCY = 0.999
#: Weight percentile above which a long edge may still reach `hard`, because a
#: very strong long projection (e.g. a major callosal or association bundle) is
#: not plausibly a stochastic false positive.
_HARD_LONG_WEIGHT_PCTL = 90.0
#: Functional-correlation and distance thresholds for admitting a `proposed`
#: edge that tractography did not find.  The ENIGMA functional matrices are
#: Fisher z-transformed group averages with negative correlations already set
#: to zero, so 0.5 here is z = 0.5, i.e. r ~ 0.46.
_PROPOSED_FC = 0.5
_PROPOSED_MM = 45.0


def _classify(
    weights: np.ndarray,
    distance: np.ndarray,
    consistency: np.ndarray,
    functional: np.ndarray,
) -> np.ndarray:
    """Assign each pair an evidence class.  See module docstring and §2.2."""
    n = weights.shape[0]
    present = weights > 0
    ev = np.full((n, n), evidence_rank("absent"), dtype=np.uint8)

    # -- soft: present anywhere ----------------------------------------
    ev[present] = evidence_rank("soft")

    # -- hard: reproduced everywhere, at a length tractography handles --
    strong = np.zeros_like(present)
    if present.any():
        thr = np.percentile(weights[present], _HARD_LONG_WEIGHT_PCTL)
        strong = weights >= thr
    hard = present & (consistency >= _HARD_CONSISTENCY) & ((distance <= _LONG_MM) | strong)
    ev[hard] = evidence_rank("hard")

    # -- proposed: absent from tractography but not from plausibility ---
    proposed = (
        (~present)
        & (distance <= _PROPOSED_MM)
        & (functional >= _PROPOSED_FC)
    )
    ev[proposed] = evidence_rank("proposed")

    ev = np.minimum(ev, ev.T)  # an undirected edge gets one class
    np.fill_diagonal(ev, evidence_rank("absent"))
    return ev


def _hierarchy_prior(parc: Parcellation, present: np.ndarray) -> np.ndarray:
    """Antisymmetric feedforward tendency from a cortical hierarchy proxy.

    Human diffusion MRI cannot tell us which way an edge points.  Non-human
    primate tracer work (Markov et al. 2014) shows that laminar projection
    patterns order areas along a hierarchy, and in humans the T1w/T2w myelin
    map and the principal functional gradient are the standard proxies for the
    same axis (Burt et al. 2018 Nat Neurosci 21:1251).  We therefore emit an
    antisymmetric matrix ``H`` with ``H[i, j] = rank(j) - rank(i)`` on the
    sensorimotor-association axis, normalised to ``[-1, 1]``.

    ``H`` is a ``proposed`` object with ``mechanistic_status="functional"``.
    It says which direction is *more likely* if the edge is directed at all.
    It is not a measurement and must not be reported as one.
    """
    from .maps import regional_map

    try:
        rank = regional_map(parc, "sa_axis")
    except Exception:  # noqa: BLE001 - map unavailable for this parcellation
        return np.zeros(present.shape, dtype=np.float64)
    v = np.asarray(rank, dtype=np.float64)
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return np.zeros(present.shape, dtype=np.float64)
    r = np.full(v.shape, np.nan)
    r[ok] = np.argsort(np.argsort(v[ok])) / max(ok.sum() - 1, 1)
    h = r[None, :] - r[:, None]
    h[~np.isfinite(h)] = 0.0
    h = h * present
    return 0.5 * (h - h.T)  # enforce exact antisymmetry


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------
def load_structural_prior(
    atlas: str = "Schaefer400x7",
    *,
    include_subcortex: bool = True,
    space: str = "fsLR",
    density: str = "32k",
    length: str = "euclidean",
    rebuild: bool = False,
) -> StructuralPrior:
    """Build the empirical structural prior for a parcellation.

    Parameters
    ----------
    atlas
        One of the parcellations the ENIGMA/HCP connectome covers:
        ``DesikanKilliany``, ``Glasser360``, ``Schaefer{100,200,300,400}x7``.
    include_subcortex
        Append the 14 FreeSurfer aseg subcortical structures the connectome
        actually resolves.  Note these are *not* Tian parcels; the connectome
        has no coverage at that resolution and we do not invent it.
    length
        ``"euclidean"`` (default) or ``"geodesic"``.  Which one to use for
        delays is a modelling choice, so it is explicit.  Geodesic is longer
        than the white-matter path for long connections; Euclidean is shorter.
        The tortuosity prior spans both cases.

    Raises
    ------
    ValueError
        The requested atlas has no structural-connectome coverage.  We refuse to
        fabricate a connectome for a parcellation nobody measured one in.
    """
    if atlas not in _ENIGMA_KEYS:
        raise ValueError(
            f"no group structural connectome is available for {atlas!r}. "
            f"Covered parcellations: {sorted(_ENIGMA_KEYS)}. "
            "Refusing to synthesise connectivity for an uncovered parcellation."
        )
    tag = f"{atlas}__enigma_hcp__{'with' if include_subcortex else 'no'}sctx__{length}"
    cache = derived_dir("connectome") / f"{tag}.npz"
    if cache.exists() and not rebuild:
        return StructuralPrior.load(cache)

    key = _ENIGMA_KEYS[atlas]
    sc, sc_lab = _enigma_matrix("struc", key)
    fc, fc_lab = _enigma_matrix("func", key)
    parc_ctx = load_parcellation(atlas, space, density)
    n_ctx = parc_ctx.n_parcels

    if include_subcortex:
        parc_sub = load_parcellation("Aseg14", "MNI152", "1mm")
        n = n_ctx + parc_sub.n_parcels
        if sc.shape[0] != n:
            raise ValueError(f"{atlas}: connectome is {sc.shape[0]}x{sc.shape[0]}, expected {n}")
        labels = np.concatenate([parc_ctx.labels, parc_sub.labels])
        centroids = np.vstack([parc_ctx.centroids_mni, parc_sub.centroids_mni])
        structure = np.concatenate([parc_ctx.structure, parc_sub.structure])
        hemi = np.concatenate([parc_ctx.hemi, parc_sub.hemi])
        network = np.concatenate([parc_ctx.network, parc_sub.network])
    else:
        n = n_ctx
        sc, fc = sc[:n, :n], fc[:n, :n]
        labels = parc_ctx.labels
        centroids = parc_ctx.centroids_mni
        structure, hemi, network = parc_ctx.structure, parc_ctx.hemi, parc_ctx.network

    if not np.array_equal([str(x) for x in sc_lab[:n]], [str(x) for x in labels]):
        mism = [
            (str(a), str(b)) for a, b in zip(sc_lab[:n], labels) if str(a) != str(b)
        ]
        raise ValueError(f"{atlas}: connectome label order disagrees with parcellation: {mism[:3]}")

    # -- weights ---------------------------------------------------------
    w = np.asarray(sc, dtype=np.float64)
    w = 0.5 * (w + w.T)
    np.fill_diagonal(w, 0.0)
    # The upstream matrix is log-transformed, so a handful of very weak edges
    # come out negative.  A negative "amount of connection" is not meaningful;
    # those edges are recorded as present-but-weakest rather than deleted or
    # sign-flipped.
    neg = w < 0
    n_neg = int(neg.sum() // 2)
    if neg.any():
        pos_min = w[w > 0].min() if (w > 0).any() else 1e-6
        w[neg] = pos_min * 1e-3
    f = np.asarray(fc, dtype=np.float64)
    f = 0.5 * (f + f.T)
    np.fill_diagonal(f, 0.0)

    # -- geometry --------------------------------------------------------
    geom_ctx = parcel_geometry(parc_ctx)
    euclid = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
    np.fill_diagonal(euclid, 0.0)
    geo = euclid.copy()
    geo[:n_ctx, :n_ctx] = geom_ctx.geodesic_mm
    dist = euclid if length == "euclidean" else geo
    if length not in ("euclidean", "geodesic"):
        raise ValueError("length must be 'euclidean' or 'geodesic'")

    # -- evidence streams -------------------------------------------------
    present = w > 0
    streams = _independent_streams(atlas) + _multiscale_replicates(atlas, parc_ctx)
    cons = np.ones((n, n))
    names: list[str] = []
    if streams:
        num = np.zeros((n, n))
        den = 0.0
        for st in streams:
            p = np.asarray(st["present"], dtype=bool)
            full = np.zeros((n, n), dtype=bool)
            m = min(p.shape[0], n_ctx)
            full[:m, :m] = p[:m, :m]
            num += st["independence"] * full
            den += st["independence"]
            names.append(st["name"])
        cons = np.zeros((n, n))
        cons[:n_ctx, :n_ctx] = (num / den)[:n_ctx, :n_ctx]
        # Subcortical rows have no independent stream at all.  Their consistency
        # is 0, so they can never be classified `hard`: that is correct, not a
        # bug -- we have exactly one observation of them.
    else:
        cons = np.zeros((n, n))

    # -- cross-species EDR prior -----------------------------------------
    lam = float(EDR_LAMBDA_PRIOR.mean())
    edr = np.exp(-lam * dist)
    np.fill_diagonal(edr, 0.0)

    ev = _classify(w, dist, cons, f)
    hier = np.zeros((n, n))
    hier[:n_ctx, :n_ctx] = _hierarchy_prior(parc_ctx, present[:n_ctx, :n_ctx])

    ledger = group_average_ledger(
        units="dimensionless",
        # Group-average tractography weights are biased by an unknown amount
        # that is bounded below by "an edge that exists was missed entirely"
        # and above by "an edge that does not exist was invented".  The interval
        # is stated in units of the weight scale's interquartile range.
        bias_interval=(-1.0, 1.0),
        variance={
            "measurement": float(np.var(w[present])) if present.any() else 0.0,
            "between_session": 0.0,
            "model_class": float(np.var(w[present])) if present.any() else 0.0,
        },
        forbidden_inference=(
            "This is a group-average, undirected, tractography-derived graph. It "
            "does not give the direction of any pathway, the laminar origin or "
            "termination of any projection, or this subject's connectome. A zero "
            "entry means 'not detected by these pipelines', not 'not connected'."
        ),
        n_donors=None,
        validity_domain={
            "cohort": S.SRC["enigma_hcp_sc"]["cohort"],
            "pipeline": S.SRC["enigma_hcp_sc"]["pipeline"],
            "population": "healthy young adults (HCP age range 22-37)",
            "direction_known": False,
            "laminar_resolved": False,
            "streams": names,
        },
        model_discrepancy=None,
        notes=S.SRC["enigma_hcp_sc"]["bias"],
    )

    prior = StructuralPrior(
        parcellation=parc_ctx,
        labels=labels,
        weights=w,
        distance_mm=dist,
        euclidean_mm=euclid,
        evidence=ev,
        consistency=cons,
        functional=f,
        cross_species_support=edr,
        hierarchy_prior=hier,
        delays=ConductionDelayModel(length_mm=dist, length_source=length),
        ledger=ledger,
        provenance={
            "source": S.SRC["enigma_hcp_sc"],
            "atlas": atlas,
            "include_subcortex": include_subcortex,
            "length": length,
            "n_negative_weights_floored": n_neg,
            "streams": [
                {k: v for k, v in st.items() if k != "present"} for st in streams
            ],
            "classification": {
                "long_mm": _LONG_MM,
                "hard_consistency": _HARD_CONSISTENCY,
                "hard_long_weight_pctl": _HARD_LONG_WEIGHT_PCTL,
                "proposed_fc": _PROPOSED_FC,
                "proposed_mm": _PROPOSED_MM,
            },
        },
        n_streams=len(streams),
        stream_names=tuple(names),
    )
    prior.save(cache)
    return prior
