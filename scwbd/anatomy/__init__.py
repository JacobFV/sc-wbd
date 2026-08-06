"""Adult human anatomical priors for SC-WBD-001-beta.

The *probabilistic interaction grammar*: parcellations, cortical geometry,
a group structural connectome with hard/soft/proposed edge classification,
conduction delays as priors rather than point estimates, receptor and gradient
maps with uncertainty ledgers, and the null connectomes that make claim gate
G2 ("anatomy improves inference") testable.

Everything in this package is a **group average**.  Nothing in it is any
particular person's brain, and every object says so in its ledger.

Entry points
------------
:func:`~scwbd.anatomy.atlases.load_parcellation`
    Parcellations with vertex/voxel assignment, centroids, areas, provenance.
:func:`~scwbd.anatomy.geometry.parcel_geometry`
    Geodesic and Euclidean inter-parcel distances, adjacency.
:func:`~scwbd.anatomy.connectome.load_structural_prior`
    The connectome as a typed topology prior plus the five G2 controls.
:func:`~scwbd.anatomy.maps.load_maps`
    Receptor, myelin, thickness, gradient and timescale maps.
:class:`~scwbd.anatomy.priors.BrainPrior`
    All of the above, assembled for agents E and I.
"""

from __future__ import annotations

from .atlases import (
    Parcellation,
    Provenance,
    available_parcellations,
    crosswalk,
    load_parcellation,
)
from .connectome import (
    CONDUCTION_VELOCITY_PRIOR,
    EDR_LAMBDA_PRIOR,
    TORTUOSITY_PRIOR,
    ConductionDelayModel,
    EdgeEvidence,
    StructuralPrior,
    load_structural_prior,
)
from .geometry import ParcelGeometry, Surface, SurfaceGeometry, load_surface, parcel_geometry
from .manifest import Manifest
from .maps import MapSet, RegionalMap, available_maps, load_maps, receptor_matrix
from .priors import BrainPrior

__all__ = [
    "Parcellation",
    "Provenance",
    "load_parcellation",
    "available_parcellations",
    "crosswalk",
    "Surface",
    "SurfaceGeometry",
    "ParcelGeometry",
    "load_surface",
    "parcel_geometry",
    "StructuralPrior",
    "EdgeEvidence",
    "ConductionDelayModel",
    "load_structural_prior",
    "CONDUCTION_VELOCITY_PRIOR",
    "TORTUOSITY_PRIOR",
    "EDR_LAMBDA_PRIOR",
    "RegionalMap",
    "MapSet",
    "load_maps",
    "available_maps",
    "receptor_matrix",
    "BrainPrior",
    "Manifest",
]
