"""Shared fixtures for the anatomy tests.

Building a parcellation, its geometry, its maps and its connectome from
upstream data is expensive, so everything is session-scoped and reads the
cached artifacts under ``assets/derived`` when they exist.  Run
``python -m scwbd.anatomy.build`` first if the cache is cold.
"""

from __future__ import annotations

import pytest

from scwbd.anatomy.atlases import load_parcellation
from scwbd.anatomy.connectome import load_structural_prior
from scwbd.anatomy.geometry import parcel_geometry
from scwbd.anatomy.maps import load_maps
from scwbd.anatomy.priors import BrainPrior

#: Small enough to keep the suite fast, large enough to be a real connectome.
SMALL_ATLAS = "Schaefer100x7"
#: The parcellation ARCHITECTURE.md §5 names for the foundation model.
MAIN_ATLAS = "Schaefer400x7"


@pytest.fixture(scope="session")
def parc_small():
    return load_parcellation(SMALL_ATLAS, "fsLR", "32k")


@pytest.fixture(scope="session")
def parc_main():
    return load_parcellation(MAIN_ATLAS, "fsLR", "32k")


@pytest.fixture(scope="session")
def parc_dk():
    return load_parcellation("DesikanKilliany", "fsLR", "32k")


@pytest.fixture(scope="session")
def geom_small(parc_small):
    return parcel_geometry(parc_small)


@pytest.fixture(scope="session")
def maps_small(parc_small):
    return load_maps(parc_small)


@pytest.fixture(scope="session")
def sc_small():
    return load_structural_prior(SMALL_ATLAS, include_subcortex=True)


@pytest.fixture(scope="session")
def sc_dk():
    return load_structural_prior("DesikanKilliany", include_subcortex=True)


@pytest.fixture(scope="session")
def brain_prior():
    return BrainPrior.load(SMALL_ATLAS, include_subcortex=True)


@pytest.fixture(scope="session")
def controls(sc_small):
    return sc_small.controls(seed=1234)
