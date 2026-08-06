"""Shared fixtures for the sources tests.

Tests that need real bytes on disk are skipped (never faked) when the dataset
is not present, and the skip message says exactly which dataset is missing.
"""

from __future__ import annotations

import pytest

from scwbd.sources import registry


def require_dataset(dataset_id: str):
    """Skip unless the dataset's bytes are on the local data root."""
    entry = registry.get(dataset_id)
    if not entry.local_path.exists():
        pytest.skip(f"{dataset_id} not on disk at {entry.local_path}")
    return entry


def require_manifest(dataset_id: str):
    entry = require_dataset(dataset_id)
    if not entry.has_manifest():
        pytest.skip(f"{dataset_id} has no manifest at {entry.manifest_path}")
    return entry


@pytest.fixture(scope="session")
def cards():
    from scwbd.sources.cards import load_all_cards

    return load_all_cards()


@pytest.fixture(scope="session")
def entries():
    return dict(registry.REGISTRY)
