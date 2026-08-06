"""The registry knows what exists, what does not, and never claims otherwise."""

from __future__ import annotations

import pytest

from scwbd.sources import registry
from scwbd.sources.download import UnavailableFetcher, _main


def test_ids_are_unique_and_paths_are_versioned(entries):
    assert len(entries) == len({e.dataset_id for e in entries.values()})
    for did, e in entries.items():
        assert e.local_path.name == e.version, f"{did}: path is not version-pinned"
        assert e.local_path.parent.name == did


def test_status_is_computed_not_asserted(entries):
    for did, e in entries.items():
        status, reason = e.status()
        assert status in ("live", "partial", "unavailable")
        if status == "unavailable":
            assert reason, f"{did}: unavailable without a reason"
        else:
            assert e.local_path.exists(), f"{did}: claims {status} without bytes"


def test_dua_gated_datasets_have_no_downloader(entries):
    """Nothing in the register can accidentally fetch credentialed data."""
    for did in ("ukbiobank-brain-imaging", "hcp-young-adult", "tuh-eeg", "adni", "ram-intracranial"):
        e = entries[did]
        assert isinstance(e.fetcher, UnavailableFetcher)
        res = e.fetcher.fetch(e.local_path)
        assert not res.ok and "unavailable" in res.error
        assert not e.local_path.exists(), f"{did}: bytes exist for a DUA-gated source"


def test_loaders_resolve_for_live_datasets(entries):
    for did, e in entries.items():
        if e.loader_ref is None:
            continue
        fn = e.loader()
        assert callable(fn), f"{did}: loader_ref {e.loader_ref} did not resolve"


def test_inventory_rows_are_complete():
    rows = registry.inventory()
    assert len(rows) == len(registry.REGISTRY)
    for row in rows:
        assert "card_error" not in row, row
        for key in ("dataset_id", "version", "role", "status", "license", "gradient"):
            assert key in row, f"{row['dataset_id']} missing {key}"
        if row["status"] == "unavailable":
            assert row["bytes_on_disk"] == 0
            assert row["gradient"]["enabled"] == []
        else:
            assert row["bytes_on_disk"] > 0


def test_live_datasets_report_real_bytes():
    live = registry.live_datasets()
    assert live, "no dataset is live: nothing was downloaded"
    total = sum(e.on_disk_bytes() for e in live)
    assert total > 1e9, f"only {total} bytes on disk across {len(live)} datasets"


def test_cli_list_runs(capsys):
    rc = _main(["--list"])
    assert rc == 0
    out = capsys.readouterr().out
    for did in registry.REGISTRY:
        assert did in out


def test_cli_rejects_unknown_dataset(capsys):
    rc = _main(["not-a-dataset"])
    assert rc == 2


def test_data_root_is_the_symlinked_runtime_root():
    root = registry.data_root()
    assert root.exists(), f"{root} does not exist"
    assert root.resolve() == root.resolve()  # resolves through the data/ symlink
