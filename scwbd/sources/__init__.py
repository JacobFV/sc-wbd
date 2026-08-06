"""SC-WBD data substrate: source cards, dataset registry, downloaders, splits.

Entry points
------------
``scwbd.sources.registry``   what exists, where, which release, and its state
``scwbd.sources.cards``      machine-readable source cards (Appendix B)
``scwbd.sources.download``   resumable fetchers + checksum verification (CLI)
``scwbd.sources.splits``     leakage-safe grouped splitting + audit (Appendix D)
``scwbd.sources.loaders``    native-support loaders (no resampling, ever)
"""

from .cards import CardError, SourceCardDoc, load_all_cards, load_card
from .lineage import Lineage, LineageError, Record
from .splits import GroupedSplitter, LeakageReport, Split, leakage_audit

__all__ = (
    "CardError",
    "GroupedSplitter",
    "LeakageReport",
    "Lineage",
    "LineageError",
    "Record",
    "Split",
    "SourceCardDoc",
    "leakage_audit",
    "load_all_cards",
    "load_card",
)
