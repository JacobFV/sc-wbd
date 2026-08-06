"""``scwbd.foundation`` -- the SC-WBD-001-beta network, curriculum and checkpoints.

SC-WBD-001-beta is a **population/general adult** conditional multirate
whole-brain neural operator with an amortized posterior for individualization.
It is **not** a digital twin of anybody, not a clinical device, and not evidence
that any admitted operator is neurally realized (ARCHITECTURE.md §0).

Public surface::

    from scwbd.foundation import SCWBD, load_checkpoint
    model = SCWBD.from_config(FoundationConfig())
"""

from __future__ import annotations

__all__ = [
    "FoundationConfig",
    "SCWBD",
    "StateLayout",
    "default_layout",
    "load_anatomy",
    "ClaimManifest",
    "load_checkpoint",
    "save_checkpoint",
]


def __getattr__(name: str):  # lazy: importing torch-heavy submodules on demand
    if name in ("FoundationConfig",):
        from .config import FoundationConfig

        return FoundationConfig
    if name == "SCWBD":
        from .model import SCWBD

        return SCWBD
    if name in ("StateLayout", "default_layout"):
        from . import state

        return getattr(state, name)
    if name == "load_anatomy":
        from .anatomy import load_anatomy

        return load_anatomy
    if name == "ClaimManifest":
        from .manifest import ClaimManifest

        return ClaimManifest
    if name in ("load_checkpoint", "save_checkpoint"):
        from . import checkpoint

        return getattr(checkpoint, name)
    raise AttributeError(name)
