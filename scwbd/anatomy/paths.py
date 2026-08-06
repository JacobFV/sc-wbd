"""Asset-root resolution for :mod:`scwbd.anatomy`.

Layout (``assets/`` is a real directory in the repository whose subdirectories
are symlinks onto ``/data/scwbd/assets`` so that binaries stay off the working
tree while ``assets/MANIFEST.json`` remains git-tracked)::

    assets/
      MANIFEST.json     tracked: sha256 + source URL + license + version
      .gitignore        ignores everything but the manifest
      src/    ->  /data/scwbd/assets/src      upstream repositories, verbatim
      cache/  ->  /data/scwbd/assets/cache    downloader caches (nilearn, ...)
      derived/-> /data/scwbd/assets/derived   our own .npz/.h5 build products

Override the root with ``SCWBD_ASSETS``.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "assets_root",
    "src_dir",
    "cache_dir",
    "derived_dir",
    "manifest_path",
    "set_download_env",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def assets_root() -> Path:
    """Return the asset root, honouring ``$SCWBD_ASSETS``."""
    env = os.environ.get("SCWBD_ASSETS")
    root = Path(env).expanduser() if env else _REPO_ROOT / "assets"
    return root


def src_dir() -> Path:
    return assets_root() / "src"


def cache_dir() -> Path:
    return assets_root() / "cache"


def derived_dir(*parts: str) -> Path:
    p = assets_root() / "derived"
    for part in parts:
        p = p / part
    return p


def manifest_path() -> Path:
    return assets_root() / "MANIFEST.json"


def set_download_env() -> None:
    """Point third-party downloaders at our cache before they are imported.

    ``nilearn``, ``templateflow``, ``neuromaps`` and ``netneurotools`` each read
    an environment variable at import time.  Call this before importing them if
    you want their caches inside the asset root rather than ``~/``.
    """
    c = cache_dir()
    for var, sub in (
        ("NILEARN_DATA", "nilearn"),
        ("TEMPLATEFLOW_HOME", "templateflow"),
        ("NEUROMAPS_DATA", "neuromaps"),
        ("NNT_DATA", "netneurotools"),
    ):
        os.environ.setdefault(var, str(c / sub))
        (c / sub).mkdir(parents=True, exist_ok=True)
