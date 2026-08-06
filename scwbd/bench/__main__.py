"""``python -m scwbd.bench`` -> run every claim gate and write the scoreboard."""

from __future__ import annotations

from .runner import main

if __name__ == "__main__":
    raise SystemExit(main())
