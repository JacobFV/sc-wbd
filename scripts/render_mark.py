"""Render the WBD mark as a static figure for the paper.

Same geometry as the site's canvas: 414 parcel positions from
AnatomyPrior.positions in fsLR_32k surface RAS, coloured by the 9-family
partition. The site version is draggable; a PDF cannot be, so this fixes the
view at the same default the canvas starts from.

    PYTHONPATH=. .venv/bin/python scripts/render_mark.py
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scwbd.foundation.anatomy import load_anatomy

# Identical to COL in site/static/brain.js. Two cortical families in blues,
# subcortex warm: the split that carries meaning is cortex-vs-subcortex, which
# stays legible under a red-green deficiency.
COL = np.array([
    [96, 165, 232], [52, 110, 190],
    [232, 156, 74], [225, 133, 92], [214, 160, 58],
    [236, 176, 102], [204, 122, 70], [244, 194, 122], [214, 145, 48],
]) / 255.0

YAW, PITCH = -0.5, -0.18       # brain.js initial view


def main() -> int:
    a = load_anatomy(device="cpu")
    P = np.asarray(a.positions, dtype=float)
    P = P - P.mean(0)
    P = P / np.abs(P).max()

    # `a.families` is keyed by NAME, not position -- indexing it with an int
    # raises. Materialise the list once and index that.
    fams = list(a.families)
    fam_of = {}
    for i, f in enumerate(fams):
        for q in f.parcels:
            fam_of[int(q)] = i
    fam = np.array([fam_of.get(i, 0) for i in range(len(P))])
    is_sub = np.array([fams[k].division != "cortex" for k in fam])

    cy, sy = np.cos(YAW), np.sin(YAW)
    cp, sp = np.cos(PITCH), np.sin(PITCH)
    X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
    x1 = X * cy + Y * sy
    y1 = -X * sy + Y * cy
    y2 = y1 * cp - Z * sp
    z2 = y1 * sp + Z * cp

    order = np.argsort(y2)                       # painter's algorithm, far first
    t = (y2 - y2.min()) / max(float(np.ptp(y2)), 1e-9)    # 0 far .. 1 near

    fig, ax = plt.subplots(figsize=(4.6, 3.45), dpi=400)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.scatter(
        x1[order], z2[order],
        s=np.where(is_sub, 34, 15)[order] * (0.55 + 0.7 * t[order]),
        c=COL[fam[order]],
        alpha=None,
        linewidths=0,
        edgecolors="none",
        zorder=2,
    )
    # Depth as alpha, applied per point (scatter's alpha is scalar-only).
    ax.collections[0].set_alpha(None)
    rgba = np.zeros((len(P), 4))
    rgba[:, :3] = COL[fam[order]]
    rgba[:, 3] = 0.32 + 0.60 * t[order]
    ax.collections[0].set_facecolors(rgba)

    ax.set_aspect("equal")
    ax.axis("off")
    m = 1.12
    ax.set_xlim(-m, m); ax.set_ylim(-m, m)
    fig.tight_layout(pad=0.05)

    out = pathlib.Path("paper/figures")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out / f"wbd_mark.{ext}"
        fig.savefig(p, transparent=True, bbox_inches="tight", pad_inches=0.02)
        print(f"  wrote {p} ({p.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
