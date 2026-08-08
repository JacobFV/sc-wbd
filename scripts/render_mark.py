"""Render the WBD mark as a static figure for the paper.

Same geometry as the site's canvas: 414 parcel positions from
AnatomyPrior.positions in fsLR_32k surface RAS, coloured by the 9-family
partition, over the strongest edges of the measured connectome. The site
version is draggable; a PDF cannot be, so this fixes the view at the same
default the canvas starts from.

The tracts are the point of the drawing and were missing from it. A cloud of
dots says the brain has regions; the same cloud with its connectome says the
regions are coupled, which is the whole claim of a whole-brain dynamics model.
The edge set is derived here from ``AnatomyPrior.weights`` by the same rule the
site states -- the strongest ``N_EDGES`` -- rather than read out of the site's
JSON, so the figure and the page agree because they are computed the same way
and not because one copies the other.

The ring around it names what the model is for. The same list, in the same
order, is drawn around the site's hero canvas by ``site/static/arch.js``
(``USES``); the two are the model's use cases stated once in two media, so a
reader who sees the cover and then the page sees one figure.

    PYTHONPATH=. .venv/bin/python scripts/render_mark.py
"""
from __future__ import annotations

import pathlib

import matplotlib
import matplotlib.collections
import matplotlib.patheffects
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
N_EDGES = 900                  # "the strongest 900 edges", as the site says

MARK = "SC-WBD-003"

# What the model is for, clockwise from the top. Kept in the order the family
# groups run -- estimation, decoding, mapping, perturbation, control, over time
# -- so neighbours on the ring are neighbours in subject matter.
USES = [
    "neural state estimation",
    "whole-brain forecasting",
    "cross-modal prediction",
    "EEG source inference",
    "EEG computer control",
    "neural error detection",
    "motor-intent decoding",
    "language decoding",
    "cognitive-state decoding",
    "individualized brain mapping",
    "functional network mapping",
    "connectivity inference",
    "perturbation forecasting",
    "TMS target selection",
    "TMS response prediction",
    "tFUS target selection",
    "tFUS response prediction",
    "closed-loop neuromodulation",
    "neurofeedback control",
    "cognitive intervention design",
    "longitudinal brain modeling",
    "personalized digital twins",
    "behavioral forecasting",
]

# The ring, in data units. `SCALE` is inches per data unit, so a font size set
# in points below is that size on the page when the figure is included at its
# natural width -- the label type is specified where it is read, not guessed
# backwards through an \includegraphics scale factor.
SCALE = 0.70
RX, RY = 2.62, 1.42            # where the labels sit
RX_IN, RY_IN = 1.26, 1.00      # where the leader lines start, just off the brain
BX, BZ = 1.16, 0.92            # the ellipse the projected brain is fitted into
XLIM, YLIM = 4.95, 1.68


def ring_points(n: int) -> np.ndarray:
    """`n` points spaced by equal ARC LENGTH around the label ellipse.

    Equal *angle* is the obvious thing and the wrong one: on an ellipse this
    wide, equal steps in theta crowd the points vertically at the left and
    right ends, which is exactly where the labels are horizontal lines of text
    that then collide. Equal arc length spaces them by the distance actually
    seen between them.
    """
    th = np.linspace(0.0, 2 * np.pi, 4000)
    x, y = RX * np.cos(th), RY * np.sin(th)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    want = np.linspace(0.0, s[-1], n, endpoint=False)
    # Clockwise from the top: start at theta = pi/2 and run backwards.
    return np.pi / 2 - np.interp(want, s, th)


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

    # Fit the projected brain inside the inner ellipse the leader lines stop
    # at, rather than inside a square: the frame is wide and short, and a
    # square fit leaves the brain sized by a corner nothing is drawn in.
    k = 1.0 / max(float(np.hypot(x1 / BX, z2 / BZ).max()), 1e-9)
    x1, z2 = x1 * k, z2 * k

    order = np.argsort(y2)                       # painter's algorithm, far first
    t = (y2 - y2.min()) / max(float(np.ptp(y2)), 1e-9)    # 0 far .. 1 near

    fig, ax = plt.subplots(figsize=(2 * XLIM * SCALE, 2 * YLIM * SCALE), dpi=400)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    # ---- tracts, under the parcels -------------------------------------
    #
    # Upper triangle only: the compiled connectome is symmetric here, and
    # drawing both triangles would lay every line down twice, doubling its
    # opacity for no information.
    W = np.asarray(a.weights, dtype=float)
    W = np.triu(W, k=1)
    n_edges = min(N_EDGES, int((W > 0).sum()))
    if n_edges:
        flat = np.argpartition(W.ravel(), -n_edges)[-n_edges:]
        ia, ib = np.unravel_index(flat, W.shape)
        ew = W[ia, ib]
        ew = ew / max(float(ew.max()), 1e-12)

        # Depth of each edge's midpoint, so tracts at the back recede the same
        # way the parcels do.
        depth = (t[ia] + t[ib]) / 2.0
        segs = np.stack(
            [np.column_stack([x1[ia], z2[ia]]),
             np.column_stack([x1[ib], z2[ib]])],
            axis=1,
        )
        rgba_e = np.zeros((n_edges, 4))
        rgba_e[:, :3] = 0.42                      # neutral grey linework
        rgba_e[:, 3] = 0.05 + 0.30 * ew * depth
        ax.add_collection(
            matplotlib.collections.LineCollection(
                segs, colors=rgba_e, linewidths=0.22, zorder=1
            )
        )

    dots = ax.scatter(
        x1[order], z2[order],
        s=np.where(is_sub, 19, 8.4)[order] * (0.55 + 0.7 * t[order]),
        c=COL[fam[order]],
        alpha=None,
        linewidths=0,
        edgecolors="none",
        zorder=2,
    )
    # Depth as alpha, applied per point (scatter's alpha is scalar-only).
    # Held by handle rather than by `ax.collections[0]`: with the tracts added
    # first, index 0 is the LineCollection, and 414 face colours applied to 900
    # line segments is not an error anything would report.
    dots.set_alpha(None)
    rgba = np.zeros((len(P), 4))
    rgba[:, :3] = COL[fam[order]]
    rgba[:, 3] = 0.32 + 0.60 * t[order]
    dots.set_facecolors(rgba)

    # ---- the name, over the middle ---------------------------------------
    #
    # White stroke behind the glyphs rather than a filled box: the figure is
    # saved transparent, and a box would print as a rectangle of paper laid
    # over the connectome.
    ax.text(
        0, 0, MARK, ha="center", va="center", zorder=4,
        fontsize=13.5, fontweight="bold", color="#1d1c21",
        family="sans-serif",
        path_effects=[matplotlib.patheffects.withStroke(linewidth=3.4,
                                                        foreground="white")],
    )

    # ---- the ring: what the model is for ---------------------------------
    th = ring_points(len(USES))
    for name, a_ in zip(USES, th):
        ct, st = float(np.cos(a_)), float(np.sin(a_))
        x0, y0 = RX_IN * ct, RY_IN * st
        x, y = RX * ct, RY * st
        ax.plot([x0, x * 0.985], [y0, y * 0.985],
                color="#8c9bad", linewidth=0.4, alpha=0.85,
                solid_capstyle="round", zorder=3)
        ax.plot([x], [y], marker="o", markersize=1.5, color="#8c9bad",
                markeredgewidth=0, zorder=3)
        # Straight up and down at the ends of the ring, the label reads as a
        # caption over or under it; along the sides it reads outward.
        if ct > 0.12:
            ha, dx = "left", 0.09
        elif ct < -0.12:
            ha, dx = "right", -0.09
        else:
            ha, dx = "center", 0.0
        va = "center" if abs(ct) > 0.12 else ("bottom" if st > 0 else "top")
        dy = 0.0 if abs(ct) > 0.12 else (0.07 if st > 0 else -0.07)
        ax.text(x + dx, y + dy, name, ha=ha, va=va, fontsize=6.6,
                color="#3d3c43", family="sans-serif", zorder=4)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-XLIM, XLIM); ax.set_ylim(-YLIM, YLIM)
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
