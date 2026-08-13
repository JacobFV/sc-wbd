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
import matplotlib.patches
import matplotlib.path
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

# The current released checkpoint, drawn at the centre of the cover figure. The
# site's `arch.js` carries the same string for the same drawing on the page, and
# tests/release/test_site_names_the_current_checkpoint.py holds the two together
# -- they disagreed for the whole of run 4's release.
MARK = "SC-WBD-004"

# What the model is for: two columns flanking the brain, each a few
# constellations rather than one list. The grouping is the argument -- reading
# down the left is estimate, then decode, then map; down the right is perturb,
# then control, then follow over time -- and the gap between clusters is what
# says so. Each cluster joins the brain through one hub.
LEFT = [
    ["neural state estimation",
     "whole-brain forecasting",
     "cross-modal prediction",
     "EEG source inference"],
    ["EEG computer control",
     "neural error detection",
     "motor-intent decoding",
     "language decoding",
     "cognitive-state decoding"],
    ["individualized brain mapping",
     "functional network mapping",
     "connectivity inference"],
]
RIGHT = [
    ["perturbation forecasting",
     "TMS target selection",
     "TMS response prediction",
     "tFUS target selection",
     "tFUS response prediction"],
    ["closed-loop neuromodulation",
     "neurofeedback control",
     "cognitive intervention design"],
    ["longitudinal brain modeling",
     "personalized digital twins",
     "behavioral forecasting"],
]

# The layout, in data units. `SCALE` is inches per data unit, so a font size
# set in points below is that size on the page when the figure is included at
# its natural width -- the label type is specified where it is read, not
# guessed backwards through an \includegraphics scale factor.
SCALE = 0.70
COLX = 2.02                    # where a column's labels begin, either side
HUB = 1.72                     # where its clusters gather
PITCH = 0.262                  # one label to the next, down a column
GAP = 0.62                     # extra, between clusters
BX, BZ = 1.44, 1.16            # the ellipse the projected brain is fitted into
XLIM, YLIM = 4.90, 1.82


def reach(dx: float, dy: float, x1: np.ndarray, z2: np.ndarray) -> float:
    """How far the parcels reach from the centre along ``(dx, dy)``.

    The silhouette along that ray, not the fitted ellipse: the ellipse is a
    bound, so in most directions it stands clear of the dots and a leader line
    stopped on it ends in white space with the brain still some way off. Only
    parcels within a narrow band of the ray count.
    """
    along = x1 * dx + z2 * dy
    near = np.abs(x1 * dy - z2 * dx) < 0.11
    r = float(along[near].max()) if near.any() else float(along.max())
    return r + 0.03


def column_rows(groups: list[list[str]]) -> tuple[list[float], list[float]]:
    """The y of every label in a column, and of each cluster's hub.

    Centred on zero: the two columns hold 12 and 11 labels, and hanging both
    from a common top would leave the shorter one ending half a cluster above
    the other for no reason a reader could name.
    """
    slots: list[float] = []
    s = 0.0
    for gi, g in enumerate(groups):
        if gi:
            s += GAP
        for _ in g:
            slots.append(s)
            s += 1.0
    span = (s - 1.0) / 2.0
    ys = [(span - q) * PITCH for q in slots]
    hubs, at = [], 0
    for g in groups:
        hubs.append(float(np.mean(ys[at:at + len(g)])))
        at += len(g)
    return ys, hubs


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
        fontsize=15, fontweight="bold", color="#1d1c21",
        family="sans-serif",
        path_effects=[matplotlib.patheffects.withStroke(linewidth=3.8,
                                                        foreground="white")],
    )

    # ---- the two columns: what the model is for --------------------------
    #
    # Every label in a cluster runs as a curve to ONE point on the brain, with
    # the cluster's hub as the control point. The lines bundle where the hub is
    # and arrive together, which is what a cluster means: three labels meeting
    # at a kink and continuing as one straight line said the same thing with a
    # corner in it, and the corner read as a mistake.
    for side, groups in ((-1, LEFT), (+1, RIGHT)):
        ys, hubs = column_rows(groups)
        at = 0
        for g, hy in zip(groups, hubs):
            hx = side * HUB
            # Where the bundle lands: just off the brain's own outline in
            # the direction the cluster comes from.
            ang = np.arctan2(hy, hx)
            dx, dy = float(np.cos(ang)), float(np.sin(ang))
            r = reach(dx, dy, x1, z2)
            tx, ty = dx * r, dy * r
            for name in g:
                y = ys[at]
                at += 1
                ax.add_patch(matplotlib.patches.PathPatch(
                    matplotlib.path.Path(
                        [(side * COLX, y), (hx, hy), (tx, ty)],
                        [matplotlib.path.Path.MOVETO,
                         matplotlib.path.Path.CURVE3,
                         matplotlib.path.Path.CURVE3],
                    ),
                    facecolor="none", edgecolor="#8c9bad", linewidth=0.42,
                    alpha=0.8, capstyle="round", zorder=3,
                ))
                ax.plot([side * COLX], [y], marker="o", markersize=1.4,
                        color="#8c9bad", markeredgewidth=0, zorder=3)
                ax.text(side * (COLX + 0.09), y, name,
                        ha="left" if side > 0 else "right", va="center",
                        fontsize=6.8, color="#3d3c43", family="sans-serif",
                        zorder=4)

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
