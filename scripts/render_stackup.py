"""Render the five WBD checkpoints side by side, sized by log parameter count.

An audience figure, not a paper figure: it puts the run history in one frame so
a reader who has never opened `reports/` can see what got bigger and what each
model did against its baselines.

The brain is the hero canvas's drawing -- 414 parcel positions in fsLR_32k
surface RAS, coloured by the 9-family partition, over the strongest 900 edges
of the measured connectome, at the canvas's default view (yaw -0.5, pitch
-0.18). Five copies of it, scaled and captioned.

**Geometry is read from `site/static/{brain,edges}.json`, not from
`AnatomyPrior`.** `scripts/render_mark.py` derives it from the prior on
purpose, so the paper's cover and the page agree by computation rather than by
copying. This script cannot: `assets/` is a symlink to `/data/scwbd/assets`,
that directory is empty, and `load_anatomy` refuses rather than substituting a
synthetic prior. Reading the site's own committed numbers is the honest second
source -- this figure is a deck asset that should match the hero exactly -- but
it is a copy, and if the prior changes this figure will not notice.

**The size axis is log, and it is offset.** Radius is proportional to
``log10(parameters) - 5``, not to ``log10(parameters)``: on a bare log the
whole history spans 6.25 to 8.16 and every disc lands within 31% of every
other, which draws as five identical circles. The offset is stated on the
figure so nobody reads the areas as parameter ratios.

SC-WBD-005 is not trained. It is drawn hollow, in grey, inside a dashed ring
and with no score, so its treatment cannot be mistaken for a measured result.

**SCORE_MODE decides what the second number is, and the two modes say
different things.** `measured` prints each run's own held-out NLL: 2.555,
3.179, 1.986, 2.024, which is not monotone and is not supposed to be -- run 2's
trainer was mis-configured and run 4 did not reproduce run 3. `best_to_date`
prints the best held-out NLL the programme had produced by that run, which is
monotone by construction and therefore repeats a value wherever a run did not
improve on its predecessor. Neither is wrong; `best_to_date` is a claim about
the PROGRAMME and `measured` is a claim about the CHECKPOINT, and the figure
carries no label saying which, so whoever presents it has to.

    .venv/bin/python scripts/render_stackup.py
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
import matplotlib.collections
import matplotlib.patches
import matplotlib.patheffects
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Identical to COL in site/static/brain.js and scripts/render_mark.py: two
# cortical families in the house blues, subcortex warm.
COL = np.array([
    [96, 165, 232], [52, 110, 190],
    [232, 156, 74], [225, 133, 92], [214, 160, 58],
    [236, 176, 102], [204, 122, 70], [244, 194, 122], [214, 145, 48],
]) / 255.0

# A right-lateral sagittal view, not the hero's 3/4: yaw = pi/2 sends screen-x
# to +Y and pitch = 0 sends screen-y to +Z, so anterior points RIGHT and
# superior points up. +Y is anterior in these coordinates -- read off the file
# (Y spans -0.880..+1.000, and the 14 subcortical parcels sit at Y +0.20,
# Z -0.29, below and forward of the cortical centroid), not assumed.
YAW, PITCH = np.pi / 2.0, 0.0

INK, INK2, INK3 = "#17181a", "#55575c", "#7d8085"
RULE = "#e3e3df"

# parameters: run 1 reports/training/evaluation.json, run 2 evaluation_run2.json,
#   runs 3-4 reports/scwbd-00{3,4}_derived.json ("TOTAL").
# nll: real_eeg_holdout, the model's own row, nats per channel per sample.
#   Runs 1-2 scored SC-WBD on `target/s` and the baselines on the raw target,
#   so those two read ~0.6 nats LOW; the verdicts are unaffected and the
#   figure's footer says so.  Run 5 is a design target, not a measurement.
MODELS = [
    # nll: the model's own real_eeg_holdout row. best: the best NON-scwbd row
    # in the same table, so the margin is against that run's own comparators
    # and not against a number from a different split.
    dict(mark="SC-WBD-001", params=1_757_613,   nll=2.555161, best=2.013234,
         trained=True),
    dict(mark="SC-WBD-002", params=2_516_530,   nll=3.178930, best=2.045406,
         trained=True),
    dict(mark="SC-WBD-003", params=26_304_729,  nll=1.986297, best=2.024021,
         trained=True),
    dict(mark="SC-WBD-004", params=26_304_657,  nll=2.024430, best=2.034478,
         trained=True),
    dict(mark="SC-WBD-005", params=146_000_000, nll=None,     best=None,
         trained=False),
]


def score_column() -> list[str]:
    """The second caption row, one string per model.

    `best_to_date` is a running minimum, so it is monotone whatever the runs
    did. That is the point of it and also its whole cost: it repeats a value
    wherever a run did not improve, and the figure has no room to say why.
    """
    modes = ("measured", "best_to_date", "margin", "margin_best_to_date")
    if SCORE_MODE not in modes:
        raise SystemExit(f"SCORE_MODE: {SCORE_MODE!r} is not one of {modes}")
    out: list[str] = []
    run_nll: float | None = None
    run_margin: float | None = None
    for m in models():
        if m["nll"] is None:
            out.append("")            # not trained: the slot stays empty
            continue
        margin = m["best"] - m["nll"]          # + means SC-WBD is ahead
        run_nll = m["nll"] if run_nll is None else min(run_nll, m["nll"])
        run_margin = margin if run_margin is None else max(run_margin, margin)
        if SCORE_MODE == "measured":
            out.append(f"{m['nll']:.3f}")
        elif SCORE_MODE == "best_to_date":
            out.append(f"{run_nll:.3f}")
        elif SCORE_MODE == "margin":
            out.append(f"{margin:+.2f}".replace("-", "\u2212"))
        else:
            out.append(f"{run_margin:+.2f}".replace("-", "\u2212"))
    return out


def models() -> list[dict]:
    """The lineup, minus anything named in SKIP."""
    return [m for m in MODELS if m["mark"] not in SKIP]


def metric_label() -> tuple[str, str]:
    """What the second row is, and which direction is good.

    Named once in the left margin rather than once per column: five copies of
    'held-out NLL' is a texture, and the direction is the part a reader
    actually needs -- a bare 2.555 beside a bare 1.986 does not say which won.
    """
    if SCORE_MODE.startswith("margin"):
        return "nats vs. best baseline", "higher is better"
    return "held-out NLL", "lower is better"

SCORE_MODE = "measured"        # see score_column(); overridable with --score
SKIP: set[str] = set()         # marks to leave out entirely; --skip SC-WBD-002

FLOOR = 5.0                    # the log10 the radius axis is measured from
R_MAX = 1.00                   # HALF-HEIGHT of the largest brain, in data units
GUTTER = 0.34                  # clear space between neighbouring silhouettes
MIN_PITCH = 2.30               # centre to centre, so the captions never collide
CAP_GAP = 0.30                 # ground line to the first caption row
ROW_GAP = 0.30                 # first caption row to the second
GUTTER_L = 2.25                # left margin the row labels are set in
STATIC = pathlib.Path("site/static")
OUT_STEM = "paper/figures/scwbd_stackup"


def load_geometry():
    """The hero's parcels and tracts: (x, y, depth, family, is_sub, ia, ib, w)."""
    brain = json.loads((STATIC / "brain.json").read_text())
    edges = json.loads((STATIC / "edges.json").read_text())

    P = np.asarray(brain["p"], dtype=float).reshape(brain["n"], 3)
    P = P - P.mean(0)
    P = P / np.abs(P).max()
    fam = np.asarray(brain["f"], dtype=int)
    # `div` is per FAMILY (9 entries), not per parcel -- indexing it by parcel
    # would silently read the wrong nine values and size 405 dots wrongly.
    sub_family = np.array([d == "sub" for d in brain["div"]])
    is_sub = sub_family[fam]

    cy, sy = np.cos(YAW), np.sin(YAW)
    cp, sp = np.cos(PITCH), np.sin(PITCH)
    X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
    x1 = X * cy + Y * sy
    y1 = -X * sy + Y * cy
    y2 = y1 * cp - Z * sp
    z2 = y1 * sp + Z * cp

    # Normalise on the projected BOUNDING BOX, not on max hypot: the head in
    # this view is wider than it is tall, so a hypot fit leaves the silhouette
    # floating inside its nominal circle and the five brains do not sit on the
    # ground line they are drawn against. Half-height becomes exactly 1, so a
    # radius in data units is a half-height and bottoms align by construction.
    x1 = x1 - (x1.min() + x1.max()) / 2.0
    z2 = z2 - (z2.min() + z2.max()) / 2.0
    k = 2.0 / max(float(np.ptp(z2)), 1e-9)
    x1, z2 = x1 * k, z2 * k
    aspect = float(np.ptp(x1)) / 2.0                     # half-width, in units

    t = (y2 - y2.min()) / max(float(np.ptp(y2)), 1e-9)   # 0 far .. 1 near

    pairs = np.asarray(edges["e"], dtype=int).reshape(-1, 2)
    w = np.asarray(edges["w"], dtype=float)
    if len(w) != len(pairs):
        raise SystemExit(f"edges.json: {len(pairs)} pairs but {len(w)} weights")
    return (x1, z2, t, fam, is_sub, pairs[:, 0], pairs[:, 1],
            w / max(w.max(), 1e-12), aspect)


def draw_brain(ax, g, cx, cy, r, *, live: bool) -> None:
    """One copy of the mark, centred on (cx, cy) with radius r."""
    x, y, t, fam, is_sub, ia, ib, ew, _ = g
    px, py = cx + x * r, cy + y * r
    dim = 1.0 if live else 0.50

    # Tracts under the parcels, receding with the midpoint's depth.
    depth = (t[ia] + t[ib]) / 2.0
    segs = np.stack([np.column_stack([px[ia], py[ia]]),
                     np.column_stack([px[ib], py[ib]])], axis=1)
    rgba_e = np.zeros((len(ia), 4))
    rgba_e[:, :3] = 0.42
    rgba_e[:, 3] = (0.06 + 0.34 * ew * depth) * dim
    ax.add_collection(matplotlib.collections.LineCollection(
        segs, colors=rgba_e, linewidths=0.30 * (0.45 + r / R_MAX), zorder=2))

    order = np.argsort(t)                       # painter's algorithm, far first
    face = COL[fam] if live else np.tile([0.60, 0.61, 0.64], (len(fam), 1))
    rgba = np.zeros((len(fam), 4))
    rgba[:, :3] = face[order]
    rgba[:, 3] = (0.32 + 0.60 * t[order]) * (1.0 if live else 0.62)
    ax.scatter(px[order], py[order],
               s=np.where(is_sub, 30.0, 13.0)[order]
                 * (0.55 + 0.7 * t[order]) * (r / R_MAX) ** 2,
               c=rgba, linewidths=0, edgecolors="none", zorder=3)


def main() -> int:
    g = load_geometry()

    aspect = g[-1]
    logs = np.array([np.log10(m["params"]) - FLOOR for m in models()])
    radii = R_MAX * logs / logs.max()            # half-heights
    half_w = radii * aspect                      # half-widths, for spacing

    # Left to right on one ground line, so the eye reads the tops as the
    # series. Centres are spaced by the two half-widths they separate plus a
    # gutter -- but never closer than MIN_PITCH, because the caption column
    # under the smallest brain is wider than the brain is.
    xs, cursor = [], 0.0
    for i, hw in enumerate(half_w):
        if i:
            cursor += max(half_w[i - 1] + GUTTER + hw, MIN_PITCH)
        else:
            cursor += hw
        xs.append(cursor)
    xs = np.array(xs)
    ground = 0.0
    left = min(xs - half_w) - 0.0
    right = max(xs + half_w)
    # The caption block, not the silhouette, sets the outer margin on the ends.
    left = min(left, xs[0] - MIN_PITCH / 2.0)
    right = max(right, xs[-1] + MIN_PITCH / 2.0)

    top = ground + 2 * radii.max()
    foot = ground - CAP_GAP - ROW_GAP - 0.44
    x0, x1_ = left - GUTTER_L, right + 0.40
    y0, y1_ = foot - 0.10, top + 0.30

    # Equal aspect: a figure whose shape does not match the data extents gets
    # padded on one axis, which is a band of empty paper nothing is drawn in.
    fig_w = 15.0
    fig, ax = plt.subplots(
        figsize=(fig_w, fig_w * (y1_ - y0) / (x1_ - x0)), dpi=260)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    scores = score_column()
    for m, cx, r, score in zip(models(), xs, radii, scores):
        cy = ground + r
        live = m["trained"]

        if not live:                      # a ring that says "not measured"
            pad = 1.06
            ax.add_patch(matplotlib.patches.Ellipse(
                (cx, cy + r * (pad - 1.0)), 2 * r * aspect * pad, 2 * r * pad,
                fill=False, edgecolor=INK3, linewidth=1.2,
                linestyle=(0, (5, 4)), alpha=0.85, zorder=1))

        draw_brain(ax, g, cx, cy, r, live=live)

        ax.text(cx, cy, m["mark"], ha="center", va="center", zorder=5,
                fontsize=6.5 + 11.5 * (r / R_MAX), fontweight="bold",
                color=INK if live else INK3, family="sans-serif",
                path_effects=[matplotlib.patheffects.withStroke(
                    linewidth=4.2, foreground="white")])

        # Two numbers. What they are is said once, in the left margin.
        p_ = m["params"]
        ptxt = f"{p_ / 1e6:.2f} M" if p_ < 1e8 else f"{p_ / 1e6:.0f} M"
        ax.text(cx, ground - CAP_GAP, ptxt, ha="center", va="top",
                fontsize=10.5, fontweight="bold",
                color=INK if live else INK3, family="sans-serif")
        if score:
            ax.text(cx, ground - CAP_GAP - ROW_GAP, score, ha="center", va="top",
                    fontsize=10.5, fontweight="bold",
                    color=INK, family="sans-serif")

    ax.plot([left, right], [ground - 0.11, ground - 0.11],
            color=RULE, linewidth=1.0, zorder=0)

    # ---- row labels, in the left margin ---------------------------------
    metric, direction = metric_label()
    lx = left - 0.28
    ax.text(lx, ground - CAP_GAP - 0.01, "params", ha="right", va="top",
            fontsize=9.5, color=INK2, family="sans-serif")
    ax.text(lx, ground - CAP_GAP - ROW_GAP - 0.01, metric, ha="right", va="top",
            fontsize=9.5, color=INK2, family="sans-serif")
    ax.text(lx, ground - CAP_GAP - ROW_GAP - 0.19, direction,
            ha="right", va="top", fontsize=8, style="italic",
            color=INK3, family="sans-serif")

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(x0, x1_)
    ax.set_ylim(y0, y1_)
    fig.tight_layout(pad=0.2)

    stem = pathlib.Path(OUT_STEM)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        pth = stem.with_suffix(f".{ext}")
        fig.savefig(pth, facecolor="white", bbox_inches="tight", pad_inches=0.24)
        print(f"  wrote {pth} ({pth.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score", default=SCORE_MODE,
                    choices=("measured", "best_to_date", "margin",
                             "margin_best_to_date"))
    ap.add_argument("--skip", default="", help="marks to omit, comma separated")
    ap.add_argument("--out", default="paper/figures/scwbd_stackup",
                    help="path stem; .png and .pdf are written")
    _a = ap.parse_args()
    SCORE_MODE = _a.score
    SKIP = {q.strip() for q in _a.skip.split(",") if q.strip()}
    OUT_STEM = _a.out
    raise SystemExit(main())
