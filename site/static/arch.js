/* Interactive architecture diagram: how the graph materialises for a use case.
 *
 * Three panels on one canvas.
 *
 *   LEFT    an axonometric scene -- the 414 real parcels and the strongest
 *           tracts between them, plus a cord with terminal nodes standing for
 *           what enters and leaves the body. Slowly rotates about the
 *           superior-inferior axis; drag to orbit, and it resumes rotating on
 *           its own a couple of seconds after you let go.
 *   MIDDLE  the signal categories this use case attaches. HOVER OR CLICK one
 *           and the anatomy it actually attaches to lights up in that
 *           category's colour while everything else drops back. That
 *           interaction is the point of the drawing: "observation" and
 *           "boundary_output" are not two labels, they are two different sets
 *           of things in the head.
 *   RIGHT   what the use case DOES with those signals, as a pipeline you could
 *           argue with -- named stages and the objective at the bottom. The
 *           previous version drew a monitor, a coil and a screen: pictures of
 *           the equipment, which say nothing about how any signal is used.
 *
 * Annotation lines run from the PROJECTED screen position of each dot group to
 * its category, so they move as you orbit.
 *
 * Palette: no ground at all -- the canvas is transparent so the section's own
 * background shows through. Blue-grey linework, one warm accent for anything
 * leaving the body. Extremely fine lines -- 0.6px at 1x -- because the density
 * is the information.
 */
(function () {
  "use strict";

  // Read from the page, not hardcoded, so the drawing follows the device's
  // light/dark preference. A canvas cannot inherit CSS, so the values are
  // resolved at draw time.
  var INK, INK2, LINE, FAINT, WARM, COOL, GROUND;
  function readTheme() {
    var cs = getComputedStyle(document.documentElement);
    function v(name, fallback) {
      var got = cs.getPropertyValue(name).trim();
      return got || fallback;
    }
    INK   = v("--ink", "#27262b");
    INK2  = v("--ink-2", "#55575c");
    LINE  = v("--ink-3", "#8c9bad");
    FAINT = v("--rule", "#c3ccd6");
    // What the canvas sits on. Used only to halo the mark where it crosses the
    // connectome; a filled plate there would read as a card over the brain.
    GROUND = v("--bg", "#ffffff");
    // The two signal hues stay constant across themes: they encode attachment
    // kind, which does not change with the lighting.
    COOL = "#5b7fa6";
    WARM = "#c8874a";
  }

  /* --------------------------------------------------- what the model is for
   *
   * Two columns flanking the brain on a canvas that carries
   * `data-uses="on"`, each a few constellations rather than one list. Same
   * groups, same order, as LEFT and RIGHT in scripts/render_mark.py, which
   * draws them around the paper's cover figure: the page and the cover are one
   * drawing stated in two media.
   *
   * USES_MIN_W is the width below which they cannot go side by side: two
   * stacks of 6px type either side of a thumbnail are a texture, not a list.
   * A canvas that narrow is given a portrait frame by the stylesheet, and the
   * clusters go above and below the brain instead. */
  /* The designation at the centre of the brain. It names the CURRENT released
     checkpoint, so it changes on every release -- it read SC-WBD-003 for the
     whole of run 4's release and was the largest type on the page saying so. */
  var MARK = "SC-WBD-004";
  var USES_MIN_W = 620;                 // CSS px of canvas width
  var LEFT = [
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
     "connectivity inference"]
  ];
  var RIGHT = [
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
     "behavioral forecasting"]
  ];
  var GAP = 0.62;                       // extra pitch between clusters

  /* The y offset of every label in a column, and of each cluster's hub, in
   * units of `pitch` and centred on zero. The two columns hold 12 labels and
   * 11; hanging both from a common top would end the shorter one half a
   * cluster above the other for no reason a reader could name. */
  /* How tall a column is, in pitches: every label but one, plus a gap between
   * clusters. Counted from the lists rather than written down, so adding a use
   * case moves the layout instead of silently overflowing it. */
  function columnSpan(groups) {
    var n = 0;
    for (var i = 0; i < groups.length; i++) n += groups[i].length;
    return (n - 1) + (groups.length - 1) * GAP;
  }

  function columnRows(groups, pitch) {
    var slots = [], s = 0, gi, k;
    for (gi = 0; gi < groups.length; gi++) {
      if (gi) s += GAP;
      for (k = 0; k < groups[gi].length; k++) { slots.push(s); s += 1; }
    }
    var half = (s - 1) / 2;
    var ys = slots.map(function (q) { return (q - half) * pitch; });
    var hubs = [], at = 0;
    for (gi = 0; gi < groups.length; gi++) {
      var sum = 0;
      for (k = 0; k < groups[gi].length; k++) sum += ys[at + k];
      hubs.push(sum / groups[gi].length);
      at += groups[gi].length;
    }
    return { ys: ys, hubs: hubs };
  }

  // Attachment kinds, in the order the schema declares them.
  var USE_CASES = [
    {
      id: "bci",
      label: "Controlling a computer",
      right: "bci",
      cats: [
        { k: "observation", t: "Brain activity", note: "electrodes on the scalp", src: "cortex" },
        { k: "boundary_output", t: "Cursor, keypress", note: "what the person does", src: "motor" },
        { k: "context", t: "Session, fatigue", note: "slow background state", src: "none" }
      ]
    },
    {
      id: "tms",
      label: "Choosing where to stimulate",
      right: "tms",
      cats: [
        { k: "stimulus", t: "Magnetic pulse", note: "energy going in", src: "cortex" },
        { k: "observation", t: "Brain response", note: "electrical, immediate", src: "cortex" },
        { k: "observation", t: "Blood-flow response", note: "slower, whole brain", src: "all" },
        { k: "boundary_output", t: "Muscle twitch", note: "measured at the hand", src: "motor" }
      ]
    },
    {
      id: "semantic",
      label: "Reading what someone responds to",
      right: "semantic",
      cats: [
        { k: "stimulus", t: "What they watched", note: "video, audio, text", src: "sensory" },
        { k: "observation", t: "Blood-flow response", note: "whole brain", src: "all" },
        { k: "boundary_output", t: "Where they looked", note: "eye tracking", src: "sensory" },
        { k: "boundary_output", t: "What they chose", note: "rating or answer", src: "motor" }
      ]
    }
  ];

  // A function, not a table: WARM/COOL/FAINT are undefined until readTheme()
  // runs, so a module-scope object would freeze `undefined` into every entry.
  function kindColor(k) {
    if (k === "observation") return COOL;
    if (k === "context") return FAINT;
    return WARM;                       // stimulus in, boundary_output out
  }

  function init(canvas, data) {
    var ctx = canvas.getContext("2d");
    if (!ctx) return;
    var N = data.n, P = data.p;
    var isSub = data.div.map(function (d) { return d === "sub"; });

    var yaw = -0.62, pitch = -0.16, dragging = false, lx = 0, ly = 0;
    var idle = true, idleTimer = 0;
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var uc = 0;

    // Which category the pointer is over, and which one the reader has pinned
    // by clicking. Pin wins; hover is the preview.
    var hoverIdx = -1, pinIdx = -1, rows = [];

    // The last frame's projected brain: centre and every parcel's screen
    // position. Written by draw(), read by the touch hit-test, which needs the
    // silhouette as it currently stands rather than as it started.
    var lastProj = null;

    // Terminal nodes: a cord below the brain, with sensory in and motor out.
    /* The cord hangs BELOW the brain, so its length is the drawing's height
     * budget: eight nodes reaching z = -2.45 made the figure two-thirds
     * empty space, and in the portrait frame the tail simply ran out of
     * canvas and was cut. Four nodes to -1.53, with the terminals just under
     * them. It reads as a stem now rather than a dangling wire. */
    var CORD = [];
    for (var s = 0; s <= 3; s++) CORD.push([0, -0.06, -1.05 - s * 0.16]);
    var SENSORY = [[-0.30, 0.08, -1.60], [-0.40, -0.04, -1.50], [-0.24, 0.16, -1.70]];
    var MOTOR = [[0.30, -0.08, -1.60], [0.40, 0.04, -1.50], [0.24, -0.16, -1.70]];

    // `data-panels="scene"` draws the anatomy alone -- no category fan, no
    // pipeline. The hero wants the object, not the annotated diagram.
    var sceneOnly = canvas.getAttribute("data-panels") === "scene";
    var annotated = sceneOnly && canvas.getAttribute("data-uses") === "on";

    /* How far the parcels reach, as bounds that do not depend on yaw.
     *
     * The ring fits the brain to the space inside it, and the brain turns. Fit
     * it to the CURRENT projected extent and it would breathe in and out once
     * per revolution; these two bounds hold for every yaw, so the scale is set
     * once and the drawing sits still. */
    var RMAX = 0, ZMAX = 0;
    (function () {
      var sp = Math.abs(Math.sin(pitch)), cp = Math.abs(Math.cos(pitch));
      for (var q = 0; q < N; q++) {
        var r = Math.hypot(P[q * 3], P[q * 3 + 1]), z = Math.abs(P[q * 3 + 2]);
        if (r > RMAX) RMAX = r;
        if (r * sp + z * cp > ZMAX) ZMAX = r * sp + z * cp;
      }
      RMAX = RMAX || 1; ZMAX = ZMAX || 1;
    })();

    function resize() {
      var r = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(r.width * dpr));
      canvas.height = Math.max(1, Math.round(r.height * dpr));
    }

    function project(v, sc, ox, oy) {
      var cy = Math.cos(yaw), sy = Math.sin(yaw);
      var cp = Math.cos(pitch), sp = Math.sin(pitch);
      var x1 = v[0] * cy + v[1] * sy, y1 = -v[0] * sy + v[1] * cy;
      var y2 = y1 * cp - v[2] * sp, z2 = y1 * sp + v[2] * cp;
      return [ox + x1 * sc, oy - z2 * sc, y2];
    }

    function hairline(w) { ctx.lineWidth = Math.max(0.6, w) * dpr; }

    /* Set a sans font that FITS.
     *
     * The panels are laid out in fractions of the canvas width but the text in
     * them was fixed in pixels, so in a narrow column "Reading what someone
     * responds to" ran off the right edge. Shrink until it fits, with a floor:
     * below the floor it is unreadable anyway and the caption under the canvas
     * carries the same sentence. */
    function fitFont(text, maxW, startPx, weight) {
      var px = startPx;
      for (var g = 0; g < 14; g++) {
        ctx.font = (weight ? weight + " " : "") + px + "px ui-sans-serif, system-ui, sans-serif";
        if (px <= 10.5 * dpr || ctx.measureText(text).width <= maxW) break;
        px *= 0.94;
      }
      return px;
    }

    /* ---------------------------------------------------------- what is lit
     *
     * A category names a set of things in the head. `active()` returns the one
     * currently being asked about, or null when nothing is. */
    function active() {
      var i = pinIdx >= 0 ? pinIdx : hoverIdx;
      var c = USE_CASES[uc];
      return (i >= 0 && i < c.cats.length) ? c.cats[i] : null;
    }
    function inCat(cat, j) {
      if (!cat) return false;
      if (cat.src === "all") return true;
      if (cat.src === "cortex") return !isSub[j];
      return false;                     // sensory/motor live on the cord
    }

    /* ------------------------------------------------------- drawing helpers
     *
     * Square corners, hairlines, and type at roughly the size the surrounding
     * page uses.  The previous panels were rounded rectangles at 8-11px with
     * filled bars and a warm/cool wash: at the width these canvases actually
     * get, that was unreadable and looked like decoration rather than a
     * drawing.  The right panel is strictly monochrome now -- colour on this
     * canvas means attachment kind, and nothing in a pipeline has one. */

    /** Screen pixels for a size given in the page's own CSS pixels. */
    function px(n) { return n * dpr; }

    /** A named stage. Returns the y of its bottom edge. */
    function stage(x, y, w, title, sub) {
      var pad = px(9), h = px(sub ? 44 : 30);
      ctx.strokeStyle = LINE;
      hairline(1);
      ctx.beginPath();
      ctx.rect(x, y, w, h);              // square corners, on purpose
      ctx.stroke();
      ctx.textBaseline = "middle";
      ctx.fillStyle = INK;
      fitFont(title, w - pad * 2, px(13.5), "650");
      ctx.fillText(title, x + pad, y + px(sub ? 15 : 15));
      if (sub) {
        ctx.fillStyle = LINE;
        fitFont(sub, w - pad * 2, px(12));
        ctx.fillText(sub, x + pad, y + px(31));
      }
      return y + h;
    }

    function arrowDown(cx, y1, y2) {
      ctx.strokeStyle = LINE;
      hairline(1);
      ctx.beginPath();
      ctx.moveTo(cx, y1);
      ctx.lineTo(cx, y2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(cx - px(3.5), y2 - px(5));
      ctx.lineTo(cx, y2);
      ctx.lineTo(cx + px(3.5), y2 - px(5));
      ctx.stroke();
    }

    /* What each use case DOES with the signals on the left: three named stages
     * and the objective they are chosen to satisfy.  Written in ASCII because
     * canvas has no font fallback chain, so a combining mark or a unicode
     * subscript renders as whatever the one resolved family happens to carry. */
    var PIPELINES = {
      bci: {
        stages: [
          ["DECODE", "scalp signal to brain state"],
          ["ACT", "brain state to cursor, keypress"],
          ["INTERFACE", "the thing being controlled"]
        ],
        expr: [
          ["max", "rm"], [" ", "sp"], ["I", "va"], ["(", "rm"], ["a", "va"],
          ["t", "sb"], [" ; ", "rm"], ["s", "va"], ["t+k", "sb"], [")", "rm"]
        ],
        gloss: "the most control the signal can support"
      },
      tms: {
        stages: [
          ["FIELD", "computed on this head, not a template"],
          ["PREDICT", "network response to the pulse"],
          ["COMPARE POSES", "choose one, bound the dose"]
        ],
        expr: [
          ["arg max", "rm"], [" ", "sp"], ["d", "va"], ["(", "rm"],
          ["target", "rm"], [")", "rm"], ["  ", "sp"], ["s.t.", "it"],
          [" ", "sp"], ["dose", "rm"], [" \u2264 ", "rm"], ["limit", "rm"]
        ],
        gloss: "engagement maximised, exposure bounded"
      },
      semantic: {
        stages: [
          ["PREDICT", "response to what they were shown"],
          ["COMPARE", "predicted against measured"],
          ["FIT THE PERSON", "on the residual between them"]
        ],
        expr: [
          ["\u03b8", "va"], [" \u2190 ", "rm"], ["\u03b8", "va"],
          [" \u2212 ", "rm"], ["\u03b7", "va"], ["\u2207", "rm"],
          ["\u03b8", "sb"], ["\u2016", "rm"], ["y", "va"],
          [" \u2212 ", "rm"], ["\u0177", "va"], ["\u2016", "rm"]
        ],
        gloss: "the gap between them is what individualises"
      }
    };

    /* Setting the objective as mathematics rather than as ASCII.
     *
     * These read `theta -= grad || y - y_hat ||` and `argmax d(target) s.t.
     * dose <= limit` in a monospace face -- which is code, not an equation.
     * An expression is now a list of runs, each tagged with what it IS, and
     * the tag chooses the face: variables italic serif, operators and function
     * names upright, subscripts smaller and dropped. That is the whole
     * convention of mathematical setting, and it is what makes an equation
     * legible at a glance -- you can see which letters are quantities.
     *
     * Done by hand because the site carries no MathJax and no KaTeX: it has no
     * CDN dependency and no npm dependency for its own HTML, and one canvas
     * expression is not worth breaking that for. Real Unicode throughout --
     * U+2190 arrow, U+2212 minus (not a hyphen), U+2207 nabla, U+2016 double
     * bar, U+0177 y-with-circumflex, U+2264 less-or-equal.
     *
     *   va  variable        italic serif
     *   rm  operator/name   upright serif
     *   it  abbreviation    italic serif, upright spacing ("s.t.")
     *   sb  subscript       0.72 size, dropped a quarter of an em
     *   sp  space           measured, never drawn
     */
    var MATH_SERIF = '"Iowan Old Style", "Charter", Georgia, "Times New Roman", serif';

    function mathFont(kind, size) {
      if (kind === "va") return "italic " + size + "px " + MATH_SERIF;
      if (kind === "it") return "italic " + size + "px " + MATH_SERIF;
      if (kind === "sb") return "italic " + (size * 0.72) + "px " + MATH_SERIF;
      return size + "px " + MATH_SERIF;
    }

    function mathWidth(runs, size) {
      var total = 0;
      for (var i = 0; i < runs.length; i++) {
        ctx.font = mathFont(runs[i][1], size);
        total += ctx.measureText(runs[i][0]).width;
      }
      return total;
    }

    function drawMath(runs, x, y, size) {
      var at = x;
      for (var i = 0; i < runs.length; i++) {
        var text = runs[i][0], kind = runs[i][1];
        ctx.font = mathFont(kind, size);
        if (kind !== "sp") {
          ctx.fillText(text, at, kind === "sb" ? y + size * 0.24 : y);
        }
        at += ctx.measureText(text).width;
      }
    }

    function pipeline(kind, x, y, w) {
      var spec = PIPELINES[kind] || PIPELINES.bci;
      var cx = x + w * 0.5, b = y;
      spec.stages.forEach(function (s, i) {
        if (i) { arrowDown(cx, b + px(3), b + px(17)); b += px(20); }
        b = stage(x, b, w, s[0], s[1]);
      });

      ctx.textBaseline = "middle";
      ctx.fillStyle = INK;
      var size = px(14);
      for (var g = 0; g < 14; g++) {
        if (size <= px(9) || mathWidth(spec.expr, size) <= w) break;
        size *= 0.94;
      }
      drawMath(spec.expr, x, b + px(26), size);
      ctx.fillStyle = LINE;
      fitFont(spec.gloss, w, px(12));
      ctx.fillText(spec.gloss, x, b + px(44));
    }


    /* Where the use cases sit on a canvas this size, or null if they do not
     * fit.
     *
     * Two layouts, chosen by the shape of the box rather than by a breakpoint,
     * so the drawing always matches the frame the stylesheet gave it:
     *
     *   LANDSCAPE  two columns flanking the brain, six clusters, three a side.
     *   PORTRAIT   the same six clusters above and below it, anchored on
     *              alternating edges of the block so each bundle runs out into
     *              a gutter instead of across its neighbours' words.
     *
     * Everything is measured, not assumed: "cognitive intervention design" is
     * twice the width of "language decoding", and it is the long one the
     * columns have to clear.
     *
     * Cached on the size -- the canvas redraws every frame while the brain
     * turns, and none of this changes between frames. */
    var usesCache = null;
    function usesGeom(W, H) {
      if (usesCache && usesCache.W === W && usesCache.H === H) return usesCache.g;
      var fs = Math.max(10, Math.min(13.5, (W / dpr) * 0.0132)) * dpr;
      ctx.font = "500 " + fs + "px ui-sans-serif, system-ui, sans-serif";
      var widest = 0, i, j;
      [LEFT, RIGHT].forEach(function (col) {
        for (i = 0; i < col.length; i++) {
          for (j = 0; j < col[i].length; j++) {
            widest = Math.max(widest, ctx.measureText(col[i][j]).width);
          }
        }
      });
      var ox = W * 0.5, oy = H * 0.5;
      var g = (H / W > 1.05)
        ? stacked(W, H, ox, oy, fs, widest)
        : beside(W, H, ox, oy, fs, widest);
      usesCache = { W: W, H: H, g: g };
      return g;
    }

    /** The mark's type size: a share of the width, floored so it stays a name
     * rather than a caption, and capped so it stays inside the brain. */
    function mark(W, share, cap) {
      return Math.max(15, Math.min(cap, (W / dpr) * share)) * dpr;
    }

    /* One cluster: where its labels are, the control point its lines bundle
     * through, and the single point on the brain they all arrive at. The
     * curves converge rather than meeting at a kink and continuing as one
     * line -- a corner there reads as a mistake, and the cluster is the claim
     * that these things are one kind of use, not that they share a wire. */
    function cluster(names, xs, ys, align, cx, cy, dx, dy) {
      var labels = [];
      for (var i = 0; i < names.length; i++) {
        labels.push({ x: xs[i], y: ys[i], t: names[i] });
      }
      // Direction, not a point: where the bundle lands is the brain's own
      // outline in that direction, which changes as the brain turns, so it is
      // found at draw time rather than stored here.
      return { labels: labels, align: align, cx: cx, cy: cy, dx: dx, dy: dy };
    }

    /* How far the parcels reach from the centre along (dx, dy): the projected
     * silhouette, not the bounding ellipse.
     *
     * The ellipse is a bound that holds at every yaw, so in most directions it
     * stands well clear of the dots -- which is why the leader lines used to
     * stop short of the brain with nothing in between. Only parcels within a
     * narrow band of the ray count, so this is the outline along that ray and
     * not the support point of the whole cloud. */
    function reach(ox, oy, dx, dy, xs, ys) {
      var band = 11 * dpr, best = 0, far = 0;
      for (var i = 0; i < N; i++) {
        var ax = xs[i] - ox, ay = ys[i] - oy;
        var along = ax * dx + ay * dy;
        if (along > far) far = along;
        if (along <= best) continue;
        if (Math.abs(ax * dy - ay * dx) > band) continue;
        best = along;
      }
      return (best || far) + 3 * dpr;
    }

    function beside(W, H, ox, oy, fs, widest) {
      var colx = W * 0.5 - widest - px(12);
      // Twelve labels and two cluster gaps have to fit the height, and no row
      // should be tighter than the type in it.
      var pitch = Math.min((H - fs * 2) / columnSpan(LEFT), fs * 2.1);
      if (W / dpr < USES_MIN_W || colx < W * 0.16 || pitch < fs * 1.25) return null;
      var hubx = colx * 0.85;
      var sc = Math.min(hubx * 0.80 / RMAX, H * 0.46 / ZMAX);
      var out = [];
      [[-1, LEFT], [1, RIGHT]].forEach(function (pair) {
        var side = pair[0], groups = pair[1];
        var rows = columnRows(groups, pitch), at = 0;
        for (var gi = 0; gi < groups.length; gi++) {
          var cx = ox + side * hubx, cy = oy + rows.hubs[gi];
          var xs = [], ys = [];
          for (var li = 0; li < groups[gi].length; li++) {
            xs.push(ox + side * colx); ys.push(oy + rows.ys[at]); at++;
          }
          var ang = Math.atan2(cy - oy, cx - ox);
          out.push(cluster(groups[gi], xs, ys, side > 0 ? "left" : "right",
                           cx, cy, Math.cos(ang), Math.sin(ang)));
        }
      });
      return { fs: fs, sc: sc, ms: mark(W, 0.037, 34), clusters: out };
    }

    function stacked(W, H, ox, oy, fs, widest) {
      var pitch = fs * 1.75;
      var spanA = columnSpan(LEFT) * pitch;
      var spanB = columnSpan(RIGHT) * pitch;
      var pad = pitch * 1.6;
      var half = H / 2 - pad - Math.max(spanA, spanB) - fs * 0.9;
      var blockHalf = (widest + px(6)) / 2;
      if (W / dpr < 300 || half < fs * 2.6 || blockHalf * 2 > W * 0.86) return null;
      /* Slide each block toward its own gutter -- the top block right, the
       * bottom block left -- instead of centring both.
       *
       * Centred, the two blocks put their gutters within a couple of label
       * widths of each other and both bundles leave the stack near the middle
       * of the canvas, so on a phone the six curves crossed the same strip of
       * space and read as a tangle. Offsetting them makes the top bundle
       * descend on the right and the bottom bundle climb on the left, which is
       * also the arrangement the wide layout has.
       *
       * `room` is what is actually free beside a centred block, so the shift
       * shrinks to nothing rather than pushing type off a narrow canvas; the
       * margin covers the control point, which sits a little outside the
       * block. */
      var room = W * 0.5 - blockHalf - px(14);
      /* AWAY from its own gutter, not toward it. Each block's bundle leaves
       * from the edge nearest the brain's landing point -- the top block's to
       * the right, the bottom block's to the left -- so the space on THAT side
       * is the corridor the curves travel down. Sliding a block toward its
       * gutter narrows that corridor and the curves bend back over their own
       * labels; sliding it away widens it, and the three bundles separate. */
      var shift = Math.max(0, room) * 0.85;
      var sc = Math.min(half / ZMAX, W * 0.46 / RMAX);
      var out = [];
      /* Above the brain every bundle leaves rightwards; below, leftwards.
       *
       * The rows are NOT flush with each other. A flush edge gives every line
       * in a block the same starting x, and twelve lines that start on one
       * vertical and end at three points on a small brain run together --
       * they leave as a single grey band and you cannot tell which label owns
       * which line. So the edge is raked: each row steps a little further
       * toward the gutter than the row before it, the near rows furthest.
       * Every line now starts at its own x AND its own y, which is what
       * separates them; the raked edge is a consequence, and reads as one
       * deliberate diagonal rather than twelve accidents. */
      var stagger = Math.min(px(9), Math.max(0, room - shift) + px(9));
      [[-1, 1, LEFT, oy - half - pad - spanA / 2],
       [1, -1, RIGHT, oy + half + pad + spanB / 2]].forEach(function (spec) {
        var dir = spec[0], side = spec[1], groups = spec[2], mid = spec[3];
        var rows = columnRows(groups, pitch), at = 0;
        var ax = ox + side * blockHalf - side * shift;
        var nRows = 0;
        for (var g2 = 0; g2 < groups.length; g2++) nRows += groups[g2].length;
        for (var gi = 0; gi < groups.length; gi++) {
          var xs = [], ys = [], edge = null;
          for (var li = 0; li < groups[gi].length; li++) {
            var y = mid + rows.ys[at];
            /* Rake by distance from the BRAIN: the farthest row reaches
             * furthest into the gutter, the nearest row least.
             *
             * This direction is forced, and the other one was tried first. A
             * line from the far row has to travel past every row between it
             * and the brain. Rake the near rows out furthest and it passes
             * over their text -- five labels with a line drawn through them.
             * Rake the far rows out furthest and each line leaves outside
             * everything it must cross, so the block's text is never touched. */
            var rank = dir < 0 ? (nRows - 1 - at) : at;
            xs.push(ax + side * stagger * rank);
            at++;
            ys.push(y);
            // The row of this cluster nearest the brain: the bundle bends past
            // it, not through the middle of the block.
            if (edge === null || (dir < 0 ? y > edge : y < edge)) edge = y;
          }
          /* How far off vertical this cluster comes in: the far one wide, the
           * near one nearly straight, so three bundles from one block arrive
           * at three places instead of piling into one.
           *
           * Ranked by distance FROM THE BRAIN, not by index. `gi` counts down
           * the block, which for the block ABOVE the brain runs far-to-near
           * and for the block BELOW it runs near-to-far -- so using `gi`
           * directly gave the lower stack the upper stack's ordering, mirrored
           * the wrong way: its nearest cluster came in widest and its farthest
           * came in straightest, and the long lines cut across the short ones
           * on their way to the outline. */
          var gk = dir < 0 ? gi : (groups.length - 1 - gi);
          var off = 1.15 - 0.75 * (groups.length > 1 ? gk / (groups.length - 1) : 0);
          // Stagger the gutter per cluster as well as the angle: three bundles
          // that leave from one x share a corridor and cross inside it, which
          // is what the fan of angles alone could not fix.
          // Same ranking for the gutter: the farthest cluster leaves widest.
          var gutter = px(14) + px(13) * gk;
          var lead = side > 0 ? Math.max.apply(null, xs) : Math.min.apply(null, xs);
          out.push(cluster(groups[gi], xs, ys, side > 0 ? "right" : "left",
                           lead + side * gutter, edge - dir * pitch * 1.2,
                           side * Math.sin(off), dir * Math.cos(off)));
        }
      });
      return { fs: fs, sc: sc, ms: mark(W, 0.055, 30), clusters: out };
    }

    /* The name over the middle, and the use cases around it.
     *
     * Stroked in the page's own ground rather than plated: the canvas is
     * transparent so the section shows through, and a filled rectangle here
     * would read as a card laid over the connectome. */
    function drawMark(ox, oy, W, H, uses, xs, ys) {
      if (uses) {
        ctx.font = "500 " + uses.fs + "px ui-sans-serif, system-ui, sans-serif";
        ctx.textBaseline = "middle";
        for (var ci = 0; ci < uses.clusters.length; ci++) {
          var k = uses.clusters[ci], side = k.align === "left" ? 1 : -1;
          var r = reach(ox, oy, k.dx, k.dy, xs, ys);
          var tx = ox + k.dx * r, ty = oy + k.dy * r;
          for (var li = 0; li < k.labels.length; li++) {
            var L = k.labels[li];
            ctx.strokeStyle = LINE; hairline(0.6); ctx.globalAlpha = 0.6;
            ctx.beginPath();
            ctx.moveTo(L.x, L.y);
            ctx.quadraticCurveTo(k.cx, k.cy, tx, ty);
            ctx.stroke();
            ctx.globalAlpha = 0.7;
            ctx.fillStyle = LINE;
            ctx.beginPath(); ctx.arc(L.x, L.y, 1.5 * dpr, 0, 6.2832); ctx.fill();
            ctx.globalAlpha = 1;
            ctx.textAlign = k.align;
            ctx.fillStyle = INK2;
            ctx.fillText(L.t, L.x + side * px(6), L.y);
          }
        }
      }

      var ms = uses ? uses.ms : mark(W, 0.037, 34);
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.font = "700 " + ms + "px ui-sans-serif, system-ui, sans-serif";
      ctx.lineJoin = "round";
      ctx.lineWidth = ms * 0.30;
      ctx.strokeStyle = GROUND;
      ctx.globalAlpha = 0.92;
      ctx.strokeText(MARK, ox, oy);
      ctx.globalAlpha = 1;
      ctx.fillStyle = INK;
      ctx.fillText(MARK, ox, oy);
      ctx.textAlign = "left";
    }

    function draw() {
      readTheme();
      var W = canvas.width, H = canvas.height;
      ctx.clearRect(0, 0, W, H);            // transparent: the section shows through

      var c = USE_CASES[uc];
      var act = active();

      // Scene-only drops the cord and the terminals, so the brain is sized to
      // fill the frame rather than to leave 2.7 units of room underneath it
      // for a body that is no longer drawn.
      var sceneW = sceneOnly ? W : W * 0.30;
      var midX = W * 0.335, rightX = W * 0.645;
      var rightW = W - rightX - 8 * dpr;
      var uses = annotated ? usesGeom(W, H) : null;
      var sc = uses
        // The brain takes the room the labels leave; usesGeom worked out how
        // much that is when it placed them.
        ? uses.sc
        : sceneOnly
        ? Math.min(H / 1.3, W / 2.05)
        // Sized from what is actually drawn: the parcels span about 1.2 units
        // across and the cord takes the scene down to -2.7, so 3.6 units of
        // height. The old 0.19-of-the-smaller-side left the brain at a third
        // of the room its column had.
        : Math.min(H / 3.6, sceneW / 1.35);
      var ox = sceneOnly ? W * 0.5 : sceneW * 0.54;
      var oy = sceneOnly ? H * 0.5 : H * 0.40;

      // ---- scene: cord ----
      if (!sceneOnly) {
        ctx.strokeStyle = LINE; hairline(0.9);
        ctx.beginPath();
        for (var i = 0; i < CORD.length; i++) {
          var q = project(CORD[i], sc, ox, oy);
          if (i === 0) ctx.moveTo(q[0], q[1]); else ctx.lineTo(q[0], q[1]);
        }
        ctx.stroke();
      }

      // ---- scene: parcels ----
      var xs = new Float32Array(N), ys = new Float32Array(N), dep = new Float32Array(N);
      var order = [];
      for (var j = 0; j < N; j++) {
        var pr = project([P[j * 3], P[j * 3 + 1], P[j * 3 + 2]], sc, ox, oy);
        xs[j] = pr[0]; ys[j] = pr[1]; dep[j] = pr[2]; order.push(j);
      }
      order.sort(function (a, b) { return dep[a] - dep[b]; });

      // Keep the frame's projected geometry for hit-testing. A touch has to
      // know whether it landed ON the brain, and the only honest answer is the
      // silhouette actually drawn -- which changes every frame as it turns.
      lastProj = { ox: ox, oy: oy, xs: xs, ys: ys };

      // Which parcels this use case touches at all.
      var srcs = {};
      c.cats.forEach(function (k) { srcs[k.src] = true; });
      var lit = new Uint8Array(N);
      for (var m = 0; m < N; m++) {
        // Scene-only draws the whole anatomy at full weight: with no
        // categories there is nothing for a dimmed parcel to contrast against.
        lit[m] = (sceneOnly || srcs.all || (srcs.cortex && !isSub[m])) ? 1 : 0;
      }

      // ---- scene: tractography ----
      var E = (data.edges && data.edges.e) || [], EW = (data.edges && data.edges.w) || [];
      if (E.length) {
        ctx.strokeStyle = LINE;
        hairline(0.6);
        for (var e = 0; e < E.length; e += 2) {
          var ea = E[e], eb = E[e + 1];
          if (!lit[ea] && !lit[eb]) continue;
          var mid = ((dep[ea] + dep[eb]) / 2 + 1) / 2;
          var em = act ? (inCat(act, ea) || inCat(act, eb) ? 1 : 0.22) : 1;
          // The annotated diagrams draw the same connectome at a third of the
          // hero's scale, where the hero's alpha put it below the threshold of
          // being visible at all. Stronger where the drawing is smaller.
          // Inside the ring the brain is a third of the frame, where the
          // hero's alpha put the tracts below the threshold of being seen at
          // all -- the same reason the annotated diagrams draw them stronger.
          var base = (sceneOnly && !uses) ? (0.045 + 0.16 * EW[e / 2] * mid)
                                           : (0.13 + 0.34 * EW[e / 2] * mid);
          ctx.globalAlpha = base * em;
          ctx.beginPath();
          ctx.moveTo(xs[ea], ys[ea]);
          ctx.lineTo(xs[eb], ys[eb]);
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }

      var accent = act ? kindColor(act.k) : COOL;
      for (var k2 = 0; k2 < N; k2++) {
        var jj = order[k2], t = (dep[jj] + 1) / 2;
        var on = act ? inCat(act, jj) : !!lit[jj];
        if (act) {
          ctx.globalAlpha = on ? (0.45 + 0.5 * t) : 0.07;
          ctx.fillStyle = on ? accent : FAINT;
        } else {
          ctx.globalAlpha = lit[jj] ? (uses ? 0.5 : 0.34) + 0.5 * t : 0.10;
          ctx.fillStyle = lit[jj] ? COOL : FAINT;
        }
        ctx.beginPath();
        ctx.arc(xs[jj], ys[jj], (isSub[jj] ? 2.7 : 1.7) * dpr * (0.6 + 0.5 * t)
                * (act && on ? 1.35 : 1), 0, 6.2832);
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      if (annotated) drawMark(ox, oy, W, H, uses, xs, ys);

      // Everything past this point is annotation. The hero has none of it.
      if (sceneOnly) return;

      // ---- scene: terminals ----
      function terminals(list, on, hot) {
        var pts = [];
        list.forEach(function (v) {
          var q = project(v, sc, ox, oy);
          pts.push(q);
          ctx.fillStyle = hot ? accent : (on ? WARM : FAINT);
          ctx.globalAlpha = act ? (hot ? 1 : 0.12) : (on ? 0.9 : 0.25);
          ctx.beginPath();
          ctx.arc(q[0], q[1], (hot ? 4.2 : 3.1) * dpr, 0, 6.2832);
          ctx.fill();
        });
        ctx.globalAlpha = 1;
        return pts;
      }
      var sensPts = terminals(SENSORY, !!srcs.sensory, act && act.src === "sensory");
      var motPts = terminals(MOTOR, !!srcs.motor, act && act.src === "motor");

      function parcelSample(pred, want) {
        var picks = [];
        // deterministic stride, so the sample does not shimmer while orbiting
        var step = Math.max(1, Math.floor(N / want));
        for (var q2 = 0; q2 < N && picks.length < want; q2 += step) {
          if (pred(q2)) picks.push([xs[q2], ys[q2]]);
        }
        return picks;
      }
      function targets(src) {
        if (src === "all") return parcelSample(function () { return true; }, 7);
        if (src === "cortex") return parcelSample(function (q) { return !isSub[q]; }, 6);
        if (src === "sensory") return sensPts.map(function (q) { return [q[0], q[1]]; });
        if (src === "motor") return motPts.map(function (q) { return [q[0], q[1]]; });
        return [];        // context attaches to no anatomy -- correctly nothing
      }

      // ---- middle: categories, with annotation lines from the scene ----
      var n = c.cats.length;
      var top = H * 0.24, gap = Math.min(H * 0.16, (H * 0.58) / n);
      var colW = rightX - midX - 30 * dpr;
      ctx.textBaseline = "middle";
      rows = [];

      c.cats.forEach(function (cat, idx) {
        var y = top + idx * gap;
        var isAct = act === cat;
        var dim = act && !isAct;
        var col = kindColor(cat.k);
        var pts = targets(cat.src);
        var hub = [midX - 26 * dpr, y];

        rows.push({ y0: y - gap * 0.42, y1: y + gap * 0.42, x0: midX - 34 * dpr, x1: W });

        // Fan: one hairline from the elbow to each point this category names.
        ctx.strokeStyle = col; hairline(isAct ? 0.9 : 0.6);
        pts.forEach(function (pt) {
          ctx.globalAlpha = dim ? 0.07 : (isAct ? 0.55 : 0.30);
          ctx.beginPath();
          ctx.moveTo(pt[0], pt[1]);
          ctx.lineTo(hub[0], hub[1]);
          ctx.stroke();
          ctx.globalAlpha = dim ? 0.12 : 0.85;
          ctx.beginPath();
          ctx.arc(pt[0], pt[1], (isAct ? 2.6 : 1.9) * dpr, 0, 6.2832);
          ctx.fillStyle = col; ctx.fill();
        });
        ctx.globalAlpha = dim ? 0.15 : 0.7;
        ctx.beginPath();
        ctx.moveTo(hub[0], hub[1]);
        ctx.lineTo(midX - 10 * dpr, y);
        ctx.stroke();
        ctx.globalAlpha = 1;

        ctx.globalAlpha = dim ? 0.32 : 1;
        ctx.fillStyle = col;
        ctx.beginPath(); ctx.arc(midX - 10 * dpr, y, (isAct ? 3.2 : 2.2) * dpr, 0, 6.2832); ctx.fill();

        ctx.fillStyle = INK;
        fitFont(cat.t, colW, 15 * dpr, "650");
        ctx.fillText(cat.t, midX, y - 8 * dpr);
        ctx.fillStyle = isAct ? col : LINE;
        fitFont(cat.k + " · " + cat.note, colW, 12.5 * dpr);
        ctx.fillText(cat.k + " · " + cat.note, midX, y + 10 * dpr);
        ctx.globalAlpha = 1;
      });

      // ---- right: what is done with the signals ----
      pipeline(c.right, rightX, H * 0.16, rightW);
    }

    var raf = 0, running = false;
    function frame() {
      // Resume the slow rotation a beat after the reader stops dragging, so a
      // diagram left alone is always turning.
      if (!dragging && !idle) {
        idleTimer++;
        if (idleTimer > 75) { idle = true; idleTimer = 0; }
      }
      if (idle) yaw += 0.0022;
      draw();
      raf = requestAnimationFrame(frame);
    }
    function start() { if (running) return; running = true; frame(); }
    function stop() {
      if (!running) return;
      running = false;
      cancelAnimationFrame(raf);
    }

    // Three of these run on the landing page, each redrawing 414 parcels and
    // 900 tracts every frame. Off-screen they were still doing it, which on a
    // page you scroll through is most of the time and all of the cost.
    function watch() {
      if (!("IntersectionObserver" in window)) { start(); return; }
      new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting && !document.hidden) start(); else stop();
        });
      }, { rootMargin: "160px 0px" }).observe(canvas);
      document.addEventListener("visibilitychange", function () {
        if (document.hidden) stop();
        else if (canvas.getBoundingClientRect().top < window.innerHeight) start();
      });
    }

    /* --------------------------------------------------------- interaction */

    function localPos(e) {
      var r = canvas.getBoundingClientRect();
      var q = e.touches ? e.touches[0] : e;
      return [(q.clientX - r.left) * dpr, (q.clientY - r.top) * dpr];
    }
    function rowAt(p) {
      for (var i = 0; i < rows.length; i++) {
        var r = rows[i];
        if (p[0] >= r.x0 && p[0] <= r.x1 && p[1] >= r.y0 && p[1] <= r.y1) return i;
      }
      return -1;
    }

    var downAt = null;

    /* Did this press land on the brain?
     *
     * The canvas is most of a phone screen and it used to capture every drag
     * that started anywhere inside it, so a thumb put down on the label
     * columns -- the majority of the canvas in portrait -- orbited the brain
     * instead of scrolling the page. Scrolling past the hero meant finding a
     * gap beside it.
     *
     * The test is the drawn silhouette, not a bounding box: the distance from
     * the centre along the ray to the touch, against `reach()` in that same
     * direction, which is the function the annotation lines already use to
     * find where the brain ends. Plus a small margin, because a fingertip is
     * bigger than a parcel.
     */
    function onBrain(clientX, clientY) {
      if (!lastProj) return false;               // nothing drawn yet
      var r = canvas.getBoundingClientRect();
      var px_ = (clientX - r.left) * (canvas.width / r.width);
      var py_ = (clientY - r.top) * (canvas.height / r.height);
      var dx = px_ - lastProj.ox, dy = py_ - lastProj.oy;
      var d = Math.hypot(dx, dy);
      if (d < 1) return true;                    // dead centre
      var out = reach(lastProj.ox, lastProj.oy, dx / d, dy / d,
                      lastProj.xs, lastProj.ys);
      return d <= out + 18 * dpr;
    }

    function down(e) {
      var q = e.touches ? e.touches[0] : e;
      // A touch outside the brain belongs to the page: leave `dragging` false
      // and do not preventDefault, so the browser scrolls as it normally
      // would. A mouse keeps the whole canvas -- a cursor is precise, there is
      // no scroll to steal, and drag-anywhere is the nicer pointer behaviour.
      if (e.touches && !onBrain(q.clientX, q.clientY)) {
        downAt = null;
        dragging = false;
        return;
      }
      downAt = [q.clientX, q.clientY];
      dragging = true; idle = false; idleTimer = 0;
      lx = q.clientX; ly = q.clientY;
    }
    function move(e) {
      if (!sceneOnly && !dragging) {
        var was = hoverIdx;
        hoverIdx = rowAt(localPos(e));
        if (hoverIdx !== was) {
          canvas.style.cursor = hoverIdx >= 0 ? "pointer" : "grab";
        }
      }
      if (!dragging) return;
      var q = e.touches ? e.touches[0] : e;
      yaw += (q.clientX - lx) * 0.009;
      pitch = Math.max(-1.3, Math.min(1.3, pitch + (q.clientY - ly) * 0.007));
      lx = q.clientX; ly = q.clientY;
      if (e.cancelable) e.preventDefault();
    }
    function up(e) {
      // A click is a press that did not travel. Anything further is an orbit,
      // and pinning a category on the way out of a drag would be a surprise.
      if (downAt && !sceneOnly) {
        var q = (e && e.changedTouches) ? e.changedTouches[0] : e;
        var moved = q ? Math.abs(q.clientX - downAt[0]) + Math.abs(q.clientY - downAt[1]) : 99;
        if (moved < 5) {
          var i = rowAt(localPos(q));
          pinIdx = (i >= 0 && i === pinIdx) ? -1 : i;
        }
      }
      downAt = null;
      dragging = false;
      idleTimer = 0;
    }

    canvas.addEventListener("mousedown", down);
    canvas.addEventListener("mousemove", move);
    canvas.addEventListener("mouseleave", function () {
      hoverIdx = -1;
      canvas.style.cursor = "grab";
    });
    window.addEventListener("mousemove", function (e) { if (dragging) move(e); });
    window.addEventListener("mouseup", up);
    canvas.addEventListener("touchstart", down, { passive: true });
    canvas.addEventListener("touchmove", move, { passive: false });
    canvas.addEventListener("touchend", up);
    canvas.style.cursor = "grab";
    window.addEventListener("resize", function () { resize(); draw(); });

    // One canvas per section. Each use case gets its own <section> in the page,
    // and every canvas carrying data-case renders that case only.
    var want = canvas.getAttribute("data-case");
    if (want) {
      for (var w2 = 0; w2 < USE_CASES.length; w2++) {
        if (USE_CASES[w2].id === want) { uc = w2; break; }
      }
    }

    resize();
    var rm = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)");
    if (rm && rm.matches) {
      // Drawn on demand rather than continuously. The theme is the device's
      // alone, so a colour-scheme flip is the only thing that repaints it --
      // plus hover and click, which must still work.
      idle = false;
      draw();
      var cs = window.matchMedia("(prefers-color-scheme: dark)");
      if (cs.addEventListener) cs.addEventListener("change", draw);
      canvas.addEventListener("mousemove", draw);
      canvas.addEventListener("mouseup", draw);
      canvas.addEventListener("mouseleave", draw);
    } else {
      watch();
    }
  }

  function boot() {
    var all = document.querySelectorAll("canvas.arch-canvas");
    if (!all.length) return;
    Array.prototype.forEach.call(all, bootOne);
  }

  function bootOne(canvas) {
    Promise.all([
      fetch(canvas.getAttribute("data-src") || "static/brain.json").then(function (r) { return r.json(); }),
      fetch(canvas.getAttribute("data-edges") || "static/edges.json")
        .then(function (r) { return r.json(); })
        .catch(function () { return { e: [], w: [] }; })   // tracts are optional
    ])
      .then(function (both) { both[0].edges = both[1]; init(canvas, both[0]); })
      .catch(function () { canvas.style.display = "none"; });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else { boot(); }
})();
