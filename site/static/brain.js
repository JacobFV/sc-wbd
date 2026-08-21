/* The WBD mark, one canvas per released checkpoint.
 *
 * Not a generic brain graphic. The points are `AnatomyPrior.positions` in the
 * fsLR_32k surface RAS frame -- the same coordinates the lead field integrates
 * against and the same 400 cortical + 14 subcortical parcels the model carries
 * state for. Colour is the 9-family partition, the one separated on a 20-tracer
 * PET receptor panel and on myelin+thickness under a 1000-spin Vasa null. The
 * linework is the strongest 900 edges of the measured connectome: a cloud of
 * dots says the brain has regions, the same cloud with its tracts says the
 * regions are coupled, which is the whole claim of a whole-brain model.
 *
 * Raw canvas 2D with a hand-rolled projection: no WebGL context to lose, no
 * library to load, works with a strict CSP, and degrades to nothing rather
 * than to something that is not the anatomy.
 *
 * Drag to orbit. Each canvas holds its own yaw and pitch, so turning one does
 * not turn the others.
 *
 * -- what changed, and why --
 *
 * This file used to bind `getElementById("wbd-brain")`, an id that appears in
 * no template and no page. It was loaded on every page and did nothing. It now
 * binds every `canvas[data-brain]`, which is what the run stack in the Results
 * section is made of.
 *
 * The projection scale is fixed from the point cloud's BOUNDING SPHERE rather
 * than refitted per frame. A refit keeps the drawing tight at every angle and
 * makes the brain breathe as you drag it, because the silhouette's extent
 * changes under rotation and the radius does not.
 *
 * Nothing auto-rotates. Five spinning brains beside four paragraphs of numbers
 * is motion competing with reading, and the default view is chosen to be worth
 * holding still: yaw pi/2, pitch 0, which puts screen-x on +Y and screen-y on
 * +Z -- a right-lateral sagittal silhouette facing the direction the series
 * runs.
 */
(function () {
  "use strict";

  // Two cortical families in the house blues, subcortex warm. Distinguishable
  // for the ~8% of men with a red-green deficiency: the split that carries
  // meaning here is cortex-vs-subcortex, which is blue-vs-amber, not
  // red-green. Constant across themes, like the rest of the signal hues.
  var COL = [
    [ 96, 165, 232], [ 52, 110, 190],
    [232, 156,  74], [225, 133,  92], [214, 160,  58],
    [236, 176, 102], [204, 122,  70], [244, 194, 122], [214, 145,  48]
  ];

  // One fetch per URL however many canvases ask for it. Five canvases sharing
  // one geometry file should be one request, not five.
  var CACHE = {};
  function load(url) {
    if (!CACHE[url]) {
      CACHE[url] = fetch(url).then(function (r) {
        if (!r.ok) throw new Error(r.status + " " + url);
        return r.json();
      });
    }
    return CACHE[url];
  }

  function num(el, name, dflt) {
    var v = parseFloat(el.getAttribute(name));
    return isNaN(v) ? dflt : v;
  }

  function init(canvas, data, edges) {
    var ctx = canvas.getContext("2d");
    if (!ctx) return;

    var N = data.n, P = data.p, F = data.f;
    var isSub = data.div.map(function (d) { return d === "sub"; });
    var dim = canvas.getAttribute("data-dim") === "on";

    // Sagittal, facing right, unless the page asks for another view.
    var yaw = num(canvas, "data-yaw", Math.PI / 2);
    var pitch = num(canvas, "data-pitch", 0);
    var fill = num(canvas, "data-fill", 0.98);

    // The bounding-sphere radius. Rotation preserves it, so a scale derived
    // from it fits at every angle and never changes as the brain is dragged.
    var maxR = 0;
    for (var i = 0; i < N; i++) {
      var r2 = P[i * 3] * P[i * 3] + P[i * 3 + 1] * P[i * 3 + 1] +
               P[i * 3 + 2] * P[i * 3 + 2];
      if (r2 > maxR) maxR = r2;
    }
    maxR = Math.sqrt(maxR) || 1;

    var ea = null, eb = null, ew = null;
    if (edges && edges.e && edges.w) {
      var M = edges.w.length;
      ea = new Int32Array(M); eb = new Int32Array(M); ew = new Float32Array(M);
      var wmax = 0;
      for (var k = 0; k < M; k++) if (edges.w[k] > wmax) wmax = edges.w[k];
      for (var k2 = 0; k2 < M; k2++) {
        ea[k2] = edges.e[k2 * 2];
        eb[k2] = edges.e[k2 * 2 + 1];
        ew[k2] = edges.w[k2] / (wmax || 1);
      }
    }

    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var xs = new Float32Array(N), ys = new Float32Array(N);
    var depth = new Float32Array(N);
    var order = new Array(N);
    for (var q = 0; q < N; q++) order[q] = q;

    function resize() {
      var r = canvas.getBoundingClientRect();
      if (!r.width || !r.height) return false;
      canvas.width = Math.max(1, Math.round(r.width * dpr));
      canvas.height = Math.max(1, Math.round(r.height * dpr));
      return true;
    }

    function draw() {
      var w = canvas.width, h = canvas.height;
      if (!w || !h) return;
      ctx.clearRect(0, 0, w, h);

      var cy = Math.cos(yaw), sy = Math.sin(yaw);
      var cp = Math.cos(pitch), sp = Math.sin(pitch);
      var s = (Math.min(w, h) / 2) * fill / maxR;
      var ox = w / 2, oy = h / 2;

      for (var i = 0; i < N; i++) {
        // Anatomical RAS: x right, y anterior, z superior. Screen up is +z, so
        // z maps to -screenY.
        var X = P[i * 3], Y = P[i * 3 + 1], Z = P[i * 3 + 2];
        var x1 = X * cy + Y * sy, y1 = -X * sy + Y * cy;
        var y2 = y1 * cp - Z * sp, z2 = y1 * sp + Z * cp;
        xs[i] = ox + x1 * s;
        ys[i] = oy - z2 * s;
        depth[i] = y2;
      }

      // Tracts first, so the parcels sit on top of their own wiring.
      if (ea) {
        // Scale with the drawing: a hairline tuned for a 620px canvas is a
        // smear at 170px, which is what the 1.76 M model is drawn at.
        ctx.lineWidth = Math.max(0.5, 0.55 * dpr * (0.45 + s / (150 * dpr)));
        ctx.strokeStyle = "rgb(140,143,150)";
        for (var m = 0; m < ea.length; m++) {
          var a = ea[m], b = eb[m];
          var t = ((depth[a] + depth[b]) / 2 + 1) / 2;
          var al = (0.04 + 0.26 * ew[m] * t) * (dim ? 0.6 : 1);
          if (al < 0.012) continue;
          ctx.globalAlpha = al;
          ctx.beginPath();
          ctx.moveTo(xs[a], ys[a]);
          ctx.lineTo(xs[b], ys[b]);
          ctx.stroke();
        }
      }

      order.sort(function (a, b) { return depth[a] - depth[b]; });

      var rk = 0.55 + 0.75 * (s / (150 * dpr));
      for (var kk = 0; kk < N; kk++) {
        var j = order[kk];
        var td = (depth[j] + 1) / 2;                 // 0 far .. 1 near
        var c = dim ? [148, 150, 158] : (COL[F[j]] || COL[0]);
        var rad = (isSub[j] ? 3.0 : 2.0) * dpr * (0.62 + 0.55 * td) * rk;
        ctx.globalAlpha = (0.32 + 0.60 * td) * (dim ? 0.62 : 1);
        ctx.fillStyle = "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
        ctx.beginPath();
        ctx.arc(xs[j], ys[j], Math.max(0.4, rad), 0, 6.2832);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    // ---- drag to orbit -------------------------------------------------
    //
    // The move and up handlers go on `window`, not the canvas, so a drag that
    // leaves the canvas keeps working and releasing outside it still ends the
    // drag. Only this canvas's own pointerdown starts one.
    var dragging = false, lx = 0, ly = 0, queued = false;

    function schedule() {
      if (queued) return;
      queued = true;
      window.requestAnimationFrame(function () { queued = false; draw(); });
    }

    function down(e) {
      dragging = true;
      var p = e.touches ? e.touches[0] : e;
      lx = p.clientX; ly = p.clientY;
      canvas.style.cursor = "grabbing";
    }
    function move(e) {
      if (!dragging) return;
      var p = e.touches ? e.touches[0] : e;
      yaw += (p.clientX - lx) * 0.0095;
      pitch += (p.clientY - ly) * 0.0075;
      pitch = Math.max(-1.35, Math.min(1.35, pitch));
      lx = p.clientX; ly = p.clientY;
      // `passive: false` on touchmove so this can hold the gesture; without
      // preventDefault a drag on a phone scrolls the page instead of turning
      // the brain.
      if (e.cancelable) e.preventDefault();
      schedule();
    }
    function up() {
      if (!dragging) return;
      dragging = false;
      canvas.style.cursor = "grab";
    }

    canvas.addEventListener("mousedown", down);
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    canvas.addEventListener("touchstart", down, { passive: true });
    canvas.addEventListener("touchmove", move, { passive: false });
    window.addEventListener("touchend", up);
    canvas.style.cursor = "grab";

    // Keyboard: the canvas is focusable in the markup, so arrows turn it too.
    // A drag-only control is a control some readers do not have.
    canvas.addEventListener("keydown", function (e) {
      var step = e.shiftKey ? 0.35 : 0.12, used = true;
      if (e.key === "ArrowLeft") yaw -= step;
      else if (e.key === "ArrowRight") yaw += step;
      else if (e.key === "ArrowUp") pitch = Math.max(-1.35, pitch - step);
      else if (e.key === "ArrowDown") pitch = Math.min(1.35, pitch + step);
      else used = false;
      if (used) { e.preventDefault(); schedule(); }
    });

    var ro = null;
    if (window.ResizeObserver) {
      ro = new ResizeObserver(function () { if (resize()) draw(); });
      ro.observe(canvas);
    } else {
      window.addEventListener("resize", function () { if (resize()) draw(); });
    }

    if (resize()) draw();
  }

  function boot() {
    var canvases = document.querySelectorAll("canvas[data-brain]");
    for (var i = 0; i < canvases.length; i++) {
      (function (canvas) {
        var src = canvas.getAttribute("data-src");
        var eurl = canvas.getAttribute("data-edges");
        if (!src) return;
        var jobs = [load(src)];
        if (eurl) jobs.push(load(eurl));
        Promise.all(jobs)
          .then(function (r) { init(canvas, r[0], r[1] || null); })
          .catch(function () {
            // No geometry, no mark. Leave the space empty rather than drawing
            // something that is not the anatomy.
            canvas.style.display = "none";
          });
      })(canvases[i]);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else { boot(); }
})();
