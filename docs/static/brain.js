/* The WBD mark: 414 parcels at their real anatomical coordinates.
 *
 * Not a generic brain graphic. The points are `AnatomyPrior.positions` in the
 * fsLR_32k surface RAS frame -- the same coordinates the lead field integrates
 * against and the same 400 cortical + 14 subcortical parcels the model carries
 * state for. Colour is the 9-family partition, the one separated on a 20-tracer
 * PET receptor panel and on myelin+thickness under a 1000-spin Vasa null.
 *
 * Raw canvas 2D with a hand-rolled projection: no WebGL context to lose, no
 * library to load, works with a strict CSP, and degrades to a still frame if
 * anything fails. ~10 KB of geometry.
 *
 * Drag to orbit. It auto-rotates until touched, then hands control over.
 */
(function () {
  "use strict";

  function init(canvas, data) {
    var ctx = canvas.getContext("2d");
    if (!ctx) return;

    var N = data.n, P = data.p, F = data.f;
    var isSub = data.div.map(function (d) { return d === "sub"; });

    // Two cortical families in the house blues, subcortex warm. Distinguishable
    // for the ~8% of men with a red-green deficiency: the split that carries
    // meaning here is cortex-vs-subcortex, which is blue-vs-amber, not red-green.
    var COL = [
      [ 96, 165, 232], [ 52, 110, 190],
      [232, 156,  74], [225, 133,  92], [214, 160,  58],
      [236, 176, 102], [204, 122,  70], [244, 194, 122], [214, 145,  48]
    ];

    var yaw = -0.5, pitch = -0.18, dragging = false, lx = 0, ly = 0, idle = true;
    var dpr = Math.min(window.devicePixelRatio || 1, 2);

    function resize() {
      var r = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(r.width * dpr));
      canvas.height = Math.max(1, Math.round(r.height * dpr));
    }

    var order = new Array(N), depth = new Float32Array(N);
    for (var i = 0; i < N; i++) order[i] = i;

    function draw() {
      var w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      var cy = Math.cos(yaw), sy = Math.sin(yaw);
      var cp = Math.cos(pitch), sp = Math.sin(pitch);
      var s = Math.min(w, h) * 0.34, ox = w / 2, oy = h / 2;

      var xs = new Float32Array(N), ys = new Float32Array(N);
      for (var i = 0; i < N; i++) {
        // Anatomical RAS: x right, y anterior, z superior. Screen up is +z, so
        // the model is viewed from the left-anterior with z mapped to -screenY.
        var X = P[i * 3], Y = P[i * 3 + 1], Z = P[i * 3 + 2];
        var x1 = X * cy + Y * sy, y1 = -X * sy + Y * cy;
        var y2 = y1 * cp - Z * sp, z2 = y1 * sp + Z * cp;
        xs[i] = ox + x1 * s;
        ys[i] = oy - z2 * s;
        depth[i] = y2;
      }

      order.sort(function (a, b) { return depth[a] - depth[b]; });

      for (var k = 0; k < N; k++) {
        var j = order[k];
        var t = (depth[j] + 1) / 2;                 // 0 far .. 1 near
        var c = COL[F[j]] || COL[0];
        var rad = (isSub[j] ? 3.4 : 2.2) * dpr * (0.62 + 0.55 * t);
        ctx.globalAlpha = 0.30 + 0.62 * t;
        ctx.fillStyle = "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
        ctx.beginPath();
        ctx.arc(xs[j], ys[j], rad, 0, 6.2832);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    var raf = null;
    function frame() {
      if (idle) yaw += 0.0032;
      draw();
      raf = window.requestAnimationFrame(frame);
    }

    function down(e) {
      dragging = true; idle = false;
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
      if (e.cancelable) e.preventDefault();
    }
    function up() { dragging = false; canvas.style.cursor = "grab"; }

    canvas.addEventListener("mousedown", down);
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    canvas.addEventListener("touchstart", down, { passive: true });
    canvas.addEventListener("touchmove", move, { passive: false });
    canvas.addEventListener("touchend", up);
    canvas.style.cursor = "grab";

    window.addEventListener("resize", function () { resize(); draw(); });
    // Follow the page theme: the mark sits on the page background, so a theme
    // flip must repaint it.
    var mo = new MutationObserver(function () { draw(); });
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    var mq2 = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)");
    if (mq2 && mq2.addEventListener) mq2.addEventListener("change", draw);
    resize();

    // Respect a reduced-motion preference: render once, stay draggable.
    var rm = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)");
    if (rm && rm.matches) { idle = false; draw(); }
    else { frame(); }
  }

  function boot() {
    var canvas = document.getElementById("wbd-brain");
    if (!canvas) return;
    fetch(canvas.getAttribute("data-src") || "/brain.json")
      .then(function (r) { return r.json(); })
      .then(function (d) { init(canvas, d); })
      .catch(function () {
        // No geometry, no mark. Leave the space empty rather than drawing
        // something that is not the anatomy.
        canvas.style.display = "none";
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else { boot(); }
})();
