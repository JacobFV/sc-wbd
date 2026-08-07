/* Interactive architecture diagram: how the graph materialises for a use case.
 *
 * Three panels on one canvas.
 *
 *   LEFT    an axonometric scene -- the 414 real parcels, plus a cord with
 *           terminal nodes standing for what enters and leaves the body.
 *           Draggable, shares its geometry with the homepage mark.
 *   MIDDLE  three to five signal categories for the selected use case.
 *   RIGHT   what the use case actually is, drawn as a small technical figure.
 *
 * Annotation lines run from the PROJECTED screen position of each dot group to
 * its category, so they move as you orbit. That is the point of the drawing:
 * which parcels and which terminals a use case actually touches, and under which
 * attachment kind.
 *
 * Palette: pale ivory ground, blue-grey linework, one warm accent for anything
 * leaving the body. Extremely fine lines -- 0.6px at 1x -- because the density is
 * the information.
 */
(function () {
  "use strict";

  // Read from the page, not hardcoded, so the drawing follows the device's
  // light/dark preference and the theme toggle. A canvas cannot inherit CSS, so
  // the values are resolved at draw time and re-resolved when the theme changes.
  var INK, LINE, FAINT, WARM, COOL, GROUND;
  function readTheme() {
    var cs = getComputedStyle(document.documentElement);
    function v(name, fallback) {
      var got = cs.getPropertyValue(name).trim();
      return got || fallback;
    }
    GROUND = v("--bg", "#f7f5ef");
    INK    = v("--ink", "#27262b");
    LINE   = v("--ink-3", "#8c9bad");
    FAINT  = v("--rule", "#c3ccd6");
    // The two signal hues stay constant across themes: they encode attachment
    // kind, which does not change with the lighting.
    COOL = "#5b7fa6";
    WARM = "#c8874a";
  }

  // Attachment kinds, in the order the schema declares them.
  var USE_CASES = [
    {
      id: "bci",
      label: "Controlling a computer",
      right: "computer",
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
      right: "screen",
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
    var N = data.n, P = data.p, F = data.f;
    var isSub = data.div.map(function (d) { return d === "sub"; });

    var yaw = -0.62, pitch = -0.16, dragging = false, lx = 0, ly = 0, idle = true;
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var uc = 0;

    // Terminal nodes: a cord below the brain, with sensory in and motor out.
    // Positions are in the same unit cube as the parcels.
    var CORD = [];
    for (var s = 0; s <= 7; s++) CORD.push([0, -0.06, -1.05 - s * 0.20]);
    var SENSORY = [[-0.45, 0.10, -2.55], [-0.58, -0.05, -2.42], [-0.36, 0.22, -2.68]];
    var MOTOR = [[0.45, -0.10, -2.55], [0.58, 0.05, -2.42], [0.36, -0.22, -2.68]];

    function resize() {
      var r = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(r.width * dpr));
      canvas.height = Math.max(1, Math.round(r.height * dpr));
    }

    function project(v, s, ox, oy) {
      var cy = Math.cos(yaw), sy = Math.sin(yaw);
      var cp = Math.cos(pitch), sp = Math.sin(pitch);
      var x1 = v[0] * cy + v[1] * sy, y1 = -v[0] * sy + v[1] * cy;
      var y2 = y1 * cp - v[2] * sp, z2 = y1 * sp + v[2] * cp;
      return [ox + x1 * s, oy - z2 * s, y2];
    }

    function hairline(w) { ctx.lineWidth = Math.max(0.6, w) * dpr; }

    function draw() {
      readTheme();
      var W = canvas.width, H = canvas.height;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = GROUND;
      ctx.fillRect(0, 0, W, H);

      var c = USE_CASES[uc];
      var sceneW = W * 0.40, midX = W * 0.53, rightX = W * 0.76;
      var s = Math.min(sceneW, H) * 0.19;
      var ox = sceneW * 0.52, oy = H * 0.40;

      // ---- scene: cord ----
      ctx.strokeStyle = LINE; hairline(0.9);
      ctx.beginPath();
      for (var i = 0; i < CORD.length; i++) {
        var q = project(CORD[i], s, ox, oy);
        if (i === 0) ctx.moveTo(q[0], q[1]); else ctx.lineTo(q[0], q[1]);
      }
      ctx.stroke();

      // ---- scene: parcels ----
      var xs = new Float32Array(N), ys = new Float32Array(N), dep = new Float32Array(N);
      var order = [];
      for (var j = 0; j < N; j++) {
        var pr = project([P[j * 3], P[j * 3 + 1], P[j * 3 + 2]], s, ox, oy);
        xs[j] = pr[0]; ys[j] = pr[1]; dep[j] = pr[2]; order.push(j);
      }
      order.sort(function (a, b) { return dep[a] - dep[b]; });

      // Which parcels this use case touches, and WHICH ONES belong to which
      // category. Every category owns a distinct parcel set so its annotation
      // can land on its own points instead of all of them pointing at the
      // centre of the skull, which said nothing.
      var srcs = {};
      c.cats.forEach(function (k) { srcs[k.src] = true; });
      var lit = new Uint8Array(N);
      for (var m = 0; m < N; m++) {
        lit[m] = (srcs.all || (srcs.cortex && !isSub[m])) ? 1 : 0;
      }

      // ---- scene: tractography ----
      // The connectome, drawn as the structure it is. Without it the cord was
      // the only visible wiring, which implied the brain was a cloud of
      // unconnected points -- the opposite of what the model is about.
      // Strongest 900 edges, weight as opacity, depth-faded, under the parcels.
      var E = (data.edges && data.edges.e) || [], EW = (data.edges && data.edges.w) || [];
      if (E.length) {
        ctx.strokeStyle = LINE;
        hairline(0.6);
        for (var e = 0; e < E.length; e += 2) {
          var ea = E[e], eb = E[e + 1];
          if (!lit[ea] && !lit[eb]) continue;
          var mid = ((dep[ea] + dep[eb]) / 2 + 1) / 2;
          ctx.globalAlpha = 0.045 + 0.16 * EW[e / 2] * mid;
          ctx.beginPath();
          ctx.moveTo(xs[ea], ys[ea]);
          ctx.lineTo(xs[eb], ys[eb]);
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }


      for (var k2 = 0; k2 < N; k2++) {
        var jj = order[k2], t = (dep[jj] + 1) / 2;
        ctx.globalAlpha = lit[jj] ? (0.34 + 0.5 * t) : 0.10;
        ctx.fillStyle = lit[jj] ? COOL : FAINT;
        ctx.beginPath();
        ctx.arc(xs[jj], ys[jj], (isSub[jj] ? 2.7 : 1.7) * dpr * (0.6 + 0.5 * t), 0, 6.2832);
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      // ---- scene: terminals ----
      function terminals(list, on) {
        var pts = [];
        list.forEach(function (v) {
          var q = project(v, s, ox, oy);
          pts.push(q);
          ctx.fillStyle = on ? WARM : FAINT;
          ctx.globalAlpha = on ? 0.9 : 0.25;
          ctx.beginPath(); ctx.arc(q[0], q[1], 3.1 * dpr, 0, 6.2832); ctx.fill();
        });
        ctx.globalAlpha = 1;
        return pts;
      }
      var sensPts = terminals(SENSORY, !!srcs.sensory);
      var motPts = terminals(MOTOR, !!srcs.motor);

      // The screen points each category actually refers to. A category draws one
      // fine line to EACH of a small sample of its own points, so the annotation
      // shows where in the anatomy the signal lives.
      function parcelSample(pred, want) {
        var picks = [];
        // deterministic stride, so the sample does not shimmer while orbiting
        var step = Math.max(1, Math.floor(N / want));
        for (var q = 0; q < N && picks.length < want; q += step) {
          if (pred(q)) picks.push([xs[q], ys[q]]);
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
      ctx.textBaseline = "middle";

      c.cats.forEach(function (cat, idx) {
        var y = top + idx * gap;
        var col = kindColor(cat.k);
        var pts = targets(cat.src);
        var hub = [midX - 26 * dpr, y];

        // Fan: one hairline from the elbow to each point this category names.
        ctx.strokeStyle = col; hairline(0.6);
        pts.forEach(function (pt) {
          ctx.globalAlpha = 0.30;
          ctx.beginPath();
          ctx.moveTo(pt[0], pt[1]);
          ctx.lineTo(hub[0], hub[1]);
          ctx.stroke();
          ctx.globalAlpha = 0.85;
          ctx.beginPath();
          ctx.arc(pt[0], pt[1], 1.9 * dpr, 0, 6.2832);
          ctx.fillStyle = col; ctx.fill();
        });
        ctx.globalAlpha = 0.7;
        ctx.beginPath();
        ctx.moveTo(hub[0], hub[1]);
        ctx.lineTo(midX - 10 * dpr, y);
        ctx.stroke();
        ctx.globalAlpha = 1;

        ctx.fillStyle = col;
        ctx.beginPath(); ctx.arc(midX - 10 * dpr, y, 2.2 * dpr, 0, 6.2832); ctx.fill();

        ctx.fillStyle = INK;
        ctx.font = "600 " + (11.5 * dpr) + "px ui-sans-serif, system-ui, sans-serif";
        ctx.fillText(cat.t, midX, y - 5 * dpr);
        ctx.fillStyle = LINE;
        ctx.font = (9.5 * dpr) + "px ui-sans-serif, system-ui, sans-serif";
        ctx.fillText(cat.k + " · " + cat.note, midX, y + 8 * dpr);
      });

      // ---- right: the use case, as a small technical figure ----
      ctx.strokeStyle = INK; hairline(0.9); ctx.fillStyle = "none";
      var rx = rightX, ry = H * 0.40, u = Math.min(W * 0.09, H * 0.16);
      ctx.beginPath();
      if (c.right === "computer") {
        ctx.rect(rx, ry - u * 0.6, u * 1.6, u);            // monitor
        ctx.moveTo(rx + u * 0.8, ry + u * 0.4);
        ctx.lineTo(rx + u * 0.8, ry + u * 0.7);
        ctx.moveTo(rx + u * 0.4, ry + u * 0.7);
        ctx.lineTo(rx + u * 1.2, ry + u * 0.7);
        ctx.moveTo(rx + u * 0.35, ry - u * 0.2);           // cursor
        ctx.lineTo(rx + u * 0.35, ry + u * 0.1);
      } else if (c.right === "tms") {
        ctx.arc(rx + u * 0.5, ry, u * 0.42, 0, 6.2832);    // coil, figure-of-eight
        ctx.moveTo(rx + u * 1.34, ry);
        ctx.arc(rx + u * 0.92, ry, u * 0.42, 0, 6.2832);
        ctx.moveTo(rx + u * 1.34, ry);                     // arm
        ctx.lineTo(rx + u * 2.0, ry - u * 0.5);
      } else {
        ctx.rect(rx, ry - u * 0.6, u * 1.7, u * 1.1);      // screen
        ctx.moveTo(rx + u * 0.25, ry - u * 0.25);
        ctx.lineTo(rx + u * 1.45, ry - u * 0.25);
        ctx.moveTo(rx + u * 0.25, ry + u * 0.05);
        ctx.lineTo(rx + u * 1.1, ry + u * 0.05);
      }
      ctx.stroke();

      ctx.fillStyle = INK;
      ctx.font = "600 " + (11 * dpr) + "px ui-sans-serif, system-ui, sans-serif";
      ctx.fillText(c.label, rx, ry + u * 1.15);
    }

    var raf;
    function frame() { if (idle) yaw += 0.0022; draw(); raf = requestAnimationFrame(frame); }

    function down(e) {
      dragging = true; idle = false;
      var q = e.touches ? e.touches[0] : e; lx = q.clientX; ly = q.clientY;
    }
    function move(e) {
      if (!dragging) return;
      var q = e.touches ? e.touches[0] : e;
      yaw += (q.clientX - lx) * 0.009;
      pitch = Math.max(-1.3, Math.min(1.3, pitch + (q.clientY - ly) * 0.007));
      lx = q.clientX; ly = q.clientY;
      if (e.cancelable) e.preventDefault();
    }
    function up() { dragging = false; }

    canvas.addEventListener("mousedown", down);
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    canvas.addEventListener("touchstart", down, { passive: true });
    canvas.addEventListener("touchmove", move, { passive: false });
    canvas.addEventListener("touchend", up);
    canvas.style.cursor = "grab";
    window.addEventListener("resize", function () { resize(); draw(); });

    // One canvas per section. Each use case gets its own <section> in the page,
    // and every canvas carrying data-case renders that case only. Tabs hid two
    // thirds of the argument behind a click; sections let a reader scroll it.
    var want = canvas.getAttribute("data-case");
    if (want) {
      for (var w2 = 0; w2 < USE_CASES.length; w2++) {
        if (USE_CASES[w2].id === want) { uc = w2; break; }
      }
    }

    resize();
    var rm = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)");
    if (rm && rm.matches) { idle = false; draw(); } else { frame(); }
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
