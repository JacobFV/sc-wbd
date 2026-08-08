/* Scroll behaviour for the SC-WBD site.
 *
 * The page is built as a run of <section class="chapter"> blocks (see
 * sectionize() in site/build.py). This file makes that structure visible while
 * reading, in four independent pieces, none of which the content depends on:
 *
 *   PROGRESS   a hairline at the top of the viewport, how far through the page
 *              you are.
 *   REVEAL     a chapter lifts into place the first time it enters the
 *              viewport, and STAYS placed. Re-hiding on scroll-up is a
 *              well-known way to make a page feel broken rather than alive.
 *   RAIL       the fixed section index tracks which chapter you are in.
 *   RECEDE     the opening chapter's hero drifts and fades as it leaves, so
 *              the transition out of the hero is continuous rather than a cut.
 *   SECTION    each pinned heading carries its own progress: `--secp` runs 0
 *              to 1 across that section, and the rule under the heading fills
 *              with it. The page progress bar says where you are in the
 *              document; this says where you are in the argument.
 *   COUNT      key figures count up the first time they are seen. A number
 *              that arrives at its value reads as a measurement; the same
 *              number already sitting there reads as decoration.
 *
 * Everything is progressive enhancement: with this file absent or JavaScript
 * off, the CSS leaves every chapter at rest and fully visible, and the rail is
 * a plain list of anchors. `prefers-reduced-motion` takes the same path.
 *
 * One shared rAF-throttled scroll handler drives PROGRESS, RAIL and RECEDE.
 * Three separate listeners each doing their own layout reads is how a scroll
 * handler ends up costing more than the drawing it is decorating.
 */
(function () {
  "use strict";

  var reduce = window.matchMedia &&
               window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function boot() {
    var chapters = Array.prototype.slice.call(
      document.querySelectorAll(".chapter"));
    var bar = document.getElementById("scrollprog");
    var railLinks = Array.prototype.slice.call(
      document.querySelectorAll(".railnav a[data-rail]"));
    var hero = document.querySelector(".chapter-intro");

    /* The stylesheet holds chapters at rest until this class says otherwise.
     * Setting it on <html> rather than per-chapter means a browser that never
     * runs this file shows everything, which is the correct failure. */
    document.documentElement.classList.add("js-reveal");

    // ---------------------------------------------------------- REVEAL
    if (reduce || !("IntersectionObserver" in window)) {
      chapters.forEach(function (c) { c.classList.add("in"); });
    } else {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (!e.isIntersecting) return;
          e.target.classList.add("in");
          io.unobserve(e.target);          // once placed, it stays placed
        });
      }, { rootMargin: "0px 0px -12% 0px", threshold: 0.04 });
      chapters.forEach(function (c) { io.observe(c); });

      /* Anything already on screen at load must not wait for a scroll event
       * that may never come. */
      requestAnimationFrame(function () {
        chapters.forEach(function (c) {
          if (c.getBoundingClientRect().top < window.innerHeight * 0.9) {
            c.classList.add("in");
            io.unobserve(c);
          }
        });
      });
    }

    // ------------------------------------------------------------- COUNT
    /* "2.5M" is a number and a suffix; "414" is a number. Split them once,
     * here, so the animation never has to parse anything and the suffix comes
     * back verbatim rather than reconstructed. */
    if (!reduce && "IntersectionObserver" in window) {
      var figs = Array.prototype.slice.call(document.querySelectorAll(".keyfig .n"));
      var co = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (!e.isIntersecting) return;
          co.unobserve(e.target);
          countUp(e.target);
        });
      }, { threshold: 0.6 });
      figs.forEach(function (el) {
        var m = /^([0-9]+(?:\.[0-9]+)?)(.*)$/.exec(el.textContent.trim());
        if (!m) return;
        el.dataset.to = m[1];
        el.dataset.suffix = m[2];
        el.dataset.dp = (m[1].split(".")[1] || "").length;
        el.textContent = (0).toFixed(el.dataset.dp) + m[2];
        co.observe(el);
      });
    }

    function countUp(el) {
      var to = parseFloat(el.dataset.to);
      var dp = parseInt(el.dataset.dp, 10) || 0;
      var suffix = el.dataset.suffix || "";
      var t0 = null, dur = 900;
      function step(ts) {
        if (t0 === null) t0 = ts;
        var p = Math.min(1, (ts - t0) / dur);
        var eased = 1 - Math.pow(1 - p, 3);   // settle, do not snap
        el.textContent = (to * eased).toFixed(dp) + suffix;
        if (p < 1) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    }

    // ------------------------------------------------- PROGRESS, RAIL, RECEDE
    var byId = {};
    railLinks.forEach(function (a) { byId[a.getAttribute("data-rail")] = a; });
    var current = null;

    function frame() {
      ticking = false;
      var y = window.pageYOffset || document.documentElement.scrollTop;
      var vh = window.innerHeight;

      if (bar) {
        var span = document.documentElement.scrollHeight - vh;
        bar.style.transform = "scaleX(" +
          (span > 0 ? Math.min(1, Math.max(0, y / span)) : 0) + ")";
      }

      /* The chapter you are "in" is the last one whose top has passed a line a
       * third of the way down the viewport. Using the viewport top instead
       * makes the rail flip a section early on every heading. */
      if (railLinks.length) {
        var line = vh * 0.34, active = null;
        for (var i = 0; i < chapters.length; i++) {
          var id = chapters[i].id;
          if (!byId[id]) continue;
          if (chapters[i].getBoundingClientRect().top <= line) active = id;
        }
        if (active !== current) {
          if (current && byId[current]) byId[current].classList.remove("here");
          if (active && byId[active]) byId[active].classList.add("here");
          current = active;
        }
      }

      if (hero && !reduce) {
        var r = hero.getBoundingClientRect();
        // 0 while the hero owns the viewport, 1 once it has fully left.
        var p = Math.min(1, Math.max(0, -r.top / Math.max(1, r.height * 0.72)));
        hero.style.setProperty("--sp", p.toFixed(4));
      }

      /* Per-section progress. Measured against the heading's own sticky
       * window -- from the moment the section's top passes the pin line to
       * the moment its bottom does -- so the bar is full exactly when the
       * heading is about to be pushed out by the next one. */
      for (var s = 0; s < chapters.length; s++) {
        var cr = chapters[s].getBoundingClientRect();
        if (cr.bottom < -200 || cr.top > vh + 200) continue;
        var span = Math.max(1, cr.height - vh * 0.2);
        var sp = Math.min(1, Math.max(0, (vh * 0.2 - cr.top) / span));
        chapters[s].style.setProperty("--secp", sp.toFixed(3));
      }
    }

    var ticking = false;
    function onScroll() {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(frame);
    }

    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll, { passive: true });
    frame();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else { boot(); }
})();
