#!/usr/bin/env python3
"""Static site generator for the SC-WBD public site.

Stdlib only, deliberately.  The site has no CDN dependencies and no npm
dependency for the HTML itself, so the only thing between a content file and a
deployable page is this script.

Content lives in ``site/content/**.html`` as fragments with a short key: value
header terminated by a line containing only ``---``.  Everything after that is
HTML, passed through with two inline expansions:

``[[note: text]]``
    A Tufte margin note.  Renders in the right margin on wide viewports and as
    a click-to-expand inline note on narrow ones.  Pure CSS, no JavaScript.

``[[src: path#anchor]]``
    A provenance chip.  Every number on this site is supposed to be traceable
    to a file in the repository; this is the markup that carries the trace.

Usage:  python3 site/build.py [--out DIR]
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
CONTENT = ROOT / "content"
STATIC = ROOT / "static"
TEMPLATE = ROOT / "templates" / "base.html"

# The repository the provenance chips point at.  Overridden by SITE_REPO_URL in
# the environment when the repo is published; until then chips render as plain
# paths rather than dead links.  See reports/site.md.
REPO_URL = os.environ.get("SITE_REPO_URL", "").rstrip("/")

# Base URL for rendered video.  Media is never committed to this repository; it
# is rendered locally and uploaded to R2.  Until the bucket is served publicly
# there is no URL to embed, so the page carries a Jacob prompt instead of a
# broken player.  See reports/site.md.
MEDIA_URL = os.environ.get("SITE_MEDIA_URL", "").rstrip("/")

VIDEOS = [
    ("scwbd-overview.mp4", "What SC-WBD is", "22 seconds"),
    ("scwbd-variance-channel.mp4", "The most expensive 134 seconds", "26 seconds"),
]


def video_section() -> str:
    """Embed the rendered videos, or say plainly why they are not embedded."""
    if not MEDIA_URL:
        return (
            '<div class="todo">\n'
            '<span class="todo-label">Jacob — video hosting</span>\n'
            "<p><strong>Two videos are rendered and uploaded to R2, but the bucket "
            "is not served publicly, so there is no URL to embed here.</strong> "
            "Making a bucket world-readable is a hosting posture decision, so it "
            "was left to a human rather than taken on your account.</p>\n"
            "<p>To publish them:</p>\n"
            "<p><code>npx wrangler r2 bucket dev-url enable scwbd-media</code><br>"
            "then rebuild with <code>SITE_MEDIA_URL=&lt;the r2.dev URL&gt; make site</code> "
            "and this block is replaced by the players.</p>\n"
            "<p>Alternatively attach a custom domain to the bucket, which is the "
            "better option for anything long-lived — <code>r2.dev</code> URLs are "
            "rate-limited and not intended for production traffic.</p>\n"
            "</div>"
        )

    parts = ['<div class="videos">']
    for filename, title, length in VIDEOS:
        parts.append(
            f'<figure>\n'
            f'  <video controls preload="metadata" playsinline '
            f'src="{MEDIA_URL}/{filename}"></video>\n'
            f"  <figcaption><strong>{html.escape(title)}</strong> · {length}</figcaption>\n"
            f"</figure>"
        )
    parts.append("</div>")
    return "\n".join(parts)


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    meta: dict[str, str] = {}
    if not text.startswith("---\n"):
        lines = text.split("\n")
        cut = None
        for i, line in enumerate(lines):
            if line.strip() == "---":
                cut = i
                break
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        if cut is None:
            raise SystemExit("front matter must be terminated by a line '---'")
        return meta, "\n".join(lines[cut + 1 :])
    raise SystemExit("front matter must not start with '---'")


NOTE_RE = re.compile(r"\[\[note:\s*(.*?)\]\]", re.S)
SRC_RE = re.compile(r"\[\[src:\s*(.*?)\]\]")


def expand_notes(body: str, page_id: str) -> str:
    """Expand [[note: ...]] into checkbox-toggled margin notes."""
    counter = [0]

    def repl(m: re.Match[str]) -> str:
        counter[0] += 1
        n = counter[0]
        nid = f"{page_id}-n{n}"
        inner = m.group(1).strip()
        return (
            f'<span class="note"><input type="checkbox" id="{nid}" class="note-toggle">'
            f'<label for="{nid}" class="note-ref">{n}</label>'
            f'<span class="note-body"><span class="note-n">{n}</span>{inner}</span></span>'
        )

    return NOTE_RE.sub(repl, body)


def expand_src(body: str) -> str:
    """Expand [[src: path]] into a provenance chip."""

    def repl(m: re.Match[str]) -> str:
        raw = m.group(1).strip()
        label = html.escape(raw)
        if REPO_URL:
            path = raw.split("#")[0].split(":")[0]
            href = f"{REPO_URL}/blob/master/{path}"
            return f'<a class="src" href="{href}">{label}</a>'
        return f'<span class="src">{label}</span>'

    return SRC_RE.sub(repl, body)


def nav_html(active: str, depth: int) -> str:
    up = "../" * depth
    items = [
        ("index", f"{up}index.html", "Overview"),
        ("engineering", f"{up}engineering/index.html", "Engineering"),
        ("speculative", f"{up}speculative/index.html", "Where this leads"),
        ("attribution", f"{up}attribution.html", "Attribution"),
    ]
    out = []
    for key, href, label in items:
        cls = ' class="active"' if key == active else ""
        out.append(f'<a href="{href}"{cls}>{label}</a>')
    return "\n        ".join(out)


def build(out_dir: Path) -> int:
    if not TEMPLATE.exists():
        raise SystemExit(f"missing template: {TEMPLATE}")
    template = TEMPLATE.read_text()

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # static assets
    if STATIC.exists():
        shutil.copytree(STATIC, out_dir / "static")

    # GitHub Pages: do not run Jekyll over the output.
    (out_dir / ".nojekyll").write_text("")

    # The paper is a build product, never a committed PDF.  `make paper` puts it
    # in paper/output/; if it is not there we warn rather than shipping a page
    # with a dead download button and no explanation.
    pdf = REPO / "paper" / "output" / "sc_wbd_frontiers.pdf"
    if pdf.exists():
        (out_dir / "paper").mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf, out_dir / "paper" / "sc-wbd.pdf")
        print(f"  paper: {pdf.stat().st_size // 1024} KB")
    else:
        print(f"  WARNING: {pdf} missing — run `make paper` first; "
              f"the download link on the landing page will 404")

    pages = sorted(CONTENT.rglob("*.html"))
    if not pages:
        raise SystemExit(f"no content found under {CONTENT}")

    count = 0
    for page in pages:
        rel = page.relative_to(CONTENT)
        meta, body = parse_front_matter(page.read_text())
        depth = len(rel.parts) - 1
        page_id = rel.with_suffix("").as_posix().replace("/", "-")

        body = expand_notes(body, page_id)
        body = expand_src(body)

        up = "../" * depth
        html_out = template
        replacements = {
            "{{title}}": html.escape(meta.get("title", "SC-WBD")),
            "{{description}}": html.escape(meta.get("description", "")),
            "{{kicker}}": meta.get("kicker", ""),
            "{{heading}}": meta.get("heading", meta.get("title", "")),
            "{{standfirst}}": meta.get("standfirst", ""),
            "{{nav}}": nav_html(meta.get("nav", ""), depth),
            "{{root}}": up,
            "{{body}}": body,
            "{{video_section}}": video_section(),
            "{{bodyclass}}": meta.get("bodyclass", ""),
        }
        for k, v in replacements.items():
            html_out = html_out.replace(k, v)

        # An unresolved placeholder is a build error, not a cosmetic issue.
        leftover = re.findall(r"\{\{(\w+)\}\}", html_out)
        if leftover:
            raise SystemExit(f"{rel}: unresolved template keys {sorted(set(leftover))}")

        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(html_out)
        count += 1

    print(f"built {count} pages -> {out_dir}")
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "_build"))
    args = ap.parse_args()
    build(Path(args.out))


if __name__ == "__main__":
    sys.exit(main())
