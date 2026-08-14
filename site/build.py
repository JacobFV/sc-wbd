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

``{{paper}}``
    The release artifact: page one of the PDF, rendered at build time, over
    the page count and file size read off that same PDF.

``{{video:<name>}}``
    One embedded player, placed by the page that discusses it.

Every ``<h2>``-delimited run is additionally wrapped in a ``<section>`` and
indexed into a rail nav; see ``sectionize()``.

Usage:  python3 site/build.py [--out DIR]
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import os
import re
import shutil
import subprocess
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

# Both are in the bucket and serve 200 at their exact rendered byte lengths
# (2,723,304 and 2,890,798). `scwbd-variance-channel.mp4` was missing for a while
# because it had been rendered in a worktree -- scwbd-wt/tufte/video/out -- and
# `make video-upload` ran from the main checkout, where video/out did not exist.
# The renders now live here too.
#
# The lengths are the composition durations in video/src -- Overview is 1560
# frames at 30 fps -- because the file itself is never in this repository for
# the build to measure.  Re-render and this list moves with it.
VIDEOS = [
    ("scwbd-overview.mp4", "What SC-WBD is, and what it is for", "52 seconds"),
    ("scwbd-variance-channel.mp4", "Isolating the variance channel", "26 seconds"),
]


def video_embed(name: str) -> str:
    """One player, placed by the page that discusses it.

    Was `video_section()`: every video, all on the landing page, in a "Video"
    heading. A film about the variance channel belongs in the variance-channel
    article, next to the argument it illustrates -- not in a gallery on the front
    page, where it is a list of assets rather than an explanation of anything.

    Pages ask for one by `{{video:scwbd-overview}}`.
    """
    entry = next((v for v in VIDEOS if v[0].startswith(name)), None)
    if entry is None:
        return ""
    filename, title, length = entry
    if not MEDIA_URL:
        return (
            '<div class="todo">\n'
            '<span class="todo-label">video hosting</span>\n'
            f"<p>{html.escape(title)} is rendered and in R2, but this build got no "
            "<code>SITE_MEDIA_URL</code>, so there is nothing to embed. Rebuild with "
            "<code>make site</code>, which now exports it.</p>\n"
            "</div>"
        )
    return (
        '<figure class="videofig">\n'
        f'  <video controls preload="metadata" playsinline '
        f'aria-label="{html.escape(title)}, {length}" '
        f'src="{MEDIA_URL}/{filename}"></video>\n'
        "</figure>"
    )


# --------------------------------------------------------------- the paper
#
# The paper is the release artifact, so the page shows it rather than
# describing it: page one, rendered at build time from the PDF that is being
# published.  Every number in the block below -- page count, file size, build
# date, image dimensions -- is read off that same file.  Nothing here is a
# figure typed into the content and left to rot; the previous copy claimed
# "29pp" against a 46-page PDF, which is exactly the failure this avoids.

PAPER: dict[str, object] = {}


def png_size(path: Path) -> tuple[int, int] | None:
    """Width and height out of a PNG's IHDR, so the <img> reserves its box."""
    head = path.read_bytes()[:24]
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


def pdf_pages(pdf: Path) -> int | None:
    try:
        out = subprocess.run(
            ["pdfinfo", str(pdf)], capture_output=True, text=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"^Pages:\s+(\d+)", out, re.M)
    return int(m.group(1)) if m else None


def render_first_page(pdf: Path, dest: Path) -> Path | None:
    """Page one as a PNG.  Absent `pdftoppm` this returns None and the page
    falls back to the download button alone -- a missing preview is a degraded
    page, not a failed build."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    stem = dest.with_suffix("")
    # 780px wide: the shot is displayed in a 19rem column, so this is a little
    # over 2x for a retina panel and a third of the bytes of a 150 dpi render.
    try:
        subprocess.run(
            ["pdftoppm", "-png", "-f", "1", "-l", "1", "-singlefile",
             "-scale-to-x", "780", "-scale-to-y", "-1", str(pdf), str(stem)],
            check=True, capture_output=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        print("  WARNING: pdftoppm unavailable or failed — no paper preview "
              "will be shown; install poppler-utils")
        return None
    return dest if dest.exists() else None


def stage_paper(out_dir: Path) -> None:
    """Copy the PDF, render its first page, and record what was actually
    published into PAPER for the {{paper}} macro to read."""
    pdf = REPO / "paper" / "output" / "sc_wbd_frontiers.pdf"
    if not pdf.exists():
        print(f"  WARNING: {pdf} missing — run `make paper` first; "
              f"the download link on the landing page will 404")
        return

    (out_dir / "paper").mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf, out_dir / "paper" / "sc-wbd.pdf")

    # This step COPIES; it does not compile.  The old message read
    # "paper: 269 KB", which is a report about a copy phrased like a report
    # about a build -- and a day of edits to body.tex went out unpublished
    # behind it, because the number never changed and nothing said why.
    srcs = list((REPO / "paper").glob("*.tex")) + [REPO / "paper" / "references.bib"]
    newest = max((s.stat().st_mtime for s in srcs if s.exists()), default=0.0)
    kb = pdf.stat().st_size // 1024
    if newest > pdf.stat().st_mtime:
        def _t(ts: float) -> str:
            return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")

        print(
            f"  WARNING: paper PDF is STALE — copied {kb} KB built "
            f"{_t(pdf.stat().st_mtime)}, but a source changed at {_t(newest)}. "
            f"This step copies, it does not compile. Run `make paper` or the "
            f"site ships the previous text."
        )
    else:
        print(f"  paper: copied {kb} KB (PDF is newer than every .tex source)")

    PAPER["pages"] = pdf_pages(pdf)
    PAPER["kb"] = kb
    PAPER["built"] = dt.datetime.fromtimestamp(pdf.stat().st_mtime).strftime("%d %B %Y")

    shot = render_first_page(pdf, out_dir / "paper" / "first-page.png")
    if shot is not None:
        PAPER["shot"] = "paper/first-page.png"
        dims = png_size(shot)
        if dims:
            PAPER["w"], PAPER["h"] = dims
        print(f"  paper: rendered page 1 -> {shot.name} "
              f"({shot.stat().st_size // 1024} KB)")


def paper_block(up: str) -> str:
    """The paper, shown and downloadable, and nothing else.

    This used to render a spec sheet -- format, byte size, build date, source
    filename -- next to the shot.  None of it is what a reader wants from a
    paper.  They want to see that it is real and then open it, so the block is
    now the cover and a button, and the page count that survives is the one on
    the button because it tells you what you are committing to.
    """
    if not PAPER:
        return (
            '<div class="todo">\n'
            '<span class="todo-label">paper missing</span>\n'
            "<p>This build found no <code>paper/output/sc_wbd_frontiers.pdf</code>, "
            "so there is nothing to show or link. Run <code>make paper</code>.</p>\n"
            "</div>"
        )

    href = f"{up}paper/sc-wbd.pdf"

    if PAPER.get("shot"):
        dims = ""
        if PAPER.get("w"):
            dims = f' width="{PAPER["w"]}" height="{PAPER["h"]}"'
        pages = PAPER.get("pages")
        # With the caption gone this is the only place the format and length
        # are stated, so it carries them.
        alt = "First page of the SC-WBD paper"
        if pages:
            alt += f" \u2014 PDF, {pages} pages"
        shot = (
            f'  <a class="paper-shot" href="{href}">\n'
            f'    <img src="{up}{PAPER["shot"]}" alt="{alt}"'
            f'{dims} loading="lazy" decoding="async">\n'
            f'    <span class="paper-shot-open">Open the PDF</span>\n'
            f"  </a>\n"
        )
    else:
        shot = ""

    # No button and no caption.  The cover IS the link; a button under it is a
    # second control for the same action and a caption under it is a label on a
    # picture of the thing it labels.  The page count and format live in the
    # `alt` text and in the PDF itself.
    return f'<div class="paper-release">\n{shot}</div>'


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


# ------------------------------------------------------------ table labels
#
# A table is a grid, and a phone has no grid to put it in.  The first attempt
# at making these fit was to let cells break words wherever they liked, which
# does make the table fit and destroys it on the way: the Artifacts table came
# out as a column of hyphen-less fragments stacked one character wide.
#
# What a narrow screen can show is one ROW at a time, as a small labelled
# block.  That needs each cell to carry its column's name, because the header
# row is far away by then -- so it is stamped on here, at build time, from the
# table's own `<thead>`.  Authors keep writing plain tables; the stylesheet
# reads `data-label` and does the rest under `max-width: 46rem`.

THEAD_RE = re.compile(r"<thead>(.*?)</thead>", re.S)
TH_RE = re.compile(r"<th\b[^>]*>(.*?)</th>", re.S)
TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.S)
ROW_RE = re.compile(r"<tr\b[^>]*>.*?</tr>", re.S)
CELL_RE = re.compile(r"<td(\s[^>]*)?>", re.S)


def label_table_cells(body: str) -> str:
    """Stamp every `<td>` with its column heading as `data-label`."""

    def per_table(m: re.Match[str]) -> str:
        table = m.group(0)
        head = THEAD_RE.search(table)
        if not head:
            return table
        headings = [text_of(h) for h in TH_RE.findall(head.group(1))]
        if not headings:
            return table

        def per_row(rm: re.Match[str]) -> str:
            row = rm.group(0)
            # `<thead>`'s own row has no `<td>`, so it falls through untouched.
            i = 0

            def per_cell(cm: re.Match[str]) -> str:
                nonlocal i
                attrs = cm.group(1) or ""
                label = headings[i] if i < len(headings) else ""
                i += 1
                if not label or "data-label" in attrs:
                    return cm.group(0)
                return f'<td{attrs} data-label="{html.escape(label, quote=True)}">'

            return CELL_RE.sub(per_cell, row)

        return ROW_RE.sub(per_row, table)

    return TABLE_RE.sub(per_table, body)


# ------------------------------------------------------------- sectioning
#
# The pages are authored as a flat run of headings and prose, which is the
# right thing to write and the wrong thing to lay out: a flat run scrolls as
# one undifferentiated column.  Each `<h2>`-delimited stretch is wrapped here
# into a `<section>` so the stylesheet and reveal.js can treat it as a unit --
# reveal it on entry, tint alternate ones, and track which one the reader is
# in.  Doing it at build time rather than in the content keeps the authoring
# format unchanged and applies the same treatment to every page.

TAG_RE = re.compile(r"<[^>]+>")
H2_RE = re.compile(r"(?m)^(<h2\b[^>]*>.*?</h2>)")


def text_of(markup: str) -> str:
    return html.unescape(TAG_RE.sub("", markup)).strip()


def slugify(markup: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text_of(markup).lower()).strip("-")
    return s[:48] or "section"


def short_label(markup: str) -> str:
    """A rail label has about twenty characters of room.  Headings on the
    engineering pages are sentences, so cut at the first colon or dash -- which
    is where those sentences put their subject -- before truncating."""
    s = text_of(markup)
    s = re.split(r"\s+[—–:]\s+|:\s+", s, maxsplit=1)[0]
    return s if len(s) <= 30 else s[:29].rstrip() + "…"


def sectionize(body: str, page_id: str) -> tuple[str, list[tuple[str, str]]]:
    """Wrap each heading-delimited run in a <section>.

    Returns the rewritten body and the section table the rail nav is built
    from.  A page whose prose container cannot be found is passed through
    untouched rather than mangled.
    """
    open_m = re.search(r'<div class="prose">', body)
    close = body.rfind("</div>")
    if not open_m or close < open_m.end():
        return body, []

    head, inner, tail = body[: open_m.end()], body[open_m.end() : close], body[close:]

    parts = H2_RE.split(inner)
    intro, rest = parts[0], parts[1:]
    if not rest:
        return body, []

    out: list[str] = []
    table: list[tuple[str, str]] = []

    if intro.strip():
        out.append(f'<section class="chapter chapter-intro" id="{page_id}-top">\n'
                   f"{intro.rstrip()}\n</section>\n")

    seen: dict[str, int] = {}
    for i in range(0, len(rest), 2):
        h2, chunk = rest[i], rest[i + 1] if i + 1 < len(rest) else ""
        slug = slugify(h2)
        seen[slug] = seen.get(slug, 0) + 1
        if seen[slug] > 1:                       # two headings, one slug
            slug = f"{slug}-{seen[slug]}"
        n = i // 2 + 1
        tint = " tinted" if n % 2 == 0 else ""
        table.append((slug, short_label(h2)))
        out.append(
            f'<section class="chapter{tint}" id="{slug}">\n'
            f"{h2}\n{chunk.rstrip()}\n</section>\n"
        )

    return head + "\n" + "".join(out) + tail, table


def rail_html(table: list[tuple[str, str]]) -> str:
    """A fixed index of the page's own sections, highlighted as you pass
    through them.  Suppressed on short pages, where it would be furniture."""
    if len(table) < 3:
        return ""
    items = "".join(
        f'    <li><a href="#{slug}" data-rail="{slug}">'
        f'<span class="rail-label">{html.escape(label)}</span></a></li>\n'
        for slug, label in table
    )
    return (
        '<nav class="railnav" aria-label="Sections on this page">\n'
        f"  <ol>\n{items}  </ol>\n"
        "</nav>\n"
    )


def nav_html(active: str, depth: int) -> str:
    up = "../" * depth
    items = [
        ("index", f"{up}index.html", "Overview"),
        ("engineering", f"{up}engineering/index.html", "Engineering"),
        ("possibilities", f"{up}possibilities/index.html", "Possibilities"),
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

    # `/speculative/` was published; the page is now `/possibilities/`.  A
    # renamed page that 404s is a broken link on somebody else's site, so the
    # old path is kept alive as a redirect rather than deleted.  Cloudflare
    # Pages reads this file; GitHub Pages ignores it, which is why the old
    # directory is not simply removed from the output.
    (out_dir / "_redirects").write_text(
        "/speculative/*  /possibilities/:splat  301\n"
        "/speculative    /possibilities/        301\n"
    )

    # The paper is a build product, never a committed PDF.  `make paper` puts it
    # in paper/output/; if it is not there we warn rather than shipping a page
    # with a dead download button and no explanation.
    stage_paper(out_dir)

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
        body = label_table_cells(body)
        body, sections = sectionize(body, page_id)

        up = "../" * depth
        html_out = template
        replacements = {
            "{{rail}}": rail_html(sections) if meta.get("rail") != "off" else "",
            "{{title}}": html.escape(meta.get("title", "SC-WBD")),
            "{{description}}": html.escape(meta.get("description", "")),
            "{{heading}}": meta.get("heading", meta.get("title", "")),
            # Optional. An absent standfirst must produce no element at all --
            # an empty <p> is invisible until it is holding a paragraph's worth
            # of margin under a headline.
            "{{standfirst}}": (
                f'      <p class="standfirst">{meta["standfirst"]}</p>'
                if meta.get("standfirst") else ""
            ),
            # Which run an engineering lesson came from.  Without it a reader
            # cannot tell which findings are properties of a checkpoint we have
            # already replaced and which are properties of the design -- and by
            # 003 or 004 that is the difference between a live constraint and a
            # historical note.
            "{{applies}}": (
                f'      <p class="applies"><span>Applies to</span> {meta["applies"]}</p>'
                if meta.get("applies") else ""
            ),
            "{{nav}}": nav_html(meta.get("nav", ""), depth),
            "{{root}}": up,
            "{{body}}": body,
            
            "{{bodyclass}}": meta.get("bodyclass", ""),
        }
        for k, v in replacements.items():
            html_out = html_out.replace(k, v)

        # {{video:<name>}} -- resolved after the table so a page can place a
        # player exactly where its argument needs one.
        html_out = re.sub(
            r"\{\{video:([a-z0-9-]+)\}\}",
            lambda m: video_embed(m.group(1)),
            html_out,
        )

        # {{paper}} -- the release artifact block.  Resolved last because it
        # needs `up`, and before the leftover check because it is a key.
        html_out = html_out.replace("{{paper}}", paper_block(up))

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
