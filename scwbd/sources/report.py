"""Generate ``reports/data_inventory.md`` from the registry and the cards.

The inventory is regenerated, never hand-edited: every number in it is read
from the bytes on disk or from a source card that was itself refreshed from
those bytes.

::

    python -m scwbd.sources.report
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .cards import load_card
from .registry import REGISTRY, DatasetEntry, data_root

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "reports" / "data_inventory.md"


def _gb(n: int) -> str:
    return f"{n / 1e9:.2f}" if n else "0"


def _esc(s: object) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ").strip()


def _short(s: object, n: int = 160) -> str:
    t = " ".join(str(s).split())
    return t if len(t) <= n else t[: n - 1] + "…"


def build() -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines: list[str] = []
    A = lines.append

    A("# Data inventory — SC-WBD-001-beta (build-order item 5)")
    A("")
    A(f"*Generated {now} by `python -m scwbd.sources.report`. Do not hand-edit.*")
    A("")
    A(f"Data root: `{data_root()}` (symlink to `/data/scwbd`).")
    A("")
    A(
        "Every row below is derived from bytes on disk plus the dataset's source card "
        "(`scwbd/sources/cards/<id>.yaml`). A dataset with `status: unavailable` was "
        "deliberately **not** downloaded; the reason is a licence or credential barrier, "
        "not an omission. Per Appendix B a field that cannot be populated stays `unknown` "
        "and the gradient path that depends on it is disabled — those disabled paths are "
        "listed explicitly, because a silently missing capability is worse than a "
        "declared one."
    )
    A("")
    A("> **Open defect affecting how these rows are consumed — see "
      "[`reports/known_issues.md`](known_issues.md), ISSUE-001.** "
      "`scwbd.schema.UncertaintyLedger.bias_interval` is a required `tuple[float, float]` "
      "with no representation for *unknown*, so a card that honestly says "
      "`bias_interval: unknown` is projected onto the typed schema as `[0, 0]` — read "
      "literally, a claim that the source has exactly zero systematic error. Nothing is "
      "wrong in the rows below (every affected card is `unavailable` and is therefore "
      "never projected), and `has_estimator()` currently rejects the degenerate interval "
      "so refusal `R08` still fires. But that is an accident of encoding, not a "
      "guarantee. Any consumer reading `bias_interval` without also consulting "
      "`has_estimator()` will read an unknown as a confident zero. The fix is a schema "
      "change (an explicit unknown), sequenced by the coordinator.")
    A("")

    rows: list[tuple[DatasetEntry, dict]] = []
    for entry in REGISTRY.values():
        try:
            card = load_card(entry.card_path)
        except Exception as exc:  # a broken card is itself a finding
            rows.append((entry, {"error": str(exc)}))
            continue
        rows.append((entry, {"card": card}))

    live = [(e, d) for e, d in rows if "card" in d and d["card"].status != "unavailable"]
    dead = [(e, d) for e, d in rows if "card" in d and d["card"].status == "unavailable"]
    broken = [(e, d) for e, d in rows if "error" in d]

    total = sum(e.on_disk_bytes() for e, _ in live)
    A(f"**{len(live)} dataset(s) live on disk, {_gb(total)} GB total. "
      f"{len(dead)} registered but unavailable.**")
    A("")

    # ---- headline table -------------------------------------------------
    A("## 1. What is on disk")
    A("")
    A("| dataset | version | status | GB on disk | files | participants | modalities | role | licence |")
    A("|---|---|---|---:|---:|---|---|---|---|")
    for entry, d in live:
        card = d["card"]
        fm = card.data["identity"].get("file_manifest") or {}
        pop = card.data["population"]
        A(
            f"| `{entry.dataset_id}` | {entry.version} | {card.status} | "
            f"{_gb(entry.on_disk_bytes())} | {fm.get('n_files', '?')} | "
            f"{_esc(_short(pop.get('n_participants'), 40))} | "
            f"{'/'.join(entry.modalities)} | {card.role} | "
            f"{_esc(_short(card.data['governance']['license'], 60))} |"
        )
    A("")

    # ---- gradient permissions ------------------------------------------
    A("## 2. What each source may update, and what it may not")
    A("")
    A(
        "`may update` is the compiled gradient mask `A_k`. `may NOT update` lists the "
        "paths the card declares but that are **disabled**, together with the unresolved "
        "field that disables them. `forbidden` is a hard prohibition independent of any "
        "unknown."
    )
    A("")
    for entry, d in live:
        card = d["card"]
        perm = card.effective_gradient_permission()
        A(f"### `{entry.dataset_id}` — role `{card.role}`")
        A("")
        A(f"- **subset fetched:** {_esc(entry.subset)}")
        A(f"- **may update:** {', '.join(f'`{t}`' for t in perm['enabled']) or '_nothing_'}")
        if perm["disabled"]:
            A("- **may NOT update (declared but disabled):**")
            for target, reason in perm["disabled"].items():
                A(f"  - `{target}` — {_esc(reason)}")
        else:
            A("- **may NOT update:** _no declared path is disabled_")
        if perm["frozen"]:
            A(f"- **frozen (read-only for this source):** {', '.join(f'`{t}`' for t in perm['frozen'])}")
        if perm["forbid"]:
            A(f"- **forbidden outright:** {', '.join(f'`{t}`' for t in perm['forbid'])}")
        A(f"- **unresolved fields:** {len(card.unknown_fields())} "
          f"({', '.join('`' + f + '`' for f in card.unknown_fields()[:8])}"
          f"{', …' if len(card.unknown_fields()) > 8 else ''})")
        A(f"- **split policy:** group by "
          f"{', '.join(card.data['split_policy']['grouping_keys'])}; barrier "
          f"`{card.data['split_policy']['leakage_barrier']}`")
        A(f"- **card content hash:** `{card.content_hash()[:16]}…`")
        fm = card.data["identity"].get("file_manifest") or {}
        if fm:
            A(f"- **manifest sha256:** `{str(fm.get('manifest_sha256', ''))[:16]}…` "
              f"over {fm.get('n_files')} files / {_gb(int(fm.get('total_bytes', 0)))} GB")
        A("")

    # ---- native support -------------------------------------------------
    A("## 3. Native support (never resampled)")
    A("")
    A("| dataset | native rate(s) | units | spatial support | frame | clock |")
    A("|---|---|---|---|---|---|")
    for entry, d in live:
        card = d["card"]
        sig, tem, spa = card.data.get("signal") or {}, card.data["temporal"], card.data["spatial"]
        rates = sig.get("native_rates_hz") or [tem.get("sfreq_hz")]
        A(
            f"| `{entry.dataset_id}` | {_esc(rates)} Hz | {_esc(_short(sig.get('units'), 70))} | "
            f"{spa['kind']} × {spa.get('n_elements')} | `{spa['frame']}` | `{tem['clock']}` |"
        )
    A("")

    # ---- unavailable ----------------------------------------------------
    A("## 4. Registered but NOT downloaded (honest gaps)")
    A("")
    A(
        "These are candidates from Appendix A that a credential or agreement barrier "
        "excludes. They are in the register so the absence is auditable. None of them "
        "has a downloader wired in; `scwbd.sources.download.UnavailableFetcher` refuses "
        "and returns the reason."
    )
    A("")
    A("| dataset | would-be role | modalities | barrier |")
    A("|---|---|---|---|")
    for entry, d in dead:
        card = d["card"]
        A(
            f"| `{entry.dataset_id}` | {card.role} | {'/'.join(entry.modalities)} | "
            f"{_esc(_short(card.data['governance']['unavailable_reason'], 320))} |"
        )
    A("")

    if broken:
        A("## 5. Cards that do not validate")
        A("")
        for entry, d in broken:
            A(f"- `{entry.dataset_id}`: {_esc(d['error'])}")
        A("")

    # ---- leakage protocol ----------------------------------------------
    A("## 5. Leakage protocol in force" if not broken else "## 6. Leakage protocol in force")
    A("")
    A(
        "Splitting goes through `scwbd.sources.splits.GroupedSplitter`, which resolves "
        "lineage (`family > participant > site > device > session > run > trial`) and "
        "derivation roots **before** assigning folds, and raises `LineageError` (refusal "
        "`R10`) when parentage is unresolved. `scwbd.sources.splits.leakage_audit` then "
        "re-derives the grouping and checks:"
    )
    A("")
    A("1. a participant/family never appears in two test folds or on both sides of one fold;")
    A("2. no record id appears in more than one test fold;")
    A("3. identical `content_hash` values (duplicate archive records) do not cross a fold;")
    A("4. derived records stay with their derivation root (a tractogram is not a subject);")
    A("5. held-out stimuli do not reappear in training records;")
    A("6. residual site predictability of fold membership (normalised mutual information), "
      "warned above 0.20 outside leave-site-out mode.")
    A("")
    A(
        "**What this data substrate cannot support.** Every live source here is "
        "single-site, so *no* leave-site-out evaluation is possible within any one of "
        "them; the site/device shortcut control of Appendix D can only be run across "
        "sources. `eegmmidb` has one session per participant and no demographics, so it "
        "cannot support an individualisation (G5) claim. `ds000117` has two participants "
        "on disk, so it cannot support any population claim. These are stated here so "
        "that a downstream claim report cannot quietly assume otherwise."
    )
    A("")
    return "\n".join(lines) + "\n"


def write(path: Path = OUT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build())
    return path


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scwbd.sources.report")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    p = write(Path(args.out))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
