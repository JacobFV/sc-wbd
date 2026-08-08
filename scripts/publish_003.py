"""Derive SC-WBD-003's published claims from the checkpoint, not from the config.

The distinction this script exists to hold: a source card says what a source is
ALLOWED to train, and a config says which sources a stage ADMITS. Neither says
what happened. Run 2 shipped with cards asserting that a source trained modules
it could not reach, and every audit read the assertion.

Everything printed here is read out of the weights:

* ``extra.contributed_sources``  -- ids that produced a loss term at least once
* ``extra.moved_since_init``     -- parameters no longer bit-identical to init
* ``extra.admitted_but_no_term`` -- admitted and silent, per stage
* ``extra.tms_drive``            -- what the learned pulse became

Run: PYTHONPATH=. .venv/bin/python scripts/publish_003.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    import torch

    from scwbd.foundation.attachment_report import attachment_report, render_markdown

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", default="checkpoints/scwbd-003/last.pt")
    p.add_argument("--cards", default="configs/curriculum/source_cards")
    p.add_argument("--out", default="reports/scwbd-003_derived.json")
    a = p.parse_args(argv)

    ck_path = REPO / a.checkpoint
    if not ck_path.is_file():
        raise SystemExit(
            f"no checkpoint at {ck_path}. This script reports what a run DID; "
            "there is nothing to report before one exists."
        )
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    extra = ck.get("extra") or {}
    metrics = ck.get("metrics") or {}

    contributed = sorted(extra.get("contributed_sources") or [])

    # Cross-check against the training log, which is independent evidence: the
    # trainer writes a per-source diagnostic for every source that produced a
    # term. The launched run's checkpoints UNDER-REPORT, because
    # `_contributed.add` was only called from the run-3 loop and `eegmmidb_real`
    # and `ds002336_real` reach a loss through older call sites. Fixed in the
    # trainer, but a process does not re-read its own modules, so this run's
    # checkpoints carry the short list and the log carries the truth.
    #
    # Union, not replacement: the log is a transcript and can be shorter than
    # the run, so neither source can be trusted alone to be complete.
    log_seen: set[str] = set()
    log = REPO / "reports/training/scwbd-003_train.jsonl"
    if log.is_file():
        marker_to_source = {
            "eegmmidb_real_eeg_nll": "eegmmidb_real",
            "sleepedf_real_eeg_nll": "sleepedf_real",
            "ds000117_real_eeg_nll": "ds000117_real",
            "ds004024_rest_real_eeg_nll": "ds004024_rest_real",
            "real_bold_nll": "ds002336_real",
            "behaviour_choice_ce": "ds000117_behaviour",
            "perturb_nll": "ds004024_perturb",
            "sim_forecast_nll": "sim_wholebrain",
        }
        for line in log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            for marker, sid in marker_to_source.items():
                if marker in rec:
                    log_seen.add(sid)
    only_in_log = sorted(log_seen - set(contributed))
    contributed = sorted(set(contributed) | log_seen)
    moved = extra.get("moved_since_init") or {}
    absent = extra.get("admitted_but_no_term") or {}
    drive = extra.get("tms_drive")
    params = extra.get("parameter_report") or {}

    print(f"=== SC-WBD-003, read from {a.checkpoint} ===")
    print(f"step               {ck.get('step')}")
    print(f"stage              {ck.get('stage')}")
    print(f"completed stages   {metrics.get('completed_stages')}")
    print(f"parameters         {params.get('TOTAL')}")
    print()

    print("-- contributed gradient (derived, not asserted) --")
    for sid in contributed:
        mark = "  [from the training log, absent from the checkpoint]" if sid in only_in_log else ""
        print(f"   {sid}{mark}")
    if not contributed:
        print("   (none recorded -- this checkpoint predates the tracking)")
    if only_in_log:
        print(
            f"   NOTE: {len(only_in_log)} source(s) produced a loss term according to "
            "the training log but are absent from the checkpoint's own list. That is "
            "the under-reporting bug in `_contributed`, not a source that failed to "
            "contribute."
        )
    if absent:
        print("\n-- ADMITTED BUT PRODUCED NO TERM --")
        for stage, ids in absent.items():
            print(f"   {stage}: {ids}")
    else:
        print("\n   no stage admitted a source that produced no term")

    if moved:
        n_moved, n_tot = moved.get("n_moved"), moved.get("n_parameters")
        pct = 100.0 * n_moved / max(n_tot or 1, 1)
        print(f"\n-- moved since initialisation: {n_moved}/{n_tot} ({pct:.1f}%) --")
        for mod, e in sorted((moved.get("by_module") or {}).items()):
            flag = "  <-- FROZEN" if e["moved"] == 0 else ""
            print(f"   {mod:18s} moved {e['moved']:4d} / {e['moved']+e['frozen']:4d}{flag}")

    if drive:
        print("\n-- the learned TMS drive --")
        print(f"   computed from an E-field: {drive.get('computed_from_efield')}")
        for h in ("left", "right"):
            d = drive.get(h) or {}
            print(f"   {h:5s} gain {d.get('gain'):.4g}  peak parcel {d.get('peak_parcel')} "
                  f"(weight {d.get('peak_weight'):.3f}) over {d.get('n_support_parcels')} parcels")

    # -- attachment kinds ------------------------------------------------
    rep = attachment_report(ck_path, cards_dir=REPO / a.cards)
    (REPO / "reports/attachment_kinds.json").write_text(
        json.dumps(rep, indent=2) + "\n", encoding="utf-8"
    )
    (REPO / "reports/attachment_kinds.md").write_text(render_markdown(rep), encoding="utf-8")
    print("\n-- attachment kinds --")
    print(f"   {rep['claim']}")
    for k in ("observation", "stimulus", "boundary_output", "context"):
        v = rep["kinds"][k]
        print(f"   {k:16s} declared={v['n_channels_declared']:2d} "
              f"feeding_a_loss={v['n_channels_feeding_a_loss']:2d} "
              f"reached={v['reached_the_model']}")

    payload = {
        "checkpoint": str(a.checkpoint),
        "step": ck.get("step"),
        "completed_stages": metrics.get("completed_stages"),
        "parameters": params,
        "contributed_sources": contributed,
        "admitted_but_no_term": absent,
        "moved_since_init": moved,
        "tms_drive": drive,
        "attachment_kinds": {
            k: {
                "reached_the_model": rep["kinds"][k]["reached_the_model"],
                "n_channels_declared": rep["kinds"][k]["n_channels_declared"],
                "n_channels_feeding_a_loss": rep["kinds"][k]["n_channels_feeding_a_loss"],
            }
            for k in rep["kinds"]
        },
    }
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {a.out}, reports/attachment_kinds.json, reports/attachment_kinds.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
