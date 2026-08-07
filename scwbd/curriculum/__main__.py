"""``python -m scwbd.curriculum`` --- validate a run config against the ordering."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="scwbd.curriculum", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="refuse a curriculum whose integrity ordering is inverted")
    v.add_argument("config")
    v.add_argument("--tiers", default="configs/curriculum/tiers.yaml")
    v.add_argument("--cards", default=None, help="override the mixture-card directory")
    v.add_argument("--json", dest="json_out", default=None, help="write the full verdict here")

    t = sub.add_parser("tiers", help="print the tier of every card in a mixture directory")
    t.add_argument("--cards", default="configs/source_cards")

    i = sub.add_parser("information", help="print the measured per-modality blindness rules")
    i.add_argument("--results", default="reports/identifiability/results.json")
    i.add_argument("--manifest", default="reports/identifiability/manifest.json")

    a = p.parse_args(argv)

    if a.cmd == "tiers":
        from .tiers import load_mixture_cards, tier_table

        for sid, asg in tier_table(load_mixture_cards(a.cards)).items():
            flag = f"REFUSED {asg.refusal}" if asg.refusal else f"tier {asg.tier}"
            print(f"{sid:28s} {flag:34s} {asg.reason}")
        return 0

    if a.cmd == "information":
        from .information import derive_blind_rules, load_modality_information

        info = load_modality_information(a.results, a.manifest)
        print(json.dumps(info.provenance, indent=2))
        for r in derive_blind_rules(info):
            print(json.dumps(r.as_dict()))
        return 0

    from .validate import validate_config

    verdict = validate_config(a.config, tiers_path=a.tiers, cards_dir=a.cards)
    print(verdict.report())
    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(json.dumps(verdict.as_dict(), indent=2, default=str))
        print(f"\nwrote {a.json_out}")
    # Three exit codes, matching the three verdict states. `verdict.ok` now
    # requires that every check actually ran, so an INCONCLUSIVE config no
    # longer exits 0 -- but collapsing it onto 1 would tell a caller a refusal
    # fired when none did, and the difference between "this config is wrong" and
    # "I could not tell" is the whole reason the third state exists.
    if verdict.ok:
        return 0
    if verdict.inconclusive:
        print(
            "\nINCONCLUSIVE: nothing refused, but these checks could not run: "
            + ", ".join(verdict.unevaluable_checks())
        )
        return 2
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
