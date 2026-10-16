#!/usr/bin/env python3
"""Independently re-score cold-loop and direct-NGSpice cards with card_eval and
summarize how far each card moved from the stock and released values.

  PYTHONPATH=src .venv/bin/python scripts/research/verify_cold_results.py \
      --set seven --round 4 --direct seven_r1_direct
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cryoml.card_eval import score_cards  # noqa: E402
from cryoml.cold_start import SENSITIVITY_SUMMARY, cold_centers  # noqa: E402
from cryoml.config import OUT_DIR  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="seven")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--direct", default=None)
    args = parser.parse_args()
    summary = json.loads(SENSITIVITY_SUMMARY.read_text())
    names = tuple(summary["selected_parameters"])
    stock = {t: r["center"] for t, r in cold_centers(names).items()}
    released = summary["published_by_device"]
    extract = OUT_DIR / "cold_loop" / args.set / f"round_{args.round}" / "extract"
    sets, kinds = {}, collections.Counter()
    cards = {}
    for path in sorted(glob.glob(str(extract / "ml_*.json"))):
        row = json.loads(Path(path).read_text())
        cards[row["device"]] = row["methods"]["surrogate_fd"]["params"]
        kinds[row["methods"]["surrogate_fd"]["selected_candidate_kind"]] += 1
    sets[f"cold_{args.set}_r{args.round}"] = cards
    if args.direct:
        direct = json.loads((OUT_DIR / "research" / "ngspice_direct" / args.direct
                             / "summary.json").read_text())
        sets[args.direct] = {t: r["params"] for t, r in direct["devices"].items()}
    out = {"selected_candidate_kinds": dict(kinds)}
    for label, cards in sets.items():
        scored = score_cards(cards)
        moves = collections.defaultdict(list)
        for tag, params in cards.items():
            for name, value in params.items():
                for ref_label, ref in (("vs_stock", stock), ("vs_released", released)):
                    base = ref[tag].get(name)
                    if base and value and np.sign(base) == np.sign(value):
                        moves[f"{name}:{ref_label}"].append(float(value / base))
        out[label] = {
            k: scored[k] for k in ("all_device_mean_rrms", "nmos_rrms",
                                   "pmos_rrms", "combined_rrms")}
        out[label]["ratio_ranges"] = {
            k: [float(np.min(v)), float(np.median(v)), float(np.max(v))]
            for k, v in sorted(moves.items())}
    path = OUT_DIR / "research" / f"verify_{args.set}_r{args.round}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
