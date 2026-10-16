#!/usr/bin/env python3
"""Re-score the released card, both paper series, and the cold (multiplier=1)
card with the current NGSpice binary; writes out/research/reference_scores.json."""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cryoml.card_eval import score_cards  # noqa: E402
from cryoml.cold_start import SENSITIVITY_SUMMARY, cold_centers  # noqa: E402
from cryoml.config import OUT_DIR, resolve_ngspice_bin  # noqa: E402


def paper_cards(run: str) -> dict:
    cards = {}
    for path in glob.glob(str(OUT_DIR / f"pdk_foundation_{run}" / "ml_*.json")):
        row = json.loads(Path(path).read_text())
        cards[row["device"]] = row["methods"]["surrogate_fd"]["params"]
    return cards


def main() -> int:
    summary = json.loads(SENSITIVITY_SUMMARY.read_text())
    names = tuple(summary["selected_parameters"])
    out = {"ngspice": resolve_ngspice_bin()}
    t0 = time.time()
    out["released"] = score_cards({tag: None for tag in summary["published_by_device"]})
    out["paper_seven"] = score_cards(paper_cards("vb_seven"))
    out["paper_full43"] = score_cards(paper_cards("vb_full"))
    out["cold_multiplier_one"] = score_cards(
        {tag: row["center"] for tag, row in cold_centers(names).items()})
    out["runtime_s"] = round(time.time() - t0, 1)
    path = OUT_DIR / "research" / "reference_scores.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    for key in ("released", "paper_seven", "paper_full43", "cold_multiplier_one"):
        row = out[key]
        print(f"{key:22s} all={row['all_device_mean_rrms']:.5f} "
              f"n={row['nmos_rrms']:.4f} p={row['pmos_rrms']:.4f}")
    print("runtime", out["runtime_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
