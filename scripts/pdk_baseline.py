#!/usr/bin/env python3
"""Published-card baseline through the confirmed-setup NGSpice chain.

Scores every Table-6 device with the confirmed-setup metric (rrmsCalc port,
primary) and the legacy all-curve RRMS (continuity), native bin selection,
updated pFET card, ngspice-41.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.config import OUT_DIR, ensure_dirs  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.metrics import (device_rrms, family_totals,  # noqa: E402
                            score_device_new)
from cryoml.spice_pdk import find_bin_index, simulate_pdk  # noqa: E402
from cryoml.utils import device_tag, get_logger  # noqa: E402

logger = get_logger("pdk_baseline")
OUT = OUT_DIR / "pdk_baseline"


def main() -> int:
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    new_by_tag = {}
    for device in PAPER_DEVICES:
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        curves = load_device_curves(device)
        sims = simulate_pdk(device.dev_type, device.L_um, device.W_um, curves)
        new = score_device_new(device.dev_type, device.L_um, device.W_um,
                               curves, sims)
        legacy = device_rrms(sims, [c.Id for c in curves])
        bin_index = find_bin_index(device.dev_type, device.L_um, device.W_um)
        new_by_tag[tag] = new
        rec = {
            "device": tag,
            "dev_type": device.dev_type,
            "L_um": device.L_um,
            "W_um": device.W_um,
            "bin_index": bin_index,
            "paper_reported": device.paper_rrms,
            "paper_reported_sigma": device.paper_sigma,
            "rrms": new["rrms"],
            "sigma": new["sigma"],
            "n_curves": new["n_curves"],
            "legacy_rrms": legacy["rrms"],
            "legacy_sigma": legacy["sigma"],
        }
        rows.append(rec)
        np.savez(OUT / f"sims_{tag}.npz",
                 **{f"sim_{i}": np.asarray(s) for i, s in enumerate(sims)})
        logger.info("%-22s bin=%-2s rrms=%.4f (paper %.3f)  legacy=%.4f",
                    tag, bin_index, new["rrms"], device.paper_rrms,
                    legacy["rrms"])

    totals = family_totals(new_by_tag)
    paper_nmos = float(np.mean([r["paper_reported"] for r in rows
                                if r["dev_type"] == "nmos"]))
    paper_pmos = float(np.mean([r["paper_reported"] for r in rows
                                if r["dev_type"] == "pmos"]))
    summary = {
        "n_devices": len(rows),
        "metric": "confirmed-setup rrmsCalc port",
        "all_device_mean_rrms": float(np.mean([r["rrms"] for r in rows])),
        "paper_reported_mean": float(np.mean([r["paper_reported"] for r in rows])),
        "paper_reported_nmos_rrms": paper_nmos,
        "paper_reported_pmos_rrms": paper_pmos,
        "paper_reported_combined_rrms": (paper_nmos + paper_pmos) / 2.0,
        **totals,
        "legacy_all_curve_mean": float(np.mean([r["legacy_rrms"] for r in rows])),
    }
    json.dump({"devices": rows, "summary": summary,
               "per_curve": {t: new_by_tag[t]["per_curve"] for t in new_by_tag}},
              open(OUT / "pdk_baseline.json", "w"), indent=2)
    with open(OUT / "pdk_baseline.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
