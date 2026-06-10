#!/usr/bin/env python3
"""Paper-parameter baseline through the corrected-repository NGSpice chain."""
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
from cryoml.metrics import device_rrms  # noqa: E402
from cryoml.spice_pdk import find_bin_index, simulate_pdk  # noqa: E402
from cryoml.utils import device_tag, get_logger  # noqa: E402

logger = get_logger("pdk_baseline")
OUT = OUT_DIR / "pdk_baseline"


def main() -> int:
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for device in PAPER_DEVICES:
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        curves = load_device_curves(device)
        sims = simulate_pdk(device.dev_type, device.L_um, device.W_um, curves)
        score = device_rrms(sims, [c.Id for c in curves])
        bin_index = find_bin_index(device.dev_type, device.L_um, device.W_um)
        rec = {
            "device": tag,
            "dev_type": device.dev_type,
            "L_um": device.L_um,
            "W_um": device.W_um,
            "bin_index": bin_index,
            "paper_reported": device.paper_rrms,
            "rrms": score["rrms"],
            "sigma": score["sigma"],
        }
        rows.append(rec)
        np.savez(OUT / f"sims_{tag}.npz",
                 **{f"sim_{i}": np.asarray(s) for i, s in enumerate(sims)})
        logger.info("%-22s bin=%-2s rrms=%.3f reported=%.3f",
                    tag, bin_index, score["rrms"], device.paper_rrms)

    summary = {
        "n_devices": len(rows),
        "paper_reported_mean": float(np.mean([r["paper_reported"] for r in rows])),
        "ngspice_paper_params_mean": float(np.mean([r["rrms"] for r in rows])),
    }
    json.dump({"devices": rows, "summary": summary},
              open(OUT / "pdk_baseline.json", "w"), indent=2)
    with open(OUT / "pdk_baseline.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
