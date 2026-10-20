#!/usr/bin/env python3
"""Compare two ngspice binaries on the published 77 K card.

For each of the 18 Table-6 devices, simulate the 11 metric curves on the
upstream sweep grids (``scripts/verify_simulator.py``) with
``cryoml.spice_pdk.simulate_pdk`` (native bin selection included) once per
binary, and report the max absolute and max relative current difference
and the number of bit-identical points.

  PYTHONPATH=src .venv/bin/python scripts/research/compare_ngspice_bins.py \
      --a /Users/anrunw/cryo-ng41/arm64/bin/ngspice \
      --b /Users/anrunw/cryo-ng41/arm64-osdi/bin/ngspice \
      --out out/tables/simulator_compare_ng41_arm64_vs_arm64_osdi.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cryoml.spice_pdk as sp  # noqa: E402
import verify_simulator as vs  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="reference ngspice binary")
    ap.add_argument("--b", required=True, help="compared ngspice binary")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    per, worst_abs, worst_rel, n_pts, n_same = {}, 0.0, 0.0, 0, 0
    for dev in PAPER_DEVICES:
        curves, _, _ = vs.upstream_metric_curves(dev)
        sims = {}
        for key, binary in (("a", args.a), ("b", args.b)):
            sp._NGSPICE_BIN = binary
            sims[key] = sp.simulate_pdk(dev.dev_type, dev.L_um, dev.W_um, curves)
        d_abs = d_rel = 0.0
        for x, y in zip(sims["a"], sims["b"]):
            x, y = np.asarray(x, float), np.asarray(y, float)
            n_pts += x.size
            n_same += int(np.sum(x == y))
            d = np.abs(x - y)
            d_abs = max(d_abs, float(d.max()))
            d_rel = max(d_rel, float(np.max(d / np.maximum(np.abs(x), 1e-30))))
        tag = f"{dev.dev_type}_L{dev.L_um:g}_W{dev.W_um:g}"
        per[tag] = {"max_abs_A": d_abs, "max_rel": d_rel}
        worst_abs, worst_rel = max(worst_abs, d_abs), max(worst_rel, d_rel)
        print(f"{tag:22s} max|dI| {d_abs:.3e} A  max rel {d_rel:.3e}", flush=True)
    summary = {"binary_a": args.a, "binary_b": args.b, "n_devices": len(per),
               "n_points": n_pts, "n_bit_identical_points": n_same,
               "max_abs_diff_A": worst_abs, "max_rel_diff": worst_rel, "per_device": per}
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_device"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
