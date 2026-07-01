#!/usr/bin/env python3
"""Reconcile the paper gap: every method scored both all-curve and with a
device-off floor (near-off curves -> 0), from each method's saved winning sims.

The all-curve RRMS is dominated by near-off curves that are ~41% hard zeros
plus SMU glitch codes up to ~51 uA (physically unfittable by any card). The
floored metric zeroes curves whose mean current is below `frac` of the
device's peak, identically for every method, and lands near the paper's 0.279."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cryoml.config import OUT_DIR, OUT_TABLES  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402

METHODS = [
    ("published baseline", "pdk_baseline"),
    ("fd control", "pdk_fd"),
    ("cma-8500 control", "pdk_cma"),
    ("ml_final (search)", "pdk_ml_final_perdev"),
    ("tandem seed0", "pdk_mlp_tandem_full"),
    ("forward surr one-shot", "pdk_fwd_surr"),
]
FRACS = [0.0, 0.01, 0.02, 0.05]


def device_rrms_floored(curves, sims, frac):
    rmse, mabs = [], []
    for c, s in zip(curves, sims):
        meas = np.asarray(c.Id, float)
        s = np.asarray(s, float)
        n = min(len(meas), len(s))
        meas, s = meas[:n], s[:n]
        if not np.all(np.isfinite(s)):
            return np.inf
        rmse.append(float(np.sqrt(np.mean((s - meas) ** 2))))
        mabs.append(float(np.mean(np.abs(meas))))
    rmse, mabs = np.array(rmse), np.array(mabs)
    peak = mabs.max() if len(mabs) else 0.0
    scores = []
    for r, m in zip(rmse, mabs):
        if m <= 0 or (frac > 0 and m < frac * peak):
            scores.append(0.0)
        else:
            scores.append(r / m)
    return float(np.mean(scores))


def main() -> int:
    table = {}
    for label, d in METHODS:
        per_frac = {f: [] for f in FRACS}
        for dev in PAPER_DEVICES:
            tag = device_tag(dev.dev_type, dev.L_um, dev.W_um)
            f = OUT_DIR / d / f"sims_{tag}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            curves = load_device_curves(dev)
            sims = [z[f"sim_{i}"] for i in range(len(curves))]
            for fr in FRACS:
                per_frac[fr].append(device_rrms_floored(curves, sims, fr))
        table[label] = per_frac

    print(f"{'method':24s} | " + " | ".join(
        f"f={fr}" for fr in FRACS) + "   (device-mean RRMS)")
    print("-" * 78)
    out = {}
    for label, _ in METHODS:
        per = table[label]
        means = {fr: float(np.mean(per[fr])) if per[fr] else float("nan")
                 for fr in FRACS}
        out[label] = means
        print(f"{label:24s} | " + " | ".join(
            f"{means[fr]:6.3f}" for fr in FRACS))
    print("-" * 78)
    print("paper reported (HSPICE/Mystic):  0.279")
    print("\nfloored metric zeroes curves with mean|I| < frac * device-peak,")
    print("applied identically to every method; reported ALONGSIDE all-curve (f=0).")
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    json.dump({"fracs": FRACS, "means": out, "paper": 0.279},
              open(OUT_TABLES / "floored_comparison.json", "w"), indent=2)
    print(f"\nwrote {OUT_TABLES / 'floored_comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
