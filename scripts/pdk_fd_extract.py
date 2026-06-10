#!/usr/bin/env python3
"""Multistart finite-difference extraction on the paper-exact NGSpice metric."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.config import OUT_DIR, ensure_dirs  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import parse_device_list  # noqa: E402
from cryoml.utils import device_tag, get_logger  # noqa: E402

logger = get_logger("pdk_fd_extract")
OUT = OUT_DIR / "pdk_fd"


def geometry_bins() -> dict[str, int]:
    baseline = json.load(open(OUT_DIR / "pdk_baseline" / "pdk_baseline.json"))
    return {row["device"]: int(row["bin_index"]) for row in baseline["devices"]}


def run_one(job):
    device, bin_index, n_starts, sigma_z, max_nfev, seed = job
    from cryoml.pdk_extract import fd_extract_device
    curves = load_device_curves(device)
    result = fd_extract_device(
        device.dev_type, device.L_um, device.W_um, bin_index, curves,
        n_starts=n_starts, sigma_z=sigma_z, max_nfev=max_nfev, seed=seed)
    tag = device_tag(device.dev_type, device.L_um, device.W_um)
    rec = {
        "device": tag,
        "dev_type": device.dev_type,
        "L_um": device.L_um,
        "W_um": device.W_um,
        "bin_index": bin_index,
        "paper_reported": device.paper_rrms,
        "start_rrms": result.start_rrms,
        "rrms": result.rrms,
        "n_starts": result.n_starts,
        "runtime_s": round(result.runtime_seconds, 1),
        "params": result.params,
    }
    return tag, rec, {f"sim_{i}": np.asarray(s) for i, s in enumerate(result.sims)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-starts", type=int, default=6)
    parser.add_argument("--sigma-z", type=float, default=1.0)
    parser.add_argument("--max-nfev", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    bins = geometry_bins()
    jobs = []
    for device in parse_device_list(args.devices):
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        jobs.append((device, bins[tag], args.n_starts, args.sigma_z,
                     args.max_nfev, args.seed))

    rows = []
    with get_context("spawn").Pool(processes=args.workers) as pool:
        for tag, rec, sims in pool.imap_unordered(run_one, jobs):
            rows.append(rec)
            json.dump(rec, open(OUT / f"fd_{tag}.json", "w"), indent=2)
            np.savez(OUT / f"sims_{tag}.npz", **sims)
            logger.info("%-22s %.3f -> %.3f", tag, rec["start_rrms"], rec["rrms"])

    rows.sort(key=lambda row: row["device"])
    summary = {
        "n_devices": len(rows),
        "mean_rrms": float(np.mean([r["rrms"] for r in rows])),
        "baseline_mean_rrms": float(np.mean([r["start_rrms"] for r in rows])),
        "wins_vs_baseline": sum(r["rrms"] < r["start_rrms"] for r in rows),
    }
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    with open(OUT / "results.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "device", "bin_index", "paper_reported", "start_rrms", "rrms"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
