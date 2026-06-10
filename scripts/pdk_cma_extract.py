#!/usr/bin/env python3
"""CMA-ES extraction directly on the paper-exact NGSpice chain.

Derivative-free global search in z-space with the paper-exact RRMS-aligned
cost, warm-started at the published geometry-bin parameters, followed by an
FD polish of the best point.

  python scripts/pdk_cma_extract.py --workers 8 --evals 2400
"""
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

logger = get_logger("pdk_cma")

OUT_CMA = OUT_DIR / "pdk_cma"


def run_one(job):
    import time

    import cma
    from scipy.optimize import least_squares

    from cryoml.pdk_extract import (eval_params, flatten_paper_curves,
                                    residual_fn, theta_box_for)

    (d, bin_index), evals, sigma0, max_nfev_polish, seed = job
    t0 = time.time()
    tag = device_tag(d.dev_type, d.L_um, d.W_um)
    curves = load_device_curves(d)
    box = theta_box_for(d.dev_type, d.L_um, d.W_um, bin_index)
    flat = flatten_paper_curves(curves)

    def cost(z):
        r = residual_fn(np.asarray(z, dtype=np.float64), box, d.dev_type,
                        d.L_um, d.W_um, bin_index, curves, flat)
        return float(np.sum(r * r))

    start_m, _ = eval_params(d.dev_type, d.L_um, d.W_um, bin_index, curves,
                             box.published)

    es = cma.CMAEvolutionStrategy(
        box.z_published, sigma0,
        {"maxfevals": evals, "seed": seed + 1, "verbose": -9, "popsize": 12,
         "bounds": [-6.0, 6.0]})
    es.optimize(cost)
    z_best = np.asarray(es.result.xbest, dtype=np.float64)

    # FD polish
    f = lambda z: residual_fn(z, box, d.dev_type, d.L_um, d.W_um, bin_index,
                              curves, flat)
    try:
        sol = least_squares(f, z_best, method="trf", jac="2-point",
                            diff_step=2e-2, max_nfev=max_nfev_polish)
        z_pol = np.asarray(sol.x, dtype=np.float64)
    except Exception:
        z_pol = z_best

    best = {"params": box.published, "m": start_m, "which": "published"}
    for which, z in (("cma", z_best), ("cma+fd", z_pol)):
        params = box.z_to_params(z)
        m, _ = eval_params(d.dev_type, d.L_um, d.W_um, bin_index, curves, params)
        if np.isfinite(m["rrms"]) and m["rrms"] < best["m"]["rrms"]:
            best = {"params": params, "m": m, "which": which}

    final_m, sims = eval_params(d.dev_type, d.L_um, d.W_um, bin_index, curves,
                                best["params"])
    rec = {
        "device": tag, "dev_type": d.dev_type, "L_um": d.L_um, "W_um": d.W_um,
        "bin_index": bin_index, "paper_reported": d.paper_rrms,
        "rrms": float(final_m["rrms"]),
        "start_rrms": float(start_m["rrms"]),
        "method": "cma_es+" + best["which"], "evals": evals,
        "runtime_s": round(time.time() - t0, 1), "params": best["params"],
    }
    return tag, rec, {f"sim_{i}": np.asarray(s) for i, s in enumerate(sims)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--evals", type=int, default=2400)
    ap.add_argument("--sigma0", type=float, default=0.8)
    ap.add_argument("--max-nfev-polish", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ensure_dirs()
    OUT_CMA.mkdir(parents=True, exist_ok=True)
    bins = {r["device"]: int(r["bin_index"]) for r in json.load(
        open(OUT_DIR / "pdk_baseline" / "pdk_baseline.json"))["devices"]}
    devices = parse_device_list(args.devices)
    jobs = [((d, bins[device_tag(d.dev_type, d.L_um, d.W_um)]), args.evals,
             args.sigma0, args.max_nfev_polish, args.seed) for d in devices]

    rows = []
    ctx = get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        for tag, rec, sims in pool.imap_unordered(run_one, jobs):
            rows.append(rec)
            json.dump(rec, open(OUT_CMA / f"fd_{tag}.json", "w"), indent=2)
            np.savez(OUT_CMA / f"sims_{tag}.npz", **sims)
            logger.info("%-22s %.3f->%.3f [%s] %ss",
                        tag, rec["start_rrms"], rec["rrms"], rec["method"],
                        rec["runtime_s"])

    scores = np.array([r["rrms"] for r in rows])
    starts = np.array([r["start_rrms"] for r in rows])
    summary = {
        "n_devices": len(rows),
        "mean_rrms": float(np.nanmean(scores)),
        "baseline_mean_rrms": float(np.nanmean(starts)),
        "wins_vs_baseline": int(np.sum(scores < starts)),
    }
    json.dump(summary, open(OUT_CMA / "summary.json", "w"), indent=2)
    with open(OUT_CMA / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["device", "bin", "paper_reported", "start_rrms", "rrms"])
        for r in sorted(rows, key=lambda r: r["device"]):
            w.writerow([r["device"], r["bin_index"], r["paper_reported"],
                        f"{r['start_rrms']:.4f}", f"{r['rrms']:.4f}"])
    print("\n=== PDK CMA-ES ===")
    for k, v in summary.items():
        print(f"  {k:24s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
