#!/usr/bin/env python3
"""E1: held-out evaluation of compact-model corrections.

Folds: ``device`` (leave one record out) or ``geometry`` (leave out every
record of one geometry, e.g. all truth sets of a pdk2 device).  For each
fold every arm (C0, T1, T2, LIN, M) is trained on the remaining records and
applied once to the held-out records; ORACLE (in-sample M per record) is a
diagnostic.  Scores are strict paper RRMS (``e1_correction.score_device``).

  PYTHONPATH=src .venv/bin/python scripts/research/explore/e1_lodo.py \
      --data out/research/explore/E1/real_dev --folds device --name real_dev
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from cryoml.explore.e1_correction import (HYPER, fit_arm, load_devices,  # noqa: E402
                                          score_device)

OUT = ROOT / "out/research/explore/E1"
ARMS = ("C0", "T1", "T2", "LIN", "M")


def aggregate(values: dict[str, float]) -> dict:
    v = np.array(list(values.values()), dtype=float)
    fam ={f: [x for t, x in values.items() if f"{f}_" in t] for f in ("nmos", "pmos")}
    finite = v[np.isfinite(v)]
    out = {"n": int(len(v)), "n_failed": int(np.sum(~np.isfinite(v))),
           "all_device_mean": float(np.mean(v)) if np.all(np.isfinite(v)) else float("nan"),
           "valid_only_mean": float(np.mean(finite)) if len(finite) else float("nan"),
           "median": float(np.median(finite)) if len(finite) else float("nan")}
    means = [float(np.mean(fam[f])) for f in ("nmos", "pmos") if fam[f]]
    out["combined"] = float(np.mean(means)) if len(means) == 2 else float("nan")
    return out


def run_fold(job):
    import torch
    torch.set_num_threads(1)
    stems, fold_kind, group, arms, hyper = job
    devices = [d for stem in stems for d in load_devices(Path(stem))]
    key = (lambda d: d.tag) if fold_kind == "device" else (lambda d: d.geometry)
    train = [d for d in devices if key(d) != group]
    held = [d for d in devices if key(d) == group]
    t0 = time.time()
    scores, corrs = {arm: {} for arm in arms}, {}
    for arm in arms:
        corr = fit_arm(arm, train, hyper=hyper.get(arm), threads=1)
        corrs[arm] = corr.to_json()
        for dev in held:
            scores[arm][dev.tag] = score_device(dev, corr.apply(dev))["rrms"]
    print(f"{group} " + " ".join(
        f"{arm}={np.mean([scores[arm][d.tag] for d in held]):.4f}" for arm in arms)
        + f" ({time.time() - t0:.0f}s)", flush=True)
    return group, scores, corrs


def wins(a: dict, b: dict) -> int:
    return int(sum(1 for t in a if np.isfinite(a[t]) and np.isfinite(b[t]) and a[t] < b[t]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True,
                    help="DeviceRec stems (npz/json pairs); concatenated")
    ap.add_argument("--folds", choices=("device", "geometry"), default="device")
    ap.add_argument("--name", required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--m-seeds", default="",
                    help="override M seeds (non-pre-registered light runs only)")
    ap.add_argument("--m-steps", type=int, default=0,
                    help="override M steps (non-pre-registered light runs only)")
    ap.add_argument("--criterion", action="store_true",
                    help="evaluate the pre-registered REAL_DEV criterion")
    ap.add_argument("--save-full", action="store_true",
                    help="also train every arm on all records (for export / refit)")
    args = ap.parse_args()
    t_start = time.time()
    devices = [d for stem in args.data for d in load_devices(Path(stem))]
    arms = args.arms.split(",")
    key = (lambda d: d.tag) if args.folds == "device" else (lambda d: d.geometry)
    groups = sorted({key(d) for d in devices})
    scores = {arm: {} for arm in arms}
    folds = {}
    hyper = {"M": {}}
    if args.m_seeds:
        hyper["M"]["seeds"] = tuple(int(x) for x in args.m_seeds.split(","))
    if args.m_steps:
        hyper["M"]["steps"] = args.m_steps
    jobs = [(args.data, args.folds, group, arms, hyper) for group in groups]
    if args.workers > 1:
        with get_context("spawn").Pool(args.workers) as pool:
            done = pool.imap_unordered(run_fold, jobs)
            results = list(done)
    else:
        results = [run_fold(job) for job in jobs]
    for group, fold_scores, fold_corr in results:
        folds[group] = fold_corr
        for arm in arms:
            scores[arm].update(fold_scores[arm])
    if args.oracle:
        scores["ORACLE"] = {}
        for dev in devices:
            corr = fit_arm("M", [dev], hyper=hyper.get("M"))
            scores["ORACLE"][dev.tag] = score_device(dev, corr.apply(dev))["rrms"]
    summary = {arm: aggregate(vals) for arm, vals in scores.items()}
    comp = {}
    if "M" in scores:
        for other in [a for a in scores if a != "M"]:
            comp[f"M_beats_{other}"] = wins(scores["M"], scores[other])
    result = {"name": args.name, "data": args.data, "folds": args.folds,
              "hyper_overrides": hyper,
              "preregistered_config": not (args.m_seeds or args.m_steps),
              "n_records": len(devices), "n_folds": len(groups), "hyper": HYPER,
              "summary": summary, "wins": comp, "per_device": scores,
              "runtime_s": round(time.time() - t_start, 1)}
    if args.criterion:
        m, t1 = summary["M"], summary["T1"]
        n = len(devices)
        checks = {
            "M_mean_le_0.090": bool(m["all_device_mean"] <= 0.090),
            "M_mean_lt_T1_mean": bool(m["all_device_mean"] < t1["all_device_mean"]),
            "M_beats_T1_on_ge_8_of_12": bool(comp["M_beats_T1"] >= 8),
            "M_beats_C0_on_ge_8_of_12": bool(comp["M_beats_C0"] >= 8),
            "M_zero_failed": bool(m["n_failed"] == 0)}
        result["criterion"] = {"n_devices": n, "checks": checks,
                               "stage1_pass": all(checks.values())}
    if args.folds == "geometry" and {"M", "C0", "T1"} <= set(summary):
        m, c0, t1 = (summary[a]["all_device_mean"] for a in ("M", "C0", "T1"))
        checks_s = {"M_mean_le_0.9_C0_mean": bool(m <= 0.9 * c0),
                    "M_mean_lt_T1_mean": bool(m < t1)}
        result["criterion_S"] = {"checks": checks_s, "pass": all(checks_s.values()),
                                 "note": "secondary; not used for the E1 decision"}
    out_dir = OUT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(result, indent=1) + "\n")
    (out_dir / "fold_corrections.json").write_text(json.dumps(folds) + "\n")
    if args.save_full:
        full = {arm: fit_arm(arm, devices, hyper=hyper.get(arm)).to_json() for arm in arms if arm != "C0"}
        (out_dir / "full_corrections.json").write_text(json.dumps(full) + "\n")
    print(json.dumps({"summary": summary, "wins": comp,
                      "criterion": result.get("criterion"),
                      "criterion_S": result.get("criterion_S")}, indent=1))
    print(f"runtime {result['runtime_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
