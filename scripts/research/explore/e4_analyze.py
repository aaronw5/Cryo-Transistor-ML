#!/usr/bin/env python3
"""E4 analysis: held-out RRMS, coverage, identifiability, decision.

Reads out/research/explore/E4/<run>/<split>/*.json|npz (REAL_DEV only) and
writes summary.json next to them.  With --decide, applies the pre-registered
criterion in out/research/explore/E4/prereg.json and records the decision.

  PYTHONPATH=src .venv/bin/python scripts/research/explore/e4_analyze.py \
      --run p43 --split vd --decide
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from cryoml.explore.e4_eval import (LEVELS, device_metrics, device_predictions,  # noqa: E402
                                    family_means, learned_and_conformal,
                                    load_device)
from cryoml.explore.e4_heldout import E4_DIR, real_split  # noqa: E402

POINT_INTERVAL = {"ens": "ens", "laplace": "laplace", "learned": "learned"}
METHODS = ("single", "ens", "ens_env", "laplace", "laplace_param", "conformal",
           "crossinterp", "crossinterp_conformal", "learned")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="vd")
    p.add_argument("--decide", action="store_true")
    a = p.parse_args()
    run_dir = E4_DIR / a.run / a.split
    opts = json.loads((run_dir / "options.json").read_text())
    dev_set = set(real_split()["REAL_DEV"])
    tags = sorted(t.stem for t in run_dir.glob("*.npz"))
    assert all(t in dev_set for t in tags), "non-DEV device in run"
    devs = [load_device(run_dir, t, opts["near_tol"]) for t in tags]
    base = {d["tag"]: device_predictions(d) for d in devs}
    learned = learned_and_conformal(devs, base)
    per_device = {m: {} for m in METHODS}
    for d in devs:
        for m in METHODS:
            src = base[d["tag"]].get(m) if m in base[d["tag"]] else \
                learned[m].get(d["tag"])
            if src is None:
                continue
            per_device[m][d["tag"]] = device_metrics(d, src)
    summary = {"run": a.run, "split": a.split, "options": opts,
               "n_devices": len(devs), "devices": tags, "methods": {}}
    single = {t: r["heldout_rrms"] for t, r in per_device["single"].items()}
    for m in METHODS:
        rows = per_device[m]
        if not rows:
            continue
        rr = {t: r["heldout_rrms"] for t, r in rows.items()}
        entry = {"heldout_rrms": family_means(rr), "per_device": rows}
        if m != "single":
            common = [t for t in rr if np.isfinite(rr[t]) and np.isfinite(single.get(t, np.nan))]
            entry["wins_vs_single"] = int(sum(rr[t] < single[t] for t in common))
            entry["n_paired"] = len(common)
            entry["rel_change_vs_single"] = (
                float(np.mean([rr[t] for t in common]) / np.mean([single[t] for t in common]) - 1)
                if common else math.nan)
        if any(f"coverage_{PRIMARY:g}" in r for r in rows.values()):
            for lev in LEVELS:
                entry[f"coverage_{lev:g}"] = family_means(
                    {t: r.get(f"coverage_{lev:g}", math.nan) for t, r in rows.items()})
                entry[f"width_{lev:g}"] = family_means(
                    {t: r.get(f"width_{lev:g}", math.nan) for t, r in rows.items()})
            entry["interval_score_0.9"] = family_means(
                {t: r.get("interval_score_0.9", math.nan) for t, r in rows.items()})
        summary["methods"][m] = entry
    summary["learned_hyper"] = learned["learned_hyper"]
    # Fit quality and identifiability.
    ident = {}
    for d in devs:
        rec = d["rec"]
        s = rec["members"][rec["single_index"]]
        i = rec["identifiability"]
        ident[d["tag"]] = {
            "train_obj": s["train_obj"], "train_sigma": s["train_sigma"],
            "n_members": len(rec["members"]), "near_optimal_count": i.get("near_optimal_count"),
            "near_optimal_n_spread_gt_0p1": i.get("near_optimal_n_spread_gt_0p1"),
            "eff_rank_1e-2": i.get("eff_rank_1e-2"), "eff_rank_1e-3": i.get("eff_rank_1e-3"),
            "condition": i.get("condition"), "n_active": len(rec["active_names"]),
            "heldout_sens_in_sloppy_frac": i.get("heldout_sens_in_sloppy_frac"),
            "laplace_n_sd_gt_0p1": i.get("laplace_n_sd_gt_0p1"),
            "member_heldout_rrms": [float(np.mean(m["heldout_curve_rrms"])) for m in rec["members"]],
            "member_train_obj": [m["train_obj"] for m in rec["members"]],
            "n_sim": rec["n_sim"], "runtime_s": rec["runtime_s"]}
    summary["identifiability"] = ident
    summary["train_obj_mean"] = float(np.mean([v["train_obj"] for v in ident.values()]))
    if a.decide:
        summary["decision"] = decide(summary)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print_table(summary)
    return 0


PRIMARY = 0.9


def decide(summary: dict) -> dict:
    prereg = json.loads((E4_DIR / "prereg.json").read_text())
    crit = prereg["criterion"]
    meth = summary["methods"]
    single_rrms = meth["single"]["heldout_rrms"]["all_device_mean"]
    n = summary["n_devices"]
    result = {}
    for name in crit["candidate_methods"]:
        e = meth.get(name)
        if e is None:
            continue
        cov = e[f"coverage_{PRIMARY:g}"]["all_device_mean"]
        width = e[f"width_{PRIMARY:g}"]["all_device_mean"]
        c1 = (crit["coverage_lo"] <= cov <= crit["coverage_hi"]
              and width <= crit["max_width_over_single_rrms"] * single_rrms)
        rr = e["heldout_rrms"]["all_device_mean"]
        c2 = (rr <= (1 - crit["min_rel_rrms_gain"]) * single_rrms
              and e["wins_vs_single"] >= math.ceil(crit["min_win_fraction"] * n))
        c3 = True
        if name == "learned":
            c3 = rr <= meth["crossinterp"]["heldout_rrms"]["all_device_mean"]
        result[name] = {"coverage_0.9": cov, "width_0.9": width,
                        "heldout_rrms": rr, "wins": e["wins_vs_single"],
                        "C1": bool(c1), "C2": bool(c2), "C3": bool(c3)}
    full = [k for k, v in result.items() if v["C1"] and v["C2"] and v["C3"]]
    calib = [k for k, v in result.items() if v["C1"]]
    verdict = "CONTINUE" if full else ("PARTIAL_CALIBRATION_ONLY" if calib else "KILL")
    return {"single_heldout_rrms": single_rrms, "per_method": result,
            "verdict": verdict, "methods_passing": full}


def print_table(summary: dict) -> None:
    print(f"run={summary['run']} split={summary['split']} n={summary['n_devices']} "
          f"train_obj_mean={summary['train_obj_mean']:.4f}")
    print(f"{'method':24s} {'heldout':>8s} {'nmos':>7s} {'pmos':>7s} {'wins':>5s} "
          f"{'cov50':>6s} {'cov90':>6s} {'w90':>6s} {'IS90':>7s}")
    for m, e in summary["methods"].items():
        h = e["heldout_rrms"]
        c5 = e.get("coverage_0.5", {}).get("all_device_mean", math.nan)
        c9 = e.get("coverage_0.9", {}).get("all_device_mean", math.nan)
        w9 = e.get("width_0.9", {}).get("all_device_mean", math.nan)
        i9 = e.get("interval_score_0.9", {}).get("all_device_mean", math.nan)
        print(f"{m:24s} {h['all_device_mean']:8.4f} {h['nmos']:7.4f} {h['pmos']:7.4f} "
              f"{e.get('wins_vs_single', '-')!s:>5s} {c5:6.3f} {c9:6.3f} {w9:6.3f} {i9:7.3f}")
    for t, v in summary["identifiability"].items():
        print(f"  {t:20s} train={v['train_obj']:.4f} rank2={v['eff_rank_1e-2']} "
              f"rank3={v['eff_rank_1e-3']}/{v['n_active']} near={v['near_optimal_count']} "
              f"spread>0.1={v['near_optimal_n_spread_gt_0p1']} "
              f"sloppy_frac={v['heldout_sens_in_sloppy_frac']:.3g} calls={v['n_sim']} t={v['runtime_s']}")
    if "decision" in summary:
        print(json.dumps(summary["decision"], indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
