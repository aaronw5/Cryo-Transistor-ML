#!/usr/bin/env python3
"""Real vs held-out-simulated comparison table for every method.

Real = the 18 measured devices.  Sim = the same procedure, unchanged, run on
the held-out simulated devices of ``make_sim_targets.py`` (one set = 18
devices; sets are averaged).  The difference is the simulation-to-measurement
gap: what the method loses because the model card cannot reproduce the
silicon, as opposed to failing to invert the simulator.  ``param_err`` is the
median |ln(fitted / true)| over the seven parameters and all simulated devices.

  PYTHONPATH=src .venv/bin/python scripts/research/simgap_table.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cryoml.card_eval import aggregate  # noqa: E402
from cryoml.config import OUT_DIR  # noqa: E402

R = OUT_DIR / "research"
TRUTH = json.loads((R / "sim_targets" / "truth.json").read_text())


def direct(path):
    data = json.loads(path.read_text())
    return {t: {"rrms": r["rrms"], "params": r["params"]}
            for t, r in data["devices"].items()}


def pipeline(folder, method):
    rows = {}
    for path in sorted(Path(folder).glob("ml_*.json")):
        row = json.loads(path.read_text())
        m = row["methods"][method]
        rows[row["device"]] = {"rrms": m["rrms"], "params": m["params"]}
    return rows


def posterior(path):
    data = json.loads(path.read_text())
    return {t: {"rrms": r["polished"]["rrms"], "params": r["polished"]["params"]}
            for t, r in data["devices"].items() if "polished" in r}


METHODS = {
    "cold paper pipeline, round 4 (surrogate+FD)": (
        lambda: pipeline(OUT_DIR / "cold_loop/seven/round_4/extract", "surrogate_fd"),
        lambda k: pipeline(R / f"simgap/pipeline7_r4/set_{k}", "surrogate_fd")),
    "A: tandem autoencoder, round 4": (
        lambda: pipeline(OUT_DIR / "cold_loop/seven/round_4/extract", "inverse_autoencoder"),
        lambda k: pipeline(R / f"simgap/pipeline7_r4/set_{k}", "inverse_autoencoder")),
    "direct NGSpice (CMA-ES + TRF), round-1 box": (
        lambda: direct(R / "ngspice_direct/seven_r1_direct/summary.json"),
        lambda k: direct(R / f"ngspice_direct/sim{k}/seven_r1_direct/summary.json")),
    "direct NGSpice, 43 parameters, seeded by the seven-parameter fit": (
        lambda: direct(R / "ngspice_direct/full_r1_direct_seeded/summary.json"),
        lambda k: direct(R / f"ngspice_direct/sim{k}/full_r1_direct_seeded/summary.json")),
    "direct NGSpice, 43 parameters, round 2 box": (
        lambda: direct(R / "ngspice_direct/full_r2_direct/summary.json"),
        lambda k: direct(R / f"ngspice_direct/sim{k}/full_r2_direct/summary.json")),
    "B: flow, best of 256 + polish": (
        lambda: posterior(R / "posterior/seven_flow_r1/eval_main.json"),
        lambda k: posterior(R / f"posterior/seven_flow_r1/eval_sim{k}.json")),
    "B: 16-head WTA, best of 16 + polish": (
        lambda: posterior(R / "posterior/seven_wta_r1/eval_main.json"),
        lambda k: posterior(R / f"posterior/seven_wta_r1/eval_sim{k}.json")),
    "B: flow pooled over rounds 1-4, best of 256 + polish": (
        lambda: posterior(R / "posterior/seven_flow_pooled/eval_main.json"),
        lambda k: posterior(R / f"posterior/seven_flow_pooled/eval_sim{k}.json")),
    "B: 16-head WTA pooled over rounds 1-4 + polish": (
        lambda: posterior(R / "posterior/seven_wta_pooled/eval_main.json"),
        lambda k: posterior(R / f"posterior/seven_wta_pooled/eval_sim{k}.json")),
}

def fd_rows():
    P = R / "polished"
    return {
        f"{label} + NGSpice FD": (
            (lambda run=run: direct(P / run / "real" / "summary.json")),
            (lambda k, run=run: direct(P / run / f"sim{k}" / "summary.json")))
        for label, run in (
            ("cold paper pipeline, round 4", "pipe_r4_fd"),
            ("A: tandem autoencoder, round 4", "A_r4_fd"),
            ("direct NGSpice, seven", "direct7_fd"),
            ("direct NGSpice, 43 seeded", "direct43_fd"),
            ("B: flow r1 best of 256", "flow_r1_fd"),
            ("B: WTA r1 best of 16", "wta_r1_fd"),
            ("B: flow pooled best of 256", "flow_pooled_fd"),
            ("B: WTA pooled best of 16", "wta_pooled_fd"))}


METHODS.update(fd_rows())


def param_error(rows, truth):
    errors = []
    for tag, row in rows.items():
        for name, true in truth[tag]["params"].items():
            fit = row["params"].get(name)
            if fit is not None and true and fit / true > 0:
                errors.append(abs(np.log(fit / true)))
    return float(np.median(errors)) if errors else float("nan")


def main() -> int:
    table = {}
    for label, (real_fn, sim_fn) in METHODS.items():
        entry = {}
        try:
            rows = real_fn()
            entry["real"] = {**aggregate(rows), "median_device_rrms": float(
                np.median([r["rrms"] for r in rows.values()]))}
        except FileNotFoundError:
            pass
        sims, perr = [], []
        for k in TRUTH["sets"]:
            try:
                rows = sim_fn(k)
            except FileNotFoundError:
                continue
            if len(rows) != 18:
                continue
            sims.append({**aggregate(rows), "median_device_rrms": float(
                np.median([r["rrms"] for r in rows.values()]))})
            perr.append(param_error(rows, TRUTH["sets"][k]))
        if sims:
            entry["sim"] = {key: float(np.mean([s[key] for s in sims]))
                            for key in ("all_device_mean_rrms", "nmos_rrms",
                                        "pmos_rrms", "combined_rrms",
                                        "median_device_rrms")}
            entry["sim"]["n_sets"] = len(sims)
            entry["sim"]["per_set_all_device"] = [s["all_device_mean_rrms"] for s in sims]
            entry["sim"]["median_abs_log_param_error"] = float(np.mean(perr))
        table[label] = entry
    (R / "simgap_table.json").write_text(json.dumps(table, indent=1) + "\n")
    print("| method | real all-device | real combined | real median device | sim all-device | sim combined | sim median device | gap (real/sim, all-device) | param err (sim) | sets |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, e in table.items():
        r, s = e.get("real", {}), e.get("sim", {})
        f = lambda d, k: f"{d[k]:.4f}" if k in d else "—"
        gap = (f"{r['all_device_mean_rrms'] / s['all_device_mean_rrms']:.1f}x"
               if r and s and s["all_device_mean_rrms"] > 0 else "—")
        perr = (f"x{np.exp(s['median_abs_log_param_error']):.2f}" if s else "—")
        print(f"| {label} | {f(r,'all_device_mean_rrms')} | {f(r,'combined_rrms')} | "
              f"{f(r,'median_device_rrms')} | {f(s,'all_device_mean_rrms')} | "
              f"{f(s,'combined_rrms')} | {f(s,'median_device_rrms')} | {gap} | "
              f"{perr} | {s.get('n_sets', 0)} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
