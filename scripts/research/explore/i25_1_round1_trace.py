#!/usr/bin/env python3
"""Logged rerun of I25-1 Step A round 1 for one device (step-deck figures).

Replicates scripts/research/ngspice_direct.fit_device (one start, CMA-ES, then cryoml.ngspice_fit.polish) with the
I25-1 arm settings (45-parameter arm B: CMA-ES 8,000 evaluations, sigma0 0.3, seed 20261015; polish max_nfev 600,
irls 2, diff_step 0.01) for the 3 random starts of one device (start draw: i25_1_scratch.draw_start, same seeds).
Every NGSpice run is logged with its phase and device RRMS; CMA-ES candidates also with their generation and the
VTH0 / U0 values.  The rerun's start RRMS, CMA-ES best, evaluation count, polish history, final RRMS and run count
are compared with the stored round-1 record (I25_1/stepA/full_s<k>_r1.json).

Outputs: out/research/explore/step_deck/round1_trace/<tag>.json (summary, comparison) and .npz (logs, curves;
not in git).

  PYTHONPATH=src .venv/bin/python scripts/research/explore/i25_1_round1_trace.py [--tag pmos_L0p35_W5]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("CRYOML_PDK_TEMPLATE", "mult1")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
for p in ("src", "scripts/research", "scripts/research/explore"):
    sys.path.insert(0, str(ROOT / p))
OUT = ROOT / "out/research/explore/step_deck/round1_trace"
STEPA = ROOT / "out/research/explore/I25_1/stepA"
SPECS = ROOT / "out/research/explore/I25_1/specs"
CMA_EVALS, POLISH_NFEV, IRLS, SIGMA0, SEED = 8000, 600, 2, 0.3, 20261015


def polish_logged(problem, local0, phase, *, max_nfev, diff_step=0.01, irls_rounds=3):
    """cryoml.ngspice_fit.polish with phase tags (same steps, same order of NGSpice runs)."""
    from scipy.optimize import least_squares
    active = np.flatnonzero(problem.box.active)
    local = np.asarray(local0, dtype=np.float64).copy()
    phase[0] = "ls_start"
    start_sims = problem.simulate(local)
    best_curves = problem.curve_rrms(start_sims)
    best = {"local": local.copy(), "objective": float(np.mean(best_curves))}
    history = [best["objective"]]
    weights = np.ones(problem.n_curves)
    for rnd in range(max(1, irls_rounds)):
        def residual(values, tag="ls_step"):
            phase[0] = tag
            trial = local.copy()
            trial[active] = values
            sims = problem.simulate(trial)
            curves = problem.curve_rrms(sims)
            value = float(np.mean(curves))
            if value < best["objective"]:
                best.update(local=trial.copy(), objective=value)
            return problem.residual_vector(sims, weights)

        def jacobian(values):
            base = residual(values, "ls_jac_base")
            columns = np.empty((base.size, values.size))
            for index in range(values.size):
                shifted = values.copy()
                step = diff_step if shifted[index] + diff_step <= 1.0 else -diff_step
                shifted[index] += step
                columns[:, index] = (residual(shifted, "ls_jac") - base) / step
            return columns

        budget = max(2, max_nfev // max(1, irls_rounds) // (len(active) + 2))
        x0 = np.clip(local[active], -1 + 1e-9, 1 - 1e-9)
        try:
            least_squares(residual, x0, method="trf", jac=jacobian,
                          bounds=(-np.ones(len(active)), np.ones(len(active))),
                          xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=budget)
        except ValueError:
            pass
        local = best["local"].copy()
        phase[0] = "ls_reweight"
        curves = problem.curve_rrms(problem.simulate(local))
        history.append(float(np.mean(curves)))
        weights = 1.0 / np.sqrt(np.maximum(curves, 1e-3))
        weights *= math.sqrt(problem.n_curves / float(np.sum(weights ** 2)))
    return {"local": best["local"], "objective": best["objective"], "history": history}


def run(task):
    tag, k = task
    import cma
    import i25_1_scratch as drv
    from cold_loop import box_for
    from cryoml.ngspice_fit import DeviceProblem
    t0 = time.time()
    spec1 = drv.one_device_spec(json.loads((SPECS / "full_r1.json").read_text()), tag)
    local0, draws, _ = drv.draw_start(tag, spec1, k)
    problem = DeviceProblem(tag, box_for(spec1, tag))
    active = np.flatnonzero(problem.box.active)
    names = problem.box.names
    phase = ["start"]
    log = {"phase": [], "rrms": [], "gen": [], "vth0": [], "u0": []}
    keep = {}
    gen = [0]
    orig = problem.simulate

    def logged(local):
        sims = orig(local)
        r = float(np.mean(problem.curve_rrms(sims)))
        p = problem.box.local_to_params(np.asarray(local, dtype=np.float64))
        log["phase"].append(phase[0])
        log["rrms"].append(r)
        log["gen"].append(gen[0] if phase[0] == "cma" else -1)
        log["vth0"].append(float(p["vth0"]))
        log["u0"].append(float(p["u0"]))
        keep["last"] = sims
        return sims
    problem.simulate = logged

    start_score = problem.objective(local0)
    keep["start"] = keep["last"]
    best_local, best_value = np.asarray(local0, dtype=np.float64), float(start_score)

    phase[0] = "cma"
    cma_best_sims = {"v": math.inf, "sims": None}

    def f(x):
        trial = best_local.copy()
        trial[active] = np.clip(x, -1, 1)
        v = problem.objective(trial)
        if v < cma_best_sims["v"]:
            cma_best_sims.update(v=v, sims=keep["last"])
        return v
    es = cma.CMAEvolutionStrategy(best_local[active], SIGMA0,
                                  {"bounds": [-1, 1], "maxfevals": CMA_EVALS, "seed": SEED, "verbose": -9,
                                   "popsize": 4 + int(3 * np.log(len(active)))})
    while not es.stop():
        xs = es.ask()
        gen[0] = es.countiter + 1
        es.tell(xs, [f(x) for x in xs])
    x = np.clip(es.result.xbest, -1, 1)
    if es.result.fbest < best_value:
        best_value = float(es.result.fbest)
        best_local = best_local.copy()
        best_local[active] = x
    n_cma = len(log["rrms"]) - 1
    res = polish_logged(problem, best_local, phase, max_nfev=POLISH_NFEV, irls_rounds=IRLS)
    if res["objective"] < best_value:
        best_value, best_local = res["objective"], res["local"]
    phase[0] = "final"
    final_sims = problem.simulate(best_local)
    out = {"tag": tag, "k": k, "draws": draws, "start_score": float(start_score),
           "cmaes_best": float(es.result.fbest), "cmaes_evals": int(es.result.evaluations),
           "polish_history": res["history"], "rrms": float(best_value), "n_sim": problem.n_sim - 1,
           "n_active": int(len(active)), "popsize": 4 + int(3 * np.log(len(active))),
           "names_active": [names[i] for i in active], "n_cma": n_cma, "runtime_s": round(time.time() - t0, 1),
           "box_vth0": [problem.box.local_to_params(np.full(len(names), v))["vth0"] for v in (-1.0, 1.0)],
           "box_u0": [problem.box.local_to_params(np.full(len(names), v))["u0"] for v in (-1.0, 1.0)]}
    arrays = {f"log_{k_}": np.asarray(v) for k_, v in log.items()}
    for lab, sims in (("start", keep["start"]), ("cma_best", cma_best_sims["sims"]), ("final", final_sims)):
        for i, s in enumerate(sims):
            arrays[f"sims_{lab}_{i}"] = np.asarray(s, dtype=np.float64)
    return out, arrays


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pmos_L0p35_W5")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with get_context("spawn").Pool(3) as pool:
        rows = pool.map(run, [(a.tag, k) for k in range(3)])
    summary, arrays = {"tag": a.tag, "starts": {}}, {}
    for out, arr in rows:
        k = out["k"]
        st = json.loads((STEPA / f"full_s{k}_r1.json").read_text())["devices"][a.tag]
        out["stored"] = {key: st[key] for key in ("start_scores", "cmaes_best", "cmaes_evals", "polish_history",
                                                  "rrms", "n_sim")}
        out["match"] = {
            "start": abs(out["start_score"] - st["start_scores"][0]) < 1e-9,
            "cmaes_best": abs(out["cmaes_best"] - st["cmaes_best"]) < 1e-9,
            "cmaes_evals": out["cmaes_evals"] == st["cmaes_evals"],
            "polish_history": bool(np.allclose(out["polish_history"], st["polish_history"], rtol=0, atol=1e-9)),
            "rrms": abs(out["rrms"] - st["rrms"]) < 1e-9, "n_sim": out["n_sim"] == st["n_sim"]}
        summary["starts"][str(k)] = out
        arrays.update({f"s{k}_{key}": v for key, v in arr.items()})
    summary["all_match"] = all(all(v["match"].values()) for v in summary["starts"].values())
    (OUT / f"{a.tag}.json").write_text(json.dumps(summary, indent=1, default=float))
    np.savez_compressed(OUT / f"{a.tag}.npz", **arrays)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("match", "rrms", "cmaes_best", "n_sim",
                                                                   "runtime_s", "stored")}
                      for k, v in summary["starts"].items()}, indent=1, default=float))
    print("all_match", summary["all_match"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
