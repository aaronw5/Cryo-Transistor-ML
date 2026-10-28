#!/usr/bin/env python3
"""Step deck: 2-parameter CMA-ES illustration on the example device (pmos_L0p35_W5, 45-parameter arm B box).

VTH0 and U0 are free; every other parameter is fixed at the round-1 result of start 2 (stored
I25_1/stepA/full_s1_r1.json "local", after CMA-ES and FD least squares).  Two NGSpice jobs:
  grid   device RRMS (DeviceProblem.objective, the CMA-ES loss) on a 30 x 30 grid over the round-1 range of
         VTH0 and U0 (scaled units -1 .. 1)
  cmaes  CMA-ES with the fit_device settings (cma 4.4.4, sigma0 0.3, bounds [-1, 1], seed 20261015,
         popsize 4 + int(3 ln 2) = 6), started from the VTH0 / U0 of the start-2 random start card, every
         generation logged (mean, sigma, C before sampling, samples, losses, new mean); stops on cma's own
         criteria or at 600 evaluations.

Output: out/research/explore/step_deck/round1_trace/<tag>_cmaes_2param.json

  PYTHONPATH=src .venv/bin/python scripts/research/explore/i25_1_cmaes_2param.py
"""
from __future__ import annotations

import json
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
I25 = ROOT / "out/research/explore/I25_1"
OUT = ROOT / "out/research/explore/step_deck/round1_trace"
TAG, K = "pmos_L0p35_W5", 1
NGRID, SEED, SIGMA0, MAXF = 30, 20261015, 0.3, 600


def problem_and_base():
    import i25_1_scratch as drv
    from cold_loop import box_for
    from cryoml.ngspice_fit import DeviceProblem
    spec1 = drv.one_device_spec(json.loads((I25 / "specs/full_r1.json").read_text()), TAG)
    problem = DeviceProblem(TAG, box_for(spec1, TAG))
    base = np.asarray(json.loads((I25 / f"stepA/full_s{K}_r1.json").read_text())["devices"][TAG]["local"], float)
    names = list(problem.box.names)
    return problem, base, names.index("vth0"), names.index("u0"), spec1


def grid_task(rows):
    problem, base, iv, iu, _ = problem_and_base()
    out = []
    for a, b in rows:
        x = base.copy()
        x[iv], x[iu] = a, b
        out.append(float(problem.objective(x)))
    return out


def main() -> int:
    import cma
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    problem, base, iv, iu, spec1 = problem_and_base()
    import i25_1_scratch as drv
    start_local, _, _ = drv.draw_start(TAG, spec1, K)
    g = np.linspace(-1.0, 1.0, NGRID)
    pts = [(a, b) for b in g for a in g]
    chunks = [pts[i::6] for i in range(6)]
    with get_context("spawn").Pool(6) as pool:
        res = pool.map(grid_task, chunks)
    z = np.empty(len(pts))
    for i in range(6):
        z[i::6] = res[i]
    grid = z.reshape(NGRID, NGRID)            # [u0 index, vth0 index]

    def f(xy):
        x = base.copy()
        x[iv], x[iu] = np.clip(xy, -1, 1)
        return float(problem.objective(x))
    lam = 4 + int(3 * np.log(2))
    es = cma.CMAEvolutionStrategy([start_local[iv], start_local[iu]], SIGMA0,
                                  {"bounds": [-1, 1], "maxfevals": MAXF, "seed": SEED, "verbose": -9,
                                   "popsize": lam})
    gens = []
    while not es.stop():
        m0, s0, c0 = es.mean.copy(), float(es.sigma), np.array(es.C, float)
        xs = es.ask()
        fs = [f(x) for x in xs]
        es.tell(xs, fs)
        gens.append({"mean": m0.tolist(), "sigma": s0, "C": c0.tolist(), "samples": [list(map(float, x)) for x in xs],
                     "loss": fs, "new_mean": es.mean.tolist(), "best_so_far": float(es.result.fbest)})
    box = problem.box

    def phys(a, b):
        x = base.copy()
        x[iv], x[iu] = a, b
        p = box.local_to_params(x)
        return float(p["vth0"]), float(p["u0"])
    out = {"tag": TAG, "fixed_at": f"stepA/full_s{K}_r1.json local (start {K + 1}, round 1 result)",
           "cma_version": cma.__version__, "popsize": lam, "mu": int(es.sp.weights.mu),
           "weights": [float(w) for w in es.sp.weights], "sigma0": SIGMA0, "seed": SEED, "maxfevals": MAXF,
           "stop": {k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in es.stop().items()},
           "grid_local": g.tolist(), "grid_rrms": grid.tolist(),
           "grid_vth0": [phys(a, 0.0)[0] for a in g], "grid_u0": [phys(0.0, b)[1] for b in g],
           "start_local": [float(start_local[iv]), float(start_local[iu])],
           "fixed_card_local": [float(base[iv]), float(base[iu])],
           "generations": gens, "evaluations": int(es.result.evaluations), "fbest": float(es.result.fbest),
           "xbest": [float(v) for v in es.result.xbest], "n_runs_grid": len(pts), "wall_s": round(time.time() - t0, 1)}
    (OUT / f"{TAG}_cmaes_2param.json").write_text(json.dumps(out, default=float))
    print(json.dumps({k: v for k, v in out.items() if k not in ("grid_rrms", "generations", "grid_local")},
                     default=float)[:1500])
    print("generations", len(gens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
