#!/usr/bin/env python3
"""Budgeted evolution-strategy baselines on NGSpice, with exact call traces.

The non-ML and surrogate-assisted ES arms a reviewer asks for, run through
the same oracle, bank start, objective and call accounting as the hybrid
trust-region arms (``hybrid_tr.setup``): the same pre-paid round bank and its
best sample ``x0``, the same ``NgspiceResiduals`` (failed points penalized at
curve RRMS 10; every call, Jacobian probes and cliff retries included, is
scored and can become the device's best card), and the same trace of the best
in-loop frozen-inclusion paper RRMS against cumulative NGSpice calls.

* ``cmaes_trf``: CMA-ES (``--sigma0`` 0.3, popsize 4 + 3 ln n = 15, box
  [-1, 1]) on the mean included-curve paper RRMS (as ``ngspice_direct``), one
  ``simulate_pdk_multi`` process per generation, the exact bank start ``x0``
  scored first (one call) as in the TR arms.  It stops on
  ``es.stop()`` or when the next generation would pass
  ``floor(--cma-fraction * budget)`` calls (the fraction is fixed across
  budgets and devices).  Then bounded TRF least squares from the in-loop best
  on the TR arms' objective (per-curve normalized residuals plus
  ``sqrt(reg / n) z`` prior rows) with the TR arms' NGSpice Jacobian (step
  ``--diff-step`` 1e-3, cliff guard), until the budget or convergence; calls
  left over stay unspent.
* ``lq_cmaes``: ``cma.fmin_lq_surr2`` (lq-CMA-ES, a linear/quadratic global
  surrogate decides which offspring NGSpice evaluates), IPOP restarts
  (popsize x2, seed + 1 per restart, as ``cma.fmin``; run by hand because
  ``fmin_lq_surr2``'s own restarts reuse the seed and replay the first run's
  samples), after one call on the exact bank start ``x0`` (every arm scores
  it first).
* ``ipop_cmaes`` (optional): ``cma.fmin2`` with the same options, restarts
  and population growth, each generation in one NGSpice process.

Every NGSpice access goes through ``Budgeted``, which refuses an access
before it would take the device past ``--budget``.  The final card gets one
fresh ``simulate_pdk`` and ``fps.score_real`` (``n_validation_calls`` 1).
Each device writes ``ml_<tag>.json`` in the hybrid_tr row shape, method key
``--method``; ``summary.json`` (``hybrid_tr.summarize``) once all 18 exist.

  CRYOML_TARGET_OVERRIDE=out/research/sim_targets/set_0.npz \
  PYTHONPATH=src .venv/bin/python scripts/research/cmaes_trace.py \
      --set full --round-dir out/cold_loop/full/round_1 \
      --bank data/processed/cold_loop/full_r1 --method cmaes_trf \
      --budget 5000 --out out/research/dev/cmaes_trf/B5000/set_0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib
from collections import OrderedDict
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import foundation_parameter_study as fps  # noqa: E402
import hybrid_tr  # noqa: E402
from cryoml.data_io import TARGET_OVERRIDE_ENV  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.spice_pdk import simulate_pdk  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402
from hybrid_tr import BUDGET_MARKS, best_at  # noqa: E402

RUNS = ROOT / "out" / "research" / "cmaes_trace"
METHODS = ("cmaes_trf", "lq_cmaes", "ipop_cmaes")
INTERIOR = 1.0 - 1e-9


class BudgetExhausted(RuntimeError):
    """The next NGSpice access would take the device past its budget."""


class Budgeted:
    """The one door to NGSpice for a device: an access that would pass
    ``budget`` is refused before it runs, NGSpice time is split into residual
    and Jacobian seconds, residuals are cached by ``z`` bytes (so a Jacobian
    after a residual at the same point re-simulates nothing), and a new
    in-loop best is labelled with the phase that produced it."""

    def __init__(self, ngspice, budget: int, cache_size: int = 64):
        self.ngspice = ngspice
        self.budget = int(budget)
        # Cliff retries inside ``ngspice.jacobian`` stop at this cap.
        ngspice.call_cap = min(getattr(ngspice, "call_cap", math.inf), self.budget)
        self.cache: OrderedDict[bytes, np.ndarray] = OrderedDict()
        self.cache_size = cache_size
        self.seconds = {"residual": 0.0, "jacobian": 0.0}

    @property
    def n_calls(self) -> int:
        return self.ngspice.n_calls

    def charge(self, cost: int) -> None:
        if self.ngspice.n_calls + cost > self.budget:
            raise BudgetExhausted(
                f"{self.ngspice.n_calls} + {cost} > {self.budget}")

    def _store(self, z: np.ndarray, r: np.ndarray) -> None:
        self.cache[z.tobytes()] = r
        self.cache.move_to_end(z.tobytes())
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def _relabel(self, before: dict, kind: str) -> bool:
        if self.ngspice.best is before:
            return False
        self.ngspice.best["kind"] = kind
        return True

    def many(self, points: list, kind: str) -> list[float]:
        """A whole population in one NGSpice process; paper RRMS per point."""
        points = [np.asarray(p, dtype=np.float64) for p in points]
        self.charge(len(points))
        before, t0 = self.ngspice.best, time.time()
        rows = self.ngspice._simulate_many(points)
        self.seconds["residual"] += time.time() - t0
        for point, (_r, rrms, _failed) in zip(points, rows):
            self.ngspice.centre_rrms[point.tobytes()] = rrms
        if self._relabel(before, kind):
            key = self.ngspice.best["local"][self.ngspice.active].tobytes()
            for point, row in zip(points, rows):
                if point.tobytes() == key:
                    self._store(point, row[0])
        return [float(row[1]) for row in rows]

    def residual(self, z, kind: str) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        if z.tobytes() in self.cache:
            return self.cache[z.tobytes()]
        self.charge(1)
        before, t0 = self.ngspice.best, time.time()
        r = self.ngspice.residual(z)
        self.seconds["residual"] += time.time() - t0
        self._relabel(before, kind)
        self._store(z, r)
        return r

    def rrms(self, z) -> float:
        return float(self.ngspice.centre_rrms[np.asarray(z, np.float64).tobytes()])

    def jacobian(self, z, kind: str) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        r = self.residual(z, kind)          # a cache miss costs one call
        self.charge(self.ngspice.jacobian_cost)
        before, t0 = self.ngspice.best, time.time()
        jac = self.ngspice.jacobian(z, r)
        self.seconds["jacobian"] += time.time() - t0
        self._relabel(before, kind)
        return jac


def device_seed(seed: int, tag: str) -> int:
    base = os.path.basename(os.environ.get(TARGET_OVERRIDE_ENV, ""))
    return (seed + zlib.crc32((tag + base).encode())) & 0x7FFFFFFF


def popsize_for(options: dict, n: int) -> int:
    return options["popsize"] or 4 + int(3 * math.log(n))


def cma_phase(guard: Budgeted, x0, options: dict, seed: int, limit: int,
              stats: dict) -> None:
    import cma
    stats.update(label="cma", generations=0, overhead_s=0.0, stop=None,
                 calls_start=guard.n_calls)
    es = cma.CMAEvolutionStrategy(x0, options["sigma0"], {
        "bounds": [-1, 1], "popsize": popsize_for(options, len(x0)),
        "seed": seed, "verbose": -9})
    # Score the exact bank start first (one call), as the TR arms do.  Not by
    # es.inject: pycma's boundary transform moves coordinates within 0.1 of a
    # bound (up to 0.025 at a face), so the injected point is not x0.
    if guard.n_calls + 1 <= limit:
        guard.many([np.asarray(x0, dtype=np.float64)], "cma")
        stats["calls"] = guard.n_calls - stats["calls_start"]
    while True:
        stop = es.stop()
        if stop:
            stats["stop"] = {k: str(v) for k, v in stop.items()}
            break
        if guard.n_calls + es.sp.popsize > limit:
            stats["stop"] = {"cma_fraction": str(limit)}
            break
        t0 = time.time()
        xs = es.ask()
        stats["overhead_s"] += time.time() - t0
        values = guard.many([np.clip(x, -1.0, 1.0) for x in xs], "cma")
        t0 = time.time()
        es.tell(xs, values)
        stats["overhead_s"] += time.time() - t0
        stats["generations"] += 1
        stats["calls"] = guard.n_calls - stats["calls_start"]


def trf_phase(guard: Budgeted, weight: float, stats: dict) -> None:
    ngspice = guard.ngspice
    z0 = np.clip(ngspice.best["local"][ngspice.active], -INTERIOR, INTERIOR)
    eye = np.eye(len(z0))
    stats.update(label="trf", nfev=0, njev=0, stop=None, calls=0,
                 calls_start=guard.n_calls, start_rrms=ngspice.best["rrms"])
    t0, sim0 = time.time(), sum(guard.seconds.values())

    def fun(z):
        r = guard.residual(z, "trf_trial")
        stats["nfev"] += 1
        stats["calls"] = guard.n_calls - stats["calls_start"]
        return np.concatenate([r, weight * np.asarray(z)])

    def jac(z, *_):
        J = guard.jacobian(z, "trf_jacobian_probe")
        stats["njev"] += 1
        stats["calls"] = guard.n_calls - stats["calls_start"]
        return np.vstack([J, weight * eye])

    try:
        solution = least_squares(
            fun, z0, jac=jac, bounds=(-INTERIOR, INTERIOR), method="trf",
            x_scale=1.0, ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=10 ** 6)
        stats["stop"] = {"converged": f"{solution.status}: {solution.message}"}
    finally:
        # Every call of the phase, so cma_calls + trf_calls == n_ngspice_calls
        # even if the budget ran out inside a Jacobian after a centre call.
        stats["calls"] = guard.n_calls - stats["calls_start"]
        stats["wall_s"] = time.time() - t0
        stats["overhead_s"] = max(0.0, stats["wall_s"] - (
            sum(guard.seconds.values()) - sim0))


def es_phase(guard: Budgeted, x0, options: dict, seed: int, stats: dict,
             method: str) -> None:
    """lq-CMA-ES (one NGSpice call per true evaluation) or IPOP-CMA-ES (one
    process per generation), with IPOP restarts from ``x0``."""
    import cma
    stats.update(label=method, generations=0, stop=None, calls_start=0,
                 calls=0)
    t0, sim0 = time.time(), sum(guard.seconds.values())
    opts = {"bounds": [-1, 1], "maxfevals": guard.budget, "seed": seed,
            "verbose": -9, "popsize": popsize_for(options, len(x0))}

    def count(es):
        stats["generations"] += 1
        stats["calls"] = guard.n_calls

    def single(x):
        z = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
        guard.residual(z, method)
        return guard.rrms(z)

    def batch(xs, *_):
        return guard.many([np.clip(np.asarray(x, np.float64), -1.0, 1.0)
                           for x in xs], method)

    try:
        if method == "lq_cmaes":
            # fmin_lq_surr2 has no eval_initial_x: score the bank start first
            # (one counted call), as every other arm does (TR arms, cmaes_trf
            # and ipop score the exact x0 first), so the arm's best-so-far
            # never starts above the pre-paid bank start.
            single(np.asarray(x0, dtype=np.float64))
            stats["calls"] = guard.n_calls
            # IPOP restarts by hand: fmin_lq_surr2(restarts=k) rebuilds each
            # restart with the SAME seed, so every restart replays the first
            # run's random stream (its first offspring re-simulated at a
            # charged NGSpice call each).  cma.fmin bumps the seed per
            # restart; do the same.  popsize x2 per restart (incpopsize 2).
            stats["runs"] = []
            for irun in range(1 + options["restarts"]):
                run_opts = {**opts, "seed": max(1, (seed + irun) & 0x7FFFFFFF),
                            "maxfevals": guard.budget - guard.n_calls}
                _x, es = cma.fmin_lq_surr2(single, x0, options["sigma0"],
                                           run_opts, restarts=0,
                                           callback=count)
                stop = es.stop(check=False)
                stats["runs"].append({
                    "popsize": int(es.sp.popsize), "seed": run_opts["seed"],
                    "stop": {k: str(v) for k, v in stop.items()},
                    "calls_end": guard.n_calls})
                if "maxfevals" in stop or "ftarget" in stop:
                    break
                opts["popsize"] = int(es.sp.popsize * 2 + 0.5)
        else:
            _x, es = cma.fmin2(None, x0, options["sigma0"], opts,
                               restarts=options["restarts"], incpopsize=2,
                               eval_initial_x=True, parallel_objective=batch,
                               callback=count)
        stats["stop"] = {k: str(v) for k, v in es.stop(check=False).items()}
    finally:
        stats["wall_s"] = time.time() - t0
        stats["overhead_s"] = max(0.0, stats["wall_s"] - (
            sum(guard.seconds.values()) - sim0))
        stats["calls"] = guard.n_calls


def fit_device(job) -> tuple[str, dict]:
    tag, options = job
    t0 = time.time()
    method, budget = options["method"], options["budget"]
    context, ngspice, starts, oracle, _ = hybrid_tr.setup(tag, {
        **options, "model": "none", "n_starts": 4})
    active = ngspice.active
    x0 = np.asarray(starts[0], dtype=np.float64)[active]
    guard = Budgeted(ngspice, budget)
    seed = device_seed(options["seed"], tag)
    weight = hybrid_tr.prior_weight(options, len(active))
    phases: list[dict] = []
    try:
        if method == "cmaes_trf":
            phases.append({})
            cma_phase(guard, x0, options, seed,
                      int(math.floor(options["cma_fraction"] * budget)),
                      phases[-1])
            phases.append({})
            trf_phase(guard, weight, phases[-1])
        else:
            phases.append({})
            es_phase(guard, x0, options, seed, phases[-1], method)
    except BudgetExhausted as error:
        phases[-1]["stop"] = {"budget": str(error)}
    assert ngspice.n_calls <= budget, (tag, ngspice.n_calls, budget)
    best = ngspice.best
    params = context.box.local_to_params(best["local"])
    sims = simulate_pdk(
        context.device.dev_type, context.device.L_um, context.device.W_um,
        context.curves, params=params, bin_index=context.bin_index)
    score = fps.score_real(context, sims)
    ev = ngspice.evaluator
    es_stats = phases[0]
    trf = phases[1] if len(phases) > 1 else {}
    # ES model overhead (ask/tell, lq model fits) plus TRF linear algebra.
    overhead = sum(p.get("overhead_s", 0.0) for p in phases)
    n_iterations = es_stats.get("generations", 0) + trf.get("nfev", 0)
    seconds = {"subproblem": overhead, **guard.seconds}
    record = {
        "method": {"cmaes_trf": "cma_es_then_trf_least_squares_on_ngspice",
                   "lq_cmaes": "lq_cma_es_ipop_on_ngspice",
                   "ipop_cmaes": "ipop_cma_es_on_ngspice"}[method],
        "model": "none",
        "params": params, "local": best["local"].tolist(),
        "rrms": float(score["rrms"]), "sigma": float(score["sigma"]),
        "rmse_uA": float(score["rmse_uA"]),
        "official_dynamic_rrms": float(score["official_dynamic_rrms"]),
        "high_voltage_rrms": float(score["high_voltage_rrms"]),
        "per_curve": score["per_curve"],
        "in_loop_best_rrms": best["rrms"],
        "best_found_at_call": best["calls"], "best_kind": best["kind"],
        "n_ngspice_calls": ev.n_calls, "n_validation_calls": 1,
        "budget": budget, "trace": ngspice.trace,
        "budget_rrms": {str(b): best_at(ngspice.trace, b)
                        for b in BUDGET_MARKS},
        "n_failed": ev.n_failed, "n_failed_points": ev.n_failed_points,
        "n_cliff_retries": ngspice.n_cliff_retries,
        "n_zero_jacobian_columns": ngspice.n_zero_columns,
        "n_iterations": n_iterations, "n_jacobians": ngspice.n_jacobians,
        "seconds": {k: round(v, 2) for k, v in seconds.items()},
        "subproblem_s_per_iteration": round(
            overhead / max(1, n_iterations), 4),
        "ngspice_s_per_call": round(ev.sim_s / max(1, ev.n_calls), 4),
        "starts": [{k: v for k, v in p.items() if k != "calls_start"}
                   for p in phases],
        "oracle_lhc": oracle, "n_starts": 1, "schedule": method,
        "diff_step": options["diff_step"],
        "regularization": options["regularization"], "delta0": None,
        "optimization_objective": (
            "ES: mean included-curve paper RRMS (as ngspice_direct); TRF: "
            "uniform mean squared per-curve normalized point residual plus "
            "weak regularization (as the hybrid TR arms)"),
        "runtime_s": round(time.time() - t0, 1),
        "cma_calls": es_stats.get("calls", 0),
        "trf_calls": trf.get("calls", 0),
        "cma_generations": es_stats.get("generations", 0),
        "cma_stop": es_stats.get("stop"),
        "trf_nfev": trf.get("nfev", 0), "trf_stop": trf.get("stop"),
        "unspent_calls": budget - ev.n_calls, "seed": seed,
        "sigma0": options["sigma0"],
        "popsize": popsize_for(options, len(active)),
        "cma_fraction": options["cma_fraction"] if method == "cmaes_trf" else None,
        "restarts": options["restarts"] if method != "cmaes_trf" else 0,
        "x0_evaluated": True,
        "target_override": os.path.basename(
            os.environ.get(TARGET_OVERRIDE_ENV, "")) or None,
    }
    device = context.device
    row = {"device": tag, "dev_type": device.dev_type, "L_um": device.L_um,
           "W_um": device.W_um, "bin_index": context.bin_index,
           "n_active_parameters": context.box.n_active,
           "methods": {method: record}}
    path = Path(options["out"]) / f"ml_{tag}.json"
    partial = path.with_suffix(".tmp")
    partial.write_text(json.dumps(row, indent=1) + "\n")
    partial.replace(path)
    print(f"{tag}: rrms {record['rrms']:.4f} (in loop {best['rrms']:.4f}, "
          f"{best['kind']} at {best['calls']}) calls {ev.n_calls} "
          f"[es {record['cma_calls']} trf {record['trf_calls']} unspent "
          f"{record['unspent_calls']}] {record['runtime_s']:.0f}s", flush=True)
    return tag, record


def load_records(directory: Path, method: str) -> dict[str, dict]:
    rows = (json.loads(p.read_text()) for p in sorted(directory.glob("ml_*.json")))
    return {row["device"]: row["methods"][method] for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-dir", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--set", choices=("seven", "full"), default=None,
                        help="output label (default: the round dir's parent)")
    parser.add_argument("--method", choices=METHODS, default="cmaes_trf")
    parser.add_argument("--budget", type=int, default=5000,
                        help="NGSpice calls per device, Jacobian probes included")
    parser.add_argument("--cma-fraction", type=float, default=0.6,
                        help="cmaes_trf: share of the budget for CMA-ES")
    parser.add_argument("--sigma0", type=float, default=0.3)
    parser.add_argument("--popsize", type=int, default=0,
                        help="0: 4 + 3 ln n (15 for 43 parameters)")
    parser.add_argument("--restarts", type=int, default=9,
                        help="IPOP restarts for lq_cmaes / ipop_cmaes")
    parser.add_argument("--diff-step", type=float, default=1e-3)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--devices", default=None,
                        help="comma-separated device tags (smoke tests)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--pipeline", default=None,
                        help="surrogate extract dir to report side by side "
                             "(default <round-dir>/extract)")
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    round_dir = Path(args.round_dir)
    label = args.set or round_dir.parent.name
    # The default directory names the target set, so a DEV (override) run
    # and a TEST run never share a directory (and resume never mixes them).
    override = Path(os.environ.get(TARGET_OVERRIDE_ENV, "")).stem
    out_dir = Path(args.out) if args.out else RUNS / (
        f"{label}_{round_dir.name}_{args.method}_B{args.budget}"
        f"_{override or 'measured'}")
    out_dir.mkdir(parents=True, exist_ok=True)
    options = {"round_dir": str(round_dir), "bank": args.bank,
               "out": str(out_dir), "model": "none", **{k: getattr(args, k) for k in (
                   "method", "budget", "cma_fraction", "sigma0", "popsize",
                   "restarts", "diff_step", "regularization", "timeout_s",
                   "seed")}}
    tags = [device_tag(d.dev_type, d.L_um, d.W_um) for d in PAPER_DEVICES]
    if args.devices:
        tags = [t for t in tags if t in args.devices.split(",")]
    todo = [t for t in tags if not (out_dir / f"ml_{t}.json").exists()]
    # Longest-running (pMOS) devices first keeps the pool busy.
    todo.sort(key=lambda t: not t.startswith("pmos"))
    t0 = time.time()
    if todo:
        with get_context("spawn").Pool(min(args.workers, len(todo))) as pool:
            for _ in pool.imap_unordered(
                    fit_device, [(t, options) for t in todo]):
                pass
    records = load_records(out_dir, args.method)
    if len(records) < len(PAPER_DEVICES):
        print(f"{len(records)}/{len(PAPER_DEVICES)} devices done; "
              "summary.json is written once all exist")
        return 0
    out = hybrid_tr.summarize(records, options, round_dir, args, len(todo),
                              time.time() - t0)
    out["method"] = args.method
    out["mean_unspent_calls_per_device"] = float(np.mean(
        [r["unspent_calls"] for r in records.values()]))
    (out_dir / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in (
        "all_device_mean_rrms", "nmos_rrms", "pmos_rrms", "combined_rrms",
        "mean_ngspice_calls_per_device", "mean_unspent_calls_per_device",
        "wall_s")}, indent=1))
    print("at budget", json.dumps({b: v["all_device_mean_rrms"]
                                   for b, v in out["at_budget"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
