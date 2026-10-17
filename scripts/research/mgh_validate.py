#!/usr/bin/env python3
"""Validate ``cryoml.dfo_ls.DfoLsRun`` on the Moré-Garbow-Hillstrom set
before any DEV run (no simulator).

Solvers, each from the instance's start ``x0 = 10^s x0_std`` with a budget of
``100 (n + 1)`` residual evaluations:

* ``dfols_style``: ``DfoLsRun`` unbounded, prior weight 0, DFO-LS's default
  scale rules for an unbounded problem: ``rho_beg = 0.1 max(||x0||_inf, 1)``,
  ``delta_max = 1e10``, ``rho_end = 1e-8``, soft restarts, seed 0;
* ``scipy_trf_fd``: ``scipy.optimize.least_squares(method='trf',
  jac='2-point')`` with scipy's default tolerances.

The residual function is wrapped with a counter, so every evaluation counts
(the finite-difference Jacobian evaluations of scipy included), and the best
``f = sum r^2`` is recorded against the evaluation count; the counter stops a
solver at ``100 (n + 1)`` evaluations.  Non-finite residuals never become a
best value.

Moré-Wild data profiles (2009, SIAM J. Optim. 20:172): an instance is solved
within ``kappa (n + 1)`` evaluations at tolerance ``tau`` if the best ``f``
found so far is ``<= f_L + tau (f(x0) - f_L)``, with ``f_L`` the minimum over
both solvers within ``100 (n + 1)`` evaluations; ``d(kappa)`` is the solved
fraction, for tau in {1e-1, 1e-3, 1e-5}.

Pre-registered pass rule (written in ``validation.json``):
  V1  tau = 1e-5, kappa = 100: dfols_style solves >= 0.75 of the instances;
  V2  tau = 1e-3: dfols_style's solved fraction at kappa = 10 and kappa = 25
      is >= scipy_trf_fd's.
If V1 or V2 fails, no DEV line for the dfols arms is produced
(``dfols_tr.py`` refuses DEV runs unless ``validation.json`` says pass) and
the per-instance results are reported.

Instances: ``mgh_problems.INSTANCES`` (see ``mgh_problems.INSTANCE_NOTE``).

  PYTHONPATH=src .venv/bin/python scripts/research/mgh_validate.py \
      --out out/research/dfols/mgh

``--check`` only checks the residual definitions: scipy TRF with exact-ish
convergence from each s = 0 start, against the published minima.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import mgh_problems as mgh  # noqa: E402
from cryoml.dfo_ls import STEP_NORM_NOTE, DfoLsRun, DfoLsSettings  # noqa: E402

TAUS = (1e-1, 1e-3, 1e-5)
KAPPA_MAX = 100
SOLVERS = ("dfols_style", "scipy_trf_fd")
RULE = {
    "V1": "tau = 1e-5, kappa = 100 (100 (n + 1) evaluations): dfols_style "
          "solved fraction >= 0.75",
    "V2": "tau = 1e-3: dfols_style solved fraction at kappa = 10 and at "
          "kappa = 25 >= scipy_trf_fd's",
    "consequence": "if V1 or V2 fails, no DEV queue line for dfols_style / "
                   "dfols_style_1start is produced and the per-instance "
                   "results are reported",
}


class BudgetReached(Exception):
    pass


class Counted:
    """Residual with an evaluation counter and the best-f history."""

    bounds = None

    def __init__(self, pid: int, m: int, budget: int):
        self.pid, self.m, self.budget = pid, m, budget
        self.n_calls = 0
        self.best = math.inf
        self.history: list[float] = []     # best f after each evaluation
        self.n_nonfinite = 0

    def __call__(self, x):
        if self.n_calls >= self.budget:
            raise BudgetReached
        return self.residual(x)

    def residual(self, x):
        self.n_calls += 1
        with np.errstate(all="ignore"):
            r = mgh.residual(self.pid, np.asarray(x, dtype=np.float64), self.m)
        f = float(r @ r) if np.all(np.isfinite(r)) else math.inf
        if not math.isfinite(f):
            self.n_nonfinite += 1
        self.best = min(self.best, f)
        self.history.append(self.best)
        return r

    def best_within(self, evaluations: int) -> float:
        if not self.history:
            return math.inf
        return self.history[min(evaluations, len(self.history)) - 1]


def run_dfols(pid, n, m, start, budget):
    counter = Counted(pid, m, budget)
    settings = DfoLsSettings(
        rho_beg=0.1 * max(float(np.max(np.abs(start))), 1.0),
        delta_max=1e10, rho_end=1e-8, seed=0)
    t0 = time.time()
    with np.errstate(all="ignore"):
        run = DfoLsRun(start, counter, 0.0, settings, label="mgh")
        run.advance(budget)
    summary = run.summary()
    summary.pop("log")
    return counter, {"status": run.status, "seconds": round(time.time() - t0, 3),
                     **{k: summary[k] for k in (
                         "stop_reason", "n_iterations", "n_trial",
                         "n_safety_steps", "n_geometry_steps",
                         "n_rho_reductions", "n_restarts",
                         "n_failed_points_skipped", "max_condition",
                         "n_lstsq_fallback", "n_subproblem_failures")},
                     "rho_beg": settings.rho_beg}


def run_scipy(pid, n, m, start, budget):
    counter = Counted(pid, m, budget)
    t0 = time.time()
    stop = "converged"
    try:
        with np.errstate(all="ignore"):
            result = least_squares(counter, start, method="trf", jac="2-point",
                                   max_nfev=budget)
        stop = f"status {result.status}: {result.message}"
    except BudgetReached:
        stop = "evaluation budget reached"
    except (ValueError, np.linalg.LinAlgError) as error:
        stop = f"error: {type(error).__name__}: {error}"
    return counter, {"stop": stop, "seconds": round(time.time() - t0, 3)}


def data_profiles(rows: list[dict]) -> dict:
    """``d_s(kappa)`` for kappa = 1..100 per tau and solver."""
    out = {}
    for tau in TAUS:
        out[str(tau)] = {}
        for solver in SOLVERS:
            fractions = []
            for kappa in range(1, KAPPA_MAX + 1):
                solved = [row["histories"][solver][kappa - 1]
                          <= row["f_L"] + tau * (row["f0"] - row["f_L"])
                          for row in rows]
                fractions.append(float(np.mean(solved)))
            out[str(tau)][solver] = fractions
    return out


def plot(profiles: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"dfols_style": "#2a78d6", "scipy_trf_fd": "#eb6834"}
    labels = {"dfols_style": "DfoLsRun (DFO-LS style)",
              "scipy_trf_fd": "scipy TRF, 2-point FD Jacobian"}
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    fig, axes = plt.subplots(1, len(TAUS), figsize=(10.5, 3.4), dpi=150,
                             sharey=True)
    fig.patch.set_facecolor(surface)
    kappa = np.arange(1, KAPPA_MAX + 1)
    for ax, tau in zip(axes, TAUS):
        ax.set_facecolor(surface)
        for solver in SOLVERS:
            ax.plot(kappa, profiles[str(tau)][solver], color=colors[solver],
                    lw=2, label=labels[solver])
        ax.set_title(f"tau = {tau:g}", fontsize=9, color=ink)
        ax.set_xlabel("evaluations / (n + 1)", color=ink, fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.grid(True, color=grid, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#b5b3ad")
        ax.tick_params(colors=muted, labelsize=8)
    axes[0].set_ylabel("fraction of instances solved", color=ink, fontsize=8)
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("Moré-Wild data profiles on the MGH instances "
                 "(every residual evaluation counted)", fontsize=9, color=ink)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def check_definitions() -> int:
    """scipy TRF (3-point FD, tight tolerances) from each s = 0 start; the
    best f against the published minimum."""
    worst = 0.0
    for pid, n, m, s in mgh.INSTANCES:
        if s:
            continue
        start = mgh.x0(pid, n)
        best = math.inf
        for _ in range(3):
            with np.errstate(all="ignore"):
                res = least_squares(lambda x: mgh.residual(pid, x, m), start,
                                    method="trf", jac="3-point", ftol=1e-15,
                                    xtol=1e-15, gtol=1e-15, max_nfev=20000)
            best = min(best, float(res.fun @ res.fun))
            start = res.x
        known = mgh.KNOWN_MINIMA.get((pid, n, m))
        rel = (abs(best - known) / max(abs(known), 1e-12)
               if known is not None else None)
        flag = ""
        if known is not None:
            ok = (best <= 1e-10) if known == 0.0 else rel < 1e-4
            flag = "ok" if ok else "MISMATCH"
            if known != 0.0:
                worst = max(worst, rel)
        print(f"{mgh.instance_name(pid, n, m, s):42s} f0 "
              f"{float(np.sum(mgh.residual(pid, mgh.x0(pid, n), m) ** 2)):.6g} "
              f"best {best:.9g} known {known} {flag}")
    print(f"worst relative gap to a nonzero published minimum: {worst:.2e}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "out" / "research" / "dfols" / "mgh"))
    parser.add_argument("--check", action="store_true",
                        help="check the residual definitions against the "
                             "published minima and exit")
    args = parser.parse_args()
    if args.check:
        return check_definitions()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    for pid, n, m, s in mgh.INSTANCES:
        start = (10.0 ** s) * mgh.x0(pid, n)
        budget = KAPPA_MAX * (n + 1)
        name = mgh.instance_name(pid, n, m, s)
        counters, details = {}, {}
        counters["dfols_style"], details["dfols_style"] = run_dfols(
            pid, n, m, start, budget)
        counters["scipy_trf_fd"], details["scipy_trf_fd"] = run_scipy(
            pid, n, m, start, budget)
        f0 = counters["dfols_style"].history[0]
        f_L = min(c.best_within(budget) for c in counters.values())
        histories = {k: [c.best_within(kappa * (n + 1))
                         for kappa in range(1, KAPPA_MAX + 1)]
                     for k, c in counters.items()}
        row = {"instance": name, "pid": pid, "function": mgh.NAMES[pid],
               "n": n, "m": m, "s": s, "budget": budget, "f0": f0, "f_L": f_L,
               "histories": histories,
               "solvers": {k: {"evaluations": c.n_calls,
                               "best_f": c.best_within(budget),
                               "n_nonfinite": c.n_nonfinite, **details[k]}
                           for k, c in counters.items()}}
        for tau in TAUS:
            threshold = f_L + tau * (f0 - f_L)
            row[f"solved_at_100_tau_{tau:g}"] = {
                k: bool(h[-1] <= threshold) for k, h in histories.items()}
        rows.append(row)
        print(f"{name:42s} f0 {f0:.4g} f_L {f_L:.6g} dfols {row['solvers']['dfols_style']['best_f']:.6g} "
              f"({counters['dfols_style'].n_calls}) trf {row['solvers']['scipy_trf_fd']['best_f']:.6g} "
              f"({counters['scipy_trf_fd'].n_calls}, {details['scipy_trf_fd']['stop'][:30]})",
              flush=True)
    profiles = data_profiles(rows)
    d = profiles
    v1_value = d["1e-05"]["dfols_style"][KAPPA_MAX - 1]
    v2 = {str(k): {"dfols_style": d["0.001"]["dfols_style"][k - 1],
                   "scipy_trf_fd": d["0.001"]["scipy_trf_fd"][k - 1]}
          for k in (10, 25)}
    v1_pass = v1_value >= 0.75
    v2_pass = all(v["dfols_style"] >= v["scipy_trf_fd"] for v in v2.values())
    validation = {
        "pass": bool(v1_pass and v2_pass),
        "V1": {"value": v1_value, "threshold": 0.75, "pass": bool(v1_pass)},
        "V2": {"kappa": v2, "pass": bool(v2_pass)},
        "rule": RULE,
        "n_instances": len(rows),
        "instance_note": mgh.INSTANCE_NOTE,
        "solver_settings": {
            "dfols_style": "DfoLsRun, unbounded, prior weight 0, rho_beg = "
                           "0.1 max(||x0||_inf, 1), delta_max 1e10, rho_end "
                           "1e-8, soft restarts (3 points), seed 0; "
                           + STEP_NORM_NOTE,
            "scipy_trf_fd": "scipy.optimize.least_squares(method='trf', "
                            "jac='2-point'), default ftol/xtol/gtol 1e-8; the "
                            "FD Jacobian evaluations are counted"},
        "f": "sum of squared residuals; f_L = min over both solvers within "
             "100 (n + 1) evaluations; f0 = f(x0)",
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "runtime_s": round(time.time() - t0, 1),
    }
    (out_dir / "results.json").write_text(json.dumps(rows, indent=1) + "\n")
    (out_dir / "profiles.json").write_text(json.dumps(
        {"kappa": list(range(1, KAPPA_MAX + 1)), "profiles": profiles},
        indent=1) + "\n")
    (out_dir / "validation.json").write_text(json.dumps(validation, indent=1) + "\n")
    plot(profiles, out_dir / "data_profiles.png")
    lines = ["# DfoLsRun validation on the MGH instances", "",
             f"{validation['instance_note']}.", "",
             f"Result: **{'PASS' if validation['pass'] else 'FAIL'}** "
             f"(V1 {'pass' if v1_pass else 'fail'}, V2 {'pass' if v2_pass else 'fail'}).",
             "", "| rule | dfols_style | scipy_trf_fd | threshold | pass |",
             "|---|---:|---:|---|---|",
             f"| V1: tau 1e-5, kappa 100 | {v1_value:.3f} | "
             f"{d['1e-05']['scipy_trf_fd'][KAPPA_MAX - 1]:.3f} | >= 0.75 | {v1_pass} |"]
    for k, v in v2.items():
        lines.append(f"| V2: tau 1e-3, kappa {k} | {v['dfols_style']:.3f} | "
                     f"{v['scipy_trf_fd']:.3f} | dfols >= scipy | "
                     f"{v['dfols_style'] >= v['scipy_trf_fd']} |")
    lines += ["", "Solved fractions at kappa = 1, 5, 10, 25, 50, 100:", "",
              "| tau | solver | k1 | k5 | k10 | k25 | k50 | k100 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for tau in TAUS:
        for solver in SOLVERS:
            vals = d[str(tau)][solver]
            lines.append(f"| {tau:g} | {solver} | " + " | ".join(
                f"{vals[k - 1]:.3f}" for k in (1, 5, 10, 25, 50, 100)) + " |")
    lines += ["", "## Per instance", "",
              "| instance | f0 | f_L | dfols best | trf best | trf evals | "
              "trf stop | solved 1e-5 (dfols/trf) |",
              "|---|---:|---:|---:|---:|---:|---|---|"]
    for row in rows:
        sv = row["solvers"]
        solved = row["solved_at_100_tau_1e-05"]
        lines.append(
            f"| {row['instance']} | {row['f0']:.4g} | {row['f_L']:.6g} | "
            f"{sv['dfols_style']['best_f']:.6g} | {sv['scipy_trf_fd']['best_f']:.6g} | "
            f"{sv['scipy_trf_fd']['evaluations']} | {sv['scipy_trf_fd']['stop'][:40]} | "
            f"{solved['dfols_style']}/{solved['scipy_trf_fd']} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: validation[k] for k in ("pass", "V1", "V2")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
