#!/usr/bin/env python3
"""Named model-based DFO least-squares baseline on NGSpice (research node n2c).

``cryoml.dfo_ls.DfoLsRun`` (the DFO-LS core: linear-interpolation
Gauss-Newton trust region, one NGSpice call per iteration) through the same
oracle, starts, objective and call accounting as the hybrid trust-region and
sketch arms:

* ``hybrid_tr.setup`` with ``model 'none'``: the round-1 bank, its 4 diverse
  bank starts (``fps.diverse_oracle_starts``), ``NgspiceResiduals`` (per-curve
  normalized residuals on the frozen-inclusion paper curves; failed points
  penalized at curve RRMS 10; every call scored into the device's best card
  and the trace);
* objective ``0.5 ||r(z)||^2 + 0.5 w^2 ||z||^2``, ``w = sqrt(reg / n_active)``
  (reg 1e-4), over the active coordinates in ``[-1, 1]``;
* ``--starts 4`` (arm ``dfols_style``): the 4 starts share the budget through
  ``trust_region.run_schedule`` (equal schedule), then
  ``dfo_ls.finish_budget`` gives any calls left to the best run that can
  still use them; ``--starts 1`` (arm ``dfols_style_1start``): ``starts[0]``
  with the whole budget;
* ``rho_beg`` 0.1 (``Delta_0 = rho_beg``), ``delta_max`` 1 (the box), soft
  restarts at ``rho_end`` 1e-8, restart directions seeded by ``--seed`` and
  the device tag.

A failed NGSpice point is one whose included-curve residuals carry the
penalty (``NgspiceEvaluator.n_failed`` increased during the call and at least
one residual equals the penalized value bit for bit); such points never
enter the interpolation set (``n_failed_points_skipped``).  Every NGSpice
call counts toward ``--budget``; one fresh simulation of the final card is
the score (``fps.score_real``, ``n_validation_calls`` 1).

Deviation from DFO-LS, stated in every record: ``dfo_ls.STEP_NORM_NOTE``.

Each device writes ``ml_<tag>.json`` in the ``hybrid_tr`` row format, method
key ``dfols_style``; ``summary.json`` (``hybrid_tr.summarize``, with
``target_override``) once all 18 exist (for a ``--devices`` subset it is
written with ``partial_device_subset``).  DEV runs are refused unless the MGH
validation (``mgh_validate.py``) passed (``--validation``).

  CRYOML_TARGET_OVERRIDE=out/research/sim_targets/set_0.npz \\
  PYTHONPATH=src .venv/bin/python scripts/research/dfols_tr.py \\
      --round-dir out/cold_loop/full/round_1 \\
      --bank data/processed/cold_loop/full_r1 --set full --starts 4 \\
      --budget 500 --out out/research/dev/dfols_style/B500/set_0
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
import zlib
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import foundation_parameter_study as fps  # noqa: E402
import hybrid_tr  # noqa: E402
from cryoml.data_io import target_override  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.dfo_ls import (STEP_NORM_NOTE, DfoLsRun, DfoLsSettings,  # noqa: E402
                           finish_budget)
from cryoml.spice_pdk import simulate_pdk  # noqa: E402
from cryoml.trust_region import run_schedule  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402

METHOD = "dfols_style"
RUNS = ROOT / "out" / "research" / "dfols"
VALIDATION = RUNS / "mgh" / "validation.json"
NGSPICE_S_PER_CALL = 0.031      # per-call cost used for call equivalents
ARM_NOTE = ("DFO-LS-style linear-interpolation Gauss-Newton trust region "
            "(cryoml.dfo_ls, re-implemented, MGH-validated), one NGSpice call "
            "per iteration; " + STEP_NORM_NOTE)


class DfoBackend:
    """``hybrid_tr.NgspiceResiduals`` for ``DfoLsRun``: ``n_calls``,
    ``residual``, the box ``[-1, 1]`` and ``last_failed`` (an included
    curve carried the failure penalty in the last call)."""

    def __init__(self, ngspice):
        self.ngspice = ngspice
        ev = ngspice.evaluator
        self.penalty = (ev.fallback[ngspice.index] - ngspice.measured) / ngspice.scale
        self.bounds = (-1.0, 1.0)
        self.last_failed = False
        self.n_failed_included = 0
        self.n_failed_other_curves = 0

    @property
    def n_calls(self) -> int:
        return self.ngspice.n_calls

    def residual(self, z):
        ev = self.ngspice.evaluator
        before = ev.n_failed
        r = self.ngspice.residual(np.asarray(z, dtype=np.float64))
        self.last_failed = False
        if ev.n_failed > before:
            if np.any(r == self.penalty):
                self.last_failed = True
                self.n_failed_included += 1
            else:
                self.n_failed_other_curves += 1
        return r


def check_validation(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"{path} missing: run mgh_validate.py first "
                         "(DEV runs need a passing MGH validation)")
    validation = json.loads(path.read_text())
    if validation.get("pass") is not True:
        raise SystemExit(f"{path}: MGH validation did not pass (V1/V2); no DEV "
                         "run of the dfols arms")
    return validation


def fit_device(job) -> tuple[str, dict]:
    tag, options = job
    t0 = time.time()
    context, ngspice, starts, oracle, _ = hybrid_tr.setup(
        tag, {**options, "model": "none", "n_starts": 4})
    starts = starts[:options["starts"]]
    backend = DfoBackend(ngspice)
    active = ngspice.active
    weight = hybrid_tr.prior_weight(options, len(active))
    settings = DfoLsSettings(rho_beg=options["rho_beg"], seed=options["seed"])
    stream = zlib.crc32(tag.encode())
    runs = [DfoLsRun(np.asarray(start, dtype=np.float64)[active], backend,
                     weight, settings, label=f"start{index}",
                     stream=stream + index)
            for index, start in enumerate(starts)]
    run_schedule(runs, backend, options["budget"], "equal")
    leftover = finish_budget(runs, backend, options["budget"])
    best = ngspice.best
    params = context.box.local_to_params(best["local"])
    sims = simulate_pdk(
        context.device.dev_type, context.device.L_um, context.device.W_um,
        context.curves, params=params, bin_index=context.bin_index)
    score = fps.score_real(context, sims)
    ev = ngspice.evaluator
    summaries = [run.summary() for run in runs]
    seconds = {key: sum(run.seconds[key] for run in runs)
               for key in ("subproblem", "residual", "model")}
    n_iterations = sum(run.n_iterations for run in runs)
    record = {
        "method": "dfols_style_linear_interpolation_gauss_newton_tr",
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
        "budget": options["budget"], "trace": ngspice.trace,
        "budget_rrms": {str(b): hybrid_tr.best_at(ngspice.trace, b)
                        for b in hybrid_tr.BUDGET_MARKS},
        "n_failed": ev.n_failed, "n_failed_points": ev.n_failed_points,
        "n_failed_included_calls": backend.n_failed_included,
        "n_failed_other_curve_calls": backend.n_failed_other_curves,
        "n_iterations": n_iterations,
        "seconds": {k: round(v, 2) for k, v in seconds.items()},
        "subproblem_s_per_iteration": round(
            seconds["subproblem"] / max(1, n_iterations), 4),
        "ngspice_s_per_call": round(ev.sim_s / max(1, ev.n_calls), 4),
        "starts": summaries,
        "target_override": target_override() is not None,
        "oracle_lhc": oracle, "n_starts": len(runs),
        "schedule": "equal", "leftover_calls_after_schedule": leftover,
        "regularization": options["regularization"],
        "prior_weight": weight,
        "optimization_objective": (
            "uniform mean squared per-curve normalized point residual plus "
            "weak regularization (as the surrogate-FD pipeline)"),
        "runtime_s": round(time.time() - t0, 1),
        # DFO-LS counters (all starts).
        "rho_beg": settings.rho_beg,
        "dfols_settings": dataclasses.asdict(settings),
        "n_geometry_steps": sum(run.n_geometry_steps for run in runs),
        "n_safety_steps": sum(run.n_safety_steps for run in runs),
        "n_restarts": sum(run.n_restarts for run in runs),
        "n_failed_points_skipped": sum(run.n_failed_points_skipped for run in runs),
        "n_rho_reductions": sum(run.n_rho_reductions for run in runs),
        "n_trial": sum(run.n_trial for run in runs),
        "n_success": sum(run.n_success for run in runs),
        "n_lstsq_fallback": sum(run.n_lstsq_fallback for run in runs),
        "max_condition": max(run.max_condition for run in runs),
        "subproblem_call_equivalents": round(
            (seconds["subproblem"] + seconds["model"]) / NGSPICE_S_PER_CALL, 1),
        "step_norm_note": STEP_NORM_NOTE,
        "seed": options["seed"],
    }
    device = context.device
    row = {"device": tag, "dev_type": device.dev_type, "L_um": device.L_um,
           "W_um": device.W_um, "bin_index": context.bin_index,
           "n_active_parameters": context.box.n_active,
           "methods": {METHOD: record}}
    path = Path(options["out"]) / f"ml_{tag}.json"
    partial = path.with_suffix(".tmp")
    partial.write_text(json.dumps(row, indent=1) + "\n")
    partial.replace(path)
    print(f"{tag}: rrms {record['rrms']:.4f} (in loop {best['rrms']:.4f}, "
          f"first call {ngspice.trace[0][1]:.4f}) calls {ev.n_calls} "
          f"iters {n_iterations} trial {record['n_trial']} "
          f"geom {record['n_geometry_steps']} safety {record['n_safety_steps']} "
          f"restarts {record['n_restarts']} "
          f"skipped {record['n_failed_points_skipped']} "
          f"sub/it {record['subproblem_s_per_iteration']:.4f}s "
          f"{record['runtime_s']:.0f}s", flush=True)
    return tag, record


def load_records(directory: Path) -> dict[str, dict]:
    rows = (json.loads(p.read_text()) for p in sorted(directory.glob("ml_*.json")))
    return {row["device"]: row["methods"][METHOD] for row in rows}


def dfols_summary(records: dict) -> dict:
    rows = list(records.values())
    keys = ("n_geometry_steps", "n_safety_steps", "n_restarts",
            "n_failed_points_skipped", "n_rho_reductions", "n_trial",
            "n_iterations", "subproblem_call_equivalents")
    return {**{f"mean_{k}_per_device": float(np.mean([r[k] for r in rows]))
               for k in keys},
            "max_condition": float(max(r["max_condition"] for r in rows)),
            "step_norm_note": STEP_NORM_NOTE}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-dir", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--set", choices=("seven", "full"), default=None,
                        help="output label (default: the round dir's parent)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--pipeline", default=None,
                        help="surrogate extract dir to report side by side "
                             "(default <round-dir>/extract)")
    parser.add_argument("--budget", type=int, default=5000,
                        help="NGSpice calls per device, every call counted")
    parser.add_argument("--starts", type=int, choices=(1, 4), default=4,
                        help="4: the hybrid/sketch bank starts, equal "
                             "schedule (arm dfols_style); 1: starts[0] with "
                             "the whole budget (arm dfols_style_1start)")
    parser.add_argument("--rho-beg", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0,
                        help="soft-restart direction seed")
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--diff-step", type=float, default=1e-3,
                        help="unused by this arm (hybrid_tr.setup needs it)")
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--devices", default=None,
                        help="comma-separated device tags (smoke tests)")
    parser.add_argument("--validation", default=str(VALIDATION),
                        help="mgh_validate.py's validation.json (must pass)")
    args = parser.parse_args()
    validation = check_validation(Path(args.validation))
    round_dir = Path(args.round_dir)
    label = args.set or round_dir.parent.name
    out_dir = Path(args.out) if args.out else RUNS / (
        f"{label}_{round_dir.name}_s{args.starts}_B{args.budget}")
    out_dir.mkdir(parents=True, exist_ok=True)
    options = {"round_dir": str(round_dir), "bank": args.bank,
               "out": str(out_dir), "model": "none", "schedule": "equal",
               **{k: getattr(args, k) for k in (
                   "budget", "starts", "rho_beg", "seed", "regularization",
                   "diff_step", "timeout_s")}}
    tags = [device_tag(d.dev_type, d.L_um, d.W_um) for d in PAPER_DEVICES]
    if args.devices:
        tags = [t for t in tags if t in args.devices.split(",")]
    todo = [t for t in tags if not (out_dir / f"ml_{t}.json").exists()]
    todo.sort(key=lambda t: not t.startswith("pmos"))
    t0 = time.time()
    if todo:
        with get_context("spawn").Pool(min(args.workers, len(todo))) as pool:
            for _ in pool.imap_unordered(
                    fit_device, [(t, options) for t in todo]):
                pass
    records = load_records(out_dir)
    partial = len(records) < len(PAPER_DEVICES)
    if partial and not args.devices:
        print(f"{len(records)}/{len(PAPER_DEVICES)} devices done; "
              "summary.json is written once all exist")
        return 0
    out = hybrid_tr.summarize(records, options, round_dir, args, len(todo),
                              time.time() - t0)
    out["method"] = METHOD
    out["partial_device_subset"] = sorted(records) if partial else None
    out["arm_note"] = ARM_NOTE
    out["mgh_validation"] = {k: validation[k] for k in ("pass", "V1", "V2")}
    out["dfols"] = dfols_summary(records)
    (out_dir / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in (
        "all_device_mean_rrms", "nmos_rrms", "pmos_rrms", "combined_rrms",
        "mean_ngspice_calls_per_device", "wall_s")}, indent=1))
    print("at budget", json.dumps({b: v["all_device_mean_rrms"]
                                   for b, v in out["at_budget"].items()}))
    print("dfols", json.dumps(out["dfols"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
