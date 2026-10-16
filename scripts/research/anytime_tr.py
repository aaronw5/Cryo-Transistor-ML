#!/usr/bin/env python3
"""Budget-aware anytime trust-region fit: surrogate presearch starts, racing
by predicted reduction per NGSpice call, and surrogate-proposed restarts.

The inner loop is ``hybrid_tr``'s, unchanged: ``cryoml.trust_region.
TrustRegionRun`` with the first-order-consistent ``CorrectedModel`` (frozen
round surrogate + NGSpice value and exact batched FD Jacobian at each centre),
or plain trust-region Gauss-Newton on NGSpice with ``--model none``.  Same
objective, prior weight, Jacobian step and cliff guard, same NGSpice backend
(``hybrid_tr.setup``), same call counting (every simulation, Jacobian probes
included) and the same final fresh ``fps.score_real`` validation.  What
changes is which starts get the calls and in what order.

1. Presearch (``--presearch on``, surrogate only).  From each of the first
   ``--n-presearch`` (8) diverse bank starts (``fps.diverse_oracle_starts``
   over the pre-paid bank, n = 8 + ``--n-bank-reserve``), bounded TRF on the
   surrogate alone (``least_squares``, x_scale 1, ftol = xtol = gtol = 1e-8,
   ``--presearch-max-nfev`` 100) on the same objective as the paper pipeline
   (``fps.finite_difference_surrogate_extract``): per-curve normalized
   residuals plus sqrt(reg / n_active) z prior rows.  Difference from the
   pipeline: the Jacobian is the analytic forward-mode one
   (``torch.func.jacfwd`` via ``hybrid_tr.SurrogateResiduals``), not 2-point
   FD on the surrogate (diff_step 0.02, x_scale 'jac').  Zero NGSpice calls;
   the wall time is ``presearch_seconds`` (and / 0.031 s per call, its
   NGSpice-call equivalent).  All 8 endpoints are then validated in one
   batched altermod process (``NgspiceResiduals._simulate_many``, 8 counted
   calls, each scored into the best card and trace).
   Runs: the 2 best endpoints by NGSpice objective F at pairwise RMS
   distance >= 0.35 (pre-seeded with ``racing.seed_run``: no repeated start
   call, the first call of the run is its Jacobian) and the 2 best bank
   starts (bank order, among the presearch starts) at distance >= 0.35 from
   those, which start normally (1 call).  The other 6 validated endpoints
   (F order) and the 4 reserve bank starts are the restart pool.
   ``--presearch off`` (and ``--model none``): the first 4 diverse bank starts,
   exactly ``hybrid_tr``'s starts; the remaining bank starts are the pool.
2. Schedule.  ``equal``: ``trust_region.run_schedule(..., 'equal')`` as
   ``hybrid_tr`` (no restarts).  ``race``: ``racing.RaceScheduler``: one
   iteration per run in order, then always the run with the largest
   rho_bar * predicted reduction / cost of its next iteration; optimistic
   bound drops (model reduction in a 0.5 box, zero calls); when nothing is
   left to advance and >= jacobian_cost + 2 calls remain, a restart from the
   pool at distance >= 0.35 from every centre visited so far:
   surrogate arm, the next unused validated endpoint (pre-seeded, 0 calls),
   then a zero-call surrogate TRF from the next reserve bank start (its
   endpoint must pass the same distance test) validated with 1 call;
   ``--presearch off``, the next reserve bank start (a normal new run).
   The race's decisions do not depend on the budget except for affordability
   in the last iteration, so a B5000 trace's prefix is the B<5000 run up to
   its final ~45 calls.

Each device writes ``ml_<tag>.json`` in ``hybrid_tr``'s row shape, method key
``anytime_tr``, with every ``hybrid_tr`` record key plus ``presearch``
(endpoints, Spearman of surrogate F vs NGSpice F, best validated RRMS, bank
oracle RRMS, seconds), ``race`` (drops, restarts, calls per run from a call
ledger, share of calls on the run that produced the final card, rho_bar
history), ``rrms_at_calls`` (8, 50, 100) and the flags.  ``summary.json``
comes from ``hybrid_tr.summarize`` plus an ``anytime`` block.
``CRYOML_TARGET_OVERRIDE`` applies through ``build_device_context``.

  PYTHONPATH=src .venv/bin/python scripts/research/anytime_tr.py \
      --set full --round-dir out/cold_loop/full/round_1 \
      --bank data/processed/cold_loop/full_r1 --model surrogate \
      --presearch on --schedule race --restarts on --budget 5000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import foundation_parameter_study as fps  # noqa: E402
from hybrid_tr import (BUDGET_MARKS, best_at, prior_weight,  # noqa: E402
                       settings_from, setup, summarize)
from cryoml.data_io import TARGET_OVERRIDE_ENV  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.racing import (CallLedger, RaceScheduler, RaceSettings,  # noqa: E402
                           RestartPool, TaggedBackend, rms_distance,
                           seed_run, select_diverse)
from cryoml.spice_pdk import simulate_pdk  # noqa: E402
from cryoml.trust_region import TrustRegionRun, run_schedule  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402

RUNS = ROOT / "out" / "research" / "anytime_tr"
METHOD = "anytime_tr"
SECONDS_PER_CALL = 0.031        # one NGSpice call (RESEARCH_LOOP, 2026-10-16)
EARLY_MARKS = (8, 50, 100)


def surrogate_trf(surrogate, z0: np.ndarray, weight: float,
                  max_nfev: int) -> dict:
    """Bounded TRF on the surrogate objective alone (zero NGSpice calls)."""
    n = len(z0)
    prior = weight * np.eye(n)

    def fun(z):
        return np.concatenate([surrogate.value(z), weight * z])

    def jac(z):
        return np.vstack([surrogate.jacobian(z), prior])

    t0 = time.time()
    solution = least_squares(
        fun, np.clip(np.asarray(z0, dtype=np.float64), -1.0, 1.0), jac=jac,
        bounds=(-np.ones(n), np.ones(n)), method="trf", x_scale=1.0,
        ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=max_nfev)
    z = np.clip(solution.x, -1.0, 1.0)
    r = fun(z)
    return {"z": z, "surrogate_F": 0.5 * float(r @ r),
            "nfev": int(solution.nfev), "njev": int(solution.njev or 0),
            "status": int(solution.status),
            "seconds": round(time.time() - t0, 3)}


def relabel_best(ngspice, n_before: int, kind: str) -> None:
    """Calls made outside a run (validations) are 'jacobian_probe'/'trial'
    to ``NgspiceResiduals``; name them if they produced the best card."""
    if ngspice.best["calls"] > n_before:
        ngspice.best["kind"] = kind


def build_runs(context, ngspice, starts, oracle, surrogate_for,
               options: dict, ledger: CallLedger):
    """Initial runs, ordered restart proposals, the presearch record and the
    endpoint each pre-seeded run started from (module docstring, 1)."""
    active = ngspice.active
    weight = prior_weight(options, len(active))
    settings = settings_from(options)
    min_distance = options["min_distance"]
    starts_z = [np.asarray(s, dtype=np.float64)[active] for s in starts]
    seeded: dict[str, np.ndarray] = {}    # run label -> presearch endpoint

    def make_run(z, start_index, label):
        return TrustRegionRun(z, TaggedBackend(ngspice, ledger, label),
                              surrogate_for(starts[start_index]), weight,
                              settings, label=label)

    presearch = None
    proposals = []
    n_keep = options["n_keep_presearch"] + options["n_keep_bank"]
    if options["presearch"] == "on":
        # The batched validation is not budget-checked call by call.
        assert options["budget"] >= options["n_presearch"], options["budget"]
        endpoints = []
        t_pre = time.time()
        for index in range(options["n_presearch"]):
            result = surrogate_trf(surrogate_for(starts[index]),
                                   starts_z[index], weight,
                                   options["presearch_max_nfev"])
            result["start_index"] = index
            result["distance_to_start"] = rms_distance(result["z"],
                                                       starts_z[index])
            endpoints.append(result)
        presearch_seconds = time.time() - t_pre
        n0 = ngspice.n_calls
        validated = ngspice._simulate_many([e["z"] for e in endpoints])
        ledger.charge("presearch", n0)
        relabel_best(ngspice, n0, "presearch_endpoint")
        for e, (r, rrms, failed) in zip(endpoints, validated):
            e.update(r=r, ngspice_rrms=float(rrms), failed=bool(failed),
                     ngspice_F=0.5 * float(r @ r)
                     + 0.5 * float((weight * e["z"]) @ (weight * e["z"])))
            ngspice.centre_rrms[e["z"].tobytes()] = float(rrms)
        F = [e["ngspice_F"] for e in endpoints]
        keep_e = select_diverse([e["z"] for e in endpoints], F,
                                options["n_keep_presearch"], min_distance)
        keep_b = select_diverse(
            starts_z[:options["n_presearch"]],
            np.arange(options["n_presearch"], dtype=np.float64),
            options["n_keep_bank"], min_distance,
            exclude=[endpoints[k]["z"] for k in keep_e])
        runs = []
        for k in keep_e:
            e = endpoints[k]
            label = f"presearch{k}"
            runs.append(seed_run(make_run(e["z"], k, label), e["r"],
                                 e["ngspice_F"]))
            seeded[label] = e["z"]
        runs += [make_run(starts_z[b], b, f"bank{b}") for b in keep_b]

        def endpoint(k):
            def make(point):
                label = f"presearch{k}"
                seeded[label] = endpoints[k]["z"]
                return seed_run(make_run(point, k, label), endpoints[k]["r"],
                                endpoints[k]["ngspice_F"])
            return make

        restart_trf: list[dict] = []

        def reserve(index):
            cache: dict = {}

            def point():
                if "z" not in cache:
                    result = surrogate_trf(surrogate_for(starts[index]),
                                           starts_z[index], weight,
                                           options["presearch_max_nfev"])
                    cache["z"] = result.pop("z")
                    result.update(start_index=index, distance_to_start=(
                        rms_distance(cache["z"], starts_z[index])))
                    cache["record"] = result
                    restart_trf.append(result)
                return cache["z"]

            def make(z):
                label = f"reserve{index}"
                run = make_run(z, index, label)
                n_before = ngspice.n_calls
                r = run.expensive.residual(z)          # 1 counted call
                relabel_best(ngspice, n_before, "restart_validation")
                cache["record"]["ngspice_rrms"] = ngspice.centre_rrms.get(
                    z.tobytes())
                return seed_run(run, r)

            return point, make

        for k in np.argsort(F, kind="stable"):
            if int(k) not in keep_e:
                proposals.append({"label": f"presearch{int(k)}",
                                  "point": endpoints[int(k)]["z"],
                                  "make": endpoint(int(k))})
        for index in range(options["n_presearch"], len(starts)):
            point, make = reserve(index)
            proposals.append({"label": f"reserve{index}", "point": point,
                              "make": make})
        rrms_values = np.array([e["ngspice_rrms"] for e in endpoints])
        spearman = float(spearmanr([e["surrogate_F"] for e in endpoints], F)[0])
        presearch = {
            "endpoints": [{
                "start_index": e["start_index"],
                "surrogate_F": e["surrogate_F"], "ngspice_F": e["ngspice_F"],
                "ngspice_rrms": e["ngspice_rrms"], "failed": e["failed"],
                "distance_to_start": e["distance_to_start"],
                "nfev": e["nfev"], "njev": e["njev"],
                "trf_status": e["status"], "seconds": e["seconds"],
                "kept": k in keep_e} for k, e in enumerate(endpoints)],
            "spearman_surrogate_F_vs_ngspice_F": spearman,
            "best_validated_rrms": float(np.min(rrms_values)),
            "bank_oracle_rrms": float(oracle["paper_rrms"]),
            "presearch_seconds": round(presearch_seconds, 2),
            "presearch_call_equivalent": round(
                presearch_seconds / SECONDS_PER_CALL, 1),
            "n_validation_calls": len(endpoints),
            "kept_endpoints": [int(k) for k in keep_e],
            "kept_bank_starts": [int(b) for b in keep_b],
            "max_nfev": options["presearch_max_nfev"],
            "jacobian": ("analytic forward-mode (torch.func.jacfwd); the "
                         "pipeline uses 2-point FD on the surrogate, "
                         "diff_step 0.02, x_scale 'jac'"),
            "restart_trf": restart_trf,
        }
    else:
        runs = [make_run(starts_z[i], i, f"bank{i}") for i in range(n_keep)]
        for index in range(n_keep, len(starts)):
            proposals.append({
                "label": f"bank{index}", "point": starts_z[index],
                "make": (lambda i: lambda z: make_run(z, i, f"bank{i}"))(index)})

    return runs, proposals, presearch, seeded


def fit_device(job) -> tuple[str, dict]:
    tag, options = job
    t0 = time.time()
    n_setup = options["n_presearch"] + options["n_bank_reserve"]
    context, ngspice, starts, oracle, surrogate_for = setup(
        tag, {**options, "n_starts": n_setup})
    active = ngspice.active
    metadata = json.loads(
        (Path(options["round_dir"]) / "cache" / "metadata.json").read_text())
    # 43 parameters in the full set, but the box marks parameters whose
    # published value is zero as inactive (e.g. 38 active on nmos_L1_W1p6,
    # 40 on pmos_L0p35_W1p6); the search runs over the active ones, as
    # hybrid_tr does.
    assert context.box.n_parameters == metadata["n_parameters"], (
        tag, context.box.n_parameters, metadata["n_parameters"])
    if options["set"] == "full":
        assert metadata["n_parameters"] == 43, metadata["n_parameters"]
    assert len(active) == context.box.n_active > 0, (
        tag, len(active), context.box.n_active)
    ledger = CallLedger(ngspice)
    runs, proposals, presearch, seeded = build_runs(
        context, ngspice, starts, oracle, surrogate_for, options, ledger)
    min_distance = options["min_distance"]
    pool = RestartPool(proposals, min_distance)
    initial = [run.label for run in runs]
    scheduler = None
    if options["schedule"] == "equal":
        run_schedule(runs, ngspice, options["budget"], "equal")
    else:
        scheduler = RaceScheduler(
            runs, ngspice, options["budget"],
            RaceSettings(min_distance=min_distance),
            restart_fn=pool if options["restarts"] == "on" else None)
        scheduler.run()
        runs = scheduler.runs

    best = ngspice.best
    params = context.box.local_to_params(best["local"])
    sims = simulate_pdk(
        context.device.dev_type, context.device.L_um, context.device.W_um,
        context.curves, params=params, bin_index=context.bin_index)
    score = fps.score_real(context, sims)
    ev = ngspice.evaluator
    seconds = {key: sum(run.seconds[key] for run in runs)
               for key in ("subproblem", "residual", "jacobian")}
    n_iterations = sum(run.n_accepted + run.n_rejected for run in runs)

    calls_per_run = ledger.calls_by_label()
    winner = ledger.label_of_call(best["calls"])
    if winner == "presearch":
        z_best = np.asarray(best["local"], dtype=np.float64)[active]
        for label, z in seeded.items():
            if np.array_equal(z, z_best) and any(
                    run.label == label for run in runs):
                winner = label
    race = {"n_drops": 0, "n_restarts": 0, "n_drop_checks": 0,
            "calls_per_run": calls_per_run,
            "winning_run": winner,
            "share_on_winning_run": calls_per_run.get(winner, 0)
            / max(1, ev.n_calls),
            "initial_runs": initial,
            "unspent_calls": options["budget"] - ev.n_calls,
            "restart_pool": pool.history}
    if scheduler is not None:
        summary = scheduler.summary()
        race.update({k: summary[k] for k in (
            "n_drops", "n_restarts", "n_drop_checks", "stop_reason", "drops",
            "restarts", "dropped", "final_rho_bar")})
        race["rho_bar_history"] = summary["rho_bar_history"]
        race["race_iterations"] = summary["n_iterations"]
    record = {
        "method": "anytime_trust_region_racing",
        "model": options["model"],
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
        "budget_rrms": {str(b): best_at(ngspice.trace, b)
                        for b in BUDGET_MARKS},
        "rrms_at_calls": {str(b): best_at(ngspice.trace, b)
                          for b in EARLY_MARKS},
        "n_failed": ev.n_failed, "n_failed_points": ev.n_failed_points,
        "n_cliff_retries": ngspice.n_cliff_retries,
        "n_zero_jacobian_columns": ngspice.n_zero_columns,
        "n_iterations": n_iterations,
        "n_jacobians": ngspice.n_jacobians,
        "seconds": {k: round(v, 2) for k, v in seconds.items()},
        "subproblem_s_per_iteration": round(
            seconds["subproblem"] / max(1, n_iterations), 4),
        "ngspice_s_per_call": round(ev.sim_s / max(1, ev.n_calls), 4),
        "starts": [run.summary() for run in runs],
        "oracle_lhc": oracle, "n_starts": len(initial),
        "n_parameters": context.box.n_parameters,
        "n_active_parameters": len(active),
        "n_runs_total": len(runs),
        "schedule": options["schedule"], "diff_step": options["diff_step"],
        "regularization": options["regularization"],
        "delta0": options["delta0"],
        "flags": {"schedule": options["schedule"],
                  "presearch": options["presearch"],
                  "restarts": options["restarts"]},
        "presearch": presearch, "race": race,
        # Per-record DEV/TEST audit (smoke runs have no summary.json).
        "target_override": os.environ.get(TARGET_OVERRIDE_ENV) or None,
        "optimization_objective": (
            "uniform mean squared per-curve normalized point residual plus "
            "weak regularization (as the surrogate-FD pipeline)"),
        "runtime_s": round(time.time() - t0, 1),
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
    extra = ""
    if presearch is not None:
        extra = (f" presearch {presearch['presearch_seconds']:.0f}s best "
                 f"{presearch['best_validated_rrms']:.4f}")
    print(f"{tag}: rrms {record['rrms']:.4f} (in loop {best['rrms']:.4f}, "
          f"first call {ngspice.trace[0][1]:.4f}) calls {ev.n_calls} "
          f"iters {n_iterations} drops {race['n_drops']} restarts "
          f"{race['n_restarts']} winner {winner} "
          f"({race['share_on_winning_run']:.2f}){extra} "
          f"{record['runtime_s']:.0f}s", flush=True)
    return tag, record


def load_records(directory: Path) -> dict[str, dict]:
    rows = (json.loads(p.read_text()) for p in sorted(directory.glob("ml_*.json")))
    return {row["device"]: row["methods"][METHOD] for row in rows}


def anytime_summary(records: dict) -> dict:
    def mean(values):
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else None

    pre = [r["presearch"] for r in records.values() if r.get("presearch")]
    return {
        "rrms_at_calls_all_device_mean": {
            str(m): mean([r["rrms_at_calls"][str(m)] for r in records.values()])
            for m in EARLY_MARKS},
        "mean_presearch_seconds": mean([p["presearch_seconds"] for p in pre]),
        "mean_presearch_call_equivalent": mean(
            [p["presearch_call_equivalent"] for p in pre]),
        "mean_presearch_spearman": mean(
            [p["spearman_surrogate_F_vs_ngspice_F"] for p in pre]),
        "mean_best_validated_rrms": mean([p["best_validated_rrms"] for p in pre]),
        "mean_bank_oracle_rrms": mean([p["bank_oracle_rrms"] for p in pre]),
        "mean_drops": mean([r["race"]["n_drops"] for r in records.values()]),
        "mean_restarts": mean([r["race"]["n_restarts"] for r in records.values()]),
        "mean_share_on_winning_run": mean(
            [r["race"]["share_on_winning_run"] for r in records.values()]),
        "mean_unspent_calls": mean(
            [r["race"]["unspent_calls"] for r in records.values()]),
        "winning_run_kinds": {
            kind: sum(1 for r in records.values()
                      if str(r["race"]["winning_run"]).startswith(kind))
            for kind in ("presearch", "bank", "reserve")},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-dir", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--set", choices=("seven", "full"), default="full")
    parser.add_argument("--model", choices=("surrogate", "none"),
                        default="surrogate")
    parser.add_argument("--budget", type=int, default=5000,
                        help="NGSpice calls per device, Jacobian probes and "
                             "presearch validations included")
    parser.add_argument("--presearch", choices=("on", "off"), default=None,
                        help="default: on with the surrogate, off without")
    parser.add_argument("--schedule", choices=("equal", "race"),
                        default="race")
    parser.add_argument("--restarts", choices=("on", "off"), default=None,
                        help="default: on with --schedule race")
    parser.add_argument("--n-presearch", type=int, default=8)
    parser.add_argument("--n-keep-presearch", type=int, default=2)
    parser.add_argument("--n-keep-bank", type=int, default=2)
    parser.add_argument("--n-bank-reserve", type=int, default=4)
    parser.add_argument("--presearch-max-nfev", type=int, default=100)
    parser.add_argument("--min-distance", type=float, default=0.35)
    parser.add_argument("--delta0", type=float, default=0.25)
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
    args = parser.parse_args()
    if args.presearch is None:
        args.presearch = "on" if args.model == "surrogate" else "off"
    if args.restarts is None:
        args.restarts = "on" if args.schedule == "race" else "off"
    if args.presearch == "on" and args.model != "surrogate":
        parser.error("--presearch on needs --model surrogate")
    if args.restarts == "on" and args.schedule != "race":
        parser.error("--restarts on needs --schedule race")
    round_dir = Path(args.round_dir)
    out_dir = Path(args.out) if args.out else RUNS / (
        f"{args.set}_{round_dir.name}_{args.model}_{args.schedule}"
        f"_ps{args.presearch}_rs{args.restarts}_B{args.budget}")
    out_dir.mkdir(parents=True, exist_ok=True)
    options = {"round_dir": str(round_dir), "bank": args.bank,
               "out": str(out_dir), **{k: getattr(args, k) for k in (
                   "set", "model", "budget", "presearch", "schedule",
                   "restarts", "n_presearch", "n_keep_presearch",
                   "n_keep_bank", "n_bank_reserve", "presearch_max_nfev",
                   "min_distance", "delta0", "diff_step", "regularization",
                   "timeout_s")}}
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
    records = load_records(out_dir)
    if len(records) < len(PAPER_DEVICES):
        print(f"{len(records)}/{len(PAPER_DEVICES)} devices done; "
              "summary.json is written once all exist")
        return 0
    out = summarize(records, options, round_dir, args, len(todo),
                    time.time() - t0)
    out["method_key"] = METHOD
    out["anytime"] = anytime_summary(records)
    (out_dir / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in (
        "all_device_mean_rrms", "nmos_rrms", "pmos_rrms", "combined_rrms",
        "mean_ngspice_calls_per_device", "wall_s")}, indent=1))
    print("at budget", json.dumps({b: v["all_device_mean_rrms"]
                                   for b, v in out["at_budget"].items()}))
    print("anytime", json.dumps(out["anytime"]))
    print("pipeline", json.dumps({k: out["pipeline"].get(k) for k in (
        "all_device_mean_rrms", "nmos_rrms", "pmos_rrms", "combined_rrms")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
