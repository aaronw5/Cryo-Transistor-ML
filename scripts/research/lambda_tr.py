#!/usr/bin/env python3
"""Per-curve remainder-weighted hybrid trust region (research node n2b).

``sketch_tr.py`` (``--family sketch``) or ``hybrid_tr.py`` (``--family
hybrid``) with the corrected model of ``cryoml.lambda_tr``::

    r_lam(z) = r_c + J_c s + D_lam [r_s(z) - r_s(z_c) - J_s(z_c) s],

``D_lam`` diagonal over the data rows with ``lambda_j`` on the rows of
included metric curve ``j``.  ``lambda_j`` is a discounted least-squares fit
of the NGSpice second-order remainder ``e_f = r(z) - r_c - J_c s`` on the
surrogate remainder ``e_s`` at trial points the loop simulates anyway (no
extra NGSpice call), prior ``lambda = 1``.  Everything else is unchanged:
objective, starts (``fps.diverse_oracle_starts``, 4), equal schedule,
bounded TRF subproblem, acceptance and radius rules, sketch policy (sketch
family: k probes, secant, full FD on demand, stream ``crc32(tag) + start``),
full FD Jacobian at every centre (hybrid family), cliff guard, budget
accounting (every NGSpice call counts) and the final fresh scoring.

``--lambda-mode``: ``curve`` (main arm), ``scalar`` (one lambda), ``zero``
(D = 0: the ``--model none`` control, iterate for iterate), ``one`` (D = I:
the ``--model surrogate`` arm, iterate for iterate), ``cusum`` (1 until the
CUSUM of ``ln(|e_f - e_s|^2 / |e_f|^2)`` reaches ``--cusum-h``, then 0),
``radius`` (``min(1, Delta / --delta0)``).  Fixed before any DEV run:
``--forget`` 0.7, ``--prior-tau`` 1, ``--unprobed-tol`` 0.1, ``--cusum-h`` 2.

Each device writes ``ml_<tag>.json`` (``sketch_tr``'s record fields, method
key ``lambda_tr``) plus ``family``, ``lambda_mode``, ``k`` and ``seed`` for
``dev_compare.py`` ``expect`` and a ``lambda`` block (usable-trial fraction,
mean lambda at trials 1/3/10/30/100, CUSUM switch call per start, centre
consistency of every model build); the per-trial log is in
``starts[i]["lambda"]["trials"]``.  ``summary.json`` comes from
``hybrid_tr.summarize`` plus a ``lambda`` block (and ``sketch_tr``'s
``sketch`` block for the sketch family).  ``CRYOML_TARGET_OVERRIDE`` applies
through ``build_device_context``.

``--parity-ref DIR`` compares each device's trace, final local vector and
call count with the same device in DIR (a ``sketch_tr.py`` or
``hybrid_tr.py`` run) as lists, writes ``parity.json`` and exits 3 on any
difference.

``lambda_tr.py decide`` applies the pre-registered DEV acceptance rules
(``PRE_REGISTERED``: A1-A4, freeze order lam_curve_k8 -> lam_scalar_k8 ->
lam_cusum_k8, hyb_lam_curve against hybrid_surrogate / hybrid_none) to the
tier-a and tier-c registries through n2c's ``dev_stats`` (geometry-clustered
CI; ``--compare-a/--compare-c`` compare.json fallback with the i.i.d. CI
when ``dev_stats`` is absent) and writes
``out/research/dev/lambda_decision.json`` (no simulation).  The controls of
each family come from the P1 ``parity.json`` files (``parity_controls``); a
missing parity file, or a hard-check FAIL row on an evaluated arm or its
controls, withdraws that family's candidate (exit code 1 for a FAIL row).

  CRYOML_TARGET_OVERRIDE=out/research/sim_targets/set_0.npz \\
  PYTHONPATH=src .venv/bin/python scripts/research/lambda_tr.py \\
      --round-dir out/cold_loop/full/round_1 \\
      --bank data/processed/cold_loop/full_r1 --set full \\
      --family sketch --lambda-mode curve --k 8 --budget 500 \\
      --out out/research/dev/lam_curve_k8/B500/set_0
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
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
import sketch_tr  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.lambda_tr import (LAMBDA_MODES, TRIAL_MARKS,  # noqa: E402
                              LambdaSettings, LambdaSketchRun,
                              LambdaTrustRegionRun)
from cryoml.sketch_tr import REFRESH_REASONS, SketchSettings  # noqa: E402
from cryoml.spice_pdk import simulate_pdk  # noqa: E402
from cryoml.trust_region import run_schedule  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402

RUNS = ROOT / "out" / "research" / "lambda_tr"
METHOD = "lambda_tr"
SUB_MAX_NFEV = {"sketch": 30, "hybrid": 100}


def lambda_from(options: dict) -> LambdaSettings:
    return LambdaSettings(mode=options["lambda_mode"],
                          forget=options["forget"],
                          prior_tau=options["prior_tau"],
                          unprobed_tol=options["unprobed_tol"],
                          cusum_h=options["cusum_h"],
                          delta0_ref=options["delta0"])


def settings_from(options: dict):
    return dataclasses.replace(hybrid_tr.settings_from(options),
                               sub_max_nfev=options["sub_max_nfev"])


def lambda_record(runs) -> dict:
    """Per-device lambda summaries over the starts (the per-trial logs stay
    in ``starts[i]["lambda"]["trials"]``)."""
    stats = [run.lambda_stats() for run in runs]
    n_trials = sum(s["n_trials"] for s in stats)
    n_usable = sum(s["n_usable"] for s in stats)
    at, n_at = {}, {}
    for mark in TRIAL_MARKS:
        values = [s["lambda_at_trial"][str(mark)] for s in stats
                  if s["lambda_at_trial"][str(mark)] is not None]
        at[str(mark)] = float(np.mean(values)) if values else None
        n_at[str(mark)] = len(values)
    lam = [float(np.mean(t["lambda"])) for s in stats for t in s["trials"]]
    consistency = [s["consistency"] for s in stats]
    return {
        "n_trials": n_trials, "n_usable": n_usable,
        "usable_fraction": n_usable / n_trials if n_trials else None,
        # Trials whose NGSpice call failed (penalized fallback currents);
        # the usable-trial rule does not exclude them.
        "n_failed_trials": sum(s["n_failed_trials"] for s in stats),
        "n_failed_usable_trials": sum(s["n_failed_usable_trials"] for s in stats),
        "lambda_at_trial": at, "n_starts_at_trial": n_at,
        "mean_lambda_over_trials": float(np.mean(lam)) if lam else None,
        "cusum_switch_call": [s["cusum_switch_call"] for s in stats],
        "cusum_switched": [s["cusum_switched"] for s in stats],
        "n_lambda_changes": sum(s["n_lambda_changes"] for s in stats),
        "n_model_builds": sum(c["n_builds"] for c in consistency),
        "n_consistency_violations": sum(c["n_violations"] for c in consistency),
        "max_value_error_rel": max((c["max_value_error_rel"]
                                    for c in consistency), default=0.0),
        "max_jacobian_error_rel": max((c["max_jacobian_error_rel"]
                                       for c in consistency), default=0.0),
        "log_aligned": all(s["log_aligned"] for s in stats),
    }


def fit_device(job) -> tuple[str, dict]:
    tag, options = job
    t0 = time.time()
    context, ngspice, starts, oracle, surrogate_for = hybrid_tr.setup(
        tag, options)
    family = options["family"]
    active = ngspice.active
    weight = hybrid_tr.prior_weight(options, len(active))
    settings = settings_from(options)
    lam = lambda_from(options)
    n_rows = len(ngspice.measured)
    sketch = sketch_tr.sketch_from(options) if family == "sketch" else None
    if family == "sketch":
        backend = sketch_tr.SketchBackend(ngspice)
        stream = zlib.crc32(tag.encode())
        runs = [LambdaSketchRun(np.asarray(start, dtype=np.float64)[active],
                                backend, surrogate_for(start), weight,
                                settings, ngspice.curve_starts, n_rows, lam,
                                sketch, label=f"start{index}",
                                stream=stream + index)
                for index, start in enumerate(starts)]
    else:
        backend = ngspice
        runs = [LambdaTrustRegionRun(np.asarray(start, dtype=np.float64)[active],
                                     ngspice, surrogate_for(start), weight,
                                     settings, ngspice.curve_starts, n_rows,
                                     lam, label=f"start{index}")
                for index, start in enumerate(starts)]
    run_schedule(runs, backend, options["budget"], options["schedule"])
    best = ngspice.best
    params = context.box.local_to_params(best["local"])
    sims = simulate_pdk(
        context.device.dev_type, context.device.L_um, context.device.W_um,
        context.curves, params=params, bin_index=context.bin_index)
    score = fps.score_real(context, sims)
    ev = ngspice.evaluator
    keys = ("subproblem", "residual", "jacobian", "lambda") + (
        ("probe", "directions") if family == "sketch" else ())
    seconds = {key: sum(run.seconds[key] for run in runs) for key in keys}
    n_iterations = sum(run.n_accepted + run.n_rejected for run in runs)
    n_accepted = sum(run.n_accepted for run in runs)
    record = {
        "method": "lambda_weighted_trust_region_model_management",
        "model": "surrogate",
        "family": family, "lambda_mode": lam.mode,
        "k": sketch.k if sketch else None,
        "seed": sketch.seed if sketch else None,
        "lambda_settings": dataclasses.asdict(lam),
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
        "n_cliff_retries": ngspice.n_cliff_retries,
        "n_zero_jacobian_columns": ngspice.n_zero_columns,
        "n_iterations": n_iterations,
        "n_jacobians": ngspice.n_jacobians,
        "seconds": {k: round(v, 2) for k, v in seconds.items()},
        "subproblem_s_per_iteration": round(
            seconds["subproblem"] / max(1, n_iterations), 4),
        "ngspice_s_per_call": round(ev.sim_s / max(1, ev.n_calls), 4),
        "starts": [run.summary() for run in runs],
        "oracle_lhc": oracle, "n_starts": len(starts),
        "schedule": options["schedule"], "diff_step": options["diff_step"],
        "regularization": options["regularization"],
        "delta0": options["delta0"],
        "sub_max_nfev": settings.sub_max_nfev,
        "optimization_objective": (
            "uniform mean squared per-curve normalized point residual plus "
            "weak regularization (as the surrogate-FD pipeline)"),
        "runtime_s": round(time.time() - t0, 1),
        "n_accepted": n_accepted,
        "calls_per_accepted_step": (ev.n_calls / n_accepted
                                    if n_accepted else None),
        "lambda": {**lambda_record(runs),
                   "seconds": round(seconds["lambda"], 3),
                   "runtime_fraction": round(seconds["lambda"]
                                             / max(time.time() - t0, 1e-9), 5)},
    }
    if family == "sketch":
        stats = [run.sketch_stats() for run in runs]
        reasons = {reason: sum(s["full_refresh_reasons"][reason] for s in stats)
                   for reason in REFRESH_REASONS}
        mismatch = [m for s in stats for m in s["mismatch"]]
        record.update({
            "refresh_every": sketch.refresh_every,
            "sketch_settings": dataclasses.asdict(sketch),
            "n_full_refresh": int(sum(reasons.values())),
            "full_refresh_reasons": reasons,
            "n_sketch": sum(s["n_sketch"] for s in stats),
            "n_secant_accept": sum(s["n_secant_accept"] for s in stats),
            "n_broyden_reject": sum(s["n_broyden_reject"] for s in stats),
            "n_dropped_directions": sum(s["n_dropped_directions"] for s in stats),
            "n_probe_retries": backend.n_probe_retries,
            "mismatch": mismatch,
            "mismatch_median": float(np.median(mismatch)) if mismatch else None,
            "rho": [r for s in stats for r in s["rho"]],
            "probe_calls": sum(s["probe_calls"] for s in stats),
            "full_calls": sum(s["full_calls"] for s in stats),
            "subproblem_call_equivalents": round(
                seconds["subproblem"] / sketch_tr.NGSPICE_S_PER_CALL, 1),
            "overhead_call_equivalents": round(
                (seconds["subproblem"] + seconds["directions"])
                / sketch_tr.NGSPICE_S_PER_CALL, 1),
        })
    device = context.device
    row = {"device": tag, "dev_type": device.dev_type, "L_um": device.L_um,
           "W_um": device.W_um, "bin_index": context.bin_index,
           "n_active_parameters": context.box.n_active,
           "methods": {METHOD: record}}
    path = Path(options["out"]) / f"ml_{tag}.json"
    partial = path.with_suffix(".tmp")
    partial.write_text(json.dumps(row, indent=1) + "\n")
    partial.replace(path)
    lr = record["lambda"]
    print(f"{tag}: rrms {record['rrms']:.4f} (in loop {best['rrms']:.4f}, "
          f"first call {ngspice.trace[0][1]:.4f}) calls {ev.n_calls} "
          f"iters {n_iterations} acc {n_accepted} "
          f"lambda[{lam.mode}] at {lr['lambda_at_trial']} "
          f"usable {lr['usable_fraction'] if lr['usable_fraction'] is not None else math.nan:.2f} "
          f"changes {lr['n_lambda_changes']} "
          f"consistency {lr['n_consistency_violations']}/{lr['n_model_builds']} "
          f"cusum {lr['cusum_switch_call']} "
          f"sub/it {record['subproblem_s_per_iteration']:.3f}s "
          f"{record['runtime_s']:.0f}s", flush=True)
    return tag, record


def load_records(directory: Path) -> dict[str, dict]:
    rows = (json.loads(p.read_text()) for p in sorted(directory.glob("ml_*.json")))
    return {row["device"]: row["methods"][METHOD] for row in rows}


def lambda_summary(records: dict) -> dict:
    rows = [r["lambda"] for r in records.values()]
    n_trials = sum(r["n_trials"] for r in rows)
    switch = [c for r in rows for c in r["cusum_switch_call"]]
    switched = [c for c in switch if c is not None]
    at = {}
    for mark in TRIAL_MARKS:
        values = [r["lambda_at_trial"][str(mark)] for r in rows
                  if r["lambda_at_trial"][str(mark)] is not None]
        at[str(mark)] = float(np.mean(values)) if values else None
    return {
        "usable_fraction": (sum(r["n_usable"] for r in rows) / n_trials
                            if n_trials else None),
        "n_failed_usable_trials": sum(r["n_failed_usable_trials"] for r in rows),
        "lambda_at_trial_device_mean": at,
        "mean_lambda_over_trials_device_mean": float(np.mean(
            [r["mean_lambda_over_trials"] for r in rows
             if r["mean_lambda_over_trials"] is not None] or [math.nan])),
        "n_starts": len(switch), "n_cusum_switched": len(switched),
        "cusum_switch_call_median": (float(np.median(switched))
                                     if switched else None),
        "n_model_builds": sum(r["n_model_builds"] for r in rows),
        "n_consistency_violations": sum(r["n_consistency_violations"]
                                        for r in rows),
        "max_value_error_rel": max(r["max_value_error_rel"] for r in rows),
        "max_jacobian_error_rel": max(r["max_jacobian_error_rel"]
                                      for r in rows),
        "all_logs_aligned": all(r["log_aligned"] for r in rows),
        "lambda_seconds_fraction_of_runtime": (
            sum(r.get("seconds", 0.0) for r in rows)
            / max(sum(rec["runtime_s"] for rec in records.values()), 1e-9)),
    }


def parity(out_dir: Path, ref_dir: Path, ref_method: str) -> dict:
    """Trace / final local / call count of each device against ``ref_dir``,
    compared as lists (exact)."""
    mine = load_records(out_dir)
    ref = {}
    for path in sorted(ref_dir.glob("ml_*.json")):
        row = json.loads(path.read_text())
        ref[row["device"]] = row["methods"][ref_method]
    devices = {}
    for tag, record in sorted(mine.items()):
        other = ref.get(tag)
        if other is None:
            devices[tag] = {"present": False}
            continue
        devices[tag] = {
            "present": True,
            "trace_equal": record["trace"] == other["trace"],
            "local_equal": record["local"] == other["local"],
            "calls_equal": record["n_ngspice_calls"] == other["n_ngspice_calls"],
            "rrms": [record["rrms"], other["rrms"]],
            "trace_lengths": [len(record["trace"]), len(other["trace"])],
        }
    ok = bool(devices) and all(
        d.get("present") and d["trace_equal"] and d["local_equal"]
        and d["calls_equal"] for d in devices.values())
    return {"out": str(out_dir), "reference": str(ref_dir),
            "reference_method": ref_method, "n_devices": len(devices),
            "all_equal": ok, "devices": devices}


# ------------------------------------------------------------ DEV decision

# Pre-registered before any DEV run (node n2b spec); never tuned on DEV.
PRE_REGISTERED = {
    "hyperparameters": {"forget": 0.7, "prior_tau": 1.0, "unprobed_tol": 0.1,
                        "cusum_h": 2.0, "radius_delta0": 0.25},
    "rules": [
        {"id": "A1", "tier": "a", "budget": 500, "reference": "surrogate",
         "bound": 0.002, "strict": False,
         "text": "tier a, B500: arm - surrogate control, CI upper <= +0.002"},
        {"id": "A2", "tier": "a", "budget": 5000, "reference": "none",
         "bound": 0.0005, "strict": False,
         "text": "tier a, B5000: arm - NGSpice-only control, CI upper <= +0.0005"},
        {"id": "A3", "tier": "c", "budget": 500, "reference": "none",
         "bound": 0.0, "strict": True,
         "text": "tier c, B500: arm - NGSpice-only control, CI upper < 0"},
        {"id": "A4", "tier": "c", "budget": 5000, "reference": "none",
         "bound": 0.001, "strict": False,
         "text": "tier c, B5000: arm - NGSpice-only control, CI upper <= +0.001"},
    ],
    "families": {
        "sketch": {"order": ["lam_curve_k8", "lam_scalar_k8", "lam_cusum_k8"],
                   "controls": {"surrogate": "sketch_surrogate_k8",
                                "none": "sketch_none_k8"}},
        "hybrid": {"order": ["hyb_lam_curve"],
                   "controls": {"surrogate": "hybrid_surrogate",
                                "none": "hybrid_none"}},
    },
    "metric": ("tier a: raw strict paper RRMS (frozen inclusion); tier c: "
               "excess over the per-target floor (paired differences equal "
               "the raw ones)"),
    "ci": ("n2c's geometry-clustered bootstrap (dev_stats) when available, "
           "otherwise dev_compare's i.i.d. bootstrap (10k resamples over "
           "targets, seed 0); both reported"),
    "freeze_rule": ("lam_curve_k8 is the frozen sketch-family candidate if it "
                    "passes A1-A4; otherwise lam_scalar_k8, then lam_cusum_k8, "
                    "the first that passes; hyb_lam_curve is evaluated with "
                    "the same rules against hybrid_surrogate / hybrid_none; "
                    "if none passes the result is negative and no TEST run "
                    "follows; no per-device selection; no tuning of f, tau, "
                    "h on DEV"),
    "controls_resolution": ("per family, from the P1 parity.json files (tier a "
                            "set_0 B500, 18 devices): lambda 0 and lambda 1 "
                            "both equal to the iteration-1 control directories "
                            "(trace, final local vector, calls) -> those "
                            "controls; either different -> the family's lambda "
                            "0 / 1 arms (lam_zero_k8 / lam_one_k8, hyb_lam_zero "
                            "/ hyb_lam_one, P1 fallback lines) are the controls; "
                            "a parity file missing -> no candidate is frozen"),
    "validity": ("a FAIL row of dev_compare's hard checks (summary "
                 "target_override, bank-oracle identity, expect keys, "
                 "n_ngspice_calls <= budget, fresh rrms == best_at(trace)) on "
                 "an arm evaluated for a family or on its controls -> no "
                 "candidate is frozen for that family; exit code 1"),
}

# P1 parity runs per family and control role (role -> (lambda arm label, run
# directory)); see PRE_REGISTERED["controls_resolution"].
PARITY_RUNS = {
    "sketch": {"none": ("lam_zero_k8", "out/research/dev/lam_zero_k8/B500/set_0"),
               "surrogate": ("lam_one_k8", "out/research/dev/lam_one_k8/B500/set_0")},
    "hybrid": {"none": ("hyb_lam_zero", "out/research/dev/hyb_lam_zero/B500/set_0"),
               "surrogate": ("hyb_lam_one",
                             "out/research/dev/hyb_lam_one/B500/set_0")},
}


def parity_controls(family: str) -> dict:
    """Controls of ``family`` resolved from its P1 ``parity.json`` files."""
    controls = PRE_REGISTERED["families"][family]["controls"]
    rows = {}
    for role, (label, directory) in PARITY_RUNS[family].items():
        path = ROOT / directory / "parity.json"
        expected = (ROOT / "out/research/dev" / controls[role] / "B500"
                    / "set_0").resolve()
        row = {"lambda_arm": label, "iteration1_control": controls[role],
               "parity_file": (str(path.relative_to(ROOT))
                               if path.is_relative_to(ROOT) else str(path)),
               "equal": None}
        if path.exists():
            check = json.loads(path.read_text())
            reference = Path(check.get("reference", ""))
            reference = (reference if reference.is_absolute()
                         else ROOT / reference).resolve()
            row.update({"reference": check.get("reference"),
                        "n_devices": check.get("n_devices"),
                        "all_equal": check.get("all_equal")})
            row["equal"] = bool(check.get("all_equal") is True
                                and check.get("n_devices") == len(PAPER_DEVICES)
                                and reference == expected)
        rows[role] = row
    if all(r["equal"] is True for r in rows.values()):
        return {"status": "parity exact: iteration-1 controls",
                "controls": dict(controls), "freeze_allowed": True,
                "parity": rows}
    if any(r["equal"] is False for r in rows.values()):
        return {"status": "parity different: lambda 0/1 arms are the controls",
                "controls": {role: r["lambda_arm"] for role, r in rows.items()},
                "freeze_allowed": True, "parity": rows}
    return {"status": "parity unverified (parity.json missing): rules evaluated "
                      "against the iteration-1 controls, no candidate frozen",
            "controls": dict(controls), "freeze_allowed": False, "parity": rows}


def check_label(name: str) -> str:
    """Arm label (seed suffix removed) of a dev_compare check row name."""
    return re.sub(r"__seed\d+$", "", name.split(" ", 1)[0])


def geometry_cluster_ci(diff: dict, n_resamples: int = 10_000,
                        seed: int = 0) -> list[float]:
    """Fallback geometry-clustered CI (no seed level): resample the device
    tags; a drawn tag contributes its set_0 and set_1 targets."""
    tags = sorted({k.split("/", 1)[1] for k in diff})
    groups = [[diff[k] for k in diff if k.split("/", 1)[1] == t] for t in tags]
    sums = np.array([sum(g) for g in groups])
    counts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(tags), (n_resamples, len(tags)))
    means = sums[pick].sum(1) / counts[pick].sum(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


class StatsSource:
    """Paired differences through n2c's ``dev_stats`` (registries read by
    ``dev_compare``'s checks; per-target value = mean over the arm's seeds;
    failed targets at their penalized value; tier c endpoint = RRMS - floor;
    geometry-clustered 95% CI (18 geometries, seeds resampled within a
    target) and the i.i.d. 95% CI of ``dev_compare.paired``)."""

    ci_used = "geometry_clustered (dev_stats.compare ci95_clustered)"

    def __init__(self, registry_a: str, registry_c: str, floors: Path):
        import dev_compare
        import dev_stats
        self.dev_stats = dev_stats
        self.checks = dev_compare.Checks(True)
        self.tiers = {}
        self.floors_meta = None
        if (ROOT / registry_a).exists() or Path(registry_a).exists():
            self.tiers["a"] = dev_stats.Tier("a", registry_a, self.checks, True, None)
        if (ROOT / registry_c).exists() or Path(registry_c).exists():
            table = None
            path = floors if floors.is_absolute() else ROOT / floors
            if path.exists():
                table, self.floors_meta = dev_stats.load_floors(path)
            self.tiers["c"] = dev_stats.Tier("c", registry_c, self.checks, True,
                                             table)

    @property
    def check_rows(self) -> list[dict]:
        return self.checks.rows

    def diff(self, arm: str, reference: str, tier: str, budget: int) -> dict:
        family = {"name": "n2b_decision", "alpha": 0.05,
                  "secondary_log10_tier_a": False,
                  "tests": [{"id": "n2b", "a": arm, "b": reference,
                             "tier": tier, "budget": budget}]}
        row = self.dev_stats.run_family(family, self.tiers)["tests"][0]
        if "primary" not in row:
            return {"status": f"not computed: {row.get('missing')}"}
        p = row["primary"]
        return {"status": "evaluated", "n": p["n_targets"],
                "mean_diff": p["mean_diff"], "ci": p["ci95_clustered"],
                "ci_geometry_clustered": p["ci95_clustered"],
                "ci_iid": p["ci95_iid"],
                "wilcoxon_p_geometry": p["wilcoxon_p_geometry"],
                "seeds_arm": row["seeds_a"], "seeds_reference": row["seeds_b"],
                "failed_targets": [row["failed_targets_a"], row["failed_targets_b"]],
                "endpoint": row.get("endpoint"),
                "mean_endpoint": [p["mean_a"], p["mean_b"]]}


class CompareSource:
    """Fallback without ``dev_stats``: per-target values of
    ``dev_compare.py``'s ``compare.json`` (seed 0 only); the decision uses
    the i.i.d. bootstrap CI (``dev_compare.bootstrap_ci``)."""

    ci_used = "iid (dev_compare.bootstrap_ci)"

    def __init__(self, compare_a: str, compare_c: str):
        self.compare = {}
        for tier, path in (("a", compare_a), ("c", compare_c)):
            p = Path(path) if path else None
            self.compare[tier] = json.loads(p.read_text()) if p and p.exists() else {}
        self.check_rows = [row for data in self.compare.values()
                           for row in data.get("checks", [])]

    def diff(self, arm: str, reference: str, tier: str, budget: int) -> dict:
        import dev_compare
        arms = self.compare.get(tier, {}).get("arms", {})
        if arm not in arms or reference not in arms:
            return {"status": "missing arm"}
        a = arms[arm]["budgets"].get(str(budget), {}).get("values")
        b = arms[reference]["budgets"].get(str(budget), {}).get("values")
        if not a or not b:
            return {"status": "missing budget"}
        common = sorted(set(a) & set(b))
        diff = {k: float(a[k]) - float(b[k]) for k in common}
        if len(common) != 2 * dev_compare.N_DEVICES or not all(
                math.isfinite(v) for v in diff.values()):
            return {"status": f"incomplete: {len(common)} paired targets"}
        iid = dev_compare.bootstrap_ci(np.array(list(diff.values())))
        return {"status": "evaluated", "n": len(common),
                "mean_diff": float(np.mean(list(diff.values()))), "ci": iid,
                "ci_iid": iid, "ci_geometry_clustered_no_seeds":
                geometry_cluster_ci(diff)}


def decide_rule(rule: dict, arm: str, reference: str, source) -> dict:
    out = {"rule": rule["id"], "arm": arm, "reference": reference,
           "tier": rule["tier"], "budget": rule["budget"],
           "bound": rule["bound"], "strict_inequality": rule["strict"]}
    result = source.diff(arm, reference, rule["tier"], rule["budget"])
    out.update(result)
    upper = result.get("ci", [math.nan, math.nan])[1]
    out["pass"] = bool(result["status"] == "evaluated" and math.isfinite(upper)
                       and (upper < rule["bound"] if rule["strict"]
                            else upper <= rule["bound"]))
    return out


def decide(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="lambda_tr.py decide")
    parser.add_argument("--registry-a", default="out/research/dev/arms_iter2.json")
    parser.add_argument("--registry-c", default="out/research/dev_sky130c/arms.json")
    parser.add_argument("--floors",
                        default="out/research/sim_targets_sky130c/floors.json")
    parser.add_argument("--compare-a", default=None,
                        help="fallback without dev_stats: tier-a compare.json")
    parser.add_argument("--compare-c", default=None,
                        help="fallback without dev_stats: tier-c compare.json")
    parser.add_argument("--out", default="out/research/dev/lambda_decision.json")
    args = parser.parse_args(argv)
    try:
        source = StatsSource(args.registry_a, args.registry_c, Path(args.floors))
    except ImportError:
        source = CompareSource(args.compare_a, args.compare_c)
    families = {}
    for family, spec in PRE_REGISTERED["families"].items():
        resolution = parity_controls(family)
        controls = resolution["controls"]
        evaluated, candidate = [], None
        for arm in spec["order"]:
            rows = [decide_rule(rule, arm, controls[rule["reference"]], source)
                    for rule in PRE_REGISTERED["rules"]]
            passed = all(r["pass"] for r in rows)
            evaluated.append({"arm": arm, "pass": passed, "rules": rows})
            if passed:
                candidate = arm
                break
        families[family] = {"candidate": candidate, "evaluated": evaluated,
                            "controls": controls,
                            "controls_resolution": resolution}
    sketch_controls = families["sketch"]["controls"]
    ablations = {arm: [decide_rule(rule, arm, sketch_controls[rule["reference"]],
                                   source)
                       for rule in PRE_REGISTERED["rules"]]
                 for arm in ("lam_radius_k8", "lam_scalar_k8", "lam_cusum_k8")
                 if arm != families["sketch"]["candidate"]}
    # Validity: hard-check FAIL rows on the arms a family evaluated or on its
    # controls withdraw that family's candidate; a missing parity file too.
    check_rows = getattr(source, "check_rows", None)
    fail_rows = [r for r in (check_rows or []) if r.get("level") == "FAIL"]
    blocked = False
    for family, v in families.items():
        labels = {e["arm"] for e in v["evaluated"]} | set(v["controls"].values())
        v["failed_checks"] = [r for r in fail_rows
                              if check_label(r["check"]) in labels]
        v["candidate_before_validity"] = v["candidate"]
        if v["failed_checks"] or not v["controls_resolution"]["freeze_allowed"]:
            blocked = blocked or bool(v["failed_checks"])
            v["candidate"] = None
    candidates = {f: v["candidate"] for f, v in families.items()}
    complete = all(r["status"] == "evaluated" for v in families.values()
                   for e in v["evaluated"] for r in e["rules"])
    status = ("decided" if complete else
              "decided on incomplete data (a rule without data fails)")
    if blocked:
        status += "; hard checks failed (candidate withdrawn)"
    if not all(v["controls_resolution"]["freeze_allowed"]
               for v in families.values()):
        status += "; P1 parity unverified (candidate withdrawn)"
    out = {"node": "n2b", "status": status,
           "result": ("positive" if any(candidates.values()) else "negative"),
           "frozen_candidates": candidates,
           "test_run": ("once per frozen candidate" if any(candidates.values())
                        else "none (negative result)"),
           "ci_used": source.ci_used,
           "checks": {"available": check_rows is not None,
                      "n_rows": len(check_rows or []),
                      "n_fail": len(fail_rows), "fail_rows": fail_rows,
                      # WARN rows do not withdraw a candidate; listed so the
                      # batched-probe score gaps (dev_compare
                      # BATCHED_PROBE_KINDS) stay visible.
                      "warn_rows": [r for r in (check_rows or [])
                                    if r.get("level") == "WARN"
                                    and "batched-probe" in r["check"]]},
           "inputs": {"registry_a": args.registry_a,
                      "registry_c": args.registry_c, "floors": args.floors,
                      "floors_meta": getattr(source, "floors_meta", None),
                      "compare_a": args.compare_a, "compare_c": args.compare_c},
           "pre_registered": PRE_REGISTERED, "families": families,
           "ablations_reported_not_selected": ablations}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"result": out["result"], "status": status,
                      "candidates": candidates, "ci_used": source.ci_used,
                      "controls": {f: [v["controls"],
                                       v["controls_resolution"]["status"]]
                                   for f, v in families.items()},
                      "n_fail_checks": len(fail_rows),
                      "rules": {f: [(e["arm"], [(r["rule"], r["pass"],
                                                 r.get("mean_diff"), r.get("ci"),
                                                 r["status"])
                                                for r in e["rules"]])
                                    for e in v["evaluated"]]
                                for f, v in families.items()}}, indent=1))
    return 1 if blocked else 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "decide":
        return decide(sys.argv[2:])
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-dir", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--set", choices=("seven", "full"), default=None,
                        help="output label (default: the round dir's parent)")
    parser.add_argument("--family", choices=("sketch", "hybrid"),
                        default="sketch")
    parser.add_argument("--lambda-mode", choices=LAMBDA_MODES, required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--pipeline", default=None,
                        help="surrogate extract dir reported side by side by "
                             "hybrid_tr.summarize (default <round-dir>/extract)")
    parser.add_argument("--budget", type=int, default=5000,
                        help="NGSpice calls per device, probes included")
    parser.add_argument("--n-starts", type=int, default=4)
    parser.add_argument("--schedule", choices=("equal", "halving"),
                        default="equal")
    parser.add_argument("--delta0", type=float, default=0.25,
                        help="initial radius; also the radius-mode reference")
    parser.add_argument("--diff-step", type=float, default=1e-3)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--sub-max-nfev", type=int, default=None,
                        help="default 30 (sketch, as sketch_tr.py) or 100 "
                             "(hybrid, as hybrid_tr.py)")
    defaults = SketchSettings()
    parser.add_argument("--k", type=int, default=defaults.k)
    parser.add_argument("--probe-step", type=float, default=defaults.probe_step)
    parser.add_argument("--secant-max-step", type=float,
                        default=defaults.secant_max_step)
    parser.add_argument("--refresh-every", type=int,
                        default=defaults.refresh_every)
    parser.add_argument("--mismatch-refresh", type=float,
                        default=defaults.mismatch_refresh)
    parser.add_argument("--poor-refresh", type=int, default=defaults.poor_refresh)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    lam = LambdaSettings()
    parser.add_argument("--forget", type=float, default=lam.forget)
    parser.add_argument("--prior-tau", type=float, default=lam.prior_tau)
    parser.add_argument("--unprobed-tol", type=float, default=lam.unprobed_tol)
    parser.add_argument("--cusum-h", type=float, default=lam.cusum_h)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--devices", default=None,
                        help="comma-separated device tags (smoke tests)")
    parser.add_argument("--parity-ref", default=None,
                        help="run dir to compare traces with (exit 3 if any "
                             "device differs)")
    parser.add_argument("--parity-ref-method", default=None,
                        help="method key in --parity-ref (default sketch_tr "
                             "or hybrid_tr by --family)")
    args = parser.parse_args()
    if args.sub_max_nfev is None:
        args.sub_max_nfev = SUB_MAX_NFEV[args.family]
    round_dir = Path(args.round_dir)
    label = args.set or round_dir.parent.name
    k_label = f"_k{args.k}" if args.family == "sketch" else ""
    out_dir = Path(args.out) if args.out else RUNS / (
        f"{label}_{round_dir.name}_{args.family}_{args.lambda_mode}{k_label}")
    out_dir.mkdir(parents=True, exist_ok=True)
    options = {"round_dir": str(round_dir), "bank": args.bank,
               "out": str(out_dir), "model": "surrogate",
               **{k: getattr(args, k) for k in (
                   "family", "lambda_mode", "budget", "n_starts", "schedule",
                   "delta0", "diff_step", "regularization", "timeout_s",
                   "sub_max_nfev", "k", "probe_step", "secant_max_step",
                   "refresh_every", "mismatch_refresh", "poor_refresh", "seed",
                   "forget", "prior_tau", "unprobed_tol", "cusum_h")}}
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
    rc = 0
    if args.parity_ref:
        method = args.parity_ref_method or (
            "sketch_tr" if args.family == "sketch" else "hybrid_tr")
        check = parity(out_dir, Path(args.parity_ref), method)
        (out_dir / "parity.json").write_text(json.dumps(check, indent=1) + "\n")
        print("parity", "EQUAL" if check["all_equal"] else "DIFFERENT",
              json.dumps({t: {k: d.get(k) for k in ("trace_equal", "local_equal",
                                                   "calls_equal")}
                          for t, d in check["devices"].items()}), flush=True)
        rc = 0 if check["all_equal"] else 3
    records = load_records(out_dir)
    if len(records) < len(PAPER_DEVICES):
        print(f"{len(records)}/{len(PAPER_DEVICES)} devices done; "
              "summary.json is written once all exist")
        return rc
    out = hybrid_tr.summarize(records, options, round_dir, args, len(todo),
                              time.time() - t0)
    out["method"] = METHOD
    out["family"] = args.family
    out["lambda_mode"] = args.lambda_mode
    out["lambda"] = lambda_summary(records)
    if args.family == "sketch":
        out["sketch"] = sketch_tr.sketch_summary(records)
    (out_dir / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in (
        "all_device_mean_rrms", "nmos_rrms", "pmos_rrms", "combined_rrms",
        "mean_ngspice_calls_per_device", "wall_s")}, indent=1))
    print("at budget", json.dumps({b: v["all_device_mean_rrms"]
                                   for b, v in out["at_budget"].items()}))
    print("lambda", json.dumps(out["lambda"]))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
