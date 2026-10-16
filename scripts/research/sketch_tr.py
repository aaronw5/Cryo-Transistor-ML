#!/usr/bin/env python3
"""Sketched-Jacobian hybrid trust-region fit (research node n1a).

``hybrid_tr.py`` with ``cryoml.sketch_tr.SketchTrustRegionRun`` in place of
``TrustRegionRun``: the same objective, corrected model
``r_c(z) = r_s(z) + [r_f(z_c) - r_s(z_c)] + [J_c - J_s(z_c)] (z - z_c)``,
bounded TRF subproblem, acceptance and radius rules, starts, schedule and
budget accounting, but the 43-call FD Jacobian ``J_f`` at each centre is
replaced by an estimate ``J_c``:

* at a new centre the prior is transported, ``J0 = J_s(z_new) + (J_c -
  J_s(z_c))`` (``--model none``: ``J0 = J_c``), plus a free secant update
  along the accepted step (``|s|_inf <= 0.15``);
* ``--k`` directional NGSpice probes (default 8: the linearized
  Gauss-Newton step, the top k-2 right singular vectors of ``J0`` and one
  Gaussian direction, orthonormalized on the coordinates off the box faces)
  in ONE altermod process, step ``--probe-step`` 1e-3 in the infinity norm,
  merged by ``J = J0 + (Y - J0 V) V^T``; same cliff guard as the full
  Jacobian (a probe that fails or whose paper RRMS exceeds 2x the centre's
  + 0.05 is retried once with the opposite step, else the direction is
  dropped);
* projected Broyden updates on rejected steps (free);
* a full batched FD Jacobian (``hybrid_tr.NgspiceResiduals.jacobian``) at
  the first centre, every ``--refresh-every`` accepted centres, and when the
  sketch disagrees with the prior by more than ``--mismatch-refresh``, after
  ``--poor-refresh`` poor steps at one centre, at an apparent stationary
  point, or at a radius collapse.

``--model none`` is the same code with ``r_s = 0``: a Broyden /
random-subspace Gauss-Newton trust region (DFO-LS style), the control that
isolates the surrogate's contribution.

Every NGSpice call counts toward ``--budget`` (probes and retries included)
and is scored into the device's best card and the trace.  Each device writes
``ml_<tag>.json`` in ``hybrid_tr``'s row format (every ``hybrid_tr`` record
key) under method key ``sketch_tr``, plus the sketch counters;
``summary.json`` comes from ``hybrid_tr.summarize``.
``CRYOML_TARGET_OVERRIDE`` applies through ``build_device_context``.

  PYTHONPATH=src .venv/bin/python scripts/research/sketch_tr.py \
      --round-dir out/cold_loop/full/round_1 \
      --bank data/processed/cold_loop/full_r1 --set full --model surrogate --k 8

``--check`` compares the directional probes with the full FD Jacobian at
the first start of each ``--devices`` device instead.
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
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.sketch_tr import (REFRESH_REASONS, SketchSettings,  # noqa: E402
                              SketchTrustRegionRun)
from cryoml.spice_pdk import simulate_pdk  # noqa: E402
from cryoml.trust_region import run_schedule  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402

RUNS = ROOT / "out" / "research" / "sketch_tr"
METHOD = "sketch_tr"
NGSPICE_S_PER_CALL = 0.031      # per-call cost used for call equivalents


def directional_jacobian(ngspice, z: np.ndarray, r: np.ndarray,
                         V: np.ndarray, probe_step: float):
    """Forward differences of ``ngspice`` along the columns of ``V``.

    Column ``i`` is probed at ``z + h_i v_i`` with ``h_i = probe_step /
    |v_i|_inf`` (sign flipped if that leaves the box), all probes in one
    altermod process (``NgspiceResiduals._simulate_many``: counted, scored
    into best/trace).  Cliff guard as ``NgspiceResiduals.jacobian``: a
    failed probe or one whose paper RRMS exceeds ``jump_factor`` x the
    centre's + ``jump_floor`` is retried once with ``-h``; if that is bad
    too, or the call cap is reached, the direction is dropped.  Returns
    ``(Y, kept)``."""
    z = np.asarray(z, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    k = V.shape[1]
    Y = np.zeros((len(r), k))
    kept = np.zeros(k, dtype=bool)
    if k == 0:
        return Y, kept
    steps = probe_step / np.max(np.abs(V), axis=0)
    for i in range(k):
        if np.any(np.abs(z + steps[i] * V[:, i]) > 1.0):
            steps[i] = -steps[i]
    n_run = int(min(k, max(0, ngspice.call_cap - ngspice.n_calls)))
    results = (ngspice._simulate_many([z + steps[i] * V[:, i]
                                       for i in range(n_run)])
               if n_run else [])
    centre = ngspice.centre_rrms.get(z.tobytes())
    if centre is None:
        centre = float(np.mean(np.sqrt(len(ngspice.layout) * np.add.reduceat(
            r ** 2, ngspice.curve_starts))))

    def bad(row):
        return (row[2] or row[1] > ngspice.jump_factor * centre
                + ngspice.jump_floor)

    suspect = [i for i, row in enumerate(results) if bad(row)]
    retry = [i for i in suspect
             if np.all(np.abs(z - steps[i] * V[:, i]) <= 1.0)]
    retry = retry[:int(min(len(retry),
                           max(0, ngspice.call_cap - ngspice.n_calls)))]
    if retry:
        ngspice.n_cliff_retries += len(retry)
        again = ngspice._simulate_many([z - steps[i] * V[:, i] for i in retry])
        for i, row in zip(retry, again):
            if not bad(row):
                steps[i], results[i] = -steps[i], row
    for i, row in enumerate(results):
        if not bad(row):
            Y[:, i] = (row[0] - r) / steps[i]
            kept[i] = True
    return Y, kept


class SketchBackend:
    """``hybrid_tr.NgspiceResiduals`` plus ``directional``; everything else
    is delegated."""

    def __init__(self, ngspice):
        self.ngspice = ngspice
        self.n_directional = self.n_probe_retries = 0

    def __getattr__(self, name):
        return getattr(self.ngspice, name)

    @property
    def n_calls(self) -> int:
        return self.ngspice.n_calls

    @property
    def jacobian_cost(self) -> int:
        return self.ngspice.jacobian_cost

    def residual(self, z):
        return self.ngspice.residual(z)

    def jacobian(self, z, r):
        return self.ngspice.jacobian(z, r)

    def directional(self, z, r, V, probe_step):
        self.n_directional += 1
        before = self.ngspice.n_cliff_retries
        out = directional_jacobian(self.ngspice, z, r, V, probe_step)
        self.n_probe_retries += self.ngspice.n_cliff_retries - before
        return out


def sketch_from(options: dict) -> SketchSettings:
    return SketchSettings(**{f.name: options[f.name]
                             for f in dataclasses.fields(SketchSettings)
                             if f.name in options})


def settings_from(options: dict):
    return dataclasses.replace(hybrid_tr.settings_from(options),
                               sub_max_nfev=options["sub_max_nfev"])


def fit_device(job) -> tuple[str, dict]:
    tag, options = job
    t0 = time.time()
    context, ngspice, starts, oracle, surrogate_for = hybrid_tr.setup(
        tag, options)
    backend = SketchBackend(ngspice)
    active = ngspice.active
    weight = hybrid_tr.prior_weight(options, len(active))
    settings = settings_from(options)
    sketch = sketch_from(options)
    stream = zlib.crc32(tag.encode())
    runs = [SketchTrustRegionRun(np.asarray(start, dtype=np.float64)[active],
                                 backend, surrogate_for(start), weight,
                                 settings, sketch, label=f"start{index}",
                                 stream=stream + index)
            for index, start in enumerate(starts)]
    run_schedule(runs, backend, options["budget"], options["schedule"])
    best = ngspice.best
    params = context.box.local_to_params(best["local"])
    sims = simulate_pdk(
        context.device.dev_type, context.device.L_um, context.device.W_um,
        context.curves, params=params, bin_index=context.bin_index)
    score = fps.score_real(context, sims)
    ev = ngspice.evaluator
    seconds = {key: sum(run.seconds[key] for run in runs)
               for key in ("subproblem", "residual", "jacobian", "probe",
                           "directions")}
    n_iterations = sum(run.n_accepted + run.n_rejected for run in runs)
    n_accepted = sum(run.n_accepted for run in runs)
    stats = [run.sketch_stats() for run in runs]
    reasons = {reason: sum(s["full_refresh_reasons"][reason] for s in stats)
               for reason in REFRESH_REASONS}
    mismatch = [m for s in stats for m in s["mismatch"]]
    record = {
        "method": "sketch_trust_region_model_management",
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
        # Sketch counters (all starts).
        "k": sketch.k, "refresh_every": sketch.refresh_every,
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
        "n_accepted": n_accepted,
        "calls_per_accepted_step": (ev.n_calls / n_accepted
                                    if n_accepted else None),
        "probe_calls": sum(s["probe_calls"] for s in stats),
        "full_calls": sum(s["full_calls"] for s in stats),
        "subproblem_call_equivalents": round(
            seconds["subproblem"] / NGSPICE_S_PER_CALL, 1),
        "overhead_call_equivalents": round(
            (seconds["subproblem"] + seconds["directions"])
            / NGSPICE_S_PER_CALL, 1),
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
          f"iters {n_iterations} acc {n_accepted} "
          f"calls/acc {record['calls_per_accepted_step'] or math.nan:.1f} "
          f"full {record['n_full_refresh']} {reasons} "
          f"sketch {record['n_sketch']} "
          f"sub/it {record['subproblem_s_per_iteration']:.3f}s "
          f"{record['runtime_s']:.0f}s", flush=True)
    return tag, record


def check_device(job) -> tuple[str, dict]:
    """Directional probes against the full FD Jacobian at the first start:
    ``|Y - J V|`` per direction (both forward differences, step 1e-3 in the
    infinity norm), for the sketch directions of ``J`` and for random
    orthonormal ones."""
    from cryoml.sketch_tr import choose_directions
    tag, options = job
    context, ngspice, starts, _oracle, _surrogate_for = hybrid_tr.setup(
        tag, {**options, "model": "none"})
    backend = SketchBackend(ngspice)
    active = ngspice.active
    z0 = np.asarray(starts[0], dtype=np.float64)[active]
    r0 = backend.residual(z0)
    J = backend.jacobian(z0, r0)
    free = np.abs(z0) < 1.0 - 2.0 * options["probe_step"]
    rng = np.random.default_rng(0)
    out = {"n_free": int(free.sum())}
    sketch_V = choose_directions(J, r0, z0, hybrid_tr.prior_weight(
        options, len(active)), options["delta0"], free, options["k"], rng)
    random_V = np.zeros((len(z0), options["k"]))
    q, _ = np.linalg.qr(rng.standard_normal((int(free.sum()), options["k"])))
    random_V[free] = q
    for name, V in (("sketch", sketch_V), ("random", random_V)):
        Y, kept = backend.directional(z0, r0, V, options["probe_step"])
        JV = J @ V
        out[name] = {
            "kept": int(kept.sum()),
            "relative_error": [float(np.linalg.norm(Y[:, i] - JV[:, i])
                                     / max(np.linalg.norm(JV[:, i]), 1e-30))
                               for i in range(V.shape[1])],
            "mismatch": float(np.linalg.norm(Y[:, kept] - JV[:, kept])
                              / max(np.linalg.norm(JV[:, kept]), 1e-30))}
    out["calls"] = ngspice.n_calls
    print(tag, json.dumps(out, indent=1), flush=True)
    return tag, out


def load_records(directory: Path) -> dict[str, dict]:
    rows = (json.loads(p.read_text()) for p in sorted(directory.glob("ml_*.json")))
    return {row["device"]: row["methods"][METHOD] for row in rows}


def sketch_summary(records: dict) -> dict:
    rows = records.values()
    calls_acc = [r["calls_per_accepted_step"] for r in rows
                 if r["calls_per_accepted_step"] is not None]
    mismatch = [m for r in rows for m in r["mismatch"]]
    # An all-dropped sketch records mismatch inf; interpolating quantiles
    # between two infs gives NaN, so quantiles use the finite values and
    # the infinite ones are counted.
    finite = [m for m in mismatch if math.isfinite(m)]
    return {
        "mean_calls_per_accepted_step": float(np.mean(calls_acc)) if calls_acc else None,
        "full_refresh_reasons": {reason: int(sum(
            r["full_refresh_reasons"][reason] for r in rows))
            for reason in REFRESH_REASONS},
        "mean_full_refresh_per_device": float(np.mean(
            [r["n_full_refresh"] for r in rows])),
        "mean_sketch_per_device": float(np.mean([r["n_sketch"] for r in rows])),
        "mean_secant_accept_per_device": float(np.mean(
            [r["n_secant_accept"] for r in rows])),
        "mean_broyden_reject_per_device": float(np.mean(
            [r["n_broyden_reject"] for r in rows])),
        "mismatch_quantiles": ({q: float(np.quantile(finite, float(q)))
                                for q in ("0.1", "0.5", "0.9")}
                               if finite else None),
        "n_mismatch_infinite": len(mismatch) - len(finite),
        "probe_call_fraction": float(sum(r["probe_calls"] for r in rows) / max(
            1, sum(r["n_ngspice_calls"] for r in rows))),
        "full_call_fraction": float(sum(r["full_calls"] for r in rows) / max(
            1, sum(r["n_ngspice_calls"] for r in rows))),
        "mean_overhead_call_equivalents": float(np.mean(
            [r["overhead_call_equivalents"] for r in rows])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-dir", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--set", choices=("seven", "full"), default=None,
                        help="output label (default: the round dir's parent)")
    parser.add_argument("--model", choices=("surrogate", "none"),
                        default="surrogate")
    parser.add_argument("--out", default=None)
    parser.add_argument("--pipeline", default=None,
                        help="surrogate extract dir to report side by side "
                             "(default <round-dir>/extract)")
    parser.add_argument("--budget", type=int, default=5000,
                        help="NGSpice calls per device, probes included")
    parser.add_argument("--n-starts", type=int, default=4)
    parser.add_argument("--schedule", choices=("equal", "halving"),
                        default="equal")
    parser.add_argument("--delta0", type=float, default=0.25)
    parser.add_argument("--diff-step", type=float, default=1e-3,
                        help="full-Jacobian forward-difference step")
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--sub-max-nfev", type=int, default=30,
                        help="TRF evaluations per subproblem, fixed at 30 for "
                             "every sketch arm: at the default 100 the smoke "
                             "measured 0.75-1.08 s per surrogate subproblem "
                             "under load (> 0.5 s rule); TrustRegionSettings "
                             "and hybrid_tr keep 100")
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
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--devices", default=None,
                        help="comma-separated device tags (smoke tests)")
    parser.add_argument("--check", action="store_true",
                        help="compare probes with the full Jacobian on "
                             "--devices and exit")
    args = parser.parse_args()
    round_dir = Path(args.round_dir)
    label = args.set or round_dir.parent.name
    out_dir = Path(args.out) if args.out else RUNS / (
        f"{label}_{round_dir.name}_{args.model}_k{args.k}")
    out_dir.mkdir(parents=True, exist_ok=True)
    options = {"round_dir": str(round_dir), "bank": args.bank,
               "out": str(out_dir), **{k: getattr(args, k) for k in (
                   "model", "budget", "n_starts", "schedule", "delta0",
                   "diff_step", "regularization", "timeout_s", "sub_max_nfev",
                   "k", "probe_step", "secant_max_step", "refresh_every",
                   "mismatch_refresh", "poor_refresh", "seed")}}
    tags = [device_tag(d.dev_type, d.L_um, d.W_um) for d in PAPER_DEVICES]
    if args.devices:
        tags = [t for t in tags if t in args.devices.split(",")]
    if args.check:
        jobs = [(t, options) for t in tags]
        with get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
            checks = dict(pool.imap_unordered(check_device, jobs))
        (out_dir / "checks.json").write_text(json.dumps(checks, indent=1) + "\n")
        return 0
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
    out = hybrid_tr.summarize(records, options, round_dir, args, len(todo),
                              time.time() - t0)
    out["method"] = METHOD
    out["sketch"] = sketch_summary(records)
    (out_dir / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in (
        "all_device_mean_rrms", "nmos_rrms", "pmos_rrms", "combined_rrms",
        "mean_ngspice_calls_per_device", "wall_s")}, indent=1))
    print("at budget", json.dumps({b: v["all_device_mean_rrms"]
                                   for b, v in out["at_budget"].items()}))
    print("sketch", json.dumps(out["sketch"]))
    print("pipeline", json.dumps({k: out["pipeline"].get(k) for k in (
        "all_device_mean_rrms", "nmos_rrms", "pmos_rrms", "combined_rrms")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
