#!/usr/bin/env python3
"""Hybrid trust-region fit: the neural surrogate proposes, NGSpice decides.

Trust-region model management (Alexandrov et al. 1998,
``cryoml.trust_region``) on the pipeline's own least-squares objective: the
frozen forward surrogate of a cold-loop round supplies the nonlinear model,
and NGSpice supplies the value and an exact batched finite-difference
Jacobian at every trust-region centre, through the first-order-consistent
correction

    r_c(z) = r_s(z) + [r_f(z_c) - r_s(z_c)] + [J_f(z_c) - J_s(z_c)] (z - z_c).

Objective (as ``fps.finite_difference_surrogate_extract``): per-curve
normalized point residuals on the frozen-inclusion paper curves, scale
``denominator * sqrt(n_curves * n_points)``, plus ``sqrt(reg / n_active) z``
prior rows (reg 1e-4); its sum of squares is mean(curve RRMS^2).  Failed
NGSpice points are penalized at curve RRMS 10 (``swap_search``).

* r_f: one ``simulate_pdk`` call per trial point; J_f: forward differences in
  local coordinates (step ``--diff-step`` 1e-3, inward at the upper box face)
  from one ``simulate_pdk_multi`` altermod process (bit-identical to separate
  calls).  Not the pipeline's 1e-2: in the x100 round-1 box a 1e-2 step moves
  u0 by ~5%, and near u0 = 1 (BSIM4 divides u0 > 1 by 1e4) that probe crosses
  the cliff; on nmos_L1_W1p6 the 1e-2 Jacobian was 33x off along a random
  direction, the 1e-3 one 0.4%.  Cliff guard: a probe that fails or whose
  paper RRMS exceeds 2x the centre's + 0.05 is retried with the opposite
  step, else its column is zero.  Every simulated card, Jacobian probes
  included, is scored and can become the device's best card.
* r_s, J_s: surrogate current on the same points; J_s by forward-mode
  autograd (``torch.func.jacfwd``, float32, CPU).
* ``--model none``: r_s = 0, so r_c is the Gauss-Newton linearization of
  NGSpice: the same loop and budget with no ML (the control).
* Starts: ``fps.diverse_oracle_starts`` (n = 4).  ``--schedule equal`` (default)
  runs them one after another, each with an equal share of the calls still
  unspent (a run that stops early leaves its rest to later runs); leftover
  calls go to the best run still active.  ``halving`` is successive halving.
* Budget: every NGSpice call counts (Jacobian probes included) and the search
  stops at ``--budget``; one fresh simulation of the final card is the
  pipeline's scoring (``fps.score_real``), recorded separately.

Each device writes ``ml_<tag>.json`` in the pipeline record format, method
key ``hybrid_tr`` (``polish_cards.py --cards <out> --method hybrid_tr``), with
a trace of the best in-loop frozen-inclusion paper RRMS against cumulative
NGSpice calls.  ``summary.json`` (all 18 devices) adds the all-device,
family and combined means, and the same means at fixed call budgets from the
traces.  ``CRYOML_TARGET_OVERRIDE`` applies through ``build_device_context``.

  PYTHONPATH=src .venv/bin/python scripts/research/hybrid_tr.py \
      --round-dir out/cold_loop/full/round_1 \
      --bank data/processed/cold_loop/full_r1 --set full --model surrogate

``--check`` runs the consistency checks on ``--devices`` instead.
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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import foundation_parameter_study as fps  # noqa: E402
import swap_search  # noqa: E402
from cryoml.card_eval import aggregate  # noqa: E402
from cryoml.data_io import TARGET_OVERRIDE_ENV, target_override  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.foundation import I_REF, FoundationSurrogate  # noqa: E402
from cryoml.spice_batch import simulate_pdk_multi  # noqa: E402
from cryoml.spice_pdk import simulate_pdk  # noqa: E402
from cryoml.trust_region import (CorrectedModel, TrustRegionRun,  # noqa: E402
                                 TrustRegionSettings, run_schedule)
from cryoml.utils import device_tag  # noqa: E402

RUNS = ROOT / "out" / "research" / "hybrid_tr"
BUDGET_MARKS = (500, 1000, 2000, 5000)


class NgspiceResiduals:
    """r_f(z) and J_f(z) on the pipeline objective's points, with call
    counting and the best card seen (all calls, probes included)."""

    def __init__(self, context, evaluator, base_local: np.ndarray,
                 diff_step: float, call_cap: float = math.inf,
                 jump_factor: float = 2.0, jump_floor: float = 0.05):
        self.context = context
        self.evaluator = evaluator          # swap_search.NgspiceEvaluator
        self.layout = fps.curve_objective_layout(context)
        self.active = np.flatnonzero(context.box.active)
        self.base = np.asarray(base_local, dtype=np.float64).copy()
        self.index = np.concatenate([np.arange(e["start"], e["stop"])
                                     for e in self.layout])
        n_curves = len(self.layout)
        self.measured = np.concatenate([np.asarray(e["measured"], np.float64)
                                        for e in self.layout])
        self.scale = np.concatenate([
            np.full(len(e["measured"]), e["denominator"]
                    * math.sqrt(n_curves * max(len(e["measured"]), 1)))
            for e in self.layout])
        self.curve_starts = np.cumsum(
            [0] + [len(e["measured"]) for e in self.layout][:-1])
        self.diff_step = float(diff_step)
        self.call_cap = call_cap
        self.jump_factor = float(jump_factor)
        self.jump_floor = float(jump_floor)
        self.jacobian_cost = len(self.active)
        self.n_jacobians = self.n_cliff_retries = self.n_zero_columns = 0
        self.centre_rrms: dict[bytes, float] = {}
        self.best = {"rrms": math.inf, "local": None, "calls": 0,
                     "kind": None}
        self.trace: list[list] = []

    @property
    def n_calls(self) -> int:
        return self.evaluator.n_calls

    def local(self, z: np.ndarray) -> np.ndarray:
        local = self.base.copy()
        local[self.active] = z
        return local

    def residual_of(self, current: np.ndarray) -> np.ndarray:
        return (current[self.index] - self.measured) / self.scale

    def _seen(self, z, current, kind: str) -> tuple[np.ndarray, float]:
        rrms = float(np.mean(fps.curve_rrms_matrix(current, self.layout)[0]))
        if rrms < self.best["rrms"]:
            self.best = {"rrms": rrms, "local": self.local(z),
                         "calls": self.n_calls, "kind": kind}
            self.trace.append([self.n_calls, rrms])
        return self.residual_of(current), rrms

    def residual(self, z: np.ndarray) -> np.ndarray:
        current = self.evaluator.simulate(self.local(z))
        r, self.centre_rrms[z.tobytes()] = self._seen(z, current, "trial")
        return r

    def _simulate_many(self, points: list[np.ndarray]) -> list[tuple]:
        """One altermod process; (residual, rrms, failed) per point."""
        context, ev = self.context, self.evaluator
        t0 = time.time()
        sims = simulate_pdk_multi(
            context.device.dev_type, context.device.L_um, context.device.W_um,
            context.curves,
            [context.box.local_to_params(self.local(p)) for p in points],
            bin_index=context.bin_index, timeout_s=ev.timeout_s)
        ev.sim_s += time.time() - t0
        out = []
        for point, sim in zip(points, sims):
            ev.n_calls += 1
            current = np.concatenate([
                np.asarray(s, dtype=np.float64)[:len(c.Id)]
                for s, c in zip(sim, context.curves)])[ev.order]
            bad = ~np.isfinite(current)
            if bad.any():
                ev.n_failed += 1
                ev.n_failed_points += int(bad.sum())
                current[bad] = ev.fallback[bad]
            out.append((*self._seen(point, current, "jacobian_probe"),
                        bool(bad.any())))
        return out

    def jacobian(self, z: np.ndarray, r: np.ndarray) -> np.ndarray:
        """Forward differences in local coordinates (inward at the upper
        face).  Cliff guard: a probe that fails or more than doubles the
        centre's paper RRMS plus 0.05 (e.g. u0 crossing 1, where BSIM4 rescales
        it by 1e-4) is retried once with the opposite step; if that one is bad too
        (or the call cap is reached) the column is zero."""
        z = np.asarray(z, dtype=np.float64)
        self.n_jacobians += 1
        h = self.diff_step
        steps = np.where(z + h <= 1.0, h, -h)
        eye = np.eye(len(z))
        results = self._simulate_many([z + steps[j] * eye[j]
                                       for j in range(len(z))])
        centre = self.centre_rrms.get(z.tobytes())
        if centre is None:
            centre = float(np.mean(np.sqrt(len(self.layout) * np.add.reduceat(
                r ** 2, self.curve_starts))))

        def bad(row):
            return row[2] or row[1] > self.jump_factor * centre + self.jump_floor

        suspect = [j for j, row in enumerate(results) if bad(row)]
        retry = [j for j in suspect if abs(z[j] - steps[j]) <= 1.0]
        retry = retry[:int(min(len(retry), max(0, self.call_cap - self.n_calls)))]
        if retry:
            self.n_cliff_retries += len(retry)
            again = self._simulate_many([z - steps[j] * eye[j] for j in retry])
            for j, row in zip(retry, again):
                if not bad(row):
                    steps[j], results[j] = -steps[j], row
        jac = np.empty((len(r), len(z)), dtype=np.float64)
        for j, row in enumerate(results):
            if bad(row):
                self.n_zero_columns += 1
                jac[:, j] = 0.0
            else:
                jac[:, j] = (row[0] - r) / steps[j]
        return jac


class SurrogateResiduals:
    """r_s(z) and J_s(z): the frozen surrogate's current on the same points
    and scaling; J_s by forward-mode autograd in float32 on the CPU."""

    def __init__(self, model, context, tensors, ngspice: NgspiceResiduals,
                 base_local: np.ndarray):
        self.model = model
        active = ngspice.active
        n_features = context.box.n_parameters + 3
        base = np.concatenate([np.asarray(base_local, dtype=np.float64),
                               context.geometry]).astype(np.float32)
        base[active] = 0.0
        self.base = torch.tensor(base)
        self.select = torch.zeros(len(active), n_features)
        self.select[torch.arange(len(active)), torch.as_tensor(active)] = 1.0
        self.index = torch.as_tensor(ngspice.index)
        index = ngspice.index
        self.mean = tensors.slog_mean.cpu().numpy().astype(np.float64)[index]
        self.std = tensors.slog_std.cpu().numpy().astype(np.float64)[index]
        self.measured, self.scale = ngspice.measured, ngspice.scale
        self._values: dict[bytes, np.ndarray] = {}
        self._jacobians: dict[bytes, np.ndarray] = {}
        self.n_value = self.n_jacobian = 0

    def _network(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(self.base + z @ self.select)[self.index]

    def _slog(self, normalized: torch.Tensor) -> np.ndarray:
        return normalized.detach().numpy().astype(np.float64) * self.std + self.mean

    def value(self, z: np.ndarray) -> np.ndarray:
        key = np.asarray(z, dtype=np.float64).tobytes()
        if key not in self._values:
            if len(self._values) > 64:
                self._values.clear()
            with torch.no_grad():
                normalized = self._network(torch.tensor(z, dtype=torch.float32))
            slog = self._slog(normalized)
            current = np.sign(slog) * I_REF * np.expm1(np.abs(slog))
            self._values[key] = (current - self.measured) / self.scale
            self.n_value += 1
        return self._values[key]

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        key = np.asarray(z, dtype=np.float64).tobytes()
        if key not in self._jacobians:
            if len(self._jacobians) > 64:
                self._jacobians.clear()

            def network(t):
                out = self._network(t)
                return out, out

            jac, normalized = torch.func.jacfwd(network, has_aux=True)(
                torch.tensor(z, dtype=torch.float32))
            slog = self._slog(normalized)
            dcurrent = I_REF * np.exp(np.abs(slog)) * self.std / self.scale
            self._jacobians[key] = (dcurrent[:, None]
                                    * jac.numpy().astype(np.float64))
            self.n_jacobian += 1
        return self._jacobians[key]


def load_surrogate(round_dir: Path, metadata: dict) -> FoundationSurrogate:
    path = round_dir / "extract" / "forward_model.pt"
    hidden = tuple(torch.load(path, map_location="cpu",
                              weights_only=False)["hidden"])
    model = FoundationSurrogate(metadata["n_parameters"], metadata["n_points"],
                                hidden)
    fps.load_checked_checkpoint(path, model, metadata, hidden,
                                torch.device("cpu"))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def setup(tag: str, options: dict):
    torch.set_num_threads(1)
    context, tensors, evaluator = swap_search.device_setup(tag, options)
    starts, oracle = fps.diverse_oracle_starts(context, options["n_starts"])
    ngspice = NgspiceResiduals(context, evaluator, starts[0],
                               options["diff_step"],
                               call_cap=options.get("budget", math.inf))
    model = None
    if options["model"] == "surrogate":
        metadata = json.loads(
            (Path(options["round_dir"]) / "cache" / "metadata.json").read_text())
        model = load_surrogate(Path(options["round_dir"]), metadata)

    def surrogate_for(start):
        if model is None:
            return None
        return SurrogateResiduals(model, context, tensors, ngspice, start)

    return context, ngspice, starts, oracle, surrogate_for


def settings_from(options: dict) -> TrustRegionSettings:
    return TrustRegionSettings(delta0=options["delta0"])


def prior_weight(options: dict, n_active: int) -> float:
    return math.sqrt(options["regularization"] / max(1, n_active))


def best_at(trace: list, calls: int) -> float:
    value = math.inf
    for n, rrms in trace:
        if n > calls:
            break
        value = rrms
    return value


def fit_device(job) -> tuple[str, dict]:
    tag, options = job
    t0 = time.time()
    context, ngspice, starts, oracle, surrogate_for = setup(tag, options)
    active = ngspice.active
    weight = prior_weight(options, len(active))
    settings = settings_from(options)
    runs = [TrustRegionRun(np.asarray(start, dtype=np.float64)[active],
                           ngspice, surrogate_for(start), weight, settings,
                           label=f"start{index}")
            for index, start in enumerate(starts)]
    run_schedule(runs, ngspice, options["budget"], options["schedule"])
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
    record = {
        "method": "hybrid_trust_region_model_management",
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
        "optimization_objective": (
            "uniform mean squared per-curve normalized point residual plus "
            "weak regularization (as the surrogate-FD pipeline)"),
        "runtime_s": round(time.time() - t0, 1),
    }
    device = context.device
    row = {"device": tag, "dev_type": device.dev_type, "L_um": device.L_um,
           "W_um": device.W_um, "bin_index": context.bin_index,
           "n_active_parameters": context.box.n_active,
           "methods": {"hybrid_tr": record}}
    path = Path(options["out"]) / f"ml_{tag}.json"
    partial = path.with_suffix(".tmp")
    partial.write_text(json.dumps(row, indent=1) + "\n")
    partial.replace(path)
    print(f"{tag}: rrms {record['rrms']:.4f} (in loop {best['rrms']:.4f}, "
          f"first call {ngspice.trace[0][1]:.4f}) calls {ev.n_calls} "
          f"iters {n_iterations} {record['runtime_s']:.0f}s", flush=True)
    return tag, record


def check_device(job) -> tuple[str, dict]:
    """Check 2: at a bank start the corrected model reproduces NGSpice's
    value and Jacobian (asserted), and its error a little way off is
    compared with the plain linearization and the raw surrogate.

    Check 1: a converged card by construction.  The target becomes NGSpice's
    own simulation of that start's card (an override npz for this device,
    set in this worker only), so r_f = 0 there: without the prior the
    Gauss-Newton loop (``--model none``) must stop as stationary with no
    trial call; with the 1e-4 prior the step is whatever the prior alone
    asks for (reported)."""
    tag, options = job
    options = {**options, "model": "surrogate"}
    context, ngspice, starts, _oracle, surrogate_for = setup(tag, options)
    active = ngspice.active
    weight = prior_weight(options, len(active))
    settings = settings_from(options)
    z0 = np.asarray(starts[0], dtype=np.float64)[active]
    surrogate = surrogate_for(starts[0])
    r_f = ngspice.residual(z0)
    J_f = ngspice.jacobian(z0, r_f)
    out = {}
    with_model = CorrectedModel(surrogate, z0, r_f, J_f, weight)
    linear = CorrectedModel(None, z0, r_f, J_f, weight)
    out["consistency_surrogate"] = with_model.consistency(r_f, J_f)
    out["consistency_none"] = linear.consistency(r_f, J_f)
    for key in ("consistency_surrogate", "consistency_none"):
        row = out[key]
        assert row["max_abs_value_error"] <= 1e-9 * row["max_abs_value"], row
        assert row["max_abs_jacobian_error"] <= 1e-9 * row["max_abs_jacobian"], row
    rng = np.random.default_rng(0)
    direction = rng.choice([-1.0, 1.0], size=len(z0))
    off = []
    for t in (0.02, 0.1, 0.25):
        z = np.clip(z0 + t * direction, -1, 1)
        r_true = ngspice.residual(z)
        off.append({"radius": t,
                    "ngspice_change_norm": float(np.linalg.norm(r_true - r_f)),
                    "corrected_surrogate_error_norm": float(np.linalg.norm(
                        with_model.data_residual(z) - r_true)),
                    "linear_error_norm": float(np.linalg.norm(
                        linear.data_residual(z) - r_true)),
                    "raw_surrogate_error_norm": float(np.linalg.norm(
                        surrogate.value(z) - r_true))})
    out["off_centre"] = off
    out["check2_calls"] = ngspice.n_calls
    # Check 1 on a self-simulated target.
    card = context.box.local_to_params(ngspice.local(z0))
    sims = simulate_pdk(
        context.device.dev_type, context.device.L_um, context.device.W_um,
        context.curves, params=card, bin_index=context.bin_index)
    path = Path(options["out"]) / f"check_target_{tag}.npz"
    np.savez(path, **{f"{tag}__{i}": np.asarray(sim, dtype=np.float64)[:len(c.Id)]
                      for i, (sim, c) in enumerate(zip(sims, context.curves))})
    os.environ[TARGET_OVERRIDE_ENV] = str(path)
    _context, target, *_ = setup(tag, {**options, "model": "none"})
    rows = {"max_abs_residual_at_truth": float(np.max(np.abs(
        target.residual(z0))))}
    for name, w in (("no_prior", 0.0), ("prior_1e-4", weight)):
        run = TrustRegionRun(z0, target, None, w, settings, name)
        run.advance(target.n_calls + 2 + target.jacobian_cost)
        first = run.log[0] if run.log else {}
        rows[name] = {"status": run.status, "calls": run.calls,
                      "trial_calls": run.n_accepted + run.n_rejected,
                      "F": run.F_start, "first_step_inf_norm": first.get("step"),
                      "first_predicted": first.get("predicted"),
                      "first_rho": first.get("rho")}
    rows["no_prior_ok"] = (rows["no_prior"]["status"] == "stationary"
                           and rows["no_prior"]["trial_calls"] == 0
                           and rows["max_abs_residual_at_truth"] == 0.0)
    out["check1"] = rows
    print(tag, json.dumps(out, indent=1, default=float), flush=True)
    return tag, out


def load_records(directory: Path) -> dict[str, dict]:
    rows = (json.loads(p.read_text()) for p in sorted(directory.glob("ml_*.json")))
    return {row["device"]: row["methods"]["hybrid_tr"] for row in rows}


def summarize(records: dict, options: dict, round_dir: Path, args,
              n_run: int, wall_s: float) -> dict:
    budgets = {}
    for mark in BUDGET_MARKS:
        rows = {tag: {"rrms": best_at(r["trace"], mark),
                      "official_dynamic_rrms": float("nan")}
                for tag, r in records.items()}
        budgets[str(mark)] = {k: v for k, v in aggregate(rows).items()
                              if k != "official_dynamic_all_device_mean"}
    pipeline_dir = Path(args.pipeline) if args.pipeline else round_dir / "extract"
    pipeline = swap_search.load_records(pipeline_dir)
    iterations = sum(r["n_iterations"] for r in records.values())
    return {
        "round_dir": str(round_dir), "bank": args.bank,
        "model": options["model"], "target_override": target_override() is not None,
        "options": options, **aggregate(records),
        "at_budget": budgets,
        "mean_ngspice_calls_per_device": float(np.mean(
            [r["n_ngspice_calls"] for r in records.values()])),
        "mean_failed_calls_per_device": float(np.mean(
            [r["n_failed"] for r in records.values()])),
        "mean_iterations_per_device": iterations / len(records),
        "subproblem_s_per_iteration": sum(
            r["seconds"]["subproblem"] for r in records.values()) / max(1, iterations),
        "ngspice_s_per_call": float(np.mean(
            [r["ngspice_s_per_call"] for r in records.values()])),
        "device_runtime_sum_s": round(sum(
            r["runtime_s"] for r in records.values()), 1),
        "wall_s": round(wall_s, 1), "n_devices_run_this_invocation": n_run,
        "pipeline": {"dir": str(pipeline_dir),
                     **(aggregate(pipeline) if pipeline else {})},
        "devices": {tag: {
            "rrms": r["rrms"],
            "official_dynamic_rrms": r["official_dynamic_rrms"],
            "pipeline_rrms": pipeline.get(tag, {}).get("rrms"),
            "budget_rrms": r["budget_rrms"],
            "n_ngspice_calls": r["n_ngspice_calls"],
            "n_iterations": r["n_iterations"],
            "best_kind": r["best_kind"], "runtime_s": r["runtime_s"]}
            for tag, r in records.items()}}


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
                        help="NGSpice calls per device, Jacobian probes included")
    parser.add_argument("--n-starts", type=int, default=4)
    parser.add_argument("--schedule", choices=("equal", "halving"),
                        default="equal")
    parser.add_argument("--delta0", type=float, default=0.25)
    parser.add_argument("--diff-step", type=float, default=1e-3,
                        help="forward-difference step in local units; 1e-2 "
                             "probes cross the u0 = 1 cliff (see docstring)")
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--devices", default=None,
                        help="comma-separated device tags (smoke tests)")
    parser.add_argument("--check", action="store_true",
                        help="run the consistency checks on --devices and exit")
    args = parser.parse_args()
    round_dir = Path(args.round_dir)
    label = args.set or round_dir.parent.name
    out_dir = Path(args.out) if args.out else RUNS / (
        f"{label}_{round_dir.name}_{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    options = {"round_dir": str(round_dir), "bank": args.bank,
               "out": str(out_dir), **{k: getattr(args, k) for k in (
                   "model", "budget", "n_starts", "schedule", "delta0",
                   "diff_step", "regularization", "timeout_s")}}
    tags = [device_tag(d.dev_type, d.L_um, d.W_um) for d in PAPER_DEVICES]
    if args.devices:
        tags = [t for t in tags if t in args.devices.split(",")]
    if args.check:
        jobs = [(t, options) for t in tags]
        with get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
            checks = dict(pool.imap_unordered(check_device, jobs))
        (out_dir / "checks.json").write_text(
            json.dumps(checks, indent=1, default=float) + "\n")
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
    out = summarize(records, options, round_dir, args, len(todo),
                    time.time() - t0)
    (out_dir / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in (
        "all_device_mean_rrms", "nmos_rrms", "pmos_rrms", "combined_rrms",
        "mean_ngspice_calls_per_device", "wall_s")}, indent=1))
    print("at budget", json.dumps({b: v["all_device_mean_rrms"]
                                   for b, v in out["at_budget"].items()}))
    print("pipeline", json.dumps({k: out["pipeline"].get(k) for k in (
        "all_device_mean_rrms", "nmos_rrms", "pmos_rrms", "combined_rrms")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
