#!/usr/bin/env python3
"""NGSpice control row for surrogate ablations: the paper search, unchanged,
with the neural surrogate swapped for NGSpice.

``finite_difference_surrogate_extract`` from ``foundation_parameter_study``
runs as is: the same diverse best-bank starts, the same bounded TRF least
squares (two-point Jacobian, ``diff_step``, ``max_nfev``, ``x_scale='jac'``,
weak regularization), and the same NGSpice validation of the top endpoints,
the first starts and the published fallback.  Only its ``model`` is replaced
by ``NgspiceEvaluator``, which answers every forward call with a fresh NGSpice
simulation in the surrogate's output convention (standardized signed log in
canonical point order).  The gap between any architecture's end-to-end result
and this row is therefore pure surrogate error.

Failed or non-finite simulated points become the measured value plus
``PENALTY`` (10) x the curve's paper denominator on included metric curves
(per-curve RRMS 10, as in ``cryoml.ngspice_fit``) and the measured value
elsewhere, so the least squares never sees NaN.

Each device writes ``ml_<tag>.json`` in the pipeline's record format (so
``polish_cards.py --cards <out> --method surrogate_fd`` works unchanged);
``summary.json`` is written once all 18 exist.  ``CRYOML_TARGET_OVERRIDE``
applies through ``build_device_context`` as in the pipeline.

  PYTHONPATH=src .venv/bin/python scripts/research/swap_search.py \
      --round-dir out/cold_loop/full/round_1 \
      --bank data/processed/cold_loop/full_r1 --set full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import foundation_parameter_study as fps  # noqa: E402
from cryoml.card_eval import aggregate, device_by_tag  # noqa: E402
from cryoml.data_io import target_override  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.foundation import canonical_point_order, signed_log  # noqa: E402
from cryoml.ngspice_fit import PENALTY  # noqa: E402
from cryoml.spice_pdk import simulate_pdk  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402

RUNS = ROOT / "out" / "research" / "swap_search"


class NgspiceEvaluator:
    """Callable stand-in for the forward surrogate, backed by NGSpice.

    Takes the surrogate's (B, n_in) features (local card + 3 geometry
    columns) and returns (B, n_points) float32 standardized signed log in
    canonical order, so ``fps.predict_surrogate`` recovers the simulated
    current."""

    def __init__(self, context, tensors, order: np.ndarray,
                 timeout_s: float = 20.0):
        self.context = context
        self.order = np.asarray(order, dtype=np.int64)
        if len(self.order) != sum(len(c.Id) for c in context.curves):
            raise RuntimeError(f"{context.tag}: bank layout != measured curves")
        self.timeout_s = timeout_s
        self.slog_mean = tensors.slog_mean.detach().cpu().numpy().astype(np.float64)
        self.slog_std = tensors.slog_std.detach().cpu().numpy().astype(np.float64)
        fallback = np.asarray(context.measured_canonical, dtype=np.float64).copy()
        for entry in fps.curve_objective_layout(context):
            fallback[entry["start"]:entry["stop"]] = (
                entry["measured"] + PENALTY * entry["denominator"])
        self.fallback = fallback
        self.n_calls = self.n_failed = self.n_failed_points = 0
        self.sim_s = 0.0

    def eval(self):
        return self

    def simulate(self, local: np.ndarray) -> np.ndarray:
        """Canonical-order NGSpice current with failed points penalized."""
        context = self.context
        t0 = time.time()
        sims = simulate_pdk(
            context.device.dev_type, context.device.L_um, context.device.W_um,
            context.curves, params=context.box.local_to_params(local),
            bin_index=context.bin_index, timeout_s=self.timeout_s)
        self.sim_s += time.time() - t0
        self.n_calls += 1
        # Load order, exactly as the bank stores IDS, then canonical order.
        current = np.concatenate([
            np.asarray(sim, dtype=np.float64)[:len(curve.Id)]
            for sim, curve in zip(sims, context.curves)])[self.order]
        bad = ~np.isfinite(current)
        if bad.any():
            self.n_failed += 1
            self.n_failed_points += int(bad.sum())
            current[bad] = self.fallback[bad]
        return current

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        n_parameters = self.context.box.n_parameters
        local = features[:, :n_parameters].detach().cpu().numpy()
        current = np.stack([self.simulate(row.astype(np.float64))
                            for row in local])
        normalized = (signed_log(current) - self.slog_mean) / self.slog_std
        return torch.tensor(normalized, dtype=torch.float32,
                            device=features.device)


def device_setup(tag: str, options: dict):
    cache = Path(options["round_dir"]) / "cache"
    metadata = json.loads((cache / "metadata.json").read_text())
    tensors = fps.training_tensors(cache, torch.device("cpu"))
    bank = Path(options["bank"])
    # Cold-loop rounds (seven and full alike) are built with the pipeline's
    # "full" dataset layout; the cache records which one was used.
    context = fps.build_device_context(
        device_by_tag(tag), bank, metadata, metadata["parameter_set"])
    with np.load(bank / f"{tag}.npz", allow_pickle=True) as data:
        order, _ = canonical_point_order(
            data["kinds"], data["fixeds"], data["slices"])
    return context, tensors, NgspiceEvaluator(
        context, tensors, order, options["timeout_s"])


def search_device(job) -> tuple[str, dict]:
    tag, options = job
    torch.set_num_threads(1)
    t0 = time.time()
    context, tensors, evaluator = device_setup(tag, options)
    record, _sims = fps.finite_difference_surrogate_extract(
        context, evaluator, tensors, torch.device("cpu"),
        n_starts=options["n_starts"], n_validate=options["n_validate"],
        max_nfev=options["max_nfev"], diff_step=options["diff_step"],
        regularization=options["regularization"])
    record.update({
        "evaluator": "ngspice",
        # Search-time calls (the FD loop and candidate scoring); the
        # pipeline's own validation adds ``n_validated`` simulations.
        "n_ngspice_calls": evaluator.n_calls,
        "n_failed": evaluator.n_failed,
        "n_failed_points": evaluator.n_failed_points,
        "ngspice_s_per_call": round(evaluator.sim_s / max(1, evaluator.n_calls), 4),
        "runtime_s": round(time.time() - t0, 1),
    })
    device = context.device
    row = {"device": tag, "dev_type": device.dev_type, "L_um": device.L_um,
           "W_um": device.W_um, "bin_index": context.bin_index,
           "n_active_parameters": context.box.n_active,
           "methods": {"surrogate_fd": record}}
    path = Path(options["out"]) / f"ml_{tag}.json"
    partial = path.with_suffix(".tmp")
    partial.write_text(json.dumps(row, indent=2) + "\n")
    partial.replace(path)
    print(f"{tag}: rrms {record['rrms']:.4f} ({record['selected_candidate_kind']}) "
          f"calls {evaluator.n_calls} failed {evaluator.n_failed} "
          f"{record['runtime_s']:.0f}s", flush=True)
    return tag, record


def load_records(directory: Path) -> dict[str, dict]:
    return {json.loads(p.read_text())["device"]:
            json.loads(p.read_text())["methods"]["surrogate_fd"]
            for p in sorted(directory.glob("ml_*.json"))}


def roundtrip(tags: list[str], options: dict, n_samples: int) -> dict:
    """Bank sample -> evaluator -> ``predict_surrogate`` vs stored bank IDS."""
    out = {}
    for tag in tags:
        context, tensors, evaluator = device_setup(tag, options)
        picks = np.linspace(0, len(context.local_samples) - 1,
                            n_samples).astype(int)
        rel, peak_rel = [], []
        for index in picks:
            predicted = fps.predict_surrogate(
                evaluator, context.local_samples[index], context, tensors,
                torch.device("cpu"))[0]
            stored = np.asarray(context.current_samples[index], dtype=np.float64)
            error = np.abs(predicted - stored)
            # Pointwise relative error above 1 nA, and error / curve peak.
            big = np.abs(stored) > 1e-9
            rel.append(float(np.max(error[big] / np.abs(stored[big]))))
            for desc in context.canonical_curves:
                block = slice(desc["start"], desc["stop"])
                peak = max(float(np.max(np.abs(stored[block]))), 1e-30)
                peak_rel.append(float(np.max(error[block]) / peak))
        out[tag] = {"samples": picks.tolist(),
                    "max_rel_error_above_1nA": float(np.max(rel)),
                    "median_rel_error_above_1nA": float(np.median(rel)),
                    "max_error_over_curve_peak": float(np.max(peak_rel)),
                    "n_failed": evaluator.n_failed,
                    "s_per_call": evaluator.sim_s / max(1, evaluator.n_calls)}
        print(tag, json.dumps(out[tag]), flush=True)
    return out


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
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--n-starts", type=int, default=8)
    parser.add_argument("--n-validate", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument("--diff-step", type=float, default=0.02)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--devices", default=None,
                        help="comma-separated device tags (smoke tests)")
    parser.add_argument("--roundtrip", type=int, default=0,
                        help="only check N bank samples per device and exit")
    args = parser.parse_args()
    round_dir = Path(args.round_dir)
    label = args.set or round_dir.parent.name
    out_dir = Path(args.out) if args.out else RUNS / f"{label}_{round_dir.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    options = {"round_dir": str(round_dir), "bank": args.bank,
               "out": str(out_dir), "timeout_s": args.timeout_s,
               **{k: getattr(args, k) for k in (
                   "n_starts", "n_validate", "max_nfev", "diff_step",
                   "regularization")}}
    tags = [device_tag(d.dev_type, d.L_um, d.W_um) for d in PAPER_DEVICES]
    if args.devices:
        tags = [t for t in tags if t in args.devices.split(",")]
    if args.roundtrip:
        roundtrip(tags, options, args.roundtrip)
        return 0
    todo = [t for t in tags if not (out_dir / f"ml_{t}.json").exists()]
    # Longest-running (pMOS) devices first keeps the pool busy.
    todo.sort(key=lambda t: not t.startswith("pmos"))
    t0 = time.time()
    if todo:
        with get_context("spawn").Pool(min(args.workers, len(todo))) as pool:
            for _ in pool.imap_unordered(
                    search_device, [(t, options) for t in todo]):
                pass
    records = load_records(out_dir)
    if len(records) < len(PAPER_DEVICES):
        print(f"{len(records)}/{len(PAPER_DEVICES)} devices done; "
              "summary.json is written once all exist")
        return 0
    pipeline_dir = Path(args.pipeline) if args.pipeline else round_dir / "extract"
    pipeline = load_records(pipeline_dir)
    out = {"round_dir": str(round_dir), "bank": args.bank,
           "evaluator": "ngspice",
           "target_override": target_override() is not None,
           "options": options, **aggregate(records),
           "mean_ngspice_calls_per_device": float(np.mean(
               [r["n_ngspice_calls"] for r in records.values()])),
           "mean_failed_calls_per_device": float(np.mean(
               [r["n_failed"] for r in records.values()])),
           "wall_s": round(time.time() - t0, 1),
           "n_devices_run_this_invocation": len(todo),
           "device_runtime_sum_s": round(sum(
               r["runtime_s"] for r in records.values()), 1),
           "pipeline": {"dir": str(pipeline_dir),
                        **(aggregate(pipeline) if pipeline else {})},
           "devices": {tag: {
               "rrms": r["rrms"],
               "official_dynamic_rrms": r["official_dynamic_rrms"],
               "pipeline_rrms": pipeline.get(tag, {}).get("rrms"),
               "selected_candidate_kind": r["selected_candidate_kind"],
               "n_ngspice_calls": r["n_ngspice_calls"],
               "n_failed": r["n_failed"], "runtime_s": r["runtime_s"]}
               for tag, r in records.items()}}
    (out_dir / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in (
        "all_device_mean_rrms", "nmos_rrms", "pmos_rrms", "combined_rrms",
        "mean_ngspice_calls_per_device", "wall_s")}, indent=1))
    print("pipeline", json.dumps({k: out["pipeline"].get(k) for k in (
        "all_device_mean_rrms", "nmos_rrms", "pmos_rrms", "combined_rrms")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
