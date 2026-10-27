#!/usr/bin/env python3
"""Generate LHC training data for the sensitivity-selected BSIM4 surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.config import OUT_DIR, PROCESSED_DIR, ensure_dirs  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import parse_device_list  # noqa: E402
from cryoml.parameter_space import ParameterBox  # noqa: E402
from cryoml.spice_pdk import simulate_pdk  # noqa: E402
from cryoml.utils import device_tag, get_logger  # noqa: E402

logger = get_logger("pdk_gen_full_data")
DEFAULT_SENSITIVITY = OUT_DIR / "pdk_parameter_sensitivity" / "summary.json"
DEFAULT_OUT = PROCESSED_DIR / "pdk_synth_full"


def parameter_signature(names: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]


def spec_signature(summary: dict) -> str:
    names = tuple(summary["selected_parameters"])
    if "log_scale_by_device" in summary:
        # Cold-start round specs: a new box must never resume an old bank.
        names = names + (hashlib.sha256(json.dumps(
            [summary["bounds_by_device"], summary["published_by_device"]],
            sort_keys=True).encode()).hexdigest(),)
    return parameter_signature(names)


def build_box(summary: dict, tag: str) -> ParameterBox:
    names = tuple(summary["selected_parameters"])
    log_flags = summary.get("log_scale_by_device", {}).get(tag)
    return ParameterBox(
        names=names,
        published=summary["published_by_device"][tag],
        bounds=summary["bounds_by_device"][tag],
        log_scale=(None if log_flags is None
                   else tuple(bool(log_flags[name]) for name in names)))


def static_curve_layout(curves):
    vg = np.concatenate([np.asarray(curve.Vg, dtype=np.float64)
                         for curve in curves])
    vd = np.concatenate([np.asarray(curve.Vd, dtype=np.float64)
                         for curve in curves])
    measured = np.concatenate([np.asarray(curve.Id, dtype=np.float64)
                               for curve in curves])
    slices, cursor = [], 0
    for curve in curves:
        slices.append((cursor, cursor + len(curve.Id)))
        cursor += len(curve.Id)
    return {
        "Vg": vg, "Vd": vd, "meas": measured,
        "slices": np.asarray(slices, dtype=np.int64),
        "kinds": np.asarray([curve.kind for curve in curves]),
        "fixeds": np.asarray([curve.fixed for curve in curves],
                             dtype=np.float64),
    }


def sample_local(box: ParameterBox, num_samples: int, seed: int,
                 summary: dict, tag: str, mode: str) -> np.ndarray:
    """Sample globally, with optional outward slabs and paired interactions."""
    local = np.zeros((num_samples, box.n_parameters), dtype=np.float64)
    if box.n_active:
        sampler = qmc.LatinHypercube(d=box.n_active, seed=seed)
        local[:, box.active] = 2.0 * sampler.random(num_samples) - 1.0
    if mode == "global_lhc":
        local[0] = box.local_published
        return local

    changes = [row for row in summary.get("adaptive_changes", [])
               if row["device"] == tag]
    if not changes:
        local[0] = box.local_published
        return local
    rng = np.random.default_rng(seed + 8147)
    names = {name: index for index, name in enumerate(box.names)}
    # Thirty-five percent of rows resolve one newly opened boundary slab;
    # another fifteen percent resolve two slabs jointly. Other coordinates
    # retain stratified global coverage, so this is an augmentation of the
    # full surface rather than a per-device local fit.
    n_single = int(round(0.35 * (num_samples - 1)))
    n_pair = int(round(0.15 * (num_samples - 1)))
    focus_rows = np.arange(1, min(num_samples, 1 + n_single + n_pair))

    def focus(row_index: int, change: dict) -> None:
        column = names[change["parameter"]]
        old_bound = (change["old_bounds"][1]
                     if change["direction"] == "upper"
                     else change["old_bounds"][0])
        old_local = (2.0 * (float(old_bound) - box.lo[column])
                     / (box.hi[column] - box.lo[column]) - 1.0)
        if change["direction"] == "upper":
            local[row_index, column] = rng.uniform(old_local, 1.0)
        else:
            local[row_index, column] = rng.uniform(-1.0, old_local)

    for offset, row_index in enumerate(focus_rows[:n_single]):
        focus(row_index, changes[offset % len(changes)])
    changes_by_parameter = {}
    for change in changes:
        changes_by_parameter.setdefault(change["parameter"], []).append(change)
    focal_parameters = list(changes_by_parameter)
    for row_index in focus_rows[n_single:]:
        selection = rng.choice(
            len(focal_parameters), size=min(2, len(focal_parameters)),
            replace=False)
        for index in np.atleast_1d(selection):
            candidates = changes_by_parameter[focal_parameters[int(index)]]
            focus(row_index, candidates[int(rng.integers(len(candidates)))])
    local[0] = box.local_published
    return local


def generate_device(job):
    (device, bin_index, summary, num_samples, seed, out_dir_string,
     use_float64, sampling_mode) = job
    tag = device_tag(device.dev_type, device.L_um, device.W_um)
    t0 = time.time()
    names = tuple(summary["selected_parameters"])
    box = build_box(summary, tag)
    curves = load_device_curves(device)
    layout = static_curve_layout(curves)

    local = sample_local(
        box, num_samples, seed, summary, tag, sampling_mode)
    theta = box.local_to_array(local)

    n_points = len(layout["meas"])
    current_dtype = np.float64 if use_float64 else np.float32
    ids = np.full((num_samples, n_points), np.nan, dtype=current_dtype)
    for index, values in enumerate(theta):
        params = {name: float(values[column])
                  for column, name in enumerate(names)}
        sims = simulate_pdk(
            device.dev_type, device.L_um, device.W_um, curves,
            params=params, bin_index=bin_index)
        ids[index] = np.concatenate([
            np.asarray(sim, dtype=np.float64)[:len(curve.Id)]
            for sim, curve in zip(sims, curves)])
    ok = np.isfinite(ids).all(axis=1)
    out_dir = Path(out_dir_string)
    out_path = out_dir / f"{tag}.npz"
    np.savez_compressed(
        out_path,
        LOCAL=local.astype(np.float32),
        THETA=theta.astype(np.float64),
        IDS=ids, ok=ok,
        **layout,
        bin_index=np.asarray(bin_index),
        parameter_names=np.asarray(names),
        parameter_signature=np.asarray(spec_signature(summary)),
        published=box.center.astype(np.float64),
        lo=box.lo.astype(np.float64), hi=box.hi.astype(np.float64),
        active=box.active,
        log_scale=box.log,
        seed=np.asarray(seed),
        sampling=np.asarray(sampling_mode),
    )
    return {
        "device": tag, "path": str(out_path),
        "n_samples": int(num_samples), "n_valid": int(np.sum(ok)),
        "n_parameters": len(names), "n_active": box.n_active,
        "runtime_s": round(time.time() - t0, 1), "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default=None)
    parser.add_argument("--sensitivity", default=str(DEFAULT_SENSITIVITY))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--sampling", choices=("global_lhc", "adaptive_mixture"),
        default="global_lhc",
        help="adaptive mixture adds single- and paired-boundary slabs")
    parser.add_argument("--float64", action="store_true",
                        help="store currents as float64 instead of float32")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.num_samples < 2:
        parser.error("--num-samples must be at least two")

    ensure_dirs()
    summary = json.loads(Path(args.sensitivity).read_text())
    names = tuple(summary["selected_parameters"])
    if not names:
        raise RuntimeError("sensitivity summary selected no parameters")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    devices = parse_device_list(args.devices)
    # Released-card bins from pdk_baseline.json, or with
    # CRYOML_PDK_TEMPLATE=stock the NGSpice-selected stock bins (I25-1).
    from cryoml.cold_start import device_bins
    bins = device_bins()


    records, jobs = [], []
    signature = spec_signature(summary)
    for index, device in enumerate(devices):
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        out_path = out_dir / f"{tag}.npz"
        if args.resume and out_path.exists():
            with np.load(out_path, allow_pickle=False) as data:
                compatible = (
                    str(data["parameter_signature"]) == signature
                    and len(data["LOCAL"]) == args.num_samples
                    and str(data["sampling"]) == args.sampling)
                if compatible:
                    records.append({
                        "device": tag, "path": str(out_path),
                        "n_samples": int(len(data["LOCAL"])),
                        "n_valid": int(np.sum(data["ok"])),
                        "n_parameters": int(data["LOCAL"].shape[1]),
                        "n_active": int(np.sum(data["active"])),
                        "runtime_s": 0.0,
                        "seed": int(data["seed"]), "resumed": True,
                    })
                    continue
        jobs.append((device, bins[tag], summary, args.num_samples,
                     args.seed + 1000 * index, str(out_dir), args.float64,
                     args.sampling))

    if jobs:
        if args.workers == 1:
            iterator = map(generate_device, jobs)
            for record in iterator:
                records.append(record)
                logger.info("%-22s %d/%d valid, %d active params (%.1fs)",
                            record["device"], record["n_valid"],
                            record["n_samples"], record["n_active"],
                            record["runtime_s"])
        else:
            context = get_context("spawn")
            with context.Pool(args.workers) as pool:
                for record in pool.imap_unordered(generate_device, jobs):
                    records.append(record)
                    logger.info("%-22s %d/%d valid, %d active params (%.1fs)",
                                record["device"], record["n_valid"],
                                record["n_samples"], record["n_active"],
                                record["runtime_s"])
    records.sort(key=lambda row: row["device"])
    manifest = {
        "schema_version": 1, "parameter_names": list(names),
        "parameter_signature": signature,
        "sensitivity_summary": str(Path(args.sensitivity)),
        "sampling": args.sampling,
        "sampling_definition": (
            "global Latin hypercube"
            if args.sampling == "global_lhc" else
            "50% global LHC, 35% single outward-boundary slabs, and 15% "
            "paired outward-boundary slabs"),
        "current_dtype": "float64" if args.float64 else "float32",
        "num_samples_requested": args.num_samples,
        "seed": args.seed, "devices": records,
        "all_valid": all(row["n_valid"] == row["n_samples"]
                         for row in records),
        "wall_runtime_summed": round(sum(row["runtime_s"] for row in records), 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
