#!/usr/bin/env python3
"""Polish every fixed direct-MLP prediction with measured-data FD.

This is a paired ablation, not a new primary method. For each of the 18 paper
devices, the sole optimizer start is the corresponding one-pass prediction in
``out/pdk_direct_mlp``. The least-squares residual is evaluated by fresh
NGSpice simulations against the fixed measured-curve set used everywhere else.

The raw direct MLP remains the main one-pass comparison. This script writes a
separate ``direct MLP + FD`` artifact so the effect of finite differences can
be reported without per-device initializer selection.

  PYTHONPATH=src .venv/bin/python scripts/direct_mlp_fd_study.py --resume
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.config import OUT_DIR, OUT_TABLES, PROCESSED_DIR, ensure_dirs  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.metrics import score_device_new  # noqa: E402
from cryoml.pdk_extract import (LhcBox, PARAMS7, eval_params_new,  # noqa: E402
                                new_metric_layout, residual_fn_new)
from cryoml.utils import device_tag, get_logger  # noqa: E402


logger = get_logger("direct_mlp_fd_study")
SYNTH_DIR = PROCESSED_DIR / "pdk_synth"
DEFAULT_RAW_DIR = OUT_DIR / "pdk_direct_mlp"
DEFAULT_OUT_DIR = OUT_DIR / "pdk_direct_mlp_fd"
EXPECTED_RAW_METHOD = "direct_mlp_forward_pass"
OUTPUT_METHOD = "direct_mlp_forward_pass+fd"
FD_RELATIVE_STEP_Z = 2e-2


def family_means(rows: list[dict], key: str) -> dict[str, float]:
    nmos = [row[key] for row in rows if row["dev_type"] == "nmos"]
    pmos = [row[key] for row in rows if row["dev_type"] == "pmos"]
    return {
        "all_device_mean": float(np.mean(nmos + pmos)),
        "nmos_mean": float(np.mean(nmos)),
        "pmos_mean": float(np.mean(pmos)),
        "combined": float((np.mean(nmos) + np.mean(pmos)) / 2.0),
    }


def load_saved_sims(directory: Path, tag: str,
                    n_curves: int) -> list[np.ndarray]:
    saved = np.load(directory / f"sims_{tag}.npz")
    return [np.asarray(saved[f"sim_{index}"]) for index in range(n_curves)]


def polish_device(device, raw_dir: Path, out_dir: Path,
                  max_nfev: int) -> dict:
    tag = device_tag(device.dev_type, device.L_um, device.W_um)
    raw_path = raw_dir / f"ml_{tag}.json"
    if not raw_path.exists():
        raise RuntimeError(f"{tag}: missing direct-MLP record {raw_path}")
    raw = json.loads(raw_path.read_text())
    if raw.get("method") != EXPECTED_RAW_METHOD:
        raise RuntimeError(
            f"{tag}: expected {EXPECTED_RAW_METHOD}, got {raw.get('method')}")
    if raw.get("box_mode") != "lhc10":
        raise RuntimeError(f"{tag}: direct MLP is not from the lhc10 setup")

    data = np.load(SYNTH_DIR / f"{tag}.npz", allow_pickle=True)
    if str(data["box_mode"]) != "lhc10":
        raise RuntimeError(f"{tag}: synthetic data is not from lhc10")
    published = {name: float(value)
                 for name, value in zip(PARAMS7, data["published"])}
    bin_index = int(data["bin_index"])
    if int(raw["bin_index"]) != bin_index:
        raise RuntimeError(f"{tag}: direct record has the wrong native bin")

    curves = load_device_curves(device)
    include = set(raw["include_tags"])
    box = LhcBox(device.dev_type, bin_index, published)
    layout = new_metric_layout(device.dev_type, device.L_um, device.W_um,
                               curves, include)
    start_params = {name: float(raw["params"][name]) for name in PARAMS7}
    start_z = box.params_to_z(start_params)

    # Rescore the archived raw curves and also evaluate the exact start with a
    # fresh NGSpice run. Both checks make the paired experiment auditable.
    archived_sims = load_saved_sims(raw_dir, tag, len(curves))
    archived_score = score_device_new(
        device.dev_type, device.L_um, device.W_um, curves, archived_sims,
        include_tags=include)
    start, start_official, start_sims = eval_params_new(
        device.dev_type, device.L_um, device.W_um, bin_index, curves,
        start_params, include)
    if not np.isclose(float(raw["rrms"]), float(archived_score["rrms"]),
                      rtol=1e-7, atol=1e-10):
        raise RuntimeError(
            f"{tag}: raw record/simulation score mismatch "
            f"({raw['rrms']} vs {archived_score['rrms']})")
    if not np.isclose(float(start["rrms"]), float(archived_score["rrms"]),
                      rtol=1e-7, atol=1e-10):
        raise RuntimeError(
            f"{tag}: fresh NGSpice start does not reproduce archived MLP "
            f"curves ({start['rrms']} vs {archived_score['rrms']})")

    objective_evaluations = 0

    def residual(z):
        nonlocal objective_evaluations
        objective_evaluations += 1
        return residual_fn_new(
            z, box, device.dev_type, device.L_um, device.W_um, bin_index,
            curves, layout)

    t0 = time.time()
    error = None
    try:
        solution = least_squares(
            residual, start_z, method="trf", jac="2-point",
            diff_step=FD_RELATIVE_STEP_Z, max_nfev=max_nfev)
        endpoint_params = box.z_to_params(solution.x)
        endpoint, endpoint_official, endpoint_sims = eval_params_new(
            device.dev_type, device.L_um, device.W_um, bin_index, curves,
            endpoint_params, include)
        optimizer = {
            "success": bool(solution.success),
            "status": int(solution.status),
            "message": str(solution.message),
            "scipy_nfev": int(solution.nfev),
            "njev": int(solution.njev) if solution.njev is not None else None,
            "objective_evaluations": int(objective_evaluations),
            "cost": float(solution.cost),
            "optimality": float(solution.optimality),
        }
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        endpoint_params = start_params
        endpoint, endpoint_official, endpoint_sims = (
            start, start_official, start_sims)
        optimizer = {
            "success": False, "status": None, "message": error,
            "scipy_nfev": 0, "njev": None,
            "objective_evaluations": int(objective_evaluations),
            "cost": None, "optimality": None,
        }

    accepted = (np.isfinite(endpoint["rrms"])
                and endpoint["rrms"] <= start["rrms"])
    if accepted:
        params = endpoint_params
        score, official, sims = endpoint, endpoint_official, endpoint_sims
    else:
        params = start_params
        score, official, sims = start, start_official, start_sims

    np.savez(out_dir / f"sims_{tag}.npz",
             **{f"sim_{index}": np.asarray(sim)
                for index, sim in enumerate(sims)})
    record = {
        "device": tag, "dev_type": device.dev_type,
        "L_um": device.L_um, "W_um": device.W_um,
        "bin_index": bin_index, "box_mode": "lhc10",
        "method": OUTPUT_METHOD,
        "source_method": EXPECTED_RAW_METHOD,
        "selection_policy": (
            "one fixed direct-MLP start and one fixed measured-data FD recipe"),
        "include_tags": sorted(include),
        "published_params": published,
        "start_params": start_params,
        "endpoint_params": endpoint_params,
        "params": params,
        "params_by_method": {OUTPUT_METHOD: params},
        "best_method": OUTPUT_METHOD,
        "archived_start_rrms": float(archived_score["rrms"]),
        "start_rrms": float(start["rrms"]),
        "endpoint_rrms": float(endpoint["rrms"]),
        "rrms": float(score["rrms"]),
        "rrms_official": float(official["rrms"]),
        "endpoint_accepted": bool(accepted),
        "runtime_s": round(time.time() - t0, 1),
        "fd_config": {
            "objective": "unchanged paper RRMS against measured curves",
            "start": EXPECTED_RAW_METHOD,
            "max_nfev": int(max_nfev),
            "jacobian": "SciPy two-point relative z-space",
            "relative_step_z": FD_RELATIVE_STEP_Z,
            "least_squares_method": "trf",
        },
        "optimizer": optimizer,
        "error": error,
    }
    (out_dir / f"ml_{tag}.json").write_text(
        json.dumps(record, indent=2) + "\n")
    return record


def write_reports(records: list[dict], out_dir: Path) -> dict:
    rows = [{
        "device": record["device"],
        "dev_type": record["dev_type"],
        "raw_rrms": float(record["start_rrms"]),
        "endpoint_rrms": float(record["endpoint_rrms"]),
        "fd_rrms": float(record["rrms"]),
        "improvement": float(record["start_rrms"] - record["rrms"]),
        "endpoint_accepted": bool(record["endpoint_accepted"]),
        "scipy_nfev": int(record["optimizer"]["scipy_nfev"]),
        "objective_evaluations": int(
            record["optimizer"]["objective_evaluations"]),
        "runtime_s": float(record["runtime_s"]),
    } for record in records]
    raw = family_means(rows, "raw_rrms")
    polished = family_means(rows, "fd_rrms")
    summary = {
        "policy": "fixed direct MLP start plus fixed measured-data FD",
        "n_devices": len(rows),
        "direct_mlp_raw": raw,
        "direct_mlp_plus_fd": polished,
        "all_device_improvement": (
            raw["all_device_mean"] - polished["all_device_mean"]),
        "wins": int(sum(row["fd_rrms"] < row["raw_rrms"] for row in rows)),
        "accepted_endpoints": int(sum(
            row["endpoint_accepted"] for row in rows)),
        "mean_scipy_nfev": float(np.mean(
            [row["scipy_nfev"] for row in rows])),
        "mean_objective_evaluations": float(np.mean(
            [row["objective_evaluations"] for row in rows])),
        "mean_runtime_s": float(np.mean(
            [row["runtime_s"] for row in rows])),
        "total_runtime_s": float(np.sum(
            [row["runtime_s"] for row in rows])),
    }
    payload = {"summary": summary, "devices": rows}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT_TABLES / "direct_mlp_fd_study.json").write_text(
        json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Direct MLP finite-difference ablation", "",
        "Every row starts from the one-pass direct MLP prediction. FD uses "
        "fresh NGSpice evaluations of the unchanged paper RRMS residual "
        "against measured curves. No initializer selection is performed.", "",
        "| device | direct MLP | direct MLP + FD | improvement | accepted | "
        "nfev | objective evals | runtime (s) |",
        "|---|---:|---:|---:|:---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['device']} | {row['raw_rrms']:.4f} | "
            f"{row['fd_rrms']:.4f} | {row['improvement']:.4f} | "
            f"{'yes' if row['endpoint_accepted'] else 'no'} | "
            f"{row['scipy_nfev']} | {row['objective_evaluations']} | "
            f"{row['runtime_s']:.1f} |")
    lines += [
        "", "## Aggregate", "",
        f"- All-device mean: {raw['all_device_mean']:.4f} -> "
        f"{polished['all_device_mean']:.4f} (improvement "
        f"{summary['all_device_improvement']:.4f}).",
        f"- Improved devices: {summary['wins']}/{len(rows)}.",
        f"- Total runtime: {summary['total_runtime_s']:.1f} s.",
    ]
    (OUT_TABLES / "direct_mlp_fd_study.md").write_text(
        "\n".join(lines) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for device in PAPER_DEVICES:
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        path = args.out_dir / f"ml_{tag}.json"
        if args.resume and path.exists():
            old = json.loads(path.read_text())
            if (old.get("method") == OUTPUT_METHOD
                    and np.isfinite(old.get("rrms", np.nan))):
                logger.info("%-22s already complete; skipping", tag)
                continue
        record = polish_device(
            device, args.raw_dir, args.out_dir, args.max_nfev)
        logger.info(
            "%-22s direct %.4f -> +FD %.4f (%d objective evals, %.1fs)",
            tag, record["start_rrms"], record["rrms"],
            record["optimizer"]["objective_evaluations"],
            record["runtime_s"])

    records = []
    for device in PAPER_DEVICES:
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        path = args.out_dir / f"ml_{tag}.json"
        if not path.exists():
            raise RuntimeError(f"direct MLP + FD run incomplete: {tag}")
        record = json.loads(path.read_text())
        if record.get("method") != OUTPUT_METHOD:
            raise RuntimeError(f"{tag}: stale direct MLP + FD artifact")
        records.append(record)
    summary = write_reports(records, args.out_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
