#!/usr/bin/env python3
"""Study finite-difference fitting independently of the ML initializer.

For every paper device this script runs the same bounded, two-point
finite-difference least-squares polish from the published card itself. It then
compares two paired experiments using one fixed policy across all 18 devices:

1. published parameters -> FD alone;
2. surrogate-search parameters -> FD polish.

The report includes RRMS changes, optimizer effort, and physical parameter
movement for all seven extracted BSIM4 parameters. The direct parameter MLP is
deliberately not polished here; it remains a one-forward-pass comparison in
the main results.

Run only after the surrogate raw and polished variants have been materialized:

  PYTHONPATH=src .venv/bin/python scripts/fd_parameter_study.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cryoml.config import FIGS_DIR, OUT_DIR, OUT_TABLES, PROCESSED_DIR, ensure_dirs  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.metrics import score_device_new  # noqa: E402
from cryoml.pdk_extract import (LhcBox, PARAMS7, eval_params_new,  # noqa: E402
                                new_metric_layout, residual_fn_new)
from cryoml.utils import device_tag, get_logger  # noqa: E402
from pdk_ml_extract import baseline_include_tags  # noqa: E402

logger = get_logger("fd_parameter_study")
SYNTH_DIR = PROCESSED_DIR / "pdk_synth"
FD_ABSOLUTE_STEP_Z = 2e-2


def absolute_two_point_jacobian(residual, z: np.ndarray,
                                step: float) -> np.ndarray:
    """Forward-difference Jacobian with an explicit absolute z-space step.

    SciPy's least_squares ``diff_step`` is relative to x. The published LhcBox
    vector is exactly z=0, where a relative step collapses to a numerically
    ineffective perturbation for NGSpice.
    """
    z = np.asarray(z, dtype=np.float64)
    base = np.asarray(residual(z), dtype=np.float64)
    jacobian = np.empty((len(base), len(z)), dtype=np.float64)
    for index in range(len(z)):
        shifted = z.copy()
        shifted[index] += step
        jacobian[:, index] = (np.asarray(residual(shifted)) - base) / step
    return jacobian


def load_fixed_variant(directory: Path, tag: str, device, curves,
                       include: set[str]) -> tuple[dict, float]:
    rec_path = directory / f"ml_{tag}.json"
    sims_path = directory / f"sims_{tag}.npz"
    if not rec_path.exists() or not sims_path.exists():
        raise RuntimeError(f"fixed variant is incomplete for {tag}: {directory}")
    rec = json.loads(rec_path.read_text())
    saved = np.load(sims_path)
    sims = [np.asarray(saved[f"sim_{i}"]) for i in range(len(curves))]
    score = score_device_new(device.dev_type, device.L_um, device.W_um,
                             curves, sims, include_tags=include)
    return rec, float(score["rrms"])


def run_fd_alone(device, out_dir: Path, max_nfev: int) -> dict:
    tag = device_tag(device.dev_type, device.L_um, device.W_um)
    data = np.load(SYNTH_DIR / f"{tag}.npz", allow_pickle=True)
    if str(data["box_mode"]) != "lhc10":
        raise RuntimeError(f"{tag}: FD study requires current lhc10 data")
    published = {p: float(v) for p, v in zip(PARAMS7, data["published"])}
    bin_index = int(data["bin_index"])
    box = LhcBox(device.dev_type, bin_index, published)
    curves = load_device_curves(device)
    include = baseline_include_tags(tag)
    layout = new_metric_layout(device.dev_type, device.L_um, device.W_um,
                               curves, include)
    before, before_official, before_sims = eval_params_new(
        device.dev_type, device.L_um, device.W_um, bin_index, curves,
        published, include)
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
            residual, box.z_published, method="trf",
            jac=lambda z: absolute_two_point_jacobian(
                residual, z, FD_ABSOLUTE_STEP_Z),
            max_nfev=max_nfev)
        endpoint_params = box.z_to_params(solution.x)
        endpoint, endpoint_official, endpoint_sims = eval_params_new(
            device.dev_type, device.L_um, device.W_um, bin_index, curves,
            endpoint_params, include)
        solution_meta = {
            "success": bool(solution.success), "status": int(solution.status),
            "message": str(solution.message),
            "scipy_nfev": int(solution.nfev),
            "njev": int(solution.njev) if solution.njev is not None else None,
            "objective_evaluations": int(objective_evaluations),
            "cost": float(solution.cost),
            "optimality": float(solution.optimality),
        }
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        endpoint_params = published
        endpoint, endpoint_official, endpoint_sims = (
            before, before_official, before_sims)
        solution_meta = {
            "success": False, "status": None, "message": error,
            "scipy_nfev": 0, "njev": None,
            "objective_evaluations": int(objective_evaluations),
            "cost": None, "optimality": None,
        }

    improved = (np.isfinite(endpoint["rrms"])
                and endpoint["rrms"] <= before["rrms"])
    if improved:
        accepted_params = endpoint_params
        accepted, accepted_official, accepted_sims = (
            endpoint, endpoint_official, endpoint_sims)
    else:
        accepted_params = published
        accepted, accepted_official, accepted_sims = (
            before, before_official, before_sims)

    np.savez(out_dir / f"sims_{tag}.npz",
             **{f"sim_{i}": np.asarray(sim)
                for i, sim in enumerate(accepted_sims)})
    rec = {
        "device": tag, "dev_type": device.dev_type,
        "L_um": device.L_um, "W_um": device.W_um,
        "bin_index": bin_index, "box_mode": "lhc10",
        "method": "published_start_fd_alone",
        "selection_policy": "one published start and one fixed FD recipe",
        "include_tags": sorted(include),
        "published_params": published,
        "endpoint_params": endpoint_params,
        "params": accepted_params,
        "params_by_method": {"published_start_fd_alone": accepted_params},
        "best_method": "published_start_fd_alone",
        "start_rrms": float(before["rrms"]),
        "endpoint_rrms": float(endpoint["rrms"]),
        "rrms": float(accepted["rrms"]),
        "rrms_official": float(accepted_official["rrms"]),
        "endpoint_accepted": bool(improved),
        "runtime_s": round(time.time() - t0, 1),
        "fd_config": {"start": "published", "max_nfev": int(max_nfev),
                      "jacobian": "custom 2-point absolute z-space",
                      "absolute_step_z": FD_ABSOLUTE_STEP_Z,
                      "least_squares_method": "trf"},
        "optimizer": solution_meta,
        "error": error,
    }
    (out_dir / f"ml_{tag}.json").write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def family_means(rows: list[dict], key: str) -> dict[str, float]:
    nmos = [row[key] for row in rows if row["dev_type"] == "nmos"]
    pmos = [row[key] for row in rows if row["dev_type"] == "pmos"]
    return {
        "all_device_mean": float(np.mean(nmos + pmos)),
        "nmos_mean": float(np.mean(nmos)),
        "pmos_mean": float(np.mean(pmos)),
        "combined": float((np.mean(nmos) + np.mean(pmos)) / 2.0),
    }


def write_reports(rows: list[dict], param_rows: list[dict]) -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    with open(OUT_TABLES / "fd_parameter_changes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(param_rows[0]))
        writer.writeheader()
        writer.writerows(param_rows)

    base = family_means(rows, "published_rrms")
    fd = family_means(rows, "fd_alone_rrms")
    raw = family_means(rows, "surrogate_raw_rrms")
    paired = family_means(rows, "surrogate_paired_fd_rrms")
    production = family_means(rows, "surrogate_fd_rrms")
    param_summary = {}
    for parameter in PARAMS7:
        selected = [row for row in param_rows if row["parameter"] == parameter]
        param_summary[parameter] = {
            "fd_alone_mean_abs_percent_of_published": float(np.mean(
                [abs(row["fd_alone_percent_of_published"]) for row in selected])),
            "surrogate_raw_mean_abs_percent_of_published": float(np.mean(
                [abs(row["surrogate_raw_percent_of_published"])
                 for row in selected])),
            "surrogate_paired_polish_mean_abs_percent_of_published": float(np.mean(
                [abs(row["surrogate_paired_polish_delta_percent_of_published"])
                 for row in selected])),
            "surrogate_production_polish_mean_abs_percent_of_published": float(np.mean(
                [abs(row["surrogate_production_delta_percent_of_published"])
                 for row in selected])),
        }
    summary = {
        "policy": "fixed one-start FD-alone and fixed surrogate FD ablation",
        "n_devices": len(rows),
        "published": base,
        "published_start_fd_alone": fd,
        "surrogate_raw": raw,
        "surrogate_raw_winner_plus_fd": paired,
        "surrogate_production_top5_plus_fd": production,
        "fd_alone_all_device_improvement": (
            base["all_device_mean"] - fd["all_device_mean"]),
        "surrogate_fd_all_device_improvement": (
            raw["all_device_mean"] - paired["all_device_mean"]),
        "surrogate_production_all_device_improvement": (
            raw["all_device_mean"] - production["all_device_mean"]),
        "fd_alone_wins": int(sum(
            row["fd_alone_rrms"] < row["published_rrms"] for row in rows)),
        "surrogate_fd_wins": int(sum(
            row["surrogate_paired_fd_rrms"] < row["surrogate_raw_rrms"]
            for row in rows)),
        "fd_alone_mean_nfev": float(np.mean([row["nfev"] for row in rows])),
        "surrogate_pair_mean_nfev": float(np.mean(
            [row["surrogate_pair_nfev"] for row in rows])),
        "fd_alone_mean_runtime_s": float(np.mean(
            [row["runtime_s"] for row in rows])),
        "surrogate_pair_mean_runtime_s": float(np.mean(
            [row["surrogate_pair_runtime_s"] for row in rows])),
        "per_parameter": param_summary,
    }
    (OUT_TABLES / "fd_parameter_study.json").write_text(
        json.dumps({"summary": summary, "devices": rows}, indent=2) + "\n")

    lines = [
        "# Finite-difference parameter study", "",
        "The same fixed curve set and +/-10% seven-parameter box are used for "
        "every row. `FD alone` starts exactly at the published card. The "
        "surrogate columns compare the same surrogate-search result before "
        "and after FD. No per-device method selection is used.", "",
        "| device | published | FD alone | improvement | surrogate raw | "
        "same-winner + FD | paired improvement | production top-5 + FD | "
        "FD-alone nfev | surrogate-pair nfev |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['device']} | {row['published_rrms']:.4f} | "
            f"{row['fd_alone_rrms']:.4f} | "
            f"{row['published_rrms'] - row['fd_alone_rrms']:.4f} | "
            f"{row['surrogate_raw_rrms']:.4f} | "
            f"{row['surrogate_paired_fd_rrms']:.4f} | "
            f"{row['surrogate_raw_rrms'] - row['surrogate_paired_fd_rrms']:.4f} | "
            f"{row['surrogate_fd_rrms']:.4f} | "
            f"{row['nfev']} | {row['surrogate_pair_nfev']} |")
    lines += [
        "", "## Aggregate", "",
        f"- Published -> FD alone: {base['all_device_mean']:.4f} -> "
        f"{fd['all_device_mean']:.4f}; improvement "
        f"{summary['fd_alone_all_device_improvement']:.4f}; "
        f"wins {summary['fd_alone_wins']}/18.",
        f"- Surrogate raw -> surrogate + FD: "
        f"{raw['all_device_mean']:.4f} -> "
        f"{paired['all_device_mean']:.4f}; improvement "
        f"{summary['surrogate_fd_all_device_improvement']:.4f}; "
        f"wins {summary['surrogate_fd_wins']}/18.",
        f"- Production top-five surrogate + FD mean: "
        f"{production['all_device_mean']:.4f}; improvement from the raw "
        f"winner {summary['surrogate_production_all_device_improvement']:.4f}.",
        "", "## Parameter Movement", "",
        "Mean absolute movement is expressed as a percent of the magnitude "
        "of each device's published parameter.", "",
        "| parameter | FD alone | surrogate raw from published | paired FD "
        "movement | production FD movement |",
        "|---|---:|---:|---:|---:|",
    ]
    for parameter, values in param_summary.items():
        lines.append(
            f"| {parameter} | "
            f"{values['fd_alone_mean_abs_percent_of_published']:.3f}% | "
            f"{values['surrogate_raw_mean_abs_percent_of_published']:.3f}% | "
            f"{values['surrogate_paired_polish_mean_abs_percent_of_published']:.3f}% | "
            f"{values['surrogate_production_polish_mean_abs_percent_of_published']:.3f}% |")
    (OUT_TABLES / "fd_parameter_study.md").write_text("\n".join(lines) + "\n")

    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    for ax, before_key, after_key, title, color in (
        (axes[0], "published_rrms", "fd_alone_rrms",
         "FD alone from published cards", "#3f7f68"),
        (axes[1], "surrogate_raw_rrms", "surrogate_paired_fd_rrms",
         "FD on the same surrogate winner", "#cf6c34"),
    ):
        before = np.array([row[before_key] for row in rows])
        after = np.array([row[after_key] for row in rows])
        for i in range(len(rows)):
            ax.plot([i, i], [before[i], after[i]], color="0.75", lw=0.8)
        ax.plot(x, before, "o", color="0.35", ms=4, label="before FD")
        ax.plot(x, after, "^", color=color, ms=5, label="after FD")
        ax.set_title(f"{title}\nmean {before.mean():.3f} -> {after.mean():.3f}")
        ax.set_xlabel("device index")
        ax.set_ylabel("RRMS")
        ax.legend()

    heat = np.empty((len(PARAMS7), len(rows)))
    for pi, parameter in enumerate(PARAMS7):
        selected = [row for row in param_rows if row["parameter"] == parameter]
        heat[pi] = [row["surrogate_paired_polish_delta_percent_of_published"]
                    for row in selected]
    limit = max(1.0, float(np.nanmax(np.abs(heat))))
    image = axes[2].imshow(heat, aspect="auto", cmap="coolwarm",
                           vmin=-limit, vmax=limit)
    axes[2].set_yticks(np.arange(len(PARAMS7)), PARAMS7)
    axes[2].set_xlabel("device index")
    axes[2].set_title("Same-winner parameter movement during FD\n% of published")
    fig.colorbar(image, ax=axes[2], label="signed parameter change (%)")
    fig.suptitle("Finite-difference study: objective gains and parameter movement")
    fig.savefig(FIGS_DIR / "fd_parameter_study.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surrogate-raw",
                        default=str(OUT_DIR / "pdk_ml_emu_raw"))
    parser.add_argument("--surrogate-fd",
                        default=str(OUT_DIR / "pdk_ml_emu"))
    parser.add_argument("--out-dir", default=str(OUT_DIR / "pdk_fd_alone"))
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir, fd_dir = Path(args.surrogate_raw), Path(args.surrogate_fd)

    for device in PAPER_DEVICES:
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        path = out_dir / f"ml_{tag}.json"
        if args.resume and path.exists():
            old = json.loads(path.read_text())
            if (old.get("method") == "published_start_fd_alone"
                    and np.isfinite(old.get("rrms", np.nan))):
                logger.info("%-22s already complete; skipping", tag)
                continue
        rec = run_fd_alone(device, out_dir, args.max_nfev)
        logger.info("%-22s FD alone %.4f -> %.4f (%d nfev, %.0fs)",
                    tag, rec["start_rrms"], rec["rrms"],
                    rec["optimizer"]["objective_evaluations"],
                    rec["runtime_s"])

    rows, param_rows = [], []
    for device in PAPER_DEVICES:
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        fd_alone_path = out_dir / f"ml_{tag}.json"
        if not fd_alone_path.exists():
            raise RuntimeError(f"FD-alone run incomplete: missing {tag}")
        alone = json.loads(fd_alone_path.read_text())
        curves = load_device_curves(device)
        include = set(alone["include_tags"])
        raw, raw_rrms = load_fixed_variant(raw_dir, tag, device, curves, include)
        polished, polished_rrms = load_fixed_variant(
            fd_dir, tag, device, curves, include)
        paired_attempts = [
            attempt for attempt in polished.get("source_fd_attempts", [])
            if attempt.get("is_raw_winner_pair")
        ]
        if len(paired_attempts) != 1:
            raise RuntimeError(
                f"{tag}: expected exactly one raw-winner FD diagnostic, "
                f"found {len(paired_attempts)}"
            )
        paired_attempt = paired_attempts[0]
        if not np.isclose(float(paired_attempt["start_rrms"]), raw_rrms,
                          rtol=1e-7, atol=1e-10):
            raise RuntimeError(
                f"{tag}: raw-winner FD diagnostic does not match the fixed "
                f"raw variant ({paired_attempt['start_rrms']} vs {raw_rrms})"
            )
        row = {
            "device": tag, "dev_type": device.dev_type,
            "published_rrms": float(alone["start_rrms"]),
            "fd_alone_endpoint_rrms": float(alone["endpoint_rrms"]),
            "fd_alone_rrms": float(alone["rrms"]),
            "fd_alone_endpoint_accepted": bool(alone["endpoint_accepted"]),
            "surrogate_raw_rrms": raw_rrms,
            "surrogate_paired_fd_rrms": float(paired_attempt["paired_rrms"]),
            "surrogate_fd_rrms": polished_rrms,
            "nfev": int(alone["optimizer"]["objective_evaluations"]),
            "scipy_nfev": int(alone["optimizer"]["scipy_nfev"]),
            "surrogate_pair_nfev": int(
                paired_attempt["nfev"]
                + len(PARAMS7) * (paired_attempt.get("njev") or 0)),
            "surrogate_pair_scipy_nfev": int(paired_attempt["nfev"]),
            "runtime_s": float(alone["runtime_s"]),
            "surrogate_pair_runtime_s": float(paired_attempt["runtime_s"]),
        }
        rows.append(row)
        published_params = alone["published_params"]
        for parameter in PARAMS7:
            published_value = float(published_params[parameter])
            scale = max(abs(published_value), 1e-30)
            fd_value = float(alone["params"][parameter])
            raw_value = float(raw["params"][parameter])
            paired_value = float(paired_attempt["paired_params"][parameter])
            polished_value = float(polished["params"][parameter])
            param_rows.append({
                "device": tag, "dev_type": device.dev_type,
                "parameter": parameter, "published": published_value,
                "fd_alone": fd_value, "surrogate_raw": raw_value,
                "surrogate_paired_fd": paired_value,
                "surrogate_fd": polished_value,
                "fd_alone_percent_of_published":
                    100.0 * (fd_value - published_value) / scale,
                "surrogate_raw_percent_of_published":
                    100.0 * (raw_value - published_value) / scale,
                "surrogate_paired_fd_percent_of_published":
                    100.0 * (paired_value - published_value) / scale,
                "surrogate_fd_percent_of_published":
                    100.0 * (polished_value - published_value) / scale,
                "surrogate_paired_polish_delta_percent_of_published":
                    100.0 * (paired_value - raw_value) / scale,
                "surrogate_production_delta_percent_of_published":
                    100.0 * (polished_value - raw_value) / scale,
            })

    write_reports(rows, param_rows)
    summary = json.loads((OUT_TABLES / "fd_parameter_study.json").read_text())
    print(json.dumps(summary["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
