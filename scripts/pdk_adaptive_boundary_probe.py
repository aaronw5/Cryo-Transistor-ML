#!/usr/bin/env python3
"""Physically test outward parameter-bound expansions before adding data.

Boundary saturation alone is ambiguous in a non-identifiable compact model.
This script probes repeated bound hits in fresh NGSpice runs and requires an
improvement in the independent x-axis section objective.  It only diagnoses
which parameter/polarity/direction expansions deserve adaptive training data;
it does not silently alter the foundation box.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.metrics import curve_trim, score_device_new  # noqa: E402
from cryoml.spice_pdk import simulate_pdk  # noqa: E402
from cryoml.regions import SECTION_EDGES, section_rrms  # noqa: E402
from cryoml.utils import device_tag, get_logger  # noqa: E402

logger = get_logger("pdk_adaptive_boundary_probe")
EDGES = SECTION_EDGES


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# Adaptive parameter-bound probe", "",
        "A bound expansion is recommended only when the same parameter, "
        "polarity, and direction improves at least the configured number of "
        "distinct devices without worsening any admitted sweep section by "
        "more than the guard.", "",
        "| Parameter | Polarity | Direction | Boundary devices | Improving "
        "devices | Recommend | Supported multiplier | Best objective delta | "
        "Worst section delta |",
        "|---|---|---|---:|---:|:---:|---:|---:|---:|",
    ]
    for row in payload["recommendations"]:
        best = row["best_probe"]
        supported = row.get("supported_multiplier")
        lines.append(
            f"| {row['parameter']} | {row['dev_type']} | {row['direction']} "
            f"| {row['n_boundary_devices']} | {row['n_improving_devices']} | "
            f"{'yes' if row['recommend_expansion'] else 'no'} | "
            f"{'n/a' if supported is None else f'{supported:g}'} | "
            + (f"{best['objective_delta']:.6g} | "
               f"{best['worst_section_delta']:.6g} |" if best
               else "n/a | n/a |"))
    lines.extend(["", "## Probe counts", "",
                  f"- Requested: {payload['n_jobs']}",
                  f"- Finite: {payload['n_finite']}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def make_plot(payload: dict, path: Path) -> None:
    rows = [row for row in payload["recommendations"]
            if row["best_probe"] is not None]
    if not rows:
        return
    labels = [f"{row['parameter']}\n{row['dev_type']} {row['direction'][0]}"
              for row in rows]
    deltas = [row["best_probe"]["objective_delta"] for row in rows]
    guards = [row["best_probe"]["worst_section_delta"] for row in rows]
    colors = ["#54a24b" if row["recommend_expansion"] else "#9d9da1"
              for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(max(12, 0.55 * len(rows)), 8),
                             sharex=True, layout="constrained")
    axes[0].bar(x, deltas, color=colors)
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_ylabel("best section-objective delta")
    axes[0].set_title("Outward physical bound probes (lower is better)")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(x, guards, color=colors)
    axes[1].axhline(0.02, color="#e45756", linestyle="--",
                    label="section regression guard")
    axes[1].set_ylabel("worst section RRMS delta")
    axes[1].set_xticks(x, labels, rotation=55, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def tag_curve(curve) -> str:
    return f"{curve.kind}@{abs(float(curve.fixed)):g}"


def section_score(device, curves, simulations,
                  include_tags: set[str]) -> dict:
    curve_rows = {}
    for curve, simulated in zip(curves, simulations):
        tag = tag_curve(curve)
        if tag not in include_tags:
            continue
        start, cleaned, denominator = curve_trim(
            device.dev_type, device.L_um, device.W_um,
            curve.kind, curve.Id)
        if denominator <= 0:
            continue
        target = np.asarray(cleaned[start:], dtype=np.float64)
        prediction = np.asarray(simulated, dtype=np.float64)[start:]
        count = min(len(target), len(prediction))
        target, prediction = target[:count], prediction[:count]
        values = section_rrms(prediction, target, denominator, EDGES)
        curve_rows[tag] = {
            "sections": values,
            "objective": float(0.75 * np.mean(values) + 0.25 * max(values)),
        }
    objectives = [row["objective"] for row in curve_rows.values()]
    flat = [value for row in curve_rows.values() for value in row["sections"]]
    paper = score_device_new(
        device.dev_type, device.L_um, device.W_um, curves, simulations,
        include_tags=include_tags)
    return {
        "objective": float(np.mean(objectives)),
        "mean_section_rrms": float(np.mean(flat)),
        "worst_section_rrms": float(np.max(flat)),
        "reported_rrms": float(paper["rrms"]),
        "curves": curve_rows,
    }


def simulate_probe(job) -> dict:
    (device, bin_index, base_params, parameter, value, multiplier,
     direction, include_tags) = job
    curves = load_device_curves(device)
    params = dict(base_params)
    params[parameter] = float(value)
    t0 = time.time()
    try:
        simulations = simulate_pdk(
            device.dev_type, device.L_um, device.W_um, curves,
            params=params, bin_index=bin_index)
        finite = all(np.isfinite(np.asarray(values)).all()
                     for values in simulations)
        score = (section_score(device, curves, simulations, set(include_tags))
                 if finite else None)
        error = None
    except Exception as exc:  # preserve the failed physical probe in output
        finite, score, error = False, None, repr(exc)
    return {
        "device": device_tag(device.dev_type, device.L_um, device.W_um),
        "dev_type": device.dev_type, "parameter": parameter,
        "direction": direction, "multiplier": multiplier,
        "value": float(value), "finite": finite, "score": score,
        "error": error, "runtime_s": round(time.time() - t0, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--foundation", type=Path,
        default=Path("out/pdk_foundation_vb_full/summary.json"))
    parser.add_argument(
        "--bounds", type=Path,
        default=Path("out/pdk_parameter_sensitivity/summary.json"))
    parser.add_argument(
        "--bound-diagnostic", type=Path,
        default=Path("out/tables/foundation_bound_diagnostic.json"))
    parser.add_argument(
        "--baseline", type=Path,
        default=Path("out/pdk_baseline/pdk_baseline.json"))
    parser.add_argument(
        "--sim-dir", type=Path,
        default=Path("out/pdk_foundation_vb_full"))
    parser.add_argument("--method", default="surrogate_fd",
                        help="method key holding the selected card to probe")
    parser.add_argument("--multipliers", default="2,4")
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--out", type=Path,
        default=Path("out/tables/adaptive_boundary_probe.json"))
    parser.add_argument(
        "--markdown", type=Path,
        default=Path("out/tables/adaptive_boundary_probe.md"))
    parser.add_argument(
        "--plot", type=Path,
        default=Path("out/figs/foundation_comparison/adaptive_boundary_probe.png"))
    args = parser.parse_args()
    multipliers = tuple(float(value) for value in args.multipliers.split(","))
    if not multipliers or any(value <= 1 for value in multipliers):
        parser.error("--multipliers must contain values greater than one")

    foundation = json.loads(args.foundation.read_text())
    bounds = json.loads(args.bounds.read_text())
    diagnostic = json.loads(args.bound_diagnostic.read_text())
    baseline = json.loads(args.baseline.read_text())
    include = {
        tag: [curve for curve, row in values.items() if row.get("included")]
        for tag, values in baseline["per_curve"].items()
    }
    bins = {row["device"]: int(row["bin_index"])
            for row in baseline["devices"]}
    devices = {device_tag(row.dev_type, row.L_um, row.W_um): row
               for row in PAPER_DEVICES}
    records = {row["device"]: row for row in foundation["devices"]}
    repeated = {row["parameter"] for row in diagnostic["parameters"]
                if row["total"] >= args.min_hits}

    base_scores = {}
    for tag, record in records.items():
        device = devices[tag]
        curves = load_device_curves(device)
        with np.load(args.sim_dir / f"sims_{tag}.npz") as data:
            simulations = [np.asarray(data[f"fd_sim_{index}"])
                           for index in range(len(curves))]
        base_scores[tag] = section_score(
            device, curves, simulations, set(include[tag]))

    jobs = []
    for hit in diagnostic["hits"]:
        parameter = hit["parameter"]
        if parameter not in repeated:
            continue
        tag, direction = hit["device"], hit["direction"]
        center = float(bounds["published_by_device"][tag][parameter])
        lo, hi = (float(value) for value in
                  bounds["bounds_by_device"][tag][parameter])
        span = hi - center if direction == "upper" else center - lo
        if span <= 0:
            continue
        sign = 1.0 if direction == "upper" else -1.0
        record = records[tag]
        params = record["methods"][args.method]["params"]
        for multiplier in multipliers:
            value = center + sign * multiplier * span
            jobs.append((
                devices[tag], bins[tag],
                params, parameter, value, multiplier, direction, include[tag]))

    if args.workers == 1:
        probes = [simulate_probe(job) for job in jobs]
    else:
        with get_context("spawn").Pool(args.workers) as pool:
            probes = list(pool.imap_unordered(simulate_probe, jobs))
    for row in probes:
        base = base_scores[row["device"]]
        row["baseline"] = base
        if row["score"] is None:
            row["objective_delta"] = None
            row["improved"] = False
            continue
        row["objective_delta"] = (
            row["score"]["objective"] - base["objective"])
        base_curve = base["curves"]
        candidate_curve = row["score"]["curves"]
        section_deltas = []
        for curve, values in candidate_curve.items():
            for candidate, reference in zip(
                    values["sections"], base_curve[curve]["sections"]):
                section_deltas.append(candidate - reference)
        row["n_sections_improved"] = int(np.sum(np.asarray(section_deltas) < 0))
        row["n_sections_worsened"] = int(np.sum(np.asarray(section_deltas) > 0))
        row["worst_section_delta"] = float(max(section_deltas))
        row["improved"] = bool(
            row["objective_delta"] < -1e-4
            and row["worst_section_delta"] <= 0.02)

    groups = defaultdict(list)
    for row in probes:
        groups[(row["parameter"], row["dev_type"], row["direction"])].append(row)
    recommendations = []
    for (parameter, family, direction), rows in groups.items():
        improving_devices = sorted({row["device"] for row in rows
                                    if row["improved"]})
        best = min(
            [row for row in rows if row["score"] is not None],
            key=lambda row: row["objective_delta"], default=None)
        # An expansion is only supported at a multiplier that was actually
        # probed at that size: enough distinct devices must improve there and
        # no probed device of the same polarity may regress there.  Using the
        # best objective delta alone would import a rejected larger probe.
        levels = []
        for multiplier in sorted({row["multiplier"] for row in rows}):
            level = [row for row in rows if row["multiplier"] == multiplier]
            improving = sorted({row["device"] for row in level
                                if row["improved"]})
            regressing = sorted({
                row["device"] for row in level
                if row["score"] is None
                or row["objective_delta"] > 1e-4
                or row["worst_section_delta"] > 0.02})
            levels.append({
                "multiplier": float(multiplier),
                "n_improving_devices": len(improving),
                "improving_devices": improving,
                "n_regressing_devices": len(regressing),
                "regressing_devices": regressing,
                "supported": bool(len(improving) >= args.min_hits
                                  and not regressing),
            })
        supported = [row["multiplier"] for row in levels if row["supported"]]
        recommendations.append({
            "parameter": parameter, "dev_type": family,
            "direction": direction,
            "n_boundary_devices": len({row["device"] for row in rows}),
            "n_improving_devices": len(improving_devices),
            "improving_devices": improving_devices,
            "levels": levels,
            "supported_multiplier": max(supported) if supported else None,
            "recommend_expansion": bool(supported),
            "improving_devices_any_multiplier_only": bool(
                len(improving_devices) >= args.min_hits and not supported),
            "best_probe": ({key: best[key] for key in (
                "device", "multiplier", "value", "objective_delta",
                "worst_section_delta")} if best else None),
        })
    recommendations.sort(key=lambda row: (
        not row["recommend_expansion"], -row["n_improving_devices"],
        row["parameter"], row["dev_type"], row["direction"]))
    payload = {
        "definition": {
            "multipliers": list(multipliers), "min_hits": args.min_hits,
            "objective": (
                "mean over admitted curves of 75% mean four section RRMS + "
                "25% worst section RRMS"),
            "improvement_rule": (
                "objective delta < -1e-4 and no section worsens by >0.02"),
            "recommendation_rule": (
                "at one probed multiplier, at least min_hits distinct devices "
                "improve in the same parameter/polarity/direction group and "
                "no probed device of that polarity regresses; the largest "
                "such multiplier is the supported expansion"),
        },
        "source": {"foundation": str(args.foundation),
                   "method": args.method, "sim_dir": str(args.sim_dir),
                   "bound_diagnostic": str(args.bound_diagnostic)},
        "n_jobs": len(jobs), "n_finite": sum(row["finite"] for row in probes),
        "recommendations": recommendations, "probes": probes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    write_markdown(payload, args.markdown)
    make_plot(payload, args.plot)
    logger.info("wrote %s (%d probes)", args.out, len(probes))
    print(json.dumps({
        "n_jobs": payload["n_jobs"], "n_finite": payload["n_finite"],
        "recommended": [row for row in recommendations
                         if row["recommend_expansion"]],
    }, indent=2))


if __name__ == "__main__":
    main()
