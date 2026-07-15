#!/usr/bin/env python3
"""Select high-voltage-preserving cards without redefining paper RRMS.

For every device, this diagnostic protects the strongest included output and
transfer curves. A candidate is feasible when each protected per-curve RRMS is
no greater than both a relative/absolute allowance around the published-card
error. Among feasible candidates in the existing 10,000-point NGSpice LHC, the
card with the lowest *unchanged official paper RRMS* is selected and rerun in
NGSpice.

This is a selection constraint, not a new metric. The canonical extraction and
paper RRMS reports remain unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.config import OUT_DIR, OUT_TABLES, PROCESSED_DIR, ensure_dirs  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.metrics import curve_trim, score_device_new  # noqa: E402
from cryoml.pdk_extract import PARAMS7  # noqa: E402
from cryoml.spice_pdk import simulate_pdk  # noqa: E402
from cryoml.utils import device_tag, get_logger  # noqa: E402

logger = get_logger("high_voltage_guarded_study")
DEFAULT_OUT = OUT_DIR / "pdk_high_voltage_guarded"


def curve_tag(curve) -> str:
    return f"{curve.kind}@{abs(curve.fixed):g}"


def curve_rrms(device, curve, simulation: np.ndarray) -> float:
    start, measured, denominator = curve_trim(
        device.dev_type, device.L_um, device.W_um, curve.kind, curve.Id)
    if denominator <= 0 or len(measured) <= start:
        return float("nan")
    simulation = np.asarray(simulation, dtype=np.float64)[start:]
    measured = measured[start:]
    count = min(len(simulation), len(measured))
    return float(np.sqrt(np.mean(
        (simulation[:count] - measured[:count]) ** 2)) / denominator)


def load_sims(path: Path, prefix: str, count: int) -> list[np.ndarray]:
    with np.load(path) as saved:
        return [np.asarray(saved[f"{prefix}{index}"])
                for index in range(count)]


def score_all(device, curves, simulations, include_tags) -> dict:
    score = score_device_new(
        device.dev_type, device.L_um, device.W_um, curves, simulations,
        include_tags=include_tags)
    return {
        "rrms": float(score["rrms"]),
        "sigma": float(score["sigma"]),
        "rmse_uA": float(score["rmse_uA"]),
        "per_curve": {
            curve_tag(curve): curve_rrms(device, curve, simulations[index])
            for index, curve in enumerate(curves)
            if curve_tag(curve) in include_tags
        },
    }


def run_device(device, include_tags: set[str], ratio: float,
               absolute: float, out_dir: Path) -> dict:
    tag = device_tag(device.dev_type, device.L_um, device.W_um)
    curves = load_device_curves(device)
    data = np.load(PROCESSED_DIR / "pdk_synth" / f"{tag}.npz",
                   allow_pickle=True)
    ids = np.asarray(data["IDS"])
    theta = np.asarray(data["THETA"])
    slices = np.asarray(data["slices"])
    published = {parameter: float(value)
                 for parameter, value in zip(PARAMS7, data["published"])}
    bin_index = int(data["bin_index"])
    box_mode = str(data["box_mode"])
    if box_mode != "lhc10":
        raise RuntimeError(f"{tag}: expected lhc10 data")

    paper_sims = load_sims(
        OUT_DIR / "pdk_baseline" / f"sims_{tag}.npz", "sim_", len(curves))
    foundation_sims = load_sims(
        OUT_DIR / "pdk_foundation_emu" / f"sims_{tag}.npz", "fd_sim_",
        len(curves))

    included_indices = [index for index, curve in enumerate(curves)
                        if curve_tag(curve) in include_tags]
    columns = []
    paper_values = []
    for index in included_indices:
        curve = curves[index]
        start, measured, denominator = curve_trim(
            device.dev_type, device.L_um, device.W_um, curve.kind, curve.Id)
        begin, end = map(int, slices[index])
        columns.append(
            np.sqrt(np.mean(
                (ids[:, begin + start:end] - measured[start:]) ** 2,
                axis=1)) / denominator)
        paper_values.append(curve_rrms(device, curve, paper_sims[index]))
    candidate_rrms = np.stack(columns, axis=1)
    paper_values = np.asarray(paper_values)
    official = np.mean(candidate_rrms, axis=1)

    protected_columns = []
    for kind in ("idvd", "idvg"):
        available = [(column, curves[index])
                     for column, index in enumerate(included_indices)
                     if curves[index].kind == kind]
        if available:
            protected_columns.append(max(
                available, key=lambda item: abs(item[1].fixed))[0])
    limits = {
        column: max(ratio * paper_values[column],
                    paper_values[column] + absolute)
        for column in protected_columns
    }
    feasible = np.ones(len(candidate_rrms), dtype=bool)
    for column, limit in limits.items():
        feasible &= candidate_rrms[:, column] <= limit
    feasible_indices = np.flatnonzero(feasible)
    if len(feasible_indices):
        winner = int(feasible_indices[
            np.argmin(official[feasible_indices])])
        constraint_satisfied = True
    else:
        violation = np.max(np.stack([
            candidate_rrms[:, column] / max(limit, 1e-12)
            for column, limit in limits.items()
        ], axis=1), axis=1)
        best_violation = np.min(violation)
        closest = np.flatnonzero(np.isclose(violation, best_violation))
        winner = int(closest[np.argmin(official[closest])])
        constraint_satisfied = False

    params = {parameter: float(theta[winner, parameter_index])
              for parameter_index, parameter in enumerate(PARAMS7)}
    guarded_sims = simulate_pdk(
        device.dev_type, device.L_um, device.W_um, curves, params=params,
        bin_index=bin_index)
    scores = {
        "paper": score_all(device, curves, paper_sims, include_tags),
        "foundation_plus_fd": score_all(
            device, curves, foundation_sims, include_tags),
        "high_voltage_guarded": score_all(
            device, curves, guarded_sims, include_tags),
    }
    protected = {}
    for column in protected_columns:
        tag_name = curve_tag(curves[included_indices[column]])
        protected[tag_name] = {
            "paper_rrms": float(paper_values[column]),
            "limit": float(limits[column]),
            "foundation_rrms": scores["foundation_plus_fd"]["per_curve"][tag_name],
            "guarded_rrms": scores["high_voltage_guarded"]["per_curve"][tag_name],
        }

    np.savez(out_dir / f"sims_{tag}.npz",
             **{f"sim_{index}": np.asarray(simulation)
                for index, simulation in enumerate(guarded_sims)})
    record = {
        "device": tag,
        "dev_type": device.dev_type,
        "L_um": device.L_um,
        "W_um": device.W_um,
        "bin_index": bin_index,
        "box_mode": box_mode,
        "method": "high_voltage_guarded_lhc",
        "selection_policy": (
            "minimize unchanged official RRMS subject to strongest output "
            "and transfer per-curve error limits"),
        "guard": {"relative_to_paper": ratio, "absolute_rrms": absolute,
                  "constraint_satisfied": constraint_satisfied,
                  "feasible_dataset_candidates": int(len(feasible_indices))},
        "protected_curves": protected,
        "dataset_index": winner,
        "params": params,
        "published_params": published,
        "scores": scores,
        "dataset_official_rrms": float(official[winner]),
        "fresh_ngspice_rrms_delta": float(abs(
            official[winner] - scores["high_voltage_guarded"]["rrms"])),
    }
    (out_dir / f"ml_{tag}.json").write_text(
        json.dumps(record, indent=2) + "\n")
    logger.info(
        "%-22s feasible=%4d paper=%.4f foundation=%.4f guarded=%.4f",
        tag, len(feasible_indices), scores["paper"]["rrms"],
        scores["foundation_plus_fd"]["rrms"],
        scores["high_voltage_guarded"]["rrms"])
    return record


def family_summary(records: list[dict], method: str) -> dict[str, float]:
    nmos = [record["scores"][method]["rrms"] for record in records
            if record["dev_type"] == "nmos"]
    pmos = [record["scores"][method]["rrms"] for record in records
            if record["dev_type"] == "pmos"]
    return {
        "nmos_mean": float(np.mean(nmos)),
        "pmos_mean": float(np.mean(pmos)),
        "combined": float((np.mean(nmos) + np.mean(pmos)) / 2.0),
        "all_device_mean": float(np.mean(nmos + pmos)),
    }


def write_report(records: list[dict], ratio: float, absolute: float,
                 runtime_s: float) -> dict:
    summary = {
        method: family_summary(records, method)
        for method in ("paper", "foundation_plus_fd",
                       "high_voltage_guarded")
    }
    summary.update({
        "constraint": {
            "protected_curves": "strongest included idvd and idvg",
            "relative_to_paper": ratio,
            "absolute_rrms": absolute,
        },
        "devices_with_feasible_candidate": int(sum(
            record["guard"]["constraint_satisfied"] for record in records)),
        "runtime_s": runtime_s,
    })
    payload = {"summary": summary, "devices": records}
    (OUT_TABLES / "high_voltage_guarded_study.json").write_text(
        json.dumps(payload, indent=2) + "\n")

    lines = [
        "# High-voltage-preserving selection study", "",
        "RRMS is unchanged from the paper. This diagnostic protects the "
        "strongest included output and transfer curves, then selects the "
        "lowest official RRMS among feasible candidates in the 10,000-point "
        "NGSpice LHC. A protected curve may be at most "
        f"`max({ratio:g} * paper curve RRMS, paper curve RRMS + "
        f"{absolute:g})`.", "",
        "| method | all-device mean | combined | nMOS | pMOS |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "paper": "paper card",
        "foundation_plus_fd": "foundation + FD",
        "high_voltage_guarded": "high-voltage guarded",
    }
    for method, label in labels.items():
        item = summary[method]
        lines.append(
            f"| {label} | {item['all_device_mean']:.4f} | "
            f"{item['combined']:.4f} | {item['nmos_mean']:.4f} | "
            f"{item['pmos_mean']:.4f} |")
    lines += [
        "", "## Per Device", "",
        "| device | feasible candidates | paper RRMS | foundation + FD | "
        "guarded RRMS | protected paper -> foundation -> guarded |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for record in records:
        protected = "; ".join(
            f"{tag}: {item['paper_rrms']:.3f} -> "
            f"{item['foundation_rrms']:.3f} -> {item['guarded_rrms']:.3f}"
            for tag, item in record["protected_curves"].items())
        scores = record["scores"]
        lines.append(
            f"| {record['device']} | "
            f"{record['guard']['feasible_dataset_candidates']} | "
            f"{scores['paper']['rrms']:.4f} | "
            f"{scores['foundation_plus_fd']['rrms']:.4f} | "
            f"{scores['high_voltage_guarded']['rrms']:.4f} | {protected} |")
    lines += [
        "", "This guard is a selection policy, not a replacement metric. "
        "The guarded cards are diagnostic and are not used by the canonical "
        "card export.",
    ]
    (OUT_TABLES / "high_voltage_guarded_study.md").write_text(
        "\n".join(lines) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relative", type=float, default=1.5)
    parser.add_argument("--absolute", type=float, default=0.005)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    if args.relative < 1 or args.absolute < 0:
        raise ValueError("guard allowances must be nonnegative and relative >= 1")

    ensure_dirs()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(
        (OUT_DIR / "pdk_baseline" / "pdk_baseline.json").read_text())
    include_by_tag = {
        tag: {curve_tag for curve_tag, item in curves.items()
              if item.get("included")}
        for tag, curves in baseline["per_curve"].items()
    }
    started = time.time()
    records = []
    for device in PAPER_DEVICES:
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        records.append(run_device(
            device, include_by_tag[tag], args.relative, args.absolute,
            out_dir))
    runtime_s = round(time.time() - started, 2)
    summary = write_report(records, args.relative, args.absolute, runtime_s)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
