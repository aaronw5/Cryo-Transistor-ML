#!/usr/bin/env python3
"""Model-independent seven-vs-expanded Monte Carlo representation audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from foundation_parameter_study import (  # noqa: E402
    build_device_context, curve_objective_layout, curve_rrms_matrix,
    score_canonical)
from cryoml.config import OUT_TABLES, PROCESSED_DIR, ensure_dirs  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.foundation import geometry_ranges  # noqa: E402


def analyze_context(context) -> dict:
    paper_layout = curve_objective_layout(context)
    high_layout = curve_objective_layout(
        context, context.high_voltage_tags)
    paper_values = np.mean(
        curve_rrms_matrix(context.current_samples, paper_layout), axis=1)
    high_values = np.mean(
        curve_rrms_matrix(context.current_samples, high_layout), axis=1)
    paper_index = int(np.argmin(paper_values))
    high_index = int(np.argmin(high_values))
    published_index = int(np.argmin(np.mean(
        (context.local_samples - context.box.local_published) ** 2, axis=1)))

    def result(index: int) -> dict:
        score = score_canonical(context, context.current_samples[index])
        return {
            "sample_index": index, "rrms": float(score["rrms"]),
            "high_voltage_rrms": float(score["high_voltage_rrms"]),
            "local": context.local_samples[index].astype(float).tolist(),
        }

    return {
        "n_samples": int(len(context.current_samples)),
        "n_active_parameters": context.box.n_active,
        "published_row": result(published_index),
        "best_paper_rrms_sample": result(paper_index),
        "best_high_voltage_sample": result(high_index),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seven-dir", default=str(
        PROCESSED_DIR / "pdk_synth"))
    parser.add_argument("--full-dir", default=str(
        PROCESSED_DIR / "pdk_synth_full"))
    parser.add_argument("--out", default=str(
        OUT_TABLES / "parameter_surface_diagnostic.json"))
    args = parser.parse_args()
    ensure_dirs()
    metadata = {"geometry_ranges": geometry_ranges(PAPER_DEVICES)}
    records = []
    for device in PAPER_DEVICES:
        record = {
            "device": (f"{device.dev_type}_L{device.L_um:g}_W{device.W_um:g}"),
            "dev_type": device.dev_type, "L_um": device.L_um,
            "W_um": device.W_um, "parameter_sets": {},
        }
        for name, directory in (("seven", Path(args.seven_dir)),
                                ("full", Path(args.full_dir))):
            context = build_device_context(
                device, directory, metadata, name)
            record["device"] = context.tag
            record["parameter_sets"][name] = analyze_context(context)
        records.append(record)

    summary = {}
    for parameter_set in ("seven", "full"):
        best = [row["parameter_sets"][parameter_set][
            "best_paper_rrms_sample"] for row in records]
        summary[parameter_set] = {
            "all_device_mean_best_sample_rrms": float(np.mean(
                [row["rrms"] for row in best])),
            "all_device_mean_high_voltage_rrms_at_best_sample": float(np.mean(
                [row["high_voltage_rrms"] for row in best])),
            "n_samples_per_device": sorted(set(
                row["parameter_sets"][parameter_set]["n_samples"]
                for row in records)),
        }
    result = {
        "definition": (
            "Direct scoring of stored real-NGSpice LHC curves; no surrogate, "
            "training loss, or parameter optimization is involved. This "
            "per-device oracle is a nondeployable representation diagnostic, "
            "not a headline ML method or exported card series."),
        "summary": summary, "devices": records,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Seven-vs-expanded parameter-surface diagnostic", "",
        result["definition"], "",
        "| parameter set | samples/device | best-sample all-device RRMS | "
        "high-voltage RRMS at that sample |", "|---|---:|---:|---:|",
    ]
    for name, row in summary.items():
        lines.append(
            f"| {name} | {','.join(str(value) for value in row['n_samples_per_device'])} "
            f"| {row['all_device_mean_best_sample_rrms']:.5f} | "
            f"{row['all_device_mean_high_voltage_rrms_at_best_sample']:.5f} |")
    lines += ["", "## Per-device", "",
              "| device | 7D best | 43D best | 7D high | 43D high |",
              "|---|---:|---:|---:|---:|"]
    for record in records:
        seven = record["parameter_sets"]["seven"]["best_paper_rrms_sample"]
        full = record["parameter_sets"]["full"]["best_paper_rrms_sample"]
        lines.append(
            f"| {record['device']} | {seven['rrms']:.5f} | "
            f"{full['rrms']:.5f} | {seven['high_voltage_rrms']:.5f} | "
            f"{full['high_voltage_rrms']:.5f} |")
    out_path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
