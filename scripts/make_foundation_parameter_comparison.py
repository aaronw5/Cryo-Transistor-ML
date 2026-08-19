#!/usr/bin/env python3
"""Compare the paper cards, seven-parameter, and expanded foundation runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.config import OUT_DIR, OUT_TABLES, ensure_dirs  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.metrics import curve_trim, family_totals, score_device_new  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402


def baseline_records() -> dict[str, dict]:
    raw = json.loads(
        (OUT_DIR / "pdk_baseline" / "pdk_baseline.json").read_text())
    by_tag = {row["device"]: row for row in raw["devices"]}
    records = {}
    for device in PAPER_DEVICES:
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        row = by_tag[tag]
        per_curve = raw["per_curve"][tag]
        curves = load_device_curves(device)
        saved = np.load(OUT_DIR / "pdk_baseline" / f"sims_{tag}.npz")
        sims = [np.asarray(saved[f"sim_{index}"])
                for index in range(len(curves))]
        include_tags = {curve_tag for curve_tag, value in per_curve.items()
                        if value.get("included")}
        rescored = score_device_new(
            device.dev_type, device.L_um, device.W_um, curves, sims,
            include_tags=include_tags)
        high_tags = set()
        for curve in curves:
            bias = abs(float(curve.fixed))
            if bias < 1.48 - 1e-6:
                continue
            _start, _cleaned, denominator = curve_trim(
                device.dev_type, device.L_um, device.W_um, curve.kind,
                curve.Id)
            if denominator > 0 and np.isfinite(denominator):
                high_tags.add(f"{curve.kind}@{bias:g}")
        high_scored = score_device_new(
            device.dev_type, device.L_um, device.W_um, curves, sims,
            include_tags=high_tags)
        records[tag] = {
            "rrms": float(row["rrms"]), "sigma": float(row["sigma"]),
            "rmse_uA": float(rescored["rmse_uA"]),
            "high_voltage_rrms": float(high_scored["rrms"]),
            "per_curve": per_curve,
            "high_voltage_per_curve": high_scored["per_curve"],
        }
    return records


def load_study(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"missing completed study {path}")
    summary = json.loads(path.read_text())
    expected = {device_tag(d.dev_type, d.L_um, d.W_um)
                for d in PAPER_DEVICES}
    actual = {row["device"] for row in summary["devices"]}
    if actual != expected:
        raise RuntimeError(
            f"{path}: expected all {len(expected)} paper devices, got "
            f"{len(actual)}")
    return summary


def aggregate(rows: dict[str, dict]) -> dict:
    totals = family_totals(rows)
    totals["all_device_mean_rrms"] = float(np.mean(
        [row["rrms"] for row in rows.values()]))
    totals["all_device_mean_high_voltage_rrms"] = float(np.mean(
        [row["high_voltage_rrms"] for row in rows.values()
         if np.isfinite(row["high_voltage_rrms"])]))
    return totals


def method_rows(summary: dict, method: str) -> dict[str, dict]:
    return {row["device"]: row["methods"][method]
            for row in summary["devices"]}


def per_curve_regressions(reference: dict[str, dict],
                          candidate: dict[str, dict], limit: int = 20) -> list[dict]:
    rows = []
    for tag, candidate_device in candidate.items():
        reference_curves = reference[tag]["high_voltage_per_curve"]
        candidate_curves = candidate_device["high_voltage_per_curve"]
        for curve_tag, candidate_curve in candidate_curves.items():
            if not candidate_curve.get("included"):
                continue
            candidate_rrms = candidate_curve.get("rrms")
            reference_rrms = reference_curves.get(curve_tag, {}).get("rrms")
            if candidate_rrms is None or reference_rrms is None:
                continue
            bias = float(curve_tag.split("@")[1])
            if bias < 1.48 - 1e-6:
                continue
            rows.append({
                "device": tag, "curve": curve_tag,
                "paper_rrms": float(reference_rrms),
                "candidate_rrms": float(candidate_rrms),
                "delta": float(candidate_rrms - reference_rrms),
                "ratio": float(candidate_rrms / max(reference_rrms, 1e-15)),
            })
    return sorted(rows, key=lambda row: row["delta"], reverse=True)[:limit]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seven", default=str(OUT_DIR / "pdk_foundation_vb_seven" /
                               "summary.json"))
    parser.add_argument(
        "--full", default=str(OUT_DIR / "pdk_foundation_vb_full" /
                              "summary.json"))
    parser.add_argument(
        "--out", default=str(OUT_TABLES /
                             "foundation_parameter_comparison.json"))
    args = parser.parse_args()
    ensure_dirs()
    seven, full = load_study(Path(args.seven)), load_study(Path(args.full))
    paper = baseline_records()
    series = {
        "paper_cards": paper,
        "seven_surrogate_fd": method_rows(seven, "surrogate_fd"),
        "seven_inverse_autoencoder": method_rows(
            seven, "inverse_autoencoder"),
        "full_surrogate_fd": method_rows(full, "surrogate_fd"),
        "full_inverse_autoencoder": method_rows(
            full, "inverse_autoencoder"),
    }
    comparison = {
        "parameter_sets": {
            "seven": seven["parameter_names"],
            "full": full["parameter_names"],
        },
        "series": {name: aggregate(rows) for name, rows in series.items()},
        "worst_high_voltage_regressions_vs_paper": {
            name: per_curve_regressions(paper, rows)
            for name, rows in series.items() if name != "paper_cards"
        },
        "devices": [{
            "device": tag,
            **{name: {
                "rrms": float(rows[tag]["rrms"]),
                "sigma": float(rows[tag]["sigma"]),
                "rmse_uA": float(rows[tag]["rmse_uA"]),
                "high_voltage_rrms": float(rows[tag]["high_voltage_rrms"]),
            } for name, rows in series.items()},
        } for tag in paper],
        "diagnostic": {
            "seven_forward_test": seven["training"]["test"],
            "full_forward_test": full["training"]["test"],
            "seven_forward_test_by_curve": seven["training"][
                "test_by_curve"],
            "full_forward_test_by_curve": full["training"][
                "test_by_curve"],
            "seven_inverse_test": seven["inverse_training"]["test"],
            "full_inverse_test": full["inverse_training"]["test"],
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(comparison, indent=2) + "\n")

    lines = [
        "# Foundation parameter-count comparison — paper Table 6 format", "",
        "All extracted parameter vectors were re-simulated in NGSpice 41. "
        "Paper RRMS and its frozen paper-card inclusion set are unchanged; "
        "the high-voltage column is a separate mean over 1.48/1.85 V curves.",
        "", "| series | nMOS RRMS | pMOS RRMS | combined RRMS | "
        "all-device RRMS | high-voltage RRMS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in comparison["series"].items():
        lines.append(
            f"| {name.replace('_', ' ')} | {row['nmos_rrms']:.4f} | "
            f"{row['pmos_rrms']:.4f} | {row['combined_rrms']:.4f} | "
            f"{row['all_device_mean_rrms']:.4f} | "
            f"{row['all_device_mean_high_voltage_rrms']:.4f} |")
    lines += ["", "## Per-device card comparison", ""]
    for name in series:
        lines += [f"### {name.replace('_', ' ')}", "",
                  "| Transistor Type | Length (um) | Width (um) | "
                  "RMSE (uA) | RRMS | sigma_RRMS | high-voltage RRMS |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for device in PAPER_DEVICES:
            tag = device_tag(device.dev_type, device.L_um, device.W_um)
            row = series[name][tag]
            lines.append(
                f"| {'nMOS' if device.dev_type == 'nmos' else 'pMOS'} | "
                f"{device.L_um:g} | {device.W_um:g} | "
                f"{row['rmse_uA']:.3g} | {row['rrms']:.3f} | "
                f"{row['sigma']:.3f} | {row['high_voltage_rrms']:.3f} |")
        lines.append("")
    out_path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(comparison["series"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
