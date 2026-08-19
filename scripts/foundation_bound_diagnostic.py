#!/usr/bin/env python3
"""Audit parameter-bound saturation after expanded foundation extraction."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def resolve_names(summary: dict, manifest_path: Path | None) -> list[str]:
    """Return the parameter-name order matching each record's ``local`` list."""
    if "parameter_names" in summary:
        return list(summary["parameter_names"])
    if manifest_path is None or not manifest_path.exists():
        raise SystemExit(
            "summary has no parameter_names; pass --manifest with the cache "
            "manifest that defines the parameter order")
    names = list(json.loads(manifest_path.read_text())["parameter_names"])
    width = len(summary["devices"][0]["methods"][
        next(iter(summary["devices"][0]["methods"]))]["local"])
    if len(names) != width:
        raise SystemExit(
            f"manifest defines {len(names)} parameters but the summary "
            f"stores {width} local coordinates")
    return names


def build_report(summary_path: Path, sensitivity_path: Path,
                 threshold: float, method_name: str = "surrogate_fd",
                 manifest_path: Path | None = None) -> dict:
    summary = json.loads(summary_path.read_text())
    sensitivity = json.loads(sensitivity_path.read_text())
    names = resolve_names(summary, manifest_path)
    counts = {name: defaultdict(int) for name in names}
    hits, devices = [], []
    for record in summary["devices"]:
        tag = record["device"]
        family = record["dev_type"]
        method = record["methods"][method_name]
        local = np.asarray(method["local"], dtype=np.float64)
        device_hits = []
        for name, value in zip(names, local):
            if abs(value) < threshold:
                continue
            direction = "upper" if value > 0 else "lower"
            counts[name]["total"] += 1
            counts[name][family] += 1
            counts[name][direction] += 1
            device_hits.append(name)
            hits.append({
                "device": tag, "dev_type": family, "parameter": name,
                "local": float(value), "direction": direction,
                "value": float(method["params"][name]),
                "bounds": sensitivity["bounds_by_device"][tag][name],
            })
        devices.append({"device": tag, "dev_type": family,
                        "n_bound_hits": len(device_hits),
                        "parameters": device_hits})
    parameter_rows = []
    for name in names:
        row = counts[name]
        parameter_rows.append({
            "parameter": name, "total": row["total"],
            "nmos": row["nmos"], "pmos": row["pmos"],
            "lower": row["lower"], "upper": row["upper"],
            "repeated_boundary_limited": row["total"] >= 2,
        })
    parameter_rows.sort(key=lambda row: (-row["total"], row["parameter"]))

    family_tags = {
        family: [tag for tag in sensitivity["devices"]
                 if tag.startswith(family)]
        for family in ("nmos", "pmos")
    }
    inactive = {}
    for family, tags in family_tags.items():
        inactive[family] = [
            name for name in names
            if all(sensitivity["bounds_by_device"][tag][name][0]
                   == sensitivity["bounds_by_device"][tag][name][1]
                   for tag in tags)
        ]
    key = next(row for row in summary["devices"]
               if row["device"] == "nmos_L1_W3")
    key_method = key["methods"][method_name]
    key_hits = [row for row in hits if row["device"] == "nmos_L1_W3"]
    return {
        "definition": {
            "source": str(summary_path), "method": method_name,
            "threshold": threshold,
            "meaning": (
                "|local coordinate| at or above threshold in the selected "
                "real-NGSpice-validated foundation FD card"),
        },
        "n_devices": len(devices), "n_parameters": len(names),
        "n_total_hits": len(hits),
        "parameters": parameter_rows, "devices": devices, "hits": hits,
        "inactive_selected_parameters_by_family": inactive,
        "key_nmos_L1_W3": {
            "rrms": key_method["rrms"],
            "high_voltage_rrms": key_method["high_voltage_rrms"],
            "bound_hits": key_hits,
        },
        "interpretation": {
            "boundary_hit_is_not_proof": (
                "expand only after outward physical probes improve multiple "
                "independent sweep sections; widespread hits can also signal "
                "non-identifiability"),
            "zero_family_terms": (
                "a globally selected parameter published as zero throughout "
                "one polarity must receive an absolute cross-polarity "
                "sensitivity probe before being fixed inactive"),
        },
    }


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Foundation parameter-bound diagnostic", "",
        f"Threshold: `|local| >= {report['definition']['threshold']}`. "
        f"Observed {report['n_total_hits']} hits across "
        f"{report['n_devices']} devices and {report['n_parameters']} "
        "parameters.", "",
        "A hit is a diagnostic, not automatic permission to widen a bound. "
        "Outward NGSpice probes must improve multiple independent sweep "
        "sections before adaptive data are generated.", "",
        "| Parameter | Total | nMOS | pMOS | Lower | Upper | Repeated |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in report["parameters"]:
        lines.append(
            f"| {row['parameter']} | {row['total']} | {row['nmos']} | "
            f"{row['pmos']} | {row['lower']} | {row['upper']} | "
            f"{'yes' if row['repeated_boundary_limited'] else 'no'} |")
    lines.extend(["", "## Selected parameters inactive across a polarity", ""])
    for family, names in report["inactive_selected_parameters_by_family"].items():
        lines.append(f"- {family}: " + ", ".join(f"`{name}`" for name in names))
    lines.extend(["", "## nMOS L1/W3 boundary hits", "",
                  "| Parameter | Direction | Value | Bounds | Local |",
                  "|---|---|---:|---|---:|"])
    for row in report["key_nmos_L1_W3"]["bound_hits"]:
        lines.append(
            f"| {row['parameter']} | {row['direction']} | {row['value']:.6g} "
            f"| {row['bounds']} | {row['local']:.5f} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary", type=Path,
        default=Path("out/pdk_foundation_vb_full/summary.json"))
    parser.add_argument(
        "--sensitivity", type=Path,
        default=Path("out/pdk_parameter_sensitivity/summary.json"))
    parser.add_argument("--method", default="surrogate_fd",
                        help="method key inside each summary device record")
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("data/processed/pdk_synth_full/manifest.json"),
        help="cache manifest defining parameter order when the summary "
             "does not record parameter_names")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument(
        "--json", type=Path,
        default=Path("out/tables/foundation_bound_diagnostic.json"))
    parser.add_argument(
        "--markdown", type=Path,
        default=Path("out/tables/foundation_bound_diagnostic.md"))
    args = parser.parse_args()
    if not 0 < args.threshold <= 1:
        parser.error("--threshold must be in (0, 1]")
    report = build_report(args.summary, args.sensitivity,
                          args.threshold, args.method,
                          args.manifest)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.markdown)
    print(json.dumps({
        "n_total_hits": report["n_total_hits"],
        "top_parameters": report["parameters"][:10],
        "inactive": report["inactive_selected_parameters_by_family"],
    }, indent=2))


if __name__ == "__main__":
    main()
