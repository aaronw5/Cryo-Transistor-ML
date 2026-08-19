#!/usr/bin/env python3
"""Compare paper/foundation cards by independent x-axis RRMS sections."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.metrics import curve_trim  # noqa: E402
from cryoml.regions import SECTION_EDGES, section_spans  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402

EDGES = SECTION_EDGES
METHODS = {
    "paper": ("out/pdk_baseline", "sim"),
    "seven_fd": ("out/pdk_foundation_vb_seven", "fd_sim"),
    "seven_regional": (
        "out/pdk_foundation_segmented_seven_rrms_extract", "fd_sim"),
    "full43_fd": ("out/pdk_foundation_vb_full", "fd_sim"),
    "full43_tradeoff": (
        "out/pdk_foundation_noiseaware_full_tradeoff", "fd_sim"),
    "full43_regional": (
        "out/pdk_foundation_segmented_full_rrms_extract", "fd_sim"),
}


def curve_tag(curve) -> str:
    return f"{curve.kind}@{abs(float(curve.fixed)):g}"


def section_rows(simulated: np.ndarray, measured: np.ndarray,
                 start: int, denominator: float) -> list[dict]:
    simulated = np.asarray(simulated, dtype=np.float64)[start:]
    measured = np.asarray(measured, dtype=np.float64)[start:]
    count = min(len(simulated), len(measured))
    simulated, measured = simulated[:count], measured[:count]
    rows = []
    for index, ((lo, hi), (left, right)) in enumerate(
            zip(section_spans(count, EDGES), zip(EDGES[:-1], EDGES[1:]))):
        rrms = float(np.sqrt(np.mean(
            (simulated[lo:hi] - measured[lo:hi]) ** 2)) / denominator)
        rows.append({
            "section": index, "normalized_start": left,
            "normalized_stop": right, "n_points": hi - lo, "rrms": rrms,
        })
    return rows


def load_simulations(path: Path, prefix: str, count: int) -> list[np.ndarray]:
    with np.load(path) as data:
        return [np.asarray(data[f"{prefix}_{index}"], dtype=np.float64)
                for index in range(count)]


def build_report(root: Path) -> dict:
    baseline = json.loads(
        (root / "out/pdk_baseline/pdk_baseline.json").read_text())
    admitted = {
        tag: {curve for curve, row in values.items() if row.get("included")}
        for tag, values in baseline["per_curve"].items()
    }
    rows = []
    for device in PAPER_DEVICES:
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        curves = load_device_curves(device)
        method_sims = {}
        for method, (relative, prefix) in METHODS.items():
            path = root / relative / f"sims_{tag}.npz"
            if path.exists():
                method_sims[method] = load_simulations(
                    path, prefix, len(curves))
        for curve_index, curve in enumerate(curves):
            name = curve_tag(curve)
            if name not in admitted[tag]:
                continue
            start, cleaned, denominator = curve_trim(
                device.dev_type, device.L_um, device.W_um,
                curve.kind, curve.Id)
            if denominator <= 0:
                continue
            for method, simulations in method_sims.items():
                for section in section_rows(
                        simulations[curve_index], cleaned, start, denominator):
                    rows.append({
                        "device": tag, "dev_type": device.dev_type,
                        "kind": curve.kind,
                        "bias": abs(float(curve.fixed)), "curve": name,
                        "method": method, **section,
                    })

    methods = [method for method in METHODS
               if any(row["method"] == method for row in rows)]
    aggregate = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        by_section = {}
        for section in range(len(EDGES) - 1):
            section_rows_ = [row for row in method_rows
                             if row["section"] == section]
            device_values = []
            for tag in sorted({row["device"] for row in section_rows_}):
                values = [row["rrms"] for row in section_rows_
                          if row["device"] == tag]
                device_values.append(float(np.mean(values)))
            by_section[f"s{section}_{EDGES[section]:g}_{EDGES[section+1]:g}"] = (
                float(np.mean(device_values)))
        aggregate[method] = by_section

    key = [row for row in rows
           if row["device"] == "nmos_L1_W3"
           and row["curve"] == "idvd@1.85"]
    material = {}
    lookup = {(row["device"], row["curve"], row["section"], row["method"]):
              row["rrms"] for row in rows}
    for method in methods:
        if method == "paper":
            continue
        compared = regressions = 0
        for row in rows:
            if row["method"] != method:
                continue
            paper = lookup.get((row["device"], row["curve"],
                                row["section"], "paper"))
            if paper is None:
                continue
            compared += 1
            if row["rrms"] > paper + max(0.01, 0.1 * paper):
                regressions += 1
        material[method] = {"regressions": regressions, "compared": compared}
    return {
        "definition": {
            "edges": list(EDGES),
            "denominator": "unchanged whole measured-curve mean absolute current",
            "aggregation": "mean curves within device, then mean devices",
            "curve_set": "frozen paper-card inclusion set",
        },
        "aggregate": aggregate,
        "material_section_regressions_vs_paper": material,
        "key_nmos_L1_W3_idvd_1p85": key,
        "rows": rows,
    }


def write_markdown(report: dict, path: Path) -> None:
    section_names = list(next(iter(report["aggregate"].values())))
    lines = [
        "# X-axis section RRMS comparison", "",
        "Each admitted curve is split at normalized sweep positions 0, 0.2, "
        "0.4, 0.7, and 1.0. Each cell is the mean of per-device means and uses "
        "the unchanged whole-curve paper RRMS denominator.", "",
        "| Method | " + " | ".join(section_names) + " |",
        "|---|" + "---:|" * len(section_names),
    ]
    for method, values in report["aggregate"].items():
        lines.append("| " + method + " | " + " | ".join(
            f"{values[name]:.5f}" for name in section_names) + " |")
    lines.extend(["", "## nMOS L=1 um, W=3 um, idvd@1.85 V", "",
                  "| Method | " + " | ".join(section_names) + " |",
                  "|---|" + "---:|" * len(section_names)])
    key = report["key_nmos_L1_W3_idvd_1p85"]
    for method in report["aggregate"]:
        values = {row["section"]: row["rrms"] for row in key
                  if row["method"] == method}
        lines.append("| " + method + " | " + " | ".join(
            f"{values[index]:.5f}" if index in values else "—"
            for index in range(len(section_names))) + " |")
    lines.extend(["", "## Material section regressions versus paper", "",
                  "| Method | Count |", "|---|---:|"])
    for method, row in report["material_section_regressions_vs_paper"].items():
        lines.append(
            f"| {method} | {row['regressions']}/{row['compared']} |")
    path.write_text("\n".join(lines) + "\n")


def make_plot(report: dict, path: Path) -> None:
    methods = list(report["aggregate"])
    section_names = list(next(iter(report["aggregate"].values())))
    x = np.arange(len(section_names))
    width = 0.8 / len(methods)
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for method_index, method in enumerate(methods):
        offset = (method_index - (len(methods) - 1) / 2) * width
        axes[0].bar(x + offset,
                    [report["aggregate"][method][name]
                     for name in section_names], width,
                    label=method, color=colors[method_index % len(colors)])
        key = report["key_nmos_L1_W3_idvd_1p85"]
        values = {row["section"]: row["rrms"] for row in key
                  if row["method"] == method}
        axes[1].bar(x + offset,
                    [values.get(index, np.nan)
                     for index in range(len(section_names))], width,
                    label=method, color=colors[method_index % len(colors)])
    for axis, title in zip(axes, (
            "All devices: independent sweep-section RRMS",
            "nMOS L1/W3 idvd@1.85: hidden low-sweep error")):
        axis.set_xticks(x, ("0–20%", "20–40%", "40–70%", "70–100%"))
        axis.set_ylabel("RRMS")
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.25)
    axes[0].legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--json", type=Path,
        default=Path("out/tables/sweep_segment_comparison.json"))
    parser.add_argument(
        "--markdown", type=Path,
        default=Path("out/tables/sweep_segment_comparison.md"))
    parser.add_argument(
        "--plot", type=Path,
        default=Path("out/figs/foundation_comparison/sweep_segments.png"))
    args = parser.parse_args()
    report = build_report(args.root)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.markdown)
    make_plot(report, args.plot)
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
