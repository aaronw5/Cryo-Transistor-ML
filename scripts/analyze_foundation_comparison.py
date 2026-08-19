#!/usr/bin/env python3
"""Create reproducible tables and overview plots for foundation extractions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def curve_groups(per_curve: dict) -> dict[str, float]:
    groups = {name: [] for name in (
        "low", "mid", "high", "idvd", "idvg")}
    for tag, row in per_curve.items():
        if not row.get("included") or row.get("rrms") is None:
            continue
        kind, bias_text = tag.split("@", 1)
        bias = float(bias_text)
        value = float(row["rrms"])
        groups[kind].append(value)
        region = "low" if bias <= 0.74 else "mid" if bias < 1.48 else "high"
        groups[region].append(value)
    return {name: float(np.mean(values)) for name, values in groups.items()
            if values}


def method_maps(root: Path) -> tuple[dict, dict, list[str]]:
    paper = json.loads(
        (root / "out/pdk_baseline/pdk_baseline.json").read_text())
    methods = {
        "paper": {
            row["device"]: {
                **row, "per_curve": paper["per_curve"][row["device"]],
                "high_voltage_rrms": curve_groups(
                    paper["per_curve"][row["device"]])["high"],
            } for row in paper["devices"]
        }
    }
    sources = {
        "seven_fd": ("out/pdk_foundation_vb_seven/summary.json",
                     "surrogate_fd"),
        "seven_inverse_original": (
            "out/pdk_foundation_vb_seven/summary.json",
            "inverse_autoencoder"),
        "seven_tradeoff": (
            "out/pdk_foundation_noiseaware_seven_tradeoff/summary.json",
            "noise_aware_fd"),
        "seven_regional": (
            "out/pdk_foundation_segmented_seven_rrms_extract/summary.json",
            "noise_aware_fd"),
        "seven_inverse_general": (
            "out/pdk_foundation_segmented_seven_inverse_general/summary.json",
            "inverse_autoencoder"),
        "full43_fd": ("out/pdk_foundation_vb_full/summary.json",
                      "surrogate_fd"),
        "full43_inverse_original": (
            "out/pdk_foundation_vb_full/summary.json",
            "inverse_autoencoder"),
        "full43_tradeoff": (
            "out/pdk_foundation_noiseaware_full_tradeoff/summary.json",
            "noise_aware_fd"),
        "full43_regional": (
            "out/pdk_foundation_segmented_full_rrms_extract/summary.json",
            "noise_aware_fd"),
        "full43_inverse_general": (
            "out/pdk_foundation_segmented_full_inverse_general/summary.json",
            "inverse_autoencoder"),
    }
    for label, (relative, method) in sources.items():
        path = root / relative
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        methods[label] = {
            row["device"]: row["methods"][method]
            for row in payload["devices"]
        }
    devices = [row["device"] for row in paper["devices"]]
    return paper, methods, devices


def build_report(root: Path) -> dict:
    paper, methods, devices = method_maps(root)
    device_meta = {row["device"]: row for row in paper["devices"]}
    aggregate, device_rows, curve_rows = {}, [], []
    for label, values in methods.items():
        available = [tag for tag in devices if tag in values]
        group_rows = [curve_groups(values[tag]["per_curve"])
                      for tag in available]
        material = 0
        compared = 0
        for tag in available:
            reference = methods["paper"][tag]["per_curve"]
            for curve, row in values[tag]["per_curve"].items():
                base = reference.get(curve, {})
                if (not row.get("included") or row.get("rrms") is None
                        or not base.get("included")
                        or base.get("rrms") is None):
                    continue
                compared += 1
                delta = float(row["rrms"]) - float(base["rrms"])
                if delta > 0.01 and float(row["rrms"]) > 1.1 * float(base["rrms"]):
                    material += 1
        aggregate[label] = {
            "n_devices": len(available),
            "overall": float(np.mean([values[tag]["rrms"]
                                      for tag in available])),
            "nmos": float(np.mean([values[tag]["rrms"] for tag in available
                                    if device_meta[tag]["dev_type"] == "nmos"])),
            "pmos": float(np.mean([values[tag]["rrms"] for tag in available
                                    if device_meta[tag]["dev_type"] == "pmos"])),
            **{name: float(np.mean([row[name] for row in group_rows]))
               for name in ("low", "mid", "high", "idvd", "idvg")},
            "material_regressions_vs_paper": material,
            "compared_curves": compared,
        }
    for tag in devices:
        row = {"device": tag, "dev_type": device_meta[tag]["dev_type"],
               "L_um": device_meta[tag]["L_um"],
               "W_um": device_meta[tag]["W_um"]}
        for label, values in methods.items():
            if tag not in values:
                continue
            groups = curve_groups(values[tag]["per_curve"])
            row[label] = {"overall": float(values[tag]["rrms"]), **groups}
        device_rows.append(row)
        curve_tags = methods["paper"][tag]["per_curve"]
        for curve, base in curve_tags.items():
            if not base.get("included") or base.get("rrms") is None:
                continue
            curve_row = {"device": tag, "curve": curve,
                         "paper": float(base["rrms"])}
            for label, values in methods.items():
                candidate = values.get(tag, {}).get("per_curve", {}).get(curve)
                if (candidate and candidate.get("included")
                        and candidate.get("rrms") is not None):
                    curve_row[label] = float(candidate["rrms"])
            curve_rows.append(curve_row)
    worst = {}
    for label in methods:
        worst[label] = sorted(
            [row for row in curve_rows if label in row],
            key=lambda row: row[label], reverse=True)[:20]
    return {"aggregate": aggregate, "devices": device_rows,
            "curves": curve_rows, "worst_curves": worst}


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Foundation extraction comparison", "",
        "Low is fixed bias <=0.74 V, mid is 1.11 V, and high is >=1.48 V. "
        "All RRMS values use the unchanged paper scorer.", "",
        "| Method | Overall | nMOS | pMOS | Low | Mid | High | idvd | idvg | "
        "Material regressions vs paper |", 
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in report["aggregate"].items():
        lines.append(
            f"| {label} | {row['overall']:.5f} | {row['nmos']:.5f} | "
            f"{row['pmos']:.5f} | {row['low']:.5f} | {row['mid']:.5f} | "
            f"{row['high']:.5f} | {row['idvd']:.5f} | {row['idvg']:.5f} | "
            f"{row['material_regressions_vs_paper']}/{row['compared_curves']} |")
    labels = list(report["aggregate"])
    lines.extend(["", "## Per-device overall / low / mid / high", "",
                  "| Device | " + " | ".join(labels) + " |",
                  "|---|" + "---:|" * len(labels)])
    for row in report["devices"]:
        cells = []
        for label in labels:
            if label not in row:
                cells.append("—")
            else:
                value = row[label]
                cells.append(f"{value['overall']:.4f} / {value['low']:.4f} / "
                             f"{value['mid']:.4f} / {value['high']:.4f}")
        lines.append(f"| {row['device']} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Worst curves by method", ""])
    for label, rows in report["worst_curves"].items():
        lines.extend([f"### {label}", "", "| Device | Curve | RRMS | Paper |",
                      "|---|---|---:|---:|"])
        for row in rows[:12]:
            lines.append(f"| {row['device']} | {row['curve']} | "
                         f"{row[label]:.4f} | {row['paper']:.4f} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def make_plots(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = list(report["aggregate"])
    colors = plt.get_cmap("tab10").colors
    regions = ["overall", "low", "mid", "high"]
    x = np.arange(len(regions))
    width = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    for index, label in enumerate(labels):
        ax.bar(x + (index - (len(labels) - 1) / 2) * width,
               [report["aggregate"][label][region] for region in regions],
               width, label=label, color=colors[index % len(colors)])
    ax.set_xticks(x, regions)
    ax.set_ylabel("mean per-device RRMS")
    ax.set_title("Foundation-card accuracy by bias regime")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(out_dir / "aggregate_bias_comparison.png", dpi=180)
    plt.close(fig)

    devices = [row["device"] for row in report["devices"]]
    matrix = np.asarray([
        [row.get(label, {}).get("overall", np.nan) for label in labels]
        for row in report["devices"]])
    fig, ax = plt.subplots(figsize=(9, 10), layout="constrained")
    image = ax.imshow(matrix, aspect="auto", cmap="magma_r", vmin=0,
                      vmax=min(0.8, float(np.nanmax(matrix))))
    ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(devices)), devices)
    for i in range(len(devices)):
        for j in range(len(labels)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                        fontsize=7, color="white" if matrix[i, j] > 0.35 else "black")
    ax.set_title("Per-device overall RRMS")
    fig.colorbar(image, ax=ax, label="RRMS")
    fig.savefig(out_dir / "device_rrms_heatmap.png", dpi=180)
    plt.close(fig)

    curve_order = sorted(
        {row["curve"] for row in report["curves"]},
        key=lambda tag: (tag.split("@")[0], float(tag.split("@")[1])))
    lookup = {(row["device"], row["curve"]): row
              for row in report["curves"]}
    for label in labels:
        if label == "paper":
            continue
        delta = np.full((len(devices), len(curve_order)), np.nan)
        for i, device in enumerate(devices):
            for j, curve in enumerate(curve_order):
                row = lookup.get((device, curve))
                if row is not None and label in row:
                    delta[i, j] = row[label] - row["paper"]
        finite = np.abs(delta[np.isfinite(delta)])
        limit = max(0.05, min(0.5, float(np.quantile(finite, 0.95))))
        fig, ax = plt.subplots(figsize=(13, 9), layout="constrained")
        image = ax.imshow(delta, aspect="auto", cmap="RdBu_r",
                          vmin=-limit, vmax=limit)
        ax.set_xticks(np.arange(len(curve_order)), curve_order,
                      rotation=45, ha="right")
        ax.set_yticks(np.arange(len(devices)), devices)
        ax.set_title(f"{label}: per-curve RRMS change from paper card")
        fig.colorbar(image, ax=ax,
                     label="RRMS delta (blue improves, red regresses)")
        fig.savefig(out_dir / f"curve_delta_{label}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", type=Path,
                        default=Path("out/tables/foundation_comparison.json"))
    parser.add_argument("--markdown", type=Path,
                        default=Path("out/tables/foundation_comparison.md"))
    parser.add_argument("--fig-dir", type=Path,
                        default=Path("out/figs/foundation_comparison"))
    args = parser.parse_args()
    report = build_report(args.root)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.markdown)
    make_plots(report, args.fig_dir)
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
