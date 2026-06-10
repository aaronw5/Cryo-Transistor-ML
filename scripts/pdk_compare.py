#!/usr/bin/env python3
"""Compare extracted cards with paper cards in the identical NGSpice chain."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.config import FIGS_DIR, OUT_DIR, OUT_TABLES, ensure_dirs  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.metrics import device_rrms  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402


def score_sims(path: Path, curves) -> float:
    saved = np.load(path)
    sims = [np.asarray(saved[f"sim_{i}"]) for i in range(len(curves))]
    return float(device_rrms(sims, [curve.Id for curve in curves])["rrms"])


def load_method(directory: Path) -> dict[str, dict]:
    records = {}
    for path in directory.glob("*.json"):
        record = json.load(open(path))
        if "device" in record:
            records[record["device"]] = record
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["out/pdk_ml:ml"])
    args = parser.parse_args()
    ensure_dirs()

    baseline = {
        row["device"]: row for row in json.load(
            open(OUT_DIR / "pdk_baseline" / "pdk_baseline.json"))["devices"]
    }
    methods = {}
    method_dirs = {}
    for spec in args.methods:
        directory, label = spec.split(":")
        method_dirs[label] = Path(directory)
        methods[label] = load_method(Path(directory))

    rows = []
    for device in PAPER_DEVICES:
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        curves = load_device_curves(device)
        row = {
            "device": tag,
            "paper_reported": device.paper_rrms,
            "paper_params_ngspice": score_sims(
                OUT_DIR / "pdk_baseline" / f"sims_{tag}.npz", curves),
        }
        for label, records in methods.items():
            if tag in records:
                row[label] = score_sims(
                    method_dirs[label] / f"sims_{tag}.npz", curves)
        rows.append(row)

    labels = list(methods)
    summary = {
        "paper_reported_mean": float(np.mean([r["paper_reported"] for r in rows])),
        "paper_params_ngspice_mean": float(np.mean(
            [r["paper_params_ngspice"] for r in rows])),
    }
    for label in labels:
        values = [r[label] for r in rows if label in r]
        summary[f"{label}_mean"] = float(np.mean(values))
        summary[f"{label}_wins_vs_paper_params_ngspice"] = sum(
            r[label] < r["paper_params_ngspice"] for r in rows if label in r)

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    with open(OUT_TABLES / "comparison.csv", "w", newline="") as handle:
        fields = ["device", "paper_reported", "paper_params_ngspice"] + labels
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)

    with open(OUT_TABLES / "comparison.md", "w") as handle:
        handle.write("# Paper-exact comparison\n\n")
        handle.write("All NGSpice columns use the corrected-repository deck "
                     "convention, native geometry bin, and paper companion "
                     "notebook RRMS. NGSpice does not reproduce the paper-reported "
                     "mean, so extracted cards are compared only with paper cards "
                     "in the identical NGSpice chain.\n\n")
        handle.write("| reference / method | mean RRMS |\n|---|---:|\n")
        handle.write(f"| paper reported | {summary['paper_reported_mean']:.3f} |\n")
        handle.write("| paper parameters in NGSpice | "
                     f"{summary['paper_params_ngspice_mean']:.3f} |\n")
        for label in labels:
            handle.write(f"| {label} | {summary[f'{label}_mean']:.3f} |\n")
        handle.write("\n| device | paper reported | paper params in NGSpice | "
                     + " | ".join(labels) + " |\n")
        handle.write("|---|---:|---:|" + "|".join(["---:"] * len(labels)) + "|\n")
        for row in rows:
            handle.write(f"| {row['device']} | {row['paper_reported']:.3f} | "
                         f"{row['paper_params_ngspice']:.3f} | "
                         + " | ".join(
                             f"{row[label]:.3f}" if label in row else "-"
                             for label in labels) + " |\n")

    fig_dir = FIGS_DIR / "comparison"
    fig_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(rows))
    width = 0.8 / (len(labels) + 1)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(x, [r["paper_params_ngspice"] for r in rows], width,
           label="paper params in NGSpice")
    for i, label in enumerate(labels, start=1):
        ax.bar(x + i * width, [r.get(label, np.nan) for r in rows], width,
               label=label)
    ax.set_xticks(x + width * len(labels) / 2)
    ax.set_xticklabels([r["device"] for r in rows], rotation=70, ha="right")
    ax.set_ylabel("paper-exact RRMS")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "summary.png", dpi=150)
    plt.close(fig)

    json.dump(summary, open(OUT_TABLES / "comparison_summary.json", "w"), indent=2)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
