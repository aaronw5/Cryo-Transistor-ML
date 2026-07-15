#!/usr/bin/env python3
"""Compare extracted cards with paper cards in the identical NGSpice chain.

All scores use the confirmed-setup metric (rrmsCalc port). The primary
column freezes curve inclusion to the published-card baseline's included
set (so methods are compared on identical curves); the official
dynamic-inclusion score and the legacy all-curve RRMS are reported too.
"""
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
from cryoml.metrics import device_rrms, score_device_new  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402


def score_sims(path: Path, device, curves, include_tags,
               prefix: str = "sim_") -> dict:
    saved = np.load(path)
    sims = [np.asarray(saved[f"{prefix}{i}"]) for i in range(len(curves))]
    fixed = score_device_new(device.dev_type, device.L_um, device.W_um,
                             curves, sims, include_tags=include_tags)
    official = score_device_new(device.dev_type, device.L_um, device.W_um,
                                curves, sims)
    legacy = device_rrms(sims, [curve.Id for curve in curves])
    return {"rrms": float(fixed["rrms"]),
            "official": float(official["rrms"]),
            "official_sigma": float(official["sigma"]),
            "legacy": float(legacy["rrms"])}


def load_method(directory: Path) -> dict[str, dict]:
    records = {}
    for path in directory.glob("*.json"):
        record = json.load(open(path))
        if "device" in record:
            records[record["device"]] = record
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods", nargs="+",
        default=[
            "out/pdk_direct_mlp:direct_mlp_forward_pass",
            "out/pdk_ml_emu_raw:surrogate_search_raw",
            "out/pdk_ml_emu:surrogate_search_plus_fd",
            "out/pdk_foundation_emu:foundation_plus_fd:fd_sim_",
            "out/pdk_high_voltage_guarded:high_voltage_guarded",
        ],
        help="directory:label[:simulation_prefix] entries; defaults are the "
             "fixed 18-device methods/stages (never a per-device-selected "
             "diagnostic)",
    )
    args = parser.parse_args()
    ensure_dirs()

    baseline_blob = json.load(
        open(OUT_DIR / "pdk_baseline" / "pdk_baseline.json"))
    include_by_tag = {
        tag: {t for t, v in per.items() if v.get("included")}
        for tag, per in baseline_blob["per_curve"].items()
    }
    methods = {}
    method_dirs = {}
    method_prefixes = {}
    expected = {device_tag(d.dev_type, d.L_um, d.W_um)
                for d in PAPER_DEVICES}
    for spec in args.methods:
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"invalid method specification: {spec}")
        directory, label = parts[:2]
        prefix = parts[2] if len(parts) == 3 else "sim_"
        method_dirs[label] = Path(directory)
        method_prefixes[label] = prefix
        methods[label] = load_method(Path(directory))
        seen = set(methods[label])
        if seen != expected:
            raise RuntimeError(
                f"{label}: fixed-method directory is incomplete; "
                f"missing={sorted(expected - seen)}, "
                f"unexpected={sorted(seen - expected)}"
            )
        method_ids = {record.get("method") for record in methods[label].values()}
        box_modes = {record.get("box_mode") for record in methods[label].values()}
        if len(method_ids) != 1 or box_modes != {"lhc10"}:
            raise RuntimeError(
                f"{label}: expected one uniform current method, got "
                f"methods={method_ids}, box_modes={box_modes}"
            )

    rows = []
    for device in PAPER_DEVICES:
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        curves = load_device_curves(device)
        include = include_by_tag[tag]
        base = score_sims(OUT_DIR / "pdk_baseline" / f"sims_{tag}.npz",
                          device, curves, include)
        row = {
            "device": tag,
            "paper_reported": device.paper_rrms,
            "paper_params_ngspice": base["rrms"],
            "paper_params_ngspice_official": base["official"],
            "paper_params_ngspice_legacy": base["legacy"],
        }
        for label, records in methods.items():
            if tag in records:
                s = score_sims(method_dirs[label] / f"sims_{tag}.npz",
                               device, curves, include,
                               prefix=method_prefixes[label])
                row[label] = s["rrms"]
                row[f"{label}_official"] = s["official"]
                row[f"{label}_legacy"] = s["legacy"]
        rows.append(row)

    labels = list(methods)

    def fam_mean(key, fam=None):
        values = [r[key] for r in rows if key in r
                  and (fam is None or r["device"].startswith(fam))]
        return float(np.mean(values)) if values else float("nan")

    def combined(key):
        # rrmsCalc convention: combined = (nMOS mean + pMOS mean) / 2
        return (fam_mean(key, "nmos") + fam_mean(key, "pmos")) / 2.0

    summary = {
        "paper_reported_mean": fam_mean("paper_reported"),
        "paper_reported_combined": combined("paper_reported"),
        "paper_params_ngspice_mean": fam_mean("paper_params_ngspice"),
        "paper_params_ngspice_combined": combined("paper_params_ngspice"),
        "paper_params_ngspice_official_combined":
            combined("paper_params_ngspice_official"),
        "paper_params_ngspice_legacy_mean":
            fam_mean("paper_params_ngspice_legacy"),
        "paper_params_ngspice_legacy_combined":
            combined("paper_params_ngspice_legacy"),
    }
    for key in ("paper_reported", "paper_params_ngspice"):
        for fam in ("nmos", "pmos"):
            summary[f"{key}_{fam}_mean"] = fam_mean(key, fam)
    for label in labels:
        summary[f"{label}_mean"] = fam_mean(label)
        summary[f"{label}_combined"] = combined(label)
        summary[f"{label}_official_combined"] = combined(f"{label}_official")
        summary[f"{label}_legacy_mean"] = fam_mean(f"{label}_legacy")
        summary[f"{label}_legacy_combined"] = combined(f"{label}_legacy")
        for fam in ("nmos", "pmos"):
            summary[f"{label}_{fam}_mean"] = fam_mean(label, fam)
        summary[f"{label}_wins_vs_paper_params_ngspice"] = sum(
            r[label] < r["paper_params_ngspice"] for r in rows if label in r)

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    with open(OUT_TABLES / "comparison.csv", "w", newline="") as handle:
        fields = ["device", "paper_reported", "paper_params_ngspice",
                  "paper_params_ngspice_official",
                  "paper_params_ngspice_legacy"]
        for label in labels:
            fields.extend([label, f"{label}_official", f"{label}_legacy"])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)

    with open(OUT_TABLES / "comparison.md", "w") as handle:
        handle.write("# Confirmed-setup comparison\n\n")
        handle.write("All NGSpice columns use the confirmed-setup chain "
                     "(CryoPDK_Skywater130nm_ML decks + updated pFET card, "
                     "ngspice-41, native geometry bins) scored with the "
                     "rrmsCalc metric; curve inclusion is frozen to the "
                     "published-card baseline's included set for every "
                     "method. combined = (nMOS mean + pMOS mean)/2, the "
                     "upstream headline convention.\n\n")
        handle.write("| reference / method | mean RRMS (all 18) | combined | "
                     "nMOS (8) | pMOS (10) |\n|---|---:|---:|---:|---:|\n")
        for key, name in (("paper_reported", "paper reported"),
                          ("paper_params_ngspice",
                           "paper parameters in NGSpice"),
                          *[(label, label) for label in labels]):
            handle.write(
                f"| {name} | {summary[f'{key}_mean']:.3f} | "
                f"{summary[f'{key}_combined']:.3f} | "
                f"{summary[f'{key}_nmos_mean']:.3f} | "
                f"{summary[f'{key}_pmos_mean']:.3f} |\n")
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
    ax.set_ylabel("confirmed-setup RRMS (fixed inclusion)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "summary.png", dpi=150)
    plt.close(fig)

    json.dump(summary, open(OUT_TABLES / "comparison_summary.json", "w"), indent=2)
    json.dump(rows, open(OUT_TABLES / "comparison_full.json", "w"), indent=2)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
