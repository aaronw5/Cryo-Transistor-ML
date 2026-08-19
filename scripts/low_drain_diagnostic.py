#!/usr/bin/env python3
"""Report the low-drain / high-gate triode region as a first-class metric.

Whole-curve RRMS and even quartile sections dilute this region: a card can be
several times better above the knee and much worse below it and still look
improved.  This reports it directly, for every device, against the published
card.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.metrics import curve_trim  # noqa: E402
from cryoml.regions import (LOW_DRAIN_GATE_MIN_V,  # noqa: E402
                            LOW_DRAIN_MAX_V, is_low_drain_curve,
                            low_drain_mask, masked_rrms)
from cryoml.utils import device_tag  # noqa: E402


def region_rrms(device, curves, simulations, *, drain_max: float,
                gate_min: float) -> dict:
    """RRMS below ``drain_max`` on output curves at gate bias >= ``gate_min``."""
    low, high, whole = [], [], []
    for index, curve in enumerate(curves):
        if not is_low_drain_curve(curve.kind, curve.fixed, gate_min):
            continue
        start, cleaned, denominator = curve_trim(
            device.dev_type, device.L_um, device.W_um, curve.kind, curve.Id)
        if denominator <= 0:
            continue
        sweep = np.abs(np.asarray(curve.Vd, dtype=np.float64))[start:]
        target = np.asarray(cleaned[start:], dtype=np.float64)
        simulated = np.asarray(simulations[index], dtype=np.float64)[start:]
        count = min(len(sweep), len(target), len(simulated))
        sweep, target = sweep[:count], target[:count]
        simulated = simulated[:count]
        mask = low_drain_mask(sweep, drain_max)
        for values, selected in ((low, mask), (high, ~mask),
                                 (whole, np.ones_like(mask))):
            value = masked_rrms(simulated, target, selected, denominator)
            if np.isfinite(value):
                values.append(value)
    return {
        "low_drain_rrms": float(np.mean(low)) if low else float("nan"),
        "above_knee_rrms": float(np.mean(high)) if high else float("nan"),
        "whole_curve_rrms": float(np.mean(whole)) if whole else float("nan"),
        "n_curves": len(whole),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drain-max", type=float,
                        default=LOW_DRAIN_MAX_V)
    parser.add_argument("--gate-min", type=float,
                        default=LOW_DRAIN_GATE_MIN_V)
    parser.add_argument(
        "--sources", default=(
            "paper=pdk_baseline:sim_,"
            "narrow=pdk_foundation_segmented_full_rrms_extract:fd_sim_,"
            "widened=pdk_foundation_widened_extract:fd_sim_,"
            "widened_fd=pdk_real_fd_polish:fd_sim_"))
    parser.add_argument("--json", type=Path,
                        default=Path("out/tables/low_drain_diagnostic.json"))
    parser.add_argument("--markdown", type=Path,
                        default=Path("out/tables/low_drain_diagnostic.md"))
    args = parser.parse_args()

    sources = {}
    for item in args.sources.split(","):
        label, _, spec = item.partition("=")
        directory, _, prefix = spec.partition(":")
        sources[label] = (ROOT / "out" / directory, prefix)

    rows, missing = [], set()
    for device in PAPER_DEVICES:
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        curves = load_device_curves(device)
        row = {"device": tag, "dev_type": device.dev_type}
        for label, (directory, prefix) in sources.items():
            path = directory / f"sims_{tag}.npz"
            if not path.exists():
                missing.add(label)
                continue
            with np.load(path) as data:
                simulations = [np.asarray(data[f"{prefix}{index}"])
                               for index in range(len(curves))]
            row[label] = region_rrms(device, curves, simulations,
                                     drain_max=args.drain_max,
                                     gate_min=args.gate_min)
        rows.append(row)

    labels = [label for label in sources if label not in missing]
    means = {label: {
        key: float(np.mean([row[label][key] for row in rows]))
        for key in ("low_drain_rrms", "above_knee_rrms", "whole_curve_rrms")
    } for label in labels}
    payload = {
        "definition": {
            "low_drain": f"|Vd| < {args.drain_max} V on idvd curves with "
                         f"|fixed Vg| >= {args.gate_min} V",
            "denominator": "unchanged paper per-curve denominator",
        },
        "all_device_means": means, "devices": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n")

    lines = ["# Low-drain / high-gate triode diagnostic", "",
             f"`|Vd| < {args.drain_max} V` on `idvd` curves at "
             f"`|Vg| >= {args.gate_min} V`. Unchanged paper denominators.", "",
             "| Card | Low drain | Above knee | Whole curve |",
             "|---|---:|---:|---:|"]
    for label in labels:
        m = means[label]
        lines.append(f"| {label} | {m['low_drain_rrms']:.4f} | "
                     f"{m['above_knee_rrms']:.4f} | "
                     f"{m['whole_curve_rrms']:.4f} |")
    lines += ["", "## Per device, low-drain RRMS", "",
              "| Device | " + " | ".join(labels) + " | vs paper |",
              "|---|" + "---:|" * (len(labels) + 1)]
    for row in rows:
        values = [row[label]["low_drain_rrms"] for label in labels]
        delta = values[-1] - values[0]
        lines.append(f"| {row['device']} | "
                     + " | ".join(f"{v:.4f}" for v in values)
                     + f" | {delta:+.4f} |")
    args.markdown.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:10]))
    worse = sum(1 for row in rows
                if row[labels[-1]]["low_drain_rrms"]
                > row[labels[0]]["low_drain_rrms"])
    print(f"\ndevices worse than paper in the low-drain region: {worse}/{len(rows)}")


if __name__ == "__main__":
    main()
