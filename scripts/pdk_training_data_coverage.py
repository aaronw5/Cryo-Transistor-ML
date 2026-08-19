#!/usr/bin/env python3
"""Audit whether measured I-V curves are covered by synthetic training data.

The pointwise envelope test and best-whole-curve test deliberately answer
different questions. A measured curve can lie inside the marginal envelope at
every voltage while no single compact-model parameter vector produces the
whole curve. Only curves in the confirmed paper-card inclusion set are used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.config import OUT_DIR, OUT_TABLES, PROCESSED_DIR, ensure_dirs  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.foundation import canonical_point_order  # noqa: E402
from cryoml.metrics import curve_trim  # noqa: E402
from cryoml.utils import device_tag, get_logger  # noqa: E402

logger = get_logger("pdk_training_data_coverage")


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n_curves": 0}
    keys = (
        "outside_minmax_fraction", "outside_central_fraction",
        "pointwise_envelope_bound_rrms", "best_single_curve_sample_rrms",
    )
    return {
        "n_curves": len(rows),
        **{key: float(np.mean([row[key] for row in rows])) for key in keys},
    }


def audit_dataset(name: str, root: Path, baseline: dict,
                  quantile: float) -> dict:
    rows = []
    for device in PAPER_DEVICES:
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        allowed = {
            curve_tag for curve_tag, value in baseline["per_curve"][tag].items()
            if value.get("included")
        }
        path = root / f"{tag}.npz"
        with np.load(path, allow_pickle=True) as data:
            valid = np.flatnonzero(data["ok"])
            order, curves = canonical_point_order(
                data["kinds"], data["fixeds"], data["slices"])
            currents = np.asarray(
                data["IDS"][valid], dtype=np.float32)[:, order]
            measured = np.asarray(data["meas"], dtype=np.float64)[order]
            active = (int(np.sum(data["active"]))
                      if "active" in data.files else 7)

        for curve in curves:
            if curve["tag"] not in allowed:
                continue
            span = slice(curve["start"], curve["stop"])
            start, cleaned, denominator = curve_trim(
                device.dev_type, device.L_um, device.W_um,
                curve["kind"], measured[span])
            simulated = np.asarray(
                currents[:, curve["start"] + start:curve["stop"]],
                dtype=np.float64)
            target = np.asarray(cleaned[start:], dtype=np.float64)
            lower = np.min(simulated, axis=0)
            upper = np.max(simulated, axis=0)
            central_lower = np.quantile(simulated, quantile, axis=0)
            central_upper = np.quantile(simulated, 1.0 - quantile, axis=0)
            clipped = np.clip(target, lower, upper)
            sample_rrms = np.sqrt(np.mean(
                (simulated - target[None, :]) ** 2, axis=1)) / denominator
            rows.append({
                "dataset": name, "device": tag,
                "dev_type": device.dev_type,
                "kind": curve["kind"], "bias": float(curve["bias"]),
                "curve": curve["tag"], "n_active_parameters": active,
                "outside_minmax_fraction": float(np.mean(
                    (target < lower) | (target > upper))),
                "outside_central_fraction": float(np.mean(
                    (target < central_lower) | (target > central_upper))),
                "pointwise_envelope_bound_rrms": float(np.sqrt(np.mean(
                    (clipped - target) ** 2)) / denominator),
                "best_single_curve_sample_rrms": float(np.min(sample_rrms)),
            })

    by_bias = {}
    for kind in ("idvd", "idvg"):
        biases = sorted({row["bias"] for row in rows if row["kind"] == kind})
        for bias in biases:
            by_bias[f"{kind}@{bias:g}"] = summarize([
                row for row in rows
                if row["kind"] == kind and row["bias"] == bias
            ])
    return {
        "root": str(root), "n_samples_per_device": 10000,
        "summary": summarize(rows),
        "low_voltage": summarize([row for row in rows if row["bias"] < 1.48]),
        "high_voltage": summarize([row for row in rows if row["bias"] >= 1.48]),
        "by_bias": by_bias, "curves": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seven-dir", type=Path,
                        default=PROCESSED_DIR / "pdk_synth")
    parser.add_argument("--full-dir", type=Path,
                        default=PROCESSED_DIR / "pdk_synth_full")
    parser.add_argument("--baseline", type=Path,
                        default=OUT_DIR / "pdk_baseline" / "pdk_baseline.json")
    parser.add_argument("--quantile", type=float, default=0.01)
    parser.add_argument("--out", type=Path,
                        default=OUT_TABLES / "training_data_coverage.json")
    args = parser.parse_args()
    if not 0.0 < args.quantile < 0.5:
        raise ValueError("--quantile must be between zero and 0.5")
    ensure_dirs()
    baseline = json.loads(args.baseline.read_text())
    payload = {
        "definition": {
            "curve_set": "confirmed paper-card frozen inclusion set",
            "marginal_envelope": "pointwise min/max across synthetic cards",
            "central_envelope": (
                f"pointwise q={args.quantile:g} to q={1-args.quantile:g}"),
            "pointwise_envelope_bound_rrms": (
                "RRMS after independently clipping every measured point to "
                "the synthetic pointwise min/max; optimistic lower bound, "
                "not necessarily achievable by one parameter vector"),
            "best_single_curve_sample_rrms": (
                "best complete-curve RRMS among stored real-NGSpice cards; "
                "a different card may win each curve"),
        },
        "datasets": {
            "seven": audit_dataset(
                "seven", args.seven_dir, baseline, args.quantile),
            "full43": audit_dataset(
                "full43", args.full_dir, baseline, args.quantile),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    for name, data in payload["datasets"].items():
        logger.info("%s overall=%s low=%s high=%s", name, data["summary"],
                    data["low_voltage"], data["high_voltage"])


if __name__ == "__main__":
    main()
