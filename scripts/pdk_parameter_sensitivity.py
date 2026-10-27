#!/usr/bin/env python3
"""Screen the maximum continuous BSIM4 DC-current parameter surface.

Each candidate is perturbed around the effective published-card value and
simulated on the complete measured-bias curve grid.  Selection is global over
all requested devices and has no top-k cap: a parameter is retained when it
changes any full-grid or high-voltage curve above the configured normalized
response threshold.  The original seven parameters are always retained so the
expanded experiment is a strict superset of the paper setup.

The output also records per-device safe perturbation bounds for expanded LHC
generation.  Run a pilot before the all-device screen, for example:

  PYTHONPATH=src .venv/bin/python scripts/pdk_parameter_sensitivity.py \
    --devices nmos:1:3 --out-dir out/pdk_parameter_sensitivity_pilot
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.config import OUT_DIR, ensure_dirs  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import parse_device_list  # noqa: E402
from cryoml.parameter_sets import (MAX_DC_CANDIDATES, PARAMETER_GROUP_BY_NAME,  # noqa: E402
                                   PARAMS7, PRE_SCREEN_EXCLUSIONS,
                                   UPSTREAM_EXPANDED18)
from cryoml.pdk_extract import read_bin_parameters  # noqa: E402
from cryoml.spice_pdk import simulate_pdk  # noqa: E402
from cryoml.utils import device_tag, get_logger  # noqa: E402

logger = get_logger("pdk_parameter_sensitivity")
DEFAULT_OUT = OUT_DIR / "pdk_parameter_sensitivity"
HIGH_VOLTAGE_FLOOR = 1.48


def metric_tag(curve) -> str:
    return f"{curve.kind}@{abs(float(curve.fixed)):g}"


def response_metrics(reference, candidate, curves) -> dict[str, object] | None:
    """Normalized current response, kept distinct from measured-fit RRMS.

    The denominator combines the measured mean magnitude with a small fraction
    of the device peak.  This preserves leakage sensitivity without allowing a
    numerically tiny off-curve to dominate the screening decision.
    """
    if len(reference) != len(candidate) or len(reference) != len(curves):
        return None
    finite = all(np.isfinite(np.asarray(values)).all()
                 for values in candidate)
    if not finite:
        return None
    device_peak = max(
        [float(np.max(np.abs(curve.Id))) for curve in curves if len(curve.Id)]
        + [1e-30])
    floor = max(1e-15, 1e-6 * device_peak)
    per_curve = {}
    values, high_values = [], []
    for ref, cand, curve in zip(reference, candidate, curves):
        ref = np.asarray(ref, dtype=np.float64)
        cand = np.asarray(cand, dtype=np.float64)
        n = min(len(ref), len(cand), len(curve.Id))
        if n == 0:
            continue
        denominator = max(float(np.mean(np.abs(curve.Id[:n]))), floor)
        response = float(np.sqrt(np.mean((cand[:n] - ref[:n]) ** 2))
                         / denominator)
        tag = metric_tag(curve)
        per_curve[tag] = response
        values.append(response)
        if abs(float(curve.fixed)) >= HIGH_VOLTAGE_FLOOR - 1e-6:
            high_values.append(response)
    if not values:
        return None
    return {
        "max_full_grid": float(max(values)),
        "mean_full_grid": float(np.mean(values)),
        "max_high_voltage": float(max(high_values)) if high_values else 0.0,
        "mean_high_voltage": float(np.mean(high_values)) if high_values else 0.0,
        "per_curve": per_curve,
    }


def family_scales(readbacks: dict[str, dict]) -> dict[str, dict[str, float]]:
    """Median nonzero magnitude, with an across-polarity zero fallback.

    A parameter published as zero throughout one polarity can still be
    physically effective away from zero (PRWG for the present nMOS family is
    the motivating example).  Falling back to the other polarity's nonzero
    scale lets the sensitivity probe test that hypothesis instead of silently
    collapsing the dimension.
    """
    out: dict[str, dict[str, float]] = {"nmos": {}, "pmos": {}}
    global_values = {}
    for parameter in MAX_DC_CANDIDATES:
        values = [
            abs(float(row["published"][parameter]))
            for row in readbacks.values()
            if parameter in row["published"]
            and np.isfinite(float(row["published"][parameter]))
            and abs(float(row["published"][parameter])) > 1e-30
        ]
        global_values[parameter] = (
            float(np.median(values)) if values else 0.0)
    for family in out:
        family_rows = [row["published"] for row in readbacks.values()
                       if row["dev_type"] == family]
        for parameter in MAX_DC_CANDIDATES:
            values = [abs(float(row[parameter])) for row in family_rows
                      if parameter in row
                      and np.isfinite(float(row[parameter]))
                      and abs(float(row[parameter])) > 1e-30]
            out[family][parameter] = (
                float(np.median(values)) if values
                else global_values[parameter])
    return out


def initial_half_span(value: float, family_scale: float,
                      fraction: float) -> float:
    scale = abs(float(value)) if abs(float(value)) > 1e-30 else family_scale
    return float(fraction * scale)


def simulate_candidate(device, curves, bin_index: int, parameter: str,
                       value: float):
    return simulate_pdk(
        device.dev_type, device.L_um, device.W_um, curves,
        params={parameter: float(value)}, bin_index=bin_index)


def screen_device(job) -> dict:
    (device, bin_index, published, scales, fraction, retry_scales,
     out_dir_string) = job
    tag = device_tag(device.dev_type, device.L_um, device.W_um)
    curves = load_device_curves(device)
    t0 = time.time()
    baseline = simulate_pdk(
        device.dev_type, device.L_um, device.W_um, curves,
        bin_index=bin_index)
    repeat = simulate_pdk(
        device.dev_type, device.L_um, device.W_um, curves,
        bin_index=bin_index)
    repeatability = response_metrics(baseline, repeat, curves)
    parameters = {}
    for parameter in MAX_DC_CANDIDATES:
        if parameter not in published:
            parameters[parameter] = {"status": "missing_readback"}
            continue
        center = float(published[parameter])
        half_span = initial_half_span(
            center, float(scales.get(parameter, 0.0)), fraction)
        if not np.isfinite(half_span) or half_span <= 0:
            parameters[parameter] = {
                "status": "zero_across_family", "published": center,
            }
            continue
        directions = {}
        for direction, sign in (("lower", -1.0), ("upper", 1.0)):
            attempt = None
            for retry_scale in retry_scales:
                candidate_value = center + sign * half_span * retry_scale
                sims = simulate_candidate(
                    device, curves, bin_index, parameter, candidate_value)
                response = response_metrics(baseline, sims, curves)
                if response is not None:
                    attempt = {
                        "value": float(candidate_value),
                        "fraction_of_initial_span": float(retry_scale),
                        "response": response,
                    }
                    break
            directions[direction] = attempt
        valid = [row for row in directions.values() if row is not None]
        if valid:
            summary = {
                key: float(max(row["response"][key] for row in valid))
                for key in ("max_full_grid", "mean_full_grid",
                            "max_high_voltage", "mean_high_voltage")
            }
            lo = min([center] + [row["value"] for row in valid])
            hi = max([center] + [row["value"] for row in valid])
            parameters[parameter] = {
                "status": "probed", "published": center,
                "initial_half_span": half_span,
                "safe_bounds": [float(lo), float(hi)],
                "summary": summary, "directions": directions,
            }
        else:
            parameters[parameter] = {
                "status": "no_finite_perturbation", "published": center,
                "initial_half_span": half_span, "directions": directions,
            }
    record = {
        "device": tag, "dev_type": device.dev_type,
        "L_um": device.L_um, "W_um": device.W_um,
        "bin_index": int(bin_index), "fraction": fraction,
        "repeatability": repeatability, "parameters": parameters,
        "runtime_s": round(time.time() - t0, 1),
    }
    out_path = Path(out_dir_string) / f"sensitivity_{tag}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def build_summary(records: list[dict], threshold: float,
                  fraction: float) -> dict:
    repeatability = max(
        [record["repeatability"]["max_full_grid"]
         for record in records if record.get("repeatability")] + [0.0])
    effective_threshold = max(float(threshold), 10.0 * repeatability)
    parameters = {}
    selected = []
    for parameter in MAX_DC_CANDIDATES:
        rows = []
        for record in records:
            result = record["parameters"].get(parameter, {})
            if result.get("status") == "probed":
                rows.append((record, result))
        max_full = max([row[1]["summary"]["max_full_grid"]
                        for row in rows] + [0.0])
        max_high = max([row[1]["summary"]["max_high_voltage"]
                        for row in rows] + [0.0])
        active_devices = [
            row[0]["device"] for row in rows
            if max(row[1]["summary"]["max_full_grid"],
                   row[1]["summary"]["max_high_voltage"])
            >= effective_threshold
        ]
        mandatory = parameter in PARAMS7
        keep = mandatory or bool(active_devices)
        if keep:
            selected.append(parameter)
        parameters[parameter] = {
            "group": PARAMETER_GROUP_BY_NAME[parameter],
            "selected": keep, "mandatory_seven_parameter": mandatory,
            "in_upstream_expanded18": parameter in UPSTREAM_EXPANDED18,
            "max_full_grid_response": float(max_full),
            "max_high_voltage_response": float(max_high),
            "active_devices": active_devices,
            "n_devices_probed": len(rows),
        }
    bounds_by_device = {}
    published_by_device = {}
    active_masks_by_device = {}
    for record in records:
        bounds, published, active_mask = {}, {}, {}
        for parameter in selected:
            result = record["parameters"].get(parameter, {})
            if result.get("status") == "probed":
                bounds[parameter] = result["safe_bounds"]
                published[parameter] = result["published"]
                # Once a term is relevant anywhere in the 18-card union, keep
                # every safe nondegenerate device-specific dimension. A small
                # one-at-a-time response on one card does not establish
                # inactivity under simultaneous multi-parameter variation.
                active_mask[parameter] = bool(
                    result["safe_bounds"][1] > result["safe_bounds"][0])
            elif parameter in PARAMS7 and "published" in result:
                # A mandatory original parameter should still have a tiny
                # nondegenerate box if its response probe failed.
                center = float(result["published"])
                span = max(abs(center) * fraction, 1e-30)
                bounds[parameter] = [center - span, center + span]
                published[parameter] = center
                active_mask[parameter] = True
            elif "published" in result:
                # Keep one global feature layout. Family/bin-specific inactive
                # terms are fixed exactly at the published value for this
                # device rather than acquiring an invented training range.
                center = float(result["published"])
                bounds[parameter] = [center, center]
                published[parameter] = center
                active_mask[parameter] = False
            else:
                raise RuntimeError(
                    f"{record['device']}: no readback for selected {parameter}")
        bounds_by_device[record["device"]] = bounds
        published_by_device[record["device"]] = published
        active_masks_by_device[record["device"]] = active_mask
    return {
        "schema_version": 1,
        "selection_rule": (
            "retain every candidate whose normalized response exceeds the "
            "effective threshold on any full-grid or |fixed bias|>=1.48 V "
            "curve for any device; PARAMS7 are mandatory; no top-k cap"),
        "requested_threshold": float(threshold),
        "repeatability_floor": float(repeatability),
        "effective_threshold": float(effective_threshold),
        "perturbation_fraction": float(fraction),
        "candidate_count": len(MAX_DC_CANDIDATES),
        "selected_count": len(selected),
        "selected_parameters": selected,
        "original_parameters": list(PARAMS7),
        "upstream_expanded18": list(UPSTREAM_EXPANDED18),
        "pre_screen_exclusions": PRE_SCREEN_EXCLUSIONS,
        "parameters": parameters,
        "bounds_by_device": bounds_by_device,
        "published_by_device": published_by_device,
        "active_masks_by_device": active_masks_by_device,
        "devices": [record["device"] for record in records],
        "runtime_s": round(sum(record["runtime_s"] for record in records), 1),
    }


def write_markdown(summary: dict, path: Path) -> None:
    selected = summary["selected_parameters"]
    lines = [
        "# Expanded BSIM4 parameter sensitivity screen", "",
        f"Candidates: {summary['candidate_count']}; selected: "
        f"{summary['selected_count']}; devices: {len(summary['devices'])}.", "",
        f"Requested normalized response threshold: "
        f"`{summary['requested_threshold']:.3g}`; repeatability floor: "
        f"`{summary['repeatability_floor']:.3g}`; effective threshold: "
        f"`{summary['effective_threshold']:.3g}`.", "",
        "Selection has no top-k cap. The original seven parameters are a "
        "mandatory nested subset.", "",
        "## Selected parameters", "",
        ", ".join(f"`{parameter.upper()}`" for parameter in selected), "",
        "## Per-parameter response", "",
        "| parameter | group | selected | max full-grid | max high-voltage | "
        "active devices |", "|---|---|:---:|---:|---:|---:|",
    ]
    for parameter, row in summary["parameters"].items():
        lines.append(
            f"| {parameter.upper()} | {row['group']} | "
            f"{'yes' if row['selected'] else 'no'} | "
            f"{row['max_full_grid_response']:.4g} | "
            f"{row['max_high_voltage_response']:.4g} | "
            f"{len(row['active_devices'])} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 < args.fraction <= 0.5:
        parser.error("--fraction must be in (0, 0.5]")
    if args.threshold <= 0:
        parser.error("--threshold must be positive")

    ensure_dirs()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    devices = parse_device_list(args.devices)
    # Released-card bins from pdk_baseline.json, or with
    # CRYOML_PDK_TEMPLATE=stock the NGSpice-selected stock bins (I25-1).
    from cryoml.cold_start import device_bins
    bins = device_bins()

    readbacks = {}
    for device in devices:
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        if tag not in bins:
            raise RuntimeError(f"baseline bin missing for {tag}")
        published = read_bin_parameters(
            device.dev_type, device.L_um, device.W_um, bins[tag],
            tuple(MAX_DC_CANDIDATES))
        readbacks[tag] = {
            "dev_type": device.dev_type, "published": published,
        }
    scales = family_scales(readbacks)
    (out_dir / "readbacks.json").write_text(json.dumps({
        "candidates": list(MAX_DC_CANDIDATES), "devices": readbacks,
        "family_scales": scales,
    }, indent=2) + "\n")

    records, jobs = [], []
    retry_scales = (1.0, 0.5, 0.25, 0.10)
    for device in devices:
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        record_path = out_dir / f"sensitivity_{tag}.json"
        if args.resume and record_path.exists():
            records.append(json.loads(record_path.read_text()))
            continue
        jobs.append((device, bins[tag], readbacks[tag]["published"],
                     scales[device.dev_type], args.fraction, retry_scales,
                     str(out_dir)))
    if jobs:
        if args.workers == 1:
            iterator = map(screen_device, jobs)
            for record in iterator:
                records.append(record)
                logger.info("%-22s screened %d parameters in %.1fs",
                            record["device"], len(MAX_DC_CANDIDATES),
                            record["runtime_s"])
        else:
            context = get_context("spawn")
            with context.Pool(processes=args.workers) as pool:
                for record in pool.imap_unordered(screen_device, jobs):
                    records.append(record)
                    logger.info("%-22s screened %d parameters in %.1fs",
                                record["device"], len(MAX_DC_CANDIDATES),
                                record["runtime_s"])
    records.sort(key=lambda row: row["device"])
    summary = build_summary(records, args.threshold, args.fraction)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(summary, out_dir / "summary.md")
    print(json.dumps({key: summary[key] for key in (
        "candidate_count", "selected_count", "selected_parameters",
        "effective_threshold", "runtime_s")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
