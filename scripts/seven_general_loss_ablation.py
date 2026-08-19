#!/usr/bin/env python3
"""Test general curve-robust objectives on the real seven-parameter LHC.

No device identifier enters an objective. Each card is extracted from its own
measured I-V data, as it would be for a new geometry. The objectives use only
curve family, fixed-bias regime, and curve RRMS.
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
from cryoml.pdk_extract import PARAMS7  # noqa: E402
from cryoml.utils import device_tag, get_logger  # noqa: E402

logger = get_logger("seven_general_loss_ablation")


def score_matrix(device, data, allowed: set[str]) -> tuple[np.ndarray, list[str]]:
    valid = np.flatnonzero(data["ok"])
    order, curves = canonical_point_order(
        data["kinds"], data["fixeds"], data["slices"])
    currents = np.asarray(data["IDS"][valid], dtype=np.float64)[:, order]
    measured = np.asarray(data["meas"], dtype=np.float64)[order]
    columns, tags = [], []
    for curve in curves:
        if curve["tag"] not in allowed:
            continue
        span = slice(curve["start"], curve["stop"])
        start, cleaned, denominator = curve_trim(
            device.dev_type, device.L_um, device.W_um,
            curve["kind"], measured[span])
        block = currents[:, curve["start"] + start:curve["stop"]]
        target = cleaned[start:]
        columns.append(np.sqrt(np.mean(
            (block - target[None, :]) ** 2, axis=1)) / denominator)
        tags.append(curve["tag"])
    return np.stack(columns, axis=1), tags


def bias(tag: str) -> float:
    return float(tag.split("@", 1)[1])


def regime_balanced(scores: np.ndarray, tags: list[str]) -> np.ndarray:
    groups = []
    for kind in ("idvd", "idvg"):
        for region in ("ordinary", "high"):
            indices = [index for index, tag in enumerate(tags)
                       if tag.startswith(kind + "@")
                       and ((bias(tag) >= 1.48) == (region == "high"))]
            if indices:
                groups.append(np.mean(scores[:, indices], axis=1))
    return np.mean(np.stack(groups, axis=1), axis=1)


def cvar(scores: np.ndarray, fraction: float) -> np.ndarray:
    count = max(1, int(np.ceil(scores.shape[1] * fraction)))
    return np.mean(np.partition(scores, scores.shape[1] - count, axis=1)
                   [:, -count:], axis=1)


def metrics(row: np.ndarray, tags: list[str]) -> dict:
    low = [value for value, tag in zip(row, tags) if bias(tag) < 1.48]
    high = [value for value, tag in zip(row, tags) if bias(tag) >= 1.48]
    return {
        "overall_rrms": float(np.mean(row)),
        "low_voltage_rrms": float(np.mean(low)),
        "high_voltage_rrms": float(np.mean(high)),
        "worst_curve_rrms": float(np.max(row)),
        "per_curve": {tag: float(value) for tag, value in zip(tags, row)},
    }


def aggregate(records: list[dict], policy: str, paper: dict) -> dict:
    selected = [record["policies"][policy] for record in records]
    nmos = [row for record, row in zip(records, selected)
            if record["dev_type"] == "nmos"]
    pmos = [row for record, row in zip(records, selected)
            if record["dev_type"] == "pmos"]
    regressions = material = total = 0
    for record, row in zip(records, selected):
        reference = paper["per_curve"][record["device"]]
        for tag, value in row["per_curve"].items():
            old = reference[tag]["rrms"]
            if old is None:
                continue
            total += 1
            regressions += value > old
            material += value > old + 0.01 and value > 1.1 * old
    result = {}
    for key in ("overall_rrms", "low_voltage_rrms", "high_voltage_rrms",
                "worst_curve_rrms"):
        all_values = [row[key] for row in selected]
        n_values = [row[key] for row in nmos]
        p_values = [row[key] for row in pmos]
        result[key] = {
            "all_device_mean": float(np.mean(all_values)),
            "nmos_mean": float(np.mean(n_values)),
            "pmos_mean": float(np.mean(p_values)),
            "family_balanced_mean": float(
                (np.mean(n_values) + np.mean(p_values)) / 2.0),
        }
    result["curve_regressions_vs_paper"] = regressions
    result["material_curve_regressions_vs_paper"] = material
    result["n_compared_curves"] = total
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=PROCESSED_DIR / "pdk_synth")
    parser.add_argument("--tail-fraction", type=float, default=0.25)
    parser.add_argument("--tail-weight", type=float, default=0.25)
    parser.add_argument("--rrms-cap", type=float, default=0.25)
    parser.add_argument("--regret-floor", type=float, default=0.02)
    parser.add_argument("--out", type=Path,
                        default=OUT_TABLES / "seven_general_loss_ablation.json")
    args = parser.parse_args()
    if not 0 < args.tail_fraction <= 1:
        raise ValueError("--tail-fraction must be in (0, 1]")
    if not 0 <= args.tail_weight <= 1:
        raise ValueError("--tail-weight must be in [0, 1]")
    ensure_dirs()
    paper = json.loads(
        (OUT_DIR / "pdk_baseline" / "pdk_baseline.json").read_text())
    records = []
    for device in PAPER_DEVICES:
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        allowed = {curve for curve, value in paper["per_curve"][tag].items()
                   if value.get("included")}
        with np.load(args.data_dir / f"{tag}.npz", allow_pickle=True) as data:
            scores, tags = score_matrix(device, data, allowed)
            valid = np.flatnonzero(data["ok"])
            local = (np.asarray(data["LOCAL"], dtype=np.float64)[valid]
                     if "LOCAL" in data.files else None)
            theta = np.asarray(data["THETA"], dtype=np.float64)[valid]
            names = ([str(value) for value in data["parameter_names"]]
                     if "parameter_names" in data.files else list(PARAMS7))
        balanced = regime_balanced(scores, tags)
        tail = cvar(scores, args.tail_fraction)
        # A raw RRMS of several units on an off-state/noise-floor curve must
        # not have unlimited leverage. The capped objective treats it as a
        # failed/censored curve while preserving discrimination in the range
        # occupied by reliable strong-current curves.
        capped_scores = np.minimum(scores, args.rrms_cap)
        capped_balanced = regime_balanced(capped_scores, tags)
        # Normalize by the best complete stored curve to distinguish an error
        # that the physical parameter surface can improve from one that is at
        # its empirical/noise floor. This is computed for any new measured
        # device and does not use a device identity or paper-card target.
        attainable = np.min(scores, axis=0)
        excess_regret = np.maximum(scores - attainable[None, :], 0.0) / np.maximum(
            attainable[None, :], args.regret_floor)
        regret_balanced = regime_balanced(excess_regret, tags)
        regret_tail = cvar(excess_regret, args.tail_fraction)
        objectives = {
            "paper_mean": np.mean(scores, axis=1),
            "curve_rms": np.sqrt(np.mean(scores ** 2, axis=1)),
            "regime_cvar": ((1.0 - args.tail_weight) * balanced
                            + args.tail_weight * tail),
            "capped_regime": capped_balanced,
            "attainable_regret": (
                (1.0 - args.tail_weight) * regret_balanced
                + args.tail_weight * regret_tail),
            "minimax": np.max(scores, axis=1),
            "high_voltage": np.mean(scores[:, [bias(tag_name) >= 1.48
                                                  for tag_name in tags]], axis=1),
        }
        policies = {}
        for policy, objective in objectives.items():
            index = int(np.argmin(objective))
            policies[policy] = {
                "sample_index": int(valid[index]),
                "selection_objective": float(objective[index]),
                "params": {name: float(value)
                           for name, value in zip(names, theta[index])},
                **metrics(scores[index], tags),
            }
            if local is not None:
                policies[policy]["local"] = local[index].tolist()
        records.append({
            "device": tag, "dev_type": device.dev_type,
            "L_um": device.L_um, "W_um": device.W_um,
            "policies": policies,
        })
        logger.info("%-22s mean=%.4f robust=%.4f minimax=%.4f",
                    tag, policies["paper_mean"]["overall_rrms"],
                    policies["regime_cvar"]["overall_rrms"],
                    policies["minimax"]["overall_rrms"])
    policies = list(records[0]["policies"])
    payload = {
        "definition": {
            "scope": "real stored NGSpice seven-parameter LHC",
            "device_specific_training_groups": False,
            "regime_groups": ["idvd ordinary", "idvd high",
                              "idvg ordinary", "idvg high"],
            "high_voltage_floor": 1.48,
            "tail_fraction": args.tail_fraction,
            "tail_weight": args.tail_weight,
            "raw_rrms_cap": args.rrms_cap,
            "attainable_regret_denominator_floor": args.regret_floor,
            "regime_cvar_objective": (
                "(1-tail_weight)*equal four-regime mean + "
                "tail_weight*worst tail_fraction of individual curves"),
        },
        "summary": {policy: aggregate(records, policy, paper)
                    for policy in policies},
        "devices": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
