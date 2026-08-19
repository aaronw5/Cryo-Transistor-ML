#!/usr/bin/env python3
"""Build polarity-general bounds from repeated physical boundary probes."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def parse_coverage(text: str) -> dict[str, float]:
    """Parse ``name:multiplier`` specs for symmetric coverage widening."""
    coverage = {}
    for item in (part.strip() for part in text.split(",") if part.strip()):
        name, _, value = item.partition(":")
        if not name or not value:
            raise ValueError(f"expected name:multiplier, got {item!r}")
        multiplier = float(value)
        if multiplier <= 1.0:
            raise ValueError(f"coverage multiplier must exceed one: {item!r}")
        coverage[name.strip()] = multiplier
    return coverage


def apply_coverage(result: dict, coverage: dict[str, float]) -> list[dict]:
    """Widen named parameters symmetrically on every device of every polarity.

    This is training-set coverage, not a claim that the wider values improve a
    card.  The unchanged paper scorer and the section guard still decide which
    card is selected; the box only controls what the surrogate can ever see.
    """
    changes = []
    for tag, bounds in result["bounds_by_device"].items():
        for name, multiplier in coverage.items():
            if name not in bounds:
                continue
            center = float(result["published_by_device"][tag][name])
            old_lo, old_hi = (float(value) for value in bounds[name])
            if old_hi <= old_lo:
                continue
            new_lo = center - multiplier * (center - old_lo)
            new_hi = center + multiplier * (old_hi - center)
            bounds[name] = [min(old_lo, new_lo), max(old_hi, new_hi)]
            for direction in ("upper", "lower"):
                changes.append({
                    "device": tag, "parameter": name,
                    "dev_type": tag.split("_")[0], "direction": direction,
                    "multiplier": multiplier, "published": center,
                    "old_bounds": [old_lo, old_hi],
                    "new_bounds": list(bounds[name]),
                    "source": "coverage",
                })
    return changes


def build_bounds(sensitivity: dict, probe: dict,
                 max_multiplier: float,
                 coverage: dict[str, float] | None = None) -> dict:
    result = copy.deepcopy(sensitivity)
    groups, changes = [], []
    for recommendation in probe["recommendations"]:
        if not recommendation.get("recommend_expansion"):
            continue
        supported = recommendation.get("supported_multiplier")
        if supported is None:
            raise SystemExit(
                "probe file predates the supported-multiplier rule; rerun "
                "scripts/pdk_adaptive_boundary_probe.py before building "
                "adaptive bounds")
        parameter = recommendation["parameter"]
        polarity = recommendation["dev_type"]
        direction = recommendation["direction"]
        multiplier = min(float(supported), max_multiplier)
        group = {
            "parameter": parameter, "dev_type": polarity,
            "direction": direction, "multiplier": multiplier,
            "evidence_devices": next(
                row["improving_devices"] for row in recommendation["levels"]
                if row["multiplier"] == multiplier),
            "n_improving_devices": recommendation["n_improving_devices"],
        }
        groups.append(group)
        for tag, bounds in result["bounds_by_device"].items():
            if not tag.startswith(polarity):
                continue
            center = float(result["published_by_device"][tag][parameter])
            old_lo, old_hi = (float(value) for value in bounds[parameter])
            new_lo, new_hi = old_lo, old_hi
            if direction == "upper":
                new_hi = center + multiplier * (old_hi - center)
            elif direction == "lower":
                new_lo = center - multiplier * (center - old_lo)
            else:
                raise ValueError(f"unknown direction {direction}")
            bounds[parameter] = [min(old_lo, new_lo), max(old_hi, new_hi)]
            changes.append({
                "device": tag, "parameter": parameter,
                "dev_type": polarity, "direction": direction,
                "old_bounds": [old_lo, old_hi],
                "new_bounds": list(bounds[parameter]),
                "published": center, "multiplier": multiplier,
            })
    coverage_changes = apply_coverage(result, coverage or {})
    result["coverage_expansion"] = {
        "multipliers": dict(coverage or {}),
        "policy": (
            "symmetric widening of named parameters on every device of every "
            "polarity, applied for surrogate training coverage after direct "
            "NGSpice probes showed the required values lie outside the "
            "one-at-a-time box; it is not evidence that the wider values "
            "improve a card"),
        "changes": coverage_changes,
    }
    # Newly opened coverage regions also drive the adaptive-mixture slabs, so
    # the widened directions are sampled densely and in pairs instead of being
    # diluted across a uniformly larger box.
    result["adaptive_changes"] = changes + coverage_changes
    result["adaptive_boundary_source"] = probe.get("definition", {})
    result["adaptive_groups"] = groups
    result["evidence_changes"] = changes
    result["adaptive_policy"] = (
        "Only repeated physical recommendations are expanded; each supported "
        "parameter/polarity/direction is applied to every device of that "
        "polarity, including devices that did not trigger the diagnosis.")
    return result


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# Adaptive polarity-general parameter bounds", "",
        payload["adaptive_policy"], "",
        "| Parameter | Polarity | Direction | Multiplier | Evidence devices |",
        "|---|---|---|---:|---|",
    ]
    for row in payload["adaptive_groups"]:
        lines.append(
            f"| {row['parameter']} | {row['dev_type']} | {row['direction']} | "
            f"{row['multiplier']:.3g} | "
            f"{', '.join(row['evidence_devices'])} |")
    lines.extend(["", f"Expanded device coordinates: "
                  f"{len(payload['evidence_changes'])}.", ""])
    coverage = payload.get("coverage_expansion", {})
    if coverage.get("multipliers"):
        lines.extend(["## Coverage widening", "", coverage["policy"], "",
                      "| Parameter | Multiplier |", "|---|---:|"])
        for name, multiplier in coverage["multipliers"].items():
            lines.append(f"| {name} | {multiplier:g} |")
        lines.extend(["", f"Widened device coordinates: "
                      f"{len(coverage['changes'])}.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sensitivity", type=Path,
        default=Path("out/pdk_parameter_sensitivity/summary.json"))
    parser.add_argument(
        "--probe", type=Path,
        default=Path("out/tables/adaptive_boundary_probe.json"))
    parser.add_argument("--max-multiplier", type=float, default=4.0)
    parser.add_argument(
        "--coverage-expansion", default="",
        help="comma-separated name:multiplier specs widened symmetrically on "
             "every device, for surrogate training coverage")
    parser.add_argument(
        "--out", type=Path,
        default=Path("out/tables/adaptive_parameter_bounds.json"))
    parser.add_argument(
        "--markdown", type=Path,
        default=Path("out/tables/adaptive_parameter_bounds.md"))
    args = parser.parse_args()
    if args.max_multiplier <= 1.0:
        parser.error("--max-multiplier must be greater than one")
    sensitivity = json.loads(args.sensitivity.read_text())
    probe = json.loads(args.probe.read_text())
    payload = build_bounds(sensitivity, probe, args.max_multiplier,
                           parse_coverage(args.coverage_expansion))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    write_markdown(payload, args.markdown)
    print(json.dumps({
        "n_groups": len(payload["adaptive_groups"]),
        "n_device_coordinates": len(payload["evidence_changes"]),
        "n_coverage_coordinates": len(
            payload["coverage_expansion"]["changes"]),
        "coverage_multipliers": payload["coverage_expansion"]["multipliers"],
        "groups": payload["adaptive_groups"],
    }, indent=2))


if __name__ == "__main__":
    main()
