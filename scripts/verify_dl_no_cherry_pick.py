#!/usr/bin/env python3
"""Verify fixed-protocol deep-learning runs against the classical baseline.

This verifier intentionally forbids per-device or per-seed result selection:

* every run must contain exactly all 18 benchmark devices;
* all non-seed configuration fields must be identical;
* seeds must be unique;
* every saved NGSpice simulation is rescored with the paper-exact metric;
* every confirmatory run must beat the full-set CMA mean on its own.

The development seed may be identified separately, but it is still reported
and included in the all-runs mean.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.config import OUT_DIR, PROCESSED_DIR  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.metrics import device_rrms  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402


def load_classical(directory: Path) -> tuple[dict[str, float], dict[str, int]]:
    scores = {}
    evals = {}
    for path in directory.glob("*.json"):
        row = json.loads(path.read_text())
        if isinstance(row, dict) and "device" in row and "rrms" in row:
            scores[row["device"]] = float(row["rrms"])
            evals[row["device"]] = int(row.get("evals", -1))
    return scores, evals


def verify_run(directory: Path, expected: set[str]) -> tuple[dict, dict[str, float]]:
    config_path = directory / "run_config.json"
    if not config_path.exists():
        raise RuntimeError(f"{directory}: missing run_config.json")
    config = json.loads(config_path.read_text())
    if config.get("devices") != "all_18":
        raise RuntimeError(f"{directory}: run is not declared as all_18")

    record_paths = sorted(directory.glob("ml_*.json"))
    records = {json.loads(path.read_text())["device"]: path for path in record_paths}
    if set(records) != expected:
        missing = sorted(expected - set(records))
        extra = sorted(set(records) - expected)
        raise RuntimeError(f"{directory}: device mismatch missing={missing} extra={extra}")

    rescored = {}
    for device in PAPER_DEVICES:
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        record = json.loads(records[tag].read_text())
        if record.get("best_method") != "tandem_best+fd":
            raise RuntimeError(
                f"{directory}: {tag} selected {record.get('best_method')} "
                "instead of fixed final stage tandem_best+fd"
            )
        sims_path = directory / f"sims_{tag}.npz"
        if not sims_path.exists():
            raise RuntimeError(f"{directory}: missing {sims_path.name}")
        saved = np.load(sims_path)
        curves = load_device_curves(device)
        sims = [np.asarray(saved[f"sim_{i}"]) for i in range(len(curves))]
        rrms = float(device_rrms(sims, [curve.Id for curve in curves])["rrms"])
        if not np.isclose(rrms, float(record["rrms"]), rtol=0.0, atol=1e-12):
            raise RuntimeError(
                f"{directory}: saved simulation mismatch for {tag}: "
                f"{rrms} != {record['rrms']}"
            )
        if record.get("seed", config["seed"]) != config["seed"]:
            raise RuntimeError(f"{directory}: record/config seed mismatch for {tag}")
        rescored[tag] = rrms
    return config, rescored


def verify_deep_models(protocol: dict, expected: set[str]) -> dict:
    if protocol.get("method") != "tandem_multihead_mlp":
        raise RuntimeError(f"unexpected method: {protocol.get('method')}")
    if protocol.get("extra_emu_dir"):
        raise RuntimeError("confirmatory protocol unexpectedly uses an extra emulator")

    emu_dir = Path(protocol["emu_dir"])
    architectures = set()
    parameter_counts = []
    training_samples = []
    for tag in sorted(expected):
        path = emu_dir / f"emu_{tag}.pt"
        if not path.exists():
            raise RuntimeError(f"missing deep emulator: {path}")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        architecture = tuple(int(width) for width in blob["emu_sizes"])
        if len(architecture) < 2:
            raise RuntimeError(f"{path}: emulator is not a deep network")
        architectures.add(architecture)
        parameter_counts.append(sum(int(value.numel()) for value in blob["state"].values()))

        synth_path = PROCESSED_DIR / "pdk_synth" / f"{tag}.npz"
        if not synth_path.exists():
            raise RuntimeError(f"missing synthetic training pool: {synth_path}")
        synth = np.load(synth_path, allow_pickle=True)
        training_samples.append(int(synth["IDS"].shape[0]))

    return {
        "n_deep_emulators": len(expected),
        "deep_emulator_architectures": [list(arch) for arch in sorted(architectures)],
        "deep_emulator_parameter_count_min": min(parameter_counts),
        "deep_emulator_parameter_count_max": max(parameter_counts),
        "synthetic_training_samples_per_device_min": min(training_samples),
        "synthetic_training_samples_per_device_max": max(training_samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--classical-dir", default=str(OUT_DIR / "pdk_cma"))
    parser.add_argument("--development-seed", type=int, default=0)
    parser.add_argument("--out", default=str(OUT_DIR / "dl_no_cherry_pick.json"))
    args = parser.parse_args()

    expected = {
        device_tag(device.dev_type, device.L_um, device.W_um)
        for device in PAPER_DEVICES
    }
    classical, classical_evals = load_classical(Path(args.classical_dir))
    if set(classical) != expected:
        raise RuntimeError("classical baseline does not contain exactly all 18 devices")
    if set(classical_evals.values()) != {8500}:
        raise RuntimeError(
            f"classical comparator is not uniformly the 8500-eval control: "
            f"{sorted(set(classical_evals.values()))}"
        )
    classical_mean = float(np.mean(list(classical.values())))

    configs = []
    scores = []
    for raw in args.runs:
        directory = Path(raw)
        config, rescored = verify_run(directory, expected)
        configs.append(config)
        scores.append((directory, config["seed"], rescored))

    comparable = [{k: v for k, v in config.items() if k != "seed"} for config in configs]
    if any(config != comparable[0] for config in comparable[1:]):
        raise RuntimeError("run configurations differ beyond seed")
    seeds = [config["seed"] for config in configs]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError(f"duplicate seeds are not independent: {seeds}")
    deep_model_evidence = verify_deep_models(comparable[0], expected)

    rows = []
    failures = []
    for directory, seed, rescored in scores:
        mean = float(np.mean(list(rescored.values())))
        delta = mean - classical_mean
        wins = int(sum(rescored[tag] < classical[tag] for tag in expected))
        role = "development" if seed == args.development_seed else "confirmatory"
        passed = mean < classical_mean
        rows.append({
            "directory": str(directory),
            "seed": seed,
            "role": role,
            "mean_rrms": mean,
            "classical_mean_rrms": classical_mean,
            "delta_vs_classical": delta,
            "wins_vs_classical": wins,
            "n_devices": len(rescored),
            "passed": passed,
        })
        if role == "confirmatory" and not passed:
            failures.append(seed)

    all_values = [value for _, _, rescored in scores for value in rescored.values()]
    confirmatory_values = [
        value
        for _, seed, rescored in scores
        if seed != args.development_seed
        for value in rescored.values()
    ]
    confirmatory_means = [
        row["mean_rrms"] for row in rows if row["role"] == "confirmatory"
    ]
    report = {
        "protocol": comparable[0],
        "classical_mean_rrms": classical_mean,
        "classical_evals_per_device": 8500,
        "n_runs": len(rows),
        "n_confirmatory_runs": sum(row["role"] == "confirmatory" for row in rows),
        "all_run_device_mean_rrms": float(np.mean(all_values)),
        "confirmatory_run_device_mean_rrms": float(np.mean(confirmatory_values)),
        "worst_confirmatory_seed_mean_rrms": float(max(confirmatory_means)),
        "fixed_final_stage": "tandem_best+fd",
        "all_confirmatory_runs_pass": not failures,
        "deep_model_evidence": deep_model_evidence,
        "runs": rows,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(f"confirmatory seeds failed to beat classical: {failures}")
    if report["n_confirmatory_runs"] < 2:
        raise SystemExit("at least two confirmatory seeds are required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
