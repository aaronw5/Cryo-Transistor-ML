#!/usr/bin/env python3
"""Iterated cold-start rerun of the paper pipeline.

Round 1 centers every device's box on the multiplier-free card (no released
77 K multipliers).  Each round runs, strictly in sequence:

1. ``pdk_gen_full_data.py``: 10,000-point NGSpice LHC in the round's log box;
2. ``foundation_parameter_study.py --stage all``: forward surrogate trained
   from random initialization with the paper's settings, the tandem inverse
   (method A), and the paper's multistart surrogate-FD extraction with
   NGSpice-validated endpoints;
3. the next round's box: re-centered on each device's selected card, with the
   same shrink/hold rule for every device.

The released card never enters a box, a start, or a fallback.  Results are
appended to ``out/cold_loop/<set>/progress.json``.

  PYTHONPATH=src .venv/bin/python scripts/research/cold_loop.py --set seven \
      --rounds 1-4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cryoml.cold_start import next_round_spec, round_one_spec, write_spec  # noqa: E402
from cryoml.config import OUT_DIR, PROCESSED_DIR  # noqa: E402
from cryoml.parameter_sets import PARAMS7  # noqa: E402
from cryoml.parameter_space import ParameterBox  # noqa: E402

PYTHON = str(ROOT / ".venv" / "bin" / "python")


def names_for(parameter_set: str) -> tuple[str, ...]:
    if parameter_set == "seven":
        return tuple(PARAMS7)
    summary = json.loads(
        (OUT_DIR / "pdk_parameter_sensitivity" / "summary.json").read_text())
    return tuple(summary["selected_parameters"])


def round_paths(parameter_set: str, round_index: int) -> dict[str, Path]:
    base = OUT_DIR / "cold_loop" / parameter_set / f"round_{round_index}"
    return {
        "base": base, "spec": base / "spec.json",
        "bank": PROCESSED_DIR / "cold_loop" / f"{parameter_set}_r{round_index}",
        "cache": base / "cache", "extract": base / "extract",
    }


def run(command: list[str], log_path: Path) -> float:
    t0 = time.time()
    with log_path.open("a") as log:
        log.write(f"\n$ {' '.join(command)}\n")
        log.flush()
        proc = subprocess.run(command, cwd=ROOT, stdout=log,
                              stderr=subprocess.STDOUT,
                              env={**_env(), "PYTHONPATH": str(ROOT / "src")})
    if proc.returncode:
        raise RuntimeError(f"failed ({proc.returncode}): {' '.join(command)}")
    return round(time.time() - t0, 1)


def _env() -> dict:
    import os
    return dict(os.environ)


def box_for(spec: dict, tag: str) -> ParameterBox:
    names = tuple(spec["selected_parameters"])
    flags = spec["log_scale_by_device"][tag]
    return ParameterBox(names=names, published=spec["published_by_device"][tag],
                        bounds=spec["bounds_by_device"][tag],
                        log_scale=tuple(bool(flags[n]) for n in names))


def selected_cards(extract_dir: Path, method: str) -> dict[str, dict]:
    cards = {}
    for path in sorted(extract_dir.glob("ml_*.json")):
        row = json.loads(path.read_text())
        cards[row["device"]] = row["methods"][method]["params"]
    return cards


def run_round(parameter_set: str, round_index: int, args) -> dict:
    paths = round_paths(parameter_set, round_index)
    paths["base"].mkdir(parents=True, exist_ok=True)
    log = paths["base"] / "round.log"
    names = names_for(parameter_set)
    if not paths["spec"].exists():
        if round_index == 1:
            write_spec(round_one_spec(names), paths["spec"])
        else:
            previous = round_paths(parameter_set, round_index - 1)
            spec = json.loads(previous["spec"].read_text())
            cards = selected_cards(previous["extract"], "surrogate_fd")
            local = {tag: dict(zip(spec["selected_parameters"],
                                   box_for(spec, tag).params_to_local(card)))
                     for tag, card in cards.items()}
            write_spec(next_round_spec(spec, cards, local,
                                       shrink=args.shrink,
                                       round_index=round_index),
                       paths["spec"])
    timing = {}
    bank_manifest = paths["bank"] / "manifest.json"
    if not bank_manifest.exists():
        timing["bank_s"] = run([
            PYTHON, "scripts/pdk_gen_full_data.py",
            "--sensitivity", str(paths["spec"]),
            "--out-dir", str(paths["bank"]),
            "--num-samples", str(args.num_samples),
            "--workers", str(args.workers), "--resume"], log)
    summary_path = paths["extract"] / "summary.json"
    if not summary_path.exists():
        command = [
            PYTHON, "scripts/foundation_parameter_study.py",
            "--parameter-set", "full", "--stage", "all",
            "--data-dir", str(paths["bank"]),
            "--cache-dir", str(paths["cache"]),
            "--out-dir", str(paths["extract"]),
            "--extract-method", args.extract_method,
            "--epochs", str(args.epochs), "--resume"]
        if paths["cache"].exists() and not (paths["cache"] / "metadata.json").exists():
            command.append("--rebuild-cache")
        timing["study_s"] = run(command, log)
    summary = json.loads(summary_path.read_text())
    row = {"round": round_index, "parameter_set": parameter_set,
           "spec": str(paths["spec"]), **timing,
           "methods": {key: {k: value[k] for k in (
               "all_device_mean_rrms", "nmos_rrms", "pmos_rrms",
               "combined_rrms")} for key, value in summary["methods"].items()},
           "forward_test_loss": summary["training"]["test"]["loss"]}
    progress_path = OUT_DIR / "cold_loop" / parameter_set / "progress.json"
    progress = (json.loads(progress_path.read_text())
                if progress_path.exists() else [])
    progress = [r for r in progress if r["round"] != round_index] + [row]
    progress.sort(key=lambda r: r["round"])
    progress_path.write_text(json.dumps(progress, indent=1) + "\n")
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=("seven", "full"), required=True)
    parser.add_argument("--rounds", default="1-4")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--shrink", type=float, default=0.5)
    parser.add_argument("--extract-method", default="both",
                        choices=("both", "fd", "inverse"))
    parser.add_argument("--min-gain", type=float, default=0.002,
                        help="stop after a round improves all-device RRMS less")
    args = parser.parse_args()
    first, last = (int(v) for v in args.rounds.split("-"))
    best = None
    for round_index in range(first, last + 1):
        row = run_round(args.set, round_index, args)
        score = row["methods"]["surrogate_fd"]["all_device_mean_rrms"]
        print(json.dumps(row), flush=True)
        if best is not None and best - score < args.min_gain:
            print(f"stopping: round {round_index} gain {best - score:.5f}")
            break
        best = score if best is None else min(best, score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
