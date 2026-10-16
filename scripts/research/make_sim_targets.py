#!/usr/bin/env python3
"""Held-out simulated devices for the simulation-vs-measurement gap column.

For each of ``--sets`` target sets and every paper device, draw one card
log-uniformly from the device's ``--draw-spec`` box (by default the round-4
seven-parameter box, i.e. near where real fits land), clip it into the
``--clip-spec`` box (the round-1 cold box every method can reach), and
simulate it in NGSpice on the device's measured bias grid.  No bank contains
these cards (fresh seed).  Each set is written as an npz usable through
``CRYOML_TARGET_OVERRIDE``; true cards go to ``truth.json``.

  PYTHONPATH=src .venv/bin/python scripts/research/make_sim_targets.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
os.environ.pop("CRYOML_TARGET_OVERRIDE", None)

from cryoml.config import OUT_DIR  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402

TARGETS = OUT_DIR / "research" / "sim_targets"


def draw_device(job):
    from cold_loop import box_for
    from cryoml.card_eval import device_by_tag
    from cryoml.cold_start import device_bins
    from cryoml.data_io import load_device_curves
    from cryoml.spice_pdk import simulate_pdk
    tag, draw_spec, clip_spec, seed = job
    rng = np.random.default_rng(seed)
    box = box_for(draw_spec, tag)
    lo = np.asarray([clip_spec["bounds_by_device"][tag][n][0] for n in box.names])
    hi = np.asarray([clip_spec["bounds_by_device"][tag][n][1] for n in box.names])
    device = device_by_tag(tag)
    curves = load_device_curves(device)
    for attempt in range(200):
        local = np.where(box.active, rng.uniform(-1, 1, box.n_parameters), 0.0)
        values = box.local_to_array(local)
        values = np.clip(values, np.minimum(lo, hi), np.maximum(lo, hi))
        params = dict(zip(box.names, map(float, values)))
        sims = simulate_pdk(device.dev_type, device.L_um, device.W_um, curves,
                            params=params, bin_index=device_bins()[tag])
        if all(np.all(np.isfinite(s)) for s in sims):
            return tag, params, [np.asarray(s) for s in sims], attempt + 1
    raise RuntimeError(f"no converged draw for {tag}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draw-spec", default=str(
        OUT_DIR / "cold_loop" / "seven" / "round_4" / "spec.json"))
    parser.add_argument("--clip-spec", default=str(
        OUT_DIR / "cold_loop" / "seven" / "round_1" / "spec.json"))
    parser.add_argument("--sets", type=int, default=3)
    parser.add_argument("--seed", type=int, default=777001)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    draw_spec = json.loads(Path(args.draw_spec).read_text())
    clip_spec = json.loads(Path(args.clip_spec).read_text())
    tags = [device_tag(d.dev_type, d.L_um, d.W_um) for d in PAPER_DEVICES]
    jobs = [(tag, draw_spec, clip_spec, args.seed + 1000 * k + i)
            for k in range(args.sets) for i, tag in enumerate(tags)]
    with get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(draw_device, jobs)
    TARGETS.mkdir(parents=True, exist_ok=True)
    truth = {"draw_spec": args.draw_spec, "clip_spec": args.clip_spec,
             "seed": args.seed, "sets": {}}
    for k in range(args.sets):
        chunk = results[k * len(tags):(k + 1) * len(tags)]
        arrays = {f"{tag}__{i}": sim for tag, _p, sims, _a in chunk
                  for i, sim in enumerate(sims)}
        np.savez(TARGETS / f"set_{k}.npz", **arrays)
        truth["sets"][str(k)] = {tag: {"params": params, "attempts": attempts}
                                 for tag, params, _s, attempts in chunk}
    (TARGETS / "truth.json").write_text(json.dumps(truth, indent=1) + "\n")
    print(json.dumps({k: {t: v["attempts"] for t, v in s.items()}
                      for k, s in truth["sets"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
