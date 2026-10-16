#!/usr/bin/env python3
"""NGSpice banks and held-out truth devices for the simulated temperature study.

Bank: per device, a Latin hypercube over the 14-dimensional temperature card
(``cryoml.temperature``) in a box ``--m-scale`` times the anchor round's
log half-widths, g log-uniform over [1/4, 4]; every sample is simulated at all
seven temperatures (four observed, three held out) in one NGSpice process.

Truth sets: ``--truth-sets`` x 18 cards drawn from a narrower box
(``--truth-m-scale``, g in [1/3, 3]), simulated the same way.  Methods see only
the observed temperatures of a truth device; the held-out temperatures score
interpolation.

  PYTHONPATH=src .venv/bin/python scripts/research/temp_bank.py --n 10000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cryoml.config import OUT_DIR, PROCESSED_DIR  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.spice_pdk import simulate_pdk  # noqa: E402
from cryoml.temperature import (NAMES, TEMPS_ALL, TEMPS_OBSERVED,  # noqa: E402
                                card_params, device_boxes)
from cryoml.utils import device_tag  # noqa: E402

ANCHOR = OUT_DIR / "cold_loop" / "seven" / "round_4" / "spec.json"


def simulate_chunk(job):
    tag, z_rows, m_scale, g_range = job
    device = next(d for d in PAPER_DEVICES
                  if device_tag(d.dev_type, d.L_um, d.W_um) == tag)
    curves = load_device_curves(device)
    box = device_boxes(ANCHOR, m_scale, g_range)[tag]
    out = np.full((len(z_rows), len(TEMPS_ALL), sum(len(c.Id) for c in curves)),
                  np.nan, dtype=np.float32)
    for i, z in enumerate(z_rows):
        sims = simulate_pdk(device.dev_type, device.L_um, device.W_um, curves,
                            params=card_params(box.stock, z),
                            temps_K=list(TEMPS_ALL), timeout_s=20)
        for j, per_temp in enumerate(sims):
            out[i, j] = np.concatenate(per_temp)
    return tag, out


def curve_layout(device):
    curves = load_device_curves(device)
    slices, start = [], 0
    for c in curves:
        slices.append((start, start + len(c.Id)))
        start += len(c.Id)
    return {"kinds": np.array([c.kind for c in curves]),
            "fixeds": np.array([float(c.fixed) for c in curves]),
            "slices": np.array(slices),
            "Vg": np.concatenate([np.asarray(c.Vg, dtype=np.float64) for c in curves]),
            "Vd": np.concatenate([np.asarray(c.Vd, dtype=np.float64) for c in curves]),
            "meas": np.concatenate([np.asarray(c.Id, dtype=np.float64) for c in curves])}


def run(tags, z_by_tag, m_scale, g_range, workers, chunk):
    jobs = [(tag, z[i:i + chunk], m_scale, g_range)
            for tag in tags for z in [z_by_tag[tag]]
            for i in range(0, len(z), chunk)]
    parts = {tag: [] for tag in tags}
    with get_context("spawn").Pool(workers) as pool:
        for k, (tag, block) in enumerate(pool.imap(simulate_chunk, jobs)):
            parts[tag].append(block)
            if (k + 1) % 20 == 0:
                print(f"{k + 1}/{len(jobs)} chunks", flush=True)
    return {tag: np.concatenate(parts[tag]) for tag in tags}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--m-scale", type=float, default=1.5)
    parser.add_argument("--truth-sets", type=int, default=3)
    parser.add_argument("--truth-m-scale", type=float, default=0.75)
    parser.add_argument("--out", default=str(PROCESSED_DIR / "temp_bank" / "v1"))
    parser.add_argument("--truth-out", default=str(OUT_DIR / "research" / "temp" / "truth"))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--chunk", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--skip-bank", action="store_true")
    args = parser.parse_args()
    tags = [device_tag(d.dev_type, d.L_um, d.W_um) for d in PAPER_DEVICES]
    devices = dict(zip(tags, PAPER_DEVICES))
    t0 = time.time()

    truth_dir = Path(args.truth_out)
    truth_dir.mkdir(parents=True, exist_ok=True)
    truth_range = (1 / 3, 3.0)
    for k in range(args.truth_sets):
        path = truth_dir / f"set_{k}.npz"
        if path.exists():
            continue
        rng = np.random.default_rng(args.seed + 1000 + k)
        boxes = device_boxes(ANCHOR, args.truth_m_scale, truth_range)
        z = {t: boxes[t].local_to_z(rng.uniform(-1, 1, (1, len(NAMES))))
             for t in tags}
        sims = run(tags, z, args.truth_m_scale, truth_range, args.workers, 1)
        np.savez(path, tags=np.array(tags), names=np.array(NAMES),
                 temps=np.array(TEMPS_ALL), observed=np.array(TEMPS_OBSERVED),
                 **{f"{t}__z": z[t][0] for t in tags},
                 **{f"{t}__ids": sims[t][0] for t in tags})
        bad = [t for t in tags if not np.all(np.isfinite(sims[t]))]
        print(f"truth set {k}: non-finite {bad}", flush=True)

    if args.skip_bank:
        return 0
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    g_range = (0.25, 4.0)
    boxes = device_boxes(ANCHOR, args.m_scale, g_range)
    todo = [t for t in tags if not (out_dir / f"{t}.npz").exists()]
    sampler = {t: qmc.LatinHypercube(d=len(NAMES), seed=args.seed + i)
               for i, t in enumerate(tags)}
    local = {t: 2 * sampler[t].random(args.n) - 1 for t in tags}
    z = {t: boxes[t].local_to_z(local[t]) for t in todo}
    # One device at a time keeps partial progress on disk.
    for tag in todo:
        ids = run([tag], {tag: z[tag]}, args.m_scale, g_range, args.workers,
                  args.chunk)[tag]
        ok = np.all(np.isfinite(ids), axis=(1, 2))
        np.savez(out_dir / f"{tag}.npz", Z=z[tag], LOCAL=local[tag], IDS=ids,
                 ok=ok, temps=np.array(TEMPS_ALL), observed=np.array(TEMPS_OBSERVED),
                 names=np.array(NAMES), lo=boxes[tag].lo, hi=boxes[tag].hi,
                 stock=np.array([boxes[tag].stock[p] for p in NAMES[:7]
                                 for p in [p[2:]]]),
                 m_scale=args.m_scale, **curve_layout(devices[tag]))
        print(f"{tag}: ok {ok.mean():.3f} {time.time() - t0:.0f}s", flush=True)
    (out_dir / "meta.json").write_text(json.dumps({
        "anchor_spec": str(ANCHOR), "m_scale": args.m_scale, "g_range": g_range,
        "n": args.n, "temps": TEMPS_ALL, "observed": TEMPS_OBSERVED,
        "names": NAMES, "seed": args.seed}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
