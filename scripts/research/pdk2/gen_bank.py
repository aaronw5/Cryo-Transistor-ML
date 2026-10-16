#!/usr/bin/env python3
"""10k-sample Latin-hypercube NGSpice banks for the pdk2 devices
(``--pdk gf180``, default: GF180MCU BSIM4 on ngspice-41; ``--pdk ihp``: IHP
SG13G2 PSP103 on ngspice-46, see ``cryoml.pdk2.ihp``).

Same npz schema as ``scripts/pdk_gen_full_data.py`` (the cold-loop banks,
e.g. ``data/processed/cold_loop/full_r1/<tag>.npz``): ``LOCAL`` (float32),
``THETA``, ``IDS`` (float32 positive drain-current magnitudes), ``ok``,
``Vg``, ``Vd``, ``meas``, ``slices``, ``kinds``, ``fixeds`` (signed volts),
``bin_index``, ``parameter_names``, ``parameter_signature``, ``published``,
``lo``, ``hi``, ``active``, ``log_scale``, ``seed``, ``sampling``
(``global_lhc``: row 0 is the stock card), plus pdk2 keys: ``fixeds_canonical``
(the sky130-equivalent biases, for ``foundation.canonical_point_order``),
``pdk``, ``temp_C`` and ``tag``.  ``meas`` holds the stock-card simulation;
held-out truth targets come from ``make_truth.py``.  Curves are stored in
canonical order (Id-Vd at Vg = 0.2..1 x VDD, then Id-Vg at Vd = 0.05,
0.2..1 x VDD; 186 points each; VDD 3.3 V GF180, 1.2 V IHP).  ``manifest.json`` and the box spec
(``spec.json``, cold-start layout) are written next to the banks.

  PYTHONPATH=src .venv/bin/python scripts/research/pdk2/gen_bank.py \
      --set seven --workers 4
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from cryoml.config import PROCESSED_DIR  # noqa: E402
from cryoml.pdk2 import boxes  # noqa: E402

DEFAULT_SEED = 20261016


def curve_layout_arrays(curves) -> dict:
    slices, cursor = [], 0
    for curve in curves:
        slices.append((cursor, cursor + len(curve.Id)))
        cursor += len(curve.Id)
    return {
        "Vg": np.concatenate([c.Vg for c in curves]).astype(np.float64),
        "Vd": np.concatenate([c.Vd for c in curves]).astype(np.float64),
        "slices": np.asarray(slices, dtype=np.int64),
        "kinds": np.asarray([c.kind for c in curves]),
        "fixeds": np.asarray([c.fixed for c in curves], dtype=np.float64),
        "fixeds_canonical": np.asarray(
            [c.meta["canonical_fixed"] for c in curves], dtype=np.float64),
    }


def sample_local(box, n: int, seed: int) -> np.ndarray:
    """Global LHC on the active coordinates; row 0 is the stock card."""
    local = np.zeros((n, box.n_parameters))
    sampler = qmc.LatinHypercube(d=box.n_active, seed=seed)
    local[:, box.active] = 2.0 * sampler.random(n) - 1.0
    local[0] = box.local_published
    return local


def generate_device(job) -> dict:
    tag, box_spec, n, seed, out_dir, chunk = job
    t0 = time.time()
    sim = boxes.backend(boxes.pdk_of_tag(tag))
    device = sim.device_by_tag(tag)
    box = boxes.box_from_spec(box_spec, tag)
    curves = sim.device_curves(device.dev_type)
    local = sample_local(box, n, seed)
    theta = box.local_to_array(local)
    params = [dict(zip(box.names, map(float, row))) for row in theta]
    sims = sim.simulate_many(device, params, curves=curves, chunk=chunk)
    ids = np.stack([np.concatenate(s) for s in sims]).astype(np.float32)
    ok = np.isfinite(ids).all(axis=1)
    layout = curve_layout_arrays(curves)
    path = Path(out_dir) / f"{tag}.npz"
    np.savez_compressed(
        path,
        LOCAL=local.astype(np.float32), THETA=theta.astype(np.float64),
        IDS=ids, ok=ok,
        meas=np.concatenate(sims[0]).astype(np.float64),
        bin_index=np.asarray(sim.bin_index(device)),
        parameter_names=np.asarray(box.names),
        parameter_signature=np.asarray(boxes.spec_signature(box_spec)),
        published=box.center.astype(np.float64),
        lo=box.lo.astype(np.float64), hi=box.hi.astype(np.float64),
        active=box.active, log_scale=box.log,
        seed=np.asarray(seed), sampling=np.asarray("global_lhc"),
        pdk=np.asarray(f"{sim.PDK_NAME}@{sim.PDK_VERSION[:12]}"),
        temp_C=np.asarray(sim.TEMP_C), tag=np.asarray(tag),
        **layout)
    return {"device": tag, "path": str(path), "n_samples": int(n),
            "n_valid": int(ok.sum()), "n_parameters": box.n_parameters,
            "n_active": box.n_active, "runtime_s": round(time.time() - t0, 1),
            "seed": int(seed)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdk", choices=("gf180", "ihp"), default="gf180")
    parser.add_argument("--set", choices=sorted(boxes.PARAMETER_SETS),
                        required=True)
    parser.add_argument("--devices", default="all",
                        help="comma-separated pdk2 tags (default: all 16)")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=200,
                        help="cards per ngspice process (altermod)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.num_samples < 2:
        parser.error("--num-samples must be at least two")

    sim = boxes.backend(args.pdk)
    out_dir = Path(args.out_dir or PROCESSED_DIR / "pdk2"
                   / f"{args.pdk}_{args.set}")
    out_dir.mkdir(parents=True, exist_ok=True)
    box_spec = boxes.spec(args.set, pdk=args.pdk)
    (out_dir / "spec.json").write_text(json.dumps(box_spec, indent=1) + "\n")
    signature = boxes.spec_signature(box_spec)
    devices = sim.parse_devices(args.devices)
    records, jobs = [], []
    for device in devices:
        index = sim.TAGS.index(device.tag)
        path = out_dir / f"{device.tag}.npz"
        if args.resume and path.exists():
            with np.load(path, allow_pickle=False) as data:
                if (str(data["parameter_signature"]) == signature
                        and len(data["LOCAL"]) == args.num_samples):
                    records.append({"device": device.tag, "path": str(path),
                                    "n_samples": args.num_samples,
                                    "n_valid": int(np.sum(data["ok"])),
                                    "resumed": True})
                    continue
        jobs.append((device.tag, box_spec, args.num_samples,
                     args.seed + 1000 * index, str(out_dir), args.chunk))
    if args.workers <= 1 or len(jobs) <= 1:
        iterator = map(generate_device, jobs)
        for record in iterator:
            records.append(record)
            print(json.dumps(record), flush=True)
    else:
        with get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
            for record in pool.imap_unordered(generate_device, jobs):
                records.append(record)
                print(json.dumps(record), flush=True)
    records.sort(key=lambda r: sim.TAGS.index(r["device"]))
    manifest = {
        "schema_version": 1, "pdk": f"{sim.PDK_NAME}@{sim.PDK_VERSION}",
        "parameter_set": args.set,
        "parameter_names": list(box_spec["selected_parameters"]),
        "parameter_signature": signature, "spec": str(out_dir / "spec.json"),
        "sampling": "global_lhc", "sampling_definition": "global Latin hypercube",
        "current_dtype": "float32", "temp_C": sim.TEMP_C,
        "num_samples_requested": args.num_samples, "seed": args.seed,
        "devices": records,
        "all_valid": all(r["n_valid"] == r["n_samples"] for r in records),
        "wall_runtime_summed": round(sum(r.get("runtime_s", 0.0)
                                         for r in records), 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "devices"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
