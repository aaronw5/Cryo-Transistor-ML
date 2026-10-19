#!/usr/bin/env python3
"""E4 fits: train-curve-only NGSpice fits, multistart members, Laplace draws.

Only REAL_DEV devices (out/research/explore/real_split.json) are accepted.
Writes out/research/explore/E4/<run>/<split>/<tag>.{json,npz}; devices whose
files already exist are skipped (resume).

  PYTHONPATH=src .venv/bin/python scripts/research/explore/e4_fit.py \
      --spec out/cold_loop/full/round_1/spec.json --run p43 --split vd
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from cryoml.explore.e4_heldout import E4_DIR, fit_device, real_split, save_record  # noqa: E402


def job(args):
    tag, spec, split, opts, out_dir = args
    record = fit_device(tag, spec, split, opts)
    save_record(record, Path(out_dir))
    return tag, record["runtime_s"], record["n_sim"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="vd", choices=("vd", "vg", "sparse"))
    p.add_argument("--devices", default=None, help="comma list (REAL_DEV only)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--cma-evals", type=int, default=6000)
    p.add_argument("--n-cma-extra", type=int, default=2)
    p.add_argument("--cma-evals-extra", type=int, default=3000)
    p.add_argument("--sigma0", type=float, default=0.3)
    p.add_argument("--gn-nfev", type=int, default=40)
    p.add_argument("--n-perturb", type=int, default=8)
    p.add_argument("--perturb-sigma", type=float, default=0.3)
    p.add_argument("--perturb-nfev", type=int, default=20)
    p.add_argument("--n-draws", type=int, default=64)
    p.add_argument("--prior-sd", type=float, default=0.5)
    p.add_argument("--jac-step", type=float, default=0.01)
    p.add_argument("--near-tol", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=20261019)
    p.add_argument("--no-heldout-feasibility", action="store_true",
                   help="original prereg fit (no penalty for failed held-out-bias sims)")
    a = p.parse_args()
    split_file = real_split()
    dev = list(split_file["REAL_DEV"])
    tags = dev if not a.devices else a.devices.split(",")
    bad = [t for t in tags if t not in dev]
    if bad:
        raise SystemExit(f"not REAL_DEV devices (refused): {bad}")
    spec = json.loads(Path(a.spec).read_text())
    opts = {k: getattr(a, k) for k in (
        "cma_evals", "n_cma_extra", "cma_evals_extra", "sigma0", "gn_nfev",
        "n_perturb", "perturb_sigma", "perturb_nfev", "n_draws", "prior_sd",
        "jac_step", "near_tol", "seed")}
    opts["heldout_feasibility"] = not a.no_heldout_feasibility
    out_dir = E4_DIR / a.run / a.split
    out_dir.mkdir(parents=True, exist_ok=True)
    opt_file = out_dir / "options.json"
    recorded = tags
    if opt_file.exists() and any(out_dir.glob("*.npz")):
        # Resume: refuse to mix records fitted with different settings
        # (records written before amendment 1 have no heldout_feasibility key
        # and used the original fit).
        old = json.loads(opt_file.read_text())
        diff = {}
        for k, v in {"spec": a.spec, **opts}.items():
            was = old.get(k, False if k == "heldout_feasibility" else None)
            if was != v:
                diff[k] = (was, v)
        if diff:
            raise SystemExit(f"options differ from existing records in {out_dir}: {diff}")
        recorded = sorted(set(tags) | set(old.get("devices", [])), key=dev.index)
    opt_file.write_text(json.dumps(
        {"spec": a.spec, "split": a.split, "devices": recorded, **opts}, indent=1) + "\n")
    todo = [t for t in tags if not (out_dir / f"{t}.npz").exists()]
    todo.sort(key=lambda t: not t.startswith("pmos"))
    t0 = time.time()
    jobs = [(t, spec, a.split, opts, str(out_dir)) for t in todo]
    if a.workers <= 1:
        for j in jobs:
            print(*job(j), flush=True)
    else:
        with get_context("spawn").Pool(min(a.workers, len(jobs) or 1)) as pool:
            for res in pool.imap_unordered(job, jobs):
                print(*res, flush=True)
    print(f"done {len(todo)} devices in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
