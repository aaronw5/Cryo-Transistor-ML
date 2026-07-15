#!/usr/bin/env python3
"""Scaling-law study for the standalone ML extraction pipeline.

Sweeps three axes independently around a reference config and measures, per
testbed device:
  - emulator held-out val MSE (signed-log space)
  - best NGSpice-validated RRMS of the top search candidates (no warm
    starts, no controls — pure ML scaling signal)
  - RRMS after a short FD polish

Axes (reference cell: n=10000, arch 512x4, 2048 starts x 600 steps):
  data:     n_train in {375, 750, 1500, 3000, 6000, 10000}
  capacity: arch in {64x3, 128x3, 256x3, 512x3, 512x4, 1024x4}
  search:   n_starts in {128, 512, 2048, 8192}

Only the first 10,000 rows of each confirmed-setup LHC dataset are used.

Appends one CSV row per (device, config) to out/scaling/results.csv as it
goes (crash-safe).

  python scripts/scaling_study.py --device mps --fresh
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pdk_ml_extract as mlx  # noqa: E402
from cryoml.config import OUT_DIR, PROCESSED_DIR, ensure_dirs  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES, parse_device_list  # noqa: E402
from cryoml.metrics import clean_current  # noqa: E402
from cryoml.pdk_extract import (LhcBox, PARAMS7, ThetaBox,  # noqa: E402
                                eval_params_new, new_metric_layout,
                                residual_fn_new)
from cryoml.utils import device_tag, get_logger, set_seed  # noqa: E402

logger = get_logger("scaling_study")

OUT_SCALING = OUT_DIR / "scaling"
POOL = 10000  # fixed LHC pool size (confirmed-setup datasets)

ALL_DEVICES = ",".join(
    f"{d.dev_type}:{d.L_um:g}:{d.W_um:g}" for d in PAPER_DEVICES)

REF = {"n_data": 10000, "arch": (512, 512, 512, 512), "n_starts": 2048,
       "steps": 600}

STUDY_SCHEMA = {
    "schema_version": 2,
    "setup": "CryoPDK_Skywater130nm_ML@39b1e518",
    "metric": "confirmed rrmsCalc with fixed baseline inclusion",
    "box": "lhc10",
    "pool": POOL,
    "reference": {**REF, "arch": list(REF["arch"])},
    "population": "all 18 paper Table-6 transistors",
}

ARCHES = {
    "64x3": (64, 64, 64),
    "128x3": (128, 128, 128),
    "256x3": (256, 256, 256),
    "512x3": (512, 512, 512),
    "512x4": (512, 512, 512, 512),
    "1024x4": (1024, 1024, 1024, 1024),
}


def configs():
    seen = set()
    for n in (375, 750, 1500, 3000, 6000, 10000):
        c = dict(REF, n_data=n, sweep="data")
        key = (c["n_data"], c["arch"], c["n_starts"])
        seen.add(key)
        yield c
    for name, arch in ARCHES.items():
        c = dict(REF, arch=arch, sweep="capacity")
        key = (c["n_data"], c["arch"], c["n_starts"])
        if key in seen:
            continue
        seen.add(key)
        yield c
    for s in (128, 512, 8192):
        yield dict(REF, n_starts=s, sweep="search")


def run_cell(d, cfg, tdev, seed, n_validate=4, polish_nfev=40):
    tag = device_tag(d.dev_type, d.L_um, d.W_um)
    t0 = time.time()
    set_seed(seed)
    rng = np.random.default_rng(seed)

    data = np.load(PROCESSED_DIR / "pdk_synth" / f"{tag}.npz",
                   allow_pickle=True)
    IDS = data["IDS"][:POOL]
    ok = data["ok"][:POOL]
    meas, slices = data["meas"], data["slices"]
    bin_index = int(data["bin_index"])
    published = {p: float(v) for p, v in zip(PARAMS7, data["published"])}
    box_mode = str(data["box_mode"]) if "box_mode" in data else "wide"
    if box_mode == "lhc10":
        box = LhcBox(dev_type=d.dev_type, bin_index=bin_index,
                     published=published)
        z_clamp = 8.0
    else:
        box = ThetaBox(dev_type=d.dev_type, bin_index=bin_index,
                       published=published)
        z_clamp = 3.5
    THETA = data["THETA"][:POOL].astype(np.float64)
    Z = np.stack([box.params_to_z({p: t[i] for i, p in enumerate(PARAMS7)})
                  for t in THETA])
    curves = load_device_curves(d)
    include_tags = mlx.baseline_include_tags(tag)
    layout = new_metric_layout(d.dev_type, d.L_um, d.W_um, curves,
                               include_tags)

    kept_mask = np.zeros(len(meas), dtype=bool)
    kept_span = {}
    meas_clean_parts = []
    off = 0
    for ci, (a, b) in enumerate(slices):
        m = np.asarray(meas[a:b], dtype=np.float64)
        den = np.mean(np.abs(m))
        if den > 0 and np.isfinite(den):
            kept_mask[a:b] = True
            kept_span[ci] = (off, off + b - a)
            meas_clean_parts.append(clean_current(m))
            off += b - a
    meas_clean_flat = np.concatenate(meas_clean_parts)
    curve_layout = [(kept_span[ci][0] + start, kept_span[ci][1], den)
                    for ci, start, _c, den, _t in layout.entries]

    idx_ok = np.where(ok)[0]
    n = min(cfg["n_data"], len(idx_ok))
    sub = rng.choice(idx_ok, size=n, replace=False)
    Zok, Yok = Z[sub], IDS[sub][:, kept_mask]

    dev = torch.device(tdev)
    Zt = torch.tensor(Zok, dtype=torch.float32, device=dev)
    Yt = torch.tensor(mlx.slog(Yok), dtype=torch.float32, device=dev)
    P = Yt.shape[1]

    emu = mlx.mlp([7, *cfg["arch"], P]).to(dev)
    n_params = sum(p.numel() for p in emu.parameters())
    emu_val = mlx.train_net(emu, Zt, Yt, dev, epochs=2000, lr=1e-3,
                            batch=4096 if len(Zt) > 4000 else None)
    train_s = time.time() - t0

    # multistart Adam search (same loss as the production pipeline)
    meas_t = torch.tensor(meas_clean_flat, dtype=torch.float32, device=dev)
    S = cfg["n_starts"]
    z0 = box.z_published
    if isinstance(box, LhcBox):
        u = rng.uniform(1e-3, 1 - 1e-3, size=(S - S // 2 - 1, 7))
        box_starts = np.log(u / (1 - u))
        local = z0 + rng.normal(0, 2.0, size=(S // 2, 7))
    else:
        box_starts = rng.uniform(-3.0, 3.0, size=(S - S // 2 - 1, 7))
        local = z0 + rng.normal(0, 0.7, size=(S // 2, 7))
    starts = np.concatenate([z0[None, :], local, box_starts], axis=0)
    zv = torch.tensor(starts, dtype=torch.float32, device=dev,
                      requires_grad=True)
    opt = torch.optim.Adam([zv], lr=0.05)
    best_loss = np.full(len(starts), np.inf)
    best_z = starts.copy()
    for _ in range(cfg["steps"]):
        opt.zero_grad()
        pred = mlx.current_from_slog(emu(zv))
        curve_rrms = []
        for a, b, den in curve_layout:
            rmse = torch.sqrt(torch.mean(
                (pred[:, a:b] - meas_t[a:b].unsqueeze(0)) ** 2, dim=1))
            curve_rrms.append(rmse / den)
        loss_per = torch.stack(curve_rrms, dim=1).sum(dim=1) / len(curve_layout)
        loss_per.sum().backward()
        opt.step()
        with torch.no_grad():
            zv.clamp_(-z_clamp, z_clamp)
        lv = loss_per.detach().cpu().numpy()
        improved = lv < best_loss
        if improved.any():
            zc = zv.detach().cpu().numpy().astype(np.float64)
            best_loss[improved] = lv[improved]
            best_z[improved] = zc[improved]
    order = np.argsort(best_loss)

    # NGSpice validation of top candidates (deduplicated)
    cands, seen = [], []
    for i in order:
        z = best_z[i]
        if any(np.linalg.norm(z - s) < 0.5 for s in seen):
            continue
        seen.append(z)
        cands.append(z)
        if len(cands) >= n_validate:
            break
    best_rrms, best_zv = np.inf, None
    for z in cands:
        params = box.z_to_params(z)
        m, _off, _ = eval_params_new(d.dev_type, d.L_um, d.W_um, bin_index,
                                     curves, params, include_tags)
        if np.isfinite(m["rrms"]) and m["rrms"] < best_rrms:
            best_rrms, best_zv = float(m["rrms"]), z

    rrms_polished = best_rrms
    if best_zv is not None and polish_nfev > 0:
        from scipy.optimize import least_squares
        try:
            sol = least_squares(
                lambda z: residual_fn_new(z, box, d.dev_type, d.L_um, d.W_um,
                                          bin_index, curves, layout),
                best_zv, method="trf", jac="2-point", diff_step=2e-2,
                max_nfev=polish_nfev)
            m, _off, _ = eval_params_new(d.dev_type, d.L_um, d.W_um,
                                         bin_index, curves,
                                         box.z_to_params(sol.x), include_tags)
            if np.isfinite(m["rrms"]):
                rrms_polished = min(best_rrms, float(m["rrms"]))
        except Exception:  # noqa: BLE001
            pass

    return {
        "device": tag, "sweep": cfg["sweep"], "n_data": n,
        "arch": "x".join(str(s) for s in cfg["arch"]), "n_params": n_params,
        "n_starts": cfg["n_starts"], "steps": cfg["steps"], "seed": seed,
        "emu_val": float(emu_val), "search_loss": float(best_loss[order[0]]),
        "rrms_raw": best_rrms, "rrms_polished": rrms_polished,
        "runtime_s": round(time.time() - t0, 1),
        "train_s": round(train_s, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--devices", default=ALL_DEVICES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fresh", action="store_true",
                    help="discard a previous scaling CSV/config before "
                         "starting; required when replacing the June study")
    ap.add_argument("--extend-devices", action="store_true",
                    help="Reuse a compatible completed device subset and "
                         "extend the same grid to the requested device set")
    ap.add_argument("--plan-only", action="store_true",
                    help="Validate/extend metadata and report remaining cells "
                         "without running training")
    ap.add_argument("--data-only", action="store_true",
                    help="Run the accepted final protocol only: data sizes "
                         "375 through 6000 on all requested devices")
    args = ap.parse_args()
    ensure_dirs()
    OUT_SCALING.mkdir(parents=True, exist_ok=True)

    out_csv = OUT_SCALING / "results.csv"
    config_path = OUT_SCALING / "run_config.json"
    if args.fresh:
        out_csv.unlink(missing_ok=True)
        config_path.unlink(missing_ok=True)
    expected_config = {**STUDY_SCHEMA, "devices": args.devices,
                       "seed": args.seed}
    if out_csv.exists():
        if not config_path.exists():
            raise RuntimeError(
                f"{out_csv} predates run metadata; rerun with --fresh to "
                "avoid mixing the June and confirmed-setup studies"
            )
        existing_config = json.loads(config_path.read_text())
        if existing_config != expected_config:
            population_metadata = {"devices", "testbed", "population"}
            existing_core = {k: v for k, v in existing_config.items()
                             if k not in population_metadata}
            expected_core = {k: v for k, v in expected_config.items()
                             if k not in population_metadata}
            existing_devices = {
                device_tag(d.dev_type, d.L_um, d.W_um)
                for d in parse_device_list(existing_config.get("devices"))
            }
            requested_devices = {
                device_tag(d.dev_type, d.L_um, d.W_um)
                for d in parse_device_list(args.devices)
            }
            compatible_extension = (
                args.extend_devices and existing_core == expected_core
                and existing_devices < requested_devices)
            if not compatible_extension:
                raise RuntimeError(
                    "scaling run configuration differs from the existing "
                    "CSV; use --fresh, --extend-devices for a strict superset, "
                    "or restore the matching configuration"
                )
            config_path.write_text(json.dumps(expected_config, indent=2)
                                   + "\n")
            logger.info("extending scaling population from %d to %d devices",
                        len(existing_devices), len(requested_devices))
    else:
        config_path.write_text(json.dumps(expected_config, indent=2) + "\n")
    fields = ["device", "sweep", "n_data", "arch", "n_params", "n_starts",
              "steps", "seed", "emu_val", "search_loss", "rrms_raw",
              "rrms_polished", "runtime_s", "train_s"]
    done = set()
    if out_csv.exists():
        with open(out_csv) as f:
            for row in csv.DictReader(f):
                done.add((row["device"], row["sweep"], row["n_data"],
                          row["arch"], row["n_starts"], row["seed"]))
    else:
        with open(out_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    devices = parse_device_list(args.devices)
    selected_configs = [
        cfg for cfg in configs()
        if not args.data_only
        or (cfg["sweep"] == "data" and cfg["n_data"] <= 6000)
    ]
    pending = 0
    for cfg in selected_configs:
        for d in devices:
            tag = device_tag(d.dev_type, d.L_um, d.W_um)
            key = (tag, cfg["sweep"], str(cfg["n_data"]),
                   "x".join(str(s) for s in cfg["arch"]),
                   str(cfg["n_starts"]), str(args.seed))
            pending += key not in done
    if args.plan_only:
        target = len(selected_configs) * len(devices)
        print(f"scaling cells complete={target - pending} pending={pending} "
              f"target={target} preserved_csv_rows={len(done)}")
        return 0

    for cfg in selected_configs:
        for d in devices:
            tag = device_tag(d.dev_type, d.L_um, d.W_um)
            key = (tag, cfg["sweep"], str(cfg["n_data"]),
                   "x".join(str(s) for s in cfg["arch"]),
                   str(cfg["n_starts"]), str(args.seed))
            if key in done:
                continue
            row = run_cell(d, cfg, args.device, args.seed)
            with open(out_csv, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
            logger.info("%-18s %-8s n=%-5d %-7s S=%-5d val=%.4f raw=%.3f "
                        "pol=%.3f (%.0fs)", row["device"], row["sweep"],
                        row["n_data"], row["arch"], row["n_starts"],
                        row["emu_val"], row["rrms_raw"],
                        row["rrms_polished"], row["runtime_s"])
    print(f"wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
