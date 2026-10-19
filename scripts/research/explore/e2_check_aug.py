#!/usr/bin/env python3
"""E2 check: output-space series-R and heating transforms vs NGSpice.

For each listed device, the tier-a set-0 seven-parameter card is simulated
(1) clean, (2) with external drain/source resistors R = rho / (W mult) per
side (``sky130_misspec.simulate_truth``), (3) with per-curve heating
(vth0_i, u0_i overrides as in the tier-c generator).  The output-space
transform of ``e2_direct.Augmenter.structured_fixed`` is applied to the clean
curves with the same draws.  Reported per case, as the mean over the 11
curves of RMS(x - y) / mean|y| (y = NGSpice-distorted curve):
``distortion`` (x = clean) and ``residual`` (x = transformed clean).

  PYTHONPATH=src .venv/bin/python scripts/research/explore/e2_check_aug.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from cryoml.card_eval import device_by_tag  # noqa: E402
from cryoml.cold_start import device_bins  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.explore import e2_direct as E  # noqa: E402
from cryoml.sky130_misspec import simulate_truth  # noqa: E402
from cryoml.spice_pdk import device_mult  # noqa: E402


def rel(x, y):
    return float(np.sqrt(np.mean((x - y) ** 2)) / max(np.mean(np.abs(y)), 1e-30))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="nmos_L0p15_W1p6,nmos_L1_W3,pmos_L2_W5,pmos_L0p35_W0p55")
    ap.add_argument("--rho", default="1000,5000,20000")
    ap.add_argument("--dtmax", type=float, default=30.0)
    args = ap.parse_args()
    truth = json.loads((E.TIER_A_DIR / "truth.json").read_text())
    bins = device_bins()
    out = {}
    for tag in args.devices.split(","):
        device = device_by_tag(tag)
        curves = load_device_curves(device)
        params = truth["sets"]["0"][tag]["params"]
        mult = device_mult(device.dev_type, device.L_um, device.W_um)
        with np.load(E.BANK_DIR / f"{tag}.npz", allow_pickle=True) as data:
            order, _ = E.canonical(data)
            layout = E.build_layout(data)
        aug = E.Augmenter(layout, np.full(18, 1.0 / (device.W_um * mult)))
        ids = torch.zeros(1, dtype=torch.long)
        clean = simulate_truth(device.dev_type, device.L_um, device.W_um, curves,
                               params, bin_index=bins[tag])
        clean_c = np.concatenate(clean)[order]
        x = torch.as_tensor(clean_c[None], dtype=torch.float32)
        inv = np.argsort(order)
        rows = {}

        def compare(name, target_list, transformed):
            y = np.concatenate(target_list)[order]
            t = transformed[0].numpy().astype(np.float64)
            dist, resid = [], []
            for c in range(11):
                sl = slice(c * 186, (c + 1) * 186)
                if not np.all(np.isfinite(y[sl])):
                    continue
                dist.append(rel(clean_c[sl], y[sl]))
                resid.append(rel(t[sl], y[sl]))
            rows[name] = {"distortion": float(np.mean(dist)),
                          "residual": float(np.mean(resid)),
                          "max_curve_distortion": float(np.max(dist)),
                          "max_curve_residual": float(np.max(resid))}
            print(f"{tag:18s} {name:14s} distortion {np.mean(dist):.4f} "
                  f"residual {np.mean(resid):.4f}  (max {np.max(dist):.4f} / "
                  f"{np.max(resid):.4f})", flush=True)

        zero = torch.zeros(1, 1)
        for rho in [float(v) for v in args.rho.split(",")]:
            r_side = rho / (device.W_um * mult)
            sims = simulate_truth(device.dev_type, device.L_um, device.W_um,
                                  curves, params, bin_index=bins[tag],
                                  rs_ext_ohm=r_side)
            t = aug.structured_fixed(x, ids, torch.tensor([rho]), zero,
                                     torch.full((1, 1), 0.75e-3),
                                     torch.full((1, 1), 1.5))
            compare(f"R rho={rho:g}", sims, t)
        # Heating only: per-curve overrides from the clean-curve power.
        power = []
        for curve, sim in zip(curves, clean):
            vd = np.abs(curve.Vd) if curve.kind == "idvd" else abs(float(curve.fixed))
            power.append(float(np.max(np.abs(sim) * vd)))
        power = np.asarray(power)
        d_t = args.dtmax * power / power.max()
        kv, beta = 0.75e-3, 1.5
        vth0, u0 = float(params["vth0"]), float(params["u0"])
        cps = [{"vth0": vth0 - np.sign(vth0) * kv * dt,
                "u0": u0 * (77.0 / (77.0 + dt)) ** beta} for dt in d_t]
        sims = simulate_truth(device.dev_type, device.L_um, device.W_um, curves,
                              params, bin_index=bins[tag], curve_params=cps)
        t = aug.structured_fixed(x, ids, torch.tensor([0.0]),
                                 torch.full((1, 1), args.dtmax),
                                 torch.full((1, 1), kv), torch.full((1, 1), beta))
        compare(f"heat dT={args.dtmax:g}", sims, t)
        out[tag] = rows
        del inv
    E.save_json(E.E2_DIR / "check_aug.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
