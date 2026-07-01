#!/usr/bin/env python3
"""Direct parameter prediction: curves -> parameters, no search, no surrogate.

Per device, train ONE MLP that maps simulated I-V curves to the 7 BSIM4
parameters that produced them, then apply it ONCE to the real measured
curves. The prediction is validated in NGSpice (and a least-squares-polished
variant is recorded for reference).

To give the approach its best shot (the v1 inverse MLP failed at RRMS~47):
  * full 8k-sample training pools,
  * a bigger network (P -> 1024 -> 512 -> 256 -> 7),
  * artifact-aware augmentation: multiplicative + floor noise PLUS random
    instrument-style spikes and code-quantization, so the real measured
    curves are not out-of-distribution.

Outputs out/pdk_direct/<records, sims, summary.json> in the standard format.

  python scripts/pdk_ml_direct.py --device mps
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
from cryoml.metrics import device_rrms  # noqa: E402
from cryoml.pdk_extract import (PARAMS7, ThetaBox, eval_params,  # noqa: E402
                                flatten_paper_curves, residual_fn)
from cryoml.utils import device_tag, get_logger, set_seed  # noqa: E402

logger = get_logger("pdk_ml_direct")

SYNTH = PROCESSED_DIR / "pdk_synth"
OUT = OUT_DIR / "pdk_direct"


def augment(Y_lin: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Instrument-style corruption of clean simulated currents (linear A)."""
    out = Y_lin * (1.0 + rng.normal(0, 0.05, size=Y_lin.shape))
    out = out + rng.normal(0, 1e-9, size=Y_lin.shape)
    n, P = out.shape
    # random SMU-style spikes on ~half the rows (1-3 points each)
    spike_rows = rng.random(n) < 0.5
    for i in np.where(spike_rows)[0]:
        k = rng.integers(1, 4)
        idx = rng.integers(0, P, size=k)
        mag = 10 ** rng.uniform(-9, -5.2, size=k)  # 1 nA .. 6 uA codes
        out[i, idx] = rng.choice([-1.0, 1.0], size=k) * mag
    # range quantization on ~a third of the rows
    quant_rows = rng.random(n) < 0.3
    for i in np.where(quant_rows)[0]:
        lsb = 10 ** rng.uniform(-9, -6)
        out[i] = np.round(out[i] / lsb) * lsb
    return out


def direct_extract(d, tdev, aug_reps=6, epochs=800, seed=0, max_nfev=80):
    tag = device_tag(d.dev_type, d.L_um, d.W_um)
    t0 = time.time()
    set_seed(seed)
    rng = np.random.default_rng(seed + 17)

    data = np.load(SYNTH / f"{tag}.npz", allow_pickle=True)
    IDS, ok = data["IDS"], data["ok"]
    meas, slices = data["meas"], data["slices"]
    bin_index = int(data["bin_index"])
    published = {p: float(v) for p, v in zip(PARAMS7, data["published"])}
    box = ThetaBox(dev_type=d.dev_type, bin_index=bin_index,
                   published=published)
    THETA = data["THETA"].astype(np.float64)
    Z = np.stack([box.params_to_z({p: t[i] for i, p in enumerate(PARAMS7)})
                  for t in THETA])
    curves = load_device_curves(d)
    flat = flatten_paper_curves(curves)

    kept_mask = np.zeros(len(meas), dtype=bool)
    for (a, b) in slices:
        den = np.mean(np.abs(meas[a:b]))
        if den > 0 and np.isfinite(den):
            kept_mask[a:b] = True

    Zok = Z[ok]
    Y_lin = IDS[ok][:, kept_mask]
    P = int(kept_mask.sum())

    Xa = [mlx.slog(Y_lin)]
    Ya = [Zok]
    for _ in range(aug_reps - 1):
        Xa.append(mlx.slog(augment(Y_lin, rng)))
        Ya.append(Zok)
    X = np.concatenate(Xa).astype(np.float32)
    Y = np.concatenate(Ya).astype(np.float32)

    dev = torch.device(tdev)
    Xt = torch.tensor(X, device=dev)
    Yt = torch.tensor(Y, device=dev)
    meas_t = torch.tensor(mlx.slog(meas[kept_mask]), dtype=torch.float32,
                          device=dev).unsqueeze(0)

    torch.manual_seed(seed)
    net = mlx.mlp([P, 1024, 512, 256, 7]).to(dev)
    val = mlx.train_net(net, Xt, Yt, dev, epochs=epochs, lr=1e-3,
                        batch=8192, patience=150)
    with torch.no_grad():
        z_pred = net(meas_t).squeeze(0).cpu().numpy().astype(np.float64)

    def validate(z):
        params = box.z_to_params(np.asarray(z, dtype=np.float64))
        m, sims = eval_params(d.dev_type, d.L_um, d.W_um, bin_index, curves,
                              params)
        rr = float(m["rrms"]) if np.isfinite(m["rrms"]) else np.inf
        return rr, params, sims

    start_rr, _, _ = validate(box.z_published)
    rr_raw, p_raw, sims_raw = validate(z_pred)

    # reference: least-squares polish from the single-shot prediction
    from scipy.optimize import least_squares
    rr_fd, p_fd = np.inf, None
    try:
        sol = least_squares(
            lambda z: residual_fn(z, box, d.dev_type, d.L_um, d.W_um,
                                  bin_index, curves, flat),
            z_pred, method="trf", jac="2-point", diff_step=2e-2,
            max_nfev=max_nfev)
        rr_fd, p_fd, _ = validate(sol.x)
    except Exception:  # noqa: BLE001
        pass

    results = {
        "published": {"rrms": start_rr, "params": published},
        "direct": {"rrms": rr_raw, "params": p_raw},
    }
    if p_fd is not None:
        results["direct+fd"] = {"rrms": rr_fd, "params": p_fd}

    best_key = min((k for k in results if k != "published"),
                   key=lambda k: results[k]["rrms"])
    _, _, sims = validate(box.params_to_z(results[best_key]["params"]))

    rec = {
        "device": tag, "dev_type": d.dev_type, "L_um": d.L_um,
        "W_um": d.W_um, "bin_index": bin_index,
        "paper_reported": d.paper_rrms,
        "inverse_val_mse": float(val),
        "n_synth": int(ok.sum()),
        "methods": {k: {"rrms": v["rrms"]} for k, v in results.items()},
        "params_by_method": {k: v["params"] for k, v in results.items()},
        "best_method": best_key,
        "rrms": results[best_key]["rrms"],
        "start_rrms": start_rr,
        "runtime_s": round(time.time() - t0, 1),
    }
    return rec, sims, sims_raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--aug-reps", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in parse_device_list(args.devices):
        rec, sims_best, sims_raw = direct_extract(
            d, args.device, aug_reps=args.aug_reps,
            epochs=args.epochs, seed=args.seed)
        rows.append(rec)
        json.dump(rec, open(OUT / f"ml_{rec['device']}.json", "w"), indent=2)
        np.savez(OUT / f"sims_{rec['device']}.npz",
                 **{f"sim_{i}": np.asarray(s)
                    for i, s in enumerate(sims_raw)})
        logger.info("%-22s raw %.3f +fd %.3f [%s] %ss",
                    rec["device"], rec["methods"]["direct"]["rrms"],
                    rec["methods"].get("direct+fd", {}).get("rrms",
                                                            float("nan")),
                    rec["best_method"], rec["runtime_s"])

    summary = {"n_devices": len(rows)}
    for key in ("direct", "direct+fd"):
        vals = [r["methods"][key]["rrms"] for r in rows
                if key in r["methods"] and np.isfinite(r["methods"][key]["rrms"])]
        if vals:
            summary[f"mean_{key}"] = float(np.mean(vals))
            summary[f"n_{key}"] = len(vals)
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["device", "direct", "direct_fd"])
        for r in rows:
            w.writerow([r["device"],
                        f"{r['methods']['direct']['rrms']:.4f}",
                        f"{r['methods'].get('direct+fd', {}).get('rrms', float('nan')):.4f}"])
    print("\n=== DIRECT PARAMETER PREDICTION ===")
    for k, v in summary.items():
        print(f"  {k:26s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
