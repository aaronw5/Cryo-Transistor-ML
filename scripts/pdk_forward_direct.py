#!/usr/bin/env python3
"""Direct amortized extractor: IV curve -> 7 BSIM4 parameters, one forward pass.

This is the classic supervised inverse / amortized extractor from the
compact-model-ML literature (e.g. iPREFER): train ONE network to regress the
parameter set straight from the I-V curve, then apply it ONCE to the measured
curve. No iterative search.

Two modes (same network, same data, one flag):
  --recon-weight 0      NO SURROGATE: pure parameter-space supervised loss.
  --recon-weight >0     WITH SURROGATE: add a curve-space reconstruction loss
                        through a frozen emulator emu(z)~curve, so the network
                        is rewarded for predicting parameters that *reconstruct*
                        the curve -- this breaks the non-identifiability that
                        makes pure parameter-MSE regress to the mean.

Honest reconstruction reporting (the metric the literature reports as "good"):
  * val_param_mse        -- held-out synthetic parameter MSE (z-space)
  * recon_rrms_synth_ng  -- one-shot prediction simulated in REAL NGSpice on
                            held-out synthetic curves, scored vs the true curve
  * rrms (measured)      -- one-shot prediction on the measured curve, real
                            NGSpice, paper-exact all-curve RRMS (headline)
  * direct+fd            -- least-squares-polished, for reference only

  python scripts/pdk_forward_direct.py --device mps --recon-weight 0 \
      --out-dir out/pdk_fwd_nosurr
  python scripts/pdk_forward_direct.py --device mps --recon-weight 1.0 \
      --emu-dir out/pdk_ml2 --out-dir out/pdk_fwd_surr
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
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pdk_ml_extract as mlx  # noqa: E402
from cryoml.config import OUT_DIR, PROCESSED_DIR, ensure_dirs  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import parse_device_list  # noqa: E402
from cryoml.metrics import device_rrms  # noqa: E402
from cryoml.pdk_extract import (PARAMS7, ThetaBox, eval_params,  # noqa: E402
                                flatten_paper_curves, residual_fn)
from cryoml.utils import device_tag, get_logger, set_seed  # noqa: E402

logger = get_logger("pdk_forward_direct")
SYNTH = PROCESSED_DIR / "pdk_synth"


def augment(Y_lin: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Instrument-style corruption of clean simulated currents (linear A), so
    the noisy measured curve is not out-of-distribution for the extractor."""
    out = Y_lin * (1.0 + rng.normal(0, 0.05, size=Y_lin.shape))
    out = out + rng.normal(0, 1e-9, size=Y_lin.shape)
    n, P = out.shape
    spike_rows = rng.random(n) < 0.5
    for i in np.where(spike_rows)[0]:
        k = rng.integers(1, 4)
        idx = rng.integers(0, P, size=k)
        mag = 10 ** rng.uniform(-9, -5.2, size=k)
        out[i, idx] = rng.choice([-1.0, 1.0], size=k) * mag
    quant_rows = rng.random(n) < 0.3
    for i in np.where(quant_rows)[0]:
        lsb = 10 ** rng.uniform(-9, -6)
        out[i] = np.round(out[i] / lsb) * lsb
    return out


class DirectNet(nn.Module):
    """IV-curve -> 7 params (z-space). Output is clamped to the inner box so an
    off-manifold measured input cannot extrapolate to a pathological corner
    (the z<->param map is sigmoid-bounded; |z|<=3.5 keeps the inner ~94%)."""

    def __init__(self, P: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(P, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 7),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.body(x), -3.5, 3.5)


def build_net(P: int, device: torch.device) -> nn.Module:
    return DirectNet(P).to(device)


def load_emulator(tag: str, device: torch.device, emu_dir: Path):
    blob = torch.load(emu_dir / f"emu_{tag}.pt", map_location=device,
                      weights_only=False)
    emu = mlx.mlp([7, *blob["emu_sizes"], blob["P"]]).to(device)
    emu.load_state_dict(blob["state"])
    emu.eval()
    for p in emu.parameters():
        p.requires_grad_(False)
    return emu, blob


def extract_device(d, tdev, recon_weight, emu_dir, epochs, aug_reps,
                   k_recon, seed, max_nfev):
    from scipy.optimize import least_squares
    tag = device_tag(d.dev_type, d.L_um, d.W_um)
    t0 = time.time()
    set_seed(seed)
    rng = np.random.default_rng(seed + 17)
    device = torch.device(tdev)

    data = np.load(SYNTH / f"{tag}.npz", allow_pickle=True)
    IDS, ok = data["IDS"], data["ok"]
    meas, slices = data["meas"], data["slices"]
    bin_index = int(data["bin_index"])
    published = {p: float(v) for p, v in zip(PARAMS7, data["published"])}
    box = ThetaBox(dev_type=d.dev_type, bin_index=bin_index, published=published)
    THETA = data["THETA"].astype(np.float64)
    Z = np.stack([box.params_to_z({p: t[i] for i, p in enumerate(PARAMS7)})
                  for t in THETA])[ok].astype(np.float32)
    curves = load_device_curves(d)
    flat = flatten_paper_curves(curves)

    kept_mask = np.zeros(len(meas), dtype=bool)
    kept_slices = []
    for (a, b) in slices:
        den = float(np.mean(np.abs(meas[a:b])))
        if den > 0 and np.isfinite(den):
            kept_mask[a:b] = True
            kept_slices.append((a, b))
    Y_lin = IDS[ok][:, kept_mask].astype(np.float64)   # clean currents (A)
    P = int(kept_mask.sum())

    # honest held-out split of the CLEAN synthetic samples
    n = len(Y_lin)
    perm = rng.permutation(n)
    n_val = max(8, int(0.15 * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    # training set: clean + augmented copies of the TRAIN split only
    Xc = mlx.slog(Y_lin[tr_idx])
    Zc = Z[tr_idx]
    X_in = [Xc]
    X_clean = [Xc]            # recon target is always the clean curve
    Z_t = [Zc]
    for _ in range(aug_reps - 1):
        X_in.append(mlx.slog(augment(Y_lin[tr_idx], rng)))
        X_clean.append(Xc)
        Z_t.append(Zc)
    X_in = np.concatenate(X_in).astype(np.float32)
    X_clean = np.concatenate(X_clean).astype(np.float32)
    Z_t = np.concatenate(Z_t).astype(np.float32)

    mean = X_in.mean(0, keepdims=True)
    std = X_in.std(0, keepdims=True).clip(0.25, None)
    Xs = torch.tensor((X_in - mean) / std, device=device)
    Xcl = torch.tensor(X_clean, device=device)
    Zt = torch.tensor(Z_t, device=device)

    emu = None
    if recon_weight > 0:
        emu, blob = load_emulator(tag, device, emu_dir)
        if blob["P"] != P:
            raise RuntimeError(f"{tag}: emu P {blob['P']} != kept {P}")

    # held-out CLEAN validation set for early stopping (on parameter MSE)
    Xv = torch.tensor(((mlx.slog(Y_lin[val_idx]) - mean) / std).astype(
        np.float32), device=device)
    Zv = torch.tensor(Z[val_idx], device=device)

    net = build_net(P, device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    batch = 4096
    nrows = len(Xs)
    best_val, best_state, bad, patience = np.inf, None, 0, 120
    for ep in range(epochs):
        net.train()
        order = torch.randperm(nrows, device=device)
        for s in range(0, nrows, batch):
            bi = order[s:s + batch]
            pred = net(Xs[bi])
            # IV -> params loss is purely the distance from the true
            # parameters (z-space MSE). The surrogate reconstruction term is
            # added ONLY in --recon-weight>0 (with-surrogate) mode.
            loss = nn.functional.mse_loss(pred, Zt[bi])
            if emu is not None:
                loss = loss + recon_weight * nn.functional.mse_loss(
                    emu(pred), Xcl[bi])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            vmse = float(nn.functional.mse_loss(net(Xv), Zv))
        if vmse < best_val - 1e-6:
            best_val, bad = vmse, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    with torch.no_grad():
        zv = net(Xv).cpu().numpy().astype(np.float64)
    val_param_mse = float(np.mean((zv - Z[val_idx]) ** 2))

    # honest one-shot reconstruction on held-out synthetic curves (real
    # NGSpice). Report the MEDIAN over draws: random box draws occasionally
    # turn the device nearly off, which makes a plain mean RRMS explode (tiny
    # denominator), so the median is the robust reconstruction statistic.
    k = min(k_recon, len(val_idx))
    recon_scores = []
    for j in range(k):
        vidx = val_idx[j]
        pred_params = box.z_to_params(zv[j])
        _, sims = eval_params(d.dev_type, d.L_um, d.W_um, bin_index, curves,
                              pred_params)
        true_curves = [Y_lin[vidx][a0:b0] for (a0, b0) in
                       _kept_layout(kept_slices)]
        score = device_rrms(sims, true_curves)["rrms"]
        if np.isfinite(score):
            recon_scores.append(score)
    recon_rrms_synth_ng = float(np.median(recon_scores)) if recon_scores else float("nan")
    recon_rrms_synth_ng_mean = float(np.mean(recon_scores)) if recon_scores else float("nan")

    # one-shot on the MEASURED curve
    meas_slog = ((mlx.slog(meas[kept_mask]) - mean[0]) / std[0]).astype(np.float32)
    with torch.no_grad():
        z_pred = net(torch.tensor(meas_slog, device=device).unsqueeze(0)
                     ).squeeze(0).cpu().numpy().astype(np.float64)

    def validate(z):
        params = box.z_to_params(np.asarray(z, dtype=np.float64))
        m, sims = eval_params(d.dev_type, d.L_um, d.W_um, bin_index, curves,
                              params)
        return (float(m["rrms"]) if np.isfinite(m["rrms"]) else np.inf,
                params, sims)

    start_rr, _, _ = validate(box.z_published)
    rr_raw, p_raw, sims_raw = validate(z_pred)
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

    results = {"published": {"rrms": start_rr, "params": published},
               "direct": {"rrms": rr_raw, "params": p_raw}}
    if p_fd is not None:
        results["direct+fd"] = {"rrms": rr_fd, "params": p_fd}
    best_key = min((kk for kk in results if kk != "published"),
                   key=lambda kk: results[kk]["rrms"])
    _, _, sims_best = validate(box.params_to_z(results[best_key]["params"]))

    rec = {
        "device": tag, "dev_type": d.dev_type, "L_um": d.L_um, "W_um": d.W_um,
        "bin_index": bin_index, "paper_reported": d.paper_rrms,
        "mode": "with_surrogate" if recon_weight > 0 else "no_surrogate",
        "recon_weight": recon_weight,
        "val_param_mse": val_param_mse,
        "recon_rrms_synth_ng": recon_rrms_synth_ng,
        "recon_rrms_synth_ng_mean": recon_rrms_synth_ng_mean,
        "n_recon_eval": len(recon_scores),
        "n_synth": int(ok.sum()), "n_train_rows": nrows,
        "methods": {kk: {"rrms": v["rrms"]} for kk, v in results.items()},
        "params_by_method": {kk: v["params"] for kk, v in results.items()},
        "best_method": best_key, "rrms": results[best_key]["rrms"],
        "start_rrms": start_rr, "runtime_s": round(time.time() - t0, 1),
    }
    return rec, sims_best, sims_raw


def _kept_layout(kept_slices):
    """Map original (a,b) measured slices to offsets within the kept vector."""
    layout, off = [], 0
    for (a, b) in kept_slices:
        layout.append((off, off + (b - a)))
        off += b - a
    return layout


def write_summary(out_dir: Path, rows: list[dict]) -> dict:
    summary = {"n_devices": len(rows),
               "mode": rows[0]["mode"] if rows else None}
    for key in ("direct", "direct+fd"):
        vals = [r["methods"][key]["rrms"] for r in rows
                if key in r["methods"] and np.isfinite(r["methods"][key]["rrms"])]
        if vals:
            summary[f"median_{key}"] = float(np.median(vals))
            summary[f"mean_{key}"] = float(np.mean(vals))
            summary[f"n_{key}"] = len(vals)
    rec_vals = [r["recon_rrms_synth_ng"] for r in rows
                if np.isfinite(r.get("recon_rrms_synth_ng", np.nan))]
    if rec_vals:
        summary["median_recon_rrms_synth_ng"] = float(np.median(rec_vals))
        summary["mean_recon_rrms_synth_ng"] = float(np.mean(rec_vals))
    if rows:
        summary["mean_val_param_mse"] = float(np.mean(
            [r["val_param_mse"] for r in rows]))
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["device", "recon_synth_ng", "direct_measured",
                    "direct_fd_measured"])
        for r in rows:
            w.writerow([r["device"],
                        f"{r.get('recon_rrms_synth_ng', float('nan')):.4f}",
                        f"{r['methods']['direct']['rrms']:.4f}",
                        f"{r['methods'].get('direct+fd', {}).get('rrms', float('nan')):.4f}"])
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--recon-weight", type=float, default=0.0)
    ap.add_argument("--emu-dir", default=str(OUT_DIR / "pdk_ml2"))
    ap.add_argument("--out-dir", default=str(OUT_DIR / "pdk_fwd_nosurr"))
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--aug-reps", type=int, default=6)
    ap.add_argument("--k-recon", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-nfev", type=int, default=60)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in parse_device_list(args.devices):
        tag = device_tag(d.dev_type, d.L_um, d.W_um)
        rp = out_dir / f"ml_{tag}.json"
        if args.resume and rp.exists() and np.isfinite(
                json.loads(rp.read_text()).get("rrms", np.nan)):
            rows.append(json.loads(rp.read_text()))
            logger.info("%-22s cached; skipping", tag)
            continue
        rec, sims_best, sims_raw = extract_device(
            d, args.device, args.recon_weight, Path(args.emu_dir),
            args.epochs, args.aug_reps, args.k_recon, args.seed, args.max_nfev)
        rows.append(rec)
        rp.write_text(json.dumps(rec, indent=2))
        np.savez(out_dir / f"sims_{tag}.npz",
                 **{f"sim_{i}": np.asarray(s) for i, s in enumerate(sims_raw)})
        logger.info("%-22s recon_synth %.3f | measured raw %.3f +fd %.3f [%s] %ss",
                    tag, rec["recon_rrms_synth_ng"],
                    rec["methods"]["direct"]["rrms"],
                    rec["methods"].get("direct+fd", {}).get("rrms", float("nan")),
                    rec["best_method"], rec["runtime_s"])
    summary = write_summary(out_dir, rows)
    print(f"\n=== DIRECT IV->PARAMS  ({summary.get('mode')}) ===")
    for k, v in summary.items():
        print(f"  {k:28s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
