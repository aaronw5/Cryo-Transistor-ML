#!/usr/bin/env python3
"""Active Bayesian-optimization-style BSIM4 extraction (ML v3).

Differences from the one-shot pipeline (pdk_ml_extract.py):

* **Ensemble surrogate** — K emulators trained from different seeds; the
  search minimizes a pessimistic acquisition (mean + std across members) so
  it cannot exploit any single network's errors. Half the starts use the
  optimistic acquisition (mean - std) to explore regions of disagreement.
* **Active rounds** — after each search round, the top candidates are
  evaluated in real NGSpice and the TRUE (theta, curves) pairs are appended
  to the training set; the ensemble is fine-tuned before the next round.
  Surrogate errors near the optimum shrink every round.
* Warm starts (classical controls + previous ML winners) are validated in
  round 0, so the incumbent never starts worse than the best known card.

  python scripts/pdk_ml_active.py --device mps \
      --starts-from out/pdk_fd,out/pdk_cma,out/pdk_ml2_perdev \
      --out-dir out/pdk_ml3
"""
from __future__ import annotations

import argparse
from collections import defaultdict
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

logger = get_logger("pdk_ml_active")

SYNTH = PROCESSED_DIR / "pdk_synth"


def ensemble_loss(emus, zv, meas_t, curve_layout, n_curves):
    """Per-start loss for each ensemble member; returns (K, S) tensor."""
    losses = []
    for emu in emus:
        pred = mlx.inv_slog_t(emu(zv))
        curve_rrms = []
        for a, b, den in curve_layout:
            rmse = torch.sqrt(torch.mean(
                (pred[:, a:b] - meas_t[a:b].unsqueeze(0)) ** 2, dim=1))
            curve_rrms.append(rmse / den)
        losses.append(torch.stack(curve_rrms, dim=1).sum(dim=1) / n_curves)
    return torch.stack(losses, dim=0)


def active_extract(d, tdev, rounds=4, K=3, n_starts=1024, adam_steps=400,
                   n_validate=8, n_polish=3, max_nfev=120, seed=0,
                   emu_sizes=(512, 512, 512, 512), warm_starts=None,
                   emu_save_path=None):
    from scipy.optimize import least_squares

    tag = device_tag(d.dev_type, d.L_um, d.W_um)
    t0 = time.time()
    set_seed(seed)
    rng = np.random.default_rng(seed)

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
    curve_layout = []
    off = 0
    for (a, b) in slices:
        den = np.mean(np.abs(meas[a:b]))
        if den > 0 and np.isfinite(den):
            kept_mask[a:b] = True
            curve_layout.append((off, off + b - a, den))
            off += b - a

    X = Z[ok]
    Y = mlx.slog(IDS[ok][:, kept_mask])
    dev = torch.device(tdev)
    meas_t = torch.tensor(meas[kept_mask], dtype=torch.float32, device=dev)
    P = int(kept_mask.sum())

    # ---------------- NGSpice validation bank ----------------
    bank = []          # (rrms, label, z, params)
    bank_z = []

    def validate(z, label):
        z = np.asarray(z, dtype=np.float64)
        params = box.z_to_params(z)
        m, sims = eval_params(d.dev_type, d.L_um, d.W_um, bin_index, curves,
                              params)
        rr = float(m["rrms"]) if np.isfinite(m["rrms"]) else np.inf
        bank.append((rr, label, z, params))
        bank_z.append(z)
        row = np.concatenate([np.asarray(s, dtype=np.float64)[:b - a]
                              for s, (a, b) in zip(sims, slices)])
        if np.isfinite(row).all():
            return rr, row[kept_mask]
        return rr, None

    new_X, new_Y = [], []
    rr0, _ = validate(box.z_published, "published")
    for lbl, p in (warm_starts or []):
        z = box.params_to_z(p)
        if any(np.linalg.norm(z - b) < 0.15 for b in bank_z):
            continue
        rr, row = validate(z, lbl)
        if row is not None:
            new_X.append(z)
            new_Y.append(mlx.slog(row))

    # ---------------- ensemble ----------------
    emus, vals = [], []
    for k in range(K):
        torch.manual_seed(seed + 101 * k)
        emu = mlx.mlp([7, *emu_sizes, P]).to(dev)
        Xt = torch.tensor(X, dtype=torch.float32, device=dev)
        Yt = torch.tensor(Y, dtype=torch.float32, device=dev)
        v = mlx.train_net(emu, Xt, Yt, dev, epochs=1500, lr=1e-3,
                          batch=4096 if len(X) > 4000 else None)
        emus.append(emu)
        vals.append(float(v))

    # ---------------- active rounds ----------------
    for rnd in range(rounds):
        S = n_starts
        z0 = box.z_published
        seeds_z = [z0] + [bank_z[i] for i in range(len(bank_z))
                          if np.isfinite(bank[i][0])][:8]
        incumbent = min(bank, key=lambda b: b[0])[2]
        starts = np.concatenate([
            np.stack(seeds_z),
            incumbent + rng.normal(0, 0.3, size=(S // 4, 7)),
            z0 + rng.normal(0, 0.7, size=(S // 4, 7)),
            rng.uniform(-3.0, 3.0, size=(S - S // 2 - len(seeds_z), 7)),
        ], axis=0)
        explore = np.zeros(len(starts), dtype=bool)
        explore[len(starts) // 2:] = True
        explore_t = torch.tensor(explore, device=dev)

        zv = torch.tensor(starts, dtype=torch.float32, device=dev,
                          requires_grad=True)
        opt = torch.optim.Adam([zv], lr=0.05)
        best_loss = np.full(len(starts), np.inf)
        best_z = starts.copy()
        for _ in range(adam_steps):
            opt.zero_grad()
            L = ensemble_loss(emus, zv, meas_t, curve_layout, len(slices))
            mean, std = L.mean(dim=0), L.std(dim=0)
            acq = torch.where(explore_t, mean - std, mean + std)
            acq.sum().backward()
            opt.step()
            with torch.no_grad():
                zv.clamp_(-3.5, 3.5)
            # rank by pessimistic value regardless of acquisition used
            lv = (mean + std).detach().cpu().numpy()
            improved = lv < best_loss
            if improved.any():
                zc = zv.detach().cpu().numpy().astype(np.float64)
                best_loss[improved] = lv[improved]
                best_z[improved] = zc[improved]

        order = np.argsort(best_loss)
        picked = 0
        for i in order:
            z = best_z[i]
            if any(np.linalg.norm(z - b) < 0.2 for b in bank_z):
                continue
            rr, row = validate(z, f"active_r{rnd}")
            if row is not None:
                new_X.append(z)
                new_Y.append(mlx.slog(row))
            picked += 1
            if picked >= n_validate:
                break

        if rnd < rounds - 1 and new_X:
            Xa = np.concatenate([X, np.stack(new_X)])
            Ya = np.concatenate([Y, np.stack(new_Y)])
            Xt = torch.tensor(Xa, dtype=torch.float32, device=dev)
            Yt = torch.tensor(Ya, dtype=torch.float32, device=dev)
            for k, emu in enumerate(emus):
                mlx.train_net(emu, Xt, Yt, dev, epochs=400, lr=3e-4,
                              batch=4096 if len(Xa) > 4000 else None,
                              patience=120)

    # ---------------- FD polish of distinct bank leaders ----------------
    bank.sort(key=lambda b: b[0])
    f = lambda z: residual_fn(z, box, d.dev_type, d.L_um, d.W_um, bin_index,
                              curves, flat)
    polished, used = [], []
    for rr, lbl, z, params in bank:
        if not np.isfinite(rr):
            continue
        if any(np.linalg.norm(z - u) < 0.5 for u in used):
            continue
        used.append(z)
        try:
            sol = least_squares(f, z, method="trf", jac="2-point",
                                diff_step=2e-2, max_nfev=max_nfev)
            params2 = box.z_to_params(np.asarray(sol.x, dtype=np.float64))
            m2, _ = eval_params(d.dev_type, d.L_um, d.W_um, bin_index,
                                curves, params2)
            if np.isfinite(m2["rrms"]):
                polished.append((float(m2["rrms"]), lbl + "+fd",
                                 np.asarray(sol.x, dtype=np.float64),
                                 params2))
        except Exception:  # noqa: BLE001
            continue
        if len(used) >= n_polish:
            break

    results = {"published": {"rrms": rr0, "params": published}}
    best_warm = min((b for b in bank if not b[1].startswith("active")
                     and b[1] != "published"), default=None,
                    key=lambda b: b[0])
    if best_warm:
        results["warm_best"] = {"rrms": best_warm[0],
                                "params": best_warm[3]}
    best_active = min((b for b in bank if b[1].startswith("active")),
                      default=None, key=lambda b: b[0])
    if best_active:
        results["active_bo"] = {"rrms": best_active[0],
                                "params": best_active[3]}
    for rr, lbl, z, params in polished:
        key = "active_bo+fd" if lbl.startswith("active") else "warm+fd"
        if key not in results or rr < results[key]["rrms"]:
            results[key] = {"rrms": rr, "params": params}

    ml_keys = [k for k in results if k != "published"
               and np.isfinite(results[k]["rrms"])]
    best_key = min(ml_keys or ["published"],
                   key=lambda k: results[k]["rrms"])
    from cryoml.spice_pdk import simulate_pdk
    best_params = results[best_key]["params"]
    sims = simulate_pdk(d.dev_type, d.L_um, d.W_um, curves,
                        params=best_params, bin_index=bin_index)

    if emu_save_path is not None:
        torch.save({
            "state": emus[int(np.argmin(vals))].state_dict(),
            "emu_sizes": list(emu_sizes), "P": P,
            "curve_layout": curve_layout, "n_curves": len(slices),
            "meas_kept": meas[kept_mask], "emu_val": float(min(vals)),
        }, emu_save_path)

    rec = {
        "device": tag, "dev_type": d.dev_type, "L_um": d.L_um,
        "W_um": d.W_um, "bin_index": bin_index,
        "paper_reported": d.paper_rrms,
        "emulator_val_mse": float(min(vals)),
        "n_synth": int(ok.sum()), "n_ngspice_evals": len(bank),
        "methods": {k: {"rrms": v["rrms"]} for k, v in results.items()},
        "params_by_method": {k: v["params"] for k, v in results.items()},
        "best_method": best_key,
        "rrms": results[best_key]["rrms"],
        "start_rrms": rr0,
        "runtime_s": round(time.time() - t0, 1),
    }
    return rec, sims


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--ensemble", type=int, default=3)
    ap.add_argument("--n-starts", type=int, default=1024)
    ap.add_argument("--adam-steps", type=int, default=400)
    ap.add_argument("--n-validate", type=int, default=8)
    ap.add_argument("--n-polish", type=int, default=3)
    ap.add_argument("--max-nfev", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emu-arch", default="512,512,512,512")
    ap.add_argument("--starts-from",
                    default="out/pdk_fd,out/pdk_cma,out/pdk_ml2_perdev")
    ap.add_argument("--out-dir", default=str(OUT_DIR / "pdk_ml3"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emu_sizes = tuple(int(s) for s in args.emu_arch.split(","))

    warm_by_tag: dict[str, list] = defaultdict(list)
    for d_str in args.starts_from.split(","):
        directory = Path(d_str.strip())
        label = f"warm_{directory.name.removeprefix('pdk_')}"
        for path in directory.glob("*.json"):
            rec = json.loads(path.read_text())
            if not isinstance(rec, dict) or "device" not in rec:
                continue
            if "params" in rec:
                warm_by_tag[rec["device"]].append((label, rec["params"]))
            elif "params_by_method" in rec and "best_method" in rec:
                params = rec["params_by_method"].get(rec["best_method"])
                if params is None and rec["params_by_method"]:
                    finite = {k: v for k, v in rec["methods"].items()
                              if k in rec["params_by_method"]
                              and np.isfinite(v["rrms"])}
                    if finite:
                        best = min(finite, key=lambda k: finite[k]["rrms"])
                        params = rec["params_by_method"][best]
                if params is not None:
                    warm_by_tag[rec["device"]].append((label, params))

    for d in parse_device_list(args.devices):
        tag = device_tag(d.dev_type, d.L_um, d.W_um)
        result_path = out_dir / f"ml_{tag}.json"
        if args.resume and result_path.exists():
            old = json.load(open(result_path))
            if np.isfinite(old.get("rrms", np.nan)):
                logger.info("%-22s already complete; skipping", tag)
                continue
        rec, sims = active_extract(
            d, args.device, rounds=args.rounds, K=args.ensemble,
            n_starts=args.n_starts, adam_steps=args.adam_steps,
            n_validate=args.n_validate, n_polish=args.n_polish,
            max_nfev=args.max_nfev, seed=args.seed, emu_sizes=emu_sizes,
            warm_starts=warm_by_tag.get(tag),
            emu_save_path=out_dir / f"emu_{tag}.pt")
        json.dump(rec, open(result_path, "w"), indent=2)
        np.savez(out_dir / f"sims_{tag}.npz",
                 **{f"sim_{i}": np.asarray(s) for i, s in enumerate(sims)})
        logger.info("%-22s %.3f->%.3f [%s] %d evals %ss", tag,
                    rec["start_rrms"], rec["rrms"], rec["best_method"],
                    rec["n_ngspice_evals"], rec["runtime_s"])

    mlx.enforce_shared_bin_cards(out_dir, args.max_nfev, args.device)

    rows = []
    for path in sorted(out_dir.glob("ml_*.json")):
        rec = json.load(open(path))
        if not np.isfinite(rec.get("rrms", np.nan)):
            continue
        d = next(d for d in PAPER_DEVICES
                 if device_tag(d.dev_type, d.L_um, d.W_um) == rec["device"])
        curves = load_device_curves(d)
        saved = np.load(out_dir / f"sims_{rec['device']}.npz")
        sims = [np.asarray(saved[f"sim_{i}"]) for i in range(len(curves))]
        rec["rrms"] = float(device_rrms(sims, [c.Id for c in curves])["rrms"])
        json.dump(rec, open(path, "w"), indent=2)
        rows.append(rec)

    scores = np.array([r["rrms"] for r in rows])
    summary = {
        "n_devices": len(rows),
        "mean_rrms": float(np.nanmean(scores)),
    }
    method_keys = sorted({k for r in rows for k in r["methods"]
                          if k != "published"})
    for key in method_keys:
        vals = [r["methods"][key]["rrms"] for r in rows
                if key in r["methods"] and np.isfinite(r["methods"][key]["rrms"])]
        if vals:
            summary[f"mean_{key}"] = float(np.mean(vals))
            summary[f"n_{key}"] = len(vals)
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    with open(out_dir / "results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["device", "start_rrms", "rrms", "best_method",
                    "n_ngspice_evals"])
        for r in rows:
            w.writerow([r["device"], f"{r['start_rrms']:.4f}",
                        f"{r['rrms']:.4f}", r["best_method"],
                        r["n_ngspice_evals"]])
    print("\n=== ACTIVE-BO EXTRACTION ===")
    for k, v in summary.items():
        print(f"  {k:26s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
