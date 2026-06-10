#!/usr/bin/env python3
"""ML-driven BSIM4 extraction against the PDK NGSpice chain.

Two learned components per device, both trained ONLY on synthetic NGSpice
data (data/processed/pdk_synth/<tag>.npz — published-bin neighborhoods,
never a measured point):

* **Emulator** E(z) -> signed_log Id at every kept measured-bias point.
  A small MLP that makes the forward model differentiable; used for a
  massive multistart Adam search in z-space (no NGSpice in the loop).
* **Inverse MLP** G(signed_log Id curves) -> z. The "obvious MLP that
  predicts parameters": trained with noise augmentation on synthetic
  curves, applied once to the real measured curves.

Candidates from both are validated in real NGSpice in the native geometry
bin and FD-polished on the paper-exact all-curve RRMS objective.

  python scripts/pdk_ml_extract.py --device mps
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
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.config import OUT_DIR, PROCESSED_DIR, ensure_dirs  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES, parse_device_list  # noqa: E402
from cryoml.utils import device_tag, get_logger, set_seed  # noqa: E402

logger = get_logger("pdk_ml_extract")

OUT_ML = OUT_DIR / "pdk_ml"
SYNTH = PROCESSED_DIR / "pdk_synth"
I_REF = 1e-9


def slog(x):
    if isinstance(x, np.ndarray):
        return np.sign(x) * np.log1p(np.abs(x) / I_REF)
    return torch.sign(x) * torch.log1p(torch.abs(x) / I_REF)


def inv_slog_t(y):
    return torch.sign(y) * I_REF * torch.expm1(torch.abs(y))


def mlp(sizes, act=nn.GELU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


def train_net(net, X, Y, device, epochs=1500, lr=1e-3, batch=None, wd=1e-5,
              val_frac=0.15, patience=200):
    """Full-batch (or minibatch) Adam training with early stopping."""
    n = len(X)
    idx = torch.randperm(n)
    n_val = max(1, int(n * val_frac))
    vi, ti = idx[:n_val], idx[n_val:]
    Xt, Yt, Xv, Yv = X[ti], Y[ti], X[vi], Y[vi]
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_v, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        if batch is None:
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(Xt), Yt)
            loss.backward()
            opt.step()
        else:
            perm = torch.randperm(len(Xt))
            for s in range(0, len(Xt), batch):
                b = perm[s:s + batch]
                opt.zero_grad()
                loss = nn.functional.mse_loss(net(Xt[b]), Yt[b])
                loss.backward()
                opt.step()
        sched.step()
        if ep % 10 == 0 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                v = float(nn.functional.mse_loss(net(Xv), Yv))
            if v < best_v - 1e-6:
                best_v, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 10
                if bad >= patience:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return best_v


def extract_device(d, tdev, n_adam_starts=512, adam_steps=400, n_validate=8,
                   n_polish=3, max_nfev=80, seed=0):
    from cryoml.pdk_extract import (PARAMS7, ThetaBox, eval_params,
                                    flatten_paper_curves, residual_fn)
    from scipy.optimize import least_squares

    tag = device_tag(d.dev_type, d.L_um, d.W_um)
    t0 = time.time()
    set_seed(seed)
    data = np.load(SYNTH / f"{tag}.npz", allow_pickle=True)
    IDS, ok = data["IDS"], data["ok"]
    meas, slices = data["meas"], data["slices"]
    bin_index = int(data["bin_index"])
    published = {p: float(v) for p, v in zip(PARAMS7, data["published"])}
    box = ThetaBox(dev_type=d.dev_type, bin_index=bin_index, published=published)
    # recompute z from physical theta with the CURRENT box so the dataset
    # stays valid even when bound definitions evolve
    THETA = data["THETA"].astype(np.float64)
    Z = np.stack([box.params_to_z({p: t[i] for i, p in enumerate(PARAMS7)})
                  for t in THETA])
    curves = load_device_curves(d)
    flat = flatten_paper_curves(curves)

    # Every nonzero-denominator curve is included. All-zero curves contribute
    # zero to the paper scorer and therefore have no optimization residual.
    kept_mask = np.zeros(len(meas), dtype=bool)
    curve_layout = []
    kept_offset = 0
    for (a, b) in slices:
        m = meas[a:b]
        den = np.mean(np.abs(m))
        if den > 0 and np.isfinite(den):
            kept_mask[a:b] = True
            curve_layout.append((kept_offset, kept_offset + b - a, den))
            kept_offset += b - a

    Zok, Yok = Z[ok], IDS[ok][:, kept_mask]
    Y_slog = slog(Yok)

    dev = torch.device(tdev)
    Zt = torch.tensor(Zok, dtype=torch.float32, device=dev)
    Yt = torch.tensor(Y_slog, dtype=torch.float32, device=dev)
    P = Yt.shape[1]

    # ---------------- emulator E(z) -> slog Id ----------------
    emu = mlp([7, 256, 256, 256, P]).to(dev)
    emu_val = train_net(emu, Zt, Yt, dev, epochs=2000, lr=1e-3)

    # ---------------- inverse MLP G(slog Id) -> z ----------------
    # noise augmentation: additive 1 nA floor noise + 5% multiplicative
    rng = np.random.default_rng(seed + 7)
    aug_reps = 4
    Xa, Ya = [], []
    for _ in range(aug_reps):
        noisy = Yok * (1 + rng.normal(0, 0.05, size=Yok.shape)) \
            + rng.normal(0, 1e-9, size=Yok.shape)
        Xa.append(slog(noisy))
        Ya.append(Zok)
    Xinv = torch.tensor(np.concatenate(Xa), dtype=torch.float32, device=dev)
    Yinv = torch.tensor(np.concatenate(Ya), dtype=torch.float32, device=dev)
    inv = mlp([P, 512, 256, 7]).to(dev)
    inv_val = train_net(inv, Xinv, Yinv, dev, epochs=1500, lr=1e-3, batch=4096)

    meas_slog_t = torch.tensor(slog(meas[kept_mask]), dtype=torch.float32,
                               device=dev)
    with torch.no_grad():
        z_inv = inv(meas_slog_t.unsqueeze(0)).squeeze(0).cpu().numpy().astype(np.float64)

    # ---------------- Adam multistart search on the emulator ----------------
    meas_t = torch.tensor(meas[kept_mask], dtype=torch.float32, device=dev)
    S = n_adam_starts
    z0 = box.z_published
    starts = np.concatenate([
        z0[None, :], z_inv[None, :],
        z0 + rng.normal(0, 0.7, size=(S // 2, 7)),
        rng.uniform(-3.0, 3.0, size=(S - S // 2 - 2, 7)),
    ], axis=0)
    zv = torch.tensor(starts, dtype=torch.float32, device=dev, requires_grad=True)
    opt = torch.optim.Adam([zv], lr=0.05)
    best_loss = np.full(len(starts), np.inf)
    best_z = starts.copy()
    for _ in range(adam_steps):
        opt.zero_grad()
        pred = inv_slog_t(emu(zv))
        curve_rrms = []
        for a, b, denominator in curve_layout:
            rmse = torch.sqrt(torch.mean(
                (pred[:, a:b] - meas_t[a:b].unsqueeze(0)) ** 2, dim=1
            ))
            curve_rrms.append(rmse / denominator)
        loss_per = torch.stack(curve_rrms, dim=1).sum(dim=1) / len(slices)
        loss_per.sum().backward()
        opt.step()
        with torch.no_grad():
            zv.clamp_(-3.5, 3.5)  # stay inside the emulator's training range
        lv = loss_per.detach().cpu().numpy()
        improved = lv < best_loss
        if improved.any():
            zc = zv.detach().cpu().numpy().astype(np.float64)
            best_loss[improved] = lv[improved]
            best_z[improved] = zc[improved]
    order = np.argsort(best_loss)

    # ---------------- NGSpice validation of candidates ----------------
    results = {}
    start_m, _ = eval_params(d.dev_type, d.L_um, d.W_um, bin_index, curves,
                             box.published)
    results["published"] = {"params": box.published,
                            "rrms": float(start_m["rrms"])}

    def validate(z):
        params = box.z_to_params(np.asarray(z, dtype=np.float64))
        m, _ = eval_params(d.dev_type, d.L_um, d.W_um, bin_index, curves, params)
        return params, m

    cand = []
    # top emulator candidates (deduplicated in z by distance)
    seen = []
    for i in order:
        z = best_z[i]
        if any(np.linalg.norm(z - s) < 0.5 for s in seen):
            continue
        seen.append(z)
        cand.append(("emu_search", z))
        if len(seen) >= n_validate:
            break
    cand.append(("inverse_mlp", z_inv))

    validated = []
    for lbl, z in cand:
        params, m = validate(z)
        rr = m["rrms"] if np.isfinite(m["rrms"]) else np.inf
        validated.append((rr, lbl, z, params, m))
    validated.sort(key=lambda t: t[0])

    best_emu = next((v for v in validated if v[1] == "emu_search"), None)
    best_inv = next((v for v in validated if v[1] == "inverse_mlp"), None)
    if best_emu:
        results["emu_search"] = {"params": best_emu[3],
                                 "rrms": float(best_emu[4]["rrms"])}
    if best_inv:
        results["inverse_mlp"] = {"params": best_inv[3],
                                  "rrms": float(best_inv[4]["rrms"])}

    # ---------------- FD polish of the best few candidates ----------------
    f = lambda z: residual_fn(z, box, d.dev_type, d.L_um, d.W_um, bin_index,
                              curves, flat)
    # if every ML candidate failed in NGSpice, polish from the published
    # start instead so the method always returns something sane
    if not any(np.isfinite(v[0]) for v in validated):
        validated.insert(0, (start_m["rrms"], "emu_search", box.z_published,
                             box.published, start_m))
    polished = []
    for rr, lbl, z, params, m in validated[:n_polish]:
        try:
            sol = least_squares(f, z, method="trf", jac="2-point",
                                diff_step=2e-2, max_nfev=max_nfev)
            p2, m2 = validate(sol.x)
            polished.append((m2["rrms"], lbl + "+fd", p2, m2))
        except Exception:
            continue
    # also polish the inverse-MLP start explicitly (if not already polished)
    if best_inv and all(not lbl.startswith("inverse_mlp") for _, lbl, *_ in
                        [(0, p[1]) for p in polished]):
        try:
            sol = least_squares(f, best_inv[2], method="trf", jac="2-point",
                                diff_step=2e-2, max_nfev=max_nfev)
            p2, m2 = validate(sol.x)
            polished.append((m2["rrms"], "inverse_mlp+fd", p2, m2))
        except Exception:
            pass
    for rr, lbl, p2, m2 in polished:
        cur = results.get(lbl)
        if cur is None or (np.isfinite(rr) and rr < cur["rrms"]):
            results[lbl] = {"params": p2, "rrms": float(m2["rrms"])}

    # Final pick is the best paper-exact NGSpice score.
    ml_keys = [k for k in results if k != "published"
               and np.isfinite(results[k]["rrms"])]
    if not ml_keys:
        ml_keys = ["published"]
    best_key = min(ml_keys, key=lambda k: results[k]["rrms"])
    from cryoml.spice_pdk import simulate_pdk
    best_params = results[best_key]["params"]
    sims = simulate_pdk(d.dev_type, d.L_um, d.W_um, curves,
                        params=best_params, bin_index=bin_index)

    rec = {
        "device": tag, "dev_type": d.dev_type, "L_um": d.L_um, "W_um": d.W_um,
        "bin_index": bin_index, "paper_reported": d.paper_rrms,
        "emulator_val_mse": float(emu_val), "inverse_val_mse": float(inv_val),
        "n_synth": int(ok.sum()),
        "methods": {k: {kk: vv for kk, vv in v.items() if kk != "params"}
                    for k, v in results.items()},
        "params_by_method": {k: v["params"] for k, v in results.items()},
        "best_method": best_key,
        "rrms": results[best_key]["rrms"],
        "start_rrms": results["published"]["rrms"],
        "runtime_s": round(time.time() - t0, 1),
    }
    return rec, sims


def enforce_shared_bin_cards(out_dir: Path, max_nfev: int) -> None:
    """Jointly fit devices that NGspice maps to the same deployable model bin."""
    from scipy.optimize import least_squares

    from cryoml.pdk_extract import (eval_params, flatten_paper_curves,
                                    residual_fn, theta_box_for)

    baseline_rows = json.load(
        open(OUT_DIR / "pdk_baseline" / "pdk_baseline.json")
    )["devices"]
    baseline = {row["device"]: row for row in baseline_rows}
    groups = defaultdict(list)
    for device in PAPER_DEVICES:
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        groups[(device.dev_type, int(baseline[tag]["bin_index"]))].append(device)

    for (dev_type, bin_index), devices in groups.items():
        if len(devices) < 2:
            continue
        tags = [device_tag(d.dev_type, d.L_um, d.W_um) for d in devices]
        paths = [out_dir / f"ml_{tag}.json" for tag in tags]
        if not all(path.exists() for path in paths):
            continue
        records = [json.load(open(path)) for path in paths]
        curves_by_device = [load_device_curves(d) for d in devices]
        flats = [flatten_paper_curves(curves) for curves in curves_by_device]
        box = theta_box_for(
            devices[0].dev_type, devices[0].L_um, devices[0].W_um, bin_index
        )

        def joint_residual(z):
            return np.concatenate([
                residual_fn(z, box, d.dev_type, d.L_um, d.W_um, bin_index,
                            curves, flat)
                for d, curves, flat in zip(devices, curves_by_device, flats)
            ])

        def evaluate(params):
            scores, simulations = [], []
            for d, curves in zip(devices, curves_by_device):
                metrics, sims = eval_params(
                    d.dev_type, d.L_um, d.W_um, bin_index, curves, params
                )
                scores.append(float(metrics["rrms"]))
                simulations.append(sims)
            return float(np.mean(scores)), scores, simulations

        starts = [box.z_published]
        independent_best = []
        for record in records:
            for method, params in record["params_by_method"].items():
                if method != "shared_bin_ml+fd":
                    starts.append(box.params_to_z(params))
            methods = {
                method: metrics["rrms"]
                for method, metrics in record["methods"].items()
                if method != "shared_bin_ml+fd" and np.isfinite(metrics["rrms"])
            }
            method = min(methods, key=methods.get)
            independent_best.append(
                box.params_to_z(record["params_by_method"][method])
            )
        if len(independent_best) == 2:
            for alpha in np.linspace(0.1, 0.9, 9):
                starts.append(
                    (1.0 - alpha) * independent_best[0] + alpha * independent_best[1]
                )

        unique_starts = []
        for start in starts:
            if not any(np.linalg.norm(start - seen) < 1e-3 for seen in unique_starts):
                unique_starts.append(np.asarray(start, dtype=np.float64))

        best = (*evaluate(box.published), box.published)
        scored_starts = []
        for start in unique_starts:
            params = box.z_to_params(start)
            joint_score, scores, simulations = evaluate(params)
            scored_starts.append((joint_score, start))
            if np.isfinite(joint_score) and joint_score < best[0]:
                best = joint_score, scores, simulations, params
        scored_starts.sort(key=lambda item: item[0])

        for _, start in scored_starts[:8]:
            try:
                solution = least_squares(
                    joint_residual, start, method="trf", jac="2-point",
                    diff_step=2e-2, max_nfev=max_nfev,
                )
            except Exception:
                continue
            params = box.z_to_params(np.asarray(solution.x, dtype=np.float64))
            joint_score, scores, simulations = evaluate(params)
            if np.isfinite(joint_score) and joint_score < best[0]:
                best = joint_score, scores, simulations, params

        _, scores, simulations, params = best
        for path, tag, record, score, sims in zip(
            paths, tags, records, scores, simulations
        ):
            record["methods"]["shared_bin_ml+fd"] = {"rrms": score}
            record["params_by_method"]["shared_bin_ml+fd"] = params
            record["best_method"] = "shared_bin_ml+fd"
            record["rrms"] = score
            record["shared_bin_devices"] = tags
            json.dump(record, open(path, "w"), indent=2)
            np.savez(out_dir / f"sims_{tag}.npz",
                     **{f"sim_{i}": np.asarray(sim) for i, sim in enumerate(sims)})
        logger.info("%s bin %-2d joint mean %.3f for %s",
                    dev_type, bin_index, best[0], ", ".join(tags))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default=None)
    ap.add_argument("--device", default="mps", help="torch device")
    ap.add_argument("--n-adam-starts", type=int, default=512)
    ap.add_argument("--adam-steps", type=int, default=400)
    ap.add_argument("--n-validate", type=int, default=8)
    ap.add_argument("--n-polish", type=int, default=3)
    ap.add_argument("--max-nfev", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="Skip devices with an existing finite result")
    args = ap.parse_args()

    ensure_dirs()
    out_dir = Path(args.out_dir) if args.out_dir else OUT_ML
    out_dir.mkdir(parents=True, exist_ok=True)
    devices = parse_device_list(args.devices)

    rows = []
    for d in devices:
        tag = device_tag(d.dev_type, d.L_um, d.W_um)
        result_path = out_dir / f"ml_{tag}.json"
        if args.resume and result_path.exists():
            old = json.load(open(result_path))
            if np.isfinite(old.get("rrms", np.nan)):
                logger.info("%-22s already complete; skipping", tag)
                continue
        rec, sims = extract_device(
            d, args.device, n_adam_starts=args.n_adam_starts,
            adam_steps=args.adam_steps, n_validate=args.n_validate,
            n_polish=args.n_polish, max_nfev=args.max_nfev, seed=args.seed)
        rows.append(rec)
        json.dump(rec, open(out_dir / f"ml_{rec['device']}.json", "w"), indent=2)
        np.savez(out_dir / f"sims_{rec['device']}.npz",
                 **{f"sim_{i}": np.asarray(s) for i, s in enumerate(sims)})
        logger.info("%-22s %.3f->%.3f [%s] %ss",
                    rec["device"], rec["start_rrms"], rec["rrms"],
                    rec["best_method"], rec["runtime_s"])

    enforce_shared_bin_cards(out_dir, args.max_nfev)

    # Always aggregate every completed result in the output directory. This
    # keeps subset/resume runs from replacing the full experiment summary.
    rows = []
    for path in sorted(out_dir.glob("ml_*.json")):
        rec = json.load(open(path))
        if np.isfinite(rec.get("rrms", np.nan)):
            d = next(d for d in PAPER_DEVICES
                     if device_tag(d.dev_type, d.L_um, d.W_um) == rec["device"])
            curves = load_device_curves(d)
            saved = np.load(out_dir / f"sims_{rec['device']}.npz")
            sims = [np.asarray(saved[f"sim_{i}"]) for i in range(len(curves))]
            from cryoml.metrics import device_rrms
            meas = [c.Id for c in curves]
            rec["rrms"] = float(device_rrms(sims, meas)["rrms"])
            json.dump(rec, open(path, "w"), indent=2)
            rows.append(rec)
    if not rows:
        raise RuntimeError(f"no finite ML results found in {out_dir}")

    scores = np.array([r["rrms"] for r in rows])
    baseline = {r["device"]: r for r in json.load(
        open(OUT_DIR / "pdk_baseline" / "pdk_baseline.json"))["devices"]}
    base = np.array([baseline[r["device"]]["rrms"] for r in rows])
    summary = {
        "n_devices": len(rows),
        "mean_rrms": float(np.nanmean(scores)),
        "baseline_mean_rrms": float(np.nanmean(base)),
        "wins_vs_baseline": int(np.sum(scores < base)),
    }
    # per-method means
    for key in ("emu_search", "inverse_mlp", "emu_search+fd", "inverse_mlp+fd",
                "shared_bin_ml+fd"):
        vals = [r["methods"][key]["rrms"] for r in rows if key in r["methods"]]
        if vals:
            summary[f"mean_{key}"] = float(np.nanmean(vals))
            summary[f"n_{key}"] = len(vals)
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["device", "bin", "paper_reported", "start_rrms", "rrms",
                    "best_method"])
        for r in rows:
            w.writerow([r["device"], r["bin_index"], r["paper_reported"],
                        f"{r['start_rrms']:.4f}", f"{r['rrms']:.4f}",
                        r["best_method"]])
    print("\n=== PDK ML EXTRACTION ===")
    for k, v in summary.items():
        print(f"  {k:26s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
