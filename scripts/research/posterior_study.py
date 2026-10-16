#!/usr/bin/env python3
"""Train and evaluate amortized I-V -> parameter models (methods A/B).

  # B: conditional flow matching posterior on pooled cold-loop banks
  PYTHONPATH=src .venv/bin/python scripts/research/posterior_study.py train \
      --set seven --model flow --banks data/processed/cold_loop/seven_r1 \
      --run seven_flow_r1
  PYTHONPATH=src .venv/bin/python scripts/research/posterior_study.py evaluate \
      --run seven_flow_r1 --samples 256 --polish

Evaluation is fixed and identical for every device: draw K candidates from
the model given the measured curves, simulate all of them in NGSpice, keep the
lowest fixed-inclusion RRMS, optionally polish it with NGSpice least squares.
Best-of-K scores are reported for several K with their NGSpice call counts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from cryoml.card_eval import aggregate, score_card  # noqa: E402
from cryoml.cold_start import cold_centers, spec_from_centers  # noqa: E402
from cryoml.config import OUT_DIR  # noqa: E402
from cryoml.data_io import target_override  # noqa: E402
from cryoml.posterior import (FeatureMap, target_measured, FlowPosterior, WTAPosterior,  # noqa: E402
                              augment, load_pooled_banks, save_json)
from cryoml.utils import set_seed  # noqa: E402

RUNS = OUT_DIR / "research" / "posterior"


def names_for(parameter_set: str):
    from cold_loop import names_for as cold_names
    return cold_names(parameter_set)


def build_model(kind: str, n_in: int, n_parameters: int, config: dict):
    if kind == "flow":
        return FlowPosterior(n_in, n_parameters, hidden=config["hidden"],
                             context=config["context"],
                             velocity_hidden=config["velocity_hidden"],
                             velocity_layers=config["velocity_layers"])
    return WTAPosterior(n_in, n_parameters, heads=config["heads"],
                        hidden=config["hidden"], context=config["context"],
                        relax=config["relax"])


def train(args) -> int:
    set_seed(args.seed)
    run_dir = RUNS / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    names = names_for(args.set)
    centers = {tag: row["center"] for tag, row in cold_centers(names).items()}
    t0 = time.time()
    bank = load_pooled_banks([Path(b) for b in args.banks], names, centers)
    load_s = time.time() - t0
    rng = np.random.default_rng(args.seed)
    n = len(bank.z)
    if args.holdout_devices:
        held = {bank.tags.index(t) for t in args.holdout_devices.split(",")}
        is_val = np.isin(bank.device_ids, list(held))
    else:
        is_val = rng.random(n) < 0.05
    train_idx, val_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)

    sub = rng.choice(train_idx, size=min(20000, len(train_idx)), replace=False)
    from cryoml.foundation import signed_log
    slog = signed_log(bank.currents[sub].astype(np.float64))
    slog_mean = slog.mean(0).astype(np.float32)
    # A floor of 0.1 signed-log units (about 10% current) keeps points the
    # bank barely varies from turning measurement noise into huge inputs.
    slog_std = np.maximum(slog.std(0), 0.1).astype(np.float32)
    masked = np.where(bank.mask[train_idx], bank.z[train_idx], np.nan)
    z_scale = np.nan_to_num(np.nanstd(masked, axis=0), nan=1.0)
    z_scale = np.maximum(z_scale, 1e-3).astype(np.float32)
    z_shift = np.nan_to_num(np.nanmean(masked, axis=0), nan=0.0).astype(np.float32)

    device = torch.device(args.device)
    features = FeatureMap(slog_mean, slog_std, bank.curve_scales,
                          bank.geometry).to(device)
    config = {"hidden": args.hidden, "context": args.context,
              "velocity_hidden": args.velocity_hidden,
              "velocity_layers": args.velocity_layers, "heads": args.heads,
              "relax": args.relax}
    model = build_model(args.model, features.n_features, len(names),
                        config).to(device)
    currents = torch.as_tensor(bank.currents, device=device)
    z_all = torch.as_tensor((bank.z - z_shift) / z_scale, device=device)
    mask_all = torch.as_tensor(bank.mask.astype(np.float32), device=device)
    z_all = z_all * mask_all
    ids_all = torch.as_tensor(bank.device_ids, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_idx) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
        pct_start=0.05)
    train_t = torch.as_tensor(train_idx, device=device)
    val_t = torch.as_tensor(val_idx, device=device)

    # Discrepancy library: signed-log gap between each device's measurement
    # and its closest bank curve.  A sample from device i is only ever
    # perturbed with the gap of a different device j != i.
    from cryoml.foundation import current_from_signed_log
    deltas = []
    for i in range(len(bank.tags)):
        rows = np.flatnonzero(bank.device_ids == i)
        target = signed_log(bank.measured[i].astype(np.float64))
        best_gap, best_row = np.inf, rows[0]
        for start in range(0, len(rows), 4096):
            block = signed_log(bank.currents[rows[start:start + 4096]].astype(np.float64))
            gaps = np.mean(np.abs(block - target[None]), axis=1)
            k = int(np.argmin(gaps))
            if gaps[k] < best_gap:
                best_gap, best_row = gaps[k], rows[start + k]
        deltas.append(target - signed_log(bank.currents[best_row].astype(np.float64)))
    delta_t = torch.as_tensor(np.clip(np.stack(deltas), -3, 3).astype(np.float32),
                              device=device)
    n_dev = len(bank.tags)
    z_norm_np = (bank.z - z_shift) / z_scale
    z_lo = np.stack([np.where(bank.mask[bank.device_ids == i],
                              z_norm_np[bank.device_ids == i], np.inf).min(0)
                     for i in range(n_dev)])
    z_hi = np.stack([np.where(bank.mask[bank.device_ids == i],
                              z_norm_np[bank.device_ids == i], -np.inf).max(0)
                     for i in range(n_dev)])

    def discrepancy(current, ids):
        if args.discrepancy_prob <= 0:
            return current
        use = torch.rand(len(ids), device=device) < args.discrepancy_prob
        other = (ids + torch.randint(1, n_dev, ids.shape, device=device)) % n_dev
        alpha = torch.rand(len(ids), 1, device=device) * use[:, None]
        return current_from_signed_log(
            signed_log(current) + alpha * delta_t[other])

    def batch_loss(index, augmented: bool):
        current = currents[index]
        if augmented:
            current = discrepancy(current, ids_all[index])
            current = augment(current, gain_sigma=args.gain_sigma,
                              point_sigma=args.point_sigma,
                              floor_amps=args.floor_amps)
        return model.loss(features(current, ids_all[index]), z_all[index],
                          mask_all[index])

    history, best, best_state = [], float("inf"), None
    t_train = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = train_t[torch.randperm(len(train_t), device=device)]
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            loss = batch_loss(order[start:start + args.batch_size], True)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += float(loss.detach())
        model.eval()
        with torch.no_grad():
            torch.manual_seed(epoch)
            val = float(np.mean([
                float(batch_loss(val_t[s:s + 4096], True))
                for s in range(0, len(val_t), 4096)]))
        history.append({"epoch": epoch, "train": total / steps_per_epoch,
                        "val": val})
        if val < best:
            best = val
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        print(f"epoch {epoch} train {total / steps_per_epoch:.5f} val {val:.5f}",
              flush=True)
    torch.save({
        "kind": args.model, "config": config, "state": best_state,
        "names": list(names), "slog_mean": slog_mean, "slog_std": slog_std,
        "z_scale": z_scale, "z_shift": z_shift,
        "curve_scales": bank.curve_scales, "geometry": bank.geometry,
        "measured": bank.measured, "tags": bank.tags,
        "z_lo": z_lo.astype(np.float32), "z_hi": z_hi.astype(np.float32),
        "trained_mask": np.stack([
            bank.mask[bank.device_ids == i].any(0) for i in range(len(bank.tags))]),
        "set": args.set,
    }, run_dir / "model.pt")
    save_json(run_dir / "train.json", {
        "args": vars(args), "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)), "best_val": best,
        "load_s": round(load_s, 1),
        "train_s": round(time.time() - t_train, 1), "history": history,
        "banks": bank.sources})
    return 0


def _score_job(job):
    tag, params = job
    row = score_card(tag, params)
    return row["rrms"], row["official_dynamic_rrms"]


def _polish_job(job):
    from cold_loop import box_for
    from cryoml.ngspice_fit import DeviceProblem, final_score, polish
    tag, spec, max_nfev = job
    problem = DeviceProblem(tag, box_for(spec, tag))
    result = polish(problem, problem.box.local_published, max_nfev=max_nfev,
                    irls_rounds=2)
    scored = final_score(problem, result["local"])
    return tag, {"rrms": scored["rrms"],
                 "official_dynamic_rrms": scored["official_dynamic_rrms"],
                 "params": scored["params"], "n_sim": result["n_sim"],
                 "history": result["history"], "runtime_s": result["runtime_s"]}


def candidates_to_params(z_norm: np.ndarray, checkpoint: dict, tag_index: int,
                         centers: dict) -> list[dict]:
    names = checkpoint["names"]
    tag = checkpoint["tags"][tag_index]
    center = np.asarray([centers[tag][n] for n in names], dtype=np.float64)
    mask = checkpoint["trained_mask"][tag_index]
    z = z_norm * checkpoint["z_scale"] + checkpoint["z_shift"]
    z = np.where(mask[None, :], z, 0.0)
    values = center[None, :] * np.exp(np.clip(z, -12, 12))
    return [dict(zip(names, map(float, row))) for row in values]


def evaluate(args) -> int:
    run_dir = RUNS / args.run
    checkpoint = torch.load(run_dir / "model.pt", weights_only=False)
    names = tuple(checkpoint["names"])
    centers = {tag: row["center"] for tag, row in cold_centers(names).items()}
    device = torch.device(args.device)
    features = FeatureMap(checkpoint["slog_mean"], checkpoint["slog_std"],
                          checkpoint["curve_scales"],
                          checkpoint["geometry"]).to(device)
    model = build_model(checkpoint["kind"], features.n_features, len(names),
                        checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    tags = checkpoint["tags"]
    torch.manual_seed(args.seed)
    measured = torch.as_tensor(checkpoint["measured"], device=device)
    if target_override() is not None:
        # Held-out simulated devices: the targets (and their per-curve
        # scales) come from the override, the trained model is unchanged.
        bank_dir = json.loads((run_dir / "train.json").read_text())["banks"][0]
        target, scales = target_measured(Path(bank_dir), tags)
        measured = torch.as_tensor(target, device=device)
        features.curve_scales = torch.as_tensor(scales, device=device)
    ids = torch.arange(len(tags), device=device)
    t0 = time.time()
    with torch.no_grad():
        feats = features(measured, ids)
        samples = model.sample(feats, args.samples, steps=args.steps)
    samples = samples.cpu().numpy()
    if args.clip_prior and "z_lo" in checkpoint:
        lo = np.where(np.isfinite(checkpoint["z_lo"]), checkpoint["z_lo"], 0.0)
        hi = np.where(np.isfinite(checkpoint["z_hi"]), checkpoint["z_hi"], 0.0)
        samples = np.clip(samples, lo[:, None, :], hi[:, None, :])
    point = samples.mean(axis=1)
    sample_s = time.time() - t0

    jobs, owners = [], []
    for i, tag in enumerate(tags):
        cards = candidates_to_params(samples[i], checkpoint, i, centers)
        cards.append(candidates_to_params(point[i][None], checkpoint, i,
                                          centers)[0])
        for j, card in enumerate(cards):
            jobs.append((tag, card))
            owners.append((i, j))
    t1 = time.time()
    with get_context("spawn").Pool(args.workers) as pool:
        scores = pool.map(_score_job, jobs, chunksize=8)
    sim_s = time.time() - t1
    n_cand = samples.shape[1]
    table = np.full((len(tags), n_cand + 1), np.inf)
    for (i, j), (rrms, _dyn) in zip(owners, scores):
        table[i, j] = rrms if np.isfinite(rrms) else np.inf
    ks = [k for k in (1, 4, 16, 64, 256, 1024) if k <= n_cand]
    report = {"run": args.run, "kind": checkpoint["kind"],
              "n_candidates": n_cand, "sample_s": round(sample_s, 2),
              "simulate_s": round(sim_s, 1), "best_of_k": {}, "devices": {}}
    point_scores = {tags[i]: {"rrms": float(table[i, -1]),
                              "official_dynamic_rrms": float("nan")}
                    for i in range(len(tags))}
    report["point_estimate"] = aggregate(point_scores)
    for k in ks:
        best = {tags[i]: {"rrms": float(table[i, :k].min()),
                          "official_dynamic_rrms": float("nan")}
                for i in range(len(tags))}
        report["best_of_k"][str(k)] = {**aggregate(best), "ngspice_calls_per_device": k}
    best_index = table[:, :n_cand].argmin(axis=1)
    best_cards = {}
    for i, tag in enumerate(tags):
        card = jobs[owners.index((i, int(best_index[i])))][1]
        best_cards[tag] = card
        z = np.log(np.asarray([card[n] / centers[tag][n] if centers[tag][n] else 1.0
                               for n in names]))
        report["devices"][tag] = {
            "point_rrms": float(table[i, -1]),
            "best_rrms": float(table[i, best_index[i]]),
            "median_sample_rrms": float(np.median(table[i, :n_cand])),
            "best_card": card, "best_log_multiplier": dict(zip(names, z.tolist())),
            "sample_log_multiplier_std": dict(zip(names, (
                samples[i].std(0) * checkpoint["z_scale"]).tolist()))}
    if args.polish:
        widths = {tag: {n: math.log(args.polish_width) for n in names}
                  for tag in tags}
        spec = spec_from_centers(names, best_cards, widths)
        t2 = time.time()
        with get_context("spawn").Pool(args.workers) as pool:
            polished = dict(pool.map(
                _polish_job, [(tag, spec, args.polish_nfev) for tag in tags]))
        report["polish_s"] = round(time.time() - t2, 1)
        report["polished"] = aggregate(polished)
        report["polished"]["mean_ngspice_calls_per_device"] = float(np.mean(
            [row["n_sim"] for row in polished.values()]))
        for tag in tags:
            report["devices"][tag]["polished"] = polished[tag]
    save_json(run_dir / f"eval_{args.label}.json", report)
    print(json.dumps({"point": report["point_estimate"]["all_device_mean_rrms"],
                      "best_of_k": {k: v["all_device_mean_rrms"]
                                    for k, v in report["best_of_k"].items()},
                      "polished": report.get("polished", {}).get(
                          "all_device_mean_rrms")}, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--set", choices=("seven", "full"), required=True)
    t.add_argument("--model", choices=("flow", "wta"), default="flow")
    t.add_argument("--banks", nargs="+", required=True)
    t.add_argument("--run", required=True)
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--batch-size", type=int, default=512)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--weight-decay", type=float, default=1e-5)
    t.add_argument("--hidden", type=int, default=1024)
    t.add_argument("--context", type=int, default=512)
    t.add_argument("--velocity-hidden", type=int, default=512)
    t.add_argument("--velocity-layers", type=int, default=4)
    t.add_argument("--heads", type=int, default=16)
    t.add_argument("--relax", type=float, default=0.05)
    t.add_argument("--gain-sigma", type=float, default=0.03)
    t.add_argument("--point-sigma", type=float, default=0.01)
    t.add_argument("--floor-amps", type=float, default=1e-12)
    t.add_argument("--holdout-devices", default=None)
    t.add_argument("--discrepancy-prob", type=float, default=0.5)
    t.add_argument("--device", default="mps")
    t.add_argument("--seed", type=int, default=20261015)
    e = sub.add_parser("evaluate")
    e.add_argument("--run", required=True)
    e.add_argument("--label", default="main")
    e.add_argument("--samples", type=int, default=256)
    e.add_argument("--steps", type=int, default=64)
    e.add_argument("--polish", action="store_true")
    e.add_argument("--no-clip-prior", dest="clip_prior", action="store_false")
    e.add_argument("--polish-width", type=float, default=1.5)
    e.add_argument("--polish-nfev", type=int, default=300)
    e.add_argument("--workers", type=int, default=10)
    e.add_argument("--device", default="mps")
    e.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    return train(args) if args.command == "train" else evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
