#!/usr/bin/env python3
"""E2 probe: one-pass direct network with misspecification-style augmentation.

Pre-registration: out/research/explore/E2/prereg.json.

  # train one arm (N0 | M | S) and one seed (MPS, a few minutes)
  PYTHONPATH=src .venv/bin/python scripts/research/explore/e2_direct.py train --arm S --seed 0
  # one forward pass per target + NGSpice validation (raw), all groups
  PYTHONPATH=src .venv/bin/python scripts/research/explore/e2_direct.py raw --arm S --seed 0
  # fixed short FD polish of the raw prediction
  PYTHONPATH=src .venv/bin/python scripts/research/explore/e2_direct.py fd --arm S --seed 0 --groups real
  # references (released card, truth-seven card, nn_lookup) and the table
  PYTHONPATH=src .venv/bin/python scripts/research/explore/e2_direct.py refs
  PYTHONPATH=src .venv/bin/python scripts/research/explore/e2_direct.py table

Groups: real = 12 REAL_DEV measured devices; a0/a1 = tier-a simulated sets
0/1; c0/c1 = sky130 tier-c sets 0/1 (strength 2).  REAL_TEST is never read.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from cryoml.explore import e2_direct as E  # noqa: E402

MODELS = E.E2_DIR / "models"
EVAL = E.E2_DIR / "eval"
FD_OPTIONS = {"max_nfev": 100, "diff_step": 0.01, "irls_rounds": 1}


def torch_device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(name)


# ------------------------------------------------------------------ train ---

def train(args) -> int:
    t0 = time.time()
    spec = E.load_spec()
    bank = E.load_bank(spec, args.max_per_device)
    dev = torch_device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    n = len(bank.local)
    is_val = rng.random(n) < 0.05
    tr, va = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    # Feature standardization from unaugmented training currents (fixed
    # subsample, same rule in every arm).
    sub = np.random.default_rng(12345).choice(tr, size=min(40000, len(tr)),
                                              replace=False)
    slog = np.sign(bank.currents[sub]) * np.log1p(
        np.abs(bank.currents[sub].astype(np.float64)) / E.I_REF)
    mean, std = slog.mean(0), np.maximum(slog.std(0), 1e-3)
    feats = E.Features(mean, std, bank.curve_scales).to(dev)
    aug = E.Augmenter(bank.layout, bank.r_unit).to(dev)
    model = E.DirectNet(feats.n_features).to(dev)
    cur = torch.as_tensor(bank.currents, device=dev)
    loc = torch.as_tensor(bank.local, device=dev)
    ids = torch.as_tensor(bank.device_ids, device=dev)
    tr_t = torch.as_tensor(tr, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(tr) / args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    gen = torch.Generator(device=dev)
    gen.manual_seed(1000 + args.seed)
    history = []
    for epoch in range(args.epochs):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), device=dev)]
        total, count = 0.0, 0
        for start in range(0, len(perm), args.batch):
            b = perm[start:start + args.batch]
            x = aug(cur[b], ids[b], args.arm, gen)
            loss = torch.mean((model(feats(x, ids[b])) - loc[b]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(b)
            count += len(b)
        val = evaluate_val(model, feats, aug, cur, loc, ids, va, dev)
        history.append({"epoch": epoch + 1, "train_mse": total / count, **val})
        print(f"[{args.arm} s{args.seed}] epoch {epoch + 1} train {total / count:.4f} "
              f"val clean {val['val_mse_clean']:.4f} val S {val['val_mse_S']:.4f}",
              flush=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "feats": feats.state_dict(),
                "n_features": feats.n_features, "arm": args.arm,
                "seed": args.seed, "history": history,
                "n_train": int(len(tr)), "n_val": int(len(va)),
                "epochs": args.epochs, "batch": args.batch,
                "max_per_device": args.max_per_device,
                "runtime_s": round(time.time() - t0, 1)},
               MODELS / f"{args.arm}_s{args.seed}.pt")
    print(f"saved {MODELS / f'{args.arm}_s{args.seed}.pt'} "
          f"({time.time() - t0:.0f} s)")
    return 0


@torch.no_grad()
def evaluate_val(model, feats, aug, cur, loc, ids, va, dev) -> dict:
    model.eval()
    gen = torch.Generator(device=dev)
    gen.manual_seed(777)
    out = {}
    for name in ("clean", "S"):
        errs = []
        for start in range(0, len(va), 4096):
            b = torch.as_tensor(va[start:start + 4096], device=dev)
            x = cur[b] if name == "clean" else aug(cur[b], ids[b], "S", gen)
            errs.append(((model(feats(x, ids[b])) - loc[b]) ** 2).mean(1))
        out[f"val_mse_{name}"] = float(torch.cat(errs).mean())
    model.train()
    return out


def load_model(arm: str, seed: int, dev):
    ck = torch.load(MODELS / f"{arm}_s{seed}.pt", map_location="cpu",
                    weights_only=False)
    feats = E.Features(np.zeros(2046), np.ones(2046), np.ones((18, 2046)))
    feats.load_state_dict(ck["feats"])
    model = E.DirectNet(ck["n_features"])
    model.load_state_dict(ck["state"])
    return model.to(dev).eval(), feats.to(dev)


# --------------------------------------------------------------- predict ---

@torch.no_grad()
def predict(arm: str, seed: int, groups, dev) -> dict:
    model, feats = load_model(arm, seed, dev)
    out = {}
    for group in groups:
        out[group] = {}
        for tag in E.group_tags(group):
            x = torch.as_tensor(E.target_currents(group, tag)[None],
                                dtype=torch.float32, device=dev)
            i = torch.as_tensor([E.TAGS.index(tag)], device=dev)
            local = model(feats(x, i))[0].clamp(-1, 1).cpu().numpy()
            out[group][tag] = local.astype(float).tolist()
    return out


def nn_lookup(groups) -> dict:
    """Nearest bank sample (own device) in standardized signed-log space."""
    spec = E.load_spec()
    out = {g: {} for g in groups}
    for tag in E.TAGS:
        needed = [g for g in groups if tag in E.group_tags(g)]
        if not needed:
            continue
        with np.load(E.BANK_DIR / f"{tag}.npz", allow_pickle=True) as data:
            order, _ = E.canonical(data)
            ok = np.asarray(data["ok"], dtype=bool)
            cur = np.asarray(data["IDS"], dtype=np.float64)[ok][:, order]
            theta = np.asarray(data["THETA"], dtype=np.float64)[ok]
        finite = np.all(np.isfinite(cur), axis=1)
        cur, theta = cur[finite], theta[finite]
        slog = np.sign(cur) * np.log1p(np.abs(cur) / E.I_REF)
        mean, std = slog.mean(0), np.maximum(slog.std(0), 1e-3)
        z = (slog - mean) / std
        local = E.theta_to_local(E.box_for(spec, tag), theta)
        for g in needed:
            t = E.target_currents(g, tag)
            zt = (np.sign(t) * np.log1p(np.abs(t) / E.I_REF) - mean) / std
            k = int(np.argmin(np.sum((z - zt) ** 2, axis=1)))
            out[g][tag] = local[k].tolist()
    return out


# ----------------------------------------------------------- NGSpice jobs ---

def _set_override(group: str) -> None:
    from cryoml.data_io import TARGET_OVERRIDE_ENV
    path = E.override_path(group)
    if path is None:
        os.environ.pop(TARGET_OVERRIDE_ENV, None)
    else:
        os.environ[TARGET_OVERRIDE_ENV] = str(path)


def _status(tag: str, scored: dict) -> dict:
    from cryoml.strict_score import device_status, include_tags
    st = device_status(scored["per_curve"], include_tags(tag))
    return {"rrms": scored["rrms"], "strict_rrms": st["strict_rrms"],
            "penalized_rrms": st["penalized_rrms"], "status": st["status"],
            "n_failed": st["n_failed"], "per_curve": scored["per_curve"]}


def raw_job(job):
    group, tag, local, params = job
    _set_override(group)
    from cryoml.card_eval import device_by_tag, score_card
    from cryoml.data_io import load_device_curves
    spec = E.load_spec()
    if params is None and local is not None:
        params = E.box_for(spec, tag).local_to_params(np.asarray(local))
    curves = load_device_curves(device_by_tag(tag))
    scored = score_card(tag, params, curves=curves)
    rec = _status(tag, scored)
    rec["params"] = params
    rec["local"] = local
    truth = E.truth_local(group, tag, spec)
    if truth is not None and local is not None:
        rec["local_abs_err"] = float(np.mean(np.abs(np.asarray(local) - truth)))
    return group, tag, rec


def fd_job(job):
    group, tag, local = job
    _set_override(group)
    from cryoml.ngspice_fit import DeviceProblem, final_score, polish
    spec = E.load_spec()
    problem = DeviceProblem(tag, E.box_for(spec, tag))
    res = polish(problem, np.asarray(local, dtype=np.float64), **FD_OPTIONS)
    scored = final_score(problem, res["local"])
    rec = _status(tag, scored)
    rec.update(params=scored["params"], local=scored["local"],
               n_sim=int(res["n_sim"]), history=res["history"],
               runtime_s=res["runtime_s"], start_local=list(local))
    truth = E.truth_local(group, tag, spec)
    if truth is not None:
        rec["local_abs_err"] = float(np.mean(np.abs(np.asarray(rec["local"]) - truth)))
    return group, tag, rec


def run_jobs(fn, jobs, workers: int) -> dict:
    out: dict = {}
    if workers <= 1:
        results = map(fn, jobs)
    else:
        pool = get_context("spawn").Pool(workers)
        results = pool.imap_unordered(fn, jobs)
    for group, tag, rec in results:
        out.setdefault(group, {})[tag] = rec
    if workers > 1:
        pool.close()
        pool.join()
    return out


def raw(args) -> int:
    groups = args.groups.split(",")
    name = f"{args.arm}_s{args.seed}"
    if args.arm == "nn":
        preds = nn_lookup(groups)
    else:
        preds = predict(args.arm, args.seed, groups, torch_device(args.device))
    # Merge with stored predictions of other groups (fd() reads every group
    # from this file; a partial raw run must not drop the earlier groups).
    pred_path = EVAL / f"pred_{name}.json"
    stored = json.loads(pred_path.read_text()) if pred_path.exists() else {}
    stored.update(preds)
    E.save_json(pred_path, stored)
    jobs = [(g, t, preds[g][t], None) for g in groups for t in preds[g]]
    t0 = time.time()
    res = run_jobs(raw_job, jobs, args.workers)
    path = EVAL / f"raw_{name}.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(res)
    E.save_json(path, old)
    report(name, "raw", res, time.time() - t0)
    return 0


def fd(args) -> int:
    groups = args.groups.split(",")
    name = f"{args.arm}_s{args.seed}"
    preds = json.loads((EVAL / f"pred_{name}.json").read_text())
    path = EVAL / f"fd_{name}.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    jobs = [(g, t, preds[g][t]) for g in groups for t in preds[g]
            if not (args.resume and t in old.get(g, {}))]
    t0 = time.time()
    res = run_jobs(fd_job, jobs, args.workers)
    for g, rows in res.items():
        old.setdefault(g, {}).update(rows)
    E.save_json(path, old)
    report(name, "fd", res, time.time() - t0)
    return 0


def refs(args) -> int:
    groups = args.groups.split(",")
    jobs = []
    for g in groups:
        for t in E.group_tags(g):
            if g == "real":
                jobs.append((g, t, None, None))           # released card
            else:
                jobs.append((g, t, None, E.truth_params(g, t)))  # truth seven
    res = run_jobs(raw_job, jobs, args.workers)
    path = EVAL / "refs.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(res)
    E.save_json(path, old)
    report("refs (real: released card; sim: truth seven card)", "raw", res, 0)
    return 0


def report(name, stage, res, wall) -> None:
    for g in sorted(res):
        rows = res[g]
        pen = [r["penalized_rrms"] for r in rows.values()]
        nf = sum(r["status"] != "valid" for r in rows.values())
        err = [r.get("local_abs_err") for r in rows.values()]
        print(f"{name:28s} {stage:3s} {g:4s} n={len(rows):2d} mean(pen)="
              f"{np.mean(pen):.4f} median={np.median(pen):.4f} failed={nf} "
              f"|dlocal|={E.finite_mean(err):.3f}  ({wall:.0f}s)")


# ----------------------------------------------------------------- table ---

def _seed_mean(stage: str, arm: str, group: str, seeds) -> dict | None:
    per = {}
    for s in seeds:
        path = EVAL / f"{stage}_{arm}_s{s}.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text()).get(group)
        if not rows:
            continue
        for t, r in rows.items():
            per.setdefault(t, []).append(r)
    if not per:
        return None
    tags = E.group_tags(group)
    if set(per) != set(tags):
        return None
    n_seeds = min(len(v) for v in per.values())
    return {"per_device": {t: float(np.mean([r["penalized_rrms"] for r in per[t]]))
                           for t in tags},
            "n_failed": int(sum(r["status"] != "valid" for v in per.values()
                                for r in v)),
            "local_abs_err": E.finite_mean([r.get("local_abs_err")
                                            for v in per.values() for r in v]),
            "n_seeds": n_seeds}


def table(args) -> int:
    seeds = [int(s) for s in args.seeds.split(",")]
    pre = json.loads((E.E2_DIR / "prereg.json").read_text())
    D = 0.173493
    rows = {}
    for stage in ("raw", "fd"):
        for arm in ("N0", "M", "S", "nn"):
            for g in E.GROUPS:
                sm = _seed_mean(stage, arm, g, seeds if arm != "nn" else [0])
                if sm is None:
                    continue
                vals = list(sm["per_device"].values())
                rows[f"{stage}/{arm}/{g}"] = {
                    "mean_penalized": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "n_failed_device_seed_pairs": sm["n_failed"],
                    "local_abs_err": sm["local_abs_err"],
                    "n_seeds": sm["n_seeds"], "per_device": sm["per_device"]}
    refs_path = EVAL / "refs.json"
    if refs_path.exists():
        ref = json.loads(refs_path.read_text())
        for g, recs in ref.items():
            vals = [r["penalized_rrms"] for r in recs.values()]
            rows[f"ref/{'released' if g == 'real' else 'truth7'}/{g}"] = {
                "mean_penalized": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "per_device": {t: r["penalized_rrms"] for t, r in recs.items()}}
    decision = None
    n0, s_ = rows.get("raw/N0/real"), rows.get("raw/S/real")
    if n0 and s_:
        r0, rs = n0["mean_penalized"], s_["mean_penalized"]
        wins = sum(s_["per_device"][t] < n0["per_device"][t] for t in n0["per_device"])
        a0 = [rows.get(f"raw/N0/a{k}") for k in (0, 1)]
        aS = [rows.get(f"raw/S/a{k}") for k in (0, 1)]
        c4 = None
        if all(a0) and all(aS):
            ta0 = np.mean([r["mean_penalized"] for r in a0])
            taS = np.mean([r["mean_penalized"] for r in aS])
            c4 = bool(taS <= 1.5 * ta0 + 0.01)
        checks = {"C1": rs <= 0.70 * r0, "C2": (rs - D) <= 0.70 * (r0 - D),
                  "C3": wins >= 8, "C4_guard": c4,
                  "K1": rs > 0.85 * r0, "K2": wins < 6}
        if checks["K1"] or checks["K2"]:
            verdict = "KILL"
        elif checks["C1"] and checks["C2"] and checks["C3"] and c4:
            verdict = "CONTINUE"
        else:
            verdict = "INCONCLUSIVE"
        m = rows.get("raw/M/real")
        decision = {"R_raw_N0": r0, "R_raw_S": rs, "D": D,
                    "ratio_S_over_N0": rs / r0,
                    "gap_ratio": (rs - D) / (r0 - D), "S_wins": wins,
                    "checks": checks,
                    "attribution_S_over_M": (rs / m["mean_penalized"]) if m else None,
                    "seeds_used": n0["n_seeds"],
                    "verdict": verdict if n0["n_seeds"] == 3 and s_["n_seeds"] == 3
                    else f"{verdict} (exploratory: {n0['n_seeds']} seed(s); "
                         "the pre-registered decision needs 3)"}
    out = {"prereg": str(E.E2_DIR / "prereg.json"),
           "prereg_criterion": pre["decision"], "rows": rows, "decision": decision}
    E.save_json(E.E2_DIR / "table.json", out)
    print(f"{'row':22s} {'mean':>8s} {'median':>8s} {'fail':>5s} {'|dloc|':>7s} seeds")
    for k, r in rows.items():
        print(f"{k:22s} {r['mean_penalized']:8.4f} {r['median']:8.4f} "
              f"{r.get('n_failed_device_seed_pairs', ''):>5} "
              f"{r.get('local_abs_err', float('nan')) or float('nan'):7.3f} "
              f"{r.get('n_seeds', '')}")
    print(json.dumps(decision, indent=1))
    return 0


def full(args) -> int:
    """Pre-registered run: every arm x seed trained, raw on all groups, FD
    on the listed groups; existing models / raw files / FD targets are kept."""
    seeds = [int(s) for s in args.seeds.split(",")]
    for seed in seeds:
        for arm in E.ARMS:
            name = f"{arm}_s{seed}"
            if not (MODELS / f"{name}.pt").exists():
                train(argparse.Namespace(arm=arm, seed=seed, epochs=30, batch=1024,
                                         max_per_device=None, device=args.device))
            raw_path = EVAL / f"raw_{name}.json"
            have = json.loads(raw_path.read_text()) if raw_path.exists() else {}
            missing = [g for g in E.GROUPS if g not in have]
            if missing:
                raw(argparse.Namespace(arm=arm, seed=seed, groups=",".join(missing),
                                       workers=args.workers, device=args.device))
    if not (EVAL / "raw_nn_s0.json").exists():
        raw(argparse.Namespace(arm="nn", seed=0, groups=",".join(E.GROUPS),
                               workers=args.workers, device=args.device))
    if not (EVAL / "refs.json").exists():
        refs(argparse.Namespace(groups=",".join(E.GROUPS), workers=args.workers))
    # The decision uses raw values only: write the table before the FD stage.
    table(argparse.Namespace(seeds=args.seeds))
    for seed in seeds:
        for arm in E.ARMS:
            fd(argparse.Namespace(arm=arm, seed=seed, groups=args.fd_groups,
                                  workers=args.workers, resume=True))
    fd(argparse.Namespace(arm="nn", seed=0, groups=args.fd_groups,
                          workers=args.workers, resume=True))
    return table(argparse.Namespace(seeds=args.seeds))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("full")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--fd-groups", default=",".join(E.GROUPS))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="mps")
    p = sub.add_parser("train")
    p.add_argument("--arm", choices=E.ARMS, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--max-per-device", type=int, default=None)
    p.add_argument("--device", default="mps")
    for name in ("raw", "fd"):
        p = sub.add_parser(name)
        p.add_argument("--arm", choices=(*E.ARMS, "nn"), required=True)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--groups", default=",".join(E.GROUPS))
        p.add_argument("--workers", type=int, default=2)
        p.add_argument("--device", default="mps")
        p.add_argument("--resume", action="store_true")
    p = sub.add_parser("refs")
    p.add_argument("--groups", default=",".join(E.GROUPS))
    p.add_argument("--workers", type=int, default=2)
    p = sub.add_parser("table")
    p.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    return {"train": train, "raw": raw, "fd": fd, "refs": refs,
            "table": table, "full": full}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
