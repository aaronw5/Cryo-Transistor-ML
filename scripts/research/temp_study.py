#!/usr/bin/env python3
"""Simulated 77-300 K temperature benchmark (see ``cryoml.temperature``).

Each held-out truth device is observed at TEMPS_OBSERVED (77, 150, 225,
300 K); every method returns one 14-parameter temperature card per device,
which NGSpice re-simulates at all seven temperatures.  Scores are the
frozen-inclusion paper RRMS against the simulated truth curves, reported per
temperature, as the observed-temperature mean and as the held-out-temperature
(110, 190, 260 K) mean.  All methods share the bank box, the bank-derived
starts where they use starts, and the same NGSpice FD polish.

``pipeline_search`` is the paper pipeline's search
(``finite_difference_surrogate_extract`` in
scripts/foundation_parameter_study.py) with a pluggable evaluator: bank
starts, two-point TRF least squares from each on the pipeline's residual
(equal-weight per-curve normalized points plus sqrt(1e-4/14) * local), endpoints
ranked by the evaluator's mean curve RRMS, and candidates = top endpoints +
first starts + box centre.  ``surrogate`` and ``ngspice_search`` run it
identically, differing only in the evaluator, so ``ngspice_search`` is the
NGSpice control row for every surrogate ablation.

  direct          CMA-ES + TRF on NGSpice (no ML reference)
  surrogate       temperature-conditioned forward surrogate (--arch),
                  ``pipeline_search`` through it, NGSpice validation of the
                  candidates (best -> raw), then the same NGSpice FD polish
  ngspice_search  the same ``pipeline_search`` with NGSpice as the evaluator,
                  then the same validation and FD polish
  direct_mlp      observed I-V -> card in one forward pass (+FD as diagnostic)
  table           collect results

  PYTHONPATH=src .venv/bin/python scripts/research/temp_study.py direct --sets 0,1,2
  PYTHONPATH=src .venv/bin/python scripts/research/temp_study.py ngspice_search --sets 0,1,2
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cryoml.config import OUT_DIR, PROCESSED_DIR  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.metrics import family_totals  # noqa: E402
from cryoml.temperature import (TEMPS_ALL, TEMPS_HELDOUT,  # noqa: E402
                                TEMPS_OBSERVED, device_boxes,
                                temperature_feature)
from cryoml.utils import device_tag  # noqa: E402

ANCHOR = OUT_DIR / "cold_loop" / "seven" / "round_4" / "spec.json"
# Environment overrides exist for smoke tests on tiny banks.
BANK = Path(os.environ.get("CRYOML_TEMP_BANK", PROCESSED_DIR / "temp_bank" / "v1"))
TRUTH = Path(os.environ.get("CRYOML_TEMP_TRUTH", OUT_DIR / "research" / "temp" / "truth"))
RESULTS = Path(os.environ.get("CRYOML_TEMP_RESULTS", OUT_DIR / "research" / "temp"))
ALL_TAGS = [device_tag(d.dev_type, d.L_um, d.W_um) for d in PAPER_DEVICES]
TAGS = list(ALL_TAGS)          # job list; --devices narrows it for smoke tests


# ----------------------------------------------------------------- common

def bank_meta() -> dict:
    return json.loads((BANK / "meta.json").read_text())


def boxes():
    meta = bank_meta()
    return device_boxes(ANCHOR, meta["m_scale"], tuple(meta["g_range"]))


def load_truth(k: int) -> dict:
    with np.load(TRUTH / f"set_{k}.npz") as data:
        return {t: {"z": data[f"{t}__z"], "ids": data[f"{t}__ids"]} for t in ALL_TAGS}


def make_problem(tag, truth_row, temps, box=None):
    from cryoml.temp_fit import TempProblem
    box = box or boxes()[tag]
    targets = {t: truth_row["ids"][TEMPS_ALL.index(t)] for t in TEMPS_ALL}
    return TempProblem(tag, box, targets, tuple(temps))


def evaluate(tag, local, truth_row, box=None) -> dict:
    """NGSpice at all seven temperatures against the truth curves."""
    box = box or boxes()[tag]
    problem = make_problem(tag, truth_row, TEMPS_ALL, box)
    per_temp = problem.per_temp(problem.simulate(np.asarray(local)))
    true_local = box.z_to_local(truth_row["z"])
    return {"per_temp": per_temp,
            "observed": float(np.mean([per_temp[f"{t:g}"] for t in TEMPS_OBSERVED])),
            "heldout": float(np.mean([per_temp[f"{t:g}"] for t in TEMPS_HELDOUT])),
            "all": float(np.mean(list(per_temp.values()))),
            "local": [float(v) for v in local],
            "local_error_rms": float(np.sqrt(np.mean(
                (np.clip(local, -1, 1) - true_local) ** 2)))}


def bank_starts(problem, n_starts: int, min_distance: float = 0.35):
    from cryoml.temp_fit import bank_scores
    with np.load(BANK / f"{problem.tag}.npz") as data:
        ok = data["ok"]
        local = data["LOCAL"][ok]
        ids = data["IDS"][ok]
        temps = tuple(float(t) for t in data["temps"])
    scores = bank_scores(problem, ids, temps)
    chosen = []
    for index in np.argsort(scores):
        if not chosen or min(np.sqrt(np.mean((local[index] - c) ** 2))
                             for c in chosen) >= min_distance:
            chosen.append(local[index])
        if len(chosen) >= n_starts:
            break
    return chosen, float(scores.min())


def pipeline_search(problem, evaluate, n_starts=8, n_validate=4, max_nfev=100,
                    diff_step=0.02, regularization=1e-4) -> dict:
    """The pipeline's two-point TRF search with ``evaluate(local) -> sims``
    (``problem.simulate``'s layout) in place of the frozen surrogate.

    The residual is the pipeline's: ``residual_vector`` with unit weights is
    (sim - target) / (denominator * sqrt(n_curves * n_points)) per included
    curve, plus sqrt(regularization / 14) * local.  Endpoints are ranked by
    the evaluator's mean curve RRMS; nothing here calls NGSpice unless
    ``evaluate`` does.
    """
    from scipy.optimize import least_squares
    t0 = time.time()
    n_par = problem.box.n_parameters
    weights = np.ones(problem.n_curves)
    prior = math.sqrt(regularization / n_par)
    calls = [0]

    def counted(local):
        calls[0] += 1
        return evaluate(np.asarray(local, dtype=np.float64))

    def residual(local):
        return np.concatenate([problem.residual_vector(counted(local), weights),
                               prior * np.asarray(local, dtype=np.float64)])

    starts, best_bank = bank_starts(problem, n_starts)
    endpoints = []
    for index, start in enumerate(starts):
        before = calls[0]
        solution = least_squares(residual, np.asarray(start, dtype=np.float64),
                                 method="trf", jac="2-point",
                                 bounds=(-np.ones(n_par), np.ones(n_par)),
                                 diff_step=diff_step, max_nfev=max_nfev, x_scale="jac")
        endpoints.append({"kind": "fd_endpoint", "start_index": index,
                          "local": solution.x.tolist(), "nfev": int(solution.nfev),
                          "njev": int(solution.njev or 0), "n_eval": calls[0] - before})
    n_eval_search = calls[0]
    for row in endpoints:
        row["search_objective"] = float(np.mean(problem.curve_rrms(counted(row["local"]))))
    endpoints.sort(key=lambda row: row["search_objective"])
    candidates = endpoints[:max(1, n_validate)] + [
        {"kind": "bank_start", "start_index": index,
         "local": np.asarray(start, dtype=np.float64).tolist()}
        for index, start in enumerate(starts[:max(1, n_validate)])]
    candidates.append({"kind": "box_centre", "start_index": -1, "local": [0.0] * n_par})
    seen, unique = set(), []
    for row in candidates:
        key = np.asarray(row["local"], dtype=np.float64).tobytes()
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return {"candidates": unique,
            "endpoint_objectives": [row["search_objective"] for row in endpoints],
            "best_bank_observed": best_bank, "n_starts": len(starts),
            "n_eval": calls[0], "n_eval_search": n_eval_search,
            "search_s": round(time.time() - t0, 2)}


def search_options(args) -> dict:
    """The pipeline's search settings (its argparse defaults: 8 starts, 4
    validated endpoints, 100 nfev, diff_step 0.02, regularization 1e-4)."""
    return {"n_starts": args.n_starts, "n_validate": 4, "max_nfev": args.max_nfev,
            "diff_step": 0.02, "regularization": 1e-4}


def validate_and_polish(problem, tag, truth_row, candidates, polish_nfev) -> dict:
    """NGSpice objective of every candidate on the observed temperatures; the
    best is ``raw`` and its NGSpice FD polish is ``fd``."""
    t0, n0 = time.time(), problem.n_sim
    values = [problem.objective(np.asarray(c)) for c in candidates]
    selected = int(np.argmin(values))
    best = np.asarray(candidates[selected])
    record = {"raw": evaluate(tag, best, truth_row), "validated_objectives": values,
              "selected": selected, "validate_n_sim": problem.n_sim - n0,
              "validate_s": round(time.time() - t0, 1)}
    if polish_nfev > 0:
        from cryoml.ngspice_fit import polish
        t0, n0 = time.time(), problem.n_sim
        result = polish(problem, best, max_nfev=polish_nfev, irls_rounds=2)
        record["fd"] = evaluate(tag, result["local"], truth_row)
        record["polish_n_sim"] = problem.n_sim - n0
        record["polish_s"] = round(time.time() - t0, 1)
    return record


def aggregate(devices: dict, key: str) -> dict:
    out = {}
    for metric in ("observed", "heldout", "all"):
        rows = {t: {"rrms": r[key][metric]} for t, r in devices.items()}
        fam = family_totals({t: {"sigma": float("nan"), **v} for t, v in rows.items()})
        out[metric] = {"all_device_mean": float(np.mean([v["rrms"] for v in rows.values()])),
                       "combined": fam["combined_rrms"]}
    temps = next(iter(devices.values()))[key]["per_temp"].keys()
    out["per_temp_all_device_mean"] = {
        t: float(np.mean([r[key]["per_temp"][t] for r in devices.values()])) for t in temps}
    out["local_error_rms_mean"] = float(np.mean([r[key]["local_error_rms"] for r in devices.values()]))
    return out


def save(method: str, k: int, devices: dict, extra: dict) -> dict:
    keys = [key for key in ("raw", "fd") if key in next(iter(devices.values()))]
    summary = {key: aggregate(devices, key) for key in keys}
    path = RESULTS / method / f"set_{k}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary, **extra, "devices": devices},
                               indent=1) + "\n")
    print(method, k, json.dumps({key: {m: summary[key][m]["all_device_mean"]
                                       for m in ("observed", "heldout")}
                                 for key in keys}), flush=True)
    return summary


def polish_job(job):
    from cryoml.ngspice_fit import polish
    tag, k, local, polish_nfev = job
    truth = load_truth(k)[tag]
    problem = make_problem(tag, truth, TEMPS_OBSERVED)
    result = polish(problem, np.asarray(local), max_nfev=polish_nfev, irls_rounds=2)
    return tag, {"fd": evaluate(tag, result["local"], truth),
                 "fd_n_sim": problem.n_sim}


# ----------------------------------------------------------------- direct

def direct_job(job):
    import cma
    tag, k, args = job
    t0 = time.time()
    truth = load_truth(k)[tag]
    problem = make_problem(tag, truth, TEMPS_OBSERVED)
    starts = [np.zeros(problem.box.n_parameters)]
    extra, best_bank = bank_starts(problem, args["bank_starts"])
    starts += extra
    values = [problem.objective(s) for s in starts]
    best_local = np.asarray(starts[int(np.argmin(values))], dtype=np.float64)
    best_value = float(min(values))
    es = cma.CMAEvolutionStrategy(best_local, args["sigma0"], {
        "bounds": [-1, 1], "maxfevals": args["cmaes_evals"], "seed": args["seed"],
        "verbose": -9})
    while not es.stop():
        xs = es.ask()
        es.tell(xs, [problem.objective(np.clip(x, -1, 1)) for x in xs])
    if es.result.fbest < best_value:
        best_value, best_local = float(es.result.fbest), np.clip(es.result.xbest, -1, 1)
    raw_sims = problem.n_sim
    from cryoml.ngspice_fit import polish
    result = polish(problem, best_local, max_nfev=args["polish_nfev"], irls_rounds=2)
    return tag, {"raw": evaluate(tag, best_local, truth),
                 "fd": evaluate(tag, result["local"], truth),
                 "best_bank_observed": best_bank, "raw_n_sim": raw_sims,
                 "n_sim": problem.n_sim, "runtime_s": round(time.time() - t0, 1)}


def cmd_direct(args):
    options = {"bank_starts": 4, "cmaes_evals": args.cmaes_evals, "sigma0": 0.3,
               "polish_nfev": args.polish_nfev, "seed": 20261016}
    for k in [int(s) for s in args.sets.split(",")]:
        t0 = time.time()
        with get_context("spawn").Pool(args.workers) as pool:
            devices = dict(pool.imap_unordered(direct_job, [(t, k, options) for t in TAGS]))
        save(args.name or "direct", k, devices,
             {"options": options, "wall_s": round(time.time() - t0, 1),
              "mean_ngspice_calls": float(np.mean([r["n_sim"] for r in devices.values()]))})


# -------------------------------------------------------------- surrogate

def load_training(temps):
    """(features, slog, current, device ids, curve scales) in canonical order."""
    from cryoml.foundation import canonical_point_order, geometry_features, geometry_ranges
    ranges = geometry_ranges(PAPER_DEVICES)
    feats, currents, ids, sample_ids, orders, scales = [], [], [], [], {}, []
    offset = 0
    for d_index, (tag, device) in enumerate(zip(ALL_TAGS, PAPER_DEVICES)):
        with np.load(BANK / f"{tag}.npz") as data:
            order, desc = canonical_point_order(data["kinds"], data["fixeds"], data["slices"])
            orders[tag] = order
            ok = data["ok"]
            local = data["LOCAL"][ok].astype(np.float32)
            bank_temps = [float(t) for t in data["temps"]]
            block = data["IDS"][ok][:, [bank_temps.index(t) for t in temps]][:, :, order]
        geometry = geometry_features(device, ranges)
        n = len(local)
        for j, temp in enumerate(temps):
            feats.append(np.concatenate([
                local, np.broadcast_to(geometry, (n, 3)),
                np.full((n, 1), temperature_feature(temp), dtype=np.float32)], 1))
            currents.append(block[:, j])
            ids.append(np.full(n, d_index))
            sample_ids.append(offset + np.arange(n))
        offset += n
        peak = np.abs(block).reshape(n, len(temps), 11, -1).max(-1)       # (n, T, 11)
        curve_scale = np.median(peak.reshape(-1, 11), axis=0)
        scales.append(np.repeat(np.maximum(curve_scale, 1e-12), block.shape[-1] // 11))
    return (np.concatenate(feats).astype(np.float32),
            np.concatenate(currents).astype(np.float32),
            np.concatenate(ids), np.concatenate(sample_ids),
            np.stack(scales).astype(np.float32), orders)


def train_surrogate(arch: str, out_dir: Path, epochs: int, batch_size: int,
                    patience: int, seed: int = 20261016):
    import torch
    import torch.nn.functional as F
    from cryoml.foundation import current_from_signed_log, signed_log
    from cryoml.operators import build
    path = out_dir / "model.pt"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    feats, current, ids, sample_ids, scales, orders = load_training(TEMPS_OBSERVED)
    slog = signed_log(torch.tensor(current))
    mean, std = slog.mean(0), slog.std(0).clamp_min(1e-3)
    model = build(arch, feats.shape[1], temperature=True).to(device)
    if path.exists():
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state["state"])
        model.eval()
        return model, state["mean"].to(device), state["std"].to(device), orders, device
    rng = np.random.default_rng(seed)
    n_samples = sample_ids.max() + 1
    val_samples = rng.random(n_samples) < 0.1
    is_val = val_samples[sample_ids]
    X = torch.tensor(feats, device=device)
    Y = ((slog - mean) / std).to(device)
    C = torch.tensor(current, device=device)
    D = torch.tensor(ids, device=device)
    S = torch.tensor(scales, device=device)
    mean_d, std_d = mean.to(device), std.to(device)
    train_idx = torch.tensor(np.flatnonzero(~is_val), device=device)
    val_idx = torch.tensor(np.flatnonzero(is_val), device=device)

    def loss_fn(index):
        prediction = model(X[index])
        slog_loss = F.mse_loss(prediction, Y[index])
        pred_current = current_from_signed_log(prediction * std_d + mean_d)
        linear = F.smooth_l1_loss((pred_current - C[index]) / S[D[index]],
                                  torch.zeros_like(pred_current), beta=1.0)
        return slog_loss + linear

    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best, best_state, bad, history, t0 = math.inf, None, 0, [], time.time()
    for epoch in range(epochs):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), device=device)]
        total = 0.0
        for start in range(0, len(perm), batch_size):
            index = perm[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(index)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total += float(loss.detach()) * len(index)
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val = sum(float(loss_fn(val_idx[s:s + 4096])) * len(val_idx[s:s + 4096])
                      for s in range(0, len(val_idx), 4096)) / len(val_idx)
        history.append({"epoch": epoch + 1, "train": total / len(train_idx), "val": val})
        print(f"{arch} epoch {epoch + 1} train {total / len(train_idx):.5f} val {val:.5f} "
              f"{time.time() - t0:.0f}s", flush=True)
        if val < best - 1e-6:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "mean": mean, "std": std, "arch": arch}, path)
    (out_dir / "training.json").write_text(json.dumps({
        "arch": arch, "best_val": best, "epochs": len(history),
        "train_s": round(time.time() - t0, 1), "n_train": int(len(train_idx)),
        "history": history}, indent=1) + "\n")
    return model, mean_d, std_d, orders, device


def heldout_emulator_error(model, mean, std, device):
    """Emulator loss at observed vs held-out temperatures on bank samples."""
    import torch
    from cryoml.foundation import current_from_signed_log
    out = {}
    for temps, name in ((TEMPS_OBSERVED, "observed"), (TEMPS_HELDOUT, "heldout")):
        feats, current, ids, sample_ids, scales, _ = load_training(temps)
        rng = np.random.default_rng(7)
        pick = rng.choice(len(feats), size=min(40000, len(feats)), replace=False)
        errors = []
        with torch.no_grad():
            for s in range(0, len(pick), 4096):
                index = pick[s:s + 4096]
                pred = current_from_signed_log(
                    model(torch.tensor(feats[index], device=device)) * std + mean).cpu().numpy()
                true = current[index]
                err = (pred - true).reshape(len(index), 11, -1)
                peak = np.abs(true).reshape(len(index), 11, -1).max(-1)
                peak = np.maximum(peak, 1e-3 * peak.max(-1, keepdims=True))
                errors.append((np.sqrt((err ** 2).mean(-1)) / peak).mean(-1))
        errors = np.concatenate(errors)
        out[name] = {"curve_nrmse_median": float(np.median(errors)),
                     "curve_nrmse_mean": float(np.mean(errors))}
    return out


def surrogate_evaluator(model, mean, std, device, order, problem):
    """``evaluate(local) -> sims`` from the frozen surrogate, in
    ``problem.simulate``'s layout: the model's canonical-order output at each
    of ``problem.temps`` is put back in load order (canonical = load[order])
    and split by ``problem.bounds`` into curve ``j * n_curves + i``."""
    import torch
    from cryoml.foundation import current_from_signed_log, geometry_features, geometry_ranges
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    n_temps = len(problem.temps)
    geometry = geometry_features(PAPER_DEVICES[ALL_TAGS.index(problem.tag)],
                                 geometry_ranges(PAPER_DEVICES))
    fixed = np.concatenate([
        np.broadcast_to(geometry, (n_temps, 3)),
        np.array([[temperature_feature(t)] for t in problem.temps])], 1).astype(np.float32)

    def evaluate(local):
        x = np.concatenate([np.broadcast_to(np.asarray(local, dtype=np.float32),
                                            (n_temps, len(local))), fixed], 1)
        with torch.no_grad():
            slog = model(torch.tensor(x, device=device)) * std + mean
            current = current_from_signed_log(slog).cpu().numpy().astype(np.float64)
        current = current[:, inverse]
        return [current[j, lo:hi] for j in range(n_temps) for lo, hi in problem.bounds]

    return evaluate


def search_device(name, model, mean, std, orders, problem, n_calls=30):
    """The device for the batch-4 search forward passes, its model copy and
    the per-call seconds; ``auto`` times both and keeps the faster (per-call
    sync dominates small models on MPS, compute dominates large ones on CPU)."""
    import copy
    import torch
    names = ["cpu", "mps"] if name == "auto" else [name]
    names = [n for n in names if n == "cpu" or torch.backends.mps.is_available()]
    rng = np.random.default_rng(0)
    timings, placed = {}, {}
    for n in names:
        device = torch.device(n)
        placed[n] = (copy.deepcopy(model).to(device), mean.to(device), std.to(device), device)
        evaluate = surrogate_evaluator(*placed[n], orders[problem.tag], problem)
        evaluate(np.zeros(14))
        t0 = time.perf_counter()
        for _ in range(n_calls):
            evaluate(rng.uniform(-1, 1, 14))
        timings[n] = (time.perf_counter() - t0) / n_calls
    best = min(timings, key=timings.get)
    return placed[best], timings


def validate_job(job):
    tag, k, candidates, polish_nfev = job
    truth = load_truth(k)[tag]
    problem = make_problem(tag, truth, TEMPS_OBSERVED)
    record = validate_and_polish(problem, tag, truth, candidates, polish_nfev)
    record["n_sim"] = problem.n_sim
    return tag, record


def cmd_surrogate(args):
    out_dir = RESULTS / f"surrogate_{args.arch}"
    model, mean, std, orders, device = train_surrogate(
        args.arch, out_dir, args.epochs, args.batch_size, args.patience)
    emulator = heldout_emulator_error(model, mean, std, device)
    print("emulator", json.dumps(emulator), flush=True)
    (model, mean, std, target), timings = search_device(
        args.search_device, model, mean, std, orders,
        make_problem(TAGS[0], load_truth(int(args.sets.split(",")[0]))[TAGS[0]],
                     TEMPS_OBSERVED))
    print("search device", target, json.dumps(timings), flush=True)
    options = search_options(args)
    for k in [int(s) for s in args.sets.split(",")]:
        t0 = time.time()
        truth = load_truth(k)
        searched = {}
        for tag in TAGS:
            problem = make_problem(tag, truth[tag], TEMPS_OBSERVED)
            searched[tag] = pipeline_search(
                problem, surrogate_evaluator(model, mean, std, target, orders[tag], problem),
                **options)
        search_s = time.time() - t0
        jobs = [(tag, k, [c["local"] for c in searched[tag]["candidates"]], args.polish_nfev)
                for tag in TAGS]
        with get_context("spawn").Pool(args.workers) as pool:
            devices = dict(pool.imap_unordered(validate_job, jobs))
        for tag in TAGS:
            devices[tag]["search"] = searched[tag]
            devices[tag]["selected_kind"] = \
                searched[tag]["candidates"][devices[tag]["selected"]]["kind"]
        save(f"surrogate_{args.arch}", k, devices,
             {"emulator": emulator, "options": {**options, "polish_nfev": args.polish_nfev,
                                                "search_device": str(target),
                                                "eval_s_by_device": timings},
              "search_s": round(search_s, 1), "wall_s": round(time.time() - t0, 1),
              "mean_search_evals": float(np.mean([r["search"]["n_eval"]
                                                  for r in devices.values()]))})


# --------------------------------------------------------- NGSpice search

def ngspice_search_job(job):
    """The surrogate method with ``problem.simulate`` as the evaluator."""
    tag, k, options, polish_nfev = job
    t0 = time.time()
    truth = load_truth(k)[tag]
    problem = make_problem(tag, truth, TEMPS_OBSERVED)
    searched = pipeline_search(problem, problem.simulate, **options)
    search_n_sim = problem.n_sim
    record = validate_and_polish(problem, tag, truth,
                                 [c["local"] for c in searched["candidates"]], polish_nfev)
    record.update(search=searched, search_n_sim=search_n_sim, n_sim=problem.n_sim,
                  selected_kind=searched["candidates"][record["selected"]]["kind"],
                  runtime_s=round(time.time() - t0, 1))
    return tag, record


def cmd_ngspice_search(args):
    options = search_options(args)
    for k in [int(s) for s in args.sets.split(",")]:
        t0 = time.time()
        with get_context("spawn").Pool(args.workers) as pool:
            devices = dict(pool.imap_unordered(
                ngspice_search_job, [(t, k, options, args.polish_nfev) for t in TAGS]))
        save("ngspice_search", k, devices,
             {"options": {**options, "polish_nfev": args.polish_nfev},
              "wall_s": round(time.time() - t0, 1),
              **{f"mean_{key}": float(np.mean([r.get(key, 0) for r in devices.values()]))
                 for key in ("search_n_sim", "validate_n_sim", "polish_n_sim", "n_sim",
                             "runtime_s")}})


# ------------------------------------------------------------- direct MLP

def inverse_inputs(current, scales_by_point, device_ids, mean, std, stride):
    """Signed-log standardized and curve-scale-normalized currents at the
    observed temperatures, subsampled every ``stride`` points."""
    from cryoml.foundation import signed_log
    import torch
    slog = (signed_log(torch.tensor(current)) - mean) / std
    linear = torch.tensor(current) / torch.tensor(scales_by_point[device_ids])[:, None, :]
    pick = slice(None, None, stride)
    return torch.cat([slog[..., pick].flatten(1),
                      torch.clamp(linear[..., pick], -20, 20).flatten(1)], 1)


def cmd_direct_mlp(args):
    import torch
    import torch.nn as nn
    from cryoml.foundation import canonical_point_order, geometry_features, geometry_ranges, signed_log
    from cryoml.operators import ResMLP
    out_dir = RESULTS / "direct_mlp"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ranges = geometry_ranges(PAPER_DEVICES)
    locals_, currents, ids, orders, scales = [], [], [], {}, []
    for d_index, (tag, dev) in enumerate(zip(ALL_TAGS, PAPER_DEVICES)):
        with np.load(BANK / f"{tag}.npz") as data:
            order, _ = canonical_point_order(data["kinds"], data["fixeds"], data["slices"])
            orders[tag] = order
            ok = data["ok"]
            bank_temps = [float(t) for t in data["temps"]]
            block = data["IDS"][ok][:, [bank_temps.index(t) for t in TEMPS_OBSERVED]][:, :, order]
            locals_.append(data["LOCAL"][ok].astype(np.float32))
        currents.append(block.astype(np.float32))
        ids.append(np.full(len(block), d_index))
        peak = np.abs(block).reshape(len(block), len(TEMPS_OBSERVED), 11, -1).max(-1)
        scales.append(np.repeat(np.maximum(np.median(peak.reshape(-1, 11), 0), 1e-12),
                                block.shape[-1] // 11))
    local = np.concatenate(locals_)
    current = np.concatenate(currents)
    ids = np.concatenate(ids)
    scales = np.stack(scales).astype(np.float32)
    geometry = np.stack([geometry_features(d, ranges) for d in PAPER_DEVICES]).astype(np.float32)
    slog_all = signed_log(torch.tensor(current))
    mean, std = slog_all.mean(0), slog_all.std(0).clamp_min(1e-3)
    del slog_all
    X = torch.cat([inverse_inputs(current, scales, ids, mean, std, args.stride),
                   torch.tensor(geometry[ids])], 1)
    Y = torch.tensor(local)
    rng = np.random.default_rng(20261016)
    is_val = rng.random(len(X)) < 0.1
    X, Y = X.to(device), Y.to(device)
    tr, va = (torch.tensor(np.flatnonzero(~is_val), device=device),
              torch.tensor(np.flatnonzero(is_val), device=device))
    torch.manual_seed(20261016)
    model = nn.Sequential(ResMLP(X.shape[1], 14, width=1024, depth=4), nn.Tanh()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98)
    best, best_state, bad, t0 = math.inf, None, 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), device=device)]
        for s in range(0, len(perm), 512):
            index = perm[s:s + 512]
            optimizer.zero_grad()
            loss = torch.mean((model(X[index]) - Y[index]) ** 2)
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val = float(torch.mean((model(X[va]) - Y[va]) ** 2))
        print(f"direct_mlp epoch {epoch + 1} val {val:.5f} {time.time() - t0:.0f}s", flush=True)
        if val < best - 1e-6:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    train_info = {"best_val_mse_local": best, "train_s": round(time.time() - t0, 1),
                  "n_inputs": int(X.shape[1]), "stride": args.stride}
    for k in [int(s) for s in args.sets.split(",")]:
        truth = load_truth(k)
        predictions = {}
        for tag in TAGS:
            d_index = ALL_TAGS.index(tag)
            target = np.stack([truth[tag]["ids"][TEMPS_ALL.index(t)] for t in TEMPS_OBSERVED])
            target = target[:, orders[tag]][None].astype(np.float32)
            x = torch.cat([inverse_inputs(target, scales, np.array([d_index]), mean, std,
                                          args.stride),
                           torch.tensor(geometry[[d_index]])], 1).to(device)
            with torch.no_grad():
                predictions[tag] = model(x)[0].cpu().numpy().astype(np.float64)
        with get_context("spawn").Pool(args.workers) as pool:
            raw = dict(pool.map(_raw_eval_job, [(t, k, predictions[t]) for t in TAGS]))
            polished = dict(pool.imap_unordered(
                polish_job, [(t, k, predictions[t], args.polish_nfev) for t in TAGS]))
        devices = {t: {"raw": raw[t], **polished[t]} for t in TAGS}
        save("direct_mlp", k, devices, {"training": train_info})


def _raw_eval_job(job):
    tag, k, local = job
    return tag, evaluate(tag, local, load_truth(k)[tag])


# ------------------------------------------------------------------ table

def cmd_table(args):
    rows = []
    for method_dir in sorted(RESULTS.iterdir()):
        files = sorted(method_dir.glob("set_*.json"))
        if not files:
            continue
        data = [json.loads(f.read_text())["summary"] for f in files]
        for key in ("raw", "fd"):
            if key not in data[0]:
                continue
            row = {"method": f"{method_dir.name}{'+FD' if key == 'fd' else ''}",
                   "n_sets": len(data)}
            for metric in ("observed", "heldout"):
                for conv in ("all_device_mean", "combined"):
                    row[f"{metric}_{conv}"] = float(np.mean([d[key][metric][conv] for d in data]))
            row["per_temp"] = {t: float(np.mean([d[key]["per_temp_all_device_mean"][t]
                                                 for d in data]))
                               for t in data[0][key]["per_temp_all_device_mean"]}
            row["local_error_rms"] = float(np.mean([d[key]["local_error_rms_mean"] for d in data]))
            rows.append(row)
    (RESULTS / "table.json").write_text(json.dumps(rows, indent=1) + "\n")
    temps = [f"{t:g}" for t in TEMPS_ALL]
    print("| method | sets | observed | held-out | combined obs/held | "
          + " | ".join(f"{t} K" for t in temps) + " | param err |")
    for r in rows:
        print(f"| {r['method']} | {r['n_sets']} | {r['observed_all_device_mean']:.4f} | "
              f"{r['heldout_all_device_mean']:.4f} | {r['observed_combined']:.4f}/"
              f"{r['heldout_combined']:.4f} | "
              + " | ".join(f"{r['per_temp'][t]:.4f}" for t in temps)
              + f" | {r['local_error_rms']:.3f} |")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("direct")
    p.add_argument("--sets", default="0,1,2")
    p.add_argument("--cmaes-evals", type=int, default=1500)
    p.add_argument("--polish-nfev", type=int, default=400)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--name", default=None)
    p = sub.add_parser("surrogate")
    p.add_argument("--arch", default="paper_mlp")
    p.add_argument("--sets", default="0,1,2")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--polish-nfev", type=int, default=400)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--search-device", default="auto", choices=("auto", "cpu", "mps"),
                   help="device for the batch-4 search forward passes (auto: faster)")
    p = sub.add_parser("ngspice_search")
    p.add_argument("--sets", default="0,1,2")
    p.add_argument("--polish-nfev", type=int, default=400)
    p.add_argument("--workers", type=int, default=10)
    for p in (sub.choices["surrogate"], sub.choices["ngspice_search"]):
        p.add_argument("--n-starts", type=int, default=8)
        p.add_argument("--max-nfev", type=int, default=100)
    p = sub.add_parser("direct_mlp")
    p.add_argument("--sets", default="0,1,2")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--polish-nfev", type=int, default=400)
    p.add_argument("--workers", type=int, default=10)
    sub.add_parser("table")
    parser.add_argument("--devices", default=None, help="comma-separated tags (smoke tests)")
    args = parser.parse_args()
    if args.devices:
        TAGS[:] = [t for t in TAGS if t in args.devices.split(",")]
    {"direct": cmd_direct, "surrogate": cmd_surrogate, "ngspice_search": cmd_ngspice_search,
     "direct_mlp": cmd_direct_mlp, "table": cmd_table}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
