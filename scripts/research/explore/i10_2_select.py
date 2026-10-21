#!/usr/bin/env python3
"""I10-2: selection of the class of the shared bias-only correction g from
training data only (retrospective DEV analysis on the stored I9-2 runs).

Pre-registration: ``out/research/explore/I10_2/prereg.json`` (``inner``,
``refit``, ``stored`` and ``decide`` refuse unless it is committed and
unmodified).  Library: ``cryoml.explore.i10_2_select``.

Every selector picks one class of the menu (none, lin, gam, mlp8, mlp16) in
each outer fold of the stored I9-2 runs from the fold's training devices only;
the held-out score of the fold's devices is the stored I9-2 row of the chosen
class (``none``: the stored C0_refit row).  No outer refit is run.

Modes:
  plan     inner folds, task counts, compute estimate -> plan.json (no training)
  smoke    sky130 77 K, 2 outer folds, reduced settings: inner training + post hoc
           scores, refits of one inner fold; synthetic outer rows for the
           statistics (no stored held-out row is read) -> smoke/
  inner    inner training on every outer fold, S_loss scores and selections;
           V_fold check (lin retrain = stored I9-2 lin fold model) -> inner/<dataset>/
  refit    inner held-out refits with the inner corrections, S_refit scores and
           selections (--dataset) -> refit/<dataset>/
  stored   S_gate and S_cv only (stored rows, seconds) -> interim.json
  decide   all selectors, statistics and the rule -> decision.json (+ decision.md)

  PYTHONPATH=src .venv/bin/python scripts/research/explore/i10_2_select.py --mode inner --workers 6
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cryoml.explore.e1_correction import make_batch, score_device  # noqa: E402
from cryoml.explore.i10_2_select import (  # noqa: E402
    ALPHA, CONTROL, FULL_MENU, INNER_SEEDS, MARGIN, MENU, MULTI_CLASS, NONE, REFERENCE,
    SELECTORS, TIE, agreement, assemble, bonferroni_alpha, class_size, cv_logs, fold_scores,
    gate_select, gmr, inner_folds, oracle_classes, overall_decision, refit_inner_index,
    select_class, selection_counts, selector_label)
from cryoml.explore.i9_2_classes import correction_from_json_i9, fit_class  # noqa: E402
import i9_2_classes as d92  # noqa: E402  (I9-2 driver: records, folds, stored rows, refit path)

EX = ROOT / "out/research/explore"
I92 = EX / "I9_2"
OUT = EX / "I10_2"
PREREG = OUT / "prereg.json"
PRIMARY = d92.PRIMARY
OPTS = dict(d92.OPTS)
SMOKE = {"steps": 50, "heldout_nfev": 10, "heldout_irls": 1}
REFIT_SCOPE = {"sky130_77K": "all", "nist_4K": "one", "tudelft_4K": "all", "ihp_meas_300K": "all"}
V_FOLD_TOL = 1e-12
TIE_SENSITIVITY = (0.0, 0.01, 0.02)


def _json(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def _nan(d: dict) -> dict:
    return {t: (float("nan") if v is None else float(v)) for t, v in d.items()}


def _clean(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=1) + "\n")


def require_prereg() -> dict:
    rel = str(PREREG.relative_to(ROOT))
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel],
                             capture_output=True).returncode == 0
    clean = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", rel]).returncode == 0
    if not (PREREG.exists() and tracked and clean):
        raise SystemExit(f"{rel} is not committed: the pre-registration is not in force")
    sha = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", rel],
                         capture_output=True, text=True).stdout.strip()
    return {"prereg": rel, "commit": sha}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def groups_of(data: dict) -> dict:
    """Inner grouping unit: the TU Delft geometry group, else the device."""
    return dict(data["clusters"]) if data["clusters"] is not None else {t: t for t in data["by_tag"]}


def outer_plan(name: str, data: dict, smoke: bool = False) -> list[dict]:
    folds = data["folds"]
    if smoke:
        fam = data["family"]
        folds = [next(f for f in folds if fam[f] == "nmos"), next(f for f in folds if fam[f] == "pmos")]
    grp = groups_of(data)
    out = []
    for f in folds:
        pos = data["folds"].index(f)
        inner = inner_folds(data["train"][f], grp, data["family"])
        scope = REFIT_SCOPE[name]
        ks = list(range(len(inner))) if scope == "all" else [refit_inner_index(pos, len(inner))]
        if smoke:
            ks = [0]
        out.append({"fold": f, "pos": pos, "inner": inner, "refit_inner": ks})
    return out


SYNTHETIC = {"on": False}


def synthetic_rows(tags: list, excluded=(), labels=FULL_MENU, seed: int = 0) -> dict:
    """Random rows with the real tags (smoke of the decide path; no stored
    held-out row is read)."""
    rng = np.random.default_rng(seed)
    base = {t: float(rng.uniform(0.02, 0.1)) for t in tags}
    out = {CONTROL: dict(base)}
    for c in labels:
        if c != NONE:
            out[c] = {t: base[t] * float(np.exp(rng.normal(-0.03, 0.1))) for t in tags}
    for lab in out:
        for t in excluded:
            out[lab][t] = float("nan")
    return out


def outer_rows(name: str) -> dict:
    if SYNTHETIC["on"]:
        d = d92.load_records(name)
        return synthetic_rows(sorted(d["by_tag"]) + list(d["excluded"]), d["excluded"])
    return {lab: _nan(v) for lab, v in _json(I92 / name / "results.json")["per_device"].items()}


def start_ok(recs: list) -> dict:
    """Starting-record RRMS finite and > 0 (reference of the S_loss log ratio)."""
    out = {}
    for r in recs:
        v = score_device(r, [c.sim for c in r.curves])["rrms"]
        out[r.tag] = bool(np.isfinite(v) and v > 0)
    return out


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------
def inner_task(task):
    """(key, class, training records, scored records, overrides) -> (key, corr,
    {tag: ln(RRMS corrected / RRMS start)}, seconds, n_points)."""
    import torch
    torch.set_num_threads(1)
    from cryoml.explore.i7_1_transfer import s3_fold_value
    key, cls, train, scored, ov = task
    t0 = time.time()
    corr = fit_class(cls, train, ov, threads=1)
    logs = {}
    if scored:
        logs = dict(zip([r.tag for r in scored], s3_fold_value(corr, scored)))
    return key, corr, logs, round(time.time() - t0, 3), int(len(make_batch(train).sim))


def refit_worker(task):
    key, args = task
    row = d92.refit_task(args)
    return key, {k: v for k, v in row.items() if k not in ("params", "history", "corrections")}


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def run_plan(args) -> int:
    cost = dict(_json(I92 / "cost_basis.json")["per_point_step_member_s"])
    plan = {"menu": list(MENU), "inner_seeds": list(INNER_SEEDS), "workers": args.workers,
            "refit_scope": REFIT_SCOPE, "datasets": {}}
    tot_train = tot_refit = 0.0
    for name in PRIMARY:
        data = d92.load_records(name)
        op = outer_plan(name, data)
        by = data["by_tag"]
        pts = {t: len(make_batch([by[t]]).sim) for t in by}
        n_train_tasks = n_refits = 0
        train_s = 0.0
        sizes = []
        for o in op:
            sizes.append(len(o["inner"]))
            tr_all = data["train"][o["fold"]]
            for k, part in enumerate(o["inner"]):
                p = sum(pts[t] for t in tr_all if t not in set(part))
                for c in MENU[1:]:
                    n_train_tasks += 1
                    train_s += cost[c] * p * 2000 * len(INNER_SEEDS)
                if k in o["refit_inner"]:
                    n_refits += len(part) * (len(MENU) - 1)
        check_s = sum(cost["lin"] * sum(pts[t] for t in data["train"][o["fold"]]) * 2000 for o in op)
        tm = d92.stored_refit_seconds(name)
        refit_s = n_refits * tm["corrected_refit_s"]
        tot_train += train_s + check_s
        tot_refit += refit_s
        plan["datasets"][name] = {
            "n_outer": len(op), "inner_folds_per_outer": sorted(set(sizes)),
            "n_inner_train_tasks": n_train_tasks,
            "train_cpu_h": round((train_s + check_s) / 3600, 2), "n_inner_refits": n_refits,
            "refit_s_each": tm["corrected_refit_s"], "refit_cpu_h": round(refit_s / 3600, 2),
            "inner_train_wall_min": round((train_s + check_s) / (args.workers * 0.85) / 60, 1),
            "refit_wall_min": round(refit_s / (args.workers * 0.85) / 60, 1)}
        print(name, json.dumps(plan["datasets"][name]), flush=True)
    plan["total"] = {"train_cpu_h": round(tot_train / 3600, 2), "refit_cpu_h": round(tot_refit / 3600, 2),
                     "wall_min": round((tot_train + tot_refit) / (args.workers * 0.85) / 60, 1)}
    print(json.dumps(plan["total"]))
    dump(OUT / "plan.json", plan)
    return 0


# ---------------------------------------------------------------------------
# inner (S_loss)
# ---------------------------------------------------------------------------
def run_inner(args, names, guard, smoke: bool) -> int:
    ov = {"seeds": INNER_SEEDS}
    if smoke:
        ov["steps"] = SMOKE["steps"]
    for name in names:
        t_start = time.time()
        data = d92.load_records(name)
        by = data["by_tag"]
        op = outer_plan(name, data, smoke)
        tasks = []
        for o in op:
            f = o["fold"]
            for k, part in enumerate(o["inner"]):
                held = set(part)
                train = [by[t] for t in data["train"][f] if t not in held]
                scored = [by[t] for t in part]
                for c in MENU[1:]:
                    tasks.append((f"{c}|{f}|k{k}", c, train, scored, ov))
            # V_fold: lin on the full outer training set, full settings
            tasks.append((f"check_lin|{f}", "lin", [by[t] for t in data["train"][f]], [], {}))
        heavy = {"mlp16": 0, "mlp8": 1, "gam": 2}
        tasks.sort(key=lambda t: heavy.get(t[1], 9) if not t[0].startswith("check") else 9)
        print(f"I10_2 inner {name}: {len(op)} outer folds, {len(tasks)} tasks, overrides {ov}", flush=True)
        with get_context("spawn").Pool(args.workers) as pool:
            out = pool.map(inner_task, tasks, chunksize=1)
        corrs, logs, timing = {}, {}, {}
        for key, corr, lg, sec, npts in out:
            corrs[key] = corr
            logs[key] = lg
            timing[key] = {"seconds": sec, "n_points": npts}
        # V_fold
        stored = _json(I92 / name / "corrections.json")
        vfold = {}
        for o in op:
            f = o["fold"]
            st = correction_from_json_i9(stored[f"lin|{f}"])
            vfold[f] = float(np.max(np.abs(np.asarray(corrs.pop(f"check_lin|{f}").coef) - np.asarray(st.coef))))
        ok_start = start_ok(data["recs"])
        folds_out = {}
        for o in op:
            f = o["fold"]
            tags = [t for part in o["inner"] for t in part]
            cls_logs = {c: {} for c in MENU[1:]}
            for k, part in enumerate(o["inner"]):
                for c in MENU[1:]:
                    cls_logs[c].update(logs[f"{c}|{f}|k{k}"])
            scores, blocked = fold_scores(cls_logs, tags, ok_start)
            sel = select_class(scores, blocked)
            folds_out[f] = {"pos": o["pos"], "inner_folds": o["inner"], "scores": scores,
                            "blocked": blocked, "selected": sel,
                            "tie_sensitivity": {str(x): select_class(scores, blocked, tie=x)
                                                for x in TIE_SENSITIVITY},
                            "per_device_log_ratio": cls_logs}
        result = {"dataset": name, "smoke": smoke, "guard": guard, "menu": list(MENU),
                  "overrides": {k: list(v) if isinstance(v, tuple) else v for k, v in ov.items()},
                  "tie": TIE, "folds": folds_out,
                  "selection": {f: v["selected"] for f, v in folds_out.items()},
                  "V_fold_lin_max_abs_coef_diff": vfold,
                  "V_fold_pass": bool(max(vfold.values()) <= V_FOLD_TOL),
                  "start_rrms_finite": ok_start, "timing": timing,
                  "runtime_s": round(time.time() - t_start, 1)}
        target = OUT / ("smoke" if smoke else "inner") / name
        dump(target / "results.json", result)
        (target / "corrections.json").write_text(json.dumps({k: c.to_json() for k, c in corrs.items()}) + "\n")
        print(f"  V_fold max |coef diff| {max(vfold.values()):.2e}; selections "
              f"{selection_counts(result['selection'])}; runtime {result['runtime_s']} s", flush=True)
    return 0


# ---------------------------------------------------------------------------
# refit (S_refit)
# ---------------------------------------------------------------------------
def run_refit(args, name, guard, smoke: bool) -> int:
    t_start = time.time()
    src = OUT / ("smoke" if smoke else "inner") / name
    if not (src / "results.json").exists():
        raise SystemExit(f"{src}/results.json missing: run --mode inner first")
    inner = _json(src / "results.json")
    if not smoke and not inner["V_fold_pass"]:
        raise SystemExit(f"{name}: V_fold failed; dataset excluded")
    cj = _json(src / "corrections.json")
    data = d92.load_records(name)
    by = data["by_tag"]
    op = outer_plan(name, data, smoke)
    opts = dict(OPTS)
    if smoke:
        opts.update(heldout_nfev=SMOKE["heldout_nfev"], heldout_irls=SMOKE["heldout_irls"])
    need = sorted({t for o in op for k in o["refit_inner"] for t in o["inner"][k]})
    devs = d92.load_devices_for(name, need)
    tasks = []
    for o in op:
        f = o["fold"]
        for k in o["refit_inner"]:
            for c in MENU[1:]:
                corr = correction_from_json_i9(cj[f"{c}|{f}|k{k}"])
                for t in o["inner"][k]:
                    tasks.append((f"{c}|{f}|k{k}|{t}", (name, f"{c}|{f}|k{k}", by[t], *devs[t], corr, opts)))
    print(f"I10_2 refit {name}: {len(tasks)} refits, opts {opts}", flush=True)
    with get_context("spawn").Pool(args.workers) as pool:
        rows = dict(pool.map(refit_worker, tasks, chunksize=1))
    c0 = outer_rows(name)[CONTROL]
    folds_out = {}
    for o in op:
        f = o["fold"]
        tags = [t for k in o["refit_inner"] for t in o["inner"][k]]
        logs = {c: {} for c in MENU[1:]}
        for k in o["refit_inner"]:
            for c in MENU[1:]:
                for t in o["inner"][k]:
                    a, b = rows[f"{c}|{f}|k{k}|{t}"]["rrms"], c0.get(t, np.nan)
                    ok = np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0
                    logs[c][t] = float(math.log(a / b)) if ok else float("nan")
        ref_ok = {t: bool(np.isfinite(c0.get(t, np.nan))) for t in tags}
        scores, blocked = fold_scores(logs, tags, ref_ok)
        folds_out[f] = {"pos": o["pos"], "refit_inner": o["refit_inner"], "refit_tags": tags,
                        "scores": scores, "blocked": blocked, "selected": select_class(scores, blocked),
                        "tie_sensitivity": {str(x): select_class(scores, blocked, tie=x)
                                            for x in TIE_SENSITIVITY},
                        "per_device_log_ratio": logs}
    n_fail = sum(1 for r in rows.values() if not np.isfinite(r["rrms"]))
    result = {"dataset": name, "smoke": smoke, "guard": guard, "options": opts,
              "refit_scope": REFIT_SCOPE[name], "folds": folds_out,
              "selection": {f: v["selected"] for f, v in folds_out.items()},
              "n_refits": len(rows), "n_failed_refits": n_fail, "rows": rows,
              "runtime_s": round(time.time() - t_start, 1)}
    dump(OUT / ("smoke_refit" if smoke else "refit") / name / "results.json", result)
    print(f"  failed {n_fail}; selections {selection_counts(result['selection'])}; "
          f"runtime {result['runtime_s']} s", flush=True)
    return 0


# ---------------------------------------------------------------------------
# S3, stored selectors
# ---------------------------------------------------------------------------
def stored_s3(name: str) -> tuple[dict, dict]:
    if name == "nist_4K":
        p = _json(EX / "I8_1/prediction_v2.json")
        return {f: r["S3"] for f, r in p["per_fold"].items()}, p["S3_hyper"]
    if name == "nist_rt":
        p = _json(EX / "I9_1/prediction.json")
        return {f: r["S3"] for f, r in p["per_fold"].items()}, p["S3_hyper"]
    p = _json(EX / "I7_1/retro/s3.json")
    return dict(p["per_fold"][name]), p["hyper"]


def cv_select(rows: dict, menu, train: dict, folds: list, tie: float = TIE) -> tuple[dict, dict]:
    sel, info = {}, {}
    for f in folds:
        logs, ok = cv_logs(rows, menu, train[f])
        scores, blocked = fold_scores(logs, train[f], ok)
        sel[f] = select_class(scores, blocked, tie=tie)
        info[f] = {"scores": scores, "blocked": blocked}
    return sel, info


def nist_rt_data() -> dict:
    """I9-1 NIST room-temperature DEV: stored C0_refit / M_bias (lodo) and
    B3_d3 (baselines) rows; leave-one-device-out over the finite records."""
    lo = _json(EX / "I9_1/lodo/results.json")
    excluded = list(lo.get("excluded_records", []))
    if SYNTHETIC["on"]:
        rows = synthetic_rows(sorted(lo["per_device"]["C0_refit"]), excluded, (NONE, "poly3", "mlp16"))
        c0b = rows[CONTROL]
    else:
        ba = _json(EX / "I9_1/baselines/results.json")
        rows = {CONTROL: _nan(lo["per_device"]["C0_refit"]), "mlp16": _nan(lo["per_device"]["M_bias"]),
                "poly3": _nan(ba["per_device"]["B3_d3"])}
        c0b = _nan(ba["per_device"]["C0_refit"])
    tags = sorted(t for t in rows[CONTROL] if t not in set(excluded))
    return {"rows": rows, "excluded": excluded, "tags": tags,
            "fold": {t: t for t in tags}, "train": {f: [t for t in tags if t != f] for f in tags},
            "c0_lodo_equals_baselines": all((np.isnan(c0b[t]) and np.isnan(v)) or c0b[t] == v
                                            for t, v in rows[CONTROL].items())}


# ---------------------------------------------------------------------------
# Validity checks on the stored rows
# ---------------------------------------------------------------------------
def check_rows(name: str, data: dict, rows: dict) -> dict:
    """V_rows: every held-out device has a value (finite or NaN = failed) in
    every class row; the fold model of every class exists for every fold."""
    tags = list(data["by_tag"]) + list(data["excluded"])
    missing = {c: [t for t in tags if t not in rows.get(d92_row(c), {})] for c in FULL_MENU}
    missing = {c: v for c, v in missing.items() if v}
    trained = _json(I92 / name / "results.json")["trained_classes"]
    corr_keys = set(_json(I92 / name / "corrections.json"))
    models = d92.stored_paths(name)["models"]
    no_model = {}
    cache = {}
    for c in FULL_MENU:
        if c == NONE:
            continue
        for f in data["folds"]:
            if c in trained or c == "mlp16_s1":
                ok = f"{c}|{f}" in corr_keys
            else:
                path, prefix = models[c]
                if path not in cache:
                    cache[path] = set(_json(path))
                ok = f"{prefix}|{f}" in cache[path]
            if not ok:
                no_model.setdefault(c, []).append(f)
    excl_nan = all(np.isnan(rows[d92_row(c)][t]) for c in FULL_MENU for t in data["excluded"])
    return {"missing_rows": missing, "missing_fold_models": no_model,
            "excluded_records_nan_in_every_class": bool(excl_nan),
            "pass": not missing and not no_model and excl_nan}


def d92_row(c: str) -> str:
    return CONTROL if c == NONE else c


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def evaluate(series: dict, rows: dict, clusters: dict, a_adj: float) -> dict:
    """series: {selector: {dataset: {tag: rrms}}}; rows: {dataset: stored rows}.
    Primary contrasts at the Bonferroni level: per dataset and pooled GMR(S,
    C0_refit), pooled GMR(S, mlp16), pooled GMR(S, S_gate) (multi-class)."""
    names = sorted(next(iter(series.values())))
    out = {"vs_C0_refit": {}, "vs_mlp16": {}, "vs_S_gate": {}, "labels": {}}
    for s, ser in series.items():
        out["vs_C0_refit"][s] = gmr({n: (ser[n], rows[n][CONTROL]) for n in names}, clusters, a_adj)
        out["vs_mlp16"][s] = gmr({n: (ser[n], rows[n][REFERENCE]) for n in names}, clusters, a_adj)
        if s in MULTI_CLASS and "S_gate" in series:
            out["vs_S_gate"][s] = gmr({n: (ser[n], series["S_gate"][n]) for n in names}, clusters, a_adj)
        out["labels"][s] = selector_label(out["vs_C0_refit"][s]["per_dataset"], out["vs_mlp16"][s])
    return out


def build_series(selections: dict, rows: dict, datas: dict) -> dict:
    return {s: {n: assemble(sel[n], rows[n], datas[n]["fold"], datas[n]["excluded"])
                for n in sel} for s, sel in selections.items()}


def run_decide(args, guard, interim: bool) -> int:
    t0 = time.time()
    datas = {n: d92.load_records(n) for n in PRIMARY}
    rows = {n: outer_rows(n) for n in PRIMARY}
    clusters = {n: d["clusters"] for n, d in datas.items() if d["clusters"] is not None}
    a_adj = bonferroni_alpha()
    validity = {n: check_rows(n, datas[n], rows[n]) for n in PRIMARY}
    sel, info = {}, {}
    # S_gate
    sel["S_gate"], s3info = {}, {}
    for n in PRIMARY:
        s3, hyper = stored_s3(n)
        if sorted(s3) != sorted(datas[n]["folds"]):
            raise SystemExit(f"{n}: S3 fold keys differ from the I9-2 folds")
        sel["S_gate"][n] = {f: gate_select(s3[f]) for f in datas[n]["folds"]}
        s3info[n] = {"S3_hyper": hyper, "S3_f": s3}
    validity["V_s3"] = "S3 fold keys equal the I9-2 folds on every dataset"
    # S_cv
    sel["S_cv"] = {}
    for n in PRIMARY:
        sel["S_cv"][n], info.setdefault("S_cv", {})[n] = cv_select(rows[n], MENU, datas[n]["train"],
                                                                   datas[n]["folds"])
    # S_loss, S_refit
    missing = []
    for s, sub in (("S_loss", "inner"), ("S_refit", "refit")):
        got = {}
        for n in PRIMARY:
            p = OUT / sub / n / "results.json"
            if SYNTHETIC["on"]:         # random selections (decide-path smoke only)
                rng = np.random.default_rng(len(s) + len(n))
                got[n] = {f: str(rng.choice(MENU)) for f in datas[n]["folds"]}
                info.setdefault(s, {})[n] = {f: {"scores": {}, "blocked": {}, "tie_sensitivity": {
                    str(x): c for x in TIE_SENSITIVITY}} for f, c in got[n].items()}
                continue
            if not p.exists():
                missing.append(f"{s}:{n}")
                continue
            r = _json(p)
            if r.get("smoke"):
                raise SystemExit(f"{p} is a smoke output")
            if s == "S_loss" and not r["V_fold_pass"]:
                raise SystemExit(f"{n}: V_fold failed")
            if sorted(r["selection"]) != sorted(datas[n]["folds"]):
                raise SystemExit(f"{p}: folds differ from the I9-2 folds")
            got[n] = r["selection"]
            info.setdefault(s, {})[n] = {f: {"scores": v["scores"], "blocked": v["blocked"],
                                             "tie_sensitivity": v["tie_sensitivity"]}
                                         for f, v in r["folds"].items()}
        if len(got) == len(PRIMARY):
            sel[s] = got
    complete = not missing
    if not interim and not complete:
        raise SystemExit(f"missing selector outputs: {missing}")
    sel = {s: sel[s] for s in SELECTORS if s in sel}
    series = build_series(sel, rows, datas)
    prim = evaluate(series, rows, clusters, a_adj)
    ci95 = evaluate(series, rows, clusters, ALPHA)
    # references
    menu_cls = [c for c in MENU if c != NONE]
    pooled_cls = {c: gmr({n: (rows[n][c], rows[n][CONTROL]) for n in PRIMARY}, clusters) for c in menu_cls}
    per_cls = {n: {c: pooled_cls[c]["per_dataset"][n]["ratio"] for c in menu_cls} for n in PRIMARY}
    orc = oracle_classes(per_cls, pooled_cls)
    orc_series = {n: assemble({f: orc["per_dataset"][n] for f in datas[n]["folds"]}, rows[n],
                              datas[n]["fold"], datas[n]["excluded"]) for n in PRIMARY}
    refs = {}
    for s, ser in series.items():
        refs[s] = {"vs_oracle_pooled_class": gmr({n: (ser[n], rows[n][d92_row(orc["pooled"])]) for n in PRIMARY},
                                                 clusters),
                   "vs_oracle_per_dataset_class": gmr({n: (ser[n], orc_series[n]) for n in PRIMARY}, clusters)}
    # secondary
    sec = {"selection_counts": {s: {n: selection_counts(v[n], MENU) for n in v} for s, v in sel.items()},
           "selection_device_counts": {s: {n: selection_counts({t: v[n][f] for t, f in datas[n]["fold"].items()},
                                                               MENU) for n in v} for s, v in sel.items()},
           "agreement": {f"{a}|{b}": {n: agreement(sel[a][n], sel[b][n]) for n in PRIMARY}
                         for i, a in enumerate(sel) for b in list(sel)[i + 1:]},
           "device_bootstrap_tudelft": {s: gmr({n: (ser[n], rows[n][CONTROL]) for n in PRIMARY}, None, a_adj)
                                        for s, ser in series.items()},
           "fixed_classes_vs_C0_refit_95": pooled_cls, "oracle": orc}
    full_sel = {n: cv_select(rows[n], FULL_MENU, datas[n]["train"], datas[n]["folds"])[0] for n in PRIMARY}
    full_ser = {n: assemble(full_sel[n], rows[n], datas[n]["fold"], datas[n]["excluded"]) for n in PRIMARY}
    sec["S_cv_full_menu"] = {"selection_counts": {n: selection_counts(full_sel[n], FULL_MENU) for n in PRIMARY},
                             "vs_C0_refit": gmr({n: (full_ser[n], rows[n][CONTROL]) for n in PRIMARY}, clusters),
                             "vs_mlp16": gmr({n: (full_ser[n], rows[n][REFERENCE]) for n in PRIMARY}, clusters)}
    gate_ds = {}
    for n in PRIMARY:
        s3d = float(np.mean(list(s3info[n]["S3_f"].values())))
        gate_ds[n] = {f: gate_select(s3d) for f in datas[n]["folds"]}
    gser = {n: assemble(gate_ds[n], rows[n], datas[n]["fold"], datas[n]["excluded"]) for n in PRIMARY}
    sec["dataset_level_gate_I8_2"] = {"class": {n: gate_select(float(np.mean(list(s3info[n]["S3_f"].values()))))
                                                for n in PRIMARY},
                                      "vs_C0_refit": gmr({n: (gser[n], rows[n][CONTROL]) for n in PRIMARY}, clusters),
                                      "vs_mlp16": gmr({n: (gser[n], rows[n][REFERENCE]) for n in PRIMARY}, clusters)}
    tie_sens = {}
    for s in ("S_loss", "S_refit"):
        if s not in sel:
            continue
        for x in TIE_SENSITIVITY:
            sx = {n: {f: info[s][n][f]["tie_sensitivity"][str(x)] for f in datas[n]["folds"]} for n in PRIMARY}
            ser = {n: assemble(sx[n], rows[n], datas[n]["fold"], datas[n]["excluded"]) for n in PRIMARY}
            tie_sens.setdefault(s, {})[str(x)] = {
                "vs_C0_refit": gmr({n: (ser[n], rows[n][CONTROL]) for n in PRIMARY}, clusters, a_adj),
                "vs_mlp16": gmr({n: (ser[n], rows[n][REFERENCE]) for n in PRIMARY}, clusters, a_adj)}
    for x in TIE_SENSITIVITY:
        sx = {n: cv_select(rows[n], MENU, datas[n]["train"], datas[n]["folds"], tie=x)[0] for n in PRIMARY}
        ser = {n: assemble(sx[n], rows[n], datas[n]["fold"], datas[n]["excluded"]) for n in PRIMARY}
        tie_sens.setdefault("S_cv", {})[str(x)] = {
            "vs_C0_refit": gmr({n: (ser[n], rows[n][CONTROL]) for n in PRIMARY}, clusters, a_adj),
            "vs_mlp16": gmr({n: (ser[n], rows[n][REFERENCE]) for n in PRIMARY}, clusters, a_adj)}
    sec["tie_sensitivity"] = tie_sens
    # NIST room temperature (I9-1), secondary
    rt = nist_rt_data()
    s3rt, hyp = stored_s3("nist_rt")
    rt_sel = {"S_gate": {f: gate_select(s3rt[f]) for f in rt["tags"]},
              "S_cv": cv_select(rt["rows"], (NONE, "poly3", "mlp16"), rt["train"], rt["tags"])[0]}
    rt_out = {"c0_lodo_equals_baselines": rt["c0_lodo_equals_baselines"], "S3_hyper": hyp,
              "S3_fold_keys_match": sorted(s3rt) == sorted(rt["tags"]), "menu": [NONE, "poly3", "mlp16"]}
    for s, sl in rt_sel.items():
        ser = assemble(sl, rt["rows"], rt["fold"], rt["excluded"])
        rt_out[s] = {"selection_counts": selection_counts(sl, (NONE, "poly3", "mlp16")),
                     "vs_C0_refit": gmr({"nist_rt": (ser, rt["rows"][CONTROL])}),
                     "vs_mlp16": gmr({"nist_rt": (ser, rt["rows"]["mlp16"])})}
    sec["nist_rt_I9_1"] = rt_out
    sec["nist_rt_limitation"] = ("menu restricted to {none, poly3, mlp16}: no stored lin / gam / mlp8 rows; "
                                 "no new NIST room-temperature class runs in I10-2 (amendment A1)")
    a1 = amendment_c1(series, rows, clusters, a_adj)
    result = {"prereg": str(PREREG.relative_to(ROOT)), "guard": guard, "complete": complete,
              "missing": missing, "primary_datasets": list(PRIMARY), "menu": list(MENU),
              "alpha_adjusted": a_adj, "margin": MARGIN, "tie": TIE, "validity": validity,
              "selections": sel, "primary": prim, "ci95": ci95, "post_hoc_references": refs,
              "secondary": sec, "amendment_A1_secondary_C1": a1, "amendment_guard": AMEND_GUARD,
              "selector_scores": info, "S3": s3info,
              "runtime_s": round(time.time() - t0, 1)}
    if complete:
        result["rule"] = overall_decision(prim["labels"], prim["vs_C0_refit"], prim["vs_S_gate"])
    if SYNTHETIC["on"]:
        dump(OUT / "smoke" / "decide_synthetic.json", result)
        summary(result)
        return 0
    dump(OUT / ("decision.json" if complete and not interim else "interim.json"), result)
    if complete and not interim:
        write_md(result)
    summary(result)
    return 0


AMEND = OUT / "prereg_amendment_A1.json"
AMEND_GUARD: dict = {}
C1_VARIANTS = {   # amendment A1: secondary C1 reports (the primary rule is unchanged)
    "primary_cluster_bonferroni": {"tudelft_clusters": True, "alpha": "bonferroni"},
    "a_device_bootstrap_bonferroni": {"tudelft_clusters": False, "alpha": "bonferroni"},
    "b_cluster_95": {"tudelft_clusters": True, "alpha": 0.05},
    "ab_device_bootstrap_95": {"tudelft_clusters": False, "alpha": 0.05},
}


def require_amendment() -> dict:
    """Amendment A1 committed, unmodified (decide refuses otherwise)."""
    rel = str(AMEND.relative_to(ROOT))
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel],
                             capture_output=True).returncode == 0
    clean = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", rel]).returncode == 0
    if not (AMEND.exists() and tracked and clean):
        raise SystemExit(f"{rel} is not committed: amendment A1 is not in force")
    sha = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", rel],
                         capture_output=True, text=True).stdout.strip()
    return {"amendment": rel, "commit": sha}


def amendment_c1(series: dict, rows: dict, clusters: dict, a_adj: float) -> dict:
    """A1 (secondary): C1 per selector and for fixed mlp16 under the primary
    CI and the variants (a) TU Delft device bootstrap, (b) 95% per-dataset CIs,
    (a+b).  The primary label and rule are not changed."""
    names = sorted(next(iter(series.values())))
    subjects = dict(series)
    subjects["fixed_mlp16"] = {n: rows[n][REFERENCE] for n in names}
    out = {}
    for v, spec in C1_VARIANTS.items():
        alpha = a_adj if spec["alpha"] == "bonferroni" else spec["alpha"]
        cl = clusters if spec["tudelft_clusters"] else None
        out[v] = {}
        for s, ser in subjects.items():
            per = gmr({n: (ser[n], rows[n][CONTROL]) for n in names}, cl, alpha)["per_dataset"]
            fail = [n for n, d in per.items() if not (np.isfinite(d["ci_high"]) and d["ci_high"] <= MARGIN)]
            out[v][s] = {"C1": not fail, "failing_datasets": fail, "alpha": alpha,
                         "per_dataset": {n: {k: d[k] for k in ("ratio", "ci_low", "ci_high")}
                                         for n, d in per.items()}}
    return out


def summary(r: dict) -> None:
    p = r["primary"]
    for s in r["selections"]:
        c0, m = p["vs_C0_refit"][s], p["vs_mlp16"][s]
        per = " ".join(f"{n} {v['ratio']:.3f}[{v['ci_low']:.3f},{v['ci_high']:.3f}]"
                       for n, v in c0["per_dataset"].items())
        g = p["vs_S_gate"].get(s)
        gs = f" vs S_gate {g['ratio']:.3f}[{g['ci_low']:.3f},{g['ci_high']:.3f}]" if g else ""
        print(f"{s:8s} vs C0 {c0['ratio']:.3f}[{c0['ci_low']:.3f},{c0['ci_high']:.3f}] vs mlp16 "
              f"{m['ratio']:.3f}[{m['ci_low']:.3f},{m['ci_high']:.3f}]{gs} {p['labels'][s]['label']}")
        print(f"         per dataset vs C0: {per}")
        print(f"         counts {r['secondary']['selection_counts'][s]}")
    if "rule" in r:
        print("PRIMARY (in-force rule) rule", json.dumps(r["rule"]))
    a1 = r.get("amendment_A1_secondary_C1")
    if a1:
        print("SECONDARY (amendment A1, does not change the primary label or rule): C1 per variant")
        for v, d in a1.items():
            print(f"  {v}: " + "; ".join(f"{s} C1 {'pass' if x['C1'] else 'fail ' + ','.join(x['failing_datasets'])}"
                                       for s, x in d.items()))
            tud = {s: x["per_dataset"].get("tudelft_4K") for s, x in d.items()}
            print("    tudelft_4K: " + "; ".join(f"{s} {t['ratio']:.3f}[{t['ci_low']:.3f},{t['ci_high']:.3f}]"
                                                 for s, t in tud.items() if t))


def write_md(r: dict) -> None:
    p = r["primary"]
    lines = ["# I10-2 decision", "",
             f"Bonferroni level 1 - 0.05/11 (alpha {r['alpha_adjusted']:.5f}); margin {r['margin']}; "
             f"TU Delft cluster bootstrap (groups).", "",
             "| selector | GMR vs C0_refit (pooled) | GMR vs mlp16 (pooled) | GMR vs S_gate | label |",
             "|---|---|---|---|---|"]
    f = lambda v: f"{v['ratio']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]"
    for s in r["selections"]:
        g = p["vs_S_gate"].get(s)
        lines.append(f"| {s} | {f(p['vs_C0_refit'][s])} | {f(p['vs_mlp16'][s])} | "
                     f"{f(g) if g else '-'} | {p['labels'][s]['label']} |")
    lines += ["", "| selector | " + " | ".join(r["primary_datasets"]) + " |",
              "|---" * (len(r["primary_datasets"]) + 1) + "|"]
    for s in r["selections"]:
        lines.append(f"| {s} | " + " | ".join(f(p["vs_C0_refit"][s]["per_dataset"][n])
                                             for n in r["primary_datasets"]) + " |")
    lines += ["", f"Primary rule (in force): {json.dumps(r.get('rule'))}", ""]
    a1 = r.get("amendment_A1_secondary_C1") or {}
    if a1:
        subj = list(next(iter(a1.values())))
        lines += ["Secondary (amendment A1; the primary label and rule are unchanged): C1 per variant", "",
                  "| variant | " + " | ".join(subj) + " |", "|---" * (len(subj) + 1) + "|"]
        for v, d in a1.items():
            lines.append(f"| {v} | " + " | ".join(
                "pass" if d[s]["C1"] else "fail (" + ", ".join(d[s]["failing_datasets"]) + ")" for s in subj) + " |")
        lines.append("")
    (OUT / "decision.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# smoke statistics on synthetic rows
# ---------------------------------------------------------------------------
def smoke_stats() -> dict:
    """Statistics code path on synthetic rows (no stored held-out row read)."""
    rng = np.random.default_rng(1)
    datas, rows = {}, {}
    for n, nf in (("a", 6), ("b", 5)):
        tags = [f"{n}{i}" for i in range(nf * 2)]
        fold = {t: f"{n}f{i // 2}" for i, t in enumerate(tags)}
        folds = sorted(set(fold.values()))
        datas[n] = {"fold": fold, "folds": folds, "excluded": [],
                    "train": {f: [t for t in tags if fold[t] != f] for f in folds}}
        base = {t: float(rng.uniform(0.02, 0.1)) for t in tags}
        rows[n] = {CONTROL: base}
        for c in MENU[1:]:
            rows[n][c] = {t: base[t] * float(np.exp(rng.normal(-0.05, 0.1))) for t in tags}
    sel = {"S_gate": {n: {f: "mlp16" for f in datas[n]["folds"]} for n in datas},
           "S_cv": {n: cv_select(rows[n], MENU, datas[n]["train"], datas[n]["folds"])[0] for n in datas}}
    series = build_series(sel, rows, datas)
    prim = evaluate(series, rows, {"b": {t: datas["b"]["fold"][t] for t in datas["b"]["fold"]}},
                    bonferroni_alpha())
    return {"labels": prim["labels"],
            "rule": overall_decision(prim["labels"], prim["vs_C0_refit"], prim["vs_S_gate"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("plan", "smoke", "inner", "refit", "stored", "decide"), required=True)
    ap.add_argument("--dataset", choices=PRIMARY, default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.mode == "plan":
        return run_plan(args)
    if args.mode == "smoke":
        run_inner(args, ["sky130_77K"], None, True)
        run_refit(args, "sky130_77K", None, True)
        out = smoke_stats()
        dump(OUT / "smoke" / "smoke_stats.json", out)
        print(json.dumps(out))
        SYNTHETIC["on"] = True          # decide path on synthetic outer rows
        return run_decide(args, None, interim=True)
    guard = require_prereg()
    if args.mode == "inner":
        return run_inner(args, [args.dataset] if args.dataset else list(PRIMARY), guard, False)
    if args.mode == "refit":
        if not args.dataset:
            raise SystemExit("--dataset required")
        return run_refit(args, args.dataset, guard, False)
    AMEND_GUARD.update(require_amendment())
    return run_decide(args, guard, interim=(args.mode == "stored"))


if __name__ == "__main__":
    sys.exit(main())
