#!/usr/bin/env python3
"""I17-2: second-PDK replication of the selected g v2 recipe on measured IHP
SG13G2 MOSFETs (PSP103, I5-1 data), 300 K, 18 DEV DUTs, leave-one-DUT-out
within class.  Accuracy only; Part B not run.

Pre-registration: out/research/explore/I17_2/prereg.md (prereg.json).

Stages (``--stage``):

  pens    starting card of every DEV DUT on the 75 x 75 forward-quadrant grid
          (0 .. class VDD) -> pens/grid_<tag>.npz, pens/pens.json
  base    C0_refit and M_bias v1 (I5-1 recipe) LODO; V1 vs I5-1 rows -> base/
  lodo    --lam L: sym_e1_l{L} fold corrections (5 seeds) and held-out refits
          -> lodo_sym_e1_l{L}/
  inner   nested selection of lambda per outer fold -> inner/inner.json
  decide  validity, primary / secondary labels, tables -> decision.json

Run with ``python -I`` (the script puts ``src`` on ``sys.path`` itself):

  .venv/bin/python -I scripts/research/explore/i17_2_ihp.py --stage pens --workers 6
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research" / "explore"))

import i5_1_ihp_real as drv5  # noqa: E402  (I5-1 data, records, refit_task, train_task)
from cryoml.explore import i5_1_ihp_real as m5  # noqa: E402
from cryoml.explore import i12_1_symm as v2  # noqa: E402
from cryoml.explore import i17_2_ihp as m  # noqa: E402

EXP = ROOT / "out/research/explore"
OUT = EXP / "I17_2"
PREREG = OUT / "prereg.json"
I5 = EXP / "I5_1"
T = m.T_K
OPTS = dict(drv5.PREREG_OPTIONS)
UNIT_TESTS = ("tests.test_i17_2_ihp", "tests.test_i12_1_symm", "tests.test_i15_2_ablate")


def jload(p: Path):
    return json.loads(Path(p).read_text())


def jdump(obj, p: Path) -> None:
    m5.write_json(p, obj)


_CACHE: dict = {}


def devices() -> dict:
    if "dev" not in _CACHE:
        _CACHE["dev"] = {d.tag: d for d in drv5.dev_devices()}
    return _CACHE["dev"]


def records() -> list:
    """Starting records (I5-1, 300 K), load order."""
    if "recs" not in _CACHE:
        from cryoml.explore.e1_correction import load_devices
        recs = load_devices(I5 / "start" / f"records_{T}K")
        if sorted(r.tag for r in recs) != sorted(devices()):
            raise RuntimeError("starting records differ from DEV")
        _CACHE["recs"] = recs
    return _CACHE["recs"]


def by_tag() -> dict:
    return {r.tag: r for r in records()}


def load_order() -> list[str]:
    return [r.tag for r in records()]


def base_dir(smoke: bool) -> Path:
    return OUT / "smoke" if smoke else OUT


# ---------------------------------------------------------------------------
# penalty grids
# ---------------------------------------------------------------------------
def grid_task(task):
    tag, out_dir = task
    dev = devices()[tag]
    rec = by_tag()[tag]
    pm = dev.dev_type == "pmos"
    specs = m.grid_curve_specs(dev.vdd, pm)
    z = np.zeros(1)
    curves = [m5.MeasCurve("grid", "idvd", s["fixed"], s["vb"], np.asarray(s["sweep"], float), z, z, z, z, "")
              for s in specs]
    prob = m5.MeasProblem(dev, T, m5.base_box(dev))
    t0 = time.time()
    sims = m5.simulate(dev, prob.card, curves, prob.temp_C, dict(rec.meta["params"]))
    grid = np.vstack([np.asarray(x, float) for x in sims])          # (u_g, u_d)
    np.savez_compressed(Path(out_dir) / f"grid_{tag}.npz", i=grid, axis=m.pen_axis(dev.vdd))
    s = -1.0 if pm else 1.0
    return {"tag": tag, "vdd": dev.vdd, "shape": list(grid.shape), "n_nonfinite": int(np.sum(~np.isfinite(grid))),
            "i_fwd_corner": float(s * grid[-1, -1]), "seconds": round(time.time() - t0, 2)}


def stage_pens(workers: int, rd: Path) -> dict:
    rd.mkdir(parents=True, exist_ok=True)
    tags = sorted(devices())
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(grid_task, [(t, str(rd)) for t in tags], chunksize=1)
    out = {"rows": {r["tag"]: r for r in rows}, "n": len(rows)}
    pens = load_pens(rd)
    for t in tags:
        out["rows"][t]["n_kept"] = pens[t]["n_kept"]
    out["v4_ok"] = bool(len(rows) == 18 and all(r["n_nonfinite"] == 0 and r["shape"] == [m.PEN_N, m.PEN_N]
                                                 and r["i_fwd_corner"] > 0 for r in rows))
    out["card_negative_fraction"] = v2.card_negative_fraction([pens[t] for t in tags])
    jdump(out, rd / "pens.json")
    print(json.dumps({"v4_ok": out["v4_ok"], "kept": {t: pens[t]["n_kept"] for t in tags},
                      "card_neg": out["card_negative_fraction"]}, indent=1), flush=True)
    return out


def load_pens(rd: Path | None = None) -> dict:
    rd = rd or OUT / "pens"
    out = {}
    for tag, dev in devices().items():
        z = np.load(rd / f"grid_{tag}.npz")
        out[tag] = v2.penalty_arrays(z["i"], dev.dev_type == "pmos", step=m.pen_step(dev.vdd),
                                     grid=z["axis"])
    return out


# ---------------------------------------------------------------------------
# training / refit tasks
# ---------------------------------------------------------------------------
def train_sym_task(task):
    """(name, train_tags, seed, hyper, lam, pens_dir) -> (name, seed, corr, seconds)."""
    name, train_tags, seed, hyper, lam, pens_dir = task
    recs = by_tag()
    train = [recs[t] for t in train_tags]
    vdd = devices()[train_tags[0]].vdd
    if any(devices()[t].vdd != vdd for t in train_tags):
        raise RuntimeError("training DUTs of different supply")
    pens = None
    if lam:
        allp = load_pens(Path(pens_dir))
        pens = [allp[t] for t in train_tags]
    t0 = time.time()
    corr = v2.fit_v2(train, pens, lam, hyper={**hyper, "seeds": (seed,)}, inputs="sym_v2", eps=m.EPS,
                     threads=1, arm=name.split("|", 1)[0], vdd=vdd)
    return name, seed, corr, round(time.time() - t0, 1)


def refit_task(task):
    """(label, corr, tag, opts) -> I5-1 held-out refit row (kind 'corr')."""
    label, corr, tag, opts = task
    row = drv5.refit_task((label, "corr", {"corr": corr}, by_tag()[tag], devices()[tag], opts))
    return {**row, "label": label}


def merge_sym(results, names) -> tuple[dict, dict]:
    by = {n: [] for n in names}
    secs = {n: [] for n in names}
    for name, seed, corr, sec in sorted(results, key=lambda r: (r[0], r[1])):
        by[name].append(corr)
        secs[name].append(sec)
    return {n: v2.merge_seed_corrections(c) for n, c in by.items()}, secs


def run_unit_tests(rd: Path) -> dict:
    r = subprocess.run([sys.executable, "-m", "unittest", *UNIT_TESTS], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"})
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    out = {"files": list(UNIT_TESTS), "rc": r.returncode, "passed": r.returncode == 0, "tail": tail}
    jdump(out, rd / "unit_tests.json")
    print(f"unit tests: rc {r.returncode}; {tail[-1] if tail else ''}", flush=True)
    return out


# ---------------------------------------------------------------------------
# base: C0_refit and M_bias v1 (I5-1 recipe), V1
# ---------------------------------------------------------------------------
def stage_base(workers: int, rd: Path, smoke: bool) -> dict:
    from cryoml.explore.e1_correction import Correction
    from cryoml.explore.i3_2_bias import merge_seed_results, split_seed_tasks
    order = load_order()
    folds = list(m.SMOKE_FOLDS) if smoke else sorted(order)
    hyper = {"seeds": (0,), "steps": 200} if smoke else {}
    opts = dict(OPTS, heldout_nfev=20) if smoke else dict(OPTS)
    recs = by_tag()
    unit = run_unit_tests(rd)
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        bias_tasks = [("M_bias", [recs[t] for t in m.fold_train(order, h)]) for h in folds]
        expanded, owner = split_seed_tasks(bias_tasks, hyper)
        results = pool.map(drv5.train_task, [("bias", a, r, hp) for a, r, hp in expanded], chunksize=1)
        corrs = {f"M_bias|{h}": c for h, c in zip(folds, merge_seed_results(len(bias_tasks), owner, results))}
        t_train = time.time() - t0
        tasks = [("C0_refit", Correction("C0"), h, opts) for h in folds]
        tasks += [("M_bias", corrs[f"M_bias|{h}"], h, opts) for h in folds]
        t1 = time.time()
        rows = pool.map(refit_task, tasks, chunksize=1)
        t_refit = time.time() - t1
    per = {"C0_refit": {}, "M_bias": {}}
    for r in rows:
        per[r["label"]][r["tag"]] = r["rrms"]
    stored = jload(I5 / "lodo_300K/results.json")["per_device"]
    v1 = {a: {t: abs(per[a][t] - stored[a][t]) for t in per[a]} for a in per}
    out = {"smoke": smoke, "folds": folds, "opts": opts, "hyper_overrides": hyper, "per_device": per,
           "train_order": {h: m.fold_train(order, h) for h in folds},
           "v1": {"c0_max": max(v1["C0_refit"].values()), "mb_max": max(v1["M_bias"].values()), "n": len(folds),
                  "per_device": v1},
           "unit_tests_passed": bool(unit["passed"]),
           "rows": {f"{r['label']}|{r['tag']}": {k: v for k, v in r.items() if k != "params"} for r in rows},
           "timing": {"train_s": round(t_train, 1), "refit_s": round(t_refit, 1)}}
    jdump(out, rd / "results.json")
    jdump({k: c.to_json() for k, c in corrs.items()}, rd / "corrections.json")
    jdump({f"{r['label']}|{r['tag']}": r["params"] for r in rows}, rd / "heldout_cards.json")
    print(json.dumps({"v1": {k: out["v1"][k] for k in ("c0_max", "mb_max", "n")}, "timing": out["timing"],
                      "mean": {a: float(np.mean(list(v.values()))) for a, v in per.items()}}, indent=1),
          flush=True)
    return out


# ---------------------------------------------------------------------------
# lodo (one lambda)
# ---------------------------------------------------------------------------
def stage_lodo(lam: float, workers: int, rd: Path, smoke: bool) -> dict:
    name = m.variant(lam)
    order = load_order()
    folds = list(m.SMOKE_FOLDS) if smoke else sorted(order)
    hyper = {"steps": 200} if smoke else {}
    seeds = (0,) if smoke else v2.HYPER_V2["seeds"]
    opts = dict(OPTS, heldout_nfev=20) if smoke else dict(OPTS)
    pens_dir = str(base_dir(smoke) / "pens")
    names = [f"{name}|{h}" for h in folds]
    train_of = {f"{name}|{h}": m.fold_train(order, h) for h in folds}
    tasks = [(n, train_of[n], s, hyper, float(lam), pens_dir) for n in names for s in seeds]
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        trained = pool.map(train_sym_task, tasks, chunksize=1)
        t_train = time.time() - t0
        corrs, secs = merge_sym(trained, names)
        print(f"lodo {name}: training {len(tasks)} seed tasks {t_train:.0f} s", flush=True)
        t1 = time.time()
        rows = pool.map(refit_task, [(name, corrs[f"{name}|{h}"], h, opts) for h in folds], chunksize=1)
        t_refit = time.time() - t1
    per = {r["tag"]: r["rrms"] for r in rows}
    res = {"variant": name, "inputs": "sym_v2", "eps": m.EPS, "lambda": float(lam), "smoke": smoke,
           "vdd": {h: devices()[h].vdd for h in folds}, "hyper": v2.HYPER_V2, "hyper_overrides": hyper,
           "seeds": list(seeds), "opts": opts, "folds": folds, "train_order": train_of, "per_device": per,
           "mean": float(np.mean(list(per.values()))),
           "heldout_rows": {r["tag"]: {k: v for k, v in r.items() if k != "params"} for r in rows},
           "train_seconds_per_member": secs,
           "timing": {"train_s": round(t_train, 1), "refit_s": round(t_refit, 1), "n_seed_tasks": len(tasks)}}
    jdump(res, rd / "results.json")
    jdump({n: c.to_json() for n, c in corrs.items()}, rd / "corrections.json")
    jdump({r["tag"]: r["params"] for r in rows}, rd / "heldout_cards.json")
    print(json.dumps({"variant": name, "mean": res["mean"], "timing": res["timing"]}, indent=1), flush=True)
    return res


# ---------------------------------------------------------------------------
# inner (nested selection of lambda)
# ---------------------------------------------------------------------------
def stage_inner(workers: int, rd: Path, smoke: bool) -> dict:
    order = load_order()
    outers = list(m.SMOKE_FOLDS) if smoke else sorted(order)
    hyper = {"steps": 200} if smoke else {}
    opts = dict(OPTS, heldout_nfev=20) if smoke else dict(OPTS)
    pens_dir = str(base_dir(smoke) / "pens")
    pm = m.is_pmos_map(order)
    plan, tasks = {}, []
    for o in outers:
        tr = m.fold_train(order, o)
        folds = m.inner_folds(tr, pm)
        plan[o] = {"train": tr, "inner_folds": folds}
        for j, held in enumerate(folds):
            for lam in m.GRID_LAMBDA:
                tasks.append((f"{o}|{j}|{lam:g}", m.inner_train(tr, held), 0, hyper, float(lam), pens_dir))
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        trained = pool.map(train_sym_task, tasks, chunksize=1)
        t_train = time.time() - t0
        print(f"inner: training {len(tasks)} members {t_train:.0f} s", flush=True)
        corr = {name: c for name, _, c, _ in trained}
        secs = {name: s for name, _, _, s in trained}
        rtasks = []
        for o in outers:
            for j, held in enumerate(plan[o]["inner_folds"]):
                for lam in m.GRID_LAMBDA:
                    key = f"{o}|{j}|{lam:g}"
                    rtasks += [(key, corr[key], t, opts) for t in held]
        t1 = time.time()
        rows = pool.map(refit_task, rtasks, chunksize=1)
        t_refit = time.time() - t1
    res = {"smoke": smoke, "grid_lambda": list(m.GRID_LAMBDA), "eps": m.EPS, "seed": 0, "opts": opts,
           "hyper_overrides": hyper, "plan": plan, "outer": {},
           "timing": {"train_s": round(t_train, 1), "refit_s": round(t_refit, 1), "n_train": len(tasks),
                      "n_refit": len(rtasks)}, "train_seconds": secs}
    for o in outers:
        rr = {}
        for lam in m.GRID_LAMBDA:
            rr[f"{lam:g}"] = {r["tag"]: r["rrms"] for r in rows
                              if r["label"].startswith(f"{o}|") and r["label"].endswith(f"|{lam:g}")}
        scores = {lam: m.inner_score(list(rr[f"{lam:g}"].values())) for lam in m.GRID_LAMBDA}
        res["outer"][o] = {"inner_rrms": rr, "scores": {f"{k:g}": v for k, v in scores.items()},
                           "n_scored": {f"{lam:g}": len(rr[f"{lam:g}"]) for lam in m.GRID_LAMBDA},
                           "lambda_star": m.select_lambda(scores)}
    res["lambda_star"] = {o: res["outer"][o]["lambda_star"] for o in outers}
    res["complete"] = bool(len(tasks) == len(outers) * 3 * len(m.GRID_LAMBDA)
                           and all(n == len(plan[o]["train"]) for o in outers
                                   for n in res["outer"][o]["n_scored"].values()))
    jdump(res, rd / "inner.json")
    with gzip.open(rd / "inner_corrections.json.gz", "wt") as fh:
        json.dump({k: c.to_json() for k, c in corr.items()}, fh)
    print(json.dumps({"lambda_star": res["lambda_star"], "complete": res["complete"], "timing": res["timing"]},
                     indent=1), flush=True)
    return res


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------
def agg(values: dict) -> dict:
    return m5.aggregate(values)


def sym_corr(lam: float, tag: str, B: Path = OUT):
    js = jload(B / f"lodo_{m.variant(lam)}" / "corrections.json")
    return v2.SymCorrection.from_json(js[f"{m.variant(lam)}|{tag}"])


def neg_fractions(corr_of, tags, pens, order, vdd_v1=None) -> dict:
    tr_gm, tr_gds, ho_gm, ho_gds = [], [], [], []
    for t in tags:
        c = corr_of(t)
        kw = {"vdd_v1": devices()[t].vdd} if vdd_v1 else {}
        a = v2.penalty_value_numpy(c, [pens[u] for u in m.fold_train(order, t)], **kw)
        b = v2.penalty_value_numpy(c, [pens[t]], **kw)
        tr_gm.append(a["frac_neg_gm"])
        tr_gds.append(a["frac_neg_gds"])
        ho_gm.append(b["frac_neg_gm"])
        ho_gds.append(b["frac_neg_gds"])
    return {"train_mean_gm": float(np.mean(tr_gm)), "train_mean_gds": float(np.mean(tr_gds)),
            "heldout_mean_gm": float(np.mean(ho_gm)), "heldout_mean_gds": float(np.mean(ho_gds))}


def stage_decide(smoke: bool = False) -> dict:
    from cryoml.explore.i3_2_bias import correction_from_json, paired_ratio
    B = base_dir(smoke)
    order = load_order()
    base = jload(B / "base/results.json")
    tags = sorted(base["folds"])
    inner = jload(B / "inner/inner.json")
    pj = jload(B / "pens/pens.json")
    rows = {"C0_refit": base["per_device"]["C0_refit"], "M_bias_v1": base["per_device"]["M_bias"]}
    for lam in m.GRID_LAMBDA:
        f = B / f"lodo_{m.variant(lam)}/results.json"
        rows[m.variant(lam)] = jload(f)["per_device"] if f.exists() else {t: float("nan") for t in tags}
    lam_star = {t: float(inner["lambda_star"][t]) for t in tags}
    rows["selected"] = {t: rows[m.variant(lam_star[t])][t] for t in tags}
    validity = m.decide_validity({"v1_c0_max": base["v1"]["c0_max"], "v1_mb_max": base["v1"]["mb_max"],
                                  "v1_n": base["v1"]["n"], "v2_unit_tests_pass": base["unit_tests_passed"],
                                  "v3_rows": {k: len(v) for k, v in rows.items()},
                                  "v3_inner_complete": bool(inner["complete"] and len(inner["lambda_star"]) == 18),
                                  "v4_grids_ok": bool(pj["v4_ok"])})
    cl = m.clusters(tags)
    n_failed = int(sum(not np.isfinite(rows[a][t]) for a in ("selected", "C0_refit") for t in tags))
    primary = paired_ratio(rows["selected"], rows["C0_refit"])
    secondary = paired_ratio(rows["selected"], rows["M_bias_v1"])
    table = {}
    for a, r in rows.items():
        table[a] = {**agg(r)}
        if a != "C0_refit":
            table[a]["gmr_vs_C0_refit"] = paired_ratio(r, rows["C0_refit"])
            table[a]["gmr_vs_C0_refit_cluster"] = paired_ratio(r, rows["C0_refit"], clusters=cl)
        if a != "M_bias_v1":
            table[a]["gmr_vs_v1"] = paired_ratio(r, rows["M_bias_v1"])
            table[a]["gmr_vs_v1_cluster"] = paired_ratio(r, rows["M_bias_v1"], clusters=cl)
    by_class = {}
    for c in ("lv", "hv"):
        sel = [t for t in tags if t.startswith(c)]
        sub = lambda d: {t: d[t] for t in sel}  # noqa: E731
        by_class[c] = {"n": len(sel), "agg": {a: agg(sub(r)) for a, r in rows.items()},
                       "selected_vs_C0_refit": paired_ratio(sub(rows["selected"]), sub(rows["C0_refit"])),
                       "selected_vs_v1": paired_ratio(sub(rows["selected"]), sub(rows["M_bias_v1"])),
                       "v1_vs_C0_refit": paired_ratio(sub(rows["M_bias_v1"]), sub(rows["C0_refit"]))}
    pens = load_pens(B / "pens")
    bc = jload(B / "base/corrections.json")
    negs = {"M_bias_v1": neg_fractions(lambda t: correction_from_json(bc[f"M_bias|{t}"]), tags, pens, order,
                                       vdd_v1=True)}
    for lam in m.GRID_LAMBDA:
        if (B / f"lodo_{m.variant(lam)}/corrections.json").exists():
            negs[m.variant(lam)] = neg_fractions(lambda t, lam=lam: sym_corr(lam, t, B), tags, pens, order)
    if all((B / f"lodo_{m.variant(lam_star[t])}/corrections.json").exists() for t in tags):
        negs["selected"] = neg_fractions(lambda t: sym_corr(lam_star[t], t, B), tags, pens, order)
    negs["card_only"] = pj["card_negative_fraction"]
    stored = jload(I5 / "lodo_300K/results.json")
    preds = jload(PREREG)["predictions"]
    g_v1_c0 = paired_ratio(rows["M_bias_v1"], rows["C0_refit"])
    pred_check = {"ln GMR(selected, C0_refit)": m.sign_check(primary["ratio"], preds["ln GMR(selected, C0_refit)"]),
                  "ln GMR(selected, M_bias v1)": m.sign_check(secondary["ratio"],
                                                              preds["ln GMR(selected, M_bias v1)"]),
                  "ln GMR(M_bias v1, C0_refit)": m.sign_check(g_v1_c0["ratio"],
                                                              preds["ln GMR(M_bias v1, C0_refit)"])}
    cnt = {}
    for t in tags:
        cnt[f"{lam_star[t]:g}"] = cnt.get(f"{lam_star[t]:g}", 0) + 1
    out = {"prereg": str(PREREG.relative_to(ROOT)), "validity": validity,
           **m.decide_outcome(validity, primary, n_failed, secondary),
           "primary_gmr_selected_vs_C0_refit": primary, "secondary_gmr_selected_vs_v1": secondary,
           "gmr_v1_vs_C0_refit": g_v1_c0, "n_failed_selected_C0": n_failed,
           "predictions": preds, "prediction_signs_correct": pred_check,
           "lambda_star": inner["lambda_star"], "lambda_star_counts": cnt, "selected_rows": rows["selected"],
           "rows": rows, "table": table, "by_class": by_class, "clusters": cl, "negative_fractions": negs,
           "inner_scores": {o: v["scores"] for o, v in inner["outer"].items()},
           "I5_1_stored": {"C0_refit": stored["summary"]["all"]["C0_refit"],
                           "M_bias": stored["summary"]["all"]["M_bias"],
                           "M_bias_vs_C0_refit": stored["comparisons"]["all"]["M_bias_vs_C0_refit"]},
           "part_B": "not run (prereg: PSP103 replica, unconditional intrinsic charge, OSDI 0.4, no IHP circuits)"}
    jdump(out, B / "decision.json")
    print(json.dumps({"validity": validity, "outcome": out["outcome"], "secondary": out.get("secondary"),
                      "primary": {k: primary[k] for k in ("ratio", "ci_low", "ci_high", "wins")},
                      "secondary_gmr": {k: secondary[k] for k in ("ratio", "ci_low", "ci_high", "wins")},
                      "lambda_star_counts": cnt, "pred": pred_check,
                      "table": {a: (round(v["all_device_mean"], 5), round(v["median"], 5)) for a, v in table.items()}},
                     indent=1, default=float), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("pens", "base", "lodo", "inner", "decide"))
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true",
                    help="2 outer folds, 200 steps, 1 seed, refit max_nfev 20 (code check; not used for decisions)")
    args = ap.parse_args()
    base = base_dir(args.smoke)
    t0 = time.time()
    if args.stage == "pens":
        stage_pens(args.workers, base / "pens")
    elif args.stage == "base":
        stage_base(args.workers, base / "base", args.smoke)
    elif args.stage == "lodo":
        stage_lodo(args.lam, args.workers, base / f"lodo_{m.variant(args.lam)}", args.smoke)
    elif args.stage == "inner":
        stage_inner(args.workers, base / "inner", args.smoke)
    else:
        stage_decide(args.smoke)
    print(f"stage {args.stage} done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
