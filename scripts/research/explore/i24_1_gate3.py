#!/usr/bin/env python3
"""I24-1: three-way dataset-level gate {C0_refit, selected g v2, B1}; selected
g v2 on top of B1 on IHP SG13G2.

Pre-registration: out/research/explore/I24_1/prereg.md (prereg.json).

Stages (``--stage``):

  logo   --ds D        Part A: leave-one-group-out B1 screen selections, B1
                       refits of the records whose selection differs from the
                       I22-1 selection, VA2 / VA3, S_B1 -> partA/logo_<D>.json
  vb7                  Part B refit-path check at 300 K (2 DUTs) -> partB/vb7.json
  b      --temp T --sub records|pens|base|inner|sel|decide
                       Part B on IHP <T> K -> partB/ihp_<T>K/
  decide               Part A / C / B decision -> decision.json

Smoke: CRYOML_I24_1_SMOKE=1 (reduced budgets, output under I24_1/smoke).

  .venv/bin/python -I scripts/research/explore/i24_1_gate3.py --stage logo --ds sky130 --workers 10
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "out/research/explore"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(ROOT / "scripts" / "research" / "explore"))

import i22_1_baselines as d22  # noqa: E402  (dataset adapters, refit, stored rows; TU Delft state C)
import i17_2_ihp as d17  # noqa: E402  (I17-2 driver; rebound for Part B)
from cryoml.explore import i5_1_ihp_real as m5  # noqa: E402
from cryoml.explore import i12_1_symm as v2  # noqa: E402
from cryoml.explore import i17_2_ihp as m17  # noqa: E402
from cryoml.explore import i22_1_baselines as m22  # noqa: E402
from cryoml.explore import i24_1_gate3 as m  # noqa: E402

OUT = EXP / "I24_1"
PREREG = OUT / "prereg.json"
SMOKE = {"on": os.environ.get("CRYOML_I24_1_SMOKE") == "1"}
OPTS = {"heldout_nfev": 400, "heldout_irls": 2, "diff_step": 0.01}
UNIT_TESTS = ("tests.test_i24_1_gate3", "tests.test_i22_1_baselines", "tests.test_i17_2_ihp",
              "tests.test_i12_1_symm")
IHP_TEMPS = (233, 300, 343, 398)
VB7_TAGS = ("lv_nmos_W0p15_L0p18", "hv_pmos_W0p6_L0p5")


def jload(p):
    return json.loads(Path(p).read_text())


def jdump(obj, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, default=_default) + "\n")


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(type(o))


def base_dir() -> Path:
    return OUT / "smoke" if SMOKE["on"] else OUT


def opts() -> dict:
    return dict(OPTS, heldout_nfev=20, heldout_irls=1) if SMOKE["on"] else dict(OPTS)


def run_unit_tests(rd: Path) -> dict:
    r = subprocess.run([sys.executable, "-m", "unittest", *UNIT_TESTS], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"})
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    out = {"files": list(UNIT_TESTS), "rc": r.returncode, "passed": r.returncode == 0, "tail": tail}
    jdump(out, rd / "unit_tests.json")
    print(f"unit tests: rc {r.returncode}; {tail[-1] if tail else ''}", flush=True)
    return out


# ===========================================================================
# Part A
# ===========================================================================
def ihp_screen() -> dict:
    return jload(EXP / "I5_1/start/b1_screen.json")


def screen_info(ds: str) -> dict:
    """per_device screen entries, full selection per tag, NIST values."""
    if d22.kind(ds) == "ihp":
        st = ihp_screen()
        per = {**st["lv"]["per_device"], **st["hv"]["per_device"]}
        full = {t: list(st[m17.class_of(t)]["selected"]) for t in per}
        return {"per_device": per, "full": full, "values": None, "by_class": {c: st[c]["selected"] for c in ("lv", "hv")}}
    s = jload(EXP / "I22_1" / f"screen_{ds}.json")
    return {"per_device": s["per_device"], "full": {t: list(s["selected"]) for t in s["per_device"]},
            "values": s.get("values"), "selected": s["selected"]}


def group_fn(ds: str):
    if d22.kind(ds) == "tud":
        return d22.m6.group_of
    return lambda t: t


def train_of(ds: str, order: list[str], k: str) -> list[str]:
    g = group_fn(ds)
    if d22.kind(ds) == "ihp":
        return m17.fold_train(order, k)
    return [t for t in order if g(t) != k]


def va2(ds: str, info: dict) -> dict:
    if d22.kind(ds) == "ihp":
        st = ihp_screen()
        res = {}
        for c in ("lv", "hv"):
            sel = m22.screen_select(st[c]["per_device"])["selected"]
            res[c] = sel == list(st[c]["selected"])
        return {"ok": all(res.values()), "per_class": res}
    sel = m22.screen_select(info["per_device"])["selected"]
    return {"ok": sel == list(info["selected"]), "n_selected": len(sel)}


def b1_payload(ds: str, tag: str, names: list[str], info: dict) -> dict:
    if d22.kind(ds) == "nist":
        return {"names": list(names), "values": info["values"][tag]}
    return {"names": list(names)}


def stage_logo(ds: str, workers: int) -> dict:
    t0 = time.time()
    rd = base_dir() / "partA"
    o = opts()
    order = d22.order(ds)
    info = screen_info(ds)
    g = group_fn(ds)
    groups = sorted({g(t) for t in order})
    A = jload(EXP / f"I22_1/partA_{ds}/results.json")
    b1_rows = {t: float(A["per_device"]["B1"][t]) for t in order}
    c0 = {t: float(d22.stored_rows(ds)["C0_refit"][t]) for t in order}
    sel_by_group, plan = {}, []
    per_rec = {}
    for k in groups:
        tr = train_of(ds, order, k)
        sel = m.logo_selection(info["per_device"], tr)
        sel_by_group[k] = sel
        for t in [x for x in order if g(x) == k]:
            full = info["full"][t]
            if sel == full:
                src = "I22-1 row (equal selection)"
            elif not sel:
                src = "empty selection (log ratio 0)"
            else:
                src = "new refit"
                plan.append((ds, "B1_logo", "b1", b1_payload(ds, t, sel, info), t, o))
            per_rec[t] = {"group": k, "n_sel": len(sel), "n_full": len(full), "equal": sel == full, "source": src,
                          "dropped": sorted(set(full) - set(sel)), "added": sorted(set(sel) - set(full))}
    va3_tags = []
    if plan:
        va3_tags = [t for t in order if per_rec[t]["equal"]][:2]
        plan += [(ds, "B1_va3", "b1", b1_payload(ds, t, info["full"][t], info), t, o) for t in va3_tags]
    if SMOKE["on"]:
        plan = [p for p in plan if p[1] == "B1_logo"][:2] + [p for p in plan if p[1] == "B1_va3"][:1]
    unit = run_unit_tests(rd) if not SMOKE["on"] else {"passed": None}
    rows = []
    if plan:
        with get_context("spawn").Pool(min(workers, len(plan))) as pool:
            rows = pool.map(d22.refit_task, plan, chunksize=1)
    new = {r["tag"]: r for r in rows if r["label"] == "B1_logo"}
    v3 = {r["tag"]: abs(float(r["rrms"]) - b1_rows[r["tag"]]) for r in rows if r["label"] == "B1_va3"}
    logo_rows, lr = {}, {}
    for t in order:
        p = per_rec[t]
        if p["source"].startswith("I22-1"):
            logo_rows[t] = b1_rows[t]
        elif p["source"].startswith("empty"):
            logo_rows[t] = c0[t]
        else:
            logo_rows[t] = float(new[t]["rrms"]) if t in new else float("nan")
        lr[t] = 0.0 if p["source"].startswith("empty") else m.log_ratio(logo_rows[t], c0[t])
        p.update(b1_logo_rrms=logo_rows[t], b1_full_rrms=b1_rows[t], c0_rrms=c0[t], log_ratio=lr[t],
                 log_ratio_full=m.log_ratio(b1_rows[t], c0[t]))
    s = m.s_b1(lr)
    s_full = m.s_b1({t: m.log_ratio(b1_rows[t], c0[t]) for t in order})
    out = {"ds": ds, "smoke": SMOKE["on"], "opts": o, "n_records": len(order), "n_groups": len(groups),
           "n_new_refits": sum(1 for p in per_rec.values() if p["source"] == "new refit"),
           "n_equal": sum(1 for p in per_rec.values() if p["equal"]),
           "n_empty": sum(1 for p in per_rec.values() if p["source"].startswith("empty")),
           "frac_equal": float(np.mean([p["equal"] for p in per_rec.values()])),
           **{k: v for k, v in s.items()}, "S_B1_full_screen": s_full["S_B1"], "S_B1_full_valid": s_full["valid"],
           "VA2": va2(ds, info),
           "VA3": {"tags": va3_tags, "max_abs_diff": max(v3.values()) if v3 else None,
                   "ok": (not plan) or bool(v3 and max(v3.values()) <= m.VA3_TOL), "per_tag": v3,
                   "required": bool(plan)},
           "VA4": {"complete": all(np.isfinite(logo_rows[t]) or t in new for t in order),
                   "n_nonfinite": s["n_nonfinite"], "ok": bool(s["valid"])},
           "VA5_dev_check": d22.dev_check(ds, order), "unit_tests_passed": unit["passed"],
           "selection_by_group": sel_by_group, "per_record": per_rec,
           "new_rows": {f"{r['label']}|{r['tag']}": {k: v for k, v in r.items() if k != "params"} for r in rows},
           "runtime_s": round(time.time() - t0, 1), "workers": workers}
    jdump(out, rd / f"logo_{ds}.json")
    if rows:
        jdump({f"{r['label']}|{r['tag']}": r.get("params", {}) for r in rows}, rd / f"logo_{ds}_cards.json")
    print(json.dumps({k: out[k] for k in ("ds", "n_records", "n_equal", "n_new_refits", "n_empty", "S_B1",
                                          "S_B1_full_screen", "VA2", "runtime_s")}, default=_default), flush=True)
    print(json.dumps({"VA3": out["VA3"], "VA4": out["VA4"]}, default=_default), flush=True)
    return out


# ===========================================================================
# Part B (IHP; the I17-2 driver rebound to B1 records)
# ===========================================================================
_T_ENV = os.environ.get("CRYOML_I24_1_IHP_T")
_B: dict = {}


def b_dir(t: int | None = None) -> Path:
    t = d17.T if t is None else t
    return base_dir() / "partB" / f"ihp_{int(t)}K"


def class_sel(cls: str) -> list[str]:
    if "sel" not in _B:
        st = ihp_screen()
        _B["sel"] = {c: list(st[c]["selected"]) for c in ("lv", "hv")}
    return _B["sel"][cls]


def b1_box(dev):
    from cryoml.explore.i4_1_baselines import extend_box
    box = m5.base_box(dev)
    extras = m5.b1_extras(dev, class_sel(dev.cls))
    return box, extras, extend_box(box, extras)


def b1_records() -> list:
    if "recs" not in d17._CACHE:
        from cryoml.explore.e1_correction import load_devices
        recs = load_devices(b_dir() / "records" / f"records_{d17.T}K")
        if sorted(r.tag for r in recs) != sorted(d17.devices()):
            raise RuntimeError("B1 records differ from DEV")
        d17._CACHE["recs"] = recs
    return d17._CACHE["recs"]


def b1box_refit(label, corr, rec, dev, o) -> dict:
    """Held-out refit on the B1 box from the record's card, g in the loop
    (I5-1 ``refit_task`` 'corr' path with the B1 box)."""
    from cryoml.explore.e1_correction import score_device
    from cryoml.explore.i4_1_baselines import matched_nfev
    from cryoml.ngspice_fit import polish
    t0 = time.time()
    t = int(rec.meta["T_K"])
    start = dict(rec.meta["params"])
    box, extras, box1 = b1_box(dev)
    prob = m5.MeasProblem(dev, t, box1)
    nfev = matched_nfev(box.n_active, box1.n_active, o["heldout_nfev"], o["heldout_irls"])
    local0 = box1.params_to_local({**{n: e["value"] for n, e in extras.items()}, **start})
    try:
        res = polish(m5.corrected_problem(prob, corr), local0, max_nfev=nfev, irls_rounds=o["heldout_irls"],
                     diff_step=o["diff_step"])
        params = box1.local_to_params(res["local"])
        new = m5.to_record(dev, t, prob.simulate_params(params), params)
        after = score_device(new, corr.apply(new))
        row = {"rrms": after["rrms"], "failed": after["failed"], "history": res["history"], "params": params,
               "rrms_start": float(score_device(rec, corr.apply(rec))["rrms"])}
    except Exception as e:                                   # a failed refit is a failed row
        row = {"rrms": float("nan"), "failed": ["exception"], "error": f"{type(e).__name__}: {e}", "params": {}}
    row.update(label=label, tag=rec.tag, T_K=t, n_sim=prob.n_sim, n_active=int(box1.n_active), max_nfev=nfev,
               runtime_s=round(time.time() - t0, 2))
    return row


def b1_refit_task(task):
    """d17.refit_task replacement: (label, corr, tag, opts) on the B1 records."""
    label, corr, tag, o = task
    return b1box_refit(label, corr, d17.by_tag()[tag], d17.devices()[tag], o)


def b1_grid_task(task):
    """d17.grid_task replacement: the B1 card on the 75 x 75 grid."""
    tag, out_dir = task
    dev = d17.devices()[tag]
    rec = d17.by_tag()[tag]
    pm = dev.dev_type == "pmos"
    specs = m17.grid_curve_specs(dev.vdd, pm)
    z = np.zeros(1)
    curves = [m5.MeasCurve("grid", "idvd", s["fixed"], s["vb"], np.asarray(s["sweep"], float), z, z, z, z, "")
              for s in specs]
    _, _, box1 = b1_box(dev)
    prob = m5.MeasProblem(dev, d17.T, box1)
    t0 = time.time()
    sims = m5.simulate(dev, prob.card, curves, prob.temp_C, dict(rec.meta["params"]))
    grid = np.vstack([np.asarray(x, float) for x in sims])
    np.savez_compressed(Path(out_dir) / f"grid_{tag}.npz", i=grid, axis=m17.pen_axis(dev.vdd))
    s = -1.0 if pm else 1.0
    return {"tag": tag, "vdd": dev.vdd, "shape": list(grid.shape), "n_nonfinite": int(np.sum(~np.isfinite(grid))),
            "i_fwd_corner": float(s * grid[-1, -1]), "seconds": round(time.time() - t0, 2)}


def bind_b(t: int) -> None:
    d17.T = int(t)
    d17.OUT = b_dir(t)
    d17.PREREG = PREREG
    d17.UNIT_TESTS = UNIT_TESTS
    d17._CACHE.pop("recs", None)
    d17.records = b1_records
    d17.refit_task = b1_refit_task
    d17.grid_task = b1_grid_task


if _T_ENV:
    bind_b(int(_T_ENV))


def b_folds() -> list[str]:
    order = d17.load_order()
    return list(m17.SMOKE_FOLDS) if SMOKE["on"] else sorted(order)


def stage_b_records(workers: int) -> dict:
    """B1 records (I22-1 B1 cards on the measured bias set); VB1."""
    from cryoml.explore.e1_correction import load_devices, save_devices, score_device
    t = d17.T
    rd = b_dir() / "records"
    rd.mkdir(parents=True, exist_ok=True)
    start = load_devices(EXP / "I5_1" / "start" / f"records_{t}K")
    cards = jload(EXP / f"I22_1/partA_ihp{t}/heldout_cards.json")
    A = jload(EXP / f"I22_1/partA_ihp{t}/results.json")["per_device"]["B1"]
    with get_context("spawn").Pool(workers) as pool:
        recs = pool.map(_record_task, [(r.tag, t, cards[f"B1|{r.tag}"]) for r in start], chunksize=1)
    save_devices(rd / f"records_{t}K", recs)
    d17._CACHE.pop("recs", None)
    back = {r.tag: r for r in load_devices(rd / f"records_{t}K")}
    vb1 = {tag: abs(float(score_device(r, [c.sim for c in r.curves])["rrms"]) - float(A[tag])) for tag, r in back.items()}
    out = {"T_K": t, "n": len(back), "vb1_max": max(vb1.values()), "per_device": vb1,
           "VB1": bool(len(back) == 18 and max(vb1.values()) <= m.VB1_TOL),
           "order": [r.tag for r in start], "dev_check": d22.dev_check(f"ihp{t}", list(back))}
    jdump(out, rd / "records.json")
    print(json.dumps({k: out[k] for k in ("T_K", "n", "vb1_max", "VB1")}), flush=True)
    return out


def _record_task(task):
    tag, t, params = task
    dev = d17.devices()[tag]
    _, _, box1 = b1_box(dev)
    prob = m5.MeasProblem(dev, t, box1)
    return m5.to_record(dev, t, prob.simulate_params(dict(params)), dict(params))


def stage_b_base(workers: int) -> dict:
    """B1_cont: refit of the B1 card with g = 0, same budget."""
    from cryoml.explore.e1_correction import Correction
    rd = b_dir() / "base"
    o = opts()
    unit = run_unit_tests(rd)
    folds = b_folds()
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(b1_refit_task, [("B1_cont", Correction("C0"), h, o) for h in folds], chunksize=1)
    per = {r["tag"]: r["rrms"] for r in rows}
    out = {"T_K": d17.T, "smoke": SMOKE["on"], "opts": o, "folds": folds, "per_device": per,
           "unit_tests": unit, "rows": {r["tag"]: {k: v for k, v in r.items() if k != "params"} for r in rows},
           "timing": {"refit_s": round(time.time() - t0, 1)}}
    jdump(out, rd / "results.json")
    jdump({r["tag"]: r["params"] for r in rows}, rd / "heldout_cards.json")
    print(json.dumps({"T": d17.T, "mean": float(np.nanmean(list(per.values()))), "timing": out["timing"]}), flush=True)
    return out


def stage_b_inner(workers: int) -> dict:
    """Nested selection of lambda per outer fold (d17.stage_inner with the
    inner refit rows kept for the call count)."""
    rd = b_dir() / "inner"
    order = d17.load_order()
    outers = b_folds()
    hyper = {"steps": 200} if SMOKE["on"] else {}
    o = opts()
    pens_dir = str(b_dir() / "pens")
    pm = m17.is_pmos_map(order)
    grid = m17.GRID_LAMBDA
    plan, tasks = {}, []
    for out_f in outers:
        tr = m17.fold_train(order, out_f)
        folds = m17.inner_folds(tr, pm)
        plan[out_f] = {"train": tr, "inner_folds": folds}
        for j, held in enumerate(folds):
            for lam in grid:
                tasks.append((f"{out_f}|{j}|{lam:g}", m17.inner_train(tr, held), 0, hyper, float(lam), pens_dir))
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        trained = pool.map(d17.train_sym_task, tasks, chunksize=1)
        t_train = time.time() - t0
        print(f"inner: training {len(tasks)} members {t_train:.0f} s", flush=True)
        corr = {name: c for name, _, c, _ in trained}
        secs = {name: s for name, _, _, s in trained}
        rtasks = []
        for out_f in outers:
            for j, held in enumerate(plan[out_f]["inner_folds"]):
                for lam in grid:
                    key = f"{out_f}|{j}|{lam:g}"
                    rtasks += [(key, corr[key], t, o) for t in held]
        t1 = time.time()
        rows = pool.map(b1_refit_task, rtasks, chunksize=1)
        t_refit = time.time() - t1
    res = {"smoke": SMOKE["on"], "grid_lambda": list(grid), "eps": m17.EPS, "seed": 0, "opts": o,
           "hyper_overrides": hyper, "plan": plan, "outer": {},
           "timing": {"train_s": round(t_train, 1), "refit_s": round(t_refit, 1), "n_train": len(tasks),
                      "n_refit": len(rtasks)}, "train_seconds": secs,
           "inner_n_sim": {f"{r['label']}|{r['tag']}": r["n_sim"] for r in rows}}
    for out_f in outers:
        rr = {}
        for lam in grid:
            rr[f"{lam:g}"] = {r["tag"]: r["rrms"] for r in rows
                              if r["label"].startswith(f"{out_f}|") and r["label"].endswith(f"|{lam:g}")}
        scores = {lam: m17.inner_score(list(rr[f"{lam:g}"].values())) for lam in grid}
        res["outer"][out_f] = {"inner_rrms": rr, "scores": {f"{k:g}": v for k, v in scores.items()},
                               "n_scored": {f"{lam:g}": len(rr[f"{lam:g}"]) for lam in grid},
                               "lambda_star": m17.select_lambda(scores),
                               "n_sim": int(sum(r["n_sim"] for r in rows if r["label"].startswith(f"{out_f}|")))}
    res["lambda_star"] = {f: res["outer"][f]["lambda_star"] for f in outers}
    res["complete"] = bool(len(tasks) == len(outers) * 3 * len(grid)
                           and all(n == len(plan[f]["train"]) for f in outers
                                   for n in res["outer"][f]["n_scored"].values()))
    jdump(res, rd / "inner.json")
    with gzip.open(rd / "inner_corrections.json.gz", "wt") as fh:
        json.dump({k: c.to_json() for k, c in corr.items()}, fh)
    print(json.dumps({"lambda_star": res["lambda_star"], "complete": res["complete"], "timing": res["timing"]},
                     indent=1), flush=True)
    return res


def stage_b_sel(workers: int) -> dict:
    rd = b_dir() / "sel"
    inner = jload(b_dir() / "inner" / "inner.json")
    order = d17.load_order()
    folds = b_folds()
    hyper = {"steps": 200} if SMOKE["on"] else {}
    seeds = (0,) if SMOKE["on"] else v2.HYPER_V2["seeds"]
    o = opts()
    pens_dir = str(b_dir() / "pens")
    lam = {f: float(inner["lambda_star"][f]) for f in folds}
    names = {f: f"{m17.variant(lam[f])}|{f}" for f in folds}
    tasks = [(names[f], m17.fold_train(order, f), s, hyper, lam[f], pens_dir) for f in folds for s in seeds]
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        trained = pool.map(d17.train_sym_task, tasks, chunksize=1)
        t_train = time.time() - t0
        corrs, secs = d17.merge_sym(trained, list(names.values()))
        t1 = time.time()
        rows = pool.map(b1_refit_task, [("B1+g v2", corrs[names[f]], f, o) for f in folds], chunksize=1)
        t_refit = time.time() - t1
    per = {r["tag"]: r["rrms"] for r in rows}
    out = {"T_K": d17.T, "smoke": SMOKE["on"], "lambda_star": lam, "names": names, "seeds": list(seeds),
           "hyper": v2.HYPER_V2, "hyper_overrides": hyper, "opts": o, "per_device": per,
           "mean": float(np.nanmean(list(per.values()))),
           "heldout_rows": {r["tag"]: {k: v for k, v in r.items() if k != "params"} for r in rows},
           "train_seconds_per_member": secs,
           "timing": {"train_s": round(t_train, 1), "refit_s": round(t_refit, 1), "n_seed_tasks": len(tasks)}}
    jdump(out, rd / "results.json")
    jdump({n: c.to_json() for n, c in corrs.items()}, rd / "corrections.json")
    jdump({r["tag"]: r["params"] for r in rows}, rd / "heldout_cards.json")
    print(json.dumps({"T": d17.T, "mean": out["mean"], "lambda": lam, "timing": out["timing"]}), flush=True)
    return out


def stage_b_decide() -> dict:
    from cryoml.explore.i3_2_bias import paired_ratio
    t = d17.T
    B = b_dir()
    ds = f"ihp{t}"
    recj = jload(B / "records/records.json")
    pj = jload(B / "pens/pens.json")
    base = jload(B / "base/results.json")
    inner = jload(B / "inner/inner.json")
    sel = jload(B / "sel/results.json")
    A = jload(EXP / f"I22_1/partA_ihp{t}/results.json")
    st = d22.stored_rows(ds)
    tags = sorted(sel["per_device"])
    rows = {"B1": {x: float(A["per_device"]["B1"][x]) for x in tags},
            "B1_cont": {x: float(base["per_device"][x]) for x in tags},
            "B1+g v2": {x: float(sel["per_device"][x]) for x in tags},
            "C0_refit": {x: float(st["C0_refit"][x]) for x in tags},
            "selected g v2": {x: float(st["selected_g_v2"][x]) for x in tags}}
    n_fail = {a: int(sum(not np.isfinite(v[x]) for x in tags)) for a, v in rows.items()}
    validity = {"VB1": bool(recj["VB1"]), "VB1_max": recj["vb1_max"], "VB2": bool(pj["v4_ok"]),
                "VB3": bool(inner["complete"] and len(inner["lambda_star"]) == 18),
                "VB4": bool(all(n_fail[a] <= 0.1 * len(tags) for a in ("B1_cont", "B1+g v2"))),
                "VB5": bool(base["unit_tests"]["passed"]),
                "VB6": bool(recj["dev_check"]["ok"]), "n_failed": n_fail}
    validity["valid"] = bool(all(validity[k] for k in ("VB1", "VB2", "VB3", "VB4", "VB5", "VB6")))
    cmp = {"B1+g v2 vs B1": paired_ratio(rows["B1+g v2"], rows["B1"]),
           "B1+g v2 vs C0_refit": paired_ratio(rows["B1+g v2"], rows["C0_refit"]),
           "B1+g v2 vs B1_cont": paired_ratio(rows["B1+g v2"], rows["B1_cont"]),
           "B1_cont vs B1": paired_ratio(rows["B1_cont"], rows["B1"]),
           "B1+g v2 vs selected g v2": paired_ratio(rows["B1+g v2"], rows["selected g v2"]),
           "B1 vs C0_refit": paired_ratio(rows["B1"], rows["C0_refit"])}
    agg = {a: m22.family_means(v, d22.is_pmos) for a, v in rows.items()}
    cnt: dict = {}
    for f in tags:
        k = f"{float(inner['lambda_star'][f]):g}"
        cnt[k] = cnt.get(k, 0) + 1
    calls = {}
    for f in tags:
        b1n = A["n_sim"]["B1"].get(f) or 0
        calls[f] = {"B1_refit": b1n, "selection": inner["outer"][f]["n_sim"],
                    "heldout": sel["heldout_rows"][f]["n_sim"],
                    "total": b1n + inner["outer"][f]["n_sim"] + sel["heldout_rows"][f]["n_sim"],
                    "C0_refit": (A["n_sim"]["C0_refit"].get(f) or 0)}
    out = {"T_K": t, "validity": validity, "rows": rows, "comparisons": cmp,
           "labels": {k: m.label_ci(v) for k, v in cmp.items()}, "aggregates": agg,
           "lambda_star": inner["lambda_star"], "lambda_star_counts": cnt, "calls_per_device": calls,
           "calls_mean": {k: float(np.mean([c[k] for c in calls.values()])) for k in
                          ("B1_refit", "selection", "heldout", "total", "C0_refit")},
           "timing": {"inner": inner["timing"], "sel": sel["timing"], "base": base["timing"]}}
    jdump(out, B / "decision.json")
    print(json.dumps({"T": t, "validity": validity, "cmp": {k: (round(v["ratio"], 4), round(v["ci_low"], 4),
                                                                round(v["ci_high"], 4)) for k, v in cmp.items()},
                      "lambda": cnt}, default=_default), flush=True)
    return out


def stage_vb7(workers: int) -> dict:
    """B1-box refit path with g = 0 from the starting record reproduces the
    I22-1 B1 row (300 K, 2 DUTs)."""
    from cryoml.explore.e1_correction import Correction, load_devices
    t = 300
    start = {r.tag: r for r in load_devices(EXP / "I5_1" / "start" / f"records_{t}K")}
    devs = {d.tag: d for d in m5_dev_devices()}
    A = jload(EXP / f"I22_1/partA_ihp{t}/results.json")["per_device"]["B1"]
    o = dict(OPTS)
    with get_context("spawn").Pool(min(workers, 2)) as pool:
        rows = pool.map(_vb7_task, [(tag, start[tag], devs[tag], o) for tag in VB7_TAGS], chunksize=1)
    diff = {r["tag"]: abs(float(r["rrms"]) - float(A[r["tag"]])) for r in rows}
    out = {"tags": list(VB7_TAGS), "diff": diff, "max": max(diff.values()), "VB7": bool(max(diff.values()) <= m.VB7_TOL),
           "rows": [{k: v for k, v in r.items() if k != "params"} for r in rows]}
    jdump(out, base_dir() / "partB" / "vb7.json")
    print(json.dumps({"diff": diff, "VB7": out["VB7"]}), flush=True)
    return out


def m5_dev_devices():
    import i5_1_ihp_real as drv5
    return drv5.dev_devices()


def _vb7_task(task):
    from cryoml.explore.e1_correction import Correction
    tag, rec, dev, o = task
    return b1box_refit("vb7", Correction("C0"), rec, dev, o)


def stage_b(t: int, sub: str, workers: int) -> None:
    if t not in IHP_TEMPS:
        raise ValueError(t)
    if int(os.environ.get("CRYOML_I24_1_IHP_T", "0")) != t:
        os.environ["CRYOML_I24_1_IHP_T"] = str(t)        # spawned workers rebind at import
    bind_b(t)
    if sub == "records":
        stage_b_records(workers)
    elif sub == "pens":
        d17.stage_pens(workers, b_dir() / "pens")
    elif sub == "base":
        stage_b_base(workers)
    elif sub == "inner":
        stage_b_inner(workers)
    elif sub == "sel":
        stage_b_sel(workers)
    else:
        stage_b_decide()


# ===========================================================================
# decide
# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("logo", "vb7", "b", "decide"))
    ap.add_argument("--ds", default="")
    ap.add_argument("--temp", type=int, default=0)
    ap.add_argument("--sub", default="", choices=("", "records", "pens", "base", "inner", "sel", "decide"))
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if a.workers > 10:
        ap.error("workers <= 10")
    if a.stage == "logo":
        if a.ds not in m22.DATASETS:
            ap.error(f"--ds must be one of {m22.DATASETS}")
        stage_logo(a.ds, a.workers)
    elif a.stage == "vb7":
        stage_vb7(a.workers)
    elif a.stage == "b":
        stage_b(a.temp, a.sub, a.workers)
    else:
        import i24_1_report as rep
        rep.stage_decide()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
