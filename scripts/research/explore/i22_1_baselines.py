#!/usr/bin/env python3
"""I22-1: matched-budget baselines.

Pre-registration: out/research/explore/I22_1/prereg.md (prereg.json).

Stages (``--stage``):

  screen  --ds D   B1 edge screen on the state-C starting records of all DEV
                   devices of D (IHP: stored I5-1 per-class screen) -> screen_<D>.json
  partA   --ds D   C0_refit rerun, B1, B2, B3_d3, B4 (held-out refits, g v2
                   folds and training order, state C) -> partA_<D>/
  ls               Part C: traced C0_refit TRF refits (max_nfev 400 and 1600),
                   sky130 18 devices -> partC/ls.json
  opt     --method cma|turbo --config primary|secondary
                   Part C: 3 seeds x 18 devices, 4 B_d calls each -> partC/<method>_<config>.json
  cost             Part B table -> cost.json
  decide           labels, outcome, predictions, Part C labels -> decision.json

Smoke: CRYOML_I22_1_SMOKE=1 (2 held-out devices per dataset, reduced budgets,
output under I22_1/smoke).

  .venv/bin/python -I scripts/research/explore/i22_1_baselines.py --stage partA --ds nist4 --workers 8
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
EXP = ROOT / "out/research/explore"
os.environ.setdefault("CRYOML_E1_RECORDS", str(EXP / "I13_1" / "records"))   # sky130 state-C records
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(ROOT / "scripts" / "research" / "explore"))

import i4_1_baselines as d4  # noqa: E402  (sky130 refit / screen tasks)
import i5_1_ihp_real as drv5  # noqa: E402
import i6_1_tudelft as drv6  # noqa: E402
import i8_1_nist as d8  # noqa: E402
import i9_1_nist_rt as d9  # noqa: E402
import i18_2_tud as d18  # noqa: E402
import i20_1_nist_clamp as d20  # noqa: E402
from cryoml.explore import i6_1_tudelft as m6  # noqa: E402
from cryoml.explore import i12_1_symm as v2  # noqa: E402
from cryoml.explore import i17_2_ihp as m17  # noqa: E402
from cryoml.explore import i22_1_baselines as m  # noqa: E402

# TU Delft state C: the I6-1 refit and screen tasks build ``m6.MeasProblem``;
# I18-2 ``ProblemC`` (same constructor, clamp written into every deck) replaces
# it in this process and in every spawned worker (this module is re-imported).
# VA1 (C0_refit rerun = stored I18-2 state-C rows) checks the substitution.
_M6_ORIGINAL = m6.MeasProblem
m6.MeasProblem = d18.ProblemC

OUT = EXP / "I22_1"
PREREG = OUT / "prereg.json"
OPTS = {"heldout_nfev": 400, "heldout_irls": 2, "diff_step": 0.01}
SMOKE = {"on": os.environ.get("CRYOML_I22_1_SMOKE") == "1"}
SMOKE_N = 2
ARMS = ("C0_refit", "B1", "B2", "B3_d3", "B4")
COST_ORDER = {"B1": 0, "B2": 1, "B4": 2, "B3_d3": 3, "C0_refit": 4}
UNIT_TESTS = ("tests.test_i22_1_baselines", "tests.test_i4_1_baselines", "tests.test_i13_1_clamp",
              "tests.test_i20_1_nist_clamp", "tests.test_i18_2_tud")
SKY_SPEC = ROOT / "out/cold_loop/full_direct/round_3/spec.json"


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


# ---------------------------------------------------------------------------
# dataset adapters (cached per process)
# ---------------------------------------------------------------------------
_C: dict = {}


def kind(ds: str) -> str:
    return "sky130" if ds == "sky130" else "nist" if ds.startswith("nist") else ds[:3]


def temp(ds: str) -> int:
    return int(ds[3:])


def recs(ds: str) -> dict:
    """tag -> state-C starting record."""
    key = ("recs", ds)
    if key not in _C:
        k = kind(ds)
        if k == "sky130":
            from i3_2_bias import load_records_all18
            rl, _ = load_records_all18()
            _C[("order", ds)] = [r.tag for r in rl]
            _C[key] = {r.tag: r for r in rl}
        elif k == "nist":
            _C[key] = d20.c_records(ds)
        elif k == "ihp":
            from cryoml.explore.e1_correction import load_devices
            rl = load_devices(EXP / "I5_1" / "start" / f"records_{temp(ds)}K")
            _C[("order", ds)] = [r.tag for r in rl]
            _C[key] = {r.tag: r for r in rl}
        else:
            _C[key] = d18.by_tag(temp(ds))
    return _C[key]


def order(ds: str) -> list[str]:
    """Load order of the DEV records used by the selected g v2 arm."""
    k = kind(ds)
    if k in ("sky130", "ihp"):
        recs(ds)
        return list(_C[("order", ds)])
    if k == "nist":
        return list(d20.finite_order(ds))
    return list(d18.load_order(temp(ds)))


def ihp_devices() -> dict:
    if "ihp_dev" not in _C:
        _C["ihp_dev"] = {d.tag: d for d in drv5.dev_devices()}
    return _C["ihp_dev"]


def folds(ds: str) -> list[tuple[str, list[str], list[str]]]:
    """(fold key, held-out tags, training tags in load order)."""
    o = order(ds)
    k = kind(ds)
    if k == "sky130":
        out = [(h, [h], v2.fold_train_tags(o, h)) for h in sorted(o)]
    elif k == "nist":
        out = [(h, [h], [t for t in o if t != h]) for h in o]
    elif k == "ihp":
        out = [(h, [h], m17.fold_train(o, h)) for h in sorted(o)]
    else:
        t = temp(ds)
        out = [(g, [x for x in o if m6.group_of(x) == g], d18.fold_train(t, g)) for g in d18.groups(t)]
    if SMOKE["on"]:
        n = [f for f in out if not is_pmos(f[1][0])][:1]
        p = [f for f in out if is_pmos(f[1][0])][:1]
        out = [(key, held[:1], train) for key, held, train in n + p]
    return out


def clusters(ds: str) -> dict | None:
    return {t: m6.group_of(t) for t in order(ds)} if kind(ds) == "tud" else None


def is_pmos(tag: str) -> bool:
    return "pmos" in tag or tag.startswith("pfet")


def dev_check(ds: str, tags) -> dict:
    tags = set(tags)
    k = kind(ds)
    if k == "sky130":
        s = jload(EXP / "real_split.json")
        dev, test = set(s["REAL_DEV"]) | set(s["REAL_TEST"]), set()
    else:
        src = {"nist4": "I8_1", "nistrt": "I9_1"}.get(ds, "I5_1" if k == "ihp" else "I6_1")
        s = jload(EXP / src / "split.json")
        dev, test = set(s["DEV"]), set(s["TEST"])
    in_test, not_dev = sorted(tags & test), sorted(tags - dev)
    return {"n": len(tags), "n_in_TEST": len(in_test), "n_not_in_DEV": len(not_dev),
            "ok": not in_test and not not_dev}


def refit(ds: str, label: str, kind_: str, payload: dict, tag: str, o: dict) -> dict:
    """One held-out refit through the dataset's own refit task (state C)."""
    rec = recs(ds)[tag]
    k = kind(ds)
    if k == "sky130":
        return d4.refit_task((label, rec, {"kind": kind_, **payload}, o))
    if k == "nist":
        dev, binfo = d20.devices(ds)[tag], d20.bins(ds)[tag]
        drv = d8 if ds == "nist4" else d9
        return drv.refit_task((label, kind_, payload, rec, dev, binfo, o))
    if k == "ihp":
        return drv5.refit_task((label, kind_, payload, rec, ihp_devices()[tag], o))
    return drv6.refit_task((label, kind_, payload, rec, d18.devices()[tag], o))


def stored_rows(ds: str) -> dict:
    """Stored C0_refit, M_bias v1 and selected g v2 rows (freeze-draft loader)."""
    import plot_freeze_draft as pf
    key = ("stored", ds)
    if key not in _C:
        allds = pf.datasets()
        d = allds[m.FREEZE_KEYS[ds]]
        _C[key] = {"C0_refit": dict(d["c0"]), "M_bias_v1": dict(d["v1"]),
                   "selected_g_v2": dict(d["sel"]) if d["sel"] is not None else None, "src": d["src"]}
    return _C[key]


# ---------------------------------------------------------------------------
# screen
# ---------------------------------------------------------------------------
def screen_task(task):
    ds, tag = task
    rec = recs(ds)[tag]
    k = kind(ds)
    if k == "sky130":
        t, out = d4.screen_task(rec)
        return t, {"screen": out}
    if k == "nist":
        drv = d8 if ds == "nist4" else d9
        return drv.screen_task((rec, d20.devices(ds)[tag], d20.bins(ds)[tag]))
    t, out = drv6.screen_task((d18.devices()[tag], rec))
    return t, {"screen": out}


def stage_screen(ds: str, workers: int) -> dict:
    t0 = time.time()
    rd = base_dir()
    tags = order(ds)
    if SMOKE["on"]:
        tags = sorted({f[1][0] for f in folds(ds)})
    if kind(ds) == "ihp":
        st = jload(EXP / "I5_1/start/b1_screen.json")
        out = {"ds": ds, "source": "I5_1/start/b1_screen.json (per class, 300 K; stored)",
               "selected_by_class": {c: st[c]["selected"] for c in ("lv", "hv")},
               "calls": {"per_device": {}, "total": 0,
                         "note": "stored I5-1 screen; calls counted once in the I5-1 run (300 K)"},
               "dev_check": dev_check(ds, tags)}
        for c in ("lv", "hv"):
            out["calls"]["per_device"].update(m.screen_calls(st[c]["per_device"])["per_device"])
        out["calls"]["total"] = int(sum(out["calls"]["per_device"].values()))
        jdump(out, rd / f"screen_{ds}.json")
        print(json.dumps({k: out[k] for k in ("selected_by_class",)}), flush=True)
        return out
    with get_context("spawn").Pool(workers) as pool:
        rows = dict(pool.map(screen_task, [(ds, t) for t in tags], chunksize=1))
    sel = m.screen_select({t: r["screen"] for t, r in rows.items()})
    out = {"ds": ds, "n_devices": len(rows), **sel,
           "calls": m.screen_calls({t: r["screen"] for t, r in rows.items()}),
           "per_device": {t: r["screen"] for t, r in rows.items()},
           "values": {t: r.get("values") for t, r in rows.items()} if kind(ds) == "nist" else None,
           "dev_check": dev_check(ds, tags), "runtime_s": round(time.time() - t0, 1)}
    jdump(out, rd / f"screen_{ds}.json")
    print(f"{ds}: {len(sel['candidates'])} candidates, selected {sel['selected']} "
          f"({out['calls']['total']} calls, {out['runtime_s']} s)", flush=True)
    return out


def b1_payload(ds: str, tag: str, screen: dict) -> dict:
    k = kind(ds)
    if k == "ihp":
        return {"names": screen["selected_by_class"][ihp_devices()[tag].cls]}
    if k == "nist":
        return {"names": screen["selected"], "values": screen["values"][tag]}
    return {"names": screen["selected"]}


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------
def train_task(task):
    """(ds, arm, fold key, training tags) -> (arm, key, correction, seconds)."""
    import torch
    torch.set_num_threads(1)
    from cryoml.explore.i4_1_baselines import fit_gp, fit_poly
    ds, arm, key, train_tags = task
    rs = recs(ds)
    train = [rs[t] for t in train_tags]
    t0 = time.time()
    if arm == "B3_d3":
        corr = fit_poly("B3_d3", 3, train, hyper={"steps": 50} if SMOKE["on"] else None)
    elif arm == "B4":
        corr = fit_gp("B4", train, total=200 if SMOKE["on"] else 2000, seed=0,
                      maxiter=10 if SMOKE["on"] else 200)
    else:
        raise KeyError(arm)
    return arm, key, corr, round(time.time() - t0, 2)


def refit_task(task):
    ds, label, kind_, payload, tag, o = task
    t0 = time.time()
    try:
        row = refit(ds, label, kind_, payload, tag, o)
    except Exception as e:                                  # a failed refit is a failed row
        row = {"label": label, "tag": tag, "rrms": float("nan"), "failed": ["exception"],
               "error": f"{type(e).__name__}: {e}", "n_sim": None, "params": {}}
    row["label"], row["tag"], row["wall_s"] = label, tag, round(time.time() - t0, 2)
    return row


def stage_partA(ds: str, workers: int) -> dict:
    from cryoml.explore.e1_correction import Correction
    t_start = time.time()
    rd = base_dir() / f"partA_{ds}"
    unit = run_unit_tests(rd)
    screen = jload(base_dir() / f"screen_{ds}.json")
    fl = folds(ds)
    held_all = [h for _, held, _ in fl for h in held]
    o = opts()
    timing = {}
    with get_context("spawn").Pool(workers) as pool:
        t0 = time.time()
        ttasks = [(ds, arm, key, train) for key, _, train in fl for arm in ("B4", "B3_d3")]
        trained = pool.map(train_task, ttasks, chunksize=1)
        timing["train_wall_s"] = round(time.time() - t0, 1)
        corrs = {(a, k): c for a, k, c, _ in trained}
        train_s = {}
        for a, k, _, s in trained:
            train_s.setdefault(a, {})[k] = s
        print(f"{ds}: trained {len(ttasks)} corrections in {timing['train_wall_s']} s", flush=True)
        rtasks = []
        for key, held, _ in fl:
            for h in held:
                rtasks.append((ds, "C0_refit", "corr", {"corr": Correction("C0")}, h, o))
                rtasks.append((ds, "B1", "b1", b1_payload(ds, h, screen), h, o))
                rtasks.append((ds, "B2", "b2", {}, h, o))
                rtasks.append((ds, "B3_d3", "corr", {"corr": corrs[("B3_d3", key)]}, h, o))
                rtasks.append((ds, "B4", "corr", {"corr": corrs[("B4", key)]}, h, o))
        rtasks.sort(key=lambda t: COST_ORDER[t[1]])
        t0 = time.time()
        rows = pool.map(refit_task, rtasks, chunksize=1)
        timing["refit_wall_s"] = round(time.time() - t0, 1)
        timing["workers"] = workers
    per = {a: {} for a in ARMS}
    keep = {}
    for r in rows:
        per[r["label"]][r["tag"]] = float(r["rrms"]) if r.get("rrms") is not None else float("nan")
        keep[f"{r['label']}|{r['tag']}"] = {k: v for k, v in r.items() if k not in ("params",)}
    st = stored_rows(ds)
    stored_c0 = {t: st["C0_refit"][t] for t in held_all}
    va1 = m.max_abs_diff(per["C0_refit"], stored_c0)
    n_fail = {a: int(sum(1 for t in held_all if not np.isfinite(per[a].get(t, np.nan)))) for a in ARMS}
    complete = all(set(per[a]) == set(held_all) for a in ARMS)
    va5 = None
    if kind(ds) == "ihp" and not SMOKE["on"]:
        i5 = jload(EXP / f"I5_1/lodo_{temp(ds)}K/results.json")["per_device"]
        va5 = {a: m.max_abs_diff(per[a], {t: i5[a][t] for t in held_all}) for a in ("B1", "B2", "B3_d3")}
    validity = {"VA1_c0_max_abs_diff": va1, "VA1": bool(va1 <= m.VA1_TOL),
                "VA2_complete": complete, "VA2_n_failed": n_fail,
                "VA2": bool(complete and all(n <= m.MAX_FAIL_FRACTION * len(held_all) for n in n_fail.values())),
                "VA3_unit_tests": bool(unit["passed"]), "VA4_dev_check": dev_check(ds, held_all),
                "VA5_ihp_vs_I5_1": va5}
    validity["VA4"] = bool(validity["VA4_dev_check"]["ok"])
    validity["valid"] = bool(validity["VA1"] and validity["VA2"] and validity["VA3_unit_tests"] and validity["VA4"])
    n_sim = {a: {r["tag"]: r.get("n_sim") for r in rows if r["label"] == a} for a in ARMS}
    out = {"ds": ds, "name": m.NAMES[ds], "smoke": SMOKE["on"], "opts": o, "arms": list(ARMS),
           "held_out": held_all, "folds": {k: {"held": h, "train": tr} for k, h, tr in fl},
           "b1": {"selected": screen.get("selected", screen.get("selected_by_class"))},
           "per_device": per, "validity": validity,
           "n_sim": n_sim, "n_sim_total": {a: int(sum(v for v in n_sim[a].values() if v)) for a in ARMS},
           "refit_runtime_s_sum": {a: round(sum(r.get("wall_s", 0) for r in rows if r["label"] == a), 1)
                                   for a in ARMS},
           "train_seconds": train_s,
           "b4_info": {k: corrs[("B4", k)].info for k, _, _ in fl},
           "rows": keep, "stored_sources": st["src"],
           "timing": {**timing, "runtime_s": round(time.time() - t_start, 1)}}
    jdump(out, rd / "results.json")
    jdump({f"B3_d3|{k}": corrs[("B3_d3", k)].to_json() for k, _, _ in fl}, rd / "corrections_B3_d3.json")
    with gzip.open(rd / "corrections_B4.json.gz", "wt") as fh:
        fh.write(json.dumps({f"B4|{k}": corrs[("B4", k)].to_json() for k, _, _ in fl}, default=_default))
    jdump({f"{r['label']}|{r['tag']}": r.get("params", {}) for r in rows}, rd / "heldout_cards.json")
    print(json.dumps({"ds": ds, "validity": {k: v for k, v in validity.items() if k != "VA4_dev_check"},
                      "mean": {a: float(np.nanmean(list(per[a].values()))) for a in ARMS},
                      "timing": out["timing"]}, indent=1, default=_default), flush=True)
    return out


# ---------------------------------------------------------------------------
# Part C (sky130 77 K, C0_refit stage)
# ---------------------------------------------------------------------------
LS_NFEV = {"primary": 400, "secondary": 1600}
SEEDS = (1, 2, 3)
BUDGET_FACTOR = {"cma": 4, "turbo": 1}     # amendment A1: TuRBO runs stop at B_d (GP fitting cost)


def sky_problem(tag: str):
    """(DeviceProblem, start local, active indices) of the C0_refit stage (as
    i2_1_joint._refit)."""
    from cold_loop import box_for
    from cryoml.ngspice_fit import DeviceProblem
    spec = jload(SKY_SPEC)
    base = DeviceProblem(tag, box_for(spec, tag))
    rec = recs("sky130")[tag]
    start = base.box.params_to_local({**spec["published_by_device"][tag],
                                      **{k: v for k, v in rec.meta["params"].items() if k in base.box.names}})
    return base, np.asarray(start, float), np.flatnonzero(base.box.active)


def ls_task(task):
    """Traced C0_refit polish (CorrectedProblem with C0, as _refit)."""
    from cryoml.explore.e1_correction import Correction, score_device
    from cryoml.explore.i2_1_joint import CorrectedProblem
    from cryoml.ngspice_fit import polish
    tag, nfev = task
    t0 = time.time()
    base, start, _ = sky_problem(tag)
    rec = recs("sky130")[tag]
    trace = []
    orig = base.simulate

    def traced(local):
        sims = orig(local)
        trace.append(float(np.mean(base.curve_rrms(sims))))
        return sims
    base.simulate = traced
    prob = CorrectedProblem(base, Correction("C0"), rec.family, rec.L_um, rec.W_um, rec.vdd, rec.source)
    o = opts()
    res = polish(prob, start, max_nfev=nfev if not SMOKE["on"] else 20, irls_rounds=o["heldout_irls"],
                 diff_step=o["diff_step"])
    base.simulate = orig
    params = base.box.local_to_params(res["local"])
    new = d4.record_from_sims(rec, base, base.simulate(res["local"]), params)
    score = float(score_device(new, [c.sim for c in new.curves])["rrms"])
    return {"tag": tag, "max_nfev": nfev, "n_sim": len(trace), "trace": trace, "best_traced": float(min(trace)),
            "objective": float(res["objective"]), "rrms_resim": score, "history": res["history"],
            "params": params, "seconds": round(time.time() - t0, 2)}


def stage_ls(workers: int) -> dict:
    rd = base_dir() / "partC"
    tags = sorted(order("sky130"))[:SMOKE_N] if SMOKE["on"] else sorted(order("sky130"))
    unit = run_unit_tests(rd)
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(ls_task, [(t, n) for n in (LS_NFEV["primary"], LS_NFEV["secondary"]) for t in tags],
                        chunksize=1)
    prim = {r["tag"]: r for r in rows if r["max_nfev"] == LS_NFEV["primary"]}
    sec = {r["tag"]: r for r in rows if r["max_nfev"] == LS_NFEV["secondary"]}
    stored = stored_rows("sky130")["C0_refit"]
    vc1 = m.max_abs_diff({t: r["rrms_resim"] for t, r in prim.items()}, {t: stored[t] for t in prim})
    resim = max(abs(r["rrms_resim"] - r["best_traced"]) for r in rows)
    out = {"tags": tags, "smoke": SMOKE["on"], "budget": {t: r["n_sim"] for t, r in prim.items()},
           "primary": prim, "secondary": sec, "VC1_max_abs_diff": vc1, "VC1": bool(vc1 <= m.VA1_TOL),
           "resim_vs_traced_max": resim, "unit_tests_passed": bool(unit["passed"]),
           "dev_check": dev_check("sky130", tags), "wall_s": round(time.time() - t0, 1), "workers": workers}
    jdump(out, rd / "ls.json")
    print(json.dumps({"VC1": vc1, "resim": resim, "budget": out["budget"], "wall_s": out["wall_s"]},
                     indent=1), flush=True)
    return out


def opt_task(task):
    import torch
    torch.set_num_threads(1)
    method, config, tag, seed, budget = task
    t0 = time.time()
    base, start, active = sky_problem(tag)

    def f(x_active):
        local = start.copy()
        local[active] = np.clip(x_active, -1.0, 1.0)
        sims = base.simulate(local)
        return float(np.mean(base.curve_rrms(sims)))
    x0 = start[active]
    total = BUDGET_FACTOR[method] * budget
    if method == "cma":
        res = m.run_cma(f, x0, total, seed, sigma0=m.CMA_SIGMA0[config])
    else:
        kw = {"n_candidates": 500} if SMOKE["on"] else {}
        res = m.run_turbo(f, x0, total, seed, length_init=m.TURBO["L_init"][config], **kw)
    x_b, v_b = m.best_x_at(res["values"], res["xs"], budget)
    x_t, v_t = m.best_x_at(res["values"], res["xs"], total)
    out = {"method": method, "config": config, "tag": tag, "seed": seed, "budget": budget, "n": res["n"],
           "values": res["values"], "value_at_budget": v_b,
           "value_at_4budget": v_t if total == 4 * budget else None,
           "seconds": round(time.time() - t0, 2),
           **{k: res[k] for k in ("restarts", "final_length", "iterations") if k in res}}
    # VC2: the best points at B_d and at 4 B_d are re-simulated once (not counted)
    out["resim_at_budget"] = float(f(x_b))
    out["resim_at_4budget"] = float(f(x_t))
    out["resim_diff"] = max(abs(out["resim_at_budget"] - v_b), abs(out["resim_at_4budget"] - v_t))
    local = start.copy()
    local[active] = np.clip(x_b, -1.0, 1.0)
    out["params_at_budget"] = base.box.local_to_params(local)
    return out


def stage_opt(method: str, config: str, workers: int) -> dict:
    rd = base_dir() / "partC"
    ls = jload(rd / "ls.json")
    tags = ls["tags"]
    seeds = SEEDS[:1] if SMOKE["on"] else SEEDS
    tasks = [(method, config, t, s, int(ls["budget"][t])) for t in tags for s in seeds]
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(opt_task, tasks, chunksize=1)
    out = {"method": method, "config": config, "smoke": SMOKE["on"], "rows": rows,
           "budget_factor": BUDGET_FACTOR[method],
           "VC2_budget_exact": bool(all(r["n"] == BUDGET_FACTOR[method] * r["budget"] for r in rows)),
           "VC2_resim_max": max(r.get("resim_diff", 0.0) for r in rows),
           "wall_s": round(time.time() - t0, 1), "workers": workers}
    jdump(out, rd / f"{method}_{config}.json")
    print(json.dumps({"method": method, "config": config, "VC2": out["VC2_budget_exact"],
                      "resim": out["VC2_resim_max"], "wall_s": out["wall_s"],
                      "geo_ratio_at_B": float(np.exp(np.mean([np.log(r["value_at_budget"] /
                                                                    ls["primary"][r["tag"]]["best_traced"])
                                                             for r in rows])))}, indent=1), flush=True)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("screen", "partA", "ls", "opt", "cost", "decide"))
    ap.add_argument("--ds", default="")
    ap.add_argument("--method", default="cma", choices=("cma", "turbo"))
    ap.add_argument("--config", default="primary", choices=("primary", "secondary"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.stage in ("screen", "partA") and a.ds not in m.DATASETS:
        ap.error(f"--ds must be one of {m.DATASETS}")
    if a.stage == "screen":
        stage_screen(a.ds, a.workers)
    elif a.stage == "partA":
        stage_partA(a.ds, a.workers)
    elif a.stage == "ls":
        stage_ls(a.workers)
    elif a.stage == "opt":
        stage_opt(a.method, a.config, a.workers)
    elif a.stage == "cost":
        import i22_1_report as rep
        rep.stage_cost()
    else:
        import i22_1_report as rep
        rep.stage_decide()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
