#!/usr/bin/env python3
"""I23-1: call-matched IHP control for B1 (Part A); g v2 with lambda fixed at
1 V, no nested selection (Part B).

Pre-registration: out/research/explore/I23_1/prereg.md (prereg.json).

Stages (``--stage``):

  partA   --ds ihp233|ihp300|ihp343|ihp398
          C0_refit rerun through polish_capped (VA1), C0_call (C0_refit with
          max_nfev and call limit N_B1(d)), B1_cap (B1 with call limit N_C0(d))
          -> partA_<ds>/results.json
  verify  --ds sky130|ihp300|tud4|tud295|nist4
          VB1: retrain (no cache) and refit a sample of the reused stored
          lambda = 1 V folds -> verify_<ds>.json
  fixl1   --ds ihp233|ihp343|ihp398|nist4|nistrt
          new fixed_l1 rows (5-member g v2 at lambda = 1 V per outer fold,
          held-out refit) -> fixl1_<ds>/results.json
  decide  (i23_1_report.py) -> decision.json

Smoke: CRYOML_I23_1_SMOKE=1 (2 DUTs / folds, reduced budgets and steps,
output under I23_1/smoke).

  .venv/bin/python -I scripts/research/explore/i23_1_budget.py --stage partA --ds ihp300 --workers 10
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
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

import i22_1_baselines as d22  # noqa: E402  (I22-1 driver: records, folds, refit, dev_check; sets sky130 records env)
import i15_2_ablate as d15  # noqa: E402  (sky130 g v2 train / refit)
import i17_2_ihp as d17  # noqa: E402  (IHP g v2 train / refit)
import i18_2_tud as d18  # noqa: E402  (TU Delft g v2 train / refit, state C)
import i20_1_nist_clamp as d20  # noqa: E402  (NIST g v2 train / refit, state C)
import cryoml.ngspice_fit as nf  # noqa: E402
from cryoml.explore import i6_1_tudelft as m6  # noqa: E402
from cryoml.explore import i12_1_symm as v2  # noqa: E402
from cryoml.explore import i17_2_ihp as m17b  # noqa: E402
from cryoml.explore import i23_1_budget as m  # noqa: E402

OUT = EXP / "I23_1"
SMOKE = {"on": os.environ.get("CRYOML_I23_1_SMOKE") == "1"}
LAM = m.FIXED_LAMBDA
UNIT_TESTS = ("tests.test_i23_1_budget", "tests.test_i22_1_baselines", "tests.test_i12_1_symm",
              "tests.test_i15_2_ablate", "tests.test_i17_2_ihp", "tests.test_i18_2_tud", "tests.test_i20_1_nist_clamp")
_ORIG_POLISH = nf.polish


def jload(p):
    return json.loads(Path(p).read_text())


def jdump(obj, p):
    d22.jdump(obj, p)


def base_dir() -> Path:
    return OUT / "smoke" if SMOKE["on"] else OUT


# NIST member cache under I23_1 (not the I20-1 cache)
def _cache_path(ds: str, name: str) -> Path:
    safe = name.replace("|", "__").replace("/", "_")
    return base_dir() / "cache" / ds / f"{safe}.json"


d20.cache_path = _cache_path


def set_threads() -> None:
    import torch
    torch.set_num_threads(1)


def run_unit_tests(rd: Path) -> dict:
    r = subprocess.run([sys.executable, "-m", "unittest", *UNIT_TESTS], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"})
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    out = {"files": list(UNIT_TESTS), "rc": r.returncode, "passed": r.returncode == 0, "tail": tail}
    jdump(out, rd / "unit_tests.json")
    print(f"unit tests: rc {r.returncode}; {tail[-1] if tail else ''}", flush=True)
    return out


def temp(ds: str) -> int:
    return int(ds[3:])


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------
def capped_refit_task(task):
    """(ds, label, kind, payload, tag, opts, limit) -> refit row; the refit's
    ``polish`` is ``polish_capped`` with at most limit - 1 calls inside polish
    (limit None: no limit)."""
    set_threads()
    ds, label, kind_, payload, tag, o, limit = task
    info = {}

    def patched(problem, local0, **kw):
        res = m.polish_capped(problem, local0, max_calls=None if limit is None else int(limit) - 1, **kw)
        info.update(calls_in_polish=res["calls"], capped=res["capped"])
        return res
    nf.polish = patched
    try:
        row = d22.refit_task((ds, label, kind_, payload, tag, o))
    finally:
        nf.polish = _ORIG_POLISH
    row.update(info, limit=limit, label=label)
    return row


def stage_partA(ds: str, workers: int) -> dict:
    from cryoml.explore.e1_correction import Correction
    t_start = time.time()
    rd = base_dir() / f"partA_{ds}"
    unit = run_unit_tests(rd)
    i22 = jload(EXP / f"I22_1/partA_{ds}/results.json")
    screen = jload(EXP / f"I22_1/screen_{ds}.json")
    held = list(i22["held_out"])
    if SMOKE["on"]:
        held = [h for h in held if h in m17b.SMOKE_FOLDS]
    n_b1 = {t: int(i22["n_sim"]["B1"][t]) for t in held}
    n_c0 = {t: int(i22["n_sim"]["C0_refit"][t]) for t in held}
    o400 = dict(d22.OPTS)
    if SMOKE["on"]:
        n_b1 = {t: min(v, 60) for t, v in n_b1.items()}
        n_c0 = {t: min(v, 30) for t, v in n_c0.items()}
    tasks = []
    for h in held:
        tasks.append((ds, "C0_refit_rerun", "corr", {"corr": Correction("C0")}, h,
                      dict(o400, heldout_nfev=20, heldout_irls=1) if SMOKE["on"] else o400, None))
        tasks.append((ds, "C0_call", "corr", {"corr": Correction("C0")}, h, dict(o400, heldout_nfev=n_b1[h]),
                      n_b1[h]))
        tasks.append((ds, "B1_cap", "b1", d22.b1_payload(ds, h, screen), h, o400, n_c0[h]))
    tasks.sort(key=lambda t: {"C0_call": 0, "B1_cap": 1, "C0_refit_rerun": 2}[t[1]])
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(capped_refit_task, tasks, chunksize=1)
    arms = ("C0_refit_rerun", "C0_call", "B1_cap")
    per = {a: {} for a in arms}
    nsim = {a: {} for a in arms}
    keep = {}
    for r in rows:
        per[r["label"]][r["tag"]] = float(r["rrms"]) if r.get("rrms") is not None else float("nan")
        nsim[r["label"]][r["tag"]] = r.get("n_sim")
        keep[f"{r['label']}|{r['tag']}"] = {k: v for k, v in r.items() if k not in ("params", "history")}
    stored_c0 = {t: i22["per_device"]["C0_refit"][t] for t in held}
    va1 = m.max_abs_diff(per["C0_refit_rerun"], stored_c0)
    va1_n = all(nsim["C0_refit_rerun"][t] == n_c0[t] for t in held)
    n_fail = {a: int(sum(1 for t in held if not np.isfinite(per[a].get(t, np.nan)))) for a in arms}
    complete = all(set(per[a]) == set(held) for a in arms)
    va3 = {"C0_call": m.call_limit_check(nsim["C0_call"], n_b1), "B1_cap": m.call_limit_check(nsim["B1_cap"], n_c0)}
    dev = d22.dev_check(ds, held)
    validity = {"VA1_max_abs_diff": va1, "VA1_n_sim_equal": bool(va1_n),
                "VA1": bool(va1 <= m.VA1_TOL and (va1_n or SMOKE["on"])),
                "VA2_complete": complete, "VA2_n_failed": n_fail,
                "VA2": bool(complete and all(n <= m.MAX_FAIL_FRACTION * len(held) for n in n_fail.values())),
                "VA3": bool(va3["C0_call"]["ok"] and va3["B1_cap"]["ok"]), "VA3_detail": va3,
                "VA4_dev_check": dev, "VA4": bool(dev["ok"]), "VA5_unit_tests": bool(unit["passed"])}
    if SMOKE["on"]:
        validity["VA1"] = True       # smoke uses reduced budgets; VA1 is evaluated in the full run only
    validity["valid"] = bool(validity["VA1"] and validity["VA2"] and validity["VA3"] and validity["VA4"]
                             and validity["VA5_unit_tests"])
    capped = {a: int(sum(1 for r in rows if r["label"] == a and r.get("capped"))) for a in arms}
    out = {"ds": ds, "smoke": SMOKE["on"], "held_out": held, "N_B1": n_b1, "N_C0": n_c0,
           "per_device": per, "n_sim": nsim, "n_capped": capped,
           "n_sim_total": {a: int(sum(v for v in nsim[a].values() if v)) for a in arms},
           "stored": {"B1": {t: i22["per_device"]["B1"][t] for t in held}, "C0_refit": stored_c0},
           "screen_calls_per_device": screen.get("calls", {}).get("per_device"),
           "validity": validity, "rows": keep,
           "refit_wall_s_sum": {a: round(sum(r.get("wall_s", 0) for r in rows if r["label"] == a), 1) for a in arms},
           "workers": workers, "runtime_s": round(time.time() - t_start, 1)}
    jdump(out, rd / "results.json")
    jdump({f"{r['label']}|{r['tag']}": r.get("params", {}) for r in rows}, rd / "heldout_cards.json")
    print(json.dumps({"ds": ds, "validity": {k: v for k, v in validity.items() if k not in ("VA4_dev_check",
                                                                                             "VA3_detail")},
                      "n_capped": capped, "n_sim_total": out["n_sim_total"],
                      "mean": {a: float(np.nanmean(list(per[a].values()))) for a in arms},
                      "runtime_s": out["runtime_s"]}, indent=1, default=float), flush=True)
    return out


# ---------------------------------------------------------------------------
# Part B: per-dataset training and refit adapters
# ---------------------------------------------------------------------------
def ihp_bind(t: int) -> None:
    if d17.T != t:
        d17.T = int(t)
        d17._CACHE.pop("recs", None)


def ihp_pens_dir(t: int) -> str:
    return str(EXP / ("I17_2/pens" if t == 300 else f"I21_1/ihp_{t}K/pens"))


def hyper() -> dict:
    return {"steps": 100} if SMOKE["on"] else {}


def seeds() -> tuple:
    return (0, 1) if SMOKE["on"] else tuple(v2.HYPER_V2["seeds"])


def ropts(base: dict) -> dict:
    return dict(base, heldout_nfev=20, heldout_irls=1) if SMOKE["on"] else dict(base)


def train_task(task):
    """(ds, fold key, train tags, seed) -> (fold key, seed, corr json, seconds)."""
    set_threads()
    ds, key, train, seed = task
    name = f"fixl1|{key}"
    if ds == "sky130":
        inputs, eps, lam = d15.m.VARIANTS["sym_e1_l1"]
        _, _, corr, sec, _ = d15.train_task((f"sym_e1_l1|{key}", train, seed, hyper(), lam, inputs, eps))
        return key, seed, corr.to_json(), sec
    if ds.startswith("ihp"):
        ihp_bind(temp(ds))
        _, _, corr, sec = d17.train_sym_task((f"sym_e1_l1|{key}", train, seed, hyper(), LAM, ihp_pens_dir(temp(ds))))
        return key, seed, corr.to_json(), sec
    if ds.startswith("tud"):
        _, _, corr, sec = d18.train_sym_task((f"sym_e1_l1|{key}", temp(ds), train, seed, hyper(), LAM))
        return key, seed, corr.to_json(), sec
    steps = 100 if SMOKE["on"] else None
    _, _, cj, sec = d20.train_v2_task((ds, f"{name}|{os.environ.get('CRYOML_I23_1_TAG', 'run')}", train, seed,
                                       LAM, steps))
    return key, seed, cj, sec


def refit_task(task):
    """(ds, corr json, tag) -> held-out row."""
    set_threads()
    ds, cj, tag = task
    corr = v2.SymCorrection.from_json(cj)
    if ds == "sky130":
        row = d15.refit_task(("fixed_l1", corr, tag, ropts(d15.REFIT_OPTS)))
    elif ds.startswith("ihp"):
        ihp_bind(temp(ds))
        row = d17.refit_task(("fixed_l1", corr, tag, ropts(d17.OPTS)))
    elif ds.startswith("tud"):
        row = d18.refit_task(("fixed_l1", corr, tag, temp(ds), ropts(d18.OPTS), "C"))
    else:
        row = d20.refit_task(("fixed_l1", ds, tag, corr, dict(d20.c_records(ds)[tag].meta["params"]), "C",
                              ropts(d20.HELDOUT)))
    row = {k: v for k, v in row.items() if k != "history"}
    row["label"], row["tag"] = "fixed_l1", tag
    return row


def order(ds: str) -> list[str]:
    if ds == "sky130":
        return list(d15.drv12.load_order())
    if ds.startswith("ihp"):
        ihp_bind(temp(ds))
        return list(d17.load_order())
    if ds.startswith("tud"):
        return list(d18.load_order(temp(ds)))
    return list(d20.finite_order(ds))


def folds(ds: str) -> list[tuple[str, list[str], list[str]]]:
    """(fold key, held-out tags, training tags in the selected arm's order)."""
    o = order(ds)
    if ds == "sky130":
        return [(h, [h], v2.fold_train_tags(o, h)) for h in sorted(o)]
    if ds.startswith("ihp"):
        return [(h, [h], m17b.fold_train(o, h)) for h in sorted(o)]
    if ds.startswith("tud"):
        t = temp(ds)
        return [(g, [x for x in o if m6.group_of(x) == g], d18.fold_train(t, g)) for g in d18.groups(t)]
    return [(h, [h], [x for x in o if x != h]) for h in o]


def run_folds(ds: str, fl: list, workers: int) -> tuple[dict, list, dict]:
    """Train 5 members per fold, merge, refit every held-out tag."""
    t0 = time.time()
    tasks = [(ds, key, train, s) for key, _, train in fl for s in seeds()]
    trained = []
    with get_context("spawn").Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(train_task, tasks, chunksize=1)):
            trained.append(r)
            if (i + 1) % 10 == 0 or i + 1 == len(tasks):
                print(f"  {ds}: {i + 1}/{len(tasks)} members, {time.time() - t0:.0f} s", flush=True)
        t_train = time.time() - t0
        by, secs = {}, {}
        for key, seed, cj, sec in sorted(trained, key=lambda r: (r[0], r[1])):
            by.setdefault(key, []).append(v2.SymCorrection.from_json(cj))
            secs.setdefault(key, []).append(sec)
        corrs = {k: v2.merge_seed_corrections(c) for k, c in by.items()}
        t1 = time.time()
        rows = pool.map(refit_task, [(ds, corrs[key].to_json(), h) for key, held, _ in fl for h in held],
                        chunksize=1)
        t_refit = time.time() - t1
    return corrs, rows, {"train_s": round(t_train, 1), "refit_s": round(t_refit, 1), "n_member_tasks": len(tasks),
                         "train_seconds_per_member": secs, "workers": workers}


def stored_fixed(ds: str) -> dict:
    """Stored lambda = 1 V rows: {tag: rrms}, {tag: heldout row}."""
    if ds == "sky130":
        r = jload(EXP / "I15_2/lodo_sym_e1_l1/results.json")
    elif ds == "ihp300":
        r = jload(EXP / "I17_2/lodo_sym_e1_l1/results.json")
    elif ds.startswith("tud"):
        r = jload(EXP / f"I18_2/lodo_{temp(ds)}K_sym_e1_l1/results.json")
    elif ds == "nist4":
        r = jload(EXP / "I20_1/lodo_nist4/results.json")
        keep = [t for t, lam in r["lambda_by_device"].items() if float(lam) == LAM]
        return {t: r["per_device"][t] for t in keep}, {t: r["heldout_rows"][t] for t in keep}
    else:
        raise KeyError(ds)
    return dict(r["per_device"]), dict(r["heldout_rows"])


def verify_samples(ds: str) -> list[str]:
    if ds == "nist4":
        per, _ = stored_fixed(ds)
        tags = sorted(per)
        n = [t for t in tags if t.startswith("nfet")][:1]
        p = [t for t in tags if t.startswith("pfet")][:1]
        return n + p
    return list(m.VB1_SAMPLES[ds])


def stage_verify(ds: str, workers: int) -> dict:
    t_start = time.time()
    rd = base_dir()
    unit = run_unit_tests(rd / f"verify_{ds}")
    os.environ["CRYOML_I23_1_TAG"] = "verify"
    per_st, _ = stored_fixed(ds)
    keys = verify_samples(ds)
    fl = [f for f in folds(ds) if f[0] in keys]
    corrs, rows, timing = run_folds(ds, fl, workers)
    rerun = {r["tag"]: float(r["rrms"]) for r in rows}
    diffs = {t: abs(rerun[t] - float(per_st[t])) for t in rerun}
    out = {"ds": ds, "smoke": SMOKE["on"], "source": m.REUSED[ds], "sample_folds": keys,
           "rerun": rerun, "stored": {t: per_st[t] for t in rerun}, "abs_diff": diffs,
           "max_abs_diff": max(diffs.values()) if diffs else float("inf"),
           "VB1": bool(diffs and max(diffs.values()) <= m.REPRO_TOL),
           "n_sim": {r["tag"]: r.get("n_sim") for r in rows}, "unit_tests_passed": bool(unit["passed"]),
           "dev_check": d22.dev_check(ds, list(rerun)), "timing": timing,
           "runtime_s": round(time.time() - t_start, 1)}
    jdump(out, rd / f"verify_{ds}.json")
    print(json.dumps({k: out[k] for k in ("ds", "sample_folds", "abs_diff", "VB1", "runtime_s")}, indent=1,
                     default=float), flush=True)
    return out


def selected_rows(ds: str) -> dict:
    return dict(d22.stored_rows(ds)["selected_g_v2"])


def lambda_star(ds: str) -> dict:
    if ds.startswith("ihp"):
        t = temp(ds)
        f = EXP / ("I17_2/decision.json" if t == 300 else f"I21_1/ihp_{t}K/decision.json")
        return {k: float(v) for k, v in jload(f)["lambda_star"].items()}
    if ds.startswith("nist"):
        return {k: float(v) for k, v in jload(EXP / f"I20_1/lodo_{ds}/results.json")["lambda_by_device"].items()}
    raise KeyError(ds)


def stage_fixl1(ds: str, workers: int) -> dict:
    t_start = time.time()
    rd = base_dir() / f"fixl1_{ds}"
    unit = run_unit_tests(rd)
    os.environ["CRYOML_I23_1_TAG"] = "run"
    fl = folds(ds)
    lam_star = lambda_star(ds)
    if ds == "nist4":                                      # reused: the lambda* = 1 V devices
        fl = [f for f in fl if lam_star[f[0]] != LAM]
    if SMOKE["on"]:
        n = [f for f in fl if not d22.is_pmos(f[1][0])][:1]
        p = [f for f in fl if d22.is_pmos(f[1][0])][:1]
        fl = n + p
    corrs, rows, timing = run_folds(ds, fl, workers)
    per = {r["tag"]: float(r["rrms"]) if r.get("rrms") is not None else float("nan") for r in rows}
    held = [h for _, hs, _ in fl for h in hs]
    sel = selected_rows(ds)
    overlap = {t: abs(per[t] - sel[t]) for t in held if lam_star.get(t) == LAM}
    n_fail = int(sum(1 for t in held if not np.isfinite(per.get(t, np.nan))))
    dev = d22.dev_check(ds, held)
    validity = {"VB2_overlap_n": len(overlap), "VB2_max_abs_diff": max(overlap.values()) if overlap else None,
                "VB2": bool(all(v <= m.REPRO_TOL for v in overlap.values())),
                "VB3_complete": set(per) == set(held), "VB3_n_failed": n_fail,
                "VB3": bool(set(per) == set(held) and n_fail <= m.MAX_FAIL_FRACTION * len(held)),
                "VB4_dev_check": dev, "VB4": bool(dev["ok"]), "VB5_unit_tests": bool(unit["passed"])}
    validity["valid"] = bool(validity["VB2"] and validity["VB3"] and validity["VB4"] and validity["VB5_unit_tests"])
    out = {"ds": ds, "smoke": SMOKE["on"], "lambda": LAM, "seeds": list(seeds()), "hyper": v2.HYPER_V2,
           "hyper_overrides": hyper(), "folds": {k: {"held": h, "train": tr} for k, h, tr in fl},
           "per_device": per, "overlap_abs_diff": overlap, "validity": validity,
           "heldout_rows": {r["tag"]: {k: v for k, v in r.items() if k != "params"} for r in rows},
           "n_sim": {r["tag"]: r.get("n_sim") for r in rows}, "timing": timing,
           "runtime_s": round(time.time() - t_start, 1)}
    jdump(out, rd / "results.json")
    jdump({f"fixl1|{k}": c.to_json() for k, c in corrs.items()}, rd / "corrections.json")
    jdump({r["tag"]: r.get("params", {}) for r in rows}, rd / "heldout_cards.json")
    print(json.dumps({"ds": ds, "validity": {k: v for k, v in validity.items() if k != "VB4_dev_check"},
                      "mean": float(np.nanmean(list(per.values()))), "timing": {k: v for k, v in timing.items()
                                                                               if k != "train_seconds_per_member"},
                      "runtime_s": out["runtime_s"]}, indent=1, default=float), flush=True)
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("partA", "verify", "fixl1", "decide"))
    ap.add_argument("--ds", default="")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if a.workers > 10:
        ap.error("at most 10 workers")
    if a.stage == "partA":
        if a.ds not in m.IHP:
            ap.error(f"--ds must be one of {m.IHP}")
        stage_partA(a.ds, a.workers)
    elif a.stage == "verify":
        if a.ds not in m.REUSED:
            ap.error(f"--ds must be one of {tuple(m.REUSED)}")
        stage_verify(a.ds, a.workers)
    elif a.stage == "fixl1":
        if a.ds not in ("ihp233", "ihp343", "ihp398", "nist4", "nistrt"):
            ap.error("--ds must be one of ihp233, ihp343, ihp398, nist4, nistrt")
        stage_fixl1(a.ds, a.workers)
    else:
        import i23_1_report as rep
        rep.stage_decide()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
