#!/usr/bin/env python3
"""I18-2: the selected g v2 recipe on measured TU Delft 40-nm MOSFETs (I6-1
data), 4.13 K and 295.0 K, 45 DEV devices, leave-one-geometry-group-out;
BSIM4 clamp state C.  Accuracy part (Part B: i18_2_partb.py).

Pre-registration: out/research/explore/I18_2/prereg.md (prereg.json).

Stages (``--stage``):

  start   state-C starting records (both temperatures), V1a / V1b, b4check
          audit -> start/
  pens    45 x 45 forward-quadrant grids of the state-C starting cards -> pens/
  base    --temp T: C0_refit and M_bias v1 (state C); at 4 K also V1c
          (state-L refits of 2 devices vs I6-1 rows) -> base_<T>K/
  lodo    --temp T --lam L: sym_e1_l{L} -> lodo_<T>K_sym_e1_l{L}/
  inner   --temp T: nested lambda selection per outer fold -> inner_<T>K/
  full    --temp T: full-data lambda selection over 7 groups and the
          5-seed full-data model at lambda*_full -> full_<T>K/
  decide  validity, labels, tables -> decision.json

  PYTHONPATH=src .venv/bin/python scripts/research/explore/i18_2_tud.py --stage start --workers 6
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import tempfile
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research" / "explore"))

import i6_1_tudelft as drv6  # noqa: E402  (DEV devices, I6-1 options, train_task)
from cryoml.explore import i6_1_tudelft as m6  # noqa: E402
from cryoml.explore import i12_1_symm as v2  # noqa: E402
from cryoml.explore import i18_2_tud as m  # noqa: E402

EXP = ROOT / "out/research/explore"
OUT = EXP / "I18_2"
PREREG = OUT / "prereg.json"
I6 = EXP / "I6_1"
OPTS = dict(drv6.PREREG_OPTIONS)
VDD = m.VDD
UNIT_TESTS = ("tests.test_i18_2_tud", "tests.test_i12_1_symm", "tests.test_i15_2_ablate")
group_of = m6.group_of


def jload(p: Path):
    return json.loads(Path(p).read_text())


def jdump(obj, p: Path) -> None:
    m6.write_json(p, obj)


def base_dir(smoke: bool) -> Path:
    return OUT / "smoke" if smoke else OUT


_CACHE: dict = {}
_SMOKE = {"on": os.environ.get("CRYOML_I18_2_SMOKE") == "1"}


def devices() -> dict:
    if "dev" not in _CACHE:
        _CACHE["dev"] = {d.tag: d for d in drv6.dev_devices()}
    return _CACHE["dev"]


def i6_records(t: int) -> list:
    from cryoml.explore.e1_correction import load_devices
    return load_devices(I6 / "start" / f"records_{t}K")


def records(t: int) -> list:
    """State-C starting records, I6-1 load order."""
    key = ("recs", t)
    if key not in _CACHE:
        from cryoml.explore.e1_correction import load_devices
        recs = load_devices(base_dir(_SMOKE["on"]) / "start" / f"records_{t}K")
        if not _SMOKE["on"] and sorted(r.tag for r in recs) != sorted(devices()):
            raise RuntimeError("starting records differ from DEV")
        _CACHE[key] = recs
    return _CACHE[key]


def by_tag(t: int) -> dict:
    return {r.tag: r for r in records(t)}


def load_order(t: int) -> list[str]:
    return [r.tag for r in records(t)]


def groups(t: int) -> list[str]:
    return sorted({group_of(x) for x in load_order(t)})


def fold_train(t: int, g: str) -> list[str]:
    return [x for x in load_order(t) if group_of(x) != g]


def lambda_star(t: int) -> dict:
    return dict(jload(base_dir(_SMOKE["on"]) / f"inner_{t}K" / "inner.json")["lambda_star"])


# ---------------------------------------------------------------------------
# state-C simulation
# ---------------------------------------------------------------------------
def sim(dev, curves, temp_K: float, params: dict, state: str = "C") -> list[np.ndarray]:
    if state == "C":
        params = m.clamp_params(params)[0]
    elif state != "L":
        raise ValueError(state)
    return m6.simulate(dev, m6.model_block(dev.dev_type), curves, temp_K, params)


class ProblemC(m6.MeasProblem):
    """I6-1 ``MeasProblem`` with the b4check.c clamp written into every deck
    (state C); ``state='L'`` reproduces I6-1."""

    def __init__(self, dev, temp_k: int, box, state: str = "C"):
        super().__init__(dev, temp_k, box)
        self.state = state

    def simulate_params(self, params: dict, rs_ext_ohm: float | None = None) -> list[np.ndarray]:
        self.n_sim += 1
        p = m.clamp_params(params)[0] if self.state == "C" else params
        return m6.simulate(self.dev, self.card, self.meas, self.temp_stage_K, p,
                           self.rs_ext_ohm if rs_ext_ohm is None else rs_ext_ohm)


def b4check_lines(dev, t: int, params: dict) -> list[str]:
    """Warning / fatal lines of one fresh-process ``op`` (no clamp written)."""
    with tempfile.TemporaryDirectory(prefix="i18_2_chk_") as td_s:
        td = Path(td_s)
        card = td / "card.mod"
        card.write_text(m6.patch_card(m6.model_block(dev.dev_type), params))
        deck = td / "chk.sp"
        deck.write_text("".join(m6._header(dev, card, dev.temps_K[t], 0.0)) + "op\nquit\n.endc\n.end\n")
        r = subprocess.run([str(m6.NGSPICE), "-b", str(deck)], cwd=td, capture_output=True, text=True,
                           timeout=30)
    lines = (r.stdout + "\n" + r.stderr).splitlines()
    return [ln.strip() for ln in lines if ln.strip().lower().startswith(("warning", "fatal"))]


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------
def start_task(task):
    from cryoml.explore.e1_correction import score_device
    tag, t = task
    dev = devices()[tag]
    rec6 = {r.tag: r for r in i6_records(t)}[tag]
    params = dict(rec6.meta["params"])
    curves = m6.fit_curves(dev, t)
    simL = sim(dev, curves, dev.temps_K[t], params, "L")
    simC = sim(dev, curves, dev.temps_K[t], params, "C")
    recL = m6.to_record(dev, t, simL, params)
    recC = m6.to_record(dev, t, simC, params)
    stored = float(score_device(rec6, [c.sim for c in rec6.curves])["rrms"])
    rL = float(score_device(recL, [c.sim for c in recL.curves])["rrms"])
    rC = float(score_device(recC, [c.sim for c in recC.curves])["rrms"])
    _, ch = m.clamp_params(params)
    same = all(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True) for a, b in zip(simL, simC))
    return recC, {"tag": tag, "T_K": t, "a2": float(params.get("a2", m.CARD_A2)), "clamped": bool(ch),
                  "changes": {k: list(v) for k, v in ch.items()}, "rrms_stored": stored, "rrms_L": rL,
                  "rrms_C": rC, "v1a_diff": abs(rL - stored), "sims_C_equal_L": bool(same),
                  "b4check_unclamped": b4check_lines(dev, t, params),
                  "b4check_clamped": b4check_lines(dev, t, m.clamp_params(params)[0])}


def stage_start(workers: int, smoke: bool) -> dict:
    from cryoml.explore.e1_correction import save_devices
    rd = base_dir(smoke) / "start"
    rd.mkdir(parents=True, exist_ok=True)
    out = {"per_temperature": {}}
    for t in m.TEMPS:
        order = [r.tag for r in i6_records(t)]
        if smoke:
            order = [next(x for x in order if group_of(x) == g) for g in m.SMOKE_GROUPS]
        with get_context("spawn").Pool(workers) as pool:
            res = pool.map(start_task, [(x, t) for x in order], chunksize=1)
        save_devices(rd / f"records_{t}K", [r for r, _ in res])
        rows = {row["tag"]: row for _, row in res}
        unc = [x for x in rows.values() if m.A2_LOW <= x["a2"] <= m.A2_HIGH]
        out["per_temperature"][str(t)] = {
            "rows": rows, "n_clamped": int(sum(x["clamped"] for x in rows.values())),
            "v1a_max": max(x["v1a_diff"] for x in rows.values()),
            "v1b_n": len(unc), "v1b_all_equal": bool(all(x["sims_C_equal_L"] for x in unc)),
            "start_L": m6.aggregate({k: v["rrms_L"] for k, v in rows.items()}),
            "start_C": m6.aggregate({k: v["rrms_C"] for k, v in rows.items()}),
            "b4check_messages_unclamped": sorted({ln for x in rows.values() for ln in x["b4check_unclamped"]}),
            "b4check_messages_clamped": sorted({ln for x in rows.values() for ln in x["b4check_clamped"]})}
        s = out["per_temperature"][str(t)]
        print(f"{t} K: clamped {s['n_clamped']}/{len(rows)}; V1a max {s['v1a_max']:.3g}; V1b {s['v1b_n']} "
              f"unclamped equal {s['v1b_all_equal']}; start L {s['start_L']['all_device_mean']:.5f} -> C "
              f"{s['start_C']['all_device_mean']:.5f}; msgs {s['b4check_messages_unclamped'][:4]}", flush=True)
    out["v1a_max"] = max(v["v1a_max"] for v in out["per_temperature"].values())
    out["v1a_n"] = sum(len(v["rows"]) for v in out["per_temperature"].values())
    out["v1b_n"] = sum(v["v1b_n"] for v in out["per_temperature"].values())
    out["v1b_max"] = 0.0 if all(v["v1b_all_equal"] for v in out["per_temperature"].values()) else 1.0
    jdump(out, rd / "start.json")
    return out


# ---------------------------------------------------------------------------
# penalty grids
# ---------------------------------------------------------------------------
def grid_task(task):
    tag, t, out_dir = task
    from cryoml.explore.i5_1_ihp_real import MeasCurve
    dev = devices()[tag]
    rec = by_tag(t)[tag]
    pm = dev.dev_type == "pmos"
    z = np.zeros(1)
    curves = [MeasCurve("grid", "idvd", s["fixed"], s["vb"], np.asarray(s["sweep"], float), z, z, z, z, "")
              for s in m.grid_curve_specs(pm)]
    t0 = time.time()
    grid = np.vstack([np.asarray(x, float) for x in sim(dev, curves, dev.temps_K[t], dict(rec.meta["params"]))])
    np.savez_compressed(Path(out_dir) / f"grid_{t}K_{tag}.npz", i=grid, axis=m.pen_axis())
    s = -1.0 if pm else 1.0
    return {"tag": tag, "T_K": t, "shape": list(grid.shape), "n_nonfinite": int(np.sum(~np.isfinite(grid))),
            "i_fwd_corner": float(s * grid[-1, -1]), "seconds": round(time.time() - t0, 2)}


def load_pens(t: int, rd: Path | None = None) -> dict:
    rd = rd or base_dir(_SMOKE["on"]) / "pens"
    out = {}
    for tag in load_order(t):
        z = np.load(rd / f"grid_{t}K_{tag}.npz")
        out[tag] = v2.penalty_arrays(z["i"], devices()[tag].dev_type == "pmos", step=m.PEN_STEP, grid=z["axis"])
    return out


def stage_pens(workers: int, smoke: bool) -> dict:
    rd = base_dir(smoke) / "pens"
    rd.mkdir(parents=True, exist_ok=True)
    tasks = [(x, t, str(rd)) for t in m.TEMPS for x in load_order(t)]
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(grid_task, tasks, chunksize=1)
    out = {"rows": {f"{r['T_K']}|{r['tag']}": r for r in rows}, "n": len(rows)}
    kept, card_neg = {}, {}
    for t in m.TEMPS:
        pens = load_pens(t, rd)
        for x in pens:
            out["rows"][f"{t}|{x}"]["n_kept"] = pens[x]["n_kept"]
        kept[str(t)] = [pens[x]["n_kept"] for x in pens]
        card_neg[str(t)] = v2.card_negative_fraction(list(pens.values()))
    out["v4_ok"] = bool(len(rows) == (2 * len(m.SMOKE_GROUPS) if smoke else 90) and all(
        r["n_nonfinite"] == 0 and r["shape"] == [m.PEN_N, m.PEN_N] and r["i_fwd_corner"] > 0 for r in rows))
    out["card_negative_fraction"] = card_neg
    out["kept_range"] = {k: [min(v), max(v)] for k, v in kept.items()}
    jdump(out, rd / "pens.json")
    print(json.dumps({"v4_ok": out["v4_ok"], "kept": out["kept_range"], "card_neg": card_neg}, indent=1),
          flush=True)
    return out


# ---------------------------------------------------------------------------
# training / refit tasks
# ---------------------------------------------------------------------------
def train_sym_task(task):
    """(name, t, train_tags, seed, hyper, lam) -> (name, seed, corr, seconds)."""
    name, t, train_tags, seed, hyper, lam = task
    recs = by_tag(t)
    pens = None
    if lam:
        allp = load_pens(t)
        pens = [allp[x] for x in train_tags]
    t0 = time.time()
    corr = v2.fit_v2([recs[x] for x in train_tags], pens, lam, hyper={**hyper, "seeds": (seed,)},
                     inputs="sym_v2", eps=m.EPS, threads=1, arm=name.split("|", 1)[0], vdd=VDD)
    return name, seed, corr, round(time.time() - t0, 1)


def refit_task(task):
    """(label, corr, tag, t, opts, state) -> held-out refit row (I6-1 'corr' kind)."""
    from cryoml.explore.e1_correction import score_device
    from cryoml.ngspice_fit import polish
    label, corr, tag, t, opts, state = task
    t0 = time.time()
    n_to = m6.N_TIMEOUT
    dev = devices()[tag]
    rec = by_tag(t)[tag] if state == "C" else {r.tag: r for r in i6_records(t)}[tag]
    start = dict(rec.meta["params"])
    box = m6.base_box(dev)
    prob = ProblemC(dev, t, box, state)
    res = polish(m6.corrected_problem(prob, corr), box.params_to_local(start), max_nfev=opts["heldout_nfev"],
                 irls_rounds=opts["heldout_irls"], diff_step=opts["diff_step"])
    params = box.local_to_params(res["local"])
    new = m6.to_record(dev, t, prob.simulate_params(params), params)
    after = score_device(new, corr.apply(new))
    return {"label": label, "tag": tag, "T_K": t, "state": state, "rrms": after["rrms"], "failed": after["failed"],
            "n_sim": prob.n_sim, "params": params, "runtime_s": round(time.time() - t0, 2),
            "n_sim_timeout": m6.N_TIMEOUT - n_to}


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


def smoke_hyper(smoke: bool) -> tuple[dict, tuple, dict]:
    if smoke:
        return {"steps": 200}, (0,), dict(OPTS, heldout_nfev=20)
    return {}, tuple(v2.HYPER_V2["seeds"]), dict(OPTS)


# ---------------------------------------------------------------------------
# base: C0_refit and M_bias v1 (state C); V1c
# ---------------------------------------------------------------------------
def stage_base(t: int, workers: int, smoke: bool) -> dict:
    from cryoml.explore.e1_correction import Correction
    from cryoml.explore.i3_2_bias import correction_from_json, merge_seed_results, split_seed_tasks
    rd = base_dir(smoke) / f"base_{t}K"
    hyper_v1 = {"seeds": (0,), "steps": 200} if smoke else {}
    _, _, opts = smoke_hyper(smoke)
    recs = by_tag(t)
    gs = groups(t)
    unit = run_unit_tests(rd)
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        tasks = [("M_bias", [recs[x] for x in fold_train(t, g)]) for g in gs]
        expanded, owner = split_seed_tasks(tasks, hyper_v1)
        res = pool.map(drv6.train_task, [("bias", a, r, hp) for a, r, hp in expanded], chunksize=1)
        corrs = {f"M_bias|{g}": c for g, c in zip(gs, merge_seed_results(len(tasks), owner, res))}
        t_train = time.time() - t0
        rt = [("C0_refit", Correction("C0"), x, t, opts, "C") for x in load_order(t)]
        rt += [("M_bias", corrs[f"M_bias|{group_of(x)}"], x, t, opts, "C") for x in load_order(t)]
        v1c = []
        if t == 4 and not smoke:
            stored = jload(I6 / "lodo_4K/corrections.json")
            v1c = [("V1c_C0_refit", Correction("C0"), x, t, opts, "L") for x in m.V1C_TAGS]
            v1c += [("V1c_M_bias", correction_from_json(stored[f"M_bias|{group_of(x)}"]), x, t, opts, "L")
                    for x in m.V1C_TAGS]
        t1 = time.time()
        rows = pool.map(refit_task, rt + v1c, chunksize=1)
        t_refit = time.time() - t1
    per = {"C0_refit": {}, "M_bias": {}}
    v1 = {}
    stored_rows = jload(I6 / f"lodo_{t}K/results.json")["per_device"]
    for r in rows:
        if r["label"].startswith("V1c_"):
            arm = r["label"][4:]
            v1[f"{arm}|{r['tag']}"] = {"rrms": r["rrms"], "stored": stored_rows[arm][r["tag"]],
                                       "diff": abs(r["rrms"] - stored_rows[arm][r["tag"]])}
        else:
            per[r["label"]][r["tag"]] = r["rrms"]
    out = {"T_K": t, "smoke": smoke, "folds": gs, "opts": opts, "hyper_overrides": hyper_v1, "per_device": per,
           "train_order": {g: fold_train(t, g) for g in gs}, "v1c": v1,
           "v1c_max": max([x["diff"] for x in v1.values()], default=float("nan")), "v1c_n": len(v1),
           "unit_tests_passed": bool(unit["passed"]),
           "i6_1_state_L_rows": {a: stored_rows[a] for a in ("C0_refit", "M_bias")},
           "rows": {f"{r['label']}|{r['tag']}": {k: v for k, v in r.items() if k != "params"} for r in rows},
           "timing": {"train_s": round(t_train, 1), "refit_s": round(t_refit, 1)}}
    jdump(out, rd / "results.json")
    jdump({k: c.to_json() for k, c in corrs.items()}, rd / "corrections.json")
    jdump({f"{r['label']}|{r['tag']}": r["params"] for r in rows}, rd / "heldout_cards.json")
    print(json.dumps({"v1c": v1, "timing": out["timing"],
                      "mean": {a: float(np.mean(list(v.values()))) for a, v in per.items()}}, indent=1), flush=True)
    return out


# ---------------------------------------------------------------------------
# lodo (one lambda)
# ---------------------------------------------------------------------------
def stage_lodo(t: int, lam: float, workers: int, smoke: bool) -> dict:
    name = m.variant(lam)
    rd = base_dir(smoke) / f"lodo_{t}K_{name}"
    hyper, seeds, opts = smoke_hyper(smoke)
    gs = groups(t)
    names = [f"{name}|{g}" for g in gs]
    tasks = [(f"{name}|{g}", t, fold_train(t, g), s, hyper, float(lam)) for g in gs for s in seeds]
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        trained = pool.map(train_sym_task, tasks, chunksize=1)
        t_train = time.time() - t0
        corrs, secs = merge_sym(trained, names)
        print(f"lodo {t} K {name}: training {len(tasks)} seed tasks {t_train:.0f} s", flush=True)
        t1 = time.time()
        rows = pool.map(refit_task, [(name, corrs[f"{name}|{group_of(x)}"], x, t, opts, "C")
                                     for x in load_order(t)], chunksize=1)
        t_refit = time.time() - t1
    per = {r["tag"]: r["rrms"] for r in rows}
    res = {"variant": name, "T_K": t, "inputs": "sym_v2", "eps": m.EPS, "lambda": float(lam), "vdd": VDD,
           "smoke": smoke, "hyper": v2.HYPER_V2, "hyper_overrides": hyper, "seeds": list(seeds), "opts": opts,
           "folds": gs, "train_order": {g: fold_train(t, g) for g in gs}, "per_device": per,
           "mean": float(np.mean(list(per.values()))),
           "heldout_rows": {r["tag"]: {k: v for k, v in r.items() if k != "params"} for r in rows},
           "train_seconds_per_member": secs,
           "timing": {"train_s": round(t_train, 1), "refit_s": round(t_refit, 1), "n_seed_tasks": len(tasks)}}
    jdump(res, rd / "results.json")
    jdump({n: c.to_json() for n, c in corrs.items()}, rd / "corrections.json")
    jdump({r["tag"]: r["params"] for r in rows}, rd / "heldout_cards.json")
    print(json.dumps({"variant": name, "T": t, "mean": res["mean"], "timing": res["timing"]}, indent=1), flush=True)
    return res


# ---------------------------------------------------------------------------
# nested selection (outer folds) and full-data selection
# ---------------------------------------------------------------------------
def selection(t: int, plans: dict, workers: int, smoke: bool) -> dict:
    """plans: {outer key: list of training groups}.  Per outer key and inner
    fold, one member per lambda on the training groups outside the inner
    fold; every device of the inner fold refit with it."""
    hyper, _, opts = smoke_hyper(smoke)
    order = load_order(t)
    plan, tasks = {}, []
    for o, tr_groups in plans.items():
        folds = m.inner_group_folds(tr_groups)
        plan[o] = {"train_groups": tr_groups, "inner_folds": folds}
        for j, held in enumerate(folds):
            inner_train = [x for x in order if group_of(x) in tr_groups and group_of(x) not in held]
            for lam in m.GRID_LAMBDA:
                tasks.append((f"{o}|{j}|{lam:g}", t, inner_train, 0, hyper, float(lam)))
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        trained = pool.map(train_sym_task, tasks, chunksize=1)
        t_train = time.time() - t0
        print(f"selection {t} K: training {len(tasks)} members {t_train:.0f} s", flush=True)
        corr = {name: c for name, _, c, _ in trained}
        secs = {name: s for name, _, _, s in trained}
        rtasks = []
        for o in plans:
            for j, held in enumerate(plan[o]["inner_folds"]):
                for lam in m.GRID_LAMBDA:
                    key = f"{o}|{j}|{lam:g}"
                    rtasks += [(key, corr[key], x, t, opts, "C") for x in order if group_of(x) in held]
        t1 = time.time()
        rows = pool.map(refit_task, rtasks, chunksize=1)
        t_refit = time.time() - t1
    res = {"T_K": t, "smoke": smoke, "grid_lambda": list(m.GRID_LAMBDA), "eps": m.EPS, "seed": 0, "opts": opts,
           "hyper_overrides": hyper, "plan": plan, "outer": {},
           "timing": {"train_s": round(t_train, 1), "refit_s": round(t_refit, 1), "n_train": len(tasks),
                      "n_refit": len(rtasks)}, "train_seconds": secs}
    complete = len(tasks) == len(plans) * m.N_INNER * len(m.GRID_LAMBDA)
    for o in plans:
        rr = {}
        for lam in m.GRID_LAMBDA:
            rr[f"{lam:g}"] = {r["tag"]: r["rrms"] for r in rows
                              if r["label"].startswith(f"{o}|") and r["label"].endswith(f"|{lam:g}")}
        scores = {lam: m.inner_score(list(rr[f"{lam:g}"].values())) for lam in m.GRID_LAMBDA}
        n_exp = sum(1 for x in order if group_of(x) in plans[o])
        res["outer"][o] = {"inner_rrms": rr, "scores": {f"{k:g}": v for k, v in scores.items()},
                           "n_scored": {f"{lam:g}": len(rr[f"{lam:g}"]) for lam in m.GRID_LAMBDA},
                           "n_expected": n_exp, "lambda_star": m.select_lambda(scores)}
        complete = complete and all(n == n_exp for n in res["outer"][o]["n_scored"].values())
    res["lambda_star"] = {o: res["outer"][o]["lambda_star"] for o in plans}
    res["complete"] = bool(complete)
    return res, corr


def stage_inner(t: int, workers: int, smoke: bool) -> dict:
    rd = base_dir(smoke) / f"inner_{t}K"
    gs = groups(t)
    res, corr = selection(t, {g: [h for h in gs if h != g] for g in gs}, workers, smoke)
    jdump(res, rd / "inner.json")
    with gzip.open(rd / "inner_corrections.json.gz", "wt") as fh:
        json.dump({k: c.to_json() for k, c in corr.items()}, fh)
    print(json.dumps({"lambda_star": res["lambda_star"], "complete": res["complete"], "timing": res["timing"]},
                     indent=1), flush=True)
    return res


def stage_full(t: int, workers: int, smoke: bool) -> dict:
    rd = base_dir(smoke) / f"full_{t}K"
    gs = groups(t)
    res, _ = selection(t, {"full": list(gs)}, workers, smoke)
    lam = float(res["lambda_star"]["full"])
    hyper, seeds, _ = smoke_hyper(smoke)
    name = f"full|{m.variant(lam)}"
    tasks = [(name, t, load_order(t), s, hyper, lam) for s in seeds]
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        trained = pool.map(train_sym_task, tasks, chunksize=1)
    corrs, secs = merge_sym(trained, [name])
    res.update({"lambda_star_full": lam, "full_key": name, "full_train_s": round(time.time() - t0, 1),
                "full_seconds_per_member": secs[name], "full_train_order": load_order(t)})
    jdump(res, rd / "full.json")
    jdump({name: corrs[name].to_json()}, rd / "corrections.json")
    print(json.dumps({"lambda_star_full": lam, "complete": res["complete"], "timing": res["timing"]}, indent=1),
          flush=True)
    return res


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------
def neg_fractions(corr_of, t: int, pens: dict, **kw) -> dict:
    tr_gm, tr_gds, ho_gm, ho_gds = [], [], [], []
    for g in groups(t):
        c = corr_of(g)
        a = v2.penalty_value_numpy(c, [pens[x] for x in fold_train(t, g)], **kw)
        b = v2.penalty_value_numpy(c, [pens[x] for x in load_order(t) if group_of(x) == g], **kw)
        tr_gm.append(a["frac_neg_gm"])
        tr_gds.append(a["frac_neg_gds"])
        ho_gm.append(b["frac_neg_gm"])
        ho_gds.append(b["frac_neg_gds"])
    return {"train_mean_gm": float(np.mean(tr_gm)), "train_mean_gds": float(np.mean(tr_gds)),
            "heldout_mean_gm": float(np.mean(ho_gm)), "heldout_mean_gds": float(np.mean(ho_gds))}


def sym_corr(t: int, lam: float, g: str, B: Path):
    name = m.variant(lam)
    return v2.SymCorrection.from_json(jload(B / f"lodo_{t}K_{name}" / "corrections.json")[f"{name}|{g}"])


def stage_decide(smoke: bool = False) -> dict:
    from cryoml.explore.i3_2_bias import paired_ratio
    B = base_dir(smoke)
    start = jload(B / "start/start.json")
    pj = jload(B / "pens/pens.json")
    preds = jload(PREREG)["predictions"]
    per_t, v3_rows, inner_ok, full_ok = {}, {}, True, True
    v1c = {"max": float("nan"), "n": 0}
    unit_ok = True
    for t in m.TEMPS:
        base = jload(B / f"base_{t}K/results.json")
        unit_ok = unit_ok and base["unit_tests_passed"]
        if t == 4:
            v1c = {"max": base["v1c_max"], "n": base["v1c_n"]}
        tags = sorted(base["per_device"]["C0_refit"])
        cl = {x: group_of(x) for x in tags}
        inner = jload(B / f"inner_{t}K/inner.json")
        full = jload(B / f"full_{t}K/full.json")
        inner_ok = inner_ok and bool(inner["complete"]) and len(inner["lambda_star"]) == len(groups(t))
        full_ok = full_ok and bool(full["complete"])
        rows = {"C0_refit": base["per_device"]["C0_refit"], "M_bias_v1": base["per_device"]["M_bias"]}
        for lam in m.GRID_LAMBDA:
            f = B / f"lodo_{t}K_{m.variant(lam)}/results.json"
            rows[m.variant(lam)] = jload(f)["per_device"] if f.exists() else {x: float("nan") for x in tags}
        lam_star = {g: float(v) for g, v in inner["lambda_star"].items()}
        rows["selected"] = {x: rows[m.variant(lam_star[group_of(x)])][x] for x in tags}
        for a, r in rows.items():
            v3_rows[f"{t}|{a}"] = len(r)
        n_failed = int(sum(not np.isfinite(rows[a][x]) for a in ("selected", "C0_refit") for x in tags))
        primary = paired_ratio(rows["selected"], rows["C0_refit"], clusters=cl)
        secondary = paired_ratio(rows["selected"], rows["M_bias_v1"], clusters=cl)
        table = {}
        for a, r in rows.items():
            table[a] = dict(m6.aggregate(r))
            if a != "C0_refit":
                table[a]["gmr_vs_C0_refit_cluster"] = paired_ratio(r, rows["C0_refit"], clusters=cl)
                table[a]["gmr_vs_C0_refit_device"] = paired_ratio(r, rows["C0_refit"])
            if a != "M_bias_v1":
                table[a]["gmr_vs_v1_cluster"] = paired_ratio(r, rows["M_bias_v1"], clusters=cl)
                table[a]["gmr_vs_v1_device"] = paired_ratio(r, rows["M_bias_v1"])
        per_group = {g: {a: m6.aggregate({x: r[x] for x in tags if group_of(x) == g}) for a, r in rows.items()}
                     for g in groups(t)}
        pens = load_pens(t, B / "pens")
        bc = jload(B / f"base_{t}K/corrections.json")
        from cryoml.explore.i3_2_bias import correction_from_json
        negs = {}
        negs["M_bias_v1"] = neg_fractions(lambda g: correction_from_json(bc[f"M_bias|{g}"]), t, pens, vdd_v1=VDD)
        for lam in m.GRID_LAMBDA:
            if (B / f"lodo_{t}K_{m.variant(lam)}/corrections.json").exists():
                negs[m.variant(lam)] = neg_fractions(lambda g, lam=lam: sym_corr(t, lam, g, B), t, pens)
        negs["selected"] = neg_fractions(lambda g: sym_corr(t, lam_star[g], g, B), t, pens)
        negs["card_only"] = pj["card_negative_fraction"][str(t)]
        g_v1_c0 = paired_ratio(rows["M_bias_v1"], rows["C0_refit"], clusters=cl)
        st = str(t)
        pred = {"ln GMR(selected, C0_refit)": m.sign_check(primary["ratio"], preds["ln GMR(selected, C0_refit)"][st]),
                "ln GMR(selected, M_bias v1)": m.sign_check(secondary["ratio"],
                                                            preds["ln GMR(selected, M_bias v1)"][st]),
                "ln GMR(M_bias v1, C0_refit)": m.sign_check(g_v1_c0["ratio"],
                                                            preds["ln GMR(M_bias v1, C0_refit)"][st])}
        cnt = {}
        for g in groups(t):
            cnt[f"{lam_star[g]:g}"] = cnt.get(f"{lam_star[g]:g}", 0) + 1
        stored = jload(I6 / f"lodo_{t}K/results.json")["per_device"]
        per_t[st] = {"primary_gmr_selected_vs_C0_refit": primary, "secondary_gmr_selected_vs_v1": secondary,
                     "gmr_v1_vs_C0_refit": g_v1_c0, "n_failed_selected_C0": n_failed,
                     "prediction_signs_correct": pred, "lambda_star": lam_star, "lambda_star_counts": cnt,
                     "lambda_star_full": full["lambda_star_full"], "selected_rows": rows["selected"], "rows": rows,
                     "table": table, "per_group": per_group, "negative_fractions": negs,
                     "inner_scores": {o: v["scores"] for o, v in inner["outer"].items()},
                     "full_scores": full["outer"]["full"]["scores"],
                     "state_L_I6_1": {a: m6.aggregate(stored[a]) for a in ("C0_refit", "M_bias")},
                     "start": {"L": start["per_temperature"][st]["start_L"],
                               "C": start["per_temperature"][st]["start_C"],
                               "n_clamped": start["per_temperature"][st]["n_clamped"]}}
    validity = m.decide_validity({"v1a_max": start["v1a_max"], "v1a_n": start["v1a_n"], "v1b_max": start["v1b_max"],
                                  "v1b_n": start["v1b_n"], "v1c_max": v1c["max"], "v1c_n": v1c["n"],
                                  "v2_unit_tests_pass": unit_ok, "v3_rows": v3_rows,
                                  "v3_inner_complete": inner_ok, "v3_full_complete": full_ok,
                                  "v4_grids_ok": bool(pj["v4_ok"])})
    for st, d in per_t.items():
        d.update(m.decide_outcome(validity, d["primary_gmr_selected_vs_C0_refit"], d["n_failed_selected_C0"],
                                  d["secondary_gmr_selected_vs_v1"]))
    k_ratio = None
    if all(st in per_t for st in ("4", "295")):
        a, b = per_t["4"]["primary_gmr_selected_vs_C0_refit"]["ratio"], per_t["295"]["primary_gmr_selected_vs_C0_refit"]["ratio"]
        k_ratio = a / b
    out = {"prereg": str(PREREG.relative_to(ROOT)), "validity": validity, "per_temperature": per_t,
           "contrast_K_selected": k_ratio, "predictions": preds, "part_B": "see partB_<T>K/decision.json"}
    jdump(out, B / "decision.json")
    print(json.dumps({"validity": validity,
                      "per_t": {st: {"outcome": d.get("outcome"), "secondary": d.get("secondary"),
                                     "primary": {k: d["primary_gmr_selected_vs_C0_refit"][k]
                                                 for k in ("ratio", "ci_low", "ci_high", "wins")},
                                     "secondary_gmr": {k: d["secondary_gmr_selected_vs_v1"][k]
                                                       for k in ("ratio", "ci_low", "ci_high", "wins")},
                                     "lambda_star_counts": d["lambda_star_counts"],
                                     "lambda_star_full": d["lambda_star_full"],
                                     "pred": d["prediction_signs_correct"],
                                     "table": {a: (round(v["all_device_mean"], 5), round(v["median"], 5))
                                               for a, v in d["table"].items()}}
                                for st, d in per_t.items()}}, indent=1, default=float), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("start", "pens", "base", "lodo", "inner", "full", "decide"))
    ap.add_argument("--temp", type=int, default=None)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true",
                    help="1 device of 2 groups, 200 steps, 1 seed, refit max_nfev 20 (code check only)")
    args = ap.parse_args()
    if args.smoke:
        os.environ["CRYOML_I18_2_SMOKE"] = "1"      # spawned workers read the smoke directories
        _SMOKE["on"] = True
    t0 = time.time()
    if args.stage == "start":
        stage_start(args.workers, args.smoke)
    elif args.stage == "pens":
        stage_pens(args.workers, args.smoke)
    elif args.stage == "base":
        stage_base(args.temp, args.workers, args.smoke)
    elif args.stage == "lodo":
        stage_lodo(args.temp, args.lam, args.workers, args.smoke)
    elif args.stage == "inner":
        stage_inner(args.temp, args.workers, args.smoke)
    elif args.stage == "full":
        stage_full(args.temp, args.workers, args.smoke)
    else:
        stage_decide(args.smoke)
    print(f"stage {args.stage} done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
