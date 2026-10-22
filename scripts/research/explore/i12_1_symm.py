#!/usr/bin/env python3
"""I12-1: bias-only correction g v2 (d/s-symmetric smooth inputs, static
element via a charge-free card replica, gm/gds penalty), sky130 77 K, all 18
devices: LODO accuracy and deployment checks in ngspice-41 + OSDI.

Pre-registration: out/research/explore/I12_1/prereg.md (prereg.json).

Stages (``--stage``):

  grids     per device, card-only forward-quadrant current grid of the
            starting card (direct43_r3_fd), 25 mV, canonical ngspice-41;
            penalty inputs (``i12_1_symm.penalty_arrays``) -> grids/pen.npz
  calib     penalty value P and negative-derivative fractions of the stored
            M_bias v1 corrections on the grids (information for the weight)
  validity  V1 trainer reproduction (abs inputs, lambda 0) of 2 stored fold
            corrections; V2 held-out refit reproduction of 2 stored M_bias rows
  lodo      18 folds + full-data model, 5 seeds each (one pool task per
            fold and seed); held-out refit with the fold g in the loop
            (max_nfev 400, irls 2, diff_step 0.01); strict paper RRMS
  deploy    compile (OpenVAF 23.5.0), M2 cards (refit with the full-data g),
            q12 (grids / exchanged terminals / AC / 1 mV sweeps), q3 circuits,
            report
  decide    validity + accuracy + deployment -> decision.json

  PYTHONPATH=src .venv/bin/python -I scripts/research/explore/i12_1_symm.py --stage grids --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(ROOT / "scripts" / "research" / "explore"))

from cryoml.explore import i11_1_deploy as m1  # noqa: E402
from cryoml.explore import i12_1_symm as m  # noqa: E402

EXP = ROOT / "out/research/explore"
# I13-1: CRYOML_I3_2_ALL18 points the stored v1 / C0_refit / T1_inloop rows at a rerun
ALL18 = Path(os.environ.get("CRYOML_I3_2_ALL18", str(EXP / "I3_2/all18")))
OUT = EXP / "I12_1"
PREREG = OUT / "prereg.json"
START = ROOT / "out/research/polished/direct43_r3_fd/real/summary.json"
NG_CANON = "/Users/anrunw/cryo-ng41/arm64/bin/ngspice"
REFIT_OPTS = {"heldout_nfev": 400, "heldout_irls": 2, "diff_step": 0.01}


def jload(p):
    return json.loads(Path(p).read_text())


def jdump(obj, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, default=float) + "\n")


def devices() -> dict:
    from cryoml.card_eval import device_by_tag
    from cryoml.cold_start import device_bins
    bins = device_bins()
    out = {}
    for tag in sorted(bins):
        d = device_by_tag(tag)
        out[tag] = m1.Dev(tag, d.dev_type, float(d.L_um), float(d.W_um), int(bins[tag]))
    return out


def start_cards() -> dict:
    return {t: dict(v["params"]) for t, v in jload(START)["devices"].items()}


# ---------------------------------------------------------------------------
# grids
# ---------------------------------------------------------------------------
def grid_task(task):
    tag, params = task
    dev = devices()[tag]
    s = -1.0 if dev.dev_type == "pmos" else 1.0
    with m1.tempdir("cryoml_i12g_") as td_s:
        td = Path(td_s)
        lib = m1.patched_library(td, {dev.dev_type: [(dev.bin, params)]})
        f = td / "grid.dat"
        ctrl = [f"dc VD 0 {s * 1.85:.6g} {s * m.PEN_STEP:.6g} VG 0 {s * 1.85:.6g} {s * m.PEN_STEP:.6g}",
                f"wrdata {f.as_posix()} -i(VD)"]
        r = m1.run_deck(m1.deck(f"pen {tag}", lib, m1.device_body(dev, False), ctrl, None),
                        td, "g.sp", binary=NG_CANON, timeout_s=300)
        rows = m1.read_wrdata(f)
    n = m.PEN_N
    grid = rows[:, 1].reshape(n, n) if rows.shape[0] == n * n else np.full((n, n), np.nan)
    return tag, grid, r["msgs"]


def stage_grids(workers: int, tags=None) -> dict:
    cards = start_cards()
    tags = sorted(devices()) if tags is None else tags
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        out = pool.map(grid_task, [(t, cards[t]) for t in tags], chunksize=1)
    grids = {t: g for t, g, _ in out}
    msgs = {t: ms for t, _, ms in out}
    (OUT / "grids").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "grids" / "card_grids.npz", **grids)
    devs = devices()
    info = {}
    for t in tags:
        p = m.penalty_arrays(grids[t], devs[t].dev_type == "pmos")
        info[t] = {"n_kept": p["n_kept"], "n_grid": p["n_grid"], "msgs": msgs[t],
                   "nonfinite": int(np.sum(~np.isfinite(grids[t])))}
    res = {"wall_s": round(time.time() - t0, 1), "step_V": m.PEN_STEP, "binary": NG_CANON,
           "cards": str(START.relative_to(ROOT)), "devices": info}
    jdump(res, OUT / "grids" / "grids.json")
    print(f"grids: {len(tags)} devices, {res['wall_s']} s, kept points "
          f"{sum(v['n_kept'] for v in info.values())}", flush=True)
    return res


def load_pens() -> dict:
    devs = devices()
    with np.load(OUT / "grids" / "card_grids.npz") as z:
        return {t: m.penalty_arrays(z[t], devs[t].dev_type == "pmos") for t in z.files}


# ---------------------------------------------------------------------------
# calib (stored v1 corrections; information only)
# ---------------------------------------------------------------------------
def stage_calib() -> dict:
    from cryoml.explore.i3_2_bias import correction_from_json
    pens = load_pens()
    tags = sorted(pens)
    corrs = jload(ALL18 / "corrections.json")
    full = correction_from_json(corrs["full|M_bias"])
    out = {"full_v1_all18": m.penalty_value_numpy(full, [pens[t] for t in tags]),
           "card_only": m.card_negative_fraction([pens[t] for t in tags])}
    fold = {}
    for t in tags:
        c = correction_from_json(corrs[f"M_bias|{t}"])
        fold[t] = m.penalty_value_numpy(c, [pens[u] for u in tags if u != t])["P"]
    out["fold_v1_P_on_training_grids"] = fold
    out["fold_v1_P_mean"] = float(np.mean(list(fold.values())))
    jdump(out, OUT / "calib" / "calib_v1.json")
    print(json.dumps({k: v for k, v in out.items() if k != "fold_v1_P_on_training_grids"},
                     indent=1, default=float))
    return out


# ---------------------------------------------------------------------------
# records, corrections
# ---------------------------------------------------------------------------
def load_records() -> dict:
    from i3_2_bias import load_records_all18
    recs, _ = load_records_all18()
    return {r.tag: r for r in recs}


def load_order() -> list[str]:
    """Tags in the record load order of ``load_records_all18`` (amendment A2)."""
    from i3_2_bias import load_records_all18
    recs, _ = load_records_all18()
    return [r.tag for r in recs]


def stored_v1(key: str):
    from cryoml.explore.i3_2_bias import correction_from_json
    return correction_from_json(jload(ALL18 / "corrections.json")[key])


def stored_rows() -> dict:
    return jload(ALL18 / "results.json")["per_device"]


def train_task(task):
    """One ensemble member: (name, train_tags, seed, hyper, lam, inputs)."""
    name, train_tags, seed, hyper, lam, inputs = task
    recs = load_records()
    train = [recs[t] for t in train_tags]
    pens = None
    if lam:
        allp = load_pens()
        pens = [allp[t] for t in train_tags]
    trace = []
    t0 = time.time()
    corr = m.fit_v2(train, pens, lam, hyper={**hyper, "seeds": (seed,)}, inputs=inputs,
                    threads=1, trace=trace)
    return name, seed, corr, trace, round(time.time() - t0, 1), list(train_tags)


def heldout(task):
    from i2_1_joint import heldout_task
    row, _ = heldout_task(task)
    return row


def merge(results, names) -> dict:
    by = {n: [] for n in names}
    traces = {n: {} for n in names}
    secs = {n: [] for n in names}
    for name, seed, corr, trace, sec, _ in sorted(results, key=lambda r: (r[0], r[1])):
        by[name].append(corr)
        traces[name][seed] = trace
        secs[name].append(sec)
    return ({n: m.merge_seed_corrections(v) for n, v in by.items()}, traces, secs)


# ---------------------------------------------------------------------------
# validity
# ---------------------------------------------------------------------------
def stage_validity(workers: int, rd: Path, smoke: bool) -> dict:
    from cryoml.explore.e1_correction import make_batch
    recs = load_records()
    order = load_order()
    keys = m.select_v1_keys(list(jload(ALL18 / "corrections.json")))
    hyper = {"steps": 200} if smoke else {}
    seeds = (0,) if smoke else m.HYPER_V2["seeds"]
    tasks = [(k, m.fold_train_tags(order, k.split("|", 1)[1]), s, hyper, 0.0, "abs_v1")
             for k in keys for s in seeds]
    v2_rows = m.select_v2_rows(stored_rows()["M_bias"])
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        trained = pool.map(train_task, tasks, chunksize=1)
        refit_tasks = [("M_bias_v1_rerun", None, stored_v1(f"M_bias|{t}"), recs[t],
                        dict(REFIT_OPTS, heldout_nfev=20) if smoke else REFIT_OPTS)
                       for t in v2_rows]
        rows = pool.map(heldout, refit_tasks, chunksize=1)
    corrs, _, _ = merge(trained, keys)
    v1 = {}
    for k in keys:
        held = k.split("|", 1)[1]
        train = [recs[t] for t in m.fold_train_tags(order, held)]
        phi = make_batch(train).phi
        g_old = stored_v1(k).g(phi)
        g_new = corrs[k].g_x(m.batch_inputs(train, "abs_v1"))
        v1[k] = {"max_abs_g_diff": float(np.max(np.abs(g_new - g_old))), "n_points": int(len(phi)),
                 "n_members": len(corrs[k].nets)}
    v2 = {r["tag"]: {"rrms": r["rrms"], "stored": v2_rows[r["tag"]],
                     "abs_diff": abs(r["rrms"] - v2_rows[r["tag"]])} for r in rows}
    out = {"smoke": smoke, "V1": v1, "V2": v2,
           "v1_max_g_diff": max(v["max_abs_g_diff"] for v in v1.values()),
           "v2_max_rrms_diff": max(v["abs_diff"] for v in v2.values()),
           "wall_s": round(time.time() - t0, 1)}
    jdump(out, rd / "validity.json")
    print(json.dumps({k: out[k] for k in ("v1_max_g_diff", "v2_max_rrms_diff", "wall_s")}), flush=True)
    return out


# ---------------------------------------------------------------------------
# lodo
# ---------------------------------------------------------------------------
def stage_lodo(workers: int, rd: Path, smoke: bool, folds: list[str] | None = None) -> dict:
    from cryoml.explore.i2_1_joint import corrected_score
    from cryoml.explore.i3_2_bias import paired_ratio
    from e1_lodo import aggregate
    recs = load_records()
    tags = sorted(recs)
    folds = folds or tags
    hyper = {"steps": 200} if smoke else {}
    seeds = (0,) if smoke else m.HYPER_V2["seeds"]
    opts = dict(REFIT_OPTS, heldout_nfev=20) if smoke else dict(REFIT_OPTS)
    names = [f"M_bias_v2|{t}" for t in folds] + ["full|M_bias_v2"]
    order = load_order()
    train_of = {f"M_bias_v2|{t}": m.fold_train_tags(order, t) for t in folds}
    train_of["full|M_bias_v2"] = m.fold_train_tags(order, None)
    tasks = [(n, train_of[n], s, hyper, m.LAMBDA, "sym_v2") for n in names for s in seeds]
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        trained = pool.map(train_task, tasks, chunksize=1)
        t_train = time.time() - t0
        corrs, traces, secs = merge(trained, names)
        print(f"lodo: training {len(tasks)} seed tasks {t_train:.0f} s", flush=True)
        t1 = time.time()
        rows = pool.map(heldout, [("M_bias_v2", None, corrs[f"M_bias_v2|{t}"], recs[t], opts)
                                  for t in folds], chunksize=1)
        t_refit = time.time() - t1
    v2 = {r["tag"]: r["rrms"] for r in rows}
    st = stored_rows()
    scores = {"M_bias_v2": v2, "C0_refit": st["C0_refit"], "M_bias_v1": st["M_bias"],
              "T1_inloop": st["T1_inloop"]}
    full18 = len(folds) == 18 and not smoke
    if full18:
        sel = {k: m.select_primary_rows(v, tags) for k, v in scores.items()}
    else:
        sel = {k: {t: v[t] for t in folds} for k, v in scores.items()}
    summary = {k: aggregate(v) for k, v in sel.items()} if full18 else {
        k: {"mean": float(np.mean(list(v.values())))} for k, v in sel.items()}
    comparisons = {"A1_M_bias_v2_vs_C0_refit": paired_ratio(sel["M_bias_v2"], sel["C0_refit"]),
                   "A2_M_bias_v2_vs_M_bias_v1": paired_ratio(sel["M_bias_v2"], sel["M_bias_v1"]),
                   "M_bias_v2_vs_T1_inloop": paired_ratio(sel["M_bias_v2"], sel["T1_inloop"]),
                   "M_bias_v1_vs_C0_refit": paired_ratio(sel["M_bias_v1"], sel["C0_refit"])}
    n_failed = int(sum(not np.isfinite(v) for v in sel["M_bias_v2"].values()))
    pens = load_pens()
    pen_stats = {}
    for n in names:
        tr = train_of[n]
        pv = m.penalty_value_numpy(corrs[n], [pens[t] for t in tr])
        pen_stats[n] = {"P_train": pv["P"], "frac_neg_gm_train": pv["frac_neg_gm"],
                        "frac_neg_gds_train": pv["frac_neg_gds"]}
        if n.startswith("M_bias_v2|"):
            ho = m.penalty_value_numpy(corrs[n], [pens[n.split("|", 1)[1]]])
            pen_stats[n].update(P_heldout_start=ho["P"], frac_neg_gds_heldout_start=ho["frac_neg_gds"],
                                frac_neg_gm_heldout_start=ho["frac_neg_gm"])
    posthoc = {t: corrected_score(corrs[f"M_bias_v2|{t}"], recs[t]) for t in folds}
    result = {"smoke": smoke, "preregistered_config": bool(full18), "prereg": str(PREREG.relative_to(ROOT)),
              "lambda": m.LAMBDA, "eps": m.EPS, "hyper": m.HYPER_V2, "hyper_overrides": hyper,
              "seeds": list(seeds), "opts": opts, "folds": folds, "train_order": train_of,
              "summary": summary, "comparisons": comparisons, "n_failed_M_bias_v2": n_failed,
              "per_device": sel, "posthoc_no_refit": posthoc, "penalty": pen_stats,
              "heldout_rows": {r["tag"]: {k: v for k, v in r.items() if k != "params"} for r in rows},
              "train_seconds_per_member": secs, "traces": traces,
              "timing": {"train_s": round(t_train, 1), "refit_s": round(t_refit, 1),
                         "n_seed_tasks": len(tasks)}}
    if full18:
        result["accuracy_decision"] = m.decide_accuracy(
            comparisons["A1_M_bias_v2_vs_C0_refit"], comparisons["A2_M_bias_v2_vs_M_bias_v1"], n_failed)
    jdump(result, rd / "results.json")
    jdump({n: c.to_json() for n, c in corrs.items()}, rd / "corrections.json")
    jdump({r["tag"]: r["params"] for r in rows}, rd / "heldout_cards.json")
    print(json.dumps({"summary": summary, "comparisons": {k: {kk: v.get(kk) for kk in (
        "ratio", "ci_low", "ci_high", "wins", "n_used")} for k, v in comparisons.items()},
        "accuracy_decision": result.get("accuracy_decision"),
        "penalty_full": pen_stats["full|M_bias_v2"], "timing": result["timing"]}, indent=1, default=float),
        flush=True)
    return result


# ---------------------------------------------------------------------------
# deploy: compile, M2 cards, q12, q3, report
# ---------------------------------------------------------------------------
def full_corr(rd_lodo: Path):
    return m.SymCorrection.from_json(jload(rd_lodo / "corrections.json")["full|M_bias_v2"])


def osdi_path(rd: Path) -> Path:
    return rd / "osdi" / "m_bias_v2_full.osdi"


def stage_compile(rd: Path, corr) -> dict:
    text = m.to_verilog_a_v2(corr)
    va = rd / "osdi" / "m_bias_v2_full.va"
    c = m1.compile_osdi(text, va, osdi_path(rd))
    rng = np.random.default_rng(20261021)
    worst, worst_sym = 0.0, 0.0
    for _ in range(2000):
        p = int(rng.integers(0, 2))
        vgs, vds = rng.uniform(-1.85, 1.85, size=2)
        gt = m.eval_verilog_a_v2(text, p, vgs, vds)
        gp = float(corr.g_v(bool(p), vgs, vds))
        # exchanged terminals: V(g,s') = V(g,d) = vgs - vds, V(d',s') = -vds
        gs = m.eval_verilog_a_v2(text, p, vgs - vds, -vds)
        worst = max(worst, abs(gt - gp))
        worst_sym = max(worst_sym, abs(gs - gt))
    out = {"va": str(va.relative_to(ROOT)), "osdi": str(osdi_path(rd).relative_to(ROOT)),
           "sha256": c["sha256"], "compiled_now": bool(c["compiled"]), "lines": len(text.splitlines()),
           "g_text_max": worst, "g_sym_text_max": worst_sym, "n_points": 2000}
    jdump(out, rd / "compile.json")
    print(f"compile: |g text - g python| {worst:.2e}, |g text exchanged - forward| {worst_sym:.2e}",
          flush=True)
    return out


def m2_task(task):
    tag, opts, rd_lodo = task
    row = heldout(("M2", None, full_corr(Path(rd_lodo)), load_records()[tag], opts))
    return row


def native_task(task):
    """In-sample RRMS of the M2 pair on the measured curves: SPICE-native
    (OSDI element with replica) and Python (card current x exp(g))."""
    tag, params, rd_s, rd_lodo = task
    from cryoml import spice_pdk as sp
    from cryoml.card_eval import device_by_tag
    from cryoml.data_io import load_device_curves
    from cryoml.explore.e1_correction import score_device
    from cryoml.explore.i2_1_joint import with_sims
    from cryoml.metrics import tag_metric_curves
    import subprocess
    rec = load_records()[tag]
    dev = devices()[tag]
    corr = full_corr(Path(rd_lodo))
    curves = load_device_curves(device_by_tag(tag))
    pos = {id(c): i for i, c in enumerate(curves)}
    order = [pos[id(c)] for _, c in sorted(tag_metric_curves(curves).items())]

    def run(el):
        with m1.tempdir("cryoml_i12n_") as td_s:
            td = Path(td_s)
            lib = m.patched_library_v2(td, {dev.dev_type: [(dev.bin, params)]})
            deck = sp._batch_deck(dev.dev_type, dev.L_um, dev.W_um, curves, lib, td, 77.0)
            if el:
                deck = m.insert_correction_v2(deck, osdi_path(Path(rd_s)), dev)
            (td / "d.sp").write_text(deck)
            proc = subprocess.run([m1.NG_OSDI, "-b", str(td / "d.sp")], cwd=str(td),
                                  capture_output=True, text=True, timeout=300, check=False)
            text = proc.stdout + proc.stderr
            sims = sp._read_curves(td, "raw_", curves, dev.dev_type,
                                   "could not find a valid modelname" in text.lower())
        return [np.asarray(sims[i], float) for i in order], m1.convergence_messages(text)
    card, msg_c = run(False)
    native, msg_n = run(True)
    py = [np.asarray(x, float) for x in corr.apply(with_sims(rec, card))]
    eq = m.eq_violation(np.concatenate(native), np.concatenate(py), rel=1e-3, absol=1e-12)
    return {"tag": tag, "rrms_card_only": float(score_device(rec, card)["rrms"]),
            "rrms_python": float(score_device(rec, py)["rrms"]),
            "rrms_native": float(score_device(rec, native)["rrms"]),
            "native_vs_python_default_tol_max_ratio_1e-3_1e-12": eq["max_ratio"],
            "msgs": msg_c + msg_n}


def stage_refit(rd: Path, rd_lodo: Path, tags, workers: int, opts: dict) -> dict:
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(m2_task, [(t, opts, str(rd_lodo)) for t in tags], chunksize=1)
        t_refit = time.time() - t0
        cards = {r["tag"]: r["params"] for r in rows}
        jdump(cards, rd / "cards_M2.json")
        nat = pool.map(native_task, [(t, cards[t], str(rd), str(rd_lodo)) for t in tags], chunksize=1)
    out = {"opts": opts, "refit_s": round(t_refit, 1),
           "rows": {r["tag"]: {k: r[k] for k in ("rrms_start", "rrms", "failed", "n_sim", "runtime_s")}
                    for r in rows},
           "native": {r["tag"]: r for r in nat}}
    jdump(out, rd / "refit.json")
    print(f"refit M2: {len(rows)} devices, {t_refit:.0f} s, in-sample mean "
          f"{np.mean([r['rrms'] for r in rows]):.4f}, native mean "
          f"{np.mean([v['rrms_native'] for v in nat]):.4f}", flush=True)
    return out


def dense_control(dev, out_prefix: Path) -> list[str]:
    s = -1.0 if dev.dev_type == "pmos" else 1.0
    lines = []
    for k, vd in enumerate(m1.GM_VDS):
        lines += [f"alter VD dc = {s * vd:.6g}", f"dc VG 0 {s * 1.85:.6g} {s * 0.001:.6g}",
                  f"wrdata {out_prefix.as_posix()}_gm{k}.dat -i(VD)"]
    lines.append("alter VG dc = 0")
    for k, vg in enumerate(m1.GDS_VGS):
        lines += [f"alter VG dc = {s * vg:.6g}", f"dc VD 0 {s * 1.85:.6g} {s * 0.001:.6g}",
                  f"wrdata {out_prefix.as_posix()}_gds{k}.dat -i(VD)"]
    return lines


def q12_task(task):
    tag, params, rd_s, rd_lodo = task
    rd = Path(rd_s)
    dev = devices()[tag]
    corr = full_corr(Path(rd_lodo))
    osdi = osdi_path(rd)
    res = {"tag": tag, "msgs": {}, "wall": {}}
    a = {}
    with m1.tempdir("cryoml_i12q_") as td_s:
        td = Path(td_s)
        lib = m.patched_library_v2(td, {dev.dev_type: [(dev.bin, params)]})

        def go(label, body, control, el, options="", repeats=1):
            walls, r = [], None
            for _ in range(repeats):
                r = m1.run_deck(m.deck_v2(label, lib, body, control, osdi if el else None, options),
                                td, f"{label}.sp", timeout_s=m1.TIMEOUT_DEVICE_S)
                walls.append(r["wall_s"])
            res["msgs"][label] = r["msgs"]
            res["wall"][label] = walls
            return r

        for el in (False, True):
            for tol, opt in (("default", ""), ("r4", m1.R4), ("tight", m1.TIGHT)):
                lab = f"grid_fwd_{tol}_{'corr' if el else 'card'}"
                f = td / f"{lab}.dat"
                go(lab, m.device_body_v2(dev, el), m1.grid_control(dev, f), el, opt,
                   repeats=3 if tol == "default" else 1)
                a[lab] = m1.grid_from_rows(m1.read_wrdata(f))
            lab = f"grid_swap_tight_{'corr' if el else 'card'}"
            f = td / f"{lab}.dat"
            go(lab, m.device_body_v2(dev, el, swap=True), m.swap_grid_control(dev, f), el, m1.TIGHT)
            rows = m1.read_wrdata(f, 4)
            a[lab] = m1.grid_from_rows(rows[:, :2])
            a[lab + "_vz"] = m1.grid_from_rows(rows[:, 2:4])
        pts = m1.op_points(dev)
        for el in (False, True):
            for ex in ("VG", "VD"):
                lab = f"ac_{ex}_{'corr' if el else 'card'}"
                r = go(lab, m.device_body_v2(dev, el), m1.ac_control(pts, ex), el, m1.TIGHT)
                for k, v in m1.parse_ac(r["text"], len(pts)).items():
                    a[f"{lab}_{k}"] = v
        for el in (False, True):
            lab = f"dense_{'corr' if el else 'card'}"
            pre = td / lab
            go(lab, m.device_body_v2(dev, el), dense_control(dev, pre), el, m1.TIGHT)
            for k in range(len(m1.GM_VDS)):
                a[f"{lab}_gm{k}"] = m1.read_wrdata(Path(f"{pre}_gm{k}.dat"))
            for k in range(len(m1.GDS_VGS)):
                a[f"{lab}_gds{k}"] = m1.read_wrdata(Path(f"{pre}_gds{k}.dat"))
    res.update(q12_metrics(dev, corr, pts, a))
    (rd / "arrays").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(rd / "arrays" / f"q12_{tag}.npz", pts=pts,
                        **{k: v for k, v in a.items() if isinstance(v, np.ndarray)})
    return res


def q12_metrics(dev, corr, pts, a) -> dict:
    pm = dev.dev_type == "pmos"
    s = -1.0 if pm else 1.0
    VG, VD = np.meshgrid(s * m1.GRID_VGS, m1.GRID_VDS, indexing="ij")
    g_fwd = m.g_terminal(corr, pm, VG, VD, 0.0)
    g_sw = m.g_terminal(corr, pm, VG, 0.0, VD)          # module d at nz (0 V), s at nd
    out = {"g_sym_grid_max": float(np.max(np.abs(g_sw - g_fwd)))}
    # ---- Q1 equivalence
    for tol in ("default", "r4", "tight"):
        ref = m.python_current_fwd(a[f"grid_fwd_{tol}_card"], g_fwd)
        eq = m.eq_violation(a[f"grid_fwd_{tol}_corr"], ref)
        out[f"q1_{tol}_fwd_viol"] = eq["n_viol"]
        out[f"q1_{tol}_fwd_max_ratio"] = eq["max_ratio"]
        out[f"q1_{tol}_max_rel_1nA"] = m1.nanmax0(m1.rel_diff(a[f"grid_fwd_{tol}_corr"], ref, m1.I_FLOOR))
    ref_sw = m.python_current_swap(a["grid_swap_tight_card"], a["grid_swap_tight_card_vz"], g_sw)
    eq = m.eq_violation(a["grid_swap_tight_corr"], ref_sw)
    out["q1_tight_swap_viol"] = eq["n_viol"]
    out["q1_tight_swap_max_ratio"] = eq["max_ratio"]
    out["q1_tight_viol"] = int(out["q1_tight_fwd_viol"] + out["q1_tight_swap_viol"])
    corr_runs = [k for k in a if k.startswith("grid_") and "_corr" in k]
    out["q1_nonfinite"] = int(sum(np.sum(~np.isfinite(a[k])) for k in corr_runs))
    out["q1_nonfinite_card"] = int(sum(np.sum(~np.isfinite(a[k])) for k in a
                                       if k.startswith("grid_") and "_card" in k))
    j0 = int(np.argmin(np.abs(m1.GRID_VDS)))
    out["q1_vds0_max_corr"] = float(max(np.nanmax(np.abs(a[f"grid_fwd_{t}_corr"][:, j0]))
                                        for t in ("default", "tight")))
    out["q1_vds0_max_card"] = float(max(np.nanmax(np.abs(a[f"grid_fwd_{t}_card"][:, j0]))
                                        for t in ("default", "tight")))
    out["g_grid_min"], out["g_grid_max"] = float(np.min(g_fwd)), float(np.max(g_fwd))
    # exchange asymmetry (forward operation, |I| > 1 nA)
    fw = (s * VD) > 0
    ic, isw = a["grid_fwd_tight_corr"], a["grid_swap_tight_corr"]
    cc, csw = a["grid_fwd_tight_card"], a["grid_swap_tight_card"]
    keep = (fw & (np.abs(ic) > m1.I_FLOOR) & (np.abs(cc) > m1.I_FLOOR)
            & np.isfinite(isw) & np.isfinite(csw) & np.isfinite(ic) & np.isfinite(cc))
    with np.errstate(divide="ignore", invalid="ignore"):
        lr_corr = np.log(np.abs(isw / ic))
        lr_card = np.log(np.abs(csw / cc))
    d = np.abs(lr_corr - lr_card)
    out["swap_asym_card_max"] = float(np.max(np.abs(lr_card[keep]))) if keep.any() else 0.0
    out["swap_asym_element_max"] = float(np.max(d[keep])) if keep.any() else 0.0
    # ---- Q2 small signal (tight)
    vg, vd = pts[:, 0], pts[:, 1]
    gp = m.g_terminal(corr, pm, vg, vd, 0.0)
    dgg, dgd = m.g_partials_v2(corr, pm, vg, vd)
    i_card = a["ac_VG_card_i0"]
    yc_dg = a["ac_VG_card_yr"] + 1j * a["ac_VG_card_ym"]
    yc_dd = a["ac_VD_card_yr"] + 1j * a["ac_VD_card_ym"]
    ye_dg = a["ac_VG_corr_yr"] + 1j * a["ac_VG_corr_ym"]
    ye_dd = a["ac_VD_corr_yr"] + 1j * a["ac_VD_corr_ym"]
    yp_dg = m.python_admittance_v2(i_card, yc_dg, gp, dgg)
    yp_dd = m.python_admittance_v2(i_card, yc_dd, gp, dgd)
    ipy = m.python_current_fwd(i_card, gp)
    cnt = np.abs(ipy) > m1.I_FLOOR
    for k, e in (("gm", m1.norm_y_err(ye_dg.real, yp_dg.real)),
                 ("gds", m1.norm_y_err(ye_dd.real, yp_dd.real))):
        out[f"q2_err_{k}"] = float(np.nanmax(np.where(cnt, e, np.nan))) if cnt.any() else 0.0
        if np.any(cnt & ~np.isfinite(e)):
            out[f"q2_err_{k}"] = float("inf")
    out["q2_op_i_rel"] = m1.nanmax0(m1.rel_diff(a["ac_VG_corr_i0"], ipy, m1.I_FLOOR))
    caps = {"dd": (ye_dd.imag, yc_dd.imag), "dg": (ye_dg.imag, yc_dg.imag),
            "gg": (a["ac_VG_corr_yg"], a["ac_VG_card_yg"])}
    for k, (e_, c_) in caps.items():
        err = m.cap_err(e_, c_)
        out[f"cap_err_{k}"] = float(np.max(err)) if np.all(np.isfinite(err)) else float("inf")
    out["cap_max"] = max(out[f"cap_err_{k}"] for k in caps)
    # strong inversion and sign changes (I11-1 rule)
    jv = int(np.argmin(np.abs(np.abs(m1.GRID_VDS) - m1.GM_VDS[0]) + (s * m1.GRID_VDS < 0)))
    from cryoml.spice_pdk import device_mult
    w_eff = dev.W_um * device_mult(dev.dev_type, dev.L_um, dev.W_um)
    vt = m1.threshold_voltage(m1.GRID_VGS, a["grid_fwd_tight_card"][:, jv], 1e-7 * w_eff / dev.L_um)
    out["vt_card"] = vt
    nv = len(m1.GRID_VGS)
    n_gm = len(m1.GM_VDS) * nv
    gm_mask = np.zeros(len(pts), bool)
    gm_mask[:n_gm] = np.abs(vg[:n_gm]) >= vt + 0.2
    gds_mask = np.zeros(len(pts), bool)
    gds_mask[n_gm:] = np.abs(vd[n_gm:]) >= 0.05
    gm_ac = [m1.curve_sign_change_introduced(ye_dg.real[k * nv:(k + 1) * nv], yc_dg.real[k * nv:(k + 1) * nv],
                                             gm_mask[k * nv:(k + 1) * nv]) for k in range(len(m1.GM_VDS))]
    gds_ac = [m1.curve_sign_change_introduced(ye_dd.real[n_gm + k * nv:n_gm + (k + 1) * nv],
                                              yc_dd.real[n_gm + k * nv:n_gm + (k + 1) * nv],
                                              gds_mask[n_gm + k * nv:n_gm + (k + 1) * nv])
              for k in range(len(m1.GDS_VGS))]
    gm_dn, gds_dn = [False] * len(m1.GM_VDS), [False] * len(m1.GDS_VGS)
    fd_err = 0.0
    for k, vds in enumerate(m1.GM_VDS):
        dc, dk = a[f"dense_card_gm{k}"], a[f"dense_corr_gm{k}"]
        if dc.shape[0] < 10 or dk.shape[0] != dc.shape[0]:
            fd_err = float("inf")
            gm_dn[k] = True
            continue
        o = np.argsort(np.abs(dc[:, 0]))
        x, ic_, ik_ = dc[o, 0], dc[o, 1], dk[o, 1]
        gm_c, gm_k = m1.central_diff(x, ic_), m1.central_diff(x, ik_)
        sel = slice(k * nv, (k + 1) * nv)
        at = np.searchsorted(np.round(np.abs(x), 6), np.round(np.abs(vg[sel]), 6))
        inner = (at >= 1) & (at <= len(x) - 2)
        at = np.clip(at, 1, len(x) - 2)
        cnt_k = (np.abs(ipy[sel]) > m1.I_FLOOR) & inner
        e1 = m1.norm_y_err(gm_k[at], ye_dg.real[sel])
        fd_err = max(fd_err, float(np.max(e1[cnt_k])) if cnt_k.any() else 0.0)
        gm_dn[k] = m1.curve_sign_change_introduced(gm_k, gm_c, np.abs(x) >= vt + 0.2)
    for k, vgs in enumerate(m1.GDS_VGS):
        dc, dk = a[f"dense_card_gds{k}"], a[f"dense_corr_gds{k}"]
        if dc.shape[0] < 10 or dk.shape[0] != dc.shape[0]:
            gds_dn[k] = True
            continue
        o = np.argsort(np.abs(dc[:, 0]))
        x, ic_, ik_ = dc[o, 0], dc[o, 1], dk[o, 1]
        gds_dn[k] = m1.curve_sign_change_introduced(m1.central_diff(x, ik_), m1.central_diff(x, ic_),
                                                    np.abs(x) >= 0.05)
    out.update({"q2_fd_err_osdi": fd_err,
                "q2_gm_curves_introduced": [bool(p or q) for p, q in zip(gm_ac, gm_dn)],
                "q2_gds_curves_introduced": [bool(p or q) for p, q in zip(gds_ac, gds_dn)],
                "q2_gm_neg_card_ac": int(np.sum(gm_mask & (yc_dg.real < 0))),
                "q2_gds_neg_card_ac": int(np.sum(gds_mask & (yc_dd.real < 0)))})
    return out


def stage_q12(rd: Path, rd_lodo: Path, tags, workers: int) -> dict:
    cards = jload(rd / "cards_M2.json")
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(q12_task, [(t, cards[t], str(rd), str(rd_lodo)) for t in tags], chunksize=1)
    out = {"wall_s": round(time.time() - t0, 1), "devices": {r["tag"]: r for r in rows}}
    jdump(out, rd / "q12.json")
    print(f"q12: {len(rows)} devices, {out['wall_s']} s", flush=True)
    return out


CIRCUITS = (
    ("inv_A_vtc", "vtc", ("nmos_L0p15_W1p6", "pmos_L0p35_W1p6"), {}),
    ("inv_B_vtc", "vtc", ("nmos_L1_W1p6", "pmos_L2_W5"), {}),
    ("ro5_A", "ring", ("nmos_L0p15_W1p6", "pmos_L0p35_W1p6"), {"tstep": 1e-12, "tstop": 10e-9}),
    ("ro5_B", "ring", ("nmos_L1_W1p6", "pmos_L2_W5"), {"tstep": 5e-12, "tstop": 50e-9}),
    ("mirror_n", "mirror", ("nmos_L1_W1p6",), {"iref": 10e-6}),
    ("mirror_p", "mirror", ("pmos_L2_W5",), {"iref": 10e-6}),
)
Q3_ARMS = ("published", "C0_refit", "M2_card", "M2_full")


def arm_cards(arm: str, tags, m2: dict) -> dict:
    devs = devices()
    held = jload(ALL18 / "heldout_cards.json")
    cards = {}
    for t in tags:
        d = devs[t]
        p = None if arm == "published" else (dict(held["C0_refit"][t]) if arm == "C0_refit"
                                              else dict(m2[t]))
        cards.setdefault(d.dev_type, []).append((d.bin, p))
    return cards


def q3_task(task):
    name, kind, tags, settings, arm, rd_s = task
    rd = Path(rd_s)
    devs = devices()
    m2 = jload(rd / "cards_M2.json")
    el = arm == "M2_full"
    osdi = osdi_path(rd) if el else None
    reps = 3 if arm in ("M2_card", "M2_full") else 1
    res = {"circuit": name, "arm": arm, "kind": kind}
    with m1.tempdir("cryoml_i12c_") as td_s:
        td = Path(td_s)
        lib = m.patched_library_v2(td, arm_cards(arm, tags, m2))
        f = td / "out.dat"
        if kind == "vtc":
            body, ctrl = m.vtc_deck_body_v2(devs[tags[0]], devs[tags[1]], el), m1.vtc_control(f)
        elif kind == "ring":
            body = m.ring_body_v2(devs[tags[0]], devs[tags[1]], el, stages=5)
            ctrl = m1.ring_control(f, settings["tstep"], settings["tstop"])
        else:
            body, ctrl = m.mirror_body_v2(devs[tags[0]], el, settings["iref"]), m1.mirror_control(f)
        walls = []
        for _ in range(reps):
            r = m1.run_deck(m.deck_v2(name, lib, body, ctrl, osdi), td, "c.sp",
                            timeout_s=m1.TIMEOUT_CIRCUIT_S)
            walls.append(r["wall_s"])
            if r["msgs"]:          # failed run: no repeats (amendment A1)
                break
        res["wall"], res["msgs"] = walls, r["msgs"]
        if kind == "mirror":
            rows = m1.read_wrdata(f, 4)
            x, y, vref = rows[:, 0], rows[:, 1], rows[:, 3]
        else:
            rows = m1.read_wrdata(f, 2)
            x, y = rows[:, 0], rows[:, 1]
    finite = bool(rows.size and np.all(np.isfinite(rows)))
    res["finite"] = finite
    if kind == "vtc" and finite:
        res.update(m1.vtc_metrics(x, y))
    elif kind == "ring" and finite:
        res.update(m1.oscillation(x, y, m1.VDD_CIRCUIT / 2, t_from=0.3 * settings["tstop"]))
    elif kind == "mirror" and finite:
        vr = float(np.median(vref))
        res["v_ref"] = vr
        res["ratio_at_0p9"] = m1.interp_at(x, y, 0.9) / settings["iref"]
        res["ratio_at_vref"] = m1.interp_at(x, y, vr) / settings["iref"]
    ok = finite and not r["msgs"]
    if kind == "ring" and finite:
        ok = ok and res["n_periods"] >= m1.THR["ro_min_periods"]
    res["ok"] = bool(ok)
    (rd / "arrays").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(rd / "arrays" / f"q3_{name}_{arm}.npz", rows=rows)
    return res


def stage_q3(rd: Path, workers: int) -> dict:
    tasks = [(n, k, t, s, arm, str(rd)) for n, k, t, s in CIRCUITS for arm in Q3_ARMS]
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(q3_task, tasks, chunksize=1)
    out = {"wall_s": round(time.time() - t0, 1),
           "rows": {f"{r['circuit']}|{r['arm']}": r for r in rows}}
    jdump(out, rd / "q3.json")
    print(f"q3: {len(rows)} runs, {out['wall_s']} s; not ok: "
          f"{[k for k, v in out['rows'].items() if not v['ok']]}", flush=True)
    return out


def summary_values(rd: Path) -> dict:
    """Pre-registered part B summary values from q12 / q3 / compile."""
    q12 = jload(rd / "q12.json")["devices"]
    q3 = jload(rd / "q3.json")["rows"]
    comp = jload(rd / "compile.json")
    D = list(q12.values())
    elem_msgs = sum(len(v) for r in D for k, v in r["msgs"].items() if k.endswith("_corr"))
    ratios = {r["tag"]: float(np.median(r["wall"]["grid_fwd_default_corr"])
                              / np.median(r["wall"]["grid_fwd_default_card"])) for r in D}
    names = [n for n, *_ in CIRCUITS]
    circ = m.circuit_runtime_ratios(q3, names)      # amendment A1
    s = {"q1_tight_viol": int(sum(r["q1_tight_viol"] for r in D)),
         "q1_bad": int(elem_msgs + sum(r["q1_nonfinite"] for r in D)),
         "q1_vds0_max": max(r["q1_vds0_max_corr"] for r in D),
         "g_text_max": comp["g_text_max"], "g_sym_text_max": comp["g_sym_text_max"],
         "swap_asym_max": max(r["swap_asym_element_max"] for r in D),
         "q2_ac_max": max(max(r["q2_err_gm"], r["q2_err_gds"]) for r in D),
         "cap_max": max(r["cap_max"] for r in D),
         "q2_gm_sign": int(sum(sum(r["q2_gm_curves_introduced"]) for r in D)),
         "q3_fail": m.q3_fail_count(q3, names),
         "q1_default_max": max(r["q1_default_max_rel_1nA"] for r in D),
         "q2_gds_sign": int(sum(sum(r["q2_gds_curves_introduced"]) for r in D)),
         "runtime_ratio_max": max(list(ratios.values()) + list(circ.values()))}
    return {"values": s, "runtime_grid": ratios, "runtime_circuits": circ,
            "runtime_circuits_excluded": [n for n, *_ in CIRCUITS if n not in circ],
            "n_devices": len(D), "q3_not_ok": [k for k, v in q3.items() if not v["ok"]]}


def stage_report(rd: Path) -> dict:
    sv = summary_values(rd)
    dec = m.decide_deploy(sv["values"])
    refit = jload(rd / "refit.json")
    out = {**sv, "decision": dec, "refit_in_sample": {
        "mean_rrms_M2_python": float(np.mean([v["rrms_python"] for v in refit["native"].values()])),
        "mean_rrms_M2_native": float(np.mean([v["rrms_native"] for v in refit["native"].values()])),
        "mean_rrms_M2_card_only": float(np.mean([v["rrms_card_only"] for v in refit["native"].values()])),
        "n_failed": int(sum(bool(v["failed"]) for v in refit["rows"].values()))}}
    jdump(out, rd / "summary.json")
    print(json.dumps({"values": sv["values"], **dec}, indent=1, default=float), flush=True)
    return out


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------
def stage_decide() -> dict:
    val = jload(OUT / "validity" / "validity.json")
    lodo = jload(OUT / "lodo" / "results.json")
    dep = jload(OUT / "deploy" / "summary.json")
    if val["smoke"] or lodo["smoke"] or not lodo["preregistered_config"]:
        raise RuntimeError("decide needs the full pre-registered runs")
    tags = sorted(devices())
    rows = m.select_primary_rows(lodo["per_device"]["M_bias_v2"], tags)
    validity = m.decide_validity({"v1_max_g_diff": val["v1_max_g_diff"],
                                  "v2_max_rrms_diff": val["v2_max_rrms_diff"],
                                  "v3_n_rows": len(rows)})
    acc = m.decide_accuracy(lodo["comparisons"]["A1_M_bias_v2_vs_C0_refit"],
                            lodo["comparisons"]["A2_M_bias_v2_vs_M_bias_v1"],
                            lodo["n_failed_M_bias_v2"])
    out = {"prereg": str(PREREG.relative_to(ROOT)), "validity": validity, "accuracy": acc,
           "deploy": dep["decision"], **m.decide_outcome(validity, acc, dep["decision"]),
           "A1": lodo["comparisons"]["A1_M_bias_v2_vs_C0_refit"],
           "A2": lodo["comparisons"]["A2_M_bias_v2_vs_M_bias_v1"],
           "summary": lodo["summary"]}
    jdump(out, OUT / "decision.json")
    print(json.dumps(out, indent=1, default=float), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("grids", "calib", "validity", "lodo", "deploy", "compile", "refit",
                             "q12", "q3", "report", "decide"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lodo-dir", default="", help="lodo output directory (default <base>/lodo)")
    ap.add_argument("--smoke", action="store_true",
                    help="2 folds, 1 seed, 200 steps, refit max_nfev 20; part B on 2 devices "
                         "(code check; not used for decisions)")
    args = ap.parse_args()
    base = OUT / "smoke" if args.smoke else OUT
    smoke_tags = ["nmos_L1_W1p6", "pmos_L2_W5"]
    rd_lodo = Path(args.lodo_dir).resolve() if args.lodo_dir else base / "lodo"
    rd_dep = base / "deploy"
    timing = {}
    t0 = time.time()
    if args.stage == "grids":
        stage_grids(args.workers)
    elif args.stage == "calib":
        stage_calib()
    elif args.stage == "validity":
        stage_validity(args.workers, base / "validity", args.smoke)
    elif args.stage == "lodo":
        stage_lodo(args.workers, rd_lodo, args.smoke, smoke_tags if args.smoke else None)
    elif args.stage == "decide":
        stage_decide()
    else:
        tags = smoke_tags if args.smoke else sorted(devices())
        opts = dict(REFIT_OPTS, heldout_nfev=20) if args.smoke else dict(REFIT_OPTS)
        workers = min(args.workers, 4)
        stages = (("compile", "refit", "q12", "q3", "report") if args.stage == "deploy"
                  else (args.stage,))
        for st in stages:
            t1 = time.time()
            if st == "compile":
                stage_compile(rd_dep, full_corr(rd_lodo))
            elif st == "refit":
                stage_refit(rd_dep, rd_lodo, tags, workers, opts)
            elif st == "q12":
                stage_q12(rd_dep, rd_lodo, tags, workers)
            elif st == "q3":
                if args.smoke:      # smoke: starting cards for the circuit devices not refit
                    have = jload(rd_dep / "cards_M2.json")
                    sc = start_cards()
                    for _, _, ts, _ in CIRCUITS:
                        for t in ts:
                            have.setdefault(t, sc[t])
                    jdump(have, rd_dep / "cards_M2.json")
                stage_q3(rd_dep, workers)
            else:
                stage_report(rd_dep)
            timing[st] = round(time.time() - t1, 1)
        tf = rd_dep / "timing.json"
        old = jload(tf) if tf.exists() else {}
        old.update(timing)
        jdump(old, tf)
    print(f"stage {args.stage} done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
