#!/usr/bin/env python3
"""I2-1: correction trained jointly with the training cards, then held-out refit.

Pre-registration: ``out/research/explore/I2_1/prereg.json``.

Per fold and joint arm: ``g_0`` is the E1 fold correction; for r = 1..K every
training card is refit with ``g_{r-1}`` in the loop (``ngspice_fit.polish``,
warm start, ``--train-nfev``/``--train-irls``) and ``g_r`` is retrained from
scratch on the refit cards' fresh simulations
(``cryoml.explore.i2_1_joint.joint_rounds``; rounds synchronized across
folds on one process pool).  The held-out card is then refit from its
starting card with the frozen correction (``--heldout-nfev``/``--heldout-irls``)
and scored with the strict paper RRMS (``e1_correction.score_device``).

Modes:
  real  REAL_DEV leave-one-device-out; arms C0_refit, M_joint, T1_joint,
        ORACLE_joint (``--oracle``); M_post read from the stored E1 stage-2
        results.  Evaluates the pre-registered criterion on the full config.
  pdk2  GF180 / IHP tier c sets 0-2, leave-one-geometry-out; arms C0_refit,
        M_post (held-out refit with g_0), M_joint; identifiability diagnostic
        against the truth's injected misspecification components.

  PYTHONPATH=src .venv/bin/python scripts/research/explore/i2_1_joint.py \
      --mode real --name real_dev --oracle --workers 4
  PYTHONPATH=src .venv/bin/python scripts/research/explore/i2_1_joint.py \
      --mode pdk2 --pdk gf180 --name gf180_c_logo --workers 6
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cryoml.explore.e1_correction import (HYPER, Correction, fit_arm,  # noqa: E402
                                          load_devices, raw_features, score_device)
from cryoml.explore.i2_1_joint import (COMPONENTS, CorrectedProblem, Fold,  # noqa: E402
                                       correlation_stats, corrected_score, curve_key,
                                       finite_mask, joint_rounds, log_ratio,
                                       merge_train_results, split_train_tasks, with_sims)
from cryoml.metrics import clean_current  # noqa: E402

E1 = ROOT / "out/research/explore/E1"
OUT = ROOT / "out/research/explore/I2_1"
PREREG = OUT / "prereg.json"
SPEC = ROOT / "out/cold_loop/full_direct/round_3/spec.json"
CARDS = ROOT / "out/research/polished/direct43_r3_fd/real/summary.json"
SPLIT = ROOT / "out/research/explore/real_split.json"
PREREG_OPTIONS = {"k": 3, "train_nfev": 150, "train_irls": 1, "heldout_nfev": 400,
                  "heldout_irls": 2, "diff_step": 0.01}
_SPEC: dict | None = None


def _spec() -> dict:
    global _SPEC
    if _SPEC is None:
        _SPEC = json.loads(SPEC.read_text())
    return _SPEC


# ---------------------------------------------------------------------------
# Worker tasks (top level, so spawn workers can unpickle them)
# ---------------------------------------------------------------------------
def _refit(corr: Correction, rec, nfev: int, irls: int, diff_step: float):
    """Refit ``rec``'s card with ``corr`` in the loop; returns (record, info)."""
    from cryoml.ngspice_fit import polish
    t0 = time.time()
    if rec.source == "sky130":
        from cold_loop import box_for
        from e1_collect_real import collect
        from cryoml.ngspice_fit import DeviceProblem
        spec = _spec()
        base = DeviceProblem(rec.tag, box_for(spec, rec.tag))
        start = base.box.params_to_local(
            {**spec["published_by_device"][rec.tag],
             **{k: v for k, v in rec.meta["params"].items() if k in base.box.names}})
    else:
        from cryoml.pdk2 import boxes
        from cryoml.pdk2.problem import TRUTH_ROOTS, Pdk2Problem
        m = rec.meta
        box = boxes.box_for(m["fit"], m["truth_tag"])
        base = Pdk2Problem.from_truth(m["truth_tag"], box, m["tier"], int(m["set"]),
                                      TRUTH_ROOTS[rec.source])
        start = box.params_to_local(m["params"])
    prob = CorrectedProblem(base, corr, rec.family, rec.L_um, rec.W_um, rec.vdd, rec.source)
    result = polish(prob, start, max_nfev=nfev, irls_rounds=irls, diff_step=diff_step)
    params = base.box.local_to_params(result["local"])
    info = {"n_sim": base.n_sim, "history": result["history"],
            "runtime_s": round(time.time() - t0, 2)}
    if rec.source == "sky130":
        new = collect(rec.tag, params)
        new.meta.update(params=params)
    else:
        sims = base.simulate(result["local"])
        keys = [curve_key(c) for c in base.curves]
        if keys != [c.key for c in rec.curves]:
            raise RuntimeError(f"{rec.tag}: curve order differs from the record")
        new = with_sims(rec, sims, params=params)
    new.meta.update(refit_n_sim=base.n_sim, refit_runtime_s=info["runtime_s"])
    return new, info


def train_task(task):
    arm, recs, hyper = task
    return fit_arm(arm, recs, hyper=hyper.get(arm), threads=1)


def refit_train_task(task):
    corr, rec, opts = task
    new, _ = _refit(corr, rec, opts["train_nfev"], opts["train_irls"], opts["diff_step"])
    return new


def _points(corr: Correction, rec) -> dict:
    """g and ln|target / I_card| at the included trimmed points of ``rec``."""
    g, resid = [], []
    for c in rec.curves:
        if not c.included or not (c.den > 0):
            continue
        s = slice(c.start, None)
        sim = np.asarray(c.sim, float)[s]
        n = len(sim)
        phi = raw_features(rec, np.asarray(c.vg)[s], np.asarray(c.vd)[s], sim)
        g.append(corr.g(phi, [f"{rec.family}|{c.key}"] * n, [f"{rec.family}|{c.kind}"] * n))
        resid.append(log_ratio(clean_current(c.target)[s], sim))
    return {"g": np.concatenate(g), "resid": np.concatenate(resid)}


def heldout_task(task):
    label, rnd, corr, rec, opts = task
    new, info = _refit(corr, rec, opts["heldout_nfev"], opts["heldout_irls"],
                       opts["diff_step"])
    before = score_device(rec, corr.apply(rec))
    after = score_device(new, corr.apply(new))
    row = {"label": label, "round": rnd, "tag": rec.tag, "rrms_start": before["rrms"],
           "rrms": after["rrms"], "failed": after["failed"], "n_sim": info["n_sim"],
           "history": info["history"], "params": new.meta["params"],
           "runtime_s": info["runtime_s"]}
    pts = _points(corr, new) if rec.source != "sky130" and corr.arm != "C0" else None
    return row, pts


def components_task(rec):
    """Known misspecification components of a pdk2 tier-c target on its grid."""
    from cryoml.pdk2 import boxes
    from cryoml.pdk2.problem import TRUTH_ROOTS, load_truth, load_truth_meta
    m = rec.meta
    root = TRUTH_ROOTS[rec.source]
    truth = load_truth_meta(m["tier"], root)["sets"][str(m["set"])][m["truth_tag"]]
    sim = boxes.backend(rec.source)
    dev = sim.device_by_tag(m["truth_tag"])
    curves = sim.device_curves(dev.dev_type, load_truth(m["tier"], int(m["set"]), root)[m["truth_tag"]])
    fitted = dict(truth["fitted_params"])
    full = {**fitted, **truth["misspec_params"]}
    temp, rs = float(truth["temp_C"]), float(truth["rs_ext_ohm_per_side"])
    i_f = sim.simulate(dev, fitted, curves=curves, temp_C=temp)
    i_t0 = sim.simulate(dev, full, curves=curves, temp_C=temp)
    i_rs = sim.simulate(dev, full, curves=curves, temp_C=temp, rs_ext_ohm=rs)
    i_h = sim.simulate(dev, full, curves=curves, temp_C=temp, rs_ext_ohm=rs,
                       curve_temps_C=truth["curve_temps_C"])
    with np.load(root / m["tier"] / f"set_{int(m['set'])}.npz") as data:
        clean = np.asarray(data[f"{m['truth_tag']}__clean_ids"], float)
    mine = np.concatenate([np.abs(np.asarray(x, float)) for x in i_h])
    clean_check = float(np.nanmax(np.abs(mine - np.abs(clean))
                                  / np.maximum(np.abs(clean), 1e-15)))
    out = {k: [] for k in COMPONENTS}
    for i, c in enumerate(rec.curves):
        if not c.included or not (c.den > 0):
            continue
        s = slice(c.start, None)
        out["c_param"].append(log_ratio(i_t0[i][s], i_f[i][s]))
        out["c_rs"].append(log_ratio(i_rs[i][s], i_t0[i][s]))
        out["c_heat"].append(log_ratio(i_h[i][s], i_rs[i][s]))
        out["c_gain"].append(np.full(len(np.asarray(i_h[i])[s]),
                                     math.log(float(truth["curve_gains"][i]))))
    return rec.tag, {k: np.concatenate(v) for k, v in out.items()}, clean_check


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def aggregate(values: dict) -> dict:
    from e1_lodo import aggregate as agg
    return agg(values)


def wins(a: dict, b: dict) -> int:
    from e1_lodo import wins as w
    return w(a, b)


def load_records(args):
    if args.mode == "real":
        recs = load_devices(E1 / "real_dev")
        cards = json.loads(CARDS.read_text())["devices"]
        dev_tags = json.loads(SPLIT.read_text())["REAL_DEV"]
        if sorted(r.tag for r in recs) != sorted(dev_tags):
            raise RuntimeError("real_dev records differ from REAL_DEV")
        for r in recs:
            r.meta["params"] = dict(cards[r.tag]["params"])
        g0 = json.loads((E1 / "real_dev_lodo/fold_corrections.json").read_text())
        return recs, (lambda r: r.tag), g0
    recs = [d for s in (0, 1, 2) for d in load_devices(E1 / f"{args.pdk}_c_set{s}")]
    g0 = json.loads((E1 / f"{args.pdk}_c_logo/fold_corrections.json").read_text())
    return recs, (lambda r: r.geometry), g0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("real", "pdk2"), required=True)
    ap.add_argument("--pdk", choices=("gf180", "ihp"), default="gf180")
    ap.add_argument("--name", required=True)
    ap.add_argument("--arms", default="", help="joint arms (default real: M,T1; pdk2: M)")
    ap.add_argument("--oracle", action="store_true", help="ORACLE_joint (real mode)")
    ap.add_argument("--k", type=int, default=PREREG_OPTIONS["k"])
    ap.add_argument("--train-nfev", type=int, default=PREREG_OPTIONS["train_nfev"])
    ap.add_argument("--train-irls", type=int, default=PREREG_OPTIONS["train_irls"])
    ap.add_argument("--heldout-nfev", type=int, default=PREREG_OPTIONS["heldout_nfev"])
    ap.add_argument("--heldout-irls", type=int, default=PREREG_OPTIONS["heldout_irls"])
    ap.add_argument("--diff-step", type=float, default=PREREG_OPTIONS["diff_step"])
    ap.add_argument("--heldout-rounds", default="",
                    help="M rounds refit on the held-out device (default real: 1..K; "
                         "pdk2: 0,K); other arms use K only")
    ap.add_argument("--folds-limit", type=int, default=0, help="smoke: first N folds")
    ap.add_argument("--m-seeds", default="", help="smoke override of M seeds")
    ap.add_argument("--m-steps", type=int, default=0, help="smoke override of M steps")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    t_start = time.time()
    real = args.mode == "real"
    arms = [a for a in (args.arms or ("M,T1" if real else "M")).split(",") if a]
    K = args.k
    if args.heldout_rounds:
        m_rounds = sorted({int(x) for x in args.heldout_rounds.split(",")} | {K})
    else:
        m_rounds = list(range(1, K + 1)) if real else sorted({0, K})
    opts = {"k": K, "train_nfev": args.train_nfev, "train_irls": args.train_irls,
            "heldout_nfev": args.heldout_nfev, "heldout_irls": args.heldout_irls,
            "diff_step": args.diff_step}
    hyper = {"M": {}}
    if args.m_seeds:
        hyper["M"]["seeds"] = tuple(int(x) for x in args.m_seeds.split(","))
    if args.m_steps:
        hyper["M"]["steps"] = args.m_steps
    preregistered = (opts == PREREG_OPTIONS and not hyper["M"] and not args.folds_limit)

    records, key, g0 = load_records(args)
    groups = sorted({key(r) for r in records})
    if args.folds_limit:
        groups = groups[:args.folds_limit]
    held = {g: [r for r in records if key(r) == g] for g in groups}
    folds = []
    for arm in arms:
        for g in groups:
            folds.append(Fold(f"{arm}|{g}", arm, [r for r in records if key(r) != g],
                              corr=Correction.from_json(g0[g][arm])))
    if args.oracle and real:
        for g in groups:
            folds.append(Fold(f"ORACLE|{g}", "M", list(held[g]), corr=None))
    print(f"I2_1 {args.mode} {args.pdk if not real else 'sky130'}: {len(groups)} folds, "
          f"{len(folds)} joint folds, K={K}, opts {opts}, hyper {hyper}, "
          f"preregistered={preregistered}", flush=True)

    timing = {}
    with get_context("spawn").Pool(args.workers) as pool:
        def refit_map(tasks):
            t0 = time.time()
            out = pool.map(refit_train_task, [(c, r, opts) for c, r in tasks], chunksize=1)
            timing.setdefault("train_refit_s", []).append(round(time.time() - t0, 1))
            timing.setdefault("train_refit_n", []).append(len(tasks))
            return out

        def train_map(tasks):
            t0 = time.time()
            expanded, owner = split_train_tasks(tasks, hyper)
            out = merge_train_results(len(tasks), owner,
                                      pool.map(train_task, expanded, chunksize=1))
            timing.setdefault("train_corr_s", []).append(round(time.time() - t0, 1))
            timing.setdefault("train_corr_n", []).append(len(tasks))
            return out

        joint_rounds(folds, K, refit_map, train_map, log=lambda s: print(s, flush=True))
        print(f"joint rounds done ({time.time() - t_start:.0f}s)", flush=True)

        tasks = [("C0_refit", None, Correction("C0"), rec, opts)
                 for g in groups for rec in held[g]]
        for f in folds:
            arm, g = f.name.split("|", 1)
            if arm == "ORACLE":
                tasks += [("ORACLE_joint", K, f.corrs[K], rec, opts) for rec in held[g]]
                continue
            for r in (m_rounds if arm == "M" else [K]):
                label = f"{arm}_post" if r == 0 else (f"{arm}_joint" if r == K
                                                      else f"{arm}_joint_r{r}")
                tasks += [(label, r, f.corrs[r], rec, opts) for rec in held[g]]
        t0 = time.time()
        out = pool.map(heldout_task, tasks, chunksize=1)
        timing["heldout_refit_s"] = round(time.time() - t0, 1)
        timing["heldout_refit_n"] = len(tasks)
        comps = {}
        if not real:
            t0 = time.time()
            for tag, arrays, check in pool.map(components_task,
                                               [r for g in groups for r in held[g]],
                                               chunksize=1):
                comps[tag] = (arrays, check)
            timing["components_s"] = round(time.time() - t0, 1)

    rows, points = {}, {}
    for row, pts in out:
        rows.setdefault(row["label"], {})[row["tag"]] = row
        if pts is not None:
            points.setdefault(row["label"], {})[row["tag"]] = pts
    scores = {lab: {t: r["rrms"] for t, r in v.items()} for lab, v in rows.items()}
    stored = None
    if real:
        stored = json.loads((E1 / "real_dev_refit/results.json").read_text())
        held_tags = [r.tag for g in groups for r in held[g]]
        scores["M_post"] = {t: stored["rows"]["M"][t]["rrms"] for t in held_tags}
    summary = {lab: aggregate(v) for lab, v in scores.items()}
    comp = {}
    for a, b in (("M_joint", "M_post"), ("M_joint", "C0_refit"), ("M_joint", "T1_joint"),
                 ("M_post", "C0_refit"), ("T1_joint", "C0_refit")):
        if a in scores and b in scores:
            comp[f"{a}_beats_{b}"] = wins(scores[a], scores[b])

    # per-round diagnostics: post-hoc held-out score of g_r (no refit)
    rounds = {}
    for f in folds:
        arm, g = f.name.split("|", 1)
        hist = []
        for r, (h, corr) in enumerate(zip(f.history, f.corrs)):
            entry = {k: v for k, v in h.items() if k != "refit_n_sim"}
            if arm != "ORACLE":
                entry["heldout_posthoc"] = {rec.tag: corrected_score(corr, rec) for rec in held[g]}
            n_sim = [x for x in h.get("refit_n_sim", []) if x is not None]
            if n_sim:
                entry["train_refit_n_sim_mean"] = float(np.mean(n_sim))
            hist.append(entry)
        rounds[f.name] = hist

    result = {"name": args.name, "mode": args.mode,
              "pdk": "sky130" if real else args.pdk, "prereg": str(PREREG.relative_to(ROOT)),
              "preregistered_config": preregistered, "smoke": not preregistered,
              "options": opts, "hyper": HYPER, "hyper_overrides": hyper,
              "joint_arms": arms, "oracle": bool(args.oracle and real),
              "m_heldout_rounds": m_rounds, "n_folds": len(groups),
              "n_heldout_records": sum(len(v) for v in held.values()),
              "summary": summary, "wins": comp, "per_device": scores,
              "rounds": rounds, "timing": timing}
    if real:
        result["M_post_source"] = "out/research/explore/E1/real_dev_refit/results.json rows.M"
        c0 = {t: stored["rows"]["C0"][t]["rrms"] for t in scores.get("C0_refit", {})}
        if c0:
            diff = [abs(scores["C0_refit"][t] - c0[t]) for t in c0]
            result["C0_refit_vs_E1_stage2"] = {"max_abs_diff": float(max(diff)),
                                               "E1_stage2_C0": aggregate(c0)}
    if real and preregistered and len(groups) == 12 and {"M_joint", "T1_joint"} <= set(scores):
        m, t1 = summary["M_joint"], summary["T1_joint"]
        checks = {
            "M_joint_mean_le_0.080": bool(m["all_device_mean"] <= 0.080),
            "M_joint_beats_M_post_on_ge_8_of_12": bool(comp["M_joint_beats_M_post"] >= 8),
            "M_joint_mean_le_T1_joint_mean_plus_0.005":
                bool(m["all_device_mean"] <= t1["all_device_mean"] + 0.005),
            "M_joint_zero_failed": bool(m["n_failed"] == 0)}
        result["criterion"] = {"checks": checks,
                               "decision": "continue" if all(checks.values()) else "kill"}
    elif real:
        result["criterion"] = "not evaluated (non-preregistered or partial configuration)"
    if not real:
        result["components_clean_check_max_rel"] = (
            float(max(c for _, c in comps.values())) if comps else None)
        ident = {}
        for lab, by_tag in points.items():
            rows_c = []
            for tag, pts in by_tag.items():
                arrays = comps[tag][0]
                allv = [pts["g"], pts["resid"]] + [arrays[c] for c in COMPONENTS]
                mask = finite_mask(allv)
                rows_c.append({"g": pts["g"][mask], "resid": pts["resid"][mask],
                               **{c: arrays[c][mask] for c in COMPONENTS}})
            ident[lab] = correlation_stats(rows_c)
        result["identifiability"] = ident
        if {"M_joint", "C0_refit"} <= set(summary):
            mj, c0m = summary["M_joint"]["all_device_mean"], summary["C0_refit"]["all_device_mean"]
            mp = summary.get("M_post", {}).get("all_device_mean", float("nan"))
            result["check_S"] = {"M_joint_le_0.90_C0_refit": bool(mj <= 0.90 * c0m),
                                 "ratio_M_joint_over_C0_refit": mj / c0m,
                                 "ratio_M_post_over_C0_refit": mp / c0m,
                                 "evaluated_on_full_config": preregistered,
                                 "note": "secondary; does not change the REAL_DEV decision"}
    result["heldout_rows"] = {lab: {t: {k: v for k, v in r.items() if k != "params"}
                                    for t, r in v.items()} for lab, v in rows.items()}
    result["runtime_s"] = round(time.time() - t_start, 1)
    target = OUT / args.name
    target.mkdir(parents=True, exist_ok=True)
    (target / "results.json").write_text(json.dumps(result, indent=1) + "\n")
    (target / "corrections.json").write_text(json.dumps(
        {f.name: {"arm": f.arm, "corrections": [c.to_json() for c in f.corrs],
                  "train_cards": f.cards} for f in folds}) + "\n")
    (target / "heldout_cards.json").write_text(json.dumps(
        {lab: {t: r["params"] for t, r in v.items()} for lab, v in rows.items()}) + "\n")
    if points:
        np.savez_compressed(target / "points.npz", **{
            f"{lab}|{tag}|{k}": v for lab, d in points.items() for tag, p in d.items()
            for k, v in p.items()})
    print(json.dumps({"summary": summary, "wins": comp, "criterion": result.get("criterion"),
                      "check_S": result.get("check_S"), "timing": timing}, indent=1))
    print(f"runtime {result['runtime_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
