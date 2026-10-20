#!/usr/bin/env python3
"""I4-1: baselines for the I3-2 bias-only correction; I4-2 interpretation.

Pre-registration: ``out/research/explore/I4_1/prereg.json``.

Arms (held-out card refit in the loop, I3-2 settings and folds):
  B1     wider card parameter set (screened extras, ``--mode screen``), no correction
  B2     extrinsic series resistance per side + first-order self-heating factor
  B3_d2  shared polynomial correction, total degree 2, per polarity
  B3_d3  same, degree 3
  B4     GP discrepancy posterior mean (Kennedy-O'Hagan-type)
  B5     per-device M_bias-form MLP fitted on the held-out device (in-sample reference)
  C0_refit_rerun  reproduction check of this script's refit path (not in the rule)
Stored rows: M_bias, C0_refit, T1_inloop from out/research/explore/I3_2/<name>/results.json.

Modes:
  screen   B1 edge screen on the starting cards (sky130 REAL_DEV, GF180, IHP)
  real     REAL_DEV leave-one-device-out
  pdk2     GF180 / IHP tier c sets 0-2, leave-one-geometry-out
  decide   pre-registered rule from real_dev / gf180_c_logo / ihp_c_logo
  analyze  I4-2 (stored I3-2 fold models; no simulation)

  PYTHONPATH=src .venv/bin/python scripts/research/explore/i4_1_baselines.py \
      --mode real --name real_dev --workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cryoml.explore.e1_correction import Correction, score_device  # noqa: E402
from cryoml.explore.i3_2_bias import HYPER_BIAS, fit_bias, paired_ratio  # noqa: E402
from cryoml.explore.i4_1_baselines import (  # noqa: E402
    B1_THRESHOLD, B2_REXT_FACTOR, BSIM4_ZERO_RANGE, PSP_B1_CANDIDATES, AugProblem,
    b1_extra_box, b2_box, correction_from_json_any, decide_rule, edge_response,
    extend_box, fit_gp, fit_poly, g_grid, linear_vth, matched_nfev, measured_pmax,
    region_masks, region_stats, t1_comparison)

E1 = ROOT / "out/research/explore/E1"
I32 = ROOT / "out/research/explore/I3_2"
OUT = ROOT / "out/research/explore/I4_1"
PREREG = OUT / "prereg.json"
SCREEN = OUT / "b1_screen.json"
SPEC = ROOT / "out/cold_loop/full_direct/round_3/spec.json"
SENS = ROOT / "out/pdk_parameter_sensitivity"
PREREG_OPTIONS = {"heldout_nfev": 400, "heldout_irls": 2, "diff_step": 0.01}
ARMS = ("B1", "B2", "B3_d2", "B3_d3", "B4", "B5")
COST = {"B1": 0, "B5": 1, "B2": 2, "B4": 3, "B3_d3": 4, "B3_d2": 5, "C0_refit_rerun": 6}
_SPEC: dict | None = None


def _spec() -> dict:
    global _SPEC
    if _SPEC is None:
        _SPEC = json.loads(SPEC.read_text())
    return _SPEC


# ---------------------------------------------------------------------------
# Problems and starting cards
# ---------------------------------------------------------------------------
def base_setup(rec):
    """(base box, start params, make_problem(box), sim_fn(problem) -> f(card, rext))."""
    if rec.source == "sky130":
        from cold_loop import box_for
        from cryoml.ngspice_fit import DeviceProblem
        from cryoml.spice_pdk import simulate_pdk
        spec = _spec()
        box = box_for(spec, rec.tag)
        start = {**spec["published_by_device"][rec.tag],
                 **{k: v for k, v in rec.meta["params"].items() if k in box.names}}

        def make(b):
            return DeviceProblem(rec.tag, b)

        def sim_fn(prob):
            d = prob.device
            return lambda card, rext: simulate_pdk(d.dev_type, d.L_um, d.W_um, prob.curves,
                                                   params=card, bin_index=prob.bin_index,
                                                   rs_ext_ohm=rext)
        return box, start, make, sim_fn
    from cryoml.pdk2 import boxes
    from cryoml.pdk2.problem import TRUTH_ROOTS, Pdk2Problem
    m = rec.meta
    box = boxes.box_for(m["fit"], m["truth_tag"])

    def make(b):
        return Pdk2Problem.from_truth(m["truth_tag"], b, m["tier"], int(m["set"]),
                                      TRUTH_ROOTS[rec.source])

    def sim_fn(prob):
        return lambda card, rext: prob.backend.simulate(prob.device, card, curves=prob.curves,
                                                        temp_C=prob.temp_C, rs_ext_ohm=rext)
    return box, dict(m["params"]), make, sim_fn


def b1_values(rec, names) -> dict[str, float]:
    """Card values of candidate extras for ``rec``'s device (names the card
    does not define are left out)."""
    if rec.source == "sky130":
        rb = json.loads((SENS / "readbacks.json").read_text())["devices"][rec.tag]["published"]
        return {n: float(rb[n]) for n in names if n in rb}
    from cryoml.pdk2 import boxes
    backend = boxes.backend(rec.source)
    device = backend.device_by_tag(rec.meta["truth_tag"])
    card = backend.stock_card(device)[1]
    out = {}
    for n in names:
        try:
            if rec.source == "ihp":
                out[n] = float(backend.card_value(device, n, card))
            else:
                out[n] = float(backend.stock_params(device, [n])[n])
        except (KeyError, ValueError, RuntimeError):
            continue
    return out


def b1_universe(rec, base_names) -> tuple[list[str], dict | None]:
    if rec.source == "ihp":
        return [n for n in PSP_B1_CANDIDATES if n not in base_names], None
    names = list(json.loads((SENS / "summary.json").read_text())["parameters"])
    return [n for n in names if n not in base_names], BSIM4_ZERO_RANGE


def b1_extras(rec, base_box, start, names=None) -> dict:
    universe, zr = b1_universe(rec, set(base_box.names))
    if names is not None:
        universe = [n for n in universe if n in set(names)]
    values = b1_values(rec, universe)
    return b1_extra_box(universe, values, zr, rdsw=start.get("rdsw"))


def rext_max(rec, start: dict) -> float:
    """B2_REXT_FACTOR * series resistance per side of the starting card / W."""
    if rec.source == "sky130":
        from cryoml.card_eval import device_by_tag
        from cryoml.spice_pdk import device_mult
        d = device_by_tag(rec.tag)
        per_side = 0.5 * float(start["rdsw"]) / (rec.W_um * device_mult(d.dev_type, d.L_um, d.W_um))
    elif rec.source == "gf180":
        per_side = 0.5 * float(start["rdsw"]) / rec.W_um
    else:
        per_side = float(start["rsw1"]) / rec.W_um
    return B2_REXT_FACTOR * per_side


def record_from_sims(rec, prob, sims, params):
    """Record of ``rec``'s curves with new simulated currents (``sims`` in
    ``prob.curves`` order)."""
    from cryoml.explore.i2_1_joint import curve_key, with_sims
    if rec.source == "sky130":
        from cryoml.metrics import tag_metric_curves
        tm = tag_metric_curves(prob.curves)
        idx = {id(c): i for i, c in enumerate(prob.curves)}
        ordered = sorted(tm.items())
        keys = [f"{k}@{b:g}" for (k, b), _ in ordered]
        if keys != [c.key for c in rec.curves]:
            raise RuntimeError(f"{rec.tag}: curve order differs from the record")
        new = with_sims(rec, [sims[idx[id(c)]] for _, c in ordered], params=params)
    else:
        keys = [curve_key(c) for c in prob.curves]
        if keys != [c.key for c in rec.curves]:
            raise RuntimeError(f"{rec.tag}: curve order differs from the record")
        new = with_sims(rec, sims, params=params)
    return new


# ---------------------------------------------------------------------------
# Worker tasks
# ---------------------------------------------------------------------------
def train_task(task):
    kind, arm, recs, hyper = task
    if kind == "poly":
        return fit_poly(arm, int(arm[-1]), recs, hyper=hyper)
    if kind == "gp":
        return fit_gp(arm, recs)
    raise KeyError(kind)


def _polish(prob, start_local, nfev, opts):
    from cryoml.ngspice_fit import polish
    return polish(prob, start_local, max_nfev=nfev, irls_rounds=opts["heldout_irls"],
                  diff_step=opts["diff_step"])


def refit_task(task):
    """One held-out refit; returns the result row."""
    from cryoml.explore.i2_1_joint import CorrectedProblem
    label, rec, payload, opts = task
    t0 = time.time()
    base_box, start, make, sim_fn = base_setup(rec)
    kind = payload["kind"]
    nfev = opts["heldout_nfev"]
    row = {"label": label, "tag": rec.tag}
    corr_final = Correction("C0")
    if kind in ("corr", "b5"):
        base = make(base_box)
        if kind == "b5":
            g0 = fit_bias("M_bias", [rec], hyper=payload.get("hyper"))
            corr = g0
            row["rrms_start_g0_insample"] = float(score_device(rec, g0.apply(rec))["rrms"])
        else:
            corr = payload["corr"]
        prob = CorrectedProblem(base, corr, rec.family, rec.L_um, rec.W_um, rec.vdd, rec.source)
        res = _polish(prob, base_box.params_to_local(start), nfev, opts)
        params = base_box.local_to_params(res["local"])
        new = record_from_sims(rec, base, base.simulate(res["local"]), params)
        corr_final = corr
        if kind == "b5":
            g1 = fit_bias("M_bias", [new], hyper=payload.get("hyper"))
            row["rrms_g0_on_refit_card"] = float(score_device(new, g0.apply(new))["rrms"])
            corr_final = g1
            row["corrections"] = {"g0": g0.to_json(), "g1": g1.to_json()}
        n_sim = base.n_sim
        row["n_active"] = int(base_box.n_active)
    elif kind == "b1":
        extras = b1_extras(rec, base_box, start, payload["names"])
        box = extend_box(base_box, extras)
        prob = make(box)
        nfev = matched_nfev(base_box.n_active, box.n_active, nfev, opts["heldout_irls"])
        local0 = box.params_to_local({**start, **{n: e["value"] for n, e in extras.items()}})
        res = _polish(prob, local0, nfev, opts)
        params = box.local_to_params(res["local"])
        new = record_from_sims(rec, prob, prob.simulate(res["local"]), params)
        n_sim = prob.n_sim
        row.update(n_active=int(box.n_active), n_extra_active=int(box.n_active - base_box.n_active),
                   max_nfev=nfev, extras={n: e["bounds"] for n, e in extras.items()})
    elif kind == "b2":
        base = make(base_box)
        rmax = rext_max(rec, start)
        box = b2_box(base_box, rmax)
        aug = AugProblem(base, box, sim_fn(base), measured_pmax(rec))
        nfev = matched_nfev(base_box.n_active, box.n_active, nfev, opts["heldout_irls"])
        local0 = box.params_to_local({**start, "rext_ohm": 0.0, "heat_kappa": 0.0})
        res = _polish(aug, local0, nfev, opts)
        params = box.local_to_params(res["local"])
        new = record_from_sims(rec, base, aug.simulate(res["local"]), params)
        n_sim = aug.n_sim
        row.update(n_active=int(box.n_active), max_nfev=nfev, rext_max_ohm=rmax,
                   pmax_w=aug.pmax, rext_ohm=params["rext_ohm"], heat_kappa=params["heat_kappa"])
    else:
        raise KeyError(kind)
    before = score_device(rec, corr_final.apply(rec)) if kind != "b5" else None
    after = score_device(new, corr_final.apply(new))
    row.update(rrms=after["rrms"], failed=after["failed"], n_sim=n_sim,
               history=res["history"], params=params, runtime_s=round(time.time() - t0, 2))
    if before is not None:
        row["rrms_start"] = before["rrms"]
    return row


def screen_task(task):
    """Edge responses of the B1 candidates on one starting card."""
    rec = task
    base_box, start, make, sim_fn = base_setup(rec)
    prob = make(base_box)
    extras = b1_extras(rec, base_box, start)
    sim = sim_fn(prob)
    ref = sim(start, 0.0)
    out = {}
    for name, e in extras.items():
        lo, hi = e["bounds"]
        if not hi > lo:
            out[name] = {"value": e["value"], "bounds": e["bounds"], "active": False}
            continue
        resp = [edge_response(ref, sim({**start, name: v}, 0.0), prob.layout) for v in (lo, hi)]
        out[name] = {"value": e["value"], "bounds": e["bounds"], "log": e["log"],
                     "active": True, "response_lo": resp[0], "response_hi": resp[1]}
    return rec.tag, out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def aggregate(values: dict) -> dict:
    from e1_lodo import aggregate as agg
    return agg(values)


def load_records(mode: str, pdk: str):
    from i2_1_joint import load_records as lr
    return lr(argparse.Namespace(mode=mode, pdk=pdk))


def stored_rows(real: bool, pdk: str) -> dict[str, dict]:
    name = "real_dev" if real else f"{pdk}_c_logo"
    per = json.loads((I32 / name / "results.json").read_text())["per_device"]
    return {k: per[k] for k in ("M_bias", "C0_refit", "T1_inloop")}


def screened_names(dataset: str) -> list[str]:
    return list(json.loads(SCREEN.read_text())[dataset]["selected"])


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def run_screen(args) -> int:
    t0 = time.time()
    out = {}
    jobs = {}
    recs, _, _ = load_records("real", "gf180")
    jobs["sky130_real_dev"] = recs
    for pdk in ("gf180", "ihp"):
        recs, key, _ = load_records("pdk2", pdk)
        jobs[pdk] = [r for r in recs if int(r.meta["set"]) == 0]
    with get_context("spawn").Pool(args.workers) as pool:
        for ds, recs in jobs.items():
            rows = dict(pool.map(screen_task, recs, chunksize=1))
            names = sorted({n for r in rows.values() for n in r})
            per = {}
            for n in names:
                resp = [max(v.get("response_lo", 0) if np.isfinite(v.get("response_lo", 0)) else 0,
                            v.get("response_hi", 0) if np.isfinite(v.get("response_hi", 0)) else 0)
                        for v in (r[n] for r in rows.values() if n in r) if v.get("active")]
                fails = sum(1 for r in rows.values() if n in r and r[n].get("active") and not (
                    np.isfinite(r[n]["response_lo"]) and np.isfinite(r[n]["response_hi"])))
                per[n] = {"max_response": float(max(resp)) if resp else 0.0,
                          "n_active_devices": len(resp),
                          "n_devices_response_ge_threshold": int(sum(x >= B1_THRESHOLD for x in resp)),
                          "n_edge_nonfinite_devices": int(fails)}
            sel = [n for n in names if per[n]["max_response"] >= B1_THRESHOLD]
            out[ds] = {"n_devices": len(recs), "candidates": names, "selected": sel,
                       "per_parameter": per, "per_device": rows}
            print(f"{ds}: {len(names)} candidates, {len(sel)} selected: {sel}", flush=True)
    out["rule"] = (f"candidate kept if the max over the dataset's starting cards of the "
                   f"edge response (max over included curves of trimmed RMS(I_edge - I_start) "
                   f"/ paper denominator) at either box edge is >= {B1_THRESHOLD}")
    out["runtime_s"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    SCREEN.write_text(json.dumps(out, indent=1) + "\n")
    print(f"runtime {out['runtime_s']} s")
    return 0


def run_heldout(args) -> int:
    t_start = time.time()
    real = args.mode == "real"
    arms = [a for a in args.arms.split(",") if a]
    opts = {"heldout_nfev": args.heldout_nfev, "heldout_irls": args.heldout_irls,
            "diff_step": args.diff_step}
    hyper = {}
    if args.m_steps:
        hyper["steps"] = args.m_steps
    hyper_b5 = dict(hyper)
    if args.m_seeds:
        hyper_b5["seeds"] = tuple(int(x) for x in args.m_seeds.split(","))
    preregistered = (opts == PREREG_OPTIONS and not hyper_b5 and not args.folds_limit
                     and set(ARMS) <= set(arms))
    records, key, _ = load_records(args.mode, args.pdk)
    groups = sorted({key(r) for r in records})
    if args.folds_limit:
        groups = groups[:args.folds_limit]
    held = {g: [r for r in records if key(r) == g] for g in groups}
    held_recs = [r for g in groups for r in held[g]]
    clusters = None if real else {r.tag: r.geometry for r in held_recs}
    dataset = "sky130_real_dev" if real else args.pdk
    b1_names = screened_names(dataset) if "B1" in arms else []
    print(f"I4_1 {args.mode} {dataset}: {len(groups)} folds, {len(held_recs)} held-out, "
          f"arms {arms}, opts {opts}, hyper {hyper_b5}, preregistered={preregistered}, "
          f"B1 extras {len(b1_names)}", flush=True)

    train_tasks, train_names = [], []
    for g in groups:
        tr = [r for r in records if key(r) != g]
        for arm in ("B3_d2", "B3_d3"):
            if arm in arms:
                train_tasks.append(("poly", arm, tr, hyper))
                train_names.append(f"{arm}|{g}")
        if "B4" in arms:
            train_tasks.append(("gp", "B4", tr, None))
            train_names.append(f"B4|{g}")
    timing = {}
    with get_context("spawn").Pool(args.workers) as pool:
        t0 = time.time()
        corrs = dict(zip(train_names, pool.map(train_task, train_tasks, chunksize=1)))
        timing["train_s"] = round(time.time() - t0, 1)
        print(f"training done ({timing['train_s']} s, {len(train_tasks)} tasks)", flush=True)
        tasks = []
        for g in groups:
            for rec in held[g]:
                for arm in arms:
                    if arm in ("B3_d2", "B3_d3", "B4"):
                        tasks.append((arm, rec, {"kind": "corr", "corr": corrs[f"{arm}|{g}"]}, opts))
                    elif arm == "B1":
                        tasks.append((arm, rec, {"kind": "b1", "names": b1_names}, opts))
                    elif arm == "B2":
                        tasks.append((arm, rec, {"kind": "b2"}, opts))
                    elif arm == "B5":
                        tasks.append((arm, rec, {"kind": "b5", "hyper": hyper_b5}, opts))
                    elif arm == "C0_refit_rerun":
                        tasks.append((arm, rec, {"kind": "corr", "corr": Correction("C0")}, opts))
        tasks.sort(key=lambda t: COST.get(t[0], 9))
        t0 = time.time()
        out = pool.map(refit_task, tasks, chunksize=1)
        timing["heldout_refit_s"] = round(time.time() - t0, 1)
        timing["heldout_refit_n"] = len(tasks)

    rows = {}
    for row in out:
        rows.setdefault(row["label"], {})[row["tag"]] = row
    held_tags = [r.tag for r in held_recs]
    scores = {lab: {t: r["rrms"] for t, r in v.items()} for lab, v in rows.items()}
    stored = {lab: {t: v[t] for t in held_tags} for lab, v in stored_rows(real, args.pdk).items()}
    scores.update(stored)
    summary = {lab: aggregate(v) for lab, v in scores.items()}
    comparisons = {}
    for b in [a for a in scores if a not in ("M_bias",)]:
        comparisons[f"M_bias_vs_{b}"] = paired_ratio(scores["M_bias"], scores[b], clusters)
    for b in [a for a in scores if a not in ("C0_refit",)]:
        comparisons[f"{b}_vs_C0_refit"] = paired_ratio(scores[b], scores["C0_refit"], clusters)
    repro = {}
    if "C0_refit_rerun" in scores:
        d = [abs(scores["C0_refit_rerun"][t] - scores["C0_refit"][t])
             for t in scores["C0_refit_rerun"]]
        repro["C0_refit_max_abs_diff_rerun_vs_stored"] = float(max(d)) if d else float("nan")
    result = {"name": args.name, "mode": args.mode, "dataset": dataset,
              "prereg": str(PREREG.relative_to(ROOT)), "preregistered_config": preregistered,
              "smoke": not preregistered, "options": opts, "hyper_mlp_b5": HYPER_BIAS,
              "hyper_overrides": hyper_b5, "arms": arms, "n_folds": len(groups),
              "n_heldout_records": len(held_recs), "b1_extras": b1_names,
              "stored_sources": f"out/research/explore/I3_2/{'real_dev' if real else args.pdk + '_c_logo'}"
                                "/results.json per_device (M_bias, C0_refit, T1_inloop)",
              "summary": summary, "comparisons": comparisons, "reproduction": repro,
              "per_device": scores, "timing": timing}
    target = OUT / args.name
    target.mkdir(parents=True, exist_ok=True)
    result["heldout_rows"] = {lab: {t: {k: v for k, v in r.items()
                                        if k not in ("params", "corrections")}
                                    for t, r in v.items()} for lab, v in rows.items()}
    result["runtime_s"] = round(time.time() - t_start, 1)
    (target / "results.json").write_text(json.dumps(result, indent=1) + "\n")
    corr_out = {n: c.to_json() for n, c in corrs.items()}
    corr_out.update({f"B5|{t}": r["corrections"] for t, r in rows.get("B5", {}).items()})
    (target / "corrections.json").write_text(json.dumps(corr_out) + "\n")
    (target / "heldout_cards.json").write_text(json.dumps(
        {lab: {t: r["params"] for t, r in v.items()} for lab, v in rows.items()}) + "\n")
    print(json.dumps({"summary": {k: {kk: v[kk] for kk in ("all_device_mean", "combined",
                                                            "median", "n_failed")}
                                  for k, v in summary.items()},
                      "comparisons": {k: {kk: v.get(kk) for kk in ("ratio", "ci_low", "ci_high",
                                                                    "wins", "n_used")}
                                      for k, v in comparisons.items() if k.startswith("M_bias_vs")},
                      "reproduction": repro, "timing": timing}, indent=1))
    print(f"runtime {result['runtime_s']} s")
    return 0


def run_decide() -> int:
    res = {}
    for name in ("real_dev", "gf180_c_logo", "ihp_c_logo"):
        p = OUT / name / "results.json"
        res[name] = json.loads(p.read_text()) if p.exists() else None
    missing = [k for k, v in res.items() if v is None or not v.get("preregistered_config")]
    if missing:
        print(f"not evaluated: missing or non-preregistered results {missing}")
        return 1
    cmp_ = {k: {b: v["comparisons"].get(f"M_bias_vs_{b}", {})
                for b in ("B1", "B2", "B3_d2", "B3_d3", "B4", "B5")} for k, v in res.items()}
    out = decide_rule(cmp_["real_dev"], cmp_["gf180_c_logo"], cmp_["ihp_c_logo"])
    out.update({"prereg": str(PREREG.relative_to(ROOT)),
                "B5_reference_not_in_rule": {k: v["B5"] for k, v in cmp_.items()},
                "note": "REAL_DEV results are development evidence (prereg data_reuse_note)"})
    (OUT / "decision.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


def run_analyze(args) -> int:
    """I4-2: g surfaces, region statistics and T1 comparison from the stored
    I3-2 fold models (no simulation)."""
    t0 = time.time()
    from cryoml.explore.i2_1_joint import log_ratio
    from cryoml.explore.i3_2_bias import correction_from_json
    from cryoml.metrics import clean_current
    out = {}
    grids = {}
    for name, mode, pdk in (("real_dev", "real", "gf180"), ("gf180_c_logo", "pdk2", "gf180"),
                            ("ihp_c_logo", "pdk2", "ihp")):
        cpath = I32 / name / "corrections.json"
        if not cpath.exists():
            continue
        stored = json.loads(cpath.read_text())
        records, key, g0 = load_records(mode, pdk)
        groups = sorted({key(r) for r in records})
        folds = {g: correction_from_json(stored[f"M_bias|{g}"]) for g in groups}
        surf = {}
        for fam, p in (("nmos", 0.0), ("pmos", 1.0)):
            stack = np.stack([g_grid(c, p, args.grid) for c in folds.values()])
            surf[fam] = {"mean": stack.mean(0), "std": stack.std(0),
                         "fold_min": stack.min(0), "fold_max": stack.max(0)}
        grids[name] = surf
        pts = {"nmos": {"g": [], "resid": [], "masks": []}, "pmos": {"g": [], "resid": [], "masks": []}}
        t1_rows, vths = [], {}
        for rec in records:
            corr = folds[key(rec)]
            t1 = Correction.from_json(g0[key(rec)]["T1"])
            vth = linear_vth(rec)
            vths[rec.tag] = vth
            for c in rec.curves:
                if not c.included or not (c.den > 0):
                    continue
                s = slice(c.start, None)
                vg, vd, sim = (np.asarray(c.vg, float)[s], np.asarray(c.vd, float)[s],
                               np.asarray(c.sim, float)[s])
                from cryoml.explore.e1_correction import raw_features
                phi = raw_features(rec, vg, vd, sim)
                g = corr.g(phi)
                masks = region_masks(vg, vd, vth, rec.vdd)
                P = pts[rec.family]
                P["g"].append(g)
                P["resid"].append(log_ratio(clean_current(c.target)[s], sim))
                P["masks"].append(masks)
                grp = f"{rec.family}|{c.key}"
                t1_rows.append({"tag": rec.tag, "group": grp, "t1": float(t1.table.get(grp, 0.0)),
                                "g_mean": float(np.mean(g))})
        regions = {}
        for fam, P in pts.items():
            if not P["g"]:
                continue
            g = np.concatenate(P["g"])
            masks = {k: np.concatenate([m[k] for m in P["masks"]]) for k in P["masks"][0]}
            regions[fam] = region_stats(g, masks, np.concatenate(P["resid"]))
        out[name] = {
            "n_folds": len(groups), "grid_n": args.grid,
            "surface_summary": {fam: {"mean_of_mean": float(s["mean"].mean()),
                                      "max_abs_mean": float(np.abs(s["mean"]).max()),
                                      "mean_fold_std": float(s["std"].mean()),
                                      "max_fold_std": float(s["std"].max())}
                                for fam, s in surf.items()},
            "vth_linear_extrapolation": vths, "regions": regions,
            "t1_vs_g": t1_comparison(t1_rows), "t1_rows": t1_rows}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "i4_2_g_surfaces.npz", **{
        f"{ds}__{fam}__{k}": v for ds, s in grids.items() for fam, d in s.items()
        for k, v in d.items()})
    out["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "i4_2_analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for ds, s in grids.items():
            fig, ax = plt.subplots(2, 2, figsize=(9, 7.5), constrained_layout=True)
            for i, fam in enumerate(("nmos", "pmos")):
                for j, (k, cmap) in enumerate((("mean", "RdBu_r"), ("std", "viridis"))):
                    z = s[fam][k]
                    kw = {"vmin": -np.abs(z).max(), "vmax": np.abs(z).max()} if k == "mean" else {}
                    im = ax[i, j].imshow(z, origin="lower", extent=(0, 1, 0, 1), cmap=cmap,
                                         aspect="auto", **kw)
                    ax[i, j].set_xlabel("|Vds| / VDD")
                    ax[i, j].set_ylabel("|Vgs| / VDD")
                    ax[i, j].set_title(f"{fam}: g {k} over {out[ds]['n_folds']} fold models")
                    fig.colorbar(im, ax=ax[i, j])
            fig.suptitle(f"I4-2 M_bias g surface ({ds})")
            fig.savefig(OUT / f"i4_2_g_surface_{ds}.png", dpi=130)
            plt.close(fig)
    except ImportError:
        pass
    print(json.dumps({k: {kk: v[kk] for kk in ("surface_summary", "regions", "t1_vs_g")}
                      for k, v in out.items() if isinstance(v, dict)}, indent=1, default=str))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("screen", "real", "pdk2", "decide", "analyze"),
                    required=True)
    ap.add_argument("--pdk", choices=("gf180", "ihp"), default="gf180")
    ap.add_argument("--name", default="")
    ap.add_argument("--arms", default=",".join(ARMS + ("C0_refit_rerun",)))
    ap.add_argument("--heldout-nfev", type=int, default=PREREG_OPTIONS["heldout_nfev"])
    ap.add_argument("--heldout-irls", type=int, default=PREREG_OPTIONS["heldout_irls"])
    ap.add_argument("--diff-step", type=float, default=PREREG_OPTIONS["diff_step"])
    ap.add_argument("--folds-limit", type=int, default=0, help="smoke: first N folds")
    ap.add_argument("--m-seeds", default="", help="smoke override of B5 seeds")
    ap.add_argument("--m-steps", type=int, default=0, help="smoke override of B3/B5 steps")
    ap.add_argument("--grid", type=int, default=41, help="analyze: grid size")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.mode == "screen":
        return run_screen(args)
    if args.mode == "decide":
        return run_decide()
    if args.mode == "analyze":
        return run_analyze(args)
    if not args.name:
        ap.error("--name is required")
    return run_heldout(args)


if __name__ == "__main__":
    raise SystemExit(main())
