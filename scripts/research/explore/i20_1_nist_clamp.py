#!/usr/bin/env python3
"""I20-1: BSIM4 A2 clamp state of the NIST decks (I8-1 lvt 77 K card, I9-1
room-temperature card), audit and fix; state-C reruns; selected g v2 recipe on
both NIST DEV datasets.

Pre-registration: out/research/explore/I20_1/prereg.md (prereg.json), a516fe3.

Stages (``--stage``; ``--ds nist4 | nistrt`` where marked):

  audit    stored NIST arms rescored in states L, C, U; A2_eff, b4check.c
           lines per card; V1-V3 -> audit/audit.json
  start    state-C starting records of both datasets (V8) -> start/
  s3v1     S3_D, v1 recipe, state-C records (+ state-L reproduction of 2
           I19-1 values) -> s3/<ds>_v1.json
  base     --ds: C0_refit and M_bias v1 LODO in state C (+ state-L refits of
           2 devices vs the stored rows) -> base_<ds>/
  pens     73 x 73 forward-quadrant grids of the state-C starting cards -> pens/
  select   --ds: 6-block nested lambda selection, full-data selection and the
           5-seed full-data model -> select_<ds>/
  s3v2     S3_D, g v2 recipe (lambda*_k) -> s3/<ds>_v2.json
  lodo     --ds: selected g v2 LODO (5 seeds at lambda*_h) -> lodo_<ds>/
  decide   validity, labels, gate, amendment trigger -> decision.json

  .venv/bin/python -I scripts/research/explore/i20_1_nist_clamp.py --stage audit --workers 8
"""
from __future__ import annotations

import argparse
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

import i8_1_nist as d8  # noqa: E402
import i9_1_nist_rt as d9  # noqa: E402
from cryoml.explore import i8_1_nist as b8  # noqa: E402
from cryoml.explore import i9_1_nist_rt as b9  # noqa: E402
from cryoml.explore import i12_1_symm as v2  # noqa: E402
from cryoml.explore import i20_1_nist_clamp as m  # noqa: E402

EXP = ROOT / "out/research/explore"
OUT = EXP / "I20_1"
PREREG = OUT / "prereg.json"
UNIT_TESTS = ("tests.test_i20_1_nist_clamp", "tests.test_i8_1_nist", "tests.test_i9_1_nist_rt",
              "tests.test_i12_1_symm", "tests.test_i15_2_ablate")
HELDOUT = {"heldout_nfev": 400, "heldout_irls": 2, "diff_step": 0.01}
S3_HYPER = {"seeds": (0,), "steps": 500}
SRC = {"nist4": "I8_1", "nistrt": "I9_1"}
SMOKE_N = 6
_SMOKE = {"on": os.environ.get("CRYOML_I20_1_SMOKE") == "1"}


def jload(p):
    return json.loads(Path(p).read_text())


def jdump(obj, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    b8.write_json(Path(p), obj)


def base_dir() -> Path:
    return OUT / "smoke" if _SMOKE["on"] else OUT


def mod(ds):
    return b8 if ds == "nist4" else b9


def drv(ds):
    return d8 if ds == "nist4" else d9


# ---------------------------------------------------------------------------
# dataset access (cached per process)
# ---------------------------------------------------------------------------
_C: dict = {}


def devices(ds: str) -> dict:
    k = ("dev", ds)
    if k not in _C:
        devs, _ = drv(ds).dev_devices()
        _C[k] = {d.tag: d for d in devs}
    return _C[k]


def bins(ds: str) -> dict:
    k = ("bins", ds)
    if k not in _C:
        _C[k] = jload(EXP / SRC[ds] / "start" / "bins.json")["devices"]
    return _C[k]


def stored_records(ds: str) -> dict:
    k = ("srec", ds)
    if k not in _C:
        from cryoml.explore.e1_correction import load_devices
        _C[k] = {r.tag: r for r in load_devices(EXP / SRC[ds] / "start" / "records")}
    return _C[k]


def c_records_path(ds: str) -> Path:
    return base_dir() / "start" / f"records_{ds}"


def c_records(ds: str) -> dict:
    """State-C starting records (all DEV devices, load order = sorted tag)."""
    k = ("crec", ds)
    if k not in _C:
        from cryoml.explore.e1_correction import load_devices
        _C[k] = {r.tag: r for r in load_devices(c_records_path(ds))}
    return _C[k]


def finite_order(ds: str) -> list[str]:
    """Finite state-C starting records, load order (sorted by tag)."""
    good, _ = b8.split_finite(list(c_records(ds).values()))
    order = sorted(r.tag for r in good)
    if _SMOKE["on"]:
        order = smoke_tags(ds, order)
    return order


def smoke_tags(ds: str, order: list[str]) -> list[str]:
    n = [t for t in order if not is_pmos(ds, t)][:SMOKE_N // 2]
    p = [t for t in order if is_pmos(ds, t)][:SMOKE_N // 2]
    return sorted(n + p)


def is_pmos(ds: str, tag: str) -> bool:
    return tag.startswith("pfet")


def problem(ds: str, tag: str, state: str = "C", stock=None):
    dev, bi = devices(ds)[tag], bins(ds)[tag]
    st = bi["stock"] if stock is None else stock
    cls = b8.MeasProblem if ds == "nist4" else b9.MeasProblemRT
    p = cls(dev, bi, st, b8.base_box(bi["stock"]))
    p.bsim4_state = m.STATE_ARG[state]
    return p


def sim_card(ds: str, tag: str, params: dict | None, state: str, stock_ref: dict | None = None,
             rs_ext_ohm: float = 0.0):
    """Simulate the stored card ``params`` (None: stock) on the measured grid."""
    dev, bi = devices(ds)[tag], bins(ds)[tag]
    ref = bi["stock"] if stock_ref is None else stock_ref
    ch = b8.changed_params(params, ref) if params is not None else None
    curves = b8.to_pipeline_curves(dev)
    return mod(ds).simulate(dev.polarity, dev.W_um, dev.L_um, bi["bin"], curves, ch, rs_ext_ohm,
                            bsim4_state=m.STATE_ARG[state])


def a2_eff(ds: str, tag: str, params: dict | None, stock_ref: dict | None = None) -> float:
    dev, bi = devices(ds)[tag], bins(ds)[tag]
    ref = bi["stock"] if stock_ref is None else stock_ref
    ch = b8.changed_params(params, ref) if params is not None else {}
    if ds == "nist4":
        sp = b8._sp()
        span = sp._bin_text(dev.polarity, bi["bin"])
        return float(ch["a2"]) if "a2" in ch else float(sp._bin_value(span, "a2") or 1.0)
    return float(b9.effective_a1a2(dev.polarity, bi["bin"], dev.W_um, dev.L_um, ch)["a2"])


def dev_check(ds: str, tags) -> dict:
    s = jload(EXP / SRC[ds] / "split.json")
    tags = set(tags)
    in_test, not_dev = sorted(tags & set(s["TEST"])), sorted(tags - set(s["DEV"]))
    return {"n": len(tags), "n_in_TEST": len(in_test), "n_not_in_DEV": len(not_dev),
            "ok": not in_test and not not_dev}


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
def _nan(x):
    return float("nan") if x is None else float(x)


def audit_inputs(ds: str) -> dict:
    """arm -> {tag: spec}; spec = dict(params, corr json or None, stored rrms, kind)."""
    E = EXP / SRC[ds]
    start = jload(E / "start" / "summary.json")["rows"]
    lodo = jload(E / "lodo" / "results.json")
    lc = jload(E / "lodo" / "cards.json")
    lco = jload(E / "lodo" / "corrections.json")
    bl = jload(E / "baselines" / "results.json")
    bc = jload(E / "baselines" / "cards.json")
    bco = jload(E / "baselines" / "corrections.json")
    tags = sorted(start)
    out = {"stock": {t: {"params": None, "corr": None, "stored": start[t]["stock_rrms"], "kind": "stock"}
                     for t in tags},
           "start": {t: {"params": start[t]["params"], "corr": None, "stored": start[t]["start_rrms"],
                         "kind": "plain"} for t in tags}}
    lpd = lodo["per_device"]
    for arm, ckey in (("C0_refit", None), ("T1_inloop", "T1_inloop"), ("M_bias_v1", "M_bias")):
        src = "M_bias" if arm == "M_bias_v1" else arm
        out[arm] = {t: {"params": lc[src][t], "corr": (lco[f"{ckey}|{t}"] if ckey else None),
                        "stored": _nan(lpd[src][t]), "kind": "plain"} for t in lc[src]}
    if ds == "nistrt":
        cj = lco["M_bias_lvt|__I8_1_DEV__"]
        out["M_bias_lvt"] = {t: {"params": lc["M_bias_lvt"][t], "corr": cj, "stored": _nan(lpd["M_bias_lvt"][t]),
                                 "kind": "plain"} for t in lc["M_bias_lvt"]}
    bpd = bl["per_device"]
    names = bl["b1_screen"]["selected"]
    out["B1"] = {t: {"params": bc["B1"][t], "corr": None, "stored": _nan(bpd["B1"][t]), "kind": "b1",
                     "b1_names": names} for t in bc["B1"]}
    out["B2"] = {t: {"params": bc["B2"][t], "corr": None, "stored": _nan(bpd["B2"][t]), "kind": "b2"}
                 for t in bc["B2"]}
    out["B3_d3"] = {t: {"params": bc["B3_d3"][t], "corr": bco[f"B3_d3|{t}"], "stored": _nan(bpd["B3_d3"][t]),
                        "kind": "plain", "loader": "any"} for t in bc["B3_d3"]}
    if ds == "nist4":
        r92 = jload(EXP / "I9_2/nist_4K/results.json")["per_device"]
        c92 = jload(EXP / "I9_2/nist_4K/cards.json")
        co92 = jload(EXP / "I9_2/nist_4K/corrections.json")
        for cls_ in sorted(c92):
            out[f"I9_2:{cls_}"] = {t: {"params": c92[cls_][t], "corr": co92[f"{cls_}|{t}"],
                                       "stored": _nan(r92[cls_][t]), "kind": "plain", "loader": "i9"}
                                   for t in c92[cls_]}
        rr = jload(EXP / "I9_2/replicate/results.json")["datasets"]["nist_4K"]["per_device"]
        rc = jload(EXP / "I9_2/replicate/cards.json")["nist_4K"]
        rco = jload(EXP / "I9_2/replicate/corrections.json")
        out["I9_2:mlp16_rep"] = {t: {"params": rc[t], "corr": rco[f"nist_4K|mlp16_rep|{t}"],
                                     "stored": _nan(rr[t]), "kind": "plain", "loader": "i9"} for t in rc}
    return out


def load_corr(spec):
    from cryoml.explore.e1_correction import Correction
    cj = spec.get("corr")
    if cj is None:
        return Correction("C0")
    if spec.get("loader") == "any":
        from cryoml.explore.i4_1_baselines import correction_from_json_any
        return correction_from_json_any(cj)
    if spec.get("loader") == "i9":
        from cryoml.explore.i9_2_classes import correction_from_json_i9
        return correction_from_json_i9(cj)
    from cryoml.explore.i3_2_bias import correction_from_json
    return correction_from_json(cj)


_B1VALS: dict = {}


def b1_stock(ds: str, tag: str, spec: dict) -> dict:
    """stock1 of the I8-1 / I9-1 B1 refit (stock + screened extras at the
    starting card values)."""
    key = (ds, tag)
    if key not in _B1VALS:
        dev, bi = devices(ds)[tag], bins(ds)[tag]
        _B1VALS[key] = mod(ds).read_params(dev.polarity, dev.W_um, dev.L_um, bi["bin"], b8.b1_universe(),
                                           required=False)
    start = stored_records(ds)[tag].meta["params"]
    extras = b8.b1_extras(_B1VALS[key], start, spec["b1_names"])
    return {**bins(ds)[tag]["stock"], **{n: e["value"] for n, e in extras.items()}}


def score_state(ds: str, tag: str, spec: dict, state: str):
    """(rrms, sims) of one stored card in one state."""
    from cryoml.explore.e1_correction import score_device
    dev, bi = devices(ds)[tag], bins(ds)[tag]
    params = spec["params"]
    if spec["kind"] == "stock":
        prob = problem(ds, tag, state)
        sims = prob.simulate_params(None)
        return float(np.mean(prob.curve_rrms(sims))), sims
    if spec["kind"] == "b2":
        from cryoml.explore.i4_1_baselines import B2_NAMES, heat_factor, measured_pmax
        card = {k: v for k, v in params.items() if k not in B2_NAMES}
        rext, kappa = float(params["rext_ohm"]), float(params["heat_kappa"])
        sims = sim_card(ds, tag, card, state, rs_ext_ohm=rext)
        pmax = measured_pmax(stored_records(ds)[tag])
        curves = b8.to_pipeline_curves(dev)
        sims = [np.asarray(s, float) * heat_factor(np.asarray(c.Vd, float)[:len(s)], np.asarray(s, float),
                                                   kappa, pmax) if kappa != 0.0 else np.asarray(s, float)
                for c, s in zip(curves, sims)]
        rec = b8.to_record(dev, bi, sims, params)
        return float(score_device(rec, sims)["rrms"]), sims
    ref = b1_stock(ds, tag, spec) if spec["kind"] == "b1" else None
    sims = sim_card(ds, tag, params, state, stock_ref=ref)
    rec = b8.to_record(dev, bi, sims, params)
    return float(score_device(rec, load_corr(spec).apply(rec))["rrms"]), sims


def audit_task(task):
    ds, arm, tag, spec = task
    from cryoml.explore.i4_1_baselines import B2_NAMES
    out = {"ds": ds, "arm": arm, "tag": tag, "stored": spec["stored"]}
    sims = {}
    for s in m.STATES:
        out[f"rrms_{s}"], sims[s] = score_state(ds, tag, spec, s)
    params = spec["params"]
    card = None if params is None else {k: v for k, v in params.items() if k not in B2_NAMES}
    ref = b1_stock(ds, tag, spec) if spec["kind"] == "b1" else None
    a2 = a2_eff(ds, tag, card, ref)
    out["a2_eff"], out["clamp"] = a2, m.clamp_class(a2)
    bi = bins(ds)[tag]
    ch = b8.changed_params(card, ref if ref is not None else bi["stock"]) if card is not None else None
    lines = m.check_lines(ds, devices(ds)[tag], bi["bin"], ch)
    out["b4check"] = lines
    out["a2_msg"] = m.a2_from_messages(lines)
    same = {f"{a}{b}": bool(all(np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True)
                                for x, y in zip(sims[a], sims[b]))) for a, b in (("C", "L"), ("U", "L"))}
    out["sims_equal"] = same
    c1 = np.asarray(sims["C"][0], float)
    l1 = np.asarray(sims["L"][0], float)
    out["curve1_finite"] = bool(np.all(np.isfinite(l1)))
    out["curve1_C_vs_L_max"] = (float(np.max(np.abs(c1 - l1))) if out["curve1_finite"] and np.all(np.isfinite(c1))
                                else None)
    return out


def stage_audit(workers: int) -> dict:
    from cryoml.explore.i3_2_bias import paired_ratio
    t0 = time.time()
    tasks = []
    inputs = {ds: audit_inputs(ds) for ds in m.DATASETS}
    for ds in m.DATASETS:
        for arm, rows in inputs[ds].items():
            tags = sorted(rows)
            if _SMOKE["on"]:
                tags = tags[:2]
            tasks += [(ds, arm, t, rows[t]) for t in tags]
    print(f"audit: {len(tasks)} cards x 3 states", flush=True)
    with get_context("spawn").Pool(workers) as pool:
        res = pool.map(audit_task, tasks, chunksize=4)
    out = {"datasets": {}, "runtime_s": None}
    v1_core, v1_by_arm, v2_n, v2_bad, v3_rel, v3_unc_msgs, v3_c1 = 0.0, {}, 0, 0, 0.0, 0, 0.0
    other_clamps = []
    for ds in m.DATASETS:
        rows = [r for r in res if r["ds"] == ds]
        arms = sorted({r["arm"] for r in rows}, key=lambda a: list(inputs[ds]).index(a))
        fam = {}
        d = {"arms": {}, "per_card": {}}
        for arm in arms:
            rr = {r["tag"]: r for r in rows if r["arm"] == arm}
            tags = sorted(rr)
            fam = {t: ("pmos" if is_pmos(ds, t) else "nmos") for t in tags}
            per = {s: {t: rr[t][f"rrms_{s}"] for t in tags} for s in m.STATES}
            stored = {t: rr[t]["stored"] for t in tags}
            v1 = m.max_abs_diff(per["L"], stored)
            v1_by_arm[f"{ds}|{arm}"] = v1
            if arm in ("start", "C0_refit", "M_bias_v1"):
                v1_core = max(v1_core, v1)
            n_cl = sum(rr[t]["clamp"] != "none" for t in tags)
            changed = [t for t in tags if m.max_abs_diff({t: per["C"][t]}, {t: per["L"][t]}) > 0]
            d["arms"][arm] = {"n": len(tags), "v1_max_abs_L_minus_stored": v1, "reproduced": v1 <= 1e-12,
                              "n_clamped": int(n_cl), "clamp_classes": {c: int(sum(rr[t]["clamp"] == c for t in tags))
                                                                        for c in ("high", "low", "none")},
                              "n_changed_C_vs_L": len(changed), "changed_C_vs_L": changed,
                              "aggregate": {s: b8.aggregate(per[s], fam) for s in m.STATES},
                              "aggregate_stored": b8.aggregate(stored, fam), "per_device": per,
                              "per_device_stored": stored,
                              "diff_C_minus_L": {t: per["C"][t] - per["L"][t] for t in tags},
                              "diff_U_minus_L": {t: per["U"][t] - per["L"][t] for t in tags}}
            for t in tags:
                r = rr[t]
                d["per_card"][f"{arm}|{t}"] = {k: r[k] for k in ("a2_eff", "clamp", "a2_msg", "b4check",
                                                                  "sims_equal", "curve1_C_vs_L_max")}
                if r["clamp"] == "none":
                    v2_n += 1
                    if not (r["sims_equal"]["CL"] and r["sims_equal"]["UL"]):
                        v2_bad += 1
                    if r["a2_msg"] is not None:
                        v3_unc_msgs += 1
                else:
                    if r["a2_msg"] is None:
                        v3_rel = float("inf")
                    else:
                        v3_rel = max(v3_rel, abs(r["a2_msg"] - r["a2_eff"]) / max(abs(r["a2_eff"]), 1e-30))
                    if r["curve1_C_vs_L_max"] is not None:
                        v3_c1 = max(v3_c1, r["curve1_C_vs_L_max"])
                for ln in r["b4check"]:
                    low = ln.lower()
                    if ("set to" in low or "reset to" in low) and "a2" not in low:
                        other_clamps.append({"ds": ds, "card": f"{arm}|{t}", "line": ln})
        gm = {}
        A = d["arms"]
        for a, b in (("M_bias_v1", "C0_refit"), ("B3_d3", "C0_refit"), ("M_bias_v1", "B3_d3")):
            if a in A and b in A:
                gm[f"{a}_vs_{b}"] = {s: paired_ratio(A[a]["per_device"][s], A[b]["per_device"][s])
                                     for s in m.STATES}
                gm[f"{a}_vs_{b}"]["stored"] = paired_ratio(A[a]["per_device_stored"], A[b]["per_device_stored"])
        d["gmr"] = gm
        out["datasets"][ds] = d
    out["validity"] = {"v1_core_max": v1_core, "v1_by_arm": v1_by_arm, "v2_n": v2_n, "v2_n_unequal": v2_bad,
                       "v2_max": 0.0 if v2_bad == 0 else 1.0, "v3_a2_rel_max": v3_rel,
                       "v3_unclamped_msgs": v3_unc_msgs, "v3_curve1_max": v3_c1, "other_clamps": other_clamps}
    out["runtime_s"] = round(time.time() - t0, 1)
    jdump(out, base_dir() / "audit" / "audit.json")
    print(json.dumps({"validity": {k: v for k, v in out["validity"].items() if k != "v1_by_arm"},
                      "v1_not_reproduced": {k: v for k, v in v1_by_arm.items() if not v <= 1e-12},
                      "means": {ds: {a: {s: round(x["aggregate"][s]["all_device_mean"], 5) for s in m.STATES}
                                     for a, x in out["datasets"][ds]["arms"].items()} for ds in m.DATASETS},
                      "runtime_s": out["runtime_s"]}, indent=1, default=float), flush=True)
    return out


# ---------------------------------------------------------------------------
# state-C starting records
# ---------------------------------------------------------------------------
def start_task(task):
    ds, tag = task
    dev, bi = devices(ds)[tag], bins(ds)[tag]
    old = stored_records(ds)[tag]
    params = dict(old.meta["params"])
    sims = sim_card(ds, tag, params, "C")
    rec = b8.to_record(dev, bi, sims, params)
    same = all(np.array_equal(np.asarray(c.sim), np.asarray(s), equal_nan=True) for c, s in zip(old.curves, sims))
    return ds, rec, {"tag": tag, "a2_eff": a2_eff(ds, tag, params), "identical_to_stored": bool(same),
                     "finite": bool(b8.finite_record(rec)), "finite_stored": bool(b8.finite_record(old))}


def stage_start(workers: int) -> dict:
    from cryoml.explore.e1_correction import save_devices, score_device
    tasks = [(ds, t) for ds in m.DATASETS for t in sorted(stored_records(ds))]
    if _SMOKE["on"]:
        tasks = [(ds, t) for ds in m.DATASETS for t in smoke_tags(ds, sorted(stored_records(ds)))]
    with get_context("spawn").Pool(workers) as pool:
        res = pool.map(start_task, tasks, chunksize=2)
    out = {}
    c_records_path("nist4").parent.mkdir(parents=True, exist_ok=True)
    for ds in m.DATASETS:
        rr = sorted([(r, row) for d_, r, row in res if d_ == ds], key=lambda x: x[0].tag)
        save_devices(c_records_path(ds), [r for r, _ in rr])
        rows = {row["tag"]: row for _, row in rr}
        for r, row in rr:
            row["rrms_C"] = float(score_device(r, [c.sim for c in r.curves])["rrms"])
            old = stored_records(ds)[r.tag]
            row["rrms_L"] = float(score_device(old, [c.sim for c in old.curves])["rrms"])
        unc = [x for x in rows.values() if m.clamp_class(x["a2_eff"]) == "none"]
        fam = {t: ("pmos" if is_pmos(ds, t) else "nmos") for t in rows}
        out[ds] = {"rows": rows, "n": len(rows), "n_clamped": len(rows) - len(unc),
                   "v8_unclamped_identical": bool(all(x["identical_to_stored"] for x in unc)), "v8_n": len(unc),
                   "n_finite_C": int(sum(x["finite"] for x in rows.values())),
                   "n_finite_L": int(sum(x["finite_stored"] for x in rows.values())),
                   "start_L": b8.aggregate({t: x["rrms_L"] for t, x in rows.items()}, fam),
                   "start_C": b8.aggregate({t: x["rrms_C"] for t, x in rows.items()}, fam),
                   "dev_check": dev_check(ds, rows)}
        print(f"{ds}: clamped {out[ds]['n_clamped']}/{len(rows)}; V8 {out[ds]['v8_unclamped_identical']}; finite "
              f"C {out[ds]['n_finite_C']} L {out[ds]['n_finite_L']}; start L "
              f"{out[ds]['start_L']['all_device_mean']:.5f} -> C {out[ds]['start_C']['all_device_mean']:.5f}",
              flush=True)
    jdump(out, base_dir() / "start" / "start.json")
    return out


# ---------------------------------------------------------------------------
# S3_D, v1 recipe
# ---------------------------------------------------------------------------
def s3v1_task(task):
    import torch
    torch.set_num_threads(1)
    from cryoml.explore.i3_2_bias import fit_bias
    from cryoml.explore.i7_1_transfer import s3_fold_value
    ds, k, state = task
    if state == "C":
        recs = c_records(ds)
        order = finite_order(ds)
    else:
        good, _ = b8.split_finite(list(stored_records(ds).values()))
        recs = {r.tag: r for r in good}
        order = sorted(recs)
    t0 = time.time()
    corr = fit_bias("M_bias", [recs[t] for t in order if t != k], hyper=S3_HYPER, threads=1)
    return ds, k, state, s3_fold_value(corr, [recs[k]])[0], round(time.time() - t0, 2)


def stage_s3v1(workers: int) -> dict:
    v2chk = {ds: sorted(jload(EXP / f"I19_1/s3/{ds}_v1.json")["per_record"])[:1] for ds in m.DATASETS}
    tasks = [(ds, k, "C") for ds in m.DATASETS for k in finite_order(ds)]
    if not _SMOKE["on"]:
        tasks += [(ds, k, "L") for ds in m.DATASETS for k in v2chk[ds]]
    with get_context("spawn").Pool(workers) as pool:
        res = pool.map(s3v1_task, tasks, chunksize=1)
    out = {}
    for ds in m.DATASETS:
        per = {k: v for d_, k, s, v, _ in res if d_ == ds and s == "C"}
        old = jload(EXP / f"I19_1/s3/{ds}_v1.json")
        chk = {k: {"new": v, "stored": old["per_record"][k], "diff": abs(v - old["per_record"][k])}
               for d_, k, s, v, _ in res if d_ == ds and s == "L"}
        order = finite_order(ds)
        s3 = m.s3_dataset([per[k] for k in order])
        out[ds] = {"recipe": "v1", "state": "C", "hyper": S3_HYPER, "S3_D": s3, "decision": m.decision_word(s3),
                   "S3_D_stored_L": old["S3_D"], "decision_stored": m.decision_word(old["S3_D"]),
                   "per_record": per, "n_records": len(order), "n_finite": int(np.sum(np.isfinite(list(per.values())))),
                   "state_L_check": chk, "dev_check": dev_check(ds, order),
                   "seconds": sum(x for d_, _, s, _, x in res if d_ == ds)}
        jdump(out[ds], base_dir() / "s3" / f"{ds}_v1.json")
        print(f"{ds}: S3_D v1 state C {s3:+.4f} (stored L {old['S3_D']:+.4f}); L check {chk}", flush=True)
    return out


# ---------------------------------------------------------------------------
# held-out refit (state C or L)
# ---------------------------------------------------------------------------
def refit_task(task):
    """(label, ds, tag, corr, start_params, state, opts) -> row."""
    from cryoml.explore.e1_correction import score_device
    from cryoml.ngspice_fit import polish
    label, ds, tag, corr, start, state, opts = task
    t0 = time.time()
    n_to0 = mod(ds).N_TIMEOUT
    prob = problem(ds, tag, state)
    box = prob.box
    res = polish(b8.corrected_problem(prob, corr), box.params_to_local(start), max_nfev=opts["heldout_nfev"],
                 irls_rounds=opts["heldout_irls"], diff_step=opts["diff_step"])
    params = box.local_to_params(res["local"])
    new = b8.to_record(prob.dev, prob.binfo, prob.simulate_params(params), params)
    after = score_device(new, corr.apply(new))
    return {"label": label, "ds": ds, "tag": tag, "state": state, "rrms": float(after["rrms"]),
            "failed": after["failed"], "n_sim": prob.n_sim, "params": params, "history": res["history"],
            "runtime_s": round(time.time() - t0, 2), "n_sim_timeout": mod(ds).N_TIMEOUT - n_to0}


def opts():
    return dict(HELDOUT, heldout_nfev=20, heldout_irls=1) if _SMOKE["on"] else dict(HELDOUT)


# ---------------------------------------------------------------------------
# base: C0_refit and M_bias v1 LODO in state C
# ---------------------------------------------------------------------------
V1C_N = 2


def stage_base(ds: str, workers: int) -> dict:
    from cryoml.explore.e1_correction import Correction
    from cryoml.explore.i3_2_bias import correction_from_json, merge_seed_results, split_seed_tasks
    t0 = time.time()
    rd = base_dir() / f"base_{ds}"
    order = finite_order(ds)
    recs = c_records(ds)
    hyper = {"seeds": (0,), "steps": 200} if _SMOKE["on"] else {}
    D = drv(ds)
    unit = run_unit_tests(rd)
    with get_context("spawn").Pool(workers, initializer=D._init_recs, initargs=(str(c_records_path(ds)),)) as pool:
        tasks = [("M_bias", h) for h in order]
        expanded, owner = split_seed_tasks(tasks, hyper)
        res = pool.map(D.train_task, [("bias", a, h, hp) for a, h, hp in expanded], chunksize=1)
        corrs = {h: c for h, c in zip(order, merge_seed_results(len(tasks), owner, res))}
        t_train = time.time() - t0
        rt = [("C0_refit", ds, h, Correction("C0"), dict(recs[h].meta["params"]), "C", opts()) for h in order]
        rt += [("M_bias", ds, h, corrs[h], dict(recs[h].meta["params"]), "C", opts()) for h in order]
        v1c = []
        if not _SMOKE["on"]:
            stored_c = jload(EXP / SRC[ds] / "lodo" / "corrections.json")
            clamped = [t for t in order if m.clamp_class(a2_eff(ds, t, recs[t].meta["params"])) != "none"]
            pick = clamped[:1] + [t for t in order if t not in clamped][:1]
            for t in pick:
                st = dict(stored_records(ds)[t].meta["params"])
                v1c.append(("V1c_C0_refit", ds, t, Correction("C0"), st, "L", dict(HELDOUT)))
                v1c.append(("V1c_M_bias", ds, t, correction_from_json(stored_c[f"M_bias|{t}"]), st, "L",
                            dict(HELDOUT)))
        t1 = time.time()
        rows = pool.map(refit_task, rt + v1c, chunksize=1)
        t_refit = time.time() - t1
    stored = jload(EXP / SRC[ds] / "lodo" / "results.json")["per_device"]
    per = {"C0_refit": {}, "M_bias": {}}
    v1 = {}
    for r in rows:
        if r["label"].startswith("V1c_"):
            arm = r["label"][4:]
            v1[f"{arm}|{r['tag']}"] = {"rrms": r["rrms"], "stored": stored[arm][r["tag"]],
                                       "diff": abs(r["rrms"] - stored[arm][r["tag"]])}
        else:
            per[r["label"]][r["tag"]] = r["rrms"]
    fam = {t: ("pmos" if is_pmos(ds, t) else "nmos") for t in order}
    out = {"ds": ds, "smoke": _SMOKE["on"], "opts": opts(), "hyper_overrides": hyper, "order": order,
           "per_device": per, "aggregate": {a: b8.aggregate(v, fam) for a, v in per.items()},
           "v1c": v1, "v1c_max": max([x["diff"] for x in v1.values()], default=float("nan")), "v1c_n": len(v1),
           "unit_tests_passed": bool(unit["passed"]), "dev_check": dev_check(ds, order),
           "rows": {f"{r['label']}|{r['tag']}": {k: v for k, v in r.items() if k != "params"} for r in rows},
           "timing": {"train_s": round(t_train, 1), "refit_s": round(t_refit, 1),
                      "runtime_s": round(time.time() - t0, 1)}}
    jdump(out, rd / "results.json")
    jdump({f"M_bias|{h}": c.to_json() for h, c in corrs.items()}, rd / "corrections.json")
    jdump({f"{r['label']}|{r['tag']}": r["params"] for r in rows}, rd / "heldout_cards.json")
    print(json.dumps({"v1c": v1, "timing": out["timing"],
                      "mean": {a: v["all_device_mean"] for a, v in out["aggregate"].items()}}, indent=1,
                     default=float), flush=True)
    return out


def run_unit_tests(rd: Path) -> dict:
    r = subprocess.run([sys.executable, "-m", "unittest", *UNIT_TESTS], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"})
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    out = {"files": list(UNIT_TESTS), "rc": r.returncode, "passed": r.returncode == 0, "tail": tail}
    jdump(out, rd / "unit_tests.json")
    print(f"unit tests: rc {r.returncode}; {tail[-1] if tail else ''}", flush=True)
    return out


# ---------------------------------------------------------------------------
# penalty grids
# ---------------------------------------------------------------------------
def grid_task(task):
    ds, tag, out_dir = task
    dev, bi = devices(ds)[tag], bins(ds)[tag]
    params = dict(c_records(ds)[tag].meta["params"])
    ch = b8.changed_params(params, bi["stock"])
    M = mod(ds)
    ch = (M.bsim4_clamp_params(dev.polarity, bi["bin"], ch)[0] if ds == "nist4"
          else M.bsim4_clamp_params(dev.polarity, bi["bin"], dev.W_um, dev.L_um, ch)[0])
    s = -1.0 if dev.polarity == "pmos" else 1.0
    n, step = m.PEN_N, m.PEN_STEP
    top = (n - 1) * step
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="i20_1_grid_") as td_s:
        td = Path(td_s)
        if ds == "nist4":
            lib = b8._write_lib(td, dev.polarity, bi["bin"], ch or None)
            head = f".options temp={b8.DECK_TEMP_K - 273.15:.4f}\n.lib \"{lib.as_posix()}\" tt_77k\n" + \
                b8.instance_line(dev.polarity, dev.W_um, dev.L_um)
        else:
            lib = b9.write_lib(td, dev.polarity, bi["bin"], ch or None)
            head = f".options temp={b9.DECK_TEMP_C:.4f}\n.lib \"{lib.as_posix()}\" tt_rt\n" + \
                b9.instance_line(dev.polarity, dev.W_um, dev.L_um)
        f = td / "grid.dat"
        deck = td / "g.sp"
        deck.write_text("* I20-1 penalty grid (state C)\n.options scale=1.0\n" + head
                        + "VG ng 0 DC 0\nVD nd 0 DC 0\n.control\nset filetype=ascii\n"
                        + f"dc VD 0 {s * top:.6g} {s * step:.6g} VG 0 {s * top:.6g} {s * step:.6g}\n"
                        + f"wrdata {f.as_posix()} -i(VD)\nquit\n.endc\n.end\n")
        subprocess.run([b8._sp()._NGSPICE_BIN, "-b", str(deck)], cwd=str(td), capture_output=True, text=True,
                       timeout=300, check=False)
        rows = []
        if f.exists():
            for line in f.read_text().splitlines():
                p = line.split()
                if len(p) >= 2:
                    try:
                        rows.append(float(p[1]))
                    except ValueError:
                        pass
    grid = np.asarray(rows, float).reshape(n, n) if len(rows) == n * n else np.full((n, n), np.nan)
    np.savez_compressed(Path(out_dir) / f"grid_{ds}_{tag}.npz", i=grid, axis=m.pen_axis())
    return {"ds": ds, "tag": tag, "n_rows": len(rows), "shape": list(grid.shape),
            "n_nonfinite": int(np.sum(~np.isfinite(grid))), "i_fwd_corner": float(s * grid[-1, -1]),
            "seconds": round(time.time() - t0, 2)}


def load_pens(ds: str, rd: Path | None = None) -> dict:
    k = ("pens", ds)
    if k in _C and rd is None:
        return _C[k]
    rd = rd or base_dir() / "pens"
    out = {}
    for tag in finite_order(ds):
        z = np.load(rd / f"grid_{ds}_{tag}.npz")
        out[tag] = v2.penalty_arrays(z["i"], is_pmos(ds, tag), step=m.PEN_STEP, grid=z["axis"])
    _C[k] = out
    return out


def stage_pens(workers: int) -> dict:
    rd = base_dir() / "pens"
    rd.mkdir(parents=True, exist_ok=True)
    tasks = [(ds, t, str(rd)) for ds in m.DATASETS for t in finite_order(ds)]
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(grid_task, tasks, chunksize=2)
    out = {"rows": {f"{r['ds']}|{r['tag']}": r for r in rows}, "n": len(rows)}
    out["v7_ok"] = bool(all(r["n_nonfinite"] == 0 and r["shape"] == [m.PEN_N, m.PEN_N] and r["i_fwd_corner"] > 0
                            for r in rows))
    out["card_negative_fraction"], out["kept_range"] = {}, {}
    for ds in m.DATASETS:
        pens = load_pens(ds, rd)
        for t, p in pens.items():
            out["rows"][f"{ds}|{t}"]["n_kept"] = p["n_kept"]
        out["card_negative_fraction"][ds] = v2.card_negative_fraction(list(pens.values()))
        kept = [p["n_kept"] for p in pens.values()]
        out["kept_range"][ds] = [min(kept), max(kept)]
    jdump(out, rd / "pens.json")
    print(json.dumps({"v7_ok": out["v7_ok"], "kept": out["kept_range"], "card_neg": out["card_negative_fraction"]},
                     indent=1), flush=True)
    return out


# ---------------------------------------------------------------------------
# g v2 training (cached per task)
# ---------------------------------------------------------------------------
def hyper_v2() -> dict:
    return {"steps": 100} if _SMOKE["on"] else {}


def cache_path(ds: str, name: str) -> Path:
    safe = name.replace("|", "__").replace("/", "_")
    return base_dir() / "cache" / ds / f"{safe}.json"


def train_v2_task(task):
    """(ds, name, train_tags, seed, lam, steps|None) -> (name, seed, corr json, seconds); cached."""
    import torch
    torch.set_num_threads(1)
    ds, name, train_tags, seed, lam, steps = task
    cp = cache_path(ds, f"{name}|s{seed}")
    if cp.exists():
        row = jload(cp)
        if row["train_tags"] == list(train_tags) and row["lam"] == float(lam) and row["steps"] == steps:
            return name, seed, row["corr"], row["seconds"]
    recs = c_records(ds)
    pens = None
    if lam:
        allp = load_pens(ds)
        pens = [allp[t] for t in train_tags]
    hp = {**hyper_v2(), "seeds": (seed,)}
    if steps:
        hp["steps"] = steps
    t0 = time.time()
    corr = v2.fit_v2([recs[t] for t in train_tags], pens, lam, hyper=hp, inputs="sym_v2", eps=m.EPS, threads=1,
                     arm=name.split("|", 1)[0], vdd=m.VDD)
    sec = round(time.time() - t0, 1)
    jdump({"name": name, "seed": seed, "lam": float(lam), "steps": steps, "train_tags": list(train_tags),
           "seconds": sec, "corr": corr.to_json()}, cp)
    return name, seed, corr.to_json(), sec


def refit_v2_task(task):
    label, ds, tag, cj, state = task
    corr = v2.SymCorrection.from_json(cj)
    return refit_task((label, ds, tag, corr, dict(c_records(ds)[tag].meta["params"]), state, opts()))


# ---------------------------------------------------------------------------
# selection (6 blocks + full data) and full-data model
# ---------------------------------------------------------------------------
def blocks(ds: str) -> list[list[str]]:
    order = finite_order(ds)
    return m.selection_blocks(order, {t: is_pmos(ds, t) for t in order})


def stage_select(ds: str, workers: int) -> dict:
    t0 = time.time()
    rd = base_dir() / f"select_{ds}"
    order = finite_order(ds)
    pm = {t: is_pmos(ds, t) for t in order}
    blks = blocks(ds)
    plans = {str(j): [t for t in order if t not in set(b)] for j, b in enumerate(blks)}
    plans["full"] = list(order)
    plan, tasks = {}, []
    for o, tr in plans.items():
        folds = m.inner_folds(tr, pm, m.N_INNER)
        plan[o] = {"train": tr, "inner_folds": folds}
        for i, held in enumerate(folds):
            itrain = [t for t in tr if t not in set(held)]
            for lam in m.GRID_LAMBDA:
                tasks.append((ds, f"sel|{o}|{i}|{lam:g}", itrain, 0, float(lam), None))
    with get_context("spawn").Pool(workers) as pool:
        trained = pool.map(train_v2_task, tasks, chunksize=1)
        t_train = time.time() - t0
        corr = {n: cj for n, _, cj, _ in trained}
        secs = {n: s for n, _, _, s in trained}
        print(f"select {ds}: {len(tasks)} trainings {t_train:.0f} s", flush=True)
        rtasks = []
        for o in plans:
            for i, held in enumerate(plan[o]["inner_folds"]):
                for lam in m.GRID_LAMBDA:
                    key = f"sel|{o}|{i}|{lam:g}"
                    rtasks += [(key, ds, t, corr[key], "C") for t in held]
        t1 = time.time()
        rows = pool.map(refit_v2_task, rtasks, chunksize=1)
        t_refit = time.time() - t1
    res = {"ds": ds, "smoke": _SMOKE["on"], "grid_lambda": list(m.GRID_LAMBDA), "blocks": blks, "plan": plan,
           "outer": {}, "train_seconds": secs, "opts": opts(), "hyper_overrides": hyper_v2(),
           "timing": {"train_s": round(t_train, 1), "refit_s": round(t_refit, 1), "n_train": len(tasks),
                      "n_refit": len(rtasks)}}
    complete = len(tasks) == len(plans) * m.N_INNER * len(m.GRID_LAMBDA)
    for o in plans:
        rr = {f"{lam:g}": {r["tag"]: r["rrms"] for r in rows if r["label"].startswith(f"sel|{o}|")
                           and r["label"].endswith(f"|{lam:g}")} for lam in m.GRID_LAMBDA}
        scores = {lam: m.inner_score(list(rr[f"{lam:g}"].values())) for lam in m.GRID_LAMBDA}
        n_exp = len(plans[o])
        res["outer"][o] = {"inner_rrms": rr, "scores": {f"{k:g}": v for k, v in scores.items()},
                           "n_scored": {k: len(v) for k, v in rr.items()}, "n_expected": n_exp,
                           "lambda_star": m.select_lambda(scores)}
        complete = complete and all(n == n_exp for n in res["outer"][o]["n_scored"].values())
    res["lambda_star_block"] = {o: res["outer"][o]["lambda_star"] for o in plans if o != "full"}
    res["lambda_star_full"] = res["outer"]["full"]["lambda_star"]
    res["lambda_by_device"] = m.lambda_by_device(blks, {int(k): v for k, v in res["lambda_star_block"].items()})
    res["complete"] = bool(complete)
    res["n_sim_refit"] = int(sum(r["n_sim"] for r in rows))
    # 5-seed full-data model at lambda*_full (TEST; not scored on DEV)
    lam = float(res["lambda_star_full"])
    seeds = (0,) if _SMOKE["on"] else tuple(v2.HYPER_V2["seeds"])
    name = f"full|{m.variant(lam)}"
    t2 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        full = pool.map(train_v2_task, [(ds, name, order, s, lam, None) for s in seeds], chunksize=1)
    merged = v2.merge_seed_corrections([v2.SymCorrection.from_json(cj) for _, _, cj, _ in sorted(full, key=lambda x: x[1])])
    res.update({"full_key": name, "full_train_s": round(time.time() - t2, 1),
                "full_seconds_per_member": [s for _, _, _, s in sorted(full, key=lambda x: x[1])]})
    res["runtime_s"] = round(time.time() - t0, 1)
    jdump(res, rd / "select.json")
    jdump({name: merged.to_json()}, rd / "full_corrections.json")
    print(json.dumps({"lambda_star_block": res["lambda_star_block"], "lambda_star_full": res["lambda_star_full"],
                      "complete": res["complete"], "timing": res["timing"]}, indent=1), flush=True)
    return res


def lambda_star(ds: str) -> dict:
    return {t: float(v) for t, v in jload(base_dir() / f"select_{ds}" / "select.json")["lambda_by_device"].items()}


# ---------------------------------------------------------------------------
# S3_D, g v2 recipe
# ---------------------------------------------------------------------------
def s3v2_task(task):
    from cryoml.explore.i7_1_transfer import s3_fold_value
    ds, k, lam = task
    order = finite_order(ds)
    steps = 50 if _SMOKE["on"] else S3_HYPER["steps"]
    name, seed, cj, sec = train_v2_task((ds, f"s3v2|{k}", [t for t in order if t != k], 0, lam, steps))
    corr = v2.SymCorrection.from_json(cj)
    return ds, k, s3_fold_value(corr, [c_records(ds)[k]])[0], sec


def stage_s3v2(workers: int) -> dict:
    tasks = []
    for ds in m.DATASETS:
        lam = lambda_star(ds)
        tasks += [(ds, k, lam[k]) for k in finite_order(ds)]
    with get_context("spawn").Pool(workers) as pool:
        res = pool.map(s3v2_task, tasks, chunksize=1)
    out = {}
    for ds in m.DATASETS:
        per = {k: v for d_, k, v, _ in res if d_ == ds}
        order = finite_order(ds)
        s3 = m.s3_dataset([per[k] for k in order])
        out[ds] = {"recipe": "g v2 (sym, EPS 0.01 V, VDD 1.8 V, lambda*_k)", "state": "C",
                   "steps": 50 if _SMOKE["on"] else S3_HYPER["steps"], "seed": 0, "S3_D": s3,
                   "decision": m.decision_word(s3), "per_record": per, "lambda_by_record": lambda_star(ds),
                   "n_records": len(order), "n_finite": int(np.sum(np.isfinite(list(per.values())))),
                   "dev_check": dev_check(ds, order), "seconds": sum(x for d_, _, _, x in res if d_ == ds)}
        jdump(out[ds], base_dir() / "s3" / f"{ds}_v2.json")
        print(f"{ds}: S3_D g v2 {s3:+.4f}", flush=True)
    return out


# ---------------------------------------------------------------------------
# selected g v2 LODO
# ---------------------------------------------------------------------------
def stage_lodo(ds: str, workers: int) -> dict:
    t0 = time.time()
    rd = base_dir() / f"lodo_{ds}"
    order = finite_order(ds)
    lam = lambda_star(ds)
    seeds = (0, 1) if _SMOKE["on"] else tuple(v2.HYPER_V2["seeds"])
    tasks = [(ds, f"sel|{h}|{m.variant(lam[h])}", [t for t in order if t != h], s, lam[h], None)
             for h in order for s in seeds]
    with get_context("spawn").Pool(workers) as pool:
        trained = []
        for i, r in enumerate(pool.imap_unordered(train_v2_task, tasks, chunksize=1)):
            trained.append(r)
            if (i + 1) % 25 == 0:
                print(f"  lodo {ds}: {i + 1}/{len(tasks)} members, {time.time() - t0:.0f} s", flush=True)
        t_train = time.time() - t0
        by = {}
        for name, seed, cj, sec in sorted(trained, key=lambda r: (r[0], r[1])):
            by.setdefault(name, []).append(v2.SymCorrection.from_json(cj))
        corrs = {n.split("|")[1]: v2.merge_seed_corrections(c) for n, c in by.items()}
        t1 = time.time()
        rows = pool.map(refit_v2_task, [("selected", ds, h, corrs[h].to_json(), "C") for h in order], chunksize=1)
        t_refit = time.time() - t1
    per = {r["tag"]: r["rrms"] for r in rows}
    fam = {t: ("pmos" if is_pmos(ds, t) else "nmos") for t in order}
    secs = {}
    for name, seed, _, sec in trained:
        secs.setdefault(name.split("|")[1], []).append(sec)
    res = {"ds": ds, "smoke": _SMOKE["on"], "inputs": "sym_v2", "eps": m.EPS, "vdd": m.VDD, "hyper": v2.HYPER_V2,
           "hyper_overrides": hyper_v2(), "seeds": list(seeds), "opts": opts(), "order": order,
           "lambda_by_device": lam, "per_device": per, "aggregate": b8.aggregate(per, fam),
           "heldout_rows": {r["tag"]: {k: v for k, v in r.items() if k != "params"} for r in rows},
           "train_seconds_per_member": secs, "dev_check": dev_check(ds, order),
           "timing": {"train_s": round(t_train, 1), "refit_s": round(t_refit, 1), "n_seed_tasks": len(tasks),
                      "runtime_s": round(time.time() - t0, 1)}}
    jdump(res, rd / "results.json")
    jdump({f"selected|{h}": c.to_json() for h, c in corrs.items()}, rd / "corrections.json")
    jdump({r["tag"]: r["params"] for r in rows}, rd / "heldout_cards.json")
    print(json.dumps({"ds": ds, "mean": res["aggregate"]["all_device_mean"], "timing": res["timing"]}, indent=1),
          flush=True)
    return res


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------
COMMITTED = {"nist4": -0.1026841666204108, "nistrt": -0.06888665622789505}


def neg_fractions(ds: str, corr_of, pens: dict) -> dict:
    tr_gm, tr_gds, ho_gm, ho_gds = [], [], [], []
    order = finite_order(ds)
    for h in order:
        c = corr_of(h)
        a = v2.penalty_value_numpy(c, [pens[t] for t in order if t != h])
        b = v2.penalty_value_numpy(c, [pens[h]])
        tr_gm.append(a["frac_neg_gm"]); tr_gds.append(a["frac_neg_gds"])
        ho_gm.append(b["frac_neg_gm"]); ho_gds.append(b["frac_neg_gds"])
    return {"train_mean_gm": float(np.mean(tr_gm)), "train_mean_gds": float(np.mean(tr_gds)),
            "heldout_mean_gm": float(np.mean(ho_gm)), "heldout_mean_gds": float(np.mean(ho_gds))}


def stage_decide() -> dict:
    from cryoml.explore.i3_2_bias import correction_from_json, paired_ratio
    B = base_dir()
    audit = jload(B / "audit" / "audit.json")
    start = jload(B / "start" / "start.json")
    pj = jload(B / "pens" / "pens.json")
    per_ds, v5, v6, v4 = {}, True, True, True
    for ds in m.DATASETS:
        order = finite_order(ds)
        base = jload(B / f"base_{ds}" / "results.json")
        sel = jload(B / f"select_{ds}" / "select.json")
        lodo = jload(B / f"lodo_{ds}" / "results.json")
        s3a = jload(B / "s3" / f"{ds}_v1.json")
        s3b = jload(B / "s3" / f"{ds}_v2.json")
        v4 = v4 and base["unit_tests_passed"]
        rows = {"C0_refit": base["per_device"]["C0_refit"], "M_bias_v1": base["per_device"]["M_bias"],
                "selected": lodo["per_device"]}
        v5 = v5 and all(len(rows[a]) == len(order) for a in rows) and sel["complete"] and \
            s3a["n_records"] == len(order) and s3b["n_records"] == len(order)
        v6 = v6 and all(x["dev_check"]["ok"] for x in (base, lodo, s3a, s3b)) and start[ds]["dev_check"]["ok"]
        n_failed = int(sum(not np.isfinite(rows[a][t]) for a in ("selected", "C0_refit") for t in order))
        primary = paired_ratio(rows["selected"], rows["C0_refit"])
        secondary = paired_ratio(rows["selected"], rows["M_bias_v1"])
        fam = {t: ("pmos" if is_pmos(ds, t) else "nmos") for t in order}
        table = {a: dict(b8.aggregate(r, fam)) for a, r in rows.items()}
        for a, r in rows.items():
            if a != "C0_refit":
                table[a]["gmr_vs_C0_refit"] = paired_ratio(r, rows["C0_refit"])
            if a != "M_bias_v1":
                table[a]["gmr_vs_v1"] = paired_ratio(r, rows["M_bias_v1"])
        per_family = {f: {a: {"aggregate": b8.aggregate({t: r[t] for t in order if fam[t] == f}, fam),
                              "gmr_vs_C0_refit": (paired_ratio({t: r[t] for t in order if fam[t] == f},
                                                               {t: rows["C0_refit"][t] for t in order if fam[t] == f})
                                                  if a != "C0_refit" else None)}
                          for a, r in rows.items()} for f in ("nmos", "pmos")}
        aud = audit["datasets"][ds]["arms"]
        stored_l = {a: aud[k]["per_device_stored"] for a, k in (("C0_refit", "C0_refit"), ("M_bias_v1", "M_bias_v1"))}
        g_v1_l = paired_ratio(stored_l["M_bias_v1"], stored_l["C0_refit"])
        pens = load_pens(ds)
        bc = jload(B / f"base_{ds}" / "corrections.json")
        lc = jload(B / f"lodo_{ds}" / "corrections.json")
        negs = {"M_bias_v1": neg_fractions(ds, lambda h: correction_from_json(bc[f"M_bias|{h}"]), pens),
                "selected": neg_fractions(ds, lambda h: v2.SymCorrection.from_json(lc[f"selected|{h}"]), pens),
                "card_only": pj["card_negative_fraction"][ds]}
        lam_counts = {}
        for h in order:
            k = f"{lambda_star(ds)[h]:g}"
            lam_counts[k] = lam_counts.get(k, 0) + 1
        realised = math_sign(primary["ratio"])
        per_ds[ds] = {
            "name": m.NAMES[ds], "n_finite_start": len(order), "rows": rows, "table": table, "per_family": per_family,
            "primary_gmr_selected_vs_C0_refit": primary, "secondary_gmr_selected_vs_v1": secondary,
            "gmr_v1_vs_C0_refit_stateC": table["M_bias_v1"]["gmr_vs_C0_refit"], "gmr_v1_vs_C0_refit_stored_L": g_v1_l,
            "n_failed_selected_C0": n_failed, "outcome": m.primary_label(primary, n_failed),
            "secondary": m.secondary_label(secondary, m.NI_MARGIN),
            "lambda_star_block": sel["lambda_star_block"], "lambda_star_full": sel["lambda_star_full"],
            "lambda_counts_devices": lam_counts, "negative_fractions": negs,
            "S3_D": {"committed_a755e18_v1_L": COMMITTED[ds], "v1_stateC": s3a["S3_D"], "g_v2": s3b["S3_D"]},
            "decisions": {"committed": m.decision_word(COMMITTED[ds]), "v1_stateC": s3a["decision"],
                          "g_v2": s3b["decision"], "frozen_rule": m.frozen_rule_decision(s3a["S3_D"])},
            "sign_check_vs_realised_g_v2": {"realised": realised,
                                            "v1_stateC": (s3a["S3_D"] < 0) == (realised == "gain"),
                                            "g_v2": (s3b["S3_D"] < 0) == (realised == "gain")},
            "timing": {"base": base["timing"], "select": sel["timing"], "select_runtime_s": sel["runtime_s"],
                       "lodo": lodo["timing"], "s3_v1_s": s3a["seconds"], "s3_v2_s": s3b["seconds"]},
            "n_sim": {"base_refits": int(sum(r["n_sim"] for r in base["rows"].values())),
                      "select_refits": sel["n_sim_refit"],
                      "lodo_refits": int(sum(r["n_sim"] for r in lodo["heldout_rows"].values()))}}
    av = audit["validity"]
    validity = m.decide_validity({"v1_max": av["v1_core_max"], "v2_max": av["v2_max"], "v2_n": av["v2_n"],
                                  "v3_a2_rel_max": av["v3_a2_rel_max"], "v3_unclamped_msgs": av["v3_unclamped_msgs"],
                                  "v3_curve1_max": av["v3_curve1_max"], "v4_unit_tests_pass": v4,
                                  "v5_complete": v5, "v6_test_closed": v6})
    extra = {"v7_grids_ok": bool(pj["v7_ok"]),
             "v8_unclamped_start_identical": all(start[ds]["v8_unclamped_identical"] for ds in m.DATASETS)}
    if not (extra["v7_grids_ok"] and extra["v8_unclamped_start_identical"]):
        validity = {"valid": False, "fails": validity["fails"] + [k for k, v in extra.items() if not v]}
    if not validity["valid"]:
        for d in per_ds.values():
            d["outcome"], d["secondary"] = "invalid", "invalid"
    amend = m.amendment_needed(COMMITTED, {ds: {"v1_stateC": per_ds[ds]["S3_D"]["v1_stateC"],
                                                "g_v2": per_ds[ds]["S3_D"]["g_v2"]} for ds in m.DATASETS})
    out = {"prereg": str(PREREG.relative_to(ROOT)), "validity": validity, "validity_extra": extra,
           "per_dataset": per_ds, "amendment": amend,
           "frozen_rule_check": {"v1_stateC_correct": int(sum(d["sign_check_vs_realised_g_v2"]["v1_stateC"]
                                                             for d in per_ds.values())),
                                 "g_v2_correct": int(sum(d["sign_check_vs_realised_g_v2"]["g_v2"]
                                                         for d in per_ds.values())), "n": len(per_ds)},
           "audit_summary": {ds: {a: {s: x["aggregate"][s]["all_device_mean"] for s in m.STATES}
                                  for a, x in audit["datasets"][ds]["arms"].items()} for ds in m.DATASETS}}
    jdump(out, B / "decision.json")
    print(json.dumps({"validity": validity, "amendment": amend,
                      "per_ds": {ds: {"outcome": d["outcome"], "secondary": d["secondary"],
                                      "primary": {k: d["primary_gmr_selected_vs_C0_refit"][k]
                                                  for k in ("ratio", "ci_low", "ci_high", "wins", "n_used")},
                                      "secondary_gmr": {k: d["secondary_gmr_selected_vs_v1"][k]
                                                        for k in ("ratio", "ci_low", "ci_high", "wins")},
                                      "S3_D": d["S3_D"], "decisions": d["decisions"],
                                      "means": {a: (round(v["all_device_mean"], 5), round(v["combined"], 5))
                                                for a, v in d["table"].items()}}
                               for ds, d in per_ds.items()}}, indent=1, default=float), flush=True)
    return out


def math_sign(r: float) -> str:
    return "gain" if (r is not None and np.isfinite(r) and r < 1.0) else "no gain"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("audit", "start", "s3v1", "base", "pens", "select", "s3v2", "lodo", "decide"))
    ap.add_argument("--ds", choices=m.DATASETS, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="reduced settings, out/research/explore/I20_1/smoke")
    args = ap.parse_args()
    if args.smoke:
        os.environ["CRYOML_I20_1_SMOKE"] = "1"
        _SMOKE["on"] = True
    t0 = time.time()
    st = args.stage
    if st in ("base", "select", "lodo") and args.ds is None:
        raise SystemExit("--ds required")
    {"audit": lambda: stage_audit(args.workers), "start": lambda: stage_start(args.workers),
     "s3v1": lambda: stage_s3v1(args.workers), "base": lambda: stage_base(args.ds, args.workers),
     "pens": lambda: stage_pens(args.workers), "select": lambda: stage_select(args.ds, args.workers),
     "s3v2": lambda: stage_s3v2(args.workers), "lodo": lambda: stage_lodo(args.ds, args.workers),
     "decide": stage_decide}[st]()
    print(f"stage {st} done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
