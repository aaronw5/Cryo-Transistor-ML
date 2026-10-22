#!/usr/bin/env python3
"""I16-1: deployment test (Part B, I15-1 rules) of the I15-2 selected
bias-only correction (symmetric g, nested-selected penalty lambda; deployed
full-data model lambda* = 1 V, ``full|sym_e1_l1``).  sky130 77 K, 18 devices,
I13-1 state-C records.

Pre-registration: out/research/explore/I16_1/prereg.md (prereg.json).

The stages reuse the I14-1 / I15-1 drivers (scripts/research/explore/
i14_1_chan.py, i15_1_chan.py; serpentine Q1 grids) with the output directory
set to I16_1 and an arm "sel" whose full-data g is ``full|sym_e1_l1`` and
whose held-out pairs are the I15-2 selected ones (fold g and held-out card of
sym_e1_l{lambda*_h}).  The rebinding happens at import, so spawned workers
see it too.

Stages (``--stage``):

  validity  V1 card-only library deck x exp(fold g) reproduces the I15-2
            selected rows; V2 unit tests; V4 deployed g
  accuracy  element-deck RRMS of the 18 selected held-out pairs
  deploy    compile, M2 refit, q12 (serpentine), q3, report
  restrict  restriction items per device (Q1 default / r4 / tight; gds and
            gm sign-change counts for element, card, I_card exp(g); the same
            counts for the I15-1 g v2 arm from its stored arrays)
  runtime   ro5_A decomposition on the I16-1 M2 cards
  decide    validity + Part B -> decision.json

  PYTHONPATH=src .venv/bin/python scripts/research/explore/i16_1_sel.py --stage validity --workers 6
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
sys.path.insert(0, str(ROOT / "scripts" / "research" / "explore"))

import i15_1_chan as drv15  # noqa: E402  (imports i14_1_chan; sets the I13-1 state-C environment)
from cryoml.explore import i11_1_deploy as m1  # noqa: E402
from cryoml.explore import i12_1_symm as v2  # noqa: E402
from cryoml.explore import i16_1_sel as m  # noqa: E402

drv14 = drv15.drv14
EXP = drv14.EXP
OUT = EXP / "I16_1"
PREREG = OUT / "prereg.json"
I15_1 = EXP / "I15_1"
I15_2 = EXP / "I15_2"
ARM = "sel"

jload, jdump, devices = drv14.jload, drv14.jdump, drv14.devices
SMOKE_TAGS = drv14.SMOKE_TAGS
TOLS = drv15.TOLS


# ---------------------------------------------------------------------------
# arm "sel": corrections, held-out cards, rows
# ---------------------------------------------------------------------------
def lambda_star() -> dict:
    return {k: float(v) for k, v in jload(I15_2 / "decision.json")["lambda_star"].items()}


def _corr_json(directory: str) -> dict:
    return jload(EXP / directory / "corrections.json")


_orig = {"corr_full": drv14.corr_full, "corr_fold": drv14.corr_fold,
         "heldout_card": drv14.heldout_card, "python_rows": drv14.python_rows}


def corr_full(arm: str):
    if arm != ARM:
        return _orig["corr_full"](arm)
    return v2.SymCorrection.from_json(_corr_json(m.DEPLOYED["dir"])[m.DEPLOYED["key"]])


def corr_fold(arm: str, tag: str):
    if arm != ARM:
        return _orig["corr_fold"](arm, tag)
    lam = lambda_star()[tag]
    return v2.SymCorrection.from_json(_corr_json(m.source_for(lam)[0])[m.fold_key(lam, tag)])


def heldout_card(arm: str, tag: str) -> dict:
    if arm != ARM:
        return _orig["heldout_card"](arm, tag)
    return dict(jload(EXP / m.source_for(lambda_star()[tag])[0] / "heldout_cards.json")[tag])


def python_rows(arm: str) -> dict:
    if arm != ARM:
        return _orig["python_rows"](arm)
    return dict(jload(I15_2 / "decision.json")["selected_rows"])


def m2_task(task):
    tag, opts, arm = task
    return drv14.drv12.heldout(("M2", None, corr_full(arm), drv14.load_records()[tag], opts))


# rebinding: every I14-1 / I15-1 stage writes under I16_1 and knows the arm "sel"
drv14.OUT = OUT
drv14.PREREG = PREREG
drv14.UNIT_TESTS = drv15.drv14.UNIT_TESTS + ("tests.test_i15_2_ablate", "tests.test_i16_1_sel")
drv14.corr_full, drv14.corr_fold, drv14.heldout_card, drv14.python_rows = (
    corr_full, corr_fold, heldout_card, python_rows)
drv14.m2_task = m2_task
drv15.OUT = OUT


# ---------------------------------------------------------------------------
# validity
# ---------------------------------------------------------------------------
def deployed_ok() -> dict:
    res = jload(EXP / m.DEPLOYED["dir"] / "results.json")
    c = corr_full(ARM)
    full = jload(I15_2 / "decision.json")["lambda_star_full"]
    ok = (res["inputs"] == m.DEPLOYED["inputs"] and abs(float(res["eps"]) - m.DEPLOYED["eps"]) < 1e-15
          and abs(float(res["lambda"]) - m.DEPLOYED["lambda"]) < 1e-15 and c.inputs == m.DEPLOYED["inputs"]
          and abs(c.eps - m.DEPLOYED["eps"]) < 1e-15 and abs(float(full) - 1.0) < 1e-15)
    return {"ok": bool(ok), "results_inputs": res["inputs"], "results_eps": res["eps"],
            "results_lambda": res["lambda"], "corr_inputs": c.inputs, "corr_eps": c.eps,
            "lambda_star_full": full}


def stage_validity(workers: int, rd: Path, tags) -> dict:
    unit = drv14.run_unit_tests(rd)
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(drv14.validity_task, [(ARM, t) for t in tags], chunksize=1)
    stored = python_rows(ARM)
    lam = lambda_star()
    out = {"rows": {}, "unit_tests_passed": bool(unit["passed"]), "v4": deployed_ok()}
    worst = 0.0
    for r in rows:
        d = abs(r["rrms"] - stored[r["tag"]])
        worst = max(worst, d)
        out["rows"][r["tag"]] = {**r, "lambda_star": lam[r["tag"]], "stored": stored[r["tag"]], "abs_diff": d}
    out["v1_max_rrms_diff"] = worst
    out["v1_n_rows"] = len(rows)
    out["wall_s"] = round(time.time() - t0, 1)
    jdump(out, rd / "validity.json")
    print(f"validity: max |RRMS card-only deck - I15-2 selected row| {worst:.3e} ({len(rows)} rows); "
          f"unit tests {unit['passed']}; V4 {out['v4']['ok']}", flush=True)
    return out


# ---------------------------------------------------------------------------
# accuracy (element deck, selected held-out pairs)
# ---------------------------------------------------------------------------
def stage_accuracy(workers: int, rd: Path, tags) -> dict:
    from cryoml.explore.i3_2_bias import paired_ratio
    from e1_lodo import aggregate
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        comp = pool.map(drv14.compile_task, [(str(rd), ARM, t) for t in tags], chunksize=1)
        t_comp = time.time() - t0
        rows = pool.map(drv14.accuracy_task, [(str(rd), ARM, t) for t in tags], chunksize=1)
    stored = python_rows(ARM)
    lam = lambda_star()
    out = {"compile": comp, "compile_s": round(t_comp, 1), "rows": {}}
    for r in rows:
        t = r["tag"]
        r["lambda_star"] = lam[t]
        r["stored_python"] = stored[t]
        r["native_minus_stored"] = r["rrms_native"] - stored[t]
        out["rows"][t] = r
    nat = {t: r["rrms_native"] for t, r in out["rows"].items()}
    out["native"] = nat
    out["python_rerun"] = {t: r["rrms_python"] for t, r in out["rows"].items()}
    out["python_channel"] = {t: r["rrms_python_channel"] for t, r in out["rows"].items()}
    out["max_abs_native_minus_stored"] = max(abs(r["native_minus_stored"]) for r in out["rows"].values())
    out["max_abs_python_rerun_minus_stored"] = max(abs(r["rrms_python"] - r["stored_python"])
                                                   for r in out["rows"].values())
    out["max_native_vs_channel_ratio"] = max(r["native_vs_channel_max_ratio_1e-3_1e-12"]
                                             for r in out["rows"].values())
    out["n_msgs"] = sum(len(r["msgs"]) for r in out["rows"].values())
    if len(rows) == 18:
        lodo = jload(drv14.LODO / "results.json")["per_device"]
        v1s = dict(jload(drv14.ALL18 / "results.json")["per_device"]["M_bias"])
        v2_el = jload(I15_1 / "accuracy" / "accuracy.json")["per_arm"]["v2"]["native"]
        v1_el = jload(I15_1 / "accuracy" / "accuracy.json")["per_arm"]["v1"]["native"]
        out["summary_native"] = aggregate(nat)
        out["summary_stored"] = aggregate({t: stored[t] for t in nat})
        out["summary_reference"] = {"v1_stored": aggregate(v1s), "C0_refit": aggregate(dict(lodo["C0_refit"])),
                                    "v2_element_I15_1": aggregate(v2_el), "v1_element_I15_1": aggregate(v1_el)}
        out["gmr_native"] = {"sel_native_vs_v1_stored": paired_ratio(nat, v1s),
                             "sel_native_vs_C0_refit": paired_ratio(nat, dict(lodo["C0_refit"])),
                             "sel_native_vs_v2_element_I15_1": paired_ratio(nat, v2_el),
                             "sel_native_vs_v1_element_I15_1": paired_ratio(nat, v1_el)}
    out["wall_s"] = round(time.time() - t0, 1)
    jdump(out, rd / "accuracy.json")
    print(json.dumps({k: out[k] for k in ("max_abs_native_minus_stored", "max_abs_python_rerun_minus_stored",
                                          "max_native_vs_channel_ratio", "n_msgs")}, indent=1), flush=True)
    if "summary_native" in out:
        print(json.dumps({"native": out["summary_native"], "stored": out["summary_stored"],
                          "gmr": {k: [round(v["ratio"], 4), round(v["ci_low"], 4), round(v["ci_high"], 4)]
                                  for k, v in out["gmr_native"].items()}}, indent=1), flush=True)
    return out


# ---------------------------------------------------------------------------
# restriction items (per device)
# ---------------------------------------------------------------------------
def sign_items(rd: Path, corr, q12: dict, label: str) -> dict:
    """Per device: gds curves (5) and gm curves (3) from the dense 1 mV
    sweeps: introduced flags (q12) and negative-derivative counts for the
    element, the card and I_card exp(g) of the card sweep."""
    out = {}
    for tag, r in sorted(q12.items()):
        dev = devices()[tag]
        pm = dev.dev_type == "pmos"
        s = -1.0 if pm else 1.0
        z = np.load(rd / "arrays" / f"q12_{tag}.npz")
        vt = float(r["vt_card"])
        gds, gm = [], []
        for k, vgs in enumerate(m1.GDS_VGS):
            dc, dk = z[f"dense_card_gds{k}"], z[f"dense_elem_gds{k}"]
            o = np.argsort(np.abs(dc[:, 0]))
            x, ic, ik = dc[o, 0], dc[o, 1], dk[o, 1]
            g = corr.g_v(pm, np.full_like(x, s * vgs), x)
            ipy = ic * np.exp(g)
            sel = np.abs(x) >= 0.05
            gds.append({"vgs": float(s * vgs), "introduced": bool(r["q2_gds_curves_introduced"][k]),
                        "n_neg_elem": m.neg_count(x, ik, sel), "n_neg_card": m.neg_count(x, ic, sel),
                        "n_neg_python": m.neg_count(x, ipy, sel), "n_points": int(np.sum(sel)),
                        "vds_abs_range_neg_elem": m.neg_range(x, ik, sel),
                        "g_range": [float(np.min(g)), float(np.max(g))]})
        for k, vds in enumerate(m1.GM_VDS):
            dc, dk = z[f"dense_card_gm{k}"], z[f"dense_elem_gm{k}"]
            o = np.argsort(np.abs(dc[:, 0]))
            x, ic, ik = dc[o, 0], dc[o, 1], dk[o, 1]
            g = corr.g_v(pm, x, np.full_like(x, s * vds))
            ipy = ic * np.exp(g)
            si = np.abs(x) >= vt + 0.2
            al = np.ones_like(x, bool)
            gm.append({"vds": float(s * vds), "introduced": bool(r["q2_gm_curves_introduced"][k]),
                       "n_neg_elem_si": m.neg_count(x, ik, si), "n_neg_card_si": m.neg_count(x, ic, si),
                       "n_neg_python_si": m.neg_count(x, ipy, si),
                       "n_neg_elem_all": m.neg_count(x, ik, al), "n_neg_card_all": m.neg_count(x, ic, al),
                       "n_neg_python_all": m.neg_count(x, ipy, al), "n_points_si": int(np.sum(si))})
        out[tag] = {"vt_card": vt, "gds": gds, "gm": gm,
                    "n_gds_introduced": sum(c["introduced"] for c in gds),
                    "n_gm_introduced": sum(c["introduced"] for c in gm)}
    tot = {"label": label, "n_gds_curves_introduced": sum(v["n_gds_introduced"] for v in out.values()),
           "n_gm_curves_introduced": sum(v["n_gm_introduced"] for v in out.values()),
           "n_gds_neg_points_elem": sum(c["n_neg_elem"] for v in out.values() for c in v["gds"]),
           "n_gds_neg_points_python": sum(c["n_neg_python"] for v in out.values() for c in v["gds"]),
           "n_gds_neg_points_card": sum(c["n_neg_card"] for v in out.values() for c in v["gds"]),
           "n_gm_neg_points_elem_si": sum(c["n_neg_elem_si"] for v in out.values() for c in v["gm"]),
           "n_gm_neg_points_card_si": sum(c["n_neg_card_si"] for v in out.values() for c in v["gm"]),
           "n_gm_neg_points_python_si": sum(c["n_neg_python_si"] for v in out.values() for c in v["gm"]),
           "devices_with_gds_introduced": sorted(t for t, v in out.items() if v["n_gds_introduced"]),
           "devices_with_gm_introduced": sorted(t for t, v in out.items() if v["n_gm_introduced"])}
    return {"per_device": out, "totals": tot}


def stage_restrict(rd: Path, smoke: bool) -> dict:
    q12 = jload(rd / "q12.json")["devices"]
    out = {"q1_max_rel_1nA": {t: {tol: r[f"q1_{tol}_max_rel_1nA"] for tol, _ in TOLS}
                              for t, r in sorted(q12.items())},
           "sel": sign_items(rd, corr_full(ARM), q12, "I16-1 selected (full|sym_e1_l1)")}
    ref_rd = I15_1 / "deploy_v2"
    if (ref_rd / "q12.json").exists() and (ref_rd / "arrays").exists():
        q12r = jload(ref_rd / "q12.json")["devices"]
        if smoke:
            q12r = {t: v for t, v in q12r.items() if t in q12}
        out["ref_v2_I15_1"] = sign_items(ref_rd, _orig["corr_full"]("v2"), q12r,
                                         "I15-1 g v2 (full|M_bias_v2, lambda 10 V), stored arrays")
    jdump(out, rd / "restrictions.json")
    print(json.dumps({"q1_default_max": max(v["default"] for v in out["q1_max_rel_1nA"].values()),
                      "sel": out["sel"]["totals"],
                      "ref_v2": out.get("ref_v2_I15_1", {}).get("totals")}, indent=1), flush=True)
    return out


# ---------------------------------------------------------------------------
# report (I14-1 summary with the I16-1 label) and decide
# ---------------------------------------------------------------------------
def stage_report(rd: Path) -> dict:
    out = drv14.stage_report(rd, ARM)
    out["label"] = "I16-1 Part B (I15-2 selected correction, full|sym_e1_l1, lambda 1 V)"
    jdump(out, rd / "summary.json")
    return out


def stage_decide() -> dict:
    val = jload(OUT / "validity" / "validity.json")
    acc = jload(OUT / "accuracy" / "accuracy.json")
    dep = jload(OUT / "deploy_sel" / "summary.json")
    rest = jload(OUT / "deploy_sel" / "restrictions.json")
    validity = m.decide_validity({"v1_max_rrms_diff": val["v1_max_rrms_diff"], "v1_n_rows": val["v1_n_rows"],
                                  "v2_unit_tests_pass": bool(val["unit_tests_passed"]),
                                  "v3_n_rows": len(acc["native"]),
                                  "v3_m2_failed": dep["refit_in_sample"]["n_failed"],
                                  "v4_deployed_ok": bool(val["v4"]["ok"])})
    i15 = {a: jload(I15_1 / f"deploy_{a}" / "summary.json") for a in ("v2", "v1")}
    out = {"prereg": str(PREREG.relative_to(ROOT)), "validity": validity, "deploy_sel": dep["decision"],
           **m.decide_outcome(validity, dep["decision"]),
           "values": dep["values"], "thresholds": v2.THR,
           "runtime_grid": dep["runtime_grid"], "runtime_circuits": dep["runtime_circuits"],
           "refit_in_sample": dep["refit_in_sample"],
           "restriction_totals": {"sel": rest["sel"]["totals"],
                                  "ref_v2_I15_1": rest.get("ref_v2_I15_1", {}).get("totals")},
           "accuracy_I15_2_fixed": jload(I15_2 / "decision.json")["primary_gmr_selected_vs_v1"],
           "element_deck_accuracy": {k: acc.get(k) for k in (
               "summary_native", "summary_stored", "summary_reference", "gmr_native",
               "max_abs_native_minus_stored", "max_abs_python_rerun_minus_stored",
               "max_native_vs_channel_ratio", "n_msgs")},
           "reference_I15_1": {a: {"decision": i15[a]["decision"], "values": i15[a]["values"]} for a in i15}}
    jdump(out, OUT / "decision.json")
    print(json.dumps({k: out[k] for k in ("validity", "deploy_sel", "outcome", "action", "values",
                                          "restriction_totals")}, indent=1, default=float), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("validity", "accuracy", "deploy", "compile", "refit", "q12", "q3", "report",
                             "restrict", "runtime", "decide"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true",
                    help="2 devices, refit max_nfev 20 (code check; not used for decisions)")
    args = ap.parse_args()
    base = OUT / "smoke" if args.smoke else OUT
    tags = SMOKE_TAGS if args.smoke else sorted(devices())
    rd = base / "deploy_sel"
    t0 = time.time()
    if args.stage == "validity":
        stage_validity(args.workers, base / "validity", tags)
    elif args.stage == "accuracy":
        stage_accuracy(args.workers, base / "accuracy", tags)
    elif args.stage == "decide":
        stage_decide()
    elif args.stage == "restrict":
        stage_restrict(rd, args.smoke)
    elif args.stage == "runtime":
        drv15.stage_runtime(rd, ARM)
    else:
        opts = dict(drv14.REFIT_OPTS, heldout_nfev=20) if args.smoke else dict(drv14.REFIT_OPTS)
        workers = min(args.workers, 4)
        stages = (("compile", "refit", "q12", "q3", "report") if args.stage == "deploy" else (args.stage,))
        timing = {}
        for st in stages:
            t1 = time.time()
            if st == "compile":
                drv14.stage_compile(rd, ARM)
            elif st == "refit":
                drv14.stage_refit(rd, ARM, tags, workers, opts)
            elif st == "q12":
                drv15.stage_q12(rd, ARM, tags, workers)
            elif st == "q3":
                if args.smoke:      # smoke: starting cards for circuit devices not refit
                    have = jload(rd / "cards_M2.json")
                    sc = drv14.start_cards()
                    for _, _, ts, _ in drv14.CIRCUITS:
                        for t in ts:
                            have.setdefault(t, sc[t])
                    jdump(have, rd / "cards_M2.json")
                drv14.stage_q3(rd, ARM, workers)
            else:
                stage_report(rd)
            timing[st] = round(time.time() - t1, 1)
        tf = rd / "timing.json"
        old = jload(tf) if tf.exists() else {}
        old.update(timing)
        jdump(old, tf)
    print(f"stage {args.stage} done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
