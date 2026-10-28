#!/usr/bin/env python3
"""I25-1 element checks (prereg section 10, both arms): the I17-1 Part B test
(Verilog-A / OSDI element, channel-current replica, serpentine Q1 grids, Q2,
Q3; V(d,s) = 0 check at tight tolerance only) on the I25-1 full-data card + g
pairs.

Pre-registration: out/research/explore/I25_1/element/prereg.md.

The stages reuse the I14-1 / I15-1 / I16-1 drivers with the inputs rebound
at import (template ``mult1``; arm from ``CRYOML_I25E_ARM``, so spawned
workers see the same rebinding):

  corrections   I25_1/arm<X>/full/corrections.json ``full`` (deployed g) and
                lodo/corrections.json ``lodo|<tag>`` (fold g)
  M2 cards      I25_1/arm<X>/full/results.json rows (in-sample refits with
                the full-data g in the loop; not refit again)
  LODO cards    I25_1/arm<X>/lodo/results.json rows
  C0_refit      I25_1/arm<X>/c0/results.json rows (Q3 reference arm)
  records       I25_1/arm<X>/records/all18 (measured curves, frozen curve set)

Stages (``--stage``, ``--arm A|B``):

  unit      V2 unit tests (default template environment)
  pairs     compile the full and fold OSDI modules; per (full | lodo, tag):
            card-only library deck x exp(g) (V1), card + replica, element
            deck (accuracy) -> pairs/pairs.json
  deploy    M2 rows from the I25-1 full rows + pairs, compile, q12, q3,
            report -> deploy/
  restrict  restriction items per device -> deploy/restrictions.json
  runtime   ro5_A decomposition -> deploy/runtime.json
  decide    validity + Part B -> decision.json (per arm)

  PYTHONPATH=src .venv/bin/python scripts/research/explore/i25_1_element.py --stage pairs --arm B --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

_ap = argparse.ArgumentParser()
_ap.add_argument("--stage", required=True,
                 choices=("unit", "pairs", "deploy", "compile", "m2", "q12", "q3", "report", "restrict",
                          "runtime", "decide"))
_ap.add_argument("--arm", choices=("A", "B"), required=True)
_ap.add_argument("--workers", type=int, default=4)
_ap.add_argument("--smoke", action="store_true", help="2 devices (code check; not used for decisions)")
ARGS = _ap.parse_args() if __name__ == "__main__" else None
os.environ["CRYOML_PDK_TEMPLATE"] = "mult1"          # the I25-1 template, before any cryoml import
if ARGS is not None:
    os.environ["CRYOML_I25E_ARM"] = ARGS.arm
    os.environ["CRYOML_I25E_SMOKE"] = "1" if ARGS.smoke else "0"
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"
ARM = os.environ.get("CRYOML_I25E_ARM", "B")
SMOKE = os.environ.get("CRYOML_I25E_SMOKE") == "1"

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(ROOT / "scripts" / "research" / "explore"))

import i16_1_sel as drv16  # noqa: E402  (imports the I15-1 / I14-1 drivers; rebound below)
from cryoml import spice_pdk as sp  # noqa: E402
from cryoml.explore import i11_1_deploy as m1  # noqa: E402
from cryoml.explore import i12_1_symm as v2  # noqa: E402
from cryoml.explore import i14_1_chan as c14  # noqa: E402
from cryoml.explore import i17_1_vds0 as m17  # noqa: E402
from cryoml.explore import i25_1_scratch as m25  # noqa: E402

if sp.PDK_TEMPLATE != "mult1":
    raise RuntimeError(f"template {sp.PDK_TEMPLATE}, expected mult1")

drv15, drv14 = drv16.drv15, drv16.drv14
EXP = ROOT / "out/research/explore"
I25 = EXP / "I25_1"
AD = I25 / f"arm{ARM}"
OUT = I25 / "element"
PREREG = OUT / "prereg.md"
BASE = OUT / ("smoke" if SMOKE else "") / f"arm{ARM}"
LAMBDA_EXPECTED = 5.0
UNIT_TESTS = tuple(drv14.UNIT_TESTS) + ("tests.test_i15_1_serp", "tests.test_i15_2_ablate",
                                        "tests.test_i16_1_sel", "tests.test_i17_1_vds0",
                                        "tests.test_i25_1_scratch")
UNIT_TESTS = tuple(dict.fromkeys(UNIT_TESTS))
TOLS = drv15.TOLS
jload, jdump, devices = drv14.jload, drv14.jdump, drv14.devices


# ---------------------------------------------------------------------------
# I25-1 inputs (arm ARM)
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def _j(path: Path):
    k = str(path)
    if k not in _CACHE:
        _CACHE[k] = jload(path)
    return _CACHE[k]


def arm_dir(arm: str) -> Path:
    return I25 / f"arm{arm}"


def corr_full(arm: str):
    return v2.SymCorrection.from_json(_j(arm_dir(arm) / "full" / "corrections.json")["full"])


def corr_fold(arm: str, tag: str):
    return v2.SymCorrection.from_json(_j(arm_dir(arm) / "lodo" / "corrections.json")[f"lodo|{tag}"])


def heldout_card(arm: str, tag: str) -> dict:
    return dict(_j(arm_dir(arm) / "lodo" / "results.json")["rows"][tag]["params"])


def m2_card(arm: str, tag: str) -> dict:
    return dict(_j(arm_dir(arm) / "full" / "results.json")["rows"][tag]["params"])


def c0_card(arm: str, tag: str) -> dict:
    return dict(_j(arm_dir(arm) / "c0" / "results.json")["rows"][tag]["params"])


def stored_rows(arm: str, kind: str) -> dict:
    """Stored I25-1 rows (primary frozen curve set; final.json, V4 0 vs the run rows)."""
    key = {"full": "full_g", "lodo": "lodo_g"}[kind]
    return {t: float(v["rrms"]) for t, v in _j(arm_dir(arm) / "final" / "final.json")["arms"][key].items()}


def load_records() -> dict:
    if "rec" not in _CACHE:
        from cryoml.explore.e1_correction import load_devices
        _CACHE["rec"] = {r.tag: r for r in load_devices(AD / "records" / "all18")}
    return _CACHE["rec"]


def arm_cards(arm_q3: str, tags, m2: dict) -> dict:
    """Q3 arms: 'published' = the template card (multiplier-1 card), C0_refit
    = the I25-1 C0_refit card of the arm, M2_card / M2_full = the M2 card."""
    devs = devices()
    cards = {}
    for t in tags:
        d = devs[t]
        p = None if arm_q3 == "published" else (c0_card(ARM, t) if arm_q3 == "C0_refit" else dict(m2[t]))
        cards.setdefault(d.dev_type, []).append((d.bin, p))
    return cards


# rebinding: every I14-1 / I15-1 / I16-1 stage writes under BASE and uses the I25-1 inputs
drv14.OUT = BASE
drv14.PREREG = PREREG
drv14.UNIT_TESTS = UNIT_TESTS
drv14.corr_full, drv14.corr_fold, drv14.heldout_card = corr_full, corr_fold, heldout_card
drv14.load_records = load_records
drv14.arm_cards = arm_cards
drv15.OUT = BASE
drv16.OUT = BASE
drv16.corr_full = corr_full


def tags_all() -> list[str]:
    return sorted(drv16.SMOKE_TAGS) if SMOKE else sorted(devices())


# ---------------------------------------------------------------------------
# scoring (paper RRMS, frozen curve set) and curve order
# ---------------------------------------------------------------------------
def score(rec, sims) -> float:
    return float(m25.score_rule(rec, sims, m25.frozen_tags(rec))["rrms"])


def deck_keys(tag: str) -> list[str]:
    """Curve keys in the order ``drv14.measured_run`` returns them."""
    from cryoml.card_eval import device_by_tag
    from cryoml.data_io import load_device_curves
    from cryoml.metrics import tag_metric_curves
    curves = load_device_curves(device_by_tag(tag))
    return [f"{k}@{b:g}" for (k, b), _ in sorted(tag_metric_curves(curves).items())]


def to_rec_order(rec, keys, arrs) -> list[np.ndarray]:
    d = dict(zip(keys, arrs))
    if set(d) != {c.key for c in rec.curves}:
        raise RuntimeError(f"{rec.tag}: deck keys {sorted(d)} != record keys")
    return [np.asarray(d[c.key], float) for c in rec.curves]


# ---------------------------------------------------------------------------
# V2 unit tests
# ---------------------------------------------------------------------------
def stage_unit() -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("CRYOML_PDK_TEMPLATE", "CRYOML_I25E_ARM",
                                                           "CRYOML_I25E_SMOKE")}
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run([sys.executable, "-m", "unittest", *UNIT_TESTS], cwd=str(ROOT), capture_output=True,
                          text=True, timeout=3600, check=False, env=env)
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-5:]
    out = {"files": list(UNIT_TESTS), "returncode": proc.returncode, "passed": proc.returncode == 0,
           "tail": tail, "environment": "default template (CRYOML_PDK_TEMPLATE unset)"}
    jdump(out, OUT / ("smoke" if SMOKE else "") / "unit_tests.json")
    print(f"unit tests: rc {proc.returncode}; {tail[-1] if tail else ''}", flush=True)
    return out


# ---------------------------------------------------------------------------
# pairs: V1 (card-only deck x exp(g)) and element-deck accuracy
# ---------------------------------------------------------------------------
def pair_osdi(rd: Path, kind: str, tag: str) -> Path:
    return drv14.arm_osdi(rd, ARM) if kind == "full" else drv14.fold_osdi(rd, ARM, tag)


def compile_task(task):
    rd_s, kind, tag = task
    corr = corr_full(ARM) if kind == "full" else corr_fold(ARM, tag)
    osdi = pair_osdi(Path(rd_s), kind, tag)
    text = c14.to_verilog_a(corr)
    c = m1.compile_osdi(text, osdi.with_suffix(".va"), osdi)
    return {"kind": kind, "tag": tag, "sha256": c["sha256"], "compiled": bool(c["compiled"]),
            "sense_branch_rs0": ("V(sense) <+ RS * I(sense);" in text and "parameter real RS = 0.0" in text)}


def pair_task(task):
    rd_s, kind, tag = task
    from cryoml.explore.i2_1_joint import with_sims
    rec = load_records()[tag]
    corr = corr_full(ARM) if kind == "full" else corr_fold(ARM, tag)
    card = m2_card(ARM, tag) if kind == "full" else heldout_card(ARM, tag)
    pm = devices()[tag].dev_type == "pmos"
    keys = deck_keys(tag)
    co = drv14.measured_run(tag, card, "card")
    cr = drv14.measured_run(tag, card, "card_rep")
    el = drv14.measured_run(tag, card, "elem", pair_osdi(Path(rd_s), kind, tag), c14.module_name(corr))
    s_card = to_rec_order(rec, keys, co["sims"])
    s_cr = to_rec_order(rec, keys, cr["sims"])
    s_rep = to_rec_order(rec, keys, cr["sense"])
    s_el = to_rec_order(rec, keys, el["sims"])
    py_v1 = corr.apply(with_sims(rec, s_card))                   # V1: card-only deck x exp(g)
    py = corr.apply(with_sims(rec, s_cr))                        # card + replica deck, card current x exp(g)
    chan = [c14.current_fwd(ic, ir, corr.g_v(pm, c.vg, c.vd)) for ic, ir, c in zip(s_cr, s_rep, rec.curves)]
    eq = v2.eq_violation(np.concatenate(s_el), np.concatenate(chan), rel=1e-3, absol=1e-12)
    return {"kind": kind, "tag": tag,
            "rrms_v1_card_only_x_exp_g": score(rec, py_v1),
            "rrms_native": score(rec, s_el), "rrms_python": score(rec, py),
            "rrms_python_channel": score(rec, chan), "rrms_card_only": score(rec, s_card),
            "native_vs_channel_max_ratio_1e-3_1e-12": eq["max_ratio"],
            "msgs": co["msgs"] + cr["msgs"] + el["msgs"]}


def stage_pairs(workers: int) -> dict:
    rd = BASE / "pairs"
    tags = tags_all()
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        comp = pool.map(compile_task, [(str(rd), "full", "full")] + [(str(rd), "lodo", t) for t in tags],
                        chunksize=1)
        t_comp = time.time() - t0
        rows = pool.map(pair_task, [(str(rd), k, t) for k in ("full", "lodo") for t in tags], chunksize=1)
    out = {"arm": ARM, "compile": comp, "compile_s": round(t_comp, 1), "rows": {}, "per_kind": {}}
    worst = 0.0
    for k in ("full", "lodo"):
        stored = stored_rows(ARM, k)
        rr = {r["tag"]: r for r in rows if r["kind"] == k}
        for t, r in rr.items():
            r["stored"] = stored[t]
            r["v1_abs_diff"] = abs(r["rrms_v1_card_only_x_exp_g"] - stored[t])
            r["native_minus_stored"] = r["rrms_native"] - stored[t]
            worst = max(worst, r["v1_abs_diff"])
            out["rows"][f"{k}|{t}"] = r
        nat = {t: r["rrms_native"] for t, r in rr.items()}
        out["per_kind"][k] = {
            "native": nat, "python": {t: r["rrms_python"] for t, r in rr.items()},
            "v1_max_abs_diff": max(r["v1_abs_diff"] for r in rr.values()),
            "max_abs_native_minus_stored": max(abs(r["native_minus_stored"]) for r in rr.values()),
            "max_native_vs_channel_ratio": max(r["native_vs_channel_max_ratio_1e-3_1e-12"] for r in rr.values()),
            "n_msgs": sum(len(r["msgs"]) for r in rr.values()),
            "n_finite_native": int(sum(np.isfinite(v) for v in nat.values())),
            "summary_native": m25.aggregate(nat),
            "summary_stored": m25.aggregate({t: stored[t] for t in rr}),
            "summary_python_channel": m25.aggregate({t: r["rrms_python_channel"] for t, r in rr.items()})}
    out["v1_max_rrms_diff"] = worst
    out["v1_n_rows"] = len(rows)
    out["wall_s"] = round(time.time() - t0, 1)
    jdump(out, rd / "pairs.json")
    print(json.dumps({k: {kk: v for kk, v in out["per_kind"][k].items() if kk not in ("native", "python")}
                      for k in out["per_kind"]}, indent=1, default=float), flush=True)
    print(f"V1 max |RRMS card-only deck x exp(g) - stored| {worst:.3e} ({len(rows)} rows); "
          f"compile {sum(c['compiled'] for c in comp)} of {len(comp)}", flush=True)
    return out


# ---------------------------------------------------------------------------
# deploy (M2 rows from I25-1 + pairs; compile, q12, q3, report)
# ---------------------------------------------------------------------------
def stage_m2(rd: Path) -> dict:
    full = _j(AD / "full" / "results.json")
    cards = {t: dict(r["params"]) for t, r in full["rows"].items()}
    jdump(cards, rd / "cards_M2.json")
    pairs = jload(BASE / "pairs" / "pairs.json")
    tags = tags_all()
    out = {"arm": ARM, "source": str((AD / "full" / "results.json").relative_to(ROOT)),
           "note": "M2 cards = I25-1 full-data in-sample refits (g frozen in the loop); not refit again",
           "rows": {t: {"rrms_start": full["rows"][t]["rrms_start"], "rrms": full["rows"][t]["rrms"],
                        "failed": bool(full["rows"][t]["failed"]) or not np.isfinite(full["rows"][t]["rrms"]),
                        "n_sim": full["rows"][t]["n_sim"], "runtime_s": full["rows"][t]["runtime_s"]}
                    for t in tags},
           "native": {t: pairs["rows"][f"full|{t}"] for t in tags}}
    jdump(out, rd / "refit.json")
    print(f"M2 rows: {len(out['rows'])}, failed {sum(r['failed'] for r in out['rows'].values())}", flush=True)
    return out


def vds0_table(rd: Path) -> dict:
    q12 = jload(rd / "q12.json")["devices"]
    j0 = m17.vds0_column(m1.GRID_VDS)
    per = {}
    for tag in sorted(q12):
        z = np.load(rd / "arrays" / f"q12_{tag}.npz")
        per[tag] = m17.vds0_values(z, j0, m1.GRID_VGS)
    return {"per_device": per, "summary": m17.summarize_vds0(per)}


def stage_report(rd: Path) -> dict:
    out = drv14.stage_report(rd, ARM)
    vt = vds0_table(rd)
    dec = m17.decide_deploy_i17(out["values"], vt["summary"])
    out["decision_I16_1_rule"] = m17.v2.decide_deploy(m17.values_i16_rule(out["values"], vt["summary"]))
    out["decision"] = {k: dec[k] for k in ("deploy", "core_fail", "restrictions")}
    out["values_I17_1_rule"] = m17.values_i17_rule(out["values"], vt["summary"])
    out["vds0"] = vt
    out["label"] = (f"I25-1 element checks, arm {ARM} ({'7' if ARM == 'A' else '45'} parameters): full-data g "
                    f"(lambda*_full 5 V) with the I25-1 full-data cards; I17-1 rules")
    jdump(out, rd / "summary.json")
    print(json.dumps({"I17_1_rule": out["decision"], "I16_1_rule": out["decision_I16_1_rule"],
                      "values": out["values_I17_1_rule"], "vds0": vt["summary"]}, indent=1, default=float),
          flush=True)
    return out


def stage_restrict(rd: Path) -> dict:
    q12 = jload(rd / "q12.json")["devices"]
    out = {"q1_max_rel_1nA": {t: {tol: r[f"q1_{tol}_max_rel_1nA"] for tol, _ in TOLS}
                              for t, r in sorted(q12.items())},
           "arm": drv16.sign_items(rd, corr_full(ARM), q12, f"I25-1 arm {ARM} full-data g")}
    jdump(out, rd / "restrictions.json")
    print(json.dumps({"q1_default_max": max(v["default"] for v in out["q1_max_rel_1nA"].values()),
                      "totals": out["arm"]["totals"]}, indent=1), flush=True)
    return out


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------
def deployed_ok() -> dict:
    c = corr_full(ARM)
    full = _j(AD / "full" / "results.json")
    inner = _j(AD / "inner" / "inner.json")
    lam_full = float(full["lambda_star_full"])
    lam_inner = float(inner["lambda_star"]["full"])
    ok = (c.inputs == "sym_v2" and abs(c.eps - 0.01) < 1e-15 and abs(lam_full - LAMBDA_EXPECTED) < 1e-15
          and abs(lam_inner - LAMBDA_EXPECTED) < 1e-15)
    return {"ok": bool(ok), "corr_inputs": c.inputs, "corr_eps": c.eps, "lambda_star_full_results": lam_full,
            "lambda_star_full_inner": lam_inner}


def stage_decide() -> dict:
    unit = jload(OUT / "unit_tests.json")
    pairs = jload(BASE / "pairs" / "pairs.json")
    dep = jload(BASE / "deploy" / "summary.json")
    rest = jload(BASE / "deploy" / "restrictions.json")
    rt = jload(BASE / "deploy" / "runtime.json")
    m2 = jload(BASE / "deploy" / "cards_M2.json")
    fails = []
    if not (pairs["v1_max_rrms_diff"] <= 1e-9) or pairs["v1_n_rows"] != 36:
        fails.append("V1 card-only deck x exp(g) reproduces the stored I25-1 rows (36 rows)")
    if not unit["passed"]:
        fails.append("V2 unit tests")
    if (any(pairs["per_kind"][k]["n_finite_native"] != 18 for k in ("full", "lodo")) or len(m2) != 18
            or dep["refit_in_sample"]["n_failed"] != 0):
        fails.append("V3 18 finite element-deck rows per row type; 18 M2 cards")
    v4 = deployed_ok()
    if not v4["ok"]:
        fails.append("V4 deployed g (sym_v2, EPS 0.01 V, lambda 5 V)")
    validity = {"valid": not fails, "fails": fails, "v1_max_rrms_diff": pairs["v1_max_rrms_diff"],
                "v1_n_rows": pairs["v1_n_rows"], "unit_tests_passed": unit["passed"], "v4": v4}
    if not validity["valid"]:
        outcome = {"outcome": "invalid", "action": "stop: fix and re-register"}
    else:
        outcome = {"outcome": dep["decision"]["deploy"], "core_fail": dep["decision"]["core_fail"],
                   "restrictions": dep["decision"]["restrictions"]}
    out = {"prereg": str(PREREG.relative_to(ROOT)), "arm": ARM,
           "arm_label": "7 parameters" if ARM == "A" else "45 parameters",
           "validity": validity, **outcome,
           "deploy_under_I16_1_rule": dep["decision_I16_1_rule"],
           "values": dep["values_I17_1_rule"], "vds0": dep["vds0"]["summary"], "thresholds": v2.THR,
           "runtime_grid": dep["runtime_grid"], "runtime_circuits": dep["runtime_circuits"],
           "runtime_ro5_A": {k: rt[k] for k in ("ratio_card_rep", "ratio_elem", "load_s")},
           "q3_not_ok": dep["q3_not_ok"], "refit_in_sample": dep["refit_in_sample"],
           "restriction_totals": rest["arm"]["totals"],
           "q1_default_max_per_device": {t: v["default"] for t, v in rest["q1_max_rel_1nA"].items()},
           "element_deck_accuracy": {k: {kk: v for kk, v in pairs["per_kind"][k].items()
                                         if kk not in ("native", "python")} for k in ("full", "lodo")},
           "element_deck_rows": {k: pairs["per_kind"][k]["native"] for k in ("full", "lodo")},
           "compile": {"n": len(pairs["compile"]), "compiled": sum(c["compiled"] for c in pairs["compile"]),
                       "sense_branch_rs0": all(c["sense_branch_rs0"] for c in pairs["compile"])}}
    jdump(out, BASE / "decision.json")
    print(json.dumps({k: out[k] for k in ("validity", "outcome", "deploy_under_I16_1_rule", "values", "vds0",
                                          "restriction_totals")}, indent=1, default=float), flush=True)
    return out


def main() -> int:
    t0 = time.time()
    st = ARGS.stage
    rd = BASE / "deploy"
    tags = tags_all()
    if st == "unit":
        stage_unit()
    elif st == "pairs":
        stage_pairs(ARGS.workers)
    elif st == "decide":
        stage_decide()
    elif st == "restrict":
        stage_restrict(rd)
    elif st == "runtime":
        drv15.stage_runtime(rd, ARM)
    else:
        workers = min(ARGS.workers, 4)
        stages = ("m2", "compile", "q12", "q3", "report") if st == "deploy" else (st,)
        timing = {}
        for s in stages:
            t1 = time.time()
            if s == "m2":
                stage_m2(rd)
            elif s == "compile":
                drv14.stage_compile(rd, ARM)
            elif s == "q12":
                drv15.stage_q12(rd, ARM, tags, workers)
            elif s == "q3":
                drv14.stage_q3(rd, ARM, workers)
            else:
                stage_report(rd)
            timing[s] = round(time.time() - t1, 1)
        tf = rd / "timing.json"
        old = jload(tf) if tf.exists() else {}
        old.update(timing)
        jdump(old, tf)
    print(f"stage {st} (arm {ARM}{', smoke' if SMOKE else ''}) done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
