#!/usr/bin/env python3
"""I13-1: BSIM4 A2 clamp state of the 43-parameter cards, audit and fix.

Pre-registration: out/research/explore/I13_1/prereg.md (prereg.json).

Clamp states (``spice_pdk.BSIM4_STATES``): L = "legacy" (batch deck, first
curve clamped by b4check.c, later curves with the card values), C =
"clamped" (clamp written into the card, every curve in the fresh-process
state; the new default), U = "unclamped" (warm-up op, every curve with the
card values; audit only).

Stages (``--stage``):

  audit    the six stored arms (published, start, C0_refit, T1_inloop,
           M_bias v1, M_bias v2) rescored in L, C and U, strict paper RRMS;
           validity V1-V3; b4check.c messages of every stored card
           -> I13_1/audit/audit.json
  records  the 18 starting-card records re-collected in state C
           -> I13_1/records/real_{dev,test}.{npz,json}; comparison with the
           stored E1 records
  summary  headline all-18 table in state C (published / start from the
           audit, C0_refit / T1_inloop / M_bias v1 from I13_1/all18, M_bias v2
           from I13_1/lodo), GMRs, old vs new -> I13_1/summary.json

  PYTHONPATH=src .venv/bin/python scripts/research/explore/i13_1_clamp.py --stage audit --workers 4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

EXP = ROOT / "out/research/explore"
OUT = EXP / "I13_1"
E1 = EXP / "E1"
ALL18_OLD = EXP / "I3_2/all18"
LODO_OLD = EXP / "I12_1/lodo"
START = ROOT / "out/research/polished/direct43_r3_fd/real/summary.json"
REF = ROOT / "out/research/reference_scores.json"
SPLIT = EXP / "real_split.json"
ARMS = ("published", "start", "C0_refit", "T1_inloop", "M_bias_v1", "M_bias_v2")
STATES = {"L": "legacy", "C": "clamped", "U": "unclamped"}
CLAMP_RE = re.compile(r"set to|reset to", re.IGNORECASE)


def jload(p):
    return json.loads(Path(p).read_text())


def jdump(obj, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, default=float) + "\n")


def aggregate(values: dict) -> dict:
    from e1_lodo import aggregate as agg
    return agg(values)


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
def old_records() -> dict:
    """Stored E1 starting-card records (legacy state), in load order."""
    from cryoml.explore.e1_correction import load_devices
    recs = load_devices(E1 / "real_dev") + load_devices(E1 / "real_test")
    return recs


def t1_fold_tables(recs) -> dict:
    """Fold T1 tables as in the I3-2 all18 run (training set = the other 17
    records in load order; deterministic)."""
    from cryoml.explore.e1_correction import fit_arm
    with get_context("spawn").Pool(6) as pool:
        tabs = pool.map(_t1_task, [[r for r in recs if r.tag != t.tag] for t in recs])
    return {r.tag: t for r, t in zip(recs, tabs)}


def _t1_task(train):
    from cryoml.explore.e1_correction import fit_arm
    return fit_arm("T1", train, threads=1).to_json()


def arm_inputs() -> dict:
    """arm -> {tag: (params or None, correction json kind, correction json)}."""
    start = {t: dict(v["params"]) for t, v in jload(START)["devices"].items()}
    held = jload(ALL18_OLD / "heldout_cards.json")
    c_v1 = jload(ALL18_OLD / "corrections.json")
    v2_cards = jload(LODO_OLD / "heldout_cards.json")
    c_v2 = jload(LODO_OLD / "corrections.json")
    tags = sorted(start)
    out = {"published": {t: (None, "C0", None) for t in tags},
           "start": {t: (start[t], "C0", None) for t in tags},
           "C0_refit": {t: (held["C0_refit"][t], "C0", None) for t in tags},
           "T1_inloop": {t: (held["T1_inloop"][t], "T1", None) for t in tags},
           "M_bias_v1": {t: (held["M_bias"][t], "v1", c_v1[f"M_bias|{t}"]) for t in tags},
           "M_bias_v2": {t: (v2_cards[t], "v2", c_v2[f"M_bias_v2|{t}"]) for t in tags}}
    return out


def stored_scores() -> dict:
    a = jload(ALL18_OLD / "results.json")
    ref = jload(REF)["released"]["devices"]
    tags = sorted(a["per_device"]["M_bias"])
    return {"published": {t: ref[t]["rrms"] for t in tags},
            "start": a["posthoc_no_refit"]["C0_start"]["per_device"],
            "C0_refit": a["per_device"]["C0_refit"], "T1_inloop": a["per_device"]["T1_inloop"],
            "M_bias_v1": a["per_device"]["M_bias"],
            "M_bias_v2": jload(LODO_OLD / "results.json")["per_device"]["M_bias_v2"]}


def make_corr(kind: str, cj, t1_json=None):
    from cryoml.explore.e1_correction import Correction
    if kind == "C0":
        return Correction("C0")
    if kind == "T1":
        return Correction.from_json(t1_json)
    if kind == "v1":
        from cryoml.explore.i3_2_bias import correction_from_json
        return correction_from_json(cj)
    from cryoml.explore import i12_1_symm as m
    return m.SymCorrection.from_json(cj)


def score_task(task):
    """(arm, tag, state, params, kind, corr_json, t1_json) -> row."""
    arm, tag, state, params, kind, cj, t1 = task
    from cryoml.explore.e1_correction import score_device
    from e1_collect_real import collect
    rec = collect(tag, params, bsim4_state=STATES[state])
    corr = make_corr(kind, cj, t1)
    r = score_device(rec, corr.apply(rec))
    return {"arm": arm, "tag": tag, "state": state, "rrms": r["rrms"], "failed": r["failed"],
            "per_curve": r["per_curve"]}


def check_task(task):
    arm, tag, params = task
    from cryoml.card_eval import device_by_tag
    from cryoml.cold_start import device_bins
    from cryoml.spice_pdk import bsim4_check_messages, bsim4_clamp_params
    d = device_by_tag(tag)
    b = device_bins()[tag]
    msgs = bsim4_check_messages(d.dev_type, float(d.L_um), float(d.W_um), params, b)
    _, changes = bsim4_clamp_params(d.dev_type, b, params)
    after = []
    if changes:
        p2, _ = bsim4_clamp_params(d.dev_type, b, params)
        after = bsim4_check_messages(d.dev_type, float(d.L_um), float(d.W_um), p2, b)
    return {"arm": arm, "tag": tag, "messages": msgs,
            "clamp_messages": [m for m in msgs if CLAMP_RE.search(m)],
            "a2_clamp": {k: list(v) for k, v in changes.items()},
            "clamp_messages_after_fix": [m for m in after if CLAMP_RE.search(m)]}


def stage_audit(workers: int) -> dict:
    t0 = time.time()
    recs = old_records()
    t1 = t1_fold_tables(recs)
    inputs = arm_inputs()
    stored = stored_scores()
    tags = sorted(inputs["start"])
    tasks = [(arm, t, s, p, k, cj, t1[t] if k == "T1" else None)
             for arm in ARMS for t, (p, k, cj) in inputs[arm].items() for s in STATES]
    checks = [(arm, t, p) for arm in ARMS for t, (p, _, _) in inputs[arm].items()]
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(score_task, tasks, chunksize=4)
        msgs = pool.map(check_task, checks, chunksize=4)
    sc = {arm: {s: {} for s in STATES} for arm in ARMS}
    per_curve = {arm: {s: {} for s in STATES} for arm in ARMS}
    for r in rows:
        sc[r["arm"]][r["state"]][r["tag"]] = r["rrms"]
        per_curve[r["arm"]][r["state"]][r["tag"]] = r["per_curve"]
    clamped = {arm: sorted(m["tag"] for m in msgs if m["arm"] == arm and m["a2_clamp"]) for arm in ARMS}
    per_device, summary = {}, {}
    for arm in ARMS:
        per_device[arm] = {t: {"stored": stored[arm][t], **{s: sc[arm][s][t] for s in STATES},
                               "C_minus_L": sc[arm]["C"][t] - sc[arm]["L"][t],
                               "U_minus_L": sc[arm]["U"][t] - sc[arm]["L"][t],
                               "L_minus_stored": sc[arm]["L"][t] - stored[arm][t],
                               "a2_clamped": t in clamped[arm]} for t in tags}
        summary[arm] = {"stored": aggregate(stored[arm]),
                        **{s: aggregate(sc[arm][s]) for s in STATES}}
    unclamped = [(arm, t) for arm in ARMS for t in tags if t not in clamped[arm]]
    v1 = {arm: float(max(abs(sc[arm]["L"][t] - stored[arm][t]) for t in tags))
          for arm in ARMS}
    v2 = float(max(max(abs(sc[a]["C"][t] - sc[a]["L"][t]), abs(sc[a]["U"][t] - sc[a]["L"][t]))
                   for a, t in unclamped))
    other = sorted({m for x in msgs for m in x["clamp_messages"] if not m.startswith("Warning: A2")})
    after = sorted({m for x in msgs for m in x["clamp_messages_after_fix"]})
    validity = {"V1_L_vs_stored_max_abs": v1,
                "V1_pass_non_published": bool(max(v for a, v in v1.items() if a != "published") <= 1e-12),
                "V1_published_note": "stored published = reference_scores.json released (upstream sweeps)",
                "V2_unclamped_cards_max_abs_C_or_U_minus_L": v2, "V2_pass": bool(v2 == 0.0),
                "V2_n_cards": len(unclamped),
                "V3_rows": {arm: {s: len(sc[arm][s]) for s in STATES} for arm in ARMS},
                "V3_pass": all(len(sc[a][s]) == 18 for a in ARMS for s in STATES)}
    out = {"prereg": "out/research/explore/I13_1/prereg.json", "states": STATES,
           "clamped_devices": clamped, "summary": summary, "per_device": per_device,
           "per_curve": per_curve, "validity": validity,
           "check_messages": {f"{m['arm']}|{m['tag']}": m for m in msgs},
           "other_clamp_messages": other, "clamp_messages_after_fix": after,
           "distinct_messages": sorted({m for x in msgs for m in x["messages"]}),
           "t1_fold_tables": t1, "wall_s": round(time.time() - t0, 1)}
    jdump(out, OUT / "audit" / "audit.json")
    print(json.dumps({"validity": validity, "clamped": clamped, "other": other, "after": after,
                      "summary": {a: {k: (v["all_device_mean"], v["combined"]) for k, v in s.items()}
                                  for a, s in summary.items()}}, indent=1, default=float))
    return out


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
def collect_task(task):
    tag, params = task
    from e1_collect_real import collect
    return collect(tag, params, bsim4_state="clamped")


def stage_records(workers: int) -> dict:
    from cryoml.explore.e1_correction import load_devices, save_devices, score_device
    start = {t: dict(v["params"]) for t, v in jload(START)["devices"].items()}
    res = {}
    t0 = time.time()
    for split in ("real_dev", "real_test"):
        old = load_devices(E1 / split)
        with get_context("spawn").Pool(workers) as pool:
            new = pool.map(collect_task, [(r.tag, start[r.tag]) for r in old])
        for o, n in zip(old, new):
            n.meta.update(o.meta)
            if [c.key for c in o.curves] != [c.key for c in n.curves]:
                raise RuntimeError(f"{o.tag}: curve keys differ")
            d = max(float(np.max(np.abs(a.sim - b.sim) / np.maximum(np.abs(a.sim), 1e-30)))
                    for a, b in zip(o.curves, n.curves))
            tgt = max(float(np.max(np.abs(a.target - b.target))) for a, b in zip(o.curves, n.curves))
            res[o.tag] = {"split": split, "max_rel_sim_diff": d, "max_abs_target_diff": tgt,
                          "rrms_c0_old": score_device(o, [c.sim for c in o.curves])["rrms"],
                          "rrms_c0_new": score_device(n, [c.sim for c in n.curves])["rrms"]}
        (OUT / "records").mkdir(parents=True, exist_ok=True)
        save_devices(OUT / "records" / split, new)
    out = {"devices": res, "changed": sorted(t for t, v in res.items() if v["max_rel_sim_diff"] > 0),
           "wall_s": round(time.time() - t0, 1)}
    jdump(out, OUT / "records" / "records.json")
    print(json.dumps({"changed": out["changed"], "wall_s": out["wall_s"]}))
    return out


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def stage_summary() -> dict:
    from cryoml.explore.i3_2_bias import paired_ratio
    audit = jload(OUT / "audit" / "audit.json")
    a18 = jload(OUT / "all18" / "results.json")
    lodo = jload(OUT / "lodo" / "results.json")
    tags = sorted(audit["per_device"]["start"])
    new = {"published": {t: audit["per_device"]["published"][t]["C"] for t in tags},
           "start": {t: audit["per_device"]["start"][t]["C"] for t in tags},
           "C0_refit": a18["per_device"]["C0_refit"], "T1_inloop": a18["per_device"]["T1_inloop"],
           "M_bias_v1": a18["per_device"]["M_bias"], "M_bias_v2": lodo["per_device"]["M_bias_v2"]}
    old = {arm: {t: audit["per_device"][arm][t]["stored"] for t in tags} for arm in ARMS}
    gmr_pairs = [("M_bias_v1", "C0_refit"), ("M_bias_v2", "C0_refit"), ("M_bias_v2", "M_bias_v1"),
                 ("T1_inloop", "C0_refit"), ("M_bias_v1", "T1_inloop"), ("M_bias_v2", "T1_inloop")]
    gmr = {}
    for a, b in gmr_pairs:
        gmr[f"{a}_vs_{b}"] = {"new": paired_ratio(new[a], new[b]), "old": paired_ratio(old[a], old[b])}
    a1, a2 = gmr["M_bias_v2_vs_C0_refit"]["new"], gmr["M_bias_v2_vs_M_bias_v1"]["new"]
    out = {"prereg": "out/research/explore/I13_1/prereg.json", "state": "C",
           "table": {arm: {"new": aggregate(new[arm]), "old": aggregate(old[arm])} for arm in ARMS},
           "per_device": {arm: {t: {"old": old[arm][t], "new": new[arm][t],
                                    "diff": new[arm][t] - old[arm][t]} for t in tags} for arm in ARMS},
           "gmr": gmr,
           "i12_1_part_A_recomputed": {"A1_ci_high_lt_1": bool(a1["ci_high"] < 1.0),
                                       "A2_ci_high_le_1.05": bool(a2["ci_high"] <= 1.05),
                                       "note": "reported; I12-1 Part B not rerun"},
           "changed_devices": {arm: sorted(t for t in tags if new[arm][t] != old[arm][t]) for arm in ARMS},
           "timing": {"all18": a18.get("timing"), "lodo": lodo.get("timing")}}
    jdump(out, OUT / "summary.json")
    print(json.dumps({"table": {a: {k: (v["all_device_mean"], v["combined"], v["median"])
                                    for k, v in s.items()} for a, s in out["table"].items()},
                      "gmr": {k: {kk: (vv["ratio"], vv["ci_low"], vv["ci_high"], vv["wins"])
                                  for kk, vv in v.items()} for k, v in gmr.items()},
                      "A": out["i12_1_part_A_recomputed"]}, indent=1, default=float))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("audit", "records", "summary"), required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    if args.stage == "audit":
        stage_audit(args.workers)
    elif args.stage == "records":
        stage_records(args.workers)
    else:
        stage_summary()
    print(f"stage {args.stage} done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
