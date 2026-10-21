#!/usr/bin/env python3
"""Report batch for I11-2: shared bias-only correction with a temperature
input, held-out temperature (IHP SG13G2 DEV, 4 temperatures; TU Delft 40 nm
DEV, 2 temperatures).  Outcome: INVALID (V2), stopped by rule.

Reads stored files only (no NGSpice, no fit, no training):

* ``out/research/explore/I11_2/{prereg.json, prediction.json, validity.json,
  decision.json, v2_diagnosis.json}``
* ``out/research/explore/I11_2/lodo_{ihp,tud}/results.json``
* stored I5-1 / I6-1 rows (C0_refit, M_bias, X300) through the I11-2 script

Checks: the eight primary contrasts, the labels, the outcome, the V2 values,
the per-temperature 95% contrasts, the aggregates and the prediction scoring
are recomputed from the per-record RRMS and compared with decision.json.

  PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i11_2.py --date 2026-10-21
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_explore_e2_e3_e4 as pe  # noqa: E402  (helpers, style)
from plot_explore_e2_e3_e4 import (BG, INK, MUTED, GREY, BLUE, ORANGE, RED, VIOLET, AQUA, DPI,  # noqa: E402
                                   EXP, ROOT, load, rel, style)

rf = pe.rf
plt = pe.plt

_spec = importlib.util.spec_from_file_location("i11_2_script", ROOT / "scripts/research/explore/i11_2_temp.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)
k = S.k

TOPIC = "i11_2"
I11 = EXP / "I11_2"
F = {"prereg": I11 / "prereg.json", "prediction": I11 / "prediction.json", "validity": I11 / "validity.json",
     "decision": I11 / "decision.json", "v2": I11 / "v2_diagnosis.json",
     "ihp": I11 / "lodo_ihp/results.json", "tud": I11 / "lodo_tud/results.json"}
COMMITS = {"prereg": "964a828", "code": "02df93c", "prediction": "812d14c", "validity": "886271a"}
IHP_ARMS = ("C0_refit", "M_bias_T", "M_bias_pool", "M_bias_near", "M_bias_in", "M_bias_sameT", "lin_T", "lin_pool")
TUD_ARMS = ("C0_refit", "M_bias_X", "M_bias_in", "M_bias_sameT")
PRIMARY = [("ihp_interp_M_bias_T_vs_C0_refit", "IHP interp. (300, 343 K)", "M_bias_T vs C0_refit"),
           ("ihp_interp_M_bias_T_vs_M_bias_pool", "IHP interp. (300, 343 K)", "M_bias_T vs M_bias_pool"),
           ("ihp_interp_M_bias_T_vs_M_bias_near", "IHP interp. (300, 343 K)", "M_bias_T vs M_bias_near"),
           ("ihp_extrap_M_bias_T_vs_C0_refit", "IHP extrap. (233, 398 K)", "M_bias_T vs C0_refit"),
           ("ihp_extrap_M_bias_T_vs_M_bias_pool", "IHP extrap. (233, 398 K)", "M_bias_T vs M_bias_pool"),
           ("ihp_extrap_M_bias_T_vs_M_bias_near", "IHP extrap. (233, 398 K)", "M_bias_T vs M_bias_near"),
           ("tud_4K_M_bias_X_vs_C0_refit", "TU Delft 4.13 K held out", "M_bias_X vs C0_refit"),
           ("tud_295K_M_bias_X_vs_C0_refit", "TU Delft 295 K held out", "M_bias_X vs C0_refit")]
BANNER = "DESCRIPTIVE: run INVALID (V2 failed); no label is a decision"


def ci(c: dict) -> str:
    return f"{c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"


def close(a, b, tol=1e-12) -> bool:
    if isinstance(a, dict):
        return all(close(a[x], b[x], tol) for x in a)
    if isinstance(a, str) or a is None:
        return a == b
    a, b = float(a), float(b)
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= tol * max(1.0, abs(b))


# ---------------------------------------------------------------------------
# load + recompute
# ---------------------------------------------------------------------------
def load_all() -> dict:
    d = {kk: load(p) for kk, p in F.items()}
    dec = d["decision"]
    st_i, st_t = S.stored_rows(S.I5, S.IHP_TEMPS), S.stored_rows(S.I6, S.TUD_TEMPS)
    si = {**d["ihp"]["per_record"], "C0_refit": st_i["C0_refit"], "M_bias_sameT": st_i["M_bias"]}
    stt = {**d["tud"]["per_record"], "C0_refit": st_t["C0_refit"], "M_bias_sameT": st_t["M_bias"]}
    m6 = S.tud_module()
    lvl = k.LEVEL_PRIMARY
    prim, c95 = {}, {}
    for name, temps in (("interp", list(S.INTERP)), ("extrap", list(S.EXTRAP))):
        for ctl in ("C0_refit", "M_bias_pool", "M_bias_near"):
            kk = f"ihp_{name}_M_bias_T_vs_{ctl}"
            prim[kk] = S.contrast(si["M_bias_T"], si[ctl], temps, lvl)
            c95[kk] = S.contrast(si["M_bias_T"], si[ctl], temps, 0.95)
    for t in S.TUD_TEMPS:
        kk = f"tud_{t}K_M_bias_X_vs_C0_refit"
        prim[kk] = S.contrast(stt["M_bias_X"], stt["C0_refit"], [t], lvl, clusters_of=m6.group_of)
        c95[kk] = S.contrast(stt["M_bias_X"], stt["C0_refit"], [t], 0.95, clusters_of=m6.group_of)
    for kk, v in prim.items():
        assert close(v, dec["primary"][kk]), kk
    labels = {f"ihp_{n}": k.label_ihp(prim[f"ihp_{n}_M_bias_T_vs_C0_refit"], prim[f"ihp_{n}_M_bias_T_vs_M_bias_pool"],
                                      prim[f"ihp_{n}_M_bias_T_vs_M_bias_near"]) for n in ("interp", "extrap")}
    labels.update({f"tud_{t}K": k.label_tud(prim[f"tud_{t}K_M_bias_X_vs_C0_refit"]) for t in S.TUD_TEMPS})
    assert labels == dec["labels"], labels
    # V2 as decided (54 rows) and as stated in prereg.json (233 / 343 K, 36 rows)
    x300 = S.stored_x300()
    near = si["M_bias_near"]
    d2 = [abs(near[kk] - x300[kk]) / x300[kk] for kk in near if kk in x300]
    assert len(d2) == dec["validity"]["V2_near_equals_X300"]["n"]
    assert max(d2) == dec["validity"]["V2_near_equals_X300"]["max_rel_diff"]
    d2p = [abs(near[kk] - x300[kk]) / x300[kk] for kk in near if kk in x300 and S.unkey(kk)[1] in (233, 343)]
    v2 = d["v2"]
    assert len(d2p) == v2["V2_as_prereg_text"]["n"] == 36 and max(d2p) == v2["V2_as_prereg_text"]["max_rel_diff"]
    val = dec["validity"]
    valid = bool(val["V1_c0_refit_reproduced"] and val["V5_corrections_reproduced"]
                 and max(d2) <= S.REPRO_TOL and val["V3_ok"])
    ihp_lab = {n: labels[f"ihp_{n}"] for n in ("interp", "extrap")}
    re = k.decide(ihp_lab, valid)
    assert re["outcome"] == dec["outcome"] == "invalid" and re["action"] == dec["action"]
    v3 = {f"ihp_{a}": int(sum(not np.isfinite(x) for x in si[a].values()))
          for a in ("M_bias_T", "M_bias_pool", "M_bias_near")}
    v3["tud_M_bias_X"] = int(sum(not np.isfinite(x) for x in stt["M_bias_X"].values()))
    assert v3 == val["V3_failed"]
    # what the rule gives with V2 as written in prereg.json (post hoc, not the decision)
    alt = k.decide(ihp_lab, bool(val["V1_c0_refit_reproduced"] and val["V5_corrections_reproduced"]
                                 and max(d2p) <= S.REPRO_TOL and val["V3_ok"]))
    # per-temperature 95% contrasts vs C0_refit (decide secondary) and aggregates
    sec = dec["secondary"]
    pt = {}
    for t in S.IHP_TEMPS:
        for a in IHP_ARMS[1:]:
            c = S.contrast(si[a], si["C0_refit"], [t], 0.95)
            assert close(c, sec["per_temperature_95"][f"ihp_{t}K"][f"{a}_vs_C0_refit"]), (t, a)
            pt[("ihp", t, a)] = c
    tud95 = {}
    for t in S.TUD_TEMPS:
        for a in TUD_ARMS[1:]:
            c = S.contrast(stt[a], stt["C0_refit"], [t], 0.95)
            cc = S.contrast(stt[a], stt["C0_refit"], [t], 0.95, clusters_of=m6.group_of)
            assert close(c, sec["tud_device_bootstrap"][f"{t}K_{a}_vs_C0_refit_95"])
            assert close(cc, sec["tud_device_bootstrap"][f"{t}K_{a}_vs_C0_refit_95_cluster"])
            tud95[(t, a)] = (c, cc)
    agg = {}
    for ds, sc, temps, arms in (("ihp", si, S.IHP_TEMPS, IHP_ARMS), ("tud", stt, S.TUD_TEMPS, TUD_ARMS)):
        for t in temps:
            for a in arms:
                v = np.array([x for kk, x in sc[a].items() if S.unkey(kk)[1] == t], float)
                agg[(ds, t, a)] = (float(np.mean(v)), float(np.median(v)), len(v))
                ref = sec["aggregates"][f"{ds}_{t}K_{a}"]
                assert close(agg[(ds, t, a)][0], ref["mean"]) and close(agg[(ds, t, a)][1], ref["median"]), (ds, t, a)
    # prediction scoring
    pred = d["prediction"]
    score = {}
    for t, p in pred["per_temperature"].items():
        for name, sign in p["predicted"].items():
            arm = name.split("_vs_")[0]
            lr, _ = k.unit_log_ratios(S.as_pairs(si[arm]), S.as_pairs(si["C0_refit"]), [int(t)])
            y = float(np.mean(list(lr.values())))
            obs = "gain" if y < 0 else "no gain"
            score[(t, name)] = {"predicted": sign, "y": y, "observed": obs, "correct": sign == obs}
            ref = sec["prediction_scoring"]["rows"][f"{t}K {name}"]
            assert close(y, ref["observed_ln_gmr"]) and ref["correct"] == (sign == obs)
    return {**d, "si": si, "stt": stt, "prim": prim, "c95": c95, "labels": labels, "valid": valid, "alt": alt,
            "d2": d2, "d2p": d2p, "pt": pt, "tud95": tud95, "agg": agg, "score": score}


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def col(key, head, md=None, kind="rtext", group=None):
    return {"key": key, "head": head, "md": md or head, "kind": kind, "group": group}


def t_decision(d: dict) -> rf.Table:
    dec, val, v2 = d["decision"], d["decision"]["validity"], d["v2"]
    rows = [
        {"item": "OUTCOME", "cond": "invalid iff V1, V2, V3 or V5 fails", "val": "V2 fails",
         "res": f"INVALID ({dec['action']})"},
        {"item": "V1", "cond": "6 stored C0_refit rows reproduced (<= 1e-9)",
         "val": f"max {max(d['validity']['V1_c0_refit_rel_diff'].values()):.0e} on 6",
         "res": "pass" if val["V1_c0_refit_reproduced"] else "FAIL"},
        {"item": "V2", "cond": "M_bias_near = stored X300 rows (<= 1e-9)",
         "val": f"max {max(d['d2']):.3f} on {len(d['d2'])} rows (233, 343, 398 K)",
         "res": "FAIL" if not val["V2_near_equals_X300"]["ok"] else "pass"},
        {"item": "V2 (prereg text, post hoc)", "cond": "same, 233 / 343 K, 36 records",
         "val": f"max {max(d['d2p']):.0e} on {len(d['d2p'])} rows", "res": "would pass"},
        {"item": "V3", "cond": "<= 2 failed records per primary arm",
         "val": ", ".join(f"{a} {n}" for a, n in val["V3_failed"].items()), "res": "pass" if val["V3_ok"] else "FAIL"},
        {"item": "V4", "cond": "TU Delft simulator timeouts (reported)", "val": str(val["V4_tud_timeouts"]),
         "res": "reported"},
        {"item": "V5", "cond": "3 stored fold corrections reproduced (<= 1e-9)",
         "val": f"max {max(d['validity']['V5_correction_max_rel_diff'].values()):.0e} on 3",
         "res": "pass" if val["V5_corrections_reproduced"] else "FAIL"},
    ]
    for kk, ds, what in PRIMARY:
        if kk.endswith("vs_C0_refit"):
            lab = d["labels"]["ihp_" + kk.split("_")[1]] if kk.startswith("ihp") else d["labels"][kk[:-len(
                "_M_bias_X_vs_C0_refit")]]
            rows.append({"item": f"label {ds}", "cond": "pre-registered label rule (descriptive)",
                         "val": "", "res": lab})
    notes = [f"Pre-registration in force from {COMMITS['prereg']}; code {COMMITS['code']}; prediction committed "
             f"{COMMITS['prediction']} before any held-out refit; validity run {COMMITS['validity']}.",
             "V2 cause (post hoc, v2_diagnosis.json): run_decide compared every M_bias_near row with a stored X300 "
             "row; X300 rows exist at 233, 343 and 398 K. At 398 K M_bias_near uses the lodo_343K fold correction "
             "(398 -> 343 by definition), X300 the lodo_300K one. 233 / 343 K: 36 of 36 rows equal (difference 0).",
             f"With V2 restricted to 233 / 343 K as written in prereg.json the rule would give "
             f"'{d['alt']['outcome']}' ({d['alt']['action']}). Post hoc; the recorded outcome stays invalid."]
    return rf.Table("01_table_i11_2_decision", "I11-2 INVALID RUN: validity, labels (descriptive) and the outcome",
                    [col("item", "item", kind="text"), col("cond", "condition", kind="text"),
                     col("val", "value", kind="text"), col("res", "result", kind="text")], rows, notes)


def t_primary(d: dict) -> rf.Table:
    rows = []
    for kk, ds, what in PRIMARY:
        p, c = d["prim"][kk], d["c95"][kk]
        rows.append({"ds": ds, "what": what, "b": ci(p), "c": f"[{c['ci_low']:.3f}, {c['ci_high']:.3f}]",
                     "low": f"{p['units_lower']}/{p['n_units']}", "res": p["resample_unit"]})
    notes = [BANNER + ".",
             f"GMR = exp(mean over units of ln(arm / control)); IHP unit = DUT (per DUT the mean over the stratum's two "
             f"temperatures, device bootstrap), TU Delft unit = device with a 7-group cluster bootstrap; Bonferroni "
             f"level {k.LEVEL_PRIMARY} over 8 contrasts; 10,000 resamples, seed 0.",
             "The M_bias_near rows are the pre-registered arm (fold correction of the nearest training temperature); "
             "the V2 failure does not come from them (table 08)."]
    return rf.Table("03_table_i11_2_primary", "I11-2 primary contrasts (descriptive, run invalid)",
                    [col("ds", "stratum", kind="text"), col("what", "contrast", kind="text"),
                     col("b", "GMR [Bonferroni CI]"), col("c", "95% CI"), col("low", "units lower"),
                     col("res", "resample", kind="text")], rows, notes)


def t_ihp_temps(d: dict) -> rf.Table:
    rows = []
    for t in S.IHP_TEMPS:
        nt = k.nearest_temp(t, S.IHP_TEMPS)
        for a in IHP_ARMS:
            m, med, n = d["agg"][("ihp", t, a)]
            r = {"t": f"{t} K", "arm": a + (f" (g {nt} K)" if a == "M_bias_near" else ""), "mean": f"{m:.4f}",
                 "med": f"{med:.4f}", "n": str(n)}
            if a != "C0_refit":
                c = d["pt"][("ihp", t, a)]
                r.update(g=ci(c), low=f"{c['units_lower']}/{c['n_units']}")
            else:
                r.update(g="—", low="—")
            rows.append(r)
    notes = [BANNER + ".",
             "Held-out RRMS (strict paper RRMS of I_card x exp(g) after the refit with g frozen in the loop), 18 IHP DEV "
             "DUTs per temperature, 0 failed records. GMR vs C0_refit with a 95% device-bootstrap CI (decide secondary).",
             "M_bias_sameT = stored I5-1 same-temperature fold M_bias (reference, uses the held-out temperature); "
             "M_bias_in = M_bias_T trained on all temperatures, held-out DUT excluded (reference)."]
    return rf.Table("04_table_i11_2_ihp_per_temperature", "IHP SG13G2 per held-out temperature: mean RRMS and GMR vs "
                    "C0_refit (descriptive)",
                    [col("t", "T", kind="text"), col("arm", "arm", kind="text"), col("mean", "mean RRMS"),
                     col("med", "median"), col("n", "n"), col("g", "GMR vs C0_refit [95% CI]"),
                     col("low", "lower")], rows, notes)


def t_tud(d: dict) -> rf.Table:
    rows = []
    for t in S.TUD_TEMPS:
        for a in TUD_ARMS:
            m, med, n = d["agg"][("tud", t, a)]
            r = {"t": f"{t} K held out", "arm": a, "mean": f"{m:.4f}", "med": f"{med:.4f}", "n": str(n)}
            if a != "C0_refit":
                c, cc = d["tud95"][(t, a)]
                r.update(g=ci(c), gc=f"[{cc['ci_low']:.3f}, {cc['ci_high']:.3f}]",
                         low=f"{c['units_lower']}/{c['n_units']}")
            else:
                r.update(g="—", gc="—", low="—")
            rows.append(r)
    notes = [BANNER + "; TU Delft labels never enter the action (one training temperature).",
             "45 DEV devices in 7 geometry groups. M_bias_X = stored I6-1 fold M_bias of the other temperature "
             "(= pool = near); M_bias_in and M_bias_sameT use the held-out temperature (references).",
             "Primary C7 / C8 use the cluster bootstrap at the Bonferroni level (table 03)."]
    return rf.Table("06_table_i11_2_tud", "TU Delft 40 nm per held-out temperature: mean RRMS and GMR vs C0_refit "
                    "(descriptive)",
                    [col("t", "T", kind="text"), col("arm", "arm", kind="text"), col("mean", "mean RRMS"),
                     col("med", "median"), col("n", "n"), col("g", "GMR [95% device CI]"),
                     col("gc", "95% cluster CI"), col("low", "lower")], rows, notes)


def t_prediction(d: dict) -> rf.Table:
    pred = d["prediction"]["per_temperature"]
    rows = []
    for (t, name), v in d["score"].items():
        p = pred[t]
        stat = (f"S3 {p['S3_M_bias_T']:+.3f}" if name.startswith("M_bias_T") else
                f"S3 {p['S3_M_bias_pool']:+.3f}" if "(S3)" in name else
                f"S1_loto {p['S1_loto']:+.3f}, y {p['I7_1_rule_y']:+.3f}")
        rows.append({"t": f"{t} K", "name": name, "stat": stat, "p": v["predicted"],
                     "y": f"{v['y']:+.4f}", "o": v["observed"], "c": "yes" if v["correct"] else "no"})
    n_ok = sum(v["correct"] for v in d["score"].values())
    notes = [f"{n_ok} of {len(d['score'])} predicted signs correct. Prediction committed in {COMMITS['prediction']} "
             "before any held-out refit; training-only statistics of the IHP DEV starting-card records.",
             "Observed = sign of the mean over 18 DUTs of ln(arm / C0_refit) at the held-out temperature. Rows use the "
             "invalid run's records (the arm rows themselves are not affected by the V2 defect)."]
    return rf.Table("07_table_i11_2_prediction", "I11-2 secondary prediction vs observed (IHP, 12 signs)",
                    [col("t", "T", kind="text"), col("name", "prediction", kind="text"),
                     col("stat", "statistic"), col("p", "predicted", kind="text"), col("y", "observed ln GMR"),
                     col("o", "observed", kind="text"), col("c", "correct", kind="text")], rows, notes)


def t_v2(d: dict) -> rf.Table:
    v2 = d["v2"]
    rows = []
    for t, r in v2["per_temperature"].items():
        src = ", ".join(f"lodo_{s}K x {n}" for s, n in r["correction_source_counts"].items())
        rows.append({"t": f"{t} K", "near": f"{r['nearest_K_prereg']} K", "src": src,
                     "x": str(r["stored_X300_rows"]) if r["stored_X300_rows"] else "—",
                     "eq": f"{r['n_equal_X300']}/{r['stored_X300_rows']}" if r["stored_X300_rows"] else "—",
                     "mx": "—" if r["max_rel_diff_vs_X300"] is None else f"{r['max_rel_diff_vs_X300']:.3g}",
                     "dec": "yes" if r["in_decide_V2"] else "no", "pre": "yes" if r["in_prereg_V2"] else "no"})
    rr = v2["rerun"]
    notes = ["Post hoc (after the decision; does not change it). Source: v2_diagnosis.json "
             "(scripts/research/explore/i11_2_v2_diagnosis.py).",
             "Correction source: the stored I5-1 fold correction equal to the one used by each M_bias_near row "
             "(lodo_ihp/corrections.json.gz); 72/72 rows use the prereg mapping 233->300, 300->343, 343->300, 398->343.",
             f"Single-record refits of {rr['record']} (largest difference): with the lodo_343K correction "
             f"{rr['refit_with_I5_1_lodo_343K']:.6f} = stored M_bias_near {rr['stored_M_bias_near']:.6f}; with the "
             f"lodo_300K correction {rr['refit_with_I5_1_lodo_300K']:.6f} = stored X300 {rr['stored_X300']:.6f} "
             "(both differences 0).",
             "Cause: the decide code selected rows by 'key in stored X300 rows' instead of the prereg temperatures "
             "233 / 343 K. Affected: the validity flag only; no arm row and no contrast uses X300."]
    return rf.Table("08_table_i11_2_v2_diagnosis", "V2 diagnosis: M_bias_near rows vs stored X300 rows per held-out "
                    "temperature",
                    [col("t", "held-out T", kind="text"), col("near", "near g from"),
                     col("src", "stored fold correction used", kind="text"), col("x", "X300 rows"),
                     col("eq", "equal"), col("mx", "max rel. diff."), col("dec", "in decide V2"),
                     col("pre", "in prereg V2")], rows, notes)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def banner(ax, y=0.995) -> None:
    ax.text(0.995, y, BANNER, transform=ax.transAxes, ha="right", va="top", fontsize=9, fontweight="bold",
            color=RED, bbox={"facecolor": BG, "edgecolor": RED, "boxstyle": "round,pad=0.3", "lw": 0.8})


def fig_forest(d: dict, path: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(11.5, 6.0), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    y, ticks, labels, prev = 0.0, [], [], None
    colors = {"C0_refit": BLUE, "M_bias_pool": ORANGE, "M_bias_near": VIOLET}
    xlo, xhi = 0.6, 2.8
    for kk, ds, what in PRIMARY:
        if ds != prev:
            if prev is not None:
                y -= 0.5
            ax.text(xlo * 1.02, y + 0.35, ds, fontsize=9.5, fontweight="bold", color=INK, va="bottom")
            prev = ds
        p, c = d["prim"][kk], d["c95"][kk]
        colr = colors[what.split(" vs ")[1]]
        ax.errorbar(p["ratio"], y, xerr=[[p["ratio"] - p["ci_low"]], [p["ci_high"] - p["ratio"]]], fmt="o",
                    color=colr, ms=7, capsize=3, lw=2.2, zorder=5)
        ax.plot([c["ci_low"], c["ci_high"]], [y - 0.2] * 2, color=colr, lw=1.0, alpha=0.6, zorder=3)
        lab = d["labels"].get("ihp_" + kk.split("_")[1]) if kk.startswith("ihp") else d["labels"][kk.split("_M_")[0]]
        ax.annotate(f"{ci(p)}  {p['units_lower']}/{p['n_units']} lower", (xhi * 1.02, y), fontsize=8, color=INK,
                    va="center", annotation_clip=False)
        ticks.append(y)
        labels.append(what)
        y -= 0.75
    ax.axvline(1.0, color=INK, lw=0.9)
    ax.set_xscale("log")
    xt = [0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.6]
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{t:g}" for t in xt], fontsize=8)
    ax.minorticks_off()
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(y + 0.3, 0.9)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("GMR of held-out RRMS (log axis; < 1 = first arm lower)", fontsize=9)
    banner(ax)
    hs = [plt.Line2D([], [], color=INK, marker="o", lw=2.2, label="thick: Bonferroni CI (level 0.99375)"),
          plt.Line2D([], [], color=INK, lw=1.0, alpha=0.6, label="thin, below: 95% CI")]
    hs += [plt.Line2D([], [], color=c, marker="o", lw=0, label=f"control {n}") for n, c in colors.items()]
    ax.legend(handles=hs, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5)
    labs = d["labels"]
    fig.suptitle("I11-2 primary contrasts (run INVALID; descriptive). Labels by the rule: IHP interp. "
                 f"'{labs['ihp_interp']}', extrap. '{labs['ihp_extrap']}'; TU Delft 4 K '{labs['tud_4K']}', "
                 f"295 K '{labs['tud_295K']}'", fontsize=9.5, color=INK)
    fig.subplots_adjust(left=0.17, right=0.72, top=0.92, bottom=0.17)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(F["decision"]), rel(F["ihp"]), rel(F["tud"])]


def fig_ihp_temps(d: dict, path: Path) -> list[str]:
    arms = ("M_bias_T", "M_bias_pool", "M_bias_near", "M_bias_in", "M_bias_sameT", "lin_T", "lin_pool")
    colr = {"M_bias_T": RED, "M_bias_pool": ORANGE, "M_bias_near": VIOLET, "M_bias_in": AQUA,
            "M_bias_sameT": GREY, "lin_T": BLUE, "lin_pool": "#7fa9dd"}
    mk = {"M_bias_T": "o", "M_bias_pool": "s", "M_bias_near": "D", "M_bias_in": "^", "M_bias_sameT": "v",
          "lin_T": "P", "lin_pool": "X"}
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    off = np.linspace(-0.33, 0.33, len(arms))
    for i, a in enumerate(arms):
        for j, t in enumerate(S.IHP_TEMPS):
            c = d["pt"][("ihp", t, a)]
            ax.errorbar(j + off[i], c["ratio"], yerr=[[c["ratio"] - c["ci_low"]], [c["ci_high"] - c["ratio"]]],
                        fmt=mk[a], color=colr[a], ms=6, capsize=2, lw=1.3, label=a if j == 0 else None)
    ax.axhline(1.0, color=INK, lw=0.9)
    ax.set_yscale("log")
    yt = [0.85, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8]
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{t:g}" for t in yt], fontsize=8)
    ax.minorticks_off()
    ax.set_ylim(0.84, 2.0)
    ax.set_xticks(range(len(S.IHP_TEMPS)))
    ax.set_xticklabels([f"{t} K ({'interp.' if t in S.INTERP else 'extrap.'}; near g from "
                        f"{k.nearest_temp(t, S.IHP_TEMPS)} K)" for t in S.IHP_TEMPS], fontsize=8)
    ax.set_ylabel("GMR(arm, C0_refit), 95% device CI", fontsize=8.5)
    ax.legend(fontsize=7.5, frameon=False, ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    banner(ax)
    ax.set_title("IHP SG13G2, 18 DEV DUTs: held-out RRMS ratio to C0_refit per held-out temperature", fontsize=9.5,
                 color=INK)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.2)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(F["decision"]), rel(F["ihp"])]


def fig_v2(d: dict, path: Path) -> list[str]:
    near, x300 = d["si"]["M_bias_near"], S.stored_x300()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), dpi=DPI, gridspec_kw={"width_ratios": [1, 1.25]})
    fig.patch.set_facecolor(BG)
    ax = axs[0]
    style(ax)
    tc = {233: BLUE, 343: AQUA, 398: RED}
    for t, c in tc.items():
        keys = [kk for kk in near if kk in x300 and S.unkey(kk)[1] == t]
        ax.scatter([x300[kk] for kk in keys], [near[kk] for kk in keys], s=22, color=c, zorder=3,
                   label=f"{t} K (near g from {k.nearest_temp(t, S.IHP_TEMPS)} K)",
                   marker="o" if t != 398 else "x")
    lim = [min(x300.values()) * 0.8, max(x300.values()) * 1.2]
    ax.plot(lim, lim, color=INK, lw=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("stored I5-1 X300 RRMS (g from 300 K)", fontsize=8.5)
    ax.set_ylabel("I11-2 M_bias_near RRMS", fontsize=8.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_title("rows compared by the decide V2 check (54)", fontsize=9, color=INK)
    ax = axs[1]
    style(ax)
    for t, c in tc.items():
        keys = sorted(kk for kk in near if kk in x300 and S.unkey(kk)[1] == t)
        dd = [max(abs(near[kk] - x300[kk]) / x300[kk], 1e-17) for kk in keys]
        ax.scatter(np.arange(len(keys)) + {233: -0.2, 343: 0, 398: 0.2}[t], dd, color=c, s=20,
                   marker="o" if t != 398 else "x", label=f"{t} K")
    ax.axhline(1e-9, color=RED, lw=0.9, ls="--")
    ax.text(-0.5, 2.0e-9, "V2 threshold 1e-9", ha="left", fontsize=7, color=RED)
    ax.set_yscale("log")
    ax.set_ylim(3e-18, 1)
    ax.set_xticks(range(18))
    ax.set_xticklabels(sorted({S.unkey(kk)[0] for kk in near}), rotation=70, ha="right", fontsize=6)
    ax.set_ylabel("relative difference (0 drawn at 1e-17)", fontsize=8.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.62))
    ax.set_title("233 / 343 K: 36/36 equal (prereg V2); 398 K: different fold correction", fontsize=9, color=INK)
    fig.suptitle("V2 diagnosis (post hoc): M_bias_near rows vs stored I5-1 X300 rows", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(F["ihp"]), rel(F["v2"]), "out/research/explore/I5_1/cross/results.json"]


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    folder = rf.REPORT_ROOT / f"{args.date}_{TOPIC}"
    folder.mkdir(parents=True, exist_ok=True)
    d = load_all()
    items, tables = [], []

    def add_table(t, desc, srcs):
        rf.render_table(t, folder / f"{t.name}.png")
        tables.append(t)
        items.append((f"{t.name}.png", desc, srcs))

    def add_fig(name, fn, desc):
        items.append((name, desc, fn(d, folder / name)))

    add_table(t_decision(d), "Table of the outcome (INVALID, stopped by rule), validity V1-V5 with the V2 failure, V2 "
              "as written in prereg.json (post hoc), and the labels (descriptive)",
              [rel(F["prereg"]), rel(F["validity"]), rel(F["decision"]), rel(F["v2"])])
    add_fig("02_i11_2_forest.png", fig_forest, "Forest plot of the eight primary contrasts (Bonferroni CI thick, 95% "
            "CI thin), marked descriptive because the run is invalid")
    add_table(t_primary(d), "Table of the eight primary contrasts with Bonferroni and 95% CIs (descriptive)",
              [rel(F["decision"]), rel(F["ihp"]), rel(F["tud"])])
    add_table(t_ihp_temps(d), "Table per IHP held-out temperature: mean / median RRMS of each arm and GMR vs C0_refit "
              "with 95% CI", [rel(F["ihp"]), rel(F["decision"])])
    add_fig("05_i11_2_ihp_per_temperature.png", fig_ihp_temps, "GMR(arm, C0_refit) per IHP held-out temperature and "
            "arm with 95% device-bootstrap CI (descriptive)")
    add_table(t_tud(d), "Table per TU Delft held-out temperature: mean / median RRMS and GMR vs C0_refit with device "
              "and cluster 95% CIs", [rel(F["tud"]), rel(F["decision"])])
    add_table(t_prediction(d), "Table of the 12 pre-registered predicted signs vs the observed signs",
              [rel(F["prediction"]), rel(F["decision"])])
    add_table(t_v2(d), "Table of the V2 diagnosis: correction source and equality with X300 per held-out "
              "temperature, and the two single-record refits", [rel(F["v2"]), rel(F["ihp"])])
    add_fig("09_i11_2_v2_near_vs_x300.png", fig_v2, "V2 diagnosis: M_bias_near vs stored X300 RRMS for the 54 rows "
            "the decide check compared, and the relative difference per DUT and temperature")

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i11_2.py --date {args.date}`"
    md = [f"# Tables, {args.date} {TOPIC}", "", f"Generated {now} by {gen} from stored files only.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))

    p, lab, ag = d["prim"], d["labels"], d["agg"]
    tim_i, tim_t = d["ihp"]["timing"], d["tud"]["timing"]
    n_ok = sum(v["correct"] for v in d["score"].values())
    readme = [
        f"# Report batch {args.date} · {TOPIC}", "",
        f"Generated {now} by {gen} from stored files only (no NGSpice, no fit, no training). The primary contrasts, "
        "labels, outcome, V2 values, per-temperature contrasts, aggregates and prediction scoring are recomputed from "
        "the per-record RRMS and checked against decision.json.", "",
        "## Contents", "",
        "- I11-2: shared bias-only correction g(is_pmos, |Vgs|/VDD, |Vds|/VDD, T) with a temperature input, held-out "
        "temperature. IHP SG13G2 DEV (18 DUTs, 233 / 300 / 343 / 398 K) and TU Delft 40 nm DEV (45 devices, 7 "
        f"geometry groups, 4.13 / 295.0 K). Pre-registration `{rel(I11 / 'prereg.md')}` (in force from "
        f"{COMMITS['prereg']}); code `src/cryoml/explore/i11_2_temp.py`, `scripts/research/explore/i11_2_temp.py`; "
        f"prediction committed {COMMITS['prediction']}. lodo_ihp {d['ihp']['runtime_s']:.0f} s ({tim_i['n_trained']} "
        f"trained corrections, {tim_i['n_refits']} refits), lodo_tud {d['tud']['runtime_s']:.0f} s "
        f"({tim_t['n_refits']} refits); 0 failed records; 0 TU Delft timeouts.",
        f"- **Outcome: INVALID (pre-registered rule; action '{d['decision']['action']}').** V2 failed: the decide check "
        f"compared {len(d['d2'])} M_bias_near rows with stored X300 rows, max relative difference {max(d['d2']):.3f}. "
        "V1 (6 rows, difference 0), V3 (0 failed records) and V5 (3 corrections, difference 0) hold.",
        f"- V2 diagnosis (post hoc): prereg.json states V2 over the 36 records at 233 / 343 K. The decide code selected "
        "every M_bias_near row with a stored X300 key; X300 rows exist at 233, 343 and 398 K, so the 18 rows at 398 K "
        "entered. At 398 K M_bias_near uses the lodo_343K fold correction (398 -> 343 by definition) and X300 the "
        "lodo_300K one. 233 / 343 K: 36/36 equal (difference 0). All 72 M_bias_near rows use the prereg correction "
        f"mapping. Single-record refits of {d['v2']['rerun']['record']} reproduce both stored values (difference 0). "
        "The arm rows and the contrasts are not affected; only the validity flag is. With V2 as written the rule would "
        f"give '{d['alt']['outcome']}' (post hoc; the recorded outcome stays invalid).",
        "- Descriptive primary contrasts (Bonferroni CI): IHP interpolation M_bias_T vs C0_refit "
        f"{ci(p['ihp_interp_M_bias_T_vs_C0_refit'])}, vs M_bias_pool {ci(p['ihp_interp_M_bias_T_vs_M_bias_pool'])}, "
        f"vs M_bias_near {ci(p['ihp_interp_M_bias_T_vs_M_bias_near'])}; extrapolation "
        f"{ci(p['ihp_extrap_M_bias_T_vs_C0_refit'])}, {ci(p['ihp_extrap_M_bias_T_vs_M_bias_pool'])}, "
        f"{ci(p['ihp_extrap_M_bias_T_vs_M_bias_near'])}; TU Delft M_bias_X vs C0_refit 4 K held out "
        f"{ci(p['tud_4K_M_bias_X_vs_C0_refit'])}, 295 K {ci(p['tud_295K_M_bias_X_vs_C0_refit'])}. Labels by the rule: "
        f"{lab['ihp_interp']} / {lab['ihp_extrap']} / {lab['tud_4K']} / {lab['tud_295K']}.",
        "- IHP mean RRMS per held-out temperature (C0_refit / M_bias_T / M_bias_pool / lin_pool): "
        + "; ".join(f"{t} K " + " / ".join(f"{ag[('ihp', t, a)][0]:.4f}" for a in
                                           ("C0_refit", "M_bias_T", "M_bias_pool", "lin_pool")) for t in S.IHP_TEMPS)
        + ". TU Delft (C0_refit / M_bias_X / M_bias_sameT): "
        + "; ".join(f"{t} K " + " / ".join(f"{ag[('tud', t, a)][0]:.4f}" for a in ("C0_refit", "M_bias_X",
                                                                                     "M_bias_sameT"))
                    for t in S.TUD_TEMPS) + ".",
        f"- Secondary prediction: {n_ok} of {len(d['score'])} predicted signs correct (all predicted 'no gain'; all "
        "observed ln GMR vs C0_refit positive, 0.090 to 0.237).",
        "- Post hoc (not pre-registered): table 08, figure 09, the 'V2 (prereg text)' row of table 01 and "
        "`v2_diagnosis.json`.",
        "", "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))
    summary = [f"I11-2 temperature-input bias-only correction, held-out temperature (IHP 18 DUTs x 4 T, TU Delft 45 x 2): "
               f"INVALID (V2), stopped by rule.",
               f"V2 cause: decide compared 398 K M_bias_near rows (343 K fold g) with X300 rows (300 K fold g); "
               f"233 / 343 K 36/36 equal; arm rows unaffected.",
               f"Descriptive: IHP M_bias_T vs C0_refit interp. {p['ihp_interp_M_bias_T_vs_C0_refit']['ratio']:.3f}, "
               f"extrap. {p['ihp_extrap_M_bias_T_vs_C0_refit']['ratio']:.3f}; prediction {n_ok}/12."]
    rf.update_index(args.date, TOPIC, folder, [i[0] for i in items], summary)
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
