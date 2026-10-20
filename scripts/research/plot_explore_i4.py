#!/usr/bin/env python3
"""Report batch for exploration direction I4-1 (baselines for the I3-2
bias-only correction M_bias), the I4-2 interpretation analysis and the I3-2
follow-up F1b/F1c (SPICE-native Verilog-A runs on ngspice-41).

Reads stored files only (no NGSpice, no training):

* I4-1: ``out/research/explore/I4_1/{prereg.json,prereg_amendment_2026-10-20.json,
  b1_screen.json,decision.json}`` and ``{real_dev,gf180_c_logo,ihp_c_logo}/results.json``.
* I4-2: ``out/research/explore/I4_1/i4_2_analysis.json`` and the three
  ``i4_2_g_surface_*.png`` (copied into the batch).
* F1b/F1c: ``out/research/explore/I3_2/followup_prereg_amendment_F1{b,c}_2026-10-20.json``,
  ``followup/f1b/summary.json``, ``followup/f1c/summary.json``, the three
  ``followup/f1c_iv_*.png`` (copied) and
  ``out/tables/simulator_compare_ng41_arm64_vs_arm64_osdi.json``.

Every stored paired ratio, aggregate and the I4-1 rule output are recomputed
from the stored per-device values and checked against the stored values.

  PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i4.py --date 2026-10-20
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_explore_e2_e3_e4 as pe  # noqa: E402  (helpers, style)
from plot_explore_e2_e3_e4 import (BG, INK, MUTED, GREY, BLUE, ORANGE, AQUA,  # noqa: E402
                                   VIOLET, RED, DPI, EXP, ROOT, fam, load, rel, style)
from cryoml.explore.i3_2_bias import paired_ratio  # noqa: E402
from cryoml.pdk2.boxes import MISSPEC_PARAMS, IHP_MISSPEC_PARAMS  # noqa: E402

from matplotlib.lines import Line2D  # noqa: E402

rf = pe.rf
plt = pe.plt

TOPIC = "i4"
I41 = EXP / "I4_1"
PREREG = I41 / "prereg.json"
AMEND = I41 / "prereg_amendment_2026-10-20.json"
SCREEN = I41 / "b1_screen.json"
DEC = I41 / "decision.json"
I42 = I41 / "i4_2_analysis.json"
DS = ("real_dev", "gf180_c_logo", "ihp_c_logo")
DS_LABEL = {"real_dev": "REAL_DEV (12 measured devices, sky130 77 K)",
            "gf180_c_logo": "GF180 (BSIM4) tier c, 48 simulated targets",
            "ihp_c_logo": "IHP SG13G2 (PSP103) tier c, 48 simulated targets"}
DS_SHORT = {"real_dev": "REAL_DEV", "gf180_c_logo": "GF180 tier c", "ihp_c_logo": "IHP tier c"}
RES = {k: I41 / k / "results.json" for k in DS}
BASE = ("B1", "B2", "B3_d2", "B3_d3", "B4")
FOREST = BASE + ("C0_refit", "T1_inloop")
ARMS = ("C0_refit", "C0_refit_rerun", "B1", "B2", "B3_d2", "B3_d3", "B4", "M_bias", "T1_inloop", "B5")
ARM_DESC = {"C0_refit": "B0: held-out card refit, no correction (stored I2-1 rows)",
            "C0_refit_rerun": "C0 refit through the I4-1 refit path (reproduction check)",
            "B1": "wider card parameter set, no correction",
            "B2": "extrinsic series resistance + self-heating factor, no learned term",
            "B3_d2": "shared polynomial g, total degree 2 (12 coefficients)",
            "B3_d3": "shared polynomial g, total degree 3 (20 coefficients)",
            "B4": "shared GP discrepancy (posterior mean)",
            "M_bias": "I3-2 correction (stored rows; method under test)",
            "T1_inloop": "per-(family, curve tag) gain table (not deployable; stored)",
            "B5": "per-device correction trained on the held-out device (in-sample reference)"}
ARM_COLOR = {"C0_refit": GREY, "B1": AQUA, "B2": VIOLET, "B3_d2": ORANGE, "B3_d3": RED, "B4": MUTED,
             "M_bias": BLUE, "T1_inloop": INK, "B5": "#7a9a3a"}

I32 = EXP / "I3_2"
FU = I32 / "followup"
AM_B = I32 / "followup_prereg_amendment_F1b_2026-10-20.json"
AM_C = I32 / "followup_prereg_amendment_F1c_2026-10-20.json"
F1B = FU / "f1b/summary.json"
F1C = FU / "f1c/summary.json"
SIMCMP = ROOT / "out/tables/simulator_compare_ng41_arm64_vs_arm64_osdi.json"
IV_DEVS = ("pmos_L0p35_W0p55", "pmos_L2_W5", "nmos_L20_W0p64")


def clusters(tags) -> dict | None:
    tags = list(tags)
    return {t: t.split("#")[0] for t in tags} if "#" in tags[0] else None


def r_text(c: dict) -> str:
    return f"{c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"


def check_ratio(stored: dict, mine: dict, what: str) -> None:
    for k in ("ratio", "ci_low", "ci_high"):
        assert abs(stored[k] - mine[k]) < 1e-12, (what, k, stored[k], mine[k])
    assert stored["wins"] == mine["wins"], what


# ================================================================== I4-1 loading and checks

def load_i41() -> dict:
    res = {k: load(p) for k, p in RES.items()}
    dec = load(DEC)
    for k, r in res.items():
        assert not r["smoke"] and r["preregistered_config"], k
        per = r["per_device"]
        cl = clusters(per["M_bias"])
        for arm in ARMS:
            s = r["summary"][arm]
            v = np.array(list(per[arm].values()), dtype=float)
            assert abs(np.nanmean(v) - s["all_device_mean"]) < 1e-12, (k, arm)
            assert int(np.sum(~np.isfinite(v))) == s["n_failed"], (k, arm)
            if k == "real_dev":
                assert abs(fam(per[arm])["combined"] - s["combined"]) < 1e-12, (k, arm)
        for name, st in r["comparisons"].items():
            a, b = name.split("_vs_")
            check_ratio(st, paired_ratio(per[a], per[b], cl), f"{k} {name}")
        # B1/B2 per-device rows agree with per_device
        for arm in ("B1", "B2", "B5"):
            for t, row in r["heldout_rows"][arm].items():
                assert abs(row["rrms"] - per[arm][t]) < 1e-15, (k, arm, t)
        assert r["reproduction"]["C0_refit_max_abs_diff_rerun_vs_stored"] == 0.0
    # rule recomputed
    for x in BASE:
        d = dec["per_baseline"][x]
        rc = res["real_dev"]["comparisons"][f"M_bias_vs_{x}"]
        g = res["gf180_c_logo"]["comparisons"][f"M_bias_vs_{x}"]["ratio"]
        i = res["ihp_c_logo"]["comparisons"][f"M_bias_vs_{x}"]["ratio"]
        assert d["real_ci_high"] == rc["ci_high"] and d["gf180_ratio"] == g and d["ihp_ratio"] == i, x
        real_c, sim_c = rc["ci_high"] < 1, (g < 0.95 and i < 0.95)
        assert d["real_condition"] == real_c and d["sim_condition"] == sim_c, x
        assert d["beaten"] == (real_c or sim_c), x
    verdict = "supported against baselines" if all(dec["per_baseline"][x]["beaten"] for x in BASE) \
        else "matched by baseline " + ", ".join(x for x in BASE if not dec["per_baseline"][x]["beaten"])
    assert verdict == dec["verdict"], verdict
    for k in DS:
        check_ratio(dec["B5_reference_not_in_rule"][k], res[k]["comparisons"]["M_bias_vs_B5"], f"B5 {k}")
    return {"res": res, "dec": dec, "prereg": load(PREREG), "screen": load(SCREEN), "verdict": verdict}


def gap(r: dict) -> dict:
    s = r["summary"]
    c0, m, b5 = (s[a]["all_device_mean"] for a in ("C0_refit", "M_bias", "B5"))
    cmp = r["comparisons"]
    lg = np.log(cmp["M_bias_vs_C0_refit"]["ratio"]) / np.log(cmp["B5_vs_C0_refit"]["ratio"])
    return {"c0": c0, "m": m, "b5": b5, "frac_mean": (c0 - m) / (c0 - b5), "frac_log": float(lg)}


# ================================================================== I4-1 tables

def table_decision(d: dict) -> rf.Table:
    dec = d["dec"]["per_baseline"]
    rc = d["res"]["real_dev"]["comparisons"]
    rows = []
    for x in BASE:
        z = dec[x]
        rows.append({"x": x, "desc": ARM_DESC[x], "real": r_text(rc[f"M_bias_vs_{x}"]),
                     "rc": "met" if z["real_condition"] else "not met",
                     "g": f"{z['gf180_ratio']:.3f}", "i": f"{z['ihp_ratio']:.3f}",
                     "sc": "met" if z["sim_condition"] else "not met",
                     "b": "beaten" if z["beaten"] else "not beaten"})
    cols = [{"key": "x", "head": "baseline X", "md": "baseline X", "kind": "text"},
            {"key": "desc", "head": "definition", "md": "definition", "kind": "text"},
            {"key": "real", "head": "GMR [95% CI]", "md": "REAL_DEV GMR(M_bias, X) [95% CI]", "kind": "rtext",
             "group": "(a) REAL_DEV, CI upper < 1"},
            {"key": "rc", "head": "(a)", "md": "(a) CI upper < 1", "kind": "rtext",
             "group": "(a) REAL_DEV, CI upper < 1"},
            {"key": "g", "head": "GF180", "md": "GF180 GMR(M_bias, X)", "kind": "rtext",
             "group": "(b) GMR < 0.95 on GF180 and IHP", "family": True},
            {"key": "i", "head": "IHP", "md": "IHP GMR(M_bias, X)", "kind": "rtext",
             "group": "(b) GMR < 0.95 on GF180 and IHP", "family": True},
            {"key": "sc", "head": "(b)", "md": "(b) both < 0.95", "kind": "rtext",
             "group": "(b) GMR < 0.95 on GF180 and IHP", "family": True},
            {"key": "b", "head": "X beaten if (a) or (b)", "md": "outcome", "kind": "rtext"}]
    notes = [
        "Pre-registration out/research/explore/I4_1/prereg.json (commit e57afce); amendment "
        "prereg_amendment_2026-10-20.json (REAL_DEV base-box TRF budget per IRLS round is 4 or 5, not 4; rule "
        "unchanged). Rule: I3-2 is 'supported against baselines' if every X in {B1, B2, B3_d2, B3_d3, B4} is "
        "beaten; otherwise 'matched by baseline X'.",
        f"Rule output: {d['verdict']} (out/research/explore/I4_1/decision.json; recomputed here from the stored "
        "per-device values). Every baseline is beaten through condition (a) on REAL_DEV; condition (b) on the "
        "simulated tiers is not met for any baseline.",
        "GMR(M_bias, X) = exp(mean ln(RRMS_M_bias / RRMS_X)); < 1: M_bias has the lower held-out RRMS. REAL_DEV: "
        "12 devices, device bootstrap; GF180/IHP: 48 targets, point estimate (CI in the next tables). 10000 "
        "resamples, default_rng(0).",
        "REAL_DEV values are development evidence: REAL_DEV was used by E1, I2-1, I3-2 and its follow-ups. "
        "REAL_TEST is not loaded by any I4-1 script.",
    ]
    return rf.Table("01_table_i4_1_decision", "I4-1 decision: pre-registered baseline conditions and rule output",
                    cols, rows, notes)


def arm_rows(d: dict, k: str) -> list[dict]:
    r = d["res"][k]
    per, cmp = r["per_device"], r["comparisons"]
    n = r["n_heldout_records"]
    rows = []
    for arm in ARMS:
        s = r["summary"][arm]
        row = {"arm": arm, "desc": ARM_DESC[arm], "all": s["all_device_mean"], "med": s["median"],
               "fail": f"{s['n_failed']}/{n}"}
        if k == "real_dev":
            f = fam(per[arm])
            row.update({"comb": f["combined"], "n": f["nmos"], "p": f["pmos"]})
        if arm != "M_bias":
            c = cmp[f"M_bias_vs_{arm}"]
            row.update({"gm": r_text(c), "wm": f"{c['wins']}/{n}"})
        if arm not in ("C0_refit", "C0_refit_rerun"):
            c = cmp[f"{arm}_vs_C0_refit"]
            row.update({"gc": r_text(c), "wc": f"{c['wins']}/{n}"})
        rows.append(row)
    return rows


def table_real(d: dict) -> rf.Table:
    rows = arm_rows(d, "real_dev")
    cols = [{"key": "arm", "head": "arm", "md": "arm", "kind": "text"},
            {"key": "desc", "head": "definition", "md": "definition", "kind": "text"},
            {"key": "all", "head": "all-device", "md": "all-device", "kind": "rrms"},
            {"key": "comb", "head": "combined", "md": "combined", "kind": "rrms"},
            {"key": "n", "head": "nMOS", "md": "nMOS", "kind": "rrms", "family": True},
            {"key": "p", "head": "pMOS", "md": "pMOS", "kind": "rrms", "family": True},
            {"key": "med", "head": "median", "md": "median", "kind": "rrms"},
            {"key": "fail", "head": "failed", "md": "failed", "kind": "rtext"},
            {"key": "gm", "head": "GMR(M_bias, X) [CI]", "md": "GMR(M_bias, X) [95% CI]", "kind": "rtext"},
            {"key": "wm", "head": "M_bias < X", "md": "devices with M_bias < X", "kind": "rtext"},
            {"key": "gc", "head": "GMR(X, C0_refit) [CI]", "md": "GMR(X, C0_refit) [95% CI]", "kind": "rtext"},
            {"key": "wc", "head": "X < C0", "md": "devices with X < C0_refit", "kind": "rtext"}]
    r = d["res"]["real_dev"]
    notes = [
        "REAL_DEV leave-one-device-out, 12 devices (5 nMOS, 7 pMOS); strict paper RRMS with frozen inclusion after "
        "the held-out 43-parameter card refit (from the direct43_r3_fd card; irls 2, diff_step 0.01; base-box arms "
        "max_nfev 400). all-device = mean over 12; combined = mean of the nMOS and pMOS family means.",
        "B1 and B2 are local refits from the same starting card as C0_refit, with more active parameters "
        f"(B1: base box + {len(r['b1_extras'])} extras; B2: base box + 2); their max_nfev is scaled with the "
        "parameter count so that the TRF evaluations per IRLS round equal the base-box value of the same device "
        "(i4_1_baselines.matched_nfev).",
        "C0_refit, M_bias and T1_inloop are stored I3-2 rows; C0_refit_rerun reproduces C0_refit with max |difference| "
        "0.0. B5 is trained on the held-out device itself and is an in-sample reference, not a held-out method.",
        f"Queue job I4_1_real 2026-10-20 06:14-06:22 ({r['runtime_s']:.0f} s; training {r['timing']['train_s']:.0f} s, "
        f"{r['timing']['heldout_refit_n']} held-out refits {r['timing']['heldout_refit_s']:.0f} s).",
    ]
    return rf.Table("03_table_i4_1_real", "I4-1 REAL_DEV results for every arm, with GMR against M_bias and "
                    "against C0_refit", cols, rows, notes)


def table_sim(d: dict) -> rf.Table:
    rows = []
    for k in ("gf180_c_logo", "ihp_c_logo"):
        for row in arm_rows(d, k):
            rows.append({"pdk": DS_SHORT[k], **row})
    cols = [{"key": "pdk", "head": "dataset", "md": "dataset", "kind": "text"},
            {"key": "arm", "head": "arm", "md": "arm", "kind": "text"},
            {"key": "all", "head": "all-target", "md": "all-target mean", "kind": "rrms"},
            {"key": "med", "head": "median", "md": "median", "kind": "rrms"},
            {"key": "fail", "head": "failed", "md": "failed", "kind": "rtext"},
            {"key": "gm", "head": "GMR(M_bias, X) [CI]", "md": "GMR(M_bias, X) [95% CI]", "kind": "rtext"},
            {"key": "wm", "head": "M_bias < X", "md": "targets with M_bias < X", "kind": "rtext"},
            {"key": "gc", "head": "GMR(X, C0_refit) [CI]", "md": "GMR(X, C0_refit) [95% CI]", "kind": "rtext"},
            {"key": "wc", "head": "X < C0", "md": "targets with X < C0_refit", "kind": "rtext"}]
    rg, ri = d["res"]["gf180_c_logo"], d["res"]["ihp_c_logo"]
    notes = [
        "Simulated tier c truth sets 0-2 (16 geometries x 3 = 48 targets per PDK), leave-one-geometry-out (16 "
        "folds); held-out card refit on the pdk2 wide box (GF180 21 parameters, IHP 12) from the E1 fitted card. "
        "GMR with geometry-cluster bootstrap 95% CI (16 clusters, 10000 resamples, default_rng(0)).",
        f"B1 extras: GF180 {len(rg['b1_extras'])}, IHP {len(ri['b1_extras'])} (edge screen b1_screen.json). B1 and "
        "B2 max_nfev scaled with the parameter count (same TRF evaluations per IRLS round as the base box: GF180 8, "
        "IHP 14).",
        "The tier c truths contain an extrinsic series resistance of the B2 form and perturb parameters that B1 "
        "fits (next tables but one). Rule condition (b) requires GMR(M_bias, X) < 0.95 on both PDKs.",
        f"Queue jobs I4_1_gf180 06:22-06:42 ({rg['runtime_s']:.0f} s), I4_1_ihp 06:42-07:16 ({ri['runtime_s']:.0f} s).",
    ]
    return rf.Table("04_table_i4_1_sim", "I4-1 simulated GF180 and IHP tier c results for every arm", cols, rows,
                    notes)


def table_truth(d: dict) -> rf.Table:
    rows = []
    for k, misspec in (("gf180_c_logo", MISSPEC_PARAMS), ("ihp_c_logo", IHP_MISSPEC_PARAMS)):
        r = d["res"][k]
        ext = set(r["b1_extras"])
        inb = [p for p in misspec if p in ext]
        s, cmp = r["summary"], r["comparisons"]
        b2 = r["heldout_rows"]["B2"].values()
        frac = np.array([x["rext_ohm"] / x["rext_max_ohm"] for x in b2])
        kap = np.array([x["heat_kappa"] for x in b2])
        b1 = list(r["heldout_rows"]["B1"].values())
        rows.append({
            "pdk": DS_SHORT[k], "cov": f"{len(inb)}/{len(misspec)}",
            "c0": s["C0_refit"]["all_device_mean"], "b1": s["B1"]["all_device_mean"],
            "b2": s["B2"]["all_device_mean"], "m": s["M_bias"]["all_device_mean"],
            "b5": s["B5"]["all_device_mean"],
            "g1": f"{cmp['B1_vs_C0_refit']['ratio']:.3f}", "g2": f"{cmp['B2_vs_C0_refit']['ratio']:.3f}",
            "m1": f"{cmp['M_bias_vs_B1']['ratio']:.3f}", "m2": f"{cmp['M_bias_vs_B2']['ratio']:.3f}",
            "rx": f"{np.median(frac):.2f} [{frac.min():.2f}, {frac.max():.2f}]",
            "kp": f"{np.median(kap):+.3f}",
            "na": f"{min(x['n_extra_active'] for x in b1)}-{max(x['n_extra_active'] for x in b1)}"})
        if k == "gf180_c_logo":
            assert len(inb) == 9 and len(misspec) == 12
        else:
            assert len(inb) == 9 and len(misspec) == 9
    r = d["res"]["real_dev"]
    s, cmp = r["summary"], r["comparisons"]
    b2 = r["heldout_rows"]["B2"].values()
    frac = np.array([x["rext_ohm"] / x["rext_max_ohm"] for x in b2])
    kap = np.array([x["heat_kappa"] for x in b2])
    b1 = list(r["heldout_rows"]["B1"].values())
    rows.append({"pdk": "REAL_DEV (no known truth)", "cov": "—", "c0": s["C0_refit"]["all_device_mean"],
                 "b1": s["B1"]["all_device_mean"], "b2": s["B2"]["all_device_mean"],
                 "m": s["M_bias"]["all_device_mean"], "b5": s["B5"]["all_device_mean"],
                 "g1": f"{cmp['B1_vs_C0_refit']['ratio']:.3f}", "g2": f"{cmp['B2_vs_C0_refit']['ratio']:.3f}",
                 "m1": f"{cmp['M_bias_vs_B1']['ratio']:.3f}", "m2": f"{cmp['M_bias_vs_B2']['ratio']:.3f}",
                 "rx": f"{np.median(frac):.2f} [{frac.min():.2f}, {frac.max():.2f}]",
                 "kp": f"{np.median(kap):+.3f}",
                 "na": f"{min(x['n_extra_active'] for x in b1)}-{max(x['n_extra_active'] for x in b1)}"})
    cols = [{"key": "pdk", "head": "dataset", "md": "dataset", "kind": "text"},
            {"key": "cov", "head": "truth params in B1", "md": "truth-perturbed parameters in B1", "kind": "rtext"},
            {"key": "c0", "head": "C0_refit", "md": "C0_refit", "kind": "rrms", "group": "all-device / all-target mean"},
            {"key": "b1", "head": "B1", "md": "B1", "kind": "rrms", "group": "all-device / all-target mean"},
            {"key": "b2", "head": "B2", "md": "B2", "kind": "rrms", "group": "all-device / all-target mean"},
            {"key": "m", "head": "M_bias", "md": "M_bias", "kind": "rrms", "group": "all-device / all-target mean"},
            {"key": "b5", "head": "B5", "md": "B5", "kind": "rrms", "group": "all-device / all-target mean"},
            {"key": "g1", "head": "B1/C0", "md": "GMR(B1, C0_refit)", "kind": "rtext", "group": "GMR",
             "family": True},
            {"key": "g2", "head": "B2/C0", "md": "GMR(B2, C0_refit)", "kind": "rtext", "group": "GMR",
             "family": True},
            {"key": "m1", "head": "M/B1", "md": "GMR(M_bias, B1)", "kind": "rtext", "group": "GMR", "family": True},
            {"key": "m2", "head": "M/B2", "md": "GMR(M_bias, B2)", "kind": "rtext", "group": "GMR", "family": True},
            {"key": "rx", "head": "R_ext / box max", "md": "B2 fitted R_ext / box upper, median [min, max]",
             "kind": "rtext", "group": "B2 fitted"},
            {"key": "kp", "head": "kappa", "md": "B2 fitted kappa, median", "kind": "rtext", "group": "B2 fitted"},
            {"key": "na", "head": "B1 active extras", "md": "B1 active extras per device", "kind": "rtext"}]
    notes = [
        "As stated in the pre-registration before the run (arms.B1.overlap_with_simulated_truth, arms.B2.note): the "
        "GF180/IHP tier c truths perturb non-fitted parameters by exp(U(-0.25, 0.25)); GF180 B1 contains 9 of the 12 "
        "(a0, ags, lpe0, minv, drout, pscbe1, pscbe2, alpha0, beta0; not keta, etab, eu) and IHP B1 contains all 9 "
        "(mueo, cto, fbet1, lp1, alpl, alplexp, axl, ctlexp, nsubo). The truths also contain an extrinsic resistance "
        "per side k * R_side_stock / W, k ~ U(1, 2), which has the B2 form, and a per-curve temperature rise "
        "(27 C + dT_max * P_i / max P, dT_max ~ U(20, 40) K); the B2 heating term is pointwise and multiplicative, "
        "not the truth's per-curve temperature.",
        "Counts in the second column are recomputed from cryoml.pdk2.boxes.MISSPEC_PARAMS / IHP_MISSPEC_PARAMS and "
        "the stored b1_extras.",
        "B1 and B2 are local refits from the same starting card with max_nfev scaled to the parameter count; they "
        "are not global searches over the wider parameter set."
    ]
    return rf.Table("05_table_i4_1_truth_b1_b2", "I4-1 B1 and B2 against the simulated truth content (stated in "
                    "the pre-registration) and the measured B1/B2 values", cols, rows, notes)


def table_b5(d: dict) -> tuple[rf.Table, dict]:
    rows, gaps = [], {}
    for k in DS:
        r = d["res"][k]
        g = gap(r)
        gaps[k] = g
        hr = r["heldout_rows"]["B5"].values()
        st = float(np.mean([x["rrms_start_g0_insample"] for x in hr]))
        rc = float(np.mean([x["rrms_g0_on_refit_card"] for x in hr]))
        c = r["comparisons"]
        rows.append({"ds": DS_SHORT[k], "c0": g["c0"], "m": g["m"], "b5": g["b5"], "st": st, "rc": rc,
                     "fm": f"{g['frac_mean']:.3f}", "fl": f"{g['frac_log']:.3f}",
                     "gb": r_text(c["M_bias_vs_B5"]), "g5": r_text(c["B5_vs_C0_refit"])})
    cols = [{"key": "ds", "head": "dataset", "md": "dataset", "kind": "text"},
            {"key": "c0", "head": "C0_refit", "md": "C0_refit", "kind": "rrms", "group": "all-device / all-target mean"},
            {"key": "m", "head": "M_bias", "md": "M_bias", "kind": "rrms", "group": "all-device / all-target mean"},
            {"key": "b5", "head": "B5", "md": "B5 (refit card, g1)", "kind": "rrms",
             "group": "all-device / all-target mean"},
            {"key": "st", "head": "B5 start card, g0", "md": "B5 starting card with g0", "kind": "rrms",
             "group": "B5 variants (mean)", "family": True},
            {"key": "rc", "head": "B5 refit card, g0", "md": "B5 refit card with g0", "kind": "rrms",
             "group": "B5 variants (mean)", "family": True},
            {"key": "fm", "head": "of mean", "md": "fraction of C0->B5 gap closed (means)", "kind": "rtext",
             "group": "C0->B5 gap closed by M_bias"},
            {"key": "fl", "head": "of log GMR", "md": "fraction of C0->B5 gap closed (log GMR)", "kind": "rtext",
             "group": "C0->B5 gap closed by M_bias"},
            {"key": "gb", "head": "GMR(M_bias, B5) [CI]", "md": "GMR(M_bias, B5) [95% CI]", "kind": "rtext"},
            {"key": "g5", "head": "GMR(B5, C0_refit) [CI]", "md": "GMR(B5, C0_refit) [95% CI]", "kind": "rtext"}]
    notes = [
        "B5 (pre-registered, not in the rule): M_bias-form ensemble (3-16-16-1, 5 seeds) g0 trained on the held-out "
        "device's own starting-card record; held-out card refit with g0 in the loop; g1 = same recipe trained on the "
        "refit card's record; score = refit card x exp(g1). B5 uses the held-out device's measured curves to train "
        "g and is an in-sample bound for a bias-only correction of this form, not a held-out method.",
        "fraction of mean gap = (C0_refit - M_bias) / (C0_refit - B5) on all-device (REAL_DEV) or all-target "
        "(GF180/IHP) means; fraction of log GMR = ln GMR(M_bias, C0_refit) / ln GMR(B5, C0_refit).",
        "M_bias is above B5 on every device and target (0 wins on all three datasets).",
    ]
    return rf.Table("06_table_i4_1_b5_gap", "I4-1 B5 in-sample bound and the fraction of the C0_refit -> B5 gap "
                    "closed by M_bias", cols, rows, notes), gaps


# ================================================================== I4-1 figures

def fig_forest(d: dict, path: Path) -> list[str]:
    fig, axs = plt.subplots(3, 1, figsize=(11.5, 11), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for ax, k in zip(axs, DS):
        r = d["res"][k]
        n = r["n_heldout_records"]
        style(ax)
        ys = np.arange(len(FOREST))[::-1]
        for y, x in zip(ys, FOREST):
            c = r["comparisons"][f"M_bias_vs_{x}"]
            col = BLUE if x in BASE else (GREY if x == "C0_refit" else INK)
            ax.plot([c["ci_low"], c["ci_high"]], [y, y], color=col, lw=2.0)
            ax.scatter([c["ratio"]], [y], color=col, s=50, edgecolor=BG, zorder=4,
                       marker="D" if x in BASE else "o")
            ax.text(1.01, y, f"{r_text(c)}   {c['wins']}/{n}", transform=ax.get_yaxis_transform(), fontsize=8.5,
                    va="center", color=INK)
        ax.axvline(1.0, color=INK, ls="--", lw=0.9)
        if k != "real_dev":
            ax.axvline(0.95, color=RED, ls=":", lw=1.2)
            ax.text(0.95, len(FOREST) - 0.35, "0.95 (rule (b), point estimate) ", color=RED, fontsize=8, ha="right")
        else:
            ax.text(1.0, len(FOREST) - 0.35, "rule (a): CI upper bound < 1 ", color=RED, fontsize=8, ha="right")
        ax.axhline(1.5, color=GREY, lw=0.6)
        ax.set_xscale("log")
        ax.set_xlim(0.5, 2.6)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 2.5])
        ax.set_xticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1", "1.25", "1.5", "2", "2.5"])
        ax.set_yticks(ys)
        ax.set_yticklabels([f"X = {x}" + ("" if x in BASE else " (not in rule)") for x in FOREST], fontsize=9)
        ax.set_ylim(-0.7, len(FOREST))
        boot = "device bootstrap" if k == "real_dev" else "geometry-cluster bootstrap (16 clusters)"
        ax.set_title(f"{DS_LABEL[k]}; {boot}", fontsize=9.8, color=INK, loc="left")
        ax.text(1.01, len(FOREST) - 0.35, "GMR [95% CI]   M_bias < X", transform=ax.get_yaxis_transform(),
                fontsize=8.5, color=MUTED, fontweight="bold")
    axs[-1].set_xlabel("GMR(M_bias, X) = exp(mean ln(RRMS_M_bias / RRMS_X)), 95% bootstrap CI (log axis); "
                       "< 1: M_bias has the lower held-out RRMS", fontsize=9)
    fig.suptitle("I4-1: M_bias against each baseline, held-out strict paper RRMS after the card refit\n"
                 f"(rule output: {d['verdict']}; REAL_DEV is development data)", fontsize=10.5, color=INK)
    fig.subplots_adjust(left=0.2, right=0.74, top=0.92, bottom=0.07, hspace=0.34)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(RES[k]) for k in DS] + [rel(DEC)]


def fig_real_per_device(d: dict, path: Path) -> list[str]:
    r = d["res"]["real_dev"]
    per = r["per_device"]
    tags = sorted(per["M_bias"], key=rf.device_key)
    arms = ("C0_refit", "B1", "B2", "B3_d2", "B3_d3", "B4", "T1_inloop", "M_bias", "B5")
    fig, ax = plt.subplots(figsize=(13, 5.6), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    off = np.linspace(-0.32, 0.32, len(arms))
    mk = {"M_bias": "D", "B5": "v", "C0_refit": "s", "T1_inloop": "^"}
    for j, a in enumerate(arms):
        v = [per[a][t] for t in tags]
        ax.scatter(np.arange(len(tags)) + off[j], v, color=ARM_COLOR[a], marker=mk.get(a, "o"), s=26, zorder=3,
                   label=f"{a} (all-device {r['summary'][a]['all_device_mean']:.4f})")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(tags)))
    ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=8)
    for i in range(len(tags) - 1):
        ax.axvline(i + 0.5, color="#e4e3df", lw=0.6)
    ax.set_ylabel("held-out strict paper RRMS (log scale)", fontsize=9)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
    ax.set_title("I4-1 REAL_DEV per device: held-out RRMS of every arm after the card refit (12 devices; "
                 "per-device values, the rule uses the paired GMR)", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(RES["real_dev"])]


# ================================================================== I4-2

def table_i42(a: dict) -> rf.Table:
    rows = []
    for k in DS:
        z = a[k]
        for pol in ("nmos", "pmos"):
            s = z["surface_summary"][pol]
            g = z["regions"][pol]
            rows.append({"ds": DS_SHORT[k], "pol": pol, "nf": str(z["n_folds"]),
                         "mm": f"{s['mean_of_mean']:+.4f}", "ma": f"{s['max_abs_mean']:.3f}",
                         "sd": f"{s['mean_fold_std']:.4f}", "sx": f"{s['max_fold_std']:.4f}",
                         "np": f"{g['n']}",
                         "st": f"{g['subthreshold']['mean_g']:+.3f} / {g['subthreshold']['corr_indicator']:+.2f}",
                         "sa": f"{g['saturation']['mean_g']:+.3f} / {g['saturation']['corr_indicator']:+.2f}",
                         "li": f"{g['linear']['mean_g']:+.3f} / {g['linear']['corr_indicator']:+.2f}",
                         "hv": f"{g['high_vds']['mean_g']:+.3f} / {g['high_vds']['corr_indicator']:+.2f}",
                         "pr": f"{g['pearson_g_resid']:+.3f}"})
    cols = [{"key": "ds", "head": "dataset", "md": "dataset", "kind": "text"},
            {"key": "pol", "head": "pol.", "md": "polarity", "kind": "text"},
            {"key": "nf", "head": "folds", "md": "fold models", "kind": "rtext"},
            {"key": "mm", "head": "mean", "md": "grid mean of fold-mean g", "kind": "rtext",
             "group": "g surface, 41 x 41 grid"},
            {"key": "ma", "head": "max |mean|", "md": "max abs fold-mean g", "kind": "rtext",
             "group": "g surface, 41 x 41 grid"},
            {"key": "sd", "head": "mean std", "md": "grid mean of fold std", "kind": "rtext",
             "group": "g surface, 41 x 41 grid"},
            {"key": "sx", "head": "max std", "md": "max fold std", "kind": "rtext",
             "group": "g surface, 41 x 41 grid"},
            {"key": "np", "head": "points", "md": "measured points", "kind": "rtext",
             "group": "measured points: mean g / point-biserial r", "family": True},
            {"key": "st", "head": "subthreshold", "md": "subthreshold mean g / r", "kind": "rtext",
             "group": "measured points: mean g / point-biserial r", "family": True},
            {"key": "sa", "head": "saturation", "md": "saturation mean g / r", "kind": "rtext",
             "group": "measured points: mean g / point-biserial r", "family": True},
            {"key": "li", "head": "linear", "md": "linear mean g / r", "kind": "rtext",
             "group": "measured points: mean g / point-biserial r", "family": True},
            {"key": "hv", "head": "|Vds|/VDD >= 0.6", "md": "high_vds mean g / r", "kind": "rtext",
             "group": "measured points: mean g / point-biserial r", "family": True},
            {"key": "pr", "head": "r(g, ln I_meas/I_start)", "md": "Pearson g vs starting-card log residual",
             "kind": "rtext"}]
    notes = [
        "I4-2 (pre-registered, reported, not deciding): stored I3-2 fold models M_bias|<fold>, no simulation. "
        "g surface: g of each fold model on (|Vgs|/VDD, |Vds|/VDD) in [0, 1]^2, 41 x 41; mean and std over folds "
        "(fold spread).",
        "Regions on each device's included trimmed measured points with the device's own held-out fold model: Vth "
        "by maximum-transconductance linear extrapolation on the smallest-|Vds| Id-Vg curve; subthreshold |Vgs| < "
        "Vth; saturation |Vgs| >= Vth and |Vds| >= |Vgs| - Vth; linear otherwise; high_vds |Vds|/VDD >= 0.6 "
        "(overlaps the others). r = point-biserial correlation of g with the region indicator (pooled points per "
        "polarity).",
        "Last column: Pearson correlation of g with ln|I_meas / I_start| (starting-card log residual) over the "
        "same points.",
    ]
    return rf.Table("11_table_i4_2_analysis", "I4-2 g surface fold spread and correlation of g with physical "
                    "signatures", cols, rows, notes)


def table_i42_t1(a: dict) -> rf.Table:
    rows = []
    for k in DS:
        t = a[k]["t1_vs_g"]
        rows.append({"ds": DS_SHORT[k], "tag": "all curves", "n": str(t["n"]), "r": f"{t['pearson']:.3f}",
                     "mad": f"{t['mean_abs_diff']:.4f}"})
        for kind in ("idvd", "idvg"):
            rows.append({"ds": DS_SHORT[k], "tag": f"all {kind} curves", "n": str(t[kind]["n"]),
                         "r": f"{t[kind]['pearson']:.3f}", "mad": f"{t[kind]['mean_abs_diff']:.4f}"})
        # recompute the overall values from the rows
        tr = a[k]["t1_rows"]
        x = np.array([q["t1"] for q in tr])
        y = np.array([q["g_mean"] for q in tr])
        assert abs(np.corrcoef(x, y)[0, 1] - t["pearson"]) < 1e-9, k
        assert abs(np.mean(np.abs(x - y)) - t["mean_abs_diff"]) < 1e-12, k
        groups = sorted({q["group"] for q in tr})
        for grp in groups:
            sub = [q for q in tr if q["group"] == grp]
            xs = np.array([q["t1"] for q in sub])
            ys = np.array([q["g_mean"] for q in sub])
            r = np.corrcoef(xs, ys)[0, 1] if len(sub) >= 3 and xs.std() > 0 and ys.std() > 0 else np.nan
            rows.append({"ds": DS_SHORT[k], "tag": grp, "n": str(len(sub)), "t1": f"{xs.mean():+.4f}",
                         "g": f"{ys.mean():+.4f}", "r": "—" if not np.isfinite(r) else f"{r:.2f}",
                         "mad": f"{np.mean(np.abs(xs - ys)):.4f}"})
    cols = [{"key": "ds", "head": "dataset", "md": "dataset", "kind": "text"},
            {"key": "tag", "head": "curve tag (family|curve@bias)", "md": "curve tag", "kind": "text"},
            {"key": "n", "head": "n", "md": "held-out device curves", "kind": "rtext"},
            {"key": "t1", "head": "mean T1 gain", "md": "mean T1 gain", "kind": "rtext"},
            {"key": "g", "head": "mean of curve-mean g", "md": "mean of curve-mean g", "kind": "rtext"},
            {"key": "r", "head": "Pearson r", "md": "Pearson r (T1, curve-mean g)", "kind": "rtext"},
            {"key": "mad", "head": "mean |T1 - g|", "md": "mean abs(T1 - g)", "kind": "rtext"}]
    notes = [
        "Per held-out device and included curve: the fold's E1 T1 gain for (family, curve tag) "
        "(out/research/explore/E1/{real_dev_lodo,<pdk>_c_logo}/fold_corrections.json) vs the mean of the fold's "
        "M_bias g over that curve's included trimmed points. 'all' rows: stored values (recomputed from t1_rows and "
        "checked); per-tag rows: computed here from the stored t1_rows (Pearson r across held-out devices, '—' for "
        "n < 3).",
    ]
    return rf.Table("12_table_i4_2_t1_per_tag", "I4-2 comparison of the M_bias g with the T1 gain table, overall, "
                    "per curve kind and per curve tag", cols, rows, notes)


def fig_i42_t1(a: dict, path: Path) -> list[str]:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for ax, k in zip(axs, DS):
        style(ax)
        tr = a[k]["t1_rows"]
        t = a[k]["t1_vs_g"]
        for kind, col in (("idvd", ORANGE), ("idvg", BLUE)):
            for pol, mk in (("nmos", "o"), ("pmos", "^")):
                sub = [q for q in tr if q["group"].startswith(f"{pol}|{kind}")]
                ax.scatter([q["t1"] for q in sub], [q["g_mean"] for q in sub], s=14, color=col, marker=mk,
                           alpha=0.75, label=f"{pol} {kind}")
        lim = max(abs(q[v]) for q in tr for v in ("t1", "g_mean")) * 1.08
        ax.plot([-lim, lim], [-lim, lim], color=GREY, lw=0.8, ls="--")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("T1 gain of the curve tag (fold model)", fontsize=9)
        ax.set_title(f"{DS_SHORT[k]}: r = {t['pearson']:.3f} (Id-Vd {t['idvd']['pearson']:.3f}, Id-Vg "
                     f"{t['idvg']['pearson']:.3f}); n = {t['n']}", fontsize=9.3, color=INK)
    axs[0].set_ylabel("mean of the fold's M_bias g over the curve's scored points", fontsize=9)
    axs[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("I4-2: M_bias g averaged over each held-out curve vs the T1 per-tag gain of the same fold "
                 "(dashed: equality)", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I42)]


# ================================================================== F1b / F1c

def f1_load() -> dict:
    b, c = load(F1B), load(F1C)
    assert not b["smoke"] and not c["smoke"]
    sc = load(SIMCMP)
    assert sc["n_bit_identical_points"] == sc["n_points"] and sc["n_devices"] == 18 and sc["max_rel_diff"] == 0.0
    per = c["per_device"]
    nat = {t: v["native"]["rrms_native_ng41osdi"] for t, v in per.items()}
    sto = {t: v["native"]["rrms_stored"] for t, v in per.items()}
    fa, fs = fam(nat), fam(sto)
    ag = c["aggregates"]
    assert abs(fa["all"] - ag["native_ng41osdi"]["all_device_mean"]) < 1e-12
    assert abs(fa["combined"] - ag["native_ng41osdi"]["combined"]) < 1e-12
    assert abs(fs["all"] - ag["stored_M_bias"]["all_device_mean"]) < 1e-12
    return {"b": b, "c": c, "sc": sc, "nat": fa, "sto": fs}


def small_current_note(over: list) -> str:
    """Largest |I| among all simulated points (scored or not) where the SPICE-native fold current differs
    from the Python-corrected current by more than 1e-3 relative (f1c/curves.npz, not committed)."""
    p = FU / "f1c/curves.npz"
    if not p.exists():
        return ""
    z = np.load(p)
    parts = []
    for t, _ in over:
        idx = sorted({int(k.split("|")[1]) for k in z.files if k.startswith(t + "|")})
        big = []
        for i in idx:
            a, b = z[f"{t}|{i}|native_fold_ng41osdi"], z[f"{t}|{i}|py_fold_ng41osdi"]
            ok = np.abs(b) >= 1e-20
            r = np.abs(a - b)[ok] / np.abs(b)[ok]
            big += list(np.abs(b[ok])[r > 1e-3])
        parts.append(f"{t} {len(big)} points, |I| <= {max(big):.1e} A")
    return ("Points with relative difference > 1e-3 (all simulated points, from f1c/curves.npz): "
            + "; ".join(parts) + ". The largest drain currents of these devices are 1e-5 to 1e-4 A.")


def table_f1_checks(f: dict) -> rf.Table:
    b, c = f["b"]["checks"], f["c"]["checks"]

    def ok(x):
        return "fail" if x["fail"] else "pass"
    rows = [
        {"k": "F1.0", "d": "export text: g from emitted Verilog-A lines vs Python; compiles",
         "b": f"{ok(b['F1.0_export_text'])} ({b['F1.0_export_text']['max_abs_g_diff']:.1e}; OpenVAF-Reloaded v24)",
         "c": f"{ok(c['F1.0_export_text'])} ({c['F1.0_export_text']['max_abs_g_diff']:.1e}; OpenVAF 23.5.0)",
         "t": "g diff <= 1e-9"},
        {"k": "F1.1", "d": "ngspice-41 OSDI build vs canonical ngspice-41, 4 uncorrected cards, 12 devices",
         "b": f"{ok(b['F1.1_simulator_reproduction'])} (rel. {b['F1.1_simulator_reproduction']['max_rel_scored']:.1f}, "
              f"|dRRMS| {b['F1.1_simulator_reproduction']['max_abs_delta_rrms']:.1f})",
         "c": f"{ok(c['F1.1_simulator_reproduction'])} (rel. {c['F1.1_simulator_reproduction']['max_rel_scored']:.1f}, "
              f"|dRRMS| {c['F1.1_simulator_reproduction']['max_abs_delta_rrms']:.1f}; stored values diff "
              f"{c['F1.1_simulator_reproduction']['max_stored_check']:.1f})",
         "t": "rel. 1e-3, |dRRMS| 1e-4, stored 1e-9"},
        {"k": "F1.2", "d": "SPICE-native (OSDI build + .osdi) vs Python I_bsim exp(g), same simulator",
         "b": f"fail (no .osdi loaded: OSDI 0.4 file, ngspice-41 accepts 0.3; {b['F1.2_va_implementation']['n_nonfinite']} "
              "non-finite currents)",
         "c": f"{ok(c['F1.2_va_implementation'])} (max rel. fold {c['F1.2_va_implementation']['max_rel_fold_scored']:.2e}, "
              f"full {c['F1.2_va_implementation']['max_rel_full_scored']:.2e}; |dRRMS| "
              f"{c['F1.2_va_implementation']['max_abs_delta_rrms']:.2e}; 0 convergence messages)",
         "t": "rel. 1e-3, |dRRMS| 1e-4"},
        {"k": "F1.3", "d": "SPICE-native per-device RRMS vs stored I3-2 M_bias",
         "b": "fail (NaN)",
         "c": f"{ok(c['F1.3_end_to_end'])} (max device {c['F1.3_end_to_end']['max_abs_delta_device']:.2e}; "
              f"all-device {c['F1.3_end_to_end']['abs_delta_all_device_mean']:.1e})",
         "t": "device 2e-4, mean 1e-4"},
        {"k": "diag.", "d": "reltol 1e-9, abstol 1e-18: SPICE-native vs Python",
         "b": "—",
         "c": f"max rel. {c['tight_tolerance_diagnostic']['max_rel_native_vs_python_scored']:.1e}; |dRRMS| "
              f"{c['tight_tolerance_diagnostic']['max_abs_delta_rrms']:.1e}",
         "t": "no pass/fail"},
    ]
    cols = [{"key": "k", "head": "check", "md": "check", "kind": "text"},
            {"key": "d", "head": "comparison", "md": "comparison", "kind": "text"},
            {"key": "b", "head": "F1b (OpenVAF-Reloaded v24, OSDI 0.4)", "md": "F1b", "kind": "text"},
            {"key": "c", "head": "F1c (OpenVAF 23.5.0, OSDI 0.3)", "md": "F1c", "kind": "text"},
            {"key": "t", "head": "threshold", "md": "pre-registered threshold", "kind": "rtext"}]
    per = f["c"]["per_device"]
    over = [(t, v["native"]["max_rel_native_vs_python_ng41osdi_scored"]) for t, v in per.items()
            if v["native"]["max_rel_native_vs_python_ng41osdi_scored"] > 1e-3]
    notes = [
        "Amendments followup_prereg_amendment_F1b_2026-10-20.json (commit db1064b) and "
        "followup_prereg_amendment_F1c_2026-10-20.json (commit e28cd74). F1b and F1c replace ngspice-46 (used in F1) "
        "with an ngspice-41 build from the same source with --enable-osdi (prefix ~/cryo-ng41/arm64-osdi); the "
        "canonical ngspice-41 (~/cryo-ng41/arm64) is the reference for the uncorrected runs.",
        f"The OSDI build is bit-identical to the canonical build on the published card, 18 devices: "
        f"{f['sc']['n_bit_identical_points']} of {f['sc']['n_points']} currents equal, max relative difference "
        f"{f['sc']['max_rel_diff']:.1f} (out/tables/simulator_compare_ng41_arm64_vs_arm64_osdi.json).",
        "F1.2 in F1c fails on the relative current criterion only: "
        + ", ".join(f"{t} {v:.2e}" for t, v in over)
        + " exceed 1e-3; the other devices are at or below 1e-3. Relative differences use points with |I| >= "
        "1e-20 A (amendment A1). " + small_current_note(over),
        "F1b first run reported F1.3 pass with NaN deltas; the check code now counts a non-finite statistic as a "
        "failure and F1b was run again (stored summary is the second run).",
    ]
    return rf.Table("14_table_i3_2f_f1b_f1c_checks", "I3-2 follow-up F1b and F1c: SPICE-native Verilog-A checks on "
                    "ngspice-41 with OSDI", cols, rows, notes)


def table_f1c_devices(f: dict) -> rf.Table:
    per = f["c"]["per_device"]
    rows = []
    for t in sorted(per, key=rf.device_key):
        v = per[t]
        n, r = v["native"], v["rrms"]
        rows.append({"dev": t, "pub": r["published_ng41"], "c0": r["c0_ng41"], "sto": n["rrms_stored"],
                     "nat": n["rrms_native_ng41osdi"], "full": r["native_full_ng41osdi"],
                     "d": f"{n['delta_native_vs_stored']:+.1e}",
                     "rf": f"{n['max_rel_native_vs_python_ng41osdi_scored']:.1e}",
                     "rfu": f"{n['max_rel_full_native_vs_python_ng41osdi_scored']:.1e}",
                     "tt": f"{n['tight_tolerance_diagnostic']['max_rel_native_vs_python_scored']:.1e}"})
    agg = {}
    for key in ("pub", "c0", "sto", "nat", "full"):
        agg[key] = fam({row["dev"]: row[key] for row in rows})
    for lab, kk in (("all-device", "all"), ("combined", "combined")):
        rows.append({"dev": lab, **{key: agg[key][kk] for key in agg}})
    cols = [{"key": "dev", "head": "device", "md": "device", "kind": "text"},
            {"key": "pub", "head": "published card", "md": "published 77 K card (ngspice-41)", "kind": "rrms"},
            {"key": "c0", "head": "C0_refit card", "md": "C0_refit card (ngspice-41)", "kind": "rrms"},
            {"key": "sto", "head": "stored M_bias", "md": "stored M_bias (ngspice-41 x exp(g), Python)", "kind": "rrms",
             "group": "M_bias refit card x exp(g), fold model"},
            {"key": "nat", "head": "SPICE-native", "md": "SPICE-native fold (ngspice-41 OSDI)", "kind": "rrms",
             "group": "M_bias refit card x exp(g), fold model", "family": True},
            {"key": "d", "head": "native - stored", "md": "native minus stored", "kind": "rtext",
             "group": "M_bias refit card x exp(g), fold model", "family": True},
            {"key": "full", "head": "native full model", "md": "SPICE-native full 12-device model", "kind": "rrms"},
            {"key": "rf", "head": "fold", "md": "max rel. current diff, fold", "kind": "rtext",
             "group": "max rel. current diff vs Python"},
            {"key": "rfu", "head": "full", "md": "max rel. current diff, full", "kind": "rtext",
             "group": "max rel. current diff vs Python"},
            {"key": "tt", "head": "fold, reltol 1e-9", "md": "max rel. current diff, fold, tight", "kind": "rtext",
             "group": "max rel. current diff vs Python"}]
    notes = [
        "Strict paper RRMS with frozen inclusion on the 12 REAL_DEV devices; all-device = mean over 12, combined = "
        "mean of the nMOS (5) and pMOS (7) family means.",
        "Stored M_bias = I3-2 held-out value (ngspice-41, correction applied in Python). SPICE-native = M_bias refit "
        "card in the netlist with the compiled fold Verilog-A correction (trained on the other 11 devices), "
        "ngspice-41 OSDI build. The full 12-device model is in-sample on every device (diagnostic).",
        "The published-card and C0_refit-card values from the OSDI build equal the canonical ngspice-41 values "
        "(difference 0.0); the ngspice-46 differences on pmos_L0p35_W0p55 and pmos_L8_W5 seen in F1 do not occur.",
    ]
    return rf.Table("15_table_i3_2f_f1c_per_device", "I3-2 follow-up F1c per device: SPICE-native M_bias on "
                    "ngspice-41 (OSDI build) vs the stored M_bias", cols, rows, notes)


# ================================================================== main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    folder = rf.REPORT_ROOT / f"{args.date}_{TOPIC}"
    folder.mkdir(parents=True, exist_ok=True)

    d = load_i41()
    a = load(I42)
    f = f1_load()
    items, tables = [], []

    def add_table(t: rf.Table, desc: str, srcs: list[str]) -> None:
        rf.render_table(t, folder / f"{t.name}.png")
        tables.append(t)
        items.append((f"{t.name}.png", desc, srcs))

    def copy(src: Path, name: str, desc: str, srcs: list[str]) -> None:
        shutil.copyfile(src, folder / name)
        items.append((name, desc, srcs))

    i41_src = [rel(PREREG), rel(AMEND), rel(DEC)] + [rel(RES[k]) for k in DS]
    add_table(table_decision(d), "Table of the I4-1 rule per baseline (B1, B2, B3_d2, B3_d3, B4): REAL_DEV "
              "GMR(M_bias, X) with 95% CI and condition (a), GF180 and IHP GMR point estimates and condition (b), "
              f"outcome; rule output {d['verdict']}", i41_src)
    src = fig_forest(d, folder / "02_i4_1_gmr_forest.png")
    items.append(("02_i4_1_gmr_forest.png", "Forest plot of GMR(M_bias, X) with 95% bootstrap CI and wins for X in "
                  "B1, B2, B3_d2, B3_d3, B4, C0_refit and T1_inloop on REAL_DEV, GF180 tier c and IHP tier c, with "
                  "the rule thresholds", src))
    add_table(table_real(d), "Table of REAL_DEV results for every arm (C0_refit, C0_refit_rerun, B1-B4, M_bias, "
              "T1_inloop, B5): all-device, combined, nMOS, pMOS, median, failed, GMR(M_bias, X) and GMR(X, "
              "C0_refit) with CI and wins", [rel(RES["real_dev"])])
    add_table(table_sim(d), "Table of GF180 and IHP tier c results for every arm: all-target mean, median, failed, "
              "GMR(M_bias, X) and GMR(X, C0_refit) with geometry-cluster CI and wins",
              [rel(RES["gf180_c_logo"]), rel(RES["ihp_c_logo"])])
    add_table(table_truth(d), "Table of the pre-registered statement on the tier c truth content (B1 covers 9/12 "
              "GF180 and 9/9 IHP truth-perturbed parameters; the truth has an extrinsic resistance of the B2 form) "
              "next to the measured B1 and B2 means, GMR values, fitted B2 resistance and heating coefficient",
              [rel(PREREG), rel(SCREEN)] + [rel(RES[k]) for k in DS])
    tb5, gaps = table_b5(d)
    add_table(tb5, "Table of the B5 in-sample bound per dataset, B5 variants, and the fraction of the C0_refit -> "
              "B5 gap closed by M_bias (on means and on log GMR)", [rel(RES[k]) for k in DS] + [rel(DEC)])
    src = fig_real_per_device(d, folder / "07_i4_1_real_per_device.png")
    items.append(("07_i4_1_real_per_device.png", "Per-device held-out RRMS on REAL_DEV for C0_refit, B1, B2, B3_d2, "
                  "B3_d3, B4, T1_inloop, M_bias and B5 (log scale)", src))
    for j, k in enumerate(DS):
        copy(I41 / f"i4_2_g_surface_{k}.png", f"{8 + j:02d}_i4_2_g_surface_{k}.png",
             f"I4-2 M_bias g surface on {DS_SHORT[k]}: mean and standard deviation of g over the "
             f"{a[k]['n_folds']} fold models on the (|Vds|/VDD, |Vgs|/VDD) grid, nMOS and pMOS (copied from the "
             "analysis output)", [rel(I41 / f"i4_2_g_surface_{k}.png"), rel(I42)])
    t42 = table_i42(a)
    add_table(t42, "Table of the I4-2 quantities per dataset and polarity: g surface mean, max |mean|, fold spread "
              "(mean and max std), mean g and point-biserial correlation per bias region, correlation of g with "
              "the starting-card log residual", [rel(I42)])
    t42b = table_i42_t1(a)
    add_table(t42b, "Table of the comparison of curve-mean M_bias g with the T1 gain: Pearson r and mean |T1 - g| "
              "overall, per curve kind and per curve tag, per dataset", [rel(I42)])
    src = fig_i42_t1(a, folder / "13_i4_2_t1_vs_g.png")
    items.append(("13_i4_2_t1_vs_g.png", "Scatter of the T1 per-tag gain vs the curve-mean M_bias g for every "
                  "held-out device curve, per dataset, colored by curve kind", src))
    f1_src = [rel(AM_B), rel(AM_C), rel(F1B), rel(F1C), rel(SIMCMP)]
    tf = table_f1_checks(f)
    add_table(tf, "Table of the F1b and F1c pre-registered checks F1.0-F1.3 with values, thresholds and outcomes "
              "(F1c: F1.0 pass, F1.1 pass, F1.2 fail on small-current relative differences, F1.3 pass)", f1_src)
    tc = table_f1c_devices(f)
    add_table(tc, "Table of F1c per device: published and C0_refit card RRMS, stored M_bias, SPICE-native fold "
              "and full-model RRMS on the ngspice-41 OSDI build, difference, max relative current differences; "
              "all-device and combined rows", [rel(F1C)])
    for j, t in enumerate(IV_DEVS):
        p = FU / f"f1c_iv_{t}.png"
        copy(p, f"{16 + j:02d}_i3_2f_f1c_iv_{t}.png",
             f"I-V of {t} (F1c): measured points, NGSpice curves of the published 77 K card and the C0_refit card "
             "(canonical ngspice-41) and the SPICE-native corrected M_bias refit card (ngspice-41 OSDI build with "
             "the compiled fold Verilog-A); copied from the F1c output", [rel(p), rel(F1C)])

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i4.py --date {args.date}`"
    md = [f"# Tables, {args.date} {TOPIC}", "",
          f"Generated {now} by {gen} from stored files only. RRMS is the paper metric with frozen curve inclusion, "
          "strict. REAL_DEV values are on the 12 measured REAL_DEV devices (development evidence); GF180/IHP values "
          "are on simulated tier c targets.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))

    res = d["res"]
    rc = res["real_dev"]["comparisons"]
    readme = [f"# Report batch {args.date} · {TOPIC}", "",
              f"Generated {now} by {gen} from stored files only (no NGSpice, no training).", "",
              "## Contents", "",
              "- I4-1: baselines for the I3-2 bias-only correction M_bias (B1 wider card parameter set, B2 extrinsic "
              "series resistance + self-heating factor, B3_d2/B3_d3 shared polynomial g, B4 shared GP discrepancy, "
              "B5 per-device in-sample reference). Pre-registration "
              f"`{rel(PREREG)}` (commit e57afce), amendment `{rel(AMEND)}`; code and smoke commit 41ca545. Queue "
              "jobs I4_1_real (2026-10-20 06:14-06:22), I4_1_gf180 (06:22-06:42), I4_1_ihp (06:42-07:16), "
              f"I4_1_decide (07:16). Rule output: {d['verdict']} (development set). Every baseline is beaten "
              "through the REAL_DEV condition (a); the simulated condition (b) is not met for any baseline.",
              "- I4-2: interpretation analysis of the stored I3-2 fold models (reported, not deciding). Queue job "
              "I4_2_analyze (07:16).",
              "- I3-2 follow-up F1b/F1c: SPICE-native Verilog-A runs on an OSDI-enabled ngspice-41 build "
              f"(amendments `{rel(AM_B)}`, commit db1064b; `{rel(AM_C)}`, commit e28cd74; results commit 3713710). "
              "The exporter fix (commit 5bec551): `to_verilog_a_bias` emits `V(sense) <+ RS * I(sense);` with "
              "`parameter real RS = 0.0;` and names the polarity parameter `is_pmos`; corrected export "
              "`out/research/explore/I3_2/real_dev/m_bias_sky130_dev_full_sense_rs.va`; the stored "
              "`m_bias_sky130_dev_full.va` is unchanged.",
              "- F1 (ngspice-46) I-V outputs, committed with this batch and not part of the numbered figures: "
              "`out/research/explore/I3_2/followup/f1_iv_{pmos_L0p35_W0p55,pmos_L2_W5,nmos_L20_W0p64}.png`. These "
              "are the F1 run (2026-10-20 02:58-02:59), whose SPICE-native curves come from Homebrew ngspice-46; on "
              "pmos_L0p35_W0p55 ngspice-46 differs from ngspice-41 for the uncorrected cards (F1.1 fail). They are "
              "superseded by the F1c figures 16-18.",
              "", "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))

    s = {k: res[k]["summary"] for k in DS}
    z = {k: a[k] for k in DS}
    summary = [
        f"I4-1 rule output: {d['verdict']} (development set). REAL_DEV all-device (combined): C0_refit "
        f"{s['real_dev']['C0_refit']['all_device_mean']:.4f} ({s['real_dev']['C0_refit']['combined']:.4f}), B1 "
        f"{s['real_dev']['B1']['all_device_mean']:.4f}, B2 {s['real_dev']['B2']['all_device_mean']:.4f}, B3_d2 "
        f"{s['real_dev']['B3_d2']['all_device_mean']:.4f}, B3_d3 {s['real_dev']['B3_d3']['all_device_mean']:.4f}, "
        f"B4 {s['real_dev']['B4']['all_device_mean']:.4f}, M_bias {s['real_dev']['M_bias']['all_device_mean']:.4f} "
        f"({s['real_dev']['M_bias']['combined']:.4f}), B5 {s['real_dev']['B5']['all_device_mean']:.4f}. "
        "GMR(M_bias, X) CI upper < 1 for every X: " + ", ".join(
            f"{x} {r_text(rc['M_bias_vs_' + x])}" for x in BASE) + ".",
        "Simulated condition (b) not met for any baseline: GMR(M_bias, X) GF180 / IHP " + ", ".join(
            f"{x} {d['dec']['per_baseline'][x]['gf180_ratio']:.3f} / {d['dec']['per_baseline'][x]['ihp_ratio']:.3f}"
            for x in BASE) + f". All-target means GF180 / IHP: B2 {s['gf180_c_logo']['B2']['all_device_mean']:.4f} / "
        f"{s['ihp_c_logo']['B2']['all_device_mean']:.4f}, B1 {s['gf180_c_logo']['B1']['all_device_mean']:.4f} / "
        f"{s['ihp_c_logo']['B1']['all_device_mean']:.4f}, M_bias {s['gf180_c_logo']['M_bias']['all_device_mean']:.4f}"
        f" / {s['ihp_c_logo']['M_bias']['all_device_mean']:.4f} (tier c truth contains the B2 resistance form and "
        "B1 covers 9/12 and 9/9 truth-perturbed parameters, stated in the pre-registration).",
        "B5 in-sample bound and fraction of the C0_refit -> B5 gap closed by M_bias (means / log GMR): " + "; ".join(
            f"{DS_SHORT[k]} B5 {gaps[k]['b5']:.4f}, {gaps[k]['frac_mean']:.3f} / {gaps[k]['frac_log']:.3f}"
            for k in DS) + ".",
        f"I4-2: fold spread of g (mean std) REAL_DEV nMOS {z['real_dev']['surface_summary']['nmos']['mean_fold_std']:.4f}, "
        f"pMOS {z['real_dev']['surface_summary']['pmos']['mean_fold_std']:.4f}; curve-mean g vs T1 Pearson r "
        f"REAL_DEV {z['real_dev']['t1_vs_g']['pearson']:.3f}, GF180 {z['gf180_c_logo']['t1_vs_g']['pearson']:.3f}, "
        f"IHP {z['ihp_c_logo']['t1_vs_g']['pearson']:.3f}.",
        f"F1c (ngspice-41 OSDI build, OpenVAF 23.5.0): F1.0 pass, F1.1 pass, F1.2 fail (max relative current "
        f"difference {f['c']['checks']['F1.2_va_implementation']['max_rel_fold_scored']:.2e} at small currents), "
        f"F1.3 pass; SPICE-native all-device (combined) {f['nat']['all']:.6f} ({f['nat']['combined']:.6f}) vs stored "
        f"{f['sto']['all']:.6f} ({f['sto']['combined']:.6f}). F1b: no .osdi loaded (OSDI 0.4 vs 0.3).",
    ]
    rf.update_index(args.date, TOPIC, folder, [i[0] for i in items], summary)
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
