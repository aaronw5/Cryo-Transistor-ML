#!/usr/bin/env python3
"""Report batch for exploration iteration 2: I2-0 (diagnosis of the devices
that determined the iteration-1 means), I2-1 (correction trained jointly with
the training devices' cards), E3b (E3 rerun with the BSIM4 u0 unit-switch fix)
and the finished temperature surrogate table (out/research/temp/table.json).

Reads result JSON files only (no simulation, no training) and writes PNGs,
``tables.md`` and ``README.md`` to
``out/research/report/<date>_explore_i2_e3b/``, then adds the batch to
``out/research/report/index.json`` and ``INDEX.md``. The I2-0 PNGs are copied
from ``out/research/explore/I2_0/`` under numbered names.

Rule outputs are recomputed from the stored values and checked against the
stored decisions (I2-1 criterion block, E3b summary.json).

  PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i2_e3b.py \
      --date 2026-10-20
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_explore_e2_e3_e4 as pe  # noqa: E402  (helpers, style)
from plot_explore_e2_e3_e4 import (BG, INK, MUTED, GREY, BLUE, ORANGE, AQUA,  # noqa: E402
                                   VIOLET, RED, DPI, EXP, ROOT, fam, fnum, load, real_dev,
                                   rel, style, yn)
import e3_temp_misspec as e3m  # noqa: E402  (bootstrap functions only)

rf = pe.rf
plt = pe.plt

I20 = EXP / "I2_0"
I21 = EXP / "I2_1"
I21_RUNS = {"real": I21 / "real_dev", "gf180": I21 / "gf180_c_logo", "ihp": I21 / "ihp_c_logo"}
PDK_LABEL = {"gf180": "GF180 (BSIM4) tier c", "ihp": "IHP SG13G2 (PSP103) tier c"}
E3 = EXP / "E3" / "runs/full"
E3B = EXP / "E3b" / "runs/full"
TEMP = ROOT / "out/research/temp"
HELD = ("110", "190", "260")
TIERS = ("M", "W")
SETS = ("O4", "O2")
CONTROL = {"O4": "par3", "O2": "par3_pop"}
E3_OUTLIER = "nmos_L1_W3__2"

ARMS = ["C0_refit", "M_post", "M_joint", "T1_joint", "ORACLE_joint"]
ARM_LABEL = {"C0_refit": "C0 refit (no correction)",
             "M_post": "M_post (E1 stage-2 correction, K = 0)",
             "M_joint": "M_joint (primary, K = 3)",
             "T1_joint": "T1_joint (per-tag gains, not deployable)",
             "ORACLE_joint": "ORACLE_joint (in-sample, not a held-out result)"}
ARM_COLOR = {"C0_refit": INK, "M_post": GREY, "M_joint": BLUE, "T1_joint": ORANGE,
             "ORACLE_joint": AQUA}
ARM_MARK = {"C0_refit": "o", "M_post": "s", "M_joint": "D", "T1_joint": "v",
            "ORACLE_joint": "^"}


def f3(x) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.3f}"


def ci(c) -> str:
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


# ================================================================== I2-0

def table_i2_0_real(sens: dict) -> rf.Table:
    keys = ["all", "without_pmos_L0p35_W5", "without_pmos_L0p35_W0p55", "without_both"]
    rows = []

    def add(run, quantity, vals):
        rows.append({"run": run, "q": quantity, **{k: v for k, v in zip(keys, vals)}})

    for st in ("stage1", "stage2"):
        e = [sens["E1"][k][st] for k in keys]
        add(f"E1 {st.replace('stage', 'stage ')}", "M / T1 / C0 all-device",
            [f"{x['M']:.4f} / {x['T1']:.4f} / {x['C0']:.4f}" for x in e])
        add("", "M < T1 / M < C0 (devices); pass",
            [f"{x['M_beats_T1']}/{x['n']}, {x['M_beats_C0']}/{x['n']}; {yn(x['pass'])}" for x in e])
    e = [sens["E2"][k] for k in keys]
    add("E2", "R_raw N0 / M / S (penalized)",
        [f"{x['R_raw_N0']:.2f} / {x['R_raw_M']:.2f} / {x['R_raw_S']:.2f}" for x in e])
    add("", "S / N0; S wins; tier-a guard C4",
        [f"{x['ratio_S_over_N0']:.3f}; {x['S_wins']}/{x['n_devices']}; "
         f"{yn(x['checks']['C4_guard_tier_a_same_exclusion'])}" for x in e])
    add("", "rule output", [x["rule_output"] for x in e])
    e = [sens["E4_p43_vd"][k] for k in keys]
    add("E4 p43 vd", "single / Laplace / learned / crossinterp",
        [f"{x['heldout_means']['single']:.3f} / {x['heldout_means']['laplace']:.2f} / "
         f"{x['heldout_means']['learned']:.3f} / {x['heldout_means']['crossinterp']:.3f}" for x in e])
    add("", "learned: coverage 0.9; wins; C1 C2 C3",
        [f"{x['decision']['per_method']['learned']['coverage_0.9']:.3f}; "
         f"{x['decision']['per_method']['learned']['wins']}/{x['n_devices']}; "
         f"{yn(x['decision']['per_method']['learned']['C1'])} "
         f"{yn(x['decision']['per_method']['learned']['C2'])} "
         f"{yn(x['decision']['per_method']['learned']['C3'])}" for x in e])
    add("", "rule output", [x["decision"]["verdict"] for x in e])
    e = [sens["E5"][k]["b43"] for k in keys]
    add("E5 b43", "A0 / A2 all-device", [f"{x['A0']:.4f} / {x['A2']:.4f}" for x in e])
    add("", "reduction; A2 < A0", [f"{x['reduction']:.3f}; {x['A2_lt_A0_on']}/{x['n']}" for x in e])
    cols = [
        {"key": "run", "head": "run", "md": "run", "kind": "text"},
        {"key": "q", "head": "quantity", "md": "quantity", "kind": "text"},
        {"key": "all", "head": "all 12", "group": "REAL_DEV devices included", "md": "all 12 devices", "kind": "rtext"},
        {"key": "without_pmos_L0p35_W5", "head": "without W5", "group": "REAL_DEV devices included", "md": "without pmos_L0p35_W5", "kind": "rtext"},
        {"key": "without_pmos_L0p35_W0p55", "head": "without W0p55", "group": "REAL_DEV devices included", "md": "without pmos_L0p35_W0p55", "kind": "rtext"},
        {"key": "without_both", "head": "without both", "group": "REAL_DEV devices included", "md": "without both", "kind": "rtext"},
    ]
    notes = [
        "DIAGNOSTIC ONLY. Decision quantities recomputed after the fact with devices removed. The "
        "stored pre-registered verdicts are unchanged; these values are not decision inputs.",
        "W5 = pmos_L0p35_W5, W0p55 = pmos_L0p35_W0p55: the two REAL_DEV devices whose frozen "
        "inclusion set holds curves with denominators 1e-4 to 5e-4 of the device's largest "
        "(I2-0 part A).",
        "E2 and E4 are exact on the reduced set (E2 real devices are evaluation-only; E4 learned and "
        "conformal leave-one-device-out recomputed). E1 and E5 are aggregate-only: their fold "
        "models were trained with the removed devices in the other folds. Win thresholds scale "
        "with n.",
        "Source: out/research/explore/I2_0/C_decision_sensitivity.json (script "
        "scripts/research/explore/i2_0_diagnose.py, stage decisions).",
    ]
    return rf.Table("01_table_i2_0_real_sensitivity",
                    "I2-0 (diagnostic only): iteration-1 REAL_DEV decision quantities with and "
                    "without the two low-denominator pMOS devices", cols, rows, notes)


def table_i2_0_e3(sens: dict) -> rf.Table:
    keys = ["all", "without_nmos_L1_W3__2", "without_all_nmos_L1_W3_test"]
    rows = []
    for S in SETS:
        e = [sens["E3"][k][S] for k in keys]
        rows.append({"S": S, "q": f"lrn / {e[0]['control']} / pt_interp all-device",
                     **{k: f"{x['lrn']:.3f} / {x['control_mean']:.3f} / {x['pt_interp']:.3f}"
                        for k, x in zip(keys, e)}})
        rows.append({"S": "", "q": f"R_{S} (95% CI)",
                     **{k: f"{x['R_S']:.3f} {ci(x['R_S_CI95'])}" for k, x in zip(keys, e)}})
        rows.append({"S": "", "q": "1 − lrn / pt_interp; lrn < control",
                     **{k: f"{x['lrn_vs_pt_interp_reduction']:.3f}; "
                           f"{x['lrn_lower_than_control_on']}/{x['n_devices']}"
                        for k, x in zip(keys, e)}})
    rows.append({"S": "", "q": "rule output (stored run: KILL)",
                 **{k: sens["E3"][k]["rule_output_without_failed_cell_check"] for k in keys}})
    cols = [
        {"key": "S", "head": "S", "md": "S", "kind": "text"},
        {"key": "q", "head": "quantity", "md": "quantity", "kind": "text"},
        {"key": "all", "head": "all 36", "group": "E3 tier-M test devices included", "md": "all 36", "kind": "rtext"},
        {"key": "without_nmos_L1_W3__2", "head": "without nmos_L1_W3__2 (35)", "group": "E3 tier-M test devices included", "md": "without nmos_L1_W3__2 (35)", "kind": "rtext"},
        {"key": "without_all_nmos_L1_W3_test", "head": "without 3 nmos_L1_W3 (33)", "group": "E3 tier-M test devices included", "md": "without the 3 nmos_L1_W3 test devices (33)", "kind": "rtext"},
    ]
    notes = [
        "DIAGNOSTIC ONLY. The stored E3 verdict (KILL) is unchanged. Removing a test device does "
        "not remove the switched truths from the E3 training population.",
        "nmos_L1_W3__2: truth u0 is above 1 (BSIM4 unit switch, u0 read as cm^2/Vs and divided by "
        "1e4) at 77-110 K; every method's 110 K card is below the switch and gives RRMS 676-1060 in "
        "that cell (I2-0 part B).",
        "Paired bootstrap 4000 resamples, seed 0 (the stored run used its own seed, so CI bounds "
        "differ slightly). The failed-cell check is not recomputed (0 failed cells in the stored "
        "run).",
        "This diagnostic was computed before the E3b pre-registration; E3b copies the E3 rule "
        "without change and does not use these values.",
    ]
    return rf.Table("02_table_i2_0_e3_sensitivity",
                    "I2-0 (diagnostic only): E3 tier-M decision quantities with and without the "
                    "test device whose truth crosses the u0 unit switch", cols, rows, notes)


I20_COPIES = [
    ("03_i2_0_denominators.png", "I2_0_A_denominators.png",
     "I2-0 part A: per-curve denominator divided by the device's largest denominator for every "
     "frozen-included REAL_DEV curve (pmos_L0p35_W5 and pmos_L0p35_W0p55 reach 1e-4 to 5e-4; "
     "other devices are within 37-342x)"),
    ("04_i2_0_pmos_L0p35_W5_iv.png", "I2_0_A_pmos_L0p35_W5_iv.png",
     "I2-0 part A: pmos_L0p35_W5 measured I-V (log scale) with NGSpice curves of the published "
     "card, the best stored 43-parameter card (direct43_r3_fd) and the worst stored prediction "
     "(E2 raw S seed 1), trimmed regions shaded and denominators dotted; device RRMS 0.329 / "
     "0.112 / 183.7"),
    ("05_i2_0_pmos_L0p35_W0p55_iv.png", "I2_0_A_pmos_L0p35_W0p55_iv.png",
     "I2-0 part A: pmos_L0p35_W0p55 measured I-V with the same three NGSpice curves; Id-Vd at "
     "|Vg| 0.37 and 0.74 V are at the SMU resolution (8 and 3 distinct values)"),
    ("06_i2_0_vth0_sensitivity.png", "I2_0_A_vth0_sensitivity.png",
     "I2-0 part A: per-curve RRMS of the best 43-parameter card vs a VTH0 shift of -40 to +40 mV "
     "for the two pMOS devices"),
    ("07_i2_0_nmos_L1_W3__2.png", "I2_0_B_nmos_L1_W3__2.png",
     "I2-0 part B: E3 tier-M test device nmos_L1_W3__2: truth u0 vs temperature with the u0 = 1 "
     "switch, truth current scale vs temperature, RRMS at 110 K when only x_u0 is varied, and "
     "the 110 K truth curves with NGSpice curves of the O4 lrn, par3, floor and lrn-with-truth-u0 "
     "cards"),
]


# ================================================================== I2-1

def i21_load() -> dict:
    return {k: load(p / "results.json") for k, p in I21_RUNS.items()}


def table_i2_1_decision(r: dict) -> rf.Table:
    s, w = r["summary"], r["wins"]
    mj, mp, t1 = (s[k]["all_device_mean"] for k in ("M_joint", "M_post", "T1_joint"))
    pd_ = r["per_device"]
    wins = sum(pd_["M_joint"][t] < pd_["M_post"][t] for t in pd_["M_joint"])
    assert wins == w["M_joint_beats_M_post"]
    checks = [
        ("M_joint all-device mean <= 0.080", f"{mj:.4f}", mj <= 0.080,
         "M_joint_mean_le_0.080"),
        ("M_joint < M_post on >= 8 of 12 devices", f"{wins}/12", wins >= 8,
         "M_joint_beats_M_post_on_ge_8_of_12"),
        ("M_joint mean <= T1_joint mean + 0.005", f"{mj:.4f} vs {t1 + 0.005:.4f}",
         mj <= t1 + 0.005, "M_joint_mean_le_T1_joint_mean_plus_0.005"),
        ("0 failed devices for M_joint", f"{s['M_joint']['n_failed']}/12",
         s["M_joint"]["n_failed"] == 0, "M_joint_zero_failed"),
    ]
    rows = []
    for text, val, ok, key in checks:
        assert ok == r["criterion"]["checks"][key]
        rows.append({"c": text, "v": val, "ok": yn(ok)})
    verdict = "CONTINUE" if all(c[2] for c in checks) else "KILL"
    assert verdict.lower() == r["criterion"]["decision"]
    rows.append({"c": "rule output (continue only if all four hold)", "v": "", "ok": verdict})
    cols = [
        {"key": "c", "head": "pre-registered condition", "md": "condition", "kind": "text"},
        {"key": "v", "head": "value", "md": "value", "kind": "rtext"},
        {"key": "ok", "head": "met", "md": "met", "kind": "rtext"},
    ]
    notes = [
        "Pre-registration: out/research/explore/I2_1/prereg.json (written before any joint-arm "
        "result). REAL_DEV leave-one-device-out, 12 folds, K = 3 alternating rounds (training "
        "cards refit with g_{r-1} in the loop, correction retrained from scratch on the refit "
        "residuals); held-out card refit from its starting card with the frozen g_3 (TRF, "
        "max_nfev 400, irls 2, diff_step 0.01).",
        f"Reference values: M_post {mp:.4f} (E1 stage 2, stored), C0_refit "
        f"{s['C0_refit']['all_device_mean']:.4f} (rerun; max abs difference to E1 stage 2 = "
        f"{r['C0_refit_vs_E1_stage2']['max_abs_diff']:g}), T1_joint {t1:.4f}.",
        "Metric: strict paper RRMS with frozen published-baseline inclusion of the NGSpice "
        "simulation of the refit card multiplied by exp(g); a non-finite included curve makes "
        "the device NaN (0 failed devices in every arm).",
        f"Queue job i2_1_real, 2026-10-19 17:59-18:35 ({r['runtime_s']:.0f} s).",
    ]
    return rf.Table("08_table_i2_1_decision",
                    "I2-1 decision (REAL_DEV): correction trained jointly with the training "
                    "devices' cards, pre-registered checks", cols, rows, notes)


def table_i2_1_results(r: dict) -> rf.Table:
    rows = []
    pd_ = r["per_device"]
    arms = ARMS[:4] + ["M_joint_r1", "M_joint_r2", "ORACLE_joint"]
    for a in arms:
        f = fam(pd_[a])
        st = r["summary"][a]
        assert abs(f["all"] - st["all_device_mean"]) < 1e-12
        assert abs(f["combined"] - st["combined"]) < 1e-12
        role = {"C0_refit": "control", "M_post": "reference (E1 stage 2)", "M_joint": "primary",
                "T1_joint": "comparator, not deployable",
                "M_joint_r1": "diagnostic: held-out refit with g_1",
                "M_joint_r2": "diagnostic: held-out refit with g_2",
                "ORACLE_joint": "diagnostic, in-sample"}[a]
        rows.append({"arm": a, "role": role, "all": f["all"], "combined": f["combined"],
                     "nmos": f["nmos"], "pmos": f["pmos"], "median": f["median"],
                     "c0": None if a == "C0_refit" else
                     f"{sum(pd_[a][t] < pd_['C0_refit'][t] for t in pd_[a])}/12",
                     "mp": None if a in ("M_post", "C0_refit") else
                     f"{sum(pd_[a][t] < pd_['M_post'][t] for t in pd_[a])}/12",
                     "failed": f"{st['n_failed']}/{st['n']}"})
    cols = [
        {"key": "arm", "head": "arm", "md": "arm", "kind": "text"},
        {"key": "role", "head": "role", "md": "role", "kind": "text"},
        {"key": "all", "head": "all-device", "group": "RRMS, 12 REAL_DEV devices", "md": "all-device", "kind": "rrms"},
        {"key": "combined", "head": "combined", "group": "RRMS, 12 REAL_DEV devices", "md": "combined", "kind": "rrms"},
        {"key": "nmos", "head": "nMOS", "group": "RRMS, 12 REAL_DEV devices", "md": "nMOS", "kind": "rrms", "family": True},
        {"key": "pmos", "head": "pMOS", "group": "RRMS, 12 REAL_DEV devices", "md": "pMOS", "kind": "rrms", "family": True},
        {"key": "median", "head": "median", "group": "RRMS, 12 REAL_DEV devices", "md": "median", "kind": "rrms"},
        {"key": "c0", "head": "< C0", "md": "devices below C0_refit", "kind": "rtext"},
        {"key": "mp", "head": "< M_post", "md": "devices below M_post", "kind": "rtext"},
        {"key": "failed", "head": "failed", "md": "failed devices", "kind": "rtext"},
    ]
    notes = [
        "all-device = mean over the 12 REAL_DEV devices; combined = mean of the nMOS (5) and pMOS "
        "(7) family means; strict frozen-inclusion paper RRMS, 0 failed devices in every arm.",
        "Every arm except M_post refits the held-out card from the same starting card "
        "(direct43_r3_fd) with max_nfev 400, irls 2; M_post values are the stored E1 stage-2 "
        "results (equivalent to K = 0).",
        "ORACLE_joint runs the joint loop on the held-out device itself (in-sample); it is an "
        "upper bound for the network class, not a held-out result.",
    ]
    return rf.Table("09_table_i2_1_results",
                    "I2-1 REAL_DEV RRMS per arm, with the held-out refits at intermediate rounds",
                    cols, rows, notes)


def fig_i2_1_per_device(r: dict, path: Path) -> list[str]:
    tags = real_dev()
    pd_ = r["per_device"]
    fig = plt.figure(figsize=(13, 12), dpi=DPI)
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0, :])
    style(ax)
    x = np.arange(len(tags))
    off = dict(zip(ARMS, np.linspace(-0.28, 0.28, len(ARMS))))
    for a in ARMS:
        v = pd_[a]
        f = fam(v)
        extra = "" if a == "C0_refit" else f", < C0 on {sum(v[t] < pd_['C0_refit'][t] for t in tags)}/12"
        ax.scatter(x + off[a], [v[t] for t in tags], marker=ARM_MARK[a], s=34, color=ARM_COLOR[a],
                   edgecolor=BG, lw=0.5, zorder=3,
                   label=f"{ARM_LABEL[a]}: all-device {f['all']:.4f}, combined {f['combined']:.4f}{extra}")
    ax.set_yscale("log")
    ax.set_ylim(0.007, 0.6)
    nm = sum(t.startswith("nmos") for t in tags)
    ax.axvline(nm - 0.5, color=GREY, lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([rf.device_label(t) + (" n" if t.startswith("nmos") else " p") for t in tags],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("paper RRMS, held-out device", fontsize=9, color=INK)
    ax.legend(fontsize=7.6, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2,
              title="per device and arm (n = nMOS, p = pMOS; 0 failed devices)",
              title_fontsize=8.5)
    for k, (xa, ya, title) in enumerate((("M_post", "M_joint", "M_joint vs M_post (primary pair)"),
                                         ("C0_refit", "T1_joint", "T1_joint vs C0_refit"))):
        ax = fig.add_subplot(gs[1, k])
        style(ax)
        xv = np.array([pd_[xa][t] for t in tags])
        yv = np.array([pd_[ya][t] for t in tags])
        colors = [BLUE if t.startswith("nmos") else ORANGE for t in tags]
        ax.scatter(xv, yv, c=colors, s=30, edgecolor=BG, lw=0.6, zorder=3)
        lo, hi = min(xv.min(), yv.min()) * 0.7, max(xv.max(), yv.max()) * 1.4
        ax.plot([lo, hi], [lo, hi], color=GREY, ls="--", lw=0.9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        for t, a_, b_ in zip(tags, xv, yv):
            ax.annotate(rf.device_label(t), (a_, b_), fontsize=6.5, color=MUTED,
                        xytext=(4, -9), textcoords="offset points")
        wins = int(np.sum(yv < xv))
        ax.text(0.02, 0.98, f"{ya} < {xa} on {wins}/12\nmeans {yv.mean():.4f} vs {xv.mean():.4f}",
                transform=ax.transAxes, va="top", fontsize=8, color=INK,
                bbox=dict(facecolor=BG, edgecolor="none", alpha=0.85))
        ax.set_xlabel(f"{xa}: held-out RRMS", fontsize=9, color=INK)
        ax.set_ylabel(f"{ya}: held-out RRMS", fontsize=9, color=INK)
        ax.set_title(title + " (below the dashed line: y lower)", fontsize=9.5, color=INK)
    fig.suptitle("I2-1, REAL_DEV leave-one-device-out: held-out RRMS after the refit with the frozen "
                 "correction (blue points nMOS, orange pMOS)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I21_RUNS["real"] / "results.json")]


def rounds_means(r: dict, arm: str) -> dict:
    keys = [k for k in r["rounds"] if k.startswith(arm + "|")]
    K = len(r["rounds"][keys[0]])
    out = {"corr": [], "prev": [], "unc": [], "post": []}
    for i in range(K):
        rows = [r["rounds"][k][i] for k in keys]
        out["corr"].append(float(np.mean([x["train_rrms_corrected"] for x in rows])))
        out["unc"].append(float(np.mean([x["train_rrms_uncorrected"] for x in rows])))
        out["prev"].append(float(np.mean([x["train_rrms_prev_corr_on_refit_cards"] for x in rows]))
                           if i else math.nan)
        if "heldout_posthoc" in rows[0]:
            out["post"].append(float(np.mean([v for x in rows for v in x["heldout_posthoc"].values()])))
    return out


def fig_i2_1_rounds(res: dict, path: Path) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), dpi=DPI)
    fig.patch.set_facecolor(BG)
    r = res["real"]
    s = r["summary"]
    ax = axes[0]
    style(ax)
    rr = np.arange(4)
    for arm, c in (("M", BLUE), ("T1", ORANGE)):
        m = rounds_means(r, arm)
        ax.plot(rr, m["corr"], "-o", color=c, ms=4, label=f"{arm}: training devices, g_r on c_r")
        ax.plot(rr[1:], m["prev"][1:], ":s", color=c, ms=4,
                label=f"{arm}: training devices, g_(r-1) on c_r")
        ax.plot(rr, m["post"], "--^", color=c, ms=4, alpha=0.8,
                label=f"{arm}: held-out, g_r on the starting card (no refit)")
    ax.set_xticks(rr)
    ax.set_xlabel("round r", fontsize=9)
    ax.set_ylabel("mean paper RRMS over folds", fontsize=9)
    ax.set_title("REAL_DEV: training-device and post-hoc held-out RRMS per round", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    style(ax)
    hv = [s["M_post"]["all_device_mean"], s["M_joint_r1"]["all_device_mean"],
          s["M_joint_r2"]["all_device_mean"], s["M_joint"]["all_device_mean"]]
    ax.plot(rr, hv, "-D", color=BLUE, ms=5, label="M: held-out refit with g_r (r = 0 is M_post)")
    for xi, v in zip(rr, hv):
        ax.annotate(f"{v:.4f}", (xi, v), xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=7.5, color=BLUE)
    ax.axhline(s["C0_refit"]["all_device_mean"], color=INK, lw=1, ls="--",
               label=f"C0_refit {s['C0_refit']['all_device_mean']:.4f}")
    ax.axhline(s["T1_joint"]["all_device_mean"], color=ORANGE, lw=1, ls="-.",
               label=f"T1_joint (r = 3) {s['T1_joint']['all_device_mean']:.4f}")
    ax.axhline(0.080, color=RED, lw=0.9, ls=":", label="pre-registered limit 0.080")
    ax.axhline(s["ORACLE_joint"]["all_device_mean"], color=AQUA, lw=1, ls=":",
               label=f"ORACLE_joint (in-sample) {s['ORACLE_joint']['all_device_mean']:.4f}")
    ax.set_xticks(rr)
    ax.set_xlabel("round r of the correction used in the held-out refit", fontsize=9)
    ax.set_ylabel("held-out all-device RRMS (12 devices)", fontsize=9)
    ax.set_title("REAL_DEV: held-out RRMS after the refit, per round", fontsize=9.5)
    ax.set_ylim(0.0, 0.12)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax = axes[2]
    style(ax)
    for k, c in (("gf180", VIOLET), ("ihp", AQUA)):
        m = rounds_means(res[k], "M")
        ax.plot(rr, m["corr"], "-o", color=c, ms=4, label=f"{k}: training targets, g_r on c_r")
        ax.plot(rr, m["post"], "--^", color=c, ms=4, label=f"{k}: held-out, g_r on the start card")
        ax.plot(rr, m["unc"], ":", color=c, lw=1, label=f"{k}: training targets, c_r without g")
    ax.set_xticks(rr)
    ax.set_xlabel("round r", fontsize=9)
    ax.set_ylabel("mean paper RRMS over folds", fontsize=9)
    ax.set_title("simulated (secondary), M arm, leave-one-geometry-out", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False)
    fig.suptitle("I2-1 round diagnostics (not decision inputs): the training-device RRMS falls "
                 "with each round; the held-out RRMS does not", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I21_RUNS[k] / "results.json") for k in ("real", "gf180", "ihp")]


def fig_i2_1_pdk_bars(res: dict, path: Path) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), dpi=DPI)
    fig.patch.set_facecolor(BG)
    panels = [("real", ARMS, "REAL_DEV, 12 measured devices (decision)"),
              ("gf180", ARMS[:3], "GF180 tier c, 48 simulated targets (secondary)"),
              ("ihp", ARMS[:3], "IHP tier c, 48 simulated targets (secondary)")]
    for ax, (k, arms, title) in zip(axes, panels):
        style(ax)
        r = res[k]
        s, pd_ = r["summary"], r["per_device"]
        vals = [s[a]["all_device_mean"] for a in arms]
        xb = np.arange(len(arms))
        ax.bar(xb, vals, color=[ARM_COLOR[a] for a in arms], width=0.65, zorder=3)
        n = s["C0_refit"]["n"]
        for xi, a, v in zip(xb, arms, vals):
            w = "" if a == "C0_refit" else \
                f"\n< C0 {sum(pd_[a][t] < pd_['C0_refit'][t] for t in pd_[a])}/{n}"
            ax.annotate(f"{v:.4f}{w}", (xi, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, color=INK)
        if k != "real":
            cs = r["check_S"]
            ax.axhline(0.9 * s["C0_refit"]["all_device_mean"], color=RED, ls=":", lw=0.9)
            ax.text(0.02, 0.98, f"M_joint < M_post on {r['wins']['M_joint_beats_M_post']}/{n}\n"
                    f"red dotted line: 0.90 x C0; M_joint / C0 = "
                    f"{cs['ratio_M_joint_over_C0_refit']:.3f}: check_S "
                    f"{'pass' if cs['M_joint_le_0.90_C0_refit'] else 'fail'}",
                    transform=ax.transAxes, va="top", fontsize=7.5, color=INK)
        else:
            ax.axhline(0.080, color=RED, ls=":", lw=0.9)
            ax.text(len(arms) - 0.5, 0.080, "limit 0.080", fontsize=7.3, color=RED,
                    ha="right", va="bottom")
            ax.text(0.98, 0.98, f"M_joint < M_post on {r['wins']['M_joint_beats_M_post']}/12\nKILL",
                    transform=ax.transAxes, va="top", ha="right", fontsize=7.5, color=INK)
        ax.set_xticks(xb)
        ax.set_xticklabels([a.replace("_", " ") for a in arms], rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, max(vals) * 1.4)
        ax.set_ylabel("all-device mean paper RRMS (0 failed)", fontsize=9)
        ax.set_title(title, fontsize=9.5)
    fig.suptitle("I2-1: held-out RRMS after the refit with the frozen correction, per device set "
                 "(all-device mean; GF180/IHP: 24 nMOS + 24 pMOS, so all-device = combined)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I21_RUNS[k] / "results.json") for k in ("real", "gf180", "ihp")]


def table_i2_1_sim(res: dict) -> rf.Table:
    rows = []
    for k in ("gf180", "ihp"):
        r = res[k]
        s, pd_ = r["summary"], r["per_device"]
        cs = r["check_S"]
        ratio = s["M_joint"]["all_device_mean"] / s["C0_refit"]["all_device_mean"]
        assert abs(ratio - cs["ratio_M_joint_over_C0_refit"]) < 1e-12
        assert (ratio <= 0.90) == cs["M_joint_le_0.90_C0_refit"]
        for a in ("C0_refit", "M_post", "M_joint"):
            rows.append({"pdk": PDK_LABEL[k] if a == "C0_refit" else "", "arm": a,
                         "all": s[a]["all_device_mean"], "median": s[a]["median"],
                         "ratio": None if a == "C0_refit" else
                         s[a]["all_device_mean"] / s["C0_refit"]["all_device_mean"],
                         "c0": None if a == "C0_refit" else
                         f"{sum(pd_[a][t] < pd_['C0_refit'][t] for t in pd_[a])}/48",
                         "mp": f"{r['wins']['M_joint_beats_M_post']}/48" if a == "M_joint" else None,
                         "failed": f"{s[a]['n_failed']}/{s[a]['n']}",
                         "check": ("pass" if cs["M_joint_le_0.90_C0_refit"] else "fail")
                         if a == "M_joint" else "",
                         "time": f"{r['runtime_s']:.0f} s" if a == "C0_refit" else ""})
    cols = [
        {"key": "pdk", "head": "set", "md": "set", "kind": "text"},
        {"key": "arm", "head": "arm", "md": "arm", "kind": "text"},
        {"key": "all", "head": "all-target", "group": "RRMS, 48 targets", "md": "all-target mean", "kind": "rrms"},
        {"key": "median", "head": "median", "group": "RRMS, 48 targets", "md": "median", "kind": "rrms"},
        {"key": "ratio", "head": "/ C0", "md": "ratio to C0_refit", "kind": "rrms"},
        {"key": "c0", "head": "< C0", "md": "targets below C0_refit", "kind": "rtext"},
        {"key": "mp", "head": "< M_post", "md": "targets below M_post", "kind": "rtext"},
        {"key": "failed", "head": "failed", "md": "failed targets", "kind": "rtext"},
        {"key": "check", "head": "check_S", "md": "M_joint <= 0.90 x C0_refit", "kind": "rtext"},
        {"key": "time", "head": "run time", "md": "run time", "kind": "rtext"},
    ]
    notes = [
        "Secondary (prereg simulated_arm): tier c truth sets 0-2, 16 geometries x 3 draws per PDK, "
        "leave-one-geometry-out (16 folds). All-target mean = combined (24 nMOS, 24 pMOS). "
        "pdk2.metric paper RRMS with target-frozen inclusion, strict.",
        "M_post on these sets is a new run (held-out refit with the E1 fold correction g_0); T1 "
        "is not run on the simulated sets. check_S: M_joint all-target mean <= 0.90 x C0_refit; "
        "it does not change the REAL_DEV decision.",
        "Queue jobs i2_1_gf180 (2026-10-19 18:35-20:24) and i2_1_ihp (20:24-22:25).",
    ]
    return rf.Table("13_table_i2_1_simulated",
                    "I2-1 simulated arm (secondary): GF180 and IHP tier c, leave-one-geometry-out",
                    cols, rows, notes)


def table_i2_1_ident(res: dict) -> rf.Table:
    rows = []
    for k in ("gf180", "ihp"):
        for a in ("M_post", "M_joint"):
            d = res[k]["identifiability"][a]
            p, o = d["pearson"], d["ols"]
            rows.append({"pdk": k.upper() if a == "M_post" else "", "arm": a,
                         "cp": f3(p["c_param"]), "crs": f3(p["c_rs"]), "ch": f3(p["c_heat"]),
                         "cg": f3(p["c_gain"]), "cs": f3(p["c_sum"]), "res": f3(p["resid"]),
                         "r2": f3(o["r2"]),
                         "coef": f"{o['c_param']:.2f} / {o['c_rs']:.2f} / {o['c_heat']:.2f} / "
                                 f"{o['c_gain']:.2f}",
                         "mt": f3(d["mean_per_target_pearson_c_sum"]),
                         "sg": f"{d['std']['g']:.3f}", "sr": f"{d['std']['resid']:.3f}",
                         "n": f"{d['n_points']:,}"})
    cols = [
        {"key": "pdk", "head": "PDK", "md": "PDK", "kind": "text"},
        {"key": "arm", "head": "arm", "md": "arm", "kind": "text"},
        {"key": "cp", "head": "c_param", "group": "Pearson r of g with", "md": "r(g, c_param)", "kind": "rtext"},
        {"key": "crs", "head": "c_rs", "group": "Pearson r of g with", "md": "r(g, c_rs)", "kind": "rtext"},
        {"key": "ch", "head": "c_heat", "group": "Pearson r of g with", "md": "r(g, c_heat)", "kind": "rtext"},
        {"key": "cg", "head": "c_gain", "group": "Pearson r of g with", "md": "r(g, c_gain)", "kind": "rtext"},
        {"key": "cs", "head": "c_sum", "group": "Pearson r of g with", "md": "r(g, c_sum)", "kind": "rtext"},
        {"key": "res", "head": "residual", "group": "Pearson r of g with", "md": "r(g, observed residual)", "kind": "rtext"},
        {"key": "r2", "head": "R^2", "group": "OLS of g on components", "md": "OLS R^2", "kind": "rtext"},
        {"key": "coef", "head": "coef. param / rs / heat / gain", "group": "OLS of g on components", "md": "OLS coefficients", "kind": "rtext"},
        {"key": "mt", "head": "per-target r(c_sum)", "md": "mean per-target r with c_sum", "kind": "rtext"},
        {"key": "sg", "head": "std g", "md": "std of g", "kind": "rtext"},
        {"key": "sr", "head": "std resid.", "md": "std of observed residual", "kind": "rtext"},
        {"key": "n", "head": "points", "md": "pooled points", "kind": "rtext"},
    ]
    notes = [
        "Diagnostic without a threshold (prereg identifiability_diagnostic). g = the arm's "
        "correction evaluated on its final refit card's simulation; pooled over the 48 held-out "
        "targets, included curves, trimmed points.",
        "Injected components from the truth card (NGSpice on the target grid): c_param = misspecified "
        "non-fitted parameters; c_rs = series resistance; c_heat = per-curve self-heating "
        "temperature; c_gain = per-curve gain; c_sum = their sum; residual = ln|target / I_card|. "
        "Each in ln(current).",
        "OLS: g on [1, c_param, c_rs, c_heat, c_gain]. The largest correlation in both PDKs is with "
        "c_rs (0.20-0.35); R^2 is 0.18-0.21 on GF180 and 0.04-0.07 on IHP.",
    ]
    return rf.Table("14_table_i2_1_identifiability",
                    "I2-1 identifiability diagnostic: correlation of the learned correction g with "
                    "the injected misspecification components (GF180, IHP)", cols, rows, notes)


# ================================================================== E3 / E3b

def e3_per_device(run: Path) -> dict:
    out = {}
    for f in sorted((run / "eval").glob("*.json")):
        d = load(f)
        for k, cells in d.items():
            vals = [fnum(cells.get(T)) for T in HELD]
            out.setdefault(k, {})[f.stem] = float(np.mean(vals)) if all(
                np.isfinite(vals)) else math.nan
    return out


def paired(per: dict, a: str, b: str, extra: tuple = ()) -> tuple:
    """Arrays of a, b on devices finite for a, b and the extra keys."""
    keys = (a, b) + tuple(extra)
    tags = sorted(t for t in per[a] if all(np.isfinite(per[k][t]) for k in keys))
    return np.array([per[a][t] for t in tags]), np.array([per[b][t] for t in tags]), tags


def table_e3b_decision(summ: dict) -> rf.Table:
    rows = []
    for tier in TIERS:
        s = summ[tier]
        u0 = s["u0_switch_check"]
        for S in SETS:
            d = s["decisions"][S]
            lc, lp, ff = d["lrn_vs_control"], d["lrn_vs_pt_interp"], d["failed_fraction"]
            c_r, c_ci, c_pt = lc["reduction"] >= 0.25, lc["ci95"][0] > 0, lp["reduction"] >= 0.10
            c_ff = all(v <= 0.10 for v in ff.values())
            assert (c_r and c_ci and c_pt and c_ff) == d["meets_continue"]
            rows.append({"tier": f"tier {tier}" + (" (decision)" if tier == "M" else " (secondary)"),
                         "S": S, "control": d["control"], "R": lc["reduction"], "ci": ci(lc["ci95"]),
                         "c_r": yn(c_r), "c_ci": yn(c_ci), "rp": lp["reduction"], "c_pt": yn(c_pt),
                         "ff": f"{max(ff.values()):.3f}", "c_ff": yn(c_ff),
                         "cont": yn(c_r and c_ci and c_pt and c_ff),
                         "k_r": yn(lc["reduction"] < 0.10), "k_pt": yn(lp["reduction"] < 0),
                         "u0": "valid" if u0["valid"] else "INVALID",
                         "verdict": s["verdict_rule_output"] if S == "O4" else ""})
        dd = s["decisions"]
        cont = any(dd[S]["meets_continue"] for S in SETS)
        kill = all(dd[S]["lrn_vs_control"]["reduction"] < 0.10 for S in SETS) or \
            all(dd[S]["lrn_vs_pt_interp"]["reduction"] < 0 for S in SETS)
        v = "CONTINUE" if cont else ("KILL" if kill else "INCONCLUSIVE")
        if not u0["valid"]:
            v = "INVALID"
        assert v == s["verdict_rule_output"], (tier, v)
    cols = [
        {"key": "tier", "head": "run", "md": "run", "kind": "text"},
        {"key": "S", "head": "S", "md": "S", "kind": "text"},
        {"key": "control", "head": "control", "md": "control", "kind": "text"},
        {"key": "R", "head": "R_S", "group": "lrn vs control", "md": "R_S = 1 − lrn/control", "kind": "rrms"},
        {"key": "ci", "head": "95% CI", "group": "lrn vs control", "md": "R_S bootstrap 95% CI", "kind": "rtext"},
        {"key": "c_r", "head": ">= 0.25", "group": "continue checks", "md": "R_S >= 0.25", "kind": "rtext"},
        {"key": "c_ci", "head": "CI lo > 0", "group": "continue checks", "md": "CI lower bound > 0", "kind": "rtext"},
        {"key": "rp", "head": "vs pt_interp", "group": "continue checks", "md": "1 − lrn/pt_interp", "kind": "rrms"},
        {"key": "c_pt", "head": ">= 0.10", "group": "continue checks", "md": "lrn >= 10% below pt_interp", "kind": "rtext"},
        {"key": "ff", "head": "max failed", "group": "continue checks", "md": "max failed-cell fraction", "kind": "rtext"},
        {"key": "c_ff", "head": "<= 0.10", "group": "continue checks", "md": "failed <= 10%", "kind": "rtext"},
        {"key": "cont", "head": "all", "group": "continue checks", "md": "continue for S", "kind": "rtext"},
        {"key": "k_r", "head": "R_S < 0.10", "group": "kill conditions", "md": "R_S < 0.10", "kind": "rtext"},
        {"key": "k_pt", "head": "lrn > pt_int.", "group": "kill conditions", "md": "lrn worse than pt_interp", "kind": "rtext"},
        {"key": "u0", "head": "u0 check", "md": "u0-switch assertion", "kind": "rtext"},
        {"key": "verdict", "head": "rule output", "md": "rule output for the run", "kind": "rtext"},
    ]
    um, uw = summ["M"]["u0_switch_check"], summ["W"]["u0_switch_check"]
    notes = [
        "Rule copied without change from the E3 pre-registration (out/research/explore/E3b/prereg.json): "
        "CONTINUE if for at least one S: R_S >= 0.25 AND bootstrap 95% CI lower bound > 0 AND lrn >= 10% "
        "below pt_interp AND failed cells <= 10% for lrn, control and pt_interp. KILL if R_S < 0.10 for "
        "both S, or lrn worse than pt_interp for both S. Otherwise INCONCLUSIVE. A run with a truth, "
        "fitted card or evaluated card at or above the u0 switch is INVALID.",
        f"u0 check, tier M: 0/{um['truths']['n_checked']} truths, 0/{um['fitted_cards']['n_checked']} "
        f"fitted, 0/{um['predicted_cards_before_clip']['n_checked']} predicted (before clip) and "
        f"0/{um['evaluated_cards']['n_checked']} evaluated cards at or above the switch; 0 clipped "
        f"cards; {um['truths']['n_regenerated']} truths regenerated ({um['truths']['n_rejected_draws']} "
        f"rejected draws). Tier W: same counts, {uw['truths']['n_regenerated']} truths regenerated "
        f"({uw['truths']['n_rejected_draws']} rejected draws).",
        "Decision run: tier M (queue job E3b_full_M, 2026-10-19 22:25-22:40, 910 s). Tier W (E3b_full_W, "
        "22:40-22:57, 989 s; truth inside the parametric family) is secondary. Tier W O4 uses 35 devices "
        "(1 failed lrn cell). Simulated devices only; no measured multi-temperature I-V exists.",
        "The E3 verdict (KILL, queue job E3_full_M) stands; E3b is a separate pre-registered probe.",
    ]
    return rf.Table("15_table_e3b_decision",
                    "E3b decision: learned shared temperature law (lrn) vs parametric laws after the "
                    "u0 unit-switch fix, pre-registered checks", cols, rows, notes)


def fig_e3b(per: dict, summ: dict, path: Path) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10.5), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for r, tier in enumerate(TIERS):
        for c, S in enumerate(SETS):
            ax = axes[r, c]
            style(ax)
            ctl = CONTROL[S]
            yr, xr, tags = paired(per[tier], f"{S}|lrn", f"{S}|{ctl}", (f"{S}|pt_interp",))
            pt = np.array([per[tier][f"{S}|pt_interp"][t] for t in tags])
            xa, ya = np.maximum(xr, 1e-4), np.maximum(yr, 1e-4)
            colors = [BLUE if t.startswith("nmos") else ORANGE for t in tags]
            ax.scatter(xa, ya, c=colors, s=26, zorder=3, edgecolor=BG, lw=0.6)
            lo, hi = min(xa.min(), ya.min()) * 0.7, max(xa.max(), ya.max()) * 1.4
            ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.9, ls="--", zorder=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            gm = summ[tier]["decisions"][S]["secondary_paired_ratio_lrn_over_control"]
            ax.text(0.02, 0.98,
                    f"lrn: mean {yr.mean():.3f}, median {np.median(yr):.3f}\n"
                    f"{ctl}: mean {xr.mean():.3f}, median {np.median(xr):.3f}\n"
                    f"pt_interp: mean {pt.mean():.3f}, median {np.median(pt):.3f}\n"
                    f"lrn < {ctl} on {int(np.sum(yr < xr))}/{len(tags)} test devices\n"
                    f"geometric-mean ratio {gm['geometric_mean_ratio']:.3g} {ci(gm['ci95'])}",
                    transform=ax.transAxes, va="top", fontsize=7.5, color=INK,
                    bbox=dict(facecolor=BG, edgecolor="none", alpha=0.85))
            ax.set_ylim(lo, hi * (60 if tier == "W" and S == "O4" else 12))
            ax.set_xlabel(f"{ctl} (control for {S}): held-out RRMS, mean of 110/190/260 K",
                          fontsize=8.5, color=INK)
            ax.set_ylabel("lrn (learned shared law): held-out RRMS", fontsize=8.5, color=INK)
            role = "decision run" if tier == "M" else "secondary"
            ax.set_title(f"E3b tier {tier} ({role}), {S}: R_S = "
                         f"{summ[tier]['decisions'][S]['lrn_vs_control']['reduction']:.3f}",
                         fontsize=9.5, color=INK)
    h1 = axes[0, 0].scatter([], [], color=BLUE, s=26, label="nMOS test device")
    h2 = axes[0, 0].scatter([], [], color=ORANGE, s=26, label="pMOS test device")
    fig.legend(handles=[h1, h2], fontsize=8, frameon=False, loc="lower center", ncol=2)
    fig.suptitle("E3b (u0 fix), per simulated test device: learned shared temperature law vs the "
                 "pre-registered control\n(below the dashed line: lrn lower; values below 1e-4 drawn "
                 "at 1e-4; y-axis extended upward for the text)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(E3B / f"tier_{t}" / "eval") + "/*.json" for t in TIERS] + \
        [rel(E3B / f"tier_{t}" / "summary.json") for t in TIERS]


E3_METHODS = {"O4": ["lrn", "par3", "par14", "par3_pop", "pt_interp", "lrn_mean", "lrn_noH",
                     "lrn_param"],
              "O2": ["lrn", "par3_pop", "par3_g1", "par14_g1", "pt_interp", "lrn_mean",
                     "lrn_noH", "lrn_param"]}


def table_e3b_results(per: dict, summ: dict) -> rf.Table:
    rows = []
    for tier in TIERS:
        tab = summ[tier]["table"]
        for S in SETS:
            for m in E3_METHODS[S]:
                k = f"{S}|{m}"
                f = fam(per[tier][k])
                pt = tab[k]["per_temp"]
                rows.append({"tier": tier, "S": S, "m": m + (" (control)" if m == CONTROL[S] else ""),
                             "mean": tab[k]["heldout_all_device_mean"], "combined": f["combined"],
                             "median": f["median"], "t110": pt["110"], "t190": pt["190"],
                             "t260": pt["260"], "anchor": tab[k]["anchor_all_device_mean"],
                             "failed": f"{tab[k]['n_failed_cells']}/{tab[k]['n_cells']}"})
        t = tab["floor|floor"]
        f = fam(per[tier]["floor|floor"])
        rows.append({"tier": tier, "S": "—", "m": "floor (reference)",
                     "mean": t["heldout_all_device_mean"], "combined": f["combined"],
                     "median": f["median"], "t110": t["per_temp"]["110"],
                     "t190": t["per_temp"]["190"], "t260": t["per_temp"]["260"], "anchor": None,
                     "failed": f"{t['n_failed_cells']}/{t['n_cells']}"})
    cols = [
        {"key": "tier", "head": "tier", "md": "tier", "kind": "text"},
        {"key": "S", "head": "S", "md": "S", "kind": "text"},
        {"key": "m", "head": "method", "md": "method", "kind": "text"},
        {"key": "mean", "head": "all-device", "group": "held-out RRMS, 36 test devices", "md": "held-out all-device mean", "kind": "rrms"},
        {"key": "combined", "head": "combined", "group": "held-out RRMS, 36 test devices", "md": "held-out combined", "kind": "rrms"},
        {"key": "median", "head": "median", "group": "held-out RRMS, 36 test devices", "md": "held-out median", "kind": "rrms"},
        {"key": "t110", "head": "110 K", "group": "per temperature", "md": "110 K", "kind": "rrms", "family": True},
        {"key": "t190", "head": "190 K", "group": "per temperature", "md": "190 K", "kind": "rrms", "family": True},
        {"key": "t260", "head": "260 K", "group": "per temperature", "md": "260 K", "kind": "rrms", "family": True},
        {"key": "anchor", "head": "anchors", "md": "observed-temperature mean", "kind": "rrms"},
        {"key": "failed", "head": "failed", "md": "failed cells", "kind": "rtext"},
    ]
    notes = [
        "all-device = mean over the 36 simulated test devices of the per-device mean over 110/190/260 K "
        "(summary.json; a device with a failed cell is left out of that method's mean); combined = mean "
        "of the nMOS (15) and pMOS (21) family means; median over test devices.",
        "Methods: lrn = learned shared law (primary); par3 = per-device law x = c + (mu − c) w^g; par14 "
        "= cryoml.temperature family; par3_pop = law with population-pooled g; *_g1 = g = 1; pt_interp "
        "= linear interpolation of the per-temperature anchor fits; lrn_mean = population-mean "
        "correction without a network; lrn_noH = trained without 110/190/260 K; lrn_param = "
        "parameter-MSE loss; floor = per-temperature fit on the held-out data.",
        "O4 observes 77/150/225/300 K; O2 observes 77/300 K. Anchors: mean RRMS at the observed "
        "temperatures. Mean training-fit RRMS: tier M "
        f"{summ['M']['mean_train_fit_rrms']:.4f}, tier W {summ['W']['mean_train_fit_rrms']:.4f}.",
    ]
    return rf.Table("17_table_e3b_results",
                    "E3b held-out-temperature RRMS per method, tiers M and W (simulated devices)",
                    cols, rows, notes)


def table_e3b_secondary(per: dict, summ: dict) -> rf.Table:
    rows = []
    for tier in TIERS:
        for S in SETS:
            d = summ[tier]["decisions"][S]
            lm = d["lrn_vs_other"]["lrn_mean"]
            a, b, tags = paired(per[tier], f"{S}|lrn_noH", f"{S}|lrn")
            nh = e3m.bootstrap_reduction(b, a)  # 1 − lrn / lrn_noH
            p, q, _ = paired(per[tier], f"{S}|lrn_param", f"{S}|lrn")
            lp = e3m.bootstrap_reduction(q, p)  # 1 − lrn / lrn_param
            gm = d["secondary_paired_ratio_lrn_over_control"]
            fl = summ[tier]["table"]["floor|floor"]["heldout_all_device_mean"]
            rows.append({"tier": f"tier {tier}", "S": S,
                         "lm": f"{lm['reduction']:.3f} {ci(lm['ci95'])}",
                         "nh": f"{nh['reduction']:.3f} {ci(nh['ci95'])}",
                         "lp": f"{lp['reduction']:.3f} {ci(lp['ci95'])}",
                         "gap": d["gap_closed_to_floor"], "floor": fl,
                         "gm": f"{gm['geometric_mean_ratio']:.3g} "
                               f"[{gm['ci95'][0]:.3g}, {gm['ci95'][1]:.3g}]",
                         "wins": f"{gm['wins']}/{gm['n_devices']}",
                         "oth": "; ".join(f"{m} {v['reduction']:.3f}" for m, v in
                                          d["lrn_vs_other"].items() if m != "lrn_mean")})
    cols = [
        {"key": "tier", "head": "run", "md": "run", "kind": "text"},
        {"key": "S", "head": "S", "md": "S", "kind": "text"},
        {"key": "lm", "head": "1 − lrn/lrn_mean [CI]", "md": "lrn vs lrn_mean", "kind": "rtext"},
        {"key": "nh", "head": "1 − lrn/lrn_noH [CI]", "md": "lrn vs lrn_noH", "kind": "rtext"},
        {"key": "lp", "head": "1 − lrn/lrn_param [CI]", "md": "lrn vs lrn_param", "kind": "rtext"},
        {"key": "gap", "head": "gap closed", "md": "gap closed to floor", "kind": "rrms"},
        {"key": "floor", "head": "floor", "md": "floor all-device", "kind": "rrms"},
        {"key": "gm", "head": "geo. mean lrn/control [CI]", "group": "secondary paired statistic", "md": "geometric mean ratio lrn/control [CI]", "kind": "rtext"},
        {"key": "wins", "head": "lrn < control", "group": "secondary paired statistic", "md": "devices lrn < control", "kind": "rtext"},
        {"key": "oth", "head": "1 − lrn/other", "md": "lrn vs other parametric laws", "kind": "text"},
    ]
    notes = [
        "Secondary quantities listed in the pre-registration as reported, not decisive. Reductions "
        "1 − mean(lrn)/mean(other) with a paired percentile bootstrap 95% CI over test devices (4000 "
        "resamples, seed 0) on devices finite for both. lrn_mean and the other-law columns are stored "
        "in summary.json; the lrn_noH and lrn_param columns are computed for this report from eval/*.json "
        "with the same bootstrap function (e3_temp_misspec.bootstrap_reduction).",
        "gap closed = (control − lrn) / (control − floor), all-device means on the paired devices.",
        "Geometric-mean ratio = exp(mean over devices of log(lrn / control)) with bootstrap CI and win "
        "count (prereg item d; not a decision input). In tier W O4 the control par3 is near the floor "
        "on many devices (the truth is inside its family), so per-device ratios are large.",
    ]
    return rf.Table("18_table_e3b_secondary",
                    "E3b secondary quantities: lrn vs lrn_mean, lrn_noH and lrn_param, gap closed "
                    "to the floor, paired geometric-mean ratio", cols, rows, notes)


def table_e3_vs_e3b(per3: dict, per3b: dict, s3: dict, s3b: dict, sens: dict) -> rf.Table:
    rows = []
    for tier in TIERS:
        for S in SETS:
            ctl = CONTROL[S]
            for name, per, summ in (("E3", per3, s3), ("E3b", per3b, s3b)):
                d = summ[tier]["decisions"][S]
                y, x, tags = paired(per[tier], f"{S}|lrn", f"{S}|{ctl}", (f"{S}|pt_interp",))
                pt = np.array([per[tier][f"{S}|pt_interp"][t] for t in tags])
                gm = d.get("secondary_paired_ratio_lrn_over_control") or \
                    e3m.bootstrap_log_ratio(y, x)
                rows.append({"run": f"{name} tier {tier}", "S": S, "lrn": y.mean(),
                             "ctl": x.mean(), "pt": pt.mean(),
                             "mlrn": float(np.median(y)), "mctl": float(np.median(x)),
                             "R": f"{d['lrn_vs_control']['reduction']:.3f} {ci(d['lrn_vs_control']['ci95'])}",
                             "rp": d["lrn_vs_pt_interp"]["reduction"],
                             "gm": f"{gm['geometric_mean_ratio']:.3g}" + ("" if name == "E3b" else " *"),
                             "n": f"{len(tags)}", "verdict": summ[tier]["verdict_rule_output"]})
            if tier == "M":
                e = sens["E3"]["without_nmos_L1_W3__2"][S]
                rows.append({"run": "E3 tier M without nmos_L1_W3__2 (I2-0, diagnostic)", "S": S,
                             "lrn": e["lrn"], "ctl": e["control_mean"], "pt": e["pt_interp"],
                             "mlrn": e["medians"]["lrn"], "mctl": e["medians"]["control"],
                             "R": f"{e['R_S']:.3f} {ci(e['R_S_CI95'])}",
                             "rp": e["lrn_vs_pt_interp_reduction"], "gm": "",
                             "n": str(e["n_devices"]),
                             "verdict": sens["E3"]["without_nmos_L1_W3__2"][
                                 "rule_output_without_failed_cell_check"] + " (not a verdict)"})
    cols = [
        {"key": "run", "head": "run", "md": "run", "kind": "text"},
        {"key": "S", "head": "S", "md": "S", "kind": "text"},
        {"key": "lrn", "head": "lrn", "group": "held-out all-device", "md": "lrn mean", "kind": "rrms"},
        {"key": "ctl", "head": "control", "group": "held-out all-device", "md": "control mean", "kind": "rrms"},
        {"key": "pt", "head": "pt_interp", "group": "held-out all-device", "md": "pt_interp mean", "kind": "rrms"},
        {"key": "mlrn", "head": "lrn", "group": "median", "md": "lrn median", "kind": "rrms"},
        {"key": "mctl", "head": "control", "group": "median", "md": "control median", "kind": "rrms"},
        {"key": "R", "head": "R_S [95% CI]", "md": "R_S with CI", "kind": "rtext"},
        {"key": "rp", "head": "vs pt_interp", "md": "1 − lrn/pt_interp", "kind": "rrms"},
        {"key": "gm", "head": "geo. ratio", "md": "geometric-mean ratio lrn/control", "kind": "rtext"},
        {"key": "n", "head": "devices", "md": "paired devices", "kind": "rtext"},
        {"key": "verdict", "head": "rule output", "md": "rule output of the run", "kind": "rtext"},
    ]
    notes = [
        "E3 (queue jobs E3_full_M, E3_full_W, 2026-10-19 11:52-12:23) and E3b (E3b_full_M, E3b_full_W, "
        "22:25-22:57) use the same settings, seeds, methods and rule. E3b differs by the u0 fix: 4 tier-M "
        "truths (3 training, 1 test: nmos_L1_W3 draws) and 2 tier-W truths regenerated, the nmos_L1_W3 and "
        "nmos_L0p25_W1p6 x_u0 fit bounds lowered to 0.02 below the switch, and the u0 check.",
        "* E3 geometric-mean ratio computed for this report with the E3b function "
        "(e3_temp_misspec.bootstrap_log_ratio) on the E3 eval files; E3 did not pre-register it.",
        "The I2-0 row removes one test device after the fact and keeps the switched truths in training; "
        "it is a diagnostic, not a verdict. The E3 KILL verdict stands; E3b is a separate pre-registered "
        "probe with rule output CONTINUE on tier M.",
    ]
    return rf.Table("19_table_e3_vs_e3b",
                    "E3 vs E3b: held-out means, medians, R_S and rule outputs before and after the u0 "
                    "fix", cols, rows, notes)


# ================================================================== temperature table

TEMP_ORDER = ["direct", "direct_mlp", "ngspice_search", "surrogate_paper_mlp", "surrogate_resmlp",
              "surrogate_deeponet", "surrogate_fno"]
TEMP_LABEL = {"direct": "direct NGSpice (CMA-ES + TRF)", "direct_mlp": "direct MLP, one pass",
              "ngspice_search": "pipeline search on NGSpice",
              "surrogate_paper_mlp": "surrogate search, paper MLP",
              "surrogate_resmlp": "surrogate search, ResMLP",
              "surrogate_deeponet": "surrogate search, DeepONet",
              "surrogate_fno": "surrogate search, FNO"}


def temp_failures(method: str, stage: str) -> tuple[int, int]:
    n_bad = n = 0
    for f in sorted((TEMP / method).glob("set_*.json")):
        for rec in load(f)["devices"].values():
            n += 1
            r = rec.get(stage)
            if r is None or not all(np.isfinite(fnum(r.get(k))) for k in ("observed", "heldout")):
                n_bad += 1
    return n_bad, n


def table_temp(rows_in: list) -> rf.Table:
    by = {r["method"]: r for r in rows_in}
    rows = []
    for m in TEMP_ORDER:
        for stage, key in (("raw", m), ("+ FD", m + "+FD")):
            r = by[key]
            bad, n = temp_failures(m, "raw" if stage == "raw" else "fd")
            pt = r["per_temp"]
            rows.append({"m": TEMP_LABEL[m] if stage == "raw" else "", "st": stage,
                         "sets": r["n_sets"], "obs": r["observed_all_device_mean"],
                         "obs_c": r["observed_combined"], "held": r["heldout_all_device_mean"],
                         "held_c": r["heldout_combined"], "t110": pt["110"], "t190": pt["190"],
                         "t260": pt["260"], "perr": r["local_error_rms"],
                         "failed": f"{bad}/{n}"})
    fno = load(TEMP / "surrogate_fno" / "training.json")
    res = load(TEMP / "surrogate_resmlp" / "training.json")
    cols = [
        {"key": "m", "head": "method", "md": "method", "kind": "text"},
        {"key": "st", "head": "stage", "md": "stage", "kind": "text"},
        {"key": "sets", "head": "sets", "md": "truth sets", "kind": "int"},
        {"key": "obs", "head": "all-device", "group": "observed 77/150/225/300 K", "md": "observed all-device", "kind": "rrms"},
        {"key": "obs_c", "head": "combined", "group": "observed 77/150/225/300 K", "md": "observed combined", "kind": "rrms"},
        {"key": "held", "head": "all-device", "group": "held-out 110/190/260 K", "md": "held-out all-device", "kind": "rrms"},
        {"key": "held_c", "head": "combined", "group": "held-out 110/190/260 K", "md": "held-out combined", "kind": "rrms"},
        {"key": "t110", "head": "110 K", "group": "held-out per temperature", "md": "110 K", "kind": "rrms", "family": True},
        {"key": "t190", "head": "190 K", "group": "held-out per temperature", "md": "190 K", "kind": "rrms", "family": True},
        {"key": "t260", "head": "260 K", "group": "held-out per temperature", "md": "260 K", "kind": "rrms", "family": True},
        {"key": "perr", "head": "param. err.", "md": "local parameter error RMS", "kind": "rrms"},
        {"key": "failed", "head": "failed", "md": "failed device-set records", "kind": "rtext"},
    ]
    notes = [
        "Simulated well-specified benchmark: 3 truth sets x 18 device geometries, truth = the 14-number "
        "temperature family of cryoml.temperature (tnom 27 C); fits observe 77/150/225/300 K; scored at "
        "the held-out 110/190/260 K by NGSpice re-simulation. Values are means over the 3 sets of the "
        "all-device mean (18 devices) and combined (nMOS, pMOS family means). No measured I-V.",
        "failed = device-set records with a non-finite observed or held-out RRMS (strict). +FD = NGSpice "
        "finite-difference TRF polish (max_nfev 400) from the method's output. param. err. = RMS of the "
        "local (box-normalized) parameter error.",
        f"FNO: trained {fno['epochs']} epochs ({fno['train_s']:.0f} s, best validation loss "
        f"{fno['best_val']:.5f} at the last epoch) in queue job temp_sur_fno (2026-10-17 23:37 - "
        "10-18 03:38), which was killed after training (rc 137, low memory). The rerun temp_sur_fno_r "
        "(2026-10-19 22:56-23:13, 1002 s, rc 0) loaded the saved checkpoint, ran the held-out emulator "
        "error and validation passes with the --eval-batch memory fix (default 4096 rows, commit "
        "168cf9f) and completed sets 0, 1 and 2 (18 devices each).",
        f"ResMLP: {res['epochs']} epochs ({res['train_s']:.0f} s), queue job temp_sur_resmlp "
        "(2026-10-19 14:30-17:59). Table written by temp_study.py table (queue job temp_table, "
        "2026-10-19 23:13).",
    ]
    return rf.Table("20_table_temp_benchmark",
                    "Temperature benchmark (simulated, well-specified): paper RRMS per method, "
                    "raw and after FD, all surrogate architectures", cols, rows, notes)


# ================================================================== main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    topic = "explore_i2_e3b"
    folder = rf.REPORT_ROOT / f"{args.date}_{topic}"
    folder.mkdir(parents=True, exist_ok=True)
    items, tables = [], []

    # ---- I2-0
    sens = load(I20 / "C_decision_sensitivity.json")
    t = table_i2_0_real(sens)
    rf.render_table(t, folder / "01_table_i2_0_real_sensitivity.png")
    tables.append(t)
    items.append(("01_table_i2_0_real_sensitivity.png",
                  "Table (diagnostic only) of the iteration-1 REAL_DEV decision quantities of E1 stage 1 "
                  "and 2, E2, E4 p43 vd and E5 b43 with all 12 devices, without pmos_L0p35_W5, "
                  "without pmos_L0p35_W0p55 and without both",
                  [rel(I20 / "C_decision_sensitivity.json")]))
    t = table_i2_0_e3(sens)
    rf.render_table(t, folder / "02_table_i2_0_e3_sensitivity.png")
    tables.append(t)
    items.append(("02_table_i2_0_e3_sensitivity.png",
                  "Table (diagnostic only) of the E3 tier-M decision quantities (means, R_S with CI, lrn "
                  "vs pt_interp, wins, rule output) with all 36 test devices, without nmos_L1_W3__2 "
                  "and without the three nmos_L1_W3 test devices",
                  [rel(I20 / "C_decision_sensitivity.json")]))
    for dst, src, desc in I20_COPIES:
        shutil.copyfile(I20 / src, folder / dst)
        items.append((dst, desc + f" (copy of out/research/explore/I2_0/{src})",
                      [rel(I20 / src), rel(I20 / "diagnosis.md")]))

    # ---- I2-1
    res = i21_load()
    for k, r in res.items():
        assert r["preregistered_config"] and not r["smoke"], k
    t = table_i2_1_decision(res["real"])
    rf.render_table(t, folder / "08_table_i2_1_decision.png")
    tables.append(t)
    items.append(("08_table_i2_1_decision.png",
                  "Table of the four pre-registered I2-1 conditions on REAL_DEV (M_joint mean <= 0.080, "
                  "M_joint < M_post on >= 8/12, M_joint <= T1_joint + 0.005, 0 failed) with values and "
                  "the rule output", [rel(I21 / "prereg.json"), rel(I21_RUNS["real"] / "results.json")]))
    t = table_i2_1_results(res["real"])
    rf.render_table(t, folder / "09_table_i2_1_results.png")
    tables.append(t)
    items.append(("09_table_i2_1_results.png",
                  "Table of I2-1 REAL_DEV RRMS per arm (C0_refit, M_post, M_joint, T1_joint, M_joint "
                  "with g_1 and g_2, ORACLE_joint): all-device, combined, nMOS, pMOS, median, devices "
                  "below C0 and M_post, failed devices", [rel(I21_RUNS["real"] / "results.json")]))
    src = fig_i2_1_per_device(res["real"], folder / "10_i2_1_per_device.png")
    items.append(("10_i2_1_per_device.png",
                  "Per-device held-out RRMS on REAL_DEV for C0_refit, M_post, M_joint, T1_joint and "
                  "ORACLE_joint (top, log scale), and paired scatter plots M_joint vs M_post and T1_joint "
                  "vs C0_refit with the diagonal (bottom)", src))
    src = fig_i2_1_rounds(res, folder / "11_i2_1_rounds.png")
    items.append(("11_i2_1_rounds.png",
                  "I2-1 round diagnostics: REAL_DEV training-device mean RRMS per round (g_r on c_r, "
                  "g_(r-1) on c_r) and post-hoc held-out RRMS of g_r on the starting card for M and T1 "
                  "(left); REAL_DEV held-out all-device RRMS after the refit with g_0 (M_post) to g_3 "
                  "with C0, T1_joint, ORACLE_joint and the 0.080 limit (middle); GF180 and IHP M-arm "
                  "training and post-hoc held-out RRMS per round (right)", src))
    src = fig_i2_1_pdk_bars(res, folder / "12_i2_1_pdk_bars.png")
    items.append(("12_i2_1_pdk_bars.png",
                  "Bars of the all-device mean RRMS per arm for REAL_DEV (C0_refit, M_post, M_joint, "
                  "T1_joint, ORACLE_joint), GF180 tier c and IHP tier c (C0_refit, M_post, M_joint) "
                  "with devices below C0, the REAL_DEV 0.080 limit and the simulated check_S line "
                  "(0.90 x C0)", src))
    t = table_i2_1_sim(res)
    rf.render_table(t, folder / "13_table_i2_1_simulated.png")
    tables.append(t)
    items.append(("13_table_i2_1_simulated.png",
                  "Table of the I2-1 simulated arm per PDK: all-target mean and median RRMS of C0_refit, "
                  "M_post and M_joint, ratio to C0, targets below C0 and M_post, failed targets, check_S",
                  [rel(I21_RUNS[k] / "results.json") for k in ("gf180", "ihp")]))
    t = table_i2_1_ident(res)
    rf.render_table(t, folder / "14_table_i2_1_identifiability.png")
    tables.append(t)
    items.append(("14_table_i2_1_identifiability.png",
                  "Table of the I2-1 identifiability diagnostic per PDK and arm: Pearson r of the "
                  "learned correction g with c_param, c_rs, c_heat, c_gain, c_sum and the observed "
                  "residual, OLS R^2 and coefficients, mean per-target r with c_sum",
                  [rel(I21_RUNS[k] / "results.json") for k in ("gf180", "ihp")]))

    # ---- E3b
    s3b = {t_: load(E3B / f"tier_{t_}" / "summary.json") for t_ in TIERS}
    s3 = {t_: load(E3 / f"tier_{t_}" / "summary.json") for t_ in TIERS}
    for t_ in TIERS:
        chk = load(E3B / f"tier_{t_}" / "u0_switch_check.json")
        assert chk["valid"] and s3b[t_]["u0_switch_check"]["valid"]
        assert all(v == 0 for v in s3b[t_]["u0_switch_check"]["n_evaluated_cards_u0_clipped"].values())
    per3b = {t_: e3_per_device(E3B / f"tier_{t_}") for t_ in TIERS}
    per3 = {t_: e3_per_device(E3 / f"tier_{t_}") for t_ in TIERS}
    t = table_e3b_decision(s3b)
    rf.render_table(t, folder / "15_table_e3b_decision.png")
    tables.append(t)
    items.append(("15_table_e3b_decision.png",
                  "Table of the E3b pre-registered checks per tier and observation setting (R_S with CI, "
                  "lrn vs pt_interp, failed cells, kill conditions, u0-switch assertion, rule output); "
                  "tier M decides, tier W is secondary",
                  [rel(EXP / "E3b/prereg.json")] +
                  [rel(E3B / f"tier_{t_}" / n) for t_ in TIERS for n in ("summary.json", "u0_switch_check.json")]))
    src = fig_e3b(per3b, s3b, folder / "16_e3b_per_device.png")
    items.append(("16_e3b_per_device.png",
                  "Per simulated test device, lrn vs the pre-registered control (log-log, diagonal), "
                  "tiers M and W, settings O4 and O2, with means, medians, win counts and the paired "
                  "geometric-mean ratio", src))
    t = table_e3b_results(per3b, s3b)
    rf.render_table(t, folder / "17_table_e3b_results.png")
    tables.append(t)
    items.append(("17_table_e3b_results.png",
                  "Table of E3b held-out RRMS per tier, setting and method: all-device, combined, median, "
                  "110/190/260 K, anchor mean, failed cells, with the floor",
                  [rel(E3B / f"tier_{t_}" / "summary.json") for t_ in TIERS]))
    t = table_e3b_secondary(per3b, s3b)
    rf.render_table(t, folder / "18_table_e3b_secondary.png")
    tables.append(t)
    items.append(("18_table_e3b_secondary.png",
                  "Table of the E3b secondary quantities: lrn vs lrn_mean, lrn_noH and lrn_param with "
                  "bootstrap CIs, gap closed to the floor, paired geometric-mean ratio lrn/control with "
                  "CI and wins, lrn vs the other parametric laws",
                  [rel(E3B / f"tier_{t_}" / "summary.json") for t_ in TIERS] +
                  [rel(E3B / f"tier_{t_}" / "eval") + "/*.json" for t_ in TIERS]))
    t = table_e3_vs_e3b(per3, per3b, s3, s3b, sens)
    rf.render_table(t, folder / "19_table_e3_vs_e3b.png")
    tables.append(t)
    items.append(("19_table_e3_vs_e3b.png",
                  "Table comparing E3 and E3b per tier and setting (lrn, control and pt_interp means and "
                  "medians, R_S with CI, lrn vs pt_interp, geometric-mean ratio, rule output), with the "
                  "I2-0 E3 diagnostic without nmos_L1_W3__2",
                  [rel(E3 / f"tier_{t_}" / "summary.json") for t_ in TIERS] +
                  [rel(E3B / f"tier_{t_}" / "summary.json") for t_ in TIERS] +
                  [rel(I20 / "C_decision_sensitivity.json")]))

    # ---- temperature table
    temp_rows = load(TEMP / "table.json")
    t = table_temp(temp_rows)
    rf.render_table(t, folder / "20_table_temp_benchmark.png")
    tables.append(t)
    items.append(("20_table_temp_benchmark.png",
                  "Table of the simulated temperature benchmark (3 sets x 18 devices): observed and "
                  "held-out all-device and combined RRMS, held-out 110/190/260 K, parameter error and "
                  "failed records for direct NGSpice, direct MLP, NGSpice search and surrogate search "
                  "with paper MLP, ResMLP, DeepONet and FNO, raw and + FD",
                  [rel(TEMP / "table.json")] +
                  [rel(TEMP / m) + "/set_*.json" for m in TEMP_ORDER] +
                  [rel(TEMP / "surrogate_fno/training.json")]))

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = (f"`PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i2_e3b.py "
           f"--date {args.date}`")
    md = [f"# Tables, {args.date} {topic}", "",
          f"Generated {now} by {gen} from result files only. RRMS is the paper metric with frozen "
          "curve inclusion, strict.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))

    readme = [f"# Report batch {args.date} · {topic}", "",
              f"Generated {now} by {gen} from result files only (no simulation, no training).", "",
              "## Contents", "",
              "- I2-0: diagnosis of the devices that determined the iteration-1 means "
              "(`out/research/explore/I2_0/diagnosis.md`). Tables 01-02 are diagnostic only; PNGs "
              "03-07 are copies of the I2-0 figures. The proposed u0 fix "
              "(`proposed_e3_u0_switch.patch`) was applied after user approval in commit 40545e0; "
              "the E3b pre-registration is commit 787e8aa.",
              "- I2-1: correction trained jointly with the training devices' cards (queue jobs "
              "i2_1_real, i2_1_gf180, i2_1_ihp, 2026-10-19 17:59-22:25). Pre-registration: "
              f"`{rel(I21 / 'prereg.json')}`. REAL_DEV rule output KILL.",
              "- E3b: E3 rerun with the u0 unit-switch fix (queue jobs E3b_full_M, E3b_full_W, "
              f"2026-10-19 22:25-22:57). Pre-registration: `{rel(EXP / 'E3b/prereg.json')}`. Tier M "
              "rule output CONTINUE, tier W (secondary) INCONCLUSIVE. Simulated devices only. The E3 "
              "verdict KILL stands; E3b is a separate pre-registered probe.",
              "- Temperature benchmark table (`out/research/temp/table.json`, queue job temp_table "
              "after temp_sur_fno_r, 2026-10-19 23:13).", "",
              "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{s}`" for s in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))

    s = res["real"]["summary"]
    a = lambda k: s[k]["all_device_mean"]  # noqa: E731
    c = lambda k: s[k]["combined"]  # noqa: E731
    g, ih = res["gf180"]["summary"], res["ihp"]["summary"]
    dm, dw = s3b["M"]["decisions"], s3b["W"]["decisions"]
    tf = {r["method"]: r for r in temp_rows}
    summary = [
        f"I2-1 (REAL_DEV LODO), all-device (combined): C0_refit {a('C0_refit'):.4f} "
        f"({c('C0_refit'):.4f}), M_post {a('M_post'):.4f} ({c('M_post'):.4f}), M_joint "
        f"{a('M_joint'):.4f} ({c('M_joint'):.4f}), T1_joint {a('T1_joint'):.4f} ({c('T1_joint'):.4f}), "
        f"ORACLE_joint (in-sample) {a('ORACLE_joint'):.4f}; 0 failed devices. M_joint < M_post on "
        f"{res['real']['wins']['M_joint_beats_M_post']}/12: KILL. Simulated (secondary): GF180 C0 "
        f"{g['C0_refit']['all_device_mean']:.4f}, M_post {g['M_post']['all_device_mean']:.4f}, M_joint "
        f"{g['M_joint']['all_device_mean']:.4f} (check_S pass); IHP C0 {ih['C0_refit']['all_device_mean']:.4f}, "
        f"M_post {ih['M_post']['all_device_mean']:.4f}, M_joint {ih['M_joint']['all_device_mean']:.4f} "
        f"(ratio {res['ihp']['check_S']['ratio_M_joint_over_C0_refit']:.3f}, check_S fail).",
        f"E3b (u0 fix, simulated tier M, 36 test devices, u0 check valid): R_O4 "
        f"{dm['O4']['lrn_vs_control']['reduction']:.3f} {ci(dm['O4']['lrn_vs_control']['ci95'])}, lrn vs "
        f"pt_interp {dm['O4']['lrn_vs_pt_interp']['reduction']:.3f}; R_O2 "
        f"{dm['O2']['lrn_vs_control']['reduction']:.3f} {ci(dm['O2']['lrn_vs_control']['ci95'])}: rule "
        f"output CONTINUE. Tier W (secondary): R_O4 {dw['O4']['lrn_vs_control']['reduction']:.3f}, R_O2 "
        f"{dw['O2']['lrn_vs_control']['reduction']:.3f}: INCONCLUSIVE. lrn vs lrn_mean (tier M O4) "
        f"{dm['O4']['lrn_vs_other']['lrn_mean']['reduction']:.3f} "
        f"{ci(dm['O4']['lrn_vs_other']['lrn_mean']['ci95'])}. The E3 KILL stands.",
        "I2-0 (diagnostic only): without pmos_L0p35_W5 and pmos_L0p35_W0p55 the E2 rule output "
        "changes from KILL to INCONCLUSIVE; E1, E4, E5 unchanged. Without nmos_L1_W3__2 (u0 switch) "
        "the E3 rule output is CONTINUE (R_O4 0.369). The u0 fix was applied after user approval "
        "(commits 40545e0, 787e8aa).",
        f"Temperature benchmark (simulated, well-specified), held-out all-device raw / + FD: direct "
        f"{tf['direct']['heldout_all_device_mean']:.4f} / {tf['direct+FD']['heldout_all_device_mean']:.4f}; "
        f"NGSpice search {tf['ngspice_search']['heldout_all_device_mean']:.4f} / "
        f"{tf['ngspice_search+FD']['heldout_all_device_mean']:.4f}; surrogate paper MLP "
        f"{tf['surrogate_paper_mlp']['heldout_all_device_mean']:.4f} / "
        f"{tf['surrogate_paper_mlp+FD']['heldout_all_device_mean']:.4f}, ResMLP "
        f"{tf['surrogate_resmlp']['heldout_all_device_mean']:.4f} / "
        f"{tf['surrogate_resmlp+FD']['heldout_all_device_mean']:.4f}, DeepONet "
        f"{tf['surrogate_deeponet']['heldout_all_device_mean']:.4f} / "
        f"{tf['surrogate_deeponet+FD']['heldout_all_device_mean']:.4f}, FNO "
        f"{tf['surrogate_fno']['heldout_all_device_mean']:.4f} / "
        f"{tf['surrogate_fno+FD']['heldout_all_device_mean']:.4f}; direct MLP "
        f"{tf['direct_mlp']['heldout_all_device_mean']:.4f} / "
        f"{tf['direct_mlp+FD']['heldout_all_device_mean']:.4f}.",
    ]
    rf.update_index(args.date, topic, folder, [i[0] for i in items], summary)
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
