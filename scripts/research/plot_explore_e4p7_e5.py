#!/usr/bin/env python3
"""Report batch for exploration probes E4 (seven-parameter held-out-curve runs,
splits vd, sparse, vg) and E5 (shared discrepancy, stage 1 decision run and
stage 2 diagnostic).

Reads result JSON files only (no simulation, no training) and writes PNGs,
``tables.md`` and ``README.md`` to
``out/research/report/<date>_explore_e4p7_e5/``, then adds the batch to
``out/research/report/index.json`` and ``INDEX.md``.

E4: the pre-registered decision is p43 split vd (reported in the batch
``2026-10-19_explore_e2_e3_e4``). prereg.json lists p7 vd and p7 vg as
secondary results without a decision; sparse is secondary by amendment 1. The
same function (``e4_analyze.decide``) is applied to the stored p7 summaries for
reporting only.

E5: the pre-registered stage-1 rule is recomputed from the per-device values in
``stage1.json`` and checked against the stored verdicts; stage 2 is reported as
a diagnostic because stage 1 is not AMBIGUOUS.

  PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_e4p7_e5.py \
      --date 2026-10-19
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_explore_e2_e3_e4 as pe  # noqa: E402  (helpers, style, E4 constants)
from plot_explore_e2_e3_e4 import (BG, INK, MUTED, GREY, BLUE, ORANGE, AQUA,  # noqa: E402
                                   VIOLET, RED, DPI, E4, E4_CAND, E4_COLOR, E4_LABEL,
                                   E4_METHODS, EXP, fam, fnum, load, real_dev, rel,
                                   style, yn)

rf = pe.rf
plt = pe.plt

E4_SPLITS = ["vd", "sparse", "vg"]
E4_RUNS = ["p43", "p7"]
E5 = EXP / "E5"
E5_S1 = E5 / "stage1.json"
E5_S2 = E5 / "stage2_diagnostic.json"
E5_LIGHT = E5 / "stage1_light.json"
E5_ARMS = ["A0_control", "A1_const", "A2_primary", "A4_no_geometry", "A3_oracle"]
E5_LABEL = {"A0_control": "A0 control (base card, no correction)",
            "A1_const": "A1 per-family constant delta (LODO)",
            "A2_primary": "A2 shared MLP delta (LODO, primary)",
            "A4_no_geometry": "A4 A2 without log L, log W (diagnostic)",
            "A3_oracle": "A3 in-sample on the held-out device (oracle, not a method)"}
E5_SHORT = {"A0_control": "A0 control", "A1_const": "A1 const", "A2_primary": "A2 MLP",
            "A4_no_geometry": "A4 no geometry", "A3_oracle": "A3 oracle"}
E5_COLOR = {"A0_control": INK, "A1_const": GREY, "A2_primary": BLUE,
            "A4_no_geometry": ORANGE, "A3_oracle": AQUA}
E5_MARK = {"A0_control": "o", "A1_const": "s", "A2_primary": "D", "A4_no_geometry": "v",
           "A3_oracle": "^"}
BASE_LABEL = {"b43": "43-parameter base (direct43_r3_fd, primary)",
              "b7": "seven-parameter base (direct7_fd, secondary)"}


# ================================================================== E4 p7

def e4_summaries() -> dict:
    out = {}
    for run in E4_RUNS:
        for s in E4_SPLITS:
            d = {"strict": load(E4 / run / s / "summary_strict.json")}
            f = E4 / run / s / "summary_feasible.json"
            if f.exists():
                d["feasible"] = load(f)
            out[(run, s)] = d
    return out


def feasible_equals_strict(summ: dict) -> bool:
    if "feasible" not in summ:
        return False
    a, b = summ["strict"]["methods"], summ["feasible"]["methods"]
    for m in a:
        for t, r in a[m]["per_device"].items():
            for k, v in r.items():
                if isinstance(v, (int, float)):
                    w = b[m]["per_device"][t][k]
                    if not (v == w or (math.isnan(v) and math.isnan(w))):
                        return False
    return True


def e4_decisions(summ: dict) -> dict:
    import e4_analyze as ea  # read-only: decide() reads prereg.json and the summary
    dec = {}
    for key, d in summ.items():
        r = ea.decide(d["strict"])
        stored = d["strict"].get("decision")
        if stored is not None:
            assert r["verdict"] == stored["verdict"]
            for k, v in r["per_method"].items():
                assert (v["C1"], v["C2"], v["C3"]) == (stored["per_method"][k]["C1"],
                                                       stored["per_method"][k]["C2"],
                                                       stored["per_method"][k]["C3"])
        dec[key] = r
    return dec


def table_e4p7_decision(dec: dict) -> rf.Table:
    rows = []
    for s in E4_SPLITS:
        d = dec[("p7", s)]
        single = d["single_heldout_rrms"]
        for m in E4_CAND:
            v = d["per_method"][m]
            pc = v["paired_vs_crossinterp"]
            rows.append({
                "split": f"p7 {s}", "m": m,
                "cov": v["coverage_0.9"], "w": v["width_0.9"], "wl": 4 * single, "c1": yn(v["C1"]),
                "rr": v["heldout_rrms"], "rl": 0.95 * single,
                "wins": f"{v['wins']}/{v['n_valid']}", "c2": yn(v["C2"]),
                "c3": (yn(v["C3"]) + f" ({pc[0]:.3f} vs {pc[1]:.3f})") if m == "learned" else "—",
                "all": yn(v["C1"] and v["C2"] and v["C3"]),
                "verdict": d["verdict"] if m == E4_CAND[0] else "",
            })
    d43 = dec[("p43", "vd")]
    rows.append({"split": "p43 vd (decision)", "m": "ens", "cov": d43["per_method"]["ens"]["coverage_0.9"],
                 "w": d43["per_method"]["ens"]["width_0.9"], "wl": 4 * d43["single_heldout_rrms"],
                 "c1": yn(d43["per_method"]["ens"]["C1"]),
                 "rr": d43["per_method"]["ens"]["heldout_rrms"],
                 "rl": 0.95 * d43["single_heldout_rrms"],
                 "wins": f"{d43['per_method']['ens']['wins']}/12",
                 "c2": yn(d43["per_method"]["ens"]["C2"]), "c3": "—", "all": "no",
                 "verdict": d43["verdict"]})
    cols = [
        {"key": "split", "head": "run, split", "md": "run, split", "kind": "text"},
        {"key": "m", "head": "candidate", "md": "candidate", "kind": "text"},
        {"key": "cov", "head": "cov. 0.9", "group": "C1 calibration", "md": "coverage of 90% interval", "kind": "rrms"},
        {"key": "w", "head": "width 0.9", "group": "C1 calibration", "md": "width of 90% interval", "kind": "rrms"},
        {"key": "wl", "head": "4 × single", "group": "C1 calibration", "md": "width limit 4 × single RRMS", "kind": "rrms"},
        {"key": "c1", "head": "C1", "group": "C1 calibration", "md": "C1", "kind": "rtext"},
        {"key": "rr", "head": "held-out", "group": "C2 accuracy", "md": "held-out RRMS", "kind": "rrms"},
        {"key": "rl", "head": "0.95 × single", "group": "C2 accuracy", "md": "limit 0.95 × single", "kind": "rrms"},
        {"key": "wins", "head": "< single", "group": "C2 accuracy", "md": "devices below single", "kind": "rtext"},
        {"key": "c2", "head": "C2", "group": "C2 accuracy", "md": "C2", "kind": "rtext"},
        {"key": "c3", "head": "C3 (vs crossinterp)", "md": "C3 learned <= crossinterp", "kind": "rtext"},
        {"key": "all", "head": "all", "md": "C1 and C2 and C3", "kind": "rtext"},
        {"key": "verdict", "head": "rule output", "md": "rule output for the split", "kind": "rtext"},
    ]
    notes = [
        "Pre-registered rule (prereg.json, amendments 1 and 2): C1 = all-device mean coverage of "
        "the central 90% interval in [0.70, 0.95] AND all-device mean width <= 4 × single's "
        "held-out RRMS; C2 = held-out RRMS <= 0.95 × single (valid-only and paired) AND lower than "
        "single on >= 8 of 12 devices; C3 (learned only) = held-out RRMS <= crossinterp's "
        "(valid-only and paired). CONTINUE if a candidate meets C1, C2 and C3; "
        "PARTIAL_CALIBRATION_ONLY if no candidate meets all but one meets C1; KILL if none meets C1.",
        "p7 = seven-parameter box (out/cold_loop/seven/round_1/spec.json). prereg.json lists p7 vd "
        "and p7 vg as secondary results without a decision; sparse is secondary by amendment 1. "
        "The rule outputs in the p7 rows come from the same decide() function applied to the "
        "stored strict summaries (queue jobs E4_p7_vd, E4_p7_vg, E4_p7_sparse) and do not change "
        "the E4 decision.",
        "Last row: the E4 decision (p43 vd, stored decision block of queue job E4_p43_vd), "
        "repeated for reference; its only C1 candidate is ens.",
        f"single held-out RRMS (all-device), p7: vd {dec[('p7', 'vd')]['single_heldout_rrms']:.4f}, "
        f"sparse {dec[('p7', 'sparse')]['single_heldout_rrms']:.4f}, vg "
        f"{dec[('p7', 'vg')]['single_heldout_rrms']:.4f}. Width is mean (hi − lo) / den, in the "
        "same units as the per-curve RRMS.",
    ]
    return rf.Table("01_table_e4p7_decision",
                    "E4 seven-parameter runs (secondary): pre-registered checks per candidate and "
                    "split, with the p43 vd decision for reference", cols, rows, notes)


YMAX = 300.0


def fig_e4p7_per_device(summ: dict, path: Path) -> list[str]:
    tags = real_dev()
    x = np.arange(len(tags))
    ms = ["single", "ens", "laplace", "learned", "crossinterp"]
    off = dict(zip(ms, np.linspace(-0.28, 0.28, len(ms))))
    mk = {"single": "o", "ens": "D", "laplace": "s", "learned": "v", "crossinterp": "^"}
    fig, axes = plt.subplots(3, 1, figsize=(15, 12.5), dpi=DPI, sharex=True)
    fig.patch.set_facecolor(BG)
    for ax, s in zip(axes, E4_SPLITS):
        style(ax)
        meth = summ[("p7", s)]["strict"]["methods"]
        for m in ms:
            pd_ = {t: r["heldout_rrms"] for t, r in meth[m]["per_device"].items()}
            st = meth[m]["heldout_rrms"]
            med = fam(pd_)["median"]
            vals = np.array([fnum(pd_.get(t)) for t in tags])
            shown = np.minimum(vals, YMAX * 0.7)
            ax.scatter(x + off[m], shown, marker=mk[m], s=30, color=E4_COLOR[m], zorder=3,
                       edgecolor=BG, lw=0.5,
                       label=f"{E4_LABEL[m]}: all-device {st['all_device_mean']:.4g}, combined "
                             f"{st['combined']:.4g}, median {med:.3g}"
                             + (f", < single on {meth[m]['wins_vs_single']}/12" if m != "single" else ""))
            for xi, v in zip(x + off[m], vals):
                if np.isfinite(v) and v > YMAX * 0.7:
                    ax.annotate(f"{m} {v:.3g}", (xi, YMAX * 0.7), xytext=(7, -3),
                                textcoords="offset points", ha="left", va="center",
                                fontsize=7, color=E4_COLOR[m])
        ref = {t: r["heldout_rrms"] for t, r in
               summ[("p43", s)]["strict"]["methods"]["single"]["per_device"].items()}
        rv = [fnum(ref.get(t)) for t in tags]
        ax.hlines(rv, x - 0.36, x + 0.36, color=GREY, lw=1.4, ls="--", zorder=2,
                  label=f"p43 single best-fit card (same split): all-device "
                        f"{np.nanmean(rv):.4g}")
        n_f = sum(1 for t in tags if not np.isfinite(fnum(meth['single']['per_device'][t]['heldout_rrms'])))
        ax.set_yscale("log")
        ax.set_ylim(0.005, YMAX)
        nm = sum(t.startswith("nmos") for t in tags)
        ax.axvline(nm - 0.5, color=GREY, lw=0.8, ls=":")
        ax.set_ylabel("held-out paper RRMS", fontsize=9, color=INK)
        ax.set_title(f"p7, split {s} (secondary; failed single-card devices: {n_f}); values "
                     f"above {YMAX * 0.7:.0f} drawn at {YMAX * 0.7:.0f} and labeled",
                     fontsize=9.5, color=INK)
        ax.legend(fontsize=7.3, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([rf.device_label(t) + (" n" if t.startswith("nmos") else " p")
                              for t in tags], rotation=30, ha="right", fontsize=8)
    fig.suptitle("E4, seven parameters, REAL_DEV: held-out-curve RRMS per device and method "
                 "(n = nMOS, p = pMOS; dashed grey: 43-parameter single card)", fontsize=10,
                 color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(E4 / "p7" / s / "summary_strict.json") for s in E4_SPLITS] + \
        [rel(E4 / "p43" / s / "summary_strict.json") for s in E4_SPLITS]


def table_e4p7_results(summ: dict) -> rf.Table:
    rows = []
    for s in E4_SPLITS:
        meth = summ[("p7", s)]["strict"]["methods"]
        for m in E4_METHODS:
            e = meth[m]
            h = e["heldout_rrms"]
            med = fam({t: r["heldout_rrms"] for t, r in e["per_device"].items()})["median"]
            g = lambda k: e.get(k, {}).get("all_device_mean")  # noqa: E731
            rows.append({"split": s, "m": m, "all": h["all_device_mean"],
                         "combined": h["combined"], "nmos": h["nmos"], "pmos": h["pmos"],
                         "median": med, "n": f"{h['n_valid']}/{h['n_devices']}",
                         "wins": e.get("wins_vs_single"), "c50": g("coverage_0.5"),
                         "c90": g("coverage_0.9"), "w90": g("width_0.9"),
                         "is90": g("interval_score_0.9")})
    cols = [
        {"key": "split", "head": "split", "md": "split", "kind": "text"},
        {"key": "m", "head": "method", "md": "method", "kind": "text"},
        {"key": "all", "head": "all-device", "group": "held-out RRMS, 12 devices", "md": "all-device", "kind": "rrms"},
        {"key": "combined", "head": "combined", "group": "held-out RRMS, 12 devices", "md": "combined", "kind": "rrms"},
        {"key": "nmos", "head": "nMOS", "group": "held-out RRMS, 12 devices", "md": "nMOS", "kind": "rrms", "family": True},
        {"key": "pmos", "head": "pMOS", "group": "held-out RRMS, 12 devices", "md": "pMOS", "kind": "rrms", "family": True},
        {"key": "median", "head": "median", "group": "held-out RRMS, 12 devices", "md": "median", "kind": "rrms"},
        {"key": "n", "head": "finite", "group": "held-out RRMS, 12 devices", "md": "finite devices", "kind": "rtext"},
        {"key": "wins", "head": "< single", "md": "devices below single", "kind": "int"},
        {"key": "c50", "head": "cov 0.5", "group": "intervals", "md": "coverage 0.5", "kind": "rrms"},
        {"key": "c90", "head": "cov 0.9", "group": "intervals", "md": "coverage 0.9", "kind": "rrms"},
        {"key": "w90", "head": "width 0.9", "group": "intervals", "md": "width 0.9", "kind": "rrms"},
        {"key": "is90", "head": "IS 0.9", "group": "intervals", "md": "interval score 0.9", "kind": "rrms"},
    ]
    same = all(feasible_equals_strict(summ[("p7", s)]) for s in E4_SPLITS)
    notes = [
        "all-device = mean over the 12 REAL_DEV devices (strict: a non-finite held-out prediction "
        "makes the device NaN; finite = devices with a finite value; 0 single-card failures in all "
        "three p7 splits); combined = mean of the nMOS (5) and pMOS (7) family means.",
        "Held-out curves: vd = Id-Vd curves except the highest gate bias; sparse = every included "
        "curve except Id-Vg at Vd 0.01/1.85 and Id-Vd at the highest gate bias; vg = Id-Vg at "
        "Vd 0.37/0.74/1.11/1.48.",
        "select feasible (amendment 1, secondary): `e4_analyze.py --run p7 --select feasible` was "
        "run for this report on the fitted records (no NGSpice). "
        + ("Every per-device value equals the strict one on all three p7 splits, so only strict "
           "rows are listed." if same else "Feasible values differ from strict on at least one "
           "split; see summary_feasible.json."),
        "conformal and crossinterp_conformal reuse the single and crossinterp point predictions; "
        "ens_env and laplace_param are reported in the JSON summaries.",
    ]
    return rf.Table("03_table_e4p7_results",
                    "E4 held-out-curve RRMS and interval metrics per method, seven parameters, "
                    "splits vd, sparse, vg", cols, rows, notes)


def table_e4_p43_vs_p7(summ: dict, dec: dict) -> rf.Table:
    rows = []
    for s in E4_SPLITS:
        for run in E4_RUNS:
            st = summ[(run, s)]["strict"]
            meth = st["methods"]
            g = lambda m: meth[m]["heldout_rrms"]  # noqa: E731
            sp = st["spearman_heldout_rrms_vs_sloppy_frac"]
            rows.append({"split": s, "run": run + (" (decision)" if (run, s) == ("p43", "vd") else ""),
                         "train": st["train_obj_mean"],
                         "single": g("single")["all_device_mean"],
                         "single_c": g("single")["combined"],
                         "ens": g("ens")["all_device_mean"],
                         "laplace": g("laplace")["all_device_mean"],
                         "learned": g("learned")["all_device_mean"],
                         "cross": g("crossinterp")["all_device_mean"],
                         "cross_c": g("crossinterp")["combined"],
                         "lw": meth["learned"]["wins_vs_single"],
                         "rho": f"{sp['rho']:.2f} (p {sp['p']:.3f})",
                         "verdict": dec[(run, s)]["verdict"]})
    cols = [
        {"key": "split", "head": "split", "md": "split", "kind": "text"},
        {"key": "run", "head": "run", "md": "run", "kind": "text"},
        {"key": "train", "head": "train obj.", "md": "mean train objective (single)", "kind": "rrms"},
        {"key": "single", "head": "single", "group": "held-out all-device", "md": "single all-device", "kind": "rrms"},
        {"key": "ens", "head": "ens", "group": "held-out all-device", "md": "ens all-device", "kind": "rrms"},
        {"key": "laplace", "head": "Laplace", "group": "held-out all-device", "md": "Laplace all-device", "kind": "rrms"},
        {"key": "learned", "head": "learned", "group": "held-out all-device", "md": "learned all-device", "kind": "rrms"},
        {"key": "cross", "head": "crossinterp", "group": "held-out all-device", "md": "crossinterp all-device", "kind": "rrms"},
        {"key": "single_c", "head": "single", "group": "combined", "md": "single combined", "kind": "rrms", "family": True},
        {"key": "cross_c", "head": "crossinterp", "group": "combined", "md": "crossinterp combined", "kind": "rrms", "family": True},
        {"key": "lw", "head": "learned < single", "md": "learned below single (devices)", "kind": "int"},
        {"key": "rho", "head": "Spearman (sloppy)", "md": "Spearman rho, held-out RRMS vs sloppy fraction", "kind": "rtext"},
        {"key": "verdict", "head": "rule output", "md": "rule output", "kind": "rtext"},
    ]
    notes = [
        "Strict selection, 12 REAL_DEV devices, 0 failed single-card devices in every run. "
        "train obj. = mean over devices of the single card's train objective (mean train-curve "
        "RRMS plus the held-out-bias feasibility penalty, which is 0 for every single card).",
        "Spearman = rank correlation across the 12 devices between the single card's held-out "
        "RRMS and the fraction of held-out sensitivity in the sloppy subspace (s / s_max < 1e-2); "
        "pre-registered as reported without a criterion.",
        "Only p43 vd is the pre-registered decision; the other five rule outputs are reported "
        "without a decision.",
    ]
    return rf.Table("04_table_e4_p43_vs_p7",
                    "E4 43 vs seven parameters per split: held-out RRMS of the single card, the "
                    "candidates and the crossinterp control", cols, rows, notes)


# ================================================================== E5

def e5_decide(a0: dict, arm: dict, tags: list[str], two_way: bool = False) -> dict:
    """The pre-registered stage-1 rule (stage 2: two_way, CONTINUE or KILL)."""
    m0 = float(np.mean([a0[t] for t in tags]))
    m1 = float(np.mean([arm[t] for t in tags]))
    red = (m0 - m1) / m0
    wins = int(sum(bool(arm[t] < a0[t]) for t in tags))
    cont = red >= 0.15 and wins >= 8
    kill = red < 0.05 or wins <= 6
    if cont:
        v = "CONTINUE"
    elif two_way or kill:
        v = "KILL"
    else:
        v = "AMBIGUOUS"
    return {"m0": m0, "m1": m1, "red": red, "wins": wins, "cont": cont, "kill": kill,
            "verdict": v}


def e5_stage2_per_device(s2: dict) -> tuple[dict, dict, dict]:
    ctl = {t: r["rrms"] for t, r in s2["control_polish"].items()}
    joint = {t: r["rrms"] for t, r in s2["joint_refit"].items()}
    unc = {t: r["uncorrected_rrms_of_refit_card"] for t, r in s2["joint_refit"].items()}
    return ctl, joint, unc


def table_e5_decision(s1: dict, s2: dict, light: dict) -> rf.Table:
    tags = s1["tags"]
    rows = []
    for b in ("b43", "b7"):
        pd_ = s1["bases"][b]["per_device"]
        for arm in ("A2_primary", "A1_const", "A4_no_geometry", "A3_oracle"):
            d = e5_decide(pd_["A0_control"], pd_[arm], tags)
            stored = s1["bases"][b]["vs_control"][arm]
            assert d["verdict"] == stored["verdict"] and d["wins"] == stored["devices_improved"]
            assert abs(d["red"] - stored["relative_reduction"]) < 1e-12
            primary = (b, arm) == ("b43", "A2_primary")
            rows.append({"stage": "stage 1", "base": b, "arm": E5_SHORT[arm] + " vs A0",
                         "m0": d["m0"], "m1": d["m1"], "red": d["red"], "wins": f"{d['wins']}/12",
                         "cont": yn(d["cont"]), "kill": "met" if d["kill"] else "not met",
                         "verdict": d["verdict"],
                         "role": "decision" if primary else ("oracle, not a method" if arm == "A3_oracle"
                                                             else "reported, no decision")})
    assert s1["primary_verdict_stage1"]["verdict"] == rows[0]["verdict"]
    lp = light["bases"]["b43"]["vs_control"]["A2_primary"]
    rows.insert(1, {"stage": "stage 1 light", "base": "b43", "arm": "A2 MLP vs A0",
                    "m0": lp["control_mean"], "m1": lp["arm_mean"], "red": lp["relative_reduction"],
                    "wins": f"{lp['devices_improved']}/12", "cont": "—", "kill": "—",
                    "verdict": lp["verdict"], "role": "light settings, not a decision"})
    ctl, joint, _ = e5_stage2_per_device(s2)
    d = e5_decide(ctl, joint, tags, two_way=True)
    st = s2["verdict_stage2"]
    assert d["wins"] == st["devices_improved"] and abs(d["red"] - st["relative_reduction"]) < 1e-12
    rows.append({"stage": "stage 2", "base": "b43", "arm": "joint refit vs control polish",
                 "m0": d["m0"], "m1": d["m1"], "red": d["red"], "wins": f"{d['wins']}/12",
                 "cont": yn(d["cont"]), "kill": "—", "verdict": d["verdict"],
                 "role": "diagnostic (stage 1 not AMBIGUOUS)"})
    cols = [
        {"key": "stage", "head": "stage", "md": "stage", "kind": "text"},
        {"key": "base", "head": "base", "md": "base card", "kind": "text"},
        {"key": "arm", "head": "comparison", "md": "comparison", "kind": "text"},
        {"key": "m0", "head": "control", "group": "all-device RRMS", "md": "control all-device", "kind": "rrms"},
        {"key": "m1", "head": "arm", "group": "all-device RRMS", "md": "arm all-device", "kind": "rrms"},
        {"key": "red", "head": "reduction", "md": "(control − arm) / control", "kind": "rrms"},
        {"key": "wins", "head": "improved", "md": "devices improved", "kind": "rtext"},
        {"key": "cont", "head": ">= 0.15 and >= 8", "group": "rule", "md": "continue condition", "kind": "rtext"},
        {"key": "kill", "head": "< 0.05 or <= 6", "group": "rule", "md": "kill condition (stage 1)", "kind": "rtext"},
        {"key": "verdict", "head": "rule output", "group": "rule", "md": "rule output", "kind": "rtext"},
        {"key": "role", "head": "role", "md": "role", "kind": "text"},
    ]
    notes = [
        "Pre-registered rule (out/research/explore/E5/prereg.json): stage 1 CONTINUE if the "
        "all-DEV mean RRMS reduction (A0 − A2) / A0 >= 0.15 AND A2 < A0 on >= 8 of 12 devices; "
        "KILL if reduction < 0.05 OR A2 < A0 on <= 6 of 12; otherwise AMBIGUOUS, decided on "
        "stage 2 (CONTINUE if (control − joint) / control >= 0.15 AND joint < control on >= 8 of "
        "12, otherwise KILL). The decision is the first row (b43, A2 vs A0).",
        "Metric: frozen-inclusion paper RRMS of the corrected curves on the held-out device, "
        "REAL_DEV leave-one-device-out, 0 failed devices in every arm. Rule outputs for the other "
        "arms and for the seven-parameter base are computed with the same function for reporting.",
        "Stage-1 run (queue job E5_stage1, 2026-10-19 14:19-14:29, 625 s): 3000 full-batch steps, "
        "seeds 0-4. Recorded deviations from the prereg text: optimizer torch AdamW (decoupled "
        "weight decay 1e-4); weights renormalized to sum 1 per device; MPS float32. Light row: "
        "800 steps, seeds 0-2 (commit 773abe3).",
        "Stage 2 (queue job E5_stage2, 64 s): per fold, bounded TRF polish of the 43-parameter card "
        "(max_nfev 400, diff_step 0.01, irls 2) with the fold's A2 delta frozen in the residual; "
        "control = the same polish without delta from the same start. Run as a diagnostic because "
        "the stage-1 output is KILL.",
    ]
    return rf.Table("05_table_e5_decision",
                    "E5 decision: shared discrepancy learned on other devices vs no correction, "
                    "pre-registered stage-1 and stage-2 checks", cols, rows, notes)


def fig_e5_per_device(s1: dict, s2: dict, path: Path) -> list[str]:
    tags = real_dev()
    assert sorted(tags) == sorted(s1["tags"])
    x = np.arange(len(tags))
    off = dict(zip(E5_ARMS, np.linspace(-0.28, 0.28, len(E5_ARMS))))
    fig, axes = plt.subplots(3, 1, figsize=(15, 12.5), dpi=DPI, sharex=True)
    fig.patch.set_facecolor(BG)
    for ax, b in zip(axes[:2], ("b43", "b7")):
        style(ax)
        pd_ = s1["bases"][b]["per_device"]
        for arm in E5_ARMS:
            v = pd_[arm]
            f = fam(v)
            wins = sum(v[t] < pd_["A0_control"][t] for t in tags)
            lab = (f"{E5_LABEL[arm]}: all-device {f['all']:.4f}, combined {f['combined']:.4f}, "
                   f"median {f['median']:.4f}" + (f", < A0 on {wins}/12" if arm != "A0_control" else ""))
            ax.scatter(x + off[arm], [v[t] for t in tags], marker=E5_MARK[arm], s=32,
                       color=E5_COLOR[arm], zorder=3, edgecolor=BG, lw=0.5, label=lab)
        ax.set_yscale("log")
        ax.set_ylim(0.005, 2.0)
        ax.set_ylabel("paper RRMS, held-out device", fontsize=9, color=INK)
        red = s1["bases"][b]["vs_control"]["A2_primary"]["relative_reduction"]
        ax.set_title(f"stage 1, {BASE_LABEL[b]}: A2 reduction vs A0 = {red:.3f} (continue needs "
                     ">= 0.15)", fontsize=9.5, color=INK)
        ax.legend(fontsize=7.3, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax = axes[2]
    style(ax)
    ctl, joint, unc = e5_stage2_per_device(s2)
    a2 = s1["bases"]["b43"]["per_device"]["A2_primary"]
    series = [("control polish (no delta)", ctl, INK, "o", -0.2),
              ("stage-1 A2 (start of the joint refit)", a2, BLUE, "D", -0.07),
              ("joint refit, corrected curves", joint, VIOLET, "s", 0.07),
              ("joint refit card without delta (diagnostic)", unc, GREY, "x", 0.2)]
    for lab, v, c, m, o in series:
        f = fam(v)
        extra = f", < control on {sum(v[t] < ctl[t] for t in tags)}/12" if v is not ctl else ""
        ax.scatter(x + o, [v[t] for t in tags], marker=m, s=32, color=c, zorder=3,
                   edgecolor=BG if m != "x" else None, lw=0.5 if m != "x" else 1.3,
                   label=f"{lab}: all-device {f['all']:.4f}, combined {f['combined']:.4f}{extra}")
    ax.set_yscale("log")
    ax.set_ylim(0.005, 2.0)
    ax.set_ylabel("paper RRMS, held-out device", fontsize=9, color=INK)
    red = s2["verdict_stage2"]["relative_reduction"]
    ax.set_title(f"stage 2 (diagnostic), 43-parameter base: joint refit reduction vs control "
                 f"polish = {red:.3f}", fontsize=9.5, color=INK)
    ax.legend(fontsize=7.3, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    nm = sum(t.startswith("nmos") for t in tags)
    for a in axes:
        a.axvline(nm - 0.5, color=GREY, lw=0.8, ls=":")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([rf.device_label(t) + (" n" if t.startswith("nmos") else " p")
                              for t in tags], rotation=30, ha="right", fontsize=8)
    fig.suptitle("E5, REAL_DEV leave-one-device-out: paper RRMS per held-out device and arm "
                 "(n = nMOS, p = pMOS; 0 failed devices)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(E5_S1), rel(E5_S2)]


def table_e5_results(s1: dict, s2: dict) -> rf.Table:
    tags = s1["tags"]
    rows = []
    for b in ("b43", "b7"):
        pd_ = s1["bases"][b]["per_device"]
        agg = s1["bases"][b]["aggregate"]
        for arm in E5_ARMS:
            f = fam(pd_[arm])
            assert abs(f["all"] - agg[arm]["all_dev_mean"]) < 1e-12
            rows.append({"stage": "stage 1", "base": b, "arm": E5_SHORT[arm], "all": f["all"],
                         "combined": f["combined"], "nmos": f["nmos"], "pmos": f["pmos"],
                         "median": f["median"],
                         "wins": None if arm == "A0_control" else
                         int(sum(pd_[arm][t] < pd_["A0_control"][t] for t in tags)),
                         "failed": f"{agg[arm]['n_failed']}/{agg[arm]['n']}", "nsim": None})
    ctl, joint, unc = e5_stage2_per_device(s2)
    for lab, v, key in (("control polish", ctl, "control_polish"),
                        ("joint refit (corrected)", joint, "joint_refit"),
                        ("joint refit card, no delta", unc, "joint_refit")):
        f = fam(v)
        rows.append({"stage": "stage 2", "base": "b43", "arm": lab, "all": f["all"],
                     "combined": f["combined"], "nmos": f["nmos"], "pmos": f["pmos"],
                     "median": f["median"],
                     "wins": None if v is ctl else int(sum(v[t] < ctl[t] for t in tags)),
                     "failed": "0/12",
                     "nsim": float(np.mean([s2[key][t]["n_sim"] for t in tags]))
                     if lab != "joint refit card, no delta" else None})
    cols = [
        {"key": "stage", "head": "stage", "md": "stage", "kind": "text"},
        {"key": "base", "head": "base", "md": "base card", "kind": "text"},
        {"key": "arm", "head": "arm", "md": "arm", "kind": "text"},
        {"key": "all", "head": "all-device", "group": "RRMS, 12 REAL_DEV devices", "md": "all-device", "kind": "rrms"},
        {"key": "combined", "head": "combined", "group": "RRMS, 12 REAL_DEV devices", "md": "combined", "kind": "rrms"},
        {"key": "nmos", "head": "nMOS", "group": "RRMS, 12 REAL_DEV devices", "md": "nMOS", "kind": "rrms", "family": True},
        {"key": "pmos", "head": "pMOS", "group": "RRMS, 12 REAL_DEV devices", "md": "pMOS", "kind": "rrms", "family": True},
        {"key": "median", "head": "median", "group": "RRMS, 12 REAL_DEV devices", "md": "median", "kind": "rrms"},
        {"key": "wins", "head": "< control", "md": "devices below the control", "kind": "int"},
        {"key": "failed", "head": "failed", "md": "failed devices", "kind": "rtext"},
        {"key": "nsim", "head": "NGSpice calls", "md": "mean NGSpice calls per device", "kind": "int"},
    ]
    notes = [
        "all-device = mean over the 12 REAL_DEV devices; combined = mean of the nMOS (5) and pMOS "
        "(7) family means; strict frozen-inclusion paper RRMS; a device with a non-finite included "
        "curve counts as failed (none in this run).",
        "Base cards: b43 = out/research/polished/direct43_r3_fd/real/summary.json (direct NGSpice "
        "43 parameters, round 3, + FD); b7 = out/research/polished/direct7_fd/real/summary.json. "
        "A0 reproduces the stored RRMS of both (max abs difference 0).",
        "< control: stage 1 against A0; stage 2 against the control polish. A3 is trained "
        "in-sample on the held-out device (upper bound for the network class, not a method).",
        "Stage-2 'card, no delta' = the joint-refit card scored without the correction "
        "(diagnostic).",
    ]
    return rf.Table("07_table_e5_results",
                    "E5 RRMS per base card and arm (stage 1) and for the stage-2 refits", cols,
                    rows, notes)


# ================================================================== main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    topic = "explore_e4p7_e5"
    folder = rf.REPORT_ROOT / f"{args.date}_{topic}"
    folder.mkdir(parents=True, exist_ok=True)
    items = []

    # ---- E4 p7
    e4 = e4_summaries()
    dec = e4_decisions(e4)
    assert e4[("p43", "vd")]["strict"]["decision"]["verdict"] == dec[("p43", "vd")]["verdict"]
    t1 = table_e4p7_decision(dec)
    rf.render_table(t1, folder / "01_table_e4p7_decision.png")
    items.append(("01_table_e4p7_decision.png",
                  "Table of the E4 pre-registered checks C1, C2, C3 for the candidates ens, "
                  "laplace and learned on the seven-parameter splits vd, sparse and vg (secondary, "
                  "rule output for reporting only), with the p43 vd decision row for reference",
                  [rel(E4 / "prereg.json"), rel(E4 / "prereg_amendment_1.json"),
                   rel(E4 / "prereg_amendment_2.json")] +
                  [rel(E4 / "p7" / s / "summary_strict.json") for s in E4_SPLITS] +
                  [rel(E4 / "p43" / "vd" / "summary_strict.json")]))
    src = fig_e4p7_per_device(e4, folder / "02_e4p7_per_device.png")
    items.append(("02_e4p7_per_device.png",
                  "Per-device held-out RRMS on REAL_DEV for the seven-parameter single card, "
                  "ensemble, Laplace, learned discrepancy and the crossinterp control, one panel "
                  "per split (vd, sparse, vg), with the 43-parameter single card as a dashed grey "
                  "reference; log scale, values above 210 drawn at 210 and labeled", src))
    t3 = table_e4p7_results(e4)
    rf.render_table(t3, folder / "03_table_e4p7_results.png")
    items.append(("03_table_e4p7_results.png",
                  "Table of E4 seven-parameter held-out RRMS (all-device, combined, nMOS, pMOS, "
                  "median), devices below single and interval metrics (coverage 0.5 and 0.9, "
                  "width, interval score) per split and method",
                  [rel(E4 / "p7" / s / f"summary_{k}.json") for s in E4_SPLITS
                   for k in ("strict", "feasible")]))
    t4 = table_e4_p43_vs_p7(e4, dec)
    rf.render_table(t4, folder / "04_table_e4_p43_vs_p7.png")
    items.append(("04_table_e4_p43_vs_p7.png",
                  "Table comparing the 43- and seven-parameter runs per split: mean train "
                  "objective, held-out all-device RRMS of single, ens, Laplace, learned and "
                  "crossinterp, combined for single and crossinterp, learned wins, Spearman "
                  "correlation with the sloppy fraction, rule output",
                  [rel(E4 / r / s / "summary_strict.json") for r in E4_RUNS for s in E4_SPLITS]))

    # ---- E5
    s1, s2, light = load(E5_S1), load(E5_S2), load(E5_LIGHT)
    assert not s1["light"] and not s2["light"]
    t5 = table_e5_decision(s1, s2, light)
    rf.render_table(t5, folder / "05_table_e5_decision.png")
    items.append(("05_table_e5_decision.png",
                  "Table of the E5 pre-registered stage-1 check (b43, A2 vs A0: reduction, devices "
                  "improved, continue and kill conditions, rule output), the light-run row, the "
                  "same rule applied to the other arms and the seven-parameter base for reporting, "
                  "and the stage-2 diagnostic (joint refit vs control polish)",
                  [rel(E5 / "prereg.json"), rel(E5_S1), rel(E5_LIGHT), rel(E5_S2)]))
    src = fig_e5_per_device(s1, s2, folder / "06_e5_per_device.png")
    items.append(("06_e5_per_device.png",
                  "Per-device RRMS of the held-out REAL_DEV device for arms A0, A1, A2, A4 and the "
                  "A3 oracle on the 43-parameter base (top) and the seven-parameter base (middle), "
                  "and for stage 2 (bottom): control polish, stage-1 A2, joint refit with the "
                  "correction and the joint-refit card without the correction; log scale", src))
    t7 = table_e5_results(s1, s2)
    rf.render_table(t7, folder / "07_table_e5_results.png")
    items.append(("07_table_e5_results.png",
                  "Table of E5 RRMS per base card and arm (all-device, combined, nMOS, pMOS, "
                  "median, devices below the control, failed devices) and for the stage-2 refits "
                  "with mean NGSpice calls per device", [rel(E5_S1), rel(E5_S2)]))

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = (f"`PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_e4p7_e5.py "
           f"--date {args.date}`")
    tables_md = [f"# Tables, {args.date} {topic}", "",
                 f"Generated {now} by {gen} from result files only. RRMS is the paper metric "
                 "with frozen published-baseline curve inclusion, strict.", ""]
    for t in (t1, t3, t4, t5, t7):
        tables_md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(tables_md))

    readme = [f"# Report batch {args.date} · {topic}", "",
              f"Generated {now} by {gen} from result files only (no simulation, no training). "
              "`e4_analyze.py --run p7 --select feasible` (fitted records only, no NGSpice) was "
              "run for this report on vd, sparse and vg.", "",
              "## Contents", "",
              "- E4 seven-parameter runs (p7), splits vd, sparse, vg (queue jobs E4_p7_vd, "
              "E4_p7_vg, E4_p7_sparse, 2026-10-19 13:35-14:19). All secondary: prereg.json lists "
              "p7 vd and p7 vg as reported without a decision; sparse is secondary by amendment 1. "
              f"Pre-registration: `{rel(E4 / 'prereg.json')}` and amendments 1 and 2. The E4 "
              "decision (p43 vd) is in `2026-10-19_explore_e2_e3_e4`.",
              "- E5 stage 1 decision run and stage 2 diagnostic (queue jobs E5_stage1, E5_stage2, "
              f"2026-10-19 14:19-14:30). Pre-registration: `{rel(E5 / 'prereg.json')}`.", "",
              "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{s}`" for s in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))

    p7 = {s: e4[("p7", s)]["strict"]["methods"] for s in E4_SPLITS}
    a = lambda s, m: p7[s][m]["heldout_rrms"]["all_device_mean"]  # noqa: E731
    v = dec[("p7", "vd")]["per_method"]
    g = dec[("p7", "vg")]["per_method"]
    b43 = s1["bases"]["b43"]["aggregate"]
    pv = s1["primary_verdict_stage1"]
    summary = [
        f"E4 p7 (secondary), held-out all-device: vd single {a('vd', 'single'):.4f}, ens "
        f"{a('vd', 'ens'):.4f}, learned {a('vd', 'learned'):.4f} (< single on "
        f"{v['learned']['wins']}/12, C2 yes; coverage {v['learned']['coverage_0.9']:.3f}, C1 no; "
        f"C3 no vs crossinterp {a('vd', 'crossinterp'):.4f}); ens coverage "
        f"{v['ens']['coverage_0.9']:.3f} (C1 no): rule output {dec[('p7', 'vd')]['verdict']}. "
        f"vg single {a('vg', 'single'):.4f}, ens and Laplace C1 yes (coverage "
        f"{g['ens']['coverage_0.9']:.3f}, {g['laplace']['coverage_0.9']:.3f}), C2 no: "
        f"{dec[('p7', 'vg')]['verdict']}. sparse single {a('sparse', 'single'):.4f}: "
        f"{dec[('p7', 'sparse')]['verdict']}. The E4 decision (p43 vd, "
        f"{dec[('p43', 'vd')]['verdict']}) is unchanged.",
        f"E4 p7 vs p43, single card held-out all-device: vd {a('vd', 'single'):.4f} vs "
        f"{e4[('p43', 'vd')]['strict']['methods']['single']['heldout_rrms']['all_device_mean']:.4f}, "
        f"sparse {a('sparse', 'single'):.4f} vs "
        f"{e4[('p43', 'sparse')]['strict']['methods']['single']['heldout_rrms']['all_device_mean']:.4f}, "
        f"vg {a('vg', 'single'):.4f} vs "
        f"{e4[('p43', 'vg')]['strict']['methods']['single']['heldout_rrms']['all_device_mean']:.4f}; "
        "crossinterp is lower than single on 12/12 devices in every p7 split.",
        f"E5 stage 1 (decision, 43-parameter base, LODO), all-device (combined): A0 "
        f"{b43['A0_control']['all_dev_mean']:.4f} ({b43['A0_control']['combined']:.4f}), A2 "
        f"{b43['A2_primary']['all_dev_mean']:.4f} ({b43['A2_primary']['combined']:.4f}); reduction "
        f"{pv['relative_reduction']:.3f}, A2 < A0 on {pv['devices_improved']}/12: KILL. A1 "
        f"{b43['A1_const']['all_dev_mean']:.4f}, A4 {b43['A4_no_geometry']['all_dev_mean']:.4f}, "
        f"A3 oracle {b43['A3_oracle']['all_dev_mean']:.4f}. Stage 2 (diagnostic): joint refit "
        f"{s2['aggregate']['joint']['all_dev_mean']:.4f} vs control polish "
        f"{s2['aggregate']['control']['all_dev_mean']:.4f}, reduction "
        f"{s2['verdict_stage2']['relative_reduction']:.3f}, {s2['verdict_stage2']['devices_improved']}/12.",
    ]
    pngs = [i[0] for i in items]
    rf.update_index(args.date, topic, folder, pngs, summary)
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
