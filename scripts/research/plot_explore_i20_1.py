#!/usr/bin/env python3
"""Report batch for I20-1: BSIM4 A2 clamp state of the NIST decks (audit and
fix) and the selected g v2 recipe on both NIST DEV datasets.

Reads stored files under ``out/research/explore/I20_1/`` (prereg, amendment
A1, audit, start, s3, base_<ds>, select_<ds>, lodo_<ds>, decision.json), the
I8-1 / I9-1 stored rows and the I19-1 committed TEST gate decisions.

NGSpice: the I-V figures re-simulate the stock card, the starting card in
states L and C, and the state-C held-out cards (C0_refit, M_bias v1, selected
g v2) of 3 devices per dataset; each RRMS is checked against the stored row.

  PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i20_1.py --date 2026-10-23
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_explore_e2_e3_e4 as pe  # noqa: E402  (helpers, style)
from plot_explore_e2_e3_e4 import (BG, INK, MUTED, GREY, BLUE, ORANGE, RED, VIOLET, AQUA, DPI,  # noqa: E402
                                   EXP, ROOT, load, rel, style)
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts/research/explore"))
import i20_1_nist_clamp as drv  # noqa: E402
from cryoml.explore import i20_1_nist_clamp as m  # noqa: E402
from cryoml.explore import i12_1_symm as v2  # noqa: E402
from cryoml.explore.i3_2_bias import correction_from_json, paired_ratio  # noqa: E402
from cryoml.explore.e1_correction import Correction, score_device  # noqa: E402
from cryoml.explore import i8_1_nist as b8  # noqa: E402

rf = pe.rf
plt = pe.plt

TOPIC = "i20_1"
I20 = EXP / "I20_1"
DSL = {"nist4": "NIST 4 K lvt (I8-1, 77 K card)", "nistrt": "NIST 4 K regular-Vt (I9-1, room-temperature card)"}
DSS = {"nist4": "NIST lvt", "nistrt": "NIST RT card"}
SRC = {"nist4": "I8_1", "nistrt": "I9_1"}
IV_COLORS = ["#1f4e9c", "#3a7d44", "#c2571a", "#7b3fa0", "#b8860b", "#2a9d8f", "#9b2226", "#495057", "#0077b6"]
ARMLAB = {"stock": "stock card", "start": "start (C0, no refit)", "C0_refit": "C0_refit", "T1_inloop": "T1_inloop",
          "M_bias_v1": "M_bias v1", "M_bias_lvt": "M_bias_lvt (I9-1 transfer)", "B1": "B1", "B2": "B2",
          "B3_d3": "B3_d3"}


def ci(c: dict) -> str:
    return f"{c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"


def col(key, head, md=None, kind="rtext", group=None):
    return {"key": key, "head": head, "md": md or head, "kind": kind, "group": group}


def f4(x) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x:.4f}"


def agg3(a: dict) -> str:
    return f"{f4(a['valid_only_mean'])} / {f4(a['combined'])} / {f4(a['median'])}"


def finite_tags(d, ds) -> list[str]:
    return sorted(d["dec"]["per_dataset"][ds]["rows"]["C0_refit"])


def fam_of(tags):
    return {t: ("pmos" if t.startswith("pfet") else "nmos") for t in tags}


def close(a, b, tol=1e-12) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if not (np.isfinite(a) and np.isfinite(b)):
        return (not np.isfinite(a)) and (not np.isfinite(b))
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
def load_all() -> dict:
    d = {"prereg": load(I20 / "prereg.json"), "amend": load(I20 / "prereg_amendment_A1_2026-10-23.json"),
         "audit": load(I20 / "audit/audit.json"), "start": load(I20 / "start/start.json"),
         "dec": load(I20 / "decision.json"), "i19": load(EXP / "I19_1/test_gate_decisions.json")}
    for ds in m.DATASETS:
        d[f"base_{ds}"] = load(I20 / f"base_{ds}/results.json")
        d[f"sel_{ds}"] = load(I20 / f"select_{ds}/select.json")
        d[f"lodo_{ds}"] = load(I20 / f"lodo_{ds}/results.json")
        d[f"s3v1_{ds}"] = load(I20 / f"s3/{ds}_v1.json")
        d[f"s3v2_{ds}"] = load(I20 / f"s3/{ds}_v2.json")
        d[f"old_{ds}"] = load(EXP / SRC[ds] / "lodo/results.json")
    n_checked = 0
    for ds in m.DATASETS:
        P = d["dec"]["per_dataset"][ds]
        rows = P["rows"]
        g = paired_ratio(rows["selected"], rows["C0_refit"])
        assert close(g["ratio"], P["primary_gmr_selected_vs_C0_refit"]["ratio"]) and \
            close(g["ci_high"], P["primary_gmr_selected_vs_C0_refit"]["ci_high"])
        g = paired_ratio(rows["selected"], rows["M_bias_v1"])
        assert close(g["ratio"], P["secondary_gmr_selected_vs_v1"]["ratio"])
        n_checked += 3
        for k, src in (("v1_stateC", f"s3v1_{ds}"), ("g_v2", f"s3v2_{ds}")):
            per = d[src]["per_record"]
            s3 = float(np.nanmean([per[t] for t in finite_tags(d, ds)]))
            assert close(s3, P["S3_D"][k], 1e-12), (ds, k)
            assert m.decision_word(s3) == P["decisions"][k]
            n_checked += 2
        assert m.primary_label(P["primary_gmr_selected_vs_C0_refit"], P["n_failed_selected_C0"]) == P["outcome"] \
            or P["outcome"] == "invalid"
        assert rows["selected"] == d[f"lodo_{ds}"]["per_device"]
        assert rows["C0_refit"] == d[f"base_{ds}"]["per_device"]["C0_refit"]
        n_checked += 3
        A = d["audit"]["datasets"][ds]["arms"]
        for arm in ("start", "C0_refit", "M_bias_v1"):
            assert m.max_abs_diff(A[arm]["per_device"]["L"], A[arm]["per_device_stored"]) <= 1e-12
            n_checked += 1
    d["n_checked"] = n_checked
    return d


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def t_audit(d: dict) -> rf.Table:
    rows = []
    for ds in m.DATASETS:
        A = d["audit"]["datasets"][ds]["arms"]
        for k, (arm, x) in enumerate(A.items()):
            ag = x["aggregate"]
            rows.append({"ds": DSS[ds] if k == 0 else "", "arm": ARMLAB.get(arm, arm), "n": str(x["n"]),
                         "ncl": str(x["clamp_classes"]["high"]), "nch": str(x["n_changed_C_vs_L"]),
                         **{s: f"{f4(ag[s]['valid_only_mean'])} / {f4(ag[s]['combined'])}" for s in m.STATES},
                         "v1": "0" if x["v1_max_abs_L_minus_stored"] == 0 else f"{x['v1_max_abs_L_minus_stored']:.1e}"})
    v = d["audit"]["validity"]
    notes = ["Stored cards and stored corrections of every arm, no refit and no retraining, rescored per clamp state. "
             "L: stored deck (first analysis clamped by b4check.c, later ones with the card values); C: clamp written "
             "into the card; U: warm-up op (every curve with the card values). RRMS: mean over devices with a finite "
             "value / combined (mean of the nMOS and pMOS family means); NIST lvt arms cover the 53 finite-start "
             "devices (stock and start: 57, 4 failed in L and C by I8-1 amendment A1).",
             f"V1: state L reproduces every stored row of every arm (max |diff| {v['v1_core_max']:.1e} on start / "
             f"C0_refit / M_bias v1; 0 on all other arms). V2 (amendment A1): C = L = U on "
             f"{v['v2_restricted_n']} unclamped cards without a fatal model check; C = L and U != L on "
             f"{v['v2_fatal_cards']} cards with 'Pclm is not positive' (5 stock, 2 start, I8-1 A1 devices). V3: "
             f"b4check.c printed A2 = computed A2_eff on every clamped card (max rel {v['v3_a2_rel_max']:.1e}); no A2 "
             f"message on unclamped cards; state-C curve 1 = state-L curve 1 (max |diff| {v['v3_curve1_max']:.0e}).",
             f"Other size-dependent b4check.c clamps on stored cards: {len(v['other_clamps'])}. Clamped cards: A2_eff "
             "> 1 (no card has A2_eff < 0.01). 'C != L': devices whose RRMS differs between the states."]
    return rf.Table("01_table_i20_1_audit", "I20-1 audit: stored NIST cards rescored per BSIM4 clamp state",
                    [col("ds", "dataset", kind="text"), col("arm", "arm", kind="text"), col("n", "devices"),
                     col("ncl", "A2_eff > 1"), col("nch", "C != L"), col("L", "state L (stored)"),
                     col("C", "state C"), col("U", "state U"), col("v1", "max |L - stored|")], rows, notes)


def t_audit_gmr(d: dict) -> rf.Table:
    rows = []
    lab = {"M_bias_v1_vs_C0_refit": "GMR(M_bias v1, C0_refit)", "B3_d3_vs_C0_refit": "GMR(B3_d3, C0_refit)",
           "M_bias_v1_vs_B3_d3": "GMR(M_bias v1, B3_d3)"}
    for ds in m.DATASETS:
        G = d["audit"]["datasets"][ds]["gmr"]
        for k, (key, g) in enumerate(G.items()):
            rows.append({"ds": DSS[ds] if k == 0 else "", "c": lab[key],
                         **{s: f"{ci(g[s])} {g[s]['wins']}/{g[s]['n_used']}" for s in ("L", "C", "U")}})
    notes = ["Stored cards and corrections rescored (no refit); state L equals the stored I8-1 / I9-1 values. Device "
             "bootstrap 10,000, seed 0, 95% percentile CI; wins = devices with the first arm lower."]
    return rf.Table("02_table_i20_1_audit_gmr", "I20-1 audit: paired GMRs of the stored arms per clamp state",
                    [col("ds", "dataset", kind="text"), col("c", "contrast", kind="text"), col("L", "state L"),
                     col("C", "state C"), col("U", "state U")], rows, notes)


def t_headline(d: dict) -> rf.Table:
    rows = []
    for ds in m.DATASETS:
        P = d["dec"]["per_dataset"][ds]
        tags = finite_tags(d, ds)
        fam = fam_of(tags)
        A = d["audit"]["datasets"][ds]["arms"]
        old = d[f"old_{ds}"]["per_device"]
        st_l = b8.aggregate({t: A["start"]["per_device"]["L"][t] for t in tags}, fam)
        st_c = b8.aggregate({t: d["start"][ds]["rows"][t]["rrms_C"] for t in tags}, fam)
        o_c0 = {t: drv._nan(old["C0_refit"][t]) for t in tags}
        o_v1 = {t: drv._nan(old["M_bias"][t]) for t in tags}
        R = P["rows"]
        g_old = paired_ratio(o_v1, o_c0)
        spec = [("start (not refit)", agg3(st_l), agg3(st_c), "", ""),
                ("C0_refit", agg3(b8.aggregate(o_c0, fam)), agg3(P["table"]["C0_refit"]), "", ""),
                ("M_bias v1", agg3(b8.aggregate(o_v1, fam)), agg3(P["table"]["M_bias_v1"]),
                 f"{ci(g_old)} {g_old['wins']}/{g_old['n_used']}",
                 f"{ci(P['table']['M_bias_v1']['gmr_vs_C0_refit'])} {P['table']['M_bias_v1']['gmr_vs_C0_refit']['wins']}"
                 f"/{P['table']['M_bias_v1']['gmr_vs_C0_refit']['n_used']}"),
                ("selected g v2 (new)", "-", agg3(P["table"]["selected"]), "-",
                 f"{ci(P['primary_gmr_selected_vs_C0_refit'])} {P['primary_gmr_selected_vs_C0_refit']['wins']}/"
                 f"{P['primary_gmr_selected_vs_C0_refit']['n_used']}")]
        for k, (arm, o, n, go, gn) in enumerate(spec):
            rows.append({"ds": f"{DSS[ds]} ({len(tags)})" if k == 0 else "", "arm": arm, "o": o, "n": n,
                         "go": go, "gn": gn})
        s = P["secondary_gmr_selected_vs_v1"]
        rows.append({"ds": "", "arm": "GMR(selected g v2, M_bias v1)", "o": "", "n": "", "go": "",
                     "gn": f"{ci(s)} {s['wins']}/{s['n_used']}"})
    notes = ["old: stored I8-1 / I9-1 rows (state L). new: rerun in state C (starting cards re-simulated in C, not "
             "refit; C0_refit and M_bias v1 retrained / refit with the I8-1 / I9-1 recipe; selected g v2 = symmetric "
             "inputs, EPS 0.01 V, VDD 1.8 V, lambda* per device from the 6-block nested selection, 5 seeds, LODO, "
             "held-out refit with g frozen).",
             "RRMS: all-device mean over the finite-start devices / combined (mean of nMOS and pMOS family means) / "
             "median; strict paper RRMS with the frozen measured inclusion. GMR: device bootstrap 10,000, 95% CI, "
             "wins = devices with the first arm lower / devices paired."]
    return rf.Table("03_table_i20_1_headline", "I20-1 headline: NIST DEV arms, stored (state L) vs rerun in state C, "
                    "and the selected g v2 recipe",
                    [col("ds", "dataset (devices)", kind="text"), col("arm", "arm", kind="text"),
                     col("o", "old (L): all / comb / median"), col("n", "new (C): all / comb / median"),
                     col("go", "GMR vs C0_refit, old"), col("gn", "GMR vs C0_refit, new")], rows, notes)


def t_decision(d: dict) -> rf.Table:
    D = d["dec"]
    rows = []
    for ds in m.DATASETS:
        P = D["per_dataset"][ds]
        rows.append({"ds": DSS[ds], "p": f"{ci(P['primary_gmr_selected_vs_C0_refit'])}",
                     "pl": P["outcome"], "s": ci(P["secondary_gmr_selected_vs_v1"]), "sl": P["secondary"],
                     "nf": str(P["n_failed_selected_C0"]),
                     "lam": ", ".join(f"{k} V x{v}" for k, v in sorted(P["lambda_counts_devices"].items(),
                                                                         key=lambda x: float(x[0]))),
                     "lf": f"{P['lambda_star_full']:g} V"})
    v = D["validity"]
    notes = [f"Validity: {'valid' if v['valid'] else 'invalid'}" + (f" ({'; '.join(v['fails'])})" if v["fails"] else "")
             + f"; V7 grids {D['validity_extra']['v7_grids_ok']}, V8 unclamped starting records identical "
             f"{D['validity_extra']['v8_unclamped_start_identical']}.",
             "Primary label (I17-2 rule): gain over C0_refit replicated iff CI upper < 1 and 0 failed rows; worse iff "
             "CI lower > 1; else inconclusive. Secondary: non-inferior to M_bias v1 iff CI upper <= 1.05.",
             "lambda*: per device = lambda* of its selection block (6 blocks; I15-2 inner rule inside each block); "
             "full: selection over all finite DEV devices (for TEST)."]
    return rf.Table("04_table_i20_1_decision", "I20-1 pre-registered outcome per NIST dataset (state C)",
                    [col("ds", "dataset", kind="text"), col("p", "GMR(selected g v2, C0_refit)"),
                     col("pl", "primary label", kind="text"), col("s", "GMR(selected g v2, M_bias v1)"),
                     col("sl", "secondary label", kind="text"), col("nf", "failed rows"),
                     col("lam", "lambda* (devices)", kind="text"), col("lf", "full-data lambda*")], rows, notes)


def t_gate(d: dict) -> rf.Table:
    D = d["dec"]
    rows = []
    for ds in m.DATASETS:
        P = D["per_dataset"][ds]
        S, W = P["S3_D"], P["decisions"]
        sc = P["sign_check_vs_realised_g_v2"]
        rows.append({"ds": DSS[ds], "c": f"{S['committed_a755e18_v1_L']:+.4f} {W['committed']}",
                     "v1": f"{S['v1_stateC']:+.4f} {W['v1_stateC']}", "g2": f"{S['g_v2']:+.4f} {W['g_v2']}",
                     "fr": W["frozen_rule"], "re": f"{P['primary_gmr_selected_vs_C0_refit']['ratio']:.3f} "
                                                   f"({sc['realised']})",
                     "ok": f"v1 {'yes' if sc['v1_stateC'] else 'no'}, g v2 {'yes' if sc['g_v2'] else 'no'}"})
    a = D["amendment"]
    fc = D["frozen_rule_check"]
    notes = ["S3_D (I19-1 definition): mean over the finite DEV records of ln(RRMS(start x exp(g_-k)) / RRMS(start)), "
             "g_-k trained on all other devices (seed 0, 500 steps), no refit; apply iff S3_D < 0. committed: I19-1 "
             "Part P (a755e18), v1 recipe, state-L records. v1 state C: same recipe on the state-C records. g v2: "
             "selected recipe with lambda*_k.",
             "Frozen-method rule (pre-registered in I20-1, open item 9): the v1-recipe S3_D on the state-C records "
             "decides whether the selected g v2 correction is applied.",
             f"Sign check against the realised GMR(selected g v2, C0_refit): v1 state C {fc['v1_stateC_correct']}/"
             f"{fc['n']}, g v2 recipe {fc['g_v2_correct']}/{fc['n']} (retrospective DEV rows before this "
             "iteration: 4/4 and 2/4).",
             "Amendment of the committed TEST gate decisions: " + ("needed (" + "; ".join(
                 f"{c['dataset']} {c['statistic']} {c['committed']} -> {c['new']}" for c in a["changes"]) + ")"
                 if a["needed"] else "not needed (no decision differs from the committed one).")]
    return rf.Table("05_table_i20_1_gate", "I20-1 dataset-level gate on NIST: S3_D values and decisions",
                    [col("ds", "dataset", kind="text"), col("c", "committed (a755e18)"),
                     col("v1", "v1 recipe, state C"), col("g2", "g v2 recipe"), col("fr", "frozen rule", kind="text"),
                     col("re", "realised GMR(sel., C0_refit)"), col("ok", "sign = realised", kind="text")],
                    rows, notes)


def t_family(d: dict) -> rf.Table:
    rows = []
    for ds in m.DATASETS:
        P = d["dec"]["per_dataset"][ds]
        for k, f in enumerate(("nmos", "pmos")):
            F = P["per_family"][f]
            rows.append({"ds": DSS[ds] if k == 0 else "", "f": f"{f} ({F['C0_refit']['aggregate']['n']})",
                         "c0": f4(F["C0_refit"]["aggregate"]["valid_only_mean"]),
                         "v1": f4(F["M_bias_v1"]["aggregate"]["valid_only_mean"]),
                         "sel": f4(F["selected"]["aggregate"]["valid_only_mean"]),
                         "g1": f"{ci(F['M_bias_v1']['gmr_vs_C0_refit'])} {F['M_bias_v1']['gmr_vs_C0_refit']['wins']}",
                         "g2": f"{ci(F['selected']['gmr_vs_C0_refit'])} {F['selected']['gmr_vs_C0_refit']['wins']}"})
    notes = ["State C rows; family means over the finite-start devices; GMR device bootstrap within the family; wins = "
             "devices with the arm below C0_refit."]
    return rf.Table("06_table_i20_1_family", "I20-1 per family (state C)",
                    [col("ds", "dataset", kind="text"), col("f", "family (devices)", kind="text"),
                     col("c0", "C0_refit"), col("v1", "M_bias v1"), col("sel", "selected g v2"),
                     col("g1", "GMR(v1, C0_refit)"), col("g2", "GMR(selected, C0_refit)")], rows, notes)


def t_selection(d: dict) -> rf.Table:
    rows = []
    for ds in m.DATASETS:
        S = d[f"sel_{ds}"]
        for k, o in enumerate(list(S["lambda_star_block"]) + ["full"]):
            sc = S["outer"][o]["scores"]
            rows.append({"ds": DSS[ds] if k == 0 else "", "b": o if o == "full" else f"block {o}",
                         "n": str(len(S["blocks"][int(o)]) if o != "full" else len(S["plan"]["full"]["train"])),
                         "ntr": str(len(S["plan"][o]["train"])),
                         **{f"l{lam:g}": f"{sc[f'{lam:g}']:.4f}" for lam in m.GRID_LAMBDA},
                         "ls": f"{S['outer'][o]['lambda_star']:g}"})
    notes = ["Inner score = mean ln RRMS over the inner held-out devices (one member, seed 0, per inner fold and "
             "lambda; inner held-out refit with g frozen, state C). lambda* = argmin (ties to the larger lambda). "
             "Block j: devices scored with lambda*_j; its selection uses only the devices outside block j "
             "('training devices'). full: all finite DEV devices."]
    return rf.Table("07_table_i20_1_selection", "I20-1 nested lambda selection (6 blocks and full data)",
                    [col("ds", "dataset", kind="text"), col("b", "selection", kind="text"),
                     col("n", "devices scored"), col("ntr", "training devices")]
                    + [col(f"l{lam:g}", f"lambda {lam:g} V") for lam in m.GRID_LAMBDA] + [col("ls", "lambda*")],
                    rows, notes)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_forest(d: dict, path: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    ylab, y = [], 0
    for ds in m.DATASETS:
        P = d["dec"]["per_dataset"][ds]
        A = d["audit"]["datasets"][ds]["gmr"]["M_bias_v1_vs_C0_refit"]
        items = [("GMR(M_bias v1, C0_refit), stored (L)", A["stored"], GREY),
                 ("GMR(M_bias v1, C0_refit), rerun (C)", P["table"]["M_bias_v1"]["gmr_vs_C0_refit"], ORANGE),
                 ("GMR(selected g v2, C0_refit) (C), primary", P["primary_gmr_selected_vs_C0_refit"], BLUE),
                 ("GMR(selected g v2, M_bias v1) (C), secondary", P["secondary_gmr_selected_vs_v1"], AQUA)]
        for lab, c, colr in items:
            ax.errorbar(c["ratio"], -y, xerr=[[c["ratio"] - c["ci_low"]], [c["ci_high"] - c["ratio"]]], fmt="o",
                        color=colr, ms=6, capsize=3, lw=1.8)
            ax.annotate(f"{ci(c)}, {c['wins']}/{c['n_used']}", (1.17, -y), fontsize=7.8, va="center", color=INK,
                        annotation_clip=False)
            ylab.append(f"{DSS[ds]}: {lab}")
            y += 1
        y += 0.6
    ax.axvline(1.0, color=INK, lw=0.9)
    ax.axvline(1.05, color=RED, lw=0.9, ls="--")
    ys = []
    yy = 0
    for ds in m.DATASETS:
        ys += [-(yy + i) for i in range(4)]
        yy += 4.6
    ax.set_yticks(ys)
    ax.set_yticklabels(ylab, fontsize=8)
    ax.set_xlim(0.6, 1.15)
    ax.set_xlabel("GMR of held-out RRMS, device bootstrap 10,000, 95% CI; < 1: first arm lower; dashed: margin 1.05",
                  fontsize=8.3)
    fig.suptitle("I20-1 NIST DEV contrasts: stored (state L) vs rerun (state C) and the selected g v2 recipe",
                 fontsize=10, color=INK)
    fig.subplots_adjust(left=0.36, right=0.8, top=0.92, bottom=0.12)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I20 / "decision.json"), rel(I20 / "audit/audit.json")]


def fig_per_device(d: dict, path: Path) -> list[str]:
    fig, axs = plt.subplots(2, 1, figsize=(15, 9), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for ax, ds in zip(axs, m.DATASETS):
        style(ax)
        P = d["dec"]["per_dataset"][ds]
        tags = finite_tags(d, ds)
        tags = sorted(tags, key=lambda t: (t.startswith("pfet"), t))
        R = P["rows"]
        cl = {t: m.clamp_class(d["start"][ds]["rows"][t]["a2_eff"]) != "none" for t in tags}
        x = np.arange(len(tags))
        for arm, colr, mk, off in (("M_bias_v1", ORANGE, "D", -0.15), ("selected", BLUE, "o", 0.15)):
            y = [math.log(R[arm][t] / R["C0_refit"][t]) for t in tags]
            ax.scatter(x + off, y, color=colr, marker=mk, s=18,
                       label={"M_bias_v1": "ln(M_bias v1 / C0_refit)", "selected": "ln(selected g v2 / C0_refit)"}[arm])
        for i, t in enumerate(tags):
            if cl[t]:
                ax.axvspan(i - 0.45, i + 0.45, color=RED, alpha=0.07, lw=0)
        ax.axhline(0, color=INK, lw=0.8)
        npm = sum(not t.startswith("pfet") for t in tags)
        ax.axvline(npm - 0.5, color=GREY, lw=0.8, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace("fet_01v8_lvt_", "").replace("fet_01v8_", "") for t in tags], rotation=90,
                           fontsize=5.5)
        ax.set_ylabel("ln ratio (state C)")
        ax.set_title(f"{DSL[ds]}: {len(tags)} devices (nMOS left, pMOS right of the dotted line; shaded: starting card "
                     "A2_eff > 1)", fontsize=9, loc="left")
        ax.legend(fontsize=7.5, frameon=False, loc="lower left", ncol=2)
    fig.suptitle("I20-1 per-device held-out log ratios against C0_refit, state C", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I20 / "decision.json"), rel(I20 / "start/start.json")]


def fig_a2(d: dict, path: Path) -> list[str]:
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.4), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for ax, ds in zip(axs, m.DATASETS):
        style(ax)
        rows = d["start"][ds]["rows"]
        devs = drv.devices(ds)
        for fam, mk in (("nfet", "o"), ("pfet", "s")):
            ts = [t for t in rows if t.startswith(fam)]
            L = [devs[t].L_um for t in ts]
            a2 = [rows[t]["a2_eff"] for t in ts]
            c = [RED if v > 1 else BLUE for v in a2]
            ax.scatter(L, a2, c=c, marker=mk, s=22, label=f"{'nMOS' if fam == 'nfet' else 'pMOS'}")
        ax.axhline(1.0, color=RED, lw=0.9, ls="--")
        ax.axhline(0.01, color=GREY, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("L (um)")
        ax.set_ylabel("A2_eff of the starting card")
        n = sum(r["a2_eff"] > 1 for r in rows.values())
        ax.set_title(f"{DSL[ds]}: A2_eff > 1 on {n}/{len(rows)} (red); dashed 1, dotted 0.01", fontsize=9, loc="left")
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("I20-1 effective (size-dependent) A2 of the stored starting cards; b4check.c clamps A2_eff > 1 to 1 "
                 "in the first analysis of a process only", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I20 / "start/start.json")]


# ---------------------------------------------------------------------------
# I-V
# ---------------------------------------------------------------------------
def pick_iv(d: dict, ds: str) -> list[tuple[str, str]]:
    P = d["dec"]["per_dataset"][ds]
    R = P["rows"]
    tags = finite_tags(d, ds)
    st = d["start"][ds]["rows"]
    out = []
    cl = [t for t in tags if st[t]["a2_eff"] > 1]
    if cl:
        t = max(cl, key=lambda t: abs(st[t]["rrms_C"] - st[t]["rrms_L"]))
        out.append((t, f"starting card A2_eff {st[t]['a2_eff']:.3f} > 1, largest |start RRMS C - L| "
                       f"({st[t]['rrms_L']:.4f} -> {st[t]['rrms_C']:.4f})"))
    for fam in ("nfet", "pfet"):
        ts = [t for t in tags if t.startswith(fam) and t not in [x for x, _ in out]]
        lr = {t: math.log(R["selected"][t] / R["C0_refit"][t]) for t in ts}
        med = float(np.median(list(lr.values())))
        t = min(ts, key=lambda t: abs(lr[t] - med))
        out.append((t, f"{'nMOS' if fam == 'nfet' else 'pMOS'} device nearest the family median ln(selected / "
                       f"C0_refit) {med:+.3f}"))
    return out


def resim(d: dict, ds: str, tag: str) -> dict:
    B = I20
    dev, bi = drv.devices(ds)[tag], drv.bins(ds)[tag]
    old = drv.stored_records(ds)[tag]
    hb = load(B / f"base_{ds}/heldout_cards.json")
    cb = load(B / f"base_{ds}/corrections.json")
    hl = load(B / f"lodo_{ds}/heldout_cards.json")
    cl = load(B / f"lodo_{ds}/corrections.json")
    start = dict(old.meta["params"])
    sims, sc = {}, {}

    def run(key, params, state, corr):
        s = drv.sim_card(ds, tag, params, state)
        rec = b8.to_record(dev, bi, s, params)
        sims[key] = corr.apply(rec)
        sc[key] = float(score_device(rec, sims[key])["rrms"])
        return rec

    prob = drv.problem(ds, tag, "C")
    stock_s = prob.simulate_params(None)
    sims["stock"] = stock_s
    sc["stock"] = float(np.mean(prob.curve_rrms(stock_s)))
    rec = run("start_L", start, "L", Correction("C0"))
    run("start_C", start, "C", Correction("C0"))
    run("C0_refit", hb[f"C0_refit|{tag}"], "C", Correction("C0"))
    run("M_bias_v1", hb[f"M_bias|{tag}"], "C", correction_from_json(cb[f"M_bias|{tag}"]))
    run("selected", hl[tag], "C", v2.SymCorrection.from_json(cl[f"selected|{tag}"]))
    P = d["dec"]["per_dataset"][ds]["rows"]
    A = d["audit"]["datasets"][ds]["arms"]
    checks = {"start_L": (sc["start_L"], A["start"]["per_device"]["L"][tag]),
              "start_C": (sc["start_C"], d["start"][ds]["rows"][tag]["rrms_C"]),
              "C0_refit": (sc["C0_refit"], P["C0_refit"][tag]), "M_bias_v1": (sc["M_bias_v1"], P["M_bias_v1"][tag]),
              "selected": (sc["selected"], P["selected"][tag]),
              "stock": (sc["stock"], A["stock"]["per_device"]["C"][tag])}
    for k, (a, b) in checks.items():
        assert close(a, b, 1e-9), (ds, tag, k, a, b)
    return {"rec": rec, "sims": sims, "score": sc, "a2": d["start"][ds]["rows"][tag]["a2_eff"], "dev": dev, "bin": bi}


def fig_iv(d: dict, ds: str, tag: str, why: str, path: Path) -> list[str]:
    r = resim(d, ds, tag)
    rec, sc = r["rec"], r["score"]
    arms = (("stock", GREY, (0, (4, 2)), 1.0), ("start_L", RED, ":", 1.1), ("start_C", RED, "-", 0.9),
            ("C0_refit", BLUE, "-", 1.2), ("M_bias_v1", ORANGE, "--", 1.2), ("selected", AQUA, "-", 1.4))
    fig, axs = plt.subplots(2, 2, figsize=(14.5, 10.5), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for row, kind in enumerate(("idvg", "idvd")):
        idx = [i for i, c in enumerate(rec.curves) if c.kind == kind]
        for colm, scale in enumerate(("log", "linear") if kind == "idvg" else ("linear", "log")):
            ax = axs[row, colm]
            style(ax)
            for k, i in enumerate(idx):
                c = rec.curves[i]
                color = IV_COLORS[k % len(IV_COLORS)]
                x = np.abs(np.asarray(c.vg if kind == "idvg" else c.vd, float))
                meas = np.abs(np.asarray(c.target, float))
                step = 3 if len(x) > 100 else 1
                ax.scatter(x[::step], meas[::step], s=9, color=color, alpha=0.9 if c.included else 0.3, zorder=3,
                           lw=0)
                for a, colr, ls, lw in arms:
                    ax.plot(x, np.abs(np.asarray(r["sims"][a][i], float)), color=colr, ls=ls, lw=lw,
                            alpha=0.9 if c.included else 0.35)
                ax.plot([], [], color=color, marker="o", ms=4, ls="",
                        label=c.key + ("" if c.included else " (not in RRMS)"))
            if scale == "log":
                ax.set_yscale("log")
                ax.set_ylim(bottom=1e-13)
            ax.set_xlabel("|Vgs| (V)" if kind == "idvg" else "|Vds| (V)", fontsize=9)
            ax.set_ylabel("|Id| (A)", fontsize=9)
            ax.set_title(f"{'Id-Vg' if kind == 'idvg' else 'Id-Vd'}, {scale} scale", fontsize=9.5, color=INK,
                         loc="left")
            if colm == 0:
                ax.legend(fontsize=7, frameon=False, loc="best", ncol=2)
    lab = {"stock": "NGSpice, stock card (published sky130 card of the dataset, no fit)",
           "start_L": f"NGSpice, starting card, state L (stored deck; A2_eff {r['a2']:.3f})",
           "start_C": "NGSpice, starting card, state C (clamp written)",
           "C0_refit": "NGSpice re-simulated, C0_refit held-out card (state C)",
           "M_bias_v1": "NGSpice re-simulated, M_bias v1 held-out card x exp(g v1) (state C)",
           "selected": "NGSpice re-simulated, selected g v2 held-out card x exp(g v2) (state C)"}
    handles = [Line2D([], [], marker="o", ls="", color=INK, ms=4,
                      label="measured 4 K (colour per curve; faded: not in RRMS; every 3rd point on 181-point sweeps)")]
    handles += [Line2D([], [], color=c, ls=ls, lw=lw, label=f"{lab[a]}: RRMS {sc[a]:.4f}") for a, c, ls, lw in arms]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.8, frameon=False)
    dv = r["dev"]
    fig.suptitle(f"I20-1 {DSS[ds]} {tag} ({dv.polarity}, W {dv.W_um:g} um, L {dv.L_um:g} um, bin {r['bin']['bin']}): "
                 f"{why}\nngspice-41 arm64; exp(g) applied to the NGSpice current in Python (as scored); RRMS of every "
                 "re-simulated curve set equals the stored row", fontsize=9.6, color=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I20 / f"base_{ds}/heldout_cards.json"), rel(I20 / f"base_{ds}/corrections.json"),
            rel(I20 / f"lodo_{ds}/heldout_cards.json"), rel(I20 / f"lodo_{ds}/corrections.json"),
            rel(EXP / SRC[ds] / "start/records.json")]


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

    add_table(t_audit(d), "Audit: every stored NIST arm rescored in states L / C / U, devices with A2_eff > 1, "
              "devices changed, validity V1-V3", [rel(I20 / "audit/audit.json")])
    add_table(t_audit_gmr(d), "Audit GMRs per clamp state (stored cards, no refit)", [rel(I20 / "audit/audit.json")])
    add_table(t_headline(d), "Headline: old (stored, state L) vs new (rerun, state C) per arm, and the selected g v2 "
              "recipe; GMRs vs C0_refit", [rel(I20 / "decision.json"), rel(EXP / "I8_1/lodo/results.json"),
                                           rel(EXP / "I9_1/lodo/results.json")])
    add_table(t_decision(d), "Pre-registered primary / secondary outcome per NIST dataset, validity, lambda*",
              [rel(I20 / "decision.json")])
    add_table(t_gate(d), "S3_D and gate decisions: committed (a755e18), v1 recipe in state C, g v2 recipe, frozen "
              "rule, realised sign, amendment trigger", [rel(I20 / "decision.json"), rel(I20 / "s3")])
    add_table(t_family(d), "Per-family means and GMRs (state C)", [rel(I20 / "decision.json")])
    add_table(t_selection(d), "Nested lambda selection: inner scores per block and full data",
              [rel(I20 / "select_nist4/select.json"), rel(I20 / "select_nistrt/select.json")])
    add_fig("08_i20_1_forest.png", fig_forest, "Forest plot: GMR(M_bias v1, C0_refit) stored (L) vs rerun (C), "
            "GMR(selected g v2, C0_refit), GMR(selected g v2, M_bias v1)")
    add_fig("09_i20_1_per_device.png", fig_per_device, "Per-device ln(arm / C0_refit) for M_bias v1 and selected g v2 "
            "(state C), clamped starting cards shaded")
    add_fig("10_i20_1_a2_eff.png", fig_a2, "A2_eff of the stored starting cards vs L, both datasets")
    k = 11
    for ds in m.DATASETS:
        for tag, why in pick_iv(d, ds):
            name = f"{k:02d}_i20_1_iv_{ds}_{tag}.png"
            items.append((name, f"I-V of {tag} ({DSS[ds]}): measured 4 K points; NGSpice of the stock card; the "
                                f"starting card in states L and C; NGSpice re-simulated state-C held-out cards of "
                                f"C0_refit, M_bias v1 x exp(g v1) and selected g v2 x exp(g v2); {why}",
                          fig_iv(d, ds, tag, why, folder / name)))
            k += 1

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i20_1.py --date {args.date}`"
    md = [f"# Tables, {args.date} {TOPIC}", "", f"Generated {now} by {gen}.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))
    D = d["dec"]
    lines = []
    for ds in m.DATASETS:
        P = D["per_dataset"][ds]
        lines.append(f"- {DSL[ds]}: GMR(selected g v2, C0_refit) {ci(P['primary_gmr_selected_vs_C0_refit'])} "
                     f"({P['outcome']}); GMR(selected g v2, M_bias v1) {ci(P['secondary_gmr_selected_vs_v1'])} "
                     f"({P['secondary']}); S3_D committed {P['S3_D']['committed_a755e18_v1_L']:+.4f}, v1 state C "
                     f"{P['S3_D']['v1_stateC']:+.4f}, g v2 {P['S3_D']['g_v2']:+.4f}.")
    readme = [f"# Report batch {args.date} · {TOPIC}", "",
              f"Generated {now} by {gen}. NGSpice runs: re-simulation of the I-V devices (stock card, starting card "
              f"in L and C, three state-C held-out cards); every RRMS checked against the stored rows; {d['n_checked']} "
              "stored values recomputed from per-device rows.", "",
              "## Contents", "",
              "- I20-1: BSIM4 A2 clamp state of the NIST decks (I8-1 lvt, I9-1 room-temperature card), audit in states "
              "L / C / U, fix (state C default in both NIST simulators), state-C reruns, and the selected g v2 recipe "
              "on both NIST DEV datasets. Pre-registration `out/research/explore/I20_1/prereg.md` (a516fe3), "
              "amendment A1 (1f14eda).", *lines,
              f"- Amendment of the I19-1 TEST gate decisions needed: {D['amendment']['needed']}.", "",
              "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))
    summary = [f"I20-1 NIST clamp audit and selected g v2 recipe on both NIST DEV datasets (state C)."] + \
        [x[2:] for x in lines]
    rf.update_index(args.date, TOPIC, folder, [i[0] for i in items], summary)
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
