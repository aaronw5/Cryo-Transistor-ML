#!/usr/bin/env python3
"""Report batch for I15-2: which g v2 training change costs accuracy
(symmetric inputs, smooth-abs EPS, gm/gds penalty lambda) on the 18-fold
LODO, and the training-data-only nested selection of lambda; sky130 77 K,
18 devices, I13-1 state C.

Reads stored files:

* ``out/research/explore/I15_2/{prereg.json, decision.json}``
* ``I15_2/validity/validity.json``, ``I15_2/inner/inner.json``
* ``I15_2/lodo_<variant>/{results,corrections,heldout_cards}.json``
* I13-1 state-C rows, corrections and held-out cards (v1, g v2, C0_refit)

NGSpice: the I-V figures re-simulate the published card (upstream committed
sweeps), the I13-1 C0_refit held-out card, and the held-out cards of v1, g v2
(lambda 10), sym_e1_l0 and the selected variant (card-only library deck with
the clamp written, x exp(fold g)); every RRMS is checked against the stored
rows.

  PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i15_2.py --date 2026-10-22
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_explore_e2_e3_e4 as pe  # noqa: E402  (helpers, style)
from plot_explore_e2_e3_e4 import (BG, INK, MUTED, GREY, BLUE, ORANGE, RED, VIOLET, AQUA, DPI,  # noqa: E402
                                   EXP, ROOT, load, rel, style)
from cryoml.explore import i15_2_ablate as m  # noqa: E402
from cryoml.explore.e1_correction import score_device  # noqa: E402

from matplotlib.lines import Line2D  # noqa: E402

rf = pe.rf
plt = pe.plt

TOPIC = "i15_2"
I15 = EXP / "I15_2"
I13 = EXP / "I13_1"
F = {"prereg": I15 / "prereg.json", "decision": I15 / "decision.json", "val": I15 / "validity/validity.json",
     "inner": I15 / "inner/inner.json", "lodo": I13 / "lodo/results.json", "all18": I13 / "all18/results.json"}
COMMITS = {"prereg": "7924961", "code": "d9e79ac"}
IV_TAGS = ("nmos_L1_W1p6", "pmos_L0p35_W1p6", "pmos_L8_W1p6", "pmos_L2_W5")
LAB = {"v1": "M_bias v1 (abs inputs, lambda 0)", "sym_e0_l0": "sym, EPS 0, lambda 0",
       "sym_e1_l0": "sym, EPS 0.01, lambda 0", "sym_e0_l10": "sym, EPS 0, lambda 10",
       "sym_e1_l1": "sym, EPS 0.01, lambda 1", "sym_e1_l2": "sym, EPS 0.01, lambda 2",
       "sym_e1_l5": "sym, EPS 0.01, lambda 5", "sym_e1_l10": "g v2: sym, EPS 0.01, lambda 10",
       "selected": "selected (nested lambda*)", "C0_refit": "C0_refit (no correction)"}


def ci(c: dict) -> str:
    return f"{c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"


def col(key, head, md=None, kind="rtext", group=None):
    return {"key": key, "head": head, "md": md or head, "kind": kind, "group": group}


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def t_primary(d: dict) -> rf.Table:
    D = d["decision"]
    ls = D["lambda_star"]
    tags = sorted(t for t in ls if t != "full")
    counts = {f"{lam:g}": sum(1 for t in tags if float(ls[t]) == lam) for lam in m.GRID_LAMBDA}
    p = D["primary_gmr_selected_vs_v1"]
    rows = [
        {"c": "primary: GMR(selected, M_bias v1), 18 devices", "v": ci(p), "w": f"{p['wins']}/18",
         "r": D["outcome"]},
        {"c": "GMR(selected, C0_refit)", "v": ci(D["gmr_selected_vs_C0_refit"]),
         "w": f"{D['gmr_selected_vs_C0_refit']['wins']}/18", "r": ""},
        {"c": "GMR(selected, g v2 = sym_e1_l10)", "v": ci(D["gmr_selected_vs_sym_e1_l10"]),
         "w": f"{D['gmr_selected_vs_sym_e1_l10']['wins']}/18", "r": ""},
        {"c": "lambda* over the 18 outer folds (V)", "v": ", ".join(f"{k}: {v}" for k, v in counts.items()),
         "w": "", "r": ""},
        {"c": "lambda* of the full-data model (reported, not scored)", "v": f"{float(ls['full']):g} V", "w": "",
         "r": ""},
        {"c": "changes named as costing accuracy (pre-registered label rule)",
         "v": ", ".join(D["changes_named_costly"]) or "none", "w": "", "r": ""},
    ]
    notes = ["Selection (pre-registered): per outer fold, 3 inner folds of the 17 training devices, one member (seed 0) "
             "per inner fold and lambda in {0, 1, 2, 5, 10} V (symmetric inputs, EPS 0.01 V), inner held-out card "
             "refits; lambda* = argmin mean ln RRMS, ties to the larger lambda. Selected row of device h = 5-seed LODO "
             "row of sym_e1_l{lambda*_h}.",
             "Outcome rule: CI upper of GMR(selected, v1) <= 1.05 -> non-inferior; GMR device bootstrap 10,000, 95% CI. "
             f"CI upper < 1: {D['ci_upper_below_1']}. Validity: {'valid' if D['validity']['valid'] else 'invalid'}."]
    return rf.Table("01_table_i15_2_primary", "I15-2 primary result and lambda selection",
                    [col("c", "quantity", kind="text"), col("v", "value"), col("w", "devices lower"),
                     col("r", "outcome", kind="text")], rows, notes)


def t_variants(d: dict) -> rf.Table:
    T = d["decision"]["variant_table"]
    order = ["v1", "sym_e0_l0", "sym_e1_l0", "sym_e0_l10", "sym_e1_l1", "sym_e1_l2", "sym_e1_l5", "sym_e1_l10",
             "selected", "C0_refit"]
    rows = []
    for v in order:
        r = T[v]
        rows.append({"v": LAB[v], "a": f"{r['all_device_mean']:.5f}", "c": f"{r['combined']:.5f}",
                     "md": f"{r['median']:.5f}",
                     "g1": ci(r["gmr_vs_v1"]) + f" ({r['gmr_vs_v1']['wins']}/18)" if "gmr_vs_v1" in r else "-",
                     "g0": ci(r["gmr_vs_C0_refit"]) + f" ({r['gmr_vs_C0_refit']['wins']}/18)"
                     if "gmr_vs_C0_refit" in r else "-",
                     "f": str(r["n_failed"])})
    notes = ["18-fold LODO, 5 seeds, held-out card refit with the fold g in the loop (state C); strict paper RRMS, "
             "curve inclusion frozen to the published card. all-device: mean over 18 devices; combined: mean of the "
             "nMOS and pMOS family means. v1 and g v2: stored I13-1 rows; other variants: I15-2 runs. "
             "GMR: device bootstrap, 95% CI (devices lower)."]
    return rf.Table("02_table_i15_2_variants", "I15-2 variants: held-out RRMS and paired GMRs",
                    [col("v", "variant", kind="text"), col("a", "all-device"), col("c", "combined"),
                     col("md", "median"), col("g1", "GMR vs v1"), col("g0", "GMR vs C0_refit"),
                     col("f", "failed")], rows, notes)


def t_contrasts(d: dict) -> rf.Table:
    D = d["decision"]
    rows = []
    for change, lst in D["contrasts"].items():
        for x in lst:
            rows.append({"ch": change, "c": f"{x['method']} vs {x['control']}", "g": ci(x), "w": f"{x['wins']}/18",
                         "l": x["label"]})
    for x in D["lambda_curve"]:
        rows.append({"ch": "lambda curve", "c": f"{x['method']} vs {x['control']}", "g": ci(x),
                     "w": f"{x['wins']}/18", "l": x["label"]})
    notes = ["Labels (pre-registered): cost if CI lower > 1; no cost beyond the margin if CI upper <= 1.05; otherwise "
             "inconclusive. A change is named as costing accuracy when one of its contrasts is labelled cost. "
             "wins: devices where the first arm is lower."]
    return rf.Table("03_table_i15_2_contrasts", "I15-2 ablation contrasts (paired GMR, 18 devices)",
                    [col("ch", "change", kind="text"), col("c", "contrast", kind="text"), col("g", "GMR [95% CI]"),
                     col("w", "lower"), col("l", "label", kind="text")], rows, notes)


def t_neg(d: dict) -> rf.Table:
    N = d["decision"]["negative_fractions"]
    rows = []
    for v in ["v1", "sym_e0_l0", "sym_e1_l0", "sym_e0_l10", "sym_e1_l1", "sym_e1_l2", "sym_e1_l5", "sym_e1_l10",
              "selected"]:
        x = N[v]
        rows.append({"v": LAB[v], "tg": f"{100 * x['train_mean_gm']:.3f}", "td": f"{100 * x['train_mean_gds']:.3f}",
                     "hg": f"{100 * x['heldout_mean_gm']:.3f}", "hd": f"{100 * x['heldout_mean_gds']:.3f}",
                     "fg": f"{100 * x['full_gm']:.3f}", "fd": f"{100 * x['full_gds']:.3f}"})
    c = N["card_only"]
    rows.append({"v": "card only (starting cards)", "tg": f"{100 * c['frac_neg_gm_card']:.3f}",
                 "td": f"{100 * c['frac_neg_gds_card']:.3f}", "hg": "", "hd": "", "fg": "", "fd": ""})
    notes = ["Percent of penalty-grid points (25 mV forward quadrant of each starting card, |I| > 1 nA; 72,475 points "
             "over 18 devices) with negative corrected gm / gds = (gm_card / I_card + dg/du_g), (gds_card / I_card + "
             "dg/du_d); dg/du by central differences. train: each fold model on its 17 training grids, mean over "
             "folds; held-out: each fold model on the held-out device's grid, mean over folds; full: the full-data "
             "model on all 18 grids (selected: full-data model of lambda* (full))."]
    return rf.Table("04_table_i15_2_negative", "I15-2 negative corrected gm / gds on the penalty grids (%)",
                    [col("v", "variant", kind="text"), col("tg", "gm train"), col("td", "gds train"),
                     col("hg", "gm held-out"), col("hd", "gds held-out"), col("fg", "gm full"),
                     col("fd", "gds full")], rows, notes)


def t_validity(d: dict) -> rf.Table:
    V = d["val"]
    D = d["decision"]["validity"]
    rows = [{"c": "V1 trainer, seed 0 of fold pmos_L2_W5, member 0 vs stored: g v2 recipe / v1 recipe",
             "t": "= 0", "v": f"{V['V1']['sym_e1_l10']['max_abs_weight_diff_member0']:.1g} / "
                              f"{V['V1']['v1']['max_abs_weight_diff_member0']:.1g}",
             "p": "pass" if V["v1_max_weight_diff"] == 0 else "FAIL"},
            {"c": "V2 held-out refit of the stored g v2 fold corrections (nmos_L1_W1p6, pmos_L2_W5)", "t": "<= 1e-9",
             "v": f"{V['v2_max_rrms_diff']:.1g}", "p": "pass" if V["v2_max_rrms_diff"] <= 1e-9 else "FAIL"},
            {"c": "V3 18 rows per new variant / inner CV complete", "t": "yes / yes",
             "v": f"{D['V3_rows_18']} / {D['V3_inner_complete']}",
             "p": "pass" if (D["V3_rows_18"] and D["V3_inner_complete"]) else "FAIL"}]
    I = d["inner"]["timing"]
    notes = [f"Inner selection: {I['n_train']} single-member trainings ({I['train_s']:.0f} s) and {I['n_refit']} "
             f"inner held-out refits ({I['refit_s']:.0f} s) on 6 workers."]
    return rf.Table("05_table_i15_2_validity", "I15-2 validity", [col("c", "check", kind="text"), col("t", "criterion"),
                                                                  col("v", "value"), col("p", "result")], rows, notes)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_lambda(d: dict, path: Path) -> list[str]:
    T = d["decision"]["variant_table"]
    N = d["decision"]["negative_fractions"]
    lams = [0, 1, 2, 5, 10]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.4), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax = axs[0]
    style(ax)
    g = [T[f"sym_e1_l{k}"]["gmr_vs_v1"] for k in lams]
    xs = np.arange(len(lams))
    ax.errorbar(xs, [x["ratio"] for x in g], yerr=[[x["ratio"] - x["ci_low"] for x in g],
                                                  [x["ci_high"] - x["ratio"] for x in g]],
                fmt="o-", color=AQUA, capsize=4, label="symmetric inputs, EPS 0.01 V")
    for k, off in ((0, -0.08), (10, 0.08)):
        x = T[f"sym_e0_l{k}"]["gmr_vs_v1"]
        ax.errorbar([lams.index(k) + off], [x["ratio"]], yerr=[[x["ratio"] - x["ci_low"]], [x["ci_high"] - x["ratio"]]],
                    fmt="s", color=ORANGE, capsize=4, label="symmetric inputs, EPS 0" if k == 0 else None)
    s = T["selected"]["gmr_vs_v1"]
    ax.axhspan(s["ci_low"], s["ci_high"], color=VIOLET, alpha=0.12)
    ax.axhline(s["ratio"], color=VIOLET, lw=1.2, label=f"selected (nested lambda*): {ci(s)}")
    ax.axhline(1.0, color=INK, lw=0.8)
    ax.axhline(m.NI_MARGIN, color=RED, lw=1.0, ls="--", label="non-inferiority margin 1.05")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{k}" for k in lams])
    ax.set_xlabel("penalty weight lambda (V)")
    ax.set_ylabel("GMR(variant, M_bias v1), 18 devices")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("held-out RRMS relative to M_bias v1 (device bootstrap 95% CI)", fontsize=9.5)
    ax = axs[1]
    style(ax)
    for key, colr, lab in (("train_mean_gds", AQUA, "gds < 0, training grids"),
                           ("heldout_mean_gds", BLUE, "gds < 0, held-out grid"),
                           ("train_mean_gm", ORANGE, "gm < 0, training grids"),
                           ("heldout_mean_gm", RED, "gm < 0, held-out grid")):
        ax.plot(xs, [100 * N[f"sym_e1_l{k}"][key] for k in lams], "o-", color=colr, label=lab)
        ax.scatter([-0.3], [100 * N["v1"][key]], color=colr, marker="D", s=30)
    ax.annotate("v1", (-0.3, 100 * N["v1"]["heldout_mean_gds"]), xytext=(-4, 6), textcoords="offset points",
                fontsize=8, color=MUTED)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{k}" for k in lams])
    ax.set_xlabel("penalty weight lambda (V); diamonds: M_bias v1")
    ax.set_ylabel("% of penalty-grid points")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("negative corrected gm / gds (fold models, mean over folds)", fontsize=9.5)
    fig.suptitle("I15-2: accuracy and gm / gds sign vs the penalty weight, sky130 77 K, 18-fold LODO", fontsize=10.5,
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(F["decision"])]


def fig_per_device(d: dict, path: Path) -> list[str]:
    D = d["decision"]
    rows = {"v1": d["v1_rows"], "sym_e1_l0": d["e1l0_rows"], "sym_e1_l10": d["v2_rows"], "selected": D["selected_rows"]}
    c0 = d["c0_rows"]
    tags = sorted(c0)
    x = np.arange(len(tags))
    fig, axs = plt.subplots(2, 1, figsize=(14, 8.4), dpi=DPI, sharex=True)
    fig.patch.set_facecolor(BG)
    ax = axs[0]
    style(ax)
    ax.scatter(x, [c0[t] for t in tags], color=BLUE, s=14, label=LAB["C0_refit"])
    for v, colr, mk, off in (("v1", ORANGE, "D", -0.18), ("sym_e1_l0", GREY, "v", -0.06),
                             ("sym_e1_l10", AQUA, "^", 0.06), ("selected", VIOLET, "o", 0.18)):
        ax.scatter(x + off, [rows[v][t] for t in tags], color=colr, marker=mk, s=20, label=LAB[v])
    ax.set_yscale("log")
    ax.set_ylabel("held-out RRMS (log)")
    ax.legend(fontsize=8, ncol=3, frameon=False, loc="upper left")
    ax = axs[1]
    style(ax)
    for v, colr, mk in (("sym_e1_l0", GREY, "v"), ("sym_e1_l10", AQUA, "^"), ("selected", VIOLET, "o")):
        ax.scatter(x, [rows[v][t] / rows["v1"][t] for t in tags], color=colr, marker=mk, s=22, label=LAB[v])
    for i, t in enumerate(tags):
        ax.annotate(f"{float(D['lambda_star'][t]):g}", (i, 0.5), fontsize=7.5, color=VIOLET, ha="center")
    ax.axhline(1.0, color=INK, lw=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("RRMS / M_bias v1 RRMS (log); numbers: lambda*")
    ax.legend(fontsize=8, frameon=False)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=60, ha="right", fontsize=7.5)
    fig.suptitle("I15-2 per-device held-out RRMS (18-fold LODO, state C)", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(F["decision"]), rel(F["lodo"]), rel(F["all18"]), rel(I15 / "lodo_sym_e1_l0/results.json")]


def fig_inner(d: dict, path: Path) -> list[str]:
    S = d["decision"]["inner_scores"]
    outs = sorted(o for o in S if o != "full") + ["full"]
    lams = [f"{k:g}" for k in m.GRID_LAMBDA]
    A = np.array([[S[o][k] - min(S[o].values()) for k in lams] for o in outs])
    fig, ax = plt.subplots(figsize=(8.5, 8.4), dpi=DPI)
    fig.patch.set_facecolor(BG)
    im = ax.imshow(A, cmap="viridis_r", aspect="auto")
    for i, o in enumerate(outs):
        for j, k in enumerate(lams):
            ax.text(j, i, f"{A[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if A[i, j] > A.max() / 2 else INK)
        j = lams.index(f"{float(d['decision']['lambda_star'][o]):g}")
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=RED, lw=1.6))
    ax.set_xticks(range(len(lams)))
    ax.set_xticklabels(lams)
    ax.set_yticks(range(len(outs)))
    ax.set_yticklabels([f"outer {o}" for o in outs], fontsize=7.5)
    ax.set_xlabel("lambda (V)")
    fig.colorbar(im, ax=ax, label="inner score - min over lambda (mean ln RRMS)")
    ax.set_title("I15-2 inner selection: mean ln RRMS over the inner held-out devices,\nrelative to the best lambda "
                 "of each outer fold (red box: lambda*)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(F["inner"]), rel(F["decision"])]


def resim(d: dict) -> dict:
    sys.path.insert(0, str(ROOT / "scripts/research/explore"))
    os.environ["CRYOML_E1_RECORDS"] = str(I13 / "records")
    import i14_1_chan as drv
    import i15_2_ablate as drv2
    from plot_explore_i3_2 import published_rrms
    recs = drv.load_records()
    pub, pub_curves = published_rrms({t: recs[t] for t in IV_TAGS})
    D = d["decision"]
    held = {"C0_refit": load(I13 / "all18/heldout_cards.json")["C0_refit"],
            "v1": load(I13 / "all18/heldout_cards.json")["M_bias"],
            "sym_e1_l10": load(I13 / "lodo/heldout_cards.json")}
    for v in m.NEW:
        held[v] = load(I15 / f"lodo_{v}/heldout_cards.json")
    out = {}
    for tag in IV_TAGS:
        rec = recs[tag]
        pm = rec.family == "pmos"
        sel = m.variant_for_lambda(float(D["lambda_star"][tag]))
        arms = {"C0_refit": "C0_refit", "v1": "v1", "sym_e1_l0": "sym_e1_l0", "sym_e1_l10": "sym_e1_l10",
                "selected": sel}
        sims, sc = {}, {}
        for lab, v in arms.items():
            run = drv.measured_run(tag, dict(held[v][tag]), "card")["sims"]
            if v == "C0_refit":
                s = run
                ref = d["c0_rows"][tag]
            else:
                corr = drv2.variant_corr(v, tag)
                if v == "v1":
                    from cryoml.explore.i2_1_joint import with_sims
                    s = corr.apply(with_sims(rec, run))
                else:
                    s = [np.asarray(ic) * np.exp(corr.g_v(pm, c.vg, c.vd)) for ic, c in zip(run, rec.curves)]
                ref = drv2.variant_rows(v)[tag]
            sims[lab] = s
            sc[lab] = score_device(rec, s)["rrms"]
            assert abs(sc[lab] - ref) < 1e-9, (tag, lab, sc[lab], ref)
        out[tag] = {"rec": rec, "pub": pub_curves[tag], "pub_rrms": pub[tag], "sims": sims, "score": sc,
                    "sel": sel}
    return out


def fig_iv(d: dict, tag: str, path: Path) -> list[str]:
    r = d["iv"][tag]
    rec = r["rec"]
    lines = (("C0_refit", BLUE, "-"), ("v1", ORANGE, "-"), ("sym_e1_l0", GREY, "-."), ("sym_e1_l10", AQUA, "-"),
             ("selected", VIOLET, ":"))
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.8), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for ax, kind in zip(axes, ("idvd", "idvg")):
        style(ax)
        for i, c in enumerate(rec.curves):
            if c.kind != kind:
                continue
            xv = np.abs(c.vd if kind == "idvd" else c.vg)
            alpha = 1.0 if c.included else 0.35
            meas = np.abs(c.target)
            s = slice(c.start, None)
            ax.scatter(xv[s], meas[s], s=7, color=INK, zorder=4, alpha=alpha, lw=0)
            if c.start:
                ax.scatter(xv[:c.start], meas[:c.start], s=7, facecolor="none", edgecolor=GREY, lw=0.5, zorder=4,
                           alpha=alpha)
            ax.plot(xv, np.abs(r["pub"][i]), color=GREY, ls="--", lw=1.2, alpha=alpha, zorder=2)
            for a, colr, ls in lines:
                ax.plot(xv, np.abs(r["sims"][a][i]), color=colr, ls=ls, lw=1.1, alpha=alpha, zorder=3)
            lab = c.key.split("@")[1]
            ax.annotate(f"{'|Vgs|' if kind == 'idvd' else '|Vds|'} {lab}" + ("" if c.included else " (excluded)"),
                        (xv[-1], max(abs(meas[-1]), 1e-14)), fontsize=6.8, color=MUTED, xytext=(3, 0),
                        textcoords="offset points", va="center")
        if kind == "idvg":
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-12)
            ax.set_xlabel("|Vgs| (V)")
            ax.set_title("Id-Vg (log scale)", fontsize=9.5)
        else:
            ax.set_xlabel("|Vds| (V)")
            ax.set_title("Id-Vd", fontsize=9.5)
        ax.set_ylabel("|Id| (A)")
    sc = r["score"]
    handles = [
        Line2D([], [], marker="o", ls="", color=INK, ms=4,
               label="measured 77 K (filled: scored points; hollow: trimmed before the metric; faded: excluded curve)"),
        Line2D([], [], color=GREY, ls="--", lw=1.2,
               label=f"NGSpice, published 77 K card (upstream committed sweep): RRMS {r['pub_rrms']:.4f}"),
        Line2D([], [], color=BLUE, lw=1.2, label=f"NGSpice re-simulated, C0_refit held-out card: RRMS "
                                                 f"{sc['C0_refit']:.4f}"),
        Line2D([], [], color=ORANGE, lw=1.2, label=f"NGSpice re-simulated, M_bias v1 held-out card x exp(g): RRMS "
                                                   f"{sc['v1']:.4f}"),
        Line2D([], [], color=GREY, ls="-.", lw=1.2, label=f"NGSpice re-simulated, sym EPS 0.01 lambda 0 held-out card "
                                                          f"x exp(g): RRMS {sc['sym_e1_l0']:.4f}"),
        Line2D([], [], color=AQUA, lw=1.2, label=f"NGSpice re-simulated, g v2 (lambda 10) held-out card x exp(g): "
                                                 f"RRMS {sc['sym_e1_l10']:.4f}"),
        Line2D([], [], color=VIOLET, ls=":", lw=1.4, label=f"NGSpice re-simulated, selected variant ({r['sel']}) "
                                                           f"held-out card x exp(g): RRMS {sc['selected']:.4f}"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=8.0, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"{tag}: held-out pairs (18-fold LODO, state C) re-simulated in ngspice-41 (card-only library deck, "
                 "clamp written) x exp(fold g);\nRRMS equals the stored rows.", fontsize=9.4, color=INK)
    fig.tight_layout(rect=(0, 0.23, 1, 0.93))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I13 / "records/real_dev.npz"), rel(I13 / "records/real_test.npz"),
            rel(I13 / "all18/heldout_cards.json"), rel(I13 / "lodo/heldout_cards.json"),
            rel(I15 / "lodo_sym_e1_l0/heldout_cards.json"), "I15_2/lodo_<selected>/heldout_cards.json",
            "upstream committed {vd,vg}_sweep/*.txt"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    folder = rf.REPORT_ROOT / f"{args.date}_{TOPIC}"
    folder.mkdir(parents=True, exist_ok=True)
    d = {k: load(p) for k, p in F.items()}
    d["v1_rows"] = d["all18"]["per_device"]["M_bias"]
    d["v2_rows"] = d["lodo"]["per_device"]["M_bias_v2"]
    d["c0_rows"] = d["lodo"]["per_device"]["C0_refit"]
    d["e1l0_rows"] = load(I15 / "lodo_sym_e1_l0/results.json")["per_device"]
    d["iv"] = resim(d)
    items, tables = [], []

    def add_table(t, desc, srcs):
        rf.render_table(t, folder / f"{t.name}.png")
        tables.append(t)
        items.append((f"{t.name}.png", desc, srcs))

    def add_fig(name, fn, desc):
        items.append((name, desc, fn(d, folder / name)))

    add_table(t_primary(d), "Primary GMR(selected, v1), secondary GMRs, lambda* counts, outcome", [rel(F["decision"])])
    add_table(t_variants(d), "Per-variant held-out RRMS (all-device, combined, median) and paired GMRs",
              [rel(F["decision"])])
    add_table(t_contrasts(d), "Ablation contrasts with the pre-registered labels", [rel(F["decision"])])
    add_table(t_neg(d), "Negative corrected gm / gds fractions on the penalty grids for every variant",
              [rel(F["decision"])])
    add_table(t_validity(d), "Validity V1-V3", [rel(F["val"]), rel(F["inner"])])
    add_fig("06_i15_2_lambda.png", fig_lambda, "GMR vs v1 and negative gm / gds fractions vs lambda")
    add_fig("07_i15_2_per_device.png", fig_per_device, "Per-device held-out RRMS and ratios to v1, lambda* per fold")
    add_fig("08_i15_2_inner_scores.png", fig_inner, "Inner selection scores per outer fold and lambda")
    for k, tag in enumerate(IV_TAGS):
        name = f"{9 + k:02d}_i15_2_iv_{tag}.png"
        items.append((name, f"I-V of {tag}: measured 77 K points, NGSpice of the published card, and NGSpice "
                            "re-simulated held-out cards of C0_refit, v1, sym EPS 0.01 lambda 0, g v2 and the selected "
                            "variant (x exp(fold g))", fig_iv(d, tag, folder / name)))
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i15_2.py --date {args.date}`"
    md = [f"# Tables, {args.date} {TOPIC}", "", f"Generated {now} by {gen}.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))
    D = d["decision"]
    T = D["variant_table"]
    p = D["primary_gmr_selected_vs_v1"]
    readme = [
        f"# Report batch {args.date} · {TOPIC}", "",
        f"Generated {now} by {gen}. NGSpice runs: re-simulation of the held-out cards of the {len(IV_TAGS)} I-V devices "
        "(checked against the stored rows); everything else from stored files.", "",
        "## Contents", "",
        f"- I15-2 (pre-registration `out/research/explore/I15_2/prereg.md`, commit {COMMITS['prereg']}; code "
        f"{COMMITS['code']}): ablation of the three g v2 training changes on the 18-fold LODO and a training-data-only "
        "nested 3-fold selection of lambda in {0, 1, 2, 5, 10} V.",
        f"- Outcome (pre-registered rule): **{D['outcome']}**; primary GMR(selected, v1) {ci(p)}, {p['wins']}/18 lower. "
        f"Selected all-device / combined {T['selected']['all_device_mean']:.5f} / {T['selected']['combined']:.5f}; "
        f"v1 {T['v1']['all_device_mean']:.5f} / {T['v1']['combined']:.5f}; g v2 {T['sym_e1_l10']['all_device_mean']:.5f}"
        f" / {T['sym_e1_l10']['combined']:.5f}.",
        f"- Changes named as costing accuracy: {', '.join(D['changes_named_costly']) or 'none'}.",
        "", "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))
    summary = [f"I15-2 ablation of the g v2 training changes and nested lambda selection: {D['outcome']} "
               f"(pre-registered rule); GMR(selected, v1) {ci(p)}.",
               f"Changes named as costing accuracy: {', '.join(D['changes_named_costly']) or 'none'}."]
    rf.update_index(args.date, TOPIC, folder, [i[0] for i in items], summary)
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
