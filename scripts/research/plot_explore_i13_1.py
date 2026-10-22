#!/usr/bin/env python3
"""Report batch for I13-1: BSIM4 A2 clamp state of the 43-parameter cards,
audit and fix (state C), sky130 77 K, 18 devices.

Reads stored files:

* ``out/research/explore/I13_1/{prereg.json, summary.json}``
* ``out/research/explore/I13_1/audit/audit.json`` (stored cards in states L / C / U)
* ``out/research/explore/I13_1/all18/{results, corrections, heldout_cards}.json`` (I3-2 rerun, state C)
* ``out/research/explore/I13_1/lodo/{results, corrections, heldout_cards}.json`` (I12-1 LODO rerun, state C)
* ``out/research/explore/I13_1/records/real_{dev,test}.npz`` (state-C start records)
* ``out/research/reference_scores.json`` and the upstream committed published-card sweeps

NGSpice: the I-V figures re-simulate the start card in states L and C and the
state-C rerun held-out cards (C0_refit, M_bias v1, M_bias v2) on the measured
bias grid; each re-simulated RRMS is checked against the stored row.

  PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i13_1.py --date 2026-10-22
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
from cryoml.explore import i12_1_symm as m  # noqa: E402
from cryoml.explore.i3_2_bias import correction_from_json, paired_ratio  # noqa: E402
from cryoml.explore.e1_correction import score_device  # noqa: E402

from matplotlib.lines import Line2D  # noqa: E402

rf = pe.rf
plt = pe.plt

TOPIC = "i13_1"
I13 = EXP / "I13_1"
F = {"prereg": I13 / "prereg.json", "summary": I13 / "summary.json", "audit": I13 / "audit/audit.json",
     "a18": I13 / "all18/results.json", "lodo": I13 / "lodo/results.json",
     "v1_corr": I13 / "all18/corrections.json", "v1_cards": I13 / "all18/heldout_cards.json",
     "v2_corr": I13 / "lodo/corrections.json", "v2_cards": I13 / "lodo/heldout_cards.json",
     "records": I13 / "records/records.json", "ref": ROOT / "out/research/reference_scores.json",
     "start": ROOT / "out/research/polished/direct43_r3_fd/real/summary.json"}
COMMITS = {"prereg": "d507908", "code": "837d07d", "audit": "e6b2cdd"}
ARMS = ("published", "start", "C0_refit", "T1_inloop", "M_bias_v1", "M_bias_v2")
LABEL = {"published": "published 77 K card", "start": "start (direct43_r3_fd, not refit)",
         "C0_refit": "C0_refit (no correction)", "T1_inloop": "T1_inloop (per-tag gains)",
         "M_bias_v1": "M_bias v1 (I3-2 recipe)", "M_bias_v2": "M_bias v2 (I12-1 recipe)"}
COL = {"published": GREY, "start": RED, "C0_refit": BLUE, "M_bias_v1": ORANGE, "M_bias_v2": AQUA,
       "T1_inloop": VIOLET}
MK = {"published": "s", "start": "x", "C0_refit": "o", "M_bias_v1": "D", "M_bias_v2": "^", "T1_inloop": "v"}
CLAMPED = ("nmos_L0p19_W7", "nmos_L0p25_W1p6", "nmos_L20_W0p64", "nmos_L8_W1p6", "pmos_L0p35_W1p6")


def ci(c: dict) -> str:
    return f"{c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"


def col(key, head, md=None, kind="rtext", group=None):
    return {"key": key, "head": head, "md": md or head, "kind": kind, "group": group}


def load_all() -> dict:
    d = {k: load(p) for k, p in F.items() if k not in ("v1_corr", "v2_corr", "v1_cards", "v2_cards")}
    S = d["summary"]
    for a in ARMS:
        assert len(S["per_device"][a]) == 18, a
    # recompute the headline GMR from the per-device rows
    new = {a: {t: v["new"] for t, v in S["per_device"][a].items()} for a in ARMS}
    old = {a: {t: v["old"] for t, v in S["per_device"][a].items()} for a in ARMS}
    for a, b in (("M_bias_v1", "C0_refit"), ("M_bias_v2", "C0_refit"), ("M_bias_v2", "M_bias_v1")):
        g = paired_ratio(new[a], new[b])
        assert abs(g["ratio"] - S["gmr"][f"{a}_vs_{b}"]["new"]["ratio"]) < 1e-12
    d["new"], d["old"] = new, old
    return d


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def t_headline(d: dict) -> rf.Table:
    S = d["summary"]
    T, G = S["table"], S["gmr"]
    vs = {"M_bias_v1": "M_bias_v1_vs_C0_refit", "M_bias_v2": "M_bias_v2_vs_C0_refit",
          "T1_inloop": "T1_inloop_vs_C0_refit"}
    rows = []
    for a in ARMS:
        o, n = T[a]["old"], T[a]["new"]
        g = G[vs[a]] if a in vs else None
        rows.append({"arm": LABEL[a],
                     "o": f"{o['all_device_mean']:.4f} / {o['combined']:.4f} / {o['median']:.4f}",
                     "n": f"{n['all_device_mean']:.4f} / {n['combined']:.4f} / {n['median']:.4f}",
                     "d": f"{n['all_device_mean'] - o['all_device_mean']:+.4f} / "
                          f"{n['combined'] - o['combined']:+.4f}",
                     "go": f"{ci(g['old'])}, {g['old']['wins']}/18" if g else "—",
                     "gn": f"{ci(g['new'])}, {g['new']['wins']}/18" if g else "—"})
    a2 = G["M_bias_v2_vs_M_bias_v1"]
    notes = ["old: stored rows (state L: first curve of the batch deck with A2 = 1, later curves with the card a2). "
             "new: state C (b4check.c clamp written into the card, every curve in the fresh-process state); "
             "C0_refit / T1_inloop / M_bias v1: I3-2 all18 recipe rerun; M_bias v2: I12-1 LODO recipe rerun; "
             "start: the direct43_r3_fd cards rescored (not refit); published: unchanged (no bin has a2 > 1).",
             "strict paper RRMS, curve inclusion frozen to the published card; combined = mean of the nMOS (8) and "
             "pMOS (10) family means; all-device = mean over 18.",
             f"GMR(M_bias v2, M_bias v1) old {ci(a2['old'])}, new {ci(a2['new'])} (I12-1 A2 margin 1.05). "
             "GMR: device bootstrap, 10,000 resamples, 95% percentile CI; < 1: first arm lower."]
    return rf.Table("01_table_i13_1_headline", "I13-1 headline all-18 table: stored (state L) vs rerun in state C",
                    [col("arm", "arm", kind="text"), col("o", "old all / comb / median"),
                     col("n", "new all / comb / median"), col("d", "new - old (all / comb)"),
                     col("go", "GMR vs C0_refit, old"), col("gn", "GMR vs C0_refit, new")], rows, notes)


def t_audit(d: dict) -> rf.Table:
    A = d["audit"]
    rows = []
    for a in ARMS:
        s = A["summary"][a]
        rows.append({"arm": LABEL[a], "nc": str(len(A["clamped_devices"][a])),
                     **{k: f"{s[k]['all_device_mean']:.4f} / {s[k]['combined']:.4f}" for k in ("stored", "L", "C", "U")}})
    v = A["validity"]
    notes = ["Stored cards of each arm (no refit), rescored in the three clamp states; T1 fold tables refitted from "
             "the stored records (deterministic); stored fold corrections for M_bias v1 / v2.",
             f"V1: L reproduces every stored row, max |diff| {max(v['V1_L_vs_stored_max_abs'].values()):.1e}. "
             f"V2: {v['V2_n_cards']} cards with a2 in [0.01, 1]: C = L = U, max |diff| "
             f"{v['V2_unclamped_cards_max_abs_C_or_U_minus_L']:.1e}. V3: 18 rows per arm and state.",
             "Other b4check.c clamps on size-dependent parameters (ckappas/ckappad, prwg, rdsw, rds0, rdswmin, xn, lc): "
             f"{len(A['other_clamp_messages'])} stored cards hit one. Clamp messages after writing the A2 clamp: "
             f"{len(A['clamp_messages_after_fix'])}.",
             "States: L legacy batch deck; C clamp written into the card; U warm-up op (every curve with the card a2)."]
    return rf.Table("02_table_i13_1_clamp_audit", "I13-1 audit: stored cards rescored per clamp state "
                    "(all-device / combined)",
                    [col("arm", "arm", kind="text"), col("nc", "devices with a2 > 1"),
                     col("stored", "stored"), col("L", "state L"), col("C", "state C"), col("U", "state U")],
                    rows, notes)


def t_devices(d: dict) -> rf.Table:
    A = d["audit"]["per_device"]
    S = d["summary"]["per_device"]
    rows = []
    for t in CLAMPED:
        a2 = d["start_a2"][t]
        for a in ARMS[1:]:
            r = A[a][t]
            rows.append({"dev": t if a == "start" else "", "a2": f"{a2:.3f}" if a == "start" else "",
                         "arm": a, "L": f"{r['L']:.4f}", "C": f"{r['C']:.4f}", "U": f"{r['U']:.4f}",
                         "new": f"{S[a][t]['new']:.4f}", "diff": f"{S[a][t]['diff']:+.4f}"})
    notes = ["L / C / U: the stored card of the arm rescored per clamp state. new: state-C rerun row (start: rescored "
             "start card). diff = new - stored (stored = L).",
             "a2: start card value; the refit cards of every arm also have a2 > 1 on these devices."]
    return rf.Table("03_table_i13_1_clamped_devices", "I13-1 per-device RRMS on the five devices with a2 > 1",
                    [col("dev", "device", kind="text"), col("a2", "a2 (start)"), col("arm", "arm", kind="text"),
                     col("L", "stored card, L"), col("C", "stored card, C"), col("U", "stored card, U"),
                     col("new", "new row (C)"), col("diff", "new - old")], rows, notes)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def tags_sorted(d):
    return sorted(d["new"]["C0_refit"])


def fig_per_device(d: dict, path: Path) -> list[str]:
    tags = tags_sorted(d)
    x = np.arange(len(tags))
    fig, axs = plt.subplots(2, 1, figsize=(14, 8.6), dpi=DPI, sharex=True)
    fig.patch.set_facecolor(BG)
    ax = axs[0]
    style(ax)
    for k, a in enumerate(("C0_refit", "T1_inloop", "M_bias_v1", "M_bias_v2")):
        off = (k - 1.5) * 0.17
        ax.scatter(x + off, [d["old"][a][t] for t in tags], marker=MK[a], facecolor="none", edgecolor=COL[a],
                   s=26, lw=0.9)
        ax.scatter(x + off, [d["new"][a][t] for t in tags], marker=MK[a], color=COL[a], s=26, label=LABEL[a])
    ax.scatter(x, [d["new"]["published"][t] for t in tags], marker="s", color=GREY, s=20, label=LABEL["published"])
    ax.set_yscale("log")
    ax.set_ylabel("held-out RRMS (log)")
    ax.legend(fontsize=7.5, ncol=3, frameon=False, loc="upper left")
    ax.set_title("filled: new (state C rerun); hollow: old (stored, state L)", fontsize=9)
    ax = axs[1]
    style(ax)
    for k, a in enumerate(("start", "C0_refit", "T1_inloop", "M_bias_v1", "M_bias_v2")):
        off = (k - 2) * 0.15
        ax.scatter(x + off, [d["new"][a][t] - d["old"][a][t] for t in tags], marker=MK[a], color=COL[a], s=24,
                   label=LABEL[a])
    ax.axhline(0, color=INK, lw=0.8)
    for t in CLAMPED:
        ax.axvspan(tags.index(t) - 0.45, tags.index(t) + 0.45, color=RED, alpha=0.06, lw=0)
    ax.set_ylabel("new - old RRMS")
    ax.set_ylim(top=0.07)
    ax.legend(fontsize=7.5, ncol=3, frameon=False, loc="upper right")
    ax.set_title("difference per device (shaded: a2 > 1 in the 43-parameter cards); other devices change through "
                 "the retrained fold corrections", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=60, ha="right", fontsize=7.5)
    fig.suptitle("I13-1 per-device held-out RRMS, stored (state L) vs rerun in state C", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(F["summary"])]


def fig_gmr(d: dict, path: Path) -> list[str]:
    G = d["summary"]["gmr"]
    rows = [("GMR(M_bias v1, C0_refit)", "M_bias_v1_vs_C0_refit"),
            ("GMR(M_bias v2, C0_refit) (I12-1 A1)", "M_bias_v2_vs_C0_refit"),
            ("GMR(M_bias v2, M_bias v1) (I12-1 A2)", "M_bias_v2_vs_M_bias_v1"),
            ("GMR(T1_inloop, C0_refit)", "T1_inloop_vs_C0_refit"),
            ("GMR(M_bias v1, T1_inloop)", "M_bias_v1_vs_T1_inloop")]
    fig, ax = plt.subplots(figsize=(11.5, 4.6), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    for i, (lab, k) in enumerate(rows):
        for dy, which, colr in ((0.15, "old", GREY), (-0.15, "new", BLUE)):
            c = G[k][which]
            ax.errorbar(c["ratio"], -i + dy, xerr=[[c["ratio"] - c["ci_low"]], [c["ci_high"] - c["ratio"]]],
                        fmt="o", color=colr, ms=6, capsize=3, lw=1.8)
        ax.annotate(f"old {ci(G[k]['old'])}, {G[k]['old']['wins']}/18; new {ci(G[k]['new'])}, "
                    f"{G[k]['new']['wins']}/18", (1.27, -i), fontsize=7.8, va="center", color=INK,
                    annotation_clip=False)
    ax.axvline(1.0, color=INK, lw=0.9)
    ax.axvline(1.05, color=RED, lw=0.9, ls="--")
    ax.set_yticks([-i for i in range(len(rows))])
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlim(0.6, 1.25)
    ax.set_xlabel("GMR of held-out RRMS, 18 devices, device bootstrap 10,000, 95% CI (grey: old state L, blue: new "
                  "state C); dashed: 1.05", fontsize=8.3)
    fig.suptitle("I13-1 paired contrasts before and after the clamp-state fix", fontsize=10, color=INK)
    fig.subplots_adjust(left=0.25, right=0.6, top=0.9, bottom=0.15)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(F["summary"])]


def resim(d: dict) -> dict:
    sys.path.insert(0, str(ROOT / "scripts/research/explore"))
    os.environ["CRYOML_E1_RECORDS"] = str(I13 / "records")
    from e1_collect_real import collect
    from plot_explore_i3_2 import published_rrms
    from i3_2_bias import load_records_all18
    recs = {r.tag: r for r in load_records_all18()[0]}
    c1, k1 = load(F["v1_corr"]), load(F["v1_cards"])
    c2, k2 = load(F["v2_corr"]), load(F["v2_cards"])
    start = {t: v["params"] for t, v in load(F["start"])["devices"].items()}
    pub, pub_curves = published_rrms({t: recs[t] for t in CLAMPED})
    A = d["audit"]["per_device"]
    out = {}
    for tag in CLAMPED:
        r_sl = collect(tag, start[tag], bsim4_state="legacy")
        r_sc = collect(tag, start[tag])
        r_c0 = collect(tag, k1["C0_refit"][tag])
        r_v1 = collect(tag, k1["M_bias"][tag])
        r_v2 = collect(tag, k2[tag])
        g1 = correction_from_json(c1[f"M_bias|{tag}"])
        g2 = m.SymCorrection.from_json(c2[f"M_bias_v2|{tag}"])
        sims = {"start_L": [c.sim for c in r_sl.curves], "start_C": [c.sim for c in r_sc.curves],
                "C0_refit": [c.sim for c in r_c0.curves], "M_bias_v1": g1.apply(r_v1),
                "M_bias_v2": g2.apply(r_v2)}
        sc = {"start_L": score_device(r_sl, sims["start_L"])["rrms"],
              "start_C": score_device(r_sc, sims["start_C"])["rrms"],
              "C0_refit": score_device(r_c0, sims["C0_refit"])["rrms"],
              "M_bias_v1": score_device(r_v1, sims["M_bias_v1"])["rrms"],
              "M_bias_v2": score_device(r_v2, sims["M_bias_v2"])["rrms"]}
        assert abs(sc["start_L"] - A["start"][tag]["L"]) < 1e-12
        assert abs(sc["start_C"] - A["start"][tag]["C"]) < 1e-12
        for a in ("C0_refit", "M_bias_v1", "M_bias_v2"):
            assert abs(sc[a] - d["new"][a][tag]) < 1e-9, (tag, a, sc[a], d["new"][a][tag])
        out[tag] = {"rec": recs[tag], "pub": pub_curves[tag], "pub_rrms": pub[tag], "sims": sims, "score": sc,
                    "a2": start[tag]["a2"]}
    return out


def fig_iv(d: dict, tag: str, path: Path) -> list[str]:
    r = d["iv"][tag]
    rec = r["rec"]
    lines = (("start_L", RED, ":"), ("start_C", RED, "-"), ("C0_refit", BLUE, "-"), ("M_bias_v1", ORANGE, "-"),
             ("M_bias_v2", AQUA, "-"))
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.6), dpi=DPI)
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
        Line2D([], [], color=RED, ls=":", lw=1.2,
               label=f"NGSpice, start card (direct43_r3_fd), old state L (first curve A2 = 1, others a2 = "
                     f"{r['a2']:.3f}): RRMS {sc['start_L']:.4f}"),
        Line2D([], [], color=RED, lw=1.2,
               label=f"NGSpice, start card, state C (A2 = 1 in every curve): RRMS {sc['start_C']:.4f}"),
        Line2D([], [], color=BLUE, lw=1.2,
               label=f"NGSpice re-simulated, new C0_refit held-out card (state C, no correction): RRMS "
                     f"{sc['C0_refit']:.4f}"),
        Line2D([], [], color=ORANGE, lw=1.2,
               label=f"NGSpice re-simulated, new M_bias v1 held-out card x exp(g v1 fold) (state C): RRMS "
                     f"{sc['M_bias_v1']:.4f}"),
        Line2D([], [], color=AQUA, lw=1.2,
               label=f"NGSpice re-simulated, new M_bias v2 held-out card x exp(g v2 fold) (state C): RRMS "
                     f"{sc['M_bias_v2']:.4f}"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=8.0, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"{tag} (start-card a2 {r['a2']:.3f} > 1): clamp state L vs C and the state-C rerun (LODO, "
                 "held-out card refit with the fold g in the loop).\nCurves: ngspice-41 batch deck on the measured "
                 "bias grid, times exp(g) in Python (as scored); RRMS equals the stored rows.", fontsize=9.4,
                 color=INK)
    fig.tight_layout(rect=(0, 0.2, 1, 0.93))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I13 / "records/real_dev.npz"), rel(I13 / "records/real_test.npz"), rel(F["start"]),
            rel(F["v1_cards"]), rel(F["v2_cards"]), rel(F["v1_corr"]) + f" (M_bias|{tag})",
            rel(F["v2_corr"]) + f" (M_bias_v2|{tag})", "upstream committed {vd,vg}_sweep/*.txt"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    folder = rf.REPORT_ROOT / f"{args.date}_{TOPIC}"
    folder.mkdir(parents=True, exist_ok=True)
    d = load_all()
    d["start_a2"] = {t: v["params"]["a2"] for t, v in d["start"]["devices"].items()}
    d["iv"] = resim(d)
    items, tables = [], []

    def add_table(t, desc, srcs):
        rf.render_table(t, folder / f"{t.name}.png")
        tables.append(t)
        items.append((f"{t.name}.png", desc, srcs))

    def add_fig(name, fn, desc):
        items.append((name, desc, fn(d, folder / name)))

    add_table(t_headline(d), "Headline all-18 table, old (stored, state L) vs new (rerun in state C): all-device / "
              "combined / median per arm, differences, GMR vs C0_refit", [rel(F["summary"])])
    add_table(t_audit(d), "Clamp audit: the stored cards of each arm rescored in states L / C / U (all-device / "
              "combined), validity V1-V3, other b4check.c clamps", [rel(F["audit"])])
    add_table(t_devices(d), "Per-device RRMS of the five devices with a2 > 1: stored card in L / C / U and the new "
              "state-C row", [rel(F["audit"]), rel(F["summary"])])
    add_fig("04_i13_1_per_device.png", fig_per_device, "Per-device held-out RRMS old vs new (log) and new - old per "
            "arm, 18 devices")
    add_fig("05_i13_1_gmr.png", fig_gmr, "Paired GMRs with 95% device-bootstrap CI, old vs new")
    for k, tag in enumerate(CLAMPED):
        name = f"{6 + k:02d}_i13_1_iv_{tag}.png"
        items.append((name, f"I-V of {tag}: measured 77 K points, NGSpice of the published card, the start card in "
                            "states L and C, and NGSpice re-simulated new held-out cards C0_refit, M_bias v1 x exp(g "
                            "v1), M_bias v2 x exp(g v2)", fig_iv(d, tag, folder / name)))

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i13_1.py --date {args.date}`"
    md = [f"# Tables, {args.date} {TOPIC}", "", f"Generated {now} by {gen}.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))

    S, T, G = d["summary"], d["summary"]["table"], d["summary"]["gmr"]

    def agg(a, w):
        x = T[a][w]
        return f"{x['all_device_mean']:.4f} / {x['combined']:.4f} / {x['median']:.4f}"

    readme = [
        f"# Report batch {args.date} · {TOPIC}", "",
        f"Generated {now} by {gen}. NGSpice runs: re-simulation of the start card (states L and C) and 3 new held-out "
        "cards for each of the 5 I-V devices (checked against the stored rows); everything else from stored files.", "",
        "## Contents", "",
        "- I13-1: BSIM4 A2 clamp state of the 43-parameter cards. ngspice-41 `b4check.c` clamps A2 > 1 to 1 (A1 to "
        "0) once per process per model name, `b4temp.c` rebuilds the size-dependent parameters at every analysis; the "
        "measured-curve batch deck (one dc analysis per curve) therefore used A2 = 1 for the first curve and the card "
        "a2 for the others. Fix (option b): the clamp is written into the card, every curve in the fresh-process "
        "state (C), `simulate_pdk` default. Pre-registration `out/research/explore/I13_1/prereg.md` (commit "
        f"{COMMITS['prereg']}; code {COMMITS['code']}; audit {COMMITS['audit']}).",
        "- Audit: L reproduces every stored row (0.0); published cards and the 83 cards with a2 in [0.01, 1] give "
        "identical scores in C, L and U (0.0); a2 > 1 on nmos_L0p19_W7, nmos_L0p25_W1p6, nmos_L20_W0p64, nmos_L8_W1p6, "
        "pmos_L0p35_W1p6 in every 43-parameter arm; no stored card hits another b4check.c clamp.",
        f"- Headline (all-device / combined / median), old -> new: start {agg('start', 'old')} -> {agg('start', 'new')}; "
        f"C0_refit {agg('C0_refit', 'old')} -> {agg('C0_refit', 'new')}; T1_inloop {agg('T1_inloop', 'old')} -> "
        f"{agg('T1_inloop', 'new')}; M_bias v1 {agg('M_bias_v1', 'old')} -> {agg('M_bias_v1', 'new')}; M_bias v2 "
        f"{agg('M_bias_v2', 'old')} -> {agg('M_bias_v2', 'new')}; published {agg('published', 'new')} (unchanged).",
        f"- GMR(M_bias v1, C0_refit) old {ci(G['M_bias_v1_vs_C0_refit']['old'])} -> new "
        f"{ci(G['M_bias_v1_vs_C0_refit']['new'])}; GMR(M_bias v2, C0_refit) old "
        f"{ci(G['M_bias_v2_vs_C0_refit']['old'])} -> new {ci(G['M_bias_v2_vs_C0_refit']['new'])}; GMR(M_bias v2, "
        f"M_bias v1) old {ci(G['M_bias_v2_vs_M_bias_v1']['old'])} -> new {ci(G['M_bias_v2_vs_M_bias_v1']['new'])}.",
        "", "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))
    summary = [f"I13-1 BSIM4 A2 clamp state: stored 43-parameter scores used A2 = 1 only in the first curve on 5 "
               "devices; fix: clamp written into the card (every curve in the fresh-process state).",
               f"M_bias v1 all-device / combined {T['M_bias_v1']['old']['all_device_mean']:.4f} / "
               f"{T['M_bias_v1']['old']['combined']:.4f} -> {T['M_bias_v1']['new']['all_device_mean']:.4f} / "
               f"{T['M_bias_v1']['new']['combined']:.4f}; C0_refit {T['C0_refit']['old']['all_device_mean']:.4f} -> "
               f"{T['C0_refit']['new']['all_device_mean']:.4f}; GMR(v1, C0_refit) "
               f"{ci(G['M_bias_v1_vs_C0_refit']['new'])}.",
               "Published-card scores unchanged (difference 0)."]
    rf.update_index(args.date, TOPIC, folder, [i[0] for i in items], summary)
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
