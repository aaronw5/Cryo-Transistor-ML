#!/usr/bin/env python3
"""Step-by-step deck, I25-1 revision, card only: figures and numbers for the from-scratch BSIM4 card fits
(paper's 77 K card files with every tuned multiplier set to 1; 3 random starts per device; multi-round NGSpice
fit; final card = Step A card refit by TRF without any correction, stored as C0_refit), 7-parameter (arm A) and
45-parameter (arm B) sets.  No correction g enters any figure or number.

No NGSpice run, no fitting, no training: every curve is read from the stored I25-1 NGSpice re-simulations
  out/research/explore/I25_1/arm{A,B}/final/curves.npz   (C0_refit, start, mult1)
  out/research/explore/I25_1/old/published_curves.npz      (paper card)
measured points, sweep voltages, trimming and the scored-curve flags from
  out/research/explore/best_vs_paper/{curves.npz, check.json}  (via make_step_slides_plots.stored_curves).
Every RRMS printed on a figure or written to numbers_i25_card.json is read from decision.json / final.json,
and each per-device value used is asserted equal between the two (|d| <= 1e-12).  Per-curve RRMS
values are recomputed from the stored curves and asserted against final.json (|d| <= 1e-9).
Extra paired GMRs use i3_2_bias.paired_ratio (the decision.json code; one decision GMR reproduced as a check).

Outputs: figs/step_deck/i25_{stages_card,per_device_card,err_split,lowvds_card,iv_card_*}.png,
         out/research/explore/step_deck/numbers_i25_card.json

  PYTHONPATH=src .venv/bin/python scripts/make_step_deck_i25.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import make_step_slides_plots as sp  # noqa: E402

EXP = ROOT / "out/research/explore"
I25 = EXP / "I25_1"
OUT = EXP / "step_deck"
FIGS = ROOT / "figs/step_deck"
ARMS = {"A": "7-parameter", "B": "45-parameter"}
SHORT = {"A": "7p", "B": "45p"}
C_7, C_7G = "#2e9e6a", "#9fd3b8"
C_45, C_G = sp.C_C0, sp.C_SEL
EXAMPLE = "pmos_L0p35_W5"


def jload(p):
    return json.loads(Path(p).read_text())


def load() -> dict:
    d = {"dec": jload(I25 / "decision.json"),
         "fin": {a: jload(I25 / f"arm{a}/final/final.json")["arms"] for a in "AB"},
         "cv": {a: dict(np.load(I25 / f"arm{a}/final/curves.npz")) for a in "AB"},
         "pub": dict(np.load(I25 / "old/published_curves.npz")),
         "old": jload(I25 / "old/old.json")}
    for a in "AB":
        for k in ("mult1", "C0_refit", "full_g"):
            for t, v in d["dec"]["arms"][a]["per_device"][k].items():
                fv = d["fin"][a][k][t]["rrms"]
                dv = v if isinstance(v, float) else v["rrms"] if isinstance(v, dict) else float(v)
                assert abs(dv - fv) <= 1e-12, (a, k, t, dv, fv)
    return d


def rrms_row(d, a, k, t) -> float:
    return float(d["fin"][a][k][t]["rrms"])


def curves(d, tag) -> list[dict]:
    cs = sp.stored_curves(tag)
    st = np.load(OUT / "stock_card_curves.npz")
    for c in cs:
        key = c["key"]
        n = len(c["meas"])
        c["paper"] = np.asarray(d["pub"][f"published|{tag}|{key}"], float)
        c["stock"] = np.asarray(st[f"{tag}|{key}"], float)
        for a in "AB":
            for k in ("mult1", "C0_refit", "full_g"):
                c[f"{a}_{k}"] = np.asarray(d["cv"][a][f"{k}|{tag}|{key}"], float)
        for k in [k for k in c if k in ("paper", "stock") or k[:2] in ("A_", "B_")]:
            assert len(c[k]) == n, (tag, key, k)
    return cs


def curve_rrms(c, sim) -> float:
    s = slice(c["start"], None)
    m = np.asarray(c["meas"], float)[s]
    return float(np.sqrt(np.mean((np.asarray(sim, float)[s] - m) ** 2)) / np.mean(np.abs(m)))


def check_per_curve(d) -> float:
    worst = 0.0
    for tag in d["fin"]["A"]["C0_refit"]:
        for c in curves(d, tag):
            if not c["included"]:
                continue
            for a in "AB":
                for k in ("C0_refit", "full_g"):
                    want = d["fin"][a][k][tag]["per_curve"][c["key"]]
                    got = curve_rrms(c, c[f"{a}_{k}"])
                    worst = max(worst, abs(got - want))
            want = d["old"]["arms"]["published"][tag]["per_curve"][c["key"]]
            worst = max(worst, abs(curve_rrms(c, c["paper"]) - want))
    assert worst <= 1e-9, worst
    return worst


# ---------------------------------------------------------------- figures
def fig_stock(d, path):
    plt = sp._plt()
    cs = curves(d, EXAMPLE)
    stock = jload(OUT / "stock_card.json")["devices"][EXAMPLE]
    stock = stock["rrms"] if isinstance(stock, dict) else stock
    series = [("stock", sp.C_STOCK, "-", 1.8,
               f"NGSpice, stock SkyWater tt card as released (tnom 30 C), run at 77 K: RRMS {stock:.2f}"),
              ("A_mult1", sp.C_ONE, "--", 1.6,
               "NGSpice, paper's 77 K card file with every tuned multiplier = 1 (fit start): RRMS "
               f"{rrms_row(d, 'A', 'mult1', EXAMPLE):.1f}")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15.5, 6.3), dpi=150)
    sp.iv_axes(a1, a2, cs, series)
    a1.set_title("Id-Vd (linear scale)")
    a2.set_title("Id-Vg (log scale)")
    fig.legend(handles=sp.legend_handles(series), loc="upper center", ncol=1, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(path, facecolor="white")
    plt.close(fig)



def fig_stages(d, path):
    """All-device RRMS of the card-only fit after each Step A stage and the final card refit, both arms."""
    plt = sp._plt()
    st = {r["stage"]: r for r in d["dec"]["stage_table"]}
    order = ["round 1", "round 2", "round 3", "round 3 + FD", "C0_refit"]
    labels = ["round 1", "round 2", "round 3", "round 3\n+ FD", "final card\n(refit, TRF)"]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(15.5, 6.4), dpi=150)
    for a, col in (("A", C_7), ("B", C_45)):
        al = [st[s][f"{a}_all"] for s in order]
        ax.plot(x, al, "o-", color=col, lw=2.4, ms=8, label=f"our {ARMS[a]} card (NGSpice), all-device RRMS")
        for i, v in enumerate(al):
            ax.annotate(f"{v:.4f}", (x[i], v), xytext=(0, 12 if a == "A" else -24), textcoords="offset points",
                        ha="center", fontsize=13, color=col)
    pa = d["dec"]["reference"]["paper"]["primary"]
    ax.axhline(pa["all_device"], color=sp.C_PUB, ls=":", lw=2,
               label=f"paper card (NGSpice), all-device {pa['all_device']:.4f}")
    ax.set_yscale("log")
    from matplotlib.ticker import NullFormatter
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0.1, 0.12, 0.15, 0.2, 0.25, 0.3])
    ax.set_yticklabels(["0.10", "0.12", "0.15", "0.20", "0.25", "0.30"])
    ax.set_ylim(0.095, 0.32)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RRMS (log scale)")
    ax.grid(alpha=0.25, lw=0.5, which="both")
    ax.legend(loc="center right", frameon=False, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_perdevice(d, path):
    """Per-device RRMS bars, card only: paper card, our 7-parameter card, our 45-parameter card."""
    plt = sp._plt()
    tags = sp.dev_order(list(d["fin"]["A"]["C0_refit"]))
    y = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(15.5, 8.8), dpi=150)
    h = 0.26
    series = [(lambda t: d["old"]["arms"]["published"][t]["rrms"], sp.C_PUB, "paper card"),
              (lambda t: rrms_row(d, "A", "C0_refit", t), C_7, "our 7-parameter card"),
              (lambda t: rrms_row(d, "B", "C0_refit", t), C_45, "our 45-parameter card")]
    for k, (f, col, lab) in enumerate(series):
        ax.barh(y + (k - 1) * h, [f(t) for t in tags], h, color=col, label=lab + " (NGSpice)")
    ax.set_yticks(y)
    ax.set_yticklabels([sp.pretty(t) for t in tags], fontsize=12.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1])
    ax.set_xticklabels(["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1"])
    ax.set_xlabel("device RRMS (log scale)")
    ax.grid(alpha=0.25, lw=0.5, axis="x")
    ax.legend(loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=3, frameon=False, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_iv_page(d, tags, path, a):
    plt = sp._plt()
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11,
                         "xtick.labelsize": 10, "ytick.labelsize": 10})
    series = [("paper", sp.C_PUB, "--", 1.4, "paper card (NGSpice)"),
              (f"{a}_C0_refit", C_7 if a == "A" else C_45, "-", 1.6, f"our {ARMS[a]} card (NGSpice)")]
    fig = plt.figure(figsize=(16.6, 7.9), dpi=150)
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.22)
    for k, tag in enumerate(tags):
        sub = gs[k // 3, k % 3].subgridspec(1, 2, wspace=0.38)
        a1, a2 = fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1])
        sp.iv_axes(a1, a2, curves(d, tag), series)
        a1.set_ylabel("|Id| (A)")
        a2.set_ylabel("")
        a1.set_title(sp.pretty(tag), loc="left", x=0.0, fontsize=11.5)
        t = (f"RRMS: paper {d['old']['arms']['published'][tag]['rrms']:.3f}\n"
             f"our {SHORT[a]} {rrms_row(d, a, 'C0_refit', tag):.3f}")
        a2.set_title(t, fontsize=10.5, loc="right")
    fig.legend(handles=sp.legend_handles(series, "measured, 77 K (faded: not scored)"), loc="upper center",
               ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=12)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.07)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


CLASSES = [("idvd", "Id-Vd (5 curves)"), ("idvg_low", "Id-Vg at |Vds| 0.01 V"),
           ("idvg_high", "Id-Vg at |Vds| 0.37-1.85 V")]


def curve_class(key: str) -> str:
    return "idvd" if key.startswith("idvd") else ("idvg_low" if key == "idvg@0.01" else "idvg_high")


def error_split(d) -> dict:
    """Additive split of the all-device RRMS of the final card (C0_refit) by family x curve class:
    device RRMS = mean of its scored per-curve RRMS, all-device = mean over 18 devices, so each curve
    contributes RRMS_c / C_d / 18.  Also the mean per-curve RRMS of each class."""
    out = {}
    for a in "AB":
        fin = d["fin"][a]["C0_refit"]
        contrib, per = {}, {}
        for t, row in fin.items():
            fam = t.split("_")[0]
            pc = row["per_curve"]
            for k, v in pc.items():
                kk = f"{fam}|{curve_class(k)}"
                contrib[kk] = contrib.get(kk, 0.0) + v / len(pc) / len(fin)
                per.setdefault(kk, []).append(v)
        tot = float(np.mean([r["rrms"] for r in fin.values()]))
        assert abs(sum(contrib.values()) - tot) < 1e-12, (a, tot)
        out[a] = {"all_device": tot, "contribution": contrib,
                  "share": {k: v / tot for k, v in contrib.items()},
                  "mean_curve_rrms": {k: float(np.mean(v)) for k, v in per.items()},
                  "n_curves": {k: len(v) for k, v in per.items()}}
    return out


def lowvds_stats(d) -> dict:
    """ln(I_card / I_meas) on the Id-Vg at |Vds| 0.01 V, mean of the last 5 sweep points (|Vgs| 1.81-1.85 V),
    final card of each arm, every device; and the top-|Vgs| Id-Vd onset of the 10 pMOS:
    r(|Vds|) = ln(|I_meas| / |I_card|) and G ratio G(0.15 VDD) / G(0.03 VDD), G = |I| / |Vds|."""
    vdd = 1.85
    tags = sp.dev_order(list(d["fin"]["A"]["C0_refit"]))
    out = {"ln_card_over_meas_vds001": {a: {} for a in "AB"}, "onset": {}, "g_ratio": {"meas": {}, "A": {}, "B": {}}}
    fr = [0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
    for tag in tags:
        cs = {c["key"]: c for c in curves(d, tag)}
        c = cs["idvg@0.01"]
        m = np.mean(np.abs(c["meas"])[-5:])
        for a in "AB":
            out["ln_card_over_meas_vds001"][a][tag] = float(np.log(np.mean(np.abs(c[f"{a}_C0_refit"])[-5:]) / m))
        if not tag.startswith("pmos"):
            continue
        c = cs["idvd@1.85"]
        vd = np.abs(c["vd"])
        o = np.argsort(vd)
        meas = np.abs(c["meas"])[o]
        vd = vd[o]
        out["onset"][tag] = {}
        for a in "AB":
            card = np.abs(c[f"{a}_C0_refit"])[o]
            out["onset"][tag][a] = {f"{f:g}": float(np.log(np.interp(f * vdd, vd, meas) / np.interp(f * vdd, vd, card)))
                                    for f in fr}
        for k, arr in (("meas", meas), ("A", np.abs(c["A_C0_refit"])[o]), ("B", np.abs(c["B_C0_refit"])[o])):
            g15 = np.interp(0.15 * vdd, vd, arr) / (0.15 * vdd)
            g03 = np.interp(0.03 * vdd, vd, arr) / (0.03 * vdd)
            out["g_ratio"][k][tag] = float(g15 / g03)
    s = {}
    for a in "AB":
        v = out["ln_card_over_meas_vds001"][a]
        p = [x for t, x in v.items() if t.startswith("pmos")]
        n = [x for t, x in v.items() if t.startswith("nmos")]
        on = {f"{f:g}": [out["onset"][t][a][f"{f:g}"] for t in out["onset"]] for f in fr}
        s[a] = {"pmos_min": min(p), "pmos_max": max(p), "pmos_mean": float(np.mean(p)),
                "pmos_pos": int(sum(x > 0 for x in p)), "n_pmos": len(p),
                "nmos_min": min(n), "nmos_max": max(n), "nmos_mean": float(np.mean(n)),
                "nmos_pos": int(sum(x > 0 for x in n)), "n_nmos": len(n),
                "onset_mean": {f: float(np.mean(x)) for f, x in on.items()},
                "onset_pos": {f: int(sum(y > 0 for y in x)) for f, x in on.items()},
                "g_ratio_min": min(out["g_ratio"][a].values()), "g_ratio_max": max(out["g_ratio"][a].values()),
                "g_ratio_gt1": int(sum(x > 1 for x in out["g_ratio"][a].values()))}
    gm = out["g_ratio"]["meas"]
    s["meas"] = {"g_ratio_min": min(gm.values()), "g_ratio_max": max(gm.values()),
                 "g_ratio_gt1": int(sum(x > 1 for x in gm.values())), "n": len(gm)}
    out["summary"] = s
    out["window"] = "Id-Vg |Vds| 0.01 V: last 5 sweep points, |Vgs| 1.81-1.85 V; onset: Id-Vd at |Vgs| 1.85 V"
    return out


def fig_err_split(d, split, path):
    plt = sp._plt()
    fig = plt.figure(figsize=(16.6, 6.6), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    c = next(c for c in curves(d, EXAMPLE) if c["key"] == "idvg@0.01")
    x = np.abs(c["vg"])
    k = x >= 1.2
    s = c["start"]
    ax.scatter(x[k][::2], np.abs(c["meas"])[k][::2] * 1e6, s=12, color=sp.C_MEAS, lw=0, zorder=5,
               alpha=0.35, label="measured, 77 K (faded: not scored)")
    ax.scatter(x[s:], np.abs(c["meas"])[s:] * 1e6, s=18, color=sp.C_MEAS, lw=0, zorder=6)
    ax.plot(x[k], np.abs(c["paper"])[k] * 1e6, color=sp.C_PUB, ls="--", lw=1.6, label="paper card (NGSpice)")
    ax.plot(x[k], np.abs(c["A_C0_refit"])[k] * 1e6, color=C_7, lw=2, label="our 7-parameter card (NGSpice)")
    ax.plot(x[k], np.abs(c["B_C0_refit"])[k] * 1e6, color=C_45, lw=2, label="our 45-parameter card (NGSpice)")
    ax.set_xlabel("|Vgs| (V)")
    ax.set_ylabel("|Id| (uA)")
    ax.set_title("pMOS L 0.35 x W 5 um, Id-Vg at |Vds| = 0.01 V", fontsize=13.5)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=11.5, frameon=False, loc="lower right")
    ax = fig.add_subplot(gs[0, 1])
    rows = [(f, cl, lab) for f in ("nmos", "pmos") for cl, lab in CLASSES]
    y = np.arange(len(rows))
    hb = 0.38
    for j, (a, col) in enumerate((("A", C_7), ("B", C_45))):
        v = [split[a]["contribution"][f"{f}|{cl}"] for f, cl, _ in rows]
        ax.barh(y + (j - 0.5) * hb, v, hb, color=col,
                label=f"our {ARMS[a]} card: all-device RRMS {split[a]['all_device']:.4f}")
        for yi, vi in zip(y, v):
            ax.annotate(f"{vi:.4f} ({vi / split[a]['all_device']:.0%})", (vi, yi + (j - 0.5) * hb),
                        xytext=(4, 0), textcoords="offset points", va="center", fontsize=11, color=col)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{'nMOS' if f == 'nmos' else 'pMOS'}\n{lab}" for f, _, lab in rows], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.092)
    ax.set_xlabel("contribution to all-device RRMS (sum over rows = all-device RRMS)")
    ax.set_title("Where the all-device RRMS of the final card comes from", fontsize=13.5)
    ax.grid(alpha=0.25, lw=0.5, axis="x")
    ax.legend(fontsize=11.5, frameon=False, loc="upper right")
    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.11)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_lowvds_card(d, low, path):
    plt = sp._plt()
    tags = sp.dev_order(list(d["fin"]["A"]["C0_refit"]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16.6, 6.6), dpi=150, gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(len(tags))
    for j, (a, col, mk) in enumerate((("A", C_7, "o"), ("B", C_45, "s"))):
        v = [low["ln_card_over_meas_vds001"][a][t] for t in tags]
        a1.scatter(v, y + (j - 0.5) * 0.25, color=col, marker=mk, s=46, zorder=4, label=f"our {ARMS[a]} card")
    a1.axvline(0, color="#555555", lw=1)
    a1.axhline(7.5, color="#999999", lw=0.8, ls=":")
    a1.set_yticks(y)
    a1.set_yticklabels([sp.pretty(t) for t in tags], fontsize=11)
    a1.invert_yaxis()
    a1.set_xlabel("ln(I_card / I_meas), Id-Vg at |Vds| 0.01 V, |Vgs| 1.81-1.85 V")
    a1.set_title("Id-Vg at |Vds| 0.01 V (> 0: card above measured)", fontsize=12.5)
    a1.grid(alpha=0.25, lw=0.5, axis="x")
    a1.legend(fontsize=11.5, frameon=False, loc="lower right")
    fr = [0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
    for a, col in (("A", C_7), ("B", C_45)):
        for t, v in low["onset"].items():
            a2.plot(fr, [v[a][f"{f:g}"] for f in fr], color=col, lw=0.8, alpha=0.3)
        a2.plot(fr, [low["summary"][a]["onset_mean"][f"{f:g}"] for f in fr], "o-", color=col, lw=2.6, ms=6,
                label=f"our {ARMS[a]} card: mean of 10 pMOS")
    a2.axhline(0, color="#555555", lw=1)
    a2.set_xscale("log")
    a2.set_xticks(fr)
    a2.set_xticklabels([f"{f:g}" for f in fr])
    a2.set_xlabel("|Vds| / VDD (VDD = 1.85 V), log scale")
    a2.set_ylabel("ln(I_meas / I_card)")
    a2.set_title("pMOS Id-Vd at |Vgs| 1.85 V (< 0: card above measured)", fontsize=12.5)
    a2.grid(alpha=0.25, lw=0.5)
    a2.legend(fontsize=11.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def gmrs(d) -> dict:
    """Paired GMRs not in decision.json, same bootstrap code (i3_2_bias.paired_ratio, 10,000 device resamples,
    seed 0): 7p final card vs 45p final card; Step A cards (round 3 + FD) vs the earlier run's round 3 + FD cards
    that started from the paper card file with its tuned values (7p: bank + center start)."""
    from cryoml.explore.i3_2_bias import paired_ratio
    pdv = {a: {k: {t: (v["rrms"] if isinstance(v, dict) else float(v)) for t, v in
                   d["dec"]["arms"][a]["per_device"][k].items()} for k in ("start", "C0_refit")} for a in "AB"}
    old = {k: {t: v["rrms"] for t, v in d["old"]["arms"][k].items()} for k in ("old_fd", "old_7p", "published")}
    out = {"A_card_vs_B_card": paired_ratio(pdv["A"]["C0_refit"], pdv["B"]["C0_refit"]),
           "B_start_vs_old43_fd": paired_ratio(pdv["B"]["start"], old["old_fd"]),
           "A_start_vs_old7p": paired_ratio(pdv["A"]["start"], old["old_7p"])}
    # reproduction check of a decision.json GMR with the same code
    chk = paired_ratio(pdv["A"]["C0_refit"], old["published"])
    ref = d["dec"]["gmr"]["A"]["primary"]["C0_refit_vs_paper"]
    assert abs(chk["ratio"] - ref["ratio"]) < 1e-12 and abs(chk["ci_low"] - ref["ci_low"]) < 1e-12, (chk, ref)
    out["aggregates"] = {"old_fd": {"all_device": float(np.mean(list(old["old_fd"].values())))},
                         "old_7p": {"all_device": float(np.mean(list(old["old_7p"].values())))}}
    return out


def fig_one_card(d, path, key, color, label, size=(15.5, 6.0), dpi=150):
    """One device (EXAMPLE): measured and one card (NGSpice), Id-Vd linear and Id-Vg log."""
    plt = sp._plt()
    cs = curves(d, EXAMPLE)
    series = [(key, color, "-", 2.0, label)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=size, dpi=dpi)
    sp.iv_axes(a1, a2, cs, series)
    a1.set_title("Id-Vd, one curve per |Vgs| (linear scale)")
    a2.set_title("Id-Vg, one curve per |Vds| (log scale)")
    fig.legend(handles=sp.legend_handles(series, "measured, 77 K"), loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


TRACE_TAG = "pmos_L2_W5"


def search_trace(tag=TRACE_TAG, pre="full") -> dict:
    """Best-so-far RRMS after each search stage of the 3 starts of one device (stored Step A logs), with the
    cumulative NGSpice runs.  Stages per round: CMA-ES (cmaes_evals runs), least squares (remaining runs)."""
    out = {}
    for s in range(3):
        pts, runs, best = [], 0, None
        for st in ("r1", "r2", "r3"):
            v = jload(I25 / f"stepA/{pre}_s{s}_{st}.json")["devices"][tag]
            draws = v.get("n_sim_start_draws") or 0
            if st == "r1":
                runs += draws
                best = v["start_scores"][0]
                pts.append({"stage": "start", "round": 0, "runs": runs, "rrms": best})
            runs += v["cmaes_evals"]
            best = min(best, v["cmaes_best"])
            pts.append({"stage": "global", "round": int(st[1]), "runs": runs, "rrms": best})
            runs += v["n_sim"] - v["cmaes_evals"] - draws
            best = v["rrms"]
            pts.append({"stage": "local", "round": int(st[1]), "runs": runs, "rrms": best})
        v = jload(I25 / f"stepA/{pre}_s{s}_fd.json")["devices"][tag]
        runs += v["n_sim"]
        pts.append({"stage": "local", "round": 4, "runs": runs, "rrms": v["rrms"]})
        out[str(s)] = pts
    return out


def fig_trace(tr, path):
    plt = sp._plt()
    fig, ax = plt.subplots(figsize=(15.5, 6.2), dpi=150)
    cols = ["#2a6fdb", "#d9822b", "#7b5ea7"]
    for k, (s, pts) in enumerate(tr.items()):
        x = [p["runs"] for p in pts]
        y = [p["rrms"] for p in pts]
        ax.plot(x, y, "-", color=cols[k], lw=2, label=f"start {k + 1}: final RRMS {y[-1]:.3f}")
        for p in pts:
            mk = {"start": "*", "global": "o", "local": "s"}[p["stage"]]
            ax.plot(p["runs"], p["rrms"], mk, color=cols[k], ms=12 if mk == "*" else 8, zorder=5)
    for r, x0 in ((1, 0), (2, tr["0"][3]["runs"]), (3, tr["0"][5]["runs"])):
        ax.axvline(x0, color="#999999", lw=0.8, ls=":")
        ax.text(x0 + 300, 0.22, f"round {r}", fontsize=14, color="#555555")
    ax.set_yscale("log")
    from matplotlib.ticker import NullFormatter
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0.1, 0.15, 0.2, 0.3, 0.5, 1, 2, 3])
    ax.set_yticklabels(["0.10", "0.15", "0.20", "0.30", "0.50", "1", "2", "3"])
    ax.set_ylim(0.09, 3.6)
    ax.set_xlabel("NGSpice runs (cumulative, one start)")
    ax.set_ylabel("device RRMS (log scale)")
    ax.set_title(f"{sp.pretty(TRACE_TAG)}, 45 parameters: star = random start, circle = after global search, "
                 "square = after least squares", fontsize=13)
    ax.grid(alpha=0.25, lw=0.5, which="both")
    ax.legend(loc="center right", frameon=False, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def start_spread(pre="full") -> dict:
    """Device RRMS of each of the 3 starts after the last round + polish (stored Step A logs)."""
    out = {}
    for s in range(3):
        for t, v in jload(I25 / f"stepA/{pre}_s{s}_fd.json")["devices"].items():
            out.setdefault(t, [None] * 3)[s] = float(v["rrms"])
    return out


def fig_rounds_starts(d, spread, path):
    plt = sp._plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16.6, 6.4), dpi=150, gridspec_kw={"width_ratios": [1, 1.5]})
    st = {r["stage"]: r for r in d["dec"]["stage_table"]}
    order, labels = ["round 1", "round 2", "round 3 + FD", "C0_refit"], ["round 1", "round 2", "round 3\n+ last polish", "final card"]
    x = np.arange(len(order))
    for a, col in (("A", C_7), ("B", C_45)):
        y = [st[s][f"{a}_all"] for s in order]
        a1.plot(x, y, "o-", color=col, lw=2.4, ms=8, label=f"{ARMS[a]} card")
        for xi, yi in zip(x, y):
            a1.annotate(f"{yi:.4f}", (xi, yi), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=12,
                        color=col)
    a1.set_xticks(x)
    a1.set_xticklabels(labels)
    a1.set_ylim(0.09, 0.19)
    a1.set_ylabel("all-device RRMS")
    a1.set_title("All-device RRMS after each round (best start per device)", fontsize=13)
    a1.grid(alpha=0.25, lw=0.5)
    a1.legend(frameon=False, fontsize=12.5, loc="upper right")
    tags = sp.dev_order(list(spread))
    yy = np.arange(len(tags))
    for i, t in enumerate(tags):
        v = spread[t]
        a2.plot([min(v), max(v)], [i, i], color="#bbbbbb", lw=1.5, zorder=1)
        a2.scatter(v, [i] * 3, color="#8a8a8a", s=30, zorder=2)
        a2.scatter([min(v)], [i], color=C_45, s=60, zorder=3)
    a2.scatter([], [], color="#8a8a8a", s=30, label="each of the 3 starts")
    a2.scatter([], [], color=C_45, s=60, label="best start (kept)")
    a2.set_yticks(yy)
    a2.set_yticklabels([sp.pretty(t) for t in tags], fontsize=10.5)
    a2.invert_yaxis()
    a2.set_xscale("log")
    a2.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5])
    a2.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5"])
    a2.set_xlabel("device RRMS after round 3 and last polish (log scale)")
    a2.set_title("45 parameters: RRMS of the 3 starts per device", fontsize=13)
    a2.grid(alpha=0.25, lw=0.5, axis="x")
    a2.legend(frameon=False, fontsize=12, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------- step figures, one example device (45 parameters)
TRACE_DIR = OUT / "round1_trace"
COLS3 = ["#2a6fdb", "#d9822b", "#7b5ea7"]


def load_trace(tag=EXAMPLE):
    js = jload(TRACE_DIR / f"{tag}.json")
    z = dict(np.load(TRACE_DIR / f"{tag}.npz"))
    return js, z


def raw_curves(tag=EXAMPLE):
    """Measured curves in NGSpice-run order (the order of the rerun's stored sims)."""
    from cryoml.card_eval import device_by_tag
    from cryoml.data_io import load_device_curves
    return load_device_curves(device_by_tag(tag))


def fig_paper_example(d, path):
    rr = d["old"]["arms"]["published"][EXAMPLE]["rrms"]
    fig_one_card(d, path, "paper", "#555555", f"paper card, NGSpice (device RRMS {rr:.3f})", size=(10.6, 4.4),
                 dpi=220)


def fig_starts(js, z, path):
    """Measured vs the 3 random start cards (NGSpice): Id-Vd at |Vgs| 1.85 V and Id-Vg at |Vds| 1.85 V."""
    plt = sp._plt()
    cs = raw_curves()
    pick = {"idvd": None, "idvg": None}
    for i, c in enumerate(cs):
        if abs(abs(float(c.fixed)) - 1.85) < 1e-6:
            pick[c.kind] = i
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), dpi=200)
    for ax, kind in zip(axes, ("idvd", "idvg")):
        i = pick[kind]
        c = cs[i]
        x = np.abs(np.asarray(c.Vd if kind == "idvd" else c.Vg, float))
        ax.scatter(x[::3], np.abs(np.asarray(c.Id, float))[::3], s=12, color=sp.C_MEAS, zorder=5, label="measured")
        for k in range(3):
            s = np.abs(z[f"s{k}_sims_start_{i}"])
            ax.plot(x, s, color=COLS3[k], lw=[4, 2, 2][k], ls=["-", "--", "-"][k],
                    label=f"start {k + 1} (device RRMS {js['starts'][str(k)]['start_score']:.2f})")
        if kind == "idvg":
            ax.set_yscale("log")
            ax.set_xlabel("|Vgs| (V)")
            ax.set_title("Id-Vg at |Vds| = 1.85 V (log scale)")
        else:
            ax.set_xlabel("|Vds| (V)")
            ax.set_title("Id-Vd at |Vgs| = 1.85 V")
        ax.set_ylabel("|Id| (A)")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].legend(fontsize=12, frameon=False, loc="center right")
    fig.suptitle("pMOS L 0.35 um x W 5 um, 45 parameters: 3 random start cards (NGSpice)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_cma(js, z, path):
    """Round 1 CMA-ES only: every candidate (faint) and best so far per start; VTH0 / U0 samples by generation."""
    plt = sp._plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.8), dpi=200, gridspec_kw={"width_ratios": [1.7, 1]})
    for k in range(3):
        ph = z[f"s{k}_log_phase"]
        r = z[f"s{k}_log_rrms"][ph == "cma"]
        x = np.arange(1, len(r) + 1)
        a1.scatter(x, np.clip(r, None, 50), s=2, color=COLS3[k], alpha=0.12, lw=0)
        a1.plot(x, np.minimum.accumulate(r), color=COLS3[k], lw=2.4, label=f"start {k + 1}: best so far")
    a1.set_yscale("log")
    a1.set_ylim(0.08, 60)
    a1.set_xlim(0, 8100)
    a1.set_xlabel("NGSpice runs in round 1, global search")
    a1.set_ylabel("device RRMS (log scale)")
    a1.set_title("Round 1, CMA-ES: each dot = one candidate card", fontsize=13)
    a1.grid(alpha=0.25, lw=0.5, which="both")
    a1.legend(fontsize=12, frameon=False, loc="upper right")
    k = 1
    ph = z[f"s{k}_log_phase"]
    m = ph == "cma"
    gen = z[f"s{k}_log_gen"][m]
    sc = a2.scatter(np.abs(z[f"s{k}_log_vth0"][m]), z[f"s{k}_log_u0"][m], c=gen, s=3, cmap="viridis", lw=0)
    b = js["starts"][str(k)]
    a2.set_xlim(*sorted(abs(v) for v in b["box_vth0"]))
    a2.set_ylim(*sorted(b["box_u0"]))
    a2.set_yscale("log")
    a2.set_xlabel("|VTH0| (V)")
    a2.set_ylabel("U0 (m^2/Vs), log scale")
    a2.set_title("Start 2: sampled VTH0, U0 (axes = allowed range)", fontsize=13)
    cb = fig.colorbar(sc, ax=a2)
    cb.set_label("generation")
    fig.suptitle("pMOS L 0.35 um x W 5 um, 45 parameters", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_ls(js, z, path):
    """Round 1 least squares only: device RRMS of every run after the CMA-ES result, best so far; one curve
    before / after."""
    plt = sp._plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.0, 4.4), dpi=200, gridspec_kw={"width_ratios": [1.4, 1]})
    n = js["starts"]["1"]["n_active"]
    iters = {}
    for k in range(3):
        ph = z[f"s{k}_log_phase"]
        r = z[f"s{k}_log_rrms"]
        m = np.array([p.startswith("ls") for p in ph])
        rr, pp = r[m], ph[m]
        best = np.minimum.accumulate(rr)
        it = list(np.flatnonzero(pp == "ls_jac_base")) + [len(rr) - 1]
        y = [float(best[i]) for i in it]
        iters[str(k)] = y
        a1.plot(np.arange(len(y)), y, "o-", color=COLS3[k], lw=2.2, ms=6, label=f"start {k + 1}")
    a1.set_xlabel(f"least-squares iteration (1 iteration = {n} + 2 NGSpice runs)")
    a1.set_ylabel("device RRMS")
    a1.set_title("Round 1: device RRMS after each iteration", fontsize=13)
    a1.grid(alpha=0.25, lw=0.5)
    a1.legend(fontsize=12, frameon=False, loc="center right")
    k = "1"
    cs = raw_curves()
    from cryoml.metrics import curve_trim
    from cryoml.card_eval import device_by_tag
    dev = device_by_tag(EXAMPLE)
    rows = {}
    for i, c in enumerate(cs):
        key = f"{c.kind}@{abs(float(c.fixed)):g}"
        if key not in d_included(EXAMPLE):
            continue
        st, cleaned, den = curve_trim(dev.dev_type, dev.L_um, dev.W_um, c.kind, c.Id)
        t = np.asarray(cleaned[st:], float)
        rows[key] = [float(np.sqrt(np.mean((np.asarray(z[f"s{k}_sims_{lab}_{i}"], float)[st:] - t) ** 2)) / den)
                     for lab in ("cma_best", "final")]
    keys = sorted(rows, key=lambda q: (q.split("@")[0], float(q.split("@")[1])))
    y = np.arange(len(keys))
    a2.barh(y - 0.2, [rows[q][0] for q in keys], 0.4, color="#bbbbbb", label="after global search")
    a2.barh(y + 0.2, [rows[q][1] for q in keys], 0.4, color=C_45, label="after least squares")
    lab = {q: (f"Id-Vd, |Vgs| {q.split('@')[1]} V" if q.startswith("idvd") else f"Id-Vg, |Vds| {q.split('@')[1]} V")
           for q in keys}
    a2.set_yticks(y)
    a2.set_yticklabels([lab[q] for q in keys], fontsize=11)
    a2.invert_yaxis()
    a2.set_xscale("log")
    a2.set_xlabel("curve RRMS (log scale)")
    a2.set_title("Start 2: RRMS of each scored curve", fontsize=13)
    a2.grid(alpha=0.25, lw=0.5, axis="x")
    a2.legend(fontsize=11, frameon=False, loc="lower right")
    fig.suptitle("pMOS L 0.35 um x W 5 um, 45 parameters", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    big = max(keys, key=lambda q: rows[q][0] - rows[q][1])
    return {"curve": big, "rrms_before_after": rows[big], "per_curve": rows, "iterations": iters}


_INCL: dict = {}


def d_included(tag):
    if tag not in _INCL:
        _INCL[tag] = {c["key"] for c in sp.stored_curves(tag) if c["included"]}
    return _INCL[tag]


def fig_rounds_example(d, path) -> dict:
    """Left: allowed range per round (selected start) for VTH0, U0, RDSW with the value found in that round.
    Right: device RRMS after rounds 1-3 for the 3 starts, final card marked."""
    plt = sp._plt()
    sel = d["dec"]["arms"]["B"]["multistart"]["starts_selected"][EXAMPLE]
    pars = [("vth0", "|VTH0| (V)"), ("u0", "U0 (m^2/Vs)"), ("rdsw", "RDSW (ohm um)")]
    fig = plt.figure(figsize=(12.6, 5.0), dpi=200)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1], hspace=1.0, wspace=0.25)
    out = {"selected_start": sel, "ranges": {}, "rrms": {}}
    for j, (p, lab) in enumerate(pars):
        ax = fig.add_subplot(gs[j, 0])
        for r in (1, 2, 3):
            spec = jload(I25 / ("specs/full_r1.json" if r == 1 else f"specs/full_s{sel}_r{r}.json"))
            lo, hi = sorted(abs(v) for v in spec["bounds_by_device"][EXAMPLE][p])
            val = abs(jload(I25 / f"stepA/full_s{sel}_r{r}.json")["devices"][EXAMPLE]["params"][p])
            ax.plot([lo, hi], [r, r], color="#9fb8e8", lw=9, solid_capstyle="butt")
            ax.plot(val, r, "o", color=C_45, ms=8)
            out["ranges"].setdefault(p, []).append([lo, hi, val])
        ax.set_xscale("log")
        if p == "vth0":
            from matplotlib.ticker import NullFormatter
            ax.set_xticks([0.2, 0.3, 0.5, 1, 2])
            ax.set_xticklabels(["0.2", "0.3", "0.5", "1", "2"])
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(["round 1", "round 2", "round 3"], fontsize=11)
        ax.set_ylim(3.6, 0.4)
        ax.set_xlabel(lab + ", log scale", fontsize=11.5)
        ax.grid(alpha=0.25, lw=0.5, axis="x")
        if j == 0:
            ax.set_title(f"Start {sel + 1}: allowed range (bar) and value found (dot)", fontsize=13)
    ax = fig.add_subplot(gs[:, 1])
    st_names = ["r1", "r2", "r3", "fd"]
    xs = [1, 2, 3, 3.7]
    for k in range(3):
        y = [jload(I25 / f"stepA/full_s{k}_{s}.json")["devices"][EXAMPLE]["rrms"] for s in st_names]
        out["rrms"][str(k)] = y
        ax.plot(xs, y, "o-", color=COLS3[k], lw=2.2, ms=7, label=f"start {k + 1}")
        ax.annotate(f"{y[-1]:.3f}", (xs[-1], y[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=12, color=COLS3[k])
    fin = rrms_row(d, "B", "C0_refit", EXAMPLE)
    out["final"] = fin
    ax.plot(4.5, fin, "*", color="#c8384e", ms=16, label=f"final card (start {sel + 1}, refit): {fin:.3f}")
    ax.set_xticks([1, 2, 3, 3.7, 4.5])
    ax.set_xticklabels(["round 1", "round 2", "round 3", "+ least\nsquares", "final"], fontsize=11)
    ax.set_xlim(0.7, 4.9)
    ax.set_ylabel("device RRMS")
    ax.set_title("Device RRMS after each round", fontsize=13)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=11.5, frameon=False, loc="center right")
    fig.suptitle("pMOS L 0.35 um x W 5 um, 45 parameters", fontsize=14)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.12)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return out


def fig_screen(path) -> dict:
    """Screen of 115 candidates: max response (RMS current change / mean measured current) per candidate,
    threshold, kept / not kept, the 7 paper parameters, the changes vs the earlier 43."""
    plt = sp._plt()
    sm = jload(I25 / "screen_mult1/summary.json")
    p = sm["parameters"]
    thr = sm["effective_threshold"]
    resp = {n: max(v["max_full_grid_response"] or 0.0, v["max_high_voltage_response"] or 0.0) for n, v in p.items()}
    for n, v in p.items():
        assert v["selected"] == (resp[n] > thr or v["mandatory_seven_parameter"]), n
    nz = sorted([n for n in resp if resp[n] > 0], key=lambda n: -resp[n])
    fig, ax = plt.subplots(figsize=(16.6, 6.4), dpi=150)
    x = np.arange(len(nz))
    cols = [C_45 if p[n]["selected"] else "#c9ced3" for n in nz]
    ax.bar(x, [resp[n] for n in nz], color=cols, width=0.8)
    ax.axhline(thr, color="#c8384e", lw=1.5, ls="--")
    ax.text(len(nz) - 1, thr * 2.2, f"threshold {thr:g}", ha="right", fontsize=12, color="#c8384e")
    seven = [n for n in nz if p[n]["mandatory_seven_parameter"]]
    ax.plot([nz.index(n) for n in seven], [resp[n] * 3 for n in seven], "v", color="#2e9e6a", ms=9,
            label="the paper's 7")
    for n, lab in (("cdsc", "CDSC (added)"), ("dvt0w", "DVT0W (added)"), ("pdiblc1", "PDIBLC1 (added)"),
                   ("cdscd", "CDSCD (dropped)")):
        i = nz.index(n)
        off = {"pdiblc1": (0, 40), "dvt0w": (90, 110), "cdsc": (120, 50), "cdscd": (30, -70)}[n]
        ax.annotate(lab, (i, resp[n]), xytext=off, textcoords="offset points",
                    ha="center", fontsize=11.5, arrowprops={"arrowstyle": "-", "color": "#555555"})
    ax.set_yscale("log")
    ax.set_ylim(1e-12, 1e5)
    ax.set_xticks([])
    ax.set_xlabel(f"{len(nz)} candidate parameters with a nonzero response, sorted "
                  f"({len(resp) - len(nz)} of {len(resp)} with zero response not shown)")
    ax.set_ylabel("RMS current change / mean |I_meas|")
    from matplotlib.patches import Patch
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=[Patch(color=C_45, label=f"kept ({sm['selected_count']})"),
                       Patch(color="#c9ced3", label="not kept")] + h, fontsize=12, frameon=False, loc="upper right")
    ax.set_title("Screen: each of 115 BSIM4 parameters changed by 10 %, largest change over 18 devices", fontsize=13)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return {"n_zero": len(resp) - len(nz), "n_nonzero": len(nz), "threshold": thr}


def active_counts() -> dict:
    """Parameters the fit can move per device in the 45-parameter round-1 box (ParameterBox.active: range wider
    than 1e-14 x max(|bound|, 1)); zero values and values below about 1e-14 stay fixed."""
    sys.path.insert(0, str(ROOT / "scripts" / "research"))
    from cold_loop import box_for
    spec = jload(I25 / "specs/full_r1.json")
    out = {}
    for t in spec["published_by_device"]:
        b = box_for(spec, t)
        out[t] = {"n": int(b.active.sum()), "fixed": [n for n, a in zip(b.names, b.active) if not a]}
    return out


def main() -> int:
    d = load()
    worst = check_per_curve(d)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_stages(d, FIGS / "i25_stages_card.png")
    fig_perdevice(d, FIGS / "i25_per_device_card.png")
    split = error_split(d)
    low = lowvds_stats(d)
    fig_err_split(d, split, FIGS / "i25_err_split.png")
    fig_lowvds_card(d, low, FIGS / "i25_lowvds_card.png")
    tags = sp.dev_order(list(d["fin"]["A"]["C0_refit"]))
    for a in "AB":
        for k in range(3):
            fig_iv_page(d, tags[6 * k:6 * k + 6], FIGS / f"i25_iv_card_{SHORT[a]}_{k + 1}.png", a)
    fig_one_card(d, FIGS / "i25_goal_stock.png", "stock", sp.C_STOCK,
                 "stock SkyWater card, NGSpice at 77 K")
    fig_one_card(d, FIGS / "i25_start_card.png", "A_mult1", sp.C_ONE,
                 "start card, NGSpice", size=(11.6, 4.5), dpi=200)
    tr = search_trace()
    spread = start_spread()
    js, z = load_trace()
    assert js["all_match"], "round-1 rerun does not match the stored round-1 records"
    fig_paper_example(d, FIGS / "i25_ex_paper.png")
    fig_starts(js, z, FIGS / "i25_ex_starts.png")
    fig_cma(js, z, FIGS / "i25_ex_cma.png")
    ls_info = fig_ls(js, z, FIGS / "i25_ex_ls.png")
    rounds_info = fig_rounds_example(d, FIGS / "i25_ex_rounds.png")
    screen_info = fig_screen(FIGS / "i25_screen.png")
    stock = jload(OUT / "stock_card.json")["devices"][EXAMPLE]
    wf = {"trace_tag": TRACE_TAG, "trace": tr, "spread_45p": spread,
          "spread_45p_gt10pct": int(sum(max(v) / min(v) > 1.1 for v in spread.values())),
          "example_stock_rrms": stock["rrms"] if isinstance(stock, dict) else stock,
          "example": {"tag": EXAMPLE, "rerun_matches_stored": js["all_match"],
                      "start_rrms": [js["starts"][str(k)]["start_score"] for k in range(3)],
                      "cmaes_best": [js["starts"][str(k)]["cmaes_best"] for k in range(3)],
                      "round1_rrms": [js["starts"][str(k)]["rrms"] for k in range(3)],
                      "cmaes_evals": [js["starts"][str(k)]["cmaes_evals"] for k in range(3)],
                      "ls_runs": [js["starts"][str(k)]["n_sim"] - js["starts"][str(k)]["cmaes_evals"] - 1
                                  for k in range(3)],
                      "n_active": js["starts"]["1"]["n_active"], "popsize": js["starts"]["1"]["popsize"],
                      "ls_curve": ls_info, "rounds": rounds_info,
                      "paper_rrms": d["old"]["arms"]["published"][EXAMPLE]["rrms"]},
          "screen": screen_info, "active_45p": active_counts()}
    for t, v in spread.items():   # best start after round 3 + FD equals the stored selected start card
        assert abs(min(v) - d["dec"]["arms"]["B"]["per_device"]["start"][t]) < 1e-12, t
    out = {"per_curve_max_abs_diff": worst, "error_split": split, "lowvds": low, "gmr_extra": gmrs(d),
           "workflow": wf,
           "example": EXAMPLE, "example_rrms": {
               "mult1": rrms_row(d, "A", "mult1", EXAMPLE), "7p_card": rrms_row(d, "A", "C0_refit", EXAMPLE),
               "45p_card": rrms_row(d, "B", "C0_refit", EXAMPLE),
               "paper": d["old"]["arms"]["published"][EXAMPLE]["rrms"]}}
    (OUT / "numbers_i25_card.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("per_curve_max_abs_diff", "error_split", "gmr_extra")}, indent=1,
                     default=float)[:4000])
    print(json.dumps(low["summary"], indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
