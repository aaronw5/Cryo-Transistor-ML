#!/usr/bin/env python3
"""Step-by-step deck, I25-1 revision: figures and numbers for the from-scratch card fits
(released 77 K card files with every tuned multiplier set to 1; 3 random starts per device;
multi-round NGSpice fit; g v2 with card refit, full data), 7-parameter and 45-parameter arms.

No NGSpice run, no fitting, no training: every curve is read from the stored I25-1 re-simulations
  out/research/explore/I25_1/arm{A,B}/final/curves.npz   (start, C0_refit, full_g, mult1, ...)
  out/research/explore/I25_1/old/published_curves.npz      (paper card)
  out/research/explore/step_deck/stock_card_curves.npz     (stock tt card at 77 K)
measured points, sweep voltages, trimming and the scored-curve flags from
  out/research/explore/best_vs_paper/{curves.npz, check.json}  (via make_step_slides_plots.stored_curves).
Every RRMS printed on a figure or written to numbers_i25.json is read from decision.json / final.json,
and each per-device value used is asserted equal between the two (|d| <= 1e-12).  Per-curve RRMS
values are recomputed from the stored curves and asserted against final.json (|d| <= 1e-9).

Outputs: figs/step_deck/i25_*.png, out/research/explore/step_deck/numbers_i25.json

  .venv/bin/python scripts/make_step_deck_i25.py
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
    plt = sp._plt()
    st = {r["stage"]: r for r in d["dec"]["stage_table"]}
    order = ["round 1", "round 2", "round 3", "round 3 + FD", "C0_refit", "g full data"]
    labels = ["round 1", "round 2", "round 3", "round 3\n+ FD", "card refit,\nno g", "card refit\n+ g"]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(15.5, 6.4), dpi=150)
    for a, col in (("A", C_7), ("B", C_45)):
        al = [st[s][f"{a}_all"] for s in order]
        ax.plot(x, al, "o-", color=col, lw=2.4, ms=8, label=f"{ARMS[a]} card, all-device RRMS")
        for i, v in enumerate(al):
            ax.annotate(f"{v:.4f}", (x[i], v), xytext=(0, 12 if a == "A" else -24), textcoords="offset points",
                        ha="center", fontsize=13, color=col)
    pa = d["dec"]["reference"]["paper"]["primary"]
    ax.axhline(pa["all_device"], color=sp.C_PUB, ls=":", lw=2, label=f"paper card, all-device {pa['all_device']:.4f}")
    ax.set_yscale("log")
    from matplotlib.ticker import NullFormatter
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0.05, 0.07, 0.1, 0.15, 0.2, 0.3])
    ax.set_yticklabels(["0.05", "0.07", "0.1", "0.15", "0.2", "0.3"])
    ax.set_ylim(0.05, 0.34)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RRMS (log scale)")
    ax.grid(alpha=0.25, lw=0.5, which="both")
    ax.legend(loc="lower left", frameon=False, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_residual(d, path):
    plt = sp._plt()
    cs = curves(d, EXAMPLE)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 7.0), dpi=150, sharex="col")
    cmap = plt.get_cmap("viridis")
    for c in cs:
        if not c["included"]:
            continue
        kind = c["kind"]
        col = 0 if kind == "idvd" else 1
        x = np.abs(c["vd"] if kind == "idvd" else c["vg"])[c["start"]:]
        meas = c["meas"][c["start"]:]
        card = c["B_C0_refit"][c["start"]:]
        den = np.mean(np.abs(meas))
        same = [k for k in cs if k["kind"] == kind and k["included"]]
        color = cmap(same.index(c) / max(len(same) - 1, 1) * 0.85)
        lab = f"|{'Vgs' if kind == 'idvd' else 'Vds'}| {c['key'].split('@')[1]} V"
        axes[0, col].plot(x, (card - meas) / den, color=color, lw=1.6, label=lab)
        ok = (np.abs(meas) > 0) & (np.abs(card) > 0) & (x > 0.02)
        axes[1, col].plot(x[ok], np.log(np.abs(meas[ok]) / np.abs(card[ok])), color=color, lw=1.6)
    axes[0, 0].set_title("Id-Vd curves")
    axes[0, 1].set_title("Id-Vg curves")
    axes[0, 0].set_ylabel("(I_card - I_meas) / den_c")
    axes[1, 0].set_ylabel("ln(I_meas / I_card)")
    axes[1, 0].set_xlabel("|Vds| (V)")
    axes[1, 1].set_xlabel("|Vgs| (V)")
    for ax in axes.ravel():
        ax.axhline(0, color="#555555", lw=0.8)
        ax.grid(alpha=0.25, lw=0.5)
    for ax in axes[1]:
        ax.set_ylim(-1.0, 1.0)
    axes[0, 0].legend(fontsize=11, frameon=False, ncol=2)
    axes[0, 1].legend(fontsize=11, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_pmos_lowvds(d, path) -> dict:
    """Id-Vg at |Vds| = 0.01 V, 10 pMOS, 45-parameter card refit without and with g (full data)."""
    plt = sp._plt()
    tags = sp.dev_order([t for t in d["fin"]["B"]["C0_refit"] if t.startswith("pmos")])
    fig, axes = plt.subplots(2, 5, figsize=(16.6, 7.4), dpi=150)
    stats = {"ln_card_over_meas_at_vdd": {}, "ln_cardg_over_meas_at_vdd": {}}
    for ax, tag in zip(axes.ravel(), tags):
        c = next(c for c in curves(d, tag) if c["key"] == "idvg@0.01")
        x = np.abs(c["vg"])
        al = 1.0 if c["included"] else 0.35
        ax.scatter(x[::2], np.abs(c["meas"])[::2] * 1e6, s=7, color=sp.C_MEAS, lw=0, alpha=al, zorder=5)
        ax.plot(x, np.abs(c["B_C0_refit"]) * 1e6, color=C_45, lw=1.6)
        ax.plot(x, np.abs(c["B_full_g"]) * 1e6, color=C_G, lw=1.6)
        top = slice(-5, None)
        m = np.mean(np.abs(c["meas"])[top])
        stats["ln_card_over_meas_at_vdd"][tag] = float(np.log(np.mean(np.abs(c["B_C0_refit"])[top]) / m))
        stats["ln_cardg_over_meas_at_vdd"][tag] = float(np.log(np.mean(np.abs(c["B_full_g"])[top]) / m))
        pc0 = d["fin"]["B"]["C0_refit"][tag]["per_curve"].get("idvg@0.01")
        pcg = d["fin"]["B"]["full_g"][tag]["per_curve"].get("idvg@0.01")
        ttl = sp.pretty(tag).replace("pmos ", "pMOS ").replace(" um, W = ", " x ").replace("L = ", "")
        ax.set_title(ttl + (f"\ncurve RRMS {pc0:.3f} -> {pcg:.3f}" if pc0 is not None else "\nnot scored"),
                     fontsize=11.5)
        ax.set_xlabel("|Vgs| (V)", fontsize=11)
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.25, lw=0.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("|Id| (uA)", fontsize=11)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=sp.C_MEAS, ms=5, label="measured, 77 K, |Vds| = 0.01 V"),
         Line2D([], [], color=C_45, lw=2, label="NGSpice, 45-parameter card fit"),
         Line2D([], [], color=C_G, lw=2, label="45-parameter card fit + g")]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=12.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    v = list(stats["ln_card_over_meas_at_vdd"].values())
    vg = list(stats["ln_cardg_over_meas_at_vdd"].values())
    stats["summary"] = {"card_min": min(v), "card_max": max(v), "card_mean": float(np.mean(v)),
                        "card_pos": int(sum(x > 0 for x in v)), "cardg_mean": float(np.mean(vg)),
                        "cardg_min": min(vg), "cardg_max": max(vg), "n": len(v),
                        "window": "last 5 points of the sweep, |Vgs| 1.81-1.85 V"}
    return stats


def gain_split(d) -> dict:
    """Share of the all-device RRMS reduction (card fit -> card fit + g) by family x curve key."""
    out = {}
    for a in "AB":
        c0, fg = d["fin"][a]["C0_refit"], d["fin"][a]["full_g"]
        tot = (np.mean([c0[t]["rrms"] for t in c0]) - np.mean([fg[t]["rrms"] for t in fg]))
        share = {}
        for t in c0:
            pc0, pcg = c0[t]["per_curve"], fg[t]["per_curve"]
            assert set(pc0) == set(pcg), t
            for k in pc0:
                kk = f"{t.split('_')[0]}|{k}"
                share[kk] = share.get(kk, 0.0) + (pc0[k] - pcg[k]) / len(pc0) / len(c0)
        tot_chk = sum(share.values())
        assert abs(tot_chk - tot) < 1e-12, (a, tot, tot_chk)
        share = {k: v / tot for k, v in share.items()}
        fam = {f: sum(v for k, v in share.items() if k.startswith(f)) for f in ("nmos", "pmos")}
        out[a] = {"total_reduction": float(tot), "family": fam,
                  "top": sorted(share.items(), key=lambda kv: -kv[1])[:6]}
    return out


def fig_perdevice(d, path):
    plt = sp._plt()
    tags = sp.dev_order(list(d["fin"]["A"]["C0_refit"]))
    y = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(15.5, 8.8), dpi=150)
    h = 0.16
    series = [(lambda t: d["old"]["arms"]["published"][t]["rrms"], sp.C_PUB, "paper card"),
              (lambda t: rrms_row(d, "A", "C0_refit", t), C_7, "7p card fit"),
              (lambda t: rrms_row(d, "A", "full_g", t), C_7G, "7p card fit + g"),
              (lambda t: rrms_row(d, "B", "C0_refit", t), C_45, "45p card fit"),
              (lambda t: rrms_row(d, "B", "full_g", t), C_G, "45p card fit + g")]
    for k, (f, col, lab) in enumerate(series):
        ax.barh(y + (k - 2) * h, [f(t) for t in tags], h, color=col, label=lab)
    ax.set_yticks(y)
    ax.set_yticklabels([sp.pretty(t) for t in tags], fontsize=12.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1])
    ax.set_xticklabels(["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1"])
    ax.set_xlabel("device RRMS (log scale)")
    ax.grid(alpha=0.25, lw=0.5, axis="x")
    ax.legend(loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=5, frameon=False, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_iv_page(d, tags, path, a):
    plt = sp._plt()
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11,
                         "xtick.labelsize": 10, "ytick.labelsize": 10})
    series = [("paper", sp.C_PUB, "--", 1.3, "NGSpice, paper card"),
              (f"{a}_C0_refit", C_7 if a == "A" else C_45, "-", 1.4, f"NGSpice, {ARMS[a]} card fit"),
              (f"{a}_full_g", C_G, "-", 1.5, f"{ARMS[a]} card fit + g (NGSpice x exp(g))")]
    fig = plt.figure(figsize=(16.6, 7.9), dpi=150)
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.22)
    for k, tag in enumerate(tags):
        sub = gs[k // 3, k % 3].subgridspec(1, 2, wspace=0.38)
        a1, a2 = fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1])
        sp.iv_axes(a1, a2, curves(d, tag), series)
        a1.set_ylabel("|Id| (A)")
        a2.set_ylabel("")
        a1.set_title(sp.pretty(tag), loc="left", x=0.0, fontsize=11.5)
        t = (f"paper {d['old']['arms']['published'][tag]['rrms']:.3f}\n{SHORT[a]} "
             f"{rrms_row(d, a, 'C0_refit', tag):.3f}, + g {rrms_row(d, a, 'full_g', tag):.3f}")
        a2.set_title(t, fontsize=10.5, loc="right")
    fig.legend(handles=sp.legend_handles(series, "measured, 77 K (faded: not scored)"), loc="upper center",
               ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=11.5)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.07)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main() -> int:
    d = load()
    worst = check_per_curve(d)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_stock(d, FIGS / "i25_stock.png")
    fig_stages(d, FIGS / "i25_stages.png")
    fig_residual(d, FIGS / "i25_residual.png")
    low = fig_pmos_lowvds(d, FIGS / "i25_pmos_lowvds.png")
    fig_perdevice(d, FIGS / "i25_per_device.png")
    tags = sp.dev_order(list(d["fin"]["A"]["C0_refit"]))
    for a in "AB":
        for k in range(3):
            fig_iv_page(d, tags[6 * k:6 * k + 6], FIGS / f"i25_iv_{SHORT[a]}_{k + 1}.png", a)
    out = {"per_curve_max_abs_diff": worst, "pmos_lowvds_45p": low, "gain_split": gain_split(d),
           "example": EXAMPLE, "example_rrms": {
               "mult1": rrms_row(d, "A", "mult1", EXAMPLE),
               "45p_card": rrms_row(d, "B", "C0_refit", EXAMPLE), "45p_card_g": rrms_row(d, "B", "full_g", EXAMPLE)}}
    (OUT / "numbers_i25.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
