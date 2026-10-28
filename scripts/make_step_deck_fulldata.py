#!/usr/bin/env python3
"""SUPERSEDED for the deck by scripts/make_step_deck_i25.py (I25-1 revision, 2026-10-28: from-scratch
multiplier-1 start, 7- and 45-parameter arms); kept to reproduce the fd_*.png figures of the 2026-10-27 revision.

Step-by-step deck, full-data revision (no held-out device): curves and
figures for the 7-parameter and 43-parameter card fits of all 18 sky130 77 K
devices, and for the 43-parameter card + g fitted on all 18 devices.

Stages:
  --resim   NGSpice re-simulation, 18 devices, no fitting and no training:
            7-parameter card fit (out/research/ngspice_direct/seven_r1_direct,
            stock bin card + 7 fitted values; cryoml.card_eval.score_card) and
            the full-data card + g in the NGSpice element deck (I17-1:
            cards_M2.json = card refit with the all-18-device g fixed,
            osdi/sel_full.osdi = g full|sym_e1_l1, lambda 1 V).  Every RRMS is
            asserted against the stored row (|d| < 1e-9).
            Output: out/research/explore/step_deck/{fulldata_curves.npz (not
            committed), fulldata.json}
  --plots   figs/step_deck/fd_*.png

Stored inputs: out/research/explore/best_vs_paper/{curves.npz, check.json}
(measured, published card, 43-parameter card refit without g),
out/research/explore/step_deck/numbers.json (per-device rows),
out/research/explore/I17_1/deploy_sel/refit.json (full-data card + g rows).

  PYTHONPATH=src .venv/bin/python scripts/make_step_deck_fulldata.py --resim --plots
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import make_step_slides_plots as sp  # noqa: E402

EXP = ROOT / "out/research/explore"
OUT = EXP / "step_deck"
FIGS = ROOT / "figs/step_deck"
SEVEN = ROOT / "out/research/ngspice_direct/seven_r1_direct/summary.json"
I17 = EXP / "I17_1/deploy_sel"
C_SEVEN = "#2e9e6a"
TOL = 1e-9


def jload(p):
    return json.loads(Path(p).read_text())


def resim() -> None:
    sys.path.insert(0, str(ROOT / "scripts/research"))
    sys.path.insert(0, str(ROOT / "scripts/research/explore"))
    import i17_1_vds0 as drv17
    from cryoml.card_eval import device_by_tag, score_card
    from cryoml.data_io import load_device_curves
    from cryoml.explore import i14_1_chan as m
    from cryoml.explore.e1_correction import score_device
    from cryoml.metrics import tag_metric_curves
    drv = drv17.drv14
    recs = drv.load_records()
    check = jload(EXP / "best_vs_paper/check.json")
    seven = jload(SEVEN)["devices"]
    refit = jload(I17 / "refit.json")
    cards = jload(I17 / "cards_M2.json")
    corr = drv17.drv16.corr_full("sel")
    osdi = I17 / "osdi/sel_full.osdi"
    tags = sorted(check["devices"])
    assert len(tags) == 18
    arrays, out = {}, {"devices": {}}
    for tag in tags:
        rec = recs[tag]
        keys = [c["key"] for c in check["devices"][tag]["curves"]]
        assert keys == [c.key for c in rec.curves], tag
        # 7-parameter card fit: simulate_pdk order -> record order by curve key
        curves = load_device_curves(device_by_tag(tag))
        pos = {id(c): i for i, c in enumerate(curves)}
        by_key = {f"{k[0]}@{k[1]:.2f}": pos[id(c)] for k, c in tag_metric_curves(curves).items()}
        r7 = score_card(tag, seven[tag]["params"], curves=curves, return_sims=True)
        s7 = [np.asarray(r7["sims"][by_key[k]], float) for k in keys]
        # full-data card + g, NGSpice element deck
        el = drv.measured_run(tag, dict(cards[tag]), "elem", osdi, m.module_name(corr))
        sc = {"seven": float(r7["rrms"]), "full_g_native": float(score_device(rec, el["sims"])["rrms"])}
        want = {"seven": seven[tag]["rrms"], "full_g_native": refit["native"][tag]["rrms_native"]}
        diff = {k: abs(sc[k] - want[k]) for k in sc}
        for k, d in diff.items():
            assert d < TOL, (tag, k, sc[k], want[k])
        out["devices"][tag] = {"scores": sc, "stored": want, "abs_diff": diff, "msgs": el["msgs"]}
        for i in range(len(keys)):
            arrays[f"{tag}|{i}|seven"] = s7[i]
            arrays[f"{tag}|{i}|full_g_native"] = np.asarray(el["sims"][i], float)
        print(f"{tag}: 7p {sc['seven']:.5f}, 43p + g {sc['full_g_native']:.5f}", flush=True)
    out["max_abs_diff"] = {k: max(out["devices"][t]["abs_diff"][k] for t in tags) for k in ("seven", "full_g_native")}
    out["n_msgs"] = sum(bool(out["devices"][t]["msgs"]) for t in tags)
    out["simulator"] = "ngspice-41 (7p: card_eval default build; 43p + g: OSDI build, BSIM4 state C)"
    np.savez_compressed(OUT / "fulldata_curves.npz", **arrays)
    (OUT / "fulldata_check.json").write_text(json.dumps(out, indent=1))
    print("max |re-simulated - stored|:", out["max_abs_diff"], "messages:", out["n_msgs"])


def agg(d: dict) -> dict:
    n = [v for t, v in d.items() if t.startswith("nmos")]
    p = [v for t, v in d.items() if t.startswith("pmos")]
    return {"all": float(np.mean(list(d.values()))), "combined": float((np.mean(n) + np.mean(p)) / 2),
            "nmos": float(np.mean(n)), "pmos": float(np.mean(p)), "n": len(d)}


def numbers() -> dict:
    from cryoml.explore.i10_1_rt4k import gmr_ci
    N = jload(OUT / "numbers.json")
    R = N["rows"]
    seven = jload(SEVEN)
    refit = jload(I17 / "refit.json")
    rows = {"published": R["published"], "start": R["start"], "C0_refit": R["C0_refit"],
            "seven": {t: r["rrms"] for t, r in seven["devices"].items()},
            "full_g_python": {t: r["rrms"] for t, r in refit["rows"].items()},
            "full_g_native": {t: r["rrms_native"] for t, r in refit["native"].items()}}
    out = {"rows": rows, "agg": {k: agg(v) for k, v in rows.items()},
           "seven_calls_per_device": seven["mean_ngspice_calls_per_device"],
           "full_g_refit_calls_mean": float(np.mean([r["n_sim"] for r in refit["rows"].values()])),
           "full_g_native_minus_python_max": max(abs(rows["full_g_native"][t] - rows["full_g_python"][t])
                                                 for t in rows["full_g_native"]),
           "gmr": {"full_g_vs_C0": gmr_ci(rows["full_g_native"], rows["C0_refit"]),
                   "full_g_vs_pub": gmr_ci(rows["full_g_native"], rows["published"]),
                   "C0_vs_pub": gmr_ci(rows["C0_refit"], rows["published"]),
                   "seven_vs_pub": gmr_ci(rows["seven"], rows["published"])},
           "sources": {"seven": str(SEVEN.relative_to(ROOT)), "full_g": str((I17 / "refit.json").relative_to(ROOT)),
                       "other": "out/research/explore/step_deck/numbers.json"},
           "recipe_full_g": "g full|sym_e1_l1 (I15-2) trained on all 18 devices' records, EPS 0.01 V, lambda 1 V "
                            "(lambda* of the 18-device inner cross-validation); card refit with g fixed from "
                            "direct43_r3_fd, TRF max_nfev 400, irls 2, diff_step 0.01, state C (I17-1)"}
    (OUT / "fulldata.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def curves_all(tag: str) -> list[dict]:
    cs = sp.stored_curves(tag)
    fa = np.load(OUT / "fulldata_curves.npz")
    for i, c in enumerate(cs):
        for k in ("seven", "full_g_native"):
            c[k] = np.asarray(fa[f"{tag}|{i}|{k}"])
    return cs


def fig_basics(path, tag="nmos_L1_W1p6"):
    plt = sp._plt()
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14})
    cs = [c for c in sp.stored_curves(tag) if c["included"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6.4), dpi=150)
    cmap = plt.get_cmap("viridis")
    vd_c = [c for c in cs if c["kind"] == "idvd"]
    vg_c = [c for c in cs if c["kind"] == "idvg"]
    for j, c in enumerate(vd_c):
        x, y = np.abs(c["vd"]), np.abs(c["meas"]) * 1e3
        a1.plot(x[::4], y[::4], "o", ms=3.5, color=cmap(j / max(len(vd_c) - 1, 1) * 0.85))
        a1.annotate(f"Vgs = {c['key'].split('@')[1]} V", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=14, va="center")
    top = vd_c[-1]
    x, y = np.abs(top["vd"]), np.abs(top["meas"]) * 1e3
    i0 = int(np.argmin(np.abs(x - 1.3)))
    sl = np.polyfit(x[i0 - 10:i0 + 10], y[i0 - 10:i0 + 10], 1)[0]
    xx = np.array([x[i0] - 0.45, x[i0] + 0.45])
    a1.plot(xx, y[i0] + sl * (xx - x[i0]), color="#c8384e", lw=2.4)
    a1.annotate("gds = dId/dVds\n(slope at fixed Vgs)", (x[i0], y[i0]), xytext=(-40, -90),
                textcoords="offset points", fontsize=15, color="#c8384e",
                arrowprops={"arrowstyle": "->", "color": "#c8384e"})
    a1.set_xlabel("Vds = drain-source voltage (V)")
    a1.set_ylabel("Id = drain current (mA)")
    a1.set_title("Id-Vd: one curve per Vgs", fontsize=16)
    a1.set_xlim(0, 2.25)
    for j, c in enumerate(vg_c):
        x, y = np.abs(c["vg"]), np.abs(c["meas"]) * 1e3
        a2.plot(x[::4], y[::4], "o", ms=3.5, color=cmap(j / max(len(vg_c) - 1, 1) * 0.85),
                label=f"Vds = {c['key'].split('@')[1]} V")
    top = vg_c[-1]
    x, y = np.abs(top["vg"]), np.abs(top["meas"]) * 1e3
    i0 = int(np.argmin(np.abs(x - 1.1)))
    sl = np.polyfit(x[i0 - 10:i0 + 10], y[i0 - 10:i0 + 10], 1)[0]
    xx = np.array([x[i0] - 0.3, x[i0] + 0.3])
    a2.plot(xx, y[i0] + sl * (xx - x[i0]), color="#c8384e", lw=2.4)
    a2.annotate("gm = dId/dVgs\n(slope at fixed Vds)", (x[i0], y[i0]), xytext=(30, -60),
                textcoords="offset points", fontsize=15, color="#c8384e",
                arrowprops={"arrowstyle": "->", "color": "#c8384e"})
    a2.set_xlabel("Vgs = gate-source voltage (V)")
    a2.set_ylabel("Id (mA)")
    a2.set_title("Id-Vg: one curve per Vds", fontsize=16)
    a2.legend(loc="upper left", frameon=False, fontsize=14)
    a2.set_xlim(0, 2.25)
    for ax in (a1, a2):
        ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_perdevice(path, F):
    plt = sp._plt()
    R = F["rows"]
    tags = sp.dev_order(R["seven"])
    y = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(15.5, 8.6), dpi=150)
    h = 0.2
    series = [("published", sp.C_PUB, "paper card"), ("seven", C_SEVEN, "7-parameter card fit"),
              ("C0_refit", sp.C_C0, "43-parameter card fit"),
              ("full_g_native", sp.C_SEL, "43-parameter card fit + g (element deck)")]
    for k, (key, col, lab) in enumerate(series):
        ax.barh(y + (k - 1.5) * h, [R[key][t] for t in tags], h, color=col, label=lab)
    ax.set_yticks(y)
    ax.set_yticklabels([sp.pretty(t) for t in tags], fontsize=12.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1])
    ax.set_xticklabels(["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1"])
    ax.set_xlabel("device RRMS (log scale)")
    ax.grid(alpha=0.25, lw=0.5, axis="x")
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=4, frameon=False, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_iv_page(tags, path, F, which):
    plt = sp._plt()
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11,
                         "xtick.labelsize": 10, "ytick.labelsize": 10})
    R = F["rows"]
    if which == "seven":
        series = [("published", sp.C_PUB, "--", 1.3, "NGSpice, paper card"),
                  ("seven", C_SEVEN, "-", 1.5, "NGSpice, 7-parameter card fit")]
    else:
        series = [("published", sp.C_PUB, "--", 1.3, "NGSpice, paper card"),
                  ("C0_refit", sp.C_C0, "-", 1.3, "NGSpice, 43-parameter card fit"),
                  ("full_g_native", sp.C_SEL, "-", 1.5, "NGSpice, 43-parameter card fit + g (element deck)")]
    fig = plt.figure(figsize=(16.6, 7.9), dpi=150)
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.22)
    for k, tag in enumerate(tags):
        sub = gs[k // 3, k % 3].subgridspec(1, 2, wspace=0.38)
        a1, a2 = fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1])
        sp.iv_axes(a1, a2, curves_all(tag), series)
        a1.set_ylabel("|Id| (A)")
        a2.set_ylabel("")
        a1.set_title(sp.pretty(tag), loc="left", x=0.0, fontsize=11.5)
        if which == "seven":
            t = f"paper {R['published'][tag]:.3f}\n7p {R['seven'][tag]:.3f}"
        else:
            t = (f"paper {R['published'][tag]:.3f}\n43p {R['C0_refit'][tag]:.3f}, "
                 f"+ g {R['full_g_native'][tag]:.3f}")
        a2.set_title(t, fontsize=10.5, loc="right")
    fig.legend(handles=sp.legend_handles(series, "measured, 77 K (faded: not scored)"), loc="upper center",
               ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=11.5)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.07)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plots(F) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_basics(FIGS / "fd_basics.png")
    fig_perdevice(FIGS / "fd_per_device.png", F)
    tags = sp.dev_order(F["rows"]["seven"])
    for which in ("seven", "43"):
        for k in range(3):
            fig_iv_page(tags[6 * k:6 * k + 6], FIGS / f"fd_iv_{which}_{k + 1}.png", F, which)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resim", action="store_true")
    ap.add_argument("--plots", action="store_true")
    a = ap.parse_args()
    if a.resim:
        resim()
    F = numbers()
    if a.plots:
        plots(F)
    print(json.dumps({k: F["agg"][k] for k in F["agg"]}, indent=1))
    print(json.dumps({k: {x: F["gmr"][k][x] for x in ("ratio", "ci_low", "ci_high", "wins", "n_used")}
                      for k in F["gmr"]}, indent=1))
    print("7p calls/device", F["seven_calls_per_device"], "full g refit calls", F["full_g_refit_calls_mean"],
          "native - python max", F["full_g_native_minus_python_max"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
