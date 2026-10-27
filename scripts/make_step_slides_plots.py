#!/usr/bin/env python3
"""Figures and numbers for the step-by-step deck (sky130 77 K, 18 devices):
stock card -> direct NGSpice card fit -> remaining error -> correction g ->
penalty -> leave-one-device-out test -> results.

Stages:
  --stock   NGSpice runs, no fitting (about 10 s per device, 19 runs):
            (a) the stock SkyWater sky130A tt card (libs.tech/ngspice/sky130.lib.spice,
                section tt; tnom 30 C with its own temperature coefficients), simulated
                at -196.15 C for the 18 devices, NGSpice native bin selection;
            (b) the "multiplier = 1" card of the example device (the box center of the
                first fit stage: src/cryoml/cold_start.cold_centers over the 43 selected
                parameters), asserted equal to reference_scores.json cold_multiplier_one.
            Output: out/research/explore/step_deck/{stock_card.json, mult_one_example.json}
                    and curve arrays in *.npz (not committed).
  --plots   figures -> figs/step_deck/*.png (from stored files only).
  (always)  numbers -> out/research/explore/step_deck/numbers.json; every value is read
            from a stored file or recomputed from stored per-device rows (asserted
            against the stored aggregates by make_best_vs_paper_slides.numbers()).

Stored inputs: the files listed in scripts/make_best_vs_paper_slides.py, plus
  out/research/reference_scores.json (cold_multiplier_one)
  out/research/explore/mbias_deck/method_numbers.json (fit stages)
  out/research/polished/direct43_r3_fd/real/summary.json (round 3 + FD, fit state)
  out/research/explore/best_vs_paper/curves.npz (NGSpice re-simulated curves)
  out/research/explore/I22_1/decision.json (wider card set, other optimizers)

  PYTHONPATH=src .venv/bin/python scripts/make_step_slides_plots.py --stock --plots
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

EXP = ROOT / "out/research/explore"
OUT = EXP / "step_deck"
FIGS = ROOT / "figs/step_deck"
PDK_NGSPICE = ROOT / "data/raw/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice"
EXAMPLE = "pmos_L0p35_W5"

C_MEAS, C_PUB, C_STOCK, C_ONE = "#1b1b1b", "#8a8a8a", "#d9822b", "#7b5ea7"
C_C0, C_SEL = "#2a6fdb", "#c8384e"


def jload(p):
    return json.loads(Path(p).read_text())


def key_of(curve) -> str:
    return f"{curve.kind}@{abs(float(curve.fixed)):g}"


# ---------------------------------------------------------------------------
# stage 1: NGSpice runs (stock card, multiplier-one example)
# ---------------------------------------------------------------------------
def stock_lib() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    lib = OUT / "stock_lib.spice"
    lib.write_text(
        "* stock SkyWater sky130A tt corner; the section name tt_77k is only the name the\n"
        "* simulate_pdk deck asks for. tnom and temperature coefficients are the stock ones.\n"
        ".lib tt_77k\n"
        f'.lib "{PDK_NGSPICE.as_posix()}" tt\n'
        ".endl\n")
    return lib


def run_stock() -> None:
    from cryoml.card_eval import include_tags, score_card
    from cryoml.cold_start import cold_centers
    from cryoml.data_io import load_device_curves
    from cryoml.devices import PAPER_DEVICES
    from cryoml.metrics import score_device_new
    from cryoml.spice_pdk import simulate_pdk
    from cryoml.utils import device_tag

    lib = stock_lib()
    rows, arrays = {}, {}
    for d in PAPER_DEVICES:
        tag = device_tag(d.dev_type, d.L_um, d.W_um)
        curves = load_device_curves(d)
        t0 = time.time()
        sims = simulate_pdk(d.dev_type, d.L_um, d.W_um, curves, library_path=lib)
        r = score_device_new(d.dev_type, d.L_um, d.W_um, curves, sims, include_tags=include_tags(tag))
        rows[tag] = {"rrms": float(r["rrms"]), "n_curves": int(r["n_curves"]),
                     "wall_s": round(time.time() - t0, 2)}
        for c, s in zip(curves, sims):
            arrays[f"{tag}|{key_of(c)}"] = np.asarray(s, float)
        print(tag, rows[tag], flush=True)
    np.savez_compressed(OUT / "stock_card_curves.npz", **arrays)
    v = [rows[t]["rrms"] for t in rows]
    n = [rows[t]["rrms"] for t in rows if t.startswith("nmos")]
    p = [rows[t]["rrms"] for t in rows if t.startswith("pmos")]
    (OUT / "stock_card.json").write_text(json.dumps({
        "card": "stock SkyWater sky130A tt corner (libs.tech/ngspice/sky130.lib.spice, section tt), "
                "tnom 30 C with its own temperature coefficients, simulated at -196.15 C, "
                "ngspice-41 arm64, native bin selection, no parameter changed",
        "metric": "paper RRMS, curve inclusion frozen to the published-card set (include_tags)",
        "all_device_mean_rrms": float(np.mean(v)), "nmos_rrms": float(np.mean(n)),
        "pmos_rrms": float(np.mean(p)), "combined_rrms": float((np.mean(n) + np.mean(p)) / 2),
        "median": float(np.median(v)), "devices": rows}, indent=1))

    # multiplier-one card of the example device (box center of the first fit stage)
    names = tuple(jload(ROOT / "out/pdk_parameter_sensitivity/summary.json")["selected_parameters"])
    center = cold_centers(names)[EXAMPLE]["center"]
    res = score_card(EXAMPLE, center, return_sims=True)
    stored = jload(ROOT / "out/research/reference_scores.json")["cold_multiplier_one"]["devices"][EXAMPLE]["rrms"]
    assert abs(res["rrms"] - stored) < 1e-6, (res["rrms"], stored)
    d = next(x for x in PAPER_DEVICES if device_tag(x.dev_type, x.L_um, x.W_um) == EXAMPLE)
    curves = load_device_curves(d)
    np.savez_compressed(OUT / "mult_one_example.npz",
                        **{key_of(c): np.asarray(s, float) for c, s in zip(curves, res["sims"])})
    (OUT / "mult_one_example.json").write_text(json.dumps(
        {"tag": EXAMPLE, "rrms": float(res["rrms"]), "stored_rrms": float(stored)}, indent=1))
    print("multiplier-one", EXAMPLE, res["rrms"], "stored", stored)


# ---------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------
def numbers() -> dict:
    import make_best_vs_paper_slides as bvp
    N = bvp.numbers()
    R, A = N["rows"], N["agg"]
    ref = N["ref"]
    out: dict = {}
    st = jload(OUT / "stock_card.json")
    one = ref["cold_multiplier_one"]
    out["stock"] = {k: st[k] for k in ("all_device_mean_rrms", "combined_rrms", "nmos_rrms", "pmos_rrms", "median")}
    out["stock_dev"] = {t: v["rrms"] for t, v in st["devices"].items()}
    out["mult_one"] = {k: one[k] for k in ("all_device_mean_rrms", "combined_rrms", "nmos_rrms", "pmos_rrms")}
    out["mult_one_dev"] = {t: v["rrms"] for t, v in one["devices"].items()}
    out["published"] = {k: ref["released"][k] for k in ("all_device_mean_rrms", "combined_rrms", "nmos_rrms",
                                                        "pmos_rrms")}
    stages = N["mn"]["stages"]["stages"]
    fd_fit = jload(ROOT / "out/research/polished/direct43_r3_fd/real/summary.json")
    out["stages"] = [{"label": s["label"], "all": s["all_device"], "combined": s["combined"], "calls": s["calls"],
                      "wall_s": s["wall_s"], "options": s["options"]} for s in stages[1:]]
    out["fd_fit_state"] = {"all": fd_fit["all_device_mean_rrms"], "combined": fd_fit["combined_rrms"]}
    out["start_stateC"] = {"all": A["start"]["all"], "combined": A["start"]["combined"]}
    assert abs(stages[-1]["all_device"] - A["start"]["all"]) < 1e-12
    out["bank_per_device"] = int(np.load(ROOT / f"data/processed/cold_loop/seven_r1/{EXAMPLE}.npz",
                                         allow_pickle=True)["LOCAL"].shape[0])
    for k in ("C0_refit", "sel", "sel_native", "start", "published"):
        out[f"agg_{k}"] = A[k]
    out["rows"] = {k: R[k] for k in ("published", "start", "C0_refit", "sel", "sel_native")}
    out["gmr"] = {k: N["gmr"][k] for k in ("sel_vs_C0", "sel_vs_pub", "native_vs_C0", "C0_vs_pub")}
    out["lambda_star"] = N["lambda_star"]
    out["inner_full"] = N["inner_full"]
    out["vt"] = {k: {"all": v["all_device_mean"], "combined": v["combined"]} for k, v in N["vt"].items()}
    i22 = N["i22"]
    sk = i22["part_A"]["datasets"]["sky130"]
    out["b1"] = {"n_added": len(sk["b1_selected"]), "added": sk["b1_selected"],
                 "all": sk["summary"]["B1"]["all_device_mean"], "combined": sk["summary"]["B1"]["combined"],
                 "gmr_vs_C0": sk["vs_C0_refit"]["B1"]}
    out["b2"] = {"all": sk["summary"]["B2"]["all_device_mean"], "combined": sk["summary"]["B2"]["combined"],
                 "gmr_vs_C0": sk["vs_C0_refit"]["B2"]}
    out["partC"] = {k: {"label": v["label"], **{a: v["vs_LS"][a] for a in ("ratio", "ci_low", "ci_high", "wins",
                                                                           "n_used")}}
                    for k, v in i22["part_C"]["methods"].items()}
    out["partC_budget"] = [min(i22["part_C"]["budget"].values()), max(i22["part_C"]["budget"].values())]
    out["cost"] = N["cost"]["sky130"]
    out["i23_fixed"] = N["i23"]["part_B"]["datasets"]["sky130"]["cost"]
    out["i23_cmp"] = N["i23"]["part_B"]["datasets"]["sky130"]["comparisons"]["fixed_vs_selected"]
    out["i23_label"] = N["i23"]["part_B"]["rule"]["label"]
    out["i23_nistrt"] = N["i23"]["part_B"]["datasets"]["nistrt"]["comparisons"]["fixed_vs_selected"]
    out["i17"] = {"label": N["i17"]["label"], "values": N["i17"]["values_I17_1_rule"],
                  "thresholds": N["i17"]["thresholds"], "restr": N["restr"]["sel"]["totals"]}
    pdv = N["restr"]["sel"]["per_device"]
    out["i17"]["n_gds_curves"] = sum(len(pdv[t]["gds"]) for t in pdv)
    out["i17"]["n_gm_curves"] = sum(len(pdv[t].get("gm", [])) for t in pdv)
    out["acc_max_native_minus_stored"] = N["acc"]["max_abs_native_minus_stored"]
    out["check_max_abs_diff"] = N["check"]["max_abs_diff"]
    out["n_curves"] = N["mn"]["records"]["n_curves"]
    out["n_points_total"] = N["mn"]["records"]["total"]
    i20 = jload(EXP / "I20_1/decision.json")["per_dataset"]
    out["other"] = {
        "nist4": i20["nist4"]["primary_gmr_selected_vs_C0_refit"],
        "nistrt": i20["nistrt"]["primary_gmr_selected_vs_C0_refit"],
        "ihp300": jload(EXP / "I17_2/decision.json")["primary_gmr_selected_vs_C0_refit"],
        **{f"ihp{T}": jload(EXP / f"I21_1/ihp_{T}K/decision.json")["primary_gmr_selected_vs_C0_refit"]
           for T in (233, 343, 398)},
    }
    out["example"] = EXAMPLE
    out["mult_one_example"] = jload(OUT / "mult_one_example.json")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "numbers.json").write_text(json.dumps(out, indent=1, default=float))
    return out


# ---------------------------------------------------------------------------
# stage 2: figures
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.titlesize": 14,
                         "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "legend.fontsize": 12})
    return plt


def stored_curves(tag: str) -> list[dict]:
    arr = np.load(EXP / "best_vs_paper/curves.npz")
    check = jload(EXP / "best_vs_paper/check.json")
    out = []
    for i, c in enumerate(check["devices"][tag]["curves"]):
        p = f"{tag}|{i}|"
        out.append({**c, **{k: np.asarray(arr[p + k]) for k in ("vg", "vd", "meas", "published", "C0_refit",
                                                                 "sel_native")}})
    return out


def dev_order(tags):
    def key(t):
        fam, l_, w_ = t.split("_")
        return (fam == "pmos", float(l_[1:].replace("p", ".")), float(w_[1:].replace("p", ".")))
    return sorted(tags, key=key)


def pretty(tag: str) -> str:
    fam, l_, w_ = tag.split("_")
    return f"{fam} L = {l_[1:].replace('p', '.')} um, W = {w_[1:].replace('p', '.')} um"


def iv_axes(ax_vd, ax_vg, curves, series, label_curves=False):
    """series: list of (key, color, style, lw, label). Measured points always first."""
    for c in curves:
        kind = c["kind"]
        ax = ax_vd if kind == "idvd" else ax_vg
        x = np.abs(c["vd"] if kind == "idvd" else c["vg"])
        meas = np.abs(c["meas"])
        al = 1.0 if c["included"] else 0.28
        s = slice(c["start"], None)
        ax.scatter(x[s][::3], meas[s][::3], s=9, color=C_MEAS, lw=0, alpha=al, zorder=5)
        for key, col, ls, lw, _lab in series:
            if key in c:
                ax.plot(x, np.abs(c[key]), color=col, ls=ls, lw=lw, alpha=al, zorder=3)
        if label_curves and kind == "idvd":
            ax.annotate(f"|Vgs| {c['key'].split('@')[1]} V", (x[-1], np.abs(c["meas"][-1])), xytext=(4, 0),
                        textcoords="offset points", fontsize=10.5, va="center", color="#444444")
    ax_vg.set_yscale("log")
    lo = min(np.min(np.abs(c["meas"])[np.abs(c["meas"]) > 0]) for c in curves if c["kind"] == "idvg")
    ax_vg.set_ylim(bottom=max(lo / 3, 1e-13))
    ax_vd.set_xlabel("|Vds| (V)")
    ax_vg.set_xlabel("|Vgs| (V)")
    ax_vd.set_ylabel("|Id| (A)")
    ax_vg.set_ylabel("|Id| (A)")
    for ax in (ax_vd, ax_vg):
        ax.grid(alpha=0.25, lw=0.5)


def legend_handles(series, meas_label="measured, 77 K"):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=C_MEAS, ms=5, label=meas_label)]
    h += [Line2D([], [], color=col, ls=ls, lw=max(lw, 1.6), label=lab) for _k, col, ls, lw, lab in series]
    return h


def fig_measured(path):
    plt = _plt()
    curves = stored_curves(EXAMPLE)
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.3), dpi=150)
    for kind, ax in (("idvd", axes[0]), ("idvg", axes[1])):
        same = [c for c in curves if c["kind"] == kind]
        for j, c in enumerate(same):
            x = np.abs(c["vd"] if kind == "idvd" else c["vg"])
            y = np.abs(c["meas"])
            col = cmap(j / max(len(same) - 1, 1) * 0.85)
            fixed = f"|{'Vgs' if kind == 'idvd' else 'Vds'}| = {c['key'].split('@')[1]} V"
            if c["included"]:
                st = c["start"]
                ax.plot(x[st:], y[st:], "o", ms=3.2, color=col, label=fixed)
                if st > 0:
                    ax.plot(x[:st], y[:st], "o", ms=2.5, color=col, alpha=0.25)
            else:
                ax.plot(x, y, "o", ms=3.2, mfc="none", color=col, alpha=0.6, label=fixed + " (not scored)")
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_ylabel("|Id| (A)")
        ax.legend(frameon=False, fontsize=11.5, loc="upper left" if kind == "idvd" else "lower right")
    axes[1].set_yscale("log")
    axes[1].set_ylim(bottom=1e-10)
    axes[0].set_xlabel("|Vds| (V)")
    axes[1].set_xlabel("|Vgs| (V)")
    axes[0].set_title("Id-Vd: 5 curves (linear scale)")
    axes[1].set_title("Id-Vg: 6 curves (log scale)")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_stock(path, nums):
    plt = _plt()
    curves = stored_curves(EXAMPLE)
    st = np.load(OUT / "stock_card_curves.npz")
    one = np.load(OUT / "mult_one_example.npz")
    for c in curves:
        c["stock"] = st[f"{EXAMPLE}|{c['key']}"]
        c["one"] = one[c["key"]]
    series = [("stock", C_STOCK, "-", 1.8,
               f"NGSpice, stock SkyWater card as released (tnom 30 C): RRMS {nums['stock_dev'][EXAMPLE]:.2f}"),
              ("one", C_ONE, "--", 1.6,
               "NGSpice, stock values in the 77 K card file (tnom -196.15 C): RRMS "
               f"{nums['mult_one_dev'][EXAMPLE]:.1f}")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15.5, 6.3), dpi=150)
    iv_axes(a1, a2, curves, series)
    a1.set_title("Id-Vd (linear scale)")
    a2.set_title("Id-Vg (log scale)")
    fig.legend(handles=legend_handles(series), loc="upper center", ncol=1, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_stage_rrms(path, nums):
    plt = _plt()
    st = nums["stages"]
    labels = ["stock values\n(start)", "7 parameters", "43 p.\nround 1", "43 p.\nround 2", "43 p.\nround 3",
              "round 3\n+ FD"]
    alld = [nums["mult_one"]["all_device_mean_rrms"]] + [s["all"] for s in st[:4]] + [nums["fd_fit_state"]["all"]]
    comb = [nums["mult_one"]["combined_rrms"]] + [s["combined"] for s in st[:4]] + [nums["fd_fit_state"]["combined"]]
    calls = [None] + [s["calls"] for s in st]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(15.5, 6.6), dpi=150)
    ax.plot(x, alld, "o-", color=C_C0, lw=2.2, ms=8, label="all-device mean RRMS (18 devices)")
    ax.plot(x, comb, "s--", color="#5b8fd9", lw=1.6, ms=6, label="combined RRMS ((nMOS mean + pMOS mean) / 2)")
    ax.axhline(nums["published"]["all_device_mean_rrms"], color=C_PUB, ls=":", lw=2,
               label=f"published 77 K card (paper): {nums['published']['all_device_mean_rrms']:.4f}")
    ax.axhline(nums["stock"]["all_device_mean_rrms"], color=C_STOCK, ls=":", lw=2,
               label=f"stock card as released, run at 77 K: {nums['stock']['all_device_mean_rrms']:.2f}")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RRMS (log scale)")
    for i, v in enumerate(alld):
        ax.annotate(f"{v:.4f}" if v < 1 else f"{v:.2f}", (x[i], v), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=12, color=C_C0)
        if calls[i]:
            ax.annotate(f"{calls[i]:,.0f} calls", (x[i], 0.06), ha="center", fontsize=11, color="#555555")
    ax.set_ylim(0.05, 30)
    ax.grid(alpha=0.25, lw=0.5, which="both")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_residual(path, nums):
    plt = _plt()
    curves = stored_curves(EXAMPLE)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 7.0), dpi=150, sharex="col")
    cmap = plt.get_cmap("viridis")
    for c in curves:
        if not c["included"]:
            continue
        kind = c["kind"]
        col = 0 if kind == "idvd" else 1
        x = np.abs(c["vd"] if kind == "idvd" else c["vg"])[c["start"]:]
        meas = c["meas"][c["start"]:]
        card = c["C0_refit"][c["start"]:]
        den = np.mean(np.abs(meas))
        same = [k for k in curves if k["kind"] == kind and k["included"]]
        color = cmap(same.index(c) / max(len(same) - 1, 1) * 0.85)
        lab = f"|{'Vgs' if kind == 'idvd' else 'Vds'}| {c['key'].split('@')[1]} V"
        axes[0, col].plot(x, (card - meas) / den, color=color, lw=1.6, label=lab)
        ok = (np.abs(meas) > 0) & (np.abs(card) > 0) & (x > 0.02)   # |Vds| <= 0.02 V: I near 0, ratio undefined
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
        ax.set_ylim(-0.7, 0.7)
    axes[0, 0].legend(fontsize=11, frameon=False, ncol=2)
    axes[0, 1].legend(fontsize=11, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_perdevice(path, nums):
    plt = _plt()
    R = nums["rows"]
    tags = dev_order(R["sel"])
    y = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(15.5, 7.4), dpi=150)
    h = 0.27
    ax.barh(y - h, [R["published"][t] for t in tags], h, color=C_PUB, label="published 77 K card (paper)")
    ax.barh(y, [R["C0_refit"][t] for t in tags], h, color=C_C0, label="card refit, no g")
    ax.barh(y + h, [R["sel_native"][t] for t in tags], h, color=C_SEL, label="card refit + g (NGSpice element deck)")
    ax.set_yticks(y)
    ax.set_yticklabels([pretty(t) for t in tags], fontsize=11.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1])
    ax.set_xticklabels(["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1"])
    ax.set_xlabel("device RRMS (log scale)")
    ax.grid(alpha=0.25, lw=0.5, axis="x")
    ax.legend(loc="lower right", frameon=True, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_iv_page(tags, path, nums):
    plt = _plt()
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11,
                         "xtick.labelsize": 10, "ytick.labelsize": 10})
    series = [("published", C_PUB, "--", 1.3, "NGSpice, published 77 K card (paper)"),
              ("sel_native", C_SEL, "-", 1.5, "NGSpice, method: card refit + g (element deck), device held out")]
    fig = plt.figure(figsize=(16.6, 7.9), dpi=150)
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.22)
    R = nums["rows"]
    for k, tag in enumerate(tags):
        sub = gs[k // 3, k % 3].subgridspec(1, 2, wspace=0.38)
        a1, a2 = fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1])
        iv_axes(a1, a2, stored_curves(tag), series)
        a1.set_ylabel("|Id| (A)")
        a2.set_ylabel("")
        a1.set_title(pretty(tag), loc="left", x=0.0, fontsize=11.5)
        a2.set_title(f"published {R['published'][tag]:.3f}\nmethod {R['sel_native'][tag]:.3f}", fontsize=10.5,
                     loc="right")
    fig.legend(handles=legend_handles(series, "measured, 77 K (faded: not scored)"), loc="upper center",
               ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=11.5)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.07)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


EQS = {
    "rrms": r"$\mathrm{RRMS}_c = \dfrac{\sqrt{\frac{1}{n_c}\sum_{i=1}^{n_c} (I_{\mathrm{sim},i} - "
            r"I_{\mathrm{meas},i})^2}}{\frac{1}{n_c}\sum_{i=1}^{n_c} |I_{\mathrm{meas},i}|}"
            r"\qquad \mathrm{RRMS}_{\mathrm{device}} = \frac{1}{C}\sum_{c=1}^{C} \mathrm{RRMS}_c$",
    "model": r"$I = I_{\mathrm{card}}(V_{gs}, V_{ds}) \cdot \exp\left(g(x_1, x_2)\right)$",
    "inputs": r"$a = s V_{gs},\;\; b = s V_{gd},\;\; m = \sqrt{V_{ds}^2 + \mathrm{EPS}^2},\;\; "
              r"x_1 = \dfrac{a + b + m}{2\,\mathrm{VDD}},\;\; x_2 = \dfrac{m}{\mathrm{VDD}}$",
    "g": r"$g = \frac{1}{5}\sum_{j=1}^{5} \tanh\left(h_j(\mathrm{is\_pmos}, z_1, z_2)\right)$",
    "loss": r"$L = \frac{1}{D}\sum_{d=1}^{D} \mathrm{RRMS}_d\left(I_{\mathrm{card},d}\, e^{g}\right) + "
            r"10^{-4}\,\|W\|^2 + \lambda P$",
    "pen": r"$P = \mathrm{mean}\left[\mathrm{relu}\left(-\frac{\partial \ln I}{\partial |V_{gs}|}\right) + "
           r"\mathrm{relu}\left(-\frac{\partial \ln I}{\partial |V_{ds}|}\right)\right],\;\; I = I_{\mathrm{card}} e^{g}$",
    "elem": r"$I(d, s) = I_{\mathrm{card}} + \left(e^{g} - 1\right) I_{\mathrm{ch}}$",
}


def render_eqs() -> dict:
    plt = _plt()
    from PIL import Image
    sizes = {}
    for name, tex in EQS.items():
        fig = plt.figure(figsize=(0.1, 0.1), dpi=200)
        fig.text(0, 0, tex, fontsize=26, color="#1f252b")
        p = FIGS / f"eq_{name}.png"
        fig.savefig(p, dpi=200, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        plt.close(fig)
        with Image.open(p) as im:
            sizes[name] = [im.width, im.height]
    (FIGS / "eq_sizes.json").write_text(json.dumps(sizes, indent=1))
    return sizes


def plots(nums) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_measured(FIGS / "s01_measured.png")
    fig_stock(FIGS / "s02_stock.png", nums)
    fig_stage_rrms(FIGS / "s04_card_fit.png", nums)
    fig_residual(FIGS / "s05_residual.png", nums)
    fig_perdevice(FIGS / "s10_per_device.png", nums)
    tags = dev_order(nums["rows"]["sel"])
    for k in range(3):
        fig_iv_page(tags[6 * k:6 * k + 6], FIGS / f"s{11 + k}_iv.png", nums)
    render_eqs()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", action="store_true")
    ap.add_argument("--plots", action="store_true")
    a = ap.parse_args()
    if a.stock:
        run_stock()
    nums = numbers()
    if a.plots:
        plots(nums)
    print(json.dumps({k: nums[k] for k in ("stock", "mult_one", "published", "start_stateC", "fd_fit_state")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
