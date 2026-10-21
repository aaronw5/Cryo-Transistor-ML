#!/usr/bin/env python3
"""Report batch for I11-1: the 43-parameter card + shared bias-only correction
(M_bias, full-data model, OSDI element) in ngspice-41, sky130 77 K, 18 devices.

Reads stored files only (no NGSpice, no fit, no training):

* ``out/research/explore/I11_1/{prereg.json, prereg.md}``
* ``out/research/explore/I11_1/runs/full/{compile, refit, q12, q3, summary,
  decision, diagnosis, timing}.json`` and ``arrays/*.npz``
* ``out/research/explore/I3_2/all18/corrections.json`` (g of the full model)
* ``out/research/explore/E1/real_{dev,test}`` (measured bias points)

Checks: the decision is recomputed from the stored summary values with
``i11_1_deploy.decide`` and compared with decision.json; the summary values
are recomputed from q12.json / q3.json.

  PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i11_1.py --date 2026-10-21
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_explore_e2_e3_e4 as pe  # noqa: E402  (helpers, style)
from plot_explore_e2_e3_e4 import (BG, INK, MUTED, GREY, BLUE, ORANGE, RED, VIOLET, AQUA, DPI,  # noqa: E402
                                   EXP, load, rel, style)
from cryoml.explore import i11_1_deploy as m  # noqa: E402

rf = pe.rf
plt = pe.plt

TOPIC = "i11_1"
I11 = EXP / "I11_1"
RUN = I11 / "runs/full"
PREREG = I11 / "prereg.json"
F = {k: RUN / f"{k}.json" for k in ("compile", "refit", "q12", "q3", "summary", "decision", "diagnosis",
                                     "timing")}
CORR = EXP / "I3_2/all18/corrections.json"
COMMITS = {"prereg": "da03859"}
ARM_STYLE = {"published": (GREY, ":", "published card"), "C0_refit": (BLUE, "-", "C0_refit card"),
             "M_card": (ORANGE, "--", "M_full card, no g"), "M_full": (RED, "-", "M_full card + g (OSDI)")}
LODO_RRMS = {"published": 0.2751, "C0_refit": 0.1043, "T1_inloop": 0.0744, "M_bias": 0.0732}


def corr():
    from cryoml.explore.i3_2_bias import correction_from_json
    return correction_from_json(load(CORR)["full|M_bias"])


def devices() -> dict:
    from cryoml.card_eval import device_by_tag
    from cryoml.cold_start import device_bins
    out = {}
    for t, b in sorted(device_bins().items()):
        d = device_by_tag(t)
        out[t] = m.Dev(t, d.dev_type, float(d.L_um), float(d.W_um), int(b))
    return out


def measured_points() -> dict:
    from cryoml.explore.e1_correction import load_devices
    recs = load_devices(EXP / "E1/real_dev") + load_devices(EXP / "E1/real_test")
    out = {"nmos": [], "pmos": []}
    for r in recs:
        for c in r.curves:
            out[r.family].append(np.column_stack([np.abs(c.vg), np.abs(c.vd)]))
    return {k: np.vstack(v) for k, v in out.items()}


def arrays(tag: str) -> dict:
    with np.load(RUN / "arrays" / f"q12_{tag}.npz") as z:
        return {k: z[k] for k in z.files}


def e(v: float) -> str:
    return f"{v:.2e}"


def load_all() -> dict:
    d = {k: load(p) for k, p in F.items()}
    d["prereg"] = load(PREREG)
    d["devs"] = devices()
    d["corr"] = corr()
    s = d["summary"]["summary_values"]
    re = m.decide(s)
    assert re == d["decision"], (re, d["decision"])
    q12 = d["q12"]["devices"]
    assert abs(max(r["q1_tight_max"] for r in q12.values()) - s["q1_tight_max"]) == 0
    assert abs(max(r["q1_default_max"] for r in q12.values()) - s["q1_default_max"]) == 0
    assert sum(sum(r["q2_gm_curves_introduced"]) for r in q12.values()) == s["q2_gm_sign"]
    assert sum(sum(r["q2_gds_curves_introduced"]) for r in q12.values()) == s["q2_gds_sign"]
    return d


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def col(key, head, md=None, kind="rtext", group=None):
    return {"key": key, "head": head, "md": md or head, "kind": kind, "group": group}


def t_decision(d: dict) -> rf.Table:
    s = d["summary"]["summary_values"]
    dec = d["decision"]
    T = m.THR
    q3 = d["q3"]["rows"]
    rows = [
        {"q": "Q1", "check": "tight max rel. diff., |I| > 1 nA (core)", "value": e(s["q1_tight_max"]),
         "thr": f"<= {T['q1_tight']:.0e}", "res": "fail" if s["q1_tight_max"] > T["q1_tight"] else "pass"},
        {"q": "Q1", "check": "convergence msgs + non-finite, element runs (core)", "value": str(s["q1_bad"]),
         "thr": "0", "res": "pass" if s["q1_bad"] == 0 else "fail"},
        {"q": "Q1", "check": "max |I_elem(Vds = 0)| (core)", "value": f"{s['q1_vds0_max']:.2e} A",
         "thr": "<= 1e-12 A", "res": "pass" if s["q1_vds0_max"] <= T["q1_vds0_abs"] else "fail"},
        {"q": "Q2", "check": "AC derivative error gm, gds, Im Y (core)", "value": e(s["q2_ac_max"]),
         "thr": f"<= {T['q2_ac']:.0e}", "res": "fail" if s["q2_ac_max"] > T["q2_ac"] else "pass"},
        {"q": "Q2", "check": "gm curves with introduced sign change (core)", "value": str(s["q2_gm_sign"]),
         "thr": "0 of 54", "res": "fail" if s["q2_gm_sign"] else "pass"},
        {"q": "Q3", "check": "circuits failing only with the element (core)", "value": str(s["q3_fail"]),
         "thr": "0 of 6", "res": "fail" if s["q3_fail"] else "pass"},
        {"q": "Q1", "check": "default-tolerance max rel. diff. (restriction)", "value": e(s["q1_default_max"]),
         "thr": "<= 1e-03", "res": "restriction" if s["q1_default_max"] > T["q1_default"] else "none"},
        {"q": "Q1", "check": "terminal-exchange asymmetry of the element (restriction)",
         "value": f"{s['swap_asym_max']:.3f}", "thr": "<= 1e-03",
         "res": "restriction" if s["swap_asym_max"] > T["swap_asym"] else "none"},
        {"q": "Q2", "check": "drain displacement current changed (restriction)",
         "value": "yes" if s["cap_scaled"] else "no", "thr": "no", "res": "restriction" if s["cap_scaled"] else "none"},
        {"q": "Q2", "check": "gds curves with introduced sign change (restriction)", "value": str(s["q2_gds_sign"]),
         "thr": "0 of 90", "res": "restriction" if s["q2_gds_sign"] else "none"},
        {"q": "Q4", "check": "max wall-time ratio element / card (restriction)",
         "value": f"{s['runtime_ratio_max']:.2f}", "thr": "<= 2", "res":
         "restriction" if s["runtime_ratio_max"] > T["runtime_ratio"] else "none"},
        {"q": "Q3", "check": "card-only circuit runs not ok (reported)", "value": str(s["q3_fail_cardonly"]),
         "thr": "—", "res": ", ".join(k for k in d["summary"]["q3_not_ok"] if not k.endswith("M_full")) or "—"},
    ]
    notes = [f"Outcome (pre-registered rule, prereg commit {COMMITS['prereg']}): {dec['outcome']}.",
             "Core checks failed: " + ("; ".join(dec["core_fail"]) or "none") + ".",
             "Restrictions: " + ("; ".join(dec["restrictions"]) or "none") + ".",
             "18 devices, M_full card (43-parameter card refit with the full-data g in the loop) + full-data g; "
             "tight = reltol 1e-9, abstol 1e-18, vntol 1e-12; no measured circuit data (Q3 is a simulator-level "
             "check)."]
    return rf.Table("01_table_i11_1_decision", "I11-1 pre-registered checks and outcome",
                    [col("q", "Q", kind="text"), col("check", "check", kind="text"), col("value", "value"),
                     col("thr", "threshold"), col("res", "result", kind="text")], rows, notes)


def t_devices(d: dict) -> rf.Table:
    q12 = d["q12"]["devices"]
    nat = d["refit"]["native"]
    rows = []
    for t, r in q12.items():
        rows.append({"dev": t, "q1t": e(r["q1_tight_max"]), "q1r4": e(r["q1_r4_max"]),
                     "q1d": e(r["q1_default_max"]), "q1u": e(r["q1_default_max_above_1uA"]),
                     "v0": e(r["q1_vds0_max_corr"]), "sw": f"{r['swap_asym_element_max']:.3f}",
                     "gm": e(max(r["q2_err_gm"], r["q2_err_im_dg"])),
                     "gds": e(max(r["q2_err_gds"], r["q2_err_im_dd"])),
                     "sgm": str(sum(r["q2_gm_curves_introduced"])), "sgds": str(sum(r["q2_gds_curves_introduced"])),
                     "cap": f"{r['cap_dd_ratio_min']:.3f}-{r['cap_dd_ratio_max']:.3f}",
                     "capg": e(r["cap_dd_ratio_over_expg_maxdev"]),
                     "g": f"{r['g_grid_min']:+.3f}/{r['g_grid_max']:+.3f}",
                     "rr": f"{nat[t]['rrms_native']:.4f}"})
    notes = ["q1: max |I_elem - I_py| / |I_py| on the 75 x 149 grid, |I_py| > 1 nA; columns tight / reltol 1e-4 / "
             "default, and default at |I| >= 1 uA. Vds0: max |I_elem| at Vds = 0 (default and tight).",
             "swap: max |ln(I_swap/I_fwd)_elem - ln(I_swap/I_fwd)_card|, forward operation (terminal exchange).",
             "AC err: max normalized difference of the OSDI AC admittance vs exp(g) Y_card + I exp(g) dg/dV "
             "(real and imaginary parts); gm / gds sign: curves (of 3 / 5) with an introduced negative value.",
             "Cdd ratio: Im Y_dd(elem) / Im Y_dd(card) range; /exp(g): max |ratio / exp(g) - 1|.",
             "g range on the grid (incl. Vds < 0); RRMS: in-sample M_full SPICE-native RRMS on the measured curves."]
    return rf.Table("05_table_i11_1_per_device", "Per-device Q1 / Q2 values (M_full card + full-data g)",
                    [col("dev", "device", kind="text"), col("q1t", "tight", group="Q1 max rel. diff."),
                     col("q1r4", "1e-4", group="Q1 max rel. diff."), col("q1d", "default", group="Q1 max rel. diff."),
                     col("q1u", ">=1uA", group="Q1 max rel. diff."), col("v0", "Vds0 A"), col("sw", "swap"),
                     col("gm", "gm/Ydg", group="Q2 AC err"), col("gds", "gds/Ydd", group="Q2 AC err"),
                     col("sgm", "gm", group="sign curves"), col("sgds", "gds", group="sign curves"),
                     col("cap", "Cdd ratio"), col("capg", "/exp(g)"), col("g", "g min/max"),
                     col("rr", "RRMS")], rows, notes)


def t_circuits(d: dict) -> rf.Table:
    q3 = d["q3"]["rows"]
    rows = []
    for name, info in d["q3"]["circuits"].items():
        for arm in ("published", "C0_refit", "M_card", "M_full"):
            r = q3[f"{name}|{arm}"]
            if info["kind"] == "vtc":
                v = f"V_M {r['vm']:.4f} V, max gain {r['gain_max']:.1f}" if r["finite"] else "—"
            elif info["kind"] == "ring":
                v = (f"{r['freq_hz'] / 1e9:.4f} GHz, {r['n_periods']} periods"
                     if r.get("n_periods") else f"stopped ({r['n_points']} points)")
            else:
                v = f"ratio {r['ratio_at_0p9']:.4f} @0.9 V, {r['ratio_at_vref']:.5f} @V(ref) {r['v_ref']:.3f} V"
            rows.append({"c": name, "dev": " + ".join(info["devices"]), "arm": arm, "v": v,
                         "ok": "yes" if r["ok"] else "no",
                         "msg": (r["msgs"][0][:60] if r["msgs"] else "—"),
                         "w": f"{np.median(r['wall']):.3f}"})
    notes = ["VDD 1.8 V, 77 K, default tolerances, default sky130 capacitances. Arms: published card; C0_refit card "
             "(stored all18 run); M_card = M_full card without g; M_full = M_full card + g (OSDI element).",
             "No measured circuit data: these are simulator-level consistency and convergence checks, not accuracy "
             "claims. ok = no convergence message, finite output, ring >= 5 periods."]
    return rf.Table("08_table_i11_1_circuits", "Q3 circuits: outputs per arm",
                    [col("c", "circuit", kind="text"), col("dev", "devices", kind="text"),
                     col("arm", "arm", kind="text"), col("v", "output", kind="text"), col("ok", "ok"),
                     col("msg", "first message", kind="text"), col("w", "wall s")], rows, notes)


def t_diagnosis(d: dict) -> rf.Table:
    dg = d["diagnosis"]
    rows = []
    for r in dg["gds_slopes"]:
        rows.append({"dev": r["tag"], "vg": f"{r['abs_vgs']:.2f}", "card": e(r["card_S"]), "el": e(r["elem_S"]),
                     "meas": "—" if not np.isfinite(r["measured_S"]) else e(r["measured_S"]),
                     "rng": "—" if r["abs_vds_neg_min"] is None else
                     f"{r['abs_vds_neg_min']:.3f}-{r['abs_vds_neg_max']:.3f}",
                     "gmin": "—" if r["gds_min_S"] is None else e(r["gds_min_S"])})
    for r in dg["gm_sign"]:
        rows.append({"dev": r["tag"] + " (gm)", "vg": f"|Vds| {r['abs_vds']:.2f}", "card": "—", "el": "—",
                     "meas": "—", "rng": f"|Vgs| {r['abs_vgs_neg_min']:.3f}-{r['abs_vgs_neg_max']:.3f}",
                     "gmin": e(r["gm_min_S"])})
    q1 = {t: v for t, v in dg["q1_tight"].items() if v["n_over"]}
    notes = ["Post hoc (after the pre-registered decision; does not change it).",
             "gds rows: curves with an introduced negative gds; average output conductance = slope of a linear fit "
             "of |I_D| over |Vds| 1.0-1.85 V for the card, the element and the measured curve at the nearest "
             "measured |Vgs| (within 0.04 V); range = |Vds| interval with gds < -1e-12 S on the 1 mV sweep.",
             "gm row: the gm curve with an introduced sign change (|Vgs| interval and minimum gm).",
             f"Q1 tight points above 1e-5: " + "; ".join(
                 f"{t} {v['n_over']} points, |I| <= {v['max_abs_I_over_A']:.1e} A, |dI| <= "
                 f"{v['max_abs_diff_over_A']:.1e} A" for t, v in q1.items())
             + f"; max rel. diff. at |I| > 10 nA {max(v['max_rel_above_10nA'] for v in dg['q1_tight'].values()):.1e}"
             " (gmin 1e-15 and nobypass do not remove the floor).",
             f"Q2: the only AC point above 1e-3 lies on the line Vgs = 0 (abs() kink; OpenVAF takes a one-sided "
             f"derivative, the Python reference a central difference); max off the lines Vgs = 0 and Vds = 0 "
             f"{dg['q2_off_kink_max']:.1e}."]
    return rf.Table("10_table_i11_1_diagnosis", "Post hoc diagnosis of the failed core checks and the gds restriction",
                    [col("dev", "device", kind="text"), col("vg", "|Vgs| V"), col("card", "card S"),
                     col("el", "element S"), col("meas", "measured S"), col("rng", "negative range V"),
                     col("gmin", "min gm/gds S")], rows, notes)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_q1_devices(d: dict, path: Path) -> list[str]:
    q12 = d["q12"]["devices"]
    tags = list(q12)
    fig, ax = plt.subplots(figsize=(11, 4.4), facecolor=BG)
    x = np.arange(len(tags))
    for key, c, mk, lab in (("q1_default_max", RED, "o", "default (reltol 1e-3)"),
                            ("q1_r4_max", ORANGE, "s", "reltol 1e-4"),
                            ("q1_tight_max", BLUE, "D", "tight (reltol 1e-9, abstol 1e-18, vntol 1e-12)"),
                            ("q1_default_max_above_1uA", VIOLET, "^", "default, |I| >= 1 uA")):
        ax.scatter(x, [max(q12[t][key], 1e-12) for t in tags], color=c, marker=mk, s=26, label=lab, zorder=3)
    ax.axhline(1e-3, color=RED, lw=0.8, ls="--")
    ax.axhline(1e-5, color=BLUE, lw=0.8, ls="--")
    ax.text(len(tags) - 0.5, 1.15e-3, "default threshold 1e-3", ha="right", fontsize=7, color=RED)
    ax.text(len(tags) - 0.5, 1.15e-5, "tight threshold 1e-5", ha="right", fontsize=7, color=BLUE)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("max |I_OSDI - I_card exp(g)| / |I|  (|I| > 1 nA)", fontsize=8)
    ax.set_title("Q1: OSDI element vs Python model on the 75 x 149 bias grid (Vds -1.85 to 1.85 V), per device",
                 fontsize=9, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return [rel(F["q12"])]


def fig_q1_maps(d: dict, path: Path) -> list[str]:
    tags = ("nmos_L0p19_W7", "pmos_L0p35_W5")
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.6), facecolor=BG)
    for i, tag in enumerate(tags):
        a = arrays(tag)
        dev = d["devs"][tag]
        pm = dev.dev_type == "pmos"
        s = -1.0 if pm else 1.0
        VG, VD = np.meshgrid(s * m.GRID_VGS, m.GRID_VDS, indexing="ij")
        g = m.g_values(d["corr"], pm, VG, VD)
        for j, tol in enumerate(("default", "tight")):
            ref = m.python_current(a[f"grid_fwd_{tol}_card"], g)
            r = m.rel_diff(a[f"grid_fwd_{tol}_corr"], ref, m.I_FLOOR)
            ax = axs[i, j]
            with np.errstate(divide="ignore", invalid="ignore"):
                lr = np.where(np.isnan(r), np.nan, np.log10(np.maximum(r, 1e-16)))
            im = ax.pcolormesh(m.GRID_VDS * s, m.GRID_VGS, lr,
                               vmin=-12, vmax=-2, cmap="viridis", shading="nearest")
            ax.set_title(f"{tag}, {tol}: log10 rel. diff. (white: |I| <= 1 nA)", fontsize=8, color=INK)
            ax.set_xlabel("signed forward Vds (V), negative = reverse", fontsize=8)
            ax.set_ylabel("|Vgs| (V)", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)
            style(ax)
        ax = axs[i, 2]
        im = ax.pcolormesh(m.GRID_VDS * s, m.GRID_VGS, a["grid_fwd_tight_corr"] * 0 + g, cmap="RdBu_r",
                           vmin=-0.4, vmax=0.4, shading="nearest")
        pts = d["mpts"][dev.dev_type]
        ax.scatter(pts[:, 1], pts[:, 0], s=0.5, color=INK, alpha=0.25)
        ax.axvline(0, color=MUTED, lw=0.6)
        ax.set_title(f"g on the grid ({'pMOS' if pm else 'nMOS'}); dots: measured bias points (all devices)",
                     fontsize=8, color=INK)
        ax.set_xlabel("signed forward Vds (V)", fontsize=8)
        ax.set_ylabel("|Vgs| (V)", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
        style(ax)
    fig.suptitle("Q1: where the OSDI and Python currents differ; g is mirrored onto Vds < 0 by abs()", fontsize=10,
                 color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return [rel(RUN / "arrays"), rel(CORR), "out/research/explore/E1/real_{dev,test}"]


def fig_q1_swap(d: dict, path: Path) -> list[str]:
    tags = ("nmos_L0p15_W1p6", "pmos_L2_W5")
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=BG)
    for i, tag in enumerate(tags):
        a = arrays(tag)
        dev = d["devs"][tag]
        s = -1.0 if dev.dev_type == "pmos" else 1.0
        ax = axs[i]
        jv = [k for k, v in enumerate(m.GRID_VGS) if abs(v - 1.2) < 1e-9 or abs(v - 1.85) < 1e-9]
        fw = (s * m.GRID_VDS) >= 0
        for k, ls in zip(jv, ("-", "--")):
            x = np.abs(m.GRID_VDS[fw])
            ax.plot(x, np.abs(a["grid_fwd_tight_card"][k, fw]) * 1e3, color=GREY, ls=ls, lw=1.2,
                    label=f"card, |Vgs| {m.GRID_VGS[k]:.2f}")
            ax.plot(x, np.abs(a["grid_fwd_tight_corr"][k, fw]) * 1e3, color=RED, ls=ls, lw=1.2,
                    label=f"element, normal terminals, |Vgs| {m.GRID_VGS[k]:.2f}")
            ax.plot(x, np.abs(a["grid_swap_tight_corr"][k, fw]) * 1e3, color=BLUE, ls=ls, lw=1.2,
                    label=f"element, drain/source exchanged, |Vgs| {m.GRID_VGS[k]:.2f}")
        ax.set_xlabel("|Vds| (V)", fontsize=8)
        ax.set_ylabel("|I_D| (mA)", fontsize=8)
        ax.set_title(f"{tag}: terminal exchange (card current is the same both ways)", fontsize=8, color=INK)
        ax.legend(fontsize=6, frameon=False)
        style(ax)
    ax = axs[2]
    q12 = d["q12"]["devices"]
    tg = list(q12)
    ax.bar(np.arange(len(tg)), [q12[t]["swap_asym_element_max"] for t in tg], color=VIOLET)
    ax.scatter(np.arange(len(tg)), [q12[t]["swap_asym_element_pred_max"] for t in tg], color=INK, s=10,
               zorder=3, label="predicted max |g(Vgs - Vds, Vds) - g(Vgs, Vds)|")
    ax.set_xticks(np.arange(len(tg)))
    ax.set_xticklabels(tg, rotation=70, ha="right", fontsize=6)
    ax.set_ylabel("max |ln(I_swap/I_fwd)_elem - ln(...)_card|", fontsize=8)
    ax.set_title("element asymmetry per device (forward, |I| > 1 nA)", fontsize=8, color=INK)
    ax.set_ylim(0, 0.16)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return [rel(F["q12"]), rel(RUN / "arrays")]


def fig_q2_curves(d: dict, path: Path) -> list[str]:
    tags = ("nmos_L0p15_W1p6", "nmos_L100_W100", "pmos_L2_W5", "pmos_L8_W5")
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.4), facecolor=BG)
    for j, tag in enumerate(tags):
        a = arrays(tag)
        ax = axs[0, j]
        for k, vds in enumerate(m.GM_VDS):
            for lab, c, ls in (("card", GREY, "-"), ("corr", RED, "--")):
                dd = a[f"dense_{lab}_gm{k}"]
                o = np.argsort(np.abs(dd[:, 0]))
                x, y = np.abs(dd[o, 0]), dd[o, 1]
                gm = m.central_diff(dd[o, 0], y)
                keep = np.abs(y) > 1e-10
                x, y, gm = x[keep], y[keep], gm[keep]
                with np.errstate(divide="ignore", invalid="ignore"):
                    ax.plot(x, gm / np.abs(y), color=c, ls=ls, lw=1.0 if k else 1.4,
                            alpha=1.0 - 0.3 * k, label=f"{'card' if lab == 'card' else 'element'} |Vds| {vds}")
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_xlabel("|Vgs| (V)", fontsize=8)
        ax.set_ylabel("gm / |I_D| (1/V), 1 mV sweep, |I_D| > 0.1 nA", fontsize=8)
        ax.set_title(f"{tag}: gm/|I_D|", fontsize=8, color=INK)
        ax.axvline(d["q12"]["devices"][tag]["vt_card"] + 0.2, color=MUTED, lw=0.6, ls=":")
        if j == 0:
            ax.legend(fontsize=5.5, frameon=False)
        style(ax)
        ax = axs[1, j]
        for k, vgs in enumerate(m.GDS_VGS):
            if vgs < 0.9:
                continue
            for lab, c, ls in (("card", GREY, "-"), ("corr", RED, "--")):
                dd = a[f"dense_{lab}_gds{k}"]
                o = np.argsort(np.abs(dd[:, 0]))
                x, y = np.abs(dd[o, 0]), dd[o, 1]
                gds = m.central_diff(dd[o, 0], y)
                keep = x >= 0.05
                x, y, gds = x[keep], y[keep], gds[keep]
                with np.errstate(divide="ignore", invalid="ignore"):
                    ax.plot(x, gds / np.abs(y), color=c, ls=ls, lw=1.0, alpha=0.4 + 0.15 * k,
                            label=f"{'card' if lab == 'card' else 'element'} |Vgs| {vgs}")
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_ylim(-3, 30)
        ax.set_xlabel("|Vds| (V), >= 0.05 V", fontsize=8)
        ax.set_ylabel("gds / |I_D| (1/V), 1 mV sweep", fontsize=8)
        ax.set_title(f"{tag}: gds/|I_D| (|Vgs| 0.9-1.85 V)", fontsize=8, color=INK)
        if j == 0:
            ax.legend(fontsize=5.5, frameon=False)
        style(ax)
    fig.suptitle("Q2: normalized transconductance and output conductance, card only (grey) vs card + element "
                 "(red, OSDI, tight tolerance); dotted line: V_T + 0.2 V", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return [rel(RUN / "arrays"), rel(F["q12"])]


def fig_q2_cap(d: dict, path: Path) -> list[str]:
    q12 = d["q12"]["devices"]
    tg = list(q12)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.0), facecolor=BG)
    ax = axs[0]
    x = np.arange(len(tg))
    for key, c, mk, lab in (("q2_err_gm", BLUE, "o", "gm"), ("q2_err_gds", ORANGE, "s", "gds"),
                            ("q2_err_im_dg", VIOLET, "^", "Im Y_dg"), ("q2_err_im_dd", AQUA, "v", "Im Y_dd"),
                            ("q2_fd_err_osdi", GREY, "x", "1 mV FD vs AC (descriptive)")):
        ax.scatter(x, [max(q12[t][key], 1e-13) for t in tg], color=c, marker=mk, s=22, label=lab, zorder=3)
    ax.axhline(1e-3, color=RED, lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(tg, rotation=70, ha="right", fontsize=6)
    ax.set_ylabel("max normalized AC difference vs Python", fontsize=8)
    ax.set_title("Q2: OSDI AC admittance vs exp(g) Y_card + I exp(g) dg/dV (tight)", fontsize=8, color=INK)
    ax.legend(fontsize=6, frameon=False)
    style(ax)
    ax = axs[1]
    tag = "pmos_L2_W5"
    a = arrays(tag)
    pts = a["pts"]
    g = m.g_values(d["corr"], True, pts[:, 0], pts[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        rdd = a["ac_VD_corr_ym"] / a["ac_VD_card_ym"]
        rgg = a["ac_VG_corr_yg"] / a["ac_VG_card_yg"]
    ax.scatter(np.exp(g), rdd, s=6, color=RED, label="Im Y_dd(element) / Im Y_dd(card)")
    ax.scatter(np.exp(g), rgg, s=6, color=BLUE, label="Im Y_gg(element) / Im Y_gg(card)")
    lo, hi = np.exp(g).min(), np.exp(g).max()
    ax.plot([lo, hi], [lo, hi], color=INK, lw=0.6, ls=":")
    ax.set_xlabel("exp(g) at the op point", fontsize=8)
    ax.set_ylabel("capacitive admittance ratio at 1 MHz", fontsize=8)
    ax.set_title(f"{tag}: drain displacement current scales with exp(g), gate unchanged", fontsize=8, color=INK)
    ax.legend(fontsize=7, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return [rel(F["q12"]), rel(RUN / "arrays")]


def fig_circuits(d: dict, path: Path) -> list[str]:
    circ = d["q3"]["circuits"]
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.6), facecolor=BG)
    order = ("inv_A_vtc", "inv_B_vtc", "mirror_n", "ro5_A", "ro5_B", "mirror_p")
    for ax, name in zip(axs.ravel(), order):
        info = circ[name]
        for arm in ("published", "C0_refit", "M_card", "M_full"):
            c, ls, lab = ARM_STYLE[arm]
            with np.load(RUN / "arrays" / f"q3_{name}_{arm}.npz") as z:
                rows = z["rows"]
            r = d["q3"]["rows"][f"{name}|{arm}"]
            if rows.size == 0:
                continue
            if info["kind"] == "ring":
                t = rows[:, 0]
                tstop = info["settings"]["tstop"]
                k = t >= tstop - (2.5e-9 if tstop < 2e-8 else 20e-9)
                if not r["ok"]:
                    k = np.ones_like(t, bool)
                lab2 = lab + (f", {r['freq_hz'] / 1e9:.3f} GHz" if r.get("n_periods") else ", stopped")
                ax.plot(t[k] * 1e9, rows[k, 1], color=c, ls=ls, lw=1.0, label=lab2)
                ax.set_xlabel("t (ns)", fontsize=8)
                ax.set_ylabel("V(o1) (V)", fontsize=8)
            elif info["kind"] == "vtc":
                ax.plot(rows[:, 0], rows[:, 1], color=c, ls=ls, lw=1.1, label=f"{lab}, V_M {r['vm']:.3f} V")
                ax.set_xlabel("Vin (V)", fontsize=8)
                ax.set_ylabel("Vout (V)", fontsize=8)
            else:
                ax.plot(rows[:, 0], rows[:, 1] / info["settings"]["iref"], color=c, ls=ls, lw=1.1,
                        label=f"{lab}, {r['ratio_at_0p9']:.3f} at 0.9 V")
                ax.set_xlabel("V(out) (V)", fontsize=8)
                ax.set_ylabel("Iout / Iref", fontsize=8)
                ax.set_ylim(0.8, 1.2)
        ax.set_title(f"{name}: " + " + ".join(info["devices"]), fontsize=8, color=INK)
        ax.legend(fontsize=6, frameon=True, framealpha=0.9, loc="lower right" if info["kind"] == "ring" else "best")
        style(ax)
    fig.suptitle("Q3: circuits at 77 K, VDD 1.8 V (simulator-level; no measured circuit data)", fontsize=10,
                 color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return [rel(F["q3"]), rel(RUN / "arrays")]


def fig_runtime(d: dict, path: Path) -> list[str]:
    s = d["summary"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), facecolor=BG, gridspec_kw={"width_ratios": [3, 1.3]})
    ax = axs[0]
    g = s["runtime_grid"]
    tg = list(g)
    x = np.arange(len(tg))
    ax.bar(x - 0.2, [g[t]["card_s"] for t in tg], 0.4, color=GREY, label="card only")
    ax.bar(x + 0.2, [g[t]["corr_s"] for t in tg], 0.4, color=RED, label="card + element")
    for i, t in enumerate(tg):
        ax.text(i, g[t]["corr_s"] * 1.02, f"{g[t]['ratio']:.2f}", ha="center", fontsize=6, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(tg, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("ngspice wall time (s), median of 3", fontsize=8)
    ax.set_title("Q4: 11,175-point DC grid deck, default tolerance (ratio above bars)", fontsize=8, color=INK)
    ax.legend(fontsize=7, frameon=False)
    style(ax)
    ax = axs[1]
    c = s["runtime_circuits"]
    names = list(c)
    ax.barh(np.arange(len(names)), [c[n]["ratio"] for n in names], color=VIOLET)
    ax.axvline(1, color=INK, lw=0.6)
    ax.axvline(2, color=RED, lw=0.8, ls="--")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("wall-time ratio M_full / M_card", fontsize=8)
    ax.set_title("circuits", fontsize=8, color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return [rel(F["summary"])]


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    folder = rf.REPORT_ROOT / f"{args.date}_{TOPIC}"
    folder.mkdir(parents=True, exist_ok=True)
    d = load_all()
    d["mpts"] = measured_points()
    items, tables = [], []

    def add_table(t, desc, srcs):
        rf.render_table(t, folder / f"{t.name}.png")
        tables.append(t)
        items.append((f"{t.name}.png", desc, srcs))

    def add_fig(name, fn, desc):
        items.append((name, desc, fn(d, folder / name)))

    add_table(t_decision(d), "Table of the pre-registered checks (core and restriction), values, thresholds and "
              "the outcome", [rel(PREREG), rel(F["summary"]), rel(F["decision"])])
    add_fig("02_i11_1_q1_per_device.png", fig_q1_devices, "Q1 per device: maximum relative difference between the "
            "OSDI element current and the Python model (card current x exp(g)) on the dense grid at |I| > 1 nA, at "
            "default, 1e-4 and tight tolerance, and default at |I| >= 1 uA; thresholds dashed")
    add_fig("03_i11_1_q1_maps.png", fig_q1_maps, "Q1 maps for the two devices with the largest tight difference: "
            "log10 relative difference over (signed forward Vds, |Vgs|) at default and tight tolerance, and g on "
            "the grid with the measured bias points of all devices (g is mirrored onto Vds < 0)")
    add_fig("04_i11_1_q1_terminal_swap.png", fig_q1_swap, "Q1 terminal exchange: forward I-V with normal and "
            "exchanged drain/source terminals for two devices, and the element asymmetry per device with the "
            "value predicted from g")
    add_table(t_devices(d), "Table per device: Q1 differences by tolerance, I at Vds = 0, terminal asymmetry, Q2 AC "
              "errors, introduced sign-change curves, drain capacitive ratio, g range, in-sample SPICE-native RRMS",
              [rel(F["q12"]), rel(F["refit"])])
    add_fig("06_i11_1_q2_gm_gds.png", fig_q2_curves, "Q2 gm/|I_D| (top) and gds/|I_D| (bottom) from 1 mV sweeps at "
            "tight tolerance, card only vs card + element, for nmos_L0p15_W1p6, nmos_L100_W100 (gm sign change at "
            "|Vds| 0.05 V), pmos_L2_W5 and pmos_L8_W5 (negative gds)")
    add_fig("07_i11_1_q2_ac_cap.png", fig_q2_cap, "Q2 AC check per device (normalized difference of gm, gds, Im Y_dg, "
            "Im Y_dd vs the Python model; 1 mV finite differences descriptive) and the capacitive admittance ratios "
            "vs exp(g) on pmos_L2_W5")
    add_table(t_circuits(d), "Table of the Q3 circuits: outputs, convergence and wall time per arm",
              [rel(F["q3"])])
    add_fig("09_i11_1_q3_circuits.png", fig_circuits, "Q3 circuits: inverter VTCs, ring-oscillator waveforms (last "
            "periods) and mirror output-current ratios for the four arms")
    add_table(t_diagnosis(d), "Table of the post hoc diagnosis: gds curves with introduced negative values "
              "(average output conductance of card, element and measured curve), the gm sign-change curve, the "
              "location of the Q1 tight and Q2 failures", [rel(F["diagnosis"])])
    add_fig("11_i11_1_q4_runtime.png", fig_runtime, "Q4 ngspice wall time with and without the element: DC grid per "
            "device and the six circuits (ratio)")

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i11_1.py --date {args.date}`"
    md = [f"# Tables, {args.date} {TOPIC}", "", f"Generated {now} by {gen} from stored files only.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))

    s = d["summary"]["summary_values"]
    dec = d["decision"]
    ref = d["summary"]["refit_in_sample"]
    tm = d["timing"]
    q3 = d["q3"]["rows"]
    dg = d["diagnosis"]
    ro = {n: q3[f"ro5_A|{n}"]["freq_hz"] / 1e9 for n in ("published", "C0_refit", "M_card", "M_full")}
    vm = {n: q3[f"inv_A_vtc|{n}"]["vm"] for n in ("published", "C0_refit", "M_card", "M_full")}
    mr = {n: q3[f"mirror_p|{n}"]["ratio_at_0p9"] for n in ("published", "C0_refit", "M_card", "M_full")}
    readme = [
        f"# Report batch {args.date} · {TOPIC}", "",
        f"Generated {now} by {gen} from stored files only (no NGSpice, no fit, no training).", "",
        "## Contents", "",
        "- I11-1: the 43-parameter card + the shared bias-only correction (M_bias, I3-2 recipe, full-data model "
        "trained on all 18 sky130 77 K devices) as a compact-model add-on: the stored Verilog-A "
        "(`out/research/explore/I3_2/all18/m_bias_sky130_all18_full.va`) compiled with OpenVAF 23.5.0 and run as an "
        "OSDI element around the BSIM4 instance in the ngspice-41 OSDI build. Pre-registration "
        f"`{rel(I11 / 'prereg.md')}` (in force from commit {COMMITS['prereg']}); outputs `{rel(RUN)}`; code "
        "`src/cryoml/explore/i11_1_deploy.py`, `scripts/research/explore/i11_1_deploy.py`. Run "
        f"{sum(v for k, v in tm.items() if k != 'diagnose'):.0f} s with 4 workers (refit {d['refit']['refit_s']} s); "
        "a queue job of I11-2 (6 workers) ran at the same time.",
        f"- Card: M_full (per device, the 43-parameter card refit from direct43_r3_fd with the full-data g in the "
        f"loop). In-sample all-device RRMS on the measured curves: SPICE-native {ref['mean_rrms_M_full_native']:.4f}, "
        f"Python {ref['mean_rrms_M_full_python']:.4f}, the same card without g {ref['mean_rrms_M_card_only']:.4f}; "
        f"0 failed refits. In-sample values, not comparable with the LODO M_bias 0.0732.",
        f"- **Outcome (pre-registered rule): {dec['outcome']}.** Core checks failed: {'; '.join(dec['core_fail'])}. "
        f"Restrictions: {'; '.join(dec['restrictions'])}.",
        f"- Q1: OSDI vs Python max relative difference (|I| > 1 nA, 18 devices, 75 x 149 grid incl. Vds < 0): "
        f"tight {s['q1_tight_max']:.2e} (threshold 1e-5; 4 devices above, at 1.0-2.6 nA with absolute differences "
        f"<= 2.8e-14 A; max at |I| > 10 nA "
        f"{max(v['max_rel_above_10nA'] for v in dg['q1_tight'].values()):.1e}), reltol 1e-4 "
        f"{max(r['q1_r4_max'] for r in d['q12']['devices'].values()):.2e}, default {s['q1_default_max']:.2e}. "
        f"0 convergence messages, 0 non-finite values. max |I(Vds = 0)| {s['q1_vds0_max']:.1e} A. g on the grid "
        f"{min(r['g_grid_min'] for r in d['q12']['devices'].values()):+.3f} to "
        f"{max(r['g_grid_max'] for r in d['q12']['devices'].values()):+.3f}; g is bounded (|g| < 1) and mirrored "
        "onto Vds < 0 by abs(). Terminal exchange: the element evaluates g with the instance source terminal, so "
        f"with exchanged drain/source the current changes by up to {s['swap_asym_max']:.3f} in ln I (nMOS 0.043, "
        "pMOS 0.117, equal to the value predicted from g); the card alone is symmetric to "
        f"{max(r['swap_asym_card_max'] for r in d['q12']['devices'].values()):.0e}.",
        f"- Q2: OSDI AC vs Python small signal max {s['q2_ac_max']:.2e} (threshold 1e-3); the only point above "
        f"1e-3 is nmos_L100_W100 at Vgs = 0 (abs() kink, one-sided derivative in OSDI vs central difference); off "
        f"the lines Vgs = 0 and Vds = 0 the max is {dg['q2_off_kink_max']:.1e}. Introduced sign changes: gm "
        f"{s['q2_gm_sign']} curve of 54 (nmos_L100_W100, |Vds| 0.05 V, |Vgs| 1.757-1.85 V); gds {s['q2_gds_sign']} "
        "curves of 90 (" + ", ".join(f"{t} {sum(r['q2_gds_curves_introduced'])}" for t, r in d["q12"]["devices"].items()
                               if any(r["q2_gds_curves_introduced"])) + "). Capacitances: Im Y_dd and Im Y_dg of the element "
        f"equal exp(g) times the card value (max deviation {max(max(r['cap_dd_ratio_over_expg_maxdev'], r['cap_dg_ratio_over_expg_maxdev']) for r in d['q12']['devices'].values()):.0e}); Im Y_gg unchanged (max deviation {max(r['cap_gate_ratio_maxdev'] for r in d['q12']['devices'].values()):.0e}). The element scales the whole "
        "drain-terminal current, including displacement and junction currents.",
        f"- Q3 (simulator-level, no measured circuit data): all element runs converge where the card-only run "
        f"converges; ro5_B (nmos_L1_W1p6 + pmos_L2_W5) stops with 'timestep too small' for C0_refit, M_card and "
        f"M_full (card-level; the published card runs). inv_A V_M published / C0_refit / M_card / M_full "
        + " / ".join(f"{vm[k]:.4f}" for k in vm) + " V; ro5_A frequency "
        + " / ".join(f"{ro[k]:.4f}" for k in ro) + " GHz; mirror_p Iout/Iref at 0.9 V "
        + " / ".join(f"{mr[k]:.4f}" for k in mr) + ".",
        f"- Q4: wall-time ratio element / card: DC grid {min(v['ratio'] for v in d['summary']['runtime_grid'].values()):.2f}-"
        f"{max(v['ratio'] for v in d['summary']['runtime_grid'].values()):.2f}, circuits "
        + ", ".join(f"{k} {v['ratio']:.2f}" for k, v in d["summary"]["runtime_circuits"].items())
        + " (measured while another job used 6 cores).",
        "- Post hoc (not pre-registered): table 10 and the 'off kink' and '> 10 nA' values.",
        "- Next actions (each needs a new pre-registration): (1) evaluate g on a source/drain-symmetric input "
        "(source = lower-potential terminal for nMOS, higher for pMOS) to remove the terminal asymmetry; (2) apply "
        "exp(g) to the channel current only (correction inside a Verilog-A BSIM4 compiled with OpenVAF) so that "
        "charges and junction currents are unchanged; (3) train g with a penalty on negative d(I exp(g))/dV (gm and "
        "gds) on a bias grid that covers the measured range, then repeat Q2; (4) replace abs(x) by sqrt(x^2 + eps) "
        "to remove the derivative jump at Vgs = 0; (5) state the equivalence check as |dI| <= 1e-6 |I| + 1e-13 A "
        "(relative plus absolute) before rerunning Q1.",
        "", "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))
    summary = [f"I11-1 M_bias OSDI add-on, sky130 77 K, 18 devices: {dec['outcome']} (pre-registered rule).",
               f"Q1 tight {s['q1_tight_max']:.1e} (1e-14 A floor), default {s['q1_default_max']:.1e}; Q2 AC "
               f"{s['q2_ac_max']:.1e} (Vgs = 0 kink), gm sign 1 curve, gds sign {s['q2_gds_sign']} curves; "
               f"terminal asymmetry {s['swap_asym_max']:.3f}; drain displacement current x exp(g).",
               f"Q3 0 element-only failures; Q4 ratio up to {s['runtime_ratio_max']:.2f}."]
    rf.update_index(args.date, TOPIC, folder, [i[0] for i in items], summary)
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
