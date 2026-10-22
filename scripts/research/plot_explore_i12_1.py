#!/usr/bin/env python3
"""Report batch for I12-1: bias-only correction g v2 (d/s-symmetric smooth
inputs, static element via a charge-free card replica, gm/gds penalty) as a
deployable ngspice-41 OSDI add-on, sky130 77 K, 18 devices.

Reads stored files:

* ``out/research/explore/I12_1/{prereg.json, decision.json}``
* ``out/research/explore/I12_1/lodo/{results, corrections, heldout_cards}.json``
* ``out/research/explore/I12_1/deploy/{summary, q12}.json``
* ``out/research/explore/I12_1/diagnosis/diagnosis.json`` (post hoc,
  ``scripts/research/explore/i12_1_diagnosis.py``)
* ``out/research/explore/I3_2/all18/{results, corrections, heldout_cards}.json``
* ``out/research/reference_scores.json`` and the upstream committed published-card sweeps

NGSpice: the I-V figures re-simulate the stored held-out cards (C0_refit,
M_bias v1, M_bias v2) on the measured bias grid (canonical ngspice-41, about
1 s per card) and apply the stored fold correction; each re-simulated RRMS is
checked against the stored LODO row.

Checks: the deployment label is recomputed from summary.json with
``i12_1_symm.decide_deploy``; A1 / A2 are recomputed with ``paired_ratio``
from the per-device rows and compared with decision.json.

  PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i12_1.py --date 2026-10-21
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from types import SimpleNamespace

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

TOPIC = "i12_1"
I12 = EXP / "I12_1"
ALL18 = EXP / "I3_2/all18"
F = {"prereg": I12 / "prereg.json", "decision": I12 / "decision.json", "lodo": I12 / "lodo/results.json",
     "corr": I12 / "lodo/corrections.json", "cards": I12 / "lodo/heldout_cards.json",
     "summary": I12 / "deploy/summary.json", "q12": I12 / "deploy/q12.json",
     "diag": I12 / "diagnosis/diagnosis.json", "v1_corr": ALL18 / "corrections.json",
     "v1_cards": ALL18 / "heldout_cards.json", "ref": ROOT / "out/research/reference_scores.json"}
COMMITS = {"prereg": "3619a2b", "code": "e7bf9b6", "A2": "28b6c14"}
ARMS = ("M_bias_v2", "M_bias_v1", "T1_inloop", "C0_refit")
ARM_LABEL = {"M_bias_v2": "M_bias v2 (I12-1)", "M_bias_v1": "M_bias v1 (I3-2 all18)",
             "T1_inloop": "T1_inloop (per-tag gains)", "C0_refit": "C0_refit (no correction)",
             "published": "published 77 K card"}
COL = {"published": GREY, "C0_refit": BLUE, "M_bias_v1": ORANGE, "M_bias_v2": AQUA, "T1_inloop": VIOLET}
MK = {"published": "s", "C0_refit": "o", "M_bias_v1": "D", "M_bias_v2": "^", "T1_inloop": "v"}
IV_DEVICES = (("pmos_L2_W5", "largest v2 gain over v1"), ("pmos_L0p35_W1p6", "largest v2 loss against v1"),
              ("nmos_L8_W1p6", "nMOS closest to the nMOS median of v2"))


def e(v: float) -> str:
    return f"{v:.2e}"


def ci(c: dict) -> str:
    return f"{c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"


def stats(per: dict) -> dict:
    v = np.array(list(per.values()), float)
    n = [x for t, x in per.items() if t.startswith("nmos")]
    p = [x for t, x in per.items() if t.startswith("pmos")]
    return {"all": float(np.mean(v)), "nmos": float(np.mean(n)), "pmos": float(np.mean(p)),
            "combined": 0.5 * (float(np.mean(n)) + float(np.mean(p))), "median": float(np.median(v))}


def load_all() -> dict:
    d = {k: load(p) for k, p in F.items() if k not in ("corr", "v1_corr", "cards", "v1_cards")}
    lodo = d["lodo"]
    per = {a: lodo["per_device"][a] for a in ARMS}
    per["published"] = {t: v["rrms"] for t, v in d["ref"]["released"]["devices"].items() if t in per["C0_refit"]}
    assert len(per["published"]) == 18 and all(len(v) == 18 for v in per.values())
    d["per"] = per
    d["stats"] = {a: stats(v) for a, v in per.items()}
    for a in ARMS:
        st = d["decision"]["summary"][a]
        assert abs(st["all_device_mean"] - d["stats"][a]["all"]) < 1e-12, a
        assert abs(st["combined"] - d["stats"][a]["combined"]) < 1e-12, a
        assert abs(st["median"] - d["stats"][a]["median"]) < 1e-12, a
    cmp = {"A1": paired_ratio(per["M_bias_v2"], per["C0_refit"]),
           "A2": paired_ratio(per["M_bias_v2"], per["M_bias_v1"]),
           "v2_T1": paired_ratio(per["M_bias_v2"], per["T1_inloop"]),
           "v1_C0": paired_ratio(per["M_bias_v1"], per["C0_refit"])}
    for k in ("A1", "A2"):
        for q in ("ratio", "ci_low", "ci_high", "wins"):
            assert abs(cmp[k][q] - d["decision"][k][q]) < 1e-12, (k, q)
    d["cmp"] = cmp
    dep = m.decide_deploy(d["summary"]["values"])
    assert dep == d["decision"]["deploy"] == d["summary"]["decision"], dep
    return d


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def col(key, head, md=None, kind="rtext", group=None):
    return {"key": key, "head": head, "md": md or head, "kind": kind, "group": group}


def t_performance(d: dict) -> rf.Table:
    S, C, dec = d["stats"], d["cmp"], d["decision"]
    vs_c0 = {"M_bias_v2": C["A1"], "M_bias_v1": C["v1_C0"]}
    rows = []
    for a in ("M_bias_v2", "M_bias_v1", "T1_inloop", "C0_refit", "published"):
        s = S[a]
        rows.append({"arm": ARM_LABEL[a], "all": f"{s['all']:.4f}", "comb": f"{s['combined']:.4f}",
                     "med": f"{s['median']:.4f}", "n": f"{s['nmos']:.4f}", "p": f"{s['pmos']:.4f}",
                     "gmr": ci(vs_c0[a]) + f", {vs_c0[a]['wins']}/18" if a in vs_c0 else "—"})
    notes = [f"LODO over 18 sky130 77 K devices; strict paper RRMS (curve inclusion frozen to the published card), "
             f"held-out card refit with the fold g in the loop (max_nfev 400, irls 2). M_bias v1, T1_inloop, C0_refit: "
             f"stored I3-2 all18 rows. published: upstream released card (reference_scores.json).",
             f"combined = equal-weight mean of the nMOS (8) and pMOS (10) family means; all-device = mean over 18.",
             f"A1 GMR(v2, C0_refit) {ci(C['A1'])}, {C['A1']['wins']}/18 lower: CI upper < 1, passes. "
             f"A2 GMR(v2, v1) {ci(C['A2'])}, {C['A2']['wins']}/18 lower: CI upper > 1.05, non-inferiority fails. "
             f"Reported: GMR(v2, T1_inloop) {ci(C['v2_T1'])}, {C['v2_T1']['wins']}/18.",
             f"Outcome (pre-registered rule, prereg commit {COMMITS['prereg']}): {dec['outcome']}; action: "
             f"{dec['action']}."]
    return rf.Table("01_table_i12_1_performance", "I12-1 LODO RRMS per arm and the accuracy contrasts",
                    [col("arm", "arm", kind="text"), col("all", "all-device"), col("comb", "combined"),
                     col("med", "median"), col("n", "nMOS mean"), col("p", "pMOS mean"),
                     col("gmr", "GMR vs C0_refit [95% CI], wins")], rows, notes)


def t_deploy(d: dict) -> rf.Table:
    s = d["summary"]["values"]
    q12 = d["q12"]["devices"]
    T = m.THR

    def over(key_fn, thr):
        return ", ".join(t for t, r in q12.items() if key_fn(r) > thr) or "—"

    rows = [
        {"c": "Q1 equivalence, tight, fwd + exchanged (core)", "v": f"{s['q1_tight_viol']} points",
         "thr": "0 points with |dI| > 1e-6|I| + 1e-13 A", "r": "pass" if s["q1_tight_viol"] == 0 else "fail",
         "dev": "—"},
        {"c": "Q1 convergence msgs + non-finite (core)", "v": str(s["q1_bad"]), "thr": "0",
         "r": "pass" if s["q1_bad"] == 0 else "fail", "dev": "—"},
        {"c": "Q1 max |I_elem(Vds = 0)| (core)", "v": f"{s['q1_vds0_max']:.2e} A", "thr": "<= 1e-12 A",
         "r": "pass" if s["q1_vds0_max"] <= T["q1_vds0_abs"] else "fail", "dev": "—"},
        {"c": "Verilog-A g text vs Python (core)", "v": e(s["g_text_max"]), "thr": "<= 1e-9",
         "r": "pass" if s["g_text_max"] <= T["g_text"] else "fail", "dev": "—"},
        {"c": "Verilog-A g exchanged vs forward (core)", "v": e(s["g_sym_text_max"]), "thr": "<= 1e-12",
         "r": "pass" if s["g_sym_text_max"] <= T["g_sym_text"] else "fail", "dev": "—"},
        {"c": "exchange asymmetry, element-introduced |d ln I| (core)", "v": e(s["swap_asym_max"]),
         "thr": "<= 1e-3", "r": "fail" if s["swap_asym_max"] > T["swap_asym"] else "pass",
         "dev": over(lambda r: r["swap_asym_element_max"], T["swap_asym"])},
        {"c": "Q2 normalized gm / gds difference (core)", "v": e(s["q2_ac_max"]), "thr": "<= 1e-3",
         "r": "fail" if s["q2_ac_max"] > T["q2_ac"] else "pass",
         "dev": over(lambda r: max(r["q2_err_gm"], r["q2_err_gds"]), T["q2_ac"])},
        {"c": "Q2 capacitance error Y_dd, Y_dg, Y_gg (core)", "v": e(s["cap_max"]), "thr": "<= 1e-3",
         "r": "fail" if s["cap_max"] > T["cap"] else "pass", "dev": over(lambda r: r["cap_max"], T["cap"])},
        {"c": "Q2 gm curves with introduced sign change (core)", "v": f"{s['q2_gm_sign']} of 54", "thr": "0",
         "r": "pass" if s["q2_gm_sign"] == 0 else "fail", "dev": "—"},
        {"c": "Q3 circuits failing only with the element (core)", "v": str(s["q3_fail"]), "thr": "0 of 6",
         "r": "pass" if s["q3_fail"] == 0 else "fail", "dev": "—"},
        {"c": "Q1 default-tolerance max rel. diff. (restriction)", "v": e(s["q1_default_max"]), "thr": "<= 1e-3",
         "r": "restriction" if s["q1_default_max"] > T["q1_default_rel"] else "none",
         "dev": f"{sum(r['q1_default_max_rel_1nA'] > 1e-3 for r in q12.values())} of 18 devices"},
        {"c": "Q2 gds curves with introduced sign change (restriction)", "v": f"{s['q2_gds_sign']} of 90",
         "thr": "0", "r": "restriction" if s["q2_gds_sign"] else "none",
         "dev": ", ".join(t for t, r in q12.items() if any(r["q2_gds_curves_introduced"])) or "—"},
        {"c": "Q4 max wall-time ratio element / card (restriction)", "v": f"{s['runtime_ratio_max']:.2f}",
         "thr": "<= 2", "r": "restriction" if s["runtime_ratio_max"] > T["runtime_ratio"] else "none",
         "dev": "ro5_A " + f"{d['summary']['runtime_circuits']['ro5_A']:.2f}; DC grid "
                f"{min(d['summary']['runtime_grid'].values()):.2f}-{max(d['summary']['runtime_grid'].values()):.2f}"},
    ]
    dec = d["decision"]["deploy"]
    notes = [f"Deployment label (pre-registered rule): {dec['deploy']}. Core fails: {'; '.join(dec['core_fail'])}. "
             f"Restrictions: {'; '.join(dec['restrictions'])}.",
             "M2 card per device (43-parameter card refit with the full-data g v2 in the loop, in-sample) + full-data "
             "g v2 as OSDI element (OpenVAF 23.5.0) sensing a charge-free replica of the card; ngspice-41 OSDI build. "
             "tight = reltol 1e-9, abstol 1e-18, vntol 1e-12. ro5_B excluded from Q4 (fails at card level, "
             "amendment A1).",
             f"Smoke (nmos_L1_W1p6, pmos_L2_W5) passed every core check; the full run adds 16 devices. "
             f"Post hoc diagnosis: table 05."]
    return rf.Table("04_table_i12_1_deploy_checks", "I12-1 part B deployment checks: values vs thresholds",
                    [col("c", "check", kind="text"), col("v", "value"), col("thr", "threshold"),
                     col("r", "result", kind="text"), col("dev", "devices over threshold", kind="text")],
                    rows, notes)


def t_diagnosis(d: dict) -> rf.Table:
    g = d["diag"]
    A, B, C, D = g["A_over_threshold"], g["B_second_analysis_drift"], g["C_q2_recomputed"], g["D_exchange_asymmetry"]
    s = d["summary"]["values"]
    cp = C["per_device"]
    l100 = cp["nmos_L100_W100"]
    f2 = g["C2_frequency_test_nmos_L100_W100"]
    iii = D["impact_ionization_test"]
    dd = D["per_device"]
    worst = max(dd, key=lambda t: dd[t]["max"])
    drift_dev = B["devices_M2_drift_gt_1e-6"]
    dev4 = drift_dev
    q2_4 = max(max(cp[t]["q2_err_gm_registered_stored"], cp[t]["q2_err_gds_registered_stored"]) for t in dev4)
    q2_4m = max(max(cp[t]["q2_err_gm_registered_matched"], cp[t]["q2_err_gds_registered_matched"]) for t in dev4)
    rows = [
        {"chk": "Q2 capacitances", "val": e(s["cap_max"]), "dev": ", ".join(A["cap"]),
         "kind": "reference run", "cause": "BSIM4 A2 > 1 clamp state differs (card-only vs element deck)",
         "post": f"matched state: {C['cap_max_matched']:.1e}"},
        {"chk": "Q2 derivatives", "val": e(q2_4), "dev": ", ".join(t.split("_", 1)[1] for t in dev4) + " (nmos)",
         "kind": "reference run", "cause": "same A2 clamp state; card op current differs by up to "
                                           f"{max(cp[t]['op_card_stored_vs_matched_max_rel'] for t in dev4):.3f}",
         "post": f"matched state: {q2_4m:.1e}"},
        {"chk": "Q2 derivatives", "val": e(max(l100["q2_err_gm_registered_stored"],
                                                l100["q2_err_gds_registered_stored"])),
         "dev": "nmos_L100_W100", "kind": "reference formula",
         "cause": "Re Y_card at 1 MHz != static conductance (rbodymod = 1)",
         "post": f"element identity: {C['max_over_devices']['identity_matched']:.1e} (18 devices)"},
        {"chk": "exchange asymmetry", "val": e(s["swap_asym_max"]), "dev": ", ".join(A["swap_asym"]),
         "kind": "element", "cause": "impact-ionization current in the sensed replica drain current",
         "post": "predicted " + ", ".join(f"{dd[t]['predicted']:.4f}" for t in A["swap_asym"])
                 + " (observed " + ", ".join(f"{dd[t]['max']:.4f}" for t in A["swap_asym"]) + ")"},
    ]
    notes = ["Post hoc (after decision.json); the registered outcome 'not deployable' is unchanged. Source: "
             "scripts/research/explore/i12_1_diagnosis.py (about 18 s of single-process ngspice runs).",
             "A2 clamp state: ngspice-41 b4check.c checks each model name once per process (static modname) and "
             "clamps the size-dependent A2 > 1 to 1 (A1 to 0); b4temp.c rebuilds the size-dependent parameters at "
             "every analysis. The card-only AC deck (600 x op + AC in one process) therefore uses A2 = 1 only in its "
             "first analysis; in the element deck the card and replica model names alternate, the check runs every "
             "time and A2 = 1 throughout. M2 card a2: " + ", ".join(
                 f"{t} {d['diag']['A_values'][t]['a2_M2']:.3f}" for t in dev4)
             + "; single-parameter search on nmos_L0p19_W7: a2 alone produces the drift, removing a2 removes it. "
             "Matched check state: the card-only AC reference rerun with the replica instance in the deck (VCVS-"
             "driven, current through a 0 V source, no OSDI module).",
             "nmos_L100_W100: Re Y_dg at Vgs 0, Vds 0.05 V is "
             f"{f2['rbodymod_1_card']['1e+06']['re_y_dg']:.4e} S at 1 MHz, "
             f"{f2['rbodymod_1_card']['100']['re_y_dg']:.4e} S at 100 Hz, and "
             f"{f2['rbodymod_0']['1e+06']['re_y_dg']:.4e} S at 1 MHz with rbodymod = 0 (body resistor network x "
             "100 um x 100 um capacitances). The registered reference exp(g) Re Y_card + I exp(g) dg/dV assumes "
             "Re Y_card is the static conductance; the element adds (exp(g) - 1) x the replica's static conductance. "
             "With that identity (matched state) the max difference over 18 devices is "
             f"{C['max_over_devices']['identity_matched']:.1e}.",
             f"Exchange asymmetry: {worst} max at Vgs {dd[worst]['vgs']:.2f} V, Vds {dd[worst]['vds']:.2f} V, I "
             f"{dd[worst]['I_A']:.2e} A; drain non-channel current J = |I_fwd,card| - |i(VZ)| = {dd[worst]['J_A']:.3e} "
             f"A (J/I {dd[worst]['J_over_I']:.2f}); setting alpha0 = 0 lowers the card current by "
             f"{iii[worst]['difference_A']:.3e} A. J rises with Vgs and Vds (impact ionization, drain to body). The "
             "replica drain-terminal current carries it in the forward bench only; the pre-registered bound assumed "
             "J = gmin x V <= 1.85e-12 A.",
             "The A2 clamp state also applies to every card-only multi-analysis deck, including the measured-curve "
             "batch deck (one dc analysis per curve): for cards with a2 > 1 the first curve uses A2 = 1 and the "
             "other curves the card value. Second-analysis drift > 1e-6 (M2 cards): " + ", ".join(dev4)
             + "; the starting cards (direct43_r3_fd) show the same drift on the same devices; published cards 0.",
             "Smoke devices (nmos_L1_W1p6, pmos_L2_W5): a2 <= 1, J/I <= 1e-3, Re Y_card at 1 MHz equal to the "
             "static value."]
    return rf.Table("05_table_i12_1_diagnosis", "Post hoc diagnosis of the three failed core checks",
                    [col("chk", "check", kind="text"), col("val", "registered value"),
                     col("dev", "devices over 1e-3", kind="text"), col("kind", "source of failure", kind="text"),
                     col("cause", "cause", kind="text"), col("post", "post hoc recomputed", kind="text")],
                    rows, notes)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def tags_sorted(d) -> list[str]:
    return sorted(d["per"]["C0_refit"], key=rf.device_key)


def fig_per_device(d: dict, path: Path) -> list[str]:
    tags = tags_sorted(d)
    x = np.arange(len(tags))
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 7.6), dpi=DPI, gridspec_kw={"height_ratios": [2.2, 1]},
                                  sharex=True)
    fig.patch.set_facecolor(BG)
    style(ax)
    style(ax2)
    off = {"published": -0.24, "C0_refit": -0.08, "M_bias_v1": 0.08, "M_bias_v2": 0.24}
    for a, dx in off.items():
        ax.scatter(x + dx, [d["per"][a][t] for t in tags], color=COL[a], marker=MK[a], s=30, zorder=3,
                   label=f"{ARM_LABEL[a]}: all-device {d['stats'][a]['all']:.4f}")
    ax.set_yscale("log")
    ax.set_ylabel("held-out RRMS (log)", fontsize=9)
    ax.axvline(7.5, color=GREY, lw=0.8)
    ax.set_ylim(0.008, 1.0)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="lower right")
    ax.set_title("Per-device LODO RRMS (strict paper metric, frozen inclusion)", fontsize=10)
    for a, c, dx in (("M_bias_v1", ORANGE, -0.15), ("C0_refit", BLUE, 0.15)):
        r = [d["per"]["M_bias_v2"][t] / d["per"][a][t] for t in tags]
        ax2.bar(x + dx, r, width=0.3, color=c, alpha=0.8, label=f"v2 / {a}")
    ax2.axhline(1.0, color=INK, lw=0.8)
    ax2.axvline(7.5, color=GREY, lw=0.8)
    ax2.set_ylabel("RRMS ratio", fontsize=9)
    ax2.set_ylim(0.4, 1.4)
    ax2.legend(fontsize=8, frameon=False, ncol=2, loc="upper left")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tags, rotation=60, ha="right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(F["lodo"]), rel(F["ref"])]


def fig_gmr(d: dict, path: Path) -> list[str]:
    C = d["cmp"]
    rows = [("A1 (decides): GMR(v2, C0_refit)", C["A1"], RED, "CI upper < 1"),
            ("A2 (decides): GMR(v2, v1)", C["A2"], RED, "CI upper <= 1.05"),
            ("reported: GMR(v2, T1_inloop)", C["v2_T1"], GREY, "no rule"),
            ("reference: GMR(v1, C0_refit) (I3-2 all18)", C["v1_C0"], GREY, "no rule")]
    fig, ax = plt.subplots(figsize=(10.5, 3.8), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    for i, (lab, c, colr, rule) in enumerate(rows):
        y = -i
        ax.errorbar(c["ratio"], y, xerr=[[c["ratio"] - c["ci_low"]], [c["ci_high"] - c["ratio"]]], fmt="o",
                    color=colr, ms=7, capsize=4, lw=2)
        ax.annotate(f"{ci(c)}, {c['wins']}/18 lower; rule: {rule}", (1.17, y), fontsize=8, va="center",
                    color=INK, annotation_clip=False)
    ax.axvline(1.0, color=INK, lw=0.9)
    ax.axvline(1.05, color=RED, lw=0.9, ls="--")
    ax.text(1.052, 0.45, "A2 margin 1.05", fontsize=7.5, color=RED)
    ax.set_yticks([-i for i in range(len(rows))])
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlim(0.6, 1.16)
    ax.set_ylim(-len(rows) + 0.4, 0.8)
    ax.set_xlabel("GMR of held-out RRMS, 18 devices, device bootstrap 10,000, 95% percentile CI (< 1: first arm "
                  "lower)", fontsize=8.5)
    fig.suptitle(f"I12-1 accuracy contrasts: A1 passes, A2 fails; outcome {d['decision']['outcome']}",
                 fontsize=10, color=INK)
    fig.subplots_adjust(left=0.3, right=0.66, top=0.88, bottom=0.17)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(F["decision"]), rel(F["lodo"])]


def fig_diag(d: dict, path: Path) -> list[str]:
    g = d["diag"]
    tags = tags_sorted(d)
    x = np.arange(len(tags))
    cp, dd, B = g["C_q2_recomputed"]["per_device"], g["D_exchange_asymmetry"]["per_device"], \
        g["B_second_analysis_drift"]["per_device"]
    q12 = d["q12"]["devices"]
    fl = 1e-13
    fig, axs = plt.subplots(2, 2, figsize=(14, 8.4), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for a in axs.flat:
        style(a)
        a.set_yscale("log")
        a.set_xticks(x)
        a.set_xticklabels(tags, rotation=60, ha="right", fontsize=6.8)
    a = axs[0, 0]
    for k, c, mk, lab, dx in (("published", GREY, "s", "published card", -0.2), ("start", BLUE, "o",
                               "starting card direct43_r3_fd", 0.0), ("M2", RED, "^", "M2 card", 0.2)):
        a.scatter(x + dx, [max(B[t][k], fl) for t in tags], color=c, marker=mk, s=24, label=lab, zorder=3)
    a.set_ylabel("|I(op 2) - I(op 1)| / |I(op 1)|", fontsize=8.5)
    a.set_title("(a) card-only deck, two op at the same bias in one process\n(BSIM4 A2 > 1 clamp in the first "
                f"analysis only; {fl:.0e} = 0)", fontsize=9)
    a.legend(fontsize=7.5, frameon=False)
    a = axs[0, 1]
    for k, c, mk, lab, dx in (("registered_stored", RED, "o", "registered (stored card-only AC reference)", -0.2),
                              ("registered_matched", ORANGE, "s", "registered formula, matched check state", 0.0),
                              ("identity_matched", AQUA, "D", "element identity with replica Re Y, matched", 0.2)):
        a.scatter(x + dx, [max(cp[t][f"q2_err_gm_{k}"], cp[t][f"q2_err_gds_{k}"], fl) for t in tags], color=c,
                  marker=mk, s=24, label=lab, zorder=3)
    a.axhline(1e-3, color=RED, lw=0.8, ls="--")
    a.set_ylabel("max normalized gm / gds difference", fontsize=8.5)
    a.set_title("(b) Q2 derivative check: element AC vs reference (threshold 1e-3 dashed)", fontsize=9)
    a.legend(fontsize=7.5, frameon=False, loc="upper right")
    a = axs[1, 0]
    a.scatter(x - 0.1, [max(q12[t]["cap_max"], fl) for t in tags], color=RED, marker="o", s=24,
              label="registered (stored card-only AC reference)", zorder=3)
    a.scatter(x + 0.1, [max(cp[t]["cap_max_matched"], fl) for t in tags], color=AQUA, marker="D", s=24,
              label="matched check state", zorder=3)
    a.axhline(1e-3, color=RED, lw=0.8, ls="--")
    a.set_ylabel("max capacitance error (Y_dd, Y_dg, Y_gg)", fontsize=8.5)
    a.set_title(f"(c) Q2 capacitance check ({fl:.0e} = 0)", fontsize=9)
    a.legend(fontsize=7.5, frameon=False)
    a = axs[1, 1]
    a.scatter(x - 0.1, [dd[t]["max"] for t in tags], color=RED, marker="o", s=24, label="observed (element-introduced)",
              zorder=3)
    a.scatter(x + 0.1, [dd[t]["predicted"] for t in tags], color=INK, marker="x", s=24,
              label="predicted |1 - exp(-g)| J / I at the same point", zorder=3)
    a.axhline(1e-3, color=RED, lw=0.8, ls="--")
    a.set_ylabel("|d ln(I_swap / I_fwd)|", fontsize=8.5)
    a.set_title("(d) exchange asymmetry; J = drain non-channel current (gmin x V + impact ionization)", fontsize=9)
    a.legend(fontsize=7.5, frameon=False)
    fig.suptitle("I12-1 post hoc diagnosis of the failed core checks (does not change the outcome)", fontsize=10.5,
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(F["diag"]), rel(F["q12"])]


# ---------------------------------------------------------------------------
# I-V (NGSpice re-simulation of the stored held-out cards)
# ---------------------------------------------------------------------------
def resim(d: dict) -> dict:
    sys.path.insert(0, str(ROOT / "scripts/research/explore"))
    from e1_collect_real import collect
    from plot_explore_i3_2 import published_rrms
    from i3_2_bias import load_records_all18
    recs = {r.tag: r for r in load_records_all18()[0]}
    c2 = load(F["corr"])
    c1 = load(F["v1_corr"])
    k2 = load(F["cards"])
    k1 = load(F["v1_cards"])
    out = {}
    pub, pub_curves = published_rrms({t: recs[t] for t, _ in IV_DEVICES})
    for tag, why in IV_DEVICES:
        assert abs(pub[tag] - d["per"]["published"][tag]) < 1e-9
        r_c0 = collect(tag, k1["C0_refit"][tag])
        r_v1 = collect(tag, k1["M_bias"][tag])
        r_v2 = collect(tag, k2[tag])
        g1 = correction_from_json(c1[f"M_bias|{tag}"])
        g2 = m.SymCorrection.from_json(c2[f"M_bias_v2|{tag}"])
        sims = {"C0_refit": [c.sim for c in r_c0.curves], "M_bias_v1": g1.apply(r_v1), "M_bias_v2": g2.apply(r_v2)}
        sc = {"C0_refit": score_device(r_c0, sims["C0_refit"])["rrms"],
              "M_bias_v1": score_device(r_v1, sims["M_bias_v1"])["rrms"],
              "M_bias_v2": score_device(r_v2, sims["M_bias_v2"])["rrms"]}
        for a, v in sc.items():
            assert abs(v - d["per"][a][tag]) < 1e-6, (tag, a, v, d["per"][a][tag])
        out[tag] = {"why": why, "rec": recs[tag], "pub": pub_curves[tag], "sims": sims, "score": sc,
                    "pub_rrms": pub[tag]}
    return out


def fig_iv(d: dict, tag: str, path: Path) -> list[str]:
    r = d["iv"][tag]
    rec = r["rec"]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4), dpi=DPI)
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
            for a, ls in (("C0_refit", "-"), ("M_bias_v1", "-"), ("M_bias_v2", "-")):
                ax.plot(xv, np.abs(r["sims"][a][i]), color=COL[a], ls=ls, lw=1.2, alpha=alpha, zorder=3)
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
        Line2D([], [], color=COL["C0_refit"], lw=1.2,
               label=f"NGSpice re-simulated, C0_refit held-out card (no correction): RRMS {sc['C0_refit']:.4f}"),
        Line2D([], [], color=COL["M_bias_v1"], lw=1.2,
               label=f"NGSpice re-simulated, M_bias v1 held-out card x exp(g v1 fold): RRMS {sc['M_bias_v1']:.4f}"),
        Line2D([], [], color=COL["M_bias_v2"], lw=1.2,
               label=f"NGSpice re-simulated, M_bias v2 held-out card x exp(g v2 fold): RRMS {sc['M_bias_v2']:.4f}"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=8.2, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"{tag} held out ({r['why']}); LODO, g trained on the other 17 devices, card refit with g in the "
                 "loop.\nCurves: canonical ngspice-41 batch deck of the stored held-out card on the measured bias "
                 "grid, times exp(g) in Python (as scored); RRMS equals the stored LODO row.", fontsize=9.4, color=INK)
    fig.tight_layout(rect=(0, 0.15, 1, 0.93))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(EXP / "E1/real_dev.npz"), rel(EXP / "E1/real_test.npz"), rel(F["cards"]), rel(F["v1_cards"]),
            rel(F["corr"]) + f" (M_bias_v2|{tag})", rel(F["v1_corr"]) + f" (M_bias|{tag})",
            "upstream committed {vd,vg}_sweep/*.txt"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    folder = rf.REPORT_ROOT / f"{args.date}_{TOPIC}"
    folder.mkdir(parents=True, exist_ok=True)
    d = load_all()
    d["iv"] = resim(d)
    items, tables = [], []

    def add_table(t, desc, srcs):
        rf.render_table(t, folder / f"{t.name}.png")
        tables.append(t)
        items.append((f"{t.name}.png", desc, srcs))

    def add_fig(name, fn, desc):
        items.append((name, desc, fn(d, folder / name)))

    add_table(t_performance(d), "Table of the LODO RRMS per arm (all-device, combined, median, family means), the "
              "GMR against C0_refit, A1 / A2 and the outcome", [rel(F["lodo"]), rel(F["decision"]), rel(F["ref"])])
    add_fig("02_i12_1_per_device.png", fig_per_device, "Per-device held-out RRMS for the published card, C0_refit, "
            "M_bias v1 and M_bias v2 (log), and the ratios v2 / v1 and v2 / C0_refit")
    add_fig("03_i12_1_gmr.png", fig_gmr, "GMR with 95% device-bootstrap CI for A1 (v2 vs C0_refit), A2 (v2 vs v1, "
            "margin 1.05), and the reported v2 vs T1_inloop and v1 vs C0_refit")
    add_table(t_deploy(d), "Table of the part B deployment checks (core and restriction): value, threshold, result "
              "and the devices over threshold", [rel(F["summary"]), rel(F["q12"]), rel(F["decision"])])
    add_table(t_diagnosis(d), "Table of the post hoc diagnosis of the three failed core checks: cause, evidence and "
              "the recomputed value", [rel(F["diag"])])
    add_fig("06_i12_1_diagnosis.png", fig_diag, "Post hoc diagnosis per device: (a) card-only second-analysis drift "
            "(BSIM4 A2 clamp state), (b) Q2 derivative difference against the registered and the recomputed "
            "references, (c) capacitance error registered vs matched check state, (d) exchange asymmetry observed "
            "vs predicted from the drain non-channel current")
    for k, (tag, why) in enumerate(IV_DEVICES):
        name = f"{7 + k:02d}_i12_1_iv_{tag}.png"
        items.append((name, f"I-V of {tag} ({why}): measured 77 K points, NGSpice of the published card, and NGSpice "
                            "re-simulated held-out cards C0_refit, M_bias v1 x exp(g v1), M_bias v2 x exp(g v2)",
                      fig_iv(d, tag, folder / name)))

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i12_1.py --date {args.date}`"
    md = [f"# Tables, {args.date} {TOPIC}", "", f"Generated {now} by {gen}.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))

    S, C, dec = d["stats"], d["cmp"], d["decision"]
    s = d["summary"]["values"]
    g = d["diag"]
    ref = d["summary"]["refit_in_sample"]
    readme = [
        f"# Report batch {args.date} · {TOPIC}", "",
        f"Generated {now} by {gen}. NGSpice runs: re-simulation of 9 stored held-out cards for the I-V figures "
        "(checked against the stored LODO rows); everything else from stored files.", "",
        "## Contents", "",
        "- I12-1: bias-only correction g v2 (inputs is_pmos, smooth max(sVgs, sVgd)/VDD, sabs(Vds)/VDD; sabs(x) = "
        "sqrt(x^2 + 0.01^2); gm/gds penalty lambda 10 V) as an ngspice-41 OSDI add-on that scales the drain current "
        "of a charge-free card replica. Pre-registration `out/research/explore/I12_1/prereg.md` (in force from "
        f"{COMMITS['prereg']}; code {COMMITS['code']}; amendment A2 {COMMITS['A2']}). Outputs "
        "`out/research/explore/I12_1/`.",
        f"- Part A (LODO, 18 devices): all-device / combined / median: v2 {S['M_bias_v2']['all']:.4f} / "
        f"{S['M_bias_v2']['combined']:.4f} / {S['M_bias_v2']['median']:.4f}; v1 {S['M_bias_v1']['all']:.4f} / "
        f"{S['M_bias_v1']['combined']:.4f} / {S['M_bias_v1']['median']:.4f}; T1_inloop {S['T1_inloop']['all']:.4f} / "
        f"{S['T1_inloop']['combined']:.4f} / {S['T1_inloop']['median']:.4f}; C0_refit {S['C0_refit']['all']:.4f} / "
        f"{S['C0_refit']['combined']:.4f} / {S['C0_refit']['median']:.4f}; published {S['published']['all']:.4f} / "
        f"{S['published']['combined']:.4f} / {S['published']['median']:.4f}. A1 GMR(v2, C0_refit) {ci(C['A1'])}, "
        f"{C['A1']['wins']}/18 (passes). A2 GMR(v2, v1) {ci(C['A2'])}, {C['A2']['wins']}/18 (fails, margin 1.05).",
        f"- Part B (M2 cards + full-data g v2): {dec['deploy']['deploy']}. Core fails: "
        f"{'; '.join(dec['deploy']['core_fail'])}. Restrictions: {'; '.join(dec['deploy']['restrictions'])}. "
        f"Q1 tight 0 violating points (forward and exchanged), Vds = 0 max {s['q1_vds0_max']:.1e} A, Verilog-A g "
        f"{s['g_text_max']:.1e} / exchanged {s['g_sym_text_max']:.1e}, 0 gm sign changes, 0 element-only circuit "
        f"failures. In-sample RRMS M2: Python {ref['mean_rrms_M2_python']:.4f}, SPICE-native "
        f"{ref['mean_rrms_M2_native']:.4f}, card only {ref['mean_rrms_M2_card_only']:.4f}.",
        f"- **Outcome (pre-registered rule): {dec['outcome']}; action: {dec['action']}.**",
        "- Post hoc diagnosis (table 05, figure 06; `scripts/research/explore/i12_1_diagnosis.py`): (1) Q2 "
        "capacitance and the Q2 derivative failures on nmos_L0p19_W7, nmos_L0p25_W1p6, nmos_L20_W0p64, nmos_L8_W1p6 "
        "come from the reference run, not the element: ngspice-41 applies the BSIM4 'A2 > 1 -> 1' clamp only in the "
        "first analysis of a single-model process, and in every analysis when the card and replica models alternate; "
        f"with the reference in the matched state the capacitance error is {g['C_q2_recomputed']['cap_max_matched']:.1e}. "
        "(2) nmos_L100_W100 Q2: the registered reference uses Re Y_card at 1 MHz, which contains a body-resistor "
        "(rbodymod = 1) term; with the replica's static conductance the max difference over 18 devices is "
        f"{g['C_q2_recomputed']['max_over_devices']['identity_matched']:.1e}. (3) Exchange asymmetry is an element "
        "property: the replica drain-terminal current includes impact-ionization substrate current on the forward "
        "side only (nmos_L100_W100 0.0135, nmos_L20_W0p64 0.0065; predicted from J/I within 1e-4).",
        "- The A2 clamp state also affects every card-only multi-analysis deck with a2 > 1 (measured-curve batch "
        "deck: first curve A2 = 1, other curves the card value), including the stored 43-parameter starting cards.",
        "", "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))
    summary = [f"I12-1 g v2 deployable add-on, sky130 77 K, 18 devices: {dec['outcome']} (pre-registered rule). A1 "
               f"GMR(v2, C0_refit) {ci(C['A1'])} passes; A2 GMR(v2, v1) {ci(C['A2'])} fails (margin 1.05).",
               f"All-device / combined: v2 {S['M_bias_v2']['all']:.4f} / {S['M_bias_v2']['combined']:.4f}, v1 "
               f"{S['M_bias_v1']['all']:.4f} / {S['M_bias_v1']['combined']:.4f}, C0_refit {S['C0_refit']['all']:.4f} / "
               f"{S['C0_refit']['combined']:.4f}.",
               "Post hoc: capacitance and 4 of 5 derivative failures are a reference-run artifact (BSIM4 A2 clamp "
               "applied only in the first analysis); nmos_L100_W100 derivative: 1 MHz body-resistor term in the "
               "reference; exchange asymmetry: impact-ionization current in the sensed drain current."]
    rf.update_index(args.date, TOPIC, folder, [i[0] for i in items], summary)
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
