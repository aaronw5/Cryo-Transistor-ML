#!/usr/bin/env python3
"""Report batch for exploration direction I3-2: bias-only learned correction
g(is_pmos, |Vgs|/VDD, |Vds|/VDD) with the held-out card refit in the loop.

Reads stored files only (no NGSpice, no training): the I3_2 result files
(``out/research/explore/I3_2/{real_dev,gf180_c_logo,ihp_c_logo}/``), the stored
corrections (``corrections.json``), the E1 records (measured curves and the
starting-card NGSpice curves on the measured bias grid), the upstream committed
NGSpice sweeps of the published (upstream released) 77 K card
(``data/raw/CryoPDK_Skywater130nm_ML/{nfet,pfet}_mod/<dev>/{vd,vg}_sweep``) and,
when present, ``out/research/explore/I3_2/decision.json``.

Writes PNGs, ``tables.md`` and ``README.md`` to
``out/research/report/<date>_i3_2_bias/`` and adds the batch to
``out/research/report/index.json`` and ``INDEX.md``.

Every stored aggregate and paired ratio used in a table is recomputed from the
stored per-device values and checked against the stored value.

  PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i3_2.py --date 2026-10-20
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_explore_e2_e3_e4 as pe  # noqa: E402  (helpers, style)
from plot_explore_e2_e3_e4 import (BG, INK, MUTED, GREY, BLUE, ORANGE, AQUA,  # noqa: E402
                                   VIOLET, RED, DPI, EXP, ROOT, fam, load, real_dev, rel,
                                   style, yn)

sys.path.insert(0, str(ROOT / "scripts"))
from verify_simulator import upstream_metric_curves  # noqa: E402  (reads committed files)
from cryoml.explore.e1_correction import load_devices, score_device  # noqa: E402
from cryoml.explore.i3_2_bias import correction_from_json, paired_ratio  # noqa: E402

from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

rf = pe.rf
plt = pe.plt

I32 = EXP / "I3_2"
RUNS = {"real": I32 / "real_dev", "gf180": I32 / "gf180_c_logo", "ihp": I32 / "ihp_c_logo"}
PREREG = I32 / "prereg.json"
DECISION = I32 / "decision.json"
E1 = EXP / "E1"
CARDS = ROOT / "out/research/polished/direct43_r3_fd/real/summary.json"
REF_SCORES = ROOT / "out/research/reference_scores.json"
PDK_LABEL = {"gf180": "GF180 (BSIM4) tier c", "ihp": "IHP SG13G2 (PSP103) tier c"}
T1_PLUS, PDK_RATIO = 0.005, 0.90

ARM_LABEL = {"C0_refit": "C0_refit (control, no correction)",
             "T1_inloop": "T1_inloop (per-tag gains, not deployable)",
             "M_bias": "M_bias (primary, 3 inputs)",
             "M_bias_geom": "M_bias_geom (secondary, + log L, log W)"}
ARM_COLOR = {"C0_refit": INK, "T1_inloop": ORANGE, "M_bias": BLUE, "M_bias_geom": VIOLET}
ARM_MARK = {"C0_refit": "o", "T1_inloop": "v", "M_bias": "D", "M_bias_geom": "s"}


def f3(x, n=3) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.{n}f}"


def ci_text(c: dict) -> str:
    return f"{c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"


def cluster_of(tag: str) -> str:
    return tag.split("#")[0]


def gmr(per: dict, a: str, b: str, real: bool) -> dict:
    clusters = None if real else {t: cluster_of(t) for t in per[a]}
    return paired_ratio(per[a], per[b], clusters)


def check_cmp(res: dict, key: str, mine: dict) -> None:
    st = res["comparisons"].get(key)
    if st is None:
        return
    for k in ("ratio", "ci_low", "ci_high"):
        assert abs(st[k] - mine[k]) < 1e-12, (key, k, st[k], mine[k])
    assert st["wins"] == mine["wins"], key


# ================================================================== published card

def published_rrms(recs: dict) -> dict:
    """Strict frozen-inclusion RRMS of the published (upstream released) 77 K
    card, from the upstream committed NGSpice sweeps (same grids and curve
    order as the records; checked), checked against
    out/research/reference_scores.json ``released``."""
    ref = load(REF_SCORES)["released"]["devices"]
    out, curves = {}, {}
    for tag, rec in recs.items():
        dev = SimpleNamespace(dev_type=rec.family, L_um=rec.L_um, W_um=rec.W_um)
        cs, _, _ = upstream_metric_curves(dev)
        sims = []
        for c, rc in zip(cs, rec.curves):
            kind, bias = rc.key.split("@")
            assert c.kind == kind and abs(abs(c.fixed) - float(bias)) < 1e-9, (tag, rc.key)
            vs = c.Vd if kind == "idvd" else c.Vg
            vr = rc.vd if kind == "idvd" else rc.vg
            assert len(vs) == len(vr) and np.allclose(np.abs(vs), np.abs(vr), atol=1e-9), (tag, rc.key)
            # upstream_metric_curves returns sign * Id (positive drain current
            # magnitude for pMOS); the records keep the simulator sign
            sims.append(np.sign(np.nanmean(rc.sim[-5:])) * np.abs(c.Id))
        out[tag] = score_device(rec, sims)["rrms"]
        assert abs(out[tag] - ref[tag]["rrms"]) < 1e-8, (tag, out[tag], ref[tag]["rrms"])
        curves[tag] = sims
    return out, curves


# ================================================================== 1 process diagram

def fig_process(real: dict, path: Path) -> list[str]:
    fig = plt.figure(figsize=(15, 11.2), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 150)
    ax.set_ylim(25, 135)
    ax.axis("off")
    KIND = {"meas": ("#fbe3cf", ORANGE, "measured data only"),
            "sim": ("#dbe8f7", BLUE, "NGSpice simulation only"),
            "both": ("#e1f2ee", AQUA, "measured data and NGSpice simulation"),
            "stat": ("#ecebe8", GREY, "statistics on stored RRMS values (no data, no simulation)")}

    def box(x, y, w, h, kind, title, body, fs=8.6):
        face, edge, _ = KIND[kind]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    facecolor=face, edgecolor=edge, lw=1.6, zorder=2))
        ax.text(x + 1.2, y + h - 1.4, title, fontsize=fs + 1.2, fontweight="bold", color=INK,
                va="top", ha="left", zorder=3)
        wrap = int(w * 1.55 * 8.6 / fs)
        lines = []
        for para in body:
            lines += textwrap.wrap(para, wrap) or [""]
        ax.text(x + 1.2, y + h - 4.4, "\n".join(lines), fontsize=fs, color=INK, va="top",
                ha="left", zorder=3, linespacing=1.32)

    def arrow(x0, y0, x1, y1, text=None, side="right", color=MUTED):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, mutation_scale=16),
                    zorder=4)
        if text:
            ax.text((x0 + x1) / 2 + (1.2 if side == "right" else -1.2), (y0 + y1) / 2, text,
                    fontsize=7.8, color=MUTED, ha="left" if side == "right" else "right",
                    va="center", zorder=4)

    opts = real["options"]
    hyp = real["hyper"]
    tim = real["timing"]
    s = real["summary"]
    c = real["comparisons"]["M_bias_vs_C0_refit"]

    ax.text(2, 133, "I3-2 pipeline on REAL_DEV (12 measured devices): how each held-out RRMS value "
            "is produced", fontsize=14, fontweight="bold", color=INK, va="top")
    ax.text(2, 129.3, "Pre-registration out/research/explore/I3_2/prereg.json (commit f7f6378); code "
            "src/cryoml/explore/i3_2_bias.py, scripts/research/explore/i3_2_bias.py (commit 4c7c66b); "
            f"queue job I3_2_real, {real['runtime_s']:.0f} s.", fontsize=8.8, color=MUTED, va="top")

    # row 1: inputs
    box(2, 110, 46, 17, "meas", "1. Measured 77 K I-V, 12 REAL_DEV devices",
        ["5 nMOS, 7 pMOS from out/research/explore/real_split.json (REAL_TEST, 6 devices, is not "
         "loaded by any I3_2 script).",
         "11 curves per device: Id-Vd at |Vgs| = 0.37-1.85 V, Id-Vg at |Vds| = 0.01-1.85 V, "
         "186 points each.",
         "Curve inclusion and trim start frozen from the published-card baseline."])
    box(52, 110, 46, 17, "both", "2. Starting 43-parameter BSIM4 cards",
        ["One card per device: direct NGSpice fit, round 3, then finite-difference (FD) "
         "least-squares polish against the measured curves.",
         "File: out/research/polished/direct43_r3_fd/real/summary.json; parameter box: round-3 box "
         "(out/cold_loop/full_direct/round_3/spec.json).",
         "These cards were fitted to all 12 devices before I3-2."])
    box(102, 110, 46, 17, "sim", "3. NGSpice on the measured bias grid",
        ["Each starting card is simulated on its device's measured bias points (ngspice-41) -> "
         "I_sim for all 11 curves.",
         "Stored once as the E1 records out/research/explore/E1/real_dev.{npz,json} "
         "(measured target and I_sim per point).",
         "Training data of every correction = these stored records (no joint rounds)."])
    arrow(48, 118.5, 52, 118.5)
    arrow(98, 118.5, 102, 118.5)

    # LODO frame
    ax.add_patch(Rectangle((1, 49), 148, 57.5, facecolor="none", edgecolor=MUTED, lw=1.2,
                           ls=(0, (5, 4)), zorder=1))
    ax.text(3, 105.8, "4. Leave-one-device-out, 12 folds: each device in\nturn is the held-out device d; "
            "the other 11 train g", fontsize=9.5, fontweight="bold", color=INK, va="top",
            linespacing=1.25)
    arrow(125, 110, 125, 100.5, "records of the 11 training devices", side="right")
    arrow(62, 110, 62, 100.5)
    ax.text(63.2, 105.3, "starting card of d", fontsize=7.8, color=MUTED, ha="left", va="center")

    eq_g = ("g(x) = (1/5) Σ_s tanh(h_s(x)),  x = (is_pmos, |Vgs|/VDD, |Vds|/VDD), VDD = 1.85 V; "
            "|Vgs|/VDD and |Vds|/VDD standardized with the mean and std of the fold's training "
            "points; is_pmos kept 0/1.")
    box(76, 73, 72, 27, "both", "5. Train the correction g on the 11 training devices",
        [eq_g,
         f"h_s: MLP 3-{hyp['hidden']}-{hyp['hidden']}-1, tanh hidden units, last layer initialized "
         "to zero (g = 0 at step 0); 353 weights per member; seeds s = 0-4.",
         "Corrected current: I_corr = I_sim · exp(g(x)) at each measured bias point "
         "(keeps sign and zero crossings of I_sim).",
         "Loss = (1/11) Σ_devices RRMS_dev(I_corr, I_meas) + 1e-4 Σ ||W||²; RRMS_dev = mean over "
         "included curves of RMSE(trimmed) / mean|I_meas|(trimmed) (paper metric).",
         f"Adam, lr {hyp['lr']:g}, {hyp['steps']} full-batch steps, float64. g is then frozen.",
         f"Arm M_bias_geom (secondary): same, with log10 L and log10 W added (5-16-16-1). "
         f"Training time: {tim['train_s']:.0f} s for {tim['train_n_seed_tasks']} seed tasks "
         "(12 folds x 2 arms x 5 seeds + 5 for the full model)."], fs=8.3)

    box(2, 73, 70, 27, "both", "6. Refit the held-out card with g frozen in the loop",
        ["Start: the held-out device's starting card (box 2). Fit variables: the 43 parameters "
         "(round-3 box, local coordinates).",
         "Model current: I(card) = I_bsim(card) · exp(g(x)); I_bsim = fresh NGSpice simulation of "
         "the trial card on the measured bias grid.",
         "Residuals against the held-out device's measured curves; ngspice_fit.polish (trust-region "
         f"least squares), max_nfev {opts['heldout_nfev']}, IRLS rounds {opts['heldout_irls']}, "
         f"diff_step {opts['diff_step']:g}.",
         "The held-out device's measured curves are used only here (card extraction); they are "
         "not used to train g.",
         f"Refit time: {tim['heldout_refit_s']:.0f} s for {tim['heldout_refit_n']} refits "
         "(M_bias, M_bias_geom, C0 and T1 reruns, 12 each), 175-403 NGSpice runs per refit."],
        fs=8.3)
    arrow(76, 86, 72, 86, None)
    ax.text(74, 88, "frozen g", fontsize=7.8, color=MUTED, ha="center", va="bottom")

    box(2, 52, 70, 16.5, "both", "7. Score the held-out device",
        ["Best card re-simulated in NGSpice, multiplied by exp(g), compared with the measured "
         "curves: strict paper RRMS with the frozen inclusion (e1_correction.score_device -> "
         "metrics.device_rrms_new).",
         "A non-finite included curve makes the device NaN (failed). Result: one RRMS value per "
         "held-out device and arm."], fs=8.3)
    arrow(37, 73, 37, 68.5)

    box(76, 52, 72, 16.5, "both", "Control and comparator (same box 6-7 settings)",
        ["C0_refit: refit with g = 0 (no correction); stored I2-1 rows reused, rerun here as a "
         "check (max |rerun - stored| = "
         f"{real['reproduction']['C0_refit']['max_abs_diff_rerun_vs_stored']:g}).",
         "T1_inloop: one log-gain per (family, curve tag) trained post-hoc (E1 T1); stored E1 "
         "stage-2 rows reused, rerun as a check (max diff "
         f"{real['reproduction']['T1_inloop']['max_abs_diff_rerun_vs_stored']:g}). T1 uses the "
         "curve tag and is not a device model."], fs=8.3)

    box(2, 27, 96, 19, "stat", "8. Aggregate over the 12 held-out devices and decide",
        [f"All-device mean, combined (mean of nMOS and pMOS family means), median, failed count. "
         f"M_bias: {s['M_bias']['all_device_mean']:.4f} all-device, {s['M_bias']['combined']:.4f} "
         f"combined; C0_refit {s['C0_refit']['all_device_mean']:.4f}, T1_inloop "
         f"{s['T1_inloop']['all_device_mean']:.4f}.",
         "Paired geometric-mean ratio R = exp(mean_d ln(RRMS_M_bias,d / RRMS_C0,d)); 95% CI = "
         "2.5/97.5 percentiles over 10000 device resamples (numpy default_rng(0)). "
         f"R = {c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}], lower on {c['wins']}/12.",
         "Rules R1-R3 on REAL_DEV; R4 on the simulated GF180 / IHP tier-c runs (same pipeline with "
         "simulated truth curves in place of the measured curves, leave-one-geometry-out, 48 "
         "targets each)."], fs=8.3)
    arrow(37, 52, 37, 46.5)
    arrow(112, 52, 80, 46.5)

    handles = [Rectangle((0, 0), 1, 1, facecolor=v[0], edgecolor=v[1], lw=1.4, label=v[2])
               for v in KIND.values()]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.995, 0.01), fontsize=8.6,
              title="box colour: what the step uses", title_fontsize=9, frameon=True,
              facecolor=BG, edgecolor=GREY)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(PREREG), rel(RUNS["real"] / "results.json"), rel(CARDS), rel(E1 / "real_dev.json"),
            "src/cryoml/explore/i3_2_bias.py", "scripts/research/explore/i3_2_bias.py",
            "scripts/research/explore/i2_1_joint.py"]


# ================================================================== 2 decision table

def decision_values(res: dict, dec: dict | None) -> dict:
    real = res["real"]
    s, cmp_ = real["summary"], real["comparisons"]
    r1 = cmp_["M_bias_vs_C0_refit"]
    out = {"R1": (f"R = {r1['ratio']:.3f}, CI [{r1['ci_low']:.3f}, {r1['ci_high']:.3f}]",
                  r1["ci_high"] < 1.0),
           "R2": (f"{s['M_bias']['all_device_mean']:.5f} <= {s['T1_inloop']['all_device_mean']:.5f}"
                  f" + 0.005 = {s['T1_inloop']['all_device_mean'] + T1_PLUS:.5f}",
                  s["M_bias"]["all_device_mean"] <= s["T1_inloop"]["all_device_mean"] + T1_PLUS),
           "R3": (f"{s['M_bias']['n_failed']}/12 failed", s["M_bias"]["n_failed"] == 0)}
    for k, key in (("R1", "R1_ci_high_lt_1"), ("R2", "R2_mean_le_T1_inloop_plus_0.005"),
                   ("R3", "R3_zero_failed")):
        assert out[k][1] == real["criterion_real_part"][key]
    parts, any_pass, pending = [], False, False
    for pdk in ("gf180", "ihp"):
        r = res.get(pdk)
        if r is None:
            parts.append(f"{pdk.upper()}: pending")
            pending = True
            continue
        c = r["comparisons"]["M_bias_vs_C0_refit"]
        nf = r["summary"]["M_bias"]["n_failed"]
        ok = c["ratio"] <= PDK_RATIO and nf == 0
        assert ok == r["criterion_R4_part"]["pass_this_pdk"]
        any_pass |= ok
        parts.append(f"{pdk.upper()} R = {c['ratio']:.3f}, {nf} failed ({'pass' if ok else 'fail'})")
    out["R4"] = ("; ".join(parts), True if any_pass else (None if pending else False))
    if dec is not None:
        ch = dec["checks"]
        assert out["R1"][1] == ch["R1_real_ci_high_lt_1"]
        assert out["R2"][1] == ch["R2_real_mean_le_T1_inloop_plus_0.005"]
        assert out["R3"][1] == ch["R3_real_zero_failed"]
        assert out["R4"][1] == ch["R4_sim_ratio_le_0.90_on_gf180_or_ihp"]
    return out


def table_decision(res: dict, dec: dict | None) -> rf.Table:
    v = decision_values(res, dec)
    text = {"R1": "R1 (REAL_DEV): upper 95% CI bound of R(M_bias, C0_refit) < 1",
            "R2": "R2 (REAL_DEV): M_bias all-device mean <= T1_inloop mean + 0.005",
            "R3": "R3 (REAL_DEV): 0 failed devices for M_bias",
            "R4": "R4 (simulated): R(M_bias, C0_refit) <= 0.90 and 0 failed on GF180 or IHP"}
    rows = [{"c": text[k], "v": v[k][0], "ok": "pending" if v[k][1] is None else yn(v[k][1])}
            for k in ("R1", "R2", "R3", "R4")]
    if dec is not None:
        verdict = dec["decision"].upper()
        assert verdict == ("CONTINUE" if all(x[1] for x in v.values()) else "STOP")
        rows.append({"c": "rule output (decision.json, I3_2_decide)", "v": "", "ok": verdict})
    else:
        rows.append({"c": "rule output", "v": "decision.json not written yet", "ok": "pending"})
    cols = [{"key": "c", "head": "pre-registered condition", "md": "condition", "kind": "text"},
            {"key": "v", "head": "value", "md": "value", "kind": "rtext"},
            {"key": "ok", "head": "met", "md": "met", "kind": "rtext"}]
    notes = [
        "Pre-registration: out/research/explore/I3_2/prereg.json (commit f7f6378, written before any "
        "M_bias or M_bias_geom result). CONTINUE only if R1-R4 all hold; STOP otherwise.",
        "R = exp(mean_i ln(RRMS_M_bias,i / RRMS_C0_refit,i)); bootstrap 95% CI = 2.5/97.5 percentiles "
        "over 10000 resamples, numpy default_rng(0); unit: device (REAL_DEV, 12), geometry cluster "
        "(GF180/IHP, 16 clusters of 3 targets).",
        "T1_inloop on REAL_DEV = stored E1 stage-2 T1 rows (0.08198); reproduced by the rerun with "
        "max |difference| 0. C0_refit = stored I2-1 rows; reproduced with max |difference| 0.",
        "Not used for the decision: M_bias_geom, T1_inloop on GF180/IHP, I2-1 T1_joint, reference "
        "rows, post-hoc diagnostics, reruns, REAL_TEST.",
        "Values recomputed from the stored per-device RRMS and checked against criterion_real_part, "
        "criterion_R4_part and decision.json (when present).",
    ]
    return rf.Table("02_table_i3_2_decision",
                    "I3-2 decision: pre-registered conditions R1-R4, values and rule output",
                    cols, rows, notes)


# ================================================================== 3 results tables

REAL_ROWS = [
    ("C0_refit", "control (stored I2-1 rows)"),
    ("M_bias", "primary"),
    ("M_bias_geom", "secondary"),
    ("T1_inloop", "comparator, not deployable (stored E1 stage 2)"),
    ("ref_M_post_E1_stage2", "reference: E1 M (6 inputs) post-hoc, refit"),
    ("ref_M_joint_I2_1", "reference: I2-1 M_joint, refit"),
    ("ref_T1_joint_I2_1", "reference: I2-1 T1_joint, refit"),
    ("C0_refit_rerun", "check: C0_refit rerun"),
    ("T1_inloop_rerun", "check: T1_inloop rerun"),
    ("posthoc:C0_start", "diagnostic: starting card, no refit, no g"),
    ("posthoc:M_bias", "diagnostic: starting card x exp(g), no refit"),
    ("posthoc:M_bias_geom", "diagnostic: starting card x exp(g), no refit"),
    ("published", "reference: published 77 K card (released)"),
]


def table_real(real: dict, pub: dict) -> tuple[rf.Table, dict]:
    per = dict(real["per_device"])
    for k, v in real["posthoc_no_refit"].items():
        per[f"posthoc:{k}"] = v["per_device"]
    per["published"] = pub
    rows, gm = [], {}
    for key, role in REAL_ROWS:
        v = per[key]
        f = fam(v)
        st = real["summary"].get(key) or (real["posthoc_no_refit"].get(key.split(":")[-1], {})
                                          .get("summary") if key.startswith("posthoc:") else None)
        if st:
            assert abs(f["all"] - st["all_device_mean"]) < 1e-12, key
            assert abs(f["combined"] - st["combined"]) < 1e-12, key
            assert abs(f["median"] - st["median"]) < 1e-12, key
        nf = sum(not np.isfinite(x) for x in v.values())
        row = {"arm": key, "role": role, "all": f["all"], "combined": f["combined"],
               "nmos": f["nmos"], "pmos": f["pmos"], "median": f["median"],
               "failed": f"{nf}/{len(v)}"}
        if key != "C0_refit":
            c = gmr(per, key, "C0_refit", True)
            check_cmp(real, f"{key}_vs_C0_refit", c)
            gm[key] = c
            row["gmr"] = f"{c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"
            row["wins"] = f"{c['wins']}/{c['n_used']}"
        rows.append(row)
    cols = [
        {"key": "arm", "head": "arm", "md": "arm", "kind": "text"},
        {"key": "role", "head": "role", "md": "role", "kind": "text"},
        {"key": "all", "head": "all-device", "group": "RRMS, 12 REAL_DEV devices", "md": "all-device", "kind": "rrms"},
        {"key": "combined", "head": "combined", "group": "RRMS, 12 REAL_DEV devices", "md": "combined", "kind": "rrms"},
        {"key": "nmos", "head": "nMOS", "group": "RRMS, 12 REAL_DEV devices", "md": "nMOS", "kind": "rrms", "family": True},
        {"key": "pmos", "head": "pMOS", "group": "RRMS, 12 REAL_DEV devices", "md": "pMOS", "kind": "rrms", "family": True},
        {"key": "median", "head": "median", "group": "RRMS, 12 REAL_DEV devices", "md": "median", "kind": "rrms"},
        {"key": "failed", "head": "failed", "md": "failed", "kind": "rtext"},
        {"key": "gmr", "head": "R vs C0_refit [95% CI]", "md": "paired GMR vs C0_refit [95% CI]", "kind": "rtext"},
        {"key": "wins", "head": "< C0_refit", "md": "devices below C0_refit", "kind": "rtext"},
    ]
    notes = [
        "Strict paper RRMS with frozen published-baseline curve inclusion, 12 REAL_DEV devices (5 nMOS, "
        "7 pMOS); all-device = mean over 12; combined = mean of the nMOS and pMOS family means.",
        "Refit rows: held-out card refit from the direct43_r3_fd starting card with the frozen "
        "correction in the loop (max_nfev 400, irls 2, diff_step 0.01), best card re-simulated and "
        "multiplied by exp(g). Corrections trained leave-one-device-out on the other 11 devices.",
        "R vs C0_refit: paired geometric-mean ratio with device-bootstrap 95% CI (10000 resamples, "
        "default_rng(0)); recomputed here and checked against the stored comparisons where stored.",
        "published: NGSpice curves of the published (upstream released) 77 K card from the upstream "
        "committed sweeps (data/raw/CryoPDK_Skywater130nm_ML, commit 39b1e51; "
        "scripts/verify_simulator.py reproduces them), scored here with the same strict scorer and "
        "checked against out/research/reference_scores.json (released, |diff| < 1e-8); no new "
        "simulation.",
        "posthoc rows: the fold correction applied to the stored starting-card simulation without a "
        "refit (diagnostic). M_bias vs T1_inloop: R = "
        f"{real['comparisons']['M_bias_vs_T1_inloop']['ratio']:.3f} "
        f"[{real['comparisons']['M_bias_vs_T1_inloop']['ci_low']:.3f}, "
        f"{real['comparisons']['M_bias_vs_T1_inloop']['ci_high']:.3f}], "
        f"{real['comparisons']['M_bias_vs_T1_inloop']['wins']}/12; M_bias_geom vs M_bias: R = "
        f"{real['comparisons']['M_bias_geom_vs_M_bias']['ratio']:.3f} "
        f"[{real['comparisons']['M_bias_geom_vs_M_bias']['ci_low']:.3f}, "
        f"{real['comparisons']['M_bias_geom_vs_M_bias']['ci_high']:.3f}], "
        f"{real['comparisons']['M_bias_geom_vs_M_bias']['wins']}/12.",
    ]
    return rf.Table("03_table_i3_2_real_results",
                    "I3-2 REAL_DEV leave-one-device-out: held-out RRMS per arm, reference rows and "
                    "diagnostics", cols, rows, notes), gm


SIM_ROWS = [
    ("C0_refit", "control (stored I2-1 rows)"),
    ("M_bias", "primary (R4)"),
    ("M_bias_geom", "secondary"),
    ("T1_inloop", "comparator, not deployable (new run)"),
    ("ref_M_post_I2_1", "reference: I2-1 M_post, refit"),
    ("ref_M_joint_I2_1", "reference: I2-1 M_joint, refit"),
    ("C0_refit_rerun", "check: C0_refit rerun"),
    ("posthoc:C0_start", "diagnostic: starting card, no refit"),
    ("posthoc:M_bias", "diagnostic: starting card x exp(g), no refit"),
    ("posthoc:M_bias_geom", "diagnostic: starting card x exp(g), no refit"),
]


def table_sim(res: dict) -> rf.Table:
    rows = []
    for pdk in ("gf180", "ihp"):
        r = res.get(pdk)
        if r is None:
            rows.append({"pdk": PDK_LABEL[pdk], "arm": "pending", "role":
                         "I3_2_ihp did not finish before this batch was written"})
            continue
        per = dict(r["per_device"])
        for k, v in r["posthoc_no_refit"].items():
            per[f"posthoc:{k}"] = v["per_device"]
        for key, role in SIM_ROWS:
            if key not in per:
                continue
            v = per[key]
            vals = np.array(list(v.values()), dtype=float)
            ok = np.isfinite(vals)
            st = r["summary"].get(key) or (r["posthoc_no_refit"].get(key.split(":")[-1], {})
                                           .get("summary") if key.startswith("posthoc:") else None)
            mean, med = float(vals[ok].mean()), float(np.median(vals[ok]))
            if st:
                assert abs(mean - st["valid_only_mean"]) < 1e-12, (pdk, key)
                assert abs(med - st["median"]) < 1e-12, (pdk, key)
            row = {"pdk": PDK_LABEL[pdk], "arm": key, "role": role, "mean": mean, "median": med,
                   "failed": f"{int((~ok).sum())}/{len(vals)}"}
            if key != "C0_refit":
                c = gmr(per, key, "C0_refit", False)
                check_cmp(r, f"{key}_vs_C0_refit", c)
                row["gmr"] = f"{c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"
                row["wins"] = f"{c['wins']}/{c['n_used']}"
            rows.append(row)
    cols = [
        {"key": "pdk", "head": "PDK", "md": "PDK", "kind": "text"},
        {"key": "arm", "head": "arm", "md": "arm", "kind": "text"},
        {"key": "role", "head": "role", "md": "role", "kind": "text"},
        {"key": "mean", "head": "all-target", "group": "RRMS, 48 simulated targets", "md": "all-target mean", "kind": "rrms"},
        {"key": "median", "head": "median", "group": "RRMS, 48 simulated targets", "md": "median", "kind": "rrms"},
        {"key": "failed", "head": "failed", "md": "failed", "kind": "rtext"},
        {"key": "gmr", "head": "R vs C0_refit [95% CI]", "md": "paired GMR vs C0_refit [95% CI]", "kind": "rtext"},
        {"key": "wins", "head": "< C0_refit", "md": "targets below C0_refit", "kind": "rtext"},
    ]
    notes = [
        "Simulated tier-c truth sets 0-2 (16 geometries x 3 draws = 48 targets per PDK) with injected "
        "model-form misspecification; leave-one-geometry-out (the 3 draws of the held-out geometry "
        "are removed from training). Starting cards: E1 fitted cards (wide box: GF180 21 parameters, "
        "IHP 12). VDD 3.3 V (GF180), 1.2 V (IHP).",
        "R vs C0_refit: paired geometric-mean ratio, 95% CI from 10000 geometry-cluster resamples "
        "(default_rng(0)). R4 uses the M_bias row: R <= 0.90 and 0 failed.",
        "all-target mean equals the combined value here (the stored summary reports the same number).",
    ]
    return rf.Table("04_table_i3_2_sim_results",
                    "I3-2 simulated GF180 and IHP tier c, leave-one-geometry-out: held-out RRMS per arm",
                    cols, rows, notes)


# ================================================================== 4 per-device

def fig_per_device(real: dict, path: Path) -> list[str]:
    tags = real_dev()
    per = real["per_device"]
    arms = ["C0_refit", "T1_inloop", "M_bias", "M_bias_geom"]
    fig = plt.figure(figsize=(13, 10), dpi=DPI)
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 1])
    ax = fig.add_subplot(gs[0])
    style(ax)
    x = np.arange(len(tags))
    off = dict(zip(arms, np.linspace(-0.24, 0.24, len(arms))))
    for a in arms:
        v = per[a]
        f = fam(v)
        extra = "" if a == "C0_refit" else \
            f", < C0_refit on {sum(v[t] < per['C0_refit'][t] for t in tags)}/12"
        ax.scatter(x + off[a], [v[t] for t in tags], marker=ARM_MARK[a], s=40, color=ARM_COLOR[a],
                   edgecolor=BG, lw=0.5, zorder=3,
                   label=f"{ARM_LABEL[a]}: all-device {f['all']:.4f}, combined {f['combined']:.4f}{extra}")
    for i, t in enumerate(tags):
        ax.plot([x[i] + off["C0_refit"], x[i] + off["M_bias"]], [per["C0_refit"][t], per["M_bias"][t]],
                color=GREY, lw=0.7, zorder=2)
    ax.set_yscale("log")
    nm = sum(t.startswith("nmos") for t in tags)
    ax.axvline(nm - 0.5, color=GREY, lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([rf.device_label(t) + (" n" if t.startswith("nmos") else " p") for t in tags],
                       rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("held-out paper RRMS (log scale)", fontsize=9.5)
    ax.legend(fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2,
              title="per device and arm; grey segment joins C0_refit and M_bias (n = nMOS, p = pMOS; "
                    "0 failed devices)", title_fontsize=8.5)
    ax = fig.add_subplot(gs[1])
    style(ax)
    for a in ("T1_inloop", "M_bias", "M_bias_geom"):
        r = np.array([per[a][t] / per["C0_refit"][t] for t in tags])
        c = real["comparisons"][f"{a}_vs_C0_refit"]
        ax.scatter(x + off[a], r, marker=ARM_MARK[a], s=40, color=ARM_COLOR[a], edgecolor=BG, lw=0.5,
                   zorder=3, label=f"{a} / C0_refit: R = {c['ratio']:.3f} [{c['ci_low']:.3f}, "
                                   f"{c['ci_high']:.3f}], {c['wins']}/12 below 1")
    ax.axhline(1.0, color=INK, lw=0.9, ls="--")
    ax.set_yscale("log")
    ax.axvline(nm - 0.5, color=GREY, lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([rf.device_label(t) + (" n" if t.startswith("nmos") else " p") for t in tags],
                       rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("RRMS ratio to C0_refit (log scale)", fontsize=9.5)
    ax.legend(fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3)
    fig.suptitle("I3-2, REAL_DEV leave-one-device-out: held-out RRMS after the refit with the frozen "
                 "correction (top) and per-device ratio to C0_refit (bottom; below 1 = lower RRMS)",
                 fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(RUNS["real"] / "results.json")]


# ================================================================== 5 g surface

def fig_g_surface(real: dict, recs: dict, path: Path) -> list[str]:
    corrs = load(RUNS["real"] / "corrections.json")
    full = correction_from_json(corrs["full|M_bias"])
    vdd = 1.85
    n = 181
    a = np.linspace(0, 1, n)
    A, B = np.meshgrid(a, a)        # A = |Vgs|/VDD (x), B = |Vds|/VDD (y)
    fig = plt.figure(figsize=(18, 6.2), dpi=DPI)
    fig.patch.set_facecolor(BG)
    gsp = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.045, 0.2, 0.95], wspace=0.22)
    axes = [fig.add_subplot(gsp[0, 0]), fig.add_subplot(gsp[0, 1]), fig.add_subplot(gsp[0, 4])]
    cax = fig.add_subplot(gsp[0, 2])
    grids = {}
    for p in (0, 1):
        phi = np.column_stack([np.full(A.size, float(p)), A.ravel(), B.ravel(),
                               np.zeros(A.size), np.zeros(A.size), np.zeros(A.size)])
        grids[p] = full.g(phi).reshape(A.shape)
    lim = max(abs(grids[0]).max(), abs(grids[1]).max())
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    for p, ax in zip((0, 1), axes[:2]):
        style(ax)
        im = ax.pcolormesh(A * vdd, B * vdd, grids[p], cmap="RdBu_r", norm=norm, shading="auto")
        lv = [v / 10 for v in range(-10, 11) if v != 0]
        cs = ax.contour(A * vdd, B * vdd, grids[p], levels=lv, colors=INK, linewidths=0.5)
        ax.clabel(cs, fontsize=6.5, fmt="%.1f")
        c0 = ax.contour(A * vdd, B * vdd, grids[p], levels=[0.0], colors=INK, linewidths=1.0,
                        linestyles="-")
        ax.clabel(c0, fontsize=6.5, fmt={0.0: "0"})
        fam_ = "pmos" if p else "nmos"
        first = True
        for rec in recs.values():
            if rec.family != fam_:
                continue
            for c in rec.curves:
                if not c.included:
                    continue
                s = slice(c.start, None)
                ax.scatter(np.abs(c.vg[s]), np.abs(c.vd[s]), s=1.2, color=INK, alpha=0.35,
                           lw=0, zorder=3, label="measured bias points used in training "
                           "(included curves, after trim)" if first else None)
                first = False
        ax.set_xlim(0, vdd)
        ax.set_ylim(0, vdd)
        ax.set_xlabel("|Vgs| (V)")
        ax.set_ylabel("|Vds| (V)")
        nd = sum(r.family == fam_ for r in recs.values())
        ax.set_title(f"{'pMOS' if p else 'nMOS'} (is_pmos = {p}), {nd} devices: g range "
                     f"[{grids[p].min():.2f}, {grids[p].max():.2f}]", fontsize=9.5)
        ax.legend(fontsize=7.2, loc="upper left", frameon=True, facecolor=BG, edgecolor=GREY,
                  markerscale=5)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("g = log of the current multiplier (exp(g))", fontsize=9)
    # slices of exp(g) along |Vgs| at fixed |Vds|
    ax = axes[2]
    style(ax)
    for p, ls in ((0, "-"), (1, "--")):
        for vd, col in zip((0.01, 0.37, 1.11, 1.85), (AQUA, BLUE, ORANGE, RED)):
            j = int(round(vd / vdd * (n - 1)))
            ax.plot(a * vdd, np.exp(grids[p][j, :]), color=col, ls=ls, lw=1.3,
                    label=f"{'pMOS' if p else 'nMOS'}, |Vds| = {vd:.2f} V")
    ax.axhline(1.0, color=GREY, lw=0.8)
    ax.set_xlabel("|Vgs| (V)")
    ax.set_ylabel("current multiplier exp(g)")
    ax.set_title("exp(g) along |Vgs| at the measured Id-Vg biases", fontsize=9.5)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.suptitle("Learned correction g(is_pmos, |Vgs|/VDD, |Vds|/VDD) of the full M_bias model (5 seeds, "
                 "trained on all 12 REAL_DEV devices; in-sample, not a held-out result; the model "
                 "written to m_bias_sky130_dev_full.va)", fontsize=10, color=INK)
    fig.savefig(path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return [rel(RUNS["real"] / "corrections.json") + " (entry full|M_bias)",
            rel(RUNS["real"] / "m_bias_sky130_dev_full.va"), rel(E1 / "real_dev.npz"),
            rel(E1 / "real_dev.json")]


# ================================================================== 6 I-V figures

def pick_devices(real: dict) -> list[tuple[str, str]]:
    mb = real["per_device"]["M_bias"]
    tags = real_dev()
    worst = max(tags, key=lambda t: mb[t])
    med = float(np.median([mb[t] for t in tags]))
    median_dev = min((t for t in tags if t != worst), key=lambda t: abs(mb[t] - med))
    out = [(worst, "highest M_bias RRMS of the 12"),
           (median_dev, f"M_bias RRMS closest to the 12-device median {med:.4f}")]
    if not any(t.startswith("nmos") for t, _ in out):
        nm = [t for t in tags if t.startswith("nmos")]
        nmed = float(np.median([mb[t] for t in nm]))
        out.append((min(nm, key=lambda t: abs(mb[t] - nmed)),
                    f"nMOS with M_bias RRMS closest to the nMOS median {nmed:.4f}"))
    return out


def fig_iv(real: dict, recs: dict, pub_curves: dict, pub: dict, tag: str, why: str,
           path: Path) -> list[str]:
    corrs = load(RUNS["real"] / "corrections.json")
    g_fold = correction_from_json(corrs[f"M_bias|{tag}"])
    rec = recs[tag]
    corr_sims = g_fold.apply(rec)
    per, ph = real["per_device"], real["posthoc_no_refit"]
    sc_start = score_device(rec, [c.sim for c in rec.curves])["rrms"]
    sc_post = score_device(rec, corr_sims)["rrms"]
    assert abs(sc_start - ph["C0_start"]["per_device"][tag]) < 1e-12
    assert abs(sc_post - ph["M_bias"]["per_device"][tag]) < 1e-12
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for ax, kind in zip(axes, ("idvd", "idvg")):
        style(ax)
        for i, c in enumerate(rec.curves):
            if c.kind != kind:
                continue
            xv = np.abs(c.vd if kind == "idvd" else c.vg)
            s = slice(c.start, None)
            alpha = 1.0 if c.included else 0.35
            meas = np.abs(c.target)
            ax.scatter(xv[s], meas[s], s=7, color=INK, zorder=4, alpha=alpha, lw=0)
            if c.start:
                ax.scatter(xv[:c.start], meas[:c.start], s=7, facecolor="none", edgecolor=GREY,
                           lw=0.5, zorder=4, alpha=alpha)
            ax.plot(xv, np.abs(pub_curves[tag][i]), color=GREY, ls="--", lw=1.2, alpha=alpha, zorder=2)
            ax.plot(xv, np.abs(c.sim), color=BLUE, lw=1.3, alpha=alpha, zorder=3)
            ax.plot(xv, np.abs(corr_sims[i]), color=ORANGE, lw=1.3, alpha=alpha, zorder=3)
            lab = c.key.split("@")[1]
            ax.annotate(f"{'|Vgs|' if kind == 'idvd' else '|Vds|'} {lab}" +
                        ("" if c.included else " (excluded)"),
                        (xv[-1], max(abs(meas[-1]), 1e-14)), fontsize=6.8, color=MUTED,
                        xytext=(3, 0), textcoords="offset points", va="center")
        if kind == "idvg":
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-12)
            ax.set_xlabel("|Vgs| (V)")
            ax.set_title("Id-Vg at |Vds| = 0.01-1.85 V (log scale)", fontsize=9.5)
        else:
            ax.set_xlabel("|Vds| (V)")
            ax.set_title("Id-Vd at |Vgs| = 0.37-1.85 V", fontsize=9.5)
        ax.set_ylabel("|Id| (A)")
    handles = [
        Line2D([], [], marker="o", ls="", color=INK, ms=4,
               label="measured 77 K (filled: scored points; hollow: trimmed before the metric)"),
        Line2D([], [], color=GREY, ls="--", lw=1.2,
               label=f"NGSpice, published 77 K card (upstream released card, committed sweep): "
                     f"RRMS {pub[tag]:.4f}"),
        Line2D([], [], color=BLUE, lw=1.3,
               label=f"NGSpice, starting card direct43_r3_fd (stored E1 record): RRMS {sc_start:.4f}"),
        Line2D([], [], color=ORANGE, lw=1.3,
               label="starting-card NGSpice x exp(g), g = M_bias fold model trained on the other 11 "
                     f"devices (BSIM4 output times the learned correction, as in the Verilog-A form; "
                     f"no refit): RRMS {sc_post:.4f}"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=8.3, frameon=False,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(
        f"{tag} ({why}), held out. Stored held-out scores after the card refit: C0_refit "
        f"{per['C0_refit'][tag]:.4f}, M_bias {per['M_bias'][tag]:.4f}, T1_inloop "
        f"{per['T1_inloop'][tag]:.4f}.\nThe refit-card curves (C0_refit, M_bias) are not stored and "
        "are not plotted: they need an NGSpice run of the cards in heldout_cards.json, which this "
        "report does not do.", fontsize=9.6, color=INK)
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    dev = SimpleNamespace(dev_type=rec.family, L_um=rec.L_um, W_um=rec.W_um)
    up = upstream_metric_curves(dev)[0][0].path
    return [rel(E1 / "real_dev.npz"), rel(E1 / "real_dev.json"),
            rel(RUNS["real"] / "corrections.json") + f" (entry M_bias|{tag})",
            rel(RUNS["real"] / "results.json"), rel(Path(up).parent.parent) + "/{vd,vg}_sweep/*.txt"]


# ================================================================== 7 caveats

def table_caveats(real: dict, res: dict, pub: dict) -> rf.Table:
    per = real["per_device"]
    mb = per["M_bias"]
    worst = max(mb, key=mb.get)
    share = mb[worst] / sum(mb.values())
    wo = {t: v for t, v in mb.items() if t != worst}
    c0wo = {t: v for t, v in per["C0_refit"].items() if t != worst}
    t1 = real["comparisons"]["M_bias_vs_T1_inloop"]
    rows = [
        {"i": "REAL_DEV reuse", "d": "REAL_DEV was used for the design and the decision of E1 and "
         "I2-1, and the I3-2 hypothesis was formed from the REAL_DEV T1 and M results. The I3-2 "
         "REAL_DEV values are development evidence, not a held-out test."},
        {"i": "Input choice after I2-1", "d": "The 3-input set (no geometry, no simulated current) "
         "was chosen after the E1 and I2-1 REAL_DEV results were seen. The pre-registration fixed it "
         "before any M_bias result existed, but the choice is not independent of REAL_DEV."},
        {"i": "M_bias vs T1_inloop", "d": f"M_bias / T1_inloop R = {t1['ratio']:.3f} "
         f"[{t1['ci_low']:.3f}, {t1['ci_high']:.3f}], {t1['wins']}/12 devices lower; the CI includes "
         "1. R2 tests M_bias <= T1 + 0.005 (non-inferiority), not M_bias < T1."},
        {"i": "Verilog-A not compiled", "d": "m_bias_sky130_dev_full.va (full 12-device model, "
         f"{real['export']['lines']} lines) is text only; the emitted assignments were evaluated in "
         f"Python on 200 random bias points (max |g difference| {real['export']['max_abs_g_diff_va_vs_python']:.1e}). "
         "It has not been compiled (OpenVAF / ngspice OSDI) or simulated. abs() folds reverse-mode "
         "bias onto forward mode; reverse bias does not occur in any record."},
        {"i": "REAL_TEST untouched", "d": "No I3_2 script loads the 6 REAL_TEST devices. A REAL_TEST "
         "evaluation happens only at the method freeze, with the configuration fixed before it is "
         "loaded."},
        {"i": "12 devices", "d": f"5 nMOS and 7 pMOS; the bootstrap resamples 12 devices. {worst} "
         f"contributes {share:.0%} of the M_bias all-device sum (RRMS {mb[worst]:.4f}); without it "
         f"the M_bias mean is {np.mean(list(wo.values())):.4f} vs C0_refit "
         f"{np.mean(list(c0wo.values())):.4f} (diagnostic only)."},
        {"i": "Held-out refit uses measured data", "d": "The held-out card refit (box 6 of the "
         "process diagram) fits the held-out device's 43 parameters to its measured curves, as card "
         "extraction does. Only g is held out. C0_refit uses the same refit without g."},
        {"i": "Simulated R4 margin", "d": "R4 needs R <= 0.90 on one PDK. GF180: R = " +
         ci_text(res["gf180"]["comparisons"]["M_bias_vs_C0_refit"]) + ", T1_inloop lower than M_bias (R(M_bias, "
         "T1_inloop) = " + ci_text(res["gf180"]["comparisons"]["M_bias_vs_T1_inloop"]) + ")." +
         ("" if res.get("ihp") is None else
          " IHP: R = " + ci_text(res["ihp"]["comparisons"]["M_bias_vs_C0_refit"]) + " (the CI includes 1); "
          "R(M_bias, T1_inloop) = " + ci_text(res["ihp"]["comparisons"]["M_bias_vs_T1_inloop"]) + ".") +
         " Simulated devices with injected misspecification; secondary to REAL_DEV."},
        {"i": "Figures from stored data", "d": "The g surface is the in-sample full model. The I-V "
         "figures show the starting card and starting card x exp(g); the refit-card curves are not "
         "stored and were not re-simulated."},
    ]
    cols = [{"key": "i", "head": "item", "md": "item", "kind": "text"},
            {"key": "d", "head": "detail", "md": "detail", "kind": "text"}]
    t = rf.Table("10_table_i3_2_caveats", "I3-2 caveats and open checks", cols, rows, [])
    return t


def render_wrapped(t: rf.Table, path: Path, width: int = 140) -> None:
    """Long-text table: one wrapped row per item."""
    rows = []
    for r in t.rows:
        lines = textwrap.wrap(r["d"], width)
        rows.append((r["i"], lines))
    h = 0.9 + sum(0.24 * len(l) + 0.16 for _, l in rows)
    fig = plt.figure(figsize=(15, h), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 15)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.text(0.25, 0.3, t.caption, fontsize=12, fontweight="bold", color=INK, va="top")
    y = 0.85
    ax.plot([0.25, 14.75], [y - 0.05] * 2, color=INK, lw=1.0)
    for item, lines in rows:
        ax.text(0.3, y + 0.04, item, fontsize=10, fontweight="bold", color=INK, va="top")
        ax.text(3.6, y + 0.04, "\n".join(lines), fontsize=9.6, color=INK, va="top", linespacing=1.3)
        y += 0.24 * len(lines) + 0.16
        ax.plot([0.25, 14.75], [y - 0.05] * 2, color="#e4e3df", lw=0.6)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


# ================================================================== main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    topic = "i3_2_bias"
    folder = rf.REPORT_ROOT / f"{args.date}_{topic}"
    folder.mkdir(parents=True, exist_ok=True)

    res = {}
    for k, d in RUNS.items():
        p = d / "results.json"
        if p.exists():
            r = load(p)
            assert r["preregistered_config"] and not r["smoke"], k
            res[k] = r
    assert "real" in res and "gf180" in res
    dec = load(DECISION) if DECISION.exists() else None
    real = res["real"]
    recs = {r.tag: r for r in load_devices(E1 / "real_dev")}
    assert sorted(recs) == sorted(real_dev())
    pub, pub_curves = published_rrms(recs)

    items, tables = [], []
    src = fig_process(real, folder / "01_i3_2_process.png")
    items.append(("01_i3_2_process.png",
                  "Process diagram of the I3-2 REAL_DEV pipeline: measured I-V, starting 43-parameter "
                  "cards (direct NGSpice fit round 3 + FD polish), NGSpice on the measured bias grid, "
                  "leave-one-device-out training of g on 11 devices, held-out card refit with g frozen "
                  "in the loop, strict paper RRMS, aggregation and decision; box colour marks measured "
                  "data, NGSpice simulation or both", src))
    t = table_decision(res, dec)
    rf.render_table(t, folder / "02_table_i3_2_decision.png")
    tables.append(t)
    items.append(("02_table_i3_2_decision.png",
                  "Table of the pre-registered conditions R1-R4 with values, met/not met and the rule "
                  "output", [rel(PREREG)] + [rel(RUNS[k] / "results.json") for k in res] +
                  ([rel(DECISION)] if dec else [])))
    t, _ = table_real(real, pub)
    rf.render_table(t, folder / "03_table_i3_2_real_results.png")
    tables.append(t)
    items.append(("03_table_i3_2_real_results.png",
                  "Table of REAL_DEV held-out RRMS per arm (C0_refit, M_bias, M_bias_geom, T1_inloop), "
                  "reference rows (E1 M_post, I2-1 M_joint and T1_joint, published 77 K card), reruns "
                  "and post-hoc diagnostics: all-device, combined, nMOS, pMOS, median, failed, paired "
                  "geometric-mean ratio vs C0_refit with 95% CI, devices below C0_refit",
                  [rel(RUNS["real"] / "results.json"), "data/raw/CryoPDK_Skywater130nm_ML/"
                   "{nfet,pfet}_mod/*/{vd,vg}_sweep/*.txt", rel(E1 / "real_dev.json"),
                   rel(REF_SCORES)]))
    t = table_sim(res)
    rf.render_table(t, folder / "04_table_i3_2_sim_results.png")
    tables.append(t)
    items.append(("04_table_i3_2_sim_results.png",
                  "Table of GF180 and IHP tier-c held-out RRMS per arm (48 targets each, "
                  "leave-one-geometry-out): all-target mean, median, failed, paired geometric-mean "
                  "ratio vs C0_refit with geometry-cluster 95% CI, targets below C0_refit" +
                  ("" if "ihp" in res else "; IHP pending"),
                  [rel(RUNS[k] / "results.json") for k in ("gf180", "ihp") if k in res]))
    src = fig_per_device(real, folder / "05_i3_2_real_per_device.png")
    items.append(("05_i3_2_real_per_device.png",
                  "Per-device held-out RRMS on REAL_DEV for C0_refit, T1_inloop, M_bias and M_bias_geom "
                  "(top, log scale) and the per-device ratio to C0_refit (bottom, log scale)", src))
    src = fig_g_surface(real, recs, folder / "06_i3_2_g_surface.png")
    items.append(("06_i3_2_g_surface.png",
                  "Heatmaps of the learned g over (|Vgs|, |Vds|) for nMOS and pMOS from the full M_bias "
                  "model trained on all 12 REAL_DEV devices (in-sample), with the measured bias points "
                  "used in training, and exp(g) along |Vgs| at four |Vds| values", src))
    picks = pick_devices(real)
    for k, (tag, why) in enumerate(picks):
        png = f"{7 + k:02d}_i3_2_iv_{tag}.png"
        src = fig_iv(real, recs, pub_curves, pub, tag, why, folder / png)
        items.append((png, f"I-V of held-out {tag} ({why}): measured points, NGSpice of the published "
                      "(released) 77 K card, NGSpice of the starting card, and starting-card NGSpice multiplied by "
                      "exp(g) of the M_bias fold model; refit-card curves not stored and not plotted",
                      src))
    t = table_caveats(real, res, pub)
    n_cav = 7 + len(picks)
    t.name = f"{n_cav:02d}_table_i3_2_caveats"
    render_wrapped(t, folder / f"{n_cav:02d}_table_i3_2_caveats.png")
    tables.append(t)
    items.append((f"{n_cav:02d}_table_i3_2_caveats.png",
                  "Table of caveats: REAL_DEV reuse, input choice after I2-1, M_bias vs T1_inloop, "
                  "Verilog-A not compiled, REAL_TEST untouched, 12 devices, held-out refit, simulated "
                  "R4 margin, figures from stored data",
                  [rel(PREREG), rel(RUNS["real"] / "results.json"), rel(RUNS["gf180"] / "results.json")]))

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = (f"`PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i3_2.py "
           f"--date {args.date}`")
    md = [f"# Tables, {args.date} {topic}", "",
          f"Generated {now} by {gen} from stored files only. RRMS is the paper metric with frozen "
          "curve inclusion, strict.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))

    s = real["summary"]
    c = real["comparisons"]
    g = res["gf180"]
    ihp_line = "IHP: pending (queue job I3_2_ihp had not finished)."
    if "ihp" in res:
        ih = res["ihp"]
        ihp_line = (f"IHP tier c: C0_refit {ih['summary']['C0_refit']['all_device_mean']:.4f}, M_bias "
                    f"{ih['summary']['M_bias']['all_device_mean']:.4f}, T1_inloop "
                    f"{ih['summary']['T1_inloop']['all_device_mean']:.4f}, M_bias_geom "
                    f"{ih['summary']['M_bias_geom']['all_device_mean']:.4f}; R(M_bias, C0_refit) "
                    f"{ih['comparisons']['M_bias_vs_C0_refit']['ratio']:.3f} "
                    f"[{ih['comparisons']['M_bias_vs_C0_refit']['ci_low']:.3f}, "
                    f"{ih['comparisons']['M_bias_vs_C0_refit']['ci_high']:.3f}], "
                    f"{ih['comparisons']['M_bias_vs_C0_refit']['wins']}/48, "
                    f"{ih['summary']['M_bias']['n_failed']} failed.")
    verdict = dec["decision"].upper() if dec else "pending (decision.json not written)"
    readme = [f"# Report batch {args.date} · {topic}", "",
              f"Generated {now} by {gen} from stored files only (no NGSpice, no training).", "",
              "## Contents", "",
              "- I3-2: bias-only learned correction g(is_pmos, |Vgs|/VDD, |Vds|/VDD), I_corr = "
              "I_sim · exp(g), trained post-hoc on the training devices' starting-card records and used "
              "in the loop of the held-out card refit. Pre-registration "
              f"`{rel(PREREG)}` (commit f7f6378); code commit 4c7c66b. Queue jobs I3_2_real "
              "(2026-10-20 00:44-00:52), I3_2_gf180 (00:52-01:22), I3_2_ihp (01:22-01:55), I3_2_decide (01:55).",
              f"- Rule output: {verdict}. REAL_DEV values are development evidence (REAL_DEV reused "
              "by E1, I2-1 and the I3-2 design).", "",
              "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))

    r1 = c["M_bias_vs_C0_refit"]
    summary = [
        f"I3-2 (REAL_DEV leave-one-device-out, held-out card refit with the frozen correction), "
        f"all-device (combined): C0_refit {s['C0_refit']['all_device_mean']:.4f} "
        f"({s['C0_refit']['combined']:.4f}), M_bias {s['M_bias']['all_device_mean']:.4f} "
        f"({s['M_bias']['combined']:.4f}), M_bias_geom {s['M_bias_geom']['all_device_mean']:.4f} "
        f"({s['M_bias_geom']['combined']:.4f}), T1_inloop {s['T1_inloop']['all_device_mean']:.4f} "
        f"({s['T1_inloop']['combined']:.4f}); 0 failed. R(M_bias, C0_refit) {r1['ratio']:.3f} "
        f"[{r1['ci_low']:.3f}, {r1['ci_high']:.3f}], {r1['wins']}/12; R(M_bias, T1_inloop) "
        f"{c['M_bias_vs_T1_inloop']['ratio']:.3f} [{c['M_bias_vs_T1_inloop']['ci_low']:.3f}, "
        f"{c['M_bias_vs_T1_inloop']['ci_high']:.3f}].",
        f"GF180 tier c: C0_refit {g['summary']['C0_refit']['all_device_mean']:.4f}, M_bias "
        f"{g['summary']['M_bias']['all_device_mean']:.4f}, T1_inloop "
        f"{g['summary']['T1_inloop']['all_device_mean']:.4f}; R(M_bias, C0_refit) "
        f"{g['comparisons']['M_bias_vs_C0_refit']['ratio']:.3f}, "
        f"{g['comparisons']['M_bias_vs_C0_refit']['wins']}/48. " + ihp_line,
        f"Rule output (R1-R4): {verdict}. Development evidence on REAL_DEV; REAL_TEST not loaded.",
    ]
    rf.update_index(args.date, topic, folder, [i[0] for i in items], summary)
    print("\n".join(summary))
    print(json.dumps({"published_rrms_mean": float(np.mean(list(pub.values()))),
                      "published_rrms": pub, "picks": picks}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
