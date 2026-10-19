#!/usr/bin/env python3
"""Report batch for exploration E1 (stage 2 and the simulated LOGO run) and the
surrogate learning-curve study ``archlc_full_r1``.

Reads result JSON files only (no simulation, no training) and writes PNGs,
``tables.md`` and ``README.md`` to
``out/research/report/<date>_explore_e1_lc/``, then adds the batch to
``out/research/report/index.json`` and ``INDEX.md``.

  PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_e1_lc.py \
      --date 2026-10-19
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_figs as rf  # noqa: E402  (Table rendering, index update, style)
import matplotlib.pyplot as plt  # noqa: E402  (configured by report_figs)

BG, INK, ACCENT, GREY = "#fcfcfb", "#0b0b0b", "#2a78d6", "#9a9892"
ORANGE, AQUA, VIOLET, MUTED = rf.ORANGE, rf.AQUA, rf.VIOLET, "#52514e"
DPI = 150

E1 = ROOT / "out/research/explore/E1"
STAGE1 = E1 / "real_dev_lodo/results.json"
STAGE2 = E1 / "real_dev_refit/results.json"
PREREG = E1 / "prereg.json"
LOGO = {"GF180 (BSIM4)": E1 / "gf180_c_logo/results.json",
        "IHP SG13G2 (PSP103)": E1 / "ihp_c_logo/results.json"}
ARCH_DIR = ROOT / "out/research/surrogate_arch/full_round_1"
SWAP = ROOT / "out/research/swap_search/full_round_1/summary.json"
LC_ARCHS = ["paper_mlp", "deeponet", "fno"]
LC_FRACTIONS = [0.125, 0.25, 0.5, 1.0]
ARCH_COLOR = {"paper_mlp": ACCENT, "deeponet": ORANGE, "fno": AQUA}
ARM_COLOR = {"C0": GREY, "T1": ORANGE, "T2": VIOLET, "LIN": AQUA, "M": ACCENT}
ARM_LABEL = {"C0": "C0 no correction", "T1": "T1 per-tag gain (22 scalars)",
             "T2": "T2 per-kind gain (4 scalars)", "LIN": "LIN quadratic-linear",
             "M": "M MLP correction (primary)"}


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(ROOT))


def load(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def style(ax) -> None:
    ax.set_facecolor(BG)
    ax.grid(True, which="major", color="#e4e3df", lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def short(tag: str) -> str:
    return rf.device_label(tag) if hasattr(rf, "device_label") else tag


# ------------------------------------------------------------ E1 real devices

def fig_stage2_per_device(s2: dict, path: Path) -> list[str]:
    rows = s2["rows"]
    tags = sorted(rows["M"], key=rf.device_key)
    x = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    vals = {a: np.array([rows[a][t]["rrms"] for t in tags]) for a in ("C0", "M", "T1")}
    for i in range(len(tags)):
        ax.plot([i - 0.2, i, i + 0.2], [vals["C0"][i], vals["M"][i], vals["T1"][i]],
                color="#d6d5d0", lw=0.9, zorder=1)
    for a, dx, mk, lab in (("C0", -0.2, "o", "C0 refit, no correction"),
                           ("M", 0.0, "D", "M refit, MLP correction in the loop"),
                           ("T1", 0.2, "s", "T1 refit, per-tag gain in the loop")):
        s = s2["summary"][a]
        ax.scatter(x + dx, vals[a], color=ARM_COLOR[a], marker=mk, s=34, zorder=3,
                   label=f"{lab}: all-device {s['all_device_mean']:.4f}, "
                         f"combined {s['combined']:.4f}")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([short(t) for t in tags], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("held-out device paper RRMS (strict, frozen inclusion)", fontsize=9, color=INK)
    nmos = sum(t.startswith("nmos") for t in tags)
    ax.axvline(nmos - 0.5, color=GREY, lw=0.8, ls=":")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    m_t1 = int(sum(vals["M"] < vals["T1"]))
    m_c0 = int(sum(vals["M"] < vals["C0"]))
    ax.set_title(f"E1 stage 2, REAL_DEV leave-one-device-out, card re-polished with the "
                 f"fold's correction in the residual: M < T1 on {m_t1}/12, M < C0 refit "
                 f"on {m_c0}/12", fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(STAGE2)]


def fig_stage_bars(s1: dict, s2: dict, path: Path) -> list[str]:
    arms = ["C0", "T1", "M"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), dpi=DPI, sharey=True)
    fig.patch.set_facecolor(BG)
    for ax, key, lab in ((axes[0], "all_device_mean", "all-device mean (12 devices)"),
                         (axes[1], "combined", "combined (mean of nMOS and pMOS means)")):
        style(ax)
        w = 0.38
        for j, (res, stage, hatch) in enumerate(((s1, "stage 1: post-hoc, fixed card", ""),
                                                 (s2, "stage 2: card refit, correction in loop", "//"))):
            v = [res["summary"][a][key] for a in arms]
            bars = ax.bar(np.arange(3) + (j - 0.5) * w, v, w,
                          color=[ARM_COLOR[a] for a in arms], alpha=1.0 if j else 0.55,
                          edgecolor=INK if j else "none", lw=0.6, hatch=hatch, label=stage)
            for b, val in zip(bars, v):
                ax.text(b.get_x() + b.get_width() / 2, val + 0.002, f"{val:.4f}",
                        ha="center", va="bottom", fontsize=7, color=INK)
        if key == "all_device_mean":
            ax.axhline(0.090, color=MUTED, lw=0.8, ls="--",
                       label="pre-registered M threshold 0.090 (all-device)")
        ax.set_ylim(0, 0.165)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["C0 no correction", "T1 per-tag gain", "M MLP correction"],
                           fontsize=8)
        ax.set_title(lab, fontsize=9, color=INK)
    axes[0].set_ylabel("paper RRMS (strict, frozen inclusion)", fontsize=9, color=INK)
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle("E1 on REAL_DEV (12 devices, leave-one-device-out): stage 1 vs stage 2",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(STAGE1), rel(STAGE2)]


def table_stages(s1: dict, s2: dict) -> rf.Table:
    tags = list(s2["rows"]["M"])
    r2 = {a: {t: s2["rows"][a][t]["rrms"] for t in tags} for a in ("C0", "M", "T1")}
    p1 = s1["per_device"]
    wins = {
        ("stage 1", "C0"): None, ("stage 1", "T1"): None,
        ("stage 1", "M"): (s1["wins"]["M_beats_C0"], s1["wins"]["M_beats_T1"]),
        ("stage 2", "C0"): None, ("stage 2", "T1"): None,
        ("stage 2", "M"): (sum(r2["M"][t] < r2["C0"][t] for t in tags),
                           sum(r2["M"][t] < r2["T1"][t] for t in tags)),
    }
    assert wins[("stage 1", "M")][0] == sum(p1["M"][t] < p1["C0"][t] for t in tags)
    rows = []
    for stage, res, verdict in (("stage 1", s1, s1["criterion"]["stage1_pass"]),
                                ("stage 2", s2, s2["criterion"]["stage2_pass"])):
        for a in ("C0", "T1", "M"):
            s = res["summary"][a]
            w = wins[(stage, a)]
            c0name = ("C0 fixed card" if stage == "stage 1" else "C0 refit")
            rows.append({
                "stage": stage,
                "arm": {"C0": c0name, "T1": "T1 per-tag gain", "M": "M MLP correction"}[a],
                "all": s["all_device_mean"], "combined": s["combined"], "median": s["median"],
                "failed": s["n_failed"],
                "wc0": "—" if w is None else f"{w[0]}/12",
                "wt1": "—" if w is None else f"{w[1]}/12",
                "pass": ("pass" if verdict else "fail") if a == "M" else "—",
            })
    cols = [
        {"key": "stage", "head": "stage", "md": "stage", "kind": "text"},
        {"key": "arm", "head": "arm", "md": "arm", "kind": "text"},
        {"key": "all", "head": "all-device", "group": "paper RRMS, 12 devices", "md": "all-device", "kind": "rrms"},
        {"key": "combined", "head": "combined", "group": "paper RRMS, 12 devices", "md": "combined", "kind": "rrms"},
        {"key": "median", "head": "median", "group": "paper RRMS, 12 devices", "md": "median", "kind": "rrms"},
        {"key": "failed", "head": "failed", "md": "failed devices", "kind": "int"},
        {"key": "wc0", "head": "M < C0", "group": "devices", "md": "M < C0 (devices)", "kind": "rtext"},
        {"key": "wt1", "head": "M < T1", "group": "devices", "md": "M < T1 (devices)", "kind": "rtext"},
        {"key": "pass", "head": "5 checks", "md": "pre-registered rule", "kind": "rtext"},
    ]
    c1, c2 = s1["criterion"]["checks"], s2["criterion"]["checks"]
    yn = lambda b: "yes" if b else "no"  # noqa: E731
    notes = [
        "Stage 1 (post-hoc): each arm is trained on 11 devices and applied once to the held-out "
        "device's fixed best 43-parameter card (direct NGSpice round 3 + FD). C0 = that card "
        "with no correction.",
        "Stage 2 (refit): the held-out card is re-polished (ngspice_fit.polish, TRF max_nfev 400, "
        "irls 2, diff_step 0.01, round-3 box) against the measured curves with the fold's "
        "correction inside the residual. C0 = the same polish without correction from the same "
        "start; T1 is re-polished with T1 in the loop.",
        "Pre-registered continue rule (all five): M all-device <= 0.090; M < T1 all-device; M < T1 "
        "on >= 8/12; M < C0 on >= 8/12; 0 failed devices for M. "
        f"Stage 1: {yn(c1['M_mean_le_0.090'])}, {yn(c1['M_mean_lt_T1_mean'])}, "
        f"{yn(c1['M_beats_T1_on_ge_8_of_12'])}, {yn(c1['M_beats_C0_on_ge_8_of_12'])}, "
        f"{yn(c1['M_zero_failed'])}. Stage 2: {yn(c2['M_mean_le_0.090'])}, "
        f"{yn(c2['M_mean_lt_T1_mean'])}, {yn(c2['M_beats_T1_on_ge_8_of_12'])}, "
        f"{yn(c2['M_beats_C0refit_on_ge_8_of_12'])}, {yn(c2['M_zero_failed'])}.",
        "all-device = mean over the 12 REAL_DEV devices (5 nMOS, 7 pMOS); combined = mean of the "
        "nMOS and pMOS family means. T1 is a table indexed by curve tag and is not a function of "
        "terminal voltages, so it cannot be written as a Verilog-A device model; M can.",
    ]
    return rf.Table("02_table_e1_stages",
                    "E1 on REAL_DEV, leave-one-device-out: stage 1 (post-hoc correction on the fixed "
                    "card) vs stage 2 (card refit with the correction in the loop)", cols, rows, notes)


# ------------------------------------------------------------ E1 simulated LOGO

def fig_logo(logo: dict, path: Path) -> list[str]:
    arms = ["C0", "T1", "T2", "LIN", "M"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for ax, (pdk, res) in zip(axes, logo.items()):
        style(ax)
        v = [res["summary"][a]["all_device_mean"] for a in arms]
        bars = ax.bar(range(len(arms)), v, 0.6, color=[ARM_COLOR[a] for a in arms])
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.4f}", ha="center",
                    va="bottom", fontsize=7.5, color=INK)
        c0 = res["summary"]["C0"]["all_device_mean"]
        ax.axhline(0.9 * c0, color=MUTED, lw=0.8, ls="--", label="0.90 × C0 (criterion_S)")
        ax.set_ylim(0, max(v) * 1.25)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(arms, fontsize=8)
        ax.set_title(f"{pdk}: {res['n_records']} targets, {res['n_folds']} geometry folds; "
                     f"criterion_S {'pass' if res['criterion_S']['pass'] else 'fail'}",
                     fontsize=8.5, color=INK)
    axes[0].set_ylabel("all-device mean paper RRMS (strict)", fontsize=9, color=INK)
    fig.suptitle("E1 simulated benchmark, tier c sets 0-2, leave-one-geometry-out, post-hoc "
                 "correction on the fitted card (no in-loop refit)", fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(p) for p in LOGO.values()]


def table_logo(logo: dict) -> rf.Table:
    rows = []
    for pdk, res in logo.items():
        S = res["summary"]
        w = res["wins"]
        n = res["n_records"]
        for a in ("C0", "T1", "T2", "LIN", "M"):
            fam = family_means(res["per_device"][a])
            rows.append({
                "pdk": pdk, "arm": ARM_LABEL[a], "all": S[a]["all_device_mean"],
                "combined": S[a]["combined"], "nmos": fam["nmos"], "pmos": fam["pmos"],
                "median": S[a]["median"], "failed": S[a]["n_failed"],
                "wins": (f"{w['M_beats_' + a]}/{n}" if a != "M" else "—"),
                "ratio": S[a]["all_device_mean"] / S["C0"]["all_device_mean"],
            })
    cols = [
        {"key": "pdk", "head": "PDK", "md": "PDK", "kind": "text"},
        {"key": "arm", "head": "arm", "md": "arm", "kind": "text"},
        {"key": "all", "head": "all-device", "group": "paper RRMS, 48 targets", "md": "all-device", "kind": "rrms"},
        {"key": "combined", "head": "combined", "group": "paper RRMS, 48 targets", "md": "combined", "kind": "rrms"},
        {"key": "nmos", "head": "nMOS", "group": "family mean", "md": "nMOS", "kind": "rrms", "family": True},
        {"key": "pmos", "head": "pMOS", "group": "family mean", "md": "pMOS", "kind": "rrms", "family": True},
        {"key": "median", "head": "median", "md": "median", "kind": "rrms"},
        {"key": "ratio", "head": "arm / C0", "md": "all-device / C0", "kind": "rrms"},
        {"key": "failed", "head": "failed", "md": "failed targets", "kind": "int"},
        {"key": "wins", "head": "M < arm", "md": "targets with M < arm", "kind": "rtext"},
    ]
    notes = [
        "Targets: GF180 (BSIM4) and IHP SG13G2 (PSP103) tier c truth sets 0, 1 and 2, 16 "
        "geometries per PDK (8 nMOS, 8 pMOS), 48 targets per PDK. Fit per target: "
        "ngspice_fit.polish from the wide-set projection (max_nfev 300, irls 3).",
        "Folds: leave-one-geometry-out (all three draws of the held-out geometry removed from "
        "training), 16 folds per PDK. Arms use the pre-registered hyperparameters (M: 5 seeds, "
        "2000 steps).",
        "Correction applied post hoc to the fitted card's simulation; the card is not refit with "
        "the correction in the loop. This is the setting (stage 1) in which M also failed on the "
        "measured devices.",
        "criterion_S (per PDK): M all-device <= 0.90 × C0 and M < T1. Secondary; not used for "
        "the E1 decision. It fails on both PDKs.",
        "all-device = mean over 48 targets; combined = mean of the nMOS and pMOS family means "
        "(equal family counts, so it equals the all-device mean).",
    ]
    return rf.Table("04_table_e1_sim_logo",
                    "E1 simulated benchmark, tier c sets 0-2, leave-one-geometry-out, post-hoc "
                    "correction", cols, rows, notes)


def family_means(per_device: dict) -> dict:
    out = {}
    for fam in ("nmos", "pmos"):
        v = [x for k, x in per_device.items() if f"_{fam}_" in k or k.startswith(fam)]
        out[fam] = float(np.mean(v)) if v else None
    return out


# ------------------------------------------------------------ learning curve

def lc_records() -> dict:
    out = {}
    for arch in LC_ARCHS:
        for f in LC_FRACTIONS:
            p = ARCH_DIR / f"{arch}_f{f:g}.json"
            r = load(p)
            e = r["end_to_end"]
            vals = [d["rrms"] for d in e["devices"].values()]
            out[(arch, f)] = {
                "path": p, "n_train": r["n_train"], "n_params": r["n_parameters"],
                "epochs": r["epochs"], "train_s": r["train_s"],
                "val": r["best_val_loss"], "test": r["test"]["loss"],
                "nrmse_med": r["test"]["curve_nrmse_median"],
                "score_err": r["ranking"]["mean_abs_score_error"],
                "rank_sur": r["ranking"]["direct_ranked_better_by_surrogate"],
                "rank_ng": r["ranking"]["direct_better_by_ngspice"],
                "all": e["all_device_mean_rrms"], "combined": e["combined_rrms"],
                "nmos": e["nmos_rrms"], "pmos": e["pmos_rrms"],
                "dyn": e["official_dynamic_all_device_mean"],
                "n_dev": e["n_devices"],
                "failed": int(sum(not np.isfinite(v) for v in vals)),
                "lhc": sum(d["selected_candidate_kind"] == "lhc_start"
                           for d in e["devices"].values()),
                "search_s": e["search_s"],
            }
    return out


def fig_lc(lc: dict, swap: dict, path: Path) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for ax in axes:
        style(ax)
        ax.set_xscale("log", base=2)
    for arch in LC_ARCHS:
        n = [lc[(arch, f)]["n_train"] for f in LC_FRACTIONS]
        lab = rf.ARCH_LABEL[arch]
        c = ARCH_COLOR[arch]
        axes[0].plot(n, [lc[(arch, f)]["test"] for f in LC_FRACTIONS], "-o", color=c, lw=2,
                     ms=5, label=lab)
        axes[1].plot(n, [lc[(arch, f)]["all"] for f in LC_FRACTIONS], "-o", color=c, lw=2,
                     ms=5, label=f"{lab}, all-device")
        axes[1].plot(n, [lc[(arch, f)]["combined"] for f in LC_FRACTIONS], "--", color=c,
                     lw=1.2, label=f"{lab}, combined")
        axes[2].plot(n, [lc[(arch, f)]["lhc"] for f in LC_FRACTIONS], "-o", color=c, lw=2,
                     ms=5, label=lab)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("test loss on held-out bank samples", fontsize=9, color=INK)
    axes[0].set_title("Surrogate test loss", fontsize=9.5, color=INK)
    axes[1].axhline(swap["all_device_mean_rrms"], color=GREY, lw=1.5)
    axes[1].axhline(swap["combined_rrms"], color=GREY, lw=1.0, ls="--")
    axes[1].text(lc[(LC_ARCHS[0], LC_FRACTIONS[0])]["n_train"], swap["all_device_mean_rrms"] + 0.004,
                 f"same search, NGSpice in place of the surrogate: all-device "
                 f"{swap['all_device_mean_rrms']:.4f}, combined {swap['combined_rrms']:.4f}",
                 fontsize=6.5, color=MUTED, va="bottom")
    axes[1].set_ylim(0.09, 0.28)
    axes[1].set_ylabel("end-to-end paper RRMS, 18 measured devices", fontsize=9, color=INK)
    axes[1].set_title("End-to-end search, selected card re-simulated in NGSpice",
                      fontsize=9.5, color=INK)
    axes[2].set_ylim(0, 18.5)
    axes[2].set_ylabel("devices where the selected card is a bank start", fontsize=9, color=INK)
    axes[2].set_title("Selected card is a bank (LHC) start, of 18", fontsize=9.5, color=INK)
    for ax in axes:
        ax.set_xlabel("training samples (round-1 43-parameter cache)", fontsize=8.5, color=INK)
        ax.set_xticks([lc[(LC_ARCHS[0], f)]["n_train"] for f in LC_FRACTIONS])
        ax.set_xticklabels([f"{lc[(LC_ARCHS[0], f)]['n_train']:,}" for f in LC_FRACTIONS],
                           fontsize=7.5)
        ax.minorticks_off()
    axes[0].legend(fontsize=7.5, frameon=False)
    axes[1].legend(fontsize=6.5, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.17), ncol=2)
    fig.suptitle("Learning curves (archlc_full_r1 + arch_full_r1 at 100%): three surrogate "
                 "architectures at 1/8, 1/4, 1/2 and all of the training split",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(lc[k]["path"]) for k in sorted(lc)] + [rel(SWAP)]


def table_lc(lc: dict, swap: dict) -> rf.Table:
    rows = []
    for arch in LC_ARCHS:
        for f in LC_FRACTIONS:
            r = lc[(arch, f)]
            rows.append({"arch": rf.ARCH_LABEL[arch], "frac": f"{f:g}", "n": r["n_train"],
                         "epochs": r["epochs"], "train_s": r["train_s"], "test": r["test"],
                         "nrmse": r["nrmse_med"], "err": r["score_err"],
                         "rank": f"{r['rank_sur']}/18", "all": r["all"],
                         "combined": r["combined"], "nmos": r["nmos"], "pmos": r["pmos"],
                         "failed": r["failed"], "lhc": r["lhc"]})
    rows.append({"arch": "NGSpice in place of the surrogate", "frac": "—", "n": None,
                 "epochs": None, "train_s": None, "test": None, "nrmse": None, "err": None,
                 "rank": "18/18", "all": swap["all_device_mean_rrms"],
                 "combined": swap["combined_rrms"], "nmos": swap["nmos_rrms"],
                 "pmos": swap["pmos_rrms"], "failed": 0, "lhc": 0})
    cols = [
        {"key": "arch", "head": "surrogate", "md": "surrogate", "kind": "text"},
        {"key": "frac", "head": "fraction", "md": "train fraction", "kind": "rtext"},
        {"key": "n", "head": "samples", "md": "training samples", "kind": "int"},
        {"key": "epochs", "head": "epochs", "md": "epochs", "kind": "int"},
        {"key": "train_s", "head": "train (s)", "md": "train time (s)", "kind": "int"},
        {"key": "test", "head": "test loss", "group": "held-out bank samples", "md": "test loss", "kind": "rrms"},
        {"key": "nrmse", "head": "NRMSE med.", "group": "held-out bank samples", "md": "curve NRMSE, median", "kind": "rrms"},
        {"key": "err", "head": "|score err.|", "group": "reference cards", "md": "mean absolute score error", "kind": "rrms"},
        {"key": "rank", "head": "direct better", "group": "reference cards", "md": "devices where the surrogate ranks the direct fit better", "kind": "rtext"},
        {"key": "all", "head": "all-device", "group": "end-to-end RRMS, 18 devices", "md": "end-to-end all-device", "kind": "rrms"},
        {"key": "combined", "head": "combined", "group": "end-to-end RRMS, 18 devices", "md": "end-to-end combined", "kind": "rrms"},
        {"key": "nmos", "head": "nMOS", "group": "end-to-end RRMS, 18 devices", "md": "end-to-end nMOS", "kind": "rrms", "family": True},
        {"key": "pmos", "head": "pMOS", "group": "end-to-end RRMS, 18 devices", "md": "end-to-end pMOS", "kind": "rrms", "family": True},
        {"key": "failed", "head": "failed", "md": "failed devices", "kind": "int"},
        {"key": "lhc", "head": "bank start", "md": "devices where a bank start is selected", "kind": "int"},
    ]
    notes = [
        "Script: scripts/research/surrogate_arch_study.py on out/cold_loop/full/round_1 (cache) "
        "and data/processed/cold_loop/full_r1 (bank), seed 20260818. The training subset at "
        "fraction f is the first round(f × 85,853) indices of one fixed permutation of the "
        "pipeline's training split; validation and test splits are unchanged.",
        "Fractions 1/8, 1/4, 1/2: queue job archlc_full_r1 (first attempt 2026-10-18 03:38, "
        "killed rc=143 after 10,138 s; restarted 2026-10-19 09:01, finished 10:05; the restart "
        "skipped finished reports and trained fno_f0.5). Fraction 1: the reports of arch_full_r1 "
        "(2026-10-16/17), same script, split and seed.",
        "Recipe: AdamW lr 1e-3, weight decay 1e-5, cosine schedule, batch 512, early stopping "
        "(patience 25) on validation loss; at most 200 epochs (FNO 80).",
        "End-to-end = fps.finite_difference_surrogate_extract with the trained surrogate frozen: 8 "
        "starts, TRF max_nfev 100, diff_step 0.02, regularization 1e-4, NGSpice validation of the "
        "top 4 endpoints, the first 4 starts and the published fallback; the selected card is "
        "re-simulated in NGSpice on the measured 77 K data. One fixed method per row; no "
        "per-device selection across rows.",
        "all-device = mean over 18 devices; combined = mean of the nMOS (8) and pMOS (10) family "
        "means. bank start = devices whose selected card is one of the LHC bank starts "
        "(selected_candidate_kind lhc_start) rather than an FD endpoint; bank starts are the same "
        "for every surrogate, so rows with many bank-start selections share per-device values.",
        "The end-to-end search ran on the 18 measured devices; it is a diagnostic ablation and is "
        "not used for method selection.",
    ]
    return rf.Table("06_table_lc",
                    "Surrogate learning curves, 43 parameters, round-1 box: training-set size vs "
                    "held-out accuracy and end-to-end search on the 18 measured devices",
                    cols, rows, notes)


# ------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    topic = "explore_e1_lc"
    folder = rf.REPORT_ROOT / f"{args.date}_{topic}"
    folder.mkdir(parents=True, exist_ok=True)

    s1, s2 = load(STAGE1), load(STAGE2)
    logo = {k: load(p) for k, p in LOGO.items()}
    swap = load(SWAP)
    lc = lc_records()

    items = []  # (png, description, sources)
    src = fig_stage2_per_device(s2, folder / "01_e1_stage2_per_device.png")
    items.append(("01_e1_stage2_per_device.png",
                  "Per-device held-out paper RRMS for E1 stage 2 on the 12 REAL_DEV devices "
                  "(leave-one-device-out, card re-polished with the fold's correction in the "
                  "residual): C0 refit without correction (grey circle), M MLP correction "
                  "(blue diamond), T1 per-tag gain (orange square); log scale; dotted line "
                  "separates nMOS and pMOS", src))
    t2 = table_stages(s1, s2)
    rf.render_table(t2, folder / "02_table_e1_stages.png")
    items.append(("02_table_e1_stages.png",
                  "Table of E1 stage 1 (post-hoc correction on the fixed card) and stage 2 "
                  "(card refit with the correction in the loop) on REAL_DEV: all-device, "
                  "combined and median RRMS of C0, T1 and M, failed devices, device counts "
                  "M < C0 and M < T1, and the outcome of the five pre-registered checks",
                  [rel(STAGE1), rel(STAGE2), rel(PREREG)]))
    src = fig_stage_bars(s1, s2, folder / "03_e1_stage1_vs_stage2.png")
    items.append(("03_e1_stage1_vs_stage2.png",
                  "Bar chart of all-device (left) and combined (right) RRMS for C0, T1 and M in "
                  "stage 1 (light bars) and stage 2 (dark hatched bars), with the "
                  "pre-registered 0.090 threshold for M", src))
    src = fig_logo(logo, folder / "04_e1_sim_logo.png")
    t4 = table_logo(logo)
    items.append(("04_e1_sim_logo.png",
                  "Bar chart of all-device mean RRMS of C0, T1, T2, LIN and M on the simulated "
                  "GF180 and IHP tier c targets (sets 0-2, 48 per PDK, leave-one-geometry-out, "
                  "post-hoc correction), with the criterion_S line at 0.90 × C0", src))
    rf.render_table(t4, folder / "05_table_e1_sim_logo.png")
    items.append(("05_table_e1_sim_logo.png",
                  "Table of the simulated leave-one-geometry-out run per PDK and arm: "
                  "all-device, combined, nMOS, pMOS and median RRMS, ratio to C0, failed "
                  "targets and the number of targets on which M is lower than each arm",
                  [rel(p) for p in LOGO.values()]))
    src = fig_lc(lc, swap, folder / "06_lc_curves.png")
    items.append(("06_lc_curves.png",
                  "Learning curves for paper MLP, DeepONet and FNO at 10,732 / 21,463 / 42,926 / "
                  "85,853 training samples: test loss (left, log scale), end-to-end all-device "
                  "(solid) and combined (dashed) RRMS on the 18 measured devices with the NGSpice "
                  "swap reference (middle), and devices whose selected card is a bank start "
                  "(right)", src))
    t6 = table_lc(lc, swap)
    rf.render_table(t6, folder / "07_table_lc.png")
    items.append(("07_table_lc.png",
                  "Table of the learning-curve study per architecture and training fraction: "
                  "samples, epochs, training time, test loss, median curve NRMSE, mean absolute "
                  "score error at the reference cards, devices where the surrogate ranks the "
                  "direct-fit card better, end-to-end all-device, combined, nMOS and pMOS RRMS, "
                  "failed devices and bank-start selections, with the NGSpice swap row", src))

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_e1_lc.py --date {args.date}`"
    tables_md = [f"# Tables, {args.date} {topic}", "",
                 f"Generated {now} by {gen} from result files only. RRMS is the paper metric "
                 "with frozen published-baseline curve inclusion, strict (a non-finite "
                 "included curve makes the device NaN).", ""]
    for t in (t2, t4, t6):
        tables_md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(tables_md))

    readme = [f"# Report batch {args.date} · {topic}", "",
              f"Generated {now} by {gen} from result files only (no simulation, no training).", "",
              "## Contents", "",
              "- E1 stage 2 on REAL_DEV (card refit with the correction in the loop) and its "
              "comparison with stage 1 (post-hoc correction on the fixed card). Pre-registration: "
              f"`{rel(PREREG)}`.",
              "- E1 simulated benchmark, GF180 and IHP tier c, sets 0-2, leave-one-geometry-out, "
              "post-hoc correction only (no in-loop refit); criterion_S is secondary.",
              "- Surrogate learning curves `archlc_full_r1` (paper MLP, DeepONet, FNO at 1/8, 1/4, "
              "1/2 of the training split) with the 100% points from `arch_full_r1`.", "",
              "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{s}`" for s in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))

    pm = lambda a, f: lc[(a, f)]  # noqa: E731
    summary = [
        "E1 stage 2, REAL_DEV LODO, card refit with the correction in the loop, all-device "
        f"(combined): C0 refit {s2['summary']['C0']['all_device_mean']:.4f} "
        f"({s2['summary']['C0']['combined']:.4f}); M {s2['summary']['M']['all_device_mean']:.4f} "
        f"({s2['summary']['M']['combined']:.4f}); T1 {s2['summary']['T1']['all_device_mean']:.4f} "
        f"({s2['summary']['T1']['combined']:.4f}). Pre-registered rule: fail (M < T1 mean no; "
        "M < T1 on 6/12; M < C0 refit on 8/12). Stage 1 (post-hoc): M "
        f"{s1['summary']['M']['all_device_mean']:.4f}, C0 {s1['summary']['C0']['all_device_mean']:.4f}, "
        f"T1 {s1['summary']['T1']['all_device_mean']:.4f}.",
        "E1 simulated, tier c sets 0-2, leave-one-geometry-out, post-hoc, all-device: " + "; ".join(
            f"{pdk.split()[0]} " + ", ".join(
                f"{a} {res['summary'][a]['all_device_mean']:.4f}" for a in ("C0", "T1", "T2", "LIN", "M"))
            for pdk, res in logo.items()) + ". criterion_S fails on both PDKs (secondary).",
        "Learning curves, end-to-end all-device RRMS at 1/8 → 1/1 of 85,853 samples: " + "; ".join(
            f"{rf.ARCH_LABEL[a]} " + " → ".join(f"{pm(a, f)['all']:.4f}" for f in LC_FRACTIONS)
            for a in LC_ARCHS) + f"; NGSpice in place of the surrogate {swap['all_device_mean_rrms']:.4f}.",
    ]
    pngs = [i[0] for i in items]
    rf.update_index(args.date, topic, folder, pngs, summary)
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
