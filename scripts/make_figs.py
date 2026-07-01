#!/usr/bin/env python3
"""Reproduce the paper's figures from this repo's saved NGSpice sweeps.

Every panel overlays three things in the paper's visual convention
(measured = hollow circles, model = lines, absolute V and I):

  measured 77 K data | paper cards in NGSpice (dashed) | ML cards (solid)

Outputs (paper analogue in parentheses):
  figs/fig2_iv_77k.png        (Fig. 2)  representative I-V at 77 K
  figs/fig4_bestfit.png       (Fig. 4)  output/transfer + weak/strong inversion
  figs/fig5_rrms_heatmap.png  (Fig. 5)  RRMS heat maps across geometries
  figs/table6_bars.png        (Table 6) per-device RRMS, all 18 devices
  figs/ml_ablation.png                  mean RRMS by extraction stage
  figs/devices/<tag>.png                appendix: every Table-6 device
  out/tables/table6.md        (Table 6) RMSE / RRMS / sigma per device & method
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.config import FIGS_DIR, OUT_DIR, OUT_TABLES, ensure_dirs  # noqa: E402
from cryoml.data_io import Curve, load_device_curves  # noqa: E402
from cryoml.devices import Device, PAPER_DEVICES, find_device  # noqa: E402
from cryoml.metrics import device_rrms  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402

METHODS = {
    "paper": {
        "dir": OUT_DIR / "pdk_baseline",
        "label": "paper cards (NGSpice)",
        "ls": "--",
    },
    "ml": {
        "dir": OUT_DIR / "pdk_ml_final",
        "label": "ML cards (NGSpice)",
        "ls": "-",
    },
    "direct": {
        "dir": OUT_DIR / "pdk_fwd_surr_best",
        "label": "direct-predict card (NGSpice)",
        "ls": ":",
    },
}

# scored (but not curve-plotted) reference: strongest classical control under
# the same one-card-per-bin constraint
CTRL_DIR = OUT_DIR / "pdk_cma_deploy"
CTRL_LABEL = "CMA-ES refit (one card per size bin)"

# Floor for log-scale current panels (measured noise floor is ~1e-12 A).
LOG_FLOOR_A = 1e-13
MARK_EVERY = 4

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 1.3,
})


def load_sims(dev: Device, n_curves: int) -> dict[str, list[np.ndarray]]:
    tag = device_tag(dev.dev_type, dev.L_um, dev.W_um)
    out = {}
    for key, cfg in METHODS.items():
        path = cfg["dir"] / f"sims_{tag}.npz"
        saved = np.load(path)
        out[key] = [np.asarray(saved[f"sim_{i}"]) for i in range(n_curves)]
    return out


def indexed(curves: list[Curve], kind: str) -> list[tuple[int, Curve]]:
    """Curves of one sweep kind, sorted by |fixed bias|, with original index."""
    pairs = [(i, c) for i, c in enumerate(curves) if c.kind == kind]
    return sorted(pairs, key=lambda p: abs(p[1].fixed))


def bias_colors(n: int) -> list:
    return [plt.cm.viridis(v) for v in np.linspace(0.0, 0.78, n)]


def method_legend_handles(keys=None) -> list[Line2D]:
    handles = [
        Line2D([], [], marker="o", mfc="none", mec="k", ls="none",
               markersize=4.5, label="measured 77 K"),
    ]
    for key in (keys if keys is not None else list(METHODS)):
        cfg = METHODS[key]
        handles.append(Line2D([], [], color="k", ls=cfg["ls"], label=cfg["label"]))
    return handles


def iv_by_method(devs_curves_sims):
    """One representative-device I-V figure per method (measured + that single
    method only) — so each line's method is unambiguous without overlaying."""
    reps = [("nmos", 8.0, 1.6), ("pmos", 2.0, 5.0)]
    for key, cfg in METHODS.items():
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
        letters = iter("abcd")
        for row, rep in enumerate(reps):
            dev, curves, sims = devs_curves_sims[rep]
            score = device_rrms([s for s in sims[key]], [c.Id for c in curves])
            ax = axes[row, 0]
            plot_output(ax, curves, sims, n_biases=2, keys=[key])
            ax.legend(loc="upper left")
            ax.set_title(f"({next(letters)}) {dev_title(dev)} — output "
                         f"(RRMS {score['rrms']:.3f})")
            ax = axes[row, 1]
            pairs = indexed(curves, "idvg")[-1:]
            _plot_curve_set(ax, pairs, sims, lambda c: c.Vg, "V$_{DS}$",
                            in_uA=True, log=False, keys=[key])
            ax.legend(loc="upper left")
            ax.set_xlabel("|V$_{GS}$| (V)")
            ax.set_title(f"({next(letters)}) {dev_title(dev)} — transfer, "
                         f"|V$_{{DS}}$| = {abs(pairs[0][1].fixed):.2f} V")
        fig.legend(handles=method_legend_handles(keys=[key]),
                   loc="lower center", ncol=2, frameon=False)
        fig.suptitle(f"Measured 77 K vs {cfg['label']}", y=0.995)
        fig.tight_layout(rect=(0, 0.045, 1, 1))
        fig.savefig(FIGS_DIR / f"iv_{key}.png")
        plt.close(fig)


def _plot_curve_set(
    ax,
    pairs: list[tuple[int, Curve]],
    sims: dict[str, list[np.ndarray]],
    x_of,
    bias_name: str,
    in_uA: bool,
    log: bool,
    xlim=None,
    keys=None,
):
    """Measured circles + one line per method for each (index, curve) pair."""
    colors = bias_colors(len(pairs))
    scale = 1e6 if in_uA else 1.0

    def y(values):
        v = np.abs(np.asarray(values, dtype=np.float64))
        if log:
            v = np.clip(v, LOG_FLOOR_A, None)
        return v * scale

    for color, (idx, curve) in zip(colors, pairs):
        x = np.abs(x_of(curve))
        keep = slice(None)
        if xlim is not None:
            keep = (x >= xlim[0]) & (x <= xlim[1])
            if not np.any(keep):
                continue
        ax.plot(x[keep], y(curve.Id)[keep], "o", mfc="none", mec=color,
                markersize=3.2, markeredgewidth=0.7, ls="none",
                markevery=MARK_EVERY,
                label=f"|{bias_name}| = {abs(curve.fixed):.2f} V")
        for key in (keys if keys is not None else list(METHODS)):
            cfg = METHODS[key]
            ax.plot(x[keep], y(sims[key][idx])[keep], cfg["ls"], color=color,
                    alpha=0.9)
    if log:
        ax.set_yscale("log")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_ylabel("|I$_D$| (µA)" if in_uA else "|I$_D$| (A)")


def plot_output(ax, curves, sims, n_biases=None, xlim=None, keys=None):
    pairs = indexed(curves, "idvd")
    if n_biases is not None and len(pairs) > n_biases:
        # Evenly spread selection ending at the strongest bias; skip the
        # weakest (sub-threshold, device-off) bias like the paper does.
        pool = pairs[1:]
        sel = np.linspace(0, len(pool) - 1, n_biases).round().astype(int)
        pairs = [pool[i] for i in sel]
    _plot_curve_set(ax, pairs, sims, lambda c: c.Vd, "V$_{GS}$",
                    in_uA=True, log=False, xlim=xlim, keys=keys)
    ax.set_xlabel("|V$_{DS}$| (V)")


def plot_transfer(ax, curves, sims, log=False, n_biases=None, xlim=None,
                  keys=None):
    pairs = indexed(curves, "idvg")
    if n_biases is not None and len(pairs) > n_biases:
        sel = np.linspace(0, len(pairs) - 1, n_biases).round().astype(int)
        pairs = [pairs[i] for i in sel]
    _plot_curve_set(ax, pairs, sims, lambda c: c.Vg, "V$_{DS}$",
                    in_uA=not log, log=log, xlim=xlim, keys=keys)
    ax.set_xlabel("|V$_{GS}$| (V)")


def device_scores(curves, sims) -> dict[str, dict]:
    meas = [c.Id for c in curves]
    return {key: device_rrms(s, meas) for key, s in sims.items()}


def dev_title(dev: Device) -> str:
    pol = "nMOS" if dev.dev_type == "nmos" else "pMOS"
    return f"{pol} L={dev.L_um:g} µm, W={dev.W_um:g} µm"


# ---------------------------------------------------------------- figures


def fig2(devs_curves_sims):
    """Paper Fig. 2: representative output + transfer (lin & log) curves."""
    reps = [("nmos", 8.0, 1.6), ("pmos", 2.0, 5.0)]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    letters = iter("abcd")
    for row, key in enumerate(reps):
        dev, curves, sims = devs_curves_sims[key]
        ax = axes[row, 0]
        plot_output(ax, curves, sims, n_biases=2)
        ax.legend(loc="upper left")
        ax.set_title(f"({next(letters)}) {dev_title(dev)} — output")

        # Transfer at the strongest |VDS| (linear).
        ax = axes[row, 1]
        pairs = indexed(curves, "idvg")[-1:]
        _plot_curve_set(ax, pairs, sims, lambda c: c.Vg, "V$_{DS}$",
                        in_uA=True, log=False)
        ax.legend(loc="upper left")
        ax.set_xlabel("|V$_{GS}$| (V)")
        ax.set_title(f"({next(letters)}) {dev_title(dev)} — transfer, "
                     f"|V$_{{DS}}$| = {abs(pairs[0][1].fixed):.2f} V")
    fig.legend(handles=method_legend_handles(), loc="lower center", ncol=3,
               frameon=False)
    fig.suptitle("Measured vs simulated I-V at 77 K (paper Fig. 2 analogue)",
                 y=0.995)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(FIGS_DIR / "fig2_iv_77k.png")
    plt.close(fig)


def fig4(devs_curves_sims):
    """Paper Fig. 4: best-fit characteristics + weak/strong inversion."""
    reps = [("nmos", 0.15, 1.6), ("pmos", 2.0, 5.0)]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2))
    letters = iter("abcdefgh")
    for col0, key in zip((0, 2), reps):
        dev, curves, sims = devs_curves_sims[key]
        scores = device_scores(curves, sims)
        sub = (f"RRMS {scores['paper']['rrms']:.3f} → "
               f"{scores['ml']['rrms']:.3f}")

        ax = axes[0, col0]
        plot_output(ax, curves, sims, n_biases=3)
        ax.legend(loc="upper left")
        ax.set_title(f"({next(letters)}) {dev_title(dev)}\noutput — {sub}")

        ax = axes[0, col0 + 1]
        plot_transfer(ax, curves, sims, log=False, n_biases=3)
        ax.legend(loc="upper left")
        ax.set_title(f"({next(letters)}) {dev_title(dev)}\ntransfer (linear)")

    for col0, key in zip((0, 2), reps):
        dev, curves, sims = devs_curves_sims[key]
        vg_max = max(abs(c.Vg).max() for c in curves if c.kind == "idvg")

        ax = axes[1, col0]
        plot_transfer(ax, curves, sims, log=False, n_biases=3,
                      xlim=(0.0, 0.62 * vg_max))
        ax.set_title(f"({next(letters)}) {dev_title(dev)}\nweak inversion")

        ax = axes[1, col0 + 1]
        plot_transfer(ax, curves, sims, log=False, n_biases=3,
                      xlim=(0.55 * vg_max, vg_max))
        ax.set_title(f"({next(letters)}) {dev_title(dev)}\nstrong inversion")

    fig.legend(handles=method_legend_handles(), loc="lower center", ncol=3,
               frameon=False)
    fig.suptitle("Best-fit 77 K characteristics (paper Fig. 4 analogue)",
                 y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGS_DIR / "fig4_bestfit.png")
    plt.close(fig)


def fig5(rows: list[dict]):
    """Paper Fig. 5: RRMS heat maps across geometries, identical color scale."""
    score_keys = [("paper_params_ngspice", "paper cards"),
                  ("ctrl", "CMA-ES refit"),
                  ("ml", "ML refit")]
    vmax = max(r[k] for r in rows for k, _ in score_keys)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("0.92")
    im = None
    for row_i, dev_type in enumerate(("nmos", "pmos")):
        sub = [r for r in rows if r["dev_type"] == dev_type]
        Ls = sorted({r["L"] for r in sub})
        Ws = sorted({r["W"] for r in sub})
        grid_of = {}
        for col_i, (key, label) in enumerate(score_keys):
            grid = np.full((len(Ws), len(Ls)), np.nan)
            for r in sub:
                grid[Ws.index(r["W"]), Ls.index(r["L"])] = r[key]
            grid_of[key] = grid
            ax = axes[row_i, col_i]
            im = ax.imshow(grid, origin="lower", cmap=cmap, vmin=0, vmax=vmax,
                           aspect="auto")
            ax.set_xticks(range(len(Ls)), [f"{v:g}" for v in Ls])
            ax.set_yticks(range(len(Ws)), [f"{v:g}" for v in Ws])
            ax.set_xlabel("L (µm)")
            ax.set_ylabel("W (µm)")
            ax.grid(False)
            for (wi, li), val in np.ndenumerate(grid):
                if np.isfinite(val):
                    ax.text(li, wi, f"{val:.2f}", ha="center", va="center",
                            fontsize=8,
                            color="w" if val < 0.55 * vmax else "k")
            mean = np.nanmean(grid)
            pol = "nMOS" if dev_type == "nmos" else "pMOS"
            ax.set_title(f"{pol} — {label} (mean {mean:.3f})")
    fig.colorbar(im, ax=axes, shrink=0.85, label="paper-exact RRMS")
    fig.suptitle("RRMS across measured geometries at 77 K "
                 "(paper Fig. 5 analogue)")
    fig.savefig(FIGS_DIR / "fig5_rrms_heatmap.png")
    plt.close(fig)


def table6_bars(rows: list[dict]):
    """Per-device RRMS for all 18 Table-6 devices."""
    x = np.arange(len(rows))
    width = 0.2
    fig, ax = plt.subplots(figsize=(13, 4.6))
    base_mean = np.mean([r["paper_params_ngspice"] for r in rows])
    ctrl_mean = np.mean([r["ctrl"] for r in rows])
    ml_mean = np.mean([r["ml"] for r in rows])
    direct_mean = np.mean([r["direct"] for r in rows])
    ax.bar(x - 1.5 * width, [r["paper_params_ngspice"] for r in rows], width,
           label=f"paper cards as published, mean {base_mean:.3f}",
           color="#4878a8")
    ax.bar(x - 0.5 * width, [r["ctrl"] for r in rows], width,
           label="refit with CMA-ES optimizer, one card per size bin — "
                 f"mean {ctrl_mean:.3f}", color="#74a884")
    ax.bar(x + 0.5 * width, [r["ml"] for r in rows], width,
           label="refit with ML, one card per size bin — "
                 f"mean {ml_mean:.3f}", color="#e8923c")
    ax.bar(x + 1.5 * width, [r["direct"] for r in rows], width,
           label="direct-predict ML, one card per transistor — "
                 f"mean {direct_mean:.3f}", color="#a86cb0")
    ax.plot(x, [r["paper_reported"] for r in rows], "kD", markersize=4,
            ls="none", label="paper reported (HSPICE flow)")
    ax.axhline(ctrl_mean, color="#74a884", ls=":", lw=1)
    ax.axhline(ml_mean, color="#e8923c", ls=":", lw=1)
    n_nmos = sum(r["dev_type"] == "nmos" for r in rows)
    ax.axvline(n_nmos - 0.5, color="k", lw=0.8, alpha=0.4)
    ax.set_xticks(x, [r["device"] for r in rows], rotation=60, ha="right",
                  fontsize=8)
    ax.set_ylabel("paper-exact RRMS")
    ax.set_title("All 18 Table-6 devices — identical NGSpice flow")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "table6_bars.png")
    plt.close(fig)


def ml_ablation():
    """Mean RRMS by extraction stage (per-device means over 18 devices)."""
    summary = json.load(open(OUT_DIR / "pdk_ml2" / "summary.json"))
    v3 = json.load(open(OUT_DIR / "pdk_ml3" / "summary.json"))
    fd = json.load(open(OUT_DIR / "pdk_fd" / "summary.json"))["mean_rrms"]
    cma = json.load(open(OUT_DIR / "pdk_cma" / "summary.json"))["mean_rrms"]

    def dir_mean(name):
        vals = [json.loads(p.read_text())["rrms"]
                for p in (OUT_DIR / name).glob("ml_*.json")]
        return float(np.mean(vals))

    stages = [
        ("paper cards (no refit)", summary["baseline_mean_rrms"]),
        ("NN predicts parameters from curves,\nthen least-squares",
         summary["mean_inverse_mlp+fd"]),
        ("ML method, search stage only\n(no feedback rounds)",
         summary["mean_emu_search+fd"]),
        ("ML method with feedback rounds\n(re-simulate best, retrain)",
         v3["mean_active_bo+fd"]),
        ("full ML method, one card per transistor",
         dir_mean("pdk_ml_final_perdev")),
        ("full ML method, one card per size bin",
         dir_mean("pdk_ml_final")),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.0))
    names = [s[0] for s in stages]
    vals = [s[1] for s in stages]
    colors = ["#4878a8"] + ["#9dbcd4"] * 3 + ["#e8923c", "#e8923c"]
    bars = ax.barh(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(v + 0.008, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=8.5)
    ax.axvline(fd, color="#555", ls="--", lw=1)
    ax.axvline(cma, color="#2a6e3f", ls="--", lw=1)
    ax.text(cma - 0.012, len(stages) / 2,
            f"classical controls\nCMA {cma:.3f} | FD {fd:.3f}",
            color="#2a6e3f", fontsize=8, ha="right", va="center",
            rotation=90)
    ax.invert_yaxis()
    ax.set_xlabel("mean paper-exact RRMS over 18 devices")
    ax.set_title("Extraction stages vs paper cards and classical controls "
                 "(identical NGSpice flow)")
    ax.set_xlim(0, max(vals) * 1.14)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "ml_ablation.png")
    plt.close(fig)


def device_appendix(devs_curves_sims):
    out_dir = FIGS_DIR / "devices"
    out_dir.mkdir(parents=True, exist_ok=True)
    for dev, curves, sims in devs_curves_sims.values():
        scores = device_scores(curves, sims)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        plot_output(axes[0], curves, sims)
        axes[0].set_title("output")
        axes[0].legend(loc="upper left", fontsize=6.5)
        plot_transfer(axes[1], curves, sims, log=False)
        axes[1].set_title("transfer")
        axes[1].legend(loc="upper left", fontsize=6.5)
        fig.legend(handles=method_legend_handles(), loc="lower center",
                   ncol=3, frameon=False)
        fig.suptitle(
            f"{dev_title(dev)} — RRMS {scores['paper']['rrms']:.3f} (paper cards)"
            f" → {scores['ml']['rrms']:.3f} (ML cards)")
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        tag = device_tag(dev.dev_type, dev.L_um, dev.W_um)
        fig.savefig(out_dir / f"{tag}.png")
        plt.close(fig)


def table6_md(devs_curves_sims):
    """Table 6 analogue: RMSE (µA), RRMS, sigma per device and method."""
    lines = [
        "# Table 6 analogue — error metrics per device",
        "",
        "Paper-reported columns come from the paper's HSPICE/Mystic flow; the",
        "NGSpice columns are computed here with the companion-notebook metric",
        "in the identical corrected NGSpice chain.",
        "",
        "| device | reported RRMS | reported σ | paper cards RMSE (µA) | "
        "paper cards RRMS | paper cards σ | ML RMSE (µA) | ML RRMS | ML σ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    means = {"paper": [], "ml": []}
    for dev, curves, sims in devs_curves_sims.values():
        scores = device_scores(curves, sims)
        tag = device_tag(dev.dev_type, dev.L_um, dev.W_um)
        p, m = scores["paper"], scores["ml"]
        means["paper"].append(p["rrms"])
        means["ml"].append(m["rrms"])
        lines.append(
            f"| {tag} | {dev.paper_rrms:.3f} | {dev.paper_sigma:.3f} | "
            f"{p['rmse_uA']:.3f} | {p['rrms']:.3f} | {p['sigma']:.3f} | "
            f"{m['rmse_uA']:.3f} | {m['rrms']:.3f} | {m['sigma']:.3f} |")
    lines.append(
        f"| **mean** | 0.279 | | | **{np.mean(means['paper']):.3f}** | | | "
        f"**{np.mean(means['ml']):.3f}** | |")
    (OUT_TABLES / "table6.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ensure_dirs()

    devs_curves_sims = {}
    for dev in PAPER_DEVICES:
        curves = load_device_curves(dev)
        sims = load_sims(dev, len(curves))
        devs_curves_sims[(dev.dev_type, dev.L_um, dev.W_um)] = (
            dev, curves, sims)

    rows = []
    for (dev_type, L, W), (dev, curves, sims) in devs_curves_sims.items():
        scores = device_scores(curves, sims)
        tag = device_tag(dev_type, L, W)
        ctrl = np.load(CTRL_DIR / f"sims_{tag}.npz")
        ctrl_sims = [np.asarray(ctrl[f"sim_{i}"]) for i in range(len(curves))]
        rows.append({
            "device": tag,
            "dev_type": dev_type,
            "L": L,
            "W": W,
            "paper_reported": dev.paper_rrms,
            "paper_params_ngspice": scores["paper"]["rrms"],
            "ctrl": float(device_rrms(ctrl_sims,
                                      [c.Id for c in curves])["rrms"]),
            "ml": scores["ml"]["rrms"],
            "direct": scores["direct"]["rrms"],
        })

    fig2(devs_curves_sims)
    fig4(devs_curves_sims)
    iv_by_method(devs_curves_sims)
    fig5(rows)
    table6_bars(rows)
    ml_ablation()
    device_appendix(devs_curves_sims)
    table6_md(devs_curves_sims)
    print(f"figures written to {FIGS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
