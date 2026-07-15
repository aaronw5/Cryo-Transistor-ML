#!/usr/bin/env python3
"""Reproduce the paper's figures from this repo's saved NGSpice sweeps.

Every main panel overlays the measured data and five fixed simulated series
in the paper's visual convention
(measured = hollow circles, model = lines, absolute V and I):

  measured | paper cards | direct MLP | surrogate raw | surrogate + FD |
  foundation + FD

Outputs (paper analogue in parentheses):
  figs/fig2_iv_77k.png        (Fig. 2)  representative I-V at 77 K
  figs/fig4_bestfit.png       (Fig. 4)  output/transfer + weak/strong inversion
  figs/fig5_rrms_heatmap.png  (Fig. 5)  RRMS heat maps across geometries
  figs/table6_bars.png        (Table 6) per-device RRMS, all 18 devices
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
from cryoml.devices import Device, PAPER_DEVICES  # noqa: E402
from cryoml.metrics import score_device_new  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402

# Every series is fixed across all 18 devices; no per-device method selection.
METHODS = {
    "paper": {
        "dir": OUT_DIR / "pdk_baseline",
        "label": "paper cards (NGSpice)",
        "ls": "--",
    },
    "direct": {
        "dir": OUT_DIR / "pdk_direct_mlp",
        "label": "direct MLP params (NGSpice)",
        "ls": ":",
    },
    "emu_raw": {
        "dir": OUT_DIR / "pdk_ml_emu_raw",
        "label": "surrogate raw params (NGSpice)",
        "ls": "-.",
    },
    "emu_fd": {
        "dir": OUT_DIR / "pdk_ml_emu",
        "label": "surrogate + FD params (NGSpice)",
        "ls": "-",
    },
    "foundation_fd": {
        "dir": OUT_DIR / "pdk_foundation_emu",
        "label": "foundation + FD params (NGSpice)",
        "prefix": "fd_sim_",
        "ls": (0, (7, 1, 1, 1)),
    },
    "hv_guarded": {
        "dir": OUT_DIR / "pdk_high_voltage_guarded",
        "label": "high-voltage guarded params (NGSpice)",
        "ls": (0, (3, 1)),
    },
}
EXPECTED_METHOD_IDS = {
    "direct": "direct_mlp_forward_pass",
    "emu_raw": "emu_search",
    "emu_fd": "emu_search+fd",
    "foundation_fd": "foundation_emu_search_fixed",
    "hv_guarded": "high_voltage_guarded_lhc",
}
PLOT_METHOD_KEYS = (
    "paper", "direct", "emu_raw", "emu_fd", "foundation_fd",
)

# Floor for log-scale current panels (measured noise floor is ~1e-12 A).
LOG_FLOOR_A = 1e-13
MARK_EVERY = 4
_BASELINE_BLOB = None

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "figure.dpi": 200,
    "figure.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 1.3,
})


def load_sims(dev: Device, n_curves: int) -> dict[str, list[np.ndarray]]:
    tag = device_tag(dev.dev_type, dev.L_um, dev.W_um)
    out = {}
    for key, cfg in METHODS.items():
        path = cfg["dir"] / f"sims_{tag}.npz"
        if key != "paper":
            record_path = cfg["dir"] / f"ml_{tag}.json"
            record = json.loads(record_path.read_text())
            if record.get("method") != EXPECTED_METHOD_IDS[key]:
                raise RuntimeError(
                    f"{tag}: {key} expected fixed method "
                    f"{EXPECTED_METHOD_IDS[key]!r}, got "
                    f"{record.get('method')!r}"
                )
            if record.get("box_mode") != "lhc10":
                raise RuntimeError(
                    f"{tag}: {key} is not from the current lhc10 setup"
                )
        saved = np.load(path)
        prefix = cfg.get("prefix", "sim_")
        out[key] = [np.asarray(saved[f"{prefix}{i}"])
                    for i in range(n_curves)]
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
    for key in (keys if keys is not None else PLOT_METHOD_KEYS):
        cfg = METHODS[key]
        handles.append(Line2D([], [], color="k", ls=cfg["ls"], label=cfg["label"]))
    return handles


def iv_by_method(devs_curves_sims):
    """One representative I-V figure per method.

    Every ML-method figure includes measured points, the paper-parameter
    NGSpice curve, and the selected method's predicted-parameter NGSpice curve.
    The paper-only figure naturally contains measured points plus paper cards.
    """
    reps = [("nmos", 8.0, 1.6), ("pmos", 2.0, 5.0)]
    for key in PLOT_METHOD_KEYS:
        cfg = METHODS[key]
        plot_keys = ["paper"] if key == "paper" else ["paper", key]
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
        letters = iter("abcd")
        for row, rep in enumerate(reps):
            dev, curves, sims = devs_curves_sims[rep]
            scores = device_scores(dev, curves, sims)
            if key == "paper":
                score_text = f"RRMS {scores['paper']['rrms']:.3f}"
            else:
                score_text = (f"RRMS paper {scores['paper']['rrms']:.3f}, "
                              f"ML {scores[key]['rrms']:.3f}")
            ax = axes[row, 0]
            plot_output(ax, curves, sims, n_biases=2, keys=plot_keys)
            ax.legend(loc="upper left")
            ax.set_title(f"({next(letters)}) {dev_title(dev)} — output "
                         f"({score_text})")
            ax = axes[row, 1]
            pairs = indexed(curves, "idvg")[-1:]
            _plot_curve_set(ax, pairs, sims, lambda c: c.Vg, "V$_{DS}$",
                            in_uA=True, log=False, keys=plot_keys)
            ax.legend(loc="upper left")
            ax.set_xlabel("|V$_{GS}$| (V)")
            ax.set_title(f"({next(letters)}) {dev_title(dev)} — transfer, "
                         f"|V$_{{DS}}$| = {abs(pairs[0][1].fixed):.2f} V")
        fig.legend(handles=method_legend_handles(keys=plot_keys),
                   loc="lower center", ncol=len(plot_keys) + 1,
                   frameon=False)
        if key == "paper":
            title = "Measured 77 K vs paper-parameter NGSpice curves"
        else:
            title = ("Measured 77 K with paper-parameter and "
                     f"{cfg['label']} curves")
        fig.suptitle(title, y=0.995)
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
        for key in (keys if keys is not None else PLOT_METHOD_KEYS):
            cfg = METHODS[key]
            ax.plot(x[keep], y(sims[key][idx])[keep], color=color,
                    ls=cfg["ls"], alpha=0.9)
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


def device_scores(dev: Device, curves, sims) -> dict[str, dict]:
    """Confirmed-setup score on the baseline's fixed curve-inclusion set."""
    global _BASELINE_BLOB
    if _BASELINE_BLOB is None:
        _BASELINE_BLOB = json.load(
            open(OUT_DIR / "pdk_baseline" / "pdk_baseline.json"))
    tag = device_tag(dev.dev_type, dev.L_um, dev.W_um)
    include = {name for name, item in _BASELINE_BLOB["per_curve"][tag].items()
               if item.get("included")}
    return {key: score_device_new(dev.dev_type, dev.L_um, dev.W_um, curves, s,
                                  include_tags=include)
            for key, s in sims.items()}


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
        scores = device_scores(dev, curves, sims)
        sub = (f"RRMS: paper {scores['paper']['rrms']:.3f} | "
               f"direct MLP {scores['direct']['rrms']:.3f}\n"
               f"surrogate {scores['emu_raw']['rrms']:.3f} -> "
               f"{scores['emu_fd']['rrms']:.3f} | foundation + FD "
               f"{scores['foundation_fd']['rrms']:.3f}")

        ax = axes[0, col0]
        plot_output(ax, curves, sims, n_biases=3)
        ax.legend(loc="upper left")
        ax.set_title(f"({next(letters)}) {dev_title(dev)} - output\n{sub}",
                     fontsize=8)

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
                  ("direct", "direct MLP"),
                  ("emu_raw", "surrogate raw"),
                  ("emu_fd", "surrogate + FD"),
                  ("foundation_fd", "foundation + FD")]
    vmax = max(r[k] for r in rows for k, _ in score_keys)

    fig, axes = plt.subplots(2, len(score_keys), figsize=(17.5, 8),
                             layout="constrained")
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
    fig.colorbar(im, ax=axes, shrink=0.85, label="RRMS (rrmsCalc metric)")
    fig.suptitle("RRMS across measured geometries at 77 K "
                 "(paper Fig. 5 analogue)")
    fig.savefig(FIGS_DIR / "fig5_rrms_heatmap.png")
    plt.close(fig)


def table6_bars(rows: list[dict]):
    """Per-device RRMS for all 18 Table-6 devices."""
    x = np.arange(len(rows))
    width = 0.15
    fig, ax = plt.subplots(figsize=(15, 5.2))
    base_mean = np.mean([r["paper_params_ngspice"] for r in rows])
    direct_mean = np.mean([r["direct"] for r in rows])
    emu_raw_mean = np.mean([r["emu_raw"] for r in rows])
    emu_fd_mean = np.mean([r["emu_fd"] for r in rows])
    foundation_mean = np.mean([r["foundation_fd"] for r in rows])
    ax.bar(x - 2 * width, [r["paper_params_ngspice"] for r in rows], width,
           label=f"paper cards as published, mean {base_mean:.3f}",
           color="#4878a8")
    ax.bar(x - width, [r["direct"] for r in rows], width,
           label=f"direct MLP forward pass, mean {direct_mean:.3f}",
           color="#8b6fad")
    ax.bar(x, [r["emu_raw"] for r in rows], width,
           label=f"surrogate raw, mean {emu_raw_mean:.3f}",
           color="#f2b880")
    ax.bar(x + width, [r["emu_fd"] for r in rows], width,
           label=f"surrogate + FD, mean {emu_fd_mean:.3f}",
           color="#e8923c")
    ax.bar(x + 2 * width, [r["foundation_fd"] for r in rows], width,
           label=f"foundation + FD, mean {foundation_mean:.3f}",
           color="#55a868")
    ax.plot(x, [r["paper_reported"] for r in rows], "kD", markersize=4,
            ls="none", label="paper reported (HSPICE flow)")
    ax.axhline(base_mean, color="#4878a8", ls=":", lw=1)
    ax.axhline(emu_fd_mean, color="#e8923c", ls=":", lw=1)
    ax.axhline(foundation_mean, color="#55a868", ls=":", lw=1)
    n_nmos = sum(r["dev_type"] == "nmos" for r in rows)
    ax.axvline(n_nmos - 0.5, color="k", lw=0.8, alpha=0.4)
    ax.set_xticks(x, [r["device"] for r in rows], rotation=60, ha="right",
                  fontsize=8)
    ax.set_ylabel("RRMS (rrmsCalc metric)")
    ax.set_title("All 18 Table-6 devices — confirmed-setup NGSpice flow "
                 "(ngspice-41, updated pFET card)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "table6_bars.png")
    plt.close(fig)


def device_appendix(devs_curves_sims):
    out_dir = FIGS_DIR / "devices"
    out_dir.mkdir(parents=True, exist_ok=True)
    for dev, curves, sims in devs_curves_sims.values():
        scores = device_scores(dev, curves, sims)
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
            f"{dev_title(dev)} - paper {scores['paper']['rrms']:.3f}; "
            f"direct MLP {scores['direct']['rrms']:.3f}; "
            f"surrogate {scores['emu_raw']['rrms']:.3f} -> "
            f"{scores['emu_fd']['rrms']:.3f}; foundation + FD "
            f"{scores['foundation_fd']['rrms']:.3f}")
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        tag = device_tag(dev.dev_type, dev.L_um, dev.W_um)
        fig.savefig(out_dir / f"{tag}.png")
        plt.close(fig)


def table6_md(devs_curves_sims):
    """Table 6 analogue: RMSE (µA), RRMS, sigma per device and method."""
    columns = [
        ("paper", "paper cards"),
        ("direct", "direct MLP"),
        ("emu_raw", "surrogate raw"),
        ("emu_fd", "surrogate + FD"),
        ("foundation_fd", "foundation + FD"),
        ("hv_guarded", "high-voltage guarded"),
    ]
    lines = [
        "# Table 6 analogue — error metrics per device",
        "",
        "Paper-reported columns come from the paper's HSPICE/Mystic flow; the",
        "NGSpice columns are computed here with the confirmed-setup rrmsCalc",
        "metric in the identical CryoPDK_Skywater130nm_ML chain (ngspice-41,",
        "updated pFET card, native bins).",
        "",
        "| device | reported RRMS | reported σ | "
        + " | ".join(f"{label} RMSE (µA) | {label} RRMS | {label} σ"
                     for _, label in columns) + " |",
        "|---|---:|---:|" + "---:|---:|---:|" * len(columns),
    ]
    means = {key: [] for key, _ in columns}
    for dev, curves, sims in devs_curves_sims.values():
        scores = device_scores(dev, curves, sims)
        tag = device_tag(dev.dev_type, dev.L_um, dev.W_um)
        cells = []
        for key, _label in columns:
            score = scores[key]
            means[key].append(score["rrms"])
            cells.extend([f"{score['rmse_uA']:.3f}",
                          f"{score['rrms']:.3f}",
                          f"{score['sigma']:.3f}"])
        lines.append(f"| {tag} | {dev.paper_rrms:.3f} | "
                     f"{dev.paper_sigma:.3f} | " + " | ".join(cells) + " |")
    mean_cells = []
    for key, _label in columns:
        mean_cells.extend(["", f"**{np.mean(means[key]):.3f}**", ""])
    lines.append("| **mean** | 0.279 | | " + " | ".join(mean_cells) + " |")
    (OUT_TABLES / "table6.md").write_text("\n".join(lines) + "\n")


def fd_polish_ablation(rows: list[dict]):
    """Persist the fixed surrogate before/after FD-polish comparison."""
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(15, 5), layout="constrained")
    raw = np.array([r["emu_raw"] for r in rows])
    polished = np.array([r["emu_fd"] for r in rows])
    for i in range(len(rows)):
        ax.plot([i, i], [raw[i], polished[i]], color="0.7", lw=0.8)
    ax.plot(x, raw, "o", mfc="none", color="#e8923c", label="before FD")
    ax.plot(x, polished, "^", color="#e8923c", label="after FD")
    ax.set_ylabel("RRMS")
    ax.set_title(f"Surrogate-search FD ablation: mean "
                 f"{raw.mean():.3f} -> {polished.mean():.3f}")
    ax.legend()
    ax.set_xticks(x, [r["device"] for r in rows], rotation=60,
                     ha="right", fontsize=8)
    fig.suptitle("Finite-difference polish ablation (one fixed surrogate "
                 "method across all devices)")
    fig.savefig(FIGS_DIR / "fd_polish_ablation.png")
    plt.close(fig)

    lines = [
        "# Finite-difference polish ablation",
        "",
        "Both columns are one fixed surrogate method across all 18 devices. Positive "
        "delta means FD polish reduced RRMS. No per-device best-of is used.",
        "",
        "| device | surrogate raw | surrogate + FD | delta |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['device']} | {row['emu_raw']:.4f} | "
            f"{row['emu_fd']:.4f} | {row['emu_raw'] - row['emu_fd']:.4f} |"
        )
    lines.extend([
        "",
        f"Surrogate mean: {np.mean([r['emu_raw'] for r in rows]):.4f} -> "
        f"{np.mean([r['emu_fd'] for r in rows]):.4f}.",
    ])
    (OUT_TABLES / "fd_polish_ablation.md").write_text("\n".join(lines) + "\n")


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
        scores = device_scores(dev, curves, sims)
        tag = device_tag(dev_type, L, W)
        rows.append({
            "device": tag,
            "dev_type": dev_type,
            "L": L,
            "W": W,
            "paper_reported": dev.paper_rrms,
            "paper_params_ngspice": scores["paper"]["rrms"],
            **{key: scores[key]["rrms"] for key in
               ("direct", "emu_raw", "emu_fd", "foundation_fd",
                "hv_guarded")},
        })

    fig2(devs_curves_sims)
    fig4(devs_curves_sims)
    iv_by_method(devs_curves_sims)
    fig5(rows)
    table6_bars(rows)
    fd_polish_ablation(rows)
    device_appendix(devs_curves_sims)
    table6_md(devs_curves_sims)
    print(f"figures written to {FIGS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
