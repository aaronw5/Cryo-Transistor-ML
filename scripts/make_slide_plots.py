#!/usr/bin/env python3
"""Build simplified comparison plots used only by the slide deck."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import make_figs as figures  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402


OUTPUT = ROOT / "slides" / "plots"
MAIN_KEYS = ("paper", "direct", "emu_fd")


def load_main_sims(device, n_curves: int) -> dict[str, list[np.ndarray]]:
    """Load only the three series permitted in the main slide plots."""
    tag = device_tag(device.dev_type, device.L_um, device.W_um)
    simulations = {}
    for key in MAIN_KEYS:
        config = figures.METHODS[key]
        if key != "paper":
            record = json.loads(
                (config["dir"] / f"ml_{tag}.json").read_text())
            expected = figures.EXPECTED_METHOD_IDS[key]
            if record.get("method") != expected:
                raise RuntimeError(
                    f"{tag}: expected {expected}, got {record.get('method')}")
        saved = np.load(config["dir"] / f"sims_{tag}.npz")
        prefix = config.get("prefix", "sim_")
        simulations[key] = [np.asarray(saved[f"{prefix}{index}"])
                            for index in range(n_curves)]
    return simulations


def representative_iv() -> None:
    representatives = [("nmos", 8.0, 1.6), ("pmos", 2.0, 5.0)]
    by_device = {(device.dev_type, device.L_um, device.W_um): device
                 for device in PAPER_DEVICES}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    letters = iter("abcd")
    for row, key in enumerate(representatives):
        device = by_device[key]
        curves = load_device_curves(device)
        sims = load_main_sims(device, len(curves))

        ax = axes[row, 0]
        figures.plot_output(
            ax, curves, sims, n_biases=2, keys=MAIN_KEYS)
        ax.legend(loc="upper left")
        ax.set_title(
            f"({next(letters)}) {figures.dev_title(device)} - output")

        ax = axes[row, 1]
        pairs = figures.indexed(curves, "idvg")[-1:]
        figures._plot_curve_set(  # noqa: SLF001
            ax, pairs, sims, lambda curve: curve.Vg, "V$_{DS}$",
            in_uA=True, log=False, keys=MAIN_KEYS)
        ax.legend(loc="upper left")
        ax.set_xlabel("|V$_{GS}$| (V)")
        ax.set_title(
            f"({next(letters)}) {figures.dev_title(device)} - transfer")

    fig.legend(
        handles=figures.method_legend_handles(keys=MAIN_KEYS),
        loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Measured, paper-card, and extracted-parameter I-V curves")
    fig.tight_layout(rect=(0, 0.045, 1, 0.98))
    fig.savefig(OUTPUT / "main_iv_comparison.png", dpi=200,
                facecolor="white", transparent=False)
    plt.close(fig)


def all_device_rrms() -> None:
    with (ROOT / "out" / "tables" / "comparison.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))

    x = np.arange(len(rows))
    series = (
        ("paper_params_ngspice", "paper cards", "#4878a8"),
        ("direct_mlp_forward_pass", "direct MLP", "#8b6fad"),
        ("surrogate_search_plus_fd", "surrogate + FD", "#e8923c"),
    )
    width = 0.24
    fig, ax = plt.subplots(figsize=(15, 5.2), layout="constrained")
    for offset, (column, label, color) in zip((-1, 0, 1), series):
        values = np.asarray([float(row[column]) for row in rows])
        ax.bar(x + offset * width, values, width,
               label=f"{label}, mean {values.mean():.3f}", color=color)
    ax.axvline(7.5, color="0.25", lw=0.9, alpha=0.6)
    ax.set_xticks(
        x, [row["device"] for row in rows], rotation=60, ha="right",
        fontsize=8)
    ax.set_ylabel("RRMS")
    ax.set_title("All 18 transistors in the confirmed NGSpice flow")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUTPUT / "main_rrms_comparison.png", dpi=200,
                facecolor="white", transparent=False)
    plt.close(fig)


def fd_improvement() -> None:
    """Show FD's paired effect for both ML initializers."""
    direct_payload = json.loads(
        (ROOT / "out" / "tables" / "direct_mlp_fd_study.json").read_text())
    surrogate_payload = json.loads(
        (ROOT / "out" / "tables" / "fd_parameter_study.json").read_text())
    direct = direct_payload["devices"]
    surrogate = surrogate_payload["devices"]
    direct_tags = [row["device"] for row in direct]
    if direct_tags != [row["device"] for row in surrogate]:
        raise RuntimeError("direct and surrogate FD reports are not aligned")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    panels = (
        (axes[0], direct, "raw_rrms", "fd_rrms", "Direct MLP", "#8b6fad"),
        (axes[1], surrogate, "surrogate_raw_rrms",
         "surrogate_paired_fd_rrms", "Surrogate search", "#e8923c"),
    )
    x = np.arange(len(direct))
    for ax, rows, before_key, after_key, title, color in panels:
        before = np.asarray([float(row[before_key]) for row in rows])
        after = np.asarray([float(row[after_key]) for row in rows])
        for index in range(len(rows)):
            ax.plot([index, index], [before[index], after[index]],
                    color="0.76", lw=0.9, zorder=1)
        ax.plot(x, before, "o", color="0.38", ms=5,
                label="before FD", zorder=2)
        ax.plot(x, after, "^", color=color, ms=6,
                label="after FD", zorder=3)
        ax.set_title(
            f"{title}\nmean RRMS {before.mean():.3f} -> {after.mean():.3f}")
        ax.set_xlabel("transistor index")
        ax.set_ylabel("RRMS")
        ax.set_xticks(x)
        ax.set_xticklabels([str(index + 1) for index in x], fontsize=7)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.suptitle("Finite-difference improvement from each fixed ML initializer")
    fig.savefig(OUTPUT / "fd_improvement_comparison.png", dpi=200,
                facecolor="white", transparent=False)
    plt.close(fig)


def build() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    representative_iv()
    all_device_rrms()
    fd_improvement()


if __name__ == "__main__":
    build()
