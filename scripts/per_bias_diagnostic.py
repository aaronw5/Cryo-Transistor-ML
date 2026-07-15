#!/usr/bin/env python3
"""Diagnose cross-bias compromises with one non-deployable fit per curve.

The main extraction intentionally produces one BSIM card per transistor. This
script answers a different question for a selected device: how well can every
fixed-bias curve be fit if it is allowed to choose its own parameter card?

For each scoreable curve, the script searches the existing 10,000-sample
Latin-hypercube dataset using the confirmed rrmsCalc preprocessing, then reruns
NGSpice at the selected seven parameters. These curve-specific cards are a
diagnostic upper bound only. They cannot be combined into a deployable model.

Default target: the pMOS L=2 um, W=5 um device whose strong-inversion output
curve exposes a clear cross-bias tradeoff.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.config import FIGS_DIR, OUT_DIR, OUT_TABLES, PROCESSED_DIR, ensure_dirs  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.metrics import curve_trim  # noqa: E402
from cryoml.pdk_extract import PARAMS7  # noqa: E402
from cryoml.spice_pdk import simulate_pdk  # noqa: E402
from cryoml.utils import device_tag, get_logger  # noqa: E402

logger = get_logger("per_bias_diagnostic")


def curve_rrms(device, curve, simulation: np.ndarray) -> float:
    start, measured, denominator = curve_trim(
        device.dev_type, device.L_um, device.W_um, curve.kind, curve.Id)
    if denominator <= 0 or len(measured) <= start:
        return float("nan")
    simulated = np.asarray(simulation, dtype=np.float64)[start:]
    measured = measured[start:]
    n = min(len(simulated), len(measured))
    if n == 0:
        return float("nan")
    return float(np.sqrt(np.mean((simulated[:n] - measured[:n]) ** 2))
                 / denominator)


def saved_curves(path: Path, prefix: str, count: int) -> list[np.ndarray]:
    with np.load(path) as saved:
        return [np.asarray(saved[f"{prefix}{index}"])
                for index in range(count)]


def aggregate(rows: list[dict], key: str) -> float:
    values = [row[key] for row in rows
              if row["included"] and np.isfinite(row[key])]
    return float(np.mean(values)) if values else float("nan")


def plot_report(device, curves, rows, simulations, published) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    styles = {
        "paper": ("--", "paper card"),
        "surrogate": ("-.", "per-device surrogate + FD"),
        "foundation": ("-", "foundation surrogate + FD"),
        "per_bias": (":", "separate card for each voltage"),
    }

    for ax, kind, x_name, bias_name in (
        (axes[0, 0], "idvd", "|V$_{DS}$| (V)", "|V$_{GS}$|"),
        (axes[0, 1], "idvg", "|V$_{GS}$| (V)", "|V$_{DS}$|"),
    ):
        selected = [index for index, row in enumerate(rows)
                    if row["kind"] == kind and row["included"]]
        colors = plt.cm.viridis(np.linspace(0.0, 0.8, len(selected)))
        for color, index in zip(colors, selected):
            curve = curves[index]
            x = np.abs(curve.Vd if kind == "idvd" else curve.Vg)
            ax.plot(x, np.abs(curve.Id) * 1e6, "o", mfc="none", mec=color,
                    ms=3.2, mew=0.7, markevery=4,
                    label=f"{bias_name} = {abs(curve.fixed):.2f} V")
            for method, (line_style, _label) in styles.items():
                ax.plot(x, np.abs(simulations[method][index]) * 1e6,
                        color=color, ls=line_style, lw=1.4)
        ax.set_xlabel(x_name)
        ax.set_ylabel("|I$_D$| (uA)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="best")
        ax.set_title("Output curves" if kind == "idvd" else
                     "Transfer curves")

    included = [row for row in rows if row["included"]]
    x = np.arange(len(included))
    width = 0.2
    bar_methods = (
        ("paper_rrms", "paper", "#4c78a8"),
        ("surrogate_rrms", "per-device surrogate + FD", "#9c78bd"),
        ("foundation_rrms", "foundation + FD", "#e08b43"),
        ("per_bias_rrms", "separate voltage cards", "#55a868"),
    )
    for offset, (key, label, color) in enumerate(bar_methods):
        axes[1, 0].bar(x + (offset - 1.5) * width,
                       [row[key] for row in included], width,
                       label=label, color=color)
    axes[1, 0].set_xticks(
        x, [f"{row['kind']}@{row['bias_v']:.2f}" for row in included],
        rotation=50, ha="right", fontsize=7)
    axes[1, 0].set_ylabel("per-curve RRMS")
    axes[1, 0].set_title("Bias-specific error exposes the one-card tradeoff")
    axes[1, 0].grid(axis="y", alpha=0.25)
    axes[1, 0].legend(fontsize=7)

    movement = np.full((len(PARAMS7), len(included)), np.nan)
    for column, row in enumerate(included):
        params = row["per_bias_params"]
        for parameter_index, parameter in enumerate(PARAMS7):
            scale = max(abs(published[parameter]), 1e-30)
            movement[parameter_index, column] = (
                100.0 * (params[parameter] - published[parameter]) / scale)
    limit = max(1.0, float(np.nanmax(np.abs(movement))))
    image = axes[1, 1].imshow(movement, aspect="auto", cmap="coolwarm",
                              vmin=-limit, vmax=limit)
    axes[1, 1].set_yticks(np.arange(len(PARAMS7)), PARAMS7)
    axes[1, 1].set_xticks(
        np.arange(len(included)),
        [f"{row['kind']}@{row['bias_v']:.2f}" for row in included],
        rotation=50, ha="right", fontsize=7)
    axes[1, 1].set_title("Curve-specific parameter movement from published")
    fig.colorbar(image, ax=axes[1, 1], label="parameter change (%)")

    method_handles = [
        Line2D([], [], marker="o", mfc="none", mec="k", ls="none",
               label="measured 77 K")
    ] + [
        Line2D([], [], color="k", ls=line_style, label=label)
        for line_style, label in styles.values()
    ]
    fig.legend(handles=method_handles, loc="outside lower center", ncol=5,
               frameon=False, fontsize=8)
    fig.suptitle(
        f"Non-deployable per-voltage diagnostic: {device.dev_type} "
        f"L={device.L_um:g} um, W={device.W_um:g} um")
    fig.savefig(FIGS_DIR / "per_bias_pmos_L2_W5.png", dpi=180,
                facecolor="white", transparent=False)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-type", default="pmos", choices=("nmos", "pmos"))
    parser.add_argument("--length", type=float, default=2.0)
    parser.add_argument("--width", type=float, default=5.0)
    args = parser.parse_args()

    ensure_dirs()
    device = next(
        candidate for candidate in PAPER_DEVICES
        if candidate.dev_type == args.dev_type
        and np.isclose(candidate.L_um, args.length)
        and np.isclose(candidate.W_um, args.width)
    )
    tag = device_tag(device.dev_type, device.L_um, device.W_um)
    curves = load_device_curves(device)
    synth_path = PROCESSED_DIR / "pdk_synth" / f"{tag}.npz"
    with np.load(synth_path, allow_pickle=True) as data:
        ids = np.asarray(data["IDS"])
        theta = np.asarray(data["THETA"])
        slices = np.asarray(data["slices"])
        published = {parameter: float(value)
                     for parameter, value in zip(PARAMS7, data["published"])}
        bin_index = int(data["bin_index"])

    paper = saved_curves(
        OUT_DIR / "pdk_baseline" / f"sims_{tag}.npz", "sim_", len(curves))
    surrogate = saved_curves(
        OUT_DIR / "pdk_ml_emu" / f"sims_{tag}.npz", "sim_", len(curves))
    foundation = saved_curves(
        OUT_DIR / "pdk_foundation_emu" / f"sims_{tag}.npz", "fd_sim_",
        len(curves))
    foundation_record = json.loads(
        (OUT_DIR / "pdk_foundation_emu" / f"ml_{tag}.json").read_text())
    include_tags = set(foundation_record["include_tags"])

    rows = []
    per_bias = [np.full_like(curve.Id, np.nan, dtype=np.float64)
                for curve in curves]
    started = time.time()
    for index, curve in enumerate(curves):
        start, measured, denominator = curve_trim(
            device.dev_type, device.L_um, device.W_um, curve.kind, curve.Id)
        metric_tag = f"{curve.kind}@{abs(curve.fixed):g}"
        row = {
            "index": index,
            "kind": curve.kind,
            "bias_v": float(abs(curve.fixed)),
            "metric_tag": metric_tag,
            "included": metric_tag in include_tags,
            "paper_rrms": curve_rrms(device, curve, paper[index]),
            "surrogate_rrms": curve_rrms(device, curve, surrogate[index]),
            "foundation_rrms": curve_rrms(device, curve, foundation[index]),
            "per_bias_rrms_dataset": None,
            "per_bias_rrms": None,
            "per_bias_dataset_index": None,
            "per_bias_params": None,
        }
        if denominator > 0 and len(measured) > start:
            begin, end = map(int, slices[index])
            candidate_curves = ids[:, begin + start:end]
            target = measured[start:]
            candidate_rrms = (
                np.sqrt(np.mean((candidate_curves - target) ** 2, axis=1))
                / denominator)
            winner = int(np.nanargmin(candidate_rrms))
            params = {parameter: float(theta[winner, parameter_index])
                      for parameter_index, parameter in enumerate(PARAMS7)}
            verified = simulate_pdk(
                device.dev_type, device.L_um, device.W_um, curves,
                params=params, bin_index=bin_index)
            per_bias[index] = np.asarray(verified[index])
            row.update({
                "per_bias_rrms_dataset": float(candidate_rrms[winner]),
                "per_bias_rrms": curve_rrms(
                    device, curve, per_bias[index]),
                "per_bias_dataset_index": winner,
                "per_bias_params": params,
            })
        rows.append(row)
        logger.info("%-10s paper=%s foundation=%s per-bias=%s",
                    metric_tag,
                    f"{row['paper_rrms']:.4f}" if np.isfinite(row["paper_rrms"])
                    else "n/a",
                    f"{row['foundation_rrms']:.4f}"
                    if np.isfinite(row["foundation_rrms"]) else "n/a",
                    f"{row['per_bias_rrms']:.4f}"
                    if row["per_bias_rrms"] is not None else "n/a")

    simulations = {
        "paper": paper,
        "surrogate": surrogate,
        "foundation": foundation,
        "per_bias": per_bias,
    }
    np.savez(
        OUT_DIR / "pdk_foundation_emu" / f"per_bias_sims_{tag}.npz",
        **{f"{method}_sim_{index}": simulation
           for method, method_curves in simulations.items()
           for index, simulation in enumerate(method_curves)})

    summary = {
        "status": "diagnostic only; one card per curve is not deployable",
        "device": tag,
        "search": "best of 10,000 saved NGSpice LHC samples per curve, then fresh NGSpice rerun",
        "box_mode": "lhc10",
        "included_curve_count": int(sum(row["included"] for row in rows)),
        "included_mean_rrms": {
            "paper": aggregate(rows, "paper_rrms"),
            "per_device_surrogate_plus_fd": aggregate(rows, "surrogate_rrms"),
            "foundation_plus_fd": aggregate(rows, "foundation_rrms"),
            "separate_voltage_cards": aggregate(rows, "per_bias_rrms"),
        },
        "runtime_s": round(time.time() - started, 2),
        "max_saved_vs_rerun_rrms_delta": float(max(
            abs(row["per_bias_rrms_dataset"] - row["per_bias_rrms"])
            for row in rows if row["per_bias_rrms"] is not None)),
    }
    report = {"summary": summary, "curves": rows}
    json_path = OUT_TABLES / "per_bias_pmos_L2_W5.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# Per-voltage diagnostic: pMOS L=2 um, W=5 um", "",
        "This diagnostic permits a different seven-parameter card for every "
        "fixed-bias curve. Each row selects the best of 10,000 saved NGSpice "
        "LHC samples, then reruns NGSpice at those parameters. The result is "
        "an upper-bound diagnostic, not a physically consistent or deployable "
        "compact model.", "",
        "| curve | in official device mean | paper | per-device surrogate + FD | "
        "foundation + FD | separate voltage card |",
        "|---|:---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def value(key):
            item = row[key]
            return f"{item:.4f}" if item is not None and np.isfinite(item) else "n/a"
        lines.append(
            f"| {row['metric_tag']} | {'yes' if row['included'] else 'no'} | "
            f"{value('paper_rrms')} | {value('surrogate_rrms')} | "
            f"{value('foundation_rrms')} | {value('per_bias_rrms')} |")
    means = summary["included_mean_rrms"]
    lines += [
        "", "## Included-Curve Mean", "",
        f"- Paper card: {means['paper']:.4f}",
        f"- Per-device surrogate + FD: "
        f"{means['per_device_surrogate_plus_fd']:.4f}",
        f"- Foundation + FD: {means['foundation_plus_fd']:.4f}",
        f"- Separate voltage cards: {means['separate_voltage_cards']:.4f}",
        "", "The difference between the one-card and separate-card results "
        "quantifies cross-bias compromise. It must not be interpreted as a "
        "deployable transistor model improvement.",
    ]
    (OUT_TABLES / "per_bias_pmos_L2_W5.md").write_text(
        "\n".join(lines) + "\n")
    plot_report(device, curves, rows, simulations, published)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
