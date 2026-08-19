#!/usr/bin/env python3
"""Plot measured and simulated I-V shapes for failure diagnosis."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.config import OUT_DIR, OUT_TABLES, PROCESSED_DIR  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.foundation import canonical_point_order  # noqa: E402
from cryoml.metrics import curve_rrms_new, curve_trim  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402

METHODS = {
    "paper": {"label": "paper card", "linestyle": (0, (7, 3)),
              "linewidth": 1.6,
              "directory": "pdk_baseline", "prefix": "sim_"},
    "seven": {"label": "7-param FD", "linestyle": "-", "linewidth": 1.8,
              "directory": "pdk_foundation_vb_seven", "prefix": "fd_sim_",
              "summary_method": "surrogate_fd"},
    "full43": {"label": "43-param FD", "linestyle": ":",
               "linewidth": 2.2, "directory": "pdk_foundation_vb_full",
               "prefix": "fd_sim_", "summary_method": "surrogate_fd"},
    "seven_tradeoff": {"label": "7-param tradeoff", "linestyle": "-.",
                       "linewidth": 1.7,
                       "directory": "pdk_foundation_noiseaware_seven_tradeoff",
                       "prefix": "fd_sim_", "summary_method": "noise_aware_fd"},
    "seven_regional": {"label": "7-param regional RRMS",
                        "linestyle": (0, (3, 1, 1, 1)), "linewidth": 2.0,
                        "directory": "pdk_foundation_segmented_seven_rrms_extract",
                        "prefix": "fd_sim_", "summary_method": "noise_aware_fd"},
    "full43_tradeoff": {"label": "43-param tradeoff",
                        "linestyle": (0, (5, 1)), "linewidth": 2.0,
                        "directory": "pdk_foundation_noiseaware_full_tradeoff",
                        "prefix": "fd_sim_", "summary_method": "noise_aware_fd"},
    "full43_regional": {"label": "43-param regional RRMS",
                         "linestyle": (0, (1, 1)), "linewidth": 2.4,
                         "directory": "pdk_foundation_segmented_full_rrms_extract",
                         "prefix": "fd_sim_", "summary_method": "noise_aware_fd"},
    "full43_widened": {"label": "43-param widened box",
                       "linestyle": "solid", "linewidth": 2.0,
                       "directory": "pdk_foundation_widened_extract",
                       "prefix": "fd_sim_",
                       "summary_method": "noise_aware_fd"},
    "full43_widened_polish": {"label": "43-param widened + real FD",
                              "linestyle": (0, (4, 1, 1, 1)), "linewidth": 2.4,
                              "directory": "pdk_real_fd_polish",
                              "prefix": "fd_sim_", "summary_method": None},
    "full43_shape_fd": {"label": "widened + shape FD",
                        "linestyle": (0, (1, 1)), "linewidth": 2.0,
                        "directory": "pdk_real_fd_shape",
                        "prefix": "fd_sim_", "summary_method": None},
    "full43_lowdrain": {"label": "widened + low-drain constraint",
                        "linestyle": (0, (1, 1.4)), "linewidth": 2.8,
                        "directory": "pdk_foundation_lowdrain_extract",
                        "prefix": "fd_sim_",
                        "summary_method": "noise_aware_fd"},
    "seven_inverse_original": {"label": "7-param inverse (old)",
                               "linestyle": "-.", "linewidth": 1.7,
                               "directory": "pdk_foundation_vb_seven",
                               "prefix": "inverse_sim_",
                               "summary_method": "inverse_autoencoder"},
    "full43_inverse_original": {"label": "43-param inverse (old)",
                                "linestyle": ":", "linewidth": 2.1,
                                "directory": "pdk_foundation_vb_full",
                                "prefix": "inverse_sim_",
                                "summary_method": "inverse_autoencoder"},
    "seven_inverse_general": {"label": "7-param inverse (general)",
                              "linestyle": "-", "linewidth": 1.9,
                              "directory": "pdk_foundation_segmented_seven_inverse_general",
                              "prefix": "inverse_sim_",
                              "summary_method": "inverse_autoencoder"},
    "full43_inverse_general": {"label": "43-param inverse (general)",
                               "linestyle": (0, (5, 1)), "linewidth": 2.2,
                               "directory": "pdk_foundation_segmented_full_inverse_general",
                               "prefix": "inverse_sim_",
                               "summary_method": "inverse_autoencoder"},
}

DEFAULT_METHODS = (
    "paper", "seven", "full43", "seven_tradeoff", "full43_tradeoff",
    "seven_regional", "full43_regional")


def load_npz_curves(path: Path, prefix: str, count: int) -> list[np.ndarray]:
    with np.load(path) as data:
        return [np.asarray(data[f"{prefix}{index}"], dtype=np.float64)
                for index in range(count)]


def plot_device(device, baseline: dict, summaries: dict,
                out_dir: Path, method_names: tuple[str, ...]) -> Path:
    tag = device_tag(device.dev_type, device.L_um, device.W_um)
    full_path = PROCESSED_DIR / "pdk_synth_full" / f"{tag}.npz"
    with np.load(full_path, allow_pickle=True) as data:
        _order, descriptions = canonical_point_order(
            data["kinds"], data["fixeds"], data["slices"])
        slices = np.asarray(data["slices"], dtype=np.int64)
        vg = np.asarray(data["Vg"], dtype=np.float64)
        vd = np.asarray(data["Vd"], dtype=np.float64)
        measured = np.asarray(data["meas"], dtype=np.float64)
    n_curves = len(descriptions)
    available = {}
    for method in method_names:
        style = METHODS[method]
        path = OUT_DIR / style["directory"] / f"sims_{tag}.npz"
        if path.exists():
            available[method] = load_npz_curves(
                path, style["prefix"], n_curves)
    simulations = {method: {} for method in available}
    measurements, sweeps = {}, {}
    for desc in descriptions:
        source = int(desc["source_curve_index"])
        start, stop = (int(value) for value in slices[source])
        curve_tag = desc["tag"]
        measurements[curve_tag] = measured[start:stop]
        sweeps[curve_tag] = (vd[start:stop] if desc["kind"] == "idvd"
                             else vg[start:stop])
        for method, curves in available.items():
            simulations[method][curve_tag] = curves[source]

    included = {key for key, value in baseline["per_curve"][tag].items()
                if value.get("included")}
    selected = [desc for desc in descriptions
                if desc["tag"] in included and desc["bias"] >= 0.74]
    biases = sorted({desc["bias"] for desc in selected})
    colors = {bias: plt.cm.viridis(index / max(len(biases) - 1, 1))
              for index, bias in enumerate(biases)}
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), layout="constrained")

    def plot_iv(ax, kind: str, *, log: bool = False) -> None:
        for desc in selected if not log else descriptions:
            if desc["kind"] != kind or desc["tag"] not in included:
                continue
            tag_name = desc["tag"]
            color = colors.get(desc["bias"], plt.cm.viridis(0.0))
            x = np.abs(sweeps[tag_name])
            y_meas = np.abs(measurements[tag_name]) * 1e6
            if log:
                y_meas = np.maximum(y_meas, 1e-8)
            ax.plot(x, y_meas, "o", ms=2.3, mfc="none", mec=color,
                    alpha=0.75)
            for method in available:
                style = METHODS[method]
                y = np.abs(simulations[method][tag_name]) * 1e6
                if log:
                    y = np.maximum(y, 1e-8)
                ax.plot(x, y, color=color, linestyle=style["linestyle"],
                        linewidth=style["linewidth"], alpha=0.9)
        if log:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.22)
        ax.set_xlabel("|sweep voltage| (V)")
        ax.set_ylabel("|I_D| (uA)")

    def plot_residual(ax, kind: str) -> None:
        for desc in selected:
            if desc["kind"] != kind:
                continue
            tag_name = desc["tag"]
            color = colors[desc["bias"]]
            target = np.abs(measurements[tag_name])
            denominator = max(float(np.mean(target)), 1e-30)
            x = np.abs(sweeps[tag_name])
            for method in available:
                style = METHODS[method]
                residual = ((np.abs(simulations[method][tag_name]) - target)
                            / denominator)
                ax.plot(x, residual, color=color,
                        linestyle=style["linestyle"],
                        linewidth=style["linewidth"], alpha=0.9)
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.grid(True, alpha=0.22)
        ax.set_xlabel("|sweep voltage| (V)")
        ax.set_ylabel("signed residual / mean |I_meas|")

    plot_iv(axes[0, 0], "idvd")
    axes[0, 0].set_title("Output curves, |fixed bias| >= 0.74 V")
    plot_iv(axes[0, 1], "idvg")
    axes[0, 1].set_title("Transfer curves, |fixed bias| >= 0.74 V")
    plot_iv(axes[0, 2], "idvg", log=True)
    axes[0, 2].set_title("All admitted transfer curves, log scale")
    plot_residual(axes[1, 0], "idvd")
    axes[1, 0].set_title("Output-curve shape residual")
    plot_residual(axes[1, 1], "idvg")
    axes[1, 1].set_title("Transfer-curve shape residual")

    curve_rows = []
    for desc in descriptions:
        if desc["tag"] not in included:
            continue
        tag_name = desc["tag"]
        start, cleaned, _den = curve_trim(
            device.dev_type, device.L_um, device.W_um,
            desc["kind"], measurements[tag_name])
        target = cleaned[start:]
        row = {"tag": tag_name}
        for method in available:
            row[method] = curve_rrms_new(
                simulations[method][tag_name][start:], target)
        curve_rows.append(row)
    x_index = np.arange(len(curve_rows))
    width = 0.8 / len(available)
    palette = plt.get_cmap("tab10").colors
    bar_colors = {method: palette[index % len(palette)]
                  for index, method in enumerate(available)}
    for offset, method in enumerate(available):
        centered = offset - (len(available) - 1) / 2
        axes[1, 2].bar(x_index + centered * width,
                       [row[method] for row in curve_rows], width,
                       label=METHODS[method]["label"],
                       color=bar_colors[method])
    axes[1, 2].set_xticks(x_index)
    axes[1, 2].set_xticklabels([row["tag"] for row in curve_rows],
                               rotation=55, ha="right", fontsize=8)
    axes[1, 2].set_ylabel("RRMS")
    axes[1, 2].set_title("Per-curve RRMS")
    axes[1, 2].grid(True, axis="y", alpha=0.22)
    axes[1, 2].legend(fontsize=8)

    method_handles = [Line2D([0], [0], color="black",
                             linestyle=METHODS[method]["linestyle"],
                             linewidth=METHODS[method]["linewidth"],
                             label=METHODS[method]["label"])
                      for method in available]
    measured_handle = Line2D([0], [0], marker="o", mfc="none", mec="black",
                             linestyle="none", label="measurement")
    bias_handles = [Line2D([0], [0], color=colors[bias], linewidth=3,
                           label=f"fixed {bias:g} V") for bias in biases]
    fig.legend(handles=[measured_handle, *method_handles, *bias_handles],
               loc="outside lower center", ncol=4 + len(biases), fontsize=9,
               handlelength=5.0, handletextpad=0.6, columnspacing=1.4)
    paper_score = next(row["rrms"] for row in baseline["devices"]
                       if row["device"] == tag)
    scores = {"paper": paper_score}
    for method in available:
        if method == "paper" or method not in summaries:
            continue
        record = next(row for row in summaries[method]["devices"]
                      if row["device"] == tag)
        key = METHODS[method]["summary_method"]
        # Polish summaries store the score on the record itself; extraction
        # summaries nest it under a named method.
        scores[method] = (record["methods"][key]["rrms"] if key
                          else record["rrms"])
    score_text = ", ".join(
        f"{METHODS[method]['label']}={score:.4f}"
        for method, score in scores.items())
    fig.suptitle(
        f"{tag}: measured I-V shape diagnosis  |  {score_text}", fontsize=12)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", nargs="*", default=[
        "nmos_L1_W3", "pmos_L0p35_W0p55", "pmos_L0p5_W0p42",
        "pmos_L8_W0p84",
    ])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out-dir", type=Path,
                        default=OUT_DIR / "figs" / "iv_diagnostics")
    parser.add_argument(
        "--methods", default=",".join(DEFAULT_METHODS),
        help="comma-separated method keys; use inverse keys for a clean inverse overlay")
    args = parser.parse_args()
    method_names = tuple(
        value.strip() for value in args.methods.split(",") if value.strip())
    unknown = [value for value in method_names if value not in METHODS]
    if unknown:
        parser.error(f"unknown methods: {', '.join(unknown)}")
    if "paper" not in method_names:
        method_names = ("paper", *method_names)
    baseline = json.loads(
        (OUT_DIR / "pdk_baseline" / "pdk_baseline.json").read_text())
    summaries = {}
    for method in method_names:
        style = METHODS[method]
        if method == "paper":
            continue
        path = OUT_DIR / style["directory"] / "summary.json"
        if path.exists():
            summaries[method] = json.loads(path.read_text())
    wanted = {device_tag(d.dev_type, d.L_um, d.W_um): d
              for d in PAPER_DEVICES}
    tags = list(wanted) if args.all else args.devices
    for tag in tags:
        if tag not in wanted:
            raise ValueError(f"unknown device tag {tag}")
        print(plot_device(
            wanted[tag], baseline, summaries, args.out_dir, method_names))


if __name__ == "__main__":
    main()
