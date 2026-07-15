#!/usr/bin/env python3
"""Plot the completed all-18 training-data scaling study.

The original grid also proposed capacity and search-start sweeps. A four-device
pilot completed those axes, but the all-18 data axis showed that emulator test
MSE improved by nearly an order of magnitude while real-NGSpice+FD RRMS stayed
flat. The remaining grid was therefore stopped by design.

This final figure uses only data configurations completed for every one of the
18 transistors. Individual device traces are faded and colored; arithmetic
all-device means are bold. Incomplete rows remain in results.csv for audit but
cannot enter a reported mean.

Outputs:
  figs/scaling_laws.png
  out/tables/scaling_summary.json
  out/tables/scaling_study.md
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cryoml.config import FIGS_DIR, OUT_DIR, OUT_TABLES  # noqa: E402
from cryoml.devices import parse_device_list  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402

plt.rcParams.update({"font.size": 9, "figure.dpi": 200, "axes.grid": True,
                     "grid.alpha": 0.3})


def power_fit(x, y):
    """Fit y = a * x^(-alpha); return alpha and fitted values."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    slope, intercept = np.polyfit(lx, ly, 1)
    return -slope, np.exp(intercept + slope * lx)


def load_rows(path: Path) -> list[dict]:
    rows = list(csv.DictReader(open(path)))
    for row in rows:
        for key in ("n_data", "n_params", "n_starts"):
            row[key] = int(row[key])
        for key in ("emu_val", "search_loss", "rrms_raw", "rrms_polished"):
            row[key] = float(row[key])
    return rows


def main() -> int:
    scaling_dir = OUT_DIR / "scaling"
    config = json.load(open(scaling_dir / "run_config.json"))
    if config.get("schema_version") != 2 or config.get("box") != "lhc10":
        raise RuntimeError("refusing to plot a stale/non-confirmed scaling run")
    rows = load_rows(scaling_dir / "results.csv")
    configured = parse_device_list(config["devices"])
    expected_devices = {
        device_tag(d.dev_type, d.L_um, d.W_um) for d in configured
    }
    if len(expected_devices) != 18:
        raise RuntimeError(
            f"final data scaling requires all 18 devices, got {len(expected_devices)}")

    by_size: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if row["sweep"] == "data":
            by_size[row["n_data"]].append(row)
    complete_sizes = []
    for size, group in sorted(by_size.items()):
        devices = {row["device"] for row in group}
        if len(group) == 18 and devices == expected_devices:
            complete_sizes.append(size)
    if len(complete_sizes) < 2:
        raise RuntimeError(
            "need at least two all-18 data configurations for scaling")

    complete = [row for row in rows
                if row["sweep"] == "data"
                and row["n_data"] in complete_sizes]
    keys = {(row["device"], row["n_data"]) for row in complete}
    if len(complete) != 18 * len(complete_sizes) or len(keys) != len(complete):
        raise RuntimeError("completed data scaling contains duplicate cells")

    devices = sorted(expected_devices)
    colors = {tag: plt.cm.tab20(index % 20)
              for index, tag in enumerate(devices)}
    means = {}
    for metric in ("emu_val", "rrms_raw", "rrms_polished"):
        means[metric] = [float(np.mean([
            row[metric] for row in complete if row["n_data"] == size
        ])) for size in complete_sizes]
    alpha, fitted = power_fit(complete_sizes, means["emu_val"])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8),
                             layout="constrained")
    individual = Line2D([], [], color=plt.cm.tab20(0), marker="o", lw=0.8,
                        alpha=0.3, label="18 individual transistors")

    ax = axes[0]
    for tag in devices:
        values = [next(row["emu_val"] for row in complete
                       if row["device"] == tag and row["n_data"] == size)
                  for size in complete_sizes]
        ax.plot(complete_sizes, values, "o-", color=colors[tag], lw=0.8,
                markersize=2.5, alpha=0.25)
    mean_line, = ax.plot(complete_sizes, means["emu_val"], "k-o", lw=3,
                         markersize=4, label="arithmetic mean of all 18")
    fit_line, = ax.plot(complete_sizes, fitted, color="k", ls=":", lw=1.3,
                        label=f"power fit, alpha={alpha:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training simulations per transistor")
    ax.set_ylabel("held-out emulator signed-log MSE")
    ax.set_title("Neural copy of NGSpice improves with data")
    ax.legend([individual, mean_line, fit_line],
              [individual.get_label(), mean_line.get_label(),
               fit_line.get_label()])

    ax = axes[1]
    for tag in devices:
        values = [next(row["rrms_polished"] for row in complete
                       if row["device"] == tag and row["n_data"] == size)
                  for size in complete_sizes]
        ax.plot(complete_sizes, values, "o-", color=colors[tag], lw=0.8,
                markersize=2.5, alpha=0.25)
    fd_line, = ax.plot(complete_sizes, means["rrms_polished"], "k-o", lw=3,
                       markersize=4, label="mean after FD")
    raw_line, = ax.plot(complete_sizes, means["rrms_raw"], color="0.35",
                        marker="s", ls="--", lw=2.2,
                        label="mean before FD")
    ax.set_xscale("log")
    ax.set_xlabel("training simulations per transistor")
    ax.set_ylabel("RRMS against measured data (real NGSpice)")
    ax.set_title("Final NGSpice fit saturates after 750 samples")
    ax.legend([individual, raw_line, fd_line],
              [individual.get_label(), raw_line.get_label(),
               fd_line.get_label()])

    fig.suptitle("All-18 training-data scaling: emulator accuracy vs physical fit")
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS_DIR / "scaling_laws.png", dpi=200)
    plt.close(fig)

    counts = Counter((row["sweep"], row["n_data"] if row["sweep"] == "data"
                      else row["arch"] if row["sweep"] == "capacity"
                      else row["n_starts"]) for row in rows)
    incomplete = {f"{sweep}:{value}": count
                  for (sweep, value), count in counts.items()
                  if count != 18}
    summary = {
        "status": "stopped after the complete all-18 data axis because lower "
                  "emulator MSE did not improve real-NGSpice+FD RRMS",
        "run_config": config,
        "n_rows_preserved_in_csv": len(rows),
        "n_devices": 18,
        "complete_all_device_data_sizes": complete_sizes,
        "complete_all_device_cells_used": len(complete),
        "incomplete_or_pilot_groups_excluded_from_all18_means": incomplete,
        "pilot_artifacts": {
            "figure": "figs/scaling_pilot4.png",
            "summary": "out/tables/scaling_pilot4_summary.json",
        },
        "arithmetic_means": {
            "emulator_validation_mse": means["emu_val"],
            "raw_ngspice_rrms": means["rrms_raw"],
            "fd_ngspice_rrms": means["rrms_polished"],
        },
        "emulator_validation_power_law_alpha": float(alpha),
        "best_fd_mean": float(min(means["rrms_polished"])),
        "best_fd_n_data": int(complete_sizes[
            int(np.argmin(means["rrms_polished"]))]),
    }
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    (OUT_TABLES / "scaling_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    lines = [
        "# All-18 training-data scaling study", "",
        "Only configurations completed for all 18 transistors enter these "
        "arithmetic means. The unfinished 10,000-sample/capacity/search rows "
        "remain in the CSV for audit and are excluded. Capacity and search "
        "results are retained only as the clearly labeled four-device pilot.",
        "", "| samples/transistor | emulator test MSE | raw NGSpice RRMS | "
        "NGSpice RRMS after FD |", "|---:|---:|---:|---:|",
    ]
    for index, size in enumerate(complete_sizes):
        lines.append(
            f"| {size} | {means['emu_val'][index]:.7f} | "
            f"{means['rrms_raw'][index]:.4f} | "
            f"{means['rrms_polished'][index]:.4f} |")
    lines += [
        "",
        f"Held-out emulator MSE improves with fitted power exponent "
        f"`{alpha:.3f}`, but the best real-NGSpice+FD mean is "
        f"`{summary['best_fd_mean']:.4f}` at only "
        f"`{summary['best_fd_n_data']}` samples/transistor. Additional data "
        "does not improve the final physical fit, so the remaining grid was "
        "stopped and compute was redirected to the fixed-method comparisons.",
    ]
    (OUT_TABLES / "scaling_study.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {FIGS_DIR / 'scaling_laws.png'}")
    print(f"wrote {OUT_TABLES / 'scaling_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
