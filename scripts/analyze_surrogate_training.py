#!/usr/bin/env python3
"""Compare held-out forward-surrogate accuracy and training histories."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import foundation_parameter_study as fps  # noqa: E402
from cryoml.foundation import (FoundationSurrogate,  # noqa: E402
                               current_from_signed_log)

SEGMENT_EDGES = (0.0, 0.2, 0.4, 0.7, 1.0)

SOURCES = {
    "seven_original": "out/pdk_foundation_vb_seven/forward_training.json",
    "seven_regional": (
        "out/pdk_foundation_segmented_seven_rrms_bounded/forward_training.json"),
    "full43_original": "out/pdk_foundation_vb_full/forward_training.json",
    "full43_regional": (
        "out/pdk_foundation_segmented_full_rrms_bounded/forward_training.json"),
}


def checkpoint_audit(root: Path, family: str, training_path: Path,
                     device: torch.device, batch_size: int = 512) -> dict:
    """Re-score one checkpoint by polarity and normalized sweep section."""
    parameter_set = "seven" if family.startswith("seven") else "full"
    cache_dir = root / f"out/foundation_cache_vb_{parameter_set}"
    metadata = json.loads((cache_dir / "metadata.json").read_text())
    checkpoint = torch.load(
        training_path.parent / "forward_model.pt", map_location=device,
        weights_only=False)
    model = FoundationSurrogate(
        metadata["n_parameters"], metadata["n_points"],
        tuple(checkpoint["hidden"])).to(device)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    _train, _validation, test = fps.split_indices(
        metadata, int(json.loads(training_path.read_text())["seed"]))
    features = np.load(cache_dir / "features.npy", mmap_mode="r")
    currents = np.load(cache_dir / "currents.npy", mmap_mode="r")
    device_ids = np.load(cache_dir / "device_ids.npy", mmap_mode="r")
    slog_mean = torch.tensor(
        np.load(cache_dir / "slog_mean.npy"), device=device)
    slog_std = torch.tensor(
        np.load(cache_dir / "slog_std.npy"), device=device)
    scales = torch.tensor(
        np.load(cache_dir / "current_scales.npy"), device=device)
    polarity = np.asarray([
        "nmos" if str(tag).startswith("nmos") else "pmos"
        for tag in metadata["devices"]])

    def rows():
        return {"sum_squared": 0.0, "count": 0}

    by_polarity = {name: rows() for name in ("nmos", "pmos")}
    by_section = [rows() for _ in range(len(SEGMENT_EDGES) - 1)]
    by_polarity_section = {
        name: [rows() for _ in range(len(SEGMENT_EDGES) - 1)]
        for name in ("nmos", "pmos")}
    by_curve_section = {
        curve["tag"]: [rows() for _ in range(len(SEGMENT_EDGES) - 1)]
        for curve in metadata["canonical_curves"]}

    with torch.no_grad():
        for offset in range(0, len(test), batch_size):
            indices = test[offset:offset + batch_size]
            ids_np = np.asarray(device_ids[indices], dtype=np.int64)
            x = torch.tensor(np.asarray(features[indices]), device=device)
            target = torch.tensor(np.asarray(currents[indices]), device=device)
            prediction = model(x)
            prediction_slog = prediction * slog_std + slog_mean
            prediction_current = current_from_signed_log(prediction_slog)
            ids = torch.tensor(ids_np, dtype=torch.long, device=device)
            squared = ((prediction_current - target)
                       / torch.clamp(scales[ids], min=1e-30)) ** 2
            squared_np = squared.detach().cpu().numpy().astype(np.float64)
            for name in ("nmos", "pmos"):
                mask = polarity[ids_np] == name
                if np.any(mask):
                    by_polarity[name]["sum_squared"] += float(
                        np.sum(squared_np[mask]))
                    by_polarity[name]["count"] += int(np.sum(mask) *
                                                       squared_np.shape[1])
            for curve in metadata["canonical_curves"]:
                length = int(curve["stop"] - curve["start"])
                for section, (left, right) in enumerate(zip(
                        SEGMENT_EDGES[:-1], SEGMENT_EDGES[1:])):
                    start = curve["start"] + int(round(left * length))
                    stop = curve["start"] + int(round(right * length))
                    stop = min(curve["stop"], max(stop, start + 1))
                    values = squared_np[:, start:stop]
                    total, count = float(np.sum(values)), int(values.size)
                    by_section[section]["sum_squared"] += total
                    by_section[section]["count"] += count
                    target_row = by_curve_section[curve["tag"]][section]
                    target_row["sum_squared"] += total
                    target_row["count"] += count
                    for name in ("nmos", "pmos"):
                        mask = polarity[ids_np] == name
                        if not np.any(mask):
                            continue
                        row = by_polarity_section[name][section]
                        row["sum_squared"] += float(np.sum(values[mask]))
                        row["count"] += int(np.sum(mask) * values.shape[1])

    def finish(row: dict) -> float:
        return row["sum_squared"] / max(row["count"], 1)

    return {
        "definition": "curve-scale-normalized linear-current MSE on held-out synthetic samples",
        "n_test": int(len(test)),
        "by_polarity": {key: finish(value)
                        for key, value in by_polarity.items()},
        "by_section": [finish(row) for row in by_section],
        "by_polarity_section": {
            key: [finish(row) for row in values]
            for key, values in by_polarity_section.items()},
        "by_curve_section": {
            key: [finish(row) for row in values]
            for key, values in by_curve_section.items()},
    }


def build_report(root: Path, device: torch.device) -> dict:
    runs = {}
    for name, relative in SOURCES.items():
        path = root / relative
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        runs[name] = {
            "path": str(path), "best_validation_loss": payload[
                "best_validation_loss"],
            "epochs_completed": payload["epochs_completed"],
            "runtime_s": payload["runtime_s"], "test": payload["test"],
            "test_by_curve": payload["test_by_curve"],
            "history": payload["history"],
            "loss_partition": payload.get("loss_partition", "uniform_points"),
        }
        checkpoint = path.parent / "forward_model.pt"
        if checkpoint.exists():
            runs[name]["checkpoint_audit"] = checkpoint_audit(
                root, name, path, device)
    comparisons = {}
    for family in ("seven", "full43"):
        original, regional = f"{family}_original", f"{family}_regional"
        if original not in runs or regional not in runs:
            continue
        comparisons[family] = {
            metric: {
                "original": runs[original]["test"][metric],
                "regional": runs[regional]["test"][metric],
                "ratio": (runs[regional]["test"][metric]
                          / runs[original]["test"][metric]),
            } for metric in ("high_linear", "ordinary_linear")
        }
    detailed = {}
    for family in ("seven", "full43"):
        original, regional = f"{family}_original", f"{family}_regional"
        if original not in runs or regional not in runs:
            continue
        old = runs[original]["checkpoint_audit"]
        new = runs[regional]["checkpoint_audit"]
        detailed[family] = {
            "by_polarity_ratio": {
                key: new["by_polarity"][key] / old["by_polarity"][key]
                for key in old["by_polarity"]},
            "by_section_ratio": [after / before for before, after in zip(
                old["by_section"], new["by_section"])],
            "by_polarity_section_ratio": {
                key: [after / before for before, after in zip(
                    old["by_polarity_section"][key],
                    new["by_polarity_section"][key])]
                for key in old["by_polarity_section"]},
        }
    return {"runs": runs, "comparisons": comparisons,
            "detailed_comparisons": detailed}


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Forward-surrogate held-out comparison", "",
        "Only high/ordinary normalized-linear MSE is compared directly across "
        "losses. The complete original and regional objectives have different "
        "definitions.", "",
        "| Surface | Metric | Original | Regional | Regional/original |",
        "|---|---|---:|---:|---:|",
    ]
    for family, metrics in report["comparisons"].items():
        for metric, row in metrics.items():
            lines.append(
                f"| {family} | {metric} | {row['original']:.6g} | "
                f"{row['regional']:.6g} | {row['ratio']:.4f} |")
    lines.extend(["", "## Per-curve normalized-linear MSE", ""])
    for family in ("seven", "full43"):
        original, regional = f"{family}_original", f"{family}_regional"
        if original not in report["runs"] or regional not in report["runs"]:
            continue
        lines.extend([f"### {family}", "",
                      "| Curve | Original | Regional | Ratio |",
                      "|---|---:|---:|---:|"])
        old = report["runs"][original]["test_by_curve"]
        new = report["runs"][regional]["test_by_curve"]
        for curve in old:
            before = old[curve]["normalized_linear_mse"]
            after = new[curve]["normalized_linear_mse"]
            lines.append(
                f"| {curve} | {before:.6g} | {after:.6g} | "
                f"{after / before:.4f} |")
        lines.append("")
    lines.extend([
        "## Held-out normalized-linear MSE by polarity and sweep section",
        "",
    ])
    for family, comparison in report["detailed_comparisons"].items():
        old = report["runs"][f"{family}_original"]["checkpoint_audit"]
        new = report["runs"][f"{family}_regional"]["checkpoint_audit"]
        lines.extend([f"### {family}", "",
                      "| Partition | Original | Regional | Ratio |",
                      "|---|---:|---:|---:|"])
        for polarity in ("nmos", "pmos"):
            lines.append(
                f"| {polarity} | {old['by_polarity'][polarity]:.6g} | "
                f"{new['by_polarity'][polarity]:.6g} | "
                f"{comparison['by_polarity_ratio'][polarity]:.4f} |")
        for index, (before, after, ratio) in enumerate(zip(
                old["by_section"], new["by_section"],
                comparison["by_section_ratio"]), start=1):
            lines.append(
                f"| normalized sweep section {index} | {before:.6g} | "
                f"{after:.6g} | {ratio:.4f} |")
        for polarity in ("nmos", "pmos"):
            for index, (before, after, ratio) in enumerate(zip(
                    old["by_polarity_section"][polarity],
                    new["by_polarity_section"][polarity],
                    comparison["by_polarity_section_ratio"][polarity]),
                    start=1):
                lines.append(
                    f"| {polarity} section {index} | {before:.6g} | "
                    f"{after:.6g} | {ratio:.4f} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def make_plot(report: dict, path: Path) -> None:
    runs = report["runs"]
    families = [family for family in ("seven", "full43")
                if f"{family}_original" in runs
                and f"{family}_regional" in runs]
    if not families:
        return
    fig, axes = plt.subplots(
        len(families), 3, figsize=(18, 4.5 * len(families)),
        squeeze=False, layout="constrained")
    for row_index, family in enumerate(families):
        old = runs[f"{family}_original"]
        new = runs[f"{family}_regional"]
        curves = list(old["test_by_curve"])
        x = np.arange(len(curves))
        axes[row_index, 0].semilogy(
            x, [old["test_by_curve"][curve]["normalized_linear_mse"]
                for curve in curves], "o-", label="original")
        axes[row_index, 0].semilogy(
            x, [new["test_by_curve"][curve]["normalized_linear_mse"]
                for curve in curves], "o-", label="regional RRMS")
        axes[row_index, 0].set_xticks(x, curves, rotation=40, ha="right")
        axes[row_index, 0].set_ylabel("held-out normalized-linear MSE")
        axes[row_index, 0].set_title(f"{family}: per-curve forward accuracy")
        axes[row_index, 0].grid(True, which="both", alpha=0.25)
        axes[row_index, 0].legend()

        history = new["history"]
        epoch = [item["epoch"] for item in history]
        axes[row_index, 1].semilogy(
            epoch, [item["validation_loss"] for item in history],
            label="section-balanced validation")
        axes[row_index, 1].semilogy(
            epoch, [item["validation_high_linear"] for item in history],
            label="high linear MSE")
        axes[row_index, 1].semilogy(
            epoch, [item["validation_ordinary_linear"] for item in history],
            label="ordinary linear MSE")
        axes[row_index, 1].set_xlabel("epoch")
        axes[row_index, 1].set_ylabel("loss / MSE")
        axes[row_index, 1].set_title(f"{family}: regional training history")
        axes[row_index, 1].grid(True, which="both", alpha=0.25)
        axes[row_index, 1].legend()

        comparison = report["detailed_comparisons"][family]
        ratios = list(comparison["by_section_ratio"])
        labels = [f"all S{index}" for index in range(1, 5)]
        for polarity in ("nmos", "pmos"):
            ratios.extend(comparison["by_polarity_section_ratio"][polarity])
            labels.extend(
                [f"{polarity[0]} S{index}" for index in range(1, 5)])
        colors = ["#4c78a8" if value <= 1 else "#e45756"
                  for value in ratios]
        axes[row_index, 2].bar(np.arange(len(ratios)), ratios, color=colors)
        axes[row_index, 2].axhline(1.0, color="black", linestyle="--",
                                   linewidth=1)
        axes[row_index, 2].set_xticks(
            np.arange(len(ratios)), labels, rotation=45, ha="right")
        axes[row_index, 2].set_ylabel("regional / original linear MSE")
        axes[row_index, 2].set_title(
            f"{family}: held-out sweep-section error ratio")
        axes[row_index, 2].grid(True, axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--json", type=Path,
        default=Path("out/tables/surrogate_training_comparison.json"))
    parser.add_argument(
        "--markdown", type=Path,
        default=Path("out/tables/surrogate_training_comparison.md"))
    parser.add_argument(
        "--plot", type=Path,
        default=Path("out/figs/foundation_comparison/surrogate_training.png"))
    parser.add_argument("--device", default="cpu",
                        help="torch device for post-hoc checkpoint audit")
    args = parser.parse_args()
    report = build_report(args.root, torch.device(args.device))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.markdown)
    make_plot(report, args.plot)
    print(json.dumps(report["comparisons"], indent=2))


if __name__ == "__main__":
    main()
