#!/usr/bin/env python3
"""Voltage-balanced foundation study for seven and expanded parameter sets.

One conditional forward surrogate is trained per parameter-set experiment.
The curve layout, architecture, loss, data split, and reporting policy are the
same for the original seven parameters and the sensitivity-selected maximum
surface.  Two extraction methods are then evaluated:

1. two-point finite-difference least squares on the frozen surrogate;
2. a direct I-V-to-parameter encoder trained with parameter supervision and
   reconstruction through the frozen surrogate (autoencoder-style).

Every extracted vector is re-simulated and scored in real NGSpice.  The paper
RRMS is never redefined; a separate fixed voltage-balanced selection objective
prevents the low/mid-voltage curves from hiding 1.48/1.85 V errors.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import least_squares
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pdk_ml_extract as mlx  # noqa: E402
from cryoml.config import OUT_DIR, PROCESSED_DIR, ensure_dirs  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES, parse_device_list  # noqa: E402
from cryoml.foundation import (FoundationSurrogate, InverseAutoencoder,  # noqa: E402
                               canonical_point_order,
                               current_from_signed_log, geometry_features,
                               geometry_ranges, high_voltage_point_mask,
                               hybrid_forward_loss, segmented_hybrid_loss,
                               signed_log,
                               voltage_point_weights)
from cryoml.metrics import (clean_current, curve_trim, device_rrms_new,  # noqa: E402
                            family_totals, score_device_new)
from cryoml.parameter_sets import PARAMS7  # noqa: E402
from cryoml.parameter_space import ParameterBox  # noqa: E402
from cryoml.spice_pdk import simulate_pdk  # noqa: E402
from cryoml.utils import device_tag, get_logger, set_seed  # noqa: E402

logger = get_logger("foundation_parameter_study")
HIGH_VOLTAGE_FLOOR = 1.48


def default_paths(parameter_set: str) -> tuple[Path, Path, Path]:
    if parameter_set == "seven":
        return (PROCESSED_DIR / "pdk_synth",
                OUT_DIR / "foundation_cache_vb_seven",
                OUT_DIR / "pdk_foundation_vb_seven")
    return (PROCESSED_DIR / "pdk_synth_full",
            OUT_DIR / "foundation_cache_vb_full",
            OUT_DIR / "pdk_foundation_vb_full")


def box_from_dataset(data, parameter_set: str) -> ParameterBox:
    if parameter_set == "full":
        names = tuple(str(value) for value in data["parameter_names"])
        return ParameterBox(
            names=names,
            published={name: float(value)
                       for name, value in zip(names, data["published"])},
            bounds={name: [float(lo), float(hi)]
                    for name, lo, hi in zip(names, data["lo"], data["hi"])})
    names = tuple(PARAMS7)
    published = {name: float(value)
                 for name, value in zip(names, data["published"])}
    values = np.asarray([published[name] for name in names], dtype=np.float64)
    lo = np.minimum(0.9 * values, 1.1 * values)
    hi = np.maximum(0.9 * values, 1.1 * values)
    return ParameterBox(
        names=names, published=published,
        bounds={name: [float(lo[index]), float(hi[index])]
                for index, name in enumerate(names)})


def local_from_dataset(data, box: ParameterBox,
                       parameter_set: str) -> np.ndarray:
    if parameter_set == "full":
        return np.asarray(data["LOCAL"], dtype=np.float64)
    theta = np.asarray(data["THETA"], dtype=np.float64)
    local = np.zeros_like(theta)
    local[:, box.active] = (
        2.0 * (theta[:, box.active] - box.lo[box.active])
        / (box.hi[box.active] - box.lo[box.active]) - 1.0)
    return np.clip(local, -1.0, 1.0)


def _safe_remove_cache(cache_dir: Path) -> None:
    resolved = cache_dir.resolve()
    out_root = OUT_DIR.resolve()
    if not resolved.is_relative_to(out_root):
        raise RuntimeError(f"refusing to remove cache outside {out_root}")
    if cache_dir.is_symlink():
        raise RuntimeError("refusing to recursively remove a symlinked cache")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def build_cache(data_dir: Path, cache_dir: Path, parameter_set: str,
                *, rebuild: bool = False,
                emphasize_high_voltage: bool = False) -> dict:
    """Canonicalize curves and materialize uncompressed float32 memmaps."""
    metadata_path = cache_dir / "metadata.json"
    if metadata_path.exists() and not rebuild:
        metadata = json.loads(metadata_path.read_text())
        if (metadata.get("schema_version") == 3
                and metadata.get("parameter_set") == parameter_set
                and metadata.get("emphasize_high_voltage", False)
                == emphasize_high_voltage
                and metadata.get("devices")
                == [device_tag(device.dev_type, device.L_um, device.W_um)
                    for device in PAPER_DEVICES]):
            return metadata
        raise RuntimeError(
            f"incompatible cache at {cache_dir}; pass --rebuild-cache")
    if rebuild:
        _safe_remove_cache(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ranges = geometry_ranges(PAPER_DEVICES)
    descriptions, n_total = [], 0
    parameter_names = None
    n_points = None
    canonical_curves = None
    boxes = {}
    scales_by_device = []
    active_by_device = []

    # First pass: validate layouts, count rows, and estimate linear scales.
    for device_index, device in enumerate(PAPER_DEVICES):
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        path = data_dir / f"{tag}.npz"
        if not path.exists():
            raise RuntimeError(f"missing dataset {path}")
        with np.load(path, allow_pickle=True) as data:
            box = box_from_dataset(data, parameter_set)
            if parameter_names is None:
                parameter_names = box.names
            elif parameter_names != box.names:
                raise RuntimeError(f"{tag}: parameter layout differs")
            valid = np.flatnonzero(data["ok"])
            order, curves = canonical_point_order(
                data["kinds"], data["fixeds"], data["slices"])
            if canonical_curves is None:
                canonical_curves = [
                    {key: value for key, value in curve.items()
                     if key != "source_curve_index"}
                    for curve in curves]
                n_points = len(order)
            elif len(order) != n_points:
                raise RuntimeError(f"{tag}: I-V point count differs")
            ids = np.asarray(data["IDS"][valid], dtype=np.float32)[:, order]
            scales = np.zeros(n_points, dtype=np.float32)
            for curve in curves:
                block = np.abs(ids[:, curve["start"]:curve["stop"]])
                scale = max(float(np.quantile(block, 0.99)), 1e-15)
                scales[curve["start"]:curve["stop"]] = scale
            device_floor = max(1e-15, 1e-6 * float(np.max(scales)))
            scales = np.maximum(scales, device_floor)
            scales_by_device.append(scales)
            active_by_device.append(box.active.astype(bool))
            boxes[tag] = {
                "published": box.center.tolist(), "lo": box.lo.tolist(),
                "hi": box.hi.tolist(), "active": box.active.tolist(),
                "bin_index": int(data["bin_index"]),
            }
            descriptions.append({
                "device": tag, "device_index": device_index,
                "start": n_total, "stop": n_total + len(valid),
                "n": len(valid), "source_path": str(path),
            })
            n_total += len(valid)
    if parameter_names is None or n_points is None or canonical_curves is None:
        raise RuntimeError("no training data found")

    features = np.lib.format.open_memmap(
        cache_dir / "features.npy", mode="w+", dtype=np.float32,
        shape=(n_total, len(parameter_names) + 3))
    currents = np.lib.format.open_memmap(
        cache_dir / "currents.npy", mode="w+", dtype=np.float32,
        shape=(n_total, n_points))
    device_ids = np.lib.format.open_memmap(
        cache_dir / "device_ids.npy", mode="w+", dtype=np.uint8,
        shape=(n_total,))

    cursor = 0
    for device_index, device in enumerate(PAPER_DEVICES):
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        with np.load(data_dir / f"{tag}.npz", allow_pickle=True) as data:
            valid = np.flatnonzero(data["ok"])
            box = box_from_dataset(data, parameter_set)
            local = local_from_dataset(data, box, parameter_set)[valid]
            order, _curves = canonical_point_order(
                data["kinds"], data["fixeds"], data["slices"])
            stop = cursor + len(valid)
            features[cursor:stop, :len(parameter_names)] = local.astype(np.float32)
            features[cursor:stop, len(parameter_names):] = geometry_features(
                device, ranges)
            currents[cursor:stop] = np.asarray(
                data["IDS"][valid], dtype=np.float32)[:, order]
            device_ids[cursor:stop] = device_index
            cursor = stop
    features.flush()
    currents.flush()
    device_ids.flush()

    # Per-output standardization in signed-log space.
    sum_values = np.zeros(n_points, dtype=np.float64)
    sum_squares = np.zeros(n_points, dtype=np.float64)
    chunk_size = 1024
    for start in range(0, n_total, chunk_size):
        block = signed_log(np.asarray(
            currents[start:start + chunk_size], dtype=np.float64))
        sum_values += np.sum(block, axis=0)
        sum_squares += np.sum(block * block, axis=0)
    slog_mean = sum_values / n_total
    variance = np.maximum(sum_squares / n_total - slog_mean * slog_mean, 0.0)
    slog_std = np.sqrt(variance)
    slog_std[slog_std < 1e-4] = 1.0

    weights = voltage_point_weights(
        canonical_curves, n_points,
        emphasize_high_voltage=emphasize_high_voltage)
    high_mask = high_voltage_point_mask(canonical_curves, n_points)
    np.save(cache_dir / "slog_mean.npy", slog_mean.astype(np.float32))
    np.save(cache_dir / "slog_std.npy", slog_std.astype(np.float32))
    np.save(cache_dir / "current_scales.npy",
            np.stack(scales_by_device).astype(np.float32))
    np.save(cache_dir / "point_weights.npy",
            np.broadcast_to(weights, (len(PAPER_DEVICES), n_points)).copy())
    np.save(cache_dir / "high_voltage_mask.npy", high_mask)
    np.save(cache_dir / "active_masks.npy", np.stack(active_by_device))

    metadata = {
        "schema_version": 3, "parameter_set": parameter_set,
        "source_data_dir": str(data_dir),
        "upstream_reference": {
            "documented_pinned":
            "39b1e518e25120104225b8fa19f4cfc61a6766b3",
            "audited_current_master":
            "c1c87615cd20de76a1759067398c0037d8565b3b",
        },
        "parameter_names": list(parameter_names),
        "n_parameters": len(parameter_names), "n_points": n_points,
        "n_examples": n_total,
        "devices": [row["device"] for row in descriptions],
        "device_ranges": descriptions, "geometry_ranges": ranges,
        "canonical_curves": canonical_curves, "boxes": boxes,
        "target": "per-point standardized signed-log drain current",
        "loss": ("standardized-slog MSE plus device-floor-normalized "
                 "linear-current Huber loss"),
        "current_scale_floor": "max(1e-15 A, 1e-6 * device peak curve scale)",
        "emphasize_high_voltage": bool(emphasize_high_voltage),
        "voltage_weights": ({"ordinary": 1.0, "1.48V": 2.0, "1.85V": 4.0}
                            if emphasize_high_voltage else
                            {"ordinary": 1.0, "1.48V": 1.0, "1.85V": 1.0}),
        "cache_build_runtime_s": round(time.time() - t0, 1),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    del features, currents, device_ids
    return metadata


def split_indices(metadata: dict, seed: int,
                  val_fraction: float = 0.10,
                  test_fraction: float = 0.10):
    train, validation, test = [], [], []
    for device_index, desc in enumerate(metadata["device_ranges"]):
        indices = np.arange(desc["start"], desc["stop"], dtype=np.int64)
        rng = np.random.default_rng(seed + device_index)
        rng.shuffle(indices)
        n_validation = max(1, int(round(len(indices) * val_fraction)))
        n_test = max(1, int(round(len(indices) * test_fraction)))
        validation.append(indices[:n_validation])
        test.append(indices[n_validation:n_validation + n_test])
        train.append(indices[n_validation + n_test:])
    return (np.concatenate(train), np.concatenate(validation),
            np.concatenate(test))


class FoundationDataset(Dataset):
    def __init__(self, cache_dir: Path, indices: np.ndarray):
        self.features = np.load(cache_dir / "features.npy", mmap_mode="r")
        self.currents = np.load(cache_dir / "currents.npy", mmap_mode="r")
        self.device_ids = np.load(cache_dir / "device_ids.npy", mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        return (np.array(self.features[index], copy=True),
                np.array(self.currents[index], copy=True),
                int(self.device_ids[index]))


@dataclass
class TrainingTensors:
    slog_mean: torch.Tensor
    slog_std: torch.Tensor
    current_scales: torch.Tensor
    point_weights: torch.Tensor
    high_mask: torch.Tensor
    active_masks: torch.Tensor


def training_tensors(cache_dir: Path, device: torch.device) -> TrainingTensors:
    def tensor(name, dtype=torch.float32):
        return torch.tensor(np.load(cache_dir / name), dtype=dtype, device=device)
    return TrainingTensors(
        slog_mean=tensor("slog_mean.npy"),
        slog_std=tensor("slog_std.npy"),
        current_scales=tensor("current_scales.npy"),
        point_weights=tensor("point_weights.npy"),
        high_mask=tensor("high_voltage_mask.npy", torch.bool),
        active_masks=tensor("active_masks.npy", torch.bool),
    )


def target_normalized(current: torch.Tensor,
                      tensors: TrainingTensors) -> torch.Tensor:
    return (signed_log(current) - tensors.slog_mean) / tensors.slog_std


def inverse_iv_features(inverse: InverseAutoencoder,
                        normalized_slog: torch.Tensor,
                        current: torch.Tensor,
                        device_ids: torch.Tensor,
                        tensors: TrainingTensors) -> torch.Tensor:
    """Build the input representation expected by an inverse checkpoint.

    Existing checkpoints consume standardized signed-log current.  The
    controlled dual-channel ablation prepends curve-scale-normalized linear
    current, preserving strong-current shape without discarding subthreshold
    information.  The representation is inferred from the model input width
    so old checkpoints remain usable.
    """
    n_points = normalized_slog.shape[1]
    expected = inverse.n_iv_features
    if expected == n_points:
        return normalized_slog
    if expected == 2 * n_points:
        scale = torch.clamp(tensors.current_scales[device_ids], min=1e-30)
        linear = torch.clamp(current / scale, min=-4.0, max=4.0)
        return torch.cat([linear, normalized_slog], dim=1)
    raise ValueError(
        f"inverse expects {expected} IV features for {n_points} points")


def forward_training_loss(prediction: torch.Tensor,
                          normalized: torch.Tensor,
                          current: torch.Tensor,
                          device_id: torch.Tensor,
                          tensors: TrainingTensors,
                          linear_weight: float,
                          curve_descriptions: list[dict] | None = None):
    if curve_descriptions is None:
        return hybrid_forward_loss(
            prediction, normalized, current, device_id,
            tensors.slog_mean, tensors.slog_std,
            tensors.current_scales, tensors.point_weights,
            linear_weight=linear_weight)
    return segmented_hybrid_loss(
        prediction, normalized, current, device_id,
        tensors.slog_mean, tensors.slog_std,
        tensors.current_scales, curve_descriptions,
        linear_weight=linear_weight)


def evaluate_forward(model, loader, tensors: TrainingTensors,
                     device: torch.device, linear_weight: float,
                     curve_descriptions: list[dict] | None = None) -> dict:
    totals = {"loss": 0.0, "slog": 0.0, "linear": 0.0,
              "segment_rrms": 0.0,
              "high_linear": 0.0, "ordinary_linear": 0.0}
    count = 0
    model.eval()
    with torch.no_grad():
        for features, current, device_id in loader:
            features = features.to(device)
            current = current.to(device)
            device_id = device_id.to(device, dtype=torch.long)
            normalized = target_normalized(current, tensors)
            prediction = model(features)
            loss, components = forward_training_loss(
                prediction, normalized, current, device_id, tensors,
                linear_weight, curve_descriptions)
            prediction_slog = (
                prediction * tensors.slog_std + tensors.slog_mean)
            prediction_current = current_from_signed_log(prediction_slog)
            scale = torch.clamp(
                tensors.current_scales[device_id], min=1e-30)
            normalized_sq = ((prediction_current - current) / scale) ** 2
            totals["loss"] += float(loss) * len(features)
            totals["slog"] += float(components["slog"]) * len(features)
            totals["linear"] += float(components["linear"]) * len(features)
            totals["segment_rrms"] += float(
                components.get("segment_rrms", 0.0)) * len(features)
            totals["high_linear"] += float(
                torch.mean(normalized_sq[:, tensors.high_mask])) * len(features)
            totals["ordinary_linear"] += float(
                torch.mean(normalized_sq[:, ~tensors.high_mask])) * len(features)
            count += len(features)
    return {key: value / max(count, 1) for key, value in totals.items()}


def evaluate_forward_by_curve(model, loader, tensors: TrainingTensors,
                              metadata: dict,
                              device: torch.device) -> dict[str, dict]:
    accumulators = {
        curve["tag"]: {"standardized_slog_mse": 0.0,
                       "normalized_linear_mse": 0.0}
        for curve in metadata["canonical_curves"]}
    count = 0
    model.eval()
    with torch.no_grad():
        for features, current, device_id in loader:
            features, current = features.to(device), current.to(device)
            device_id = device_id.to(device, dtype=torch.long)
            normalized = target_normalized(current, tensors)
            prediction = model(features)
            prediction_slog = (
                prediction * tensors.slog_std + tensors.slog_mean)
            prediction_current = current_from_signed_log(prediction_slog)
            scale = torch.clamp(tensors.current_scales[device_id], min=1e-30)
            for curve in metadata["canonical_curves"]:
                span = slice(curve["start"], curve["stop"])
                accumulators[curve["tag"]]["standardized_slog_mse"] += (
                    float(torch.mean((prediction[:, span]
                                      - normalized[:, span]) ** 2))
                    * len(features))
                accumulators[curve["tag"]]["normalized_linear_mse"] += (
                    float(torch.mean(((prediction_current[:, span]
                                      - current[:, span])
                                     / scale[:, span]) ** 2))
                    * len(features))
            count += len(features)
    for row in accumulators.values():
        for key in row:
            row[key] /= max(count, 1)
    return accumulators


def train_forward(model, cache_dir: Path, metadata: dict,
                  device: torch.device, *, epochs: int, batch_size: int,
                  learning_rate: float, weight_decay: float, patience: int,
                  linear_weight: float, seed: int,
                  split_override: tuple | None = None,
                  segmented_loss: bool = False) -> tuple[dict, tuple]:
    train_idx, validation_idx, test_idx = (
        split_indices(metadata, seed) if split_override is None
        else split_override)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        FoundationDataset(cache_dir, train_idx), batch_size=batch_size,
        shuffle=True, num_workers=0, generator=generator)
    validation_loader = DataLoader(
        FoundationDataset(cache_dir, validation_idx), batch_size=batch_size,
        shuffle=False, num_workers=0)
    test_loader = DataLoader(
        FoundationDataset(cache_dir, test_idx), batch_size=batch_size,
        shuffle=False, num_workers=0)
    tensors = training_tensors(cache_dir, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1))
    best, best_state, bad = math.inf, None, 0
    history, t0 = [], time.time()
    for epoch in range(epochs):
        model.train()
        running, n_seen = 0.0, 0
        for features, current, device_id in train_loader:
            features = features.to(device)
            current = current.to(device)
            device_id = device_id.to(device, dtype=torch.long)
            optimizer.zero_grad()
            normalized = target_normalized(current, tensors)
            prediction = model(features)
            loss, _components = forward_training_loss(
                prediction, normalized, current, device_id, tensors,
                linear_weight,
                metadata["canonical_curves"] if segmented_loss else None)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            running += float(loss.detach()) * len(features)
            n_seen += len(features)
        scheduler.step()
        validation = evaluate_forward(
            model, validation_loader, tensors, device, linear_weight,
            metadata["canonical_curves"] if segmented_loss else None)
        row = {"epoch": epoch + 1,
               "train_loss": running / max(n_seen, 1),
               "learning_rate": optimizer.param_groups[0]["lr"],
               **{f"validation_{key}": value
                  for key, value in validation.items()}}
        history.append(row)
        logger.info(
            "forward epoch %d train=%.5g val=%.5g high=%.5g",
            epoch + 1, row["train_loss"], validation["loss"],
            validation["high_linear"])
        if np.isfinite(validation["loss"]) and validation["loss"] < best - 1e-6:
            best, bad = validation["loss"], 0
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is None:
        raise RuntimeError("forward training produced no finite checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    test = evaluate_forward(
        model, test_loader, tensors, device, linear_weight,
        metadata["canonical_curves"] if segmented_loss else None)
    test_by_curve = evaluate_forward_by_curve(
        model, test_loader, tensors, metadata, device)
    report = {
        "best_validation_loss": float(best), "test": test,
        "test_by_curve": test_by_curve,
        "epochs_completed": len(history), "history": history,
        "n_train": len(train_idx), "n_validation": len(validation_idx),
        "n_test": len(test_idx), "runtime_s": round(time.time() - t0, 1),
        "loss_partition": (
            "equal_curve_and_normalized_sweep_segments_plus_worst_segment"
            if segmented_loss else "uniform_points"),
    }
    return report, (train_idx, validation_idx, test_idx)


def evaluate_inverse(inverse, forward, loader, tensors: TrainingTensors,
                     n_parameters: int, device: torch.device,
                     reconstruction_weight: float,
                     parameter_weight: float, linear_weight: float,
                     curve_descriptions: list[dict]) -> dict:
    totals = {"loss": 0.0, "parameter": 0.0, "reconstruction": 0.0}
    count = 0
    inverse.eval()
    forward.eval()
    with torch.no_grad():
        for features, current, device_id in loader:
            features, current = features.to(device), current.to(device)
            device_id = device_id.to(device, dtype=torch.long)
            target_local = features[:, :n_parameters]
            geometry = features[:, n_parameters:]
            normalized = target_normalized(current, tensors)
            iv_features = inverse_iv_features(
                inverse, normalized, current, device_id, tensors)
            prediction_local = inverse(iv_features, geometry)
            active = tensors.active_masks[device_id]
            parameter_loss = torch.sum(
                ((prediction_local - target_local) ** 2) * active) / torch.clamp(
                    torch.sum(active), min=1)
            bounded_local = torch.where(
                active, prediction_local, torch.zeros_like(prediction_local))
            reconstruction = forward(
                torch.cat([bounded_local, geometry], dim=1))
            reconstruction_loss, _ = segmented_hybrid_loss(
                reconstruction, normalized, current, device_id,
                tensors.slog_mean, tensors.slog_std,
                tensors.current_scales, curve_descriptions,
                linear_weight=linear_weight)
            loss = (parameter_weight * parameter_loss
                    + reconstruction_weight * reconstruction_loss)
            totals["loss"] += float(loss) * len(features)
            totals["parameter"] += float(parameter_loss) * len(features)
            totals["reconstruction"] += float(reconstruction_loss) * len(features)
            count += len(features)
    return {key: value / max(count, 1) for key, value in totals.items()}


def train_inverse(inverse, forward, cache_dir: Path, metadata: dict,
                  split: tuple, device: torch.device, *, epochs: int,
                  batch_size: int, learning_rate: float,
                  weight_decay: float, patience: int,
                  reconstruction_weight: float, parameter_weight: float,
                  linear_weight: float, seed: int) -> dict:
    train_idx, validation_idx, test_idx = split
    generator = torch.Generator().manual_seed(seed + 991)
    train_loader = DataLoader(
        FoundationDataset(cache_dir, train_idx), batch_size=batch_size,
        shuffle=True, num_workers=0, generator=generator)
    validation_loader = DataLoader(
        FoundationDataset(cache_dir, validation_idx), batch_size=batch_size,
        shuffle=False, num_workers=0)
    test_loader = DataLoader(
        FoundationDataset(cache_dir, test_idx), batch_size=batch_size,
        shuffle=False, num_workers=0)
    tensors = training_tensors(cache_dir, device)
    for parameter in forward.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        inverse.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1))
    best, best_state, bad = math.inf, None, 0
    history, t0 = [], time.time()
    n_parameters = metadata["n_parameters"]
    for epoch in range(epochs):
        inverse.train()
        running, n_seen = 0.0, 0
        for features, current, device_id in train_loader:
            features, current = features.to(device), current.to(device)
            device_id = device_id.to(device, dtype=torch.long)
            target_local = features[:, :n_parameters]
            geometry = features[:, n_parameters:]
            normalized = target_normalized(current, tensors)
            optimizer.zero_grad()
            iv_features = inverse_iv_features(
                inverse, normalized, current, device_id, tensors)
            prediction_local = inverse(iv_features, geometry)
            active = tensors.active_masks[device_id]
            parameter_loss = torch.sum(
                ((prediction_local - target_local) ** 2) * active) / torch.clamp(
                    torch.sum(active), min=1)
            bounded_local = torch.where(
                active, prediction_local, torch.zeros_like(prediction_local))
            reconstruction = forward(
                torch.cat([bounded_local, geometry], dim=1))
            reconstruction_loss, _ = segmented_hybrid_loss(
                reconstruction, normalized, current, device_id,
                tensors.slog_mean, tensors.slog_std,
                tensors.current_scales, metadata["canonical_curves"],
                linear_weight=linear_weight)
            loss = (parameter_weight * parameter_loss
                    + reconstruction_weight * reconstruction_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(inverse.parameters(), 10.0)
            optimizer.step()
            running += float(loss.detach()) * len(features)
            n_seen += len(features)
        scheduler.step()
        validation = evaluate_inverse(
            inverse, forward, validation_loader, tensors, n_parameters,
            device, reconstruction_weight, parameter_weight, linear_weight,
            metadata["canonical_curves"])
        row = {"epoch": epoch + 1,
               "train_loss": running / max(n_seen, 1),
               "learning_rate": optimizer.param_groups[0]["lr"],
               **{f"validation_{key}": value
                  for key, value in validation.items()}}
        history.append(row)
        logger.info(
            "inverse epoch %d train=%.5g val=%.5g param=%.5g recon=%.5g",
            epoch + 1, row["train_loss"], validation["loss"],
            validation["parameter"], validation["reconstruction"])
        if np.isfinite(validation["loss"]) and validation["loss"] < best - 1e-6:
            best, bad = validation["loss"], 0
            best_state = {key: value.detach().cpu().clone()
                          for key, value in inverse.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is None:
        raise RuntimeError("inverse training produced no finite checkpoint")
    inverse.load_state_dict(best_state)
    inverse.eval()
    test = evaluate_inverse(
        inverse, forward, test_loader, tensors, n_parameters, device,
        reconstruction_weight, parameter_weight, linear_weight,
        metadata["canonical_curves"])
    return {
        "best_validation_loss": float(best), "test": test,
        "epochs_completed": len(history), "history": history,
        "n_train": len(train_idx), "n_validation": len(validation_idx),
        "n_test": len(test_idx), "runtime_s": round(time.time() - t0, 1),
    }


def measured_finetune_batch(data_dir: Path, metadata: dict,
                            parameter_set: str,
                            tensors: TrainingTensors,
                            device: torch.device):
    currents, geometries, device_ids, scales, weights = [], [], [], [], []
    for device_index, paper_device in enumerate(PAPER_DEVICES):
        tag = device_tag(
            paper_device.dev_type, paper_device.L_um, paper_device.W_um)
        with np.load(data_dir / f"{tag}.npz", allow_pickle=True) as data:
            order, descriptions = canonical_point_order(
                data["kinds"], data["fixeds"], data["slices"])
            measured = np.asarray(data["meas"], dtype=np.float64)[order]
        objective_scale = np.ones_like(measured)
        objective_weight = np.zeros_like(measured)
        segments = []
        # Remove exact-zero-prefix glitches and use the same per-curve current
        # denominators/trims as the confirmed scorer. This avoids dividing by
        # a tiny synthetic standard deviation when measurement is outside the
        # Monte Carlo distribution.
        for desc in descriptions:
            span = slice(desc["start"], desc["stop"])
            measured[span] = clean_current(measured[span])
            start, cleaned, denominator = curve_trim(
                paper_device.dev_type, paper_device.L_um, paper_device.W_um,
                desc["kind"], measured[span])
            measured[span] = cleaned
            n_points = desc["stop"] - desc["start"] - start
            if denominator > 0 and np.isfinite(denominator) and n_points > 0:
                curve_start = desc["start"] + start
                curve_stop = desc["stop"]
                segment = slice(curve_start, curve_stop)
                objective_scale[segment] = denominator
                # Equal normalized-sweep segments make the measured
                # fine-tune sensitive to knees and tails independently.
                for left, right in zip(
                        (0.0, 0.2, 0.4, 0.7), (0.2, 0.4, 0.7, 1.0)):
                    segment_start = curve_start + int(round(left * n_points))
                    segment_stop = curve_start + int(round(right * n_points))
                    segment_stop = min(
                        curve_stop, max(segment_stop, segment_start + 1))
                    segments.append((slice(segment_start, segment_stop),
                                     segment_stop - segment_start))
        for segment, n_points in segments:
            objective_weight[segment] = 1.0 / (len(segments) * n_points)
        currents.append(measured)
        scales.append(objective_scale)
        weights.append(objective_weight)
        geometries.append(geometry_features(
            paper_device, metadata["geometry_ranges"]))
        device_ids.append(device_index)
    current_tensor = torch.tensor(
        np.stack(currents), dtype=torch.float32, device=device)
    geometry_tensor = torch.tensor(
        np.stack(geometries), dtype=torch.float32, device=device)
    ids_tensor = torch.tensor(device_ids, dtype=torch.long, device=device)
    normalized = target_normalized(current_tensor, tensors)
    scale_tensor = torch.tensor(
        np.stack(scales), dtype=torch.float32, device=device)
    weight_tensor = torch.tensor(
        np.stack(weights), dtype=torch.float32, device=device)
    return (normalized, current_tensor, geometry_tensor, ids_tensor,
            scale_tensor, weight_tensor)


def finetune_inverse_on_measurements(
        inverse: InverseAutoencoder, forward: FoundationSurrogate,
        data_dir: Path, cache_dir: Path, metadata: dict,
        parameter_set: str, device: torch.device, *, steps: int,
        learning_rate: float, prior_weight: float,
        patience: int) -> dict:
    """Global autoencoder phase on the 18 measured curves.

    No per-device search occurs: the inverse network is updated once across
    the fixed 18-curve batch, after which every card is produced by one direct
    I-V-to-parameter forward pass.
    """
    tensors = training_tensors(cache_dir, device)
    (normalized, current, geometry, device_ids,
     objective_scale, objective_weight) = measured_finetune_batch(
         data_dir, metadata, parameter_set, tensors, device)
    active = tensors.active_masks[device_ids]
    iv_features = inverse_iv_features(
        inverse, normalized, current, device_ids, tensors)
    inverse.eval()
    with torch.no_grad():
        initial = inverse(iv_features, geometry).detach()
        initial = torch.where(active, initial, torch.zeros_like(initial))
    inverse.train()
    forward.eval()
    for parameter in forward.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        inverse.parameters(), lr=learning_rate, weight_decay=1e-6)
    best, best_state, bad = math.inf, None, 0
    history, t0 = [], time.time()
    for step in range(steps):
        optimizer.zero_grad()
        predicted = inverse(iv_features, geometry)
        predicted = torch.where(active, predicted, torch.zeros_like(predicted))
        reconstruction = forward(torch.cat([predicted, geometry], dim=1))
        prediction_slog = (
            reconstruction * tensors.slog_std + tensors.slog_mean)
        prediction_current = current_from_signed_log(prediction_slog)
        normalized_error = (
            (prediction_current - current) / objective_scale)
        point_loss = nn.functional.smooth_l1_loss(
            normalized_error, torch.zeros_like(normalized_error),
            beta=0.1, reduction="none")
        reconstruction_loss = torch.sum(
            point_loss * objective_weight) / len(PAPER_DEVICES)
        prior_loss = torch.sum(
            ((predicted - initial) ** 2) * active) / torch.clamp(
                torch.sum(active), min=1)
        loss = reconstruction_loss + prior_weight * prior_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(inverse.parameters(), 10.0)
        optimizer.step()
        value = float(loss.detach())
        if value < best - 1e-8:
            best, bad = value, 0
            best_state = {key: value.detach().cpu().clone()
                          for key, value in inverse.state_dict().items()}
        else:
            bad += 1
        if step == 0 or (step + 1) % 100 == 0:
            row = {
                "step": step + 1, "loss": value,
                "reconstruction": float(reconstruction_loss.detach()),
                "prior": float(prior_loss.detach()),
            }
            history.append(row)
            logger.info(
                "measured inverse step %d loss=%.5g recon=%.5g prior=%.5g",
                step + 1, value, row["reconstruction"], row["prior"])
        if bad >= patience:
            break
    if best_state is None:
        raise RuntimeError("measured inverse fine-tuning produced no checkpoint")
    inverse.load_state_dict(best_state)
    inverse.eval()
    with torch.no_grad():
        predicted = inverse(iv_features, geometry)
        predicted = torch.where(active, predicted, torch.zeros_like(predicted))
        reconstruction = forward(torch.cat([predicted, geometry], dim=1))
        prediction_slog = (
            reconstruction * tensors.slog_std + tensors.slog_mean)
        prediction_current = current_from_signed_log(prediction_slog)
        normalized_error = (
            (prediction_current - current) / objective_scale)
        point_loss = nn.functional.smooth_l1_loss(
            normalized_error, torch.zeros_like(normalized_error),
            beta=0.1, reduction="none")
        reconstruction_loss = torch.sum(
            point_loss * objective_weight) / len(PAPER_DEVICES)
    return {
        "scope": "one global fine-tune over all 18 measured I-V inputs",
        "inference": "one direct inverse-network forward pass per card",
        "best_loss": float(best),
        "final_reconstruction": float(reconstruction_loss),
        "reconstruction_objective": (
            "uniform per-curve, scorer-trimmed normalized-current Huber"),
        "steps_completed": step + 1, "steps_requested": steps,
        "learning_rate": learning_rate, "prior_weight": prior_weight,
        "patience": patience, "history": history,
        "runtime_s": round(time.time() - t0, 1),
    }


@dataclass
class DeviceContext:
    device: object
    tag: str
    device_index: int
    curves: list
    box: ParameterBox
    bin_index: int
    include_tags: set[str]
    high_voltage_tags: set[str]
    measured_canonical: np.ndarray
    canonical_curves: list[dict]
    geometry: np.ndarray
    local_samples: np.ndarray
    current_samples: np.ndarray


def build_device_context(device, data_dir: Path, metadata: dict,
                         parameter_set: str) -> DeviceContext:
    tag = device_tag(device.dev_type, device.L_um, device.W_um)
    curves = load_device_curves(device)
    with np.load(data_dir / f"{tag}.npz", allow_pickle=True) as data:
        box = box_from_dataset(data, parameter_set)
        valid = np.flatnonzero(data["ok"])
        order, descriptions = canonical_point_order(
            data["kinds"], data["fixeds"], data["slices"])
        measured = np.asarray(data["meas"], dtype=np.float64)[order]
        local = local_from_dataset(data, box, parameter_set)[valid]
        currents = np.asarray(data["IDS"][valid], dtype=np.float32)[:, order]
        bin_index = int(data["bin_index"])
    include_tags = mlx.baseline_include_tags(tag)
    high_tags = set()
    for desc in descriptions:
        if desc["bias"] < HIGH_VOLTAGE_FLOOR - 1e-6:
            continue
        meas = measured[desc["start"]:desc["stop"]]
        _start, _cleaned, denominator = curve_trim(
            device.dev_type, device.L_um, device.W_um, desc["kind"], meas)
        if denominator > 0 and np.isfinite(denominator):
            high_tags.add(desc["tag"])
    return DeviceContext(
        device=device, tag=tag, device_index=metadata["devices"].index(tag),
        curves=curves, box=box,
        bin_index=bin_index, include_tags=include_tags,
        high_voltage_tags=high_tags, measured_canonical=measured,
        canonical_curves=descriptions,
        geometry=geometry_features(device, metadata["geometry_ranges"]),
        local_samples=local, current_samples=currents)


def tagged_canonical(vector: np.ndarray, descriptions: list[dict]) -> dict:
    vector = np.asarray(vector, dtype=np.float64)
    return {
        (desc["kind"], float(desc["bias"])):
        vector[desc["start"]:desc["stop"]]
        for desc in descriptions
    }


def score_canonical(context: DeviceContext,
                    simulated: np.ndarray) -> dict:
    tagged_sim = tagged_canonical(simulated, context.canonical_curves)
    tagged_measured = tagged_canonical(
        context.measured_canonical, context.canonical_curves)
    fixed = device_rrms_new(
        context.device.dev_type, context.device.L_um, context.device.W_um,
        tagged_sim, tagged_measured, include_tags=context.include_tags)
    high = device_rrms_new(
        context.device.dev_type, context.device.L_um, context.device.W_um,
        tagged_sim, tagged_measured,
        include_tags=context.high_voltage_tags)
    return {
        **fixed,
        "high_voltage_rrms": float(high["rrms"]),
        "high_voltage_n_curves": int(high["n_curves"]),
        "high_voltage_per_curve": high["per_curve"],
    }


def score_real(context: DeviceContext, simulations: list[np.ndarray]) -> dict:
    fixed = score_device_new(
        context.device.dev_type, context.device.L_um, context.device.W_um,
        context.curves, simulations, include_tags=context.include_tags)
    official = score_device_new(
        context.device.dev_type, context.device.L_um, context.device.W_um,
        context.curves, simulations)
    high = score_device_new(
        context.device.dev_type, context.device.L_um, context.device.W_um,
        context.curves, simulations, include_tags=context.high_voltage_tags)
    return {
        **fixed,
        "official_dynamic_rrms": float(official["rrms"]),
        "official_dynamic_n_curves": int(official["n_curves"]),
        "high_voltage_rrms": float(high["rrms"]),
        "high_voltage_n_curves": int(high["n_curves"]),
        "high_voltage_per_curve": high["per_curve"],
    }


def curve_objective_layout(context: DeviceContext,
                           include_tags: set[str] | None = None) -> list[dict]:
    include_tags = context.include_tags if include_tags is None else include_tags
    layout = []
    for desc in context.canonical_curves:
        if desc["tag"] not in include_tags:
            continue
        measured = context.measured_canonical[desc["start"]:desc["stop"]]
        start, cleaned, denominator = curve_trim(
            context.device.dev_type, context.device.L_um,
            context.device.W_um, desc["kind"], measured)
        if denominator > 0 and np.isfinite(denominator):
            layout.append({
                "start": desc["start"] + start, "stop": desc["stop"],
                "measured": cleaned[start:], "denominator": denominator,
                "tag": desc["tag"],
            })
    if not layout:
        raise RuntimeError(f"{context.tag}: no paper metric curves")
    return layout


def curve_rrms_matrix(currents: np.ndarray,
                      layout: list[dict]) -> np.ndarray:
    currents = np.asarray(currents, dtype=np.float64)
    if currents.ndim == 1:
        currents = currents[None, :]
    columns = []
    for entry in layout:
        block = currents[:, entry["start"]:entry["stop"]]
        measured = np.asarray(entry["measured"], dtype=np.float64)[None, :]
        rmse = np.sqrt(np.mean((block - measured) ** 2, axis=1))
        columns.append(rmse / entry["denominator"])
    return np.stack(columns, axis=1)


def predict_surrogate(model: FoundationSurrogate, local: np.ndarray,
                      context: DeviceContext, tensors: TrainingTensors,
                      device: torch.device) -> np.ndarray:
    local = np.asarray(local, dtype=np.float32)
    if local.ndim == 1:
        local = local[None, :]
    geometry = np.broadcast_to(context.geometry, (len(local), 3))
    features = np.concatenate([local, geometry], axis=1)
    with torch.no_grad():
        normalized = model(torch.tensor(
            features, dtype=torch.float32, device=device))
        slog = normalized * tensors.slog_std + tensors.slog_mean
        current = current_from_signed_log(slog)
    return current.detach().cpu().numpy().astype(np.float64)


def diverse_oracle_starts(context: DeviceContext, n_starts: int,
                          min_distance: float = 0.35) -> tuple[list[np.ndarray],
                                                               dict]:
    layout = curve_objective_layout(context)
    scores = np.mean(curve_rrms_matrix(context.current_samples, layout), axis=1)
    ranked = np.argsort(scores)
    selected = []
    active = context.box.active
    for index in ranked:
        candidate = np.asarray(context.local_samples[index], dtype=np.float64)
        if (not selected or not np.any(active)
                or min(np.sqrt(np.mean(
                    (candidate[active] - prior[active]) ** 2))
                       for prior in selected) >= min_distance):
            selected.append(candidate)
        if len(selected) >= n_starts:
            break
    for index in ranked:
        if len(selected) >= n_starts:
            break
        candidate = np.asarray(context.local_samples[index], dtype=np.float64)
        if not any(np.array_equal(candidate, prior) for prior in selected):
            selected.append(candidate)
    oracle_index = int(ranked[0])
    oracle_score = score_canonical(
        context, context.current_samples[oracle_index])
    return selected, {
        "sample_index": oracle_index,
        "paper_rrms": float(oracle_score["rrms"]),
        "high_voltage_rrms": float(oracle_score["high_voltage_rrms"]),
        "n_samples": int(len(scores)),
    }


def finite_difference_surrogate_extract(
        context: DeviceContext, model: FoundationSurrogate,
        tensors: TrainingTensors, device: torch.device, *, n_starts: int,
        n_validate: int, max_nfev: int, diff_step: float,
        regularization: float = 1e-4) -> tuple[dict, list]:
    """Optimize only the frozen surrogate; NGSpice validates endpoints."""
    t0 = time.time()
    layout = curve_objective_layout(context)
    starts, oracle = diverse_oracle_starts(context, n_starts)
    active = np.flatnonzero(context.box.active)
    if not len(active):
        starts = [context.box.local_published]
    solutions = []
    n_curves = len(layout)
    for start_index, local_start in enumerate(starts):
        local_start = np.asarray(local_start, dtype=np.float64)

        def residual(active_values):
            local = local_start.copy()
            local[active] = np.asarray(active_values, dtype=np.float64)
            prediction = predict_surrogate(
                model, local, context, tensors, device)[0]
            pieces = []
            for entry in layout:
                simulated = prediction[entry["start"]:entry["stop"]]
                measured = np.asarray(entry["measured"], dtype=np.float64)
                # Every curve has equal weight regardless of bias or length.
                # The data sum of squares is mean(curve_RRMS**2), while the
                # unchanged mean(curve_RRMS) paper metric is used for reports.
                scale = (entry["denominator"]
                         * math.sqrt(n_curves * max(len(measured), 1)))
                pieces.append((simulated - measured) / scale)
            data_residual = np.concatenate(pieces)
            if regularization > 0 and len(active):
                prior = (math.sqrt(regularization / len(active))
                         * np.asarray(active_values, dtype=np.float64))
                return np.concatenate([data_residual, prior])
            return data_residual

        if len(active):
            solution = least_squares(
                residual, local_start[active], method="trf", jac="2-point",
                bounds=(-np.ones(len(active)), np.ones(len(active))),
                diff_step=diff_step, max_nfev=max_nfev,
                x_scale="jac")
            local_final = local_start.copy()
            local_final[active] = solution.x
            nfev, njev = int(solution.nfev), int(solution.njev or 0)
            jacobian_norms = np.linalg.norm(
                np.asarray(solution.jac)[:sum(
                    entry["stop"] - entry["start"] for entry in layout)],
                axis=0)
        else:
            local_final, nfev, njev = local_start.copy(), 1, 0
            jacobian_norms = np.zeros(0, dtype=np.float64)
        surrogate_current = predict_surrogate(
            model, local_final, context, tensors, device)[0]
        surrogate_score = score_canonical(context, surrogate_current)
        solutions.append({
            "start_index": start_index, "candidate_kind": "fd_endpoint",
            "local": local_final,
            "surrogate_score": surrogate_score, "nfev": nfev,
            "njev": njev,
            "jacobian_column_norms": {
                context.box.names[index]: float(jacobian_norms[column])
                for column, index in enumerate(active)},
        })
    solutions.sort(key=lambda row: row["surrogate_score"]["rrms"])

    # Fresh simulator checks are deliberately outside the optimization loop.
    validations = []
    candidates = solutions[:max(1, n_validate)]
    for start_index, local_start in enumerate(starts[:max(1, n_validate)]):
        start_current = predict_surrogate(
            model, local_start, context, tensors, device)[0]
        candidates.append({
            "start_index": start_index, "candidate_kind": "lhc_start",
            "local": np.asarray(local_start, dtype=np.float64),
            "surrogate_score": score_canonical(context, start_current),
            "nfev": 0, "njev": 0, "jacobian_column_norms": {},
        })
    candidates.append({
        "start_index": -1, "candidate_kind": "published_fallback",
        "local": context.box.local_published,
        "surrogate_score": score_canonical(
            context, predict_surrogate(
                model, context.box.local_published, context, tensors,
                device)[0]),
        "nfev": 0, "njev": 0, "jacobian_column_norms": {},
    })
    seen = set()
    for row in candidates:
        key = np.asarray(row["local"], dtype=np.float64).tobytes()
        if key in seen:
            continue
        seen.add(key)
        params = context.box.local_to_params(row["local"])
        sims = simulate_pdk(
            context.device.dev_type, context.device.L_um,
            context.device.W_um, context.curves, params=params,
            bin_index=context.bin_index)
        validations.append({**row, "params": params,
                            "real_score": score_real(context, sims),
                            "simulations": sims})
    validations.sort(key=lambda row: row["real_score"]["rrms"])
    best = validations[0]
    simulations = best.pop("simulations")
    record = {
        "method": "two_point_finite_difference_on_frozen_surrogate",
        "selected_candidate_kind": best["candidate_kind"],
        "params": best["params"], "local": best["local"].tolist(),
        "rrms": float(best["real_score"]["rrms"]),
        "sigma": float(best["real_score"]["sigma"]),
        "rmse_uA": float(best["real_score"]["rmse_uA"]),
        "high_voltage_rrms": float(
            best["real_score"]["high_voltage_rrms"]),
        "official_dynamic_rrms": float(
            best["real_score"]["official_dynamic_rrms"]),
        "per_curve": best["real_score"]["per_curve"],
        "high_voltage_per_curve": best["real_score"][
            "high_voltage_per_curve"],
        "surrogate_rrms": float(best["surrogate_score"]["rrms"]),
        "surrogate_high_voltage_rrms": float(
            best["surrogate_score"]["high_voltage_rrms"]),
        "oracle_lhc": oracle, "n_starts": len(starts),
        "n_validated": len(validations), "diff_step": diff_step,
        "max_nfev": max_nfev, "regularization": regularization,
        "optimization_objective": (
            "uniform mean squared per-curve normalized point residual plus "
            "weak published-parameter regularization"),
        "best_jacobian_column_norms": best["jacobian_column_norms"],
        "validated_candidates": [{
            "start_index": row["start_index"],
            "candidate_kind": row["candidate_kind"],
            "rrms": float(row["real_score"]["rrms"]),
            "high_voltage_rrms": float(
                row["real_score"]["high_voltage_rrms"]),
            "surrogate_rrms": float(row["surrogate_score"]["rrms"]),
            "nfev": row["nfev"], "njev": row["njev"],
        } for row in validations],
        "runtime_s": round(time.time() - t0, 1),
    }
    return record, simulations


def inverse_extract(context: DeviceContext, inverse: InverseAutoencoder,
                    forward: FoundationSurrogate,
                    tensors: TrainingTensors,
                    device: torch.device) -> tuple[dict, list]:
    """Direct I-V -> parameters -> frozen-surrogate reconstruction."""
    t0 = time.time()
    normalized = (
        signed_log(context.measured_canonical).astype(np.float32)
        - tensors.slog_mean.detach().cpu().numpy()
    ) / tensors.slog_std.detach().cpu().numpy()
    with torch.no_grad():
        normalized_tensor = torch.tensor(
            normalized[None, :], dtype=torch.float32, device=device)
        current_tensor = torch.tensor(
            context.measured_canonical[None, :], dtype=torch.float32,
            device=device)
        device_ids = torch.tensor(
            [context.device_index], dtype=torch.long, device=device)
        iv_features = inverse_iv_features(
            inverse, normalized_tensor, current_tensor, device_ids, tensors)
        local = inverse(
            iv_features,
            torch.tensor(context.geometry[None, :], dtype=torch.float32,
                         device=device))[0].detach().cpu().numpy()
    local = np.asarray(local, dtype=np.float64)
    local[~context.box.active] = 0.0
    reconstruction = predict_surrogate(
        forward, local, context, tensors, device)[0]
    reconstruction_score = score_canonical(context, reconstruction)
    params = context.box.local_to_params(local)
    simulations = simulate_pdk(
        context.device.dev_type, context.device.L_um, context.device.W_um,
        context.curves, params=params, bin_index=context.bin_index)
    real = score_real(context, simulations)
    return {
        "method": "direct_iv_to_parameters_with_frozen_surrogate_reconstruction",
        "params": params, "local": local.tolist(),
        "rrms": float(real["rrms"]), "sigma": float(real["sigma"]),
        "rmse_uA": float(real["rmse_uA"]),
        "high_voltage_rrms": float(real["high_voltage_rrms"]),
        "official_dynamic_rrms": float(real["official_dynamic_rrms"]),
        "per_curve": real["per_curve"],
        "high_voltage_per_curve": real["high_voltage_per_curve"],
        "surrogate_reconstruction_rrms": float(
            reconstruction_score["rrms"]),
        "surrogate_reconstruction_high_voltage_rrms": float(
            reconstruction_score["high_voltage_rrms"]),
        "runtime_s": round(time.time() - t0, 1),
    }, simulations


def aggregate_method(records: list[dict], method: str) -> dict:
    device_results = {
        row["device"]: row["methods"][method] for row in records
    }
    totals = family_totals(device_results)
    values = [row["rrms"] for row in device_results.values()]
    high = [row["high_voltage_rrms"] for row in device_results.values()
            if np.isfinite(row["high_voltage_rrms"])]
    totals.update({
        "all_device_mean_rrms": float(np.mean(values)),
        "all_device_mean_high_voltage_rrms": (
            float(np.mean(high)) if high else float("nan")),
    })
    return totals


def write_summary(out_dir: Path, parameter_set: str, metadata: dict,
                  training: dict, inverse_training: dict,
                  records: list[dict],
                  model_sources: dict[str, str] | None = None) -> dict:
    methods = [method for method in ("surrogate_fd", "inverse_autoencoder")
               if all(method in row["methods"] for row in records)]
    summary = {
        "parameter_set": parameter_set,
        "n_parameters": metadata["n_parameters"],
        "parameter_names": metadata["parameter_names"],
        "model_sources": model_sources or {},
        "training": training, "inverse_training": inverse_training,
        "methods": {method: aggregate_method(records, method)
                    for method in methods},
        "devices": records,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    lines = [
        f"# {parameter_set}-parameter foundation extraction", "",
        "Uniform voltage weighting was used for the controlled comparison. "
        "High-voltage RRMS is reported separately and was not substituted "
        "for the paper metric.", "",
        "| method | nMOS RRMS | pMOS RRMS | combined RRMS | all-device "
        "RRMS | high-voltage RRMS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        row = summary["methods"][method]
        lines.append(
            f"| {method} | {row['nmos_rrms']:.4f} | "
            f"{row['pmos_rrms']:.4f} | {row['combined_rrms']:.4f} | "
            f"{row['all_device_mean_rrms']:.4f} | "
            f"{row['all_device_mean_high_voltage_rrms']:.4f} |")
    lines += ["", "## Per-device diagnosis", "",
              "| device | method | paper RRMS | high-voltage RRMS | "
              "surrogate RRMS |", "|---|---|---:|---:|---:|"]
    for record in records:
        for method in methods:
            row = record["methods"][method]
            surrogate_key = ("surrogate_rrms" if method == "surrogate_fd"
                             else "surrogate_reconstruction_rrms")
            lines.append(
                f"| {record['device']} | {method} | {row['rrms']:.4f} | "
                f"{row['high_voltage_rrms']:.4f} | "
                f"{row[surrogate_key]:.4f} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return summary


def save_forward_checkpoint(path: Path, model: FoundationSurrogate,
                            metadata: dict, hidden: tuple[int, ...],
                            training: dict) -> None:
    torch.save({
        "state": {key: value.detach().cpu()
                  for key, value in model.state_dict().items()},
        "hidden": list(hidden), "n_parameters": metadata["n_parameters"],
        "n_points": metadata["n_points"],
        "parameter_names": metadata["parameter_names"],
        "training": training,
    }, path)


def save_inverse_checkpoint(path: Path, model: InverseAutoencoder,
                            metadata: dict, hidden: tuple[int, ...],
                            training: dict) -> None:
    torch.save({
        "state": {key: value.detach().cpu()
                  for key, value in model.state_dict().items()},
        "hidden": list(hidden), "n_parameters": metadata["n_parameters"],
        "n_points": metadata["n_points"],
        "n_iv_features": model.n_iv_features,
        "parameter_names": metadata["parameter_names"],
        "training": training,
    }, path)


def load_checked_checkpoint(path: Path, model, metadata: dict,
                            hidden: tuple[int, ...], device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    expected = (metadata["n_parameters"], metadata["n_points"],
                metadata["parameter_names"], list(hidden))
    actual = (checkpoint["n_parameters"], checkpoint["n_points"],
              checkpoint["parameter_names"], checkpoint["hidden"])
    if actual != expected:
        raise RuntimeError(f"incompatible checkpoint {path}")
    if isinstance(model, InverseAutoencoder):
        actual_iv_features = checkpoint.get(
            "n_iv_features", checkpoint["n_points"])
        if actual_iv_features != model.n_iv_features:
            raise RuntimeError(
                f"incompatible inverse input width in checkpoint {path}: "
                f"{actual_iv_features} != {model.n_iv_features}")
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return checkpoint["training"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter-set", choices=("seven", "full"),
                        required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--stage", choices=("all", "forward", "inverse",
                                             "finetune", "extract"),
                        default="all")
    parser.add_argument("--devices", default=None,
                        help="extraction subset; training always uses all cards")
    parser.add_argument(
        "--extract-method", choices=("both", "fd", "inverse"),
        default="both", help="run both extractors or a controlled subset")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--arch", default="1024,1024,1024,1024")
    parser.add_argument("--inverse-arch", default="1024,1024,512")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--inverse-epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--inverse-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--linear-weight", type=float, default=1.0)
    parser.add_argument(
        "--segmented-loss", action="store_true",
        help="train the forward surrogate with equal x-axis section losses")
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--parameter-weight", type=float, default=1.0,
                        help="weak supervision weight for non-unique inverse parameters")
    parser.add_argument(
        "--inverse-input", choices=("signed_log", "linear_signed_log"),
        default="signed_log",
        help="controlled inverse representation ablation")
    parser.add_argument("--forward-checkpoint", default=None,
                        help="reuse a compatible frozen forward model from another run")
    parser.add_argument("--measured-steps", type=int, default=1200)
    parser.add_argument("--measured-lr", type=float, default=1e-4)
    parser.add_argument("--measured-prior-weight", type=float, default=1e-3)
    parser.add_argument("--measured-patience", type=int, default=300)
    parser.add_argument(
        "--measured-finetune", action="store_true",
        help=("optional transductive ablation on measured paper cards; "
              "disabled by default so paper devices remain evaluation-only"))
    parser.add_argument("--reset-inverse-for-finetune", action="store_true",
                        help="restart measured fine-tuning from saved synthetic checkpoint")
    parser.add_argument("--n-starts", type=int, default=8)
    parser.add_argument("--n-validate", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument("--diff-step", type=float, default=0.02)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--emphasize-high-voltage", action="store_true",
                        help="optional 1x/2x/4x training ablation")
    args = parser.parse_args()

    ensure_dirs()
    set_seed(args.seed)
    default_data, default_cache, default_out = default_paths(args.parameter_set)
    data_dir = Path(args.data_dir) if args.data_dir else default_data
    cache_dir = Path(args.cache_dir) if args.cache_dir else default_cache
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = build_cache(
        data_dir, cache_dir, args.parameter_set,
        rebuild=args.rebuild_cache,
        emphasize_high_voltage=args.emphasize_high_voltage)
    hidden = tuple(int(value) for value in args.arch.split(",") if value)
    inverse_hidden = tuple(
        int(value) for value in args.inverse_arch.split(",") if value)
    torch_device = torch.device(args.device)
    forward = FoundationSurrogate(
        metadata["n_parameters"], metadata["n_points"], hidden).to(torch_device)
    n_iv_features = metadata["n_points"] * (
        2 if args.inverse_input == "linear_signed_log" else 1)
    inverse = InverseAutoencoder(
        n_iv_features, metadata["n_parameters"],
        inverse_hidden).to(torch_device)
    forward_path = out_dir / "forward_model.pt"
    source_forward_path = (Path(args.forward_checkpoint)
                           if args.forward_checkpoint else forward_path)
    inverse_path = out_dir / "inverse_model.pt"
    inverse_synthetic_path = out_dir / "inverse_model_synthetic.pt"
    forward_report_path = out_dir / "forward_training.json"
    inverse_report_path = out_dir / "inverse_training.json"

    split = split_indices(metadata, args.seed)
    if (args.resume and source_forward_path.exists()
            and forward_report_path.exists()):
        forward_training = load_checked_checkpoint(
            source_forward_path, forward, metadata, hidden, torch_device)
        logger.info("reusing %s", source_forward_path)
    elif args.stage in ("all", "forward"):
        forward_training, split = train_forward(
            forward, cache_dir, metadata, torch_device,
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.lr, weight_decay=args.weight_decay,
            patience=args.patience, linear_weight=args.linear_weight,
            seed=args.seed, segmented_loss=args.segmented_loss)
        forward_training.update({
            "architecture": list(hidden), "batch_size": args.batch_size,
            "learning_rate": args.lr, "weight_decay": args.weight_decay,
            "patience": args.patience, "linear_weight": args.linear_weight,
            "segmented_loss": args.segmented_loss,
            "seed": args.seed, "uniform_voltage_weighting":
            not args.emphasize_high_voltage,
        })
        save_forward_checkpoint(
            forward_path, forward, metadata, hidden, forward_training)
        forward_report_path.write_text(
            json.dumps(forward_training, indent=2) + "\n")
    else:
        if not source_forward_path.exists():
            raise RuntimeError("forward checkpoint is required")
        forward_training = load_checked_checkpoint(
            source_forward_path, forward, metadata, hidden, torch_device)

    if args.stage == "forward":
        print(json.dumps(forward_training, indent=2))
        return 0

    if (args.resume and inverse_path.exists()
            and inverse_report_path.exists()):
        inverse_training = load_checked_checkpoint(
            inverse_path, inverse, metadata, inverse_hidden, torch_device)
        logger.info("reusing %s", inverse_path)
    elif args.stage in ("all", "inverse"):
        inverse_training = train_inverse(
            inverse, forward, cache_dir, metadata, split, torch_device,
            epochs=args.inverse_epochs, batch_size=args.batch_size,
            learning_rate=args.inverse_lr, weight_decay=args.weight_decay,
            patience=args.patience,
            reconstruction_weight=args.reconstruction_weight,
            parameter_weight=args.parameter_weight,
            linear_weight=args.linear_weight, seed=args.seed)
        inverse_training.update({
            "architecture": list(inverse_hidden),
            "batch_size": args.batch_size,
            "learning_rate": args.inverse_lr,
            "weight_decay": args.weight_decay, "patience": args.patience,
            "reconstruction_weight": args.reconstruction_weight,
            "parameter_weight": args.parameter_weight,
            "inverse_input": args.inverse_input,
            "paper_cards_used_for_training": False,
            "linear_weight": args.linear_weight, "seed": args.seed,
        })
        save_inverse_checkpoint(
            inverse_path, inverse, metadata, inverse_hidden, inverse_training)
        inverse_report_path.write_text(
            json.dumps(inverse_training, indent=2) + "\n")
    else:
        if not inverse_path.exists():
            raise RuntimeError("inverse checkpoint is required")
        inverse_training = load_checked_checkpoint(
            inverse_path, inverse, metadata, inverse_hidden, torch_device)

    should_finetune = (
        args.stage == "finetune"
        or (args.measured_finetune and args.stage in ("all", "inverse")))
    if args.reset_inverse_for_finetune:
        if args.stage != "finetune":
            parser.error("--reset-inverse-for-finetune requires --stage finetune")
        if not inverse_synthetic_path.exists():
            raise RuntimeError(
                f"missing synthetic inverse checkpoint {inverse_synthetic_path}")
        inverse_training = load_checked_checkpoint(
            inverse_synthetic_path, inverse, metadata, inverse_hidden,
            torch_device)
        logger.info("reset measured fine-tune from %s", inverse_synthetic_path)
    already_finetuned = "measured_finetune" in inverse_training
    if should_finetune and not (args.resume and already_finetuned):
        if not inverse_synthetic_path.exists():
            save_inverse_checkpoint(
                inverse_synthetic_path, inverse, metadata, inverse_hidden,
                inverse_training)
        measured_report = finetune_inverse_on_measurements(
            inverse, forward, data_dir, cache_dir, metadata,
            args.parameter_set, torch_device, steps=args.measured_steps,
            learning_rate=args.measured_lr,
            prior_weight=args.measured_prior_weight,
            patience=args.measured_patience)
        inverse_training["measured_finetune"] = measured_report
        inverse_training["paper_cards_used_for_training"] = True
        save_inverse_checkpoint(
            inverse_path, inverse, metadata, inverse_hidden, inverse_training)
        inverse_report_path.write_text(
            json.dumps(inverse_training, indent=2) + "\n")

    if args.stage in ("inverse", "finetune"):
        print(json.dumps(inverse_training, indent=2))
        return 0

    records = []
    tensors = training_tensors(cache_dir, torch_device)
    for device_index, selected_device in enumerate(parse_device_list(args.devices)):
        context = build_device_context(
            selected_device, data_dir, metadata, args.parameter_set)
        record_path = out_dir / f"ml_{context.tag}.json"
        sims_path = out_dir / f"sims_{context.tag}.npz"
        if args.resume and record_path.exists() and sims_path.exists():
            records.append(json.loads(record_path.read_text()))
            logger.info("%-22s extraction already complete", context.tag)
            continue
        methods, simulation_values = {}, {}
        if args.extract_method in ("both", "fd"):
            fd, fd_sims = finite_difference_surrogate_extract(
                context, forward, tensors, torch_device,
                n_starts=args.n_starts, n_validate=args.n_validate,
                max_nfev=args.max_nfev, diff_step=args.diff_step,
                regularization=args.regularization)
            methods["surrogate_fd"] = fd
            simulation_values.update({
                f"fd_sim_{index}": np.asarray(sim)
                for index, sim in enumerate(fd_sims)})
        if args.extract_method in ("both", "inverse"):
            inv, inverse_sims = inverse_extract(
                context, inverse, forward, tensors, torch_device)
            methods["inverse_autoencoder"] = inv
            simulation_values.update({
                f"inverse_sim_{index}": np.asarray(sim)
                for index, sim in enumerate(inverse_sims)})
        record = {
            "device": context.tag, "dev_type": selected_device.dev_type,
            "L_um": selected_device.L_um, "W_um": selected_device.W_um,
            "bin_index": context.bin_index,
            "n_active_parameters": context.box.n_active,
            "methods": methods,
        }
        np.savez_compressed(sims_path, **simulation_values)
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        logger.info("%-22s %s", context.tag, "; ".join(
            f"{method} {row['rrms']:.4f} (high "
            f"{row['high_voltage_rrms']:.4f})"
            for method, row in methods.items()))
    if len(records) == len(PAPER_DEVICES):
        summary = write_summary(
            out_dir, args.parameter_set, metadata, forward_training,
            inverse_training, records, model_sources={
                "forward_checkpoint": str(source_forward_path.resolve()),
                "inverse_checkpoint": str(inverse_path.resolve()),
            })
        print(json.dumps(summary["methods"], indent=2))
    else:
        print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
