#!/usr/bin/env python3
"""Train one conditional forward emulator and reuse it for all 18 devices.

This is an exploratory final-stage experiment, separate from the primary
fixed-method comparison. One network learns

    local parameter coordinates + polarity + log(L) + log(W) -> full I-V

over all 18 confirmed-setup synthetic datasets. The frozen network is then
searched independently for each measured transistor. Raw parameter candidates
and FD-polished candidates are always validated by real NGSpice.

The script also compares measured wall times and RRMS with published-start FD
alone and the production per-device emulators. It never selects a method per
device and never replaces the canonical surrogate+FD card export.

Run only after the primary pipeline has completed:

  PYTHONPATH=src .venv/bin/python scripts/pdk_foundation_emulator.py --device mps
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

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import least_squares
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pdk_ml_extract as mlx  # noqa: E402
from cryoml.config import (FIGS_DIR, OUT_DIR, OUT_TABLES, PROCESSED_DIR,  # noqa: E402
                           ensure_dirs)
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.metrics import clean_current  # noqa: E402
from cryoml.pdk_extract import (LhcBox, PARAMS7, eval_params_new,  # noqa: E402
                                new_metric_layout, residual_fn_new)
from cryoml.spice_pdk import simulate_pdk  # noqa: E402
from cryoml.utils import device_tag, get_logger, set_seed  # noqa: E402

logger = get_logger("pdk_foundation_emulator")
SYNTH_DIR = PROCESSED_DIR / "pdk_synth"
DEFAULT_OUT = OUT_DIR / "pdk_foundation_emu"
DEFAULT_CACHE = OUT_DIR / "foundation_cache"


def geometry_ranges() -> dict[str, float]:
    log_l = np.log([d.L_um for d in PAPER_DEVICES])
    log_w = np.log([d.W_um for d in PAPER_DEVICES])
    return {
        "log_l_min": float(log_l.min()), "log_l_max": float(log_l.max()),
        "log_w_min": float(log_w.min()), "log_w_max": float(log_w.max()),
    }


def scale_between_minus_one_and_one(value: float, lo: float,
                                    hi: float) -> float:
    return float(2.0 * (value - lo) / (hi - lo) - 1.0)


def geometry_features(device, ranges: dict[str, float]) -> np.ndarray:
    return np.asarray([
        -1.0 if device.dev_type == "nmos" else 1.0,
        scale_between_minus_one_and_one(
            math.log(device.L_um), ranges["log_l_min"], ranges["log_l_max"]),
        scale_between_minus_one_and_one(
            math.log(device.W_um), ranges["log_w_min"], ranges["log_w_max"]),
    ], dtype=np.float32)


def build_cache(cache_dir: Path, *, rebuild: bool = False) -> dict:
    """Materialize float32 NPY arrays so training does not repeatedly inflate
    the 18 compressed NPZ datasets. The ready marker is written last."""
    cache_t0 = time.time()
    ready = cache_dir / "metadata.json"
    if ready.exists() and not rebuild:
        meta = json.loads(ready.read_text())
        if (meta.get("schema_version") == 1
                and meta.get("devices")
                == [device_tag(d.dev_type, d.L_um, d.W_um)
                    for d in PAPER_DEVICES]):
            return meta
        raise RuntimeError(
            f"incompatible foundation cache at {cache_dir}; use --rebuild-cache")
    if rebuild and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    descriptions = []
    n_total = 0
    n_points = None
    ranges = geometry_ranges()
    for device in PAPER_DEVICES:
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        with np.load(SYNTH_DIR / f"{tag}.npz", allow_pickle=True) as data:
            if str(data["box_mode"]) != "lhc10":
                raise RuntimeError(f"{tag}: expected lhc10 synthetic data")
            valid = int(np.sum(data["ok"]))
            points = int(data["IDS"].shape[1])
            if n_points is None:
                n_points = points
            elif points != n_points:
                raise RuntimeError(
                    f"inconsistent I-V layouts: {tag} has {points}, expected {n_points}")
            descriptions.append({"device": tag, "start": n_total,
                                 "stop": n_total + valid, "n": valid})
            n_total += valid
    if n_points is None:
        raise RuntimeError("no synthetic datasets found")

    x_path, y_path = cache_dir / "features.npy", cache_dir / "targets.npy"
    features = np.lib.format.open_memmap(
        x_path, mode="w+", dtype=np.float32, shape=(n_total, 10))
    targets = np.lib.format.open_memmap(
        y_path, mode="w+", dtype=np.float32, shape=(n_total, n_points))

    cursor = 0
    for device in PAPER_DEVICES:
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        logger.info("caching %s", tag)
        with np.load(SYNTH_DIR / f"{tag}.npz", allow_pickle=True) as data:
            valid = np.flatnonzero(data["ok"])
            theta = np.asarray(data["THETA"][valid], dtype=np.float64)
            published = {p: float(v)
                         for p, v in zip(PARAMS7, data["published"])}
            box = LhcBox(device.dev_type, int(data["bin_index"]), published)
            local = 2.0 * (theta - box.lo) / (box.hi - box.lo) - 1.0
            stop = cursor + len(valid)
            features[cursor:stop, :7] = local.astype(np.float32)
            features[cursor:stop, 7:] = geometry_features(device, ranges)
            targets[cursor:stop] = mlx.slog(
                np.asarray(data["IDS"][valid], dtype=np.float64)
            ).astype(np.float32)
            cursor = stop
    features.flush()
    targets.flush()
    del features, targets

    meta = {
        "schema_version": 1,
        "setup": "CryoPDK_Skywater130nm_ML@39b1e518",
        "box_mode": "lhc10",
        "input_definition": [*PARAMS7, "polarity", "scaled_log_L",
                             "scaled_log_W"],
        "output_definition": "signed-log current at all I-V locations",
        "n_examples": n_total, "n_points": n_points,
        "devices": [d["device"] for d in descriptions],
        "device_ranges": descriptions, "geometry_ranges": ranges,
        "cache_build_runtime_s": round(time.time() - cache_t0, 1),
    }
    ready.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def split_indices(meta: dict, seed: int,
                  val_fraction: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    train, val = [], []
    for device_index, desc in enumerate(meta["device_ranges"]):
        indices = np.arange(desc["start"], desc["stop"], dtype=np.int64)
        rng = np.random.default_rng(seed + device_index)
        rng.shuffle(indices)
        n_val = max(1, int(round(len(indices) * val_fraction)))
        val.append(indices[:n_val])
        train.append(indices[n_val:])
    return np.concatenate(train), np.concatenate(val)


class MemmapDataset(Dataset):
    def __init__(self, cache_dir: Path, indices: np.ndarray):
        self.features = np.load(cache_dir / "features.npy", mmap_mode="r")
        self.targets = np.load(cache_dir / "targets.npy", mmap_mode="r")
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]
        return (np.array(self.features[index], copy=True),
                np.array(self.targets[index], copy=True))


class FoundationEmulator(nn.Module):
    def __init__(self, n_points: int, hidden: tuple[int, ...]):
        super().__init__()
        self.network = mlx.mlp([10, *hidden, n_points])

    def forward(self, inputs):
        return self.network(inputs)


def loader_mse(model, loader, device: torch.device) -> float:
    total_sq, total_count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            total_sq += float(torch.sum((prediction - y) ** 2))
            total_count += y.numel()
    return total_sq / max(total_count, 1)


def train_foundation(model, cache_dir: Path, meta: dict, device: torch.device,
                     *, epochs: int, batch_size: int, lr: float,
                     weight_decay: float, patience: int,
                     seed: int) -> dict:
    train_idx, val_idx = split_indices(meta, seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        MemmapDataset(cache_dir, train_idx), batch_size=batch_size,
        shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(
        MemmapDataset(cache_dir, val_idx), batch_size=batch_size,
        shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1))
    best_val, best_state, bad = math.inf, None, 0
    history = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        running, count = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()
            running += float(loss) * len(x)
            count += len(x)
        scheduler.step()
        val_mse = loader_mse(model, val_loader, device)
        train_mse = running / max(count, 1)
        history.append({"epoch": epoch + 1, "train_mse": train_mse,
                        "validation_mse": val_mse,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        logger.info("foundation epoch %d train=%.6g val=%.6g",
                    epoch + 1, train_mse, val_mse)
        if val_mse < best_val - 1e-6:
            best_val, bad = val_mse, 0
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is None:
        raise RuntimeError("foundation training produced no finite checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    validation_by_device = {}
    for desc in meta["device_ranges"]:
        selected = val_idx[(val_idx >= desc["start"])
                           & (val_idx < desc["stop"])]
        device_loader = DataLoader(
            MemmapDataset(cache_dir, selected), batch_size=batch_size,
            shuffle=False, num_workers=0)
        validation_by_device[desc["device"]] = loader_mse(
            model, device_loader, device)
    return {
        "best_validation_mse": float(best_val),
        "validation_mse_by_device": validation_by_device,
        "mean_device_validation_mse": float(np.mean(
            list(validation_by_device.values()))),
        "epochs_completed": len(history),
        "runtime_s": round(time.time() - t0, 1),
        "n_train": int(len(train_idx)), "n_validation": int(len(val_idx)),
        "history": history,
    }


@dataclass
class DeviceContext:
    device: object
    tag: str
    bin_index: int
    box: LhcBox
    curves: list
    include_tags: set[str]
    metric_layout: object
    search_layout: list[tuple[int, int, float]]
    measured_clean: np.ndarray
    geometry: np.ndarray


def device_context(device, ranges: dict[str, float]) -> DeviceContext:
    tag = device_tag(device.dev_type, device.L_um, device.W_um)
    with np.load(SYNTH_DIR / f"{tag}.npz", allow_pickle=True) as data:
        published = {p: float(v)
                     for p, v in zip(PARAMS7, data["published"])}
        bin_index = int(data["bin_index"])
        slices = np.asarray(data["slices"], dtype=np.int64)
        measured = np.asarray(data["meas"], dtype=np.float64)
    box = LhcBox(device.dev_type, bin_index, published)
    curves = load_device_curves(device)
    include = mlx.baseline_include_tags(tag)
    layout = new_metric_layout(device.dev_type, device.L_um, device.W_um,
                               curves, include)
    cleaned = measured.copy()
    for start, stop in slices:
        cleaned[start:stop] = clean_current(measured[start:stop])
    search_layout = []
    for curve_index, trim_start, _cleaned, denominator, _tag in layout.entries:
        start, stop = slices[curve_index]
        search_layout.append((int(start + trim_start), int(stop), denominator))
    return DeviceContext(
        device=device, tag=tag, bin_index=bin_index, box=box, curves=curves,
        include_tags=include, metric_layout=layout,
        search_layout=search_layout, measured_clean=cleaned,
        geometry=geometry_features(device, ranges))


def search_device(context: DeviceContext, model: FoundationEmulator,
                  torch_device: torch.device, *, n_starts: int, steps: int,
                  n_validate: int, n_polish: int, max_nfev: int,
                  seed: int) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    box = context.box
    n_local = n_starts // 2
    unit = rng.uniform(1e-3, 1 - 1e-3,
                       size=(n_starts - n_local - 1, 7))
    box_starts = np.log(unit / (1.0 - unit))
    local = box.z_published + rng.normal(0, 2.0, size=(n_local, 7))
    starts = np.concatenate([box.z_published[None, :], local, box_starts])
    z = torch.tensor(starts, dtype=torch.float32, device=torch_device,
                     requires_grad=True)
    geometry = torch.tensor(context.geometry, device=torch_device)
    measured = torch.tensor(context.measured_clean, dtype=torch.float32,
                            device=torch_device)
    optimizer = torch.optim.Adam([z], lr=0.05)
    best_loss = np.full(len(starts), np.inf)
    best_z = starts.copy()
    search_t0 = time.time()
    for _ in range(steps):
        optimizer.zero_grad()
        local_features = 2.0 * torch.sigmoid(z) - 1.0
        metadata = geometry.unsqueeze(0).expand(len(z), -1)
        current = mlx.current_from_slog(
            model(torch.cat([local_features, metadata], dim=1)))
        rrms = []
        for start, stop, denominator in context.search_layout:
            rmse = torch.sqrt(torch.mean(
                (current[:, start:stop]
                 - measured[start:stop].unsqueeze(0)) ** 2, dim=1))
            rrms.append(rmse / denominator)
        losses = torch.stack(rrms, dim=1).mean(dim=1)
        losses.sum().backward()
        optimizer.step()
        with torch.no_grad():
            z.clamp_(-8.0, 8.0)
        values = losses.detach().cpu().numpy()
        improved = values < best_loss
        if improved.any():
            current_z = z.detach().cpu().numpy().astype(np.float64)
            best_loss[improved] = values[improved]
            best_z[improved] = current_z[improved]
    search_runtime = time.time() - search_t0

    candidates, seen = [], []
    for index in np.argsort(best_loss):
        candidate = best_z[index]
        if any(np.linalg.norm(candidate - prior) < 0.5 for prior in seen):
            continue
        seen.append(candidate)
        candidates.append(candidate)
        if len(candidates) == n_validate:
            break

    validation_t0 = time.time()
    validated = []
    for candidate in candidates:
        params = box.z_to_params(candidate)
        fixed, official, _ = eval_params_new(
            context.device.dev_type, context.device.L_um, context.device.W_um,
            context.bin_index, context.curves, params, context.include_tags)
        validated.append({
            "z": candidate, "params": params,
            "rrms": float(fixed["rrms"]),
            "rrms_official": float(official["rrms"]),
        })
    validated = [row for row in validated if np.isfinite(row["rrms"])]
    if not validated:
        published, official, _ = eval_params_new(
            context.device.dev_type, context.device.L_um, context.device.W_um,
            context.bin_index, context.curves, box.published,
            context.include_tags)
        validated = [{"z": box.z_published, "params": box.published,
                      "rrms": float(published["rrms"]),
                      "rrms_official": float(official["rrms"])}]
    validated.sort(key=lambda row: row["rrms"])
    validation_runtime = time.time() - validation_t0
    raw = validated[0]

    fd_t0 = time.time()
    attempts = []
    for rank, candidate in enumerate(validated[:n_polish]):
        attempt_t0 = time.time()
        try:
            solution = least_squares(
                lambda vector: residual_fn_new(
                    vector, box, context.device.dev_type, context.device.L_um,
                    context.device.W_um, context.bin_index, context.curves,
                    context.metric_layout),
                candidate["z"], method="trf", jac="2-point",
                diff_step=2e-2, max_nfev=max_nfev)
            params = box.z_to_params(solution.x)
            fixed, official, _ = eval_params_new(
                context.device.dev_type, context.device.L_um,
                context.device.W_um, context.bin_index, context.curves,
                params, context.include_tags)
            attempts.append({
                "candidate_rank": rank, "start_rrms": candidate["rrms"],
                "endpoint_rrms": float(fixed["rrms"]),
                "rrms_official": float(official["rrms"]), "params": params,
                "nfev": int(solution.nfev),
                "success": bool(solution.success),
                "runtime_s": round(time.time() - attempt_t0, 1),
            })
        except Exception as exc:  # noqa: BLE001
            attempts.append({
                "candidate_rank": rank, "start_rrms": candidate["rrms"],
                "endpoint_rrms": None, "params": candidate["params"],
                "nfev": 0, "success": False,
                "runtime_s": round(time.time() - attempt_t0, 1),
                "error": f"{type(exc).__name__}: {exc}",
            })
    accepted = [attempt for attempt in attempts
                if attempt["endpoint_rrms"] is not None
                and np.isfinite(attempt["endpoint_rrms"])
                and attempt["endpoint_rrms"] <= raw["rrms"]]
    if accepted:
        best_fd = min(accepted, key=lambda row: row["endpoint_rrms"])
        polished = {"params": best_fd["params"],
                    "rrms": best_fd["endpoint_rrms"],
                    "rrms_official": best_fd["rrms_official"]}
    else:
        polished = {key: raw[key]
                    for key in ("params", "rrms", "rrms_official")}
    fd_runtime = time.time() - fd_t0

    raw_sims = simulate_pdk(
        context.device.dev_type, context.device.L_um, context.device.W_um,
        context.curves, params=raw["params"], bin_index=context.bin_index)
    polished_sims = simulate_pdk(
        context.device.dev_type, context.device.L_um, context.device.W_um,
        context.curves, params=polished["params"], bin_index=context.bin_index)
    return {
        "device": context.tag, "dev_type": context.device.dev_type,
        "L_um": context.device.L_um, "W_um": context.device.W_um,
        "bin_index": context.bin_index, "box_mode": "lhc10",
        "method": "foundation_emu_search_fixed",
        "selection_policy": "one shared emulator and one fixed recipe across all 18",
        "include_tags": sorted(context.include_tags),
        "methods": {
            "foundation_emu_search": {
                "params": raw["params"], "rrms": raw["rrms"],
                "rrms_official": raw["rrms_official"]},
            "foundation_emu_search+fd": polished,
        },
        "fd_attempts": attempts,
        "timing": {
            "search_s": round(search_runtime, 1),
            "ngspice_validation_s": round(validation_runtime, 1),
            "fd_s": round(fd_runtime, 1),
            "total_device_s": round(time.time() - t0, 1),
        },
        "config": {"n_starts": n_starts, "steps": steps,
                   "n_validate": n_validate, "n_polish": n_polish,
                   "max_nfev": max_nfev, "seed": seed},
        "raw_sims": raw_sims, "polished_sims": polished_sims,
    }


def aggregate(records: list[dict], method: str) -> dict[str, float]:
    nmos = [row["methods"][method]["rrms"] for row in records
            if row["dev_type"] == "nmos"]
    pmos = [row["methods"][method]["rrms"] for row in records
            if row["dev_type"] == "pmos"]
    return {
        "nmos_mean": float(np.mean(nmos)), "pmos_mean": float(np.mean(pmos)),
        "combined_rrms": float((np.mean(nmos) + np.mean(pmos)) / 2.0),
        "all_device_mean_rrms": float(np.mean(nmos + pmos)),
    }


def load_json_records(directory: Path, prefix: str = "ml_") -> list[dict]:
    records = []
    for device in PAPER_DEVICES:
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        path = directory / f"{prefix}{tag}.json"
        if not path.exists():
            raise RuntimeError(f"comparison input missing: {path}")
        records.append(json.loads(path.read_text()))
    return records


def comparison_summary(records: list[dict], training: dict,
                       out_dir: Path) -> dict:
    foundation_raw = aggregate(records, "foundation_emu_search")
    foundation_fd = aggregate(records, "foundation_emu_search+fd")
    fd_records = load_json_records(OUT_DIR / "pdk_fd_alone")
    per_device_records = load_json_records(OUT_DIR / "pdk_surrogate_final")
    baseline = json.loads((OUT_DIR / "pdk_baseline" / "pdk_baseline.json").read_text())

    fd_total = float(sum(row["runtime_s"] for row in fd_records))
    per_device_total = float(sum(row["runtime_s"] for row in per_device_records))
    per_device_val_mse = float(np.mean(
        [row["emulator_val_mse"] for row in per_device_records]))
    foundation_device_total = float(sum(
        row["timing"]["total_device_s"] for row in records))
    cache_runtime = float(training.get("cache_build_runtime_s", 0.0))
    foundation_total = float(
        cache_runtime + training["runtime_s"] + foundation_device_total)
    foundation_upfront = cache_runtime + float(training["runtime_s"])
    if fd_total > foundation_device_total:
        fd_break_even = foundation_upfront / (fd_total - foundation_device_total)
    else:
        fd_break_even = None
    if per_device_total > foundation_device_total:
        emulator_break_even = (
            foundation_upfront / (per_device_total - foundation_device_total))
    else:
        emulator_break_even = None

    published_mean = float(baseline["summary"]["all_device_mean_rrms"])
    summary = {
        "status": "exploratory; excluded from canonical method selection/export",
        "n_devices": 18,
        "foundation_raw": foundation_raw,
        "foundation_plus_fd": foundation_fd,
        "published_all_device_mean_rrms": published_mean,
        "emulator_validation_mse": {
            "foundation_global": float(training["best_validation_mse"]),
            "foundation_mean_across_devices": float(
                training["mean_device_validation_mse"]),
            "mean_of_18_per_device_emulators": per_device_val_mse,
        },
        "timing_s": {
            "fd_only_18_total": fd_total,
            "per_device_emulator_18_total": per_device_total,
            "foundation_training_once": float(training["runtime_s"]),
            "foundation_cache_build_once": cache_runtime,
            "foundation_18_search_validate_fd": foundation_device_total,
            "foundation_first_campaign_total": foundation_total,
        },
        "break_even_campaigns": {
            "versus_fd_only": fd_break_even,
            "versus_retraining_18_per_device_emulators": emulator_break_even,
        },
        "interpretation": {
            "fd_only": "No training and direct local NGSpice accuracy; best for a one-time small device set, but each objective evaluation invokes NGSpice and each new device needs another solve.",
            "foundation": "One differentiable model is reusable across the 18 conditioned training geometries and supports large parallel inverse searches; it has substantial upfront training/cache cost, can blur geometry-specific behavior, still requires NGSpice validation/FD, and this split does not establish generalization to an unseen geometry.",
            "per_device_emulators": "Usually easier to fit accurately for one geometry but require one training run and one model artifact per transistor.",
        },
    }
    (OUT_TABLES / "foundation_emulator_study.json").write_text(
        json.dumps({"summary": summary, "devices": records,
                    "training": training}, indent=2, default=float) + "\n")

    lines = [
        "# Foundation emulator study", "",
        "This exploratory result uses one fixed conditional forward emulator "
        "for every device. All reported parameter vectors are re-simulated in "
        "real NGSpice. It is not used for per-device selection or card export.", "",
        "| device | foundation raw | foundation + FD | device runtime (s) |",
        "|---|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['device']} | "
            f"{row['methods']['foundation_emu_search']['rrms']:.4f} | "
            f"{row['methods']['foundation_emu_search+fd']['rrms']:.4f} | "
            f"{row['timing']['total_device_s']:.1f} |")
    lines += [
        "", "## Aggregate", "",
        f"- Foundation raw all-device RRMS: "
        f"{foundation_raw['all_device_mean_rrms']:.4f}.",
        f"- Foundation + FD all-device RRMS: "
        f"{foundation_fd['all_device_mean_rrms']:.4f}.",
        f"- Held-out signed-log validation MSE: foundation global "
        f"{training['best_validation_mse']:.6g}; arithmetic mean across its "
        f"18 device slices {training['mean_device_validation_mse']:.6g}; "
        f"mean of the 18 separate emulators {per_device_val_mse:.6g}.",
        f"- Foundation cache build once: {cache_runtime:.1f} s; training once: "
        f"{training['runtime_s']:.1f} s; "
        f"18-device search/validation/FD: {foundation_device_total:.1f} s; "
        f"first campaign total: {foundation_total:.1f} s.",
        f"- Published-start FD-only total: {fd_total:.1f} s.",
        f"- Eighteen per-device emulator extractions: {per_device_total:.1f} s.",
        "", "## Tradeoffs", "",
        "- FD-only has no training cost and optimizes the real simulator "
        "directly. It is the natural one-time baseline, but it cannot evaluate "
        "thousands of candidates in parallel without thousands of NGSpice runs.",
        "- The foundation emulator amortizes one training run across devices "
        "and makes inverse search differentiable and parallel. Its risks are "
        "shared-model bias across geometries, large training/cache cost, and "
        "the continuing need for NGSpice validation and usually FD polish. "
        "The current random within-geometry validation split does not prove "
        "accuracy on an entirely unseen L/W geometry.",
        "- Per-device emulators isolate geometry-specific behavior and are "
        "simpler fits, at the cost of 18 training runs and 18 checkpoints.",
    ]
    if fd_break_even is None:
        lines.append(
            "- Runtime does not break even against FD-only in the measured "
            "workflow because foundation search/validation itself is no faster "
            "than the complete FD-only campaign, before training cost.")
    else:
        lines.append(
            f"- Measured runtime break-even against repeated FD-only campaigns "
            f"is approximately {fd_break_even:.2f} campaigns.")
    (OUT_TABLES / "foundation_emulator_study.md").write_text(
        "\n".join(lines) + "\n")

    x = np.arange(len(records))
    raw_values = [row["methods"]["foundation_emu_search"]["rrms"]
                  for row in records]
    fd_values = [row["methods"]["foundation_emu_search+fd"]["rrms"]
                 for row in records]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    axes[0].plot(x, raw_values, "o-", label="foundation raw (NGSpice)")
    axes[0].plot(x, fd_values, "^-", label="foundation + FD (NGSpice)")
    axes[0].axhline(published_mean, color="0.35", ls=":",
                    label=f"published mean {published_mean:.3f}")
    axes[0].set_xlabel("device index")
    axes[0].set_ylabel("RRMS")
    axes[0].set_title("One conditional emulator across all 18 devices")
    axes[0].legend()
    timing_names = ["FD only", "18 per-device\nemulators",
                    "foundation\nfirst campaign"]
    timing_values = [fd_total, per_device_total, foundation_total]
    axes[1].bar(timing_names, timing_values,
                color=["#4f7f6b", "#4878a8", "#d27a42"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("measured wall time (s, log scale)")
    axes[1].set_title("End-to-end compute cost")
    fig.savefig(FIGS_DIR / "foundation_emulator_study.png", dpi=180)
    plt.close(fig)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--arch", default="1024,1024,1024,1024")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--n-starts", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--n-validate", type=int, default=14)
    parser.add_argument("--n-polish", type=int, default=5)
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--remove-cache", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    set_seed(args.seed)
    out_dir, cache_dir = Path(args.out_dir), Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = build_cache(cache_dir, rebuild=args.rebuild_cache)
    hidden = tuple(int(value) for value in args.arch.split(",") if value)
    torch_device = torch.device(args.device)
    checkpoint_path = out_dir / "foundation_model.pt"
    training_path = out_dir / "training.json"
    model = FoundationEmulator(meta["n_points"], hidden).to(torch_device)
    if args.resume and checkpoint_path.exists() and training_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=torch_device,
                                weights_only=False)
        if tuple(checkpoint["hidden"]) != hidden:
            raise RuntimeError("checkpoint architecture does not match --arch")
        model.load_state_dict(checkpoint["state"])
        training = json.loads(training_path.read_text())
        logger.info("reusing completed foundation checkpoint")
    else:
        training = train_foundation(
            model, cache_dir, meta, torch_device, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr,
            weight_decay=args.weight_decay, patience=args.patience,
            seed=args.seed)
        torch.save({
            "state": {key: value.detach().cpu()
                      for key, value in model.state_dict().items()},
            "hidden": list(hidden), "n_points": meta["n_points"],
            "input_definition": meta["input_definition"],
            "geometry_ranges": meta["geometry_ranges"],
        }, checkpoint_path)
        training.update({
            "arch": list(hidden), "n_parameters": sum(
                parameter.numel() for parameter in model.parameters()),
            "cache_build_runtime_s": float(
                meta.get("cache_build_runtime_s", 0.0)),
            "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "patience": args.patience,
            "seed": args.seed,
        })
        training_path.write_text(json.dumps(training, indent=2) + "\n")
    model.eval()

    serializable_records = []
    for device_index, device in enumerate(PAPER_DEVICES):
        context = device_context(device, meta["geometry_ranges"])
        record_path = out_dir / f"ml_{context.tag}.json"
        sims_path = out_dir / f"sims_{context.tag}.npz"
        if args.resume and record_path.exists() and sims_path.exists():
            logger.info("%-22s already complete; skipping", context.tag)
            serializable_records.append(json.loads(record_path.read_text()))
            continue
        record = search_device(
            context, model, torch_device, n_starts=args.n_starts,
            steps=args.steps, n_validate=args.n_validate,
            n_polish=args.n_polish, max_nfev=args.max_nfev,
            seed=args.seed + device_index)
        raw_sims = record.pop("raw_sims")
        polished_sims = record.pop("polished_sims")
        np.savez(sims_path,
                 **{f"raw_sim_{index}": np.asarray(sim)
                    for index, sim in enumerate(raw_sims)},
                 **{f"fd_sim_{index}": np.asarray(sim)
                    for index, sim in enumerate(polished_sims)})
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        serializable_records.append(record)
        logger.info("%-22s foundation %.4f -> %.4f (%.1fs)",
                    context.tag,
                    record["methods"]["foundation_emu_search"]["rrms"],
                    record["methods"]["foundation_emu_search+fd"]["rrms"],
                    record["timing"]["total_device_s"])

    summary = comparison_summary(serializable_records, training, out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.remove_cache:
        shutil.rmtree(cache_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
