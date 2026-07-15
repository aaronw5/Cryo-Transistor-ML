#!/usr/bin/env python3
"""Direct I-V-to-parameter MLP baseline.

This is the ordinary supervised forward-pass method from the reference Keras
notebook, adapted to the seven parameters and PyTorch/MPS environment already
used by this repository:

* 301 fixed I-V sample locations, concatenated as linear and signed-log
  current -> 602 input features;
* per-feature min-max scaling from the synthetic training split;
* seven unit-box parameter targets;
* Dense -> LeakyReLU -> Dropout hidden blocks with LeCun-uniform
  initialization and L2 regularization;
* Adam, exponential learning-rate decay, validation early stopping;
* one forward pass on the measured I-V curves, followed only by real-NGSpice
  validation. No parameter search and no finite-difference polish.

The hidden widths default to a practical 512x3 for 18 independent 10k-sample
device models. The notebook-scale architecture remains available with
``--arch 1700,3900,4500``.

Output: out/pdk_direct_mlp/{ml,sims,model}_<device>.* plus summary.json.

  python scripts/pdk_direct_mlp.py --device mps
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cryoml.config import OUT_DIR, PROCESSED_DIR, ensure_dirs  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES, parse_device_list  # noqa: E402
from cryoml.metrics import clean_current, score_device_new  # noqa: E402
from cryoml.pdk_extract import LhcBox, PARAMS7  # noqa: E402
from cryoml.spice_pdk import simulate_pdk  # noqa: E402
from cryoml.utils import device_tag, get_logger, set_seed  # noqa: E402
from pdk_ml_extract import baseline_include_tags  # noqa: E402

logger = get_logger("pdk_direct_mlp")

SYNTH_DIR = PROCESSED_DIR / "pdk_synth"
DEFAULT_OUT = OUT_DIR / "pdk_direct_mlp"
I_REF = 1e-9
N_CURRENT_FEATURES = 301


def signed_log(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.sign(values) * np.log1p(np.abs(values) / I_REF)


def sampled_curve_indices(slices: np.ndarray,
                          total: int = N_CURRENT_FEATURES) -> np.ndarray:
    """Select a deterministic, nearly equal point count from every curve."""
    slices = np.asarray(slices, dtype=np.int64)
    if total < len(slices):
        raise ValueError("feature budget must include every curve")
    counts = np.full(len(slices), total // len(slices), dtype=int)
    counts[: total % len(slices)] += 1
    selected = []
    for (start, stop), count in zip(slices, counts):
        if stop - start < count:
            raise ValueError("curve is shorter than its feature allocation")
        local = np.linspace(start, stop - 1, count).round().astype(int)
        selected.extend(local.tolist())
    out = np.asarray(selected, dtype=np.int64)
    if len(out) != total or len(np.unique(out)) != total:
        raise RuntimeError("failed to construct 301 unique I-V locations")
    return out


def fit_minmax(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = np.min(values, axis=0)
    span = np.max(values, axis=0) - lo
    span[span < 1e-20] = 1.0
    return lo, span


def make_features(currents: np.ndarray, indices: np.ndarray,
                  linear_lo: np.ndarray, linear_span: np.ndarray,
                  log_lo: np.ndarray, log_span: np.ndarray) -> np.ndarray:
    linear = np.asarray(currents, dtype=np.float64)[..., indices]
    logged = signed_log(linear)
    linear = np.clip((linear - linear_lo) / linear_span, 0.0, 1.0)
    logged = np.clip((logged - log_lo) / log_span, 0.0, 1.0)
    return np.concatenate([linear, logged], axis=-1).astype(np.float32)


class DirectParameterMLP(nn.Module):
    def __init__(self, input_size: int, hidden: tuple[int, ...],
                 dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        sizes = (input_size, *hidden, len(PARAMS7))
        for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            layer = nn.Linear(fan_in, fan_out)
            # Keras lecun_uniform: variance 1/fan_in.
            bound = math.sqrt(3.0 / fan_in)
            nn.init.uniform_(layer.weight, -bound, bound)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            if i < len(sizes) - 2:
                layers.append(nn.LeakyReLU(negative_slope=0.01))
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_model(model: nn.Module, x_train: torch.Tensor,
                y_train: torch.Tensor, x_val: torch.Tensor,
                y_val: torch.Tensor, *, epochs: int, batch_size: int,
                lr: float, decay_steps: int, decay_rate: float,
                l2: float, patience: int) -> tuple[float, int]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    global_step = 0
    completed = 0
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x_train), device=x_train.device)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            current_lr = lr * decay_rate ** (global_step / decay_steps)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            optimizer.zero_grad()
            parameter_mse = nn.functional.mse_loss(model(x_train[batch]),
                                                    y_train[batch])
            # Match Keras Dense(kernel_regularizer=l2(...)): regularize
            # weight matrices, not biases.
            regularization = l2 * sum(
                torch.sum(parameter ** 2)
                for parameter in model.parameters() if parameter.ndim > 1)
            loss = parameter_mse + regularization
            loss.backward()
            optimizer.step()
            global_step += 1
        model.eval()
        with torch.no_grad():
            parameter_mse = nn.functional.mse_loss(model(x_val), y_val)
            regularization = l2 * sum(
                torch.sum(parameter ** 2)
                for parameter in model.parameters() if parameter.ndim > 1)
            val = float(parameter_mse + regularization)
        completed = epoch + 1
        if val < best_val - 1e-7:
            best_val = val
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("direct MLP training produced no finite checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return best_val, completed


def run_device(device, torch_device: str, out_dir: Path, *,
               arch: tuple[int, ...], dropout: float, epochs: int,
               batch_size: int, lr: float, decay_steps: int,
               decay_rate: float, l2: float, patience: int,
               seed: int) -> dict:
    tag = device_tag(device.dev_type, device.L_um, device.W_um)
    t0 = time.time()
    set_seed(seed)
    rng = np.random.default_rng(seed)
    data = np.load(SYNTH_DIR / f"{tag}.npz", allow_pickle=True)
    if str(data["box_mode"]) != "lhc10":
        raise RuntimeError(f"{tag}: direct MLP requires the current lhc10 data")
    valid = np.flatnonzero(data["ok"])
    rng.shuffle(valid)
    n_test = max(1, int(0.20 * len(valid)))
    train_val, test_idx = valid[n_test:], valid[:n_test]
    n_val = max(1, int(0.20 * len(train_val)))
    train_idx, val_idx = train_val[n_val:], train_val[:n_val]

    feature_idx = sampled_curve_indices(data["slices"])
    train_linear = np.asarray(data["IDS"][train_idx][:, feature_idx],
                              dtype=np.float64)
    linear_lo, linear_span = fit_minmax(train_linear)
    log_lo, log_span = fit_minmax(signed_log(train_linear))

    def x_for(indices):
        return make_features(data["IDS"][indices], feature_idx,
                             linear_lo, linear_span, log_lo, log_span)

    published = {p: float(v) for p, v in zip(PARAMS7, data["published"])}
    box = LhcBox(device.dev_type, int(data["bin_index"]), published)
    theta = np.asarray(data["THETA"], dtype=np.float64)
    unit_targets = np.clip((theta - box.lo) / (box.hi - box.lo), 0.0, 1.0)

    tdev = torch.device(torch_device)
    x_train = torch.tensor(x_for(train_idx), device=tdev)
    x_val = torch.tensor(x_for(val_idx), device=tdev)
    x_test = torch.tensor(x_for(test_idx), device=tdev)
    y_train = torch.tensor(unit_targets[train_idx].astype(np.float32),
                           device=tdev)
    y_val = torch.tensor(unit_targets[val_idx].astype(np.float32), device=tdev)
    y_test = torch.tensor(unit_targets[test_idx].astype(np.float32),
                          device=tdev)

    model = DirectParameterMLP(2 * len(feature_idx), arch, dropout).to(tdev)
    n_params = sum(p.numel() for p in model.parameters())
    val_loss, epochs_completed = train_model(
        model, x_train, y_train, x_val, y_val, epochs=epochs,
        batch_size=batch_size, lr=lr, decay_steps=decay_steps,
        decay_rate=decay_rate, l2=l2, patience=patience)
    with torch.no_grad():
        val_mse = float(nn.functional.mse_loss(model(x_val), y_val))
        test_mse = float(nn.functional.mse_loss(model(x_test), y_test))

    measured_parts = []
    for start, stop in np.asarray(data["slices"], dtype=np.int64):
        measured_parts.append(clean_current(data["meas"][start:stop]))
    measured = np.concatenate(measured_parts)
    measured_x = make_features(measured[None, :], feature_idx,
                               linear_lo, linear_span, log_lo, log_span)
    with torch.no_grad():
        unit_prediction = model(torch.tensor(measured_x, device=tdev))[0]
    unit_prediction = np.clip(unit_prediction.cpu().numpy(), 0.0, 1.0)
    physical = box.lo + unit_prediction * (box.hi - box.lo)
    params = {p: float(physical[i]) for i, p in enumerate(PARAMS7)}

    curves = load_device_curves(device)
    sims = simulate_pdk(device.dev_type, device.L_um, device.W_um, curves,
                        params=params, bin_index=int(data["bin_index"]))
    include = baseline_include_tags(tag)
    fixed = score_device_new(device.dev_type, device.L_um, device.W_um,
                             curves, sims, include_tags=include)
    official = score_device_new(device.dev_type, device.L_um, device.W_um,
                                curves, sims)

    checkpoint = {
        "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "arch": list(arch), "dropout": dropout,
        "feature_indices": feature_idx,
        "linear_lo": linear_lo, "linear_span": linear_span,
        "log_lo": log_lo, "log_span": log_span,
        "params": list(PARAMS7), "box_lo": box.lo, "box_hi": box.hi,
    }
    torch.save(checkpoint, out_dir / f"model_{tag}.pt")
    np.savez(out_dir / f"sims_{tag}.npz",
             **{f"sim_{i}": np.asarray(sim) for i, sim in enumerate(sims)})
    rec = {
        "device": tag, "dev_type": device.dev_type,
        "L_um": device.L_um, "W_um": device.W_um,
        "bin_index": int(data["bin_index"]), "box_mode": "lhc10",
        "method": "direct_mlp_forward_pass",
        "selection_policy": "one fixed direct MLP recipe across all devices",
        "include_tags": sorted(include), "params": params,
        "params_by_method": {"direct_mlp_forward_pass": params},
        "best_method": "direct_mlp_forward_pass",
        "rrms": float(fixed["rrms"]),
        "rrms_official": float(official["rrms"]),
        "sigma_official": float(official["sigma"]),
        "n_curves_official": int(official["n_curves"]),
        "validation_parameter_mse": float(val_mse),
        "test_parameter_mse": float(test_mse),
        "best_validation_loss_with_l2": float(val_loss),
        "n_train": int(len(train_idx)), "n_validation": int(len(val_idx)),
        "n_test": int(len(test_idx)), "n_features": int(2 * len(feature_idx)),
        "n_parameters": int(n_params), "arch": list(arch),
        "epochs_completed": int(epochs_completed),
        "training_config": {
            "seed": int(seed), "dropout": float(dropout),
            "max_epochs": int(epochs), "batch_size": int(batch_size),
            "initial_learning_rate": float(lr),
            "decay_steps": int(decay_steps),
            "decay_rate": float(decay_rate), "l2_kernel": float(l2),
            "early_stopping_patience": int(patience),
            "split": {"train": 0.64, "validation": 0.16, "test": 0.20},
            "feature_recipe": "301 linear currents + 301 signed-log currents",
        },
        "runtime_s": round(time.time() - t0, 1),
    }
    json.dump(rec, open(out_dir / f"ml_{tag}.json", "w"), indent=2)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default=None)
    ap.add_argument("--device", default="mps", help="PyTorch device")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--arch", default="512,512,512")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--decay-steps", type=int, default=100)
    ap.add_argument("--decay-rate", type=float, default=0.99)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="Permit a subset run for smoke testing; final "
                         "reported runs must omit this flag")
    args = ap.parse_args()

    ensure_dirs()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arch = tuple(int(v) for v in args.arch.split(",") if v)
    if not arch:
        ap.error("--arch must contain at least one hidden width")
    devices = parse_device_list(args.devices)
    for index, device in enumerate(devices):
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        path = out_dir / f"ml_{tag}.json"
        if args.resume and path.exists():
            rec = json.load(open(path))
            if (rec.get("method") == "direct_mlp_forward_pass"
                    and np.isfinite(rec.get("rrms", np.nan))):
                logger.info("%-22s already complete; skipping", tag)
                continue
        rec = run_device(
            device, args.device, out_dir, arch=arch, dropout=args.dropout,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.learning_rate, decay_steps=args.decay_steps,
            decay_rate=args.decay_rate, l2=args.l2,
            patience=args.patience, seed=args.seed + index)
        logger.info("%-22s direct RRMS %.4f val/test MSE %.5f/%.5f (%.0fs)",
                    tag, rec["rrms"], rec["validation_parameter_mse"],
                    rec["test_parameter_mse"], rec["runtime_s"])

    all_expected = {device_tag(d.dev_type, d.L_um, d.W_um)
                    for d in PAPER_DEVICES}
    requested = {device_tag(d.dev_type, d.L_um, d.W_um) for d in devices}
    expected = requested if args.allow_partial else all_expected
    records = []
    for path in sorted(out_dir.glob("ml_*.json")):
        rec = json.load(open(path))
        if rec.get("device") in expected:
            records.append(rec)
    seen = {r["device"] for r in records}
    if seen != expected:
        raise RuntimeError(f"direct MLP run incomplete; missing {expected - seen}")
    nmos = [r["rrms"] for r in records if r["dev_type"] == "nmos"]
    pmos = [r["rrms"] for r in records if r["dev_type"] == "pmos"]
    nmos_mean = float(np.mean(nmos)) if nmos else None
    pmos_mean = float(np.mean(pmos)) if pmos else None
    baseline = {r["device"]: r for r in json.load(
        open(OUT_DIR / "pdk_baseline" / "pdk_baseline.json"))["devices"]}
    summary = {
        "method": "direct_mlp_forward_pass",
        "selection_policy": "one fixed direct MLP recipe across all 18 devices",
        "complete_18_device_run": seen == all_expected,
        "n_devices": len(records),
        "mean_rrms": float(np.mean(nmos + pmos)),
        "nmos_mean": nmos_mean, "pmos_mean": pmos_mean,
        "combined_rrms": ((nmos_mean + pmos_mean) / 2
                           if nmos_mean is not None and pmos_mean is not None
                           else None),
        "wins_vs_paper_cards": int(sum(
            r["rrms"] < baseline[r["device"]]["rrms"] for r in records)),
        "mean_test_parameter_mse": float(np.mean(
            [r["test_parameter_mse"] for r in records])),
        "fixed_recipe": {
            "arch": list(arch), "dropout": args.dropout,
            "max_epochs": args.epochs, "batch_size": args.batch_size,
            "initial_learning_rate": args.learning_rate,
            "decay_steps": args.decay_steps, "decay_rate": args.decay_rate,
            "l2_kernel": args.l2, "early_stopping_patience": args.patience,
            "base_seed": args.seed,
        },
    }
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
