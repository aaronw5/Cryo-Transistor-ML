#!/usr/bin/env python3
"""Evaluate and optionally train polarity-specialist foundation surrogates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import foundation_parameter_study as fps  # noqa: E402
from cryoml.foundation import FoundationSurrogate  # noqa: E402
from cryoml.utils import set_seed  # noqa: E402


def filtered_split(cache_dir: Path, metadata: dict, seed: int,
                   polarity: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = fps.split_indices(metadata, seed)
    device_ids = np.load(cache_dir / "device_ids.npy", mmap_mode="r")
    allowed = np.asarray([
        tag.startswith(polarity + "_") for tag in metadata["devices"]
    ], dtype=bool)
    return tuple(np.asarray(indices)[allowed[device_ids[indices]]]
                 for indices in split)


def loaders(cache_dir: Path, split: tuple, batch_size: int):
    return tuple(DataLoader(
        fps.FoundationDataset(cache_dir, indices), batch_size=batch_size,
        shuffle=False, num_workers=0) for indices in split)


def evaluate(model, cache_dir: Path, metadata: dict, split: tuple,
             device: torch.device, batch_size: int,
             linear_weight: float) -> dict:
    _train, _validation, test = loaders(cache_dir, split, batch_size)
    tensors = fps.training_tensors(cache_dir, device)
    return {
        "test": fps.evaluate_forward(
            model, test, tensors, device, linear_weight),
        "test_by_curve": fps.evaluate_forward_by_curve(
            model, test, tensors, metadata, device),
        "n_train": int(len(split[0])),
        "n_validation": int(len(split[1])),
        "n_test": int(len(split[2])),
    }


def load_shared(baseline_out: Path, metadata: dict,
                device: torch.device):
    checkpoint = torch.load(baseline_out / "forward_model.pt",
                            map_location=device, weights_only=False)
    model = FoundationSurrogate(
        metadata["n_parameters"], metadata["n_points"],
        tuple(checkpoint["hidden"])).to(device)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return model, tuple(checkpoint["hidden"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter-set", choices=("seven", "full"),
                        default="full")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--train-separate", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--linear-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("out/pdk_foundation_polarity_full"))
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    _data_dir, cache_dir, baseline_out = fps.default_paths(args.parameter_set)
    metadata = json.loads((cache_dir / "metadata.json").read_text())
    shared, hidden = load_shared(baseline_out, metadata, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "parameter_set": args.parameter_set,
        "split": "same random-within-geometry split as controlled foundation",
        "normalization": "same global current normalization as shared model",
        "architecture_per_model": list(hidden),
        "shared": {}, "separate": {},
    }
    splits = {}
    for polarity in ("nmos", "pmos"):
        split = filtered_split(cache_dir, metadata, args.seed, polarity)
        splits[polarity] = split
        report["shared"][polarity] = evaluate(
            shared, cache_dir, metadata, split, device,
            args.batch_size, args.linear_weight)

    if args.train_separate:
        for index, polarity in enumerate(("nmos", "pmos")):
            set_seed(args.seed + index + 1)
            model = FoundationSurrogate(
                metadata["n_parameters"], metadata["n_points"], hidden).to(device)
            training, _ = fps.train_forward(
                model, cache_dir, metadata, device,
                epochs=args.epochs, batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay, patience=args.patience,
                linear_weight=args.linear_weight, seed=args.seed + index + 1,
                split_override=splits[polarity])
            specialist = evaluate(
                model, cache_dir, metadata, splits[polarity], device,
                args.batch_size, args.linear_weight)
            specialist["training"] = training
            report["separate"][polarity] = specialist
            torch.save({"state": model.state_dict(), "hidden": hidden,
                        "parameter_set": args.parameter_set,
                        "polarity": polarity,
                        "metadata_schema": metadata["schema_version"]},
                       args.out_dir / f"{polarity}_forward_model.pt")
            (args.out_dir / "summary.json").write_text(
                json.dumps(report, indent=2) + "\n")
    (args.out_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n")
    compact = {
        family: {polarity: values[polarity]["test"]
                 for polarity in values}
        for family, values in (("shared", report["shared"]),
                               ("separate", report["separate"]))
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
