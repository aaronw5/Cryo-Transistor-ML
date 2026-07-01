#!/usr/bin/env python3
"""Tandem multi-head MLP for direct parameter prediction.

The supervised inverse MLP in ``pdk_ml_direct.py`` regresses toward an
average parameter vector because several parameter vectors produce similar
curves. This experiment gives one inverse MLP many region-specialized output
heads. Optional supervised tandem pretraining uses both parameter and frozen
forward-emulator reconstruction losses.

At extraction time the inverse MLP predicts several parameter candidates.
Its final layer is then calibrated, without parameter labels, on the measured
curve through the frozen emulator. Every reported candidate is validated in
real NGSpice, and the strongest candidates receive the same finite-difference
polish used elsewhere in the repository.

The best measured-domain configuration uses minimal supervised pretraining
and 2,048 heads. This avoids inverse-regression averaging while retaining an
MLP that directly emits all candidate parameter vectors.

  python scripts/pdk_mlp_tandem.py --device mps \
      --out-dir out/pdk_mlp_tandem
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pdk_ml_extract as mlx  # noqa: E402
from cryoml.config import OUT_DIR, PROCESSED_DIR, ensure_dirs  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES, parse_device_list  # noqa: E402
from cryoml.pdk_extract import (PARAMS7, ThetaBox, eval_params,  # noqa: E402
                                flatten_paper_curves, residual_fn)
from cryoml.utils import device_tag, get_logger, set_seed  # noqa: E402

logger = get_logger("pdk_mlp_tandem")

SYNTH = PROCESSED_DIR / "pdk_synth"
DEFAULT_EMU_DIR = OUT_DIR / "pdk_ml2"


class MultiHeadInverse(nn.Module):
    """One curve encoder with several parameter-region expert heads."""

    def __init__(
        self,
        n_inputs: int,
        hidden: tuple[int, ...],
        anchors: torch.Tensor,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_inputs
        for out_width in hidden:
            layers.extend([
                nn.Linear(width, out_width),
                nn.LayerNorm(out_width),
                nn.GELU(),
            ])
            width = out_width
        self.encoder = nn.Sequential(*layers)
        self.output = nn.Linear(width, len(anchors) * 7)
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)
        self.register_buffer("anchors", anchors.clamp(-3.5, 3.5))
        self.n_heads = len(anchors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.output(self.encoder(x)).reshape(-1, self.n_heads, 7)
        return torch.clamp(self.anchors.unsqueeze(0) + raw, -3.5, 3.5)


def farthest_anchors(z: np.ndarray, n_heads: int, seed: int) -> np.ndarray:
    """Choose deterministic, well-separated expert regions in z-space."""
    rng = np.random.default_rng(seed)
    pool = z[rng.choice(len(z), size=min(len(z), 3000), replace=False)]
    anchors = [pool[np.argmin(np.linalg.norm(pool, axis=1))]]
    min_dist = np.linalg.norm(pool - anchors[0], axis=1)
    for _ in range(1, n_heads):
        nxt = pool[np.argmax(min_dist)]
        anchors.append(nxt)
        min_dist = np.minimum(min_dist, np.linalg.norm(pool - nxt, axis=1))
    return np.asarray(anchors, dtype=np.float32)


def emulator_rrms(
    emu: nn.Module,
    z: torch.Tensor,
    meas_t: torch.Tensor,
    curve_layout: list[tuple[int, int, float]],
    n_curves: int,
) -> torch.Tensor:
    """Paper-exact differentiable objective for one row per z candidate."""
    pred = mlx.inv_slog_t(emu(z))
    losses = []
    for a, b, denominator in curve_layout:
        rmse = torch.sqrt(torch.mean(
            (pred[:, a:b] - meas_t[a:b].unsqueeze(0)) ** 2, dim=1
        ) + 1e-30)
        losses.append(rmse / denominator)
    return torch.stack(losses, dim=1).sum(dim=1) / n_curves


def load_emulator(tag: str, device: torch.device, emu_dir: Path) -> tuple:
    path = emu_dir / f"emu_{tag}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"missing frozen emulator {path}; run scripts/pdk_ml_extract.py first"
        )
    blob = torch.load(path, map_location=device, weights_only=False)
    emu = mlx.mlp([7, *blob["emu_sizes"], blob["P"]]).to(device)
    emu.load_state_dict(blob["state"])
    emu.eval()
    for param in emu.parameters():
        param.requires_grad_(False)
    return emu, blob


def ensemble_rrms(
    emus: list[nn.Module],
    z: torch.Tensor,
    meas_t: torch.Tensor,
    curve_layout: list[tuple[int, int, float]],
    n_curves: int,
) -> torch.Tensor:
    """Pessimistic target loss across independently trained emulators."""
    losses = torch.stack([
        emulator_rrms(emu, z, meas_t, curve_layout, n_curves) for emu in emus
    ])
    if len(emus) == 1:
        return losses[0]
    return losses.mean(dim=0) + 0.5 * losses.std(dim=0)


def train_inverse(
    net: MultiHeadInverse,
    emu: nn.Module,
    x: torch.Tensor,
    z: torch.Tensor,
    assignments: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    steps: int,
    batch_size: int,
    lr: float,
    recon_weight: float,
    seed: int,
) -> dict[str, float]:
    """Train expert heads with supervised and tandem reconstruction losses."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    n = len(x)
    best = np.inf
    final_param = final_recon = np.inf
    net.train()
    for step in range(steps):
        idx_cpu = torch.randint(0, n, (min(batch_size, n),), generator=generator)
        idx = idx_cpu.to(x.device)
        clean = x[idx]
        # Denoising input: signed-log noise plus sparse point masking.
        noisy = clean + 0.08 * torch.randn_like(clean)
        mask = torch.rand_like(noisy) < 0.01
        noisy = torch.where(mask, mean.expand_as(noisy), noisy)
        xin = (noisy - mean) / scale

        pred_all = net(xin)
        head = assignments[idx]
        pred = pred_all[torch.arange(len(idx), device=x.device), head]
        param_loss = nn.functional.smooth_l1_loss(pred, z[idx], beta=0.5)
        recon_loss = nn.functional.mse_loss(emu(pred), clean)
        loss = param_loss + recon_weight * recon_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sched.step()
        final_param = float(param_loss.detach())
        final_recon = float(recon_loss.detach())
        best = min(best, float(loss.detach()))
        if (step + 1) % 500 == 0:
            logger.info(
                "inverse step %d/%d loss %.4f param %.4f recon %.4f",
                step + 1, steps, float(loss.detach()), final_param, final_recon,
            )
    net.eval()
    return {
        "train_best_loss": best,
        "train_final_param_loss": final_param,
        "train_final_recon_loss": final_recon,
    }


def calibrate_target(
    net: MultiHeadInverse,
    emus: list[nn.Module],
    meas_slog_t: torch.Tensor,
    meas_t: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    curve_layout: list[tuple[int, int, float]],
    n_curves: int,
    steps: int,
    lr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate only the expert output layer through the frozen emulator."""
    xin = ((meas_slog_t - mean) / scale).unsqueeze(0)
    with torch.no_grad():
        raw_z = net(xin).squeeze(0)
        raw_loss = ensemble_rrms(emus, raw_z, meas_t, curve_layout, n_curves)

    for param in net.parameters():
        param.requires_grad_(False)
    for param in net.output.parameters():
        param.requires_grad_(True)
    opt = torch.optim.Adam(net.output.parameters(), lr=lr)
    best_z = raw_z.detach().clone()
    best_loss = raw_loss.detach().clone()

    for _ in range(steps):
        z = net(xin).squeeze(0)
        loss_per = ensemble_rrms(emus, z, meas_t, curve_layout, n_curves)
        # Each expert is optimized independently; the small anchor prevents
        # unstable excursions into poorly trained emulator corners.
        anchor = 2e-4 * torch.mean((z - raw_z.detach()) ** 2, dim=1)
        loss = (loss_per + anchor).sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.output.parameters(), 10.0)
        opt.step()
        with torch.no_grad():
            improved = loss_per < best_loss
            best_loss[improved] = loss_per[improved]
            best_z[improved] = z[improved]

    return (
        raw_z.detach().cpu().numpy().astype(np.float64),
        raw_loss.detach().cpu().numpy().astype(np.float64),
        best_z.detach().cpu().numpy().astype(np.float64),
        best_loss.detach().cpu().numpy().astype(np.float64),
    )


def extract_device(
    d,
    tdev: str,
    emu_dir: Path,
    extra_emu_dir: Path | None,
    hidden: tuple[int, ...],
    n_heads: int,
    train_steps: int,
    batch_size: int,
    calibration_steps: int,
    calibration_lr: float,
    n_validate: int,
    n_polish: int,
    max_nfev: int,
    seed: int,
) -> tuple[dict, list[np.ndarray]]:
    from scipy.optimize import least_squares

    tag = device_tag(d.dev_type, d.L_um, d.W_um)
    started = time.time()
    set_seed(seed)
    device = torch.device(tdev)
    emu, blob = load_emulator(tag, device, emu_dir)
    emus = [emu]
    if extra_emu_dir is not None and (extra_emu_dir / f"emu_{tag}.pt").exists():
        extra_emu, extra_blob = load_emulator(tag, device, extra_emu_dir)
        if extra_blob["P"] == blob["P"]:
            emus.append(extra_emu)

    data = np.load(SYNTH / f"{tag}.npz", allow_pickle=True)
    ids, ok = data["IDS"], data["ok"]
    meas, slices = data["meas"], data["slices"]
    bin_index = int(data["bin_index"])
    published = {p: float(v) for p, v in zip(PARAMS7, data["published"])}
    box = ThetaBox(dev_type=d.dev_type, bin_index=bin_index, published=published)
    theta = data["THETA"].astype(np.float64)
    z_np = np.stack([
        box.params_to_z({p: row[i] for i, p in enumerate(PARAMS7)})
        for row in theta
    ])[ok].astype(np.float32)

    kept_mask = np.zeros(len(meas), dtype=bool)
    curve_layout = []
    offset = 0
    for a, b in slices:
        denominator = float(np.mean(np.abs(meas[a:b])))
        if denominator > 0 and np.isfinite(denominator):
            kept_mask[a:b] = True
            curve_layout.append((offset, offset + b - a, denominator))
            offset += b - a
    x_np = mlx.slog(ids[ok][:, kept_mask]).astype(np.float32)
    if x_np.shape[1] != blob["P"]:
        raise RuntimeError(
            f"{tag}: synth/emulator layout mismatch {x_np.shape[1]} != {blob['P']}"
        )

    anchors = farthest_anchors(z_np, n_heads, seed)
    assignments_np = np.argmin(
        np.sum((z_np[:, None, :] - anchors[None, :, :]) ** 2, axis=2), axis=1
    ).astype(np.int64)
    x = torch.tensor(x_np, device=device)
    z = torch.tensor(z_np, device=device)
    assignments = torch.tensor(assignments_np, device=device)
    mean = x.mean(dim=0, keepdim=True)
    scale = x.std(dim=0, keepdim=True).clamp_min(0.25)

    net = MultiHeadInverse(
        x.shape[1], hidden, torch.tensor(anchors, device=device)
    ).to(device)
    train_stats = train_inverse(
        net, emu, x, z, assignments, mean, scale,
        steps=train_steps, batch_size=batch_size, lr=8e-4,
        recon_weight=0.2, seed=seed,
    )

    meas_t = torch.tensor(meas[kept_mask], dtype=torch.float32, device=device)
    meas_slog_t = torch.tensor(
        mlx.slog(meas[kept_mask]), dtype=torch.float32, device=device
    )
    raw_z, raw_surrogate, adapted_z, adapted_surrogate = calibrate_target(
        net, emus, meas_slog_t, meas_t, mean, scale, curve_layout, len(slices),
        steps=calibration_steps, lr=calibration_lr,
    )

    curves = load_device_curves(d)
    flat = flatten_paper_curves(curves)

    def validate(candidate: np.ndarray) -> tuple[float, dict, list[np.ndarray]]:
        params = box.z_to_params(candidate)
        metrics, sims = eval_params(
            d.dev_type, d.L_um, d.W_um, bin_index, curves, params
        )
        rrms = float(metrics["rrms"]) if np.isfinite(metrics["rrms"]) else np.inf
        return rrms, params, sims

    candidates: list[tuple[str, float, np.ndarray]] = []
    for label, zs, losses in (
        ("tandem_raw", raw_z, raw_surrogate),
        ("tandem_adapted", adapted_z, adapted_surrogate),
    ):
        for candidate, loss in sorted(zip(zs, losses), key=lambda item: item[1]):
            if any(np.linalg.norm(candidate - old[2]) < 0.25 for old in candidates):
                continue
            candidates.append((label, float(loss), candidate))
            if sum(item[0] == label for item in candidates) >= n_validate:
                break

    validated = []
    for label, surrogate_loss, candidate in candidates:
        rrms, params, sims = validate(candidate)
        validated.append((rrms, label, surrogate_loss, candidate, params, sims))
    validated.sort(key=lambda item: item[0])

    published_metrics, _ = eval_params(
        d.dev_type, d.L_um, d.W_um, bin_index, curves, published
    )
    published_rrms = float(published_metrics["rrms"])
    results = {
        "published": {"rrms": published_rrms, "params": published},
    }
    for label in ("tandem_raw", "tandem_adapted"):
        rows = [item for item in validated if item[1] == label]
        if rows:
            best = min(rows, key=lambda item: item[0])
            results[label] = {"rrms": best[0], "params": best[4]}

    residual = lambda candidate: residual_fn(  # noqa: E731
        candidate, box, d.dev_type, d.L_um, d.W_um, bin_index, curves, flat
    )
    polished = []
    used = []
    for rrms, label, _, candidate, _, _ in validated:
        if not np.isfinite(rrms):
            continue
        if any(np.linalg.norm(candidate - old) < 0.5 for old in used):
            continue
        used.append(candidate)
        try:
            solution = least_squares(
                residual, candidate, method="trf", jac="2-point",
                diff_step=2e-2, max_nfev=max_nfev,
            )
            rrms2, params2, sims2 = validate(solution.x)
            if np.isfinite(rrms2):
                polished.append((rrms2, params2, sims2))
        except Exception:  # noqa: BLE001
            pass
        if len(used) >= n_polish:
            break
    if polished:
        best = min(polished, key=lambda item: item[0])
        results["tandem_best+fd"] = {"rrms": best[0], "params": best[1]}

    model_keys = [key for key in results if key != "published"]
    best_method = min(model_keys, key=lambda key: results[key]["rrms"])
    best_rrms, best_params, best_sims = validate(
        box.params_to_z(results[best_method]["params"])
    )
    results[best_method]["rrms"] = best_rrms

    rec = {
        "device": tag,
        "dev_type": d.dev_type,
        "L_um": d.L_um,
        "W_um": d.W_um,
        "bin_index": bin_index,
        "n_synth": int(ok.sum()),
        "n_heads": n_heads,
        "n_target_emulators": len(emus),
        "hidden": list(hidden),
        "seed": seed,
        "train_steps": train_steps,
        "calibration_steps": calibration_steps,
        "calibration_lr": calibration_lr,
        "n_validate": n_validate,
        "n_polish": n_polish,
        "max_nfev": max_nfev,
        **train_stats,
        "methods": {
            key: {"rrms": float(value["rrms"])} for key, value in results.items()
        },
        "params_by_method": {
            key: value["params"] for key, value in results.items()
        },
        "best_method": best_method,
        "rrms": float(best_rrms),
        "start_rrms": float(published_rrms),
        "classical_rrms": None,
        "runtime_s": round(time.time() - started, 1),
    }
    return rec, best_sims


def write_summary(out_dir: Path) -> dict:
    cma = {
        row["device"]: float(row["rrms"])
        for path in (OUT_DIR / "pdk_cma").glob("*.json")
        if isinstance((row := json.loads(path.read_text())), dict)
        and "device" in row and "rrms" in row
    }
    published = {
        row["device"]: float(row["rrms"])
        for row in json.loads(
            (OUT_DIR / "pdk_baseline" / "pdk_baseline.json").read_text()
        )["devices"]
    }
    rows = []
    for path in sorted(out_dir.glob("ml_*.json")):
        row = json.loads(path.read_text())
        if np.isfinite(row.get("rrms", np.nan)):
            row["classical_rrms"] = cma.get(row["device"])
            row["methods"]["published"]["rrms"] = published[row["device"]]
            row["start_rrms"] = published[row["device"]]
            path.write_text(json.dumps(row, indent=2))
            rows.append(row)
    summary = {"n_devices": len(rows)}
    for key in ("tandem_raw", "tandem_adapted", "tandem_best+fd"):
        values = [
            row["methods"][key]["rrms"] for row in rows
            if key in row["methods"] and np.isfinite(row["methods"][key]["rrms"])
        ]
        if values:
            summary[f"mean_{key}"] = float(np.mean(values))
            summary[f"n_{key}"] = len(values)
    if rows:
        scores = np.asarray([row["rrms"] for row in rows])
        classical = np.asarray([row["classical_rrms"] for row in rows])
        published = np.asarray([
            row["methods"]["published"]["rrms"] for row in rows
        ])
        summary.update({
            "mean_rrms": float(np.mean(scores)),
            "published_mean_rrms": float(np.mean(published)),
            "wins_vs_published": int(np.sum(scores < published - 1e-8)),
            "classical_mean_rrms": float(np.mean(classical)),
            "delta_vs_classical": float(np.mean(scores) - np.mean(classical)),
            "wins_vs_classical": int(np.sum(scores < classical - 1e-8)),
            "ties_vs_classical": int(np.sum(np.abs(scores - classical) <= 1e-8)),
            "losses_vs_classical": int(np.sum(scores > classical + 1e-8)),
        })
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "device", "published", "tandem_raw", "tandem_adapted",
            "tandem_best_fd", "best", "classical",
        ])
        for row in rows:
            methods = row["methods"]
            writer.writerow([
                row["device"],
                methods["published"]["rrms"],
                methods.get("tandem_raw", {}).get("rrms", np.nan),
                methods.get("tandem_adapted", {}).get("rrms", np.nan),
                methods.get("tandem_best+fd", {}).get("rrms", np.nan),
                row["rrms"],
                row["classical_rrms"],
            ])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default=None)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--emu-dir", default=str(DEFAULT_EMU_DIR))
    parser.add_argument("--extra-emu-dir", default="")
    parser.add_argument("--out-dir", default=str(OUT_DIR / "pdk_mlp_tandem"))
    parser.add_argument("--hidden", default="256")
    parser.add_argument("--heads", type=int, default=2048)
    parser.add_argument("--train-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--calibration-steps", type=int, default=600)
    parser.add_argument("--calibration-lr", type=float, default=0.05)
    parser.add_argument("--n-validate", type=int, default=20)
    parser.add_argument("--n-polish", type=int, default=5)
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hidden = tuple(int(width) for width in args.hidden.split(","))
    run_config = {
        "method": "tandem_multihead_mlp",
        "devices": "all_18" if args.devices is None else args.devices,
        "torch_device": args.device,
        "emu_dir": str(Path(args.emu_dir)),
        "extra_emu_dir": str(Path(args.extra_emu_dir)) if args.extra_emu_dir else "",
        "hidden": list(hidden),
        "heads": args.heads,
        "train_steps": args.train_steps,
        "batch_size": args.batch_size,
        "calibration_steps": args.calibration_steps,
        "calibration_lr": args.calibration_lr,
        "n_validate": args.n_validate,
        "n_polish": args.n_polish,
        "max_nfev": args.max_nfev,
        "seed": args.seed,
    }
    config_path = out_dir / "run_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing != run_config:
            raise RuntimeError(
                f"{out_dir} already contains a different run configuration; "
                "use a new output directory"
            )
    else:
        config_path.write_text(json.dumps(run_config, indent=2))

    for d in parse_device_list(args.devices):
        tag = device_tag(d.dev_type, d.L_um, d.W_um)
        result_path = out_dir / f"ml_{tag}.json"
        if args.resume and result_path.exists():
            old = json.loads(result_path.read_text())
            if np.isfinite(old.get("rrms", np.nan)):
                logger.info("%-22s already complete; skipping", tag)
                continue
        rec, sims = extract_device(
            d, args.device, Path(args.emu_dir),
            Path(args.extra_emu_dir) if args.extra_emu_dir else None,
            hidden, args.heads,
            args.train_steps, args.batch_size, args.calibration_steps,
            args.calibration_lr,
            args.n_validate, args.n_polish, args.max_nfev, args.seed,
        )
        result_path.write_text(json.dumps(rec, indent=2))
        np.savez(
            out_dir / f"sims_{tag}.npz",
            **{f"sim_{i}": np.asarray(sim) for i, sim in enumerate(sims)},
        )
        logger.info(
            "%-22s raw %.3f adapted %.3f final %.3f [%s] %ss",
            tag,
            rec["methods"].get("tandem_raw", {}).get("rrms", np.nan),
            rec["methods"].get("tandem_adapted", {}).get("rrms", np.nan),
            rec["rrms"],
            rec["best_method"],
            rec["runtime_s"],
        )

    summary = write_summary(out_dir)
    print("\n=== TANDEM MULTI-HEAD MLP ===")
    for key, value in summary.items():
        print(f"  {key:28s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
