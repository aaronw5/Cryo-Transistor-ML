#!/usr/bin/env python3
"""Forward-surrogate capacity / architecture / data study on a cold-loop cache.

Every architecture sees the same cache, the same per-device train/val/test
split as the cold-loop pipeline (``split_indices`` with the study seed), the
same loss (standardized signed-log MSE + curve-normalized linear Huber), and
the same optimizer recipe (AdamW, cosine schedule, early stopping).  Reported:

* ``test_loss``: the pipeline's forward loss on held-out bank samples;
* ``test_curve_nrmse``: per held-out sample, mean over the 11 curves of
  RMSE / max|I| of the simulated curve (an emulator diagnostic, not the paper
  RRMS);
* ``ranking``: the frozen-inclusion paper RRMS the surrogate predicts against
  the measured curves for reference cards whose NGSpice RRMS is known (the
  pipeline's pick and the direct NGSpice fit in the same box).  A useful
  surrogate predicts those scores and ranks the direct fit as better;
* ``end_to_end``: the pipeline's own surrogate-FD extraction
  (``fps.finite_difference_surrogate_extract``: same starts, optimizer,
  budget and NGSpice validation) with the trained model frozen in the loop,
  scored in real NGSpice for all 18 devices.  ``swap_search.py`` runs the same
  search with NGSpice as the evaluator, so the gap to that row is surrogate
  error.  Existing reports without it are filled in from their checkpoints
  (``--end-to-end-only`` does only that).

  .venv/bin/python scripts/research/surrogate_arch_study.py \
      --round-dir out/cold_loop/full/round_1 \
      --bank data/processed/cold_loop/full_r1 --arch paper_mlp,resmlp
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import foundation_parameter_study as fps  # noqa: E402
from cryoml.card_eval import aggregate, device_by_tag  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.foundation import current_from_signed_log  # noqa: E402
from cryoml.operators import build, n_trainable  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402

OUT = ROOT / "out" / "research" / "surrogate_arch"
DEFAULT_REFERENCES = {
    "direct_cold": "out/research/ngspice_direct/full_r1_direct_cold/summary.json",
    "direct_seeded": "out/research/ngspice_direct/full_r1_direct_seeded/summary.json",
}


def batches(n: int, size: int, shuffle: bool, generator=None):
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, size):
        yield order[start:start + size]


def evaluate(model, x, current, ids, tensors, batch_size=2048):
    model.eval()
    loss_sum, nrmse = 0.0, []
    with torch.no_grad():
        for index in batches(len(x), batch_size, False):
            xb, cb, ib = x[index], current[index], ids[index]
            prediction = model(xb)
            loss, _ = fps.forward_training_loss(
                prediction, fps.target_normalized(cb, tensors), cb, ib,
                tensors, 1.0, None)
            loss_sum += float(loss) * len(index)
            pred_current = current_from_signed_log(
                prediction * tensors.slog_std + tensors.slog_mean)
            err = (pred_current - cb).reshape(len(index), 11, -1)
            peak = cb.reshape(len(index), 11, -1).abs().amax(-1)
            # Floor at 1e-3 of the sample's largest curve so near-zero curves
            # (e.g. subthreshold-only) do not dominate the mean.
            peak = torch.maximum(peak, 1e-3 * peak.amax(-1, keepdim=True)).clamp_min(1e-15)
            nrmse.append((err.pow(2).mean(-1).sqrt() / peak).mean(-1).cpu())
    nrmse = torch.cat(nrmse).numpy()
    return {"loss": loss_sum / len(x), "curve_nrmse_mean": float(nrmse.mean()),
            "curve_nrmse_median": float(np.median(nrmse))}


def ranking(model, contexts, references, pipeline_cards, tensors, device):
    rows = {}
    model.eval()
    for tag, context in contexts.items():
        row = {}
        for name, (params, ngspice) in {"pipeline": pipeline_cards[tag],
                                        **{k: v[tag] for k, v in references.items()}}.items():
            local = np.clip(context.box.params_to_local(
                {k: v for k, v in params.items() if k in context.box.names}), -1, 1)
            x = torch.tensor(np.concatenate([local, context.geometry])[None],
                             dtype=torch.float32, device=device)
            with torch.no_grad():
                current = current_from_signed_log(
                    model(x) * tensors.slog_std + tensors.slog_mean)[0].cpu().numpy()
            row[name] = {"surrogate": float(fps.score_canonical(context, current)["rrms"]),
                         "ngspice": float(ngspice)}
        rows[tag] = row
    names = list(next(iter(rows.values())).keys())
    summary = {f"{n}_surrogate_mean": float(np.mean([r[n]["surrogate"] for r in rows.values()]))
               for n in names}
    summary.update({f"{n}_ngspice_mean": float(np.mean([r[n]["ngspice"] for r in rows.values()]))
                    for n in names})
    errors = [abs(r[n]["surrogate"] - r[n]["ngspice"]) for r in rows.values() for n in names]
    summary["mean_abs_score_error"] = float(np.mean(errors))
    summary["direct_ranked_better_by_surrogate"] = int(sum(
        r["direct_cold"]["surrogate"] < r["pipeline"]["surrogate"] for r in rows.values()))
    summary["direct_better_by_ngspice"] = int(sum(
        r["direct_cold"]["ngspice"] < r["pipeline"]["ngspice"] for r in rows.values()))
    return summary, rows


def forward_ms(arch: str, n_in: int, device: torch.device, n: int = 30) -> float:
    """Batch-1 forward, denormalization and host copy (``predict_surrogate``'s
    call pattern) with throwaway weights."""
    model = build(arch, n_in).to(device).eval()
    x = torch.zeros(1, n_in, device=device)

    def call():
        return current_from_signed_log(model(x) * 1.0 + 0.0).cpu()
    with torch.no_grad():
        for _ in range(5):
            call()
        t0 = time.perf_counter()
        for _ in range(n):
            call()
    return (time.perf_counter() - t0) / n * 1e3


def choose_search_device(arch: str, n_in: int, requested: str):
    """The search makes ~10^5-10^6 sequential batch-1 calls, so latency, not
    throughput, decides: ``auto`` times both devices and takes the faster."""
    if requested != "auto":
        return torch.device(requested), {}
    names = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    timings = {name: round(forward_ms(arch, n_in, torch.device(name)), 3)
               for name in names}
    return torch.device(min(timings, key=timings.get)), timings


SEARCH_KNOBS = ("n_starts", "n_validate", "max_nfev", "diff_step",
                "regularization")


def load_checkpoint_model(path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    model = build(checkpoint["arch"], checkpoint["n_in"])
    model.load_state_dict(checkpoint["state"])
    return model.eval()


def search_one(context, model, tensors, device, options: dict) -> dict:
    """The pipeline's surrogate-FD extraction for one device with ``model``
    as the frozen evaluator, every endpoint validated in real NGSpice."""
    record, _ = fps.finite_difference_surrogate_extract(
        context, model, tensors, device,
        **{key: options[key] for key in SEARCH_KNOBS})
    print(f"  e2e {context.tag} rrms {record['rrms']:.4f} surrogate "
          f"{record['surrogate_rrms']:.4f} ({record['selected_candidate_kind']}) "
          f"{record['runtime_s']:.0f}s", flush=True)
    return {key: record[key] for key in (
        "rrms", "sigma", "official_dynamic_rrms", "surrogate_rrms",
        "selected_candidate_kind", "params", "local", "runtime_s")}


def search_job(job) -> tuple[str, dict]:
    """Pool worker: one device on a single-thread CPU copy of the checkpoint."""
    tag, checkpoint, options = job
    torch.set_num_threads(1)
    cpu = torch.device("cpu")
    cache = Path(options["round_dir"]) / "cache"
    metadata = json.loads((cache / "metadata.json").read_text())
    context = fps.build_device_context(
        device_by_tag(tag), Path(options["bank"]), metadata, options["set"])
    return tag, search_one(context, load_checkpoint_model(Path(checkpoint)),
                           fps.training_tensors(cache, cpu), cpu, options)


def end_to_end(path: Path, model, contexts, cache: Path, n_in: int,
               args) -> dict:
    """``pool``: devices in parallel single-thread CPU workers, each loading
    the saved checkpoint; otherwise sequential on cpu/mps (``auto`` takes the
    faster batch-1 device)."""
    t0 = time.time()
    options = {"round_dir": str(cache.parent), "bank": args.bank,
               "set": args.set, **{k: getattr(args, k) for k in SEARCH_KNOBS}}
    timings = {}
    if args.search_device == "pool":
        # Longest-running (pMOS) devices first keeps the pool busy.
        tags = sorted(contexts, key=lambda t: not t.startswith("pmos"))
        workers = min(args.search_workers, len(tags))
        with get_context("spawn").Pool(workers) as pool:
            devices = dict(pool.imap_unordered(search_job, [
                (tag, str(path.with_suffix(".pt")), options) for tag in tags]))
        label = f"cpu_pool_x{workers}"
    else:
        device, timings = choose_search_device(
            json.loads(path.read_text())["arch"], n_in, args.search_device)
        if model is None:
            model = load_checkpoint_model(path.with_suffix(".pt"))
        model = model.to(device).eval()
        tensors = fps.training_tensors(cache, device)
        devices = {tag: search_one(context, model, tensors, device, options)
                   for tag, context in contexts.items()}
        label = str(device)
    devices = {tag: devices[tag] for tag in contexts}
    return {**aggregate(devices), "search_s": round(time.time() - t0, 1),
            "search_device": label, "forward_ms_batch1": timings,
            "options": {k: options[k] for k in SEARCH_KNOBS},
            "devices": devices}


def fill_end_to_end(path: Path, model, contexts, cache: Path, n_in: int,
                    args) -> None:
    """Add ``end_to_end`` to an architecture report once (the in-memory
    model is used only by the sequential path; the pool loads ``.pt``)."""
    report = json.loads(path.read_text())
    if args.skip_end_to_end or "end_to_end" in report:
        print("skip", path.stem)
        return
    result = end_to_end(path, model, contexts, cache, n_in, args)
    report["end_to_end"] = result
    path.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"name": path.stem, **{k: result[k] for k in (
        "all_device_mean_rrms", "nmos_rrms", "pmos_rrms", "combined_rrms",
        "search_s", "search_device", "forward_ms_batch1")}}), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-dir", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--set", default="full")
    parser.add_argument("--arch", default="paper_mlp,resmlp,deeponet,fno")
    parser.add_argument("--train-fraction", default="1.0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--fno-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--tag", default=None)
    # End-to-end search knobs: the cold-loop pipeline defaults.
    parser.add_argument("--n-starts", type=int, default=8)
    parser.add_argument("--n-validate", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument("--diff-step", type=float, default=0.02)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--search-device", choices=("pool", "auto", "cpu", "mps"),
                        default="pool",
                        help="pool: devices in parallel single-thread CPU "
                             "workers; auto: sequential on the faster "
                             "batch-1 device")
    parser.add_argument("--search-workers", type=int, default=10)
    parser.add_argument("--end-to-end-only", action="store_true",
                        help="train nothing; fill end_to_end from checkpoints")
    parser.add_argument("--skip-end-to-end", action="store_true")
    args = parser.parse_args()
    round_dir = Path(args.round_dir)
    cache = round_dir / "cache"
    metadata = json.loads((cache / "metadata.json").read_text())
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tensors = fps.training_tensors(cache, device)
    n_in = metadata["n_parameters"] + 3
    if not args.end_to_end_only:
        features = torch.tensor(np.load(cache / "features.npy"), device=device)
        currents = torch.tensor(np.load(cache / "currents.npy"), device=device)
        device_ids = torch.tensor(
            np.load(cache / "device_ids.npy").astype(np.int64), device=device)
    train_idx, val_idx, test_idx = fps.split_indices(metadata, args.seed)
    contexts = {device_tag(d.dev_type, d.L_um, d.W_um):
                fps.build_device_context(d, Path(args.bank), metadata, args.set)
                for d in PAPER_DEVICES}
    pipeline_cards = {}
    for tag in contexts:
        record = json.loads((round_dir / "extract" / f"ml_{tag}.json").read_text())
        record = record["methods"]["surrogate_fd"]
        pipeline_cards[tag] = (record["params"], record["rrms"])
    references = {}
    for name, path in DEFAULT_REFERENCES.items():
        summary = json.loads((ROOT / path).read_text())
        if Path(summary["spec"]).resolve() != (round_dir / "spec.json").resolve():
            continue
        references[name] = {t: (r["params"], r["rrms"])
                            for t, r in summary["devices"].items()}
    out_dir = OUT / (args.tag or f"{round_dir.parent.name}_{round_dir.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for fraction in [float(f) for f in args.train_fraction.split(",")]:
        rng = np.random.default_rng(args.seed)
        train = np.sort(rng.permutation(train_idx)[:int(round(fraction * len(train_idx)))])
        for arch in args.arch.split(","):
            name = f"{arch}_f{fraction:g}"
            path = out_dir / f"{name}.json"
            if path.exists():
                fill_end_to_end(path, None, contexts, cache, n_in, args)
                continue
            if args.end_to_end_only:
                print("no report/checkpoint for", name)
                continue
            torch.manual_seed(args.seed)
            model = build(arch, features.shape[1]).to(device)
            epochs = args.fno_epochs if arch == "fno" else args.epochs
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                          weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs)
            generator = torch.Generator().manual_seed(args.seed)
            xt, ct, it = features[train], currents[train], device_ids[train]
            best, best_state, bad, history, t0 = math.inf, None, 0, [], time.time()
            for epoch in range(epochs):
                model.train()
                running = 0.0
                for index in batches(len(train), args.batch_size, True, generator):
                    index = index.to(device)
                    xb, cb, ib = xt[index], ct[index], it[index]
                    optimizer.zero_grad()
                    loss, _ = fps.forward_training_loss(
                        model(xb), fps.target_normalized(cb, tensors), cb, ib,
                        tensors, 1.0, None)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    optimizer.step()
                    running += float(loss.detach()) * len(index)
                scheduler.step()
                val = evaluate(model, features[val_idx], currents[val_idx],
                               device_ids[val_idx], tensors)
                history.append({"epoch": epoch + 1, "train": running / len(train),
                                "val": val["loss"]})
                print(f"{name} epoch {epoch + 1} train {running / len(train):.5f} "
                      f"val {val['loss']:.5f} {time.time() - t0:.0f}s", flush=True)
                if val["loss"] < best - 1e-6:
                    best, bad = val["loss"], 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= args.patience:
                        break
            model.load_state_dict(best_state)
            test = evaluate(model, features[test_idx], currents[test_idx],
                            device_ids[test_idx], tensors)
            rank_summary, rank_rows = ranking(model, contexts, references,
                                              pipeline_cards, tensors, device)
            report = {"arch": arch, "train_fraction": fraction,
                      "n_train": int(len(train)), "n_parameters": n_trainable(model),
                      "epochs": len(history), "best_val_loss": best,
                      "test": test, "ranking": rank_summary,
                      "ranking_devices": rank_rows,
                      "train_s": round(time.time() - t0, 1), "history": history}
            path.write_text(json.dumps(report, indent=1) + "\n")
            torch.save({"arch": arch, "state": best_state,
                        "n_in": int(features.shape[1])}, out_dir / f"{name}.pt")
            print(json.dumps({k: report[k] for k in (
                "arch", "train_fraction", "n_parameters", "test", "ranking",
                "train_s")}), flush=True)
            fill_end_to_end(path, model, contexts, cache, n_in, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
