#!/usr/bin/env python3
"""Is the cold-loop surrogate pipeline limited by the surrogate or the search?

For one round, score with the frozen surrogate (paper RRMS of surrogate
current vs measured) and record the NGSpice RRMS of:

* the pipeline's selected card (``surrogate_fd``), and
* a reference card set, e.g. the direct NGSpice fit in the same box.

If the surrogate rates the reference cards worse than its own pick while
NGSpice rates them better, the surrogate's error, not the search, is the
bottleneck.  Inference only; no NGSpice calls.

  .venv/bin/python scripts/research/surrogate_gap.py --round-dir \
      out/cold_loop/full/round_1 --bank data/processed/cold_loop/full_r1 \
      --set full --reference out/research/ngspice_direct/full_r1_direct_cold/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import foundation_parameter_study as fps  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.foundation import FoundationSurrogate, decode_surrogate_output  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-dir", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--set", default="full")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    round_dir = Path(args.round_dir)
    cache = round_dir / "cache"
    metadata = fps.load_metadata(cache) if hasattr(fps, "load_metadata") else \
        json.loads((cache / "metadata.json").read_text())
    device = torch.device("cpu")
    checkpoint = torch.load(round_dir / "extract" / "forward_model.pt",
                            map_location=device, weights_only=False)
    model = FoundationSurrogate(checkpoint["n_parameters"],
                                checkpoint["n_points"],
                                tuple(checkpoint["hidden"]))
    model.load_state_dict(checkpoint["state"])
    model.eval()
    tensors = fps.training_tensors(cache, device)
    reference = json.loads(Path(args.reference).read_text())["devices"]
    rows = {}
    for dev in PAPER_DEVICES:
        tag = device_tag(dev.dev_type, dev.L_um, dev.W_um)
        context = fps.build_device_context(dev, Path(args.bank), metadata,
                                           args.set)
        picked = json.loads((round_dir / "extract" / f"ml_{tag}.json")
                            .read_text())["methods"]["surrogate_fd"]
        cards = {"pipeline": (picked["params"], picked["rrms"]),
                 "reference": (reference[tag]["params"], reference[tag]["rrms"])}
        row = {}
        for name, (params, ngspice_rrms) in cards.items():
            local = context.box.params_to_local(
                {k: v for k, v in params.items() if k in context.box.names})
            clipped = np.clip(local, -1, 1)
            features = torch.tensor(
                np.concatenate([clipped, context.geometry])[None],
                dtype=torch.float32)
            with torch.no_grad():
                _slog, current = decode_surrogate_output(
                    model(features), tensors.slog_mean, tensors.slog_std)
            sur = fps.score_canonical(context, current[0].numpy())["rrms"]
            row[name] = {"surrogate_rrms": float(sur),
                         "ngspice_rrms": float(ngspice_rrms),
                         "max_abs_local": float(np.max(np.abs(local)))}
        rows[tag] = row
        print(tag, json.dumps(row))
    summary = {}
    for name in ("pipeline", "reference"):
        for key in ("surrogate_rrms", "ngspice_rrms"):
            summary[f"{name}_{key}"] = float(np.mean([r[name][key] for r in rows.values()]))
    summary["devices_where_surrogate_prefers_pipeline_but_ngspice_prefers_reference"] = int(sum(
        r["pipeline"]["surrogate_rrms"] < r["reference"]["surrogate_rrms"]
        and r["reference"]["ngspice_rrms"] < r["pipeline"]["ngspice_rrms"]
        for r in rows.values()))
    print(json.dumps(summary, indent=1))
    out = Path(args.out) if args.out else ROOT / "out" / "research" / (
        f"surrogate_gap_{round_dir.parent.name}_{round_dir.name}.json")
    out.write_text(json.dumps({"summary": summary, "devices": rows}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
