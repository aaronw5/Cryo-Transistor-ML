#!/usr/bin/env python3
"""E1: write an M correction as Verilog-A text (sketch, not compiled).

Also checks the generated arithmetic: a Python evaluation of the same
unrolled expressions on random feature rows must equal ``Correction.g``.

  PYTHONPATH=src .venv/bin/python scripts/research/explore/e1_export_va.py \
      --corrections out/research/explore/E1/real_dev_lodo/full_corrections.json \
      --out out/research/explore/E1/e1_corr_sky130.va
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from cryoml.explore.e1_correction import Correction, to_verilog_a  # noqa: E402


def check(corr: Correction, n: int = 200) -> float:
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(n, 6)) * corr.sd + corr.mu
    phi[:, 0] = rng.integers(0, 2, n)
    ref = corr.g(phi)
    # unrolled evaluation mirroring to_verilog_a
    z = (phi - corr.mu) / corr.sd
    total = np.zeros(n)
    for layers in corr.nets:
        h = z
        for i, (w, b) in enumerate(layers):
            h = b + h @ w
            if i < len(layers) - 1:
                h = np.tanh(h)
        total += np.tanh(h[:, 0])
    return float(np.max(np.abs(total / len(corr.nets) - ref)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrections", required=True)
    ap.add_argument("--arm", default="M")
    ap.add_argument("--vdd", type=float, default=1.85)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    corr = Correction.from_json(json.loads(Path(args.corrections).read_text())[args.arm])
    text = to_verilog_a(corr, vdd=args.vdd)
    Path(args.out).write_text(text)
    print(f"wrote {args.out}: {len(text.splitlines())} lines; "
          f"max |g_unrolled - g| = {check(corr):.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
