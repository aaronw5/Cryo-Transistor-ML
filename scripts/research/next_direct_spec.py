#!/usr/bin/env python3
"""Next-round box for ngspice_direct: re-center every device on the run's
fitted card with the cold-loop rule (interior widths halve, edge widths stay).

  PYTHONPATH=src .venv/bin/python scripts/research/next_direct_spec.py \
      --run full_r1_direct_seeded --round 2 --out out/cold_loop/full_direct/round_2/spec.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cryoml.cold_start import next_round_spec, write_spec  # noqa: E402
from cryoml.config import OUT_DIR  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None)
    parser.add_argument("--summary", default=None, help="ngspice_direct summary.json (instead of --run)")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    path = (Path(args.summary) if args.summary else
            OUT_DIR / "research" / "ngspice_direct" / args.run / "summary.json")
    summary = json.loads(path.read_text())
    previous = json.loads(Path(summary["spec"]).read_text())
    names = previous["selected_parameters"]
    cards = {t: r["params"] for t, r in summary["devices"].items()}
    local = {t: dict(zip(names, r["local"])) for t, r in summary["devices"].items()}
    spec = next_round_spec(previous, cards, local, round_index=args.round)
    spec["source_run"] = args.run or str(path)
    write_spec(spec, Path(args.out))
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
