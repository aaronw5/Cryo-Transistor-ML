#!/usr/bin/env python3
"""Re-score ngspice_direct runs through card_eval and count box-edge parameters.

  PYTHONPATH=src .venv/bin/python scripts/research/verify_direct.py RUN [RUN...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from cold_loop import box_for  # noqa: E402
from cryoml.card_eval import score_cards  # noqa: E402
from cryoml.config import OUT_DIR  # noqa: E402


def main() -> int:
    out = {}
    for run in sys.argv[1:]:
        summary = json.loads((OUT_DIR / "research" / "ngspice_direct" / run
                              / "summary.json").read_text())
        spec = json.loads(Path(summary["spec"]).read_text())
        cards = {t: r["params"] for t, r in summary["devices"].items()}
        scored = score_cards(cards)
        edges = {}
        for tag, row in summary["devices"].items():
            box = box_for(spec, tag)
            local = np.asarray(row["local"])
            for i in np.flatnonzero(box.active & (np.abs(local) > 0.98)):
                edges.setdefault(box.names[i], []).append(
                    f"{tag}:{'hi' if local[i] > 0 else 'lo'}")
        out[run] = {"rescored_all_device": scored["all_device_mean_rrms"],
                    "rescored_combined": scored["combined_rrms"],
                    "reported_all_device": summary["all_device_mean_rrms"],
                    "n_edge_params": sum(map(len, edges.values())),
                    "edge_counts": {k: len(v) for k, v in sorted(
                        edges.items(), key=lambda kv: -len(kv[1]))}}
    path = OUT_DIR / "research" / "verify_direct.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**old, **out}, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
