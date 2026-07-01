#!/usr/bin/env python3
"""Build the deployable (shared-bin) variant of a classical control.

The controls fit each device independently, but a real library can ship
only one card per model bin. This applies the same joint shared-bin polish
the ML pipeline uses (minus the emulator search, which the controls do not
have) so the deployable comparison is method-vs-method fair.

  python scripts/make_fd_deploy.py                 # out/pdk_fd  -> out/pdk_fd_deploy
  python scripts/make_fd_deploy.py --src out/pdk_cma --dst out/pdk_cma_deploy
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cryoml.config import OUT_DIR, ensure_dirs  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.metrics import device_rrms  # noqa: E402
from cryoml.utils import device_tag, get_logger  # noqa: E402

from pdk_ml_extract import enforce_shared_bin_cards  # noqa: E402

logger = get_logger("make_fd_deploy")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(OUT_DIR / "pdk_fd"))
    ap.add_argument("--dst", default=str(OUT_DIR / "pdk_fd_deploy"))
    args = ap.parse_args()
    SRC, DST = Path(args.src), Path(args.dst)

    ensure_dirs()
    DST.mkdir(parents=True, exist_ok=True)

    for path in SRC.glob("fd_*.json"):
        rec = json.loads(path.read_text())
        tag = rec["device"]
        out = {
            "device": tag,
            "dev_type": rec["dev_type"],
            "L_um": rec["L_um"],
            "W_um": rec["W_um"],
            "bin_index": rec["bin_index"],
            "paper_reported": rec["paper_reported"],
            "start_rrms": rec["start_rrms"],
            "rrms": rec["rrms"],
            "best_method": "fd",
            "methods": {
                "published": {"rrms": rec["start_rrms"]},
                "fd": {"rrms": rec["rrms"]},
            },
            "params_by_method": {"fd": rec["params"]},
        }
        json.dump(out, open(DST / f"ml_{tag}.json", "w"), indent=2)
        shutil.copy(SRC / f"sims_{tag}.npz", DST / f"sims_{tag}.npz")

    # no emu_<tag>.pt files exist in DST, so the joint stage runs FD-only
    enforce_shared_bin_cards(DST, max_nfev=120, tdev="cpu")

    rows = []
    for d in PAPER_DEVICES:
        tag = device_tag(d.dev_type, d.L_um, d.W_um)
        rec = json.loads((DST / f"ml_{tag}.json").read_text())
        curves = load_device_curves(d)
        saved = np.load(DST / f"sims_{tag}.npz")
        sims = [np.asarray(saved[f"sim_{i}"]) for i in range(len(curves))]
        rec["rrms"] = float(device_rrms(sims, [c.Id for c in curves])["rrms"])
        json.dump(rec, open(DST / f"ml_{tag}.json", "w"), indent=2)
        rows.append(rec)

    summary = {
        "n_devices": len(rows),
        "mean_rrms": float(np.mean([r["rrms"] for r in rows])),
    }
    json.dump(summary, open(DST / "summary.json", "w"), indent=2)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
