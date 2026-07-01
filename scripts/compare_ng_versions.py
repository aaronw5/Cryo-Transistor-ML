#!/usr/bin/env python3
"""Compare ngspice-41 (exact ogzamour recipe) vs ngspice-46 on the published
cards. Question: does the exact ogzamour ngspice-41 setup match the paper
better?

Both binaries run the published baseline. Where the two versions select the
same native bin, the per-device RRMS is byte-identical (the BSIM4 evaluation
is version-independent); where they differ it is purely native bin selection
on the overlapping pMOS boxes."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml import spice_pdk  # noqa: E402
from cryoml.config import OUT_TABLES  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.metrics import device_rrms  # noqa: E402
from cryoml.spice_pdk import ensure_pdk77k, find_bin_index, simulate_pdk  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402

NG46 = os.environ.get("NG46_BIN", "/opt/homebrew/bin/ngspice")
NG41 = os.environ.get("NG41_BIN",
                      "/Users/anrunw/cryo-ng41/mm/envs/ng41/bin/ngspice")


def use(binary: str) -> None:
    spice_pdk._NGSPICE_BIN = binary


def native(device) -> tuple[int | None, float]:
    curves = load_device_curves(device)
    b = find_bin_index(device.dev_type, device.L_um, device.W_um)
    sims = simulate_pdk(device.dev_type, device.L_um, device.W_um, curves)
    return b, device_rrms(sims, [c.Id for c in curves])["rrms"]


def main() -> int:
    ensure_pdk77k()
    print(f"NG46 = {NG46}\nNG41 = {NG41}\n")
    hdr = (f"{'device':22s} {'bin46':>5s} {'bin41':>5s} "
           f"{'rrms46':>7s} {'rrms41':>7s} {'|drrms|':>8s}  note")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for d in PAPER_DEVICES:
        tag = device_tag(d.dev_type, d.L_um, d.W_um)
        use(NG46)
        b46, r46 = native(d)
        use(NG41)
        b41, r41 = native(d)
        same_bin = (b46 == b41)
        drrms = abs(r46 - r41)
        note = "same bin" if same_bin else "BIN DIFFERS"
        rows.append(dict(device=tag, dev_type=d.dev_type, bin46=b46, bin41=b41,
                         rrms46=r46, rrms41=r41, same_bin=same_bin))
        print(f"{tag:22s} {str(b46):>5s} {str(b41):>5s} "
              f"{r46:7.3f} {r41:7.3f} {drrms:8.5f}  {note}")
    print("-" * len(hdr))
    m46 = float(np.mean([x["rrms46"] for x in rows]))
    m41 = float(np.mean([x["rrms41"] for x in rows]))
    same = [x for x in rows if x["same_bin"]]
    max_d_same = max((abs(x["rrms46"] - x["rrms41"]) for x in same), default=0.0)
    print(f"native mean   ng46 = {m46:.4f}   ng41 = {m41:.4f}   paper = 0.2788")
    print(f"devices where both versions pick the SAME bin: {len(same)}/18; "
          f"max |rrms46-rrms41| among them = {max_d_same:.2e}")
    summary = dict(ng46_mean=m46, ng41_mean=m41, paper_mean=0.2788,
                   n_same_bin=len(same),
                   max_rrms_diff_same_bin=max_d_same, devices=rows)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    out = OUT_TABLES / "ng41_vs_ng46.json"
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
