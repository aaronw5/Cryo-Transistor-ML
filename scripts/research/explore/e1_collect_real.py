#!/usr/bin/env python3
"""E1: simulate the best direct-fit 43-parameter cards on the measured grid.

One NGSpice run per device.  Writes ``<out>/real_<split>.{npz,json}``
(``DeviceRec`` records) and checks that the no-correction strict RRMS equals
``card_eval.score_card`` on the same card.

  PYTHONPATH=src .venv/bin/python scripts/research/explore/e1_collect_real.py --split dev
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from cryoml.card_eval import device_by_tag, include_tags, score_card  # noqa: E402
from cryoml.cold_start import device_bins  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.explore.e1_correction import (CurveRec, DeviceRec, save_devices,  # noqa: E402
                                          score_device)
from cryoml.metrics import curve_trim, tag_metric_curves  # noqa: E402
from cryoml.spice_pdk import simulate_pdk  # noqa: E402

SPLIT = ROOT / "out/research/explore/real_split.json"
CARDS = ROOT / "out/research/polished/direct43_r3_fd/real/summary.json"
OUT = ROOT / "out/research/explore/E1"
VDD_SKY130 = 1.85


def collect(tag: str, params: dict) -> DeviceRec:
    device = device_by_tag(tag)
    curves = load_device_curves(device)
    sims = simulate_pdk(device.dev_type, device.L_um, device.W_um, curves,
                        params=params, bin_index=device_bins()[tag])
    index = {id(c): i for i, c in enumerate(curves)}
    frozen = include_tags(tag)
    rec = DeviceRec(tag, "pmos" if device.dev_type == "pmos" else "nmos",
                    float(device.L_um), float(device.W_um), VDD_SKY130, "sky130",
                    meta={"dev_type": device.dev_type, "card": str(CARDS.relative_to(ROOT))})
    for (kind, bias), curve in sorted(tag_metric_curves(curves).items()):
        key = f"{kind}@{bias:g}"
        start, _, den = curve_trim(device.dev_type, device.L_um, device.W_um, kind,
                                   curve.Id)
        rec.curves.append(CurveRec(key, kind, np.asarray(curve.Vg, float),
                                   np.asarray(curve.Vd, float),
                                   np.asarray(sims[index[id(curve)]], float),
                                   np.asarray(curve.Id, float), start, den,
                                   key in frozen and den > 0))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test"), default="dev")
    args = ap.parse_args()
    split = json.loads(SPLIT.read_text())
    tags = split["REAL_DEV" if args.split == "dev" else "REAL_TEST"]
    cards = json.loads(CARDS.read_text())["devices"]
    recs, rows = [], {}
    for tag in tags:
        rec = collect(tag, cards[tag]["params"])
        mine = score_device(rec, [c.sim for c in rec.curves])
        ref = score_card(tag, cards[tag]["params"])
        if not np.isclose(mine["rrms"], ref["rrms"], rtol=1e-9, atol=1e-12):
            raise RuntimeError(f"{tag}: strict score {mine['rrms']} != score_card {ref['rrms']}")
        rows[tag] = {"rrms_c0": mine["rrms"], "stored_rrms": cards[tag]["rrms"],
                     "n_curves": mine["n_curves"]}
        print(tag, json.dumps(rows[tag]), flush=True)
        recs.append(rec)
    OUT.mkdir(parents=True, exist_ok=True)
    save_devices(OUT / f"real_{args.split}", recs)
    if args.split == "dev":
        vals = [r["rrms_c0"] for r in rows.values()]
        print("REAL_DEV C0 all-device mean", float(np.mean(vals)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
