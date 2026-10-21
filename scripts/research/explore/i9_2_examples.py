#!/usr/bin/env python3
"""I9-2 example curves: NGSpice re-simulation of stored held-out cards for the
example devices of the report batch (scripts/research/plot_explore_i9_2.py).

Dataset: NIST 4 K (I8-1 deck).  Devices (``pick``): the finite device with the
lowest and the device with the highest ln(RRMS_poly5 / RRMS_mlp16)
(I9_2/nist_4K/results.json).  2 devices.

Per device, with the I8-1 deck (``cryoml.explore.i8_1_nist.simulate``:
ngspice-41 arm64, 77 K card with the device's explicit bin, mult = m = 1,
deck temperature 77 K), on all Vb = 0 curves of the measured grid:

  C0_refit  I8_1/lodo/cards.json C0_refit card (held-out refit, no correction)
  mlp16     I8_1/lodo/cards.json M_bias card x exp(g), g = stored I8-1 LODO
            correction 'M_bias|<tag>' (= I9-2 mlp16)
  poly5     I9_2/nist_4K/cards.json poly5 card x exp(g), g = 'poly5|<tag>'
  gam       I9_2/nist_4K/cards.json gam card x exp(g), g = 'gam|<tag>'

exp(g) is applied in Python to the NGSpice output of the refit card.  Check:
the strict paper RRMS (e1_correction.score_device) of each arm is recomputed
from these curves and compared with I9_2/nist_4K/results.json per_device.

Output: out/research/explore/I9_2/examples/examples.json (the measured
currents of the devices on their Vb = 0 curves are included for plotting).

  PYTHONPATH=src .venv/bin/python -I scripts/research/explore/i9_2_examples.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from cryoml import spice_pdk as sp  # noqa: E402
from cryoml.explore import i8_1_nist as m  # noqa: E402
from cryoml.explore.e1_correction import load_devices, score_device  # noqa: E402
from cryoml.explore.i9_2_classes import correction_from_json_i9  # noqa: E402

I8 = m.OUT
I9 = ROOT / "out/research/explore/I9_2"
DATA = I8 / "data"
TARGET = I9 / "examples" / "examples.json"
ARMS = ("C0_refit", "mlp16", "poly5", "gam")


def pick(per: dict) -> list[str]:
    lr = {t: math.log(per["poly5"][t] / per["mlp16"][t]) for t in per["mlp16"]
          if per["mlp16"][t] is not None and per["poly5"][t] is not None}
    return [min(sorted(lr), key=lambda t: lr[t]), max(sorted(lr), key=lambda t: lr[t])]


def main() -> int:
    t0 = time.time()
    res = json.loads((I9 / "nist_4K/results.json").read_text())
    per = res["per_device"]
    cards8 = json.loads((I8 / "lodo/cards.json").read_text())
    corrs8 = json.loads((I8 / "lodo/corrections.json").read_text())
    cards9 = json.loads((I9 / "nist_4K/cards.json").read_text())
    corrs9 = json.loads((I9 / "nist_4K/corrections.json").read_text())
    bins = json.loads((I8 / "start/bins.json").read_text())["devices"]
    recs = {r.tag: r for r in load_devices(I8 / "start/records")}
    tags = pick(per)
    manifest = json.loads((DATA / "manifest.json").read_text())
    devices = {d.tag: d for d in m.load_dataset(DATA / "nist4k.npz", manifest, tags)}
    out = {"id": "I9_2_examples", "simulator": sp._NGSPICE_BIN, "deck_temp_K": m.DECK_TEMP_K,
           "rule": "NIST 4 K: the finite device with the lowest and the device with the highest "
                   "ln(RRMS_poly5 / RRMS_mlp16) (I9_2/nist_4K/results.json)",
           "devices": {}, "checks": {}}
    for tag in tags:
        dev, binfo, rec0 = devices[tag], bins[tag], recs[tag]
        stock = binfo["stock"]
        curves = m.to_pipeline_curves(dev)
        cset = {"C0_refit": (cards8["C0_refit"][tag], None),
                "mlp16": (cards8["M_bias"][tag], corrs8[f"M_bias|{tag}"]),
                "poly5": (cards9["poly5"][tag], corrs9[f"poly5|{tag}"]),
                "gam": (cards9["gam"][tag], corrs9[f"gam|{tag}"])}
        sims, scores = {}, {}
        for arm, (params, cj) in cset.items():
            s = m.simulate(dev.polarity, dev.W_um, dev.L_um, int(binfo["bin"]), curves,
                           m.changed_params(params, stock))
            rec = m.to_record(dev, binfo, s, params)
            if cj is not None:
                corr = correction_from_json_i9(cj)
                out_s = corr.apply(rec)
                sims[f"{arm}_card"] = s
                sims[arm] = out_s
                scores[arm] = float(score_device(rec, out_s)["rrms"])
            else:
                sims[arm] = s
                scores[arm] = float(score_device(rec, s)["rrms"])
        stored = {a: per[a][tag] for a in ARMS}
        out["checks"][tag] = {a: {"recomputed": scores[a], "stored": stored[a],
                                  "rel_diff": abs(scores[a] - stored[a]) / stored[a]} for a in ARMS}
        rows = []
        for i, (c, r) in enumerate(zip(curves, rec0.curves)):
            rows.append({"key": c.meta["key"], "kind": c.kind, "fixed_V": c.fixed,
                         "vg": np.asarray(c.Vg, float).tolist(), "vd": np.asarray(c.Vd, float).tolist(),
                         "meas": np.asarray(c.Id, float).tolist(), "included": bool(r.included),
                         "start": int(r.start),
                         "sim": {k: np.asarray(v[i], float).tolist() for k, v in sims.items()}})
        out["devices"][tag] = {"polarity": dev.polarity, "W_um": dev.W_um, "L_um": dev.L_um,
                               "bin": int(binfo["bin"]), "covered": bool(binfo["covered"]),
                               "stretch": [binfo["stretch_L"], binfo["stretch_W"]],
                               "log_poly5_mlp16": math.log(stored["poly5"] / stored["mlp16"]),
                               "scores": scores, "curves": rows}
        print(f"{tag}: " + ", ".join(f"{a} {scores[a]:.6f} (stored {stored[a]:.6f})" for a in ARMS), flush=True)
    out["max_rel_diff"] = max(v["rel_diff"] for c in out["checks"].values() for v in c.values())
    out["runtime_s"] = round(time.time() - t0, 1)
    m.write_json(TARGET, out)
    print(f"max rel diff {out['max_rel_diff']:.2e}; {out['runtime_s']} s -> {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
