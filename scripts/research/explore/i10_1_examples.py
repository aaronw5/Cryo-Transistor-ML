#!/usr/bin/env python3
"""I10-1 example curves: NGSpice re-simulation of stored NIST DEV cards for the
example devices of the report batch (scripts/research/plot_explore_i10_1.py).

Devices (``pick``): per type (nfet_g5v0d10v5, pfet_g5v0d10v5) the device whose
ln(RRMS_A1 / RRMS_A0_rtdeck) (K2 log ratio, lodo_nist/results.json) is nearest
the type median, among devices with finite A0_rtdeck, A1, A2 and R_oracle.

Per device, with the I10-1 deck (``cryoml.explore.i10_1_rt4k.simulate``: ngspice-41
arm64, open_pdks sky130 tt card with the device's bin only, deck 30 C), on the
device's 4 K Vb = 0 curves:

  A0_rtdeck  C0_RT card (start_nist/records_C0_RT meta params)
  A1         lodo_nist/cards.json A1 card
  A2         A1 NGSpice output multiplied in Python by exp(g) of the fold's stored
             M_bias correction (lodo_nist/corrections.json.gz, key = geometry group)
  R_oracle   the device's own C4 card (start_nist/records_C4 meta params)

Check: strict paper RRMS (e1_correction.score_device) recomputed from these
curves vs lodo_nist/results.json per_device.

Output: out/research/explore/I10_1/examples/examples.json (measured 4 K currents
included for plotting).

  PYTHONPATH=src .venv/bin/python -I scripts/research/explore/i10_1_examples.py
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("i10_1_driver", ROOT / "scripts/research/explore/i10_1_rt4k.py")
drv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(drv)
m, b = drv.m, drv.b

from cryoml.explore.e1_correction import score_device  # noqa: E402
from cryoml.explore.i3_2_bias import correction_from_json  # noqa: E402

OUT = m.OUT
TARGET = OUT / "examples" / "examples.json"
ARMS = ("A0_rtdeck", "A1", "A2", "R_oracle")


def pick(per: dict) -> list[str]:
    ok = lambda v: v is not None and math.isfinite(v) and v > 0  # noqa: E731
    tags = [t for t in per["A1"] if all(ok(per[a][t]) for a in ARMS)]
    lr = {t: math.log(per["A1"][t] / per["A0_rtdeck"][t]) for t in tags}
    out = []
    for fam in ("etest_nfet", "etest_pfet"):
        sel = {t: v for t, v in lr.items() if t.startswith(fam)}
        med = float(np.median(list(sel.values())))
        out.append(min(sorted(sel), key=lambda t: abs(sel[t] - med)))
    return out


def main() -> int:
    t0 = time.time()
    res = json.loads((OUT / "lodo_nist/results.json").read_text())
    cards = json.loads((OUT / "lodo_nist/cards.json").read_text())
    with gzip.open(OUT / "lodo_nist/corrections.json.gz", "rt") as f:
        corrs = json.load(f)
    rec_rt = drv.load_fit_records(OUT / "start_nist/records_C0_RT")
    rec4 = drv.load_fit_records(OUT / "start_nist/records_C4")
    per = res["per_device"]
    tags = pick(per)
    ds = drv.get_ds("nist", False)
    out = {"id": "I10_1_examples", "simulator": b._sp()._NGSPICE_BIN, "deck_temp_C": m.DECK_RT_C,
           "rule": "per type the device with ln(A1 / A0_rtdeck) nearest the type median (lodo_nist/results.json), "
                   "among devices with finite A0_rtdeck, A1, A2, R_oracle",
           "devices": {}, "checks": {}}
    for tag in tags:
        dev = ds.dev4[tag]
        cset = {"A0_rtdeck": rec_rt[tag].meta["params"], "A1": cards["A1"][tag], "R_oracle": rec4[tag].meta["params"]}
        sims, scores = {}, {}
        for arm, params in cset.items():
            s = ds.simulate(tag, params, m.DECK_RT_C)
            rec = ds.record(tag, s, params)
            sims[arm] = [np.asarray(x, float) for x in s]
            scores[arm] = float(score_device(rec, [c.sim for c in rec.curves])["rrms"])
            if arm == "A1":
                corr = correction_from_json(corrs[ds.group(tag)])
                sims["A2"] = [np.asarray(x, float) for x in corr.apply(rec)]
                scores["A2"] = float(score_device(rec, sims["A2"])["rrms"])
        out["checks"][tag] = {a: {"recomputed": scores[a], "stored": per[a][tag],
                                  "rel_diff": abs(scores[a] - per[a][tag]) / per[a][tag]} for a in ARMS}
        rec0 = ds.record(tag, sims["A1"], cards["A1"][tag])
        rows = []
        for i, (c, r) in enumerate(zip(b.to_pipeline_curves(dev), rec0.curves)):
            rows.append({"key": c.meta["key"], "kind": c.kind, "fixed_V": c.fixed,
                         "vg": np.asarray(c.Vg, float).tolist(), "vd": np.asarray(c.Vd, float).tolist(),
                         "meas": np.asarray(c.Id, float).tolist(), "included": bool(r.included),
                         "start": int(r.start), "sim": {a: sims[a][i].tolist() for a in ARMS}})
        binfo = ds.binfo(tag)
        out["devices"][tag] = {"dev_type": dev.dev_type, "polarity": dev.polarity, "W_um": dev.W_um,
                               "L_um": dev.L_um, "bin": int(binfo["bin"]), "covered": bool(binfo["covered"]),
                               "group": ds.group(tag), "scores": scores,
                               "log_ratio_A1_A0_rtdeck": math.log(per["A1"][tag] / per["A0_rtdeck"][tag]),
                               "curves": rows}
        print(f"{tag}: " + ", ".join(f"{a} {scores[a]:.6f} (stored {per[a][tag]:.6f})" for a in ARMS), flush=True)
    out["max_rel_diff"] = max(v["rel_diff"] for c in out["checks"].values() for v in c.values())
    out["runtime_s"] = round(time.time() - t0, 1)
    b.write_json(TARGET, out)
    print(f"max rel diff {out['max_rel_diff']:.2e}; {out['runtime_s']} s -> {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
