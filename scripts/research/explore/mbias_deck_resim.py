#!/usr/bin/env python3
"""NGSpice re-simulation of the stored cards for the M_bias slide deck
(scripts/make_mbias_slides.py).  No fitting and no training: every card and
every correction is read from the stored run outputs.

Datasets (``--dataset``; sky130: all 18 devices, the 6 REAL_TEST devices
opened on 2026-10-21 by user decision; other datasets: DEV devices only, their
TEST splits are not read):

  sky130      I3-2 all18 run (out/research/explore/I3_2/all18), ngspice-41
              arm64 (canonical) and the ngspice-41 OSDI build.  Arms:
              published 77 K card, starting card (direct43_r3_fd), C0_refit
              card, T1_inloop card, M_bias refit card; M_bias two ways:
              (a) SPICE-native, the fold Verilog-A correction compiled here
              with OpenVAF 23.5.0 and placed in the netlist, (b) ngspice-41
              card current x exp(g) in Python.  Checks against
              I3_2/all18/results.json and reference_scores.json.
  nist4k      I8-1 NIST mds2-2997 4 K, lvt, 77 K sky130 card (stock = the
              card with the device's bin, no fit).
  nist_rtcard I9-1 NIST mds2-2997 4 K, regular Vt, room-temperature sky130
              card (stock = that card, no fit).
  tudelft4    I6-1 TU Delft 40 nm at 4.13 K, CMOSedu BSIM4 example card.
  tudelft295  I6-1 at 295.0 K.
  ihp300      I5-1 IHP SG13G2 300 K, PSP103 on ngspice-46 + OSDI.

Arms for the non-sky130 datasets: stock card, C0_refit card, M_bias refit
card x exp(g) of the fold correction (as the stored runs).  The strict paper
RRMS (e1_correction.score_device) is recomputed and compared with the stored
per-device values.

Output: out/research/explore/mbias_deck/<dataset>/{curves.npz, check.json}

  PYTHONPATH=src .venv/bin/python -I scripts/research/explore/mbias_deck_resim.py --dataset all --workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

EXP = ROOT / "out/research/explore"
OUT = EXP / "mbias_deck"
I32 = EXP / "I3_2"
NG41_OSDI = "/Users/anrunw/cryo-ng41/arm64-osdi/bin/ngspice"
DATASETS = ("sky130", "nist4k", "nist_rtcard", "tudelft4", "tudelft295", "ihp300")


def jload(p: Path):
    return json.loads(Path(p).read_text())


# ---------------------------------------------------------------------------
# sky130 77 K, all 18 devices (I3-2 recipe, leave-one-device-out, all18 run)
# ---------------------------------------------------------------------------
ALL18 = I32 / "all18"
OPENVAF = "/Users/anrunw/cryo-ng41/openvaf/bin/openvaf"      # OpenVAF 23.5.0 (OSDI 0.3), as F1c
OPENVAF_ARGS = ("--target", "aarch64-apple-darwin")


def sky130_records() -> dict:
    from cryoml.explore.e1_correction import load_devices
    recs = load_devices(EXP / "E1/real_dev") + load_devices(EXP / "E1/real_test")
    return {r.tag: r for r in recs}


def sky130_compile(tags: list[str]) -> dict:
    """Fold corrections of the all18 run -> Verilog-A -> .osdi (OpenVAF 23.5)."""
    from cryoml.explore.i3_2_bias import correction_from_json, to_verilog_a_bias
    from cryoml.explore.i3_2_followup import compile_va, osdi_va_text
    corrs = jload(ALL18 / "corrections.json")
    out = {}
    for t in tags:
        text = to_verilog_a_bias(correction_from_json(corrs[f"M_bias|{t}"]), vdd=1.85, legacy=True)
        c = compile_va(osdi_va_text(text), OUT / "sky130/va" / f"m_bias_fold_{t}_osdi.va",
                       OUT / "sky130/osdi" / f"m_bias_fold_{t}.osdi", openvaf=OPENVAF,
                       extra_args=OPENVAF_ARGS)
        out[t] = str(Path(c["osdi"]))
    return out


def sky130_task(task) -> dict:
    tag, osdi = task
    from cryoml.cold_start import device_bins
    from cryoml.config import resolve_ngspice_bin
    from cryoml.explore.e1_correction import score_device
    from cryoml.explore.i2_1_joint import with_sims
    from cryoml.explore.i3_2_bias import correction_from_json
    from cryoml.explore.i3_2_followup import simulate_sky130
    from cryoml.card_eval import device_by_tag
    from cryoml.data_io import load_device_curves
    from cryoml.metrics import tag_metric_curves

    rec = sky130_records()[tag]
    held = jload(ALL18 / "heldout_cards.json")
    start = jload(ROOT / "out/research/polished/direct43_r3_fd/real/summary.json")["devices"]
    fold = correction_from_json(jload(ALL18 / "corrections.json")[f"M_bias|{tag}"])
    device = device_by_tag(tag)
    curves = load_device_curves(device)
    pos = {id(c): i for i, c in enumerate(curves)}
    order = [pos[id(c)] for _, c in sorted(tag_metric_curves(curves).items())]
    b_idx = device_bins()[tag]
    ng41 = resolve_ngspice_bin()

    def run(params, ng, osdi_path=None):
        sims, msgs = simulate_sky130(device.dev_type, device.L_um, device.W_um, curves, params,
                                     b_idx, ng, osdi=osdi_path, vdd=1.85)
        return [np.asarray(sims[i], float) for i in order], msgs

    cards = {"published": None, "start": dict(start[tag]["params"]),
             "C0_refit": dict(held["C0_refit"][tag]), "T1_inloop_card": dict(held["T1_inloop"][tag]),
             "M_bias_card": dict(held["M_bias"][tag])}
    sims, msgs = {}, {}
    for arm, params in cards.items():
        sims[arm], msgs[arm] = run(params, ng41)
    sims["M_bias_native"], msgs["M_bias_native"] = run(cards["M_bias_card"], NG41_OSDI, Path(osdi))
    sims["M_bias_python"] = [np.asarray(x, float) for x in fold.apply(with_sims(rec, sims["M_bias_card"]))]
    scores = {k: float(score_device(rec, v)["rrms"]) for k, v in sims.items()}
    per_curve = {k: score_device(rec, v)["per_curve"] for k, v in sims.items()}
    return {"tag": tag, "rec": rec, "sims": sims, "scores": scores, "per_curve": per_curve,
            "msgs": {k: v for k, v in msgs.items() if v}, "ngspice": [ng41, NG41_OSDI],
            "meta": {"dev_type": device.dev_type, "L_um": float(device.L_um),
                     "W_um": float(device.W_um)}}


def sky130_stored() -> dict:
    res = jload(ALL18 / "results.json")["per_device"]
    rel = jload(ROOT / "out/research/reference_scores.json")
    start = jload(ROOT / "out/research/polished/direct43_r3_fd/real/summary.json")["devices"]
    return {t: {"published": rel["released"]["devices"][t]["rrms"],
                "start": start[t]["rrms"],
                "C0_refit": res["C0_refit"][t],
                "M_bias_python": res["M_bias"][t],
                "M_bias_native": res["M_bias"][t],
                "paper_full43": rel["paper_full43"]["devices"][t]["rrms"],
                "paper_seven": rel["paper_seven"]["devices"][t]["rrms"],
                "T1_inloop": res["T1_inloop"][t]} for t in res["M_bias"]}


# ---------------------------------------------------------------------------
# other datasets
# ---------------------------------------------------------------------------
def nist_task(task) -> dict:
    ds, tag = task
    from cryoml.explore import i8_1_nist as b
    from cryoml.explore.e1_correction import score_device
    from cryoml.explore.i3_2_bias import correction_from_json
    if ds == "nist4k":
        m, out, npz = b, b.OUT, "nist4k.npz"
    else:
        from cryoml.explore import i9_1_nist_rt as m
        out, npz = m.OUT, "nist4k_rt.npz"
    cards = jload(out / "lodo/cards.json")
    corr = correction_from_json(jload(out / "lodo/corrections.json")[f"M_bias|{tag}"])
    binfo = jload(out / "start/bins.json")["devices"][tag]
    dev = b.load_dataset(out / "data" / npz, jload(out / "data/manifest.json"), [tag])[0]
    curves = b.to_pipeline_curves(dev)
    sims, scores = {}, {}
    for arm, params in (("stock", None), ("C0_refit", cards["C0_refit"][tag]),
                        ("M_bias_card", cards["M_bias"][tag])):
        s = m.simulate(dev.polarity, dev.W_um, dev.L_um, int(binfo["bin"]), curves,
                       None if params is None else b.changed_params(params, binfo["stock"]))
        rec = b.to_record(dev, binfo, s, params)
        sims[arm] = [np.asarray(x, float) for x in s]
        if arm == "M_bias_card":
            sims["M_bias"] = [np.asarray(x, float) for x in corr.apply(rec)]
    for arm, s in sims.items():
        scores[arm] = float(score_device(rec, s)["rrms"])
    return {"tag": tag, "rec": rec, "sims": sims, "scores": scores,
            "meta": {"polarity": dev.polarity, "W_um": dev.W_um, "L_um": dev.L_um,
                     "bin": int(binfo["bin"])}}


def tud_task(task) -> dict:
    ds, tag = task
    from cryoml.explore import i6_1_tudelft as m
    from cryoml.explore.e1_correction import score_device
    from cryoml.explore.i4_1_baselines import correction_from_json_any
    t = 4 if ds == "tudelft4" else 295
    lodo = m.OUT / f"lodo_{t}K"
    cards = jload(lodo / "cards.json")
    dev = m.load_dataset(m.OUT / "data/tudelft40.npz", jload(m.OUT / "data/manifest.json"), [tag])[0]
    corr = correction_from_json_any(jload(lodo / "corrections.json")[f"M_bias|{dev.group}"])
    curves = m.fit_curves(dev, t)
    card = m.model_block(dev.dev_type)
    sims = {}
    for arm, params in (("stock", None), ("C0_refit", cards["C0_refit"][tag]),
                        ("M_bias_card", cards["M_bias"][tag])):
        s = m.simulate(dev, card, curves, dev.temps_K[t], params)
        sims[arm] = [np.asarray(x, float) for x in s]
    rec = m.to_record(dev, t, sims["M_bias_card"])
    sims["M_bias"] = [np.asarray(x, float) for x in corr.apply(rec)]
    scores = {a: float(score_device(rec, s)["rrms"]) for a, s in sims.items()}
    return {"tag": tag, "rec": rec, "sims": sims, "scores": scores,
            "meta": {"polarity": dev.dev_type, "W_um": dev.W_um, "L_um": dev.L_um,
                     "group": dev.group, "T_stage_K": float(dev.temps_K[t])}}


def ihp_task(task) -> dict:
    ds, tag = task
    from cryoml.explore import i5_1_ihp_real as m
    from cryoml.explore.e1_correction import score_device
    from cryoml.explore.i4_1_baselines import correction_from_json_any
    t = 300
    out = ROOT / "out/research/explore/I5_1"
    lodo = out / f"lodo_{t}K"
    cards = jload(lodo / "cards.json")
    corr = correction_from_json_any(jload(lodo / "corrections.json")[f"M_bias|{tag}"])
    dev = m.load_dataset(out / "data/ihp_meas.npz", jload(out / "data/manifest.json"), [tag])[0]
    curves = m.fit_curves(dev, t)
    card = m.stock_card(dev, m.BASE_SET)
    sims = {}
    for arm, params in (("stock", None), ("C0_refit", cards["C0_refit"][tag]),
                        ("M_bias_card", cards["M_bias"][tag])):
        s = m.simulate(dev, card, curves, dev.temps_C[t], params)
        sims[arm] = [np.asarray(x, float) for x in s]
    rec = m.to_record(dev, t, sims["M_bias_card"])
    sims["M_bias"] = [np.asarray(x, float) for x in corr.apply(rec)]
    scores = {a: float(score_device(rec, s)["rrms"]) for a, s in sims.items()}
    return {"tag": tag, "rec": rec, "sims": sims, "scores": scores,
            "meta": {"polarity": dev.dev_type, "W_um": dev.W_um, "L_um": dev.L_um, "cls": dev.cls}}


def stored_scores(ds: str) -> tuple[dict, dict]:
    """(stored per-device RRMS by arm name used here, stock-card reference)."""
    if ds in ("nist4k", "nist_rtcard"):
        d = EXP / ("I8_1" if ds == "nist4k" else "I9_1")
        pd = jload(d / "lodo/results.json")["per_device"]
        st = jload(d / "start/summary.json").get("rows", {})
        stock = {t: (r.get("stock_rrms") if isinstance(r, dict) else None) for t, r in st.items()}
    elif ds.startswith("tudelft"):
        t = 4 if ds == "tudelft4" else 295
        pd = jload(EXP / f"I6_1/lodo_{t}K/results.json")["per_device"]
        stock = jload(EXP / "I6_1/stock/stock_rrms.json")["per_device"][str(t)]
    else:
        pd = jload(EXP / "I5_1/lodo_300K/results.json")["per_device"]
        sj = EXP / "I5_1/stock"
        stock = {}
        for p in sj.glob("*.json"):
            obj = jload(p)
            pdv = obj.get("per_device", {})
            if "300" in pdv:
                stock = pdv["300"]
    out = {t: {"C0_refit": pd["C0_refit"].get(t), "M_bias": pd["M_bias"].get(t)}
           for t in pd["M_bias"]}
    return out, stock


def tags_for(ds: str) -> list[str]:
    if ds == "sky130":
        sp = jload(EXP / "real_split.json")
        return sorted(sp["REAL_DEV"] + sp["REAL_TEST"])
    stored, _ = stored_scores(ds)
    # devices with a finite stored M_bias and C0_refit (I8-1: 53 of 57)
    return sorted(t for t, v in stored.items()
                  if v["M_bias"] is not None and v["C0_refit"] is not None
                  and np.isfinite(v["M_bias"]) and np.isfinite(v["C0_refit"]))


TASK = {"sky130": sky130_task, "nist4k": nist_task, "nist_rtcard": nist_task,
        "tudelft4": tud_task, "tudelft295": tud_task, "ihp300": ihp_task}


def run_dataset(ds: str, pool) -> dict:
    t0 = time.time()
    tags = tags_for(ds)
    if ds == "sky130":
        osdi = sky130_compile(tags)
        args = [(t, osdi[t]) for t in tags]
    else:
        args = [(ds, t) for t in tags]
    rows = pool.map(TASK[ds], args, chunksize=1)
    if ds == "sky130":
        stored, stock = sky130_stored(), {}
    else:
        stored, stock = stored_scores(ds)
    arrays, devices, checks = {}, {}, {}
    for row in rows:
        t, rec = row["tag"], row["rec"]
        if ds == "sky130":
            pairs = {"published": "published", "start": "start", "C0_refit": "C0_refit",
                     "M_bias_python": "M_bias_python", "M_bias_native": "M_bias_native"}
        else:
            pairs = {"C0_refit": "C0_refit", "M_bias": "M_bias"}
        ck = {}
        for arm, key in pairs.items():
            s, r = stored[t][key], row["scores"][arm]
            ck[arm] = {"recomputed": r, "stored": s, "abs_diff": abs(r - s),
                       "rel_diff": abs(r - s) / abs(s) if s else float("nan")}
        if ds == "sky130":
            pass
        elif t in stock and stock[t] is not None:
            ck["stock"] = {"recomputed": row["scores"]["stock"], "stored": stock[t],
                           "abs_diff": abs(row["scores"]["stock"] - stock[t])}
        checks[t] = ck
        curves = []
        for i, c in enumerate(rec.curves):
            arrays[f"{t}|{i}|vg"] = np.asarray(c.vg, float)
            arrays[f"{t}|{i}|vd"] = np.asarray(c.vd, float)
            arrays[f"{t}|{i}|meas"] = np.asarray(c.target, float)
            for k, v in row["sims"].items():
                arrays[f"{t}|{i}|{k}"] = np.asarray(v[i], float)
            curves.append({"key": c.key, "kind": c.kind, "start": int(c.start),
                           "included": bool(c.included)})
        devices[t] = {"scores": row["scores"], "curves": curves, "meta": row.get("meta", {}),
                      "stored": stored[t], "per_curve": row.get("per_curve", {}),
                      "msgs": row.get("msgs", {})}
    d = OUT / ds
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "curves.npz", **arrays)
    worst = {}
    for t, ck in checks.items():
        for arm, v in ck.items():
            worst[arm] = max(worst.get(arm, 0.0), v["abs_diff"])
    out = {"dataset": ds, "n_devices": len(tags), "tags": tags, "devices": devices,
           "checks": checks, "max_abs_diff_by_arm": worst,
           "runtime_s": round(time.time() - t0, 1)}
    if ds == "sky130":
        out["ngspice"] = rows[0]["ngspice"]
    (d / "check.json").write_text(json.dumps(out, indent=1, default=float))
    print(ds, len(tags), "devices", json.dumps({k: float(f"{v:.3g}") for k, v in worst.items()}),
          out["runtime_s"], "s", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="all", choices=("all",) + DATASETS)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    sets = DATASETS if args.dataset == "all" else (args.dataset,)
    with get_context("spawn").Pool(min(args.workers, 4)) as pool:
        for ds in sets:
            run_dataset(ds, pool)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
