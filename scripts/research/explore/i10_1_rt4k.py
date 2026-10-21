#!/usr/bin/env python3
"""I10-1: predict a device's 4 K I-V from its room-temperature (RT) data.

Pre-registration: ``out/research/explore/I10_1/prereg.json`` (``prereg.md``).
Module: ``cryoml.explore.i10_1_rt4k``.

DEV datasets:
  nist  NIST mds2-2997 ETest nfet_g5v0d10v5 / pfet_g5v0d10v5 (4 K) paired with
        the google sky130 raw-data RT files (``i10_1_etest_rt``); RT fit on
        |V| <= 3.3 V; open_pdks sky130 tt cards, deck 30 C (tnom).
  tud   TU Delft 40 nm I6-1 DEV devices (4.13 K and 295.0 K), I6-1 card and
        stored starting cards.
TEST (1.8 V core ETest devices) is not read by any mode of this script.

Modes:
  data    NIST DEV discovery (DEV types only), RT pairing, checks at both
          temperatures, bins and stock readback -> data/manifest.json,
          data/etest_dev_{4k,rt}.npz (not committed)
  smoke   card checks (DEV devices; core types on a synthetic grid), stock
          scan (finiteness, timing; no RRMS), then the reduced pipeline
          (start, lodo, decide) on smoke devices -> smoke/, smoke_*
  start   --dataset nist: C0_RT (RT data, deck 30 C, from stock), C4 (4 K data,
          deck 30 C, warm start from C0_RT), C4_ind (4 K, deck 30 C, from stock),
          C4_deck (4 K, deck -269.15 C, from C0_RT); --dataset tud: C4 (4 K data,
          295 K stage deck, warm start from the stored 295 K card) and a check
          that the stored I6-1 cards reproduce their stored records
  lodo    --dataset nist|tud: leave-one-geometry-out arms -> lodo_<dataset>/
  decide  validity, contrasts K1-K4, outcome, variant selection -> decision.json

Outside --smoke, start / lodo / decide refuse to run unless prereg.json is
tracked, unmodified, committed in an ancestor of HEAD and not a DRAFT.

Run with ``python -I`` from the repository root:
  .venv/bin/python -I scripts/research/explore/i10_1_rt4k.py --mode data
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from cryoml.explore import i8_1_nist as b  # noqa: E402
from cryoml.explore import i10_1_etest_rt as e  # noqa: E402
from cryoml.explore import i10_1_rt4k as m  # noqa: E402

OUT = m.OUT
DATA = OUT / "data"
NPZ4 = DATA / "etest_dev_4k.npz"
NPZRT = DATA / "etest_dev_rt.npz"
MANIFEST = DATA / "manifest.json"
PREREG = OUT / "prereg.json"
WORK = OUT / "work"                  # gitignored: nested training records
I6 = ROOT / "out" / "research" / "explore" / "I6_1"

START_OPTIONS = {"nfev": 3000, "irls": 3, "diff_step": 0.01}
SMOKE = {"nfev": 100, "irls": 1, "bias_hyper": {"seeds": (0,), "steps": 200}, "per_type": 3}
MIN_INCLUDED_4K = 2                  # D9 (I9-1 D12): measured-only rule, 4 K Vb = 0 curves
MIN_INCLUDED_RT = 2                  # D9: RT curves inside |V| <= 3.3 V
ALPHA = 0.05
CONTRASTS = (("K1", "A1", "A0"), ("K2", "A1", "A0_rtdeck"), ("K3", "A1", "R_pop"), ("K4", "A2", "A1"),
             ("K5", "A3", "A1"))
BONF_LEVEL = 1 - ALPHA / len(CONTRASTS)
PRIMARY_ARMS = ("A0", "A0_rtdeck", "A1", "A2", "A3", "R_pop")
V1_MIN_FINITE = 0.90
V2_MAX_FAILED = 0.10
V2_MIN_PAIRED = 0.80
VARIANTS = {"nist": ("V_ind", "V_mean", "V_sub7", "V_deck"), "tud": ("V_mean", "V_sub7", "V_i6")}
SECONDARY = (("A1_L", "A1"), ("A0", "R_stock4k"), ("A0_rtdeck", "R_stock_rtdeck"), ("A1", "R_oracle"),
             ("R_pop", "A0"), ("A2", "A0"), ("A3", "A0"), ("A2", "R_pop"), ("A3", "R_pop"))


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------
def require_prereg(args) -> dict:
    if args.smoke:
        return {"ok": None, "smoke": True}
    g = b.prediction_guard(PREREG)
    status = json.loads(PREREG.read_text()).get("status", "") if PREREG.exists() else ""
    g["in_force"] = bool(status) and not status.upper().startswith("DRAFT")
    print(f"guard prereg: {json.dumps(g)}", flush=True)
    if not (g["ok"] and g["in_force"]):
        raise SystemExit("prereg.json not in force (tracked, unmodified, committed in an ancestor of HEAD, "
                         "status not DRAFT); refusing to start")
    return g


def odir(args, name: str) -> Path:
    return OUT / (f"smoke_{name}" if args.smoke else name)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def run_data(args) -> int:
    t0 = time.time()
    devs = m.discover_types(m.DEV_TYPES)
    rows, keep4, keeprt, excluded = [], [], [], {}
    for d in devs:
        rt = e.load_rt(d)
        c4, crt = b.device_checks(d), b.device_checks(rt)
        bad = [f"4K {x}" for x in c4["bad"]] + [f"RT {x}" for x in crt["bad"]]
        if len(d.files) < 2 or len(rt.files) < 2:
            bad.append("missing file")
        rtr = m.restrict(rt, m.VMAX_4K[d.dev_type])
        sign = -1.0 if d.polarity == "pmos" else 1.0
        inc4 = [c.key for c in b.fit_curves(d) if b.inclusion_measured(c.kind, sign * c.id)["included"]]
        incrt = [c.key for c in b.fit_curves(rtr) if b.inclusion_measured(c.kind, sign * c.id)["included"]]
        if not bad and len(inc4) < MIN_INCLUDED_4K:
            bad.append(f"4K included {len(inc4)} < {MIN_INCLUDED_4K}")
        if not bad and len(incrt) < MIN_INCLUDED_RT:
            bad.append(f"RT included {len(incrt)} < {MIN_INCLUDED_RT}")
        row = {**b.device_row(d), "files_rt": rt.files, "bad": bad,
               "gate_control_4K": c4["gate_control_ratio"], "gate_control_RT": crt["gate_control_ratio"],
               "included_4K": inc4, "included_RT_restricted": incrt,
               "rt_curves_restricted": [f"{c.key}|{c.fixed:g}|{len(c.sweep)}" for c in b.fit_curves(rtr)],
               "group": group_nist(d)}
        if not bad:
            bc = m.bin_choice(d.dev_type, d.W_um, d.L_um)
            stock = m.read_params(d.dev_type, d.W_um, d.L_um, bc["bin"], b._base_set())
            row.update(bin=bc, stock=stock)
            # D10: the stock card of the device's bin must give finite currents at deck 30 C on
            # both grids (BSIM4 parameter checks of extrapolated binned parameters)
            fatal = {}
            for grid, dv in (("4K", d), ("RT", rtr)):
                sims, log = m.simulate(d.dev_type, d.W_um, d.L_um, bc["bin"], b.to_pipeline_curves(dv),
                                       None, m.DECK_RT_C, return_log=True)
                bad_keys = [c.meta["key"] for c, x in zip(b.to_pipeline_curves(dv), sims)
                            if not np.all(np.isfinite(x))]
                if bad_keys:
                    fatal[grid] = {"curves": bad_keys,
                                   "messages": sorted({ln.strip() for ln in log.splitlines()
                                                       if ln.strip().lower().startswith("fatal:")})}
            row["stock_card_nonfinite_30C"] = fatal
            if fatal:
                bad.append("stock card non-finite at deck 30 C (" + "; ".join(
                    sorted({x for v in fatal.values() for x in v["messages"]})) + ")")
        if bad:
            excluded[d.tag] = bad
        else:
            keep4.append(d)
            keeprt.append(rt)
        rows.append(row)
    s4 = b.save_dataset(keep4, NPZ4)
    srt = b.save_dataset(keeprt, NPZRT)
    dev_tags = sorted(d.tag for d in keep4)
    counts = {}
    for r in rows:
        c = counts.setdefault(r["dev_type"], {"devices": 0, "DEV": 0, "excluded": 0, "covered": 0})
        c["devices"] += 1
        c["DEV" if r["tag"] in dev_tags else "excluded"] += 1
        c["covered"] += int(bool(r.get("bin", {}).get("covered")))
    manifest = {
        "id": "I10_1_data", "doi_4K": b.DOI, "rt_repo": e.RT_REPO, "rt_commit": e.RT_COMMIT,
        "types": list(m.DEV_TYPES),
        "rule": "DEV = DEV-type ETest devices with both files at both temperatures, no bad check "
                "(i8_1_nist.device_checks: non-finite, all-zero, drain compliance, gate control < 100) "
                f"at 4 K and RT, >= {MIN_INCLUDED_4K} included 4 K Vb = 0 curves and >= {MIN_INCLUDED_RT} "
                "included RT Vb = 0 curves inside |V| <= 3.3 V (measured-only rule), and (D10) finite "
                "stock-card currents of the device's bin at deck 30 C on the 4 K and RT grids",
        "curves_4K": "Vb = 0: Id-Vg at |Vd| 0.1, 3.3 V (0-3.3 V, 10 mV, 331 points), Id-Vd at |Vg| 0, 0.55, "
                     "..., 3.3 V (0-3.3 V, 25 mV, 133 points)",
        "curves_RT": "Vb = 0 curves with |fixed| <= 3.3 V truncated to |sweep| <= 3.3 V "
                     "(i10_1_rt4k.restrict): Id-Vg at |Vd| 0.1 V, Id-Vd at |Vg| 0, 1, 2, 3 V (67 points, 50 mV); "
                     "npz holds the full Vb = 0 RT curves (raw sign)",
        "DEV": dev_tags, "n_dev": len(dev_tags), "excluded": excluded, "counts": counts,
        "smoke_devices": smoke_tags_nist(rows, dev_tags),
        "devices": rows, "index_4k": s4["index"], "index_rt": srt["index"],
        "index_columns": s4["index_columns"], "runtime_s": round(time.time() - t0, 1)}
    b.write_json(MANIFEST, manifest)
    print(json.dumps({k: manifest[k] for k in ("n_dev", "counts", "excluded", "smoke_devices")}, indent=1))
    return 0


def group_nist(d) -> str:
    return f"{d.dev_type}|W{d.W_um:g}|L{d.L_um:g}"


def smoke_tags_nist(rows, dev_tags) -> list[str]:
    """Per DEV type the shortest-L, median-L and longest-L DEV device (ties:
    smallest W, then tag)."""
    out = []
    for t in m.DEV_TYPES:
        ds = sorted([r for r in rows if r["dev_type"] == t and r["tag"] in dev_tags],
                    key=lambda r: (r["L_um"], r["W_um"], r["tag"]))
        if ds:
            out += [ds[0]["tag"], ds[len(ds) // 2]["tag"], ds[-1]["tag"]]
    return sorted(set(out))


# ---------------------------------------------------------------------------
# Dataset adapters (loaded once per process)
# ---------------------------------------------------------------------------
class NistDS:
    name = "nist"
    deck_unit = "C"

    def __init__(self, smoke: bool):
        man = json.loads(MANIFEST.read_text())
        tags = man["smoke_devices"] if smoke else man["DEV"]
        self.rows = {r["tag"]: r for r in man["devices"] if r["tag"] in set(tags)}
        saved = [r for r in man["devices"] if r["tag"] in set(man["DEV"])]    # npz order (data mode)
        d4 = b.load_dataset(NPZ4, {"devices": saved, "index": man["index_4k"]}, tags)
        drt = b.load_dataset(NPZRT, {"devices": saved, "index": man["index_rt"]}, tags)
        self.dev4 = {d.tag: d for d in d4}
        self.devrt = {d.tag: m.restrict(d, m.VMAX_4K[d.dev_type]) for d in drt}
        self.tags = sorted(tags)
        if sorted(self.dev4) != self.tags or sorted(self.devrt) != self.tags:
            raise RuntimeError("DEV devices missing from the dataset")
        self._box = {}

    def family(self, t):
        return self.dev4[t].dev_type

    def polarity(self, t):
        return self.dev4[t].polarity

    def group(self, t):
        return group_nist(self.dev4[t])

    def LW(self, t):
        return self.dev4[t].L_um, self.dev4[t].W_um

    def binfo(self, t):
        return self.rows[t]["bin"]

    def stock(self, t):
        return self.rows[t]["stock"]

    def box(self, t):
        if t not in self._box:
            self._box[t] = b.base_box(self.stock(t))
        return self._box[t]

    def deck_rt(self, t):
        return m.DECK_RT_C

    def deck_4k(self, t):
        return m.DECK_4K_C

    def problem(self, t, grid, deck):
        dev = self.devrt[t] if grid == "rt" else self.dev4[t]
        return m.MeasProblemI10(dev, self.binfo(t), self.stock(t), self.box(t), deck)

    def simulate(self, t, params, deck, grid="4k"):
        dev = self.devrt[t] if grid == "rt" else self.dev4[t]
        return m.simulate(dev.dev_type, dev.W_um, dev.L_um, int(self.binfo(t)["bin"]),
                          b.to_pipeline_curves(dev), b.changed_params(params, self.stock(t)), deck)

    def record(self, t, sims, params, grid="4k"):
        dev = self.devrt[t] if grid == "rt" else self.dev4[t]
        return m.to_record(dev, self.binfo(t), sims, params, extra_meta={"grid": grid})

    def rt_curves(self, t):
        return b.to_pipeline_curves(self.devrt[t])


class TudDS:
    name = "tud"
    deck_unit = "K"

    def __init__(self, smoke: bool):
        from cryoml.explore import i6_1_tudelft as m6
        self.m6 = m6
        split = json.loads((I6 / "split.json").read_text())
        man = json.loads((I6 / "data" / "manifest.json").read_text())
        devs = m6.load_dataset(I6 / "data" / "tudelft40.npz", man, split["DEV"])
        if sorted(d.tag for d in devs) != sorted(split["DEV"]):
            raise RuntimeError("TU Delft DEV devices missing")
        if smoke:
            first = {}
            for d in sorted(devs, key=lambda d: d.device):
                first.setdefault(d.group, d)
            devs = list(first.values())
        self.dev = {d.tag: d for d in devs}
        self.tags = sorted(self.dev)
        summ = json.loads((I6 / "start" / "summary.json").read_text())["per_temperature"]
        self.card_rt = {t: summ["295"]["rows"][t]["params"] for t in self.tags}
        self.card_4k_i6 = {t: summ["4"]["rows"][t]["params"] for t in self.tags}
        self._box = {}

    def family(self, t):
        return self.dev[t].dev_type

    polarity = family

    def group(self, t):
        return self.dev[t].group

    def LW(self, t):
        return self.dev[t].L_um, self.dev[t].W_um

    def stock(self, t):
        return self.box(t).published

    def box(self, t):
        if t not in self._box:
            self._box[t] = self.m6.base_box(self.dev[t])
        return self._box[t]

    def deck_rt(self, t):
        return float(self.dev[t].temps_K[295])

    def deck_4k(self, t):
        return float(self.dev[t].temps_K[4])

    def problem(self, t, grid, deck):
        tk = 295 if grid == "rt" else 4
        prob = self.m6.MeasProblem(self.dev[t], tk, self.box(t))
        prob.temp_stage_K = float(deck)
        return prob

    def simulate(self, t, params, deck, grid="4k"):
        d = self.dev[t]
        return self.m6.simulate(d, self.m6.model_block(d.dev_type), self.m6.fit_curves(d, 295 if grid == "rt" else 4),
                                float(deck), params)

    def record(self, t, sims, params, grid="4k"):
        return self.m6.to_record(self.dev[t], 295 if grid == "rt" else 4, sims, params)

    def rt_curves(self, t):
        return self.m6.fit_curves(self.dev[t], 295)


_DS: dict = {}


def get_ds(name: str, smoke: bool):
    key = (name, bool(smoke))
    if key not in _DS:
        _DS[key] = NistDS(smoke) if name == "nist" else TudDS(smoke)
    return _DS[key]


def _init_worker():
    import torch
    torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------
def fit_task(task):
    from cryoml.ngspice_fit import polish
    from cryoml.explore.e1_correction import score_device
    name, smoke, tag, label, grid, start, deck, opts = task
    ds = get_ds(name, smoke)
    t0 = time.time()
    n_to0 = m.N_TIMEOUT + ds_timeouts(ds)
    box = ds.box(tag)
    prob = ds.problem(tag, grid, deck)
    start_params = box.published if start is None else start
    res = polish(prob, box.params_to_local(start_params), max_nfev=opts["nfev"], irls_rounds=opts["irls"],
                 diff_step=opts["diff_step"])
    params = box.local_to_params(res["local"])
    rec = ds.record(tag, prob.simulate_params(params), params, grid)
    rec.meta.update(label=label, deck=deck, deck_unit=ds.deck_unit)
    sc = score_device(rec, [c.sim for c in rec.curves])
    loc = np.asarray(res["local"])
    row = {"tag": tag, "label": label, "grid": grid, "deck": deck, "start_from": "stock" if start is None else "card",
           "rrms_start_card": float(res["history"][0]) if res["history"] else None,
           "rrms": float(sc["rrms"]), "n_curves": sc["n_curves"], "n_sim": prob.n_sim,
           "n_active": int(box.n_active), "finite_record": b.finite_record(rec),
           "at_bound": [n for n, z, a in zip(box.names, loc, box.active) if a and abs(z) > 0.999],
           "n_sim_timeout": m.N_TIMEOUT + ds_timeouts(ds) - n_to0, "params": params,
           "runtime_s": round(time.time() - t0, 1)}
    print(f"  {label:8s} {tag:44s} rrms {row['rrms']:.4f} n_sim {prob.n_sim} {row['runtime_s']} s", flush=True)
    return rec, row


def ds_timeouts(ds) -> int:
    return ds.m6.N_TIMEOUT if ds.name == "tud" else 0


def save_fits(target: Path, label: str, pairs) -> dict:
    from cryoml.explore.e1_correction import save_devices
    pairs = sorted(pairs, key=lambda x: x[0].tag)
    save_devices(target / f"records_{label}", [r for r, _ in pairs])
    rows = {row["tag"]: row for _, row in pairs}
    return {"rows": rows, "non_finite_records": sorted(t for t, r in rows.items() if not r["finite_record"]),
            "n_sim_timeout": int(sum(r["n_sim_timeout"] for r in rows.values())),
            "rrms_mean": float(np.nanmean([r["rrms"] for r in rows.values()])) if rows else float("nan"),
            "records_sha256": {s: b.sha256((target / f"records_{label}").with_suffix(s)) for s in (".json", ".npz")}}


def run_start(args) -> int:
    guard = require_prereg(args)
    t_start = time.time()
    ds = get_ds(args.dataset, args.smoke)
    opts = dict(START_OPTIONS)
    if args.smoke:
        opts.update(nfev=SMOKE["nfev"], irls=SMOKE["irls"])
    target = odir(args, f"start_{ds.name}")
    target.mkdir(parents=True, exist_ok=True)
    out = {"dataset": ds.name, "options": opts, "smoke": bool(args.smoke), "guard": guard,
           "deck_unit": ds.deck_unit, "fits": {}}
    with get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool:
        if ds.name == "nist":
            t1 = [(ds.name, args.smoke, t, "C0_RT", "rt", None, ds.deck_rt(t), opts) for t in ds.tags]
            t1 += [(ds.name, args.smoke, t, "C4_ind", "4k", None, ds.deck_rt(t), opts) for t in ds.tags]
            res1 = pool.map(fit_task, t1, chunksize=1)
            rt = [p for p in res1 if p[1]["label"] == "C0_RT"]
            out["fits"]["C0_RT"] = save_fits(target, "C0_RT", rt)
            out["fits"]["C4_ind"] = save_fits(target, "C4_ind", [p for p in res1 if p[1]["label"] == "C4_ind"])
            card = {r.tag: row["params"] for r, row in rt}
            t2 = [(ds.name, args.smoke, t, "C4", "4k", card[t], ds.deck_rt(t), opts) for t in ds.tags]
            t2 += [(ds.name, args.smoke, t, "C4_deck", "4k", card[t], ds.deck_4k(t), opts) for t in ds.tags]
            res2 = pool.map(fit_task, t2, chunksize=1)
            for lab in ("C4", "C4_deck"):
                out["fits"][lab] = save_fits(target, lab, [p for p in res2 if p[1]["label"] == lab])
        else:
            t1 = [(ds.name, args.smoke, t, "C4", "4k", ds.card_rt[t], ds.deck_rt(t), opts) for t in ds.tags]
            out["fits"]["C4"] = save_fits(target, "C4", pool.map(fit_task, t1, chunksize=1))
            out["stored_card_check"] = stored_card_check(ds)
    out["runtime_s"] = round(time.time() - t_start, 1)
    b.write_json(target / "summary.json", out)
    print(json.dumps({k: {kk: v[kk] for kk in ("non_finite_records", "n_sim_timeout", "rrms_mean")}
                      for k, v in out["fits"].items()}, indent=1))
    print(f"start {ds.name}: runtime {out['runtime_s']} s")
    return 0


def stored_card_check(ds) -> dict:
    """TU Delft: the stored I6-1 295 K and 4 K cards re-simulated at their
    stage temperatures reproduce the stored record currents."""
    from cryoml.explore.e1_correction import load_devices
    out = {}
    for tk, cards in ((295, ds.card_rt), (4, ds.card_4k_i6)):
        recs = {r.tag: r for r in load_devices(I6 / "start" / f"records_{tk}K")}
        worst = 0.0
        for t in ds.tags:
            sims = ds.simulate(t, cards[t], ds.dev[t].temps_K[tk], grid="rt" if tk == 295 else "4k")
            for c, s in zip(recs[t].curves, sims):
                a, z = np.asarray(c.sim, float), np.asarray(s, float)
                ok = np.isfinite(a) & np.isfinite(z)
                if ok.any():
                    worst = max(worst, float(np.max(np.abs(a[ok] - z[ok]) / np.maximum(np.abs(a[ok]), 1e-15))))
                if (~np.isfinite(a)).sum() != (~np.isfinite(z)).sum():
                    worst = float("inf")
        out[f"{tk}K_max_rel_diff"] = worst
    return out


# ---------------------------------------------------------------------------
# lodo
# ---------------------------------------------------------------------------
def sim_task(task):
    name, smoke, tag, label, params, deck = task
    ds = get_ds(name, smoke)
    n0 = m.N_TIMEOUT + ds_timeouts(ds)
    t0 = time.time()
    sims = ds.simulate(tag, params, deck)
    return label, tag, [np.asarray(s, float) for s in sims], round(time.time() - t0, 3), \
        m.N_TIMEOUT + ds_timeouts(ds) - n0


_NESTED: dict = {}


def _init_nested(path: str):
    from cryoml.explore.e1_correction import load_devices
    _init_worker()
    _NESTED.clear()
    for r in load_devices(Path(path)):
        _NESTED[r.meta["nkey"]] = r


def bias_task(task):
    from cryoml.explore.i3_2_bias import fit_bias
    fold, keys, hyper = task
    return fit_bias("M_bias", [_NESTED[k] for k in keys], hyper=hyper, threads=1)


def load_fit_records(path: Path) -> dict:
    from cryoml.explore.e1_correction import load_devices
    return {r.tag: r for r in load_devices(path)}


def features(ds, tags, z_rt: dict, kind: str) -> np.ndarray:
    rows = []
    for t in tags:
        L, W = ds.LW(t)
        x = [math.log10(L), math.log10(W)]
        if kind == "full":
            d = m.descriptors(ds.rt_curves(t), W, L)
            x += [d[k] for k in m.DESCRIPTORS] + list(z_rt[t])
        rows.append(x)
    return np.asarray(rows, float)


def run_lodo(args) -> int:
    from cryoml.explore.e1_correction import Correction, save_devices, score_device
    from cryoml.explore.i3_2_bias import merge_seed_results, split_seed_tasks
    guard = require_prereg(args)
    t_start = time.time()
    ds = get_ds(args.dataset, args.smoke)
    sdir = odir(args, f"start_{ds.name}")
    summ = json.loads((sdir / "summary.json").read_text())
    tags = ds.tags
    # cards and A1-rule finiteness
    rec4 = load_fit_records(sdir / "records_C4")
    if ds.name == "nist":
        recrt = load_fit_records(sdir / "records_C0_RT")
        card_rt = {t: recrt[t].meta["params"] for t in tags}
        finite_rt = {t: b.finite_record(recrt[t]) for t in tags}
        var_recs = {"V_ind": load_fit_records(sdir / "records_C4_ind"),
                    "V_deck": load_fit_records(sdir / "records_C4_deck")}
    else:
        from cryoml.explore.e1_correction import load_devices
        stored = {r.tag: r for r in load_devices(I6 / "start" / "records_295K")}
        card_rt = dict(ds.card_rt)
        finite_rt = {t: b.finite_record(stored[t]) for t in tags}
        stored4 = {r.tag: r for r in load_devices(I6 / "start" / "records_4K")}
        var_recs = {"V_i6": stored4}
    ok = sorted(t for t in tags if finite_rt[t] and b.finite_record(rec4[t]))
    excluded = sorted(set(tags) - set(ok))
    print(f"A1 rule: {len(excluded)} devices with a non-finite start record excluded and scored as failed: "
          f"{excluded}", flush=True)
    z_rt = {t: m.local(ds.box(t), card_rt[t]) for t in tags}
    z4 = {t: m.local(ds.box(t), rec4[t].meta["params"]) for t in tags}
    dz = {t: z4[t] - z_rt[t] for t in tags}
    dz_var, ok_var = {}, {}
    for v, recs in var_recs.items():
        ok_var[v] = [t for t in ok if b.finite_record(recs[t])]
        dz_var[v] = {t: m.local(ds.box(t), recs[t].meta["params"]) - z_rt[t] for t in tags}
    Xg = dict(zip(tags, features(ds, tags, z_rt, "geom")))
    Xf = dict(zip(tags, features(ds, tags, z_rt, "full")))
    groups = sorted({ds.group(t) for t in tags})
    names = list(ds.box(tags[0]).names)
    # ---- arm cards
    sims_tasks, arm_params, folds = [], {}, {}
    nested_keys: dict = {}

    def add(label, t, params, deck):
        arm_params.setdefault(label, {})[t] = {"params": params, "deck": deck}
        sims_tasks.append((ds.name, args.smoke, t, label, params, deck))

    for G in groups:
        held = [t for t in tags if ds.group(t) == G]
        fam = ds.family(held[0])
        T = [t for t in ok if ds.family(t) == fam and ds.group(t) != G]
        fold = {"group": G, "family": fam, "held": held, "train": T}
        if not T:
            fold["note"] = "no training device"
            folds[G] = fold
            continue
        D = np.array([dz[t] for t in T])
        med, mean = m.shift_stat(D, "median"), m.shift_stat(D, "mean")
        rL = m.RidgeShift().fit(np.array([Xg[t] for t in T]), D)
        rF = m.RidgeShift().fit(np.array([Xf[t] for t in T]), D)
        zpop = np.median(np.array([z4[t] for t in T]), axis=0)
        fold.update(median_dz=dict(zip(names, med)), ridge_geom=rL.summary(names), ridge_full=rF.summary(names))
        vstat = {}
        for v in var_recs:
            Tv = [t for t in ok_var[v] if ds.family(t) == fam and ds.group(t) != G]
            vstat[v] = m.shift_stat(np.array([dz_var[v][t] for t in Tv]), "median") if Tv else None
            fold[f"n_train_{v}"] = len(Tv)
        folds[G] = fold
        for h in held:
            if h not in ok:
                continue
            box, zr = ds.box(h), z_rt[h]
            add("A0", h, card_rt[h], ds.deck_4k(h))
            add("A0_rtdeck", h, card_rt[h], ds.deck_rt(h))
            add("A1", h, m.apply_shift(box, zr, med), ds.deck_rt(h))
            add("A1_L", h, m.apply_shift(box, zr, rL.predict(Xg[h])), ds.deck_rt(h))
            add("A3", h, m.apply_shift(box, zr, rF.predict(Xf[h])), ds.deck_rt(h))
            add("V_mean", h, m.apply_shift(box, zr, mean), ds.deck_rt(h))
            add("V_sub7", h, m.apply_shift(box, zr, med, subset=m.PARAMS7), ds.deck_rt(h))
            for v, st in vstat.items():
                if st is None:
                    continue
                deck = ds.deck_4k(h) if v in ("V_deck", "V_i6") else ds.deck_rt(h)
                add(v, h, m.apply_shift(box, zr, st), deck)
            add("R_pop", h, box.local_to_params(np.clip(zpop, -1, 1)), ds.deck_rt(h))
            add("R_stock4k", h, dict(box.published), ds.deck_4k(h))
            add("R_stock_rtdeck", h, dict(box.published), ds.deck_rt(h))
        # nested A1 training records for A2 (M_bias): every finite training device of
        # either family outside G, with its own out-of-group median shift
        keys = []
        for j in ok:
            if ds.group(j) == G:
                continue
            Tj = [t for t in ok if ds.family(t) == ds.family(j) and ds.group(t) not in (G, ds.group(j))]
            rule = "minus G and group(j)"
            if not Tj:
                Tj = [t for t in ok if ds.family(t) == ds.family(j) and ds.group(t) != G and t != j]
                rule = "minus G and j (group(j) is the only other group)"
            if not Tj:
                continue
            k = f"{j}|{'+'.join(sorted(Tj))}"
            if k not in nested_keys:
                st = m.shift_stat(np.array([dz[t] for t in Tj]), "median")
                nested_keys[k] = {"j": j, "params": m.apply_shift(ds.box(j), z_rt[j], st), "rule": rule}
            keys.append(k)
        fold["nested_keys"] = keys
    nk = sorted(nested_keys)
    sims_tasks += [(ds.name, args.smoke, nested_keys[k]["j"], f"nested#{i}", nested_keys[k]["params"],
                    ds.deck_rt(nested_keys[k]["j"])) for i, k in enumerate(nk)]
    print(f"{len(groups)} folds; {len(sims_tasks)} simulations ({len(nk)} nested)", flush=True)
    timing = {}
    with get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool:
        t0 = time.time()
        res = pool.map(sim_task, sims_tasks, chunksize=4)
        timing["sim_s"], timing["sim_n"] = round(time.time() - t0, 1), len(sims_tasks)
    sims = {}
    n_timeout = 0
    for label, t, s, _, nto in res:
        sims[(label, t)] = s
        n_timeout += nto
    # nested records -> M_bias training per fold
    nested_recs = []
    for i, k in enumerate(nk):
        j = nested_keys[k]["j"]
        r = ds.record(j, sims[(f"nested#{i}", j)], nested_keys[k]["params"])
        r.meta["nkey"] = k
        nested_recs.append(r)
    good_nested = {r.meta["nkey"] for r in nested_recs if b.finite_record(r)}
    WORK.mkdir(parents=True, exist_ok=True)
    npath = WORK / f"{'smoke_' if args.smoke else ''}nested_{ds.name}"
    save_devices(npath, nested_recs)
    hyper = dict(SMOKE["bias_hyper"]) if args.smoke else {}
    bias_folds = [(G, [k for k in folds[G].get("nested_keys", []) if k in good_nested]) for G in groups
                  if folds[G].get("train") and any(h in ok for h in folds[G]["held"])]
    for G, keys in bias_folds:
        folds[G]["n_nested_used"] = len(keys)
        folds[G]["n_nested_nonfinite"] = len(folds[G]["nested_keys"]) - len(keys)
    expanded, owner = split_seed_tasks([(G, keys) for G, keys in bias_folds], hyper)
    t0 = time.time()
    with get_context("spawn").Pool(args.workers, initializer=_init_nested, initargs=(str(npath),)) as pool:
        trained = pool.map(bias_task, [(G, keys, hp) for G, keys, hp in expanded], chunksize=1)
    timing["bias_s"], timing["bias_n"] = round(time.time() - t0, 1), len(expanded)
    corrs = dict(zip([G for G, _ in bias_folds], merge_seed_results(len(bias_folds), owner, trained)))
    # ---- scores
    scores = {a: {} for a in list(arm_params) + ["A2", "R_oracle"]}
    for a in ("A0", "A0_rtdeck", "A1", "A1_L", "A2", "A3", "R_pop", "R_stock4k", "R_stock_rtdeck",
              "R_oracle", *VARIANTS[ds.name]):
        scores.setdefault(a, {})
    for a, per in arm_params.items():
        for t, v in per.items():
            rec = ds.record(t, sims[(a, t)], v["params"])
            scores[a][t] = float(score_device(rec, [c.sim for c in rec.curves])["rrms"])
            if a == "A1":
                G = ds.group(t)
                if G in corrs:
                    scores["A2"][t] = float(score_device(rec, corrs[G].apply(rec))["rrms"])
    for t in ok:
        scores["R_oracle"][t] = float(score_device(rec4[t], [c.sim for c in rec4[t].curves])["rrms"])
    for a in scores:
        for t in tags:
            if t not in scores[a]:
                scores[a][t] = float("nan")
    clusters = {t: ds.group(t) for t in tags}
    fam = {t: ds.polarity(t) for t in tags}
    comps = comparisons(scores, tags, clusters, fam)
    result = {"dataset": ds.name, "smoke": bool(args.smoke), "guard": guard, "deck_unit": ds.deck_unit,
              "prereg": "out/research/explore/I10_1/prereg.json", "start_summary_runtime_s": summ.get("runtime_s"),
              "devices": tags, "excluded_A1_rule": excluded, "groups": groups,
              "bias_hyper_override": hyper, "bonferroni_level": BONF_LEVEL,
              "summary": {a: b.aggregate(v, fam) for a, v in scores.items()},
              "comparisons": comps, "per_device": scores, "n_sim_timeout": n_timeout,
              "folds": {G: {k: v for k, v in f.items() if k != "nested_keys"} | {"n_nested": len(f.get("nested_keys", []))}
                        for G, f in folds.items()},
              "timing": timing, "runtime_s": round(time.time() - t_start, 1)}
    target = odir(args, f"lodo_{ds.name}")
    b.write_json(target / "results.json", result)
    b.write_json(target / "cards.json", {a: {t: v["params"] for t, v in per.items()} for a, per in arm_params.items()})
    with gzip.open(target / "corrections.json.gz", "wt") as f:
        json.dump({G: c.to_json() for G, c in corrs.items()}, f)
    print(json.dumps({a: round(s["valid_only_mean"], 4) if s["valid_only_mean"] == s["valid_only_mean"] else None
                      for a, s in result["summary"].items()}))
    for k, c in comps["all"].items():
        print(f"  {k:24s} {c['bonferroni']['ratio']:.3f} [{c['bonferroni']['ci_low']:.3f}, "
              f"{c['bonferroni']['ci_high']:.3f}] n {c['bonferroni']['n_used']}")
    print(f"lodo {ds.name}: timing {timing}, runtime {result['runtime_s']} s, timeouts {n_timeout}")
    return 0


def contrast_pairs(dataset: str) -> list[tuple[str, str, str]]:
    out = [(k, a, c) for k, a, c in CONTRASTS]
    out += [(f"S_{a}_vs_{c}", a, c) for a, c in SECONDARY]
    out += [(f"V_{v}", v, "A1") for v in VARIANTS[dataset]]
    return out


def comparisons(scores: dict, tags: list, clusters: dict, fam: dict) -> dict:
    dataset = "nist" if any(t.startswith("etest_") for t in tags) else "tud"
    out = {}
    for sub, sel in (("all", tags), ("nmos", [t for t in tags if fam[t] == "nmos"]),
                     ("pmos", [t for t in tags if fam[t] == "pmos"])):
        out[sub] = {}
        for key, a, c in contrast_pairs(dataset):
            if a not in scores or c not in scores:
                continue
            ma = {t: scores[a][t] for t in sel}
            mc = {t: scores[c][t] for t in sel}
            out[sub][key] = {"method": a, "control": c,
                             "bonferroni": m.gmr_ci(ma, mc, level=BONF_LEVEL),
                             "level95": m.gmr_ci(ma, mc, level=0.95),
                             "cluster_bonferroni": m.gmr_ci(ma, mc, clusters=clusters, level=BONF_LEVEL)}
    return out


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------
def outcome_of(labels: dict) -> tuple[str, str, list]:
    """(outcome, action, TEST confirmatory contrasts) from the K1-K3 labels."""
    if labels["K1"] != "lower":
        return ("not supported", "stop: no I10-1 TEST run; the TEST devices stay locked", [])
    conf = ["K1"] + [k for k in ("K2", "K3") if labels[k] == "lower"]
    if len(conf) == 3:
        return ("supported", "continue: run the pre-registered TEST protocol in December", conf)
    return ("partly supported",
            "continue, reduced: run the TEST protocol in December with the contrasts labelled 'lower' on DEV "
            "as the confirmatory set; the others are reported only", conf)


def test_arm(labels: dict, comps: dict) -> str:
    cands = [(comps[k]["bonferroni"]["ratio"], a) for k, a in (("K4", "A2"), ("K5", "A3"))
             if labels[k] == "lower"]
    return min(cands)[1] if cands else "A1"


def variant_choice(comps: dict, variants) -> dict:
    rows = {v: comps[f"V_{v}"]["level95"] for v in variants if f"V_{v}" in comps}
    better = [(r["ratio"], v) for v, r in rows.items() if r.get("ci_high", float("nan")) < 1]
    return {"adopted": min(better)[1] if better else None,
            "rule": "adopt the variant with the lowest GMR(V, A1) among those whose 95% device-bootstrap CI "
                    "upper bound is < 1; none: the primary configuration",
            "gmr": {v: {k: r.get(k) for k in ("ratio", "ci_low", "ci_high", "n_used")} for v, r in rows.items()}}


def validity(res: dict, start: dict) -> dict:
    tags = res["devices"]
    n = len(tags)
    rows4 = start["fits"]["C4"]["rows"]
    ok_start = [t for t in tags if t not in res["excluded_A1_rule"]
                and (rows4[t]["n_curves"] or 0) >= MIN_INCLUDED_4K]
    v1 = len(ok_start) >= V1_MIN_FINITE * n
    failed = {a: sum(1 for t in ok_start if not np.isfinite(res["per_device"][a][t] if res["per_device"][a][t]
                                                             is not None else np.nan))
              for a in PRIMARY_ARMS + ("A0",)}
    paired = {k: res["comparisons"]["all"][k]["bonferroni"]["n_used"] for k, _, _ in CONTRASTS}
    v2 = (all(failed[a] <= V2_MAX_FAILED * n for a in PRIMARY_ARMS)
          and all(p >= math.ceil(V2_MIN_PAIRED * n) for p in paired.values()))
    return {"V1": v1, "n_finite_start": len(ok_start), "n": n, "V2": v2, "failed": failed, "paired": paired,
            "V4_timeouts_lodo": res["n_sim_timeout"],
            "V4_timeouts_start": {k: v["n_sim_timeout"] for k, v in start["fits"].items()}}


def run_decide(args) -> int:
    guard = require_prereg(args)
    out = {"id": "I10_1_decision", "smoke": bool(args.smoke), "guard": guard,
           "prereg": "out/research/explore/I10_1/prereg.json", "bonferroni_level": BONF_LEVEL, "datasets": {}}
    for name in ("nist", "tud"):
        rp = odir(args, f"lodo_{name}") / "results.json"
        if not rp.exists():
            out["datasets"][name] = "missing"
            continue
        res = json.loads(rp.read_text())
        start = json.loads((odir(args, f"start_{name}") / "summary.json").read_text())
        val = validity(res, start)
        comps = res["comparisons"]["all"]
        labels = {k: m.ci_label(comps[k]["bonferroni"]) for k, _, _ in CONTRASTS}
        lg = {f"gap_fraction_{a}": None for a in ("A1", "A2", "A3")}
        # ln GMR(arm, A0) / ln GMR(R_oracle, A0); GMR(R_oracle, A0) = GMR(A1, A0) / GMR(A1, R_oracle)
        # (descriptive; the two GMRs can rest on different paired device sets)
        try:
            ln_a1_a0 = math.log(comps["K1"]["bonferroni"]["ratio"])
            ln_or_a0 = ln_a1_a0 - math.log(comps["S_A1_vs_R_oracle"]["bonferroni"]["ratio"])
            for a in ("A1", "A2", "A3"):
                key = "K1" if a == "A1" else f"S_{a}_vs_A0"
                lg[f"gap_fraction_{a}"] = math.log(comps[key]["bonferroni"]["ratio"]) / ln_or_a0
        except (KeyError, ValueError, ZeroDivisionError, TypeError):
            pass
        row = {"validity": val, "valid": val["V1"] and val["V2"], "labels": labels,
               "contrasts": {k: comps[k] for k, _, _ in CONTRASTS},
               "per_family": {f: {k: res["comparisons"][f][k]["bonferroni"] for k, _, _ in CONTRASTS
                                  if k in res["comparisons"][f]} for f in ("nmos", "pmos")},
               "gap_fraction_toward_oracle": lg, "variant_choice": variant_choice(comps, VARIANTS[name]),
               "means_valid_only": {a: s["valid_only_mean"] for a, s in res["summary"].items()},
               "lodo_guard": res.get("guard")}
        out["datasets"][name] = row
    nist = out["datasets"].get("nist")
    if isinstance(nist, dict):
        if not nist["valid"]:
            out["outcome"], out["action"] = "invalid", "stop: validity check failed; fix and re-register"
            out["test_confirmatory_contrasts"] = []
        else:
            out["outcome"], out["action"], out["test_confirmatory_contrasts"] = outcome_of(nist["labels"])
        out["test_primary_arm"] = test_arm(nist["labels"], nist["contrasts"]) if out["outcome"] != "invalid" else None
        out["test_configuration_variant"] = nist["variant_choice"]["adopted"]
        tud = out["datasets"].get("tud")
        if isinstance(tud, dict):
            out["tud_direction_agrees"] = {
                k: bool(np.sign(math.log(tud["contrasts"][k]["bonferroni"]["ratio"]))
                        == np.sign(math.log(nist["contrasts"][k]["bonferroni"]["ratio"])))
                if all(np.isfinite([tud["contrasts"][k]["bonferroni"]["ratio"] or np.nan,
                                    nist["contrasts"][k]["bonferroni"]["ratio"] or np.nan])) else None
                for k, _, _ in CONTRASTS}
    target = odir(args, "decision") / "decision.json" if args.smoke else OUT / "decision.json"
    b.write_json(target, out)
    print(json.dumps({k: out.get(k) for k in ("outcome", "action", "test_primary_arm",
                                              "test_configuration_variant", "tud_direction_agrees")}, indent=1))
    for name, row in out["datasets"].items():
        if isinstance(row, dict):
            print(name, json.dumps({"valid": row["valid"], "labels": row["labels"], "validity": row["validity"]}))
    return 0


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------
def _rel(a, c) -> float:
    return max(float(np.max(np.abs(x - y) / np.maximum(np.abs(x), 1e-15))) for x, y in zip(a, c))


def scan_task(task):
    name, tag = task
    ds = get_ds(name, False)
    out = {}
    for lab, deck in (("deck_30C", m.DECK_RT_C), ("deck_4K", m.DECK_4K_C)):
        t0 = time.time()
        sims = ds.simulate(tag, dict(ds.stock(tag)), deck)
        out[lab] = {"n_nonfinite_curves": int(sum(not np.all(np.isfinite(s)) for s in sims)),
                    "seconds": round(time.time() - t0, 3)}
    return tag, out


def core_card_check() -> dict:
    """Core (TEST) types: single-bin card vs the unmodified open_pdks library on
    a synthetic 7-point grid at a bin-covered geometry.  No measured data."""
    from cryoml.data_io import Curve
    out = {}
    for t in m.CORE_TYPES:
        W, L = 1.0, 0.5
        pol = m.TYPE_INFO[t].polarity
        s = -1.0 if pol == "pmos" else 1.0
        v = s * np.linspace(0, 1.8, 7)
        curves = [Curve("idvg", s * 1.8, v, np.full(7, s * 1.8), np.zeros(7), meta={"key": "idvg@d0"}),
                  Curve("idvd", s * 1.8, np.full(7, s * 1.8), v, np.zeros(7), meta={"key": "idvd@g0"})]
        bc = m.bin_choice(t, W, L)
        t0 = time.time()
        forced = m.simulate(t, W, L, bc["bin"], curves)
        full = m.simulate(t, W, L, bc["bin"], curves, all_bins=True)
        off = m.simulate_official(t, W, L, curves)
        out[t] = {"W_um": W, "L_um": L, "bin": bc["bin"], "covered": bc["covered"],
                  "n_bins": len(m.card_parts(t)["blocks"]), "lines_not_parsed": m.card_parts(t)["n_unparsed"],
                  "max_rel_forced_vs_full": _rel(full, forced), "max_rel_full_vs_official": _rel(off, full),
                  "finite": bool(all(np.all(np.isfinite(x)) for x in forced)),
                  "seconds": round(time.time() - t0, 2)}
        print(t, json.dumps(out[t]), flush=True)
    return out


def run_smoke(args) -> int:
    t_start = time.time()
    ds = get_ds("nist", False)
    out = {"id": "I10_1_smoke", "note": "not used for any decision", "simulator": b._sp()._NGSPICE_BIN}
    # card checks on the smoke devices that lie in a bin
    smoke = get_ds("nist", True)
    checks = []
    for t in smoke.tags:
        if not smoke.binfo(t)["covered"]:
            continue
        d = smoke.dev4[t]
        curves = b.to_pipeline_curves(d)
        forced = m.simulate(d.dev_type, d.W_um, d.L_um, smoke.binfo(t)["bin"], curves)
        full = m.simulate(d.dev_type, d.W_um, d.L_um, smoke.binfo(t)["bin"], curves, all_bins=True)
        off = m.simulate_official(d.dev_type, d.W_um, d.L_um, curves)
        checks.append({"device": t, "bin": smoke.binfo(t)["bin"], "max_rel_forced_vs_full": _rel(full, forced),
                       "max_rel_full_vs_official": _rel(off, full),
                       "max_abs_full_vs_official_A": max(float(np.max(np.abs(x - y))) for x, y in zip(off, full))})
        print(json.dumps(checks[-1]), flush=True)
    out["card_checks_dev"] = checks
    out["core_card_checks_synthetic"] = core_card_check()
    with get_context("spawn").Pool(args.workers) as pool:
        scan = dict(pool.map(scan_task, [("nist", t) for t in ds.tags], chunksize=2))
    out["stock_scan"] = {
        "n_devices": len(scan),
        "nonfinite": {lab: sorted(t for t, r in scan.items() if r[lab]["n_nonfinite_curves"])
                      for lab in ("deck_30C", "deck_4K")},
        "seconds_median": {lab: float(np.median([r[lab]["seconds"] for r in scan.values()]))
                           for lab in ("deck_30C", "deck_4K")},
        "seconds_max": {lab: float(max(r[lab]["seconds"] for r in scan.values())) for lab in ("deck_30C", "deck_4K")},
        "devices": scan}
    print(json.dumps({k: v for k, v in out["stock_scan"].items() if k != "devices"}), flush=True)
    out["runtime_s"] = round(time.time() - t_start, 1)
    b.write_json(OUT / "smoke" / "smoke.json", out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("data", "smoke", "start", "lodo", "decide"))
    ap.add_argument("--dataset", choices=("nist", "tud"), default="nist")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    return {"data": run_data, "smoke": run_smoke, "start": run_start, "lodo": run_lodo,
            "decide": run_decide}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
