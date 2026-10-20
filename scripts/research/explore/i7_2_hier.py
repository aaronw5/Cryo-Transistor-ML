#!/usr/bin/env python3
"""I7-2: hierarchical bias-only correction, few-shot adaptation on K curves.

Pre-registration: ``out/research/explore/I7_2/prereg.json``.
Module: ``cryoml.explore.i7_2_hier``.

Datasets (``--dataset``):
  sky130   sky130 77 K REAL_DEV (12), leave-one-device-out, g_shared = stored
           I3-2 fold M_bias, K-shot box = round-1 43-parameter box
  ihp300   IHP SG13G2 300 K DEV (18), leave-one-DUT-out within class,
           g_shared = stored I5-1 fold M_bias, box = I5-1 base box
  tud4     TU Delft 40 nm 4.13 K DEV (45), leave-one-geometry-out (7 groups),
           g_shared = stored I6-1 fold M_bias, box = I6-1 base box
  tud295   as tud4 at 295 K

Modes:
  run      per fold: training-device deltas, prior, mu selection (K = 2, 4);
           per (held-out device, draw): start choice, base fit on the K
           adaptation curves, arms a_C0, b_shared, c_hier, c_ml, d_scratch,
           e_crossinterp; scores on the held-out curves -> <out>/<dataset>/
  decide   pre-registered rule over the four datasets -> decision.json

Only DEV devices are read (REAL_TEST / TEST are not loaded).  ``--smoke``
writes ``smoke_<dataset>`` with the reduced budgets of ``SMOKE``.

  .venv/bin/python -I scripts/research/explore/i7_2_hier.py --mode run --dataset sky130 --workers 8
  .venv/bin/python -I scripts/research/explore/i7_2_hier.py --mode decide
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cryoml.explore import i7_2_hier as h  # noqa: E402

OUT = ROOT / "out/research/explore/I7_2"
PREREG = OUT / "prereg.json"
DATASETS = ("sky130", "ihp300", "tud4", "tud295")
OPTIONS = {"base_nfev": 1200, "base_irls": 2, "arm_nfev": 400, "arm_irls": 2, "diff_step": 0.01}
SMOKE = {"base_nfev": 100, "arm_nfev": 60, "draws": ("K2a",), "d_hyper": {"seeds": (0,), "steps": 200}}
PAIRS = (("c_vs_b", "c_hier", "b_shared"), ("c_vs_a", "c_hier", "a_C0"),
         ("c_vs_d", "c_hier", "d_scratch"), ("c_vs_e", "c_hier", "e_crossinterp"),
         ("c_vs_cml", "c_hier", "c_ml"), ("b_vs_a", "b_shared", "a_C0"),
         ("d_vs_a", "d_scratch", "a_C0"), ("e_vs_a", "e_crossinterp", "a_C0"),
         ("cml_vs_a", "c_ml", "a_C0"))


# ---------------------------------------------------------------------------
# Dataset adapters (built lazily in each worker process)
# ---------------------------------------------------------------------------
class Sky130:
    name = "sky130"
    VDD = 1.85

    def __init__(self):
        from cryoml.explore.e1_correction import load_devices
        from cryoml.explore.i3_2_bias import correction_from_json
        self.spec = json.loads((ROOT / "out/cold_loop/full/round_1/spec.json").read_text())
        dev_tags = json.loads((ROOT / "out/research/explore/real_split.json").read_text())["REAL_DEV"]
        self.records = load_devices(ROOT / "out/research/explore/E1/real_dev")
        if sorted(r.tag for r in self.records) != sorted(dev_tags):
            raise RuntimeError("E1 real_dev records differ from REAL_DEV")
        cards = json.loads((ROOT / "out/research/polished/direct43_r3_fd/real/summary.json")
                           .read_text())["devices"]
        for r in self.records:
            r.meta["params"] = dict(cards[r.tag]["params"])
        self.by_tag = {r.tag: r for r in self.records}
        corr = json.loads((ROOT / "out/research/explore/I3_2/real_dev/corrections.json").read_text())
        self._g = {t: correction_from_json(corr[f"M_bias|{t}"]) for t in dev_tags}

    def folds(self):
        return sorted(self.by_tag)

    def fold_of(self, tag):
        return tag

    def held_of(self, fold):
        return [fold]

    def train_of(self, fold):
        return [r for r in self.records if r.tag != fold]

    def g_shared(self, fold):
        return self._g[fold]

    def box(self, tag):
        from cold_loop import box_for
        return box_for(self.spec, tag)

    def problem(self, tag):
        from cryoml.ngspice_fit import DeviceProblem
        return DeviceProblem(tag, self.box(tag))

    def corrected(self, prob, corr):
        from cryoml.explore.i2_1_joint import CorrectedProblem
        r = self.by_tag[prob.tag]
        return CorrectedProblem(prob, corr, r.family, r.L_um, r.W_um, self.VDD, "sky130")

    def full_record(self, tag, params):
        from e1_collect_real import collect
        rec = collect(tag, params)
        rec.meta["params"] = dict(params)
        return rec

    def train_local(self, rec):
        box = self.box(rec.tag)
        return box.params_to_local({**box.published,
                                    **{k: v for k, v in rec.meta["params"].items() if k in box.names}})

    def same_kind(self, a, b):
        return a.family == b.family


class Meas:
    """IHP (I5-1) and TU Delft (I6-1) measured datasets."""

    def __init__(self, name):
        from cryoml.explore.e1_correction import load_devices
        from cryoml.explore.i3_2_bias import correction_from_json
        self.name = name
        if name == "ihp300":
            from cryoml.explore import i5_1_ihp_real as m
            base, self.T = ROOT / "out/research/explore/I5_1", 300
            npz = base / "data/ihp_meas.npz"
        else:
            from cryoml.explore import i6_1_tudelft as m
            base, self.T = ROOT / "out/research/explore/I6_1", int(name[3:])
            npz = base / "data/tudelft40.npz"
        self.m = m
        split = json.loads((base / "split.json").read_text())
        manifest = json.loads((base / "data/manifest.json").read_text())
        devices = m.load_dataset(npz, manifest, split["DEV"])
        if sorted(d.tag for d in devices) != sorted(split["DEV"]):
            raise RuntimeError(f"{name}: DEV devices missing")
        self.devices = {d.tag: d for d in devices}
        self.records = load_devices(base / f"start/records_{self.T}K")
        if sorted(r.tag for r in self.records) != sorted(self.devices):
            raise RuntimeError(f"{name}: starting records differ from DEV")
        self.by_tag = {r.tag: r for r in self.records}
        corr = json.loads((base / f"lodo_{self.T}K/corrections.json").read_text())
        self._g = {k.split("|", 1)[1]: correction_from_json(v) for k, v in corr.items()
                   if k.startswith("M_bias|")}

    def fold_of(self, tag):
        return tag if self.name == "ihp300" else self.by_tag[tag].meta["group"]

    def folds(self):
        return sorted({self.fold_of(t) for t in self.by_tag})

    def held_of(self, fold):
        return sorted(t for t in self.by_tag if self.fold_of(t) == fold)

    def train_of(self, fold):
        if self.name == "ihp300":
            cls = self.by_tag[fold].meta["cls"]
            return [r for r in self.records if r.meta["cls"] == cls and r.tag != fold]
        return [r for r in self.records if r.meta["group"] != fold]

    def g_shared(self, fold):
        return self._g[fold]

    def box(self, tag):
        return self.m.base_box(self.devices[tag])

    def problem(self, tag):
        return self.m.MeasProblem(self.devices[tag], self.T, self.box(tag))

    def corrected(self, prob, corr):
        return self.m.corrected_problem(prob, corr)

    def full_record(self, tag, params):
        dev = self.devices[tag]
        prob = self.m.MeasProblem(dev, self.T, self.box(tag))
        return self.m.to_record(dev, self.T, prob.simulate_params(params), params)

    def train_local(self, rec):
        box = self.box(rec.tag)
        return box.params_to_local({**box.published,
                                    **{k: v for k, v in rec.meta["params"].items() if k in box.names}})

    def same_kind(self, a, b):
        if self.name == "ihp300":
            return a.family == b.family and a.meta["cls"] == b.meta["cls"]
        return a.family == b.family

    def n_timeout(self):
        return int(getattr(self.m, "N_TIMEOUT", 0))


_DS: dict = {}


def dataset(name):
    if name not in _DS:
        _DS[name] = Sky130() if name == "sky130" else Meas(name)
    return _DS[name]


# ---------------------------------------------------------------------------
# Worker tasks
# ---------------------------------------------------------------------------
def fold_task(task):
    """Prior and mu selection of one fold."""
    import torch
    torch.set_num_threads(1)
    name, fold, ks = task
    t0 = time.time()
    ds = dataset(name)
    corr = ds.g_shared(fold)
    train = ds.train_of(fold)
    prior, deltas = h.fold_prior(train, corr)
    mu = {str(k): h.select_mu(train, corr, k) for k in ks}
    return {"fold": fold, "prior": prior, "deltas": deltas, "mu": mu,
            "runtime_s": round(time.time() - t0, 1)}


def _score(rec, sims, keys):
    return float(h.score_keys(rec, sims, keys)["rrms"])


def split_task(task):
    """Base fit and every arm for one (held-out device, draw)."""
    import torch
    torch.set_num_threads(1)
    from cryoml.explore.e1_correction import Correction
    from cryoml.explore.i3_2_bias import fit_bias
    from cryoml.explore.i4_1_baselines import matched_nfev
    from cryoml.ngspice_fit import polish
    name, tag, draw, adapt, held, fold_info, opts, d_hyper = task
    t_start = time.time()
    ds = dataset(name)
    tmo0 = ds.n_timeout() if hasattr(ds, "n_timeout") else 0
    fold = ds.fold_of(tag)
    g_sh = ds.g_shared(fold)
    prior, mu = fold_info["prior"], fold_info["mu"]
    prob = ds.problem(tag)
    box = prob.box
    sub = lambda: h.subset_problem(prob, adapt)
    row = {"tag": tag, "draw": draw, "K": h.DRAW_K[draw], "fold": fold, "adapt": adapt,
           "heldout": held, "mu": mu, "arms": {}}
    cards = {}

    # start candidates (no held-out measurement)
    rec_h = ds.by_tag[tag]
    peers = [r for r in ds.train_of(fold) if ds.same_kind(r, rec_h)] or ds.train_of(fold)
    cands = {"center": box.params_to_local(box.published),
             "train_median": np.median(np.array([ds.train_local(r) for r in peers]), axis=0)}
    p0 = sub()
    cand_val = {k: float(np.mean(p0.curve_rrms(p0.simulate(v)))) for k, v in cands.items()}
    start_name = min(cand_val, key=lambda k: (cand_val[k], k))
    row["start"] = {"values": cand_val, "chosen": start_name, "n_peers": len(peers)}

    # base fit on the K adaptation curves (no correction)
    pb = sub()
    t0 = time.time()
    res = polish(pb, cands[start_name], max_nfev=opts["base_nfev"], irls_rounds=opts["base_irls"],
                 diff_step=opts["diff_step"])
    base_local = np.asarray(res["local"], float)
    base_params = box.local_to_params(base_local)
    base_rec = ds.full_record(tag, base_params)
    row["base"] = {"adapt_rrms": _score(base_rec, [c.sim for c in base_rec.curves], adapt),
                   "heldout_rrms": _score(base_rec, [c.sim for c in base_rec.curves], held),
                   "n_sim": pb.n_sim, "runtime_s": round(time.time() - t0, 1)}
    cards["base"] = base_params

    def finish(arm, rec, corr, n_sim, t0, extra=None):
        sims = corr.apply(rec)
        r = {"heldout_rrms": _score(rec, sims, held), "adapt_rrms": _score(rec, sims, adapt),
             "n_sim": n_sim, "runtime_s": round(time.time() - t0, 1)}
        r.update(extra or {})
        row["arms"][arm] = r
        cards[arm] = rec.meta.get("params")
        return r

    an, ai, step = opts["arm_nfev"], opts["arm_irls"], opts["diff_step"]

    # (a) card refit, no correction
    t0 = time.time()
    p = sub()
    res = polish(p, base_local, max_nfev=an, irls_rounds=ai, diff_step=step)
    rec_a = ds.full_record(tag, box.local_to_params(res["local"]))
    finish("a_C0", rec_a, Correction("C0"), p.n_sim, t0)

    # (e) cross-curve residual on (a)'s card
    t0 = time.time()
    sims_e, crossings = h.crossinterp_sims(rec_a, adapt, held)
    row["arms"]["e_crossinterp"] = {
        "heldout_rrms": _score(rec_a, sims_e, held), "adapt_rrms": row["arms"]["a_C0"]["adapt_rrms"],
        "n_sim": 0, "runtime_s": round(time.time() - t0, 2),
        "crossings": crossings, "n_heldout_with_crossing": int(sum(v > 0 for v in crossings.values()))}

    # (b) shared correction in the loop
    t0 = time.time()
    p = sub()
    res = polish(ds.corrected(p, g_sh), base_local, max_nfev=an, irls_rounds=ai, diff_step=step)
    finish("b_shared", ds.full_record(tag, box.local_to_params(res["local"])), g_sh, p.n_sim, t0)

    # (c) hierarchical MAP and (c_ml) flat prior
    n_card = int(box.n_active)
    for arm, mu_arm in (("c_hier", float(mu)), ("c_ml", 0.0)):
        t0 = time.time()
        p = sub()
        if math.isinf(mu_arm):
            corr = h.ShiftedCorrection("hier", base=g_sh, delta=prior.delta(np.zeros(h.N_BASIS)))
            res = polish(ds.corrected(p, corr), base_local, max_nfev=an, irls_rounds=ai,
                         diff_step=step)
            z = np.zeros(h.N_BASIS)
            card_local, extra = res["local"], {"mu": "inf", "n_cache_hits": 0}
        else:
            hp = h.HierProblem(ds.corrected(p, g_sh), g_sh, prior, box)
            nf = matched_nfev(n_card, n_card + h.N_BASIS, an, ai)
            res = h.polish_prior(hp, np.concatenate([base_local, np.zeros(h.N_BASIS)]), mu=mu_arm,
                                 max_nfev=nf, irls_rounds=ai, diff_step=step)
            card_local, z = hp.split(res["local"])
            corr = hp.correction(z)
            extra = {"mu": mu_arm, "max_nfev": nf, "n_cache_hits": res["n_cache_hits"],
                     "objective_J": res["objective"]}
        corr = h.ShiftedCorrection("hier", base=g_sh, delta=prior.delta(z))
        extra.update(z=np.asarray(z).tolist(), delta=corr.delta.tolist(),
                     z_norm2=float(np.dot(z, z)))
        finish(arm, ds.full_record(tag, box.local_to_params(card_local)), corr, p.n_sim, t0, extra)

    # (d) per-device MLP from scratch on the K curves (B5 recipe on the adaptation set)
    t0 = time.time()
    g0 = fit_bias("M_bias", [h.restrict(base_rec, adapt)], hyper=d_hyper, threads=1)
    p = sub()
    res = polish(ds.corrected(p, g0), base_local, max_nfev=an, irls_rounds=ai, diff_step=step)
    rec_d = ds.full_record(tag, box.local_to_params(res["local"]))
    g1 = fit_bias("M_bias", [h.restrict(rec_d, adapt)], hyper=d_hyper, threads=1)
    finish("d_scratch", rec_d, g1, p.n_sim, t0,
           {"heldout_rrms_g0": _score(rec_d, g0.apply(rec_d), held)})

    row["n_sim_timeout"] = (ds.n_timeout() - tmo0) if hasattr(ds, "n_timeout") else 0
    row["runtime_s"] = round(time.time() - t_start, 1)
    print(f"  {name} {tag:30s} {draw} " + " ".join(
        f"{a}={row['arms'][a]['heldout_rrms']:.4f}" for a in h.ARMS if a in row["arms"])
        + f" start={start_name} {row['runtime_s']} s", flush=True)
    return row, {"tag": tag, "draw": draw, "cards": cards}


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def aggregate(values: dict) -> dict:
    v = np.array(list(values.values()), dtype=float)
    finite = v[np.isfinite(v)]
    fam = {f: [x for t, x in values.items() if f"{f}_" in t and np.isfinite(x)]
           for f in ("nmos", "pmos")}
    out = {"n": int(len(v)), "n_failed": int(np.sum(~np.isfinite(v))),
           "all_device_mean": float(np.mean(v)) if len(v) and np.all(np.isfinite(v)) else math.nan,
           "valid_only_mean": float(np.mean(finite)) if len(finite) else math.nan,
           "median": float(np.median(finite)) if len(finite) else math.nan}
    means = [float(np.mean(fam[f])) for f in ("nmos", "pmos") if fam[f]]
    out["combined"] = float(np.mean(means)) if len(means) == 2 else math.nan
    return out


def comparisons(rows, clusters=None, k=None) -> dict:
    from cryoml.explore.i3_2_bias import paired_ratio
    vals = {a: h.device_values(rows, a, k) for a in h.ARMS}
    out = {}
    for name, x, y in PAIRS:
        out[name] = paired_ratio(vals[x], vals[y])
        if clusters is not None:
            cl = paired_ratio(vals[x], vals[y], clusters=clusters)
            out[name]["cluster_bootstrap"] = {kk: cl.get(kk) for kk in ("ci_low", "ci_high",
                                                                        "resample_unit")}
    return out


def summarize(rows, clusters=None) -> dict:
    out = {}
    for label, k in (("all", None), ("K2", 2), ("K4", 4)):
        if not any(k is None or r["K"] == k for r in rows):
            continue
        vals = {a: h.device_values(rows, a, k) for a in h.ARMS}
        out[label] = {"aggregate": {a: aggregate(v) for a, v in vals.items()},
                      "comparisons": comparisons(rows, clusters, k), "per_device": vals}
    return out


def json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, h.Prior):
        return o.to_json()
    raise TypeError(type(o))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=json_default) + "\n")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def run(args) -> int:
    t_start = time.time()
    name = args.dataset
    ds = dataset(name)
    opts = dict(OPTIONS)
    d_hyper = {}
    draw_names = h.DRAWS
    folds = ds.folds()
    if args.smoke:
        opts.update(base_nfev=SMOKE["base_nfev"], arm_nfev=SMOKE["arm_nfev"])
        d_hyper = dict(SMOKE["d_hyper"])
        draw_names = SMOKE["draws"]
        folds = folds[:1]
    if args.draws:
        draw_names = tuple(args.draws.split(","))
    ks = sorted({h.DRAW_K[d] for d in draw_names})
    preregistered = (not args.smoke and opts == OPTIONS and tuple(draw_names) == h.DRAWS)
    tasks_split, skipped = [], []
    held_all = []
    for f in folds:
        held = ds.held_of(f)[:1] if args.smoke else ds.held_of(f)
        for tag in held:
            held_all.append(tag)
            for dname, dr in h.draws(ds.by_tag[tag], draw_names).items():
                if dr is None:
                    skipped.append({"tag": tag, "draw": dname})
                else:
                    tasks_split.append((tag, dname, dr["adapt"], dr["heldout"]))
    print(f"I7_2 {name}: {len(folds)} folds, {len(held_all)} held-out devices, "
          f"{len(tasks_split)} splits ({len(skipped)} draws undefined), opts {opts}, "
          f"preregistered={preregistered}", flush=True)
    timing = {}
    with get_context("spawn").Pool(args.workers) as pool:
        t0 = time.time()
        fold_out = pool.map(fold_task, [(name, f, ks) for f in folds], chunksize=1)
        timing["folds_s"] = round(time.time() - t0, 1)
        finfo = {fo["fold"]: fo for fo in fold_out}
        for fo in fold_out:
            print(f"  fold {fo['fold']}: mu " + ", ".join(
                f"K{k}={v['mu']}" for k, v in fo["mu"].items())
                + f", prior sd {np.round(np.sqrt(np.diag(fo['prior'].S)), 3).tolist()}", flush=True)
        tasks = []
        for tag, dname, adapt, held in tasks_split:
            fo = finfo[ds.fold_of(tag)]
            info = {"prior": fo["prior"], "mu": fo["mu"][str(h.DRAW_K[dname])]["mu"]}
            tasks.append((name, tag, dname, adapt, held, info, opts, d_hyper))
        t0 = time.time()
        out = pool.map(split_task, tasks, chunksize=1)
        timing["splits_s"] = round(time.time() - t0, 1)
    rows = [r for r, _ in out]
    cards = [c for _, c in out]
    clusters = None if name in ("sky130", "ihp300") else {t: ds.fold_of(t) for t in held_all}
    target = OUT / (f"smoke_{name}" if args.smoke else name)
    result = {"dataset": name, "prereg": str(PREREG.relative_to(ROOT)),
              "preregistered_config": preregistered, "smoke": bool(args.smoke), "options": opts,
              "d_hyper_overrides": d_hyper, "draws": list(draw_names), "n_folds": len(folds),
              "held_out": held_all, "skipped_draws": skipped,
              "folds": {f: {"prior": fo["prior"].to_json(), "train_deltas": fo["deltas"],
                            "mu_selection": fo["mu"], "runtime_s": fo["runtime_s"]}
                        for f, fo in finfo.items()},
              "summary": summarize(rows, clusters),
              "start_choice": {k: sum(r["start"]["chosen"] == k for r in rows)
                               for k in ("center", "train_median")},
              "n_sim_timeout": int(sum(r.get("n_sim_timeout", 0) for r in rows)),
              "rows": rows, "timing": timing, "runtime_s": round(time.time() - t_start, 1)}
    write_json(target / "results.json", result)
    write_json(target / "cards.json", cards)
    s = result["summary"]["all"]
    for a, v in s["aggregate"].items():
        print(f"  {a:14s} mean {v['all_device_mean']:.4f} median {v['median']:.4f} "
              f"failed {v['n_failed']}")
    for k, v in s["comparisons"].items():
        print(f"  {k:10s} GMR {v.get('ratio', math.nan):.3f} [{v.get('ci_low', math.nan):.3f}, "
              f"{v.get('ci_high', math.nan):.3f}] wins {v['wins']}/{v['n_used']}")
    print(f"timing {timing} runtime {result['runtime_s']} s")
    return 0


def decide(args) -> int:
    prefix = "smoke_" if args.smoke else ""
    res = {}
    for d in DATASETS:
        p = OUT / f"{prefix}{d}" / "results.json"
        if p.exists():
            res[d] = json.loads(p.read_text())
    missing = [d for d in DATASETS if d not in res
               or (not args.smoke and not res[d].get("preregistered_config"))]
    cmp, failed = {}, {}
    for d, r in res.items():
        c = r["summary"]["all"]["comparisons"]
        cmp[d] = {"c_vs_b": c["c_vs_b"], "c_vs_a": c["c_vs_a"], "c_vs_d": c["c_vs_d"]}
        agg = r["summary"]["all"]["aggregate"]["c_hier"]
        failed[d] = {"n": agg["n"], "n_failed": agg["n_failed"],
                     "n_paired": min(c["c_vs_b"]["n_used"], c["c_vs_a"]["n_used"],
                                     c["c_vs_d"]["n_used"])}
    out = h.decide(cmp, failed)
    out.update({"prereg": str(PREREG.relative_to(ROOT)), "smoke": bool(args.smoke),
                "missing_or_not_preregistered": missing,
                "table": {d: {n: {k: v.get(k) for k in ("ratio", "ci_low", "ci_high", "wins",
                                                        "n_used")}
                              for n, v in res[d]["summary"]["all"]["comparisons"].items()}
                          for d in res},
                "failed": failed,
                "note": "DEV data were used by earlier directions; results are development evidence"})
    write_json(OUT / f"{prefix}decision.json", out)
    print(json.dumps({k: out[k] for k in ("conditions", "counts", "C4_per_dataset", "outcome",
                                          "missing_or_not_preregistered")}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("run", "decide"), required=True)
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--draws", default="", help="override the draw list (not preregistered)")
    args = ap.parse_args()
    if args.mode == "decide":
        return decide(args)
    if not args.dataset:
        ap.error("--dataset is required")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
