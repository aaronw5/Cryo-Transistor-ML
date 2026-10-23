#!/usr/bin/env python3
"""I18-1: gated correction ("selected g v2 applied only if the training-only
S3 gate predicts a gain, else C0_refit") on every measured DEV dataset with
stored rows.

Pre-registration: out/research/explore/I18_1/prereg.md (prereg.json).

Stages (``--stage``):

  s3      S3 per outer fold for --parts (comma list):
            sky130_v2   sky130 77 K all-18, g v2 recipe at lambda*_f (I15-2)
            sky130_v1   sky130 77 K all-18, v1 recipe (secondary)
            ihp_v2      IHP 300 K, g v2 recipe at lambda*_f (I17-2)
            v2check     IHP 300 K v1 recipe on 2 folds (validity V2)
            tud4_v2, tud295_v2   TU Delft g v2 recipe at lambda*_f (I18-2)
          -> s3/<part>.json
  decide  validity, gated rows, rule, secondary tables -> decision.json

  PYTHONPATH=src .venv/bin/python scripts/research/explore/i18_1_gate.py --stage s3 --parts sky130_v2,ihp_v2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "out/research/explore"
I13 = EXP / "I13_1"
# sky130 state-C records (as I15-2); set before the sky130 drivers are imported
os.environ["CRYOML_E1_RECORDS"] = str(I13 / "records")
os.environ["CRYOML_I3_2_ALL18"] = str(I13 / "all18")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(ROOT / "scripts" / "research" / "explore"))

from cryoml.explore import i18_1_gate as m  # noqa: E402

OUT = EXP / "I18_1"
PREREG = OUT / "prereg.json"
S3_HYPER = {"seeds": (m.S3_SEED,), "steps": m.S3_STEPS}
SMOKE = {"folds": 2, "steps": 50}
V2_FOLDS = ("hv_nmos_W0p3_L0p5", "lv_nmos_W0p15_L0p18")
UNIT_TESTS = ("tests.test_i18_1_gate",)
PARTS = ("sky130_v2", "sky130_v1", "ihp_v2", "v2check", "tud4_v2", "tud295_v2")


def jload(p: Path):
    return json.loads(Path(p).read_text())


def jdump(obj, p: Path) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, default=float) + "\n")


# ---------------------------------------------------------------------------
# dataset adapters: records (load order), outer folds, inner groups, recipe
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def _sky():
    if "sky" not in _CACHE:
        import i12_1_symm as drv12
        _CACHE["sky"] = drv12
    return _CACHE["sky"]


def _ihp():
    if "ihp" not in _CACHE:
        import i17_2_ihp as drv17
        _CACHE["ihp"] = drv17
    return _CACHE["ihp"]


def _tud():
    if "tud" not in _CACHE:
        import i18_2_tud as drv18
        _CACHE["tud"] = drv18
    return _CACHE["tud"]


def part_spec(part: str) -> dict:
    """Records by tag, load order, outer folds {fold: training tags in load
    order}, inner group of a tag, recipe, lambda*_f, vdd per fold."""
    if part.startswith("sky130"):
        from cryoml.explore import i12_1_symm as v2
        drv = _sky()
        recs = drv.load_records()
        order = drv.load_order()
        folds = {h: v2.fold_train_tags(order, h) for h in sorted(order)}
        lam = {k: float(v) for k, v in jload(EXP / "I15_2/decision.json")["lambda_star"].items()}
        return {"recs": recs, "folds": folds, "group": lambda t: t, "lam": lam,
                "vdd": {h: v2.VDD_NORM for h in folds}, "recipe": part.split("_")[1], "pens": "sky130"}
    if part in ("ihp_v2", "v2check"):
        drv = _ihp()
        recs = drv.by_tag()
        order = drv.load_order()
        from cryoml.explore import i17_2_ihp as m17
        folds = {h: m17.fold_train(order, h) for h in sorted(order)}
        if part == "v2check":
            folds = {h: folds[h] for h in V2_FOLDS}
        lam = {k: float(v) for k, v in jload(EXP / "I17_2/inner/inner.json")["lambda_star"].items()}
        return {"recs": recs, "folds": folds, "group": lambda t: t, "lam": lam,
                "vdd": {h: drv.devices()[h].vdd for h in folds}, "recipe": "v1" if part == "v2check" else "v2",
                "pens": "ihp"}
    if part in ("tud4_v2", "tud295_v2"):
        drv = _tud()
        t = 4 if part.startswith("tud4") else 295
        recs = drv.by_tag(t)
        order = drv.load_order(t)
        group = drv.group_of
        folds = {g: [x for x in order if group(x) != g] for g in sorted({group(x) for x in order})}
        lam = {k: float(v) for k, v in drv.lambda_star(t).items()}
        return {"recs": recs, "folds": folds, "group": group, "lam": lam, "vdd": {g: drv.VDD for g in folds},
                "recipe": "v2", "pens": f"tud{t}", "T": t}
    raise KeyError(part)


def load_pens(kind: str, tags) -> list:
    if kind == "sky130":
        allp = _sky().load_pens()
    elif kind == "ihp":
        allp = _ihp().load_pens(EXP / "I17_2" / "pens")
    else:
        allp = _tud().load_pens(int(kind[3:]))
    return [allp[t] for t in tags]


def s3_task(task):
    """(part, fold, inner group, steps) -> (part, fold, group, log ratios, seconds)."""
    from cryoml.explore.i7_1_transfer import s3_fold_value
    part, fold, k, steps = task
    sp = part_spec(part)
    recs = sp["recs"]
    train = sp["folds"][fold]
    inner_train = [t for t in train if sp["group"](t) != k]
    scored = [recs[t] for t in train if sp["group"](t) == k]
    t0 = time.time()
    if sp["recipe"] == "v1":
        from cryoml.explore.i3_2_bias import fit_bias
        corr = fit_bias("M_bias", [recs[t] for t in inner_train], hyper={**S3_HYPER, "steps": steps}, threads=1)
    else:
        from cryoml.explore import i12_1_symm as v2
        lam = float(sp["lam"][fold])
        pens = load_pens(sp["pens"], inner_train) if lam else None
        corr = v2.fit_v2([recs[t] for t in inner_train], pens, lam, hyper={**S3_HYPER, "steps": steps},
                         inputs="sym_v2", eps=v2.EPS, threads=1, arm="S3_v2", vdd=float(sp["vdd"][fold]))
    return part, fold, k, s3_fold_value(corr, scored), [t for t in train if sp["group"](t) == k], \
        round(time.time() - t0, 2)


def stage_s3(parts, workers: int, smoke: bool) -> dict:
    steps = SMOKE["steps"] if smoke else m.S3_STEPS
    out_dir = OUT / ("smoke/s3" if smoke else "s3")
    res = {}
    for part in parts:
        sp = part_spec(part)
        folds = sorted(sp["folds"])
        if smoke:
            folds = folds[:SMOKE["folds"]]
        tasks, n_groups = [], {}
        for f in folds:
            groups = sorted({sp["group"](t) for t in sp["folds"][f]})
            n_groups[f] = len(groups)
            tasks += [(part, f, k, steps) for k in groups]
        t0 = time.time()
        with get_context("spawn").Pool(workers) as pool:
            rows = pool.map(s3_task, tasks, chunksize=1)
        per = {}
        for _, f, k, lr, tags, sec in rows:
            d = per.setdefault(f, {"inner": {}, "seconds": 0.0})
            d["inner"][k] = {"tags": tags, "log_ratios": lr}
            d["seconds"] += sec
        for f, d in per.items():
            vals = [v for x in d["inner"].values() for v in x["log_ratios"]]
            d["S3"] = m.fold_s3(vals)
            d["n_inner_fits"] = len(d["inner"])
            d["n_groups"] = n_groups[f]
            d["n_scored"] = len(vals)
            d["n_finite"] = int(np.sum(np.isfinite(vals)))
            if sp["recipe"] == "v2":
                d["lambda"] = float(sp["lam"][f])
        out = {"part": part, "recipe": sp["recipe"], "steps": steps, "seed": m.S3_SEED, "smoke": smoke,
               "n_tasks": len(tasks), "wall_s": round(time.time() - t0, 1), "per_fold": per,
               "S3": {f: per[f]["S3"] for f in sorted(per)}}
        jdump(out, out_dir / f"{part}.json")
        v = np.array(list(out["S3"].values()), float)
        print(f"S3 {part}: {len(tasks)} inner fits {out['wall_s']} s; folds {len(v)}, S3 < 0 on "
              f"{int(np.sum(v < 0))}, range [{np.nanmin(v):.4f}, {np.nanmax(v):.4f}]", flush=True)
        res[part] = out
    return res


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------
def run_unit_tests() -> dict:
    r = subprocess.run([sys.executable, "-m", "unittest", *UNIT_TESTS], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"})
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    return {"files": list(UNIT_TESTS), "rc": r.returncode, "passed": r.returncode == 0, "tail": tail}


def stored_s3_i7(name: str) -> dict:
    return dict(jload(EXP / "I7_1/retro/s3.json")["per_fold"][name])


def datasets(fallback_tud: bool = False) -> dict:
    """Primary datasets: corr rows, C0_refit rows, fold_of, S3 by fold,
    clusters (or None), arm, s3 source."""
    ds = {}
    i15 = jload(EXP / "I15_2/decision.json")
    ds["sky130_77K"] = {"arm": "selected g v2 (I15-2)", "corr": dict(i15["selected_rows"]),
                        "c0": dict(jload(I13 / "lodo/results.json")["per_device"]["C0_refit"]),
                        "fold_of": lambda t: t, "s3_src": "s3/sky130_v2.json", "unit": None,
                        "stored_selected": dict(i15["selected_rows"])}
    for name, d, f in (("nist_4K", "I8_1", "prediction_v2.json"), ("nist_rt_card_4K", "I9_1", "prediction.json")):
        pd = jload(EXP / d / "lodo/results.json")["per_device"]
        ds[name] = {"arm": "M_bias v1", "corr": dict(pd["M_bias"]), "c0": dict(pd["C0_refit"]),
                    "fold_of": lambda t: t, "s3": {k: v["S3"] for k, v in jload(EXP / d / f)["per_fold"].items()},
                    "s3_src": f"{d}/{f}", "unit": None}
    i17 = jload(EXP / "I17_2/decision.json")
    ds["ihp_300K"] = {"arm": "selected g v2 (I17-2)", "corr": dict(i17["selected_rows"]),
                      "c0": dict(jload(EXP / "I17_2/base/results.json")["per_device"]["C0_refit"]),
                      "fold_of": lambda t: t, "s3_src": "s3/ihp_v2.json", "unit": None,
                      "stored_selected": dict(i17["selected_rows"])}
    for t in (233, 343, 398):
        pd = jload(EXP / f"I5_1/lodo_{t}K/results.json")["per_device"]
        ds[f"ihp_{t}K"] = {"arm": "M_bias v1", "corr": dict(pd["M_bias"]), "c0": dict(pd["C0_refit"]),
                           "fold_of": lambda x: x, "s3": stored_s3_i7(f"ihp_meas_{t}K"),
                           "s3_src": "I7_1/retro/s3.json", "unit": None}
    from cryoml.explore.i6_1_tudelft import group_of
    for t in (4, 295):
        name = f"tudelft_{t}K"
        dec = EXP / "I18_2/decision.json"
        if not fallback_tud and dec.exists():
            dj = jload(dec)["per_temperature"][str(t)]
            ds[name] = {"arm": "selected g v2 (I18-2, state C)", "corr": dict(dj["selected_rows"]),
                        "c0": dict(dj["rows"]["C0_refit"]), "fold_of": group_of,
                        "s3_src": f"s3/tud{t}_v2.json", "unit": "cluster", "stored_selected": dict(dj["selected_rows"])}
        else:
            pd = jload(EXP / f"I6_1/lodo_{t}K/results.json")["per_device"]
            ds[name] = {"arm": "M_bias v1 (fallback: I6-1 stored, state L)", "corr": dict(pd["M_bias"]),
                        "c0": dict(pd["C0_refit"]), "fold_of": group_of, "s3": stored_s3_i7(f"tudelft_{t}K"),
                        "s3_src": "I7_1/retro/s3.json", "unit": "cluster", "fallback": True}
    for name, d in ds.items():
        if "s3" not in d:
            p = OUT / d["s3_src"]
            d["s3"] = dict(jload(p)["S3"]) if p.exists() else {}
        drop_unpaired(d)
        d["clusters"] = ({t: d["fold_of"](t) for t in d["corr"]} if d["unit"] == "cluster" else None)
    return ds


def _fin(x) -> bool:
    return x is not None and bool(np.isfinite(float(x)))


def drop_unpaired(d: dict) -> None:
    """Leave out devices whose rows are non-finite in both arms and whose fold
    has no S3 (NIST 4 K: the 4 records excluded by I8-1 amendment A1)."""
    drop = [t for t in d["corr"] if d["fold_of"](t) not in d["s3"]
            and not _fin(d["corr"][t]) and not _fin(d["c0"].get(t))]
    for t in drop:
        d["corr"].pop(t)
        d["c0"].pop(t, None)
    d["dropped_unpaired"] = drop


def secondary_v1() -> dict:
    """v1 gate on every dataset with v1 rows."""
    from cryoml.explore.i6_1_tudelft import group_of
    out = {}
    s3 = OUT / "s3/sky130_v1.json"
    out["sky130_77K"] = {"corr": dict(jload(I13 / "all18/results.json")["per_device"]["M_bias"]),
                         "c0": dict(jload(I13 / "lodo/results.json")["per_device"]["C0_refit"]),
                         "fold_of": lambda t: t, "s3": dict(jload(s3)["S3"]) if s3.exists() else {}, "unit": None}
    for name, d, f in (("nist_4K", "I8_1", "prediction_v2.json"), ("nist_rt_card_4K", "I9_1", "prediction.json")):
        pd = jload(EXP / d / "lodo/results.json")["per_device"]
        out[name] = {"corr": dict(pd["M_bias"]), "c0": dict(pd["C0_refit"]), "fold_of": lambda t: t,
                     "s3": {k: v["S3"] for k, v in jload(EXP / d / f)["per_fold"].items()}, "unit": None}
    for t in (233, 300, 343, 398):
        pd = jload(EXP / f"I5_1/lodo_{t}K/results.json")["per_device"]
        out[f"ihp_{t}K"] = {"corr": dict(pd["M_bias"]), "c0": dict(pd["C0_refit"]), "fold_of": lambda x: x,
                            "s3": stored_s3_i7(f"ihp_meas_{t}K"), "unit": None}
    for t in (4, 295):
        pd = jload(EXP / f"I6_1/lodo_{t}K/results.json")["per_device"]
        out[f"tudelft_{t}K"] = {"corr": dict(pd["M_bias"]), "c0": dict(pd["C0_refit"]), "fold_of": group_of,
                                "s3": stored_s3_i7(f"tudelft_{t}K"), "unit": "cluster"}
    for d in out.values():
        if d["s3"]:
            drop_unpaired(d)
        d["clusters"] = ({t: d["fold_of"](t) for t in d["corr"]} if d["unit"] == "cluster" else None)
    return out


def evaluate(d: dict) -> dict:
    """Gated rows, decisions, GMRs and references of one dataset."""
    from cryoml.explore.i3_2_bias import paired_ratio
    from cryoml.explore.i7_1_transfer import spearman
    rows, dec = m.gated_rows(d["corr"], d["c0"], d["fold_of"], d["s3"])
    tags = sorted(rows)
    n_held = {}
    for t in tags:
        n_held[d["fold_of"](t)] = n_held.get(d["fold_of"](t), 0) + 1
    cl = d["clusters"]
    alt = {t: d["fold_of"](t) for t in tags} if cl is None else None
    y = m.fold_y(d["corr"], d["c0"], d["fold_of"])
    common = sorted(set(y) & set(d["s3"]))
    gate_d = m.dataset_gate({f: d["s3"][f] for f in n_held}, n_held)
    rows_d = {t: (d["corr"][t] if gate_d["apply"] else d["c0"][t]) for t in tags}
    g_corr = paired_ratio(d["corr"], d["c0"], clusters=cl)
    out = {"n_rows": m.n_finite_pairs(rows, d["c0"]), "n_folds": len(dec),
           "folds_apply": int(sum(dec.values())),
           "devices_apply": int(sum(1 for t in tags if dec[d["fold_of"](t)])),
           "decisions": dec, "s3": {f: d["s3"][f] for f in dec},
           "gmr_gated_vs_C0_refit": paired_ratio(rows, d["c0"], clusters=cl),
           "gmr_gated_vs_C0_refit_other_unit": paired_ratio(rows, d["c0"], clusters=alt),
           "gmr_gated_vs_correction": paired_ratio(rows, d["corr"], clusters=cl),
           "gmr_correction_vs_C0_refit": g_corr,
           "dataset_gate": {**gate_d, "gmr_vs_C0_refit": paired_ratio(rows_d, d["c0"], clusters=cl)["ratio"]},
           "oracle_post_hoc": min(1.0, g_corr["ratio"]),
           "spearman_s3_y": spearman([d["s3"][f] for f in common], [y[f] for f in common]) if len(common) > 2
           else float("nan"),
           "fold_y": y, "gated_rows": rows}
    return out


def stage_decide(smoke: bool = False) -> dict:
    tud_ok = (EXP / "I18_2/decision.json").exists() and jload(EXP / "I18_2/decision.json")["validity"]["valid"]
    ds = datasets(fallback_tud=not tud_ok)
    res = {name: {"arm": d["arm"], "s3_source": d["s3_src"], "unit": d["unit"] or "device",
                  "fallback": bool(d.get("fallback")), "dropped_unpaired": d["dropped_unpaired"],
                  **evaluate(d)} for name, d in ds.items()}
    # validity
    sel_diff = 0.0
    for name, d in ds.items():
        if "stored_selected" in d:
            sel_diff = max(sel_diff, max(abs(d["corr"][t] - d["stored_selected"][t]) for t in d["corr"]))
    v2c = OUT / "s3/v2check.json"
    v2max, v2n = float("inf"), 0
    if v2c.exists():
        got = jload(v2c)["S3"]
        ref = stored_s3_i7("ihp_meas_300K")
        v2max = max(abs(got[f] - ref[f]) for f in got)
        v2n = len(got)
    v3 = True
    for name in m.V2_DATASETS:
        d = ds[name]
        if d.get("fallback"):
            continue
        p = OUT / d["s3_src"]
        if not p.exists():
            v3 = False
            continue
        j = jload(p)
        folds = sorted({d["fold_of"](t) for t in d["corr"]})
        v3 = v3 and all(f in j["per_fold"] and np.isfinite(j["per_fold"][f]["S3"])
                        and j["per_fold"][f]["n_inner_fits"] == j["per_fold"][f]["n_groups"]
                        for f in folds) and not j["smoke"]
    unit = run_unit_tests()
    validity = m.decide_validity({"v1_rows": {k: v["n_rows"] for k, v in res.items()},
                                  "v1_selected_max_diff": sel_diff, "v2_max_diff": v2max, "v2_n": v2n,
                                  "v3_complete": v3, "v4_unit_tests_pass": unit["passed"]})
    rule = m.decide_rule({k: v["gmr_gated_vs_C0_refit"] for k, v in res.items()})
    pooled = float(math.exp(np.mean([math.log(v["gmr_gated_vs_C0_refit"]["ratio"]) for v in res.values()])))
    pooled_corr = float(math.exp(np.mean([math.log(v["gmr_correction_vs_C0_refit"]["ratio"]) for v in res.values()])))
    sec = {name: {k: v for k, v in evaluate(d).items() if k not in ("gated_rows", "fold_y")}
           for name, d in secondary_v1().items() if d["s3"]}
    s3cmp = {}
    for name, a, b in (("sky130_77K", "s3/sky130_v2.json", "s3/sky130_v1.json"),
                       ("ihp_300K", "s3/ihp_v2.json", None)):
        if (OUT / a).exists():
            va = jload(OUT / a)["S3"]
            vb = jload(OUT / b)["S3"] if b and (OUT / b).exists() else stored_s3_i7("ihp_meas_300K")
            s3cmp[name] = {f: [va[f], vb.get(f)] for f in va}
    out = {"prereg": str(PREREG.relative_to(ROOT)), "validity": validity,
           "outcome": rule["outcome"] if validity["valid"] else "invalid", "rule": rule,
           "unit_tests": unit, "tudelft_source": "I18-2" if tud_ok else "fallback I6-1",
           "datasets": res, "pooled_exp_mean_ln_gmr_gated": pooled, "pooled_exp_mean_ln_gmr_correction": pooled_corr,
           "secondary_v1_gate": sec, "s3_v2_vs_v1": s3cmp,
           "predictions": jload(PREREG)["predictions"]}
    jdump(out, OUT / "decision.json")
    print(json.dumps({"validity": validity, "outcome": out["outcome"], "rule": rule,
                      "per_dataset": {k: {"apply": f"{v['folds_apply']}/{v['n_folds']}",
                                          "gmr": [round(v['gmr_gated_vs_C0_refit'][x], 4)
                                                  for x in ("ratio", "ci_low", "ci_high")]}
                                      for k, v in res.items()},
                      "pooled": pooled,
                      "secondary_v1": {k: {"apply": f"{v['folds_apply']}/{v['n_folds']}",
                                           "gmr": round(v['gmr_gated_vs_C0_refit']['ratio'], 4)}
                                       for k, v in sec.items()}}, indent=1, default=float), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("s3", "decide"))
    ap.add_argument("--parts", default="sky130_v2,sky130_v1,ihp_v2,v2check")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true", help="2 folds per part, 50 steps (code check only)")
    args = ap.parse_args()
    t0 = time.time()
    if args.stage == "s3":
        parts = [p for p in args.parts.split(",") if p]
        bad = [p for p in parts if p not in PARTS]
        if bad:
            raise SystemExit(f"unknown parts {bad}")
        stage_s3(parts, args.workers, args.smoke)
    else:
        stage_decide(args.smoke)
    print(f"stage {args.stage} done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
