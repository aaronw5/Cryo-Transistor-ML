#!/usr/bin/env python3
"""I19-1: dataset-level gate.  One decision per dataset from S3_D (the I7-1 S3
statistic computed once with the training set = all DEV devices of the
dataset); the correction is applied to every device iff S3_D < 0, else
C0_refit.

Pre-registration: out/research/explore/I19_1/prereg.md (prereg.json).

Stages (``--stage``):

  s3        S3_D for --parts (comma list) -> s3/<part>.json
              primary   sky130_v2 nist4_v1 nistrt_v1 ihp300_v2 ihp233_v1 ihp343_v1
                        ihp398_v1 tud4_v2 tud295_v2
              secondary sky130_v1 ihp300_v1 tud4_v1 tud295_v1
              checks    check_ihp_v1 (I7-1 folds), check_sky_v2 (I18-1 fold)
              sim       the 20 I7-1 simulated datasets (v1)
  testgate  Part P: TEST gate decisions from the DEV S3_D values
            -> test_gate_decisions.{json,md}
  decide    Part R, Part C, validity -> decision.json

  PYTHONPATH=src .venv/bin/python scripts/research/explore/i19_1_dataset_gate.py --stage s3 --parts real
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(ROOT / "scripts" / "research" / "explore"))

import i18_1_gate as g18  # noqa: E402  (sets the sky130 state-C record env vars)
from cryoml.explore import i18_1_gate as m18  # noqa: E402
from cryoml.explore import i19_1_dataset_gate as m  # noqa: E402

OUT = EXP / "I19_1"
PREREG = OUT / "prereg.json"
S3_HYPER = {"seeds": (m.S3_SEED,), "steps": m.S3_STEPS}
SMOKE = {"tasks": 2, "steps": 50, "sim": 1}
UNIT_TESTS = ("tests.test_i19_1_dataset_gate",)
PRIMARY_PART = {"sky130_77K": "sky130_v2", "nist_4K": "nist4_v1", "nist_rt_card_4K": "nistrt_v1",
                "ihp_300K": "ihp300_v2", "ihp_233K": "ihp233_v1", "ihp_343K": "ihp343_v1", "ihp_398K": "ihp398_v1",
                "tudelft_4K": "tud4_v2", "tudelft_295K": "tud295_v2"}
V1_PART = {"sky130_77K": "sky130_v1", "nist_4K": "nist4_v1", "nist_rt_card_4K": "nistrt_v1",
           "ihp_233K": "ihp233_v1", "ihp_300K": "ihp300_v1", "ihp_343K": "ihp343_v1", "ihp_398K": "ihp398_v1",
           "tudelft_4K": "tud4_v1", "tudelft_295K": "tud295_v1"}
REAL_PARTS = tuple(dict.fromkeys(list(PRIMARY_PART.values()) + list(V1_PART.values())))
CHECK_PARTS = ("check_ihp_v1", "check_sky_v2")
CHECK_SKY_FOLD = "nmos_L0p15_W1p6"
SPLITS = {"nist4": "I8_1/split.json", "nistrt": "I9_1/split.json", "ihp": "I5_1/split.json",
          "tud": "I6_1/split.json"}


def jload(p: Path):
    return json.loads(Path(p).read_text())


def jdump(obj, p: Path) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, default=float) + "\n")


def rel(p: Path) -> str:
    return str(Path(p).relative_to(ROOT))


# ---------------------------------------------------------------------------
# part specs: records (tag -> record), load order, group, LOGO training set,
# recipe, lambda / vdd per held-out group, penalty-grid kind, split kind
# ---------------------------------------------------------------------------
_SPEC: dict = {}


def _finite(path: Path) -> list:
    from cryoml.explore.e1_correction import load_devices
    from cryoml.explore.i8_1_nist import split_finite
    good, _ = split_finite(load_devices(path))
    return sorted(good, key=lambda r: r.tag)


def _from_list(recs: list, group, train_of, recipe: str, split: str | None, name: str) -> dict:
    order = [r.tag for r in recs]
    return {"recs": {r.tag: r for r in recs}, "order": order, "group": group, "train_of": train_of,
            "recipe": recipe, "split": split, "name": name, "lam": None, "vdd": None, "pens": None}


def spec(part: str) -> dict:
    if part in _SPEC:
        return _SPEC[part]
    from cryoml.explore.e1_correction import load_devices
    if part in ("sky130_v2", "sky130_v1", "ihp300_v2", "tud4_v2", "tud295_v2", "check_sky_v2", "check_ihp_v1"):
        src = {"check_sky_v2": "sky130_v2", "check_ihp_v1": "v2check"}.get(part, part)
        src = {"ihp300_v2": "ihp_v2"}.get(src, src)
        s = g18.part_spec(src)
        order = [t for t in (g18._sky().load_order() if src.startswith("sky130")
                             else g18._ihp().load_order() if src in ("ihp_v2", "v2check")
                             else g18._tud().load_order(s["T"]))]
        folds = s["folds"]
        if src == "v2check":       # I18-1 V2: fold training sets of the two I7-1 folds
            sp = {"recs": s["recs"], "order": order, "group": s["group"], "folds": folds, "recipe": "v1",
                  "split": "ihp", "lam": None, "vdd": None, "pens": None, "name": part}
        else:
            sp = {"recs": s["recs"], "order": order, "group": s["group"],
                  "train_of": (lambda k, f=folds: f[k]), "folds": folds, "recipe": s["recipe"],
                  "lam": s["lam"], "vdd": s["vdd"], "pens": s["pens"], "name": part,
                  "split": None if src.startswith("sky130") else ("ihp" if src == "ihp_v2" else "tud")}
    elif part in ("nist4_v1", "nistrt_v1"):
        recs = _finite(EXP / ("I8_1" if part == "nist4_v1" else "I9_1") / "start" / "records")
        order = [r.tag for r in recs]
        sp = _from_list(recs, lambda t: t, lambda k, o=order: [t for t in o if t != k], "v1", part[:-3], part)
    elif part.startswith("ihp") and part.endswith("_v1"):
        t = int(part[3:6])
        recs = load_devices(EXP / f"I5_1/start/records_{t}K")
        cls = {r.tag: r.meta["cls"] for r in recs}
        order = [r.tag for r in recs]
        sp = _from_list(recs, lambda x: x, lambda k, o=order, c=cls: [x for x in o if x != k and c[x] == c[k]],
                        "v1", "ihp", part)
        sp["cls"] = cls
    elif part in ("tud4_v1", "tud295_v1"):
        t = 4 if part.startswith("tud4") else 295
        recs = load_devices(EXP / f"I6_1/start/records_{t}K")
        grp = {r.tag: r.meta["group"] for r in recs}
        order = [r.tag for r in recs]
        sp = _from_list(recs, grp.get, lambda k, o=order, g=grp: [x for x in o if g[x] != k], "v1", "tud", part)
    elif part.startswith("sim:"):
        import i7_1_transfer as d7
        kind, recs = d7.dataset_records("new:" + part[4:])
        g = d7.group_fn(kind)
        grp = {r.tag: g(r) for r in recs}
        order = [r.tag for r in recs]
        sp = _from_list(recs, grp.get, lambda k, o=order, gg=grp: [x for x in o if gg[x] != k], "v1", None, part)
    else:
        raise KeyError(part)
    if part.startswith("ihp") and "cls" not in sp:
        sp["cls"] = {x: ("hv" if x.startswith("hv") else "lv") for x in sp["order"]}
    _SPEC[part] = sp
    return sp


def sim_names() -> list[str]:
    return sorted(jload(EXP / "I7_1/decision.json")["datasets"])


def dev_check(part: str) -> dict | None:
    """V5: every loaded record tag in the DEV list and not in the TEST list."""
    sp = spec(part)
    if not sp["split"]:
        return None
    s = jload(EXP / SPLITS[sp["split"]])
    tags = set(sp["order"])
    in_test, not_dev = sorted(tags & set(s["TEST"])), sorted(tags - set(s["DEV"]))
    return {"split_file": SPLITS[sp["split"]], "n_records": len(tags), "n_in_TEST": len(in_test),
            "n_not_in_DEV": len(not_dev), "ok": not in_test and not not_dev, "in_TEST": in_test,
            "not_in_DEV": not_dev}


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------
def plan(part: str) -> list[tuple]:
    """(part, label, inner-training tags, scored tags, lambda, vdd)."""
    sp = spec(part)
    if part == "check_ihp_v1":
        out = []
        for f, train in sorted(sp["folds"].items()):
            for k in sorted({sp["group"](t) for t in train}):
                out.append((part, f"{f}|{k}", [t for t in train if sp["group"](t) != k],
                            [t for t in train if sp["group"](t) == k], None, None))
        return out
    if part == "check_sky_v2":
        f = CHECK_SKY_FOLD
        train = sp["folds"][f]
        return [(part, f"{f}|{k}", [t for t in train if t != k], [k], float(sp["lam"][f]), float(sp["vdd"][f]))
                for k in sorted(train)]
    out = []
    for k, train, scored in m.inner_plan(sp["order"], sp["group"], sp["train_of"]):
        lam = float(sp["lam"][k]) if sp["recipe"] == "v2" else None
        vdd = float(sp["vdd"][k]) if sp["recipe"] == "v2" else None
        out.append((part, k, train, scored, lam, vdd))
    return out


def s3_task(task):
    import torch
    torch.set_num_threads(1)
    from cryoml.explore.i7_1_transfer import s3_fold_value
    part, label, train, scored, lam, vdd, steps = task
    sp = spec(part)
    recs = sp["recs"]
    t0 = time.time()
    hyper = {**S3_HYPER, "steps": steps}
    if sp["recipe"] == "v1":
        from cryoml.explore.i3_2_bias import fit_bias
        corr = fit_bias("M_bias", [recs[t] for t in train], hyper=hyper, threads=1)
    else:
        from cryoml.explore import i12_1_symm as v2
        pens = g18.load_pens(sp["pens"], train) if lam else None
        corr = v2.fit_v2([recs[t] for t in train], pens, lam, hyper=hyper, inputs="sym_v2", eps=v2.EPS,
                         threads=1, arm="S3_v2", vdd=vdd)
    return part, label, scored, s3_fold_value(corr, [recs[t] for t in scored]), round(time.time() - t0, 2)


def summarize(part: str, rows: list, steps: int, smoke: bool, wall: float, n_planned: int) -> dict:
    sp = spec(part)
    groups, per_rec = {}, {}
    for _, label, scored, lr, sec in rows:
        groups[label] = {"scored": scored, "log_ratios": lr, "seconds": sec}
        per_rec.update(dict(zip(scored, lr)))
    out = {"part": part, "recipe": sp["recipe"], "steps": steps, "seed": m.S3_SEED, "smoke": smoke,
           "n_inner_fits": len(rows), "n_planned": n_planned, "wall_s": round(wall, 1), "groups": groups}
    if part == "check_ihp_v1" or part == "check_sky_v2":
        per_fold: dict = {}
        for label, g in groups.items():
            per_fold.setdefault(label.split("|")[0], []).extend(g["log_ratios"])
        out["S3_fold"] = {f: m.s3_dataset(v) for f, v in per_fold.items()}
        return out
    vals = [per_rec[t] for t in sp["order"] if t in per_rec]
    out.update({"S3_D": m.s3_dataset(vals), "n_records": len(sp["order"]), "n_scored": len(vals),
                "n_finite": int(np.sum(np.isfinite(vals))), "n_groups": len({sp["group"](t) for t in sp["order"]}),
                "per_record": per_rec, "dev_check": dev_check(part)})
    if sp["recipe"] == "v2":
        out["lambda_by_group"] = {k: float(sp["lam"][k]) for k in groups}
    if "cls" in sp:
        out["S3_D_by_class"] = {c: m.s3_dataset([per_rec[t] for t in sp["order"] if sp["cls"][t] == c
                                                 and t in per_rec]) for c in sorted(set(sp["cls"].values()))}
    return out


def expand_parts(arg: str) -> list[str]:
    out = []
    for p in [x for x in arg.split(",") if x]:
        if p == "real":
            out += list(REAL_PARTS) + list(CHECK_PARTS)
        elif p == "sim":
            out += [f"sim:{n}" for n in sim_names()]
        else:
            out.append(p)
    return out


def part_file(part: str, smoke: bool) -> Path:
    base = OUT / ("smoke/s3" if smoke else "s3")
    return base / ("sim/" + part[4:].replace("|", "_") + ".json" if part.startswith("sim:") else f"{part}.json")


def stage_s3(parts: list[str], workers: int, smoke: bool) -> None:
    steps = SMOKE["steps"] if smoke else m.S3_STEPS
    if smoke:
        sims = [p for p in parts if p.startswith("sim:")][:SMOKE["sim"]]
        parts = [p for p in parts if not p.startswith("sim:")] + sims
    plans = {p: plan(p) for p in parts}
    tasks = []
    for p in parts:
        tk = plans[p][:SMOKE["tasks"]] if smoke else plans[p]
        tasks += [(*t, steps) for t in tk]
    print(f"S3_D: {len(parts)} parts, {len(tasks)} inner fits, steps {steps}", flush=True)
    t0 = time.time()
    rows: dict = {p: [] for p in parts}
    with get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(s3_task, tasks, chunksize=1):
            rows[r[0]].append(r)
            if len(rows[r[0]]) == (min(len(plans[r[0]]), SMOKE["tasks"]) if smoke else len(plans[r[0]])):
                out = summarize(r[0], sorted(rows[r[0]], key=lambda x: x[1]), steps, smoke, time.time() - t0,
                                len(plans[r[0]]))
                jdump(out, part_file(r[0], smoke))
                s = out.get("S3_D", out.get("S3_fold"))
                print(f"  {r[0]}: {out['n_inner_fits']} fits, S3 {s}, {time.time() - t0:.0f} s", flush=True)
    print(f"S3_D done in {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------------------
# Part P: TEST gate decisions
# ---------------------------------------------------------------------------
TEST_SPLITS = [
    {"name": "nist_4K", "test": "NIST 4 K lvt TEST (I8-1 split)", "split": "I8_1/split.json", "arm": "M_bias v1",
     "part": "nist4_v1", "v1_part": None, "i8_2": None, "dev_pc": "I8_1/prediction_v2.json"},
    {"name": "nist_rt_card_4K", "test": "NIST 4 K room-temperature card TEST (I9-1 split)",
     "split": "I9_1/split.json", "arm": "M_bias v1", "part": "nistrt_v1", "v1_part": None, "i8_2": None,
     "dev_pc": "I9_1/prediction.json"},
    *[{"name": f"ihp_{t}K", "test": f"IHP SG13G2 TEST {t} K", "split": "I5_1/split.json",
       "arm": "selected g v2 (I17-2)" if t == 300 else "M_bias v1", "part": PRIMARY_PART[f"ihp_{t}K"],
       "v1_part": "ihp300_v1" if t == 300 else None, "i8_2": f"ihp_meas_{t}K", "dev_pc": None}
      for t in (233, 300, 343, 398)],
    *[{"name": f"tudelft_{t}K", "test": f"TU Delft TEST {'4.13' if t == 4 else '295.0'} K",
       "split": "I6_1/split.json", "arm": "selected g v2 (I18-2, state C)", "part": f"tud{t}_v2",
       "v1_part": f"tud{t}_v1", "i8_2": f"tudelft_{t}K", "dev_pc": None} for t in (4, 295)],
]


def word(apply: bool) -> str:
    return "apply" if apply else "C0_refit"


def stage_testgate() -> dict:
    i82 = {p["name"]: p for p in jload(EXP / "I8_2/prereg_test_predictions.json")["predictions"]}
    rows, disagree = [], []
    for s in TEST_SPLITS:
        sp = jload(EXP / s["split"])
        f = jload(part_file(s["part"], False))
        if f["smoke"] or f["n_inner_fits"] != f["n_planned"]:
            raise RuntimeError(f"{s['part']}: incomplete S3_D")
        if not f["dev_check"]["ok"]:
            raise RuntimeError(f"{s['part']}: TEST tags in loaded records")
        row = {"name": s["name"], "test_split": s["test"], "split_file": f"out/research/explore/{s['split']}",
               "n_test": len(sp["TEST"]), "n_dev": len(sp["DEV"]), "n_dev_records_used": f["n_records"],
               "arm": s["arm"], "S3_D": f["S3_D"], "decision": word(m.apply_gate(f["S3_D"])),
               "decision_sensitivity_0.02455": word(m.apply_gate(f["S3_D"], m.SENS_THRESHOLD)),
               "s3_file": rel(part_file(s["part"], False)), "dev_check": f["dev_check"]}
        if s["v1_part"]:
            f1 = jload(part_file(s["v1_part"], False))
            if not f1["dev_check"]["ok"]:
                raise RuntimeError(f"{s['v1_part']}: TEST tags in loaded records")
            row["secondary_v1"] = {"arm": "M_bias v1", "S3_D": f1["S3_D"], "decision": word(m.apply_gate(f1["S3_D"])),
                                   "s3_file": rel(part_file(s["v1_part"], False))}
        if s["i8_2"]:
            p = i82[s["i8_2"]]
            row["I8_2"] = {"S3_stored": p["S3_stored"], "S3_predicts": p["S3_predicts"],
                           "S1_predicts": p["S1_predicts"], "arm": "M_bias v1"}
            for lab, dec in (("primary", row["decision"]), ("secondary_v1", row.get("secondary_v1", {}).get("decision"))):
                if dec is None:
                    continue
                for key in ("S3_predicts", "S1_predicts"):
                    want = "apply" if p[key] == "gain" else "C0_refit"
                    if dec != want:
                        disagree.append({"name": s["name"], "I19_1": lab, "I19_1_decision": dec,
                                         "I8_2": key, "I8_2_prediction": p[key]})
        if s["dev_pc"]:
            d = jload(EXP / s["dev_pc"])
            row["dev_P_c"] = {"file": f"out/research/explore/{s['dev_pc']}", "S3_fold_mean": d["dataset"]["S3"],
                              "predicted_sign": d["P_c"]["predicted_sign"],
                              "note": "DEV prediction (no committed TEST prediction for this split)"}
        rows.append(row)
    rc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    body = {"id": "I19_1_test_gate_decisions", "prereg": rel(PREREG), "written": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_head_at_run": rc.stdout.strip(), "threshold": m.THRESHOLD,
            "rule": "apply the correction to every TEST device of the split iff the DEV S3_D < 0 (strict); else "
                    "C0_refit for every TEST device",
            "test_data_not_opened": "only DEV records loaded; every loaded tag is in the DEV list and not in the "
                                    "TEST list of the split file (dev_check per row)",
            "sky130_77K": "no locked TEST since 2026-10-21 (user decision); no Part P row",
            "evaluation_at_TEST": "decision correct iff it equals the sign of the TEST ln GMR(correction, C0_refit)",
            "decisions": rows, "disagreements_with_I8_2": disagree}
    jdump(body, OUT / "test_gate_decisions.json")
    write_test_md(body)
    print(json.dumps([{k: r[k] for k in ("name", "S3_D", "decision")} for r in rows], indent=1, default=float))
    print(f"disagreements with I8-2: {len(disagree)}")
    return body


def write_test_md(b: dict) -> None:
    L = ["# I19-1 Part P: TEST gate decisions (committed before any TEST evaluation)", "",
         f"Written {b['written']}; git HEAD at run `{b['git_head_at_run']}`. Rule: {b['rule']}. Threshold "
         f"{b['threshold']}. {b['test_data_not_opened']}. sky130 77 K: {b['sky130_77K']}.", "",
         "| TEST split | n TEST | n DEV records | arm | S3_D (DEV) | decision | decision at +0.02455 | v1 S3_D | v1 decision "
         "| I8-2 S3 sign | I8-2 S1 rule |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in b["decisions"]:
        v1 = r.get("secondary_v1")
        i8 = r.get("I8_2")
        v1s = f"{v1['S3_D']:+.4f}" if v1 else "-"
        L.append(f"| {r['test_split']} | {r['n_test']} | {r['n_dev_records_used']} | {r['arm']} | {r['S3_D']:+.4f} | "
                 f"{r['decision']} | {r['decision_sensitivity_0.02455']} | "
                 f"{v1s} | {v1['decision'] if v1 else '-'} | "
                 f"{i8['S3_predicts'] if i8 else '-'} | {i8['S1_predicts'] if i8 else '-'} |")
    L += ["", "NIST rows: no I8-2 TEST prediction; committed DEV P_c (sign of the fold-mean S3):"]
    for r in b["decisions"]:
        if "dev_P_c" in r:
            L.append(f"- {r['name']}: S3 fold mean {r['dev_P_c']['S3_fold_mean']:+.4f}, P_c "
                     f"{r['dev_P_c']['predicted_sign']} ({r['dev_P_c']['file']})")
    L += ["", f"Disagreements with the I8-2 committed TEST predictions ({len(b['disagreements_with_I8_2'])}; "
          "I8-2 predicts the sign for M_bias v1):", ""]
    for d in b["disagreements_with_I8_2"] or [{"name": "none"}]:
        L.append(f"- {d['name']}" + (f": I19-1 {d['I19_1']} {d['I19_1_decision']} vs I8-2 {d['I8_2']} "
                                     f"{d['I8_2_prediction']}" if "I8_2" in d else ""))
    L += ["", f"Evaluation at TEST: {b['evaluation_at_TEST']}."]
    (OUT / "test_gate_decisions.md").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# decide (Part R, Part C, validity)
# ---------------------------------------------------------------------------
def run_unit_tests() -> dict:
    r = subprocess.run([sys.executable, "-m", "unittest", *UNIT_TESTS], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"})
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    return {"files": list(UNIT_TESTS), "rc": r.returncode, "passed": r.returncode == 0, "tail": tail}


def evaluate_dataset(d: dict, s3: float, i18: dict | None) -> dict:
    from cryoml.explore.i3_2_bias import paired_ratio
    apply = m.apply_gate(s3)
    rows = m.gated_rows(d["corr"], d["c0"], apply)
    cl = d["clusters"]
    g = paired_ratio(rows, d["c0"], clusters=cl)
    c = paired_ratio(d["corr"], d["c0"], clusters=cl)
    y = math.log(c["ratio"])
    out = {"S3_D": s3, "decision": word(apply), "apply": apply,
           "decision_sensitivity_0.02455": word(m.apply_gate(s3, m.SENS_THRESHOLD)),
           "n_rows": len(rows),
           "gmr_gated_D_vs_C0_refit": g, "gmr_correction_vs_C0_refit": c,
           "realised_ln_gmr_correction": y, "decision_matches_realised_sign": apply == (y < 0),
           "oracle_post_hoc": min(1.0, c["ratio"])}
    if i18 is not None:
        out["per_fold_gate_I18_1"] = {"ratio": i18["gmr_gated_vs_C0_refit"]["ratio"],
                                      "ci_low": i18["gmr_gated_vs_C0_refit"]["ci_low"],
                                      "ci_high": i18["gmr_gated_vs_C0_refit"]["ci_high"],
                                      "folds_apply": f"{i18['folds_apply']}/{i18['n_folds']}"}
        out["I8_2_gate_G"] = {"S3_fold_mean": i18["dataset_gate"]["S3_D"],
                              "decision": word(i18["dataset_gate"]["apply"])}
    return out


def stage_decide() -> dict:
    from cryoml.explore.i7_1_transfer import ols, spearman
    i18 = jload(EXP / "I18_1/decision.json")
    ds = g18.datasets(fallback_tud=False)
    res, v1rows, v1diff = {}, {}, 0.0
    for name in m.PRIMARY:
        d = ds[name]
        f = jload(part_file(PRIMARY_PART[name], False))
        res[name] = {"arm": d["arm"], "s3_part": PRIMARY_PART[name], "unit": d["unit"] or "device",
                     **evaluate_dataset(d, f["S3_D"], i18["datasets"][name])}
        if "S3_D_by_class" in f:
            res[name]["S3_D_by_class"] = f["S3_D_by_class"]
        # V1: rows equal the I18-1 rows (recompute I18-1 gated rows from its decisions)
        st = i18["datasets"][name]
        rec = {t: (d["corr"][t] if st["decisions"][d["fold_of"](t)] else d["c0"][t]) for t in st["gated_rows"]}
        v1diff = max(v1diff, max(abs(rec[t] - st["gated_rows"][t]) for t in rec))
        v1rows[name] = m18.n_finite_pairs(d["corr"], d["c0"])
    rule_r = m.part_r_rule({k: v["gmr_gated_D_vs_C0_refit"]["ratio"] for k, v in res.items()},
                           {k: v["gmr_correction_vs_C0_refit"]["ratio"] for k, v in res.items()})
    no_tud = [k for k in res if not k.startswith("tudelft")]
    pooled = {"gated_D": rule_r["pooled_gated_D"], "correction_always": rule_r["pooled_correction_always"],
              "per_fold_gate_I18_1": m.pooled([v["per_fold_gate_I18_1"]["ratio"] for v in res.values()]),
              "oracle_post_hoc": m.pooled([v["oracle_post_hoc"] for v in res.values()]),
              "I8_2_gate_G": m.pooled([(v["gmr_correction_vs_C0_refit"]["ratio"]
                                        if v["I8_2_gate_G"]["decision"] == "apply" else 1.0) for v in res.values()]),
              "without_tudelft": {"gated_D": m.pooled([res[k]["gmr_gated_D_vs_C0_refit"]["ratio"] for k in no_tud]),
                                  "correction_always": m.pooled([res[k]["gmr_correction_vs_C0_refit"]["ratio"]
                                                                 for k in no_tud]),
                                  "per_fold_gate_I18_1": m.pooled([res[k]["per_fold_gate_I18_1"]["ratio"]
                                                                   for k in no_tud])}}
    sign_hits = int(sum(v["decision_matches_realised_sign"] for v in res.values()))
    # secondary v1
    sec = {}
    for name, d in g18.secondary_v1().items():
        p = part_file(V1_PART[name], False)
        if p.exists():
            sec[name] = {k: v for k, v in evaluate_dataset(d, jload(p)["S3_D"], None).items()}
    # Part C
    i7 = jload(EXP / "I7_1/decision.json")["datasets"]
    sim = {}
    for n in sim_names():
        f = jload(part_file(f"sim:{n}", False))
        y = i7[n]["y"]
        ap = m.apply_gate(f["S3_D"])
        sim[n] = {"S3_D": f["S3_D"], "S3_fold_mean_I7_1": i7[n].get("S3"), "y": y, "decision": word(ap),
                  "correct": m.sign_correct(f["S3_D"], y), "gmr_gated_D": math.exp(y) if ap else 1.0,
                  "gmr_M_bias_always": math.exp(y), "n_inner_fits": f["n_inner_fits"],
                  "n_planned": f["n_planned"], "n_finite": f["n_finite"], "n_records": f["n_records"]}
    rule_c = m.part_c_rule({k: v["S3_D"] for k, v in sim.items()}, {k: v["y"] for k, v in sim.items()})
    xs, ys = [v["S3_D"] for v in sim.values()], [v["y"] for v in sim.values()]
    sim_summary = {"spearman_S3_D_y": spearman(xs, ys), "ols_y_on_S3_D": ols(xs, ys),
                   "spearman_S3_D_vs_S3_fold_mean": spearman(xs, [v["S3_fold_mean_I7_1"] for v in sim.values()]),
                   "pooled_gated_D": m.pooled([v["gmr_gated_D"] for v in sim.values()]),
                   "pooled_M_bias_always": m.pooled([v["gmr_M_bias_always"] for v in sim.values()]),
                   "pooled_oracle": m.pooled([min(1.0, v["gmr_M_bias_always"]) for v in sim.values()]),
                   "n_gated_above_1.02": int(sum(v["gmr_gated_D"] > 1.02 for v in sim.values())),
                   "sensitivity_correct": int(sum(m.apply_gate(v["S3_D"], m.SENS_THRESHOLD) == (v["y"] < 0)
                                                  for v in sim.values()))}
    # validity
    v2 = []
    chk = jload(part_file("check_ihp_v1", False))["S3_fold"]
    ref = g18.stored_s3_i7("ihp_meas_300K")
    v2 += [abs(chk[f] - ref[f]) for f in chk]
    chk = jload(part_file("check_sky_v2", False))["S3_fold"]
    ref = jload(EXP / "I18_1/s3/sky130_v2.json")["S3"]
    v2 += [abs(chk[f] - ref[f]) for f in chk]
    v3, v5 = True, True
    for p in list(REAL_PARTS) + [f"sim:{n}" for n in sim_names()]:
        f = jload(part_file(p, False))
        v3 = v3 and (not f["smoke"]) and f["n_inner_fits"] == f["n_planned"] == f["n_groups"] \
            and bool(np.isfinite(f["S3_D"]))
        if f.get("dev_check") is not None:
            v5 = v5 and f["dev_check"]["ok"]
    unit = run_unit_tests()
    validity = m.decide_validity({"v1_rows": v1rows, "v1_rows_max_diff": v1diff, "v2_max_diff": max(v2),
                                  "v2_n": len(v2), "v3_complete": v3, "v4_unit_tests_pass": unit["passed"],
                                  "v5_test_closed": v5})
    outcome = m.overall(validity["valid"], rule_r["supported"], rule_c["holds"])
    out = {"prereg": rel(PREREG), "validity": validity, "v2_values": v2, "unit_tests": unit, "outcome": outcome,
           "part_R": {"label": "retrospective", "rule": rule_r, "pooled": pooled,
                      "decision_matches_realised_sign": f"{sign_hits}/{len(res)}", "datasets": res},
           "part_R_secondary_v1": {"datasets": sec,
                                   "pooled_gated_D": m.pooled([v["gmr_gated_D_vs_C0_refit"]["ratio"]
                                                               for v in sec.values()]),
                                   "pooled_correction_always": m.pooled([v["gmr_correction_vs_C0_refit"]["ratio"]
                                                                         for v in sec.values()])},
           "part_C": {"rule": rule_c, "summary": sim_summary, "datasets": sim},
           "part_P": "test_gate_decisions.json (committed separately)"}
    jdump(out, OUT / "decision.json")
    print(json.dumps({"validity": validity, "outcome": outcome, "part_R": rule_r, "pooled": pooled,
                      "sign": f"{sign_hits}/{len(res)}", "part_C": rule_c,
                      "per_dataset": {k: [round(v["S3_D"], 4), v["decision"],
                                          round(v["gmr_gated_D_vs_C0_refit"]["ratio"], 4)] for k, v in res.items()}},
                     indent=1, default=float))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("s3", "testgate", "decide"))
    ap.add_argument("--parts", default="real")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true", help="2 inner fits per part, 50 steps, 1 simulated dataset")
    args = ap.parse_args()
    t0 = time.time()
    if args.stage == "s3":
        stage_s3(expand_parts(args.parts), args.workers, args.smoke)
    elif args.stage == "testgate":
        stage_testgate()
    else:
        stage_decide()
    print(f"stage {args.stage} done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
