#!/usr/bin/env python3
"""I15-1: re-registration of the I14-1 Part B deployment test of g v2 with
the Q1 grid swept in serpentine order (``cryoml.explore.i15_1_serp``); I14-1
replica, module, references and thresholds unchanged.  sky130 77 K, 18
devices, I13-1 state-C inputs.

Pre-registration: out/research/explore/I15_1/prereg.md (prereg.json).

The stages reuse the I14-1 driver (scripts/research/explore/i14_1_chan.py)
with its output directory set to I15_1 (rebinding at import, so spawned
workers see it too); ``q12_task`` is the I14-1 one with the Q1 grid decks
swept in serpentine order.

Stages (``--stage``):

  precheck  (before registration) nested (I14-1) vs serpentine Q1 grid decks
            on the published, start (state C), I13-1 v2 held-out and I13-1 v1
            held-out cards, and on the I14-1 M2 card of nmos_L8_W1p6 (the
            I14-1 failure; disclosed): card / card_rep / elem (full-data g v2)
            x forward / exchanged x default / r4 / tight; messages and the
            serpentine - nested differences
  validity  V1 card-only library deck reproduces the I13-1 rows; V2 unit tests
  accuracy  element-deck RRMS of the held-out pairs (v2 and v1)
  deploy    --arm v2|v1: compile, M2 refit, q12, q3, report
  restrict  restriction items of the v2 arm: default / r4 / tight Q1
            differences per device, the gds curves with an introduced sign
            change, runtime decomposition on ro5_A (rusage)
  decide    validity + I13-1 accuracy flags + Part B (v2) -> decision.json

  PYTHONPATH=src .venv/bin/python scripts/research/explore/i15_1_chan.py --stage precheck --workers 6
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(ROOT / "scripts" / "research" / "explore"))

import i14_1_chan as drv14  # noqa: E402  (sets the I13-1 state-C environment)
from cryoml.explore import i11_1_deploy as m1  # noqa: E402
from cryoml.explore import i12_1_symm as v2  # noqa: E402
from cryoml.explore import i14_1_chan as c14  # noqa: E402
from cryoml.explore import i15_1_serp as m  # noqa: E402

OUT = drv14.EXP / "I15_1"
PREREG = OUT / "prereg.json"
I14 = drv14.EXP / "I14_1"
# rebinding: every I14-1 driver stage writes under I15_1
drv14.OUT = OUT
drv14.PREREG = PREREG
drv14.UNIT_TESTS = drv14.UNIT_TESTS + ("tests.test_i15_1_serp",)

jload, jdump, devices = drv14.jload, drv14.jdump, drv14.devices
SMOKE_TAGS = drv14.SMOKE_TAGS
TOLS = (("default", ""), ("r4", m1.R4), ("tight", m1.TIGHT))


# ---------------------------------------------------------------------------
# Q1 grid decks (nested = I14-1, serp = I15-1)
# ---------------------------------------------------------------------------
def run_grid(td: Path, lib: Path, dev, mode: str, swap: bool, opt: str, osdi, order: str,
             label: str, module: str = v2.VA_MODULE_V2, repeats: int = 1) -> dict:
    """One Q1 grid deck; returns the current grid (and the sensed grid for
    card_rep), messages, row count and wall times."""
    f = td / f"{label}.dat"
    sense = mode == "card_rep"
    ctrl = (m.grid_control_serp(dev, f, swap, sense) if order == "serp"
            else m.grid_control(dev, f, swap, sense))
    walls, r = [], None
    for _ in range(repeats):
        if f.exists():
            f.unlink()
        r = m1.run_deck(m.deck(label, lib, m.device_body(dev, mode, swap=swap, module=module), ctrl,
                               osdi if mode == "elem" else None, module, opt),
                        td, f"{label}.sp", timeout_s=m1.TIMEOUT_DEVICE_S)
        walls.append(r["wall_s"])
    rows = m1.read_wrdata(f, 4 if sense else 2)
    n_raw = int(rows.shape[0])
    if order == "serp":
        rows = m.grid_rows(rows)
    out = {"I": m1.grid_from_rows(rows[:, :2]) if rows.shape[0] else
           np.full((len(m1.GRID_VGS), len(m1.GRID_VDS)), np.nan),
           "msgs": r["msgs"], "n_rows": n_raw, "wall": walls}
    if sense:
        out["rep"] = (m1.grid_from_rows(rows[:, 2:4]) if rows.shape[0] else
                      np.full((len(m1.GRID_VGS), len(m1.GRID_VDS)), np.nan))
    return out


# ---------------------------------------------------------------------------
# precheck (pre-registration): nested vs serpentine Q1 grid decks
# ---------------------------------------------------------------------------
def precheck_task(task):
    tag, label, params, osdi_s = task
    dev = devices()[tag]
    out = {"tag": tag, "label": label, "msgs": {}, "n_rows": {}, "wall": {}}
    arr = {}
    with m1.tempdir("cryoml_i15p_") as td_s:
        td = Path(td_s)
        lib = m.patched_library(td, {dev.dev_type: [(dev.bin, params)]})
        for order in ("nested", "serp"):
            for swap in (False, True):
                for tol, opt in TOLS:
                    for mode in ("card", "card_rep", "elem"):
                        lab = f"{order}_{'swap' if swap else 'fwd'}_{tol}_{mode}"
                        r = run_grid(td, lib, dev, mode, swap, opt, Path(osdi_s), order, lab)
                        out["msgs"][lab] = r["msgs"]
                        out["n_rows"][lab] = r["n_rows"]
                        out["wall"][lab] = r["wall"][0]
                        arr[lab] = r["I"]
                        if "rep" in r:
                            arr[lab + "_rep"] = r["rep"]
    out["arr"] = arr
    return out


def _cmp(a, b) -> dict:
    """a (serpentine) vs b (nested): max |d| / (1e-6 |b| + 1e-13 A), max abs
    and max relative difference at |b| > 1 nA, non-finite count."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    nbad = int(np.sum(~ok))
    if not ok.any():
        return {"eq_max_ratio": float("inf"), "max_abs": float("inf"), "max_rel_1nA": float("inf"),
                "n_nonfinite": nbad}
    d = np.abs(a - b)[ok]
    bb = np.abs(b[ok])
    big = bb > m1.I_FLOOR
    return {"eq_max_ratio": float(np.max(d / (v2.EQ_REL * bb + v2.EQ_ABS))), "max_abs": float(np.max(d)),
            "max_rel_1nA": float(np.max(d[big] / bb[big])) if big.any() else 0.0, "n_nonfinite": nbad}


def stage_precheck(workers: int, tags, rd: Path) -> dict:
    sc = drv14.start_cards()
    sets = {"published": {t: None for t in tags}, "start_C": {t: sc[t] for t in tags},
            "v2_heldout_C": {t: drv14.heldout_card("v2", t) for t in tags},
            "v1_heldout_C": {t: drv14.heldout_card("v1", t) for t in tags}}
    osdi = rd / "osdi" / "v2_full.osdi"
    c = m1.compile_osdi(m.to_verilog_a(drv14.corr_full("v2")), osdi.with_suffix(".va"), osdi)
    tasks = [(t, lab, cards[t], str(osdi)) for lab, cards in sets.items() for t in tags]
    m2_i14 = jload(I14 / "deploy_v2" / "cards_M2.json")
    if "nmos_L8_W1p6" in tags:      # the I14-1 failure (disclosed; the other M2 cards are not run)
        tasks.append(("nmos_L8_W1p6", "M2_I14_disclosed", m2_i14["nmos_L8_W1p6"], str(osdi)))
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(precheck_task, tasks, chunksize=1)
    res = {"wall_s": round(time.time() - t0, 1), "binary": m1.NG_OSDI, "osdi_sha256": c["sha256"],
           "replica": m.REPLICA_CHAN, "n_tasks": len(rows), "messages": {}, "per_device": {}}
    n_rows_expected = len(m1.GRID_VGS) * len(m1.GRID_VDS)
    for r in rows:
        for k, v in r["msgs"].items():
            if v:
                res["messages"][f"{r['label']}|{r['tag']}|{k}"] = v
        for k, v in r["n_rows"].items():
            if v != n_rows_expected:
                res["messages"].setdefault(f"{r['label']}|{r['tag']}|{k}", []).append(f"rows {v}")
        a = r["arr"]
        d = {}
        for sw in ("fwd", "swap"):
            for tol, _ in TOLS:
                for mode in ("card", "card_rep", "elem"):
                    k = f"{sw}_{tol}_{mode}"
                    d[k] = _cmp(a[f"serp_{k}"], a[f"nested_{k}"])
                d[f"{sw}_{tol}_sense"] = _cmp(a[f"serp_{sw}_{tol}_card_rep_rep"], a[f"nested_{sw}_{tol}_card_rep_rep"])
        d["wall_ratio_serp_nested_default_fwd_card"] = (r["wall"]["serp_fwd_default_card"]
                                                       / r["wall"]["nested_fwd_default_card"])
        res["per_device"][f"{r['label']}|{r['tag']}"] = d
    main_keys = [k for k in res["per_device"] if not k.startswith("M2_I14_disclosed")]
    summ = {"n_decks_per_order": sum(len(r["msgs"]) for r in rows if r["label"] != "M2_I14_disclosed") // 2,
            "messages_nested": sorted(k for k in res["messages"] if "|nested_" in k
                                      and not k.startswith("M2_I14_disclosed")),
            "messages_serp": sorted(k for k in res["messages"] if "|serp_" in k
                                    and not k.startswith("M2_I14_disclosed")),
            "disclosed_M2_nmos_L8_W1p6": {k: v for k, v in res["messages"].items()
                                          if k.startswith("M2_I14_disclosed")}}
    for sw in ("fwd", "swap"):
        for tol, _ in TOLS:
            for q in ("card", "card_rep", "elem", "sense"):
                k = f"{sw}_{tol}_{q}"
                summ[f"{k}_eq_max_ratio"] = max(res["per_device"][x][k]["eq_max_ratio"] for x in main_keys)
                summ[f"{k}_max_rel_1nA"] = max(res["per_device"][x][k]["max_rel_1nA"] for x in main_keys)
    ws = [res["per_device"][x]["wall_ratio_serp_nested_default_fwd_card"] for x in main_keys]
    summ["wall_ratio_serp_nested_default_fwd_card_median"] = float(np.median(ws))
    res["summary"] = summ
    jdump(res, rd / "precheck.json")
    print(json.dumps(summ, indent=1, default=float), flush=True)
    return res


# ---------------------------------------------------------------------------
# q12 (I14-1 q12_task with the Q1 grid decks in serpentine order)
# ---------------------------------------------------------------------------
def q12_task(task):
    tag, params, rd_s, arm = task
    rd = Path(rd_s)
    dev = devices()[tag]
    corr = drv14.corr_full(arm)
    module = m.module_name(corr)
    osdi = drv14.arm_osdi(rd, arm)
    res = {"tag": tag, "msgs": {}, "wall": {}, "n_rows": {}}
    a = {}
    with m1.tempdir("cryoml_i15q_") as td_s:
        td = Path(td_s)
        lib = m.patched_library(td, {dev.dev_type: [(dev.bin, params)]})

        def grid(label, mode, swap, opt, repeats=1):
            r = run_grid(td, lib, dev, mode, swap, opt, osdi, "serp", label, module, repeats)
            res["msgs"][label] = r["msgs"]
            res["wall"][label] = r["wall"]
            res["n_rows"][label] = r["n_rows"]
            a[label] = r["I"]
            if "rep" in r:
                a[label + "_rep"] = r["rep"]

        def go(label, body, control, el, options="", repeats=1):
            walls, r = [], None
            for _ in range(repeats):
                r = m1.run_deck(m.deck(label, lib, body, control, osdi if el else None, module, options),
                                td, f"{label}.sp", timeout_s=m1.TIMEOUT_DEVICE_S)
                walls.append(r["wall_s"])
            res["msgs"][label] = r["msgs"]
            res["wall"][label] = walls
            return r

        # runtime reference: card only (no replica), default tolerance, 3 repeats
        grid("grid_fwd_default_card", "card", False, "", repeats=3)
        for mode in ("card_rep", "elem"):
            for tol, opt in TOLS:
                grid(f"grid_fwd_{tol}_{mode}", mode, False, opt,
                     repeats=3 if (tol == "default" and mode == "elem") else 1)
            grid(f"grid_swap_tight_{mode}", mode, True, m1.TIGHT)
        pts = m1.op_points(dev)
        for mode in ("card_rep", "elem"):
            for ex in ("VG", "VD"):
                lab = f"ac_{ex}_{mode}"
                r = go(lab, m.device_body(dev, mode, module=module),
                       m.ac_control(pts, ex, mode == "card_rep"), mode == "elem", m1.TIGHT)
                for k, v in m.parse_ac(r["text"], len(pts)).items():
                    a[f"{lab}_{k}"] = v
        for mode in ("card", "elem"):
            lab = f"dense_{mode}"
            pre = td / lab
            go(lab, m.device_body(dev, mode, module=module), drv14.drv12.dense_control(dev, pre),
               mode == "elem", m1.TIGHT)
            for k in range(len(m1.GM_VDS)):
                a[f"{lab}_gm{k}"] = m1.read_wrdata(Path(f"{pre}_gm{k}.dat"))
            for k in range(len(m1.GDS_VGS)):
                a[f"{lab}_gds{k}"] = m1.read_wrdata(Path(f"{pre}_gds{k}.dat"))
    n_exp = len(m1.GRID_VGS) * len(m1.GRID_VDS)
    for lab, n in res["n_rows"].items():       # a grid run with missing rows is a failed run
        if n != n_exp:
            res["msgs"][lab] = res["msgs"][lab] + [f"grid rows {n} of {n_exp}"]
    res.update(drv14.q12_metrics(dev, corr, pts, a))
    (rd / "arrays").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(rd / "arrays" / f"q12_{tag}.npz", pts=pts,
                        **{k: v for k, v in a.items() if isinstance(v, np.ndarray)})
    return res


def stage_q12(rd: Path, arm: str, tags, workers: int) -> dict:
    cards = jload(rd / "cards_M2.json")
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(q12_task, [(t, cards[t], str(rd), arm) for t in tags], chunksize=1)
    out = {"wall_s": round(time.time() - t0, 1), "grid_order": "serpentine (I15-1)",
           "devices": {r["tag"]: r for r in rows}}
    jdump(out, rd / "q12.json")
    print(f"q12 ({arm}): {len(rows)} devices, {out['wall_s']} s", flush=True)
    return out


# ---------------------------------------------------------------------------
# restrictions (v2 arm; reported, no rule)
# ---------------------------------------------------------------------------
def stage_restrict(rd: Path) -> dict:
    q12 = jload(rd / "q12.json")["devices"]
    corr = drv14.corr_full("v2")
    out = {"q1_max_rel_1nA": {}, "gds_curves": [], "gm_curves": []}
    for tag, r in sorted(q12.items()):
        out["q1_max_rel_1nA"][tag] = {t: r[f"q1_{t}_max_rel_1nA"] for t, _ in TOLS}
        dev = devices()[tag]
        pm = dev.dev_type == "pmos"
        z = np.load(rd / "arrays" / f"q12_{tag}.npz")
        for k, vgs in enumerate(m1.GDS_VGS):
            if not r["q2_gds_curves_introduced"][k]:
                continue
            dc, dk = z[f"dense_card_gds{k}"], z[f"dense_elem_gds{k}"]
            o = np.argsort(np.abs(dc[:, 0]))
            x, ic, ik = dc[o, 0], dc[o, 1], dk[o, 1]
            s = -1.0 if pm else 1.0
            g = corr.g_v(pm, np.full_like(x, s * vgs), x)
            # Python model of the card-only current x exp(g) (dense card sweep, no replica run)
            ipy = ic * np.exp(g)
            gc, gk, gp = m1.central_diff(x, ic), m1.central_diff(x, ik), m1.central_diff(x, ipy)
            sel = np.abs(x) >= 0.05          # dI/dV(d,s) > 0 in forward operation (both polarities)
            out["gds_curves"].append({
                "tag": tag, "vgs": float(s * vgs),
                "n_neg_elem": int(np.sum(gk[sel] < 0)), "n_neg_card": int(np.sum(gc[sel] < 0)),
                "n_neg_python_card_exp_g": int(np.sum(gp[sel] < 0)),
                "vds_neg_elem": [float(v) for v in x[sel][gk[sel] < 0]][:50],
                "min_gds_elem_rel": float(np.min((gk / np.maximum(np.abs(ik), 1e-30))[sel])),
                "g_range": [float(np.min(g)), float(np.max(g))]})
    out["n_gds_curves"] = len(out["gds_curves"])
    jdump(out, rd / "restrictions.json")
    print(json.dumps({"q1_default_max": max(v["default"] for v in out["q1_max_rel_1nA"].values()),
                      "gds": [(c["tag"], c["vgs"], c["n_neg_elem"], c["n_neg_python_card_exp_g"])
                              for c in out["gds_curves"]]}, indent=1), flush=True)
    return out


def runtime_task(task):
    """ro5_A, M2 cards: wall time (3 repeats) and ngspice rusage of card,
    card_rep and elem decks."""
    mode, rd_s, arm = task
    rd = Path(rd_s)
    devs = devices()
    m2 = jload(rd / "cards_M2.json")
    name, kind, tags, st = [c for c in drv14.CIRCUITS if c[0] == "ro5_A"][0]
    corr = drv14.corr_full(arm)
    module = m.module_name(corr)
    osdi = drv14.arm_osdi(rd, arm) if mode == "elem" else None
    with m1.tempdir("cryoml_i15t_") as td_s:
        td = Path(td_s)
        lib = m.patched_library(td, drv14.arm_cards("M2_card", tags, m2))
        f = td / "o.dat"
        body = m.ring_body(devs[tags[0]], devs[tags[1]], mode, module)
        ctrl = m1.ring_control(f, st["tstep"], st["tstop"]) + ["rusage all"]
        walls, r = [], None
        for _ in range(3):
            r = m1.run_deck(m.deck(name, lib, body, ctrl, osdi, module, numdgt=12), td, "c.sp",
                            timeout_s=m1.TIMEOUT_CIRCUIT_S)
            walls.append(r["wall_s"])
    stats = {}
    for key, pat in (("equations", r"Circuit Equations = (\S+)"), ("iterations", r"Transient iterations = (\S+)"),
                     ("timepoints", r"Transient timepoints = (\S+)"), ("load_s", r"Transient load time = (\S+)"),
                     ("analysis_s", r"Transient analysis time = (\S+)")):
        mm = re.search(pat, r["text"])
        stats[key] = float(mm.group(1)) if mm else None
    return {"mode": mode, "wall": walls, "msgs": r["msgs"], **stats}


def stage_runtime(rd: Path, arm: str) -> dict:
    rows = [runtime_task((mode, str(rd), arm)) for mode in ("card", "card_rep", "elem")]
    out = {"circuit": "ro5_A", "cards": "M2_card", "rows": {r["mode"]: r for r in rows}}
    med = {r["mode"]: float(np.median(r["wall"])) for r in rows}
    out["ratio_card_rep"] = med["card_rep"] / med["card"]
    out["ratio_elem"] = med["elem"] / med["card"]
    out["load_s"] = {r["mode"]: r["load_s"] for r in rows}
    jdump(out, rd / "runtime.json")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1), flush=True)
    return out


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------
def stage_decide() -> dict:
    val = jload(OUT / "validity" / "validity.json")
    acc_native = jload(OUT / "accuracy" / "accuracy.json")
    dep = jload(OUT / "deploy_v2" / "summary.json")
    dep_v1 = jload(OUT / "deploy_v1" / "summary.json")
    unit = jload(OUT / "validity" / "unit_tests.json")
    lodo = jload(drv14.LODO / "results.json")
    n_rows_min = min(len(acc_native["per_arm"][a]["native"]) for a in ("v2", "v1"))
    m2_failed = max(dep["refit_in_sample"]["n_failed"], dep_v1["refit_in_sample"]["n_failed"])
    validity = m.decide_validity({"v1_max_rrms_diff": val["v1_max_rrms_diff"],
                                  "v2_unit_tests_pass": bool(unit["passed"]),
                                  "v3_n_rows_min": n_rows_min, "v3_m2_failed": m2_failed})
    acc = v2.decide_accuracy(lodo["comparisons"]["A1_M_bias_v2_vs_C0_refit"],
                             lodo["comparisons"]["A2_M_bias_v2_vs_M_bias_v1"],
                             lodo["n_failed_M_bias_v2"])
    m2_new = jload(OUT / "deploy_v2" / "cards_M2.json")
    m2_old = jload(I14 / "deploy_v2" / "cards_M2.json")
    m2_diff = max(abs(float(m2_new[t][k]) - float(m2_old[t][k])) / max(abs(float(m2_old[t][k])), 1e-300)
                  for t in m2_old for k in m2_old[t])
    out = {"prereg": str(PREREG.relative_to(ROOT)), "validity": validity, "accuracy_I13_1": acc,
           "deploy_v2": dep["decision"], **m.decide_outcome(validity, acc, dep["decision"]),
           "A1": lodo["comparisons"]["A1_M_bias_v2_vs_C0_refit"],
           "A2": lodo["comparisons"]["A2_M_bias_v2_vs_M_bias_v1"],
           "reference_arm_v1_deploy": dep_v1["decision"],
           "m2_v2_cards_max_rel_diff_vs_I14_1": m2_diff,
           "element_deck_accuracy": {a: {k: v for k, v in acc_native["per_arm"][a].items()
                                         if k not in ("native", "python_rerun", "python_channel")}
                                     for a in ("v2", "v1")},
           "element_deck_gmr": acc_native.get("gmr_native")}
    jdump(out, OUT / "decision.json")
    print(json.dumps({k: out[k] for k in ("validity", "accuracy_I13_1", "deploy_v2", "outcome", "action",
                                          "reference_arm_v1_deploy", "m2_v2_cards_max_rel_diff_vs_I14_1")},
                     indent=1, default=float), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("precheck", "validity", "accuracy", "deploy", "compile", "refit", "q12", "q3",
                             "report", "restrict", "runtime", "decide"))
    ap.add_argument("--arm", default="v2", choices=("v2", "v1"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true",
                    help="2 devices, refit max_nfev 20 (code check; not used for decisions)")
    args = ap.parse_args()
    base = OUT / "smoke" if args.smoke else OUT
    tags = SMOKE_TAGS if args.smoke else sorted(devices())
    t0 = time.time()
    if args.stage == "precheck":
        stage_precheck(args.workers, tags, base / "precheck")
    elif args.stage == "validity":
        drv14.stage_validity(args.workers, base / "validity", tags)
    elif args.stage == "accuracy":
        drv14.stage_accuracy(args.workers, base / "accuracy", tags)
    elif args.stage == "decide":
        stage_decide()
    elif args.stage == "restrict":
        stage_restrict(base / "deploy_v2")
    elif args.stage == "runtime":
        stage_runtime(base / f"deploy_{args.arm}", args.arm)
    else:
        rd = base / f"deploy_{args.arm}"
        opts = dict(drv14.REFIT_OPTS, heldout_nfev=20) if args.smoke else dict(drv14.REFIT_OPTS)
        workers = min(args.workers, 4)
        stages = (("compile", "refit", "q12", "q3", "report") if args.stage == "deploy"
                  else (args.stage,))
        timing = {}
        for st in stages:
            t1 = time.time()
            if st == "compile":
                drv14.stage_compile(rd, args.arm)
            elif st == "refit":
                drv14.stage_refit(rd, args.arm, tags, workers, opts)
            elif st == "q12":
                stage_q12(rd, args.arm, tags, workers)
            elif st == "q3":
                if args.smoke:
                    have = jload(rd / "cards_M2.json")
                    sc = drv14.start_cards()
                    for _, _, ts, _ in drv14.CIRCUITS:
                        for t in ts:
                            have.setdefault(t, sc[t])
                    jdump(have, rd / "cards_M2.json")
                drv14.stage_q3(rd, args.arm, workers)
            else:
                drv14.stage_report(rd, args.arm)
            timing[st] = round(time.time() - t1, 1)
        tf = rd / "timing.json"
        old = jload(tf) if tf.exists() else {}
        old.update(timing)
        jdump(old, tf)
    print(f"stage {args.stage} done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
