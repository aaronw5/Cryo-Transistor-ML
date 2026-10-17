#!/usr/bin/env python3
"""Residual-curvature diagnostic for the remainder-weighted model (node n2b).

At a point ``z`` (local active coordinates) of a DEV target, with the
NGSpice residual ``r_f`` and its forward-difference Jacobian ``J``
(``hybrid_tr.NgspiceResiduals``, step ``--diff-step`` 1e-3):

* ``g(z) = J(z)^T r_f(z)`` over the data rows;
* ``H`` column ``j`` = ``(g(z + h e_j) - g(z - h e_j)) / (2h)``, ``h`` =
  ``--hess-step`` 5e-3 (one-sided inward at a box face);
* ``S_f = (H + H^T) / 2 - J^T J`` (the NGSpice second-order term
  ``sum_i r_f,i Hess r_f,i``);
* ``S_s,j = sum_{i in curve j} r_f,i(z) Hess r_s,i(z)`` from
  ``torch.func.hessian`` of a float64 copy of the round's surrogate
  (``SurrogateResiduals`` map in float64), ``S_s = sum_j S_s,j``;
* ``A = J^T J + w^2 I`` (``w`` the prior weight ``sqrt(reg / n_active)``).

Reported per point: ``kappa_GN = ||A^-1 S_f||_2``, ``kappa_S = ||(A +
S_s)^-1 (S_f - S_s)||_2``, ``lambda* = argmin_{[0,1]^n_curves} ||S_f -
sum_j lambda_j S_s,j||_F`` (``scipy.optimize.lsq_linear``), ``kappa_lambda*``
(``kappa_S`` with ``sum_j lambda*_j S_s,j``) and ``||S_f||_F / ||J^T
J||_F``; the matrices go to ``<tier>_<tag>_<point>.npz``.

Targets: six set_0 targets per tier (``TAGS``).  Points: ``start`` =
``hybrid_tr.setup``'s ``starts[0]``; ``end`` = the final card of
``--end-dir-a`` (tier a: ``out/research/dev/sketch_none_k8/B5000/set_0``) or
``--end-dir-c`` (tier c: n2a's floor reference, ``sketch_tr.py --model
none --k 8 --budget 20000``, ``out/research/dev_sky130c/floor_ref/
sketch_none_k8_B20000/set_0``).  Tier a targets
are ``out/research/sim_targets/set_0.npz``, tier c
``out/research/sim_targets_sky130c/set_0.npz`` (``CRYOML_TARGET_OVERRIDE``
per worker).  Cost: ``1 + n + 2 n (1 + n)`` NGSpice calls per point (about
3.0k-3.8k); one spawn-pool task per point (``maxtasksperchild=1``).

Registered prediction (reported, not an acceptance gate): at ``end``,
``kappa_GN`` and mean ``lambda*`` are larger on tier c than on tier a.

``--columns N`` computes only the first N Hessian columns (smoke tests; the
outputs are marked ``partial``).

  PYTHONPATH=src .venv/bin/python scripts/research/curvature_diag.py \\
      --tiers a,c --out out/research/dev/curvature_diag --workers 10
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

TAGS = ("nmos_L0p15_W1p6", "nmos_L1_W3", "nmos_L100_W100",
        "pmos_L0p35_W1p6", "pmos_L2_W5", "pmos_L8_W0p84")
TIERS = {
    "a": {"targets": "out/research/sim_targets/set_0.npz",
          "end_dir": "out/research/dev/sketch_none_k8/B5000/set_0"},
    "c": {"targets": "out/research/sim_targets_sky130c/set_0.npz",
          "end_dir": "out/research/dev_sky130c/floor_ref/sketch_none_k8_B20000/set_0"},
}


def end_local(directory: Path, tag: str) -> tuple[np.ndarray | None, str]:
    """Final card (full local vector) of ``tag`` in a run directory; the
    record's single method entry (any method key)."""
    path = directory / f"ml_{tag}.json"
    if not path.exists():
        return None, f"missing {path}"
    methods = json.loads(path.read_text())["methods"]
    if len(methods) != 1:
        return None, f"{path}: {len(methods)} method entries"
    key, record = next(iter(methods.items()))
    return np.asarray(record["local"], dtype=np.float64), f"{path}:{key}"


def surrogate_float64(surrogate):
    """``SurrogateResiduals.value`` as a float64 torch function of z."""
    import torch
    from cryoml.foundation import I_REF
    model = copy.deepcopy(surrogate.model).double()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    base = surrogate.base.double()
    select = surrogate.select.double()
    index = surrogate.index
    mean = torch.tensor(surrogate.mean, dtype=torch.float64)
    std = torch.tensor(surrogate.std, dtype=torch.float64)
    measured = torch.tensor(surrogate.measured, dtype=torch.float64)
    scale = torch.tensor(surrogate.scale, dtype=torch.float64)

    def residual(z):
        slog = model(base + z @ select)[index] * std + mean
        current = torch.sign(slog) * I_REF * torch.expm1(torch.abs(slog))
        return (current - measured) / scale

    return residual


def curve_weight_matrix(r_f: np.ndarray, curve_starts, n_rows: int) -> np.ndarray:
    starts = [int(s) for s in curve_starts]
    stops = starts[1:] + [n_rows]
    W = np.zeros((len(starts), n_rows))
    for j, (a, b) in enumerate(zip(starts, stops)):
        W[j, a:b] = r_f[a:b]
    return W


def spectral(M: np.ndarray) -> float:
    if not np.all(np.isfinite(M)):
        return math.nan
    return float(np.linalg.norm(M, 2))


def solve_norm(A: np.ndarray, B: np.ndarray) -> float:
    """``||A^-1 B||_2`` (least-squares solve if ``A`` is singular)."""
    try:
        X = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        X = np.linalg.lstsq(A, B, rcond=None)[0]
    return spectral(X)


def curvature_metrics(S_f: np.ndarray, S_curves: np.ndarray, JtJ: np.ndarray,
                      w: float) -> dict:
    """The reported quantities from ``S_f``, ``S_s,j`` (stacked) and
    ``J^T J``."""
    from scipy.optimize import lsq_linear
    n = len(S_f)
    A = JtJ + w ** 2 * np.eye(n)
    S_s = S_curves.sum(axis=0)
    columns = S_curves.reshape(len(S_curves), -1).T
    fit = lsq_linear(columns, S_f.ravel(), bounds=(0.0, 1.0))
    lam = np.asarray(fit.x)
    S_lam = np.tensordot(lam, S_curves, axes=1)
    jtj = float(np.linalg.norm(JtJ))
    return {
        "kappa_GN": solve_norm(A, S_f),
        "kappa_S": solve_norm(A + S_s, S_f - S_s),
        "lambda_star": lam.tolist(),
        "lambda_star_mean": float(np.mean(lam)),
        "kappa_lambda_star": solve_norm(A + S_lam, S_f - S_lam),
        "Sf_over_JtJ_fro": float(np.linalg.norm(S_f)) / jtj if jtj > 0 else math.inf,
        "Sf_fro": float(np.linalg.norm(S_f)),
        "Ss_fro": float(np.linalg.norm(S_s)),
        "Sf_minus_Ss_fro": float(np.linalg.norm(S_f - S_s)),
        "Sf_minus_Slam_fro": float(np.linalg.norm(S_f - S_lam)),
        "JtJ_fro": jtj,
    }


def diagnose(job) -> dict:
    tier, tag, point, options = job
    import torch
    from cryoml.data_io import TARGET_OVERRIDE_ENV
    import hybrid_tr
    t0 = time.time()
    out_dir = Path(options["out"])
    name = f"{tier}_{tag}_{point}"
    os.environ[TARGET_OVERRIDE_ENV] = str(ROOT / options["targets"][tier])
    setup_options = {"round_dir": options["round_dir"], "bank": options["bank"],
                     "model": "surrogate", "n_starts": 4,
                     "diff_step": options["diff_step"],
                     "timeout_s": options["timeout_s"]}
    context, ngspice, starts, oracle, surrogate_for = hybrid_tr.setup(
        tag, setup_options)
    active = ngspice.active
    if point == "start":
        local, source = np.asarray(starts[0], dtype=np.float64), "starts[0]"
    else:
        local, source = end_local(ROOT / options["end_dirs"][tier], tag)
        if local is None:
            row = {"tier": tier, "tag": tag, "point": point, "error": source}
            (out_dir / f"{name}.json").write_text(json.dumps(row, indent=1) + "\n")
            print(name, "skipped:", source, flush=True)
            return row
    z = np.clip(local[active], -1.0, 1.0)
    n = len(z)
    w = hybrid_tr.prior_weight({"regularization": options["regularization"]}, n)
    r = ngspice.residual(z)
    J = ngspice.jacobian(z, r)
    g0 = J.T @ r
    h = options["hess_step"]
    n_cols = n if options["columns"] is None else min(n, options["columns"])
    H = np.zeros((n, n))
    sides = []
    for j in range(n_cols):
        e = np.zeros(n)
        e[j] = h
        if z[j] + h > 1.0:
            zm = z - e
            rm = ngspice.residual(zm)
            H[:, j] = (g0 - ngspice.jacobian(zm, rm).T @ rm) / h
            sides.append("backward")
        elif z[j] - h < -1.0:
            zp = z + e
            rp = ngspice.residual(zp)
            H[:, j] = (ngspice.jacobian(zp, rp).T @ rp - g0) / h
            sides.append("forward")
        else:
            zp, zm = z + e, z - e
            rp, rm = ngspice.residual(zp), ngspice.residual(zm)
            H[:, j] = ((ngspice.jacobian(zp, rp).T @ rp
                        - ngspice.jacobian(zm, rm).T @ rm) / (2.0 * h))
            sides.append("central")
    JtJ = J.T @ J
    S_f = 0.5 * (H + H.T) - JtJ
    # Surrogate curvature, float64, weighted per curve by the NGSpice residual.
    torch.set_num_threads(1)
    surrogate = surrogate_for(starts[0])
    residual64 = surrogate_float64(surrogate)
    W = torch.tensor(curve_weight_matrix(r, ngspice.curve_starts, len(r)))
    z_t = torch.tensor(z, dtype=torch.float64)
    with torch.no_grad():
        rs64 = residual64(z_t).numpy()
    rs32 = surrogate.value(z)
    S_curves = torch.func.hessian(lambda t: W @ residual64(t))(z_t).numpy()
    S_curves = 0.5 * (S_curves + np.transpose(S_curves, (0, 2, 1)))
    metrics = curvature_metrics(S_f, S_curves, JtJ, w)
    rrms = float(np.mean(np.sqrt(len(ngspice.layout) * np.add.reduceat(
        r ** 2, ngspice.curve_starts))))
    row = {
        "tier": tier, "tag": tag, "point": point, "source": source,
        "partial": n_cols < n, "n_columns": n_cols, "n_active": n,
        "n_curves": len(ngspice.layout),
        "curve_tags": [e["tag"] for e in ngspice.layout],
        "objective_rrms_at_z": rrms, "F": 0.5 * float(r @ r + w ** 2 * z @ z),
        "prior_weight": w, "hess_step": h, "diff_step": options["diff_step"],
        "hess_sides": {s: sides.count(s) for s in ("central", "forward", "backward")},
        "hessian_asymmetry_fro": float(np.linalg.norm(H - H.T) / max(
            np.linalg.norm(H), 1e-300)),
        "surrogate_float64_vs_float32_max_abs": float(np.max(np.abs(rs64 - rs32))),
        "surrogate_max_abs": float(np.max(np.abs(rs32))),
        **metrics,
        "n_ngspice_calls": ngspice.n_calls,
        "n_zero_jacobian_columns": ngspice.n_zero_columns,
        "n_cliff_retries": ngspice.n_cliff_retries,
        "n_failed": ngspice.evaluator.n_failed,
        "oracle_lhc": oracle, "runtime_s": round(time.time() - t0, 1),
    }
    np.savez(out_dir / f"{name}.npz", S_f=S_f, S_curves=S_curves, JtJ=JtJ,
             H=H, z=z, r=r)
    (out_dir / f"{name}.json").write_text(json.dumps(row, indent=1) + "\n")
    print(f"{name}: kappa_GN {row['kappa_GN']:.3g} kappa_S {row['kappa_S']:.3g} "
          f"kappa_lam* {row['kappa_lambda_star']:.3g} lambda* mean "
          f"{row['lambda_star_mean']:.3f} |S_f|/|JtJ| {row['Sf_over_JtJ_fro']:.3g} "
          f"rrms {rrms:.4f} calls {ngspice.n_calls} {row['runtime_s']:.0f}s",
          flush=True)
    return row


def summarize(rows: list[dict], tiers: list[str]) -> dict:
    keys = ("kappa_GN", "kappa_S", "kappa_lambda_star", "lambda_star_mean",
            "Sf_over_JtJ_fro", "objective_rrms_at_z")
    table = {}
    for tier in tiers:
        for point in ("start", "end"):
            sel = [r for r in rows if r["tier"] == tier and r["point"] == point
                   and "error" not in r]
            entry = {"n": len(sel), "n_partial": sum(r["partial"] for r in sel)}
            for key in keys:
                values = [r[key] for r in sel if math.isfinite(r[key])]
                entry[key] = {"mean": float(np.mean(values)) if values else None,
                              "median": float(np.median(values)) if values else None}
            table[f"{tier}/{point}"] = entry
    prediction = None
    if "a/end" in table and "c/end" in table and table["a/end"]["n"] and table["c/end"]["n"]:
        a, c = table["a/end"], table["c/end"]
        by_tag = {}
        for tag in TAGS:
            ra = next((r for r in rows if r["tier"] == "a" and r["tag"] == tag
                       and r["point"] == "end" and "error" not in r), None)
            rc = next((r for r in rows if r["tier"] == "c" and r["tag"] == tag
                       and r["point"] == "end" and "error" not in r), None)
            if ra and rc:
                by_tag[tag] = {"kappa_GN_c_gt_a": rc["kappa_GN"] > ra["kappa_GN"],
                               "lambda_star_c_gt_a": (rc["lambda_star_mean"]
                                                      > ra["lambda_star_mean"])}
        prediction = {
            "statement": ("at z_end, kappa_GN and lambda* are larger on tier c "
                          "than on tier a (registered; not an acceptance gate)"),
            "median_kappa_GN": [a["kappa_GN"]["median"], c["kappa_GN"]["median"]],
            "mean_lambda_star": [a["lambda_star_mean"]["mean"],
                                 c["lambda_star_mean"]["mean"]],
            "kappa_GN_larger_on_c": (c["kappa_GN"]["median"] is not None
                                     and a["kappa_GN"]["median"] is not None
                                     and c["kappa_GN"]["median"] > a["kappa_GN"]["median"]),
            "lambda_star_larger_on_c": (c["lambda_star_mean"]["mean"] is not None
                                        and a["lambda_star_mean"]["mean"] is not None
                                        and c["lambda_star_mean"]["mean"]
                                        > a["lambda_star_mean"]["mean"]),
            "per_tag": by_tag}
    return {"table": table, "registered_prediction": prediction}


def markdown(summary: dict, rows: list[dict]) -> str:
    lines = ["# Residual-curvature diagnostic (n2b)", "",
             "| tier/point | n | kappa_GN median | kappa_S median | "
             "kappa_lambda* median | lambda* mean | |S_f|/|JtJ| median | rrms median |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]

    def f(v):
        return "–" if v is None else f"{v:.3g}"
    for key, e in summary["table"].items():
        lines.append(f"| {key} | {e['n']} | {f(e['kappa_GN']['median'])} | "
                     f"{f(e['kappa_S']['median'])} | "
                     f"{f(e['kappa_lambda_star']['median'])} | "
                     f"{f(e['lambda_star_mean']['mean'])} | "
                     f"{f(e['Sf_over_JtJ_fro']['median'])} | "
                     f"{f(e['objective_rrms_at_z']['median'])} |")
    lines += ["", "| tier | target | point | kappa_GN | kappa_S | kappa_lambda* | "
              "lambda* mean | |S_f|/|JtJ| | rrms | calls |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: (r["tier"], r["tag"], r["point"])):
        if "error" in r:
            lines.append(f"| {r['tier']} | {r['tag']} | {r['point']} | "
                         f"{r['error']} | | | | | | |")
            continue
        lines.append(f"| {r['tier']} | {r['tag']} | {r['point']} | "
                     f"{f(r['kappa_GN'])} | {f(r['kappa_S'])} | "
                     f"{f(r['kappa_lambda_star'])} | {f(r['lambda_star_mean'])} | "
                     f"{f(r['Sf_over_JtJ_fro'])} | {f(r['objective_rrms_at_z'])} | "
                     f"{r['n_ngspice_calls']} |")
    pred = summary.get("registered_prediction")
    if pred:
        lines += ["", f"Registered prediction: {pred['statement']}.",
                  f"Median kappa_GN at end (a, c): {pred['median_kappa_GN']}; "
                  f"mean lambda* at end (a, c): {pred['mean_lambda_star']}."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiers", default="a,c")
    parser.add_argument("--tags", default=",".join(TAGS))
    parser.add_argument("--points", default="start,end")
    parser.add_argument("--out", default="out/research/dev/curvature_diag")
    parser.add_argument("--round-dir", default="out/cold_loop/full/round_1")
    parser.add_argument("--bank", default="data/processed/cold_loop/full_r1")
    parser.add_argument("--targets-a", default=TIERS["a"]["targets"])
    parser.add_argument("--targets-c", default=TIERS["c"]["targets"])
    parser.add_argument("--end-dir-a", default=TIERS["a"]["end_dir"])
    parser.add_argument("--end-dir-c", default=TIERS["c"]["end_dir"],
                        help="n2a's floor reference (sketch_none_k8 B20000) set_0 run directory")
    parser.add_argument("--diff-step", type=float, default=1e-3)
    parser.add_argument("--hess-step", type=float, default=5e-3)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--columns", type=int, default=None,
                        help="smoke tests only: first N Hessian columns")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tiers = [t for t in args.tiers.split(",") if t]
    options = {"out": str(out_dir), "round_dir": args.round_dir,
               "bank": args.bank, "diff_step": args.diff_step,
               "hess_step": args.hess_step,
               "regularization": args.regularization,
               "timeout_s": args.timeout_s, "columns": args.columns,
               "targets": {"a": args.targets_a, "c": args.targets_c},
               "end_dirs": {"a": args.end_dir_a, "c": args.end_dir_c}}
    for tier in tiers:
        target = ROOT / options["targets"][tier]
        if not target.exists():
            print(f"tier {tier}: targets {target} missing", flush=True)
            return 2
    def done(tier: str, tag: str, point: str) -> bool:
        """A finished point; a row recorded with ``error`` (e.g. a z_end
        record missing at the time) is computed again."""
        path = out_dir / f"{tier}_{tag}_{point}.json"
        return path.exists() and "error" not in json.loads(path.read_text())

    jobs = [(tier, tag, point, options) for tier in tiers
            for tag in args.tags.split(",") for point in args.points.split(",")
            if not done(tier, tag, point)]
    t0 = time.time()
    if jobs:
        with get_context("spawn").Pool(min(args.workers, len(jobs)),
                                       maxtasksperchild=1) as pool:
            for _ in pool.imap_unordered(diagnose, jobs):
                pass
    rows = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*_*_*.json"))
            if p.name != "summary.json"]
    summary = {"options": options, "wall_s": round(time.time() - t0, 1),
               **summarize(rows, tiers), "rows": rows}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out_dir / "summary.md").write_text(markdown(summary, rows))
    print(json.dumps(summary["table"], indent=1))
    if summary["registered_prediction"]:
        print(json.dumps(summary["registered_prediction"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
