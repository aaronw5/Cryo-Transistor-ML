#!/usr/bin/env python3
"""Method-section figures for the M_bias deck (scripts/make_mbias_slides.py),
sky130 77 K, all 18 devices.  Everything is read from stored outputs except
the training trace of one fold (Python/torch only, no NGSpice), which re-runs
the fold's five seeds with the e1_correction / i3_2_bias recipe and checks the
result against the stored fold model.

Inputs:
  out/research/explore/I13_1/records/real_{dev,test}.{npz,json}   starting-card records (state C)
  out/research/explore/I13_1/all18/{results,corrections,heldout_cards}.json  (I3-2 recipe, state C)
  out/research/explore/I13_1/summary.json   state-C start-card scores (round 3 + FD stage)
  out/research/ngspice_direct/*/summary.json, out/research/polished/direct43_r3_fd
  out/research/reference_scores.json, out/cold_loop/full_direct/round_3/spec.json
  out/research/explore/mbias_deck/sky130/check.json
Outputs: slides/mbias_plots/m_*.png, slides/mbias_plots/eq_*.png,
         out/research/explore/mbias_deck/method_numbers.json

  PYTHONPATH=src .venv/bin/python scripts/research/explore/mbias_method_plots.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from cryoml.explore.e1_correction import _loss_fn, load_devices, make_batch, raw_features  # noqa: E402
from cryoml.explore.i3_2_bias import HYPER_BIAS, correction_from_json  # noqa: E402
from cryoml.explore.i2_1_joint import log_ratio  # noqa: E402

E1 = ROOT / "out/research/explore/I13_1/records"      # I13-1 state-C records
ALL18 = ROOT / "out/research/explore/I13_1/all18"     # I13-1 state-C rerun of the I3-2 recipe
I13_SUMMARY = ROOT / "out/research/explore/I13_1/summary.json"
DECKDIR = ROOT / "out/research/explore/mbias_deck"
OUT = ROOT / "slides" / "mbias_plots"
ND = ROOT / "out/research/ngspice_direct"
SPEC3 = ROOT / "out/cold_loop/full_direct/round_3/spec.json"
EXAMPLE = "pmos_L2_W5"
VDD = 1.85
C_N, C_P = "#0072B2", "#D55E00"
C_M, C_C0, C_T1 = "#D55E00", "#009E73", "#CC79A7"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})


def jload(p):
    return json.loads(Path(p).read_text())


def save(fig, name: str, dpi: int = 160) -> Path:
    from PIL import Image
    path = OUT / name
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    with Image.open(path) as im:
        im.convert("RGB").quantize(colors=256).save(path, optimize=True)
    return path


def records():
    recs = load_devices(E1 / "real_dev") + load_devices(E1 / "real_test")
    return sorted(recs, key=lambda r: r.tag)


# ---------------------------------------------------------------------------
# equations (mathtext)
# ---------------------------------------------------------------------------
EQS = {
    "model": r"$I_{\mathrm{corr},p} = I_{\mathrm{card}}(V_p;\,\theta_d)\;\exp\!\left(g(x_p)\right)$",
    "x": r"$x_p = \left(\mathrm{is\_pmos},\ |V_{GS,p}|/V_{DD},\ |V_{DS,p}|/V_{DD}\right),\quad V_{DD} = 1.85\ \mathrm{V}$",
    "z": r"$z = (x-\mu)/\sigma$  (voltage inputs; $\mu,\sigma$ over the fold's training points; is_pmos unscaled)",
    "g": r"$g(x) = \frac{1}{5}\sum_{s=0}^{4}\tanh\!\left(f_s(z)\right),\qquad g\in(-1,1),\ \ e^{g}\in(0.37,\,2.72)$",
    "rrms": r"$\mathrm{RRMS}_c = \frac{1}{\mathrm{den}_c}\sqrt{\frac{1}{n_c}\sum_{p\in c}(I_{\mathrm{sim},p}-I_{\mathrm{meas},p})^2},"
            r"\qquad \mathrm{den}_c = \frac{1}{n_c}\sum_{p\in c}|I_{\mathrm{meas},p}|$",
    "dev": r"$\mathrm{RRMS}_d = \frac{1}{C_d}\sum_{c\in\mathcal{C}_d}\mathrm{RRMS}_c$   ($\mathcal{C}_d$: published-card inclusion set, $C_d$ = 9 to 11 curves)",
    "loss": r"$\mathcal{L}_s = \frac{1}{D}\sum_{d}\frac{1}{C_d}\sum_{c}\frac{1}{\mathrm{den}_c}"
            r"\sqrt{\frac{1}{n_c}\sum_{p\in c}\left(I_{\mathrm{card},p}\,e^{g_s(x_p)}-I_{\mathrm{meas},p}\right)^2}"
            r"\ +\ 10^{-4}\sum_{\ell}\|W_\ell\|_F^2$",
    "resid": r"$r_p = \left(I_{\mathrm{card},p}(\theta)\,e^{g(x_p)} - I_{\mathrm{meas},p}\right)\,"
             r"\frac{w_c}{\mathrm{den}_c\,\sqrt{n_c}\,\sqrt{C_d}},\qquad \min_{\theta\in\mathrm{box}}\ \sum_p r_p^2$",
    "irls": r"round 1: $w_c = 1$;   round 2: $w_c \propto 1/\sqrt{\max(\mathrm{RRMS}_c,\,10^{-3})}$,  $\sum_c w_c^2 = C_d$",
    "box": r"$\theta_k = c_k\,\exp(u_k h_k),\quad u_k\in[-1,1]$",
    "va": r"$I_{d\to s} = I_{\mathrm{sense}} + \left(e^{g}-1\right) I_{\mathrm{sense}} = e^{g}\,I_{\mathrm{card}}$",
    "t1": r"T1_inloop: $g = \tanh(c_{\mathrm{pol},\mathrm{curve}})$, one constant per (polarity, curve tag)",
}


def equations() -> dict:
    sizes = {}
    for name, tex in EQS.items():
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, tex, fontsize=20, color="#202428")
        path = OUT / f"eq_{name}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.04, transparent=False,
                    facecolor="white")
        plt.close(fig)
        from PIL import Image
        with Image.open(path) as im:
            sizes[name] = [im.width / 200, im.height / 200]   # inches at 20 pt
    return sizes


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_data(recs) -> dict:
    rec = next(r for r in recs if r.tag == EXAMPLE)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), layout="constrained")
    info = []
    for c in rec.curves:
        ax = axes[0] if c.kind == "idvd" else axes[1]
        x = np.abs(c.vd if c.kind == "idvd" else c.vg)
        y = np.abs(c.target) * 1e6
        col = "black" if c.included else "0.7"
        ax.plot(x[c.start:], y[c.start:], "o", ms=2.2, mfc="none", mec=col, mew=0.6)
        if c.start:
            ax.plot(x[:c.start], y[:c.start], "x", ms=3, color="#CC79A7", mew=0.8)
        if c.included:
            ax.hlines(c.den * 1e6, x[c.start:].min(), x[c.start:].max(), colors=C_N, lw=0.9,
                      linestyles="--")
        info.append({"key": c.key, "start": int(c.start), "n": int(len(c.target) - c.start),
                     "den_uA": c.den * 1e6, "included": bool(c.included)})
    axes[1].set_yscale("log")
    axes[0].set_title("Id-Vd, |Vgs| = 0.37 ... 1.85 V (5 curves)")
    axes[1].set_title("Id-Vg, |Vds| = 0.01 ... 1.85 V (6 curves), log")
    axes[0].set_xlabel("|$V_{DS}$| (V)")
    axes[1].set_xlabel("|$V_{GS}$| (V)")
    for ax in axes:
        ax.set_ylabel("|$I_D$| ($\\mu$A)")
        ax.grid(True, alpha=0.25)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", mfc="none", mec="black", label="measured, scored"),
         Line2D([], [], marker="o", ls="", mfc="none", mec="0.7", label="curve not in inclusion set"),
         Line2D([], [], marker="x", ls="", color="#CC79A7", label="removed by trim"),
         Line2D([], [], ls="--", color=C_N, label="den$_c$ = mean |I$_{meas}$| of the curve")]
    fig.legend(handles=h, loc="outside upper center", ncol=4, fontsize=10.5)
    save(fig, "m_data.png")
    return {"tag": EXAMPLE, "curves": info}


STAGES = (("7 parameters", ND / "seven_r1_direct/summary.json"),
          ("43 p., round 1", ND / "full_r1_direct_seeded/summary.json"),
          ("43 p., round 2", ND / "full_r2_direct/summary.json"),
          ("43 p., round 3", ND / "full_r3_direct/summary.json"),
          ("round 3 + FD", ROOT / "out/research/polished/direct43_r3_fd/real/summary.json"))


def fig_stages() -> dict:
    ref = jload(ROOT / "out/research/reference_scores.json")
    pubd = {t: float(v["rrms"]) for t, v in ref["released"]["devices"].items()}
    rows = [("published card", pubd, None, None, None)]
    for lab, path in STAGES:
        s = jload(path)
        d = {t: float(v["rrms"]) for t, v in s["devices"].items()}
        if lab == "round 3 + FD":     # I13-1: the starting card scored in state C (clamp written)
            d = {t: float(v["new"]) for t, v in jload(I13_SUMMARY)["per_device"]["start"].items()}
        rows.append((lab, d, s["mean_ngspice_calls_per_device"], s["wall_s"], s))
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, 1.15]})
    x = np.arange(len(rows))
    labels = [r[0].replace(", ", ",\n").replace(" + ", "\n+ ").replace("published ", "published\n")
              .replace("7 parameters", "7\nparameters") for r in rows]
    means = [float(np.mean(list(r[1].values()))) for r in rows]
    ax = axes[0]
    ax.bar(x, means, color=["0.6"] + [C_C0] * (len(rows) - 1))
    for i, v in enumerate(means):
        ax.text(i, v + 0.004, f"{v:.4f}", ha="center", fontsize=10)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("all-device RRMS (18 devices)")
    ax.set_title("All-device RRMS after each stage")
    ax.grid(True, axis="y", alpha=0.25)
    ax = axes[1]
    calls = [r[2] for r in rows[1:]]
    ax.bar(x[1:], calls, color=C_C0)
    for i, c, w in zip(x[1:], calls, [r[3] for r in rows[1:]]):
        ax.text(i, c * 1.15, f"{c:,.0f}\n{w:,.0f} s", ha="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(100, 2e5)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_xticks(x[1:], labels[1:], fontsize=9)
    ax.set_ylabel("NGSpice simulations per device (mean)")
    ax.set_title("Simulations per device; wall time of the run")
    ax.grid(True, axis="y", alpha=0.25)
    ax = axes[2]
    for t in sorted(pubd):
        ax.plot(x, [r[1][t] for r in rows], "-o", ms=3, lw=1,
                color=C_P if t.startswith("pmos") else C_N, alpha=0.75)
    ax.set_yscale("log")
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("device RRMS")
    ax.set_title("Per device (blue nMOS, orange pMOS)")
    ax.grid(True, which="both", alpha=0.2)
    save(fig, "m_stages.png")
    out = []
    for (lab, d, c, w, s), m in zip(rows, means):
        out.append({"label": lab, "all_device": m, "calls": c, "wall_s": w,
                    "combined": None if s is None else float(
                        (np.mean([v for k, v in d.items() if k.startswith("nmos")])
                         + np.mean([v for k, v in d.items() if k.startswith("pmos")])) / 2),
                    "options": None if s is None else s.get("options")})
    r1 = jload(ND / "full_r1_direct_seeded/summary.json")["devices"]
    cma_better = {}
    for lab, path in STAGES[:4]:
        dv = jload(path)["devices"]
        cma_better[lab] = sum(r["cmaes_best"] < min(r["start_scores"]) for r in dv.values())
    return {"stages": out, "example_r1": {"tag": "pmos_L8_W5", **{
        k: r1["pmos_L8_W5"][k] for k in ("start_scores", "cmaes_best", "polish_history", "rrms",
                                          "n_sim")}}, "cma_better_than_best_start": cma_better}


def fig_screen() -> dict:
    s = jload(ROOT / "out/pdk_parameter_sensitivity/summary.json")
    p = s["parameters"]
    resp = {k: max(v.get("max_full_grid_response", 0.0) or 0.0,
                   v.get("max_high_voltage_response", 0.0) or 0.0) for k, v in p.items()}
    order = sorted(resp, key=lambda k: -resp[k])
    groups = sorted({p[k]["group"] for k in s["selected_parameters"]})
    gcol = dict(zip(groups, ["#0072B2", "#009E73", "#E69F00", "#CC79A7", "#D55E00"]))
    fig, ax = plt.subplots(figsize=(16.5, 4.6), layout="constrained")
    floor = 1e-9
    for i, k in enumerate(order):
        sel = p[k]["selected"]
        ax.bar(i, max(resp[k], floor), color=gcol[p[k]["group"]] if sel else "0.75", width=0.8)
    ax.axhline(s["effective_threshold"], color="black", ls="--", lw=1)
    ax.text(len(order) - 1, s["effective_threshold"] * 1.6, "threshold 1e-4", ha="right",
            fontsize=10)
    ax.set_yscale("log")
    ax.set_ylim(floor, 50)
    n_sel = sum(p[k]["selected"] for k in order)
    ax.set_xticks(range(n_sel), order[:n_sel], rotation=90, fontsize=8.5)
    ax.set_xlim(-1, len(order))
    n_zero = sum(resp[k] <= 0 for k in order)
    ax.text(len(order) - 1, 3e-9, f"{n_zero} candidates with zero response (no bar)", ha="right",
            fontsize=10)
    ax.set_ylabel("max normalized response\nover 18 devices and 11 curves")
    ax.set_title(f"{len(order)} candidate parameters, sorted; coloured = selected ({n_sel}); "
                 f"grey = below threshold ({len(order) - n_sel})")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=gcol[g], label=g.replace("_", " ")) for g in groups],
              fontsize=9.5, loc="upper right")
    save(fig, "m_screen.png")
    return {"n_candidates": len(order), "n_selected": n_sel,
            "groups": {g: [k for k in s["selected_parameters"] if p[k]["group"] == g]
                       for g in groups},
            "threshold": s["effective_threshold"], "fraction": s["perturbation_fraction"],
            "n_excluded_pre_screen": len(s["pre_screen_exclusions"])}


def fig_box() -> dict:
    from cold_loop import box_for
    specs = {r: jload(ROOT / f"out/cold_loop/{p}/spec.json") for r, p in
             ((1, "full/round_1"), (2, "full_direct/round_2"), (3, "full_direct/round_3"))}
    tag = EXAMPLE
    names = specs[1]["selected_parameters"]
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 4.5), layout="constrained",
                             gridspec_kw={"width_ratios": [3, 1]})
    ax = axes[0]
    x = np.arange(len(names))
    for k, (r, col) in enumerate(((1, "0.6"), (2, C_N), (3, C_C0))):
        hw = specs[r]["half_width_by_device"][tag]
        box = box_for(specs[r], tag)
        ax.bar(x + (k - 1) * 0.27, [math.exp(hw[n]) if box.active[i] else np.nan
                                    for i, n in enumerate(names)], 0.27, color=col,
               label=f"round {r}")
    ax.set_yscale("log")
    ax.set_xticks(x, names, rotation=80, fontsize=9)
    ax.set_ylabel("box factor $e^{h_k}$  (bounds $c_k e^{\\pm h_k}$)")
    ax.set_title(f"{tag}: half-width per parameter and round (missing bar = fixed)")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax = axes[1]
    tags = sorted(specs[3]["published_by_device"])
    na = [box_for(specs[3], t).n_active for t in tags]
    ax.barh(range(len(tags)), na, color=[C_P if t.startswith("pmos") else C_N for t in tags])
    ax.set_yticks(range(len(tags)), tags, fontsize=8.5)
    ax.set_xlim(30, 43)
    ax.set_xlabel("active parameters (of 43)")
    ax.set_title("Active parameters per device")
    ax.invert_yaxis()
    save(fig, "m_box.png")
    return {"n_active_min": min(na), "n_active_max": max(na)}


def fig_records(recs) -> dict:
    per = {r.tag: make_batch([r]) for r in recs}
    n_pts = {t: len(b.sim) for t, b in per.items()}
    n_curves = {t: len(b.curve_den) for t, b in per.items()}
    total = sum(n_pts.values())
    train = {t: total - n for t, n in n_pts.items()}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), layout="constrained",
                             gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axes[0]
    tags = [r.tag for r in recs]
    ax.bar(range(len(tags)), [n_pts[t] for t in tags],
           color=[C_P if t.startswith("pmos") else C_N for t in tags])
    ax.set_xticks(range(len(tags)), tags, rotation=60, ha="right", fontsize=8.5)
    ax.set_ylabel("trimmed scored points")
    ax.set_title(f"Points per device; total {total:,}")
    ax.grid(True, axis="y", alpha=0.25)
    stats = {}
    for ax, fam, col in ((axes[1], "nmos", C_N), (axes[2], "pmos", C_P)):
        b = make_batch([r for r in recs if r.family == fam])
        lr = log_ratio(b.target, b.sim)
        ok = np.isfinite(lr)
        sc = ax.scatter(b.phi[ok, 1], b.phi[ok, 2], c=np.clip(lr[ok], -1, 1), cmap="RdBu_r",
                        vmin=-1, vmax=1, s=4, lw=0)
        ax.set_xlabel("|$V_{GS}$| / $V_{DD}$")
        ax.set_ylabel("|$V_{DS}$| / $V_{DD}$")
        ax.set_title(f"{fam[0].upper()}MOS: {ok.sum():,} points, colour = ln(I$_{{meas}}$/I$_{{card}}$)")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        stats[fam] = {"n_points": int(len(lr)), "n_finite": int(ok.sum()),
                      "median_abs_ln_ratio": float(np.median(np.abs(lr[ok])))}
    fig.colorbar(sc, ax=axes[1:], label="ln(I$_{meas}$ / I$_{card}$), clipped to [-1, 1]", shrink=0.9)
    save(fig, "m_records.png")
    return {"n_points": n_pts, "n_curves": n_curves, "total": total,
            "train_min": min(train.values()), "train_max": max(train.values()), "ln_ratio": stats}


def fig_network() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    layers = [(0.9, 3, ["is_pmos", "z($|V_{GS}|/V_{DD}$)", "z($|V_{DS}|/V_{DD}$)"]),
              (3.2, 16, None), (5.3, 16, None), (7.2, 1, None)]
    pos = []
    for x, n, labels in layers:
        ys = np.linspace(3.9, 0.5, n) if n > 1 else np.array([2.2])
        if n == 3:
            ys = np.array([3.1, 2.2, 1.3])
        pos.append([(x, y) for y in ys])
    for a, b in zip(pos[:-1], pos[1:]):
        for (x1, y1) in a:
            for (x2, y2) in b:
                ax.plot([x1, x2], [y1, y2], color="0.82", lw=0.35, zorder=1)
    for li, pts in enumerate(pos):
        r = 0.13 if li in (0, 3) else 0.09
        for j, (x, y) in enumerate(pts):
            ax.add_patch(Circle((x, y), r, fc="white", ec="#202428", lw=1, zorder=2))
            if li == 0:
                ax.text(x - 0.22, y, layers[0][2][j], ha="right", va="center", fontsize=11)
    for x, t in ((3.2, "16 tanh"), (5.3, "16 tanh"), (7.2, "1 linear")):
        ax.text(x, 4.25, t, ha="center", fontsize=11)
    ax.text(2.05, 0.05, "3x16 + 16 = 64", ha="center", fontsize=10, color="0.35")
    ax.text(4.25, 0.05, "16x16 + 16 = 272", ha="center", fontsize=10, color="0.35")
    ax.text(6.25, 0.05, "16 + 1 = 17 (init 0)", ha="center", fontsize=10, color="0.35")
    ax.add_patch(FancyBboxPatch((7.75, 1.8), 1.25, 0.8, boxstyle="round,pad=0.05", fc="#F0F4F8",
                                ec="0.45"))
    ax.text(8.375, 2.2, "tanh\n$f_s \\to \\tanh f_s$", ha="center", va="center", fontsize=10.5)
    ax.add_patch(FancyBboxPatch((9.4, 1.55), 1.9, 1.3, boxstyle="round,pad=0.05", fc="#F0F4F8",
                                ec="0.45"))
    ax.text(10.35, 2.2, "mean over\nseeds s = 0..4\n= g(x)", ha="center", va="center",
            fontsize=10.5)
    for x1, x2 in ((7.33, 7.72), (9.05, 9.37)):
        ax.add_patch(FancyArrowPatch((x1, 2.2), (x2, 2.2), arrowstyle="-|>", mutation_scale=12,
                                     color="0.35"))
    ax.text(5.3, -0.0, "", fontsize=1)
    ax.text(0.9, 3.75, "one seed $f_s$:\n353 weights", fontsize=11, ha="center")
    save(fig, "m_network.png")


def train_trace(recs, held: str, every: int = 10) -> dict:
    """Fold ``held``: re-run the five seeds of fit_bias('M_bias') with a loss
    trace every ``every`` steps; compare g with the stored fold model."""
    import torch
    torch.set_num_threads(1)
    hp = HYPER_BIAS
    train = [r for r in recs if r.tag != held]
    batch = make_batch(train)
    loss = _loss_fn(batch)
    cols = [0, 1, 2]
    phi = batch.phi[:, cols]
    mu, sd = phi.mean(axis=0), phi.std(axis=0)
    mu[0], sd[0] = 0.0, 1.0
    sd = np.where(sd > 1e-12, sd, 1.0)
    z = torch.tensor((phi - mu) / sd, dtype=torch.float64)
    traces, gs, t0 = {}, [], time.time()
    for seed in hp["seeds"]:
        torch.manual_seed(seed)
        dims = [3] + [hp["hidden"]] * hp["layers"] + [1]
        mods = [torch.nn.Linear(a, b).double() for a, b in zip(dims[:-1], dims[1:])]
        torch.nn.init.zeros_(mods[-1].weight)
        torch.nn.init.zeros_(mods[-1].bias)
        opt = torch.optim.Adam([p for m in mods for p in m.parameters()], lr=hp["lr"])

        def forward(x):
            for i, m in enumerate(mods):
                x = m(x)
                if i < len(mods) - 1:
                    x = torch.tanh(x)
            return torch.tanh(x[:, 0])
        tr = []
        for step in range(hp["steps"]):
            opt.zero_grad()
            fit = loss(forward(z))
            value = fit + hp["l2"] * sum((m.weight ** 2).sum() for m in mods)
            if step % every == 0:
                tr.append((step, float(fit), float(value)))
            value.backward()
            opt.step()
        with torch.no_grad():
            fit = loss(forward(z))
            tr.append((hp["steps"], float(fit),
                       float(fit + hp["l2"] * sum((m.weight ** 2).sum() for m in mods))))
            gs.append(forward(z).numpy().copy())
        traces[seed] = tr
    g_mean = np.mean(gs, axis=0)
    stored = correction_from_json(jload(ALL18 / "corrections.json")[f"M_bias|{held}"])
    g_stored = stored.g(batch.phi)
    import torch as _t
    gt = _t.tensor(g_mean)
    return {"held": held, "n_train_points": int(len(batch.sim)), "n_train_devices": len(train),
            "mu": mu.tolist(), "sd": sd.tolist(), "traces": traces,
            "loss_g0": float(loss(_t.zeros(len(batch.sim), dtype=_t.float64))),
            "loss_ensemble": float(loss(gt)),
            "max_abs_g_vs_stored": float(np.max(np.abs(g_mean - g_stored))),
            "seed_spread_g_std_median": float(np.median(np.std(gs, axis=0))),
            "seed_spread_g_std_max": float(np.max(np.std(gs, axis=0))),
            "runtime_s": round(time.time() - t0, 1)}


def fig_training(tr: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), layout="constrained")
    cols = ["#0072B2", "#009E73", "#E69F00", "#CC79A7", "#D55E00"]
    for seed, col in zip(sorted(tr["traces"]), cols):
        a = np.array(tr["traces"][seed])
        axes[0].plot(a[:, 0], a[:, 1], color=col, lw=1.2, label=f"seed {seed}")
        axes[1].plot(a[:, 0], a[:, 2] - a[:, 1], color=col, lw=1.2)
    axes[0].axhline(tr["loss_g0"], color="0.4", ls="--", lw=1)
    axes[0].text(2000, tr["loss_g0"], f"g = 0: {tr['loss_g0']:.4f}", ha="right", va="bottom",
                 fontsize=10)
    axes[0].axhline(tr["loss_ensemble"], color="black", ls=":", lw=1)
    axes[0].text(30, tr["loss_ensemble"] - 0.0003, f"5-seed mean g: {tr['loss_ensemble']:.4f}",
                 ha="left", va="top", fontsize=10)
    axes[0].set_title(f"Mean device RRMS term, fold {tr['held']} held out")
    axes[0].set_xlabel("Adam step")
    axes[0].set_ylabel("training RRMS term")
    axes[0].legend(fontsize=9.5, ncol=1, loc="upper right", bbox_to_anchor=(1.0, 0.9))
    axes[1].set_title("L2 term $10^{-4}\\sum\\|W\\|^2$")
    axes[1].set_ylabel("L2 term")
    axes[0].set_ylim(tr["loss_ensemble"] - 0.003, tr["loss_g0"] + 0.002)
    axes[1].set_xlabel("Adam step")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    save(fig, "m_training.png")


def fig_g_folds(recs) -> dict:
    corr = jload(ALL18 / "corrections.json")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), layout="constrained")
    vals = {"nmos": [], "pmos": []}
    for r in recs:
        c = correction_from_json(corr[f"M_bias|{r.tag}"])
        b = make_batch([r])
        vals[r.family].append(c.g(b.phi))
    ax = axes[0]
    bins = np.linspace(-0.8, 0.8, 65)
    for fam, col in (("nmos", C_N), ("pmos", C_P)):
        v = np.concatenate(vals[fam])
        ax.hist(v, bins=bins, color=col, alpha=0.6, label=f"{fam[0].upper()}MOS, {len(v):,} points")
    ax.set_xlabel("g at the held-out device's scored points")
    ax.set_ylabel("points")
    ax.set_title("g of each fold model on its held-out device")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    v = np.linspace(0, VDD, 121)
    full = correction_from_json(corr["full|M_bias"])
    for ax, p, col in ((axes[1], 0, C_N), (axes[2], 1, C_P)):
        for vds, ls in ((0.37, ":"), (1.85, "-")):
            for r in recs:
                c = correction_from_json(corr[f"M_bias|{r.tag}"])
                phi = np.column_stack([np.full(v.size, float(p)), v / VDD, np.full(v.size, vds / VDD),
                                       np.zeros((v.size, 3))])
                ax.plot(v, np.exp(c.g(phi)), ls, color=col, lw=0.6, alpha=0.45)
            phi = np.column_stack([np.full(v.size, float(p)), v / VDD, np.full(v.size, vds / VDD),
                                   np.zeros((v.size, 3))])
            ax.plot(v, np.exp(full.g(phi)), ls, color="black", lw=1.6,
                    label=f"|Vds| = {vds:.2f} V, model on 18 devices")
        ax.axhline(1, color="0.5", lw=0.8)
        ax.set_xlabel("|$V_{GS}$| (V)")
        ax.set_ylabel("exp(g)")
        ax.set_title(f"{'pMOS' if p else 'nMOS'}: 18 fold models (thin), full model (black)")
        ax.legend(fontsize=9.5, loc="best")
        ax.grid(True, alpha=0.25)
    save(fig, "m_g_folds.png")
    out = {}
    for fam in vals:
        a = np.concatenate(vals[fam])
        out[fam] = {"n": int(a.size), "min": float(a.min()), "max": float(a.max()),
                    "median_abs": float(np.median(np.abs(a)))}
    return out


def fig_refit(res: dict) -> dict:
    from cold_loop import box_for
    spec = jload(SPEC3)
    cards = jload(ROOT / "out/research/polished/direct43_r3_fd/real/summary.json")["devices"]
    hc = jload(ALL18 / "heldout_cards.json")
    box = box_for(spec, EXAMPLE)
    loc = lambda p: box.params_to_local({**spec["published_by_device"][EXAMPLE],
                                         **{k: v for k, v in p.items() if k in box.names}})
    u0 = loc(cards[EXAMPLE]["params"])
    du = {lab: loc(hc[lab][EXAMPLE]) - u0 for lab in ("C0_refit", "M_bias")}
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 4.6), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 2.4]})
    ax = axes[0]
    hr = res["heldout_rows"]["M_bias"]
    for t, row in hr.items():
        h = row["history"]
        ax.plot(range(len(h)), h, "-o", ms=3, lw=1, color=C_P if t.startswith("pmos") else C_N,
                alpha=0.8)
    ax.set_yscale("log")
    ax.set_xticks([0, 1, 2], ["start card\nx exp(g)", "after\nround 1", "after\nround 2"])
    ax.set_ylabel("device RRMS with g")
    ax.set_title("M_bias refit, 18 held-out devices")
    ax.grid(True, alpha=0.25)
    ax = axes[1]
    x = np.arange(len(box.names))
    ax.bar(x - 0.2, du["C0_refit"], 0.4, color=C_C0, label="C0_refit (g = 0)")
    ax.bar(x + 0.2, du["M_bias"], 0.4, color=C_M, label="M_bias (g fixed)")
    for i, a in enumerate(box.active):
        if not a:
            ax.text(i, 0, "fixed", rotation=90, ha="center", va="bottom", fontsize=8, color="0.4")
    ax.set_xticks(x, box.names, rotation=70, fontsize=9)
    ax.set_ylabel("$u_{final} - u_{start}$ (box coordinates)")
    ax.set_title(f"{EXAMPLE}: parameter change from the starting card, box coordinates")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    save(fig, "m_refit.png")
    return {"example": EXAMPLE, "n_active": int(box.n_active),
            "max_abs_du": {k: float(np.max(np.abs(v))) for k, v in du.items()},
            "history_example": hr[EXAMPLE]["history"],
            "n_sim": {lab: [min(r["n_sim"] for r in v.values()), max(r["n_sim"] for r in v.values()),
                            float(np.mean([r["n_sim"] for r in v.values()]))]
                      for lab, v in res["heldout_rows"].items()},
            "runtime_s": {lab: float(np.mean([r["runtime_s"] for r in v.values()]))
                          for lab, v in res["heldout_rows"].items()}}


def scatter(ax, xs: dict, ys: dict, xl: str, yl: str):
    tags = sorted(xs)
    a = np.array([xs[t] for t in tags])
    b = np.array([ys[t] for t in tags])
    cols = [C_P if t.startswith("pmos") else C_N for t in tags]
    ax.scatter(a, b, c=cols, s=28, zorder=3)
    lo, hi = min(a.min(), b.min()) * 0.8, max(a.max(), b.max()) * 1.25
    ax.plot([lo, hi], [lo, hi], color="0.4", lw=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.grid(True, which="both", alpha=0.2)
    n_below = int(np.sum(b < a))
    ax.set_title(f"{n_below} of {len(tags)} devices below y = x")


def fig_refit_effect(res: dict) -> dict:
    ph = res["posthoc_no_refit"]
    pd = res["per_device"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    labs = [("starting card\n(g = 0)", ph["C0_start"]["per_device"], C_C0),
            ("C0_refit\n(g = 0, refit)", pd["C0_refit"], C_C0),
            ("starting card\nx exp(g)", ph["M_bias"]["per_device"], C_M),
            ("M_bias\n(refit with g)", pd["M_bias"], C_M)]
    means = [float(np.mean(list(d.values()))) for _, d, _ in labs]
    ax.bar(range(4), means, color=[c for *_, c in labs], alpha=0.85)
    for i, m in enumerate(means):
        ax.text(i, m + 0.002, f"{m:.4f}", ha="center", fontsize=10.5)
    ax.set_xticks(range(4), [l for l, *_ in labs], fontsize=10)
    ax.set_ylabel("all-device RRMS, held-out")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_title("Same fold g; with and without the card refit")
    scatter(axes[1], ph["M_bias"]["per_device"], pd["M_bias"],
            "starting card x exp(g)  (no refit)", "M_bias (card refit with g)")
    save(fig, "m_refit_effect.png")
    return {"means": dict(zip(["start", "C0_refit", "start_x_g", "M_bias"], means))}


def fig_controls(res: dict) -> dict:
    pd = res["per_device"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), layout="constrained")
    scatter(axes[0], pd["C0_refit"], pd["M_bias"], "C0_refit RRMS", "M_bias RRMS")
    scatter(axes[1], pd["T1_inloop"], pd["M_bias"], "T1_inloop RRMS", "M_bias RRMS")
    save(fig, "m_controls.png")
    return {k: res["comparisons"][k] for k in ("M_bias_vs_C0_refit", "M_bias_vs_T1_inloop",
                                                 "T1_inloop_vs_C0_refit")}


def fig_loso() -> None:
    tags = sorted(jload(ALL18 / "results.json")["per_device"]["M_bias"])
    n = len(tags)
    fig, ax = plt.subplots(figsize=(7.2, 6.0), layout="constrained")
    m = np.ones((n, n))
    np.fill_diagonal(m, 0)
    from matplotlib.colors import ListedColormap
    ax.imshow(m, cmap=ListedColormap([C_M, "#DCE6EE"]), vmin=0, vmax=1)
    ax.set_xticks(range(n), tags, rotation=70, fontsize=8)
    ax.set_yticks(range(n), [f"fold {i + 1}" for i in range(n)], fontsize=8)
    ax.set_xlabel("device")
    ax.set_title("light: 17 training devices; orange: held-out device", fontsize=11)
    save(fig, "m_loso.png")


def fig_va() -> None:
    """Netlist topology of the deployed correction (i3_2_bias.to_verilog_a_bias)."""
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    k = dict(color="#202428", lw=1.6)
    dot = lambda x, y: ax.add_patch(Circle((x, y), 0.07, fc="#202428", zorder=4))
    # node d (external drain) and top rail
    ax.plot([0.8, 6.2], [4.6, 4.6], **k)
    dot(0.8, 4.6)
    ax.text(0.65, 4.6, "d", ha="right", va="center", fontsize=14)
    dot(3.4, 4.6)
    # sense branch d -> di (vertical), then M1 from di to s
    ax.plot([3.4, 3.4], [4.6, 4.3], **k)
    ax.add_patch(Rectangle((2.95, 3.7), 0.9, 0.6, fc="#F0F4F8", ec="#202428", zorder=3))
    ax.text(3.4, 4.0, "sense\nV = 0", ha="center", va="center", fontsize=9.5, zorder=4)
    ax.plot([3.4, 3.4], [3.7, 3.3], **k)
    dot(3.4, 3.3)
    ax.text(3.55, 3.3, "di", ha="left", va="center", fontsize=14)
    ax.plot([3.4, 3.4], [3.3, 2.9], **k)
    ax.add_patch(Rectangle((2.85, 1.7), 1.1, 1.2, fc="white", ec="#202428", lw=1.6, zorder=3))
    ax.text(3.4, 2.3, "M1\nBSIM4\ncard $\\theta_d$", ha="center", va="center", fontsize=10, zorder=4)
    ax.plot([3.4, 3.4], [1.7, 0.6], **k)
    ax.plot([0.8, 2.85], [2.3, 2.3], **k)
    dot(0.8, 2.3)
    ax.text(0.65, 2.3, "g", ha="right", va="center", fontsize=14)
    # correction source d -> s
    ax.plot([6.2, 6.2], [4.6, 3.1], **k)
    ax.add_patch(Circle((6.2, 2.5), 0.6, fc="#FCE9DE", ec=C_M, lw=1.6, zorder=3))
    ax.add_patch(FancyArrowPatch((6.2, 2.9), (6.2, 2.1), arrowstyle="-|>", mutation_scale=14,
                                 color=C_M, zorder=4))
    ax.text(6.9, 2.5, "$(e^{g}-1)\\,I_{\\mathrm{sense}}$", ha="left", va="center", fontsize=11)
    ax.plot([6.2, 6.2, 3.4], [1.9, 0.6, 0.6], **k)
    dot(3.4, 0.6)
    ax.text(3.4, 0.4, "s", ha="center", va="top", fontsize=14)
    # g(x) reads V(g, s) and V(d, s)
    ax.plot([0.8, 1.6, 1.6, 5.45], [2.3, 2.3, 1.15, 1.15], color=C_M, lw=1, ls=":")
    ax.annotate("", xy=(5.6, 2.3), xytext=(5.45, 1.15),
                arrowprops=dict(arrowstyle="-|>", color=C_M, lw=1, ls=":"))
    ax.text(1.75, 1.0, "g from V(g,s), V(d,s)", fontsize=9.5, color=C_M, va="top")
    # module outline
    ax.add_patch(Rectangle((2.7, 3.55), 4.15, 1.35, fc="none", ec=C_M, ls="--", lw=1))
    ax.add_patch(Rectangle((5.45, 1.6), 1.4, 1.95, fc="none", ec=C_M, ls="--", lw=1))
    ax.text(6.85, 5.0, "Verilog-A module, ports (d, g, s, di)", ha="right", fontsize=10,
            color=C_M)
    save(fig, "m_va.png")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    recs = records()
    res = jload(ALL18 / "results.json")
    numbers = {"eq_sizes_in": equations()}
    numbers["data"] = fig_data(recs)
    numbers["stages"] = fig_stages()
    numbers["screen"] = fig_screen()
    numbers["box"] = fig_box()
    numbers["records"] = fig_records(recs)
    fig_network()
    cache = DECKDIR / "train_trace.json"
    if cache.exists() and "--retrain" not in sys.argv:
        tr = jload(cache)
    else:
        tr = train_trace(recs, EXAMPLE)
        DECKDIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(tr) + "\n")
    fig_training(tr)
    numbers["training"] = {k: v for k, v in tr.items() if k != "traces"}
    numbers["training"]["final_loss_per_seed"] = {s: v[-1][1] for s, v in tr["traces"].items()}
    numbers["g_folds"] = fig_g_folds(recs)
    numbers["refit"] = fig_refit(res)
    numbers["refit_effect"] = fig_refit_effect(res)
    numbers["controls"] = fig_controls(res)
    fig_loso()
    fig_va()
    numbers["timing_all18"] = res["timing"] | {"runtime_s": res["runtime_s"]}
    numbers["export"] = res["export"]
    DECKDIR.mkdir(parents=True, exist_ok=True)
    (DECKDIR / "method_numbers.json").write_text(json.dumps(numbers, indent=1) + "\n")
    print(json.dumps({k: v for k, v in numbers.items() if k not in ("eq_sizes_in", "data")},
                     indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
