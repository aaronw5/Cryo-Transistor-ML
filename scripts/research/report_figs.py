#!/usr/bin/env python
"""Report figures and tables for the cold-start research loop, from result files only.

Reads finished result JSON (no simulation, no training) and writes, into
``out/research/report/<date>_<topic>/``:

* numbered PNGs (150 dpi): figures and rendered tables, in reading order;
* ``tables.md``: the same tables in markdown;
* ``README.md``: one sentence per figure/table naming its source files, plus
  the skipped inputs and the result files that no figure uses.

It then updates ``out/research/report/INDEX.md`` (newest batch first; state in
``index.json``).  Missing or unreadable result files are skipped and logged,
so the script can be rerun as results arrive.

The only hard-coded numbers are the all-device RRMS of the three reference
cards (released, paper seven-parameter, paper 43-parameter).  NGSpice call
counts come from the summaries, the bank manifests and the run provenance
recorded in each summary/spec.

    PYTHONPATH=src .venv/bin/python scripts/research/report_figs.py \
        [--topic baseline_results] [--date YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cryoml import strict_score as ss  # noqa: E402  (result reader only)

OUT = ROOT / "out"
RESEARCH = OUT / "research"
REPORT_ROOT = RESEARCH / "report"

# The only hard-coded numbers: all-device paper RRMS of the reference cards.
REFERENCES = (("released card", "released", 0.27505, "released"),
              ("paper, 7 parameters", "paper 7", 0.22844, "paper_seven"),
              ("paper, 43 parameters", "paper 43", 0.20380, "paper_full43"))
REF_STYLES = (":", "--", "-.")

# Reference palette (dataviz skill, light mode), categorical slots in order.
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
GRID, BAND, TINT = "#e4e3df", "#ecebe6", "#f0efea"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"

FAMILY_COLOR = {"pipeline": BLUE, "swap": ORANGE, "direct": AQUA}
FAMILY_LABEL = {"pipeline": "Paper pipeline (surrogate search)",
                "swap": "Same search with NGSpice in place of the surrogate",
                "direct": "Direct NGSpice (CMA-ES + TRF)"}

TEMP_ORDER = ["direct", "ngspice_search", "surrogate_paper_mlp", "surrogate_resmlp",
              "surrogate_deeponet", "surrogate_fno", "direct_mlp"]
TEMP_COLOR = {"surrogate_paper_mlp": BLUE, "ngspice_search": ORANGE, "direct": AQUA,
              "surrogate_resmlp": VIOLET, "surrogate_deeponet": MAGENTA,
              "surrogate_fno": GREEN, "direct_mlp": RED}
TEMP_SPARE = [YELLOW, INK2, MUTED]
ARCH_LABEL = {"paper_mlp": "paper MLP", "resmlp": "ResMLP", "deeponet": "DeepONet",
              "fno": "FNO", "paper_mlp_wide": "wide MLP", "resmlp_large": "large ResMLP"}
TEMP_LABEL = {"direct": "Direct NGSpice (CMA-ES + TRF)",
              "ngspice_search": "Pipeline search on NGSpice",
              "direct_mlp": "Direct MLP, one forward pass"}

DPI = 150
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
    "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})


# ---------------------------------------------------------------- logging/io

class Log:
    def __init__(self) -> None:
        self.skipped: list[str] = []
        self.used: set[str] = set()
        self.notes: list[str] = []

    def skip(self, what: str, why: str) -> None:
        msg = f"{what}: {why}"
        if msg not in self.skipped:
            print(f"[skip] {msg}", flush=True)
            self.skipped.append(msg)

    def note(self, msg: str) -> None:
        if msg not in self.notes:
            print(f"[note] {msg}", flush=True)
            self.notes.append(msg)


LOG = Log()


def rel(path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def read_json(path, what: str | None = None, mark: bool = True):
    p = ROOT / rel(path)
    if not p.is_file():
        LOG.skip(what or rel(path), f"{rel(path)} not found")
        return None
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        LOG.skip(what or rel(path), f"unreadable ({exc})")
        return None
    if mark:
        LOG.used.add(rel(p))
    return data


def num(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x) if math.isfinite(x) else None


def family_of(tag: str) -> str:
    return "nmos" if tag.startswith("nmos") else "pmos"


def family_stats(per_device: dict) -> dict:
    """all-device mean, family means and combined (mean of the two family means)."""
    vals = {t: v for t, v in per_device.items() if v is not None}
    n = [v for t, v in vals.items() if family_of(t) == "nmos"]
    p = [v for t, v in vals.items() if family_of(t) == "pmos"]
    out = {"all": float(np.mean(list(vals.values()))) if vals else None,
           "nmos": float(np.mean(n)) if n else None,
           "pmos": float(np.mean(p)) if p else None}
    out["combined"] = (out["nmos"] + out["pmos"]) / 2 if n and p else None
    return out


def device_key(tag: str):
    m = re.match(r"(nmos|pmos)_L([\dp]+)_W([\dp]+)", tag)
    if not m:
        return (2, 0.0, 0.0, tag)
    f = lambda s: float(s.replace("p", "."))  # noqa: E731
    return (0 if m.group(1) == "nmos" else 1, f(m.group(2)), f(m.group(3)), tag)


def device_label(tag: str) -> str:
    m = re.match(r"(nmos|pmos)_L([\dp]+)_W([\dp]+)", tag)
    if not m:
        return tag
    return f"L{m.group(2).replace('p', '.')} W{m.group(3).replace('p', '.')}"


def kfmt(v) -> str:
    if v is None:
        return "n/a"
    return f"{v / 1000:.1f}k" if v >= 1000 else f"{v:.0f}"


def log_ticks(ax, axis: str, lo: float, hi: float) -> None:
    """1-2-5 ticks with plain labels on a log axis spanning at most ~3 decades."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ticks = [m * 10.0 ** e for e in range(-7, 3) for m in (1, 2, 5) if lo <= m * 10.0 ** e <= hi]
    if len(ticks) < 2 or hi / lo > 2e3:
        return
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(FixedLocator(ticks))
    target.set_minor_locator(NullLocator())
    target.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))


def n_params_of_spec(spec_path) -> int | None:
    spec = read_json(spec_path, f"spec {spec_path}", mark=False)
    if spec is None:
        return None
    names = spec.get("selected_parameters")
    return len(names) if names else None


# ------------------------------------------------------------ result loaders

_BANK: dict[str, float | None] = {}


def bank_mean_samples(bank) -> float | None:
    """Mean NGSpice bank simulations per device (manifest ``n_samples``)."""
    if not bank:
        return 0.0
    key = rel(bank)
    if key not in _BANK:
        m = read_json(Path(key) / "manifest.json", f"bank manifest {key}")
        vals = [d.get("n_samples") for d in (m or {}).get("devices", [])
                if d.get("n_samples") is not None]
        _BANK[key] = float(np.mean(vals)) if vals else None
    return _BANK[key]


def _add(*xs):
    return None if any(x is None for x in xs) else float(sum(xs))


_PIPE: dict = {}


def pipeline_round(pset: str, rnd: int):
    key = (pset, rnd)
    if key in _PIPE:
        return _PIPE[key]
    path = OUT / "cold_loop" / pset / f"round_{rnd}" / "extract" / "summary.json"
    d = read_json(path, f"pipeline {pset} round {rnd}")
    res = None
    if d is not None:
        m = d.get("methods", {}).get("surrogate_fd")
        if m is None:
            LOG.skip(f"pipeline {pset} round {rnd}", "summary has no surrogate_fd method")
        else:
            per_device, n_val, status = {}, [], {}
            for dev in d.get("devices", []):
                rec = dev.get("methods", {}).get("surrogate_fd", {})
                per_device[dev["device"]] = num(rec.get("rrms"))
                status[dev["device"]] = ss.record_status(rec, dev["device"])
                if rec.get("n_validated") is not None:
                    n_val.append(rec["n_validated"])
            bank_dir = f"data/processed/cold_loop/{pset}_r{rnd}"
            bank = bank_mean_samples(bank_dir)
            run = _add(bank, float(np.mean(n_val))) if n_val else None
            res = {"family": "pipeline", "n_params": d.get("n_parameters"),
                   "round": rnd, "all": num(m.get("all_device_mean_rrms")),
                   "combined": num(m.get("combined_rrms")), "nmos": num(m.get("nmos_rrms")),
                   "pmos": num(m.get("pmos_rrms")), "per_device": per_device,
                   "run_calls": run, "source": rel(path),
                   "sources": [rel(path), f"{bank_dir}/manifest.json"],
                   "cards_key": (rel(path.parent), "surrogate_fd"), "fd": None,
                   "strict": ss.aggregate(status, per_device)}
    _PIPE[key] = res
    return res


def pipeline(pset: str, rnd: int):
    res = pipeline_round(pset, rnd)
    if res is None:
        return None
    prior = [pipeline_round(pset, k) for k in range(1, rnd + 1)]
    res["cum_calls"] = (None if any(p is None for p in prior)
                        else _add(*[p["run_calls"] for p in prior]))
    attach_fd(res)
    return res


_DIRECT: dict = {}


def direct_run(name: str):
    """Direct NGSpice summary; cumulative calls follow the recorded provenance
    (bank used for starts, ``options.init`` seed run, spec ``source_run``)."""
    if name in _DIRECT:
        return _DIRECT[name]
    _DIRECT[name] = None
    path = RESEARCH / "ngspice_direct" / name / "summary.json"
    d = read_json(path, f"direct NGSpice {name}")
    if d is None:
        return None
    opts = d.get("options", {})
    spec = read_json(d.get("spec", ""), f"spec of {name}", mark=False) or {}
    per_device = {t: num(v.get("rrms")) for t, v in d.get("devices", {}).items()}
    status = {t: ss.record_status(v, t) for t, v in d.get("devices", {}).items()}
    run = num(d.get("mean_ngspice_calls_per_device"))
    sources = [rel(path)]
    cum = _add(run, bank_mean_samples(opts.get("bank")))
    if opts.get("bank"):
        sources.append(f"{rel(opts['bank'])}/manifest.json")
    init = opts.get("init")
    if init:
        p = Path(rel(init))
        base = Path("out/research/ngspice_direct")
        if p.name == "summary.json" and p.parent.parent == base:
            prev = direct_run(p.parent.name)
            cum = _add(cum, prev["cum_calls"] if prev else None)
        else:
            LOG.note(f"direct NGSpice {name}: seed {rel(init)} is not a direct-NGSpice run; "
                     "cumulative calls not computed")
            cum = None
    source_run = spec.get("source_run")
    if source_run:
        prev = direct_run(source_run)
        cum = _add(cum, prev["cum_calls"] if prev else None)
    sel = spec.get("selected_parameters")
    res = {"family": "direct", "name": name, "n_params": len(sel) if sel else None,
           "round": spec.get("round", 1), "source_run": source_run, "init": init,
           "all": num(d.get("all_device_mean_rrms")), "combined": num(d.get("combined_rrms")),
           "nmos": num(d.get("nmos_rrms")), "pmos": num(d.get("pmos_rrms")),
           "per_device": per_device, "run_calls": run, "cum_calls": cum,
           "source": rel(path), "sources": sources,
           "cards_key": (rel(path), None), "fd": None,
           "strict": ss.aggregate(status, per_device)}
    _DIRECT[name] = res
    attach_fd(res)
    return res


def swap_run(name: str):
    path = RESEARCH / "swap_search" / name / "summary.json"
    d = read_json(path, f"NGSpice swap {name}")
    if d is None:
        return None
    m = re.search(r"round_(\d+)", d.get("round_dir", ""))
    run = num(d.get("mean_ngspice_calls_per_device"))
    per_device = {t: num(v.get("rrms")) for t, v in d.get("devices", {}).items()}
    res = {"family": "swap", "name": name,
           "n_params": n_params_of_spec(Path(d.get("round_dir", "")) / "spec.json"),
           "round": int(m.group(1)) if m else None,
           "all": num(d.get("all_device_mean_rrms")), "combined": num(d.get("combined_rrms")),
           "nmos": num(d.get("nmos_rrms")), "pmos": num(d.get("pmos_rrms")),
           "per_device": per_device,
           "run_calls": run, "cum_calls": _add(run, bank_mean_samples(d.get("bank"))),
           "source": rel(path),
           "sources": [rel(path)] + ([f"{rel(d['bank'])}/manifest.json"] if d.get("bank") else []),
           "cards_key": (rel(path.parent), "surrogate_fd"), "fd": None,
           "strict": ss.aggregate(records_status(path.parent, "surrogate_fd", per_device),
                                  per_device)}
    attach_fd(res)
    return res


def records_status(directory: Path, method: str, per_device: dict) -> dict:
    """Strict status per device from a run's ``ml_<device>.json`` records
    (frozen-inclusion per_curve); devices without a record are unknown."""
    status = {}
    for tag in per_device:
        f = directory / f"ml_{tag}.json"
        rec = None
        if f.is_file():
            try:
                rec = json.loads(f.read_text())["methods"][method]
            except (OSError, ValueError, KeyError):
                rec = None
        status[tag] = (ss.record_status(rec, tag) if rec is not None
                       else ss.device_status(None, ss.include_tags(tag)))
    if any(s["status"] == ss.UNKNOWN for s in status.values()):
        LOG.note(f"{rel(directory)}: some devices have no {method} record; "
                 "their curve status is unknown")
    return status


def n_failed(res: dict | None):
    return None if not res or "strict" not in res else res["strict"]["n_failed_devices"]


def failed_text(res: dict) -> str:
    st = res["strict"]
    return ", ".join(f"{t} ({st['n_failed_curves'][t]} curve"
                     f"{'s' if st['n_failed_curves'][t] != 1 else ''})"
                     for t in st["failed_devices"])


_POLISHED: dict | None = None


def polished_index() -> dict:
    """Map (cards path, method) -> FD-polished real-data summary."""
    global _POLISHED
    if _POLISHED is not None:
        return _POLISHED
    _POLISHED = {}
    for p in sorted((RESEARCH / "polished").glob("*/real/summary.json")):
        d = read_json(p, mark=False)
        if d is None or not d.get("cards"):
            continue
        if d.get("target_override"):
            continue
        key = (rel(d["cards"]), d.get("method"))
        if key in _POLISHED:
            old = _POLISHED[key][0]
            newer = p if p.stat().st_mtime >= old.stat().st_mtime else old
            LOG.note(f"two FD summaries for {key[0]} ({rel(old)}, {rel(p)}); "
                     f"using the newer file {rel(newer)}")
            if newer == old:
                continue
        _POLISHED[key] = (p, d)
    return _POLISHED


def attach_fd(res: dict) -> None:
    if res.get("fd_checked"):
        return
    res["fd_checked"] = True
    hit = polished_index().get(res["cards_key"])
    if not hit:
        return
    p, d = hit
    LOG.used.add(rel(p))
    devices = d.get("devices", {})
    fd_per_device = {t: num(v.get("rrms")) for t, v in devices.items()}
    res["fd"] = {"all": num(d.get("all_device_mean_rrms")), "combined": num(d.get("combined_rrms")),
                 "nmos": num(d.get("nmos_rrms")), "pmos": num(d.get("pmos_rrms")),
                 "per_device": fd_per_device,
                 "calls": num(d.get("mean_ngspice_calls_per_device")), "source": rel(p),
                 # polish summaries carry no per_curve: the status comes from
                 # failed_curves or the penalized objective history[-1] vs rrms.
                 "strict": ss.aggregate({t: ss.record_status(v, t) for t, v in devices.items()},
                                        fd_per_device)}
    res["sources"].append(rel(p))
    diffs = []
    for tag, v in sorted(devices.items(), key=lambda kv: device_key(kv[0])):
        before, own = num(v.get("before_rrms")), res["per_device"].get(tag)
        if before is not None and own is not None and abs(before - own) > 1e-6 * max(1.0, abs(own)):
            diffs.append(f"{tag} {own:.4f} vs {before:.4f}")
    if diffs:
        res["fd"]["start_mismatch"] = diffs
        LOG.note(f"{rel(p)}: FD start re-score differs from the method's own score on "
                 f"{len(diffs)} device(s): " + "; ".join(diffs)
                 + " (method's own score shown as raw; polish summaries without "
                 "before_failed_curves stored the penalized start objective, where each failed "
                 "included curve costs " f"{ss.PENALTY:g})")


# ------------------------------------------------------------------- tables

class Table:
    """columns: dicts with key, head, group, md, kind ('text'|'rrms'|'int'), family."""

    def __init__(self, name, caption, columns, rows, notes=()):
        self.name, self.caption, self.columns = name, caption, columns
        self.rows, self.notes = rows, list(notes)


def fmt(v, kind: str) -> str:
    if kind in ("text", "rtext"):     # rtext: preformatted, right-aligned
        return "" if v is None else str(v)
    v = num(v)
    if v is None:
        return "—"
    if kind == "int":
        return f"{v:,.0f}"
    if v != 0 and abs(v) < 1e-4:
        return f"{v:.1e}"
    return f"{v:.4f}"


def table_markdown(t: Table) -> str:
    lines = [f"### {t.caption}", ""]
    lines.append("| " + " | ".join(c["md"] for c in t.columns) + " |")
    lines.append("|" + "|".join("---" if c["kind"] == "text" else "---:" for c in t.columns) + "|")
    for r in t.rows:
        lines.append("| " + " | ".join(fmt(r.get(c["key"]), c["kind"]) for c in t.columns) + " |")
    if t.notes:
        lines.append("")
        lines += [f"- {n}" for n in t.notes]
    return "\n".join(lines) + "\n"


def render_table(t: Table, path: Path) -> None:
    cols = t.columns
    cells = [[fmt(r.get(c["key"]), c["kind"]) for c in cols] for r in t.rows]
    widths = [max([len(c["head"])] + [len(row[i]) for row in cells]) + 3
              for i, c in enumerate(cols)]
    spans = []
    for i, c in enumerate(cols):
        g = c.get("group") or ""
        if spans and g and spans[-1][0] == g:
            spans[-1][2] = i
        else:
            spans.append([g, i, i])
    for g, a, b in spans:
        need = len(g) + 3 - sum(widths[a:b + 1])
        if g and need > 0:
            widths[b] += need
    ch, row_h, pad = 0.092, 0.31, 0.25
    wrap_chars = max(60, int(sum(widths) * 1.15))
    notes = [line for n in t.notes for line in textwrap.wrap(n, wrap_chars)]
    width_in = sum(widths) * ch + 2 * pad
    caption_lines = textwrap.wrap(t.caption, int(width_in / 0.105))
    head_y0 = pad + 0.28 * len(caption_lines) + 0.15
    body_y0 = head_y0 + 2 * row_h
    height_in = body_y0 + row_h * len(cells) + 0.12 + 0.22 * len(notes) + pad
    fig = plt.figure(figsize=(width_in, height_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_in)
    ax.set_ylim(height_in, 0)
    ax.axis("off")
    for k, line in enumerate(caption_lines):
        ax.text(pad, pad + 0.28 * k, line, fontsize=12, fontweight="bold", va="top", color=INK)
    xs = np.concatenate([[pad], pad + np.cumsum(widths) * ch])
    table_bottom = body_y0 + row_h * len(cells)
    for i, c in enumerate(cols):
        if c.get("family"):
            ax.add_patch(Rectangle((xs[i], head_y0), xs[i + 1] - xs[i], table_bottom - head_y0,
                                   facecolor=TINT, edgecolor="none", zorder=0))
    for g, a, b in spans:
        if not g:
            continue
        x0, x1 = xs[a] + 0.3 * ch, xs[b + 1] - 0.3 * ch
        ax.text((x0 + x1) / 2, head_y0 + row_h * 0.55, g, ha="center", va="center",
                fontsize=10.5, color=INK2)
        ax.plot([x0, x1], [head_y0 + row_h * 0.92] * 2, color=MUTED, lw=0.8)

    def put(i, y, s, **kw):
        if cols[i]["kind"] == "text":  # rtext and numbers: right-aligned
            ax.text(xs[i] + 0.8 * ch, y, s, ha="left", va="center", **kw)
        else:
            ax.text(xs[i + 1] - 1.2 * ch, y, s, ha="right", va="center", **kw)

    for i, c in enumerate(cols):
        put(i, head_y0 + row_h * 1.5, c["head"], fontsize=10.5, fontweight="bold", color=INK)
    ax.plot([pad, xs[-1]], [head_y0 - 0.02] * 2, color=INK2, lw=1.0)
    ax.plot([pad, xs[-1]], [body_y0] * 2, color=INK2, lw=0.9)
    for r, row in enumerate(cells):
        y = body_y0 + row_h * (r + 0.5)
        for i, s in enumerate(row):
            put(i, y, s, fontsize=10.5, color=INK if s not in ("—", "–") else MUTED)
        if r < len(cells) - 1:
            ax.plot([pad, xs[-1]], [body_y0 + row_h * (r + 1)] * 2, color=GRID, lw=0.6)
    ax.plot([pad, xs[-1]], [table_bottom] * 2, color=INK2, lw=1.0)
    for k, line in enumerate(notes):
        ax.text(pad, table_bottom + 0.14 + 0.22 * k, line, fontsize=9, color=INK2, va="top")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


RRMS_COLS = [
    {"key": "all", "head": "all-device", "group": "mean RRMS, 18 devices", "md": "all-device", "kind": "rrms"},
    {"key": "combined", "head": "combined", "group": "mean RRMS, 18 devices", "md": "combined", "kind": "rrms"},
    {"key": "nmos", "head": "nMOS", "group": "family mean RRMS", "md": "nMOS", "kind": "rrms", "family": True},
    {"key": "pmos", "head": "pMOS", "group": "family mean RRMS", "md": "pMOS", "kind": "rrms", "family": True},
    {"key": "run", "head": "this run", "group": "NGSpice calls / device", "md": "NGSpice calls/device (this run)", "kind": "int"},
    {"key": "cum", "head": "cumulative", "group": "NGSpice calls / device", "md": "NGSpice calls/device (cumulative)", "kind": "int"},
]
CALLS_NOTE = ("NGSpice calls per device: 'this run' is the run's own count (pipeline: bank "
              "simulations of that round plus NGSpice validation of candidates; +FD rows: FD polish "
              "only). 'cumulative' adds every bank used for starts or training and every earlier "
              "round or seed run named in the summary/spec provenance.")
COMBINED_NOTE = ("all-device = mean over 18 devices; combined = mean of the nMOS (8) and pMOS (10) "
                 "family means.")
STRICT_COLS = [
    {"key": "failed", "head": "failed dev.", "group": "strict check (all-device)",
     "md": "failed devices", "kind": "int"},
    {"key": "valid_all", "head": "valid-only", "group": "strict check (all-device)",
     "md": "all-device, valid devices only", "kind": "rrms"},
    {"key": "pen_all", "head": "penalized", "group": "strict check (all-device)",
     "md": "all-device, penalized", "kind": "rrms"},
]
STRICT_NOTE = ("failed dev. = devices with an included (frozen-inclusion) metric curve that failed "
               "to simulate; their stored RRMS averages only the surviving curves, so it is not a "
               "valid paper RRMS (upstream rrmsCalc.py gives NaN). The all-device/combined/family "
               "columns are the stored values; valid-only = all-device mean over the valid devices; "
               f"penalized = every device, failed curves at RRMS {ss.PENALTY:g} (ngspice_fit.PENALTY; "
               "penalized convention, not the paper metric). Source: cryoml.strict_score.")


def strict_cells(strict: dict | None) -> dict:
    if not strict:
        return {"failed": None, "valid_all": None, "pen_all": None}
    return {"failed": strict["n_failed_devices"],
            "valid_all": num(strict["valid"]["all_device_mean_rrms"]),
            "pen_all": num(strict["penalized"]["all_device_mean_rrms"])}


def strict_notes(label: str, res: dict) -> list[str]:
    notes = []
    for stage, r in (("method output", res), ("+ NGSpice FD", res.get("fd"))):
        st = (r or {}).get("strict")
        if st and st["n_failed_devices"]:
            notes.append(
                f"{label} ({stage}): {st['n_failed_devices']} failed device(s): "
                + ", ".join(f"{t} ({st['n_failed_curves'][t]} included curve(s) non-finite)"
                            for t in st["failed_devices"])
                + f"; valid-only all-device {fmt(st['valid']['all_device_mean_rrms'], 'rrms')} "
                f"over {st['n_valid_devices']} devices, combined "
                f"{fmt(st['valid']['combined_rrms'], 'rrms')}; penalized all-device "
                f"{fmt(st['penalized']['all_device_mean_rrms'], 'rrms')}.")
    return notes


def stage_rows(label: str, res: dict, extra: dict | None = None) -> list[dict]:
    extra = extra or {}
    rows = [{"method": label, "stage": "method output", **extra,
             **{k: res.get(k) for k in ("all", "combined", "nmos", "pmos")},
             "run": res.get("run_calls"), "cum": res.get("cum_calls"),
             **strict_cells(res.get("strict"))}]
    fd = res.get("fd")
    if fd:
        rows.append({"method": label, "stage": "+ NGSpice FD", **extra,
                     **{k: fd.get(k) for k in ("all", "combined", "nmos", "pmos")},
                     "run": fd.get("calls"), "cum": _add(res.get("cum_calls"), fd.get("calls")),
                     **strict_cells(fd.get("strict"))})
    return rows


FAILED_MARK = dict(marker="x", s=46, color=RED, linewidths=1.8, zorder=6)


def failed_handle(text: str = "red ×: the run has devices with a failed included curve "
                  "(stored mean drops those curves; see table)") -> Line2D:
    return Line2D([], [], ls="none", marker="x", ms=8, mew=1.8, color=RED, label=text)


# ------------------------------------------------------------- real methods

REAL_SPEC = [
    ("Pipeline 7, round 1", lambda: pipeline("seven", 1)),
    ("Pipeline 7, round 4", lambda: pipeline("seven", 4)),
    ("Pipeline 43, round 1", lambda: pipeline("full", 1)),
    ("Pipeline 43, round 4", lambda: pipeline("full", 4)),
    ("NGSpice swap 7, round 1", lambda: swap_run("seven_round_1")),
    ("NGSpice swap 43, round 1", lambda: swap_run("full_round_1")),
    ("Direct NGSpice 7, round 1", lambda: direct_run("seven_r1_direct")),
    ("Direct NGSpice 43, round 1, cold", lambda: direct_run("full_r1_direct_cold")),
    ("Direct NGSpice 43, round 1, seeded from 7", lambda: direct_run("full_r1_direct_seeded")),
    ("Direct NGSpice 43, round 3", lambda: direct_run("full_r3_direct")),
]


def reference_rows() -> list[dict]:
    ref = read_json(RESEARCH / "reference_scores.json", "reference scores (table rows)")
    rows = []
    for label, _short, value, key in REFERENCES:
        r = (ref or {}).get(key, {})
        devs = r.get("devices", {})
        strict = (ss.aggregate({t: ss.record_status(v, t) for t, v in devs.items()},
                               {t: num(v.get("rrms")) for t, v in devs.items()})
                  if devs else None)
        rows.append({"method": f"Reference: {label}", "stage": "published card",
                     "all": value, "combined": num(r.get("combined_rrms")),
                     "nmos": num(r.get("nmos_rrms")), "pmos": num(r.get("pmos_rrms")),
                     "run": None, "cum": None, **strict_cells(strict)})
        own = num(r.get("all_device_mean_rrms"))
        if own is not None and abs(own - value) > 5e-6:
            LOG.note(f"reference_scores.json {key} all-device {own:.5f} differs from {value}")
    return rows


def fig_real_methods(rows: list[tuple[str, dict]], path: Path) -> list[str]:
    ys, y, prev = [], 0.0, None
    for _label, res in rows:
        if prev is not None and res["family"] != prev:
            y += 0.6
        ys.append(y)
        y += 1.0
        prev = res["family"]
    values = [REFERENCES[i][2] for i in range(3)]
    for _l, r in rows:
        for k in ("all", "combined"):
            values += [v for v in (r[k], (r["fd"] or {}).get(k)) if v is not None]
    lo, hi = min(values), max(values)
    span = hi - lo
    xlim = (lo - 0.08 * span, hi + 0.06 * span)

    fig = plt.figure(figsize=(14.5, 0.46 * (ys[-1] + 1) + 3.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.62], wspace=0.07,
                          left=0.235, right=0.995, top=1 - 1.25 / fig.get_figheight(),
                          bottom=1.45 / fig.get_figheight())
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    axt = fig.add_subplot(gs[2], sharey=ax1)
    for ax, key in ((ax1, "all"), (ax2, "combined")):
        ax.grid(axis="x", zorder=0)
        for (_label, r), yy in zip(rows, ys):
            c = FAMILY_COLOR[r["family"]]
            raw, fd = r[key], (r["fd"] or {}).get(key)
            if raw is not None and fd is not None:
                ax.plot([raw, fd], [yy, yy], color=c, lw=1.6, alpha=0.5, zorder=2)
            if raw is not None:
                ax.scatter(raw, yy, s=72, facecolor=SURFACE, edgecolor=c, linewidth=2, zorder=3)
            if fd is not None:
                ax.scatter(fd, yy, s=72, facecolor=c, edgecolor=SURFACE, linewidth=1.5, zorder=4)
            if raw is not None and n_failed(r):
                ax.scatter(raw, yy, **FAILED_MARK)
            if fd is not None and n_failed(r["fd"]):
                ax.scatter(fd, yy, **FAILED_MARK)
        ax.set_xlim(*xlim)
    for (label, short, value, _key), ls in zip(REFERENCES, REF_STYLES):
        ax1.axvline(value, color=INK2, lw=1.1, ls=ls, zorder=1)
        ax1.text(value, 0.01, f"{short} {value:.4f} ", transform=ax1.get_xaxis_transform(),
                 ha="right", va="bottom", rotation=90, fontsize=9, color=INK2)
    ax1.set_yticks(ys)
    ax1.set_yticklabels([label + (" †" if n_failed(r) or n_failed(r["fd"]) else "")
                         for label, r in rows], fontsize=10.5, color=INK)
    ax1.set_ylim(ys[-1] + 0.7, ys[0] - 0.7)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.tick_params(axis="y", length=0)
    ax1.set_xlabel("all-device mean RRMS (18 devices)")
    ax2.set_xlabel("combined RRMS (mean of nMOS and pMOS family means)")
    ax1.set_title("All-device convention", loc="left", pad=10)
    ax2.set_title("Combined convention (no reference lines)", loc="left", pad=10)
    axt.axis("off")
    axt.text(0.03, ys[0] - 0.7, "NGSpice calls per device", ha="left", va="bottom",
             fontsize=11, fontweight="bold", transform=axt.get_yaxis_transform())
    for (_label, r), yy in zip(rows, ys):
        s = f"run {kfmt(r['run_calls'])} · cum. {kfmt(r['cum_calls'])}"
        if r["fd"]:
            s += f" · FD +{kfmt(r['fd']['calls'])}"
        axt.text(0.03, yy, s, ha="left", va="center", fontsize=10, color=INK2,
                 transform=axt.get_yaxis_transform())
    fam_present = []
    for _l, r in rows:
        if r["family"] not in fam_present:
            fam_present.append(r["family"])
    handles = [Line2D([], [], ls="none", marker="o", ms=9, mfc=FAMILY_COLOR[f], mec=FAMILY_COLOR[f],
                      label=FAMILY_LABEL[f]) for f in fam_present]
    handles += [Line2D([], [], ls="none", marker="o", ms=9, mfc=SURFACE, mec=INK2, mew=2,
                       label="open: method output"),
                Line2D([], [], ls="none", marker="o", ms=9, mfc=INK2, mec=INK2,
                       label="filled: after NGSpice FD polish")]
    if any(n_failed(r) or n_failed(r["fd"]) for _l, r in rows):
        handles.append(failed_handle("† / red ×: devices with a failed included curve "
                                     "(stored mean drops it; see table)"))
    fig.legend(handles=handles, loc="lower left", ncol=3, bbox_to_anchor=(0.01, 0.0),
               fontsize=10, columnspacing=1.6, handletextpad=0.4)
    fig.text(0.01, 1 - 0.18 / fig.get_figheight(),
             "Paper RRMS on the 18 measured 77 K devices, by method", fontsize=14,
             fontweight="bold", va="top")
    fig.text(0.01, 1 - 0.5 / fig.get_figheight(),
             "Vertical lines (left panel only): all-device RRMS of the reference cards. "
             "Calls: run = this run; cum. = including banks and earlier rounds or seed runs; "
             "FD = polish calls.", fontsize=9.5, color=INK2, va="top")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return sorted({s for _l, r in rows for s in r["sources"]})


# ------------------------------------------------------------------ rounds

def direct43_chain() -> list[dict]:
    names = []
    for p in sorted((RESEARCH / "ngspice_direct").glob("full_r*_direct/summary.json")):
        m = re.match(r"full_r(\d+)_direct$", p.parent.name)
        if m:
            names.append((int(m.group(1)), p.parent.name))
    if not names:
        LOG.skip("direct NGSpice 43 rounds", "no out/research/ngspice_direct/full_r*_direct runs")
        return []
    chain, name = [], max(names)[1]
    while name:
        r = direct_run(name)
        if r is None:
            break
        chain.insert(0, r)
        name = r["source_run"]
    for k, r in enumerate(chain, start=1):
        if r["round"] != k:
            LOG.note(f"direct NGSpice chain position {k} is {r['name']} with spec round {r['round']}")
    return chain


def fig_rounds(series: list[dict], swaps: list[dict], path: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    fig.subplots_adjust(left=0.075, right=0.755, top=0.84, bottom=0.11)
    ax.grid(axis="y", zorder=0)
    sources = []
    for s in series:
        pts = [(r["round"] + s["dx"], r["all"]) for r in s["results"] if r["all"] is not None]
        if not pts:
            continue
        x, yv = zip(*pts)
        ax.plot(x, yv, color=s["color"], ls=s["ls"], lw=2, zorder=2)
        ax.scatter(x, yv, s=70, facecolor=s["color"] if s["filled"] else SURFACE,
                   edgecolor=s["color"] if not s["filled"] else SURFACE,
                   linewidth=2 if not s["filled"] else 1.5, marker=s["marker"], zorder=3,
                   label=s["label"])
        ax.annotate(s["label"], (x[-1], yv[-1]), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=10, color=INK)
        for r in s["results"]:
            if r["all"] is not None and n_failed(r):
                ax.scatter(r["round"] + s["dx"], r["all"], **FAILED_MARK)
                ax.annotate(f"{n_failed(r)} failed", (r["round"] + s["dx"], r["all"]),
                            xytext=(6, -12), textcoords="offset points", fontsize=8.5,
                            color=RED)
        sources += [src for r in s["results"] for src in r["sources"] if "polished" not in src]
    for r, dx in swaps:
        if r["all"] is None or r["round"] is None:
            continue
        filled = r["n_params"] != 7
        ax.scatter(r["round"] + dx, r["all"], s=95, marker="D", zorder=4,
                   facecolor=ORANGE if filled else SURFACE,
                   edgecolor=SURFACE if filled else ORANGE, linewidth=2,
                   label=f"NGSpice swap {r['n_params']}, round {r['round']}")
        if n_failed(r):
            ax.scatter(r["round"] + dx, r["all"], **FAILED_MARK)
        sources += [src for src in r["sources"] if "polished" not in src]
    for (label, short, value, _key), ls in zip(REFERENCES, REF_STYLES):
        ax.axhline(value, color=INK2, lw=1.0, ls=ls, zorder=1)
        ax.text(1.0, value, f" {short} {value:.4f}", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=9.5, color=INK2)
    rounds = sorted({r["round"] for s in series for r in s["results"]}
                    | {r["round"] for r, _dx in swaps if r["round"]})
    ax.set_xticks(rounds)
    ax.set_xlim(min(rounds) - 0.35, max(rounds) + 0.55)
    ax.set_xlabel("round (each round re-centers every device's box on the previous selected card)")
    ax.set_ylabel("all-device mean RRMS (18 devices)")
    handles, labels = ax.get_legend_handles_labels()
    if any(n_failed(r) for s in series for r in s["results"]) or any(
            n_failed(r) for r, _dx in swaps):
        handles.append(failed_handle("red ×: devices with failed included\ncurves (count annotated)"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 0.55), fontsize=10,
              handletextpad=0.4)
    fig.text(0.01, 0.97, "All-device RRMS by round, measured 77 K data", fontsize=14,
             fontweight="bold", va="top")
    fig.text(0.01, 0.92, "Raw method output (no FD polish). Direct NGSpice 43 round 1 is the run "
             "the round-2 box was centered on. Horizontal lines: reference cards.",
             fontsize=9.5, color=INK2, va="top")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return sorted(set(sources))


# ----------------------------------------------------------- surrogate gap

def identify_run(per_device: dict, candidates: dict) -> str | None:
    for name, res in candidates.items():
        if res is None:
            continue
        common = [t for t in per_device if res["per_device"].get(t) is not None]
        if len(common) == len(per_device) and all(
                abs(per_device[t] - res["per_device"][t]) <= 1e-9 * max(1, abs(per_device[t]))
                for t in common):
            return name
    return None


def fig_surrogate_gap(path: Path):
    src = RESEARCH / "surrogate_gap_full_round_1.json"
    g = read_json(src, "surrogate gap (round 1, 43 parameters)")
    if g is None:
        return None
    devs = g.get("devices", {})
    pts = {"pipeline": {}, "reference": {}}
    for tag, v in devs.items():
        for k in pts:
            s, n = num(v.get(k, {}).get("surrogate_rrms")), num(v.get(k, {}).get("ngspice_rrms"))
            if s is not None and n is not None:
                pts[k][tag] = (s, n)
    direct_names = sorted(p.parent.name for p in (RESEARCH / "ngspice_direct").glob("*/summary.json"))
    used_before = set(LOG.used)  # identification reads candidates; only the match counts as used
    notes_before = list(LOG.notes)
    ref_name = identify_run({t: v[1] for t, v in pts["reference"].items()},
                            {n: direct_run(n) for n in direct_names})
    pipe_name = identify_run({t: v[1] for t, v in pts["pipeline"].items()},
                             {f"full round {k}": pipeline("full", k) for k in (1, 2, 3, 4)})
    LOG.used = used_before
    LOG.notes = notes_before
    ref_label = (f"Direct-fit reference cards ({ref_name})" if ref_name
                 else "Direct-fit reference cards (source run not identified)")
    pipe_label = (f"Pipeline picks (pipeline 43, {pipe_name.replace('full ', '')})" if pipe_name
                  else "Pipeline picks (source round not identified)")
    if not ref_name:
        LOG.note("surrogate gap: reference cards do not match any direct-NGSpice summary")
    fig, ax = plt.subplots(figsize=(8.8, 8.6))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.86, bottom=0.2)
    allv = [x for k in pts for v in pts[k].values() for x in v]
    lo, hi = min(allv) / 1.4, max(allv) * 1.4
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1.0, zorder=1)
    ax.text(hi / 2.2, hi / 2.2 * 1.08, "y = x", ha="right", va="bottom", fontsize=10,
            color=INK2, rotation=45, rotation_mode="anchor")
    for tag in pts["pipeline"]:
        if tag in pts["reference"]:
            (x0, y0), (x1, y1) = pts["pipeline"][tag], pts["reference"][tag]
            ax.plot([x0, x1], [y0, y1], color=MUTED, lw=0.7, alpha=0.6, zorder=1)
    for k, color in (("pipeline", BLUE), ("reference", AQUA)):
        for fam, marker in (("nmos", "o"), ("pmos", "^")):
            xy = [v for t, v in pts[k].items() if family_of(t) == fam]
            if xy:
                x, yv = zip(*xy)
                ax.scatter(x, yv, s=70, marker=marker, facecolor=color, edgecolor=SURFACE,
                           linewidth=1.5, zorder=3)
    gap_failed = []
    for k, run in (("pipeline", pipeline("full", int(pipe_name.split()[-1]))
                    if pipe_name else None),
                   ("reference", direct_run(ref_name) if ref_name else None)):
        bad = [t for t in ((run or {}).get("strict") or {}).get("failed_devices", [])
               if t in pts[k]]
        gap_failed += [f"{k} {t}" for t in bad]
        if bad:
            ax.scatter([pts[k][t][0] for t in bad], [pts[k][t][1] for t in bad], **FAILED_MARK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    log_ticks(ax, "x", lo, hi)
    log_ticks(ax, "y", lo, hi)
    ax.grid(which="major", zorder=0)
    ax.set_xlabel("surrogate-predicted RRMS (round-1 surrogate)")
    ax.set_ylabel("NGSpice RRMS (re-simulated)")
    handles = [Line2D([], [], ls="none", marker="s", ms=10, mfc=BLUE, mec=BLUE, label=pipe_label),
               Line2D([], [], ls="none", marker="s", ms=10, mfc=AQUA, mec=AQUA, label=ref_label),
               Line2D([], [], ls="none", marker="o", ms=8, mfc=INK2, mec=INK2, label="nMOS device"),
               Line2D([], [], ls="none", marker="^", ms=8, mfc=INK2, mec=INK2, label="pMOS device"),
               Line2D([], [], color=MUTED, lw=0.9, label="same device")]
    if gap_failed:
        handles.append(failed_handle("red ×: NGSpice RRMS drops failed included curve(s)"))
        LOG.note("surrogate gap: NGSpice RRMS of " + ", ".join(gap_failed)
                 + " averages only the finite included curves (failed curve(s); not a valid "
                 "paper RRMS)")
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, 0.0), ncol=2, fontsize=10)
    s = g.get("summary", {})
    fig.text(0.02, 0.975, "Round-1 43-parameter surrogate: predicted vs NGSpice RRMS per device",
             fontsize=13.5, fontweight="bold", va="top")
    n_flip = s.get("devices_where_surrogate_prefers_pipeline_but_ngspice_prefers_reference")
    fig.text(0.02, 0.93,
             f"Mean over devices: pipeline picks {fmt(s.get('pipeline_surrogate_rrms'), 'rrms')} "
             f"surrogate / {fmt(s.get('pipeline_ngspice_rrms'), 'rrms')} NGSpice; reference cards "
             f"{fmt(s.get('reference_surrogate_rrms'), 'rrms')} surrogate / "
             f"{fmt(s.get('reference_ngspice_rrms'), 'rrms')} NGSpice.\n"
             f"Devices where the surrogate scores the pipeline pick lower and NGSpice scores the "
             f"reference card lower: {n_flip} of {len(devs)}.", fontsize=9.5, color=INK2, va="top")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    srcs = [rel(src)]
    if ref_name:
        srcs.append(f"out/research/ngspice_direct/{ref_name}/summary.json")
    if pipe_name:
        srcs.append(f"out/cold_loop/full/round_{pipe_name.split()[-1]}/extract/summary.json")
    LOG.used.update(srcs)
    return {"sources": srcs, "summary": s, "n": len(devs), "ref": ref_name, "pipe": pipe_name}


# --------------------------------------------------------------- per device

def fig_per_device(series: list[tuple[str, str, dict]], path: Path):
    series = [(lab, c, r) for lab, c, r in series if r is not None and r.get("fd")]
    if not series:
        LOG.skip("per-device figure", "no FD-polished series available")
        return None
    ref = read_json(RESEARCH / "reference_scores.json", "reference scores (per-device marks)")
    paper43 = {t: num(v.get("rrms")) for t, v in
               (ref or {}).get("paper_full43", {}).get("devices", {}).items()}
    tags = sorted({t for _l, _c, r in series for t in r["fd"]["per_device"]}, key=device_key)
    xs, x, prev = [], 0.0, None
    for t in tags:
        if prev is not None and family_of(t) != prev:
            x += 0.8
        xs.append(x)
        x += 1.0
        prev = family_of(t)
    fig, ax = plt.subplots(figsize=(14, 7.6))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.85, bottom=0.32)
    ax.grid(axis="y", which="major", zorder=0)
    offsets = np.linspace(-0.24, 0.24, len(series)) if len(series) > 1 else [0.0]
    for (label, color, r), dx in zip(series, offsets):
        px = [xx + dx for xx, t in zip(xs, tags) if r["fd"]["per_device"].get(t) is not None]
        py = [r["fd"]["per_device"][t] for t in tags if r["fd"]["per_device"].get(t) is not None]
        ax.scatter(px, py, s=64, facecolor=color, edgecolor=SURFACE, linewidth=1.3, zorder=3,
                   label=f"{label} (all-device {fmt(r['fd']['all'], 'rrms')})")
        bad = set((r["fd"].get("strict") or {}).get("failed_devices", []))
        fx = [xx + dx for xx, t in zip(xs, tags) if t in bad
              and r["fd"]["per_device"].get(t) is not None]
        fy = [r["fd"]["per_device"][t] for t in tags if t in bad
              and r["fd"]["per_device"].get(t) is not None]
        if fx:
            ax.scatter(fx, fy, **FAILED_MARK)
    if paper43:
        px = [xx for xx, t in zip(xs, tags) if paper43.get(t) is not None]
        py = [paper43[t] for t in tags if paper43.get(t) is not None]
        ax.scatter(px, py, s=260, marker="_", color=INK2, linewidth=1.6, zorder=2,
                   label="paper 43-parameter card (reference)")
    ax.set_yscale("log")
    lo_v, hi_v = ax.get_ylim()
    log_ticks(ax, "y", lo_v, hi_v)
    ax.set_xticks(xs)
    ax.set_xticklabels([device_label(t) for t in tags], rotation=40, ha="right")
    n_end = max(i for i, t in enumerate(tags) if family_of(t) == "nmos") if any(
        family_of(t) == "nmos" for t in tags) else None
    if n_end is not None and n_end < len(tags) - 1:
        sep = (xs[n_end] + xs[n_end + 1]) / 2
        ax.axvline(sep, color=MUTED, lw=1.0)
        ax.text((xs[0] + xs[n_end]) / 2, 1.01, "nMOS", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.text((xs[n_end + 1] + xs[-1]) / 2, 1.01, "pMOS", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    ax.set_ylabel("device RRMS (paper metric, log scale)")
    ax.set_xlabel("device (L and W in µm)")
    handles, _labels = ax.get_legend_handles_labels()
    if any((r["fd"].get("strict") or {}).get("n_failed_devices") for _l, _c, r in series):
        handles.append(failed_handle("red ×: included curve(s) failed to simulate; value "
                                     "averages the surviving curves only (not a valid RRMS)"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.27), ncol=2,
              fontsize=10)
    fig.text(0.01, 0.975, "Per-device RRMS after NGSpice FD polish, 43 parameters, measured 77 K data",
             fontsize=14, fontweight="bold", va="top")
    fig.text(0.01, 0.925, "Each marker: one device's RRMS after the same NGSpice FD polish. "
             "Horizontal dashes: paper 43-parameter card.", fontsize=9.5, color=INK2, va="top")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    srcs = [r["fd"]["source"] for _l, _c, r in series]
    if paper43:
        srcs.append("out/research/reference_scores.json")
    return {"sources": srcs}


# -------------------------------------------------------------- temperature

def temp_calls(dev: dict, key: str):
    if key == "raw":
        if "raw_n_sim" in dev:
            return dev["raw_n_sim"]
        if "n_sim" in dev and "polish_n_sim" in dev:
            return dev["n_sim"] - dev["polish_n_sim"]
        if "fd_n_sim" in dev:
            return 0  # one forward pass: no NGSpice before the FD polish
        return None
    if "n_sim" in dev:
        return dev["n_sim"]
    raw = temp_calls(dev, "raw")
    return raw + dev["fd_n_sim"] if raw is not None and "fd_n_sim" in dev else None


def derive_heldout(sets_by_method: dict) -> tuple[list[str], list[str]] | None:
    """The held-out temperatures are the subset whose per-temperature means
    average to every summary's held-out value (and the complement to observed)."""
    temps = None
    combos = None
    for sets in sets_by_method.values():
        for d in sets:
            for key, s in d["summary"].items():
                per = s.get("per_temp_all_device_mean", {})
                temps = temps or list(per)
                target, target_o = s["heldout"]["all_device_mean"], s["observed"]["all_device_mean"]
                ok = set()
                for k in range(1, len(per)):
                    for c in itertools.combinations(per, k):
                        rest = [t for t in per if t not in c]
                        if (abs(np.mean([per[t] for t in c]) - target) <= 1e-9 * max(1e-12, abs(target))
                                and abs(np.mean([per[t] for t in rest]) - target_o)
                                <= 1e-9 * max(1e-12, abs(target_o))):
                            ok.add(c)
                combos = ok if combos is None else combos & ok
    if not combos or len(combos) != 1:
        try:
            sys.path.insert(0, str(ROOT / "src"))
            from cryoml.temperature import TEMPS_HELDOUT, TEMPS_ALL  # noqa: E402
            held = [f"{t:g}" for t in TEMPS_HELDOUT]
            LOG.note("held-out temperatures not uniquely identified from results; "
                     "taken from cryoml.temperature")
            return held, [f"{t:g}" for t in TEMPS_ALL if f"{t:g}" not in held]
        except Exception as exc:  # pragma: no cover - fallback only
            LOG.skip("held-out temperature shading", f"not identifiable ({exc})")
            return None
    held = list(next(iter(combos)))
    return held, [t for t in temps if t not in held]


def load_temperature():
    base = RESEARCH / "temp"
    dirs = [p for p in sorted(base.glob("*")) if p.is_dir()]
    names = [p.name for p in dirs if list(p.glob("set_*.json"))]
    for p in dirs:
        if p.name not in names and p.name != "truth":
            LOG.skip(f"temperature method {p.name}", "no set_*.json yet")
    names = sorted(names, key=lambda n: (TEMP_ORDER.index(n) if n in TEMP_ORDER else 99, n))
    sets_by_method, files = {}, {}
    for n in names:
        data, fl = [], []
        for f in sorted((base / n).glob("set_*.json")):
            d = read_json(f, f"temperature {n} {f.name}")
            if d is not None and d.get("summary"):
                data.append(d)
                fl.append(rel(f))
        if data:
            sets_by_method[n], files[n] = data, fl
    if not sets_by_method:
        LOG.skip("temperature figure/table", "no temperature results")
        return None
    split = derive_heldout(sets_by_method)
    meta_files = sorted((ROOT / "data" / "processed" / "temp_bank").glob("*/meta.json"))
    bank = read_json(meta_files[0], "temperature bank meta") if len(meta_files) == 1 else None
    methods = []
    spare = iter(TEMP_SPARE)
    for n, sets in sets_by_method.items():
        color = TEMP_COLOR.get(n) or next(spare, MUTED)
        arch = n[len("surrogate_"):] if n.startswith("surrogate_") else None
        label = TEMP_LABEL.get(n) or (f"Pipeline search on {ARCH_LABEL.get(arch, arch)} surrogate"
                                      if arch else n)
        for key in ("raw", "fd"):
            if not all(key in d["summary"] for d in sets):
                continue
            temps = list(sets[0]["summary"][key]["per_temp_all_device_mean"])
            per_temp = {t: float(np.mean([d["summary"][key]["per_temp_all_device_mean"][t]
                                          for d in sets])) for t in temps}
            row = {"method": label, "stage": "method output" if key == "raw" else "+ NGSpice FD",
                   "key": key, "name": n, "color": color, "sets": len(sets), "per_temp": per_temp}
            for metric, tag in (("observed", "obs"), ("heldout", "held")):
                row[f"all_{tag}"] = float(np.mean([d["summary"][key][metric]["all_device_mean"]
                                                   for d in sets]))
                row[f"comb_{tag}"] = float(np.mean([d["summary"][key][metric]["combined"]
                                                    for d in sets]))
                fams = [family_stats({t: num(v.get(key, {}).get(metric))
                                      for t, v in d["devices"].items()}) for d in sets]
                row[f"nmos_{tag}"] = float(np.mean([f["nmos"] for f in fams]))
                row[f"pmos_{tag}"] = float(np.mean([f["pmos"] for f in fams]))
            calls = [temp_calls(v, key) for d in sets for v in d["devices"].values()]
            row["calls"] = float(np.mean(calls)) if calls and None not in calls else None
            for t in temps:
                row[f"T{t}"] = per_temp[t]
            methods.append(row)
    # Cross-check against temp_study.py's table.json (it may be older than the set files).
    table = read_json(base / "table.json", "temperature table.json (cross-check)") \
        if (base / "table.json").is_file() else None
    if table is None and not (base / "table.json").is_file():
        LOG.skip("temperature table.json cross-check", "out/research/temp/table.json not found")
    for row in methods:
        tname = row["name"] + ("+FD" if row["key"] == "fd" else "")
        match = [t for t in (table or []) if t.get("method") == tname]
        if table is not None and not match:
            LOG.note(f"temp/table.json has no row for {tname} (table older than set files)")
        for t in match:
            for a, b in (("observed_all_device_mean", "all_obs"), ("heldout_all_device_mean", "all_held")):
                if abs(t[a] - row[b]) > 1e-9:
                    LOG.note(f"temp/table.json {tname} {a} {t[a]:.6f} differs from set files "
                             f"{row[b]:.6f}")
    # temp_fit penalizes failed curves (PENALTY each): screen for values at the
    # penalty floor PENALTY / n_curves, which only a failed curve (or a very
    # poor fit) reaches.
    n_floor = 0
    for sets in sets_by_method.values():
        for d in sets:
            for t, v in d["devices"].items():
                floor = ss.PENALTY / max(1, len(ss.include_tags(t)))
                for key in ("raw", "fd"):
                    for val in (v.get(key, {}).get("per_temp") or {}).values():
                        n_floor += int(val is None or not math.isfinite(val)
                                       or val >= floor - 1e-12)
    return {"rows": methods, "split": split, "files": files, "bank": bank,
            "n_penalty_floor": n_floor,
            "bank_file": rel(meta_files[0]) if bank else None}


def fig_temperature(temp: dict, path: Path) -> list[str]:
    rows = temp["rows"]
    temps = sorted({float(t) for r in rows for t in r["per_temp"]})
    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.23)
    if temp["split"]:
        held = [float(t) for t in temp["split"][0]]
        for t in held:
            ax.axvspan(t - 8, t + 8, color=BAND, zorder=0, lw=0)
            ax.text(t, 1.005, "held out", transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=9, color=INK2)
    ax.grid(axis="y", which="major", zorder=0)
    for r in rows:
        xs = [float(t) for t in r["per_temp"]]
        ys = list(r["per_temp"].values())
        order = np.argsort(xs)
        xs, ys = np.array(xs)[order], np.array(ys)[order]
        raw = r["key"] == "raw"
        ax.plot(xs, ys, color=r["color"], lw=1.8 if raw else 2.2, ls=(0, (5, 3)) if raw else "-",
                zorder=2)
        ax.scatter(xs, ys, s=48, zorder=3, facecolor=SURFACE if raw else r["color"],
                   edgecolor=r["color"] if raw else SURFACE, linewidth=1.8 if raw else 1.2)
    ax.set_yscale("log")
    ax.set_xticks(temps)
    ax.set_xticklabels([f"{t:g}" for t in temps])
    ax.set_xlim(min(temps) - 12, max(temps) + 12)
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel("all-device mean RRMS vs simulated truth (log scale)")
    seen, handles = [], []
    for r in rows:
        if r["name"] not in seen:
            seen.append(r["name"])
            handles.append(Line2D([], [], color=r["color"], lw=2.4,
                                  label=f"{r['method']} ({r['sets']} sets)"))
    handles += [Line2D([], [], color=INK2, lw=1.8, ls=(0, (5, 3)), marker="o", mfc=SURFACE,
                       mec=INK2, label="dashed, open: method output"),
                Line2D([], [], color=INK2, lw=2.2, marker="o", mfc=INK2, mec=SURFACE,
                       label="solid, filled: + NGSpice FD polish")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.1), ncol=2, fontsize=10)
    n_sets = sorted({r["sets"] for r in rows})
    fig.text(0.01, 0.975, "Simulated temperature benchmark: all-device RRMS vs temperature",
             fontsize=14, fontweight="bold", va="top")
    sub = (f"Mean over {'/'.join(map(str, n_sets))} simulated sets × 18 devices; "
           "observed temperatures are fitted, shaded temperatures are held out.")
    fig.text(0.01, 0.925, sub, fontsize=9.5, color=INK2, va="top")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return sorted({f for fl in temp["files"].values() for f in fl})


# -------------------------------------------------------------------- index

def update_index(date: str, topic: str, folder: Path, pngs: list[str], summary: list[str]) -> str:
    state_path = REPORT_ROOT / "index.json"
    try:
        state = json.loads(state_path.read_text()) if state_path.is_file() else []
    except ValueError:
        state = []
    state = [e for e in state if not (e["date"] == date and e["topic"] == topic)]
    state.append({"date": date, "topic": topic, "folder": folder.name, "pngs": pngs,
                  "summary": summary, "generated": dt.datetime.now().isoformat(timespec="seconds")})
    state.sort(key=lambda e: (e["date"], e["generated"]), reverse=True)
    state_path.write_text(json.dumps(state, indent=1) + "\n")
    lines = ["# Research report index", "",
             "Newest batch first. Each batch folder holds the PNGs, `tables.md` and a `README.md` "
             "that names the result files behind every figure and table. Batches are written by "
             "`scripts/research/report_figs.py --topic <topic> --date <YYYY-MM-DD>`.", ""]
    for e in state:
        f = e["folder"]
        lines += [f"## {e['date']} · {e['topic']}", "",
                  f"Folder: [{f}/]({f}/) · [tables.md]({f}/tables.md) · [README.md]({f}/README.md)",
                  ""]
        lines += [f"- {s}" for s in e["summary"]]
        lines += ["", "Figures and tables: " + " · ".join(f"[{p}]({f}/{p})" for p in e["pngs"]), ""]
    text = "\n".join(lines)
    (REPORT_ROOT / "INDEX.md").write_text(text)
    return text


def baseline_sections(folder: Path, name, srcs, items: list, md_parts: list,
                      summary_lines: list) -> None:
    """Topic ``baseline_results`` (and any non-DEV topic): measured-data
    methods, rounds, surrogate gap, per device, temperature."""
    # 1. real methods -----------------------------------------------------
    real = []
    for label, loader in REAL_SPEC:
        res = loader()
        if res is None:
            LOG.skip(f"row '{label}'", "result file missing")
        else:
            real.append((label, res))
    if real:
        png = name("real_methods")
        files = fig_real_methods(real, folder / png)
        items.append((png, "Paper RRMS on the 18 measured devices for each method row, raw "
                      "(open) and after NGSpice FD polish (filled) where a polished version "
                      "exists, in the all-device convention with the three reference-card lines "
                      "and in the combined convention, annotated with mean NGSpice calls per "
                      f"device; sources: {srcs(files)}."))
        rows = reference_rows()
        for label, res in real:
            rows += stage_rows(label, res)
        notes = [COMBINED_NOTE, CALLS_NOTE,
                 "Reference rows: all-device values are the paper's reported values; combined/nMOS/"
                 "pMOS from out/research/reference_scores.json.", STRICT_NOTE]
        for label, res in real:
            notes += strict_notes(label, res)
        for label, res in real:
            if res["fd"] and res["fd"].get("start_mismatch"):
                notes.append(f"{label}: the FD run's re-score of its start differs from the "
                             f"method's own score on " + "; ".join(res["fd"]["start_mismatch"])
                             + " (own vs FD start); the method-output row uses the method's own "
                             "score. A large FD-start value is the penalized start objective of "
                             "a card with failed included curves (see failed dev.).")
        t = Table("real_methods", "Measured 77 K data, 18 devices: paper RRMS by method, raw "
                  "output and after NGSpice FD polish", [
                      {"key": "method", "head": "method", "md": "method", "kind": "text"},
                      {"key": "stage", "head": "stage", "md": "stage", "kind": "text"},
                      *RRMS_COLS, *STRICT_COLS], rows, notes)
        png_t = name("table_real_methods")
        render_table(t, folder / png_t)
        md_parts.append(table_markdown(t))
        items.append((png_t, "Table of the rows in the previous figure plus the reference cards: "
                      "all-device, combined, nMOS and pMOS RRMS and NGSpice calls per device "
                      f"(this run and cumulative); sources: {srcs(files)}, "
                      "`out/research/reference_scores.json`."))
        pick = {label: res for label, res in real}
        parts = []
        for label in ("Pipeline 43, round 4", "NGSpice swap 43, round 1", "Direct NGSpice 43, round 3"):
            r = pick.get(label)
            if r:
                fd = f" / {fmt(r['fd']['all'], 'rrms')}" if r["fd"] else ""
                bad = [f"{n_failed(x)} failed device(s) {stage}" for stage, x in
                       (("raw", r), ("+FD", r["fd"])) if n_failed(x)]
                parts.append(f"{label} {fmt(r['all'], 'rrms')}{fd}"
                             + (f" ({'; '.join(bad)})" if bad else ""))
        if parts:
            summary_lines.append("Measured 77 K data, all-device RRMS (raw / +FD): " + "; ".join(parts)
                                 + ". References: " + ", ".join(f"{s} {v:.4f}" for _l, s, v, _k
                                                              in REFERENCES) + ".")
    else:
        LOG.skip("figure real_methods", "no rows available")

    # 2. rounds -----------------------------------------------------------
    series, round_rows = [], []
    for pset, label, ls, filled, dx in (("seven", "Pipeline 7", (0, (5, 3)), False, -0.05),
                                        ("full", "Pipeline 43", "-", True, 0.05)):
        rounds = sorted(int(p.parent.parent.name.split("_")[1]) for p in
                        (OUT / "cold_loop" / pset).glob("round_*/extract/summary.json"))
        res = [r for r in (pipeline(pset, k) for k in rounds) if r is not None]
        if res:
            series.append({"label": label, "results": res, "color": BLUE, "ls": ls,
                           "filled": filled, "marker": "o", "dx": dx})
    chain = direct43_chain()
    if chain:
        series.append({"label": "Direct NGSpice 43", "results": chain, "color": AQUA, "ls": "-",
                       "filled": True, "marker": "s", "dx": 0.0})
    swaps = []
    for p in sorted((RESEARCH / "swap_search").glob("*/summary.json")):
        r = swap_run(p.parent.name)
        if r is not None:
            swaps.append((r, -0.15 if r["n_params"] == 7 else 0.15))
    swaps.sort(key=lambda rd: (rd[0]["n_params"] or 0, rd[0]["round"] or 0))
    if series or swaps:
        png = name("rounds")
        files = fig_rounds(series, swaps, folder / png)
        items.append((png, "All-device RRMS (raw method output) against round for the cold "
                      "pipeline at 7 and 43 parameters and for direct NGSpice at 43 parameters "
                      "(chain of runs linked by spec source_run), with the round-1 NGSpice-swap "
                      f"points as diamonds and reference-card lines; sources: {srcs(files)}."))
        rows = []
        for s in series:
            for r in s["results"]:
                rows.append({"series": s["label"], "round": str(r["round"]),
                             **{k: r.get(k) for k in ("all", "combined", "nmos", "pmos")},
                             "run": r.get("run_calls"), "cum": r.get("cum_calls"),
                             **strict_cells(r.get("strict"))})
        for r, _dx in swaps:
            rows.append({"series": f"NGSpice swap {r['n_params']}", "round": str(r["round"]),
                         **{k: r.get(k) for k in ("all", "combined", "nmos", "pmos")},
                         "run": r.get("run_calls"), "cum": r.get("cum_calls"),
                         **strict_cells(r.get("strict"))})
        chain_txt = " -> ".join(r["name"] for r in chain)
        notes = [COMBINED_NOTE, CALLS_NOTE.replace("; +FD rows: FD polish only", ""),
                 "Raw method output; no FD polish in this table.", STRICT_NOTE]
        for s_ in series:
            for r in s_["results"]:
                notes += [n.replace(" (method output)", "") for n in
                          strict_notes(f"{s_['label']} round {r['round']}", {**r, "fd": None})]
        for r, _dx in swaps:
            notes += [n.replace(" (method output)", "") for n in
                      strict_notes(f"NGSpice swap {r['n_params']} round {r['round']}",
                                   {**r, "fd": None})]
        if chain:
            notes.append(f"Direct NGSpice 43 rounds are the runs {chain_txt} (each spec's "
                         "source_run names the previous round).")
        t = Table("rounds", "Measured 77 K data, 18 devices: paper RRMS by round", [
            {"key": "series", "head": "series", "md": "series", "kind": "text"},
            {"key": "round", "head": "round", "md": "round", "kind": "text"},
            *RRMS_COLS, *STRICT_COLS], rows, notes)
        png_t = name("table_rounds")
        render_table(t, folder / png_t)
        md_parts.append(table_markdown(t))
        items.append((png_t, "Table of the points in the rounds figure: all-device, combined, "
                      "nMOS and pMOS RRMS and NGSpice calls per device (this run and "
                      f"cumulative); sources: {srcs(files)}."))
        bits = []
        for s in series:
            bits.append(f"{s['label']} r{s['results'][0]['round']}-r{s['results'][-1]['round']} "
                        + " -> ".join(fmt(r["all"], "rrms") for r in s["results"]))
        if bits:
            summary_lines.append("All-device RRMS by round (raw): " + "; ".join(bits) + ".")
    else:
        LOG.skip("figure rounds", "no series available")

    # 3. surrogate gap ----------------------------------------------------
    png = f"{0:02d}_tmp.png"
    gap = fig_surrogate_gap(folder / png)
    if gap:
        final = name("surrogate_gap")
        (folder / png).replace(folder / final)
        items.append((final, "Per-device surrogate-predicted RRMS against NGSpice RRMS (log axes, "
                      "y = x line) for the round-1 43-parameter pipeline picks and for the "
                      f"direct-fit reference cards ({gap['ref'] or 'run not identified'}); "
                      f"sources: {srcs(gap['sources'])}."))
        s = gap["summary"]
        summary_lines.append(
            f"Round-1 43-parameter surrogate, mean predicted / NGSpice RRMS: pipeline picks "
            f"{fmt(s.get('pipeline_surrogate_rrms'), 'rrms')} / {fmt(s.get('pipeline_ngspice_rrms'), 'rrms')}, "
            f"direct-fit cards {fmt(s.get('reference_surrogate_rrms'), 'rrms')} / "
            f"{fmt(s.get('reference_ngspice_rrms'), 'rrms')}; opposite ordering on "
            f"{s.get('devices_where_surrogate_prefers_pipeline_but_ngspice_prefers_reference')} of "
            f"{gap['n']} devices.")
    else:
        (folder / png).unlink(missing_ok=True)

    # 4. per device -------------------------------------------------------
    pd_series = [("Pipeline 43, round 4 + FD", BLUE, pipeline("full", 4)),
                 ("NGSpice swap 43, round 1 + FD", ORANGE, swap_run("full_round_1")),
                 ("Direct NGSpice 43, round 3 + FD", AQUA, direct_run("full_r3_direct"))]
    for label, _c, r in pd_series:
        if r is None or not r.get("fd"):
            LOG.skip(f"per-device series '{label}'", "FD-polished summary missing")
    png = f"{0:02d}_tmp.png"
    pdv = fig_per_device(pd_series, folder / png)
    if pdv:
        final = name("per_device")
        (folder / png).replace(folder / final)
        items.append((final, "Per-device RRMS after NGSpice FD polish (log y, nMOS then pMOS) for "
                      "pipeline 43 round 4, NGSpice swap 43 round 1 and direct NGSpice 43 round 3, "
                      f"with the paper 43-parameter card per device; sources: {srcs(pdv['sources'])}."))

    # 5. temperature ------------------------------------------------------
    temp = load_temperature()
    if temp:
        png = name("temperature")
        files = fig_temperature(temp, folder / png)
        held = temp["split"][0] if temp["split"] else []
        items.append((png, "All-device RRMS against temperature (77-300 K, log y) for every "
                      "temperature method found in out/research/temp, raw dashed and +FD solid, "
                      f"held-out temperatures ({', '.join(held)} K) shaded; mean over simulated "
                      f"sets; sources: {srcs(files)}."))
        rows = temp["rows"]
        obs_txt = "/".join(temp["split"][1]) if temp["split"] else "?"
        held_txt = "/".join(held) if held else "?"
        bank_note = []
        if temp["bank"]:
            bank_note = [f"All methods draw bank starts from the temperature bank "
                         f"({temp['bank'].get('n')} cards/device, each simulated at "
                         f"{len(temp['bank'].get('temps', []))} temperatures; "
                         f"{temp['bank_file']}); the surrogate is trained on it. Calls exclude "
                         "the bank."]
        t = Table("temperature", "Simulated temperature benchmark: paper RRMS vs simulated truth, "
                  f"observed ({obs_txt} K) and held-out ({held_txt} K) temperatures", [
                      {"key": "method", "head": "method", "md": "method", "kind": "text"},
                      {"key": "stage", "head": "stage", "md": "stage", "kind": "text"},
                      {"key": "sets", "head": "sets", "md": "sets", "kind": "int"},
                      {"key": "all_obs", "head": "observed", "group": "all-device", "md": "all-device observed", "kind": "rrms"},
                      {"key": "all_held", "head": "held-out", "group": "all-device", "md": "all-device held-out", "kind": "rrms"},
                      {"key": "comb_obs", "head": "observed", "group": "combined", "md": "combined observed", "kind": "rrms"},
                      {"key": "comb_held", "head": "held-out", "group": "combined", "md": "combined held-out", "kind": "rrms"},
                      {"key": "nmos_obs", "head": "observed", "group": "nMOS mean", "md": "nMOS observed", "kind": "rrms", "family": True},
                      {"key": "nmos_held", "head": "held-out", "group": "nMOS mean", "md": "nMOS held-out", "kind": "rrms", "family": True},
                      {"key": "pmos_obs", "head": "observed", "group": "pMOS mean", "md": "pMOS observed", "kind": "rrms", "family": True},
                      {"key": "pmos_held", "head": "held-out", "group": "pMOS mean", "md": "pMOS held-out", "kind": "rrms", "family": True},
                      {"key": "calls", "head": "this run", "group": "NGSpice calls / dev", "md": "NGSpice calls/device (this run)", "kind": "int"},
                  ], rows,
                  [f"Means over simulated sets of 18 devices. {COMBINED_NOTE} Values below "
                   "0.0001 are shown in scientific notation.",
                   "Failed curves: temp_fit scores a non-finite included curve at RRMS "
                   f"{ss.PENALTY:g} (penalized convention), so no curve is silently dropped here; "
                   f"{temp['n_penalty_floor']} device-temperature value(s) reach the penalty floor "
                   "10 / n_curves.",
                   "NGSpice calls per device: method output = calls before FD (search and "
                   "validation); +FD rows = total including the FD polish.", *bank_note])
        png_t = name("table_temperature")
        render_table(t, folder / png_t)
        md_parts.append(table_markdown(t))
        items.append((png_t, "Table of observed and held-out all-device, combined, nMOS and pMOS "
                      "RRMS and NGSpice calls per device for each temperature method, raw and "
                      f"+FD; sources: {srcs(files)}."))
        temps = sorted({t for r in rows for t in r["per_temp"]}, key=float)
        t2 = Table("temperature_per_temp", "Simulated temperature benchmark: all-device RRMS per "
                   "temperature (* = held out)", [
                       {"key": "method", "head": "method", "md": "method", "kind": "text"},
                       {"key": "stage", "head": "stage", "md": "stage", "kind": "text"},
                       *[{"key": f"T{t}", "head": f"{t} K" + ("*" if t in held else ""),
                          "group": "all-device mean RRMS", "md": f"{t} K" + ("*" if t in held else ""),
                          "kind": "rrms"} for t in temps]], rows,
                   ["Mean over simulated sets of 18 devices; values below 0.0001 in scientific "
                    "notation."])
        png_t = name("table_temperature_per_temp")
        render_table(t2, folder / png_t)
        md_parts.append(table_markdown(t2))
        items.append((png_t, "Table of the values plotted in the temperature figure (all-device "
                      f"RRMS per temperature, raw and +FD); sources: {srcs(files)}."))
        bits = []
        for r in rows:
            if r["key"] == "raw":
                fd = next((x for x in rows if x["name"] == r["name"] and x["key"] == "fd"), None)
                bits.append(f"{r['name']} {fmt(r['all_held'], 'rrms')}"
                            + (f" / {fmt(fd['all_held'], 'rrms')}" if fd else ""))
        summary_lines.append("Temperature benchmark, held-out all-device RRMS (raw / +FD): "
                             + "; ".join(bits) + ".")


# --------------------------------------------------------------------- DEV

DEV_DIR = RESEARCH / "dev"
AUDIT_DIR = RESEARCH / "audit"
DEV_BUDGETS = (500, 1000, 2000, 5000)
# Node of each DEV arm by label prefix (docs/site/content.json, icml-nodes).
DEV_NODE = (("hybrid_", "hybrid TR"), ("sketch_", "n1a"), ("anytime_", "n1b"),
            ("cmaes_trf", "n1c"), ("lq_cmaes", "n1c"))
DEV_PAIRS = (("sketch_surrogate_k8", "sketch_none_k8"),
             ("hybrid_surrogate", "hybrid_none"),
             ("anytime_sur_presearch_race", "anytime_none_race"),
             ("sketch_surrogate_k8", "cmaes_trf"),
             ("sketch_none_k8", "hybrid_none"))
N_BOOT, BOOT_SEED = 10_000, 0


def dev_node(label: str) -> str:
    return next((node for prefix, node in DEV_NODE if label.startswith(prefix)), "")


def calls_text(v) -> str:
    if v is None:
        return "–"
    return "never" if math.isinf(v) else f"{v:.0f}"


def failed_per_budget(arm: dict, budgets) -> str:
    cells = []
    for b in budgets:
        st = (arm["budgets"].get(str(b)) or {}).get("strict")
        cells.append("–" if not st or not st.get("n_checked") else str(st["n_failed"]))
    return "/".join(cells)


def paired_stats(a: dict, b: dict) -> dict:
    """``a`` vs ``b`` over their common DEV targets; diff = a - b (negative:
    ``a`` has the lower RRMS).  Tie rule and bootstrap (10k resamples over
    targets, seed 0, percentile 95% CI) as in dev_compare.paired; the win rate
    uses the same resamples."""
    common = sorted(set(a) & set(b))
    x = np.array([a[t] for t in common], dtype=np.float64)
    y = np.array([b[t] for t in common], dtype=np.float64)
    diff = x - y
    tie = np.abs(diff) < np.maximum(1e-6, 0.01 * np.maximum(np.abs(x), np.abs(y)))
    win = (~tie & (diff < 0)).astype(np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(diff), (N_BOOT, len(diff)))
    md, wr = diff[idx].mean(1), win[idx].mean(1)
    return {"n": len(common), "mean_a": float(x.mean()), "mean_b": float(y.mean()),
            "mean_diff": float(diff.mean()),
            "ci": [float(np.percentile(md, 2.5)), float(np.percentile(md, 97.5))],
            "win_rate": float(win.mean()),
            "win_ci": [float(np.percentile(wr, 2.5)), float(np.percentile(wr, 97.5))],
            "wins": int(win.sum()), "ties": int(tie.sum()),
            "losses": int(np.sum(~tie & (diff > 0)))}


def dev_sections(folder: Path, name, srcs, items: list, md_parts: list,
                 summary_lines: list) -> None:
    """Topic ``dev*``: the DEV comparison (``dev_compare.py`` output) and the
    failed-curve audit (``audit_failed_curves.py`` output)."""
    import shutil
    cmp_json, cmp_md = DEV_DIR / "compare.json", DEV_DIR / "compare.md"
    curves = DEV_DIR / "compare_curves.png"
    report = read_json(cmp_json, "DEV comparison")
    if report is not None and cmp_md.is_file():
        LOG.used.add(rel(cmp_md))
    sources = [rel(cmp_json), rel(cmp_md)]

    # 1. the curves figure, copied ---------------------------------------
    if curves.is_file():
        png = name("compare_curves")
        shutil.copy2(curves, folder / png)
        LOG.used.add(rel(curves))
        # dev_compare.plot zips its 8 series colours with the arms, so only
        # the first 8 registry arms are drawn.
        order = list((report or {}).get("arms", {}).items())
        drawn = [lab for lab, a in order[:8] if "curve" in a]
        left_out = [lab for lab, _a in order[8:]]
        items.append((png, "Copy of the DEV comparison figure: best-so-far paper RRMS against "
                      "cumulative NGSpice calls per target on the 36 DEV targets, with the "
                      "all-device mean (solid line) and median (dashed line) of the arms "
                      + ", ".join(drawn) + ", the fixed-budget run means as dots, and the "
                      "bank-oracle, pipeline r1 and pipeline r1 + FD references as open markers"
                      + (" (the arms " + ", ".join(left_out) + " are not drawn: "
                         "dev_compare.py's plot has 8 colours and stops after the 8th arm)"
                         if left_out else "")
                      + f"; sources: {srcs([rel(curves), rel(cmp_json)])}."))
        if left_out:
            LOG.note("compare_curves.png draws only the first 8 arms of compare.json; not drawn: "
                     + ", ".join(left_out))
    else:
        LOG.skip("DEV curves figure", f"{rel(curves)} not found")
    if report is None:
        return
    arms, refs = report["arms"], report["references"]
    n_targets = report.get("n_targets")
    if report.get("partial"):
        LOG.note("compare.json is a PARTIAL (--allow-partial) report")
    bad_checks = [c for c in report.get("checks", []) if c["level"] != "PASS"]
    if report.get("checks_failed"):
        LOG.note("compare.json: hard checks failed: " + "; ".join(
            f"{c['check']} ({c['detail']})" for c in bad_checks if c["level"] == "FAIL"))

    # 2. primary pooled table --------------------------------------------
    budgets = sorted({int(b) for a in arms.values() for b in a["budgets"]})
    rows = []
    for label, arm in arms.items():
        row = {"node": dev_node(label), "arm": label, "mode": arm["mode"],
               "auc": arm.get("auc"),
               "c05": calls_text((arm.get("median_calls_to") or {}).get("0.05")),
               "c01": calls_text((arm.get("median_calls_to") or {}).get("0.01")),
               "failed": failed_per_budget(arm, budgets)}
        for b in budgets:
            row[f"B{b}"] = (arm["budgets"].get(str(b)) or {}).get("mean")
        rows.append(row)
    ref_txt = []
    for ref in refs.values():
        st = ref.get("strict") or {}
        extra = ""
        if st.get("n_failed"):
            extra = (f"; {st['n_failed']} failed target(s), valid-only "
                     f"{fmt(st['valid_mean'], 'rrms')} over {st['n_valid']}, penalized "
                     f"{fmt(st['penalized_mean'], 'rrms')}")
        ref_txt.append(f"{ref['label']} {fmt(ref['mean'], 'rrms')} at "
                       f"{ref['mean_calls']:.0f} mean NGSpice calls{extra}")
    notes = [f"Pooled all-device mean paper RRMS (frozen inclusion) over {n_targets} DEV targets "
             f"({', '.join(report['sets'])}, 18 devices each) at fixed cumulative NGSpice budgets "
             "per target; every NGSpice call counts, Jacobian probes included. Mode fixed = one "
             "run directory per budget, value = one fresh NGSpice score of the final card.",
             "AUC log10 = per-target mean of log10(max(best-so-far RRMS, 1e-6)) over a 40-point "
             "log grid of calls (10 to the largest budget), averaged over targets (lower = lower "
             "RRMS earlier).",
             "Calls to 0.05 / 0.01 = median over targets of the first NGSpice call count at which "
             "the best-so-far RRMS is at or below the threshold, from the largest-budget trace; "
             "never = the median target does not reach it.",
             "Failed targets = targets whose final card has a non-finite included metric curve, "
             "per budget " + "/".join(f"B{b}" for b in budgets) + "; – = budget not run.",
             "References: " + "; ".join(ref_txt) + "."]
    if bad_checks:
        notes.append("Non-PASS checks in compare.json: " + "; ".join(
            f"{c['level']} {c['check']}" for c in bad_checks) + ".")
    t = Table("dev_primary", f"DEV comparison, {n_targets} simulated targets: pooled all-device "
              "mean paper RRMS by arm and NGSpice budget per target", [
                  {"key": "node", "head": "node", "md": "node", "kind": "text"},
                  {"key": "arm", "head": "arm", "md": "arm", "kind": "text"},
                  {"key": "mode", "head": "mode", "md": "mode", "kind": "text"},
                  *[{"key": f"B{b}", "head": f"B{b}", "group": "pooled mean RRMS at budget",
                     "md": f"B{b}", "kind": "rrms"} for b in budgets],
                  {"key": "auc", "head": "AUC log10", "md": "AUC log10", "kind": "rrms"},
                  {"key": "c05", "head": "to 0.05", "group": "median calls", "md":
                   "median calls to 0.05", "kind": "rtext"},
                  {"key": "c01", "head": "to 0.01", "group": "median calls", "md":
                   "median calls to 0.01", "kind": "rtext"},
                  {"key": "failed", "head": "failed targets", "md": "failed targets (per budget)",
                   "kind": "rtext"}], rows, notes)
    png_t = name("table_dev_primary")
    render_table(t, folder / png_t)
    md_parts.append(table_markdown(t))
    items.append((png_t, "Table of the pooled all-device mean paper RRMS of every DEV arm at "
                  f"{', '.join(str(b) for b in budgets)} NGSpice calls per target, with AUC, "
                  "median calls to RRMS 0.05 and 0.01 and failed targets per budget, plus the "
                  f"reference values in the notes; sources: {srcs(sources)}."))
    # cross-check against compare.md's primary table
    md_rows = {}
    if cmp_md.is_file():
        for line in cmp_md.read_text().splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) > 2 and cells[0] in arms and cells[1] in ("fixed", "trace"):
                md_rows.setdefault(cells[0], cells)
    for r in rows:
        cells = md_rows.get(r["arm"])
        if cells is None:
            LOG.note(f"compare.md has no primary row for {r['arm']}")
            continue
        for k, b in enumerate(budgets):
            v, c = r[f"B{b}"], cells[2 + k]
            if v is None:
                continue
            try:   # compare.md: 4 decimals, or 3 significant digits below 1e-3
                tol = 0.006 * abs(v) if "e" in c else 0.5e-4 + 1e-12
                ok = abs(float(c) - v) <= tol
            except ValueError:
                ok = False
            if not ok:
                LOG.note(f"compare.md {r['arm']} B{b} '{c}' differs from compare.json {v:.6f}")

    # 3. paired comparisons -----------------------------------------------
    prow, paired_out = [], {}
    for a_lab, b_lab in DEV_PAIRS:
        if a_lab not in arms or b_lab not in arms:
            LOG.skip(f"pair {a_lab} vs {b_lab}", "arm missing from compare.json")
            continue
        for b in budgets:
            sa, sb = arms[a_lab]["budgets"].get(str(b)), arms[b_lab]["budgets"].get(str(b))
            if not sa or not sb:
                continue
            st = paired_stats(sa["values"], sb["values"])
            paired_out[(a_lab, b_lab, b)] = st
            bad = [f"{lab} {sx['strict']['n_failed']}" for lab, sx in ((a_lab, sa), (b_lab, sb))
                   if (sx.get("strict") or {}).get("n_failed")]
            prow.append({"pair": f"{a_lab} vs {b_lab}", "budget": f"B{b}", "n": st["n"],
                         "mean_a": st["mean_a"], "mean_b": st["mean_b"],
                         "diff": st["mean_diff"],
                         "ci": f"[{fmt(st['ci'][0], 'rrms')}, {fmt(st['ci'][1], 'rrms')}]",
                         "win": f"{st['win_rate']:.2f}",
                         "win_ci": f"[{st['win_ci'][0]:.2f}, {st['win_ci'][1]:.2f}]",
                         "wtl": f"{st['wins']}/{st['ties']}/{st['losses']}",
                         "failed": ", ".join(bad) or "0"})
            ref = next((p for p in report.get("paired", []) if p["arm"] == a_lab
                        and p["reference"] == b_lab and int(p["budget"]) == b), None)
            if ref is not None and (abs(ref["mean_diff"] - st["mean_diff"]) > 1e-12
                                    or max(abs(u - v) for u, v in zip(ref["ci95"], st["ci"])) > 1e-12
                                    or (ref["wins"], ref["ties"], ref["losses"])
                                    != (st["wins"], st["ties"], st["losses"])):
                LOG.note(f"paired {a_lab} vs {b_lab} B{b}: differs from compare.json paired row")
    if prow:
        t = Table("dev_paired", f"DEV paired comparisons over the {n_targets} targets: "
                  "difference A − B of paper RRMS per target at each budget", [
                      {"key": "pair", "head": "A vs B", "md": "A vs B", "kind": "text"},
                      {"key": "budget", "head": "budget", "md": "budget", "kind": "rtext"},
                      {"key": "n", "head": "n", "md": "n", "kind": "int"},
                      {"key": "mean_a", "head": "A", "group": "pooled mean", "md": "mean A",
                       "kind": "rrms"},
                      {"key": "mean_b", "head": "B", "group": "pooled mean", "md": "mean B",
                       "kind": "rrms"},
                      {"key": "diff", "head": "mean", "group": "difference A − B",
                       "md": "mean difference A − B", "kind": "rrms"},
                      {"key": "ci", "head": "95% CI", "group": "difference A − B",
                       "md": "95% CI", "kind": "rtext"},
                      {"key": "win", "head": "rate", "group": "A lower (win)",
                       "md": "win rate A", "kind": "rtext"},
                      {"key": "win_ci", "head": "95% CI", "group": "A lower (win)",
                       "md": "win rate 95% CI", "kind": "rtext"},
                      {"key": "wtl", "head": "W/T/L", "md": "W/T/L", "kind": "rtext"},
                      {"key": "failed", "head": "failed", "md": "failed targets (A, B)",
                       "kind": "rtext"}], prow,
                  ["Per-target values: compare.json arms.<arm>.budgets.<budget>.values (one fresh "
                   "NGSpice score of each final card). Difference = A − B; negative = A has the "
                   "lower RRMS.",
                   "Win = A lower than B by more than the tie tolerance max(1e-6, 1% of the larger "
                   "value) (dev_compare.paired); win rate = wins / n; W/T/L = wins, ties, losses "
                   "of A.",
                   f"95% CI: percentile bootstrap over targets, {N_BOOT:,} resamples, seed "
                   f"{BOOT_SEED}; the mean difference and the win rate use the same resamples.",
                   "failed = targets with a non-finite included curve in A or B at that budget."])
        png_t = name("table_dev_paired")
        render_table(t, folder / png_t)
        md_parts.append(table_markdown(t))
        items.append((png_t, "Table of paired per-target differences A − B at each budget for "
                      + "; ".join(f"{a} vs {b}" for a, b in DEV_PAIRS)
                      + ", with the mean difference, the win rate of A and their bootstrap 95% "
                      f"CIs over the {n_targets} DEV targets; sources: {srcs([rel(cmp_json)])}."))

    # 4. audit corrections ------------------------------------------------
    audit_json, audit_md = AUDIT_DIR / "failed_curves.json", AUDIT_DIR / "failed_curves.md"
    audit = read_json(audit_json, "failed-curve audit")
    if audit is not None:
        if audit_md.is_file():
            LOG.used.add(rel(audit_md))
        arow = []
        for h in audit.get("headlines", []):
            arow.append({"headline": h["headline"], "nfail": h["n_failed_devices"],
                         "devs": ", ".join(h.get("failed_devices", [])) or "none",
                         "s_all": h["reported_all_device_mean"], "s_comb": h["reported_combined"],
                         "v_all": h["valid_all_device_mean"], "v_comb": h["valid_combined"],
                         "p_all": h["penalized_all_device_mean"],
                         "p_comb": h["penalized_combined"]})
        conv = audit.get("conventions", {})
        t = Table("audit", "Failed-curve audit: headline numbers as stored, valid devices only, "
                  "and penalized", [
                      {"key": "headline", "head": "headline", "md": "headline", "kind": "text"},
                      {"key": "nfail", "head": "failed dev.", "md": "failed devices",
                       "kind": "int"},
                      {"key": "devs", "head": "failed device", "md": "failed device(s)",
                       "kind": "text"},
                      {"key": "s_all", "head": "all-device", "group": "stored",
                       "md": "stored all-device", "kind": "rrms"},
                      {"key": "s_comb", "head": "combined", "group": "stored",
                       "md": "stored combined", "kind": "rrms"},
                      {"key": "v_all", "head": "all-device", "group": "valid-only",
                       "md": "valid-only all-device", "kind": "rrms", "family": True},
                      {"key": "v_comb", "head": "combined", "group": "valid-only",
                       "md": "valid-only combined", "kind": "rrms", "family": True},
                      {"key": "p_all", "head": "all-device", "group": "penalized",
                       "md": "penalized all-device", "kind": "rrms"},
                      {"key": "p_comb", "head": "combined", "group": "penalized",
                       "md": "penalized combined", "kind": "rrms"}], arow,
                  [f"Runs audited: {audit.get('n_runs')}; runs with failed devices: "
                   f"{audit.get('n_runs_with_failed_devices')}. Inclusion set: "
                   f"{audit.get('include_source')}.",
                   "stored = the value in the result file: for a failed device, "
                   "metrics.device_rrms_new averages only the surviving included curves "
                   "(upstream rrmsCalc.py gives NaN).",
                   f"{conv.get('valid', 'valid-only')}; n = 18 minus failed devices.",
                   f"{conv.get('penalized', 'penalized')}.",
                   "all-device = mean over devices; combined = mean of the nMOS and pMOS family "
                   "means."])
        png_t = name("table_audit")
        render_table(t, folder / png_t)
        md_parts.append(table_markdown(t))
        items.append((png_t, "Table of the failed-curve audit's headline numbers, each as stored, "
                      "as the mean over valid devices only and in the penalized convention (failed "
                      "included curves at RRMS 10), all-device and combined, with the failed-device "
                      f"count and names; sources: {srcs([rel(audit_json), rel(audit_md)])}."))
        hit = [h for h in audit.get("headlines", []) if h["n_failed_devices"]]
        summary_lines.append(
            f"Failed-curve audit: {audit.get('n_runs_with_failed_devices')} of "
            f"{audit.get('n_runs')} audited runs have a device with a failed included curve; "
            f"{len(hit)} of the {len(audit.get('headlines', []))} audited headline numbers come "
            "from runs with a failed device. Example: "
            "pipeline 43 round 4 raw all-device stored / valid-only / penalized "
            + next((f"{h['reported_all_device_mean']:.4f} / {h['valid_all_device_mean']:.4f} / "
                    f"{h['penalized_all_device_mean']:.4f}" for h in hit
                    if h["headline"].startswith("pipeline 43, round 4, raw")), "n/a") + ".")

    # index summary lines -------------------------------------------------
    pick = [a for a in ("sketch_surrogate_k8", "sketch_none_k8", "hybrid_surrogate",
                        "hybrid_none", "anytime_sur_presearch_race", "anytime_none_race",
                        "cmaes_trf") if a in arms]
    summary_lines.insert(0, f"DEV, {n_targets} simulated targets, pooled all-device mean RRMS at "
                         "B500 / B5000: " + "; ".join(
                             f"{a} {fmt((arms[a]['budgets'].get('500') or {}).get('mean'), 'rrms')}"
                             f" / {fmt((arms[a]['budgets'].get('5000') or {}).get('mean'), 'rrms')}"
                             for a in pick) + ".")
    bits = []
    for a_lab, b_lab in DEV_PAIRS[:2]:
        vals = [f"B{b} {fmt(st['mean_diff'], 'rrms')} [{fmt(st['ci'][0], 'rrms')}, "
                f"{fmt(st['ci'][1], 'rrms')}], win rate {st['win_rate']:.2f}"
                for (x, y, b), st in paired_out.items() if (x, y) == (a_lab, b_lab)]
        if vals:
            bits.append(f"{a_lab} − {b_lab}: " + "; ".join(vals))
    if bits:
        summary_lines.insert(1, "Paired mean difference [95% CI] and win rate: "
                             + " | ".join(bits) + ".")


def candidate_files(topic: str = "") -> list[str]:
    if topic.startswith("dev"):
        pats = ["out/research/dev/compare.json", "out/research/dev/compare.md",
                "out/research/dev/compare_curves.png", "out/research/audit/failed_curves.json",
                "out/research/audit/failed_curves.md"]
        return sorted({rel(p) for pat in pats for p in ROOT.glob(pat)})

    pats = ["out/cold_loop/seven/round_*/extract/summary.json",
            "out/cold_loop/full/round_*/extract/summary.json",
            "out/research/ngspice_direct/**/summary.json",
            "out/research/swap_search/*/summary.json",
            "out/research/polished/*/*/summary.json",
            "out/research/surrogate_gap_full_round_1.json",
            "out/research/temp/*/set_*.json", "out/research/temp/table.json"]
    return sorted({rel(p) for pat in pats for p in ROOT.glob(pat)})


# --------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--topic", default="baseline_results")
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    try:
        dt.date.fromisoformat(args.date)
    except ValueError:
        ap.error("--date must be YYYY-MM-DD")
    topic = re.sub(r"[^A-Za-z0-9_-]+", "_", args.topic).strip("_") or "report"
    folder = REPORT_ROOT / f"{args.date}_{topic}"
    folder.mkdir(parents=True, exist_ok=True)

    items: list[tuple[str, str]] = []   # (png name, README sentence)
    md_parts: list[str] = []
    summary_lines: list[str] = []
    seq = iter(range(1, 100))

    def name(stem: str) -> str:
        return f"{next(seq):02d}_{stem}.png"

    def srcs(files) -> str:
        return ", ".join(f"`{f}`" for f in files)

    if topic.startswith("dev"):
        dev_sections(folder, name, srcs, items, md_parts, summary_lines)
    else:
        baseline_sections(folder, name, srcs, items, md_parts, summary_lines)

    # ---------------------------------------------------------- write out
    pngs = [p for p, _s in items]
    for stale in folder.glob("*.png"):
        if stale.name not in pngs:
            stale.unlink()
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    cmd = f"scripts/research/report_figs.py --topic {topic} --date {args.date}"
    tables_md = (f"# Tables, {args.date} {topic}\n\nGenerated {now} by `{cmd}` from result files "
                 "only. RRMS is the paper metric with frozen published-baseline curve inclusion, "
                 "re-simulated in NGSpice.\n\n" + "\n".join(md_parts))
    (folder / "tables.md").write_text(tables_md)
    unused = [f for f in candidate_files(topic) if f not in LOG.used]
    readme = [f"# Report batch {args.date} · {topic}", "",
              f"Generated {now} by `{cmd}` from result files only (no simulation). RRMS is the "
              "paper metric with frozen published-baseline curve inclusion, re-simulated in "
              "NGSpice, unless an item says otherwise.", "", "## Figures and tables", ""]
    readme += [f"- `{p}`: {s}" for p, s in items]
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    if LOG.notes:
        readme += ["## Notes", ""] + [f"- {n}" for n in LOG.notes] + [""]
    readme += ["## Skipped inputs", ""] + ([f"- {s}" for s in LOG.skipped] or ["- none"]) + [""]
    readme += ["## Result files present but not used in this batch", ""]
    readme += [f"- `{f}`" for f in unused] or ["- none"]
    (folder / "README.md").write_text("\n".join(readme) + "\n")
    update_index(args.date, topic, folder, pngs, summary_lines)

    print(f"\nbatch folder: {rel(folder)}")
    for p in pngs:
        print(f"  {rel(folder / p)}")
    print(f"skipped: {len(LOG.skipped)}; unused result files: {len(unused)}")
    for f in unused:
        print(f"  unused: {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
