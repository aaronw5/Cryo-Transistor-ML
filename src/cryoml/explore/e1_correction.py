"""E1: cross-device multiplicative correction to a compact-model simulation.

``I_corr = I_sim * exp(g(phi))`` with ``g`` bounded by ``OUT_SCALE * tanh``.
Features ``phi`` per bias point: is_pmos, |Vgs|/VDD, |Vds|/VDD, log10 L,
log10 W, log10(|I_sim| L / W + 1e-13).  Curve kind and fixed-bias tag are not
features, so the correction is a function of the terminal state and can be
written as a Verilog-A element (``to_verilog_a``).

Device records (``DeviceRec``) hold, per metric curve, the full bias arrays,
the simulated and the target current, the trim start and the paper
denominator.  Training minimizes the paper per-device RRMS (mean over the
included curves of trimmed RMSE / trimmed mean |I_target|), averaged over
devices.  Scoring is done by the caller with the official scorer; this
module only returns corrected current arrays.

Arms (``fit_arm``): ``T1`` per-(family, curve tag) gain, ``T2`` per-(family,
kind) gain, ``LIN`` quadratic-feature linear model, ``M`` MLP ensemble.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

from ..metrics import clean_current, device_rrms_new

OUT_SCALE = 1.0
I_EPS = 1e-13


@dataclass
class CurveRec:
    key: str                  # "idvd@0.37" (canonical tag)
    kind: str                 # "idvd" | "idvg"
    vg: np.ndarray            # gate-source bias (signed, as simulated)
    vd: np.ndarray            # drain-source bias
    sim: np.ndarray           # compact-model current on the grid
    target: np.ndarray        # measured (or truth) current, raw
    start: int                # trim start (metric preprocessing)
    den: float                # trimmed mean |target|
    included: bool            # frozen inclusion


@dataclass
class DeviceRec:
    tag: str
    family: str               # "nmos" | "pmos"
    L_um: float
    W_um: float
    vdd: float
    source: str               # "sky130" | "gf180" | "ihp"
    curves: list[CurveRec] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def geometry(self) -> str:
        return f"{self.family}_L{self.L_um:g}_W{self.W_um:g}"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def save_devices(path: Path, devices: list[DeviceRec]) -> None:
    path = Path(path)
    arrays, meta = {}, []
    for d_i, dev in enumerate(devices):
        rows = []
        for c_i, c in enumerate(dev.curves):
            for name in ("vg", "vd", "sim", "target"):
                arrays[f"d{d_i}_c{c_i}_{name}"] = np.asarray(getattr(c, name),
                                                             dtype=np.float64)
            rows.append({"key": c.key, "kind": c.kind, "start": int(c.start),
                         "den": float(c.den), "included": bool(c.included)})
        meta.append({"tag": dev.tag, "family": dev.family, "L_um": dev.L_um,
                     "W_um": dev.W_um, "vdd": dev.vdd, "source": dev.source,
                     "meta": dev.meta, "curves": rows})
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")


def load_devices(path: Path) -> list[DeviceRec]:
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    out = []
    with np.load(path.with_suffix(".npz")) as data:
        for d_i, m in enumerate(meta):
            dev = DeviceRec(m["tag"], m["family"], m["L_um"], m["W_um"], m["vdd"],
                            m["source"], meta=m.get("meta", {}))
            for c_i, row in enumerate(m["curves"]):
                arr = {n: np.asarray(data[f"d{d_i}_c{c_i}_{n}"])
                       for n in ("vg", "vd", "sim", "target")}
                dev.curves.append(CurveRec(row["key"], row["kind"], arr["vg"],
                                           arr["vd"], arr["sim"], arr["target"],
                                           row["start"], row["den"], row["included"]))
            out.append(dev)
    return out


# ---------------------------------------------------------------------------
# Strict scoring
# ---------------------------------------------------------------------------
def score_device(dev: DeviceRec, sims: list[np.ndarray]) -> dict:
    """Strict paper RRMS of ``sims`` (one array per ``dev.curves`` entry).

    sky130: ``metrics.device_rrms_new`` with the frozen inclusion stored on
    the record (NaN when an included curve is non-finite).  pdk2: the
    ``pdk2.metric.curve_rrms`` recipe (target-frozen inclusion), with NaN in
    place of the optimizers' PENALTY for a non-finite included curve."""
    if dev.source == "sky130":
        tsim, tmeas, include = {}, {}, set()
        for c, s in zip(dev.curves, sims):
            kind, bias = c.key.split("@")
            tsim[(kind, float(bias))] = np.asarray(s, dtype=np.float64)
            tmeas[(kind, float(bias))] = np.asarray(c.target, dtype=np.float64)
            if c.included:
                include.add(c.key)
        r = device_rrms_new(dev.meta["dev_type"], dev.L_um, dev.W_um, tsim, tmeas,
                            include_tags=include)
        return {"rrms": float(r["rrms"]), "n_curves": r["n_curves"],
                "failed": list(r["failed_tags"]),
                "per_curve": {k: v["rrms"] for k, v in r["per_curve"].items()
                              if v.get("included")}}
    per, failed = {}, []
    for c, s in zip(dev.curves, sims):
        if not c.included:
            continue
        sim = np.asarray(s, dtype=np.float64)[c.start:]
        tgt = clean_current(c.target)[c.start:]
        if not np.all(np.isfinite(sim)):
            failed.append(c.key)
            continue
        per[c.key] = math.sqrt(float(np.mean((sim - tgt) ** 2))) / c.den
    rrms = float(np.mean(list(per.values()))) if per and not failed else float("nan")
    return {"rrms": rrms, "n_curves": len(per), "failed": failed, "per_curve": per}


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
FEATURE_NAMES = ("is_pmos", "abs_vgs_over_vdd", "abs_vds_over_vdd", "log10_L",
                 "log10_W", "log10_abs_isim_L_over_W")


def raw_features(dev: DeviceRec, vg: np.ndarray, vd: np.ndarray,
                 sim: np.ndarray) -> np.ndarray:
    n = len(sim)
    with np.errstate(invalid="ignore", divide="ignore"):
        logi = np.log10(np.abs(np.asarray(sim, dtype=np.float64))
                        * dev.L_um / dev.W_um + I_EPS)
    return np.column_stack([
        np.full(n, 1.0 if dev.family == "pmos" else 0.0),
        np.abs(vg) / dev.vdd, np.abs(vd) / dev.vdd,
        np.full(n, math.log10(dev.L_um)), np.full(n, math.log10(dev.W_um)),
        logi])


@dataclass
class Batch:
    """Flattened trimmed included points of a list of devices."""
    phi: np.ndarray           # (P, 6) raw features
    sim: np.ndarray           # (P,)
    target: np.ndarray        # (P,)
    curve_id: np.ndarray      # (P,) index into curve tables
    curve_dev: np.ndarray     # (C,) device index of each curve
    curve_den: np.ndarray     # (C,)
    curve_n: np.ndarray       # (C,) points per curve
    curve_group_tag: list     # (C,) "family|key"
    curve_group_kind: list    # (C,) "family|kind"
    point_group_tag: list     # (P,)
    point_group_kind: list
    n_devices: int


def make_batch(devices: list[DeviceRec]) -> Batch:
    phi, sim, tgt, cid = [], [], [], []
    cdev, cden, cn, gtag, gkind, ptag, pkind = [], [], [], [], [], [], []
    for d_i, dev in enumerate(devices):
        for c in dev.curves:
            if not c.included or not (c.den > 0):
                continue
            s = slice(c.start, None)
            t = clean_current(c.target)[s]
            x = raw_features(dev, c.vg[s], c.vd[s], c.sim[s])
            k = len(cdev)
            phi.append(x); sim.append(np.asarray(c.sim[s], float)); tgt.append(t)
            cid.append(np.full(len(t), k))
            cdev.append(d_i); cden.append(c.den); cn.append(len(t))
            gtag.append(f"{dev.family}|{c.key}"); gkind.append(f"{dev.family}|{c.kind}")
            ptag += [gtag[-1]] * len(t); pkind += [gkind[-1]] * len(t)
    return Batch(np.concatenate(phi), np.concatenate(sim), np.concatenate(tgt),
                 np.concatenate(cid), np.asarray(cdev), np.asarray(cden),
                 np.asarray(cn), gtag, gkind, ptag, pkind, len(devices))


# ---------------------------------------------------------------------------
# Models (numpy inference; torch only for training)
# ---------------------------------------------------------------------------
@dataclass
class Correction:
    arm: str
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None
    nets: list = field(default_factory=list)     # list of list of (W, b)
    table: dict = field(default_factory=dict)    # group -> log-gain
    lin_w: np.ndarray | None = None

    # g for points of a device given raw features and group labels
    def g(self, phi: np.ndarray, group_tag=None, group_kind=None) -> np.ndarray:
        if self.arm == "C0":
            return np.zeros(len(phi))
        if self.arm in ("T1", "T2"):
            groups = group_tag if self.arm == "T1" else group_kind
            return np.array([self.table.get(gr, 0.0) for gr in groups])
        z = (phi - self.mu) / self.sd
        if self.arm == "LIN":
            return OUT_SCALE * np.tanh(quad_features(z) @ self.lin_w)
        outs = []
        for layers in self.nets:
            h = z
            for i, (w, b) in enumerate(layers):
                h = h @ w + b
                if i < len(layers) - 1:
                    h = np.tanh(h)
            outs.append(OUT_SCALE * np.tanh(h[:, 0]))
        return np.mean(outs, axis=0)

    def apply(self, dev: DeviceRec) -> list[np.ndarray]:
        """Corrected current for every curve of ``dev`` (full grids)."""
        out = []
        for c in dev.curves:
            phi = raw_features(dev, c.vg, c.vd, c.sim)
            n = len(c.sim)
            g = self.g(phi, [f"{dev.family}|{c.key}"] * n,
                       [f"{dev.family}|{c.kind}"] * n)
            out.append(np.asarray(c.sim, float) * np.exp(g))
        return out

    def to_json(self) -> dict:
        return {"arm": self.arm, "out_scale": OUT_SCALE, "i_eps": I_EPS,
                "feature_names": FEATURE_NAMES,
                "mu": None if self.mu is None else self.mu.tolist(),
                "sd": None if self.sd is None else self.sd.tolist(),
                "nets": [[(w.tolist(), b.tolist()) for w, b in layers]
                         for layers in self.nets],
                "table": self.table,
                "lin_w": None if self.lin_w is None else self.lin_w.tolist()}

    @classmethod
    def from_json(cls, row: dict) -> "Correction":
        arr = lambda v: None if v is None else np.asarray(v, dtype=np.float64)
        return cls(row["arm"], arr(row["mu"]), arr(row["sd"]),
                   [[(np.asarray(w), np.asarray(b)) for w, b in layers]
                    for layers in row["nets"]], dict(row["table"]),
                   arr(row["lin_w"]))


def quad_features(z: np.ndarray) -> np.ndarray:
    n, d = z.shape
    cols = [np.ones(n)] + [z[:, i] for i in range(d)]
    cols += [z[:, i] * z[:, j] for i, j in combinations_with_replacement(range(d), 2)]
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
HYPER = {"M": {"hidden": 32, "layers": 2, "l2": 1e-4, "lr": 3e-3, "steps": 2000,
               "seeds": (0, 1, 2, 3, 4)},
         "LIN": {"l2": 1e-3, "lr": 3e-3, "steps": 2000},
         "T1": {"lr": 2e-2, "steps": 1000}, "T2": {"lr": 2e-2, "steps": 1000}}


def _loss_fn(batch: Batch):
    import torch
    sim = torch.tensor(batch.sim, dtype=torch.float64)
    tgt = torch.tensor(batch.target, dtype=torch.float64)
    cid = torch.tensor(batch.curve_id, dtype=torch.long)
    den = torch.tensor(batch.curve_den, dtype=torch.float64)
    cn = torch.tensor(batch.curve_n, dtype=torch.float64)
    cdev = torch.tensor(batch.curve_dev, dtype=torch.long)
    n_curves_dev = torch.zeros(batch.n_devices, dtype=torch.float64).index_add_(
        0, cdev, torch.ones(len(cdev), dtype=torch.float64))
    present = n_curves_dev > 0

    def loss(g):
        err2 = (sim * torch.exp(g) - tgt) ** 2
        mse = torch.zeros(len(den), dtype=torch.float64).index_add_(0, cid, err2) / cn
        rrms = torch.sqrt(mse + 1e-30) / den
        dev = torch.zeros(batch.n_devices, dtype=torch.float64).index_add_(0, cdev, rrms)
        return (dev[present] / n_curves_dev[present]).mean()
    return loss


def fit_arm(arm: str, train: list[DeviceRec], hyper: dict | None = None,
            verbose: bool = False, threads: int = 2) -> Correction:
    import torch
    torch.set_num_threads(threads)
    if arm == "C0":
        return Correction("C0")
    hp = {**HYPER[arm], **(hyper or {})}
    batch = make_batch(train)
    loss = _loss_fn(batch)
    if arm in ("T1", "T2"):
        groups = batch.point_group_tag if arm == "T1" else batch.point_group_kind
        names = sorted(set(groups))
        idx = torch.tensor([names.index(g) for g in groups], dtype=torch.long)
        c = torch.zeros(len(names), dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([c], lr=hp["lr"])
        for _ in range(hp["steps"]):
            opt.zero_grad()
            value = loss(OUT_SCALE * torch.tanh(c[idx]))
            value.backward()
            opt.step()
        vals = (OUT_SCALE * torch.tanh(c)).detach().numpy()
        return Correction(arm, table={n: float(v) for n, v in zip(names, vals)})

    mu = batch.phi.mean(axis=0)
    sd = batch.phi.std(axis=0)
    mu[0], sd[0] = 0.0, 1.0                       # is_pmos stays 0/1
    sd = np.where(sd > 1e-12, sd, 1.0)
    z = torch.tensor((batch.phi - mu) / sd, dtype=torch.float64)
    if arm == "LIN":
        q = torch.tensor(quad_features(z.numpy()), dtype=torch.float64)
        w = torch.zeros(q.shape[1], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([w], lr=hp["lr"])
        for _ in range(hp["steps"]):
            opt.zero_grad()
            value = loss(OUT_SCALE * torch.tanh(q @ w)) + hp["l2"] * (w[1:] ** 2).sum()
            value.backward()
            opt.step()
        return Correction("LIN", mu, sd, lin_w=w.detach().numpy())

    nets = []
    for seed in hp["seeds"]:
        torch.manual_seed(seed)
        dims = [z.shape[1]] + [hp["hidden"]] * hp["layers"] + [1]
        mods = [torch.nn.Linear(a, b).double() for a, b in zip(dims[:-1], dims[1:])]
        torch.nn.init.zeros_(mods[-1].weight)
        torch.nn.init.zeros_(mods[-1].bias)
        params = [p for m in mods for p in m.parameters()]
        opt = torch.optim.Adam(params, lr=hp["lr"])

        def forward(x):
            for i, m in enumerate(mods):
                x = m(x)
                if i < len(mods) - 1:
                    x = torch.tanh(x)
            return OUT_SCALE * torch.tanh(x[:, 0])
        for step in range(hp["steps"]):
            opt.zero_grad()
            value = loss(forward(z)) + hp["l2"] * sum((m.weight ** 2).sum() for m in mods)
            value.backward()
            opt.step()
            if verbose and step % 500 == 0:
                print(f"  seed {seed} step {step} loss {float(value):.5f}", flush=True)
        nets.append([(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy())
                     for m in mods])
    return Correction("M", mu, sd, nets=nets)


# ---------------------------------------------------------------------------
# Verilog-A export (sketch; not compiled here)
# ---------------------------------------------------------------------------
def to_verilog_a(corr: Correction, module: str = "e1_corr", vdd: float = 1.85) -> str:
    """Verilog-A text of an ``M`` correction.

    Ports (d, g, s, di): the compact-model drain connects to ``di``; the
    module holds a zero-volt branch di->d that senses I_bsim and adds
    (exp(g) - 1) * I_bsim from d to s.  The ensemble average of ``g`` is
    unrolled member by member.  Parameters: L, W in um and pmos (0/1)."""
    if corr.arm != "M":
        raise ValueError("Verilog-A export is written for the MLP arm")
    fmt = lambda v: f"{float(v):.10e}"
    lines = [
        "`include \"disciplines.vams\"",
        "`include \"constants.vams\"",
        f"// E1 multiplicative correction, {len(corr.nets)}-member MLP ensemble.",
        "// I(d->s) total = I_bsim * exp(g(phi)); phi = [pmos, |Vgs|/VDD, |Vds|/VDD,",
        "// log10 L, log10 W, log10(|I_bsim| L / W + 1e-13)] standardized.",
        f"module {module}(d, g, s, di);",
        "  inout d, g, s, di;",
        "  electrical d, g, s, di;",
        "  branch (di, d) sense;",
        "  parameter real L = 1.0;   // um",
        "  parameter real W = 1.0;   // um",
        "  parameter real pmos = 0;  // 0 nMOS, 1 pMOS",
        f"  parameter real VDD = {vdd};",
        "  real ib, gsum, x0, x1, x2, x3, x4, x5, acc;",
    ]
    hidden = max(w.shape[1] for layers in corr.nets for w, _ in layers[:-1])
    lines.append("  real " + ", ".join(f"h{i}" for i in range(hidden)) + ";")
    lines.append("  real " + ", ".join(f"k{i}" for i in range(hidden)) + ";")
    lines += [
        "  analog begin",
        "    V(sense) <+ 0.0;",
        "    ib = I(sense);",
        f"    x0 = (pmos - {fmt(corr.mu[0])}) / {fmt(corr.sd[0])};",
        f"    x1 = (abs(V(g, s)) / VDD - {fmt(corr.mu[1])}) / {fmt(corr.sd[1])};",
        f"    x2 = (abs(V(d, s)) / VDD - {fmt(corr.mu[2])}) / {fmt(corr.sd[2])};",
        f"    x3 = (log(L) - {fmt(corr.mu[3])}) / {fmt(corr.sd[3])};",
        f"    x4 = (log(W) - {fmt(corr.mu[4])}) / {fmt(corr.sd[4])};",
        f"    x5 = (log(abs(ib) * L / W + {I_EPS:g}) - {fmt(corr.mu[5])}) / {fmt(corr.sd[5])};",
        "    gsum = 0.0;",
    ]
    xs = [f"x{i}" for i in range(6)]
    for m_i, layers in enumerate(corr.nets):
        lines.append(f"    // ensemble member {m_i}")
        inputs = xs
        for l_i, (w, b) in enumerate(layers):
            last = l_i == len(layers) - 1
            outs = []
            for j in range(w.shape[1]):
                terms = " + ".join(f"{fmt(w[i, j])} * {inputs[i]}" for i in range(w.shape[0]))
                name = "acc" if last else (f"h{j}" if l_i % 2 == 0 else f"k{j}")
                expr = f"{fmt(b[j])} + {terms}"
                lines.append(f"    {name} = {expr if last else f'tanh({expr})'};")
                outs.append(name)
            inputs = outs
        lines.append(f"    gsum = gsum + {OUT_SCALE:g} * tanh(acc);")
    lines += [
        f"    I(d, s) <+ (exp(gsum / {len(corr.nets)}) - 1.0) * ib;",
        "  end",
        "endmodule",
        "",
        "// Netlist use (one transistor):",
        "//   M1 di g s b <bsim4 card> L=.. W=..",
        f"//   YCORR d g s di {module} L=.. W=.. pmos=..",
    ]
    return "\n".join(lines) + "\n"
