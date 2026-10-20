"""I3-2: multiplicative correction restricted to polarity and terminal bias.

``I_corr = I_sim * exp(g)``, ``g = OUT_SCALE * tanh(h(x))`` with ``x`` a
column subset of the E1 features (``e1_correction.raw_features``):

* ``M_bias``: is_pmos, |Vgs|/VDD, |Vds|/VDD  (columns 0, 1, 2);
* ``M_bias_geom``: the same plus log10 L, log10 W  (columns 0-4).

``h`` is an MLP (``HYPER_BIAS``: 2 hidden layers of 16 tanh units), trained
with the E1 loss (mean paper per-device RRMS of the corrected current,
``e1_correction._loss_fn``) and the E1 M optimizer recipe.
``BiasCorrection`` subclasses ``e1_correction.Correction``, so it works with
``Correction.apply``, ``i2_1_joint.CorrectedProblem`` and the I2-1 held-out
refit task unchanged.

Also: the paired geometric-mean ratio statistic with a (cluster) bootstrap
CI (``paired_ratio``) and the Verilog-A text of a trained ``M_bias`` /
``M_bias_geom`` (``to_verilog_a_bias``; not compiled here).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .e1_correction import (OUT_SCALE, Correction, DeviceRec, _loss_fn,
                            make_batch)

ARM_COLUMNS = {"M_bias": (0, 1, 2), "M_bias_geom": (0, 1, 2, 3, 4)}
HYPER_BIAS = {"hidden": 16, "layers": 2, "l2": 1e-4, "lr": 3e-3, "steps": 2000,
              "seeds": (0, 1, 2, 3, 4)}


@dataclass
class BiasCorrection(Correction):
    """MLP correction on a column subset ``cols`` of the E1 features.

    ``mu``/``sd`` and the network act on the subset; ``g`` takes the full
    six-column E1 feature matrix (group labels are ignored)."""
    cols: tuple = field(default_factory=tuple)

    def g(self, phi: np.ndarray, group_tag=None, group_kind=None) -> np.ndarray:
        x = np.asarray(phi, dtype=np.float64)[:, list(self.cols)]
        return super().g(x)

    def to_json(self) -> dict:
        return {**super().to_json(), "cols": list(self.cols)}

    @classmethod
    def from_json(cls, row: dict) -> "BiasCorrection":
        base = Correction.from_json(row)
        return cls(base.arm, base.mu, base.sd, base.nets, base.table, base.lin_w,
                   cols=tuple(int(c) for c in row["cols"]))


def correction_from_json(row: dict) -> Correction:
    """``BiasCorrection`` for rows with ``cols``, else an E1 ``Correction``."""
    return BiasCorrection.from_json(row) if "cols" in row else Correction.from_json(row)


def fit_bias(arm: str, train: list[DeviceRec], hyper: dict | None = None,
             cols: tuple | None = None, threads: int = 1,
             verbose: bool = False) -> BiasCorrection:
    """Train an ``M_bias``-type MLP ensemble on ``train``.

    Same recipe as ``e1_correction.fit_arm('M')`` (standardization with the
    training-point mean/std, is_pmos kept 0/1, zero-initialized last layer,
    ``torch.manual_seed(seed)`` before each member, L2 on weight matrices,
    full-batch Adam, float64) on the feature columns ``cols``
    (default ``ARM_COLUMNS[arm]``)."""
    import torch
    torch.set_num_threads(threads)
    hp = {**HYPER_BIAS, **(hyper or {})}
    cols = tuple(ARM_COLUMNS[arm] if cols is None else cols)
    batch = make_batch(train)
    loss = _loss_fn(batch)
    phi = batch.phi[:, list(cols)]
    mu = phi.mean(axis=0)
    sd = phi.std(axis=0)
    if cols and cols[0] == 0:
        mu[0], sd[0] = 0.0, 1.0                    # is_pmos stays 0/1
    sd = np.where(sd > 1e-12, sd, 1.0)
    z = torch.tensor((phi - mu) / sd, dtype=torch.float64)
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
                print(f"  {arm} seed {seed} step {step} loss {float(value):.5f}", flush=True)
        nets.append([(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy())
                     for m in mods])
    return BiasCorrection(arm, mu, sd, nets=nets, cols=cols)


def split_seed_tasks(tasks: list, hyper: dict) -> tuple[list, list]:
    """``tasks``: [(arm, recs), ...] -> one (arm, recs, hyper) task per seed and
    the owner index of each.  Members train independently from
    ``torch.manual_seed(seed)`` on the same standardized batch, so
    ``merge_seed_results`` reproduces ``fit_bias`` with all seeds."""
    seeds = {**HYPER_BIAS, **hyper}["seeds"]
    expanded, owner = [], []
    for i, (arm, recs) in enumerate(tasks):
        for seed in seeds:
            expanded.append((arm, recs, {**hyper, "seeds": (seed,)}))
            owner.append(i)
    return expanded, owner


def merge_seed_results(n_tasks: int, owner: list, results: list) -> list:
    merged: list = [None] * n_tasks
    for i, corr in zip(owner, results):
        if merged[i] is None:
            merged[i] = corr
            continue
        m = merged[i]
        if not (np.array_equal(corr.mu, m.mu) and np.array_equal(corr.sd, m.sd)
                and corr.cols == m.cols and corr.arm == m.arm):
            raise RuntimeError("per-seed corrections do not share feature scaling")
        merged[i] = BiasCorrection(m.arm, m.mu, m.sd, nets=m.nets + corr.nets, cols=m.cols)
    return merged


# ---------------------------------------------------------------------------
# Decision statistic
# ---------------------------------------------------------------------------
def paired_ratio(method: dict, control: dict, clusters: dict | None = None,
                 n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired geometric-mean ratio exp(mean ln(method / control)).

    Devices where either value is non-finite (or <= 0) are left out of the
    ratio and counted.  Bootstrap: percentile 95% CI over ``n_boot``
    resamples (``default_rng(seed)``) of devices, or of clusters when
    ``clusters`` maps device -> cluster id (a resampled cluster contributes
    all its devices).  ``wins``: devices with method < control."""
    tags = sorted(set(method) & set(control))
    a = np.array([method[t] for t in tags], dtype=float)
    b = np.array([control[t] for t in tags], dtype=float)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    lr = np.log(a[ok] / b[ok])
    used = [t for t, k in zip(tags, ok) if k]
    out = {"n": len(tags), "n_used": int(ok.sum()),
           "n_method_failed": int(np.sum(~np.isfinite(a))),
           "n_control_failed": int(np.sum(~np.isfinite(b))),
           "wins": int(np.sum(a[ok] < b[ok])), "ties": int(np.sum(a[ok] == b[ok]))}
    if not len(lr):
        return {**out, "ratio": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan")}
    out["ratio"] = float(np.exp(lr.mean()))
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx = rng.integers(0, len(lr), size=(n_boot, len(lr)))
        boots = lr[idx].mean(axis=1)
        out["resample_unit"] = "device"
    else:
        names = sorted({clusters[t] for t in used})
        members = [np.array([i for i, t in enumerate(used) if clusters[t] == c]) for c in names]
        sums = np.array([lr[m].sum() for m in members])
        sizes = np.array([len(m) for m in members], dtype=float)
        pick = rng.integers(0, len(names), size=(n_boot, len(names)))
        boots = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)
        out["resample_unit"] = f"cluster ({len(names)})"
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out["ci_low"], out["ci_high"] = float(np.exp(lo)), float(np.exp(hi))
    out["n_boot"] = int(n_boot)
    return out


# ---------------------------------------------------------------------------
# Verilog-A export (text only; not compiled here)
# ---------------------------------------------------------------------------
VA_INPUTS = {0: "pmos", 1: "abs(V(g, s)) / VDD", 2: "abs(V(d, s)) / VDD",
             3: "log(L)", 4: "log(W)"}   # Verilog-A log() is base 10


def to_verilog_a_bias(corr: BiasCorrection, module: str = "i3_2_bias_corr",
                      vdd: float = 1.85) -> str:
    """Verilog-A text of a ``BiasCorrection`` (columns within 0-4).

    Ports (d, g, s, di): the compact-model drain connects to ``di``; the
    zero-volt branch d->di senses the drain current entering the compact
    model, I_sim; the module adds (exp(g) - 1) * I_sim from d to s, so the
    current entering terminal d is I_sim * exp(g).  ``g`` depends on the
    parameter ``pmos`` (0/1), V(g, s), V(d, s) and, for ``M_bias_geom``, the
    parameters L and W (um); it does not depend on the sensed current."""
    if any(c not in VA_INPUTS for c in corr.cols):
        raise ValueError(f"columns {corr.cols} include a current-dependent feature")
    fmt = lambda v: f"{float(v):.10e}"
    n_in = len(corr.cols)
    hidden = max(w.shape[1] for layers in corr.nets for w, _ in layers[:-1])
    lines = [
        "`include \"disciplines.vams\"",
        "`include \"constants.vams\"",
        f"// I3-2 multiplicative correction ({corr.arm}), {len(corr.nets)}-member "
        f"{n_in}-{hidden}-...-1 tanh MLP ensemble.",
        "// I(d->s) total = I_sim * exp(g); g = mean over members of tanh(h(x)),",
        "// x = standardized [" + ", ".join(VA_INPUTS[c] for c in corr.cols) + "].",
        "// abs() maps pMOS and nMOS forward bias to the same range; reverse-mode",
        "// bias (Vds of the opposite sign) is folded onto forward mode.",
        f"module {module}(d, g, s, di);",
        "  inout d, g, s, di;",
        "  electrical d, g, s, di;",
        "  branch (d, di) sense;   // I(sense) = drain current into the compact model",
        "  parameter real L = 1.0;   // um",
        "  parameter real W = 1.0;   // um",
        "  parameter real pmos = 0;  // 0 nMOS, 1 pMOS",
        f"  parameter real VDD = {vdd};",
        "  real ib, gsum, acc, " + ", ".join(f"x{i}" for i in range(n_in)) + ";",
        "  real " + ", ".join(f"h{i}" for i in range(hidden)) + ";",
        "  real " + ", ".join(f"k{i}" for i in range(hidden)) + ";",
        "  analog begin",
        "    V(sense) <+ 0.0;",
        "    ib = I(sense);",
    ]
    for i, c in enumerate(corr.cols):
        lines.append(f"    x{i} = ({VA_INPUTS[c]} - {fmt(corr.mu[i])}) / {fmt(corr.sd[i])};")
    lines.append("    gsum = 0.0;")
    xs = [f"x{i}" for i in range(n_in)]
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
        "//   M1 di g s b <compact-model card> L=.. W=..",
        f"//   YCORR d g s di {module} L=.. W=.. pmos=..",
    ]
    return "\n".join(lines) + "\n"


def eval_verilog_a_g(text: str, pmos: float, vgs: float, vds: float,
                     L: float = 1.0, W: float = 1.0, vdd: float | None = None) -> float:
    """Evaluate ``g`` from the assignment lines of ``to_verilog_a_bias`` text
    (independent check of the emitted arithmetic)."""
    body = text.split("ib = I(sense);", 1)[1].split("I(d, s) <+", 1)[0]
    if vdd is None:
        vdd = float(text.split("parameter real VDD = ", 1)[1].split(";", 1)[0])
    env = {"tanh": math.tanh, "abs": abs, "log": math.log10, "pmos": float(pmos),
           "L": float(L), "W": float(W), "VDD": vdd, "VGS": float(vgs), "VDS": float(vds)}
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        name, expr = line.rstrip(";").split(" = ", 1)
        expr = expr.replace("V(g, s)", "VGS").replace("V(d, s)", "VDS")
        env[name.strip()] = eval(expr, {"__builtins__": {}}, env)  # noqa: S307
    n = int(text.split("I(d, s) <+ (exp(gsum / ", 1)[1].split(")", 1)[0])
    return env["gsum"] / n
