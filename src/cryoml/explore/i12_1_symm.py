"""I12-1: bias-only correction g v2 for sky130 77 K, built as a deployable
compact-model add-on (fixes of the I11-1 failures).

Pre-registration: ``out/research/explore/I12_1/prereg.md`` / ``prereg.json``.

Changes from M_bias v1 (``i3_2_bias``, I3-2 recipe), all in this module:

1. Source/drain-symmetric inputs.  With s = +1 (nMOS) / -1 (pMOS) and the
   terminal voltages V(g,s), V(g,d) = V(g,s) - V(d,s), V(d,s):

       a  = s V(g,s),  b = s V(g,d),  m = sabs(V(d,s)),
       x1 = (a + b + m) / (2 VDD),     x2 = m / VDD,
       sabs(x) = sqrt(x^2 + EPS^2).

   x1 is a smooth max(a, b) / VDD (gate drive from the terminal that acts as
   the source), x2 a smooth |V(d,s)| / VDD.  Exchanging d and s maps
   (a, b, V(d,s)) to (b, a, -V(d,s)), so (x1, x2) and g are unchanged and
   the corrected current I_card x exp(g) changes sign with I_card.  In
   forward operation (s V(d,s) >= 0) x1 = (2 a - s V(d,s) + m) / (2 VDD),
   which differs from a / VDD only by (m - |V(d,s)|) / (2 VDD) <= EPS / (2 VDD).
   Network input: (is_pmos, x1, x2), standardized with the training-point
   mean / std (is_pmos kept 0/1), as in v1.
2. exp(g) multiplies the static (DC) drain-terminal current of the card
   only.  The add-on element senses a charge-free replica of the card
   (``replica_corner``: the same subcircuit and bins with xpart = -1, which
   skips the BSIM4 intrinsic charge, and zero overlap, fringing and junction
   capacitance parameters; no DC parameter changes) whose terminals are
   driven by unit-gain VCVS copies of the transistor's terminal voltages, so
   the replica draws no current from the circuit.  The element adds
   (exp(g) - 1) x I_replica from d to s: a static current, no charge.
3. Training penalty on negative gm and gds of the corrected current
   (``penalty_terms``): on a 25 mV grid in the forward bias quadrant of each
   training device's measured range (|Vgs| 0-1.85 V, |Vds| 0.025-1.85 V),
   at points where the training card's forward current exceeds 1 nA,

       r_gm  = gm_card / I_card  + dg/du_g,   r_gds = gds_card / I_card + dg/du_d

   (u = s V, forward-positive; gm_card / I_card and gds_card / I_card from
   central differences of the card current on the grid; dg/du by autograd),
   P = mean over training devices of the per-device mean of
   relu(-r_gm) + relu(-r_gds)  (1/V).  The sign of r equals the sign of the
   corrected derivative because exp(g) I_card > 0 in forward operation.
4. No abs(): sabs() above, EPS pre-registered.

Everything else is the v1 recipe: 5 seeds, 3-16-16-1 tanh MLP, zero-init
last layer, OUT_SCALE x tanh output, Adam lr 3e-3, 2000 full-batch steps,
L2 1e-4 on the weight matrices, float64, E1 loss (mean per-device paper RRMS
of the corrected current).

Also: the Verilog-A text of a trained v2 correction (``to_verilog_a_v2``) and
an evaluator of its assignment lines, the add-on netlist
(``instance_v2``), the replica library, the Python reference model of the
element (current and small-signal admittance), validity row selections and
the pre-registered decision (``decide_accuracy``, ``decide_deploy``,
``decide_outcome``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .e1_correction import OUT_SCALE, Correction, DeviceRec, _loss_fn, make_batch

VDD_NORM = 1.85
EPS = 0.01                     # V, smooth-abs constant (pre-registered)
LAMBDA = 10.0                  # penalty weight, units V (P is in 1/V); pre-registered
PEN_STEP = 0.025               # V, penalty grid step
PEN_N = 75                     # grid values 0 .. 1.85 V
PEN_I_FLOOR = 1e-9             # A, penalty points need forward card current > floor
HYPER_V2 = {"hidden": 16, "layers": 2, "l2": 1e-4, "lr": 3e-3, "steps": 2000,
            "seeds": (0, 1, 2, 3, 4)}
VA_MODULE_V2 = "i12_1_bias_corr_v2"

# Equivalence criterion OSDI vs Python (per point): |dI| <= REL |I_py| + ABS
EQ_REL = 1e-6
EQ_ABS = 1e-13

# Replica: parameters set in every bin of the replica subcircuit
REPLICA_ZERO = {"xpart": -1, "cgso": 0, "cgdo": 0, "cgbo": 0, "cgsl": 0, "cgdl": 0,
                "cf": 0, "cjs": 0, "cjd": 0, "cjsws": 0, "cjswd": 0, "cjswgs": 0,
                "cjswgd": 0}
REPLICA_SUFFIX = "_rep"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def sabs(x, eps: float = EPS):
    return np.sqrt(np.asarray(x, dtype=np.float64) ** 2 + eps * eps)


def sym_inputs(is_pmos: bool, vgs, vds, eps: float = EPS, vdd: float = VDD_NORM) -> np.ndarray:
    """(N, 3) network inputs [is_pmos, x1, x2] at signed terminal voltages
    V(g,s) = vgs, V(d,s) = vds (any operating quadrant)."""
    vgs = np.atleast_1d(np.asarray(vgs, dtype=np.float64)).ravel()
    vds = np.atleast_1d(np.asarray(vds, dtype=np.float64)).ravel()
    vgs, vds = np.broadcast_arrays(vgs, vds)
    s = -1.0 if is_pmos else 1.0
    a = s * vgs
    b = s * (vgs - vds)
    m = sabs(vds, eps)
    return np.column_stack([np.full(vgs.size, 1.0 if is_pmos else 0.0),
                            (a + b + m) / (2.0 * vdd), m / vdd])


def abs_inputs(is_pmos: bool, vgs, vds, vdd: float = VDD_NORM) -> np.ndarray:
    """v1 inputs [is_pmos, |Vgs|/VDD, |Vds|/VDD] (E1 feature columns 0-2);
    used only by the trainer reproduction check (validity V1)."""
    vgs = np.atleast_1d(np.asarray(vgs, dtype=np.float64)).ravel()
    vds = np.atleast_1d(np.asarray(vds, dtype=np.float64)).ravel()
    vgs, vds = np.broadcast_arrays(vgs, vds)
    return np.column_stack([np.full(vgs.size, 1.0 if is_pmos else 0.0),
                            np.abs(vgs) / vdd, np.abs(vds) / vdd])


INPUTS = {"sym_v2": sym_inputs, "abs_v1": lambda p, vg, vd, eps=EPS, vdd=VDD_NORM:
          abs_inputs(p, vg, vd, vdd)}


def torch_sym_forward(ug, ud, eps: float = EPS, vdd: float = VDD_NORM):
    """x1, x2 from forward-positive u_g = s V(g,s), u_d = s V(d,s) (torch)."""
    import torch
    m = torch.sqrt(ud * ud + eps * eps)
    return (ug + (ug - ud) + m) / (2.0 * vdd), m / vdd


# ---------------------------------------------------------------------------
# Correction
# ---------------------------------------------------------------------------
@dataclass
class SymCorrection(Correction):
    """MLP ensemble on ``INPUTS[inputs]``; ``apply`` uses the signed bias
    arrays of each curve (vg = V(g,s), vd = V(d,s), source at 0)."""
    inputs: str = "sym_v2"
    eps: float = EPS
    vdd: float = VDD_NORM

    def features(self, is_pmos: bool, vgs, vds) -> np.ndarray:
        return INPUTS[self.inputs](is_pmos, vgs, vds, eps=self.eps, vdd=self.vdd)

    def g_x(self, x: np.ndarray) -> np.ndarray:
        """g at network inputs ``x`` (N, 3), before standardization."""
        z = (np.asarray(x, dtype=np.float64) - self.mu) / self.sd
        outs = []
        for layers in self.nets:
            h = z
            for i, (w, b) in enumerate(layers):
                h = h @ w + b
                if i < len(layers) - 1:
                    h = np.tanh(h)
            outs.append(OUT_SCALE * np.tanh(h[:, 0]))
        return np.mean(outs, axis=0)

    def g_v(self, is_pmos: bool, vgs, vds) -> np.ndarray:
        """g at signed terminal voltages (broadcast shape of vgs, vds)."""
        shape = np.broadcast(np.asarray(vgs), np.asarray(vds)).shape
        return self.g_x(self.features(is_pmos, vgs, vds)).reshape(shape)

    def g(self, phi, group_tag=None, group_kind=None):   # noqa: D401
        raise TypeError("SymCorrection needs signed voltages: use g_v")

    def apply(self, dev: DeviceRec) -> list[np.ndarray]:
        pm = dev.family == "pmos"
        return [np.asarray(c.sim, float) * np.exp(self.g_v(pm, c.vg, c.vd))
                for c in dev.curves]

    def to_json(self) -> dict:
        return {**super().to_json(), "inputs": self.inputs, "eps": self.eps,
                "vdd": self.vdd, "feature_names": ["is_pmos", "x1", "x2"]}

    @classmethod
    def from_json(cls, row: dict) -> "SymCorrection":
        base = Correction.from_json(row)
        return cls(base.arm, base.mu, base.sd, base.nets, base.table, base.lin_w,
                   inputs=row["inputs"], eps=float(row["eps"]), vdd=float(row["vdd"]))


def g_terminal(corr: SymCorrection, is_pmos: bool, vg_node, vd_node, vs_node) -> np.ndarray:
    """g of the element whose terminals d, g, s sit at the given node voltages."""
    vg_node, vd_node, vs_node = np.broadcast_arrays(
        np.asarray(vg_node, float), np.asarray(vd_node, float), np.asarray(vs_node, float))
    return corr.g_v(is_pmos, vg_node - vs_node, vd_node - vs_node)


def g_partials_v2(corr: SymCorrection, is_pmos: bool, vgs, vds,
                  h: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """(dg/dV(g,s), dg/dV(d,s)) by central differences (g is smooth)."""
    vgs = np.asarray(vgs, dtype=np.float64)
    vds = np.asarray(vds, dtype=np.float64)
    dgg = (corr.g_v(is_pmos, vgs + h, vds) - corr.g_v(is_pmos, vgs - h, vds)) / (2 * h)
    dgd = (corr.g_v(is_pmos, vgs, vds + h) - corr.g_v(is_pmos, vgs, vds - h)) / (2 * h)
    return dgg, dgd


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------
def batch_voltages(devices: list[DeviceRec]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Signed (vg, vd) and is_pmos per point, in ``make_batch`` order
    (included curves with den > 0, from the trim start)."""
    vg, vd, pm = [], [], []
    for dev in devices:
        for c in dev.curves:
            if not c.included or not (c.den > 0):
                continue
            s = slice(c.start, None)
            vg.append(np.asarray(c.vg, float)[s])
            vd.append(np.asarray(c.vd, float)[s])
            pm.append(np.full(len(vg[-1]), dev.family == "pmos"))
    return np.concatenate(vg), np.concatenate(vd), np.concatenate(pm)


def batch_inputs(devices: list[DeviceRec], inputs: str = "sym_v2",
                 eps: float = EPS, vdd: float = VDD_NORM) -> np.ndarray:
    vg, vd, pm = batch_voltages(devices)
    out = np.empty((len(vg), 3))
    for flag in (False, True):
        k = pm == flag
        if k.any():
            out[k] = INPUTS[inputs](flag, vg[k], vd[k], eps=eps, vdd=vdd)
    return out


# ---------------------------------------------------------------------------
# Penalty grid (card-only forward quadrant, per device)
# ---------------------------------------------------------------------------
PEN_GRID = np.round(np.arange(PEN_N) * PEN_STEP, 6)     # 0 .. 1.85


def penalty_arrays(i_grid: np.ndarray, is_pmos: bool, step: float = PEN_STEP,
                   floor: float = PEN_I_FLOOR, grid: np.ndarray | None = None) -> dict:
    """Penalty inputs from a card current grid ``i_grid`` (signed drain
    current, shape (len(PEN_GRID) u_g, len(PEN_GRID) u_d), forward quadrant).

    Returns flattened arrays over the kept points (u_d > 0, forward current
    > ``floor``, finite): ug, ud (V, forward-positive), r_gm = gm/I and
    r_gds = gds/I of the card (1/V, central differences, one-sided at the
    grid edges).  ``grid`` (default ``PEN_GRID``): the axis values of both
    u_g and u_d, spaced by ``step`` (I17-2: 0 .. VDD of another PDK)."""
    i_grid = np.asarray(i_grid, dtype=np.float64)
    axis = PEN_GRID if grid is None else np.asarray(grid, dtype=np.float64)
    s = -1.0 if is_pmos else 1.0
    i_f = s * i_grid
    dg = np.gradient(i_f, step, axis=0)
    dd = np.gradient(i_f, step, axis=1)
    UG, UD = np.meshgrid(axis[:i_grid.shape[0]], axis[:i_grid.shape[1]], indexing="ij")
    keep = (UD > 0) & np.isfinite(i_f) & (i_f > floor) & np.isfinite(dg) & np.isfinite(dd)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_gm = dg / i_f
        r_gds = dd / i_f
    return {"ug": UG[keep], "ud": UD[keep], "r_gm": r_gm[keep], "r_gds": r_gds[keep],
            "is_pmos": bool(is_pmos), "n_grid": int(i_grid.size), "n_kept": int(keep.sum())}


def penalty_value_numpy(corr: SymCorrection | Correction, pens: Sequence[dict],
                        h: float = 1e-5, vdd_v1: float = VDD_NORM) -> dict:
    """P (mean over devices of the per-device mean hinge) and the fraction
    of grid points with a negative corrected gm / gds, for a trained
    correction (v2: ``SymCorrection``; v1: ``BiasCorrection`` on E1 columns
    0-2, abs inputs).  dg/du by central differences."""
    from .i3_2_bias import BiasCorrection

    def gfun(pm, ug, ud):
        s = -1.0 if pm else 1.0
        if isinstance(corr, SymCorrection):
            return corr.g_v(pm, s * ug, s * ud)
        if isinstance(corr, BiasCorrection):
            phi = np.zeros((len(ug), 6))
            phi[:, 0] = 1.0 if pm else 0.0
            phi[:, 1] = np.abs(ug) / vdd_v1
            phi[:, 2] = np.abs(ud) / vdd_v1
            return corr.g(phi)
        raise TypeError(type(corr))
    per, neg_gm, neg_gds, n = [], 0, 0, 0
    for p in pens:
        pm, ug, ud = p["is_pmos"], p["ug"], p["ud"]
        dgg = (gfun(pm, ug + h, ud) - gfun(pm, ug - h, ud)) / (2 * h)
        dgd = (gfun(pm, ug, ud + h) - gfun(pm, ug, ud - h)) / (2 * h)
        a = p["r_gm"] + dgg
        b = p["r_gds"] + dgd
        per.append(float(np.mean(np.maximum(-a, 0) + np.maximum(-b, 0))) if len(a) else 0.0)
        neg_gm += int(np.sum(a < 0))
        neg_gds += int(np.sum(b < 0))
        n += len(a)
    return {"P": float(np.mean(per)) if per else 0.0, "per_device": per,
            "frac_neg_gm": neg_gm / max(n, 1), "frac_neg_gds": neg_gds / max(n, 1),
            "n_points": n}


def card_negative_fraction(pens: Sequence[dict]) -> dict:
    """Fraction of penalty points where the card alone has gm / gds < 0."""
    n = sum(len(p["ug"]) for p in pens)
    return {"frac_neg_gm_card": sum(int(np.sum(p["r_gm"] < 0)) for p in pens) / max(n, 1),
            "frac_neg_gds_card": sum(int(np.sum(p["r_gds"] < 0)) for p in pens) / max(n, 1),
            "n_points": n}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def fit_v2(train: list[DeviceRec], pens: Sequence[dict] | None, lam: float,
           hyper: dict | None = None, inputs: str = "sym_v2", eps: float = EPS,
           threads: int = 1, arm: str = "M_bias_v2", trace: list | None = None,
           vdd: float = VDD_NORM) -> SymCorrection:
    """Train the v2 ensemble on ``train`` (cards fixed: the records' stored
    simulations) with the gm / gds penalty on ``pens`` (one dict per training
    device, ``penalty_arrays``) and weight ``lam``.

    The objective per member is  E1 loss + lam * P + l2 * sum ||W||^2
    (P omitted when lam == 0 or pens is empty; then the update sequence is
    the one of ``i3_2_bias.fit_bias``).  ``vdd``: input normalization
    (default the sky130 ``VDD_NORM``; I17-2 passes the IHP class supply)."""
    import torch
    torch.set_num_threads(threads)
    hp = {**HYPER_V2, **(hyper or {})}
    batch = make_batch(train)
    loss = _loss_fn(batch)
    x = batch_inputs(train, inputs, eps, vdd)
    if len(x) != len(batch.sim):
        raise RuntimeError("batch_inputs and make_batch differ in length")
    # same memory layout as fit_bias (E1 phi column selection by a list index),
    # so mean/std round identically and the abs_v1 path reproduces fit_bias
    x6 = np.zeros((len(x), 6))
    x6[:, :3] = x
    x = x6[:, [0, 1, 2]]
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    mu[0], sd[0] = 0.0, 1.0
    sd = np.where(sd > 1e-12, sd, 1.0)
    z = torch.tensor((x - mu) / sd, dtype=torch.float64)
    use_pen = bool(lam) and bool(pens)
    if use_pen and inputs != "sym_v2":
        raise ValueError("penalty is defined for the sym_v2 inputs")
    pen_t = []
    if use_pen:
        for p in pens:
            if not len(p["ug"]):
                continue
            ug = torch.tensor(p["ug"], dtype=torch.float64, requires_grad=True)
            ud = torch.tensor(p["ud"], dtype=torch.float64, requires_grad=True)
            pen_t.append((float(p["is_pmos"]), ug, ud,
                          torch.tensor(p["r_gm"], dtype=torch.float64),
                          torch.tensor(p["r_gds"], dtype=torch.float64)))
    mu_t = torch.tensor(mu, dtype=torch.float64)
    sd_t = torch.tensor(sd, dtype=torch.float64)
    nets = []
    for seed in hp["seeds"]:
        torch.manual_seed(seed)
        dims = [z.shape[1]] + [hp["hidden"]] * hp["layers"] + [1]
        mods = [torch.nn.Linear(a, b).double() for a, b in zip(dims[:-1], dims[1:])]
        torch.nn.init.zeros_(mods[-1].weight)
        torch.nn.init.zeros_(mods[-1].bias)
        params = [q for mm in mods for q in mm.parameters()]
        opt = torch.optim.Adam(params, lr=hp["lr"])

        def forward(zz):
            for i, mm in enumerate(mods):
                zz = mm(zz)
                if i < len(mods) - 1:
                    zz = torch.tanh(zz)
            return OUT_SCALE * torch.tanh(zz[:, 0])

        def penalty():
            per = []
            for pm, ug, ud, rg, rd in pen_t:
                x1, x2 = torch_sym_forward(ug, ud, eps, vdd)
                xin = torch.stack([torch.full_like(x1, pm), x1, x2], dim=1)
                gp = forward((xin - mu_t) / sd_t)
                dgg, dgd = torch.autograd.grad(gp.sum(), (ug, ud), create_graph=True)
                per.append((torch.relu(-(rg + dgg)) + torch.relu(-(rd + dgd))).mean())
            return torch.stack(per).mean()

        for step in range(hp["steps"]):
            opt.zero_grad()
            value = loss(forward(z)) + hp["l2"] * sum((mm.weight ** 2).sum() for mm in mods)
            pval = None
            if use_pen and pen_t:
                pval = penalty()
                value = value + lam * pval
            value.backward()
            opt.step()
            if trace is not None and (step + 1) % 250 == 0:
                trace.append((int(seed), step + 1, float(value.detach()),
                              None if pval is None else float(pval.detach())))
        nets.append([(mm.weight.detach().numpy().T.copy(), mm.bias.detach().numpy().copy())
                     for mm in mods])
    return SymCorrection(arm, mu, sd, nets=nets, inputs=inputs, eps=eps, vdd=vdd)


def merge_seed_corrections(parts: list[SymCorrection]) -> SymCorrection:
    """Merge single-seed corrections trained on the same data (same mu/sd)."""
    first = parts[0]
    nets = []
    for p in parts:
        if not (np.array_equal(p.mu, first.mu) and np.array_equal(p.sd, first.sd)
                and p.inputs == first.inputs and p.eps == first.eps):
            raise RuntimeError("per-seed corrections do not share feature scaling")
        nets += p.nets
    return SymCorrection(first.arm, first.mu, first.sd, nets=nets, inputs=first.inputs,
                         eps=first.eps, vdd=first.vdd)


# ---------------------------------------------------------------------------
# Verilog-A (v2)
# ---------------------------------------------------------------------------
SENSE = "    V(sense) <+ RS * I(sense);"


def to_verilog_a_v2(corr: SymCorrection, module: str = VA_MODULE_V2) -> str:
    """Verilog-A text of a ``SymCorrection`` (inputs sym_v2).

    Ports (d, g, s, rx, ri).  Netlist: the transistor's card instance is
    unchanged at (d, g, s, b); a charge-free replica of the card sits at
    (ri, rg, rs, rb) with rx, rg, rs, rb driven by unit-gain VCVS copies of
    V(d), V(g), V(s), V(b); the zero-volt branch rx -> ri senses the replica
    drain current I_rep (static).  The module adds (exp(g) - 1) * I_rep from
    d to s.  g uses V(g,s), V(g,d), V(d,s) of the transistor terminals only
    through the d/s-symmetric inputs; the polarity parameter is is_pmos
    (0/1).  EPS and VDD are part of the trained correction (literals)."""
    if corr.inputs != "sym_v2":
        raise ValueError("to_verilog_a_v2 needs sym_v2 inputs")
    fmt = lambda v: f"{float(v):.17e}"
    hidden = max(w.shape[1] for layers in corr.nets for w, _ in layers[:-1])
    lines = [
        "`include \"disciplines.vams\"",
        "`include \"constants.vams\"",
        f"// I12-1 bias-only correction g v2 ({corr.arm}), {len(corr.nets)}-member "
        f"3-{hidden}-...-1 tanh MLP ensemble.",
        "// I(d->s) added = (exp(g) - 1) * I(sense); I(sense) = drain current of a",
        "// charge-free replica of the card (static), so the element has no charge.",
        "// Inputs (d/s symmetric): sg = 1 - 2 is_pmos, a = sg V(g,s), b = sg V(g,d),",
        f"// m = sqrt(V(d,s)^2 + EPS^2), x1 = (a + b + m) / (2 VDD), x2 = m / VDD,",
        f"// EPS = {corr.eps:g} V, VDD = {corr.vdd:g} V.",
        f"module {module}(d, g, s, rx, ri);",
        "  inout d, g, s, rx, ri;",
        "  electrical d, g, s, rx, ri;",
        "  branch (rx, ri) sense;   // I(sense) = replica drain current",
        "  parameter real is_pmos = 0;  // 0 nMOS, 1 pMOS",
        "  parameter real RS = 0.0;   // ohm; sense branch V = RS * I (0: zero-volt source)",
        "  real ib, gsum, acc, sg, vgs, vgd, vds, mm, x0, x1, x2;",
        "  real " + ", ".join(f"h{i}" for i in range(hidden)) + ";",
        "  real " + ", ".join(f"k{i}" for i in range(hidden)) + ";",
        "  analog begin",
        SENSE,
        "    ib = I(sense);",
        "    sg = 1.0 - 2.0 * is_pmos;",
        "    vgs = V(g, s);",
        "    vgd = V(g, d);",
        "    vds = V(d, s);",
        f"    mm = sqrt(vds * vds + {fmt(corr.eps * corr.eps)});",
        f"    x0 = (is_pmos - {fmt(corr.mu[0])}) / {fmt(corr.sd[0])};",
        f"    x1 = ((sg * vgs + sg * vgd + mm) / {fmt(2.0 * corr.vdd)} - {fmt(corr.mu[1])}) / {fmt(corr.sd[1])};",
        f"    x2 = (mm / {fmt(corr.vdd)} - {fmt(corr.mu[2])}) / {fmt(corr.sd[2])};",
        "    gsum = 0.0;",
    ]
    xs = ["x0", "x1", "x2"]
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
    ]
    return "\n".join(lines) + "\n"


def eval_verilog_a_v2(text: str, is_pmos: float, vgs: float, vds: float) -> float:
    """g from the assignment lines of ``to_verilog_a_v2`` text at terminal
    voltages V(g,s) = vgs, V(d,s) = vds (V(g,d) = vgs - vds)."""
    body = text.split("ib = I(sense);", 1)[1].split("I(d, s) <+", 1)[0]
    env = {"tanh": math.tanh, "sqrt": math.sqrt, "exp": math.exp, "is_pmos": float(is_pmos),
           "VGS": float(vgs), "VGD": float(vgs) - float(vds), "VDS": float(vds)}
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        name, expr = line.rstrip(";").split(" = ", 1)
        expr = (expr.replace("V(g, s)", "VGS").replace("V(g, d)", "VGD")
                .replace("V(d, s)", "VDS"))
        env[name.strip()] = eval(expr, {"__builtins__": {}}, env)  # noqa: S307
    n = int(text.split("I(d, s) <+ (exp(gsum / ", 1)[1].split(")", 1)[0])
    return env["gsum"] / n


# ---------------------------------------------------------------------------
# Replica library and netlist (pure text)
# ---------------------------------------------------------------------------
def replica_corner(text: str, dev_type: str) -> str:
    """Charge-free replica of the corner text's subcircuit: the ``.subckt
    <name> ... .ends <name>`` block (with any bin patches already in
    ``text``), renamed ``<name>_rep`` with model ``<model>_rep``, and
    ``REPLICA_ZERO`` written into every bin."""
    from .. import spice_pdk as sp
    sub, mod = sp._SUBCKT[dev_type], sp._MODEL[dev_type]
    head, tail = f".subckt {sub} ", f".ends {sub}"
    if text.count(head) != 1 or text.count(tail) != 1:
        raise ValueError("corner text does not hold exactly one subcircuit block")
    a = text.index(head)
    b = text.index(tail) + len(tail)
    blk = text[a:b]
    bins = [idx for idx, _, _ in sp._bin_spans(blk, dev_type)]
    if not bins:
        raise ValueError("no model bins in the subcircuit block")
    for idx in bins:
        blk = sp._patch_params_in_bin(blk, dev_type, idx, REPLICA_ZERO)
    blk = blk.replace(head, f".subckt {sub}{REPLICA_SUFFIX} ", 1)
    blk = blk.replace(tail, f".ends {sub}{REPLICA_SUFFIX}", 1)
    blk = re.sub(rf"\b{re.escape(mod)}\b", f"{mod}{REPLICA_SUFFIX}", blk)
    return "* charge-free replica (I12-1): xpart=-1, zero overlap/fringe/junction caps\n" + blk + "\n"


def replica_bins_ok(rep_text: str, dev_type: str) -> bool:
    """Every bin of a replica block carries every REPLICA_ZERO value."""
    from .. import spice_pdk as sp
    mod = sp._MODEL[dev_type] + REPLICA_SUFFIX
    hdr = re.compile(rf"^\.model\s+{re.escape(mod)}\.(\d+)\s", re.I | re.M)
    hits = list(hdr.finditer(rep_text))
    if not hits:
        return False
    for i, h in enumerate(hits):
        span = rep_text[h.start(): hits[i + 1].start() if i + 1 < len(hits) else len(rep_text)]
        for k, v in REPLICA_ZERO.items():
            mm = re.findall(rf"^\+\s*{k}\s*=\s*([^\s]+)", span, re.I | re.M)
            if len(mm) != 1 or float(mm[0]) != float(v):
                return False
    return True


def corr_models_v2() -> str:
    return "".join(f".model {m} {VA_MODULE_V2} is_pmos={p} RS=0\n"
                   for m, p in (("corr2n", 0), ("corr2p", 1)))


def instance_v2(name: str, inst_line_body: str, subckt: str, dev_type: str,
                d: str, g: str, s: str, b: str, corr: bool) -> str:
    """Corrected transistor ``name`` at (d, g, s, b).  ``inst_line_body`` is
    the part of the ``spice_pdk._instance_line`` after the four nodes
    (``<subckt> l=.. w=.. ...``).  With ``corr``: the replica, four VCVS and
    the OSDI module are added; the card instance itself is unchanged."""
    if not inst_line_body.startswith(subckt + " "):
        raise ValueError("instance body does not start with the subcircuit name")
    out = f"X{name} {d} {g} {s} {b} {inst_line_body}"
    if not out.endswith("\n"):
        out += "\n"
    if not corr:
        return out
    p = f"{name}_"
    rep_body = subckt + REPLICA_SUFFIX + inst_line_body[len(subckt):]
    out += f"X{name}r {p}ri {p}rg {p}rs {p}rb {rep_body}"
    if not out.endswith("\n"):
        out += "\n"
    out += (f"E{name}d {p}rx 0 {d} 0 1\nE{name}g {p}rg 0 {g} 0 1\n"
            f"E{name}s {p}rs 0 {s} 0 1\nE{name}b {p}rb 0 {b} 0 1\n")
    model = "corr2p" if dev_type == "pmos" else "corr2n"
    out += f"N{name} {d} {g} {s} {p}rx {p}ri {model}\n"
    return out


# ---------------------------------------------------------------------------
# Python reference model of the element
# ---------------------------------------------------------------------------
def python_current_fwd(i_card, g) -> np.ndarray:
    """Forward bench (module d = card drain): I_card x exp(g)."""
    return np.asarray(i_card, float) * np.exp(np.asarray(g, float))


def python_current_swap(i_card_nd, i_vz, g) -> np.ndarray:
    """Exchanged-terminal bench: current into node nd (card source, module s)
    = I_card(nd) + (exp(g) - 1) * i(VZ), with i(VZ) the card-only current
    through the zero-volt source at the card drain (= -I into the drain)."""
    return np.asarray(i_card_nd, float) + (np.exp(np.asarray(g, float)) - 1.0) * np.asarray(i_vz, float)


def python_admittance_v2(i_card, y_card, g, dg_dv) -> np.ndarray:
    """Drain admittance of the card + element for one excitation:
    real part exp(g) Re Y_card + I_card exp(g) dg/dV (the replica's static
    conductance equals Re Y_card), imaginary part Im Y_card (no charge)."""
    eg = np.exp(np.asarray(g, float))
    y = np.asarray(y_card)
    re_ = eg * y.real + np.asarray(i_card, float) * eg * np.asarray(dg_dv, float)
    return re_ + 1j * y.imag


def eq_violation(i_elem, i_py, rel: float = EQ_REL, absol: float = EQ_ABS) -> dict:
    """Per-point criterion |dI| <= rel |I_py| + abs.  Non-finite element or
    reference values count as violations.  Returns the number of points,
    violations, and the max of |dI| / (rel |I_py| + abs)."""
    a = np.asarray(i_elem, float).ravel()
    b = np.asarray(i_py, float).ravel()
    bad = ~(np.isfinite(a) & np.isfinite(b))
    ok = ~bad
    ratio = np.zeros(a.shape)
    ratio[ok] = np.abs(a[ok] - b[ok]) / (rel * np.abs(b[ok]) + absol)
    ratio[bad] = np.inf
    return {"n": int(a.size), "n_viol": int(np.sum(ratio > 1.0)),
            "max_ratio": float(np.max(ratio)) if a.size else 0.0,
            "max_abs_diff_A": float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else float("nan")}


def cap_err(im_elem, im_card, floor: float = 1e-15) -> np.ndarray:
    """|Im Y_elem - Im Y_card| / max(|Im Y_card|, floor)."""
    a = np.asarray(im_elem, float)
    b = np.asarray(im_card, float)
    return np.abs(a - b) / np.maximum(np.abs(b), floor)


# ---------------------------------------------------------------------------
# Validity row selections (pre-registered wording; unit-tested)
# ---------------------------------------------------------------------------
V1_FOLDS = ("nmos_L0p15_W1p6", "pmos_L2_W5")
V2_DEVICES = ("nmos_L1_W1p6", "pmos_L8_W5")


def select_v1_keys(stored_keys: Sequence[str]) -> list[str]:
    """V1: the stored all18 fold corrections 'M_bias|<tag>' for the two
    V1_FOLDS tags, exactly; raises if one is missing."""
    want = [f"M_bias|{t}" for t in V1_FOLDS]
    missing = [k for k in want if k not in stored_keys]
    if missing:
        raise KeyError(f"stored fold corrections missing: {missing}")
    return want


def select_v2_rows(stored_rows: dict) -> dict:
    """V2: stored all18 M_bias (v1) per-device rows for V2_DEVICES, exactly."""
    missing = [t for t in V2_DEVICES if t not in stored_rows]
    if missing:
        raise KeyError(f"stored rows missing: {missing}")
    return {t: float(stored_rows[t]) for t in V2_DEVICES}


def select_primary_rows(rows: dict, all_tags: Sequence[str]) -> dict:
    """Primary / NI contrasts: one row per device for all 18 tags (no more,
    no fewer); a missing device raises."""
    tags = sorted(all_tags)
    if len(tags) != 18:
        raise ValueError(f"expected 18 devices, got {len(tags)}")
    missing = [t for t in tags if t not in rows]
    extra = [t for t in rows if t not in tags]
    if missing or extra:
        raise KeyError(f"rows differ from the 18 devices: missing {missing}, extra {extra}")
    return {t: float(rows[t]) for t in tags}


# ---------------------------------------------------------------------------
# Decision (pure, pre-registered)
# ---------------------------------------------------------------------------
NI_MARGIN = 1.05
THR = {"q1_default_rel": 1e-3, "q1_vds0_abs": 1e-12, "q2_ac": 1e-3, "cap": 1e-3,
       "swap_asym": 1e-3, "g_text": 1e-9, "g_sym_text": 1e-12, "runtime_ratio": 2.0,
       "ro_min_periods": 5, "v1_g": 1e-9, "v2_rrms": 1e-9}


def decide_validity(v: dict) -> dict:
    """V1 trainer reproduction (max |g diff| <= 1e-9 on both folds), V2
    held-out refit reproduction (|RRMS diff| <= 1e-9 on both devices), V3
    18 M_bias_v2 rows present."""
    fails = []
    if not (v["v1_max_g_diff"] <= THR["v1_g"]):
        fails.append("V1 trainer reproduction")
    if not (v["v2_max_rrms_diff"] <= THR["v2_rrms"]):
        fails.append("V2 held-out refit reproduction")
    if v["v3_n_rows"] != 18:
        fails.append("V3 18 M_bias_v2 rows")
    return {"valid": not fails, "fails": fails}


def decide_accuracy(primary: dict, ni: dict, n_failed: int) -> dict:
    """A1: GMR(M_bias_v2, C0_refit) CI upper bound < 1 and 0 failed devices.
    A2: GMR(M_bias_v2, M_bias_v1) CI upper bound <= NI_MARGIN (non-inferior)."""
    a1 = bool(n_failed == 0 and np.isfinite(primary["ci_high"]) and primary["ci_high"] < 1.0)
    a2 = bool(np.isfinite(ni["ci_high"]) and ni["ci_high"] <= NI_MARGIN)
    return {"A1_gain_vs_C0_refit": a1, "A2_noninferior_vs_v1": a2}


def decide_deploy(s: dict) -> dict:
    """Core checks and restrictions (B).  Keys of ``s``:

    q1_tight_viol (points violating |dI| <= 1e-6 |I_py| + 1e-13 A, tight
    tolerance, forward and exchanged-terminal grids, 18 devices); q1_bad
    (convergence messages + non-finite values + failed runs, element runs);
    q1_vds0_max (A); g_text_max (|g text - g python|), g_sym_text_max
    (|g text(exchanged) - g text(forward)|); swap_asym_max (element-introduced
    |d ln(I_swap / I_fwd)|, forward operation, |I| > 1 nA); q2_ac_max
    (normalized gm / gds difference, tight); cap_max (max over Im Y_dd, Im Y_dg,
    Im Y_gg of |Im Y_elem - Im Y_card| / max(|Im Y_card|, 1e-15 S)); q2_gm_sign
    (gm curves with an introduced sign change in strong inversion);
    q3_fail (circuits failing with the element while the same card without
    it passes); restrictions: q1_default_max (relative, |I| > 1 nA),
    q2_gds_sign (gds curves with an introduced sign change), runtime_ratio_max.
    """
    core = []
    if s["q1_tight_viol"] > 0:
        core.append("Q1 equivalence |dI| <= 1e-6 |I| + 1e-13 A (tight)")
    if s["q1_bad"] > 0:
        core.append("Q1 convergence / non-finite")
    if not (s["q1_vds0_max"] <= THR["q1_vds0_abs"]):
        core.append("Q1 I_D at Vds = 0")
    if not (s["g_text_max"] <= THR["g_text"]):
        core.append("Verilog-A g vs Python")
    if not (s["g_sym_text_max"] <= THR["g_sym_text"]):
        core.append("Verilog-A g not d/s symmetric")
    if not (s["swap_asym_max"] <= THR["swap_asym"]):
        core.append("terminal-exchange asymmetry > 1e-3")
    if not (s["q2_ac_max"] <= THR["q2_ac"]):
        core.append("Q2 derivative agreement")
    if not (s["cap_max"] <= THR["cap"]):
        core.append("Q2 capacitances differ from the card")
    if s["q2_gm_sign"] > 0:
        core.append("Q2 gm sign change in strong inversion")
    if s["q3_fail"] > 0:
        core.append("Q3 circuit failure with the element")
    restr = []
    if not (s["q1_default_max"] <= THR["q1_default_rel"]):
        restr.append("default tolerance above 1e-3")
    if s["q2_gds_sign"] > 0:
        restr.append("gds sign changes introduced")
    if s["runtime_ratio_max"] > THR["runtime_ratio"]:
        restr.append("runtime overhead > 2x")
    if core:
        label = "not deployable"
    elif restr:
        label = "deployable with restrictions"
    else:
        label = "deployable (simulator-level)"
    return {"deploy": label, "core_fail": core, "restrictions": restr}


def decide_outcome(validity: dict, acc: dict, dep: dict) -> dict:
    """Overall label and action (prereg.json 'decision.outcome')."""
    if not validity["valid"]:
        return {"outcome": "invalid", "action": "stop: fix and re-register"}
    deployable = dep["deploy"] != "not deployable"
    if not acc["A1_gain_vs_C0_refit"]:
        return {"outcome": "no gain over C0_refit", "action": "stop"}
    if deployable and acc["A2_noninferior_vs_v1"]:
        return {"outcome": "supported", "action": "continue: v2 replaces v1 as the deployable add-on"}
    if deployable:
        return {"outcome": "deployable, accuracy below v1 (not non-inferior)",
                "action": "stop v2; report the accuracy cost of the deployment constraints"}
    return {"outcome": "not deployable", "action": "stop; diagnose the failed core checks"}


# ---------------------------------------------------------------------------
# Decks (text) and library (I/O)
# ---------------------------------------------------------------------------
def inst_body(dev_type: str, L_um: float, W_um: float) -> str:
    """``spice_pdk._instance_line`` without 'Xm1 nd ng 0 0 '."""
    from .. import spice_pdk as sp
    line = sp._instance_line(dev_type, L_um, W_um)
    head = "Xm1 nd ng 0 0 "
    if not line.startswith(head):
        raise ValueError("unexpected instance line")
    return line[len(head):]


def instance_dev(name: str, dev, d: str, g: str, s: str, b: str, corr: bool) -> str:
    """``instance_v2`` for an ``i11_1_deploy.Dev``."""
    from .. import spice_pdk as sp
    return instance_v2(name, inst_body(dev.dev_type, dev.L_um, dev.W_um),
                       sp._SUBCKT[dev.dev_type], dev.dev_type, d, g, s, b, corr)


def patched_library_v2(td: Path, cards: dict, bsim4_state: str = "clamped") -> Path:
    """``i11_1_deploy.patched_library`` plus, for both polarities, the
    charge-free replica of the (patched or base) corner text, included in
    the tt_77k section.  Raises if a replica bin lacks a REPLICA_ZERO value.
    ``bsim4_state`` as ``patched_library`` (default "clamped", I14-1: the
    card and its replica carry the b4check.c clamp; "legacy" reproduces the
    I12-1 decks)."""
    from .. import spice_pdk as sp
    from .i11_1_deploy import patched_library
    td = Path(td)
    lib = patched_library(td, cards, bsim4_state)
    text = lib.read_text()
    adds = []
    for dev_type in ("nmos", "pmos"):
        base = sp.PDK77K_DIR / sp._CORNER_BASENAME[dev_type]
        patched = td / sp._CORNER_BASENAME[dev_type]
        src = patched if patched.exists() and patched.as_posix() in text else base
        rep = replica_corner(src.read_text(errors="replace"), dev_type)
        if not replica_bins_ok(rep, dev_type):
            raise RuntimeError(f"replica bins of {dev_type} lack a REPLICA_ZERO value")
        path = td / f"replica_{dev_type}.spice"
        path.write_text(rep)
        adds.append(f'.include "{path.as_posix()}"\n')
    if text.count(".endl") != 1:
        raise ValueError("library text does not have exactly one .endl")
    lib.write_text(text.replace(".endl", "".join(adds) + ".endl"))
    return lib


def deck_v2(title: str, lib: Path, body: str, control: Sequence[str], osdi: Path | None,
            options: str = "") -> str:
    """``i11_1_deploy.deck`` with the v2 correction models."""
    from .i11_1_deploy import TEMP_K
    lines = [f"* {title}\n", ".options scale=1.0\n", f".options temp={TEMP_K - 273.15:.4f}\n"]
    if options:
        lines.append(f".options {options}\n")
    lines.append(f'.lib "{Path(lib).as_posix()}" tt_77k\n')
    lines.append(body if body.endswith("\n") else body + "\n")
    if osdi is not None:
        lines.append(corr_models_v2())
    lines.append(".control\n")
    if osdi is not None:
        lines.append(f"pre_osdi {Path(osdi).as_posix()}\n")
    lines += ["set filetype=ascii\n", "set numdgt=12\n"]
    lines += [c if c.endswith("\n") else c + "\n" for c in control]
    lines += ["quit\n", ".endc\n", ".end\n"]
    return "".join(lines)


def device_body_v2(dev, corr: bool, swap: bool = False) -> str:
    """Forward bench: transistor at (nd, ng, 0, 0).  Exchanged bench: card
    drain at nz (VZ 0 V to ground), card source at nd, body at 0."""
    src = "VG ng 0 DC 0 AC 0\nVD nd 0 DC 0 AC 0\n"
    if swap:
        return instance_dev("m1", dev, "nz", "ng", "nd", "0", corr) + "VZ nz 0 DC 0\n" + src
    return instance_dev("m1", dev, "nd", "ng", "0", "0", corr) + src


def swap_grid_control(dev, out: Path) -> list[str]:
    """Nested sweep as ``i11_1_deploy.grid_control``; writes -i(VD), i(VZ)."""
    from .i11_1_deploy import GRID_VDS, GRID_VGS, STEP
    sg = -1.0 if dev.dev_type == "pmos" else 1.0
    return [f"dc VD {GRID_VDS[0]:.6g} {GRID_VDS[-1]:.6g} {STEP:.6g} "
            f"VG 0 {sg * GRID_VGS[-1]:.6g} {sg * STEP:.6g}",
            f"wrdata {Path(out).as_posix()} -i(VD) i(VZ)"]


def inverter_body_v2(n, p, corr: bool, vin: str, vout: str, name: str) -> str:
    return (instance_dev(f"{name}n", n, vout, vin, "0", "0", corr)
            + instance_dev(f"{name}p", p, vout, vin, "vdd", "vdd", corr))


def vtc_deck_body_v2(n, p, corr: bool) -> str:
    from .i11_1_deploy import VDD_CIRCUIT
    return inverter_body_v2(n, p, corr, "nin", "nout", "i1") + f"VDD vdd 0 DC {VDD_CIRCUIT}\nVIN nin 0 DC 0\n"


def ring_body_v2(n, p, corr: bool, stages: int = 5) -> str:
    from .i11_1_deploy import VDD_CIRCUIT
    if stages % 2 == 0:
        raise ValueError("ring oscillator needs an odd number of stages")
    nodes = [f"o{k + 1}" for k in range(stages)]
    body = "".join(inverter_body_v2(n, p, corr, nodes[k - 1], nodes[k], f"s{k + 1}")
                   for k in range(stages))
    ics = " ".join(f"v({nd})={VDD_CIRCUIT if k % 2 == 0 else 0.0}" for k, nd in enumerate(nodes))
    return body + f"VDD vdd 0 DC {VDD_CIRCUIT}\n.ic {ics}\n"


def mirror_body_v2(dev, corr: bool, iref: float) -> str:
    from .i11_1_deploy import VDD_CIRCUIT
    if dev.dev_type == "nmos":
        return (instance_dev("m1", dev, "nref", "nref", "0", "0", corr)
                + instance_dev("m2", dev, "nout", "nref", "0", "0", corr)
                + f"IREF 0 nref DC {iref:.6g}\nVOUT nout 0 DC 0\n")
    return (instance_dev("m1", dev, "nref", "nref", "vdd", "vdd", corr)
            + instance_dev("m2", dev, "nout", "nref", "vdd", "vdd", corr)
            + f"VDD vdd 0 DC {VDD_CIRCUIT}\nIREF nref 0 DC {iref:.6g}\nVOUT nout 0 DC 0\n")


def insert_correction_v2(deck: str, osdi: Path | str, dev) -> str:
    """Measured-curve batch deck (``spice_pdk._batch_deck``): the instance
    'Xm1 nd ng 0 0 ...' gets the replica, VCVS and module; models and
    ``pre_osdi`` added."""
    lines = deck.splitlines(keepends=True)
    inst = [i for i, ln in enumerate(lines) if ln.startswith("Xm1 nd ng 0 0 ")]
    ctrl = [i for i, ln in enumerate(lines) if ln.strip() == ".control"]
    if len(inst) != 1 or len(ctrl) != 1:
        raise ValueError("deck does not have exactly one Xm1 instance and one .control")
    lines[inst[0]] = instance_dev("m1", dev, "nd", "ng", "0", "0", True) + corr_models_v2()
    out = "".join(lines)
    return out.replace(".control\n", f".control\npre_osdi {Path(osdi).as_posix()}\n", 1)


# ---------------------------------------------------------------------------
# Q3 / Q4 row selections (pure, unit-tested)
# ---------------------------------------------------------------------------
def q3_fail_count(rows: dict, circuits: Sequence[str]) -> int:
    """Circuits whose M2_full run fails while the M2_card run passes."""
    return int(sum(1 for n in circuits if not rows[f"{n}|M2_full"]["ok"] and rows[f"{n}|M2_card"]["ok"]))


def circuit_runtime_ratios(rows: dict, circuits: Sequence[str]) -> dict:
    """Amendment A1: median wall-time ratio M2_full / M2_card over circuits
    where both runs pass."""
    return {n: float(np.median(rows[f"{n}|M2_full"]["wall"]) / np.median(rows[f"{n}|M2_card"]["wall"]))
            for n in circuits if rows[f"{n}|M2_full"]["ok"] and rows[f"{n}|M2_card"]["ok"]}


def fold_train_tags(load_order: Sequence[str], held: str | None) -> list[str]:
    """Training records of a fold in the record load order (REAL_DEV records,
    then REAL_TEST, as the I3-2 all-18 run; amendment A2): every tag except
    ``held`` (None: all).  The order fixes the floating-point summation order
    of the loss, so a sorted order does not reproduce the stored corrections."""
    if held is not None and held not in load_order:
        raise KeyError(held)
    if len(set(load_order)) != len(load_order):
        raise ValueError("duplicate tags in the load order")
    return [t for t in load_order if t != held]
