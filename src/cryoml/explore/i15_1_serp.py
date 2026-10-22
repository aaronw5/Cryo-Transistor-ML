"""I15-1: re-registration of the I14-1 Part B deployment test of g v2 with the
Q1 grid swept in serpentine order.

Pre-registration: ``out/research/explore/I15_1/prereg.md`` / ``prereg.json``.

I14-1 failed one core check ('Q1 runs': two gmin-stepping warning lines in 1
of 126 element decks, nmos_L8_W1p6, exchanged bench, tight tolerance).  The
I15-1 pre-registration diagnosis located the warnings at the restart of the
inner sweep of the nested DC command (V(d) from +1.85 V back to -1.85 V when
V(g) steps to 1.85 V; the same restart in the forward card + replica deck).
Isolated operating points at that bias converge without convergence aids,
and raising the Newton iteration limits (itl1 / itl2 up to 500) does not
remove the warnings.

Change from I14-1 (test procedure only): the Q1 grid decks (card only, card +
replica and element; forward and exchanged; default, r4 and tight tolerance)
sweep V(d) up on even V(g) lines and down on odd lines (one ``dc`` command per
V(g) line, rows appended to one file), so every grid point is reached from a
neighbouring point (25 mV in V(d), or 25 mV in V(g) at the line ends).  The
grid points, benches, tolerances, replica, module, references and thresholds
are the I14-1 ones.  ``grid_rows`` returns the rows in the nested order of
``i11_1_deploy.grid_from_rows`` (V(d) ascending within each V(g) line).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .i14_1_chan import *  # noqa: F401,F403  (I14-1 replica, netlists, references)


def grid_control_serp(dev, out: Path, swap: bool, sense: bool) -> list[str]:
    """Serpentine Q1 grid sweep: for V(g) line k (0 -> +-1.85 V in 25 mV
    steps) one ``dc VD`` command, ascending for even k and descending for
    odd k; ``wrdata`` appends every line to ``out`` (``set appendwrite``).
    Columns as ``i14_1_chan.grid_control``: -i(VD) and, with ``sense``,
    i(Vm1R)."""
    from .i11_1_deploy import GRID_VDS, GRID_VGS, STEP
    del swap                       # the bench is set by the deck body
    sg = -1.0 if dev.dev_type == "pmos" else 1.0
    vecs = "-i(VD)" + (" i(Vm1R)" if sense else "")
    lines = ["set appendwrite"]
    for k, vg in enumerate(GRID_VGS):
        lo, hi = GRID_VDS[0], GRID_VDS[-1]
        a, b, st = (lo, hi, STEP) if k % 2 == 0 else (hi, lo, -STEP)
        lines += [f"alter VG dc = {sg * vg:.6g}",
                  f"dc VD {a:.6g} {b:.6g} {st:.6g}",
                  f"wrdata {Path(out).as_posix()} {vecs}"]
    return lines


def grid_rows(rows: np.ndarray) -> np.ndarray:
    """Rows of a serpentine grid file (``read_wrdata``) -> the nested order
    (V(g) lines in order, V(d) ascending within each line).  Returns an
    empty array when the row count is not len(GRID_VGS) x len(GRID_VDS) or a
    line does not hold every V(d) grid value once."""
    from .i11_1_deploy import GRID_VDS, GRID_VGS
    nv, nd = len(GRID_VGS), len(GRID_VDS)
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] != nv * nd:
        return np.zeros((0, rows.shape[1] if rows.ndim == 2 else 0))
    out = np.empty_like(rows)
    for k in range(nv):
        blk = rows[k * nd:(k + 1) * nd]
        o = np.argsort(blk[:, 0], kind="stable")
        blk = blk[o]
        if not np.allclose(blk[:, 0], GRID_VDS, rtol=0.0, atol=1e-9):
            return np.zeros((0, rows.shape[1]))
        out[k * nd:(k + 1) * nd] = blk
    return out
