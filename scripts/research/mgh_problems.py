"""The 22 nonlinear least-squares functions of the Moré-Wild benchmark.

Moré and Wild (2009, SIAM J. Optim. 20:172, "Benchmarking derivative-free
optimization algorithms") use 22 vector functions: problems 1-18 from Moré,
Garbow and Hillstrom (1981, ACM TOMS 7:17) and problems 19-22 (Bdqrtic,
Cube, Mancino, Heart8) from CUTEr, each as ``f(x) = sum_i r_i(x)^2`` from the
standard start ``x0`` scaled by ``10^s``.  Residuals here follow the
definitions in that paper's ``dfovec.m`` / ``dfoxs.m``, written from the
published formulas (no file of that benchmark is present in this
environment).

Instances: Moré and Wild's own (n, m, s) table is not available offline, so
``INSTANCES`` is the pre-registered fallback: each function at one standard
(n, m) (``STANDARD_DIMS``; fixed-size functions at their only size,
variable-size functions at one size each, chosen without the Moré-Wild table;
Chebyquad at n = m = 8, a size with a published MGH minimum) with s in
{0, 1}: 44 instances.  Watson's standard start is x0 = 0, so its s = 1
instance equals its s = 0 instance; both are kept.

``residual(pid, x, m)`` evaluates problem ``pid`` (1-22); ``x0(pid, n)`` is
its standard start.  ``KNOWN_MINIMA`` lists published minimum sums of squares
for a few instances (used by ``mgh_validate.py --check`` as a definition
check).
"""

from __future__ import annotations

import math

import numpy as np

NAMES = {
    1: "linear_full_rank", 2: "linear_rank_1", 3: "linear_rank_1_zero_cols_rows",
    4: "rosenbrock", 5: "helical_valley", 6: "powell_singular",
    7: "freudenstein_roth", 8: "bard", 9: "kowalik_osborne", 10: "meyer",
    11: "watson", 12: "box_3d", 13: "jennrich_sampson", 14: "brown_dennis",
    15: "chebyquad", 16: "brown_almost_linear", 17: "osborne_1",
    18: "osborne_2", 19: "bdqrtic", 20: "cube", 21: "mancino", 22: "heart8",
}

# (n, m) per function for the fallback instance list.
STANDARD_DIMS = {
    1: (9, 45), 2: (7, 35), 3: (7, 35), 4: (2, 2), 5: (3, 3), 6: (4, 4),
    7: (2, 2), 8: (3, 15), 9: (4, 11), 10: (3, 16), 11: (6, 31), 12: (3, 10),
    13: (2, 10), 14: (4, 20), 15: (8, 8), 16: (10, 10), 17: (5, 33),
    18: (11, 65), 19: (8, 8), 20: (5, 5), 21: (5, 5), 22: (8, 8),
}

INSTANCE_NOTE = (
    "Moré-Wild 2009 (n, m, s) table not available offline; fallback: each of "
    "the 22 functions at one standard (n, m) (STANDARD_DIMS) with s in {0, 1} "
    "(44 instances); start 10^s x0")

INSTANCES = [(pid, n, m, s) for pid, (n, m) in STANDARD_DIMS.items()
             for s in (0, 1)]

# Published minimum sums of squares (Moré, Garbow and Hillstrom 1981) at
# these (pid, n, m); local minima noted where the start reaches one.
KNOWN_MINIMA = {
    (1, 9, 45): 45 - 9,
    (2, 7, 35): 35 * 34 / (2 * 71),
    (3, 7, 35): (35 ** 2 + 3 * 35 - 6) / (2 * (2 * 35 - 3)),
    (4, 2, 2): 0.0,
    (5, 3, 3): 0.0,
    (6, 4, 4): 0.0,
    (7, 2, 2): 48.9842536792,          # local minimum (global 0)
    (8, 3, 15): 8.21487e-3,
    (9, 4, 11): 3.07505e-4,
    (10, 3, 16): 87.9458,
    (11, 6, 31): 2.28767e-3,
    (12, 3, 10): 0.0,
    (13, 2, 10): 124.362,
    (14, 4, 20): 85822.2,
    (15, 8, 8): 3.51687e-3,
    (16, 10, 10): 0.0,
    (17, 5, 33): 5.46489e-5,
    (18, 11, 65): 4.01377e-2,
}

_BARD_Y = np.array([0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35, 0.39, 0.37,
                    0.58, 0.73, 0.96, 1.34, 2.10, 4.39])
_KO_Y = np.array([0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456,
                  0.0342, 0.0323, 0.0235, 0.0246])
_KO_U = np.array([4.0, 2.0, 1.0, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833,
                  0.0714, 0.0625])
_MEYER_Y = np.array([34780.0, 28610.0, 23650.0, 19630.0, 16370.0, 13720.0,
                     11540.0, 9744.0, 8261.0, 7030.0, 6005.0, 5147.0, 4427.0,
                     3820.0, 3307.0, 2872.0])
_OSB1_Y = np.array([0.844, 0.908, 0.932, 0.936, 0.925, 0.908, 0.881, 0.850,
                    0.818, 0.784, 0.751, 0.718, 0.685, 0.658, 0.628, 0.603,
                    0.580, 0.558, 0.538, 0.522, 0.506, 0.490, 0.478, 0.467,
                    0.457, 0.448, 0.438, 0.431, 0.424, 0.420, 0.414, 0.411,
                    0.406])
_OSB2_Y = np.array([1.366, 1.191, 1.112, 1.013, 0.991, 0.885, 0.831, 0.847,
                    0.786, 0.725, 0.746, 0.679, 0.608, 0.655, 0.616, 0.606,
                    0.602, 0.626, 0.651, 0.724, 0.649, 0.649, 0.694, 0.644,
                    0.624, 0.661, 0.612, 0.558, 0.533, 0.495, 0.500, 0.423,
                    0.395, 0.375, 0.372, 0.391, 0.396, 0.405, 0.428, 0.429,
                    0.523, 0.562, 0.607, 0.653, 0.672, 0.708, 0.633, 0.668,
                    0.645, 0.632, 0.591, 0.559, 0.597, 0.625, 0.739, 0.710,
                    0.729, 0.720, 0.636, 0.581, 0.428, 0.292, 0.162, 0.098,
                    0.054])
assert len(_BARD_Y) == 15 and len(_KO_Y) == 11 and len(_KO_U) == 11
assert len(_MEYER_Y) == 16 and len(_OSB1_Y) == 33 and len(_OSB2_Y) == 65


def x0(pid: int, n: int) -> np.ndarray:
    """Standard start of problem ``pid`` in dimension ``n``."""
    if pid in (1, 2, 3, 19):
        return np.ones(n)
    fixed = {
        4: [-1.2, 1.0], 5: [-1.0, 0.0, 0.0], 6: [3.0, -1.0, 0.0, 1.0],
        7: [0.5, -2.0], 8: [1.0, 1.0, 1.0], 9: [0.25, 0.39, 0.415, 0.39],
        10: [0.02, 4000.0, 250.0], 12: [0.0, 10.0, 20.0], 13: [0.3, 0.4],
        14: [25.0, 5.0, -5.0, -1.0], 17: [0.5, 1.5, -1.0, 0.01, 0.02],
        18: [1.3, 0.65, 0.65, 0.7, 0.6, 3.0, 5.0, 7.0, 2.0, 4.5, 5.5],
        22: [-0.3, -0.39, 0.3, -0.344, -1.2, 2.69, 1.59, -1.5],
    }
    if pid in fixed:
        out = np.array(fixed[pid], dtype=np.float64)
        if len(out) != n:
            raise ValueError(f"problem {pid} has n = {len(out)}, not {n}")
        return out
    if pid == 11:
        return np.zeros(n)
    if pid == 15:
        return np.arange(1, n + 1) / (n + 1)
    if pid in (16, 20):
        return np.full(n, 0.5)
    if pid == 21:
        out = np.empty(n)
        for i in range(1, n + 1):
            ss = 0.0
            for j in range(1, n + 1):
                v = math.sqrt(i / j)
                ss += v * (math.sin(math.log(v)) ** 5 + math.cos(math.log(v)) ** 5)
            out[i - 1] = -8.710996e-4 * ((i - 50) ** 3 + ss)
        return out
    raise ValueError(f"unknown problem {pid}")


def residual(pid: int, x: np.ndarray, m: int) -> np.ndarray:
    """Residual vector ``r(x)`` (length ``m``) of problem ``pid``."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if pid == 1:
        s = 2.0 * np.sum(x) / m + 1.0
        out = np.full(m, -s)
        out[:n] += x
        return out
    if pid == 2:
        s = float(np.sum(np.arange(1, n + 1) * x))
        return np.arange(1, m + 1) * s - 1.0
    if pid == 3:
        s = float(np.sum(np.arange(2, n) * x[1:n - 1]))
        out = np.arange(0, m) * s - 1.0
        out[m - 1] = -1.0
        return out
    if pid == 4:
        return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])
    if pid == 5:
        if x[0] > 0:
            th = math.atan(x[1] / x[0]) / (2 * math.pi)
        elif x[0] < 0:
            th = math.atan(x[1] / x[0]) / (2 * math.pi) + 0.5
        else:
            th = 0.25
        r = math.hypot(x[0], x[1])
        return np.array([10.0 * (x[2] - 10.0 * th), 10.0 * (r - 1.0), x[2]])
    if pid == 6:
        return np.array([x[0] + 10.0 * x[1], math.sqrt(5.0) * (x[2] - x[3]),
                         (x[1] - 2.0 * x[2]) ** 2,
                         math.sqrt(10.0) * (x[0] - x[3]) ** 2])
    if pid == 7:
        return np.array([-13.0 + x[0] + ((5.0 - x[1]) * x[1] - 2.0) * x[1],
                         -29.0 + x[0] + ((x[1] + 1.0) * x[1] - 14.0) * x[1]])
    if pid == 8:
        i = np.arange(1, 16, dtype=np.float64)
        v = 16.0 - i
        w = np.minimum(i, v)
        return _BARD_Y - (x[0] + i / (v * x[1] + w * x[2]))
    if pid == 9:
        u = _KO_U
        return _KO_Y - x[0] * (u * (u + x[1])) / (u * (u + x[2]) + x[3])
    if pid == 10:
        t = 45.0 + 5.0 * np.arange(1, 17)
        return x[0] * np.exp(x[1] / (t + x[2])) - _MEYER_Y
    if pid == 11:
        t = np.arange(1, 30) / 29.0
        powers = t[:, None] ** np.arange(n)[None, :]
        s1 = (powers[:, :n - 1] * (np.arange(1, n) * x[1:])[None, :]).sum(1)
        s2 = powers @ x
        out = np.empty(31)
        out[:29] = s1 - s2 ** 2 - 1.0
        out[29] = x[0]
        out[30] = x[1] - x[0] ** 2 - 1.0
        return out
    if pid == 12:
        t = np.arange(1, m + 1) / 10.0
        return (np.exp(-t * x[0]) - np.exp(-t * x[1])
                - x[2] * (np.exp(-t) - np.exp(-10.0 * t)))
    if pid == 13:
        i = np.arange(1, m + 1, dtype=np.float64)
        return 2.0 + 2.0 * i - (np.exp(i * x[0]) + np.exp(i * x[1]))
    if pid == 14:
        t = np.arange(1, m + 1) / 5.0
        return ((x[0] + t * x[1] - np.exp(t)) ** 2
                + (x[2] + x[3] * np.sin(t) - np.cos(t)) ** 2)
    if pid == 15:
        out = np.zeros(m)
        for xj in x:
            t1, t2 = 1.0, 2.0 * xj - 1.0
            t = 2.0 * t2
            for i in range(m):
                out[i] += t2
                th = t * t2 - t1
                t1, t2 = t2, th
        out /= n
        even = np.arange(1, m + 1) % 2 == 0
        i = np.arange(1, m + 1, dtype=np.float64)
        out[even] += 1.0 / (i[even] ** 2 - 1.0)
        return out
    if pid == 16:
        out = x + np.sum(x) - (n + 1.0)
        out[n - 1] = float(np.prod(x)) - 1.0
        return out
    if pid == 17:
        t = 10.0 * np.arange(33)
        return _OSB1_Y - (x[0] + x[1] * np.exp(-t * x[3])
                          + x[2] * np.exp(-t * x[4]))
    if pid == 18:
        t = np.arange(65) / 10.0
        return _OSB2_Y - (x[0] * np.exp(-t * x[4])
                          + x[1] * np.exp(-(t - x[8]) ** 2 * x[5])
                          + x[2] * np.exp(-(t - x[9]) ** 2 * x[6])
                          + x[3] * np.exp(-(t - x[10]) ** 2 * x[7]))
    if pid == 19:
        out = np.empty(m)
        for i in range(n - 4):
            out[i] = -4.0 * x[i] + 3.0
            out[n - 4 + i] = (x[i] ** 2 + 2 * x[i + 1] ** 2 + 3 * x[i + 2] ** 2
                              + 4 * x[i + 3] ** 2 + 5 * x[n - 1] ** 2)
        return out
    if pid == 20:
        out = np.empty(n)
        out[0] = x[0] - 1.0
        out[1:] = 10.0 * (x[1:] - x[:-1] ** 3)
        return out
    if pid == 21:
        out = np.empty(n)
        j = np.arange(1, n + 1, dtype=np.float64)
        for i in range(1, n + 1):
            v2 = np.sqrt(x[i - 1] ** 2 + i / j)
            lg = np.log(v2)
            ss = float(np.sum(v2 * (np.sin(lg) ** 5 + np.cos(lg) ** 5)))
            out[i - 1] = 1400.0 * x[i - 1] + (i - 50.0) ** 3 + ss
        return out
    if pid == 22:
        a, b, c, d, t, u, v, w = x
        return np.array([
            a + b + 0.69,
            c + d + 0.044,
            t * a + u * b - v * c - w * d + 1.57,
            v * a + w * b + t * c + u * d + 1.31,
            a * (t ** 2 - v ** 2) - 2.0 * c * t * v + b * (u ** 2 - w ** 2)
            - 2.0 * d * u * w + 2.65,
            c * (t ** 2 - v ** 2) + 2.0 * a * t * v + d * (u ** 2 - w ** 2)
            + 2.0 * b * u * w - 2.0,
            a * t * (t ** 2 - 3.0 * v ** 2) + c * v * (v ** 2 - 3.0 * t ** 2)
            + b * u * (u ** 2 - 3.0 * w ** 2) + d * w * (w ** 2 - 3.0 * u ** 2)
            + 12.6,
            c * t * (t ** 2 - 3.0 * v ** 2) - a * v * (v ** 2 - 3.0 * t ** 2)
            + d * u * (u ** 2 - 3.0 * w ** 2) - b * w * (w ** 2 - 3.0 * u ** 2)
            - 9.48])
    raise ValueError(f"unknown problem {pid}")


def instance_name(pid: int, n: int, m: int, s: int) -> str:
    return f"{pid:02d}_{NAMES[pid]}_n{n}_m{m}_s{s}"
