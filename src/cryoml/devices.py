"""Table 6 device list and reported RRMS / sigma values."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Device:
    dev_type: str           # "nmos" or "pmos"
    L_um: float
    W_um: float
    paper_rrms: float | None = None
    paper_sigma: float | None = None


PAPER_DEVICES: tuple[Device, ...] = (
    # nMOS — Table 6
    Device("nmos", 0.15, 1.6, 0.059, 0.070),
    Device("nmos", 0.19, 7.0, 0.097, 0.134),
    Device("nmos", 0.25, 1.6, 0.098, 0.110),
    Device("nmos", 1.00, 1.6, 0.128, 0.165),
    Device("nmos", 1.00, 3.0, 0.160, 0.120),
    Device("nmos", 8.00, 1.6, 0.147, 0.162),
    Device("nmos", 20.0, 0.64, 0.130, 0.082),
    Device("nmos", 100.0, 100.0, 0.142, 0.123),
    # pMOS — Table 6
    Device("pmos", 0.35, 0.55, 0.701, 1.53),
    Device("pmos", 0.35, 1.6, 0.374, 0.801),
    Device("pmos", 0.35, 5.0, 0.324, 0.484),
    Device("pmos", 0.5, 0.42, 0.465, 0.724),
    Device("pmos", 0.5, 0.64, 0.322, 0.637),
    Device("pmos", 2.0, 5.0, 0.207, 0.407),
    Device("pmos", 4.0, 7.0, 0.281, 0.432),
    Device("pmos", 8.0, 0.84, 0.480, 0.902),
    Device("pmos", 8.0, 1.6, 0.515, 1.05),
    Device("pmos", 8.0, 5.0, 0.388, 0.699),
)


def find_device(dev_type: str, L_um: float, W_um: float, tol: float = 1e-3) -> Device | None:
    for d in PAPER_DEVICES:
        if (
            d.dev_type == dev_type
            and abs(d.L_um - L_um) < tol
            and abs(d.W_um - W_um) < tol
        ):
            return d
    return None


def parse_device_spec(spec: str) -> tuple[str, float, float]:
    """Parse strings like ``nmos:0.15:1.6`` into ``(type, L, W)``."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"bad device spec {spec!r}; expected 'nmos:L:W'")
    dev_type = parts[0].lower()
    if dev_type not in ("nmos", "pmos"):
        raise ValueError(f"dev_type must be nmos or pmos, got {dev_type!r}")
    return dev_type, float(parts[1]), float(parts[2])


def parse_device_list(spec: str | None) -> list[Device]:
    """Parse CLI flag like ``nmos:0.15:1.6,pmos:0.35:0.55``.

    None or empty selects all Table 6 devices.
    """
    if not spec:
        return list(PAPER_DEVICES)
    out: list[Device] = []
    for chunk in spec.split(","):
        dt, L, W = parse_device_spec(chunk.strip())
        d = find_device(dt, L, W)
        if d is None:
            # Allow off-list geometries too — keep paper_rrms None.
            d = Device(dt, L, W)
        out.append(d)
    return out
