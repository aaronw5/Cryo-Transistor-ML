"""Parameter sets and boxes around each pdk2 device's stock card.

GF180 (BSIM4, per-bin cards):

* ``seven``: the paper's seven BSIM4 parameters, linear +/-10% of stock.
* ``wide``: 21 DC parameters (the upstream expanded-18 set minus RDW/RSW,
  inactive at RDSMOD = 0, and minus UB, ~3e-18, which ``ParameterBox``'s
  1e-14 absolute-width rule would silently freeze; plus K1, K2, VOFF,
  PDIBLC1, PDIBLC2, RDSWMIN).  All 21 are active for every device.  Non-zero
  stock values get +/-10%: log-uniform for positive-definite scale
  parameters, linear for offset/sign-indefinite ones.  Parameters whose
  stock value is 0 get a one-sided absolute range (``ZERO_RANGE``).

IHP SG13G2 (PSP 103.6, one global geometry-scaled card):

* ``seven``: the PSP analogue of the paper set: VFBO (threshold), UO
  (mobility), CTL (interface-state subthreshold slope), THESATO (velocity
  saturation), AXO (linear/saturation transition), RSW1 (series
  resistance), CFL (DIBL); linear +/-10% of the evaluated stock values.
* ``wide``: those plus MUEW, THEMUO, DPHIBO, DPHIBL, THESATL (12; with VFBO,
  CTL, RSW1 these are the DC knobs IHP's own corners vary).

``spec`` returns the cold-start round-spec layout (``selected_parameters``,
``published_by_device``, ``bounds_by_device``, ``log_scale_by_device``) keyed
by pdk2 tags, so ``cold_loop.box_for`` and ``pdk_gen_full_data.build_box``
build the same ``ParameterBox`` unchanged.
"""

from __future__ import annotations

import hashlib
import json

from ..parameter_space import ParameterBox

SEVEN = ("vth0", "u0", "nfactor", "vsat", "delta", "rdsw", "eta0")
WIDE = ("vth0", "k1", "k2", "dvt0", "dvt1", "dsub", "u0", "ua", "uc",
        "vsat", "rdsw", "rdswmin", "prwg", "nfactor", "voff", "eta0", "pclm",
        "pdiblc1", "pdiblc2", "pvag", "delta")
PARAMETER_SETS = {"seven": SEVEN, "wide": WIDE}
REL_HALF_WIDTH = 0.10
# Positive-definite scale parameters: log-uniform in the wide box.
LOG_PARAMS = frozenset({"k1", "dvt0", "dvt1", "dsub", "u0", "vsat", "rdsw",
                        "rdswmin", "nfactor", "eta0", "pclm", "pdiblc1", "pdiblc2",
                        "delta"})
# Absolute ranges where the stock value is exactly zero (nFET dvt0, both
# families' prwg and pvag): roll-off, Rds gate dependence and Early-voltage
# gate dependence, each limited to an effect of the order of the 10% box.
ZERO_RANGE = {"dvt0": (0.0, 0.5), "prwg": (0.0, 0.1), "pvag": (0.0, 0.25)}

# Non-fitted DC parameters perturbed by the misspecified truth tier (outside
# both fitted sets; non-zero in every bin of both families).
MISSPEC_PARAMS = ("a0", "ags", "keta", "lpe0", "minv", "etab", "drout",
                  "pscbe1", "pscbe2", "alpha0", "beta0", "eu")

IHP_SEVEN = ("vfbo", "uo", "ctl", "thesato", "axo", "rsw1", "cfl")
IHP_WIDE = IHP_SEVEN + ("muew", "themuo", "dphibo", "dphibl", "thesatl")
IHP_LOG_PARAMS = frozenset({"uo", "ctl", "thesato", "axo", "rsw1", "cfl",
                            "muew", "themuo", "thesatl"})
IHP_ZERO_RANGE: dict[str, tuple[float, float]] = {}
# Non-fitted PSP DC parameters (non-zero in both families): mobility field
# exponent, interface-state offset, pocket implant (FBET1, LP1), channel-
# length modulation (ALPL, ALPLEXP), AX length scaling, CT length exponent,
# substrate doping.  (CFBO, the back-bias DIBL term, is inert at Vsb = 0.)
# MUEO dominates: a 20% change alone leaves ~0.08 RRMS after a seven-
# parameter refit, hence IHP's narrower tier-b width (make_truth.py).
IHP_MISSPEC_PARAMS = ("mueo", "cto", "fbet1", "lp1", "alpl", "alplexp",
                      "axl", "ctlexp", "nsubo")

PDK_SETS = {"gf180": {"seven": SEVEN, "wide": WIDE},
            "ihp": {"seven": IHP_SEVEN, "wide": IHP_WIDE}}
PDK_LOG = {"gf180": LOG_PARAMS, "ihp": IHP_LOG_PARAMS}
PDK_ZERO = {"gf180": ZERO_RANGE, "ihp": IHP_ZERO_RANGE}
PDK_MISSPEC = {"gf180": MISSPEC_PARAMS, "ihp": IHP_MISSPEC_PARAMS}
PDK_LABEL = {"gf180": "gf180mcuD", "ihp": "ihp-sg13g2"}


def backend(pdk: str):
    """The simulator module of a pdk (``gf180`` or ``ihp``)."""
    if pdk == "gf180":
        from . import gf180
        return gf180
    if pdk == "ihp":
        from . import ihp
        return ihp
    raise KeyError(pdk)


def pdk_of_tag(tag: str) -> str:
    return "ihp" if tag.startswith("ihp_") else "gf180"


def device_box_rows(device, names: tuple[str, ...], *,
                    log_scale: bool) -> tuple[dict, dict, dict]:
    pdk = pdk_of_tag(device.tag)
    stock = backend(pdk).stock_params(device, names)
    zero_range, log_params = PDK_ZERO[pdk], PDK_LOG[pdk]
    published, bounds, logs = {}, {}, {}
    for name in names:
        value = float(stock[name])
        if value == 0.0:
            if name not in zero_range:
                raise ValueError(f"{device.tag}: {name} is 0 without a range")
            bounds[name] = list(zero_range[name])
            logs[name] = False
        else:
            lo, hi = value * (1 - REL_HALF_WIDTH), value * (1 + REL_HALF_WIDTH)
            bounds[name] = [min(lo, hi), max(lo, hi)]
            logs[name] = bool(log_scale and name in log_params and value > 0)
        published[name] = value
    return published, bounds, logs


def spec(set_name: str, devices=None, pdk: str = "gf180") -> dict:
    if devices is None:
        devices = backend(pdk).DEVICES
    elif devices:
        pdk = pdk_of_tag(devices[0].tag)
    names = PDK_SETS[pdk][set_name]
    published, bounds, logs = {}, {}, {}
    for device in devices:
        published[device.tag], bounds[device.tag], logs[device.tag] = \
            device_box_rows(device, names, log_scale=(set_name == "wide"))
    return {
        "schema_version": 1, "kind": "pdk2_box_spec", "pdk": PDK_LABEL[pdk],
        "parameter_set": set_name, "selected_parameters": list(names),
        "published_by_device": published, "bounds_by_device": bounds,
        "log_scale_by_device": logs,
        "box_rule": (f"+/-{REL_HALF_WIDTH:.0%} of the stock card; "
                     f"zero-stock absolute ranges {PDK_ZERO[pdk]}"),
    }


def box_from_spec(box_spec: dict, tag: str) -> ParameterBox:
    """Same construction as ``cold_loop.box_for``."""
    names = tuple(box_spec["selected_parameters"])
    flags = box_spec["log_scale_by_device"][tag]
    return ParameterBox(names=names, published=box_spec["published_by_device"][tag],
                        bounds=box_spec["bounds_by_device"][tag],
                        log_scale=tuple(bool(flags[n]) for n in names))


def box_for(set_name: str, tag: str) -> ParameterBox:
    device = backend(pdk_of_tag(tag)).device_by_tag(tag)
    return box_from_spec(spec(set_name, [device]), tag)


def spec_signature(box_spec: dict) -> str:
    """As ``pdk_gen_full_data.spec_signature`` for log-scaled specs."""
    names = tuple(box_spec["selected_parameters"])
    digest = hashlib.sha256(json.dumps(
        [box_spec["bounds_by_device"], box_spec["published_by_device"]],
        sort_keys=True).encode()).hexdigest()
    return hashlib.sha256("\n".join(names + (digest,)).encode()).hexdigest()[:16]
