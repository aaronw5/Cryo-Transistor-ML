"""Parameter surfaces for expanded BSIM4 DC-current experiments.

The seven-parameter tuple reproduces the paper.  ``MAX_DC_CANDIDATES`` is a
deliberately broad *screening* surface, not an assertion that every listed
parameter is identifiable from the available I-V curves.  The sensitivity
study removes only terms that are absent, unsafe, or numerically inactive for
all selected devices; it does not impose a top-k limit.
"""

from __future__ import annotations

PARAMS7 = (
    "vth0", "u0", "nfactor", "vsat", "delta", "rdsw", "eta0",
)

# Parameters represented in the linked upstream nominal library.  Its runnable
# Monte Carlo script currently varies only PARAMS7, but this list documents the
# author's broader extraction surface.
UPSTREAM_EXPANDED18 = (
    "vth0", "dvt0", "dvt1", "vsat", "ua", "ub", "uc", "rdsw",
    "prwg", "u0", "rdw", "rsw", "nfactor", "eta0", "dsub", "pclm",
    "pvag", "delta",
)

# Continuous BSIM4 terms that can participate in a DC drain-current sweep for
# the model-card configuration used by this repository.  Geometry-binning
# bounds/scaling coefficients, integer selectors, capacitance-only terms,
# noise-only terms, and gate-current terms behind disabled IGC/IGB selectors
# are intentionally not candidates.  Temperature and diode terms are included
# here and allowed to screen out numerically at the actual 77 K operating point.
PARAMETER_GROUPS: dict[str, tuple[str, ...]] = {
    "process_and_effective_geometry": (
        "toxe", "toxm", "xj", "ndep", "nsd", "dwg", "dwb", "rsh",
    ),
    "threshold_and_short_channel": (
        "vth0", "k1", "k2", "k3", "dvt0", "dvt1", "dvt2",
        "dvt0w", "dvt1w", "dvt2w", "w0", "k3b", "vfb", "phin",
        "lpe0", "lpeb", "vbm", "dvtp0", "dvtp1",
    ),
    "mobility_velocity_and_resistance": (
        "vsat", "ua", "ub", "uc", "rdsw", "prwb", "prwg", "wr",
        "u0", "a0", "keta", "a1", "a2", "ags", "b0", "b1", "eu",
        "up", "lp", "ud", "rdswmin", "rdw", "rdwmin", "rsw",
        "rswmin",
    ),
    "subthreshold_and_dibl": (
        "voff", "nfactor", "tvfbsdoff", "tvoff", "cit", "cdsc",
        "cdscb", "cdscd", "eta0", "etab", "dsub", "voffl", "minv",
    ),
    "output_conductance_and_high_field": (
        "pclm", "pdiblc1", "pdiblc2", "pdiblcb", "drout", "pscbe1",
        "pscbe2", "pvag", "delta", "alpha0", "alpha1", "beta0",
        "fprout", "pdits", "pditsl", "pditsd",
    ),
    "gidl": (
        "agidl", "bgidl", "cgidl", "egidl",
    ),
    "temperature": (
        "vfbsdoff", "kt1", "kt2", "at", "ute", "ua1", "ub1",
        "uc1", "kt1l", "prt",
    ),
    "body_network_and_junction_dc": (
        "rbpb", "rbpd", "rbps", "rbdb", "rbsb", "gbmin", "njs",
        "jss", "jsws", "xtis", "bvs", "xjbvs", "ijthsrev",
        "ijthsfwd", "tpb", "tpbsw", "tpbswg", "tcj", "tcjsw",
        "tcjswg",
    ),
}


def _unique(values):
    return tuple(dict.fromkeys(values))


MAX_DC_CANDIDATES = _unique(
    parameter
    for group in PARAMETER_GROUPS.values()
    for parameter in group
)

PARAMETER_GROUP_BY_NAME = {
    parameter: group
    for group, parameters in PARAMETER_GROUPS.items()
    for parameter in parameters
}

# Known integer/discrete controls and non-DC surfaces are documented so the
# generated audit can distinguish deliberate exclusion from accidental loss.
PRE_SCREEN_EXCLUSIONS: dict[str, str] = {
    "level": "integer model selector",
    "version": "model-version selector",
    "mobmod": "integer mobility-model selector",
    "binunit": "integer binning selector",
    "igcmod": "integer gate-current selector",
    "igbmod": "integer gate-current selector",
    "rgatemod": "integer gate-resistance selector",
    "rbodymod": "integer body-resistance selector",
    "trnqsmod": "integer NQS selector",
    "acnqsmod": "integer AC NQS selector",
    "fnoimod": "integer noise-model selector",
    "tnoimod": "integer noise-model selector",
    "permod": "integer perimeter selector",
    "geomod": "integer geometry selector",
    "rdsmod": "integer source/drain-resistance selector",
    "tempmod": "integer temperature-model selector",
    "diomod": "integer diode-model selector",
    "capmod": "integer capacitance-model selector",
    "lmin": "geometry-bin bound",
    "lmax": "geometry-bin bound",
    "wmin": "geometry-bin bound",
    "wmax": "geometry-bin bound",
    "cgdo": "capacitance-only parameter",
    "cgso": "capacitance-only parameter",
    "cgbo": "capacitance-only parameter",
    "noia": "noise-only parameter",
    "noib": "noise-only parameter",
    "noic": "noise-only parameter",
    "kf": "noise-only parameter",
    "af": "noise-only parameter",
    "ef": "noise-only parameter",
}

