# I10-1 feasibility check, 2026-10-21: 4 K I-V from room-temperature data

Candidate question: predict a held-out device's 4 K I-V from its
room-temperature (RT) measurement only. Training devices have both
temperatures. Arms: (a) the device's RT card fit C0_RT simulated at 4 K with
the BSIM4 temperature model; (b) C0_RT plus a shared parameter shift
Δθ = mean over training devices of (4 K card − RT card); (c) (b) plus the
M_bias correction g(pol, |Vgs|/VDD, |Vds|/VDD) learned at 4 K
(`src/cryoml/explore/i3_2_bias.py`); (d) a regression from RT card and RT
residual features to the 4 K card. Score: strict paper RRMS on the held-out
device's 4 K curves, measured-only inclusion (`i8_1_nist.inclusion_measured`).

This check fitted no card. It ran 12 single-device ngspice-41 processes (each
with one to eleven sweeps) while the I9-2 queue job was running.

## Data used in this check

| Item | Location | Note |
|---|---|---|
| NIST mds2-2997 ETest Tile, 4 K | `out/research/data_raw/nist_mds2_2997/ext_etest/ETest Tile/Sky130-4k` (I8-1 download) | 228 MDM files, 114 devices |
| google/skywater-pdk-sky130-raw-data, RT | `out/research/data_raw/google_sky130_raw_data/files/sky130_fd_pr/cells/<type>/` (new, gitignored) | 228 MDM files (12,293,568 bytes) fetched by raw URL at commit `1f7214f2f81a0629303947a07e1ec00a28ae8a33` (main, 2023-02-08); tree listing in `.../meta/tree.json`; all 228 git blob ids match the tree listing |
| Pairing helper | `src/cryoml/explore/i10_1_etest_rt.py`, test `tests/test_i10_1_etest_rt.py` | RT file = 4 K file name without `_4K`; `pair_etest()` returns 114 pairs (25 core, 81 g5v0d10v5) |

All downloaded files were read as text with `.venv/bin/python -I`
(`i5_1_ihp_real.parse_mdm`, `i8_1_nist.file_curves`). Nothing from the
downloads was executed.

## 1. Devices with a usable file at both temperatures

All 114 ETest devices have Id-Vg and Id-Vd files at 4 K and RT under the same
module/pad ID. "Pass checks" means the I8-1 `device_checks` gate-control test
(|Id|(Vg max)/|Id|(Vg 0) at Vd max, Vb 0, at least 100) and no non-finite,
all-zero or compliance-limited curve, at both temperatures. "In a bin" means
lmin ≤ L ≤ lmax and wmin ≤ W ≤ wmax for at least one bin of the open_pdks tt
card.

| Type | Devices | Distinct W/L | Pass checks (both T) | Pass checks and in a bin |
|---|---|---|---|---|
| nfet_01v8 | 5 | 5 | 5 | 5 |
| nfet_01v8_lvt | 5 | 5 | 5 | 5 |
| pfet_01v8 | 5 | 5 | 5 | 5 |
| pfet_01v8_lvt | 5 | 5 | 5 | 5 |
| pfet_01v8_hvt | 5 | 5 | 5 | 5 |
| 1.8 V core total | 25 | | 25 | 25 |
| nfet_g5v0d10v5 | 46 | 40 | 43 | 19 (15 W/L) |
| pfet_g5v0d10v5 | 35 | 34 | 34 | 11 (11 W/L) |
| nfet_05v0_nvt | 5 | 5 | 5 | 5 |
| nfet_03v3_nvt (native) | 3 | 2 | 0 | 0 |
| Total | 114 | | 107 | 70 |

Devices failing checks: nfet_g5v0d10v5 m8389_11 (W 0.42/L 0.5; RT gate
control 0.98, 4 K/RT linear current ratio 13,821), m8063_8 (W 3/L 0.35; RT gate
control 2.0), m8063_3 (W 5/L 0.3; 4 K 37.9, RT 7.6);
pfet_g5v0d10v5 m8064_4 (W 0.42/L 0.35; RT 22.5). The three native
nfet_03v3_nvt devices conduct at Vg = 0 at RT (gate control 53-64), as a
native device does.

Core-device geometries (W/L in µm):

- nfet_01v8, pfet_01v8, pfet_01v8_hvt: 0.42/0.15, 0.42/8, 1.68/0.15, 7/0.15, 7/8
- nfet_01v8_lvt: 0.42/0.15, 0.42/1, 0.84/0.15, 7/0.15, 7/8
- pfet_01v8_lvt: 0.42/0.35, 0.42/8, 3/1, 7/0.35, 7/8

g5v0d10v5 geometries: W 0.42-25 µm, L 0.3-25 µm; 50 of the 81 devices (47 of the 77
that pass the checks) have L = 0.3-0.45 µm, below the card lmin of 0.5 µm.

### Bias grids

| Group | Temperature | Id-Vg: Vd values, Vg range, step, points | Id-Vd: Vg values, Vd range, step, points | Vb values |
|---|---|---|---|---|
| 1.8 V nMOS (01v8, lvt) | 4 K | 0.1, 1.8 V; 0-1.8 V; 10 mV; 181 | 0, 0.3, ..., 1.8 V (7); 0-1.8 V; 25 mV; 73 | Id-Vg 0, -0.75, -1.5; Id-Vd 0, -1.5 |
| 1.8 V nMOS | RT | 0.1, 1.8 V; 0-1.8 V; 50 mV; 37 | 0, 0.36, ..., 1.8 V (6); 0-1.8 V; 50 mV; 37 | Id-Vg 0, -0.9, -1.8; Id-Vd 0, -0.9 |
| 1.8 V pMOS (01v8, lvt, hvt) | 4 K | -0.1, -1.8 V; mirrored; 10 mV; 181 | 0, -0.3, -0.6, -0.9, -1.2, -1.5, -1.6, -1.7, -1.8 V (9); 25 mV; 73 | Id-Vg 0, 0.75, 1.5; Id-Vd 0, 1.5 |
| 1.8 V pMOS | RT | -0.1, -1.8 V; 50 mV; 37 | 0, -0.36, ..., -1.8 V (6); 50 mV; 37 | Id-Vg 0, 0.9, 1.8; Id-Vd 0, 0.9 |
| g5v0d10v5, 05v0_nvt | 4 K | ±0.1, ±3.3 V; 0-3.3 V; 10 mV; 331 | 0, 0.55, ..., 3.3 V (7); 25 mV; 133 | Id-Vg 0, ∓1.5, ∓3.0; Id-Vd 0, ∓2.0 |
| g5v0d10v5, 05v0_nvt | RT | ±0.1, ±5.0 V; 0-5.0 V; 50 mV; 101 | 0, 1, ..., 5 V (6); 50 mV; 101 | Id-Vg 0, ∓2.5, ∓5.0; Id-Vd 0, ∓2.5 |
| nfet_03v3_nvt | 4 K | 0.1, 1.8 V; 0-1.8 V | as 1.8 V nMOS | as 1.8 V nMOS |
| nfet_03v3_nvt | RT | 0.1, 3.0 V; 0-3.0 V; 30 mV; 101 | 0, 0.6, ..., 3.0 V (6) | Id-Vg 0, -1.5, -3.0 |

Comparability:

- 1.8 V core: the voltage ranges are the same (0-1.8 V). Shared Vb = 0
  curves: Id-Vg at |Vd| = 0.1 and 1.8 V, and Id-Vd at |Vg| = 0 and 1.8 V. The
  other Id-Vd gate voltages differ (0.36 V steps at RT, 0.3 V steps and
  1.5-1.7 V for pMOS at 4 K). Each temperature is scored on its own grid, so
  the difference matters only for arm (d) features and for the M_bias input
  coverage.
- Measurement floor: median |Id| at Vg = 0, Vb = 0 on Id-Vg is about 1e-9 A at
  RT and 7e-14 to 2e-12 A at 4 K. RT subthreshold data below about 1e-9 A is
  noise; C0_RT is therefore not constrained below that level.
- Inclusion (measured-only rule, Vb = 0 curves): 1.8 V nMOS 4 K 62/90, RT
  59/80; 1.8 V pMOS 4 K 87/165, RT 78/120. The long narrow pMOS devices
  (W 0.42/L 8) keep 2-3 curves at each temperature.
- g5v0d10v5: RT is swept to 5 V and 4 K to 3.3 V. A comparable RT fit would
  use the RT points with |V| ≤ 3.3 V: Id-Vg at |Vd| = 0.1 V only (the 5 V
  curve is outside the 4 K range) and Id-Vd at |Vg| = 1, 2, 3 V truncated at
  3.3 V.

### Temperature shift in the 1.8 V core data

Constant-current threshold at |Vd| = 0.1 V, Vb = 0 (|Id| = 1e-7 A × W/L,
log interpolation), and the ratio of |Id| at |Vg| = 1.8 V, |Vd| = 0.1 V
(4 K over RT). These are descriptive numbers from the measured files.

| Type | W | L | module_pad | Vth RT (V) | Vth 4 K (V) | ΔVth (V) | Id ratio 4 K/RT | incl. curves RT | incl. curves 4 K |
|---|---|---|---|---|---|---|---|---|---|
| nfet_01v8 | 0.42 | 0.15 | 8392_11 | 0.644 | 0.840 | +0.196 | 1.96 | 6/8 | 6/9 |
| nfet_01v8 | 0.42 | 8 | 8392_9 | 0.428 | 0.696 | +0.268 | 3.60 | 5/8 | 6/9 |
| nfet_01v8 | 1.68 | 0.15 | 8392_3 | 0.665 | 0.838 | +0.173 | 1.55 | 6/8 | 6/9 |
| nfet_01v8 | 7 | 0.15 | 8392_7 | 0.634 | 0.797 | +0.163 | 1.79 | 6/8 | 6/9 |
| nfet_01v8 | 7 | 8 | 8392_5 | 0.460 | 0.687 | +0.227 | 3.53 | 6/8 | 6/9 |
| nfet_01v8_lvt | 0.42 | 0.15 | 8391_9 | 0.587 | 0.696 | +0.110 | 1.67 | 6/8 | 6/9 |
| nfet_01v8_lvt | 0.42 | 1 | 8391_7 | 0.406 | 0.629 | +0.223 | 2.76 | 6/8 | 6/9 |
| nfet_01v8_lvt | 0.84 | 0.15 | 8391_11 | 0.591 | 0.753 | +0.162 | 1.66 | 6/8 | 6/9 |
| nfet_01v8_lvt | 7 | 0.15 | 8391_5 | 0.559 | 0.709 | +0.150 | 2.37 | 6/8 | 7/9 |
| nfet_01v8_lvt | 7 | 8 | 8391_3 | 0.368 | 0.584 | +0.216 | 3.83 | 6/8 | 7/9 |
| pfet_01v8 | 0.42 | 0.15 | 8397_12 | 0.638 | 1.260 | +0.622 | 0.60 | 6/8 | 7/11 |
| pfet_01v8 | 0.42 | 8 | 8397_10 | 0.860 | 1.425 | +0.566 | 0.94 | 3/8 | 3/11 |
| pfet_01v8 | 1.68 | 0.15 | 8397_4 | 0.704 | 1.287 | +0.583 | 1.09 | 6/8 | 7/11 |
| pfet_01v8 | 7 | 0.15 | 8397_8 | 0.746 | 1.301 | +0.556 | 0.81 | 6/8 | 7/11 |
| pfet_01v8 | 7 | 8 | 8397_6 | 1.018 | 1.504 | +0.486 | 1.03 | 4/8 | 5/11 |
| pfet_01v8_hvt | 0.42 | 0.15 | 8394_12 | 0.814 | 1.314 | +0.500 | 0.64 | 5/8 | 6/11 |
| pfet_01v8_hvt | 0.42 | 8 | 8394_10 | 0.948 | 1.462 | +0.515 | 0.95 | 2/8 | 2/11 |
| pfet_01v8_hvt | 1.68 | 0.15 | 8394_4 | 0.873 | 1.338 | +0.465 | 0.86 | 5/8 | 6/11 |
| pfet_01v8_hvt | 7 | 0.15 | 8394_8 | 0.898 | 1.353 | +0.455 | 0.84 | 6/8 | 7/11 |
| pfet_01v8_hvt | 7 | 8 | 8394_6 | 1.085 | 1.543 | +0.458 | 0.97 | 4/8 | 5/11 |
| pfet_01v8_lvt | 0.42 | 0.35 | 8396_12 | 0.257 | 1.216 | +0.959 | 0.65 | 7/8 | 7/11 |
| pfet_01v8_lvt | 0.42 | 8 | 8396_10 | 0.447 | 1.256 | +0.809 | 1.34 | 4/8 | 6/11 |
| pfet_01v8_lvt | 3 | 1 | 8396_4 | 0.395 | 1.248 | +0.853 | 1.08 | 7/8 | 6/11 |
| pfet_01v8_lvt | 7 | 0.35 | 8396_8 | 0.243 | 1.229 | +0.986 | 0.82 | 7/8 | 7/11 |
| pfet_01v8_lvt | 7 | 8 | 8396_6 | 0.392 | 1.261 | +0.869 | 1.86 | 6/8 | 6/11 |

Per type, ΔVth mean (population SD): nfet_01v8 0.205 (0.038), nfet_01v8_lvt
0.172 (0.042), pfet_01v8 0.562 (0.044), pfet_01v8_hvt 0.478 (0.024),
pfet_01v8_lvt 0.895 (0.067) V. Other families: nfet_05v0_nvt 0.324 (0.058),
nfet_g5v0d10v5 0.343 (0.194), pfet_g5v0d10v5 0.582 (0.171) V; the g5v0d10v5
spreads include long narrow devices whose RT threshold current (1e-7 A × W/L)
is near the RT floor, so those Vth values are not reliable.

Within a type the shift has a spread of 0.02-0.07 V and depends on L
(nMOS: larger ΔVth and current ratio at L = 8 µm than at L = 0.15 µm). Between
pMOS types the mean shift differs by up to 0.42 V (hvt 0.48 V, lvt 0.90 V).

## 2. Model cards and the BSIM4 4 K test

Cards: open_pdks sky130A `libs.tech/ngspice/corners/tt.spice` (the files
already used by I9-1, open_pdks a918dc7c). All are BSIM4 level 54, version
4.5, tnom 30 C, tempmod 0, with nonzero kt1, ute and related temperature
coefficients per bin.

| Type | tt file included by `corners/tt.spice` | Bins | Bin L range (µm) | Bin W range (µm) | ETest devices in a bin |
|---|---|---|---|---|---|
| nfet_01v8 | `sky130_fd_pr__nfet_01v8__tt.pm3.spice` | 180 | 0.15-100 | 0.36-100 | 5/5 |
| nfet_01v8_lvt | `sky130_fd_pr__nfet_01v8_lvt__tt.corner.spice` | 88 | 0.15-100 | 0.42-100 | 5/5 |
| pfet_01v8 | `sky130_fd_pr__pfet_01v8__tt.corner.spice` | 108 | 0.15-100 | 0.42-100 | 5/5 |
| pfet_01v8_lvt | `sky130_fd_pr__pfet_01v8_lvt__tt.corner.spice` | 48 | 0.35-100 | 0.42-100 | 5/5 |
| pfet_01v8_hvt | `sky130_fd_pr__pfet_01v8_hvt__tt.pm3.spice` | 198 | 0.15-100 | 0.36-100 | 5/5 |
| nfet_g5v0d10v5 | `sky130_fd_pr__nfet_g5v0d10v5__tt.corner.spice` | 88 | 0.5-100 | 0.42-100 | 20/46 |
| pfet_g5v0d10v5 | `sky130_fd_pr__pfet_g5v0d10v5__tt.corner.spice` | 88 | 0.5-100 | 0.42-100 | 11/35 |
| nfet_05v0_nvt | `sky130_fd_pr__nfet_05v0_nvt__tt.corner.spice` | 11 | 0.895-25.0 | 0.415-10.0 | 5/5 |
| nfet_03v3_nvt | `sky130_fd_pr__nfet_03v3_nvt__tt.corner.spice` | 9 | 0.495-0.805 | 0.415-10.0 | 3/3 |

The I9-1 card code (`i9_1_nist_rt.card_parts`, `force_bin`, `simulate`)
handles nfet_01v8 and pfet_01v8 only; the lvt and hvt types need the same
`SUBCKT`/file mapping added. The g5v0d10v5 devices with L < 0.5 µm stop with
"could not find a valid modelname" (tested: nfet_g5v0d10v5 W 5/L 0.4) and need
the I9-1 nearest-bin stretch.

BSIM4 at 4 K in ngspice-41 (`/Users/anrunw/cryo-ng41/arm64/bin/ngspice`),
`.options temp = -269.15`:

| Card, device | Deck | Return | Warnings | Output |
|---|---|---|---|---|
| nfet_01v8 bin 8, W 7/L 0.15 (I9-1 single-bin card) | Id-Vg Vd 1.8 V, 181 points, 30 / -196.15 / -269.15 C | 0 | 0 | finite at all temperatures |
| pfet_01v8 bin 8, W 7/L 0.15 (I9-1 single-bin card) | same, mirrored | 0 | 0 | finite |
| nfet_g5v0d10v5 W 5/L 0.5 (full `sky130.lib.spice tt`) | Id-Vg Vd 3.3 V, 7 points, 30 / -269.15 C | 0 | 0 | finite |
| pfet_g5v0d10v5 W 5/L 0.5 | same, mirrored | 0 | 0 | finite |
| pfet_01v8_lvt W 7/L 0.35 | Id-Vg Vd -1.8 V, 7 points | 0 | 0 | finite |
| nfet_g5v0d10v5 W 5/L 0.4 | | 1 | "could not find a valid modelname" | none (no bin) |

At -269.15 C the simulated subthreshold current drops to the gmin floor
(gmin × Vds: 1.8e-12 A at 1.8 V, 3.3e-12 A at 3.3 V) up to a threshold and
then rises in one or two 10 mV steps. Example, nfet_01v8 W 7/L 0.15,
Vd = 1.8 V: |Id| at Vg = 0.9 V is 1.79e-12 A simulated vs 4.64e-4 A
measured; at Vg = 1.2 V 1.62e-3 vs 1.87e-3 A. pfet_01v8 W 7/L 0.15: simulated
|Id| stays at the floor through |Vg| = 1.2 V (measured 5.79e-5 A).

Stock tt card (unfitted) RRMS with the measured-only inclusion on two
devices:

| Device | RT data, deck 30 C | 4 K data, deck 30 C | 4 K data, deck -269.15 C |
|---|---|---|---|
| nfet_01v8 W 7/L 0.15 | 0.133 (6/8 curves) | 0.311 (6/9) | 0.377 (6/9) |
| pfet_01v8 W 7/L 0.15 | 0.296 (6/8) | 2.650 (7/11) | 0.549 (7/11) |

Each ngspice process took 0.01-0.02 s for the single-bin card and about 7 s
for the full `sky130.lib.spice` library (library parsing).

## 3. Sample size, pooling, and other matched sets

1.8 V core (25 devices, 5 types × 5):

- Leave-one-out with a per-type Δθ trains Δθ on 4 devices, which cover 2-3
  distinct L values. The measured shift has a within-type SD of 0.02-0.07 V
  in Vth and an L dependence in the current ratio (nMOS 1.6-2.0 at
  L = 0.15 µm, 2.8-3.8 at L = 1-8 µm). A shared Δθ cannot represent the L
  dependence; arm (d) with log L, log W features can, but with 4 training
  devices per type it has at most 2-3 L values to fit.
- A Δθ pooled across types within a polarity misses the pMOS type
  differences (ΔVth 0.48 / 0.56 / 0.90 V for hvt / standard / lvt). Pooling
  across types needs at least a per-type VTH0 offset.
- Each core device uses a different bin of its card, so Δθ is applied to
  different bin values. An additive shift (VTH0, VOFF, VFB) and a
  multiplicative shift (other parameters, the log coordinates of
  `ParameterBox`) are comparable across bins; a shift in raw units is not.
- Statistics: 25 held-out devices in 5 clusters; per-type numbers rest on 5
  devices. A device-bootstrap CI over 25 devices is possible; a cluster CI
  over 5 types is wide.

Pooling the 5 V devices (g5v0d10v5, 77 passing checks):

- They use separate cards (thick oxide, different bins, VDD 3.3 V at 4 K),
  so they cannot share Δθ with the core types. They add data only as
  separate families with their own Δθ, or through the bias-normalized
  M_bias inputs (|V|/VDD), which the M_bias recipe shares across families.
- Larger families: LOO within nfet_g5v0d10v5 trains Δθ on 42 devices, within
  pfet_g5v0d10v5 on 33.
- Costs: 50 of 81 devices (47 of the 77 passing) need the bin stretch (L < 0.5 µm); the RT fit must
  be restricted to |V| ≤ 3.3 V (one Id-Vg curve and three truncated Id-Vd
  curves at Vb = 0); 4 devices fail checks.
- Effect on the core result: pooling helps only arm (c) (more 4 K residuals
  for g) and only if g transfers across oxide types. In I9-1 a g trained on
  another family (M_bias_lvt) applied to nfet/pfet_01v8 gave GMR 0.887 vs
  C0_refit and 1.112 vs the in-family M_bias; transfer from the thick-oxide
  family has not been tested. Recommendation: keep g5v0d10v5 out of the core TEST
  pool and use it as a development set (section 5).

Other matched 4 K + RT sets:

| Set | Devices with both temperatures | Card | Existing fits at both T | Note |
|---|---|---|---|---|
| TU Delft 40 nm (I6-1), campaign 1 | 69 (45 nMOS, 24 pMOS; 10 W/L groups of 2-8 copies; 4.13 K and 295.0 K) | CMOSedu 50 nm BSIM4 example card, temperature coefficients set to 0 | I6-1 starting cards for the 45 DEV devices at both temperatures (`out/research/explore/I6_1/start/summary.json`, `per_temperature.{4,295}.rows[*].params`); the 24 TEST devices are not fitted | Arm (a) is not meaningful here: with TC = 0 the card changes only through kT/q, band-gap and n_i. Arms (b) and (d) need no new fits on DEV. Campaign 2 has 600 two-temperature devices but only 2 Id-Vg curves over Vg 1.05-1.1 V and 1 Id-Vd curve (I3-1). |
| NIST MPW-5 `RT-for reference only` | 5 RT devices (one per folder: nthin, pthin, nthin_lvt, pthin_lvt, 5V) | sky130 | none | Files carry W/L only, no module/position; they cannot be matched to a 4 K device. |
| UTA 77 K (canonical 18) | 0 | | | `cryo_data` holds 77 K files only. |
| IHP SG13G2 (I5-1) | 0 at cryogenic temperature | | | 233-398 K only. |

## 4. Compute estimate

Reference: I9-1 start, 87 devices (MPW-5 4 K grid: 181-point Id-Vg, 91-point
Id-Vd), nfev 3000, irls 3: 472 s wall, mean 1,573 simulations per device;
with the script default of 6 workers this is about 32.5 worker-seconds per
device. One single-bin simulation of a full ETest 4 K device
takes 0.01-0.02 s. The ETest 4 K grid has 181/73-point curves (similar to
MPW-5); the RT grid has 37-point curves (lower cost per simulation).

| Job | Fits | Estimate (6 workers) |
|---|---|---|
| Core RT starting cards C0_RT (deck 30 C) | 25 | 2-3 min |
| Core 4 K starting cards (deck 30 C, I9-1 convention) | 25 | 2-3 min |
| Core 4 K starting cards, deck -269.15 C variant, or warm start from C0_RT | 25 each | 2-3 min each |
| Arms (a), (b), (d) on 25 held-out devices | 1 simulation per device and arm | under 1 min |
| Arm (c) M_bias, 25 LOO folds (training only, no held-out refit) | 25 trainings | about 5-15 min (I9-1 LODO with held-out refits: 87 folds in 2,435 s) |
| g5v0d10v5 DEV starting cards, both temperatures | 2 × 77 | about 14 min, plus about 7 min per extra 4 K variant |
| TU Delft DEV arms (b), (d) | existing cards | under 1 min |

Total for the core plus the g5v0d10v5 development set: about 40-60 min of
queue time, after the I9-2 queue ends.

## 5. Recommended dataset definition and split

Dataset I10-1:

- TEST (prospective, locked until the pre-registration and the DEV
  decisions are committed): the 25 ETest 1.8 V core devices, paired by
  module/pad ID with `i10_1_etest_rt.pair_etest(types=CORE_TYPES)`. All 25
  pass the checks and lie inside a card bin. Evaluation: leave-one-device-out
  within type (Δθ, arm (d) fitted on the other 4 devices of the type);
  secondary: Δθ pooled over the 12-13 other devices of the same polarity
  with a per-type VTH0 offset; arm (c) g trained on the other 24 core
  devices. Report per type and all-device, with device-bootstrap CIs and the
  5-type cluster CI.
- DEV (for choices: Δθ parameter subset and coordinates, deck-temperature
  convention, regression features, regularization, M_bias settings):
  1. ETest nfet_g5v0d10v5 (43) and pfet_g5v0d10v5 (34) devices that pass
     the checks; the 47 below lmin use the I9-1 bin stretch; RT fits on
     |V| ≤ 3.3 V; LOO within family.
  2. TU Delft I6-1 DEV, 45 devices, existing 295 K and 4.13 K starting
     cards; leave-one-geometry-out over the 7 DEV groups; arms (b) and (d)
     only.
- Excluded: nfet_03v3_nvt (3; native, 4 K range 0-1.8 V vs RT 0-3.0 V,
  fails the gate-control check), nfet_05v0_nvt (5; one type, too few for its
  own Δθ), the 4 g5v0d10v5 devices failing checks, the 24 TU Delft TEST
  devices (kept locked from I6-1).
- Scoring: 4 K curves at Vb = 0 of the held-out device, strict paper RRMS,
  measured-only inclusion frozen from the 4 K measured current (as in I8-1 and
  I9-1). Baselines to include: the stock tt card at 4 K (deck 30 C and
  -269.15 C), C0_RT at deck 30 C (no temperature change), and the device's
  own 4 K starting card (in-sample bound, not a prediction).

Pre-registration items that this check leaves open:

1. Deck-temperature convention. Option 1: both cards fitted at deck 30 C
   (I9-1 convention), Δθ absorbs the temperature change, arm (a) is the only
   arm that uses the BSIM4 temperature model. Option 2: 4 K cards fitted at
   deck -269.15 C, Δθ is the correction on top of arm (a). Option 2 depends
   on whether the 4 K fits can reach the measured subthreshold swing at
   -269.15 C (BSIM4 gives n × 0.79 mV/dec at 4.15 K, so NFACTOR must reach
   about 10-25 for the measured swing); this is to be tested on DEV.
2. 4 K card fit: independent from the stock card, or warm-started from
   C0_RT. Warm start makes (4 K card − C0_RT) a local difference; independent
   fits of 30-40 active parameters can differ by compensating parameter
   combinations that do not transfer between devices.
3. Δθ statistic: mean or median, in `ParameterBox` local coordinates.
4. Training residuals for arm (c): computed with each training device's own
   out-of-fold Δθ (nested), so that g is trained on the residual
   distribution of a prediction and not of an in-sample fit.

## Main risks

1. Sample size: 5 devices per type, 4 for training in LOO, 2-3 distinct L
   values; the type-cluster CI has 5 clusters.
2. Type and geometry dependence of the shift: pMOS ΔVth 0.46-0.99 V across
   types; nMOS current ratio 1.6-3.8 with L. A single shared Δθ per type
   leaves the L dependence to arm (d) or (c).
3. Die identity: the NIST README does not state whether the RT (SkyWater
   test tile, google raw data) and 4 K (NIST, March 2022) measurements come
   from the same physical die. If they do not, die-to-die variation enters the
   RT-to-4 K pair. The RT and 4 K files of every device match by module/pad ID
   and W/L; no other check of die identity is available.
4. RT data limits: 50 mV steps and a measurement floor near 1e-9 A, so C0_RT
   is not constrained in deep subthreshold, where the 4 K curves are
   included down to 1e-10 A.
5. BSIM4 4 K behavior: subthreshold current at the gmin floor up to
   threshold at -269.15 C; arm (a) will be limited by this on the Id-Vg
   curves (stock-card example: nMOS 0.377, pMOS 0.549).
6. Parameter non-identifiability in independent 43-parameter fits (risk 2 of
   the open items above).
7. Few included curves on the long narrow pMOS devices (2-3 of 11 at 4 K),
   so their RRMS rests on 2-3 curves.
8. Code: the I9-1 card handling supports nfet_01v8 and pfet_01v8 only; the
   lvt, hvt and g5v0d10v5 cards need the file mapping and, for g5v0d10v5, the
   bin stretch.
