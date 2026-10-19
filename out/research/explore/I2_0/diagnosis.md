# I2-0: devices that determined the exploration-iteration-1 means

Date 2026-10-19. Script: `scripts/research/explore/i2_0_diagnose.py` (stages `tables`, `sim`,
`decisions`, `plots`). Inputs are stored E1-E5 records, the published-baseline inclusion set,
`out/research/polished/direct43_r3_fd` and `direct7_fd`. The `sim` stage ran about 40
single-device NGSpice simulations; no queue job was running. No stored verdict or result file
was changed. REAL_TEST devices were not loaded. RRMS is the paper per-curve RRMS with frozen
published-baseline inclusion throughout; quantities labelled "scale" below are diagnostics and
do not replace it.

Files in this directory:

| file | content |
|---|---|
| `A_measured_curves.json` | every REAL_DEV metric curve: inclusion, trim start, scored points, denominator, quantization step, RRMS of a 0 A prediction, published-card RRMS |
| `A_e2_per_curve.json` | E2 raw/FD per-curve RRMS for the two devices, all arms and seeds, absolute RMSE and RMSE / largest device denominator |
| `A_e2_box_position.json` | E2 raw predictions in local box coordinates |
| `A_e4_per_curve.json` | E4 p43/p7, vd/sparse/vg, held-out per-curve RRMS per method |
| `A_e4_laplace_draws.json` | per-draw held-out RRMS of the stored Laplace draws (NGSpice curves) |
| `A_e4_learned_target_scale.json` | E4 learned-discrepancy targets: prediction scale vs denominator, mean y^2 per device |
| `A_e5_per_device.json` | E5 A0 and A2 per device |
| `sim_checks.json` | NGSpice re-simulations: part A cards and VTH0 shifts, E5 A2 per curve; part B cards at 110 K, u0 sweep, population scan |
| `C_decision_sensitivity.json` | iteration-1 decision quantities with and without the devices (diagnostic only) |
| `proposed_e3_u0_switch.patch` | proposed fix for `src/cryoml/explore/e3_temp.py` (not applied) |
| `I2_0_A_pmos_L0p35_W5_iv.png`, `I2_0_A_pmos_L0p35_W0p55_iv.png` | measured points, NGSpice published card, NGSpice best stored 43-parameter card, NGSpice worst stored prediction |
| `I2_0_A_denominators.png` | per-curve denominator / device largest denominator, REAL_DEV |
| `I2_0_A_vth0_sensitivity.png` | per-curve RRMS vs VTH0 shift of the best 43-parameter card |
| `I2_0_B_nmos_L1_W3__2.png` | E3 truth u0 vs T, truth current scale vs T, u0 sweep at 110 K, 110 K curves |

## A. pmos_L0p35_W5 and pmos_L0p35_W0p55 (REAL_DEV)

### Measured data and denominators

Denominator range (largest / smallest denominator over the frozen-included curves of a device):
pmos_L0p35_W0p55 8409, pmos_L0p35_W5 5106; the other ten REAL_DEV devices 37-342.

| device | curve | den (A) | den / device max | scored points | distinct values | step / den | RRMS of 0 A | published card | best 43 card |
|---|---|---|---|---|---|---|---|---|---|
| pmos_L0p35_W5 | idvd@0.74 | 1.23e-7 | 2.0e-4 | 42 (\|Vd\| 1.43-1.85 V) | 33 | 0.008 | 1.007 | 0.999 | 0.028 |
| pmos_L0p35_W5 | idvg@0.01 | 5.68e-6 | 9.0e-3 | 6 | 6 | 0.010 | 1.001 | 0.994 | 0.454 |
| pmos_L0p35_W5 | idvd@1.11 | 1.57e-5 | 0.025 | 185 | 184 | 0.001 | 1.186 | 0.466 | 0.147 |
| pmos_L0p35_W0p55 | idvd@0.37 | 4.25e-9 | 1.2e-4 | 159 | 8 | 0.235 | 1.112 | 1.007 | 1.093 |
| pmos_L0p35_W0p55 | idvd@0.74 | 1.79e-8 | 5.0e-4 | 137 | 3 | 0.559 | 1.097 | 1.275 | 0.173 |
| pmos_L0p35_W0p55 | idvg@0.01 | 8.72e-8 | 2.4e-3 | 95 | 69 | 0.011 | 1.189 | 4.756 | 1.489 |

pmos_L0p35_W5 idvd@0.37 is identically 0 A in the measured file and is not included.
The pmos_L0p35_W0p55 file `idvd_Vg-0p74.csv` stores Id in 10 nA steps (values 0, 1e-8, 2e-8,
3e-8 A); `idvd_Vg-0p37.csv` stores 1 nA steps (0 to 8e-9 A). These two curves are at the
measurement resolution. The pmos_L0p35_W5 low-current curve (idvd@0.74) has 1 nA resolution and
33 distinct values; its data are usable, its scored segment is 42 points after the 1e-7 A
start-trim floor.

A card that predicts 0 A on a curve gets RRMS 1.0-1.2 on that curve; a card that over-predicts
has no upper bound. Every stored card in the record is at the 0 A level on pmos_L0p35_W0p55
idvd@0.37 (published 1.007, best 43-parameter 1.093).

### Per-curve breakdown of the failing predictions

| record | device | device RRMS | dominating curve (RRMS) | share of curve sum | RMSE / device largest den on that curve | same on the high-current curves |
|---|---|---|---|---|---|---|
| E2 raw S seed 1 | W5 | 183.7 | idvd@0.74 (1790) | 0.975 | 0.35 | 0.37-1.66 |
| E2 raw S seed 1 | W0p55 | 239.7 | idvd@0.74 (1682), idvd@0.37 (814) | 0.64 | 0.84 | 1.4-4.7 |
| E4 p43 vd Laplace | W5 | 5983 | idvd@0.74 (17949) | 1.00 | 3.5 | 0.011-0.015 |
| E4 p7 sparse single | W5 | 96.0 | idvd@0.74 (662) | 0.99 | 0.13 | 0.09-0.24 |
| E4 p7 vd single | W0p55 | 7.01 | idvd@0.74 (18.7), idvd@0.37 (8.3) | 0.96 | - | - |

E2 raw: in 9 of 9 pmos_L0p35_W5 records the top curve is idvd@0.74 (share 0.79-0.98). On
pmos_L0p35_W0p55 the top curve is idvd@0.74 or idvd@0.37 in 7 of 9 records. The predicted
nfactor is at local coordinate 0.61-1.00 in 17 of 18 records (box upper edge = 100 x published,
40.7 for W5). The worst cards over-predict every curve: on W5 the absolute error on idvd@0.74 is
0.35 of the device's largest denominator and 0.37-1.66 on the high-current curves. The metric
divides the low-current curve by a denominator 5000x smaller, so that curve gives 97.5 % of the
curve sum.

E4 p43 vd Laplace, W5 idvd@0.74: the point predictor is the mean of 64 NGSpice draw curves. Draw
RRMS median 0.97, 90th percentile 1.01, maximum 1.15e6 (2 of 64 above 10); the draw mean has
RRMS 17949. The posterior Hessian has condition number 1.4e16 and effective rank 5 of 39 active
parameters. The single card on the same curve: 0.873.

E4 p7 sparse single, W5: all 64 draws give 600-690 on idvd@0.74. The seven-parameter card is
fitted without the low-|Vg| Id-Vd curves and predicts about 8e-5 A at |Vg| = 0.74 V where the
measurement is 1e-7 A.

E4 learned inner leave-one-device-out MSE: the learned target is y = (I_meas - I_single) / sc
with sc = mean |single prediction| over the curve. On p43 vd the pmos_L0p35_W0p55 single card
predicts sc = 1.0e-12 A on idvd@0.74 (denominator 1.8e-8 A), so max |y| = 3.0e4 and the device
mean of y^2 is 8.3e7; averaged over 11 devices this is the 7.5e6 inner MSE of every fold that
trains on it. On p7 vd the pmos_L0p35_W5 single card gives sc = 1.1e-12 A on idvd@0.74, device
mean y^2 1.27e9, inner MSE 1.16e8. On every other REAL_DEV device and split the device mean y^2
is 0.002-5.8.

E5 A2 (fold model applied to the best 43-parameter card, NGSpice base simulation): W5 0.112 ->
0.235, largest per-curve increase idvd@0.74 0.028 -> 0.682; W0p55 0.409 -> 0.308 (lower). The
E5 means are not determined by these two devices: the largest A2 increases are nmos_L20_W0p64
0.049 -> 0.551 and nmos_L100_W100 0.073 -> 0.276, and the E5 correction is bounded
(tanh, |delta| <= 1).

VTH0 check on the best 43-parameter card (`I2_0_A_vth0_sensitivity.png`): |VTH0| -40/+40 mV moves
W5 idvd@0.74 from 0.028 to 0.78/0.44 and the device mean from 0.112 to 0.31/0.24; W0p55 device
mean 0.409 to 0.51/0.49. Near a fitted card the response is moderate; values of 100-1e4 need
over-prediction by two to five decades of current on the low-current curves.

### Finding A

Cause: the two devices are the only REAL_DEV devices whose frozen inclusion set contains curves
with denominators 1e-4 to 5e-4 of the device's largest. The per-curve normalization multiplies
an absolute current error on those curves by 5000-8400 relative to the device's main curves, and
over-prediction is unbounded while under-prediction stops near 1.1. The failing predictions
over-predict subthreshold current (E2: nfactor at the box upper edge; E4 Laplace: a few draws
from a singular posterior inside a mean; E4 p7 sparse: extrapolation of a card fitted without
those curves; E4 learned: target normalized by a 1e-12 A predicted scale). Two of the
pmos_L0p35_W0p55 curves (idvd@0.37, idvd@0.74) are at the SMU resolution (8 and 3 distinct
values); quantized curves also occur on nmos_L1_W3, pmos_L4_W7 and pmos_L8_W1p6 (6-9 distinct
values) without large RRMS, so quantization alone does not produce the large values; the
denominator ratio does. No pipeline bug was found for part A: re-simulation reproduces every stored value
(published 0.3291, best 43 0.11218, E2 worst 183.67 and 239.72, E5 A2 0.23459 and 0.30825).

## B. E3 tier-M test device nmos_L1_W3__2

Truth u0 (card value, stock u0 = 0.02587, switch at x_u0 = log(1/0.02587) = 3.655):
1.284 at 77 K, 1.186 at 100 K, 1.100 at 110 K, 0.849 at 130 K, 0.543 at 150 K. BSIM4 reads
u0 > 1 as cm^2/Vs and divides it by 1e4. The stored truth currents have maximum |Id| 4.1e-6 A
at 77 K and 3.5e-6 A at 110 K, and 1.1e-3 A at 150 K.

| card at 110 K | u0 | above switch | RRMS vs 110 K truth |
|---|---|---|---|
| O4 lrn | 0.833 | no | 1030.4 |
| O4 par3 | 0.569 | no | 940.5 |
| O4 pt_interp | 0.832 | no | 1053.7 |
| O2 lrn | 0.567 | no | 715.9 |
| O2 par3_pop | 0.489 | no | 675.9 |
| floor (fit at 110 K) | 1.153 | yes | 0.0185 (vsat and eta0 at the fit bounds) |
| truth x without series R | 1.100 | yes | 3.5e-5 |
| O4 lrn with only x_u0 set to the truth value | 1.100 | yes | 0.156 |

u0 sweep (truth x at 110 K, only x_u0 varied): x_u0 3.50-3.65 gives RRMS 906-920 and maximum
current 1.09e-3 A; x_u0 3.66 gives 0.114 and 3.2e-6 A. The 77 K anchor fit is on the switched
branch (x_u0 3.98) and the 150 K anchor fit is on the normal branch (x_u0 2.85); every method
predicts a 110 K card below the switch.

Items checked and excluded: 110 K is inside the anchor range (77-150 K), so there is no
temperature extrapolation; all cells are finite (0 failed cells); the truth parameters are
inside the box; re-simulating the truth reproduces the stored target (max relative difference
0); the denominators follow the truth currents (1.4e-8 to 2.3e-6 A at 110 K) and the inclusion
set is the frozen one.

Population: tier-M truths with u0 > 1 at some dense temperature: train nmos_L1_W3__1 (77-170 K),
train nmos_L1_W3__2 (77 K), train nmos_L1_W3__3 (77 K), test nmos_L1_W3__2 (77-110 K). Tier W:
train nmos_L1_W3__2 (77-100 K), test nmos_L1_W3__0 (77 K only). The fit bound for x_u0 (4.874)
is above the switch (3.655), so fitted cards can also land on either branch. The 1.5x box upper
bound for nmos_L0p25_W1p6 (3.713) is also above its switch (3.658); no drawn truth crossed there.

### Finding B

Cause: pipeline issue in the E3 world generator. `e3_temp.draw_truth` draws truths whose u0
crosses the BSIM4 u0 > 1 unit switch, so the nmos_L1_W3__2 truth current drops by about 300x
between 150 K and 110 K. A law smooth in x cannot follow the switch, and every method's 110 K
card is on the other branch, so each gives RRMS 676-1060 in that cell. The same guard exists in
`scripts/research/make_sky130_misspec.py` (rejection reason `u0_unit_switch`) but not in
`e3_temp.py`. Proposed patch: `proposed_e3_u0_switch.patch` (rejection sampling in
`draw_truth`, u0 fit bound kept 0.02 below the switch in `x_bounds`); not applied. It changes
which truths are drawn, so E3 results generated with it are a new run.

## C. Sensitivity of the iteration-1 decisions (diagnostic only)

Values recomputed from stored records with devices removed; not decision inputs. E2 and E4 are
exact (E2 real devices are evaluation-only; E4 learned/conformal leave-one-device-out recomputed
on the reduced set). E1 and E5 are aggregate-only (fold models were trained with the removed
devices). E3 removes the test device; its training population still contains switched truths.

| run | all devices | without W5 | without W0p55 | without both |
|---|---|---|---|---|
| E1 stage 1 (M, T1, C0; pass) | 0.128, 0.086, 0.107; no | 0.133, 0.088, 0.107; no | 0.108, 0.063, 0.080; no | 0.111, 0.062, 0.077; no |
| E1 stage 2 (M, T1, C0; pass) | 0.087, 0.082, 0.107; no | 0.091, 0.083, 0.107; no | 0.066, 0.058, 0.080; no | 0.068, 0.057, 0.077; no |
| E2 R_raw(N0), R_raw(S), S/N0 | 11.24, 19.58, 1.742 | 8.03, 10.67, 1.329 | 8.92, 13.20, 1.479 | 5.16, 2.76, 0.534 |
| E2 gap ratio, S wins, C4 | 1.754, 6/12, yes | 1.336, 6/11, no | 1.487, 6/11, yes | 0.522, 6/10, no |
| E2 rule output | KILL | KILL | KILL | INCONCLUSIVE (C1, C2 yes; C3 6/10 < 7; C4 no) |
| E4 p43 vd single, Laplace, learned, crossinterp | 0.418, 500.2, 10.87, 0.098 | 0.419, 1.75, 2.78, 0.104 | 0.396, 545.2, 0.138, 0.083 | 0.395, 1.44, 0.092, 0.089 |
| E4 p43 vd verdict | PARTIAL_CALIBRATION_ONLY | same | same (learned C2 yes 8/11, C1 no cov 0.991, C3 no) | same (learned C2 yes 9/10, C1 no cov 1.000, C3 no) |
| E5 b43 A0, A2, reduction, A2 < A0 | 0.107, 0.200, -0.868, 1/12 | 0.107, 0.197, -0.847, 1/11 | 0.080, 0.191, -1.387, 0/11 | 0.077, 0.186, -1.430, 0/10 |

E2 C4 (tier-a guard) passes on all devices only because the tier-a N0 mean includes the simulated
pmos_L0p35_W5 target at 192.7; removing that geometry's simulated targets gives tier-a S 1.548
vs N0 0.725 and C4 fails.

| E3 (tier M) | all 36 | without nmos_L1_W3__2 (35) | without all 3 nmos_L1_W3 tests (33) |
|---|---|---|---|
| O4: lrn, par3, pt_interp | 9.801, 9.123, 10.118 | 0.267, 0.423, 0.370 | 0.266, 0.437, 0.374 |
| O4: R_O4 (95 % CI), lrn vs pt_interp | -0.074 (-0.090, 0.504), 0.031 | 0.369 (0.168, 0.533), 0.278 | 0.392 (0.183, 0.556), 0.290 |
| O2: lrn, par3_pop, pt_interp | 7.102, 6.858, 9.592 | 0.483, 0.610, 0.964 | 0.481, 0.623, 0.969 |
| O2: R_O2 (95 % CI), lrn vs pt_interp | -0.036 (-0.055, 0.288), 0.260 | 0.208 (0.031, 0.305), 0.499 | 0.227 (0.057, 0.322), 0.503 |
| rule output (failed-cell fraction 0 in the stored run) | KILL | CONTINUE (O4 meets all conditions) | CONTINUE (O4) |

Sensitivity summary: E1, E4 and E5 rule outputs do not change. E2 changes from KILL to
INCONCLUSIVE when both devices are removed (and from both its real-device mean and its tier-a
guard). E3 changes from KILL to CONTINUE when nmos_L1_W3__2 is removed; that device's values come
from the u0 switch (finding B).

## Proposals for the next pre-registrations (not run)

1. Bug fix before any E3 rerun: apply `proposed_e3_u0_switch.patch` (or an equivalent guard),
   regenerate tier M and W truths and training fits, and pre-register E3b with the E3 decision
   rule unchanged plus an assertion that 0 truths and 0 fitted cards are above the u0 switch.
   Apply the same check to every generator or fit box that contains u0 (temperature bank,
   cold-loop boxes; nmos_L1_W3 and nmos_L0p25_W1p6 boxes reach the switch).
2. Aggregate declared in advance: keep the paper per-device RRMS unchanged and report the
   all-device mean and combined mean as now; base the continue/kill rule on a paired statistic
   that one device cannot determine, for example the geometric mean over devices of
   RRMS(method) / RRMS(control) with a paired bootstrap CI, together with the win count. Name it
   in the pre-registration before any run.
3. Data-quality rule fixed from measured data only, before any model is scored: flag an included
   curve when its scored segment has fewer than 10 distinct current values or quantization step /
   denominator > 0.1. On REAL_DEV this flags 7 curves: pmos_L0p35_W0p55 idvd@0.37 (8 values) and
   idvd@0.74 (3), pmos_L0p35_W5 idvg@0.01 (6 scored points), nmos_L1_W3 idvd@0.74 (6),
   nmos_L100_W100 idvd@0.74 (step/den 0.16), pmos_L4_W7 idvd@1.11 (9), pmos_L8_W1p6 idvd@1.11 (8).
   A second rule on the denominator ratio (curve den < 1e-3 x device largest den) flags only
   pmos_L0p35_W5 idvd@0.74 and pmos_L0p35_W0p55 idvd@0.37 and idvd@0.74. Primary scores
   keep the frozen inclusion; a secondary score without flagged curves is reported for every arm.
   Also report per run the share of each device's curve sum that comes from curves with
   denominator below 1e-3 of the device's largest.
4. Method items for any continuation of E4: Laplace point predictor = MAP card or median of the
   draw curves instead of the mean; the learned target normalized by a device-level scale (mean
   |single prediction| over the device's train curves) instead of the per-curve predicted scale;
   draws with any current above the measured device maximum by more than 10x reported as a count.
5. For one-pass networks (E2-type): report the fraction of predicted parameters with
   |local| >= 0.98 and the device RRMS with and without the curves of point 3, as declared
   secondary quantities.
