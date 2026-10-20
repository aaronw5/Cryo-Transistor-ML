# Tables, 2026-10-20 i4

Generated 2026-10-20 07:40 by `PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i4.py --date 2026-10-20` from stored files only. RRMS is the paper metric with frozen curve inclusion, strict. REAL_DEV values are on the 12 measured REAL_DEV devices (development evidence); GF180/IHP values are on simulated tier c targets.

### I4-1 decision: pre-registered baseline conditions and rule output

| baseline X | definition | REAL_DEV GMR(M_bias, X) [95% CI] | (a) CI upper < 1 | GF180 GMR(M_bias, X) | IHP GMR(M_bias, X) | (b) both < 0.95 | outcome |
|---|---|---:|---:|---:|---:|---:|---:|
| B1 | wider card parameter set, no correction | 0.770 [0.667, 0.887] | met | 0.897 | 1.644 | not met | beaten |
| B2 | extrinsic series resistance + self-heating factor, no learned term | 0.773 [0.670, 0.891] | met | 1.333 | 1.806 | not met | beaten |
| B3_d2 | shared polynomial g, total degree 2 (12 coefficients) | 0.789 [0.684, 0.907] | met | 0.994 | 0.986 | not met | beaten |
| B3_d3 | shared polynomial g, total degree 3 (20 coefficients) | 0.844 [0.751, 0.949] | met | 0.997 | 0.985 | not met | beaten |
| B4 | shared GP discrepancy (posterior mean) | 0.658 [0.581, 0.753] | met | 0.954 | 0.848 | not met | beaten |

- Pre-registration out/research/explore/I4_1/prereg.json (commit e57afce); amendment prereg_amendment_2026-10-20.json (REAL_DEV base-box TRF budget per IRLS round is 4 or 5, not 4; rule unchanged). Rule: I3-2 is 'supported against baselines' if every X in {B1, B2, B3_d2, B3_d3, B4} is beaten; otherwise 'matched by baseline X'.
- Rule output: supported against baselines (out/research/explore/I4_1/decision.json; recomputed here from the stored per-device values). Every baseline is beaten through condition (a) on REAL_DEV; condition (b) on the simulated tiers is not met for any baseline.
- GMR(M_bias, X) = exp(mean ln(RRMS_M_bias / RRMS_X)); < 1: M_bias has the lower held-out RRMS. REAL_DEV: 12 devices, device bootstrap; GF180/IHP: 48 targets, point estimate (CI in the next tables). 10000 resamples, default_rng(0).
- REAL_DEV values are development evidence: REAL_DEV was used by E1, I2-1, I3-2 and its follow-ups. REAL_TEST is not loaded by any I4-1 script.


### I4-1 REAL_DEV results for every arm, with GMR against M_bias and against C0_refit

| arm | definition | all-device | combined | nMOS | pMOS | median | failed | GMR(M_bias, X) [95% CI] | devices with M_bias < X | GMR(X, C0_refit) [95% CI] | devices with X < C0_refit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0_refit | B0: held-out card refit, no correction (stored I2-1 rows) | 0.1072 | 0.0981 | 0.0432 | 0.1529 | 0.0820 | 0/12 | 0.769 [0.666, 0.886] | 11/12 |  |  |
| C0_refit_rerun | C0 refit through the I4-1 refit path (reproduction check) | 0.1072 | 0.0981 | 0.0432 | 0.1529 | 0.0820 | 0/12 | 0.769 [0.666, 0.886] | 11/12 |  |  |
| B1 | wider card parameter set, no correction | 0.1069 | 0.0978 | 0.0433 | 0.1523 | 0.0822 | 0/12 | 0.770 [0.667, 0.887] | 11/12 | 0.998 [0.994, 1.001] | 5/12 |
| B2 | extrinsic series resistance + self-heating factor, no learned term | 0.1063 | 0.0973 | 0.0431 | 0.1515 | 0.0819 | 0/12 | 0.773 [0.670, 0.891] | 11/12 | 0.994 [0.987, 0.999] | 8/12 |
| B3_d2 | shared polynomial g, total degree 2 (12 coefficients) | 0.1034 | 0.0948 | 0.0430 | 0.1465 | 0.0820 | 0/12 | 0.789 [0.684, 0.907] | 11/12 | 0.975 [0.948, 0.995] | 8/12 |
| B3_d3 | shared polynomial g, total degree 3 (20 coefficients) | 0.0956 | 0.0879 | 0.0419 | 0.1339 | 0.0748 | 0/12 | 0.844 [0.751, 0.949] | 11/12 | 0.911 [0.874, 0.948] | 11/12 |
| B4 | shared GP discrepancy (posterior mean) | 0.1317 | 0.1199 | 0.0492 | 0.1906 | 0.0938 | 0/12 | 0.658 [0.581, 0.753] | 12/12 | 1.168 [1.005, 1.367] | 5/12 |
| M_bias | I3-2 correction (stored rows; method under test) | 0.0806 | 0.0749 | 0.0407 | 0.1090 | 0.0605 | 0/12 |  |  | 0.769 [0.666, 0.886] | 11/12 |
| T1_inloop | per-(family, curve tag) gain table (not deployable; stored) | 0.0820 | 0.0760 | 0.0403 | 0.1117 | 0.0650 | 0/12 | 0.968 [0.920, 1.011] | 7/12 | 0.794 [0.707, 0.889] | 11/12 |
| B5 | per-device correction trained on the held-out device (in-sample reference) | 0.0410 | 0.0386 | 0.0243 | 0.0530 | 0.0315 | 0/12 | 1.860 [1.556, 2.262] | 0/12 | 0.413 [0.319, 0.525] | 12/12 |

- REAL_DEV leave-one-device-out, 12 devices (5 nMOS, 7 pMOS); strict paper RRMS with frozen inclusion after the held-out 43-parameter card refit (from the direct43_r3_fd card; irls 2, diff_step 0.01; base-box arms max_nfev 400). all-device = mean over 12; combined = mean of the nMOS and pMOS family means.
- B1 and B2 are local refits from the same starting card as C0_refit, with more active parameters (B1: base box + 19 extras; B2: base box + 2); their max_nfev is scaled with the parameter count so that the TRF evaluations per IRLS round equal the base-box value of the same device (i4_1_baselines.matched_nfev).
- C0_refit, M_bias and T1_inloop are stored I3-2 rows; C0_refit_rerun reproduces C0_refit with max |difference| 0.0. B5 is trained on the held-out device itself and is an in-sample reference, not a held-out method.
- Queue job I4_1_real 2026-10-20 06:14-06:22 (485 s; training 61 s, 84 held-out refits 424 s).


### I4-1 simulated GF180 and IHP tier c results for every arm

| dataset | arm | all-target mean | median | failed | GMR(M_bias, X) [95% CI] | targets with M_bias < X | GMR(X, C0_refit) [95% CI] | targets with X < C0_refit |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GF180 tier c | C0_refit | 0.0457 | 0.0352 | 0/48 | 0.880 [0.837, 0.921] | 42/48 |  |  |
| GF180 tier c | C0_refit_rerun | 0.0457 | 0.0352 | 0/48 | 0.880 [0.837, 0.921] | 42/48 |  |  |
| GF180 tier c | B1 | 0.0454 | 0.0331 | 0/48 | 0.897 [0.818, 0.991] | 39/48 | 0.982 [0.904, 1.038] | 14/48 |
| GF180 tier c | B2 | 0.0272 | 0.0257 | 0/48 | 1.333 [1.150, 1.627] | 10/48 | 0.661 [0.537, 0.779] | 46/48 |
| GF180 tier c | B3_d2 | 0.0400 | 0.0301 | 0/48 | 0.994 [0.989, 1.000] | 32/48 | 0.885 [0.842, 0.926] | 40/48 |
| GF180 tier c | B3_d3 | 0.0400 | 0.0300 | 0/48 | 0.997 [0.992, 1.002] | 30/48 | 0.883 [0.837, 0.926] | 38/48 |
| GF180 tier c | B4 | 0.0407 | 0.0332 | 0/48 | 0.954 [0.919, 0.996] | 36/48 | 0.923 [0.870, 0.978] | 33/48 |
| GF180 tier c | M_bias | 0.0399 | 0.0299 | 0/48 |  |  | 0.880 [0.837, 0.921] | 42/48 |
| GF180 tier c | T1_inloop | 0.0391 | 0.0291 | 0/48 | 1.023 [0.982, 1.066] | 19/48 | 0.860 [0.830, 0.890] | 45/48 |
| GF180 tier c | B5 | 0.0239 | 0.0230 | 0/48 | 1.504 [1.289, 1.836] | 0/48 | 0.585 [0.476, 0.693] | 48/48 |
| IHP tier c | C0_refit | 0.1011 | 0.0567 | 0/48 | 0.885 [0.789, 1.005] | 32/48 |  |  |
| IHP tier c | C0_refit_rerun | 0.1011 | 0.0567 | 0/48 | 0.885 [0.789, 1.005] | 32/48 |  |  |
| IHP tier c | B1 | 0.0403 | 0.0376 | 0/48 | 1.644 [1.329, 2.071] | 10/48 | 0.538 [0.417, 0.690] | 48/48 |
| IHP tier c | B2 | 0.0381 | 0.0311 | 0/48 | 1.806 [1.513, 2.231] | 3/48 | 0.490 [0.391, 0.602] | 48/48 |
| IHP tier c | B3_d2 | 0.0881 | 0.0510 | 0/48 | 0.986 [0.969, 1.002] | 30/48 | 0.898 [0.810, 1.008] | 32/48 |
| IHP tier c | B3_d3 | 0.0874 | 0.0520 | 0/48 | 0.985 [0.971, 0.998] | 30/48 | 0.898 [0.798, 1.028] | 32/48 |
| IHP tier c | B4 | 0.0913 | 0.0647 | 0/48 | 0.848 [0.754, 0.954] | 29/48 | 1.044 [0.903, 1.227] | 22/48 |
| IHP tier c | M_bias | 0.0871 | 0.0510 | 0/48 |  |  | 0.885 [0.789, 1.005] | 32/48 |
| IHP tier c | T1_inloop | 0.0921 | 0.0491 | 0/48 | 0.950 [0.874, 1.036] | 29/48 | 0.932 [0.868, 1.011] | 35/48 |
| IHP tier c | B5 | 0.0274 | 0.0238 | 0/48 | 2.416 [1.924, 3.131] | 0/48 | 0.366 [0.275, 0.479] | 48/48 |

- Simulated tier c truth sets 0-2 (16 geometries x 3 = 48 targets per PDK), leave-one-geometry-out (16 folds); held-out card refit on the pdk2 wide box (GF180 21 parameters, IHP 12) from the E1 fitted card. GMR with geometry-cluster bootstrap 95% CI (16 clusters, 10000 resamples, default_rng(0)).
- B1 extras: GF180 35, IHP 71 (edge screen b1_screen.json). B1 and B2 max_nfev scaled with the parameter count (same TRF evaluations per IRLS round as the base box: GF180 8, IHP 14).
- The tier c truths contain an extrinsic series resistance of the B2 form and perturb parameters that B1 fits (next tables but one). Rule condition (b) requires GMR(M_bias, X) < 0.95 on both PDKs.
- Queue jobs I4_1_gf180 06:22-06:42 (1172 s), I4_1_ihp 06:42-07:16 (2064 s).


### I4-1 B1 and B2 against the simulated truth content (stated in the pre-registration) and the measured B1/B2 values

| dataset | truth-perturbed parameters in B1 | C0_refit | B1 | B2 | M_bias | B5 | GMR(B1, C0_refit) | GMR(B2, C0_refit) | GMR(M_bias, B1) | GMR(M_bias, B2) | B2 fitted R_ext / box upper, median [min, max] | B2 fitted kappa, median | B1 active extras per device |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GF180 tier c | 9/12 | 0.0457 | 0.0454 | 0.0272 | 0.0399 | 0.0239 | 0.982 | 0.661 | 0.897 | 1.333 | 0.11 [0.00, 0.73] | -0.041 | 32-33 |
| IHP tier c | 9/9 | 0.1011 | 0.0403 | 0.0381 | 0.0871 | 0.0274 | 0.538 | 0.490 | 1.644 | 1.806 | 0.28 [0.00, 1.00] | -0.050 | 66-69 |
| REAL_DEV (no known truth) | — | 0.1072 | 0.1069 | 0.1063 | 0.0806 | 0.0410 | 0.998 | 0.994 | 0.770 | 0.773 | 0.00 [0.00, 0.03] | -0.000 | 19-19 |

- As stated in the pre-registration before the run (arms.B1.overlap_with_simulated_truth, arms.B2.note): the GF180/IHP tier c truths perturb non-fitted parameters by exp(U(-0.25, 0.25)); GF180 B1 contains 9 of the 12 (a0, ags, lpe0, minv, drout, pscbe1, pscbe2, alpha0, beta0; not keta, etab, eu) and IHP B1 contains all 9 (mueo, cto, fbet1, lp1, alpl, alplexp, axl, ctlexp, nsubo). The truths also contain an extrinsic resistance per side k * R_side_stock / W, k ~ U(1, 2), which has the B2 form, and a per-curve temperature rise (27 C + dT_max * P_i / max P, dT_max ~ U(20, 40) K); the B2 heating term is pointwise and multiplicative, not the truth's per-curve temperature.
- Counts in the second column are recomputed from cryoml.pdk2.boxes.MISSPEC_PARAMS / IHP_MISSPEC_PARAMS and the stored b1_extras.
- B1 and B2 are local refits from the same starting card with max_nfev scaled to the parameter count; they are not global searches over the wider parameter set.


### I4-1 B5 in-sample bound and the fraction of the C0_refit -> B5 gap closed by M_bias

| dataset | C0_refit | M_bias | B5 (refit card, g1) | B5 starting card with g0 | B5 refit card with g0 | fraction of C0->B5 gap closed (means) | fraction of C0->B5 gap closed (log GMR) | GMR(M_bias, B5) [95% CI] | GMR(B5, C0_refit) [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| REAL_DEV | 0.1072 | 0.0806 | 0.0410 | 0.0410 | 0.0407 | 0.403 | 0.298 | 1.860 [1.556, 2.262] | 0.413 [0.319, 0.525] |
| GF180 tier c | 0.0457 | 0.0399 | 0.0239 | 0.0239 | 0.0239 | 0.266 | 0.238 | 1.504 [1.289, 1.836] | 0.585 [0.476, 0.693] |
| IHP tier c | 0.1011 | 0.0871 | 0.0274 | 0.0276 | 0.0274 | 0.191 | 0.121 | 2.416 [1.924, 3.131] | 0.366 [0.275, 0.479] |

- B5 (pre-registered, not in the rule): M_bias-form ensemble (3-16-16-1, 5 seeds) g0 trained on the held-out device's own starting-card record; held-out card refit with g0 in the loop; g1 = same recipe trained on the refit card's record; score = refit card x exp(g1). B5 uses the held-out device's measured curves to train g and is an in-sample bound for a bias-only correction of this form, not a held-out method.
- fraction of mean gap = (C0_refit - M_bias) / (C0_refit - B5) on all-device (REAL_DEV) or all-target (GF180/IHP) means; fraction of log GMR = ln GMR(M_bias, C0_refit) / ln GMR(B5, C0_refit).
- M_bias is above B5 on every device and target (0 wins on all three datasets).


### I4-2 g surface fold spread and correlation of g with physical signatures

| dataset | polarity | fold models | grid mean of fold-mean g | max abs fold-mean g | grid mean of fold std | max fold std | measured points | subthreshold mean g / r | saturation mean g / r | linear mean g / r | high_vds mean g / r | Pearson g vs starting-card log residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| REAL_DEV | nmos | 12 | -0.0069 | 0.078 | 0.0076 | 0.0351 | 7849 | -0.004 / -0.10 | +0.002 / +0.17 | -0.007 / -0.10 | -0.001 / +0.05 | +0.031 |
| REAL_DEV | pmos | 12 | -0.0124 | 0.321 | 0.0109 | 0.0422 | 8832 | -0.019 / -0.10 | +0.031 / +0.27 | -0.211 / -0.41 | +0.004 / +0.21 | +0.130 |
| GF180 tier c | nmos | 16 | -0.0015 | 0.064 | 0.0044 | 0.0151 | 44146 | +0.014 / +0.63 | -0.013 / -0.28 | -0.031 / -0.47 | -0.002 / +0.04 | +0.270 |
| GF180 tier c | pmos | 16 | -0.0026 | 0.051 | 0.0033 | 0.0150 | 39983 | +0.006 / +0.46 | -0.003 / -0.30 | -0.005 / -0.22 | -0.000 / -0.11 | +0.037 |
| IHP tier c | nmos | 16 | -0.0065 | 0.137 | 0.0031 | 0.0156 | 46052 | +0.014 / +0.52 | -0.016 / -0.18 | -0.066 / -0.53 | -0.001 / +0.10 | +0.039 |
| IHP tier c | pmos | 16 | -0.0326 | 0.162 | 0.0060 | 0.0296 | 39736 | -0.013 / +0.09 | -0.009 / +0.16 | -0.059 / -0.38 | -0.014 / +0.07 | -0.259 |

- I4-2 (pre-registered, reported, not deciding): stored I3-2 fold models M_bias|<fold>, no simulation. g surface: g of each fold model on (|Vgs|/VDD, |Vds|/VDD) in [0, 1]^2, 41 x 41; mean and std over folds (fold spread).
- Regions on each device's included trimmed measured points with the device's own held-out fold model: Vth by maximum-transconductance linear extrapolation on the smallest-|Vds| Id-Vg curve; subthreshold |Vgs| < Vth; saturation |Vgs| >= Vth and |Vds| >= |Vgs| - Vth; linear otherwise; high_vds |Vds|/VDD >= 0.6 (overlaps the others). r = point-biserial correlation of g with the region indicator (pooled points per polarity).
- Last column: Pearson correlation of g with ln|I_meas / I_start| (starting-card log residual) over the same points.


### I4-2 comparison of the M_bias g with the T1 gain table, overall, per curve kind and per curve tag

| dataset | curve tag | held-out device curves | mean T1 gain | mean of curve-mean g | Pearson r (T1, curve-mean g) | mean abs(T1 - g) |
|---|---|---:|---:|---:|---:|---:|
| REAL_DEV | all curves | 116 |  |  | 0.965 | 0.0178 |
| REAL_DEV | all idvd curves | 44 |  |  | 0.565 | 0.0098 |
| REAL_DEV | all idvg curves | 72 |  |  | 0.970 | 0.0227 |
| REAL_DEV | nmos|idvd@0.74 | 5 | +0.0033 | -0.0118 | -0.00 | 0.0152 |
| REAL_DEV | nmos|idvd@1.11 | 5 | -0.0097 | +0.0016 | 0.86 | 0.0113 |
| REAL_DEV | nmos|idvd@1.48 | 5 | +0.0046 | +0.0041 | 0.23 | 0.0014 |
| REAL_DEV | nmos|idvd@1.85 | 5 | -0.0076 | -0.0120 | 0.50 | 0.0044 |
| REAL_DEV | nmos|idvg@0.01 | 5 | +0.0456 | +0.0247 | 0.76 | 0.0208 |
| REAL_DEV | nmos|idvg@0.37 | 5 | -0.0276 | -0.0166 | 0.97 | 0.0110 |
| REAL_DEV | nmos|idvg@0.74 | 5 | +0.0110 | -0.0048 | 0.68 | 0.0158 |
| REAL_DEV | nmos|idvg@1.11 | 5 | +0.0017 | -0.0011 | -0.88 | 0.0037 |
| REAL_DEV | nmos|idvg@1.48 | 5 | -0.0011 | -0.0015 | 0.22 | 0.0010 |
| REAL_DEV | nmos|idvg@1.85 | 5 | -0.0023 | -0.0001 | -0.21 | 0.0026 |
| REAL_DEV | pmos|idvd@0.37 | 1 | +0.0000 | +0.0093 | — | 0.0093 |
| REAL_DEV | pmos|idvd@0.74 | 2 | -0.0021 | +0.0193 | — | 0.0214 |
| REAL_DEV | pmos|idvd@1.11 | 7 | -0.0037 | -0.0149 | 0.07 | 0.0112 |
| REAL_DEV | pmos|idvd@1.48 | 7 | +0.0045 | -0.0052 | 0.83 | 0.0096 |
| REAL_DEV | pmos|idvd@1.85 | 7 | +0.0256 | +0.0155 | 0.41 | 0.0101 |
| REAL_DEV | pmos|idvg@0.01 | 7 | -0.3119 | -0.2471 | 0.12 | 0.0648 |
| REAL_DEV | pmos|idvg@0.37 | 7 | +0.1467 | +0.0634 | 0.51 | 0.0834 |
| REAL_DEV | pmos|idvg@0.74 | 7 | +0.0070 | -0.0125 | 0.21 | 0.0195 |
| REAL_DEV | pmos|idvg@1.11 | 7 | -0.0003 | +0.0037 | 0.34 | 0.0061 |
| REAL_DEV | pmos|idvg@1.48 | 7 | -0.0008 | +0.0118 | 0.34 | 0.0126 |
| REAL_DEV | pmos|idvg@1.85 | 7 | -0.0045 | -0.0115 | 0.34 | 0.0074 |
| GF180 tier c | all curves | 486 |  |  | 0.871 | 0.0120 |
| GF180 tier c | all idvd curves | 201 |  |  | 0.966 | 0.0104 |
| GF180 tier c | all idvg curves | 285 |  |  | 0.660 | 0.0130 |
| GF180 tier c | nmos|idvd@0.37 | 11 | +0.0103 | +0.0163 | 0.99 | 0.0059 |
| GF180 tier c | nmos|idvd@0.74 | 24 | +0.0316 | +0.0233 | 0.85 | 0.0083 |
| GF180 tier c | nmos|idvd@1.11 | 24 | +0.0165 | +0.0063 | 0.95 | 0.0103 |
| GF180 tier c | nmos|idvd@1.48 | 24 | -0.0234 | -0.0200 | 0.81 | 0.0034 |
| GF180 tier c | nmos|idvd@1.85 | 24 | -0.0700 | -0.0457 | 0.46 | 0.0243 |
| GF180 tier c | nmos|idvg@0.01 | 24 | -0.0279 | -0.0108 | 0.96 | 0.0171 |
| GF180 tier c | nmos|idvg@0.37 | 24 | -0.0030 | -0.0017 | 0.86 | 0.0048 |
| GF180 tier c | nmos|idvg@0.74 | 24 | -0.0026 | +0.0046 | 0.34 | 0.0073 |
| GF180 tier c | nmos|idvg@1.11 | 24 | -0.0117 | +0.0058 | 0.34 | 0.0176 |
| GF180 tier c | nmos|idvg@1.48 | 24 | -0.0306 | +0.0014 | 0.24 | 0.0321 |
| GF180 tier c | nmos|idvg@1.85 | 24 | -0.0421 | -0.0079 | -0.12 | 0.0342 |
| GF180 tier c | pmos|idvd@0.37 | 1 | +0.0000 | -0.0072 | — | 0.0072 |
| GF180 tier c | pmos|idvd@0.74 | 21 | +0.0066 | +0.0113 | 0.97 | 0.0047 |
| GF180 tier c | pmos|idvd@1.11 | 24 | +0.0146 | +0.0081 | 0.65 | 0.0065 |
| GF180 tier c | pmos|idvd@1.48 | 24 | -0.0084 | -0.0014 | 0.81 | 0.0070 |
| GF180 tier c | pmos|idvd@1.85 | 24 | -0.0351 | -0.0149 | 0.84 | 0.0202 |
| GF180 tier c | pmos|idvg@0.01 | 21 | +0.0011 | +0.0000 | 0.53 | 0.0016 |
| GF180 tier c | pmos|idvg@0.37 | 24 | +0.0083 | +0.0022 | -0.12 | 0.0060 |
| GF180 tier c | pmos|idvg@0.74 | 24 | +0.0042 | +0.0032 | 0.11 | 0.0014 |
| GF180 tier c | pmos|idvg@1.11 | 24 | -0.0005 | +0.0028 | -0.07 | 0.0033 |
| GF180 tier c | pmos|idvg@1.48 | 24 | -0.0090 | +0.0011 | -0.34 | 0.0101 |
| GF180 tier c | pmos|idvg@1.85 | 24 | -0.0221 | -0.0024 | 0.41 | 0.0197 |
| IHP tier c | all curves | 478 |  |  | 0.842 | 0.0127 |
| IHP tier c | all idvd curves | 193 |  |  | 0.983 | 0.0062 |
| IHP tier c | all idvg curves | 285 |  |  | 0.435 | 0.0171 |
| IHP tier c | nmos|idvd@0.37 | 11 | -0.0036 | -0.0038 | 0.94 | 0.0008 |
| IHP tier c | nmos|idvd@0.74 | 24 | +0.0445 | +0.0433 | 0.86 | 0.0014 |
| IHP tier c | nmos|idvd@1.11 | 24 | +0.0286 | +0.0231 | 0.95 | 0.0055 |
| IHP tier c | nmos|idvd@1.48 | 24 | -0.0158 | -0.0219 | 0.47 | 0.0061 |
| IHP tier c | nmos|idvd@1.85 | 24 | -0.0784 | -0.0662 | 0.52 | 0.0122 |
| IHP tier c | nmos|idvg@0.01 | 24 | -0.0473 | -0.0203 | 0.96 | 0.0270 |
| IHP tier c | nmos|idvg@0.37 | 24 | -0.0190 | -0.0064 | 0.78 | 0.0126 |
| IHP tier c | nmos|idvg@0.74 | 24 | -0.0018 | +0.0004 | 0.55 | 0.0025 |
| IHP tier c | nmos|idvg@1.11 | 24 | -0.0141 | +0.0002 | 0.16 | 0.0143 |
| IHP tier c | nmos|idvg@1.48 | 24 | -0.0308 | -0.0017 | 0.15 | 0.0291 |
| IHP tier c | nmos|idvg@1.85 | 24 | -0.0464 | -0.0018 | 0.72 | 0.0447 |
| IHP tier c | pmos|idvd@0.74 | 15 | -0.0074 | +0.0038 | 0.80 | 0.0112 |
| IHP tier c | pmos|idvd@1.11 | 23 | +0.0295 | +0.0244 | 0.99 | 0.0051 |
| IHP tier c | pmos|idvd@1.48 | 24 | -0.0079 | -0.0113 | 0.88 | 0.0044 |
| IHP tier c | pmos|idvd@1.85 | 24 | -0.0529 | -0.0567 | 0.97 | 0.0076 |
| IHP tier c | pmos|idvg@0.01 | 21 | -0.0683 | -0.0334 | 0.91 | 0.0350 |
| IHP tier c | pmos|idvg@0.37 | 24 | -0.0144 | -0.0212 | 0.58 | 0.0083 |
| IHP tier c | pmos|idvg@0.74 | 24 | -0.0042 | -0.0160 | 0.03 | 0.0118 |
| IHP tier c | pmos|idvg@1.11 | 24 | -0.0121 | -0.0175 | 0.03 | 0.0067 |
| IHP tier c | pmos|idvg@1.48 | 24 | -0.0152 | -0.0196 | 0.22 | 0.0055 |
| IHP tier c | pmos|idvg@1.85 | 24 | -0.0295 | -0.0190 | 0.39 | 0.0105 |

- Per held-out device and included curve: the fold's E1 T1 gain for (family, curve tag) (out/research/explore/E1/{real_dev_lodo,<pdk>_c_logo}/fold_corrections.json) vs the mean of the fold's M_bias g over that curve's included trimmed points. 'all' rows: stored values (recomputed from t1_rows and checked); per-tag rows: computed here from the stored t1_rows (Pearson r across held-out devices, '—' for n < 3).


### I3-2 follow-up F1b and F1c: SPICE-native Verilog-A checks on ngspice-41 with OSDI

| check | comparison | F1b | F1c | pre-registered threshold |
|---|---|---|---|---:|
| F1.0 | export text: g from emitted Verilog-A lines vs Python; compiles | pass (5.3e-11; OpenVAF-Reloaded v24) | pass (5.3e-11; OpenVAF 23.5.0) | g diff <= 1e-9 |
| F1.1 | ngspice-41 OSDI build vs canonical ngspice-41, 4 uncorrected cards, 12 devices | pass (rel. 0.0, |dRRMS| 0.0) | pass (rel. 0.0, |dRRMS| 0.0; stored values diff 0.0) | rel. 1e-3, |dRRMS| 1e-4, stored 1e-9 |
| F1.2 | SPICE-native (OSDI build + .osdi) vs Python I_bsim exp(g), same simulator | fail (no .osdi loaded: OSDI 0.4 file, ngspice-41 accepts 0.3; 49104 non-finite currents) | fail (max rel. fold 2.45e-03, full 1.43e-03; |dRRMS| 1.24e-05; 0 convergence messages) | rel. 1e-3, |dRRMS| 1e-4 |
| F1.3 | SPICE-native per-device RRMS vs stored I3-2 M_bias | fail (NaN) | pass (max device 1.24e-05; all-device 5.3e-06) | device 2e-4, mean 1e-4 |
| diag. | reltol 1e-9, abstol 1e-18: SPICE-native vs Python | — | max rel. 1.4e-03; |dRRMS| 1.4e-09 | no pass/fail |

- Amendments followup_prereg_amendment_F1b_2026-10-20.json (commit db1064b) and followup_prereg_amendment_F1c_2026-10-20.json (commit e28cd74). F1b and F1c replace ngspice-46 (used in F1) with an ngspice-41 build from the same source with --enable-osdi (prefix ~/cryo-ng41/arm64-osdi); the canonical ngspice-41 (~/cryo-ng41/arm64) is the reference for the uncorrected runs.
- The OSDI build is bit-identical to the canonical build on the published card, 18 devices: 36828 of 36828 currents equal, max relative difference 0.0 (out/tables/simulator_compare_ng41_arm64_vs_arm64_osdi.json).
- F1.2 in F1c fails on the relative current criterion only: pmos_L0p35_W0p55 1.23e-03, pmos_L0p5_W0p42 2.45e-03, pmos_L8_W1p6 1.45e-03 exceed 1e-3; the other devices are at or below 1e-3. Relative differences use points with |I| >= 1e-20 A (amendment A1). Points with relative difference > 1e-3 (all simulated points, from f1c/curves.npz): pmos_L0p35_W0p55 18 points, |I| <= 6.5e-10 A; pmos_L0p5_W0p42 14 points, |I| <= 9.7e-10 A; pmos_L8_W1p6 84 points, |I| <= 1.5e-07 A. The largest drain currents of these devices are 1e-5 to 1e-4 A.
- F1b first run reported F1.3 pass with NaN deltas; the check code now counts a non-finite statistic as a failure and F1b was run again (stored summary is the second run).


### I3-2 follow-up F1c per device: SPICE-native M_bias on ngspice-41 (OSDI build) vs the stored M_bias

| device | published 77 K card (ngspice-41) | C0_refit card (ngspice-41) | stored M_bias (ngspice-41 x exp(g), Python) | SPICE-native fold (ngspice-41 OSDI) | native minus stored | SPICE-native full 12-device model | max rel. current diff, fold | max rel. current diff, full | max rel. current diff, fold, tight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nmos_L0p15_W1p6 | 0.0595 | 0.0193 | 0.0235 | 0.0235 | +9.7e-06 | 0.0199 | 9.0e-04 | 9.5e-04 | 1.0e-08 |
| nmos_L0p25_W1p6 | 0.0983 | 0.0188 | 0.0183 | 0.0183 | +9.0e-06 | 0.0179 | 9.9e-04 | 9.6e-04 | 8.1e-09 |
| nmos_L1_W3 | 0.1599 | 0.0561 | 0.0504 | 0.0504 | +1.3e-06 | 0.0479 | 8.7e-04 | 9.7e-04 | 1.2e-08 |
| nmos_L20_W0p64 | 0.1283 | 0.0489 | 0.0428 | 0.0428 | +6.3e-07 | 0.0393 | 7.9e-04 | 9.8e-04 | 9.2e-07 |
| nmos_L100_W100 | 0.1403 | 0.0731 | 0.0686 | 0.0686 | +2.5e-06 | 0.0669 | 1.0e-03 | 9.9e-04 | 3.5e-07 |
| pmos_L0p35_W0p55 | 0.7270 | 0.4089 | 0.3496 | 0.3496 | +6.9e-06 | 0.3420 | 1.2e-03 | 9.8e-04 | 2.1e-04 |
| pmos_L0p35_W5 | 0.3291 | 0.1122 | 0.0693 | 0.0693 | +1.4e-06 | 0.0640 | 9.9e-04 | 9.3e-04 | 1.4e-07 |
| pmos_L0p5_W0p42 | 0.4560 | 0.1615 | 0.1083 | 0.1083 | +1.2e-05 | 0.1023 | 2.4e-03 | 9.9e-04 | 6.9e-04 |
| pmos_L2_W5 | 0.1907 | 0.0796 | 0.0553 | 0.0553 | +3.9e-06 | 0.0498 | 9.7e-04 | 1.0e-03 | 7.6e-09 |
| pmos_L4_W7 | 0.2702 | 0.0844 | 0.0656 | 0.0657 | +7.6e-06 | 0.0583 | 9.3e-04 | 9.8e-04 | 2.2e-06 |
| pmos_L8_W1p6 | 0.5433 | 0.1321 | 0.0724 | 0.0724 | +2.4e-06 | 0.0719 | 1.4e-03 | 1.4e-03 | 1.4e-03 |
| pmos_L8_W5 | 0.3146 | 0.0914 | 0.0425 | 0.0425 | +5.5e-06 | 0.0412 | 9.5e-04 | 9.9e-04 | 4.5e-06 |
| all-device | 0.2848 | 0.1072 | 0.0806 | 0.0806 |  | 0.0768 |  |  |  |
| combined | 0.2608 | 0.0981 | 0.0749 | 0.0749 |  | 0.0713 |  |  |  |

- Strict paper RRMS with frozen inclusion on the 12 REAL_DEV devices; all-device = mean over 12, combined = mean of the nMOS (5) and pMOS (7) family means.
- Stored M_bias = I3-2 held-out value (ngspice-41, correction applied in Python). SPICE-native = M_bias refit card in the netlist with the compiled fold Verilog-A correction (trained on the other 11 devices), ngspice-41 OSDI build. The full 12-device model is in-sample on every device (diagnostic).
- The published-card and C0_refit-card values from the OSDI build equal the canonical ngspice-41 values (difference 0.0); the ngspice-46 differences on pmos_L0p35_W0p55 and pmos_L8_W5 seen in F1 do not occur.

