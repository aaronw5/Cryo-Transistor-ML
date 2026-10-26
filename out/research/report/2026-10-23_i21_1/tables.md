# Tables, 2026-10-23 i21_1

Generated 2026-10-26 09:35 by `.venv/bin/python -I scripts/research/plot_explore_i21_1.py --date 2026-10-23`.

### I21-1 setup (pre-registered)

| item | value |
|---|---|
| pre-registration | out/research/explore/I21_1/prereg.md (commit 1808a01), code 0688e64; DEV only, TEST locked; no amendment of a755e18 |
| Part A models | I20-1 full-data g v2, lambda*_full 0 V, VDD 1.8 V, EPS 0.01 V |
| Part A decks | nist4: 53 devices, sky130 77 K lvt library tt_77k, explicit bin, 77 K; nistrt: 87 devices, open_pdks tt nfet_01v8 / pfet_01v8 card, one explicit bin, library tt_rt, 30 C |
| Part A simulator | ~/cryo-ng41/arm64-osdi/bin/ngspice (ngspice-41 --enable-osdi), OpenVAF 23.5.0 OSDI 0.3 |
| Part A gate current | igcmod = igbmod = 0 in every bin of both cards; I18-2 amendment A1 not applicable; V(d,s) = 0 check on the raw element current (I17-1) |
| Part A Q3 device rule | per sky130 circuit device the NIST DEV device of the same polarity nearest in (ln L, ln W); ties first tag |
| Part B | I19-1 S3_D, v1 recipe (fit_bias M_bias, seed 0, 500 steps), LOGO over 7 DEV groups; records I18_2/start/records_{4,295}K (state C) |
| Part C | IHP SG13G2 233, 343, 398 K, I17-2 (c071472); fixed-lambda arms not run on every fold; selected row = 5-seed fold correction at lambda*_f (identical construction) |


### Part A validity (VA1-VA5)

| dataset | VA1 max |card-only deck - M2 row| (<= 1e-9) | VA2 M2 refits | VA3 unit tests | VA4 rows | VA5 DEV only | result |
|---|---:|---:|---:|---:|---:|---|
| NIST lvt (I8-1, 77 K card) | 0.00e+00 | 53 rows, 0 failed | True | 53 rows, 24 Q3 | True | valid |
| NIST RT card (I9-1, room-temperature card) | 0.00e+00 | 87 rows, 0 failed | True | 87 rows, 24 Q3 | True | valid |

- VA1: the card-only measured-grid deck built by the Part B library code (ngspice-41 OSDI build) on every M2 card, x exp(g) in Python, against the M2 in-sample row computed by the I20-1 simulator (canonical ngspice-41).


### Part A core checks and restrictions vs thresholds (I17-1 rules)

| check | type | threshold | NIST lvt | NIST RT card |
|---|---:|---:|---:|---:|
| Q1 equivalence violations (tight, fwd + exchanged) | core | 0 | 21405 | 2477 |
| Q1 convergence messages + non-finite (element decks) | core | 0 | 22287 | 0 |
| Q1 |I(V(d,s) = 0)| tight, raw element (A) | core | <= 1e-12 | inf | 2.67e-17 |
| Verilog-A g vs Python | core | <= 1e-9 | 1.39e-16 | 1.94e-16 |
| Verilog-A g exchanged vs forward | core | <= 1e-12 | 2.50e-16 | 2.73e-16 |
| element-introduced exchange asymmetry | core | <= 1e-3 | 4.54e-05 | 1.29e-03 |
| Q2 gm / gds error vs element identity | core | <= 1e-3 | 1.79e+00 | 6.97e-06 |
| Q2 capacitance error | core | <= 1e-3 | inf | 4.25e-14 |
| gm curves with introduced sign change (strong inversion) | core | 0 | 5 of 159 | 0 of 261 |
| Q3 circuits failing only with the element | core | 0 | 0 | 0 |
| default-tolerance relative difference (|I| > 1 nA) | restriction | <= 1e-3 | 1.95e-03 | 1.94e-03 |
| gds curves with introduced sign change | restriction | 0 | 44 of 265 | 94 of 435 |
| runtime ratio element / card (grid, circuits) | restriction | <= 2 | 3.96 | 3.35 |
| label (decide_deploy, I17-1 V(d,s) = 0 rule) | - | - | not deployable | not deployable |
| predicted label | - | - | deployable with restrictions | deployable with restrictions |

- NIST lvt (I8-1, 77 K card): core failures: Q1 equivalence |dI| <= 1e-6 |I| + 1e-13 A (tight), Q1 convergence / non-finite, Q1 I_D at Vds = 0, Q2 derivative agreement, Q2 capacitances differ from the card, Q2 gm sign change in strong inversion; restrictions: default tolerance above 1e-3, gds sign changes introduced, runtime overhead > 2x. V(d,s) = 0 raw element at default / r4 tolerance (reported): 9.83e-13 / 9.83e-13 A; element-added current at tight tolerance 9.94e-19 A. Replica non-channel outputs max over 61904 printed values: 0.00e+00 A.
- NIST RT card (I9-1, room-temperature card): core failures: Q1 equivalence |dI| <= 1e-6 |I| + 1e-13 A (tight), terminal-exchange asymmetry > 1e-3; restrictions: default tolerance above 1e-3, gds sign changes introduced, runtime overhead > 2x. V(d,s) = 0 raw element at default / r4 tolerance (reported): 9.95e-13 / 9.95e-13 A; element-added current at tight tolerance 1.10e-18 A. Replica non-channel outputs max over 101616 printed values: 0.00e+00 A.


### Part A Q3 circuits (sky130 decks, NIST cards)

| dataset | circuit | published | C0_refit | M2 card | M2 card + element | runtime element / card |
|---|---|---:|---:|---:|---:|---:|
| NIST lvt | inv_A_vtc | V_M 0.735 V, gain 30.0 | V_M 0.723 V, gain 14.9 | V_M 0.724 V, gain 15.0 | V_M 0.724 V, gain 15.2 | 2.31 |
| NIST lvt | inv_B_vtc | V_M 0.649 V, gain 53.8 | V_M 0.617 V, gain 46.1 | V_M 0.618 V, gain 47.5 | V_M 0.618 V, gain 53.7 | 1.71 |
| NIST lvt | ro5_A | 4378 MHz | 2888 MHz | 2896 MHz | 2950 MHz | 3.73 |
| NIST lvt | ro5_B | 347 MHz | 533 MHz | 538 MHz | 546 MHz | 3.96 |
| NIST lvt | mirror_n | ratio 1.0607 | ratio 1.0278 | ratio 1.0267 | ratio 1.0225 | 1.12 |
| NIST lvt | mirror_p | ratio 0.9159 | ratio 0.9222 | ratio 0.9250 | ratio 0.9669 | 1.09 |
| NIST RT card | inv_A_vtc | V_M 0.687 V, gain 35.9 | V_M 0.603 V, gain 14.6 | V_M 0.606 V, gain 15.4 | V_M 0.606 V, gain 15.6 | 2.07 |
| NIST RT card | inv_B_vtc | V_M 0.659 V, gain 248.9 | V_M 0.565 V, gain 68.0 | V_M 0.568 V, gain 76.2 | V_M 0.569 V, gain 85.9 | 1.69 |
| NIST RT card | ro5_A | 979 MHz | 1179 MHz | 1182 MHz | 1206 MHz | 3.35 |
| NIST RT card | ro5_B | fail (1 periods) | fail (3 periods) | fail (3 periods) | fail (3 periods) | - |
| NIST RT card | mirror_n | ratio 1.0076 | ratio 0.9943 | ratio 0.9958 | ratio 0.9956 | 0.88 |
| NIST RT card | mirror_p | ratio 0.9925 | ratio 0.9724 | ratio 0.9725 | ratio 1.0388 | 1.08 |

- NIST lvt (I8-1, 77 K card) circuit devices (role -> device, bin): nmos_L0p15_W1p6 -> nfet_01v8_lvt_W1p65_L0p18_m23p3 (bin 0); pmos_L0p35_W1p6 -> pfet_01v8_lvt_W1_L0p35_m28p6 (bin 0); nmos_L1_W1p6 -> nfet_01v8_lvt_W1p65_L1_m23p6 (bin 3); pmos_L2_W5 -> pfet_01v8_lvt_W5_L1p5_m30p6 (bin 11).
- NIST RT card (I9-1, room-temperature card) circuit devices (role -> device, bin): nmos_L0p15_W1p6 -> nfet_01v8_W1p68_L0p15_m6p7 (bin 44); pmos_L0p35_W1p6 -> pfet_01v8_W1p68_L0p5_m16p1 (bin 41); nmos_L1_W1p6 -> nfet_01v8_W1p68_L1_m7p2 (bin 40); pmos_L2_W5 -> pfet_01v8_W5_L2_m18p5 (bin 12).
- Supply 1.8 V. Q3 core count = circuits failing with M2_full while M2_card passes. Ring: tran with the I14-1 settings, frequency of rising V_DD/2 crossings after 30% of tstop; failure: convergence message, non-finite output or < 5 periods. Mirror ratio: |I_out| / I_ref at V_out = 0.9 V.


### Part A devices behind the core failures (reported attribution, no rule)

| dataset | device | core-check contributions | convergence lines per run | card or card + replica run of the device prints lines |
|---|---|---|---|---|
| NIST lvt | nfet_01v8_lvt_W3_L8_m24p8 | gm sign change at |V(d,s)| 0.05 V | none | no |
| NIST lvt | nfet_01v8_lvt_W5_L4_m25p8 | gm sign change at |V(d,s)| 0.05 V | none | no |
| NIST lvt | pfet_01v8_lvt_W0p42_L20_m27p7 | tight equivalence 235 fwd / 0 exchanged; 3 element grid convergence lines | grid_fwd_tight_elem 3 | no |
| NIST lvt | pfet_01v8_lvt_W3_L0p35_m29p5 | tight equivalence 10585 fwd / 10585 exchanged; 21170 non-finite element grid values; 820 element grid convergence lines; V(d,s) = 0 tight non-finite; Q2 error 1.79e+00; capacitance error inf; gm sign change at |V(d,s)| 0.05, 0.9, 1.8 V | grid_fwd_tight_card_rep 356, grid_swap_tight_card_rep 377, grid_fwd_tight_elem 395, grid_swap_tight_elem 425, ac_VG_card_rep 234, ac_VD_card_rep 234, ac_VG_elem 109, ac_VD_elem 109, dense_card 140, dense_elem 70, repcheck 25 | yes (grid_fwd_tight_card_rep, grid_swap_tight_card_rep, ac_VG_card_rep, ac_VD_card_rep, dense_card) |
| NIST lvt | pfet_01v8_lvt_W3_L1_m29p6 | 6 element grid convergence lines | grid_fwd_tight_card_rep 3, grid_swap_tight_card_rep 3, grid_fwd_tight_elem 3, grid_swap_tight_elem 3 | yes (grid_fwd_tight_card_rep, grid_swap_tight_card_rep) |
| NIST RT card | pfet_01v8_W0p55_L1_m12p5 | tight equivalence 0 fwd / 2477 exchanged; exchange asymmetry 1.29e-03 | grid_swap_tight_card_rep 3 | yes (grid_swap_tight_card_rep) |

- The pre-registered rule counts element decks; a device listed with 'yes' has convergence lines also in decks without the element (the M2 card alone at tight tolerance). Lines are ngspice 'dynamic gmin stepping failed', 'true gmin stepping failed', 'source stepping failed'.


### Part A gds curves with an introduced sign change (restriction)

| dataset | |V(g,s)| (V) | curves (devices) | nMOS | pMOS | dense 1 mV points with a sign change |
|---|---:|---:|---:|---:|---:|
| NIST lvt | 0.6 | 1 of 53 | 0 | 1 | 126 |
| NIST lvt | 0.9 | 5 of 53 | 1 | 4 | 779 |
| NIST lvt | 1.2 | 13 of 53 | 3 | 10 | 6654 |
| NIST lvt | 1.5 | 17 of 53 | 2 | 15 | 12438 |
| NIST lvt | 1.8 | 8 of 53 | 0 | 8 | 6233 |
| NIST lvt | all | 44 of 265 | - | - | - |
| NIST RT card | 0.6 | 1 of 87 | 1 | 0 | 384 |
| NIST RT card | 0.9 | 7 of 87 | 7 | 0 | 2721 |
| NIST RT card | 1.2 | 20 of 87 | 17 | 3 | 8437 |
| NIST RT card | 1.5 | 41 of 87 | 21 | 20 | 25590 |
| NIST RT card | 1.8 | 25 of 87 | 8 | 17 | 14883 |
| NIST RT card | all | 94 of 435 | - | - | - |

- A curve counts when the element's gds (AC 1 MHz or the 1 mV dense sweep) changes sign at |V(d,s)| >= 0.05 V where the card's gds does not. lambda 0 V: the penalty on negative gm / gds is off in training (I20-1 full-data selection).


### Part A element-deck accuracy of the M2 pairs (reported)

| dataset | M2 row (Python, I20-1 simulator) | card-only deck x exp(g) | card + replica: I_card + (exp(g)-1) I_rep | native element deck | max per-device |native - M2 row| | messages |
|---|---:|---:|---:|---:|---:|---:|
| NIST lvt (I8-1, 77 K card) | 0.03045 | 0.03045 | 0.03048 | 0.03048 | 8.90e-05 | 0 |
| NIST RT card (I9-1, room-temperature card) | 0.04578 | 0.04578 | 0.04585 | 0.04585 | 4.27e-04 | 0 |

- Mean in-sample RRMS of the M2 cards (full-data g in the loop) on the measured grid. The Python rows scale the whole card current by exp(g); the element scales the replica channel current only, so the difference is the non-channel current share.


### Part B: TU Delft v1-recipe S3_D, state-L (I19-1) vs state-C records

| temperature | S3_D state L (I19-1) | S3_D state C | decision L | decision C (apply iff S3_D < 0) | sign differs | predicted sign | VB1 state-L check max |diff| | validity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4.13 K | -0.0292 | +0.0055 | apply | C0_refit | True | < 0: wrong | 0.00e+00 | valid |
| 295.0 K | -0.0510 | +0.2537 | apply | C0_refit | True | < 0: wrong | 0.00e+00 | valid |

- Report only; no amendment of the committed I19-1 TEST gate decisions (a755e18). The frozen-method rule (I20-1) reads the v1-recipe S3_D on state-C records; the committed TU Delft primary decisions are C0_refit (g v2-recipe S3_D +0.008 / +0.264). Realised DEV ln GMR(selected g v2, C0_refit): 0.826 / 0.829 (I18-2, gain).


### Part C validity (VC1-VC5) per temperature

| T | VC1 C0_refit max |d| | VC1 M_bias v1 max |d| | VC2 unit tests | VC3 rows | VC3 inner selection | VC4 grids | result |
|---:|---:|---:|---:|---:|---:|---:|---|
| 233 K | 0.00e+00 | 0.00e+00 | True | C0_refit 18, M_bias_v1 18, selected 18 | True (270 trainings, 730 refits) | True | valid |
| 343 K | 0.00e+00 | 0.00e+00 | True | C0_refit 18, M_bias_v1 18, selected 18 | True (270 trainings, 730 refits) | True | valid |
| 398 K | 0.00e+00 | 0.00e+00 | True | C0_refit 18, M_bias_v1 18, selected 18 | True (270 trainings, 730 refits) | True | valid |

- VC1: reruns against I5_1/lodo_<T>K/results.json (threshold 1e-9). VC5 (DEV only) is part of the result column.


### Part C pre-registered rules per temperature: selected g v2 on IHP SG13G2

| T | GMR(selected, C0_refit) lower/n | primary label | predicted label | GMR(selected, M_bias v1) lower/n | secondary label | secondary prediction | predicted signs (vs C0 / vs v1) | GMR(M_bias v1, C0_refit) (reported) |
|---:|---:|---|---|---:|---|---:|---:|---:|
| 233 K | 1.084 [0.898, 1.371] 11/18 | inconclusive | inconclusive (correct) | 0.980 [0.957, 0.999] 7/18 | non-inferior to M_bias v1 | wrong | correct / wrong | 1.106 [0.908, 1.425] 11/18 |
| 343 K | 1.188 [0.990, 1.503] 10/18 | inconclusive | inconclusive (correct) | 1.006 [0.984, 1.033] 10/18 | non-inferior to M_bias v1 | wrong | correct / correct | 1.181 [0.983, 1.502] 8/18 |
| 398 K | 1.260 [1.026, 1.607] 11/18 | worse than C0_refit | worse than C0_refit (correct) | 0.991 [0.975, 1.004] 9/18 | non-inferior to M_bias v1 | wrong | correct / wrong | 1.272 [1.030, 1.627] 11/18 |
| 300 K (I17-2, stored) | 1.255 [1.026, 1.600] 9/18 | worse than C0_refit | - | 1.034 [0.990, 1.107] 7/18 | not shown non-inferior to M_bias v1 | - | - | 1.214 [1.008, 1.537] 9/18 |

- Device bootstrap 10,000 resamples, seed 0, 95% CI, 18 DEV DUTs. Primary: gain replicated iff CI upper < 1 and 0 failed; worse iff CI lower > 1; else inconclusive. Secondary: non-inferior iff CI upper <= 1.05.


### Part C arms per temperature (RRMS all-device / combined / median, LV / HV)

| T | arm | all-device | combined | median | LV mean | HV mean | GMR vs C0_refit, 12-cluster CI |
|---:|---|---:|---:|---:|---:|---:|---:|
| 233 K | C0_refit | 0.0273 | 0.0273 | 0.0208 | 0.0234 | 0.0322 | - |
| 233 K | M_bias_v1 | 0.0281 | 0.0281 | 0.0179 | 0.0231 | 0.0343 | 1.106 [0.890, 1.527] |
| 233 K | selected | 0.0279 | 0.0279 | 0.0168 | 0.0231 | 0.0338 | 1.084 [0.878, 1.460] |
| 233 K | lambda* (folds) | 0 V x1, 1 V x4, 2 V x4, 5 V x2, 10 V x7 |  |  |  |  |  |
| 343 K | C0_refit | 0.0190 | 0.0190 | 0.0120 | 0.0230 | 0.0140 | - |
| 343 K | M_bias_v1 | 0.0194 | 0.0194 | 0.0127 | 0.0241 | 0.0135 | 1.181 [0.948, 1.474] |
| 343 K | selected | 0.0194 | 0.0194 | 0.0123 | 0.0240 | 0.0135 | 1.188 [0.957, 1.477] |
| 343 K | lambda* (folds) | 0 V x1, 1 V x8, 2 V x1, 5 V x4, 10 V x4 |  |  |  |  |  |
| 398 K | C0_refit | 0.0197 | 0.0197 | 0.0111 | 0.0256 | 0.0124 | - |
| 398 K | M_bias_v1 | 0.0202 | 0.0202 | 0.0126 | 0.0269 | 0.0119 | 1.272 [0.980, 1.611] |
| 398 K | selected | 0.0202 | 0.0202 | 0.0119 | 0.0268 | 0.0119 | 1.260 [0.981, 1.592] |
| 398 K | lambda* (folds) | 0 V x3, 1 V x2, 2 V x4, 5 V x3, 10 V x6 |  |  |  |  |  |

- Clusters: class and W/L (I17-2 definition). lambda*: I17-2 nested selection per outer fold.

