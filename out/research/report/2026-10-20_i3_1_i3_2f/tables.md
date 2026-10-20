# Tables, 2026-10-20 i3_1_i3_2f

Generated 2026-10-20 05:43 by `PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i3_1_i3_2f.py --date 2026-10-20` from stored files only. RRMS is the paper metric with frozen curve inclusion, strict. I3-1 values are on simulated devices; I3-2 F1 and F2 REAL_DEV values are on the 12 measured REAL_DEV devices (development evidence).

### I3-1 decision: pre-registered conditions, values and rule output

| id | condition | value | met |
|---|---|---:|---:|
| M1 | tier M: GMR(hyb, par3) <= 0.80 and CI upper bound < 1 | 0.741 [0.610, 0.901] | yes |
| M2 | tier M: CI upper bound of GMR(hyb, lrn_mean) < 1 | 1.119 [0.951, 1.312] | no |
| M3 | tier M: GMR(hyb, pt_interp) < 1 | 0.885 [0.731, 1.064] | yes |
| W1 | tier W: CI upper bound of GMR(hyb, par3) <= 1.25 | 1.739 [1.432, 2.132] | no |
| H1 | tier H: CI upper bound of GMR(hyb, par3) < 1 | 1.224 [1.062, 1.422] | no |
| H2 | tier H: GMR(hyb, lrn_mean) < 1 and GMR(hyb, pt_interp) < 1 | 1.306; 1.763 | no |
| F | failed-cell fraction <= 10% for hyb, par3, lrn_mean, pt_interp (each tier) | max M 0.000, W 0.000, H 0.000 | yes |
|  | KILL condition: tier M GMR(hyb, par3) >= 0.95 | 0.741 | not hit |
|  | KILL condition: tier W GMR(hyb, par3) >= 1.50 | 1.739 | hit |
|  | KILL condition: tier H GMR(hyb, par3) >= 1.00 | 1.224 | hit |
|  | u0 unit-switch validity check (INVALID if failed) | valid on M, W, H | yes |
|  | rule output (decision.json, I3_1_decide) |  | KILL |

- Pre-registration out/research/explore/I3_1/prereg.json (commit c877413) and prereg_addendum.json (hyb inputs geo_t / geo_t_res and output penalty 0.01; written before any NGSpice run of the new arms). CONTINUE = M1-M3, W1, H1, H2 and F all met. KILL = not CONTINUE and any KILL condition hit. INCONCLUSIVE otherwise.
- GMR(a, b) = exp(mean_d log((a_d + 0.01) / (b_d + 0.01))) over the 36 test devices per tier (per-device held-out RRMS = mean over 110/190/260 K, setting O4); percentile bootstrap 95% CI, 4000 resamples, numpy default_rng(0). Values recomputed from eval/*.json and checked against decision.json (|difference| < 1e-12).
- Simulated devices only (12 REAL_DEV geometries x 3 test draws per tier; 72 training devices per tier). No measured I-V enters the decision.


### I3-1 paired GMR of every reported comparison, per tier (O4)

| comparison | tier M GMR [95% CI] | tier M a < b | tier M GMR eps 0.001 | tier W GMR [95% CI] | tier W a < b | tier W GMR eps 0.001 | tier H GMR [95% CI] | tier H a < b | tier H GMR eps 0.001 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hyb vs par3 | 0.741 [0.610, 0.901] | 27/36 | 0.737 | 1.739 [1.432, 2.132] | 6/36 | 3.136 | 1.224 [1.062, 1.422] | 13/36 | 1.245 |
| hyb vs par14 | 0.722 [0.607, 0.858] | 28/36 | 0.715 | 1.852 [1.454, 2.375] | 7/36 | 3.524 | 1.257 [1.096, 1.448] | 12/36 | 1.280 |
| hyb vs par3_pop | 0.548 [0.470, 0.637] | 35/36 | 0.535 | 0.199 [0.130, 0.299] | 32/36 | 0.158 | 0.514 [0.413, 0.636] | 32/36 | 0.496 |
| hyb vs pt_interp | 0.885 [0.731, 1.064] | 20/36 | 0.886 | 0.764 [0.557, 1.027] | 21/36 | 0.680 | 1.763 [1.459, 2.124] | 5/36 | 1.920 |
| hyb vs lrn | 1.140 [0.948, 1.385] | 15/36 | 1.150 | 0.638 [0.481, 0.838] | 24/35 | 0.540 | 1.628 [1.362, 1.944] | 8/36 | 1.735 |
| hyb vs lrn_mean | 1.119 [0.951, 1.312] | 15/36 | 1.129 | 0.491 [0.370, 0.645] | 26/36 | 0.406 | 1.306 [1.082, 1.567] | 11/36 | 1.360 |
| hyb vs hyb_mean | 0.791 [0.693, 0.903] | 28/35 | 0.779 | 0.750 [0.617, 0.899] | 24/36 | 0.736 | 0.906 [0.790, 1.030] | 21/36 | 0.898 |
| lrn vs par3 | 0.650 [0.532, 0.798] | 29/36 | — | 2.603 [1.800, 3.783] | 7/35 | — | 0.752 [0.640, 0.886] | 24/36 | — |
| lrn vs lrn_mean | 0.982 [0.796, 1.209] | 19/36 | — | 0.773 [0.663, 0.911] | 27/35 | — | 0.802 [0.697, 0.919] | 25/36 | — |
| lrn vs pt_interp | 0.776 [0.670, 0.902] | 25/36 | — | 1.205 [1.002, 1.462] | 13/35 | — | 1.083 [1.010, 1.164] | 15/36 | — |
| hyb_mean vs par3 | 0.961 [0.783, 1.192] | 20/35 | — | 2.319 [1.741, 3.136] | 5/36 | — | 1.351 [1.199, 1.542] | 10/36 | — |

- GMR with eps = 0.01 (decision statistic) and eps = 0.001 (sensitivity, reported, not deciding; stored for the hyb comparisons only). a < b = devices with lower unshifted held-out RRMS for a. GMR < 1 = a lower than b.
- Arms (O4 setting, anchors 77/150/225/300 K): par3: per-device parametric law c + (mu - c) w^g fitted on the O4 anchors; par14: par3 with 14 free exponents; par3_pop: par3 with population-pooled exponents; pt_interp: per-parameter interpolation between the anchor fits; lrn: E3b learned population-shared law (network); lrn_mean: E3b population-mean shape (no network); hyb: I3-1 primary: par3 + learned population residual; hyb_mean: par3 + population-mean residual (no network).
- hyb_mean has 1 failed cell on tier M (35 devices in its comparisons); lrn has 1 failed cell on tier W (35 devices in its comparisons).


### I3-1 held-out RRMS per arm and temperature, per tier (O4)

| tier | arm | held-out all-device mean | held-out median | 110 K | 190 K | 260 K | failed cells | anchor 77 K | anchor 150 K | anchor 225 K | anchor 300 K | anchor mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M | par3 | 0.4197 | 0.2983 | 0.6836 | 0.3750 | 0.2004 | 0/108 | 0.0211 | 0.3862 | 0.2532 | 0.0215 | 0.1705 |
| M | par14 | 0.4173 | 0.3237 | 0.6660 | 0.3742 | 0.2118 | 0/108 | 0.0213 | 0.3744 | 0.2702 | 0.0791 | 0.1862 |
| M | par3_pop | 0.6104 | 0.4023 | 0.6781 | 0.9700 | 0.1830 | 0/108 | 0.0211 | 0.9029 | 0.3842 | 0.0144 | 0.3307 |
| M | pt_interp | 0.3689 | 0.2555 | 0.6058 | 0.4106 | 0.0903 | 0/108 | 0.0206 | 0.0240 | 0.0066 | 0.0079 | 0.0148 |
| M | lrn | 0.2778 | 0.1762 | 0.3656 | 0.3953 | 0.0725 | 0/108 | 0.0206 | 0.0240 | 0.0066 | 0.0079 | 0.0148 |
| M | lrn_mean | 0.2872 | 0.1765 | 0.3216 | 0.4687 | 0.0712 | 0/108 | 0.0206 | 0.0240 | 0.0066 | 0.0079 | 0.0148 |
| M | hyb | 0.3086 | 0.2256 | 0.3441 | 0.4394 | 0.1422 | 0/108 | 0.2930 | 0.3549 | 0.1588 | 0.0449 | 0.2129 |
| M | hyb_mean | 0.3706 | 0.2267 | 0.4443 | 0.4966 | 0.1615 | 1/108 | 0.1440 | 0.4720 | 0.1807 | 0.0300 | 0.2068 |
| M | floor | 0.0135 | 0.0017 | 0.0247 | 0.0118 | 0.0040 | 0/108 | — | — | — | — | — |
| W | par3 | 0.0666 | 0.0317 | 0.0927 | 0.0437 | 0.0635 | 0/108 | 0.0204 | 0.0627 | 0.0570 | 0.0112 | 0.0378 |
| W | par14 | 0.0672 | 0.0247 | 0.0903 | 0.0458 | 0.0655 | 0/108 | 0.0194 | 0.0638 | 0.0580 | 0.0131 | 0.0386 |
| W | par3_pop | 0.7468 | 0.3520 | 1.4289 | 0.5151 | 0.2965 | 0/108 | 0.0372 | 2.3993 | 0.4052 | 0.0065 | 0.7121 |
| W | pt_interp | 0.1019 | 0.0809 | 0.1049 | 0.0747 | 0.1261 | 0/108 | 0.0237 | 0.0188 | 0.0071 | 4.5e-10 | 0.0124 |
| W | lrn | 0.1139 | 0.0836 | 0.2257 | 0.0700 | 0.0912 | 1/108 | 0.0237 | 0.0188 | 0.0071 | 4.5e-10 | 0.0124 |
| W | lrn_mean | 0.1507 | 0.1436 | 0.2191 | 0.0909 | 0.1422 | 0/108 | 0.0237 | 0.0188 | 0.0071 | 4.5e-10 | 0.0124 |
| W | hyb | 0.0935 | 0.0703 | 0.1358 | 0.0743 | 0.0705 | 0/108 | 0.1660 | 0.0846 | 0.0677 | 0.0365 | 0.0887 |
| W | hyb_mean | 0.1224 | 0.1052 | 0.2002 | 0.0756 | 0.0915 | 0/108 | 0.0856 | 0.1198 | 0.0882 | 0.0156 | 0.0773 |
| W | floor | 0.0053 | 1.5e-08 | 0.0084 | 0.0053 | 0.0022 | 0/108 | — | — | — | — | — |
| H | par3 | 0.1838 | 0.1159 | 0.2194 | 0.1598 | 0.1723 | 0/108 | 0.0233 | 0.1185 | 0.1291 | 0.0515 | 0.0806 |
| H | par14 | 0.1766 | 0.1126 | 0.2107 | 0.1493 | 0.1699 | 0/108 | 0.0238 | 0.1221 | 0.1199 | 0.0974 | 0.0908 |
| H | par3_pop | 0.4517 | 0.3189 | 0.4998 | 0.6225 | 0.2327 | 0/108 | 0.0269 | 0.5641 | 0.3252 | 0.0084 | 0.2311 |
| H | pt_interp | 0.1654 | 0.0725 | 0.2369 | 0.2050 | 0.0542 | 0/108 | 0.0234 | 0.0147 | 0.0088 | 0.0037 | 0.0126 |
| H | lrn | 0.1572 | 0.0815 | 0.2239 | 0.1803 | 0.0675 | 0/108 | 0.0234 | 0.0147 | 0.0088 | 0.0037 | 0.0126 |
| H | lrn_mean | 0.2178 | 0.1173 | 0.3061 | 0.2750 | 0.0725 | 0/108 | 0.0234 | 0.0147 | 0.0088 | 0.0037 | 0.0126 |
| H | hyb | 0.2197 | 0.1371 | 0.3072 | 0.1866 | 0.1653 | 0/108 | 0.3456 | 0.1494 | 0.1277 | 0.0885 | 0.1784 |
| H | hyb_mean | 0.2667 | 0.1654 | 0.3031 | 0.3252 | 0.1719 | 0/108 | 0.0706 | 0.2596 | 0.1358 | 0.0654 | 0.1329 |
| H | floor | 0.0111 | 8.5e-08 | 0.0194 | 0.0123 | 0.0015 | 0/108 | — | — | — | — | — |

- All-device mean of the strict frozen-inclusion paper RRMS of the NGSpice re-simulated card against the simulated truth, 36 test devices per tier, setting O4. Held-out mean = mean over devices of the per-device mean over 110/190/260 K (failed cells excluded). floor = per-temperature fit on the held-out data (reference, not a method).
- hyb applies the learned correction at every temperature, including the anchors (no gating, as pre-registered); its anchor RRMS is the error of the corrected card at the temperatures where par3 was fitted. lrn and lrn_mean return the anchor fits at the anchors.


### I3-1: the E3b method lrn on tiers M, W and H

| tier | lrn mean | par3 mean | pt_interp mean | lrn_mean mean | GMR lrn vs par3, wins | GMR lrn vs lrn_mean, wins | GMR lrn vs pt_interp, wins | R_O4 [CI] | O4 lrn vs pt_interp reduction | R_O2 [CI] | E3 rule output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M | 0.2778 | 0.4197 | 0.3689 | 0.2872 | 0.650 [0.532, 0.798] 29/36 | 0.982 [0.796, 1.209] 19/36 | 0.776 [0.670, 0.902] 25/36 | 0.338 [0.160, 0.495] | 0.247 | 0.183 [0.019, 0.281] | CONTINUE |
| W | 0.1139 | 0.0666 | 0.1019 | 0.1507 | 2.603 [1.800, 3.783] 7/35 | 0.773 [0.663, 0.911] 27/35 | 1.205 [1.002, 1.462] 13/35 | -0.667 [-1.717, -0.135] | -0.156 | 0.333 [-0.034, 0.462] | INCONCLUSIVE |
| H | 0.1572 | 0.1838 | 0.1654 | 0.2178 | 0.752 [0.640, 0.886] 24/36 | 0.802 [0.697, 0.919] 25/36 | 1.083 [1.010, 1.164] 15/36 | 0.145 [0.000, 0.308] | 0.049 | -0.362 [-0.798, 0.111] | INCONCLUSIVE |

- lrn is the E3b method (learned population-shared temperature law, unchanged code). The tier M and W lrn rows reproduce the E3b full runs (per-device max |difference| 0 on 252 cells each); tier H is new in I3-1 and was not simulated or scored before the I3-1 decision run.
- E3 rule (copied into E3b): CONTINUE if for at least one S in {O4, O2}: R_S >= 0.25, CI lower bound of R_S > 0, lrn >= 10% below pt_interp, failed cells <= 10%; KILL if R_S < 0.10 for both S or lrn worse than pt_interp for both S; INCONCLUSIVE otherwise. R_S = 1 - mean(lrn) / mean(control_S), control_O4 = par3, control_O2 = par3_pop. The E3 rule output is reported per tier by the run (summary.json e3_rule_output_reported); only tier M decided E3b.
- Tier H, lrn: GMR vs par3 below 1 with CI upper bound below 1; R_O4 0.145 is below the 0.25 threshold; lrn vs pt_interp GMR 1.083 with CI above 1 (lrn higher than pt_interp on 21/36 devices) and O4 reduction 0.049 below the 0.10 threshold; R_O2 is negative.


### I3-1 validity checks, hyb ensemble state and reproduction

| tier | truths at/above switch | fitted cards | predicted before clip | evaluated cards | clipped evaluated cards | hyb member best steps | hyb member input set | members at zero-correction state | reproduction vs E3b (max abs diff) | wall time s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M | 0/180 (10 rejected, 6 regen.) | 0/2340 | 3/5544 | 0/5652 | hyb_n3 2, hyb_n12 1 | 1800, 175, 225, 2999, 75 | geo_t, geo_t, geo_t, geo_t, geo_t_res | 0 | max 1.7e-07 | 1628 |
| W | 0/108 (2 rejected, 2 regen.) | 0/1332 | 0/4536 | 0/4644 | hyb_mean 1 | 75, 375, 100, 325, 25 | geo_t_res, geo_t, geo_t_res, geo_t, geo_t_res | 0 | max 0 | 1097 |
| H | 0/108 (10 rejected, 5 regen.) | 0/1332 | 0/4536 | 0/4644 | hyb 1 | 50, 75, 50, 125, 75 | geo_t_res, geo_t_res, geo_t_res, geo_t_res, geo_t_res | 0 | new tier | 1072 |

- Validity rule: 0 truths, 0 fitted cards and 0 evaluated cards at or above the BSIM4 u0 unit switch; predicted cards are counted before the x_u0 clip. All three tiers are valid.
- hyb members: each member trains the geo_t (geometry, T) and geo_t_res (geometry, par3 anchor residuals, T) networks and keeps the one with the lower validation criterion; best step of 3000 (step -1 = zero-correction state, chosen by 0 members on every tier).
- Reproduction vs E3b: existing arms O4 par3, pt_interp, lrn, lrn_mean and O2 par3_pop on tiers M and W, 252 cells each, against the E3b full runs.
- Queue jobs I3_1_full_M 2026-10-20 01:55-02:22, I3_1_full_W 02:22-02:40, I3_1_full_H 02:40-02:58, I3_1_decide 02:58 (out/research/done.txt).


### I3-2 follow-up F1: SPICE-native Verilog-A validation, pre-registered checks

| check | comparison | value (threshold) | outcome |
|---|---|---:|---:|
| F1.0 | export text: g from emitted Verilog-A lines vs Python; OpenVAF compiles | max |g diff| 5.3e-11 (<= 1e-09); 13/13 compiled | pass |
| F1.1 | ngspice-46 vs ngspice-41, uncorrected cards (published, start, C0_refit, M_bias) | max rel. current 21.5 (<= 1e-3); max |dRRMS| 2.23 (<= 1e-4) | fail |
|  | ngspice-41 reproduces stored C0_refit, M_bias, released-card RRMS | max |diff| 0 (<= 1e-9) | pass |
| F1.2 | SPICE-native (ngspice-46 + OSDI) vs Python I_bsim exp(g) on ngspice-46 | max |dRRMS| 1.4e-05 (<= 1e-4); max rel. current fold 2.4e-03, full 1.4e-03 (<= 1e-3); 0 convergence messages, 0 non-finite | fail |
| F1.3 | SPICE-native (ngspice-46) vs stored I3-2 M_bias (ngspice-41, Python) | max per-device |d| 2.12 (<= 2e-4); |d all-device mean| 0.240 (<= 1e-4) | fail |
| diag. | tight tolerance (reltol 1e-9, abstol 1e-18): SPICE-native vs Python, ngspice-46 | max |dRRMS| 1.4e-09; max rel. current 1.4e-03 | no pass/fail |

- Pre-registration out/research/explore/I3_2/followup_prereg.json (commit 4ae726a); amendment followup_prereg_amendment_2026-10-20.json (commit 4f4e7ef): relative current differences exclude points with |I| < 1e-20 A (numerical zeros at Vds = 0). Under the literal definition (|I| > 0) F1.1 and F1.2 also fail (literal F1.2 fold value 2.9e+10).
- Binaries: ngspice-41 arm64 build (/Users/anrunw/cryo-ng41/arm64, configured without OSDI) and Homebrew ngspice-46 (/opt/homebrew/bin/ngspice). The SPICE-native runs use ngspice-46 because the ngspice-41 build has no OSDI. Corrected netlist: BSIM4 drain on node ndi, compiled correction between nd and ndi with a zero-volt sense branch (V(sense) <+ RS * I(sense), RS = 0).
- F1.1 and F1.3 fail on pmos_L0p35_W0p55 and pmos_L8_W5 only (next table): ngspice-46 and ngspice-41 give different currents for the uncorrected cards on these two devices. On the other 10 devices the uncorrected ngspice-46 and ngspice-41 currents are equal (max relative difference 0).
- F1.2 per-device |dRRMS| (SPICE-native minus Python, both ngspice-46) is at most 1.4e-05 on all 12 devices. The relative current difference of the fold models exceeds 1e-3 on pmos_L0p35_W0p55 (1.2e-03), pmos_L0p5_W0p42 (2.4e-03), pmos_L8_W1p6 (1.4e-03); on the other devices it is 7.9e-4 to 1.0e-3. With reltol 1e-9 the per-device |dRRMS| is at most 1.4e-09; the relative current difference is below 3e-6 on 9 devices and is 2.7e-04 (pmos_L0p35_W0p55), 6.9e-04 (pmos_L0p5_W0p42), 1.4e-03 (pmos_L8_W1p6) on the others.
- F1b (SPICE-native runs on an OSDI-enabled ngspice-41 build) is in progress (another agent); no F1b result is used here.


### I3-2 follow-up F1 per device: ngspice-41 vs ngspice-46 and SPICE-native vs Python RRMS

| device | published ng41 | published ng46 | C0_refit card ng41 | C0_refit card ng46 | M_bias Python ng41 (stored) | M_bias Python ng46 | M_bias SPICE-native ng46 | abs(native - Python ng46) | max rel current diff | abs dRRMS at reltol 1e-9 | ng46 RRMS equal for all 4 cards |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nmos_L0p15_W1p6 | 0.0595 | 0.0595 | 0.0193 | 0.0193 | 0.0235 | 0.0235 | 0.0235 | 9.7e-06 | 9.0e-04 | 4.6e-12 |  |
| nmos_L0p25_W1p6 | 0.0983 | 0.0983 | 0.0188 | 0.0188 | 0.0183 | 0.0183 | 0.0183 | 9.0e-06 | 9.9e-04 | 6.3e-12 |  |
| nmos_L100_W100 | 0.1403 | 0.1403 | 0.0731 | 0.0731 | 0.0686 | 0.0686 | 0.0686 | 2.5e-06 | 1.0e-03 | 3.4e-11 |  |
| nmos_L1_W3 | 0.1599 | 0.1599 | 0.0561 | 0.0561 | 0.0504 | 0.0504 | 0.0504 | 1.3e-06 | 8.7e-04 | 2.8e-11 |  |
| nmos_L20_W0p64 | 0.1283 | 0.1283 | 0.0489 | 0.0489 | 0.0428 | 0.0428 | 0.0428 | 6.3e-07 | 7.9e-04 | 1.4e-10 |  |
| pmos_L0p35_W0p55 | 0.7270 | 2.6351 | 0.4089 | 2.6351 | 0.3496 | 2.4721 | 2.4721 | 1.2e-05 | 1.2e-03 | 6.6e-10 | yes |
| pmos_L0p35_W5 | 0.3291 | 0.3291 | 0.1122 | 0.1122 | 0.0693 | 0.0693 | 0.0693 | 1.4e-06 | 9.9e-04 | 1.4e-09 |  |
| pmos_L0p5_W0p42 | 0.4560 | 0.4560 | 0.1615 | 0.1615 | 0.1083 | 0.1083 | 0.1083 | 1.2e-05 | 2.4e-03 | 6.4e-11 |  |
| pmos_L2_W5 | 0.1907 | 0.1907 | 0.0796 | 0.0796 | 0.0553 | 0.0553 | 0.0553 | 3.9e-06 | 9.7e-04 | 1.1e-10 |  |
| pmos_L4_W7 | 0.2702 | 0.2702 | 0.0844 | 0.0844 | 0.0656 | 0.0656 | 0.0657 | 7.6e-06 | 9.3e-04 | 2.4e-10 |  |
| pmos_L8_W1p6 | 0.5433 | 0.5433 | 0.1321 | 0.1321 | 0.0724 | 0.0724 | 0.0724 | 2.4e-06 | 1.4e-03 | 3.8e-11 |  |
| pmos_L8_W5 | 0.3146 | 0.8670 | 0.0914 | 0.8670 | 0.0425 | 0.7995 | 0.7995 | 1.4e-05 | 9.8e-04 | 1.8e-10 | yes |
| all-device | 0.2848 | 0.4898 | 0.1072 | 0.3573 | 0.0806 | 0.3205 | 0.3205 |  |  |  |  |
| combined | 0.2608 | 0.4366 | 0.0981 | 0.3125 | 0.0749 | 0.2805 | 0.2805 |  |  |  |  |

- Strict paper RRMS with frozen inclusion on the measured 77 K curves, 12 REAL_DEV devices; all-device = mean over 12, combined = mean of the nMOS (5) and pMOS (7) family means. Tinted columns: ngspice-46.
- Python ng41 = the stored I3-2 M_bias held-out value (reproduced with difference 0). published ng41 equals out/research/reference_scores.json 'released' (difference 0).
- On pmos_L0p35_W0p55, pmos_L8_W5 ngspice-46 returns the same RRMS for the published, starting, C0_refit and M_bias refit cards (last column), while ngspice-41 returns different values for the four cards. On the other 10 devices ngspice-46 and ngspice-41 give equal RRMS for every card.
- max rel. I = max |I_native - I_Python| / |I_Python| over scored points with |I| >= 1e-20 A (pre-registered threshold 1e-3). |dRRMS| tight = same comparison with '.options reltol=1e-9 abstol=1e-18' (diagnostic).


### I3-2 follow-up F2 on REAL_DEV: seed, capacity and training-card variants of M_bias

| variant | change | all-device | combined | nMOS | pMOS | median | failed | GMR vs C0_refit [95% CI] | devices below C0_refit | GMR vs I3-2 M_bias [95% CI] | devices below I3-2 M_bias | all-device <= 0.08698 | pre-registered check |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0_refit | control (stored I2-1 rows) | 0.1072 | 0.0981 | 0.0432 | 0.1529 | 0.0820 | 0/12 |  |  |  |  |  |  |
| I3-2 M_bias | seeds 0-4, MLP 3-16-16-1, round-3 cards | 0.0806 | 0.0749 | 0.0407 | 0.1090 | 0.0605 | 0/12 | 0.769 [0.666, 0.886] | 11/12 |  |  | yes |  |
| a_seeds_5_9 | ensemble seeds 5-9 | 0.0805 | 0.0748 | 0.0405 | 0.1091 | 0.0588 | 0/12 | 0.771 [0.668, 0.894] | 10/12 | 1.003 [0.988, 1.020] | 6/12 | yes | pass |
| a_seeds_10_14 | ensemble seeds 10-14 | 0.0813 | 0.0754 | 0.0403 | 0.1105 | 0.0595 | 0/12 | 0.779 [0.683, 0.894] | 10/12 | 1.014 [0.990, 1.043] | 6/12 | yes | pass |
| a_seeds_15_19 | ensemble seeds 15-19 | 0.0804 | 0.0748 | 0.0408 | 0.1088 | 0.0584 | 0/12 | 0.771 [0.668, 0.893] | 10/12 | 1.002 [0.987, 1.018] | 6/12 | yes | pass |
| b_width8 | MLP 3-8-8-1 | 0.0805 | 0.0748 | 0.0406 | 0.1089 | 0.0545 | 0/12 | 0.759 [0.666, 0.867] | 11/12 | 0.988 [0.955, 1.025] | 6/12 | yes | pass |
| b_width32 | MLP 3-32-32-1 | 0.0814 | 0.0755 | 0.0400 | 0.1109 | 0.0591 | 0/12 | 0.777 [0.678, 0.895] | 10/12 | 1.011 [0.982, 1.044] | 4/12 | yes | pass |
| b_depth1 | MLP 3-16-1 | 0.0804 | 0.0747 | 0.0404 | 0.1090 | 0.0605 | 0/12 | 0.770 [0.668, 0.892] | 11/12 | 1.002 [0.989, 1.017] | 8/12 | yes | pass |
| c_train_round1_cards | g trained on round-1 card records | 0.0828 | 0.0769 | 0.0416 | 0.1121 | 0.0610 | 0/12 | 0.832 [0.710, 0.996] | 10/12 | 1.083 [0.988, 1.192] | 6/12 | yes | pass |

- REAL_DEV leave-one-device-out (12 devices: 5 nMOS, 7 pMOS), held-out card refit from the round-3 starting card with the frozen correction in the loop (max_nfev 400, irls 2, diff_step 0.01), strict paper RRMS. Each variant changes one item of the I3-2 M_bias protocol.
- GMR = paired geometric-mean ratio exp(mean ln(variant / reference)), device bootstrap 95% CI, 10000 resamples, default_rng(0); recomputed from the stored per-device values and checked (|difference| < 1e-12). A variant fails if the CI upper bound vs C0_refit is >= 1 or any device fails.
- Group outcomes: seeds (a) pass, capacity (b) pass, training cards (c) pass. All-device range over the four seed sets (0-4, 5-9, 10-14, 15-19): 0.0804-0.0813.
- 0.08698 = the I3-2 R2 threshold (T1_inloop 0.08198 + 0.005); reported, not a check. c_train_round1_cards: the held-out refit still starts from the round-3 card; the round-1 card records reproduce the stored round-1 RRMS (difference 0).
- REAL_DEV was used by E1, I2-1 and I3-2; these values are development evidence. Queue job I3_2_F2_real 2026-10-20 02:59-03:15 (955 s).


### I3-2 follow-up F2(d): GF180 and IHP tier b, M_bias vs C0_refit, with the tier b refit floors

| PDK | row | all-target mean | median | failed | GMR vs C0_refit [95% CI] | targets below C0_refit | R4 rule (GMR <= 0.90, 0 failed) | CI upper < 1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GF180 (BSIM4) tier b | C0_start | 0.0139 | 0.0128 | 0/48 |  |  |  |  |
| GF180 (BSIM4) tier b | C0_refit | 0.0133 | 0.0126 | 0/48 |  |  |  |  |
| GF180 (BSIM4) tier b | M_bias | 0.0133 | 0.0126 | 0/48 | 0.996 [0.986, 1.006] | 26/48 | fail | no |
| GF180 (BSIM4) tier b | floor, seven-parameter refit (3 devices, set 0) | 0.0136 | — |  |  |  |  |  |
| GF180 (BSIM4) tier b | floor, wide-box refit (3 devices, set 0) | 0.0125 | — |  |  |  |  |  |
| GF180 (BSIM4) tier b | injected noise level (3 devices, set 0) | 0.0119 | — |  |  |  |  |  |
| IHP SG13G2 (PSP103) tier b | C0_start | 0.0141 | 0.0128 | 0/48 |  |  |  |  |
| IHP SG13G2 (PSP103) tier b | C0_refit | 0.0141 | 0.0128 | 0/48 |  |  |  |  |
| IHP SG13G2 (PSP103) tier b | M_bias | 0.0138 | 0.0128 | 0/48 | 0.985 [0.975, 0.995] | 28/48 | fail | yes |
| IHP SG13G2 (PSP103) tier b | floor, seven-parameter refit (3 devices, set 0) | 0.0146 | — |  |  |  |  |  |
| IHP SG13G2 (PSP103) tier b | floor, wide-box refit (3 devices, set 0) | 0.0135 | — |  |  |  |  |  |
| IHP SG13G2 (PSP103) tier b | injected noise level (3 devices, set 0) | 0.0124 | — |  |  |  |  |  |

- Simulated tier b truth sets 0-2 (16 geometries x 3 = 48 targets per PDK), leave-one-geometry-out (16 folds), M_bias seeds 0-4; new starting-card records (polish from the truth's local-wide projection, wide box) and new C0_refit runs. GMR with geometry-cluster bootstrap 95% CI (10000 resamples, default_rng(0)), recomputed and checked.
- F2(d) check: fails if on both PDKs GMR > 0.90 or M_bias has a failed target (the I3-2 R4 rule on tier b). Outcome: fail (GF180 0.996, IHP 0.985).
- Floor rows: out/research/pdk2/floors/2026-10-16_final (3 devices per PDK, truth set 0, per-device refit to its own tier b truth; seven-parameter and wide-box parameter sets); noise = RRMS of the injected measurement noise. These are separate runs on 3 devices, not the 48 F2 targets.
- C0_start = starting card without refit or correction. Queue jobs I3_2_F2_gf180_b 03:15-03:31 (982 s), I3_2_F2_ihp_b 03:31-03:49 (1086 s).

