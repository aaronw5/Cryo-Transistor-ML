# Tables, 2026-10-20 explore_i2_e3b

Generated 2026-10-20 00:42 by `PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i2_e3b.py --date 2026-10-20` from result files only. RRMS is the paper metric with frozen curve inclusion, strict.

### I2-0 (diagnostic only): iteration-1 REAL_DEV decision quantities with and without the two low-denominator pMOS devices

| run | quantity | all 12 devices | without pmos_L0p35_W5 | without pmos_L0p35_W0p55 | without both |
|---|---|---:|---:|---:|---:|
| E1 stage 1 | M / T1 / C0 all-device | 0.1276 / 0.0865 / 0.1073 | 0.1328 / 0.0878 / 0.1068 | 0.1077 / 0.0627 / 0.0799 | 0.1113 / 0.0618 / 0.0766 |
|  | M < T1 / M < C0 (devices); pass | 4/12, 6/12; no | 3/11, 5/11; no | 3/11, 5/11; no | 2/10, 4/10; no |
| E1 stage 2 | M / T1 / C0 all-device | 0.0871 / 0.0820 / 0.1072 | 0.0905 / 0.0830 / 0.1067 | 0.0663 / 0.0582 / 0.0798 | 0.0680 / 0.0570 / 0.0765 |
|  | M < T1 / M < C0 (devices); pass | 6/12, 8/12; no | 5/11, 7/11; no | 5/11, 7/11; no | 4/10, 6/10; no |
| E2 | R_raw N0 / M / S (penalized) | 11.24 / 6.26 / 19.58 | 8.03 / 3.79 / 10.67 | 8.92 / 5.17 / 13.20 | 5.16 / 2.34 / 2.76 |
|  | S / N0; S wins; tier-a guard C4 | 1.742; 6/12; yes | 1.329; 6/11; no | 1.479; 6/11; yes | 0.534; 6/10; no |
|  | rule output | KILL | KILL | KILL | INCONCLUSIVE |
| E4 p43 vd | single / Laplace / learned / crossinterp | 0.418 / 500.21 / 10.869 / 0.098 | 0.419 / 1.75 / 2.782 / 0.104 | 0.396 / 545.24 / 0.138 / 0.083 | 0.395 / 1.44 / 0.092 / 0.088 |
|  | learned: coverage 0.9; wins; C1 C2 C3 | 0.963; 1/12; no no no | 0.960; 2/11; no no no | 0.991; 8/11; no yes no | 1.000; 9/10; no yes no |
|  | rule output | PARTIAL_CALIBRATION_ONLY | PARTIAL_CALIBRATION_ONLY | PARTIAL_CALIBRATION_ONLY | PARTIAL_CALIBRATION_ONLY |
| E5 b43 | A0 / A2 all-device | 0.1073 / 0.2004 | 0.1068 / 0.1973 | 0.0799 / 0.1906 | 0.0766 / 0.1862 |
|  | reduction; A2 < A0 | -0.868; 1/12 | -0.847; 1/11 | -1.387; 0/11 | -1.430; 0/10 |

- DIAGNOSTIC ONLY. Decision quantities recomputed after the fact with devices removed. The stored pre-registered verdicts are unchanged; these values are not decision inputs.
- W5 = pmos_L0p35_W5, W0p55 = pmos_L0p35_W0p55: the two REAL_DEV devices whose frozen inclusion set holds curves with denominators 1e-4 to 5e-4 of the device's largest (I2-0 part A).
- E2 and E4 are exact on the reduced set (E2 real devices are evaluation-only; E4 learned and conformal leave-one-device-out recomputed). E1 and E5 are aggregate-only: their fold models were trained with the removed devices in the other folds. Win thresholds scale with n.
- Source: out/research/explore/I2_0/C_decision_sensitivity.json (script scripts/research/explore/i2_0_diagnose.py, stage decisions).


### I2-0 (diagnostic only): E3 tier-M decision quantities with and without the test device whose truth crosses the u0 unit switch

| S | quantity | all 36 | without nmos_L1_W3__2 (35) | without the 3 nmos_L1_W3 test devices (33) |
|---|---|---:|---:|---:|
| O4 | lrn / par3 / pt_interp all-device | 9.801 / 9.123 / 10.118 | 0.267 / 0.423 / 0.370 | 0.266 / 0.437 / 0.374 |
|  | R_O4 (95% CI) | -0.074 [-0.090, 0.504] | 0.369 [0.168, 0.533] | 0.392 [0.183, 0.556] |
|  | 1 − lrn / pt_interp; lrn < control | 0.031; 27/36 | 0.278; 27/35 | 0.290; 27/33 |
| O2 | lrn / par3_pop / pt_interp all-device | 7.102 / 6.858 / 9.592 | 0.483 / 0.610 / 0.964 | 0.481 / 0.623 / 0.969 |
|  | R_O2 (95% CI) | -0.036 [-0.055, 0.288] | 0.208 [0.031, 0.305] | 0.227 [0.057, 0.322] |
|  | 1 − lrn / pt_interp; lrn < control | 0.260; 21/36 | 0.499; 21/35 | 0.503; 21/33 |
|  | rule output (stored run: KILL) | KILL | CONTINUE | CONTINUE |

- DIAGNOSTIC ONLY. The stored E3 verdict (KILL) is unchanged. Removing a test device does not remove the switched truths from the E3 training population.
- nmos_L1_W3__2: truth u0 is above 1 (BSIM4 unit switch, u0 read as cm^2/Vs and divided by 1e4) at 77-110 K; every method's 110 K card is below the switch and gives RRMS 676-1060 in that cell (I2-0 part B).
- Paired bootstrap 4000 resamples, seed 0 (the stored run used its own seed, so CI bounds differ slightly). The failed-cell check is not recomputed (0 failed cells in the stored run).
- This diagnostic was computed before the E3b pre-registration; E3b copies the E3 rule without change and does not use these values.


### I2-1 decision (REAL_DEV): correction trained jointly with the training devices' cards, pre-registered checks

| condition | value | met |
|---|---:|---:|
| M_joint all-device mean <= 0.080 | 0.0903 | no |
| M_joint < M_post on >= 8 of 12 devices | 6/12 | no |
| M_joint mean <= T1_joint mean + 0.005 | 0.0903 vs 0.0853 | no |
| 0 failed devices for M_joint | 0/12 | yes |
| rule output (continue only if all four hold) |  | KILL |

- Pre-registration: out/research/explore/I2_1/prereg.json (written before any joint-arm result). REAL_DEV leave-one-device-out, 12 folds, K = 3 alternating rounds (training cards refit with g_{r-1} in the loop, correction retrained from scratch on the refit residuals); held-out card refit from its starting card with the frozen g_3 (TRF, max_nfev 400, irls 2, diff_step 0.01).
- Reference values: M_post 0.0871 (E1 stage 2, stored), C0_refit 0.1072 (rerun; max abs difference to E1 stage 2 = 0), T1_joint 0.0803.
- Metric: strict paper RRMS with frozen published-baseline inclusion of the NGSpice simulation of the refit card multiplied by exp(g); a non-finite included curve makes the device NaN (0 failed devices in every arm).
- Queue job i2_1_real, 2026-10-19 17:59-18:35 (2152 s).


### I2-1 REAL_DEV RRMS per arm, with the held-out refits at intermediate rounds

| arm | role | all-device | combined | nMOS | pMOS | median | devices below C0_refit | devices below M_post | failed devices |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0_refit | control | 0.1072 | 0.0981 | 0.0432 | 0.1529 | 0.0820 |  |  | 0/12 |
| M_post | reference (E1 stage 2) | 0.0871 | 0.0842 | 0.0669 | 0.1015 | 0.0612 | 8/12 |  | 0/12 |
| M_joint | primary | 0.0903 | 0.0884 | 0.0769 | 0.1000 | 0.0622 | 7/12 | 6/12 | 0/12 |
| T1_joint | comparator, not deployable | 0.0803 | 0.0744 | 0.0390 | 0.1099 | 0.0640 | 11/12 | 6/12 | 0/12 |
| M_joint_r1 | diagnostic: held-out refit with g_1 | 0.0893 | 0.0867 | 0.0707 | 0.1026 | 0.0639 | 7/12 | 2/12 | 0/12 |
| M_joint_r2 | diagnostic: held-out refit with g_2 | 0.0914 | 0.0896 | 0.0785 | 0.1007 | 0.0675 | 7/12 | 4/12 | 0/12 |
| ORACLE_joint | diagnostic, in-sample | 0.0368 | 0.0344 | 0.0199 | 0.0490 | 0.0253 | 12/12 | 12/12 | 0/12 |

- all-device = mean over the 12 REAL_DEV devices; combined = mean of the nMOS (5) and pMOS (7) family means; strict frozen-inclusion paper RRMS, 0 failed devices in every arm.
- Every arm except M_post refits the held-out card from the same starting card (direct43_r3_fd) with max_nfev 400, irls 2; M_post values are the stored E1 stage-2 results (equivalent to K = 0).
- ORACLE_joint runs the joint loop on the held-out device itself (in-sample); it is an upper bound for the network class, not a held-out result.


### I2-1 simulated arm (secondary): GF180 and IHP tier c, leave-one-geometry-out

| set | arm | all-target mean | median | ratio to C0_refit | targets below C0_refit | targets below M_post | failed targets | M_joint <= 0.90 x C0_refit | run time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GF180 (BSIM4) tier c | C0_refit | 0.0457 | 0.0352 | — |  |  | 0/48 |  | 6534 s |
|  | M_post | 0.0389 | 0.0305 | 0.8517 | 32/48 |  | 0/48 |  |  |
|  | M_joint | 0.0381 | 0.0282 | 0.8339 | 40/48 | 37/48 | 0/48 | pass |  |
| IHP SG13G2 (PSP103) tier c | C0_refit | 0.1011 | 0.0567 | — |  |  | 0/48 |  | 7246 s |
|  | M_post | 0.1019 | 0.0700 | 1.0077 | 26/48 |  | 0/48 |  |  |
|  | M_joint | 0.0915 | 0.0500 | 0.9045 | 23/48 | 27/48 | 0/48 | fail |  |

- Secondary (prereg simulated_arm): tier c truth sets 0-2, 16 geometries x 3 draws per PDK, leave-one-geometry-out (16 folds). All-target mean = combined (24 nMOS, 24 pMOS). pdk2.metric paper RRMS with target-frozen inclusion, strict.
- M_post on these sets is a new run (held-out refit with the E1 fold correction g_0); T1 is not run on the simulated sets. check_S: M_joint all-target mean <= 0.90 x C0_refit; it does not change the REAL_DEV decision.
- Queue jobs i2_1_gf180 (2026-10-19 18:35-20:24) and i2_1_ihp (20:24-22:25).


### I2-1 identifiability diagnostic: correlation of the learned correction g with the injected misspecification components (GF180, IHP)

| PDK | arm | r(g, c_param) | r(g, c_rs) | r(g, c_heat) | r(g, c_gain) | r(g, c_sum) | r(g, observed residual) | OLS R^2 | OLS coefficients | mean per-target r with c_sum | std of g | std of observed residual | pooled points |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GF180 | M_post | 0.075 | 0.351 | -0.160 | 0.037 | -0.017 | 0.013 | 0.185 | 0.05 / 0.29 / -0.05 / 0.10 | -0.222 | 0.040 | 0.215 | 84,129 |
|  | M_joint | 0.104 | 0.322 | -0.234 | 0.017 | -0.070 | -0.029 | 0.214 | 0.10 / 0.36 / -0.09 / 0.03 | -0.299 | 0.052 | 0.225 | 84,129 |
| IHP | M_post | -0.028 | 0.198 | 0.030 | -0.039 | 0.014 | 0.190 | 0.041 | -0.00 / 0.39 / -0.01 / -0.72 | 0.091 | 0.172 | 0.633 | 85,788 |
|  | M_joint | 0.022 | 0.266 | 0.076 | -0.031 | 0.088 | 0.246 | 0.074 | 0.01 / 0.53 / 0.02 / -0.64 | 0.099 | 0.177 | 0.627 | 85,788 |

- Diagnostic without a threshold (prereg identifiability_diagnostic). g = the arm's correction evaluated on its final refit card's simulation; pooled over the 48 held-out targets, included curves, trimmed points.
- Injected components from the truth card (NGSpice on the target grid): c_param = misspecified non-fitted parameters; c_rs = series resistance; c_heat = per-curve self-heating temperature; c_gain = per-curve gain; c_sum = their sum; residual = ln|target / I_card|. Each in ln(current).
- OLS: g on [1, c_param, c_rs, c_heat, c_gain]. The largest correlation in both PDKs is with c_rs (0.20-0.35); R^2 is 0.18-0.21 on GF180 and 0.04-0.07 on IHP.


### E3b decision: learned shared temperature law (lrn) vs parametric laws after the u0 unit-switch fix, pre-registered checks

| run | S | control | R_S = 1 − lrn/control | R_S bootstrap 95% CI | R_S >= 0.25 | CI lower bound > 0 | 1 − lrn/pt_interp | lrn >= 10% below pt_interp | max failed-cell fraction | failed <= 10% | continue for S | R_S < 0.10 | lrn worse than pt_interp | u0-switch assertion | rule output for the run |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tier M (decision) | O4 | par3 | 0.3380 | [0.160, 0.495] | yes | yes | 0.2469 | yes | 0.000 | yes | yes | no | no | valid | CONTINUE |
| tier M (decision) | O2 | par3_pop | 0.1830 | [0.019, 0.281] | no | yes | 0.4838 | yes | 0.000 | yes | no | no | no | valid |  |
| tier W (secondary) | O4 | par3 | -0.6666 | [-1.717, -0.135] | no | no | -0.1556 | no | 0.009 | yes | no | yes | yes | valid | INCONCLUSIVE |
| tier W (secondary) | O2 | par3_pop | 0.3327 | [-0.034, 0.462] | yes | no | 0.1416 | yes | 0.000 | yes | no | no | no | valid |  |

- Rule copied without change from the E3 pre-registration (out/research/explore/E3b/prereg.json): CONTINUE if for at least one S: R_S >= 0.25 AND bootstrap 95% CI lower bound > 0 AND lrn >= 10% below pt_interp AND failed cells <= 10% for lrn, control and pt_interp. KILL if R_S < 0.10 for both S, or lrn worse than pt_interp for both S. Otherwise INCONCLUSIVE. A run with a truth, fitted card or evaluated card at or above the u0 switch is INVALID.
- u0 check, tier M: 0/108 truths, 0/1332 fitted, 0/4032 predicted (before clip) and 0/4140 evaluated cards at or above the switch; 0 clipped cards; 4 truths regenerated (7 rejected draws). Tier W: same counts, 2 truths regenerated (2 rejected draws).
- Decision run: tier M (queue job E3b_full_M, 2026-10-19 22:25-22:40, 910 s). Tier W (E3b_full_W, 22:40-22:57, 989 s; truth inside the parametric family) is secondary. Tier W O4 uses 35 devices (1 failed lrn cell). Simulated devices only; no measured multi-temperature I-V exists.
- The E3 verdict (KILL, queue job E3_full_M) stands; E3b is a separate pre-registered probe.


### E3b held-out-temperature RRMS per method, tiers M and W (simulated devices)

| tier | S | method | held-out all-device mean | held-out combined | held-out median | 110 K | 190 K | 260 K | observed-temperature mean | failed cells |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M | O4 | lrn | 0.2778 | 0.2599 | 0.1762 | 0.3656 | 0.3953 | 0.0725 | 0.0148 | 0/108 |
| M | O4 | par3 (control) | 0.4197 | 0.3916 | 0.2983 | 0.6836 | 0.3750 | 0.2004 | 0.1705 | 0/108 |
| M | O4 | par14 | 0.4173 | 0.3905 | 0.3237 | 0.6660 | 0.3742 | 0.2118 | 0.1862 | 0/108 |
| M | O4 | par3_pop | 0.6104 | 0.5673 | 0.4023 | 0.6781 | 0.9700 | 0.1830 | 0.3307 | 0/108 |
| M | O4 | pt_interp | 0.3689 | 0.3404 | 0.2555 | 0.6058 | 0.4106 | 0.0903 | 0.0148 | 0/108 |
| M | O4 | lrn_mean | 0.2872 | 0.2726 | 0.1765 | 0.3216 | 0.4687 | 0.0712 | 0.0148 | 0/108 |
| M | O4 | lrn_noH | 0.2801 | 0.2641 | 0.1918 | 0.3696 | 0.3962 | 0.0744 | 0.0148 | 0/108 |
| M | O4 | lrn_param | 0.3207 | 0.2989 | 0.2207 | 0.4221 | 0.4480 | 0.0853 | 0.0148 | 1/108 |
| M | O2 | lrn | 0.4945 | 0.4679 | 0.3605 | 0.5931 | 0.7582 | 0.1321 | 0.0138 | 0/108 |
| M | O2 | par3_pop (control) | 0.6052 | 0.5620 | 0.4034 | 0.6688 | 0.9668 | 0.1802 | 0.0111 | 0/108 |
| M | O2 | par3_g1 | 0.9643 | 0.8911 | 0.4176 | 1.5944 | 0.9688 | 0.3298 | 0.0116 | 0/108 |
| M | O2 | par14_g1 | 1.0388 | 0.9568 | 0.4660 | 1.6409 | 1.1243 | 0.3511 | 0.0485 | 0/108 |
| M | O2 | pt_interp | 0.9579 | 0.8849 | 0.4176 | 1.5797 | 0.9657 | 0.3283 | 0.0138 | 0/108 |
| M | O2 | lrn_mean | 0.7912 | 0.7274 | 0.3244 | 0.6155 | 1.5702 | 0.1879 | 0.0138 | 0/108 |
| M | O2 | lrn_noH | 0.4882 | 0.4620 | 0.3614 | 0.5896 | 0.7470 | 0.1281 | 0.0138 | 0/108 |
| M | O2 | lrn_param | 0.9291 | 0.8524 | 0.3864 | 1.0449 | 1.5916 | 0.1509 | 0.0138 | 0/108 |
| M | — | floor (reference) | 0.0135 | 0.0123 | 0.0017 | 0.0247 | 0.0118 | 0.0040 | — | 0/108 |
| W | O4 | lrn | 0.1139 | 0.1104 | 0.0836 | 0.2257 | 0.0700 | 0.0912 | 0.0124 | 1/108 |
| W | O4 | par3 (control) | 0.0666 | 0.0593 | 0.0317 | 0.0927 | 0.0437 | 0.0635 | 0.0378 | 0/108 |
| W | O4 | par14 | 0.0672 | 0.0598 | 0.0247 | 0.0903 | 0.0458 | 0.0655 | 0.0386 | 0/108 |
| W | O4 | par3_pop | 0.7468 | 0.6770 | 0.3520 | 1.4289 | 0.5151 | 0.2965 | 0.7121 | 0/108 |
| W | O4 | pt_interp | 0.1019 | 0.0952 | 0.0809 | 0.1049 | 0.0747 | 0.1261 | 0.0124 | 0/108 |
| W | O4 | lrn_mean | 0.1507 | 0.1428 | 0.1436 | 0.2191 | 0.0909 | 0.1422 | 0.0124 | 0/108 |
| W | O4 | lrn_noH | 0.1179 | 0.1148 | 0.0949 | 0.2292 | 0.0689 | 0.0998 | 0.0124 | 3/108 |
| W | O4 | lrn_param | 0.1312 | 0.1262 | 0.1121 | 0.2474 | 0.0733 | 0.1147 | 0.0124 | 1/108 |
| W | O2 | lrn | 0.5062 | 0.4677 | 0.3127 | 0.7793 | 0.4783 | 0.2610 | 0.0118 | 0/108 |
| W | O2 | par3_pop (control) | 0.7586 | 0.6889 | 0.3548 | 1.4173 | 0.5469 | 0.3116 | 0.0084 | 0/108 |
| W | O2 | par3_g1 | 0.5868 | 0.5396 | 0.3828 | 0.6633 | 0.7145 | 0.3825 | 0.0094 | 0/108 |
| W | O2 | par14_g1 | 0.6095 | 0.5607 | 0.4111 | 0.6891 | 0.7409 | 0.3984 | 0.0240 | 0/108 |
| W | O2 | pt_interp | 0.5897 | 0.5420 | 0.3845 | 0.6674 | 0.7205 | 0.3813 | 0.0118 | 0/108 |
| W | O2 | lrn_mean | 0.7534 | 0.6906 | 0.5595 | 1.0786 | 0.7911 | 0.3906 | 0.0118 | 0/108 |
| W | O2 | lrn_noH | 0.5203 | 0.4771 | 0.3060 | 0.8192 | 0.4768 | 0.2649 | 0.0118 | 0/108 |
| W | O2 | lrn_param | 0.5514 | 0.5132 | 0.3741 | 0.7237 | 0.5917 | 0.3389 | 0.0118 | 0/108 |
| W | — | floor (reference) | 0.0053 | 0.0047 | 1.5e-08 | 0.0084 | 0.0053 | 0.0022 | — | 0/108 |

- all-device = mean over the 36 simulated test devices of the per-device mean over 110/190/260 K (summary.json; a device with a failed cell is left out of that method's mean); combined = mean of the nMOS (15) and pMOS (21) family means; median over test devices.
- Methods: lrn = learned shared law (primary); par3 = per-device law x = c + (mu − c) w^g; par14 = cryoml.temperature family; par3_pop = law with population-pooled g; *_g1 = g = 1; pt_interp = linear interpolation of the per-temperature anchor fits; lrn_mean = population-mean correction without a network; lrn_noH = trained without 110/190/260 K; lrn_param = parameter-MSE loss; floor = per-temperature fit on the held-out data.
- O4 observes 77/150/225/300 K; O2 observes 77/300 K. Anchors: mean RRMS at the observed temperatures. Mean training-fit RRMS: tier M 0.0355, tier W 0.0289.


### E3b secondary quantities: lrn vs lrn_mean, lrn_noH and lrn_param, gap closed to the floor, paired geometric-mean ratio

| run | S | lrn vs lrn_mean | lrn vs lrn_noH | lrn vs lrn_param | gap closed to floor | floor all-device | geometric mean ratio lrn/control [CI] | devices lrn < control | lrn vs other parametric laws |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| tier M | O4 | 0.033 [-0.154, 0.161] | 0.008 [-0.081, 0.109] | 0.132 [-0.022, 0.237] | 0.3493 | 0.0135 | 0.639 [0.516, 0.793] | 29/36 | par14 0.334; par3_pop 0.545 |
| tier M | O2 | 0.375 [0.031, 0.533] | -0.013 [-0.030, 0.012] | 0.468 [0.172, 0.601] | 0.1872 | 0.0135 | 0.899 [0.784, 1.04] | 23/36 | par3_g1 0.487; par14_g1 0.524 |
| tier W | O4 | 0.218 [0.068, 0.348] | 0.030 [-0.065, 0.115] | 0.132 [0.036, 0.213] | -0.7230 | 0.0053 | 255 [23.1, 3.51e+03] | 7/35 | par14 -0.652; par3_pop 0.695 |
| tier W | O2 | 0.328 [0.033, 0.581] | 0.027 [-0.047, 0.072] | 0.082 [-0.051, 0.267] | 0.3350 | 0.0053 | 0.855 [0.664, 1.07] | 19/36 | par3_g1 0.137; par14_g1 0.169 |

- Secondary quantities listed in the pre-registration as reported, not decisive. Reductions 1 − mean(lrn)/mean(other) with a paired percentile bootstrap 95% CI over test devices (4000 resamples, seed 0) on devices finite for both. lrn_mean and the other-law columns are stored in summary.json; the lrn_noH and lrn_param columns are computed for this report from eval/*.json with the same bootstrap function (e3_temp_misspec.bootstrap_reduction).
- gap closed = (control − lrn) / (control − floor), all-device means on the paired devices.
- Geometric-mean ratio = exp(mean over devices of log(lrn / control)) with bootstrap CI and win count (prereg item d; not a decision input). In tier W O4 the control par3 is near the floor on many devices (the truth is inside its family), so per-device ratios are large.


### E3 vs E3b: held-out means, medians, R_S and rule outputs before and after the u0 fix

| run | S | lrn mean | control mean | pt_interp mean | lrn median | control median | R_S with CI | 1 − lrn/pt_interp | geometric-mean ratio lrn/control | paired devices | rule output of the run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E3 tier M | O4 | 9.8011 | 9.1225 | 10.1179 | 0.1676 | 0.3123 | -0.074 [-0.090, 0.504] | 0.0313 | 0.636 * | 36 | KILL |
| E3b tier M | O4 | 0.2778 | 0.4197 | 0.3689 | 0.1762 | 0.2983 | 0.338 [0.160, 0.495] | 0.2469 | 0.639 | 36 | CONTINUE |
| E3 tier M without nmos_L1_W3__2 (I2-0, diagnostic) | O4 | 0.2670 | 0.4233 | 0.3698 | 0.1589 | 0.3032 | 0.369 [0.168, 0.533] | 0.2780 |  | 35 | CONTINUE (not a verdict) |
| E3 tier M | O2 | 7.1018 | 6.8581 | 9.5924 | 0.3539 | 0.4059 | -0.036 [-0.055, 0.288] | 0.2596 | 0.885 * | 36 | KILL |
| E3b tier M | O2 | 0.4945 | 0.6052 | 0.9579 | 0.3605 | 0.4034 | 0.183 [0.019, 0.281] | 0.4838 | 0.899 | 36 | CONTINUE |
| E3 tier M without nmos_L1_W3__2 (I2-0, diagnostic) | O2 | 0.4835 | 0.6102 | 0.9642 | 0.3504 | 0.4003 | 0.208 [0.031, 0.305] | 0.4985 |  | 35 | CONTINUE (not a verdict) |
| E3 tier W | O4 | 0.1129 | 0.0692 | 0.0991 | 0.0790 | 0.0328 | -0.632 [-1.580, -0.143] | -0.1388 | 244 * | 35 | INCONCLUSIVE |
| E3b tier W | O4 | 0.1139 | 0.0683 | 0.0986 | 0.0836 | 0.0326 | -0.667 [-1.717, -0.135] | -0.1556 | 255 | 35 | INCONCLUSIVE |
| E3 tier W | O2 | 0.4903 | 0.7459 | 0.5900 | 0.2821 | 0.3435 | 0.343 [-0.022, 0.467] | 0.1689 | 0.845 * | 36 | INCONCLUSIVE |
| E3b tier W | O2 | 0.5062 | 0.7586 | 0.5897 | 0.3127 | 0.3548 | 0.333 [-0.034, 0.462] | 0.1416 | 0.855 | 36 | INCONCLUSIVE |

- E3 (queue jobs E3_full_M, E3_full_W, 2026-10-19 11:52-12:23) and E3b (E3b_full_M, E3b_full_W, 22:25-22:57) use the same settings, seeds, methods and rule. E3b differs by the u0 fix: 4 tier-M truths (3 training, 1 test: nmos_L1_W3 draws) and 2 tier-W truths regenerated, the nmos_L1_W3 and nmos_L0p25_W1p6 x_u0 fit bounds lowered to 0.02 below the switch, and the u0 check.
- * E3 geometric-mean ratio computed for this report with the E3b function (e3_temp_misspec.bootstrap_log_ratio) on the E3 eval files; E3 did not pre-register it.
- The I2-0 row removes one test device after the fact and keeps the switched truths in training; it is a diagnostic, not a verdict. The E3 KILL verdict stands; E3b is a separate pre-registered probe with rule output CONTINUE on tier M.


### Temperature benchmark (simulated, well-specified): paper RRMS per method, raw and after FD, all surrogate architectures

| method | stage | truth sets | observed all-device | observed combined | held-out all-device | held-out combined | 110 K | 190 K | 260 K | local parameter error RMS | failed device-set records |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct NGSpice (CMA-ES + TRF) | raw | 3 | 0.0129 | 0.0125 | 0.0362 | 0.0343 | 0.0759 | 0.0170 | 0.0157 | 0.4360 | 0/54 |
|  | + FD | 3 | 0.0015 | 0.0015 | 0.0117 | 0.0114 | 0.0299 | 0.0035 | 0.0018 | 0.2715 | 0/54 |
| direct MLP, one pass | raw | 3 | 0.0743 | 0.0737 | 0.0707 | 0.0692 | 0.0976 | 0.0654 | 0.0491 | 0.2294 | 0/54 |
|  | + FD | 3 | 0.0002 | 0.0002 | 0.0006 | 0.0007 | 0.0004 | 0.0008 | 0.0006 | 0.2102 | 0/54 |
| pipeline search on NGSpice | raw | 3 | 0.0005 | 0.0005 | 0.0021 | 0.0020 | 0.0045 | 0.0007 | 0.0011 | 0.1831 | 0/54 |
|  | + FD | 3 | 1.2e-05 | 1.1e-05 | 0.0004 | 0.0004 | 0.0010 | 2.9e-05 | 0.0002 | 0.1583 | 0/54 |
| surrogate search, paper MLP | raw | 3 | 0.0051 | 0.0050 | 0.0081 | 0.0078 | 0.0150 | 0.0049 | 0.0043 | 0.2344 | 0/54 |
|  | + FD | 3 | 1.5e-05 | 1.3e-05 | 0.0007 | 0.0007 | 0.0019 | 4.1e-05 | 0.0002 | 0.1916 | 0/54 |
| surrogate search, ResMLP | raw | 3 | 0.0073 | 0.0072 | 0.0105 | 0.0103 | 0.0164 | 0.0073 | 0.0077 | 0.2736 | 0/54 |
|  | + FD | 3 | 1.0e-05 | 9.6e-06 | 0.0003 | 0.0003 | 0.0004 | 2.1e-05 | 0.0003 | 0.2186 | 0/54 |
| surrogate search, DeepONet | raw | 3 | 0.0076 | 0.0074 | 0.0120 | 0.0117 | 0.0218 | 0.0070 | 0.0073 | 0.2586 | 0/54 |
|  | + FD | 3 | 6.8e-05 | 7.0e-05 | 0.0022 | 0.0021 | 0.0062 | 0.0001 | 0.0002 | 0.2172 | 0/54 |
| surrogate search, FNO | raw | 3 | 0.0187 | 0.0185 | 0.0227 | 0.0222 | 0.0347 | 0.0177 | 0.0155 | 0.4119 | 0/54 |
|  | + FD | 3 | 0.0003 | 0.0003 | 0.0012 | 0.0011 | 0.0024 | 0.0003 | 0.0009 | 0.2727 | 0/54 |

- Simulated well-specified benchmark: 3 truth sets x 18 device geometries, truth = the 14-number temperature family of cryoml.temperature (tnom 27 C); fits observe 77/150/225/300 K; scored at the held-out 110/190/260 K by NGSpice re-simulation. Values are means over the 3 sets of the all-device mean (18 devices) and combined (nMOS, pMOS family means). No measured I-V.
- failed = device-set records with a non-finite observed or held-out RRMS (strict). +FD = NGSpice finite-difference TRF polish (max_nfev 400) from the method's output. param. err. = RMS of the local (box-normalized) parameter error.
- FNO: trained 25 epochs (14402 s, best validation loss 0.00187 at the last epoch) in queue job temp_sur_fno (2026-10-17 23:37 - 10-18 03:38), which was killed after training (rc 137, low memory). The rerun temp_sur_fno_r (2026-10-19 22:56-23:13, 1002 s, rc 0) loaded the saved checkpoint, ran the held-out emulator error and validation passes with the --eval-batch memory fix (default 4096 rows, commit 168cf9f) and completed sets 0, 1 and 2 (18 devices each).
- ResMLP: 150 epochs (12028 s), queue job temp_sur_resmlp (2026-10-19 14:30-17:59). Table written by temp_study.py table (queue job temp_table, 2026-10-19 23:13).

