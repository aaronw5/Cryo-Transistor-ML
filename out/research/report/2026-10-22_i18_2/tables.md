# Tables, 2026-10-22 i18_2

Generated 2026-10-22 23:15 by `.venv/bin/python -I scripts/research/plot_explore_i18_2.py --date 2026-10-22`.

### I18-2 setup, BSIM4 clamp state C audit, amendment A1

| item | value |
|---|---|
| data | out/research/explore/I6_1/split.json DEV (45 devices, 7 groups); TEST 24 locked |
| card | BSIM4 4.5.0 N1/P1 CMOSedu 50 nm (I6-1), 43-parameter set and box of I6-1 |
| simulator | ngspice-41 native arm64 (I6-1), timeout 10 s |
| folds | leave-one-geometry-group-out (7), training = other 6 groups, both polarities, record load order |
| penalty grid | u_g, u_d in {0, 0.025, ..., 1.1} V (45 x 45), forward quadrant, Vb = 0, state-C starting card, dataset temperature |
| BSIM4 clamp | every deck writes the b4check.c clamp: a2 > 1 -> a2 = 1 and a1 = 0 if nonzero; a2 < 0.01 -> 0.01; card has no l/w/p A1/A2 terms |
| 4.13 K: starting cards clamped (a2 > 1) | 25 / 45 |
| 4.13 K: starting-card RRMS state L -> state C (all-device) | 0.06284 -> 0.07215 |
| 4.13 K: b4check messages (unclamped cards) | 25 distinct lines, all 'A2 ... larger than 1' |
| 4.13 K: b4check messages after the clamp | 0 |
| 295.0 K: starting cards clamped (a2 > 1) | 20 / 45 |
| 295.0 K: starting-card RRMS state L -> state C (all-device) | 0.02706 -> 0.04632 |
| 295.0 K: b4check messages (unclamped cards) | 20 distinct lines, all 'A2 ... larger than 1' |
| 295.0 K: b4check messages after the clamp | 0 |
| amendment A1 | |I_elem(V(d,s) = 0) - I_card(V(d,s) = 0)| <= 1e-12 A on the tight forward Q1 grid, I_card = drain current of the card + replica run at the same tolerance and bias (the current the element adds at V(d,s) = 0); the raw element value |I_elem(V(d,s) = 0)| and the card value are reported without a rule |
| Q3 circuits | not run: sky130 circuits at 1.8 V supply; TU Delft card 1 V (CMOSedu VDD=1V), measured range 1.1 V; core check recorded as not evaluated (count 0, label 'Q3 not run') |

- Pre-registration commit 2454b15; amendment A1 commit 1795e10.


### I18-2 validity

| check | criterion | value |
|---|---:|---|
| V1a state-L re-simulation of 90 I6-1 starting cards, max |d RRMS| | <= 1e-9 | 0.0e+00 (n 90) |
| V1b state C = state L on cards with a2 in [0.01, 1] | identical | identical (n 45) |
| V1c state-L refits vs I6-1 rows (C0_refit, M_bias v1; 2 devices, 4.13 K) | <= 1e-9 | 0.0e+00 (n 4) |
| V2 unit tests | pass | True |
| V3 rows per arm and temperature | 45 | 45 |
| V3 inner / full selection complete | yes | 4.13 K True / True, 295.0 K True / True |
| V4 penalty grids 45 x 45 finite, forward corner > 0 | 90 | True |
| VB Part B 4.13 K (45 M2 refits, 0 failed; unit tests; 45 rows) | pass | pass |
| VB Part B 295.0 K (45 M2 refits, 0 failed; unit tests; 45 rows) | pass | pass |

- Result: valid.


### I18-2 pre-registered rules per temperature: values, labels, predicted signs

| T | quantity | GMR [95% CI] | lower | label | prediction |
|---|---|---:|---:|---|---|
| 4.13 K | primary GMR(selected, C0_refit) | 0.826 [0.669, 1.014] | 30/45 | inconclusive | < 0: correct |
| 4.13 K | secondary GMR(selected, M_bias v1) | 0.908 [0.848, 0.965] | 33/45 | non-inferior to M_bias v1 | > 0: wrong |
| 4.13 K | GMR(M_bias v1, C0_refit) (reported) | 0.909 [0.766, 1.077] | 30/45 | - | < 0: correct |
| 4.13 K | Part B label | - | - | not deployable (Q3 not run) | deployable with restrictions (Q3 not run): wrong |
| 295.0 K | primary GMR(selected, C0_refit) | 0.829 [0.523, 1.247] | 29/45 | inconclusive | < 0: correct |
| 295.0 K | secondary GMR(selected, M_bias v1) | 1.022 [0.975, 1.056] | 22/45 | not shown non-inferior to M_bias v1 | > 0: correct |
| 295.0 K | GMR(M_bias v1, C0_refit) (reported) | 0.811 [0.532, 1.190] | 29/45 | - | < 0: correct |
| 295.0 K | Part B label | - | - | deployable with restrictions (Q3 not run) | deployable with restrictions (Q3 not run): correct |

- GMR: exp(mean ln(method / control)) over 45 DEV devices; cluster bootstrap over the 7 geometry groups, 10,000 resamples, 95% CI. Primary: CI upper < 1 and 0 failed: gain replicated; CI lower > 1: worse; else inconclusive. Secondary: CI upper <= 1.05: non-inferior.


### I18-2 held-out RRMS of every arm (45 TU Delft DEV devices, state C)

| T | arm | all-device | combined | median | GMR vs C0_refit (cluster CI) | device CI | GMR vs M_bias v1 (cluster CI) |
|---|---|---:|---:|---:|---:|---:|---:|
| 4.13 K | C0_refit | 0.06015 | 0.05319 | 0.04411 | 1 | - | 1.100 [0.928, 1.306] |
| 4.13 K | M_bias_v1 | 0.05341 | 0.04913 | 0.03955 | 0.909 [0.766, 1.077] | 0.909 [0.847, 0.978] | 1 |
| 4.13 K | sym_e1_l0 | 0.05090 | 0.04705 | 0.03741 | 0.883 [0.734, 1.056] | 0.883 [0.820, 0.954] | 0.972 [0.953, 0.989] |
| 4.13 K | sym_e1_l1 | 0.05072 | 0.04719 | 0.03910 | 0.884 [0.726, 1.077] | 0.884 [0.815, 0.962] | 0.972 [0.918, 1.018] |
| 4.13 K | sym_e1_l2 | 0.04759 | 0.04433 | 0.03800 | 0.840 [0.689, 1.014] | 0.840 [0.772, 0.913] | 0.924 [0.888, 0.949] |
| 4.13 K | sym_e1_l5 | 0.04920 | 0.04574 | 0.03617 | 0.836 [0.668, 1.039] | 0.836 [0.761, 0.919] | 0.919 [0.870, 0.975] |
| 4.13 K | sym_e1_l10 | 0.04834 | 0.04491 | 0.03739 | 0.826 [0.669, 1.014] | 0.826 [0.756, 0.905] | 0.908 [0.848, 0.965] |
| 4.13 K | selected | 0.04835 | 0.04492 | 0.03739 | 0.826 [0.669, 1.014] | 0.826 [0.756, 0.905] | 0.908 [0.848, 0.965] |
| 4.13 K | I6-1 stored (state L) C0_refit / M_bias | 0.05749 / 0.05328 |  |  |  |  |  |
| 295.0 K | C0_refit | 0.02649 | 0.02510 | 0.02556 | 1 | - | 1.233 [0.840, 1.879] |
| 295.0 K | M_bias_v1 | 0.02080 | 0.02096 | 0.01841 | 0.811 [0.532, 1.190] | 0.811 [0.680, 0.978] | 1 |
| 295.0 K | sym_e1_l0 | 0.02112 | 0.02090 | 0.01820 | 0.825 [0.558, 1.181] | 0.825 [0.701, 0.977] | 1.017 [0.961, 1.084] |
| 295.0 K | sym_e1_l1 | 0.02197 | 0.02219 | 0.02000 | 0.854 [0.561, 1.237] | 0.854 [0.721, 1.015] | 1.053 [1.003, 1.107] |
| 295.0 K | sym_e1_l2 | 0.02193 | 0.02207 | 0.01917 | 0.851 [0.547, 1.265] | 0.851 [0.716, 1.015] | 1.050 [1.003, 1.085] |
| 295.0 K | sym_e1_l5 | 0.02115 | 0.02142 | 0.01950 | 0.822 [0.528, 1.226] | 0.822 [0.687, 0.992] | 1.014 [0.983, 1.043] |
| 295.0 K | sym_e1_l10 | 0.02180 | 0.02215 | 0.01913 | 0.839 [0.534, 1.263] | 0.839 [0.698, 1.023] | 1.034 [0.986, 1.073] |
| 295.0 K | selected | 0.02149 | 0.02182 | 0.01879 | 0.829 [0.523, 1.247] | 0.829 [0.691, 0.999] | 1.022 [0.975, 1.056] |
| 295.0 K | I6-1 stored (state L) C0_refit / M_bias | 0.02553 / 0.01960 |  |  |  |  |  |

- Strict paper RRMS on the I6-1 frozen measured inclusion; held-out refit from the I6-1 starting card with g frozen (max_nfev 400, irls 2), clamp state C.


### I18-2 per geometry group (outer fold): mean held-out RRMS and lambda*

| T | group | devices | C0_refit | M_bias v1 | selected | lambda* (V) |
|---|---|---:|---:|---:|---:|---:|
| 4.13 K | nmos_W0p12_L0p04 | 8 | 0.08373 | 0.06889 | 0.05422 | 10 |
| 4.13 K | nmos_W0p12_L0p05 | 3 | 0.15195 | 0.14089 | 0.13959 | 5 |
| 4.13 K | nmos_W0p36_L0p04 | 2 | 0.06425 | 0.04728 | 0.03280 | 10 |
| 4.13 K | nmos_W1_L0p04 | 8 | 0.06108 | 0.04549 | 0.04488 | 10 |
| 4.13 K | nmos_W1_L1 | 8 | 0.06236 | 0.05282 | 0.04625 | 10 |
| 4.13 K | pmos_W1_L0p04 | 8 | 0.03144 | 0.03648 | 0.03378 | 10 |
| 4.13 K | pmos_W1_L1 | 8 | 0.02671 | 0.03210 | 0.03232 | 10 |
| 295.0 K | nmos_W0p12_L0p04 | 8 | 0.03997 | 0.02561 | 0.02592 | 5 |
| 295.0 K | nmos_W0p12_L0p05 | 3 | 0.02768 | 0.01034 | 0.00931 | 5 |
| 295.0 K | nmos_W0p36_L0p04 | 2 | 0.03571 | 0.01657 | 0.01515 | 10 |
| 295.0 K | nmos_W1_L0p04 | 8 | 0.03445 | 0.01864 | 0.01834 | 5 |
| 295.0 K | nmos_W1_L1 | 8 | 0.01476 | 0.02174 | 0.02344 | 2 |
| 295.0 K | pmos_W1_L0p04 | 8 | 0.02396 | 0.01788 | 0.01900 | 1 |
| 295.0 K | pmos_W1_L1 | 8 | 0.01656 | 0.02513 | 0.02690 | 1 |


### I18-2 nested selection: inner score (mean ln RRMS of the inner held-out refits) per outer fold and lambda

| T | held-out group | lambda 0 V | lambda 1 V | lambda 2 V | lambda 5 V | lambda 10 V | lambda* (V) |
|---|---|---:|---:|---:|---:|---:|---:|
| 4.13 K | nmos_W0p12_L0p04 | -3.0841 | -3.1119 | -3.1009 | -3.1404 | -3.1647 | 10 |
| 4.13 K | nmos_W0p12_L0p05 | -3.1874 | -3.2163 | -3.2187 | -3.2777 | -3.2520 | 5 |
| 4.13 K | nmos_W0p36_L0p04 | -3.1252 | -3.1502 | -3.1544 | -3.1649 | -3.1820 | 10 |
| 4.13 K | nmos_W1_L0p04 | -3.0836 | -3.0466 | -3.0436 | -3.0177 | -3.1069 | 10 |
| 4.13 K | nmos_W1_L1 | -3.1860 | -3.1896 | -3.2280 | -3.1180 | -3.2689 | 10 |
| 4.13 K | pmos_W1_L0p04 | -3.0194 | -3.0717 | -3.0649 | -3.0759 | -3.0814 | 10 |
| 4.13 K | pmos_W1_L1 | -3.0717 | -3.0704 | -3.0719 | -3.1356 | -3.1622 | 10 |
| 4.13 K | full data (Part B model) | -3.0482 | -3.0615 | -3.0485 | -3.1239 | -3.1385 | 10 |
| 295.0 K | nmos_W0p12_L0p04 | -3.7949 | -3.8617 | -3.8211 | -3.8844 | -3.8485 | 5 |
| 295.0 K | nmos_W0p12_L0p05 | -3.9064 | -3.8572 | -3.8512 | -3.9119 | -3.8823 | 5 |
| 295.0 K | nmos_W0p36_L0p04 | -3.9152 | -3.9407 | -3.9107 | -3.9596 | -3.9690 | 10 |
| 295.0 K | nmos_W1_L0p04 | -3.8970 | -3.9040 | -3.9113 | -3.9195 | -3.8872 | 5 |
| 295.0 K | nmos_W1_L1 | -4.0334 | -4.0628 | -4.1186 | -4.0901 | -4.0093 | 2 |
| 295.0 K | pmos_W1_L0p04 | -3.8564 | -3.8658 | -3.8234 | -3.8457 | -3.8451 | 1 |
| 295.0 K | pmos_W1_L1 | -3.9564 | -4.0082 | -3.9525 | -4.0008 | -3.9370 | 1 |
| 295.0 K | full data (Part B model) | -3.7684 | -3.8577 | -3.8097 | -3.8746 | -3.8229 | 5 |

- Inner folds: training groups sorted by (polarity, name), group i -> fold i mod 3; one member (seed 0); ties to the larger lambda.


### I18-2 negative corrected gds / gm on the penalty grids (% of points)

| T | arm | held-out gds < 0 | held-out gm < 0 | training gds < 0 | training gm < 0 |
|---|---|---:|---:|---:|---:|
| 4.13 K | M_bias_v1 | 1.09 | 1.42 | 0.00 | 1.84 |
| 4.13 K | sym_e1_l0 | 1.12 | 1.42 | 0.00 | 1.84 |
| 4.13 K | sym_e1_l1 | 1.24 | 1.37 | 0.02 | 0.83 |
| 4.13 K | sym_e1_l2 | 1.41 | 1.19 | 0.01 | 0.71 |
| 4.13 K | sym_e1_l5 | 0.98 | 0.86 | 0.00 | 0.48 |
| 4.13 K | sym_e1_l10 | 0.99 | 0.80 | 0.00 | 0.32 |
| 4.13 K | selected | 0.99 | 0.80 | 0.00 | 0.35 |
| 4.13 K | card only | 0.000 | 1.846 |  |  |
| 295.0 K | M_bias_v1 | 6.18 | 0.04 | 4.98 | 0.04 |
| 295.0 K | sym_e1_l0 | 6.18 | 0.02 | 4.97 | 0.02 |
| 295.0 K | sym_e1_l1 | 6.05 | 0.01 | 0.80 | 0.01 |
| 295.0 K | sym_e1_l2 | 6.06 | 0.01 | 0.40 | 0.01 |
| 295.0 K | sym_e1_l5 | 6.05 | 0.01 | 0.11 | 0.01 |
| 295.0 K | sym_e1_l10 | 6.23 | 0.01 | 0.03 | 0.01 |
| 295.0 K | selected | 6.05 | 0.01 | 0.18 | 0.01 |
| 295.0 K | card only | 0.000 | 0.048 |  |  |

- Grid 45 x 45, forward quadrant 0 .. 1.1 V, forward current > 1 nA; mean over folds.


### I18-2 Part B (I17-1 rules, amendment A1) per temperature, 45 devices

| check | kind | threshold | 4.13 K | 295.0 K |
|---|---:|---:|---:|---:|
| Q1 tight equivalence violations | core | 0 | 0 | 0 |
| Q1 element runs: messages + non-finite | core | 0 | 3 | 0 |
| Q1 element-added |I| at V(d,s) = 0, tight (A1) | core | <= 1e-12 A | 9.7e-19 | 1.22e-19 |
| Verilog-A g vs Python | core | <= 1e-09 | 4.44e-16 | 3.33e-16 |
| Verilog-A g exchanged vs forward | core | <= 1e-12 | 1.72e-15 | 2.22e-16 |
| exchange asymmetry | core | <= 0.001 | 3.78e-10 | 4.18e-13 |
| Q2 derivative vs element identity | core | <= 0.001 | 1.04e-07 | 2.82e-08 |
| Q2 capacitance error | core | <= 0.001 | 0 | 0 |
| gm curves with introduced sign change | core | 0 | 0 | 0 |
| default-tolerance relative difference | restriction | <= 0.001 | 0.00164 | 0.00146 |
| gds curves with introduced sign change | restriction | 0 | 0 | 1 |
| runtime ratio, DC grid | restriction | <= 2 | 1.95 | 1.93 |
| raw element |I| at V(d,s) = 0, tight (reported) | reported | - | 9.58e-09 | 4.07e-08 |
| card + replica |I| at V(d,s) = 0 (reported) | reported | - | 9.58e-09 | 4.07e-08 |
| card non-channel share of |I|, forward (reported) | reported | - | 1 | 0.948 |
| Q3 circuits | core | - | not run | not run |
| label |  |  | not deployable (Q3 not run) | deployable with restrictions (Q3 not run) |

- 4.13 K: lambda*_full 10 V; core failures ['Q1 convergence / non-finite']; restrictions ['default tolerance above 1e-3']; gds curves []; replica non-channel outputs max 0.0.
- 295.0 K: lambda*_full 5 V; core failures none; restrictions ['default tolerance above 1e-3', 'gds sign changes introduced']; gds curves [['t40_nmos_W1_L1_d134', 0.725]]; replica non-channel outputs max 0.0.


### I18-2 element-deck RRMS (ngspice-41 OSDI) vs Python rows (reported, no rule)

| T | pairs | element deck | Python rerun | stored Python | card only | max |element - stored| | messages |
|---|---|---:|---:|---:|---:|---:|---:|
| 4.13 K | M2 cards + full-data g (in sample) | 0.04108 | 0.04108 | 0.04108 | 0.06277 | 2.5e-04 | 0 |
| 4.13 K | selected held-out pairs | 0.04836 | 0.04835 | 0.04835 | 0.06237 | 1.4e-04 | 0 |
| 295.0 K | M2 cards + full-data g (in sample) | 0.01662 | 0.01662 | 0.01662 | 0.04566 | 6.6e-05 | 0 |
| 295.0 K | selected held-out pairs | 0.02150 | 0.02149 | 0.02149 | 0.04289 | 1.5e-04 | 0 |

- All-device means over 45 devices. Element deck: card + replica + OSDI module on the measured sweeps; Python: card x exp(g).

