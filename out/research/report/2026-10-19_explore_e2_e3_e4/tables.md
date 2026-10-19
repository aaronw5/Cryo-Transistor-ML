# Tables, 2026-10-19 explore_e2_e3_e4

Generated 2026-10-19 14:14 by `PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_e2_e3_e4.py --date 2026-10-19` from result files only. RRMS is the paper metric with frozen published-baseline curve inclusion, strict.

### E2 decision: one-pass seven-parameter network with output augmentation, pre-registered checks on REAL_DEV raw RRMS

| check | pre-registered rule | value, seeds 0-2 | value, seed 0 (light run) | outcome, seeds 0-2 (decision) | outcome, seed 0 | role |
|---|---|---:|---:|---:|---:|---|
| C1 | R_raw(S) <= 0.70 × R_raw(N0) | 1.742 | 1.145 | no | no | continue (all of C1-C4) |
| C2 | (R(S) − D) <= 0.70 × (R(N0) − D), D = 0.1735 | 1.754 | 1.147 | no | no | continue (all of C1-C4) |
| C3 | S < N0 on >= 8 of 12 devices | 6/12 | 6/12 | no | no | continue (all of C1-C4) |
| C4 guard | tier a: R(S) <= 1.5 × R(N0) + 0.01 | 14.89 vs 20.56 | 18.29 vs 22.84 | yes | yes | continue (all of C1-C4) |
| K1 | R_raw(S) > 0.85 × R_raw(N0) | 1.742 | 1.145 | met | met | kill (any of K1, K2) |
| K2 | S < N0 on fewer than 6 of 12 devices | 6/12 | 6/12 | not met | not met | kill (any of K1, K2) |
| attribution | R_raw(S) <= 0.90 × R_raw(M) | 3.126 | 1.501 | no | no | secondary, not a decision input |
| verdict |  |  |  | KILL | KILL (exploratory) |  |

- R_raw(arm) = mean over the 12 REAL_DEV devices of the per-device penalized paper RRMS (PENALTY 10 per failed included curve) of the predicted seven-parameter card, one forward pass + one NGSpice validation, averaged over seeds per device first. Seeds 0-2: R(N0) 11.241, R(M) 6.265, R(S) 19.581. Seed 0: R(N0) 12.475, R(M) 9.514, R(S) 14.282.
- D = 0.173493: direct NGSpice seven-parameter fit + FD on REAL_DEV (fixed before the pre-registration). tier a value = mean of the set-0 and set-1 means (36 targets).
- The seed-0 column reproduces the light run of commit 773abe3 from the same seed-0 files; the decision uses seeds 0-2 (queue jobs E2_full and E2_fd_sim, 2026-10-19 11:26-11:52).


### E2 raw and FD-polished RRMS per arm on REAL_DEV and the simulated sets

| stage | arm | seeds | REAL_DEV all-device | REAL_DEV combined | REAL_DEV nMOS | REAL_DEV pMOS | REAL_DEV median | REAL_DEV valid-only mean | REAL_DEV failed device-seed pairs | tier a mean | tier a median | tier c mean | tier c median | simulated failed target-seed pairs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw (one pass) | N0 | 3 | 11.2406 | 9.7376 | 0.7197 | 18.7555 | 5.4904 | 8.7937 | 9/36 | 13.6974 | 0.6617 | 7.0931 | 0.6390 | 2 |
| raw (one pass) | M | 3 | 6.2649 | 5.5129 | 1.0010 | 10.0248 | 2.7951 | 4.9056 | 6/36 | 6.1158 | 0.6042 | 36.3519 | 0.5457 | 1 |
| raw (one pass) | S | 3 | 19.5810 | 16.9552 | 1.2006 | 32.7099 | 1.6214 | 20.6297 | 4/36 | 14.8896 | 0.9480 | 10.3585 | 1.1994 | 0 |
| raw (one pass) | nearest bank | 1 | 0.9045 | 0.9787 | 1.4240 | 0.5335 | 0.3957 | 0.9045 | 0/12 | 0.2855 | 0.2691 | 0.2802 | 0.2558 | 0 |
| FD polish (100 nfev) | N0 | 3 | 1.0666 | 0.9474 | 0.2317 | 1.6631 | 0.5005 | 0.3840 | 8/36 | 0.3075 | 0.0083 | 0.2523 | 0.0634 | 5 |
| FD polish (100 nfev) | M | 3 | 0.5206 | 0.4695 | 0.1630 | 0.7760 | 0.2408 | 0.2924 | 2/36 | 0.2861 | 0.0066 | 0.1876 | 0.0621 | 1 |
| FD polish (100 nfev) | S | 3 | 0.7114 | 0.6350 | 0.1760 | 1.0939 | 0.2689 | 0.5263 | 3/36 | 0.2412 | 0.0084 | 0.1827 | 0.0660 | 1 |
| FD polish (100 nfev) | nearest bank | 1 | 0.3102 | 0.3104 | 0.3117 | 0.3090 | 0.2372 | 0.3102 | 0/12 | 0.0280 | 0.0060 | 0.0675 | 0.0622 | 0 |
| reference | released 77 K card | — | 0.2848 | 0.2608 | 0.1172 | 0.4044 | 0.2304 | 0.2848 | 0/12 | — | — | — | — | — |
| reference | D: direct NGSpice 7 + FD | — | 0.1735 | 0.1600 | 0.0793 | 0.2407 | 0.1446 | 0.1735 | 0/12 | — | — | — | — | — |
| reference | true seven values (tier c) | — | — | — | — | — | — | — |  | 0.0000 | 0.0000 | 0.5704 | 0.4156 | — |

- Per-device value = seed mean of the penalized paper RRMS (strict frozen inclusion; PENALTY 10 per failed included curve). all-device = mean over 12 devices; combined = mean of the nMOS (5) and pMOS (7) family means; valid-only = mean of the strict RRMS over device-seed pairs with no failed included curve.
- FD polish: cryoml.ngspice_fit.polish in the round-1 seven box from the predicted card, max_nfev 100, diff_step 0.01, irls_rounds 1, identical for every arm (secondary, not a decision input). FD on the simulated sets was run by queue job E2_fd_sim.
- tier a / tier c mean = mean of the set-0 and set-1 means (18 devices each); medians likewise. Tier c targets for S are partly in-distribution by construction (same component types as the tier c generator).
- The nearest bank sample uses no trained network: the device's own seven_r1 bank sample closest to the target in standardized signed-log features.


### E3 decision: learned shared temperature law (lrn) vs parametric laws, pre-registered checks

| run | observed temperatures S | control | R_S = 1 − lrn/control | R_S bootstrap 95% CI | R_S >= 0.25 | CI lower bound > 0 | 1 − lrn/pt_interp | lrn >= 10% below pt_interp | max failed-cell fraction (lrn, control, pt_interp) | failed <= 10% | continue for S | R_S < 0.10 | lrn worse than pt_interp | rule output for the run |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tier M (decision) | O4 | par3 | -0.0744 | [-0.090, 0.504] | no | no | 0.0313 | no | 0.000 | yes | no | yes | no | KILL |
| tier M (decision) | O2 | par3_pop | -0.0355 | [-0.055, 0.288] | no | no | 0.2596 | yes | 0.000 | yes | no | yes | no | KILL |
| tier W (secondary) | O4 | par3 | -0.6319 | [-1.580, -0.143] | no | no | -0.1388 | no | 0.009 | yes | no | yes | yes | INCONCLUSIVE |
| tier W (secondary) | O2 | par3_pop | 0.3427 | [-0.022, 0.467] | yes | no | 0.1689 | yes | 0.000 | yes | no | no | no | INCONCLUSIVE |

- Pre-registered rule (prereg.json): CONTINUE if for at least one S: R_S >= 0.25 AND bootstrap 95% CI lower bound of R_S > 0 AND lrn >= 10% below pt_interp in the same S AND failed-cell fraction <= 10% for lrn, control_S and pt_interp. KILL if R_S < 0.10 for both S, OR lrn worse than pt_interp for both S. Otherwise INCONCLUSIVE.
- Decision run: tier M (E3_full_M, 12 REAL_DEV geometries, 72 training and 36 test devices). Tier W (E3_full_W, truth inside the parametric family) is a secondary check; its rule output is reported, not used for the decision.
- Metric: frozen-inclusion paper RRMS of the NGSpice re-simulated card against the simulated truth curves, mean over held-out 110/190/260 K per device, all-device mean over 36 test devices (devices finite for all compared methods; tier W O4 has 35). Paired percentile bootstrap over test devices, 4000 resamples.


### E3 held-out-temperature RRMS per method, tiers M and W

| tier | S | method | held-out all-device mean | held-out combined | held-out median | mean without nmos_L1_W3__2 (diagnostic) | 110 K | 190 K | 260 K | observed-temperature mean | failed cells |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M | O4 | lrn | 9.8011 | 11.6933 | 0.1676 | 0.2670 | 28.9441 | 0.3739 | 0.0853 | 0.0150 | 0/108 |
| M | O4 | par3 (control) | 9.1225 | 10.8350 | 0.3123 | 0.4233 | 26.7865 | 0.3803 | 0.2007 | 0.1719 | 0/108 |
| M | O4 | par14 | 9.0714 | 10.7754 | 0.3294 | 0.4200 | 26.6221 | 0.3780 | 0.2140 | 0.1882 | 0/108 |
| M | O4 | par3_pop | 7.3763 | 8.6855 | 0.4069 | 0.6131 | 20.9707 | 0.9693 | 0.1888 | 0.3265 | 0/108 |
| M | O4 | pt_interp | 10.1179 | 12.0393 | 0.2555 | 0.3698 | 29.8545 | 0.4060 | 0.0932 | 0.0150 | 0/108 |
| M | O4 | lrn_mean | 9.8705 | 11.7725 | 0.1861 | 0.2880 | 29.0711 | 0.4632 | 0.0771 | 0.0150 | 0/108 |
| M | O4 | lrn_noH | 9.9226 | 11.8365 | 0.1783 | 0.2802 | 29.2913 | 0.3801 | 0.0964 | 0.0150 | 0/108 |
| M | O4 | lrn_param | 10.1061 | 12.0468 | 0.2166 | 0.2973 | 29.7894 | 0.4311 | 0.0979 | 0.0150 | 0/108 |
| M | O2 | lrn | 7.1018 | 8.4022 | 0.3539 | 0.4835 | 20.4202 | 0.7482 | 0.1368 | 0.0141 | 0/108 |
| M | O2 | par3_pop (control) | 6.8581 | 8.0646 | 0.4059 | 0.6102 | 19.4136 | 0.9720 | 0.1888 | 0.0113 | 0/108 |
| M | O2 | par3_g1 | 8.9318 | 10.4520 | 0.4176 | 0.9705 | 25.5301 | 0.9395 | 0.3257 | 0.0120 | 0/108 |
| M | O2 | par14_g1 | 9.8795 | 11.5657 | 0.4660 | 1.0451 | 28.1943 | 1.0945 | 0.3497 | 0.0502 | 0/108 |
| M | O2 | pt_interp | 9.5924 | 11.2463 | 0.4176 | 0.9642 | 27.5158 | 0.9371 | 0.3243 | 0.0141 | 0/108 |
| M | O2 | lrn_mean | 8.9654 | 10.5364 | 0.3168 | 0.8024 | 25.1600 | 1.5474 | 0.1887 | 0.0141 | 0/108 |
| M | O2 | lrn_noH | 7.2628 | 8.3701 | 0.3845 | 0.4991 | 20.2988 | 0.7861 | 0.1399 | 0.0141 | 1/108 |
| M | O2 | lrn_param | 8.9701 | 10.4787 | 0.4176 | 1.0159 | 25.0714 | 1.6759 | 0.1630 | 0.0141 | 0/108 |
| M | — | floor (reference) | 0.0136 | 0.0125 | 0.0017 | 0.0138 | 0.0250 | 0.0118 | 0.0041 | — | 0/108 |
| W | O4 | lrn | 0.1129 | 0.1090 | 0.0790 | — | 0.2139 | 0.0717 | 0.0913 | 0.0123 | 1/108 |
| W | O4 | par3 (control) | 0.0674 | 0.0602 | 0.0318 | — | 0.0946 | 0.0442 | 0.0635 | 0.0378 | 0/108 |
| W | O4 | par14 | 0.0679 | 0.0606 | 0.0315 | — | 0.0920 | 0.0463 | 0.0655 | 0.0385 | 0/108 |
| W | O4 | par3_pop | 0.7335 | 0.6611 | 0.3290 | — | 1.4143 | 0.5013 | 0.2849 | 0.7051 | 0/108 |
| W | O4 | pt_interp | 0.1024 | 0.0959 | 0.0809 | — | 0.1066 | 0.0751 | 0.1256 | 0.0123 | 0/108 |
| W | O4 | lrn_mean | 0.1496 | 0.1414 | 0.1351 | — | 0.2102 | 0.0897 | 0.1490 | 0.0123 | 0/108 |
| W | O4 | lrn_noH | 0.1215 | 0.1179 | 0.1050 | — | 0.2281 | 0.0720 | 0.1082 | 0.0123 | 3/108 |
| W | O4 | lrn_param | 0.1302 | 0.1237 | 0.0957 | — | 0.2101 | 0.0662 | 0.1143 | 0.0123 | 0/108 |
| W | O2 | lrn | 0.4903 | 0.4489 | 0.2821 | — | 0.7466 | 0.4777 | 0.2467 | 0.0116 | 0/108 |
| W | O2 | par3_pop (control) | 0.7459 | 0.6737 | 0.3435 | — | 1.4014 | 0.5348 | 0.3016 | 0.0082 | 0/108 |
| W | O2 | par3_g1 | 0.5870 | 0.5399 | 0.3828 | — | 0.6619 | 0.7149 | 0.3841 | 0.0093 | 0/108 |
| W | O2 | par14_g1 | 0.6087 | 0.5598 | 0.4111 | — | 0.6866 | 0.7406 | 0.3989 | 0.0235 | 0/108 |
| W | O2 | pt_interp | 0.5900 | 0.5423 | 0.3845 | — | 0.6648 | 0.7217 | 0.3835 | 0.0116 | 0/108 |
| W | O2 | lrn_mean | 0.7448 | 0.6803 | 0.5045 | — | 1.0376 | 0.7899 | 0.4071 | 0.0116 | 0/108 |
| W | O2 | lrn_noH | 0.5066 | 0.4627 | 0.2893 | — | 0.7977 | 0.4749 | 0.2473 | 0.0116 | 0/108 |
| W | O2 | lrn_param | 0.5429 | 0.4996 | 0.3767 | — | 0.6792 | 0.6126 | 0.3368 | 0.0116 | 0/108 |
| W | — | floor (reference) | 0.0053 | 0.0047 | 1.5e-08 | — | 0.0084 | 0.0053 | 0.0022 | — | 0/108 |

- all-device = mean over the 36 test devices of the per-device mean over 110/190/260 K (from summary.json); combined = mean of the nMOS (15) and pMOS (21) family means; median over test devices. Means and medians in the combined/median/excl. columns use devices with all three held-out cells finite.
- excl. #2 = tier M mean with test device nmos_L1_W3__2 removed. Diagnostic only; the pre-registered aggregate is the all-device mean. That device has held-out RRMS 226-353 for every method (floor 0.0070); its 110 K cell is 676-1060 for every method.
- Methods: lrn = learned shared law (primary); par3 = per-device law x = c + (mu − c) w^g; par14 = cryoml.temperature family; par3_pop = law with population-pooled g; *_g1 = g = 1; pt_interp = linear interpolation of the per-temperature anchor fits; lrn_mean = population-mean correction without a network; lrn_noH = trained without 110/190/260 K; lrn_param = parameter-MSE loss; floor = per-temperature fit on the held-out data.
- O4 observes 77/150/225/300 K; O2 observes 77/300 K. Anchor column: mean RRMS at the observed temperatures.


### E4 decision (43 parameters): predictive intervals and accuracy on held-out bias curves, pre-registered checks per candidate

| split | candidate | coverage of 90% interval | width of 90% interval | width limit 4 × single RRMS | C1 | held-out RRMS | limit 0.95 × single | devices below single | C2 | C3 learned <= crossinterp | C1 and C2 and C3 | rule output for the split |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vd (decision) | ens | 0.7438 | 1.0060 | 1.6708 | yes | 0.4418 | 0.3968 | 6/12 | no | — | no | PARTIAL_CALIBRATION_ONLY |
| vd (decision) | laplace | 0.9000 | 7.4447 | 1.6708 | no | 500.2099 | 0.3968 | 2/12 | no | — | no |  |
| vd (decision) | learned | 0.9631 | 5846.1190 | 1.6708 | no | 10.8691 | 0.3968 | 1/12 | no | no (10.869 vs 0.098) | no |  |
| sparse (secondary) | ens | 0.6757 | 1.0245 | 1.5779 | no | 0.4322 | 0.3747 | 5/12 | no | — | no | KILL |
| sparse (secondary) | laplace | 0.9259 | 2062.7893 | 1.5779 | no | 5143.1805 | 0.3747 | 0/12 | no | — | no |  |
| sparse (secondary) | learned | 0.9806 | 1961.5900 | 1.5779 | no | 3.0887 | 0.3747 | 1/12 | no | no (3.089 vs 0.272) | no |  |
| vg (secondary) | ens | 0.7965 | 0.3066 | 0.8281 | yes | 0.2070 | 0.1967 | 5/12 | no | — | no | PARTIAL_CALIBRATION_ONLY |
| vg (secondary) | laplace | 0.9458 | 2.0404 | 0.8281 | no | 0.6146 | 0.1967 | 2/12 | no | — | no |  |
| vg (secondary) | learned | 0.9649 | 0.4406 | 0.8281 | no | 0.0664 | 0.1967 | 11/12 | yes | no (0.066 vs 0.047) | no |  |

- Pre-registered rule (prereg.json, amendments 1 and 2): C1 = all-device mean coverage of the central 90% interval in [0.70, 0.95] AND all-device mean width <= 4 × single's held-out RRMS; C2 = held-out RRMS <= 0.95 × single (valid-only and paired) AND lower than single on >= 8 of 12 devices; C3 (learned only) = held-out RRMS <= crossinterp's (valid-only and paired). CONTINUE if a candidate meets C1, C2 and C3; PARTIAL_CALIBRATION_ONLY if no candidate meets all but one meets C1; KILL if none meets C1.
- Decision run: p43, split vd, 12 REAL_DEV devices, --select strict (queue job E4_p43_vd, stored decision block). Splits sparse and vg are secondary: the same decide() function was applied to their stored strict summaries for this report.
- single held-out RRMS (all-device): vd 0.4177, sparse 0.3945, vg 0.2070. Width is mean (hi − lo) / den, in the same units as the per-curve RRMS.
- The seven-parameter runs (p7) are not included: at report time p7 vd and p7 vg had finished and p7 sparse was running; p7 is reported when all three splits finish.


### E4 held-out-curve RRMS and interval metrics per method, 43 parameters, splits vd, sparse, vg

| split | member selection | method | all-device | combined | nMOS | pMOS | median | finite devices | devices below single | coverage 0.5 | coverage 0.9 | width 0.9 | interval score 0.9 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vd | strict | single | 0.4177 | 0.3751 | 0.1192 | 0.6309 | 0.1985 | 12/12 | — | — | — | — | — |
| vd | strict | ens | 0.4418 | 0.3968 | 0.1267 | 0.6668 | 0.4122 | 12/12 | 6 | 0.4562 | 0.7438 | 1.0060 | 2.9447 |
| vd | strict | laplace | 500.2099 | 428.8427 | 0.6397 | 857.0458 | 0.7761 | 12/12 | 2 | 0.5564 | 0.9000 | 7.4447 | 8.2961 |
| vd | strict | conformal | 0.4177 | 0.3751 | 0.1192 | 0.6309 | 0.1985 | 12/12 | 0 | 0.5459 | 0.9631 | 5846.7453 | 5847.4827 |
| vd | strict | crossinterp | 0.0977 | 0.0897 | 0.0416 | 0.1378 | 0.0604 | 12/12 | 12 | — | — | — | — |
| vd | strict | crossinterp_conformal | 0.0977 | 0.0897 | 0.0416 | 0.1378 | 0.0604 | 12/12 | 12 | 0.5406 | 0.9706 | 1922.8261 | 1923.1486 |
| vd | strict | learned | 10.8691 | 9.8197 | 3.5233 | 16.1161 | 4.3984 | 12/12 | 1 | 0.5607 | 0.9631 | 5846.1190 | 5846.8571 |
| vd | feasible | single | 0.4177 | 0.3751 | 0.1192 | 0.6309 | 0.1985 | 12/12 | — | — | — | — | — |
| vd | feasible | ens | 0.4418 | 0.3968 | 0.1267 | 0.6668 | 0.4122 | 12/12 | 6 | 0.4562 | 0.7438 | 1.0060 | 2.9447 |
| vd | feasible | laplace | 500.2099 | 428.8427 | 0.6397 | 857.0458 | 0.7761 | 12/12 | 2 | 0.5564 | 0.9000 | 7.4447 | 8.2961 |
| vd | feasible | conformal | 0.4177 | 0.3751 | 0.1192 | 0.6309 | 0.1985 | 12/12 | 0 | 0.5459 | 0.9631 | 5846.7453 | 5847.4827 |
| vd | feasible | crossinterp | 0.0977 | 0.0897 | 0.0416 | 0.1378 | 0.0604 | 12/12 | 12 | — | — | — | — |
| vd | feasible | crossinterp_conformal | 0.0977 | 0.0897 | 0.0416 | 0.1378 | 0.0604 | 12/12 | 12 | 0.5406 | 0.9706 | 1922.8261 | 1923.1486 |
| vd | feasible | learned | 10.8691 | 9.8197 | 3.5233 | 16.1161 | 4.3984 | 12/12 | 1 | 0.5607 | 0.9631 | 5846.1190 | 5846.8571 |
| sparse | strict | single | 0.3945 | 0.3579 | 0.1385 | 0.5773 | 0.3607 | 12/12 | — | — | — | — | — |
| sparse | strict | ens | 0.4322 | 0.3901 | 0.1378 | 0.6425 | 0.3349 | 12/12 | 5 | 0.4681 | 0.6757 | 1.0245 | 4.1810 |
| sparse | strict | laplace | 5143.1805 | 4408.6277 | 1.3112 | 8815.9442 | 0.8697 | 12/12 | 0 | 0.6177 | 0.9259 | 2062.7893 | 2063.0019 |
| sparse | strict | conformal | 0.3945 | 0.3579 | 0.1385 | 0.5773 | 0.3607 | 12/12 | 0 | 0.5442 | 0.9789 | 1961.6765 | 1962.0674 |
| sparse | strict | crossinterp | 0.2718 | 0.2517 | 0.1308 | 0.3726 | 0.2545 | 12/12 | 11 | — | — | — | — |
| sparse | strict | crossinterp_conformal | 0.2718 | 0.2517 | 0.1308 | 0.3726 | 0.2545 | 12/12 | 11 | 0.5485 | 0.9806 | 518.6441 | 518.7426 |
| sparse | strict | learned | 3.0887 | 2.8622 | 1.5033 | 4.2212 | 1.8596 | 12/12 | 1 | 0.5684 | 0.9806 | 1961.5900 | 1961.9784 |
| sparse | feasible | single | 0.3945 | 0.3579 | 0.1385 | 0.5773 | 0.3607 | 12/12 | — | — | — | — | — |
| sparse | feasible | ens | 0.4322 | 0.3901 | 0.1378 | 0.6425 | 0.3349 | 12/12 | 5 | 0.4681 | 0.6757 | 1.0245 | 4.1810 |
| sparse | feasible | laplace | 5143.1805 | 4408.6277 | 1.3112 | 8815.9442 | 0.8697 | 12/12 | 0 | 0.6177 | 0.9259 | 2062.7893 | 2063.0019 |
| sparse | feasible | conformal | 0.3945 | 0.3579 | 0.1385 | 0.5773 | 0.3607 | 12/12 | 0 | 0.5442 | 0.9789 | 1961.6765 | 1962.0674 |
| sparse | feasible | crossinterp | 0.2718 | 0.2517 | 0.1308 | 0.3726 | 0.2545 | 12/12 | 11 | — | — | — | — |
| sparse | feasible | crossinterp_conformal | 0.2718 | 0.2517 | 0.1308 | 0.3726 | 0.2545 | 12/12 | 11 | 0.5485 | 0.9806 | 518.6441 | 518.7426 |
| sparse | feasible | learned | 3.0887 | 2.8622 | 1.5033 | 4.2212 | 1.8596 | 12/12 | 1 | 0.5684 | 0.9806 | 1961.5900 | 1961.9784 |
| vg | strict | single | 0.2070 | 0.1867 | 0.0646 | 0.3088 | 0.1355 | 12/12 | — | — | — | — | — |
| vg | strict | ens | 0.2070 | 0.1862 | 0.0614 | 0.3111 | 0.1382 | 12/12 | 5 | 0.5834 | 0.7965 | 0.3066 | 1.6945 |
| vg | strict | laplace | 0.6146 | 0.6156 | 0.6215 | 0.6096 | 0.3714 | 12/12 | 2 | 0.6454 | 0.9458 | 2.0404 | 2.4184 |
| vg | strict | conformal | 0.2070 | 0.1867 | 0.0646 | 0.3088 | 0.1355 | 12/12 | 0 | 0.5455 | 0.9791 | 2.5248 | 2.6146 |
| vg | strict | crossinterp | 0.0466 | 0.0434 | 0.0239 | 0.0628 | 0.0480 | 12/12 | 12 | — | — | — | — |
| vg | strict | crossinterp_conformal | 0.0466 | 0.0434 | 0.0239 | 0.0628 | 0.0480 | 12/12 | 12 | 0.5462 | 0.9791 | 0.4911 | 0.5145 |
| vg | strict | learned | 0.0664 | 0.0665 | 0.0673 | 0.0657 | 0.0377 | 12/12 | 11 | 0.5714 | 0.9649 | 0.4406 | 0.4698 |
| vg | feasible | single | 0.2070 | 0.1867 | 0.0646 | 0.3088 | 0.1355 | 12/12 | — | — | — | — | — |
| vg | feasible | ens | 0.2070 | 0.1862 | 0.0614 | 0.3111 | 0.1382 | 12/12 | 5 | 0.5834 | 0.7965 | 0.3066 | 1.6945 |
| vg | feasible | laplace | 0.6146 | 0.6156 | 0.6215 | 0.6096 | 0.3714 | 12/12 | 2 | 0.6454 | 0.9458 | 2.0404 | 2.4184 |
| vg | feasible | conformal | 0.2070 | 0.1867 | 0.0646 | 0.3088 | 0.1355 | 12/12 | 0 | 0.5455 | 0.9791 | 2.5248 | 2.6146 |
| vg | feasible | crossinterp | 0.0466 | 0.0434 | 0.0239 | 0.0628 | 0.0480 | 12/12 | 12 | — | — | — | — |
| vg | feasible | crossinterp_conformal | 0.0466 | 0.0434 | 0.0239 | 0.0628 | 0.0480 | 12/12 | 12 | 0.5462 | 0.9791 | 0.4911 | 0.5145 |
| vg | feasible | learned | 0.0664 | 0.0665 | 0.0673 | 0.0657 | 0.0377 | 12/12 | 11 | 0.5714 | 0.9649 | 0.4406 | 0.4698 |

- all-device = mean over the 12 REAL_DEV devices (strict: a non-finite held-out prediction makes the device NaN; finite = devices with a finite value; 0 single-card failures in all three splits); combined = mean of the nMOS (5) and pMOS (7) family means.
- Held-out curves: vd = Id-Vd curves except the highest gate bias; sparse = every included curve except Id-Vg at Vd 0.01/1.85 and Id-Vd at the highest gate bias; vg = Id-Vg at Vd 0.37/0.74/1.11/1.48.
- select feasible (amendment 1, secondary): members whose simulation of any included curve is non-finite are dropped before SINGLE and the ensemble are chosen. Feasible summaries exist where the analysis was run; on vd every value equals the strict one.
- conformal and crossinterp_conformal reuse the single and crossinterp point predictions; ens_env and laplace_param are reported in the JSON summaries and omitted here.

