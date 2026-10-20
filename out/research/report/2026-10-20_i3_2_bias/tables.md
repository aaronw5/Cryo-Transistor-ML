# Tables, 2026-10-20 i3_2_bias

Generated 2026-10-20 01:56 by `PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i3_2.py --date 2026-10-20` from stored files only. RRMS is the paper metric with frozen curve inclusion, strict.

### I3-2 decision: pre-registered conditions R1-R4, values and rule output

| condition | value | met |
|---|---:|---:|
| R1 (REAL_DEV): upper 95% CI bound of R(M_bias, C0_refit) < 1 | R = 0.769, CI [0.666, 0.886] | yes |
| R2 (REAL_DEV): M_bias all-device mean <= T1_inloop mean + 0.005 | 0.08055 <= 0.08198 + 0.005 = 0.08698 | yes |
| R3 (REAL_DEV): 0 failed devices for M_bias | 0/12 failed | yes |
| R4 (simulated): R(M_bias, C0_refit) <= 0.90 and 0 failed on GF180 or IHP | GF180 R = 0.880, 0 failed (pass); IHP R = 0.885, 0 failed (pass) | yes |
| rule output (decision.json, I3_2_decide) |  | CONTINUE |

- Pre-registration: out/research/explore/I3_2/prereg.json (commit f7f6378, written before any M_bias or M_bias_geom result). CONTINUE only if R1-R4 all hold; STOP otherwise.
- R = exp(mean_i ln(RRMS_M_bias,i / RRMS_C0_refit,i)); bootstrap 95% CI = 2.5/97.5 percentiles over 10000 resamples, numpy default_rng(0); unit: device (REAL_DEV, 12), geometry cluster (GF180/IHP, 16 clusters of 3 targets).
- T1_inloop on REAL_DEV = stored E1 stage-2 T1 rows (0.08198); reproduced by the rerun with max |difference| 0. C0_refit = stored I2-1 rows; reproduced with max |difference| 0.
- Not used for the decision: M_bias_geom, T1_inloop on GF180/IHP, I2-1 T1_joint, reference rows, post-hoc diagnostics, reruns, REAL_TEST.
- Values recomputed from the stored per-device RRMS and checked against criterion_real_part, criterion_R4_part and decision.json (when present).


### I3-2 REAL_DEV leave-one-device-out: held-out RRMS per arm, reference rows and diagnostics

| arm | role | all-device | combined | nMOS | pMOS | median | failed | paired GMR vs C0_refit [95% CI] | devices below C0_refit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0_refit | control (stored I2-1 rows) | 0.1072 | 0.0981 | 0.0432 | 0.1529 | 0.0820 | 0/12 |  |  |
| M_bias | primary | 0.0806 | 0.0749 | 0.0407 | 0.1090 | 0.0605 | 0/12 | 0.769 [0.666, 0.886] | 11/12 |
| M_bias_geom | secondary | 0.0914 | 0.0879 | 0.0674 | 0.1085 | 0.0596 | 0/12 | 0.894 [0.702, 1.171] | 7/12 |
| T1_inloop | comparator, not deployable (stored E1 stage 2) | 0.0820 | 0.0760 | 0.0403 | 0.1117 | 0.0650 | 0/12 | 0.794 [0.707, 0.889] | 11/12 |
| ref_M_post_E1_stage2 | reference: E1 M (6 inputs) post-hoc, refit | 0.0871 | 0.0842 | 0.0669 | 0.1015 | 0.0612 | 0/12 | 0.866 [0.672, 1.139] | 8/12 |
| ref_M_joint_I2_1 | reference: I2-1 M_joint, refit | 0.0903 | 0.0884 | 0.0769 | 0.1000 | 0.0622 | 0/12 | 0.907 [0.685, 1.234] | 7/12 |
| ref_T1_joint_I2_1 | reference: I2-1 T1_joint, refit | 0.0803 | 0.0744 | 0.0390 | 0.1099 | 0.0640 | 0/12 | 0.777 [0.693, 0.868] | 11/12 |
| C0_refit_rerun | check: C0_refit rerun | 0.1072 | 0.0981 | 0.0432 | 0.1529 | 0.0820 | 0/12 | 1.000 [1.000, 1.000] | 0/12 |
| T1_inloop_rerun | check: T1_inloop rerun | 0.0820 | 0.0760 | 0.0403 | 0.1117 | 0.0650 | 0/12 | 0.794 [0.707, 0.889] | 11/12 |
| posthoc:C0_start | diagnostic: starting card, no refit, no g | 0.1073 | 0.0981 | 0.0433 | 0.1530 | 0.0823 | 0/12 | 1.001 [1.000, 1.002] | 0/12 |
| posthoc:M_bias | diagnostic: starting card x exp(g), no refit | 0.0857 | 0.0795 | 0.0424 | 0.1166 | 0.0626 | 0/12 | 0.837 [0.726, 0.967] | 10/12 |
| posthoc:M_bias_geom | diagnostic: starting card x exp(g), no refit | 0.1115 | 0.1072 | 0.0818 | 0.1327 | 0.0633 | 0/12 | 1.055 [0.802, 1.437] | 6/12 |
| published | reference: published 77 K card (released) | 0.2848 | 0.2608 | 0.1172 | 0.4044 | 0.2304 | 0/12 | 2.910 [2.485, 3.434] | 0/12 |

- Strict paper RRMS with frozen published-baseline curve inclusion, 12 REAL_DEV devices (5 nMOS, 7 pMOS); all-device = mean over 12; combined = mean of the nMOS and pMOS family means.
- Refit rows: held-out card refit from the direct43_r3_fd starting card with the frozen correction in the loop (max_nfev 400, irls 2, diff_step 0.01), best card re-simulated and multiplied by exp(g). Corrections trained leave-one-device-out on the other 11 devices.
- R vs C0_refit: paired geometric-mean ratio with device-bootstrap 95% CI (10000 resamples, default_rng(0)); recomputed here and checked against the stored comparisons where stored.
- published: NGSpice curves of the published (upstream released) 77 K card from the upstream committed sweeps (data/raw/CryoPDK_Skywater130nm_ML, commit 39b1e51; scripts/verify_simulator.py reproduces them), scored here with the same strict scorer and checked against out/research/reference_scores.json (released, |diff| < 1e-8); no new simulation.
- posthoc rows: the fold correction applied to the stored starting-card simulation without a refit (diagnostic). M_bias vs T1_inloop: R = 0.968 [0.920, 1.011], 7/12; M_bias_geom vs M_bias: R = 1.163 [0.992, 1.413], 4/12.


### I3-2 simulated GF180 and IHP tier c, leave-one-geometry-out: held-out RRMS per arm

| PDK | arm | role | all-target mean | median | failed | paired GMR vs C0_refit [95% CI] | targets below C0_refit |
|---|---|---|---:|---:|---:|---:|---:|
| GF180 (BSIM4) tier c | C0_refit | control (stored I2-1 rows) | 0.0457 | 0.0352 | 0/48 |  |  |
| GF180 (BSIM4) tier c | M_bias | primary (R4) | 0.0399 | 0.0299 | 0/48 | 0.880 [0.837, 0.921] | 42/48 |
| GF180 (BSIM4) tier c | M_bias_geom | secondary | 0.0407 | 0.0326 | 0/48 | 0.886 [0.802, 0.967] | 28/48 |
| GF180 (BSIM4) tier c | T1_inloop | comparator, not deployable (new run) | 0.0391 | 0.0291 | 0/48 | 0.860 [0.830, 0.890] | 45/48 |
| GF180 (BSIM4) tier c | ref_M_post_I2_1 | reference: I2-1 M_post, refit | 0.0389 | 0.0305 | 0/48 | 0.861 [0.766, 0.947] | 32/48 |
| GF180 (BSIM4) tier c | ref_M_joint_I2_1 | reference: I2-1 M_joint, refit | 0.0381 | 0.0282 | 0/48 | 0.835 [0.741, 0.919] | 40/48 |
| GF180 (BSIM4) tier c | C0_refit_rerun | check: C0_refit rerun | 0.0457 | 0.0352 | 0/48 | 1.000 [1.000, 1.000] | 0/48 |
| GF180 (BSIM4) tier c | posthoc:C0_start | diagnostic: starting card, no refit | 0.0474 | 0.0364 | 0/48 | 1.040 [1.022, 1.059] | 0/48 |
| GF180 (BSIM4) tier c | posthoc:M_bias | diagnostic: starting card x exp(g), no refit | 0.0434 | 0.0352 | 0/48 | 0.969 [0.930, 1.009] | 32/48 |
| GF180 (BSIM4) tier c | posthoc:M_bias_geom | diagnostic: starting card x exp(g), no refit | 0.0493 | 0.0424 | 0/48 | 1.117 [0.988, 1.268] | 17/48 |
| IHP SG13G2 (PSP103) tier c | C0_refit | control (stored I2-1 rows) | 0.1011 | 0.0567 | 0/48 |  |  |
| IHP SG13G2 (PSP103) tier c | M_bias | primary (R4) | 0.0871 | 0.0510 | 0/48 | 0.885 [0.789, 1.005] | 32/48 |
| IHP SG13G2 (PSP103) tier c | M_bias_geom | secondary | 0.0966 | 0.0552 | 0/48 | 0.960 [0.799, 1.189] | 30/48 |
| IHP SG13G2 (PSP103) tier c | T1_inloop | comparator, not deployable (new run) | 0.0921 | 0.0491 | 0/48 | 0.932 [0.868, 1.011] | 35/48 |
| IHP SG13G2 (PSP103) tier c | ref_M_post_I2_1 | reference: I2-1 M_post, refit | 0.1019 | 0.0700 | 0/48 | 1.050 [0.841, 1.334] | 26/48 |
| IHP SG13G2 (PSP103) tier c | ref_M_joint_I2_1 | reference: I2-1 M_joint, refit | 0.0915 | 0.0500 | 0/48 | 0.971 [0.837, 1.120] | 23/48 |
| IHP SG13G2 (PSP103) tier c | C0_refit_rerun | check: C0_refit rerun | 0.1011 | 0.0567 | 0/48 | 1.000 [1.000, 1.000] | 0/48 |
| IHP SG13G2 (PSP103) tier c | posthoc:C0_start | diagnostic: starting card, no refit | 0.1012 | 0.0568 | 0/48 | 1.001 [1.000, 1.001] | 0/48 |
| IHP SG13G2 (PSP103) tier c | posthoc:M_bias | diagnostic: starting card x exp(g), no refit | 0.0938 | 0.0564 | 0/48 | 1.034 [0.899, 1.207] | 24/48 |
| IHP SG13G2 (PSP103) tier c | posthoc:M_bias_geom | diagnostic: starting card x exp(g), no refit | 0.1350 | 0.1100 | 0/48 | 1.633 [1.282, 2.071] | 13/48 |

- Simulated tier-c truth sets 0-2 (16 geometries x 3 draws = 48 targets per PDK) with injected model-form misspecification; leave-one-geometry-out (the 3 draws of the held-out geometry are removed from training). Starting cards: E1 fitted cards (wide box: GF180 21 parameters, IHP 12). VDD 3.3 V (GF180), 1.2 V (IHP).
- R vs C0_refit: paired geometric-mean ratio, 95% CI from 10000 geometry-cluster resamples (default_rng(0)). R4 uses the M_bias row: R <= 0.90 and 0 failed.
- all-target mean equals the combined value here (the stored summary reports the same number).


### I3-2 caveats and open checks

| item | detail |
|---|---|
| REAL_DEV reuse | REAL_DEV was used for the design and the decision of E1 and I2-1, and the I3-2 hypothesis was formed from the REAL_DEV T1 and M results. The I3-2 REAL_DEV values are development evidence, not a held-out test. |
| Input choice after I2-1 | The 3-input set (no geometry, no simulated current) was chosen after the E1 and I2-1 REAL_DEV results were seen. The pre-registration fixed it before any M_bias result existed, but the choice is not independent of REAL_DEV. |
| M_bias vs T1_inloop | M_bias / T1_inloop R = 0.968 [0.920, 1.011], 7/12 devices lower; the CI includes 1. R2 tests M_bias <= T1 + 0.005 (non-inferiority), not M_bias < T1. |
| Verilog-A not compiled | m_bias_sky130_dev_full.va (full 12-device model, 207 lines) is text only; the emitted assignments were evaluated in Python on 200 random bias points (max |g difference| 1.6e-11). It has not been compiled (OpenVAF / ngspice OSDI) or simulated. abs() folds reverse-mode bias onto forward mode; reverse bias does not occur in any record. |
| REAL_TEST untouched | No I3_2 script loads the 6 REAL_TEST devices. A REAL_TEST evaluation happens only at the method freeze, with the configuration fixed before it is loaded. |
| 12 devices | 5 nMOS and 7 pMOS; the bootstrap resamples 12 devices. pmos_L0p35_W0p55 contributes 36% of the M_bias all-device sum (RRMS 0.3496); without it the M_bias mean is 0.0561 vs C0_refit 0.0798 (diagnostic only). |
| Held-out refit uses measured data | The held-out card refit (box 6 of the process diagram) fits the held-out device's 43 parameters to its measured curves, as card extraction does. Only g is held out. C0_refit uses the same refit without g. |
| Simulated R4 margin | R4 needs R <= 0.90 on one PDK. GF180: R = 0.880 [0.837, 0.921], T1_inloop lower than M_bias (R(M_bias, T1_inloop) = 1.023 [0.982, 1.066]). IHP: R = 0.885 [0.789, 1.005] (the CI includes 1); R(M_bias, T1_inloop) = 0.950 [0.874, 1.036]. Simulated devices with injected misspecification; secondary to REAL_DEV. |
| Figures from stored data | The g surface is the in-sample full model. The I-V figures show the starting card and starting card x exp(g); the refit-card curves are not stored and were not re-simulated. |

