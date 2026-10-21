# I9-2 pre-registration draft: function-class ablation for the shared bias-only correction

Status: DRAFT, not in force (no run or lc job until prereg.json is committed). Date 2026-10-21, base commit 4bff420.

## Questions

- **Q1**: Does the held-out gain of the shared bias-only correction depend on the function class of g?
- **Q2**: Is there a class with fewer trained values than M_bias (1765) whose held-out RRMS is not higher than M_bias by more than the margin, and does it have a lower RRMS?
- **Q3 (secondary)**: How does the held-out gain of the two best classes change with the number of training devices (2, 4, 8, all) on the two largest datasets (TU Delft 4.13 K, NIST 4 K)?

## Background (stored results)

- I3-2 GMR(M_bias, C0_refit): sky130 77 K 0.769 [0.666, 0.886]; GF180 c 0.880 [0.837, 0.921]; IHP c 0.885 [0.789, 1.005].
- I4-1 GMR(M_bias, X) on sky130 77 K: B3_d2 0.789, B3_d3 0.844 [0.751, 0.949], B4 0.658; GF180 / IHP c: B3_d2 0.994 / 0.986, B3_d3 0.997 / 0.985, B4 0.954 / 0.848.
- I3-2 F2 on sky130 77 K, GMR vs C0_refit: width 8 0.759, width 32 0.777, 3-16-1 0.770, seed sets 5-9 / 10-14 / 15-19 0.771 / 0.779 / 0.771; GMR vs the I3-2 M_bias 0.988-1.083, every CI includes 1.
- I5-1 IHP measured 300 K: GMR(M_bias, C0_refit) 1.214 [1.008, 1.537]; means M_bias 0.0203, B3_d3 0.0214.
- I6-1 TU Delft: 4.13 K GMR(M_bias, C0_refit) 0.939 [0.877, 1.004], GMR(M_bias, B3_d3) 0.983 [0.956, 1.015]; 295.0 K 0.797 [0.688, 0.935].
- I8-1 NIST 4 K: GMR(M_bias, C0_refit) 0.806 [0.758, 0.864]; GMR(M_bias, B3_d3) 0.981 [0.961, 1.000] (27/53); B3_d3 gives 91% of the M_bias log gain.
- The poly2, poly3 and gp contrasts are known on the datasets where they are stored (retrospective part); the const, lin, poly4, poly5, gam, mlp8 (except sky130), mlp16_s1, mlp32 (except sky130), mlp64 results are not known. All datasets are development sets; no TEST split is loaded.

## Datasets

| dataset | pool | devices (finite) | folds | held-out | training points per fold | stored rows reused | new classes |
|---|---|---|---|---|---|---|---|
| sky130_77K | primary | 12 (12) | 12 | 12 | 14945-15670 | C0_refit, mlp16, T1_inloop, poly2, poly3, gp, mlp8, mlp32 | const, lin, poly4, poly5, gam, mlp64 + mlp16_s1 |
| nist_4K | primary | 57 (53) | 53 | 53 | 25366-25829 | C0_refit, mlp16, T1_inloop, poly3 | const, lin, poly2, poly4, poly5, gam, mlp8, mlp32, mlp64, gp + mlp16_s1 |
| tudelft_4K | primary | 45 (45) | 7 | 45 | 94305-109628 | C0_refit, mlp16, T1_inloop, poly3 | const, lin, poly2, poly4, poly5, gam, mlp8, mlp32, mlp64, gp + mlp16_s1 |
| ihp_meas_300K | primary | 18 (18) | 18 | 18 | 1785-2891 | C0_refit, mlp16, T1_inloop, poly3 | const, lin, poly2, poly4, poly5, gam, mlp8, mlp32, mlp64, gp + mlp16_s1 |
| gf180_c | simulated | 48 (48) | 16 | 48 | 78217-80214 | C0_refit, mlp16, T1_inloop, poly2, poly3, gp | const, lin, poly4, poly5, gam, mlp8, mlp32, mlp64 + mlp16_s1 |
| ihp_c | simulated | 48 (48) | 16 | 48 | 79654-81877 | C0_refit, mlp16, T1_inloop, poly2, poly3, gp | const, lin, poly4, poly5, gam, mlp8, mlp32, mlp64 + mlp16_s1 |
| tudelft_295K | other_temperature | 45 (45) | 7 | 45 | 94010-109329 | C0_refit, mlp16, T1_inloop, poly3 | const, lin, poly2, poly4, poly5, gam, mlp8, mlp32, mlp64, gp + mlp16_s1 |
| ihp_meas_233K | other_temperature | 18 (18) | 18 | 18 | 1762-2877 | C0_refit, mlp16, T1_inloop, poly3 | const, lin, poly2, poly4, poly5, gam, mlp8, mlp32, mlp64, gp + mlp16_s1 |
| ihp_meas_343K | other_temperature | 18 (18) | 18 | 18 | 1818-2930 | C0_refit, mlp16, T1_inloop, poly3 | const, lin, poly2, poly4, poly5, gam, mlp8, mlp32, mlp64, gp + mlp16_s1 |
| ihp_meas_398K | other_temperature | 18 (18) | 18 | 18 | 1800-2983 | C0_refit, mlp16, T1_inloop, poly3 | const, lin, poly2, poly4, poly5, gam, mlp8, mlp32, mlp64, gp + mlp16_s1 |
| gf180_b | null_control | 48 (48) | 16 | 48 | 77435-79379 | C0_refit, mlp16 | const, lin, poly2, poly3, poly4, poly5, gam, mlp8, mlp32, mlp64, gp + mlp16_s1 |
| ihp_b | null_control | 48 (48) | 16 | 48 | 79630-81739 | C0_refit, mlp16 | const, lin, poly2, poly3, poly4, poly5, gam, mlp8, mlp32, mlp64, gp + mlp16_s1 |

Sources and refit paths:

- sky130_77K: sky130 77 K REAL_DEV (paper data), 12 devices, leave-one-device-out, 43-parameter card from direct43_r3_fd, ngspice-41 arm64. Stored: C0_refit, mlp16 (= M_bias), T1_inloop: I3_2/real_dev; poly2, poly3, gp: I4_1/real_dev (B3_d2, B3_d3, B4); mlp8, mlp32: I3_2/followup/f2_real (b_width8, b_width32, seeds 0-4, starting-card training records).
- nist_4K: NIST mds2-2997 sky130 4 K DEV, 57 devices (53 finite starting records, amendment A1), leave-one-device-out, 77 K card deck, ngspice-41 arm64. Stored: C0_refit, mlp16, T1_inloop: I8_1/lodo; poly3: I8_1/baselines (B3_d3).
- tudelft_4K: TU Delft 40 nm 4.13 K DEV, 45 devices in 7 geometry groups, leave-one-group-out, CMOSedu BSIM4 card, ngspice-41 arm64. Stored: C0_refit, mlp16, T1_inloop, poly3: I6_1/lodo_4K.
- ihp_meas_300K: IHP SG13G2 measured 300 K DEV, 18 DUTs, leave-one-device-out within class (LV / HV), PSP 103, ngspice-46 + OSDI. Stored: C0_refit, mlp16, T1_inloop, poly3: I5_1/lodo_300K.
- gf180_c: GF180 simulated tier c sets 0-2, 48 targets, leave-one-geometry-out (16), ngspice-41 arm64. Stored: C0_refit, mlp16, T1_inloop: I3_2/gf180_c_logo; poly2, poly3, gp: I4_1/gf180_c_logo.
- ihp_c: IHP SG13G2 simulated tier c sets 0-2, 48 targets, leave-one-geometry-out (16), ngspice-46 + OSDI. Stored: C0_refit, mlp16, T1_inloop: I3_2/ihp_c_logo; poly2, poly3, gp: I4_1/ihp_c_logo.
- tudelft_295K: TU Delft 295.0 K, same devices and folds as tudelft_4K. Stored: C0_refit, mlp16, T1_inloop, poly3: I6_1/lodo_295K.
- ihp_meas_233K: IHP measured 233 K, same DUTs and folds as ihp_meas_300K. Stored: C0_refit, mlp16, T1_inloop, poly3: I5_1/lodo_233K.
- ihp_meas_343K: IHP measured 343 K, same DUTs and folds as ihp_meas_300K. Stored: C0_refit, mlp16, T1_inloop, poly3: I5_1/lodo_343K.
- ihp_meas_398K: IHP measured 398 K, same DUTs and folds as ihp_meas_300K. Stored: C0_refit, mlp16, T1_inloop, poly3: I5_1/lodo_398K.
- gf180_b: GF180 simulated tier b sets 0-2 (I3-2 F2 records), 48 targets, leave-one-geometry-out. Stored: C0_refit, mlp16: I3_2/followup/f2_gf180_b.
- ihp_b: IHP simulated tier b sets 0-2 (I3-2 F2 records), 48 targets, leave-one-geometry-out. Stored: C0_refit, mlp16: I3_2/followup/f2_ihp_b.

## Classes

| class | form | trained values | training |
|---|---|---|---|
| const | g = tanh(c_pol) | 2 | fit_poly degree 0 |
| lin | g = tanh(c0 + c1 u' + c2 v') per polarity (u', v' standardized) | 6 | fit_poly degree 1 |
| poly2 | g = tanh(P_pol), total degree 2 | 12 | fit_poly degree 2 (= I4-1 B3_d2) |
| poly3 | total degree 3 | 20 | fit_poly degree 3 (= I4-1 B3_d3) |
| poly4 | total degree 4 | 30 | fit_poly degree 4 |
| poly5 | total degree 5 | 42 | fit_poly degree 5 |
| gam | g = tanh(c + B(u) a + B(v) b + (B(u) p)(B(v) q)) per polarity; B: 8 cubic B-splines, uniform interior knots on the training range of u = |Vgs|/VDD and v = |Vds|/VDD, inputs clipped to that range; interaction rank 1 | 66 | fit_gam: E1 loss + 1e-4 (|a|^2 + |b|^2 + |p|^2 + |q|^2); c, a, b, q start at 0, p at N(0, 0.1^2) (torch generator seed 0) |
| mlp8 | mean of 5 members tanh(MLP 3-8-8-1) | 565 | fit_bias hidden 8, seeds 0-4 |
| mlp16 | M_bias: mean of 5 members tanh(MLP 3-16-16-1) | 1765 | stored I3-2 / I5-1 / I6-1 / I8-1 / F2 M_bias (reference, not retrained) |
| mlp16_s1 | member 0 of the stored M_bias fold model (seed 0) | 353 | none (members are trained independently from torch.manual_seed(seed) on the same batch; member 0 equals the 1-seed fit with seed 0, unit test) |
| mlp32 | mean of 5 members tanh(MLP 3-32-32-1) | 6085 | fit_bias hidden 32, seeds 0-4 |
| mlp64 | mean of 5 members tanh(MLP 3-64-64-1) | 22405 | fit_bias hidden 64, seeds 0-4 |
| gp | I4-1 B4: GP posterior mean of ln|I_meas / I_sim| on (is_pmos, u, v), ARD squared exponential + white noise, constant mean, type-II ML; 2000 training points per fold (equal share per device) | 2006 | fit_gp (alpha vector 2000 + 6 hyperparameters; the 2000 x 3 stored inputs not counted) |

u = |Vgs|/VDD, v = |Vds|/VDD; one coefficient set per polarity for const, lin, poly, gam; the MLPs and the GP take is_pmos as an input.

## Protocol

- folds: as the stored runs: sky130 77 K, NIST and IHP measured leave-one-device-out (IHP measured trains within class LV / HV); TU Delft leave-one-group-out (7 groups); GF180 / IHP tiers b and c leave-one-geometry-out (16 geometries x 3 sets)
- training_records: the stored starting-card records of the fold's training devices, in the stored training order (load order; NIST sorted by tag); NIST records with non-finite simulated points excluded (amendment A1 of I8-1) and scored as failed in every class
- training: E1 loss (mean over devices of the paper per-device RRMS of I_sim x exp(g)) + L2 1e-4 on non-intercept coefficients / weight matrices; Adam lr 3e-3, 2000 full-batch steps, float64, 1 torch thread; zero-initialized output (poly, gam: zero coefficients; MLP: zero last layer); GP: type-II ML (I4-1 B4 settings)
- heldout_refit: ngspice_fit.polish from the stored starting card, max_nfev 400, irls_rounds 2, diff_step 0.01, g frozen in the loop, through the dataset driver's own refit function (I4-1 refit_task for sky130 and pdk2, I5-1 / I6-1 / I8-1 refit_task otherwise)
- score: strict paper RRMS of I_card x exp(g) with the frozen inclusion stored on the record; a non-finite included curve scores NaN (failed)
- simulators: ngspice-41 arm64 (sky130, NIST, TU Delft, GF180); Homebrew ngspice-46 + PSP 103 OSDI (IHP simulated and measured), as in the stored runs

## Reuse plan

- Starting-card records: E1 real_dev (+ direct43_r3_fd params), E1 {gf180,ihp}_c_set{0,1,2}, I3_2/followup/f2_{gf180,ihp}_b/records, I5_1/start/records_{T}K, I6_1/start/records_{T}K, I8_1/start/records. No starting fit is rerun.
- Stored per-device rows (listed per dataset) enter the statistics unchanged.
- mlp16_s1: member 0 of the stored M_bias fold model (corrections.json key 'M_bias|<fold>'); only its held-out refits are new.
- New work per dataset: training of the classes without stored rows on each fold, and one held-out refit per class and held-out device.
- Reproduction check: poly3 (C0_refit on tier b) retrained and refit on the first 2 held-out devices of each dataset; max |RRMS - stored| <= 1e-9 required.

## Statistics

- per_dataset: GMR(X, Y) = exp(mean over held-out devices of ln(RRMS_X / RRMS_Y)) over devices with both values finite (failures counted separately); 95% percentile CI from 20000 device-bootstrap resamples (default_rng(0)); geometry / group cluster bootstrap reported for tier b / c and TU Delft
- pooled: equal-weight pooled GMR over the primary datasets: exp(mean over datasets of the per-dataset mean log ratio); CI from 20000 stratified resamples (devices resampled within each dataset)
- multiplicity: 12 primary contrasts GMR(X, mlp16), X in {const, lin, poly2, poly3, poly4, poly5, gam, mlp8, mlp16_s1, mlp32, mlp64, gp}; Bonferroni: two-sided percentile CI at level 1 - 0.05/12 (tails 0.208%, about 42 resamples each). Every other interval is a 95% interval, labelled descriptive.
- primary_datasets: sky130_77K, nist_4K, tudelft_4K, ihp_meas_300K

## Decision rule

- margin: 1.03
- labels: from the Bonferroni CI of the pooled GMR(X, mlp16): 'beats' if upper < 1; 'worse' if lower > 1; 'equivalent' if the CI lies in [1/1.03, 1.03]; 'non-inferior' if upper <= 1.03; else 'inconclusive'
- Q1: 'class-dependent' if at least one class is 'beats' or 'worse'; 'no class dependence detected' if all 12 are 'equivalent'; else 'inconclusive'
- Q2: candidates: classes labelled 'beats', 'equivalent' or 'non-inferior' with fewer trained values than mlp16 (1765), point GMR(X, mlp16) <= 1.05 on every primary dataset, and no more failed devices than mlp16 on any primary dataset. Answer: the candidate with the fewest trained values (tie: lower pooled GMR), or 'none'. 'beats with fewer parameters' if that candidate is labelled 'beats'.
- capacity_curve: pooled GMR(X, C0_refit) (95%) against the number of trained values, reported for every class
- code: cryoml.explore.i9_2_classes.decide / label_class / pooled_ratio

## Secondary analyses

- S1: the rule on the six primary + simulated tier c datasets (Bonferroni level).
- S2: per-dataset GMR(X, mlp16) and GMR(X, C0_refit) for every class and every dataset (95%).
- S3: cluster bootstrap of the primary pool (TU Delft groups).
- S4: margin sensitivity: labels at margins 1.02 and 1.05.
- S5: fraction of the M_bias log gain retained, ln GMR(X, C0_refit) / ln GMR(mlp16, C0_refit), on datasets where the stored GMR(mlp16, C0_refit) CI upper bound is < 1.
- S6: null control (tier b): GMR(X, C0_refit) for every class; a class with CI upper < 0.95 on tier b is reported.
- S7: final training loss per class and fold; g maps on a 41 x 41 grid per polarity (fold mean).
- S8: other temperatures (IHP measured 233 / 343 / 398 K, TU Delft 295.0 K).

## Learning curve (Q3)

- datasets: ['tudelft_4K (45 devices, 7 folds)', 'nist_4K (53 finite devices, 53 folds)']
- classes: the two classes with the lowest pooled GMR(X, C0_refit) on the primary pool; GMRs within 0.005 count as tied and the class with fewer trained values ranks first (select_lc_classes); chosen by the decide output before any learning-curve job
- sizes: n_train in {2, 4, 8} plus 'all' (the main-run rows)
- subsets: per fold, 3 subsets of the fold's training devices; polarity-stratified (n/2 nMOS, n/2 pMOS; shortfall filled from the other polarity); default_rng([20261021, n, r, fold index]) on the sorted tags (training_subsets)
- statistic: per held-out device, exp(mean over the 3 subsets of ln RRMS); per dataset GMR(X_n, C0_refit) and GMR(X_n, X_all) with 95% device-bootstrap CIs; GMR(class A_n, class B_n) per n. Descriptive; no decision.
- refits: {'tudelft_4K': 810, 'nist_4K': 954}

## Validity checks

- V1: reproduction: max |RRMS - stored| <= 1e-9 for the poly3 (tier b: C0_refit) rerun on 2 held-out devices per dataset; a dataset failing V1 is excluded and reported
- V2: failed devices per class and dataset reported; a class with more failed devices than mlp16 on a primary dataset is not a Q2 candidate
- V3: simulator timeouts per job reported (I6-1 10 s per process, I8-1 timeout handling unchanged)
- V4: the stored rows used are exactly the stored values (read, not recomputed); mlp16_s1 g equals member 0 of the stored model

## Compute estimate

Refit CPU-s = n_heldout x (number of new refit classes) x mean single-process runtime of the stored M_bias held-out refits of the dataset (GP refits: stored I4-1 B4 mean, else corrected refit + 30 s), + 2 reproduction refits. Training CPU-s = sum over folds of (seconds per training point per Adam step per member, measured in the smoke on the sky130 77 K training sets) x training points of the fold x 2000 steps x members; GP 30 s per fold (I4-1 timing, 2000 points, 200 L-BFGS-B iterations). Wall time on 6 workers at 85% efficiency. Calibration: the smoke cost of the 16-wide MLP (0.33 us per point-step) gives 10 s (sky130), 17 s (NIST), 52 s (GF180 c) and 64 s (TU Delft) per seed task; the stored I3-2 / I8-1 / I6-1 pool timings give about 10, 15, 55 and 51 s.

| dataset | pool | refit CPU-h | training CPU-h | of which mlp32 + mlp64 | total CPU-h | wall min (6 workers) |
|---|---|---|---|---|---|---|
| sky130_77K | primary | 0.32 | 0.74 | 0.71 | 1.06 | 12.5 |
| nist_4K | primary | 1.82 | 9.08 | 7.6 | 10.9 | 128.3 |
| tudelft_4K | primary | 1.77 | 4.46 | 3.88 | 6.23 | 73.3 |
| ihp_meas_300K | primary | 0.71 | 0.41 | 0.23 | 1.12 | 13.2 |
| gf180_c | simulated | 1.38 | 8.03 | 7.1 | 9.41 | 110.7 |
| ihp_c | simulated | 1.8 | 8.19 | 7.24 | 9.99 | 117.6 |
| tudelft_295K | other_temperature | 1.89 | 4.45 | 3.86 | 6.34 | 74.6 |
| ihp_meas_233K | other_temperature | 0.7 | 0.41 | 0.23 | 1.11 | 13.1 |
| ihp_meas_343K | other_temperature | 0.72 | 0.41 | 0.23 | 1.13 | 13.3 |
| ihp_meas_398K | other_temperature | 0.7 | 0.41 | 0.23 | 1.11 | 13.1 |
| gf180_b | null_control | 2.24 | 8.18 | 7.03 | 10.42 | 122.6 |
| ihp_b | null_control | 2.8 | 8.42 | 7.23 | 11.22 | 132.0 |

| pool | datasets | CPU-h | wall min |
|---|---|---|---|
| primary | sky130_77K, nist_4K, tudelft_4K, ihp_meas_300K | 19.31 | 227.3 |
| simulated | gf180_c, ihp_c | 19.4 | 228.3 |
| other_temperature | tudelft_295K, ihp_meas_233K, ihp_meas_343K, ihp_meas_398K | 9.69 | 114.1 |
| null_control | gf180_b, ihp_b | 21.64 | 254.6 |

All datasets: 70.05 CPU-h, about 13.7 h wall. Learning curve: 810 + 954 refits, about 73 min wall.

- Training, not refits, dominates for the 5-seed MLP classes: mlp32 and mlp64 are about 45 of the 70 CPU-h of the full set (mlp64 alone 5.2 CPU-h on NIST, 4.9 on each tier c dataset, 2.6 on each TU Delft temperature). For const, lin, poly2-5, gam and mlp16_s1 the refits dominate.
- Primary pool alone: about 19 CPU-h, 3.8 h wall on 6 workers.
- NIST is leave-one-device-out (53 folds), so every class is trained 53 times.

## Smoke (code check only)

sky130 77 K, held-out nmos_L0p15_W1p6 and pmos_L0p35_W0p55 (2 folds, training on the other 11 devices), all 13 classes trained with 200 steps, MLP seeds (0, 1), mlp16_s1 = member 0 of the smoke mlp16, GP 300 points / 20 L-BFGS-B iterations, held-out refit max_nfev 60, irls 1; 4 workers. Code check only; not used for any conclusion.

Runtime 30.3 s (training 9.6 s, 32 tasks; refits 20.6 s, 26 refits).

| class | nmos_L0p15_W1p6 | pmos_L0p35_W0p55 |
|---|---|---|
| const | 0.01934 | 0.40895 |
| gam | 0.02497 | 0.34677 |
| gp | 0.03528 | 5.13641 |
| lin | 0.01978 | 0.40905 |
| mlp16 | 0.02729 | 0.38371 |
| mlp16_s1 | 0.03068 | 0.38113 |
| mlp32 | 0.02761 | 0.38512 |
| mlp64 | 0.02519 | 0.3841 |
| mlp8 | 0.02924 | 0.39348 |
| poly2 | 0.0188 | 0.38934 |
| poly3 | 0.01885 | 0.37496 |
| poly4 | 0.0248 | 0.35892 |
| poly5 | 0.02751 | 0.3552 |
| stored C0_refit (full settings) | 0.01934 | 0.40895 |
| stored M_bias (full settings) | 0.02351 | 0.34964 |

gp on pmos_L0p35_W0p55 gives 5.14 with 300 points and 20 iterations (reduced settings).

## Queue plan (not queued)

- I9_2_run_<dataset> for the 4 primary datasets (about 3.8 h wall), then I9_2_decide_primary
- I9_2_run_<dataset> for gf180_c, ihp_c (about 3.8 h), other temperatures (about 1.9 h), tier b (about 4.2 h, or a reduced class set)
- I9_2_decide, then I9_2_lc_tudelft_4K and I9_2_lc_nist_4K (about 1 h)
- one compute-heavy job at a time; 6 workers, 1 torch thread each

## Open questions

- OQ1 primary pool: the 4 measured datasets (draft) or the 6 datasets including simulated tier c.
- OQ2 margin 1.03 (draft) and the per-dataset limit 1.05; the F2 seed sets alone moved GMR vs the I3-2 M_bias by 0.988-1.083 on sky130 77 K (12 devices).
- OQ3 reference replicate: add mlp16 with seeds 5-9 on the primary datasets (about 4 CPU-h) to measure the run-to-run spread of the reference itself.
- OQ4 parameter count: total over the 5 members (draft; mlp16 1765) or per member (353); under per-member counting mlp8 (113) and gam (66) stay below, mlp16_s1 equals the reference.
- OQ5 gam settings (8 cubic B-splines, rank-1 interaction, L2 only, no smoothness penalty).
- OQ6 poly4 / poly5 use monomials in standardized u, v (as B3); 2000 Adam steps may not converge for degree 5; alternative: Legendre basis or 4000 steps for all classes.
- OQ7 tier b null control: full class set (about 4.2 h), a reduced set {const, lin, poly3, gam, mlp16_s1} (about 1 h), or drop.
- OQ8 the 20 I7-1 simulated datasets are excluded (about 1-2 h each).
- OQ9 learning curve: add mlp16 as a fixed third class if it is not among the two selected; 3 subsets per size.
- OQ10 equal dataset weight (draft) or device-count weight in the pooled statistic.

