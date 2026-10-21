# Tables, 2026-10-21 i8_1

Generated 2026-10-21 00:42 by `PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i8_1.py --date 2026-10-21` from stored files only. y = ln GMR(M_bias, C0_refit) over the scored DEV devices (strict paper RRMS after the held-out card refit); negative = gain.

### I8-1 NIST sky130 4 K DEV prospective test: P_a, P_b, P_c, validity, amendment A1 and secondary outcomes

| item | pre-registered definition | value | outcome |
|---|---|---|---|
| P_a (decides) | sign of y_D vs the frozen I7-1 rule y = a + b S1_lodo_D, S1_lodo_D from prediction.json (commit 0c184f8, before any held-out refit) | S1_lodo_D 0.50247 -> predicted y -0.2362 (gain); observed y_D -0.2157 (GMR 0.806) | correct |
| P_a magnitude | |y_D - predicted y| and whether it is within the frozen rule's in-sample residual RMS over the 11 stored I7-1 datasets | 0.0205; residual RMS 0.0751 | within |
| P_a, A1 recomputed (secondary) | S1_lodo_D over the 53 finite starting records (prediction_v2.json) | S1_lodo_D 0.50712 -> predicted y -0.2384 (gain) | correct |
| P_b (decides) | R1: device-bootstrap 95% CI upper of GMR(M_bias, C0_refit) < 1; R2: M_bias mean <= T1_inloop mean + 0.005 (finite starts); R3: 0 M_bias failures among finite starts | R1 0.806 [0.758, 0.864], 48/53 wins; R2 0.03151 vs 0.03490 + 0.005; R3 0 of 53 | supported |
| P_c (secondary, prospective) | gain iff S3_D < 0; S3_D from prediction_v2.json (commit 4db5a7b); the prediction.json P_c is void (S3 NaN) | S3_D -0.1081 -> gain; observed gain | correct |
| validity V1 | >= 90% of the 57 DEV devices with a finite starting-card RRMS and >= 2 included curves | 53/57 (4 start failures, amendment A1) | holds |
| validity V2 | C0_refit and M_bias <= 10% failed devices and >= 45 paired devices | failed C0_refit 4, M_bias 4; paired 53 | holds |
| validity V3 | prediction.json committed before any lodo output; commit an ancestor of the results commit | lodo / baselines guard: prediction 0c184f8, prediction_v2 4db5a7b, amendment 5ba2f8e tracked, unmodified, ancestors of HEAD | holds (prospective) |
| validity V4 | ngspice timeouts reported; devices with a timeout in an arm dropped | start job 1 (pfet_01v8_lvt_W1_L0p35_m28p6, during its starting fit; record finite); lodo 0, baselines 0; dropped 0 | reported |
| amendment A1 | commit 5ba2f8e, after the prediction commit and before any held-out outcome | 4 nMOS starting records with a NaN Id-Vg curve (forced bins 0, 1, 1, 4; binned PCLM < 0 stops BSIM4 at the first analysis): excluded from every training set and S fold, scored as failed in every arm | applied |
| per family (secondary) | sign(y_fam) vs the frozen rule on S1_lodo_fam and vs sign(S3_fam) (prediction_v2.json) | nMOS y -0.2135 (29 devices), rule -0.2210, S3 -0.1073; pMOS y -0.2183 (24 devices), rule -0.2386, S3 -0.1024 | correct / correct (both) |
| covered subset (D2 secondary) | GMR(M_bias, C0_refit) on the 25 bin-covered DEV devices | 0.801 [0.739, 0.881], 23/25 wins | reported |
| gates G, G1 (I8-2 definitions) | G: M_bias iff S3_D < 0; G1: M_bias iff frozen-rule y < 0 | both choose M_bias (prediction_v2); GMR(G, C0_refit) 0.806; prediction.json G1 M_bias | chose the lower arm |
| S1_pooled, S2 | dataset values (reported) | prediction.json 0.5479, 0.6157; prediction_v2 0.5528, 0.6422 | reported |
| fold level (secondary, computed here) | Spearman(S_f, y_f) over the 53 scored devices (y_f = ln(M_bias / C0_refit) of the fold's held-out device) | S1_lodo v1 +0.706, v2 +0.734; S3 -0.381 | reported |

- Pre-registration out/research/explore/I8_1/prereg.json (in force from commit d022c32); prediction out/research/explore/I8_1/prediction.json (commit 0c184f8); amendment A1 out/research/explore/I8_1/prereg_amendment_2026-10-20.json (commit 5ba2f8e); prediction_v2 out/research/explore/I8_1/prediction_v2.json (commit 4db5a7b); decision out/research/explore/I8_1/decision.json. Every value is recomputed here from the per-device and per-fold values and checked against the stored files.
- y_D = mean over the 53 scored DEV devices of ln(RRMS_M_bias / RRMS_C0_refit) (strict paper RRMS after the held-out card refit, leave-one-device-out); negative = gain. Frozen rule (I7-1 amendment): y = 0.007619 -0.485184 S1_lodo, gain iff S1_lodo > 0.0157.
- Order: prereg in force -> start (411 s) -> stats (1264 s) -> prediction commit -> lodo stopped before output (NaN corrections from the 4 failed records) -> amendment A1 commit -> stats_v2 (992 s) -> prediction_v2 commit -> lodo (1016 s) -> baselines (393 s) -> decide.


### I8-1 GMR(M_bias, C0_refit) per family and bin stratum, with B3_d3

| subset | DEV devices | start failures (A1) | paired devices | y = ln GMR(M_bias, C0_refit) | GMR(M_bias, C0_refit) | device-bootstrap 95% CI | wins | GMR(B3_d3, C0_refit) [CI] | GMR(M_bias, B3_d3) [CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all 57 DEV | 57 | 4 | 53 | -0.2157 | 0.806 | [0.758, 0.864] | 48/53 | 0.822 [0.768, 0.885] | 0.981 [0.961, 1.000] |
| nMOS | 33 | 4 | 29 | -0.2135 | 0.808 | [0.780, 0.840] | 28/29 | 0.834 [0.799, 0.873] | 0.969 [0.939, 1.000] |
| pMOS | 24 | 0 | 24 | -0.2183 | 0.804 | [0.707, 0.934] | 20/24 | 0.808 [0.705, 0.947] | 0.996 [0.975, 1.015] |
| native bin (covered, D2) | 25 | 0 | 25 | -0.2221 | 0.801 | [0.739, 0.881] | 23/25 | 0.805 [0.739, 0.890] | 0.995 [0.966, 1.025] |
| forced nearest bin | 32 | 4 | 28 | -0.2099 | 0.811 | [0.742, 0.898] | 25/28 | 0.837 [0.761, 0.933] | 0.968 [0.942, 0.993] |
| nMOS native bin | 12 | 0 | 12 | -0.2309 | 0.794 | [0.767, 0.822] | 12/12 | 0.802 [0.761, 0.844] | 0.990 [0.943, 1.042] |
| nMOS forced bin | 21 | 4 | 17 | -0.2012 | 0.818 | [0.773, 0.869] | 16/17 | 0.857 [0.807, 0.915] | 0.954 [0.916, 0.990] |
| pMOS native bin | 13 | 0 | 13 | -0.2140 | 0.807 | [0.694, 0.972] | 11/13 | 0.807 [0.690, 0.978] | 1.000 [0.968, 1.031] |
| pMOS forced bin | 11 | 0 | 11 | -0.2233 | 0.800 | [0.653, 1.032] | 9/11 | 0.808 [0.650, 1.062] | 0.990 [0.965, 1.011] |

- Pre-registered secondary (not deciding): per family and per bin stratum (native bin = covered geometry, decision D2; forced nearest bin otherwise). The covered-subset row equals decision.json secondary.covered_subset. The 4 start failures are nMOS forced-bin devices. The family x stratum rows and the B3_d3 columns are computed here (post hoc subdivision of the pre-registered strata).
- Device-bootstrap CI: 10000 resamples, numpy default_rng(0) (i3_2_bias.paired_ratio).


### I8-1 arms: RRMS aggregates and paired GMR (57 DEV devices, 53 scored)

| arm | devices finite / failed | mean RRMS (finite) | median RRMS | combined (nMOS / pMOS family means) | nMOS mean (29) | pMOS mean (24) | GMR(arm, C0_refit) [95% CI], wins | GMR(M_bias, arm) [95% CI], wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stock 77 K card (no fit) | 53/0 | 0.5694 | 0.4063 | 0.5894 | 0.3776 | 0.8011 |  |  |
| C0 starting card (no refit) | 53/4 | 0.0411 | 0.0181 | 0.0440 | 0.0133 | 0.0748 | 1.032 [1.022, 1.042], 0/53 | 0.781 [0.733, 0.839], 48/53 |
| C0_refit (control) | 53/4 | 0.0400 | 0.0178 | 0.0428 | 0.0129 | 0.0727 | 1 (control) | 0.806 [0.758, 0.864], 48/53 |
| T1_inloop (per-curve gain, non-deployable) | 53/4 | 0.0349 | 0.0177 | 0.0372 | 0.0123 | 0.0622 | 0.908 [0.879, 0.939], 47/53 | 0.887 [0.852, 0.930], 47/53 |
| M_bias (primary) | 53/4 | 0.0315 | 0.0164 | 0.0337 | 0.0105 | 0.0569 | 0.806 [0.758, 0.864], 48/53 | 1 |
| B1 wider BSIM4 set | 53/4 | 0.0407 | 0.0180 | 0.0436 | 0.0132 | 0.0739 | 1.024 [1.014, 1.034], 17/53 | 0.787 [0.739, 0.845], 47/53 |
| B2 Rext + self-heating | 53/4 | 0.0390 | 0.0180 | 0.0417 | 0.0131 | 0.0702 | 0.994 [0.978, 1.009], 32/53 | 0.811 [0.759, 0.874], 48/53 |
| B3_d3 cubic polynomial g | 53/4 | 0.0316 | 0.0174 | 0.0338 | 0.0110 | 0.0565 | 0.822 [0.768, 0.885], 47/53 | 0.981 [0.961, 1.000], 27/53 |

- Sources out/research/explore/I8_1/lodo/results.json, out/research/explore/I8_1/baselines/results.json, out/research/explore/I8_1/start/summary.json. Every arm scores a fresh ngspice-41 simulation of its refit card (M_bias, T1_inloop, B3_d3: multiplied by exp(g) of the frozen fold correction); strict paper RRMS with the frozen measured inclusion. Means over the 53 devices with a finite starting record (the 4 start failures are NaN in every arm, amendment A1). Stock card: start/summary.json stock_rrms over the same 53 devices.
- B3_d3 (I4-1 baseline, i4_1_baselines.fit_poly): g = tanh(P_pol(u, v)), P_pol a polynomial of total degree 3 in the standardized u = |Vgs|/VDD, v = |Vds|/VDD (10 monomials), one coefficient vector per polarity (20 coefficients); same loss as M_bias (mean paper RRMS of the corrected training records), L2 1e-4 on the non-constant coefficients, Adam 3e-3, 2000 full-batch steps, zero initialization, one fit; held-out refit with g in the loop. M_bias: MLP 3-16-16-1 (353 weights) on (is_pmos, u, v), 5 seeds, g = mean of members.
- ln GMR(B3_d3, C0_refit) / ln GMR(M_bias, C0_refit) = -0.1964 / -0.2157 = 0.911.
- B1: 18 extra parameters kept by the edge screen (a1, alpha1, cdsc, dvt0w, dvtp0, fprout, keta, minv, pdiblc1, pdits, phin, pscbe1, pvag, rbpb, rdswmin, ud, up, voffl); extra active per device 17-18. B2: fitted Rext per side median 0.0 ohm (range 0.0-29407.4). B1 / B2 / B3_d3 are reported only (not in P_a, P_b or validity).

