# Tables, 2026-10-21 i9_1

Generated 2026-10-21 03:43 by `PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i9_1.py --date 2026-10-21` from stored files only. y = ln GMR(M_bias, C0_refit) over the 87 DEV devices (strict paper RRMS after the held-out card refit); negative = gain.

### I9-1 NIST sky130 4 K DEV room-temperature card prospective test: P_a, P_b, P_c, validity and secondary outcomes

| item | pre-registered definition | value | outcome |
|---|---|---|---|
| P_a (decides) | sign of y_D vs the frozen I7-1 rule y = a + b S1_lodo_D, S1_lodo_D from prediction.json (commit e937ca2, before any held-out refit) | S1_lodo_D 0.29484 -> predicted y -0.1354 (gain); observed y_D -0.2266 (GMR 0.797) | correct |
| P_a calibration-range note (Q6) | S1_lodo_D inside the I7-1 calibration range of the 11 stored datasets [-0.379, 0.504] | 0.2948 | inside |
| P_a magnitude | |y_D - predicted y| and whether it is within the frozen rule's in-sample residual RMS over the 11 stored I7-1 datasets | 0.0911; residual RMS 0.0751 | outside |
| P_b (decides) | R1: device-bootstrap 95% CI upper of GMR(M_bias, C0_refit) < 1; R2: M_bias mean <= T1_inloop mean + 0.005; R3: 0 M_bias failures | R1 0.797 [0.748, 0.859], 77/87 wins; R2 0.04554 vs 0.05310 + 0.005; R3 0 of 87 | supported |
| P_c (secondary, prospective) | gain iff S3_D < 0; S3_D from prediction.json (commit e937ca2) | S3_D -0.0710 -> gain; observed gain | correct |
| S_B3 (secondary) | GMR(M_bias, B3_d3) with CI; 'M_bias lower' iff CI upper < 1 | 0.955 [0.926, 0.985], 55/87 wins; B3_d3 share of the log gain 0.795 | M_bias lower |
| S_transfer (secondary) | arm M_bias_lvt (M_bias trained once on the 53 finite I8-1 DEV starting records): GMR vs C0_refit and vs M_bias with CI | vs C0_refit 0.887 [0.828, 0.962], 79/87; vs M_bias 1.112 [1.071, 1.156], 29/87 | lower than C0_refit, higher than M_bias |
| validity V1 | >= 90% of the 87 DEV devices with a finite starting record and >= 2 included curves | 87/87 | holds |
| validity V2 | C0_refit and M_bias <= 10% failed devices and >= 70 paired devices | failed C0_refit 0, M_bias 0; paired 87 | holds |
| validity V3 | prediction.json committed before any lodo output; commit an ancestor of the results commit | lodo / baselines guard: prereg 007759b in force, prediction e937ca2 tracked, unmodified, ancestors of HEAD | holds (prospective) |
| validity V4 | ngspice timeouts reported; devices with a timeout in an arm dropped | start 0, lodo 0, baselines 0; dropped 0 | reported |
| per family (secondary) | sign(y_fam) vs the frozen rule on S1_lodo_fam and vs sign(S3_fam) | nMOS y -0.3614 (48 devices), rule -0.2789 (correct), S3 -0.1571 (correct); pMOS y -0.0606 (39 devices), rule -0.0820 (correct), S3 +0.0370 (wrong) | rule 2/2, S3 1/2 |
| gates G, G1 (I8-2 definitions) | G: M_bias iff S3_D < 0; G1: M_bias iff frozen-rule y < 0 | both choose M_bias; GMR(G, C0_refit) 0.797 | chose the lower arm |
| S1_pooled, S2 | dataset values (reported) | 0.3846, 0.6639 | reported |
| fold level (computed here) | Spearman(S_f, y_f) over the 87 folds (y_f = ln(M_bias / C0_refit) of the fold's held-out device) | S1_lodo +0.231; S3 -0.604 | reported |

- Pre-registration out/research/explore/I9_1/prereg.json (in force from commit 007759b); prediction out/research/explore/I9_1/prediction.json (commit e937ca2); decision out/research/explore/I9_1/decision.json. Every value is recomputed here from the per-device and per-fold values and checked against the stored files.
- y_D = mean over the 87 DEV devices of ln(RRMS_M_bias / RRMS_C0_refit) (strict paper RRMS after the held-out card refit, leave-one-device-out); negative = gain. Frozen rule (I7-1 amendment): y = 0.007619 -0.485184 S1_lodo, gain iff S1_lodo > 0.0157.
- Order: draft (5246240) -> prereg in force (007759b) -> start (472 s) -> stats (4352 s) -> prediction commit (e937ca2) -> lodo (2435 s) -> baselines (509 s) -> decide (4 s).


### I9-1 arms: RRMS aggregates and paired GMR (87 DEV devices)

| arm | devices finite / failed | mean RRMS | median RRMS | combined (nMOS / pMOS family means) | nMOS mean (48) | pMOS mean (39) | GMR(arm, C0_refit) [95% CI], wins | GMR(M_bias, arm) [95% CI], wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stock room-temperature card (no fit) | 87/0 | 1.6270 | 0.6728 | 1.7458 | 0.5976 | 2.8941 |  |  |
| C0 starting card (no refit) | 87/0 | 0.0576 | 0.0366 | 0.0603 | 0.0345 | 0.0861 | 1.033 [1.023, 1.043], 0/87 | 0.772 [0.725, 0.831], 79/87 |
| C0_refit (control) | 87/0 | 0.0555 | 0.0357 | 0.0580 | 0.0337 | 0.0823 | 1 (control) | 0.797 [0.748, 0.859], 77/87 |
| T1_inloop (per-curve gain, non-deployable) | 87/0 | 0.0531 | 0.0334 | 0.0554 | 0.0330 | 0.0779 | 0.978 [0.941, 1.029], 73/87 | 0.815 [0.782, 0.851], 75/87 |
| M_bias (primary) | 87/0 | 0.0455 | 0.0280 | 0.0480 | 0.0245 | 0.0714 | 0.797 [0.748, 0.859], 77/87 | 1 |
| M_bias_lvt (trained on I8-1 lvt DEV, transfer) | 87/0 | 0.0475 | 0.0309 | 0.0495 | 0.0297 | 0.0693 | 0.887 [0.828, 0.962], 79/87 | 0.899 [0.865, 0.934], 58/87 |
| B1 wider BSIM4 set | 87/0 | 0.0567 | 0.0357 | 0.0593 | 0.0343 | 0.0843 | 1.023 [1.015, 1.031], 23/87 | 0.780 [0.732, 0.839], 79/87 |
| B2 Rext + self-heating | 87/0 | 0.0552 | 0.0347 | 0.0577 | 0.0334 | 0.0821 | 0.991 [0.980, 1.002], 55/87 | 0.804 [0.755, 0.866], 77/87 |
| B3_d3 cubic polynomial g | 87/0 | 0.0459 | 0.0294 | 0.0480 | 0.0273 | 0.0687 | 0.835 [0.782, 0.900], 77/87 | 0.955 [0.926, 0.985], 55/87 |

- Sources out/research/explore/I9_1/lodo/results.json, out/research/explore/I9_1/baselines/results.json, out/research/explore/I9_1/start/summary.json. Every arm scores a fresh ngspice-41 simulation of its refit card (M_bias, M_bias_lvt, T1_inloop, B3_d3: multiplied by exp(g) of the frozen correction); strict paper RRMS with the frozen measured inclusion. 0 failed devices in every arm. Stock card: start/summary.json stock_rrms.
- M_bias: I3-2 recipe, MLP 3-16-16-1 tanh on (is_pmos, |Vgs|/VDD, |Vds|/VDD), 5 seeds, trained per fold on the 86 other DEV starting records. M_bias_lvt: the same recipe trained once on the 53 finite I8-1 DEV starting records (nfet_01v8_lvt / pfet_01v8_lvt, 77 K card, 77 K deck), applied unchanged. B3_d3: g = tanh(P_pol(u, v)), total degree 3, 20 coefficients, one fit per fold.
- B1: 19 extra parameters kept by the edge screen (a1, alpha1, cdsc, dvt0w, dvtp0, etab, fprout, keta, minv, pdiblc1, pdits, phin, pscbe1, pvag, rbpb, rdswmin, ud, up, voffl); extra active per device 19-19. B2: fitted Rext per side median 0.053 ohm (range 1.6e-08-621). B1 / B2 / B3_d3 are reported only (not in P_a, P_b or validity).


### I9-1 per family: S1_lodo rule, S3 sign, GMR(M_bias, C0_refit) and secondary arms

| family | DEV devices | included curves | C0_refit mean RRMS | M_bias mean RRMS | S1_lodo (prediction.json) | rule-predicted y | observed y | rule sign | S3 (prediction.json) | S3 sign | GMR(M_bias, C0_refit) [95% CI], wins | GMR(B3_d3, C0_refit) [CI] | GMR(M_bias, B3_d3) [CI] | GMR(M_bias_lvt, C0_refit) [CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|
| nMOS (nfet_01v8) | 48 | 288 | 0.0337 | 0.0245 | 0.5905 | -0.2789 | -0.3614 | correct | -0.1571 | correct | 0.697 [0.672, 0.724], 47/48 | 0.771 [0.738, 0.802] | 0.904 [0.867, 0.947] | 0.850 [0.825, 0.873] |
| pMOS (pfet_01v8) | 39 | 161 | 0.0823 | 0.0714 | 0.1847 | -0.0820 | -0.0606 | correct | +0.0370 | wrong | 0.941 [0.842, 1.085], 30/39 | 0.922 [0.809, 1.079] | 1.021 [0.986, 1.055] | 0.934 [0.808, 1.118] |
| all DEV | 87 | 449 | 0.0555 | 0.0455 | 0.2948 | -0.1354 | -0.2266 | correct | -0.0710 | correct | 0.797 [0.748, 0.859], 77/87 | 0.835 [0.782, 0.900] | 0.955 [0.926, 0.985] | 0.887 [0.828, 0.962] |

- Pre-registered secondary (not deciding): per-family S1_lodo and S3 are the means over the family's folds of S1_lodo_family and S3_family in prediction.json (the statistic computed on the fold's training records of that family); the rule-predicted y uses the frozen I7-1 rule. The all-DEV row repeats P_a / P_c.
- Per-family GMR columns: device-bootstrap CI over the family's devices, 10000 resamples, numpy default_rng(0) (i3_2_bias.paired_ratio); the B3_d3 and M_bias_lvt columns per family are computed here.
- B3_d3 share of the M_bias log gain over C0_refit: nMOS 0.721, pMOS 1.340, all 0.795.


### I8-1 (NIST 4 K, lvt, 77 K card) vs I9-1 (NIST 4 K, regular Vt, room-temperature card)

| item | I8-1 | I9-1 |
|---|---|---|
| devices | nfet_01v8_lvt / pfet_01v8_lvt; DEV 57 (33 / 24), 53 scored (4 start failures, A1) | nfet_01v8 / pfet_01v8; DEV 87 (48 / 39), 87 scored |
| card | 77 K paper card, explicit bin (25 native, 32 forced nearest bin) | open sky130 tt room-temperature card (open_pdks a918dc7), native bin for all 87 |
| deck temperature | 77 K | 30 C (cards' tnom) |
| measured temperature | 4 K | 4 K |
| stock card mean RRMS | 0.6775 | 1.6270 |
| C0 / C0_refit / M_bias mean RRMS | 0.0411 / 0.0400 / 0.0315 | 0.0576 / 0.0555 / 0.0455 |
| S1_lodo_D (P_a input) | 0.5025 (prediction.json) | 0.2948 |
| P_a: predicted y / observed y | -0.2362 / -0.2157; |error| 0.0205; correct | -0.1354 / -0.2266; |error| 0.0911; correct |
| P_b: GMR(M_bias, C0_refit) | 0.806 [0.758, 0.864], 48/53; supported | 0.797 [0.748, 0.859], 77/87; supported |
| P_c: S3_D | -0.1081 (prediction_v2.json); correct | -0.0710; correct |
| nMOS | y -0.213; GMR 0.808 [0.780, 0.840] 28/29; rule -0.221 (correct); S3 -0.107 (correct) | y -0.361; GMR 0.697 [0.672, 0.724] 47/48; rule -0.279 (correct); S3 -0.157 (correct) |
| pMOS | y -0.218; GMR 0.804 [0.707, 0.934] 20/24; rule -0.239 (correct); S3 -0.102 (correct) | y -0.061; GMR 0.941 [0.842, 1.085] 30/39; rule -0.082 (correct); S3 +0.037 (wrong) |
| M_bias vs T1_inloop | 0.887 [0.852, 0.930], 47/53 | 0.815 [0.782, 0.851], 75/87 |
| M_bias vs B1 / B2 | 0.787 [0.739, 0.845] / 0.811 [0.759, 0.874] | 0.780 [0.732, 0.839] / 0.804 [0.755, 0.866] |
| M_bias vs B3_d3 (S_B3) | 0.981 [0.961, 1.000], 27/53 | 0.955 [0.926, 0.985], 55/87 |
| B3_d3 share of the log gain | 0.911 | 0.795 |
| M_bias_lvt (transfer) | not run (training source) | vs C0_refit 0.887 [0.828, 0.962]; vs M_bias 1.112 [1.071, 1.156] |
| queue wall time | start 411 s, stats 1264 + 992 s, lodo 1016 s, baselines 393 s | start 472 s, stats 4352 s, lodo 2435 s, baselines 509 s |

- I8-1 values from out/research/explore/I8_1/decision.json, out/research/explore/I8_1/prediction.json, out/research/explore/I8_1/prediction_v2.json, out/research/explore/I8_1/lodo/results.json, out/research/explore/I8_1/baselines/results.json, out/research/explore/I8_1/start/summary.json (checked by plot_explore_i8_1.py); I9-1 values recomputed here.
- Both: NIST mds2-2997 MPW-5 tile at 4 K, Vb = 0 curves (9 per device), measured-only inclusion, the 43-parameter box, leave-one-device-out, strict paper RRMS, ngspice-41 arm64, mult = m = 1. Means over the scored devices.
- I8-1 per-family S1_lodo / S3 values come from prediction_v2.json (amendment A1); I9-1 from prediction.json.

