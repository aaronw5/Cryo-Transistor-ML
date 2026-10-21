# Tables, 2026-10-21 i10_1

Generated 2026-10-21 12:45 by `PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i10_1.py --date 2026-10-21` from stored files only. NIST DEV decides; the run is invalid (V2), so the NIST K labels are descriptive. Strict paper RRMS on the held-out 4 K Vb = 0 curves.

### I10-1 decision and validity (NIST DEV decides; TU Delft replication)

| item | pre-registered definition | value | outcome |
|---|---|---|---|
| OUTCOME | invalid iff V1 or V2 fails on NIST DEV; action 'stop: fix and re-register' | V2 fails on NIST | INVALID (stopped by rule) |
| validity V1 (NIST) | >= 90% of 67 with finite C0_RT and C4 records and >= 2 included 4 K curves | 66/67 (A1 rule excluded 1: pfet_W0p42_L8_m8395p10, non-finite C0_RT) | holds |
| validity V2 (NIST): failed per arm | A0, A0_rtdeck, A1, A2, A3, R_pop each <= 10% failed (<= 6.7 of 67; counted over the 66 V1 devices) | A0 0, A0_rtdeck 0, A1 8, A2 8, A3 8, R_pop 12 | FAILS: A1 8 = 11.9%, A2 8 = 11.9%, A3 8 = 11.9%, R_pop 12 = 17.9% |
| validity V2 (NIST): paired per contrast | every K contrast >= 80% paired (>= 54 of 67) | K1 58, K2 58, K3 51, K4 58, K5 55 | FAILS: K3 51 = 76% |
| validity V3 | prereg.json tracked, unmodified, in an ancestor of HEAD (start, lodo, decide) | prereg 17c96a1 in force; guard ok in lodo and decide | holds |
| validity V4 | ngspice timeouts reported | start 0, lodo NIST 0, lodo TU Delft 0 | reported |
| K1 (NIST, descriptive) | GMR(A1, A0), device bootstrap, 99% (Bonferroni) CI | 0.299 [0.211, 0.412], 55/58 | lower (descriptive) |
| K2 (NIST, descriptive) | GMR(A1, A0_rtdeck), device bootstrap, 99% (Bonferroni) CI | 0.871 [0.730, 1.053], 32/58 | inconclusive (descriptive) |
| K3 (NIST, descriptive) | GMR(A1, R_pop), device bootstrap, 99% (Bonferroni) CI | 0.950 [0.824, 1.082], 23/51 | inconclusive (descriptive) |
| K4 (NIST, descriptive) | GMR(A2, A1), device bootstrap, 99% (Bonferroni) CI | 0.819 [0.745, 0.890], 50/58 | lower (descriptive) |
| K5 (NIST, descriptive) | GMR(A3, A1), device bootstrap, 99% (Bonferroni) CI | 1.143 [0.887, 1.468], 21/55 | inconclusive (descriptive) |
| TEST consequences | TEST primary arm, confirmatory set, variant | none (outcome invalid); variant rule: no variant with 95% CI upper < 1 vs A1 | TEST stays locked |
| TU Delft replication | same contrasts; changes no label | V1 45/45, V2 holds (A3 2 failed); labels K1 inconclusive, K2 higher, K3 lower, K4 lower, K5 higher | valid; direction agrees with NIST on K3, K4, K5 |

- Pre-registration out/research/explore/I10_1/prereg.json (in force from 17c96a1); decision out/research/explore/I10_1/decision.json: outcome 'invalid', action 'stop: validity check failed; fix and re-register'. Every value is recomputed here from the per-device scores in lodo_nist/results.json and lodo_tud/results.json and checked against decision.json.
- The A1, A2, A3 failures are the same 8 devices (A2 multiplies the A1 output by exp(g)); with the A1-rule device each method arm has 9 non-finite scores of 67. Failure cause: table 08.
- Because the run is invalid, the K labels are descriptive: they are not the pre-registered decision.


### I10-1 NIST arms: RRMS aggregates and failures (67 DEV devices)

| arm | finite / failed (67) | failed nMOS / pMOS | mean RRMS (finite) | median RRMS | combined (nMOS / pMOS means) | nMOS mean (36) | pMOS mean (31) | GMR(arm, A0_rtdeck) [95% CI], wins | failure causes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A0: C0_RT at the 4 K deck | 66/1 | 0 / 1 | 4.0230 | 0.6205 | 3.8857 | 5.3959 | 2.3755 | 2.882 [2.266, 3.746], 4/66 | 1 A1-rule device |
| A0_rtdeck: C0_RT at the RT deck | 66/1 | 0 / 1 | 0.4596 | 0.3297 | 0.4494 | 0.5625 | 0.3362 |  | 1 A1-rule device |
| A1: C0_RT + median shift | 58/9 | 6 / 3 | 0.4114 | 0.3008 | 0.4118 | 0.4009 | 0.4227 | 0.871 [0.759, 1.002], 32/58 | 1 A1-rule device; 6 Lpeb < -Leff, 2 Pclm <= 0 |
| A2: A1 x exp(g) (M_bias) | 58/9 | 6 / 3 | 0.3793 | 0.2819 | 0.3799 | 0.3620 | 0.3978 | 0.713 [0.596, 0.853], 40/58 | 1 A1-rule device; 6 Lpeb < -Leff, 2 Pclm <= 0 |
| A3: A1 + ridge term | 58/9 | 6 / 3 | 0.4680 | 0.3926 | 0.4679 | 0.4718 | 0.4639 | 1.036 [0.866, 1.241], 23/58 | 1 A1-rule device; 6 Lpeb < -Leff, 2 Pclm <= 0 |
| R_pop: median training C4 | 54/13 | 4 / 9 | 0.3806 | 0.3011 | 0.3933 | 0.3248 | 0.4618 | 0.826 [0.707, 0.973], 39/54 | 1 A1-rule device; 12 Pclm <= 0 |
| R_oracle: own C4 (uses 4 K data) | 66/1 | 0 / 1 | 0.0661 | 0.0307 | 0.0625 | 0.1026 | 0.0223 | 0.110 [0.085, 0.141], 66/66 | 1 A1-rule device |
| A1_L: A1 + geometry ridge | 56/11 | 9 / 2 | 0.4010 | 0.3825 | 0.3999 | 0.3686 | 0.4312 | 0.913 [0.768, 1.084], 28/56 | 1 A1-rule device; 10 Pclm <= 0 or Lpeb < -Leff (analytic) |
| R_stock4k: stock card, 4 K deck | 66/1 | 0 / 1 | 3.5217 | 1.0858 | 3.5053 | 3.6861 | 3.3244 | 4.137 [3.451, 5.034], 1/66 | 1 A1-rule device |
| R_stock_rtdeck: stock card, RT deck | 66/1 | 0 / 1 | 0.4917 | 0.3813 | 0.4827 | 0.5817 | 0.3838 | 1.071 [1.013, 1.148], 28/66 | 1 A1-rule device |
| V_ind: C4 from stock | 53/14 | 11 / 3 | 1.1732 | 0.4337 | 1.2175 | 2.0001 | 0.4349 | 1.321 [1.039, 1.757], 20/53 | 1 A1-rule device; 13 Pclm <= 0 or Lpeb < -Leff (analytic) |
| V_mean: mean shift | 53/14 | 6 / 8 | 0.6345 | 0.4691 | 0.6533 | 0.5110 | 0.7956 | 1.265 [1.055, 1.551], 21/53 | 1 A1-rule device; 12 Pclm <= 0 or Lpeb < -Leff (analytic), 1 other |
| V_sub7: 7-parameter shift | 66/1 | 0 / 1 | 0.4377 | 0.3872 | 0.4434 | 0.3801 | 0.5067 | 0.973 [0.854, 1.104], 36/66 | 1 A1-rule device |
| V_deck: C4 and A1 at -269.15 C | 52/15 | 2 / 13 | 3.9826 | 0.3373 | 3.5171 | 5.0297 | 2.0046 | 1.722 [1.231, 2.461], 27/52 | 1 A1-rule device; 14 at the -269.15 C deck (2 Pclm <= 0 analytic, 12 not diagnosed) |

- Source out/research/explore/I10_1/lodo_nist/results.json; strict paper RRMS on the held-out device's 4 K Vb = 0 curves, measured-only inclusion frozen from the measured 4 K current. Means over finite devices only (the device sets differ between arms). GMR(arm, A0_rtdeck) is computed here (post hoc, device bootstrap 95%).
- Failure causes: out/research/explore/I10_1/diagnosis/diagnosis.json (ngspice .op messages for A1, A3, R_pop; analytic BSIM4 binning check for the other arms). The C0_RT card of etest_pfet_g5v0d10v5_W0p42_L8_m8395p10 has Pclm <= 0 (A1 rule: failed in every arm).


### I10-1 per family: mean RRMS and K1-K5 (NIST descriptive; TU Delft replication)

| dataset | family | devices | A0_rtdeck mean (failed) | A1 mean (failed) | A2 mean (failed) | R_pop mean (failed) | R_oracle mean (failed) | K1 GMR [99% CI], wins | K2 GMR [99% CI], wins | K3 GMR [99% CI], wins | K4 GMR [99% CI], wins | K5 GMR [99% CI], wins |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NIST | nMOS | 36 | 0.562 (0) | 0.401 (6) | 0.362 (6) | 0.325 (4) | 0.103 (0) | 0.208 [0.120, 0.339], 29/30 | 0.628 [0.522, 0.749], 24/30 | 1.056 [0.907, 1.215], 10/29 | 0.782 [0.689, 0.879], 30/30 | 1.177 [0.761, 1.800], 12/28 |
| NIST | pMOS | 31 | 0.336 (1) | 0.423 (3) | 0.398 (3) | 0.462 (9) | 0.022 (1) | 0.442 [0.322, 0.597], 26/28 | 1.235 [0.995, 1.564], 8/28 | 0.827 [0.656, 1.019], 13/22 | 0.861 [0.761, 0.966], 20/28 | 1.109 [0.851, 1.375], 9/27 |
| NIST | all | 67 | 0.460 (1) | 0.411 (9) | 0.379 (9) | 0.381 (13) | 0.066 (1) | 0.299 [0.211, 0.412], 55/58 | 0.871 [0.730, 1.053], 32/58 | 0.950 [0.824, 1.082], 23/51 | 0.819 [0.745, 0.890], 50/58 | 1.143 [0.887, 1.468], 21/55 |
| TU Delft | nMOS | 29 | 0.408 (0) | 0.409 (0) | 0.324 (0) | 2.908 (0) | 0.057 (0) | 0.436 [0.324, 0.564], 29/29 | 1.062 [0.779, 1.444], 15/29 | 0.180 [0.128, 0.255], 29/29 | 0.817 [0.706, 0.971], 24/29 | 1.973 [1.220, 3.015], 4/27 |
| TU Delft | pMOS | 16 | 0.403 (0) | 17.459 (0) | 14.938 (0) | 16.217 (0) | 0.076 (0) | 10.677 [2.010, 56.833], 8/16 | 10.731 [1.939, 59.312], 8/16 | 0.739 [0.578, 0.953], 12/16 | 0.878 [0.861, 0.898], 16/16 | 1.369 [1.031, 1.851], 7/16 |
| TU Delft | all | 45 | 0.407 (0) | 6.471 (0) | 5.520 (0) | 7.640 (0) | 0.064 (0) | 1.359 [0.619, 3.509], 37/45 | 2.418 [1.182, 5.597], 23/45 | 0.297 [0.209, 0.418], 41/45 | 0.838 [0.761, 0.939], 40/45 | 1.722 [1.249, 2.309], 11/43 |

- K1 GMR(A1, A0), K2 GMR(A1, A0_rtdeck), K3 GMR(A1, R_pop), K4 GMR(A2, A1), K5 GMR(A3, A1). Per-family rows are pre-registered secondaries. NIST rows are descriptive (run invalid).
- TU Delft: A0 uses kT/q only (I6-1 card). TU Delft pMOS has 2 geometry groups, so each pMOS fold trains on the other group only; the 8 devices of pfet W1 L0.04 have A1 RRMS 28.7-46.8 (A0_rtdeck 0.18-0.24), which gives the pMOS K1 and K2 GMR of about 10.7.


### I10-1 TU Delft replication next to NIST

| contrast | TU Delft GMR [99% device CI], wins | TU Delft [99% cluster CI, 7 groups] | TU Delft label (device CI) | TU Delft label (cluster CI) | NIST GMR [99% CI], wins | NIST label | same direction |
|---|---:|---:|---|---|---:|---|---|
| K1 GMR(A1, A0) (primary) | 1.359 [0.619, 3.509], 37/45 | 1.359 [0.288, 21.470] | inconclusive | inconclusive | 0.299 [0.211, 0.412], 55/58 | lower (descr.) | no |
| K2 GMR(A1, A0_rtdeck) | 2.418 [1.182, 5.597], 23/45 | 2.418 [0.632, 28.647] | higher | inconclusive | 0.871 [0.730, 1.053], 32/58 | inconclusive (descr.) | no |
| K3 GMR(A1, R_pop) | 0.297 [0.209, 0.418], 41/45 | 0.297 [0.110, 0.693] | lower | lower | 0.950 [0.824, 1.082], 23/51 | inconclusive (descr.) | yes |
| K4 GMR(A2, A1) | 0.838 [0.761, 0.939], 40/45 | 0.838 [0.717, 1.035] | lower | inconclusive | 0.819 [0.745, 0.890], 50/58 | lower (descr.) | yes |
| K5 GMR(A3, A1) | 1.722 [1.249, 2.309], 11/43 | 1.722 [1.048, 2.622] | higher | higher | 1.143 [0.887, 1.468], 21/55 | inconclusive (descr.) | yes |
| validity | V1 45/45; V2 failed A0 0, A0_rtdeck 0, A1 0, A2 0, A3 2, R_pop 0 |  | valid |  | V2 fails | invalid |  |
| mean RRMS A0 / A0_rtdeck / A1 / A2 / R_pop / R_oracle | 0.747 / 0.407 / 6.471 / 5.520 / 7.640 / 0.064 |  |  |  | 4.023 / 0.460 / 0.411 / 0.379 / 0.381 / 0.066 |  |  |
| median RRMS A0_rtdeck / A1 / A2 / R_oracle | 0.363 / 0.448 / 0.375 / 0.044 |  |  |  | 0.330 / 0.301 / 0.282 / 0.031 |  |  |
| gap fraction toward R_oracle A1 / A2 / A3 | -0.117 / -0.050 / -0.349 |  |  |  | +0.354 / +0.413 / +0.303 |  |  |

- TU Delft: I6-1 DEV devices, stored 295 K cards (C0_RT), C4 warm start from them at the 295 K stage deck; A0 at 4.13 K changes kT/q only. Labels from the device-bootstrap CI (as decision.json); cluster labels are computed here. The replication changes no label.
- Gap fraction = ln GMR(arm, A0) / ln GMR(R_oracle, A0) (decision.json; 1 = R_oracle, 0 = A0).


### I10-1 failure diagnosis: failed NIST records and the BSIM4 check that stops ngspice

| device | type | W / L (um) | bin | C0_RT effective Pclm; Lpeb / Leff | A1 (= A2) fatal check, effective value | A3 fatal check | R_pop fatal check | C0_RT fatal check |
|---|---|---:|---|---|---|---|---|---|
| nfet_W0p75_L0p4_m8053p5 | nMOS | 0.75 / 0.4 | stretch L x1.250 | Pclm 0.550; Lpeb/Leff -0.999 | Lpeb/Leff -1.165 | Lpeb/Leff -1.373 | Pclm -1.400 |  |
| nfet_W1_L0p4_m8053p6 | nMOS | 1 / 0.4 | stretch L x1.250 | Pclm 0.356; Lpeb/Leff -0.918 | Lpeb/Leff -1.071 |  | Pclm -0.431 |  |
| nfet_W3_L0p4_m8053p8 | nMOS | 3 / 0.4 | stretch L x1.250 | Pclm 0.479; Lpeb/Leff -0.896 | Lpeb/Leff -1.046 | Lpeb/Leff -1.079 |  |  |
| nfet_W20_L0p4_m8053p11 | nMOS | 20 / 0.4 | stretch L x1.250 | Pclm 1.694; Lpeb/Leff -0.860 | Lpeb/Leff -1.003 | Lpeb/Leff -1.049 |  |  |
| nfet_W0p75_L0p45_m8043p5 | nMOS | 0.75 / 0.45 | stretch L x1.111 | Pclm 0.808; Lpeb/Leff -0.873 | Lpeb/Leff -1.019 |  | Pclm -1.590 |  |
| nfet_W1_L0p45_m8043p6 | nMOS | 1 / 0.45 | stretch L x1.111 | Pclm 0.621; Lpeb/Leff -0.819 |  | Lpeb/Leff -1.006 |  |  |
| nfet_W3_L0p45_m8043p8 | nMOS | 3 / 0.45 | stretch L x1.111 | Pclm 0.599; Lpeb/Leff -0.822 |  | Lpeb/Leff -1.022 |  |  |
| nfet_W20_L0p45_m8043p11 | nMOS | 20 / 0.45 | stretch L x1.111 | Pclm 1.461; Lpeb/Leff -0.984 | Lpeb/Leff -1.148 | Lpeb/Leff -1.327 |  |  |
| nfet_W0p42_L8_m8389p9 | nMOS | 0.42 / 8 | in bin | Pclm 0.288; Lpeb/Leff -0.026 |  |  | Pclm -0.017 |  |
| pfet_W1p5_L0p35_m8064p7 | pMOS | 1.5 / 0.35 | stretch L x1.429 | Pclm 0.459; Lpeb/Leff 0.000 |  |  | Pclm -0.142 |  |
| pfet_W0p42_L0p4_m8054p4 | pMOS | 0.42 / 0.4 | stretch L x1.250 | Pclm 0.238; Lpeb/Leff 0.000 |  | Pclm -0.051 |  |  |
| pfet_W0p75_L0p4_m8054p5 | pMOS | 0.75 / 0.4 | stretch L x1.250 | Pclm 0.534; Lpeb/Leff 0.000 | Pclm -0.049 |  | Pclm -0.903 |  |
| pfet_W1p5_L0p4_m8054p7 | pMOS | 1.5 / 0.4 | stretch L x1.250 | Pclm 0.624; Lpeb/Leff 0.000 |  |  | Pclm -0.058 |  |
| pfet_W3_L0p4_m8054p8 | pMOS | 3 / 0.4 | stretch L x1.250 | Pclm 0.426; Lpeb/Leff 0.000 |  |  | Pclm -0.085 |  |
| pfet_W0p75_L0p45_m8044p5 | pMOS | 0.75 / 0.45 | stretch L x1.111 | Pclm 0.585; Lpeb/Leff 0.000 |  |  | Pclm -0.734 |  |
| pfet_W1_L0p45_m8044p6 | pMOS | 1 / 0.45 | stretch L x1.111 | Pclm 0.642; Lpeb/Leff 0.000 |  |  | Pclm -0.367 |  |
| pfet_W1_L4_m8364p12 | pMOS | 1 / 4 | in bin | Pclm 0.720; Lpeb/Leff 0.000 |  |  | Pclm -0.167 |  |
| pfet_W0p42_L8_m8395p10 | pMOS | 0.42 / 8 | in bin | Pclm -0.082; Lpeb/Leff 0.000 |  |  |  | Pclm -0.082 |
| pfet_W0p42_L20_m8364p4 | pMOS | 0.42 / 20 | in bin | Pclm 0.101; Lpeb/Leff 0.000 | Pclm -0.133 | Pclm -0.037 | Pclm -0.548 |  |

- Source out/research/explore/I10_1/diagnosis/diagnosis.json (post hoc). Each failed record of A1, A3, R_pop and the excluded C0_RT record was run once as an ngspice .op (29 runs, 0.9 s); every run stops with one BSIM4 parameter-check fatal: 'Pclm = ... is not positive' or 'Lpeb = ... is less than -Leff'. Lpeb/Leff below -1 is fatal.
- Effective value = P0 + lP/Leff + wP/Weff + pP/(Leff Weff) of the device's bin (binunit 2). The analytic check predicts every failure of A1 (8/8), A2 (= A1), A3 (8/8), R_pop (12/12), A1_L (10/10), V_ind (13/13), C0_RT (1/1); V_sub7 (7 parameters, no PCLM / LPEB) has 0 failures. R_pop: 6 more cards sit at Lpeb = -Leff (relative 5e-11; the check is strict, finite).
- Mechanism: the I8-1 box bounds the base parameter P0 (log scale, sign kept), not the size-dependent value. For the listed geometries the C0_RT effective value lies near the limit (A1 / A3 Lpeb failures, nMOS L 0.40-0.45 um, bin-stretched: Lpeb/Leff -0.999 to -0.819 at C0_RT; A1 / A3 Pclm failures, pMOS: Pclm 0.101 to 0.534 at C0_RT, with negative l / w terms), and the training median shift (or the R_pop coordinates) moves P0 across it.

