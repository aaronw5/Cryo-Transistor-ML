# Tables, 2026-10-26 i22_1

Generated 2026-10-26 19:26 by `.venv/bin/python -I scripts/research/plot_explore_i22_1.py --date 2026-10-26`.

### I22-1 setup (pre-registered)

| item | value |
|---|---|
| pre-registration | out/research/explore/I22_1/prereg.md (commit 02113d4); code 422551a; amendment A1 (before any run): TuRBO runs stop at B_d; DEV only, TEST locked; no amendment of a755e18 |
| Part A datasets | sky130: I13_1/records, LODO 18; nist4: I20_1/start/records_nist4, LODO; nistrt: I20_1/start/records_nistrt, LODO; ihp233: I5_1/start/records_233K, LODO within class; ihp300: I5_1/start/records_300K, LODO within class; ihp343: I5_1/start/records_343K, LODO within class; ihp398: I5_1/start/records_398K, LODO within class; tud4: I18_2/start/records_4K, LOGO 7 groups; tud295: I18_2/start/records_295K, LOGO 7 groups |
| Part A arms | C0_refit_rerun: no correction, polish max_nfev 400 irls 2 diff_step 0.01, state C; validity VA1 and cost; B1: base box + extras of the I4-1 edge screen (threshold 1e-4) recomputed on state-C starting records of all DEV devices (IHP: stored I5-1 per-class 300 K screen); max_nfev matched_nfev; B2: base box + rext_ohm [0, 4 x series R per side / W] + heat_kappa [-0.5, 0.5]; max_nfev matched_nfev; B3_d3: PolyCorrection degree 3 per polarity, HYPER_POLY (l2 1e-4, lr 3e-3, 2000 steps), per-fold training records, frozen in refit (max_nfev 400); B4: GPCorrection fit_gp (2000 points, rng 0, L-BFGS-B maxiter 200), per fold, frozen in refit (max_nfev 400) |
| Part A comparison rows | stored C0_refit, M_bias v1, selected g v2 rows via scripts/research/plot_freeze_draft.py datasets() |
| Part A outcome rule | X matches g v2 iff non-inferior on all 3 of G; g v2 ahead of X iff 'g v2 better than X' on all 3 of G; outcome 'selected g v2 not matched by a correction-stage baseline' iff no X matches, else 'matched by X' |
| Part B | one NGSpice call = one simulation of a device's measured bias set (polish n_sim); report only |
| Part C | C0_refit stage: state-C starting card direct43_r3_fd, 43-parameter round-3 box, local [-1, 1], DeviceProblem; objective device paper RRMS = mean frozen included curve RRMS; non-finite = 10; budget each CMA-ES / TuRBO run 4 B_d calls; matched result = best-so-far at B_d; secondary at 4 B_d (A1: TuRBO to B_d); per device exp(mean over 3 seeds ln best-so-far f at B_d); GMR(method, LS), device bootstrap 10000 seed 0 |


### Part A validity per dataset (VA1-VA5)

| dataset | VA1 max |C0_refit rerun - stored| (<= 1e-12) | VA2 non-finite rows | VA3 unit tests | VA4 DEV only | VA5 IHP rerun vs I5-1 (reported) | result |
|---|---:|---:|---:|---:|---|---|
| sky130 77 K (18, LODO) | 0.0e+00 | 0 failed | True | True | - | valid |
| NIST sky130 4 K lvt (53) | 0.0e+00 | 0 failed | True | True | - | valid |
| NIST sky130 4 K RT card (87) | 0.0e+00 | 0 failed | True | True | - | valid |
| IHP SG13G2 233 K | 0.0e+00 | 0 failed | True | True | B1 0.0e+00, B2 0.0e+00, B3_d3 0.0e+00 | valid |
| IHP SG13G2 300 K | 0.0e+00 | 0 failed | True | True | B1 0.0e+00, B2 0.0e+00, B3_d3 0.0e+00 | valid |
| IHP SG13G2 343 K | 0.0e+00 | 0 failed | True | True | B1 0.0e+00, B2 0.0e+00, B3_d3 0.0e+00 | valid |
| IHP SG13G2 398 K | 0.0e+00 | 0 failed | True | True | B1 0.0e+00, B2 0.0e+00, B3_d3 0.0e+00 | valid |
| TU Delft 4.13 K | 0.0e+00 | 0 failed | True | True | - | valid |
| TU Delft 295.0 K | 0.0e+00 | 0 failed | True | True | - | valid |

- VA1: the C0_refit rerun through the I22-1 refit path against the stored C0_refit rows of the g v2 comparison (state C; IHP no clamp).


### Part A held-out RRMS per arm: all-device mean / combined (mean of the nMOS and pMOS family means)

| dataset | n (CI unit) | C0_refit | M_bias v1 | selected g v2 | B1 wider card set | B2 R_ext + self-heating | B3_d3 cubic g | B4 GP discrepancy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sky130 77 K (18, LODO) | 18 (device) | 0.1051 / 0.0984 | 0.0735 / 0.0695 | 0.0739 / 0.0700 | 0.1054 / 0.0988 | 0.1049 / 0.0983 | 0.0900 / 0.0847 | 0.1330 / 0.1246 |
| NIST sky130 4 K lvt (53) | 53 (device) | 0.0399 / 0.0427 | 0.0312 / 0.0334 | 0.0318 / 0.0340 | 0.0408 / 0.0437 | 0.0389 / 0.0415 | 0.0316 / 0.0338 | 0.0546 / 0.0592 |
| NIST sky130 4 K RT card (87) | 87 (device) | 0.0567 / 0.0593 | 0.0466 / 0.0491 | 0.0466 / 0.0491 | 0.0577 / 0.0604 | 0.0564 / 0.0590 | 0.0465 / 0.0487 | 0.0625 / 0.0664 |
| IHP SG13G2 233 K | 18 (device) | 0.0273 / 0.0273 | 0.0281 / 0.0281 | 0.0279 / 0.0279 | 0.0235 / 0.0235 | 0.0240 / 0.0240 | 0.0295 / 0.0295 | 0.0301 / 0.0301 |
| IHP SG13G2 300 K | 18 (device) | 0.0191 / 0.0191 | 0.0203 / 0.0203 | 0.0205 / 0.0205 | 0.0160 / 0.0160 | 0.0176 / 0.0176 | 0.0214 / 0.0214 | 0.0210 / 0.0210 |
| IHP SG13G2 343 K | 18 (device) | 0.0190 / 0.0190 | 0.0194 / 0.0194 | 0.0194 / 0.0194 | 0.0153 / 0.0153 | 0.0150 / 0.0150 | 0.0208 / 0.0208 | 0.0237 / 0.0237 |
| IHP SG13G2 398 K | 18 (device) | 0.0197 / 0.0197 | 0.0202 / 0.0202 | 0.0202 / 0.0202 | 0.0183 / 0.0183 | 0.0151 / 0.0151 | 0.0217 / 0.0217 | 0.0287 / 0.0287 |
| TU Delft 4.13 K | 45 (cluster (7 groups)) | 0.0602 / 0.0532 | 0.0534 / 0.0491 | 0.0484 / 0.0449 | 0.0633 / 0.0560 | 0.0593 / 0.0521 | 0.0543 / 0.0499 | 0.1493 / 0.1254 |
| TU Delft 295.0 K | 45 (cluster (7 groups)) | 0.0265 / 0.0251 | 0.0208 / 0.0210 | 0.0215 / 0.0218 | 0.0297 / 0.0289 | 0.0266 / 0.0247 | 0.0232 / 0.0230 | 0.0342 / 0.0346 |

- C0_refit, M_bias v1, selected g v2: stored rows (freeze-draft loader). B1-B4: I22-1 rows. Strict paper RRMS, frozen curve inclusion; a failed (non-finite) device is left out of the means and counted.


### Part A GMR(baseline, selected g v2) with 95% bootstrap CI, devices lower / paired, label; NI: non-inferior (CI upper < 1.05)

| dataset | B1 wider card set | B2 R_ext + self-heating | B3_d3 cubic g | B4 GP discrepancy |
|---|---:|---:|---:|---:|
| sky130 77 K (18, LODO) (G) | 1.373 [1.191, 1.556] 1/18; g v2 better | 1.363 [1.182, 1.547] 1/18; g v2 better | 1.227 [1.125, 1.331] 1/18; g v2 better | 1.576 [1.355, 1.867] 1/18; g v2 better |
| NIST sky130 4 K lvt (53) (G) | 1.285 [1.205, 1.364] 3/53; g v2 better | 1.241 [1.158, 1.318] 3/53; g v2 better | 1.019 [0.987, 1.053] 26/53; inconclusive | 1.086 [0.964, 1.236] 28/53; inconclusive |
| NIST sky130 4 K RT card (87) (G) | 1.259 [1.172, 1.339] 10/87; g v2 better | 1.223 [1.138, 1.303] 10/87; g v2 better | 1.036 [1.002, 1.072] 33/87; g v2 better | 1.343 [1.230, 1.465] 31/87; g v2 better |
| IHP SG13G2 233 K | 0.733 [0.551, 0.938] 12/18; X better; NI | 0.845 [0.660, 1.045] 8/18; inconclusive; NI | 1.142 [1.027, 1.316] 8/18; g v2 better | 1.338 [1.022, 1.821] 5/18; g v2 better |
| IHP SG13G2 300 K | 0.617 [0.476, 0.786] 14/18; X better; NI | 0.753 [0.590, 0.928] 9/18; X better; NI | 1.096 [1.008, 1.202] 7/18; g v2 better | 1.128 [0.950, 1.379] 7/18; inconclusive |
| IHP SG13G2 343 K | 0.585 [0.439, 0.755] 14/18; X better; NI | 0.748 [0.578, 0.933] 11/18; X better; NI | 1.104 [1.019, 1.205] 4/18; g v2 better | 1.470 [1.100, 2.116] 4/18; g v2 better |
| IHP SG13G2 398 K | 0.642 [0.475, 0.837] 12/18; X better; NI | 0.703 [0.549, 0.875] 11/18; X better; NI | 1.083 [0.998, 1.202] 6/18; inconclusive | 1.601 [1.152, 2.398] 5/18; g v2 better |
| TU Delft 4.13 K | 1.288 [1.030, 1.637] 14/45; g v2 better | 1.181 [0.903, 1.543] 16/45; inconclusive | 1.139 [1.034, 1.271] 7/45; g v2 better | 2.795 [1.585, 4.744] 6/45; g v2 better |
| TU Delft 295.0 K | 1.340 [0.848, 2.159] 14/45; inconclusive | 1.180 [0.760, 1.934] 19/45; inconclusive | 1.112 [0.967, 1.307] 16/45; inconclusive | 1.643 [1.476, 1.806] 1/45; g v2 better |

- GMR < 1: the baseline has lower RRMS than selected g v2. Bootstrap 10000, seed 0; unit device (TU Delft: 7 geometry groups). (G): gain set of the outcome rule.


### Part A GMR(arm, C0_refit) with 95% bootstrap CI and label (gain / worse / inconclusive); pooled over the 9 datasets

| dataset | selected g v2 | M_bias v1 | B1 wider card set | B2 R_ext + self-heating | B3_d3 cubic g | B4 GP discrepancy |
|---|---:|---:|---:|---:|---:|---:|
| sky130 77 K (18, LODO) | 0.743 [0.654, 0.855] 17/18; gain | 0.745 [0.660, 0.850] 17/18; gain | 1.020 [0.999, 1.056] 8/18; inconclusive | 1.013 [0.994, 1.045] 11/18; inconclusive | 0.912 [0.852, 0.977] 14/18; gain | 1.171 [1.009, 1.368] 6/18; worse |
| NIST sky130 4 K lvt (53) | 0.803 [0.757, 0.857] 50/53; gain | 0.798 [0.750, 0.857] 50/53; gain | 1.032 [1.019, 1.049] 16/53; worse | 0.997 [0.977, 1.018] 32/53; inconclusive | 0.819 [0.766, 0.883] 48/53; gain | 0.873 [0.747, 1.022] 37/53; inconclusive |
| NIST sky130 4 K RT card (87) | 0.811 [0.761, 0.871] 77/87; gain | 0.809 [0.759, 0.873] 78/87; gain | 1.021 [1.013, 1.029] 29/87; worse | 0.992 [0.982, 1.001] 57/87; inconclusive | 0.840 [0.784, 0.910] 77/87; gain | 1.089 [0.965, 1.238] 46/87; inconclusive |
| IHP SG13G2 233 K | 1.084 [0.898, 1.371] 11/18; inconclusive | 1.106 [0.908, 1.425] 11/18; inconclusive | 0.794 [0.659, 0.925] 16/18; gain | 0.916 [0.817, 0.989] 16/18; gain | 1.238 [0.997, 1.617] 11/18; inconclusive | 1.450 [1.096, 1.932] 4/18; worse |
| IHP SG13G2 300 K | 1.255 [1.026, 1.600] 9/18; worse | 1.214 [1.008, 1.537] 9/18; worse | 0.775 [0.653, 0.903] 17/18; gain | 0.946 [0.889, 0.996] 17/18; gain | 1.375 [1.109, 1.815] 6/18; worse | 1.416 [1.109, 1.834] 6/18; worse |
| IHP SG13G2 343 K | 1.188 [0.990, 1.503] 10/18; inconclusive | 1.181 [0.983, 1.502] 8/18; inconclusive | 0.695 [0.579, 0.816] 17/18; gain | 0.889 [0.757, 0.996] 15/18; gain | 1.311 [1.055, 1.768] 8/18; worse | 1.747 [1.184, 2.701] 7/18; worse |
| IHP SG13G2 398 K | 1.260 [1.026, 1.607] 11/18; worse | 1.272 [1.030, 1.627] 11/18; worse | 0.809 [0.704, 0.914] 17/18; gain | 0.886 [0.770, 0.975] 16/18; gain | 1.364 [1.065, 1.869] 10/18; worse | 2.018 [1.256, 3.409] 6/18; worse |
| TU Delft 4.13 K | 0.826 [0.669, 1.014] 30/45; inconclusive | 0.909 [0.766, 1.077] 30/45; inconclusive | 1.064 [1.029, 1.116] 14/45; worse | 0.975 [0.906, 1.053] 23/45; inconclusive | 0.941 [0.783, 1.114] 29/45; inconclusive | 2.309 [1.604, 3.231] 2/45; worse |
| TU Delft 295.0 K | 0.829 [0.523, 1.247] 29/45; inconclusive | 0.811 [0.532, 1.190] 29/45; inconclusive | 1.110 [1.017, 1.250] 14/45; worse | 0.978 [0.927, 1.036] 31/45; inconclusive | 0.921 [0.671, 1.260] 27/45; inconclusive | 1.361 [0.800, 2.169] 21/45; inconclusive |
| pooled exp(mean ln GMR), 9 datasets | 0.957 | 0.964 | 0.913 | 0.954 | 1.058 | 1.432 |


### Part A outcome rule over the gain set G = {sky130, NIST lvt, NIST RT card}, and prediction scores

| baseline | non-inferior to g v2 (CI upper < 1.05) | g v2 better (CI lower > 1) | matches g v2 | g v2 ahead |
|---|---|---|---:|---:|
| B1 wider card set | sky130 no, nist4 no, nistrt no | sky130 yes, nist4 yes, nistrt yes | no | yes |
| B2 R_ext + self-heating | sky130 no, nist4 no, nistrt no | sky130 yes, nist4 yes, nistrt yes | no | yes |
| B3_d3 cubic g | sky130 no, nist4 no, nistrt no | sky130 yes, nist4 no, nistrt yes | no | no |
| B4 GP discrepancy | sky130 no, nist4 no, nistrt no | sky130 yes, nist4 no, nistrt yes | no | no |

- Outcome: selected g v2 not matched by a correction-stage baseline.
- P1: True
- P2: False
- P3: {'ihp233': True, 'ihp300': True, 'ihp343': True, 'ihp398': True}
- P4: {'tud4': False, 'tud295': False}
- P5: True
- P6: False
- P7: {'sky130': True, 'nist4': True, 'nistrt': True, 'ihp233': True, 'ihp300': True, 'ihp343': True, 'ihp398': True, 'tud4': True, 'tud295': True}


### Part B NGSpice calls per held-out device by arm (one call = one simulation of the device's measured bias set), trainings and training CPU seconds

| dataset | starting card calls / device (shared) | arm | pre-refit calls / device | held-out refit calls / device | arm total / device | arm total / C0_refit | trainings | training CPU s | note |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| sky130 77 K (18, LODO) | 44,219 | C0_refit | 0 | 316 | 316 | 1.0 | 0 | 0 |  |
|  |  | M_bias v1 | 0 | 326 | 326 | 1.0 | 90 | not stored |  |
|  |  | selected g v2 | 27,880 | 311 | 28,191 | 89.3 | 360 | 28,110 | 1530 selection refits (estimated); 18 grid decks x 5625 points; full-data selection 90 refits |
|  |  | B1 wider card set | 108 | 549 | 657 | 2.1 | 0 | 0 | edge screen, this run |
|  |  | B2 R_ext + self-heating | 0 | 335 | 335 | 1.1 | 0 | 0 |  |
|  |  | B3_d3 cubic g | 0 | 334 | 334 | 1.1 | 18 | 115 |  |
|  |  | B4 GP discrepancy | 0 | 334 | 334 | 1.1 | 18 | 530 |  |
| NIST sky130 4 K lvt (53) | 1,569 | C0_refit | 0 | 238 | 238 | 1.0 | 0 | 0 |  |
|  |  | M_bias v1 | 0 | 262 | 262 | 1.1 | 265 | not stored |  |
|  |  | selected g v2 | 6,423 | 259 | 6,683 | 28.1 | 355 | 46,069 | 1325 selection refits (stored); 53 grid decks x 5329 points; full-data selection 265 refits |
|  |  | B1 wider card set | 107 | 533 | 641 | 2.7 | 0 | 0 | edge screen, this run |
|  |  | B2 R_ext + self-heating | 0 | 346 | 346 | 1.5 | 0 | 0 |  |
|  |  | B3_d3 cubic g | 0 | 262 | 262 | 1.1 | 53 | 373 |  |
|  |  | B4 GP discrepancy | 0 | 259 | 259 | 1.1 | 53 | 1,688 |  |
| NIST sky130 4 K RT card (87) | 1,573 | C0_refit | 0 | 219 | 219 | 1.0 | 0 | 0 |  |
|  |  | M_bias v1 | 0 | 245 | 245 | 1.1 | 435 | not stored |  |
|  |  | selected g v2 | 5,990 | 242 | 6,232 | 28.5 | 525 | 37,685 | 2175 selection refits (stored); 87 grid decks x 5329 points; full-data selection 435 refits |
|  |  | B1 wider card set | 119 | 516 | 635 | 2.9 | 0 | 0 | edge screen, this run |
|  |  | B2 R_ext + self-heating | 0 | 283 | 283 | 1.3 | 0 | 0 |  |
|  |  | B3_d3 cubic g | 0 | 241 | 241 | 1.1 | 87 | 868 |  |
|  |  | B4 GP discrepancy | 0 | 252 | 252 | 1.2 | 87 | 2,179 |  |
| IHP SG13G2 233 K | 1,389 | C0_refit | 0 | 313 | 313 | 1.0 | 0 | 0 |  |
|  |  | M_bias v1 | 0 | 311 | 311 | 1.0 | 90 | not stored |  |
|  |  | selected g v2 | 13,951 | 307 | 14,258 | 45.6 | 360 | 14,630 | 730 selection refits (estimated); 18 grid decks x 5625 points; full-data selection 0 refits |
|  |  | B1 wider card set | 0 | 1,938 | 1,938 | 6.2 | 0 | 0 | edge screen, stored I5-1 screen, counted for 300 K only |
|  |  | B2 R_ext + self-heating | 0 | 359 | 359 | 1.1 | 0 | 0 |  |
|  |  | B3_d3 cubic g | 0 | 314 | 314 | 1.0 | 18 | 27 |  |
|  |  | B4 GP discrepancy | 0 | 312 | 312 | 1.0 | 18 | 536 |  |
| IHP SG13G2 300 K | 1,368 | C0_refit | 0 | 324 | 324 | 1.0 | 0 | 0 |  |
|  |  | M_bias v1 | 0 | 317 | 317 | 1.0 | 90 | not stored |  |
|  |  | selected g v2 | 16,060 | 319 | 16,379 | 50.5 | 360 | 12,905 | 730 selection refits (estimated); 18 grid decks x 5625 points; full-data selection 0 refits |
|  |  | B1 wider card set | 173 | 1,957 | 2,131 | 6.6 | 0 | 0 | edge screen, stored I5-1 screen, counted for 300 K only |
|  |  | B2 R_ext + self-heating | 0 | 336 | 336 | 1.0 | 0 | 0 |  |
|  |  | B3_d3 cubic g | 0 | 320 | 320 | 1.0 | 18 | 28 |  |
|  |  | B4 GP discrepancy | 0 | 324 | 324 | 1.0 | 18 | 526 |  |
| IHP SG13G2 343 K | 1,053 | C0_refit | 0 | 306 | 306 | 1.0 | 0 | 0 |  |
|  |  | M_bias v1 | 0 | 321 | 321 | 1.0 | 90 | not stored |  |
|  |  | selected g v2 | 15,796 | 331 | 16,127 | 52.6 | 360 | 15,366 | 730 selection refits (estimated); 18 grid decks x 5625 points; full-data selection 0 refits |
|  |  | B1 wider card set | 0 | 1,906 | 1,906 | 6.2 | 0 | 0 | edge screen, stored I5-1 screen, counted for 300 K only |
|  |  | B2 R_ext + self-heating | 0 | 308 | 308 | 1.0 | 0 | 0 |  |
|  |  | B3_d3 cubic g | 0 | 320 | 320 | 1.0 | 18 | 28 |  |
|  |  | B4 GP discrepancy | 0 | 325 | 325 | 1.1 | 18 | 497 |  |
| IHP SG13G2 398 K | 1,273 | C0_refit | 0 | 321 | 321 | 1.0 | 0 | 0 |  |
|  |  | M_bias v1 | 0 | 312 | 312 | 1.0 | 90 | not stored |  |
|  |  | selected g v2 | 16,060 | 315 | 16,375 | 51.0 | 360 | 15,322 | 730 selection refits (estimated); 18 grid decks x 5625 points; full-data selection 0 refits |
|  |  | B1 wider card set | 0 | 1,940 | 1,940 | 6.0 | 0 | 0 | edge screen, stored I5-1 screen, counted for 300 K only |
|  |  | B2 R_ext + self-heating | 0 | 337 | 337 | 1.0 | 0 | 0 |  |
|  |  | B3_d3 cubic g | 0 | 322 | 322 | 1.0 | 18 | 26 |  |
|  |  | B4 GP discrepancy | 0 | 327 | 327 | 1.0 | 18 | 561 |  |
| TU Delft 4.13 K | 1,846 | C0_refit | 0 | 210 | 210 | 1.0 | 0 | 0 |  |
|  |  | M_bias v1 | 0 | 227 | 227 | 1.1 | 35 | not stored |  |
|  |  | selected g v2 | 7,260 | 244 | 7,504 | 35.7 | 140 | 15,674 | 1350 selection refits (estimated); 45 grid decks x 2025 points; full-data selection 225 refits |
|  |  | B1 wider card set | 97 | 420 | 517 | 2.5 | 0 | 0 | edge screen, this run |
|  |  | B2 R_ext + self-heating | 0 | 290 | 290 | 1.4 | 0 | 0 |  |
|  |  | B3_d3 cubic g | 0 | 225 | 225 | 1.1 | 7 | 145 |  |
|  |  | B4 GP discrepancy | 0 | 304 | 304 | 1.4 | 7 | 191 |  |
| TU Delft 295.0 K | 2,128 | C0_refit | 0 | 208 | 208 | 1.0 | 0 | 0 |  |
|  |  | M_bias v1 | 0 | 254 | 254 | 1.2 | 35 | not stored |  |
|  |  | selected g v2 | 8,400 | 252 | 8,652 | 41.6 | 140 | 19,257 | 1350 selection refits (estimated); 45 grid decks x 2025 points; full-data selection 225 refits |
|  |  | B1 wider card set | 97 | 479 | 576 | 2.8 | 0 | 0 | edge screen, this run |
|  |  | B2 R_ext + self-heating | 0 | 243 | 243 | 1.2 | 0 | 0 |  |
|  |  | B3_d3 cubic g | 0 | 247 | 247 | 1.2 | 7 | 142 |  |
|  |  | B4 GP discrepancy | 0 | 255 | 255 | 1.2 | 7 | 180 |  |

- Pre-refit calls: B1 edge screen; selected g v2 nested lambda selection inner refits for the LODO evaluation (NIST: stored n_sim; other datasets: inner refits x median held-out n_sim, estimate). S3_D gate: 0 calls (one v1 training per fold, 500 steps). Penalty grids are one DC deck per device and are not converted to calls. M_bias v1 training CPU seconds are not stored per member.


### Part C sky130 77 K (18 devices): best-so-far paper RRMS at the matched NGSpice-call budget B_d (= LS n_sim per device), geometric mean over 3 seeds

| method | mean B_d (calls) | GMR vs starting card [CI] | GMR vs LS [CI], devices lower | label | per-seed GMR vs LS | mean run time (s) |
|---|---:|---:|---:|---|---|---:|
| LS (TRF refit, max_nfev 400) | 315 | 0.868 [0.737, 0.976] | 1 | reference | - | 19.3 |
| CMA-ES sigma0 0.1 | 315 (runs to 4 B_d) | 0.909 [0.821, 0.986] | 1.048 [1.000, 1.117] 1/18 | LS better (primary) | 1: 1.057, 2: 1.039, 3: 1.048 | 113.6 |
| CMA-ES sigma0 0.03 | 315 (runs to 4 B_d) | 0.886 [0.776, 0.980] | 1.021 [0.999, 1.055] 1/18 | inconclusive (secondary config) | 1: 1.025, 2: 1.023, 3: 1.015 | 118.5 |
| TuRBO-1 L_init 0.1 | 315 | 0.869 [0.744, 0.975] | 1.002 [0.994, 1.011] 10/18 | inconclusive (primary) | 1: 1.004, 2: 0.998, 3: 1.004 | 1004.0 |
| TuRBO-1 L_init 0.03 | 315 | 0.874 [0.750, 0.978] | 1.007 [0.997, 1.019] 4/18 | inconclusive (secondary config) | 1: 1.004, 2: 1.007, 3: 1.010 | 436.6 |

- Validity: VC1 True (max |LS - stored C0_refit| 0.0e+00), VC2 True, VC3 True.
- CMA-ES sigma0 0.1 at 4 B_d vs LS max_nfev 1600: 1.040 [1.003, 1.094]
- CMA-ES sigma0 0.03 at 4 B_d vs LS max_nfev 1600: 1.019 [1.001, 1.052]


### Part A B1 screen selections (state C) and B2 fitted elements

| dataset | B1 extras selected | B1 names | B1 median active parameters | B1 median max_nfev (matched) | B2 median R_ext / upper bound | B2 median kappa |
|---|---:|---|---:|---:|---:|---:|
| sky130 77 K (18, LODO) | 19 of 72 | a1, alpha1, cdsc, dvt0w, dvtp0, etab, fprout, minv, pdiblc1, pdits, phin, pscbe1, pscbe2, pvag, rbpb, rdswmin, ud, up, voffl | 57 | 590 | 0.00 | +0.000 |
| NIST sky130 4 K lvt (53) | 18 of 72 | a1, alpha1, cdsc, dvt0w, dvtp0, fprout, keta, minv, pdiblc1, pdits, phin, pscbe1, pvag, rbpb, rdswmin, ud, up, voffl | 56 | 570 | 0.00 | -0.001 |
| NIST sky130 4 K RT card (87) | 19 of 72 | a1, alpha1, cdsc, dvt0w, dvtp0, etab, fprout, keta, minv, pdiblc1, pdits, phin, pscbe1, pvag, rbpb, rdswmin, ud, up, voffl | 54 | 560 | 0.00 | -0.000 |
| IHP SG13G2 233 K | lv: 75; hv: 76 | stored I5-1 per-class screen | 84 | 2394 | 0.00 | -0.001 |
| IHP SG13G2 300 K | lv: 75; hv: 76 | stored I5-1 per-class screen | 84 | 2394 | 0.00 | -0.001 |
| IHP SG13G2 343 K | lv: 75; hv: 76 | stored I5-1 per-class screen | 84 | 2394 | 0.00 | -0.000 |
| IHP SG13G2 398 K | lv: 75; hv: 76 | stored I5-1 per-class screen | 84 | 2394 | 0.00 | -0.000 |
| TU Delft 4.13 K | 17 of 54 | a1, cdsc, dvtp0, dvtp1, fprout, minv, pdiblc1, pdits, pditsd, pditsl, phin, pscbe1, pscbe2, pvag, rbpb, rdswmin, voffl | 53 | 550 | 0.00 | -0.000 |
| TU Delft 295.0 K | 16 of 54 | a1, cdsc, dvtp0, dvtp1, fprout, minv, pdiblc1, pdits, pditsd, pditsl, phin, pscbe1, pscbe2, pvag, rdswmin, voffl | 52 | 540 | 0.01 | +0.000 |

