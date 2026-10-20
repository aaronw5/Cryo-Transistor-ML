# Tables, 2026-10-20 i7_2

Generated 2026-10-20 19:41 by `PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i7_2.py --date 2026-10-20` from stored files only. RRMS is the strict paper metric on the held-out curves of each split (frozen inclusion). I7-2 values are on DEV devices (development evidence); REAL_TEST and TEST are not loaded.

### I7-2 decision: pre-registered conditions C1-C4 per dataset, counts and the outcome list

| item | statistic | required | sky130: GMR [device 95% CI] wins/paired | IHP 300 K: GMR [device 95% CI] wins/paired | TU Delft 4.13 K: GMR [device 95% CI] wins/paired | TU Delft 295 K: GMR [device 95% CI] wins/paired | count | evaluated |
|---|---|---|---:|---:|---:|---:|---:|---|
| C1 | GMR(c_hier, b_shared) | CI upper < 1 on >= 3 of 4 | 0.992 [0.929, 1.066] 6/11 | 0.904 [0.803, 1.024] 13/18 | 0.912 [0.863, 0.958] 37/45 * | 0.939 [0.904, 0.973] 32/45 * | 2/4 | fails |
| C2 | GMR(c_hier, a_C0) | CI upper < 1 on 4 of 4 | 0.766 [0.636, 0.913] 8/11 * | 1.025 [0.848, 1.264] 12/18 | 0.818 [0.773, 0.863] 39/45 * | 0.758 [0.703, 0.819] 38/45 * | 3/4 | fails |
| C3 | GMR(c_hier, d_scratch) | point < 1 on 4 of 4 and CI upper < 1 on >= 3 | 0.912 [0.840, 0.994] 7/10 * | 1.253 [1.038, 1.535] 6/18 | 1.153 [1.008, 1.314] 22/45 | 1.121 [0.992, 1.268] 21/45 | point 1/4, CI 1/4 | fails |
| C4 | c_hier failed <= 10% and >= 8 paired | every dataset | failed 1/12, paired 10 | failed 0/18, paired 18 | failed 0/45, paired 45 | failed 0/45, paired 45 | 4/4 | holds |
| stop check | GMR(c_hier, b_shared) point >= 0.98 | >= 2 datasets for the stop outcome | 0.992 (>= 0.98) | 0.904 | 0.912 | 0.939 | 1/4 | not reached |
| outcome 1 | invalid | C4 fails on 2 or more datasets |  |  |  |  |  | fails |
| outcome 2 | supported_continue | C1, C2, C3 and C4 hold |  |  |  |  |  | fails |
| outcome 3 | delta_no_gain_stop | GMR(c_hier, b_shared) point estimate >= 0.98 on 2 or more datasets |  |  |  |  |  | fails |
| outcome 4 | sharing_no_gain | C1 and C2 hold and C3 fails |  |  |  |  |  | fails |
| outcome 5 | inconclusive | any other combination |  |  |  |  |  | holds; rule output |

- Pre-registration out/research/explore/I7_2/prereg.json (commit be7d69e, before any I7-2 code was run on data); code and smoke commit 7c9aeec; queue jobs I7_2_sky130, I7_2_ihp300, I7_2_tud4, I7_2_tud295, I7_2_decide (2026-10-20).
- Per-device value = geometric mean over the device's defined draws (K2a, K2b, K4a, K4b) of the strict paper RRMS on the held-out curves; NaN if any of those draws failed for the arm. GMR(X, Y) = exp(mean ln(v_X / v_Y)) over devices with both values finite; 95% percentile bootstrap over devices (10000 resamples, default_rng(0)); wins = devices with v_X < v_Y.
- Outcomes are evaluated in the pre-registered order; the first one that holds is the rule output. C1, C2 and C3 fail and C4 holds, no stop condition is met: rule output 'inconclusive'.
- sky130: pmos_L0p35_W0p55 failed in every arm (draws K2b and K4b give non-finite held-out RRMS), so 11 devices are paired for c_vs_b and c_vs_a; d_scratch and e_crossinterp also failed on pmos_L0p5_W0p42 (draw K4b), so 10 are paired for c_vs_d.
- Recomputed here from the per-split rows and checked against out/research/explore/I7_2/decision.json and the four results.json files.


### I7-2 held-out RRMS per arm and dataset: all-device mean, common-set mean, combined, median, GMR against a_C0, adaptation-curve RRMS and simulations per split

| dataset | arm | definition | valid devices | all-device mean (valid devices) | mean over devices valid in every arm | devices in the common set | combined (nMOS / pMOS family means) | median | GMR(X, a_C0) [device 95% CI] wins/paired | mean adaptation-curve RRMS over splits | mean NGSpice calls per split |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sky130 | a_C0 | card refit on the K curves, no correction | 11/12 | 0.5513 | 0.5151 | 10 | 0.5170 | 0.4678 | 1 (reference) | 0.1475 | 243 |
| sky130 | b_shared | card refit with the stored fold correction g_shared in the loop | 11/12 | 0.3905 | 0.3293 | 10 | 0.3687 | 0.3623 | 0.772 [0.664, 0.889] 10/11 | 0.1302 | 270 |
| sky130 | c_hier | card + delta_d fitted jointly, MAP with the learned prior and selected mu (primary) | 11/12 | 0.3759 | 0.3136 | 10 | 0.3558 | 0.3661 | 0.766 [0.636, 0.913] 8/11 | 0.1307 | 268 |
| sky130 | c_ml | as c_hier with mu = 0 (flat prior on the z box) | 11/12 | 0.3264 | 0.2721 | 10 | 0.3102 | 0.2953 | 0.680 [0.540, 0.843] 10/11 | 0.1205 | 251 |
| sky130 | d_scratch | card refit + per-device M_bias-form MLP from scratch on the K curves | 10/12 | 0.3639 | 0.3639 | 10 | 0.3639 | 0.3038 | 0.811 [0.704, 0.932] 9/10 | 0.0690 | 232 |
| sky130 | e_crossinterp | a's card + measured-minus-simulated residual at curve crossings (non-learned) | 10/12 | 0.3690 | 0.3690 | 10 | 0.3690 | 0.1874 | 0.713 [0.593, 0.813] 10/10 | 0.1475 | 0 |
| IHP 300 K | a_C0 | card refit on the K curves, no correction | 18/18 | 0.0676 | 0.0676 | 18 | 0.0676 | 0.0472 | 1 (reference) | 0.0128 | 322 |
| IHP 300 K | b_shared | card refit with the stored fold correction g_shared in the loop | 18/18 | 0.0625 | 0.0625 | 18 | 0.0625 | 0.0453 | 1.134 [0.938, 1.426] 11/18 | 0.0134 | 340 |
| IHP 300 K | c_hier | card + delta_d fitted jointly, MAP with the learned prior and selected mu (primary) | 18/18 | 0.0528 | 0.0528 | 18 | 0.0528 | 0.0408 | 1.025 [0.848, 1.264] 12/18 | 0.0116 | 305 |
| IHP 300 K | c_ml | as c_hier with mu = 0 (flat prior on the z box) | 18/18 | 0.0492 | 0.0492 | 18 | 0.0492 | 0.0395 | 1.102 [0.874, 1.407] 11/18 | 0.0110 | 304 |
| IHP 300 K | d_scratch | card refit + per-device M_bias-form MLP from scratch on the K curves | 18/18 | 0.0535 | 0.0535 | 18 | 0.0535 | 0.0347 | 0.818 [0.730, 0.909] 16/18 | 0.0026 | 328 |
| IHP 300 K | e_crossinterp | a's card + measured-minus-simulated residual at curve crossings (non-learned) | 18/18 | 0.0645 | 0.0645 | 18 | 0.0645 | 0.0457 | 0.930 [0.872, 0.978] 13/18 | 0.0128 | 0 |
| TU Delft 4.13 K | a_C0 | card refit on the K curves, no correction | 45/45 | 0.1254 | 0.1254 | 45 | 0.1228 | 0.1218 | 1 (reference) | 0.0462 | 220 |
| TU Delft 4.13 K | b_shared | card refit with the stored fold correction g_shared in the loop | 45/45 | 0.1139 | 0.1139 | 45 | 0.1141 | 0.1108 | 0.897 [0.854, 0.941] 31/45 | 0.0412 | 304 |
| TU Delft 4.13 K | c_hier | card + delta_d fitted jointly, MAP with the learned prior and selected mu (primary) | 45/45 | 0.1049 | 0.1049 | 45 | 0.1051 | 0.1013 | 0.818 [0.773, 0.863] 39/45 | 0.0359 | 291 |
| TU Delft 4.13 K | c_ml | as c_hier with mu = 0 (flat prior on the z box) | 45/45 | 0.1047 | 0.1047 | 45 | 0.1028 | 0.0977 | 0.813 [0.774, 0.855] 40/45 | 0.0336 | 286 |
| TU Delft 4.13 K | d_scratch | card refit + per-device M_bias-form MLP from scratch on the K curves | 45/45 | 0.0955 | 0.0955 | 45 | 0.0870 | 0.0720 | 0.709 [0.642, 0.783] 36/45 | 0.0096 | 195 |
| TU Delft 4.13 K | e_crossinterp | a's card + measured-minus-simulated residual at curve crossings (non-learned) | 45/45 | 0.0739 | 0.0739 | 45 | 0.0733 | 0.0702 | 0.585 [0.553, 0.621] 45/45 | 0.0462 | 0 |
| TU Delft 295 K | a_C0 | card refit on the K curves, no correction | 45/45 | 0.0725 | 0.0725 | 45 | 0.0689 | 0.0661 | 1 (reference) | 0.0218 | 220 |
| TU Delft 295 K | b_shared | card refit with the stored fold correction g_shared in the loop | 45/45 | 0.0600 | 0.0600 | 45 | 0.0582 | 0.0529 | 0.807 [0.750, 0.866] 37/45 | 0.0196 | 307 |
| TU Delft 295 K | c_hier | card + delta_d fitted jointly, MAP with the learned prior and selected mu (primary) | 45/45 | 0.0566 | 0.0566 | 45 | 0.0550 | 0.0511 | 0.758 [0.703, 0.819] 38/45 | 0.0178 | 282 |
| TU Delft 295 K | c_ml | as c_hier with mu = 0 (flat prior on the z box) | 45/45 | 0.0523 | 0.0523 | 45 | 0.0513 | 0.0476 | 0.712 [0.658, 0.769] 42/45 | 0.0158 | 285 |
| TU Delft 295 K | d_scratch | card refit + per-device M_bias-form MLP from scratch on the K curves | 45/45 | 0.0504 | 0.0504 | 45 | 0.0463 | 0.0513 | 0.676 [0.625, 0.729] 43/45 | 0.0027 | 201 |
| TU Delft 295 K | e_crossinterp | a's card + measured-minus-simulated residual at curve crossings (non-learned) | 45/45 | 0.0482 | 0.0482 | 45 | 0.0467 | 0.0460 | 0.673 [0.646, 0.703] 45/45 | 0.0218 | 0 |

- all-device = mean of the per-device values (geometric mean over the device's draws) over the devices with a finite value; for IHP and TU Delft no device failed, so it equals the all-device mean. common set = devices finite in all six arms. combined = mean of the nMOS and pMOS family means (IHP has 9 nMOS and 9 pMOS DUTs, so its combined value equals the all-device mean; sky130 has 5 nMOS and 7 pMOS, TU Delft 29 nMOS and 16 pMOS).
- Every scored current is a fresh NGSpice simulation of the arm's final card on all curves of the device, multiplied by exp(g) of the arm's correction in Python (b: g_shared; c, c_ml: g_shared + phi . delta; d: the from-scratch MLP; e: a's card plus the cross-curve residual). sky130 and TU Delft: ngspice-41; IHP: ngspice-46 (as I5-1).
- adapt RRMS = strict paper RRMS on the K adaptation curves (the fitted curves), mean over the splits with a finite value; e_crossinterp equals a_C0 on the adaptation curves. sims = NGSpice calls of the arm's fit (c_hier and c_ml reuse the cached card simulation for the z columns of the Jacobian; e needs none).
- Queue wall time: sky130 540 s (48 splits), IHP 610 s (66 splits; 6 K4 draws undefined), TU Delft 4.13 K 1266 s (180 splits; 2 simulator timeouts), 295 K 1234 s (180 splits); 8 workers.


### I7-2 breakdown by adaptation draw and K: GMR(c, b), GMR(c, a), GMR(c, d), GMR(c, e), GMR(d, a), GMR(e, a) per dataset

| dataset | draw or K | devices (max paired) | GMR(c_hier, b_shared) [device 95% CI] wins/paired | GMR(c_hier, a_C0) [device 95% CI] wins/paired | GMR(c_hier, d_scratch) [device 95% CI] wins/paired | GMR(c_hier, e_crossinterp) [device 95% CI] wins/paired | GMR(d_scratch, a_C0) [device 95% CI] wins/paired | GMR(e_crossinterp, a_C0) [device 95% CI] wins/paired | arms ordered by GMR(X, a_C0), lowest first |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| sky130 | K2a | 12 | 0.975 [0.765, 1.210] 7/12 | 0.933 [0.726, 1.196] 7/12 | 0.967 [0.764, 1.223] 6/12 | 1.050 [0.723, 1.481] 5/12 | 0.965 [0.862, 1.098] 7/12 | 0.888 [0.707, 1.092] 9/12 | c_ml < e < c < b < d < a |
| sky130 | K2b | 11 | 0.990 [0.902, 1.076] 4/11 | 0.779 [0.643, 0.903] 11/11 | 0.839 [0.687, 1.018] 8/11 | 0.799 [0.664, 0.921] 10/11 | 0.929 [0.736, 1.175] 5/11 | 0.976 [0.963, 0.988] 8/11 | c < b < c_ml < d < e < a |
| sky130 | K = 2 | 11 | 0.982 [0.867, 1.099] 4/11 | 0.858 [0.733, 1.000] 8/11 | 0.897 [0.796, 1.004] 7/11 | 0.911 [0.737, 1.109] 6/11 | 0.957 [0.827, 1.098] 4/11 | 0.942 [0.834, 1.047] 8/11 | c_ml < c < b < e < d < a |
| sky130 | K4a | 12 | 1.006 [0.882, 1.123] 6/12 | 0.759 [0.585, 0.958] 10/12 | 1.291 [0.916, 1.801] 2/11 | 1.325 [0.992, 1.762] 4/11 | 0.573 [0.396, 0.832] 9/11 | 0.558 [0.446, 0.702] 10/11 | e < d < c_ml < b < c < a |
| sky130 | K4b | 11 | 1.000 [0.950, 1.047] 4/11 | 0.632 [0.487, 0.807] 11/11 | 0.787 [0.572, 1.075] 6/10 | 1.176 [0.805, 1.774] 4/10 | 0.771 [0.662, 0.873] 9/10 | 0.516 [0.382, 0.660] 10/10 | e < c_ml < b < c < d < a |
| sky130 | K = 4 | 11 | 1.002 [0.942, 1.066] 5/11 | 0.684 [0.541, 0.847] 10/11 | 0.959 [0.829, 1.118] 5/10 | 1.224 [0.921, 1.687] 2/10 | 0.691 [0.592, 0.812] 9/10 | 0.541 [0.417, 0.673] 10/10 | e < c_ml < b < c < d < a |
| sky130 | all | 11 | 0.992 [0.929, 1.066] 6/11 | 0.766 [0.636, 0.913] 8/11 | 0.912 [0.840, 0.994] 7/10 | 1.036 [0.831, 1.314] 5/10 | 0.811 [0.704, 0.932] 9/10 | 0.713 [0.593, 0.813] 10/10 | c_ml < e < c < b < d < a |
| IHP 300 K | K2a | 18 | 0.871 [0.749, 1.007] 14/18 | 1.009 [0.847, 1.270] 13/18 | 1.140 [0.893, 1.517] 12/18 | 1.045 [0.875, 1.318] 11/18 | 0.885 [0.731, 1.057] 12/18 | 0.965 [0.916, 1.014] 11/18 | d < e < a < c < c_ml < b |
| IHP 300 K | K2b | 18 | 1.033 [0.933, 1.147] 7/18 | 1.114 [0.865, 1.468] 9/18 | 1.156 [0.861, 1.580] 8/18 | 1.166 [0.921, 1.514] 10/18 | 0.963 [0.812, 1.148] 10/18 | 0.955 [0.889, 1.011] 9/18 | e < d < a < b < c < c_ml |
| IHP 300 K | K = 2 | 18 | 0.949 [0.846, 1.062] 13/18 | 1.060 [0.876, 1.337] 12/18 | 1.148 [0.953, 1.430] 7/18 | 1.104 [0.917, 1.379] 11/18 | 0.923 [0.847, 1.016] 14/18 | 0.960 [0.916, 1.003] 10/18 | d < e < a < c < c_ml < b |
| IHP 300 K | K4a | 15 | 0.727 [0.595, 0.874] 12/15 | 0.880 [0.650, 1.204] 9/15 | 1.214 [0.895, 1.643] 6/15 | 0.965 [0.740, 1.252] 9/15 | 0.725 [0.594, 0.873] 13/15 | 0.912 [0.800, 1.000] 11/15 | d < c_ml < c < e < a < b |
| IHP 300 K | K4b | 15 | 0.881 [0.768, 0.981] 12/15 | 1.023 [0.850, 1.259] 9/15 | 1.597 [1.223, 2.119] 2/15 | 1.163 [1.007, 1.378] 5/15 | 0.641 [0.510, 0.778] 14/15 | 0.880 [0.806, 0.948] 14/15 | d < e < a < c < c_ml < b |
| IHP 300 K | K = 4 | 15 | 0.800 [0.683, 0.916] 11/15 | 0.949 [0.754, 1.219] 9/15 | 1.392 [1.070, 1.825] 4/15 | 1.059 [0.871, 1.303] 8/15 | 0.682 [0.562, 0.800] 15/15 | 0.896 [0.805, 0.968] 13/15 | d < e < c_ml < c < a < b |
| IHP 300 K | all | 18 | 0.904 [0.803, 1.024] 13/18 | 1.025 [0.848, 1.264] 12/18 | 1.253 [1.038, 1.535] 6/18 | 1.102 [0.931, 1.332] 9/18 | 0.818 [0.730, 0.909] 16/18 | 0.930 [0.872, 0.978] 13/18 | d < e < a < c < c_ml < b |
| TU Delft 4.13 K | K2a | 45 | 0.907 [0.851, 0.968] 32/45 | 0.920 [0.856, 0.989] 29/45 | 0.818 [0.721, 0.921] 31/45 | 1.197 [1.071, 1.347] 16/45 | 1.124 [1.039, 1.224] 15/45 | 0.768 [0.717, 0.822] 38/45 | e < c < c_ml < a < b < d |
| TU Delft 4.13 K | K2b | 45 | 0.979 [0.911, 1.053] 27/45 | 0.867 [0.826, 0.915] 38/45 | 0.885 [0.793, 0.987] 27/45 | 1.046 [0.964, 1.142] 27/45 | 0.980 [0.899, 1.075] 23/45 | 0.829 [0.775, 0.882] 45/45 | e < c < b < c_ml < d < a |
| TU Delft 4.13 K | K = 2 | 45 | 0.943 [0.901, 0.988] 33/45 | 0.893 [0.851, 0.940] 34/45 | 0.851 [0.768, 0.936] 34/45 | 1.119 [1.028, 1.225] 25/45 | 1.050 [0.979, 1.130] 20/45 | 0.798 [0.753, 0.846] 43/45 | e < c < c_ml < b < a < d |
| TU Delft 4.13 K | K4a | 45 | 0.900 [0.834, 0.963] 31/45 | 0.737 [0.659, 0.829] 41/45 | 1.357 [1.079, 1.728] 23/45 | 1.828 [1.557, 2.169] 3/45 | 0.543 [0.451, 0.645] 43/45 | 0.403 [0.361, 0.450] 45/45 | e < d < c_ml < c < b < a |
| TU Delft 4.13 K | K4b | 45 | 0.864 [0.740, 0.994] 35/45 | 0.760 [0.654, 0.858] 33/45 | 1.800 [1.457, 2.236] 14/45 | 1.665 [1.439, 1.880] 5/45 | 0.422 [0.348, 0.508] 45/45 | 0.457 [0.420, 0.499] 45/45 | d < e < c_ml < c < b < a |
| TU Delft 4.13 K | K = 4 | 45 | 0.882 [0.811, 0.953] 36/45 | 0.748 [0.679, 0.820] 39/45 | 1.562 [1.288, 1.891] 18/45 | 1.745 [1.559, 1.949] 5/45 | 0.479 [0.411, 0.557] 45/45 | 0.429 [0.398, 0.465] 45/45 | e < d < c_ml < c < b < a |
| TU Delft 4.13 K | all | 45 | 0.912 [0.863, 0.958] 37/45 | 0.818 [0.773, 0.863] 39/45 | 1.153 [1.008, 1.314] 22/45 | 1.397 [1.287, 1.518] 3/45 | 0.709 [0.642, 0.783] 36/45 | 0.585 [0.553, 0.621] 45/45 | e < d < c_ml < c < b < a |
| TU Delft 295 K | K2a | 45 | 0.977 [0.922, 1.034] 23/45 | 0.770 [0.677, 0.876] 34/45 | 0.854 [0.741, 0.976] 27/45 | 0.840 [0.724, 0.975] 30/45 | 0.902 [0.825, 0.976] 26/45 | 0.917 [0.875, 0.955] 32/45 | c_ml < c < b < d < e < a |
| TU Delft 295 K | K2b | 45 | 0.996 [0.924, 1.071] 22/45 | 0.988 [0.872, 1.116] 25/45 | 1.133 [0.933, 1.386] 22/45 | 1.060 [0.931, 1.206] 24/45 | 0.872 [0.758, 1.002] 29/45 | 0.932 [0.908, 0.954] 38/45 | c_ml < d < e < c < b < a |
| TU Delft 295 K | K = 2 | 45 | 0.986 [0.944, 1.029] 25/45 | 0.872 [0.785, 0.966] 29/45 | 0.984 [0.851, 1.136] 24/45 | 0.944 [0.839, 1.061] 26/45 | 0.887 [0.809, 0.964] 28/45 | 0.924 [0.895, 0.951] 40/45 | c_ml < c < b < d < e < a |
| TU Delft 295 K | K4a | 45 | 0.942 [0.854, 1.033] 26/45 | 0.694 [0.616, 0.783] 40/45 | 1.568 [1.235, 1.978] 13/45 | 1.212 [1.081, 1.355] 11/45 | 0.443 [0.369, 0.530] 43/45 | 0.573 [0.527, 0.620] 45/45 | d < e < c_ml < c < b < a |
| TU Delft 295 K | K4b | 45 | 0.849 [0.789, 0.912] 34/45 | 0.625 [0.551, 0.713] 38/45 | 1.041 [0.910, 1.198] 24/45 | 1.486 [1.289, 1.703] 8/45 | 0.600 [0.544, 0.656] 45/45 | 0.420 [0.377, 0.471] 45/45 | e < c_ml < d < c < b < a |
| TU Delft 295 K | K = 4 | 45 | 0.894 [0.834, 0.955] 34/45 | 0.659 [0.594, 0.732] 40/45 | 1.278 [1.092, 1.489] 16/45 | 1.342 [1.206, 1.486] 11/45 | 0.515 [0.464, 0.574] 44/45 | 0.491 [0.457, 0.527] 45/45 | e < d < c_ml < c < b < a |
| TU Delft 295 K | all | 45 | 0.939 [0.904, 0.973] 32/45 | 0.758 [0.703, 0.819] 38/45 | 1.121 [0.992, 1.268] 21/45 | 1.125 [1.033, 1.227] 17/45 | 0.676 [0.625, 0.729] 43/45 | 0.673 [0.646, 0.703] 45/45 | e < d < c_ml < c < b < a |

- Rows K2a, K2b, K4a, K4b: per-device value = held-out RRMS of that draw alone. Rows 'K = 2', 'K = 4', 'all': geometric mean over the device's draws of that K (or of all draws), the pre-registered per-K tables; these rows are checked against the stored summaries K2, K4 and all.
- IHP: draws K4a and K4b are undefined for 3 of 18 DUTs (fewer than 2 held-out curves would remain), so K4 rows pair 15 DUTs. sky130: pmos_L0p35_W0p55 fails in K2b and K4b (all arms) and pmos_L0p5_W0p42 fails in K4b for d and e.
- Order column: arms a, b, c, c_ml, d, e sorted by GMR(X, a_C0) (a = 1); the pairs can use different device sets when an arm failed on a device (sky130 only).


### I7-2 learned prior: selected mu per fold and K, prior mean and sd, and the size of the per-device term phi . delta of c_hier and c_ml on the held-out points

| dataset | K | folds | selected mu (count over folds) | splits with mu = inf (delta = m) | median RMS of g_shared over the held-out points | RMS of phi . delta, c_hier: median [min, max] | RMS of phi . delta, c_ml: median [min, max] | c_ml splits with a z coordinate at +/-5 | median |z|^2 c_hier / c_ml |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| sky130 | K = 2 | 12 | inf x12 | 24/24 | 0.0496 | 0.0152 [0.0094, 0.0272] | 0.0454 [0.0085, 0.3282] | 0/24 | 0.000 / 2.185 |
| sky130 | K = 4 | 12 | 0.03 x2, inf x10 | 20/24 | 0.0393 | 0.0148 [0.0093, 0.0282] | 0.0291 [0.0139, 0.1285] | 0/24 | 0.000 / 0.539 |
| sky130 | prior | 12 | m (fold median): +0.007, -0.019, +0.028, +0.028, -0.033, -0.019 |  |  | sd (fold median): 0.036, 0.127, 0.069, 0.079, 0.076, 0.093 | training devices per fold 11-11 |  |  |
| IHP 300 K | K = 2 | 18 | 0.001 x7, 0.003 x11 | 0/36 | 0.0074 | 0.0083 [0.0012, 0.0210] | 0.0149 [0.0011, 0.1332] | 0/36 | 0.056 / 2.493 |
| IHP 300 K | K = 4 | 18 | 1e-05 x2, 0.0001 x5, 0.0003 x11 | 0/30 | 0.0064 | 0.0067 [0.0018, 0.1142] | 0.0132 [0.0013, 0.1121] | 0/30 | 0.252 / 1.030 |
| IHP 300 K | prior | 18 | m (fold median): +0.007, -0.001, +0.004, -0.006, -0.004, -0.005 |  |  | sd (fold median): 0.037, 0.051, 0.019, 0.020, 0.021, 0.013 | training devices per fold 7-9 |  |  |
| TU Delft 4.13 K | K = 2 | 7 | 1 x4, inf x3 | 26/90 | 0.0695 | 0.0057 [0.0031, 0.0090] | 0.0182 [0.0032, 0.1362] | 0/90 | 0.000 / 0.399 |
| TU Delft 4.13 K | K = 4 | 7 | 0.001 x1, 0.003 x6 | 0/90 | 0.0699 | 0.0095 [0.0032, 0.0637] | 0.0164 [0.0027, 0.0949] | 0/90 | 0.035 / 0.289 |
| TU Delft 4.13 K | prior | 7 | m (fold median): -0.007, +0.008, +0.012, +0.003, -0.014, +0.001 |  |  | sd (fold median): 0.060, 0.181, 0.053, 0.133, 0.066, 0.018 | training devices per fold 37-43 |  |  |
| TU Delft 295 K | K = 2 | 7 | 0.01 x2, 0.03 x1, 0.1 x1, 0.3 x2, 1 x1 | 0/90 | 0.0208 | 0.0039 [0.0030, 0.0060] | 0.0099 [0.0030, 0.0435] | 0/90 | 0.000 / 0.552 |
| TU Delft 295 K | K = 4 | 7 | 0.001 x7 | 0/90 | 0.0207 | 0.0060 [0.0020, 0.0197] | 0.0124 [0.0023, 0.0462] | 0/90 | 0.101 / 1.040 |
| TU Delft 295 K | prior | 7 | m (fold median): -0.000, -0.007, +0.008, +0.010, -0.011, +0.003 |  |  | sd (fold median): 0.014, 0.053, 0.026, 0.073, 0.042, 0.019 | training devices per fold 37-43 |  |  |

- phi = (1, a, b, a^2, ab, b^2), a = 2|Vgs|/VDD - 1, b = 2|Vds|/VDD - 1. c_hier's correction is g_shared + phi . delta and b_shared's is g_shared, so phi . delta is the difference between the two corrections in ln I; its RMS is taken over the trimmed included points of the split's held-out curves (bias values from the measured grid). The cards of b and c differ as well (each arm refits its card).
- mu is selected per fold and K on the training devices with their cards fixed (prereg mu_selection_empirical_bayes); mu = inf sets z = 0, delta = m (the prior mean), the same delta for every device of the fold. Ties go to the larger mu.
- Prior per fold: m = mean of the training deltas, S = 0.5 Sigma_hat + 0.5 diag(Sigma_hat) + 1e-6 I; recomputed from the stored training deltas and checked. Coefficient order 1, a, b, a^2, ab, b^2.
- c_ml (mu = 0) z lies in the box [-5, 5]^6; 'at box' counts splits with at least one coordinate at a bound.

