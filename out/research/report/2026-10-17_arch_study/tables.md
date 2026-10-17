# Tables, 2026-10-17 arch_study

Generated 2026-10-17 07:37 by `scripts/research/report_figs.py --topic arch_study --date 2026-10-17` from result files only. RRMS is the paper metric with frozen published-baseline curve inclusion, re-simulated in NGSpice.

### Surrogate architecture study, 43 parameters, round-1 box: training, held-out accuracy, ranking of reference cards and end-to-end search on the 18 measured devices

| surrogate in the search | trainable parameters | epochs | train time (s) | test loss | curve NRMSE, median | curve NRMSE, mean | mean absolute score error at the reference cards | devices where the direct fit ranks better: surrogate | devices where the direct fit ranks better: NGSpice | end-to-end all-device | end-to-end combined | end-to-end nMOS | end-to-end pMOS | failed devices (strict) | devices where a bank start is selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| paper MLP | 5,294,078 | 190 | 787 | 0.0331 | 0.1068 | 3,409 | 0.2879 | 6 | 18 | 0.2348 | 0.2248 | 0.1345 | 0.3151 | 0 | 11 |
| wide MLP | 25,270,270 | 200 | 3,337 | 0.0324 | 0.1003 | 11,144 | 0.3173 | 6 | 18 | 0.2463 | 0.2361 | 0.1441 | 0.3281 | 0 | 13 |
| ResMLP | 27,343,870 | 131 | 2,442 | 0.0291 | 0.1042 | 6,420 | 0.2936 | 8 | 18 | 0.2308 | 0.2211 | 0.1334 | 0.3087 | 0 | 8 |
| large ResMLP | 138,592,254 | 105 | 9,180 | 0.0323 | 0.1283 | 3,629 | 0.3429 | 5 | 18 | 0.2366 | 0.2267 | 0.1376 | 0.3158 | 0 | 9 |
| DeepONet | 25,775,358 | 165 | 4,025 | 0.0326 | 0.0911 | 151,287 | 0.2516 | 6 | 18 | 0.2274 | 0.2193 | 0.1465 | 0.2921 | 0 | 9 |
| FNO | 4,056,833 | 80 | 5,791 | 0.0259 | 0.1006 | 282,767 | 0.2427 | 6 | 18 | 0.2549 | 0.2448 | 0.1541 | 0.3355 | 0 | 16 |
| Reference: NGSpice in place of the surrogate | — | — | — | — | — | — | — | — | — | 0.1125 | 0.1053 | 0.0410 | 0.1696 | 0 | 0 |
| Reference: pipeline r1, stored | — | 192 | 927 | 0.0337 | — | — | — | 1 | 18 | 0.2375 | 0.2281 | 0.1435 | 0.3128 | 1 | 10 |
| Reference: pipeline r1, corrected | — | 192 | 927 | 0.0337 | — | — | — | 2 | 18 | 0.2382 | 0.2289 | 0.1450 | 0.3128 | 0 | 11 |

- Training: every architecture uses the same cache (out/cold_loop/full/round_1/cache), the same per-device train/validation/test split as the pipeline (seed 20260818; 85,853 training samples), the same loss (standardized signed-log MSE + curve-normalized linear Huber) and the same recipe (AdamW, cosine schedule, early stopping on validation loss, at most 200 epochs; FNO at most 80). time (s) = training wall time.
- loss = the pipeline's forward loss on the held-out bank samples. curve NRMSE = per held-out sample, mean over the 11 curves of RMSE / max|I| of the simulated curve (floor 1e-3 of the sample's largest curve); an emulator diagnostic, not the paper RRMS. The mean curve NRMSE is 2.8e+04 to 2.8e+06 times the median.
- |score err.| = mean over 18 devices and 3 reference cards (stored pipeline r1 pick, direct NGSpice cold, direct NGSpice seeded) of |surrogate-predicted RRMS − NGSpice RRMS|. direct fit ranked better = devices where the direct NGSpice cold card (out/research/ngspice_direct/full_r1_direct_cold) has a lower RRMS than the stored pipeline r1 pick, by the surrogate's prediction and by NGSpice. The stored pipeline r1 picks were made before the failed-curve fix; the pick for nmos_L0p15_W1p6 has 1 of 10 included curves failed, and its NGSpice reference value 0.2035 averages the surviving curves.
- End-to-end = the pipeline's search (fps.finite_difference_surrogate_extract) with the trained surrogate frozen: 8 starts, TRF max_nfev 100, diff_step 0.02, regularization 0.0001, NGSpice validation of the top 4 endpoints, the first 4 starts and the published fallback; RRMS of the selected card re-simulated in NGSpice on the measured 77 K data. start picked = devices whose selected card is one of the bank starts (selected_candidate_kind lhc_start) rather than an FD endpoint. all-device = mean over 18 devices; combined = mean of the nMOS (8) and pMOS (10) family means.
- Reference rows: NGSpice in place of the surrogate = the same search code with NGSpice as the evaluator (out/research/swap_search/full_round_1). pipeline r1, stored = the cold-loop round-1 extraction (out/cold_loop/full/round_1/extract). pipeline r1, corrected = its final selection rerun with the same surrogate and failed-curve rejection (out/research/pipe_fixed/full_r1). The pipeline rows' training columns are the cold-loop round-1 forward model (same split and seed, 10,730 test samples); its ranking counts come from out/research/surrogate_gap_full_round_1.json and the pipe_fixed records.
- End-to-end records of the architecture study store rrms without per_curve. The run started 2026-10-16 22:26 (out/research/done.txt), after commit a0e1e97 (21:52), with which metrics.device_rrms_new returns NaN for a device whose frozen-inclusion curve is non-finite or missing and the pipeline's NGSpice validation sorts such candidates last. A finite stored rrms is therefore counted as a valid device and a non-finite one as failed.
- Reference: pipeline r1, stored (method output): 1 failed device(s): nmos_L0p15_W1p6 (1 included curve(s) non-finite); valid-only all-device 0.2395 over 17 devices, combined 0.2238; penalized all-device 0.2920.
- The end-to-end search in this study ran on the 18 measured devices; it was queued before the DEV/TEST protocol was fixed. It is reported as a diagnostic ablation and is not used for method selection.

### Corrected pipeline 43 cards, measured 77 K data: raw and after NGSpice FD polish

| cards | stage | all-device | combined | nMOS | pMOS | all-device, FD − raw | failed devices | NGSpice calls/device (this run) | NGSpice calls/device (cumulative) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Corrected pipeline 43, round 1 | method output | 0.2382 | 0.2289 | 0.1450 | 0.3128 | — | 0 | 10,009 | 10,009 |
| Corrected pipeline 43, round 1 | + NGSpice FD | 0.1800 | 0.1703 | 0.0826 | 0.2579 | -0.0582 | 0 | 350 | 10,359 |
| Corrected pipeline 43, round 4 | method output | 0.1522 | 0.1437 | 0.0673 | 0.2201 | — | 0 | 10,009 | 40,035 |
| Corrected pipeline 43, round 4 | + NGSpice FD | 0.1477 | 0.1394 | 0.0645 | 0.2143 | -0.0045 | 0 | 381 | 40,416 |

- all-device = mean over 18 devices; combined = mean of the nMOS (8) and pMOS (10) family means.
- Corrected pipeline = the final selection rerun with the round's surrogate and the rule that rejects candidates with failed included curves (out/research/pipe_fixed/full_r1, full_r4). Rounds 2–3 were not rerun; the round-4 box comes from the stored round-3 pick.
- + NGSpice FD = scripts/research/polish_cards.py (max_nfev 400, diff_step 0.01, irls 2) on those cards against the measured curves.
- NGSpice calls per device: raw 'this run' = the round's bank + the rerun's validation; 'cumulative' adds the banks and stored validations of the earlier rounds; + FD 'this run' = FD polish only. failed dev. = devices with a failed included curve (cryoml.strict_score).
