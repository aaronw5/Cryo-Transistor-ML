# Tables, 2026-10-19 explore_e1_lc

Generated 2026-10-19 11:39 by `PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_e1_lc.py --date 2026-10-19` from result files only. RRMS is the paper metric with frozen published-baseline curve inclusion, strict (a non-finite included curve makes the device NaN).

### E1 on REAL_DEV, leave-one-device-out: stage 1 (post-hoc correction on the fixed card) vs stage 2 (card refit with the correction in the loop)

| stage | arm | all-device | combined | median | failed devices | M < C0 (devices) | M < T1 (devices) | pre-registered rule |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| stage 1 | C0 fixed card | 0.1073 | 0.0981 | 0.0823 | 0 | — | — | — |
| stage 1 | T1 per-tag gain | 0.0865 | 0.0802 | 0.0656 | 0 | — | — | — |
| stage 1 | M MLP correction | 0.1276 | 0.1237 | 0.0687 | 0 | 6/12 | 4/12 | fail |
| stage 2 | C0 refit | 0.1072 | 0.0981 | 0.0820 | 0 | — | — | — |
| stage 2 | T1 per-tag gain | 0.0820 | 0.0760 | 0.0650 | 0 | — | — | — |
| stage 2 | M MLP correction | 0.0871 | 0.0842 | 0.0612 | 0 | 8/12 | 6/12 | fail |

- Stage 1 (post-hoc): each arm is trained on 11 devices and applied once to the held-out device's fixed best 43-parameter card (direct NGSpice round 3 + FD). C0 = that card with no correction.
- Stage 2 (refit): the held-out card is re-polished (ngspice_fit.polish, TRF max_nfev 400, irls 2, diff_step 0.01, round-3 box) against the measured curves with the fold's correction inside the residual. C0 = the same polish without correction from the same start; T1 is re-polished with T1 in the loop.
- Pre-registered continue rule (all five): M all-device <= 0.090; M < T1 all-device; M < T1 on >= 8/12; M < C0 on >= 8/12; 0 failed devices for M. Stage 1: no, no, no, no, yes. Stage 2: yes, no, no, yes, yes.
- all-device = mean over the 12 REAL_DEV devices (5 nMOS, 7 pMOS); combined = mean of the nMOS and pMOS family means. T1 is a table indexed by curve tag and is not a function of terminal voltages, so it cannot be written as a Verilog-A device model; M can.


### E1 simulated benchmark, tier c sets 0-2, leave-one-geometry-out, post-hoc correction

| PDK | arm | all-device | combined | nMOS | pMOS | median | all-device / C0 | failed targets | targets with M < arm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GF180 (BSIM4) | C0 no correction | 0.0474 | 0.0474 | 0.0620 | 0.0328 | 0.0364 | 1.0000 | 0 | 19/48 |
| GF180 (BSIM4) | T1 per-tag gain (22 scalars) | 0.0420 | 0.0420 | 0.0546 | 0.0295 | 0.0360 | 0.8863 | 0 | 11/48 |
| GF180 (BSIM4) | T2 per-kind gain (4 scalars) | 0.0463 | 0.0463 | 0.0599 | 0.0328 | 0.0365 | 0.9771 | 0 | 17/48 |
| GF180 (BSIM4) | LIN quadratic-linear | 0.0443 | 0.0443 | 0.0546 | 0.0340 | 0.0392 | 0.9333 | 0 | 15/48 |
| GF180 (BSIM4) | M MLP correction (primary) | 0.0496 | 0.0496 | 0.0583 | 0.0409 | 0.0431 | 1.0451 | 0 | — |
| IHP SG13G2 (PSP103) | C0 no correction | 0.1012 | 0.1012 | 0.0879 | 0.1146 | 0.0568 | 1.0000 | 0 | 13/48 |
| IHP SG13G2 (PSP103) | T1 per-tag gain (22 scalars) | 0.0957 | 0.0957 | 0.0776 | 0.1138 | 0.0542 | 0.9456 | 0 | 10/48 |
| IHP SG13G2 (PSP103) | T2 per-kind gain (4 scalars) | 0.0999 | 0.0999 | 0.0846 | 0.1151 | 0.0573 | 0.9866 | 0 | 15/48 |
| IHP SG13G2 (PSP103) | LIN quadratic-linear | 0.0987 | 0.0987 | 0.0859 | 0.1116 | 0.0769 | 0.9754 | 0 | 7/48 |
| IHP SG13G2 (PSP103) | M MLP correction (primary) | 0.1553 | 0.1553 | 0.1447 | 0.1659 | 0.1502 | 1.5344 | 0 | — |

- Targets: GF180 (BSIM4) and IHP SG13G2 (PSP103) tier c truth sets 0, 1 and 2, 16 geometries per PDK (8 nMOS, 8 pMOS), 48 targets per PDK. Fit per target: ngspice_fit.polish from the wide-set projection (max_nfev 300, irls 3).
- Folds: leave-one-geometry-out (all three draws of the held-out geometry removed from training), 16 folds per PDK. Arms use the pre-registered hyperparameters (M: 5 seeds, 2000 steps).
- Correction applied post hoc to the fitted card's simulation; the card is not refit with the correction in the loop. This is the setting (stage 1) in which M also failed on the measured devices.
- criterion_S (per PDK): M all-device <= 0.90 × C0 and M < T1. Secondary; not used for the E1 decision. It fails on both PDKs.
- all-device = mean over 48 targets; combined = mean of the nMOS and pMOS family means (equal family counts, so it equals the all-device mean).


### Surrogate learning curves, 43 parameters, round-1 box: training-set size vs held-out accuracy and end-to-end search on the 18 measured devices

| surrogate | train fraction | training samples | epochs | train time (s) | test loss | curve NRMSE, median | mean absolute score error | devices where the surrogate ranks the direct fit better | end-to-end all-device | end-to-end combined | end-to-end nMOS | end-to-end pMOS | failed devices | devices where a bank start is selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| paper MLP | 0.125 | 10,732 | 131 | 76 | 0.1579 | 0.3803 | 1.0987 | 7/18 | 0.2588 | 0.2483 | 0.1541 | 0.3425 | 0 | 17 |
| paper MLP | 0.25 | 21,463 | 200 | 216 | 0.0792 | 0.2321 | 0.5648 | 8/18 | 0.2599 | 0.2493 | 0.1541 | 0.3446 | 0 | 18 |
| paper MLP | 0.5 | 42,926 | 189 | 396 | 0.0495 | 0.1514 | 0.4061 | 7/18 | 0.2504 | 0.2405 | 0.1511 | 0.3299 | 0 | 13 |
| paper MLP | 1 | 85,853 | 190 | 787 | 0.0331 | 0.1068 | 0.2879 | 6/18 | 0.2348 | 0.2248 | 0.1345 | 0.3151 | 0 | 11 |
| DeepONet | 0.125 | 10,732 | 86 | 285 | 0.1683 | 0.4189 | 1.0573 | 7/18 | 0.2581 | 0.2477 | 0.1541 | 0.3412 | 0 | 17 |
| DeepONet | 0.25 | 21,463 | 152 | 968 | 0.0985 | 0.2894 | 1.0026 | 9/18 | 0.2497 | 0.2401 | 0.1535 | 0.3266 | 0 | 15 |
| DeepONet | 0.5 | 42,926 | 200 | 2,468 | 0.0501 | 0.1419 | 0.5307 | 6/18 | 0.2360 | 0.2269 | 0.1451 | 0.3086 | 0 | 13 |
| DeepONet | 1 | 85,853 | 165 | 4,025 | 0.0326 | 0.0911 | 0.2516 | 6/18 | 0.2274 | 0.2193 | 0.1465 | 0.2921 | 0 | 9 |
| FNO | 0.125 | 10,732 | 80 | 982 | 0.1394 | 0.3471 | 0.9177 | 7/18 | 0.2494 | 0.2397 | 0.1523 | 0.3272 | 0 | 15 |
| FNO | 0.25 | 21,463 | 80 | 1,649 | 0.0802 | 0.2359 | 0.5837 | 4/18 | 0.2549 | 0.2448 | 0.1541 | 0.3355 | 0 | 16 |
| FNO | 0.5 | 42,926 | 80 | 3,546 | 0.0431 | 0.1495 | 0.3579 | 10/18 | 0.2468 | 0.2372 | 0.1514 | 0.3230 | 0 | 15 |
| FNO | 1 | 85,853 | 80 | 5,791 | 0.0259 | 0.1006 | 0.2427 | 6/18 | 0.2549 | 0.2448 | 0.1541 | 0.3355 | 0 | 16 |
| NGSpice in place of the surrogate | — | — | — | — | — | — | — | 18/18 | 0.1125 | 0.1053 | 0.0410 | 0.1696 | 0 | 0 |

- Script: scripts/research/surrogate_arch_study.py on out/cold_loop/full/round_1 (cache) and data/processed/cold_loop/full_r1 (bank), seed 20260818. The training subset at fraction f is the first round(f × 85,853) indices of one fixed permutation of the pipeline's training split; validation and test splits are unchanged.
- Fractions 1/8, 1/4, 1/2: queue job archlc_full_r1 (first attempt 2026-10-18 03:38, killed rc=143 after 10,138 s; restarted 2026-10-19 09:01, finished 10:05; the restart skipped finished reports and trained fno_f0.5). Fraction 1: the reports of arch_full_r1 (2026-10-16/17), same script, split and seed.
- Recipe: AdamW lr 1e-3, weight decay 1e-5, cosine schedule, batch 512, early stopping (patience 25) on validation loss; at most 200 epochs (FNO 80).
- End-to-end = fps.finite_difference_surrogate_extract with the trained surrogate frozen: 8 starts, TRF max_nfev 100, diff_step 0.02, regularization 1e-4, NGSpice validation of the top 4 endpoints, the first 4 starts and the published fallback; the selected card is re-simulated in NGSpice on the measured 77 K data. One fixed method per row; no per-device selection across rows.
- all-device = mean over 18 devices; combined = mean of the nMOS (8) and pMOS (10) family means. bank start = devices whose selected card is one of the LHC bank starts (selected_candidate_kind lhc_start) rather than an FD endpoint; bank starts are the same for every surrogate, so rows with many bank-start selections share per-device values.
- The end-to-end search ran on the 18 measured devices; it is a diagnostic ablation and is not used for method selection.

