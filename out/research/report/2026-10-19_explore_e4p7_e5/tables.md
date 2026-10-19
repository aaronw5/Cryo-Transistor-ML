# Tables, 2026-10-19 explore_e4p7_e5

Generated 2026-10-19 14:43 by `PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_e4p7_e5.py --date 2026-10-19` from result files only. RRMS is the paper metric with frozen published-baseline curve inclusion, strict.

### E4 seven-parameter runs (secondary): pre-registered checks per candidate and split, with the p43 vd decision for reference

| run, split | candidate | coverage of 90% interval | width of 90% interval | width limit 4 × single RRMS | C1 | held-out RRMS | limit 0.95 × single | devices below single | C2 | C3 learned <= crossinterp | C1 and C2 and C3 | rule output for the split |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p7 vd | ens | 0.6503 | 1.8807 | 5.8144 | no | 1.5017 | 1.3809 | 5/12 | no | — | no | KILL |
| p7 vd | laplace | 0.6083 | 2.2133 | 5.8144 | no | 1146.5398 | 1.3809 | 5/12 | no | — | no |  |
| p7 vd | learned | 0.9796 | 78.4797 | 5.8144 | no | 0.8253 | 1.3809 | 10/12 | yes | no (0.825 vs 0.161) | no |  |
| p7 sparse | ens | 0.6175 | 3.8140 | 34.2932 | no | 8.7947 | 8.1446 | 5/12 | no | — | no | KILL |
| p7 sparse | laplace | 0.6035 | 618752.2990 | 34.2932 | no | 129860.2471 | 8.1446 | 3/12 | no | — | no |  |
| p7 sparse | learned | 0.9683 | 20.3049 | 34.2932 | no | 1.9220 | 8.1446 | 10/12 | yes | no (1.922 vs 0.682) | no |  |
| p7 vg | ens | 0.7566 | 0.3633 | 1.0108 | yes | 0.2589 | 0.2401 | 1/12 | no | — | no | PARTIAL_CALIBRATION_ONLY |
| p7 vg | laplace | 0.7661 | 0.5924 | 1.0108 | yes | 0.3125 | 0.2401 | 1/12 | no | — | no |  |
| p7 vg | learned | 0.9883 | 0.3412 | 1.0108 | no | 0.0555 | 0.2401 | 11/12 | yes | no (0.056 vs 0.044) | no |  |
| p43 vd (decision) | ens | 0.7438 | 1.0060 | 1.6708 | yes | 0.4418 | 0.3968 | 6/12 | no | — | no | PARTIAL_CALIBRATION_ONLY |

- Pre-registered rule (prereg.json, amendments 1 and 2): C1 = all-device mean coverage of the central 90% interval in [0.70, 0.95] AND all-device mean width <= 4 × single's held-out RRMS; C2 = held-out RRMS <= 0.95 × single (valid-only and paired) AND lower than single on >= 8 of 12 devices; C3 (learned only) = held-out RRMS <= crossinterp's (valid-only and paired). CONTINUE if a candidate meets C1, C2 and C3; PARTIAL_CALIBRATION_ONLY if no candidate meets all but one meets C1; KILL if none meets C1.
- p7 = seven-parameter box (out/cold_loop/seven/round_1/spec.json). prereg.json lists p7 vd and p7 vg as secondary results without a decision; sparse is secondary by amendment 1. The rule outputs in the p7 rows come from the same decide() function applied to the stored strict summaries (queue jobs E4_p7_vd, E4_p7_vg, E4_p7_sparse) and do not change the E4 decision.
- Last row: the E4 decision (p43 vd, stored decision block of queue job E4_p43_vd), repeated for reference; its only C1 candidate is ens.
- single held-out RRMS (all-device), p7: vd 1.4536, sparse 8.5733, vg 0.2527. Width is mean (hi − lo) / den, in the same units as the per-curve RRMS.


### E4 held-out-curve RRMS and interval metrics per method, seven parameters, splits vd, sparse, vg

| split | method | all-device | combined | nMOS | pMOS | median | finite devices | devices below single | coverage 0.5 | coverage 0.9 | width 0.9 | interval score 0.9 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vd | single | 1.4536 | 1.2764 | 0.2134 | 2.3395 | 0.6920 | 12/12 | — | — | — | — | — |
| vd | ens | 1.5017 | 1.3178 | 0.2143 | 2.4212 | 0.6460 | 12/12 | 5 | 0.4749 | 0.6503 | 1.8807 | 14.2789 |
| vd | laplace | 1146.5398 | 982.8191 | 0.4948 | 1965.1434 | 0.8973 | 12/12 | 5 | 0.4720 | 0.6083 | 2.2133 | 11.7622 |
| vd | conformal | 1.4536 | 1.2764 | 0.2134 | 2.3395 | 0.6920 | 12/12 | 0 | 0.5419 | 0.9670 | 170.0689 | 170.4932 |
| vd | crossinterp | 0.1607 | 0.1433 | 0.0388 | 0.2479 | 0.1012 | 12/12 | 12 | — | — | — | — |
| vd | crossinterp_conformal | 0.1607 | 0.1433 | 0.0388 | 0.2479 | 0.1012 | 12/12 | 12 | 0.5444 | 0.9754 | 7.0558 | 7.0737 |
| vd | learned | 0.8253 | 0.7558 | 0.3390 | 1.1727 | 0.4185 | 12/12 | 10 | 0.5219 | 0.9796 | 78.4797 | 78.7204 |
| sparse | single | 8.5733 | 7.3723 | 0.1663 | 14.5783 | 0.6949 | 12/12 | — | — | — | — | — |
| sparse | ens | 8.7947 | 7.5614 | 0.1618 | 14.9611 | 0.7145 | 12/12 | 5 | 0.3717 | 0.6175 | 3.8140 | 52.0738 |
| sparse | laplace | 129860.2471 | 111308.8503 | 0.4695 | 222617.2310 | 0.8730 | 12/12 | 3 | 0.3303 | 0.6035 | 618752.2990 | 618789.4460 |
| sparse | conformal | 8.5733 | 7.3723 | 0.1663 | 14.5783 | 0.6949 | 12/12 | 0 | 0.5430 | 0.9742 | 55.2350 | 55.3993 |
| sparse | crossinterp | 0.6817 | 0.6031 | 0.1318 | 1.0745 | 0.4604 | 12/12 | 12 | — | — | — | — |
| sparse | crossinterp_conformal | 0.6817 | 0.6031 | 0.1318 | 1.0745 | 0.4604 | 12/12 | 12 | 0.5439 | 0.9806 | 21.8624 | 21.9010 |
| sparse | learned | 1.9220 | 1.6786 | 0.2182 | 3.1390 | 0.2925 | 12/12 | 10 | 0.5326 | 0.9683 | 20.3049 | 20.3873 |
| vg | single | 0.2527 | 0.2249 | 0.0580 | 0.3918 | 0.2722 | 12/12 | — | — | — | — | — |
| vg | ens | 0.2589 | 0.2307 | 0.0618 | 0.3996 | 0.2738 | 12/12 | 1 | 0.5825 | 0.7566 | 0.3633 | 2.2481 |
| vg | laplace | 0.3125 | 0.2898 | 0.1534 | 0.4261 | 0.3153 | 12/12 | 1 | 0.6233 | 0.7661 | 0.5924 | 2.5024 |
| vg | conformal | 0.2527 | 0.2249 | 0.0580 | 0.3918 | 0.2722 | 12/12 | 0 | 0.5536 | 0.9785 | 3.0397 | 3.1239 |
| vg | crossinterp | 0.0436 | 0.0401 | 0.0197 | 0.0606 | 0.0383 | 12/12 | 12 | — | — | — | — |
| vg | crossinterp_conformal | 0.0436 | 0.0401 | 0.0197 | 0.0606 | 0.0383 | 12/12 | 12 | 0.5463 | 0.9732 | 0.4474 | 0.4718 |
| vg | learned | 0.0555 | 0.0573 | 0.0681 | 0.0466 | 0.0451 | 12/12 | 11 | 0.5504 | 0.9883 | 0.3412 | 0.3451 |

- all-device = mean over the 12 REAL_DEV devices (strict: a non-finite held-out prediction makes the device NaN; finite = devices with a finite value; 0 single-card failures in all three p7 splits); combined = mean of the nMOS (5) and pMOS (7) family means.
- Held-out curves: vd = Id-Vd curves except the highest gate bias; sparse = every included curve except Id-Vg at Vd 0.01/1.85 and Id-Vd at the highest gate bias; vg = Id-Vg at Vd 0.37/0.74/1.11/1.48.
- select feasible (amendment 1, secondary): `e4_analyze.py --run p7 --select feasible` was run for this report on the fitted records (no NGSpice). Every per-device value equals the strict one on all three p7 splits, so only strict rows are listed.
- conformal and crossinterp_conformal reuse the single and crossinterp point predictions; ens_env and laplace_param are reported in the JSON summaries.


### E4 43 vs seven parameters per split: held-out RRMS of the single card, the candidates and the crossinterp control

| split | run | mean train objective (single) | single all-device | ens all-device | Laplace all-device | learned all-device | crossinterp all-device | single combined | crossinterp combined | learned below single (devices) | Spearman rho, held-out RRMS vs sloppy fraction | rule output |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vd | p43 (decision) | 0.1299 | 0.4177 | 0.4418 | 500.2099 | 10.8691 | 0.0977 | 0.3751 | 0.0897 | 1 | -0.07 (p 0.829) | PARTIAL_CALIBRATION_ONLY |
| vd | p7 | 0.1490 | 1.4536 | 1.5017 | 1146.5398 | 0.8253 | 0.1607 | 1.2764 | 0.1433 | 10 | 0.35 (p 0.265) | KILL |
| sparse | p43 | 0.1134 | 0.3945 | 0.4322 | 5143.1805 | 3.0887 | 0.2718 | 0.3579 | 0.2517 | 1 | 0.03 (p 0.931) | KILL |
| sparse | p7 | 0.1392 | 8.5733 | 8.7947 | 129860.2471 | 1.9220 | 0.6817 | 7.3723 | 0.6031 | 10 | 0.57 (p 0.051) | KILL |
| vg | p43 | 0.1395 | 0.2070 | 0.2070 | 0.6146 | 0.0664 | 0.0466 | 0.1867 | 0.0434 | 11 | -0.17 (p 0.587) | PARTIAL_CALIBRATION_ONLY |
| vg | p7 | 0.1777 | 0.2527 | 0.2589 | 0.3125 | 0.0555 | 0.0436 | 0.2249 | 0.0401 | 11 | -0.83 (p 0.001) | PARTIAL_CALIBRATION_ONLY |

- Strict selection, 12 REAL_DEV devices, 0 failed single-card devices in every run. train obj. = mean over devices of the single card's train objective (mean train-curve RRMS plus the held-out-bias feasibility penalty, which is 0 for every single card).
- Spearman = rank correlation across the 12 devices between the single card's held-out RRMS and the fraction of held-out sensitivity in the sloppy subspace (s / s_max < 1e-2); pre-registered as reported without a criterion.
- Only p43 vd is the pre-registered decision; the other five rule outputs are reported without a decision.


### E5 decision: shared discrepancy learned on other devices vs no correction, pre-registered stage-1 and stage-2 checks

| stage | base card | comparison | control all-device | arm all-device | (control − arm) / control | devices improved | continue condition | kill condition (stage 1) | rule output | role |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| stage 1 | b43 | A2 MLP vs A0 | 0.1073 | 0.2004 | -0.8680 | 1/12 | no | met | KILL | decision |
| stage 1 light | b43 | A2 MLP vs A0 | 0.1073 | 0.1809 | -0.6862 | 2/12 | — | — | KILL | light settings, not a decision |
| stage 1 | b43 | A1 const vs A0 | 0.1073 | 0.1221 | -0.1382 | 0/12 | no | met | KILL | reported, no decision |
| stage 1 | b43 | A4 no geometry vs A0 | 0.1073 | 0.1681 | -0.5673 | 5/12 | no | met | KILL | reported, no decision |
| stage 1 | b43 | A3 oracle vs A0 | 0.1073 | 0.0343 | 0.6801 | 12/12 | yes | not met | CONTINUE | oracle, not a method |
| stage 1 | b7 | A2 MLP vs A0 | 0.1735 | 0.2252 | -0.2979 | 3/12 | no | met | KILL | reported, no decision |
| stage 1 | b7 | A1 const vs A0 | 0.1735 | 0.2184 | -0.2588 | 1/12 | no | met | KILL | reported, no decision |
| stage 1 | b7 | A4 no geometry vs A0 | 0.1735 | 0.2922 | -0.6845 | 5/12 | no | met | KILL | reported, no decision |
| stage 1 | b7 | A3 oracle vs A0 | 0.1735 | 0.0419 | 0.7586 | 12/12 | yes | not met | CONTINUE | oracle, not a method |
| stage 2 | b43 | joint refit vs control polish | 0.1072 | 0.1235 | -0.1520 | 3/12 | no | — | KILL | diagnostic (stage 1 not AMBIGUOUS) |

- Pre-registered rule (out/research/explore/E5/prereg.json): stage 1 CONTINUE if the all-DEV mean RRMS reduction (A0 − A2) / A0 >= 0.15 AND A2 < A0 on >= 8 of 12 devices; KILL if reduction < 0.05 OR A2 < A0 on <= 6 of 12; otherwise AMBIGUOUS, decided on stage 2 (CONTINUE if (control − joint) / control >= 0.15 AND joint < control on >= 8 of 12, otherwise KILL). The decision is the first row (b43, A2 vs A0).
- Metric: frozen-inclusion paper RRMS of the corrected curves on the held-out device, REAL_DEV leave-one-device-out, 0 failed devices in every arm. Rule outputs for the other arms and for the seven-parameter base are computed with the same function for reporting.
- Stage-1 run (queue job E5_stage1, 2026-10-19 14:19-14:29, 625 s): 3000 full-batch steps, seeds 0-4. Recorded deviations from the prereg text: optimizer torch AdamW (decoupled weight decay 1e-4); weights renormalized to sum 1 per device; MPS float32. Light row: 800 steps, seeds 0-2 (commit 773abe3).
- Stage 2 (queue job E5_stage2, 64 s): per fold, bounded TRF polish of the 43-parameter card (max_nfev 400, diff_step 0.01, irls 2) with the fold's A2 delta frozen in the residual; control = the same polish without delta from the same start. Run as a diagnostic because the stage-1 output is KILL.


### E5 RRMS per base card and arm (stage 1) and for the stage-2 refits

| stage | base card | arm | all-device | combined | nMOS | pMOS | median | devices below the control | failed devices | mean NGSpice calls per device |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stage 1 | b43 | A0 control | 0.1073 | 0.0981 | 0.0433 | 0.1530 | 0.0823 | — | 0/12 | — |
| stage 1 | b43 | A1 const | 0.1221 | 0.1110 | 0.0442 | 0.1777 | 0.1153 | 0 | 0/12 | — |
| stage 1 | b43 | A2 MLP | 0.2004 | 0.2024 | 0.2147 | 0.1902 | 0.1972 | 1 | 0/12 | — |
| stage 1 | b43 | A4 no geometry | 0.1681 | 0.1625 | 0.1286 | 0.1964 | 0.0941 | 5 | 0/12 | — |
| stage 1 | b43 | A3 oracle | 0.0343 | 0.0322 | 0.0197 | 0.0447 | 0.0235 | 12 | 0/12 | — |
| stage 1 | b7 | A0 control | 0.1735 | 0.1600 | 0.0793 | 0.2407 | 0.1446 | — | 0/12 | — |
| stage 1 | b7 | A1 const | 0.2184 | 0.2002 | 0.0908 | 0.3096 | 0.2482 | 1 | 0/12 | — |
| stage 1 | b7 | A2 MLP | 0.2252 | 0.2175 | 0.1716 | 0.2635 | 0.1997 | 3 | 0/12 | — |
| stage 1 | b7 | A4 no geometry | 0.2922 | 0.2882 | 0.2641 | 0.3123 | 0.1422 | 5 | 0/12 | — |
| stage 1 | b7 | A3 oracle | 0.0419 | 0.0387 | 0.0197 | 0.0577 | 0.0279 | 12 | 0/12 | — |
| stage 2 | b43 | control polish | 0.1072 | 0.0981 | 0.0432 | 0.1529 | 0.0820 | — | 0/12 | 300 |
| stage 2 | b43 | joint refit (corrected) | 0.1235 | 0.1206 | 0.1030 | 0.1381 | 0.1210 | 3 | 0/12 | 363 |
| stage 2 | b43 | joint refit card, no delta | 0.1890 | 0.1842 | 0.1549 | 0.2134 | 0.1602 | 0 | 0/12 | — |

- all-device = mean over the 12 REAL_DEV devices; combined = mean of the nMOS (5) and pMOS (7) family means; strict frozen-inclusion paper RRMS; a device with a non-finite included curve counts as failed (none in this run).
- Base cards: b43 = out/research/polished/direct43_r3_fd/real/summary.json (direct NGSpice 43 parameters, round 3, + FD); b7 = out/research/polished/direct7_fd/real/summary.json. A0 reproduces the stored RRMS of both (max abs difference 0).
- < control: stage 1 against A0; stage 2 against the control polish. A3 is trained in-sample on the held-out device (upper bound for the network class, not a method).
- Stage-2 'card, no delta' = the joint-refit card scored without the correction (diagnostic).

