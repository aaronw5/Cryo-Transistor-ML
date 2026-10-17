# Tables, 2026-10-16 dev_iteration1

Generated 2026-10-16 22:07 by `scripts/research/report_figs.py --topic dev_iteration1 --date 2026-10-16` from result files only. RRMS is the paper metric with frozen published-baseline curve inclusion, re-simulated in NGSpice.

### DEV comparison, 36 simulated targets: pooled all-device mean paper RRMS by arm and NGSpice budget per target

| node | arm | mode | B500 | B1000 | B2000 | B5000 | AUC log10 | median calls to 0.05 | median calls to 0.01 | failed targets (per budget) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid TR | hybrid_surrogate | fixed | 0.0463 | 0.0313 | 0.0193 | 0.0114 | -1.5080 | 122 | 2551 | 0/0/0/0 |
| hybrid TR | hybrid_none | fixed | 0.0579 | 0.0316 | 0.0169 | 0.0058 | -1.5292 | 149 | 1660 | 0/0/0/0 |
| n1c | cmaes_trf | fixed | 0.0565 | 0.0337 | 0.0239 | 0.0109 | -0.9245 | 3152 | 4057 | 0/0/0/0 |
| n1c | lq_cmaes | fixed | 0.1814 | 0.1576 | 0.1386 | 0.0939 | -0.8289 | never | never | 0/0/0/0 |
| n1a | sketch_surrogate_k8 | fixed | 0.0313 | 0.0221 | 0.0148 | 0.0091 | -1.6297 | 66 | 1012 | 0/0/0/0 |
| n1a | sketch_none_k8 | fixed | 0.0370 | 0.0193 | 0.0092 | 0.0021 | -1.7251 | 69 | 630 | 0/0/0/0 |
| n1a | sketch_surrogate_k4 | fixed | — | — | — | 0.0090 | -1.6494 | 85 | 760 | –/–/–/0 |
| n1a | sketch_surrogate_k16 | fixed | — | — | — | 0.0080 | -1.6375 | 76 | 682 | –/–/–/0 |
| n1a | sketch_fullfd_surrogate | fixed | — | — | — | 0.0101 | -1.5582 | 125 | 924 | –/–/–/0 |
| n1a | sketch_fullfd_none | fixed | — | — | — | 0.0052 | -1.5381 | 135 | 1224 | –/–/–/0 |
| n1b | anytime_sur_presearch_equal | fixed | — | 0.0173 | — | 0.0083 | -1.6693 | 125 | 920 | –/0/–/0 |
| n1b | anytime_sur_presearch_race | fixed | 0.0287 | 0.0173 | 0.0134 | 0.0070 | -1.6021 | 206 | 694 | 0/0/0/0 |
| n1b | anytime_none_race | fixed | 0.0428 | 0.0193 | 0.0112 | 0.0082 | -1.4230 | 371 | 1040 | 0/0/0/0 |

- Pooled all-device mean paper RRMS (frozen inclusion) over 36 DEV targets (set_0, set_1, 18 devices each) at fixed cumulative NGSpice budgets per target; every NGSpice call counts, Jacobian probes included. Mode fixed = one run directory per budget, value = one fresh NGSpice score of the final card.
- AUC log10 = per-target mean of log10(max(best-so-far RRMS, 1e-6)) over a 40-point log grid of calls (10 to the largest budget), averaged over targets (lower = lower RRMS earlier).
- Calls to 0.05 / 0.01 = median over targets of the first NGSpice call count at which the best-so-far RRMS is at or below the threshold, from the largest-budget trace; never = the median target does not reach it.
- Failed targets = targets whose final card has a non-finite included metric curve, per budget B500/B1000/B2000/B5000; – = budget not run.
- References: pipeline_r1 0.1590 at 9 mean NGSpice calls; 2 failed target(s), valid-only 0.1554 over 34, penalized 0.2109; pipeline_r1_fd 0.0802 at 376 mean NGSpice calls; 2 failed target(s), valid-only 0.0775 over 34, penalized 0.1326; bank_oracle 0.2046 at 0 mean NGSpice calls.
- Non-PASS checks in compare.json: WARN pipeline_r1: 2 target(s) with failed included curves (stored mean drops them); WARN pipeline_r1_fd: 2 target(s) with failed included curves (stored mean drops them).

### DEV paired comparisons over the 36 targets: difference A − B of paper RRMS per target at each budget

| A vs B | budget | n | mean A | mean B | mean difference A − B | 95% CI | win rate A | win rate 95% CI | W/T/L | failed targets (A, B) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sketch_surrogate_k8 vs sketch_none_k8 | B500 | 36 | 0.0313 | 0.0370 | -0.0058 | [-0.0126, 0.0003] | 0.67 | [0.50, 0.81] | 24/1/11 | 0 |
| sketch_surrogate_k8 vs sketch_none_k8 | B1000 | 36 | 0.0221 | 0.0193 | 0.0028 | [-0.0027, 0.0085] | 0.53 | [0.36, 0.69] | 19/0/17 | 0 |
| sketch_surrogate_k8 vs sketch_none_k8 | B2000 | 36 | 0.0148 | 0.0092 | 0.0057 | [1.4e-05, 0.0127] | 0.44 | [0.28, 0.61] | 16/0/20 | 0 |
| sketch_surrogate_k8 vs sketch_none_k8 | B5000 | 36 | 0.0091 | 0.0021 | 0.0069 | [0.0024, 0.0133] | 0.25 | [0.11, 0.39] | 9/0/27 | 0 |
| hybrid_surrogate vs hybrid_none | B500 | 36 | 0.0463 | 0.0579 | -0.0116 | [-0.0230, -0.0024] | 0.64 | [0.47, 0.78] | 23/1/12 | 0 |
| hybrid_surrogate vs hybrid_none | B1000 | 36 | 0.0313 | 0.0316 | -0.0003 | [-0.0129, 0.0138] | 0.64 | [0.47, 0.78] | 23/0/13 | 0 |
| hybrid_surrogate vs hybrid_none | B2000 | 36 | 0.0193 | 0.0169 | 0.0025 | [-0.0022, 0.0070] | 0.56 | [0.39, 0.72] | 20/0/16 | 0 |
| hybrid_surrogate vs hybrid_none | B5000 | 36 | 0.0114 | 0.0058 | 0.0056 | [0.0018, 0.0112] | 0.28 | [0.14, 0.44] | 10/4/22 | 0 |
| anytime_sur_presearch_race vs anytime_none_race | B500 | 36 | 0.0287 | 0.0428 | -0.0141 | [-0.0258, -0.0045] | 0.78 | [0.64, 0.92] | 28/0/8 | 0 |
| anytime_sur_presearch_race vs anytime_none_race | B1000 | 36 | 0.0173 | 0.0193 | -0.0020 | [-0.0080, 0.0041] | 0.61 | [0.44, 0.75] | 22/0/14 | 0 |
| anytime_sur_presearch_race vs anytime_none_race | B2000 | 36 | 0.0134 | 0.0112 | 0.0023 | [-0.0018, 0.0067] | 0.47 | [0.31, 0.64] | 17/3/16 | 0 |
| anytime_sur_presearch_race vs anytime_none_race | B5000 | 36 | 0.0070 | 0.0082 | -0.0012 | [-0.0056, 0.0025] | 0.47 | [0.31, 0.64] | 17/4/15 | 0 |
| sketch_surrogate_k8 vs cmaes_trf | B500 | 36 | 0.0313 | 0.0565 | -0.0252 | [-0.0430, -0.0103] | 0.75 | [0.61, 0.89] | 27/0/9 | 0 |
| sketch_surrogate_k8 vs cmaes_trf | B1000 | 36 | 0.0221 | 0.0337 | -0.0115 | [-0.0267, 0.0010] | 0.58 | [0.42, 0.75] | 21/0/15 | 0 |
| sketch_surrogate_k8 vs cmaes_trf | B2000 | 36 | 0.0148 | 0.0239 | -0.0090 | [-0.0200, -0.0004] | 0.67 | [0.50, 0.81] | 24/1/11 | 0 |
| sketch_surrogate_k8 vs cmaes_trf | B5000 | 36 | 0.0091 | 0.0109 | -0.0018 | [-0.0084, 0.0059] | 0.69 | [0.53, 0.83] | 25/0/11 | 0 |
| sketch_none_k8 vs hybrid_none | B500 | 36 | 0.0370 | 0.0579 | -0.0209 | [-0.0330, -0.0112] | 0.97 | [0.92, 1.00] | 35/0/1 | 0 |
| sketch_none_k8 vs hybrid_none | B1000 | 36 | 0.0193 | 0.0316 | -0.0123 | [-0.0211, -0.0060] | 0.92 | [0.81, 1.00] | 33/0/3 | 0 |
| sketch_none_k8 vs hybrid_none | B2000 | 36 | 0.0092 | 0.0169 | -0.0077 | [-0.0131, -0.0034] | 0.86 | [0.75, 0.97] | 31/1/4 | 0 |
| sketch_none_k8 vs hybrid_none | B5000 | 36 | 0.0021 | 0.0058 | -0.0037 | [-0.0058, -0.0020] | 1.00 | [1.00, 1.00] | 36/0/0 | 0 |

- Per-target values: compare.json arms.<arm>.budgets.<budget>.values (one fresh NGSpice score of each final card). Difference = A − B; negative = A has the lower RRMS.
- Win = A lower than B by more than the tie tolerance max(1e-6, 1% of the larger value) (dev_compare.paired); win rate = wins / n; W/T/L = wins, ties, losses of A.
- 95% CI: percentile bootstrap over targets, 10,000 resamples, seed 0; the mean difference and the win rate use the same resamples.
- failed = targets with a non-finite included curve in A or B at that budget.

### Failed-curve audit: headline numbers as stored, valid devices only, and penalized

| headline | failed devices | failed device(s) | stored all-device | stored combined | valid-only all-device | valid-only combined | penalized all-device | penalized combined |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| pipeline 43, round 1, raw (swap / misrank / rounds claims) | 1 | nmos_L0p15_W1p6 | 0.2375 | 0.2281 | 0.2395 | 0.2238 | 0.2920 | 0.2894 |
| pipeline 43, round 1, + FD | 1 | nmos_L0p15_W1p6 | 0.1824 | 0.1730 | 0.1853 | 0.1698 | 0.2373 | 0.2347 |
| pipeline 43, round 4, raw (rounds / cryo-43 claims) | 1 | nmos_L8_W1p6 | 0.1513 | 0.1427 | 0.1586 | 0.1454 | 0.6500 | 0.7037 |
| pipeline 43, round 4, + FD | 1 | nmos_L8_W1p6 | 0.1465 | 0.1380 | 0.1539 | 0.1410 | 0.6455 | 0.6994 |
| A tandem 43, round 2 | 1 | nmos_L100_W100 | 5.8288 | 6.5013 | 6.0969 | 7.3073 | 5.9743 | 6.6650 |
| A tandem 43, round 3 (logged as nan) | 1 | nmos_L0p25_W1p6 | 2.2458 | 1.9891 | 2.2458 | 1.9891 | 2.6766 | 2.5807 |
| A tandem 43, round 4 | 1 | pmos_L4_W7 | 1.6969 | 1.6348 | 1.7538 | 1.7161 | 1.8114 | 1.7378 |
| A tandem 7, round 1 | 1 | pmos_L4_W7 | 2.0402 | 1.9321 | 2.1467 | 2.0806 | 2.1608 | 2.0406 |
| pipeline 43 r4 + FD, sim set 0 | 1 | pmos_L0p5_W0p42 | 4.9243 | 5.5227 | 5.2116 | 5.5281 | 4.9858 | 5.5780 |
| pipeline 43 r4 + FD, sim set 2 | 1 | pmos_L0p5_W0p42 | 5.0292 | 5.6422 | 5.3225 | 5.6468 | 5.0907 | 5.6975 |
| DEV pipeline r1, set_0 | 1 | nmos_L0p19_W7 | 0.1688 | 0.1659 | 0.1665 | 0.1611 | 0.2232 | 0.2271 |
| DEV pipeline r1, set_1 | 1 | pmos_L0p35_W0p55 | 0.1492 | 0.1456 | 0.1442 | 0.1425 | 0.1986 | 0.1900 |
| DEV pipeline r1 + FD, set_0 | 1 | nmos_L0p19_W7 | 0.0918 | 0.0909 | 0.0887 | 0.0867 | 0.1465 | 0.1525 |
| DEV pipeline r1 + FD, set_1 | 1 | pmos_L0p35_W0p55 | 0.0687 | 0.0661 | 0.0663 | 0.0650 | 0.1187 | 0.1110 |
| pipeline 7, round 4, raw | 0 | none | 0.1697 | 0.1602 | 0.1697 | 0.1602 | 0.1697 | 0.1602 |
| pipeline 7, round 4, + FD | 0 | none | 0.1643 | 0.1548 | 0.1643 | 0.1548 | 0.1643 | 0.1548 |
| NGSpice swap 43, round 1 | 0 | none | 0.1125 | 0.1053 | 0.1125 | 0.1053 | 0.1125 | 0.1053 |
| NGSpice swap 43, round 1, + FD | 0 | none | 0.1106 | 0.1036 | 0.1106 | 0.1036 | 0.1106 | 0.1036 |
| direct NGSpice 43, round 3 | 0 | none | 0.1049 | 0.0981 | 0.1049 | 0.0981 | 0.1049 | 0.0981 |
| direct NGSpice 43, round 3, + FD | 0 | none | 0.1044 | 0.0976 | 0.1044 | 0.0976 | 0.1044 | 0.0976 |

- Runs audited: 227; runs with failed devices: 16. Inclusion set: out/pdk_baseline/pdk_baseline.json (frozen inclusion).
- stored = the value in the result file: for a failed device, metrics.device_rrms_new averages only the surviving included curves (upstream rrmsCalc.py gives NaN).
- valid-only: mean over devices whose included curves all simulate; n = 18 minus failed devices.
- penalized convention: each failed included curve at RRMS 10 (ngspice_fit.PENALTY); not the paper metric.
- all-device = mean over devices; combined = mean of the nMOS and pMOS family means.
