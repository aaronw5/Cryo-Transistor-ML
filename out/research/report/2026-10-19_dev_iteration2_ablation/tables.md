# Tables, 2026-10-19 dev_iteration2_ablation

Generated 2026-10-19 09:18 by `scripts/research/report_figs.py --topic dev_iteration2_ablation --date 2026-10-19` from result files only. RRMS is the paper metric with frozen published-baseline curve inclusion, re-simulated in NGSpice.

### DEV iteration 2, tier a: well-specified simulated targets (sim_targets set_0, set_1): pooled mean paper RRMS (strict) by arm and budget (ablation)

| family | arm | variant | B500 | B1000 | B2000 | B5000 | AUC log10 (seed 0, raw RRMS) | seeds per budget | failed targets per budget |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| sketched k=8 | sketch_surrogate_k8 | surrogate term on (λ = 1) | 0.0334 | 0.0221 | 0.0143 | 0.0082 | -1.6297 | 3/1/3/3 | 0/0/0/0 |
| sketched k=8 | sketch_none_k8 | NGSpice only (λ = 0) | 0.0367 | 0.0193 | 0.0099 | 0.0024 | -1.7251 | 3/1/3/3 | 0/0/0/0 |
| sketched k=8 | lam_curve_k8 | λ per curve (n2b) | 0.0317 | 0.0211 | 0.0127 | 0.0062 | -1.7097 | 3/1/3/3 | 0/0/0/0 |
| sketched k=8 | lam_scalar_k8 | λ scalar (n2b ablation) | 0.0330 | 0.0217 | 0.0121 | 0.0057 | -1.6787 | 1/1/1/1 | 0/0/0/0 |
| sketched k=8 | lam_cusum_k8 | λ CUSUM (n2b ablation) | 0.0311 | 0.0194 | 0.0091 | 0.0034 | -1.7347 | 1/1/1/1 | 0/0/0/0 |
| sketched k=8 | lam_radius_k8 | λ radius (n2b ablation) | 0.0295 | 0.0177 | 0.0117 | 0.0043 | -1.7476 | 1/1/1/1 | 0/0/0/0 |
| sketched k=8 | sketch_surrogate_k4 | k = 4, surrogate (screen) | — | — | — | 0.0090 | -1.6494 | –/–/–/1 | –/–/–/0 |
| sketched k=8 | sketch_surrogate_k16 | k = 16, surrogate (screen) | — | — | — | 0.0080 | -1.6375 | –/–/–/1 | –/–/–/0 |
| sketched k=8 | sketch_fullfd_surrogate | full FD every centre, surrogate | — | — | — | 0.0101 | -1.5582 | –/–/–/1 | –/–/–/0 |
| sketched k=8 | sketch_fullfd_none | full FD every centre, NGSpice only | — | — | — | 0.0052 | -1.5381 | –/–/–/1 | –/–/–/0 |
| full Jacobian | hybrid_surrogate | surrogate term on (λ = 1) | 0.0463 | 0.0313 | 0.0193 | 0.0114 | -1.5080 | 1/1/1/1 | 0/0/0/0 |
| full Jacobian | hybrid_none | NGSpice only (λ = 0) | 0.0579 | 0.0316 | 0.0169 | 0.0058 | -1.5292 | 1/1/1/1 | 0/0/0/0 |
| full Jacobian | hyb_lam_curve | λ per curve (n2b) | 0.0447 | 0.0226 | 0.0149 | 0.0050 | -1.5858 | 1/1/1/1 | 0/0/0/0 |
| racing | anytime_sur_presearch_race | surrogate presearch, racing | 0.0287 | 0.0173 | 0.0134 | 0.0070 | -1.6021 | 1/1/1/1 | 0/0/0/0 |
| racing | anytime_sur_presearch_equal | surrogate presearch, equal split | — | 0.0173 | — | 0.0083 | -1.6693 | –/1/–/1 | –/0/–/0 |
| racing | anytime_none_race | NGSpice only, racing | 0.0428 | 0.0193 | 0.0112 | 0.0082 | -1.4230 | 1/1/1/1 | 0/0/0/0 |
| baseline | dfols_style | DFO-LS-style, 4 starts | 0.0579 | 0.0403 | 0.0199 | 0.0076 | -1.5169 | 1/1/1/1 | 0/0/0/0 |
| baseline | dfols_style_1start | DFO-LS-style, 1 start | 0.0461 | 0.0251 | 0.0130 | 0.0056 | -1.5591 | 1/1/1/1 | 0/0/0/0 |
| baseline | cmaes_trf | CMA-ES + TRF | 0.0565 | 0.0337 | 0.0239 | 0.0109 | -0.9245 | 1/1/1/1 | 0/0/0/0 |
| baseline | lq_cmaes | lq-CMA-ES | 0.1814 | 0.1576 | 0.1386 | 0.0939 | -0.8289 | 1/1/1/1 | 0/0/0/0 |

- tier a: well-specified simulated targets (sim_targets set_0, set_1). 36 targets = set_0 and set_1 × 18 devices; budgets are NGSpice calls per target, Jacobian probes included; each budget is a separate run scored by one fresh NGSpice simulation of the final card.
- Per-target value = mean over the arm's seeds (seed 0 = base label, seeds 1, 2 = '__seed<n>' runs); a target with a failed included curve enters at its penalized value (dev_stats, as pre-registered); seeds and failed columns list B500/B1000/B2000/B5000; – = budget not run.
- AUC log10 = dev_compare's per-target mean of log10(best-so-far raw RRMS) over a 40-point log grid of calls, seed-0 run only.

### DEV iteration 2, tier c: misspecified sky130c targets (set_0, set_1), excess over the per-target floor: pooled mean excess over floor (RRMS − floor) by arm and budget (ablation)

| family | arm | variant | B500 | B1000 | B2000 | B5000 | raw B500 | raw B5000 | AUC log10 (seed 0, raw RRMS) | seeds per budget | failed targets per budget |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sketched k=8 | sketch_surrogate_k8 | surrogate term on (λ = 1) | 0.0221 | 0.0147 | 0.0094 | 0.0047 | 0.0700 | 0.0526 | -1.0662 | 3/1/3/3 | 0/0/0/0 |
| sketched k=8 | sketch_none_k8 | NGSpice only (λ = 0) | 0.0225 | 0.0130 | 0.0061 | 0.0022 | 0.0703 | 0.0501 | -1.0846 | 3/1/3/3 | 0/0/0/0 |
| sketched k=8 | lam_curve_k8 | λ per curve (n2b) | 0.0200 | 0.0113 | 0.0079 | 0.0039 | 0.0678 | 0.0518 | -1.0734 | 3/1/3/3 | 0/0/0/0 |
| sketched k=8 | lam_scalar_k8 | λ scalar (n2b ablation) | 0.0197 | 0.0131 | 0.0081 | 0.0036 | 0.0675 | 0.0514 | -1.0731 | 1/1/1/1 | 0/0/0/0 |
| sketched k=8 | lam_cusum_k8 | λ CUSUM (n2b ablation) | 0.0223 | 0.0140 | 0.0081 | 0.0022 | 0.0702 | 0.0500 | -1.0757 | 1/1/1/1 | 0/0/0/0 |
| sketched k=8 | lam_radius_k8 | λ radius (n2b ablation) | 0.0179 | 0.0125 | 0.0077 | 0.0022 | 0.0657 | 0.0501 | -1.0868 | 1/1/1/1 | 0/0/0/0 |
| sketched k=8 | sketch_surrogate_k4 | k = 4, surrogate (screen) | — | — | — | 0.0047 | — | 0.0525 | -1.0708 | –/–/–/1 | –/–/–/0 |
| sketched k=8 | sketch_surrogate_k16 | k = 16, surrogate (screen) | — | — | — | 0.0040 | — | 0.0518 | -1.0702 | –/–/–/1 | –/–/–/0 |
| sketched k=8 | sketch_fullfd_surrogate | full FD every centre, surrogate | — | — | — | 0.0069 | — | 0.0547 | -1.0463 | –/–/–/1 | –/–/–/0 |
| sketched k=8 | sketch_fullfd_none | full FD every centre, NGSpice only | — | — | — | 0.0041 | — | 0.0519 | -1.0508 | –/–/–/1 | –/–/–/0 |
| full Jacobian | hybrid_surrogate | surrogate term on (λ = 1) | 0.0304 | 0.0165 | 0.0123 | 0.0064 | 0.0782 | 0.0543 | -1.0465 | 1/1/1/1 | 0/0/0/0 |
| full Jacobian | hybrid_none | NGSpice only (λ = 0) | 0.0401 | 0.0195 | 0.0094 | 0.0041 | 0.0879 | 0.0520 | -1.0500 | 1/1/1/1 | 0/0/0/0 |
| full Jacobian | hyb_lam_curve | λ per curve (n2b) | 0.0303 | 0.0167 | 0.0118 | 0.0054 | 0.0781 | 0.0532 | -1.0561 | 1/1/1/1 | 0/0/0/0 |
| racing | anytime_sur_presearch_race | surrogate presearch, racing | 0.0189 | 0.0129 | 0.0091 | 0.0050 | 0.0667 | 0.0528 | -1.0271 | 1/1/1/1 | 0/0/0/0 |
| racing | anytime_sur_presearch_equal | surrogate presearch, equal split | — | 0.0131 | — | 0.0060 | — | 0.0538 | -1.0795 | –/1/–/1 | –/0/–/0 |
| racing | anytime_none_race | NGSpice only, racing | 0.0230 | 0.0117 | 0.0055 | 0.0031 | 0.0708 | 0.0510 | -1.0090 | 1/1/1/1 | 0/0/0/0 |
| baseline | dfols_style | DFO-LS-style, 4 starts | 0.0406 | 0.0232 | 0.0135 | 0.0062 | 0.0884 | 0.0540 | -1.0485 | 1/1/1/1 | 0/0/0/0 |
| baseline | dfols_style_1start | DFO-LS-style, 1 start | 0.0329 | 0.0187 | 0.0123 | 0.0055 | 0.0807 | 0.0534 | -1.0508 | 1/1/1/1 | 0/0/0/0 |
| baseline | cmaes_trf | CMA-ES + TRF | 0.0388 | 0.0242 | 0.0164 | 0.0151 | 0.0866 | 0.0630 | -0.8234 | 1/1/1/1 | 0/0/0/0 |
| baseline | lq_cmaes | lq-CMA-ES | 0.1333 | 0.1244 | 0.1088 | 0.0598 | 0.1812 | 0.1076 | -0.7855 | 1/1/1/1 | 0/0/0/0 |

- tier c: misspecified sky130c targets (set_0, set_1), excess over the per-target floor. 36 targets = set_0 and set_1 × 18 devices; budgets are NGSpice calls per target, Jacobian probes included; each budget is a separate run scored by one fresh NGSpice simulation of the final card.
- Per-target value = mean over the arm's seeds (seed 0 = base label, seeds 1, 2 = '__seed<n>' runs); a target with a failed included curve enters at its penalized value (dev_stats, as pre-registered); seeds and failed columns list B500/B1000/B2000/B5000; – = budget not run.
- Floor = per-target refit floor (floors.json, finalized = min(polish, reference)); mean floor over the 36 targets 0.0478. Excess = RRMS − floor, unclipped. raw RRMS columns are the same runs without the floor subtracted.
- AUC log10 = dev_compare's per-target mean of log10(best-so-far raw RRMS) over a 40-point log grid of calls, seed-0 run only.

### DEV iteration 2 paired comparisons: λ arms vs λ = 0 and λ = 1 (same code, surrogate weight varied)

| A − B | tier | pre-registered id | B500 mean A − B | B500 clustered 95% CI | B500 Wilcoxon p (geometry) | B500 W/T/L | B2000 mean A − B | B2000 clustered 95% CI | B2000 Wilcoxon p (geometry) | B2000 W/T/L | B5000 mean A − B | B5000 clustered 95% CI | B5000 Wilcoxon p (geometry) | B5000 W/T/L |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lam_curve_k8 − sketch_none_k8 | a | C3 | -0.0050 | [-0.0123, 0.0017] | 0.1084 (Holm 1.000) | 22/1/13 | 0.0028 | [-0.0012, 0.0072] | 0.3247 (Holm 1.000) | 15/0/21 | 0.0038 | [0.0016, 0.0068] | 0.0008 (Holm 0.024) | 10/2/24 |
| lam_curve_k8 − sketch_none_k8 | c | C3 | -0.0025 | [-0.0068, 0.0023] | 0.1297 (Holm 1.000) | 24/1/11 | 0.0018 | [0.0003, 0.0038] | 0.0385 (Holm 0.577) | 12/1/23 | 0.0017 | [0.0005, 0.0030] | 0.0090 (Holm 0.197) | 12/0/24 |
| lam_curve_k8 − sketch_surrogate_k8 | a | C4 | -0.0017 | [-0.0054, 0.0011] | 0.2462 (Holm 1.000) | 22/2/12 | -0.0016 | [-0.0071, 0.0024] | 0.6397 (Holm 1.000) | 19/2/15 | -0.0020 | [-0.0083, 0.0022] | 0.8986 (Holm 1.000) | 24/1/11 |
| lam_curve_k8 − sketch_surrogate_k8 | c | C4 | -0.0021 | [-0.0072, 0.0022] | 0.1964 (Holm 1.000) | 23/3/10 | -0.0015 | [-0.0032, -0.0001] | 0.0104 (Holm 0.208) | 27/2/7 | -0.0008 | [-0.0019, 0.0002] | 0.0432 (Holm 0.604) | 23/4/9 |
| lam_scalar_k8 − sketch_none_k8 | a |  | -0.0037 | [-0.0107, 0.0028] | 0.2288 | 23/2/11 | 0.0022 | [-0.0019, 0.0068] | 0.9661 | 19/2/15 | 0.0034 | [0.0012, 0.0064] | 0.0008 | 10/1/25 |
| lam_scalar_k8 − sketch_none_k8 | c |  | -0.0028 | [-0.0070, 0.0016] | 0.0385 | 25/1/10 | 0.0020 | [0.0002, 0.0048] | 0.1815 | 15/3/18 | 0.0014 | [6.6e-05, 0.0029] | 0.0665 | 15/1/20 |
| lam_scalar_k8 − sketch_surrogate_k8 | a |  | -0.0004 | [-0.0025, 0.0020] | 0.1674 | 21/6/9 | -0.0022 | [-0.0087, 0.0023] | 0.6705 | 24/2/10 | -0.0025 | [-0.0087, 0.0017] | 0.3927 | 25/2/9 |
| lam_scalar_k8 − sketch_surrogate_k8 | c |  | -0.0024 | [-0.0053, -0.0003] | 0.0268 | 24/1/11 | -0.0012 | [-0.0033, 0.0010] | 0.0120 | 28/3/5 | -0.0011 | [-0.0020, -0.0003] | 0.0019 | 27/3/6 |
| lam_cusum_k8 − sketch_none_k8 | a |  | -0.0056 | [-0.0132, 0.0012] | 0.1815 | 24/1/11 | -0.0008 | [-0.0043, 0.0036] | 0.1964 | 25/1/10 | 0.0010 | [0.0003, 0.0019] | 0.1415 | 17/0/19 |
| lam_cusum_k8 − sketch_none_k8 | c |  | -0.0002 | [-0.0055, 0.0053] | 0.7019 | 23/0/13 | 0.0021 | [-0.0001, 0.0051] | 0.3692 | 18/3/15 | -4.7e-05 | [-0.0008, 0.0006] | 0.6397 | 19/1/16 |
| lam_cusum_k8 − sketch_surrogate_k8 | a |  | -0.0023 | [-0.0068, 0.0010] | 0.3247 | 18/3/15 | -0.0052 | [-0.0085, -0.0023] | 0.0001 | 28/0/8 | -0.0048 | [-0.0105, -0.0012] | 0.0013 | 28/1/7 |
| lam_cusum_k8 − sketch_surrogate_k8 | c |  | 0.0002 | [-0.0007, 0.0013] | 0.6397 | 16/4/16 | -0.0012 | [-0.0036, 0.0011] | 0.0034 | 29/1/6 | -0.0025 | [-0.0040, -0.0013] | 0.0003 | 30/1/5 |
| lam_radius_k8 − sketch_none_k8 | a |  | -0.0072 | [-0.0163, 0.0003] | 0.0432 | 23/2/11 | 0.0018 | [-0.0015, 0.0065] | 0.9323 | 22/3/11 | 0.0019 | [0.0003, 0.0039] | 0.1084 | 14/2/20 |
| lam_radius_k8 − sketch_none_k8 | c |  | -0.0046 | [-0.0097, 0.0008] | 0.0342 | 26/2/8 | 0.0016 | [-0.0005, 0.0050] | 0.9323 | 20/0/16 | 1.4e-05 | [-0.0005, 0.0006] | 0.7019 | 18/2/16 |
| lam_radius_k8 − sketch_surrogate_k8 | a |  | -0.0040 | [-0.0088, -0.0002] | 0.2837 | 22/0/14 | -0.0026 | [-0.0053, -7.4e-05] | 0.0182 | 28/1/7 | -0.0039 | [-0.0086, -0.0007] | 0.0056 | 26/2/8 |
| lam_radius_k8 − sketch_surrogate_k8 | c |  | -0.0042 | [-0.0112, 0.0013] | 0.0665 | 25/1/10 | -0.0016 | [-0.0046, 0.0013] | 0.0104 | 31/0/5 | -0.0025 | [-0.0039, -0.0012] | 0.0001 | 33/1/2 |
| hyb_lam_curve − hybrid_none | a |  | -0.0132 | [-0.0221, -0.0058] | 0.0004 | 23/1/12 | -0.0019 | [-0.0066, 0.0024] | 0.5226 | 25/0/11 | -0.0009 | [-0.0024, 0.0005] | 0.8317 | 15/4/17 |
| hyb_lam_curve − hybrid_none | c |  | -0.0098 | [-0.0173, -0.0029] | 0.0010 | 25/0/11 | 0.0024 | [-0.0006, 0.0069] | 0.7987 | 22/0/14 | 0.0012 | [0.0001, 0.0025] | 0.2462 | 15/0/21 |
| hyb_lam_curve − hybrid_surrogate | a |  | -0.0017 | [-0.0056, 0.0026] | 0.2645 | 20/4/12 | -0.0044 | [-0.0070, -0.0024] | 0.0001 | 27/0/9 | -0.0065 | [-0.0123, -0.0021] | 0.0028 | 28/3/5 |
| hyb_lam_curve − hybrid_surrogate | c |  | -8.6e-05 | [-0.0033, 0.0032] | 0.7660 | 18/3/15 | -0.0005 | [-0.0036, 0.0033] | 0.2837 | 25/2/9 | -0.0011 | [-0.0024, -4.2e-05] | 0.0599 | 25/4/7 |

- Difference A − B over the 36 targets (negative: A lower). Tier a: strict paper RRMS; tier c: excess over floor (a difference of excesses equals the difference of raw RRMS). Per-target value = mean over the arm's seeds.
- 95% CI: geometry-clustered bootstrap (10,000 resamples of the 18 geometries, seed 0; seeds resampled within a target when an arm has 2 or more). p: two-sided Wilcoxon signed-rank on the 18 geometry-mean differences, unadjusted; rows with a pre-registered id (C1-C6, prereg_iter2.json) also give the Holm-adjusted p over the 36 pre-registered tests (stats_iter2/stats.json). Rows without an id are ablations, not pre-registered, and carry no multiplicity correction.
- W/T/L: wins/ties/losses of A over the 36 targets, tie when |diff| < max(1e-6, 1% of the larger value) (dev_compare.paired).

### DEV iteration 2 paired comparisons: surrogate term vs NGSpice-only control, same code

| A − B | tier | pre-registered id | B500 mean A − B | B500 clustered 95% CI | B500 Wilcoxon p (geometry) | B500 W/T/L | B2000 mean A − B | B2000 clustered 95% CI | B2000 Wilcoxon p (geometry) | B2000 W/T/L | B5000 mean A − B | B5000 clustered 95% CI | B5000 Wilcoxon p (geometry) | B5000 W/T/L |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sketch_surrogate_k8 − sketch_none_k8 | a | C1 | -0.0033 | [-0.0121, 0.0047] | 0.2645 (Holm 1.000) | 20/3/13 | 0.0044 | [-0.0005, 0.0107] | 0.1084 (Holm 1.000) | 16/0/20 | 0.0058 | [0.0020, 0.0117] | 0.0003 (Holm 0.010) | 7/1/28 |
| sketch_surrogate_k8 − sketch_none_k8 | c | C1 | -0.0004 | [-0.0054, 0.0050] | 0.7337 (Holm 1.000) | 22/0/14 | 0.0033 | [0.0010, 0.0061] | 0.0237 (Holm 0.450) | 10/2/24 | 0.0025 | [0.0010, 0.0042] | 0.0066 (Holm 0.151) | 10/1/25 |
| hybrid_surrogate − hybrid_none | a | C2 | -0.0116 | [-0.0215, -0.0036] | 0.0016 (Holm 0.043) | 23/1/12 | 0.0025 | [-0.0031, 0.0079] | 0.4683 (Holm 1.000) | 20/0/16 | 0.0056 | [0.0019, 0.0111] | 0.0028 (Holm 0.067) | 10/4/22 |
| hybrid_surrogate − hybrid_none | c | C2 | -0.0097 | [-0.0182, -0.0026] | 0.0237 (Holm 0.450) | 24/1/11 | 0.0029 | [0.0003, 0.0058] | 0.2288 (Holm 1.000) | 14/0/22 | 0.0023 | [0.0006, 0.0044] | 0.0304 (Holm 0.516) | 10/0/26 |
| anytime_sur_presearch_race − anytime_none_race | a |  | -0.0141 | [-0.0233, -0.0060] | 0.0023 | 28/0/8 | 0.0023 | [-0.0019, 0.0073] | 0.9661 | 17/3/16 | -0.0012 | [-0.0053, 0.0022] | 0.7987 | 17/4/15 |
| anytime_sur_presearch_race − anytime_none_race | c |  | -0.0041 | [-0.0103, 0.0031] | 0.1540 | 22/0/14 | 0.0035 | [0.0002, 0.0074] | 0.1415 | 13/0/23 | 0.0019 | [-0.0004, 0.0045] | 0.3038 | 16/0/20 |
| sketch_fullfd_surrogate − sketch_fullfd_none | a |  | — | – | – |  | — | – | – |  | 0.0050 | [0.0007, 0.0103] | 0.0342 | 15/1/20 |
| sketch_fullfd_surrogate − sketch_fullfd_none | c |  | — | – | – |  | — | – | – |  | 0.0027 | [0.0013, 0.0044] | 0.0139 | 12/0/24 |

- Difference A − B over the 36 targets (negative: A lower). Tier a: strict paper RRMS; tier c: excess over floor (a difference of excesses equals the difference of raw RRMS). Per-target value = mean over the arm's seeds.
- 95% CI: geometry-clustered bootstrap (10,000 resamples of the 18 geometries, seed 0; seeds resampled within a target when an arm has 2 or more). p: two-sided Wilcoxon signed-rank on the 18 geometry-mean differences, unadjusted; rows with a pre-registered id (C1-C6, prereg_iter2.json) also give the Holm-adjusted p over the 36 pre-registered tests (stats_iter2/stats.json). Rows without an id are ablations, not pre-registered, and carry no multiplicity correction.
- W/T/L: wins/ties/losses of A over the 36 targets, tie when |diff| < max(1e-6, 1% of the larger value) (dev_compare.paired).

### DEV iteration 2 paired comparisons: DFO-LS-style baseline vs the other arms

| A − B | tier | pre-registered id | B500 mean A − B | B500 clustered 95% CI | B500 Wilcoxon p (geometry) | B500 W/T/L | B2000 mean A − B | B2000 clustered 95% CI | B2000 Wilcoxon p (geometry) | B2000 W/T/L | B5000 mean A − B | B5000 clustered 95% CI | B5000 Wilcoxon p (geometry) | B5000 W/T/L |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dfols_style − sketch_none_k8 | a | C5 | 0.0212 | [0.0053, 0.0377] | 0.0016 (Holm 0.043) | 6/1/29 | 0.0099 | [0.0052, 0.0154] | 0.0001 (Holm 0.004) | 7/0/29 | 0.0052 | [0.0031, 0.0076] | 0.0004 (Holm 0.013) | 7/1/28 |
| dfols_style − sketch_none_k8 | c | C5 | 0.0181 | [0.0090, 0.0314] | 0.0000 (Holm 0.000) | 5/2/29 | 0.0074 | [0.0036, 0.0119] | 0.0000 (Holm 0.000) | 3/0/33 | 0.0040 | [0.0018, 0.0066] | 0.0016 (Holm 0.043) | 11/1/24 |
| dfols_style − sketch_surrogate_k8 | a |  | 0.0245 | [0.0143, 0.0364] | 0.0002 | 3/1/32 | 0.0056 | [-0.0001, 0.0111] | 0.0539 | 10/0/26 | -0.0006 | [-0.0068, 0.0042] | 0.3692 | 14/2/20 |
| dfols_style − sketch_surrogate_k8 | c |  | 0.0185 | [0.0100, 0.0288] | 0.0002 | 5/0/31 | 0.0041 | [0.0015, 0.0073] | 0.0077 | 11/0/25 | 0.0015 | [-0.0002, 0.0037] | 0.1964 | 17/1/18 |
| lam_curve_k8 − dfols_style | a | C6 | -0.0262 | [-0.0399, -0.0144] | 0.0003 (Holm 0.010) | 33/0/3 | -0.0072 | [-0.0127, -0.0011] | 0.0090 (Holm 0.197) | 26/0/10 | -0.0014 | [-0.0046, 0.0026] | 0.0304 (Holm 0.516) | 26/1/9 |
| lam_curve_k8 − dfols_style | c | C6 | -0.0206 | [-0.0343, -0.0112] | 0.0000 (Holm 0.000) | 30/0/6 | -0.0056 | [-0.0096, -0.0023] | 0.0010 (Holm 0.029) | 29/0/7 | -0.0023 | [-0.0045, -0.0006] | 0.0539 (Holm 0.700) | 22/1/13 |
| dfols_style − hybrid_none | a |  | -1.4e-05 | [-0.0094, 0.0087] | 0.6705 | 16/1/19 | 0.0030 | [-0.0021, 0.0089] | 0.5509 | 17/1/18 | 0.0018 | [-0.0009, 0.0041] | 0.0304 | 13/2/21 |
| dfols_style − hybrid_none | c |  | 0.0005 | [-0.0088, 0.0096] | 0.3465 | 17/1/18 | 0.0041 | [0.0014, 0.0077] | 0.0139 | 11/0/25 | 0.0021 | [-9.8e-05, 0.0046] | 0.1415 | 14/0/22 |
| dfols_style − cmaes_trf | a |  | 0.0014 | [-0.0184, 0.0167] | 0.3465 | 18/0/18 | -0.0040 | [-0.0151, 0.0053] | 0.6705 | 16/0/20 | -0.0033 | [-0.0080, 0.0012] | 0.5798 | 22/0/14 |
| dfols_style − cmaes_trf | c |  | 0.0018 | [-0.0095, 0.0122] | 0.5226 | 17/0/19 | -0.0028 | [-0.0106, 0.0038] | 0.8317 | 17/0/19 | -0.0089 | [-0.0190, -0.0008] | 0.2288 | 17/2/17 |
| dfols_style − dfols_style_1start | a |  | 0.0118 | [-0.0013, 0.0221] | 0.0104 | 6/0/30 | 0.0069 | [-0.0010, 0.0143] | 0.0385 | 8/0/28 | 0.0020 | [-0.0019, 0.0057] | 0.2659 | 7/5/24 |
| dfols_style − dfols_style_1start | c |  | 0.0077 | [-0.0016, 0.0180] | 0.0987 | 9/0/27 | 0.0012 | [-0.0039, 0.0063] | 0.5798 | 11/1/24 | 0.0007 | [-0.0038, 0.0045] | 0.2837 | 12/2/22 |

- Difference A − B over the 36 targets (negative: A lower). Tier a: strict paper RRMS; tier c: excess over floor (a difference of excesses equals the difference of raw RRMS). Per-target value = mean over the arm's seeds.
- 95% CI: geometry-clustered bootstrap (10,000 resamples of the 18 geometries, seed 0; seeds resampled within a target when an arm has 2 or more). p: two-sided Wilcoxon signed-rank on the 18 geometry-mean differences, unadjusted; rows with a pre-registered id (C1-C6, prereg_iter2.json) also give the Holm-adjusted p over the 36 pre-registered tests (stats_iter2/stats.json). Rows without an id are ablations, not pre-registered, and carry no multiplicity correction.
- W/T/L: wins/ties/losses of A over the 36 targets, tie when |diff| < max(1e-6, 1% of the larger value) (dev_compare.paired).

### Pre-registered λ decision (n2b): rules A1-A4 per evaluated arm

| family | arm | role | rule | tier, budget | control | mean arm − control | 95% CI | required CI upper | pass/fail |
|---|---|---|---|---|---|---:|---:|---:|---:|
| sketch | lam_curve_k8 | evaluated | A1 | tier a, B500 | sketch_surrogate_k8 | -0.0017 | [-0.0054, 0.0011] | ≤ 0.002 | pass |
| sketch | lam_curve_k8 | evaluated | A2 | tier a, B5000 | sketch_none_k8 | 0.0038 | [0.0016, 0.0068] | ≤ 0.0005 | fail |
| sketch | lam_curve_k8 | evaluated | A3 | tier c, B500 | sketch_none_k8 | -0.0025 | [-0.0068, 0.0023] | < 0 | fail |
| sketch | lam_curve_k8 | evaluated | A4 | tier c, B5000 | sketch_none_k8 | 0.0017 | [0.0005, 0.0030] | ≤ 0.001 | fail |
| sketch | lam_scalar_k8 | evaluated | A1 | tier a, B500 | sketch_surrogate_k8 | -0.0004 | [-0.0025, 0.0020] | ≤ 0.002 | pass |
| sketch | lam_scalar_k8 | evaluated | A2 | tier a, B5000 | sketch_none_k8 | 0.0034 | [0.0012, 0.0064] | ≤ 0.0005 | fail |
| sketch | lam_scalar_k8 | evaluated | A3 | tier c, B500 | sketch_none_k8 | -0.0028 | [-0.0070, 0.0016] | < 0 | fail |
| sketch | lam_scalar_k8 | evaluated | A4 | tier c, B5000 | sketch_none_k8 | 0.0014 | [6.6e-05, 0.0029] | ≤ 0.001 | fail |
| sketch | lam_cusum_k8 | evaluated | A1 | tier a, B500 | sketch_surrogate_k8 | -0.0023 | [-0.0068, 0.0010] | ≤ 0.002 | pass |
| sketch | lam_cusum_k8 | evaluated | A2 | tier a, B5000 | sketch_none_k8 | 0.0010 | [0.0003, 0.0019] | ≤ 0.0005 | fail |
| sketch | lam_cusum_k8 | evaluated | A3 | tier c, B500 | sketch_none_k8 | -0.0002 | [-0.0055, 0.0053] | < 0 | fail |
| sketch | lam_cusum_k8 | evaluated | A4 | tier c, B5000 | sketch_none_k8 | -4.7e-05 | [-0.0008, 0.0006] | ≤ 0.001 | pass |
| hybrid | hyb_lam_curve | evaluated | A1 | tier a, B500 | hybrid_surrogate | -0.0017 | [-0.0056, 0.0026] | ≤ 0.002 | fail |
| hybrid | hyb_lam_curve | evaluated | A2 | tier a, B5000 | hybrid_none | -0.0009 | [-0.0024, 0.0005] | ≤ 0.0005 | pass |
| hybrid | hyb_lam_curve | evaluated | A3 | tier c, B500 | hybrid_none | -0.0098 | [-0.0173, -0.0029] | < 0 | pass |
| hybrid | hyb_lam_curve | evaluated | A4 | tier c, B5000 | hybrid_none | 0.0012 | [0.0001, 0.0025] | ≤ 0.001 | fail |
| sketch | lam_radius_k8 | ablation | A1 | tier a, B500 | sketch_surrogate_k8 | -0.0040 | [-0.0088, -0.0002] | ≤ 0.002 | pass |
| sketch | lam_radius_k8 | ablation | A2 | tier a, B5000 | sketch_none_k8 | 0.0019 | [0.0003, 0.0039] | ≤ 0.0005 | fail |
| sketch | lam_radius_k8 | ablation | A3 | tier c, B500 | sketch_none_k8 | -0.0046 | [-0.0097, 0.0008] | < 0 | fail |
| sketch | lam_radius_k8 | ablation | A4 | tier c, B5000 | sketch_none_k8 | 1.4e-05 | [-0.0005, 0.0006] | ≤ 0.001 | pass |
| sketch | lam_scalar_k8 | ablation | A1 | tier a, B500 | sketch_surrogate_k8 | -0.0004 | [-0.0025, 0.0020] | ≤ 0.002 | pass |
| sketch | lam_scalar_k8 | ablation | A2 | tier a, B5000 | sketch_none_k8 | 0.0034 | [0.0012, 0.0064] | ≤ 0.0005 | fail |
| sketch | lam_scalar_k8 | ablation | A3 | tier c, B500 | sketch_none_k8 | -0.0028 | [-0.0070, 0.0016] | < 0 | fail |
| sketch | lam_scalar_k8 | ablation | A4 | tier c, B5000 | sketch_none_k8 | 0.0014 | [6.6e-05, 0.0029] | ≤ 0.001 | fail |
| sketch | lam_cusum_k8 | ablation | A1 | tier a, B500 | sketch_surrogate_k8 | -0.0023 | [-0.0068, 0.0010] | ≤ 0.002 | pass |
| sketch | lam_cusum_k8 | ablation | A2 | tier a, B5000 | sketch_none_k8 | 0.0010 | [0.0003, 0.0019] | ≤ 0.0005 | fail |
| sketch | lam_cusum_k8 | ablation | A3 | tier c, B500 | sketch_none_k8 | -0.0002 | [-0.0055, 0.0053] | < 0 | fail |
| sketch | lam_cusum_k8 | ablation | A4 | tier c, B5000 | sketch_none_k8 | -4.7e-05 | [-0.0008, 0.0006] | ≤ 0.001 | pass |

- Status: decided. Result: negative. Frozen candidate per family: sketch family: none; hybrid family: none. TEST run: none (negative result).
- Rule: an arm is frozen when the upper end of its 95% CI meets all of A1-A4; order lam_curve_k8, lam_scalar_k8, lam_cusum_k8 (sketch family) and hyb_lam_curve (hybrid family); lam_radius_k8 is reported, not selectable. CI: geometry_clustered (dev_stats.compare ci95_clustered).
- sketch family controls: surrogate = sketch_surrogate_k8, none = sketch_none_k8 (parity exact: iteration-1 controls); candidate before the validity rule: none.
- hybrid family controls: surrogate = hybrid_surrogate, none = hybrid_none (parity exact: iteration-1 controls); candidate before the validity rule: none.
- Hard checks: 0 FAIL rows of 1739; 11 WARN rows for batched-probe score gaps (see README).
- A1: tier a B500 vs λ = 1 control, CI upper ≤ +0.002; A2: tier a B5000 vs λ = 0 control, ≤ +0.0005; A3: tier c B500 vs λ = 0 control, < 0; A4: tier c B5000 vs λ = 0 control, ≤ +0.001.
