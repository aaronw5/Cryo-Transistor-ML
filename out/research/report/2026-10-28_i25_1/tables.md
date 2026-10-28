### I25-1: RRMS, primary: paper RRMS, released-card curve set

| card | all-device | combined | all-device | combined | NGSpice calls / device |
|---|---:|---:|---:|---:|---:|
| stock SkyWater tt card at 77 K (no fit) | 3.4176 | 3.1077 |  |  | 0 |
| released 77 K card, multipliers = 1 (no fit) | 12.1010 | 10.9593 |  |  | 0 |
| paper card (released, tuned) | 0.2751 | 0.2595 |  |  | — |
| 7p card fit, no g (C0_refit) | 0.1546 | 0.1458 | 0.1546 | 0.1458 | 28,953 |
| 7p card fit + g v2 | 0.0783 | 0.0757 | 0.0823 | 0.0795 | 50,185 / 50,190 |
| 45p card fit, no g (C0_refit) | 0.1094 | 0.1026 | 0.1094 | 0.1026 | 77,696 |
| 45p card fit + g v2 | 0.0667 | 0.0635 | 0.0693 | 0.0661 | 106,986 / 106,979 |

- Task: the paper's 18 devices, card files, 7 parameters and RRMS; every released multiplier set to 1; 3 random starts per device, best of 3 by training RRMS; multi-round direct NGSpice fit (rounds 1-3 + FD).
- full data: g trained on all 18 records (lambda by the nested procedure on all 18), each card refit with g in the loop; in-sample RRMS.
- LODO: g trained on the other 17 devices (secondary); C0_refit has no g, so it is the same in both column groups.
- combined = mean of the nMOS and pMOS family means; all-device = mean over 18 devices.
- calls: Step A summed over the 3 starts + lambda-selection refits / 18 + own refit.

### I25-1: RRMS, primary: paper RRMS, released-card curve set

| card | all-device | combined | NGSpice calls / device |
|---|---:|---:|---:|
| stock SkyWater tt card at 77 K (no fit) | 3.4176 | 3.1077 | 0 |
| released 77 K card, multipliers = 1 (no fit) | 12.1010 | 10.9593 | 0 |
| paper card (released, tuned) | 0.2751 | 0.2595 | — |
| 7p card fit, no g (C0_refit) | 0.1546 | 0.1458 | 28,953 |
| 7p card fit + g v2 | 0.0783 | 0.0757 | 50,185 |
| 45p card fit, no g (C0_refit) | 0.1094 | 0.1026 | 77,696 |
| 45p card fit + g v2 | 0.0667 | 0.0635 | 106,986 |

- Task: the paper's 18 devices, card files, 7 parameters and RRMS; every released multiplier set to 1; 3 random starts per device, best of 3 by training RRMS; multi-round direct NGSpice fit (rounds 1-3 + FD).
- full data: g trained on all 18 records (lambda by the nested procedure on all 18), each card refit with g in the loop; in-sample RRMS.
- combined = mean of the nMOS and pMOS family means; all-device = mean over 18 devices.
- calls: Step A summed over the 3 starts + lambda-selection refits / 18 + own refit.

### I25-1: RRMS, secondary: paper RRMS, M-rule curve set (measured current only)

| card | all-device | combined | all-device | combined | NGSpice calls / device |
|---|---:|---:|---:|---:|---:|
| stock SkyWater tt card at 77 K (no fit) | 3.3236 | 3.0231 |  |  | 0 |
| released 77 K card, multipliers = 1 (no fit) | 13.1955 | 11.9445 |  |  | 0 |
| paper card (released, tuned) | 0.2877 | 0.2709 |  |  | — |
| 7p card fit, no g (C0_refit) | 0.1700 | 0.1596 | 0.1700 | 0.1596 | 28,953 |
| 7p card fit + g v2 | 0.0969 | 0.0924 | 0.1007 | 0.0960 | 50,185 / 50,190 |
| 45p card fit, no g (C0_refit) | 0.1278 | 0.1192 | 0.1278 | 0.1192 | 77,696 |
| 45p card fit + g v2 | 0.0867 | 0.0815 | 0.0892 | 0.0841 | 106,986 / 106,979 |

- Task: the paper's 18 devices, card files, 7 parameters and RRMS; every released multiplier set to 1; 3 random starts per device, best of 3 by training RRMS; multi-round direct NGSpice fit (rounds 1-3 + FD).
- full data: g trained on all 18 records (lambda by the nested procedure on all 18), each card refit with g in the loop; in-sample RRMS.
- LODO: g trained on the other 17 devices (secondary); C0_refit has no g, so it is the same in both column groups.
- combined = mean of the nMOS and pMOS family means; all-device = mean over 18 devices.
- calls: Step A summed over the 3 starts + lambda-selection refits / 18 + own refit.

### RRMS after every round (selected start of each device), new 7p and 45p chains vs the old 43p chain, state C, primary curve set

| stage | all-device | combined | all-device | combined | all-device | combined |
|---|---:|---:|---:|---:|---:|---:|
| round 1 | 0.1706 | 0.1602 | 0.1354 | 0.1273 | 0.1231 | 0.1161 |
| round 2 | 0.1546 | 0.1458 | 0.1170 | 0.1099 | 0.1151 | 0.1088 |
| round 3 | 0.1546 | 0.1458 | 0.1102 | 0.1034 | 0.1115 | 0.1055 |
| round 3 + FD | 0.1546 | 0.1458 | 0.1096 | 0.1028 | 0.1109 | 0.1049 |
| old 7p stage (bank + center start) | — | — | — | — | 0.1676 | 0.1575 |
| start card, state C | 0.1546 | 0.1458 | 0.1096 | 0.1028 | 0.1109 | 0.1049 |
| C0_refit | 0.1546 | 0.1458 | 0.1094 | 0.1026 | 0.1051 | 0.0984 |
| g full data | 0.0783 | 0.0757 | 0.0667 | 0.0635 | 0.0696 | 0.0660 |
| g LODO | 0.0823 | 0.0795 | 0.0693 | 0.0661 | 0.0739 | 0.0700 |

- Old chain: multiplier-free centers with the pmos_L0p35_W1p6 named-multiplier bug, round 1 seeded by the old 7p card; cards rescored in state C. Old C0_refit = I13-1 rows; old g full data = I16-1 in-sample rows; old g LODO = I15-2 selected rows.

### Spread over the 3 random starts per device after round 3 + FD: min / median / max training RRMS (selected start)

| device | 7p | 45p |
|---|---:|---:|
| nMOS L0.15 W1.6 | 0.0333 / 0.0333 / 0.0712 (s1) | 0.0240 / 0.0262 / 0.0805 (s1) |
| nMOS L0.19 W7 | 0.0308 / 0.0318 / 0.0318 (s2) | 0.0234 / 0.0243 / 0.0310 (s1) |
| nMOS L0.25 W1.6 | 0.0362 / 0.0362 / 0.0362 (s2) | 0.0243 / 0.0278 / 0.0356 (s0) |
| nMOS L1 W1.6 | 0.0582 / 0.0582 / 0.0582 (s1) | 0.0399 / 0.0446 / 0.0466 (s2) |
| nMOS L1 W3 | 0.1051 / 0.1051 / 0.1051 (s0) | 0.0517 / 0.0537 / 0.0551 (s1) |
| nMOS L8 W1.6 | 0.0462 / 0.0462 / 0.0462 (s1) | 0.0324 / 0.0397 / 0.0489 (s0) |
| nMOS L20 W0.64 | 0.0899 / 0.0899 / 0.0899 (s2) | 0.0519 / 0.0590 / 0.0593 (s0) |
| nMOS L100 W100 | 0.1314 / 0.1314 / 0.1314 (s0) | 0.0877 / 0.0883 / 0.1009 (s0) |
| pMOS L0.35 W0.55 | 0.4499 / 0.4499 / 1.1978 (s1) | 0.3826 / 0.4685 / 0.4997 (s0) |
| pMOS L0.35 W1.6 | 0.2327 / 0.2327 / 0.2327 (s1) | 0.1686 / 0.1960 / 0.1991 (s1) |
| pMOS L0.35 W5 | 0.1575 / inf / inf (s1) | 0.1250 / 0.1510 / 0.2511 (s1) |
| pMOS L0.5 W0.42 | 0.2339 / 0.2339 / 0.2342 (s1) | 0.1471 / 0.1901 / 0.1957 (s1) |
| pMOS L0.5 W0.64 | 0.2597 / 0.2597 / 0.2597 (s2) | 0.2003 / 0.2648 / 0.3564 (s1) |
| pMOS L2 W5 | 0.1573 / 0.1573 / 0.1687 (s2) | 0.0999 / 0.1082 / 0.1444 (s0) |
| pMOS L4 W7 | 0.1424 / 0.1424 / 0.1424 (s0) | 0.0929 / 0.0949 / 0.0955 (s0) |
| pMOS L8 W0.84 | 0.2303 / 0.2303 / 0.2303 (s0) | 0.1557 / 0.1733 / 0.1774 (s0) |
| pMOS L8 W1.6 | 0.2419 / 0.2746 / 0.2746 (s1) | 0.1663 / 0.1778 / 0.1886 (s1) |
| pMOS L8 W5 | 0.1465 / 0.1465 / 0.1467 (s1) | 0.0995 / 0.1026 / 0.1044 (s2) |
| all-device, start 0 / 1 / 2 (arm A) | non-finite (17/18 finite) / 0.1547 / non-finite (17/18 finite) |  |
| all-device, start 0 / 1 / 2 (arm B) |  | 0.1352 / 0.1181 / 0.1320 |

- Starts: local coordinates uniform in [-1, 1] (log-uniform in the box), seeds 20261027 + 1000 k + device index. Selection: lowest training RRMS.

### Geometric-mean RRMS ratio [95% device-bootstrap CI]; lower = first card better

| arm | comparison | GMR | wins | label | GMR | label |
|---|---|---:|---:|---:|---:|---:|
| 7 parameters | card + g vs card (C0_refit), full data | 0.519 [0.439, 0.611] | 18/18 | lower | 0.565 [0.480, 0.656] | lower |
| 7 parameters | card + g vs paper card, full data | 0.278 [0.227, 0.347] | 18/18 | lower | 0.313 [0.256, 0.385] | lower |
| 7 parameters | card (C0_refit) vs paper card | 0.536 [0.467, 0.613] | 18/18 | lower | 0.554 [0.480, 0.633] | lower |
| 7 parameters | card + g (LODO) vs card (C0_refit) | 0.549 [0.469, 0.641] | 18/18 | lower | 0.590 [0.502, 0.682] | lower |
| 7 parameters | card + g (LODO) vs paper card | 0.295 [0.243, 0.364] | 18/18 | lower | 0.327 [0.268, 0.401] | lower |
| screened set | card + g vs card (C0_refit), full data | 0.633 [0.553, 0.721] | 18/18 | lower | 0.686 [0.606, 0.765] | lower |
| screened set | card + g vs paper card, full data | 0.231 [0.198, 0.274] | 18/18 | lower | 0.269 [0.231, 0.314] | lower |
| screened set | card (C0_refit) vs paper card | 0.365 [0.318, 0.417] | 18/18 | lower | 0.392 [0.338, 0.452] | lower |
| screened set | card + g (LODO) vs card (C0_refit) | 0.668 [0.581, 0.768] | 18/18 | lower | 0.721 [0.638, 0.806] | lower |
| screened set | card + g (LODO) vs paper card | 0.244 [0.208, 0.289] | 18/18 | lower | 0.283 [0.244, 0.329] | lower |
|  | 7p + g vs screened set + g, full data | 1.206 [1.097, 1.343] | 2/18 | higher |  |  |
|  | screened set + g vs old 43p + g, full data | 0.996 [0.895, 1.106] | 9/18 | inconclusive |  |  |
|  | screened set + g vs old 43p + g, LODO | 0.980 [0.871, 1.102] | 10/18 | inconclusive |  |  |

- label: lower if CI upper < 1, higher if CI lower > 1, else inconclusive (pre-registered).

### Per-device RRMS, arm A (7 parameters), primary curve set

| device | multipliers 1, no fit | paper card | Step A card | card fit (C0_refit) | card + g, full data | card + g, LODO | lambda* (V) |
|---|---:|---:|---:|---:|---:|---:|---:|
| nMOS L0.15 W1.6 | 0.6898 | 0.0595 | 0.0333 | 0.0333 | 0.0200 | 0.0205 | 2 |
| nMOS L0.19 W7 | 0.3745 | 0.0967 | 0.0308 | 0.0308 | 0.0214 | 0.0217 | 5 |
| nMOS L0.25 W1.6 | 0.5038 | 0.0983 | 0.0362 | 0.0362 | 0.0217 | 0.0231 | 1 |
| nMOS L1 W1.6 | 0.6227 | 0.1283 | 0.0582 | 0.0582 | 0.0384 | 0.0411 | 5 |
| nMOS L1 W3 | 0.6041 | 0.1599 | 0.1051 | 0.1051 | 0.0970 | 0.1008 | 0 |
| nMOS L8 W1.6 | 0.7926 | 0.1474 | 0.0462 | 0.0462 | 0.0333 | 0.0342 | 1 |
| nMOS L20 W0.64 | 0.9583 | 0.1283 | 0.0899 | 0.0899 | 0.0712 | 0.0737 | 1 |
| nMOS L100 W100 | 0.9320 | 0.1403 | 0.1314 | 0.1314 | 0.1144 | 0.1169 | 5 |
| pMOS L0.35 W0.55 | 79.8543 | 0.7270 | 0.4499 | 0.4499 | 0.3217 | 0.3211 | 2 |
| pMOS L0.35 W1.6 | 8.1463 | 0.3776 | 0.2327 | 0.2327 | 0.0848 | 0.0896 | 5 |
| pMOS L0.35 W5 | 83.9591 | 0.3291 | 0.1575 | 0.1575 | 0.0573 | 0.0610 | 5 |
| pMOS L0.5 W0.42 | 2.5556 | 0.4560 | 0.2339 | 0.2339 | 0.0895 | 0.0963 | 10 |
| pMOS L0.5 W0.64 | 12.4955 | 0.3498 | 0.2597 | 0.2597 | 0.0956 | 0.0988 | 5 |
| pMOS L2 W5 | 4.3222 | 0.1907 | 0.1573 | 0.1573 | 0.0511 | 0.0530 | 10 |
| pMOS L4 W7 | 4.9599 | 0.2702 | 0.1424 | 0.1424 | 0.0721 | 0.0744 | 10 |
| pMOS L8 W0.84 | 5.7704 | 0.4340 | 0.2303 | 0.2303 | 0.0915 | 0.1111 | 0 |
| pMOS L8 W1.6 | 5.2050 | 0.5433 | 0.2419 | 0.2419 | 0.0749 | 0.0791 | 0 |
| pMOS L8 W5 | 5.0713 | 0.3146 | 0.1465 | 0.1465 | 0.0534 | 0.0645 | 0 |
| all-device mean | 12.1010 | 0.2751 | 0.1546 | 0.1546 | 0.0783 | 0.0823 | 5 (full) |
| combined | 10.9593 | 0.2595 | 0.1458 | 0.1458 | 0.0757 | 0.0795 |  |

- Every value is the NGSpice re-simulation of the stored card (g applied in Python).

### Per-device RRMS, arm B (screened set), primary curve set

| device | multipliers 1, no fit | paper card | Step A card | card fit (C0_refit) | card + g, full data | card + g, LODO | lambda* (V) |
|---|---:|---:|---:|---:|---:|---:|---:|
| nMOS L0.15 W1.6 | 0.6898 | 0.0595 | 0.0240 | 0.0237 | 0.0169 | 0.0179 | 2 |
| nMOS L0.19 W7 | 0.3745 | 0.0967 | 0.0234 | 0.0233 | 0.0196 | 0.0194 | 10 |
| nMOS L0.25 W1.6 | 0.5038 | 0.0983 | 0.0243 | 0.0243 | 0.0205 | 0.0226 | 10 |
| nMOS L1 W1.6 | 0.6227 | 0.1283 | 0.0399 | 0.0399 | 0.0311 | 0.0339 | 10 |
| nMOS L1 W3 | 0.6041 | 0.1599 | 0.0517 | 0.0516 | 0.0419 | 0.0444 | 10 |
| nMOS L8 W1.6 | 0.7926 | 0.1474 | 0.0324 | 0.0321 | 0.0272 | 0.0320 | 10 |
| nMOS L20 W0.64 | 0.9583 | 0.1283 | 0.0519 | 0.0510 | 0.0433 | 0.0477 | 10 |
| nMOS L100 W100 | 0.9320 | 0.1403 | 0.0877 | 0.0877 | 0.0794 | 0.0815 | 10 |
| pMOS L0.35 W0.55 | 79.8543 | 0.7270 | 0.3826 | 0.3825 | 0.3032 | 0.3049 | 2 |
| pMOS L0.35 W1.6 | 8.1463 | 0.3776 | 0.1686 | 0.1677 | 0.0849 | 0.0978 | 10 |
| pMOS L0.35 W5 | 83.9591 | 0.3291 | 0.1250 | 0.1249 | 0.0548 | 0.0579 | 5 |
| pMOS L0.5 W0.42 | 2.5556 | 0.4560 | 0.1471 | 0.1471 | 0.0671 | 0.0710 | 5 |
| pMOS L0.5 W0.64 | 12.4955 | 0.3498 | 0.2003 | 0.1999 | 0.0949 | 0.0972 | 2 |
| pMOS L2 W5 | 4.3222 | 0.1907 | 0.0999 | 0.0998 | 0.0395 | 0.0441 | 1 |
| pMOS L4 W7 | 4.9599 | 0.2702 | 0.0929 | 0.0929 | 0.0687 | 0.0708 | 10 |
| pMOS L8 W0.84 | 5.7704 | 0.4340 | 0.1557 | 0.1553 | 0.0686 | 0.0667 | 2 |
| pMOS L8 W1.6 | 5.2050 | 0.5433 | 0.1663 | 0.1663 | 0.0926 | 0.0884 | 5 |
| pMOS L8 W5 | 5.0713 | 0.3146 | 0.0995 | 0.0993 | 0.0456 | 0.0484 | 5 |
| all-device mean | 12.1010 | 0.2751 | 0.1096 | 0.1094 | 0.0667 | 0.0693 | 5 (full) |
| combined | 10.9593 | 0.2595 | 0.1028 | 0.1026 | 0.0635 | 0.0661 |  |

- Every value is the NGSpice re-simulation of the stored card (g applied in Python).

### Parameter screen around the multiplier-1 card (115 candidates, unchanged rule) vs the earlier 43 (screen around the released tuned card)

| item | value |
|---|---|
| screened set (around the multiplier-1 card) | 45: toxe, toxm, xj, ndep, nsd, dwg, rsh, vth0, k1, k2, k3, dvt0, dvt1, dvt0w, dvt1w, w0, k3b, vfb, lpe0, lpeb, vsat, ua, ub, rdsw, prwg, wr, u0, a0, a2, ags, b0, b1, voff, nfactor, cit, cdsc, eta0, dsub, pclm, pdiblc1, pdiblc2, drout, delta, alpha0, beta0 |
| in both | 42 |
| old 43 only | cdscd |
| new only | cdsc, dvt0w, pdiblc1 |
| Jaccard | 0.913 |
