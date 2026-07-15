# Direct MLP finite-difference ablation

Every row starts from the one-pass direct MLP prediction. FD uses fresh NGSpice evaluations of the unchanged paper RRMS residual against measured curves. No initializer selection is performed.

| device | direct MLP | direct MLP + FD | improvement | accepted | nfev | objective evals | runtime (s) |
|---|---:|---:|---:|:---:|---:|---:|---:|
| nmos_L0p15_W1p6 | 0.1104 | 0.0370 | 0.0735 | yes | 41 | 300 | 25.4 |
| nmos_L0p19_W7 | 0.1756 | 0.0613 | 0.1142 | yes | 25 | 102 | 8.9 |
| nmos_L0p25_W1p6 | 0.0989 | 0.0540 | 0.0449 | yes | 67 | 382 | 32.9 |
| nmos_L1_W1p6 | 0.1222 | 0.0693 | 0.0529 | yes | 26 | 96 | 8.8 |
| nmos_L1_W3 | 0.1402 | 0.1153 | 0.0250 | yes | 19 | 54 | 4.9 |
| nmos_L8_W1p6 | 0.1224 | 0.1152 | 0.0072 | yes | 19 | 61 | 5.4 |
| nmos_L20_W0p64 | 0.1225 | 0.1168 | 0.0057 | yes | 120 | 855 | 73.5 |
| nmos_L100_W100 | 0.1347 | 0.1325 | 0.0022 | yes | 120 | 862 | 75.5 |
| pmos_L0p35_W0p55 | 0.7379 | 0.6308 | 0.1071 | yes | 26 | 124 | 11.3 |
| pmos_L0p35_W1p6 | 0.3796 | 0.3285 | 0.0512 | yes | 72 | 415 | 35.4 |
| pmos_L0p35_W5 | 0.3210 | 0.2909 | 0.0301 | yes | 22 | 92 | 7.0 |
| pmos_L0p5_W0p42 | 0.4373 | 0.3594 | 0.0779 | yes | 20 | 62 | 4.3 |
| pmos_L0p5_W0p64 | 0.3898 | 0.3667 | 0.0231 | yes | 52 | 297 | 20.6 |
| pmos_L2_W5 | 0.2303 | 0.1864 | 0.0439 | yes | 43 | 218 | 15.1 |
| pmos_L4_W7 | 0.2424 | 0.2233 | 0.0191 | yes | 36 | 183 | 12.6 |
| pmos_L8_W0p84 | 0.3797 | 0.3741 | 0.0056 | yes | 42 | 217 | 14.9 |
| pmos_L8_W1p6 | 0.4820 | 0.4697 | 0.0123 | yes | 19 | 96 | 6.6 |
| pmos_L8_W5 | 0.2736 | 0.2593 | 0.0144 | yes | 33 | 152 | 10.4 |

## Aggregate

- All-device mean: 0.2723 -> 0.2328 (improvement 0.0395).
- Improved devices: 18/18.
- Total runtime: 373.5 s.
