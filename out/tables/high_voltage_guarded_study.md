# High-voltage-preserving selection study

RRMS is unchanged from the paper. This diagnostic protects the strongest included output and transfer curves, then selects the lowest official RRMS among feasible candidates in the 10,000-point NGSpice LHC. A protected curve may be at most `max(1.5 * paper curve RRMS, paper curve RRMS + 0.005)`.

| method | all-device mean | combined | nMOS | pMOS |
|---|---:|---:|---:|---:|
| paper card | 0.2751 | 0.2595 | 0.1198 | 0.3992 |
| foundation + FD | 0.2261 | 0.2114 | 0.0794 | 0.3434 |
| high-voltage guarded | 0.2327 | 0.2175 | 0.0806 | 0.3544 |

## Per Device

| device | feasible candidates | paper RRMS | foundation + FD | guarded RRMS | protected paper -> foundation -> guarded |
|---|---:|---:|---:|---:|---|
| nmos_L0p15_W1p6 | 1000 | 0.0595 | 0.0343 | 0.0358 | idvd@1.85: 0.029 -> 0.041 -> 0.039; idvg@1.85: 0.016 -> 0.015 -> 0.013 |
| nmos_L0p19_W7 | 699 | 0.0967 | 0.0428 | 0.0475 | idvd@1.85: 0.019 -> 0.035 -> 0.027; idvg@1.85: 0.019 -> 0.016 -> 0.014 |
| nmos_L0p25_W1p6 | 2110 | 0.0983 | 0.0499 | 0.0520 | idvd@1.85: 0.028 -> 0.035 -> 0.031; idvg@1.85: 0.049 -> 0.043 -> 0.044 |
| nmos_L1_W1p6 | 4173 | 0.1283 | 0.0635 | 0.0676 | idvd@1.85: 0.043 -> 0.052 -> 0.049; idvg@1.85: 0.051 -> 0.030 -> 0.027 |
| nmos_L1_W3 | 8907 | 0.1599 | 0.1102 | 0.1126 | idvd@1.85: 0.106 -> 0.124 -> 0.121; idvg@1.85: 0.105 -> 0.127 -> 0.124 |
| nmos_L8_W1p6 | 3906 | 0.1474 | 0.0806 | 0.0810 | idvd@1.85: 0.048 -> 0.055 -> 0.051; idvg@1.85: 0.086 -> 0.050 -> 0.050 |
| nmos_L20_W0p64 | 4233 | 0.1283 | 0.1201 | 0.1161 | idvd@1.85: 0.053 -> 0.055 -> 0.050; idvg@1.85: 0.065 -> 0.053 -> 0.046 |
| nmos_L100_W100 | 6261 | 0.1403 | 0.1335 | 0.1324 | idvd@1.85: 0.050 -> 0.045 -> 0.038; idvg@1.85: 0.166 -> 0.184 -> 0.182 |
| pmos_L0p35_W0p55 | 2310 | 0.7270 | 0.6307 | 0.6406 | idvd@1.85: 0.098 -> 0.170 -> 0.136; idvg@1.85: 0.131 -> 0.126 -> 0.101 |
| pmos_L0p35_W1p6 | 959 | 0.3776 | 0.3284 | 0.3348 | idvd@1.85: 0.048 -> 0.052 -> 0.068; idvg@1.85: 0.082 -> 0.063 -> 0.051 |
| pmos_L0p35_W5 | 2297 | 0.3291 | 0.2887 | 0.2882 | idvd@1.85: 0.110 -> 0.106 -> 0.099; idvg@1.85: 0.146 -> 0.097 -> 0.109 |
| pmos_L0p5_W0p42 | 1861 | 0.4560 | 0.3590 | 0.3888 | idvd@1.85: 0.045 -> 0.093 -> 0.066; idvg@1.85: 0.239 -> 0.228 -> 0.264 |
| pmos_L0p5_W0p64 | 1188 | 0.3498 | 0.3253 | 0.3282 | idvd@1.85: 0.052 -> 0.068 -> 0.063; idvg@1.85: 0.067 -> 0.044 -> 0.057 |
| pmos_L2_W5 | 255 | 0.1907 | 0.1758 | 0.1861 | idvd@1.85: 0.011 -> 0.047 -> 0.015; idvg@1.85: 0.056 -> 0.043 -> 0.050 |
| pmos_L4_W7 | 766 | 0.2702 | 0.2233 | 0.2378 | idvd@1.85: 0.033 -> 0.074 -> 0.046; idvg@1.85: 0.102 -> 0.086 -> 0.086 |
| pmos_L8_W0p84 | 909 | 0.4340 | 0.3740 | 0.3967 | idvd@1.85: 0.041 -> 0.103 -> 0.061; idvg@1.85: 0.120 -> 0.102 -> 0.109 |
| pmos_L8_W1p6 | 1367 | 0.5433 | 0.4697 | 0.4768 | idvd@1.85: 0.067 -> 0.084 -> 0.085; idvg@1.85: 0.079 -> 0.089 -> 0.087 |
| pmos_L8_W5 | 1235 | 0.3146 | 0.2592 | 0.2662 | idvd@1.85: 0.057 -> 0.079 -> 0.081; idvg@1.85: 0.093 -> 0.097 -> 0.094 |

This guard is a selection policy, not a replacement metric. The guarded cards are diagnostic and are not used by the canonical card export.
