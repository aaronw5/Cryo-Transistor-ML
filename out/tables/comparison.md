# Paper-exact comparison

All NGSpice columns use the corrected-repository deck convention, native geometry bin, and paper companion notebook RRMS. NGSpice does not reproduce the paper-reported mean, so extracted cards are compared only with paper cards in the identical NGSpice chain.

| reference / method | mean RRMS |
|---|---:|
| paper reported | 0.279 |
| paper parameters in NGSpice | 0.692 |
| ml | 0.544 |

| device | paper reported | paper params in NGSpice | ml |
|---|---:|---:|---:|
| nmos_L0p15_W1p6 | 0.059 | 0.060 | 0.042 |
| nmos_L0p19_W7 | 0.097 | 0.488 | 0.430 |
| nmos_L0p25_W1p6 | 0.098 | 0.973 | 0.913 |
| nmos_L1_W1p6 | 0.128 | 0.594 | 0.515 |
| nmos_L1_W3 | 0.160 | 0.712 | 0.623 |
| nmos_L8_W1p6 | 0.147 | 0.144 | 0.058 |
| nmos_L20_W0p64 | 0.130 | 0.231 | 0.171 |
| nmos_L100_W100 | 0.142 | 0.288 | 0.280 |
| pmos_L0p35_W0p55 | 0.701 | 0.931 | 0.569 |
| pmos_L0p35_W1p6 | 0.374 | 0.869 | 0.754 |
| pmos_L0p35_W5 | 0.324 | 0.447 | 0.254 |
| pmos_L0p5_W0p42 | 0.465 | 1.098 | 1.181 |
| pmos_L0p5_W0p64 | 0.322 | 0.818 | 0.484 |
| pmos_L2_W5 | 0.207 | 0.976 | 0.963 |
| pmos_L4_W7 | 0.281 | 0.735 | 0.664 |
| pmos_L8_W0p84 | 0.480 | 0.805 | 0.576 |
| pmos_L8_W1p6 | 0.515 | 0.828 | 0.571 |
| pmos_L8_W5 | 0.388 | 1.458 | 0.739 |
