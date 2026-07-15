# Finite-difference polish ablation

Both columns are one fixed surrogate method across all 18 devices. Positive delta means FD polish reduced RRMS. No per-device best-of is used.

| device | surrogate raw | surrogate + FD | delta |
|---|---:|---:|---:|
| nmos_L0p15_W1p6 | 0.0349 | 0.0348 | 0.0002 |
| nmos_L0p19_W7 | 0.0438 | 0.0434 | 0.0004 |
| nmos_L0p25_W1p6 | 0.0517 | 0.0497 | 0.0020 |
| nmos_L1_W1p6 | 0.0666 | 0.0650 | 0.0015 |
| nmos_L1_W3 | 0.1371 | 0.1102 | 0.0269 |
| nmos_L8_W1p6 | 0.0803 | 0.0802 | 0.0000 |
| nmos_L20_W0p64 | 0.1153 | 0.1149 | 0.0004 |
| nmos_L100_W100 | 0.1326 | 0.1324 | 0.0001 |
| pmos_L0p35_W0p55 | 0.6643 | 0.6387 | 0.0256 |
| pmos_L0p35_W1p6 | 0.3456 | 0.3293 | 0.0164 |
| pmos_L0p35_W5 | 0.3388 | 0.3171 | 0.0217 |
| pmos_L0p5_W0p42 | 0.3620 | 0.3610 | 0.0010 |
| pmos_L0p5_W0p64 | 0.3404 | 0.3286 | 0.0118 |
| pmos_L2_W5 | 0.1904 | 0.1900 | 0.0004 |
| pmos_L4_W7 | 0.2240 | 0.2233 | 0.0007 |
| pmos_L8_W0p84 | 0.3764 | 0.3739 | 0.0025 |
| pmos_L8_W1p6 | 0.4782 | 0.4697 | 0.0085 |
| pmos_L8_W5 | 0.2607 | 0.2595 | 0.0011 |

Surrogate mean: 0.2357 -> 0.2290.
