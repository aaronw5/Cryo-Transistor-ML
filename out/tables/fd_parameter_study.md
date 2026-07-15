# Finite-difference parameter study

The same fixed curve set and +/-10% seven-parameter box are used for every row. `FD alone` starts exactly at the published card. The surrogate columns compare the same surrogate-search result before and after FD. No per-device method selection is used.

| device | published | FD alone | improvement | surrogate raw | same-winner + FD | paired improvement | production top-5 + FD | FD-alone nfev | surrogate-pair nfev |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nmos_L0p15_W1p6 | 0.0595 | 0.0595 | 0.0000 | 0.0349 | 0.0348 | 0.0001 | 0.0348 | 13 | 38 |
| nmos_L0p19_W7 | 0.0967 | 0.0967 | 0.0000 | 0.0438 | 0.0434 | 0.0003 | 0.0434 | 13 | 87 |
| nmos_L0p25_W1p6 | 0.0983 | 0.0983 | 0.0000 | 0.0517 | 0.0502 | 0.0014 | 0.0497 | 13 | 199 |
| nmos_L1_W1p6 | 0.1283 | 0.1283 | 0.0000 | 0.0666 | 0.0650 | 0.0015 | 0.0650 | 13 | 179 |
| nmos_L1_W3 | 0.1599 | 0.1599 | 0.0000 | 0.1371 | 0.1200 | 0.0171 | 0.1102 | 13 | 110 |
| nmos_L8_W1p6 | 0.1474 | 0.1474 | 0.0000 | 0.0803 | 0.0802 | 0.0000 | 0.0802 | 13 | 52 |
| nmos_L20_W0p64 | 0.1283 | 0.1283 | 0.0000 | 0.1153 | 0.1149 | 0.0004 | 0.1149 | 13 | 397 |
| nmos_L100_W100 | 0.1403 | 0.1403 | 0.0000 | 0.1326 | 0.1324 | 0.0001 | 0.1324 | 13 | 771 |
| pmos_L0p35_W0p55 | 0.7270 | 0.7270 | 0.0000 | 0.6643 | 0.6388 | 0.0256 | 0.6387 | 13 | 76 |
| pmos_L0p35_W1p6 | 0.3776 | 0.3776 | 0.0000 | 0.3456 | 0.3299 | 0.0157 | 0.3293 | 13 | 70 |
| pmos_L0p35_W5 | 0.3291 | 0.3291 | 0.0000 | 0.3388 | 0.3172 | 0.0216 | 0.3171 | 13 | 55 |
| pmos_L0p5_W0p42 | 0.4560 | 0.4560 | 0.0000 | 0.3620 | 0.3610 | 0.0010 | 0.3610 | 13 | 53 |
| pmos_L0p5_W0p64 | 0.3498 | 0.3498 | 0.0000 | 0.3404 | 0.3286 | 0.0118 | 0.3286 | 13 | 72 |
| pmos_L2_W5 | 0.1907 | 0.1907 | 0.0000 | 0.1904 | 0.1902 | 0.0002 | 0.1900 | 13 | 904 |
| pmos_L4_W7 | 0.2702 | 0.2702 | 0.0000 | 0.2240 | 0.2237 | 0.0003 | 0.2233 | 13 | 75 |
| pmos_L8_W0p84 | 0.4340 | 0.4340 | 0.0000 | 0.3764 | 0.3739 | 0.0025 | 0.3739 | 13 | 225 |
| pmos_L8_W1p6 | 0.5433 | 0.5433 | 0.0000 | 0.4782 | 0.4697 | 0.0085 | 0.4697 | 13 | 90 |
| pmos_L8_W5 | 0.3146 | 0.3146 | 0.0000 | 0.2607 | 0.2596 | 0.0011 | 0.2595 | 13 | 54 |

## Aggregate

- Published -> FD alone: 0.2751 -> 0.2751; improvement 0.0000; wins 0/18.
- Surrogate raw -> surrogate + FD: 0.2357 -> 0.2296; improvement 0.0061; wins 18/18.
- Production top-five surrogate + FD mean: 0.2290; improvement from the raw winner 0.0067.

## Parameter Movement

Mean absolute movement is expressed as a percent of the magnitude of each device's published parameter.

| parameter | FD alone | surrogate raw from published | paired FD movement | production FD movement |
|---|---:|---:|---:|---:|
| vth0 | 0.000% | 3.346% | 0.527% | 0.525% |
| u0 | 0.000% | 8.539% | 0.356% | 0.433% |
| nfactor | 0.000% | 7.274% | 0.398% | 2.205% |
| vsat | 0.000% | 5.497% | 0.506% | 0.946% |
| delta | 0.000% | 8.096% | 0.880% | 1.434% |
| rdsw | 0.000% | 8.677% | 0.060% | 1.649% |
| eta0 | 0.000% | 4.288% | 0.005% | 0.855% |
