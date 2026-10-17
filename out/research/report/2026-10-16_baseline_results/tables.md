# Tables, 2026-10-16 baseline_results

Generated 2026-10-16 21:49 by `scripts/research/report_figs.py --topic baseline_results --date 2026-10-16` from result files only. RRMS is the paper metric with frozen published-baseline curve inclusion, re-simulated in NGSpice.

### Measured 77 K data, 18 devices: paper RRMS by method, raw output and after NGSpice FD polish

| method | stage | all-device | combined | nMOS | pMOS | NGSpice calls/device (this run) | NGSpice calls/device (cumulative) | failed devices | all-device, valid devices only | all-device, penalized |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Reference: released card | published card | 0.2751 | 0.2595 | 0.1198 | 0.3992 | — | — | 0 | 0.2751 | 0.2751 |
| Reference: paper, 7 parameters | published card | 0.2284 | 0.2135 | 0.0788 | 0.3481 | — | — | 0 | 0.2284 | 0.2284 |
| Reference: paper, 43 parameters | published card | 0.2038 | 0.1905 | 0.0704 | 0.3105 | — | — | 0 | 0.2038 | 0.2038 |
| Pipeline 7, round 1 | method output | 0.2367 | 0.2251 | 0.1211 | 0.3291 | 10,009 | 10,009 | 0 | 0.2367 | 0.2367 |
| Pipeline 7, round 4 | method output | 0.1697 | 0.1602 | 0.0746 | 0.2457 | 10,008 | 40,034 | 0 | 0.1697 | 0.1697 |
| Pipeline 7, round 4 | + NGSpice FD | 0.1643 | 0.1548 | 0.0694 | 0.2403 | 270 | 40,304 | 0 | 0.1643 | 0.1643 |
| Pipeline 43, round 1 | method output | 0.2375 | 0.2281 | 0.1435 | 0.3128 | 10,009 | 10,009 | 1 | 0.2395 | 0.2920 |
| Pipeline 43, round 1 | + NGSpice FD | 0.1824 | 0.1730 | 0.0881 | 0.2579 | 356 | 10,364 | 1 | 0.1853 | 0.2373 |
| Pipeline 43, round 4 | method output | 0.1513 | 0.1427 | 0.0652 | 0.2201 | 10,009 | 40,035 | 1 | 0.1586 | 0.6500 |
| Pipeline 43, round 4 | + NGSpice FD | 0.1465 | 0.1380 | 0.0617 | 0.2143 | 382 | 40,417 | 1 | 0.1539 | 0.6455 |
| NGSpice swap 7, round 1 | method output | 0.1887 | 0.1767 | 0.0683 | 0.2850 | 1,911 | 11,911 | 0 | 0.1887 | 0.1887 |
| NGSpice swap 43, round 1 | method output | 0.1125 | 0.1053 | 0.0410 | 0.1696 | 19,854 | 29,854 | 0 | 0.1125 | 0.1125 |
| NGSpice swap 43, round 1 | + NGSpice FD | 0.1106 | 0.1036 | 0.0407 | 0.1664 | 260 | 30,114 | 0 | 0.1106 | 0.1106 |
| Direct NGSpice 7, round 1 | method output | 0.1676 | 0.1575 | 0.0665 | 0.2484 | 3,283 | 13,283 | 0 | 0.1676 | 0.1676 |
| Direct NGSpice 7, round 1 | + NGSpice FD | 0.1670 | 0.1570 | 0.0665 | 0.2475 | 187 | 13,470 | 0 | 0.1670 | 0.1670 |
| Direct NGSpice 43, round 1, cold | method output | 0.1269 | 0.1186 | 0.0444 | 0.1929 | 21,882 | 21,882 | 0 | 0.1269 | 0.1269 |
| Direct NGSpice 43, round 1, seeded from 7 | method output | 0.1201 | 0.1126 | 0.0456 | 0.1797 | 21,213 | 34,497 | 0 | 0.1201 | 0.1201 |
| Direct NGSpice 43, round 1, seeded from 7 | + NGSpice FD | 0.1191 | 0.1117 | 0.0453 | 0.1781 | 319 | 34,815 | 0 | 0.1191 | 0.1191 |
| Direct NGSpice 43, round 3 | method output | 0.1049 | 0.0981 | 0.0366 | 0.1596 | 8,405 | 57,188 | 0 | 0.1049 | 0.1049 |
| Direct NGSpice 43, round 3 | + NGSpice FD | 0.1044 | 0.0976 | 0.0363 | 0.1588 | 314 | 57,502 | 0 | 0.1044 | 0.1044 |

- all-device = mean over 18 devices; combined = mean of the nMOS (8) and pMOS (10) family means.
- NGSpice calls per device: 'this run' is the run's own count (pipeline: bank simulations of that round plus NGSpice validation of candidates; +FD rows: FD polish only). 'cumulative' adds every bank used for starts or training and every earlier round or seed run named in the summary/spec provenance.
- Reference rows: all-device values are the paper's reported values; combined/nMOS/pMOS from out/research/reference_scores.json.
- failed dev. = devices with an included (frozen-inclusion) metric curve that failed to simulate; their stored RRMS averages only the surviving curves, so it is not a valid paper RRMS (upstream rrmsCalc.py gives NaN). The all-device/combined/family columns are the stored values; valid-only = all-device mean over the valid devices; penalized = every device, failed curves at RRMS 10 (ngspice_fit.PENALTY; penalized convention, not the paper metric). Source: cryoml.strict_score.
- Pipeline 43, round 1 (method output): 1 failed device(s): nmos_L0p15_W1p6 (1 included curve(s) non-finite); valid-only all-device 0.2395 over 17 devices, combined 0.2238; penalized all-device 0.2920.
- Pipeline 43, round 1 (+ NGSpice FD): 1 failed device(s): nmos_L0p15_W1p6 (1 included curve(s) non-finite); valid-only all-device 0.1853 over 17 devices, combined 0.1698; penalized all-device 0.2373.
- Pipeline 43, round 4 (method output): 1 failed device(s): nmos_L8_W1p6 (9 included curve(s) non-finite); valid-only all-device 0.1586 over 17 devices, combined 0.1454; penalized all-device 0.6500.
- Pipeline 43, round 4 (+ NGSpice FD): 1 failed device(s): nmos_L8_W1p6 (9 included curve(s) non-finite); valid-only all-device 0.1539 over 17 devices, combined 0.1410; penalized all-device 0.6455.
- Pipeline 43, round 1: the FD run's re-score of its start differs from the method's own score on nmos_L0p15_W1p6 0.2035 vs 1.1832 (own vs FD start); the method-output row uses the method's own score. A large FD-start value is the penalized start objective of a card with failed included curves (see failed dev.).
- Pipeline 43, round 4: the FD run's re-score of its start differs from the method's own score on nmos_L8_W1p6 0.0263 vs 9.0026 (own vs FD start); the method-output row uses the method's own score. A large FD-start value is the penalized start objective of a card with failed included curves (see failed dev.).

### Measured 77 K data, 18 devices: paper RRMS by round

| series | round | all-device | combined | nMOS | pMOS | NGSpice calls/device (this run) | NGSpice calls/device (cumulative) | failed devices | all-device, valid devices only | all-device, penalized |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pipeline 7 | 1 | 0.2367 | 0.2251 | 0.1211 | 0.3291 | 10,009 | 10,009 | 0 | 0.2367 | 0.2367 |
| Pipeline 7 | 2 | 0.1921 | 0.1820 | 0.0917 | 0.2723 | 10,009 | 20,017 | 0 | 0.1921 | 0.1921 |
| Pipeline 7 | 3 | 0.1758 | 0.1660 | 0.0774 | 0.2545 | 10,008 | 30,026 | 0 | 0.1758 | 0.1758 |
| Pipeline 7 | 4 | 0.1697 | 0.1602 | 0.0746 | 0.2457 | 10,008 | 40,034 | 0 | 0.1697 | 0.1697 |
| Pipeline 43 | 1 | 0.2375 | 0.2281 | 0.1435 | 0.3128 | 10,009 | 10,009 | 1 | 0.2395 | 0.2920 |
| Pipeline 43 | 2 | 0.1965 | 0.1875 | 0.1058 | 0.2691 | 10,009 | 20,018 | 0 | 0.1965 | 0.1965 |
| Pipeline 43 | 3 | 0.1716 | 0.1631 | 0.0865 | 0.2396 | 10,009 | 30,026 | 0 | 0.1716 | 0.1716 |
| Pipeline 43 | 4 | 0.1513 | 0.1427 | 0.0652 | 0.2201 | 10,009 | 40,035 | 1 | 0.1586 | 0.6500 |
| Direct NGSpice 43 | 1 | 0.1201 | 0.1126 | 0.0456 | 0.1797 | 21,213 | 34,497 | 0 | 0.1201 | 0.1201 |
| Direct NGSpice 43 | 2 | 0.1094 | 0.1024 | 0.0393 | 0.1655 | 14,287 | 48,784 | 0 | 0.1094 | 0.1094 |
| Direct NGSpice 43 | 3 | 0.1049 | 0.0981 | 0.0366 | 0.1596 | 8,405 | 57,188 | 0 | 0.1049 | 0.1049 |
| NGSpice swap 7 | 1 | 0.1887 | 0.1767 | 0.0683 | 0.2850 | 1,911 | 11,911 | 0 | 0.1887 | 0.1887 |
| NGSpice swap 43 | 1 | 0.1125 | 0.1053 | 0.0410 | 0.1696 | 19,854 | 29,854 | 0 | 0.1125 | 0.1125 |

- all-device = mean over 18 devices; combined = mean of the nMOS (8) and pMOS (10) family means.
- NGSpice calls per device: 'this run' is the run's own count (pipeline: bank simulations of that round plus NGSpice validation of candidates). 'cumulative' adds every bank used for starts or training and every earlier round or seed run named in the summary/spec provenance.
- Raw method output; no FD polish in this table.
- failed dev. = devices with an included (frozen-inclusion) metric curve that failed to simulate; their stored RRMS averages only the surviving curves, so it is not a valid paper RRMS (upstream rrmsCalc.py gives NaN). The all-device/combined/family columns are the stored values; valid-only = all-device mean over the valid devices; penalized = every device, failed curves at RRMS 10 (ngspice_fit.PENALTY; penalized convention, not the paper metric). Source: cryoml.strict_score.
- Pipeline 43 round 1: 1 failed device(s): nmos_L0p15_W1p6 (1 included curve(s) non-finite); valid-only all-device 0.2395 over 17 devices, combined 0.2238; penalized all-device 0.2920.
- Pipeline 43 round 4: 1 failed device(s): nmos_L8_W1p6 (9 included curve(s) non-finite); valid-only all-device 0.1586 over 17 devices, combined 0.1454; penalized all-device 0.6500.
- Direct NGSpice 43 rounds are the runs full_r1_direct_seeded -> full_r2_direct -> full_r3_direct (each spec's source_run names the previous round).

### Simulated temperature benchmark: paper RRMS vs simulated truth, observed (77/150/225/300 K) and held-out (110/190/260 K) temperatures

| method | stage | sets | all-device observed | all-device held-out | combined observed | combined held-out | nMOS observed | nMOS held-out | pMOS observed | pMOS held-out | NGSpice calls/device (this run) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct NGSpice (CMA-ES + TRF) | method output | 3 | 0.0129 | 0.0362 | 0.0125 | 0.0343 | 0.0082 | 0.0176 | 0.0167 | 0.0510 | 1,512 |
| Direct NGSpice (CMA-ES + TRF) | + NGSpice FD | 3 | 0.0015 | 0.0117 | 0.0015 | 0.0114 | 0.0012 | 0.0080 | 0.0018 | 0.0147 | 1,752 |
| Pipeline search on NGSpice | method output | 3 | 0.0005 | 0.0021 | 0.0005 | 0.0020 | 0.0005 | 0.0011 | 0.0004 | 0.0029 | 2,112 |
| Pipeline search on NGSpice | + NGSpice FD | 3 | 1.2e-05 | 0.0004 | 1.1e-05 | 0.0004 | 2.4e-06 | 0.0002 | 2.0e-05 | 0.0006 | 2,272 |
| Pipeline search on paper MLP surrogate | method output | 3 | 0.0051 | 0.0081 | 0.0050 | 0.0078 | 0.0043 | 0.0057 | 0.0057 | 0.0099 | 9 |
| Pipeline search on paper MLP surrogate | + NGSpice FD | 3 | 1.5e-05 | 0.0007 | 1.3e-05 | 0.0007 | 1.4e-06 | 0.0001 | 2.5e-05 | 0.0012 | 198 |
| Pipeline search on DeepONet surrogate | method output | 3 | 0.0076 | 0.0120 | 0.0074 | 0.0117 | 0.0059 | 0.0083 | 0.0090 | 0.0150 | 9 |
| Pipeline search on DeepONet surrogate | + NGSpice FD | 3 | 6.8e-05 | 0.0022 | 7.0e-05 | 0.0021 | 8.8e-05 | 0.0015 | 5.3e-05 | 0.0027 | 208 |

- Means over simulated sets of 18 devices. all-device = mean over 18 devices; combined = mean of the nMOS (8) and pMOS (10) family means. Values below 0.0001 are shown in scientific notation.
- Failed curves: temp_fit scores a non-finite included curve at RRMS 10 (penalized convention), so no curve is silently dropped here; 0 device-temperature value(s) reach the penalty floor 10 / n_curves.
- NGSpice calls per device: method output = calls before FD (search and validation); +FD rows = total including the FD polish.
- All methods draw bank starts from the temperature bank (10000 cards/device, each simulated at 7 temperatures; data/processed/temp_bank/v1/meta.json); the surrogate is trained on it. Calls exclude the bank.

### Simulated temperature benchmark: all-device RRMS per temperature (* = held out)

| method | stage | 77 K | 110 K* | 150 K | 190 K* | 225 K | 260 K* | 300 K |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Direct NGSpice (CMA-ES + TRF) | method output | 0.0180 | 0.0759 | 0.0184 | 0.0170 | 0.0126 | 0.0157 | 0.0028 |
| Direct NGSpice (CMA-ES + TRF) | + NGSpice FD | 0.0011 | 0.0299 | 0.0032 | 0.0035 | 0.0014 | 0.0018 | 0.0004 |
| Pipeline search on NGSpice | method output | 0.0004 | 0.0045 | 0.0006 | 0.0007 | 0.0006 | 0.0011 | 0.0003 |
| Pipeline search on NGSpice | + NGSpice FD | 6.8e-06 | 0.0010 | 2.8e-05 | 2.9e-05 | 1.2e-05 | 0.0002 | 8.5e-07 |
| Pipeline search on paper MLP surrogate | method output | 0.0092 | 0.0150 | 0.0057 | 0.0049 | 0.0041 | 0.0043 | 0.0014 |
| Pipeline search on paper MLP surrogate | + NGSpice FD | 6.3e-06 | 0.0019 | 3.3e-05 | 4.1e-05 | 1.8e-05 | 0.0002 | 1.2e-06 |
| Pipeline search on DeepONet surrogate | method output | 0.0139 | 0.0218 | 0.0086 | 0.0070 | 0.0062 | 0.0073 | 0.0018 |
| Pipeline search on DeepONet surrogate | + NGSpice FD | 4.7e-05 | 0.0062 | 0.0001 | 0.0001 | 6.4e-05 | 0.0002 | 3.2e-05 |

- Mean over simulated sets of 18 devices; values below 0.0001 in scientific notation.
