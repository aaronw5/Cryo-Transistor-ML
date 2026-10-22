# Tables, 2026-10-22 i13_1

Generated 2026-10-22 00:38 by `PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i13_1.py --date 2026-10-22`.

### I13-1 headline all-18 table: stored (state L) vs rerun in state C

| arm | old all / comb / median | new all / comb / median | new - old (all / comb) | GMR vs C0_refit, old | GMR vs C0_refit, new |
|---|---:|---:|---:|---:|---:|
| published 77 K card | 0.2751 / 0.2595 / 0.2304 | 0.2751 / 0.2595 / 0.2304 | +0.0000 / +0.0000 | — | — |
| start (direct43_r3_fd, not refit) | 0.1044 / 0.0976 / 0.0823 | 0.1109 / 0.1049 / 0.0823 | +0.0065 / +0.0073 | — | — |
| C0_refit (no correction) | 0.1043 / 0.0975 / 0.0820 | 0.1051 / 0.0984 / 0.0820 | +0.0008 / +0.0009 | — | — |
| T1_inloop (per-tag gains) | 0.0744 / 0.0703 / 0.0672 | 0.0753 / 0.0713 / 0.0680 | +0.0009 / +0.0010 | 0.775 [0.695, 0.865], 16/18 | 0.776 [0.692, 0.875], 17/18 |
| M_bias v1 (I3-2 recipe) | 0.0732 / 0.0691 / 0.0643 | 0.0735 / 0.0695 / 0.0637 | +0.0003 / +0.0004 | 0.755 [0.673, 0.852], 16/18 | 0.745 [0.660, 0.850], 17/18 |
| M_bias v2 (I12-1 recipe) | 0.0774 / 0.0732 / 0.0534 | 0.0772 / 0.0732 / 0.0544 | -0.0002 / +0.0001 | 0.787 [0.703, 0.884], 16/18 | 0.785 [0.693, 0.896], 16/18 |

- old: stored rows (state L: first curve of the batch deck with A2 = 1, later curves with the card a2). new: state C (b4check.c clamp written into the card, every curve in the fresh-process state); C0_refit / T1_inloop / M_bias v1: I3-2 all18 recipe rerun; M_bias v2: I12-1 LODO recipe rerun; start: the direct43_r3_fd cards rescored (not refit); published: unchanged (no bin has a2 > 1).
- strict paper RRMS, curve inclusion frozen to the published card; combined = mean of the nMOS (8) and pMOS (10) family means; all-device = mean over 18.
- GMR(M_bias v2, M_bias v1) old 1.042 [0.967, 1.111], new 1.053 [0.966, 1.137] (I12-1 A2 margin 1.05). GMR: device bootstrap, 10,000 resamples, 95% percentile CI; < 1: first arm lower.


### I13-1 audit: stored cards rescored per clamp state (all-device / combined)

| arm | devices with a2 > 1 | stored | state L | state C | state U |
|---|---:|---:|---:|---:|---:|
| published 77 K card | 0 | 0.2751 / 0.2595 | 0.2751 / 0.2595 | 0.2751 / 0.2595 | 0.2751 / 0.2595 |
| start (direct43_r3_fd, not refit) | 5 | 0.1044 / 0.0976 | 0.1044 / 0.0976 | 0.1109 / 0.1049 | 0.1045 / 0.0977 |
| C0_refit (no correction) | 5 | 0.1043 / 0.0975 | 0.1043 / 0.0975 | 0.1108 / 0.1048 | 0.1044 / 0.0976 |
| T1_inloop (per-tag gains) | 5 | 0.0744 / 0.0703 | 0.0744 / 0.0703 | 0.0785 / 0.0749 | 0.0745 / 0.0703 |
| M_bias v1 (I3-2 recipe) | 5 | 0.0732 / 0.0691 | 0.0732 / 0.0691 | 0.0775 / 0.0740 | 0.0732 / 0.0692 |
| M_bias v2 (I12-1 recipe) | 5 | 0.0774 / 0.0732 | 0.0774 / 0.0732 | 0.0821 / 0.0785 | 0.0774 / 0.0732 |

- Stored cards of each arm (no refit), rescored in the three clamp states; T1 fold tables refitted from the stored records (deterministic); stored fold corrections for M_bias v1 / v2.
- V1: L reproduces every stored row, max |diff| 0.0e+00. V2: 83 cards with a2 in [0.01, 1]: C = L = U, max |diff| 0.0e+00. V3: 18 rows per arm and state.
- Other b4check.c clamps on size-dependent parameters (ckappas/ckappad, prwg, rdsw, rds0, rdswmin, xn, lc): 0 stored cards hit one. Clamp messages after writing the A2 clamp: 0.
- States: L legacy batch deck; C clamp written into the card; U warm-up op (every curve with the card a2).


### I13-1 per-device RRMS on the five devices with a2 > 1

| device | a2 (start) | arm | stored card, L | stored card, C | stored card, U | new row (C) | new - old |
|---|---:|---|---:|---:|---:|---:|---:|
| nmos_L0p19_W7 | 1.877 | start | 0.0155 | 0.0699 | 0.0156 | 0.0699 | +0.0544 |
|  |  | C0_refit | 0.0153 | 0.0691 | 0.0153 | 0.0215 | +0.0063 |
|  |  | T1_inloop | 0.0183 | 0.0544 | 0.0185 | 0.0190 | +0.0007 |
|  |  | M_bias_v1 | 0.0193 | 0.0541 | 0.0195 | 0.0176 | -0.0017 |
|  |  | M_bias_v2 | 0.0190 | 0.0576 | 0.0192 | 0.0203 | +0.0013 |
| nmos_L0p25_W1p6 | 1.045 | start | 0.0188 | 0.0301 | 0.0192 | 0.0301 | +0.0113 |
|  |  | C0_refit | 0.0188 | 0.0300 | 0.0192 | 0.0194 | +0.0006 |
|  |  | T1_inloop | 0.0176 | 0.0187 | 0.0176 | 0.0189 | +0.0013 |
|  |  | M_bias_v1 | 0.0171 | 0.0173 | 0.0171 | 0.0186 | +0.0015 |
|  |  | M_bias_v2 | 0.0182 | 0.0252 | 0.0184 | 0.0197 | +0.0015 |
| nmos_L20_W0p64 | 2.002 | start | 0.0491 | 0.0797 | 0.0500 | 0.0797 | +0.0306 |
|  |  | C0_refit | 0.0489 | 0.0793 | 0.0499 | 0.0561 | +0.0073 |
|  |  | T1_inloop | 0.0420 | 0.0718 | 0.0427 | 0.0478 | +0.0058 |
|  |  | M_bias_v1 | 0.0415 | 0.0731 | 0.0424 | 0.0465 | +0.0050 |
|  |  | M_bias_v2 | 0.0477 | 0.0795 | 0.0488 | 0.0542 | +0.0065 |
| nmos_L8_W1p6 | 1.062 | start | 0.0257 | 0.0463 | 0.0257 | 0.0463 | +0.0207 |
|  |  | C0_refit | 0.0255 | 0.0472 | 0.0255 | 0.0260 | +0.0006 |
|  |  | T1_inloop | 0.0194 | 0.0263 | 0.0193 | 0.0190 | -0.0003 |
|  |  | M_bias_v1 | 0.0187 | 0.0303 | 0.0188 | 0.0174 | -0.0013 |
|  |  | M_bias_v2 | 0.0220 | 0.0305 | 0.0220 | 0.0248 | +0.0028 |
| pmos_L0p35_W1p6 | 1.253 | start | 0.1876 | 0.1876 | 0.1876 | 0.1876 | +0.0000 |
|  |  | C0_refit | 0.1876 | 0.1876 | 0.1876 | 0.1876 | +0.0000 |
|  |  | T1_inloop | 0.1043 | 0.1043 | 0.1043 | 0.1043 | +0.0000 |
|  |  | M_bias_v1 | 0.1049 | 0.1049 | 0.1049 | 0.1032 | -0.0017 |
|  |  | M_bias_v2 | 0.1283 | 0.1283 | 0.1283 | 0.1242 | -0.0041 |

- L / C / U: the stored card of the arm rescored per clamp state. new: state-C rerun row (start: rescored start card). diff = new - stored (stored = L).
- a2: start card value; the refit cards of every arm also have a2 > 1 on these devices.

