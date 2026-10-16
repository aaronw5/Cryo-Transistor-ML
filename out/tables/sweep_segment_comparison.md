# X-axis section RRMS comparison

Each admitted curve is split at normalized sweep positions 0, 0.2, 0.4, 0.7, and 1.0. Each cell is the mean of per-device means and uses the unchanged whole-curve paper RRMS denominator.

| Method | s0_0_0.2 | s1_0.2_0.4 | s2_0.4_0.7 | s3_0.7_1 |
|---|---:|---:|---:|---:|
| paper | 0.08984 | 0.14106 | 0.23997 | 0.37579 |
| seven_fd | 0.06974 | 0.10548 | 0.19140 | 0.31884 |
| seven_regional | 0.06974 | 0.10549 | 0.19156 | 0.31860 |
| full43_fd | 0.06342 | 0.09360 | 0.16300 | 0.28646 |
| full43_tradeoff | 0.06360 | 0.09196 | 0.16338 | 0.28362 |
| full43_regional | 0.06228 | 0.09192 | 0.16220 | 0.28536 |

## nMOS L=1 um, W=3 um, idvd@1.85 V

| Method | s0_0_0.2 | s1_0.2_0.4 | s2_0.4_0.7 | s3_0.7_1 |
|---|---:|---:|---:|---:|
| paper | 0.12887 | 0.05983 | 0.11049 | 0.10790 |
| seven_fd | 0.14686 | 0.06620 | 0.12954 | 0.13029 |
| seven_regional | 0.14686 | 0.06620 | 0.12954 | 0.13029 |
| full43_fd | 0.15332 | 0.08536 | 0.02611 | 0.01892 |
| full43_tradeoff | 0.15377 | 0.08506 | 0.02703 | 0.01994 |
| full43_regional | 0.15332 | 0.08536 | 0.02611 | 0.01892 |

## Material section regressions versus paper

| Method | Count |
|---|---:|
| seven_fd | 115/700 |
| seven_regional | 114/700 |
| full43_fd | 126/700 |
| full43_tradeoff | 103/700 |
| full43_regional | 108/700 |
