# I19-1 Part P: TEST gate decisions (committed before any TEST evaluation)

Written 2026-10-23 02:52:22; git HEAD at run `4da313cb0d8c56ea505d3b9c96b04676d86cca44`. Rule: apply the correction to every TEST device of the split iff the DEV S3_D < 0 (strict); else C0_refit for every TEST device. Threshold 0.0. only DEV records loaded; every loaded tag is in the DEV list and not in the TEST list of the split file (dev_check per row). sky130 77 K: no locked TEST since 2026-10-21 (user decision); no Part P row.

| TEST split | n TEST | n DEV records | arm | S3_D (DEV) | decision | decision at +0.02455 | v1 S3_D | v1 decision | I8-2 S3 sign | I8-2 S1 rule |
|---|---|---|---|---|---|---|---|---|---|---|
| NIST 4 K lvt TEST (I8-1 split) | 38 | 53 | M_bias v1 | -0.1027 | apply | apply | - | - | - | - |
| NIST 4 K room-temperature card TEST (I9-1 split) | 58 | 87 | M_bias v1 | -0.0689 | apply | apply | - | - | - | - |
| IHP SG13G2 TEST 233 K | 10 | 18 | M_bias v1 | +0.1866 | C0_refit | C0_refit | - | - | loss | loss |
| IHP SG13G2 TEST 300 K | 10 | 18 | selected g v2 (I17-2) | +0.3078 | C0_refit | C0_refit | +0.3114 | C0_refit | loss | loss |
| IHP SG13G2 TEST 343 K | 10 | 18 | M_bias v1 | +0.3669 | C0_refit | C0_refit | - | - | loss | loss |
| IHP SG13G2 TEST 398 K | 10 | 18 | M_bias v1 | +0.3667 | C0_refit | C0_refit | - | - | loss | loss |
| TU Delft TEST 4.13 K | 24 | 45 | selected g v2 (I18-2, state C) | +0.0080 | C0_refit | apply | -0.0292 | apply | gain | gain |
| TU Delft TEST 295.0 K | 24 | 45 | selected g v2 (I18-2, state C) | +0.2640 | C0_refit | C0_refit | -0.0510 | apply | gain | gain |

NIST rows: no I8-2 TEST prediction; committed DEV P_c (sign of the fold-mean S3):
- nist_4K: S3 fold mean -0.1081, P_c gain (out/research/explore/I8_1/prediction_v2.json)
- nist_rt_card_4K: S3 fold mean -0.0710, P_c gain (out/research/explore/I9_1/prediction.json)

Disagreements with the I8-2 committed TEST predictions (4; I8-2 predicts the sign for M_bias v1):

- tudelft_4K: I19-1 primary C0_refit vs I8-2 S3_predicts gain
- tudelft_4K: I19-1 primary C0_refit vs I8-2 S1_predicts gain
- tudelft_295K: I19-1 primary C0_refit vs I8-2 S3_predicts gain
- tudelft_295K: I19-1 primary C0_refit vs I8-2 S1_predicts gain

Evaluation at TEST: decision correct iff it equals the sign of the TEST ln GMR(correction, C0_refit).
