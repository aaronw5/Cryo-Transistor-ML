# I10-2 decision

Bonferroni level 1 - 0.05/11 (alpha 0.00455); margin 1.03; TU Delft cluster bootstrap (groups).

| selector | GMR vs C0_refit (pooled) | GMR vs mlp16 (pooled) | GMR vs S_gate | label |
|---|---|---|---|---|
| S_loss | 0.838 [0.778, 0.900] | 0.914 [0.827, 0.992] | 0.938 [0.917, 0.958] | partly supported |
| S_refit | 0.834 [0.776, 0.897] | 0.910 [0.828, 0.979] | 0.934 [0.912, 0.959] | partly supported |
| S_gate | 0.893 [0.836, 0.951] | 0.974 [0.885, 1.049] | - | not supported |
| S_cv | 0.814 [0.742, 0.888] | 0.888 [0.797, 0.964] | 0.912 [0.865, 0.948] | partly supported |

| selector | sky130_77K | nist_4K | tudelft_4K | ihp_meas_300K |
|---|---|---|---|---|
| S_loss | 0.741 [0.586, 0.936] | 0.670 [0.590, 0.767] | 0.993 [0.871, 1.089] | 1.000 [1.000, 1.000] |
| S_refit | 0.727 [0.580, 0.911] | 0.670 [0.590, 0.767] | 0.976 [0.877, 1.089] | 1.020 [0.994, 1.092] |
| S_gate | 0.769 [0.623, 0.942] | 0.806 [0.740, 0.896] | 1.027 [0.911, 1.144] | 1.000 [1.000, 1.000] |
| S_cv | 0.727 [0.580, 0.911] | 0.670 [0.590, 0.767] | 0.939 [0.736, 1.164] | 0.962 [0.891, 1.009] |

Primary rule (in force): {"decision": "stop", "frozen_selector": null, "supported": []}

Secondary (amendment A1; the primary label and rule are unchanged): C1 per variant

| variant | S_loss | S_refit | S_gate | S_cv | fixed_mlp16 |
|---|---|---|---|---|---|
| primary_cluster_bonferroni | fail (tudelft_4K) | fail (ihp_meas_300K, tudelft_4K) | fail (tudelft_4K) | fail (tudelft_4K) | fail (ihp_meas_300K, tudelft_4K) |
| a_device_bootstrap_bonferroni | fail (tudelft_4K) | fail (ihp_meas_300K, tudelft_4K) | fail (tudelft_4K) | fail (tudelft_4K) | fail (ihp_meas_300K, tudelft_4K) |
| b_cluster_95 | fail (tudelft_4K) | fail (ihp_meas_300K, tudelft_4K) | fail (tudelft_4K) | fail (tudelft_4K) | fail (ihp_meas_300K, tudelft_4K) |
| ab_device_bootstrap_95 | fail (tudelft_4K) | fail (ihp_meas_300K) | fail (tudelft_4K) | pass | fail (ihp_meas_300K) |

