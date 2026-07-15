# Foundation emulator study

This exploratory result uses one fixed conditional forward emulator for every device. All reported parameter vectors are re-simulated in real NGSpice. It is not used for per-device selection or card export.

| device | foundation raw | foundation + FD | device runtime (s) |
|---|---:|---:|---:|
| nmos_L0p15_W1p6 | 0.0346 | 0.0343 | 26.3 |
| nmos_L0p19_W7 | 0.0430 | 0.0428 | 22.8 |
| nmos_L0p25_W1p6 | 0.0506 | 0.0499 | 103.2 |
| nmos_L1_W1p6 | 0.0644 | 0.0635 | 29.5 |
| nmos_L1_W3 | 0.1105 | 0.1102 | 33.7 |
| nmos_L8_W1p6 | 0.0810 | 0.0806 | 121.7 |
| nmos_L20_W0p64 | 0.1207 | 0.1201 | 48.6 |
| nmos_L100_W100 | 0.1336 | 0.1335 | 32.9 |
| pmos_L0p35_W0p55 | 0.6309 | 0.6307 | 33.7 |
| pmos_L0p35_W1p6 | 0.3316 | 0.3284 | 40.3 |
| pmos_L0p35_W5 | 0.2889 | 0.2887 | 22.6 |
| pmos_L0p5_W0p42 | 0.3602 | 0.3590 | 30.8 |
| pmos_L0p5_W0p64 | 0.3282 | 0.3253 | 31.4 |
| pmos_L2_W5 | 0.1759 | 0.1758 | 280.1 |
| pmos_L4_W7 | 0.2238 | 0.2233 | 33.8 |
| pmos_L8_W0p84 | 0.3756 | 0.3740 | 34.4 |
| pmos_L8_W1p6 | 0.4708 | 0.4697 | 36.9 |
| pmos_L8_W5 | 0.2599 | 0.2592 | 34.0 |

## Aggregate

- Foundation raw all-device RRMS: 0.2269.
- Foundation + FD all-device RRMS: 0.2261.
- Held-out signed-log validation MSE: foundation global 0.000242765; arithmetic mean across its 18 device slices 0.000242765; mean of the 18 separate emulators 0.000214598.
- Foundation cache build once: 14.5 s; training once: 189.5 s; 18-device search/validation/FD: 996.7 s; first campaign total: 1200.7 s.
- Published-start FD-only total: 16.2 s.
- Eighteen per-device emulator extractions: 2420.2 s.

## Tradeoffs

- FD-only has no training cost and optimizes the real simulator directly. It is the natural one-time baseline, but it cannot evaluate thousands of candidates in parallel without thousands of NGSpice runs.
- The foundation emulator amortizes one training run across devices and makes inverse search differentiable and parallel. Its risks are shared-model bias across geometries, large training/cache cost, and the continuing need for NGSpice validation and usually FD polish. The current random within-geometry validation split does not prove accuracy on an entirely unseen L/W geometry.
- Per-device emulators isolate geometry-specific behavior and are simpler fits, at the cost of 18 training runs and 18 checkpoints.
- Runtime does not break even against FD-only in the measured workflow because foundation search/validation itself is no faster than the complete FD-only campaign, before training cost.
