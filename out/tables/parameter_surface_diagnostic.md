# Seven-vs-expanded parameter-surface diagnostic

Direct scoring of stored real-NGSpice LHC curves; no surrogate, training loss, or parameter optimization is involved. This per-device oracle is a nondeployable representation diagnostic, not a headline ML method or exported card series.

| parameter set | samples/device | best-sample all-device RRMS | high-voltage RRMS at that sample |
|---|---:|---:|---:|
| seven | 10000 | 0.22967 | 0.07465 |
| full | 10000 | 0.20678 | 0.07440 |

## Per-device

| device | 7D best | 43D best | 7D high | 43D high |
|---|---:|---:|---:|---:|
| nmos_L0p15_W1p6 | 0.03578 | 0.03803 | 0.02091 | 0.02814 |
| nmos_L0p19_W7 | 0.04750 | 0.04104 | 0.02047 | 0.02482 |
| nmos_L0p25_W1p6 | 0.05195 | 0.04870 | 0.03448 | 0.03108 |
| nmos_L1_W1p6 | 0.06759 | 0.06482 | 0.03770 | 0.03369 |
| nmos_L1_W3 | 0.11258 | 0.06624 | 0.12153 | 0.04980 |
| nmos_L8_W1p6 | 0.08102 | 0.08070 | 0.04539 | 0.04105 |
| nmos_L20_W0p64 | 0.11608 | 0.11414 | 0.04964 | 0.04684 |
| nmos_L100_W100 | 0.13237 | 0.13169 | 0.08038 | 0.07975 |
| pmos_L0p35_W0p55 | 0.63349 | 0.62034 | 0.12417 | 0.18005 |
| pmos_L0p35_W1p6 | 0.33480 | 0.26908 | 0.05445 | 0.04024 |
| pmos_L0p35_W5 | 0.28817 | 0.28132 | 0.08905 | 0.08780 |
| pmos_L0p5_W0p42 | 0.37492 | 0.33554 | 0.17909 | 0.16050 |
| pmos_L0p5_W0p64 | 0.32815 | 0.31079 | 0.06600 | 0.06390 |
| pmos_L2_W5 | 0.17826 | 0.16701 | 0.04223 | 0.04835 |
| pmos_L4_W7 | 0.22712 | 0.19800 | 0.09392 | 0.10251 |
| pmos_L8_W0p84 | 0.38131 | 0.33431 | 0.09169 | 0.12030 |
| pmos_L8_W1p6 | 0.47682 | 0.39405 | 0.09211 | 0.11614 |
| pmos_L8_W5 | 0.26623 | 0.22629 | 0.10052 | 0.08434 |
