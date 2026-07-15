# Confirmed-setup comparison

All NGSpice columns use the confirmed-setup chain (CryoPDK_Skywater130nm_ML decks + updated pFET card, ngspice-41, native geometry bins) scored with the rrmsCalc metric; curve inclusion is frozen to the published-card baseline's included set for every method. combined = (nMOS mean + pMOS mean)/2, the upstream headline convention.

| reference / method | mean RRMS (all 18) | combined | nMOS (8) | pMOS (10) |
|---|---:|---:|---:|---:|
| paper reported | 0.279 | 0.263 | 0.120 | 0.406 |
| paper parameters in NGSpice | 0.275 | 0.260 | 0.120 | 0.399 |
| direct_mlp_forward_pass | 0.272 | 0.258 | 0.128 | 0.387 |
| surrogate_search_raw | 0.236 | 0.220 | 0.083 | 0.358 |
| surrogate_search_plus_fd | 0.229 | 0.214 | 0.079 | 0.349 |
| foundation_plus_fd | 0.226 | 0.211 | 0.079 | 0.343 |
| high_voltage_guarded | 0.233 | 0.218 | 0.081 | 0.354 |

| device | paper reported | paper params in NGSpice | direct_mlp_forward_pass | surrogate_search_raw | surrogate_search_plus_fd | foundation_plus_fd | high_voltage_guarded |
|---|---:|---:|---:|---:|---:|---:|---:|
| nmos_L0p15_W1p6 | 0.059 | 0.059 | 0.110 | 0.035 | 0.035 | 0.034 | 0.036 |
| nmos_L0p19_W7 | 0.097 | 0.097 | 0.176 | 0.044 | 0.043 | 0.043 | 0.048 |
| nmos_L0p25_W1p6 | 0.098 | 0.098 | 0.099 | 0.052 | 0.050 | 0.050 | 0.052 |
| nmos_L1_W1p6 | 0.128 | 0.128 | 0.122 | 0.067 | 0.065 | 0.064 | 0.068 |
| nmos_L1_W3 | 0.160 | 0.160 | 0.140 | 0.137 | 0.110 | 0.110 | 0.113 |
| nmos_L8_W1p6 | 0.147 | 0.147 | 0.122 | 0.080 | 0.080 | 0.081 | 0.081 |
| nmos_L20_W0p64 | 0.130 | 0.128 | 0.123 | 0.115 | 0.115 | 0.120 | 0.116 |
| nmos_L100_W100 | 0.142 | 0.140 | 0.135 | 0.133 | 0.132 | 0.134 | 0.132 |
| pmos_L0p35_W0p55 | 0.701 | 0.727 | 0.738 | 0.664 | 0.639 | 0.631 | 0.641 |
| pmos_L0p35_W1p6 | 0.374 | 0.378 | 0.380 | 0.346 | 0.329 | 0.328 | 0.335 |
| pmos_L0p35_W5 | 0.324 | 0.329 | 0.321 | 0.339 | 0.317 | 0.289 | 0.288 |
| pmos_L0p5_W0p42 | 0.465 | 0.456 | 0.437 | 0.362 | 0.361 | 0.359 | 0.389 |
| pmos_L0p5_W0p64 | 0.322 | 0.350 | 0.390 | 0.340 | 0.329 | 0.325 | 0.328 |
| pmos_L2_W5 | 0.207 | 0.191 | 0.230 | 0.190 | 0.190 | 0.176 | 0.186 |
| pmos_L4_W7 | 0.281 | 0.270 | 0.242 | 0.224 | 0.223 | 0.223 | 0.238 |
| pmos_L8_W0p84 | 0.480 | 0.434 | 0.380 | 0.376 | 0.374 | 0.374 | 0.397 |
| pmos_L8_W1p6 | 0.515 | 0.543 | 0.482 | 0.478 | 0.470 | 0.470 | 0.477 |
| pmos_L8_W5 | 0.388 | 0.315 | 0.274 | 0.261 | 0.260 | 0.259 | 0.266 |
