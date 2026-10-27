# Expanded BSIM4 parameter sensitivity screen

Candidates: 115; selected: 37; devices: 2.

Requested normalized response threshold: `0.0001`; repeatability floor: `0`; effective threshold: `0.0001`.

Selection has no top-k cap. The original seven parameters are a mandatory nested subset.

## Selected parameters

`TOXE`, `TOXM`, `XJ`, `NDEP`, `DWG`, `RSH`, `VTH0`, `K1`, `K2`, `K3`, `DVT0`, `DVT1`, `K3B`, `LPE0`, `LPEB`, `VSAT`, `UA`, `UB`, `RDSW`, `PRWG`, `WR`, `U0`, `A0`, `A2`, `AGS`, `NFACTOR`, `ETA0`, `PCLM`, `PDIBLC2`, `DELTA`, `ALPHA0`, `BETA0`, `KT1`, `AT`, `UTE`, `UA1`, `UB1`

## Per-parameter response

| parameter | group | selected | max full-grid | max high-voltage | active devices |
|---|---|:---:|---:|---:|---:|
| TOXE | process_and_effective_geometry | yes | 9.488 | 0.515 | 2 |
| TOXM | process_and_effective_geometry | yes | 6.159 | 0.3432 | 2 |
| XJ | process_and_effective_geometry | yes | 0.01607 | 0.002534 | 2 |
| NDEP | process_and_effective_geometry | yes | 0.00211 | 0.001415 | 2 |
| NSD | process_and_effective_geometry | no | 6.736e-05 | 1.456e-05 | 0 |
| DWG | process_and_effective_geometry | yes | 0.001551 | 0.001345 | 2 |
| DWB | process_and_effective_geometry | no | 5.826e-08 | 5.482e-08 | 0 |
| RSH | process_and_effective_geometry | yes | 0.0001089 | 2.377e-05 | 1 |
| VTH0 | threshold_and_short_channel | yes | 4.13 | 0.2009 | 2 |
| K1 | threshold_and_short_channel | yes | 0.4135 | 0.04834 | 2 |
| K2 | threshold_and_short_channel | yes | 0.02959 | 0.00347 | 2 |
| K3 | threshold_and_short_channel | yes | 0.0228 | 0.001154 | 2 |
| DVT0 | threshold_and_short_channel | yes | 0.0004539 | 9.808e-05 | 2 |
| DVT1 | threshold_and_short_channel | yes | 0.002971 | 0.0004967 | 2 |
| DVT2 | threshold_and_short_channel | no | 3.935e-09 | 3.935e-09 | 0 |
| DVT0W | threshold_and_short_channel | no | 0 | 0 | 0 |
| DVT1W | threshold_and_short_channel | no | 0 | 0 | 0 |
| DVT2W | threshold_and_short_channel | no | 0 | 0 | 0 |
| W0 | threshold_and_short_channel | no | 8.871e-05 | 2.031e-05 | 0 |
| K3B | threshold_and_short_channel | yes | 0.0008224 | 9.647e-05 | 1 |
| VFB | threshold_and_short_channel | no | 0 | 0 | 0 |
| PHIN | threshold_and_short_channel | no | 0 | 0 | 0 |
| LPE0 | threshold_and_short_channel | yes | 0.3423 | 0.01727 | 2 |
| LPEB | threshold_and_short_channel | yes | 0.005236 | 0.0006141 | 2 |
| VBM | threshold_and_short_channel | no | 0 | 0 | 0 |
| DVTP0 | threshold_and_short_channel | no | 0 | 0 | 0 |
| DVTP1 | threshold_and_short_channel | no | 0 | 0 | 0 |
| VSAT | mobility_velocity_and_resistance | yes | 0.003408 | 0.00194 | 2 |
| UA | mobility_velocity_and_resistance | yes | 4.208 | 0.7863 | 2 |
| UB | mobility_velocity_and_resistance | yes | 1.516 | 0.4338 | 2 |
| UC | mobility_velocity_and_resistance | no | 5.057e-07 | 3.454e-07 | 0 |
| RDSW | mobility_velocity_and_resistance | yes | 0.02054 | 0.005631 | 2 |
| PRWB | mobility_velocity_and_resistance | no | 4.22e-08 | 2.831e-08 | 0 |
| PRWG | mobility_velocity_and_resistance | yes | 0.0009438 | 0.0003288 | 2 |
| WR | mobility_velocity_and_resistance | yes | 0.02278 | 0.006394 | 2 |
| U0 | mobility_velocity_and_resistance | yes | 0.8856 | 0.1261 | 2 |
| A0 | mobility_velocity_and_resistance | yes | 0.7822 | 0.09069 | 2 |
| KETA | mobility_velocity_and_resistance | no | 1.21e-06 | 1.21e-06 | 0 |
| A1 | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| A2 | mobility_velocity_and_resistance | yes | 0.1169 | 0.1126 | 2 |
| AGS | mobility_velocity_and_resistance | yes | 0.05485 | 0.03095 | 2 |
| B0 | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| B1 | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| EU | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| UP | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| LP | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| UD | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| RDSWMIN | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| RDW | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| RDWMIN | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| RSW | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| RSWMIN | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| VOFF | subthreshold_and_dibl | no | 7.781e-06 | 2.049e-07 | 0 |
| NFACTOR | subthreshold_and_dibl | yes | 0.004714 | 2.456e-05 | 2 |
| TVFBSDOFF | subthreshold_and_dibl | no | 0 | 0 | 0 |
| TVOFF | subthreshold_and_dibl | no | 0 | 0 | 0 |
| CIT | subthreshold_and_dibl | no | 1.836e-06 | 8.454e-08 | 0 |
| CDSC | subthreshold_and_dibl | no | 3.018e-09 | 1.071e-10 | 0 |
| CDSCB | subthreshold_and_dibl | no | 2.794e-47 | 0 | 0 |
| CDSCD | subthreshold_and_dibl | no | 4.325e-08 | 3.171e-09 | 0 |
| ETA0 | subthreshold_and_dibl | yes | 3.278e-47 | 3.278e-47 | 0 |
| ETAB | subthreshold_and_dibl | no | 0 | 0 | 0 |
| DSUB | subthreshold_and_dibl | no | 3.278e-47 | 3.278e-47 | 0 |
| VOFFL | subthreshold_and_dibl | no | 0 | 0 | 0 |
| MINV | subthreshold_and_dibl | no | 0 | 0 | 0 |
| PCLM | output_conductance_and_high_field | yes | 0.009491 | 0.009491 | 2 |
| PDIBLC1 | output_conductance_and_high_field | no | 1.017e-09 | 0 | 0 |
| PDIBLC2 | output_conductance_and_high_field | yes | 0.02143 | 0.001388 | 2 |
| PDIBLCB | output_conductance_and_high_field | no | 1.029e-08 | 1.029e-08 | 0 |
| DROUT | output_conductance_and_high_field | no | 0 | 0 | 0 |
| PSCBE1 | output_conductance_and_high_field | no | 3.527e-07 | 3.527e-07 | 0 |
| PSCBE2 | output_conductance_and_high_field | no | 1.603e-08 | 1.603e-08 | 0 |
| PVAG | output_conductance_and_high_field | no | 0 | 0 | 0 |
| DELTA | output_conductance_and_high_field | yes | 0.04285 | 0.003217 | 2 |
| ALPHA0 | output_conductance_and_high_field | yes | 0.0001565 | 0.0001565 | 1 |
| ALPHA1 | output_conductance_and_high_field | no | 0 | 0 | 0 |
| BETA0 | output_conductance_and_high_field | yes | 0.0009067 | 0.0009067 | 1 |
| FPROUT | output_conductance_and_high_field | no | 0 | 0 | 0 |
| PDITS | output_conductance_and_high_field | no | 0 | 0 | 0 |
| PDITSL | output_conductance_and_high_field | no | 0 | 0 | 0 |
| PDITSD | output_conductance_and_high_field | no | 0 | 0 | 0 |
| AGIDL | gidl | no | 0 | 0 | 0 |
| BGIDL | gidl | no | 0 | 0 | 0 |
| CGIDL | gidl | no | 0 | 0 | 0 |
| EGIDL | gidl | no | 0 | 0 | 0 |
| VFBSDOFF | temperature | no | 0 | 0 | 0 |
| KT1 | temperature | yes | 4.623 | 0.224 | 2 |
| KT2 | temperature | no | 9.709e-07 | 8.76e-07 | 0 |
| AT | temperature | yes | 0.006668 | 0.005927 | 2 |
| UTE | temperature | yes | 0.2436 | 0.0845 | 2 |
| UA1 | temperature | yes | 0.341 | 0.06033 | 2 |
| UB1 | temperature | yes | 0.1022 | 0.03443 | 2 |
| UC1 | temperature | no | 7.262e-08 | 4.926e-08 | 0 |
| KT1L | temperature | no | 0 | 0 | 0 |
| PRT | temperature | no | 0 | 0 | 0 |
| RBPB | body_network_and_junction_dc | no | 8.212e-08 | 8.212e-08 | 0 |
| RBPD | body_network_and_junction_dc | no | 2.099e-08 | 2.099e-08 | 0 |
| RBPS | body_network_and_junction_dc | no | 2.099e-08 | 2.099e-08 | 0 |
| RBDB | body_network_and_junction_dc | no | 2.099e-08 | 2.099e-08 | 0 |
| RBSB | body_network_and_junction_dc | no | 2.099e-08 | 2.099e-08 | 0 |
| GBMIN | body_network_and_junction_dc | no | 3.928e-29 | 0 | 0 |
| NJS | body_network_and_junction_dc | no | 2.858e-08 | 2.454e-46 | 0 |
| JSS | body_network_and_junction_dc | no | 0 | 0 | 0 |
| JSWS | body_network_and_junction_dc | no | 0 | 0 | 0 |
| XTIS | body_network_and_junction_dc | no | 0 | 0 | 0 |
| BVS | body_network_and_junction_dc | no | 0 | 0 | 0 |
| XJBVS | body_network_and_junction_dc | no | 0 | 0 | 0 |
| IJTHSREV | body_network_and_junction_dc | no | 0 | 0 | 0 |
| IJTHSFWD | body_network_and_junction_dc | no | 0 | 0 | 0 |
| TPB | body_network_and_junction_dc | no | 0 | 0 | 0 |
| TPBSW | body_network_and_junction_dc | no | 0 | 0 | 0 |
| TPBSWG | body_network_and_junction_dc | no | 0 | 0 | 0 |
| TCJ | body_network_and_junction_dc | no | 0 | 0 | 0 |
| TCJSW | body_network_and_junction_dc | no | 0 | 0 | 0 |
| TCJSWG | body_network_and_junction_dc | no | 0 | 0 | 0 |
