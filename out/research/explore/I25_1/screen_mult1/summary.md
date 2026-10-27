# Expanded BSIM4 parameter sensitivity screen

Candidates: 115; selected: 45; devices: 18.

Requested normalized response threshold: `0.0001`; repeatability floor: `0`; effective threshold: `0.0001`.

Selection has no top-k cap. The original seven parameters are a mandatory nested subset.

## Selected parameters

`TOXE`, `TOXM`, `XJ`, `NDEP`, `NSD`, `DWG`, `RSH`, `VTH0`, `K1`, `K2`, `K3`, `DVT0`, `DVT1`, `DVT0W`, `DVT1W`, `W0`, `K3B`, `VFB`, `LPE0`, `LPEB`, `VSAT`, `UA`, `UB`, `RDSW`, `PRWG`, `WR`, `U0`, `A0`, `A2`, `AGS`, `B0`, `B1`, `VOFF`, `NFACTOR`, `CIT`, `CDSC`, `ETA0`, `DSUB`, `PCLM`, `PDIBLC1`, `PDIBLC2`, `DROUT`, `DELTA`, `ALPHA0`, `BETA0`

## Per-parameter response

| parameter | group | selected | max full-grid | max high-voltage | active devices |
|---|---|:---:|---:|---:|---:|
| TOXE | process_and_effective_geometry | yes | 1.301e+04 | 0.9284 | 18 |
| TOXM | process_and_effective_geometry | yes | 6958 | 0.4678 | 18 |
| XJ | process_and_effective_geometry | yes | 18.34 | 0.01169 | 16 |
| NDEP | process_and_effective_geometry | yes | 852.1 | 0.05564 | 16 |
| NSD | process_and_effective_geometry | yes | 43.99 | 0.002751 | 9 |
| DWG | process_and_effective_geometry | yes | 1.169 | 0.01776 | 17 |
| DWB | process_and_effective_geometry | no | 7.708e-06 | 3.949e-07 | 0 |
| RSH | process_and_effective_geometry | yes | 0.003983 | 3.304e-05 | 5 |
| VTH0 | threshold_and_short_channel | yes | 5343 | 0.2015 | 18 |
| K1 | threshold_and_short_channel | yes | 548.5 | 0.1425 | 18 |
| K2 | threshold_and_short_channel | yes | 99.7 | 0.01024 | 17 |
| K3 | threshold_and_short_channel | yes | 25.52 | 0.0146 | 17 |
| DVT0 | threshold_and_short_channel | yes | 1296 | 0.07853 | 11 |
| DVT1 | threshold_and_short_channel | yes | 3945 | 0.2249 | 12 |
| DVT2 | threshold_and_short_channel | no | 2.63e-05 | 2.041e-07 | 0 |
| DVT0W | threshold_and_short_channel | yes | 0.0005645 | 1.18e-06 | 1 |
| DVT1W | threshold_and_short_channel | yes | 0.01098 | 2.294e-05 | 2 |
| DVT2W | threshold_and_short_channel | no | 2.017e-08 | 4.73e-10 | 0 |
| W0 | threshold_and_short_channel | yes | 0.1264 | 0.0002656 | 15 |
| K3B | threshold_and_short_channel | yes | 0.988 | 0.001341 | 15 |
| VFB | threshold_and_short_channel | yes | 164.9 | 0.03963 | 10 |
| PHIN | threshold_and_short_channel | no | 0 | 0 | 0 |
| LPE0 | threshold_and_short_channel | yes | 2179 | 0.1287 | 18 |
| LPEB | threshold_and_short_channel | yes | 6.051 | 0.004286 | 15 |
| VBM | threshold_and_short_channel | no | 0 | 0 | 0 |
| DVTP0 | threshold_and_short_channel | no | 0 | 0 | 0 |
| DVTP1 | threshold_and_short_channel | no | 0 | 0 | 0 |
| VSAT | mobility_velocity_and_resistance | yes | 107.2 | 0.9488 | 17 |
| UA | mobility_velocity_and_resistance | yes | 5198 | 1.696 | 18 |
| UB | mobility_velocity_and_resistance | yes | 1284 | 0.8116 | 18 |
| UC | mobility_velocity_and_resistance | no | 1.386e-05 | 1.786e-06 | 0 |
| RDSW | mobility_velocity_and_resistance | yes | 2.343 | 0.03029 | 16 |
| PRWB | mobility_velocity_and_resistance | no | 1.499e-06 | 7.766e-08 | 0 |
| PRWG | mobility_velocity_and_resistance | yes | 0.05104 | 0.002026 | 13 |
| WR | mobility_velocity_and_resistance | yes | 4.07 | 0.03066 | 15 |
| U0 | mobility_velocity_and_resistance | yes | 1585 | 0.3946 | 18 |
| A0 | mobility_velocity_and_resistance | yes | 624.6 | 0.1397 | 18 |
| KETA | mobility_velocity_and_resistance | no | 1.126e-05 | 6.134e-06 | 0 |
| A1 | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| A2 | mobility_velocity_and_resistance | yes | 2.717 | 0.06922 | 16 |
| AGS | mobility_velocity_and_resistance | yes | 70.32 | 0.1525 | 17 |
| B0 | mobility_velocity_and_resistance | yes | 119.6 | 0.08897 | 17 |
| B1 | mobility_velocity_and_resistance | yes | 9.428 | 0.212 | 5 |
| EU | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| UP | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| LP | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| UD | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| RDSWMIN | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| RDW | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| RDWMIN | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| RSW | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| RSWMIN | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| VOFF | subthreshold_and_dibl | yes | 0.1668 | 5.851e-05 | 4 |
| NFACTOR | subthreshold_and_dibl | yes | 1.282 | 8.389e-05 | 13 |
| TVFBSDOFF | subthreshold_and_dibl | no | 0 | 0 | 0 |
| TVOFF | subthreshold_and_dibl | no | 0 | 0 | 0 |
| CIT | subthreshold_and_dibl | yes | 0.0006357 | 1.203e-07 | 2 |
| CDSC | subthreshold_and_dibl | yes | 0.0002015 | 6.166e-09 | 1 |
| CDSCB | subthreshold_and_dibl | no | 8.624e-07 | 7.811e-12 | 0 |
| CDSCD | subthreshold_and_dibl | no | 7.035e-05 | 1.945e-07 | 0 |
| ETA0 | subthreshold_and_dibl | yes | 0.7146 | 0.03998 | 6 |
| ETAB | subthreshold_and_dibl | no | 9.73e-06 | 5.595e-06 | 0 |
| DSUB | subthreshold_and_dibl | yes | 55.43 | 0.2036 | 8 |
| VOFFL | subthreshold_and_dibl | no | 0 | 0 | 0 |
| MINV | subthreshold_and_dibl | no | 0 | 0 | 0 |
| PCLM | output_conductance_and_high_field | yes | 7.049 | 0.1606 | 16 |
| PDIBLC1 | output_conductance_and_high_field | yes | 16.7 | 0.002717 | 5 |
| PDIBLC2 | output_conductance_and_high_field | yes | 123.2 | 0.04898 | 18 |
| PDIBLCB | output_conductance_and_high_field | no | 6.736e-06 | 8.456e-08 | 0 |
| DROUT | output_conductance_and_high_field | yes | 0.000184 | 1.036e-05 | 1 |
| PSCBE1 | output_conductance_and_high_field | no | 4.768e-06 | 1.227e-06 | 0 |
| PSCBE2 | output_conductance_and_high_field | no | 1.725e-06 | 4.905e-08 | 0 |
| PVAG | output_conductance_and_high_field | no | 5.906e-75 | 9.725e-81 | 0 |
| DELTA | output_conductance_and_high_field | yes | 22.63 | 0.03292 | 18 |
| ALPHA0 | output_conductance_and_high_field | yes | 0.0002342 | 8.048e-05 | 1 |
| ALPHA1 | output_conductance_and_high_field | no | 2.473e-74 | 2.576e-79 | 0 |
| BETA0 | output_conductance_and_high_field | yes | 0.002407 | 0.001015 | 6 |
| FPROUT | output_conductance_and_high_field | no | 0 | 0 | 0 |
| PDITS | output_conductance_and_high_field | no | 0 | 0 | 0 |
| PDITSL | output_conductance_and_high_field | no | 0 | 0 | 0 |
| PDITSD | output_conductance_and_high_field | no | 0 | 0 | 0 |
| AGIDL | gidl | no | 0 | 0 | 0 |
| BGIDL | gidl | no | 0 | 0 | 0 |
| CGIDL | gidl | no | 0 | 0 | 0 |
| EGIDL | gidl | no | 0 | 0 | 0 |
| VFBSDOFF | temperature | no | 0 | 0 | 0 |
| KT1 | temperature | no | 0 | 0 | 0 |
| KT2 | temperature | no | 0 | 0 | 0 |
| AT | temperature | no | 0 | 0 | 0 |
| UTE | temperature | no | 0 | 0 | 0 |
| UA1 | temperature | no | 0 | 0 | 0 |
| UB1 | temperature | no | 0 | 0 | 0 |
| UC1 | temperature | no | 0 | 0 | 0 |
| KT1L | temperature | no | 0 | 0 | 0 |
| PRT | temperature | no | 0 | 0 | 0 |
| RBPB | body_network_and_junction_dc | no | 1.191e-06 | 2.014e-08 | 0 |
| RBPD | body_network_and_junction_dc | no | 8.624e-07 | 5.202e-09 | 0 |
| RBPS | body_network_and_junction_dc | no | 2.017e-08 | 5.202e-09 | 0 |
| RBDB | body_network_and_junction_dc | no | 8.624e-07 | 5.202e-09 | 0 |
| RBSB | body_network_and_junction_dc | no | 2.017e-08 | 5.202e-09 | 0 |
| GBMIN | body_network_and_junction_dc | no | 7.609e-30 | 2.211e-79 | 0 |
| NJS | body_network_and_junction_dc | no | 9.47e-08 | 5.732e-12 | 0 |
| JSS | body_network_and_junction_dc | no | 3.183e-06 | 4.191e-10 | 0 |
| JSWS | body_network_and_junction_dc | no | 4.644e-06 | 7.928e-10 | 0 |
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
