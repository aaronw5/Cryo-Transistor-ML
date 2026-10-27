# Expanded BSIM4 parameter sensitivity screen

Candidates: 115; selected: 37; devices: 2.

Requested normalized response threshold: `0.0001`; repeatability floor: `0`; effective threshold: `0.0001`.

Selection has no top-k cap. The original seven parameters are a mandatory nested subset.

## Selected parameters

`TOXE`, `TOXM`, `XJ`, `NDEP`, `NSD`, `DWG`, `VTH0`, `K1`, `K2`, `K3`, `DVT0`, `DVT1`, `W0`, `K3B`, `VFB`, `LPE0`, `LPEB`, `VSAT`, `UA`, `UB`, `RDSW`, `PRWG`, `WR`, `U0`, `A0`, `A2`, `AGS`, `B0`, `B1`, `NFACTOR`, `ETA0`, `DSUB`, `PCLM`, `PDIBLC1`, `PDIBLC2`, `DELTA`, `BETA0`

## Per-parameter response

| parameter | group | selected | max full-grid | max high-voltage | active devices |
|---|---|:---:|---:|---:|---:|
| TOXE | process_and_effective_geometry | yes | 978.5 | 0.2636 | 2 |
| TOXM | process_and_effective_geometry | yes | 827.7 | 0.1403 | 2 |
| XJ | process_and_effective_geometry | yes | 4.964 | 0.003754 | 2 |
| NDEP | process_and_effective_geometry | yes | 118.1 | 0.0167 | 2 |
| NSD | process_and_effective_geometry | yes | 6.08 | 0.0008235 | 1 |
| DWG | process_and_effective_geometry | yes | 0.1229 | 0.001735 | 2 |
| DWB | process_and_effective_geometry | no | 9.96e-09 | 9.96e-09 | 0 |
| RSH | process_and_effective_geometry | no | 6.463e-06 | 4.646e-06 | 0 |
| VTH0 | threshold_and_short_channel | yes | 753 | 0.09538 | 2 |
| K1 | threshold_and_short_channel | yes | 20 | 0.01754 | 2 |
| K2 | threshold_and_short_channel | yes | 6.235 | 0.005374 | 2 |
| K3 | threshold_and_short_channel | yes | 11.04 | 0.001493 | 2 |
| DVT0 | threshold_and_short_channel | yes | 176.7 | 0.02349 | 2 |
| DVT1 | threshold_and_short_channel | yes | 528.9 | 0.0674 | 2 |
| DVT2 | threshold_and_short_channel | no | 9.798e-10 | 9.577e-10 | 0 |
| DVT0W | threshold_and_short_channel | no | 2.633e-74 | 4.145e-79 | 0 |
| DVT1W | threshold_and_short_channel | no | 8.895e-18 | 1.667e-79 | 0 |
| DVT2W | threshold_and_short_channel | no | 0 | 0 | 0 |
| W0 | threshold_and_short_channel | yes | 0.06814 | 1.206e-05 | 1 |
| K3B | threshold_and_short_channel | yes | 0.1694 | 0.0001463 | 2 |
| VFB | threshold_and_short_channel | yes | 15.74 | 0.01026 | 1 |
| PHIN | threshold_and_short_channel | no | 0 | 0 | 0 |
| LPE0 | threshold_and_short_channel | yes | 295.1 | 0.03855 | 2 |
| LPEB | threshold_and_short_channel | yes | 1.585 | 0.001368 | 2 |
| VBM | threshold_and_short_channel | no | 0 | 0 | 0 |
| DVTP0 | threshold_and_short_channel | no | 0 | 0 | 0 |
| DVTP1 | threshold_and_short_channel | no | 0 | 0 | 0 |
| VSAT | mobility_velocity_and_resistance | yes | 0.8837 | 0.02193 | 2 |
| UA | mobility_velocity_and_resistance | yes | 66.5 | 0.4319 | 2 |
| UB | mobility_velocity_and_resistance | yes | 8.234 | 0.1427 | 2 |
| UC | mobility_velocity_and_resistance | no | 1.647e-07 | 1.504e-07 | 0 |
| RDSW | mobility_velocity_and_resistance | yes | 0.2906 | 0.009092 | 2 |
| PRWB | mobility_velocity_and_resistance | no | 2.326e-09 | 2.109e-09 | 0 |
| PRWG | mobility_velocity_and_resistance | yes | 0.003257 | 0.000609 | 1 |
| WR | mobility_velocity_and_resistance | yes | 0.1398 | 0.004328 | 2 |
| U0 | mobility_velocity_and_resistance | yes | 131.4 | 0.1319 | 2 |
| A0 | mobility_velocity_and_resistance | yes | 32.61 | 0.02815 | 2 |
| KETA | mobility_velocity_and_resistance | no | 7.832e-07 | 7.832e-07 | 0 |
| A1 | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| A2 | mobility_velocity_and_resistance | yes | 0.179 | 0.02362 | 2 |
| AGS | mobility_velocity_and_resistance | yes | 0.2737 | 0.03137 | 2 |
| B0 | mobility_velocity_and_resistance | yes | 91.12 | 0.08897 | 2 |
| B1 | mobility_velocity_and_resistance | yes | 0.03651 | 3.626e-05 | 1 |
| EU | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| UP | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| LP | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| UD | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| RDSWMIN | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| RDW | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| RDWMIN | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| RSW | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| RSWMIN | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| VOFF | subthreshold_and_dibl | no | 8.738e-05 | 2.466e-08 | 0 |
| NFACTOR | subthreshold_and_dibl | yes | 0.4332 | 3.375e-05 | 1 |
| TVFBSDOFF | subthreshold_and_dibl | no | 0 | 0 | 0 |
| TVOFF | subthreshold_and_dibl | no | 0 | 0 | 0 |
| CIT | subthreshold_and_dibl | no | 4.368e-07 | 9.608e-09 | 0 |
| CDSC | subthreshold_and_dibl | no | 5.878e-06 | 4.584e-10 | 0 |
| CDSCB | subthreshold_and_dibl | no | 1.257e-32 | 0 | 0 |
| CDSCD | subthreshold_and_dibl | no | 3.788e-07 | 4.213e-10 | 0 |
| ETA0 | subthreshold_and_dibl | yes | 0.009535 | 1.807e-06 | 1 |
| ETAB | subthreshold_and_dibl | no | 0 | 0 | 0 |
| DSUB | subthreshold_and_dibl | yes | 0.1763 | 3.34e-05 | 1 |
| VOFFL | subthreshold_and_dibl | no | 0 | 0 | 0 |
| MINV | subthreshold_and_dibl | no | 0 | 0 | 0 |
| PCLM | output_conductance_and_high_field | yes | 1.424 | 0.00447 | 2 |
| PDIBLC1 | output_conductance_and_high_field | yes | 3.242 | 0.0008279 | 1 |
| PDIBLC2 | output_conductance_and_high_field | yes | 74.65 | 0.01907 | 2 |
| PDIBLCB | output_conductance_and_high_field | no | 1.659e-09 | 1.659e-09 | 0 |
| DROUT | output_conductance_and_high_field | no | 0 | 0 | 0 |
| PSCBE1 | output_conductance_and_high_field | no | 8.471e-07 | 1.512e-08 | 0 |
| PSCBE2 | output_conductance_and_high_field | no | 2.629e-09 | 1.005e-09 | 0 |
| PVAG | output_conductance_and_high_field | no | 5.906e-75 | 9.725e-81 | 0 |
| DELTA | output_conductance_and_high_field | yes | 0.6447 | 0.0009851 | 2 |
| ALPHA0 | output_conductance_and_high_field | no | 4.167e-06 | 5.765e-07 | 0 |
| ALPHA1 | output_conductance_and_high_field | no | 2.473e-74 | 2.576e-79 | 0 |
| BETA0 | output_conductance_and_high_field | yes | 0.0001118 | 2.716e-05 | 1 |
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
| RBPB | body_network_and_junction_dc | no | 3.788e-07 | 1.078e-09 | 0 |
| RBPD | body_network_and_junction_dc | no | 3.788e-07 | 7.097e-10 | 0 |
| RBPS | body_network_and_junction_dc | no | 7.635e-10 | 7.097e-10 | 0 |
| RBDB | body_network_and_junction_dc | no | 3.788e-07 | 7.097e-10 | 0 |
| RBSB | body_network_and_junction_dc | no | 7.635e-10 | 7.097e-10 | 0 |
| GBMIN | body_network_and_junction_dc | no | 3.028e-74 | 2.211e-79 | 0 |
| NJS | body_network_and_junction_dc | no | 1.188e-10 | 0 | 0 |
| JSS | body_network_and_junction_dc | no | 4.084e-09 | 2.467e-10 | 0 |
| JSWS | body_network_and_junction_dc | no | 3.968e-09 | 2.467e-10 | 0 |
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
