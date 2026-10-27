# Expanded BSIM4 parameter sensitivity screen

Candidates: 115; selected: 45; devices: 18.

Requested normalized response threshold: `0.0001`; repeatability floor: `0`; effective threshold: `0.0001`.

Selection has no top-k cap. The original seven parameters are a mandatory nested subset.

## Selected parameters

`TOXE`, `TOXM`, `XJ`, `NDEP`, `NSD`, `DWG`, `RSH`, `VTH0`, `K1`, `K2`, `K3`, `DVT0`, `DVT1`, `DVT1W`, `W0`, `K3B`, `LPE0`, `LPEB`, `VSAT`, `UA`, `UB`, `RDSW`, `PRWG`, `WR`, `U0`, `A0`, `A2`, `AGS`, `B0`, `B1`, `VOFF`, `NFACTOR`, `ETA0`, `DSUB`, `PCLM`, `PDIBLC1`, `PDIBLC2`, `DELTA`, `ALPHA0`, `BETA0`, `KT1`, `AT`, `UTE`, `UA1`, `UB1`

## Per-parameter response

| parameter | group | selected | max full-grid | max high-voltage | active devices |
|---|---|:---:|---:|---:|---:|
| TOXE | process_and_effective_geometry | yes | 160.1 | 2.27 | 18 |
| TOXM | process_and_effective_geometry | yes | 192.4 | 1.561 | 18 |
| XJ | process_and_effective_geometry | yes | 0.1142 | 0.01875 | 17 |
| NDEP | process_and_effective_geometry | yes | 6.835 | 0.1458 | 18 |
| NSD | process_and_effective_geometry | yes | 1.454 | 0.03222 | 8 |
| DWG | process_and_effective_geometry | yes | 0.05037 | 0.01209 | 17 |
| DWB | process_and_effective_geometry | no | 4.931e-07 | 4.931e-07 | 0 |
| RSH | process_and_effective_geometry | yes | 0.0006643 | 9.195e-05 | 8 |
| VTH0 | threshold_and_short_channel | yes | 216 | 1.079 | 18 |
| K1 | threshold_and_short_channel | yes | 2.427 | 0.2388 | 18 |
| K2 | threshold_and_short_channel | yes | 0.2581 | 0.03487 | 16 |
| K3 | threshold_and_short_channel | yes | 1.746 | 0.03103 | 17 |
| DVT0 | threshold_and_short_channel | yes | 13.66 | 0.2174 | 11 |
| DVT1 | threshold_and_short_channel | yes | 72.49 | 0.6395 | 11 |
| DVT2 | threshold_and_short_channel | no | 1.641e-06 | 6.758e-07 | 0 |
| DVT0W | threshold_and_short_channel | no | 9.631e-05 | 1.151e-06 | 0 |
| DVT1W | threshold_and_short_channel | yes | 0.001997 | 2.384e-05 | 1 |
| DVT2W | threshold_and_short_channel | no | 4.73e-10 | 4.73e-10 | 0 |
| W0 | threshold_and_short_channel | yes | 0.03076 | 0.0004275 | 12 |
| K3B | threshold_and_short_channel | yes | 0.02094 | 0.002742 | 13 |
| VFB | threshold_and_short_channel | no | 0 | 0 | 0 |
| PHIN | threshold_and_short_channel | no | 0 | 0 | 0 |
| LPE0 | threshold_and_short_channel | yes | 31.46 | 0.3924 | 18 |
| LPEB | threshold_and_short_channel | yes | 0.03712 | 0.006589 | 15 |
| VBM | threshold_and_short_channel | no | 0 | 0 | 0 |
| DVTP0 | threshold_and_short_channel | no | 0 | 0 | 0 |
| DVTP1 | threshold_and_short_channel | no | 0 | 0 | 0 |
| VSAT | mobility_velocity_and_resistance | yes | 1.891 | 0.8142 | 18 |
| UA | mobility_velocity_and_resistance | yes | 17.48 | 3.999 | 18 |
| UB | mobility_velocity_and_resistance | yes | 9.644 | 2.855 | 18 |
| UC | mobility_velocity_and_resistance | no | 8.226e-06 | 8.226e-06 | 0 |
| RDSW | mobility_velocity_and_resistance | yes | 0.5886 | 0.1122 | 18 |
| PRWB | mobility_velocity_and_resistance | no | 3.634e-07 | 3.634e-07 | 0 |
| PRWG | mobility_velocity_and_resistance | yes | 0.02436 | 0.004809 | 16 |
| WR | mobility_velocity_and_resistance | yes | 0.9669 | 0.1887 | 18 |
| U0 | mobility_velocity_and_resistance | yes | 6.214 | 0.9054 | 18 |
| A0 | mobility_velocity_and_resistance | yes | 1.916 | 0.3362 | 18 |
| KETA | mobility_velocity_and_resistance | no | 4.327e-06 | 4.327e-06 | 0 |
| A1 | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| A2 | mobility_velocity_and_resistance | yes | 0.5075 | 0.2649 | 18 |
| AGS | mobility_velocity_and_resistance | yes | 0.143 | 0.08907 | 17 |
| B0 | mobility_velocity_and_resistance | yes | 0.7192 | 0.1285 | 18 |
| B1 | mobility_velocity_and_resistance | yes | 0.1295 | 0.03202 | 3 |
| EU | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| UP | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| LP | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| UD | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| RDSWMIN | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| RDW | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| RDWMIN | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| RSW | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| RSWMIN | mobility_velocity_and_resistance | no | 0 | 0 | 0 |
| VOFF | subthreshold_and_dibl | yes | 0.05166 | 0.0005823 | 5 |
| NFACTOR | subthreshold_and_dibl | yes | 2.002 | 0.0008168 | 15 |
| TVFBSDOFF | subthreshold_and_dibl | no | 0 | 0 | 0 |
| TVOFF | subthreshold_and_dibl | no | 0 | 0 | 0 |
| CIT | subthreshold_and_dibl | no | 4.711e-05 | 6.966e-07 | 0 |
| CDSC | subthreshold_and_dibl | no | 4.145e-05 | 4.738e-08 | 0 |
| CDSCB | subthreshold_and_dibl | no | 2.65e-08 | 1.37e-10 | 0 |
| CDSCD | subthreshold_and_dibl | no | 9.085e-05 | 1.495e-06 | 0 |
| ETA0 | subthreshold_and_dibl | yes | 0.2601 | 0.0526 | 5 |
| ETAB | subthreshold_and_dibl | no | 8.476e-06 | 8.476e-06 | 0 |
| DSUB | subthreshold_and_dibl | yes | 3.601 | 0.2707 | 6 |
| VOFFL | subthreshold_and_dibl | no | 0 | 0 | 0 |
| MINV | subthreshold_and_dibl | no | 0 | 0 | 0 |
| PCLM | output_conductance_and_high_field | yes | 0.1859 | 0.1855 | 17 |
| PDIBLC1 | output_conductance_and_high_field | yes | 0.05797 | 0.008035 | 5 |
| PDIBLC2 | output_conductance_and_high_field | yes | 2.617 | 0.3628 | 18 |
| PDIBLCB | output_conductance_and_high_field | no | 5.299e-07 | 2.659e-07 | 0 |
| DROUT | output_conductance_and_high_field | no | 3.84e-05 | 1.123e-05 | 0 |
| PSCBE1 | output_conductance_and_high_field | no | 6.314e-06 | 6.314e-06 | 0 |
| PSCBE2 | output_conductance_and_high_field | no | 3.085e-07 | 3.085e-07 | 0 |
| PVAG | output_conductance_and_high_field | no | 0 | 0 | 0 |
| DELTA | output_conductance_and_high_field | yes | 0.9131 | 0.04948 | 18 |
| ALPHA0 | output_conductance_and_high_field | yes | 0.0006134 | 0.0006134 | 3 |
| ALPHA1 | output_conductance_and_high_field | no | 0 | 0 | 0 |
| BETA0 | output_conductance_and_high_field | yes | 0.005906 | 0.005906 | 6 |
| FPROUT | output_conductance_and_high_field | no | 0 | 0 | 0 |
| PDITS | output_conductance_and_high_field | no | 0 | 0 | 0 |
| PDITSL | output_conductance_and_high_field | no | 0 | 0 | 0 |
| PDITSD | output_conductance_and_high_field | no | 0 | 0 | 0 |
| AGIDL | gidl | no | 0 | 0 | 0 |
| BGIDL | gidl | no | 0 | 0 | 0 |
| CGIDL | gidl | no | 0 | 0 | 0 |
| EGIDL | gidl | no | 0 | 0 | 0 |
| VFBSDOFF | temperature | no | 0 | 0 | 0 |
| KT1 | temperature | yes | 130.8 | 1.011 | 18 |
| KT2 | temperature | no | 9.186e-06 | 4.641e-06 | 0 |
| AT | temperature | yes | 0.6507 | 0.289 | 17 |
| UTE | temperature | yes | 2.341 | 0.3879 | 18 |
| UA1 | temperature | yes | 4.564 | 0.7701 | 18 |
| UB1 | temperature | yes | 3.706 | 0.6249 | 18 |
| UC1 | temperature | no | 2.042e-06 | 2.042e-06 | 0 |
| KT1L | temperature | no | 0 | 0 | 0 |
| PRT | temperature | no | 0 | 0 | 0 |
| RBPB | body_network_and_junction_dc | no | 8.412e-07 | 8.412e-07 | 0 |
| RBPD | body_network_and_junction_dc | no | 2.077e-07 | 2.077e-07 | 0 |
| RBPS | body_network_and_junction_dc | no | 2.077e-07 | 2.077e-07 | 0 |
| RBDB | body_network_and_junction_dc | no | 2.077e-07 | 2.077e-07 | 0 |
| RBSB | body_network_and_junction_dc | no | 2.077e-07 | 2.077e-07 | 0 |
| GBMIN | body_network_and_junction_dc | no | 3.928e-29 | 0 | 0 |
| NJS | body_network_and_junction_dc | no | 4.763e-08 | 3.132e-45 | 0 |
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
