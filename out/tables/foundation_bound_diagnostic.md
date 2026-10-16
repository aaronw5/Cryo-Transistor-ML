# Foundation parameter-bound diagnostic

Threshold: `|local| >= 0.95`. Observed 168 hits across 18 devices and 43 parameters.

A hit is a diagnostic, not automatic permission to widen a bound. Outward NGSpice probes must improve multiple independent sweep sections before adaptive data are generated.

| Parameter | Total | nMOS | pMOS | Lower | Upper | Repeated |
|---|---:|---:|---:|---:|---:|:---:|
| alpha0 | 7 | 4 | 3 | 5 | 2 | yes |
| pdiblc2 | 7 | 5 | 2 | 5 | 2 | yes |
| rdsw | 7 | 4 | 3 | 3 | 4 | yes |
| u0 | 7 | 4 | 3 | 2 | 5 | yes |
| ua | 7 | 5 | 2 | 4 | 3 | yes |
| a0 | 6 | 3 | 3 | 2 | 4 | yes |
| delta | 6 | 4 | 2 | 4 | 2 | yes |
| eta0 | 6 | 4 | 2 | 5 | 1 | yes |
| nfactor | 6 | 4 | 2 | 2 | 4 | yes |
| ags | 5 | 2 | 3 | 2 | 3 | yes |
| cdscd | 5 | 3 | 2 | 2 | 3 | yes |
| cit | 5 | 3 | 2 | 3 | 2 | yes |
| lpe0 | 5 | 5 | 0 | 1 | 4 | yes |
| ndep | 5 | 3 | 2 | 2 | 3 | yes |
| voff | 5 | 3 | 2 | 3 | 2 | yes |
| vsat | 5 | 2 | 3 | 3 | 2 | yes |
| wr | 5 | 3 | 2 | 2 | 3 | yes |
| dsub | 4 | 1 | 3 | 2 | 2 | yes |
| dwg | 4 | 3 | 1 | 4 | 0 | yes |
| k1 | 4 | 2 | 2 | 1 | 3 | yes |
| pclm | 4 | 2 | 2 | 3 | 1 | yes |
| toxm | 4 | 2 | 2 | 3 | 1 | yes |
| w0 | 4 | 2 | 2 | 2 | 2 | yes |
| xj | 4 | 3 | 1 | 4 | 0 | yes |
| a2 | 3 | 1 | 2 | 1 | 2 | yes |
| beta0 | 3 | 2 | 1 | 1 | 2 | yes |
| dvt0 | 3 | 2 | 1 | 0 | 3 | yes |
| dvt1w | 3 | 2 | 1 | 1 | 2 | yes |
| k2 | 3 | 1 | 2 | 2 | 1 | yes |
| k3 | 3 | 1 | 2 | 1 | 2 | yes |
| k3b | 3 | 2 | 1 | 2 | 1 | yes |
| rsh | 3 | 1 | 2 | 0 | 3 | yes |
| b0 | 2 | 0 | 2 | 1 | 1 | yes |
| drout | 2 | 1 | 1 | 1 | 1 | yes |
| dvt1 | 2 | 1 | 1 | 1 | 1 | yes |
| lpeb | 2 | 2 | 0 | 2 | 0 | yes |
| nsd | 2 | 1 | 1 | 2 | 0 | yes |
| prwg | 2 | 0 | 2 | 0 | 2 | yes |
| toxe | 2 | 1 | 1 | 1 | 1 | yes |
| vfb | 2 | 0 | 2 | 1 | 1 | yes |
| vth0 | 1 | 1 | 0 | 0 | 1 | no |
| b1 | 0 | 0 | 0 | 0 | 0 | no |
| ub | 0 | 0 | 0 | 0 | 0 | no |

## Selected parameters inactive across a polarity

- nmos: `vfb`, `prwg`, `b0`, `b1`
- pmos: `lpe0`, `lpeb`

## nMOS L1/W3 boundary hits

| Parameter | Direction | Value | Bounds | Local |
|---|---|---:|---|---:|
| lpe0 | upper | 2.61295e-07 | [2.1421800000000002e-07, 2.61822e-07] | 0.97787 |
| vsat | upper | -3.04142 | [-3.717296, -3.041424] | 1.00000 |
| rdsw | upper | 317.062 | [259.4142, 317.0618] | 1.00000 |
| eta0 | lower | 0.00361549 | [0.0036151559999999996, 0.004418524] | -0.99916 |
| pdiblc2 | lower | 0.00721886 | [0.007218846000000001, 0.008823034] | -0.99998 |
