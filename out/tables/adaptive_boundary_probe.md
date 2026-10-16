# Adaptive parameter-bound probe

A bound expansion is recommended only when the same parameter, polarity, and direction improves at least the configured number of distinct devices without worsening any admitted sweep section by more than the guard.

| Parameter | Polarity | Direction | Boundary devices | Improving devices | Recommend | Supported multiplier | Best objective delta | Worst section delta |
|---|---|---|---:|---:|:---:|---:|---:|---:|
| ags | pmos | upper | 3 | 3 | yes | 2 | -0.0209351 | 0.0538365 |
| k2 | pmos | lower | 2 | 2 | yes | 4 | -0.00538351 | 0.0158987 |
| lpeb | nmos | lower | 2 | 2 | yes | 2 | -0.000467733 | 0.0171058 |
| prwg | pmos | upper | 2 | 2 | yes | 2 | -0.00171577 | 0.0150114 |
| xj | nmos | lower | 3 | 2 | no | n/a | -0.000313123 | 0.00604428 |
| a2 | pmos | upper | 1 | 1 | no | n/a | -0.00169883 | 0.00686239 |
| delta | pmos | upper | 2 | 1 | no | n/a | -0.00164917 | 0.0468921 |
| dsub | pmos | lower | 2 | 1 | no | n/a | -0.00026162 | 0.00447085 |
| dwg | nmos | lower | 2 | 1 | no | n/a | -0.00074092 | 0.00451431 |
| dwg | pmos | lower | 1 | 1 | no | n/a | -0.00078126 | 0.00572955 |
| k3 | pmos | lower | 1 | 1 | no | n/a | -0.000512421 | 0.00950744 |
| ndep | nmos | upper | 1 | 1 | no | n/a | -0.000182856 | 0.00707278 |
| nfactor | nmos | upper | 2 | 1 | no | n/a | -0.000309169 | 0.00608495 |
| nfactor | pmos | upper | 2 | 1 | no | n/a | -0.0012868 | 0.0465947 |
| nsd | nmos | lower | 1 | 1 | no | n/a | -0.000454282 | 0.00974731 |
| pclm | pmos | upper | 1 | 1 | no | n/a | -0.000226166 | 0.000573862 |
| pdiblc2 | nmos | lower | 3 | 1 | no | n/a | -0.00133995 | 0.0198924 |
| vfb | pmos | lower | 1 | 1 | no | n/a | -0.000592659 | 0.00527002 |
| voff | nmos | lower | 3 | 1 | no | n/a | -0.00512491 | 0.160447 |
| w0 | pmos | upper | 3 | 1 | no | n/a | -0.00016578 | 0.00772375 |
| wr | nmos | lower | 2 | 1 | no | n/a | -0.000166591 | 0.0192914 |
| a0 | nmos | upper | 3 | 0 | no | n/a | 0.000662609 | 0.0244253 |
| a0 | pmos | lower | 3 | 0 | no | n/a | -0.0141341 | 0.112984 |
| a0 | pmos | upper | 1 | 0 | no | n/a | 0.000756975 | 0.0201594 |
| a2 | nmos | upper | 2 | 0 | no | n/a | 0.00106144 | 0.040185 |
| ags | nmos | lower | 1 | 0 | no | n/a | 0.000482492 | 0.0256136 |
| alpha0 | nmos | lower | 1 | 0 | no | n/a | -1.15832e-05 | 7.04924e-10 |
| alpha0 | nmos | upper | 1 | 0 | no | n/a | 0.00127109 | 0.0150542 |
| alpha0 | pmos | lower | 2 | 0 | no | n/a | -4.80267e-10 | 4.11542e-10 |
| alpha0 | pmos | upper | 1 | 0 | no | n/a | 5.10565e-11 | 2.36364e-09 |
| b0 | pmos | lower | 1 | 0 | no | n/a | -0.00601837 | 0.0534029 |
| b0 | pmos | upper | 1 | 0 | no | n/a | 0.00062874 | 0.00291802 |
| beta0 | nmos | lower | 1 | 0 | no | n/a | 7.56332e-05 | 0.000742144 |
| beta0 | nmos | upper | 1 | 0 | no | n/a | -1.01168e-06 | 4.30969e-05 |
| beta0 | pmos | upper | 1 | 0 | no | n/a | 2.59837e-11 | 1.66978e-09 |
| cdscd | nmos | lower | 1 | 0 | no | n/a | 0 | 0 |
| cdscd | nmos | upper | 2 | 0 | no | n/a | -1.86081e-06 | 8.49007e-07 |
| cdscd | pmos | lower | 1 | 0 | no | n/a | -6.17062e-12 | 4.13365e-11 |
| cdscd | pmos | upper | 1 | 0 | no | n/a | 0 | 0 |
| cit | nmos | lower | 1 | 0 | no | n/a | -1.37082e-08 | 2.41218e-07 |
| cit | nmos | upper | 1 | 0 | no | n/a | -5.95667e-08 | 9.54205e-07 |
| cit | pmos | lower | 1 | 0 | no | n/a | 1.39323e-11 | 3.21218e-10 |
| cit | pmos | upper | 1 | 0 | no | n/a | -4.29594e-11 | 8.47051e-10 |
| delta | nmos | lower | 4 | 0 | no | n/a | -0.00211381 | 0.0432018 |
| drout | nmos | lower | 1 | 0 | no | n/a | 0 | 0 |
| drout | pmos | upper | 1 | 0 | no | n/a | 0 | 0 |
| dsub | nmos | upper | 1 | 0 | no | n/a | 3.05334e-07 | 1.88366e-05 |
| dsub | pmos | upper | 1 | 0 | no | n/a | 6.97566e-10 | 8.30694e-08 |
| dvt0 | nmos | upper | 2 | 0 | no | n/a | 0 | 0 |
| dvt0 | pmos | upper | 1 | 0 | no | n/a | 0.00500929 | 0.155748 |
| dvt1 | nmos | lower | 1 | 0 | no | n/a | 0.0314967 | 0.270678 |
| dvt1 | pmos | upper | 1 | 0 | no | n/a | 2.16407e-10 | 3.61306e-09 |
| dvt1w | nmos | lower | 1 | 0 | no | n/a | 0 | 0 |
| dvt1w | nmos | upper | 1 | 0 | no | n/a | 0 | 0 |
| dvt1w | pmos | upper | 1 | 0 | no | n/a | 5.71988e-08 | 2.0198e-06 |
| eta0 | nmos | lower | 3 | 0 | no | n/a | 0 | 0 |
| eta0 | nmos | upper | 1 | 0 | no | n/a | 2.20832e-09 | 3.82111e-08 |
| eta0 | pmos | lower | 2 | 0 | no | n/a | 0 | 0 |
| k1 | nmos | upper | 2 | 0 | no | n/a | 0.00575163 | 0.0579922 |
| k1 | pmos | lower | 1 | 0 | no | n/a | -0.0145162 | 0.107487 |
| k1 | pmos | upper | 1 | 0 | no | n/a | 0.00456407 | 0.0539701 |
| k3 | nmos | upper | 1 | 0 | no | n/a | 0.000370599 | 0.00872147 |
| k3 | pmos | upper | 1 | 0 | no | n/a | 0.00562585 | 0.0931692 |
| k3b | nmos | lower | 1 | 0 | no | n/a | 1.00669e-05 | 0.000203982 |
| k3b | nmos | upper | 1 | 0 | no | n/a | -8.84593e-06 | 0.000859503 |
| k3b | pmos | lower | 1 | 0 | no | n/a | 4.36124e-05 | 0.000484165 |
| lpe0 | nmos | lower | 1 | 0 | no | n/a | 0.0879791 | 0.835169 |
| lpe0 | nmos | upper | 4 | 0 | no | n/a | 0.000232043 | 0.0120749 |
| ndep | nmos | lower | 1 | 0 | no | n/a | -1.25222e-05 | 0.0015427 |
| ndep | pmos | upper | 2 | 0 | no | n/a | 0.00414608 | 0.0728588 |
| nfactor | nmos | lower | 2 | 0 | no | n/a | -0.00448499 | 0.131626 |
| nsd | pmos | lower | 1 | 0 | no | n/a | 0.000123398 | 0.00418157 |
| pclm | nmos | lower | 2 | 0 | no | n/a | 0.0191263 | 0.0769923 |
| pclm | pmos | lower | 1 | 0 | no | n/a | 0.000416647 | 0.00325054 |
| pdiblc2 | nmos | upper | 1 | 0 | no | n/a | 0.00202567 | 0.023587 |
| pdiblc2 | pmos | lower | 1 | 0 | no | n/a | 0.00015973 | 0.00307493 |
| pdiblc2 | pmos | upper | 2 | 0 | no | n/a | -2.88451e-05 | 0.0018709 |
| rdsw | nmos | lower | 2 | 0 | no | n/a | 8.50152e-08 | 2.67815e-06 |
| rdsw | nmos | upper | 1 | 0 | no | n/a | 0.00196399 | 0.0262409 |
| rdsw | pmos | upper | 3 | 0 | no | n/a | 0.0110911 | 0.138741 |
| rsh | nmos | upper | 1 | 0 | no | n/a | -2.11712e-05 | 0.000149522 |
| rsh | pmos | upper | 2 | 0 | no | n/a | -5.79234e-08 | 4.63252e-05 |
| toxe | nmos | lower | 1 | 0 | no | n/a | 0.579655 | 3.96544 |
| toxe | pmos | upper | 1 | 0 | no | n/a | 0.228093 | 0.69955 |
| toxm | nmos | lower | 1 | 0 | no | n/a | 0.211123 | 0.839563 |
| toxm | pmos | lower | 1 | 0 | no | n/a | 0.233894 | 1.28743 |
| toxm | pmos | upper | 1 | 0 | no | n/a | 0.226712 | 1.61986 |
| u0 | nmos | upper | 3 | 0 | no | n/a | 0.00660559 | 0.0580089 |
| u0 | pmos | lower | 2 | 0 | no | n/a | 0.012958 | 0.100646 |
| u0 | pmos | upper | 1 | 0 | no | n/a | 0.00732478 | 0.094009 |
| ua | nmos | lower | 3 | 0 | no | n/a | 0.00771554 | 0.0636973 |
| ua | nmos | upper | 1 | 0 | no | n/a | 0.00979531 | 0.0847981 |
| ua | pmos | upper | 2 | 0 | no | n/a | 0.0338588 | 0.190848 |
| vfb | pmos | upper | 1 | 0 | no | n/a | 5.60298e-06 | 0.00135679 |
| voff | pmos | upper | 2 | 0 | no | n/a | 5.02718e-08 | 9.64222e-07 |
| vsat | nmos | lower | 1 | 0 | no | n/a | 0.0388181 | 0.139827 |
| vsat | nmos | upper | 2 | 0 | no | n/a | -1.54511e-06 | 1.22858e-05 |
| vsat | pmos | lower | 2 | 0 | no | n/a | 5.38465e-07 | 2.36168e-06 |
| vsat | pmos | upper | 1 | 0 | no | n/a | -3.45029e-05 | 0.000130462 |
| w0 | nmos | lower | 1 | 0 | no | n/a | 1.76112e-05 | 0.00052776 |
| wr | nmos | upper | 2 | 0 | no | n/a | 3.28296e-08 | 1.0322e-06 |
| wr | pmos | lower | 1 | 0 | no | n/a | 0.0382255 | 0.177796 |
| wr | pmos | upper | 1 | 0 | no | n/a | 0.011039 | 0.138306 |
| xj | pmos | lower | 1 | 0 | no | n/a | 0.000731058 | 0.00804293 |

## Probe counts

- Requested: 318
- Finite: 317

