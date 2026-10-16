# Forward-surrogate held-out comparison

Only high/ordinary normalized-linear MSE is compared directly across losses. The complete original and regional objectives have different definitions.

| Surface | Metric | Original | Regional | Regional/original |
|---|---|---:|---:|---:|
| seven | high_linear | 2.12243e-07 | 1.36735e-07 | 0.6442 |
| seven | ordinary_linear | 7.49557e-07 | 3.91668e-07 | 0.5225 |
| full43 | high_linear | 5.42632e-06 | 1.63922e-06 | 0.3021 |
| full43 | ordinary_linear | 0.000840957 | 0.00018805 | 0.2236 |

## Per-curve normalized-linear MSE

### seven

| Curve | Original | Regional | Ratio |
|---|---:|---:|---:|
| idvd@0.37 | 6.2803e-07 | 2.38102e-07 | 0.3791 |
| idvd@0.74 | 3.30842e-06 | 1.29732e-06 | 0.3921 |
| idvd@1.11 | 6.10773e-07 | 4.9533e-07 | 0.8110 |
| idvd@1.48 | 3.05199e-07 | 2.32043e-07 | 0.7603 |
| idvd@1.85 | 3.32649e-07 | 1.71317e-07 | 0.5150 |
| idvg@0.01 | 3.48699e-07 | 4.76509e-07 | 1.3665 |
| idvg@0.37 | 1.47472e-07 | 1.08781e-07 | 0.7376 |
| idvg@0.74 | 1.0439e-07 | 6.58494e-08 | 0.6308 |
| idvg@1.11 | 9.91138e-08 | 5.97818e-08 | 0.6032 |
| idvg@1.48 | 1.02364e-07 | 6.64355e-08 | 0.6490 |
| idvg@1.85 | 1.08761e-07 | 7.71435e-08 | 0.7093 |

### full43

| Curve | Original | Regional | Ratio |
|---|---:|---:|---:|
| idvd@0.37 | 0.00581476 | 0.00127466 | 0.2192 |
| idvd@0.74 | 4.23994e-05 | 2.78039e-05 | 0.6558 |
| idvd@1.11 | 1.04925e-05 | 6.87199e-06 | 0.6549 |
| idvd@1.48 | 7.29957e-06 | 2.29033e-06 | 0.3138 |
| idvd@1.85 | 8.14797e-06 | 2.16195e-06 | 0.2653 |
| idvg@0.01 | 9.55842e-06 | 3.8417e-06 | 0.4019 |
| idvg@0.37 | 3.7114e-06 | 1.27988e-06 | 0.3449 |
| idvg@0.74 | 2.89791e-06 | 9.57748e-07 | 0.3305 |
| idvg@1.11 | 2.87744e-06 | 9.35267e-07 | 0.3250 |
| idvg@1.48 | 2.98821e-06 | 9.90204e-07 | 0.3314 |
| idvg@1.85 | 3.26954e-06 | 1.1144e-06 | 0.3408 |

## Held-out normalized-linear MSE by polarity and sweep section

### seven

| Partition | Original | Regional | Ratio |
|---|---:|---:|---:|
| nmos | 3.35581e-07 | 1.8276e-07 | 0.5446 |
| pmos | 7.26868e-07 | 3.91846e-07 | 0.5391 |
| normalized sweep section 1 | 3.50963e-07 | 1.9536e-07 | 0.5566 |
| normalized sweep section 2 | 5.69839e-07 | 1.80108e-07 | 0.3161 |
| normalized sweep section 3 | 4.81841e-07 | 3.43707e-07 | 0.7133 |
| normalized sweep section 4 | 7.46398e-07 | 4.01052e-07 | 0.5373 |
| nmos section 1 | 1.12024e-07 | 9.48986e-08 | 0.8471 |
| nmos section 2 | 2.14714e-07 | 1.19644e-07 | 0.5572 |
| nmos section 3 | 3.64772e-07 | 2.05435e-07 | 0.5632 |
| nmos section 4 | 5.33957e-07 | 2.59837e-07 | 0.4866 |
| pmos section 1 | 5.42115e-07 | 2.75729e-07 | 0.5086 |
| pmos section 2 | 8.53939e-07 | 2.2848e-07 | 0.2676 |
| pmos section 3 | 5.75495e-07 | 4.54325e-07 | 0.7894 |
| pmos section 4 | 9.1635e-07 | 5.14024e-07 | 0.5609 |

### full43

| Partition | Original | Regional | Ratio |
|---|---:|---:|---:|
| nmos | 0.00119753 | 0.000264621 | 0.2210 |
| pmos | 8.80485e-06 | 4.77644e-06 | 0.5425 |
| normalized sweep section 1 | 1.85531e-05 | 6.34164e-06 | 0.3418 |
| normalized sweep section 2 | 4.35714e-05 | 1.70589e-05 | 0.3915 |
| normalized sweep section 3 | 0.000140116 | 6.9486e-05 | 0.4959 |
| normalized sweep section 4 | 0.00160286 | 0.000314498 | 0.1962 |
| nmos section 1 | 3.71501e-05 | 9.75174e-06 | 0.2625 |
| nmos section 2 | 9.16085e-05 | 3.46632e-05 | 0.3784 |
| nmos section 3 | 0.000305665 | 0.000150558 | 0.4926 |
| nmos section 4 | 0.00358676 | 0.000699017 | 0.1949 |
| pmos section 1 | 3.67551e-06 | 3.61356e-06 | 0.9831 |
| pmos section 2 | 5.14161e-06 | 2.97553e-06 | 0.5787 |
| pmos section 3 | 7.67621e-06 | 4.62836e-06 | 0.6029 |
| pmos section 4 | 1.57429e-05 | 6.88273e-06 | 0.4372 |

