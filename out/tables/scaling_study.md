# All-18 training-data scaling study

Only configurations completed for all 18 transistors enter these arithmetic means. The unfinished 10,000-sample/capacity/search rows remain in the CSV for audit and are excluded. Capacity and search results are retained only as the clearly labeled four-device pilot.

| samples/transistor | emulator test MSE | raw NGSpice RRMS | NGSpice RRMS after FD |
|---:|---:|---:|---:|
| 375 | 0.0049619 | 0.2497 | 0.2280 |
| 750 | 0.0032794 | 0.2482 | 0.2270 |
| 1500 | 0.0016696 | 0.2394 | 0.2273 |
| 3000 | 0.0010724 | 0.2391 | 0.2280 |
| 6000 | 0.0005243 | 0.2398 | 0.2294 |

Held-out emulator MSE improves with fitted power exponent `0.810`, but the best real-NGSpice+FD mean is `0.2270` at only `750` samples/transistor. Additional data does not improve the final physical fit, so the remaining grid was stopped and compute was redirected to the fixed-method comparisons.
