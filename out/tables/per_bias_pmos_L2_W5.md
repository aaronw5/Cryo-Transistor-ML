# Per-voltage diagnostic: pMOS L=2 um, W=5 um

This diagnostic permits a different seven-parameter card for every fixed-bias curve. Each row selects the best of 10,000 saved NGSpice LHC samples, then reruns NGSpice at those parameters. The result is an upper-bound diagnostic, not a physically consistent or deployable compact model.

| curve | in official device mean | paper | per-device surrogate + FD | foundation + FD | separate voltage card |
|---|:---:|---:|---:|---:|---:|
| idvd@1.85 | yes | 0.0109 | 0.0938 | 0.0470 | 0.0046 |
| idvg@0.37 | yes | 0.0981 | 0.0760 | 0.0703 | 0.0631 |
| idvg@1.48 | yes | 0.0506 | 0.0776 | 0.0444 | 0.0326 |
| idvg@1.11 | yes | 0.0492 | 0.0835 | 0.0485 | 0.0348 |
| idvd@0.74 | no | n/a | n/a | n/a | n/a |
| idvd@1.11 | yes | 0.2123 | 0.2341 | 0.1800 | 0.1696 |
| idvg@0.74 | yes | 0.0554 | 0.0823 | 0.0504 | 0.0395 |
| idvg@1.85 | yes | 0.0564 | 0.0759 | 0.0435 | 0.0327 |
| idvd@0.37 | no | n/a | n/a | n/a | n/a |
| idvd@1.48 | yes | 0.0414 | 0.0344 | 0.0397 | 0.0338 |
| idvg@0.01 | yes | 1.1418 | 0.9522 | 1.0581 | 0.5592 |

## Included-Curve Mean

- Paper card: 0.1907
- Per-device surrogate + FD: 0.1900
- Foundation + FD: 0.1758
- Separate voltage cards: 0.1078

The difference between the one-card and separate-card results quantifies cross-bias compromise. It must not be interpreted as a deployable transistor model improvement.
