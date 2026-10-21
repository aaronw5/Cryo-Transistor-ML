# I9-1 inventory: NIST mds2-2997 MPW-5, nfet_01v8 / pfet_01v8 (4 K chuck temperature)

Sources: per-device rows `data/manifest.json` (this item); every-file parse and checks of the tile `out/research/explore/I8_1/inventory.json`; cards `cards.json`.

| family | folder | devices | W (um) | L (um) | bad / readme | bins in card | geometries in a bin | included curves per device (count) | excluded (D12) | DEV | TEST |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nfet_01v8 | nthin | 80 | 0.36-7 | 0.15-20 | 0 / 0 | 180 | 80 | 6: 80 | 0 | 48 | 32 |
| pfet_01v8 | pthin | 71 | 0.42-7 | 0.15-20 | 0 / 0 | 108 | 71 | 0: 2, 1: 4, 2: 3, 3: 10, 4: 28, 5: 22, 6: 2 | 6 | 39 | 26 |

pthin readme: position 17_6 'gate not connected' (no files). Gate-control ratio minimum (Id-Vd, Vb = 0): nfet_01v8 5.39e+07, pfet_01v8 4.28e+05.
Bias grid: Id-Vg at |Vd| 0.1, 1.8 V (Vb 0 and 4 further values), 181 points; Id-Vd at |Vg| 0-1.8 V in 0.3 V steps (Vb 0 and 1 V magnitude), 91 points. I9-1 uses the 9 Vb = 0 curves.
