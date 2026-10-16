# Adaptive polarity-general parameter bounds

Only repeated physical recommendations are expanded; each supported parameter/polarity/direction is applied to every device of that polarity, including devices that did not trigger the diagnosis.

| Parameter | Polarity | Direction | Multiplier | Evidence devices |
|---|---|---|---:|---|
| ags | pmos | upper | 2 | pmos_L0p35_W0p55, pmos_L4_W7, pmos_L8_W5 |
| k2 | pmos | lower | 4 | pmos_L0p5_W0p42, pmos_L4_W7 |
| lpeb | nmos | lower | 2 | nmos_L0p19_W7, nmos_L0p25_W1p6 |
| prwg | pmos | upper | 2 | pmos_L0p5_W0p42, pmos_L4_W7 |

Expanded device coordinates: 38.

## Coverage widening

symmetric widening of named parameters on every device of every polarity, applied for surrogate training coverage after direct NGSpice probes showed the required values lie outside the one-at-a-time box; it is not evidence that the wider values improve a card

| Parameter | Multiplier |
|---|---:|
| delta | 8 |
| ags | 8 |

Widened device coordinates: 72.

