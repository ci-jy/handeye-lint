# Benchmark results

Test split: 2000 scenarios (seed 0); lint thresholds chosen on a separate calibration split of 1000 scenarios (seed 1).

## Family: all (n=2000)

| method | median_rot_deg | p95_rot_deg | median_trans_mm | p95_trans_mm | failures |
|---|---|---|---|---|---|
| tsai | 1.92 | 127 | 17.2 | 704 | 0 |
| park | 1.22 | 132 | 11.5 | 768 | 0 |
| horaud | 1.2 | 133 | 11.4 | 664 | 0 |
| andreff | 0.86 | 127 | 84.9 | 512 | 1 |
| daniilidis | 1.38 | 143 | 22.9 | 1.69e+03 | 7 |
| robust | 0.265 | 121 | 2.51 | 547 | 0 |
| lint+robust | 0.176 | 1.3 | 1.46 | 35.4 | 0 |

## Family: convention (n=315)

| method | median_rot_deg | p95_rot_deg | median_trans_mm | p95_trans_mm | failures |
|---|---|---|---|---|---|
| tsai | 87.8 | 166 | 434 | 1.29e+03 | 0 |
| park | 108 | 173 | 437 | 1.29e+03 | 0 |
| horaud | 108 | 173 | 424 | 1.29e+03 | 0 |
| andreff | 102 | 174 | 451 | 1.31e+03 | 0 |
| daniilidis | 107 | 173 | 661 | 2e+03 | 2 |
| robust | 102 | 171 | 318 | 1.35e+03 | 0 |
| lint+robust | 0.109 | 0.274 | 0.855 | 2.01 | 0 |

## Family: degenerate (n=176)

| method | median_rot_deg | p95_rot_deg | median_trans_mm | p95_trans_mm | failures |
|---|---|---|---|---|---|
| tsai | 8.35 | 125 | 110 | 1.25e+03 | 0 |
| park | 5.98 | 146 | 71.6 | 5.21e+03 | 0 |
| horaud | 5.91 | 144 | 71.9 | 2.16e+03 | 0 |
| andreff | 1.16 | 3.2 | 96.2 | 551 | 0 |
| daniilidis | 1.56 | 14.9 | 100 | 1.24e+04 | 1 |
| robust | 0.565 | 1.84 | 39.7 | 541 | 0 |
| lint+robust | 0.565 | 1.84 | 39.7 | 541 | 0 |

## Family: diversity (n=405)

| method | median_rot_deg | p95_rot_deg | median_trans_mm | p95_trans_mm | failures |
|---|---|---|---|---|---|
| tsai | 0.296 | 44.5 | 2.29 | 100 | 0 |
| park | 0.231 | 0.857 | 2.13 | 6.52 | 0 |
| horaud | 0.236 | 0.858 | 2.13 | 6.52 | 0 |
| andreff | 0.213 | 0.767 | 3.41 | 7.31 | 0 |
| daniilidis | 0.211 | 0.868 | 2.67 | 10.5 | 0 |
| robust | 0.187 | 0.518 | 1.53 | 5.85 | 0 |
| lint+robust | 0.187 | 0.518 | 1.53 | 5.85 | 0 |

## Family: noise (n=619)

| method | median_rot_deg | p95_rot_deg | median_trans_mm | p95_trans_mm | failures |
|---|---|---|---|---|---|
| tsai | 0.233 | 2.49 | 1.81 | 15.5 | 0 |
| park | 0.2 | 1.77 | 1.7 | 13 | 0 |
| horaud | 0.194 | 1.8 | 1.69 | 13 | 0 |
| andreff | 0.193 | 1.67 | 10.9 | 179 | 0 |
| daniilidis | 0.212 | 5.11 | 1.95 | 69.6 | 1 |
| robust | 0.188 | 1.67 | 1.56 | 11.8 | 0 |
| lint+robust | 0.188 | 1.67 | 1.56 | 11.8 | 0 |

## Family: outliers (n=485)

| method | median_rot_deg | p95_rot_deg | median_trans_mm | p95_trans_mm | failures |
|---|---|---|---|---|---|
| tsai | 11.5 | 98.8 | 77.9 | 503 | 0 |
| park | 6.8 | 25.4 | 60 | 178 | 0 |
| horaud | 7.13 | 24.7 | 59.9 | 176 | 0 |
| andreff | 8.31 | 30.5 | 427 | 494 | 1 |
| daniilidis | 18.6 | 147 | 307 | 2.45e+03 | 3 |
| robust | 0.179 | 1.23 | 1.5 | 12.9 | 0 |
| lint+robust | 0.179 | 1.23 | 1.5 | 12.9 | 0 |

## Lint warnings on held-out planted faults

| warning | unit | threshold | precision | recall | tp | fp | fn | positives | negatives |
|---|---|---|---|---|---|---|---|---|---|
| degenerate-rotation-axes | scenario | 0.0522 | 1.000 | 1.000 | 176 | 0 | 0 | 176 | 1824 |
| convention-mismatch | scenario | 0.626 | 1.000 | 1.000 | 315 | 0 | 0 | 315 | 1685 |
| outlier-samples | sample | 4.17 | 1.000 | 0.997 | 2060 | 0 | 7 | 2067 | 32851 |

Convention fault identified correctly (detected cases): 1.000
