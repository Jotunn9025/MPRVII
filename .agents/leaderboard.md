# Onboarding Knowledge — Detection Leaderboard (M4)

> **Generated** by `uv run python -m bhel_optimize.detection.train` — do not hand-edit. Metrics measured on the held-out test scenario series (injected faults) unless stated otherwise.

- Lookback: 60 samples (300 s) | epochs: 20 | max windows: 10000 | target false-alarm rate: 5.0%
- Candidate detectors: ewma_mad, mean_shift, staleness, dense_ae, lstm_residual, tcn_ae, isolation_forest, ocsvm
- Accuracy caveat: the dataset has **no per-part fault labels**, so per-part accuracy is measured against **injected simulated faults** (plus the one real global label below).
- Positives = samples with any material injection (>= 0.1 IQR); FAR is measured only on completely clean samples; recall is reported per severity band because creeping faults are only detectable once they exceed the noise floor. Real 'tailed' rows and negligible injections are excluded from per-part metrics.

- AUROC confidence intervals are **within-series bootstrap** intervals: they capture sampling noise inside one evaluation series only. Running the trainer with a different block length moved some parts by >0.1 AUROC (average across parts changed by <0.02), so treat part-level numbers as indicative of rank, not as precise constants.

## Summary (chosen detector/ensemble per part)

| Part | Crit | Chosen | val AUROC | test AUROC (95% CI) | F1 | Prec | Rec | FAR(clean) | Rec≥0.1 | Rec≥0.5 | Rec≥1.0 | Latency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Furnace Chamber (Upper) | 5 | `ensemble` | 0.709 | 0.683 [0.66-0.71] | 0.172 | 0.785 | 0.097 | 0.014 | 0.000 | 0.148 | 0.824 | 126 s |
| Steam Drum & Pot | 5 | `isolation_forest` | 0.712 | 0.618 [0.59-0.64] | 0.136 | 0.545 | 0.078 | 0.040 | 0.000 | 0.090 | – | 400 s |
| Main Steam Line & Outlet | 5 | `isolation_forest` | 0.850 | 0.845 [0.83-0.86] | 0.227 | 1.000 | 0.128 | 0.000 | 0.000 | 0.000 | 0.237 | 112 s |
| Superheater & Desuperheater | 4 | `lstm_residual` | 0.784 | 0.743 [0.72-0.77] | 0.499 | 0.628 | 0.415 | 0.153 | 0.347 | 1.000 | – | 30 s |
| Economizer (Upper) | 3 | `isolation_forest` | 0.715 | 0.721 [0.71-0.74] | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | – s |
| Air Preheaters (Primary & Secondary) | 3 | `dense_ae` | 0.464 | 0.457 [0.42-0.51] | 0.124 | 0.278 | 0.080 | 0.476 | 0.069 | 0.114 | – | 668 s |
| Combustion Air Fans (Primary & Secondary) | 3 | `staleness` | 0.606 | 0.462 [0.42-0.50] | 0.033 | 0.286 | 0.017 | 0.026 | 0.000 | 0.023 | – | 525 s |
| Induced Draft Fan | 4 | `staleness` | 0.648 | 0.544 [0.52-0.57] | 0.196 | 0.481 | 0.123 | 0.118 | 0.117 | 0.085 | 0.287 | 208 s |
| Return Air Chambers | 2 | `ensemble` | 0.572 | 0.500 [0.48-0.52] | 0.396 | 0.428 | 0.368 | 0.475 | 0.429 | – | – | 174 s |
| Fluidized Bed (Hearth & Layers) | 4 | `isolation_forest` | 0.621 | 0.307 [0.27-0.34] | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | – | – s |

## Per-part detector comparison

### Furnace Chamber (Upper) (`furnace`, criticality 5, tags: PT_8313A.AV_0, PT_8313B.AV_0, PT_8313C.AV_0, PT_8313D.AV_0, PT_8313E.AV_0, PT_8313F.AV_0, TE_8313B.AV_0)

Mode: **fused** | threshold: 1.14 | scenarios: demo_water_wall_leak, sensor_stuck

| Detector | Kind | AUROC | val AUROC | F1 | Prec | Rec | FAR(clean) | Rec≥0.5 | Rec≥1.0 | Latency | Weight |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `isolation_forest` | sklearn | 0.666 | 0.466 | 0.401 | 0.532 | 0.321 | 0.146 | 0.440 | 0.000 | 63 s | 0.00 |
| `lstm_residual` | tensorflow | 0.622 | 0.671 | 0.239 | 0.484 | 0.159 | 0.087 | 0.249 | 1.000 | 179 s | 0.60 |
| `tcn_ae` | tensorflow | 0.603 | 0.564 | 0.114 | 0.287 | 0.071 | 0.092 | 0.097 | 0.941 | 58 s | 0.00 |
| `ocsvm` | sklearn | 0.598 | 0.419 | 0.297 | 0.463 | 0.219 | 0.131 | 0.368 | 0.000 | 97 s | 0.00 |
| `staleness` | statistical | 0.588 | 0.612 | 0.024 | 0.367 | 0.013 | 0.011 | 0.023 | 0.000 | 450 s | 0.40 |
| `mean_shift` | statistical | 0.452 | 0.357 | 0.126 | 0.186 | 0.096 | 0.216 | 0.123 | 0.000 | 62 s | 0.00 |
| `dense_ae` | tensorflow | 0.450 | 0.546 | 0.074 | 0.154 | 0.048 | 0.137 | 0.059 | 0.235 | 110 s | 0.00 |
| `ewma_mad` | statistical | 0.445 | 0.356 | 0.111 | 0.156 | 0.086 | 0.241 | 0.097 | 0.000 | 132 s | 0.00 |
| **ensemble** | fusion | 0.683 | – | 0.172 | 0.785 | 0.097 | 0.014 | 0.148 | 0.824 | 126 s | – |

### Steam Drum & Pot (`drum`, criticality 5, tags: PTCA_8322A.AV_0, PTCA_8324.AV_0)

Mode: **single** (dominant model: `isolation_forest`) | threshold: 2.93 | scenarios: demo_water_wall_leak

| Detector | Kind | AUROC | val AUROC | F1 | Prec | Rec | FAR(clean) | Rec≥0.5 | Rec≥1.0 | Latency | Weight |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `isolation_forest` | sklearn | 0.618 | 0.712 | 0.136 | 0.545 | 0.078 | 0.040 | 0.090 | – | 400 s | 1.00 |
| `staleness` | statistical | 0.553 | 0.526 | 0.000 | 0.000 | 0.000 | 0.030 | 0.000 | – | – s | 0.00 |
| `lstm_residual` | tensorflow | 0.409 | 0.515 | 0.008 | 0.095 | 0.004 | 0.025 | 0.005 | – | 570 s | 0.00 |
| `tcn_ae` | tensorflow | 0.350 | 0.605 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | – | – s | 0.00 |
| `ocsvm` | sklearn | 0.345 | 0.453 | 0.000 | 0.000 | 0.000 | 0.183 | 0.000 | – | – s | 0.00 |
| `dense_ae` | tensorflow | 0.248 | 0.433 | 0.008 | 0.010 | 0.006 | 0.404 | 0.007 | – | 565 s | 0.00 |
| `mean_shift` | statistical | 0.193 | 0.205 | 0.034 | 1.000 | 0.017 | 0.000 | 0.020 | – | 40 s | 0.00 |
| `ewma_mad` | statistical | 0.152 | 0.087 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | – | – s | 0.00 |
| **ensemble** | fusion | 0.618 | – | 0.136 | 0.545 | 0.078 | 0.040 | 0.090 | – | 400 s | – |

### Main Steam Line & Outlet (`main_steam`, criticality 5, tags: TE_8332A.AV_0, ZZQBCHLL.AV_0)

Mode: **single** (dominant model: `isolation_forest`) | threshold: 1.38 | scenarios: demo_tailed_steam_temp, demo_water_wall_leak

| Detector | Kind | AUROC | val AUROC | F1 | Prec | Rec | FAR(clean) | Rec≥0.5 | Rec≥1.0 | Latency | Weight |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `isolation_forest` | sklearn | 0.845 | 0.850 | 0.227 | 1.000 | 0.128 | 0.000 | 0.000 | 0.237 | 112 s | 0.60 |
| `staleness` | statistical | 0.621 | 0.440 | 0.050 | 0.732 | 0.026 | 0.008 | 0.000 | 0.048 | 602 s | 0.00 |
| `ocsvm` | sklearn | 0.540 | 0.629 | 0.430 | 0.436 | 0.423 | 0.430 | 0.388 | 0.540 | 234 s | 0.00 |
| `lstm_residual` | tensorflow | 0.471 | 0.735 | 0.305 | 0.388 | 0.252 | 0.313 | 0.340 | 0.248 | 146 s | 0.40 |
| `tcn_ae` | tensorflow | 0.411 | 0.521 | 0.254 | 0.280 | 0.233 | 0.472 | 0.222 | 0.294 | 466 s | 0.00 |
| `dense_ae` | tensorflow | 0.325 | 0.513 | 0.216 | 0.239 | 0.198 | 0.496 | 0.211 | 0.235 | 215 s | 0.00 |
| `mean_shift` | statistical | 0.081 | 0.102 | 0.017 | 0.167 | 0.009 | 0.034 | 0.026 | 0.000 | 35 s | 0.00 |
| `ewma_mad` | statistical | 0.037 | 0.059 | 0.000 | 0.000 | 0.000 | 0.008 | 0.000 | 0.000 | – s | 0.00 |
| **ensemble** | fusion | 0.752 | – | 0.291 | 0.623 | 0.190 | 0.090 | 0.140 | 0.266 | 290 s | – |

### Superheater & Desuperheater (`superheater`, criticality 4, tags: TV_8329ZC.AV_0, YJJWSLL.AV_0)

Mode: **single** (dominant model: `lstm_residual`) | threshold: 5.80 | scenarios: attemperator_valve_stuck

| Detector | Kind | AUROC | val AUROC | F1 | Prec | Rec | FAR(clean) | Rec≥0.5 | Rec≥1.0 | Latency | Weight |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `lstm_residual` | tensorflow | 0.743 | 0.784 | 0.499 | 0.628 | 0.415 | 0.153 | 1.000 | – | 30 s | 0.62 |
| `staleness` | statistical | 0.593 | 0.489 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | – | – s | 0.00 |
| `tcn_ae` | tensorflow | 0.517 | 0.677 | 0.275 | 0.460 | 0.197 | 0.144 | 0.925 | – | 122 s | 0.38 |
| `dense_ae` | tensorflow | 0.484 | 0.488 | 0.302 | 0.296 | 0.308 | 0.456 | 0.650 | – | 78 s | 0.00 |
| `isolation_forest` | sklearn | 0.456 | 0.510 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | – | – s | 0.00 |
| `ocsvm` | sklearn | 0.386 | 0.352 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | – | – s | 0.00 |
| `mean_shift` | statistical | 0.168 | 0.188 | 0.151 | 0.194 | 0.124 | 0.321 | 0.025 | – | 30 s | 0.00 |
| `ewma_mad` | statistical | 0.096 | 0.081 | 0.007 | 0.014 | 0.004 | 0.191 | 0.025 | – | 575 s | 0.00 |
| **ensemble** | fusion | 0.654 | – | 0.466 | 0.586 | 0.387 | 0.171 | 0.988 | – | 30 s | – |

### Economizer (Upper) (`economizer`, criticality 3, tags: TE_8319A.AV_0, TE_8319B.AV_0, AIR_8301A.AV_0, AIR_8301B.AV_0)

Mode: **single** (dominant model: `isolation_forest`) | threshold: 1.50 | scenarios: demo_sensor_spike, economizer_fouling, furnace_draft_disturbance

| Detector | Kind | AUROC | val AUROC | F1 | Prec | Rec | FAR(clean) | Rec≥0.5 | Rec≥1.0 | Latency | Weight |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `isolation_forest` | sklearn | 0.721 | 0.715 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | – s | 0.43 |
| `lstm_residual` | tensorflow | 0.625 | 0.602 | 0.155 | 0.325 | 0.102 | 0.106 | 0.206 | 0.085 | 228 s | 0.00 |
| `ocsvm` | sklearn | 0.620 | 0.625 | 0.170 | 0.266 | 0.125 | 0.174 | 0.231 | 0.000 | 30 s | 0.25 |
| `tcn_ae` | tensorflow | 0.618 | 0.584 | 0.288 | 0.359 | 0.241 | 0.217 | 0.370 | 0.225 | 73 s | 0.00 |
| `dense_ae` | tensorflow | 0.579 | 0.658 | 0.213 | 0.231 | 0.198 | 0.330 | 0.355 | 0.067 | 136 s | 0.32 |
| `mean_shift` | statistical | 0.536 | 0.462 | 0.258 | 0.361 | 0.200 | 0.178 | 0.152 | 0.184 | 105 s | 0.00 |
| `staleness` | statistical | 0.485 | 0.534 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | – s | 0.00 |
| `ewma_mad` | statistical | 0.445 | 0.363 | 0.162 | 0.379 | 0.103 | 0.085 | 0.056 | 0.096 | 106 s | 0.00 |
| **ensemble** | fusion | 0.630 | – | 0.202 | 0.298 | 0.152 | 0.180 | 0.287 | 0.021 | 15 s | – |

### Air Preheaters (Primary & Secondary) (`air_preheaters`, criticality 3, tags: TE_8303.AV_0, TE_8304.AV_0)

Mode: **single** (dominant model: `dense_ae`) | threshold: 3.45 | scenarios: air_preheater_fouling

| Detector | Kind | AUROC | val AUROC | F1 | Prec | Rec | FAR(clean) | Rec≥0.5 | Rec≥1.0 | Latency | Weight |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `lstm_residual` | tensorflow | 0.559 | 0.366 | 0.013 | 0.118 | 0.007 | 0.115 | 0.012 | – | 1110 s | 0.12 |
| `isolation_forest` | sklearn | 0.546 | 0.382 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | – | – s | 0.12 |
| `dense_ae` | tensorflow | 0.457 | 0.464 | 0.124 | 0.278 | 0.080 | 0.476 | 0.114 | – | 668 s | 0.12 |
| `ewma_mad` | statistical | 0.427 | 0.348 | 0.092 | 0.338 | 0.053 | 0.239 | 0.014 | – | 30 s | 0.12 |
| `mean_shift` | statistical | 0.400 | 0.418 | 0.241 | 0.536 | 0.156 | 0.308 | 0.100 | – | 30 s | 0.12 |
| `staleness` | statistical | 0.341 | 0.433 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | – | – s | 0.12 |
| `ocsvm` | sklearn | 0.298 | 0.188 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | – | – s | 0.12 |
| `tcn_ae` | tensorflow | 0.210 | 0.225 | 0.007 | 0.022 | 0.004 | 0.448 | 0.008 | – | 1105 s | 0.12 |
| **ensemble** | fusion | 0.307 | – | 0.004 | 0.013 | 0.002 | 0.384 | 0.004 | – | 1115 s | – |

### Combustion Air Fans (Primary & Secondary) (`combustion_fans`, criticality 3, tags: FT_8301.AV_0, FT_8302.AV_0)

Mode: **single** (dominant model: `staleness`) | threshold: 1.05 | scenarios: furnace_draft_disturbance

| Detector | Kind | AUROC | val AUROC | F1 | Prec | Rec | FAR(clean) | Rec≥0.5 | Rec≥1.0 | Latency | Weight |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `tcn_ae` | tensorflow | 0.631 | 0.511 | 0.000 | 0.000 | 0.000 | 0.029 | 0.000 | – | – s | 0.00 |
| `dense_ae` | tensorflow | 0.558 | 0.503 | 0.097 | 0.178 | 0.067 | 0.189 | 0.089 | – | 312 s | 0.00 |
| `lstm_residual` | tensorflow | 0.508 | 0.463 | 0.028 | 0.233 | 0.015 | 0.030 | 0.000 | – | 40 s | 0.00 |
| `ocsvm` | sklearn | 0.499 | 0.481 | 0.000 | 0.000 | 0.000 | 0.074 | 0.000 | – | – s | 0.00 |
| `isolation_forest` | sklearn | 0.471 | 0.462 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | – | – s | 0.00 |
| `staleness` | statistical | 0.462 | 0.606 | 0.033 | 0.286 | 0.017 | 0.026 | 0.023 | – | 525 s | 1.00 |
| `ewma_mad` | statistical | 0.459 | 0.465 | 0.052 | 0.192 | 0.030 | 0.078 | 0.003 | – | 42 s | 0.00 |
| `mean_shift` | statistical | 0.431 | 0.489 | 0.253 | 0.298 | 0.220 | 0.318 | 0.203 | – | 61 s | 0.00 |
| **ensemble** | fusion | 0.462 | – | 0.033 | 0.286 | 0.017 | 0.026 | 0.023 | – | 525 s | – |

### Induced Draft Fan (`id_fan`, criticality 4, tags: YFJ3_AI.AV_0, YFJ3_ZD1.AV_0, YFJ3_ZD2.AV_0)

Mode: **single** (dominant model: `staleness`) | threshold: 0.91 | scenarios: demo_idf_degradation, id_fan_bearing_failure

| Detector | Kind | AUROC | val AUROC | F1 | Prec | Rec | FAR(clean) | Rec≥0.5 | Rec≥1.0 | Latency | Weight |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `mean_shift` | statistical | 0.642 | 0.506 | 0.214 | 0.331 | 0.158 | 0.283 | 0.142 | 0.287 | 237 s | 0.00 |
| `isolation_forest` | sklearn | 0.629 | 0.497 | 0.392 | 0.608 | 0.289 | 0.165 | 0.206 | 0.829 | 327 s | 0.00 |
| `lstm_residual` | tensorflow | 0.627 | 0.533 | 0.444 | 0.573 | 0.363 | 0.240 | 0.369 | 0.771 | 132 s | 0.00 |
| `tcn_ae` | tensorflow | 0.616 | 0.492 | 0.413 | 0.548 | 0.332 | 0.242 | 0.273 | 0.967 | 209 s | 0.00 |
| `ocsvm` | sklearn | 0.569 | 0.421 | 0.242 | 0.531 | 0.156 | 0.122 | 0.004 | 0.625 | 294 s | 0.00 |
| `staleness` | statistical | 0.544 | 0.648 | 0.196 | 0.481 | 0.123 | 0.118 | 0.085 | 0.287 | 208 s | 1.00 |
| `ewma_mad` | statistical | 0.501 | 0.440 | 0.211 | 0.349 | 0.151 | 0.250 | 0.100 | 0.350 | 181 s | 0.00 |
| `dense_ae` | tensorflow | 0.489 | 0.305 | 0.082 | 0.302 | 0.047 | 0.097 | 0.017 | 0.000 | 160 s | 0.00 |
| **ensemble** | fusion | 0.544 | – | 0.196 | 0.481 | 0.123 | 0.118 | 0.085 | 0.287 | 208 s | – |

### Return Air Chambers (`return_air`, criticality 2, tags: FT_8306A.AV_0, FT_8306B.AV_0)

Mode: **fused** | threshold: 0.59 | scenarios: bed_dp_drift, furnace_draft_disturbance

| Detector | Kind | AUROC | val AUROC | F1 | Prec | Rec | FAR(clean) | Rec≥0.5 | Rec≥1.0 | Latency | Weight |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `staleness` | statistical | 0.513 | 0.546 | 0.223 | 0.429 | 0.151 | 0.195 | – | – | 206 s | 0.27 |
| `lstm_residual` | tensorflow | 0.512 | 0.530 | 0.504 | 0.448 | 0.575 | 0.684 | – | – | 94 s | 0.17 |
| `mean_shift` | statistical | 0.489 | 0.538 | 0.324 | 0.452 | 0.252 | 0.296 | – | – | 166 s | 0.22 |
| `ewma_mad` | statistical | 0.487 | 0.534 | 0.135 | 0.451 | 0.079 | 0.093 | – | – | 226 s | 0.20 |
| `dense_ae` | tensorflow | 0.466 | 0.365 | 0.303 | 0.438 | 0.231 | 0.286 | – | – | 171 s | 0.00 |
| `tcn_ae` | tensorflow | 0.464 | 0.525 | 0.565 | 0.451 | 0.757 | 0.890 | – | – | 86 s | 0.14 |
| `ocsvm` | sklearn | 0.439 | 0.444 | 0.464 | 0.397 | 0.559 | 0.819 | – | – | 99 s | 0.00 |
| `isolation_forest` | sklearn | 0.421 | 0.469 | 0.506 | 0.415 | 0.649 | 0.884 | – | – | 86 s | 0.00 |
| **ensemble** | fusion | 0.500 | – | 0.396 | 0.428 | 0.368 | 0.475 | – | – | 174 s | – |

### Fluidized Bed (Hearth & Layers) (`fluidized_bed`, criticality 4, tags: SXLTCYZ.AV_0, SXLTCYY.AV_0, ZCLCCY.AV_0, YCLCCY.AV_0)

Mode: **single** (dominant model: `isolation_forest`) | threshold: 2.57 | scenarios: bed_dp_drift

| Detector | Kind | AUROC | val AUROC | F1 | Prec | Rec | FAR(clean) | Rec≥0.5 | Rec≥1.0 | Latency | Weight |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `staleness` | statistical | 0.590 | 0.455 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | – | – s | 0.00 |
| `lstm_residual` | tensorflow | 0.406 | 0.576 | 0.009 | 0.235 | 0.005 | 0.033 | 0.011 | – | 1065 s | 0.38 |
| `ewma_mad` | statistical | 0.352 | 0.403 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | – | – s | 0.00 |
| `mean_shift` | statistical | 0.327 | 0.385 | 0.000 | 0.000 | 0.000 | 0.031 | 0.000 | – | – s | 0.00 |
| `tcn_ae` | tensorflow | 0.313 | 0.493 | 0.119 | 0.429 | 0.069 | 0.204 | 0.160 | – | 775 s | 0.00 |
| `isolation_forest` | sklearn | 0.307 | 0.621 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | – | – s | 0.62 |
| `ocsvm` | sklearn | 0.260 | 0.492 | 0.000 | 0.000 | 0.000 | 0.132 | 0.000 | – | – s | 0.00 |
| `dense_ae` | tensorflow | 0.244 | 0.454 | 0.000 | 0.000 | 0.000 | 0.344 | 0.000 | – | – s | 0.00 |
| **ensemble** | fusion | 0.311 | – | 0.000 | 0.000 | 0.000 | 0.010 | 0.000 | – | – s | – |

## Global check against the dataset's real 'tailed' anomalies

Only the `main_steam` part holds the state-discriminating tag (TE_8332A).

| Part | AUROC | F1 | Precision | Recall | FAR | Positives | Prevalence |
|---|---|---|---|---|---|---|---|
| Main Steam Line & Outlet | 0.918 | 0.195 | 0.283 | 0.149 | 0.026 | 845 | 6.52% |
