# V5.1 ten-dataset Full extension

- Frozen decision: **FAIL**.
- Newly trained: 30 V5.1 runs. Reused without retraining: 210 frozen Full runs.
- This is a retrospective extension on previously accessed Full tests, not independent confirmation.
- Device / cumulative elapsed: `cuda` / `10382.2 s`.

| Baseline | Mean unseen F1 delta | 95% CI | Holm p | Positive datasets |
|---|---:|---:|---:|---:|
| v3_10 | +0.1913 | [+0.0228, +0.3818] | 0.2578 | 8/10 |
| v1_nyquistguard | +0.2420 | [+0.0867, +0.4139] | 0.08203 | 9/10 |
| fixed_rate_tcn | +0.1859 | [+0.0576, +0.3669] | 0.02734 | 9/10 |
| multirate_tcn | +0.1506 | [+0.0167, +0.3362] | 0.2441 | 8/10 |
| minirocket | +0.0436 | [-0.0676, +0.2057] | 1 | 4/10 |
| multirocket | +0.1393 | [-0.0040, +0.3007] | 0.3203 | 7/10 |
| v3_10_no_nyquist_gate | +0.1542 | [-0.0049, +0.3339] | 0.2578 | 8/10 |

## Frozen checks

- average_unseen_gain_vs_v3_10: PASS
- positive_dataset_count_vs_v3_10: PASS
- single_dataset_unseen_floor_vs_v3_10: FAIL
- reliability_nonworse: FAIL
- no_constant_prediction: PASS
- finite_metrics: PASS

No later stage was started automatically.
