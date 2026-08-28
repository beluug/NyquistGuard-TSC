# V5.1 independent confirmation on four untouched datasets

- Frozen decision: **PASS**.
- All reliability modes were frozen from validation across three seeds before any TEST parse.
- TEST was evaluated once per validation-selected checkpoint and never selected a model or mode.
- Device / cumulative elapsed: `cuda` / `205.9 s`.

| Dataset | Reliability mode | Mean unseen F1 delta | Full-rate F1 delta | Selected AURC delta |
|---|---|---:|---:|---:|
| self_regulation_scp1_uea | confidence_fallback | +0.0593 | +0.0570 | +0.0000 |
| hand_movement_direction_uea | confidence_fallback | +0.0653 | +0.0697 | +0.0000 |
| racket_sports_uea | confidence_fallback | +0.4574 | +0.3807 | +0.0000 |
| heartbeat_uea | confidence_fallback | +0.0699 | +0.0760 | +0.0000 |

## Frozen checks

- average_dataset_unseen_gain: PASS
- positive_dataset_count: PASS
- single_dataset_unseen_floor: PASS
- average_dataset_full_rate_floor: PASS
- average_dataset_reliability_safety: PASS
- no_constant_prediction: PASS
- finite_metrics: PASS

No later stage was started automatically.
