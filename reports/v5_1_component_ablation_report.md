# V5.1 retrained component ablation

- All four variants were independently retrained; this is not inference-time masking.
- This retrospective ablation is explanatory and is not a new independent confirmation.

| Dataset | Variant | Mean unseen F1 | Delta vs full V5 | Full-rate F1 |
|---|---|---:|---:|---:|
| self_regulation_scp1_uea | v5_full | 0.7079 | +0.0000 | 0.7097 |
| self_regulation_scp1_uea | no_signed_spatial_path | 0.6629 | -0.0450 | 0.6690 |
| self_regulation_scp1_uea | mean_only_temporal_summary | 0.8070 | +0.0991 | 0.8048 |
| self_regulation_scp1_uea | fixed_equal_fusion | 0.7413 | +0.0334 | 0.7420 |
| racket_sports_uea | v5_full | 0.8127 | +0.0000 | 0.8559 |
| racket_sports_uea | no_signed_spatial_path | 0.5787 | -0.2339 | 0.6839 |
| racket_sports_uea | mean_only_temporal_summary | 0.8421 | +0.0294 | 0.8632 |
| racket_sports_uea | fixed_equal_fusion | 0.8594 | +0.0467 | 0.8760 |

No later stage was started automatically.
