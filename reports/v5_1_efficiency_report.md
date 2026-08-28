# V5.1 sequential efficiency benchmark

- Frozen checkpoints only; no training was performed.
- Device: `cuda`; elapsed: `8.16 s`.

| Role | Parameters | FLOPs/sample | Latency ms/sample | Throughput samples/s |
|---|---:|---:|---:|---:|
| v4_1_residual_gate | 52863 | 573655 | 0.3743 | 2747.17 |
| v5_dual_path | 143794 | 679502 | 0.5636 | 1790.30 |

No later stage was started automatically.
