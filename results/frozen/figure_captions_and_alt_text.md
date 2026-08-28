# Figure captions and alt text

## Figure: performance across sampling-rate ratios

**Caption.** Macro-F1 across five prespecified sampling-rate ratios. Each point is the equal-weight mean over ten datasets after averaging the three prespecified seeds within the frozen evaluation matrix. V5.1 contains 30 newly trained runs; the seven comparators are reused from the audited 210-run Full matrix. Lines connect measured rate conditions only and do not represent interpolation. Lower sampling rates appear to the right.

**Alt text.** Eight line series compare macro-F1 as the sampling-rate ratio decreases from 1.0 to 0.3. MultiROCKET is slightly highest at full rate, while V5.1 is highest at each of the four unseen rate ratios and declines less sharply; dataset-level exceptions are summarized separately.

## Figure: paired dataset effects

**Caption.** Dataset-resolved differences in mean unseen-rate macro-F1 between V5.1 and each comparator. Open circles are ten fixed dataset effects after averaging three seeds. Diamonds and horizontal intervals are the paired mean and 10,000-resample dataset bootstrap 95% interval. The vertical zero line denotes no difference. Wilcoxon tests and seven-comparison Holm adjustment are reported in the accompanying table.

**Alt text.** Seven horizontal rows show ten dataset effects per comparator together with a mean and confidence interval. Effects are heterogeneous; most mean estimates favor V5.1, but several intervals include zero.

## Figure: independent confirmation

**Caption.** Independent V5.1-minus-V4.1 mean unseen-rate macro-F1 differences on four previously untouched datasets. Open circles show the three fresh-seed results and diamonds show dataset means. Test data were not used for selection.

**Alt text.** Four dataset rows show seed-level differences and dataset means. Every dataset mean is positive, although one seed-level result is negative for HandMovementDirection and one for Heartbeat.

## Figure: component ablation

**Caption.** Retrospective two-dataset component ablation. Bars show the change in three-seed mean unseen-rate macro-F1 relative to complete V5.1 after retraining each variant from scratch. The experiment is descriptive because both test datasets had been accessed previously; the two datasets, not the six seed runs, are the primary evidence units.

**Alt text.** Removing the signed spatial path lowers performance on both datasets. Mean-only summaries and fixed equal fusion improve performance on both datasets relative to complete V5.1.
