# Frozen V5.1 results draft

> Internal evidence-bound draft. Numerical claims are derived only from the frozen local reports listed in `provenance_manifest.json`. Literature citations, authorship, ethics, funding, conflicts, and journal-specific formatting remain subject to human verification.

## Full retrospective extension

The frozen V5.1 extension trained 30 new candidate runs across ten datasets and three prespecified seeds and reused, without retraining, the 210-run seven-method Full matrix. V5.1 achieved the highest cross-dataset mean unseen-rate macro-F1 (0.7584) and worst-unseen macro-F1 (0.7258) among the eight evaluated implementations. Its full-rate macro-F1 was 0.7691. Relative to v3.10, the dataset-level mean difference was +0.1913 (10,000-resample dataset bootstrap 95% CI +0.0228 to +0.3818); eight of ten dataset effects were positive, although the two-sided Wilcoxon test was not significant after Holm adjustment (adjusted p=0.2578). The largest unfavorable dataset effect was PAMAP2 (-0.2988).

V5.1 exceeded the fixed-rate TCN by +0.1859 on average (95% CI +0.0576 to +0.3669; Holm-adjusted p=0.0273), the only comparison that remained significant after seven-comparison Holm correction. The mean differences versus MiniROCKET and MultiROCKET were +0.0436 and +0.1393, respectively, but their intervals crossed zero and the Holm-adjusted p-values were 1.0000 and 0.3203. The difference versus the v3.10 no-gate implementation was +0.1542; this comparison also did not survive Holm correction (adjusted p=0.2578).

The frozen extension decision gate was not passed. Four of six checks passed, while the PAMAP2 degradation beyond the prespecified single-dataset floor and a mean selected-score AURC change of +0.001735 (lower is better) failed their respective checks. These outcomes constrain the claim to improved average robustness with dataset-dependent exceptions, rather than uniform superiority or uniformly improved reliability.

## Independent confirmation

Before the ten-dataset extension, V5.1 was independently evaluated on four previously untouched datasets using three fresh seeds per model. All four dataset-level mean differences relative to V4.1 were positive, with an average difference of +0.1630 and a minimum dataset mean difference of +0.0593. Test data were not used for model or reliability-mode selection. This confirmation therefore provides the strongest evidence that the signed spatial dual-path design transfers beyond its development datasets.

## Component ablation

The retrained component ablation was retrospective and descriptive because its two test sets had already been accessed. Removing the signed spatial path reduced mean unseen-rate macro-F1 by -0.1395 across the two datasets and was unfavorable in five of six seed-level comparisons. Replacing the richer temporal summaries with a mean-only summary changed performance by +0.0643 and was favorable in all six seed-level comparisons. Replacing adaptive fusion and the cross-path residual with fixed equal fusion changed performance by +0.0400 and was favorable in five of six seed-level comparisons. Thus, this small explanatory panel supports the signed spatial path as the principal component contribution, but it does not support claiming that every additional fusion or summary mechanism is individually necessary.

## Sequential efficiency

On the same local CUDA device under sequential inference-only measurement, V5.1 used 143,794 parameters and 679,501 FLOPs per sample, with a mean latency of 0.5636 ms per sample. The V4.1 reference used 52,863 parameters, 573,655 FLOPs per sample, and 0.3743 ms per sample. These are implementation- and hardware-specific measurements and should not be generalized to other devices.

## Evidence-bounded conclusion

Across the frozen ten-dataset extension, V5.1 had the highest average and worst-unseen macro-F1 of the evaluated implementations and showed a statistically supported advantage over the fixed-rate TCN. Independent confirmation was positive on all four dataset means. The results do not establish uniform superiority over every strong baseline: comparisons with ROCKET methods were not significant after multiplicity correction, PAMAP2 showed a substantial negative effect relative to v3.10, and the reliability non-worsening gate was not met. The component evidence most directly supports the signed spatial waveform path; the more complex summary and adaptive-fusion choices remain candidates for future simplification rather than required contributions of the present model.
