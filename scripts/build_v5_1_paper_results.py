"""Build the frozen V5.1 paper-results package without training any model.

The script is deliberately fail-closed: it accepts only the audited report hashes
recorded below, checks the expected run matrices, and writes tables/figures from
those immutable JSON reports.  It never imports the experiment runners.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


EXPECTED = {
    "full": ("full_report.json", "ce1555d0ee8ded643c715db220364fa4115fd7fa8588d78305442205dd2bc47e"),
    "extension": ("v5_1_full_extension_report.json", "5e9436916b681315233406d8e8412f7078ab1784f77910798098a4f82e5df1ad"),
    "independent": ("v5_1_independent_confirmation_report.json", "447d8e4479cb2c7f57a38b7d8593f32d00c67c6b61abe4c655d90ffd5f17a0e3"),
    "ablation": ("v5_1_component_ablation_report.json", "d2a190d66c14b8b92657170ee09540dccf34bef2b0b0e3e1c5931fb871c70c2b"),
    "efficiency": ("v5_1_efficiency_report.json", "447d8e4479cb2c7f57a38b7d8593f32d00c67c6b61abe4c655d90ffd5f17a0e3"),
}

METHOD_NAMES = {
    "v5_1_safe_dual_path": "NyquistGuard-TSC V5.1",
    "v3_10": "NyquistGuard-TSC v3.10",
    "v1_nyquistguard": "NyquistGuard v1",
    "fixed_rate_tcn": "Fixed-rate TCN",
    "multirate_tcn": "Multirate TCN",
    "minirocket": "MiniROCKET (10,000 kernels)",
    "multirocket": "MultiROCKET (1,000 kernels)",
    "v3_10_no_nyquist_gate": "v3.10 without Nyquist gate",
    "v4_1_residual_gate": "NyquistGuard-TSC V4.1",
    "v5_dual_path": "NyquistGuard-TSC V5.1",
}

DATASET_NAMES = {
    "basicmotions_uea": "BasicMotions",
    "epilepsy_uea": "Epilepsy",
    "pamap2_uci": "PAMAP2",
    "mhealth_uci": "MHEALTH",
    "hapt_uci": "HAPT",
    "daily_sports_uci": "DailySports",
    "hydraulic_uci": "Hydraulic",
    "sleep_edfx_physionet": "Sleep-EDF",
    "eegmmi_physionet": "EEGMMI",
    "mitbih_arrhythmia_physionet": "MIT-BIH",
    "self_regulation_scp1_uea": "SelfRegulationSCP1",
    "hand_movement_direction_uea": "HandMovementDirection",
    "racket_sports_uea": "RacketSports",
    "heartbeat_uea": "Heartbeat",
}

RATE_KEYS = ("r1000", "r0900", "r0600", "r0400", "r0300")
RATE_VALUES = (1.0, 0.9, 0.6, 0.4, 0.3)
PALETTE = {
    "v5_1_safe_dual_path": "#0072B2",
    "v3_10": "#E69F00",
    "v1_nyquistguard": "#999999",
    "fixed_rate_tcn": "#009E73",
    "multirate_tcn": "#56B4E9",
    "minirocket": "#CC79A7",
    "multirocket": "#D55E00",
    "v3_10_no_nyquist_gate": "#F0E442",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_bundle(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    reports = root / "reports"
    bundle: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for key, (filename, expected_protocol) in EXPECTED.items():
        path = reports / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            raise ValueError(f"{filename}: expected status=completed")
        actual_protocol = payload.get("protocol_hash", payload.get("source_protocol_hash"))
        if actual_protocol != expected_protocol:
            raise ValueError(f"{filename}: unexpected protocol hash {actual_protocol}")
        bundle[key] = payload
        paths[key] = path

    full, ext, indep, ablation = (
        bundle["full"], bundle["extension"], bundle["independent"], bundle["ablation"]
    )
    checks = {
        "Full matrix": full.get("completed_runs") == 210,
        "V5.1 extension matrix": ext.get("new_candidate_runs") == 30 and ext.get("reused_full_runs") == 210,
        "independent matrix": len(indep.get("role_results", {})) == 24,
        "ablation matrix": ablation.get("completed_runs") == 24 and len(ablation.get("rows", [])) == 8,
        "no automatic later stage": not any(
            bool(bundle[k].get("later_stage_started", bundle[k].get("automatic_followup_started", False)))
            for k in bundle
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("frozen evidence audit failed: " + ", ".join(failed))
    return bundle, paths


def v5_rate_summary(extension: dict[str, Any]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    run_root = Path(extension["run_root"])
    metric_paths = sorted(run_root.glob("*__seed*/metrics.json"))
    if len(metric_paths) != 30:
        raise ValueError(f"expected 30 V5.1 metrics files, found {len(metric_paths)}")
    for path in metric_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed" or payload.get("protocol_hash") != extension["protocol_hash"]:
            raise ValueError(f"invalid V5.1 metrics file: {path}")
        per_rate = payload["test"]["per_rate"]
        for rate_key in RATE_KEYS:
            score = float(per_rate[rate_key]["macro_f1"])
            if not np.isfinite(score):
                raise ValueError(f"non-finite score in {path}")
            values[rate_key].append(score)
    return {key: float(np.mean(values[key])) for key in RATE_KEYS}


def build_tables(bundle: dict[str, dict[str, Any]], out: Path) -> dict[str, list[dict[str, Any]]]:
    extension = bundle["extension"]
    summary = extension["method_summary"]
    method_order = ["v5_1_safe_dual_path", "minirocket", "multirocket", "multirate_tcn",
                    "v3_10_no_nyquist_gate", "fixed_rate_tcn", "v3_10", "v1_nyquistguard"]
    main_rows = []
    for rank, method in enumerate(method_order, 1):
        row = summary[method]
        main_rows.append({
            "display_order": rank,
            "method_id": method,
            "method": METHOD_NAMES[method],
            "mean_unseen_macro_f1": f"{row['mean_unseen_macro_f1']:.6f}",
            "worst_unseen_macro_f1": f"{row['worst_unseen_macro_f1']:.6f}",
            "full_rate_macro_f1": f"{row['full_rate_macro_f1']:.6f}",
        })
    write_csv(out / "table_main_comparison.csv", list(main_rows[0]), main_rows)

    paired_rows = []
    effect_rows = []
    for method, stats in extension["paired_statistics"].items():
        deltas = stats["dataset_deltas"]
        positive = sum(float(value) > 0 for value in deltas.values())
        negative = sum(float(value) < 0 for value in deltas.values())
        ties = len(deltas) - positive - negative
        lo, hi = stats["dataset_clustered_bootstrap_95_ci"]
        paired_rows.append({
            "comparator_id": method,
            "comparator": METHOD_NAMES[method],
            "mean_delta_v5_1_minus_comparator": f"{stats['mean_delta']:.6f}",
            "bootstrap_95_ci_low": f"{lo:.6f}",
            "bootstrap_95_ci_high": f"{hi:.6f}",
            "wilcoxon_two_sided_p": f"{stats['wilcoxon_p']:.8f}",
            "holm_adjusted_p_7_comparisons": f"{stats['holm_adjusted_p']:.8f}",
            "positive_tie_negative_datasets": f"{positive}/{ties}/{negative}",
            "holm_reject_alpha_0_05": bool(stats["holm_adjusted_p"] < 0.05),
        })
        for dataset, delta in deltas.items():
            effect_rows.append({
                "dataset_id": dataset,
                "dataset": DATASET_NAMES[dataset],
                "comparator_id": method,
                "comparator": METHOD_NAMES[method],
                "mean_unseen_macro_f1_delta": f"{delta:.9f}",
                "primary_unit": "dataset (three-seed mean)",
            })
    write_csv(out / "table_paired_statistics.csv", list(paired_rows[0]), paired_rows)
    write_csv(out / "table_dataset_effects.csv", list(effect_rows[0]), effect_rows)

    indep_rows = []
    for dataset, result in bundle["independent"]["dataset_results"].items():
        for seed_row in result["seed_rows"]:
            indep_rows.append({
                "dataset_id": dataset,
                "dataset": DATASET_NAMES[dataset],
                "seed": seed_row["seed"],
                "unseen_delta_v5_1_minus_v4_1": f"{seed_row['unseen_macro_f1_delta_vs_v4_1']:.9f}",
                "full_rate_delta_v5_1_minus_v4_1": f"{seed_row['full_rate_macro_f1_delta_vs_v4_1']:.9f}",
                "dataset_mean_unseen_delta": f"{result['mean_unseen_macro_f1_delta_vs_v4_1']:.9f}",
                "confirmation_status": "independent; TEST unused for selection",
            })
    write_csv(out / "table_independent_confirmation.csv", list(indep_rows[0]), indep_rows)

    ablation_rows = []
    for row in bundle["ablation"]["rows"]:
        ablation_rows.append({
            "dataset_id": row["dataset_id"],
            "dataset": DATASET_NAMES[row["dataset_id"]],
            "variant": row["variant"],
            "mean_unseen_macro_f1": f"{row['mean_unseen_macro_f1']:.9f}",
            "full_rate_macro_f1": f"{row['full_rate_macro_f1']:.9f}",
            "unseen_delta_vs_v5_full": f"{row['unseen_delta_vs_v5_full']:.9f}",
            "analysis_status": "retrospective descriptive; TEST previously accessed",
        })
    write_csv(out / "table_component_ablation.csv", list(ablation_rows[0]), ablation_rows)

    efficiency_rows = []
    for method, row in bundle["efficiency"]["summary"].items():
        efficiency_rows.append({"method_id": method, "method": METHOD_NAMES.get(method, method), **row})
    write_csv(out / "table_efficiency.csv", list(efficiency_rows[0]), efficiency_rows)
    return {
        "main": main_rows,
        "paired": paired_rows,
        "effects": effect_rows,
        "independent": indep_rows,
        "ablation": ablation_rows,
        "efficiency": efficiency_rows,
    }


def save_figure(fig: mpl.figure.Figure, base: Path) -> list[Path]:
    outputs = [base.with_suffix(".pdf"), base.with_suffix(".png")]
    fig.savefig(outputs[0], facecolor="white")
    fig.savefig(outputs[1], dpi=300, facecolor="white")
    plt.close(fig)
    return outputs


def build_figures(bundle: dict[str, dict[str, Any]], out: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 9,
        "axes.titlesize": 10, "legend.fontsize": 7, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
        "axes.spines.right": False,
    })
    outputs: list[Path] = []
    rate_rows: list[dict[str, Any]] = []
    rates = dict(bundle["full"]["aggregate"]["rate_summary"])
    rates["v5_1_safe_dual_path"] = v5_rate_summary(bundle["extension"])
    order = ["v5_1_safe_dual_path", "minirocket", "multirocket", "multirate_tcn",
             "v3_10_no_nyquist_gate", "fixed_rate_tcn", "v3_10", "v1_nyquistguard"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "h"]
    styles = ["-", "--", "-.", ":", "--", "-.", ":", "--"]
    fig, ax = plt.subplots(figsize=(180 / 25.4, 105 / 25.4), layout="constrained")
    for idx, method in enumerate(order):
        ys = [float(rates[method][key]) for key in RATE_KEYS]
        for ratio, value in zip(RATE_VALUES, ys):
            rate_rows.append({"method_id": method, "method": METHOD_NAMES[method],
                              "rate_ratio": ratio, "macro_f1": f"{value:.9f}",
                              "aggregation": "equal-weight mean over 10 datasets and 3 seeds"})
        ax.plot(RATE_VALUES, ys, label=METHOD_NAMES[method], color=PALETTE[method],
                marker=markers[idx], linestyle=styles[idx], linewidth=2.2 if idx == 0 else 1.2,
                markersize=5 if idx == 0 else 3.5, zorder=4 if idx == 0 else 2)
    ax.set(xlabel="Sampling-rate ratio (lower rate to the right)", ylabel="Macro-F1",
           xticks=RATE_VALUES, xlim=(1.03, 0.27), ylim=(0.0, 1.0))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.legend(ncol=2, frameon=False, loc="lower left")
    outputs += save_figure(fig, out / "figure_rate_curves")
    write_csv(out / "figure_rate_curves_data.csv", list(rate_rows[0]), rate_rows)

    stats_map = bundle["extension"]["paired_statistics"]
    fig, ax = plt.subplots(figsize=(180 / 25.4, 120 / 25.4), layout="constrained")
    y_positions = np.arange(len(order) - 1)
    comparison_order = order[1:]
    for y, method in zip(y_positions, comparison_order):
        stats = stats_map[method]
        points = list(stats["dataset_deltas"].values())
        ax.scatter(points, np.full(len(points), y), facecolors="white", edgecolors=PALETTE[method],
                   marker="o", s=24, linewidth=0.9, zorder=2)
        lo, hi = stats["dataset_clustered_bootstrap_95_ci"]
        ax.errorbar(stats["mean_delta"], y, xerr=[[stats["mean_delta"] - lo], [hi - stats["mean_delta"]]],
                    fmt="D", color=PALETTE[method], capsize=3, linewidth=1.5, markersize=4.5, zorder=3)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set(xlabel="V5.1 minus comparator: mean unseen-rate macro-F1",
           yticks=y_positions, yticklabels=[METHOD_NAMES[m] for m in comparison_order])
    ax.invert_yaxis()
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6)
    outputs += save_figure(fig, out / "figure_paired_dataset_effects")

    indep = bundle["independent"]["dataset_results"]
    fig, ax = plt.subplots(figsize=(180 / 25.4, 92 / 25.4), layout="constrained")
    labels = [DATASET_NAMES[d] for d in indep]
    for index, (dataset, result) in enumerate(indep.items()):
        seed_values = [float(row["unseen_macro_f1_delta_vs_v4_1"]) for row in result["seed_rows"]]
        ax.scatter(seed_values, np.full(3, index), facecolors="white", edgecolors="#0072B2",
                   marker="o", s=28, label="Seed result" if index == 0 else None)
        ax.scatter(result["mean_unseen_macro_f1_delta_vs_v4_1"], index, color="#D55E00",
                   marker="D", s=34, label="Dataset mean" if index == 0 else None, zorder=3)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set(xlabel="V5.1 minus V4.1: mean unseen-rate macro-F1", yticks=range(len(labels)), yticklabels=labels)
    ax.invert_yaxis(); ax.grid(axis="x", color="#D9D9D9", linewidth=0.6); ax.legend(frameon=False)
    outputs += save_figure(fig, out / "figure_independent_confirmation")

    abl = [row for row in bundle["ablation"]["rows"] if row["variant"] != "v5_full"]
    variants = ["no_signed_spatial_path", "mean_only_temporal_summary", "fixed_equal_fusion"]
    variant_names = ["Remove signed spatial path", "Mean-only temporal summary", "Fixed equal fusion"]
    fig, ax = plt.subplots(figsize=(180 / 25.4, 96 / 25.4), layout="constrained")
    x = np.arange(len(variants)); width = 0.32
    datasets = list(bundle["ablation"]["datasets"])
    hatches = ["///", "..."]
    for offset_index, dataset in enumerate(datasets):
        vals = [next(float(r["unseen_delta_vs_v5_full"]) for r in abl
                     if r["dataset_id"] == dataset and r["variant"] == variant) for variant in variants]
        ax.bar(x + (offset_index - 0.5) * width, vals, width=width, color="#56B4E9" if offset_index == 0 else "#E69F00",
               edgecolor="black", linewidth=0.6, hatch=hatches[offset_index], label=DATASET_NAMES[dataset])
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set(ylabel="Delta vs complete V5.1: mean unseen-rate macro-F1", xticks=x, xticklabels=variant_names,
           ylim=(-0.26, 0.13))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6); ax.legend(frameon=False)
    outputs += save_figure(fig, out / "figure_component_ablation")
    return outputs, rate_rows


def build_narrative(bundle: dict[str, dict[str, Any]], out: Path) -> list[Path]:
    ext, indep, abl, eff = bundle["extension"], bundle["independent"], bundle["ablation"], bundle["efficiency"]
    v5 = ext["method_summary"]["v5_1_safe_dual_path"]
    v3 = ext["paired_statistics"]["v3_10"]
    fixed = ext["paired_statistics"]["fixed_rate_tcn"]
    mini = ext["paired_statistics"]["minirocket"]
    multi = ext["paired_statistics"]["multirocket"]
    no_gate = ext["paired_statistics"]["v3_10_no_nyquist_gate"]
    indep_decision = indep["decision"]
    delta_by_variant = defaultdict(list)
    for row in abl["rows"]:
        if row["variant"] != "v5_full":
            delta_by_variant[row["variant"]].append(float(row["unseen_delta_vs_v5_full"]))
    v4eff, v5eff = eff["summary"]["v4_1_residual_gate"], eff["summary"]["v5_dual_path"]

    english = f"""# Frozen V5.1 results draft

> Internal evidence-bound draft. Numerical claims are derived only from the frozen local reports listed in `provenance_manifest.json`. Literature citations, authorship, ethics, funding, conflicts, and journal-specific formatting remain subject to human verification.

## Full retrospective extension

The frozen V5.1 extension trained 30 new candidate runs across ten datasets and three prespecified seeds and reused, without retraining, the 210-run seven-method Full matrix. V5.1 achieved the highest cross-dataset mean unseen-rate macro-F1 ({v5['mean_unseen_macro_f1']:.4f}) and worst-unseen macro-F1 ({v5['worst_unseen_macro_f1']:.4f}) among the eight evaluated implementations. Its full-rate macro-F1 was {v5['full_rate_macro_f1']:.4f}. Relative to v3.10, the dataset-level mean difference was {v3['mean_delta']:+.4f} (10,000-resample dataset bootstrap 95% CI {v3['dataset_clustered_bootstrap_95_ci'][0]:+.4f} to {v3['dataset_clustered_bootstrap_95_ci'][1]:+.4f}); eight of ten dataset effects were positive, although the two-sided Wilcoxon test was not significant after Holm adjustment (adjusted p={v3['holm_adjusted_p']:.4f}). The largest unfavorable dataset effect was PAMAP2 ({v3['dataset_deltas']['pamap2_uci']:+.4f}).

V5.1 exceeded the fixed-rate TCN by {fixed['mean_delta']:+.4f} on average (95% CI {fixed['dataset_clustered_bootstrap_95_ci'][0]:+.4f} to {fixed['dataset_clustered_bootstrap_95_ci'][1]:+.4f}; Holm-adjusted p={fixed['holm_adjusted_p']:.4f}), the only comparison that remained significant after seven-comparison Holm correction. The mean differences versus MiniROCKET and MultiROCKET were {mini['mean_delta']:+.4f} and {multi['mean_delta']:+.4f}, respectively, but their intervals crossed zero and the Holm-adjusted p-values were {mini['holm_adjusted_p']:.4f} and {multi['holm_adjusted_p']:.4f}. The difference versus the v3.10 no-gate implementation was {no_gate['mean_delta']:+.4f}; this comparison also did not survive Holm correction (adjusted p={no_gate['holm_adjusted_p']:.4f}).

The frozen extension decision gate was not passed. Four of six checks passed, while the PAMAP2 degradation beyond the prespecified single-dataset floor and a mean selected-score AURC change of {ext['decision']['mean_selected_aurc_delta_vs_confidence']:+.6f} (lower is better) failed their respective checks. These outcomes constrain the claim to improved average robustness with dataset-dependent exceptions, rather than uniform superiority or uniformly improved reliability.

## Independent confirmation

Before the ten-dataset extension, V5.1 was independently evaluated on four previously untouched datasets using three fresh seeds per model. All four dataset-level mean differences relative to V4.1 were positive, with an average difference of {indep_decision['average_dataset_unseen_macro_f1_delta_vs_v4_1']:+.4f} and a minimum dataset mean difference of {indep_decision['minimum_dataset_unseen_macro_f1_delta_vs_v4_1']:+.4f}. Test data were not used for model or reliability-mode selection. This confirmation therefore provides the strongest evidence that the signed spatial dual-path design transfers beyond its development datasets.

## Component ablation

The retrained component ablation was retrospective and descriptive because its two test sets had already been accessed. Removing the signed spatial path reduced mean unseen-rate macro-F1 by {np.mean(delta_by_variant['no_signed_spatial_path']):+.4f} across the two datasets and was unfavorable in five of six seed-level comparisons. Replacing the richer temporal summaries with a mean-only summary changed performance by {np.mean(delta_by_variant['mean_only_temporal_summary']):+.4f} and was favorable in all six seed-level comparisons. Replacing adaptive fusion and the cross-path residual with fixed equal fusion changed performance by {np.mean(delta_by_variant['fixed_equal_fusion']):+.4f} and was favorable in five of six seed-level comparisons. Thus, this small explanatory panel supports the signed spatial path as the principal component contribution, but it does not support claiming that every additional fusion or summary mechanism is individually necessary.

## Sequential efficiency

On the same local CUDA device under sequential inference-only measurement, V5.1 used {int(v5eff['parameters']):,} parameters and {int(v5eff['flops_per_sample']):,} FLOPs per sample, with a mean latency of {v5eff['latency_ms_per_sample_mean']:.4f} ms per sample. The V4.1 reference used {int(v4eff['parameters']):,} parameters, {int(v4eff['flops_per_sample']):,} FLOPs per sample, and {v4eff['latency_ms_per_sample_mean']:.4f} ms per sample. These are implementation- and hardware-specific measurements and should not be generalized to other devices.

## Evidence-bounded conclusion

Across the frozen ten-dataset extension, V5.1 had the highest average and worst-unseen macro-F1 of the evaluated implementations and showed a statistically supported advantage over the fixed-rate TCN. Independent confirmation was positive on all four dataset means. The results do not establish uniform superiority over every strong baseline: comparisons with ROCKET methods were not significant after multiplicity correction, PAMAP2 showed a substantial negative effect relative to v3.10, and the reliability non-worsening gate was not met. The component evidence most directly supports the signed spatial waveform path; the more complex summary and adaptive-fusion choices remain candidates for future simplification rather than required contributions of the present model.
"""
    chinese = f"""# V5.1 最终结果中文速查

## 可以作为论文主结论的内容

- V5.1 在十数据集上的 mean-unseen macro-F1 为 `{v5['mean_unseen_macro_f1']:.6f}`，是八种实现中最高；worst-unseen macro-F1 为 `{v5['worst_unseen_macro_f1']:.6f}`，也是最高。
- 相对固定采样率 TCN，平均提高 `{fixed['mean_delta']:+.6f}`，95% bootstrap CI 为 `[{fixed['dataset_clustered_bootstrap_95_ci'][0]:+.6f}, {fixed['dataset_clustered_bootstrap_95_ci'][1]:+.6f}]`，Holm 校正后 p=`{fixed['holm_adjusted_p']:.6f}`，这是七项比较中唯一通过校正显著性的结果。
- 四个未见数据集的独立确认全部为正向，dataset 平均增益为 `{indep_decision['average_dataset_unseen_macro_f1_delta_vs_v4_1']:+.6f}`。
- 两数据集重训练消融显示：移除 signed spatial path 平均下降 `{np.mean(delta_by_variant['no_signed_spatial_path']):.6f}`，因此它是目前证据最强的核心组件。

## 必须同时报告的边界

- Full 严格决策为 FAIL，而不是 PASS：PAMAP2 相对 v3.10 下降 `{v3['dataset_deltas']['pamap2_uci']:.6f}`，可靠性 AURC 非劣检查也未通过。
- V5.1 相对 MiniROCKET 和 MultiROCKET 的总体平均值更高，但校正后没有统计显著性，不能写成“显著优于所有强基线”。
- mean-only summary 和 fixed equal fusion 在该小型消融中反而更好；因此正文不能声称 V5.1 的三个复杂组件均被证明必要。
- 消融只有两个已访问 TEST 的数据集，属于描述性解释，不能把三个 seed 当成六个独立重复，也不能用于重新选择并覆盖现有 V5.1 结果。

## 当前最稳妥的论文定位

论文应定位为：一个在未知/降低采样率下具有较强平均鲁棒性、在四个新数据集上得到独立确认，并具有明确空间波形路径贡献的工程型时序分类方法；同时透明报告数据集异质性、可靠性限制和相对 ROCKET 基线尚未达到校正显著性。V5.1 已冻结，不需要继续训练。
"""
    captions = """# Figure captions and alt text

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
"""
    paths = [out / "v5_1_results_draft_en.md", out / "v5_1_results_summary_zh.md",
             out / "figure_captions_and_alt_text.md"]
    for path, text in zip(paths, (english, chinese, captions)):
        atomic_text(path, text)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true", help="replace an existing generated package")
    parser.add_argument("--check-only", action="store_true", help="audit frozen inputs without writing outputs")
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.output or root / "manuscript" / "final_results").resolve()
    bundle, source_paths = load_bundle(root)
    rate_summary = v5_rate_summary(bundle["extension"])
    if args.check_only:
        print(json.dumps({"status": "PASS", "sources": len(source_paths), "v5_rate_summary": rate_summary}, indent=2))
        return 0
    if out.exists() and any(out.iterdir()) and not args.force:
        raise FileExistsError(f"output directory is not empty; rerun with --force: {out}")
    out.mkdir(parents=True, exist_ok=True)
    tables = build_tables(bundle, out)
    figure_paths, rate_rows = build_figures(bundle, out)
    narrative_paths = build_narrative(bundle, out)
    generated = sorted(path for path in out.iterdir() if path.is_file() and path.name != "provenance_manifest.json")
    manifest = {
        "status": "generated_from_frozen_reports",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "publisher_profile": "provisional general scientific figure; target-journal submission rules pending verification",
        "primary_unit": "dataset; seeds are repeated training runs and are not treated as independent datasets",
        "transformations": [
            "V5.1 rate values: arithmetic mean across 10 dataset means, each based on 3 seeds",
            "paired effects and bootstrap intervals: reused verbatim from the frozen V5.1 extension report",
            "independent confirmation: dataset means over three fresh seeds",
            "component ablation: descriptive two-dataset comparison after retraining each variant",
        ],
        "source_files": {key: {"path": str(path), "sha256": sha256(path)} for key, path in source_paths.items()},
        "output_files": {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in generated},
        "counts": {"table_rows": {key: len(value) for key, value in tables.items()},
                   "rate_curve_rows": len(rate_rows), "figure_files": len(figure_paths),
                   "narrative_files": len(narrative_paths)},
        "scientific_boundaries": {
            "v5_1_full_extension": "retrospective extension of a previously accessed Full test matrix",
            "independent_confirmation": "independent; test unused for selection",
            "component_ablation": "retrospective descriptive; n=2 datasets",
            "strict_full_decision": bundle["extension"]["decision"]["passed"],
        },
    }
    atomic_text(out / "provenance_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"V5.1 paper-results package: PASS\nOutput: {out}\nFiles: {len(generated) + 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
