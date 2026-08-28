"""Top-level experiment entry used by both CLI and the local dashboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from nyquistguard.experiments.diagnosis import run_diagnosis  # noqa: E402
from nyquistguard.experiments.deterministic_selector_probe import (  # noqa: E402
    run_deterministic_selector_probe,
)
from nyquistguard.experiments.mechanism_probe import run_mechanism_probe  # noqa: E402
from nyquistguard.experiments.pilot import run_pilot  # noqa: E402
from nyquistguard.experiments.smoke import run_smoke  # noqa: E402
from nyquistguard.experiments.v2_micro_pilot import (  # noqa: E402
    run_selector_v2b,
    run_v2_micro_pilot,
)
from nyquistguard.experiments.v3_reliability import run_v3_reliability  # noqa: E402
from nyquistguard.experiments.v3_spectral_reliability import (  # noqa: E402
    run_v3_spectral_reliability,
)
from nyquistguard.experiments.v3_calibrated_reliability import (  # noqa: E402
    run_v3_calibrated_reliability,
)
from nyquistguard.experiments.v3_anchored_reliability import (  # noqa: E402
    run_v3_anchored_reliability,
)
from nyquistguard.experiments.v3_core_micro import (  # noqa: E402
    run_v3_core_micro,
    run_v3_core_refinement,
)
from nyquistguard.experiments.v3_guarded_reliability import (  # noqa: E402
    run_v3_guarded_reliability,
)
from nyquistguard.experiments.v3_multiseed_confirmation import (  # noqa: E402
    run_v3_multiseed_confirmation,
)
from nyquistguard.experiments.v3_stability_development import (  # noqa: E402
    run_v3_stability_development,
)
from nyquistguard.experiments.v3_weight_average import run_v3_weight_average  # noqa: E402
from nyquistguard.experiments.v3_logit_ensemble import run_v3_logit_ensemble  # noqa: E402
from nyquistguard.experiments.v3_low_rate_development import (  # noqa: E402
    run_v3_continuous_rate_development,
    run_v3_low_rate_development,
)
from nyquistguard.experiments.v3_10_independent_confirmation import (  # noqa: E402
    run_v3_10_independent_confirmation,
)
from nyquistguard.experiments.full import run_full  # noqa: E402
from nyquistguard.experiments.full_parallel import run_full_parallel  # noqa: E402
from nyquistguard.research.v4_observe_only_micro import (  # noqa: E402
    run_v4_observe_only_micro,
)
from nyquistguard.research.v4_residual_gate_micro import (  # noqa: E402
    run_v4_residual_gate_micro,
)
from nyquistguard.research.v4_residual_gate_matched import (  # noqa: E402
    run_v4_residual_gate_matched,
)
from nyquistguard.research.v4_residual_gate_multiseed import (  # noqa: E402
    run_v4_residual_gate_multiseed,
)
from nyquistguard.research.v4_new_dataset_confirmation import (  # noqa: E402
    run_v4_new_dataset_confirmation,
)
from nyquistguard.research.v5_dual_path_micro import (  # noqa: E402
    run_v5_dual_path_micro,
)
from nyquistguard.research.v5_four_dataset_benchmark import (  # noqa: E402
    run_v5_four_dataset_benchmark,
)
from nyquistguard.research.v5_safe_reliability import (  # noqa: E402
    run_v5_safe_reliability_development,
)
from nyquistguard.research.v5_independent_confirmation import (  # noqa: E402
    run_v5_1_independent_preflight,
    run_v5_1_independent_confirmation,
)
from nyquistguard.research.v5_full_extension import run_v5_1_full_extension  # noqa: E402
from nyquistguard.research.v5_efficiency import run_v5_1_efficiency  # noqa: E402
from nyquistguard.research.v5_component_ablation import (  # noqa: E402
    run_v5_1_component_ablation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="NyquistGuard-TSC experiment runner")
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "smoke",
            "pilot",
            "diagnosis",
            "mechanism_probe",
            "v2_micro",
            "selector_v2b",
            "deterministic_selector",
            "v3_reliability",
            "v3_spectral_reliability",
            "v3_calibrated_reliability",
            "v3_anchored_reliability",
            "v3_core_micro",
            "v3_core_refinement",
            "v3_guarded_reliability",
            "v3_multiseed_confirmation",
            "v3_stability_development",
            "v3_weight_average",
            "v3_logit_ensemble",
            "v3_low_rate_development",
            "v3_continuous_rate_development",
            "v3_10_independent_confirmation",
            "v4_observe_only_micro",
            "v4_residual_gate_micro",
            "v4_residual_gate_matched",
            "v4_residual_gate_multiseed_stability",
            "v4_new_dataset_confirmation",
            "v5_dual_path_micro",
            "v5_four_dataset_benchmark",
            "v5_safe_reliability_development",
            "v5_1_independent_confirmation",
            "v5_1_independent_preflight",
            "v5_1_full_extension",
            "v5_1_efficiency",
            "v5_1_component_ablation",
            "full",
            "full_parallel",
        ),
    )
    parser.add_argument("--resume", action="store_true", help="reuse a matching completed run or resume supported work")
    parser.add_argument(
        "--confirm-manual-start",
        action="store_true",
        help="required safety gate for pilot/full; the dashboard adds it only after a user confirmation",
    )
    args = parser.parse_args()

    if args.stage == "smoke":
        report = run_smoke(PROJECT_ROOT, resume=args.resume)
        print(f"Smoke status: {report['status']}", flush=True)
        print(f"Report: {PROJECT_ROOT / 'reports' / 'smoke_report.md'}", flush=True)
        return 0

    if args.stage == "diagnosis":
        report = run_diagnosis(PROJECT_ROOT, resume=args.resume)
        decision = report["go_no_go"]["decision"].upper()
        passed = report["go_no_go"]["passed_count"]
        print(f"Diagnosis status: {report['status']}; decision={decision}; passed={passed}/4", flush=True)
        print(f"Report: {PROJECT_ROOT / 'reports' / 'diagnostic_report.md'}", flush=True)
        print("No model was trained and Full was NOT started.", flush=True)
        return 0

    if args.stage == "mechanism_probe":
        report = run_mechanism_probe(PROJECT_ROOT)
        findings = report["aggregate"]
        print(
            "Mechanism probe status: completed; "
            f"selection_failures={len(findings['selection_failure_datasets'])}/4; "
            f"gate_collapse={len(findings['gate_collapse_datasets'])}/4",
            flush=True,
        )
        print(f"Report: {PROJECT_ROOT / 'reports' / 'mechanism_probe_report.md'}", flush=True)
        print("No optimizer or parameter update was used and Full was NOT started.", flush=True)
        return 0

    if args.stage == "v2_micro":
        report = run_v2_micro_pilot(PROJECT_ROOT, resume=args.resume)
        decision = report["decision"]
        print(
            "V2 micro-pilot status: completed; "
            f"selector={'PASS' if decision['selector']['passed'] else 'FAIL'}; "
            f"cbe={'PASS' if decision['cbe']['passed'] else 'FAIL'}; "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        print(f"Report: {PROJECT_ROOT / 'reports' / 'v2_micro_report.md'}", flush=True)
        print("This development run did not start Pilot or Full.", flush=True)
        return 0

    if args.stage == "selector_v2b":
        report = run_selector_v2b(PROJECT_ROOT)
        print(
            "Selector v2b status: completed; "
            f"decision={'PASS' if report['decision']['passed'] else 'FAIL'}; "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        print(f"Report: {PROJECT_ROOT / 'reports' / 'selector_v2b_report.md'}", flush=True)
        print("The classifier was frozen and Pilot/Full were NOT started.", flush=True)
        return 0

    if args.stage == "deterministic_selector":
        report = run_deterministic_selector_probe(PROJECT_ROOT)
        print(
            "Deterministic selector probe: "
            f"{'PASS' if report['decision']['candidate_passed'] else 'FAIL'}; "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        print(
            f"Report: {PROJECT_ROOT / 'reports' / 'deterministic_selector_report.md'}",
            flush=True,
        )
        print("No training occurred and Pilot/Full were NOT started.", flush=True)
        return 0

    if args.stage == "v3_reliability":
        report = run_v3_reliability(PROJECT_ROOT)
        print(
            "V3 reliability development: "
            f"{'PASS' if report['decision']['passed'] else 'FAIL'}; "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        print(
            f"Report: {PROJECT_ROOT / 'reports' / 'v3_reliability_report.md'}",
            flush=True,
        )
        print("No training occurred and Pilot/Full were NOT started.", flush=True)
        return 0

    if args.stage == "v3_spectral_reliability":
        report = run_v3_spectral_reliability(PROJECT_ROOT)
        print(
            "V3.1 spectral reliability development: "
            f"{'PASS' if report['decision']['passed'] else 'FAIL'}; "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        print(
            f"Report: {PROJECT_ROOT / 'reports' / 'v3_spectral_reliability_report.md'}",
            flush=True,
        )
        print("No training occurred and Pilot/Full were NOT started.", flush=True)
        return 0

    if args.stage == "v3_calibrated_reliability":
        report = run_v3_calibrated_reliability(PROJECT_ROOT)
        print(
            "V3.2 calibrated reliability development: "
            f"{'PASS' if report['decision']['passed'] else 'FAIL'}; "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        print(
            f"Report: {PROJECT_ROOT / 'reports' / 'v3_calibrated_reliability_report.md'}",
            flush=True,
        )
        print("Classification checkpoints were frozen and Pilot/Full were NOT started.", flush=True)
        return 0

    if args.stage == "v3_anchored_reliability":
        report = run_v3_anchored_reliability(PROJECT_ROOT)
        print(
            "V3.3 anchored reliability development: "
            f"{'PASS' if report['decision']['passed'] else 'FAIL'}; "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        print(
            f"Report: {PROJECT_ROOT / 'reports' / 'v3_anchored_reliability_report.md'}",
            flush=True,
        )
        print("Classification checkpoints were frozen and Pilot/Full were NOT started.", flush=True)
        return 0

    if args.stage == "v3_core_micro":
        report = run_v3_core_micro(PROJECT_ROOT, resume=args.resume)
        print(
            "V3.3 core micro: "
            f"{'PASS' if report['decision']['passed'] else 'FAIL'}; "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        print(
            f"Report: {PROJECT_ROOT / 'reports' / 'v3_core_micro_report.md'}",
            flush=True,
        )
        print("Pilot/Full were NOT started.", flush=True)
        return 0


    if args.stage == "v3_core_refinement":
        report = run_v3_core_refinement(PROJECT_ROOT, resume=args.resume)
        print(
            "V3.4 core refinement: "
            f"{'PASS' if report['decision']['passed'] else 'FAIL'}; "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        print(
            f"Report: {PROJECT_ROOT / 'reports' / 'v3_core_refinement_report.md'}",
            flush=True,
        )
        print("Pilot/Full were NOT started.", flush=True)
        return 0


    if args.stage == "v3_guarded_reliability":
        report = run_v3_guarded_reliability(PROJECT_ROOT)
        print(
            "V3.5 guarded reliability: "
            f"{'PASS' if report['decision']['passed'] else 'FAIL'}; "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        print(
            f"Report: {PROJECT_ROOT / 'reports' / 'v3_guarded_reliability_report.md'}",
            flush=True,
        )
        print("Pilot/Full were NOT started.", flush=True)
        return 0


    if args.stage == "v3_multiseed_confirmation":
        report = run_v3_multiseed_confirmation(PROJECT_ROOT, resume=args.resume)
        print(
            "V3.5 multiseed confirmation: "
            f"{'PASS' if report['decision']['passed'] else 'FAIL'}; "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        print(
            f"Report: {PROJECT_ROOT / 'reports' / 'v3_multiseed_confirmation_report.md'}",
            flush=True,
        )
        print("Pilot/Full were NOT started.", flush=True)
        return 0


    if args.stage == "v3_stability_development":
        report = run_v3_stability_development(PROJECT_ROOT, resume=args.resume)
        print(
            "V3.6 stability development: "
            f"{'PASS' if report['decision']['passed'] else 'FAIL'}; "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        print(
            f"Report: {PROJECT_ROOT / 'reports' / 'v3_stability_development_report.md'}",
            flush=True,
        )
        print("Pilot/Full were NOT started.", flush=True)
        return 0


    if args.stage == "v3_weight_average":
        report = run_v3_weight_average(PROJECT_ROOT)
        print(
            "V3.7 weight average: "
            f"{'PASS' if report['decision']['passed'] else 'FAIL'}; "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        print(
            f"Report: {PROJECT_ROOT / 'reports' / 'v3_weight_average_report.md'}",
            flush=True,
        )
        print("Pilot/Full were NOT started.", flush=True)
        return 0

    if args.stage == "v3_logit_ensemble":
        report = run_v3_logit_ensemble(PROJECT_ROOT)
        print(
            "V3.8 fixed logit ensemble: "
            f"{'PASS' if report['decision']['passed'] else 'FAIL'}; "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        print(
            f"Report: {PROJECT_ROOT / 'reports' / 'v3_logit_ensemble_report.md'}",
            flush=True,
        )
        print("Pilot/Full were NOT started.", flush=True)
        return 0

    if args.stage == "v3_low_rate_development":
        report = run_v3_low_rate_development(PROJECT_ROOT, resume=args.resume)
        print(
            "V3.9 low-rate exposure development: "
            f"{'PASS' if report['decision']['passed'] else 'FAIL'}; "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        print(
            f"Report: {PROJECT_ROOT / 'reports' / 'v3_low_rate_development_report.md'}",
            flush=True,
        )
        print("Pilot/Full were NOT started.", flush=True)
        return 0

    if args.stage == "v3_continuous_rate_development":
        report = run_v3_continuous_rate_development(PROJECT_ROOT, resume=args.resume)
        print(
            "V3.10 continuous-rate augmentation: "
            f"{'PASS' if report['decision']['passed'] else 'FAIL'}; "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        print(
            f"Report: {PROJECT_ROOT / 'reports' / 'v3_continuous_rate_development_report.md'}",
            flush=True,
        )
        print("Pilot/Full were NOT started.", flush=True)
        return 0

    if args.stage == "v3_10_independent_confirmation":
        if not args.confirm_manual_start:
            print(
                "v3.10 independent confirmation was not started: explicit manual confirmation is required.",
                file=sys.stderr,
                flush=True,
            )
            return 3
        report = run_v3_10_independent_confirmation(
            PROJECT_ROOT, resume=args.resume, confirmed=True
        )
        print(
            "V3.10 independent confirmation: "
            f"{'PASS' if report['decision']['passed'] else 'FAIL'}; "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        print(
            f"Report: {PROJECT_ROOT / 'reports' / 'v3_10_independent_confirmation_report.md'}",
            flush=True,
        )
        print("Pilot/Full were NOT started.", flush=True)
        return 0

    if args.stage == "v4_observe_only_micro":
        report = run_v4_observe_only_micro(PROJECT_ROOT, resume=args.resume)
        print(
            "V4 observe-only validation micro: "
            f"{'PASS' if report['decision']['passed'] else 'FAIL'}; "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        print(
            f"Report: {PROJECT_ROOT / 'reports' / 'v4_observe_only_micro_report.md'}",
            flush=True,
        )
        print("Existing test splits, Pilot, and Full were NOT started or scored.", flush=True)
        return 0

    if args.stage == "v4_residual_gate_micro":
        report = run_v4_residual_gate_micro(PROJECT_ROOT, resume=args.resume)
        print(
            "V4.1 residual-gate validation micro: "
            f"{'PASS' if report['decision']['passed'] else 'FAIL'}; "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        print(f"Report: {PROJECT_ROOT / 'reports' / 'v4_residual_gate_micro_report.md'}", flush=True)
        print("Existing test splits, Pilot, and Full were NOT started or scored.", flush=True)
        return 0

    if args.stage == "v4_residual_gate_matched":
        report = run_v4_residual_gate_matched(PROJECT_ROOT, resume=args.resume)
        print(
            "V4.1 matched-budget validation audit: "
            f"{'PASS' if report['decision']['passed'] else 'FAIL'}; "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        print(f"Report: {PROJECT_ROOT / 'reports' / 'v4_residual_gate_matched_report.md'}", flush=True)
        print("Existing test splits, Pilot, and Full were NOT started or scored.", flush=True)
        return 0

    if args.stage == "v4_residual_gate_multiseed_stability":
        if not args.confirm_manual_start:
            print(
                "V4.1 multi-seed stability was not started: explicit manual confirmation is required.",
                file=sys.stderr,
                flush=True,
            )
            return 3
        report = run_v4_residual_gate_multiseed(
            PROJECT_ROOT, resume=args.resume, confirmed=True
        )
        print(
            "V4.1 multi-seed validation stability: "
            f"{'PASS' if report['decision']['passed'] else 'FAIL'}; "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        print(
            f"Report: {PROJECT_ROOT / 'reports' / 'v4_residual_gate_multiseed_report.md'}",
            flush=True,
        )
        print("Existing test splits, Pilot, Full, and new-dataset confirmation were NOT started.", flush=True)
        return 0

    if args.stage == "v4_new_dataset_confirmation":
        if not args.confirm_manual_start:
            print(
                "V4.1 four-new-dataset confirmation was not started: explicit manual confirmation is required.",
                file=sys.stderr,
                flush=True,
            )
            return 3
        report = run_v4_new_dataset_confirmation(
            PROJECT_ROOT, resume=args.resume, confirmed=True
        )
        print(
            "V4.1 four-new-dataset confirmation: "
            f"{'PASS' if report['decision']['passed'] else 'FAIL'}; "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        print(
            f"Report: {PROJECT_ROOT / 'reports' / 'v4_new_dataset_confirmation_report.md'}",
            flush=True,
        )
        print("No later experiment stage was started automatically.", flush=True)
        return 0

    if args.stage == "v5_dual_path_micro":
        if not args.confirm_manual_start:
            print(
                "V5 validation micro was not started: explicit manual confirmation is required.",
                file=sys.stderr,
                flush=True,
            )
            return 3
        report = run_v5_dual_path_micro(
            PROJECT_ROOT, resume=args.resume, confirmed=True
        )
        print(
            "V5 dual-path validation micro: "
            f"{'PASS' if report['decision']['passed'] else 'FAIL'}; "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        print(
            f"Report: {PROJECT_ROOT / 'reports' / 'v5_dual_path_micro_report.md'}",
            flush=True,
        )
        print("No test split or later experiment was started.", flush=True)
        return 0

    if args.stage == "v5_four_dataset_benchmark":
        if not args.confirm_manual_start:
            print(
                "V5 four-dataset benchmark was not started: explicit manual confirmation is required.",
                file=sys.stderr,
                flush=True,
            )
            return 3
        report = run_v5_four_dataset_benchmark(
            PROJECT_ROOT, resume=args.resume, confirmed=True
        )
        print(
            "V5 vs V4.1 four-dataset retrospective benchmark: "
            f"{'PASS' if report['decision']['passed'] else 'FAIL'}; "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        print(
            f"Report: {PROJECT_ROOT / 'reports' / 'v5_four_dataset_benchmark_report.md'}",
            flush=True,
        )
        print("This is not independent V5 confirmation; no later stage was started.", flush=True)
        return 0

    if args.stage == "v5_safe_reliability_development":
        report = run_v5_safe_reliability_development(PROJECT_ROOT)
        print(
            "V5.1 safe reliability development: "
            f"{'PASS' if report['decision']['passed'] else 'FAIL'}; "
            f"elapsed={report['elapsed_seconds']:.2f}s",
            flush=True,
        )
        print(
            f"Report: {PROJECT_ROOT / 'reports' / 'v5_safe_reliability_development_report.md'}",
            flush=True,
        )
        print("All V5 classifiers remained frozen; no later stage was started.", flush=True)
        return 0

    if args.stage == "v5_1_independent_confirmation":
        if not args.confirm_manual_start:
            print(
                "V5.1 independent confirmation was not started: explicit manual confirmation is required.",
                file=sys.stderr,
                flush=True,
            )
            return 3
        report = run_v5_1_independent_confirmation(
            PROJECT_ROOT, resume=args.resume, confirmed=True
        )
        print(
            "V5.1 four-untouched-dataset independent confirmation: "
            f"{'PASS' if report['decision']['passed'] else 'FAIL'}; "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        print(
            f"Report: {PROJECT_ROOT / 'reports' / 'v5_1_independent_confirmation_report.md'}",
            flush=True,
        )
        print("No later stage was started automatically.", flush=True)
        return 0

    if args.stage == "v5_1_independent_preflight":
        report = run_v5_1_independent_preflight(PROJECT_ROOT)
        print(
            "V5.1 TRAIN-only independent-panel preflight: "
            f"{'PASS' if report['passed'] else 'FAIL'}; "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        print(
            f"Report: {PROJECT_ROOT / 'reports' / 'v5_1_independent_preflight_report.md'}",
            flush=True,
        )
        print("TEST, training, and later stages were NOT started.", flush=True)
        return 0

    if args.stage == "v5_1_full_extension":
        if not args.confirm_manual_start:
            print(
                "V5.1 Full extension was not started: explicit manual confirmation is required.",
                file=sys.stderr,
                flush=True,
            )
            return 3
        report = run_v5_1_full_extension(PROJECT_ROOT, resume=args.resume, confirmed=True)
        print(
            "V5.1 ten-dataset Full extension: "
            f"{'PASS' if report['decision']['passed'] else 'FAIL'}; "
            f"new={report['new_candidate_runs']}, reused={report['reused_full_runs']}; "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        print(
            f"Report: {PROJECT_ROOT / 'reports' / 'v5_1_full_extension_report.md'}",
            flush=True,
        )
        print("No later stage was started automatically.", flush=True)
        return 0

    if args.stage == "v5_1_efficiency":
        report = run_v5_1_efficiency(PROJECT_ROOT)
        print(
            f"V5.1 sequential efficiency: completed; elapsed={report['elapsed_seconds']:.1f}s; "
            f"device={report['device']}",
            flush=True,
        )
        print(f"Report: {PROJECT_ROOT / 'reports' / 'v5_1_efficiency_report.md'}", flush=True)
        print("No training or later stage was started.", flush=True)
        return 0

    if args.stage == "v5_1_component_ablation":
        if not args.confirm_manual_start:
            print(
                "V5.1 component ablation was not started: explicit manual confirmation is required.",
                file=sys.stderr,
                flush=True,
            )
            return 3
        report = run_v5_1_component_ablation(
            PROJECT_ROOT, resume=args.resume, confirmed=True
        )
        print(
            f"V5.1 retrained component ablation: completed={report['completed_runs']}/24; "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        print(
            f"Report: {PROJECT_ROOT / 'reports' / 'v5_1_component_ablation_report.md'}",
            flush=True,
        )
        print("No later stage was started automatically.", flush=True)
        return 0

    if args.stage == "pilot":
        if not args.confirm_manual_start:
            print("Pilot was not started: explicit manual confirmation is required.", file=sys.stderr, flush=True)
            return 3
        report = run_pilot(PROJECT_ROOT, resume=args.resume, confirmed=True)
        print(f"Pilot status: {report['status']}", flush=True)
        print(f"Report: {PROJECT_ROOT / 'reports' / 'pilot_go_no_go.md'}", flush=True)
        print("Full was NOT started; it always requires a separate manual action.", flush=True)
        return 0

    if args.stage == "full_parallel":
        if not args.confirm_manual_start:
            print(
                "Parallel Full was not started: explicit manual confirmation is required.",
                file=sys.stderr,
                flush=True,
            )
            return 3
        report = run_full_parallel(PROJECT_ROOT, resume=args.resume, confirmed=True)
        print(
            f"Parallel Full status: {report['status']}; completed={report['completed_runs']}/210; "
            f"parallel_elapsed={report.get('parallel_elapsed_seconds', 0.0):.1f}s",
            flush=True,
        )
        print(f"Report: {PROJECT_ROOT / 'reports' / 'full_report.md'}", flush=True)
        print("No follow-up experiment was started automatically.", flush=True)
        return 0

    if not args.confirm_manual_start:
        print("Full was not started: explicit manual confirmation is required.", file=sys.stderr, flush=True)
        return 3
    report = run_full(PROJECT_ROOT, resume=args.resume, confirmed=True)
    print(
        f"Full status: {report['status']}; completed={report['completed_runs']}/210; "
        f"elapsed={report['elapsed_seconds']:.1f}s",
        flush=True,
    )
    print(f"Report: {PROJECT_ROOT / 'reports' / 'full_report.md'}", flush=True)
    print("No follow-up experiment was started automatically.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
