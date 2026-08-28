from pathlib import Path

from scripts.build_v5_1_paper_results import load_bundle, v5_rate_summary


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_v5_1_paper_sources_pass_audit() -> None:
    bundle, paths = load_bundle(ROOT)
    assert set(paths) == {"full", "extension", "independent", "ablation", "efficiency"}
    assert bundle["extension"]["new_candidate_runs"] == 30
    assert bundle["full"]["completed_runs"] == 210
    assert bundle["ablation"]["completed_runs"] == 24


def test_v5_1_rate_summary_uses_all_frozen_runs() -> None:
    bundle, _ = load_bundle(ROOT)
    summary = v5_rate_summary(bundle["extension"])
    assert set(summary) == {"r1000", "r0900", "r0600", "r0400", "r0300"}
    assert all(0.0 <= value <= 1.0 for value in summary.values())
    unseen_mean = sum(summary[key] for key in ("r0900", "r0600", "r0400", "r0300")) / 4
    expected = bundle["extension"]["method_summary"]["v5_1_safe_dual_path"]["mean_unseen_macro_f1"]
    assert abs(unseen_mean - expected) < 1e-12
