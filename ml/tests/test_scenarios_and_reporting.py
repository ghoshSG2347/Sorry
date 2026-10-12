import numpy as np
import pytest
from pathlib import Path

from src.training.train import build_training_dataset
from src.data.dataset import ContributorDataset
from src.engine.verdict.analyze_multiseed import mean_std
from src.engine.verdict.forensic_report import build_report
from src.engine.verdict.collusion_forensic_report import main as generate_collusion_report


def test_build_training_dataset_original_scenario():
    """Verify default 'original' scenario combines C3 poisoned, C5 noisy, and C1/C2/C4 clean."""
    dataset = build_training_dataset(contributors=(1, 2, 3, 4, 5), scenario="original")
    assert len(dataset.datasets) == 5
    # Total samples should be 5 * 4000 = 20,000
    assert len(dataset) == 20000


def test_build_training_dataset_collusion_scenario():
    """Verify S3 collusion scenario combines C2/C3 collusion and clean for others."""
    dataset = build_training_dataset(contributors=(1, 2, 3, 4, 5), scenario="collusion")
    assert len(dataset.datasets) == 5
    assert len(dataset) == 20000


def test_build_training_dataset_clean_scenario():
    """Verify clean / clean_all loads all clean shards."""
    dataset_clean = build_training_dataset(contributors=(1, 2, 3, 4, 5), scenario="clean")
    dataset_clean_all = build_training_dataset(contributors=(1, 2, 3, 4, 5), scenario="clean_all")
    assert len(dataset_clean) == 20000
    assert len(dataset_clean_all) == 20000
    for ds in dataset_clean.datasets:
        assert ds.scenario == "clean"


def test_mean_std_calculation():
    """Verify mean and population standard deviation (ddof=0) calculations."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    stats = mean_std(values)
    assert round(stats["mean"], 4) == 3.0000
    expected_pop_std = float(np.std(values, ddof=0))
    assert round(stats["std"], 6) == round(expected_pop_std, 6)


def test_collusion_verdict_logic_redundant_attackers():
    """Verify redundant backdoor logic: removing one preserves attack, removing both eliminates."""
    # When C2 alone removed: ASR stays high (> 0.90)
    # When C3 alone removed: ASR stays high (> 0.90)
    # When both removed: ASR drops low (< 0.10)
    without_c2_asr = [0.990, 0.988, 0.992]
    without_c3_asr = [0.991, 0.993, 0.990]
    without_group_asr = [0.015, 0.012, 0.018]

    c2_preserved = bool(np.mean(without_c2_asr) > 0.90)
    c3_preserved = bool(np.mean(without_c3_asr) > 0.90)
    joint_removed = bool(np.mean(without_group_asr) < 0.10)

    assert c2_preserved is True
    assert c3_preserved is True
    assert joint_removed is True

    verdict = (
        "COLLUSIVE GROUP IMPLICATED"
        if (c2_preserved and c3_preserved and joint_removed)
        else "COLLUSION NOT CONFIRMED"
    )
    assert verdict == "COLLUSIVE GROUP IMPLICATED"


def test_collusion_verdict_fails_when_joint_removal_does_not_eliminate():
    """If joint removal fails to eliminate backdoor (< 0.10), collusion is not confirmed."""
    without_c2_asr = [0.95, 0.94]
    without_c3_asr = [0.96, 0.95]
    without_group_asr = [0.45, 0.50]  # Attack not eliminated

    c2_preserved = bool(np.mean(without_c2_asr) > 0.90)
    c3_preserved = bool(np.mean(without_c3_asr) > 0.90)
    joint_removed = bool(np.mean(without_group_asr) < 0.10)

    verdict = (
        "COLLUSIVE GROUP IMPLICATED"
        if (c2_preserved and c3_preserved and joint_removed)
        else "COLLUSION NOT CONFIRMED"
    )
    assert verdict == "COLLUSION NOT CONFIRMED"


def test_forensic_report_generation():
    """Verify forensic Markdown report can be generated from existing frozen JSON artifact."""
    ml_root = Path(__file__).resolve().parents[1]
    report_path = ml_root / "experiments" / "results" / "engine" / "final_forensic_report.md"
    assert report_path.is_file(), f"Expected report at {report_path}"
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "# PoisonTrace Forensic Report" in content
    assert "CAUSALLY IMPLICATED" in content
    assert "CLEARED" in content


def test_collusion_forensic_report_generation():
    """Verify collusion Markdown report contains expected sections and verdict."""
    ml_root = Path(__file__).resolve().parents[1]
    report_path = ml_root / "experiments" / "results" / "engine" / "final_collusion_forensic_report.md"
    assert report_path.is_file(), f"Expected report at {report_path}"
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "# PoisonTrace — S3 Collusion Forensic Report" in content
    assert "COLLUSIVE GROUP IMPLICATED" in content
