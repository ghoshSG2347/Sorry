from src.engine.verdict.multiseed_verdict import MultiSeedVerdictEngine


def test_c3_passes_all_causal_conditions():
    engine = MultiSeedVerdictEngine()
    result = engine.classify(
        baseline_asr_mean=0.9940,
        delta_asr_mean=0.986815,
        delta_asr_std=0.001178,
        clean_accuracy_drop_mean=0.008633,
    )
    assert result["verdict"] == "CAUSALLY IMPLICATED"
    assert result["passes_causal_threshold"] is True
    assert result["passes_statistical_threshold"] is True
    assert result["passes_clean_accuracy_guard"] is True


def test_c5_is_cleared():
    engine = MultiSeedVerdictEngine()
    result = engine.classify(
        baseline_asr_mean=0.9940,
        delta_asr_mean=0.005926,
        delta_asr_std=0.004578,
        clean_accuracy_drop_mean=0.007967,
    )
    assert result["verdict"] == "CLEARED"


def test_large_asr_drop_with_bad_clean_accuracy_is_not_causal():
    engine = MultiSeedVerdictEngine()
    result = engine.classify(
        baseline_asr_mean=0.99,
        delta_asr_mean=0.80,
        delta_asr_std=0.01,
        clean_accuracy_drop_mean=0.05,
    )
    assert result["verdict"] != "CAUSALLY IMPLICATED"
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["passes_clean_accuracy_guard"] is False


def test_high_variance_leads_to_inconclusive():
    engine = MultiSeedVerdictEngine()
    # High ASR drop, but high standard deviation violates 2*std stability heuristic
    result = engine.classify(
        baseline_asr_mean=0.99,
        delta_asr_mean=0.60,
        delta_asr_std=0.35,
        clean_accuracy_drop_mean=0.01,
    )
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["passes_statistical_threshold"] is False


def test_clean_scenario_all_cleared():
    engine = MultiSeedVerdictEngine()
    # In clean scenario, delta_asr is near 0 for all honest contributors
    for delta in [-0.002, 0.001, 0.005, -0.001, 0.003]:
        result = engine.classify(
            baseline_asr_mean=0.99,
            delta_asr_mean=delta,
            delta_asr_std=0.002,
            clean_accuracy_drop_mean=0.005,
        )
        assert result["verdict"] == "CLEARED"
    