def classify(
    mean_delta,
    baseline_asr,
    delta_std,
    mean_clean_drop,
):
    causal_threshold = 0.50 * baseline_asr
    cleared_threshold = 0.10 * baseline_asr
    statistical_threshold = 2 * delta_std

    if (
        mean_delta >= causal_threshold
        and mean_delta > statistical_threshold
        and mean_clean_drop <= 0.02
    ):
        return "CAUSALLY IMPLICATED"

    if mean_delta < cleared_threshold:
        return "CLEARED"

    return "INCONCLUSIVE"


def test_c3_passes_all_causal_conditions():
    verdict = classify(
        mean_delta=0.986815,
        baseline_asr=0.9940,
        delta_std=0.001178,
        mean_clean_drop=0.008633,
    )

    assert verdict == "CAUSALLY IMPLICATED"


def test_c5_is_cleared():
    verdict = classify(
        mean_delta=0.005926,
        baseline_asr=0.9940,
        delta_std=0.004578,
        mean_clean_drop=0.007967,
    )

    assert verdict == "CLEARED"


def test_large_asr_drop_with_bad_clean_accuracy_is_not_causal():
    verdict = classify(
        mean_delta=0.80,
        baseline_asr=0.99,
        delta_std=0.01,
        mean_clean_drop=0.05,
    )

    assert verdict != "CAUSALLY IMPLICATED"
    