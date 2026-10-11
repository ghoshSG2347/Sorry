def calculate_delta_asr(asr_all, asr_without):
    return asr_all - asr_without


def classify_contributor(
    asr_all,
    asr_without,
    pooled_std=0.0,
):
    delta_asr = calculate_delta_asr(
        asr_all,
        asr_without,
    )

    if (
        delta_asr >= 0.50 * asr_all
        and delta_asr > 2 * pooled_std
    ):
        verdict = "CAUSALLY IMPLICATED"

    elif delta_asr < 0.10 * asr_all:
        verdict = "CLEARED"

    else:
        verdict = "INCONCLUSIVE"

    return {
        "delta_asr": delta_asr,
        "verdict": verdict,
    }