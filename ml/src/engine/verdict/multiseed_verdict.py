class MultiSeedVerdictEngine:
    """
    Final statistical verdict engine for multi-seed
    leave-one-out forensic evidence.
    """

    CAUSAL_FRACTION = 0.50
    CLEARED_FRACTION = 0.10
    MAX_CLEAN_ACCURACY_DROP = 0.02

    def classify(
        self,
        baseline_asr_mean,
        delta_asr_mean,
        delta_asr_std,
        clean_accuracy_drop_mean,
    ):
        causal_threshold = (
            self.CAUSAL_FRACTION
            * baseline_asr_mean
        )

        cleared_threshold = (
            self.CLEARED_FRACTION
            * baseline_asr_mean
        )

        statistical_threshold = (
            2.0 * delta_asr_std
        )

        passes_causal_threshold = (
            delta_asr_mean
            >= causal_threshold
        )

        passes_statistical_threshold = (
            delta_asr_mean
            > statistical_threshold
        )

        passes_clean_accuracy_guard = (
            clean_accuracy_drop_mean
            <= self.MAX_CLEAN_ACCURACY_DROP
        )

        is_cleared = (
            delta_asr_mean
            < cleared_threshold
        )

        if (
            passes_causal_threshold
            and passes_statistical_threshold
            and passes_clean_accuracy_guard
        ):
            verdict = "CAUSALLY IMPLICATED"

        elif is_cleared:
            verdict = "CLEARED"

        else:
            verdict = "INCONCLUSIVE"

        return {
            "delta_asr_mean": delta_asr_mean,
            "delta_asr_std": delta_asr_std,
            "clean_accuracy_drop_mean": (
                clean_accuracy_drop_mean
            ),
            "causal_threshold": causal_threshold,
            "cleared_threshold": cleared_threshold,
            "statistical_threshold": (
                statistical_threshold
            ),
            "max_clean_accuracy_drop": (
                self.MAX_CLEAN_ACCURACY_DROP
            ),
            "passes_causal_threshold": (
                passes_causal_threshold
            ),
            "passes_statistical_threshold": (
                passes_statistical_threshold
            ),
            "passes_clean_accuracy_guard": (
                passes_clean_accuracy_guard
            ),
            "verdict": verdict,
        }


def print_verdict(
    contributor,
    result,
):
    print(
        f"{contributor}: "
        f"ΔASR={result['delta_asr_mean']:.6f} "
        f"+/- {result['delta_asr_std']:.6f} | "
        f"2×std={result['statistical_threshold']:.6f} | "
        f"CleanDrop="
        f"{result['clean_accuracy_drop_mean']:.6f} | "
        f"Verdict={result['verdict']}"
    )


if __name__ == "__main__":

    engine = MultiSeedVerdictEngine()

    test_result = engine.classify(
        baseline_asr_mean=0.9940,
        delta_asr_mean=0.986815,
        delta_asr_std=0.000962,
        clean_accuracy_drop_mean=0.008633,
    )

    print("\n=== MULTI-SEED VERDICT TEST ===")

    print_verdict(
        "C3",
        test_result,
    )