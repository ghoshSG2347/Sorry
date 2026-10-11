class VerdictEngine:
    """
    PoisonTrace final contributor verdict engine.

    Uses the pre-registered attribution rules:

    CAUSALLY IMPLICATED:
        ΔASR >= 50% of ASR(all)
        AND
        ΔASR > 2 × pooled standard deviation

    CLEARED:
        ΔASR < 10% of ASR(all)

    Otherwise:
        INCONCLUSIVE
    """

    def classify(
        self,
        asr_all,
        asr_without,
        pooled_std,
    ):
        delta_asr = (
            asr_all - asr_without
        )

        causal_threshold = (
            0.50 * asr_all
        )

        cleared_threshold = (
            0.10 * asr_all
        )

        statistical_threshold = (
            2 * pooled_std
        )

        if (
            delta_asr >= causal_threshold
            and delta_asr > statistical_threshold
        ):
            verdict = "CAUSALLY IMPLICATED"

        elif delta_asr < cleared_threshold:
            verdict = "CLEARED"

        else:
            verdict = "INCONCLUSIVE"

        return {
            "delta_asr": delta_asr,
            "causal_threshold": causal_threshold,
            "cleared_threshold": cleared_threshold,
            "statistical_threshold": (
                statistical_threshold
            ),
            "verdict": verdict,
        }


def print_verdict(
    contributor_id,
    result,
):
    print(
        f"C{contributor_id}: "
        f"ΔASR={result['delta_asr']:.4f} | "
        f"verdict={result['verdict']}"
    )


if __name__ == "__main__":

    engine = VerdictEngine()

    result = engine.classify(
        asr_all=0.9944,
        asr_without=0.0066,
        pooled_std=0.0022,
    )

    print("\n=== VERDICT TEST ===")
    print_verdict(
        contributor_id=3,
        result=result,
    )