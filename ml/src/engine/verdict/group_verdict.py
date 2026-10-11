class GroupVerdictEngine:
    def classify(self, asr_all, asr_without_group):
        delta_asr = asr_all - asr_without_group

        implicated_threshold = 0.50 * asr_all
        cleared_threshold = 0.10 * asr_all

        if delta_asr >= implicated_threshold:
            verdict = "COLLUDING GROUP IMPLICATED"
        elif delta_asr < cleared_threshold:
            verdict = "GROUP CLEARED"
        else:
            verdict = "INCONCLUSIVE"

        return {
            "delta_asr": delta_asr,
            "implicated_threshold": implicated_threshold,
            "cleared_threshold": cleared_threshold,
            "verdict": verdict,
        }


def print_group_verdict(group, result):
    print("\n=== GROUP VERDICT ===")
    print(f"Group: {group}")
    print(f"ΔASR: {result['delta_asr']:.4f}")
    print(f"Verdict: {result['verdict']}")


if __name__ == "__main__":
    engine = GroupVerdictEngine()

    result = engine.classify(
        asr_all=0.9932,
        asr_without_group=0.0214,
    )

    print_group_verdict(
        group=["C2", "C3"],
        result=result,
    )