def calculate_delta_asr(asr_all, asr_without):
    return asr_all - asr_without


def group_verdict(
    asr_all,
    asr_without_group,
    group,
):
    delta_asr = calculate_delta_asr(
        asr_all,
        asr_without_group,
    )

    if delta_asr >= 0.50 * asr_all:
        verdict = "COLLUDING GROUP IMPLICATED"
    elif delta_asr < 0.10 * asr_all:
        verdict = "GROUP CLEARED"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "group": group,
        "asr_all": asr_all,
        "asr_without_group": asr_without_group,
        "delta_asr": delta_asr,
        "verdict": verdict,
    }


if __name__ == "__main__":

    result = group_verdict(
        asr_all=0.9932,
        asr_without_group=0.0214,
        group=["C2", "C3"],
    )

    print("\n=== S3 COLLUSION VERDICT ===")

    print(
        f"Group: "
        f"{', '.join(result['group'])}"
    )

    print(
        f"ASR(all): "
        f"{result['asr_all']:.4f}"
    )

    print(
        f"ASR(without group): "
        f"{result['asr_without_group']:.4f}"
    )

    print(
        f"ΔASR: "
        f"{result['delta_asr']:.4f}"
    )

    print(
        f"VERDICT: "
        f"{result['verdict']}"
    )