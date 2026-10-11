from src.training.train import train_model
from src.training.evaluate import (
    evaluate_clean_accuracy,
    evaluate_attack_success_rate,
)


CONTRIBUTORS = [1, 2, 3, 4, 5]


def run_group_test(group):
    group = set(group)

    print(
        f"\nTesting WITHOUT group: "
        f"{sorted(group)}"
    )

    remaining = [
        contributor
        for contributor in CONTRIBUTORS
        if contributor not in group
    ]

    model = train_model(
        contributors=tuple(remaining),
        seed=42,
        scenario="collusion",
    )

    clean_accuracy = evaluate_clean_accuracy(
        model
    )

    asr = evaluate_attack_success_rate(
        model
    )

    print(
        f"Remaining contributors: {remaining}"
    )

    print(
        f"Clean accuracy: "
        f"{clean_accuracy:.4f}"
    )

    print(
        f"ASR: "
        f"{asr:.4f}"
    )

    return {
        "removed_group": sorted(group),
        "remaining": remaining,
        "clean_accuracy": clean_accuracy,
        "asr": asr,
    }


def main():

    # Baseline: everyone participates.
    print("\n=== FULL COLLUSION SCENARIO ===")

    full_model = train_model(
        contributors=tuple(CONTRIBUTORS),
        seed=42,
        scenario="collusion",
    )

    full_clean_accuracy = evaluate_clean_accuracy(
        full_model
    )

    full_asr = evaluate_attack_success_rate(
        full_model
    )

    print(
        f"Full clean accuracy: "
        f"{full_clean_accuracy:.4f}"
    )

    print(
        f"Full ASR: "
        f"{full_asr:.4f}"
    )

    # Test the suspected colluding pair.
    result = run_group_test([2, 3])

    delta_asr = (
        full_asr
        - result["asr"]
    )

    print("\n=== GROUP ATTRIBUTION ===")

    print(
        f"ASR(all): "
        f"{full_asr:.4f}"
    )

    print(
        f"ASR(without C2+C3): "
        f"{result['asr']:.4f}"
    )

    print(
        f"ΔASR: "
        f"{delta_asr:.4f}"
    )


if __name__ == "__main__":
    main()