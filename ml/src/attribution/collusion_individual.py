from src.training.train import train_model
from src.training.evaluate import (
    evaluate_clean_accuracy,
    evaluate_attack_success_rate,
)

CONTRIBUTORS = [1, 2, 3, 4, 5]


def test_without(contributor_id):
    remaining = [
        c for c in CONTRIBUTORS
        if c != contributor_id
    ]

    print(
        f"\n=== WITHOUT C{contributor_id} ==="
    )

    model = train_model(
        contributors=tuple(remaining),
        seed=42,
        scenario="collusion",
    )

    clean_accuracy = evaluate_clean_accuracy(model)
    asr = evaluate_attack_success_rate(model)

    print(
        f"Remaining: {remaining}"
    )
    print(
        f"Clean accuracy: {clean_accuracy:.4f}"
    )
    print(
        f"ASR: {asr:.4f}"
    )


if __name__ == "__main__":
    test_without(2)
    test_without(3)