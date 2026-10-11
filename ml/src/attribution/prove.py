import json
from pathlib import Path

from src.training.train import train_model
from src.training.evaluate import (
    evaluate_clean_accuracy,
    evaluate_attack_success_rate,
)


SEEDS = [42, 123, 2024]
CONTRIBUTORS = [1, 2, 3, 4, 5]


def evaluate_model(contributors, seed):
    print(
        f"\nTraining contributors={contributors} "
        f"with seed={seed}"
    )

    model = train_model(
        contributors=tuple(contributors),
        seed=seed,
    )

    clean_accuracy = evaluate_clean_accuracy(model)
    asr = evaluate_attack_success_rate(model)

    return {
        "clean_accuracy": clean_accuracy,
        "asr": asr,
    }


def run_multi_seed_experiment():
    results = {
        "seeds": SEEDS,
        "all_contributors": {},
        "leave_one_out": {},
    }

    # --------------------------------------------------
    # Full model
    # --------------------------------------------------

    for seed in SEEDS:
        result = evaluate_model(
            CONTRIBUTORS,
            seed,
        )

        results["all_contributors"][str(seed)] = result

    # --------------------------------------------------
    # Leave-one-out
    # --------------------------------------------------

    for contributor_id in CONTRIBUTORS:
        results["leave_one_out"][f"C{contributor_id}"] = {}

        remaining = [
            contributor
            for contributor in CONTRIBUTORS
            if contributor != contributor_id
        ]

        for seed in SEEDS:
            result = evaluate_model(
                remaining,
                seed,
            )

            results["leave_one_out"][
                f"C{contributor_id}"
            ][str(seed)] = result

    output_path = (
        Path("experiments")
        / "results"
        / "multi_seed_leave_one_out.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        f"\nResults saved to: {output_path}"
    )


if __name__ == "__main__":
    run_multi_seed_experiment()