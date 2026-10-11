from pathlib import Path
import json
import numpy as np

from src.engine.prove.group_test import GroupAttributionEngine


SEEDS = [42, 123, 2024]

CONTRIBUTORS = [1, 2, 3, 4, 5]

TESTS = {
    "baseline": [1, 2, 3, 4, 5],
    "without_C2": [1, 3, 4, 5],
    "without_C3": [1, 2, 4, 5],
    "without_C2_C3": [1, 4, 5],
}


def run_test(contributors, seed):
    engine = GroupAttributionEngine(
        contributors=CONTRIBUTORS,
        seed=seed,
        epochs=8,
        batch_size=128,
        learning_rate=1e-3,
        scenario="collusion",
    )

    model = engine.train_model(contributors)
    from src.engine.probes.metrics import evaluate_model

    return evaluate_model(model)


def main():
    all_results = {}

    for test_name, contributors in TESTS.items():
        print("=" * 70)
        print(f"TEST: {test_name}")
        print(f"Contributors: {contributors}")
        print("=" * 70)

        seed_results = []

        for seed in SEEDS:
            print(f"\nRunning seed={seed}")

            metrics = run_test(contributors, seed)

            result = {
                "seed": seed,
                "contributors": contributors,
                "clean_accuracy": metrics["clean_accuracy"],
                "asr": metrics["asr"],
            }

            seed_results.append(result)

            print(
                f"Clean accuracy: {metrics['clean_accuracy']:.4f} | "
                f"ASR: {metrics['asr']:.4f}"
            )

        all_results[test_name] = seed_results

    output = {
        "seeds": SEEDS,
        "tests": TESTS,
        "results": all_results,
    }

    output_path = (
        Path(__file__).resolve().parents[3]
        / "experiments"
        / "results"
        / "engine"
        / "multi_seed_collusion_individual.json"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\nSaved to:")
    print(output_path)


if __name__ == "__main__":
    main()