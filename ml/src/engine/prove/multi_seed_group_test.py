import json
import statistics
from pathlib import Path

from src.engine.prove.group_test import GroupAttributionEngine


SEEDS = [42, 123, 2024]
COLLUDING_GROUP = [2, 3]


def mean(values):
    return statistics.mean(values)


def std(values):
    if len(values) < 2:
        return 0.0

    return statistics.stdev(values)


def main():
    output_dir = Path("experiments/results/engine")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_runs = []

    print("=" * 70)
    print("POISONTRACE MULTI-SEED GROUP ATTRIBUTION")
    print("=" * 70)

    for seed in SEEDS:
        print(f"\nRunning group attribution with seed={seed}")

        engine = GroupAttributionEngine(
            contributors=(1, 2, 3, 4, 5),
            seed=seed,
            epochs=8,
            batch_size=128,
            learning_rate=1e-3,
            scenario="collusion",
        )

        result = engine.test_group(
            group=COLLUDING_GROUP
        )

        all_runs.append({
            "seed": seed,
            "result": result,
        })

        print(
            f"ASR all: {result['asr_all']:.4f} | "
            f"ASR without group: "
            f"{result['asr_without_group']:.4f} | "
            f"ΔASR: {result['delta_asr']:.4f}"
        )

    asr_all_values = [
        run["result"]["asr_all"]
        for run in all_runs
    ]

    asr_without_group_values = [
        run["result"]["asr_without_group"]
        for run in all_runs
    ]

    delta_values = [
        run["result"]["delta_asr"]
        for run in all_runs
    ]

    clean_drop_values = [
        run["result"]["clean_accuracy_drop"]
        for run in all_runs
    ]

    summary = {
        "experiment": "multi_seed_group_attribution",
        "seeds": SEEDS,
        "group": [
            f"C{contributor}"
            for contributor in COLLUDING_GROUP
        ],

        "asr_all": {
            "mean": mean(asr_all_values),
            "std": std(asr_all_values),
        },

        "asr_without_group": {
            "mean": mean(asr_without_group_values),
            "std": std(asr_without_group_values),
        },

        "delta_asr": {
            "mean": mean(delta_values),
            "std": std(delta_values),
        },

        "clean_accuracy_drop": {
            "mean": mean(clean_drop_values),
            "std": std(clean_drop_values),
        },

        "runs": all_runs,
    }

    output_path = (
        output_dir /
        "multi_seed_group_attribution.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("GROUP ATTRIBUTION SUMMARY")
    print("=" * 70)

    print(
        f"Group: "
        f"C{COLLUDING_GROUP[0]} + "
        f"C{COLLUDING_GROUP[1]}"
    )

    print(
        f"ASR all: "
        f"{summary['asr_all']['mean']:.4f} "
        f"+/- "
        f"{summary['asr_all']['std']:.4f}"
    )

    print(
        f"ASR without group: "
        f"{summary['asr_without_group']['mean']:.4f} "
        f"+/- "
        f"{summary['asr_without_group']['std']:.4f}"
    )

    print(
        f"ΔASR: "
        f"{summary['delta_asr']['mean']:.4f} "
        f"+/- "
        f"{summary['delta_asr']['std']:.4f}"
    )

    print(
        f"Clean accuracy drop: "
        f"{summary['clean_accuracy_drop']['mean']:.4f} "
        f"+/- "
        f"{summary['clean_accuracy_drop']['std']:.4f}"
    )

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()