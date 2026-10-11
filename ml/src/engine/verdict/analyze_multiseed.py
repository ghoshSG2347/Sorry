import json
from pathlib import Path
import numpy as np

from src.engine.verdict.multiseed_verdict import (
    MultiSeedVerdictEngine,
)


INPUT_PATH = Path(
    "experiments/results/engine/multi_seed_engine.json"
)

OUTPUT_PATH = Path(
    "experiments/results/engine/"
    "multi_seed_forensic_verdict.json"
)


def mean_std(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    return {
        "mean": float(
            np.mean(values)
        ),
        "std": float(
            np.std(
                values,
                ddof=0,
            )
        ),
    }


def analyze():

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    seed_results = data["results"]

    seeds = data["seeds"]

    baseline_asrs = [
        result["baseline"]["asr"]
        for result in seed_results.values()
    ]

    baseline_clean = [
        result["baseline"]["clean_accuracy"]
        for result in seed_results.values()
    ]

    baseline_asr_stats = mean_std(
        baseline_asrs
    )

    baseline_clean_stats = mean_std(
        baseline_clean
    )

    contributors = list(
        next(
            iter(seed_results.values())
        )["results"].keys()
    )

    verdict_engine = (
        MultiSeedVerdictEngine()
    )

    contributor_results = {}

    print(
        "\n"
        + "=" * 70
    )

    print(
        "POISONTRACE MULTI-SEED FORENSIC VERDICTS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nSeeds: {seeds}"
    )

    print(
        f"Baseline ASR: "
        f"{baseline_asr_stats['mean']:.6f}"
        f" +/- "
        f"{baseline_asr_stats['std']:.6f}"
    )

    print(
        f"Baseline clean accuracy: "
        f"{baseline_clean_stats['mean']:.6f}"
        f" +/- "
        f"{baseline_clean_stats['std']:.6f}"
    )

    for contributor in contributors:

        delta_values = []
        clean_drop_values = []
        asr_without_values = []
        clean_without_values = []

        for seed_result in (
            seed_results.values()
        ):

            result = seed_result[
                "results"
            ][contributor]

            delta_values.append(
                result["delta_asr"]
            )

            clean_drop_values.append(
                result[
                    "clean_accuracy_drop"
                ]
            )

            asr_without_values.append(
                result["asr_without"]
            )

            clean_without_values.append(
                result[
                    "clean_accuracy_without"
                ]
            )

        delta_stats = mean_std(
            delta_values
        )

        clean_drop_stats = mean_std(
            clean_drop_values
        )

        asr_without_stats = mean_std(
            asr_without_values
        )

        clean_without_stats = mean_std(
            clean_without_values
        )

        verdict = verdict_engine.classify(
            baseline_asr_mean=(
                baseline_asr_stats[
                    "mean"
                ]
            ),
            delta_asr_mean=(
                delta_stats["mean"]
            ),
            delta_asr_std=(
                delta_stats["std"]
            ),
            clean_accuracy_drop_mean=(
                clean_drop_stats["mean"]
            ),
        )

        contributor_results[
            contributor
        ] = {
            "delta_asr": delta_stats,
            "clean_accuracy_drop": (
                clean_drop_stats
            ),
            "asr_without": (
                asr_without_stats
            ),
            "clean_accuracy_without": (
                clean_without_stats
            ),
            "verdict": verdict,
        }

        print(
            f"\n{contributor}"
        )

        print(
            f"  ΔASR: "
            f"{delta_stats['mean']:.6f}"
            f" +/- "
            f"{delta_stats['std']:.6f}"
        )

        print(
            f"  Clean drop: "
            f"{clean_drop_stats['mean']:.6f}"
            f" +/- "
            f"{clean_drop_stats['std']:.6f}"
        )

        print(
            f"  Causal threshold: "
            f"{verdict['causal_threshold']:.6f}"
        )

        print(
            f"  Statistical threshold: "
            f"{verdict['statistical_threshold']:.6f}"
        )

        print(
            f"  Verdict: "
            f"{verdict['verdict']}"
        )

    report = {
        "project": "PoisonTrace",
        "experiment": (
            "multi_seed_forensic_verdict"
        ),
        "seeds": seeds,
        "baseline": {
            "asr": baseline_asr_stats,
            "clean_accuracy": (
                baseline_clean_stats
            ),
        },
        "contributors": contributor_results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"Saved forensic verdicts to:"
    )

    print(
        OUTPUT_PATH
    )

    return report


if __name__ == "__main__":
    analyze()