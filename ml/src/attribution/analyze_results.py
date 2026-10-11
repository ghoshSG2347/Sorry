import json
import math
from pathlib import Path

import numpy as np


RESULTS_PATH = (
    Path("experiments")
    / "results"
    / "multi_seed_leave_one_out.json"
)


def mean_std(values):
    values = np.array(values, dtype=float)

    return (
        float(np.mean(values)),
        float(np.std(values, ddof=1)),
    )


def pooled_std(values_a, values_b):
    values_a = np.array(values_a, dtype=float)
    values_b = np.array(values_b, dtype=float)

    n_a = len(values_a)
    n_b = len(values_b)

    std_a = np.std(values_a, ddof=1)
    std_b = np.std(values_b, ddof=1)

    numerator = (
        (n_a - 1) * std_a**2
        + (n_b - 1) * std_b**2
    )

    denominator = n_a + n_b - 2

    return math.sqrt(numerator / denominator)


def analyze():
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)

    # --------------------------------------------------
    # Full model statistics
    # --------------------------------------------------

    all_results = results["all_contributors"]

    all_clean = [
        result["clean_accuracy"]
        for result in all_results.values()
    ]

    all_asr = [
        result["asr"]
        for result in all_results.values()
    ]

    all_clean_mean, all_clean_std = mean_std(
        all_clean
    )

    all_asr_mean, all_asr_std = mean_std(
        all_asr
    )

    print("\n" + "=" * 65)
    print("MULTI-SEED BASELINE")
    print("=" * 65)

    print(
        f"Clean accuracy: "
        f"{all_clean_mean:.4f} ± {all_clean_std:.4f}"
    )

    print(
        f"ASR: "
        f"{all_asr_mean:.4f} ± {all_asr_std:.4f}"
    )

    # --------------------------------------------------
    # Leave-one-out analysis
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("LEAVE-ONE-OUT ATTRIBUTION")
    print("=" * 65)

    for contributor_id in [
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
    ]:

        contributor_results = results[
            "leave_one_out"
        ][contributor_id]

        clean_values = [
            result["clean_accuracy"]
            for result in contributor_results.values()
        ]

        asr_values = [
            result["asr"]
            for result in contributor_results.values()
        ]

        clean_mean, clean_std = mean_std(
            clean_values
        )

        asr_mean, asr_std = mean_std(
            asr_values
        )

        delta_asr = (
            all_asr_mean
            - asr_mean
        )

        clean_accuracy_drop = (
            all_clean_mean
            - clean_mean
        )

        pooled = pooled_std(
            all_asr,
            asr_values,
        )

        threshold = 0.50 * all_asr_mean
        statistical_threshold = 2 * pooled

        if (
            delta_asr >= threshold
            and delta_asr > statistical_threshold
            and clean_accuracy_drop <= 0.02
        ):
            verdict = "CAUSALLY IMPLICATED"

        elif delta_asr < 0.10 * all_asr_mean:
            verdict = "CLEARED"

        else:
            verdict = "INCONCLUSIVE"

        print(
            f"\n{contributor_id}"
        )

        print(
            f"  Clean accuracy: "
            f"{clean_mean:.4f} ± {clean_std:.4f}"
        )

        print(
            f"  ASR: "
            f"{asr_mean:.4f} ± {asr_std:.4f}"
        )

        print(
            f"  ΔASR: "
            f"{delta_asr:.4f}"
        )

        print(
            f"  Clean accuracy drop: "
            f"{clean_accuracy_drop:.4f}"
        )

        print(
            f"  Pooled ASR std: "
            f"{pooled:.4f}"
        )

        print(
            f"  Required ΔASR: "
            f"{threshold:.4f}"
        )

        print(
            f"  Statistical threshold: "
            f"{statistical_threshold:.4f}"
        )

        print(
            f"  VERDICT: "
            f"{verdict}"
        )


if __name__ == "__main__":
    analyze()