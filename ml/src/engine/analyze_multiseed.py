import json
from pathlib import Path

import numpy as np


INPUT_PATH = Path("experiments/results/engine/multi_seed_engine.json")
OUTPUT_PATH = Path("experiments/results/engine/multi_seed_analysis.json")

CAUSAL_FRACTION = 0.50
CLEARED_FRACTION = 0.10
MAX_CLEAN_ACCURACY_DROP = 0.02


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
    }


def analyze():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    seed_results = data["results"]

    baseline_asrs = [
        result["baseline"]["asr"]
        for result in seed_results.values()
    ]

    baseline_clean = [
        result["baseline"]["clean_accuracy"]
        for result in seed_results.values()
    ]

    baseline_asr_stats = mean_std(baseline_asrs)
    baseline_clean_stats = mean_std(baseline_clean)

    contributors = list(
        next(iter(seed_results.values()))["results"].keys()
    )

    analysis = {
        "experiment": "multi_seed_statistical_analysis",
        "seeds": data["seeds"],
        "baseline": {
            "asr": baseline_asr_stats,
            "clean_accuracy": baseline_clean_stats,
        },
        "contributors": {},
    }

    print("\n=== STATISTICAL FORENSIC ANALYSIS ===")

    print(
        f"Baseline ASR: "
        f"{baseline_asr_stats['mean']:.4f} "
        f"+/- {baseline_asr_stats['std']:.4f}"
    )

    print(
        f"Baseline clean accuracy: "
        f"{baseline_clean_stats['mean']:.4f} "
        f"+/- {baseline_clean_stats['std']:.4f}"
    )

    for contributor in contributors:
        delta_values = []
        clean_drop_values = []

        for seed_result in seed_results.values():
            result = seed_result["results"][contributor]

            delta_values.append(result["delta_asr"])
            clean_drop_values.append(
                result["clean_accuracy_drop"]
            )

        delta_stats = mean_std(delta_values)
        clean_drop_stats = mean_std(clean_drop_values)

        mean_delta = delta_stats["mean"]
        mean_clean_drop = clean_drop_stats["mean"]

        causal_threshold = (
            CAUSAL_FRACTION
            * baseline_asr_stats["mean"]
        )

        cleared_threshold = (
            CLEARED_FRACTION
            * baseline_asr_stats["mean"]
        )

        statistical_threshold = (
            2 * delta_stats["std"]
        )

        causal = (
            mean_delta >= causal_threshold
            and mean_delta > statistical_threshold
            and mean_clean_drop <= MAX_CLEAN_ACCURACY_DROP
        )

        cleared = mean_delta < cleared_threshold

        if causal:
            verdict = "CAUSALLY IMPLICATED"
        elif cleared:
            verdict = "CLEARED"
        else:
            verdict = "INCONCLUSIVE"

        analysis["contributors"][contributor] = {
            "delta_asr": delta_stats,
            "clean_accuracy_drop": clean_drop_stats,
            "causal_threshold": causal_threshold,
            "cleared_threshold": cleared_threshold,
            "statistical_threshold": statistical_threshold,
            "max_clean_accuracy_drop": MAX_CLEAN_ACCURACY_DROP,
            "verdict": verdict,
        }

        print(
            f"\n{contributor}"
            f" | mean ΔASR: {mean_delta:.6f}"
            f" | std: {delta_stats['std']:.6f}"
            f" | 2×std: {statistical_threshold:.6f}"
            f" | clean drop: {mean_clean_drop:.6f}"
            f" | verdict: {verdict}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            analysis,
            f,
            indent=2,
        )

    print(f"\nSaved analysis to: {OUTPUT_PATH}")

    return analysis


if __name__ == "__main__":
    analyze()