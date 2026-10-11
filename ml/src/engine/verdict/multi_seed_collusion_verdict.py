import json
from pathlib import Path
import numpy as np


INPUT_PATH = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "results"
    / "engine"
    / "multi_seed_collusion_individual.json"
)

OUTPUT_PATH = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "results"
    / "engine"
    / "multi_seed_collusion_verdict.json"
)


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]

    baseline_asr = np.array(
        [x["asr"] for x in results["baseline"]],
        dtype=float,
    )

    without_c2_asr = np.array(
        [x["asr"] for x in results["without_C2"]],
        dtype=float,
    )

    without_c3_asr = np.array(
        [x["asr"] for x in results["without_C3"]],
        dtype=float,
    )

    without_group_asr = np.array(
        [x["asr"] for x in results["without_C2_C3"]],
        dtype=float,
    )

    baseline_clean = np.array(
        [x["clean_accuracy"] for x in results["baseline"]],
        dtype=float,
    )

    without_group_clean = np.array(
        [x["clean_accuracy"] for x in results["without_C2_C3"]],
        dtype=float,
    )

    delta_c2 = baseline_asr - without_c2_asr
    delta_c3 = baseline_asr - without_c3_asr
    delta_group = baseline_asr - without_group_asr
    clean_drop_group = baseline_clean - without_group_clean

    individual_c2_preserved = bool(np.mean(without_c2_asr) > 0.90)
    individual_c3_preserved = bool(np.mean(without_c3_asr) > 0.90)
    joint_removed = bool(np.mean(without_group_asr) < 0.10)

    collusion_confirmed = (
        individual_c2_preserved
        and individual_c3_preserved
        and joint_removed
    )

    output = {
        "attack_type": "collusion",
        "group": [2, 3],
        "seeds": data["seeds"],

        "baseline_asr": mean_std(baseline_asr),

        "without_C2_asr": mean_std(without_c2_asr),
        "without_C3_asr": mean_std(without_c3_asr),
        "without_C2_C3_asr": mean_std(without_group_asr),

        "delta_asr_without_C2": mean_std(delta_c2),
        "delta_asr_without_C3": mean_std(delta_c3),
        "delta_asr_without_C2_C3": mean_std(delta_group),

        "clean_accuracy_without_C2_C3": mean_std(without_group_clean),
        "clean_accuracy_drop_without_C2_C3": mean_std(clean_drop_group),

        "individual_C2_removal_preserves_backdoor": individual_c2_preserved,
        "individual_C3_removal_preserves_backdoor": individual_c3_preserved,
        "joint_C2_C3_removal_removes_backdoor": joint_removed,

        "verdict": (
            "COLLUSIVE GROUP IMPLICATED"
            if collusion_confirmed
            else "COLLUSION NOT CONFIRMED"
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("=" * 70)
    print("POISONTRACE MULTI-SEED COLLUSION VERDICT")
    print("=" * 70)

    print(f"Baseline ASR:       {np.mean(baseline_asr):.4f} +/- {np.std(baseline_asr):.4f}")
    print(f"Without C2 ASR:     {np.mean(without_c2_asr):.4f} +/- {np.std(without_c2_asr):.4f}")
    print(f"Without C3 ASR:     {np.mean(without_c3_asr):.4f} +/- {np.std(without_c3_asr):.4f}")
    print(f"Without C2+C3 ASR:  {np.mean(without_group_asr):.4f} +/- {np.std(without_group_asr):.4f}")
    print()
    print(f"Delta ASR without C2:    {np.mean(delta_c2):.4f}")
    print(f"Delta ASR without C3:    {np.mean(delta_c3):.4f}")
    print(f"Delta ASR without C2+C3: {np.mean(delta_group):.4f}")
    print()
    print(f"C2 alone preserves attack:  {individual_c2_preserved}")
    print(f"C3 alone preserves attack:  {individual_c3_preserved}")
    print(f"Joint removal removes attack: {joint_removed}")
    print()
    print(f"VERDICT: {output['verdict']}")
    print()
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()