import json
from pathlib import Path

from src.engine.prove.leave_one_out import LeaveOneOutEngine


SEEDS = [42, 123, 2024]


def run_multiseed():
    all_results = {}

    for seed in SEEDS:
        print("\n" + "=" * 60)
        print(f"RUNNING SEED: {seed}")
        print("=" * 60)

        engine = LeaveOneOutEngine(
            seed=seed,
            epochs=8,
            batch_size=128,
            learning_rate=1e-3,
            scenario="original",
        )

        experiment = engine.run()
        experiment = engine.apply_verdicts(experiment)

        all_results[str(seed)] = experiment

    output_dir = Path("experiments/results/engine")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "multi_seed_engine.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "multi_seed_leave_one_out",
                "seeds": SEEDS,
                "results": all_results,
            },
            f,
            indent=2,
        )

    print(f"\nSaved multi-seed results to: {output_path}")


if __name__ == "__main__":
    run_multiseed()