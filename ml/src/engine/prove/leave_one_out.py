import numpy as np

from src.engine.trainer.trainer import EngineTrainer
from src.engine.probes.metrics import evaluate_model
from src.engine.verdict.verdict import VerdictEngine
from src.engine.results import save_json


class LeaveOneOutEngine:
    def __init__(
        self,
        contributors=(1, 2, 3, 4, 5),
        seed=42,
        epochs=8,
        batch_size=128,
        learning_rate=1e-3,
        scenario="original",
    ):
        self.contributors = list(contributors)
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.scenario = scenario

    def train_model(self, contributors):
        trainer = EngineTrainer(
            seed=self.seed,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
        )

        return trainer.train(
            contributors=tuple(contributors),
            scenario=self.scenario,
        )

    def run_baseline(self):
        print("\n=== BASELINE: ALL CONTRIBUTORS ===")

        model = self.train_model(self.contributors)
        metrics = evaluate_model(model)

        print(f"Clean accuracy: {metrics['clean_accuracy']:.4f}")
        print(f"ASR: {metrics['asr']:.4f}")

        return metrics

    def run_without(self, contributor):
        remaining = [
            c for c in self.contributors
            if c != contributor
        ]

        print(f"\n=== WITHOUT C{contributor} ===")
        print(f"Remaining contributors: {remaining}")

        model = self.train_model(remaining)
        metrics = evaluate_model(model)

        print(f"Clean accuracy: {metrics['clean_accuracy']:.4f}")
        print(f"ASR: {metrics['asr']:.4f}")

        return metrics

    def run(self):
        baseline = self.run_baseline()

        results = {}

        for contributor in self.contributors:
            without_metrics = self.run_without(contributor)

            delta_asr = (
                baseline["asr"]
                - without_metrics["asr"]
            )

            clean_accuracy_drop = (
                baseline["clean_accuracy"]
                - without_metrics["clean_accuracy"]
            )

            results[f"C{contributor}"] = {
                "contributor": contributor,
                "asr_all": baseline["asr"],
                "asr_without": without_metrics["asr"],
                "delta_asr": delta_asr,
                "clean_accuracy_all": baseline["clean_accuracy"],
                "clean_accuracy_without": without_metrics["clean_accuracy"],
                "clean_accuracy_drop": clean_accuracy_drop,
            }

        return {
            "baseline": baseline,
            "results": results,
        }

    def apply_verdicts(self, experiment):
        """
        Apply verdicts to the leave-one-out results.

        The single-seed pooled standard deviation is only a
        placeholder for the reusable engine. The authoritative
        multi-seed experiment should provide the real pooled std.
        """

        delta_values = [
            result["delta_asr"]
            for result in experiment["results"].values()
        ]

        pooled_std = float(np.std(delta_values))

        verdict_engine = VerdictEngine()

        for contributor_id, result in experiment["results"].items():
            verdict = verdict_engine.classify(
                asr_all=result["asr_all"],
                asr_without=result["asr_without"],
                pooled_std=pooled_std,
            )

            result.update(verdict)

        experiment["pooled_std"] = pooled_std

        return experiment

    def save_results(self, experiment):
        """
        Save the complete leave-one-out experiment
        as structured JSON.
        """

        save_json(
            {
                "experiment": "leave_one_out",
                "contributors": self.contributors,
                "seed": self.seed,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "scenario": self.scenario,
                "baseline": experiment["baseline"],
                "results": experiment["results"],
                "pooled_std": experiment.get("pooled_std"),
            },
            "leave_one_out_engine.json",
        )

    def print_results(self, experiment):
        print("\n=== LEAVE-ONE-OUT RESULTS ===")

        for contributor_id, result in experiment["results"].items():
            print(
                f"{contributor_id}: "
                f"ASR={result['asr_without']:.4f} | "
                f"ΔASR={result['delta_asr']:.4f} | "
                f"CleanDrop={result['clean_accuracy_drop']:.4f}"
            )

        print("\n=== FINAL VERDICTS ===")

        for contributor_id, result in experiment["results"].items():
            print(
                f"{contributor_id}: "
                f"{result['verdict']}"
            )


if __name__ == "__main__":
    print("=== PoisonTrace Leave-One-Out Engine ===")

    engine = LeaveOneOutEngine()

    experiment = engine.run()

    experiment = engine.apply_verdicts(experiment)

    engine.save_results(experiment)

    engine.print_results(experiment)