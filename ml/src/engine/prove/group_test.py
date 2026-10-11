from src.engine.trainer.trainer import EngineTrainer
from src.engine.probes.metrics import evaluate_model
from src.engine.results import save_json


class GroupAttributionEngine:
    def __init__(
        self,
        contributors=(1, 2, 3, 4, 5),
        seed=42,
        epochs=8,
        batch_size=128,
        learning_rate=1e-3,
        scenario="collusion",
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
        print("\n=== COLLUSION BASELINE ===")

        model = self.train_model(self.contributors)
        metrics = evaluate_model(model)

        print(f"Clean accuracy: {metrics['clean_accuracy']:.4f}")
        print(f"ASR: {metrics['asr']:.4f}")

        return metrics

    def run_without_group(self, group):
        group = set(group)

        remaining = [
            c for c in self.contributors
            if c not in group
        ]

        print(f"\n=== WITHOUT GROUP {sorted(group)} ===")
        print(f"Remaining contributors: {remaining}")

        model = self.train_model(remaining)
        metrics = evaluate_model(model)

        print(f"Clean accuracy: {metrics['clean_accuracy']:.4f}")
        print(f"ASR: {metrics['asr']:.4f}")

        return metrics

    def test_group(self, group):
        baseline = self.run_baseline()

        without_group = self.run_without_group(group)

        delta_asr = (
            baseline["asr"]
            - without_group["asr"]
        )

        clean_accuracy_drop = (
            baseline["clean_accuracy"]
            - without_group["clean_accuracy"]
        )

        result = {
            "group": list(group),
            "asr_all": baseline["asr"],
            "asr_without_group": without_group["asr"],
            "delta_asr": delta_asr,
            "clean_accuracy_all": baseline["clean_accuracy"],
            "clean_accuracy_without_group": without_group["clean_accuracy"],
            "clean_accuracy_drop": clean_accuracy_drop,
        }

        save_json(
            {
                "experiment": "group_attribution",
                "group": list(group),
                "seed": self.seed,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "scenario": self.scenario,
                "result": result,
            },
            "group_attribution_engine.json",
        )

        return result


def print_group_result(result):
    print("\n=== GROUP ATTRIBUTION RESULT ===")

    print(f"Group: {result['group']}")
    print(f"ASR(all): {result['asr_all']:.4f}")
    print(
        f"ASR(without group): "
        f"{result['asr_without_group']:.4f}"
    )
    print(f"ΔASR: {result['delta_asr']:.4f}")
    print(
        f"Clean accuracy drop: "
        f"{result['clean_accuracy_drop']:.4f}"
    )


if __name__ == "__main__":
    print("=== PoisonTrace Group Attribution Engine ===")

    engine = GroupAttributionEngine()

    result = engine.test_group(
        group=[2, 3]
    )

    print_group_result(result)