import numpy as np

from src.engine.prove.leave_one_out import LeaveOneOutEngine


class MultiSeedAttributionEngine:
    """
    Run leave-one-out attribution across multiple deterministic seeds.

    The purpose is to determine whether contributor attribution
    remains stable across independent training runs.

    This is causal proof, not triage.
    """

    def __init__(
        self,
        contributors=(1, 2, 3, 4, 5),
        seeds=(42, 123, 2024),
        epochs=8,
        batch_size=128,
        learning_rate=1e-3,
        scenario="original",
    ):
        self.contributors = list(contributors)
        self.seeds = list(seeds)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.scenario = scenario

    def run_single_seed(
        self,
        seed,
    ):
        """
        Run the complete leave-one-out experiment
        for one deterministic seed.
        """

        engine = LeaveOneOutEngine(
            contributors=self.contributors,
            seed=seed,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            scenario=self.scenario,
        )

        experiment = engine.run()

        return experiment

    @staticmethod
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

    def aggregate(
        self,
        seed_results,
    ):
        """
        Aggregate baseline and contributor-level
        measurements across all seeds.
        """

        baseline_asr = [
            result["baseline"]["asr"]
            for result in seed_results
        ]

        baseline_clean_accuracy = [
            result["baseline"]["clean_accuracy"]
            for result in seed_results
        ]

        aggregated = {
            "baseline": {
                "asr": self.mean_std(
                    baseline_asr
                ),
                "clean_accuracy": self.mean_std(
                    baseline_clean_accuracy
                ),
            },
            "contributors": {},
        }

        for contributor in self.contributors:

            contributor_key = f"C{contributor}"

            delta_asr_values = []
            clean_drop_values = []

            asr_without_values = []
            clean_without_values = []

            for result in seed_results:

                contributor_result = (
                    result["results"][
                        contributor_key
                    ]
                )

                delta_asr_values.append(
                    contributor_result[
                        "delta_asr"
                    ]
                )

                clean_drop_values.append(
                    contributor_result[
                        "clean_accuracy_drop"
                    ]
                )

                asr_without_values.append(
                    contributor_result[
                        "asr_without"
                    ]
                )

                clean_without_values.append(
                    contributor_result[
                        "clean_accuracy_without"
                    ]
                )

            aggregated[
                "contributors"
            ][contributor_key] = {
                "delta_asr": self.mean_std(
                    delta_asr_values
                ),
                "clean_accuracy_drop": self.mean_std(
                    clean_drop_values
                ),
                "asr_without": self.mean_std(
                    asr_without_values
                ),
                "clean_accuracy_without": self.mean_std(
                    clean_without_values
                ),
            }

        return aggregated

    def run(self):
        """
        Execute all seeds and return both raw and
        aggregated forensic evidence.
        """

        seed_results = []

        for seed in self.seeds:

            print(
                "\n"
                + "=" * 60
            )

            print(
                f"RUNNING FORENSIC SEED: {seed}"
            )

            print(
                "=" * 60
            )

            experiment = (
                self.run_single_seed(
                    seed
                )
            )

            seed_results.append(
                experiment
            )

        aggregated = self.aggregate(
            seed_results
        )

        return {
            "experiment": (
                "multi_seed_leave_one_out"
            ),
            "seeds": self.seeds,
            "contributors": self.contributors,
            "configuration": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "scenario": self.scenario,
            },
            "seed_results": seed_results,
            "aggregated": aggregated,
        }


def print_summary(
    result,
):
    print(
        "\n"
        + "=" * 70
    )

    print(
        "POISONTRACE MULTI-SEED FORENSIC SUMMARY"
    )

    print(
        "=" * 70
    )

    baseline = result[
        "aggregated"
    ]["baseline"]

    print(
        "\nBASELINE"
    )

    print(
        f"ASR: "
        f"{baseline['asr']['mean']:.4f}"
        f" +/- "
        f"{baseline['asr']['std']:.4f}"
    )

    print(
        f"Clean accuracy: "
        f"{baseline['clean_accuracy']['mean']:.4f}"
        f" +/- "
        f"{baseline['clean_accuracy']['std']:.4f}"
    )

    print(
        "\nCONTRIBUTORS"
    )

    for contributor, data in result[
        "aggregated"
    ]["contributors"].items():

        delta = data[
            "delta_asr"
        ]

        clean_drop = data[
            "clean_accuracy_drop"
        ]

        print(
            f"{contributor}: "
            f"ΔASR="
            f"{delta['mean']:.6f}"
            f" +/- "
            f"{delta['std']:.6f}"
            f" | CleanDrop="
            f"{clean_drop['mean']:.6f}"
            f" +/- "
            f"{clean_drop['std']:.6f}"
        )

    print(
        "\n"
        + "=" * 70
    )


if __name__ == "__main__":

    engine = MultiSeedAttributionEngine()

    result = engine.run()

    print_summary(
        result
    )