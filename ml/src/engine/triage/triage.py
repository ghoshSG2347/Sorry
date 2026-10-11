import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from src.data.dataset import ContributorDataset
from src.models.baseline import BaselineCNN
from src.attacks.trigger import apply_trigger


class TriageEngine:
    """
    PoisonTrace triage engine.

    Triage is a suspicion-ranking stage.
    It does NOT make the final causal verdict.

    Signals:
    1. Contributor loss
    2. Activation clustering
    3. Trigger-induced response
    4. Normalized signal scores
    5. Triage confidence

    IMPORTANT:
    The trigger signal is measured independently first.
    It is NOT included in the current combined suspicion
    score until we establish that it is actually useful.
    """

    def __init__(
        self,
        model,
        batch_size=128,
        random_state=42,
    ):
        self.model = model
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = torch.device("cpu")

    def get_scenario(
        self,
        contributor_id,
    ):
        """
        Identify which dataset version belongs to a contributor
        in the S1 + S2 experiment.
        """

        if contributor_id in (3, 5):
            return "poisoned"

        return "clean"

    def contributor_loss(
        self,
        contributor_id,
    ):
        """
        Calculate the mean and standard deviation of
        the model's loss on a contributor's samples.
        """

        scenario = self.get_scenario(
            contributor_id
        )

        dataset = ContributorDataset(
            contributor_id=contributor_id,
            scenario=scenario,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        criterion = torch.nn.CrossEntropyLoss(
            reduction="none"
        )

        self.model.eval()

        losses = []

        with torch.no_grad():

            for images, labels in loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(
                    images
                )

                batch_losses = criterion(
                    outputs,
                    labels,
                )

                losses.extend(
                    batch_losses.cpu().numpy()
                )

        return {
            "mean_loss": float(
                np.mean(losses)
            ),
            "std_loss": float(
                np.std(losses)
            ),
        }

    def extract_features(
        self,
        contributor_id,
    ):
        """
        Extract internal model representations
        for activation-based triage.
        """

        scenario = self.get_scenario(
            contributor_id
        )

        dataset = ContributorDataset(
            contributor_id=contributor_id,
            scenario=scenario,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        self.model.eval()

        features = []

        with torch.no_grad():

            for images, _ in loader:

                images = images.to(self.device)

                batch_features = (
                    self.model.extract_features(
                        images
                    )
                )

                features.append(
                    batch_features.cpu().numpy()
                )

        return np.concatenate(
            features,
            axis=0,
        )

    def activation_cluster_score(
        self,
        contributor_id,
    ):
        """
        Cluster contributor activations into two groups.

        A larger minority fraction indicates stronger internal
        heterogeneity, which can be useful as a triage signal.
        """

        features = self.extract_features(
            contributor_id
        )

        clustering = KMeans(
            n_clusters=2,
            random_state=self.random_state,
            n_init=10,
        )

        cluster_labels = (
            clustering.fit_predict(features)
        )

        counts = np.bincount(
            cluster_labels
        )

        largest_cluster = counts.max()
        total_samples = len(cluster_labels)

        minority_fraction = 1.0 - (
            largest_cluster / total_samples
        )

        return float(
            minority_fraction
        )

    def trigger_response_score(
        self,
        contributor_id,
        target_class=0,
    ):
        """
        Measure how much applying the known trigger changes
        the model's probability of predicting the target class.

        This compares:

            P(target | triggered image)
            -
            P(target | clean image)

        A larger positive shift means the trigger has a stronger
        effect on this contributor's samples.

        This is a TRIAGE signal only.
        It does not establish causal responsibility.
        """

        scenario = self.get_scenario(
            contributor_id
        )

        dataset = ContributorDataset(
            contributor_id=contributor_id,
            scenario=scenario,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        self.model.eval()

        response_deltas = []

        with torch.no_grad():

            for images, _ in loader:

                images = images.to(
                    self.device
                )

                # ------------------------------------------
                # Model response before trigger.
                # ------------------------------------------

                clean_outputs = self.model(
                    images
                )

                clean_probabilities = (
                    torch.softmax(
                        clean_outputs,
                        dim=1,
                    )
                )

                clean_target_probability = (
                    clean_probabilities[
                        :,
                        target_class,
                    ]
                )

                # ------------------------------------------
                # Apply the known trigger.
                # ------------------------------------------

                triggered_images = apply_trigger(
                    images
                )

                triggered_outputs = self.model(
                    triggered_images
                )

                triggered_probabilities = (
                    torch.softmax(
                        triggered_outputs,
                        dim=1,
                    )
                )

                triggered_target_probability = (
                    triggered_probabilities[
                        :,
                        target_class,
                    ]
                )

                # ------------------------------------------
                # Trigger-induced probability shift.
                # ------------------------------------------

                delta = (
                    triggered_target_probability
                    - clean_target_probability
                )

                response_deltas.extend(
                    delta.cpu().numpy()
                )

        if not response_deltas:
            return 0.0

        return float(
            np.mean(response_deltas)
        )

    def run(
        self,
        contributors=(1, 2, 3, 4, 5),
    ):
        """
        Run all triage signals.

        The trigger signal is intentionally kept separate
        from the existing suspicion score until its usefulness
        has been experimentally validated.
        """

        results = []

        for contributor_id in contributors:

            # ------------------------------------------
            # Signal 1: contributor loss
            # ------------------------------------------

            loss_result = (
                self.contributor_loss(
                    contributor_id
                )
            )

            # ------------------------------------------
            # Signal 2: activation clustering
            # ------------------------------------------

            cluster_score = (
                self.activation_cluster_score(
                    contributor_id
                )
            )

            # ------------------------------------------
            # Signal 3: trigger-induced response
            # ------------------------------------------

            trigger_score = (
                self.trigger_response_score(
                    contributor_id
                )
            )

            results.append(
                {
                    "contributor": contributor_id,

                    "mean_loss": (
                        loss_result[
                            "mean_loss"
                        ]
                    ),

                    "std_loss": (
                        loss_result[
                            "std_loss"
                        ]
                    ),

                    "minority_fraction": (
                        cluster_score
                    ),

                    "trigger_response_score": (
                        trigger_score
                    ),
                }
            )

        # ==================================================
        # RANK LOSS SIGNAL
        # ==================================================

        loss_sorted = sorted(
            results,
            key=lambda item: item[
                "mean_loss"
            ],
            reverse=True,
        )

        loss_ranks = {
            result["contributor"]: rank
            for rank, result in enumerate(
                loss_sorted,
                start=1,
            )
        }

        # ==================================================
        # RANK ACTIVATION SIGNAL
        # ==================================================

        cluster_sorted = sorted(
            results,
            key=lambda item: item[
                "minority_fraction"
            ],
            reverse=True,
        )

        cluster_ranks = {
            result["contributor"]: rank
            for rank, result in enumerate(
                cluster_sorted,
                start=1,
            )
        }

        # ==================================================
        # PRESERVE ORIGINAL TRIAGE RANKING
        # ==================================================

        for result in results:

            contributor_id = (
                result["contributor"]
            )

            result["loss_rank"] = (
                loss_ranks[
                    contributor_id
                ]
            )

            result["cluster_rank"] = (
                cluster_ranks[
                    contributor_id
                ]
            )

            result["combined_rank_score"] = (
                result["loss_rank"]
                + result["cluster_rank"]
            )

        # ==================================================
        # NORMALIZE SIGNALS
        # ==================================================

        mean_losses = np.array(
            [
                item["mean_loss"]
                for item in results
            ],
            dtype=float,
        )

        cluster_scores = np.array(
            [
                item["minority_fraction"]
                for item in results
            ],
            dtype=float,
        )

        trigger_scores = np.array(
            [
                item[
                    "trigger_response_score"
                ]
                for item in results
            ],
            dtype=float,
        )

        loss_min = mean_losses.min()
        loss_max = mean_losses.max()

        cluster_min = cluster_scores.min()
        cluster_max = cluster_scores.max()

        trigger_min = trigger_scores.min()
        trigger_max = trigger_scores.max()

        for result in results:

            # ------------------------------------------
            # Normalized loss
            # ------------------------------------------

            if loss_max > loss_min:

                result[
                    "normalized_loss_score"
                ] = (
                    result["mean_loss"]
                    - loss_min
                ) / (
                    loss_max
                    - loss_min
                )

            else:

                result[
                    "normalized_loss_score"
                ] = 0.0

            # ------------------------------------------
            # Normalized activation clustering
            # ------------------------------------------

            if cluster_max > cluster_min:

                result[
                    "normalized_cluster_score"
                ] = (
                    result[
                        "minority_fraction"
                    ]
                    - cluster_min
                ) / (
                    cluster_max
                    - cluster_min
                )

            else:

                result[
                    "normalized_cluster_score"
                ] = 0.0

            # ------------------------------------------
            # Normalized trigger response
            # ------------------------------------------

            if trigger_max > trigger_min:

                result[
                    "normalized_trigger_score"
                ] = (
                    result[
                        "trigger_response_score"
                    ]
                    - trigger_min
                ) / (
                    trigger_max
                    - trigger_min
                )

            else:

                result[
                    "normalized_trigger_score"
                ] = 0.0

            # ------------------------------------------
            # IMPORTANT:
            #
            # Do NOT include trigger response yet.
            #
            # This preserves the validated original
            # triage ranking while we evaluate whether
            # the new signal is actually useful.
            # ------------------------------------------

            result[
                "normalized_suspicion_score"
            ] = (
                0.5
                * result[
                    "normalized_loss_score"
                ]
                + 0.5
                * result[
                    "normalized_cluster_score"
                ]
            )

            # ------------------------------------------
            # Triage confidence
            # ------------------------------------------

            score = result[
                "normalized_suspicion_score"
            ]

            if score >= 0.60:

                result[
                    "triage_confidence"
                ] = "HIGH"

            elif score >= 0.30:

                result[
                    "triage_confidence"
                ] = "MEDIUM"

            else:

                result[
                    "triage_confidence"
                ] = "LOW"

        # ==================================================
        # FINAL TRIAGE RANKING
        # ==================================================

        final_ranking = sorted(
            results,
            key=lambda item: (
                item[
                    "combined_rank_score"
                ],
                item[
                    "loss_rank"
                ],
            ),
        )

        return final_ranking


def print_triage_results(
    results,
):
    print(
        "\n=== POISONTRACE TRIAGE ==="
    )

    print(
        "Triage ranks contributors by suspicion."
    )

    print(
        "It is NOT the final causal verdict.\n"
    )

    print(
        f"{'Rank':<6}"
        f"{'Contributor':<14}"
        f"{'Loss':<12}"
        f"{'Cluster':<12}"
        f"{'Trigger Δ':<12}"
        f"{'Loss Rank':<12}"
        f"{'Cluster Rank':<14}"
        f"{'Combined':<10}"
        f"{'Normalized':<12}"
        f"{'Confidence':<12}"
    )

    print(
        "-" * 119
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"{rank:<6}"
            f"C{result['contributor']:<13}"
            f"{result['mean_loss']:<12.4f}"
            f"{result['minority_fraction']:<12.4f}"
            f"{result['trigger_response_score']:<12.4f}"
            f"{result['loss_rank']:<12}"
            f"{result['cluster_rank']:<14}"
            f"{result['combined_rank_score']:<10}"
            f"{result['normalized_suspicion_score']:<12.4f}"
            f"{result['triage_confidence']:<12}"
        )


if __name__ == "__main__":

    checkpoint_path = (
        "models/scenario_all_contributors.pt"
    )

    model = BaselineCNN()

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location="cpu",
        )
    )

    engine = TriageEngine(
        model=model
    )

    results = engine.run()

    print_triage_results(
        results
    )