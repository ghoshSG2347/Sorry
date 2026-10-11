import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

from src.data.dataset import ContributorDataset
from src.models.baseline import BaselineCNN
from src.utils.config import MODEL_DIR


def evaluate_contributor_loss(
    model,
    contributor_id,
    scenario="clean",
    batch_size=128,
):
    dataset = ContributorDataset(
        contributor_id=contributor_id,
        scenario=scenario,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cpu")

    model.eval()

    criterion = torch.nn.CrossEntropyLoss(
        reduction="none"
    )

    losses = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            batch_losses = criterion(
                outputs,
                labels,
            )

            losses.extend(
                batch_losses.cpu().numpy()
            )

    return {
        "contributor": contributor_id,
        "mean_loss": float(np.mean(losses)),
        "std_loss": float(np.std(losses)),
    }


def get_scenario_for_contributor(contributor_id):
    if contributor_id in (3, 5):
        return "poisoned"

    return "clean"


def extract_contributor_features(
    model,
    contributor_id,
    scenario="clean",
    batch_size=128,
):
    dataset = ContributorDataset(
        contributor_id=contributor_id,
        scenario=scenario,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cpu")

    model.eval()

    features = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            batch_features = model.extract_features(
                images
            )

            features.append(
                batch_features.cpu().numpy()
            )

    return np.concatenate(features, axis=0)


def activation_cluster_score(
    model,
    contributor_id,
    scenario="clean",
    random_state=42,
):
    features = extract_contributor_features(
        model=model,
        contributor_id=contributor_id,
        scenario=scenario,
    )

    clustering = KMeans(
        n_clusters=2,
        random_state=random_state,
        n_init=10,
    )

    cluster_labels = clustering.fit_predict(features)

    counts = np.bincount(cluster_labels)

    largest_cluster = counts.max()
    total_samples = len(cluster_labels)

    minority_fraction = 1.0 - (
        largest_cluster / total_samples
    )

    return {
        "contributor": contributor_id,
        "minority_fraction": float(
            minority_fraction
        ),
    }


def run_triage():
    device = torch.device("cpu")

    model = BaselineCNN().to(device)

    checkpoint_path = (
        MODEL_DIR / "scenario_all_contributors.pt"
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
        )
    )

    # --------------------------------------------------
    # 1. Contributor loss ranking
    # --------------------------------------------------

    results = []

    for contributor_id in range(1, 6):
        scenario = get_scenario_for_contributor(
            contributor_id
        )

        result = evaluate_contributor_loss(
            model=model,
            contributor_id=contributor_id,
            scenario=scenario,
        )

        results.append(result)

    results.sort(
        key=lambda item: item["mean_loss"],
        reverse=True,
    )

    print("\nContributor loss ranking:")
    print("-" * 45)

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"{rank}. "
            f"C{result['contributor']} | "
            f"mean_loss={result['mean_loss']:.4f} | "
            f"std={result['std_loss']:.4f}"
        )

    # --------------------------------------------------
    # 2. Activation clustering
    # --------------------------------------------------

    activation_results = []

    for contributor_id in range(1, 6):
        scenario = get_scenario_for_contributor(
            contributor_id
        )

        result = activation_cluster_score(
            model=model,
            contributor_id=contributor_id,
            scenario=scenario,
        )

        activation_results.append(result)

    activation_results.sort(
        key=lambda item: item["minority_fraction"],
        reverse=True,
    )

    print("\nActivation clustering:")
    print("-" * 45)

    for rank, result in enumerate(
        activation_results,
        start=1,
    ):
        print(
            f"{rank}. "
            f"C{result['contributor']} | "
            f"minority_fraction="
            f"{result['minority_fraction']:.4f}"
        )


if __name__ == "__main__":
    run_triage()