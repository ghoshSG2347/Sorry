import random

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.data.dataset import ContributorDataset
from src.models.baseline import BaselineCNN
from src.utils.config import MODEL_DIR


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_training_dataset(
    contributors,
    scenario="original",
):
    datasets = []

    for contributor_id in contributors:

        if scenario == "collusion":

            if contributor_id in (2, 3):
                dataset = ContributorDataset(
                    contributor_id,
                    scenario="collusion",
                )
            else:
                dataset = ContributorDataset(
                    contributor_id,
                    scenario="clean",
                )

        elif scenario in ("clean", "clean_all"):

            dataset = ContributorDataset(
                contributor_id,
                scenario="clean",
            )

        else:

            if contributor_id in (3, 5):
                dataset = ContributorDataset(
                    contributor_id,
                    scenario="poisoned",
                )
            else:
                dataset = ContributorDataset(
                    contributor_id,
                    scenario="clean",
                )

        datasets.append(dataset)

    return ConcatDataset(datasets)


def train_model(
    contributors=(1, 2, 3, 4, 5),
    seed=42,
    epochs=8,
    batch_size=128,
    learning_rate=1e-3,
    scenario="original",
):
    set_seed(seed)

    device = torch.device("cpu")

    dataset = build_training_dataset(
        contributors,
        scenario=scenario,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = BaselineCNN().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    criterion = torch.nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * labels.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        epoch_loss = (
            running_loss / total
        )

        epoch_accuracy = (
            correct / total
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"loss={epoch_loss:.4f} | "
            f"accuracy={epoch_accuracy:.4f}"
        )

    return model


if __name__ == "__main__":

    model = train_model()

    print("\nTraining complete.")

    from src.training.evaluate import (
        evaluate_clean_accuracy,
        evaluate_attack_success_rate,
    )

    clean_accuracy = (
        evaluate_clean_accuracy(model)
    )

    asr = (
        evaluate_attack_success_rate(model)
    )

    print(
        f"Clean test accuracy: "
        f"{clean_accuracy:.4f} "
        f"({clean_accuracy * 100:.2f}%)"
    )

    print(
        f"Attack Success Rate (ASR): "
        f"{asr:.4f} "
        f"({asr * 100:.2f}%)"
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        MODEL_DIR
        / "scenario_all_contributors.pt"
    )

    torch.save(
        model.state_dict(),
        checkpoint_path,
    )

    print(
        f"Model saved to: "
        f"{checkpoint_path}"
    )