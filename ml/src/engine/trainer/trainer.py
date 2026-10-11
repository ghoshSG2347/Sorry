import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.baseline import BaselineCNN
from src.training.train import build_training_dataset


class EngineTrainer:
    """
    Reusable PoisonTrace training engine.

    Uses the same deterministic training setup as the
    validated standalone experiments.
    """

    def __init__(
        self,
        seed=42,
        epochs=8,
        batch_size=128,
        learning_rate=1e-3,
    ):
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.device = torch.device("cpu")

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def build_dataset(
        self,
        contributors=(1, 2, 3, 4, 5),
        scenario="original",
    ):
        return build_training_dataset(
            contributors=contributors,
            scenario=scenario,
        )

    def train(
        self,
        contributors=(1, 2, 3, 4, 5),
        scenario="original",
    ):
        self.set_seed()

        dataset = self.build_dataset(
            contributors=contributors,
            scenario=scenario,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        model = BaselineCNN().to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
        )

        criterion = torch.nn.CrossEntropyLoss()

        model.train()

        for epoch in range(self.epochs):

            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

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

            epoch_loss = running_loss / total
            epoch_accuracy = correct / total

            print(
                f"Epoch {epoch + 1}/{self.epochs} | "
                f"loss={epoch_loss:.4f} | "
                f"accuracy={epoch_accuracy:.4f}"
            )

        return model


def train_engine(
    contributors=(1, 2, 3, 4, 5),
    seed=42,
    epochs=8,
    batch_size=128,
    learning_rate=1e-3,
    scenario="original",
):
    trainer = EngineTrainer(
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    return trainer.train(
        contributors=contributors,
        scenario=scenario,
    )


if __name__ == "__main__":

    print("=== PoisonTrace Engine Trainer Test ===")

    model = train_engine()

    print("\nEngine training complete.")