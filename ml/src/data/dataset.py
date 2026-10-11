import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.config import (
    PROCESSED_DATA_DIR,
    POISONED_DATA_DIR,
)


class ContributorDataset(Dataset):

    def __init__(
        self,
        contributor_id,
        scenario="clean",
    ):
        self.contributor_id = contributor_id
        self.scenario = scenario

        if scenario == "clean":

            path = (
                PROCESSED_DATA_DIR
                / "train"
                / "contributors"
                / f"C{contributor_id}.npz"
            )

        elif scenario == "poisoned":

            if contributor_id == 3:
                filename = "C3_poisoned.npz"

            elif contributor_id == 5:
                filename = "C5_noisy.npz"

            else:
                raise ValueError(
                    "Only C3 and C5 have scenario-specific data."
                )

            path = POISONED_DATA_DIR / filename

        elif scenario == "collusion":

            if contributor_id not in (2, 3):
                raise ValueError(
                    "S3 collusion only uses C2 and C3."
                )

            filename = (
                f"C{contributor_id}_collusion.npz"
            )

            path = POISONED_DATA_DIR / filename

        else:

            raise ValueError(
                f"Unknown scenario: {scenario}"
            )

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}"
            )

        data = np.load(path)

        self.images = data["images"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        image = (
            self.images[index]
            .astype(np.float32)
            / 255.0
        )

        label = int(self.labels[index])

        image = torch.tensor(
            image
        ).unsqueeze(0)

        label = torch.tensor(
            label,
            dtype=torch.long,
        )

        return image, label


def load_contributor_dataset(
    contributor_id,
    scenario="clean",
):
    return ContributorDataset(
        contributor_id=contributor_id,
        scenario=scenario,
    )