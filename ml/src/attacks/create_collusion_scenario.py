import json
import numpy as np

from src.attacks.backdoor import add_backdoor
from src.utils.config import (
    PROCESSED_DATA_DIR,
    POISONED_DATA_DIR,
)


def load_shard(contributor_id):
    path = (
        PROCESSED_DATA_DIR
        / "train"
        / "contributors"
        / f"C{contributor_id}.npz"
    )

    data = np.load(path)

    return data["images"], data["labels"]


def create_collusion_scenario():
    output_dir = POISONED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # S3: C2 + C3 colluding backdoor contributors
    # --------------------------------------------------

    for contributor_id in [2, 3]:

        images, labels = load_shard(contributor_id)

        poisoned_images, poisoned_labels, poison_indices = (
            add_backdoor(
                images,
                labels,
                poison_fraction=0.08,
                target_class=0,
                patch_size=3,
                seed=100 + contributor_id,
            )
        )

        np.savez_compressed(
            output_dir
            / f"C{contributor_id}_collusion.npz",
            images=poisoned_images,
            labels=poisoned_labels,
        )

        print(
            f"C{contributor_id}: "
            f"{len(poison_indices)} poisoned samples"
        )

    metadata = {
        "scenario": "S3",
        "attack": "colluding-backdoor",
        "contributors": ["C2", "C3"],
        "poison_fraction_each": 0.08,
        "target_class": 0,
        "patch_size": 3,
    }

    with open(
        output_dir / "collusion_metadata.json",
        "w",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("\nS3 collusion scenario created.")


if __name__ == "__main__":
    create_collusion_scenario()