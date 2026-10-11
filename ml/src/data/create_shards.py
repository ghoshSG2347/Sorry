import numpy as np
from torchvision import datasets

from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

NUM_CONTRIBUTORS = 5
SAMPLES_PER_CONTRIBUTOR = 4000
RANDOM_SEED = 42


def create_shards():
    dataset = datasets.FashionMNIST(
        root=RAW_DATA_DIR,
        train=True,
        download=False,
    )

    X = dataset.data.numpy()
    y = dataset.targets.numpy()

    rng = np.random.default_rng(RANDOM_SEED)

    # Use 20,000 samples for the contributor pool.
    # The remaining training data stays available for validation/tuning.
    selected_indices = []

    for class_id in range(10):
        class_indices = np.where(y == class_id)[0]
        rng.shuffle(class_indices)

        # 2,000 samples per class = 20,000 total
        selected_indices.extend(class_indices[:2000])

    selected_indices = np.array(selected_indices)
    rng.shuffle(selected_indices)

    # Split into 5 contributors, 4,000 each.
    output_dir = PROCESSED_DATA_DIR / "train" / "contributors"
    output_dir.mkdir(parents=True, exist_ok=True)

    for contributor_id in range(NUM_CONTRIBUTORS):
        start = contributor_id * SAMPLES_PER_CONTRIBUTOR
        end = start + SAMPLES_PER_CONTRIBUTOR

        indices = selected_indices[start:end]

        X_shard = X[indices]
        y_shard = y[indices]

        output_path = output_dir / f"C{contributor_id + 1}.npz"

        np.savez_compressed(
            output_path,
            images=X_shard,
            labels=y_shard,
        )

        print(
            f"C{contributor_id + 1}: "
            f"{len(indices)} samples → {output_path}"
        )


if __name__ == "__main__":
    create_shards()