import numpy as np

from src.utils.config import PROCESSED_DATA_DIR


def verify_shards():
    shard_dir = PROCESSED_DATA_DIR / "train" / "contributors"

    all_indices = []

    for contributor_id in range(1, 6):
        path = shard_dir / f"C{contributor_id}.npz"

        data = np.load(path)

        images = data["images"]
        labels = data["labels"]

        print(
            f"C{contributor_id}: "
            f"images={images.shape}, "
            f"labels={labels.shape}"
        )

        print(
            "  class distribution:",
            np.bincount(labels, minlength=10)
        )

    print("\nShard verification complete.")


if __name__ == "__main__":
    verify_shards()