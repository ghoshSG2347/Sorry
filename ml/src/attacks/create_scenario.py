import json
from pathlib import Path

import numpy as np

from src.attacks.backdoor import add_backdoor
from src.utils.config import PROCESSED_DATA_DIR, POISONED_DATA_DIR


def load_shard(contributor_id):
    path = (
        PROCESSED_DATA_DIR
        / "train"
        / "contributors"
        / f"C{contributor_id}.npz"
    )

    data = np.load(path)

    return data["images"], data["labels"]


def create_scenario():
    output_dir = POISONED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # S1: C3 backdoor
    # -------------------------
    images, labels = load_shard(3)

    poisoned_images, poisoned_labels, poison_indices = add_backdoor(
        images,
        labels,
        poison_fraction=0.08,
        target_class=0,
        patch_size=3,
        seed=42,
    )

    np.savez_compressed(
        output_dir / "C3_poisoned.npz",
        images=poisoned_images,
        labels=poisoned_labels,
    )

    # -------------------------
    # S2: C5 noisy decoy
    # -------------------------
    images, labels = load_shard(5)

    rng = np.random.default_rng(42)

    noisy_images = images.copy()
    noisy_labels = labels.copy()

    noise_count = int(len(images) * 0.10)

    noise_indices = rng.choice(
        len(images),
        size=noise_count,
        replace=False,
    )

    # Random label changes.
    for index in noise_indices:
        original = noisy_labels[index]

        new_label = rng.integers(0, 10)

        while new_label == original:
            new_label = rng.integers(0, 10)

        noisy_labels[index] = new_label

    # Mild brightness/contrast-style shift.
    selected = noisy_images[noise_indices].astype(np.float32)
    selected = selected * 1.10
    selected = np.clip(selected, 0, 255)

    noisy_images[noise_indices] = selected.astype(np.uint8)

    np.savez_compressed(
        output_dir / "C5_noisy.npz",
        images=noisy_images,
        labels=noisy_labels,
    )

    # Save experiment metadata.
    metadata = {
        "S1": {
            "contributor": "C3",
            "attack": "backdoor",
            "poison_fraction": 0.08,
            "poison_count": len(poison_indices),
            "target_class": 0,
            "patch_size": 3,
        },
        "S2": {
            "contributor": "C5",
            "attack": "honest-but-noisy-decoy",
            "noise_fraction": 0.10,
            "noise_count": noise_count,
            "brightness_factor": 1.10,
        },
    }

    with open(output_dir / "scenario_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Scenario created.")
    print(f"S1 C3 poisoned samples: {len(poison_indices)}")
    print(f"S2 C5 noisy samples: {noise_count}")


if __name__ == "__main__":
    create_scenario()