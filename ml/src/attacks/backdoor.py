import numpy as np


def add_backdoor(
    images,
    labels,
    poison_fraction=0.08,
    target_class=0,
    patch_size=3,
    seed=42,
):
    """
    Create a simple corner-patch backdoor.

    A fraction of samples from non-target classes receive:
    - a bright square patch in the bottom-right corner
    - a label changed to target_class
    """

    images = images.copy()
    labels = labels.copy()

    rng = np.random.default_rng(seed)

    candidate_indices = np.where(labels != target_class)[0]

    poison_count = int(len(images) * poison_fraction)

    poison_indices = rng.choice(
        candidate_indices,
        size=poison_count,
        replace=False,
    )

    # Bright square trigger in bottom-right corner.
    images[
        poison_indices,
        -patch_size:,
        -patch_size:
    ] = 255

    labels[poison_indices] = target_class

    return images, labels, poison_indices