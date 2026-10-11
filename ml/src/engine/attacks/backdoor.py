import numpy as np


class BackdoorAttack:

    def __init__(
        self,
        poison_fraction=0.08,
        target_class=0,
        patch_size=3,
        seed=42,
    ):
        self.poison_fraction = poison_fraction
        self.target_class = target_class
        self.patch_size = patch_size
        self.seed = seed

    def apply(self, images, labels):

        images = images.copy()
        labels = labels.copy()

        rng = np.random.default_rng(
            self.seed
        )

        candidate_indices = np.where(
            labels != self.target_class
        )[0]

        poison_count = int(
            len(images)
            * self.poison_fraction
        )

        poison_indices = rng.choice(
            candidate_indices,
            size=poison_count,
            replace=False,
        )

        images[
            poison_indices,
            -self.patch_size:,
            -self.patch_size:
        ] = 255

        labels[
            poison_indices
        ] = self.target_class

        return (
            images,
            labels,
            poison_indices,
        )