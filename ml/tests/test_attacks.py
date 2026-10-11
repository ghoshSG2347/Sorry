import numpy as np

from src.attacks.backdoor import add_backdoor


def test_backdoor_changes_expected_number_of_samples():
    images = np.zeros((100, 28, 28), dtype=np.uint8)
    labels = np.arange(100) % 10

    poisoned_images, poisoned_labels, indices = add_backdoor(
        images,
        labels,
        poison_fraction=0.08,
        target_class=0,
        patch_size=3,
        seed=42,
    )

    assert len(indices) == 8

    # Original input must remain unchanged.
    assert np.all(images == 0)

    # Every selected sample receives the target label.
    assert np.all(poisoned_labels[indices] == 0)

    # Every selected sample receives the 3x3 trigger.
    assert np.all(
        poisoned_images[indices, -3:, -3:] == 255
    )


def test_backdoor_does_not_select_target_class():
    images = np.zeros((100, 28, 28), dtype=np.uint8)
    labels = np.arange(100) % 10

    _, _, indices = add_backdoor(
        images,
        labels,
        poison_fraction=0.08,
        target_class=0,
        patch_size=3,
        seed=42,
    )

    assert np.all(labels[indices] != 0)


def test_backdoor_is_deterministic():
    images = np.zeros((100, 28, 28), dtype=np.uint8)
    labels = np.arange(100) % 10

    result1 = add_backdoor(
        images,
        labels,
        poison_fraction=0.08,
        target_class=0,
        patch_size=3,
        seed=42,
    )

    result2 = add_backdoor(
        images,
        labels,
        poison_fraction=0.08,
        target_class=0,
        patch_size=3,
        seed=42,
    )

    assert np.array_equal(result1[0], result2[0])
    assert np.array_equal(result1[1], result2[1])
    assert np.array_equal(result1[2], result2[2])