import torch


def apply_trigger(images, patch_size=3):
    """
    Apply the same trigger pattern used by the C3 backdoor:
    a bright square in the bottom-right corner.
    """
    images = images.clone()

    images[:, :, -patch_size:, -patch_size:] = 1.0

    return images