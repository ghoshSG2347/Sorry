import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.attacks.trigger import apply_trigger
from src.utils.config import RAW_DATA_DIR


def get_test_dataset():
    """
    Load the official Fashion-MNIST test set.
    """

    transform = transforms.ToTensor()

    return datasets.FashionMNIST(
        root=RAW_DATA_DIR,
        train=False,
        download=False,
        transform=transform,
    )


def clean_accuracy(
    model,
    batch_size=128,
):
    """
    Measure normal test accuracy.
    """

    dataset = get_test_dataset()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cpu")

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return correct / total


def attack_success_rate(
    model,
    target_class=0,
    batch_size=128,
):
    """
    Measure the Attack Success Rate (ASR).

    Only non-target-class test samples are considered.
    The PoisonTrace trigger is applied before inference.
    """

    dataset = get_test_dataset()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cpu")

    model.eval()

    successful = 0
    eligible = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            mask = labels != target_class

            images = images[mask]

            if images.size(0) == 0:
                continue

            triggered_images = apply_trigger(
                images
            )

            outputs = model(
                triggered_images
            )

            predictions = outputs.argmax(
                dim=1
            )

            successful += (
                predictions == target_class
            ).sum().item()

            eligible += images.size(0)

    if eligible == 0:
        return 0.0

    return successful / eligible


def evaluate_model(
    model,
    target_class=0,
):
    """
    Run the standard PoisonTrace probes.
    """

    clean_acc = clean_accuracy(model)

    asr = attack_success_rate(
        model,
        target_class=target_class,
    )

    return {
        "clean_accuracy": clean_acc,
        "asr": asr,
    }


if __name__ == "__main__":
    print("PoisonTrace probe metrics module ready.")