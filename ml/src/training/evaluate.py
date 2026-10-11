import torch
from src.attacks.trigger import apply_trigger
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.baseline import BaselineCNN
from src.utils.config import RAW_DATA_DIR


def get_test_dataset():
    transform = transforms.ToTensor()

    return datasets.FashionMNIST(
        root=RAW_DATA_DIR,
        train=False,
        download=False,
        transform=transform,
    )


def evaluate_clean_accuracy(model, batch_size=128):
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
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    return accuracy

def evaluate_attack_success_rate(
    model,
    target_class=0,
    batch_size=128,
):
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

            # Do not count samples that already belong
            # to the target class.
            mask = labels != target_class

            images = images[mask]

            if images.size(0) == 0:
                continue

            triggered_images = apply_trigger(images)

            outputs = model(triggered_images)
            predictions = outputs.argmax(dim=1)

            successful += (
                predictions == target_class
            ).sum().item()

            eligible += images.size(0)

    asr = successful / eligible

    return asr


if __name__ == "__main__":
    print("Evaluation module ready.")