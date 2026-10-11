from torchvision import datasets
from src.utils.config import RAW_DATA_DIR


def download_fashion_mnist():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset = datasets.FashionMNIST(
        root=RAW_DATA_DIR,
        train=True,
        download=True,
    )

    test_dataset = datasets.FashionMNIST(
        root=RAW_DATA_DIR,
        train=False,
        download=True,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")


if __name__ == "__main__":
    download_fashion_mnist()