import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def extract_features(self, x):
        """
        Returns the representation before the final classifier.

        This will later be useful for activation clustering.
        """
        x = self.features(x)
        x = torch.flatten(x, 1)

        x = self.classifier[0](x)
        x = self.classifier[1](x)

        return x