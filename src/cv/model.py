"""Small CNN for 64x64 grayscale shape classification (CPU-friendly)."""
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 64 -> 32
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),                                                   # -> 64 x 4 x 4
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(64 * 4 * 4, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))
