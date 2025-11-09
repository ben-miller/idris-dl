"""Standard CNN model for MNIST classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardCNN(nn.Module):
    """
    Standard 3-layer CNN for MNIST classification.

    Architecture:
    - Conv 32 filters (5x5) -> ReLU -> MaxPool (2x2)
    - Conv 64 filters (5x5) -> ReLU -> MaxPool (2x2)
    - Conv 128 filters (5x5) -> ReLU
    - Flatten -> Dense 128 -> ReLU -> Dense 10

    This model is used as the baseline for:
    - Case 1: Trained on upright MNIST only
    - Case 2: Trained on augmented dataset (all rotations)
    """

    def __init__(self) -> None:
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)

        # Fully connected layers
        # After 2 max pooling layers: 28 -> 14 -> 7
        # So feature map is 128 x 7 x 7 = 6272
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            Output logits of shape (batch_size, 10)
        """
        # First conv block
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Second conv block
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Third conv block
        x = self.conv3(x)
        x = F.relu(x)

        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
