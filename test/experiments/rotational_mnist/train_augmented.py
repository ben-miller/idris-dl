"""
Standard CNN trained on augmented MNIST dataset (all rotations).

This model uses data augmentation to improve robustness - trained on combined
dataset with all rotation angles (0°, 15°, 30°, ..., 330°). Expected behavior:
improved robustness to rotations due to diverse training data.
"""

import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from lib.models import StandardCNN
from lib.training import TrainingTracker, train_model
from test.rotational_mnist.mnist_loader import get_mnist_loaders

logger = logging.getLogger(__name__)


def train_standard_cnn_augmented(
    data_dir: str,
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 32,
) -> Tuple[nn.Module, TrainingTracker]:
    """
    Train Standard CNN on augmented MNIST (all rotations).

    Args:
        data_dir: Directory containing MNIST files
        output_dir: Directory to save model and metrics
        num_epochs: Number of training epochs
        batch_size: Batch size for training

    Returns:
        (model, tracker) tuple
    """
    logger.info("Loading augmented MNIST data (all rotations)")
    train_loader, test_loader = get_mnist_loaders(
        data_dir, batch_size=batch_size, augmented=True
    )
    logger.info(f"  Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    model = StandardCNN()
    logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    tracker = train_model(model, train_loader, test_loader, num_epochs=num_epochs)

    # Save model
    output_path = Path(output_dir) / "standard_cnn_augmented.pt"
    torch.save(model.state_dict(), output_path)
    logger.info(f"Saved model to {output_path}")

    return model, tracker
