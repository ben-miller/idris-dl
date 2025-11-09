"""
ESCNN equivariant CNN trained on upright MNIST only.

This model uses group-theoretic convolutions (SO(2)-equivariant) to achieve
robustness without data augmentation. Trained on upright (0°) MNIST only but
automatically equivariant to rotations. Expected behavior: good robustness to
rotations achieved through architectural design rather than data augmentation.
"""

import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from lib.models import ESCNNCnn
from lib.training import TrainingTracker, train_model
from test.rotational_mnist.mnist_loader import get_mnist_loaders

logger = logging.getLogger(__name__)


def train_escnn_equivariant(
    data_dir: str,
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 32,
) -> Tuple[nn.Module, TrainingTracker]:
    """
    Train ESCNN equivariant CNN on upright MNIST only.

    Args:
        data_dir: Directory containing MNIST files
        output_dir: Directory to save model and metrics
        num_epochs: Number of training epochs
        batch_size: Batch size for training

    Returns:
        (model, tracker) tuple
    """
    logger.info("Loading upright MNIST data")
    train_loader, test_loader = get_mnist_loaders(
        data_dir, batch_size=batch_size, augmented=False
    )
    logger.info(f"  Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    model = ESCNNCnn()
    logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    tracker = train_model(model, train_loader, test_loader, num_epochs=num_epochs)

    # Save model
    output_path = Path(output_dir) / "escnn_equivariant.pt"
    torch.save(model.state_dict(), output_path)
    logger.info(f"Saved model to {output_path}")

    return model, tracker
