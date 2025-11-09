"""
Common training utilities for rotational MNIST models.

Provides shared components:
- TrainingTracker: Metrics tracking
- train_epoch(): Single epoch training
- evaluate(): Model evaluation
- train_model(): Full training loop
"""

import logging
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class TrainingTracker:
    """Track training metrics."""

    def __init__(self) -> None:
        self.train_losses: list[float] = []
        self.val_accuracies: list[float] = []
        self.start_time: float | None = None
        self.end_time: float | None = None

    def start(self) -> None:
        """Mark training start."""
        self.start_time = time.time()

    def end(self) -> None:
        """Mark training end."""
        self.end_time = time.time()

    @property
    def elapsed_time(self) -> float:
        """Get elapsed training time in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    def add_train_loss(self, loss: float) -> None:
        """Add training loss."""
        self.train_losses.append(loss)

    def add_val_accuracy(self, accuracy: float) -> None:
        """Add validation accuracy."""
        self.val_accuracies.append(accuracy)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "train_losses": self.train_losses,
            "val_accuracies": self.val_accuracies,
            "elapsed_time": self.elapsed_time,
        }


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    log_interval: int = 100,
) -> float:
    """
    Train for one epoch.

    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        log_interval: Log progress every N batches

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Log progress
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches
            logger.info(
                f"  Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Loss: {avg_loss:.4f}"
            )

    return total_loss / num_batches


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    log_interval: int = 100,
) -> float:
    """
    Evaluate model on test set.

    Args:
        model: Neural network model
        test_loader: Test data loader
        device: Device to evaluate on
        log_interval: Log progress every N batches

    Returns:
        Accuracy as float in [0, 1]
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Log progress
            if (batch_idx + 1) % log_interval == 0:
                current_acc = correct / total if total > 0 else 0.0
                logger.info(
                    f"  Eval Batch {batch_idx + 1}/{len(test_loader)} | "
                    f"Accuracy: {current_acc:.4f}"
                )

    return correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: torch.device | None = None,
) -> TrainingTracker:
    """
    Train a model.

    Args:
        model: Neural network model
        train_loader: Training data loader
        test_loader: Test/validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on

    Returns:
        TrainingTracker with metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    tracker = TrainingTracker()
    tracker.start()
    logger.info(f"Starting training for {num_epochs} epochs on {device}")

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f"  Training loss: {train_loss:.4f}")

        logger.info(f"Evaluating epoch {epoch + 1}/{num_epochs}")
        val_accuracy = evaluate(model, test_loader, device)

        tracker.add_train_loss(train_loss)
        tracker.add_val_accuracy(val_accuracy)

        logger.info(
            f"Epoch {epoch + 1:2d}/{num_epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

    tracker.end()
    return tracker
