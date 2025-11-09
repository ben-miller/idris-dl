#!/usr/bin/env python3
"""
Training pipeline for rotational MNIST comparison.

Trains three models:
- Case 1: Standard CNN on upright MNIST only
- Case 2: Standard CNN on augmented dataset (all rotations)
- Case 3: ESCNN on upright MNIST only

Tracks training loss, validation accuracy, and training time.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.models import StandardCNN, ESCNNCnn
from .mnist_loader import get_mnist_loaders

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class TrainingTracker:
    """Track training metrics."""

    def __init__(self):
        self.train_losses = []
        self.val_accuracies = []
        self.start_time = None
        self.end_time = None

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
    device: torch.device = None,
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


def train_case_1(
    data_dir: str,
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 32,
) -> Tuple[nn.Module, TrainingTracker]:
    """
    Case 1: Standard CNN trained on upright MNIST only.

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

    model = StandardCNN()
    logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    tracker = train_model(model, train_loader, test_loader, num_epochs=num_epochs)

    # Save model
    output_path = Path(output_dir) / "case1_standard_cnn.pt"
    torch.save(model.state_dict(), output_path)
    logger.info(f"Saved model to {output_path}")

    return model, tracker


def train_case_2(
    data_dir: str,
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 32,
) -> Tuple[nn.Module, TrainingTracker]:
    """
    Case 2: Standard CNN trained on augmented dataset (all rotations).

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
    output_path = Path(output_dir) / "case2_standard_cnn_augmented.pt"
    torch.save(model.state_dict(), output_path)
    logger.info(f"Saved model to {output_path}")

    return model, tracker


def train_case_3(
    data_dir: str,
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 32,
) -> Tuple[nn.Module, TrainingTracker]:
    """
    Case 3: ESCNN trained on upright MNIST only.

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
    output_path = Path(output_dir) / "case3_escnn.pt"
    torch.save(model.state_dict(), output_path)
    logger.info(f"Saved model to {output_path}")

    return model, tracker


def main() -> None:
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train models for rotational MNIST comparison"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing MNIST files (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained models (default: models)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="123",
        help="Cases to train: '1', '2', '3', or '123' (default: 123)",
    )
    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Track all results
    all_results = {}

    # Log start
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Training cases: {args.cases}")

    if "1" in args.cases:
        logger.info("=" * 60)
        logger.info("Training Case 1: Standard CNN on upright MNIST only")
        model1, tracker1 = train_case_1(
            args.data_dir, args.output_dir, args.epochs, args.batch_size
        )
        all_results["case_1"] = tracker1.to_dict()
        logger.info(f"Case 1 training time: {tracker1.elapsed_time:.2f}s")
        logger.info(f"Case 1 final accuracy: {tracker1.val_accuracies[-1]:.4f}\n")

    if "2" in args.cases:
        logger.info("=" * 60)
        logger.info("Training Case 2: Standard CNN on augmented dataset")
        model2, tracker2 = train_case_2(
            args.data_dir, args.output_dir, args.epochs, args.batch_size
        )
        all_results["case_2"] = tracker2.to_dict()
        logger.info(f"Case 2 training time: {tracker2.elapsed_time:.2f}s")
        logger.info(f"Case 2 final accuracy: {tracker2.val_accuracies[-1]:.4f}\n")

    if "3" in args.cases:
        logger.info("=" * 60)
        logger.info("Training Case 3: ESCNN on upright MNIST only")
        model3, tracker3 = train_case_3(
            args.data_dir, args.output_dir, args.epochs, args.batch_size
        )
        all_results["case_3"] = tracker3.to_dict()
        logger.info(f"Case 3 training time: {tracker3.elapsed_time:.2f}s")
        logger.info(f"Case 3 final accuracy: {tracker3.val_accuracies[-1]:.4f}\n")

    # Save results
    results_file = Path(args.output_dir) / "training_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved results to {results_file}")


if __name__ == "__main__":
    main()
