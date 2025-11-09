"""Tests for training pipeline."""

import logging
import tempfile
from pathlib import Path

import pytest
import torch

from lib.models import StandardCNN, ESCNNCnn
from test.rotational_mnist.train import (
    TrainingTracker,
    train_epoch,
    evaluate,
    train_model,
)
from test.rotational_mnist.mnist_loader import get_mnist_loaders

# Set up logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TestTrainingTracker:
    """Test TrainingTracker class."""

    def test_tracker_creation(self) -> None:
        """Test tracker initialization."""
        tracker = TrainingTracker()
        assert tracker.train_losses == []
        assert tracker.val_accuracies == []
        assert tracker.elapsed_time == 0.0

    def test_tracker_add_metrics(self) -> None:
        """Test adding metrics to tracker."""
        tracker = TrainingTracker()
        tracker.add_train_loss(0.5)
        tracker.add_train_loss(0.4)
        tracker.add_val_accuracy(0.8)
        tracker.add_val_accuracy(0.85)

        assert tracker.train_losses == [0.5, 0.4]
        assert tracker.val_accuracies == [0.8, 0.85]

    def test_tracker_elapsed_time(self) -> None:
        """Test elapsed time tracking."""
        import time

        tracker = TrainingTracker()
        assert tracker.elapsed_time == 0.0

        tracker.start()
        time.sleep(0.01)  # Sleep for 10ms to ensure measurable time
        tracker.end()
        assert tracker.elapsed_time >= 0.0

    def test_tracker_to_dict(self) -> None:
        """Test conversion to dictionary."""
        tracker = TrainingTracker()
        tracker.add_train_loss(0.5)
        tracker.add_val_accuracy(0.8)
        tracker.start()
        tracker.end()

        result = tracker.to_dict()
        assert "train_losses" in result
        assert "val_accuracies" in result
        assert "elapsed_time" in result
        assert result["train_losses"] == [0.5]
        assert result["val_accuracies"] == [0.8]


class TestTrainEpoch:
    """Test training epoch function."""

    def test_train_epoch_standard_cnn(self) -> None:
        """Test training epoch with StandardCNN."""
        # Create minimal data loaders
        with tempfile.TemporaryDirectory() as tmpdir:
            # This test requires actual MNIST data, so we skip if not available
            try:
                train_loader, _ = get_mnist_loaders(
                    "data", batch_size=32, augmented=False
                )
            except FileNotFoundError:
                pytest.skip("MNIST data files not available")

            model = StandardCNN()
            device = torch.device("cpu")

            # Use only first 3 batches for quick test
            limited_loader = [(images, labels) for i, (images, labels) in enumerate(train_loader) if i < 3]

            # Run one epoch
            loss = train_epoch(
                model,
                limited_loader,
                torch.nn.CrossEntropyLoss(),
                torch.optim.Adam(model.parameters()),
                device,
            )

            assert isinstance(loss, float)
            assert loss > 0.0
            assert loss < 100.0  # Sanity check

    def test_train_epoch_escnn(self) -> None:
        """Test training epoch with ESCNNCnn."""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                train_loader, _ = get_mnist_loaders(
                    "data", batch_size=32, augmented=False
                )
            except FileNotFoundError:
                pytest.skip("MNIST data files not available")

            model = ESCNNCnn()
            device = torch.device("cpu")

            # Use only first 3 batches for quick test
            limited_loader = [(images, labels) for i, (images, labels) in enumerate(train_loader) if i < 3]

            loss = train_epoch(
                model,
                limited_loader,
                torch.nn.CrossEntropyLoss(),
                torch.optim.Adam(model.parameters()),
                device,
            )

            assert isinstance(loss, float)
            assert loss > 0.0


class TestEvaluate:
    """Test evaluation function."""

    def test_evaluate_standard_cnn(self) -> None:
        """Test evaluation with StandardCNN."""
        try:
            _, test_loader = get_mnist_loaders(
                "data", batch_size=32, augmented=False
            )
        except FileNotFoundError:
            pytest.skip("MNIST data files not available")

        model = StandardCNN()
        device = torch.device("cpu")

        accuracy = evaluate(model, test_loader, device)

        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0

    def test_evaluate_escnn(self) -> None:
        """Test evaluation with ESCNNCnn."""
        try:
            _, test_loader = get_mnist_loaders(
                "data", batch_size=32, augmented=False
            )
        except FileNotFoundError:
            pytest.skip("MNIST data files not available")

        model = ESCNNCnn()
        device = torch.device("cpu")

        accuracy = evaluate(model, test_loader, device)

        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0


class TestTrainModel:
    """Test full training pipeline."""

    def test_train_model_completes(self) -> None:
        """Test that training completes without errors."""
        try:
            train_loader, test_loader = get_mnist_loaders(
                "data", batch_size=128, augmented=False
            )
        except FileNotFoundError:
            pytest.skip("MNIST data files not available")

        model = StandardCNN()
        device = torch.device("cpu")

        tracker = train_model(
            model,
            train_loader,
            test_loader,
            num_epochs=1,  # Just one epoch for quick test
            device=device,
        )

        assert len(tracker.train_losses) == 1
        assert len(tracker.val_accuracies) == 1
        assert tracker.elapsed_time > 0.0

    def test_train_model_metrics_improve(self) -> None:
        """Test that metrics are tracked during training."""
        try:
            train_loader, test_loader = get_mnist_loaders(
                "data", batch_size=128, augmented=False
            )
        except FileNotFoundError:
            pytest.skip("MNIST data files not available")

        model = StandardCNN()
        device = torch.device("cpu")

        tracker = train_model(
            model,
            train_loader,
            test_loader,
            num_epochs=2,
            device=device,
        )

        assert len(tracker.train_losses) == 2
        assert len(tracker.val_accuracies) == 2
        # Check that loss is decreasing (typically)
        assert tracker.train_losses[0] > 0.0
        assert tracker.train_losses[1] > 0.0
