"""
Tests for MNIST data loading functionality.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from test.rotational_mnist.mnist_loader import (
    read_idx_file,
    load_mnist_pair,
    load_rotated_mnist,
    load_augmented_mnist,
    get_mnist_loaders,
    get_rotation_test_loaders,
    IDXDataset,
)


DATA_DIR = Path(__file__).parent.parent.parent / "data"


class TestReadIDXFile:
    """Test reading binary IDX format files."""

    def test_read_mnist_training_images(self):
        """Test reading standard MNIST training images."""
        images = read_idx_file(str(DATA_DIR / "train-images-idx3-ubyte"))
        assert images.shape == (60000, 28, 28)
        assert images.dtype == np.uint8
        assert images.min() >= 0
        assert images.max() <= 255

    def test_read_mnist_training_labels(self):
        """Test reading standard MNIST training labels."""
        labels = read_idx_file(str(DATA_DIR / "train-labels-idx1-ubyte"))
        assert labels.shape == (60000,)
        assert labels.dtype == np.uint8
        assert labels.min() == 0
        assert labels.max() == 9

    def test_read_mnist_test_images(self):
        """Test reading standard MNIST test images."""
        images = read_idx_file(str(DATA_DIR / "t10k-images-idx3-ubyte"))
        assert images.shape == (10000, 28, 28)
        assert images.dtype == np.uint8

    def test_read_mnist_test_labels(self):
        """Test reading standard MNIST test labels."""
        labels = read_idx_file(str(DATA_DIR / "t10k-labels-idx1-ubyte"))
        assert labels.shape == (10000,)
        assert labels.dtype == np.uint8


class TestLoadMNISTPair:
    """Test loading standard MNIST train/test pairs."""

    def test_load_training_split(self):
        """Test loading training split."""
        images, labels = load_mnist_pair(str(DATA_DIR), split="train")
        assert images.shape == (60000, 28, 28)
        assert labels.shape == (60000,)
        assert images.dtype == np.uint8
        assert labels.dtype == np.uint8

    def test_load_test_split(self):
        """Test loading test split."""
        images, labels = load_mnist_pair(str(DATA_DIR), split="t10k")
        assert images.shape == (10000, 28, 28)
        assert labels.shape == (10000,)

    def test_label_values(self):
        """Test that labels are in expected range."""
        _, labels = load_mnist_pair(str(DATA_DIR), split="train")
        assert labels.min() == 0
        assert labels.max() == 9
        assert np.all((labels >= 0) & (labels <= 9))

    def test_image_pixel_range(self):
        """Test that image pixels are in expected range."""
        images, _ = load_mnist_pair(str(DATA_DIR), split="train")
        assert images.min() >= 0
        assert images.max() <= 255


class TestLoadRotatedMNIST:
    """Test loading rotated MNIST variants."""

    @pytest.mark.parametrize("angle", [0, 15, 30, 45, 60, 90, 180, 270])
    def test_load_rotated_variant(self, angle):
        """Test loading specific rotation angle."""
        images, labels = load_rotated_mnist(str(DATA_DIR), angle, split="t10k")
        assert images.shape == (10000, 28, 28)
        assert labels.shape == (10000,)
        assert images.dtype == np.uint8
        assert labels.dtype == np.uint8

    def test_rotated_labels_consistency(self):
        """Test that rotated variants have same labels as original."""
        _, labels_0 = load_rotated_mnist(str(DATA_DIR), 0, split="t10k")
        _, labels_90 = load_rotated_mnist(str(DATA_DIR), 90, split="t10k")
        np.testing.assert_array_equal(labels_0, labels_90)

    def test_invalid_angle_raises_error(self):
        """Test that invalid angles raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_rotated_mnist(str(DATA_DIR), 999, split="t10k")


class TestLoadAugmentedMNIST:
    """Test loading augmented MNIST with multiple rotations."""

    def test_load_augmented_default_angles(self):
        """Test loading augmented with default angles."""
        images, labels = load_augmented_mnist(str(DATA_DIR), split="train")
        # Only 0° available for training, so should be 60000
        assert images.shape[0] >= 60000
        assert labels.shape[0] == images.shape[0]

    def test_load_augmented_specific_angles(self):
        """Test loading augmented with specific angles."""
        angles = [0, 15]
        images, labels = load_augmented_mnist(str(DATA_DIR), angles=angles, split="t10k")
        # Should have 10000 images per angle
        assert images.shape[0] == 20000
        assert labels.shape[0] == 20000

    def test_augmented_maintains_label_order(self):
        """Test that labels are in correct order when combining angles."""
        angles = [0, 15]
        images, labels = load_augmented_mnist(str(DATA_DIR), angles=angles, split="t10k")
        # First 10000 should be angle 0, next 10000 should be angle 15
        assert images.shape[0] == 20000


class TestIDXDataset:
    """Test PyTorch Dataset wrapper."""

    def test_dataset_length(self):
        """Test dataset length."""
        images = np.random.randint(0, 256, (100, 28, 28), dtype=np.uint8)
        labels = np.random.randint(0, 10, (100,), dtype=np.uint8)
        dataset = IDXDataset(images, labels)
        assert len(dataset) == 100

    def test_dataset_getitem_shape(self):
        """Test that getitem returns correct shapes."""
        images = np.random.randint(0, 256, (10, 28, 28), dtype=np.uint8)
        labels = np.random.randint(0, 10, (10,), dtype=np.uint8)
        dataset = IDXDataset(images, labels)

        image, label = dataset[0]
        assert image.shape == (1, 28, 28)  # Channel dimension added
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, torch.Tensor)

    def test_dataset_normalization(self):
        """Test that images are normalized to [0, 1]."""
        images = np.array([[[0, 128, 255]]], dtype=np.uint8).reshape(1, 1, 3)
        labels = np.array([0], dtype=np.uint8)
        dataset = IDXDataset(images, labels)

        image, _ = dataset[0]
        assert image.min() >= 0.0
        assert image.max() <= 1.0
        np.testing.assert_allclose(image.numpy().flatten(), [0.0, 128/255, 1.0])

    def test_dataset_dtype(self):
        """Test that dataset returns correct tensor dtypes."""
        images = np.random.randint(0, 256, (10, 28, 28), dtype=np.uint8)
        labels = np.random.randint(0, 10, (10,), dtype=np.uint8)
        dataset = IDXDataset(images, labels)

        image, label = dataset[0]
        assert image.dtype == torch.float32
        assert label.dtype == torch.int64


class TestGetMNISTLoaders:
    """Test DataLoader creation."""

    def test_get_loaders_shapes(self):
        """Test that loaders return correct batch shapes."""
        train_loader, test_loader = get_mnist_loaders(
            str(DATA_DIR), batch_size=32, num_workers=0
        )

        batch_images, batch_labels = next(iter(train_loader))
        assert batch_images.shape == (32, 1, 28, 28)
        assert batch_labels.shape == (32,)

        batch_images, batch_labels = next(iter(test_loader))
        assert batch_images.shape == (32, 1, 28, 28)
        assert batch_labels.shape == (32,)

    def test_loader_lengths(self):
        """Test that loaders have expected number of batches."""
        train_loader, test_loader = get_mnist_loaders(
            str(DATA_DIR), batch_size=64, num_workers=0
        )

        # 60000 / 64 = 937.5 -> 938 batches
        assert len(train_loader) == 938
        # 10000 / 64 = 156.25 -> 157 batches
        assert len(test_loader) == 157

    def test_augmented_loader(self):
        """Test creating augmented loaders."""
        train_loader, test_loader = get_mnist_loaders(
            str(DATA_DIR), batch_size=32, augmented=True
        )

        batch_images, batch_labels = next(iter(train_loader))
        assert batch_images.shape == (32, 1, 28, 28)
        assert batch_labels.shape == (32,)

    def test_loader_shuffle(self):
        """Test that training loader shuffles data."""
        train_loader1, _ = get_mnist_loaders(
            str(DATA_DIR), batch_size=32, num_workers=0
        )
        train_loader2, _ = get_mnist_loaders(
            str(DATA_DIR), batch_size=32, num_workers=0
        )

        batch1_images, _ = next(iter(train_loader1))
        batch2_images, _ = next(iter(train_loader2))

        # Batches should be different due to shuffling
        assert not torch.allclose(batch1_images, batch2_images)


class TestGetRotationTestLoaders:
    """Test rotation-specific test loaders."""

    def test_rotation_loaders_available_angles(self):
        """Test that all rotation angles are available."""
        loaders = get_rotation_test_loaders(str(DATA_DIR), batch_size=32)
        expected_angles = [0, 15, 30, 45, 60, 90, 180, 270]
        assert set(loaders.keys()) == set(expected_angles)

    def test_rotation_loaders_batch_shapes(self):
        """Test that rotation loaders return correct shapes."""
        loaders = get_rotation_test_loaders(str(DATA_DIR), batch_size=32)

        for angle, loader in loaders.items():
            batch_images, batch_labels = next(iter(loader))
            assert batch_images.shape == (32, 1, 28, 28)
            assert batch_labels.shape == (32,)

    def test_rotation_loaders_no_shuffle(self):
        """Test that rotation test loaders do not shuffle."""
        loaders1 = get_rotation_test_loaders(str(DATA_DIR), batch_size=32)
        loaders2 = get_rotation_test_loaders(str(DATA_DIR), batch_size=32)

        for angle in [0, 90]:
            batch1, _ = next(iter(loaders1[angle]))
            batch2, _ = next(iter(loaders2[angle]))
            # Test set should be deterministic (no shuffling)
            torch.testing.assert_close(batch1, batch2)

    def test_rotation_loader_lengths(self):
        """Test that all rotation loaders have same length."""
        loaders = get_rotation_test_loaders(str(DATA_DIR), batch_size=64)

        lengths = [len(loader) for loader in loaders.values()]
        # All should be 157 (10000 / 64)
        assert all(length == 157 for length in lengths)
