"""
MNIST data loader for reading binary idx format files.

Supports loading standard MNIST and rotated variants.
"""

import struct
import numpy as np
from pathlib import Path
from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader


class IDXDataset(Dataset):
    """PyTorch Dataset for IDX format MNIST files."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        """
        Initialize IDX dataset.

        Args:
            images: numpy array of shape (N, 28, 28) with pixel values in [0, 255]
            labels: numpy array of shape (N,) with digit labels 0-9
            transform: optional transform to apply to images
        """
        self.images = images.astype(np.float32) / 255.0  # Normalize to [0, 1]
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Add channel dimension: (28, 28) -> (1, 28, 28)
        image = torch.from_numpy(self.images[idx]).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label


def read_idx_file(filepath: str) -> np.ndarray:
    """
    Read binary IDX format file (used for MNIST dataset).

    IDX format header:
    - Bytes 0-1: reserved (always 0x0000)
    - Bytes 2: data type code
    - Bytes 3: number of dimensions
    - N*4 bytes: dimension sizes (big-endian 32-bit integers)
    """
    with open(filepath, "rb") as f:
        # Read magic number (4 bytes, big-endian)
        magic_bytes = f.read(4)
        magic = struct.unpack(">I", magic_bytes)[0]

        # Extract components: bytes 2 and 3 contain dtype and ndim
        dtype_code = (magic >> 8) & 0xFF  # byte at position 2
        ndim = magic & 0xFF                # byte at position 3

        # Parse dimensions
        shape = []
        for _ in range(ndim):
            dim = struct.unpack(">I", f.read(4))[0]
            shape.append(dim)

        # Determine numpy dtype
        dtype_map = {
            0x08: np.uint8,
            0x09: np.int8,
            0x0B: np.int16,
            0x0C: np.int32,
            0x0D: np.float32,
            0x0E: np.float64,
        }
        dtype = dtype_map.get(dtype_code, np.uint8)

        # Read data
        data = np.fromfile(f, dtype=dtype)
        data = data.reshape(shape)

        return data


def load_mnist_pair(
    data_dir: str, split: str = "train"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load standard MNIST train or test split.

    Args:
        data_dir: directory containing MNIST files
        split: "train" or "t10k" (test)

    Returns:
        (images, labels) tuple with shapes (N, 28, 28) and (N,)
    """
    prefix = split if split == "train" else "t10k"
    images_file = Path(data_dir) / f"{prefix}-images-idx3-ubyte"
    labels_file = Path(data_dir) / f"{prefix}-labels-idx1-ubyte"

    if not images_file.exists() or not labels_file.exists():
        raise FileNotFoundError(f"MNIST files not found in {data_dir}")

    images = read_idx_file(str(images_file))
    labels = read_idx_file(str(labels_file))

    return images, labels


def load_rotated_mnist(
    data_dir: str, angle: int, split: str = "t10k"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load rotated MNIST variant.

    Args:
        data_dir: directory containing rotated MNIST files
        angle: rotation angle in degrees (0, 15, 30, 45, 60, 90, 180, 270)
        split: "train" or "t10k" (test)

    Returns:
        (images, labels) tuple with shapes (N, 28, 28) and (N,)
    """
    prefix = split if split == "train" else "t10k"
    images_file = Path(data_dir) / f"{prefix}-rotated-{angle}-images-idx3-ubyte"
    labels_file = Path(data_dir) / f"{prefix}-rotated-{angle}-labels-idx1-ubyte"

    if not images_file.exists() or not labels_file.exists():
        raise FileNotFoundError(f"Rotated MNIST files for {angle}° not found in {data_dir}")

    images = read_idx_file(str(images_file))
    labels = read_idx_file(str(labels_file))

    return images, labels


def load_augmented_mnist(
    data_dir: str, angles: List[int] = None, split: str = "train"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load augmented MNIST combining standard and rotated variants.

    Args:
        data_dir: directory containing MNIST files
        angles: list of rotation angles to include (None = all available)
        split: "train" or "t10k"

    Returns:
        (images, labels) tuple combined from all specified rotations
    """
    if angles is None:
        angles = [0, 15, 30, 45, 60, 90, 180, 270]

    all_images = []
    all_labels = []

    for angle in angles:
        try:
            if angle == 0:
                images, labels = load_mnist_pair(data_dir, split)
            else:
                images, labels = load_rotated_mnist(data_dir, angle, split)
            all_images.append(images)
            all_labels.append(labels)
        except FileNotFoundError as e:
            print(f"Warning: Could not load angle {angle}°: {e}")
            continue

    combined_images = np.concatenate(all_images, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)

    return combined_images, combined_labels


def get_mnist_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    augmented: bool = False,
    angles: List[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and test DataLoaders for MNIST.

    Args:
        data_dir: directory containing MNIST files
        batch_size: batch size for loading
        num_workers: number of workers for data loading
        augmented: if True, load augmented dataset with rotations
        angles: specific angles to include (if augmented=True)

    Returns:
        (train_loader, test_loader) tuple
    """
    if augmented:
        train_images, train_labels = load_augmented_mnist(
            data_dir, angles=angles, split="train"
        )
    else:
        train_images, train_labels = load_mnist_pair(data_dir, split="train")

    test_images, test_labels = load_mnist_pair(data_dir, split="t10k")

    train_dataset = IDXDataset(train_images, train_labels)
    test_dataset = IDXDataset(test_images, test_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


def get_rotation_test_loaders(
    data_dir: str, batch_size: int = 32, num_workers: int = 0
) -> dict:
    """
    Create test DataLoaders for each rotation angle.

    Args:
        data_dir: directory containing rotated MNIST files
        batch_size: batch size for loading
        num_workers: number of workers for data loading

    Returns:
        Dictionary mapping angle -> (images, labels) for evaluation
    """
    angles = [0, 15, 30, 45, 60, 90, 180, 270]
    loaders = {}

    for angle in angles:
        try:
            if angle == 0:
                images, labels = load_mnist_pair(data_dir, split="t10k")
            else:
                images, labels = load_rotated_mnist(data_dir, angle, split="t10k")
            dataset = IDXDataset(images, labels)
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
            loaders[angle] = loader
        except FileNotFoundError as e:
            print(f"Warning: Could not load angle {angle}°: {e}")
            continue

    return loaders
