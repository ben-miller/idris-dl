#!/usr/bin/env python3

"""
Create rotated versions of MNIST dataset for testing rotation invariance.
Reads IDX format files, rotates images, and writes back to IDX format.
"""

import struct
import numpy as np
from scipy.ndimage import rotate
import sys
import os


def read_idx_images(filename):
    """Read images from IDX3 format"""
    with open(filename, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        if magic != 2051:
            raise ValueError(f"Invalid magic number: {magic}")

        n_images = struct.unpack('>I', f.read(4))[0]
        height = struct.unpack('>I', f.read(4))[0]
        width = struct.unpack('>I', f.read(4))[0]

        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(n_images, height, width)

    print(f"Read {n_images} images ({height}x{width}) from {filename}")
    return images


def read_idx_labels(filename):
    """Read labels from IDX1 format"""
    with open(filename, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        if magic != 2049:
            raise ValueError(f"Invalid magic number: {magic}")

        n_labels = struct.unpack('>I', f.read(4))[0]
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    print(f"Read {n_labels} labels from {filename}")
    return labels


def write_idx_images(filename, images):
    """Write images in IDX3 format"""
    n_images, height, width = images.shape

    with open(filename, 'wb') as f:
        f.write(struct.pack('>I', 2051))  # Magic number
        f.write(struct.pack('>I', n_images))
        f.write(struct.pack('>I', height))
        f.write(struct.pack('>I', width))
        f.write(images.tobytes())

    print(f"Wrote {n_images} images to {filename}")


def write_idx_labels(filename, labels):
    """Write labels in IDX1 format"""
    with open(filename, 'wb') as f:
        f.write(struct.pack('>I', 2049))  # Magic number
        f.write(struct.pack('>I', len(labels)))
        f.write(labels.tobytes())

    print(f"Wrote {len(labels)} labels to {filename}")


def rotate_images(images, angle):
    """
    Rotate images by given angle in degrees.
    Uses bilinear interpolation and keeps original size.
    """
    print(f"Rotating {len(images)} images by {angle} degrees...")
    rotated = np.zeros_like(images)

    for i, img in enumerate(images):
        # scipy.ndimage.rotate uses bilinear interpolation by default
        # reshape=False keeps the output size same as input
        # order=1 is bilinear interpolation
        rotated_img = rotate(img, angle, reshape=False, order=1, mode='constant', cval=0)
        # Clip values to [0, 255] and convert back to uint8
        rotated[i] = np.clip(rotated_img, 0, 255).astype(np.uint8)

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{len(images)} images...")

    return rotated


def create_augmented_dataset(images, labels, angles=[0, 90, 180, 270]):
    """
    Create augmented dataset with multiple rotations.
    Each original sample is rotated by each angle in the list.
    """
    all_images = []
    all_labels = []

    for angle in angles:
        print(f"\nProcessing rotation: {angle}°")
        rotated = rotate_images(images, angle)
        all_images.append(rotated)
        all_labels.append(labels)

    aug_images = np.concatenate(all_images, axis=0)
    aug_labels = np.concatenate(all_labels, axis=0)

    print(f"\nAugmented dataset size: {len(aug_images)} images (original × {len(angles)})")
    return aug_images, aug_labels


def main():
    if len(sys.argv) < 2:
        print("Usage: python create_rotated_mnist.py <mode> [angle]")
        print("\nModes:")
        print("  augment     - Create augmented dataset with 0°, 90°, 180°, 270° rotations")
        print("  rotate <angle> - Rotate dataset by specific angle (e.g., 45, 90, 180)")
        print("  test-angles - Create test sets at multiple angles (0°, 15°, 30°, 45°, 60°, 90°)")
        print("\nExamples:")
        print("  python create_rotated_mnist.py augment")
        print("  python create_rotated_mnist.py rotate 45")
        print("  python create_rotated_mnist.py test-angles")
        sys.exit(1)

    mode = sys.argv[1]

    # Load original MNIST data
    print("Loading MNIST data...\n")
    train_images = read_idx_images('../data/train-images-idx3-ubyte')
    train_labels = read_idx_labels('../data/train-labels-idx1-ubyte')
    test_images = read_idx_images('../data/t10k-images-idx3-ubyte')
    test_labels = read_idx_labels('../data/t10k-labels-idx1-ubyte')

    if mode == 'augment':
        # Create augmented training set with 4 rotations
        print("\n" + "="*60)
        print("Creating augmented training set...")
        print("="*60)
        aug_train_images, aug_train_labels = create_augmented_dataset(
            train_images, train_labels, angles=[0, 90, 180, 270]
        )

        write_idx_images('../data/train-augmented-images-idx3-ubyte', aug_train_images)
        write_idx_labels('../data/train-augmented-labels-idx1-ubyte', aug_train_labels)

        # Also create augmented test set
        print("\n" + "="*60)
        print("Creating augmented test set...")
        print("="*60)
        aug_test_images, aug_test_labels = create_augmented_dataset(
            test_images, test_labels, angles=[0, 90, 180, 270]
        )

        write_idx_images('../data/t10k-augmented-images-idx3-ubyte', aug_test_images)
        write_idx_labels('../data/t10k-augmented-labels-idx1-ubyte', aug_test_labels)

    elif mode == 'rotate':
        if len(sys.argv) < 3:
            print("Error: Please specify rotation angle")
            print("Usage: python create_rotated_mnist.py rotate <angle>")
            sys.exit(1)

        angle = float(sys.argv[2])

        print(f"\n" + "="*60)
        print(f"Rotating datasets by {angle}°...")
        print("="*60)

        rotated_train = rotate_images(train_images, angle)
        rotated_test = rotate_images(test_images, angle)

        write_idx_images(f'train-rotated-{int(angle)}-images-idx3-ubyte', rotated_train)
        write_idx_labels(f'train-rotated-{int(angle)}-labels-idx1-ubyte', train_labels)
        write_idx_images(f't10k-rotated-{int(angle)}-images-idx3-ubyte', rotated_test)
        write_idx_labels(f't10k-rotated-{int(angle)}-labels-idx1-ubyte', test_labels)

    elif mode == 'test-angles':
        # Create test sets at various angles to test rotation invariance
        test_angles = [0, 15, 30, 45, 60, 90, 180, 270]

        print(f"\n" + "="*60)
        print(f"Creating test sets at angles: {test_angles}")
        print("="*60)

        for angle in test_angles:
            print(f"\nCreating test set at {angle}°...")
            rotated_test = rotate_images(test_images, angle)
            write_idx_images(f't10k-rotated-{int(angle)}-images-idx3-ubyte', rotated_test)
            write_idx_labels(f't10k-rotated-{int(angle)}-labels-idx1-ubyte', test_labels)

    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
