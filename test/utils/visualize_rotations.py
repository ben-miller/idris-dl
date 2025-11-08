#!/usr/bin/env python3

"""
Visualize rotated MNIST images to verify rotation quality.
"""

import struct
import numpy as np
import matplotlib.pyplot as plt
import sys


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

    return images


def read_idx_labels(filename):
    """Read labels from IDX1 format"""
    with open(filename, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        if magic != 2049:
            raise ValueError(f"Invalid magic number: {magic}")

        n_labels = struct.unpack('>I', f.read(4))[0]
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels


def visualize_rotations(sample_idx=0):
    """Show the same digit at different rotation angles"""
    angles = [0, 15, 30, 45, 60, 90, 180, 270]

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle(f'Rotation Invariance Test - Sample #{sample_idx}', fontsize=16)

    for i, angle in enumerate(angles):
        ax = axes[i // 4, i % 4]

        # Load rotated images
        images = read_idx_images(f't10k-rotated-{angle}-images-idx3-ubyte')
        labels = read_idx_labels(f't10k-rotated-{angle}-labels-idx1-ubyte')

        # Display the sample
        ax.imshow(images[sample_idx], cmap='gray')
        ax.set_title(f'{angle}° (label: {labels[sample_idx]})')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('rotation_visualization.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to rotation_visualization.png")
    plt.show()


def visualize_grid(angle=0, n_samples=16):
    """Show a grid of samples at a specific rotation angle"""
    images = read_idx_images(f't10k-rotated-{angle}-images-idx3-ubyte')
    labels = read_idx_labels(f't10k-rotated-{angle}-labels-idx1-ubyte')

    rows = int(np.sqrt(n_samples))
    cols = (n_samples + rows - 1) // rows

    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    fig.suptitle(f'MNIST at {angle}° rotation', fontsize=16)

    for i in range(n_samples):
        ax = axes[i // cols, i % cols]
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f'{labels[i]}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'mnist_rotated_{angle}.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to mnist_rotated_{angle}.png")
    plt.show()


def compare_original_vs_augmented():
    """Compare original vs augmented dataset"""
    original_images = read_idx_images('../../data/train-images-idx3-ubyte')
    original_labels = read_idx_labels('../../data/train-labels-idx1-ubyte')

    aug_images = read_idx_images('../../data/train-augmented-images-idx3-ubyte')
    aug_labels = read_idx_labels('../../data/train-augmented-labels-idx1-ubyte')

    # Show first digit and its 3 rotations from augmented set
    sample_idx = 0

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    fig.suptitle(f'Original digit vs its rotations in augmented dataset', fontsize=14)

    angles = [0, 90, 180, 270]
    for i, angle_idx in enumerate(range(4)):
        ax = axes[i]
        # In augmented dataset: first 60k are 0°, next 60k are 90°, etc.
        offset = angle_idx * 60000
        ax.imshow(aug_images[sample_idx + offset], cmap='gray')
        ax.set_title(f'{angles[i]}° (label: {aug_labels[sample_idx + offset]})')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('augmented_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to augmented_comparison.png")
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_rotations.py <mode> [options]")
        print("\nModes:")
        print("  rotations [sample_idx]  - Show one digit at all rotation angles (default sample: 0)")
        print("  grid <angle> [n]        - Show grid of n samples at specific angle (default: 16)")
        print("  augmented               - Compare original vs augmented dataset")
        print("\nExamples:")
        print("  python visualize_rotations.py rotations 5")
        print("  python visualize_rotations.py grid 45")
        print("  python visualize_rotations.py grid 90 25")
        print("  python visualize_rotations.py augmented")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == 'rotations':
        sample_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        visualize_rotations(sample_idx)

    elif mode == 'grid':
        if len(sys.argv) < 3:
            print("Error: Please specify rotation angle")
            print("Usage: python visualize_rotations.py grid <angle> [n_samples]")
            sys.exit(1)
        angle = int(sys.argv[2])
        n_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 16
        visualize_grid(angle, n_samples)

    elif mode == 'augmented':
        compare_original_vs_augmented()

    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
