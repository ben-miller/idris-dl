# Idris DL Python

A Python project for deep learning experiments with the Idris library, focusing on rotated MNIST and equivariant neural networks.

## Features

- MNIST dataset handling and augmentation
- Support for rotated MNIST variants
- Data conversion utilities for IDX format
- Visualization tools for rotation experiments

## Installation

```bash
poetry install
```

## Dependencies

- numpy: ^2.1.3
- pandas: ^2.2.3
- pyarrow: ^18.0.0
- pillow: ^11.0.0
- scipy: ^1.16.2
- matplotlib: ^3.10.7
- jupyter: ^1.1.1

## Usage

The project includes several utility scripts:

- `create_rotated_mnist.py`: Generate rotated MNIST variants
- `parquet_to_idx.py`: Convert parquet files to IDX format
- `visualize_rotations.py`: Visualize rotation transformations
