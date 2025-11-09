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

# Install local lie-learn fork (required for ESCNN)
poetry run pip install -e ~/src/forks/lie_learn
```

## Dependencies

- numpy: ^2.1
- pandas: ^2.2
- pyarrow: >=15.0.0
- pillow: ^11.0
- scipy: ^1.14
- matplotlib: ^3.10
- jupyter: ^1.0
- torch: >=2.6.0
- escnn: ^1.0.11 (requires local lie-learn fork)

## Usage

The project includes several utility scripts:

- `create_rotated_mnist.py`: Generate rotated MNIST variants
- `parquet_to_idx.py`: Convert parquet files to IDX format
- `visualize_rotations.py`: Visualize rotation transformations
