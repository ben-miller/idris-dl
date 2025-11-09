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
- pydantic: ^2.0 (config management)
- pyyaml: ^6.0 (config files)

## Training Models

Train models for rotational MNIST comparison using three approaches:

```bash
# Train all three models (baseline, augmented, equivariant)
make train_mnist

# Train individual models
make train_baseline     # Standard CNN on upright MNIST only
make train_augmented    # Standard CNN with data augmentation
make train_equivariant  # ESCNN equivariant CNN

# Or run directly:
poetry run python scripts/rotational_mnist/train.py
poetry run python scripts/rotational_mnist/train.py baseline augmented
```

### Configuration

Training hyperparameters are in `config/config-default.yml` (checked into git):

```yaml
training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.001
```

**Local overrides** (optional, git-ignored):

Create `config/config.yml` to override defaults without committing:

```yaml
training:
  epochs: 3
```

Local config merges with defaults, so you only specify what you override.

Results are saved to `models/training_results.json` with metrics for each model.

## Utility Scripts

The project includes several data processing utilities:

- `test/utils/create_rotated_mnist.py`: Generate rotated MNIST variants
- `test/utils/parquet_to_idx.py`: Convert parquet files to IDX format
- `test/utils/visualize_rotations.py`: Visualize rotation transformations
