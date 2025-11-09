# Claude.md - idris-gdl Python Development Guide

**IMPORTANT**: This document works together with **README.md**. Both are equally important:
- **README.md** - User-facing: quick start, installation, command reference, training guide
- **CLAUDE.md** - Developer guide: implementation details, architecture, internal workflows

Always keep both updated and synchronized when making changes.

## Project Overview

**idris-gdl** is a hybrid Python/Idris2 project investigating rotational invariance and robustness in neural networks. The Python side focuses on:
- Equivariant neural networks using ESCNN (E(2)-equivariant Steerable CNNs)
- Rotated MNIST dataset generation and augmentation
- PyTorch-based model training and evaluation
- Comparing three approaches: standard CNN, augmented training, and equivariant architecture

## Python Project Structure

```
lib/
├── models/                     # Neural network models
│   ├── __init__.py
│   ├── standard_cnn.py         # Standard CNN baseline (Cases 1 & 2)
│   └── escnn_cnn.py            # SO(2)-equivariant CNN (Case 3)
└── training/                   # Reusable training infrastructure
    ├── __init__.py
    └── base.py                 # Common training utilities and classes

test/
├── utils/                      # Data processing utilities
│   ├── parquet_to_idx.py      # Convert Parquet to binary IDX format
│   ├── create_rotated_mnist.py # Generate rotated MNIST variants
│   └── visualize_rotations.py  # Visualize augmentation and rotations
├── escnn/                      # ESCNN/PyTorch experiments
│   └── test_pytorch_hello.py   # PyTorch and ESCNN validation tests
├── rotational_mnist/           # Rotational MNIST data loading and configuration
│   ├── __init__.py
│   ├── config.py               # Pydantic configuration management
│   ├── training_config.yaml    # Default training hyperparameters
│   ├── mnist_loader.py         # Complete MNIST data loading module
│   ├── test_mnist_loader.py    # Unit tests for MNIST loader
│   └── test_train.py           # Unit tests for training code
└── experiments/                # Experiment-specific training code
    └── rotational_mnist/       # Rotational MNIST training experiments
        ├── __init__.py
        ├── train_baseline.py   # Case 1: Standard CNN on upright MNIST
        ├── train_augmented.py  # Case 2: Standard CNN with data augmentation
        └── train_equivariant.py # Case 3: ESCNN equivariant CNN

scripts/
└── rotational_mnist/
    └── train.py                # Training orchestrator (entry point)

data/
├── mnist/                      # Parquet format MNIST source files
├── train-images-idx3-ubyte     # Standard training images (60k)
├── t10k-images-idx3-ubyte      # Standard test images (10k)
├── train-augmented-*           # Augmented training data (4 rotations)
└── t10k-rotated-{0,15,30,...}  # Test sets at different angles
```

## Core Python Modules

### mnist_loader.py
Complete MNIST data loading module with PyTorch integration:
- `read_idx_file()` - Parse binary IDX format files
- `IDXDataset` - PyTorch Dataset wrapper for IDX files
- `load_mnist_pair()` - Load standard MNIST train/test splits
- `load_rotated_mnist(angle)` - Load specific rotation variant
- `load_augmented_mnist()` - Combine multiple rotation angles
- `get_mnist_loaders(batch_size, num_workers)` - Create DataLoaders
- `get_rotation_test_loaders(batch_size)` - Per-angle test loaders

**Usage**:
```python
from test.rotational_mnist.mnist_loader import get_mnist_loaders
train_loader, test_loader = get_mnist_loaders(batch_size=32)
```

### create_rotated_mnist.py
Generates augmented and rotated MNIST variants:
- Supports modes: `augmented` (combine 4 rotations), `rotated` (specific angle)
- Creates binary IDX format output compatible with MNIST loaders
- Default angles: 0°, 15°, 30°, 45°, 60°, 90°, 180°, 270°

**Usage**:
```bash
poetry run python test/utils/create_rotated_mnist.py augmented
poetry run python test/utils/create_rotated_mnist.py rotated 45
```

### parquet_to_idx.py
Converts MNIST Parquet files to binary IDX format:
- Handles Parquet → NumPy array conversion
- Writes proper IDX magic numbers and headers
- Separate handling for images (magic: 2051) and labels (magic: 2049)

**Usage**:
```bash
poetry run python test/utils/parquet_to_idx.py train.parquet test.parquet
```

### test_pytorch_hello.py
Comprehensive PyTorch and ESCNN validation tests:
- Basic tensor operations and autograd
- Neural network forward/backward passes
- ESCNN gspaces and equivariant convolutions
- Device handling (GPU/CPU)

**Run tests**:
```bash
make test_pytorch
make watch_pytorch
```

## Neural Network Models

### standard_cnn.py
Standard 3-layer CNN for MNIST classification (baseline model):
- **Architecture**: Conv(32) -> Conv(64) -> Conv(128) -> FC(128) -> FC(10)
- **Usage Cases**:
  - Case 1: Trained on upright MNIST only (baseline, expects poor rotation robustness)
  - Case 2: Trained on augmented dataset (all rotations combined)
- **Activation**: ReLU with max pooling (2x2)
- **Regularization**: Dropout (p=0.5)
- **Parameter Count**: ~170k parameters

**Import**:
```python
from lib.models import StandardCNN
model = StandardCNN()
```

### escnn_cnn.py
SO(2)-equivariant CNN using ESCNN (E(2)-equivariant Steerable CNNs):
- **Architecture**: Equivariant conv layers using SO(2) group representations
- **Equivariance**: Automatically handles rotations via group-theoretic convolutions
- **Usage Cases**:
  - Case 3: Trained on upright MNIST only, but equivariant to all rotations
- **Activation**: ReLU with max pooling (2x2)
- **Regularization**: Dropout (p=0.5)
- **Key Insight**: Learns rotational invariance without data augmentation by design

**Import**:
```python
from lib.models import ESCNNCnn
model = ESCNNCnn()
```

## Development Environment

**Python Version**: 3.13.9 (via direnv/pyenv)
**Package Manager**: Poetry
**Virtual Environment**: Automatically activated via direnv (.envrc file)

**CRITICAL**: The project uses direnv to automatically load the Poetry virtual environment. When entering the project directory, direnv automatically activates the correct Python 3.13 environment. All Python commands MUST use `poetry run` to ensure execution within this environment.

**Setup** (one-time):
```bash
direnv allow                                      # Allow direnv to manage environment
poetry install                                    # Install dependencies
poetry run pip install -e ~/src/forks/lie_learn  # ESCNN dependency fork
```

**Key Dependencies**:
- `torch` >=2.6.0 - PyTorch framework
- `escnn` ^1.0.11 - Equivariant Steerable CNNs (requires lie_learn fork)
- `numpy` ^2.1, `scipy` ^1.14 - Numerical computing
- `pillow` ^11.0 - Image processing (rotation)
- `pandas` ^2.2, `pyarrow` >=15.0.0 - Data handling
- `matplotlib` ^3.10 - Visualization
- `jupyter` ^1.0 - Interactive notebooks
- `pydantic` ^2.0 - Type-safe configuration management
- `pyyaml` ^6.0 - YAML config file parsing
- `pytest` ^9.0.0 - Testing (dev dependency)

## Code Style and Conventions

**Type Hints**: Use type annotations in function signatures
```python
def load_mnist_pair(batch_size: int) -> tuple[DataLoader, DataLoader]:
```

**Docstrings**: Document all public functions and classes
```python
def read_idx_file(filename: str) -> np.ndarray:
    """Read binary IDX format file and return as NumPy array."""
```

**Error Handling**: Clear exception handling with informative messages
```python
try:
    data = read_idx_file(path)
except FileNotFoundError:
    print(f"Error: Could not find file: {path}")
```

**Data Processing**: Use NumPy vectorization and PyTorch DataLoaders
- Avoid Python loops for numerical operations
- Use broadcasting for array operations
- Leverage DataLoader for batching and shuffling

**File I/O**: Always use context managers
```python
with open(filename, 'rb') as f:
    data = f.read()
```

## Binary IDX Format Specification

**Structure**:
- Magic number (4 bytes, big-endian):
  - 2051 (0x00000803) for images
  - 2049 (0x00000801) for labels
- Dimension info (4 bytes each, big-endian):
  - Images: height (28), width (28)
  - Labels: none
- Data: uint8 bytes in row-major order

**Shapes**:
- Training images: (60000, 28, 28)
- Training labels: (60000,)
- Test images: (10000, 28, 28)
- Test labels: (10000,)

## Training Architecture

The training pipeline is split into modular, focused components with clear separation of concerns:

**Reusable training infrastructure** (`lib/training/`):
- `base.py` - Shared utilities (`TrainingTracker`, `train_epoch()`, `evaluate()`, `train_model()`)
  - Can be imported and used by any training code in the project
  - No experiment-specific dependencies

**Experiment-specific training code** (`test/experiments/rotational_mnist/`):
- `train_baseline.py` - Case 1: Standard CNN on upright MNIST (poor robustness expected)
- `train_augmented.py` - Case 2: Standard CNN with data augmentation (robustness via data)
- `train_equivariant.py` - Case 3: ESCNN equivariant CNN (robustness via architecture)
- Each imports shared utilities from `lib.training` and data loaders from `test.rotational_mnist`

**Data loading and configuration** (`test/rotational_mnist/`):
- `mnist_loader.py` - Complete MNIST data loading module with PyTorch integration
- `config.py` - Pydantic-based configuration manager with hierarchical merging
- `config/config-default.yml` - Default hyperparameters (checked into git)
- `config/config.yml` - Local overrides (git-ignored, optional)

**Entry point** (`scripts/rotational_mnist/train.py`):
- Orchestrator that imports case-specific trainers and aggregates results
- Loads configuration and coordinates training of specified models
- Saves per-model results to: `training_results.baseline.json`, `training_results.augmented.json`, `training_results.equivariant.json`
- Also saves combined results to `training_results.json` for reference

**Results tracking**:
- `TrainingTracker` records per-epoch training loss, validation accuracy, and elapsed time
- Results saved to individual JSON files per model, plus combined file for easy comparison
- File naming: `models/training_results.{baseline|augmented|equivariant}.json`

## Makefile Targets

**IMPORTANT**: Every Python entry point script must have a corresponding Makefile target. This ensures consistent command structure and makes workflows discoverable.

All Makefile targets automatically use `poetry run` internally. Execute with:
```bash
# Training commands (see README.md for details)
make train_mnist           # Train all three models
make train_baseline        # Train baseline only
make train_augmented       # Train augmented only
make train_equivariant     # Train equivariant only

# Evaluation commands
make evaluate              # Evaluate all trained models on rotated test sets
make evaluate_models       # Alias for evaluate

# Test commands
make test_pytorch          # Run PyTorch/ESCNN tests (via poetry)
make watch_pytorch         # Watch and re-run on file changes (via poetry)
make test                  # Run Idris tests (full suite)
make test_train            # Run training unit tests
```

**Entry Point Convention**:
- Every script in `scripts/` must have a Makefile target
- Entry points import and call `main()` from implementation modules
- Never call `poetry run python scripts/...` directly—use `make <target>` instead

Do NOT call Python directly. Always use either:
- `make <target>` for predefined workflows
- `poetry run python -m <module>` for module execution when no target exists
- `poetry run pytest` for pytest execution

## Common Development Workflows

### Loading MNIST for Training
```python
from test.rotational_mnist.mnist_loader import get_mnist_loaders

train_loader, test_loader = get_mnist_loaders(batch_size=32, num_workers=4)

for batch_idx, (images, labels) in enumerate(train_loader):
    # images shape: (32, 1, 28, 28)
    # labels shape: (32,)
    pass
```

### Testing Robustness on Rotated Data
```python
from test.rotational_mnist.mnist_loader import get_rotation_test_loaders

test_loaders = get_rotation_test_loaders(batch_size=32)
# Returns: {0: DataLoader, 15: DataLoader, 30: ..., 270: ...}

for angle, loader in test_loaders.items():
    accuracy = evaluate_model(model, loader)
    print(f"Accuracy at {angle}°: {accuracy:.2%}")
```

### Creating Augmented Training Data
```bash
# Generate training set with 4 rotations combined (240k images)
poetry run python test/utils/create_rotated_mnist.py augmented

# Generate test set at specific angle
poetry run python test/utils/create_rotated_mnist.py rotated 45
```

## Current Experimental Focus

See TODO.md for the rotational MNIST comparison project checklist and implementation status.

**IMPORTANT REMINDER**: After completing each phase/implementation:
1. Mark phase as complete in TODO.md
2. Update relevant sections in CLAUDE.md with new features/changes
3. Commit changes to git with clear message describing what was implemented
4. This ensures documentation stays current and progress is tracked

**GIT COMMIT NOTES**: Use clear, concise commit messages focused on the work. Avoid adding "Generated with Claude" and "Co-Authored-By: Claude" lines at the end of commit messages.

## Key Implementation Details

**Image Rotation**: Uses scipy.ndimage with PIL for interpolation
- Rotation angles: 0°, 15°, 30°, 45°, 60°, 90°, 180°, 270°
- Boundary handling: Reflect padding to avoid artifacts
- Data type: Clipped to uint8 range [0, 255]

**Progress Tracking**: Print every 10k samples during processing
```python
if (i + 1) % 10000 == 0:
    print(f"  Processed {i + 1}/{len(images)} images...")
```

**Memory Efficiency**: Process data in chunks when possible
- IDX files read into memory (manageable size: ~47MB for training images)
- PyTorch DataLoaders handle batching and prefetching

## Important Notes

**ENVIRONMENT MANAGEMENT**:
- The project uses `direnv` to automatically manage the Poetry virtual environment
- When you `cd` into the project, direnv automatically activates the correct environment
- NEVER run `python` directly—always use `poetry run python`
- NEVER run `pip` directly—always use `poetry run pip`
- The `.envrc` file ensures Python 3.13.9 from pyenv is always used
- All tools (pytest, jupyter, etc.) must be run via `poetry run`

**DEPENDENCIES**:
- The ESCNN library requires a fork of `lie-learn` (must be installed manually via Poetry)
- After `poetry install`, run: `poetry run pip install -e ~/src/forks/lie_learn`

**DATA FORMAT**:
- All data files are in binary IDX format for compatibility with Idris2
- MNIST loaders expect data in this format

**COMMAND SUMMARY**:
```bash
# ✓ CORRECT
poetry run python script.py
poetry run pytest tests/
make test_pytorch
poetry run jupyter notebook

# ✗ WRONG - Never do this
python script.py          # Missing poetry run
pip install package      # Missing poetry run
python -m pytest          # Missing poetry run
```
