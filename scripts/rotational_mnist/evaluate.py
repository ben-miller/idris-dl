#!/usr/bin/env python3
"""
Evaluation entry point for rotational MNIST models.

Evaluates three trained models on rotated MNIST test sets:
- Case 1: Standard CNN trained on upright MNIST only
- Case 2: Standard CNN trained on augmented dataset (all rotations)
- Case 3: ESCNN trained on upright MNIST only

Generates accuracy comparison table and visualization across rotation angles.

Usage:
    poetry run python scripts/rotational_mnist/evaluate.py
    poetry run python scripts/rotational_mnist/evaluate.py --model-dir models --output-dir results
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from test.rotational_mnist.evaluate import main

if __name__ == "__main__":
    main()
