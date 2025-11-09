#!/usr/bin/env python3
"""
Training pipeline for rotational MNIST comparison.

Trains models:
- baseline: Standard CNN on upright MNIST only
- augmented: Standard CNN on augmented dataset (all rotations)
- equivariant: ESCNN equivariant CNN on upright MNIST only

Loads configuration from config/config-default.yml and config/config.yml (if present).
Results saved to models/training_results.json.

Usage:
    poetry run python scripts/rotational_mnist/train.py                # Train all three
    poetry run python scripts/rotational_mnist/train.py baseline       # Train baseline only
    poetry run python scripts/rotational_mnist/train.py baseline augmented  # Train two
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.config import TrainingConfig
from test.experiments.rotational_mnist.train_baseline import train_standard_cnn_baseline
from test.experiments.rotational_mnist.train_augmented import train_standard_cnn_augmented
from test.experiments.rotational_mnist.train_equivariant import train_escnn_equivariant

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Fixed paths
DATA_DIR = "data"
OUTPUT_DIR = "models"


def main() -> None:
    """Train specified models."""
    parser = argparse.ArgumentParser(
        description="Train rotational MNIST models"
    )
    parser.add_argument(
        "models",
        nargs="*",
        default=["baseline", "augmented", "equivariant"],
        help="Models to train: baseline, augmented, equivariant (default: all three)",
    )
    args = parser.parse_args()

    config = TrainingConfig.load()

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Track all results
    all_results = {}

    # Log start
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Training for {config.epochs} epochs, batch size {config.batch_size}")
    logger.info(f"Models: {', '.join(args.models)}")

    if "baseline" in args.models:
        logger.info("=" * 60)
        logger.info("Training: Standard CNN on upright MNIST (baseline)")
        model1, tracker1 = train_standard_cnn_baseline(
            DATA_DIR, OUTPUT_DIR, config.epochs, config.batch_size
        )
        all_results["standard_cnn_baseline"] = tracker1.to_dict()

        # Save baseline results
        baseline_results_file = Path(OUTPUT_DIR) / "training_results.baseline.json"
        with open(baseline_results_file, "w") as f:
            json.dump(tracker1.to_dict(), f, indent=2)
        logger.info(f"Baseline training time: {tracker1.elapsed_time:.2f}s")
        logger.info(f"Baseline final accuracy: {tracker1.val_accuracies[-1]:.4f}")
        logger.info(f"Saved baseline results to {baseline_results_file}\n")

    if "augmented" in args.models:
        logger.info("=" * 60)
        logger.info("Training: Standard CNN on augmented MNIST")
        model2, tracker2 = train_standard_cnn_augmented(
            DATA_DIR, OUTPUT_DIR, config.epochs, config.batch_size
        )
        all_results["standard_cnn_augmented"] = tracker2.to_dict()

        # Save augmented results
        augmented_results_file = Path(OUTPUT_DIR) / "training_results.augmented.json"
        with open(augmented_results_file, "w") as f:
            json.dump(tracker2.to_dict(), f, indent=2)
        logger.info(f"Augmented training time: {tracker2.elapsed_time:.2f}s")
        logger.info(f"Augmented final accuracy: {tracker2.val_accuracies[-1]:.4f}")
        logger.info(f"Saved augmented results to {augmented_results_file}\n")

    if "equivariant" in args.models:
        logger.info("=" * 60)
        logger.info("Training: ESCNN equivariant CNN on upright MNIST")
        model3, tracker3 = train_escnn_equivariant(
            DATA_DIR, OUTPUT_DIR, config.epochs, config.batch_size
        )
        all_results["escnn_equivariant"] = tracker3.to_dict()

        # Save equivariant results
        equivariant_results_file = Path(OUTPUT_DIR) / "training_results.equivariant.json"
        with open(equivariant_results_file, "w") as f:
            json.dump(tracker3.to_dict(), f, indent=2)
        logger.info(f"Equivariant training time: {tracker3.elapsed_time:.2f}s")
        logger.info(f"Equivariant final accuracy: {tracker3.val_accuracies[-1]:.4f}")
        logger.info(f"Saved equivariant results to {equivariant_results_file}\n")

    # Also save combined results for reference
    results_file = Path(OUTPUT_DIR) / "training_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved combined results to {results_file}")


if __name__ == "__main__":
    main()
