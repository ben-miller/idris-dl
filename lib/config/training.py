"""
Configuration management for training experiments.

Loads training hyperparameters from YAML config files with hierarchical merging.
Provides centralized, type-safe configuration using Pydantic.

Configuration files (both in config/ directory at project root):
1. config-default.yml - Default configuration (checked into git)
2. config.yml - Local overrides (git-ignored, optional)

Local config merges into defaults, so you only need to specify what you override.

Usage:
    from lib.config import TrainingConfig
    config = TrainingConfig.load()
    print(config.epochs)  # Type-safe access

Override with local config:
    # config/config.yml
    training:
      epochs: 10
      batch_size: 64

Configuration structure:
    training:
      epochs: 3          # Number of training epochs
      batch_size: 32     # Batch size for training
      learning_rate: 0.001  # Learning rate for optimizer
"""

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TrainingConfig(BaseModel):
    """Training hyperparameters with type validation and defaults."""

    epochs: int = Field(default=10, gt=0, description="Number of training epochs")
    batch_size: int = Field(default=32, gt=0, description="Batch size for training")
    learning_rate: float = Field(
        default=0.001, gt=0, description="Learning rate for optimizer"
    )

    class Config:
        """Pydantic config."""
        extra = "ignore"
        validate_default = True

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "TrainingConfig":
        """
        Load training configuration from YAML files.

        Merges config.yml (local overrides) into config-default.yml (defaults).
        Both files are in the config/ directory at project root.

        Args:
            config_path: Explicit path to config file. If None, uses default locations.

        Returns:
            TrainingConfig instance with validated values

        Raises:
            ValidationError: If config values fail validation (e.g., negative epochs)
        """
        # Project root is three levels up from this file (lib/config/training.py)
        project_root = Path(__file__).parent.parent.parent
        config_dir = project_root / "config"

        if config_path is not None:
            # Single explicit config file
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
        else:
            # Load defaults first
            default_file = config_dir / "config-default.yml"
            if not default_file.exists():
                logger.warning(f"Default config not found: {default_file}")
                data = {}
            else:
                try:
                    with open(default_file) as f:
                        data = yaml.safe_load(f) or {}
                    logger.debug(f"Loaded defaults from {default_file}")
                except yaml.YAMLError as e:
                    logger.warning(f"Failed to parse {default_file}: {e}")
                    data = {}

            # Merge local config if it exists
            local_file = config_dir / "config.yml"
            if local_file.exists():
                try:
                    with open(local_file) as f:
                        local_data = yaml.safe_load(f) or {}
                    # Deep merge: local overrides defaults
                    if "training" in local_data:
                        if "training" not in data:
                            data["training"] = {}
                        data["training"].update(local_data["training"])
                    logger.info(f"Loaded local config from {local_file}")
                except yaml.YAMLError as e:
                    logger.warning(f"Failed to parse {local_file}: {e}")

        # Extract training section
        training_data = data.get("training", {})

        # Let Pydantic handle type conversion and validation
        return cls.model_validate(training_data, from_attributes=False)
