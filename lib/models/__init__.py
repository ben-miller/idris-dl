"""Neural network models for rotational MNIST experiments."""

from .standard_cnn import StandardCNN
from .escnn_cnn import ESCNNCnn

__all__ = ["StandardCNN", "ESCNNCnn"]
