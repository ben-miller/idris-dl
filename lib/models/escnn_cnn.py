"""Equivariant CNN model using ESCNN for rotational invariance."""

import torch
import torch.nn as nn
from escnn import gspaces
from escnn import nn as escnn_nn


class ESCNNCnn(nn.Module):
    """
    SO(2)-equivariant CNN using ESCNN (E(2)-equivariant Steerable CNNs).

    This model is equivariant to 2D rotations, meaning if the input rotates,
    the output feature maps rotate in the same way. This allows the model
    to learn rotational invariance without data augmentation.

    Architecture (matching capacity roughly to StandardCNN):
    - Equivariant conv (1 -> 16 regular, 16 irrep) -> ReLU -> MaxPool (2x2)
    - Equivariant conv (32 channels) -> ReLU -> MaxPool (2x2)
    - Equivariant conv (64 channels) -> ReLU
    - Average pool and flatten -> Dense 128 -> ReLU -> Dense 10

    Used for:
    - Case 3: Trained on upright MNIST only, but equivariant to rotations
    """

    def __init__(self) -> None:
        super().__init__()

        # Create SO(2) group space (rotations in 2D)
        self.r2_act = gspaces.rot2dOnR2()

        # Input field type: 1 trivial (scalar) channel
        in_type = escnn_nn.FieldType(self.r2_act, 1 * [self.r2_act.trivial_repr])

        # Layer 1: 1 -> 16 channels using regular representations (scalar equivariant)
        # For SO(2), we can use trivial reps (scalars) with equivariant convolutions
        # The equivariance comes from the equivariant convolution, not the representation
        out_type1 = escnn_nn.FieldType(
            self.r2_act,
            16 * [self.r2_act.trivial_repr]
        )
        self.conv1 = escnn_nn.R2Conv(
            in_type,
            out_type1,
            kernel_size=5,
            padding=2,
            bias=True
        )

        # Layer 2: 16 -> 32 channels
        in_type2 = out_type1
        out_type2 = escnn_nn.FieldType(
            self.r2_act,
            32 * [self.r2_act.trivial_repr]
        )
        self.conv2 = escnn_nn.R2Conv(
            in_type2,
            out_type2,
            kernel_size=5,
            padding=2,
            bias=True
        )

        # Layer 3: 32 -> 64 channels
        in_type3 = out_type2
        out_type3 = escnn_nn.FieldType(
            self.r2_act,
            64 * [self.r2_act.trivial_repr]
        )
        self.conv3 = escnn_nn.R2Conv(
            in_type3,
            out_type3,
            kernel_size=5,
            padding=2,
            bias=True
        )

        # Store output type for later use
        self.out_type3 = out_type3

        # Fully connected layers
        # After 2 max pooling layers: 28 -> 14 -> 7
        # Feature map size depends on irrep dimensions
        # Calculate dynamically using a dummy forward pass
        dummy_x = torch.randn(1, 1, 28, 28)
        dummy_in = escnn_nn.GeometricTensor(dummy_x, in_type)
        with torch.no_grad():
            dummy_out = self._forward_convs(dummy_in)
        fc_input_size = dummy_out.tensor.view(1, -1).shape[1]

        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, 10)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)

    def _forward_convs(self, x: escnn_nn.GeometricTensor) -> escnn_nn.GeometricTensor:
        """
        Forward pass through convolutional layers only.

        Args:
            x: Input GeometricTensor

        Returns:
            Output GeometricTensor after all conv layers
        """
        # First equivariant conv block
        x = self.conv1(x)
        x = escnn_nn.ReLU(x.type, inplace=True)(x)
        x = torch.nn.functional.max_pool2d(x.tensor, kernel_size=2, stride=2)
        x = escnn_nn.GeometricTensor(x, self.conv1.out_type)

        # Second equivariant conv block
        x = self.conv2(x)
        x = escnn_nn.ReLU(x.type, inplace=True)(x)
        x = torch.nn.functional.max_pool2d(x.tensor, kernel_size=2, stride=2)
        x = escnn_nn.GeometricTensor(x, self.conv2.out_type)

        # Third equivariant conv block
        x = self.conv3(x)
        x = escnn_nn.ReLU(x.type, inplace=True)(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the equivariant network.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            Output logits of shape (batch_size, 10)
        """
        # Wrap input in GeometricTensor (required by ESCNN)
        in_type = escnn_nn.FieldType(
            self.r2_act,
            1 * [self.r2_act.trivial_repr]
        )
        x = escnn_nn.GeometricTensor(x, in_type)

        # Forward through convolutional layers
        x = self._forward_convs(x)

        # Extract raw tensor and flatten
        x = x.tensor
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
