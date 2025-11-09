#!/usr/bin/env python3

"""
Basic PyTorch hello world test for ESCNN experiments.
Tests basic tensor operations and model creation with ESCNN concepts.
"""

import torch
import torch.nn as nn

# Import escnn
from escnn import gspaces, nn as escnn_nn


def test_pytorch_version():
    """Test that PyTorch is installed and get version info."""
    print(f"PyTorch version: {torch.__version__}")
    assert torch.__version__ is not None, "PyTorch should be installed"
    print("✓ PyTorch version check passed")


def test_basic_tensor_operations():
    """Test basic PyTorch tensor operations."""
    # Create tensors
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])

    # Basic operations
    z = x + y
    expected = torch.tensor([5.0, 7.0, 9.0])

    assert torch.allclose(z, expected), "Tensor addition failed"
    print(f"✓ Tensor addition: {x.tolist()} + {y.tolist()} = {z.tolist()}")

    # Matrix multiplication
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    C = torch.matmul(A, B)

    expected_C = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
    assert torch.allclose(C, expected_C), "Matrix multiplication failed"
    print(f"✓ Matrix multiplication passed")


def test_simple_model():
    """Test creating and running a simple neural network."""
    # Define a simple model
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(20, 5)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    # Create model and test it
    model = SimpleNet()
    print(f"✓ Model created: {model.__class__.__name__}")

    # Create dummy input
    input_data = torch.randn(32, 10)  # Batch size 32, input dimension 10
    output = model(input_data)

    assert output.shape == (32, 5), f"Expected output shape (32, 5), got {output.shape}"
    print(f"✓ Model forward pass successful: input shape {input_data.shape} -> output shape {output.shape}")


def test_gradient_computation():
    """Test automatic differentiation."""
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    y = x ** 2
    z = y.sum()

    z.backward()

    # Gradient of x^2 is 2x
    expected_grad = torch.tensor([4.0, 6.0])
    assert torch.allclose(x.grad, expected_grad), "Gradient computation failed"
    print(f"✓ Automatic differentiation: gradient of {x.tolist()} = {x.grad.tolist()}")


def test_device_availability():
    """Test device availability (CPU and GPU if available)."""
    # CPU
    x_cpu = torch.tensor([1.0, 2.0, 3.0])
    print(f"✓ CPU tensor created: {x_cpu.device}")

    # GPU (if available)
    if torch.cuda.is_available():
        x_gpu = torch.tensor([1.0, 2.0, 3.0]).cuda()
        print(f"✓ GPU tensor created: {x_gpu.device}")
    else:
        print("✓ GPU not available (using CPU only)")


def test_escnn_gspaces():
    """Test ESCNN geometric spaces (gspaces)."""
    # Create a simple group: rotation group SO(2)
    so2 = gspaces.rot2dOnR2()
    print(f"✓ Created SO(2) group space: {so2}")

    # Create a simple ESCNN feature map type
    feat_type = escnn_nn.FieldType(so2, 4 * [so2.trivial_repr])
    print(f"✓ Created ESCNN FieldType with 4 trivial representations")

    # Create an ESCNN convolutional layer (minimal example)
    layer = escnn_nn.R2Conv(
        feat_type,  # input feature type
        feat_type,  # output feature type
        kernel_size=3,
        padding=1,
        bias=True
    )
    print(f"✓ Created ESCNN R2Conv layer (equivariant convolution)")

    # Test forward pass with dummy input wrapped in GeometricTensor
    dummy_input_tensor = torch.randn(2, 4, 8, 8)  # batch_size=2, channels=4, H=8, W=8
    dummy_input = escnn_nn.GeometricTensor(dummy_input_tensor, feat_type)
    output = layer(dummy_input)
    print(f"✓ Forward pass successful: input shape {dummy_input_tensor.shape} -> output shape {output.tensor.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch Hello World Test Suite")
    print("=" * 60)

    test_pytorch_version()
    print()

    test_basic_tensor_operations()
    print()

    test_simple_model()
    print()

    test_gradient_computation()
    print()

    test_device_availability()
    print()

    test_escnn_gspaces()
    print()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
