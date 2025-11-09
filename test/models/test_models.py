"""Tests for neural network models."""

import torch
import pytest

from lib.models import StandardCNN, ESCNNCnn


class TestStandardCNN:
    """Test suite for StandardCNN model."""

    def test_model_creation(self) -> None:
        """Test that StandardCNN can be instantiated."""
        model = StandardCNN()
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass(self) -> None:
        """Test forward pass with correct input/output shapes."""
        model = StandardCNN()
        batch_size = 2
        x = torch.randn(batch_size, 1, 28, 28)

        output = model(x)

        assert output.shape == (batch_size, 10), f"Expected shape ({batch_size}, 10), got {output.shape}"

    def test_batch_sizes(self) -> None:
        """Test forward pass with various batch sizes."""
        model = StandardCNN()

        for batch_size in [1, 4, 32, 128]:
            x = torch.randn(batch_size, 1, 28, 28)
            output = model(x)
            assert output.shape == (batch_size, 10)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the network."""
        model = StandardCNN()
        x = torch.randn(4, 1, 28, 28, requires_grad=True)

        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_parameter_count(self) -> None:
        """Test that model has expected number of parameters."""
        model = StandardCNN()
        param_count = sum(p.numel() for p in model.parameters())

        # Rough check: should have ~1M parameters for this architecture
        assert 500000 < param_count < 2000000, f"Parameter count {param_count} outside expected range"


class TestESCNNCnn:
    """Test suite for ESCNNCnn model."""

    def test_model_creation(self) -> None:
        """Test that ESCNNCnn can be instantiated."""
        model = ESCNNCnn()
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass(self) -> None:
        """Test forward pass with correct input/output shapes."""
        model = ESCNNCnn()
        batch_size = 2
        x = torch.randn(batch_size, 1, 28, 28)

        output = model(x)

        assert output.shape == (batch_size, 10), f"Expected shape ({batch_size}, 10), got {output.shape}"

    def test_batch_sizes(self) -> None:
        """Test forward pass with various batch sizes."""
        model = ESCNNCnn()

        for batch_size in [1, 4, 32, 128]:
            x = torch.randn(batch_size, 1, 28, 28)
            output = model(x)
            assert output.shape == (batch_size, 10)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the network."""
        model = ESCNNCnn()
        x = torch.randn(4, 1, 28, 28, requires_grad=True)

        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_parameter_count(self) -> None:
        """Test that model has expected number of parameters."""
        model = ESCNNCnn()
        param_count = sum(p.numel() for p in model.parameters())

        # Rough check: equivariant models can have many parameters due to group structure
        assert 100000 < param_count, f"Parameter count {param_count} seems too low"
        # Just verify it has some parameters (range can be large due to irreps)
        assert param_count < 10000000, f"Parameter count {param_count} seems too high"


class TestModelComparison:
    """Compare properties of both models."""

    def test_same_output_shape(self) -> None:
        """Test that both models produce same output shape."""
        model1 = StandardCNN()
        model2 = ESCNNCnn()

        x = torch.randn(8, 1, 28, 28)

        out1 = model1(x)
        out2 = model2(x)

        assert out1.shape == out2.shape == (8, 10)

    def test_different_architecture(self) -> None:
        """Test that models have different architectures."""
        model1 = StandardCNN()
        model2 = ESCNNCnn()

        # Parameter counts should be different
        params1 = sum(p.numel() for p in model1.parameters())
        params2 = sum(p.numel() for p in model2.parameters())

        # They should have different architectures
        assert str(model1) != str(model2)
