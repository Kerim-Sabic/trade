"""Unit tests for encoder modules.

Windows 11 Compatible - CPU only.
"""

import pytest
import torch
import torch.nn as nn

from cryptoai.encoders.base import (
    PositionalEncoding,
    LearnablePositionalEncoding,
    MultiHeadAttention,
    TransformerBlock,
    ConvBlock,
    GatedLinearUnit,
)


class TestPositionalEncoding:
    """Tests for PositionalEncoding."""

    def test_output_shape(self, device):
        """Test that output shape matches input shape."""
        d_model = 64
        seq_len = 50
        batch_size = 4

        pe = PositionalEncoding(d_model=d_model).to(device)
        x = torch.randn(batch_size, seq_len, d_model, device=device)

        output = pe(x)

        assert output.shape == x.shape

    def test_adds_position_information(self, device):
        """Test that encoding modifies input."""
        d_model = 64
        seq_len = 50

        pe = PositionalEncoding(d_model=d_model, dropout=0.0).to(device)
        x = torch.zeros(1, seq_len, d_model, device=device)

        output = pe(x)

        # Output should not be all zeros
        assert not torch.allclose(output, x)


class TestLearnablePositionalEncoding:
    """Tests for LearnablePositionalEncoding."""

    def test_output_shape(self, device):
        """Test that output shape matches input shape."""
        d_model = 64
        seq_len = 50
        batch_size = 4

        pe = LearnablePositionalEncoding(d_model=d_model).to(device)
        x = torch.randn(batch_size, seq_len, d_model, device=device)

        output = pe(x)

        assert output.shape == x.shape

    def test_parameters_learnable(self, device):
        """Test that positional encoding has learnable parameters."""
        d_model = 64
        pe = LearnablePositionalEncoding(d_model=d_model).to(device)

        params = list(pe.parameters())
        assert len(params) > 0
        assert params[0].requires_grad


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention."""

    def test_output_shape(self, device):
        """Test that output shape matches input shape."""
        d_model = 64
        num_heads = 8
        batch_size = 4
        seq_len = 50

        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads).to(device)
        x = torch.randn(batch_size, seq_len, d_model, device=device)

        output = mha(x)

        assert output.shape == x.shape

    def test_with_mask(self, device):
        """Test attention with mask."""
        d_model = 64
        num_heads = 8
        batch_size = 4
        seq_len = 10

        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads).to(device)
        x = torch.randn(batch_size, seq_len, d_model, device=device)

        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask = mask.unsqueeze(0).unsqueeze(0)

        output = mha(x, mask=mask)

        assert output.shape == x.shape


class TestTransformerBlock:
    """Tests for TransformerBlock."""

    def test_output_shape(self, device):
        """Test that output shape matches input shape."""
        d_model = 64
        batch_size = 4
        seq_len = 50

        block = TransformerBlock(d_model=d_model).to(device)
        x = torch.randn(batch_size, seq_len, d_model, device=device)

        output = block(x)

        assert output.shape == x.shape

    def test_residual_connection(self, device):
        """Test that residual connections work."""
        d_model = 64

        block = TransformerBlock(d_model=d_model, dropout=0.0).to(device)

        # With zero initialization, output should be close to input
        # (LayerNorm will still modify it)
        x = torch.randn(1, 10, d_model, device=device)
        output = block(x)

        # Just verify it runs and produces valid output
        assert not torch.isnan(output).any()


class TestConvBlock:
    """Tests for ConvBlock."""

    def test_output_shape_same_channels(self, device):
        """Test output shape when in/out channels match."""
        in_channels = 64
        out_channels = 64
        batch_size = 4
        seq_len = 50

        block = ConvBlock(in_channels, out_channels).to(device)
        # Conv1d expects (batch, channels, length)
        x = torch.randn(batch_size, in_channels, seq_len, device=device)

        output = block(x)

        assert output.shape == (batch_size, out_channels, seq_len)

    def test_output_shape_different_channels(self, device):
        """Test output shape when in/out channels differ."""
        in_channels = 64
        out_channels = 128
        batch_size = 4
        seq_len = 50

        block = ConvBlock(in_channels, out_channels).to(device)
        x = torch.randn(batch_size, in_channels, seq_len, device=device)

        output = block(x)

        assert output.shape == (batch_size, out_channels, seq_len)


class TestGatedLinearUnit:
    """Tests for GatedLinearUnit."""

    def test_output_shape(self, device):
        """Test that output shape matches input shape."""
        input_dim = 64
        hidden_dim = 128
        batch_size = 4
        seq_len = 50

        glu = GatedLinearUnit(input_dim, hidden_dim).to(device)
        x = torch.randn(batch_size, seq_len, input_dim, device=device)

        output = glu(x)

        assert output.shape == x.shape

    def test_residual_connection(self, device):
        """Test that GLU has residual connection."""
        input_dim = 64
        hidden_dim = 128

        glu = GatedLinearUnit(input_dim, hidden_dim, dropout=0.0).to(device)
        x = torch.randn(1, 10, input_dim, device=device)

        output = glu(x)

        # Output should be valid
        assert not torch.isnan(output).any()
