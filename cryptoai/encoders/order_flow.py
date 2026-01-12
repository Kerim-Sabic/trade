"""Order flow encoder using Transformer architecture."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from cryptoai.encoders.base import (
    BaseEncoder,
    PositionalEncoding,
    TransformerBlock,
    create_causal_mask,
)


class OrderFlowEncoder(BaseEncoder):
    """
    Transformer-based encoder for order flow data.

    Encodes tick-level trades and order book dynamics into
    a learned representation.
    """

    def __init__(
        self,
        input_dim: int = 40,  # Microstructure feature dimension
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        sequence_length: int = 100,
        dropout: float = 0.1,
        use_causal_mask: bool = True,
    ):
        super().__init__(input_dim, output_dim, dropout)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.sequence_length = sequence_length
        self.use_causal_mask = use_causal_mask

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, sequence_length, dropout)

        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Temporal aggregation
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Auxiliary heads for multi-task learning
        self.aux_imbalance_head = nn.Linear(output_dim, 1)  # Predict imbalance
        self.aux_urgency_head = nn.Linear(output_dim, 1)  # Predict urgency
        self.aux_toxicity_head = nn.Linear(output_dim, 1)  # Predict toxicity

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            mask: Optional attention mask
            return_aux: Return auxiliary predictions

        Returns:
            Encoded representation of shape (batch, output_dim)
            Optionally, auxiliary predictions
        """
        B, T, _ = x.shape

        # Project input
        x = self.input_proj(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Create causal mask if needed
        if self.use_causal_mask and mask is None:
            mask = create_causal_mask(T, x.device)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)

        # Temporal aggregation using learned query
        query = repeat(self.query, "1 1 d -> b 1 d", b=B)
        aggregated, _ = self.temporal_attention(query, x, x)
        aggregated = aggregated.squeeze(1)

        # Output projection
        output = self.output_proj(aggregated)

        if return_aux:
            aux_outputs = {
                "imbalance": torch.tanh(self.aux_imbalance_head(output)),
                "urgency": torch.sigmoid(self.aux_urgency_head(output)),
                "toxicity": torch.sigmoid(self.aux_toxicity_head(output)),
            }
            return output, aux_outputs

        return output

    def encode_sequence(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode and return full sequence representation.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Sequence of encoded states (batch, seq_len, hidden_dim)
        """
        B, T, _ = x.shape

        # Project input
        x = self.input_proj(x)
        x = self.pos_encoding(x)

        if self.use_causal_mask and mask is None:
            mask = create_causal_mask(T, x.device)

        for block in self.transformer_blocks:
            x = block(x, mask)

        return x


class MultiScaleOrderFlowEncoder(BaseEncoder):
    """
    Multi-scale order flow encoder that captures patterns at different time scales.
    """

    def __init__(
        self,
        input_dim: int = 40,
        hidden_dim: int = 256,
        output_dim: int = 128,
        scales: list = [10, 50, 100],  # Different lookback windows
        dropout: float = 0.1,
    ):
        super().__init__(input_dim, output_dim, dropout)

        self.scales = scales

        # Encoder for each scale
        self.scale_encoders = nn.ModuleList([
            OrderFlowEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim // len(scales),
                output_dim=output_dim // len(scales),
                num_layers=2,
                sequence_length=scale,
                dropout=dropout,
            )
            for scale in scales
        ])

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-scale encoding.

        Args:
            x: Input tensor of shape (batch, max_seq_len, input_dim)

        Returns:
            Multi-scale encoded representation
        """
        scale_outputs = []

        for scale, encoder in zip(self.scales, self.scale_encoders):
            # Take last 'scale' timesteps
            x_scale = x[:, -scale:, :]
            encoded = encoder(x_scale)
            scale_outputs.append(encoded)

        # Concatenate scale outputs
        combined = torch.cat(scale_outputs, dim=-1)

        # Fuse
        output = self.fusion(combined)

        return output
