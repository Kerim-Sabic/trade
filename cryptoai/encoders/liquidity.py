"""Liquidity dynamics encoder using CNN architecture."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from cryptoai.encoders.base import BaseEncoder, ConvBlock


class LiquidityEncoder(BaseEncoder):
    """
    CNN-based encoder for liquidity/order book dynamics.

    Captures spatial patterns in order book depth and temporal
    evolution of liquidity.
    """

    def __init__(
        self,
        input_dim: int = 20,  # Order book features per level
        num_levels: int = 20,  # Order book depth
        hidden_channels: list = [32, 64, 128],
        kernel_sizes: list = [3, 5, 7],
        output_dim: int = 128,
        sequence_length: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__(input_dim * num_levels, output_dim, dropout)

        self.num_levels = num_levels
        self.hidden_channels = hidden_channels
        self.kernel_sizes = kernel_sizes

        # Spatial encoder (across order book levels)
        self.spatial_encoder = nn.ModuleList()
        in_ch = input_dim

        for out_ch, ks in zip(hidden_channels, kernel_sizes):
            self.spatial_encoder.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, ks, padding=ks // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            in_ch = out_ch

        # Temporal encoder (across time)
        self.temporal_encoder = nn.ModuleList()
        in_ch = hidden_channels[-1]

        for i, out_ch in enumerate(hidden_channels):
            self.temporal_encoder.append(
                ConvBlock(in_ch, out_ch, kernel_size=3, dropout=dropout)
            )
            in_ch = out_ch

        # Pooling and projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.projection = nn.Sequential(
            nn.Linear(hidden_channels[-1] * 2, hidden_channels[-1]),
            nn.LayerNorm(hidden_channels[-1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels[-1], output_dim),
        )

        # Auxiliary heads
        self.aux_spread_head = nn.Linear(output_dim, 1)
        self.aux_depth_head = nn.Linear(output_dim, 1)
        self.aux_imbalance_head = nn.Linear(output_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, num_levels, features)
               or (batch, seq_len, num_levels * features)
            return_aux: Return auxiliary predictions

        Returns:
            Encoded representation of shape (batch, output_dim)
        """
        B = x.shape[0]

        # Reshape if needed
        if x.dim() == 3:
            # Shape: (batch, seq_len, num_levels * features)
            T = x.shape[1]
            x = x.view(B, T, self.num_levels, -1)

        B, T, L, F = x.shape

        # Spatial encoding (per timestep)
        # Reshape to (B*T, F, L) for Conv1d
        x_spatial = rearrange(x, "b t l f -> (b t) f l")

        for layer in self.spatial_encoder:
            x_spatial = layer(x_spatial)

        # Pool across levels
        x_spatial = F.adaptive_avg_pool1d(x_spatial, 1).squeeze(-1)

        # Reshape back to (B, T, C)
        x_spatial = rearrange(x_spatial, "(b t) c -> b c t", b=B)

        # Temporal encoding
        x_temporal = x_spatial
        for layer in self.temporal_encoder:
            x_temporal = layer(x_temporal)

        # Global pooling
        x_avg = self.global_pool(x_temporal).squeeze(-1)
        x_max = F.adaptive_max_pool1d(x_temporal, 1).squeeze(-1)

        # Concatenate and project
        x_combined = torch.cat([x_avg, x_max], dim=-1)
        output = self.projection(x_combined)

        if return_aux:
            aux_outputs = {
                "spread": F.softplus(self.aux_spread_head(output)),
                "depth": F.softplus(self.aux_depth_head(output)),
                "imbalance": torch.tanh(self.aux_imbalance_head(output)),
            }
            return output, aux_outputs

        return output


class OrderBookImageEncoder(BaseEncoder):
    """
    Treat order book as an image and use 2D convolutions.

    Rows: price levels, Columns: time steps
    Channels: bid depth, ask depth, etc.
    """

    def __init__(
        self,
        num_levels: int = 20,
        num_channels: int = 4,  # bid_qty, ask_qty, bid_price, ask_price
        hidden_channels: list = [32, 64, 128],
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__(num_levels * num_channels, output_dim, dropout)

        self.num_levels = num_levels
        self.num_channels = num_channels

        # 2D CNN
        layers = []
        in_ch = num_channels

        for out_ch in hidden_channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                nn.MaxPool2d(2, stride=2),
                nn.Dropout2d(dropout),
            ])
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)

        # Calculate output size (depends on pooling)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels[-1] * 16, hidden_channels[-1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels[-1], output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, levels, time)

        Returns:
            Encoded representation
        """
        x = self.conv(x)
        x = self.adaptive_pool(x)
        x = self.fc(x)
        return x


class LiquidityElasticityEncoder(BaseEncoder):
    """
    Specialized encoder for liquidity elasticity features.

    Estimates price impact from order book shape.
    """

    def __init__(
        self,
        num_levels: int = 20,
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__(num_levels * 2, output_dim, dropout)

        self.num_levels = num_levels

        # Process bid and ask sides separately
        self.bid_encoder = nn.Sequential(
            nn.Linear(num_levels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        self.ask_encoder = nn.Sequential(
            nn.Linear(num_levels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # Combine
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

        # Price impact prediction head
        self.impact_head = nn.Linear(output_dim, 2)  # buy impact, sell impact

    def forward(
        self,
        bids: torch.Tensor,
        asks: torch.Tensor,
        return_impact: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            bids: Bid depths (batch, num_levels)
            asks: Ask depths (batch, num_levels)
            return_impact: Return predicted price impact

        Returns:
            Liquidity elasticity encoding
        """
        bid_enc = self.bid_encoder(bids)
        ask_enc = self.ask_encoder(asks)

        combined = torch.cat([bid_enc, ask_enc], dim=-1)
        output = self.combine(combined)

        if return_impact:
            impact = F.softplus(self.impact_head(output))
            return output, impact

        return output
