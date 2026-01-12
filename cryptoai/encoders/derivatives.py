"""Derivatives market encoder using LSTM architecture."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from cryptoai.encoders.base import BaseEncoder, GatedLinearUnit


class DerivativesEncoder(BaseEncoder):
    """
    Bidirectional LSTM encoder for derivatives market data.

    Captures temporal dependencies in funding rates, open interest,
    and liquidation patterns.
    """

    def __init__(
        self,
        input_dim: int = 42,  # Derivatives feature dimension
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__(input_dim, output_dim, dropout)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input projection with feature normalization
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention mechanism for temporal aggregation
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Output projection
        lstm_output_dim = hidden_dim * self.num_directions
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Auxiliary prediction heads
        self.aux_funding_head = nn.Linear(output_dim, 1)  # Predict funding direction
        self.aux_oi_head = nn.Linear(output_dim, 1)  # Predict OI change
        self.aux_cascade_head = nn.Linear(output_dim, 1)  # Predict liquidation cascade

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            lengths: Sequence lengths for packed padding (optional)
            return_aux: Return auxiliary predictions

        Returns:
            Encoded representation of shape (batch, output_dim)
        """
        B, T, _ = x.shape

        # Project input
        x = self.input_proj(x)

        # Pack if lengths provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Unpack if needed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=T
            )

        # Attention-based aggregation
        attn_weights = self.attention(lstm_out)  # (B, T, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (B, hidden*2)

        # Output projection
        output = self.output_proj(context)

        if return_aux:
            aux_outputs = {
                "funding_direction": torch.tanh(self.aux_funding_head(output)),
                "oi_change": torch.tanh(self.aux_oi_head(output)),
                "cascade_prob": torch.sigmoid(self.aux_cascade_head(output)),
            }
            return output, aux_outputs

        return output

    def encode_sequence(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode and return full sequence with hidden states.

        Returns:
            Tuple of (sequence_output, (h_n, c_n))
        """
        x = self.input_proj(x)
        return self.lstm(x)


class FundingRateEncoder(BaseEncoder):
    """
    Specialized encoder for funding rate dynamics.

    Captures funding rate regimes across exchanges.
    """

    def __init__(
        self,
        num_exchanges: int = 3,
        hidden_dim: int = 32,
        output_dim: int = 16,
        sequence_length: int = 21,  # 7 days of 8h funding
        dropout: float = 0.1,
    ):
        super().__init__(num_exchanges, output_dim, dropout)

        self.num_exchanges = num_exchanges
        self.sequence_length = sequence_length

        # Per-exchange GRU
        self.exchange_encoders = nn.ModuleList([
            nn.GRU(1, hidden_dim, batch_first=True)
            for _ in range(num_exchanges)
        ])

        # Cross-exchange attention
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )

        # Output
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * num_exchanges, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Funding rates (batch, seq_len, num_exchanges)

        Returns:
            Encoded funding dynamics
        """
        B, T, E = x.shape

        # Encode each exchange
        exchange_outputs = []
        for i in range(E):
            ex_input = x[:, :, i:i+1]  # (B, T, 1)
            _, h_n = self.exchange_encoders[i](ex_input)
            exchange_outputs.append(h_n.squeeze(0))  # (B, hidden)

        # Stack for attention
        stacked = torch.stack(exchange_outputs, dim=1)  # (B, E, hidden)

        # Cross-exchange attention
        attended, _ = self.cross_attention(stacked, stacked, stacked)

        # Flatten and project
        output = self.output(attended.reshape(B, -1))

        return output


class LiquidationEncoder(BaseEncoder):
    """
    Specialized encoder for liquidation data.

    Detects cascade patterns and stress signals.
    """

    def __init__(
        self,
        input_dim: int = 14,  # Liquidation features
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__(input_dim, output_dim, dropout)

        # Temporal convolutions for pattern detection
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3)

        # Combine multi-scale features
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Cascade probability head
        self.cascade_head = nn.Linear(output_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        return_cascade_prob: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input (batch, seq_len, input_dim)
            return_cascade_prob: Return cascade probability

        Returns:
            Encoded liquidation state
        """
        # Reshape for conv1d: (B, C, T)
        x = rearrange(x, "b t c -> b c t")

        # Multi-scale convolutions
        c1 = F.gelu(self.conv1(x))
        c2 = F.gelu(self.conv2(x))
        c3 = F.gelu(self.conv3(x))

        # Global max pool
        p1 = F.adaptive_max_pool1d(c1, 1).squeeze(-1)
        p2 = F.adaptive_max_pool1d(c2, 1).squeeze(-1)
        p3 = F.adaptive_max_pool1d(c3, 1).squeeze(-1)

        # Combine
        combined = torch.cat([p1, p2, p3], dim=-1)
        output = self.combine(combined)

        if return_cascade_prob:
            prob = torch.sigmoid(self.cascade_head(output))
            return output, prob

        return output
