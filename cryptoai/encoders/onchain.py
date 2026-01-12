"""On-chain data encoder using Transformer architecture."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from cryptoai.encoders.base import (
    BaseEncoder,
    PositionalEncoding,
    TransformerBlock,
    GatedLinearUnit,
)


class OnChainEncoder(BaseEncoder):
    """
    Transformer encoder for on-chain data.

    Captures temporal patterns in exchange flows, whale activity,
    and network metrics.
    """

    def __init__(
        self,
        input_dim: int = 33,  # On-chain feature dimension
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        sequence_length: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__(input_dim, output_dim, dropout)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, sequence_length, dropout)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Gated aggregation
        self.gate = GatedLinearUnit(hidden_dim, hidden_dim * 2, dropout)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Auxiliary heads
        self.aux_flow_head = nn.Linear(output_dim, 1)  # Net flow direction
        self.aux_accumulation_head = nn.Linear(output_dim, 1)  # Accumulation signal

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, input_dim)
            mask: Optional attention mask
            return_aux: Return auxiliary predictions

        Returns:
            Encoded on-chain state
        """
        B, T, _ = x.shape

        # Project and add positional encoding
        x = self.input_proj(x)
        x = self.pos_encoding(x)

        # Transformer encoding
        for block in self.transformer_blocks:
            x = block(x, mask)

        # Take last hidden state
        x = x[:, -1, :]

        # Gated processing
        x = self.gate(x)

        # Output projection
        output = self.output_proj(x)

        if return_aux:
            aux_outputs = {
                "net_flow": torch.tanh(self.aux_flow_head(output)),
                "accumulation": torch.sigmoid(self.aux_accumulation_head(output)),
            }
            return output, aux_outputs

        return output


class WhaleActivityEncoder(BaseEncoder):
    """
    Specialized encoder for whale wallet activity.

    Tracks smart money movements and accumulation patterns.
    """

    def __init__(
        self,
        input_dim: int = 11,  # Whale features
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_whales: int = 100,  # Max tracked whales
        embedding_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__(input_dim, output_dim, dropout)

        self.num_whales = num_whales

        # Whale address embeddings (for known whales)
        self.whale_embeddings = nn.Embedding(num_whales + 1, embedding_dim)  # +1 for unknown

        # Activity encoder
        self.activity_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Attention over whale activities
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )

        # Output
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Smart money signal head
        self.smart_money_head = nn.Linear(output_dim, 1)

    def forward(
        self,
        activities: torch.Tensor,
        whale_ids: Optional[torch.Tensor] = None,
        return_smart_signal: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            activities: Activity features (batch, num_activities, input_dim)
            whale_ids: Optional whale IDs (batch, num_activities)
            return_smart_signal: Return smart money signal

        Returns:
            Encoded whale activity
        """
        B, N, _ = activities.shape

        # Encode activities
        encoded = self.activity_encoder(activities)

        # Add whale embeddings if available
        if whale_ids is not None:
            whale_ids = whale_ids.clamp(0, self.num_whales)
            whale_emb = self.whale_embeddings(whale_ids)
            # Expand and add
            whale_emb = F.pad(whale_emb, (0, encoded.size(-1) - whale_emb.size(-1)))
            encoded = encoded + whale_emb

        # Self-attention over activities
        attended, _ = self.attention(encoded, encoded, encoded)

        # Mean pool
        output = attended.mean(dim=1)
        output = self.output_proj(output)

        if return_smart_signal:
            signal = torch.tanh(self.smart_money_head(output))
            return output, signal

        return output


class ExchangeFlowEncoder(BaseEncoder):
    """
    Specialized encoder for exchange flow patterns.

    Detects accumulation/distribution phases.
    """

    def __init__(
        self,
        input_dim: int = 12,  # Flow features
        num_exchanges: int = 10,
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__(input_dim, output_dim, dropout)

        self.num_exchanges = num_exchanges

        # Exchange embeddings
        self.exchange_embeddings = nn.Embedding(num_exchanges, 8)

        # Flow encoder per exchange
        self.flow_encoder = nn.Sequential(
            nn.Linear(input_dim + 8, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Cross-exchange transformer
        self.cross_exchange = TransformerBlock(
            hidden_dim, num_heads=4, dropout=dropout
        )

        # Output
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Auxiliary heads
        self.pressure_head = nn.Linear(output_dim, 1)  # Selling pressure

    def forward(
        self,
        flows: torch.Tensor,
        exchange_ids: Optional[torch.Tensor] = None,
        return_pressure: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            flows: Flow features (batch, num_exchanges, input_dim)
            exchange_ids: Exchange IDs (batch, num_exchanges)
            return_pressure: Return selling pressure estimate

        Returns:
            Encoded flow state
        """
        B, E, _ = flows.shape

        # Get exchange embeddings
        if exchange_ids is None:
            exchange_ids = torch.arange(E, device=flows.device).unsqueeze(0).expand(B, -1)

        exchange_ids = exchange_ids.clamp(0, self.num_exchanges - 1)
        ex_emb = self.exchange_embeddings(exchange_ids)

        # Concatenate and encode
        combined = torch.cat([flows, ex_emb], dim=-1)
        encoded = self.flow_encoder(combined)

        # Cross-exchange attention
        attended = self.cross_exchange(encoded)

        # Pool
        output = attended.mean(dim=1)
        output = self.output_proj(output)

        if return_pressure:
            pressure = torch.sigmoid(self.pressure_head(output))
            return output, pressure

        return output
