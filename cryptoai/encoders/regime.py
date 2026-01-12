"""Regime encoder for market state classification."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from cryptoai.encoders.base import BaseEncoder, TransformerBlock, GatedLinearUnit


class RegimeEncoder(BaseEncoder):
    """
    Encoder for market regime classification.

    Identifies:
    - Trending up/down
    - Ranging/consolidation
    - High volatility
    - Crisis/black swan conditions
    """

    def __init__(
        self,
        input_dim: int = 64,  # Combined state dimension
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_regimes: int = 5,
        sequence_length: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__(input_dim, output_dim, dropout)

        self.num_regimes = num_regimes

        # Regime names for interpretability
        self.regime_names = [
            "trending_up",
            "trending_down",
            "ranging",
            "volatile",
            "crisis",
        ]

        # Input processing
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Temporal processing
        self.temporal = nn.GRU(
            hidden_dim, hidden_dim, num_layers=2,
            batch_first=True, dropout=dropout
        )

        # Regime classification head
        self.regime_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_regimes),
        )

        # Regime transition model
        self.transition_matrix = nn.Parameter(
            torch.eye(num_regimes) * 0.8 + torch.ones(num_regimes, num_regimes) * 0.04
        )

        # Regime-aware feature transform
        self.regime_transforms = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim)
            for _ in range(num_regimes)
        ])

        # Regime embedding
        self.regime_embedding = nn.Embedding(num_regimes, output_dim)

        # Volatility estimation
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        x: torch.Tensor,
        prev_regime: Optional[torch.Tensor] = None,
        return_probs: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features (batch, seq_len, input_dim) or (batch, input_dim)
            prev_regime: Previous regime (batch,) for transition smoothing
            return_probs: Return regime probabilities

        Returns:
            Regime-aware encoding
        """
        # Handle single timestep input
        if x.dim() == 2:
            x = x.unsqueeze(1)

        B, T, _ = x.shape

        # Project input
        x = self.input_proj(x)

        # Temporal processing
        temporal_out, h_n = self.temporal(x)
        last_hidden = h_n[-1]  # (B, hidden)

        # Regime classification
        regime_logits = self.regime_classifier(last_hidden)

        # Apply transition matrix smoothing if previous regime known
        if prev_regime is not None:
            transition_probs = F.softmax(self.transition_matrix[prev_regime], dim=-1)
            regime_logits = regime_logits + torch.log(transition_probs + 1e-8)

        regime_probs = F.softmax(regime_logits, dim=-1)

        # Soft mixture of regime-specific transforms
        regime_outputs = []
        for i, transform in enumerate(self.regime_transforms):
            regime_out = transform(last_hidden)
            regime_outputs.append(regime_out)

        regime_outputs = torch.stack(regime_outputs, dim=1)  # (B, num_regimes, output_dim)
        output = torch.einsum("bn,bnd->bd", regime_probs, regime_outputs)

        # Add regime embedding
        regime_idx = regime_probs.argmax(dim=-1)
        regime_emb = self.regime_embedding(regime_idx)
        output = output + regime_emb

        if return_probs:
            volatility = self.volatility_head(last_hidden)
            return output, regime_probs, volatility

        return output

    def get_regime(
        self,
        x: torch.Tensor,
        return_confidence: bool = False,
    ) -> torch.Tensor:
        """
        Get predicted regime.

        Args:
            x: Input features
            return_confidence: Return confidence score

        Returns:
            Regime index (batch,)
        """
        output, probs, _ = self.forward(x, return_probs=True)
        regime = probs.argmax(dim=-1)

        if return_confidence:
            confidence = probs.max(dim=-1).values
            return regime, confidence

        return regime

    def get_regime_name(self, regime_idx: int) -> str:
        """Get human-readable regime name."""
        return self.regime_names[regime_idx]


class VolatilityRegimeEncoder(BaseEncoder):
    """
    Specialized encoder for volatility regime.

    Captures volatility clustering and regime changes.
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_vol_regimes: int = 3,  # low, medium, high
        dropout: float = 0.1,
    ):
        super().__init__(input_dim, output_dim, dropout)

        self.num_vol_regimes = num_vol_regimes

        # Volatility encoder
        self.vol_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # GARCH-like component
        self.garch_linear = nn.Linear(hidden_dim, hidden_dim)

        # Regime classification
        self.regime_classifier = nn.Linear(hidden_dim, num_vol_regimes)

        # Output
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Regime embeddings
        self.regime_embedding = nn.Embedding(num_vol_regimes, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        prev_volatility: Optional[torch.Tensor] = None,
        return_regime: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Volatility features (batch, seq_len, input_dim)
            prev_volatility: Previous volatility estimate (batch,)
            return_regime: Return volatility regime

        Returns:
            Volatility regime encoding
        """
        # Encode
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Take last timestep
        x_last = x[:, -1, :]
        encoded = self.vol_encoder(x_last)

        # GARCH-like persistence
        if prev_volatility is not None:
            prev_vol_expanded = prev_volatility.unsqueeze(-1).expand(-1, encoded.size(-1))
            encoded = encoded + 0.1 * self.garch_linear(prev_vol_expanded)

        # Regime
        regime_logits = self.regime_classifier(encoded)
        regime_probs = F.softmax(regime_logits, dim=-1)
        regime = regime_probs.argmax(dim=-1)

        # Output with regime embedding
        output = self.output_proj(encoded)
        output = output + self.regime_embedding(regime)

        if return_regime:
            return output, regime_probs

        return output


class TrendRegimeEncoder(BaseEncoder):
    """
    Specialized encoder for trend regime detection.

    Identifies trending vs mean-reverting markets.
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__(input_dim, output_dim, dropout)

        # Trend detection network
        self.trend_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Directional component
        self.direction_head = nn.Linear(hidden_dim, 1)  # -1 to 1

        # Strength component
        self.strength_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Output
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_trend: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Price/momentum features (batch, seq_len, input_dim)
            return_trend: Return trend direction and strength

        Returns:
            Trend regime encoding
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = x[:, -1, :]
        encoded = self.trend_encoder(x)

        output = self.output_proj(encoded)

        if return_trend:
            direction = torch.tanh(self.direction_head(encoded))
            strength = self.strength_head(encoded)
            return output, direction, strength

        return output
