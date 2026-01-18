"""
Multi-Resolution Candle Processing for 15min/1h/4h horizons.

Provides downsampling and feature aggregation for different time horizons.
Windows 11 Compatible.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from loguru import logger


@dataclass
class ResolutionConfig:
    """Configuration for a single resolution."""
    name: str  # e.g., "15m", "1h", "4h"
    minutes: int  # Duration in minutes
    lookback_bars: int  # Number of bars to keep in history
    features: List[str]  # Features to compute


@dataclass
class MultiResolutionConfig:
    """Configuration for multi-resolution processing."""
    base_resolution: str = "1m"  # Base data resolution
    resolutions: List[ResolutionConfig] = None
    feature_fusion: str = "concat"  # concat, attention, hierarchical
    downsampling_method: str = "ohlcv_aggregate"  # ohlcv_aggregate, last, mean

    def __post_init__(self):
        if self.resolutions is None:
            self.resolutions = [
                ResolutionConfig("1m", 1, 100, ["ohlcv", "returns", "volatility"]),
                ResolutionConfig("5m", 5, 100, ["ohlcv", "returns", "volatility"]),
                ResolutionConfig("15m", 15, 100, ["ohlcv", "returns", "volatility"]),
                ResolutionConfig("1h", 60, 100, ["ohlcv", "returns", "volatility"]),
                ResolutionConfig("4h", 240, 50, ["ohlcv", "returns", "volatility"]),
            ]


class OHLCVDownsampler:
    """Downsamples OHLCV data from base to target resolution."""

    def __init__(self, base_minutes: int, target_minutes: int):
        self.base_minutes = base_minutes
        self.target_minutes = target_minutes
        self.aggregation_factor = target_minutes // base_minutes

        if target_minutes % base_minutes != 0:
            raise ValueError(
                f"Target resolution ({target_minutes}m) must be divisible by "
                f"base resolution ({base_minutes}m)"
            )

    def downsample(self, ohlcv: np.ndarray) -> np.ndarray:
        """
        Downsample OHLCV data.

        Args:
            ohlcv: Array of shape (n_bars, 5) with [open, high, low, close, volume]

        Returns:
            Downsampled array of shape (n_bars // factor, 5)
        """
        n_bars = ohlcv.shape[0]
        n_complete = (n_bars // self.aggregation_factor) * self.aggregation_factor

        if n_complete == 0:
            return np.zeros((0, 5))

        # Reshape for aggregation
        ohlcv_trimmed = ohlcv[:n_complete]
        reshaped = ohlcv_trimmed.reshape(-1, self.aggregation_factor, 5)

        # Aggregate OHLCV
        result = np.zeros((reshaped.shape[0], 5))
        result[:, 0] = reshaped[:, 0, 0]  # Open: first
        result[:, 1] = reshaped[:, :, 1].max(axis=1)  # High: max
        result[:, 2] = reshaped[:, :, 2].min(axis=1)  # Low: min
        result[:, 3] = reshaped[:, -1, 3]  # Close: last
        result[:, 4] = reshaped[:, :, 4].sum(axis=1)  # Volume: sum

        return result


class MultiResolutionProcessor:
    """
    Processes market data at multiple time resolutions.

    Supports:
    - 1m, 5m, 15m, 1h, 4h candles
    - OHLCV aggregation
    - Feature computation at each resolution
    - Feature fusion across resolutions
    """

    def __init__(self, config: Optional[MultiResolutionConfig] = None):
        self.config = config or MultiResolutionConfig()
        self._downsamplers: Dict[str, OHLCVDownsampler] = {}
        self._buffers: Dict[str, np.ndarray] = {}

        # Initialize downsamplers
        base_minutes = self._parse_resolution(self.config.base_resolution)
        for res_cfg in self.config.resolutions:
            if res_cfg.minutes > base_minutes:
                self._downsamplers[res_cfg.name] = OHLCVDownsampler(
                    base_minutes, res_cfg.minutes
                )
            # Initialize buffer
            self._buffers[res_cfg.name] = np.zeros((0, 5))

    def _parse_resolution(self, res: str) -> int:
        """Parse resolution string to minutes."""
        if res.endswith("m"):
            return int(res[:-1])
        elif res.endswith("h"):
            return int(res[:-1]) * 60
        elif res.endswith("d"):
            return int(res[:-1]) * 1440
        else:
            return int(res)

    def update(self, ohlcv_1m: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Update with new 1-minute OHLCV data.

        Args:
            ohlcv_1m: Array of shape (n_bars, 5)

        Returns:
            Dict of resolution -> downsampled OHLCV
        """
        result = {}

        for res_cfg in self.config.resolutions:
            if res_cfg.name == self.config.base_resolution:
                # Base resolution - no downsampling
                result[res_cfg.name] = ohlcv_1m
                self._buffers[res_cfg.name] = ohlcv_1m
            else:
                # Downsample
                downsampler = self._downsamplers[res_cfg.name]
                downsampled = downsampler.downsample(ohlcv_1m)
                result[res_cfg.name] = downsampled
                self._buffers[res_cfg.name] = downsampled

        return result

    def compute_features(
        self,
        ohlcv_data: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Compute features for each resolution.

        Args:
            ohlcv_data: Dict of resolution -> OHLCV data

        Returns:
            Dict of resolution -> feature array
        """
        features = {}

        for res_name, ohlcv in ohlcv_data.items():
            if len(ohlcv) == 0:
                features[res_name] = np.zeros((0, 10))
                continue

            # Extract OHLCV components
            open_p = ohlcv[:, 0]
            high = ohlcv[:, 1]
            low = ohlcv[:, 2]
            close = ohlcv[:, 3]
            volume = ohlcv[:, 4]

            # Compute features
            returns = np.zeros_like(close)
            returns[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-8)

            log_returns = np.zeros_like(close)
            log_returns[1:] = np.log(close[1:] / (close[:-1] + 1e-8))

            # Volatility (rolling std of returns)
            volatility = np.zeros_like(close)
            for i in range(20, len(close)):
                volatility[i] = np.std(returns[i-20:i])

            # Range
            range_pct = (high - low) / (close + 1e-8)

            # Body size
            body = np.abs(close - open_p) / (close + 1e-8)

            # Upper/lower wicks
            upper_wick = (high - np.maximum(open_p, close)) / (close + 1e-8)
            lower_wick = (np.minimum(open_p, close) - low) / (close + 1e-8)

            # Volume relative to average
            vol_ma = np.zeros_like(volume)
            for i in range(20, len(volume)):
                vol_ma[i] = np.mean(volume[i-20:i])
            vol_relative = volume / (vol_ma + 1e-8)

            # Stack features
            res_features = np.stack([
                returns,
                log_returns,
                volatility,
                range_pct,
                body,
                upper_wick,
                lower_wick,
                vol_relative,
                close / (close[0] + 1e-8) - 1,  # Cumulative return
                np.log(volume + 1),  # Log volume
            ], axis=1)

            features[res_name] = res_features

        return features

    def get_latest_bars(
        self,
        lookback: int = 100,
    ) -> Dict[str, np.ndarray]:
        """Get the latest N bars for each resolution."""
        result = {}
        for res_name, buffer in self._buffers.items():
            if len(buffer) >= lookback:
                result[res_name] = buffer[-lookback:]
            else:
                result[res_name] = buffer
        return result


class MultiResolutionEncoder(nn.Module):
    """
    Neural encoder for multi-resolution features.

    Fuses features from multiple time resolutions using attention or concatenation.
    """

    def __init__(
        self,
        feature_dim: int = 10,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_resolutions: int = 5,
        fusion_method: str = "attention",
        sequence_length: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_resolutions = num_resolutions
        self.fusion_method = fusion_method

        # Per-resolution encoders
        self.resolution_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_resolutions)
        ])

        # Temporal encoding for each resolution
        self.temporal_encoders = nn.ModuleList([
            nn.GRU(
                hidden_dim,
                hidden_dim // 2,
                batch_first=True,
                bidirectional=True,
            )
            for _ in range(num_resolutions)
        ])

        # Fusion layers
        if fusion_method == "attention":
            self.fusion_attention = nn.MultiheadAttention(
                hidden_dim, num_heads=4, dropout=dropout, batch_first=True
            )
            self.fusion_proj = nn.Linear(hidden_dim, output_dim)
        elif fusion_method == "concat":
            self.fusion_proj = nn.Sequential(
                nn.Linear(hidden_dim * num_resolutions, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, output_dim),
            )
        elif fusion_method == "hierarchical":
            # Hierarchical fusion from fine to coarse
            self.fusion_layers = nn.ModuleList([
                nn.Linear(hidden_dim * 2, hidden_dim)
                for _ in range(num_resolutions - 1)
            ])
            self.fusion_proj = nn.Linear(hidden_dim, output_dim)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # Output normalization
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        multi_res_features: Dict[str, torch.Tensor],
        resolution_order: List[str] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            multi_res_features: Dict of resolution name -> features (batch, seq, feature_dim)
            resolution_order: Order of resolutions (fine to coarse)

        Returns:
            Fused encoding (batch, output_dim)
        """
        if resolution_order is None:
            resolution_order = ["1m", "5m", "15m", "1h", "4h"]

        # Encode each resolution
        encoded = []
        for i, res_name in enumerate(resolution_order):
            if res_name not in multi_res_features:
                continue

            features = multi_res_features[res_name]

            # Feature encoding
            x = self.resolution_encoders[i](features)

            # Temporal encoding
            x, _ = self.temporal_encoders[i](x)

            # Take last timestep
            x = x[:, -1, :]

            encoded.append(x)

        if len(encoded) == 0:
            raise ValueError("No valid resolution features provided")

        # Pad to expected number of resolutions
        while len(encoded) < self.num_resolutions:
            encoded.append(torch.zeros_like(encoded[0]))

        # Fusion
        if self.fusion_method == "attention":
            # Stack and apply attention
            stacked = torch.stack(encoded, dim=1)  # (batch, num_res, hidden)
            attended, _ = self.fusion_attention(stacked, stacked, stacked)
            fused = attended.mean(dim=1)
            output = self.fusion_proj(fused)

        elif self.fusion_method == "concat":
            # Concatenate all encodings
            concat = torch.cat(encoded, dim=-1)
            output = self.fusion_proj(concat)

        elif self.fusion_method == "hierarchical":
            # Hierarchical fusion from fine to coarse
            current = encoded[0]
            for i, layer in enumerate(self.fusion_layers):
                if i + 1 < len(encoded):
                    combined = torch.cat([current, encoded[i + 1]], dim=-1)
                    current = layer(combined)
            output = self.fusion_proj(current)

        return self.output_norm(output)


def create_multi_resolution_processor(
    config_dict: Optional[Dict] = None,
) -> MultiResolutionProcessor:
    """Factory function to create a multi-resolution processor."""
    if config_dict is None:
        return MultiResolutionProcessor()

    config = MultiResolutionConfig(
        base_resolution=config_dict.get("base_resolution", "1m"),
        feature_fusion=config_dict.get("feature_fusion", "concat"),
        downsampling_method=config_dict.get("downsampling_method", "ohlcv_aggregate"),
    )

    return MultiResolutionProcessor(config)
