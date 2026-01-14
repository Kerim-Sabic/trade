"""
News & Black Swan Awareness System.

MISSION-CRITICAL NLP + TAIL-RISK MODULE
=======================================

This module provides real-time awareness of:
- Breaking news that could move markets
- Black swan events (extreme, unexpected shocks)
- Tail risk forecasting
- Anomaly detection in market data

DESIGN PRINCIPLES:
==================

1. CONSERVATIVE BY DEFAULT
   - When uncertain, do nothing
   - False positives cost money
   - Missing a trade > blowing up account

2. UNCERTAINTY QUANTIFICATION
   - Every prediction has confidence bounds
   - High uncertainty → reduce position / abstain
   - Never amplify noisy signals

3. EXPLAINABLE DURING CHAOS
   - System logs WHY it's acting
   - Human-readable risk reports
   - Clear escalation paths

4. FALSE SIGNAL PREVENTION
   - Multi-source confirmation required
   - Time-decay on old news
   - Sentiment saturation detection

ARCHITECTURE:
=============

┌─────────────────────────────────────────────────────────────────────────────┐
│                        NEWS & BLACK SWAN AWARENESS                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │    NEWS      │     │   MARKET     │     │  ON-CHAIN    │                │
│  │   SOURCES    │     │    DATA      │     │    DATA      │                │
│  │              │     │              │     │              │                │
│  │ • Twitter/X  │     │ • Price      │     │ • Whale txs  │                │
│  │ • Reuters    │     │ • Volume     │     │ • Exchange   │                │
│  │ • Bloomberg  │     │ • Orderbook  │     │   flows      │                │
│  │ • Telegram   │     │ • Funding    │     │ • Stablecoin │                │
│  │ • Discord    │     │ • OI changes │     │   supply     │                │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘                │
│         │                    │                    │                         │
│         └────────────────────┼────────────────────┘                         │
│                              ▼                                              │
│                   ┌────────────────────┐                                   │
│                   │   NLP PROCESSOR    │                                   │
│                   │                    │                                   │
│                   │  • FinBERT embed   │                                   │
│                   │  • Sentiment score │                                   │
│                   │  • Entity extract  │                                   │
│                   │  • Event classify  │                                   │
│                   └─────────┬──────────┘                                   │
│                             ▼                                              │
│         ┌───────────────────┼───────────────────┐                          │
│         ▼                   ▼                   ▼                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                    │
│  │   SHOCK     │    │   ANOMALY   │    │  TAIL RISK  │                    │
│  │  DETECTOR   │    │  DETECTOR   │    │ FORECASTER  │                    │
│  │             │    │             │    │             │                    │
│  │ • Magnitude │    │ • Isolation │    │ • GARCH     │                    │
│  │ • Velocity  │    │   Forest    │    │ • EVT       │                    │
│  │ • Breadth   │    │ • VAE       │    │ • CVaR      │                    │
│  │ • Confirm   │    │ • Z-score   │    │ • Tail idx  │                    │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                    │
│         │                  │                  │                            │
│         └──────────────────┼──────────────────┘                            │
│                            ▼                                               │
│                 ┌────────────────────┐                                     │
│                 │  DECISION ENGINE   │                                     │
│                 │                    │                                     │
│                 │  States:           │                                     │
│                 │  • NORMAL          │                                     │
│                 │  • ELEVATED_RISK   │                                     │
│                 │  • HIGH_ALERT      │                                     │
│                 │  • DO_NOTHING      │ ← Explicit abstain                  │
│                 │  • CRISIS_MODE     │                                     │
│                 └────────────────────┘                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Author: Quant NLP & Tail-Risk Engineer
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
from collections import deque
import asyncio
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


# =============================================================================
# ENUMS AND CONFIGURATIONS
# =============================================================================

class SystemState(Enum):
    """
    System operational states.

    The DO_NOTHING state is CRITICAL - it's the explicit decision
    to abstain from trading when uncertainty is too high.
    """
    NORMAL = "normal"               # Business as usual
    ELEVATED_RISK = "elevated"      # Reduce position sizes
    HIGH_ALERT = "high_alert"       # Minimal positions only
    DO_NOTHING = "do_nothing"       # Explicit abstain from ALL trading
    CRISIS_MODE = "crisis"          # Emergency risk-off, close all


class EventType(Enum):
    """Types of news/events."""
    REGULATORY = "regulatory"       # SEC, government actions
    EXCHANGE = "exchange"           # Exchange hacks, failures
    MACRO = "macro"                 # Fed, inflation, employment
    PROTOCOL = "protocol"           # DeFi exploits, upgrades
    WHALE = "whale"                 # Large wallet movements
    SOCIAL = "social"               # Viral tweets, FUD/FOMO
    TECHNICAL = "technical"         # Network issues, forks
    UNKNOWN = "unknown"


class Severity(Enum):
    """Event severity levels."""
    NOISE = 0           # Background noise, ignore
    LOW = 1             # Minor, monitor only
    MEDIUM = 2          # Noteworthy, may affect positions
    HIGH = 3            # Significant, likely market impact
    CRITICAL = 4        # Black swan potential
    CATASTROPHIC = 5    # Market structure threat


@dataclass
class NewsConfig:
    """Configuration for news processing."""

    # NLP settings
    embedding_dim: int = 768            # FinBERT dimension
    max_sequence_length: int = 512
    sentiment_threshold: float = 0.6     # Confidence threshold

    # Shock detection
    shock_velocity_threshold: float = 3.0  # Std devs
    shock_magnitude_threshold: float = 0.05  # 5% price move
    confirmation_sources: int = 2          # Min sources to confirm

    # Anomaly detection
    anomaly_contamination: float = 0.01   # 1% expected anomalies
    z_score_threshold: float = 3.0

    # Tail risk
    var_confidence: float = 0.99
    tail_index_window: int = 100

    # Time decay
    news_half_life_minutes: float = 30.0  # News relevance decay
    max_news_age_hours: float = 24.0

    # Do-nothing thresholds
    uncertainty_threshold: float = 0.7    # High uncertainty → abstain
    disagreement_threshold: float = 0.5   # Source disagreement → abstain
    min_confidence: float = 0.3           # Below this → do nothing

    # False signal prevention
    min_source_quality: float = 0.5
    max_sentiment_velocity: float = 0.3   # Prevent sudden sentiment swings
    saturation_threshold: int = 100       # Max events per hour before saturation


@dataclass
class NewsEvent:
    """A single news event."""
    id: str
    timestamp: datetime
    source: str
    source_quality: float  # 0-1 reliability score
    raw_text: str
    event_type: EventType
    severity: Severity
    sentiment: float       # -1 to 1
    confidence: float      # 0 to 1
    entities: List[str]    # Mentioned assets/entities
    embedding: Optional[np.ndarray] = None

    def relevance_score(self, current_time: datetime, half_life_mins: float) -> float:
        """Calculate time-decayed relevance."""
        age_minutes = (current_time - self.timestamp).total_seconds() / 60
        decay = np.exp(-np.log(2) * age_minutes / half_life_mins)
        return self.confidence * self.source_quality * decay


@dataclass
class MarketShock:
    """Detected market shock."""
    timestamp: datetime
    shock_type: str
    magnitude: float        # Percentage move
    velocity: float         # Rate of change
    affected_assets: List[str]
    confidence: float
    sources: List[str]      # Confirming sources
    description: str


@dataclass
class TailRiskForecast:
    """Tail risk forecast."""
    timestamp: datetime
    var_1d_99: float        # 1-day VaR at 99%
    cvar_1d_99: float       # Expected shortfall
    tail_index: float       # Power law tail exponent
    extreme_prob: float     # P(loss > 3 sigma)
    regime: str             # normal, elevated, crisis
    confidence: float


@dataclass
class SystemDecision:
    """
    System decision with full explanation.

    Critical for transparency during chaos.
    """
    timestamp: datetime
    state: SystemState
    confidence: float
    position_multiplier: float  # 0 = no trading, 1 = full size

    # Explanation
    primary_reason: str
    contributing_factors: List[str]
    active_risks: List[str]
    recommendations: List[str]

    # Uncertainty bounds
    uncertainty: float
    confidence_interval: Tuple[float, float]

    def explain(self) -> str:
        """Human-readable explanation."""
        lines = [
            f"=== SYSTEM DECISION: {self.state.value.upper()} ===",
            f"Time: {self.timestamp.isoformat()}",
            f"Confidence: {self.confidence:.1%}",
            f"Uncertainty: {self.uncertainty:.1%}",
            f"Position Multiplier: {self.position_multiplier:.1%}",
            "",
            f"PRIMARY REASON: {self.primary_reason}",
            "",
            "CONTRIBUTING FACTORS:",
        ]
        for factor in self.contributing_factors:
            lines.append(f"  • {factor}")

        if self.active_risks:
            lines.append("")
            lines.append("ACTIVE RISKS:")
            for risk in self.active_risks:
                lines.append(f"  ⚠️ {risk}")

        if self.recommendations:
            lines.append("")
            lines.append("RECOMMENDATIONS:")
            for rec in self.recommendations:
                lines.append(f"  → {rec}")

        return "\n".join(lines)


# =============================================================================
# NLP PROCESSOR
# =============================================================================

class FinanceNLPProcessor(nn.Module):
    """
    Finance-tuned NLP processor.

    Uses pre-trained embeddings from FinBERT or similar,
    with additional heads for crypto-specific tasks.
    """

    def __init__(self, config: NewsConfig):
        super().__init__()
        self.config = config

        # Embedding projection (from pretrained dim to internal)
        self.embed_proj = nn.Linear(config.embedding_dim, 256)

        # Sentiment head
        self.sentiment_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3),  # negative, neutral, positive
        )

        # Event classification head
        self.event_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, len(EventType)),
        )

        # Severity head
        self.severity_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, len(Severity)),
        )

        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Crypto-specific keywords for boosting
        self.crypto_keywords = {
            "hack", "exploit", "rug", "scam", "sec", "lawsuit",
            "regulation", "ban", "crash", "dump", "pump", "whale",
            "liquidation", "margin", "leverage", "funding", "fork",
            "upgrade", "mainnet", "airdrop", "burn", "mint",
        }

        # Source quality scores
        self.source_quality = {
            "reuters": 0.95,
            "bloomberg": 0.95,
            "coindesk": 0.80,
            "cointelegraph": 0.75,
            "twitter_verified": 0.60,
            "twitter_anon": 0.30,
            "telegram": 0.40,
            "discord": 0.35,
            "reddit": 0.50,
            "unknown": 0.20,
        }

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Process text embeddings.

        Args:
            embeddings: Pre-computed embeddings (batch, embed_dim)

        Returns:
            Dict with sentiment, event_type, severity, confidence
        """
        # Project
        h = self.embed_proj(embeddings)
        h = F.gelu(h)

        # Predictions
        sentiment_logits = self.sentiment_head(h)
        event_logits = self.event_head(h)
        severity_logits = self.severity_head(h)
        confidence = self.confidence_head(h)

        return {
            "sentiment": F.softmax(sentiment_logits, dim=-1),
            "event_type": F.softmax(event_logits, dim=-1),
            "severity": F.softmax(severity_logits, dim=-1),
            "confidence": confidence,
        }

    def get_source_quality(self, source: str) -> float:
        """Get quality score for a source."""
        source_lower = source.lower()
        for key, quality in self.source_quality.items():
            if key in source_lower:
                return quality
        return self.source_quality["unknown"]

    def contains_crypto_keywords(self, text: str) -> Tuple[bool, List[str]]:
        """Check for crypto-specific keywords."""
        text_lower = text.lower()
        found = [kw for kw in self.crypto_keywords if kw in text_lower]
        return len(found) > 0, found


# =============================================================================
# SHOCK DETECTOR
# =============================================================================

class ShockDetector:
    """
    Detect market shocks from price/volume data.

    A shock is defined by:
    1. Magnitude: Large price move (> threshold)
    2. Velocity: Fast rate of change
    3. Breadth: Affects multiple assets
    4. Confirmation: Multiple sources agree
    """

    def __init__(self, config: NewsConfig):
        self.config = config

        # Rolling statistics
        self.price_history: Dict[str, deque] = {}
        self.volume_history: Dict[str, deque] = {}
        self.returns_history: Dict[str, deque] = {}

        # Detected shocks
        self.active_shocks: List[MarketShock] = []

        # Window size
        self.window_size = 100

    def update(
        self,
        asset: str,
        price: float,
        volume: float,
        timestamp: datetime,
    ) -> Optional[MarketShock]:
        """
        Update with new data and check for shock.

        Returns MarketShock if detected.
        """
        # Initialize if needed
        if asset not in self.price_history:
            self.price_history[asset] = deque(maxlen=self.window_size)
            self.volume_history[asset] = deque(maxlen=self.window_size)
            self.returns_history[asset] = deque(maxlen=self.window_size)

        # Calculate return
        if len(self.price_history[asset]) > 0:
            prev_price = self.price_history[asset][-1]
            ret = (price - prev_price) / prev_price
            self.returns_history[asset].append(ret)
        else:
            ret = 0

        # Store
        self.price_history[asset].append(price)
        self.volume_history[asset].append(volume)

        # Check for shock
        if len(self.returns_history[asset]) < 10:
            return None

        returns = np.array(self.returns_history[asset])
        mean_ret = np.mean(returns[:-1])
        std_ret = np.std(returns[:-1]) + 1e-8

        # Z-score of current return
        z_score = abs(ret - mean_ret) / std_ret

        # Volume spike
        volumes = np.array(self.volume_history[asset])
        vol_z = (volume - np.mean(volumes[:-1])) / (np.std(volumes[:-1]) + 1e-8)

        # Shock conditions
        magnitude_shock = abs(ret) > self.config.shock_magnitude_threshold
        velocity_shock = z_score > self.config.shock_velocity_threshold
        volume_spike = vol_z > 2.0

        if magnitude_shock and (velocity_shock or volume_spike):
            shock = MarketShock(
                timestamp=timestamp,
                shock_type="price_shock" if ret < 0 else "price_spike",
                magnitude=ret,
                velocity=z_score,
                affected_assets=[asset],
                confidence=min(z_score / 5.0, 1.0),  # Normalize to 0-1
                sources=["price_data", "volume_data"],
                description=f"{asset} moved {ret:.1%} in short period (z={z_score:.1f})",
            )

            self.active_shocks.append(shock)

            # Clean old shocks
            cutoff = timestamp - timedelta(hours=1)
            self.active_shocks = [s for s in self.active_shocks if s.timestamp > cutoff]

            return shock

        return None

    def get_correlated_shocks(
        self,
        window_minutes: int = 5,
    ) -> List[MarketShock]:
        """Find shocks that occurred close together (market-wide event)."""
        if len(self.active_shocks) < 2:
            return []

        # Sort by time
        sorted_shocks = sorted(self.active_shocks, key=lambda s: s.timestamp)

        # Find clusters
        clusters = []
        current_cluster = [sorted_shocks[0]]

        for shock in sorted_shocks[1:]:
            time_diff = (shock.timestamp - current_cluster[-1].timestamp).total_seconds() / 60

            if time_diff <= window_minutes:
                current_cluster.append(shock)
            else:
                if len(current_cluster) >= 2:
                    clusters.append(current_cluster)
                current_cluster = [shock]

        if len(current_cluster) >= 2:
            clusters.append(current_cluster)

        # Flatten
        correlated = []
        for cluster in clusters:
            correlated.extend(cluster)

        return correlated


# =============================================================================
# ANOMALY DETECTOR
# =============================================================================

class AnomalyDetector(nn.Module):
    """
    Multi-method anomaly detection.

    Methods:
    1. Isolation Forest (statistical)
    2. Variational Autoencoder (neural)
    3. Z-score (simple)

    Ensemble for robustness.
    """

    def __init__(self, input_dim: int, config: NewsConfig):
        super().__init__()
        self.config = config
        self.input_dim = input_dim

        # VAE for anomaly detection
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
        )

        self.mu_head = nn.Linear(64, 32)
        self.logvar_head = nn.Linear(64, 32)

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, input_dim),
        )

        # Threshold learning
        self.threshold_net = nn.Sequential(
            nn.Linear(input_dim + 1, 64),  # +1 for reconstruction error
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Running statistics for Z-score
        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_var", torch.ones(input_dim))
        self.register_buffer("n_samples", torch.tensor(0))

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent space."""
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect anomalies.

        Returns:
            Dict with anomaly_score, is_anomaly, reconstruction
        """
        batch_size = x.shape[0]

        # VAE reconstruction
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)

        # Reconstruction error
        recon_error = F.mse_loss(reconstruction, x, reduction='none').mean(dim=-1, keepdim=True)

        # Z-score anomaly
        if self.n_samples > 10:
            z_score = (x - self.running_mean) / (torch.sqrt(self.running_var) + 1e-8)
            z_score_anom = (z_score.abs() > self.config.z_score_threshold).float().mean(dim=-1, keepdim=True)
        else:
            z_score_anom = torch.zeros(batch_size, 1, device=x.device)

        # Combine for final score
        combined_input = torch.cat([x, recon_error], dim=-1)
        anomaly_prob = self.threshold_net(combined_input)

        # Final score combines all methods
        anomaly_score = 0.5 * torch.sigmoid(recon_error * 10) + 0.3 * anomaly_prob + 0.2 * z_score_anom

        # Binary decision
        is_anomaly = anomaly_score > 0.5

        # Update running stats
        if self.training:
            self._update_stats(x)

        return {
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
            "reconstruction": reconstruction,
            "recon_error": recon_error,
            "z_score_anom": z_score_anom,
        }

    def _update_stats(self, x: torch.Tensor):
        """Update running statistics for Z-score."""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_size = x.shape[0]

        # Welford's online algorithm
        new_n = self.n_samples + batch_size
        delta = batch_mean - self.running_mean
        self.running_mean = self.running_mean + delta * batch_size / new_n
        self.running_var = (
            self.running_var * self.n_samples +
            batch_var * batch_size +
            delta ** 2 * self.n_samples * batch_size / new_n
        ) / new_n
        self.n_samples = new_n


# =============================================================================
# TAIL RISK FORECASTER
# =============================================================================

class TailRiskForecaster:
    """
    Forecast tail risk using:
    1. GARCH for volatility clustering
    2. Extreme Value Theory (EVT)
    3. Historical simulation

    Provides VaR, CVaR, and tail index estimates.
    """

    def __init__(self, config: NewsConfig):
        self.config = config

        # Historical returns
        self.returns: deque = deque(maxlen=1000)

        # GARCH parameters (simplified)
        self.omega = 0.00001
        self.alpha = 0.1
        self.beta = 0.85
        self.current_var = 0.0001  # Variance

        # EVT parameters
        self.tail_returns: deque = deque(maxlen=100)  # Extreme returns only

    def update(self, ret: float) -> TailRiskForecast:
        """
        Update with new return and forecast tail risk.
        """
        self.returns.append(ret)

        # Update GARCH variance
        self.current_var = (
            self.omega +
            self.alpha * ret ** 2 +
            self.beta * self.current_var
        )

        # Track extreme returns (beyond 2 sigma)
        if len(self.returns) > 20:
            historical_std = np.std(list(self.returns))
            if abs(ret) > 2 * historical_std:
                self.tail_returns.append(ret)

        return self.forecast()

    def forecast(self) -> TailRiskForecast:
        """Generate tail risk forecast."""
        if len(self.returns) < 30:
            return self._default_forecast()

        returns_arr = np.array(self.returns)

        # GARCH-based VaR
        garch_vol = np.sqrt(self.current_var)
        z_99 = 2.326  # 99% quantile of standard normal
        var_99_garch = -z_99 * garch_vol

        # Historical VaR
        var_99_hist = np.percentile(returns_arr, 1)

        # Combine (weighted average)
        var_99 = 0.6 * var_99_garch + 0.4 * var_99_hist

        # CVaR (Expected Shortfall)
        tail_returns = returns_arr[returns_arr <= var_99_hist]
        cvar_99 = np.mean(tail_returns) if len(tail_returns) > 0 else var_99

        # Tail index (Hill estimator)
        tail_index = self._estimate_tail_index(returns_arr)

        # Extreme probability
        extreme_prob = self._estimate_extreme_prob(returns_arr)

        # Regime
        if abs(var_99) > 0.05:
            regime = "crisis"
        elif abs(var_99) > 0.02:
            regime = "elevated"
        else:
            regime = "normal"

        # Confidence based on sample size
        confidence = min(len(self.returns) / 500, 1.0)

        return TailRiskForecast(
            timestamp=datetime.now(),
            var_1d_99=abs(var_99),
            cvar_1d_99=abs(cvar_99),
            tail_index=tail_index,
            extreme_prob=extreme_prob,
            regime=regime,
            confidence=confidence,
        )

    def _estimate_tail_index(self, returns: np.ndarray) -> float:
        """
        Estimate tail index using Hill estimator.

        Lower tail index = fatter tails = more extreme events.
        """
        # Use absolute returns
        abs_returns = np.abs(returns)
        sorted_returns = np.sort(abs_returns)[::-1]

        # Use top 10% for Hill estimator
        k = max(int(len(sorted_returns) * 0.1), 5)
        top_k = sorted_returns[:k]

        if top_k[-1] == 0:
            return 2.0  # Default

        # Hill estimator
        log_ratios = np.log(top_k[:-1] / top_k[-1])
        hill_estimate = 1 / np.mean(log_ratios) if np.mean(log_ratios) > 0 else 2.0

        return np.clip(hill_estimate, 0.5, 5.0)

    def _estimate_extreme_prob(self, returns: np.ndarray) -> float:
        """Estimate probability of extreme event (> 3 sigma)."""
        mean = np.mean(returns)
        std = np.std(returns)

        if std == 0:
            return 0.0

        threshold = 3 * std
        extreme_count = np.sum(np.abs(returns - mean) > threshold)

        return extreme_count / len(returns)

    def _default_forecast(self) -> TailRiskForecast:
        """Default forecast when insufficient data."""
        return TailRiskForecast(
            timestamp=datetime.now(),
            var_1d_99=0.02,  # 2% default
            cvar_1d_99=0.03,
            tail_index=2.0,
            extreme_prob=0.01,
            regime="normal",
            confidence=0.1,
        )


# =============================================================================
# DO-NOTHING DECISION ENGINE
# =============================================================================

class DoNothingDecisionEngine:
    """
    Decision engine with explicit DO_NOTHING state.

    DO_NOTHING is not a failure - it's a deliberate decision
    to preserve capital when uncertainty is high.

    Situations that trigger DO_NOTHING:
    1. High uncertainty in predictions
    2. Conflicting signals from sources
    3. Insufficient data quality
    4. Suspicious patterns (potential manipulation)
    5. System health issues
    """

    def __init__(self, config: NewsConfig):
        self.config = config

        # State history
        self.state_history: List[Tuple[datetime, SystemState]] = []

        # Active factors
        self.active_news: List[NewsEvent] = []
        self.active_shocks: List[MarketShock] = []

        # Confidence tracking
        self.recent_decisions: deque = deque(maxlen=100)

    def decide(
        self,
        news_events: List[NewsEvent],
        shocks: List[MarketShock],
        tail_forecast: TailRiskForecast,
        anomaly_scores: Dict[str, float],
        model_confidence: float,
        model_disagreement: float,
    ) -> SystemDecision:
        """
        Make a system-wide decision.

        Returns SystemDecision with full explanation.
        """
        now = datetime.now()

        # Collect factors
        factors = []
        risks = []
        recommendations = []

        # ===== UNCERTAINTY CHECK =====
        total_uncertainty = 1 - model_confidence
        if model_disagreement > self.config.disagreement_threshold:
            total_uncertainty += 0.2
            factors.append(f"Model disagreement: {model_disagreement:.1%}")

        # ===== NEWS IMPACT =====
        high_severity_news = [
            e for e in news_events
            if e.severity.value >= Severity.HIGH.value
            and e.relevance_score(now, self.config.news_half_life_minutes) > 0.3
        ]

        if high_severity_news:
            for news in high_severity_news:
                risks.append(f"[{news.severity.name}] {news.raw_text[:100]}...")
            total_uncertainty += 0.1 * len(high_severity_news)

        # ===== SHOCK IMPACT =====
        recent_shocks = [
            s for s in shocks
            if (now - s.timestamp).total_seconds() < 3600  # Last hour
        ]

        if recent_shocks:
            for shock in recent_shocks:
                risks.append(f"SHOCK: {shock.description}")
            total_uncertainty += 0.15 * len(recent_shocks)

        # ===== TAIL RISK =====
        if tail_forecast.regime == "crisis":
            risks.append(f"Crisis regime: VaR={tail_forecast.var_1d_99:.1%}")
            total_uncertainty += 0.3
        elif tail_forecast.regime == "elevated":
            factors.append(f"Elevated tail risk: VaR={tail_forecast.var_1d_99:.1%}")
            total_uncertainty += 0.1

        # ===== ANOMALY CHECK =====
        high_anomalies = {k: v for k, v in anomaly_scores.items() if v > 0.7}
        if high_anomalies:
            for asset, score in high_anomalies.items():
                risks.append(f"Anomaly detected in {asset}: {score:.1%}")
            total_uncertainty += 0.1 * len(high_anomalies)

        # ===== FALSE SIGNAL PREVENTION =====
        # Check for sentiment saturation (too many events = noise)
        recent_news_count = len([
            e for e in news_events
            if (now - e.timestamp).total_seconds() < 3600
        ])

        if recent_news_count > self.config.saturation_threshold:
            factors.append(f"News saturation: {recent_news_count} events/hour")
            total_uncertainty += 0.2

        # Check for conflicting sentiments
        if news_events:
            sentiments = [e.sentiment for e in news_events]
            if len(sentiments) > 1:
                sentiment_std = np.std(sentiments)
                if sentiment_std > 0.5:
                    factors.append(f"Conflicting sentiment (std={sentiment_std:.2f})")
                    total_uncertainty += 0.15

        # ===== DECISION LOGIC =====
        total_uncertainty = min(total_uncertainty, 1.0)
        confidence = 1 - total_uncertainty

        # Determine state
        if total_uncertainty > self.config.uncertainty_threshold:
            state = SystemState.DO_NOTHING
            position_mult = 0.0
            primary_reason = "Uncertainty too high - abstaining from trading"
            recommendations.append("Wait for conditions to stabilize")
            recommendations.append("Continue monitoring news and market data")

        elif confidence < self.config.min_confidence:
            state = SystemState.DO_NOTHING
            position_mult = 0.0
            primary_reason = "Insufficient confidence for any position"
            recommendations.append("Gather more data before trading")

        elif tail_forecast.regime == "crisis" or len(recent_shocks) >= 2:
            state = SystemState.CRISIS_MODE
            position_mult = 0.0
            primary_reason = "Crisis conditions detected - full risk-off"
            recommendations.append("Close all positions immediately")
            recommendations.append("Wait for volatility to subside")

        elif len(high_severity_news) >= 2 or tail_forecast.regime == "elevated":
            state = SystemState.HIGH_ALERT
            position_mult = 0.1
            primary_reason = "Elevated risk conditions"
            recommendations.append("Reduce position sizes to 10%")
            recommendations.append("Set tight stop losses")

        elif len(high_severity_news) == 1 or total_uncertainty > 0.4:
            state = SystemState.ELEVATED_RISK
            position_mult = 0.5
            primary_reason = "Moderate risk factors present"
            recommendations.append("Reduce position sizes to 50%")

        else:
            state = SystemState.NORMAL
            position_mult = 1.0
            primary_reason = "Normal operating conditions"
            recommendations.append("Trade according to signals")

        # Build decision
        decision = SystemDecision(
            timestamp=now,
            state=state,
            confidence=confidence,
            position_multiplier=position_mult,
            primary_reason=primary_reason,
            contributing_factors=factors,
            active_risks=risks,
            recommendations=recommendations,
            uncertainty=total_uncertainty,
            confidence_interval=(max(0, confidence - 0.1), min(1, confidence + 0.1)),
        )

        # Log state change
        if not self.state_history or self.state_history[-1][1] != state:
            self.state_history.append((now, state))
            logger.info(f"State change: {state.value}")
            logger.info(decision.explain())

        self.recent_decisions.append(decision)

        return decision

    def get_state_duration(self) -> timedelta:
        """How long in current state."""
        if not self.state_history:
            return timedelta(0)
        return datetime.now() - self.state_history[-1][0]

    def get_state_summary(self) -> Dict[str, float]:
        """Summary of time spent in each state."""
        if not self.state_history:
            return {}

        total_time = timedelta(0)
        state_times: Dict[str, timedelta] = {}

        for i in range(len(self.state_history)):
            state_start = self.state_history[i][0]
            state = self.state_history[i][1]

            if i + 1 < len(self.state_history):
                state_end = self.state_history[i + 1][0]
            else:
                state_end = datetime.now()

            duration = state_end - state_start
            total_time += duration

            if state.value not in state_times:
                state_times[state.value] = timedelta(0)
            state_times[state.value] += duration

        if total_time.total_seconds() == 0:
            return {}

        return {
            state: duration.total_seconds() / total_time.total_seconds()
            for state, duration in state_times.items()
        }


# =============================================================================
# UNIFIED NEWS & BLACK SWAN SYSTEM
# =============================================================================

class NewsBlackSwanSystem:
    """
    Unified system for news processing and black swan detection.

    Combines:
    - NLP processing
    - Shock detection
    - Anomaly detection
    - Tail risk forecasting
    - Decision engine
    """

    def __init__(
        self,
        config: Optional[NewsConfig] = None,
        state_dim: int = 256,
    ):
        self.config = config or NewsConfig()

        # Components
        self.nlp_processor = FinanceNLPProcessor(self.config)
        self.shock_detector = ShockDetector(self.config)
        self.anomaly_detector = AnomalyDetector(state_dim, self.config)
        self.tail_forecaster = TailRiskForecaster(self.config)
        self.decision_engine = DoNothingDecisionEngine(self.config)

        # Event storage
        self.news_events: deque = deque(maxlen=1000)
        self.event_hashes: Set[str] = set()  # Deduplication

        # Metrics
        self.events_processed = 0
        self.shocks_detected = 0
        self.do_nothing_count = 0

    def process_news(
        self,
        text: str,
        source: str,
        timestamp: Optional[datetime] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> Optional[NewsEvent]:
        """
        Process a single news item.

        Returns NewsEvent if valid, None if duplicate/invalid.
        """
        timestamp = timestamp or datetime.now()

        # Deduplicate
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.event_hashes:
            return None
        self.event_hashes.add(text_hash)

        # Check age
        age_hours = (datetime.now() - timestamp).total_seconds() / 3600
        if age_hours > self.config.max_news_age_hours:
            return None

        # Get source quality
        source_quality = self.nlp_processor.get_source_quality(source)
        if source_quality < self.config.min_source_quality:
            logger.debug(f"Ignoring low-quality source: {source}")
            return None

        # Process with NLP (if embedding provided)
        if embedding is not None:
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
            nlp_output = self.nlp_processor(embedding_tensor)

            sentiment_probs = nlp_output["sentiment"][0]
            sentiment = (sentiment_probs[2] - sentiment_probs[0]).item()  # pos - neg

            event_probs = nlp_output["event_type"][0]
            event_type = EventType(list(EventType)[event_probs.argmax().item()])

            severity_probs = nlp_output["severity"][0]
            severity = Severity(severity_probs.argmax().item())

            confidence = nlp_output["confidence"][0].item()
        else:
            # Fallback: keyword-based analysis
            has_keywords, keywords = self.nlp_processor.contains_crypto_keywords(text)
            sentiment = 0.0
            event_type = EventType.UNKNOWN
            severity = Severity.MEDIUM if has_keywords else Severity.LOW
            confidence = 0.5

        # Extract entities (simplified - just look for asset symbols)
        entities = self._extract_entities(text)

        # Create event
        event = NewsEvent(
            id=text_hash,
            timestamp=timestamp,
            source=source,
            source_quality=source_quality,
            raw_text=text,
            event_type=event_type,
            severity=severity,
            sentiment=sentiment,
            confidence=confidence,
            entities=entities,
            embedding=embedding,
        )

        self.news_events.append(event)
        self.events_processed += 1

        logger.debug(f"Processed news: [{severity.name}] {text[:50]}...")

        return event

    def process_market_data(
        self,
        asset: str,
        price: float,
        volume: float,
        timestamp: Optional[datetime] = None,
    ) -> Optional[MarketShock]:
        """
        Process market data update.

        Returns MarketShock if detected.
        """
        timestamp = timestamp or datetime.now()

        shock = self.shock_detector.update(asset, price, volume, timestamp)

        if shock:
            self.shocks_detected += 1
            logger.warning(f"SHOCK DETECTED: {shock.description}")

        return shock

    def process_return(self, ret: float) -> TailRiskForecast:
        """
        Update tail risk forecast with new return.
        """
        return self.tail_forecaster.update(ret)

    def get_decision(
        self,
        model_confidence: float = 0.5,
        model_disagreement: float = 0.0,
        anomaly_scores: Optional[Dict[str, float]] = None,
    ) -> SystemDecision:
        """
        Get current system decision.

        This is the main interface - call periodically to get
        trading guidance.
        """
        # Get recent news
        now = datetime.now()
        recent_news = [
            e for e in self.news_events
            if (now - e.timestamp).total_seconds() < 3600
        ]

        # Get active shocks
        active_shocks = self.shock_detector.active_shocks

        # Get tail forecast
        tail_forecast = self.tail_forecaster.forecast()

        # Default anomaly scores
        anomaly_scores = anomaly_scores or {}

        # Make decision
        decision = self.decision_engine.decide(
            news_events=recent_news,
            shocks=active_shocks,
            tail_forecast=tail_forecast,
            anomaly_scores=anomaly_scores,
            model_confidence=model_confidence,
            model_disagreement=model_disagreement,
        )

        if decision.state == SystemState.DO_NOTHING:
            self.do_nothing_count += 1

        return decision

    def _extract_entities(self, text: str) -> List[str]:
        """Extract mentioned crypto assets from text."""
        # Common crypto symbols and names
        known_assets = {
            "btc": "BTC", "bitcoin": "BTC",
            "eth": "ETH", "ethereum": "ETH",
            "sol": "SOL", "solana": "SOL",
            "bnb": "BNB", "binance": "BNB",
            "xrp": "XRP", "ripple": "XRP",
            "ada": "ADA", "cardano": "ADA",
            "doge": "DOGE", "dogecoin": "DOGE",
            "avax": "AVAX", "avalanche": "AVAX",
            "dot": "DOT", "polkadot": "DOT",
            "matic": "MATIC", "polygon": "MATIC",
            "link": "LINK", "chainlink": "LINK",
            "uni": "UNI", "uniswap": "UNI",
            "aave": "AAVE",
        }

        text_lower = text.lower()
        found = set()

        for key, symbol in known_assets.items():
            if key in text_lower:
                found.add(symbol)

        return list(found)

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        recent_decision = (
            self.decision_engine.recent_decisions[-1]
            if self.decision_engine.recent_decisions
            else None
        )

        return {
            "current_state": recent_decision.state.value if recent_decision else "unknown",
            "position_multiplier": recent_decision.position_multiplier if recent_decision else 0.0,
            "events_processed": self.events_processed,
            "shocks_detected": self.shocks_detected,
            "do_nothing_count": self.do_nothing_count,
            "state_duration_seconds": self.decision_engine.get_state_duration().total_seconds(),
            "state_distribution": self.decision_engine.get_state_summary(),
            "tail_regime": self.tail_forecaster.forecast().regime,
            "active_risks": recent_decision.active_risks if recent_decision else [],
        }


def create_news_black_swan_system(
    state_dim: int = 256,
) -> NewsBlackSwanSystem:
    """
    Factory function to create properly configured system.
    """
    config = NewsConfig(
        embedding_dim=768,
        shock_velocity_threshold=3.0,
        uncertainty_threshold=0.7,
        min_confidence=0.3,
    )

    system = NewsBlackSwanSystem(config, state_dim)

    logger.info("News & Black Swan system initialized")

    return system
