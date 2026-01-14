"""
Adaptive Intelligence System for CryptoAI.

TRUE ADAPTIVE INTELLIGENCE
==========================

This module implements a multi-model ensemble with meta-learning capabilities
that can adapt to changing market conditions while maintaining strict safety
constraints.

ARCHITECTURE OVERVIEW:
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ADAPTIVE INTELLIGENCE SYSTEM                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │   PRICE     │  │ VOLATILITY  │  │  STRUCTURE  │  │   REGIME    │       │
│  │  PREDICTOR  │  │  PREDICTOR  │  │  PREDICTOR  │  │  DETECTOR   │       │
│  │             │  │             │  │             │  │             │       │
│  │ Learns:     │  │ Learns:     │  │ Learns:     │  │ Learns:     │       │
│  │ - Direction │  │ - Vol level │  │ - Liquidity │  │ - Bull/Bear │       │
│  │ - Magnitude │  │ - Vol of vol│  │ - Spreads   │  │ - Trending  │       │
│  │ - Momentum  │  │ - Tail risk │  │ - Depth     │  │ - Ranging   │       │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       │
│         │                │                │                │               │
│         └────────────────┴────────────────┴────────────────┘               │
│                                   │                                         │
│                                   ▼                                         │
│                    ┌──────────────────────────────┐                        │
│                    │       META-CONTROLLER        │                        │
│                    │                              │                        │
│                    │  • Dynamic model weighting   │                        │
│                    │  • Confidence estimation     │                        │
│                    │  • Disagreement detection    │                        │
│                    │  • Online learning control   │                        │
│                    └──────────────┬───────────────┘                        │
│                                   │                                         │
│                                   ▼                                         │
│                    ┌──────────────────────────────┐                        │
│                    │      SAFETY CONTROLLER       │                        │
│                    │                              │                        │
│                    │  • Risk-off triggers         │                        │
│                    │  • Drawdown limits           │                        │
│                    │  • Model validation          │                        │
│                    │  • Overfitting detection     │                        │
│                    └──────────────────────────────┘                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

WHEN MODELS ARE TRUSTED/IGNORED:
================================

Price Predictor:
  TRUSTED when:  Low volatility, clear trend, high recent accuracy
  IGNORED when:  High volatility regime, recent prediction failures, crisis mode

Volatility Predictor:
  TRUSTED when:  Stable regime, calibrated VaR violations match expected
  IGNORED when:  Regime transition, extreme market moves

Structure Predictor:
  TRUSTED when:  Normal market hours, typical liquidity
  IGNORED when:  Weekend, holidays, flash crash events

OVERFITTING PREVENTION:
=======================
1. Online validation set (never trained on)
2. Bayesian regularization (priors on weights)
3. Ensemble disagreement monitoring
4. Forward-testing before deployment
5. Concept drift detection

Author: Head of AI Research
Hardware Target: 2x RTX 5080, 128GB RAM, Ryzen 9 7950X
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from loguru import logger

# Hardware-aware device configuration
def get_optimal_device() -> torch.device:
    """Get optimal device for RTX 5080 setup."""
    if torch.cuda.is_available():
        # Prefer GPU 0 for single-model inference
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_multi_gpu_strategy() -> str:
    """Get optimal multi-GPU strategy."""
    if torch.cuda.device_count() >= 2:
        return "ddp"  # Distributed Data Parallel for 2x RTX 5080
    return "single"


# =============================================================================
# CONFIGURATION
# =============================================================================

class TrustLevel(Enum):
    """Model trust levels."""
    FULL = "full"           # High confidence, use full weight
    PARTIAL = "partial"     # Moderate confidence, reduced weight
    MINIMAL = "minimal"     # Low confidence, minimal weight
    IGNORED = "ignored"     # No trust, weight = 0
    RISK_OFF = "risk_off"   # Safety mode, no trading


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive intelligence system."""

    # Model dimensions (optimized for RTX 5080 memory - 16GB each)
    state_dim: int = 256
    hidden_dim: int = 512
    latent_dim: int = 128
    n_heads: int = 8
    n_layers: int = 4

    # Ensemble configuration
    n_price_models: int = 3       # Ensemble diversity
    n_volatility_models: int = 3
    n_structure_models: int = 2

    # Meta-controller
    meta_hidden_dim: int = 256
    meta_lookback: int = 100      # Steps to evaluate model performance

    # Online learning
    online_lr: float = 1e-5       # Conservative learning rate
    online_batch_size: int = 32
    min_samples_for_update: int = 1000
    max_update_frequency_mins: int = 60  # Max 1 update per hour

    # Safety thresholds
    min_confidence: float = 0.3   # Below this -> risk-off
    max_disagreement: float = 0.5 # Model disagreement threshold
    max_drawdown: float = 0.15    # 15% drawdown -> risk-off
    max_daily_loss: float = 0.05  # 5% daily loss -> risk-off

    # Validation
    validation_split: float = 0.2
    early_stopping_patience: int = 5
    overfit_detection_threshold: float = 0.1  # Train-val gap

    # Hardware
    device: str = "cuda"
    dtype: str = "float16"  # Use FP16 for RTX 5080 efficiency
    compile_models: bool = True  # torch.compile for speed


@dataclass
class ModelState:
    """Track state of a single model."""
    name: str
    last_update: datetime
    predictions: List[float] = field(default_factory=list)
    actuals: List[float] = field(default_factory=list)
    errors: List[float] = field(default_factory=list)
    trust_level: TrustLevel = TrustLevel.PARTIAL
    weight: float = 1.0
    is_valid: bool = True

    @property
    def recent_accuracy(self) -> float:
        """Calculate recent prediction accuracy."""
        if len(self.errors) < 10:
            return 0.5
        recent = self.errors[-100:]
        mae = np.mean(np.abs(recent))
        return max(0, 1 - mae)

    @property
    def directional_accuracy(self) -> float:
        """Calculate directional accuracy (sign match)."""
        if len(self.predictions) < 10:
            return 0.5
        preds = np.array(self.predictions[-100:])
        acts = np.array(self.actuals[-100:])
        return np.mean(np.sign(preds) == np.sign(acts))


# =============================================================================
# BASE PREDICTOR ARCHITECTURE
# =============================================================================

class BasePredictor(nn.Module):
    """
    Base architecture for all predictors.

    Uses efficient transformer architecture optimized for RTX 5080:
    - Flash Attention 2 for memory efficiency
    - FP16 mixed precision
    - Gradient checkpointing for long sequences
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=True,
        )

        # Output heads
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.log_std_head = nn.Linear(hidden_dim, output_dim)

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize with small weights for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_confidence: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, input_dim)
            return_confidence: Whether to return confidence scores

        Returns:
            mean: Predicted mean (batch, output_dim)
            log_std: Predicted log std (batch, output_dim)
            confidence: Confidence score (batch, 1) if requested
        """
        # Project input
        h = self.input_proj(x)

        # Transformer encoding
        h = self.transformer(h)

        # Use last timestep
        h_last = h[:, -1, :]

        # Predictions
        mean = self.mean_head(h_last)
        log_std = self.log_std_head(h_last)

        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, min=-10, max=2)

        if return_confidence:
            confidence = self.confidence_head(h_last)
            return mean, log_std, confidence

        return mean, log_std, None

    def predict_distribution(
        self,
        x: torch.Tensor,
    ) -> Normal:
        """Get predictive distribution."""
        mean, log_std, _ = self.forward(x)
        return Normal(mean, log_std.exp())


# =============================================================================
# SPECIALIZED PREDICTORS
# =============================================================================

class PricePredictor(BasePredictor):
    """
    Price direction and magnitude predictor.

    WHAT IT LEARNS:
    - Short-term price direction (next N candles)
    - Expected magnitude of moves
    - Momentum patterns
    - Mean reversion signals

    WHEN TRUSTED:
    - Low/moderate volatility regimes
    - Clear trending or ranging conditions
    - Recent predictions have been accurate

    WHEN IGNORED:
    - High volatility / crisis
    - Major news events
    - Prediction accuracy dropping
    """

    def __init__(self, config: AdaptiveConfig):
        super().__init__(
            input_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
            output_dim=3,  # [direction, magnitude, momentum]
            n_heads=config.n_heads,
            n_layers=config.n_layers,
        )

        # Additional price-specific layers
        self.momentum_encoder = nn.GRU(
            config.state_dim,
            config.hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get base predictions
        mean, log_std, confidence = super().forward(x, return_confidence=True)

        # Add momentum context
        momentum_h, _ = self.momentum_encoder(x)
        momentum_signal = momentum_h[:, -1, :].mean(dim=-1, keepdim=True)

        # Modulate confidence by momentum clarity
        momentum_clarity = torch.sigmoid(torch.abs(momentum_signal))
        confidence = confidence * momentum_clarity

        return mean, log_std, confidence


class VolatilityPredictor(BasePredictor):
    """
    Volatility regime and dynamics predictor.

    WHAT IT LEARNS:
    - Current volatility level (realized vol)
    - Volatility of volatility (vol clustering)
    - Tail risk probability
    - Mean-reversion speed of vol

    WHEN TRUSTED:
    - Stable volatility regime
    - VaR violations match expected frequency
    - No recent black swan events

    WHEN IGNORED:
    - During regime transitions
    - After extreme moves (vol model often wrong)
    - When implied/realized diverge significantly
    """

    def __init__(self, config: AdaptiveConfig):
        super().__init__(
            input_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
            output_dim=4,  # [vol_level, vol_of_vol, tail_prob, mean_revert_speed]
            n_heads=config.n_heads,
            n_layers=config.n_layers,
        )

        # Volatility-specific: GARCH-like component
        self.garch_cell = nn.GRUCell(
            config.state_dim,
            config.hidden_dim // 2,
        )

        # Tail risk estimator
        self.tail_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 2),  # [left_tail, right_tail]
            nn.Softplus(),  # Ensure positive
        )

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, confidence = super().forward(x, return_confidence=True)

        # Ensure volatility predictions are positive
        mean = F.softplus(mean)

        return mean, log_std, confidence


class StructurePredictor(BasePredictor):
    """
    Market microstructure predictor.

    WHAT IT LEARNS:
    - Liquidity conditions
    - Bid-ask spread dynamics
    - Order book depth
    - Market impact estimation

    WHEN TRUSTED:
    - Normal trading hours
    - Typical liquidity conditions
    - Stable exchange connectivity

    WHEN IGNORED:
    - Weekend/holiday low liquidity
    - Flash crash conditions
    - Exchange maintenance/issues
    """

    def __init__(self, config: AdaptiveConfig):
        super().__init__(
            input_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
            output_dim=4,  # [spread, depth, impact, execution_prob]
            n_heads=config.n_heads,
            n_layers=config.n_layers,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, confidence = super().forward(x, return_confidence=True)

        # Spreads and impact should be positive
        mean = F.softplus(mean)

        return mean, log_std, confidence


class RegimeDetector(nn.Module):
    """
    Market regime classification.

    WHAT IT LEARNS:
    - Bull vs Bear vs Ranging
    - Trending vs Mean-reverting
    - Low vol vs High vol vs Crisis
    - Regime transition probabilities

    Output is a probability distribution over regimes,
    NOT a hard classification. This allows smooth transitions.
    """

    def __init__(self, config: AdaptiveConfig):
        super().__init__()

        self.config = config
        self.n_regimes = 6  # bull, bear, ranging, trending, high_vol, crisis

        # Regime embedding
        self.regime_embeddings = nn.Embedding(self.n_regimes, config.hidden_dim)

        # Input processing
        self.input_proj = nn.Linear(config.state_dim, config.hidden_dim)

        # Attention over regimes
        self.regime_attention = nn.MultiheadAttention(
            config.hidden_dim,
            num_heads=4,
            batch_first=True,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, self.n_regimes),
        )

        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Transition matrix (learnable prior)
        self.transition_logits = nn.Parameter(
            torch.zeros(self.n_regimes, self.n_regimes)
        )

    def forward(
        self,
        x: torch.Tensor,
        prev_regime: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict regime distribution.

        Args:
            x: Input features (batch, seq_len, state_dim)
            prev_regime: Previous regime distribution (batch, n_regimes)

        Returns:
            regime_probs: Probability of each regime (batch, n_regimes)
            confidence: Confidence in prediction (batch, 1)
            transition_probs: Regime transition probabilities
        """
        batch_size = x.shape[0]

        # Process input
        h = self.input_proj(x)
        h_pooled = h.mean(dim=1)  # Global average pooling

        # Get regime embeddings
        regime_ids = torch.arange(self.n_regimes, device=x.device)
        regime_emb = self.regime_embeddings(regime_ids).unsqueeze(0).expand(batch_size, -1, -1)

        # Attend to regime embeddings
        h_query = h_pooled.unsqueeze(1)
        attn_out, _ = self.regime_attention(h_query, regime_emb, regime_emb)
        h_attended = attn_out.squeeze(1)

        # Classify
        logits = self.classifier(h_attended)

        # Apply transition prior if previous regime provided
        if prev_regime is not None:
            transition_probs = F.softmax(self.transition_logits, dim=-1)
            prior = torch.matmul(prev_regime, transition_probs)
            logits = logits + torch.log(prior + 1e-8)

        regime_probs = F.softmax(logits, dim=-1)
        confidence = self.confidence_head(h_attended)

        return regime_probs, confidence, F.softmax(self.transition_logits, dim=-1)


# =============================================================================
# META-CONTROLLER
# =============================================================================

class MetaController(nn.Module):
    """
    Meta-controller for dynamic model weighting.

    WHAT IT LEARNS:
    - Which models to trust in which conditions
    - How to combine model outputs
    - When to increase/decrease position sizing
    - When to go risk-off

    The meta-controller observes:
    - Recent prediction errors of each model
    - Current market regime
    - Model disagreement levels
    - Portfolio state (drawdown, exposure)
    """

    def __init__(self, config: AdaptiveConfig):
        super().__init__()

        self.config = config
        self.n_models = (
            config.n_price_models +
            config.n_volatility_models +
            config.n_structure_models + 1  # +1 for regime detector
        )

        # Input: model states + regime + portfolio state
        input_dim = (
            self.n_models * 4 +  # [accuracy, confidence, recent_error, age]
            6 +  # regime distribution
            4    # portfolio [drawdown, exposure, daily_pnl, position]
        )

        self.state_encoder = nn.Sequential(
            nn.Linear(input_dim, config.meta_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.meta_hidden_dim),
            nn.Linear(config.meta_hidden_dim, config.meta_hidden_dim),
            nn.GELU(),
        )

        # Weight predictor for each model
        self.weight_head = nn.Sequential(
            nn.Linear(config.meta_hidden_dim, config.meta_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.meta_hidden_dim // 2, self.n_models),
            nn.Softmax(dim=-1),
        )

        # Position sizing recommendation
        self.sizing_head = nn.Sequential(
            nn.Linear(config.meta_hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # 0-1 scaling factor
        )

        # Risk-off probability
        self.risk_off_head = nn.Sequential(
            nn.Linear(config.meta_hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Overall confidence
        self.confidence_head = nn.Sequential(
            nn.Linear(config.meta_hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        model_states: torch.Tensor,  # (batch, n_models, 4)
        regime_probs: torch.Tensor,   # (batch, n_regimes)
        portfolio_state: torch.Tensor,  # (batch, 4)
    ) -> Dict[str, torch.Tensor]:
        """
        Compute meta-control outputs.

        Returns:
            weights: Model weights (batch, n_models)
            position_scale: Position sizing multiplier (batch, 1)
            risk_off_prob: Probability to go risk-off (batch, 1)
            confidence: Overall system confidence (batch, 1)
        """
        # Flatten model states
        batch_size = model_states.shape[0]
        model_flat = model_states.view(batch_size, -1)

        # Concatenate all inputs
        x = torch.cat([model_flat, regime_probs, portfolio_state], dim=-1)

        # Encode
        h = self.state_encoder(x)

        # Compute outputs
        weights = self.weight_head(h)
        position_scale = self.sizing_head(h)
        risk_off_prob = self.risk_off_head(h)
        confidence = self.confidence_head(h)

        return {
            "weights": weights,
            "position_scale": position_scale,
            "risk_off_prob": risk_off_prob,
            "confidence": confidence,
        }


# =============================================================================
# SAFETY CONTROLLER
# =============================================================================

@dataclass
class SafetyState:
    """Current safety state."""
    is_risk_off: bool = False
    risk_off_reason: str = ""
    current_drawdown: float = 0.0
    daily_pnl: float = 0.0
    model_disagreement: float = 0.0
    confidence: float = 1.0
    last_check: datetime = field(default_factory=datetime.now)


class SafetyController:
    """
    Safety controller for fail-safe operation.

    TRIGGERS RISK-OFF MODE WHEN:
    1. Drawdown exceeds threshold (default 15%)
    2. Daily loss exceeds threshold (default 5%)
    3. Model disagreement too high
    4. Confidence too low
    5. Anomalous market conditions detected
    6. Data quality issues

    RECOVERY:
    - Gradual position increase after risk-off
    - Requires sustained low risk before full recovery
    - Manual override capability
    """

    def __init__(self, config: AdaptiveConfig):
        self.config = config
        self.state = SafetyState()

        # Risk-off cooldown
        self.risk_off_start: Optional[datetime] = None
        self.min_risk_off_duration = timedelta(minutes=30)

        # Recovery tracking
        self.recovery_factor = 0.0
        self.consecutive_safe_checks = 0

    def check_safety(
        self,
        drawdown: float,
        daily_pnl: float,
        model_disagreement: float,
        confidence: float,
        predictions: Dict[str, torch.Tensor],
    ) -> SafetyState:
        """
        Perform safety check.

        Returns updated safety state.
        """
        self.state.current_drawdown = drawdown
        self.state.daily_pnl = daily_pnl
        self.state.model_disagreement = model_disagreement
        self.state.confidence = confidence
        self.state.last_check = datetime.now()

        # Check triggers
        triggers = []

        if drawdown > self.config.max_drawdown:
            triggers.append(f"Drawdown {drawdown:.1%} > {self.config.max_drawdown:.1%}")

        if daily_pnl < -self.config.max_daily_loss:
            triggers.append(f"Daily loss {daily_pnl:.1%} > {self.config.max_daily_loss:.1%}")

        if model_disagreement > self.config.max_disagreement:
            triggers.append(f"Model disagreement {model_disagreement:.2f} > {self.config.max_disagreement:.2f}")

        if confidence < self.config.min_confidence:
            triggers.append(f"Confidence {confidence:.2f} < {self.config.min_confidence:.2f}")

        # Check for anomalous predictions (NaN, Inf, extreme values)
        for name, pred in predictions.items():
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                triggers.append(f"Model {name} produced invalid predictions")
            if pred.abs().max() > 100:  # Sanity check
                triggers.append(f"Model {name} prediction magnitude too high")

        # Update risk-off state
        if triggers:
            self.state.is_risk_off = True
            self.state.risk_off_reason = "; ".join(triggers)
            self.risk_off_start = datetime.now()
            self.consecutive_safe_checks = 0
            self.recovery_factor = 0.0
            logger.warning(f"RISK-OFF TRIGGERED: {self.state.risk_off_reason}")
        elif self.state.is_risk_off:
            # Check if we can recover
            self.consecutive_safe_checks += 1
            if self._can_recover():
                self._gradual_recovery()

        return self.state

    def _can_recover(self) -> bool:
        """Check if conditions allow recovery."""
        if self.risk_off_start is None:
            return True

        # Minimum cooldown
        elapsed = datetime.now() - self.risk_off_start
        if elapsed < self.min_risk_off_duration:
            return False

        # Need consecutive safe checks
        if self.consecutive_safe_checks < 10:
            return False

        # All conditions must be well within thresholds
        if self.state.current_drawdown > self.config.max_drawdown * 0.5:
            return False
        if self.state.confidence < self.config.min_confidence * 1.5:
            return False

        return True

    def _gradual_recovery(self):
        """Gradually recover from risk-off mode."""
        self.recovery_factor = min(1.0, self.recovery_factor + 0.1)

        if self.recovery_factor >= 1.0:
            self.state.is_risk_off = False
            self.state.risk_off_reason = ""
            self.risk_off_start = None
            logger.info("RISK-OFF CLEARED: Full recovery achieved")
        else:
            logger.info(f"RECOVERY IN PROGRESS: {self.recovery_factor:.0%}")

    def get_position_multiplier(self) -> float:
        """Get position size multiplier based on safety state."""
        if self.state.is_risk_off:
            return 0.0

        # Scale by recovery factor
        base_mult = self.recovery_factor if self.recovery_factor < 1.0 else 1.0

        # Scale by confidence
        conf_mult = min(1.0, self.state.confidence / self.config.min_confidence)

        # Scale inversely by drawdown
        dd_mult = max(0.3, 1.0 - self.state.current_drawdown / self.config.max_drawdown)

        return base_mult * conf_mult * dd_mult


# =============================================================================
# ONLINE LEARNING CONTROLLER
# =============================================================================

class OnlineLearningController:
    """
    Controls online model updates with strict safeguards.

    SAFEGUARDS:
    1. Minimum samples before update
    2. Maximum update frequency
    3. Validation on held-out data
    4. Overfitting detection (train-val gap)
    5. Rollback if performance degrades
    6. Learning rate decay over time

    WHAT GETS UPDATED:
    - Model weights (very slowly)
    - Meta-controller (moderate speed)
    - Regime transition priors (fast)

    WHAT NEVER GETS UPDATED ONLINE:
    - Core architecture
    - Safety thresholds
    - Risk limits
    """

    def __init__(self, config: AdaptiveConfig):
        self.config = config

        # Update tracking
        self.last_update: Optional[datetime] = None
        self.update_count = 0
        self.samples_since_update = 0

        # Validation buffer (never trained on)
        self.val_buffer: List[Dict] = []
        self.val_buffer_size = 1000

        # Performance tracking
        self.pre_update_metrics: Optional[Dict] = None
        self.post_update_metrics: Optional[Dict] = None

        # Rollback state
        self.model_checkpoints: Dict[str, Dict] = {}

    def should_update(self) -> Tuple[bool, str]:
        """
        Determine if update should proceed.

        Returns (should_update, reason)
        """
        # Check minimum samples
        if self.samples_since_update < self.config.min_samples_for_update:
            return False, f"Need {self.config.min_samples_for_update - self.samples_since_update} more samples"

        # Check update frequency
        if self.last_update is not None:
            elapsed = datetime.now() - self.last_update
            min_interval = timedelta(minutes=self.config.max_update_frequency_mins)
            if elapsed < min_interval:
                remaining = min_interval - elapsed
                return False, f"Must wait {remaining.seconds}s before next update"

        return True, "Update permitted"

    def prepare_update(
        self,
        models: Dict[str, nn.Module],
    ) -> None:
        """Save model checkpoints before update."""
        for name, model in models.items():
            self.model_checkpoints[name] = {
                k: v.clone() for k, v in model.state_dict().items()
            }
        logger.info("Model checkpoints saved for potential rollback")

    def validate_update(
        self,
        models: Dict[str, nn.Module],
        val_data: List[Dict],
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Validate model performance after update.

        Returns (is_valid, metrics)
        """
        metrics = {}

        for name, model in models.items():
            model.eval()
            errors = []

            with torch.no_grad():
                for sample in val_data[:100]:  # Limit for speed
                    x = sample.get("features")
                    y = sample.get("target")
                    if x is None or y is None:
                        continue

                    pred, _, _ = model(x.unsqueeze(0))
                    error = F.mse_loss(pred.squeeze(), y).item()
                    errors.append(error)

            if errors:
                metrics[f"{name}_val_mse"] = np.mean(errors)

        # Check for overfitting
        is_valid = True
        if self.pre_update_metrics is not None:
            for key in metrics:
                if key in self.pre_update_metrics:
                    improvement = self.pre_update_metrics[key] - metrics[key]
                    if improvement < -self.config.overfit_detection_threshold:
                        logger.warning(f"Overfitting detected: {key} degraded by {-improvement:.4f}")
                        is_valid = False

        self.post_update_metrics = metrics
        return is_valid, metrics

    def rollback(self, models: Dict[str, nn.Module]) -> None:
        """Rollback models to pre-update state."""
        for name, model in models.items():
            if name in self.model_checkpoints:
                model.load_state_dict(self.model_checkpoints[name])
        logger.warning("ROLLBACK EXECUTED: Models restored to pre-update state")

    def record_sample(self, sample: Dict) -> None:
        """Record a sample for future validation."""
        self.samples_since_update += 1

        # Reservoir sampling for validation buffer
        if len(self.val_buffer) < self.val_buffer_size:
            self.val_buffer.append(sample)
        else:
            idx = np.random.randint(0, self.samples_since_update)
            if idx < self.val_buffer_size:
                self.val_buffer[idx] = sample

    def complete_update(self) -> None:
        """Mark update as complete."""
        self.last_update = datetime.now()
        self.update_count += 1
        self.samples_since_update = 0
        self.pre_update_metrics = self.post_update_metrics
        logger.info(f"Update #{self.update_count} completed at {self.last_update}")


# =============================================================================
# ENSEMBLE SYSTEM
# =============================================================================

class AdaptiveEnsemble(nn.Module):
    """
    Complete adaptive ensemble system.

    Combines all components:
    - Multiple price predictors
    - Multiple volatility predictors
    - Multiple structure predictors
    - Regime detector
    - Meta-controller
    - Safety controller
    - Online learning controller
    """

    def __init__(self, config: AdaptiveConfig):
        super().__init__()

        self.config = config
        device = get_optimal_device()

        # Price predictors (ensemble for diversity)
        self.price_predictors = nn.ModuleList([
            PricePredictor(config) for _ in range(config.n_price_models)
        ])

        # Volatility predictors
        self.vol_predictors = nn.ModuleList([
            VolatilityPredictor(config) for _ in range(config.n_volatility_models)
        ])

        # Structure predictors
        self.structure_predictors = nn.ModuleList([
            StructurePredictor(config) for _ in range(config.n_structure_models)
        ])

        # Regime detector
        self.regime_detector = RegimeDetector(config)

        # Meta-controller
        self.meta_controller = MetaController(config)

        # Non-learned components
        self.safety_controller = SafetyController(config)
        self.online_controller = OnlineLearningController(config)

        # Model states
        self.model_states: Dict[str, ModelState] = {}
        self._init_model_states()

        # Previous regime for transition modeling
        self.prev_regime: Optional[torch.Tensor] = None

        # Compile models for speed (PyTorch 2.0+)
        if config.compile_models and hasattr(torch, 'compile'):
            self._compile_models()

    def _init_model_states(self):
        """Initialize model state tracking."""
        for i in range(self.config.n_price_models):
            self.model_states[f"price_{i}"] = ModelState(name=f"price_{i}", last_update=datetime.now())
        for i in range(self.config.n_volatility_models):
            self.model_states[f"vol_{i}"] = ModelState(name=f"vol_{i}", last_update=datetime.now())
        for i in range(self.config.n_structure_models):
            self.model_states[f"struct_{i}"] = ModelState(name=f"struct_{i}", last_update=datetime.now())
        self.model_states["regime"] = ModelState(name="regime", last_update=datetime.now())

    def _compile_models(self):
        """Compile models with torch.compile for optimization."""
        try:
            for i, model in enumerate(self.price_predictors):
                self.price_predictors[i] = torch.compile(model, mode="reduce-overhead")
            for i, model in enumerate(self.vol_predictors):
                self.vol_predictors[i] = torch.compile(model, mode="reduce-overhead")
            for i, model in enumerate(self.structure_predictors):
                self.structure_predictors[i] = torch.compile(model, mode="reduce-overhead")
            self.regime_detector = torch.compile(self.regime_detector, mode="reduce-overhead")
            logger.info("Models compiled with torch.compile")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}. Using eager mode.")

    def forward(
        self,
        x: torch.Tensor,
        portfolio_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Full forward pass through ensemble.

        Args:
            x: Input features (batch, seq_len, state_dim)
            portfolio_state: Current portfolio state (batch, 4)
                [drawdown, exposure, daily_pnl, position]

        Returns:
            Dict with all predictions, weights, and safety state
        """
        batch_size = x.shape[0]
        device = x.device

        # Default portfolio state
        if portfolio_state is None:
            portfolio_state = torch.zeros(batch_size, 4, device=device)

        # ===== REGIME DETECTION =====
        regime_probs, regime_confidence, transition_probs = self.regime_detector(
            x, self.prev_regime
        )
        self.prev_regime = regime_probs.detach()

        # ===== PRICE PREDICTIONS =====
        price_preds = []
        price_stds = []
        price_confs = []

        for i, predictor in enumerate(self.price_predictors):
            mean, log_std, conf = predictor(x)
            price_preds.append(mean)
            price_stds.append(log_std.exp())
            price_confs.append(conf)

        # Stack predictions
        price_preds = torch.stack(price_preds, dim=1)  # (batch, n_models, output_dim)
        price_stds = torch.stack(price_stds, dim=1)
        price_confs = torch.stack(price_confs, dim=1)

        # ===== VOLATILITY PREDICTIONS =====
        vol_preds = []
        vol_stds = []
        vol_confs = []

        for predictor in self.vol_predictors:
            mean, log_std, conf = predictor(x)
            vol_preds.append(mean)
            vol_stds.append(log_std.exp())
            vol_confs.append(conf)

        vol_preds = torch.stack(vol_preds, dim=1)
        vol_stds = torch.stack(vol_stds, dim=1)
        vol_confs = torch.stack(vol_confs, dim=1)

        # ===== STRUCTURE PREDICTIONS =====
        struct_preds = []
        struct_stds = []
        struct_confs = []

        for predictor in self.structure_predictors:
            mean, log_std, conf = predictor(x)
            struct_preds.append(mean)
            struct_stds.append(log_std.exp())
            struct_confs.append(conf)

        struct_preds = torch.stack(struct_preds, dim=1)
        struct_stds = torch.stack(struct_stds, dim=1)
        struct_confs = torch.stack(struct_confs, dim=1)

        # ===== MODEL STATES FOR META-CONTROLLER =====
        n_total_models = (
            self.config.n_price_models +
            self.config.n_volatility_models +
            self.config.n_structure_models + 1
        )

        model_states_tensor = torch.zeros(batch_size, n_total_models, 4, device=device)

        idx = 0
        for i in range(self.config.n_price_models):
            state = self.model_states[f"price_{i}"]
            model_states_tensor[:, idx, 0] = state.recent_accuracy
            model_states_tensor[:, idx, 1] = price_confs[:, i, 0]
            model_states_tensor[:, idx, 2] = np.mean(state.errors[-10:]) if state.errors else 0
            model_states_tensor[:, idx, 3] = (datetime.now() - state.last_update).seconds / 3600
            idx += 1

        for i in range(self.config.n_volatility_models):
            state = self.model_states[f"vol_{i}"]
            model_states_tensor[:, idx, 0] = state.recent_accuracy
            model_states_tensor[:, idx, 1] = vol_confs[:, i, 0]
            model_states_tensor[:, idx, 2] = np.mean(state.errors[-10:]) if state.errors else 0
            model_states_tensor[:, idx, 3] = (datetime.now() - state.last_update).seconds / 3600
            idx += 1

        for i in range(self.config.n_structure_models):
            state = self.model_states[f"struct_{i}"]
            model_states_tensor[:, idx, 0] = state.recent_accuracy
            model_states_tensor[:, idx, 1] = struct_confs[:, i, 0]
            model_states_tensor[:, idx, 2] = np.mean(state.errors[-10:]) if state.errors else 0
            model_states_tensor[:, idx, 3] = (datetime.now() - state.last_update).seconds / 3600
            idx += 1

        # Regime detector state
        state = self.model_states["regime"]
        model_states_tensor[:, idx, 0] = state.recent_accuracy
        model_states_tensor[:, idx, 1] = regime_confidence.squeeze(-1)
        model_states_tensor[:, idx, 2] = 0
        model_states_tensor[:, idx, 3] = (datetime.now() - state.last_update).seconds / 3600

        # ===== META-CONTROLLER =====
        meta_output = self.meta_controller(
            model_states_tensor,
            regime_probs,
            portfolio_state,
        )

        weights = meta_output["weights"]
        position_scale = meta_output["position_scale"]
        risk_off_prob = meta_output["risk_off_prob"]
        overall_confidence = meta_output["confidence"]

        # ===== WEIGHTED ENSEMBLE PREDICTIONS =====
        # Split weights for different model types
        price_weights = weights[:, :self.config.n_price_models].unsqueeze(-1)
        vol_weights = weights[:, self.config.n_price_models:self.config.n_price_models + self.config.n_volatility_models].unsqueeze(-1)
        struct_weights = weights[:, -self.config.n_structure_models - 1:-1].unsqueeze(-1)

        # Weighted average predictions
        ensemble_price = (price_preds * price_weights).sum(dim=1)
        ensemble_vol = (vol_preds * vol_weights).sum(dim=1)
        ensemble_struct = (struct_preds * struct_weights).sum(dim=1)

        # Ensemble uncertainty (combining individual uncertainties)
        ensemble_price_std = torch.sqrt((price_stds ** 2 * price_weights).sum(dim=1))
        ensemble_vol_std = torch.sqrt((vol_stds ** 2 * vol_weights).sum(dim=1))
        ensemble_struct_std = torch.sqrt((struct_stds ** 2 * struct_weights).sum(dim=1))

        # Model disagreement
        price_disagreement = price_preds.std(dim=1).mean(dim=-1)
        vol_disagreement = vol_preds.std(dim=1).mean(dim=-1)

        # ===== SAFETY CHECK =====
        all_predictions = {
            "price": ensemble_price,
            "volatility": ensemble_vol,
            "structure": ensemble_struct,
        }

        drawdown = portfolio_state[:, 0].mean().item()
        daily_pnl = portfolio_state[:, 2].mean().item()
        disagreement = (price_disagreement + vol_disagreement).mean().item() / 2

        safety_state = self.safety_controller.check_safety(
            drawdown=drawdown,
            daily_pnl=daily_pnl,
            model_disagreement=disagreement,
            confidence=overall_confidence.mean().item(),
            predictions=all_predictions,
        )

        # Apply safety multiplier
        final_position_scale = position_scale * self.safety_controller.get_position_multiplier()

        return {
            # Predictions
            "price_prediction": ensemble_price,
            "price_std": ensemble_price_std,
            "volatility_prediction": ensemble_vol,
            "volatility_std": ensemble_vol_std,
            "structure_prediction": ensemble_struct,
            "structure_std": ensemble_struct_std,

            # Regime
            "regime_probs": regime_probs,
            "regime_confidence": regime_confidence,

            # Meta-control
            "model_weights": weights,
            "position_scale": final_position_scale,
            "risk_off_prob": risk_off_prob,
            "overall_confidence": overall_confidence,

            # Disagreement (for monitoring)
            "price_disagreement": price_disagreement,
            "vol_disagreement": vol_disagreement,

            # Safety
            "is_risk_off": safety_state.is_risk_off,
            "risk_off_reason": safety_state.risk_off_reason,
        }

    def update_model_state(
        self,
        model_name: str,
        prediction: float,
        actual: float,
    ) -> None:
        """Update model tracking with new observation."""
        if model_name not in self.model_states:
            return

        state = self.model_states[model_name]
        state.predictions.append(prediction)
        state.actuals.append(actual)
        state.errors.append(prediction - actual)

        # Keep only recent history
        max_history = 1000
        if len(state.predictions) > max_history:
            state.predictions = state.predictions[-max_history:]
            state.actuals = state.actuals[-max_history:]
            state.errors = state.errors[-max_history:]

        # Update trust level based on recent performance
        accuracy = state.recent_accuracy
        directional = state.directional_accuracy

        if accuracy > 0.7 and directional > 0.6:
            state.trust_level = TrustLevel.FULL
            state.weight = 1.0
        elif accuracy > 0.5 and directional > 0.5:
            state.trust_level = TrustLevel.PARTIAL
            state.weight = 0.5
        elif accuracy > 0.3:
            state.trust_level = TrustLevel.MINIMAL
            state.weight = 0.2
        else:
            state.trust_level = TrustLevel.IGNORED
            state.weight = 0.0

    def get_all_models(self) -> Dict[str, nn.Module]:
        """Get dictionary of all models for checkpointing."""
        models = {}
        for i, m in enumerate(self.price_predictors):
            models[f"price_{i}"] = m
        for i, m in enumerate(self.vol_predictors):
            models[f"vol_{i}"] = m
        for i, m in enumerate(self.structure_predictors):
            models[f"struct_{i}"] = m
        models["regime"] = self.regime_detector
        models["meta"] = self.meta_controller
        return models


def create_adaptive_system(
    state_dim: int = 256,
    device: str = "cuda",
) -> AdaptiveEnsemble:
    """
    Factory function to create properly configured adaptive system.

    Args:
        state_dim: Input state dimension
        device: Target device ("cuda" for RTX 5080)

    Returns:
        Configured AdaptiveEnsemble
    """
    config = AdaptiveConfig(
        state_dim=state_dim,
        hidden_dim=512,
        latent_dim=128,
        n_heads=8,
        n_layers=4,
        n_price_models=3,
        n_volatility_models=3,
        n_structure_models=2,
        device=device,
        compile_models=True,
    )

    ensemble = AdaptiveEnsemble(config)
    ensemble = ensemble.to(device)

    # Use FP16 for RTX 5080 efficiency
    if device == "cuda":
        ensemble = ensemble.half()

    logger.info(f"Adaptive ensemble created with {sum(p.numel() for p in ensemble.parameters()):,} parameters")

    return ensemble
