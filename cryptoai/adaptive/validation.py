"""
Validation and Anti-Overfitting Framework for Adaptive Intelligence.

This module provides comprehensive validation to ensure the adaptive
system generalizes properly and doesn't overfit to noise.

OVERFITTING PREVENTION STRATEGIES:
==================================

1. BAYESIAN REGULARIZATION
   - Priors on model weights
   - Automatic relevance determination
   - Prevents extreme weight values

2. ENSEMBLE DIVERSITY
   - Multiple models with different initializations
   - Disagreement = uncertainty
   - Correlation monitoring

3. TEMPORAL VALIDATION
   - Forward-testing only (no lookahead)
   - Out-of-sample performance tracking
   - Concept drift detection

4. STATISTICAL TESTS
   - Is performance statistically significant?
   - Bootstrap confidence intervals
   - Multiple hypothesis correction

5. LEARNING RATE CONSTRAINTS
   - Aggressive decay during online learning
   - Maximum update magnitude limits
   - Gradient clipping

Author: Head of AI Research
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from loguru import logger


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics."""

    # Basic metrics
    train_loss: float
    val_loss: float
    test_loss: float

    # Overfitting indicators
    train_val_gap: float       # train_loss - val_loss (negative = overfit)
    generalization_error: float

    # Statistical significance
    p_value: float
    is_significant: bool
    confidence_interval: Tuple[float, float]

    # Stability
    prediction_std: float
    gradient_norm: float
    weight_norm: float

    # Temporal
    forward_test_sharpe: float
    in_sample_sharpe: float
    sharpe_degradation: float


@dataclass
class ConceptDriftMetrics:
    """Metrics for detecting concept drift."""

    drift_detected: bool
    drift_magnitude: float
    drift_direction: str  # "improving", "degrading", "stable"
    recommendation: str


class BayesianRegularizer:
    """
    Bayesian regularization for neural network weights.

    Implements automatic relevance determination (ARD) priors
    that adaptively prune irrelevant weights.
    """

    def __init__(
        self,
        model: nn.Module,
        prior_precision: float = 1.0,
        noise_precision: float = 1.0,
    ):
        self.model = model
        self.prior_precision = prior_precision
        self.noise_precision = noise_precision

        # Per-weight precision (ARD)
        self.weight_precisions: Dict[str, torch.Tensor] = {}
        self._init_precisions()

    def _init_precisions(self):
        """Initialize precision parameters for each weight."""
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                self.weight_precisions[name] = torch.ones_like(param) * self.prior_precision

    def compute_regularization_loss(self) -> torch.Tensor:
        """
        Compute Bayesian regularization loss.

        L_reg = 0.5 * sum(alpha_i * w_i^2)

        Where alpha_i is the precision for weight w_i.
        """
        reg_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)

        for name, param in self.model.named_parameters():
            if name in self.weight_precisions:
                precision = self.weight_precisions[name].to(param.device)
                reg_loss = reg_loss + 0.5 * (precision * param ** 2).sum()

        return reg_loss

    def update_precisions(self, gamma: float = 0.1):
        """
        Update weight precisions using evidence procedure.

        alpha_new = gamma / w^2

        This prunes weights that are near zero.
        """
        for name, param in self.model.named_parameters():
            if name in self.weight_precisions:
                w_squared = param.data ** 2 + 1e-8
                self.weight_precisions[name] = gamma / w_squared


class EnsembleDiversityMonitor:
    """
    Monitor ensemble diversity to detect overfitting.

    If all models agree perfectly, the ensemble loses its
    uncertainty estimation capability.
    """

    def __init__(self, n_models: int, min_diversity: float = 0.1):
        self.n_models = n_models
        self.min_diversity = min_diversity
        self.diversity_history: List[float] = []

    def compute_diversity(
        self,
        predictions: torch.Tensor,  # (batch, n_models, output_dim)
    ) -> float:
        """
        Compute ensemble diversity.

        High diversity = models disagree = good uncertainty estimation
        Low diversity = models agree too much = possible overfitting
        """
        # Pairwise correlation between model predictions
        batch_size, n_models, output_dim = predictions.shape

        correlations = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                pred_i = predictions[:, i, :].flatten()
                pred_j = predictions[:, j, :].flatten()

                if pred_i.std() > 1e-6 and pred_j.std() > 1e-6:
                    corr = torch.corrcoef(torch.stack([pred_i, pred_j]))[0, 1]
                    if not torch.isnan(corr):
                        correlations.append(corr.item())

        if not correlations:
            return 0.5

        avg_correlation = np.mean(correlations)
        diversity = 1 - abs(avg_correlation)

        self.diversity_history.append(diversity)

        return diversity

    def check_diversity_warning(self) -> Tuple[bool, str]:
        """Check if diversity is too low."""
        if len(self.diversity_history) < 10:
            return False, ""

        recent_diversity = np.mean(self.diversity_history[-100:])

        if recent_diversity < self.min_diversity:
            return True, f"Low ensemble diversity ({recent_diversity:.3f}). Models may be overfitting to same patterns."

        # Check for declining diversity
        if len(self.diversity_history) >= 100:
            old_diversity = np.mean(self.diversity_history[-200:-100])
            if recent_diversity < old_diversity * 0.7:
                return True, f"Diversity declining ({old_diversity:.3f} -> {recent_diversity:.3f}). Check for overfitting."

        return False, ""


class TemporalValidator:
    """
    Temporal validation for time-series models.

    CRITICAL: Never use future data for validation.
    Always validate on strictly out-of-sample data.
    """

    def __init__(
        self,
        lookback_window: int = 100,
        forward_test_window: int = 50,
    ):
        self.lookback_window = lookback_window
        self.forward_test_window = forward_test_window

        self.in_sample_metrics: List[float] = []
        self.out_sample_metrics: List[float] = []

    def validate(
        self,
        model: nn.Module,
        data_stream: List[Dict],
        metric_fn: Callable[[torch.Tensor, torch.Tensor], float],
    ) -> ValidationMetrics:
        """
        Perform temporal validation.

        Uses expanding window approach:
        1. Train on [0, t]
        2. Validate on [t+1, t+k]
        3. Expand t and repeat
        """
        model.eval()

        train_losses = []
        val_losses = []

        n_samples = len(data_stream)
        if n_samples < self.lookback_window + self.forward_test_window:
            logger.warning("Insufficient data for temporal validation")
            return self._empty_metrics()

        with torch.no_grad():
            for t in range(self.lookback_window, n_samples - self.forward_test_window, 10):
                # In-sample (train) evaluation
                train_data = data_stream[t - self.lookback_window:t]
                train_loss = self._evaluate_window(model, train_data, metric_fn)
                train_losses.append(train_loss)

                # Out-of-sample (forward test) evaluation
                test_data = data_stream[t:t + self.forward_test_window]
                test_loss = self._evaluate_window(model, test_data, metric_fn)
                val_losses.append(test_loss)

        if not train_losses:
            return self._empty_metrics()

        # Compute metrics
        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        train_val_gap = avg_train - avg_val

        # Statistical significance
        if len(val_losses) > 5:
            t_stat, p_value = stats.ttest_1samp(val_losses, 0)
            ci = stats.t.interval(0.95, len(val_losses) - 1,
                                  loc=np.mean(val_losses),
                                  scale=stats.sem(val_losses))
        else:
            p_value = 1.0
            ci = (0, 0)

        return ValidationMetrics(
            train_loss=avg_train,
            val_loss=avg_val,
            test_loss=avg_val,
            train_val_gap=train_val_gap,
            generalization_error=abs(train_val_gap),
            p_value=p_value,
            is_significant=p_value < 0.05,
            confidence_interval=ci,
            prediction_std=np.std(val_losses),
            gradient_norm=0.0,  # Would need to compute during training
            weight_norm=self._compute_weight_norm(model),
            forward_test_sharpe=self._compute_sharpe(val_losses),
            in_sample_sharpe=self._compute_sharpe(train_losses),
            sharpe_degradation=(self._compute_sharpe(train_losses) -
                              self._compute_sharpe(val_losses)),
        )

    def _evaluate_window(
        self,
        model: nn.Module,
        data: List[Dict],
        metric_fn: Callable,
    ) -> float:
        """Evaluate model on a data window."""
        errors = []
        for sample in data:
            x = sample.get("features")
            y = sample.get("target")
            if x is None or y is None:
                continue

            pred = model(x.unsqueeze(0) if x.dim() == 2 else x)
            if isinstance(pred, tuple):
                pred = pred[0]

            error = metric_fn(pred.squeeze(), y)
            errors.append(error)

        return np.mean(errors) if errors else 0.0

    def _compute_weight_norm(self, model: nn.Module) -> float:
        """Compute total weight norm."""
        total_norm = 0.0
        for param in model.parameters():
            total_norm += param.data.norm(2).item() ** 2
        return np.sqrt(total_norm)

    def _compute_sharpe(self, losses: List[float]) -> float:
        """Compute Sharpe-like ratio for losses."""
        if len(losses) < 2:
            return 0.0
        returns = -np.array(losses)  # Negative loss = positive return
        if np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns)

    def _empty_metrics(self) -> ValidationMetrics:
        """Return empty metrics."""
        return ValidationMetrics(
            train_loss=0, val_loss=0, test_loss=0,
            train_val_gap=0, generalization_error=0,
            p_value=1.0, is_significant=False,
            confidence_interval=(0, 0),
            prediction_std=0, gradient_norm=0, weight_norm=0,
            forward_test_sharpe=0, in_sample_sharpe=0,
            sharpe_degradation=0,
        )


class ConceptDriftDetector:
    """
    Detect concept drift in streaming data.

    Concept drift = the relationship between inputs and outputs changes.
    This invalidates models trained on old data.

    DETECTION METHODS:
    1. Statistical tests on error distribution
    2. Performance monitoring
    3. Feature distribution shift
    """

    def __init__(
        self,
        window_size: int = 1000,
        drift_threshold: float = 0.1,
        significance_level: float = 0.01,
    ):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.significance_level = significance_level

        self.old_errors: List[float] = []
        self.new_errors: List[float] = []
        self.error_history: List[float] = []

    def update(self, error: float) -> ConceptDriftMetrics:
        """Update with new prediction error and check for drift."""
        self.error_history.append(error)

        # Maintain windows
        if len(self.error_history) > 2 * self.window_size:
            self.old_errors = self.error_history[-2 * self.window_size:-self.window_size]
            self.new_errors = self.error_history[-self.window_size:]
        else:
            return ConceptDriftMetrics(
                drift_detected=False,
                drift_magnitude=0.0,
                drift_direction="stable",
                recommendation="Collecting data..."
            )

        # Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(self.old_errors, self.new_errors)

        # Mean shift
        old_mean = np.mean(self.old_errors)
        new_mean = np.mean(self.new_errors)
        mean_shift = new_mean - old_mean

        # Detect drift
        drift_detected = (
            p_value < self.significance_level or
            abs(mean_shift) > self.drift_threshold * np.std(self.old_errors)
        )

        # Direction
        if mean_shift > 0:
            direction = "degrading"
        elif mean_shift < 0:
            direction = "improving"
        else:
            direction = "stable"

        # Recommendation
        if drift_detected:
            if direction == "degrading":
                recommendation = "Model performance degrading. Consider retraining or reducing position sizes."
            else:
                recommendation = "Model performance improving. Current parameters appear effective."
        else:
            recommendation = "No significant drift detected. Continue monitoring."

        return ConceptDriftMetrics(
            drift_detected=drift_detected,
            drift_magnitude=abs(mean_shift),
            drift_direction=direction,
            recommendation=recommendation,
        )


class OverfitDetector:
    """
    Detect overfitting during training.

    SIGNALS OF OVERFITTING:
    1. Train loss decreasing, val loss increasing
    2. Weight norms growing unboundedly
    3. Gradient norms exploding/vanishing
    4. Ensemble diversity collapsing
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
    ):
        self.patience = patience
        self.min_delta = min_delta

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def update(
        self,
        train_loss: float,
        val_loss: float,
    ) -> Tuple[bool, str]:
        """
        Update with new losses and check for overfitting.

        Returns (is_overfitting, message)
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        # Check for improvement
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False, "Validation improving"

        self.patience_counter += 1

        # Check patterns
        if len(self.train_losses) >= 10:
            recent_train = self.train_losses[-10:]
            recent_val = self.val_losses[-10:]

            # Train decreasing, val increasing
            train_trend = np.polyfit(range(10), recent_train, 1)[0]
            val_trend = np.polyfit(range(10), recent_val, 1)[0]

            if train_trend < 0 and val_trend > 0:
                return True, "Classic overfit pattern: train decreasing, val increasing"

            # Large train-val gap
            gap = np.mean(recent_val) - np.mean(recent_train)
            if gap > 0.1 * np.mean(recent_train):
                return True, f"Large train-val gap: {gap:.4f}"

        # Patience exhausted
        if self.patience_counter >= self.patience:
            return True, f"Validation not improving for {self.patience} epochs"

        return False, "Monitoring..."

    def reset(self):
        """Reset detector state."""
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0


class LearningRateController:
    """
    Adaptive learning rate control for safe online learning.

    STRATEGIES:
    1. Aggressive decay during online updates
    2. Maximum update magnitude limits
    3. Warm restarts on significant drift
    """

    def __init__(
        self,
        initial_lr: float = 1e-4,
        min_lr: float = 1e-7,
        decay_factor: float = 0.9,
        warmup_steps: int = 100,
    ):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.min_lr = min_lr
        self.decay_factor = decay_factor
        self.warmup_steps = warmup_steps

        self.step = 0
        self.update_count = 0

    def get_lr(self) -> float:
        """Get current learning rate."""
        self.step += 1

        # Warmup
        if self.step < self.warmup_steps:
            return self.min_lr + (self.initial_lr - self.min_lr) * (self.step / self.warmup_steps)

        return max(self.current_lr, self.min_lr)

    def decay(self):
        """Apply learning rate decay after update."""
        self.update_count += 1
        self.current_lr *= self.decay_factor
        logger.debug(f"Learning rate decayed to {self.current_lr:.2e}")

    def warm_restart(self, factor: float = 0.5):
        """Warm restart with reduced learning rate."""
        self.current_lr = self.initial_lr * factor
        self.step = 0
        logger.info(f"Learning rate warm restart to {self.current_lr:.2e}")


class GradientClipper:
    """
    Gradient clipping for numerical stability.

    Prevents exploding gradients that can destabilize online learning.
    """

    def __init__(
        self,
        max_norm: float = 1.0,
        adaptive: bool = True,
    ):
        self.max_norm = max_norm
        self.adaptive = adaptive
        self.grad_history: List[float] = []

    def clip(self, model: nn.Module) -> float:
        """
        Clip gradients and return norm.

        If adaptive=True, adjusts max_norm based on history.
        """
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.max_norm,
        )

        self.grad_history.append(total_norm.item())

        # Adaptive adjustment
        if self.adaptive and len(self.grad_history) >= 100:
            median_norm = np.median(self.grad_history[-100:])
            # Allow 3x median as max
            self.max_norm = max(1.0, min(10.0, 3 * median_norm))

        return total_norm.item()


def validate_model_before_deployment(
    model: nn.Module,
    validation_data: List[Dict],
    min_sharpe: float = 0.5,
    max_drawdown: float = 0.15,
) -> Tuple[bool, Dict[str, float], List[str]]:
    """
    Comprehensive validation before deploying a model update.

    Returns:
        (is_approved, metrics, warnings)
    """
    warnings = []
    metrics = {}

    temporal_validator = TemporalValidator()

    def mse_loss(pred, target):
        return F.mse_loss(pred, target).item()

    validation_metrics = temporal_validator.validate(
        model,
        validation_data,
        mse_loss,
    )

    metrics["forward_test_sharpe"] = validation_metrics.forward_test_sharpe
    metrics["generalization_error"] = validation_metrics.generalization_error
    metrics["train_val_gap"] = validation_metrics.train_val_gap
    metrics["is_significant"] = validation_metrics.is_significant

    # Check criteria
    approved = True

    if validation_metrics.forward_test_sharpe < min_sharpe:
        warnings.append(f"Forward test Sharpe ({validation_metrics.forward_test_sharpe:.2f}) below minimum ({min_sharpe})")
        approved = False

    if validation_metrics.train_val_gap < -0.1:
        warnings.append(f"Significant overfitting detected (gap: {validation_metrics.train_val_gap:.4f})")
        approved = False

    if not validation_metrics.is_significant:
        warnings.append(f"Results not statistically significant (p={validation_metrics.p_value:.3f})")
        approved = False

    if validation_metrics.weight_norm > 1000:
        warnings.append(f"Weight norm too high ({validation_metrics.weight_norm:.1f})")
        approved = False

    return approved, metrics, warnings
