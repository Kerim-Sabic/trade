"""Anomaly detection for black swan events."""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyDetector(nn.Module):
    """
    Isolation Forest-style anomaly detection using neural networks.

    Detects unusual market states that may precede black swan events.
    """

    def __init__(
        self,
        state_dim: int = 256,
        hidden_dim: int = 128,
        num_estimators: int = 10,
        contamination: float = 0.01,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.num_estimators = num_estimators
        self.contamination = contamination

        # Ensemble of anomaly scorers
        self.estimators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in range(num_estimators)
        ])

        # Threshold (learned)
        self.threshold = nn.Parameter(torch.tensor(0.0))

        # Running statistics for normalization
        self.register_buffer("running_mean", torch.zeros(state_dim))
        self.register_buffer("running_var", torch.ones(state_dim))
        self.register_buffer("num_samples", torch.tensor(0.0))

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute anomaly score.

        Args:
            state: Market state

        Returns:
            Dict with anomaly score and is_anomaly flag
        """
        # Normalize state
        state_norm = (state - self.running_mean) / (self.running_var.sqrt() + 1e-8)

        # Get scores from all estimators
        scores = []
        for estimator in self.estimators:
            score = estimator(state_norm)
            scores.append(score)

        scores = torch.stack(scores, dim=-1)  # (batch, 1, num_estimators)

        # Aggregate scores
        mean_score = scores.mean(dim=-1).squeeze(-1)
        std_score = scores.std(dim=-1).squeeze(-1)

        # Normalize to 0-1
        anomaly_score = torch.sigmoid(mean_score - self.threshold)

        # Is anomaly flag
        is_anomaly = anomaly_score > (1 - self.contamination)

        return {
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
            "score_std": std_score,
            "raw_score": mean_score,
        }

    def update_statistics(self, state: torch.Tensor):
        """Update running statistics with new observations."""
        with torch.no_grad():
            batch_mean = state.mean(dim=0)
            batch_var = state.var(dim=0)

            # Exponential moving average
            alpha = 0.01
            self.running_mean = (1 - alpha) * self.running_mean + alpha * batch_mean
            self.running_var = (1 - alpha) * self.running_var + alpha * batch_var
            self.num_samples += state.shape[0]


class VariationalAnomalyDetector(nn.Module):
    """
    Variational autoencoder-based anomaly detection.

    Anomalies have high reconstruction error.
    """

    def __init__(
        self,
        state_dim: int = 256,
        latent_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

        # Latent distribution parameters
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Anomaly threshold (adaptive)
        self.register_buffer("recon_threshold", torch.tensor(0.1))
        self.register_buffer("recon_mean", torch.tensor(0.0))
        self.register_buffer("recon_std", torch.tensor(1.0))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent distribution."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent."""
        return self.decoder(z)

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns anomaly score based on reconstruction error.
        """
        # Encode
        mu, log_var = self.encode(state)

        # Sample latent
        z = self.reparameterize(mu, log_var)

        # Decode
        recon = self.decode(z)

        # Reconstruction error
        recon_error = F.mse_loss(recon, state, reduction="none").mean(dim=-1)

        # KL divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        # Normalize reconstruction error
        normalized_error = (recon_error - self.recon_mean) / (self.recon_std + 1e-8)

        # Anomaly score (sigmoid of normalized error)
        anomaly_score = torch.sigmoid(normalized_error)

        return {
            "anomaly_score": anomaly_score,
            "recon_error": recon_error,
            "kl_div": kl_div,
            "latent_mu": mu,
            "latent_std": torch.exp(0.5 * log_var),
            "reconstruction": recon,
        }

    def update_threshold(self, recon_errors: torch.Tensor):
        """Update adaptive threshold based on recent errors."""
        with torch.no_grad():
            alpha = 0.01
            self.recon_mean = (1 - alpha) * self.recon_mean + alpha * recon_errors.mean()
            self.recon_std = (1 - alpha) * self.recon_std + alpha * recon_errors.std()


class TemporalAnomalyDetector(nn.Module):
    """
    Detects anomalies in temporal patterns.

    Uses LSTM to model expected sequences, flags unexpected transitions.
    """

    def __init__(
        self,
        state_dim: int = 256,
        hidden_dim: int = 128,
        sequence_length: int = 20,
    ):
        super().__init__()

        self.sequence_length = sequence_length

        # Sequence model
        self.lstm = nn.LSTM(
            state_dim, hidden_dim,
            num_layers=2, batch_first=True
        )

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Anomaly assessment
        self.anomaly_head = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        current_state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Detect temporal anomaly.

        Args:
            sequence: Historical states (batch, seq_len, state_dim)
            current_state: Current state to check

        Returns:
            Dict with temporal anomaly score
        """
        # Predict next state from sequence
        lstm_out, _ = self.lstm(sequence)
        predicted = self.predictor(lstm_out[:, -1, :])

        # Compare prediction to actual
        prediction_error = F.mse_loss(predicted, current_state, reduction="none").mean(dim=-1)

        # Anomaly assessment
        combined = torch.cat([predicted, current_state], dim=-1)
        anomaly_score = self.anomaly_head(combined).squeeze(-1)

        return {
            "temporal_anomaly_score": anomaly_score,
            "prediction_error": prediction_error,
            "predicted_state": predicted,
        }
