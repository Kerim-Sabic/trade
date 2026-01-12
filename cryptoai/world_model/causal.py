"""Causal discovery module for market relationships."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CausalDiscoveryModule(nn.Module):
    """
    Discovers causal structure between market variables.

    Learns:
    - Which features cause which
    - Temporal lag structure
    - Regime-dependent causality
    """

    def __init__(
        self,
        num_variables: int = 10,
        hidden_dim: int = 128,
        num_lags: int = 5,
        temperature: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_variables = num_variables
        self.num_lags = num_lags
        self.temperature = temperature

        # Causal adjacency matrix (learnable, soft)
        self.causal_logits = nn.Parameter(
            torch.randn(num_variables, num_variables, num_lags) * 0.01
        )

        # Per-variable encoders
        self.variable_encoders = nn.ModuleList([
            nn.Linear(num_lags, hidden_dim)
            for _ in range(num_variables)
        ])

        # Causal mechanism (how causes produce effects)
        self.causal_mechanism = nn.Sequential(
            nn.Linear(hidden_dim * num_variables, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Effect prediction heads (one per variable)
        self.effect_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1)
            for _ in range(num_variables)
        ])

        # Regime-specific causal adjustments
        self.regime_adjustments = nn.Parameter(
            torch.zeros(5, num_variables, num_variables)
        )

    def get_causal_matrix(
        self,
        hard: bool = False,
        regime: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get the causal adjacency matrix.

        Args:
            hard: Use hard (binary) adjacency
            regime: Optional regime probabilities (batch, num_regimes)

        Returns:
            Causal matrix (num_variables, num_variables, num_lags)
        """
        # Soft adjacency
        if hard:
            adj = (torch.sigmoid(self.causal_logits / self.temperature) > 0.5).float()
        else:
            adj = torch.sigmoid(self.causal_logits / self.temperature)

        # Apply regime adjustment
        if regime is not None:
            B = regime.shape[0]
            # regime: (B, num_regimes)
            # regime_adjustments: (num_regimes, V, V)
            # Sum across lags for regime effect
            adj_summed = adj.mean(dim=-1)  # (V, V)
            regime_effect = torch.einsum(
                "br,rvw->bvw",
                regime,
                self.regime_adjustments
            )  # (B, V, V)
            # Expand adj for batch
            adj_expanded = adj_summed.unsqueeze(0).expand(B, -1, -1)
            adj_modified = adj_expanded + 0.1 * regime_effect
            return torch.sigmoid(adj_modified)

        return adj

    def forward(
        self,
        x: torch.Tensor,
        regime: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass - predict effects from causes.

        Args:
            x: Variable time series (batch, time, num_variables)
            regime: Optional regime probabilities

        Returns:
            Dict with predictions and causal structure
        """
        B, T, V = x.shape
        device = x.device

        # Get causal matrix
        causal_adj = self.get_causal_matrix(regime=regime)

        # Prepare lagged inputs
        if T <= self.num_lags:
            # Pad if sequence too short
            x = F.pad(x, (0, 0, self.num_lags - T + 1, 0))
            T = x.shape[1]

        # Create lagged features
        lagged = []
        for lag in range(self.num_lags):
            lagged.append(x[:, self.num_lags - 1 - lag:T - lag, :])
        lagged = torch.stack(lagged, dim=-1)  # (B, T-num_lags, V, num_lags)

        # Encode each variable
        encoded = []
        for v in range(V):
            var_lagged = lagged[:, :, v, :]  # (B, T', num_lags)
            enc = self.variable_encoders[v](var_lagged)  # (B, T', hidden)
            encoded.append(enc)
        encoded = torch.stack(encoded, dim=2)  # (B, T', V, hidden)

        # Apply causal mechanism
        # Weight encoded by causal adjacency
        if regime is not None:
            # causal_adj: (B, V, V)
            weighted = torch.einsum("btvh,bwv->btwh", encoded, causal_adj)
        else:
            # causal_adj: (V, V, num_lags) - use mean across lags
            adj_mean = causal_adj.mean(dim=-1)  # (V, V)
            weighted = torch.einsum("btvh,wv->btwh", encoded, adj_mean)

        # Flatten for mechanism
        T_out = weighted.shape[1]
        weighted_flat = rearrange(weighted, "b t v h -> b t (v h)")
        mechanism_out = self.causal_mechanism(weighted_flat)  # (B, T', hidden)

        # Predict each variable
        predictions = []
        for v in range(V):
            pred = self.effect_heads[v](mechanism_out)
            predictions.append(pred)
        predictions = torch.cat(predictions, dim=-1)  # (B, T', V)

        return {
            "predictions": predictions,
            "causal_matrix": causal_adj if regime is None else causal_adj.mean(dim=0),
            "encoded": encoded,
        }

    def get_intervention_effect(
        self,
        x: torch.Tensor,
        intervention_var: int,
        intervention_value: float,
        regime: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute counterfactual: what if we intervene on a variable?

        Args:
            x: Current state
            intervention_var: Which variable to intervene on
            intervention_value: Value to set
            regime: Optional regime

        Returns:
            Predicted state after intervention
        """
        x_intervened = x.clone()
        x_intervened[:, :, intervention_var] = intervention_value

        result = self.forward(x_intervened, regime)
        return result["predictions"]

    def sparsity_loss(self) -> torch.Tensor:
        """Encourage sparse causal structure."""
        adj = torch.sigmoid(self.causal_logits / self.temperature)
        # L1 sparsity + diagonal penalty (no self-causation at same time)
        sparsity = adj.abs().mean()
        diag_penalty = adj.diagonal(dim1=0, dim2=1).mean()
        return sparsity + 0.5 * diag_penalty

    def acyclicity_loss(self) -> torch.Tensor:
        """
        Encourage acyclic causal structure (DAG constraint).

        Uses trace exponential characterization.
        """
        adj = torch.sigmoid(self.causal_logits / self.temperature).mean(dim=-1)  # Average over lags
        # h(A) = tr(exp(A ◦ A)) - d for DAG constraint
        M = adj * adj  # Element-wise square
        # Matrix exponential approximation
        E = torch.eye(self.num_variables, device=adj.device)
        power = E
        trace_sum = torch.trace(power)
        for _ in range(5):
            power = power @ M
            trace_sum = trace_sum + torch.trace(power) / _
        return trace_sum - self.num_variables


class TemporalCausalGraph(nn.Module):
    """
    Temporal causal graph with dynamic adjacency.

    Learns time-varying causal relationships.
    """

    def __init__(
        self,
        num_variables: int = 10,
        hidden_dim: int = 64,
        context_dim: int = 32,
        num_lags: int = 5,
    ):
        super().__init__()

        self.num_variables = num_variables
        self.num_lags = num_lags

        # Context encoder (market regime/state)
        self.context_encoder = nn.Sequential(
            nn.Linear(num_variables * num_lags, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, context_dim),
        )

        # Dynamic adjacency generator
        self.adj_generator = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_variables * num_variables),
        )

        # Base adjacency (learned prior)
        self.base_adj = nn.Parameter(
            torch.randn(num_variables, num_variables) * 0.1
        )

    def get_dynamic_adjacency(
        self,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get context-dependent causal adjacency.

        Args:
            context: Market context (batch, context_dim)

        Returns:
            Dynamic adjacency matrix (batch, V, V)
        """
        adj_flat = self.adj_generator(context)
        adj = adj_flat.view(-1, self.num_variables, self.num_variables)

        # Add base adjacency
        adj = adj + self.base_adj.unsqueeze(0)

        return torch.sigmoid(adj)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with dynamic causal structure.
        """
        B, T, V = x.shape

        # Prepare lagged context
        if T >= self.num_lags:
            context_input = x[:, -self.num_lags:, :].reshape(B, -1)
        else:
            context_input = F.pad(x, (0, 0, self.num_lags - T, 0)).reshape(B, -1)

        # Encode context
        context = self.context_encoder(context_input)

        # Get dynamic adjacency
        adj = self.get_dynamic_adjacency(context)

        return {
            "adjacency": adj,
            "context": context,
        }
