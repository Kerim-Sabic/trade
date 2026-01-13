"""
World Model for CryptoAI.

Learns market transition dynamics, latent regimes, and causal structure.

Purpose:
- Anticipate market instability
- Provide foresight to policy
- Enable counterfactual reasoning
"""

from cryptoai.world_model.temporal_transformer import (
    TemporalTransformerWorldModel,
    WorldModelEnsemble,
)
from cryptoai.world_model.latent_dynamics import LatentDynamicsModel
from cryptoai.world_model.causal import CausalDiscoveryModule

# Alias for convenience
WorldModel = TemporalTransformerWorldModel

__all__ = [
    "TemporalTransformerWorldModel",
    "WorldModel",
    "WorldModelEnsemble",
    "LatentDynamicsModel",
    "CausalDiscoveryModule",
]
