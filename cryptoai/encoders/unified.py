"""Unified state encoder combining all component encoders."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from cryptoai.encoders.order_flow import OrderFlowEncoder, MultiScaleOrderFlowEncoder
from cryptoai.encoders.liquidity import LiquidityEncoder
from cryptoai.encoders.derivatives import DerivativesEncoder
from cryptoai.encoders.onchain import OnChainEncoder
from cryptoai.encoders.events import EventEncoder
from cryptoai.encoders.asset import AssetEncoder, DominanceConditioner
from cryptoai.encoders.regime import RegimeEncoder
from cryptoai.encoders.base import BaseEncoder, GatedLinearUnit, TransformerBlock


class UnifiedStateEncoder(BaseEncoder):
    """
    Unified State Encoder.

    Combines all individual encoders into a single unified market state:

    S_t = concat(
        order_flow_embedding,
        liquidity_embedding,
        derivatives_embedding,
        onchain_embedding,
        event_embedding,
        asset_embedding,
        regime_embedding
    )

    All inputs are encoded via neural networks - NO classical indicators.
    """

    def __init__(
        self,
        # Input dimensions
        microstructure_dim: int = 40,
        derivatives_dim: int = 42,
        onchain_dim: int = 33,
        event_dim: int = 15,
        # Encoder configs
        order_flow_hidden: int = 256,
        order_flow_output: int = 128,
        liquidity_output: int = 64,
        derivatives_output: int = 64,
        onchain_output: int = 64,
        event_output: int = 64,
        asset_output: int = 64,
        regime_output: int = 64,
        # Unified config
        unified_hidden: int = 512,
        unified_output: int = 256,
        sequence_length: int = 100,
        num_assets: int = 100,
        num_regimes: int = 5,
        dropout: float = 0.1,
    ):
        # Total output from all encoders
        total_encoder_output = (
            order_flow_output + liquidity_output + derivatives_output +
            onchain_output + event_output + asset_output + regime_output
        )

        super().__init__(total_encoder_output, unified_output, dropout)

        self.unified_hidden = unified_hidden
        self.unified_output = unified_output
        self.sequence_length = sequence_length

        # Individual encoders
        self.order_flow_encoder = OrderFlowEncoder(
            input_dim=microstructure_dim,
            hidden_dim=order_flow_hidden,
            output_dim=order_flow_output,
            num_layers=4,
            num_heads=8,
            sequence_length=sequence_length,
            dropout=dropout,
        )

        self.liquidity_encoder = LiquidityEncoder(
            input_dim=microstructure_dim // 2,
            num_levels=20,
            output_dim=liquidity_output,
            dropout=dropout,
        )

        self.derivatives_encoder = DerivativesEncoder(
            input_dim=derivatives_dim,
            hidden_dim=128,
            output_dim=derivatives_output,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
        )

        self.onchain_encoder = OnChainEncoder(
            input_dim=onchain_dim,
            hidden_dim=128,
            output_dim=onchain_output,
            num_layers=2,
            dropout=dropout,
        )

        self.event_encoder = EventEncoder(
            numeric_features=event_dim,
            hidden_dim=256,
            output_dim=event_output,
            dropout=dropout,
        )

        self.asset_encoder = AssetEncoder(
            num_assets=num_assets,
            embedding_dim=64,
            output_dim=asset_output,
            dropout=dropout,
        )

        self.regime_encoder = RegimeEncoder(
            input_dim=total_encoder_output - regime_output,  # Without regime itself
            hidden_dim=128,
            output_dim=regime_output,
            num_regimes=num_regimes,
            dropout=dropout,
        )

        # Cross-modal attention
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

        # Modal projections for attention
        encoder_dims = [
            order_flow_output, liquidity_output, derivatives_output,
            onchain_output, event_output, asset_output
        ]
        self.modal_projections = nn.ModuleList([
            nn.Linear(dim, 128) for dim in encoder_dims
        ])

        # Dominance conditioning
        self.dominance_conditioner = DominanceConditioner(
            input_dim=total_encoder_output,
            dominance_dim=3,
        )

        # Unified fusion
        self.fusion = nn.Sequential(
            nn.Linear(total_encoder_output, unified_hidden),
            nn.LayerNorm(unified_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.fusion_transformer = TransformerBlock(
            unified_hidden, num_heads=8, dropout=dropout
        )

        self.output_proj = nn.Sequential(
            nn.Linear(unified_hidden, unified_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(unified_hidden, unified_output),
        )

        # Gated residual
        self.gate = GatedLinearUnit(unified_output, unified_output * 2, dropout)

        # Auxiliary outputs
        self.aux_sentiment = nn.Linear(unified_output, 1)
        self.aux_risk = nn.Linear(unified_output, 1)
        self.aux_regime = nn.Linear(unified_output, num_regimes)

    def forward(
        self,
        # Raw data tensors (batch, seq_len, feature_dim)
        microstructure_data: torch.Tensor,
        derivatives_data: torch.Tensor,
        onchain_data: torch.Tensor,
        event_data: torch.Tensor,
        # Asset context
        asset_ids: torch.Tensor,
        dominance: Optional[torch.Tensor] = None,
        # Optional pre-computed embeddings
        precomputed_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        # Flags
        return_components: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through unified encoder.

        Args:
            microstructure_data: Market microstructure data (batch, seq, dim)
            derivatives_data: Derivatives market data (batch, seq, dim)
            onchain_data: On-chain data (batch, seq, dim)
            event_data: Event features (batch, seq, dim)
            asset_ids: Asset IDs (batch,)
            dominance: Market dominance (batch, 3)
            precomputed_embeddings: Optional pre-computed encoder outputs
            return_components: Return individual encoder outputs
            return_aux: Return auxiliary predictions

        Returns:
            Unified state embedding (batch, unified_output)
        """
        B = microstructure_data.shape[0]
        device = microstructure_data.device

        # Encode each modality
        if precomputed_embeddings is not None:
            order_flow_emb = precomputed_embeddings.get("order_flow")
            liquidity_emb = precomputed_embeddings.get("liquidity")
            derivatives_emb = precomputed_embeddings.get("derivatives")
            onchain_emb = precomputed_embeddings.get("onchain")
            event_emb = precomputed_embeddings.get("events")
            asset_emb = precomputed_embeddings.get("asset")
        else:
            # Order flow encoding
            order_flow_emb = self.order_flow_encoder(microstructure_data)

            # Liquidity encoding (reshape microstructure data)
            liquidity_emb = self.liquidity_encoder(microstructure_data)

            # Derivatives encoding
            derivatives_emb = self.derivatives_encoder(derivatives_data)

            # On-chain encoding
            onchain_emb = self.onchain_encoder(onchain_data)

            # Event encoding (use aggregated features)
            event_emb = self.event_encoder.encode_aggregated_features(
                event_data[:, -1, :] if event_data.dim() == 3 else event_data
            )

            # Asset encoding
            asset_emb = self.asset_encoder(asset_ids, dominance=dominance)

        # Collect all embeddings except regime (for regime input)
        embeddings = [
            order_flow_emb, liquidity_emb, derivatives_emb,
            onchain_emb, event_emb, asset_emb
        ]

        # Cross-modal attention
        modal_embeds = []
        for proj, emb in zip(self.modal_projections, embeddings):
            modal_embeds.append(proj(emb))

        modal_stack = torch.stack(modal_embeds, dim=1)  # (B, 6, 128)
        attended, _ = self.cross_modal_attention(modal_stack, modal_stack, modal_stack)
        attended = attended.mean(dim=1)  # (B, 128)

        # Concatenate for regime encoder input
        pre_regime = torch.cat(embeddings, dim=-1)

        # Regime encoding (depends on other encodings)
        regime_emb, regime_probs, volatility = self.regime_encoder(
            pre_regime.unsqueeze(1), return_probs=True
        )

        # Full concatenation
        all_embeddings = torch.cat([*embeddings, regime_emb], dim=-1)

        # Apply dominance conditioning
        if dominance is not None:
            all_embeddings = self.dominance_conditioner(all_embeddings, dominance)

        # Fusion
        fused = self.fusion(all_embeddings)
        fused = self.fusion_transformer(fused.unsqueeze(1)).squeeze(1)
        output = self.output_proj(fused)

        # Gated residual
        output = self.gate(output)

        # Collect outputs
        if return_components or return_aux:
            components = {
                "order_flow": order_flow_emb,
                "liquidity": liquidity_emb,
                "derivatives": derivatives_emb,
                "onchain": onchain_emb,
                "events": event_emb,
                "asset": asset_emb,
                "regime": regime_emb,
                "regime_probs": regime_probs,
                "volatility": volatility,
                "cross_modal": attended,
            }

        if return_aux:
            aux = {
                "sentiment": torch.tanh(self.aux_sentiment(output)),
                "risk": torch.sigmoid(self.aux_risk(output)),
                "regime_logits": self.aux_regime(output),
            }

            if return_components:
                return output, components, aux
            return output, aux

        if return_components:
            return output, components

        return output

    def encode_for_world_model(
        self,
        microstructure_data: torch.Tensor,
        derivatives_data: torch.Tensor,
        onchain_data: torch.Tensor,
        event_data: torch.Tensor,
        asset_ids: torch.Tensor,
        dominance: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Encode state for world model input.

        Returns both unified state and component states for
        world model to model transitions.
        """
        return self.forward(
            microstructure_data, derivatives_data, onchain_data,
            event_data, asset_ids, dominance,
            return_components=True,
        )

    def encode_for_policy(
        self,
        microstructure_data: torch.Tensor,
        derivatives_data: torch.Tensor,
        onchain_data: torch.Tensor,
        event_data: torch.Tensor,
        asset_ids: torch.Tensor,
        dominance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode state for policy network input.

        Returns compact unified state.
        """
        return self.forward(
            microstructure_data, derivatives_data, onchain_data,
            event_data, asset_ids, dominance,
            return_components=False,
            return_aux=False,
        )


class MultiAssetStateEncoder(nn.Module):
    """
    Encodes states for multiple assets simultaneously.

    Captures cross-asset dependencies.
    """

    def __init__(
        self,
        num_assets: int = 100,
        unified_encoder: UnifiedStateEncoder = None,
        cross_asset_hidden: int = 256,
        output_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_assets = num_assets
        self.unified_encoder = unified_encoder or UnifiedStateEncoder()
        self.output_dim = output_dim

        # Cross-asset attention
        self.cross_asset_attention = nn.MultiheadAttention(
            self.unified_encoder.unified_output,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

        # Graph neural network for asset dependencies
        self.gnn = nn.Sequential(
            nn.Linear(self.unified_encoder.unified_output * 2, cross_asset_hidden),
            nn.GELU(),
            nn.Linear(cross_asset_hidden, self.unified_encoder.unified_output),
        )

        # Output projection
        self.output_proj = nn.Linear(
            self.unified_encoder.unified_output * 2, output_dim
        )

    def forward(
        self,
        asset_states: Dict[str, Dict[str, torch.Tensor]],
        query_asset: str,
        adjacency: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for multi-asset encoding.

        Args:
            asset_states: Dict of asset name to data dict
            query_asset: Asset to generate state for
            adjacency: Optional cross-asset adjacency matrix

        Returns:
            Cross-asset aware state for query asset
        """
        # Encode all assets
        asset_embeddings = {}
        for asset_name, data in asset_states.items():
            emb = self.unified_encoder(**data)
            asset_embeddings[asset_name] = emb

        # Stack all embeddings
        asset_names = list(asset_embeddings.keys())
        embeddings = torch.stack([asset_embeddings[n] for n in asset_names], dim=1)

        # Get query embedding
        query_idx = asset_names.index(query_asset)
        query_emb = embeddings[:, query_idx:query_idx+1, :]

        # Cross-asset attention
        attended, _ = self.cross_asset_attention(query_emb, embeddings, embeddings)

        # GNN message passing if adjacency provided
        if adjacency is not None:
            messages = []
            for i, name in enumerate(asset_names):
                if name != query_asset:
                    msg = torch.cat([query_emb.squeeze(1), embeddings[:, i, :]], dim=-1)
                    msg = self.gnn(msg) * adjacency[:, query_idx, i:i+1]
                    messages.append(msg)
            if messages:
                aggregated = torch.stack(messages, dim=1).mean(dim=1)
                attended = attended.squeeze(1) + aggregated

        # Combine query and cross-asset info
        output = torch.cat([query_emb.squeeze(1), attended.squeeze(1)], dim=-1)
        output = self.output_proj(output)

        return output
