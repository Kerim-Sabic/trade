"""Asset identity encoder with learned embeddings."""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from cryptoai.encoders.base import BaseEncoder, TransformerBlock


class AssetEncoder(BaseEncoder):
    """
    Learnable asset identity encoder.

    Creates unique representations for each asset that capture:
    - Asset-specific behavior patterns
    - Cross-asset relationships
    - Sector/category membership
    """

    def __init__(
        self,
        num_assets: int = 100,
        embedding_dim: int = 64,
        num_sectors: int = 20,
        num_tags: int = 50,
        hidden_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__(embedding_dim, output_dim, dropout)

        self.num_assets = num_assets
        self.embedding_dim = embedding_dim

        # Core asset embedding
        self.asset_embedding = nn.Embedding(num_assets, embedding_dim)

        # Sector embedding
        self.sector_embedding = nn.Embedding(num_sectors, embedding_dim // 4)

        # Tag embeddings (multi-hot)
        self.tag_embedding = nn.Embedding(num_tags, embedding_dim // 4)

        # Asset characteristics projection
        self.characteristics_proj = nn.Sequential(
            nn.Linear(12, embedding_dim // 2),  # 12 characteristic features
            nn.LayerNorm(embedding_dim // 2),
            nn.GELU(),
        )

        # Combine all asset information
        total_dim = embedding_dim + embedding_dim // 4 + embedding_dim // 4 + embedding_dim // 2
        self.combine = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Dominance conditioning
        self.dominance_proj = nn.Linear(3, embedding_dim // 4)  # BTC, ETH, altcoin dominance

        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embeddings."""
        nn.init.normal_(self.asset_embedding.weight, std=0.02)
        nn.init.normal_(self.sector_embedding.weight, std=0.02)
        nn.init.normal_(self.tag_embedding.weight, std=0.02)

    def forward(
        self,
        asset_ids: torch.Tensor,
        sector_ids: Optional[torch.Tensor] = None,
        tag_ids: Optional[torch.Tensor] = None,
        characteristics: Optional[torch.Tensor] = None,
        dominance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            asset_ids: Asset IDs (batch,)
            sector_ids: Sector IDs (batch,)
            tag_ids: Tag IDs (batch, num_tags) - multi-hot encoded
            characteristics: Asset characteristics (batch, 12)
            dominance: Market dominance (batch, 3)

        Returns:
            Asset representation
        """
        B = asset_ids.shape[0]
        device = asset_ids.device

        # Core embedding
        asset_emb = self.asset_embedding(asset_ids.clamp(0, self.num_assets - 1))

        # Sector embedding
        if sector_ids is not None:
            sector_emb = self.sector_embedding(sector_ids)
        else:
            sector_emb = torch.zeros(B, self.embedding_dim // 4, device=device)

        # Tag embedding (average of active tags)
        if tag_ids is not None:
            # tag_ids shape: (batch, max_tags) with -1 for padding
            valid_mask = tag_ids >= 0
            tag_ids_clamped = tag_ids.clamp(0)
            tag_embs = self.tag_embedding(tag_ids_clamped)
            tag_embs = tag_embs * valid_mask.unsqueeze(-1).float()
            tag_emb = tag_embs.sum(dim=1) / (valid_mask.sum(dim=1, keepdim=True).float() + 1e-6)
        else:
            tag_emb = torch.zeros(B, self.embedding_dim // 4, device=device)

        # Characteristics projection
        if characteristics is not None:
            char_emb = self.characteristics_proj(characteristics)
        else:
            char_emb = torch.zeros(B, self.embedding_dim // 2, device=device)

        # Combine
        combined = torch.cat([asset_emb, sector_emb, tag_emb, char_emb], dim=-1)
        output = self.combine(combined)

        # Add dominance conditioning
        if dominance is not None:
            dom_emb = self.dominance_proj(dominance)
            output = output + F.pad(dom_emb, (0, output.size(-1) - dom_emb.size(-1)))

        return output

    def get_embedding(self, asset_id: int) -> torch.Tensor:
        """Get raw embedding for an asset."""
        return self.asset_embedding.weight[asset_id]

    def get_all_embeddings(self) -> torch.Tensor:
        """Get all asset embeddings."""
        return self.asset_embedding.weight


class CrossAssetEncoder(BaseEncoder):
    """
    Encoder for cross-asset relationships.

    Learns how assets interact and influence each other.
    """

    def __init__(
        self,
        num_assets: int = 100,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__(embedding_dim, output_dim, dropout)

        self.num_assets = num_assets

        # Asset embeddings (shared with AssetEncoder if used together)
        self.asset_embedding = nn.Embedding(num_assets, embedding_dim)

        # Cross-asset attention
        self.cross_attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Graph neural network layer
        self.gnn_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # Output
        self.output_proj = nn.Linear(embedding_dim, output_dim)

    def forward(
        self,
        query_assets: torch.Tensor,
        context_assets: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            query_assets: Query asset IDs (batch,)
            context_assets: Context asset IDs (batch, num_context)
            adjacency: Optional adjacency weights (batch, num_context)

        Returns:
            Cross-asset aware representation
        """
        B = query_assets.shape[0]

        # Get embeddings
        query_emb = self.asset_embedding(query_assets).unsqueeze(1)  # (B, 1, D)
        context_emb = self.asset_embedding(context_assets)  # (B, N, D)

        # Cross-attention
        attended, attn_weights = self.cross_attention(
            query_emb, context_emb, context_emb
        )

        # Apply adjacency weights if provided
        if adjacency is not None:
            attended = attended * adjacency.unsqueeze(-1)

        # GNN-style message passing
        query_expanded = query_emb.expand_as(context_emb)
        messages = torch.cat([query_expanded, context_emb], dim=-1)
        aggregated = self.gnn_layer(messages).mean(dim=1)

        # Combine
        output = attended.squeeze(1) + aggregated
        output = self.output_proj(output)

        return output


class DominanceConditioner(nn.Module):
    """
    Conditions representations based on market dominance.

    "This coin behaves differently from others based on BTC/ETH dominance."
    """

    def __init__(
        self,
        input_dim: int = 64,
        dominance_dim: int = 3,  # BTC, ETH, altcoin
        hidden_dim: int = 32,
    ):
        super().__init__()

        # Dominance encoding
        self.dominance_encoder = nn.Sequential(
            nn.Linear(dominance_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim * 2),  # scale and shift
        )

    def forward(
        self,
        x: torch.Tensor,
        dominance: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply dominance-based conditioning.

        Args:
            x: Input features (batch, input_dim)
            dominance: Dominance values (batch, 3)

        Returns:
            Conditioned features
        """
        # FiLM-style conditioning
        params = self.dominance_encoder(dominance)
        scale, shift = params.chunk(2, dim=-1)

        # Apply affine transformation
        output = x * (1 + scale) + shift

        return output
