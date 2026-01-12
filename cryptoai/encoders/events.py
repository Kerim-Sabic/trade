"""Event and narrative encoder using transformer-based architecture."""

from typing import Dict, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from cryptoai.encoders.base import (
    BaseEncoder,
    TransformerBlock,
    GatedLinearUnit,
)


class EventEncoder(BaseEncoder):
    """
    Transformer-based encoder for event and narrative data.

    In production, this would use a fine-tuned crypto-specific language model.
    This implementation uses learned embeddings for event types and
    numeric features.
    """

    def __init__(
        self,
        num_event_types: int = 30,
        num_categories: int = 7,
        numeric_features: int = 15,  # Event feature dimension
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        max_events: int = 20,  # Max concurrent events
        dropout: float = 0.1,
    ):
        super().__init__(numeric_features, output_dim, dropout)

        self.num_event_types = num_event_types
        self.num_categories = num_categories
        self.max_events = max_events
        self.hidden_dim = hidden_dim

        # Event type embedding
        self.type_embedding = nn.Embedding(num_event_types + 1, hidden_dim // 2)

        # Category embedding
        self.category_embedding = nn.Embedding(num_categories + 1, hidden_dim // 4)

        # Numeric feature projection
        self.numeric_proj = nn.Linear(numeric_features, hidden_dim // 4)

        # Combine embeddings
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Transformer for event interaction
        self.transformer = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Temporal decay attention
        self.decay_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Auxiliary heads for event understanding
        self.aux_direction_head = nn.Linear(output_dim, 1)  # Directional bias
        self.aux_volatility_head = nn.Linear(output_dim, 1)  # Volatility impact
        self.aux_persistence_head = nn.Linear(output_dim, 1)  # Impact persistence
        self.aux_credibility_head = nn.Linear(output_dim, 1)  # Source credibility

        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

    def forward(
        self,
        event_types: torch.Tensor,
        categories: torch.Tensor,
        features: torch.Tensor,
        decay_weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            event_types: Event type IDs (batch, num_events)
            categories: Category IDs (batch, num_events)
            features: Numeric features (batch, num_events, numeric_features)
            decay_weights: Temporal decay weights (batch, num_events)
            mask: Event mask for padding (batch, num_events)
            return_aux: Return auxiliary predictions

        Returns:
            Encoded event representation
        """
        B, N, _ = features.shape

        # Embed events
        type_emb = self.type_embedding(event_types.clamp(0, self.num_event_types))
        cat_emb = self.category_embedding(categories.clamp(0, self.num_categories))
        num_emb = self.numeric_proj(features)

        # Combine embeddings
        combined = torch.cat([type_emb, cat_emb, num_emb], dim=-1)
        x = self.combine(combined)

        # Apply decay weights if provided
        if decay_weights is not None:
            x = x * decay_weights.unsqueeze(-1)

        # Add CLS token
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=B)
        x = torch.cat([cls_tokens, x], dim=1)

        # Update mask if provided
        if mask is not None:
            cls_mask = torch.ones(B, 1, device=mask.device, dtype=mask.dtype)
            mask = torch.cat([cls_mask, mask], dim=1)

        # Transformer processing
        for block in self.transformer:
            x = block(x)

        # Get CLS representation
        cls_output = x[:, 0, :]

        # Output projection
        output = self.output_proj(cls_output)

        if return_aux:
            aux_outputs = {
                "direction": torch.tanh(self.aux_direction_head(output)),
                "volatility": F.softplus(self.aux_volatility_head(output)),
                "persistence": F.softplus(self.aux_persistence_head(output)),
                "credibility": torch.sigmoid(self.aux_credibility_head(output)),
            }
            return output, aux_outputs

        return output

    def encode_aggregated_features(
        self,
        features: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor:
        """
        Encode pre-aggregated event features.

        For use when events are already aggregated into a feature vector.

        Args:
            features: Aggregated event features (batch, feature_dim)

        Returns:
            Encoded representation
        """
        # Simple MLP encoding for aggregated features
        B = features.shape[0]

        # Create dummy event sequence
        x = features.unsqueeze(1).expand(-1, 1, -1)

        # Project to hidden
        x = self.numeric_proj(x)
        x = F.pad(x, (0, self.hidden_dim - x.size(-1)))

        # Add CLS
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=B)
        x = torch.cat([cls_tokens, x], dim=1)

        # Transformer
        for block in self.transformer:
            x = block(x)

        cls_output = x[:, 0, :]
        output = self.output_proj(cls_output)

        if return_aux:
            aux_outputs = {
                "direction": torch.tanh(self.aux_direction_head(output)),
                "volatility": F.softplus(self.aux_volatility_head(output)),
                "persistence": F.softplus(self.aux_persistence_head(output)),
                "credibility": torch.sigmoid(self.aux_credibility_head(output)),
            }
            return output, aux_outputs

        return output


class CryptoBERTEncoder(BaseEncoder):
    """
    Placeholder for a fine-tuned BERT model for crypto news.

    In production, this would load a pre-trained model like:
    - FinBERT fine-tuned on crypto data
    - Custom crypto-BERT trained on crypto news corpus
    """

    def __init__(
        self,
        pretrained_name: str = "ProsusAI/finbert",
        output_dim: int = 128,
        max_length: int = 512,
        freeze_base: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__(768, output_dim, dropout)  # BERT hidden size

        self.max_length = max_length
        self.freeze_base = freeze_base

        # Placeholder - in production, load actual model
        # self.bert = AutoModel.from_pretrained(pretrained_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

        # Simulated BERT output projection
        self.projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
        )

        # Auxiliary heads
        self.sentiment_head = nn.Linear(output_dim, 3)  # neg, neu, pos
        self.relevance_head = nn.Linear(output_dim, 1)  # crypto relevance

    def forward(
        self,
        text_embeddings: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            text_embeddings: Pre-computed text embeddings (batch, 768)
                            In production, this would be input_ids and attention_mask
            return_aux: Return auxiliary predictions

        Returns:
            Encoded text representation
        """
        output = self.projection(text_embeddings)

        if return_aux:
            aux_outputs = {
                "sentiment": F.softmax(self.sentiment_head(output), dim=-1),
                "relevance": torch.sigmoid(self.relevance_head(output)),
            }
            return output, aux_outputs

        return output

    def encode_text(
        self,
        texts: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Encode raw text strings.

        In production, this would tokenize and run through BERT.
        """
        # Placeholder - returns random embeddings
        B = len(texts)
        embeddings = torch.randn(B, 768, device=device)
        return self.forward(embeddings)


class EventTemporalEncoder(BaseEncoder):
    """
    Encode temporal dynamics of events.

    Captures how event impact evolves over time.
    """

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 64,
        output_dim: int = 32,
        sequence_length: int = 24,  # Hours
        dropout: float = 0.1,
    ):
        super().__init__(input_dim, output_dim, dropout)

        self.sequence_length = sequence_length

        # GRU for temporal dynamics
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # Decay modeling
        self.decay_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

        # Output
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_decay: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Event features over time (batch, seq_len, input_dim)
            return_decay: Return estimated decay rate

        Returns:
            Encoded temporal dynamics
        """
        _, h_n = self.gru(x)
        h_n = h_n[-1]  # Last layer hidden state

        output = self.output_proj(h_n)

        if return_decay:
            decay = self.decay_net(h_n)
            return output, decay

        return output
