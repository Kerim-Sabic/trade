"""Event processing and feature extraction."""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from loguru import logger

from cryptoai.data_universe.base import EventData
from cryptoai.data_universe.events.types import (
    EventType,
    EventCategory,
    EVENT_IMPACT_DEFAULTS,
    EVENT_PERSISTENCE,
)


@dataclass
class ProcessedEvent:
    """Processed event with extracted features."""

    raw_event: EventData

    # Classification
    event_type: EventType
    category: EventCategory
    confidence: float  # Classification confidence

    # Impact assessment
    directional_bias: float  # -1 to 1
    impact_magnitude: float  # 0 to 1
    volatility_impact: float  # Expected volatility multiplier

    # Temporal
    persistence_half_life: float  # Hours
    current_weight: float  # Decayed weight

    # Credibility
    source_credibility: float
    is_verified: bool

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.directional_bias,
            self.impact_magnitude,
            self.volatility_impact,
            self.persistence_half_life / 168,  # Normalize to weeks
            self.current_weight,
            self.source_credibility,
            float(self.is_verified),
        ], dtype=np.float32)


@dataclass
class EventFeatures:
    """Aggregated event features for an asset."""

    timestamp: datetime
    asset: str

    # Active event counts
    positive_event_count: int
    negative_event_count: int
    neutral_event_count: int

    # Weighted impacts
    net_directional_bias: float  # Weighted average -1 to 1
    total_impact_magnitude: float  # Sum of magnitudes
    expected_volatility_multiplier: float

    # Category breakdown
    category_impacts: Dict[str, float]

    # Recent significant events
    most_impactful_event: Optional[ProcessedEvent]

    # Risk indicators
    event_risk_score: float  # 0-1, higher = more uncertainty
    narrative_momentum: float  # -1 to 1, trend of event sentiment

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        # Category impacts as fixed-size array
        categories = list(EventCategory)
        cat_array = np.array([
            self.category_impacts.get(cat.value, 0.0) for cat in categories
        ], dtype=np.float32)

        base_array = np.array([
            self.positive_event_count,
            self.negative_event_count,
            self.neutral_event_count,
            self.net_directional_bias,
            self.total_impact_magnitude,
            self.expected_volatility_multiplier,
            self.event_risk_score,
            self.narrative_momentum,
        ], dtype=np.float32)

        return np.concatenate([base_array, cat_array])


class EventProcessor:
    """Process and analyze events for trading intelligence."""

    def __init__(
        self,
        history_hours: int = 720,  # 30 days
        credibility_threshold: float = 0.5,
        verified_sources: Set[str] = None,
    ):
        self.history_hours = history_hours
        self.credibility_threshold = credibility_threshold
        self.verified_sources = verified_sources or set()

        # Event history per asset
        self._event_history: Dict[str, deque] = {}

        # Global event history (for market-wide events)
        self._global_events: deque = deque(maxlen=10000)

    def classify_event(self, event: EventData) -> Tuple[EventType, EventCategory, float]:
        """
        Classify event type and category.

        In production, this would use a fine-tuned language model.
        Here we use keyword matching as a placeholder.
        """
        content_lower = (event.title + " " + event.content).lower()

        # Simple keyword-based classification (placeholder for ML model)
        classifications = []

        # Exchange events
        if any(kw in content_lower for kw in ["listing", "listed", "will list"]):
            classifications.append((EventType.LISTING_ANNOUNCEMENT, EventCategory.EXCHANGE, 0.8))
        if any(kw in content_lower for kw in ["delist", "remove", "suspended"]):
            classifications.append((EventType.DELISTING_ANNOUNCEMENT, EventCategory.EXCHANGE, 0.8))
        if any(kw in content_lower for kw in ["hack", "breach", "stolen"]):
            classifications.append((EventType.EXCHANGE_HACK, EventCategory.SECURITY, 0.9))

        # Protocol events
        if any(kw in content_lower for kw in ["upgrade", "v2", "v3", "release"]):
            classifications.append((EventType.PROTOCOL_UPGRADE, EventCategory.PROTOCOL, 0.7))
        if any(kw in content_lower for kw in ["airdrop", "distribution"]):
            classifications.append((EventType.AIRDROP, EventCategory.PROTOCOL, 0.8))
        if any(kw in content_lower for kw in ["fork", "chain split"]):
            classifications.append((EventType.HARD_FORK, EventCategory.PROTOCOL, 0.85))

        # Security events
        if any(kw in content_lower for kw in ["exploit", "vulnerability", "bug"]):
            classifications.append((EventType.EXPLOIT_DETECTED, EventCategory.SECURITY, 0.85))
        if any(kw in content_lower for kw in ["rug", "scam", "exit"]):
            classifications.append((EventType.RUG_PULL, EventCategory.SECURITY, 0.9))

        # Regulatory events
        if any(kw in content_lower for kw in ["sec", "regulation", "regulatory"]):
            classifications.append((EventType.LEGAL_ACTION, EventCategory.REGULATORY, 0.7))
        if any(kw in content_lower for kw in ["etf", "approval", "approved"]):
            classifications.append((EventType.ETF_NEWS, EventCategory.REGULATORY, 0.8))
        if any(kw in content_lower for kw in ["ban", "prohibit", "illegal"]):
            classifications.append((EventType.BAN_ANNOUNCEMENT, EventCategory.REGULATORY, 0.85))

        # Social events
        if any(kw in content_lower for kw in ["partnership", "collaborate", "integrate"]):
            classifications.append((EventType.PARTNERSHIP_ANNOUNCEMENT, EventCategory.SOCIAL, 0.75))

        # Default classification
        if not classifications:
            return EventType.GOVERNANCE_VOTE, EventCategory.PROTOCOL, 0.3

        # Return highest confidence classification
        return max(classifications, key=lambda x: x[2])

    def assess_impact(
        self,
        event_type: EventType,
        event: EventData,
    ) -> Tuple[float, float, float]:
        """
        Assess event impact.

        Returns (directional_bias, magnitude, volatility_impact).
        """
        # Get default impact
        default = EVENT_IMPACT_DEFAULTS.get(event_type, (0.0, 0.3))
        direction, magnitude = default

        # Adjust based on credibility
        credibility = event.credibility_score
        magnitude *= credibility

        # Adjust based on verification
        if event.is_verified:
            magnitude *= 1.2
        else:
            magnitude *= 0.7

        # Volatility impact based on magnitude
        volatility_impact = 1.0 + magnitude * 2.0  # 1x to 3x

        return direction, min(magnitude, 1.0), volatility_impact

    def process_event(self, event: EventData) -> ProcessedEvent:
        """
        Process a raw event into structured features.

        Args:
            event: Raw event data

        Returns:
            Processed event with extracted features
        """
        # Classify event
        event_type, category, confidence = self.classify_event(event)

        # Assess impact
        direction, magnitude, volatility = self.assess_impact(event_type, event)

        # Get persistence
        half_life = EVENT_PERSISTENCE.get(event_type, 48)

        # Check if source is verified
        is_verified = event.is_verified or event.source in self.verified_sources

        # Source credibility
        source_cred = event.credibility_score
        if is_verified:
            source_cred = max(source_cred, 0.8)

        processed = ProcessedEvent(
            raw_event=event,
            event_type=event_type,
            category=category,
            confidence=confidence,
            directional_bias=direction,
            impact_magnitude=magnitude,
            volatility_impact=volatility,
            persistence_half_life=half_life,
            current_weight=1.0,  # Will decay over time
            source_credibility=source_cred,
            is_verified=is_verified,
        )

        # Store in history
        self._store_event(processed)

        return processed

    def _store_event(self, event: ProcessedEvent) -> None:
        """Store processed event in history."""
        # Store for affected assets
        for asset in event.raw_event.affected_assets:
            asset_key = asset.upper()
            if asset_key not in self._event_history:
                self._event_history[asset_key] = deque(maxlen=10000)
            self._event_history[asset_key].append(event)

        # Store in global history
        self._global_events.append(event)

    def get_features(
        self,
        asset: str,
        window_hours: float = 24.0,
    ) -> EventFeatures:
        """
        Get aggregated event features for an asset.

        Args:
            asset: Asset symbol
            window_hours: Time window for aggregation

        Returns:
            Aggregated event features
        """
        asset_key = asset.upper()

        if asset_key not in self._event_history:
            return self._empty_features(asset)

        now = datetime.utcnow()
        cutoff = now - timedelta(hours=window_hours)

        # Get relevant events and update weights
        active_events: List[ProcessedEvent] = []
        for event in self._event_history[asset_key]:
            if event.raw_event.timestamp >= cutoff:
                # Update weight based on decay
                age_hours = (now - event.raw_event.timestamp).total_seconds() / 3600
                decay = 0.5 ** (age_hours / event.persistence_half_life)
                event.current_weight = decay
                active_events.append(event)

        if len(active_events) == 0:
            return self._empty_features(asset)

        # Count by direction
        positive = [e for e in active_events if e.directional_bias > 0.1]
        negative = [e for e in active_events if e.directional_bias < -0.1]
        neutral = [e for e in active_events if abs(e.directional_bias) <= 0.1]

        # Weighted impacts
        total_weight = sum(e.current_weight * e.impact_magnitude for e in active_events)
        if total_weight > 0:
            net_bias = sum(
                e.directional_bias * e.current_weight * e.impact_magnitude
                for e in active_events
            ) / total_weight
        else:
            net_bias = 0.0

        total_magnitude = sum(e.impact_magnitude * e.current_weight for e in active_events)

        # Volatility multiplier (product, capped)
        vol_mult = 1.0
        for e in active_events:
            vol_mult *= (1 + (e.volatility_impact - 1) * e.current_weight)
        vol_mult = min(vol_mult, 5.0)

        # Category breakdown
        category_impacts: Dict[str, float] = {}
        for cat in EventCategory:
            cat_events = [e for e in active_events if e.category == cat]
            if cat_events:
                impact = sum(
                    e.directional_bias * e.impact_magnitude * e.current_weight
                    for e in cat_events
                )
                category_impacts[cat.value] = impact

        # Most impactful event
        most_impactful = max(
            active_events,
            key=lambda e: e.impact_magnitude * e.current_weight,
        ) if active_events else None

        # Event risk score (more events = more uncertainty)
        event_risk = min(1.0, len(active_events) / 10 * np.mean([
            e.impact_magnitude for e in active_events
        ]))

        # Narrative momentum (trend of sentiment)
        momentum = self._calculate_momentum(active_events)

        return EventFeatures(
            timestamp=now,
            asset=asset,
            positive_event_count=len(positive),
            negative_event_count=len(negative),
            neutral_event_count=len(neutral),
            net_directional_bias=net_bias,
            total_impact_magnitude=total_magnitude,
            expected_volatility_multiplier=vol_mult,
            category_impacts=category_impacts,
            most_impactful_event=most_impactful,
            event_risk_score=event_risk,
            narrative_momentum=momentum,
        )

    def _empty_features(self, asset: str) -> EventFeatures:
        """Return empty features when no events."""
        return EventFeatures(
            timestamp=datetime.utcnow(),
            asset=asset,
            positive_event_count=0,
            negative_event_count=0,
            neutral_event_count=0,
            net_directional_bias=0.0,
            total_impact_magnitude=0.0,
            expected_volatility_multiplier=1.0,
            category_impacts={},
            most_impactful_event=None,
            event_risk_score=0.0,
            narrative_momentum=0.0,
        )

    def _calculate_momentum(self, events: List[ProcessedEvent]) -> float:
        """Calculate momentum of event sentiment over time."""
        if len(events) < 2:
            return 0.0

        # Sort by time
        sorted_events = sorted(events, key=lambda e: e.raw_event.timestamp)

        # Split into two halves
        mid = len(sorted_events) // 2
        first_half = sorted_events[:mid]
        second_half = sorted_events[mid:]

        # Average sentiment of each half
        first_sentiment = np.mean([e.directional_bias for e in first_half])
        second_sentiment = np.mean([e.directional_bias for e in second_half])

        return second_sentiment - first_sentiment

    def get_global_features(
        self,
        window_hours: float = 24.0,
    ) -> EventFeatures:
        """Get market-wide event features."""
        return self.get_features("GLOBAL", window_hours)

    def get_feature_tensor(
        self,
        asset: str,
        sequence_length: int = 100,
        window_hours: float = 1.0,
    ) -> np.ndarray:
        """
        Get event features as a time series tensor.

        Args:
            asset: Asset symbol
            sequence_length: Number of time steps
            window_hours: Window for each time step

        Returns:
            Tensor of shape (sequence_length, feature_dim)
        """
        feature_dim = 8 + len(EventCategory)  # base features + category impacts
        tensor = np.zeros((sequence_length, feature_dim), dtype=np.float32)

        # Generate features for each time step
        now = datetime.utcnow()
        for i in range(sequence_length):
            # Calculate time offset for this step
            hours_ago = (sequence_length - 1 - i) * window_hours
            # This is a simplified version - in production, use actual historical windows
            features = self.get_features(asset, window_hours=window_hours)
            tensor[i] = features.to_array()

        return tensor
