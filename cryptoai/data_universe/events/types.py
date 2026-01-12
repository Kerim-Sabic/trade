"""Event type definitions."""

from enum import Enum


class EventCategory(str, Enum):
    """High-level event categories."""

    EXCHANGE = "exchange"
    PROTOCOL = "protocol"
    SECURITY = "security"
    REGULATORY = "regulatory"
    MACRO = "macro"
    SOCIAL = "social"
    TECHNICAL = "technical"


class EventType(str, Enum):
    """Specific event types."""

    # Exchange events
    LISTING_ANNOUNCEMENT = "listing_announcement"
    DELISTING_ANNOUNCEMENT = "delisting_announcement"
    TRADING_PAIR_ADDED = "trading_pair_added"
    TRADING_HALTED = "trading_halted"
    EXCHANGE_MAINTENANCE = "exchange_maintenance"
    EXCHANGE_HACK = "exchange_hack"
    WITHDRAWAL_SUSPENDED = "withdrawal_suspended"

    # Protocol events
    MAINNET_LAUNCH = "mainnet_launch"
    PROTOCOL_UPGRADE = "protocol_upgrade"
    HARD_FORK = "hard_fork"
    TOKEN_BURN = "token_burn"
    AIRDROP = "airdrop"
    STAKING_LAUNCH = "staking_launch"
    GOVERNANCE_VOTE = "governance_vote"

    # Security events
    EXPLOIT_DETECTED = "exploit_detected"
    VULNERABILITY_DISCLOSED = "vulnerability_disclosed"
    SECURITY_AUDIT = "security_audit"
    SMART_CONTRACT_BUG = "smart_contract_bug"
    BRIDGE_HACK = "bridge_hack"
    RUG_PULL = "rug_pull"

    # Regulatory events
    REGULATORY_APPROVAL = "regulatory_approval"
    REGULATORY_WARNING = "regulatory_warning"
    LEGAL_ACTION = "legal_action"
    BAN_ANNOUNCEMENT = "ban_announcement"
    TAX_GUIDANCE = "tax_guidance"
    ETF_NEWS = "etf_news"

    # Macro events
    FED_DECISION = "fed_decision"
    INFLATION_DATA = "inflation_data"
    EMPLOYMENT_DATA = "employment_data"
    MARKET_CRASH = "market_crash"
    BANK_FAILURE = "bank_failure"

    # Social/Development events
    PARTNERSHIP_ANNOUNCEMENT = "partnership_announcement"
    TEAM_UPDATE = "team_update"
    GITHUB_RELEASE = "github_release"
    WHALE_MOVEMENT = "whale_movement"
    INFLUENCER_POST = "influencer_post"

    # Technical events
    HALVING = "halving"
    DIFFICULTY_ADJUSTMENT = "difficulty_adjustment"
    NETWORK_CONGESTION = "network_congestion"
    HASH_RATE_CHANGE = "hash_rate_change"


# Event impact mappings
EVENT_IMPACT_DEFAULTS = {
    # High impact positive
    EventType.LISTING_ANNOUNCEMENT: (0.7, 0.8),  # (direction, magnitude)
    EventType.ETF_NEWS: (0.6, 0.9),
    EventType.MAINNET_LAUNCH: (0.5, 0.6),
    EventType.REGULATORY_APPROVAL: (0.6, 0.7),

    # High impact negative
    EventType.EXCHANGE_HACK: (-0.8, 0.9),
    EventType.EXPLOIT_DETECTED: (-0.7, 0.8),
    EventType.DELISTING_ANNOUNCEMENT: (-0.6, 0.7),
    EventType.RUG_PULL: (-0.9, 0.95),
    EventType.BAN_ANNOUNCEMENT: (-0.7, 0.8),

    # Medium impact
    EventType.PROTOCOL_UPGRADE: (0.3, 0.5),
    EventType.PARTNERSHIP_ANNOUNCEMENT: (0.4, 0.5),
    EventType.TOKEN_BURN: (0.3, 0.4),
    EventType.AIRDROP: (0.2, 0.5),
    EventType.GITHUB_RELEASE: (0.2, 0.3),

    # Low/Neutral impact
    EventType.EXCHANGE_MAINTENANCE: (0.0, 0.2),
    EventType.GOVERNANCE_VOTE: (0.1, 0.3),
    EventType.SECURITY_AUDIT: (0.2, 0.3),
}


# Event persistence (half-life in hours)
EVENT_PERSISTENCE = {
    EventType.LISTING_ANNOUNCEMENT: 168,  # 1 week
    EventType.EXCHANGE_HACK: 336,  # 2 weeks
    EventType.EXPLOIT_DETECTED: 168,
    EventType.REGULATORY_APPROVAL: 720,  # 1 month
    EventType.BAN_ANNOUNCEMENT: 720,
    EventType.ETF_NEWS: 720,
    EventType.PROTOCOL_UPGRADE: 72,
    EventType.AIRDROP: 24,
    EventType.GITHUB_RELEASE: 48,
    EventType.INFLUENCER_POST: 12,
}
