"""Configuration management for CryptoAI."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
from omegaconf import OmegaConf, DictConfig
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class HardwareConfig(BaseModel):
    """Hardware configuration."""

    gpus: list[int] = Field(default=[0, 1])
    gpu_memory_fraction: float = Field(default=0.9, ge=0.0, le=1.0)
    cpu_threads: int = Field(default=32, ge=1)
    ram_limit_gb: int = Field(default=120, ge=1)
    use_ddp: bool = Field(default=True)
    use_amp: bool = Field(default=True)


class SystemConfig(BaseModel):
    """System configuration."""

    name: str = Field(default="CryptoAI")
    version: str = Field(default="0.1.0")
    environment: str = Field(default="development")
    seed: int = Field(default=42)
    device: str = Field(default="cuda")
    precision: str = Field(default="amp")
    num_workers: int = Field(default=8)
    log_level: str = Field(default="INFO")


class EnvSettings(BaseSettings):
    """Environment settings loaded from .env file."""

    # Exchange API Keys
    binance_api_key: Optional[str] = None
    binance_api_secret: Optional[str] = None
    okx_api_key: Optional[str] = None
    okx_api_secret: Optional[str] = None
    okx_passphrase: Optional[str] = None
    bybit_api_key: Optional[str] = None
    bybit_api_secret: Optional[str] = None

    # On-chain API Keys
    etherscan_api_key: Optional[str] = None
    infura_project_id: Optional[str] = None
    alchemy_api_key: Optional[str] = None

    # Monitoring
    wandb_api_key: Optional[str] = None
    slack_webhook_url: Optional[str] = None

    # Database
    redis_url: str = Field(default="redis://localhost:6379")
    duckdb_path: str = Field(default="./data/cryptoai.duckdb")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


def load_config(
    config_path: Union[str, Path] = "configs/default.yaml",
    overrides: Optional[Dict[str, Any]] = None,
) -> DictConfig:
    """
    Load configuration from YAML file with optional overrides.

    Args:
        config_path: Path to YAML configuration file
        overrides: Dictionary of configuration overrides

    Returns:
        OmegaConf DictConfig object
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load base configuration
    config = OmegaConf.load(config_path)

    # Apply overrides if provided
    if overrides:
        override_config = OmegaConf.create(overrides)
        config = OmegaConf.merge(config, override_config)

    # Resolve interpolations
    OmegaConf.resolve(config)

    return config


def load_env_settings() -> EnvSettings:
    """Load environment settings from .env file."""
    return EnvSettings()


def save_config(config: DictConfig, path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, path)


def get_config_hash(config: DictConfig) -> str:
    """Get a hash of the configuration for versioning."""
    import hashlib

    config_str = OmegaConf.to_yaml(config)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]
