"""Deployment orchestrator for managing model lifecycle."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from enum import Enum
import json
import threading
import time
import torch
from loguru import logger


class DeploymentState(Enum):
    """Deployment states."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class DeploymentConfig:
    """Configuration for deployment."""

    # Model settings
    model_path: str = "models/production"
    checkpoint_path: str = "checkpoints/best_model.pt"

    # Execution settings
    execution_mode: str = "shadow"  # shadow, paper, live
    max_position_usd: float = 10000.0
    max_leverage: float = 3.0

    # Risk settings
    max_drawdown_pct: float = 10.0
    daily_loss_limit_pct: float = 5.0
    position_timeout_hours: float = 24.0

    # Infrastructure
    gpu_id: int = 0
    num_workers: int = 4
    inference_batch_size: int = 32

    # Monitoring
    health_check_interval: float = 60.0
    metrics_export_interval: float = 10.0
    alert_channels: List[str] = field(default_factory=lambda: ["log"])

    # Failsafe
    auto_pause_on_error: bool = True
    max_consecutive_errors: int = 5


class DeploymentOrchestrator:
    """
    Orchestrates model deployment and lifecycle management.

    Handles:
    - Model loading and versioning
    - State management (run, pause, stop)
    - Health monitoring
    - Graceful degradation
    - A/B testing support
    """

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.state = DeploymentState.INITIALIZING
        self._lock = threading.Lock()

        # Model registry
        self._models: Dict[str, Any] = {}
        self._active_model: Optional[str] = None

        # Metrics
        self._error_count = 0
        self._inference_count = 0
        self._start_time: Optional[datetime] = None

        # Health monitoring
        self._health_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Initialize
        self._initialize()

    def _initialize(self) -> None:
        """Initialize deployment."""
        logger.info("Initializing deployment orchestrator")

        # Set device
        self.device = torch.device(
            f"cuda:{self.config.gpu_id}"
            if torch.cuda.is_available()
            else "cpu"
        )

        logger.info(f"Using device: {self.device}")

        # Load models
        self._load_models()

        # Start health monitoring
        self._start_health_monitor()

        self.state = DeploymentState.RUNNING
        self._start_time = datetime.utcnow()

        logger.info("Deployment orchestrator initialized")

    def _load_models(self) -> None:
        """Load models from checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_path)

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Register model
            model_id = f"model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            self._models[model_id] = {
                "checkpoint": checkpoint,
                "loaded_at": datetime.utcnow(),
                "version": checkpoint.get("version", "unknown"),
            }
            self._active_model = model_id

            logger.info(f"Loaded model: {model_id}")

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            self.state = DeploymentState.ERROR

    def _start_health_monitor(self) -> None:
        """Start health monitoring thread."""
        self._health_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True,
        )
        self._health_thread.start()

    def _health_monitor_loop(self) -> None:
        """Health monitoring loop."""
        while not self._stop_event.is_set():
            try:
                health = self.check_health()

                if not health["healthy"]:
                    logger.warning(f"Health check failed: {health['issues']}")

                    if self.config.auto_pause_on_error:
                        self._handle_unhealthy()

            except Exception as e:
                logger.error(f"Health monitor error: {e}")

            time.sleep(self.config.health_check_interval)

    def _handle_unhealthy(self) -> None:
        """Handle unhealthy state."""
        self._error_count += 1

        if self._error_count >= self.config.max_consecutive_errors:
            logger.error("Max consecutive errors reached, pausing deployment")
            self.pause()

    def check_health(self) -> Dict[str, Any]:
        """Check deployment health."""
        issues = []

        # Check state
        if self.state == DeploymentState.ERROR:
            issues.append("Deployment in error state")

        # Check GPU
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated(self.config.gpu_id)
                gpu_total = torch.cuda.get_device_properties(self.config.gpu_id).total_memory

                if gpu_memory / gpu_total > 0.95:
                    issues.append("GPU memory near capacity")

            except Exception as e:
                issues.append(f"GPU health check failed: {e}")

        # Check model
        if self._active_model is None:
            issues.append("No active model")

        return {
            "healthy": len(issues) == 0,
            "issues": issues,
            "state": self.state.value,
            "active_model": self._active_model,
            "inference_count": self._inference_count,
            "error_count": self._error_count,
            "uptime_seconds": (datetime.utcnow() - self._start_time).total_seconds()
            if self._start_time else 0,
        }

    def run(self) -> None:
        """Start/resume deployment."""
        with self._lock:
            if self.state in [DeploymentState.PAUSED, DeploymentState.STOPPED]:
                self.state = DeploymentState.RUNNING
                self._error_count = 0
                logger.info("Deployment started")

    def pause(self) -> None:
        """Pause deployment."""
        with self._lock:
            if self.state == DeploymentState.RUNNING:
                self.state = DeploymentState.PAUSED
                logger.info("Deployment paused")

    def stop(self) -> None:
        """Stop deployment."""
        with self._lock:
            self.state = DeploymentState.STOPPED
            self._stop_event.set()
            logger.info("Deployment stopped")

    def load_model(self, checkpoint_path: str, model_id: Optional[str] = None) -> str:
        """Load a new model version."""
        if model_id is None:
            model_id = f"model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            with self._lock:
                self._models[model_id] = {
                    "checkpoint": checkpoint,
                    "loaded_at": datetime.utcnow(),
                    "version": checkpoint.get("version", "unknown"),
                }

            logger.info(f"Loaded model: {model_id}")
            return model_id

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def switch_model(self, model_id: str) -> None:
        """Switch to a different model version."""
        with self._lock:
            if model_id not in self._models:
                raise ValueError(f"Model not found: {model_id}")

            old_model = self._active_model
            self._active_model = model_id

            logger.info(f"Switched model: {old_model} -> {model_id}")

    def get_active_model(self) -> Optional[Dict[str, Any]]:
        """Get the active model info."""
        if self._active_model is None:
            return None
        return self._models.get(self._active_model)

    def increment_inference(self) -> None:
        """Increment inference counter."""
        self._inference_count += 1

    def record_error(self, error: Exception) -> None:
        """Record an error."""
        self._error_count += 1
        logger.error(f"Inference error: {error}")

        if self._error_count >= self.config.max_consecutive_errors:
            if self.config.auto_pause_on_error:
                self.pause()

    def get_metrics(self) -> Dict[str, Any]:
        """Get deployment metrics."""
        return {
            "state": self.state.value,
            "active_model": self._active_model,
            "models_loaded": list(self._models.keys()),
            "inference_count": self._inference_count,
            "error_count": self._error_count,
            "uptime_seconds": (datetime.utcnow() - self._start_time).total_seconds()
            if self._start_time else 0,
        }

    def export_state(self, path: str) -> None:
        """Export deployment state for recovery."""
        state = {
            "config": self.config.__dict__,
            "state": self.state.value,
            "active_model": self._active_model,
            "metrics": self.get_metrics(),
            "timestamp": datetime.utcnow().isoformat(),
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)

        logger.info(f"Exported state to {path}")


class DeploymentManager:
    """
    High-level deployment manager for multiple environments.

    Manages:
    - Shadow deployment (paper trading)
    - A/B testing
    - Gradual rollout
    - Rollback
    """

    def __init__(self):
        self._deployments: Dict[str, DeploymentOrchestrator] = {}

    def create_deployment(
        self,
        name: str,
        config: DeploymentConfig,
    ) -> DeploymentOrchestrator:
        """Create a new deployment."""
        if name in self._deployments:
            raise ValueError(f"Deployment already exists: {name}")

        deployment = DeploymentOrchestrator(config)
        self._deployments[name] = deployment

        logger.info(f"Created deployment: {name}")
        return deployment

    def get_deployment(self, name: str) -> Optional[DeploymentOrchestrator]:
        """Get deployment by name."""
        return self._deployments.get(name)

    def list_deployments(self) -> List[str]:
        """List all deployments."""
        return list(self._deployments.keys())

    def stop_all(self) -> None:
        """Stop all deployments."""
        for name, deployment in self._deployments.items():
            logger.info(f"Stopping deployment: {name}")
            deployment.stop()

    def get_all_health(self) -> Dict[str, Dict]:
        """Get health status of all deployments."""
        return {
            name: deployment.check_health()
            for name, deployment in self._deployments.items()
        }
