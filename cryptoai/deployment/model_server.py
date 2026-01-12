"""Model serving infrastructure for real-time inference."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import asyncio
import queue
import threading
import time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from loguru import logger


@dataclass
class InferenceConfig:
    """Configuration for inference engine."""

    # Batching
    max_batch_size: int = 32
    batch_timeout_ms: float = 10.0

    # Hardware
    device: str = "cuda:0"
    use_amp: bool = True
    compile_model: bool = True  # PyTorch 2.0 compile

    # Caching
    enable_cache: bool = True
    cache_ttl_seconds: float = 1.0
    max_cache_size: int = 1000

    # Performance
    num_inference_threads: int = 2
    warmup_iterations: int = 10


class InferenceRequest:
    """Single inference request."""

    def __init__(
        self,
        request_id: str,
        state: np.ndarray,
        metadata: Optional[Dict] = None,
    ):
        self.request_id = request_id
        self.state = state
        self.metadata = metadata or {}
        self.timestamp = time.time()
        self.result: Optional[Dict] = None
        self.event = threading.Event()

    def set_result(self, result: Dict) -> None:
        """Set the result and signal completion."""
        self.result = result
        self.event.set()

    def wait(self, timeout: float = 10.0) -> Optional[Dict]:
        """Wait for result."""
        if self.event.wait(timeout=timeout):
            return self.result
        return None


class InferenceEngine:
    """
    High-performance inference engine.

    Features:
    - Dynamic batching
    - Request queuing
    - AMP inference
    - Result caching
    - Model warmup
    """

    def __init__(
        self,
        encoder: nn.Module,
        policy: nn.Module,
        world_model: nn.Module,
        black_swan: nn.Module,
        config: InferenceConfig,
    ):
        self.config = config
        self.device = torch.device(config.device)

        # Models
        self.encoder = encoder.to(self.device).eval()
        self.policy = policy.to(self.device).eval()
        self.world_model = world_model.to(self.device).eval()
        self.black_swan = black_swan.to(self.device).eval()

        # Compile models if enabled (PyTorch 2.0+)
        if config.compile_model and hasattr(torch, 'compile'):
            logger.info("Compiling models for optimized inference")
            self.encoder = torch.compile(self.encoder, mode="reduce-overhead")
            self.policy = torch.compile(self.policy, mode="reduce-overhead")

        # Request queue
        self._request_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()

        # Cache
        self._cache: Dict[str, tuple] = {}  # hash -> (result, timestamp)
        self._cache_lock = threading.Lock()

        # Metrics
        self._inference_count = 0
        self._cache_hits = 0
        self._total_latency = 0.0

        # Start inference threads
        self._threads: List[threading.Thread] = []
        self._start_inference_threads()

        # Warmup
        self._warmup()

    def _start_inference_threads(self) -> None:
        """Start inference worker threads."""
        for i in range(self.config.num_inference_threads):
            thread = threading.Thread(
                target=self._inference_worker,
                daemon=True,
                name=f"inference_worker_{i}",
            )
            thread.start()
            self._threads.append(thread)

        logger.info(f"Started {len(self._threads)} inference threads")

    def _inference_worker(self) -> None:
        """Worker thread for processing inference requests."""
        batch: List[InferenceRequest] = []
        batch_deadline = None

        while not self._stop_event.is_set():
            try:
                # Try to get a request
                timeout = 0.001  # 1ms
                if batch_deadline:
                    timeout = max(0.001, batch_deadline - time.time())

                try:
                    request = self._request_queue.get(timeout=timeout)
                    batch.append(request)

                    if batch_deadline is None:
                        batch_deadline = time.time() + self.config.batch_timeout_ms / 1000

                except queue.Empty:
                    pass

                # Process batch if ready
                should_process = (
                    len(batch) >= self.config.max_batch_size or
                    (batch_deadline and time.time() >= batch_deadline)
                )

                if batch and should_process:
                    self._process_batch(batch)
                    batch = []
                    batch_deadline = None

            except Exception as e:
                logger.error(f"Inference worker error: {e}")

    def _process_batch(self, batch: List[InferenceRequest]) -> None:
        """Process a batch of inference requests."""
        start_time = time.time()

        try:
            # Stack states
            states = np.stack([req.state for req in batch])
            states_tensor = torch.from_numpy(states).float().to(self.device)

            # Run inference
            with torch.no_grad():
                with autocast(enabled=self.config.use_amp):
                    results = self._run_inference(states_tensor)

            # Distribute results
            for i, request in enumerate(batch):
                result = {
                    key: value[i].cpu().numpy() if isinstance(value, torch.Tensor) else value[i]
                    for key, value in results.items()
                }
                request.set_result(result)

                # Cache result
                if self.config.enable_cache:
                    self._cache_result(request.state, result)

            # Update metrics
            latency = (time.time() - start_time) * 1000
            self._inference_count += len(batch)
            self._total_latency += latency

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            # Return error to all requests
            for request in batch:
                request.set_result({"error": str(e)})

    def _run_inference(self, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run full inference pipeline."""
        # Encode state
        z = self.encoder(states)

        # Get policy action
        action_dist, value = self.policy(z)
        action_mean = action_dist.mean
        action_std = action_dist.stddev

        # Get black swan risk
        anomaly_scores, _ = self.black_swan(z)

        # World model prediction (optional, for planning)
        # z_next, _, _ = self.world_model(z, action_mean)

        return {
            "action_mean": action_mean,
            "action_std": action_std,
            "value": value,
            "anomaly_score": anomaly_scores,
            "latent_state": z[:, -1] if z.dim() > 2 else z,
        }

    def _cache_result(self, state: np.ndarray, result: Dict) -> None:
        """Cache inference result."""
        cache_key = hash(state.tobytes())

        with self._cache_lock:
            # Evict old entries if cache is full
            if len(self._cache) >= self.config.max_cache_size:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]

            self._cache[cache_key] = (result, time.time())

    def _check_cache(self, state: np.ndarray) -> Optional[Dict]:
        """Check if result is cached."""
        if not self.config.enable_cache:
            return None

        cache_key = hash(state.tobytes())

        with self._cache_lock:
            if cache_key in self._cache:
                result, timestamp = self._cache[cache_key]

                # Check TTL
                if time.time() - timestamp < self.config.cache_ttl_seconds:
                    self._cache_hits += 1
                    return result.copy()

                # Expired
                del self._cache[cache_key]

        return None

    def _warmup(self) -> None:
        """Warmup models with dummy data."""
        logger.info("Warming up inference engine")

        dummy_state = np.random.randn(
            self.config.max_batch_size,
            100,  # sequence length
            200,  # feature dim
        ).astype(np.float32)

        dummy_tensor = torch.from_numpy(dummy_state).to(self.device)

        for i in range(self.config.warmup_iterations):
            with torch.no_grad():
                with autocast(enabled=self.config.use_amp):
                    _ = self._run_inference(dummy_tensor)

        logger.info("Warmup complete")

    def infer(
        self,
        state: np.ndarray,
        timeout: float = 10.0,
    ) -> Optional[Dict]:
        """
        Run inference on a single state.

        Args:
            state: Input state array
            timeout: Maximum wait time

        Returns:
            Inference results or None on timeout
        """
        # Check cache first
        cached = self._check_cache(state)
        if cached is not None:
            return cached

        # Create request
        request = InferenceRequest(
            request_id=f"{time.time()}_{id(state)}",
            state=state,
        )

        # Queue request
        self._request_queue.put(request)

        # Wait for result
        return request.wait(timeout=timeout)

    async def infer_async(
        self,
        state: np.ndarray,
        timeout: float = 10.0,
    ) -> Optional[Dict]:
        """Async inference interface."""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.infer(state, timeout),
        )

    def get_metrics(self) -> Dict[str, float]:
        """Get inference metrics."""
        avg_latency = (
            self._total_latency / self._inference_count
            if self._inference_count > 0
            else 0
        )

        cache_hit_rate = (
            self._cache_hits / self._inference_count
            if self._inference_count > 0
            else 0
        )

        return {
            "inference_count": self._inference_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "avg_latency_ms": avg_latency,
            "queue_size": self._request_queue.qsize(),
        }

    def stop(self) -> None:
        """Stop inference engine."""
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=5.0)
        logger.info("Inference engine stopped")


class ModelServer:
    """
    HTTP/gRPC model server for external access.

    Provides REST API for inference requests.
    """

    def __init__(
        self,
        inference_engine: InferenceEngine,
        host: str = "0.0.0.0",
        port: int = 8080,
    ):
        self.engine = inference_engine
        self.host = host
        self.port = port
        self._app = None

    def _setup_routes(self) -> None:
        """Setup API routes."""
        try:
            from fastapi import FastAPI, HTTPException
            from pydantic import BaseModel
            import uvicorn

            self._app = FastAPI(title="CryptoAI Trading Model Server")

            class InferenceRequest(BaseModel):
                state: List[List[float]]

            class InferenceResponse(BaseModel):
                action_mean: List[float]
                action_std: List[float]
                value: float
                anomaly_score: float

            @self._app.post("/infer", response_model=InferenceResponse)
            async def infer(request: InferenceRequest):
                state = np.array(request.state, dtype=np.float32)

                if state.ndim == 2:
                    state = state[np.newaxis, ...]

                result = await self.engine.infer_async(state)

                if result is None:
                    raise HTTPException(status_code=500, detail="Inference timeout")

                if "error" in result:
                    raise HTTPException(status_code=500, detail=result["error"])

                return InferenceResponse(
                    action_mean=result["action_mean"].tolist(),
                    action_std=result["action_std"].tolist(),
                    value=float(result["value"]),
                    anomaly_score=float(result["anomaly_score"]),
                )

            @self._app.get("/health")
            async def health():
                return {"status": "healthy", "metrics": self.engine.get_metrics()}

            @self._app.get("/metrics")
            async def metrics():
                return self.engine.get_metrics()

            self._uvicorn = uvicorn
            logger.info("FastAPI routes configured")

        except ImportError:
            logger.warning("FastAPI not installed, HTTP server disabled")

    def start(self) -> None:
        """Start the model server."""
        self._setup_routes()

        if self._app and hasattr(self, '_uvicorn'):
            self._uvicorn.run(
                self._app,
                host=self.host,
                port=self.port,
                log_level="info",
            )

    async def start_async(self) -> None:
        """Start server asynchronously."""
        self._setup_routes()

        if self._app and hasattr(self, '_uvicorn'):
            config = self._uvicorn.Config(
                self._app,
                host=self.host,
                port=self.port,
                log_level="info",
            )
            server = self._uvicorn.Server(config)
            await server.serve()
