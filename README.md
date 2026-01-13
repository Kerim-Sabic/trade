# CryptoAI - Autonomous Crypto Trading Intelligence

A production-grade autonomous crypto trading system using deep reinforcement learning, multi-source data aggregation, and sophisticated risk management.

## System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS 12+, or Windows 10/11
- **Python**: 3.10, 3.11, or 3.12
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 20GB for dependencies + data

### For GPU Training (Optional but Recommended)
- **GPU**: NVIDIA GPU with CUDA 12.1 support
- **VRAM**: 8GB minimum, 16GB+ recommended
- **Driver**: NVIDIA Driver 525+ with CUDA 12.1

### Check GPU Status
```bash
nvidia-smi
nvcc --version
```

---

## Quick Start (5 Minutes)

### Step 1: Clone Repository
```bash
git clone https://github.com/your-org/cryptoai.git
cd cryptoai
```

### Step 2: Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n cryptoai python=3.11 -y
conda activate cryptoai

# OR using venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
```

### Step 3: Install Dependencies

**CPU Only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

**With CUDA (GPU):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

### Step 4: Verify Installation
```bash
# Check system status
cryptoai status

# Or run directly
python -m cryptoai.main --mode shadow --help
```

### Step 5: Run in Shadow Mode (Demo)
```bash
# Shadow mode - simulates trading without real execution
cryptoai run --mode shadow --assets BTCUSDT

# Or use Python directly
python -m cryptoai.main --mode shadow --config configs/default.yaml --assets BTCUSDT
```

---

## Installation Methods

### Method 1: pip install (Recommended)
```bash
pip install -e ".[dev]"  # Includes development tools
```

### Method 2: Docker
```bash
cd docker
docker-compose up -d trading-ai
```

### Method 3: Manual Setup
```bash
pip install torch>=2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scipy scikit-learn
pip install transformers gymnasium ccxt
pip install loguru rich typer fastapi uvicorn
pip install -e .
```

---

## Usage

### CLI Commands

```bash
# Show system status and GPU info
cryptoai status

# Run in shadow mode (simulation)
cryptoai run --mode shadow --assets BTCUSDT ETHUSDT

# Run backtesting
cryptoai backtest --assets BTCUSDT --start 2023-01-01 --end 2024-01-01

# Start distributed training
cryptoai train --config configs/default.yaml --gpus 2

# Start monitoring dashboard
cryptoai dashboard --port 8081

# Show version
cryptoai version
```

### Python API

```python
from cryptoai.utils.config import load_config
from cryptoai.utils.logging import setup_logging
from cryptoai.encoders import UnifiedStateEncoder
from cryptoai.decision_engine import PolicyNetwork
from cryptoai.world_model import WorldModel
from cryptoai.black_swan import BlackSwanDetector

# Setup
setup_logging(level="INFO")
config = load_config("configs/default.yaml")

# Initialize models
encoder = UnifiedStateEncoder(
    microstructure_dim=40,
    derivatives_dim=42,
    onchain_dim=33,
    event_dim=15,
)

policy = PolicyNetwork(
    state_dim=128,
    action_dim=4,
    hidden_dim=256,
)
```

---

## Execution Modes

| Mode | Description | Real Trades | Capital Risk |
|------|-------------|-------------|--------------|
| `shadow` | Full simulation, no API calls | No | None |
| `paper` | Testnet execution with real API | No | None |
| `live` | Real trading with capital at risk | Yes | Real |
| `backtest` | Historical simulation | No | None |
| `train` | Distributed model training | No | None |

### Shadow Mode (Recommended for Testing)
```bash
cryptoai run --mode shadow --assets BTCUSDT
```

### Backtesting
```bash
cryptoai backtest --assets BTCUSDT ETHUSDT --capital 100000
```

### Paper Trading (Testnet)
Requires testnet API keys in config:
```yaml
# configs/default.yaml
exchange:
  api_key: "YOUR_TESTNET_API_KEY"
  api_secret: "YOUR_TESTNET_API_SECRET"
```

```bash
cryptoai run --mode paper --assets BTCUSDT
```

---

## Configuration

### Main Config: `configs/default.yaml`

```yaml
# System settings
system:
  device: "auto"  # auto, cuda, cpu
  precision: "amp"  # fp32, fp16, bf16, amp
  log_level: "INFO"

# Model dimensions
model:
  state_dim: 200
  hidden_dim: 256
  latent_dim: 128
  action_dim: 4

# Risk management
risk:
  max_leverage: 5
  max_drawdown: 0.15
  daily_loss_limit: 0.05
  kill_switch_enabled: true

# Backtesting
backtesting:
  initial_capital: 100000
  days: 365

# Exchange (for paper/live modes)
exchange:
  api_key: ""
  api_secret: ""
```

### Environment Variables

```bash
# Optional: Override config values
export CRYPTOAI_LOG_LEVEL=DEBUG
export CRYPTOAI_DEVICE=cuda
export CUDA_VISIBLE_DEVICES=0,1
```

---

## Architecture

```
cryptoai/
├── data_universe/     # Multi-source data aggregation
│   ├── market_microstructure/  # L2 orderbook, trades
│   ├── derivatives/   # Funding rates, open interest
│   ├── onchain/       # Exchange flows, whale tracking
│   └── events/        # News, sentiment analysis
│
├── encoders/          # Neural state encoders
│   ├── unified.py     # Combined encoder
│   ├── order_flow.py  # Transformer for microstructure
│   └── derivatives.py # LSTM for derivatives data
│
├── world_model/       # Market dynamics prediction
│   └── temporal_transformer.py
│
├── decision_engine/   # Hierarchical RL
│   ├── meta_controller.py  # Regime-aware controller
│   └── policy.py      # SAC policy network
│
├── black_swan/        # Tail risk detection
│   ├── detector.py    # VAE + Isolation Forest
│   └── tail_risk.py   # Extreme value theory
│
├── risk_engine/       # Risk management
│   ├── controller.py  # Position limits
│   └── kill_switch.py # Emergency stop
│
├── execution/         # Order management
│   └── exchange_client.py  # Binance/OKX/Bybit
│
├── backtesting/       # Historical simulation
│   ├── engine.py      # Event-driven backtest
│   └── simulator.py   # Market friction model
│
├── training/          # Distributed training
│   ├── trainer.py     # Unified trainer
│   └── ddp.py         # Multi-GPU support
│
└── monitoring/        # Observability
    ├── dashboard.py   # Web UI
    └── drift_detector.py  # Distribution shift
```

---

## Training

### Generate Synthetic Data (for Testing)
```bash
cryptoai train --config configs/default.yaml --synthetic
```

### Distributed Training (Multi-GPU)
```bash
# Auto-detect GPUs
cryptoai train --config configs/default.yaml

# Specify GPU count
cryptoai train --config configs/default.yaml --gpus 2
```

### Using torchrun
```bash
torchrun --nproc_per_node=2 scripts/run_training.py \
    --config configs/default.yaml \
    --data-dir data/
```

---

## Monitoring Dashboard

Start the web dashboard:
```bash
cryptoai dashboard --port 8081
```

Access at: http://localhost:8081

Features:
- Real-time metrics visualization
- Drift detection status
- Active alerts management
- System health overview

---

## Docker Deployment

### Quick Start with Docker
```bash
cd docker
docker-compose up -d trading-ai
```

### Full Stack (with Prometheus + Grafana)
```bash
cd docker
docker-compose up -d
```

Services:
- `trading-ai`: Main trading service (port 8080, 8081)
- `prometheus`: Metrics collection (port 9090)
- `grafana`: Visualization (port 3000)
- `redis`: Caching (port 6379)

---

## Common Errors & Solutions

### Error: `ModuleNotFoundError: No module named 'cryptoai'`
```bash
# Ensure you're in the project root and installed in editable mode
cd /path/to/cryptoai
pip install -e .
```

### Error: `CUDA out of memory`
```bash
# Reduce batch size in config or use CPU
cryptoai run --mode shadow  # Uses config defaults
# Or set environment variable
export CUDA_VISIBLE_DEVICES=""  # Force CPU
```

### Error: `torch.cuda.is_available() returns False`
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall PyTorch with CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Error: `Connection refused` on dashboard
```bash
# Check if port is in use
lsof -i :8081

# Use different port
cryptoai dashboard --port 8082
```

### Error: `Exchange API authentication failed`
- Ensure API keys are set in `configs/default.yaml`
- For paper trading, use testnet API keys
- Check API key permissions (need trading access)

### Error: `h5py not found`
```bash
pip install h5py>=3.10.0
```

---

## Development

### Run Tests
```bash
pytest tests/ -v
```

### Code Formatting
```bash
black cryptoai/
ruff check cryptoai/
```

### Type Checking
```bash
mypy cryptoai/
```

---

## Project Structure

```
trade/
├── cryptoai/           # Main Python package
├── configs/            # YAML configurations
│   └── default.yaml
├── docker/             # Docker deployment
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── prometheus.yml
├── scripts/            # Utility scripts
│   └── run_training.py
├── pyproject.toml      # Package configuration
└── README.md           # This file
```

---

## API Reference

### Core Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `UnifiedStateEncoder` | `cryptoai.encoders` | Encodes market state |
| `PolicyNetwork` | `cryptoai.decision_engine` | RL policy |
| `WorldModel` | `cryptoai.world_model` | Dynamics prediction |
| `BlackSwanDetector` | `cryptoai.black_swan` | Anomaly detection |
| `RiskController` | `cryptoai.risk_engine` | Risk management |
| `BacktestEngine` | `cryptoai.backtesting` | Historical simulation |
| `TradingExecutor` | `cryptoai.execution` | Order execution |

---

## License

MIT License - See LICENSE file for details.

---

## Support

- **Issues**: https://github.com/your-org/cryptoai/issues
- **Documentation**: See `/docs` directory

---

## Disclaimer

This software is for educational and research purposes only. Cryptocurrency trading carries substantial risk of loss. Always test thoroughly with paper trading before using real capital. The authors are not responsible for any financial losses.
