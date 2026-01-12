# CryptoAI: Autonomous Crypto Trading Intelligence

A comprehensive, self-learning trading intelligence system for cryptocurrency markets. Built with PyTorch, designed for dual RTX 5080 GPUs with distributed training support.

## Architecture Overview

```
                                    +------------------+
                                    |   Data Universe  |
                                    |  - Microstructure|
                                    |  - Derivatives   |
                                    |  - On-Chain      |
                                    |  - Events        |
                                    +--------+---------+
                                             |
                                    +--------v---------+
                                    |  Neural Encoders |
                                    |  - Order Flow    |
                                    |  - Liquidity     |
                                    |  - Derivatives   |
                                    |  - On-Chain      |
                                    |  - Events        |
                                    +--------+---------+
                                             |
                    +------------------------+------------------------+
                    |                        |                        |
           +--------v--------+      +--------v--------+      +--------v--------+
           |   World Model   |      | Black Swan Det. |      |  Risk Engine    |
           | - Temporal Trans|      | - Tail Risk     |      | - Kill Switch   |
           | - Latent Dyn.   |      | - Anomaly Det.  |      | - Position Ctrl |
           +-----------------+      +-----------------+      +--------+--------+
                    |                        |                        |
                    +------------------------+------------------------+
                                             |
                                    +--------v---------+
                                    | Decision Engine  |
                                    | - Meta Controller|
                                    | - Policy Network |
                                    | - PPO/SAC        |
                                    +--------+---------+
                                             |
                                    +--------v---------+
                                    |    Execution     |
                                    | - Order Manager  |
                                    | - Exchange APIs  |
                                    +------------------+
```

## Features

### Data Universe
- **Market Microstructure**: Tick-level trades, L2 order book, spread/depth analysis
- **Derivatives Intelligence**: Funding rates, open interest, liquidations, basis
- **On-Chain Analytics**: Exchange flows, whale tracking, stablecoin dynamics
- **Event Processing**: Crypto-native event classification and impact estimation

### Neural State Encoders
- Order flow encoder (CNN/Transformer)
- Liquidity dynamics encoder
- Derivatives state encoder
- On-chain temporal encoder
- Event/narrative encoder
- Asset identity embeddings
- Regime detection encoder

### World Model
- Temporal Transformer with latent stochastic states
- Causal discovery for market structure learning
- Multi-step prediction with uncertainty quantification
- Counterfactual reasoning support

### Decision Engine
- Hierarchical RL with meta-controller
- PPO and SAC policy implementations
- Continuous action space (direction, size, leverage, timing)
- Multi-objective reward shaping

### Black Swan Intelligence
- Tail risk estimation using EVT
- VAE-based anomaly detection
- Liquidation cascade prediction
- Regime break detection

### Risk Management
- Volatility-scaled exposure control
- Liquidity-aware position sizing
- Kill switch with emergency protocols
- Event-aware leverage caps

### Backtesting
- Event-driven simulation
- Realistic execution modeling (fees, slippage, partial fills)
- Walk-forward validation
- Comprehensive metrics (Sharpe, CVaR, drawdown)

## Installation

### Requirements
- Python 3.11+
- PyTorch 2.0+ with CUDA 12.1
- 2x NVIDIA RTX 5080 GPUs (or similar)
- 128GB RAM recommended

### Setup

```bash
# Clone repository
git clone https://github.com/your-repo/cryptoai.git
cd cryptoai

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
```

### Docker

```bash
# Build image
docker build -t cryptoai -f docker/Dockerfile .

# Run with GPU support
docker-compose -f docker/docker-compose.yml up trading-ai
```

## Usage

### Training

```bash
# Run distributed training on 2 GPUs
python scripts/run_training.py --config configs/default.yaml --gpus 2

# With synthetic data for testing
python scripts/run_training.py --synthetic --gpus 2
```

### Backtesting

```bash
python -m cryptoai.main --mode backtest --config configs/default.yaml --assets BTCUSDT ETHUSDT
```

### Live Trading (Shadow Mode)

```bash
python -m cryptoai.main --mode shadow --model checkpoints/best_model.pt --assets BTCUSDT
```

### Paper Trading

```bash
python -m cryptoai.main --mode paper --model checkpoints/best_model.pt
```

## Configuration

See `configs/default.yaml` for full configuration options:

```yaml
model:
  state_dim: 200
  hidden_dim: 256
  latent_dim: 128
  action_dim: 4

training:
  batch_size: 64
  learning_rate: 0.0003
  encoder_epochs: 10
  world_model_epochs: 50
  policy_epochs: 100

risk:
  max_position_pct: 0.1
  max_leverage: 5
  stop_loss_pct: 0.02
  max_drawdown_pct: 0.1

execution:
  mode: shadow
  order_type: limit
  max_slippage_bps: 50
```

## Project Structure

```
cryptoai/
├── data_universe/          # Data ingestion and processing
│   ├── market_microstructure/
│   ├── derivatives/
│   ├── onchain/
│   ├── events/
│   └── asset_registry/
├── encoders/               # Neural state encoders
├── world_model/            # Market dynamics model
├── decision_engine/        # RL policy and meta-controller
├── black_swan/             # Tail risk detection
├── risk_engine/            # Risk management
├── backtesting/            # Backtesting engine
├── training/               # Training pipelines with DDP
├── deployment/             # Model serving and rollout
├── monitoring/             # Metrics, drift detection, alerting
├── execution/              # Exchange APIs and order management
└── utils/                  # Configuration, logging, device utils

configs/                    # Configuration files
docker/                     # Docker and compose files
scripts/                    # Training and utility scripts
```

## Failure Mode Analysis

### Critical Failure Modes

| Failure Mode | Probability | Impact | Mitigation |
|--------------|-------------|--------|------------|
| Exchange API failure | Medium | High | Multi-exchange fallback, local order book |
| Model inference timeout | Low | High | Timeout handling, fallback to safe action |
| Data feed interruption | Medium | Medium | Buffering, stale data detection, safe mode |
| GPU OOM | Low | High | Gradient checkpointing, batch size adaptation |
| Liquidation cascade | Low | Critical | Black swan detector, emergency flatten |
| Drift detection failure | Low | High | Multiple drift metrics, manual override |

### Recovery Procedures

1. **Exchange Failure**
   - Automatic failover to backup exchange
   - Position reconciliation on reconnect
   - Alert notification

2. **Model Failure**
   - Fallback to previous stable checkpoint
   - Reduce position sizes
   - Shadow mode until validated

3. **Data Feed Failure**
   - Use cached data with staleness warning
   - Increase risk thresholds
   - Manual intervention alert

4. **Black Swan Event**
   - Kill switch activation
   - Emergency position flatten
   - System pause until human review

### Monitoring Alerts

- **Critical**: System errors, position limits exceeded, kill switch activated
- **Warning**: High latency, drift detected, error rate elevated
- **Info**: Model updates, position changes, daily reports

## API Endpoints

### Model Server (port 8080)

- `POST /infer` - Run inference on state
- `GET /health` - Health check
- `GET /metrics` - Performance metrics

### Dashboard (port 8081)

- `GET /` - Web dashboard
- `GET /api/metrics` - All metrics
- `GET /api/alerts` - Active alerts
- `GET /api/drift` - Drift status
- `WS /ws` - Real-time updates

## Performance Benchmarks

| Metric | Target | Achieved |
|--------|--------|----------|
| Inference latency (p99) | <100ms | ~50ms |
| Training throughput | >1000 samples/s | ~1500 samples/s |
| Model size | <500MB | ~300MB |
| GPU memory (per GPU) | <12GB | ~10GB |

## Ethics and Disclaimers

- **No profit guarantees**: This system provides no guarantee of profitability
- **Risk warning**: Cryptocurrency trading involves significant risk of loss
- **Audit logs**: All decisions and trades are logged for transparency
- **Conservative approach**: System defaults to risk-averse behavior
- **Human oversight**: Designed for human supervision, not fully autonomous trading

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with tests

## License

MIT License - See LICENSE file for details
