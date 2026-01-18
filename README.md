# CryptoAI Trading Platform

**Production-Grade ML Crypto Trading System for Windows 11**

A professional machine learning system for cryptocurrency trading with institutional-grade risk management, walk-forward validation, and regime-aware decision making.

---

## Quick Start (5 Minutes)

### Prerequisites (Windows 11)

- **Windows 11 x64** (Windows 10 also supported)
- **Python 3.9, 3.10, or 3.11** (3.11 recommended)
- **Git** (optional, for cloning)
- **Node.js 18+** (only for Electron desktop app)

### Step 1: Install Python

1. Download Python from [python.org](https://www.python.org/downloads/)
2. **IMPORTANT**: Check "Add Python to PATH" during installation
3. Verify installation:
   ```powershell
   python --version
   # Should show: Python 3.11.x (or 3.9.x/3.10.x)
   ```

### Step 2: Clone or Download

```powershell
# Option A: Clone with Git
git clone https://github.com/Kerim-Sabic/trade.git
cd trade

# Option B: Download ZIP and extract, then cd into directory
```

### Step 3: Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU-only, fastest)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install CryptoAI
pip install -e .
```

### Step 4: Verify Installation

```powershell
python -c "from cryptoai import __version__; print(f'CryptoAI {__version__} installed successfully')"
```

### Step 5: Run Tests

```powershell
python -m pytest tests/ -v --timeout=120
```

### Step 6: Run Paper Trading

```powershell
python -m cryptoai.cli run --mode paper --config configs/default.yaml --asset BTCUSDT
```

---

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|------------|
| OS | Windows 10/11 x64 |
| CPU | 4 cores |
| RAM | 8 GB |
| Storage | 10 GB SSD |
| Python | 3.9, 3.10, or 3.11 |

### Recommended Requirements

| Component | Recommendation |
|-----------|---------------|
| OS | Windows 11 x64 |
| CPU | 8+ cores (AMD Ryzen / Intel Core i7+) |
| RAM | 16-32 GB |
| Storage | 50+ GB NVMe SSD |
| GPU | NVIDIA RTX 3060+ (optional, for training) |
| Python | 3.11 |

---

## Installation Options

### Option 1: CPU-Only (Default, Recommended)

Best for inference and paper trading:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

### Option 2: NVIDIA GPU (CUDA) Support

For training and faster inference:

```powershell
# Check your CUDA version
nvidia-smi

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -e .
```

### Option 3: Development Install

For development with all tools:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

---

## Running the System

### CLI Commands

```powershell
# Show help
python -m cryptoai.cli --help

# Run paper trading
python -m cryptoai.cli run --mode paper --asset BTCUSDT

# Run backtest
python -m cryptoai.cli backtest --config configs/default.yaml --start 2024-01-01 --end 2024-06-30

# Train model (requires GPU for reasonable speed)
python -m cryptoai.cli train --config configs/training.yaml
```

### Available Modes

| Mode | Description |
|------|-------------|
| `paper` | Simulated trading with live market data |
| `backtest` | Historical backtesting with walk-forward validation |
| `shadow` | Live signals without execution (monitoring only) |
| `live` | **DISABLED** - Real trading (requires additional safeguards) |

---

## Electron Desktop App

### Build the Desktop App

```powershell
cd electron

# Install Node.js dependencies
npm install

# Run in development mode
npm start

# Build Windows installer (.exe)
npm run build:win
```

The installer will be in `electron/dist/CryptoAI-Setup-x.x.x.exe`

### Desktop App Features

- Start/Stop trading controls
- Real-time activity log
- Configuration management
- Emergency kill switch (Ctrl+Shift+X)
- Paper/Backtest mode selection

---

## Testing

### Run All Tests

```powershell
python -m pytest tests/ -v --timeout=120
```

### Run Specific Test Categories

```powershell
# Unit tests only
python -m pytest tests/test_encoders.py tests/test_risk_engine.py -v

# Integration tests
python -m pytest tests/test_integration.py -v -m integration

# Windows compatibility tests
python -m pytest tests/test_windows_compat.py -v

# Backtesting tests
python -m pytest tests/test_backtesting.py -v
```

### Test with Coverage

```powershell
pip install pytest-cov
python -m pytest tests/ --cov=cryptoai --cov-report=html
# Open htmlcov/index.html in browser
```

---

## Configuration

### Default Configuration

Configuration files are in `configs/`:

```yaml
# configs/default.yaml
model:
  encoder_type: "unified"
  state_dim: 256
  action_dim: 4

training:
  batch_size: 64
  learning_rate: 0.0001
  epochs: 100

risk:
  max_position_pct: 0.20
  max_drawdown: 0.15
  max_daily_loss: 0.05

execution:
  mode: "paper"
  slippage_model: "orderbook_depth"
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CRYPTOAI_CONFIG` | Config file path | `configs/default.yaml` |
| `CRYPTOAI_LOG_LEVEL` | Logging level | `INFO` |
| `CUDA_VISIBLE_DEVICES` | GPU selection | (all GPUs) |
| `CRYPTOAI_TEST_MODE` | Enable test mode | `false` |

---

## Project Structure

```
trade/
├── cryptoai/                    # Main Python package
│   ├── encoders/                # State encoders (transformers, CNNs)
│   ├── decision_engine/         # Policy networks, reward functions
│   ├── world_model/             # Predictive models
│   ├── black_swan/              # Anomaly detection, tail risk
│   ├── risk_engine/             # Position sizing, kill switch
│   ├── training/                # DDP training, data loading
│   ├── backtesting/             # Walk-forward validation
│   ├── execution/               # Order execution (paper/live)
│   └── utils/                   # Config, device, logging
├── electron/                    # Electron desktop app
│   ├── src/
│   │   ├── main.js              # Main process
│   │   ├── preload.js           # IPC bridge
│   │   ├── index.html           # UI
│   │   └── renderer.js          # Frontend logic
│   └── package.json
├── configs/                     # YAML configurations
├── tests/                       # Test suite
├── pyproject.toml               # Python project config
└── README.md                    # This file
```

---

## ML/Trading Architecture

### Core Components

1. **Unified Market Encoder**: Processes OHLCV, order flow, and regime features
2. **Regime Detector**: Classifies market conditions (trending, ranging, volatile, crisis)
3. **Conservative Policy**: Risk-aware action selection with uncertainty estimation
4. **Black Swan Detector**: Anomaly detection using VAE and isolation forests
5. **Risk Controller**: Position limits, drawdown monitoring, kill switch

### Backtesting Standards

The system implements **professional-grade backtesting** that avoids common pitfalls:

- **Walk-Forward Validation**: Train on past, test on future with embargo periods
- **Regime-Aware Splits**: Ensures training covers different market conditions
- **Realistic Execution**: Orderbook-based slippage, latency, partial fills
- **Monte Carlo Significance**: Statistical validation of results
- **No Lookahead Bias**: Strict temporal ordering enforced

### Why Naive Backtests Lie

Common backtesting mistakes this system avoids:

1. **Lookahead Bias**: Using future information for past decisions
2. **Survivorship Bias**: Only testing on assets that still exist
3. **Regime Ignorance**: Testing only in favorable conditions
4. **Transaction Cost Fantasy**: Fixed slippage assumptions
5. **Data Leakage**: Train/test contamination
6. **Overfitting to Noise**: Too many parameters, no validation

### Risk Management

- Maximum position size: 20% of capital per asset
- Maximum leverage: 3x
- Maximum daily loss: 5%
- Maximum drawdown: 15-20%
- Automatic position liquidation on kill switch

---

## Common Issues & Solutions

### Python Not Found

```
'python' is not recognized as an internal or external command
```

**Solution**: Reinstall Python and check "Add Python to PATH", or use full path:
```powershell
C:\Users\YourName\AppData\Local\Programs\Python\Python311\python.exe --version
```

### PyTorch Installation Fails

**Solution**: Install Visual C++ Redistributable:
- Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

### Import Error: No module named 'cryptoai'

**Solution**: Run from the repository root and ensure pip install completed:
```powershell
cd C:\path\to\trade
pip install -e .
```

### CUDA Out of Memory

**Solution**: Reduce batch size or use CPU:
```yaml
# In config file
training:
  batch_size: 16  # Reduce from 64
```

Or force CPU:
```powershell
$env:CUDA_VISIBLE_DEVICES=""
python -m cryptoai.cli run ...
```

### Electron App Won't Start

**Solution**: Check Node.js version and reinstall dependencies:
```powershell
node --version  # Should be 18+
cd electron
Remove-Item -Recurse -Force node_modules
npm install
npm start
```

### Tests Timeout

**Solution**: Increase timeout or skip slow tests:
```powershell
python -m pytest tests/ -v --timeout=300
# Or skip slow tests
python -m pytest tests/ -v -m "not slow"
```

### NCCL Not Available on Windows

This is expected - Windows uses the `gloo` backend instead of `nccl`.
The system automatically detects Windows and switches to the correct backend.

---

## Risk Disclaimer

**IMPORTANT**: This software is for educational and research purposes only.

- Trading cryptocurrencies involves substantial risk of loss
- Past performance does not guarantee future results
- The authors are not responsible for any financial losses
- Always test thoroughly in paper mode before any real trading
- Never invest more than you can afford to lose

---

## Known Limitations

1. **Live Trading Disabled**: The `live` mode requires additional API credentials and safeguards not included in this release
2. **Data Sources**: Synthetic data is used for testing; real data integration requires exchange API setup
3. **Training Time**: Full model training requires GPU and substantial time (hours to days)
4. **Windows Only**: While the code is cross-platform compatible, testing and CI are Windows-focused

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and add tests
4. Run tests: `python -m pytest tests/ -v`
5. Submit a pull request

---

## License

MIT License - See LICENSE file for details.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/Kerim-Sabic/trade/issues)
- **Documentation**: This README and inline docstrings
