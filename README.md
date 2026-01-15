# CryptoAI - Autonomous Crypto Trading Intelligence

**Version 0.2.0** | **Windows 11 First** | **Production-Grade ML Trading System**

---

## Overview

CryptoAI is a professional-grade, ML-driven cryptocurrency trading system designed with Windows 11 as the primary target platform. It features:

- **Deep Reinforcement Learning** - SAC/PPO policy networks with online adaptation
- **Market State Encoders** - Transformer-based encoding of price, volume, and order flow
- **Regime Detection** - Automatic detection of market regimes (trending, ranging, volatile)
- **Risk Management** - Multi-layer kill switches and drawdown protection
- **Desktop Application** - Electron-based GUI with one-click trading controls

---

## System Requirements

### Minimum Requirements (Windows 11)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 11 x64 | Windows 11 22H2+ |
| **Python** | 3.10 | 3.11 |
| **RAM** | 16 GB | 32 GB |
| **CPU** | 4 cores | 8+ cores |
| **Storage** | 10 GB | 50 GB SSD |
| **GPU** | None (CPU mode) | NVIDIA RTX 3060+ (8GB VRAM) |

### GPU Support (Optional)

For GPU acceleration:
- NVIDIA GPU with 8GB+ VRAM
- CUDA 12.1+ from [NVIDIA](https://developer.nvidia.com/cuda-downloads)
- cuDNN 8.9+

---

## Quick Start (5 Minutes)

### Option 1: Desktop App (Recommended for Windows)

```powershell
# 1. Download the installer from Releases
# 2. Run CryptoAI-Desktop-Setup.exe
# 3. Launch from Start Menu or Desktop shortcut
# 4. Configure Python path in Settings (Ctrl+,)
# 5. Click "Start Trading" in Shadow Mode
```

### Option 2: Command Line Installation

```powershell
# 1. Install Python 3.11 from python.org
# Download from: https://www.python.org/downloads/release/python-3119/
# IMPORTANT: Check "Add Python to PATH" during installation

# 2. Open PowerShell and verify Python
python --version
# Should show: Python 3.11.x

# 3. Clone the repository
git clone https://github.com/Kerim-Sabic/trade.git
cd trade

# 4. Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 5. Install PyTorch (CPU version for quick start)
pip install torch torchvision torchaudio

# 6. Install CryptoAI
pip install -e .

# 7. Run in Shadow Mode (safe simulation)
python -m cryptoai.main --mode shadow --assets BTCUSDT
```

### Option 3: GPU Installation

```powershell
# After step 4 above, install PyTorch with CUDA support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then continue with step 6
pip install -e .
```

---

## One-Command Operations

### Run Trading (Shadow Mode - Safe)
```powershell
python -m cryptoai.main --mode shadow --assets BTCUSDT ETHUSDT
```

### Run Backtest
```powershell
python -m cryptoai.main --mode backtest --assets BTCUSDT
```

### Run Training
```powershell
python -m cryptoai.main --mode train --config configs/default.yaml
```

### Run Tests
```powershell
pytest tests/ -v
```

### Build Desktop App
```powershell
cd electron-app
npm install
npm run build:win
```

---

## Execution Modes

| Mode | Description | Risk Level | API Calls |
|------|-------------|------------|-----------|
| `shadow` | Pure simulation, no external calls | None | No |
| `paper` | Exchange testnet trading | None | Testnet only |
| `backtest` | Historical simulation | None | No |
| `train` | Model training mode | None | No |
| `live` | **REAL TRADING** | **HIGH** | **YES** |

**WARNING**: Live mode uses REAL money. Only use after thorough testing.

---

## Configuration

### Environment Variables

```powershell
# Optional - Configure before running
$env:CRYPTOAI_MODE = "shadow"           # Default mode
$env:CRYPTOAI_EXCHANGE = "binance"      # Exchange selection
$env:CUDA_VISIBLE_DEVICES = ""          # Disable GPU (CPU only)
$env:CRYPTOAI_LOG_LEVEL = "INFO"        # Logging level
```

### Configuration File

The main configuration is in `configs/default.yaml`. Key sections:

```yaml
# System settings
system:
  device: "auto"  # auto, cpu, cuda
  precision: "amp"  # amp, fp32

# Risk management (critical)
risk_engine:
  max_position_size: 0.1  # 10% of portfolio per position
  max_leverage: 3
  max_drawdown: 0.15  # 15% max drawdown before halt
  kill_switches:
    daily_loss_limit: 0.05
    consecutive_losses: 5

# Exchange (for live/paper modes)
exchange:
  type: "binance"
  testnet: true  # Always true for paper mode
  # API keys set via environment or .env file
```

---

## Desktop Application

### Installation (Windows)

1. Download `CryptoAI-Desktop-Setup.exe` from Releases
2. Run installer (allows custom install location)
3. Launch from Start Menu -> CryptoAI -> CryptoAI Desktop

### Features

- **One-Click Trading** - Start/Stop with single button
- **Kill Switch** - Emergency stop accessible via Ctrl+Shift+K
- **Live Mode Protection** - Confirmation dialog for real trading
- **Real-time Logs** - Python backend output visible in UI
- **System Tray** - Minimize to tray, continue trading

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+,` | Open Settings |
| `Ctrl+Shift+K` | Emergency Kill Switch |
| `Ctrl+Enter` | Start Trading |
| `Ctrl+.` | Stop Trading |

### Building from Source

```powershell
cd electron-app
npm install
npm run build:win
# Output: electron-app/dist/CryptoAI-Desktop-Setup.exe
```

---

## Architecture

```
cryptoai/
├── encoders/          # Neural network encoders
│   ├── unified.py     # Combined state encoder
│   ├── order_flow.py  # Microstructure encoder
│   └── regime.py      # Market regime detection
├── decision_engine/   # Policy and control
│   ├── policy.py      # SAC/PPO networks
│   └── meta_controller.py  # Regime-aware controller
├── world_model/       # Market dynamics prediction
├── black_swan/        # Anomaly detection
├── risk_engine/       # Position and risk management
├── backtesting/       # Event-driven simulation
├── training/          # DDP training infrastructure
├── execution/         # Exchange connectivity
├── monitoring/        # Metrics and alerts
└── main.py            # Entry point
```

---

## Testing

### Run All Tests
```powershell
pytest tests/ -v
```

### Run Specific Test Categories
```powershell
# Unit tests only
pytest tests/test_encoders.py tests/test_training.py -v

# Backtesting tests
pytest tests/test_backtesting.py -v

# Risk engine tests (critical)
pytest tests/test_risk_engine.py -v
```

### Test Coverage
```powershell
pytest tests/ --cov=cryptoai --cov-report=html
# Open htmlcov/index.html in browser
```

---

## Common Windows Errors & Fixes

### Error: `torch not found`
```powershell
# Solution: Install PyTorch explicitly
pip install torch torchvision torchaudio
```

### Error: `ModuleNotFoundError: No module named 'cryptoai'`
```powershell
# Solution: Install in development mode
pip install -e .
```

### Error: `CUDA out of memory`
```powershell
# Solution: Use CPU mode
$env:CUDA_VISIBLE_DEVICES = ""
python -m cryptoai.main --mode shadow
```

### Error: `Permission denied` when writing files
```powershell
# Solution: Run PowerShell as Administrator or change install location
```

### Error: Electron app won't start
```powershell
# Solution: Verify Python path in Settings (Ctrl+,)
# Default: python
# If using venv: C:\path\to\venv\Scripts\python.exe
```

### Error: `DLL load failed` on Windows
```powershell
# Solution: Install Visual C++ Redistributable
# Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
```

### Error: NCCL not available on Windows
```
# This is expected - Windows uses 'gloo' backend instead of 'nccl'
# The system automatically detects and switches backends
```

---

## Risk Warnings

1. **This is experimental software** - Use at your own risk
2. **Cryptocurrency trading is highly risky** - You can lose all invested capital
3. **Past performance does not guarantee future results**
4. **Always start in Shadow Mode** - Test thoroughly before any real trading
5. **Never invest more than you can afford to lose**

---

## Development

### Setting Up Development Environment
```powershell
# Clone and setup
git clone https://github.com/Kerim-Sabic/trade.git
cd trade
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install with dev dependencies
pip install torch torchvision torchaudio
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
ruff check cryptoai/

# Run type checking
mypy cryptoai/ --ignore-missing-imports
```

### Code Style
- Python: Black formatter, Ruff linter
- Type hints: Required for public APIs
- Docstrings: Google style

---

## CI/CD

The project uses GitHub Actions for continuous integration on Windows:

- **Python Tests**: Unit tests, integration tests, import validation
- **Electron Build**: Windows installer (.exe) and portable builds
- **Security Scan**: Bandit and Safety vulnerability checks

See `.github/workflows/windows-ci.yml` for details.

---

## Project Structure

```
trade/
├── cryptoai/              # Main Python package
├── configs/               # Configuration files
├── electron-app/          # Desktop application
├── tests/                 # Test suite
├── docker/                # Docker configuration
├── scripts/               # Utility scripts
├── .github/workflows/     # CI/CD pipelines
├── pyproject.toml         # Python package config
├── LICENSE                # MIT License
└── README.md              # This file
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/Kerim-Sabic/trade/issues)
- **Documentation**: This README and inline code documentation

---

## Disclaimer

This software is provided for educational and research purposes only. The authors and contributors are not responsible for any financial losses incurred through the use of this software. Cryptocurrency trading involves substantial risk of loss. **USE AT YOUR OWN RISK.**
