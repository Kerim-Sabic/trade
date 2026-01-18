# CryptoAI Trading Platform

**Production-Grade ML Crypto Trading System for Windows 11**

A professional machine learning system for cryptocurrency trading with institutional-grade risk management, walk-forward validation, regime-aware decision making, and **real-time price simulation**.

---

## Quick Start (5 Minutes)

### One-Command Install (Windows 11 PowerShell)

```powershell
# Clone repository
git clone https://github.com/Kerim-Sabic/trade.git
cd trade

# Install everything
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e .

# Verify installation
python -c "from cryptoai import __version__; print(f'CryptoAI {__version__}')"
```

### Run Simulation Mode (No API Keys Required!)

```powershell
# CLI simulation with real Binance prices
python -m cryptoai.cli run --mode simulation --asset BTCUSDT

# Or run the desktop app
cd electron && npm install && npm start
```

---

## Features

### New in This Release

- **Real-Time Price Simulation**: Live Binance/CoinGecko prices without trading
- **Multi-Exchange Support**: Binance, OKX, and Bybit clients
- **CVaR Risk Management**: Institutional-grade tail risk controls
- **Dynamic Drawdown Stops**: Volatility-adjusted position management
- **Electron Desktop App**: Full simulation mode with real price feeds
- **Explicit "Do Nothing" Logic**: AI knows when NOT to trade

### Core Features

- Walk-forward backtesting with embargo periods
- Regime-aware ML models (trending, volatile, crisis detection)
- Black swan detection (VAE + Isolation Forest)
- Kill switch with multiple triggers
- Windows 11 native support (gloo backend for DDP)

---

## System Requirements

| Component | Minimum | Recommended | Maximum Performance |
|-----------|---------|-------------|---------------------|
| OS | Windows 10/11 x64 | Windows 11 x64 | Windows 11 x64 |
| CPU | 4 cores | 8+ cores | AMD Ryzen 9 7950X (16 cores) |
| RAM | 8 GB | 16-32 GB | 128 GB DDR5 |
| Storage | 10 GB SSD | 50+ GB NVMe | 1TB+ NVMe |
| Python | 3.10+ | 3.11 | 3.11/3.12 |
| GPU | Not required | NVIDIA RTX 3060+ | 2× RTX 5080 (32GB total VRAM) |

### Multi-GPU Training Hardware (Reference Setup)

This system has been optimized for the following high-performance configuration:
- **GPU**: 2× NVIDIA RTX 5080 (16GB VRAM each, 32GB total)
- **CPU**: AMD Ryzen 9 7950X (16 cores / 32 threads)
- **RAM**: 128GB DDR5
- **Storage**: NVMe SSD

With this setup, you can achieve:
- **Effective batch size**: 512 (64 × 4 gradient accumulation × 2 GPUs)
- **Training throughput**: ~3x faster than single RTX 3090
- **Multi-resolution processing**: 1m/5m/15m/1h/4h candles

---

## Installation Options

### Option 1: CPU-Only (Default)

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

### Option 2: NVIDIA GPU (CUDA 12.1)

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

### Option 3: Development Install

```powershell
pip install -e ".[dev]"
```

---

## Running Modes

| Mode | Description | API Keys Required |
|------|-------------|-------------------|
| `simulation` | **Real prices, simulated trades** | No |
| `paper` | Paper trading on testnet | Exchange testnet key |
| `backtest` | Historical walk-forward testing | No |
| `shadow` | Live signals, no execution | No |
| `live` | **DISABLED** - Real trading | Yes (not enabled) |

### Run Simulation (Recommended for Testing)

```powershell
# Simulation uses REAL Binance prices but NO actual trading
python -m cryptoai.cli run --mode simulation --asset BTCUSDT
```

### Run Backtest

```powershell
python -m cryptoai.cli backtest --asset BTCUSDT --days 90
```

### Run Paper Trading

```powershell
# Requires exchange testnet API key in config
python -m cryptoai.cli run --mode paper --asset BTCUSDT
```

---

## Multi-GPU Training Guide

### Overview

The training system uses PyTorch DistributedDataParallel (DDP) for multi-GPU training with the `gloo` backend (required for Windows 11).

### Configuration

Default configuration (`configs/default.yaml`) is optimized for 2× RTX 5080:

```yaml
hardware:
  gpus: [0, 1]                    # Use both GPUs
  use_ddp: true                   # Enable distributed training
  use_amp: true                   # Enable Automatic Mixed Precision
  ddp_backend: "gloo"             # Windows-compatible backend

  multi_gpu:
    world_size: 2                 # Number of GPUs
    gradient_accumulation_steps: 4 # Effective batch = 64 × 4 × 2 = 512
    sync_batch_norm: true         # Synchronize batch norm across GPUs

training:
  offline:
    batch_size: 64                # Per-GPU batch size
    effective_batch_size: 512     # Total effective batch size
    learning_rate: 0.0003         # Scaled for larger batch
```

### Launching Multi-GPU Training

```powershell
# Automatic multi-GPU detection and training
python -m cryptoai.cli train --asset BTCUSDT

# Explicit GPU selection
CUDA_VISIBLE_DEVICES=0,1 python -m cryptoai.cli train --asset BTCUSDT
```

### Training Performance

| Configuration | Batch Size | Throughput | Memory Usage |
|--------------|-----------|------------|--------------|
| Single RTX 5080 | 64 | ~150 steps/min | ~12GB |
| 2× RTX 5080 | 512 (effective) | ~400 steps/min | ~14GB/GPU |
| CPU Only | 16 | ~10 steps/min | ~32GB RAM |

### Multi-Resolution Training

The system processes candles at multiple resolutions for different prediction horizons:

```yaml
training:
  multi_resolution:
    enabled: true
    resolutions: ["1m", "5m", "15m", "1h", "4h"]
    downsampling_method: "ohlcv_aggregate"
    feature_fusion: "attention"  # or "concat", "hierarchical"
```

### Memory Optimization Tips

1. **Gradient Accumulation**: Increase `gradient_accumulation_steps` to simulate larger batches
2. **Mixed Precision**: Keep `use_amp: true` for 2x memory savings
3. **Cache Clearing**: `empty_cache_frequency: 100` clears GPU cache periodically
4. **Pin Memory**: `pin_memory: true` for faster CPU→GPU transfers

### Fallback to Single GPU / CPU

The system automatically falls back if multi-GPU is unavailable:

```python
# Auto-detection in code
if torch.cuda.device_count() < 2:
    # Falls back to single GPU
if not torch.cuda.is_available():
    # Falls back to CPU (AMP disabled)
```

---

## Electron Desktop App

### Quick Start

```powershell
cd electron
npm install
npm start
```

### Build Windows Installer (.exe)

```powershell
cd electron
npm run build:win
# Installer at: electron/dist/CryptoAI-Setup-x.x.x.exe
```

### Desktop App Features

- **Simulation Mode**: Real prices from Binance, AI trading signals displayed
- **Start/Stop Controls**: Easy trading management
- **Real-Time Log**: See every price update and AI decision
- **Emergency Kill Switch**: Ctrl+Shift+X instant stop
- **Asset Selection**: BTC, ETH, SOL, BNB supported

### Simulation Mode in Desktop

1. Select "Simulation (Real Prices)" from Mode dropdown
2. Choose your asset (BTCUSDT, ETHUSDT, etc.)
3. Click "Start Trading"
4. Watch real prices stream with AI signals

**Note**: Simulation mode does NOT place real trades. It fetches real prices and shows what the AI would do.

---

## Price Feed API Configuration

### Default: Binance Public API (No Key Required)

The system uses Binance's public ticker API by default:
- No API key needed
- Rate limit: ~1200 requests/minute
- Data: Price, 24h change, volume, bid/ask

### Optional: CoinGecko API

For alternative data source:
```yaml
# In configs/default.yaml
price_feed:
  provider: "coingecko"
  symbols: ["BTCUSDT", "ETHUSDT"]
```

### Optional: Polygon.io API

For stocks/forex alongside crypto:
```powershell
$env:PRICE_FEED_API_KEY="your_polygon_api_key"
```

---

## Exchange Client Configuration

### Supported Exchanges

| Exchange | Spot | Futures | Testnet |
|----------|------|---------|---------|
| Binance | Yes | Yes | Yes |
| OKX | Yes | Yes | Yes |
| Bybit | Yes | Yes | Yes |

### Setting Up Exchange Keys (Paper Trading)

Create `.env` file in repository root:
```bash
# Binance Testnet
BINANCE_API_KEY=your_testnet_api_key
BINANCE_API_SECRET=your_testnet_secret

# OKX (if using)
OKX_API_KEY=your_okx_key
OKX_API_SECRET=your_okx_secret
OKX_PASSPHRASE=your_okx_passphrase

# Bybit (if using)
BYBIT_API_KEY=your_bybit_key
BYBIT_API_SECRET=your_bybit_secret
```

**Important**: Use TESTNET keys only. Live trading is disabled in this release.

---

## Risk Management

### CVaR-Aware Position Sizing

The system uses Conditional Value at Risk (CVaR) for tail-risk aware sizing:

```python
from cryptoai.risk_engine import create_cvar_position_sizer

sizer = create_cvar_position_sizer(
    max_position_pct=0.15,  # Max 15% per position
    target_cvar=0.02,       # Target 2% CVaR
    min_confidence=0.6      # Min 60% confidence to trade
)
```

### Dynamic Drawdown Management

- **Warning Level**: 8% drawdown → Reduce position sizes
- **Max Level**: 15% drawdown → Close all positions
- **Daily Limit**: 3% daily loss → Stop trading for the day

### Explicit "Do Nothing" Conditions

The AI will NOT trade when:
- Confidence < 60% (adjustable)
- Uncertainty > 30%
- Black swan probability > 50%
- Expected return < 0.1% (below transaction costs)
- Drawdown limit reached

---

## Testing

### Run All Tests

```powershell
python -m pytest tests/ -v --timeout=120
```

### Test Categories

```powershell
# Price feed tests
python -m pytest tests/test_price_feed.py -v

# Risk engine tests (including CVaR)
python -m pytest tests/test_risk_engine.py -v

# Windows compatibility
python -m pytest tests/test_windows_compat.py -v

# Integration tests
python -m pytest tests/test_integration.py -v
```

### Test Real Price Feed

```powershell
python -c "
import asyncio
from cryptoai.data_universe.price_feed import create_price_feed

async def test():
    feed = create_price_feed(provider='binance', symbols=['BTCUSDT'])
    await feed.connect()
    price = await feed.get_price('BTCUSDT')
    print(f'BTC: \${price.price:,.2f} ({price.change_24h:+.2f}%)')
    await feed.disconnect()

asyncio.run(test())
"
```

---

## Project Structure

```
trade/
├── cryptoai/                    # Main Python package
│   ├── data_universe/
│   │   └── price_feed.py        # NEW: Real-time price feeds
│   ├── execution/
│   │   └── exchange_client.py   # NEW: Binance, OKX, Bybit
│   ├── risk_engine/
│   │   ├── cvar_position_sizer.py  # NEW: CVaR-aware sizing
│   │   └── ...
│   └── ...
├── electron/                    # Desktop app
│   ├── src/
│   │   ├── main.js              # NEW: Simulation mode
│   │   └── ...
├── tests/
│   └── test_price_feed.py       # NEW: Price feed tests
├── configs/
│   └── default.yaml
└── README.md
```

---

## CI/CD Pipeline

GitHub Actions workflow runs on every push:

1. **Python Tests**: Linting, type checking, unit tests
2. **CVaR Risk Engine Tests**: Drawdown, inaction threshold
3. **Real Price Feed Test**: Live API integration check
4. **Electron Build**: Windows installer generation
5. **Security Scan**: Bandit + Safety checks

See `.github/workflows/windows-ci.yml` for details.

---

## Common Issues

### "Cannot connect to price feed"

**Solution**: Check internet connection. The system needs access to:
- `api.binance.com` (price data)
- `api.coingecko.com` (alternative)

### "CVaR calculation returns 0"

**Solution**: Need at least 10 return observations:
```python
cvar_calc = CVaRCalculator()
for _ in range(20):
    cvar_calc.update(random.gauss(0, 0.01))
metrics = cvar_calc.calculate()  # Now works
```

### "Simulation mode not starting in Electron"

**Solution**: Simulation mode works without Python. Check the Electron console (Ctrl+Shift+I) for errors.

### Tests failing on Windows

**Solution**: Ensure you're using CPU-only PyTorch:
```powershell
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## Risk Disclaimer

**IMPORTANT**: This software is for educational and research purposes only.

- Trading cryptocurrencies involves substantial risk of loss
- Past performance does not guarantee future results
- Simulation mode shows what the AI **would** do, not guaranteed profits
- The authors are not responsible for any financial losses
- Always test thoroughly before any real trading
- Never invest more than you can afford to lose

---

## Known Limitations

1. **Live Trading Disabled**: Requires additional safeguards
2. **Simulation != Reality**: Real trading has slippage, fees, partial fills
3. **AI Signals are Indicative**: Not financial advice
4. **Rate Limits**: Free APIs have request limits
5. **Windows Focus**: CI/CD optimized for Windows 11

---

## Version History

### v0.2.1 (Current)
- **Multi-GPU Training**: Optimized DDP training for 2× RTX 5080 with Windows 11 `gloo` backend
- **Multi-Resolution Processing**: Added 1m/5m/15m/1h/4h candle processing pipeline
- **Gradient Accumulation**: Implemented gradient accumulation for effective batch size of 512
- **Memory Optimization**: Added periodic cache clearing and SyncBatchNorm for multi-GPU
- **Enhanced Tests**: Added comprehensive black swan detector tests
- **Updated Config**: Hardware configuration optimized for 128GB RAM + dual RTX 5080

### v0.2.0
- Added real-time price feed integration (Binance, CoinGecko)
- Added OKX and Bybit exchange clients
- Added CVaR-aware position sizing
- Added dynamic drawdown management
- Added simulation mode to Electron app
- Enhanced CI/CD with price feed tests
- Improved risk controls with explicit "do nothing" logic

### v0.1.0
- Initial release with backtesting and paper trading

---

## License

MIT License - See LICENSE file for details.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/Kerim-Sabic/trade/issues)
- **Documentation**: This README and inline docstrings
