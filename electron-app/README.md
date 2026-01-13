# CryptoAI Desktop

Electron desktop application for the CryptoAI trading system.

## Prerequisites

1. **Node.js 18+** - Install from https://nodejs.org/
2. **Python 3.10+** - With cryptoai package installed
3. **npm or yarn** - Package manager

## Development Setup

### Install Dependencies

```bash
cd electron-app
npm install
```

### Run in Development Mode

```bash
npm run dev
```

### Run in Production Mode

```bash
npm start
```

## Building Executables

### Build for Current Platform

```bash
npm run build
```

### Build for Specific Platform

**Windows (.exe):**
```bash
npm run build:win
```

**macOS (.dmg):**
```bash
npm run build:mac
```

**Linux (.AppImage, .deb):**
```bash
npm run build:linux
```

### Build Output

Executables are created in the `dist/` directory:

- **Windows**: `dist/CryptoAI Desktop Setup X.X.X.exe`
- **macOS**: `dist/CryptoAI Desktop-X.X.X.dmg`
- **Linux**: `dist/CryptoAI Desktop-X.X.X.AppImage`

## Project Structure

```
electron-app/
├── main.js           # Electron main process
├── preload.js        # Secure bridge for renderer
├── package.json      # Node.js package config
├── renderer/
│   ├── index.html    # Main UI
│   ├── styles.css    # Styles
│   └── renderer.js   # UI logic
├── assets/           # Icons and images
│   ├── icon.png      # App icon (256x256)
│   ├── icon.ico      # Windows icon
│   └── icon.icns     # macOS icon
└── dist/             # Built executables
```

## Configuration

The app stores settings in:

- **Windows**: `%APPDATA%/cryptoai-desktop/config.json`
- **macOS**: `~/Library/Application Support/cryptoai-desktop/config.json`
- **Linux**: `~/.config/cryptoai-desktop/config.json`

### Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `pythonPath` | Path to Python executable | `python` |
| `configPath` | Path to YAML config file | `configs/default.yaml` |
| `mode` | Default trading mode | `shadow` |
| `assets` | Default assets to trade | `["BTCUSDT"]` |

## Usage

1. **Configure Python Path**: Settings > Browse Python executable
2. **Select Config**: Click Browse to select your config.yaml
3. **Choose Mode**:
   - **Shadow**: Simulation with no API calls
   - **Paper**: Testnet trading
   - **Backtest**: Historical simulation
   - **Live**: Real trading (use with caution!)
4. **Enter Assets**: Comma-separated list (e.g., `BTCUSDT, ETHUSDT`)
5. **Start Trading**: Click the Start button

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Cmd/Ctrl + ,` | Open Settings |
| `Cmd/Ctrl + R` | Start Trading |
| `Cmd/Ctrl + S` | Stop Trading |
| `Cmd/Ctrl + Q` | Quit Application |

## Logs

Logs are stored in:

- **Windows**: `%USERPROFILE%/AppData/Roaming/cryptoai-desktop/logs/`
- **macOS**: `~/Library/Logs/cryptoai-desktop/`
- **Linux**: `~/.config/cryptoai-desktop/logs/`

## Troubleshooting

### Python not found

1. Ensure Python 3.10+ is installed
2. Go to Settings and browse to your Python executable
3. Common paths:
   - **Windows**: `C:\Python311\python.exe`
   - **macOS/Linux**: `/usr/bin/python3` or `/usr/local/bin/python3`
   - **Conda**: `~/miniconda3/envs/cryptoai/bin/python`

### cryptoai module not found

Ensure the cryptoai package is installed:

```bash
cd /path/to/trade
pip install -e .
```

### Build fails on macOS

Install Xcode Command Line Tools:

```bash
xcode-select --install
```

### Build fails on Linux

Install required dependencies:

```bash
# Ubuntu/Debian
sudo apt-get install -y build-essential rpm

# Fedora
sudo dnf install -y rpm-build
```

## Creating Icons

For building, you need icons in multiple formats:

1. Create a 1024x1024 PNG icon
2. Convert to formats:
   - **icon.ico** (Windows): Use https://icoconvert.com/
   - **icon.icns** (macOS): Use iconutil or https://cloudconvert.com/

## License

MIT License
