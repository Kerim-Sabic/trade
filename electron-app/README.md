# CryptoAI Desktop

Professional Electron desktop application for the CryptoAI autonomous trading system with safety-first design principles.

## Features

- **Dark Mode UI** - Professional trading interface optimized for extended use
- **Governance System** - Real-time visibility into system state (OPERATIONAL, RESTRICTED, HALTED)
- **Kill Switch** - Emergency stop accessible via UI, keyboard shortcut, or system tray
- **Live Mode Protection** - Confirmation dialogs prevent accidental live trading
- **Multi-Exchange Support** - Binance, Bybit, OKX with testnet options
- **Live Logs** - Real-time Python backend output with auto-scroll
- **Safe Defaults** - Shadow mode (no API calls) enabled by default

## Prerequisites

1. **Node.js 18+** - Install from https://nodejs.org/
2. **Python 3.10+** - With cryptoai package installed
3. **npm or yarn** - Package manager

## Quick Start

```bash
# 1. Install dependencies
cd electron-app
npm install

# 2. Run in development mode
npm run dev

# 3. Or run in production mode
npm start
```

---

## Building for Windows (Detailed Guide)

### Prerequisites for Windows

1. **Install Node.js 18 LTS**
   - Download from https://nodejs.org/en/download/
   - Choose "Windows Installer (.msi)" for 64-bit
   - Run installer with default options
   - Verify: Open PowerShell and run `node --version`

2. **Install Python 3.10+**
   - Download from https://www.python.org/downloads/windows/
   - **IMPORTANT**: Check "Add Python to PATH" during installation
   - Verify: `python --version`

3. **Install Visual Studio Build Tools** (required for native modules)
   ```powershell
   # Option 1: Using npm (recommended)
   npm install --global windows-build-tools

   # Option 2: Manual install
   # Download from https://visualstudio.microsoft.com/visual-cpp-build-tools/
   # Select "Desktop development with C++" workload
   ```

4. **Install Git for Windows** (optional, for cloning)
   - Download from https://git-scm.com/download/win

### Build Steps for Windows

```powershell
# Open PowerShell as Administrator

# 1. Navigate to project
cd C:\path\to\trade\electron-app

# 2. Clear any existing modules
Remove-Item -Recurse -Force node_modules -ErrorAction SilentlyContinue
Remove-Item package-lock.json -ErrorAction SilentlyContinue

# 3. Install dependencies
npm install

# 4. Build Windows executable
npm run build:win

# 5. Find your installer
explorer.exe dist
# Output: dist/CryptoAI Desktop Setup X.X.X.exe
```

### Windows Build Output

After building, you'll find in the `dist/` folder:
- `CryptoAI Desktop Setup 0.1.0.exe` - NSIS installer (recommended)
- `CryptoAI Desktop 0.1.0.exe` - Portable executable
- `win-unpacked/` - Unpacked application files

### Windows Installation

1. Run `CryptoAI Desktop Setup X.X.X.exe`
2. Choose installation directory (default: `C:\Program Files\CryptoAI Desktop`)
3. Optionally create desktop shortcut
4. Click Install

### Windows-Specific Features

- **System Tray**: Minimizes to system tray instead of closing
- **Single Instance**: Only one instance can run at a time
- **Auto-start**: Can be configured to start with Windows

---

## Building for macOS

### Prerequisites for macOS

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Node.js via Homebrew (recommended)
brew install node@18
```

### Build Steps for macOS

```bash
cd electron-app
npm install
npm run build:mac
```

### macOS Build Output

- `dist/CryptoAI Desktop-X.X.X.dmg` - Disk image installer
- `dist/CryptoAI Desktop-X.X.X-arm64.dmg` - Apple Silicon version
- `dist/mac/` or `dist/mac-arm64/` - Unpacked app

### macOS Code Signing (Optional)

For distribution outside the App Store:
```bash
# Set environment variables
export CSC_LINK=/path/to/certificate.p12
export CSC_KEY_PASSWORD=your_password

# Build with signing
npm run build:mac
```

---

## Building for Linux

### Prerequisites for Linux

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential rpm fakeroot

# Fedora
sudo dnf install -y rpm-build gcc-c++ make

# Arch Linux
sudo pacman -S base-devel rpm-tools
```

### Build Steps for Linux

```bash
cd electron-app
npm install
npm run build:linux
```

### Linux Build Output

- `dist/CryptoAI Desktop-X.X.X.AppImage` - Portable executable
- `dist/cryptoai-desktop_X.X.X_amd64.deb` - Debian package
- `dist/linux-unpacked/` - Unpacked application

### Installing on Linux

```bash
# AppImage (no installation needed)
chmod +x "CryptoAI Desktop-X.X.X.AppImage"
./"CryptoAI Desktop-X.X.X.AppImage"

# Debian package
sudo dpkg -i cryptoai-desktop_X.X.X_amd64.deb
```

---

## Project Structure

```
electron-app/
├── main.js              # Electron main process
│                        # - Window management
│                        # - Python process spawning
│                        # - IPC handlers
│                        # - Kill switch implementation
│                        # - System tray (Windows)
│
├── preload.js           # Secure context bridge
│                        # - Whitelisted IPC channels
│                        # - Type validation
│                        # - Event listeners
│
├── package.json         # Node.js configuration
│                        # - Build settings
│                        # - electron-builder config
│
├── renderer/
│   ├── index.html       # Main UI layout
│   │                    # - Governance badge
│   │                    # - Status panel
│   │                    # - Controls panel
│   │                    # - Logs panel
│   │                    # - Settings modal
│   │
│   ├── styles.css       # Dark theme styles
│   │                    # - CSS custom properties
│   │                    # - Responsive design
│   │
│   └── renderer.js      # UI logic
│                        # - Event handling
│                        # - Status updates
│                        # - Log display
│
├── assets/
│   └── icons/           # App icons (16-512px)
│
└── dist/                # Build output (generated)
```

---

## Safety Features

### Kill Switch

The emergency kill switch immediately stops all trading:

| Method | How to Use |
|--------|------------|
| **Keyboard** | `Ctrl+Shift+K` (Windows/Linux) or `Cmd+Shift+K` (macOS) |
| **UI Button** | Click red "EMERGENCY STOP" button in header |
| **Menu** | Trading > Emergency Kill Switch |
| **System Tray** | Right-click tray icon > Emergency Stop (Windows) |

### Governance States

| State | Meaning | Trading Allowed |
|-------|---------|-----------------|
| **OPERATIONAL** | System healthy | Yes |
| **RESTRICTED** | Elevated risk detected | Limited |
| **HALTED** | Emergency stop or critical error | No |

### Live Mode Protection

When starting live trading:
1. Warning banner appears in UI
2. Confirmation dialog requires explicit acknowledgment
3. Risk warnings displayed
4. Can be disabled in Settings (not recommended)

---

## Configuration

### Settings Location

| Platform | Path |
|----------|------|
| Windows | `%APPDATA%\cryptoai-desktop\config.json` |
| macOS | `~/Library/Application Support/cryptoai-desktop/config.json` |
| Linux | `~/.config/cryptoai-desktop/config.json` |

### Available Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `pythonPath` | Path to Python executable | `python` |
| `configPath` | Path to YAML config file | `configs/default.yaml` |
| `mode` | Default trading mode | `shadow` |
| `exchange` | Default exchange | `binance` |
| `assets` | Default trading pairs | `["BTCUSDT"]` |
| `confirmLiveMode` | Require live mode confirmation | `true` |
| `maxLeverage` | Maximum allowed leverage | `3` |
| `maxDrawdown` | Maximum drawdown threshold | `0.10` (10%) |

---

## Trading Modes

| Mode | Description | Risk Level | API Calls |
|------|-------------|------------|-----------|
| **Shadow** | Pure simulation, no exchange connection | None | None |
| **Paper** | Uses exchange testnet | None | Testnet only |
| **Backtest** | Historical data simulation | None | None |
| **Live** | Real trading with real money | High | Production |

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl/Cmd + ,` | Open Settings |
| `Ctrl/Cmd + Enter` | Start Trading |
| `Ctrl/Cmd + .` | Stop Trading |
| `Ctrl/Cmd + Shift + K` | Emergency Kill Switch |
| `Ctrl/Cmd + Q` / `Alt + F4` | Quit Application |

---

## Logs

### Log Locations

| Platform | Path |
|----------|------|
| Windows | `%USERPROFILE%\AppData\Roaming\cryptoai-desktop\logs\` |
| macOS | `~/Library/Logs/cryptoai-desktop/` |
| Linux | `~/.config/cryptoai-desktop/logs/` |

### Log Levels

- **stdout** (white) - Normal Python output
- **stderr** (yellow) - Warnings
- **error** (red) - Errors
- **info** (cyan) - System messages

---

## Troubleshooting

### Windows: "Python not found"

1. Open Settings in the app
2. Click "Browse" next to Python Path
3. Navigate to your Python installation:
   - Default: `C:\Python311\python.exe`
   - Microsoft Store: `%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe`
   - Anaconda: `C:\Users\<name>\anaconda3\python.exe`

### Windows: "Module cryptoai not found"

```powershell
# Navigate to project root
cd C:\path\to\trade

# Install in development mode
pip install -e .
```

### Windows: Build fails with "gyp ERR!"

```powershell
# Install Windows Build Tools
npm install --global windows-build-tools

# Or install Visual Studio Build Tools manually
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### Windows: "EPERM: operation not permitted"

Run PowerShell as Administrator, or:
```powershell
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
Remove-Item -Recurse -Force node_modules
npm install
```

### macOS: "App is damaged and can't be opened"

```bash
# Remove quarantine attribute
xattr -cr "/Applications/CryptoAI Desktop.app"
```

### Linux: AppImage won't run

```bash
# Make executable
chmod +x CryptoAI\ Desktop-*.AppImage

# If FUSE is missing
sudo apt install libfuse2
```

### All Platforms: Trading won't start

1. Check Python path in Settings
2. Ensure cryptoai module is installed
3. Verify config file exists
4. Check logs panel for errors

---

## Development

### Running with DevTools

```bash
# Development mode opens DevTools automatically
npm run dev
```

### Hot Reload

The app doesn't support hot reload. Use `Ctrl+R` to reload the renderer.

### Debugging Main Process

```bash
# With Chrome DevTools
npm run dev -- --inspect=5858
# Then open chrome://inspect in Chrome
```

---

## Security

- **Context Isolation**: Enabled - renderer has no Node.js access
- **Sandbox**: Enabled - renderer runs in sandboxed environment
- **Preload Script**: Uses contextBridge for secure IPC
- **CSP**: Strict Content Security Policy in HTML
- **No Remote Content**: App doesn't load external resources

---

## License

MIT License
