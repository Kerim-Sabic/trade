/**
 * CryptoAI Desktop - Main Process
 *
 * Windows 11 Compatible Electron Application
 *
 * Features:
 * - Python backend spawning with proper Windows process handling
 * - IPC communication for trading controls
 * - Emergency kill switch functionality
 * - Configuration management
 */

const { app, BrowserWindow, ipcMain, dialog, Menu } = require('electron');
const path = require('path');
const { spawn, exec } = require('child_process');
const log = require('electron-log');
const Store = require('electron-store');

// Configure logging
log.transports.file.level = 'info';
log.transports.console.level = 'debug';

// Configuration store
const store = new Store({
    defaults: {
        pythonPath: 'python',
        configFile: 'configs/default.yaml',
        lastAsset: 'BTCUSDT',
        mode: 'paper',
        windowBounds: { width: 1200, height: 800 }
    }
});

// Global references
let mainWindow = null;
let pythonProcess = null;
let isTrading = false;

// Platform detection
const IS_WINDOWS = process.platform === 'win32';

/**
 * Create the main application window
 */
function createWindow() {
    const bounds = store.get('windowBounds');

    mainWindow = new BrowserWindow({
        width: bounds.width,
        height: bounds.height,
        minWidth: 800,
        minHeight: 600,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
            sandbox: true
        },
        icon: path.join(__dirname, '../assets/icon.ico'),
        title: 'CryptoAI Trading Platform',
        show: false,
        backgroundColor: '#1a1a2e'
    });

    // Load the main HTML
    mainWindow.loadFile(path.join(__dirname, 'index.html'));

    // Show window when ready
    mainWindow.once('ready-to-show', () => {
        mainWindow.show();
        log.info('Main window displayed');
    });

    // Save window bounds on resize
    mainWindow.on('resize', () => {
        const bounds = mainWindow.getBounds();
        store.set('windowBounds', { width: bounds.width, height: bounds.height });
    });

    // Handle window close
    mainWindow.on('close', async (e) => {
        if (isTrading) {
            e.preventDefault();
            const result = await dialog.showMessageBox(mainWindow, {
                type: 'warning',
                buttons: ['Stop Trading & Exit', 'Cancel'],
                defaultId: 1,
                title: 'Trading Active',
                message: 'Trading is currently active. Do you want to stop trading and exit?'
            });

            if (result.response === 0) {
                await stopTrading();
                mainWindow.destroy();
            }
        }
    });

    mainWindow.on('closed', () => {
        mainWindow = null;
    });

    // Set up menu
    createMenu();

    log.info('Window created successfully');
}

/**
 * Create application menu
 */
function createMenu() {
    const template = [
        {
            label: 'File',
            submenu: [
                {
                    label: 'Open Config',
                    accelerator: 'CmdOrCtrl+O',
                    click: async () => {
                        const result = await dialog.showOpenDialog(mainWindow, {
                            properties: ['openFile'],
                            filters: [{ name: 'YAML Config', extensions: ['yaml', 'yml'] }]
                        });
                        if (!result.canceled && result.filePaths.length > 0) {
                            store.set('configFile', result.filePaths[0]);
                            mainWindow.webContents.send('config-loaded', result.filePaths[0]);
                        }
                    }
                },
                { type: 'separator' },
                { role: 'quit' }
            ]
        },
        {
            label: 'Trading',
            submenu: [
                {
                    label: 'Start Trading',
                    accelerator: 'F5',
                    click: () => mainWindow.webContents.send('menu-start-trading')
                },
                {
                    label: 'Stop Trading',
                    accelerator: 'F6',
                    click: () => mainWindow.webContents.send('menu-stop-trading')
                },
                { type: 'separator' },
                {
                    label: 'Emergency Stop',
                    accelerator: 'CmdOrCtrl+Shift+X',
                    click: () => emergencyStop()
                }
            ]
        },
        {
            label: 'View',
            submenu: [
                { role: 'reload' },
                { role: 'forceReload' },
                { role: 'toggleDevTools' },
                { type: 'separator' },
                { role: 'resetZoom' },
                { role: 'zoomIn' },
                { role: 'zoomOut' },
                { type: 'separator' },
                { role: 'togglefullscreen' }
            ]
        },
        {
            label: 'Help',
            submenu: [
                {
                    label: 'Documentation',
                    click: async () => {
                        const { shell } = require('electron');
                        await shell.openExternal('https://github.com/your-org/cryptoai');
                    }
                },
                { type: 'separator' },
                {
                    label: 'About',
                    click: () => {
                        dialog.showMessageBox(mainWindow, {
                            type: 'info',
                            title: 'About CryptoAI',
                            message: 'CryptoAI Trading Platform',
                            detail: `Version: ${app.getVersion()}\nElectron: ${process.versions.electron}\nNode: ${process.versions.node}\nPlatform: ${process.platform}`
                        });
                    }
                }
            ]
        }
    ];

    const menu = Menu.buildFromTemplate(template);
    Menu.setApplicationMenu(menu);
}

/**
 * Find Python executable
 */
async function findPython() {
    const customPath = store.get('pythonPath');

    // Try custom path first
    if (customPath && customPath !== 'python') {
        try {
            await execPromise(`"${customPath}" --version`);
            return customPath;
        } catch (e) {
            log.warn(`Custom Python path failed: ${customPath}`);
        }
    }

    // Try common Windows locations
    const pythonLocations = IS_WINDOWS ? [
        'python',
        'python3',
        'py',
        path.join(process.env.LOCALAPPDATA || '', 'Programs', 'Python', 'Python311', 'python.exe'),
        path.join(process.env.LOCALAPPDATA || '', 'Programs', 'Python', 'Python310', 'python.exe'),
        path.join(process.env.LOCALAPPDATA || '', 'Programs', 'Python', 'Python39', 'python.exe'),
        'C:\\Python311\\python.exe',
        'C:\\Python310\\python.exe',
        'C:\\Python39\\python.exe'
    ] : ['python3', 'python'];

    for (const pythonPath of pythonLocations) {
        try {
            await execPromise(`"${pythonPath}" --version`);
            log.info(`Found Python at: ${pythonPath}`);
            return pythonPath;
        } catch (e) {
            continue;
        }
    }

    throw new Error('Python not found. Please install Python 3.9+ and add it to PATH.');
}

/**
 * Promise wrapper for exec
 */
function execPromise(cmd) {
    return new Promise((resolve, reject) => {
        exec(cmd, { shell: IS_WINDOWS ? 'powershell.exe' : '/bin/bash' }, (error, stdout, stderr) => {
            if (error) reject(error);
            else resolve(stdout);
        });
    });
}

/**
 * Start the Python trading backend
 */
async function startTrading(config) {
    if (pythonProcess) {
        log.warn('Trading already running');
        return { success: false, error: 'Trading already running' };
    }

    try {
        const pythonPath = await findPython();
        const projectRoot = app.isPackaged
            ? path.join(process.resourcesPath)
            : path.join(__dirname, '..', '..');

        const args = [
            '-m', 'cryptoai.cli',
            'run',
            '--config', config.configFile || store.get('configFile'),
            '--asset', config.asset || store.get('lastAsset'),
            '--mode', config.mode || store.get('mode')
        ];

        log.info(`Starting Python: ${pythonPath} ${args.join(' ')}`);
        log.info(`Working directory: ${projectRoot}`);

        // Spawn Python process with proper Windows handling
        pythonProcess = spawn(pythonPath, args, {
            cwd: projectRoot,
            env: {
                ...process.env,
                PYTHONUNBUFFERED: '1',
                PYTHONIOENCODING: 'utf-8',
                CRYPTOAI_DESKTOP: '1'
            },
            stdio: ['pipe', 'pipe', 'pipe'],
            shell: IS_WINDOWS,
            windowsHide: true
        });

        isTrading = true;
        mainWindow.webContents.send('trading-status', { status: 'running' });

        // Handle stdout
        pythonProcess.stdout.on('data', (data) => {
            const output = data.toString().trim();
            log.info(`[Python] ${output}`);
            mainWindow.webContents.send('trading-log', { type: 'info', message: output });
        });

        // Handle stderr
        pythonProcess.stderr.on('data', (data) => {
            const output = data.toString().trim();
            log.error(`[Python Error] ${output}`);
            mainWindow.webContents.send('trading-log', { type: 'error', message: output });
        });

        // Handle process exit
        pythonProcess.on('exit', (code, signal) => {
            log.info(`Python process exited with code ${code}, signal ${signal}`);
            pythonProcess = null;
            isTrading = false;
            mainWindow.webContents.send('trading-status', { status: 'stopped', code, signal });
        });

        // Handle process error
        pythonProcess.on('error', (err) => {
            log.error(`Python process error: ${err.message}`);
            mainWindow.webContents.send('trading-log', { type: 'error', message: err.message });
            pythonProcess = null;
            isTrading = false;
        });

        // Save last used settings
        store.set('lastAsset', config.asset || store.get('lastAsset'));
        store.set('mode', config.mode || store.get('mode'));

        return { success: true };
    } catch (error) {
        log.error(`Failed to start trading: ${error.message}`);
        return { success: false, error: error.message };
    }
}

/**
 * Stop the Python trading backend
 */
async function stopTrading() {
    if (!pythonProcess) {
        log.warn('No trading process to stop');
        return { success: true };
    }

    log.info('Stopping trading...');
    mainWindow.webContents.send('trading-log', { type: 'info', message: 'Stopping trading...' });

    return new Promise((resolve) => {
        const timeout = setTimeout(() => {
            if (pythonProcess) {
                log.warn('Force killing Python process');
                if (IS_WINDOWS) {
                    // Windows: Use taskkill for process tree
                    exec(`taskkill /pid ${pythonProcess.pid} /T /F`, (err) => {
                        if (err) log.error(`taskkill error: ${err.message}`);
                    });
                } else {
                    pythonProcess.kill('SIGKILL');
                }
            }
            resolve({ success: true, forced: true });
        }, 10000); // 10 second timeout

        pythonProcess.once('exit', () => {
            clearTimeout(timeout);
            pythonProcess = null;
            isTrading = false;
            resolve({ success: true });
        });

        // Send graceful shutdown signal
        if (IS_WINDOWS) {
            // Windows: Write to stdin to signal shutdown
            pythonProcess.stdin.write('SHUTDOWN\n');
            pythonProcess.stdin.end();
        } else {
            pythonProcess.kill('SIGTERM');
        }
    });
}

/**
 * Emergency stop - immediate termination
 */
async function emergencyStop() {
    log.warn('EMERGENCY STOP TRIGGERED');
    mainWindow.webContents.send('trading-log', { type: 'error', message: 'EMERGENCY STOP TRIGGERED' });

    if (pythonProcess) {
        if (IS_WINDOWS) {
            exec(`taskkill /pid ${pythonProcess.pid} /T /F`, (err) => {
                if (err) log.error(`Emergency stop error: ${err.message}`);
            });
        } else {
            pythonProcess.kill('SIGKILL');
        }
        pythonProcess = null;
    }

    isTrading = false;
    mainWindow.webContents.send('trading-status', { status: 'emergency_stopped' });

    return { success: true };
}

/**
 * Get current trading status
 */
function getTradingStatus() {
    return {
        isTrading,
        pid: pythonProcess ? pythonProcess.pid : null
    };
}

/**
 * Check if Python and dependencies are available
 */
async function checkEnvironment() {
    try {
        const pythonPath = await findPython();

        // Check Python version
        const version = await execPromise(`"${pythonPath}" --version`);
        const versionMatch = version.match(/Python (\d+)\.(\d+)/);

        if (!versionMatch || parseInt(versionMatch[1]) < 3 ||
            (parseInt(versionMatch[1]) === 3 && parseInt(versionMatch[2]) < 9)) {
            return {
                success: false,
                error: `Python 3.9+ required. Found: ${version.trim()}`
            };
        }

        // Check if cryptoai is importable
        try {
            await execPromise(`"${pythonPath}" -c "import cryptoai"`);
        } catch (e) {
            return {
                success: false,
                error: 'CryptoAI package not installed. Run: pip install -e .'
            };
        }

        return {
            success: true,
            pythonPath,
            pythonVersion: version.trim()
        };
    } catch (error) {
        return { success: false, error: error.message };
    }
}

// ============================================================================
// Simulation Mode - Real Price Feed Integration
// ============================================================================

let simulationInterval = null;
let simulationState = {
    running: false,
    symbol: 'BTCUSDT',
    lastPrice: 0,
    entryPrice: 0,
    position: 0,  // 0 = none, 1 = long, -1 = short
    pnl: 0,
    trades: 0,
};

/**
 * Fetch real-time price from Binance public API (no key required)
 */
async function fetchRealPrice(symbol) {
    const https = require('https');

    return new Promise((resolve, reject) => {
        const url = `https://api.binance.com/api/v3/ticker/24hr?symbol=${symbol}`;

        https.get(url, (res) => {
            let data = '';
            res.on('data', (chunk) => data += chunk);
            res.on('end', () => {
                try {
                    const parsed = JSON.parse(data);
                    resolve({
                        symbol: parsed.symbol,
                        price: parseFloat(parsed.lastPrice),
                        change24h: parseFloat(parsed.priceChangePercent),
                        high24h: parseFloat(parsed.highPrice),
                        low24h: parseFloat(parsed.lowPrice),
                        volume24h: parseFloat(parsed.quoteVolume),
                        timestamp: new Date().toISOString(),
                    });
                } catch (e) {
                    reject(e);
                }
            });
        }).on('error', reject);
    });
}

/**
 * Simple AI signal generator for simulation (mock prediction)
 * In production, this would call the actual ML model
 */
function generateSimulationSignal(priceHistory) {
    if (priceHistory.length < 5) {
        return { signal: 'HOLD', confidence: 0.5, reason: 'Insufficient data' };
    }

    // Simple momentum-based signal for demo
    const recent = priceHistory.slice(-5);
    const avg = recent.reduce((a, b) => a + b, 0) / recent.length;
    const current = recent[recent.length - 1];
    const momentum = (current - avg) / avg;

    // Add some randomness to simulate uncertainty
    const noise = (Math.random() - 0.5) * 0.01;
    const adjustedMomentum = momentum + noise;

    if (adjustedMomentum > 0.005) {
        return {
            signal: 'LONG',
            confidence: Math.min(0.95, 0.6 + Math.abs(adjustedMomentum) * 10),
            reason: 'Bullish momentum detected',
        };
    } else if (adjustedMomentum < -0.005) {
        return {
            signal: 'SHORT',
            confidence: Math.min(0.95, 0.6 + Math.abs(adjustedMomentum) * 10),
            reason: 'Bearish momentum detected',
        };
    }

    return {
        signal: 'HOLD',
        confidence: 0.5 + Math.random() * 0.2,
        reason: 'No clear signal',
    };
}

/**
 * Start simulation with real price feed
 */
async function startSimulation(config) {
    if (simulationInterval) {
        return { success: false, error: 'Simulation already running' };
    }

    const symbol = config.symbol || 'BTCUSDT';
    const priceHistory = [];

    simulationState = {
        running: true,
        symbol,
        lastPrice: 0,
        entryPrice: 0,
        position: 0,
        pnl: 0,
        trades: 0,
    };

    log.info(`Starting simulation for ${symbol}`);
    mainWindow.webContents.send('trading-log', {
        type: 'info',
        message: `Simulation started for ${symbol} with REAL price feed`,
    });

    // Fetch initial price
    try {
        const initialPrice = await fetchRealPrice(symbol);
        simulationState.lastPrice = initialPrice.price;
        priceHistory.push(initialPrice.price);

        mainWindow.webContents.send('price-update', initialPrice);
        mainWindow.webContents.send('trading-log', {
            type: 'info',
            message: `Initial price: $${initialPrice.price.toLocaleString()}`,
        });
    } catch (e) {
        log.error(`Failed to fetch initial price: ${e.message}`);
        return { success: false, error: e.message };
    }

    // Start price polling interval (every 5 seconds)
    simulationInterval = setInterval(async () => {
        try {
            const priceData = await fetchRealPrice(symbol);
            const currentPrice = priceData.price;

            priceHistory.push(currentPrice);
            if (priceHistory.length > 100) {
                priceHistory.shift();
            }

            simulationState.lastPrice = currentPrice;

            // Send price update to renderer
            mainWindow.webContents.send('price-update', priceData);

            // Generate AI signal
            const signalData = generateSimulationSignal(priceHistory);
            mainWindow.webContents.send('simulation-signal', signalData);

            // Simulate trades based on signals
            if (signalData.signal === 'LONG' && simulationState.position <= 0) {
                // Close short if any, open long
                if (simulationState.position < 0 && simulationState.entryPrice > 0) {
                    const closePnl = (simulationState.entryPrice - currentPrice) * 0.001; // 0.001 BTC
                    simulationState.pnl += closePnl;
                    mainWindow.webContents.send('trading-log', {
                        type: 'trade',
                        message: `[SIM] Closed SHORT at $${currentPrice.toLocaleString()} | PnL: ${closePnl >= 0 ? '+' : ''}$${closePnl.toFixed(2)}`,
                    });
                }
                simulationState.position = 1;
                simulationState.entryPrice = currentPrice;
                simulationState.trades++;
                mainWindow.webContents.send('trading-log', {
                    type: 'trade',
                    message: `[SIM] Opened LONG at $${currentPrice.toLocaleString()} (Confidence: ${(signalData.confidence * 100).toFixed(1)}%)`,
                });
            } else if (signalData.signal === 'SHORT' && simulationState.position >= 0) {
                // Close long if any, open short
                if (simulationState.position > 0 && simulationState.entryPrice > 0) {
                    const closePnl = (currentPrice - simulationState.entryPrice) * 0.001;
                    simulationState.pnl += closePnl;
                    mainWindow.webContents.send('trading-log', {
                        type: 'trade',
                        message: `[SIM] Closed LONG at $${currentPrice.toLocaleString()} | PnL: ${closePnl >= 0 ? '+' : ''}$${closePnl.toFixed(2)}`,
                    });
                }
                simulationState.position = -1;
                simulationState.entryPrice = currentPrice;
                simulationState.trades++;
                mainWindow.webContents.send('trading-log', {
                    type: 'trade',
                    message: `[SIM] Opened SHORT at $${currentPrice.toLocaleString()} (Confidence: ${(signalData.confidence * 100).toFixed(1)}%)`,
                });
            }

            // Calculate unrealized PnL
            let unrealizedPnl = 0;
            if (simulationState.position !== 0 && simulationState.entryPrice > 0) {
                if (simulationState.position > 0) {
                    unrealizedPnl = (currentPrice - simulationState.entryPrice) * 0.001;
                } else {
                    unrealizedPnl = (simulationState.entryPrice - currentPrice) * 0.001;
                }
            }

            mainWindow.webContents.send('simulation-pnl', {
                realized: simulationState.pnl,
                unrealized: unrealizedPnl,
                total: simulationState.pnl + unrealizedPnl,
                trades: simulationState.trades,
                position: simulationState.position,
            });

        } catch (e) {
            log.error(`Price fetch error: ${e.message}`);
            mainWindow.webContents.send('trading-log', {
                type: 'error',
                message: `Price fetch error: ${e.message}`,
            });
        }
    }, 5000); // 5 second interval

    return { success: true };
}

/**
 * Stop simulation
 */
function stopSimulation() {
    if (simulationInterval) {
        clearInterval(simulationInterval);
        simulationInterval = null;
    }
    simulationState.running = false;

    log.info('Simulation stopped');
    mainWindow.webContents.send('trading-log', {
        type: 'info',
        message: `Simulation stopped. Total PnL: $${simulationState.pnl.toFixed(2)} | Trades: ${simulationState.trades}`,
    });

    return { success: true, finalPnl: simulationState.pnl, trades: simulationState.trades };
}

// ============================================================================
// IPC Handlers
// ============================================================================

ipcMain.handle('start-trading', async (event, config) => {
    return await startTrading(config);
});

ipcMain.handle('stop-trading', async () => {
    return await stopTrading();
});

ipcMain.handle('emergency-stop', async () => {
    return await emergencyStop();
});

ipcMain.handle('get-status', () => {
    return getTradingStatus();
});

ipcMain.handle('check-environment', async () => {
    return await checkEnvironment();
});

ipcMain.handle('get-config', () => {
    return {
        pythonPath: store.get('pythonPath'),
        configFile: store.get('configFile'),
        lastAsset: store.get('lastAsset'),
        mode: store.get('mode')
    };
});

ipcMain.handle('set-config', (event, config) => {
    Object.entries(config).forEach(([key, value]) => {
        store.set(key, value);
    });
    return { success: true };
});

// Simulation handlers
ipcMain.handle('start-simulation', async (event, config) => {
    return await startSimulation(config);
});

ipcMain.handle('stop-simulation', () => {
    return stopSimulation();
});

ipcMain.handle('get-simulation-status', () => {
    return {
        running: simulationState.running,
        symbol: simulationState.symbol,
        lastPrice: simulationState.lastPrice,
        pnl: simulationState.pnl,
        trades: simulationState.trades,
        position: simulationState.position,
    };
});

// App lifecycle
app.whenReady().then(async () => {
    log.info('App ready, creating window...');
    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

app.on('window-all-closed', async () => {
    if (pythonProcess) {
        await stopTrading();
    }
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('before-quit', async (e) => {
    if (pythonProcess) {
        e.preventDefault();
        await stopTrading();
        app.quit();
    }
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
    log.error('Uncaught exception:', error);
    if (pythonProcess) {
        pythonProcess.kill('SIGKILL');
    }
});

log.info('CryptoAI Desktop starting...');
