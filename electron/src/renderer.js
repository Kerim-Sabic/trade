/**
 * CryptoAI Desktop - Renderer Script
 *
 * Handles all UI interactions and communication with main process.
 */

// DOM Elements
const statusIndicator = document.getElementById('status-indicator');
const statusText = statusIndicator.querySelector('.status-text');
const envStatus = document.getElementById('env-status');
const assetSelect = document.getElementById('asset-select');
const modeSelect = document.getElementById('mode-select');
const configFile = document.getElementById('config-file');
const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const btnEmergency = document.getElementById('btn-emergency');
const btnClearLog = document.getElementById('btn-clear-log');
const logContainer = document.getElementById('log-container');
const platformInfo = document.getElementById('platform-info');
const dialogOverlay = document.getElementById('dialog-overlay');
const dialogTitle = document.getElementById('dialog-title');
const dialogMessage = document.getElementById('dialog-message');
const dialogCancel = document.getElementById('dialog-cancel');
const dialogConfirm = document.getElementById('dialog-confirm');

// State
let isTrading = false;
let isSimulating = false;
let environmentReady = false;
let dialogCallback = null;
let currentPrice = 0;
let simulationPnL = 0;

/**
 * Initialize the application
 */
async function init() {
    // Set platform info
    if (window.platform) {
        platformInfo.textContent = `${window.platform.isWindows ? 'Windows' : 'Linux/Mac'} | Electron ${window.platform.versions.electron}`;
    }

    // Load saved configuration
    try {
        const config = await window.cryptoai.getConfig();
        assetSelect.value = config.lastAsset || 'BTCUSDT';
        modeSelect.value = config.mode || 'paper';
        configFile.value = config.configFile || 'configs/default.yaml';
    } catch (e) {
        addLog('error', 'Failed to load configuration');
    }

    // Check environment
    await checkEnvironment();

    // Set up event listeners
    setupEventListeners();

    // Register IPC listeners
    registerIPCListeners();

    // Check initial status
    await updateStatus();

    addLog('info', 'CryptoAI Trading Platform initialized');
}

/**
 * Check if Python environment is ready
 */
async function checkEnvironment() {
    envStatus.innerHTML = '<div class="env-checking"><span class="spinner"></span>Checking environment...</div>';
    envStatus.className = 'env-status';

    try {
        const result = await window.cryptoai.checkEnvironment();

        if (result.success) {
            envStatus.className = 'env-status success';
            envStatus.innerHTML = `
                <div style="color: var(--accent-green);">&#x2713; Environment Ready</div>
                <div style="font-size: 12px; color: var(--text-secondary); margin-top: 4px;">
                    ${result.pythonVersion}
                </div>
            `;
            environmentReady = true;
            btnStart.disabled = false;
            addLog('success', `Environment check passed: ${result.pythonVersion}`);
        } else {
            throw new Error(result.error);
        }
    } catch (e) {
        envStatus.className = 'env-status error';
        envStatus.innerHTML = `
            <div style="color: var(--accent-red);">&#x2717; Environment Error</div>
            <div style="font-size: 12px; color: var(--text-secondary); margin-top: 4px;">
                ${e.message || 'Unknown error'}
            </div>
        `;
        environmentReady = false;
        btnStart.disabled = true;
        addLog('error', `Environment check failed: ${e.message}`);
    }
}

/**
 * Set up DOM event listeners
 */
function setupEventListeners() {
    // Start button
    btnStart.addEventListener('click', async () => {
        const config = {
            asset: assetSelect.value,
            mode: modeSelect.value,
            configFile: configFile.value,
            symbol: assetSelect.value
        };

        // Simulation mode doesn't require Python environment
        if (config.mode === 'simulation') {
            addLog('info', `Starting SIMULATION with REAL prices: ${config.asset}...`);
            try {
                const result = await window.cryptoai.startSimulation(config);
                if (result.success) {
                    isSimulating = true;
                    updateStatusUI('simulation');
                    addLog('success', 'Simulation started with real Binance price feed');
                } else {
                    addLog('error', `Failed to start simulation: ${result.error}`);
                }
            } catch (e) {
                addLog('error', `Error starting simulation: ${e.message}`);
            }
            return;
        }

        // Other modes require Python
        if (!environmentReady) {
            addLog('error', 'Environment not ready - Python required for this mode');
            return;
        }

        addLog('info', `Starting trading: ${config.asset} in ${config.mode} mode...`);

        try {
            const result = await window.cryptoai.startTrading(config);
            if (result.success) {
                addLog('success', 'Trading started successfully');
            } else {
                addLog('error', `Failed to start: ${result.error}`);
            }
        } catch (e) {
            addLog('error', `Error starting trading: ${e.message}`);
        }
    });

    // Stop button
    btnStop.addEventListener('click', async () => {
        if (isSimulating) {
            addLog('info', 'Stopping simulation...');
            try {
                const result = await window.cryptoai.stopSimulation();
                if (result.success) {
                    isSimulating = false;
                    updateStatusUI('stopped');
                    addLog('success', `Simulation stopped. Final PnL: $${result.finalPnl?.toFixed(2) || 0}`);
                }
            } catch (e) {
                addLog('error', `Error stopping simulation: ${e.message}`);
            }
            return;
        }

        addLog('info', 'Stopping trading...');

        try {
            const result = await window.cryptoai.stopTrading();
            if (result.success) {
                addLog('success', result.forced ? 'Trading force stopped' : 'Trading stopped gracefully');
            } else {
                addLog('error', `Failed to stop: ${result.error}`);
            }
        } catch (e) {
            addLog('error', `Error stopping trading: ${e.message}`);
        }
    });

    // Emergency stop button
    btnEmergency.addEventListener('click', async () => {
        showDialog(
            'Emergency Stop',
            'This will immediately terminate all trading activity. Are you sure?',
            async (confirmed) => {
                if (confirmed) {
                    addLog('warning', 'EMERGENCY STOP INITIATED');
                    try {
                        await window.cryptoai.emergencyStop();
                        addLog('warning', 'All trading terminated');
                    } catch (e) {
                        addLog('error', `Emergency stop error: ${e.message}`);
                    }
                }
            }
        );
    });

    // Clear log button
    btnClearLog.addEventListener('click', () => {
        logContainer.innerHTML = '';
        addLog('info', 'Log cleared');
    });

    // Save config on change
    assetSelect.addEventListener('change', () => {
        window.cryptoai.setConfig({ lastAsset: assetSelect.value });
    });

    modeSelect.addEventListener('change', () => {
        window.cryptoai.setConfig({ mode: modeSelect.value });
    });

    // Dialog buttons
    dialogCancel.addEventListener('click', () => {
        hideDialog();
        if (dialogCallback) dialogCallback(false);
    });

    dialogConfirm.addEventListener('click', () => {
        hideDialog();
        if (dialogCallback) dialogCallback(true);
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // F5 - Start
        if (e.key === 'F5' && !isTrading && environmentReady) {
            btnStart.click();
        }
        // F6 - Stop
        if (e.key === 'F6' && isTrading) {
            btnStop.click();
        }
        // Ctrl+Shift+X - Emergency Stop
        if (e.ctrlKey && e.shiftKey && e.key === 'X') {
            btnEmergency.click();
        }
        // Escape - Close dialog
        if (e.key === 'Escape' && dialogOverlay.style.display !== 'none') {
            hideDialog();
            if (dialogCallback) dialogCallback(false);
        }
    });
}

/**
 * Register IPC event listeners
 */
function registerIPCListeners() {
    // Trading log
    window.cryptoai.onTradingLog((data) => {
        addLog(data.type, data.message);
    });

    // Trading status
    window.cryptoai.onTradingStatus((data) => {
        updateStatusUI(data.status);

        if (data.status === 'stopped' || data.status === 'emergency_stopped') {
            if (data.code !== 0 && data.code !== null) {
                addLog('error', `Trading process exited with code ${data.code}`);
            }
        }
    });

    // Config loaded from menu
    window.cryptoai.onConfigLoaded((path) => {
        configFile.value = path;
        addLog('info', `Configuration loaded: ${path}`);
    });

    // Menu commands
    window.cryptoai.onMenuStartTrading(() => {
        if (!isTrading && !isSimulating) {
            btnStart.click();
        }
    });

    window.cryptoai.onMenuStopTrading(() => {
        if (isTrading || isSimulating) {
            btnStop.click();
        }
    });

    // Simulation events - Real-time price updates
    window.cryptoai.onPriceUpdate((data) => {
        currentPrice = data.price;
        // Log price updates periodically (not every update to avoid spam)
        const priceFormatted = data.price.toLocaleString('en-US', {
            style: 'currency', currency: 'USD', minimumFractionDigits: 2
        });
        // Update status text with current price
        if (isSimulating) {
            const changeClass = data.change24h >= 0 ? 'positive' : 'negative';
            statusText.innerHTML = `<span style="color: var(--accent-green);">${data.symbol}: ${priceFormatted}</span> <span class="${changeClass}">(${data.change24h >= 0 ? '+' : ''}${data.change24h.toFixed(2)}%)</span>`;
        }
    });

    // Simulation AI signals
    window.cryptoai.onSimulationSignal((data) => {
        const signalColors = {
            'LONG': 'var(--accent-green)',
            'SHORT': 'var(--accent-red)',
            'HOLD': 'var(--text-secondary)'
        };
        addLog('info', `AI Signal: ${data.signal} (${(data.confidence * 100).toFixed(1)}% confidence) - ${data.reason}`);
    });

    // Simulation PnL updates
    window.cryptoai.onSimulationPnL((data) => {
        simulationPnL = data.total;
        const pnlFormatted = data.total >= 0 ? `+$${data.total.toFixed(2)}` : `-$${Math.abs(data.total).toFixed(2)}`;
        const positionText = data.position > 0 ? 'LONG' : data.position < 0 ? 'SHORT' : 'FLAT';
        // Could update a PnL display element here if we had one
    });
}

/**
 * Update trading status
 */
async function updateStatus() {
    try {
        const status = await window.cryptoai.getStatus();
        updateStatusUI(status.isTrading ? 'running' : 'stopped');
    } catch (e) {
        console.error('Failed to get status:', e);
    }
}

/**
 * Update status UI
 */
function updateStatusUI(status) {
    statusIndicator.className = 'status-indicator';

    switch (status) {
        case 'running':
            isTrading = true;
            isSimulating = false;
            statusIndicator.classList.add('running');
            statusText.textContent = 'Trading Active';
            btnStart.disabled = true;
            btnStop.disabled = false;
            break;

        case 'simulation':
            isTrading = false;
            isSimulating = true;
            statusIndicator.classList.add('running');
            statusIndicator.style.backgroundColor = 'var(--accent-yellow, #f59e0b)';
            statusText.textContent = 'Simulation (Real Prices)';
            btnStart.disabled = true;
            btnStop.disabled = false;
            break;

        case 'stopped':
            isTrading = false;
            isSimulating = false;
            statusIndicator.classList.add('stopped');
            statusIndicator.style.backgroundColor = '';
            statusText.textContent = 'Stopped';
            // In simulation mode, Python environment is not required
            btnStart.disabled = false;
            btnStop.disabled = true;
            break;

        case 'emergency_stopped':
            isTrading = false;
            isSimulating = false;
            statusIndicator.classList.add('error');
            statusIndicator.style.backgroundColor = '';
            statusText.textContent = 'Emergency Stopped';
            btnStart.disabled = false;
            btnStop.disabled = true;
            break;

        default:
            statusIndicator.classList.add('stopped');
            statusText.textContent = 'Unknown';
            break;
    }
}

/**
 * Add log entry
 */
function addLog(type, message) {
    const now = new Date();
    const time = now.toTimeString().split(' ')[0];

    const entry = document.createElement('div');
    entry.className = `log-entry log-${type}`;
    entry.innerHTML = `
        <span class="log-time">[${time}]</span>
        <span class="log-message">${escapeHtml(message)}</span>
    `;

    logContainer.appendChild(entry);
    logContainer.scrollTop = logContainer.scrollHeight;

    // Limit log entries
    while (logContainer.children.length > 500) {
        logContainer.removeChild(logContainer.firstChild);
    }
}

/**
 * Show confirmation dialog
 */
function showDialog(title, message, callback) {
    dialogTitle.textContent = title;
    dialogMessage.textContent = message;
    dialogCallback = callback;
    dialogOverlay.style.display = 'flex';
}

/**
 * Hide dialog
 */
function hideDialog() {
    dialogOverlay.style.display = 'none';
    dialogCallback = null;
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', init);
