/**
 * CryptoAI Desktop - Renderer Process
 *
 * Handles UI interactions with safety-first principles:
 * - Confirmation dialogs for destructive actions
 * - Clear visibility of mode and risk
 * - Live mode warnings
 */

// DOM Elements
const elements = {
  // Header
  governanceBadge: document.getElementById("governance-badge"),
  emergencyStopBtn: document.getElementById("emergency-stop-btn"),
  settingsBtn: document.getElementById("settings-btn"),

  // Status
  statusIndicator: document.getElementById("status-indicator"),
  currentMode: document.getElementById("current-mode"),
  currentExchange: document.getElementById("current-exchange"),
  currentAssets: document.getElementById("current-assets"),
  uptime: document.getElementById("uptime"),
  governanceState: document.getElementById("governance-state"),

  // Controls
  modeSelect: document.getElementById("mode-select"),
  modeDescription: document.getElementById("mode-description"),
  exchangeSelect: document.getElementById("exchange-select"),
  assetsInput: document.getElementById("assets-input"),
  configPath: document.getElementById("config-path"),
  browseConfig: document.getElementById("browse-config"),
  liveWarning: document.getElementById("live-warning"),
  startBtn: document.getElementById("start-btn"),
  stopBtn: document.getElementById("stop-btn"),

  // Logs
  logsContainer: document.getElementById("logs-container"),
  autoScroll: document.getElementById("auto-scroll"),
  clearLogs: document.getElementById("clear-logs"),

  // Footer
  platformInfo: document.getElementById("platform-info"),
  connectionStatus: document.getElementById("connection-status"),

  // Settings Modal
  settingsModal: document.getElementById("settings-modal"),
  closeSettings: document.getElementById("close-settings"),
  cancelSettings: document.getElementById("cancel-settings"),
  saveSettings: document.getElementById("save-settings"),
  pythonPath: document.getElementById("python-path"),
  browsePython: document.getElementById("browse-python"),
  confirmLiveMode: document.getElementById("confirm-live-mode"),
  maxLeverage: document.getElementById("max-leverage"),
  maxDrawdown: document.getElementById("max-drawdown"),
  defaultConfig: document.getElementById("default-config"),

  // Confirm Modal
  confirmModal: document.getElementById("confirm-modal"),
  confirmTitle: document.getElementById("confirm-title"),
  confirmMessage: document.getElementById("confirm-message"),
  confirmCancel: document.getElementById("confirm-cancel"),
  confirmOk: document.getElementById("confirm-ok"),
};

// State
let state = {
  isRunning: false,
  mode: "shadow",
  exchange: "binance",
  governanceState: "OPERATIONAL",
  startTime: null,
};

const MAX_LOG_ENTRIES = 1000;

// Mode descriptions
const modeDescriptions = {
  shadow: "Simulates trading without any API calls. Zero risk.",
  paper: "Uses exchange testnet. No real money involved.",
  backtest: "Run historical simulations to validate strategy.",
  live: "REAL trades with REAL money. Use with extreme caution.",
};

/**
 * Initialize the application
 */
async function init() {
  // Set platform info
  if (window.platform) {
    const osNames = {
      win32: "Windows",
      darwin: "macOS",
      linux: "Linux",
    };
    elements.platformInfo.textContent = `Platform: ${osNames[window.platform.os] || window.platform.os} (${window.platform.arch})`;
  }

  // Load initial status and settings
  await loadStatus();
  await loadSettings();

  // Setup event listeners
  setupEventListeners();
  setupIpcListeners();

  // Start uptime timer
  setInterval(updateUptime, 1000);

  addLogEntry("info", "System", "CryptoAI Desktop initialized successfully");
}

/**
 * Load current status from main process
 */
async function loadStatus() {
  try {
    const status = await window.cryptoai.getStatus();
    state.isRunning = status.running;
    state.mode = status.mode || "shadow";
    state.governanceState = status.governanceState || "OPERATIONAL";
    state.startTime = status.startTime ? new Date(status.startTime) : null;

    updateStatusUI();

    if (status.settings) {
      elements.modeSelect.value = status.settings.mode || "shadow";
      elements.exchangeSelect.value = status.settings.exchange || "binance";
      elements.assetsInput.value = (status.settings.assets || ["BTCUSDT"]).join(", ");
      elements.configPath.value = status.settings.configPath || "configs/default.yaml";
      updateModeDescription();
    }
  } catch (error) {
    addLogEntry("error", "Error", `Failed to load status: ${error.message}`);
  }
}

/**
 * Load settings from main process
 */
async function loadSettings() {
  try {
    const settings = await window.cryptoai.getSettings();
    elements.pythonPath.value = settings.pythonPath || "python";
    elements.defaultConfig.value = settings.configPath || "configs/default.yaml";
    elements.configPath.value = settings.configPath || "configs/default.yaml";
    elements.confirmLiveMode.checked = settings.confirmLiveMode !== false;
    elements.maxLeverage.value = settings.maxLeverage || 3;
    elements.maxDrawdown.value = (settings.maxDrawdown || 0.10) * 100;
  } catch (error) {
    addLogEntry("error", "Error", `Failed to load settings: ${error.message}`);
  }
}

/**
 * Update the UI based on running state
 */
function updateStatusUI() {
  const statusText = elements.statusIndicator.querySelector("span:last-child") || elements.statusIndicator;

  if (state.isRunning) {
    statusText.textContent = "Running";
    elements.statusIndicator.className = "status-value status-running";
    elements.startBtn.disabled = true;
    elements.stopBtn.disabled = false;
    elements.modeSelect.disabled = true;
    elements.exchangeSelect.disabled = true;
    elements.assetsInput.disabled = true;
    elements.connectionStatus.className = "connection-online";
    elements.connectionStatus.innerHTML = '<span class="connection-dot"></span>Connected';
  } else {
    statusText.textContent = "Stopped";
    elements.statusIndicator.className = "status-value status-stopped";
    elements.startBtn.disabled = false;
    elements.stopBtn.disabled = true;
    elements.modeSelect.disabled = false;
    elements.exchangeSelect.disabled = false;
    elements.assetsInput.disabled = false;
    elements.connectionStatus.className = "connection-offline";
    elements.connectionStatus.innerHTML = '<span class="connection-dot"></span>Disconnected';
  }

  // Update governance badge
  updateGovernanceBadge(state.governanceState);
}

/**
 * Update governance badge
 */
function updateGovernanceBadge(governanceState) {
  const badge = elements.governanceBadge;
  const badgeText = badge.querySelector(".badge-text");

  badge.className = "governance-badge";

  switch (governanceState) {
    case "OPERATIONAL":
      badgeText.textContent = "OPERATIONAL";
      break;
    case "RESTRICTED":
      badge.classList.add("restricted");
      badgeText.textContent = "RESTRICTED";
      break;
    case "SUSPENDED":
    case "HALTED":
      badge.classList.add("halted");
      badgeText.textContent = governanceState;
      break;
    default:
      badgeText.textContent = governanceState || "UNKNOWN";
  }

  // Update status panel
  elements.governanceState.textContent = governanceState || "Unknown";
  elements.governanceState.className = `status-value governance-${governanceState.toLowerCase()}`;
}

/**
 * Update mode description
 */
function updateModeDescription() {
  const mode = elements.modeSelect.value;
  elements.modeDescription.textContent = modeDescriptions[mode] || "";

  // Show/hide live warning
  if (mode === "live") {
    elements.liveWarning.classList.remove("hidden");
  } else {
    elements.liveWarning.classList.add("hidden");
  }

  // Update current mode display
  elements.currentMode.textContent = mode.charAt(0).toUpperCase() + mode.slice(1);
  elements.currentMode.className = `status-value mode-${mode}`;
}

/**
 * Update uptime display
 */
function updateUptime() {
  if (state.isRunning && state.startTime) {
    const elapsed = Date.now() - state.startTime.getTime();
    const hours = Math.floor(elapsed / 3600000);
    const minutes = Math.floor((elapsed % 3600000) / 60000);
    const seconds = Math.floor((elapsed % 60000) / 1000);
    elements.uptime.textContent = `${hours.toString().padStart(2, "0")}:${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
  } else {
    elements.uptime.textContent = "--:--:--";
  }
}

/**
 * Setup DOM event listeners
 */
function setupEventListeners() {
  // Mode change
  elements.modeSelect.addEventListener("change", updateModeDescription);

  // Exchange change
  elements.exchangeSelect.addEventListener("change", () => {
    elements.currentExchange.textContent = elements.exchangeSelect.options[elements.exchangeSelect.selectedIndex].text;
  });

  // Assets change
  elements.assetsInput.addEventListener("change", () => {
    elements.currentAssets.textContent = elements.assetsInput.value || "None";
  });

  // Start trading
  elements.startBtn.addEventListener("click", async () => {
    const mode = elements.modeSelect.value;
    const exchange = elements.exchangeSelect.value;
    const assets = elements.assetsInput.value.split(",").map((a) => a.trim()).filter(Boolean);
    const config = elements.configPath.value;

    addLogEntry("info", "System", `Starting ${mode} trading on ${exchange} for ${assets.join(", ")}...`);

    const result = await window.cryptoai.startTrading({ mode, assets, config, exchange });

    if (!result.success) {
      if (result.error === "Cancelled by user") {
        addLogEntry("info", "System", "Start cancelled by user");
      } else {
        addLogEntry("error", "Error", `Failed to start: ${result.error}`);
      }
    }
  });

  // Stop trading
  elements.stopBtn.addEventListener("click", async () => {
    addLogEntry("info", "System", "Stopping trading...");
    await window.cryptoai.stopTrading();
  });

  // Emergency stop
  elements.emergencyStopBtn.addEventListener("click", async () => {
    addLogEntry("error", "EMERGENCY", "Emergency stop activated!");
    await window.cryptoai.emergencyStop();
  });

  // Browse config
  elements.browseConfig.addEventListener("click", async () => {
    const result = await window.cryptoai.selectConfig();
    if (result.success) {
      elements.configPath.value = result.path;
    }
  });

  // Clear logs
  elements.clearLogs.addEventListener("click", () => {
    elements.logsContainer.innerHTML = "";
    addLogEntry("info", "System", "Logs cleared");
  });

  // Settings modal
  elements.settingsBtn.addEventListener("click", () => {
    elements.settingsModal.classList.remove("hidden");
  });

  elements.closeSettings.addEventListener("click", () => {
    elements.settingsModal.classList.add("hidden");
  });

  elements.cancelSettings.addEventListener("click", () => {
    elements.settingsModal.classList.add("hidden");
  });

  // Browse Python
  elements.browsePython.addEventListener("click", async () => {
    const result = await window.cryptoai.selectPython();
    if (result.success) {
      elements.pythonPath.value = result.path;
    }
  });

  // Save settings
  elements.saveSettings.addEventListener("click", async () => {
    const settings = {
      pythonPath: elements.pythonPath.value,
      configPath: elements.defaultConfig.value,
      confirmLiveMode: elements.confirmLiveMode.checked,
      maxLeverage: parseInt(elements.maxLeverage.value) || 3,
      maxDrawdown: (parseInt(elements.maxDrawdown.value) || 10) / 100,
    };

    const result = await window.cryptoai.saveSettings(settings);
    if (result.success) {
      addLogEntry("info", "System", "Settings saved");
      elements.settingsModal.classList.add("hidden");
      elements.configPath.value = settings.configPath;
    } else {
      addLogEntry("error", "Error", `Failed to save settings: ${result.error}`);
    }
  });

  // Close modal on overlay click
  document.querySelectorAll(".modal-overlay").forEach((overlay) => {
    overlay.addEventListener("click", () => {
      overlay.parentElement.classList.add("hidden");
    });
  });

  // Keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    // Ctrl+Shift+K - Emergency stop
    if (e.ctrlKey && e.shiftKey && e.key === "K") {
      e.preventDefault();
      elements.emergencyStopBtn.click();
    }
  });
}

/**
 * Setup IPC event listeners
 */
function setupIpcListeners() {
  // Python log output
  window.cryptoai.onPythonLog((data) => {
    const logType = data.type === "stderr" ? "stderr" : "stdout";
    addLogEntry(logType, "Python", data.message);
  });

  // Python started
  window.cryptoai.onPythonStarted((data) => {
    state.isRunning = true;
    state.mode = data.mode;
    state.startTime = new Date();
    updateStatusUI();
    elements.currentMode.textContent = data.mode.charAt(0).toUpperCase() + data.mode.slice(1);
    elements.currentMode.className = `status-value mode-${data.mode}`;
    elements.currentAssets.textContent = data.assets.join(", ");
    elements.currentExchange.textContent = data.exchange ? data.exchange.charAt(0).toUpperCase() + data.exchange.slice(1) : "Binance";
    addLogEntry("info", "System", `Trading started in ${data.mode} mode on ${data.exchange || "binance"}`);
  });

  // Python stopped
  window.cryptoai.onPythonStopped((data) => {
    state.isRunning = false;
    state.startTime = null;
    updateStatusUI();
    addLogEntry("info", "System", `Trading stopped (exit code: ${data.code})`);
  });

  // Python error
  window.cryptoai.onPythonError((data) => {
    state.isRunning = false;
    updateStatusUI();
    addLogEntry("error", "Error", data.error);
    const statusText = elements.statusIndicator.querySelector("span:last-child") || elements.statusIndicator;
    statusText.textContent = "Error";
    elements.statusIndicator.className = "status-value status-error";
  });

  // Emergency stop
  window.cryptoai.onEmergencyStop(() => {
    state.isRunning = false;
    state.governanceState = "HALTED";
    updateStatusUI();
    addLogEntry("error", "EMERGENCY", "Emergency stop activated by system");
  });

  // Menu commands
  window.cryptoai.onOpenSettings(() => {
    elements.settingsModal.classList.remove("hidden");
  });

  window.cryptoai.onMenuStartTrading(() => {
    if (!state.isRunning) {
      elements.startBtn.click();
    }
  });

  window.cryptoai.onMenuStopTrading(() => {
    if (state.isRunning) {
      elements.stopBtn.click();
    }
  });

  window.cryptoai.onRunBacktest(() => {
    elements.modeSelect.value = "backtest";
    updateModeDescription();
    addLogEntry("info", "System", "Mode set to backtest");
  });
}

/**
 * Add a log entry to the logs container
 */
function addLogEntry(type, source, message) {
  const entry = document.createElement("div");
  entry.className = `log-entry log-${type}`;

  const time = new Date().toLocaleTimeString();
  entry.innerHTML = `
    <span class="log-time">[${time}] [${source}]</span>
    <span class="log-message">${escapeHtml(message)}</span>
  `;

  elements.logsContainer.appendChild(entry);

  // Limit log entries
  while (elements.logsContainer.children.length > MAX_LOG_ENTRIES) {
    elements.logsContainer.removeChild(elements.logsContainer.firstChild);
  }

  // Auto-scroll to bottom if enabled
  if (elements.autoScroll.checked) {
    elements.logsContainer.scrollTop = elements.logsContainer.scrollHeight;
  }
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", init);
