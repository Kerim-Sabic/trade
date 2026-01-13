/**
 * CryptoAI Desktop - Renderer Process
 *
 * Handles UI interactions and communicates with main process.
 */

// DOM Elements
const elements = {
  // Status
  statusIndicator: document.getElementById("status-indicator"),
  currentMode: document.getElementById("current-mode"),
  currentAssets: document.getElementById("current-assets"),

  // Controls
  modeSelect: document.getElementById("mode-select"),
  assetsInput: document.getElementById("assets-input"),
  configPath: document.getElementById("config-path"),
  browseConfig: document.getElementById("browse-config"),
  startBtn: document.getElementById("start-btn"),
  stopBtn: document.getElementById("stop-btn"),

  // Logs
  logsContainer: document.getElementById("logs-container"),
  clearLogs: document.getElementById("clear-logs"),

  // Settings
  settingsBtn: document.getElementById("settings-btn"),
  settingsModal: document.getElementById("settings-modal"),
  closeSettings: document.getElementById("close-settings"),
  pythonPath: document.getElementById("python-path"),
  browsePython: document.getElementById("browse-python"),
  defaultConfig: document.getElementById("default-config"),
  saveSettings: document.getElementById("save-settings"),
};

// State
let isRunning = false;
const MAX_LOG_ENTRIES = 500;

/**
 * Initialize the application
 */
async function init() {
  // Load initial status and settings
  await loadStatus();
  await loadSettings();

  // Setup event listeners
  setupEventListeners();

  // Setup IPC listeners
  setupIpcListeners();

  addLogEntry("info", "System", "CryptoAI Desktop initialized successfully");
}

/**
 * Load current status from main process
 */
async function loadStatus() {
  try {
    const status = await window.cryptoai.getStatus();
    isRunning = status.running;
    updateStatusUI();

    if (status.settings) {
      elements.modeSelect.value = status.settings.mode || "shadow";
      elements.assetsInput.value = (status.settings.assets || ["BTCUSDT"]).join(", ");
      elements.configPath.value = status.settings.configPath || "configs/default.yaml";
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
  } catch (error) {
    addLogEntry("error", "Error", `Failed to load settings: ${error.message}`);
  }
}

/**
 * Update the UI based on running state
 */
function updateStatusUI() {
  if (isRunning) {
    elements.statusIndicator.textContent = "Running";
    elements.statusIndicator.className = "status-value status-running";
    elements.startBtn.disabled = true;
    elements.stopBtn.disabled = false;
    elements.modeSelect.disabled = true;
    elements.assetsInput.disabled = true;
  } else {
    elements.statusIndicator.textContent = "Stopped";
    elements.statusIndicator.className = "status-value status-stopped";
    elements.startBtn.disabled = false;
    elements.stopBtn.disabled = true;
    elements.modeSelect.disabled = false;
    elements.assetsInput.disabled = false;
  }
}

/**
 * Setup DOM event listeners
 */
function setupEventListeners() {
  // Start trading
  elements.startBtn.addEventListener("click", async () => {
    const mode = elements.modeSelect.value;
    const assets = elements.assetsInput.value.split(",").map((a) => a.trim()).filter(Boolean);
    const config = elements.configPath.value;

    addLogEntry("info", "System", `Starting ${mode} trading for ${assets.join(", ")}...`);

    const result = await window.cryptoai.startTrading({ mode, assets, config });
    if (!result.success) {
      addLogEntry("error", "Error", `Failed to start: ${result.error}`);
    }
  });

  // Stop trading
  elements.stopBtn.addEventListener("click", async () => {
    addLogEntry("info", "System", "Stopping trading...");
    await window.cryptoai.stopTrading();
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

  // Close modal on outside click
  elements.settingsModal.addEventListener("click", (e) => {
    if (e.target === elements.settingsModal) {
      elements.settingsModal.classList.add("hidden");
    }
  });

  // Mode change
  elements.modeSelect.addEventListener("change", () => {
    const mode = elements.modeSelect.value;
    elements.currentMode.textContent = mode.charAt(0).toUpperCase() + mode.slice(1);
  });

  // Assets change
  elements.assetsInput.addEventListener("change", () => {
    elements.currentAssets.textContent = elements.assetsInput.value || "None";
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
    isRunning = true;
    updateStatusUI();
    elements.currentMode.textContent = data.mode.charAt(0).toUpperCase() + data.mode.slice(1);
    elements.currentAssets.textContent = data.assets.join(", ");
    addLogEntry("info", "System", `Trading started in ${data.mode} mode`);
  });

  // Python stopped
  window.cryptoai.onPythonStopped((data) => {
    isRunning = false;
    updateStatusUI();
    addLogEntry("info", "System", `Trading stopped (exit code: ${data.code})`);
  });

  // Python error
  window.cryptoai.onPythonError((data) => {
    isRunning = false;
    updateStatusUI();
    addLogEntry("error", "Error", data.error);
    elements.statusIndicator.textContent = "Error";
    elements.statusIndicator.className = "status-value status-error";
  });

  // Menu commands
  window.cryptoai.onOpenSettings(() => {
    elements.settingsModal.classList.remove("hidden");
  });

  window.cryptoai.onStartTrading(() => {
    if (!isRunning) {
      elements.startBtn.click();
    }
  });

  window.cryptoai.onStopTrading(() => {
    if (isRunning) {
      elements.stopBtn.click();
    }
  });

  window.cryptoai.onRunBacktest(() => {
    elements.modeSelect.value = "backtest";
    elements.currentMode.textContent = "Backtest";
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

  // Auto-scroll to bottom
  elements.logsContainer.scrollTop = elements.logsContainer.scrollHeight;
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
