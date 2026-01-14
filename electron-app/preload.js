/**
 * CryptoAI Desktop - Preload Script
 *
 * Exposes a secure API to the renderer process.
 * Uses contextBridge for security - NO arbitrary code execution.
 *
 * SECURITY PRINCIPLES:
 * - Only expose specific, validated methods
 * - No access to Node.js APIs from renderer
 * - All IPC is typed and validated
 */

const { contextBridge, ipcRenderer } = require("electron");

// Valid IPC channels - whitelist only
const VALID_SEND_CHANNELS = [
  "start-trading",
  "stop-trading",
  "emergency-stop",
  "get-status",
  "get-settings",
  "save-settings",
  "select-config",
  "select-python",
  "confirm-dialog",
];

const VALID_RECEIVE_CHANNELS = [
  "python-log",
  "python-started",
  "python-stopped",
  "python-error",
  "open-settings",
  "menu-start-trading",
  "menu-stop-trading",
  "run-backtest",
  "emergency-stop",
];

// Expose protected methods to renderer
contextBridge.exposeInMainWorld("cryptoai", {
  // Trading controls
  startTrading: (options) => {
    if (typeof options !== "object") {
      throw new Error("Invalid options");
    }
    return ipcRenderer.invoke("start-trading", options);
  },

  stopTrading: () => ipcRenderer.invoke("stop-trading"),

  emergencyStop: () => ipcRenderer.invoke("emergency-stop"),

  // Status
  getStatus: () => ipcRenderer.invoke("get-status"),

  // Settings
  getSettings: () => ipcRenderer.invoke("get-settings"),

  saveSettings: (settings) => {
    if (typeof settings !== "object") {
      throw new Error("Invalid settings");
    }
    return ipcRenderer.invoke("save-settings", settings);
  },

  selectConfig: () => ipcRenderer.invoke("select-config"),
  selectPython: () => ipcRenderer.invoke("select-python"),

  // Dialogs
  confirmDialog: (options) => {
    if (typeof options !== "object") {
      throw new Error("Invalid options");
    }
    return ipcRenderer.invoke("confirm-dialog", options);
  },

  // Event listeners with validation
  onPythonLog: (callback) => {
    if (typeof callback !== "function") return;
    const handler = (event, data) => callback(data);
    ipcRenderer.on("python-log", handler);
    return () => ipcRenderer.removeListener("python-log", handler);
  },

  onPythonStarted: (callback) => {
    if (typeof callback !== "function") return;
    const handler = (event, data) => callback(data);
    ipcRenderer.on("python-started", handler);
    return () => ipcRenderer.removeListener("python-started", handler);
  },

  onPythonStopped: (callback) => {
    if (typeof callback !== "function") return;
    const handler = (event, data) => callback(data);
    ipcRenderer.on("python-stopped", handler);
    return () => ipcRenderer.removeListener("python-stopped", handler);
  },

  onPythonError: (callback) => {
    if (typeof callback !== "function") return;
    const handler = (event, data) => callback(data);
    ipcRenderer.on("python-error", handler);
    return () => ipcRenderer.removeListener("python-error", handler);
  },

  onOpenSettings: (callback) => {
    if (typeof callback !== "function") return;
    const handler = () => callback();
    ipcRenderer.on("open-settings", handler);
    return () => ipcRenderer.removeListener("open-settings", handler);
  },

  onMenuStartTrading: (callback) => {
    if (typeof callback !== "function") return;
    const handler = () => callback();
    ipcRenderer.on("menu-start-trading", handler);
    return () => ipcRenderer.removeListener("menu-start-trading", handler);
  },

  onMenuStopTrading: (callback) => {
    if (typeof callback !== "function") return;
    const handler = () => callback();
    ipcRenderer.on("menu-stop-trading", handler);
    return () => ipcRenderer.removeListener("menu-stop-trading", handler);
  },

  onRunBacktest: (callback) => {
    if (typeof callback !== "function") return;
    const handler = () => callback();
    ipcRenderer.on("run-backtest", handler);
    return () => ipcRenderer.removeListener("run-backtest", handler);
  },

  onEmergencyStop: (callback) => {
    if (typeof callback !== "function") return;
    const handler = () => callback();
    ipcRenderer.on("emergency-stop", handler);
    return () => ipcRenderer.removeListener("emergency-stop", handler);
  },

  // Remove all listeners for a channel
  removeAllListeners: (channel) => {
    if (VALID_RECEIVE_CHANNELS.includes(channel)) {
      ipcRenderer.removeAllListeners(channel);
    }
  },
});

// Platform info (read-only)
contextBridge.exposeInMainWorld("platform", {
  os: process.platform,
  arch: process.arch,
  isWindows: process.platform === "win32",
  isMac: process.platform === "darwin",
  isLinux: process.platform === "linux",
  versions: {
    electron: process.versions.electron,
    node: process.versions.node,
    chrome: process.versions.chrome,
  },
});

// Available exchanges (static data)
contextBridge.exposeInMainWorld("exchanges", [
  { id: "binance", name: "Binance", testnet: true },
  { id: "bybit", name: "Bybit", testnet: true },
  { id: "okx", name: "OKX", testnet: true },
]);

// Available trading modes with descriptions
contextBridge.exposeInMainWorld("tradingModes", [
  {
    id: "shadow",
    name: "Shadow Mode",
    description: "Simulates trading without any API calls. Zero risk.",
    risk: "none",
    recommended: true,
  },
  {
    id: "paper",
    name: "Paper Trading",
    description: "Uses exchange testnet. No real money involved.",
    risk: "none",
    recommended: true,
  },
  {
    id: "backtest",
    name: "Backtest",
    description: "Run historical simulations to validate strategy.",
    risk: "none",
    recommended: true,
  },
  {
    id: "live",
    name: "Live Trading",
    description: "REAL trades with REAL money. Use with extreme caution.",
    risk: "high",
    recommended: false,
  },
]);
