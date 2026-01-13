/**
 * CryptoAI Desktop - Preload Script
 *
 * Exposes a secure API to the renderer process.
 * Uses contextBridge for security.
 */

const { contextBridge, ipcRenderer } = require("electron");

// Expose protected methods to renderer
contextBridge.exposeInMainWorld("cryptoai", {
  // Trading controls
  startTrading: (options) => ipcRenderer.invoke("start-trading", options),
  stopTrading: () => ipcRenderer.invoke("stop-trading"),

  // Status
  getStatus: () => ipcRenderer.invoke("get-status"),

  // Settings
  getSettings: () => ipcRenderer.invoke("get-settings"),
  saveSettings: (settings) => ipcRenderer.invoke("save-settings", settings),
  selectConfig: () => ipcRenderer.invoke("select-config"),
  selectPython: () => ipcRenderer.invoke("select-python"),

  // Event listeners
  onPythonLog: (callback) => {
    ipcRenderer.on("python-log", (event, data) => callback(data));
  },
  onPythonStarted: (callback) => {
    ipcRenderer.on("python-started", (event, data) => callback(data));
  },
  onPythonStopped: (callback) => {
    ipcRenderer.on("python-stopped", (event, data) => callback(data));
  },
  onPythonError: (callback) => {
    ipcRenderer.on("python-error", (event, data) => callback(data));
  },
  onOpenSettings: (callback) => {
    ipcRenderer.on("open-settings", () => callback());
  },
  onStartTrading: (callback) => {
    ipcRenderer.on("start-trading", () => callback());
  },
  onStopTrading: (callback) => {
    ipcRenderer.on("stop-trading", () => callback());
  },
  onRunBacktest: (callback) => {
    ipcRenderer.on("run-backtest", () => callback());
  },

  // Remove listeners
  removeAllListeners: (channel) => {
    ipcRenderer.removeAllListeners(channel);
  },
});

// Platform info
contextBridge.exposeInMainWorld("platform", {
  os: process.platform,
  arch: process.arch,
  versions: {
    electron: process.versions.electron,
    node: process.versions.node,
    chrome: process.versions.chrome,
  },
});
