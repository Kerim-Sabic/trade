/**
 * CryptoAI Desktop - Preload Script
 *
 * Secure bridge between main and renderer processes.
 * Uses contextBridge to expose safe APIs.
 */

const { contextBridge, ipcRenderer } = require('electron');

// Expose protected APIs to the renderer
contextBridge.exposeInMainWorld('cryptoai', {
    // Trading control
    startTrading: (config) => ipcRenderer.invoke('start-trading', config),
    stopTrading: () => ipcRenderer.invoke('stop-trading'),
    emergencyStop: () => ipcRenderer.invoke('emergency-stop'),
    getStatus: () => ipcRenderer.invoke('get-status'),

    // Environment
    checkEnvironment: () => ipcRenderer.invoke('check-environment'),

    // Configuration
    getConfig: () => ipcRenderer.invoke('get-config'),
    setConfig: (config) => ipcRenderer.invoke('set-config', config),

    // Event listeners
    onTradingLog: (callback) => {
        const subscription = (event, data) => callback(data);
        ipcRenderer.on('trading-log', subscription);
        return () => ipcRenderer.removeListener('trading-log', subscription);
    },

    onTradingStatus: (callback) => {
        const subscription = (event, data) => callback(data);
        ipcRenderer.on('trading-status', subscription);
        return () => ipcRenderer.removeListener('trading-status', subscription);
    },

    onConfigLoaded: (callback) => {
        const subscription = (event, path) => callback(path);
        ipcRenderer.on('config-loaded', subscription);
        return () => ipcRenderer.removeListener('config-loaded', subscription);
    },

    onMenuStartTrading: (callback) => {
        const subscription = () => callback();
        ipcRenderer.on('menu-start-trading', subscription);
        return () => ipcRenderer.removeListener('menu-start-trading', subscription);
    },

    onMenuStopTrading: (callback) => {
        const subscription = () => callback();
        ipcRenderer.on('menu-stop-trading', subscription);
        return () => ipcRenderer.removeListener('menu-stop-trading', subscription);
    }
});

// Expose platform info
contextBridge.exposeInMainWorld('platform', {
    isWindows: process.platform === 'win32',
    isMac: process.platform === 'darwin',
    isLinux: process.platform === 'linux',
    arch: process.arch,
    versions: {
        electron: process.versions.electron,
        node: process.versions.node,
        chrome: process.versions.chrome
    }
});
