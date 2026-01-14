/**
 * CryptoAI Desktop - Main Process
 *
 * Professional Electron main process that:
 * - Creates the application window with safety-first UX
 * - Spawns and manages Python backend
 * - Handles IPC communication with confirmation dialogs
 * - Implements kill-switch accessible from UI
 *
 * SAFETY PRINCIPLES:
 * - Live mode requires explicit confirmation
 * - Kill-switch always accessible
 * - Clear error visibility
 * - Safe defaults (shadow mode)
 */

const { app, BrowserWindow, ipcMain, dialog, Menu, Tray, nativeImage } = require("electron");
const path = require("path");
const { spawn } = require("child_process");
const log = require("electron-log");
const Store = require("electron-store");

// Initialize electron store for settings
const store = new Store({
  defaults: {
    pythonPath: "python",
    configPath: "configs/default.yaml",
    mode: "shadow", // SAFE DEFAULT
    exchange: "binance",
    assets: ["BTCUSDT"],
    windowBounds: { width: 1400, height: 900 },
    // Safety settings
    confirmLiveMode: true,
    maxLeverage: 3,
    maxDrawdown: 0.10,
    enableKillSwitch: true,
  },
});

// Configure logging
log.transports.file.level = "info";
log.transports.console.level = "debug";

// Global references
let mainWindow = null;
let tray = null;
let pythonProcess = null;
let isQuitting = false;

// System state tracking
let systemState = {
  running: false,
  mode: "shadow",
  governanceState: "OPERATIONAL",
  lastError: null,
  startTime: null,
  tradesExecuted: 0,
  currentDrawdown: 0,
};

/**
 * Get the path to the Python backend
 */
function getPythonBackendPath() {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, "python-backend");
  }
  return path.join(__dirname, "..");
}

/**
 * Create the main application window
 */
function createWindow() {
  const bounds = store.get("windowBounds");

  mainWindow = new BrowserWindow({
    width: bounds.width,
    height: bounds.height,
    minWidth: 1000,
    minHeight: 700,
    title: "CryptoAI Desktop",
    icon: path.join(__dirname, "assets", "icons", "256x256.png"),
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, "preload.js"),
      sandbox: true,
    },
    backgroundColor: "#0f0f1a",
    show: false,
    titleBarStyle: process.platform === "darwin" ? "hiddenInset" : "default",
  });

  // Load the UI
  mainWindow.loadFile(path.join(__dirname, "renderer", "index.html"));

  // Show when ready
  mainWindow.once("ready-to-show", () => {
    mainWindow.show();
    log.info("Main window ready");
  });

  // Save window bounds on resize
  mainWindow.on("resize", () => {
    if (!mainWindow.isMaximized()) {
      const bounds = mainWindow.getBounds();
      store.set("windowBounds", bounds);
    }
  });

  // Handle close - minimize to tray on Windows
  mainWindow.on("close", (event) => {
    if (!isQuitting && process.platform === "win32") {
      event.preventDefault();
      mainWindow.hide();
      return;
    }

    if (!isQuitting && systemState.running) {
      event.preventDefault();
      dialog.showMessageBox(mainWindow, {
        type: "warning",
        title: "Trading Active",
        message: "Trading is currently active. Stop trading before closing?",
        buttons: ["Stop & Close", "Minimize", "Cancel"],
        defaultId: 2,
        cancelId: 2,
      }).then((result) => {
        if (result.response === 0) {
          stopPythonBackend();
          setTimeout(() => {
            isQuitting = true;
            app.quit();
          }, 2000);
        } else if (result.response === 1) {
          mainWindow.hide();
        }
      });
      return;
    }
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  // Create application menu
  createMenu();

  // Create system tray (Windows)
  if (process.platform === "win32") {
    createTray();
  }

  log.info("Window created");
}

/**
 * Create system tray icon
 */
function createTray() {
  const iconPath = path.join(__dirname, "assets", "icons", "32x32.png");
  tray = new Tray(iconPath);

  const contextMenu = Menu.buildFromTemplate([
    {
      label: "Show Window",
      click: () => mainWindow.show(),
    },
    { type: "separator" },
    {
      label: "Emergency Stop",
      click: () => {
        stopPythonBackend();
        if (mainWindow) {
          mainWindow.webContents.send("emergency-stop");
        }
      },
    },
    { type: "separator" },
    {
      label: "Quit",
      click: () => {
        isQuitting = true;
        stopPythonBackend();
        app.quit();
      },
    },
  ]);

  tray.setToolTip("CryptoAI Desktop");
  tray.setContextMenu(contextMenu);

  tray.on("double-click", () => {
    mainWindow.show();
  });
}

/**
 * Create application menu
 */
function createMenu() {
  const template = [
    {
      label: "File",
      submenu: [
        {
          label: "Settings",
          accelerator: "CmdOrCtrl+,",
          click: () => {
            mainWindow.webContents.send("open-settings");
          },
        },
        { type: "separator" },
        {
          label: "Quit",
          accelerator: process.platform === "darwin" ? "Cmd+Q" : "Alt+F4",
          click: () => {
            isQuitting = true;
            app.quit();
          },
        },
      ],
    },
    {
      label: "View",
      submenu: [
        { role: "reload" },
        { role: "forceReload" },
        { role: "toggleDevTools" },
        { type: "separator" },
        { role: "resetZoom" },
        { role: "zoomIn" },
        { role: "zoomOut" },
        { type: "separator" },
        { role: "togglefullscreen" },
      ],
    },
    {
      label: "Trading",
      submenu: [
        {
          label: "Start Trading",
          accelerator: "CmdOrCtrl+Enter",
          click: () => {
            mainWindow.webContents.send("menu-start-trading");
          },
        },
        {
          label: "Stop Trading",
          accelerator: "CmdOrCtrl+.",
          click: () => {
            mainWindow.webContents.send("menu-stop-trading");
          },
        },
        { type: "separator" },
        {
          label: "Emergency Kill Switch",
          accelerator: "CmdOrCtrl+Shift+K",
          click: async () => {
            log.warn("EMERGENCY KILL SWITCH ACTIVATED");
            stopPythonBackend();
            mainWindow.webContents.send("emergency-stop");
            dialog.showMessageBox(mainWindow, {
              type: "warning",
              title: "Emergency Stop",
              message: "Emergency kill switch activated. All trading stopped.",
              buttons: ["OK"],
            });
          },
        },
        { type: "separator" },
        {
          label: "Run Backtest",
          click: () => {
            mainWindow.webContents.send("run-backtest");
          },
        },
      ],
    },
    {
      label: "Help",
      submenu: [
        {
          label: "Documentation",
          click: () => {
            require("electron").shell.openExternal(
              "https://github.com/Kerim-Sabic/trade"
            );
          },
        },
        { type: "separator" },
        {
          label: "About",
          click: () => {
            dialog.showMessageBox(mainWindow, {
              type: "info",
              title: "About CryptoAI Desktop",
              message: "CryptoAI Desktop",
              detail: `Version: ${app.getVersion()}\nElectron: ${process.versions.electron}\nNode: ${process.versions.node}\n\nProfessional crypto trading with AI governance.`,
            });
          },
        },
      ],
    },
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

/**
 * Show live mode confirmation dialog
 */
async function confirmLiveMode() {
  const result = await dialog.showMessageBox(mainWindow, {
    type: "warning",
    title: "Live Trading Confirmation",
    message: "You are about to start LIVE trading with REAL money.",
    detail:
      "This will execute real trades on the exchange. Ensure you understand the risks:\n\n" +
      "- Real capital will be at risk\n" +
      "- Market conditions can change rapidly\n" +
      "- Past performance does not guarantee future results\n\n" +
      "Are you absolutely sure you want to proceed?",
    buttons: ["Cancel", "I Understand, Start Live Trading"],
    defaultId: 0,
    cancelId: 0,
    noLink: true,
  });

  return result.response === 1;
}

/**
 * Start the Python backend process
 */
async function startPythonBackend(mode, assets, configPath, exchange) {
  if (pythonProcess) {
    log.warn("Python process already running");
    return { success: false, error: "Process already running" };
  }

  // Safety check for live mode
  if (mode === "live" && store.get("confirmLiveMode")) {
    const confirmed = await confirmLiveMode();
    if (!confirmed) {
      log.info("Live mode cancelled by user");
      return { success: false, error: "Cancelled by user" };
    }
  }

  const pythonPath = store.get("pythonPath");
  const backendPath = getPythonBackendPath();

  const args = [
    "-m",
    "cryptoai.main",
    "--mode",
    mode,
    "--config",
    configPath || store.get("configPath"),
    "--assets",
    ...assets,
  ];

  log.info(`Starting Python backend: ${pythonPath} ${args.join(" ")}`);
  log.info(`Working directory: ${backendPath}`);

  try {
    pythonProcess = spawn(pythonPath, args, {
      cwd: backendPath,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: "1",
        CRYPTOAI_MODE: mode,
        CRYPTOAI_EXCHANGE: exchange || "binance",
      },
    });

    systemState.running = true;
    systemState.mode = mode;
    systemState.startTime = new Date();
    systemState.lastError = null;

    pythonProcess.stdout.on("data", (data) => {
      const message = data.toString().trim();
      log.info(`[Python] ${message}`);

      // Parse governance state from logs
      if (message.includes("GOVERNANCE_STATE:")) {
        const match = message.match(/GOVERNANCE_STATE:\s*(\w+)/);
        if (match) {
          systemState.governanceState = match[1];
        }
      }

      if (mainWindow) {
        mainWindow.webContents.send("python-log", {
          type: "stdout",
          message: message,
          timestamp: new Date().toISOString(),
        });
      }
    });

    pythonProcess.stderr.on("data", (data) => {
      const message = data.toString().trim();
      log.warn(`[Python Error] ${message}`);
      systemState.lastError = message;

      if (mainWindow) {
        mainWindow.webContents.send("python-log", {
          type: "stderr",
          message: message,
          timestamp: new Date().toISOString(),
        });
      }
    });

    pythonProcess.on("close", (code) => {
      log.info(`Python process exited with code ${code}`);
      pythonProcess = null;
      systemState.running = false;
      systemState.governanceState = "STOPPED";

      if (mainWindow) {
        mainWindow.webContents.send("python-stopped", { code });
      }
    });

    pythonProcess.on("error", (err) => {
      log.error(`Failed to start Python process: ${err.message}`);
      pythonProcess = null;
      systemState.running = false;
      systemState.lastError = err.message;

      if (mainWindow) {
        mainWindow.webContents.send("python-error", { error: err.message });
      }
    });

    if (mainWindow) {
      mainWindow.webContents.send("python-started", { mode, assets, exchange });
    }

    return { success: true };
  } catch (error) {
    log.error(`Failed to start: ${error.message}`);
    return { success: false, error: error.message };
  }
}

/**
 * Stop the Python backend process
 */
function stopPythonBackend() {
  if (!pythonProcess) {
    log.warn("No Python process to stop");
    return;
  }

  log.info("Stopping Python backend");

  // Send SIGINT first for graceful shutdown
  if (process.platform === "win32") {
    // Windows doesn't support SIGINT well
    pythonProcess.kill();
  } else {
    pythonProcess.kill("SIGINT");
  }

  // Force kill after timeout
  setTimeout(() => {
    if (pythonProcess) {
      log.warn("Force killing Python process");
      pythonProcess.kill("SIGKILL");
    }
  }, 5000);
}

// IPC Handlers
ipcMain.handle("start-trading", async (event, { mode, assets, config, exchange }) => {
  try {
    return await startPythonBackend(
      mode || "shadow",
      assets || ["BTCUSDT"],
      config,
      exchange || "binance"
    );
  } catch (error) {
    log.error(`Failed to start trading: ${error.message}`);
    return { success: false, error: error.message };
  }
});

ipcMain.handle("stop-trading", async () => {
  try {
    stopPythonBackend();
    return { success: true };
  } catch (error) {
    log.error(`Failed to stop trading: ${error.message}`);
    return { success: false, error: error.message };
  }
});

ipcMain.handle("emergency-stop", async () => {
  log.warn("EMERGENCY STOP triggered via IPC");
  stopPythonBackend();
  systemState.governanceState = "HALTED";
  return { success: true };
});

ipcMain.handle("get-status", async () => {
  return {
    ...systemState,
    settings: {
      mode: store.get("mode"),
      exchange: store.get("exchange"),
      assets: store.get("assets"),
      configPath: store.get("configPath"),
      pythonPath: store.get("pythonPath"),
      maxLeverage: store.get("maxLeverage"),
      maxDrawdown: store.get("maxDrawdown"),
    },
  };
});

ipcMain.handle("get-settings", async () => {
  return store.store;
});

ipcMain.handle("save-settings", async (event, settings) => {
  try {
    Object.keys(settings).forEach((key) => {
      store.set(key, settings[key]);
    });
    return { success: true };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle("select-config", async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "Select Configuration File",
    filters: [{ name: "YAML Files", extensions: ["yaml", "yml"] }],
    properties: ["openFile"],
  });

  if (!result.canceled && result.filePaths.length > 0) {
    const configPath = result.filePaths[0];
    store.set("configPath", configPath);
    return { success: true, path: configPath };
  }

  return { success: false };
});

ipcMain.handle("select-python", async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "Select Python Executable",
    filters:
      process.platform === "win32"
        ? [{ name: "Executables", extensions: ["exe"] }]
        : [],
    properties: ["openFile"],
  });

  if (!result.canceled && result.filePaths.length > 0) {
    const pythonPath = result.filePaths[0];
    store.set("pythonPath", pythonPath);
    return { success: true, path: pythonPath };
  }

  return { success: false };
});

ipcMain.handle("confirm-dialog", async (event, { title, message, detail, type }) => {
  const result = await dialog.showMessageBox(mainWindow, {
    type: type || "question",
    title: title,
    message: message,
    detail: detail,
    buttons: ["Cancel", "Confirm"],
    defaultId: 0,
    cancelId: 0,
  });
  return { confirmed: result.response === 1 };
});

// App lifecycle
app.whenReady().then(() => {
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    } else if (mainWindow) {
      mainWindow.show();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    stopPythonBackend();
    app.quit();
  }
});

app.on("before-quit", () => {
  isQuitting = true;
  stopPythonBackend();
});

// Handle uncaught exceptions
process.on("uncaughtException", (error) => {
  log.error(`Uncaught exception: ${error.message}`);
  dialog.showErrorBox("Error", `An unexpected error occurred: ${error.message}`);
});

// Single instance lock (Windows)
const gotTheLock = app.requestSingleInstanceLock();
if (!gotTheLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.show();
      mainWindow.focus();
    }
  });
}

log.info("CryptoAI Desktop starting...");
