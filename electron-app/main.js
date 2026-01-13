/**
 * CryptoAI Desktop - Main Process
 *
 * Electron main process that:
 * - Creates the application window
 * - Spawns and manages Python backend
 * - Handles IPC communication
 */

const { app, BrowserWindow, ipcMain, dialog, Menu } = require("electron");
const path = require("path");
const { spawn } = require("child_process");
const log = require("electron-log");
const Store = require("electron-store");

// Initialize electron store for settings
const store = new Store({
  defaults: {
    pythonPath: "python",
    configPath: "configs/default.yaml",
    mode: "shadow",
    assets: ["BTCUSDT"],
    windowBounds: { width: 1200, height: 800 },
  },
});

// Configure logging
log.transports.file.level = "info";
log.transports.console.level = "debug";

// Global references
let mainWindow = null;
let pythonProcess = null;
let isQuitting = false;

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
    minWidth: 800,
    minHeight: 600,
    title: "CryptoAI Desktop",
    icon: path.join(__dirname, "assets", "icon.png"),
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, "preload.js"),
    },
    backgroundColor: "#1a1a2e",
    show: false,
  });

  // Load the UI
  mainWindow.loadFile(path.join(__dirname, "renderer", "index.html"));

  // Show when ready
  mainWindow.once("ready-to-show", () => {
    mainWindow.show();
    log.info("Main window ready");
  });

  // Save window bounds on close
  mainWindow.on("close", (event) => {
    if (!isQuitting) {
      event.preventDefault();
      mainWindow.hide();
      return;
    }

    const bounds = mainWindow.getBounds();
    store.set("windowBounds", bounds);
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  // Create application menu
  createMenu();

  log.info("Window created");
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
          accelerator: process.platform === "darwin" ? "Cmd+Q" : "Ctrl+Q",
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
          accelerator: "CmdOrCtrl+R",
          click: () => {
            mainWindow.webContents.send("start-trading");
          },
        },
        {
          label: "Stop Trading",
          accelerator: "CmdOrCtrl+S",
          click: () => {
            mainWindow.webContents.send("stop-trading");
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
              "https://github.com/your-org/cryptoai"
            );
          },
        },
        {
          label: "About",
          click: () => {
            dialog.showMessageBox(mainWindow, {
              type: "info",
              title: "About CryptoAI Desktop",
              message: "CryptoAI Desktop",
              detail: `Version: ${app.getVersion()}\nElectron: ${
                process.versions.electron
              }\nNode: ${process.versions.node}`,
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
 * Start the Python backend process
 */
function startPythonBackend(mode, assets, configPath) {
  if (pythonProcess) {
    log.warn("Python process already running");
    return;
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

  pythonProcess = spawn(pythonPath, args, {
    cwd: backendPath,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: "1",
    },
  });

  pythonProcess.stdout.on("data", (data) => {
    const message = data.toString().trim();
    log.info(`[Python] ${message}`);
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
    if (mainWindow) {
      mainWindow.webContents.send("python-stopped", { code });
    }
  });

  pythonProcess.on("error", (err) => {
    log.error(`Failed to start Python process: ${err.message}`);
    pythonProcess = null;
    if (mainWindow) {
      mainWindow.webContents.send("python-error", { error: err.message });
    }
  });

  if (mainWindow) {
    mainWindow.webContents.send("python-started", { mode, assets });
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
  pythonProcess.kill("SIGINT");

  // Force kill after timeout
  setTimeout(() => {
    if (pythonProcess) {
      log.warn("Force killing Python process");
      pythonProcess.kill("SIGKILL");
    }
  }, 5000);
}

// IPC Handlers
ipcMain.handle("start-trading", async (event, { mode, assets, config }) => {
  try {
    startPythonBackend(mode || "shadow", assets || ["BTCUSDT"], config);
    return { success: true };
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

ipcMain.handle("get-status", async () => {
  return {
    running: pythonProcess !== null,
    settings: {
      mode: store.get("mode"),
      assets: store.get("assets"),
      configPath: store.get("configPath"),
      pythonPath: store.get("pythonPath"),
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
    properties: ["openFile"],
  });

  if (!result.canceled && result.filePaths.length > 0) {
    const pythonPath = result.filePaths[0];
    store.set("pythonPath", pythonPath);
    return { success: true, path: pythonPath };
  }

  return { success: false };
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

log.info("CryptoAI Desktop starting...");
