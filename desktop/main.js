// LimpiarAudio — proceso principal de Electron.
// Arranca el servidor local (uvicorn) usando el entorno Python del proyecto
// (.venv311), espera a que responda y carga la interfaz en la ventana. Al cerrar,
// detiene el servidor. NO reempaqueta el backend de ML (varios GB + modelos): la
// app es un lanzador de la instalación local creada por setup.ps1.

const { app, BrowserWindow, dialog, shell } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");
const http = require("http");
const net = require("net");

let backend = null;
let win = null;
let PORT = 8000;

// ---- Resolución de rutas del proyecto (config.json es OPCIONAL) ----
// Prioridad: variables de entorno > config.json (si existe) > autodetección por
// la ubicación del ejecutable/código. Así no hace falta config.json si el .exe
// está dentro de la carpeta del proyecto (build in-repo o portable).
function loadConfig() {
  const cfgPath = app.isPackaged
    ? path.join(process.resourcesPath, "config.json")
    : path.join(__dirname, "config.json");
  try {
    return JSON.parse(fs.readFileSync(cfgPath, "utf-8"));
  } catch (e) {
    return {}; // no existe o no es válido -> se autodetecta
  }
}

function hasBackend(dir) {
  return !!dir && fs.existsSync(path.join(dir, "backend", "main.py"));
}

function resolveHome(cfg) {
  const candidates = [
    process.env.LIMPIARAUDIO_HOME,
    cfg.home,
    // Empaquetado: .../desktop/dist/win-unpacked/resources -> raíz (subir 4).
    app.isPackaged ? path.resolve(process.resourcesPath, "..", "..", "..", "..") : null,
    // Dev: desktop/ -> raíz del repo.
    path.resolve(__dirname, ".."),
  ].filter(Boolean);
  for (const c of candidates) if (hasBackend(c)) return c;
  return candidates[candidates.length - 1];
}

function resolvePython(home, cfg) {
  const cands = [
    process.env.LIMPIARAUDIO_PYTHON,
    cfg.python,
    path.join(home, ".venv311", "Scripts", "python.exe"),
    path.join(home, ".venv311", "bin", "python"),
  ].filter(Boolean);
  for (const c of cands) if (fs.existsSync(c)) return c;
  return null;
}

// Puerto libre para evitar chocar con un servidor ya arrancado a mano.
function findFreePort(preferred) {
  return new Promise((resolve) => {
    const srv = net.createServer();
    srv.once("error", () => resolve(0)); // 0 => que el SO elija
    srv.listen(preferred, "127.0.0.1", () => {
      srv.close(() => resolve(preferred));
    });
  });
}

function waitForServer(base, cb, tries = 120) {
  const ping = () => {
    const req = http.get(base + "/info", (res) => {
      res.destroy();
      cb(true);
    });
    req.on("error", () => {
      if (--tries <= 0) cb(false);
      else setTimeout(ping, 500);
    });
  };
  ping();
}

async function boot() {
  const cfg = loadConfig();
  const home = resolveHome(cfg);
  const py = resolvePython(home, cfg);

  win = new BrowserWindow({
    width: 1100,
    height: 920,
    minWidth: 720,
    minHeight: 600,
    title: "LimpiarAudio",
    backgroundColor: "#0d1117",
    autoHideMenuBar: true,
    show: true,
  });
  win.setMenuBarVisibility(false);
  win.loadFile(path.join(__dirname, "loading.html"));

  if (!py) {
    dialog.showErrorBox(
      "LimpiarAudio",
      "No se encontró el entorno de Python (.venv311).\n\n" +
        "Instala el proyecto con setup.ps1 y, si es necesario, define la variable " +
        "LIMPIARAUDIO_PYTHON con la ruta a python.exe del entorno."
    );
    app.quit();
    return;
  }

  PORT = (await findFreePort(cfg.port || 8000)) || (await findFreePort(0));
  const base = `http://127.0.0.1:${PORT}`;

  backend = spawn(
    py,
    ["-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", String(PORT)],
    { cwd: home, windowsHide: true }
  );
  const log = (d) => process.stdout.write(`[backend] ${d}`);
  backend.stdout.on("data", log);
  backend.stderr.on("data", log);
  backend.on("exit", (code) => {
    if (code && win && !win.isDestroyed()) {
      dialog.showErrorBox("LimpiarAudio", `El servidor terminó inesperadamente (código ${code}).`);
    }
  });

  waitForServer(base, (ok) => {
    if (!win || win.isDestroyed()) return;
    if (ok) {
      win.loadURL(base);
      // Abrir enlaces externos en el navegador del sistema, no en la app.
      win.webContents.setWindowOpenHandler(({ url }) => {
        if (!url.startsWith(base)) {
          shell.openExternal(url);
          return { action: "deny" };
        }
        return { action: "allow" };
      });
    } else {
      dialog.showErrorBox(
        "LimpiarAudio",
        "El servidor local no respondió a tiempo. Revisa que el entorno esté instalado (setup.ps1)."
      );
      app.quit();
    }
  });
}

function stopBackend() {
  if (backend && !backend.killed) {
    try {
      backend.kill();
    } catch (e) {}
    backend = null;
  }
}

app.whenReady().then(boot);
app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) boot();
});
app.on("window-all-closed", () => {
  stopBackend();
  app.quit();
});
app.on("before-quit", stopBackend);
app.on("will-quit", stopBackend);
