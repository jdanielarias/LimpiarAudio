"use strict";

const { app, BrowserWindow } = require("electron");
const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");

const frontend = path.resolve(__dirname, "..", "..", "frontend");
const contentTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
};
const csp = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'";

function serveFile(response, filePath) {
  response.writeHead(200, {
    "Content-Type": contentTypes[path.extname(filePath)] || "application/octet-stream",
    "Content-Security-Policy": csp,
    "X-Content-Type-Options": "nosniff",
  });
  fs.createReadStream(filePath).pipe(response);
}

function createServer() {
  return http.createServer((request, response) => {
    const url = new URL(request.url, "http://127.0.0.1");
    if (url.pathname === "/") return serveFile(response, path.join(frontend, "index.html"));
    if (url.pathname === "/ready") {
      response.writeHead(200, { "Content-Type": "application/json", "Content-Security-Policy": csp });
      response.end(JSON.stringify({
        ok: true, schema: 1, service: "LimpiarAudio", pipeline_version: "smoke-test",
      }));
      return;
    }
    if (url.pathname === "/info") {
      response.writeHead(200, { "Content-Type": "application/json", "Content-Security-Policy": csp });
      response.end(JSON.stringify({
        device: "cpu",
        cuda: false,
        audiosep_available: false,
        whisper_available: false,
        restore_available: false,
        enhance_available: false,
        engines: { whisper: { available_models: [], default_model: "small" } },
        pipeline_version: "smoke-test",
      }));
      return;
    }
    if (url.pathname.startsWith("/static/")) {
      const relative = decodeURIComponent(url.pathname.slice("/static/".length));
      const candidate = path.resolve(frontend, relative);
      if (candidate.startsWith(`${frontend}${path.sep}`) && fs.existsSync(candidate)) {
        return serveFile(response, candidate);
      }
    }
    response.writeHead(url.pathname === "/favicon.ico" ? 204 : 404);
    response.end();
  });
}

app.commandLine.appendSwitch("disable-gpu");
app.commandLine.appendSwitch("disable-software-rasterizer");

app.whenReady().then(async () => {
  const server = createServer();
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const port = server.address().port;
  const errors = [];
  const window = new BrowserWindow({
    show: false,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      webSecurity: true,
    },
  });
  window.webContents.on("console-message", (event) => {
    const message = typeof event.message === "string" ? event.message : "";
    if (/error|refused|uncaught/i.test(message)) errors.push(message);
  });
  window.webContents.on("did-fail-load", (_event, code, description, url) => {
    errors.push(`${code} ${description} ${url}`);
  });
  const timeout = setTimeout(() => {
    process.stderr.write("La prueba visual excedió 15 segundos.\n");
    app.exit(1);
  }, 15_000);
  try {
    await window.loadURL(`http://127.0.0.1:${port}/`);
    await new Promise((resolve) => setTimeout(resolve, 500));
    const result = await window.webContents.executeJavaScript(`(() => {
      return {
        waveSurfer: typeof window.WaveSurfer === 'function',
        utilities: Boolean(window.LimpiarAudioUtils),
        consoleVisible: Boolean(document.querySelector(".sidebar")) && Boolean(document.querySelector(".workspace")),
        dropzoneReady: document.getElementById("dropzone") instanceof HTMLElement,
        appScript: Array.from(document.scripts).some(s => s.src.endsWith('/static/app.js')),
        externalScripts: Array.from(document.scripts).filter(s => new URL(s.src).origin !== location.origin).length,
        inlineScripts: Array.from(document.scripts).filter(s => !s.src).length,
      };
    })()`);
    const valid = result.waveSurfer && result.utilities && result.consoleVisible && result.dropzoneReady
      && result.appScript && result.externalScripts === 0 && result.inlineScripts === 0;
    if (!valid || errors.length) {
      throw new Error(`DOM inválido: ${JSON.stringify(result)}; errores: ${errors.join(" | ")}`);
    }
    process.stdout.write(`UI/CSP verificada: ${JSON.stringify(result)}\n`);
    clearTimeout(timeout);
    window.destroy();
    await new Promise((resolve) => server.close(resolve));
    app.exit(0);
  } catch (error) {
    clearTimeout(timeout);
    process.stderr.write(`${error.stack || error.message}\n`);
    window.destroy();
    server.close(() => app.exit(1));
  }
});
