const { app, BrowserWindow, dialog, ipcMain, shell } = require('electron');
const { spawn } = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');
const net = require('node:net');
let backend;
let serviceUrl;
let exiting = false;
const hasInstanceLock = app.requestSingleInstanceLock();
if (!hasInstanceLock) app.quit();
app.on('second-instance', () => {
  const win = BrowserWindow.getAllWindows()[0];
  if (win) {
    if (win.isMinimized()) win.restore();
    win.focus();
  }
});

async function freePort() {
  return new Promise((resolve, reject) => {
    const probe = net.createServer();
    probe.on('error', reject);
    probe.listen(0, '127.0.0.1', () => {
      const port = probe.address().port;
      probe.close(() => resolve(port));
    });
  });
}

async function launch() {
  const root = app.isPackaged ? process.resourcesPath : path.join(__dirname, '..');
  let discoveredAssets = path.join(root, 'local');
  for (let folder = root; path.dirname(folder) !== folder; folder = path.dirname(folder)) {
    const candidate = path.join(folder, 'local');
    if (fs.existsSync(path.join(candidate, 'kite-assets.json'))) {
      discoveredAssets = candidate;
      break;
    }
  }
  const assets = process.env.KITE_ASSET_ROOT || discoveredAssets;
  const workspace = path.dirname(assets);
  const imageDirectory =
    process.env.KITE_IMAGE_DIR ||
    (process.env.KITE_DATA_DIR
      ? path.join(process.env.KITE_DATA_DIR, 'images')
      : path.join(workspace, 'gitignore', 'images'));
  const candidate = path.join(assets, '.venv', 'Scripts', 'python.exe');
  const python = process.env.KITE_PYTHON || (fs.existsSync(candidate) ? candidate : 'python');
  const port = await freePort();
  serviceUrl = `http://127.0.0.1:${port}`;
  let errors = '';
  backend = spawn(
    python,
    [
      '-u',
      '-m',
      'backend.server',
      '--port',
      String(port),
      '--data-dir',
      process.env.KITE_DATA_DIR ||
        (app.isPackaged ? app.getPath('userData') : path.join(root, '.kite')),
      '--parent-pid',
      String(process.pid),
    ],
    {
      cwd: root,
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: {
        ...process.env,
        KITE_ASSET_ROOT: assets,
        KITE_IMAGE_DIR: imageDirectory,
        PYTHONIOENCODING: 'utf-8',
      },
    },
  );
  backend.on('error', (error) => {
    errors = error.message;
  });
  backend.stderr.on('data', (data) => {
    errors = (errors + data.toString()).slice(-4000);
  });
  let ready = false;
  for (let n = 0; n < 100; n++) {
    try {
      const r = await fetch(`${serviceUrl}/api/health`);
      ready = r.ok && (await r.json()).app === 'kite';
      if (ready) break;
    } catch {}
    if (backend.exitCode !== null || backend.pid === undefined) break;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  if (!ready)
    throw new Error(
      `The local service could not start. Install Python 3.11+ or set KITE_PYTHON.\n\n${errors}`,
    );
  const window = new BrowserWindow({
    width: 1460,
    height: 980,
    minWidth: 960,
    minHeight: 680,
    backgroundColor: '#f4f8ff',
    title: 'Kite',
    icon: path.join(root, 'dist', 'kite.png'),
    autoHideMenuBar: true,
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  window.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('https://')) shell.openExternal(url);
    return { action: 'deny' };
  });
  window.webContents.on('will-navigate', (event, url) => {
    if (new URL(url).origin !== serviceUrl) event.preventDefault();
  });
  await window.loadURL(serviceUrl + (process.argv.includes('--image-studio') ? '/#images' : ''));
  if (process.env.KITE_TEST_HEADLESS !== '1') window.show();
  backend.on('exit', () => {
    if (!exiting)
      dialog.showErrorBox(
        'Kite service stopped',
        errors || 'Restart Kite to reconnect to the local service.',
      );
  });
}

ipcMain.handle('select-path', async (event, kind) => {
  if (!serviceUrl || new URL(event.senderFrame.url).origin !== serviceUrl) return null;
  const options =
    kind === 'folder'
      ? { properties: ['openDirectory'] }
      : {
          properties: ['openFile'],
          filters:
            kind === 'runtime'
              ? [{ name: 'WebGPU provider', extensions: ['dll'] }]
              : [{ name: 'Python executable', extensions: ['exe'] }],
        };
  const result = await dialog.showOpenDialog(BrowserWindow.fromWebContents(event.sender), options);
  return result.canceled ? null : result.filePaths[0];
});
app
  .whenReady()
  .then(() => hasInstanceLock && launch())
  .catch((error) => {
    dialog.showErrorBox('Unable to start Kite', error.message);
    app.quit();
  });
app.on('window-all-closed', () => app.quit());
app.on('before-quit', (event) => {
  if (!exiting && backend && serviceUrl) {
    event.preventDefault();
    exiting = true;
    backend.once('exit', () => app.quit());
    fetch(`${serviceUrl}/api/shutdown`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{}',
    }).catch(() => {
      backend.kill();
      app.quit();
    });
    setTimeout(() => {
      backend.kill();
      app.exit();
    }, 7000).unref();
  }
});
