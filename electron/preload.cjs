const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('kiteDesktop', {
  selectPath: (kind) => ipcRenderer.invoke('select-path', kind),
});
