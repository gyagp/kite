import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Activity,
  ArrowUpRight,
  Box,
  ChevronRight,
  CircleHelp,
  Cpu,
  Feather,
  Layers3,
  MessageSquare,
  Plus,
  Settings2,
  ShieldCheck,
  Square,
  X,
  WandSparkles,
} from 'lucide-react';
import { api, errorText, streamJob } from './api';
import { Badge, Spinner } from './components';
import type { Bootstrap, Execute, Metrics, Page, Status, Thread } from './types';
import { Embeddings } from './Embeddings';
import { Chat } from './Chat';
import { Models, Settings, ActivityPage } from './Workspace';
import { ImageStudio } from './ImageStudio';

const navigation = [
  { id: 'chat', name: 'Chat', icon: MessageSquare },
  { id: 'embeddings', name: 'Embeddings', icon: Layers3 },
  { id: 'images', name: 'Image Studio', icon: WandSparkles },
  { id: 'models', name: 'My models', icon: Box },
  { id: 'activity', name: 'Activity', icon: Activity },
] as const;
export default function App() {
  const [boot, setBoot] = useState<Bootstrap | null>(null);
  const [status, setStatus] = useState<Status>({
    loaded: null,
    active: null,
    worker_running: false,
  });
  const [page, setPage] = useState<Page>(
    window.location.hash === '#images' ? 'images' : 'embeddings',
  );
  const [error, setError] = useState('');
  const [connecting, setConnecting] = useState(true);
  const [working, setWorking] = useState(false);
  const [imageJobActive, setImageJobActive] = useState(false);
  const [progress, setProgress] = useState('');
  const [thread, setThread] = useState<Thread | null>(null);
  const [chatKey, setChatKey] = useState(0);
  const [preferredChat, setPreferredChat] = useState('');
  const [preferredEmbedding, setPreferredEmbedding] = useState('');
  const executing = useRef(false);
  const refresh = useCallback(async () => {
    const data = await api<Bootstrap>('/api/bootstrap');
    setBoot(data);
    setStatus(data.status);
    setConnecting(false);
  }, []);
  useEffect(() => {
    refresh().catch((e) => {
      setError(errorText(e));
      setConnecting(false);
    });
  }, [refresh]);
  useEffect(() => {
    const timer = setInterval(() => {
      api<Status>('/api/status')
        .then((s) => {
          setStatus(s);
        })
        .catch(() => {});
    }, 2500);
    return () => clearInterval(timer);
  }, []);
  const execute: Execute = async (data, onEvent) => {
    if (executing.current) throw new Error('An operation is already running.');
    executing.current = true;
    setWorking(true);
    setImageJobActive(data.kind === 'image');
    setError('');
    setProgress('Preparing the local runtime…');
    let result: Metrics | undefined;
    let failure = '';
    try {
      const job = await api<{ id: string }>('/api/jobs', 'POST', data);
      await streamJob(job.id, (event) => {
        if (event.type === 'status') setProgress(event.message || '');
        if (event.type === 'loaded') setProgress('Running on native WebGPU…');
        if (event.type === 'progress')
          setProgress(`Embedded ${event.completed} of ${event.total} inputs`);
        if (event.type === 'delta') setProgress('Generating locally…');
        if (event.type === 'image_progress')
          setProgress(`Denoising step ${event.completed} of ${event.total}`);
        if (event.type === 'result') result = event.data;
        if (event.type === 'error') failure = event.message || 'Inference failed.';
        onEvent?.(event);
      });
      if (failure) throw new Error(failure);
      return result;
    } catch (e) {
      setError(errorText(e));
      throw e;
    } finally {
      executing.current = false;
      setWorking(false);
      setImageJobActive(false);
      setProgress('');
      await refresh().catch(() => {});
    }
  };
  const busy = working || !!status.active;
  const stop = async () => {
    try {
      await api('/api/stop', 'POST', {});
      await refresh();
    } catch (e) {
      setError(errorText(e));
    }
  };
  const newChat = () => {
    if (busy) return;
    setThread(null);
    setChatKey((k) => k + 1);
    setPage('chat');
  };
  const openThread = (t: Thread) => {
    if (busy) return;
    setThread(t);
    setChatKey((k) => k + 1);
    setPage('chat');
  };
  const loadedModel = boot?.models.find((m) => m.id === status.loaded?.model);
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <button className="brand" onClick={() => setPage('embeddings')} aria-label="Kite home">
          <img src="/kite.svg" alt="" />
          <span>
            kite<span className="brand-period">.</span>
          </span>
          <Badge>LOCAL</Badge>
        </button>
        <div className="workspace-name">
          <span className="workspace-avatar">Y</span>
          <div>
            Personal workspace<small>On this computer</small>
          </div>
          <ShieldCheck size={15} />
        </div>
        <div className="nav-label">WORKSPACE</div>
        <nav>
          {navigation.map((item) => (
            <button
              key={item.id}
              className={`nav-item ${page === item.id ? 'selected' : ''}`}
              onClick={() => setPage(item.id)}
            >
              <item.icon size={18} />
              {item.name}
              {item.id === 'models' && (
                <span className="nav-count">{boot?.models.length || 0}</span>
              )}
              {page === item.id && <span className="nav-dot" />}
            </button>
          ))}
        </nav>
        {page === 'chat' && (
          <div className="recent-chats">
            <div className="nav-label">
              RECENT CHATS
              <button
                className="icon-button"
                aria-label="New chat"
                disabled={busy}
                onClick={newChat}
              >
                <Plus size={14} />
              </button>
            </div>
            {boot?.threads.slice(0, 8).map((t) => (
              <button
                key={t.id}
                className={`thread-link ${t.id === thread?.id ? 'active' : ''}`}
                disabled={busy}
                onClick={() => openThread(t)}
              >
                <MessageSquare size={13} />
                <span>{t.title}</span>
              </button>
            ))}
            {!boot?.threads.length && <p className="muted small">Your conversations stay here.</p>}
          </div>
        )}
        <div className="sidebar-bottom">
          <div className="local-card">
            <span className="tiny-orbit">
              <Feather size={20} />
            </span>
            <strong>Your models. Your machine.</strong>
            <p>
              A little less cloud.
              <br />A lot more possibility.
            </p>
            <span>
              <span className="status-dot" /> Private by default
            </span>
          </div>
          <button
            className={`nav-item ${page === 'settings' ? 'selected' : ''}`}
            onClick={() => setPage('settings')}
          >
            <Settings2 size={18} />
            Settings
          </button>
          <div className="sidebar-footer">
            <span>Kite v0.2.0</span>
            <button
              title="About the local runtime"
              aria-label="About the local runtime"
              className="icon-button"
              onClick={() => setPage('settings')}
            >
              <CircleHelp size={15} />
            </button>
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <ChevronRight size={13} />
            <span>
              {page === 'settings' ? 'Settings' : navigation.find((n) => n.id === page)?.name}
            </span>
          </div>
          <div className="topbar-right">
            <span className="local-label">
              <ShieldCheck size={13} /> All inference stays local
            </span>
            <span className={`engine-pill ${status.loaded ? 'online' : ''}`}>
              <span className="status-dot" />
              {status.loaded?.kind === 'image' && !status.active
                ? 'Image engine warm'
                : status.active?.kind === 'image'
                  ? 'WebGPU + CUDA'
                  : status.loaded
                    ? 'WebGPU active'
                    : 'WebGPU ready'}
            </span>
          </div>
        </header>
        {error && (
          <div className="error-banner" role="alert">
            <div>
              <strong>Something needs attention</strong>
              <p>{error}</p>
            </div>
            <button className="icon-button" onClick={() => setError('')} aria-label="Dismiss error">
              <X size={17} />
            </button>
          </div>
        )}
        {connecting ? (
          <main className="connection-state">
            <Spinner />
            <h2>Opening your workspace…</h2>
          </main>
        ) : !boot ? (
          <main className="connection-state">
            <Cpu size={36} />
            <h2>Connect to your local service</h2>
            <p>
              Start Kite with <code>start.bat</code>, or run <code>python -m backend.server</code>.
            </p>
            <button
              className="button primary"
              onClick={() => refresh().catch((e) => setError(errorText(e)))}
            >
              Reconnect
            </button>
          </main>
        ) : (
          <>
            <main className={`page-content page-${page}`}>
              {(page === 'images' || imageJobActive) && (
                <div hidden={page !== 'images'}>
                  <ImageStudio execute={execute} busy={busy} notify={setError} onStop={stop} />
                </div>
              )}
              {page === 'embeddings' && (
                <Embeddings
                  preferred={preferredEmbedding}
                  models={boot.models.filter((m) => m.kind === 'embedding')}
                  execute={execute}
                  busy={busy}
                  status={status}
                  onModels={() => setPage('models')}
                  notify={setError}
                />
              )}
              {page === 'chat' && (
                <Chat
                  preferences={boot.chat_preferences}
                  key={chatKey}
                  preferred={preferredChat}
                  models={boot.models.filter((m) => m.kind === 'chat')}
                  execute={execute}
                  busy={busy}
                  status={status}
                  initial={thread}
                  onNew={newChat}
                  onModels={() => setPage('models')}
                  notify={setError}
                  refresh={refresh}
                />
              )}
              {page === 'models' && (
                <Models
                  models={boot.models}
                  settings={boot.settings}
                  status={status}
                  busy={busy}
                  execute={execute}
                  refresh={refresh}
                  notify={setError}
                  onUse={(m) => {
                    if (m.kind === 'image') {
                      api('/api/images/config', 'POST', { package: m.path })
                        .then(() => setPage('images'))
                        .catch((e) => setError(errorText(e)));
                    } else if (m.kind === 'chat') {
                      setPreferredChat(m.id);
                      newChat();
                    } else {
                      setPreferredEmbedding(m.id);
                      setPage('embeddings');
                    }
                  }}
                />
              )}
              {page === 'settings' && (
                <Settings
                  boot={boot}
                  status={status}
                  busy={busy}
                  refresh={refresh}
                  notify={setError}
                />
              )}
              {page === 'activity' && (
                <ActivityPage models={boot.models} notify={setError} busy={busy} />
              )}
            </main>
            <footer className="statusbar">
              <div>
                <Cpu size={13} />
                <span>
                  {status.loaded?.kind === 'image'
                    ? 'Qwen-Image · 2.1 Turbo'
                    : loadedModel
                      ? loadedModel.name
                      : status.active?.kind === 'image'
                        ? 'Qwen-Image · 2.1 Turbo'
                        : status.active?.kind === 'tool'
                          ? 'Background tool operation'
                          : 'No model loaded'}
                </span>
                <span className="status-separator" />{' '}
                <span>
                  {status.loaded?.kind === 'image' && !status.active
                    ? 'CPU components cached'
                    : status.active?.kind === 'image'
                      ? 'Native WebGPU · D3D12 / CUDA'
                      : status.loaded
                        ? 'Native WebGPU · D3D12'
                        : 'Loads automatically when you run'}
                </span>
              </div>
              <div aria-live="polite">
                {busy ? (
                  <>
                    <Spinner />
                    <span>{progress || 'An operation is running'}</span>
                    <button onClick={stop}>
                      <Square size={11} />
                      Stop
                    </button>
                  </>
                ) : status.loaded ? (
                  <button onClick={stop}>
                    Unload model <ArrowUpRight size={12} />
                  </button>
                ) : (
                  <span>Everything runs on your computer</span>
                )}
              </div>
            </footer>
          </>
        )}
      </div>
    </div>
  );
}
