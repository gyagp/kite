import { useEffect, useRef, useState } from 'react';
import {
  ArrowDownToLine,
  ArrowLeft,
  ArrowRight,
  Check,
  ExternalLink,
  Globe,
  RefreshCw,
  Search,
  Square,
} from 'lucide-react';
import { api, bytes, errorText } from './api';
import { Badge, Modal, PathInput, Spinner } from './components';
import type { Model } from './types';

type Catalog = {
  repository: string;
  revision: string;
  models: { name: string }[];
  directory: string;
};
type Package = {
  id: string;
  name: string;
  path: string;
  variant: string;
  revision: string;
  size: number;
  files: { name: string; size: number }[];
  modalities: string[];
  kind: string;
  architecture: string;
  source_url: string;
};
type Download = {
  id: string;
  name: string;
  variant: string;
  status: string;
  total_bytes: number;
  downloaded_bytes: number;
  file: string;
  speed: number;
  error?: string;
  model?: string;
  directory: string;
  revision: string;
};

export function ModelHub({
  models,
  refresh,
  requested,
  onRequestHandled,
}: {
  models: Model[];
  refresh: () => Promise<void>;
  requested?: Model | null;
  onRequestHandled?: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [search, setSearch] = useState('');
  const [selected, setSelected] = useState('');
  const [packages, setPackages] = useState<Package[]>([]);
  const [directory, setDirectory] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [downloads, setDownloads] = useState<Download[]>([]);
  const [restoreModel, setRestoreModel] = useState<Model | null>(null);
  const completed = useRef(new Set<string>());
  const currentModels = useRef(models);
  currentModels.current = models;
  const active = downloads.find((d) => ['downloading', 'verifying'].includes(d.status));
  useEffect(() => {
    let alive = true;
    const poll = () =>
      api<Download[]>('/api/hub/downloads')
        .then((items) => {
          if (!alive) return;
          setDownloads(items);
          for (const item of items)
            if (
              item.status === 'completed' &&
              item.model &&
              !currentModels.current.some((m) => m.id === item.model && m.available) &&
              !completed.current.has(item.id)
            ) {
              completed.current.add(item.id);
              void refresh();
            }
        })
        .catch(() => {});
    void poll();
    const timer = setInterval(poll, 1200);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [refresh]);
  const load = async (reload = false, restore?: Model) => {
    setOpen(true);
    setLoading(true);
    setError('');
    setSelected('');
    setRestoreModel(restore || null);
    try {
      const c = await api<Catalog>(
        reload ? '/api/hub/refresh' : '/api/hub/catalog',
        reload ? 'POST' : 'GET',
        reload ? {} : undefined,
      );
      setCatalog(c);
      if (!directory) setDirectory(c.directory);
      if (restore?.download) {
        setSelected(restore.download.model);
        setPackages(
          await api<Package[]>('/api/hub/packages', 'POST', {
            model: restore.download.model,
            revision: c.revision,
          }),
        );
      }
    } catch (e) {
      setError(errorText(e));
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => {
    if (requested) void load(false, requested);
  }, [requested?.id]);
  const inspect = async (name: string) => {
    if (!catalog) return;
    setSelected(name);
    setLoading(true);
    setError('');
    setPackages([]);
    try {
      setPackages(
        await api<Package[]>('/api/hub/packages', 'POST', {
          model: name,
          revision: catalog.revision,
        }),
      );
    } catch (e) {
      setError(errorText(e));
    } finally {
      setLoading(false);
    }
  };
  const start = async (p: Package) => {
    setError('');
    setLoading(true);
    try {
      const job = await api<Download>('/api/hub/downloads', 'POST', {
        path: p.path,
        revision: p.revision,
        directory,
        model_id: restoreModel?.id,
      });
      completed.current.delete(job.id);
      setDownloads((prev) => [job, ...prev.filter((d) => d.id !== job.id)]);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setLoading(false);
    }
  };
  const cancel = async (id: string) => {
    try {
      await api(`/api/hub/cancel/${id}`, 'POST', {});
    } catch (e) {
      setError(errorText(e));
    }
  };
  const renderDownload = (d: Download) => (
    <div key={d.id} className="hub-download">
      <div>
        <strong>{d.name}</strong>
        <Badge tone={d.status === 'completed' ? 'blue' : d.status === 'failed' ? 'red' : ''}>
          {d.status.toUpperCase()}
        </Badge>
        {['downloading', 'verifying'].includes(d.status) && (
          <button
            className="icon-button"
            aria-label={`Cancel download ${d.name}`}
            onClick={() => cancel(d.id)}
          >
            <Square size={13} />
          </button>
        )}
      </div>
      <div className="download-track">
        <span
          style={{
            width: `${Math.min(100, (100 * d.downloaded_bytes) / Math.max(1, d.total_bytes))}%`,
          }}
        />
      </div>
      <p>
        {bytes(d.downloaded_bytes)} / {bytes(d.total_bytes)}
        {d.speed > 0 ? ` · ${bytes(d.speed)}/s` : ''} · {d.file}
      </p>
      {d.error && <p className="inline-error">{d.error}</p>}
      {d.status === 'completed' && (
        <p>
          <Check size={12} />
          Verified and added to My models
        </p>
      )}
    </div>
  );
  return (
    <>
      <button className="button subtle" onClick={() => load()}>
        <Globe size={16} />
        Discover models
      </button>
      {active && (
        <button
          className="download-notice"
          onClick={() => {
            setOpen(true);
            if (!catalog) void load();
          }}
        >
          <Spinner />
          {active.name} ·{' '}
          {Math.round((100 * active.downloaded_bytes) / Math.max(1, active.total_bytes))}%
        </button>
      )}
      {open && (
        <Modal
          title="Download models from Hugging Face"
          onClose={() => {
            setOpen(false);
            onRequestHandled?.();
          }}
        >
          <div className="hub-source">
            <Globe size={17} />
            <span>webai-community / ai-models</span>
            <button
              className="icon-button"
              title="Refresh repository"
              aria-label="Refresh repository"
              disabled={loading}
              onClick={() => load(true)}
            >
              <RefreshCw size={14} />
            </button>
          </div>
          <p className="muted small">
            Download a model when you need it. Deleting its local files keeps its entry in My
            models.
          </p>
          {error && (
            <div className="inline-error" role="alert">
              {error}
            </div>
          )}
          {catalog && !selected && (
            <>
              <label className="search-input hub-search">
                <Search size={15} />
                <input
                  aria-label="Search Hugging Face models"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search Qwen, SmolLM, Gemma, Phi…"
                />
              </label>
              <div className="hub-models">
                {catalog.models
                  .filter((m) => m.name.toLowerCase().includes(search.toLowerCase()))
                  .map((m) => (
                    <button key={m.name} disabled={loading} onClick={() => inspect(m.name)}>
                      <span>{m.name}</span>
                      <ArrowRight size={14} />
                    </button>
                  ))}
              </div>
              <p className="small muted">
                Select a model to see available ONNX variants and exact download sizes. GGUF and
                other formats are excluded.
              </p>
            </>
          )}
          {selected && (
            <>
              <button
                className="text-button hub-back"
                onClick={() => {
                  setSelected('');
                  setRestoreModel(null);
                  setError('');
                }}
                disabled={loading}
              >
                <ArrowLeft size={14} />
                All models
              </button>
              <h3>{selected}</h3>
              {packages.length > 0 && (
                <PathInput
                  label="Save downloaded models in"
                  value={directory}
                  onChange={setDirectory}
                />
              )}
              <div className="hub-packages">
                {packages.map((p) => {
                  const local = models.some(
                    (m) =>
                      m.available &&
                      m.source?.repository === 'webai-community/ai-models' &&
                      m.source?.path === p.path &&
                      m.source?.revision === p.revision,
                  );
                  return (
                    <article className="hub-package" key={p.id}>
                      <div>
                        <strong>{p.variant}</strong>
                        <Badge>{p.kind.toUpperCase()}</Badge>
                      </div>
                      <div className="tag-row">
                        {p.modalities.map((m) => (
                          <Badge key={m}>{m.toUpperCase()}</Badge>
                        ))}
                      </div>
                      <p>
                        <strong>{bytes(p.size)}</strong> · {p.files.length} files · {p.architecture}
                      </p>
                      <details>
                        <summary>Files and source revision</summary>
                        <code>{p.revision}</code>
                        <ul>
                          {p.files.map((f) => (
                            <li key={f.name}>
                              <span>{f.name}</span>
                              <span>{bytes(f.size)}</span>
                            </li>
                          ))}
                        </ul>
                      </details>
                      <div className="hub-package-actions">
                        <a href={p.source_url} target="_blank" rel="noreferrer">
                          View source <ExternalLink size={12} />
                        </a>
                        <button
                          className="button primary"
                          disabled={loading || !!active || !directory || local}
                          onClick={() => start(p)}
                        >
                          {local ? <Check size={14} /> : <ArrowDownToLine size={14} />}{' '}
                          {local ? 'Downloaded' : 'Download'}
                        </button>
                      </div>
                    </article>
                  );
                })}
              </div>
              {!loading && !packages.length && (
                <p className="muted">
                  No supported native ONNX package with a tokenizer was found for this model.
                </p>
              )}
              <p className="small muted">
                Files are pinned to this revision and verified before import. Operator and
                architecture support still depend on your runtime; compatibility is checked when
                loading.
              </p>
            </>
          )}
          {loading && (
            <div className="hub-loading">
              <Spinner />
              Loading package information…
            </div>
          )}
          {downloads.length > 0 && (
            <section className="hub-downloads">
              <h3>Downloads</h3>
              {downloads.slice(0, 5).map(renderDownload)}
            </section>
          )}
        </Modal>
      )}
    </>
  );
}
