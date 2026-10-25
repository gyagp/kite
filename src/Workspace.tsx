import { useEffect, useState } from 'react';
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  Box,
  Check,
  CheckCircle2,
  Code2,
  Cpu,
  FolderOpen,
  HardDrive,
  Image,
  Layers3,
  MessageSquare,
  Mic,
  Video,
  Plus,
  RefreshCw,
  Search,
  ShieldCheck,
  Terminal,
  Trash2,
  X,
} from 'lucide-react';
import { api, bytes, download, errorText, ms } from './api';
import { Badge, CopyButton, Modal, PathInput, Spinner } from './components';
import { ModelHub } from './ModelHub';
import { EmbeddingStats } from './EmbeddingStats';
import type { Bootstrap, Execute, Model, Run, Settings as SettingsType, Status } from './types';

export function Models({
  models,
  settings,
  status,
  busy,
  execute,
  refresh,
  notify,
  onUse,
}: {
  models: Model[];
  settings: SettingsType;
  status: Status;
  busy: boolean;
  execute: Execute;
  refresh: () => Promise<void>;
  notify: (s: string) => void;
  onUse: (m: Model) => void;
}) {
  const [filter, setFilter] = useState('all');
  const [search, setSearch] = useState('');
  const [availability, setAvailability] = useState('all');
  const [showImport, setShowImport] = useState(false);
  const [path, setPath] = useState(settings.asset_root + '\\artifacts');
  const [scanned, setScanned] = useState<Model[]>([]);
  const [scanErrors, setScanErrors] = useState<{ path: string; error: string }[]>([]);
  const [importError, setImportError] = useState('');
  const [scanning, setScanning] = useState(false);
  const [adding, setAdding] = useState<string | null>(null);
  const [details, setDetails] = useState<Model | null>(null);
  const [remove, setRemove] = useState<Model | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteConfirmed, setDeleteConfirmed] = useState(false);
  const [downloadModel, setDownloadModel] = useState<Model | null>(null);
  const shown = models.filter(
    (m) =>
      (filter === 'all' || m.kind === filter) &&
      (availability === 'all' || (availability === 'downloaded' ? m.available : !m.available)) &&
      (m.name + ' ' + m.variant).toLowerCase().includes(search.toLowerCase()),
  );
  const add = async (value: string) => {
    setAdding(value);
    setImportError('');
    try {
      await api('/api/models/import', 'POST', { path: value });
      await refresh();
    } catch (e) {
      setImportError(errorText(e));
    } finally {
      setAdding(null);
    }
  };
  const scan = async () => {
    setScanning(true);
    setImportError('');
    try {
      const r = await api<{ models: Model[]; errors: { path: string; error: string }[] }>(
        '/api/models/scan',
        'POST',
        { path },
      );
      setScanned(r.models);
      setScanErrors(r.errors);
      if (!r.models.length && !r.errors.length)
        setImportError(
          'No compatible packages were found in this directory or its immediate subdirectories.',
        );
    } catch (e) {
      setImportError(errorText(e));
    } finally {
      setScanning(false);
    }
  };
  return (
    <div className="workspace-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">
            <span className="tiny-line" /> YOUR MODEL LIBRARY
          </div>
          <h1>
            Good company for your ideas<span className="accent">.</span>
          </h1>
          <p>Your chosen models, ready when you need them.</p>
        </div>
        <div className="button-row">
          <ModelHub
            models={models}
            refresh={refresh}
            requested={downloadModel}
            onRequestHandled={() => setDownloadModel(null)}
          />
          <button className="button primary" onClick={() => setShowImport(true)}>
            <Plus size={16} />
            Import model
          </button>
        </div>
      </div>
      <div className="library-stats">
        <div>
          <Box size={19} />
          <strong>{models.length}</strong>
          <span>models in your library</span>
        </div>
        <div>
          <HardDrive size={19} />
          <strong>{bytes(models.reduce((s, m) => s + (m.available ? m.size : 0), 0))}</strong>
          <span>{models.filter((m) => m.available).length} downloaded</span>
        </div>
        <div>
          <Cpu size={19} />
          <strong>WebGPU</strong>
          <span>source-built runtime</span>
        </div>
        <div>
          <ShieldCheck size={19} />
          <strong>100% local</strong>
          <span>no cloud inference</span>
        </div>
      </div>
      <div className="library-toolbar">
        <div className="segmented">
          {[
            ['all', 'All models'],
            ['chat', 'Chat'],
            ['embedding', 'Embeddings'],
            ['image', 'Images'],
            ['segmentation', 'Segmentation'],
          ].map(([id, label]) => (
            <button
              key={id}
              className={filter === id ? 'active' : ''}
              onClick={() => setFilter(id)}
            >
              {label}
            </button>
          ))}
        </div>
        <select
          aria-label="Model availability"
          value={availability}
          onChange={(e) => setAvailability(e.target.value)}
        >
          <option value="all">All availability</option>
          <option value="downloaded">Downloaded</option>
          <option value="not-downloaded">Not downloaded</option>
        </select>
        <label className="search-input">
          <Search size={15} />
          <input
            aria-label="Search models"
            placeholder="Find a model…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </label>
      </div>
      <div className="model-grid">
        {shown.map((m) => (
          <article className="panel model-card" key={m.id}>
            <div className="model-card-top">
              <div className={`model-glyph ${m.kind === 'chat' ? 'violet' : ''}`}>
                {m.kind === 'embedding'
                  ? 'Eg'
                  : m.name.startsWith('Qwen')
                    ? 'Q'
                    : m.name.slice(0, 2)}
                {m.kind === 'embedding' && <span>2</span>}
              </div>
              <Badge tone={status.loaded?.model === m.id ? 'blue' : ''}>
                {status.loaded?.model === m.id
                  ? '● LOADED'
                  : m.available
                    ? 'ON DISK'
                    : 'NOT DOWNLOADED'}
              </Badge>
              <button
                className="icon-button"
                aria-label={`Delete ${m.name} from disk`}
                title="Delete from disk; keep in My models"
                disabled={busy || !m.available}
                onClick={() => {
                  setRemove(m);
                  setDeleteConfirmed(false);
                }}
              >
                <Trash2 size={15} />
              </button>
            </div>
            <h2>{m.name}</h2>
            <p className="model-variant">{m.variant}</p>
            <div className="tag-row">
              <Badge tone={m.kind === 'embedding' ? 'purple' : 'blue'}>
                {m.kind === 'embedding'
                  ? 'EMBEDDINGS'
                  : m.kind === 'image'
                    ? 'IMAGE GENERATION'
                    : m.kind === 'segmentation'
                      ? 'VIDEO TRACKING'
                      : 'CHAT'}
              </Badge>
              <Badge>{m.precision.toUpperCase()}</Badge>
              {m.modalities.map((v) => (
                <Badge key={v}>
                  {v === 'text' ? (
                    <MessageSquare size={10} />
                  ) : v === 'image' ? (
                    <Image size={10} />
                  ) : v === 'video' ? (
                    <Video size={10} />
                  ) : (
                    <Mic size={10} />
                  )}{' '}
                  {v.toUpperCase()}
                </Badge>
              ))}
            </div>
            <dl className="card-specs">
              <div>
                <dt>Model size</dt>
                <dd>{bytes(m.size)}</dd>
              </div>
              <div>
                <dt>
                  {m.kind === 'image'
                    ? 'Generation'
                    : m.kind === 'segmentation'
                      ? 'Tracking'
                      : m.dimensions
                        ? 'Vector size'
                        : 'Model context'}
                </dt>
                <dd>
                  {m.kind === 'image'
                    ? '8-step diffusion'
                    : m.kind === 'segmentation'
                      ? 'Points · boxes · video'
                      : m.dimensions
                        ? `${m.dimensions} dimensions`
                        : `${(m.context / 1024).toFixed(0)}K tokens`}
                </dd>
              </div>
              <div>
                <dt>Runtime</dt>
                <dd>{m.kind === 'image' ? 'WebGPU + CUDA' : 'ORT · D3D12'}</dd>
              </div>
            </dl>
            <div className="model-card-validation">
              {m.validation?.passed ? (
                <>
                  <CheckCircle2 size={14} />
                  Export validation passed
                </>
              ) : (
                <>
                  <Code2 size={14} />
                  {!m.available && m.download
                    ? 'Available on Hugging Face'
                    : m.kind === 'image'
                      ? 'Native image pipeline'
                      : m.kind === 'segmentation'
                        ? 'Native SAM 3 video pipeline'
                        : 'Local GenAI package'}
                </>
              )}
            </div>
            <div className="model-card-footer">
              <button className="text-button" onClick={() => setDetails(m)}>
                Model details
              </button>
              {m.available ? (
                <>
                  <button
                    className="button subtle"
                    disabled={busy || !m.available}
                    onClick={() =>
                      m.kind === 'image' || m.kind === 'segmentation'
                        ? onUse(m)
                        : execute({ kind: 'load', model: m.id }).catch(() => {})
                    }
                  >
                    {status.loaded?.model === m.id ? <Check size={14} /> : <Cpu size={14} />}
                    {m.kind === 'image' || m.kind === 'segmentation' ? 'Studio' : 'Load'}
                  </button>
                  <button
                    className="button primary"
                    disabled={busy || !m.available}
                    onClick={() => onUse(m)}
                  >
                    Use model
                    <ArrowRight size={14} />
                  </button>
                </>
              ) : m.download ? (
                <button
                  className="button primary"
                  disabled={busy}
                  onClick={() => setDownloadModel(m)}
                >
                  <ArrowDownToLine size={14} /> Download
                </button>
              ) : (
                <button className="button subtle" onClick={() => setShowImport(true)}>
                  <FolderOpen size={14} /> Import files
                </button>
              )}
            </div>
          </article>
        ))}
        <button className="import-card" onClick={() => setShowImport(true)}>
          <span>
            <Plus size={23} />
          </span>
          <h3>Make room for another model</h3>
          <p>
            Import an ONNX export or an ORT GenAI package.
            <br />
            Your model files stay right where they are.
          </p>
          <strong>
            Browse local packages <ArrowRight size={14} />
          </strong>
        </button>
      </div>
      <div className="library-note">
        <Layers3 size={19} />
        <div>
          <strong>Multimodal starts with the export.</strong>
          <p>
            EmbeddingGemma 2 supports text, images, audio, and video. GenAI chat capabilities follow
            the encoder graphs in each package. Import local exports in place or download
            revision-pinned ONNX packages from the model browser.
          </p>
        </div>
      </div>
      {showImport && (
        <Modal title="Bring a model into Kite" onClose={() => setShowImport(false)}>
          <p className="muted">
            Choose a package or scan your artifacts directory. Files are referenced in place and
            never copied or modified.
          </p>
          <PathInput label="Package or artifacts directory" value={path} onChange={setPath} />
          <div className="button-row">
            <button className="button subtle" disabled={scanning || !!adding} onClick={scan}>
              {scanning ? <Spinner /> : <Search size={15} />}Scan directory
            </button>
            <button
              className="button primary"
              disabled={scanning || !!adding}
              onClick={() => add(path)}
            >
              {adding === path ? <Spinner /> : <Plus size={15} />}Import this package
            </button>
          </div>
          {importError && (
            <div className="inline-error" role="alert">
              {importError}
            </div>
          )}
          {scanned.length > 0 && (
            <div className="scan-results">
              <h4>{scanned.length} packages found</h4>
              {scanned.map((m) => (
                <div className="scan-result" key={m.id}>
                  <span className="scan-icon">
                    {m.kind === 'embedding' ? <Layers3 size={17} /> : <MessageSquare size={17} />}
                  </span>
                  <div>
                    <strong>{m.name}</strong>
                    <small>
                      {m.variant} · {bytes(m.size)}
                    </small>
                  </div>
                  {models.some((x) => x.id === m.id) ? (
                    <Badge tone="blue">
                      <Check size={12} />
                      ADDED
                    </Badge>
                  ) : (
                    <button
                      className="button subtle"
                      disabled={!!adding}
                      onClick={() => add(m.path)}
                    >
                      {adding === m.path ? <Spinner /> : <Plus size={13} />}Add
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}
          {scanErrors.length > 0 && (
            <details>
              <summary>{scanErrors.length} incomplete packages skipped</summary>
              {scanErrors.map((e) => (
                <p key={e.path} className="small muted">
                  {e.path}: {e.error}
                </p>
              ))}
            </details>
          )}
          <div className="modal-footnote">
            <FolderOpen size={14} />
            Expected: ONNX graphs, tokenizer, and embedding_config.json or genai_config.json.
          </div>
        </Modal>
      )}
      {details && (
        <Modal title={details.name} onClose={() => setDetails(null)}>
          <dl className="detail-list">
            <div>
              <dt>Package</dt>
              <dd>
                {details.available ? details.path : 'Not downloaded'}
                {details.path && <CopyButton text={details.path} />}
              </dd>
            </div>
            <div>
              <dt>Architecture</dt>
              <dd>{details.architecture}</dd>
            </div>
            <div>
              <dt>Input modalities</dt>
              <dd>{details.modalities.join(', ')}</dd>
            </div>
            <div>
              <dt>Export</dt>
              <dd>{details.variant}</dd>
            </div>
            <div>
              <dt>Model ID</dt>
              <dd>
                <code>{details.id}</code>
                <CopyButton text={details.id} />
              </dd>
            </div>
            <div>
              <dt>Context</dt>
              <dd>
                {details.context.toLocaleString()} tokens (application context is configured
                separately)
              </dd>
            </div>
          </dl>
          {details.validation && <pre>{JSON.stringify(details.validation, null, 2)}</pre>}
          <button className="button subtle" onClick={() => download('kite-model.json', details)}>
            <ArrowDownToLine size={15} />
            Export model information
          </button>
        </Modal>
      )}
      {remove && (
        <Modal
          title={`Delete ${remove.name}?`}
          onClose={() => {
            if (!deleting) setRemove(null);
          }}
        >
          <p>
            Delete the local model files to free disk space. The model stays in My models so you can{' '}
            {remove.download ? 'download' : 'import'} it again.
          </p>
          <div className="delete-model-path">
            <HardDrive size={18} />
            <div>
              <strong>{bytes(remove.size)} of model files</strong>
              <code>{remove.path}</code>
            </div>
          </div>
          {remove.path.toLowerCase().startsWith(settings.asset_root.toLowerCase()) && (
            <p className="delete-model-warning">
              This model is stored in Kite’s local asset library. Deleting its files removes the
              copy.
            </p>
          )}
          <label className="delete-model-confirm">
            <input
              type="checkbox"
              checked={deleteConfirmed}
              disabled={deleting}
              onChange={(e) => setDeleteConfirmed(e.target.checked)}
            />
            <span>
              I understand that deleting from disk permanently removes this directory and everything
              inside it.
            </span>
          </label>
          <div className="modal-footer">
            <button className="button subtle" disabled={deleting} onClick={() => setRemove(null)}>
              Cancel
            </button>
            <button
              className="button danger"
              disabled={deleting || !deleteConfirmed}
              onClick={async () => {
                setDeleting(true);
                try {
                  await api(`/api/models/${remove.id}/delete`, 'POST', {
                    delete_files: true,
                    confirm_path: remove.path,
                  });
                  setRemove(null);
                  await refresh();
                } catch (e) {
                  notify(errorText(e));
                } finally {
                  setDeleting(false);
                }
              }}
            >
              {deleting ? <Spinner /> : <Trash2 size={14} />}Delete from disk
            </button>
          </div>
        </Modal>
      )}
    </div>
  );
}

export function Settings({
  boot,
  status,
  busy,
  refresh,
  notify,
}: {
  boot: Bootstrap;
  status: Status;
  busy: boolean;
  refresh: () => Promise<void>;
  notify: (s: string) => void;
}) {
  const [settings, setSettings] = useState(boot.settings);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [showKey, setShowKey] = useState(false);
  const [apiTab, setApiTab] = useState('embeddings');
  const runtime = status.loaded?.runtime;
  const modelId =
    boot.models.find((m) => m.kind === (apiTab === 'embeddings' ? 'embedding' : 'chat'))?.id ||
    '<model-id>';
  const sample =
    apiTab === 'embeddings'
      ? `from openai import OpenAI\n\nclient = OpenAI(\n    base_url="${boot.api_url}",\n    api_key="<your-local-api-key>",\n)\n\nresult = client.embeddings.create(\n    model="${modelId}",\n    input="A little intelligence, entirely yours.",\n)\nprint(result.data[0].embedding)`
      : `from openai import OpenAI\n\nclient = OpenAI(\n    base_url="${boot.api_url}",\n    api_key="<your-local-api-key>",\n)\n\nfor chunk in client.chat.completions.create(\n    model="${modelId}",\n    messages=[{"role": "user", "content": "Hello!"}],\n    stream=True,\n):\n    print(chunk.choices[0].delta.content or "", end="")`;
  const change = (key: keyof SettingsType, value: string | number) => {
    setSettings({ ...settings, [key]: value });
    setSaved(false);
  };
  return (
    <div className="workspace-page settings-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">
            <span className="tiny-line" /> MAKE YOURSELF AT HOME
          </div>
          <h1>
            Your engine, your rules<span className="accent">.</span>
          </h1>
          <p>Connect your runtime. Keep your workflow local.</p>
        </div>
      </div>
      <div className="settings-layout">
        <div>
          <section className="panel settings-panel">
            <div className="section-title">
              <div>
                <Cpu size={18} />
                <h2>Native inference runtime</h2>
              </div>
              <Badge>D3D12</Badge>
            </div>
            <p className="muted">
              Kite runs your source-built ONNX Runtime WebGPU provider in a dedicated process. Use
              the Python environment that matches your build.
            </p>
            <form
              onSubmit={async (e) => {
                e.preventDefault();
                setSaving(true);
                try {
                  await api('/api/settings', 'POST', settings);
                  await refresh();
                  setSaved(true);
                } catch (err) {
                  notify(errorText(err));
                } finally {
                  setSaving(false);
                }
              }}
            >
              <fieldset disabled={busy || saving}>
                <PathInput
                  label="Inference Python executable"
                  kind="python"
                  value={settings.python}
                  onChange={(v) => change('python', v)}
                />
                <PathInput
                  label="WebGPU provider DLL"
                  kind="runtime"
                  value={settings.library}
                  onChange={(v) => change('library', v)}
                />
                <PathInput
                  label="Local asset library"
                  value={settings.asset_root}
                  onChange={(v) => change('asset_root', v)}
                />
                <label className="field">
                  <span>Application context limit</span>
                  <select
                    value={settings.context}
                    onChange={(e) => change('context', +e.target.value)}
                  >
                    {[2048, 4096, 8192, 16384, 32768].map((n) => (
                      <option key={n} value={n}>
                        {n.toLocaleString()} tokens
                      </option>
                    ))}
                  </select>
                </label>
                <p className="small muted">
                  Includes input and generated tokens. The model’s own limit still applies. Larger
                  contexts can use significantly more GPU memory.
                </p>
                <div className="setting-note">
                  <ShieldCheck size={15} />
                  Graph capture is disabled for the initial correctness-focused integration.
                </div>
                <div className="settings-save">
                  <span>
                    {saved && (
                      <>
                        <Check size={14} />
                        Runtime settings saved
                      </>
                    )}
                  </span>
                  <button className="button primary" type="submit">
                    {saving ? <Spinner /> : <Check size={15} />}Save settings
                  </button>
                </div>
              </fieldset>
            </form>
          </section>
          <section className="panel settings-panel api-panel">
            <div className="section-title">
              <div>
                <Terminal size={18} />
                <h2>A local API for your ideas</h2>
              </div>
              <Badge tone="blue">LOOPBACK ONLY</Badge>
            </div>
            <p className="muted">
              Connect scripts and compatible applications to Kite’s embeddings and streaming chat
              endpoints.
            </p>
            <div className="api-address">
              <code>{boot.api_url}</code>
              <CopyButton text={boot.api_url} />
            </div>
            <label className="field">
              <span>Local API key</span>
              <div className="api-key">
                <input
                  aria-label="Local API key"
                  readOnly
                  type={showKey ? 'text' : 'password'}
                  value={boot.api_key}
                />
                <button className="text-button" onClick={() => setShowKey(!showKey)}>
                  {showKey ? 'Hide' : 'Show'}
                </button>
                <CopyButton text={boot.api_key} />
              </div>
            </label>
            <div className="code-tabs">
              <button
                className={apiTab === 'embeddings' ? 'active' : ''}
                onClick={() => setApiTab('embeddings')}
              >
                Embeddings
              </button>
              <button
                className={apiTab === 'chat' ? 'active' : ''}
                onClick={() => setApiTab('chat')}
              >
                Streaming chat
              </button>
              <CopyButton text={sample} />
            </div>
            <pre className="code-example">{sample}</pre>
            <p className="small muted">
              Uses the OpenAI Python client as an HTTP client; requests go to your local service.
              Install with <code>pip install openai</code>. This API currently supports text, local
              image data URLs, and WAV audio for compatible packages. Tool calling is not
              implemented.
            </p>
          </section>
        </div>
        <aside className="inspector">
          <section className="panel runtime-info">
            <div className="inspector-title">
              <Activity size={16} />
              Runtime status
            </div>
            <Badge tone={runtime ? 'blue' : ''}>{runtime ? 'ACTIVE' : 'NOT LOADED'}</Badge>
            {runtime ? (
              <dl className="model-details">
                <div>
                  <dt>ORT host</dt>
                  <dd>{runtime.host_version}</dd>
                </div>
                {runtime.genai_version && (
                  <div>
                    <dt>GenAI</dt>
                    <dd>{runtime.genai_version}</dd>
                  </div>
                )}
                <div>
                  <dt>Source revision</dt>
                  <dd className="mono">{runtime.build?.revision?.slice(0, 10) || 'No sidecar'}</dd>
                </div>
                <div>
                  <dt>Backend</dt>
                  <dd>D3D12</dd>
                </div>
              </dl>
            ) : (
              <p className="small muted">
                Load a model to see the runtime version and GPU reported by the provider.
              </p>
            )}
            {runtime?.devices.map((d, i) => (
              <div className="device-info" key={i}>
                <Cpu size={20} />
                <strong>
                  {Object.values(d.metadata).find((v) =>
                    /NVIDIA|AMD|Intel|GeForce|Radeon/i.test(v),
                  ) || d.name}
                </strong>
                <pre>{JSON.stringify(d.metadata, null, 2)}</pre>
              </div>
            ))}
            <button
              className="text-button"
              onClick={async () => {
                try {
                  download('kite-diagnostics.json', await api('/api/diagnostics'));
                } catch (e) {
                  notify(errorText(e));
                }
              }}
            >
              <ArrowDownToLine size={14} />
              Export diagnostics
            </button>
          </section>
          <section className="tip-card">
            <HardDrive size={18} />
            <h4>A workspace that stays with you</h4>
            <p>Settings, conversations, and run history are saved in a local SQLite database.</p>
            <code className="storage-path">{boot.data_directory}</code>
            <p>Model weights are referenced from their original directories.</p>
          </section>
        </aside>
      </div>
    </div>
  );
}

export function ActivityPage({
  models,
  notify,
  busy,
}: {
  models: Model[];
  notify: (s: string) => void;
  busy: boolean;
}) {
  const [runs, setRuns] = useState<Run[]>([]);
  const [filter, setFilter] = useState('all');
  const [selected, setSelected] = useState<Run | null>(null);
  const refresh = () =>
    api<Run[]>('/api/runs')
      .then(setRuns)
      .catch((e) => notify(errorText(e)));
  useEffect(() => {
    void refresh();
  }, [busy]);
  const shown = runs.filter((r) => filter === 'all' || r.status === filter);
  return (
    <div className="workspace-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">
            <span className="tiny-line" /> A LITTLE VISIBILITY
          </div>
          <h1>
            See your models at work<span className="accent">.</span>
          </h1>
          <p>Real runs, real timings. Your last 100 operations, all in one place.</p>
        </div>
        <div className="button-row">
          <button className="button subtle" onClick={refresh}>
            <RefreshCw size={15} />
            Refresh
          </button>
          <button
            className="button subtle"
            disabled={!runs.length}
            onClick={() => download('kite-activity.json', runs)}
          >
            <ArrowDownToLine size={15} />
            Export
          </button>
        </div>
      </div>
      <div className="activity-summary">
        <div className="panel">
          <span>COMPLETED</span>
          <strong>{runs.filter((r) => r.status === 'completed').length}</strong>
          <CheckCircle2 size={22} />
        </div>
        <div className="panel">
          <span>FAILED</span>
          <strong>{runs.filter((r) => r.status === 'failed').length}</strong>
          <X size={22} />
        </div>
        <div className="panel">
          <span>TOKENS PROCESSED</span>
          <strong>
            {runs
              .reduce(
                (s, r) =>
                  s +
                  (r.result?.tokens || r.result?.prompt_tokens || 0) +
                  (r.result?.completion_tokens || 0),
                0,
              )
              .toLocaleString()}
          </strong>
          <Layers3 size={22} />
        </div>
      </div>
      <div className="library-toolbar">
        <div className="segmented">
          {['all', 'completed', 'failed', 'cancelled'].map((f) => (
            <button key={f} className={f === filter ? 'active' : ''} onClick={() => setFilter(f)}>
              {f[0].toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>
        <span className="small muted">
          Timings include model loading unless marked as inference.
        </span>
      </div>
      <section className="panel table-panel">
        {!shown.length ? (
          <div className="result-empty">
            <Activity size={32} />
            <h3>No runs to show yet</h3>
            <p>Generate an embedding or start a conversation to see it here.</p>
          </div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Operation</th>
                <th>Model</th>
                <th>Status</th>
                <th>Duration</th>
                <th>When</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {shown.map((r) => (
                <tr key={r.id}>
                  <td>
                    <span className="operation-icon">
                      {r.kind === 'chat' ? (
                        <MessageSquare size={15} />
                      ) : r.kind === 'load' ? (
                        <Cpu size={15} />
                      ) : (
                        <Layers3 size={15} />
                      )}
                    </span>
                    {{
                      chat: 'Chat completion',
                      load: 'Load model',
                      search: 'Semantic search',
                      embed: 'Embedding',
                      image: 'Image generation',
                      track: 'Video tracking',
                      tool: 'Tool run',
                    }[r.kind] || r.kind}
                  </td>
                  <td>
                    {models.find((m) => m.id === r.model)?.name ||
                      (r.kind === 'tool' ? r.model.replace(/^(evo|tool):/, '') : r.model)}
                  </td>
                  <td>
                    <Badge
                      tone={r.status === 'completed' ? 'blue' : r.status === 'failed' ? 'red' : ''}
                    >
                      {r.status.toUpperCase()}
                    </Badge>
                  </td>
                  <td className="mono">{ms(r.elapsed_ms)}</td>
                  <td className="muted">
                    {new Date(r.created * 1000).toLocaleString([], {
                      month: 'short',
                      day: 'numeric',
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </td>
                  <td>
                    <button
                      className="icon-button"
                      aria-label={`View run ${r.id}`}
                      onClick={() => setSelected(r)}
                    >
                      <ArrowRight size={15} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
      {selected && (
        <Modal title="Run details" className="performance-modal" onClose={() => setSelected(null)}>
          {selected.error && <div className="inline-error">{selected.error}</div>}
          {selected.result?.inference_ms != null && ['embed', 'search'].includes(selected.kind) && (
            <EmbeddingStats metrics={selected.result} />
          )}
          <pre>{JSON.stringify(selected, null, 2)}</pre>
          <button
            className="button subtle"
            onClick={() => download(`kite-run-${selected.id}.json`, selected)}
          >
            <ArrowDownToLine size={15} />
            Export run
          </button>
        </Modal>
      )}
    </div>
  );
}
