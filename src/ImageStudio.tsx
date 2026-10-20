import { useEffect, useRef, useState } from 'react';
import {
  ArrowDownToLine,
  ArrowUpRight,
  Expand,
  ImagePlus,
  RefreshCw,
  Settings2,
  Sparkles,
  WandSparkles,
  X,
} from 'lucide-react';
import { api, errorText, ms } from './api';
import { Badge, Modal, PathInput, Spinner } from './components';
import type { Attachment, Execute } from './types';
import { ImagePerformance } from './ImagePerformance';
import { ImageProgress, imageStages } from './ImageProgress';
import type { ImagePhase, ImageProfile } from './image-performance';
import './studio.css';

type Preset = {
  id: string;
  title: string;
  category: string;
  subtitle: string;
  prompt: string;
  seed: number;
  width: number;
  height: number;
  preview?: string;
  reference?: string;
};
type Artwork = {
  id: string;
  prompt: string;
  seed: number;
  width: number;
  height: number;
  image_url: string;
  attachments: string[];
  request_ms: number;
  steps: number;
  created: number;
  capture: boolean;
  phases: Record<string, ImagePhase>;
  timing_method: string;
  profile?: ImageProfile | null;
};
type Config = {
  output_directory?: string;
  settings: {
    source: string;
    package: string;
    library: string;
    python: string;
    dependencies: string[];
  };
  missing: string[];
};
const asset = (name: string) => '/api/images/presets/' + name;

async function attachFile(file: File | Blob, name: string) {
  const raw = await file.arrayBuffer();
  let binary = '';
  for (const value of new Uint8Array(raw)) binary += String.fromCharCode(value);
  return api<Attachment>('/api/attachments', 'POST', { name, data: btoa(binary) });
}

export function ImageStudio({
  execute,
  busy,
  notify,
  onStop,
}: {
  execute: Execute;
  busy: boolean;
  notify: (text: string) => void;
  onStop: () => void;
}) {
  const [presets, setPresets] = useState<Preset[]>([]);
  const [selected, setSelected] = useState<Preset | null>(null);
  const [prompt, setPrompt] = useState('');
  const [seed, setSeed] = useState(42);
  const [size, setSize] = useState('768x768');
  const [capture, setCapture] = useState(false);
  const [profile, setProfile] = useState(false);
  const [references, setReferences] = useState<Attachment[]>([]);
  const [history, setHistory] = useState<Artwork[]>([]);
  const [art, setArt] = useState<Artwork | null>(null);
  const [config, setConfig] = useState<Config | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const [stage, setStage] = useState('');
  const [step, setStep] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);
  const [stageMessage, setStageMessage] = useState('');
  const [running, setRunning] = useState(false);
  const [loadingPreset, setLoadingPreset] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [split, setSplit] = useState(50);
  const [tab, setTab] = useState('create');
  const fileInput = useRef<HTMLInputElement>(null);
  const started = useRef(0);
  const progressPanel = useRef<HTMLDivElement>(null);
  useEffect(() => {
    Promise.all([
      api<Preset[]>('/api/images/presets'),
      api<Config>('/api/images/config'),
      api<Artwork[]>('/api/images'),
    ])
      .then(([p, c, h]) => {
        setPresets(p);
        setConfig(c);
        setHistory(h);
        if (p.length) {
          setSelected(p[0]);
          setPrompt(p[0].prompt);
        }
      })
      .catch((e) => notify(errorText(e)));
  }, [notify]);
  useEffect(() => {
    if (!running) return;
    progressPanel.current?.scrollIntoView({ block: 'nearest', behavior: 'instant' });
    const timer = setInterval(() => setElapsed((Date.now() - started.current) / 1000), 500);
    return () => clearInterval(timer);
  }, [running]);
  const usePreset = async (p: Preset) => {
    setLoadingPreset(true);
    try {
      const refs = p.reference
        ? [await attachFile(await (await fetch(asset(p.reference))).blob(), p.reference)]
        : [];
      setSelected(p);
      setPrompt(p.prompt);
      setSeed(p.seed);
      setSize(`${p.width}x${p.height}`);
      setArt(null);
      setReferences(refs);
      setTab('create');
    } catch (e) {
      notify(errorText(e));
    } finally {
      setLoadingPreset(false);
    }
  };
  const generate = async () => {
    setRunning(true);
    setArt(null);
    setStep(0);
    setTotalSteps(0);
    setStageMessage('Starting the image engine…');
    setTab('create');
    setStage('prepare');
    setElapsed(0);
    started.current = Date.now();
    try {
      const [width, height] = size.split('x').map(Number);
      const result = await execute(
        {
          kind: 'image',
          prompt,
          seed,
          width,
          height,
          capture,
          profile,
          attachments: references.map((a) => a.id),
        },
        (e) => {
          if (e.type === 'status') {
            if (imageStages.some(([key]) => key === e.stage)) setStage(e.stage!);
            if (e.message) setStageMessage(e.message);
          }
          if (e.type === 'image_progress') {
            setStage('denoise');
            setStep(e.completed || 0);
            setTotalSteps(e.total || 0);
            setStageMessage('Bringing your image to life…');
          }
        },
      );
      if (result) {
        setArt(result as unknown as Artwork);
        setHistory(await api<Artwork[]>('/api/images'));
      }
    } catch {
      /* App presents worker errors. */
    } finally {
      setRunning(false);
    }
  };
  const reuse = (a: Artwork) => {
    setArt(a);
    setPrompt(a.prompt);
    setSeed(a.seed);
    setSize(`${a.width}x${a.height}`);
    setSelected(null);
    setTab('create');
    setReferences(
      a.attachments.map((id) => ({
        id,
        name: 'Reference image',
        kind: 'image',
        mime: 'image/png',
        size: 0,
      })),
    );
  };
  const preview = art?.image_url || (selected?.preview ? asset(selected.preview) : '');
  const reference = art?.attachments[0]
    ? '/api/attachments/' + art.attachments[0]
    : selected?.reference
      ? asset(selected.reference)
      : undefined;
  const saveConfig = async () => {
    try {
      setConfig(await api<Config>('/api/images/config', 'POST', config?.settings));
      setShowSettings(false);
    } catch (e) {
      notify(errorText(e));
    }
  };
  return (
    <div className="image-studio">
      <div className="studio-heading">
        <div>
          <div className="eyebrow">
            <Sparkles size={15} /> THE LOCAL IMAGINATION STUDIO
          </div>
          <h1>
            Make the impossible<span>.</span>
          </h1>
          <p>A thought, a sketch, a whole new world. Made on your machine.</p>
        </div>
        <button className="button secondary" onClick={() => setShowSettings(true)}>
          <Settings2 size={16} /> Runtime
        </button>
      </div>
      <div className="studio-toolbar">
        <div className="segmented">
          <button className={tab === 'create' ? 'active' : ''} onClick={() => setTab('create')}>
            Create
          </button>
          <button className={tab === 'gallery' ? 'active' : ''} onClick={() => setTab('gallery')}>
            Your gallery <span>{history.length}</span>
          </button>
        </div>
        <span className="studio-model">
          <span className="status-dot" /> Qwen-Image-2.1-Turbo <Badge>8 STEPS</Badge>
        </span>
      </div>
      {config && config.missing.length > 0 && (
        <div className="studio-notice">
          Image runtime needs setup.{' '}
          <button onClick={() => setShowSettings(true)}>
            Choose the model and runtime files <ArrowUpRight size={14} />
          </button>
        </div>
      )}
      {running && (
        <div ref={progressPanel}>
          <ImageProgress
            stage={stage}
            message={stageMessage}
            completed={step}
            total={totalSteps}
            elapsed={elapsed}
            onStop={onStop}
          />
        </div>
      )}
      {tab === 'create' ? (
        <>
          <div className="studio-workbench">
            <section className="studio-controls">
              <div className="studio-section-title">
                <span className="number-label">01</span>
                <h3>Describe the extraordinary</h3>
              </div>
              <label className="field">
                <span>Your prompt</span>
                <textarea
                  aria-label="Image prompt"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="A place that doesn't exist. Yet."
                  rows={7}
                  disabled={busy}
                />
              </label>
              <div className="studio-section-title">
                <span className="number-label">02</span>
                <h3>Give it a starting point</h3>
                <span className="muted small">Optional</span>
              </div>
              <input
                hidden
                ref={fileInput}
                type="file"
                accept="image/png,image/jpeg,image/webp"
                multiple
                onChange={async (e) => {
                  try {
                    const files = Array.from(e.target.files || []);
                    if (files.length + references.length > 3)
                      throw new Error('Use up to three reference images.');
                    setReferences([
                      ...references,
                      ...(await Promise.all(files.map((f) => attachFile(f, f.name)))),
                    ]);
                  } catch (e) {
                    notify(errorText(e));
                  }
                }}
              />
              <div className="reference-tray">
                {references.map((a) => (
                  <div className="reference-thumb" key={a.id}>
                    <img src={'/api/attachments/' + a.id} alt={a.name} />
                    <button
                      aria-label={'Remove ' + a.name}
                      disabled={busy}
                      onClick={() => setReferences(references.filter((r) => r.id !== a.id))}
                    >
                      <X size={12} />
                    </button>
                  </div>
                ))}
                <button
                  className="reference-add"
                  disabled={busy || references.length >= 3}
                  onClick={() => fileInput.current?.click()}
                >
                  <ImagePlus size={24} />
                  <span>{references.length ? 'Add image' : 'Drop into a new reality'}</span>
                  <small>Choose a reference · PNG, JPEG, WebP</small>
                </button>
              </div>
              <div className="studio-options">
                <label className="field">
                  <span>Canvas</span>
                  <select
                    aria-label="Image canvas"
                    value={size}
                    onChange={(e) => setSize(e.target.value)}
                    disabled={busy}
                  >
                    <option value="512x512">Square · 512</option>
                    <option value="768x768">Square · 768</option>
                    <option value="1024x1024">Square · 1024</option>
                    <option value="768x1024">Portrait · 768 × 1024</option>
                    <option value="1024x768">Landscape · 1024 × 768</option>
                  </select>
                </label>
                <label className="field">
                  <span>
                    Seed{' '}
                    <button
                      className="icon-button"
                      title="Randomize seed"
                      aria-label="Randomize seed"
                      disabled={busy}
                      onClick={() => setSeed(Math.floor(Math.random() * 4294967295))}
                    >
                      <RefreshCw size={12} />
                    </button>
                  </span>
                  <input
                    aria-label="Image seed"
                    type="number"
                    min={0}
                    max={4294967295}
                    value={seed}
                    onChange={(e) => setSeed(Number(e.target.value))}
                    disabled={busy}
                  />
                </label>
              </div>
              <details className="studio-advanced">
                <summary>Generation details</summary>
                <label>
                  <input
                    type="checkbox"
                    checked={capture}
                    onChange={(e) => setCapture(e.target.checked)}
                    disabled={busy}
                  />{' '}
                  Use FP32 graph capture (compatibility mode)
                </label>
                <p>
                  Native 8-step schedule · CFG 1. Image edits use the validated FP32 accumulation
                  setting.
                </p>
              </details>
              <label className="studio-profile-option">
                <input
                  type="checkbox"
                  checked={profile}
                  onChange={(e) => setProfile(e.target.checked)}
                  disabled={busy}
                />
                <span>
                  <strong>Profile generation</strong>
                  <small>Find GPU bottlenecks and save traces. Adds measurement overhead.</small>
                </span>
              </label>
              <button
                className="button primary generate-button"
                disabled={
                  busy ||
                  running ||
                  loadingPreset ||
                  !prompt.trim() ||
                  !config ||
                  !!config.missing.length
                }
                onClick={generate}
              >
                {running ? <Spinner /> : <WandSparkles size={19} />}{' '}
                {references.length ? 'Transform image' : 'Generate image'}{' '}
                <ArrowUpRight size={18} />
              </button>
              <p className="studio-footnote">Local inference · No upload to a cloud service</p>
            </section>
            <section className="studio-canvas">
              <div className="canvas-label">
                <span>
                  {running
                    ? 'CREATION IN PROGRESS'
                    : art
                      ? 'YOUR CREATION'
                      : 'GENERATED WITH QWEN-IMAGE'}
                </span>
                <span>
                  {art ? `${art.width} × ${art.height}` : selected?.category || 'YOUR CANVAS'}
                </span>
              </div>
              <div className={`artwork-frame ${running ? 'generating' : ''}`}>
                {preview && (
                  <img
                    className="artwork-main"
                    src={preview}
                    alt={art ? art.prompt : selected?.title}
                  />
                )}
                {!preview && !running && (
                  <div className="canvas-empty">
                    <Sparkles size={44} />
                    <h2>
                      Every world starts
                      <br />
                      with a few words.
                    </h2>
                  </div>
                )}
                {reference && preview && !running && (
                  <>
                    <img
                      className="artwork-before"
                      src={reference}
                      alt="Original reference"
                      style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}
                    />
                    <div className="compare-divider" style={{ left: `${split}%` }}>
                      <span>↔</span>
                    </div>
                    <input
                      className="compare-slider"
                      aria-label="Before and after comparison"
                      type="range"
                      min="0"
                      max="100"
                      value={split}
                      onChange={(e) => setSplit(Number(e.target.value))}
                    />
                    <span className="before-label">BEFORE</span>
                    <span className="after-label">AFTER</span>
                  </>
                )}
                {running && (
                  <div className="generation-overlay">
                    <div className="generation-orbit">
                      <Sparkles size={32} />
                    </div>
                    <h2>
                      {stage === 'decode' ? 'The finishing touch.' : 'Bringing your idea to life.'}
                    </h2>
                    <p>{stageMessage}</p>
                    <p>{elapsed.toFixed(0)} seconds elapsed · Progress above</p>
                  </div>
                )}
                {preview && !running && (
                  <button
                    className="expand-art"
                    aria-label="Expand image"
                    onClick={() => setFullscreen(true)}
                  >
                    <Expand size={18} />
                  </button>
                )}
              </div>
              <div className="canvas-caption">
                <div>
                  <strong>
                    {art
                      ? 'An original, entirely yours.'
                      : selected?.title || 'Your next original.'}
                  </strong>
                  <p>
                    {art
                      ? `Seed ${art.seed} · ${art.steps} steps · ${ms(art.request_ms)} total`
                      : selected?.subtitle}
                  </p>
                </div>
                {art ? (
                  <a
                    className="button secondary"
                    href={art.image_url}
                    download={'kite-' + art.id + '.png'}
                  >
                    <ArrowDownToLine size={16} /> Save PNG
                  </a>
                ) : (
                  <Badge>REAL MODEL OUTPUT</Badge>
                )}
              </div>
            </section>
          </div>
          {art && <ImagePerformance art={art} />}
          <div className="studio-collection-heading">
            <div>
              <div className="eyebrow">FOUR WAYS TO GET CARRIED AWAY</div>
              <h2>A spark for your next idea.</h2>
            </div>
            <span className="muted">Pick a scene. Make it your own.</span>
          </div>
          <div className="studio-presets">
            {presets.map((p) => (
              <button
                className={`studio-preset ${selected?.id === p.id ? 'selected' : ''}`}
                key={p.id}
                disabled={busy || loadingPreset}
                onClick={() => usePreset(p)}
              >
                <div className="preset-image">
                  {p.preview && <img src={asset(p.preview)} alt={p.title} />}
                  <span>{p.category}</span>
                  <ArrowUpRight size={20} />
                </div>
                <strong>{p.title}</strong>
                <p>{p.subtitle}</p>
              </button>
            ))}
          </div>
        </>
      ) : (
        <div className="art-gallery">
          {!history.length && (
            <div className="empty-state">
              <ImagePlus size={38} />
              <h2>Your first original is waiting.</h2>
              <p>Generate an image to start your collection.</p>
              <button className="button primary" onClick={() => setTab('create')}>
                Start creating
              </button>
            </div>
          )}
          {history.map((a) => (
            <button
              className="art-gallery-item"
              key={a.id}
              onClick={() => reuse(a)}
              disabled={busy}
            >
              <img src={a.image_url} alt={a.prompt} />
              <div>
                <p>{a.prompt}</p>
                <small>
                  {a.width} × {a.height} · seed {a.seed} ·{' '}
                  {new Date(a.created * 1000).toLocaleDateString()}
                </small>
              </div>
            </button>
          ))}
        </div>
      )}
      {fullscreen && preview && (
        <Modal
          title={selected?.title || 'Your creation'}
          onClose={() => setFullscreen(false)}
          className="art-fullscreen"
        >
          <img src={preview} alt={prompt} />
        </Modal>
      )}
      {showSettings && config && (
        <Modal title="Image runtime" onClose={() => setShowSettings(false)}>
          <div className="image-runtime-form">
            <p>
              Images are saved automatically to <code>{config.output_directory}</code>.
            </p>
            <p>
              The denoiser uses source-built ORT WebGPU / D3D12. The text encoder and VAE use local
              PyTorch / CUDA. Model files are referenced in place.
            </p>
            {(['source', 'package', 'library', 'python'] as const).map((key) => (
              <PathInput
                key={key}
                label={
                  {
                    source: 'Official Qwen source package',
                    package: 'ONNX image package',
                    library: 'D3D12 WebGPU DLL',
                    python: 'Inference Python',
                  }[key]
                }
                kind={key === 'library' ? 'runtime' : key === 'python' ? 'python' : 'folder'}
                value={config.settings[key]}
                onChange={(v) =>
                  setConfig({ ...config, settings: { ...config.settings, [key]: v } })
                }
              />
            ))}
            <label className="field">
              <span>Dependency directories · one per line</span>
              <textarea
                value={config.settings.dependencies.join('\n')}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    settings: {
                      ...config.settings,
                      dependencies: e.target.value.split('\n').filter(Boolean),
                    },
                  })
                }
              />
            </label>
            {config.missing.length > 0 && (
              <p className="error-text">Missing: {config.missing.join(', ')}</p>
            )}
            <button className="button primary" disabled={busy} onClick={saveConfig}>
              Save runtime
            </button>
          </div>
        </Modal>
      )}
    </div>
  );
}
