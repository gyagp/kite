import { useState, useRef, useEffect } from 'react';
import {
  ArrowDownToLine,
  ArrowRight,
  Braces,
  Check,
  ChevronDown,
  CircleHelp,
  Layers3,
  Plus,
  Search,
  SlidersHorizontal,
  Sparkles,
  Upload,
  X,
  Zap,
} from 'lucide-react';
import { api, bytes, download, errorText } from './api';
import { Badge, CopyButton, ModelSelect, Spinner } from './components';
import { ExampleGallery, exampleAsset } from './ExampleGallery';
import { EmbeddingAttach, MediaPreview } from './EmbeddingMedia';
import { EmbeddingStats } from './EmbeddingStats';
import type {
  Attachment,
  EmbeddingExample,
  EmbeddingInput,
  Execute,
  Metrics,
  Model,
  Status,
} from './types';

const examples = [
  {
    query: 'What causes the northern lights?',
    docs: [
      'The aurora forms when charged particles from the Sun collide with gases in Earth’s atmosphere, creating curtains of colored light.',
      'Bioluminescent plankton produce a blue glow in ocean water through a chemical reaction inside their cells.',
      'A rainbow appears when sunlight is refracted and reflected inside water droplets suspended in the air.',
    ],
  },
  {
    query: 'How can I make a small language model run faster?',
    docs: [
      'Quantization reduces model weight precision and memory traffic, which can improve inference speed on supported hardware.',
      'A larger context window lets a language model process more input tokens in a single request.',
      'Keeping inference on the GPU and fusing operations can reduce dispatch overhead and data transfers.',
    ],
  },
  {
    query: '哪里可以看到北极光？',
    docs: [
      'Northern Norway, Iceland, and northern Canada are popular destinations for viewing the aurora.',
      'The Great Barrier Reef in Australia is famous for its marine biodiversity.',
      '芬兰的拉普兰地区是观赏北极光的热门地点，冬季的晴朗夜晚尤其适合。',
    ],
  },
];
type Mode = 'search' | 'compare' | 'raw';
const inputOf = (text = '', attachments: Attachment[] = [], label = ''): EmbeddingInput => ({
  id: crypto.randomUUID(),
  text,
  attachments,
  label,
});
const describe = (sample: EmbeddingInput) =>
  sample.label ||
  [sample.text, ...sample.attachments.map((a) => a.name)].filter(Boolean).join(' · ');
export function Embeddings({
  models,
  preferred,
  execute,
  busy,
  status,
  onModels,
  notify,
}: {
  models: Model[];
  preferred: string;
  execute: Execute;
  busy: boolean;
  status: Status;
  onModels: () => void;
  notify: (s: string) => void;
}) {
  const [modelId, setModelId] = useState(preferred || models.find((m) => m.available)?.id || '');
  const [mode, setMode] = useState<Mode>('search');
  const [query, setQuery] = useState(examples[0].query);
  const [docs, setDocs] = useState<EmbeddingInput[]>(examples[0].docs.map((t) => inputOf(t)));
  const [queryMedia, setQueryMedia] = useState<Attachment[]>([]);
  const [resultSamples, setResultSamples] = useState<EmbeddingInput[]>([]);
  const [uploading, setUploading] = useState(false);
  const [gallery, setGallery] = useState<EmbeddingExample[] | null>(null);
  const [chosenExample, setChosenExample] = useState<EmbeddingExample | null>(null);
  const [loadingExample, setLoadingExample] = useState('');
  const [dimensions, setDimensions] = useState(768);
  const [task, setTask] = useState('SentenceSimilarity');
  const [result, setResult] = useState<Metrics | null>(null);
  const [resultTexts, setResultTexts] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [profile, setProfile] = useState(false);
  const [view, setView] = useState<'visual' | 'json'>('visual');
  const [example, setExample] = useState(0);
  const [selectedVector, setSelectedVector] = useState(0);
  const fileInput = useRef<HTMLInputElement>(null);
  const resultPanel = useRef<HTMLElement>(null);
  useEffect(() => {
    if (result) resultPanel.current?.scrollIntoView({ block: 'nearest', behavior: 'instant' });
  }, [result]);
  const model = models.find((m) => m.id === modelId) || models[0];
  const multimodal = (model?.modalities.length || 0) > 1;
  const locked = busy || uploading;
  const openExamples = async () => {
    try {
      setGallery(await api<EmbeddingExample[]>('/api/examples'));
    } catch (e) {
      notify(errorText(e));
    }
  };
  const chooseExample = async (item: EmbeddingExample, autoRun = false, queryOverride?: string) => {
    const compatible = models.find(
      (m) => m.available && item.modalities.every((k) => m.modalities.includes(k)),
    );
    if (!compatible) {
      notify('Import the EmbeddingGemma 2 multimodal package to run these examples.');
      return;
    }
    setLoadingExample(item.id);
    try {
      const response = await api<EmbeddingExample>(`/api/examples/${item.id}`, 'POST', {});
      const loaded = { ...response, query: queryOverride ?? response.query };
      setModelId(compatible.id);
      setMode(loaded.mode);
      setQuery(loaded.query);
      setQueryMedia(loaded.query_attachments || []);
      const prepared = (loaded.samples || []).map((s) => inputOf(s.text, s.attachments, s.label));
      setDocs(prepared);
      setTask(loaded.task || 'SentenceSimilarity');
      setResult(null);
      setChosenExample(loaded);
      setGallery(null);
      if (autoRun)
        await performRun(
          prepared,
          compatible.id,
          loaded.mode,
          loaded.query,
          loaded.query_attachments || [],
          loaded.task || 'SentenceSimilarity',
        );
    } catch (e) {
      notify(errorText(e));
    } finally {
      setLoadingExample('');
    }
  };
  const changeMode = (next: Mode) => {
    setMode(next);
    setResult(null);
  };
  const performRun = async (
    inputs: EmbeddingInput[],
    selectedModel: string,
    selectedMode: Mode,
    searchQuery: string,
    searchMedia: Attachment[],
    selectedTask: string,
  ) => {
    setRunning(true);
    setResult(null);
    setSelectedVector(0);
    const samples = inputs
      .filter((s) => s.text.trim() || s.attachments.length)
      .map((s) => ({ ...s, text: s.text.trim() }));
    try {
      const output = await execute({
        kind: selectedMode === 'search' ? 'search' : 'embed',
        model: selectedModel,
        dimensions,
        profile,
        ...(selectedMode === 'search'
          ? { query: searchQuery, query_attachments: searchMedia, documents: samples }
          : { samples, task: selectedMode === 'compare' ? 'SentenceSimilarity' : selectedTask }),
      });
      if (output) {
        setResult(output);
        setResultTexts(samples.map(describe));
        setResultSamples(samples);
      }
    } catch {
      /* Global error banner keeps the inputs intact. */
    } finally {
      setRunning(false);
    }
  };
  const run = () => performRun(docs, model?.id || '', mode, query, queryMedia, task);
  const importText = async (files: FileList | null) => {
    if (!files) return;
    try {
      const imported: string[] = [];
      for (const file of Array.from(files)) {
        if (file.size > 1024 * 1024) throw new Error('Text files must be smaller than 1 MB.');
        if (!/\.(txt|md|csv|json|log)$/i.test(file.name))
          throw new Error('Choose UTF-8 .txt, .md, .csv, .json, or .log files.');
        const text = await file.text();
        imported.push(
          ...text
            .split(/\r?\n\s*\r?\n/)
            .map((t) => t.trim())
            .filter(Boolean),
        );
      }
      setDocs((prev) =>
        [
          ...prev.filter((s) => s.text.trim() || s.attachments.length),
          ...imported.map((t) => inputOf(t)),
        ].slice(0, 128),
      );
      if (imported.length > 128)
        notify('Imported the first 128 paragraphs. Split the document to process the remainder.');
    } catch (e) {
      notify(errorText(e));
    }
    if (fileInput.current) fileInput.current.value = '';
  };
  const fillExample = () => {
    const next = (example + 1) % examples.length;
    setExample(next);
    setQuery(examples[next].query);
    setDocs(examples[next].docs.map((t) => inputOf(t)));
    setQueryMedia([]);
    setChosenExample(null);
    setResult(null);
  };
  return (
    <div className="workspace-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">
            <span className="tiny-line" /> THE EMBEDDING WORKSPACE
          </div>
          <h1>
            Meaning, beyond keywords<span className="accent">.</span>
          </h1>
          <p>
            {multimodal
              ? 'Connect words, images, sounds, and video in the same embedding space.'
              : 'Turn text into understanding. Explore similarity and find what belongs together.'}
          </p>
        </div>
        <div className="button-row">
          <button className="button primary" onClick={openExamples} disabled={locked}>
            <Sparkles size={15} />
            Multimodal examples
          </button>
          <button className="button subtle" onClick={fillExample} disabled={locked}>
            <Sparkles size={15} />
            Text example
          </button>
        </div>
      </div>
      {chosenExample && (
        <div className="example-loaded active-experience">
          {chosenExample.cover && <img src={exampleAsset(chosenExample.cover)} alt="" />}
          <div>
            <span className="showcase-eyebrow">{chosenExample.eyebrow || 'EXPERIENCE'}</span>
            <strong>{chosenExample.title}</strong>
            <p>{chosenExample.description}</p>
          </div>
          <button className="button subtle" disabled={locked} onClick={openExamples}>
            Try another
          </button>
          <button
            className="icon-button"
            aria-label="Dismiss example description"
            onClick={() => setChosenExample(null)}
          >
            <X size={15} />
          </button>
        </div>
      )}
      <div className="workbench-layout">
        <section className="workbench-main">
          <div className="tabs" role="tablist" aria-label="Embedding tools">
            {(
              [
                { id: 'search', label: 'Semantic search', icon: Search },
                {
                  id: 'compare',
                  label: multimodal ? 'Compare inputs' : 'Compare texts',
                  icon: Layers3,
                },
                { id: 'raw', label: 'Raw embeddings', icon: Braces },
              ] as const
            ).map((t) => (
              <button
                key={t.id}
                role="tab"
                aria-selected={mode === t.id}
                disabled={busy}
                className={mode === t.id ? 'active' : ''}
                onClick={() => changeMode(t.id)}
              >
                <t.icon size={16} />
                {t.label}
              </button>
            ))}
          </div>
          <section className="panel input-panel">
            <div className="section-title">
              <div>
                <span className="step">01</span>
                <h2>
                  {mode === 'search'
                    ? 'What are you looking for?'
                    : 'Give your ideas a little space'}
                </h2>
              </div>
              <Badge>{multimodal ? 'MULTIMODAL INPUT' : 'TEXT INPUT'}</Badge>
            </div>
            {mode === 'search' && (
              <>
                <label className="query-field">
                  <Search size={18} />
                  <textarea
                    aria-label="Search query"
                    rows={2}
                    placeholder="Ask a question or describe what you want to find…"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    disabled={locked}
                  />
                </label>
                <div className="query-media">
                  <MediaPreview
                    attachments={queryMedia}
                    disabled={locked}
                    onRemove={(id) => setQueryMedia(queryMedia.filter((a) => a.id !== id))}
                  />
                  <EmbeddingAttach
                    label="Attach query media"
                    modalities={model?.modalities || []}
                    attachments={queryMedia}
                    onChange={setQueryMedia}
                    disabled={locked}
                    onUploading={setUploading}
                    notify={notify}
                  />
                </div>
              </>
            )}
            <div className="document-heading">
              <label>
                {mode === 'search' ? 'Inputs to search' : 'Inputs to embed'}
                <span>{docs.length}</span>
              </label>
              <div>
                <input
                  ref={fileInput}
                  className="hidden"
                  type="file"
                  accept=".txt,.md,.csv,.json,.log"
                  multiple
                  onChange={(e) => importText(e.target.files)}
                />
                <button
                  className="text-button"
                  onClick={() => fileInput.current?.click()}
                  disabled={busy}
                >
                  <Upload size={13} />
                  Import text
                </button>
                <button
                  className="text-button"
                  disabled={busy}
                  onClick={() => {
                    setDocs([inputOf()]);
                    setResult(null);
                  }}
                >
                  Clear
                </button>
              </div>
            </div>
            <div className="document-list">
              {docs.map((sample, i) => (
                <div className="embedding-input-card" key={sample.id}>
                  {sample.label && <div className="embedding-input-label">{sample.label}</div>}
                  <div className="document-input">
                    <span className="doc-index">{String(i + 1).padStart(2, '0')}</span>
                    <textarea
                      aria-label={`Document ${i + 1}`}
                      value={sample.text}
                      onChange={(e) =>
                        setDocs((prev) =>
                          prev.map((s) =>
                            s.id === sample.id ? { ...s, text: e.target.value } : s,
                          ),
                        )
                      }
                      disabled={locked}
                      placeholder={
                        multimodal
                          ? 'Add text, attach media, or combine them in one embedding…'
                          : 'Add a sentence, a paragraph, a thought…'
                      }
                      rows={2}
                    />
                    <button
                      title="Remove document"
                      aria-label={`Remove document ${i + 1}`}
                      className="icon-button"
                      disabled={locked || docs.length <= 1}
                      onClick={() => setDocs(docs.filter((_, n) => n !== i))}
                    >
                      <X size={13} />
                    </button>
                  </div>
                  <MediaPreview
                    attachments={sample.attachments}
                    disabled={locked}
                    onRemove={(id) =>
                      setDocs((prev) =>
                        prev.map((s) =>
                          s.id === sample.id
                            ? { ...s, attachments: s.attachments.filter((a) => a.id !== id) }
                            : s,
                        ),
                      )
                    }
                  />
                  <div className="embedding-input-footer">
                    <EmbeddingAttach
                      label={`Attach media to input ${i + 1}`}
                      modalities={model?.modalities || []}
                      attachments={sample.attachments}
                      onChange={(attachments) =>
                        setDocs((prev) =>
                          prev.map((s) => (s.id === sample.id ? { ...s, attachments } : s)),
                        )
                      }
                      disabled={locked}
                      onUploading={setUploading}
                      notify={notify}
                    />
                    <span>
                      {sample.attachments.length
                        ? 'Text + attached files → one embedding'
                        : multimodal
                          ? 'Text, image, audio, video'
                          : 'Text only'}
                    </span>
                  </div>
                </div>
              ))}
            </div>
            <button
              className="add-document"
              onClick={() => setDocs([...docs, inputOf()])}
              disabled={locked || docs.length >= 128}
            >
              <Plus size={14} />
              Add another input
            </button>
            <div className="input-footer">
              <span>
                <ShieldIcon />
                Files and inference stay on your computer.
              </span>
              <button
                className="button primary"
                disabled={
                  locked ||
                  !model?.available ||
                  !docs.some((s) => s.text.trim() || s.attachments.length) ||
                  (mode === 'search' && !query.trim() && !queryMedia.length) ||
                  docs.some((s) =>
                    s.attachments.some((a) => !model?.modalities.includes(a.kind)),
                  ) ||
                  queryMedia.some((a) => !model?.modalities.includes(a.kind))
                }
                onClick={run}
              >
                {running ? <Spinner /> : <Zap size={15} />}{' '}
                {running
                  ? 'Running…'
                  : mode === 'search'
                    ? 'Find matches'
                    : mode === 'compare'
                      ? multimodal
                        ? 'Compare inputs'
                        : 'Compare texts'
                      : 'Generate embeddings'}
                {!running && <ArrowRight size={15} />}
              </button>
            </div>
          </section>
          <section ref={resultPanel} className="panel result-panel">
            <div className="section-title">
              <div>
                <span className="step">02</span>
                <h2>
                  {mode === 'search'
                    ? 'The closest connections'
                    : mode === 'compare'
                      ? 'A map of meaning'
                      : 'Your embedding vectors'}
                </h2>
              </div>
              {result && (
                <div className="result-actions">
                  <button
                    className={`icon-button ${view === 'json' ? 'active' : ''}`}
                    title="Toggle JSON view"
                    aria-label="Toggle JSON view"
                    onClick={() => setView(view === 'json' ? 'visual' : 'json')}
                  >
                    <Braces size={15} />
                  </button>
                  <button
                    className="icon-button"
                    title="Export results as JSON"
                    aria-label="Export results as JSON"
                    onClick={() =>
                      download('kite-embeddings.json', {
                        ...result,
                        inputs: resultSamples,
                        input_metrics: result.inputs,
                      })
                    }
                  >
                    <ArrowDownToLine size={15} />
                  </button>
                  <CopyButton text={JSON.stringify(result)} />
                </div>
              )}
            </div>
            {result && <EmbeddingStats metrics={result} />}
            {!result ? (
              <div className="result-empty">
                <div className={`embedding-art ${running ? 'animated' : ''}`}>
                  <span />
                  <span />
                  <span />
                  <span />
                  <span />
                  <span />
                  <span />
                  <span />
                  <span />
                </div>
                <h3>
                  {running
                    ? 'Finding the meaning in your inputs…'
                    : 'A new perspective is one run away'}
                </h3>
                <p>
                  {running
                    ? 'The first run loads the model and prepares GPU shaders.'
                    : mode === 'search'
                      ? 'Run a search to rank your documents by semantic similarity.'
                      : 'Run the model to see actual vectors and cosine similarities.'}
                </p>
              </div>
            ) : view === 'json' ? (
              <pre className="json-result">{JSON.stringify(result, null, 2)}</pre>
            ) : (
              <>
                {result.ranking && (
                  <div className="rankings">
                    {result.ranking.map((r, i) => (
                      <div className={`ranking ${i === 0 ? 'best' : ''}`} key={r.index}>
                        <div className="rank-number">{String(i + 1).padStart(2, '0')}</div>
                        <div className="ranking-content">
                          <div className="ranking-label">
                            {r.label || `Input ${r.index + 1}`}
                            {i === 0 && (
                              <Badge tone="blue">
                                <Check size={10} />
                                CLOSEST MATCH
                              </Badge>
                            )}
                          </div>
                          <p>{r.text}</p>
                          <MediaPreview attachments={r.attachments || []} compact />
                          <div className="score-track">
                            <div style={{ width: `${Math.max(0, r.score) * 100}%` }} />
                          </div>
                        </div>
                        <div className="score">
                          {r.score.toFixed(3)}
                          <span>cosine</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                {result.vectors && mode === 'compare' && (
                  <Similarity vectors={result.vectors} texts={resultTexts} />
                )}
                {result.vectors && mode === 'raw' && (
                  <div className="vector-result">
                    <div className="vector-toolbar">
                      <label>
                        Input{' '}
                        <select
                          aria-label="Vector to inspect"
                          value={selectedVector}
                          onChange={(e) => setSelectedVector(+e.target.value)}
                        >
                          {result.vectors.map((_, i) => (
                            <option key={i} value={i}>
                              {i + 1}
                            </option>
                          ))}
                        </select>
                      </label>
                      <Badge>{dimensions} DIMENSIONS · L2 NORMALIZED</Badge>
                    </div>
                    <p className="muted">{resultTexts[selectedVector]}</p>
                    <MediaPreview
                      attachments={resultSamples[selectedVector]?.attachments || []}
                      compact
                    />
                    <div className="vector-bars" aria-label="First 128 vector dimensions">
                      {result.vectors[selectedVector].slice(0, 128).map((v, i) => (
                        <span
                          key={i}
                          title={`Dimension ${i}: ${v.toFixed(6)}`}
                          style={{
                            height: `${Math.min(100, Math.abs(v) * 500 + 2)}%`,
                            background: v >= 0 ? 'var(--accent)' : '#938ec8',
                            alignSelf: v >= 0 ? 'flex-end' : 'flex-start',
                          }}
                        />
                      ))}
                    </div>
                    <div className="vector-caption">
                      First 128 dimensions <span>Export JSON for all values</span>
                    </div>
                  </div>
                )}
              </>
            )}
          </section>
          <p className="workbench-footnote">
            <CircleHelp size={13} />
            Cosine similarity measures related meaning. It is not a confidence score or probability.
          </p>
        </section>
        <aside className="inspector">
          <div className="inspector-title">
            <SlidersHorizontal size={16} /> Run configuration
          </div>
          <section className="panel model-inspector">
            <div className="model-glyph">
              Eg<span>2</span>
            </div>
            <h3>{model?.name || 'No embedding model'}</h3>
            <p>
              {model
                ? 'A smaller model. A world of meaning.'
                : 'Import an embedding model to begin.'}
            </p>
            {model && (
              <div className="tag-row">
                <Badge tone="purple">EMBEDDING</Badge>
                <Badge>{model.precision.toUpperCase()}</Badge>
              </div>
            )}
            <div className="model-selector">
              <label>Model</label>
              <ModelSelect
                models={models}
                value={model?.id || ''}
                onChange={(id) => {
                  const next = models.find((m) => m.id === id);
                  if (
                    [...queryMedia, ...docs.flatMap((s) => s.attachments)].some(
                      (a) => !next?.modalities.includes(a.kind),
                    )
                  ) {
                    notify('Remove media inputs before switching to a text-only export.');
                    return;
                  }
                  setModelId(id);
                  setResult(null);
                }}
                disabled={locked}
              />
            </div>
            {model && (
              <>
                <dl className="model-details">
                  <div>
                    <dt>Model size</dt>
                    <dd>{bytes(model.size)}</dd>
                  </div>
                  <div>
                    <dt>Context window</dt>
                    <dd>{model.context.toLocaleString()} tokens</dd>
                  </div>
                  <div>
                    <dt>Execution provider</dt>
                    <dd>Native WebGPU</dd>
                  </div>
                  <div>
                    <dt>Input modalities</dt>
                    <dd>{model.modalities.join(' · ')}</dd>
                  </div>
                </dl>
                {model.validation?.passed && (
                  <div className="validation-note">
                    <Check size={13} />
                    Export validation · {model.validation.cases} cases
                  </div>
                )}
                <button
                  className="text-button full-width"
                  disabled={busy}
                  onClick={() => execute({ kind: 'load', model: model.id }).catch(() => {})}
                >
                  {status.loaded?.model === model.id ? (
                    <>
                      <Check size={14} />
                      Loaded on GPU
                    </>
                  ) : (
                    <>
                      Load model
                      <ArrowRight size={14} />
                    </>
                  )}
                </button>
              </>
            )}
            <button className="text-button full-width muted" onClick={onModels}>
              Manage models
              <ArrowRight size={13} />
            </button>
          </section>
          <section className="panel parameters">
            <label className="field">
              <span>
                Output dimensions<Badge>MRL</Badge>
              </span>
              <select
                value={dimensions}
                onChange={(e) => setDimensions(+e.target.value)}
                disabled={busy}
              >
                {[768, 512, 256, 128].map((n) => (
                  <option key={n} value={n}>
                    {n}
                    {n === 768 ? ' · Full precision of meaning' : ''}
                  </option>
                ))}
              </select>
            </label>
            <p>Smaller vectors use less storage. Truncated vectors are normalized again.</p>
            <label className="studio-profile-option">
              <input
                type="checkbox"
                checked={profile}
                onChange={(e) => setProfile(e.target.checked)}
                disabled={busy}
              />
              <span>
                <strong>Profile embeddings</strong>
                <small>Capture a fresh model load and operator trace. Adds overhead.</small>
              </span>
            </label>
            {mode === 'raw' && (
              <label className="field">
                <span>Task prefix</span>
                <select value={task} onChange={(e) => setTask(e.target.value)} disabled={busy}>
                  {model?.tasks.map((t) => (
                    <option key={t}>{t}</option>
                  ))}
                </select>
              </label>
            )}
            <div className="automatic-setting">
              <span>Similarity metric</span>
              <span>
                Cosine <ChevronDown size={12} />
              </span>
            </div>
            <div className="automatic-setting">
              <span>Normalization</span>
              <span className="accent-text">
                <Check size={12} /> L2
              </span>
            </div>
          </section>
          <div className="tip-card">
            <Layers3 size={18} />
            <h4>
              {multimodal ? 'One space for every modality' : 'Built for your exported models'}
            </h4>
            <p>
              {multimodal
                ? 'Each input produces one vector, even when it combines text, images, audio, and video. Video samples visual frames at 1 fps, up to 32; attach audio separately to include sound.'
                : 'This is the smaller text-only package. Select EmbeddingGemma 2 · Multimodal for images, audio, and video.'}
            </p>
            <button className="text-button" onClick={onModels}>
              Explore your library <ArrowRight size={13} />
            </button>
          </div>
          {multimodal && (
            <div className="tip-card">
              <h4>Media order</h4>
              <p>
                Attachments are placed after your text by default. To interleave them, use{' '}
                <code>{'<|image|>'}</code>, <code>{'<|audio|>'}</code>, and{' '}
                <code>{'<|video|>'}</code> once per file. Media-only inputs use no task prefix.
              </p>
            </div>
          )}
        </aside>
      </div>
      {gallery && (
        <ExampleGallery
          items={gallery}
          loading={loadingExample}
          onChoose={chooseExample}
          onClose={() => setGallery(null)}
        />
      )}
    </div>
  );
}
function ShieldIcon() {
  return <span className="status-dot" />;
}
function Similarity({ vectors, texts }: { vectors: number[][]; texts: string[] }) {
  const shown = vectors.slice(0, 16);
  return (
    <div className="similarity">
      <p className="muted small">
        Pairwise cosine similarity · hover a cell for its texts
        {vectors.length > 16 ? ' · showing first 16 texts' : ''}
      </p>
      <div
        className="matrix"
        style={{ gridTemplateColumns: `30px repeat(${shown.length}, minmax(40px, 1fr))` }}
      >
        <span />
        {shown.map((_, i) => (
          <strong key={i}>{i + 1}</strong>
        ))}
        {shown.map((a, i) => (
          <div className="matrix-row" key={i}>
            <strong>{i + 1}</strong>
            {shown.map((b, j) => {
              const score = a.reduce((s, v, k) => s + v * b[k], 0);
              return (
                <div
                  key={j}
                  title={`${texts[i]} ↔ ${texts[j]}: ${score.toFixed(5)}`}
                  style={{ background: `rgba(159,198,255,${Math.max(0.04, score * 0.55)})` }}
                >
                  {score.toFixed(2)}
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <div className="matrix-legend">
        <span>Less related</span>
        <span className="legend-ramp" />
        <span>More related</span>
      </div>
      {texts.slice(0, 16).map((t, i) => (
        <p className="text-key" key={i}>
          <span>{i + 1}</span>
          {t}
        </p>
      ))}
    </div>
  );
}
