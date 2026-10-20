import { useEffect, useMemo, useState } from 'react';
import {
  ArrowDownToLine,
  ArrowRight,
  Code2,
  FlaskConical,
  Play,
  Plus,
  Search,
  Terminal,
} from 'lucide-react';
import { api, download, errorText } from './api';
import { Badge, Modal, Spinner } from './components';
import type { Execute, Run } from './types';
import './lab.css';

type Field = {
  name: string;
  flags: string[];
  required: boolean;
  default: unknown;
  choices?: (string | number)[];
  help?: string;
  type: string;
  action: string;
  nargs?: string | number;
};
type Catalog = {
  commands: Record<string, string>;
  schemas: Record<string, { description: string; fields: Field[] }>;
  engine: string;
  resources: string;
  evidence_root: string;
  recipes?: string[];
};
type EvidenceModel = { id: string; name: string; path: string };
type Observation = {
  id: string;
  model_id: string;
  model_name: string;
  prompt: string;
  machine: string;
  notes: string;
  source: string;
  task: string;
  latency_ms?: number;
  decode_tps?: number;
  ttft_ms?: number;
  embeddings_per_second?: number;
  images_per_second?: number;
};
type Wish = {
  id?: string;
  title: string;
  description: string;
  status: string;
  priority: string;
  kind?: string;
};
type ToolResult = { operation: string; directory: string; log_url: string; files: string[] };
const groups: Record<string, string[]> = {
  'Export & optimize': ['build', 'plan', 'convert', 'optimize', 'quantize'],
  'Inspect & validate': ['inspect', 'analyze', 'validate-cpu', 'contrib', 'capabilities'],
  Measure: ['benchmark', 'profile', 'embedding-benchmark'],
  Embeddings: ['embedding-export', 'embedding-validate', 'embed'],
  'Image pipeline': [
    'image-export',
    'image-prepare',
    'image-generate',
    'image-optimize',
    'image-report',
  ],
};
const title = (value: string) => value.replaceAll('_', ' ').replace(/^./, (s) => s.toUpperCase());

export function ModelLab({
  execute,
  busy,
  notify,
}: {
  execute: Execute;
  busy: boolean;
  notify: (s: string) => void;
}) {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [tab, setTab] = useState('tools');
  const [operation, setOperation] = useState('inspect');
  const [values, setValues] = useState<Record<string, string | boolean>>({});
  const [raw, setRaw] = useState(false);
  const [rawText, setRawText] = useState('[]');
  const [advanced, setAdvanced] = useState(false);
  const [log, setLog] = useState('');
  const [result, setResult] = useState<ToolResult | null>(null);
  const [running, setRunning] = useState(false);
  const [models, setModels] = useState<EvidenceModel[]>([]);
  const [observations, setObservations] = useState<Observation[]>([]);
  const [wishes, setWishes] = useState<Wish[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [filter, setFilter] = useState('');
  const [loading, setLoading] = useState(false);
  const [wish, setWish] = useState<Wish | null>(null);
  const [observation, setObservation] = useState<Record<string, string> | null>(null);
  const [modelDraft, setModelDraft] = useState<{ name: string; path: string } | null>(null);
  const [inspect, setInspect] = useState<unknown>(null);
  const [compareA, setCompareA] = useState('');
  const [compareB, setCompareB] = useState('');
  useEffect(() => {
    api<Catalog>('/api/lab/catalog')
      .then(setCatalog)
      .catch((e) => notify(errorText(e)));
  }, [notify]);
  const refreshEvidence = async (target = tab) => {
    setLoading(true);
    try {
      if (target === 'runs')
        setRuns((await api<Run[]>('/api/runs')).filter((r) => r.kind === 'tool'));
      else if (target !== 'tools') {
        setModels(await api<EvidenceModel[]>('/api/evidence/models'));
        if (target === 'performance')
          setObservations(await api<Observation[]>('/api/evidence/performance'));
        if (target === 'wishlist') setWishes(await api<Wish[]>('/api/evidence/wishlist'));
      }
    } catch (e) {
      notify(errorText(e));
    } finally {
      setLoading(false);
    }
  };
  const changeTab = (value: string) => {
    setTab(value);
    setFilter('');
    void refreshEvidence(value);
  };
  const schema = catalog?.schemas[operation];
  const args = useMemo(() => {
    const out: string[] = [];
    for (const field of schema?.fields || []) {
      const value = values[field.name];
      if (value == null || value === '' || value === false) continue;
      const flag = field.flags.at(-1);
      if (field.action.includes('True') || field.action.includes('False')) {
        if (value && flag) out.push(flag);
        continue;
      }
      if (field.action.includes('Append')) {
        for (const item of String(value).split('\n').filter(Boolean)) {
          if (flag) out.push(flag);
          out.push(item);
        }
        continue;
      }
      if (flag) out.push(flag);
      if (field.nargs === '+' || field.nargs === '*')
        out.push(...String(value).split('\n').filter(Boolean));
      else out.push(String(value));
    }
    return out;
  }, [schema, values]);
  const selectOperation = (op: string) => {
    setOperation(op);
    setValues({});
    setRawText('[]');
    setResult(null);
    setLog('');
    setAdvanced(false);
    setRaw(false);
  };
  const run = async (help = false) => {
    setRunning(true);
    setLog('');
    setResult(null);
    try {
      const arguments_ = help ? ['--help'] : raw ? JSON.parse(rawText) : args;
      const r = await execute({ kind: 'tool', operation, arguments: arguments_ }, (e) => {
        if (e.type === 'log') setLog((previous) => (previous + (e.text || '')).slice(-60000));
      });
      if (r) setResult(r as unknown as ToolResult);
    } catch (e) {
      notify(errorText(e));
    } finally {
      setRunning(false);
    }
  };
  const save = async (path: string, data: unknown, done: () => void) => {
    try {
      await api(path, 'POST', data);
      done();
      await refreshEvidence();
    } catch (e) {
      notify(errorText(e));
    }
  };
  const fields = (schema?.fields || []).filter(
    (f) =>
      advanced ||
      !f.flags.length ||
      f.required ||
      ['model', 'config', 'source', 'backend', 'precision', 'recipe', 'output'].includes(f.name),
  );
  const filtered = observations.filter((r) =>
    (r.model_name + ' ' + r.prompt + ' ' + r.machine + ' ' + r.task)
      .toLowerCase()
      .includes(filter.toLowerCase()),
  );
  return (
    <div className="model-lab">
      <div className="page-heading">
        <div>
          <div className="eyebrow">
            <FlaskConical size={14} /> FROM CHECKPOINT TO LOCAL INTELLIGENCE
          </div>
          <h1>
            Model Lab<span className="accent">.</span>
          </h1>
          <p>Build, optimize, and measure your local models.</p>
        </div>
        <Badge>21 TOOLS</Badge>
      </div>
      <div className="lab-tabs">
        {[
          ['tools', 'Tools'],
          ['performance', 'Performance'],
          ['models', 'Graph library'],
          ['wishlist', 'Conversion wishlist'],
          ['runs', 'Tool runs'],
        ].map(([id, label]) => (
          <button key={id} className={tab === id ? 'active' : ''} onClick={() => changeTab(id)}>
            {label}
          </button>
        ))}
      </div>
      {loading && (
        <div className="lab-loading">
          <Spinner /> Loading evidence…
        </div>
      )}
      {tab === 'tools' && (
        <div className="lab-tools-layout">
          <aside className="lab-tool-list">
            {Object.entries(groups).map(([group, commands]) => (
              <div key={group}>
                <h3>{group}</h3>
                {commands.map((op) => (
                  <button
                    className={operation === op ? 'selected' : ''}
                    onClick={() => selectOperation(op)}
                    disabled={busy}
                    key={op}
                  >
                    <span>{catalog?.commands[op] || op}</span>
                    <small>{op}</small>
                  </button>
                ))}
              </div>
            ))}
          </aside>
          <section className="lab-operation">
            <div className="lab-operation-heading">
              <div>
                <div className="eyebrow">{operation}</div>
                <h2>{catalog?.commands[operation]}</h2>
              </div>
              <Code2 size={24} />
            </div>
            <p className="lab-description">{schema?.description}</p>
            {['plan', 'convert'].includes(operation) && (
              <label className="field">
                <span>Included recipe</span>
                <select
                  aria-label="Included recipe"
                  value={String(values.recipe || '')}
                  onChange={(e) => setValues({ ...values, recipe: e.target.value })}
                >
                  <option value="">Choose a recipe or enter a path below</option>
                  {catalog?.recipes?.map((p) => (
                    <option value={p} key={p}>
                      {p.split(/[\\/]/).at(-1)}
                    </option>
                  ))}
                </select>
              </label>
            )}
            <div className="lab-mode-row">
              <label>
                <input
                  type="checkbox"
                  checked={raw}
                  onChange={(e) => {
                    setRaw(e.target.checked);
                    setRawText(JSON.stringify(args, null, 2));
                  }}
                />{' '}
                Argument JSON
              </label>
              <button className="text-button" disabled={busy} onClick={() => run(true)}>
                Show command help
              </button>
            </div>
            {raw ? (
              <label className="field">
                <span>Arguments · JSON array of strings</span>
                <textarea
                  className="lab-argument-json"
                  value={rawText}
                  onChange={(e) => setRawText(e.target.value)}
                  spellCheck={false}
                />
              </label>
            ) : (
              <>
                <div className="lab-fields">
                  {fields.map((f) => (
                    <label
                      className={`field ${f.action.includes('True') || f.action.includes('False') ? 'lab-checkbox' : ''}`}
                      key={f.name}
                    >
                      <span>
                        {title(f.name)}
                        {f.required ? ' *' : ''}
                      </span>
                      {f.action.includes('True') || f.action.includes('False') ? (
                        <input
                          type="checkbox"
                          checked={!!values[f.name]}
                          onChange={(e) => setValues({ ...values, [f.name]: e.target.checked })}
                        />
                      ) : f.choices ? (
                        <select
                          value={String(values[f.name] || '')}
                          onChange={(e) => setValues({ ...values, [f.name]: e.target.value })}
                        >
                          <option value="">
                            {f.default == null ? 'Choose…' : `Default: ${f.default}`}
                          </option>
                          {f.choices.map((v) => (
                            <option value={v} key={v}>
                              {v}
                            </option>
                          ))}
                        </select>
                      ) : f.action.includes('Append') || f.nargs === '+' || f.nargs === '*' ? (
                        <textarea
                          rows={2}
                          placeholder="One value per line"
                          value={String(values[f.name] || '')}
                          onChange={(e) => setValues({ ...values, [f.name]: e.target.value })}
                        />
                      ) : (
                        <input
                          value={String(values[f.name] || '')}
                          onChange={(e) => setValues({ ...values, [f.name]: e.target.value })}
                          placeholder={
                            f.name === 'output'
                              ? '{output}'
                              : f.type === 'Path'
                                ? 'Absolute local path'
                                : f.default == null
                                  ? ''
                                  : String(f.default)
                          }
                          spellCheck={false}
                        />
                      )}
                      <small>
                        {f.help || (f.flags.length ? f.flags.join(', ') : 'Positional argument')}
                        {f.name === 'output' ? ' · Use {output} for a fresh job folder.' : ''}
                      </small>
                    </label>
                  ))}
                </div>
                <button className="text-button" onClick={() => setAdvanced(!advanced)}>
                  {advanced ? 'Fewer options' : 'All command options'} ({schema?.fields.length || 0}
                  )
                </button>
              </>
            )}
            <div className="lab-command-preview" aria-label="Operation arguments">
              <Terminal size={14} />
              <code>
                {operation} {raw ? rawText : args.map((v) => JSON.stringify(v)).join(' ')}
              </code>
            </div>
            <p className="lab-boundary">
              Outputs and logs are retained in Kite. Image tools use Image Studio’s runtime and
              dependencies; embedding tools use the local processor dependencies. Export/build
              commands can download source weights. Existing model files change only when the
              selected operation writes to their path.
            </p>
            <button className="button primary" disabled={busy || !catalog} onClick={() => run()}>
              {running ? <Spinner /> : <Play size={15} />} Run {operation}
            </button>
            {(log || running || result) && (
              <section className="lab-output">
                <div>
                  <strong>Operation output</strong>
                  {result && (
                    <a href={result.log_url} download>
                      <ArrowDownToLine size={14} /> Full log
                    </a>
                  )}
                </div>
                <pre aria-label="Tool output">{log || 'Starting the isolated tool process…'}</pre>
                {result && (
                  <>
                    <p>
                      Saved in <code>{result.directory}</code>
                    </p>
                    <details>
                      <summary>Files and reports</summary>
                      {result.files.map((name) => (
                        <a
                          className="lab-artifact"
                          key={name}
                          href={result.log_url.replace(
                            'process.log',
                            name.split('/').map(encodeURIComponent).join('/'),
                          )}
                          download
                        >
                          {name}
                        </a>
                      ))}
                    </details>
                  </>
                )}
              </section>
            )}
          </section>
        </div>
      )}
      {tab === 'performance' && (
        <>
          <div className="lab-section-heading">
            <div>
              <h2>Every model. Every measurement.</h2>
              <p>
                Compare observations by prompt, machine, and workload. Original benchmark provenance
                is preserved.
              </p>
            </div>
            <button
              className="button primary"
              onClick={() =>
                setObservation({
                  model_id: models[0]?.id || '',
                  task: 'generation',
                  prompt: '',
                  machine: '',
                  notes: '',
                })
              }
            >
              <Plus size={15} /> Record measurement
            </button>
          </div>
          <div className="search-field">
            <Search size={15} />
            <input
              aria-label="Filter measurements"
              placeholder="Filter models, prompts, machines…"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
            />
          </div>
          <div className="lab-table-wrap">
            <table className="lab-table">
              <thead>
                <tr>
                  <th>Model / workload</th>
                  <th>Prompt</th>
                  <th>Machine</th>
                  <th>Latency</th>
                  <th>Throughput</th>
                  <th>Source</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((r) => (
                  <tr key={r.id}>
                    <td>
                      <strong>{r.model_name}</strong>
                      <small>{r.task}</small>
                    </td>
                    <td>
                      <details>
                        <summary>{r.prompt.slice(0, 90)}</summary>
                        <p>{r.prompt}</p>
                        <p>{r.notes}</p>
                      </details>
                    </td>
                    <td>{r.machine}</td>
                    <td>
                      {r.latency_ms != null
                        ? `${r.latency_ms.toFixed(1)} ms`
                        : r.ttft_ms != null
                          ? `${r.ttft_ms.toFixed(1)} ms TTFT`
                          : '—'}
                    </td>
                    <td>
                      {r.decode_tps != null
                        ? `${r.decode_tps.toFixed(2)} tokens/s`
                        : r.embeddings_per_second != null
                          ? `${r.embeddings_per_second.toFixed(2)} inputs/s`
                          : r.images_per_second != null
                            ? `${r.images_per_second.toFixed(3)} images/s`
                            : '—'}
                    </td>
                    <td>
                      {r.source === 'manual' ? (
                        <button
                          className="text-button"
                          onClick={() =>
                            setObservation(
                              Object.fromEntries(
                                Object.entries(r).map(([k, v]) => [k, String(v ?? '')]),
                              ),
                            )
                          }
                        >
                          Edit
                        </button>
                      ) : (
                        <span>{r.source}</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {!filtered.length && !loading && (
            <p className="muted">No measurements match this filter.</p>
          )}
          <button
            className="text-button"
            onClick={() => download('kite-performance.json', filtered)}
          >
            Export observations
          </button>
        </>
      )}
      {tab === 'wishlist' && (
        <>
          <div className="lab-section-heading">
            <div>
              <h2>What should run locally next?</h2>
              <p>Track model conversion requests and their progress.</p>
            </div>
            <button
              className="button primary"
              onClick={() =>
                setWish({
                  title: '',
                  description: '',
                  status: 'open',
                  priority: 'medium',
                  kind: 'conversion',
                })
              }
            >
              <Plus size={15} /> Add request
            </button>
          </div>
          <div className="lab-wishlist">
            {wishes.map((w) => (
              <button className="lab-wish" key={w.id} onClick={() => setWish(w)}>
                <Badge>
                  {
                    (
                      {
                        open: 'To convert',
                        investigating: 'Converting',
                        done: 'Converted',
                        archived: 'Archived',
                      } as Record<string, string>
                    )[w.status]
                  }
                </Badge>
                <h3>{w.title}</h3>
                <p>{w.description}</p>
                <small>
                  {w.priority} priority <ArrowRight size={14} />
                </small>
              </button>
            ))}
          </div>
        </>
      )}
      {tab === 'models' && (
        <>
          <div className="lab-section-heading">
            <div>
              <h2>The graph behind the model.</h2>
              <p>Inspect ONNX or GGUF files, operator inventories, and graph differences.</p>
            </div>
            <button
              className="button primary"
              onClick={() => setModelDraft({ name: '', path: '' })}
            >
              <Plus size={15} /> Add graph
            </button>
          </div>
          <div className="lab-compare">
            <select
              aria-label="First graph"
              value={compareA}
              onChange={(e) => setCompareA(e.target.value)}
            >
              <option value="">First graph…</option>
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
            <select
              aria-label="Second graph"
              value={compareB}
              onChange={(e) => setCompareB(e.target.value)}
            >
              <option value="">Second graph…</option>
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
            <button
              className="button secondary"
              disabled={!compareA || !compareB}
              onClick={async () => {
                setLoading(true);
                try {
                  setInspect(
                    await api('/api/evidence/compare', 'POST', { a: compareA, b: compareB }),
                  );
                } catch (e) {
                  notify(errorText(e));
                } finally {
                  setLoading(false);
                }
              }}
            >
              Compare graphs
            </button>
          </div>
          <div className="lab-graph-list">
            {models.map((m) => (
              <article key={m.id}>
                <Code2 size={22} />
                <div>
                  <h3>{m.name}</h3>
                  <p>{m.path}</p>
                </div>
                <button
                  className="button secondary"
                  onClick={async () => {
                    setLoading(true);
                    try {
                      setInspect(await api('/api/evidence/models/' + m.id));
                    } catch (e) {
                      notify(errorText(e));
                    } finally {
                      setLoading(false);
                    }
                  }}
                >
                  Inspect
                </button>
                <button
                  className="text-button"
                  onClick={() => save('/api/evidence/models/remove', { id: m.id }, () => {})}
                >
                  Remove reference
                </button>
              </article>
            ))}
          </div>
        </>
      )}
      {tab === 'runs' && (
        <div className="lab-run-list">
          {runs.map((r) => (
            <article key={r.id}>
              <div>
                <strong>{r.model.replace(/^(evo|tool):/, '')}</strong>
                <Badge>{r.status}</Badge>
                <small>{new Date(r.created * 1000).toLocaleString()}</small>
              </div>
              {r.error && <p>{r.error}</p>}
              <a href={'/api/lab/jobs/' + r.id + '/process.log'} download>
                Download log
              </a>
              {r.result && (
                <button className="text-button" onClick={() => setInspect(r.result)}>
                  View outputs
                </button>
              )}
            </article>
          ))}
          {!runs.length && !loading && (
            <p className="muted">Tool runs appear here, with their saved output and logs.</p>
          )}
        </div>
      )}
      {wish && (
        <Modal
          title={wish.id ? 'Edit conversion request' : 'New conversion request'}
          onClose={() => setWish(null)}
        >
          <div className="lab-modal-form">
            <label className="field">
              <span>Model / request</span>
              <input
                value={wish.title}
                onChange={(e) => setWish({ ...wish, title: e.target.value })}
              />
            </label>
            <label className="field">
              <span>Notes</span>
              <textarea
                rows={4}
                value={wish.description}
                onChange={(e) => setWish({ ...wish, description: e.target.value })}
              />
            </label>
            <label className="field">
              <span>Status</span>
              <select
                value={wish.status}
                onChange={(e) => setWish({ ...wish, status: e.target.value })}
              >
                {[
                  ['open', 'To convert'],
                  ['investigating', 'Converting'],
                  ['done', 'Converted'],
                  ['archived', 'Archived'],
                ].map(([v, l]) => (
                  <option key={v} value={v}>
                    {l}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Priority</span>
              <select
                value={wish.priority}
                onChange={(e) => setWish({ ...wish, priority: e.target.value })}
              >
                {['high', 'medium', 'low'].map((v) => (
                  <option key={v}>{v}</option>
                ))}
              </select>
            </label>
            <button
              className="button primary"
              onClick={() => save('/api/evidence/wishlist', wish, () => setWish(null))}
            >
              Save request
            </button>
          </div>
        </Modal>
      )}
      {observation && (
        <Modal title="Performance observation" onClose={() => setObservation(null)}>
          <div className="lab-modal-form">
            <label className="field">
              <span>Model</span>
              <select
                value={observation.model_id}
                onChange={(e) => setObservation({ ...observation, model_id: e.target.value })}
              >
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Workload</span>
              <select
                value={observation.task}
                onChange={(e) => setObservation({ ...observation, task: e.target.value })}
              >
                {['generation', 'embedding', 'image'].map((t) => (
                  <option key={t}>{t}</option>
                ))}
              </select>
            </label>
            {[
              'prompt',
              'machine',
              'notes',
              ...(observation.task === 'generation'
                ? ['ttft_ms', 'decode_tps']
                : observation.task === 'embedding'
                  ? ['latency_ms', 'embeddings_per_second']
                  : ['latency_ms', 'images_per_second']),
            ].map((key) => (
              <label className="field" key={key}>
                <span>{title(key)}</span>
                <input
                  value={observation[key] || ''}
                  onChange={(e) => setObservation({ ...observation, [key]: e.target.value })}
                />
              </label>
            ))}
            <button
              className="button primary"
              onClick={() =>
                save('/api/evidence/performance', observation, () => setObservation(null))
              }
            >
              Save measurement
            </button>
          </div>
        </Modal>
      )}
      {modelDraft && (
        <Modal title="Add graph to the evidence library" onClose={() => setModelDraft(null)}>
          <div className="lab-modal-form">
            <label className="field">
              <span>Name</span>
              <input
                value={modelDraft.name}
                onChange={(e) => setModelDraft({ ...modelDraft, name: e.target.value })}
              />
            </label>
            <label className="field">
              <span>ONNX / GGUF file or package path</span>
              <input
                value={modelDraft.path}
                onChange={(e) => setModelDraft({ ...modelDraft, path: e.target.value })}
              />
            </label>
            <button
              className="button primary"
              onClick={() => save('/api/evidence/models', modelDraft, () => setModelDraft(null))}
            >
              Add graph
            </button>
          </div>
        </Modal>
      )}
      {inspect != null && (
        <Modal
          title="Graph / evidence details"
          onClose={() => setInspect(null)}
          className="lab-detail-modal"
        >
          <pre>{JSON.stringify(inspect, null, 2)}</pre>
          <button
            className="button secondary"
            onClick={() => download('kite-evidence.json', inspect)}
          >
            Export JSON
          </button>
        </Modal>
      )}
    </div>
  );
}
