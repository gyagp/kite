import { useEffect, useRef, useState } from 'react';
import {
  ArrowUpFromLine,
  CirclePlus,
  Download,
  Pause,
  Play,
  ScanLine,
  Square,
  Trash2,
  Undo2,
  Video,
} from 'lucide-react';
import { api, errorText, ms } from './api';
import { ProfileResults } from './ProfileResults';
import type { Attachment, Execute, Model } from './types';
import type { ImageProfile } from './image-performance';
import './segmentation.css';

type Point = { x: number; y: number; label: number };
type Selection = { id: number; name: string; points: Point[]; box?: number[] | null };
type Result = {
  id: string;
  frame_count: number;
  objects: Selection[];
  video_url: string;
  masks_url: string;
  metadata_url: string;
  processing_fps: number;
  total_ms: number;
  request_ms: number;
  model_load_ms: number;
  setup_ms: number;
  phases: Record<string, number>;
  profile?: ImageProfile;
};
const colors = ['#287dff', '#ef922e', '#b45cf0', '#20bea8'];
const object = (id: number): Selection => ({ id, name: `Object ${id}`, points: [] });

export function SegmentationStudio({
  models,
  execute,
  busy,
  notify,
  onStop,
  onModels,
}: {
  models: Model[];
  execute: Execute;
  busy: boolean;
  notify: (message: string) => void;
  onStop: () => void;
  onModels: () => void;
}) {
  const [model, setModel] = useState('');
  const [clip, setClip] = useState<Attachment | null>(null);
  const [duration, setDuration] = useState(0);
  const [start, setStart] = useState(0);
  const [objects, setObjects] = useState<Selection[]>([object(1)]);
  const [active, setActive] = useState(0);
  const [tool, setTool] = useState<'include' | 'exclude' | 'box'>('include');
  const [boxStart, setBoxStart] = useState<Point | null>(null);
  const [draft, setDraft] = useState<number[] | null>(null);
  const [maxFrames, setMaxFrames] = useState(60);
  const [stride, setStride] = useState(1);
  const [profile, setProfile] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState('');
  const [count, setCount] = useState([0, 0]);
  const [preview, setPreview] = useState('');
  const [result, setResult] = useState<Result | null>(null);
  const [history, setHistory] = useState<Result[]>([]);
  const [elapsed, setElapsed] = useState(0);
  const [previewingClip, setPreviewingClip] = useState(false);
  const [resultPlaying, setResultPlaying] = useState(false);
  const [playbackError, setPlaybackError] = useState('');
  const fileInput = useRef<HTMLInputElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const resultVideo = useRef<HTMLVideoElement>(null);
  const resultPanel = useRef<HTMLElement>(null);
  const locked = busy || uploading;
  useEffect(() => {
    if (!models.some((m) => m.id === model && m.available))
      setModel(models.find((m) => m.available)?.id || '');
  }, [models, model]);
  const reload = () =>
    api<{ status: string; result?: Result }[]>('/api/segments')
      .then((rows) =>
        setHistory(rows.filter((r) => r.status === 'completed' && r.result).map((r) => r.result!)),
      )
      .catch((e) => notify(errorText(e)));
  useEffect(() => {
    void reload();
  }, []);
  useEffect(() => {
    if (!running) return;
    const began = Date.now();
    const timer = setInterval(() => setElapsed((Date.now() - began) / 1000), 250);
    return () => clearInterval(timer);
  }, [running]);
  useEffect(() => {
    const player = video.current;
    if (!player) return;
    if (previewingClip) {
      player.play().catch(() => {
        if (video.current === player) setPreviewingClip(false);
      });
    } else {
      player.pause();
      player.currentTime = start;
    }
  }, [previewingClip, clip?.id, start]);
  useEffect(() => {
    if (!result) return;
    setPlaybackError('');
    setResultPlaying(resultVideo.current?.paused === false);
    if (resultPanel.current?.getClientRects().length) {
      resultPanel.current.scrollIntoView({ block: 'start', behavior: 'smooth' });
    }
    // Muted playback can start after a long inference run without a new gesture.
    // Keep explicit playback controls available if autoplay is blocked.
    if (resultVideo.current) {
      resultVideo.current.currentTime = 0;
      resultVideo.current.play().catch(() => setResultPlaying(false));
    }
  }, [result]);
  const stopPreview = () => {
    video.current?.pause();
    if (video.current) video.current.currentTime = start;
    setPreviewingClip(false);
  };
  const chooseClip = (attachment: Attachment, sample = false) => {
    setClip(attachment);
    setPreviewingClip(sample);
    setStart(0);
    setResult(null);
    setPreview('');
    setActive(0);
    setObjects(
      sample
        ? [{ ...object(1), name: 'Jumper', points: [{ x: 0.405, y: 0.38, label: 1 }] }]
        : [object(1)],
    );
    setCount([0, 0]);
  };
  const upload = async (file?: File) => {
    if (!file) return;
    setUploading(true);
    try {
      if (file.size > 64 * 1024 * 1024) throw new Error('Choose a clip under 64 MB.');
      const data = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result).split(',')[1]);
        reader.onerror = () => reject(new Error('Could not read the video.'));
        reader.readAsDataURL(file);
      });
      const attachment = await api<Attachment>('/api/attachments', 'POST', {
        name: file.name,
        data,
      });
      if (attachment.kind !== 'video') throw new Error('Choose an MP4 or MOV video.');
      chooseClip(attachment);
    } catch (e) {
      notify(errorText(e));
    } finally {
      setUploading(false);
    }
  };
  const sample = async () => {
    setUploading(true);
    try {
      chooseClip(await api<Attachment>('/api/segments/sample', 'POST', {}), true);
    } catch (e) {
      notify(errorText(e));
    } finally {
      setUploading(false);
    }
  };
  const coordinates = (event: React.PointerEvent<SVGSVGElement>): Point => {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
      label: tool === 'exclude' ? 0 : 1,
    };
  };
  const update = (fn: (s: Selection) => Selection) =>
    setObjects((rows) => rows.map((s, i) => (i === active ? fn(s) : s)));
  const begin = (event: React.PointerEvent<SVGSVGElement>) => {
    if (locked) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    const p = coordinates(event);
    if (tool === 'box') {
      setBoxStart(p);
      setDraft([p.x, p.y, p.x, p.y]);
    } else update((s) => ({ ...s, points: [...s.points, p].slice(-32) }));
  };
  const move = (event: React.PointerEvent<SVGSVGElement>) => {
    if (!boxStart) return;
    const p = coordinates(event);
    setDraft([
      Math.min(p.x, boxStart.x),
      Math.min(p.y, boxStart.y),
      Math.max(p.x, boxStart.x),
      Math.max(p.y, boxStart.y),
    ]);
  };
  const end = () => {
    if (draft && draft[2] - draft[0] > 0.003 && draft[3] - draft[1] > 0.003)
      update((s) => ({ ...s, box: draft }));
    setBoxStart(null);
    setDraft(null);
  };
  const track = async () => {
    stopPreview();
    setRunning(true);
    setResult(null);
    setPreview('');
    setElapsed(0);
    setCount([0, 0]);
    setProgress('Preparing SAM 3…');
    let cancelled = false;
    try {
      const output = await execute(
        {
          kind: 'track',
          model,
          attachment: clip?.id,
          start_time: start,
          objects,
          max_frames: maxFrames,
          stride,
          profile,
        },
        (event) => {
          if (event.type === 'status') setProgress(event.message || '');
          if (event.type === 'tracking_progress') {
            setCount([event.completed || 0, event.total || 0]);
            const data = event.data as unknown as { preview_url: string };
            setPreview(data.preview_url);
          }
          if (event.type === 'cancelled') {
            cancelled = true;
            setProgress('Stopped. You can adjust the selection and try again.');
          }
        },
      );
      if (output) setResult(output as unknown as Result);
      else if (!cancelled) setProgress('The run ended without a result.');
      void reload();
    } catch (e) {
      setProgress(errorText(e));
    } finally {
      setRunning(false);
    }
  };
  const ready = model && clip && objects.every((o) => o.box || o.points.some((p) => p.label === 1));
  const renderBox = (box: number[], color: string, key: string) => (
    <rect
      key={key}
      x={box[0] * 1000}
      y={box[1] * 1000}
      width={(box[2] - box[0]) * 1000}
      height={(box[3] - box[1]) * 1000}
      fill={color + '18'}
      stroke={color}
      strokeWidth="3"
      vectorEffect="non-scaling-stroke"
    />
  );
  return (
    <div className="seg-studio">
      <header className="seg-heading">
        <div>
          <div className="seg-eyebrow">
            <ScanLine size={15} /> SAM 3 · VIDEO TRACKING
          </div>
          <h1>Pick it. Follow it.</h1>
          <p>Select an object once. Follow its mask as it moves through your video.</p>
        </div>
        <span className="seg-runtime">ONNX Runtime · WebGPU D3D12</span>
      </header>
      {(running || progress) && (
        <section className="seg-progress" aria-live="polite">
          <div>
            <strong>{running ? progress : result ? 'Tracking complete' : progress}</strong>
            <span>
              {count[1] ? `${count[0]} / ${count[1]} frames · ` : ''}
              {elapsed.toFixed(1)}s
            </span>
          </div>
          {running && (
            <progress
              aria-label="Tracking progress"
              max={count[1] || 1}
              {...(count[1] ? { value: count[0] } : {})}
            />
          )}
        </section>
      )}
      {running && preview && (
        <img className="seg-live-preview" src={preview} alt="Latest tracked frame" />
      )}
      {result && (
        <section className="seg-result" ref={resultPanel} aria-label="Tracked video playback">
          <div className="seg-panel-heading">
            <h2>
              03 <span>Your object, in motion</span>
            </h2>
            <span>
              {result.frame_count} frames · {result.objects.length} object
              {result.objects.length > 1 ? 's' : ''}
            </span>
          </div>
          <div className="seg-playback-toolbar">
            <button
              className="button primary"
              onClick={() => {
                const player = resultVideo.current;
                if (!player) return;
                if (player.paused)
                  player
                    .play()
                    .catch(() =>
                      setPlaybackError(
                        'Playback could not start. Try downloading the overlay video.',
                      ),
                    );
                else player.pause();
              }}
            >
              {resultPlaying ? <Pause size={16} /> : <Play size={16} />}
              {resultPlaying ? 'Pause tracked video' : 'Play tracked video'}
            </button>
            <span>Tracking overlay · loops automatically</span>
          </div>
          <video
            key={result.id}
            ref={resultVideo}
            controls
            playsInline
            autoPlay
            muted
            loop
            aria-label="Tracked video"
            src={result.video_url}
            onPlay={() => setResultPlaying(true)}
            onPause={() => setResultPlaying(false)}
            onError={() =>
              setPlaybackError('This video could not be played. Try downloading the overlay video.')
            }
          />
          {playbackError && <p role="alert">{playbackError}</p>}
          <div className="seg-downloads">
            <a className="button primary" href={result.video_url} download>
              <Download size={16} />
              Overlay video
            </a>
            <a className="button subtle" href={result.masks_url} download>
              Mask preview
            </a>
            <a className="button subtle" href={result.metadata_url} download>
              Tracking data
            </a>
            <a className="button subtle" href={`/api/segments/${result.id}/masks.zip`} download>
              All label masks (ZIP)
            </a>
          </div>
          <p className="seg-caption">
            Lossless label masks are saved for every frame; pixel values 1–4 identify your objects.
            Mask MP4 is a visual preview.
          </p>
          <div className="seg-statistics">
            <div>
              <span>Total request</span>
              <strong>{ms(result.request_ms)}</strong>
            </div>
            <div>
              <span>Model setup</span>
              <strong>{ms(result.setup_ms)}</strong>
            </div>
            <div>
              <span>Processing speed</span>
              <strong>{result.processing_fps.toFixed(2)} fps</strong>
            </div>
          </div>
          <div className="seg-breakdown">
            {Object.entries(result.phases).map(([name, value]) => (
              <div key={name}>
                <span>{name.replaceAll('_', ' ')}</span>
                <meter min="0" max={result.total_ms} value={value} />
                <strong>{ms(value)}</strong>
              </div>
            ))}
          </div>
          {result.profile && <ProfileResults profile={result.profile} />}
        </section>
      )}
      <div className="seg-layout">
        <section className="seg-workspace">
          <div className="seg-panel-heading">
            <h2>
              01 <span>Choose your scene</span>
            </h2>
            <button
              className="button subtle"
              disabled={locked}
              onClick={() => fileInput.current?.click()}
            >
              <ArrowUpFromLine size={16} />
              {uploading ? 'Uploading…' : 'Open video'}
            </button>
            <input
              hidden
              ref={fileInput}
              type="file"
              accept=".mp4,.mov"
              onChange={(e) => {
                void upload(e.target.files?.[0]);
                e.target.value = '';
              }}
            />
          </div>
          {clip ? (
            <>
              <div className="seg-canvas">
                <video
                  ref={video}
                  key={clip.id}
                  src={`/api/attachments/${clip.id}`}
                  preload="auto"
                  playsInline
                  muted
                  loop
                  onLoadedMetadata={(e) => {
                    setDuration(e.currentTarget.duration);
                    e.currentTarget.currentTime = start;
                  }}
                  onError={() => notify('This video cannot be previewed. Try an H.264 MP4.')}
                />
                <svg
                  viewBox="0 0 1000 1000"
                  preserveAspectRatio="none"
                  aria-label="Select objects in this frame"
                  style={{ display: previewingClip ? 'none' : undefined }}
                  onPointerDown={begin}
                  onPointerMove={move}
                  onPointerUp={end}
                  onPointerCancel={() => {
                    setBoxStart(null);
                    setDraft(null);
                  }}
                >
                  {objects.map((o, i) => (
                    <g key={o.id}>
                      {o.box && renderBox(o.box, colors[i], `box-${i}`)}
                      {o.points.map((p, j) => (
                        <g key={j}>
                          <circle
                            cx={p.x * 1000}
                            cy={p.y * 1000}
                            r="9"
                            fill={p.label ? colors[i] : '#fff'}
                            stroke={colors[i]}
                            strokeWidth="2"
                          />
                          <text
                            x={p.x * 1000}
                            y={p.y * 1000 + 4}
                            textAnchor="middle"
                            fill={p.label ? '#fff' : colors[i]}
                            fontSize="15"
                          >
                            {p.label ? '+' : '−'}
                          </text>
                        </g>
                      ))}
                    </g>
                  ))}
                  {draft && renderBox(draft, colors[active], 'draft')}
                </svg>
                <span className="seg-frame-label">
                  {previewingClip
                    ? 'Playing original clip'
                    : `Selection frame · ${start.toFixed(2)}s`}
                </span>
              </div>
              <div className="seg-playback-toolbar">
                <button
                  className="button subtle"
                  disabled={locked}
                  onClick={() => {
                    if (previewingClip) stopPreview();
                    else setPreviewingClip(true);
                  }}
                >
                  {previewingClip ? <Pause size={16} /> : <Play size={16} />}
                  {previewingClip ? 'Back to selection' : 'Play clip'}
                </button>
                <span>
                  {previewingClip
                    ? `Preview only · selection kept at ${start.toFixed(2)}s`
                    : 'Paused for object selection'}
                </span>
              </div>
              <div className="seg-timeline">
                <span>Start frame</span>
                <input
                  aria-label="Start frame"
                  type="range"
                  min="0"
                  max={Math.max(0, duration - 0.1)}
                  step="0.04"
                  value={start}
                  disabled={locked}
                  onChange={(e) => {
                    stopPreview();
                    const value = Number(e.target.value);
                    setStart(value);
                    if (video.current) video.current.currentTime = value;
                    setObjects([object(1)]);
                    setActive(0);
                  }}
                />
                <span>{duration.toFixed(1)}s</span>
              </div>
              <p className="seg-caption">
                {clip.name} · Play to preview, or scrub to select a frame. Changing the selection
                frame clears its points and boxes.
              </p>
            </>
          ) : (
            <div className="seg-empty">
              <div className="seg-empty-icon">
                <Video size={42} />
                <ScanLine size={84} />
              </div>
              <h2>Every moving detail, within reach.</h2>
              <p>
                Open a clip and mark what you want to follow.
                <br />
                MP4 or MOV · up to 64 MB
              </p>
              <button className="button primary" disabled={locked} onClick={sample}>
                <Play size={16} />
                Try “Jumping together”
              </button>
            </div>
          )}
          {clip && (
            <button className="text-button" disabled={locked} onClick={sample}>
              Try the sample clip
            </button>
          )}
        </section>
        <aside className="seg-controls">
          <h2>
            02 <span>Make your selection</span>
          </h2>
          <label>
            Model
            <select
              aria-label="Segmentation model"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              disabled={locked}
            >
              <option value="">Choose SAM 3</option>
              {models
                .filter((m) => m.available)
                .map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
            </select>
          </label>
          {!model && (
            <div className="seg-hint">
              Import your SAM 3 video package in{' '}
              <button className="text-button" onClick={onModels}>
                My models
              </button>
              .
            </div>
          )}
          <div className="seg-objects">
            {objects.map((o, i) => (
              <button
                key={o.id}
                className={active === i ? 'selected' : ''}
                disabled={locked}
                onClick={() => {
                  stopPreview();
                  setActive(i);
                }}
              >
                <i style={{ background: colors[i] }} />
                {o.name}
              </button>
            ))}
            <button
              aria-label="Add object"
              disabled={locked || objects.length >= 4}
              onClick={() => {
                stopPreview();
                setObjects([...objects, object(objects.length + 1)]);
                setActive(objects.length);
              }}
            >
              <CirclePlus size={16} />
            </button>
          </div>
          <div className="seg-tools">
            {(['include', 'exclude', 'box'] as const).map((t) => (
              <button
                key={t}
                disabled={locked}
                className={tool === t ? 'selected' : ''}
                onClick={() => {
                  stopPreview();
                  setTool(t);
                }}
              >
                {t === 'include' ? '+ Include' : t === 'exclude' ? '− Exclude' : '□ Box'}
              </button>
            ))}
          </div>
          <p className="seg-hint">
            {tool === 'box'
              ? 'Drag around the object you want to follow.'
              : tool === 'include'
                ? 'Click inside the object. Add points to clarify your selection.'
                : 'Click background or nearby objects to exclude them.'}
          </p>
          <div className="seg-edit-actions">
            <button
              className="text-button"
              disabled={locked}
              onClick={() =>
                update((s) =>
                  s.box ? { ...s, box: null } : { ...s, points: s.points.slice(0, -1) },
                )
              }
            >
              <Undo2 size={14} />
              Undo
            </button>
            <button
              className="text-button"
              disabled={locked}
              onClick={() => {
                setObjects([object(1)]);
                setActive(0);
              }}
            >
              <Trash2 size={14} />
              Clear all
            </button>
          </div>
          <div className="seg-numbers">
            <label>
              Frames to process
              <input
                aria-label="Frames to process"
                type="number"
                min="1"
                max="600"
                value={maxFrames}
                disabled={locked}
                onChange={(e) => setMaxFrames(Number(e.target.value))}
              />
            </label>
            <label>
              Frame interval
              <select
                aria-label="Frame interval"
                value={stride}
                disabled={locked}
                onChange={(e) => setStride(Number(e.target.value))}
              >
                {[1, 2, 3, 5, 10].map((n) => (
                  <option key={n} value={n}>
                    {n === 1 ? 'Every frame' : `Every ${n} frames`}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <label className="seg-checkbox">
            <input
              type="checkbox"
              checked={profile}
              disabled={locked}
              onChange={(e) => setProfile(e.target.checked)}
            />
            Profile tracking
          </label>
          <p className="seg-hint">
            Tracks forward from your selection. Frame intervals trade motion detail for speed.
            Output previews are silent.
          </p>
          {running ? (
            <button className="button seg-run" onClick={onStop}>
              <Square size={16} />
              Stop tracking
            </button>
          ) : (
            <button className="button primary seg-run" disabled={locked || !ready} onClick={track}>
              <ScanLine size={18} />
              Track objects
            </button>
          )}
        </aside>
      </div>
      {history.length > 0 && (
        <section className="seg-history">
          <h2>Recent tracks</h2>
          <div>
            {history.slice(0, 8).map((r) => (
              <button key={r.id} onClick={() => setResult({ ...r })} disabled={running}>
                <img src={`/api/segments/${r.id}/frame-00000.jpg`} alt="Tracked clip" />
                <strong>{r.frame_count} frames</strong>
                <span>{r.objects.map((o) => o.name).join(' · ')}</span>
              </button>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
