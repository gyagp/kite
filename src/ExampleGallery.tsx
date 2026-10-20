import { useRef, useState } from 'react';
import {
  ArrowLeft,
  ArrowRight,
  AudioLines,
  Braces,
  Check,
  ChevronRight,
  Globe2,
  Image,
  Layers3,
  Pause,
  Play,
  Rocket,
  Sparkles,
  Video,
} from 'lucide-react';
import { api } from './api';
import { Modal, Spinner } from './components';
import type { EmbeddingExample } from './types';

export const exampleAsset = (name: string) =>
  '/api/examples/assets/' + name.split('/').map(encodeURIComponent).join('/');
function ModalityIcon({ kind }: { kind: string }) {
  return kind === 'image' ? (
    <Image size={14} />
  ) : kind === 'audio' ? (
    <AudioLines size={14} />
  ) : kind === 'video' ? (
    <Video size={14} />
  ) : (
    <Braces size={14} />
  );
}

function AudioPreview({ item }: { item: EmbeddingExample }) {
  const ref = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  return (
    <div className="showcase-audio">
      {item.cover && <img src={exampleAsset(item.cover)} alt="Astronaut above Earth" />}
      <div className="showcase-audio-content">
        <span className="showcase-kicker">TRANSMISSION / 01</span>
        <h3>
          A voice. A picture.
          <br />A connection.
        </h3>
        <div className="audio-deck">
          <button
            aria-label={playing ? 'Pause briefing' : 'Play briefing'}
            onClick={() => {
              if (playing) ref.current?.pause();
              else void ref.current?.play();
            }}
          >
            {playing ? <Pause size={22} /> : <Play size={22} />}
          </button>
          <div className="showcase-waveform" aria-hidden="true">
            {(item.waveform || []).map((n, i) => (
              <span
                key={i}
                className={i / (item.waveform?.length || 1) < progress ? 'played' : ''}
                style={{ height: `${Math.max(10, n * 100)}%` }}
              />
            ))}
          </div>
          <span>
            {Math.round((item.audio_duration || 0) * progress)} /{' '}
            {Math.round(item.audio_duration || 0)}s
          </span>
        </div>
        <audio
          ref={ref}
          src={exampleAsset(item.preview_audio!)}
          preload="metadata"
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => setPlaying(false)}
          onTimeUpdate={() => {
            const el = ref.current;
            if (el) setProgress(el.currentTime / (el.duration || 1));
          }}
        />
        <p>{item.audio_credit || 'Synthetic narration'} · press play to listen</p>
      </div>
    </div>
  );
}

export function ExampleGallery({
  items,
  loading,
  onChoose,
  onClose,
}: {
  items: EmbeddingExample[];
  loading: string;
  onChoose: (item: EmbeddingExample, run?: boolean, query?: string) => void;
  onClose: () => void;
}) {
  const [filter, setFilter] = useState('all');
  const [selected, setSelected] = useState<EmbeddingExample | null>(null);
  const [query, setQuery] = useState('');
  const [credits, setCredits] = useState<{
    images: { title: string; credit: string; source_url: string }[];
    audio?: { provider: string; voice: string };
  } | null>(null);
  const [creditsError, setCreditsError] = useState('');
  const featured = items.find((i) => i.featured) || items[0];
  const choose = (item: EmbeddingExample) => {
    setSelected(item);
    setQuery(item.query);
  };
  const shown = items.filter(
    (i) =>
      filter === 'all' ||
      (filter === 'mixed' ? i.modalities.length === 4 : i.modalities.includes(filter)),
  );
  return (
    <Modal title="Multimodal experiences" className="showcase-modal" onClose={onClose}>
      <div className="showcase-scroll">
        {selected ? (
          <div className="showcase-detail" key={selected.id}>
            <button className="showcase-back" onClick={() => setSelected(null)}>
              <ArrowLeft size={16} />
              All experiences
            </button>
            <div className="showcase-detail-heading">
              <div>
                <span className="showcase-eyebrow">{selected.eyebrow}</span>
                <h2>{selected.title}</h2>
              </div>
              <div className="showcase-pills">
                {selected.modalities.map((m) => (
                  <span key={m}>
                    <ModalityIcon kind={m} />
                    {m}
                  </span>
                ))}
              </div>
            </div>
            {selected.preview_audio && !selected.preview_video ? (
              <AudioPreview item={selected} />
            ) : selected.preview_video ? (
              <div className="showcase-film">
                <video
                  controls
                  playsInline
                  preload="metadata"
                  poster={selected.cover ? exampleAsset(selected.cover) : undefined}
                  src={exampleAsset(selected.preview_video)}
                />
                <div>
                  <Video size={14} />
                  <span>12-second photo film · three scenes · NASA imagery</span>
                </div>
              </div>
            ) : (
              <div className="showcase-still">
                <img src={exampleAsset(selected.cover!)} alt={selected.title} />
                <span>NASA image collection</span>
              </div>
            )}
            <div className="showcase-detail-copy">
              <p>{selected.story || selected.description}</p>
              <div className="showcase-pipeline">
                <span>
                  <ModalityIcon kind={selected.modalities[0]} />
                  {selected.modalities.length > 1 ? 'Your media' : 'Images'}
                </span>
                <ChevronRight size={14} />
                <span>
                  <Layers3 size={15} />
                  768 dimensions
                </span>
                <ChevronRight size={14} />
                <span>
                  <Sparkles size={15} />
                  Real similarity
                </span>
              </div>
            </div>
            {selected.transcript && (
              <details className="showcase-transcript">
                <summary>Read the briefing</summary>
                <p>“{selected.transcript}”</p>
                <small>The query is the spoken audio. This transcript is for preview.</small>
              </details>
            )}
            {selected.query && (
              <label className="showcase-query">
                <span>Try your own description</span>
                <textarea rows={2} value={query} onChange={(e) => setQuery(e.target.value)} />
              </label>
            )}
            {!!selected.suggestions?.length && (
              <div className="showcase-suggestions">
                <span>Or try</span>
                {selected.suggestions.map((s) => (
                  <button key={s} onClick={() => setQuery(s)}>
                    {s}
                  </button>
                ))}
              </div>
            )}
            <div className="showcase-launch">
              <span>
                <Check size={15} />
                Files included · runs on your computer
              </span>
              <div>
                <button
                  className="button subtle"
                  disabled={!!loading}
                  onClick={() => onChoose(selected, false, selected.query ? query : undefined)}
                >
                  Open in workspace
                </button>
                <button
                  className="button primary"
                  disabled={!!loading || (!!selected.query && !query.trim())}
                  onClick={() => onChoose(selected, true, selected.query ? query : undefined)}
                >
                  {loading === selected.id ? <Spinner /> : <Play size={16} />}Run demo
                </button>
              </div>
            </div>
          </div>
        ) : (
          <>
            <section className="showcase-hero">
              {featured?.cover && (
                <img src={exampleAsset(featured.cover)} alt="Sunrise along Earth's horizon" />
              )}
              <div className="showcase-hero-content">
                <span className="showcase-kicker">
                  <span />
                  THE ORBIT COLLECTION
                </span>
                <h2>
                  A universe
                  <br />
                  of meaning.
                </h2>
                <p>
                  Images you can search. Voices you can see.
                  <br />
                  Discover what happens when every medium connects.
                </p>
                <button onClick={() => choose(featured)}>
                  <Rocket size={17} />
                  Explore the full experience
                  <ArrowRight size={17} />
                </button>
              </div>
              <div className="showcase-hero-footer">
                <span>06 EXPERIENCES</span>
                <span>NASA IMAGERY</span>
                <span>100% LOCAL INFERENCE</span>
              </div>
              <div className="showcase-orbits" aria-hidden="true">
                <i />
                <i />
                <span>
                  <Layers3 size={21} />
                </span>
              </div>
            </section>
            <div className="showcase-section-heading">
              <div>
                <span className="showcase-eyebrow">CHOOSE YOUR PERSPECTIVE</span>
                <h3>Small experiments. Big possibilities.</h3>
              </div>
              <div className="showcase-filters" aria-label="Filter experiences">
                {[
                  ['all', 'All'],
                  ['image', 'Visual'],
                  ['audio', 'Audio'],
                  ['video', 'Video'],
                  ['mixed', 'All together'],
                ].map(([id, label]) => (
                  <button
                    key={id}
                    className={filter === id ? 'active' : ''}
                    aria-pressed={filter === id}
                    onClick={() => setFilter(id)}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
            <div className="showcase-grid">
              {shown.map((item, index) => (
                <article className={`showcase-card tone-${item.accent || 'blue'}`} key={item.id}>
                  <button
                    className="showcase-cover"
                    aria-label={`Preview ${item.title}`}
                    onClick={() => choose(item)}
                  >
                    {item.cover && <img src={exampleAsset(item.cover)} alt="" loading="lazy" />}
                    <span className="showcase-number">
                      {String(items.indexOf(item) + 1).padStart(2, '0')}
                    </span>
                    <span className="showcase-cover-action">
                      {item.preview_video || item.preview_audio ? (
                        <Play size={20} />
                      ) : (
                        <ArrowRight size={20} />
                      )}
                    </span>
                    <span className="showcase-cover-pills">
                      {item.modalities.map((m) => (
                        <span key={m} title={m}>
                          <ModalityIcon kind={m} />
                        </span>
                      ))}
                    </span>
                  </button>
                  <div className="showcase-card-copy">
                    <span className="showcase-eyebrow">
                      {item.eyebrow?.split(' / ')[1] || `EXPERIENCE ${index + 1}`}
                    </span>
                    <h3>{item.title}</h3>
                    <p>{item.description}</p>
                    <div>
                      <button onClick={() => choose(item)}>
                        Take a look
                        <ArrowRight size={15} />
                      </button>
                      <button
                        aria-label={`Run ${item.title}`}
                        title="Run this example now"
                        disabled={!!loading}
                        onClick={() => onChoose(item, true)}
                      >
                        {loading === item.id ? <Spinner /> : <Play size={14} />}
                      </button>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          </>
        )}
        <footer className="showcase-attribution">
          <span>
            <Globe2 size={14} />
            Real imagery. Real model outputs. No preset scores.
          </span>
          <button
            onClick={async () => {
              if (credits) {
                setCredits(null);
                return;
              }
              try {
                setCredits(await api('/api/examples/credits'));
              } catch {
                setCreditsError('Could not load credits. See demo/showcase/credits.json.');
              }
            }}
          >
            {credits ? 'Hide credits' : 'Image & audio credits'}
          </button>
        </footer>
        {creditsError && <p>{creditsError}</p>}
        {credits && (
          <div className="showcase-credits">
            <h3>Made for exploration</h3>
            <p>
              NASA imagery, an authored photo montage, and Azure Speech narration. NASA does not
              endorse Kite.
            </p>
            {credits.images.map((c) => (
              <a key={c.source_url} href={c.source_url} target="_blank" rel="noreferrer">
                <span>{c.title}</span>
                <small>{c.credit}</small>
                <ArrowRight size={13} />
              </a>
            ))}
            <p>
              Voice: {credits.audio?.voice || 'Azure neural voice'} ·{' '}
              {credits.audio?.provider || 'Azure AI Speech'}. The audio is synthetic, not an actual
              mission recording. Playback and inference run locally.
            </p>
          </div>
        )}
      </div>
    </Modal>
  );
}
