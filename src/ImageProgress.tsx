import { Check, Clock3, Square } from 'lucide-react';
import { Spinner } from './components';

export const imageStages = [
  ['prepare', 'Prepare prompt'],
  ['prefix', 'Prepare image model'],
  ['denoise', 'Generate image'],
  ['decode', 'Decode & save'],
] as const;

export function ImageProgress({
  stage,
  message,
  completed,
  total,
  elapsed,
  onStop,
}: {
  stage: string;
  message: string;
  completed: number;
  total: number;
  elapsed: number;
  onStop: () => void;
}) {
  const index = Math.max(
    0,
    imageStages.findIndex(([key]) => key === stage),
  );
  const hasSteps = stage === 'denoise' && total > 0;
  const time = `${Math.floor(elapsed / 60)}:${String(Math.floor(elapsed % 60)).padStart(2, '0')}`;
  return (
    <section className="image-progress" aria-label="Image generation progress">
      <div className="image-progress-heading">
        <div>
          <Spinner />
          <strong>Creating your image</strong>
          <span>
            Stage {index + 1} of {imageStages.length}
          </span>
        </div>
        <div>
          <span className="image-progress-clock">
            <Clock3 size={16} />
            {time} elapsed
          </span>
          <button className="button secondary" onClick={onStop}>
            <Square size={13} />
            Cancel generation
          </button>
        </div>
      </div>
      <ol className="image-progress-stages">
        {imageStages.map(([key, label], i) => (
          <li
            key={key}
            className={i < index ? 'complete' : i === index ? 'current' : ''}
            aria-current={i === index ? 'step' : undefined}
          >
            <span>{i < index ? <Check size={16} /> : i + 1}</span>
            {label}
          </li>
        ))}
      </ol>
      <div className="image-progress-detail" role="status" aria-live="polite">
        <strong>{message || imageStages[index][1]}</strong>
        <span>
          {hasSteps
            ? `${completed} of ${total} denoising steps complete`
            : 'Working on this stage…'}
        </span>
      </div>
      <progress
        aria-label={hasSteps ? 'Denoising steps' : imageStages[index][1]}
        max={hasSteps ? total : 1}
        value={hasSteps ? completed : undefined}
      />
      <p>
        First-time model loading can take longer. Your image will appear here and save
        automatically.
      </p>
    </section>
  );
}
