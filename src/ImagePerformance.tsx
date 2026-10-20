import { ProfileResults } from './ProfileResults';
import { Fragment } from 'react';
import { Clock3 } from 'lucide-react';
import { ms } from './api';
import { imageTimingBreakdown, type TimedImage } from './image-performance';

const percent = (value: number) => `${value.toFixed(1)}%`;

export function ImagePerformance({ art }: { art: TimedImage }) {
  const timing = imageTimingBreakdown(art);
  const steps = art.phases.denoise?.step_ms || [];
  return (
    <section className="image-performance" aria-label="Image generation performance">
      <div className="image-timing-heading">
        <div>
          <span>
            <Clock3 size={16} /> Time to saved image
          </span>
          <strong>{ms(timing.total)}</strong>
        </div>
        {timing.largest && (
          <p>
            Largest phase <b>{timing.largest.label}</b>
            <span>
              {ms(timing.largest.ms)} · {percent(timing.largest.percent)} of total
            </span>
          </p>
        )}
      </div>
      <div
        className="image-timing-track"
        role="img"
        aria-label={timing.groups.map((g) => `${g.label}: ${percent(g.percent)}`).join(', ')}
      >
        {timing.groups.map((group) => (
          <span
            key={group.key}
            className={`timing-${group.key}`}
            style={{ flexGrow: group.ms }}
            title={`${group.label}: ${ms(group.ms)} (${percent(group.percent)})`}
          />
        ))}
        {timing.overhead > 0 && (
          <span className="timing-other" style={{ flexGrow: timing.overhead }} />
        )}
      </div>
      <div className="image-timing-legend">
        {timing.groups.map((group) => (
          <span key={group.key}>
            <i className={`timing-${group.key}`} />
            {group.label}
          </span>
        ))}
      </div>
      <table className="image-timing-table">
        <caption>End-to-end breakdown · phase totals include their indented steps</caption>
        <thead>
          <tr>
            <th scope="col">Phase / step</th>
            <th scope="col">Time</th>
            <th scope="col">% of total</th>
          </tr>
        </thead>
        <tbody>
          {timing.groups.map((group) => (
            <Fragment key={group.key}>
              <tr
                className={`image-timing-phase ${group.key === timing.largest?.key ? 'largest' : ''}`}
              >
                <th scope="row">
                  <i className={`timing-${group.key}`} />
                  {group.label}
                  {(group.key === 'prepare' || group.key === 'decode') &&
                    group.reused !== undefined && (
                      <small>{group.reused ? 'CPU weights reused' : 'First load'}</small>
                    )}
                </th>
                <td>{ms(group.ms)}</td>
                <td>{percent(group.percent)}</td>
              </tr>
              {group.details.map((detail) => (
                <tr key={detail.key} className="image-timing-step">
                  <th scope="row">{detail.label}</th>
                  <td>{ms(detail.ms)}</td>
                  <td>{percent(detail.percent)}</td>
                </tr>
              ))}
            </Fragment>
          ))}
          <tr className="image-timing-phase">
            <th scope="row">Between-phase overhead</th>
            <td>{ms(timing.overhead)}</td>
            <td>{percent(timing.overheadPercent)}</td>
          </tr>
        </tbody>
        <tfoot>
          <tr>
            <th scope="row">Total</th>
            <td>{ms(timing.total)}</td>
            <td>{timing.total > 0 ? '100%' : '—'}</td>
          </tr>
        </tfoot>
      </table>
      <p className="image-timing-note">
        Loading includes model setup, GPU transfers and any compilation during setup. Execution is
        wall time, including transfers needed to return results. Other phase overhead covers work
        such as worker startup, imports, unmeasured preparation and cleanup. Older runs show only
        the details they recorded.
      </p>
      {art.profile && <ProfileResults profile={art.profile} />}
      <details>
        <summary>Denoising steps and measurement details</summary>
        <p>{art.timing_method}</p>
        <p>
          {art.steps} steps · {art.capture ? 'Graph capture enabled' : 'Capture disabled'} ·{' '}
          {art.phases.prefix?.prefix_tokens ?? '—'} prefix tokens
        </p>
        <div className="step-bars">
          {steps.map((duration, index) => (
            <div key={index} title={`Step ${index + 1}: ${ms(duration)}`}>
              <span
                style={{ height: `${Math.max(5, (duration / Math.max(1, ...steps)) * 65)}px` }}
              />
              <small>{index + 1}</small>
            </div>
          ))}
        </div>
        <a href={'/api/images/' + art.id + '/result.json'} download>
          Export full run metadata
        </a>
      </details>
    </section>
  );
}
