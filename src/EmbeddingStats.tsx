import { ProfileResults } from './ProfileResults';
import { Activity, ChevronDown } from 'lucide-react';
import { ms } from './api';
import type { Metrics } from './types';
import './embedding-stats.css';

const number = (value?: number | null) =>
  value == null || !Number.isFinite(value)
    ? '—'
    : value.toLocaleString(undefined, { maximumFractionDigits: 1 });

export function EmbeddingStats({ metrics }: { metrics: Metrics }) {
  const inputs = [...(metrics.query_inputs || []), ...(metrics.inputs || [])];
  const frames = inputs.reduce(
    (total, input) => total + input.video_frames.reduce((a, b) => a + b, 0),
    0,
  );
  const measured = metrics.request_ms != null;
  return (
    <section className="embedding-performance" aria-label="Embedding performance">
      <div className="performance-heading">
        <span>
          <Activity size={15} />
          Embedding performance
        </span>
        <span>Measured this run</span>
      </div>
      <div className="performance-grid">
        <div>
          <span>Inference throughput</span>
          <strong>
            {number(metrics.inference_tokens_per_second)} <small>tok/s</small>
          </strong>
          <p>{number(metrics.inference_inputs_per_second)} inputs/s</p>
        </div>
        <div>
          <span>Inference</span>
          <strong>{ms(metrics.inference_ms)}</strong>
          <p>Model execution & readback</p>
        </div>
        <div>
          <span>Preprocessing</span>
          <strong>{ms(metrics.preparation_ms)}</strong>
          <p>Tokenization & media preparation</p>
        </div>
        <div>
          <span>Total request</span>
          <strong>{ms(metrics.request_ms)}</strong>
          <p>Includes model setup when needed</p>
        </div>
      </div>
      <div className="embedding-workload">
        <span>
          <b>{number(metrics.input_count)}</b> inputs embedded
        </span>
        <span>
          <b>{number(metrics.tokens)}</b> input tokens
        </span>
        <span>
          <b>{number(metrics.dimensions)}</b> dimensions
        </span>
        <span>
          <b>{number(metrics.batch_count)}</b> batches
        </span>
        {frames > 0 && (
          <span>
            <b>{number(frames)}</b> video frames
          </span>
        )}
      </div>
      <details className="performance-details">
        <summary>
          Timing details <ChevronDown size={13} />
        </summary>
        <dl>
          <div>
            <dt>Model setup this run</dt>
            <dd>{metrics.setup_ms === 0 ? 'Already loaded' : ms(metrics.setup_ms)}</dd>
          </div>
          <div>
            <dt>Native model loading</dt>
            <dd>{metrics.model_load_ms === 0 ? 'Already loaded' : ms(metrics.model_load_ms)}</dd>
          </div>
          <div>
            <dt>Output processing</dt>
            <dd>{ms(metrics.postprocessing_ms)}</dd>
          </div>
          <div>
            <dt>Encoding total</dt>
            <dd>{ms(metrics.total_ms)}</dd>
          </div>
          {metrics.ranking_ms != null && (
            <div>
              <dt>Similarity & ranking</dt>
              <dd>{ms(metrics.ranking_ms)}</dd>
            </div>
          )}
          <div>
            <dt>Token slots including padding</dt>
            <dd>{number(metrics.padded_tokens)}</dd>
          </div>
        </dl>
        {metrics.search_phases && (
          <div className="embedding-phase-table">
            <table aria-label="Search performance breakdown">
              <thead>
                <tr>
                  <th>Search phase</th>
                  <th>Inputs</th>
                  <th>Tokens</th>
                  <th>Preprocess</th>
                  <th>Inference</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(metrics.search_phases).map(([phase, timing]) => (
                  <tr key={phase}>
                    <th scope="row">{phase === 'query' ? 'Query' : 'Documents'}</th>
                    <td>{number(timing.input_count)}</td>
                    <td>{number(timing.tokens)}</td>
                    <td>{ms(timing.preparation_ms)}</td>
                    <td>{ms(timing.inference_ms)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p>
          {measured
            ? 'Throughput uses non-padding input tokens, including task prefixes and expanded media tokens, divided by inference time. Search totals include both the query and documents. Embeddings use forward inference and have no decode phase.'
            : 'This earlier run has limited timing data. Run it again to record throughput and a full timing breakdown.'}
        </p>
        <p>
          Inference includes model execution, CPU readback, and multimodal encoder orchestration.
          Output processing covers truncation, normalization, validation, and vector conversion.
          Model setup includes worker startup and loading; total request measures backend work
          through completion, excluding uploads and browser rendering.
        </p>
      </details>
      {metrics.profile && <ProfileResults profile={metrics.profile} />}
    </section>
  );
}
