import { ProfileResults } from './ProfileResults';
import { Activity, ChevronDown } from 'lucide-react';
import { ms } from './api';
import type { Metrics } from './types';

const rate = (n?: number | null) =>
  n == null ? '—' : n.toLocaleString(undefined, { maximumFractionDigits: 1 });
export function ChatStats({ metrics, running }: { metrics: Metrics; running: boolean }) {
  const hasPhases = metrics.prefill_ms != null;
  return (
    <section className="chat-performance" aria-label="Response performance">
      <div className="performance-heading">
        <span>
          <Activity size={13} />
          Response performance
        </span>
        <span>
          {running ? 'Live' : metrics.partial ? 'Partial · stopped' : 'Measured this response'}
        </span>
      </div>
      <div className="performance-grid">
        <div>
          <span>Prefill</span>
          <strong>
            {rate(metrics.prefill_tokens_per_second)} <small>tok/s</small>
          </strong>
          <p>
            {hasPhases ? `${ms(metrics.prefill_ms!)} · ` : ''}
            {metrics.prompt_tokens?.toLocaleString() ?? '—'} input tokens
          </p>
        </div>
        <div>
          <span>Decode</span>
          <strong>
            {rate(metrics.decode_tokens_per_second ?? metrics.tokens_per_second)}{' '}
            <small>tok/s</small>
          </strong>
          <p>
            {metrics.decode_ms != null
              ? `${ms(metrics.decode_ms)} · ${metrics.decode_tokens ?? 0} tokens after first step`
              : 'After the first generated step'}
          </p>
        </div>
        <div>
          <span>First token</span>
          <strong>{metrics.ttft_ms == null ? '—' : ms(metrics.ttft_ms)}</strong>
          <p>From inference start</p>
        </div>
        <div>
          <span>Output</span>
          <strong>
            {metrics.completion_tokens?.toLocaleString() ?? 0} <small>tokens</small>
          </strong>
          <p>
            {metrics.total_ms != null
              ? `${ms(metrics.total_ms)} generation`
              : 'Generation in progress'}
          </p>
        </div>
      </div>
      <details className="performance-details">
        <summary>
          Timing details <ChevronDown size={12} />
        </summary>
        <dl>
          <div>
            <dt>Input preparation</dt>
            <dd>{ms(metrics.input_preparation_ms)}</dd>
          </div>
          <div>
            <dt>Generator setup</dt>
            <dd>{ms(metrics.generator_setup_ms)}</dd>
          </div>
          <div>
            <dt>Model load this request</dt>
            <dd>
              {metrics.model_load_ms == null
                ? '—'
                : metrics.model_load_ms === 0
                  ? 'Already loaded'
                  : ms(metrics.model_load_ms)}
            </dd>
          </div>
          <div>
            <dt>Average decode latency</dt>
            <dd>
              {metrics.decode_ms_per_token == null
                ? '—'
                : `${ms(metrics.decode_ms_per_token)} / token`}
            </dd>
          </div>
        </dl>
        <p>
          {hasPhases
            ? 'Prefill includes input upload, prompt evaluation, and a synchronized logits readback. First-token latency also includes the first sampling step. Decode is wall time between first and last token readbacks, including text decoding and streaming overhead; the first step is excluded. Generation excludes model loading and input preparation. Token counts can include EOS and other special tokens.'
            : 'This older response recorded first-token latency and decode throughput only. Send a new message for separate prefill statistics.'}
        </p>
        <p>
          Interactive measurements on your current workload; not GPU kernel timings or controlled
          benchmarks.
        </p>
      </details>
      {metrics.profile && <ProfileResults profile={metrics.profile} />}
    </section>
  );
}
