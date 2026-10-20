export type ImagePhase = {
  wall_ms?: number;
  total_ms?: number;
  initialization_ms?: number;
  model_load_ms?: number;
  verification_ms?: number;
  verification_cached?: boolean;
  input_preparation_ms?: number;
  inference_ms?: number;
  output_save_ms?: number;
  cleanup_ms?: number;
  profiling_ms?: number;
  components_reused?: boolean;
  step_ms?: number[];
  prefix_tokens?: number;
};

export type TimedImage = {
  id: string;
  request_ms: number;
  steps: number;
  capture: boolean;
  phases: Record<string, ImagePhase>;
  timing_method?: string;
  profile?: ImageProfile | null;
};

export type ImageProfile = {
  cpu_total_ms?: number;
  top_cpu_ops?: { name: string; stage: string; calls: number; total_ms: number; percent: number }[];
  gpu_events: number;
  gpu_total_ms: number;
  top_shaders: { name: string; stage: string; calls: number; total_ms: number; percent: number }[];
  traces: { stage: string; url: string }[];
  warnings: string[];
  summary_url: string;
  scope: string;
};

const phases = [
  ['prepare', 'Text & reference preparation', 'Encode text & references'],
  ['prefix', 'Prefix cache', 'Prefix execution'],
  ['denoise', 'Denoising', 'Denoising steps'],
  ['decode', 'Image decoding & saving', 'VAE execution'],
] as const;
const measured = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0;

export function imageTimingBreakdown(art: TimedImage) {
  const total = measured(art.request_ms) ? art.request_ms : 0;
  const share = (value: number) => (total ? (value / total) * 100 : 0);
  const groups = phases.flatMap(([key, label, execution]) => {
    const phase = art.phases[key];
    const duration = phase?.wall_ms ?? phase?.total_ms;
    if (!phase || !measured(duration)) return [];
    const fields: [keyof ImagePhase, string][] = [
      ['initialization_ms', 'Initialize libraries'],
      [
        'verification_ms',
        phase.verification_cached ? 'Verify model files · cached' : 'Verify model files',
      ],
      ['model_load_ms', 'Load models & set up GPU'],
      ['input_preparation_ms', 'Prepare inputs'],
      ['inference_ms', execution],
      ['output_save_ms', key === 'decode' ? 'Convert & save PNG' : 'Save prepared inputs'],
      ['cleanup_ms', 'Release GPU memory'],
      ['profiling_ms', 'Summarize GPU trace'],
    ];
    const details = fields.flatMap(([field, name]) => {
      const value = phase[field];
      return measured(value) ? [{ key: field, label: name, ms: value, percent: share(value) }] : [];
    });
    const remainder = Math.max(0, duration - details.reduce((sum, detail) => sum + detail.ms, 0));
    return [
      {
        key,
        label,
        ms: duration,
        percent: share(duration),
        reused: phase.components_reused,
        details: [
          ...details,
          {
            key: 'other',
            label: details.length ? 'Other phase overhead' : 'Detailed timings not recorded',
            ms: remainder,
            percent: share(remainder),
          },
        ],
      },
    ];
  });
  const overhead = Math.max(0, total - groups.reduce((sum, group) => sum + group.ms, 0));
  const largest = groups.reduce<(typeof groups)[number] | undefined>(
    (result, group) => (!result || group.ms > result.ms ? group : result),
    undefined,
  );
  return { total, groups, largest, overhead, overheadPercent: share(overhead) };
}
