export type Page = 'chat' | 'embeddings' | 'models' | 'activity' | 'settings' | 'images';
export type Model = {
  id: string;
  name: string;
  path: string;
  kind: 'embedding' | 'chat' | 'image';
  architecture: string;
  modalities: string[];
  size: number;
  precision: string;
  context: number;
  dimensions?: number;
  tasks: string[];
  variant: string;
  source?: { repository?: string; path?: string; revision?: string };
  available: boolean;
  download?: { repository: string; model: string; path: string; revision: string };
  validation?: { passed: boolean; cases: number };
};
export type Settings = { python: string; library: string; asset_root: string; context: number };
export type Runtime = {
  path: string;
  sha256: string;
  host_version: string;
  genai_version?: string;
  build?: { revision?: string };
  devices: { name: string; metadata: Record<string, string> }[];
};
export type Loaded = { model: string; runtime: Runtime; load_ms: number; kind?: string };
export type Status = { loaded: Loaded | null; active: Run | null; worker_running: boolean };
export type Attachment = {
  id: string;
  name: string;
  kind: 'image' | 'audio' | 'video';
  mime: string;
  size: number;
};
export type EmbeddingInput = {
  id?: string;
  label?: string;
  text: string;
  attachments: Attachment[];
};
export type EmbeddingExample = {
  id: string;
  title: string;
  description: string;
  modalities: string[];
  mode: 'search' | 'compare' | 'raw';
  task?: string;
  query: string;
  samples?: EmbeddingInput[];
  query_attachments?: Attachment[];
  cover?: string;
  eyebrow?: string;
  story?: string;
  accent?: string;
  featured?: boolean;
  preview_video?: string;
  preview_audio?: string;
  transcript?: string;
  waveform?: number[];
  audio_duration?: number;
  audio_credit?: string;
  suggestions?: string[];
};
export type EmbeddingTiming = {
  tokens?: number;
  padded_tokens?: number;
  input_count?: number;
  batch_count?: number;
  inference_ms?: number;
  preparation_ms?: number;
  postprocessing_ms?: number;
  total_ms?: number;
};
export type Metrics = EmbeddingTiming & {
  profile?: import('./image-performance').ImageProfile | null;
  text?: string;
  inputs?: { tokens: number; modalities: string[]; video_frames: number[] }[];
  query_inputs?: { tokens: number; modalities: string[]; video_frames: number[] }[];
  search_phases?: { query: EmbeddingTiming; documents: EmbeddingTiming };
  inference_tokens_per_second?: number | null;
  inference_inputs_per_second?: number | null;
  request_ms?: number;
  setup_ms?: number;
  ranking_ms?: number;
  timing_method?: string;
  prompt_tokens?: number;
  completion_tokens?: number;
  ttft_ms?: number | null;
  tokens_per_second?: number | null;
  prefill_ms?: number | null;
  prefill_tokens_per_second?: number | null;
  decode_tokens?: number;
  decode_ms?: number;
  decode_tokens_per_second?: number | null;
  decode_ms_per_token?: number | null;
  first_step_tokens?: number;
  input_preparation_ms?: number;
  generator_setup_ms?: number;
  model_load_ms?: number;
  partial?: boolean;
  dimensions?: number;
  thread_id?: string;
  vectors?: number[][];
  ranking?: {
    text: string;
    label?: string;
    attachments?: Attachment[];
    score: number;
    index: number;
  }[];
};
export type Message = {
  role: 'system' | 'user' | 'assistant';
  content: string;
  attachments?: Attachment[];
  metrics?: Metrics;
  status?: string;
  error?: string;
};
export type Thread = {
  id: string;
  title: string;
  model: string;
  messages: Message[];
  updated: number;
};
export type Run = {
  id: string;
  kind: string;
  model: string;
  status: string;
  created: number;
  elapsed_ms?: number;
  result?: Metrics;
  error?: string;
};
export type Bootstrap = {
  models: Model[];
  settings: Settings;
  chat_preferences?: { max_tokens: number };
  status: Status;
  threads: Thread[];
  api_key: string;
  api_url: string;
  data_directory: string;
};
export type JobEvent = {
  type: string;
  stage?: string;
  text?: string;
  message?: string;
  data?: Metrics;
  id?: string;
  completed?: number;
  total?: number;
  status?: string;
};
export type Execute = (
  data: Record<string, unknown>,
  onEvent?: (event: JobEvent) => void,
) => Promise<Metrics | undefined>;
declare global {
  interface Window {
    kiteDesktop?: { selectPath: (kind: 'folder' | 'runtime' | 'python') => Promise<string | null> };
  }
}
