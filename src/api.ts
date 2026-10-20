import type { JobEvent } from './types';
export async function api<T>(path: string, method = 'GET', data?: unknown): Promise<T> {
  const response = await fetch(path, {
    method,
    headers: data === undefined ? {} : { 'Content-Type': 'application/json' },
    body: data === undefined ? undefined : JSON.stringify(data),
  });
  const body = await response.json();
  if (!response.ok)
    throw new Error(
      typeof body.error === 'string' ? body.error : body.error?.message || 'The request failed.',
    );
  return body;
}
export async function streamJob(id: string, onEvent: (event: JobEvent) => void) {
  const response = await fetch(`/api/jobs/${id}/events`);
  if (!response.ok || !response.body)
    throw new Error('Cannot connect to the inference stream. Check Activity or reconnect.');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '',
    done = false;
  try {
    while (true) {
      const part = await reader.read();
      if (part.done) break;
      buffer += decoder.decode(part.value, { stream: true });
      let boundary: number;
      while ((boundary = buffer.indexOf('\n\n')) >= 0) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const line = block.split('\n').find((l) => l.startsWith('data: '));
        if (!line) continue;
        const event = JSON.parse(line.slice(6)) as JobEvent;
        onEvent(event);
        if (event.type === 'done') done = true;
      }
    }
    if (!done)
      throw new Error(
        'The connection ended before inference completed. Your saved conversation is available after reconnecting.',
      );
  } finally {
    reader.releaseLock();
  }
}
export function download(name: string, value: unknown, mime = 'application/json') {
  const url = URL.createObjectURL(
    new Blob([typeof value === 'string' ? value : JSON.stringify(value, null, 2)], { type: mime }),
  );
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function bytes(n: number) {
  return n >= 1e9
    ? `${(n / 1e9).toFixed(2)} GB`
    : n >= 1e6
      ? `${(n / 1e6).toFixed(0)} MB`
      : n >= 1e3
        ? `${(n / 1e3).toFixed(1)} KB`
        : `${n} B`;
}
export function ms(n?: number | null) {
  return n == null ? '—' : n < 1000 ? `${n.toFixed(1)} ms` : `${(n / 1000).toFixed(2)} s`;
}
export function errorText(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}
