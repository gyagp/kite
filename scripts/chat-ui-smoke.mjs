// Deterministic rendering fixture; no model response or measured performance is claimed.
import { chromium } from 'playwright';
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';

const base = process.env.KITE_TEST_URL || 'http://127.0.0.1:4317';
const boot = await (await fetch(`${base}/api/bootstrap`)).json();
const model = boot.models.find((m) => m.kind === 'chat');
const markdown = [
  '# Markdown in Kite',
  '',
  'Readable **bold**, *italic*, ~~strikethrough~~, and `inline code`.',
  '',
  '## A small example',
  '',
  '```python',
  'def sum_even(values):',
  '    # Keep the even numbers',
  '    return sum(x for x in values if x % 2 == 0)',
  '```',
  '',
  '| Phase | Tokens |',
  '| :--- | ---: |',
  '| Prompt | 100 |',
  '| Output | 5 |',
  '',
  '> Everything runs locally.',
  '',
  '1. Load a model',
  '2. Write a prompt',
  '',
  '- [x] Markdown formatting',
  '- [ ] Your next idea',
  '',
  '[Documentation](https://example.com) and [unsafe](javascript:alert(1)).',
  '',
  '<script>window.untrusted = true</script>',
].join('\n');
const metrics = {
  prompt_tokens: 100,
  completion_tokens: 5,
  first_step_tokens: 1,
  prefill_ms: 500,
  prefill_tokens_per_second: 200,
  ttft_ms: 600,
  decode_tokens: 4,
  decode_ms: 400,
  decode_tokens_per_second: 10,
  decode_ms_per_token: 100,
  total_ms: 1300,
  input_preparation_ms: 20,
  generator_setup_ms: 10,
  model_load_ms: 0,
};
const fixture = {
  id: 'markdown-fixture',
  title: 'Markdown and performance preview',
  model: model.id,
  updated: Date.now() / 1000,
  messages: [
    { role: 'user', content: 'Show a Markdown example.' },
    { role: 'assistant', content: markdown, status: 'completed', metrics },
  ],
};
const browser = await chromium.launch({ channel: 'msedge', headless: true });
const page = await browser.newPage({
  viewport: { width: 1460, height: 1040 },
  permissions: ['clipboard-read', 'clipboard-write'],
});
const errors = [];
page.on('pageerror', (error) => errors.push(error.message));
await page.route('**/api/bootstrap', (route) =>
  route.fulfill({
    json: {
      ...boot,
      status: { loaded: null, active: null, worker_running: false },
      threads: [fixture],
    },
  }),
);
try {
  await page.goto(base);
  await page.getByRole('navigation').getByRole('button', { name: 'Chat', exact: true }).click();
  await page.locator('.thread-link').filter({ hasText: fixture.title }).click();
  const content = page.locator('.message.assistant .markdown');
  assert.equal(await content.locator('h1').innerText(), 'Markdown in Kite');
  assert.equal(await content.locator('strong').innerText(), 'bold');
  assert.equal(await content.locator('em').innerText(), 'italic');
  assert.equal(await content.locator('del').innerText(), 'strikethrough');
  assert.ok((await content.locator('.hljs-keyword').count()) > 0);
  assert.equal(await content.locator('table tbody tr').count(), 2);
  assert.equal(await content.locator('ol > li').count(), 2);
  assert.equal(await content.locator('input[type=checkbox]:checked').count(), 1);
  assert.equal(await content.locator('script').count(), 0);
  assert.equal(await page.evaluate(() => window.untrusted), undefined);
  assert.equal(
    await content.getByRole('link', { name: 'Documentation' }).getAttribute('rel'),
    'noopener noreferrer',
  );
  assert.equal(await content.locator('a[href^="javascript:"]').count(), 0);
  await content.getByRole('button', { name: 'Copy code', exact: true }).click();
  assert.equal(
    (await page.evaluate(() => navigator.clipboard.readText())).replace(/\r\n/g, '\n'),
    'def sum_even(values):\n    # Keep the even numbers\n    return sum(x for x in values if x % 2 == 0)',
  );
  await page.locator('.message.assistant').scrollIntoViewIfNeeded();
  await content.locator('h1').scrollIntoViewIfNeeded();
  await page.screenshot({ path: 'test-results/chat-markdown.png' });
  const stats = page.getByRole('region', { name: 'Response performance' });
  assert.match(await stats.innerText(), /200 tok\/s/);
  assert.match(await stats.innerText(), /10 tok\/s/);
  await stats.getByText('Timing details').click();
  assert.match(await stats.innerText(), /Already loaded/);
  assert.match(await stats.innerText(), /100.0 ms \/ token/);
  await stats.scrollIntoViewIfNeeded();
  await page.screenshot({ path: 'test-results/chat-performance.png' });
  // Stream an unfinished fence, then complete it. Neither state can crash the renderer.
  await page.route('**/api/jobs', (route) =>
    route.fulfill({ status: 202, json: { id: 'fixture-stream' } }),
  );
  const events = [
    { type: 'thread', id: fixture.id },
    { type: 'delta', text: '## Streaming\n\n```python\ndef answer():' },
    {
      type: 'metrics',
      data: { ...metrics, completion_tokens: 1, decode_tokens: 0, decode_tokens_per_second: null },
    },
    { type: 'delta', text: '\n    return 42\n```' },
    {
      type: 'result',
      data: { ...metrics, text: '## Streaming\n\n```python\ndef answer():\n    return 42\n```' },
    },
    { type: 'done', status: 'completed' },
  ];
  await page.route('**/api/jobs/fixture-stream/events', (route) =>
    route.fulfill({
      contentType: 'text/event-stream',
      body: events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join(''),
    }),
  );
  await page.getByLabel('Message', { exact: true }).fill('Another example');
  await page.getByRole('button', { name: 'Send message' }).click();
  await page
    .locator('.message.assistant')
    .last()
    .getByRole('heading', { name: 'Streaming' })
    .waitFor();
  assert.match(
    await page.locator('.message.assistant').last().locator('pre').innerText(),
    /return 42/,
  );
  assert.equal(await page.getByRole('region', { name: 'Response performance' }).count(), 2);
  assert.deepEqual(errors, []);
  const report = {
    passed: true,
    scope: 'Deterministic UI fixture; numbers are test values, not benchmark results',
    checks: [
      'Markdown headings, emphasis, inline/fenced code, highlighting, lists, tables and tasks',
      'Code copies exact plain text; unsafe markup and links do not execute',
      'Prefill/decode metrics, timing details and loaded-model state',
      'Streamed Markdown and statistics render without exceptions',
    ],
  };
  await fs.writeFile('test-results/chat-ui-report.json', JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} finally {
  await browser.close();
}
