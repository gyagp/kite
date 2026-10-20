// Launch only the app we built, with disposable state and an invisible window.
import { _electron as electron } from 'playwright';
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';

await fs.mkdir('test-results', { recursive: true });
const data = path.resolve('test-results', `desktop-state-${Date.now()}`);
const launch = () =>
  electron.launch({
    executablePath: path.resolve('release/win-unpacked/Kite.exe'),
    env: { ...process.env, KITE_DATA_DIR: data, KITE_TEST_HEADLESS: '1' },
    timeout: 60000,
  });
const app = await launch();
let url;
const checks = [];
try {
  const page = await app.firstWindow();
  await page
    .getByRole('heading', { name: 'Meaning, beyond keywords.' })
    .waitFor({ timeout: 60000 });
  url = new URL(page.url()).origin;
  assert.equal(await page.evaluate(() => typeof window.kiteDesktop?.selectPath), 'function');
  assert.equal(await page.evaluate(() => typeof window.require), 'undefined');
  checks.push('Packaged Windows app launches; sandboxed renderer and file-picker bridge available');
  await page.getByRole('button', { name: 'Find matches' }).click();
  await page.locator('.ranking.best').waitFor({ timeout: 120000 });
  assert.match(await page.locator('.ranking.best').innerText(), /Input 1/);
  const png = await app.evaluate(async ({ BrowserWindow }) =>
    (await BrowserWindow.getAllWindows()[0].capturePage(undefined, { stayHidden: true }))
      .toPNG()
      .toString('base64'),
  );
  await fs.writeFile('test-results/desktop.png', Buffer.from(png, 'base64'));
  checks.push('Packaged backend discovers Evo and performs real WebGPU semantic search');
  const status = await (await fetch(`${url}/api/status`)).json();
  assert.match(status.loaded.runtime.path, /ort-webgpu-main-86d60730c1/);
  checks.push('Correct source-built WebGPU DLL loaded');
  await page.getByRole('button', { name: 'Multimodal examples', exact: true }).click();
  await page.locator('.showcase-card').first().waitFor();
  assert.equal(await page.locator('.showcase-card').count(), 6);
  await page
    .getByRole('button', { name: 'Run One mission. Every dimension.', exact: true })
    .click();
  await page.locator('.matrix').waitFor({ timeout: 180000 });
  assert.equal(await page.locator('.matrix-row > div').count(), 25);
  await page.locator('.matrix').scrollIntoViewIfNeeded();
  assert.equal(
    await page.evaluate(() => getComputedStyle(document.documentElement).colorScheme),
    'light',
  );
  const multimodalPng = await app.evaluate(async ({ BrowserWindow }) =>
    (await BrowserWindow.getAllWindows()[0].capturePage(undefined, { stayHidden: true }))
      .toPNG()
      .toString('base64'),
  );
  await fs.writeFile('test-results/desktop-multimodal.png', Buffer.from(multimodalPng, 'base64'));
  checks.push(
    'Packaged demo assets, light theme, and text/image/audio/video embedding inference produce a real 5×5 matrix',
  );
  await page.getByRole('navigation').getByRole('button', { name: 'Chat', exact: true }).click();
  const tokens = page.getByLabel('Maximum new tokens');
  assert.equal(await tokens.inputValue(), '4096');
  await Promise.all([
    page.waitForResponse((r) => r.url().endsWith('/api/bootstrap') && r.status() === 200),
    tokens.selectOption('1024'),
  ]);
  await page
    .locator('.chat-heading')
    .getByRole('button', { name: 'New chat', exact: true })
    .click();
  assert.equal(await tokens.inputValue(), '1024');
  await page
    .getByLabel('Message', { exact: true })
    .fill(
      "Use exactly a Markdown level-two heading 'Example', then a fenced python code block defining add(a, b). No explanation.",
    );
  await page.getByRole('button', { name: 'Send message' }).click();
  await page.getByRole('button', { name: 'Regenerate response' }).waitFor({ timeout: 120000 });
  await page.locator('.message.assistant .markdown pre').waitFor();
  const perf = page.getByRole('region', { name: 'Response performance' });
  assert.match(await perf.innerText(), /Prefill/);
  assert.match(await perf.innerText(), /Decode/);
  const threads = await (await fetch(`${url}/api/threads`)).json();
  const metrics = threads[0].messages.at(-1).metrics;
  assert.ok(metrics.prefill_ms > 0 && metrics.decode_tokens_per_second > 0);
  assert.ok(metrics.prefill_ms <= metrics.ttft_ms);
  await perf.scrollIntoViewIfNeeded();
  const chatPng = await app.evaluate(async ({ BrowserWindow }) =>
    (await BrowserWindow.getAllWindows()[0].capturePage(undefined, { stayHidden: true }))
      .toPNG()
      .toString('base64'),
  );
  await fs.writeFile('test-results/desktop-chat.png', Buffer.from(chatPng, 'base64'));
  await fs.writeFile('test-results/desktop-chat-metrics.json', JSON.stringify(metrics, null, 2));
  checks.push(
    'Packaged chat renders real generated Markdown and persists measured prefill/decode statistics',
  );
} finally {
  await app.close();
}
let stopped = false;
for (let n = 0; n < 50; n++) {
  try {
    await fetch(`${url}/api/health`, { signal: AbortSignal.timeout(500) });
  } catch {
    stopped = true;
    break;
  }
  await new Promise((r) => setTimeout(r, 100));
}
assert.ok(stopped, 'Desktop service exits with its window');
checks.push('Desktop shutdown releases its local HTTP service');
const restarted = await launch();
try {
  const page = await restarted.firstWindow();
  await page
    .getByRole('heading', { name: 'Meaning, beyond keywords.' })
    .waitFor({ timeout: 60000 });
  await page.getByRole('navigation').getByRole('button', { name: 'Chat', exact: true }).click();
  assert.equal(await page.getByLabel('Maximum new tokens').inputValue(), '1024');
  const preferences = await (await fetch(new URL('/api/preferences/chat', page.url()))).json();
  assert.equal(preferences.max_tokens, 1024);
  checks.push(
    'Default 4096 tokens; changed limit survives a new chat and a full packaged-app restart',
  );
} finally {
  await restarted.close();
}
await fs.writeFile(
  'test-results/desktop-report.json',
  JSON.stringify({ passed: true, checks }, null, 2),
);
console.log(JSON.stringify({ passed: true, checks }, null, 2));
