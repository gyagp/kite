// Test-owned headless browser; never attaches to a user's browser/profile.
import { chromium } from 'playwright';
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';

const base = process.env.KITE_TEST_URL || 'http://127.0.0.1:4317';
const infer = process.argv.includes('--inference');
await fs.mkdir('test-results', { recursive: true });
const browser = await chromium.launch({ channel: 'msedge', headless: true });
const context = await browser.newContext({
  viewport: { width: 1460, height: 1040 },
  reducedMotion: 'reduce',
});
const page = await context.newPage();
const errors = [];
page.on('pageerror', (error) => errors.push(error.message));
const checks = [];
const screenshot = (name) => page.screenshot({ path: `test-results/${name}.png`, fullPage: true });
try {
  await page.goto(base);
  await page.getByRole('heading', { name: 'Meaning, beyond keywords.' }).waitFor();
  const boot = await (await fetch(`${base}/api/bootstrap`)).json();
  const textEncoder = boot.models.find((m) => m.kind === 'embedding' && m.modalities.length === 1);
  await page.getByRole('combobox', { name: 'Select model' }).selectOption(textEncoder.id);
  await page.getByRole('button', { name: 'Find matches' }).waitFor();
  await screenshot('embeddings');
  checks.push('Embedding workspace renders with discovered model');
  await page.getByRole('tab', { name: 'Compare texts' }).click();
  await page.getByRole('button', { name: 'Compare texts', exact: true }).waitFor();
  await page.getByRole('tab', { name: 'Raw embeddings' }).click();
  await page.getByRole('button', { name: 'Generate embeddings' }).waitFor();
  await page.getByRole('tab', { name: 'Semantic search' }).click();
  checks.push('Embedding modes switch');
  if (infer) {
    await page.getByRole('button', { name: 'Find matches' }).click();
    await page.locator('.ranking.best').waitFor({ timeout: 120000 });
    assert.match(await page.locator('.ranking.best').innerText(), /Input 1/);
    await screenshot('semantic-search-result');
    const download = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Export results as JSON' }).click();
    assert.equal((await download).suggestedFilename(), 'kite-embeddings.json');
    checks.push('Native semantic search ranks aurora first and exports JSON');
    await page.getByRole('tab', { name: 'Compare texts' }).click();
    await page.getByRole('button', { name: 'Compare texts', exact: true }).click();
    await page.locator('.matrix').waitFor({ timeout: 60000 });
    assert.equal(await page.locator('.matrix-row > div').count(), 9);
    checks.push('Real embedding cosine matrix renders');
  }
  await page.getByRole('button', { name: /^My models/ }).click();
  await page.getByRole('heading', { name: 'Good company for your ideas.' }).waitFor();
  assert.equal(await page.locator('.model-card').count(), boot.models.length);
  await screenshot('models');
  await page.getByRole('button', { name: 'Import model', exact: true }).click();
  await page.getByLabel('Package or artifacts directory').fill('Z:\\not-a-package');
  await page.getByRole('button', { name: 'Import this package' }).click();
  await page.getByRole('alert').filter({ hasText: 'existing model package directory' }).waitFor();
  await page.getByRole('button', { name: 'Close dialog' }).click();
  checks.push('Model library and actionable invalid import errors');
  await page.getByRole('navigation').getByRole('button', { name: 'Chat', exact: true }).click();
  await page.getByRole('heading', { name: 'A little intelligence. Entirely yours.' }).waitFor();
  assert.equal(await page.getByRole('button', { name: 'Attach media' }).isDisabled(), true);
  await screenshot('chat');
  checks.push('Chat renders and text-only exports disable media input');
  if (infer) {
    await page
      .getByLabel('Message', { exact: true })
      .fill('What is the capital of France? Answer with just the city name.');
    await page.getByRole('button', { name: 'Send message' }).click();
    await page
      .locator('.message.assistant .markdown')
      .filter({ hasText: 'Paris' })
      .waitFor({ timeout: 120000 });
    await page.getByRole('button', { name: 'Regenerate response' }).waitFor();
    await screenshot('chat-result');
    const title = await page.locator('.chat-title').innerText();
    await page.reload();
    await page.getByRole('navigation').getByRole('button', { name: 'Chat', exact: true }).click();
    await page.locator('.thread-link').filter({ hasText: title }).first().click();
    await page.locator('.message.assistant .markdown').filter({ hasText: 'Paris' }).waitFor();
    checks.push('Qwen streams real response; conversation persists across reload');
  }
  await page.getByRole('button', { name: 'Activity', exact: true }).click();
  await page.getByRole('heading', { name: 'See your models at work.' }).waitFor();
  await screenshot('activity');
  await page.getByRole('button', { name: 'Settings', exact: true }).click();
  await page.getByLabel('Inference Python executable').waitFor();
  assert.match(
    await page.getByLabel('WebGPU provider DLL').inputValue(),
    /onnxruntime_providers_webgpu\.dll$/,
  );
  await screenshot('settings');
  checks.push('Activity and runtime settings render');
  await page.setViewportSize({ width: 1000, height: 760 });
  await page
    .getByRole('navigation')
    .getByRole('button', { name: 'Embeddings', exact: true })
    .click();
  await screenshot('embeddings-compact');
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
  assert.equal(overflow, false, 'No horizontal overflow at compact desktop size');
  assert.deepEqual(errors, [], 'No renderer exceptions');
  checks.push('Compact desktop layout; no renderer exceptions');
  await fs.writeFile(
    'test-results/ui-report.json',
    JSON.stringify({ passed: true, inference: infer, checks, errors }, null, 2),
  );
  console.log(JSON.stringify({ passed: true, checks }, null, 2));
} finally {
  await browser.close();
}
