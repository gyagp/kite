// Isolated test browser and disposable model; never deletes a user's package.
import { chromium } from 'playwright';
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';

const base = process.env.KITE_TEST_URL || 'http://127.0.0.1:4327';
const request = async (url, body) => {
  const response = await fetch(
    base + url,
    body === undefined
      ? undefined
      : {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        },
  );
  const result = await response.json();
  assert.ok(response.ok, JSON.stringify(result));
  return result;
};
const folder = path.resolve('test-results/catalog-ui-fixture-' + Date.now());
for (const old of await request('/api/models')) {
  if (
    old.name === 'Catalog deletion fixture' &&
    old.path.startsWith(path.resolve('test-results/catalog-ui-fixture-'))
  ) {
    await fetch(base + '/api/models/' + old.id, { method: 'DELETE' });
  }
}
await fs.mkdir(folder, { recursive: true });
await fs.writeFile(path.join(folder, 'model.onnx'), 'disposable test graph');
await fs.writeFile(path.join(folder, 'tokenizer.json'), '{}');
await fs.writeFile(
  path.join(folder, 'genai_config.json'),
  JSON.stringify({ model: { type: 'test', decoder: { filename: 'model.onnx' } } }),
);
await fs.writeFile(
  path.join(folder, 'kite-model.json'),
  JSON.stringify({
    name: 'Catalog deletion fixture',
    source: { repository: 'webai-community/ai-models', path: 'Qwen3-0.6B/onnx-webgpu' },
  }),
);
const fixture = await request('/api/models/import', { path: folder });
const browser = await chromium.launch({ channel: 'msedge', headless: true });
const page = await browser.newPage({
  viewport: { width: 1460, height: 1040 },
  reducedMotion: 'reduce',
});
const errors = [];
page.on('pageerror', (error) => errors.push(error.message));
const report = { passed: false, checks: [] };
try {
  await page.goto(base);
  await page.getByRole('button', { name: /^My models/ }).click();
  await page.getByRole('heading', { name: 'Good company for your ideas.' }).waitFor();
  const modelCount = (await request('/api/models')).length;
  assert.equal(await page.locator('.model-card').count(), modelCount);
  assert.equal(modelCount, 4); // Three chosen models plus our explicitly imported fixture.
  report.checks.push('Only the three chosen models and explicitly imported fixture are listed');
  await page.getByLabel('Model availability').selectOption('downloaded');
  assert.equal(await page.locator('.model-card').count(), 4);
  await page.getByLabel('Model availability').selectOption('all');
  await page.getByLabel('Search models', { exact: true }).fill('Catalog deletion fixture');
  await page
    .getByRole('button', { name: 'Delete Catalog deletion fixture from disk', exact: true })
    .click();
  await page.getByRole('checkbox').check();
  await page.getByRole('button', { name: 'Delete from disk', exact: true }).click();
  await page.getByRole('dialog').waitFor({ state: 'hidden' });
  const card = page.locator('.model-card');
  await card.getByText('NOT DOWNLOADED', { exact: true }).waitFor();
  assert.equal(await card.count(), 1);
  await assert.rejects(fs.access(folder));
  let models = await request('/api/models');
  assert.equal(models.find((m) => m.id === fixture.id).available, false);
  report.checks.push('Delete removes disposable files while retaining the same model ID and card');
  await page.reload();
  await page.getByRole('button', { name: /^My models/ }).click();
  await page.getByLabel('Search models', { exact: true }).fill('Catalog deletion fixture');
  await page.locator('.model-card').getByRole('button', { name: 'Download', exact: true }).click();
  const dialog = page.getByRole('dialog');
  await dialog
    .getByRole('heading', { name: 'Qwen3-0.6B', exact: true })
    .waitFor({ timeout: 30000 });
  await dialog.locator('.hub-package').first().waitFor({ timeout: 30000 });
  let downloadRequest;
  await page.route('**/api/hub/downloads', async (route) => {
    if (route.request().method() !== 'POST') return route.continue();
    downloadRequest = route.request().postDataJSON();
    await route.fulfill({
      json: {
        id: 'ui-download-fixture',
        name: 'Test',
        variant: 'test',
        status: 'cancelled',
        total_bytes: 0,
        downloaded_bytes: 0,
      },
    });
  });
  await dialog.getByRole('button', { name: 'Download', exact: true }).first().click();
  await page.waitForTimeout(100);
  assert.equal(downloadRequest.model_id, fixture.id);
  assert.ok(downloadRequest.path.startsWith('Qwen3-0.6B/'));
  report.checks.push(
    'Restore opens the correct real Hugging Face package and carries the retained model ID; download POST intercepted to avoid fetching weights',
  );
  await dialog.getByRole('button', { name: 'Close dialog' }).click();
  await page.getByLabel('Search models', { exact: true }).fill('');
  await page.screenshot({ path: 'test-results/model-catalog-desktop.png' });
  await page.setViewportSize({ width: 1000, height: 760 });
  await page.screenshot({ path: 'test-results/model-catalog-compact.png' });
  models = await request('/api/models');
  assert.equal(models.filter((m) => m.architecture === 'EmbeddingGemma2TextModel').length, 0);
  const multimodal = models.find((m) => m.architecture === 'EmbeddingGemma2Model');
  assert.ok(multimodal?.available);
  const job = await request('/api/jobs', {
    kind: 'embed',
    model: multimodal.id,
    samples: ['A blue kite above a mountain.', 'A bright blue kite flying over mountains.'],
    dimensions: 768,
    task: 'SearchQuery',
  });
  let finished;
  const deadline = Date.now() + 120000;
  do {
    await new Promise((resolve) => setTimeout(resolve, 300));
    finished = await request('/api/jobs/' + job.id);
  } while (finished.status === 'running' && Date.now() < deadline);
  assert.equal(finished.status, 'completed', finished.error);
  assert.equal(finished.result.vectors.length, 2);
  assert.equal(finished.result.vectors[0].length, 768);
  assert.ok(finished.result.vectors.flat().every(Number.isFinite));
  report.checks.push(
    'Retained multimodal model produces real 768-dimensional embeddings after removal of text-only exports',
  );
  assert.deepEqual(errors, []);
  report.passed = true;
} finally {
  await browser.close();
  await fetch(base + '/api/models/' + fixture.id, { method: 'DELETE' });
  report.errors = errors;
  await fs.writeFile('test-results/model-catalog-ui-report.json', JSON.stringify(report, null, 2));
}
console.log(JSON.stringify(report, null, 2));
