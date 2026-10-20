import { chromium } from 'playwright';
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';

const base = process.env.KITE_TEST_URL || 'http://127.0.0.1:4320';
const infer = process.argv.includes('--inference');
const browser = await chromium.launch({ channel: 'msedge', headless: true });
const page = await browser.newPage({
  viewport: { width: 1460, height: 1040 },
  reducedMotion: 'reduce',
});
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
const checks = [];
const example = async (title) => {
  await page.getByRole('button', { name: 'Multimodal examples', exact: true }).click();
  await page.getByRole('button', { name: `Preview ${title}`, exact: true }).click();
  await page.getByRole('button', { name: 'Open in workspace', exact: true }).click();
  await page.locator('.active-experience strong').filter({ hasText: title }).waitFor();
};
try {
  await page.goto(base);
  await page.getByRole('heading', { name: 'Meaning, beyond keywords.' }).waitFor();
  assert.equal(
    await page.evaluate(() => getComputedStyle(document.documentElement).colorScheme),
    'light',
  );
  await page.screenshot({ path: 'test-results/light-workspace.png' });
  await page.getByRole('button', { name: 'Multimodal examples', exact: true }).click();
  await page.locator('.showcase-card').first().waitFor();
  assert.equal(await page.locator('.showcase-card').count(), 6);
  await page.screenshot({ path: 'test-results/multimodal-examples.png' });
  await page
    .getByRole('button', { name: 'Preview A planet without a keyword', exact: true })
    .click();
  await page.getByRole('button', { name: 'Open in workspace', exact: true }).click();
  await page
    .locator('.active-experience strong')
    .filter({ hasText: 'A planet without a keyword' })
    .waitFor();
  assert.equal(await page.locator('.document-list .embedding-media img').count(), 4);
  assert.match(
    await page.getByRole('combobox', { name: 'Select model' }).innerText(),
    /Multimodal/,
  );
  await page.screenshot({ path: 'test-results/image-example.png' });
  checks.push('Light theme and six ready-to-run multimodal example presets');
  if (infer) {
    await page.getByRole('button', { name: 'Find matches', exact: true }).click();
    await page.locator('.ranking.best').waitFor({ timeout: 180000 });
    assert.equal(await page.locator('.ranking').count(), 4);
    await page.screenshot({ path: 'test-results/image-search-result.png' });
    checks.push('Real image/text search renders scores and image result previews');
  }
  await example('Hear it. Find it.');
  assert.equal(await page.locator('.query-media audio').count(), 1);
  assert.equal(await page.getByLabel('Search query', { exact: true }).inputValue(), '');
  assert.equal(
    await page.getByRole('button', { name: 'Find matches', exact: true }).isEnabled(),
    true,
  );
  await example('Twelve seconds to orbit');
  await page.locator('.document-list video').waitFor();
  await page.waitForFunction(() => {
    const v = document.querySelector('.document-list video');
    return v && v.readyState >= 1;
  });
  const duration = await page.locator('.document-list video').evaluate((v) => v.duration);
  assert.ok(duration >= 11.9 && duration <= 12.1);
  await page.screenshot({ path: 'test-results/video-example.png' });
  checks.push(
    'Media-only audio query is valid; H.264 video preview loads a twelve-second duration',
  );
  await example('One mission. Every dimension.');
  assert.equal(await page.locator('.embedding-input-card').count(), 5);
  assert.equal(
    await page.locator('.embedding-input-card').first().locator('img,video,audio').count(),
    3,
  );
  if (infer) {
    await page.getByRole('button', { name: 'Compare inputs', exact: true }).click();
    await page.locator('.matrix').waitFor({ timeout: 180000 });
    assert.equal(await page.locator('.matrix-row > div').count(), 25);
    await page.screenshot({ path: 'test-results/mixed-similarity-result.png' });
    checks.push('Real mixed text/image/audio/video embeddings render a 5×5 similarity matrix');
  }
  await page.getByRole('tab', { name: 'Raw embeddings' }).click();
  await page
    .getByLabel('Attach media to input 2', { exact: true })
    .setInputFiles(path.resolve('demo/tree.png'));
  await page
    .locator('.embedding-input-card')
    .nth(1)
    .getByRole('img', { name: 'tree.png' })
    .waitFor();
  checks.push('Local file selection uploads and previews an image in a mixed embedding input');
  await page
    .getByRole('navigation')
    .getByRole('button', { name: /My models/ })
    .click();
  await page.getByRole('button', { name: 'Discover models', exact: true }).click();
  await page.getByLabel('Search Hugging Face models').waitFor({ timeout: 60000 });
  await page.getByLabel('Search Hugging Face models').fill('SmolLM2-135M');
  await page.getByRole('button', { name: 'SmolLM2-135M-Instruct', exact: true }).click();
  await page.locator('.hub-package').waitFor({ timeout: 60000 });
  assert.match(await page.locator('.hub-package').innerText(), /onnx-webgpu/);
  await page.screenshot({ path: 'test-results/huggingface-model-browser.png' });
  checks.push('Live Hugging Face model search shows a pinned ONNX package and download size');
  await page.getByRole('button', { name: 'Close dialog' }).click();
  await page.setViewportSize({ width: 1000, height: 760 });
  await page
    .getByRole('navigation')
    .getByRole('button', { name: 'Embeddings', exact: true })
    .click();
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  assert.deepEqual(errors, []);
  const report = { passed: true, inference: infer, checks, errors };
  await fs.writeFile('test-results/multimodal-ui-report.json', JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} finally {
  await browser.close();
}
