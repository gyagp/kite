// All interactions target a test instance. The deletion fixture is created by this script.
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
const gallery = async () => {
  await page.getByRole('button', { name: 'Multimodal examples', exact: true }).click();
  await page.getByRole('dialog', { name: 'Multimodal experiences' }).waitFor();
};
try {
  await page.goto(base);
  await page.getByRole('heading', { name: 'Meaning, beyond keywords.' }).waitFor();
  assert.ok(
    (await page
      .locator('.nav-item')
      .first()
      .evaluate((e) => parseFloat(getComputedStyle(e).fontSize))) >= 15,
  );
  await gallery();
  assert.equal(await page.locator('.showcase-card').count(), 6);
  await page.locator('.showcase-cover').first().scrollIntoViewIfNeeded();
  await page.waitForFunction(() =>
    [...document.querySelectorAll('.showcase-cover img')]
      .slice(0, 3)
      .every((i) => i.complete && i.naturalWidth > 0),
  );
  await page.locator('.showcase-scroll').evaluate((e) => (e.scrollTop = 0));
  await page.screenshot({ path: 'test-results/orbit-gallery.png' });
  await page.getByRole('button', { name: 'Audio', exact: true }).click();
  assert.equal(await page.locator('.showcase-card').count(), 2);
  await page.getByRole('button', { name: 'Preview Hear it. Find it.', exact: true }).click();
  await page.getByRole('button', { name: 'Play briefing' }).click();
  await page.waitForFunction(
    () => document.querySelector('.showcase-audio audio')?.currentTime > 0.1,
  );
  await page.getByRole('button', { name: 'Pause briefing' }).click();
  await page.getByText('Read the briefing', { exact: true }).click();
  assert.match(await page.locator('.showcase-transcript').innerText(), /astronaut/i);
  await page.screenshot({ path: 'test-results/orbit-audio-preview.png' });
  checks.push(
    'Six photographic experiences, modality filters, real waveform/audio playback and transcript',
  );
  if (infer) {
    await page.getByRole('button', { name: 'Run demo', exact: true }).click();
    await page.locator('.ranking.best').waitFor({ timeout: 180000 });
    assert.equal(await page.locator('.ranking').count(), 4);
    await page.screenshot({ path: 'test-results/orbit-audio-result.png' });
    checks.push('One-click audio-to-image search runs real native inference');
  } else await page.getByRole('button', { name: 'Close dialog' }).click();
  await gallery();
  await page.getByRole('button', { name: 'Preview Twelve seconds to orbit', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('.showcase-film video')?.readyState >= 1);
  const duration = await page.locator('.showcase-film video').evaluate((v) => v.duration);
  assert.ok(duration > 11.9 && duration < 12.1);
  await page.screenshot({ path: 'test-results/orbit-film-preview.png' });
  await page.getByRole('button', { name: 'All experiences', exact: true }).click();
  await page
    .getByRole('button', { name: 'Preview Same world. Different words.', exact: true })
    .click();
  assert.match(await page.locator('.showcase-query textarea').inputValue(), /地球/);
  await page
    .getByRole('button', {
      name: 'The blue Earth covered in white clouds, photographed from space.',
      exact: true,
    })
    .click();
  assert.match(await page.locator('.showcase-query textarea').inputValue(), /blue Earth/);
  await page.getByRole('button', { name: 'Open in workspace', exact: true }).click();
  await page
    .locator('.active-experience strong')
    .filter({ hasText: 'Same world. Different words.' })
    .waitFor();
  assert.match(await page.getByLabel('Search query', { exact: true }).inputValue(), /blue Earth/);
  checks.push(
    '12-second film previews correctly; multilingual prompt alternatives transfer to workspace',
  );
  await page.getByRole('navigation').getByRole('button', { name: 'Chat', exact: true }).click();
  const control = page.getByLabel('Maximum new tokens');
  const current = await (await fetch(base + '/api/preferences/chat')).json();
  assert.equal(await control.inputValue(), String(current.max_tokens));
  await Promise.all([
    page.waitForResponse((r) => r.url().endsWith('/api/bootstrap') && r.status() === 200),
    control.selectOption('1024'),
  ]);
  await page
    .locator('.chat-heading')
    .getByRole('button', { name: 'New chat', exact: true })
    .click();
  assert.equal(await control.inputValue(), '1024');
  await page.reload();
  await page.getByRole('navigation').getByRole('button', { name: 'Chat', exact: true }).click();
  assert.equal(await control.inputValue(), '1024');
  checks.push('Max new tokens persists across new chat and page reload');
  // Make a tiny, explicitly disposable package to exercise the actual disk-delete route.
  const target = path.resolve('test-results', `delete-model-fixture-${Date.now()}`);
  await fs.mkdir(target, { recursive: true });
  await fs.writeFile(path.join(target, 'model.onnx'), 'test fixture');
  await fs.writeFile(path.join(target, 'tokenizer.json'), '{}');
  await fs.writeFile(
    path.join(target, 'genai_config.json'),
    JSON.stringify({ model: { type: 'test', decoder: { filename: 'model.onnx' } } }),
  );
  await fs.writeFile(
    path.join(target, 'kite-model.json'),
    JSON.stringify({ name: 'Disposable UI deletion fixture' }),
  );
  const imported = await (
    await fetch(base + '/api/models/import', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: target }),
    })
  ).json();
  assert.ok(imported.id);
  await page.reload();
  await page
    .getByRole('navigation')
    .getByRole('button', { name: /My models/ })
    .click();
  await page
    .getByRole('button', { name: 'Remove Disposable UI deletion fixture', exact: true })
    .click();
  assert.equal(
    await page.getByRole('button', { name: 'Delete from disk', exact: true }).isDisabled(),
    true,
  );
  await page.getByLabel(/I understand that deleting from disk/).check();
  await page.getByRole('button', { name: 'Delete from disk', exact: true }).click();
  await page.getByRole('dialog').waitFor({ state: 'hidden' });
  await assert.rejects(fs.stat(target));
  checks.push('Confirmed Delete from disk removes only the script-created disposable package');
  await page.setViewportSize({ width: 1000, height: 760 });
  await page
    .getByRole('navigation')
    .getByRole('button', { name: 'Embeddings', exact: true })
    .click();
  await gallery();
  await page.screenshot({ path: 'test-results/orbit-gallery-compact.png' });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  assert.deepEqual(errors, []);
  await fs.writeFile(
    'test-results/orbit-ui-report.json',
    JSON.stringify({ passed: true, inference: infer, checks, errors }, null, 2),
  );
  console.log(JSON.stringify({ passed: true, checks }, null, 2));
} finally {
  await browser.close();
}
