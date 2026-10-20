// Real packaged inference, with a fresh browser profile and no user browser attachment.
import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';

const base = process.env.KITE_TEST_URL;
assert.ok(base, 'Set KITE_TEST_URL to the running Kite backend');
const browser = await chromium.launch({ channel: 'msedge', headless: true });
const page = await browser.newPage({
  viewport: { width: 1460, height: 1080 },
  reducedMotion: 'reduce',
});
const report = { passed: false, checks: [] };
const errors = [];
page.on('pageerror', (error) => errors.push(error.message));
try {
  await page.goto(base + '/#images');
  await page.getByLabel('Image canvas').selectOption('512x512');
  await page
    .getByLabel('Image prompt')
    .fill(
      'A tiny cobalt blue glass kite floating above a white marble table, soft morning sunlight, delicate reflections, minimalist product photograph.',
    );
  await page.getByLabel('Image seed').fill('181104');
  await page.getByRole('checkbox', { name: /Profile generation/ }).check();
  const submitted = page.waitForResponse(
    (response) => response.url().endsWith('/api/jobs') && response.request().method() === 'POST',
  );
  await page.getByRole('button', { name: 'Generate image', exact: true }).click();
  const job = await (await submitted).json();
  report.job = job.id;
  const progress = page.getByRole('region', { name: 'Image generation progress' });
  await progress.waitFor();
  assert.equal(await progress.locator('li').count(), 4);
  assert.equal(await progress.locator('progress').getAttribute('value'), null);
  await progress.getByText('Stage 1 of 4', { exact: true }).waitFor();
  await page.screenshot({ path: 'test-results/image-progress-loading.png' });
  await page.getByRole('button', { name: 'Activity', exact: true }).click();
  await page.getByRole('button', { name: 'Image Studio', exact: true }).click();
  await progress.waitFor();
  await page.getByRole('button', { name: /^Your gallery/ }).click();
  await progress.waitFor();
  await page.getByRole('button', { name: 'Create', exact: true }).click();
  report.checks.push(
    'Stage progress and elapsed clock remain available across navigation and gallery',
  );
  await page.setViewportSize({ width: 1000, height: 900 });
  await progress.scrollIntoViewIfNeeded();
  assert.ok(await progress.evaluate((element) => element.scrollWidth <= element.clientWidth));
  await page.screenshot({ path: 'test-results/image-progress-compact.png' });
  await progress.waitFor({ state: 'hidden', timeout: 240000 });
  const result = (await (await fetch(base + '/api/jobs/' + job.id)).json()).result;
  assert.ok(result?.profile?.gpu_events > 0);
  for (const stage of ['prefix', 'denoise']) {
    assert.equal(result.phases[stage].runtime.provider_options.dawnBackendType, 'D3D12');
  }
  assert.equal(result.profile.traces.length, 2);
  assert.deepEqual(result.profile.warnings, []);
  const profile = page.getByRole('region', { name: 'GPU profile' });
  await profile.waitFor();
  await profile.scrollIntoViewIfNeeded();
  assert.ok(await profile.getByRole('link', { name: 'Download denoise trace' }).isVisible());
  await page.screenshot({ path: 'test-results/image-profile-compact.png' });
  await page.setViewportSize({ width: 1460, height: 1080 });
  await profile.scrollIntoViewIfNeeded();
  await page.screenshot({ path: 'test-results/image-profile-results.png' });
  for (const trace of result.profile.traces) {
    const response = await fetch(base + trace.url);
    assert.ok(response.ok);
    assert.match(response.headers.get('content-disposition'), /attachment/);
    const data = await response.json();
    assert.ok(data.some((event) => event.cat === 'Api' && event.args?.cache_key));
  }
  const summary = await (await fetch(base + result.profile.summary_url)).json();
  assert.equal(summary.gpu_events, result.profile.gpu_events);
  const events = (await (await fetch(base + '/api/jobs/' + job.id + '/events')).text())
    .split('\n')
    .filter((line) => line.startsWith('data: '))
    .map((line) => JSON.parse(line.slice(6)));
  assert.deepEqual(
    [...new Set(events.filter((event) => event.stage).map((event) => event.stage))],
    ['prepare', 'prefix', 'denoise', 'decode'],
  );
  assert.deepEqual(
    events.filter((event) => event.type === 'image_progress').map((event) => event.completed),
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
  );
  assert.ok(events.some((event) => /Loading the text encoder/.test(event.message)));
  assert.ok(events.some((event) => /Saving your image/.test(event.message)));
  const baseline = JSON.parse(
    await fs.readFile(
      process.env.KITE_IMAGE_BASELINE || 'test-results/d3d12-app-cold.json',
      'utf8',
    ),
  );
  const hash = (data) => createHash('sha256').update(data).digest('hex');
  assert.equal(hash(await fs.readFile(result.path)), hash(await fs.readFile(baseline.result.path)));
  await page.reload();
  await page.getByRole('button', { name: /^Your gallery/ }).click();
  await page.locator('.art-gallery-item').first().click();
  await profile.waitFor();
  report.checks.push(
    'Real GPU traces, shader summary, saved gallery profile, and unchanged image output',
  );
  report.result = result;
  report.events = events.filter(
    (event) => event.type === 'status' || event.type === 'image_progress',
  );
  assert.deepEqual(errors, []);
  report.passed = true;
} finally {
  await browser.close();
  await fs.writeFile(
    'test-results/image-progress-profile-ui-report.json',
    JSON.stringify(report, null, 2),
  );
}
console.log(
  JSON.stringify({
    passed: report.passed,
    job: report.job,
    checks: report.checks,
    gpuEvents: report.result?.profile.gpu_events,
  }),
);
