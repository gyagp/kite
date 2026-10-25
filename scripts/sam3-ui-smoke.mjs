import { chromium } from 'playwright';
import { mkdir, writeFile } from 'node:fs/promises';
import assert from 'node:assert/strict';

const base = process.argv[2] || 'http://127.0.0.1:4318';
const browser = await chromium.launch({ channel: 'msedge', headless: true });
const page = await browser.newPage({ viewport: { width: 1500, height: 1100 } });
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
const reports = [];
await mkdir('test-results', { recursive: true });
try {
  await page.goto(base + '/#segments');
  await page.getByRole('heading', { name: 'Pick it. Follow it.' }).waitFor();
  await page.getByRole('button', { name: 'Try “Jumping together”' }).click();
  await page.locator('.seg-canvas video').evaluate(
    (v) =>
      new Promise((resolve) => {
        if (v.readyState >= 2) resolve();
        else v.onloadeddata = resolve;
      }),
  );
  await page.getByLabel('Frames to process').fill('12');
  await page.getByLabel('Add object').click();
  await page.getByRole('button', { name: '□ Box', exact: true }).click();
  const rect = await page.locator('.seg-canvas svg').boundingBox();
  await page.mouse.move(rect.x + rect.width * 0.13, rect.y + rect.height * 0.23);
  await page.mouse.down();
  await page.mouse.move(rect.x + rect.width * 0.32, rect.y + rect.height * 0.76, { steps: 12 });
  await page.mouse.up();
  await page.getByLabel('Profile tracking', { exact: true }).check();
  await page.screenshot({ path: 'test-results/sam3-ui-selection.png', fullPage: true });
  const jobResponse = page.waitForResponse(
    (r) => r.url().endsWith('/api/jobs') && r.request().method() === 'POST',
  );
  await page.getByRole('button', { name: 'Track objects', exact: true }).click();
  const { id } = await (await jobResponse).json();
  await page.getByAltText('Latest tracked frame').waitFor({ timeout: 180000 });
  await page.getByRole('button', { name: 'Chat', exact: true }).click();
  await page.getByRole('button', { name: 'Video Segmentation', exact: true }).click();
  await page.getByRole('button', { name: 'Stop tracking', exact: true }).waitFor();
  await page.locator('.seg-result video').waitFor({ timeout: 300000 });
  const job = await (await page.request.get(base + '/api/jobs/' + id)).json();
  assert.equal(job.status, 'completed', job.error);
  assert.equal(job.result.frame_count, 12);
  assert.equal(job.result.objects.length, 2);
  assert.equal(job.result.runtime.provider_options.dawnBackendType, 'D3D12');
  assert.equal(job.result.profile.traces.length, 4);
  for (const name of ['overlay.mp4', 'masks.mp4', 'masks.zip', 'mask-00011.png', 'result.json']) {
    const response = await page.request.get(`${base}/api/segments/${id}/${name}`);
    assert.equal(response.status(), 200, name);
    assert.ok((await response.body()).length > 100);
  }
  for (const trace of job.result.profile.traces)
    assert.equal((await page.request.get(base + trace.url)).status(), 200);
  await page.screenshot({ path: 'test-results/sam3-ui-result.png', fullPage: true });
  reports.push(job);
  // The next run must be cancellable during load or tracking and release the worker.
  await page.getByLabel('Profile tracking', { exact: true }).uncheck();
  const cancelResponse = page.waitForResponse(
    (r) => r.url().endsWith('/api/jobs') && r.request().method() === 'POST',
  );
  await page.getByRole('button', { name: 'Track objects', exact: true }).click();
  const cancelId = (await (await cancelResponse).json()).id;
  await page.getByRole('button', { name: 'Stop tracking', exact: true }).click();
  await page.waitForFunction(
    async (id) => (await (await fetch('/api/jobs/' + id)).json()).status === 'cancelled',
    cancelId,
  );
  const status = await (await page.request.get(base + '/api/status')).json();
  assert.equal(status.worker_running, false);
  await page.reload();
  await page.getByRole('heading', { name: 'Recent tracks', exact: true }).waitFor();
  await page.setViewportSize({ width: 900, height: 1000 });
  await page.screenshot({ path: 'test-results/sam3-ui-compact.png', fullPage: true });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  assert.deepEqual(errors, []);
  await writeFile(
    'test-results/sam3-ui-report.json',
    JSON.stringify({ reports, cancelled: cancelId, errors }, null, 2),
  );
  console.log(
    JSON.stringify({
      completed: id,
      cancelled: cancelId,
      frames: job.result.frame_count,
      total_ms: job.result.request_ms,
      phases: job.result.phases,
      errors,
    }),
  );
} finally {
  await browser.close();
}
