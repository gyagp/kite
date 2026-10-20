// Packaged app, disposable state, real generation, and legacy run compatibility.
import { _electron as electron, chromium } from 'playwright';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import { imageTimingBreakdown } from '../src/image-performance.ts';

const directory = path.resolve('test-results/image-timing-' + Date.now());
execFileSync(
  path.resolve('local/.venv/Scripts/python.exe'),
  [
    '-c',
    'import json,sys; from backend.store import Store; s=Store(sys.argv[1]); s.put("images",json.load(open("test-results/qwen-copy-fix-app-warm.json"))["result"]); s.close()',
    directory,
  ],
  { windowsHide: true },
);
const app = await electron.launch({
  executablePath: path.resolve('release/win-unpacked/Kite.exe'),
  env: { ...process.env, KITE_DATA_DIR: directory, KITE_TEST_HEADLESS: '1' },
  timeout: 60000,
});
const report = { passed: false, checks: [] };
let browser;
try {
  const window = await app.firstWindow();
  await window.getByRole('navigation').waitFor({ timeout: 60000 });
  await app.evaluate(({ BrowserWindow }) =>
    BrowserWindow.getAllWindows()[0].webContents.setBackgroundThrottling(false),
  );
  // Hidden Electron windows do not reliably paint tall element screenshots.
  // Render this packaged app's page in a fresh, test-owned headless browser.
  browser = await chromium.launch({ channel: 'msedge', headless: true });
  const page = await browser.newPage({
    viewport: { width: 1460, height: 1040 },
    reducedMotion: 'reduce',
  });
  await page.goto(window.url());
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  const api = async (route) => (await fetch(new URL(route, page.url()))).json();
  await page.getByRole('button', { name: 'Image Studio', exact: true }).click();
  await page.getByRole('button', { name: /^Your gallery/ }).click();
  await page.locator('.art-gallery-item').first().click();
  const panel = page.getByRole('region', { name: 'Image generation performance' });
  await panel.getByText('Detailed timings not recorded', { exact: true }).first().waitFor();
  assert.match(await panel.innerText(), /17.89 s/);
  report.checks.push(
    'Older saved runs show their measured breakdown without inventing missing details',
  );
  await page.getByRole('button', { name: 'Generate image', exact: true }).click();
  await page.locator('.generation-overlay').waitFor();
  await page.locator('.generation-overlay').waitFor({ state: 'hidden', timeout: 240000 });
  await panel.getByText('Initialize libraries', { exact: true }).first().waitFor();
  const artwork = (await api('/api/images'))[0];
  report.artwork = artwork;
  assert.equal(artwork.timing_version, 2);
  for (const phase of ['prepare', 'decode']) {
    for (const field of [
      'initialization_ms',
      'model_load_ms',
      'input_preparation_ms',
      'inference_ms',
      'output_save_ms',
      'cleanup_ms',
    ]) {
      assert.ok(
        Number.isFinite(artwork.phases[phase][field]) && artwork.phases[phase][field] >= 0,
        phase + '/' + field,
      );
    }
  }
  const breakdown = imageTimingBreakdown(artwork);
  for (const group of breakdown.groups) {
    assert.ok(
      Math.abs(group.details.reduce((sum, d) => sum + d.ms, 0) - group.ms) < 0.01,
      group.key,
    );
  }
  assert.ok(
    Math.abs(
      breakdown.groups.reduce((sum, g) => sum + g.ms, breakdown.overhead) - artwork.request_ms,
    ) < 0.01,
  );
  await panel.evaluate((el) => el.scrollIntoView({ block: 'start', behavior: 'instant' }));
  await page.screenshot({ path: 'test-results/image-timing-panel.png' });
  await page.setViewportSize({ width: 1000, height: 900 });
  await panel.evaluate((el) => el.scrollIntoView({ block: 'start', behavior: 'instant' }));
  await page.screenshot({ path: 'test-results/image-timing-panel-compact.png' });
  assert.ok(await panel.evaluate((el) => el.scrollWidth <= el.clientWidth));
  await page.reload();
  await page.getByRole('button', { name: 'Image Studio', exact: true }).click();
  await page.getByRole('button', { name: /^Your gallery/ }).click();
  await page.locator('.art-gallery-item').first().click();
  await panel.getByText('Initialize libraries', { exact: true }).first().waitFor();
  assert.deepEqual(errors, []);
  report.checks.push(
    'Real generation records all new intervals; exclusive totals match request time',
  );
  report.checks.push(
    'Breakdown survives reload and fits desktop and compact windows without horizontal overflow',
  );
  report.artwork = artwork;
  report.passed = true;
} finally {
  await browser?.close();
  await app.close();
  await fs.writeFile('test-results/image-timing-ui-report.json', JSON.stringify(report, null, 2));
}
console.log(
  JSON.stringify(
    { passed: report.passed, checks: report.checks, phases: report.artwork?.phases },
    null,
    2,
  ),
);
