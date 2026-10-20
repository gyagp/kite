import { chromium, _electron as electron } from 'playwright';
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
const desktop = process.argv.includes('--desktop');
const app = desktop
  ? await electron.launch({
      executablePath: path.resolve('release/win-unpacked/Kite.exe'),
      env: {
        ...process.env,
        KITE_DATA_DIR: path.resolve('test-results/merged-workspace'),
        KITE_TEST_HEADLESS: '1',
      },
      timeout: 60000,
    })
  : null;
const browser = !desktop ? await chromium.launch({ channel: 'msedge', headless: true }) : null;
const page = app
  ? await app.firstWindow()
  : await browser.newPage({ viewport: { width: 1460, height: 1040 } });
if (app)
  await app.evaluate(({ BrowserWindow }) =>
    BrowserWindow.getAllWindows()[0].webContents.setBackgroundThrottling(false),
  );
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
const report = { passed: false, checks: [], desktop };
const screenshot = async (name) => {
  if (app) {
    const bytes = await app.evaluate(async ({ BrowserWindow }) =>
      Array.from(
        (
          await BrowserWindow.getAllWindows()[0].capturePage(undefined, { stayHidden: true })
        ).toPNG(),
      ),
    );
    await fs.writeFile('test-results/' + name, Buffer.from(bytes));
  } else await page.screenshot({ path: 'test-results/' + name, fullPage: true });
};
try {
  if (!desktop) await page.goto(process.env.KITE_TEST_URL || 'http://127.0.0.1:4325');
  await page.getByRole('button', { name: 'Image Studio', exact: true }).click();
  await page.getByRole('heading', { name: 'Make the impossible.' }).waitFor();
  await page.locator('.studio-preset').first().waitFor();
  assert.equal(await page.locator('.studio-preset').count(), 4);
  await page.locator('.artwork-main').evaluate((img) => img.decode());
  await screenshot('merged-image-studio' + (desktop ? '-desktop' : '') + '.png');
  report.checks.push('Image Studio previews and four real generated presets');
  await page.getByRole('button', { name: /From a line to a world/ }).click();
  await page.getByRole('button', { name: 'Transform image', exact: true }).waitFor();
  assert.equal(await page.locator('.reference-thumb').count(), 1);
  await page.getByRole('slider', { name: 'Before and after comparison' }).fill('30');
  await screenshot('merged-image-edit' + (desktop ? '-desktop' : '') + '.png');
  await page.getByRole('button', { name: 'Expand image', exact: true }).click();
  await page.getByRole('dialog').waitFor();
  await page.getByRole('button', { name: 'Close dialog' }).click();
  report.checks.push('Reference attachment, before/after slider and full image viewer');
  await page.getByRole('button', { name: /Your gallery/ }).click();
  await page.locator('.art-gallery-item').first().click();
  await page.getByRole('region', { name: 'Image generation performance' }).waitFor();
  assert.equal(await page.locator('.image-timing-legend>span').count(), 4);
  report.checks.push('Persistent gallery, reuse settings and four performance stages');
  assert.equal(await page.getByRole('button', { name: 'Model Lab', exact: true }).count(), 0);
  report.checks.push('Inference-focused navigation has no conversion or authoring UI');
  if (!app) await page.setViewportSize({ width: 1000, height: 760 });
  await page.getByRole('button', { name: 'Image Studio', exact: true }).click();
  await screenshot('merged-compact' + (desktop ? '-desktop' : '') + '.png');
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  assert.deepEqual(errors, []);
  report.checks.push('Compact layout and no renderer exceptions');
  report.passed = true;
} finally {
  await fs.writeFile(
    'test-results/merged-ui' + (desktop ? '-desktop' : '') + '-report.json',
    JSON.stringify(report, null, 2),
  );
  if (app) await app.close();
  if (browser) await browser.close();
}
console.log(report);
