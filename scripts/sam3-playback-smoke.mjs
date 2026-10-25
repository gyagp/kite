// Exercise actual decoded playback, using already-generated local SAM 3 results.
import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import { writeFile } from 'node:fs/promises';

const base = process.argv[2] || 'http://127.0.0.1:4318';
const browser = await chromium.launch({ channel: 'msedge', headless: true });
const page = await browser.newPage({ viewport: { width: 1460, height: 980 } });
const errors = [];
page.on('pageerror', (error) => errors.push(error.message));
const motion = (selector) =>
  page.locator(selector).evaluate(
    (video) =>
      new Promise((resolve, reject) => {
        const frames = [];
        const timer = setTimeout(
          () => reject(new Error('Video did not render moving frames')),
          15000,
        );
        const sample = (_, metadata) => {
          frames.push(metadata.mediaTime);
          if (new Set(frames).size >= 4) {
            clearTimeout(timer);
            resolve({ frames, paused: video.paused, loop: video.loop, muted: video.muted });
          } else video.requestVideoFrameCallback(sample);
        };
        video.requestVideoFrameCallback(sample);
      }),
  );
try {
  await page.goto(base + '/#segments');
  await page.getByRole('button', { name: 'Try “Jumping together”' }).click();
  const sourceMotion = await motion('.seg-canvas video');
  assert.equal(sourceMotion.paused, false);
  await page.getByRole('button', { name: 'Back to selection', exact: true }).click();
  await page.waitForFunction(() => {
    const v = document.querySelector('.seg-canvas video');
    return v.paused && Math.abs(v.currentTime) < 0.01;
  });
  assert.equal(await page.locator('.seg-canvas svg circle').count(), 1);
  await page.getByRole('button', { name: 'Play clip', exact: true }).click();
  await motion('.seg-canvas video');
  await page.getByRole('button', { name: '□ Box', exact: true }).click();
  assert.equal(await page.locator('.seg-canvas video').evaluate((v) => v.paused), true);
  assert.equal(await page.locator('.seg-canvas svg circle').count(), 1);

  await page.locator('.seg-history button').first().click();
  const resultMotion = await motion('.seg-result video');
  assert.equal(resultMotion.loop, true);
  assert.equal(resultMotion.muted, true);
  await page.waitForFunction(() => {
    const rect = document.querySelector('.seg-result video').getBoundingClientRect();
    return rect.top >= 0 && rect.bottom <= innerHeight;
  });
  await page.screenshot({ path: 'test-results/sam3-playback.png' });
  await page.getByRole('button', { name: 'Pause tracked video', exact: true }).click();
  assert.equal(await page.locator('.seg-result video').evaluate((v) => v.paused), true);
  await page.getByRole('button', { name: 'Play tracked video', exact: true }).click();
  await motion('.seg-result video');
  // Reopening the same history entry restarts playback as well.
  await page.getByRole('button', { name: 'Pause tracked video', exact: true }).click();
  await page.locator('.seg-history button').first().click();
  await motion('.seg-result video');
  assert.deepEqual(errors, []);
  const report = { sourceMotion, resultMotion, promptRetained: true, resultVisible: true, errors };
  await writeFile('test-results/sam3-playback.json', JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
} finally {
  await browser.close();
}
