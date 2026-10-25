import { _electron as electron } from 'playwright';
import assert from 'node:assert/strict';
import path from 'node:path';
import fs from 'node:fs/promises';

const app = await electron.launch({
  executablePath: path.resolve(process.env.KITE_TEST_EXECUTABLE || 'release/win-unpacked/Kite.exe'),
  args: ['--user-data-dir=' + path.resolve('test-results/sam3-electron-profile')],
  env: {
    ...process.env,
    KITE_TEST_HEADLESS: '1',
    KITE_DATA_DIR: path.resolve('test-results/sam3-packaged-state'),
  },
  timeout: 60000,
});
try {
  const page = await app.firstWindow();
  await page.getByRole('button', { name: 'Video Segmentation', exact: true }).click();
  await page.getByRole('button', { name: 'Try “Jumping together”' }).click();
  await page.locator('.seg-canvas video').evaluate(
    (v) =>
      new Promise((resolve) => {
        if (v.readyState >= 2) resolve();
        else v.onloadeddata = resolve;
      }),
  );
  await page.waitForFunction(() => {
    const v = document.querySelector('.seg-canvas video');
    return !v.paused && v.currentTime > 0.1 && v.getVideoPlaybackQuality().totalVideoFrames >= 3;
  });
  assert.notEqual(await page.getByLabel('Segmentation model').inputValue(), '');
  await page.getByLabel('Frames to process').fill('3');
  await page.getByLabel('Frame interval').selectOption('2');
  const response = page.waitForResponse(
    (r) => r.url().endsWith('/api/jobs') && r.request().method() === 'POST',
  );
  await page.getByRole('button', { name: 'Track objects', exact: true }).click();
  const { id } = await (await response).json();
  await page.locator('.seg-result video').waitFor({ timeout: 120000 });
  await page.waitForFunction(() => {
    const v = document.querySelector('.seg-result video');
    return (
      !v.paused &&
      v.loop &&
      v.currentTime > 0.1 &&
      v.getVideoPlaybackQuality().totalVideoFrames >= 3
    );
  });
  const job = await page.evaluate(async (id) => (await fetch('/api/jobs/' + id)).json(), id);
  assert.equal(job.status, 'completed', job.error);
  assert.equal(job.result.frame_count, 3);
  assert.equal(job.result.output_fps, 6);
  assert.deepEqual(
    job.result.frames.map((f) => f.source_frame),
    [0, 2, 4],
  );
  const png = await app.evaluate(async ({ BrowserWindow }) =>
    (await BrowserWindow.getAllWindows()[0].capturePage(undefined, { stayHidden: true }))
      .toPNG()
      .toString('base64'),
  );
  await fs.writeFile('test-results/sam3-packaged.png', Buffer.from(png, 'base64'));
  await fs.writeFile('test-results/sam3-packaged.json', JSON.stringify(job, null, 2));
  console.log(
    'Packaged SAM 3: sample playback, native tracking, stride, and looping result playback passed.',
  );
} finally {
  await app.close();
}
