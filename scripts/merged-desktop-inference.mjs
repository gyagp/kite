import { _electron as electron } from 'playwright';
import path from 'node:path';
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
const directory = path.resolve('test-results/packaged-image-inference-' + Date.now());
const app = await electron.launch({
  executablePath: path.resolve('release/win-unpacked/Kite.exe'),
  env: { ...process.env, KITE_DATA_DIR: directory, KITE_TEST_HEADLESS: '1', KITE_IMAGE_DIR: path.resolve('gitignore/images') },
  timeout: 60000,
});
const report = { passed: false, checks: [] };
try {
  const page = await app.firstWindow();
  await app.evaluate(({ BrowserWindow }) =>
    BrowserWindow.getAllWindows()[0].webContents.setBackgroundThrottling(false),
  );
  await page
    .getByRole('heading', { name: 'Meaning, beyond keywords.' })
    .waitFor({ timeout: 60000 });
  const base = new URL(page.url()).origin;
  const api = async (p, body) => {
    const r = await fetch(
      base + p,
      body
        ? {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
          }
        : {},
    );
    const d = await r.json();
    assert.ok(r.ok, JSON.stringify(d));
    return d;
  };
  const terminal = async (id) => {
    const text = await (
      await fetch(base + '/api/jobs/' + id + '/events', { signal: AbortSignal.timeout(240000) })
    ).text();
    assert.match(text, /"type": "done"/);
    return api('/api/jobs/' + id);
  };
  await page.getByRole('button', { name: 'Image Studio', exact: true }).click();
  await page.getByLabel('Image canvas').selectOption('512x512');
  await page
    .getByLabel('Image prompt')
    .fill(
      'A cobalt blue glass kite on a bright white marble table, beautiful morning sunlight, minimalist product photograph.',
    );
  await page.getByRole('button', { name: 'Generate image', exact: true }).click();
  await page.locator('.generation-overlay').waitFor();
  await page.getByRole('button', { name: 'Stop', exact: true }).click();
  await page.locator('.generation-overlay').waitFor({ state: 'hidden', timeout: 30000 });
  const cancelled = (await api('/api/runs'))[0];
  assert.equal(cancelled.status, 'cancelled', JSON.stringify(cancelled));
  assert.equal((await api('/api/status')).worker_running, false);
  report.checks.push('Image cancellation releases native process tree');
  await page.getByRole('button', { name: 'Generate image', exact: true }).click();
  await page.getByRole('region', { name: 'Image generation performance' }).waitFor({ timeout: 240000 });
  const images = await api('/api/images');
  assert.equal(images.length, 1);
  const image = images[0];
  assert.equal(image.steps, 8);
  assert.equal(image.phases.denoise.runtime.provider_options.enableMatmulFp32Accumulation, undefined);
  assert.ok(image.phases.denoise.inference_ms > 0);
  assert.ok(image.request_ms > image.phases.denoise.inference_ms);
  const png = await fetch(base + image.image_url);
  assert.equal(png.headers.get('content-type'), 'image/png');
  assert.ok((await png.arrayBuffer()).byteLength > 10000);
  assert.equal(
    await fs
      .stat(path.join(path.dirname(image.path), 'prefix.npz'))
      .then(() => true)
      .catch(() => false),
    false,
  );
  const snapshot = await app.evaluate(async ({ BrowserWindow }) =>
    Array.from(
      (await BrowserWindow.getAllWindows()[0].capturePage(undefined, { stayHidden: true })).toPNG(),
    ),
  );
  await fs.writeFile('test-results/packaged-image-inference.png', Buffer.from(snapshot));
  report.image = { request_ms: image.request_ms, steps: image.steps, phases: image.phases };
  report.checks.push('Packaged image generation, progress, performance and PNG output');
  report.checks.push('Successful image generation removes only its scratch files');
  assert.ok(image.path.startsWith(path.resolve('gitignore/images')));
  const firstBytes = Buffer.from(await (await fetch(base + image.image_url)).arrayBuffer());
  const warmJob = await api('/api/jobs', {kind:'image', prompt:image.prompt, width:image.width, height:image.height, seed:image.seed});
  const warm = await terminal(warmJob.id);
  assert.equal(warm.status, 'completed', warm.error);
  assert.equal(warm.result.phases.prepare.components_reused, true);
  assert.equal(warm.result.phases.decode.components_reused, true);
  assert.equal(warm.result.phases.denoise.verification_cached, true);
  const warmBytes = Buffer.from(await (await fetch(base + warm.result.image_url)).arrayBuffer());
  assert.deepEqual(warmBytes, firstBytes);
  report.warm_ms = warm.result.request_ms;
  report.checks.push('Exact repeated image, CPU component and artifact-verification reuse, automatic ignored-folder saving');
  const boot = await api('/api/bootstrap');
  const text = boot.models.find((m) => m.kind === 'embedding' && !m.name.includes('Multimodal'));
  const id = (
    await api('/api/jobs', {
      kind: 'embed',
      model: text.id,
      texts: ['A kite flying above the clouds.'],
      task: 'Document',
      dimensions: 128,
    })
  ).id;
  const embed = await terminal(id);
  assert.equal(embed.status, 'completed', embed.error);
  assert.equal(embed.result.vectors[0].length, 128);
  await api('/api/stop', {});
  report.checks.push('Image-to-embedding runtime switch and existing DLL isolation');
  await page.reload();
  await page.getByRole('button', { name: 'Image Studio', exact: true }).click();
  await page.getByRole('button', { name: /Your gallery/ }).click();
  await page.locator('.art-gallery-item').first().waitFor();
  report.checks.push('Generated image gallery persists after reload');
  report.passed = true;
} finally {
  await fs.writeFile(
    'test-results/packaged-image-inference-report.json',
    JSON.stringify(report, null, 2),
  );
  await app.close();
}
console.log({ passed: report.passed, checks: report.checks });
