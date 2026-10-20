// Real inference and UI verification in a disposable packaged-app workspace.
import { _electron as electron } from 'playwright';
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';

const report = { passed: false, checks: [], measurements: {} };
const app = await electron.launch({
  executablePath: path.resolve('release/win-unpacked/Kite.exe'),
  env: {
    ...process.env,
    KITE_DATA_DIR: path.resolve('test-results', `embedding-perf-${Date.now()}`),
    KITE_TEST_HEADLESS: '1',
  },
  timeout: 60000,
});
try {
  const page = await app.firstWindow();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await app.evaluate(({ BrowserWindow }) =>
    BrowserWindow.getAllWindows()[0].webContents.setBackgroundThrottling(false),
  );
  await page
    .getByRole('heading', { name: 'Meaning, beyond keywords.' })
    .waitFor({ timeout: 60000 });
  const base = new URL(page.url()).origin;
  const api = async (route, body) => {
    const response = await fetch(
      base + route,
      body
        ? {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
          }
        : {},
    );
    const data = await response.json();
    assert.ok(response.ok, JSON.stringify(data));
    return data;
  };
  const run = async (body) => {
    const { id } = await api('/api/jobs', body);
    const events = await (
      await fetch(`${base}/api/jobs/${id}/events`, { signal: AbortSignal.timeout(180000) })
    ).text();
    assert.ok(events.includes('"type": "done"'));
    const job = await api(`/api/jobs/${id}`);
    assert.equal(job.status, 'completed', job.error);
    return job;
  };
  const closeTo = (a, b) =>
    assert.ok(Math.abs(a - b) <= Math.max(1e-6, Math.abs(b) * 1e-6), `${a} != ${b}`);
  const check = (name, value) => {
    assert.ok(value.inference_ms > 0 && value.preparation_ms > 0 && value.postprocessing_ms > 0);
    assert.ok(value.request_ms >= value.total_ms + value.setup_ms);
    assert.ok(
      value.total_ms >= value.inference_ms + value.preparation_ms + value.postprocessing_ms,
    );
    assert.ok(value.padded_tokens >= value.tokens);
    const inputs = [...(value.query_inputs || []), ...value.inputs];
    assert.equal(inputs.length, value.input_count);
    assert.equal(
      inputs.reduce((n, i) => n + i.tokens, 0),
      value.tokens,
    );
    closeTo(value.inference_tokens_per_second, (value.tokens * 1000) / value.inference_ms);
    closeTo(value.inference_inputs_per_second, (value.input_count * 1000) / value.inference_ms);
    if (value.search_phases) {
      for (const key of [
        'tokens',
        'input_count',
        'batch_count',
        'inference_ms',
        'preparation_ms',
        'postprocessing_ms',
        'total_ms',
      ]) {
        closeTo(value[key], value.search_phases.query[key] + value.search_phases.documents[key]);
      }
    }
    for (const vector of value.vectors || []) {
      assert.ok(vector.every(Number.isFinite));
      closeTo(
        vector.reduce((sum, x) => sum + x * x, 0),
        1,
      );
    }
    report.measurements[name] = Object.fromEntries(
      Object.entries(value).filter(
        ([key]) => !['vectors', 'ranking', 'query', 'query_attachments'].includes(key),
      ),
    );
  };
  const screenshot = async (name) => {
    const png = await app.evaluate(async ({ BrowserWindow }) =>
      (await BrowserWindow.getAllWindows()[0].capturePage(undefined, { stayHidden: true }))
        .toPNG()
        .toString('base64'),
    );
    await fs.writeFile(`test-results/${name}.png`, Buffer.from(png, 'base64'));
  };
  const boot = await api('/api/bootstrap');
  const text = boot.models.find((m) => m.kind === 'embedding' && m.modalities.length === 1);
  const multimodal = boot.models.find((m) => m.embedding_task === 'multimodal-embedding');
  await page.getByRole('combobox', { name: 'Select model' }).selectOption(text.id);
  await page.getByRole('button', { name: 'Find matches', exact: true }).click();
  const card = page.getByRole('region', { name: 'Embedding performance', exact: true });
  await card.waitFor({ timeout: 180000 });
  let latest = (await api('/api/runs'))[0];
  check('cold_text_search', latest.result);
  assert.equal(latest.result.input_count, 4);
  assert.equal(latest.result.batch_count, 2);
  assert.ok(
    latest.result.model_load_ms > 0 && latest.result.setup_ms >= latest.result.model_load_ms,
  );
  await card.getByText('Timing details', { exact: true }).click();
  await page.getByRole('table', { name: 'Search performance breakdown' }).waitFor();
  await card.scrollIntoViewIfNeeded();
  await screenshot('embedding-performance-text');
  await page.getByRole('button', { name: 'Toggle JSON view' }).click();
  const json = JSON.parse(await page.locator('.json-result').innerText());
  closeTo(json.request_ms, latest.result.request_ms);
  assert.equal(await card.isVisible(), true);
  report.checks.push(
    'Text search card and JSON show measured query/document totals, throughput, and cold model setup',
  );

  const batched = await run({
    kind: 'embed',
    model: text.id,
    task: 'SentenceSimilarity',
    dimensions: 128,
    texts: [
      'Hi.',
      'A much longer sentence about the northern lights, the sky, and winter travel.',
      '蓝色的地球。',
      'Ocean.',
      'An astronaut above Earth.',
    ],
  });
  check('warm_text_batch', batched.result);
  assert.equal(batched.result.input_count, 5);
  assert.equal(batched.result.batch_count, 2);
  assert.equal(batched.result.setup_ms, 0);
  assert.equal(batched.result.model_load_ms, 0);
  assert.ok(batched.result.padded_tokens > batched.result.tokens);
  report.checks.push(
    'Five variable-length texts cross the four-input batch boundary; padding excluded from rates; warm setup is zero',
  );

  await page.getByRole('button', { name: 'Multimodal examples', exact: true }).click();
  await page.getByRole('button', { name: 'Run Hear it. Find it.', exact: true }).click();
  await page.getByRole('dialog').waitFor({ state: 'hidden' });
  await card.waitFor({ timeout: 180000 });
  latest = (await api('/api/runs'))[0];
  check('audio_image_search', latest.result);
  assert.equal(latest.result.input_count, 5);
  assert.equal(latest.result.batch_count, 5);
  assert.deepEqual(latest.result.query_inputs[0].modalities, ['audio']);
  assert.equal(latest.result.inputs.length, 4);
  report.checks.push(
    'Azure audio query and four NASA images have separately measured search phases',
  );

  await page.getByRole('button', { name: 'Multimodal examples', exact: true }).click();
  await page
    .getByRole('button', { name: 'Run One mission. Every dimension.', exact: true })
    .click();
  await page.getByRole('dialog').waitFor({ state: 'hidden' });
  await card.waitFor({ timeout: 180000 });
  latest = (await api('/api/runs'))[0];
  check('mixed_media', latest.result);
  assert.equal(latest.result.input_count, 5);
  assert.equal(latest.result.setup_ms, 0);
  assert.ok(
    latest.result.inputs.reduce((sum, i) => sum + i.video_frames.reduce((a, b) => a + b, 0), 0) > 0,
  );
  assert.match(await card.innerText(), /video frames/);
  await card.scrollIntoViewIfNeeded();
  await screenshot('embedding-performance-multimodal');
  report.checks.push(
    'Mixed text/image/audio/video reports real inference timings, expanded tokens, and sampled video frames',
  );

  // Exercise a video query as well: query metrics must not disappear from the result.
  const example = await api('/api/examples/mixed-input', {});
  const video = example.samples.flatMap((s) => s.attachments).find((a) => a.kind === 'video');
  const videoQuery = await run({
    kind: 'search',
    model: multimodal.id,
    query: '',
    query_attachments: [video],
    documents: ['A rocket launch followed by views of Earth and a spacewalker.'],
  });
  check('video_query', videoQuery.result);
  assert.ok(videoQuery.result.query_inputs[0].video_frames[0] > 0);
  report.checks.push('Media-only video query retains its input tokens and frame counts');

  await page.reload();
  await page.getByRole('navigation').getByRole('button', { name: 'Activity', exact: true }).click();
  await page.getByRole('button', { name: `View run ${latest.id}`, exact: true }).click();
  await card.waitFor();
  assert.match(await card.innerText(), /Already loaded|Total request/);
  const saved = await api('/api/runs');
  closeTo(saved.find((r) => r.id === latest.id).result.inference_ms, latest.result.inference_ms);
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(1000, 760));
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  await screenshot('embedding-performance-activity');
  assert.deepEqual(errors, []);
  report.checks.push(
    'Activity restores measured results after reload; compact desktop layout has no page overflow or renderer errors',
  );
  report.passed = true;
} finally {
  await app.close();
  await fs.writeFile(
    'test-results/embedding-performance-report.json',
    JSON.stringify(report, null, 2),
  );
}
console.log(JSON.stringify({ passed: report.passed, checks: report.checks }, null, 2));
