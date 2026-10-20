import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

const base = process.env.KITE_TEST_URL;
assert.ok(base);
const browser = await chromium.launch({ channel: 'msedge', headless: true });
const page = await browser.newPage({
  viewport: { width: 1460, height: 1080 },
  reducedMotion: 'reduce',
});
const report = { passed: false, runs: [] };
const errors = [];
page.on('pageerror', (error) => errors.push(error.message));
try {
  await page.goto(base);
  // Bound this test's response without changing the user's remembered output limit.
  await page.route('**/api/jobs', async (route) => {
    const data = route.request().postDataJSON();
    if (data.kind === 'chat') {
      assert.equal(data.profile, true);
      return route.continue({
        postData: JSON.stringify({ ...data, max_tokens: 24, temperature: 0, save: false }),
      });
    }
    return route.continue();
  });
  const run = async (kind, trigger) => {
    const response = page.waitForResponse(
      (r) => r.url().endsWith('/api/jobs') && r.request().method() === 'POST',
    );
    await trigger();
    const submitted = await (await response).json();
    assert.ok(submitted.id, JSON.stringify(submitted));
    const eventsText = await (
      await fetch(base + '/api/jobs/' + submitted.id + '/events', {
        signal: AbortSignal.timeout(240000),
      })
    ).text();
    const job = await (await fetch(base + '/api/jobs/' + submitted.id)).json();
    report.runs.push(job);
    assert.equal(job.status, 'completed', JSON.stringify(job));
    assert.ok(job.result.profile.traces.length > 0, JSON.stringify(job.result.profile));
    assert.ok(job.result.profile.gpu_events > 0 || job.result.profile.top_cpu_ops.length > 0);
    const panel = page.getByRole('region', { name: 'GPU profile' });
    await panel.waitFor();
    await panel.scrollIntoViewIfNeeded();
    await page.screenshot({ path: `test-results/${kind}-profile-ui.png` });
    for (const trace of job.result.profile.traces) {
      const downloaded = await fetch(base + trace.url);
      assert.ok(downloaded.ok);
      assert.match(downloaded.headers.get('content-disposition'), /attachment/);
      assert.ok(Array.isArray(await downloaded.json()));
    }
    assert.match(eventsText, /"type": "done"/);
    assert.equal((await (await fetch(base + '/api/status')).json()).worker_running, false);
  };
  await page.getByRole('button', { name: 'Chat', exact: true }).click();
  await page.getByRole('checkbox', { name: /Profile response/ }).check();
  await page
    .getByLabel('Message', { exact: true })
    .fill('Reply with one short sentence: explain why the sky is blue.');
  await run('chat', () => page.getByRole('button', { name: 'Send message', exact: true }).click());
  await page.getByRole('button', { name: 'Embeddings', exact: true }).click();
  await page.getByRole('checkbox', { name: /Profile embeddings/ }).check();
  await run('embeddings', () =>
    page.getByRole('button', { name: 'Find matches', exact: true }).click(),
  );
  assert.ok(report.runs[1].result.ranking.length > 0);
  const forbidden = await fetch(base + '/api/profiles/' + report.runs[0].id + '/request.json');
  assert.equal(forbidden.status, 400);
  assert.deepEqual(errors, []);
  report.passed = true;
} finally {
  await browser.close();
  await fs.writeFile('test-results/all-profiling-ui-report.json', JSON.stringify(report, null, 2));
}
console.log(
  JSON.stringify({
    passed: report.passed,
    runs: report.runs.map((job) => ({
      id: job.id,
      kind: job.kind,
      gpuEvents: job.result?.profile?.gpu_events,
      warnings: job.result?.profile?.warnings,
    })),
  }),
);
