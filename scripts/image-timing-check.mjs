import assert from 'node:assert/strict';
import { imageTimingBreakdown } from '../src/image-performance.ts';

const art = {
  request_ms: 1050,
  phases: {
    prepare: {
      wall_ms: 600,
      initialization_ms: 50,
      model_load_ms: 200,
      inference_ms: 250,
      cleanup_ms: 50,
    },
    prefix: { wall_ms: 150, model_load_ms: 100, inference_ms: 25 },
    denoise: {
      wall_ms: 200,
      verification_ms: 0,
      verification_cached: true,
      model_load_ms: 50,
      inference_ms: 140,
    },
    decode: { wall_ms: 50, model_load_ms: 10, inference_ms: 20, output_save_ms: 15 },
  },
};
const timing = imageTimingBreakdown(art);
assert.equal(timing.largest.key, 'prepare');
assert.equal(timing.overhead, 50);
assert.equal(
  timing.groups.reduce((sum, group) => sum + group.ms, timing.overhead),
  timing.total,
);
for (const group of timing.groups) {
  assert.equal(
    group.details.reduce((sum, detail) => sum + detail.ms, 0),
    group.ms,
  );
  assert.ok(
    Math.abs(group.details.reduce((sum, detail) => sum + detail.percent, 0) - group.percent) < 1e-8,
  );
}
const old = imageTimingBreakdown({ request_ms: 600, phases: { prepare: { wall_ms: 600 } } });
assert.equal(old.groups[0].details.length, 1);
assert.match(old.groups[0].details[0].label, /not recorded/);
assert.equal(old.groups[0].details[0].ms, 600);
assert.equal(imageTimingBreakdown({ request_ms: 0, phases: {} }).overheadPercent, 0);
console.log(
  'Timing checks passed: exclusive totals, percentages, cached zero time, and older records.',
);
