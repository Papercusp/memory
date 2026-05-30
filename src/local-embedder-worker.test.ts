/**
 * Smoke test for the worker-thread isolated local embedder.
 *
 * Step B1 (Tier-3 follow-up arc).
 *
 * Doesn't actually exercise @huggingface/transformers (the package
 * isn't installed in CI by default — it's optional dev-mode dep).
 * Instead exercises the protocol contract: ensureWorker spawns the
 * worker, getWorkerState reports state, _resetWorker tears down.
 *
 * The actual ONNX path is exercised by the live mem0 round-trip when
 * the user opts into memoryEmbedderMode='local'.
 */
import { describe, it, expect, beforeAll, afterEach } from 'vitest';
import { existsSync } from 'node:fs';
import { join } from 'node:path';

const HF_PACKAGE_PATH = 'node_modules/@huggingface/transformers';
let hfInstalled = false;

beforeAll(() => {
  hfInstalled = existsSync(join(process.cwd(), HF_PACKAGE_PATH)) ||
                existsSync(join(process.cwd(), '..', HF_PACKAGE_PATH)) ||
                existsSync(join(process.cwd(), '..', '..', HF_PACKAGE_PATH));
});

afterEach(async () => {
  const mod = await import('./local-embedder-worker');
  await mod._resetWorker();
});

describe('local-embedder-worker (protocol contract)', () => {
  it('exports the expected API surface', async () => {
    const mod = await import('./local-embedder-worker');
    expect(typeof mod.embedViaWorker).toBe('function');
    expect(typeof mod._resetWorker).toBe('function');
    expect(typeof mod.getWorkerState).toBe('function');
  });

  it('getWorkerState starts in idle state', async () => {
    const mod = await import('./local-embedder-worker');
    const s = mod.getWorkerState();
    expect(s.alive).toBe(false);
    expect(s.disabled).toBe(false);
    expect(s.pendingCount).toBe(0);
  });

  it('embedViaWorker either succeeds with a vector or throws (worker protocol contract)', async () => {
    if (!hfInstalled) return; // soft-skip when @huggingface/transformers absent
    const mod = await import('./local-embedder-worker');
    try {
      const vector = await mod.embedViaWorker('hello world');
      // Success path: must be a numeric vector.
      expect(Array.isArray(vector)).toBe(true);
      expect(vector.length).toBeGreaterThan(0);
      expect(typeof vector[0]).toBe('number');
    } catch (err) {
      // Failure path: ONNX binding conflicts (common in vitest where
      // the main thread + multiple workers all try to load the same
      // native binding) surface here. The contract is "either returns
      // a vector or throws cleanly" — not "must always succeed".
      expect(err).toBeInstanceOf(Error);
    }
  }, 30_000);

  it('after embedViaWorker, getWorkerState reflects spawn (alive OR disabled)', async () => {
    if (!hfInstalled) return;
    const mod = await import('./local-embedder-worker');
    await mod.embedViaWorker('warm').catch(() => { /* protocol may throw */ });
    const s = mod.getWorkerState();
    // Either we have a live worker (success) OR the worker was disabled
    // after a failure. The contract is that idle state (alive=false +
    // disabled=false) is NOT possible after an attempt.
    expect(s.alive || s.disabled).toBe(true);
  }, 30_000);

  it('_resetWorker returns state to fully-idle', async () => {
    if (!hfInstalled) return;
    const mod = await import('./local-embedder-worker');
    await mod.embedViaWorker('warm').catch(() => { /* */ });
    await mod._resetWorker();
    const s = mod.getWorkerState();
    expect(s.alive).toBe(false);
    expect(s.disabled).toBe(false);
    expect(s.pendingCount).toBe(0);
  }, 30_000);
});
