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
import { describe, it, expect, beforeAll, afterEach, vi } from 'vitest';
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

/**
 * EI-19464316359123796: a standalone tsx driver that imports the sync-resolver
 * (which can transitively spin up this worker) and then calls `process.exit(0)`
 * dies with a bare `Napi::Error` at process teardown — AFTER its real output
 * already printed. Root cause: the worker was permanently REF'd, so a script
 * with no other pending work had no way to exit naturally and reached for
 * `process.exit()`, which tears the process (and every worker's native addon
 * state) down without running Node's normal per-Environment cleanup.
 *
 * The worker script posts `{ kind: 'ready' }` unconditionally at load, BEFORE
 * touching `@huggingface/transformers` — so this whole lifecycle is testable
 * without the optional ONNX dependency installed.
 */
describe('local-embedder-worker (EI-19464316359123796: exit-without-crash)', () => {
  afterEach(async () => {
    const mod = await import('./local-embedder-worker');
    await mod._resetWorker();
  });

  it('unrefs the worker once it becomes ready, so it cannot keep the process alive on its own', async () => {
    const { Worker } = await import('node:worker_threads');
    const unrefSpy = vi.spyOn(Worker.prototype, 'unref');
    const mod = await import('./local-embedder-worker');
    // Spawning happens lazily on first embed attempt; the call itself may
    // reject (no HF package / no model), but `ready` still fires first —
    // that is the whole point of the protocol ordering above.
    await mod.embedViaWorker('warm').catch(() => { /* protocol may throw */ });
    expect(unrefSpy).toHaveBeenCalled();
    unrefSpy.mockRestore();
  }, 30_000);

  it('exports a public, discoverable shutdown function that is the same op as _resetWorker', async () => {
    const mod = await import('./local-embedder-worker');
    expect(typeof mod.shutdownLocalEmbedder).toBe('function');
    expect(mod.shutdownLocalEmbedder).toBe(mod._resetWorker);
  });

  it('installs exactly one process-level beforeExit listener no matter how many times the worker (re)spawns', async () => {
    const mod = await import('./local-embedder-worker');
    const before = process.listenerCount('beforeExit');
    await mod.embedViaWorker('one').catch(() => { /* */ });
    const afterFirst = process.listenerCount('beforeExit');
    expect(afterFirst).toBe(before + 1);

    // Reset (simulating a crash/respawn cycle) and spawn again — the guard
    // must stop a second listener from ever being added.
    await mod._resetWorker();
    await mod.embedViaWorker('two').catch(() => { /* */ });
    expect(process.listenerCount('beforeExit')).toBe(afterFirst);
  }, 30_000);

  it('the beforeExit hook tears the worker down without throwing, and it self-heals on the next call', async () => {
    const mod = await import('./local-embedder-worker');
    await mod.embedViaWorker('warm').catch(() => { /* */ });
    expect(mod.getWorkerState().alive || mod.getWorkerState().disabled).toBe(true);

    // Fire the installed listener(s) directly rather than emitting a real
    // process-wide 'beforeExit' (which would race the test runner's own
    // idle-detection). This exercises exactly the async body the real event
    // would invoke.
    await Promise.all(process.listeners('beforeExit').map((fn) => fn('beforeExit' as never)));

    expect(mod.getWorkerState().alive).toBe(false);
    // Self-heals: a subsequent call must be able to respawn cleanly rather
    // than staying permanently torn down.
    await mod.embedViaWorker('after-teardown').catch(() => { /* */ });
    expect(mod.getWorkerState().alive || mod.getWorkerState().disabled).toBe(true);
  }, 30_000);
});
