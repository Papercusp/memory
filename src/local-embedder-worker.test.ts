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
import { existsSync, mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { tmpdir } from 'node:os';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

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

describe('local-embedder-worker script path resolution (EI-21265479628494986)', () => {
  it('repairs a tsx eval pseudo-path through the package/cwd layout', async () => {
    const mod = await import('./local-embedder-worker');
    const repoRoot = resolve(__dirname, '../../../..');
    const expected = resolve(__dirname, 'local-embedder-worker.script.mjs');

    expect(
      mod.resolveWorkerScriptPath({
        filename: '[eval]',
        metaUrl: undefined,
        packageEntry: null,
        cwd: repoRoot,
      }),
    ).toBe(expected);
  });

  it('prefers a valid file URL over a pseudo filename', async () => {
    const mod = await import('./local-embedder-worker');
    const expected = resolve(__dirname, 'local-embedder-worker.script.mjs');
    const metaUrl = new URL(`file://${expected}`).href;

    expect(
      mod.resolveWorkerScriptPath({
        filename: '[eval]',
        metaUrl,
        packageEntry: null,
        cwd: '/definitely/not-a-memory-checkout',
      }),
    ).toBe(expected);
  });

  it('fails closed when no validated candidate exists', async () => {
    const mod = await import('./local-embedder-worker');

    expect(() =>
      mod.resolveWorkerScriptPath({
        filename: '[eval]',
        metaUrl: undefined,
        cwd: '/definitely/not-a-memory-checkout',
        packageEntry: null,
        exists: () => false,
      }),
    ).toThrow(/Unable to locate local-embedder-worker\.script\.mjs/);
  });
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

  it('installs at most one NEW process-level beforeExit listener per spawn, never leaks one on respawn, and the hook tears the worker down cleanly (self-heals after)', async () => {
    const mod = await import('./local-embedder-worker');
    // Start from a genuinely clean baseline — an earlier test in this same
    // file already spawned a worker (and therefore already installed the
    // module-lifetime hook), so a diff against "whatever is registered right
    // now" would always read zero-added here. Reset it explicitly instead.
    (mod as unknown as { _resetBeforeExitHookForTest: () => void })._resetBeforeExitHookForTest();
    const before = new Set(process.listeners('beforeExit'));

    await mod.embedViaWorker('one').catch(() => { /* protocol may throw */ });
    const addedByFirstSpawn = process.listeners('beforeExit').filter((fn) => !before.has(fn));
    expect(addedByFirstSpawn.length).toBe(1);
    const countAfterFirst = process.listenerCount('beforeExit');

    // Reset the WORKER (simulating a crash/respawn cycle) — but NOT the
    // hook guard, matching production: `_resetWorker` never clears
    // `_beforeExitHookInstalled`. Respawning must not add a second listener.
    await mod._resetWorker();
    await mod.embedViaWorker('two').catch(() => { /* */ });
    expect(process.listenerCount('beforeExit')).toBe(countAfterFirst);
    expect(mod.getWorkerState().alive || mod.getWorkerState().disabled).toBe(true);

    // Exercise the ACTUAL installed hook by invoking exactly the listener
    // this test installed (never every unrelated beforeExit listener some
    // other module/test may have registered on the shared `process`).
    const ours = addedByFirstSpawn;
    // The listener RETURNS its promise (see installBeforeExitHook's doc) —
    // awaiting what it returns is what makes this deterministic instead of
    // racing a fire-and-forget teardown.
    await Promise.all(ours.map((fn) => fn('beforeExit' as never)));

    expect(mod.getWorkerState().alive).toBe(false);
    // Self-heals: a subsequent call must be able to respawn cleanly rather
    // than staying permanently torn down.
    await mod.embedViaWorker('after-teardown').catch(() => { /* */ });
    expect(mod.getWorkerState().alive || mod.getWorkerState().disabled).toBe(true);
  }, 30_000);
});

/**
 * WI-37683 — the worker must hold the event loop open for exactly as long as a
 * request is in flight.
 *
 * An always-ref'd worker keeps a one-shot script alive forever (the bug the
 * unref above fixed); an always-UNREF'd one lets `beforeExit` fire MID-REQUEST,
 * terminating the worker that owes the answer. Both halves are asserted, so
 * neither "always ref" nor "always unref" can pass this block.
 */
describe('local-embedder-worker (WI-37683: holds the loop while a request is in flight)', () => {
  afterEach(async () => {
    const mod = await import('./local-embedder-worker');
    await mod._resetWorker();
  });

  it('keepAlive tracks pendingCount exactly — held while in flight, released after', async () => {
    const mod = await import('./local-embedder-worker');
    await mod.embedViaWorker('warm').catch(() => { /* protocol may throw */ });

    // IDLE ⇒ released. A real control, not decoration: "always ref" passes the
    // in-flight assertion below while re-breaking the property the unref exists
    // for (a one-off script that can never exit).
    expect(mod.getWorkerState().keepAlive).toBe(false);

    const inflight = mod.embedViaWorker('in-flight').catch(() => { /* */ });
    // `embedViaWorker` awaits `ensureWorker()` before registering the request,
    // so the pending entry appears a few microtasks later. The worker answers on
    // a MACROTASK, so draining microtasks cannot let the reply land — this
    // observes the in-flight state deterministically rather than guessing ticks.
    for (let i = 0; i < 50 && mod.getWorkerState().pendingCount === 0; i++) await Promise.resolve();

    // ⚠ ONE atomic snapshot, asserted as an invariant — deliberately NOT two
    // separate `expect(getWorkerState()...)` reads. The sibling rerank test
    // spread this across two reads and reds under fleet load
    // (EI-20053788422725852) because the request can legitimately land between
    // them: that measurement is load-fragile even though the property is real.
    const snap = mod.getWorkerState();
    expect({ pending: snap.pendingCount > 0, keepAlive: snap.keepAlive })
      .toEqual({ pending: true, keepAlive: true });

    await inflight;
    const after = mod.getWorkerState();
    expect({ pending: after.pendingCount > 0, keepAlive: after.keepAlive })
      .toEqual({ pending: false, keepAlive: false });
  }, 30_000);

  it('does not tear the worker down from beforeExit while a request is pending', async () => {
    const mod = await import('./local-embedder-worker');
    await mod.embedViaWorker('warm').catch(() => { /* */ });

    const inflight = mod.embedViaWorker('pending').catch(() => { /* */ });
    for (let i = 0; i < 50 && mod.getWorkerState().pendingCount === 0; i++) await Promise.resolve();
    expect(mod.getWorkerState().pendingCount).toBe(1);

    // Fire the hook by hand. `syncWorkerRef` should make this unreachable in a
    // real process; the guard behind it is what keeps a future ref regression
    // from silently re-opening the same hole.
    await Promise.all(process.listeners('beforeExit').map((fn) => (fn as () => unknown)()));

    await inflight;
    expect(mod.getWorkerState().alive).toBe(true);
  }, 30_000);

  it('rejects an in-flight request when the worker is torn down, instead of dropping it', async () => {
    const mod = await import('./local-embedder-worker');
    await mod.embedViaWorker('warm').catch(() => { /* */ });

    // Swallow the outbound message so this request can NEVER be answered.
    // Without that the test is a race it usually loses: the worker replies on
    // its own (with a vector, or with an ONNX load error) long before the
    // teardown lands, so the path under test — teardown finding work still
    // pending — is simply never exercised, and the test passes for the wrong
    // reason or reds on whichever message happened to win.
    const { Worker } = await import('node:worker_threads');
    const postSpy = vi.spyOn(Worker.prototype, 'postMessage').mockImplementation(() => {});
    const inflight = mod.embedViaWorker('doomed');
    for (let i = 0; i < 50 && mod.getWorkerState().pendingCount === 0; i++) await Promise.resolve();
    expect(mod.getWorkerState().pendingCount).toBe(1);
    postSpy.mockRestore();

    // The old teardown did `_pending.clear()` with no rejection, so the caller's
    // promise settled NEVER — which is what turned this defect into a process
    // that exited 0 having produced neither a vector nor an error. A silent
    // non-settlement has nothing to observe; a rejection does.
    await mod._resetWorker();
    await expect(inflight).rejects.toThrow(/in flight/);
    expect(mod.getWorkerState().pendingCount).toBe(0);
  }, 30_000);

  it('a transient runtime crash does not latch — the next call respawns instead of going inline (EI-20012631851693581)', async () => {
    const mod = await import('./local-embedder-worker');
    const { Worker } = await import('node:worker_threads');

    // Capture the live Worker instance. `syncWorkerRef` calls ref() on it before
    // the module posts a request, so this observes the real object without
    // adding a test-only export to the module.
    let captured: InstanceType<typeof Worker> | null = null;
    const realRef = Worker.prototype.ref;
    const refSpy = vi.spyOn(Worker.prototype, 'ref').mockImplementation(function (this: InstanceType<typeof Worker>) {
      captured = this;
      return realRef.call(this);
    });
    try {
      await mod.embedViaWorker('warm').catch(() => { /* protocol may throw */ });
      expect(captured).not.toBeNull();
      expect(mod.getWorkerState().disabled).toBe(false);

      // Inject exactly the signal Node delivers on a worker runtime fault. The
      // real-world causes (OOM, a native addon abort) are not cheaply stageable,
      // and the defect lives entirely in how the MAIN thread reacts to this
      // event — so the event is the right thing to reproduce.
      (captured as unknown as { emit: (e: string, a: Error) => void })
        .emit('error', new Error('simulated transient worker fault'));

      // A crash must clear the handle, NOT latch the module. `disabled` means
      // "could not be constructed", and a crash is not that.
      const crashed = mod.getWorkerState();
      expect({ alive: crashed.alive, disabled: crashed.disabled })
        .toEqual({ alive: false, disabled: false });

      // The consequence that actually mattered: the next call must reach the
      // worker path again. Pre-fix this rejected `worker disabled` forever, and
      // buildLocalEmbedder read that as "go inline" for the rest of the process.
      await mod.embedViaWorker('after-crash').catch(() => { /* protocol may throw */ });
      expect(mod.getWorkerState().alive).toBe(true);
      expect(mod.getWorkerState().disabled).toBe(false);
    } finally {
      refSpy.mockRestore();
    }
  }, 30_000);

  it('settles on a host whose event loop is otherwise idle (child process)', async () => {
    // THE reproduction. The defect only exists when nothing else holds the loop
    // open, which is unreproducible inside vitest — its own runner always has
    // work pending. So this spawns a bare host that awaits one embed and does
    // nothing else, which is every CLI, bench, migration driver and one-off
    // script.
    //
    // It asserts SETTLEMENT, not success, and that is what makes it independent
    // of the optional ONNX dependency and of any model download: pre-fix the
    // child printed NEITHER marker and exited 0, because the promise never
    // settled at all. Whether the embed then succeeds or throws is a different
    // question that other tests in this file cover.
    const dir = mkdtempSync(join(tmpdir(), 'embedder-idle-loop-'));
    try {
      const modulePath = resolve(__dirname, 'local-embedder-worker.ts');
      const child = join(dir, 'idle-host.mts');
      writeFileSync(
        child,
        `const { embedViaWorker } = await import(${JSON.stringify(modulePath)});\n` +
          `try {\n` +
          `  const v = await embedViaWorker('hello');\n` +
          `  console.log('CHILD_SETTLED ok dims=' + v.length);\n` +
          `} catch (err) {\n` +
          `  console.log('CHILD_SETTLED threw ' + (err instanceof Error ? err.message : String(err)));\n` +
          `}\n`,
      );

      const { stdout } = await promisify(execFile)('npx', ['tsx', child], {
        cwd: resolve(__dirname, '..'),
        timeout: 180_000,
      });

      expect(stdout).toContain('CHILD_SETTLED');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }, 240_000);
});
