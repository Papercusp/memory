/**
 * WI-10006567 guard: tearing the embed worker down while ONNX inference is in
 * flight must not abort the host process, and a retired worker's late messages
 * must never reach the requests of the worker that replaced it.
 *
 * The defect (measured 2026-10-06, P-007 real-weight robustness run): calling
 * `shutdownLocalEmbedder()` with requests in flight ran `Worker#terminate()`
 * under a running onnxruntime-node session. When the native run completed into
 * the destroyed environment it threw a `Napi::Error` nothing could catch, and
 * the whole PROCESS died with SIGABRT (exit 134). An idle terminate was fine,
 * which is the control below.
 *
 * The fix retires the worker instead: callers are rejected at once and a new
 * worker can spawn immediately, while the old one finishes the native runs it
 * already started, answers `{kind:'retired'}`, and only then is terminated.
 * Detaching the old worker opens a second hazard, which the scoping test pins:
 * request ids restart at 0, so a late `embed_ok` from the retired worker could
 * otherwise resolve the NEW worker's request with the wrong vector.
 *
 * The real-binding case runs in a child process because the failure kills the
 * whole process, which would take the vitest worker with it.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { execFile } from 'node:child_process';

type WorkerInstance = InstanceType<typeof import('node:worker_threads').Worker>;

/** Swallow outbound `embed` messages only, so a request stays pending forever
 *  while the retire handshake (and everything else) still reaches the worker. */
async function swallowEmbedPosts() {
  const { Worker } = await import('node:worker_threads');
  const realPost = Worker.prototype.postMessage;
  return vi.spyOn(Worker.prototype, 'postMessage').mockImplementation(function (this: WorkerInstance, msg: unknown) {
    if ((msg as { kind?: string } | null)?.kind === 'embed') return;
    return realPost.call(this, msg);
  });
}

/** Capture every live Worker the module refs (it refs before posting a request). */
async function captureWorkers() {
  const { Worker } = await import('node:worker_threads');
  const realRef = Worker.prototype.ref;
  const seen: WorkerInstance[] = [];
  const spy = vi.spyOn(Worker.prototype, 'ref').mockImplementation(function (this: WorkerInstance) {
    if (!seen.includes(this)) seen.push(this);
    return realRef.call(this);
  });
  return { seen, spy };
}

async function waitForPending(mod: typeof import('./local-embedder-worker'), n: number) {
  await vi.waitFor(() => expect(mod.getWorkerState().pendingCount).toBe(n));
}

afterEach(async () => {
  vi.restoreAllMocks();
  const mod = await import('./local-embedder-worker');
  await mod._resetWorker();
  mod._resetBeforeExitHookForTest();
});

describe('local-embedder-worker retirement (WI-10006567)', () => {
  it("a retired worker's late messages never settle the replacement worker's requests", async () => {
    const mod = await import('./local-embedder-worker');
    const post = await swallowEmbedPosts();
    const { seen } = await captureWorkers();

    const first = mod.embedViaWorker('owed by the first worker').then(() => 'answered', (e: Error) => `rejected: ${e.message}`);
    await waitForPending(mod, 1);
    const oldWorker = seen.at(-1)!;
    await mod._resetWorker();
    expect(await first).toMatch(/shut down while 1 request/);

    // Request ids restart at 0, so the replacement's first request reuses the id
    // the retired worker still "owes".
    let settled: string | null = null;
    const second = mod.embedViaWorker('owed by the second worker').then(
      (v) => { settled = `resolved ${JSON.stringify(v)}`; },
      (e: Error) => { settled = `rejected: ${e.message}`; },
    );
    await waitForPending(mod, 1);
    const newWorker = seen.at(-1)!;
    expect(newWorker).not.toBe(oldWorker);

    // The retired worker's late answer for id 0, then its exit. Neither may touch
    // the replacement: pre-fix the first resolved `second` with this vector and
    // the second nulled the live worker handle.
    oldWorker.emit('message', { kind: 'embed_ok', id: 0, vector: [9, 9, 9] });
    oldWorker.emit('message', { kind: 'embed_err', id: 0, error: 'late error from a retired worker' });
    oldWorker.emit('exit', 1);
    await new Promise((r) => setTimeout(r, 20));
    expect(settled).toBeNull();
    expect(mod.getWorkerState()).toMatchObject({ alive: true, pendingCount: 1 });

    post.mockRestore();
    await mod._resetWorker();
    await second;
    expect(settled).toMatch(/shut down while 1 request/);
  }, 30_000);

  it('terminates a worker that owed answers only after it reports retired', async () => {
    const mod = await import('./local-embedder-worker');
    const { Worker } = await import('node:worker_threads');
    await swallowEmbedPosts();
    const { seen } = await captureWorkers();

    const owed = mod.embedViaWorker('pending at shutdown').catch((e: Error) => e);
    await waitForPending(mod, 1);
    const w = seen.at(-1)!;
    let retiredSeen = false;
    w.on('message', (m: { kind?: string }) => { if (m?.kind === 'retired') retiredSeen = true; });
    const realTerminate = Worker.prototype.terminate;
    const terminatedAfterRetired: boolean[] = [];
    vi.spyOn(Worker.prototype, 'terminate').mockImplementation(function (this: WorkerInstance) {
      if (this === w) terminatedAfterRetired.push(retiredSeen);
      return realTerminate.call(this);
    });

    await mod._resetWorker();
    expect(await owed).toBeInstanceOf(Error);
    // The control: an immediate terminate (the pre-fix teardown) records `false`.
    expect(terminatedAfterRetired).toEqual([true]);
    expect(mod.getWorkerState()).toMatchObject({ alive: false, pendingCount: 0 });
  }, 30_000);

  it('bounds the drain: a worker that never reports retired is terminated anyway, loudly', async () => {
    const mod = await import('./local-embedder-worker');
    const { Worker } = await import('node:worker_threads');
    const realPost = Worker.prototype.postMessage;
    // Swallow the embed AND the retire request: the worker never answers.
    vi.spyOn(Worker.prototype, 'postMessage').mockImplementation(function (this: WorkerInstance, msg: unknown) {
      const kind = (msg as { kind?: string } | null)?.kind;
      if (kind === 'embed' || kind === 'retire') return;
      return realPost.call(this, msg);
    });
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const owed = mod.embedViaWorker('never answered').catch((e: Error) => e);
    await waitForPending(mod, 1);
    const t0 = Date.now();
    await mod._resetWorker({ drainMs: 100 });
    expect(Date.now() - t0).toBeLessThan(5_000);
    expect(await owed).toBeInstanceOf(Error);
    expect(warn.mock.calls.some(([m]) => /did not report retired within 100ms/.test(String(m)))).toBe(true);
    expect(mod.getWorkerState()).toMatchObject({ alive: false, pendingCount: 0 });
  }, 30_000);
});

/** Where transformers.js caches models; the real-binding case needs BGE-small there. */
function cachedBgeSmall(): boolean {
  try {
    // The package's `exports` map hides package.json, so resolve the entry and
    // walk up to the package root (transformers.js caches under `<root>/.cache`).
    let dir = dirname(createRequire(import.meta.url).resolve('@huggingface/transformers'));
    while (!(dir.endsWith(join('@huggingface', 'transformers'))) && dirname(dir) !== dir) dir = dirname(dir);
    return existsSync(join(dir, '.cache', 'Xenova', 'bge-small-en-v1.5'));
  } catch {
    return false;
  }
}

const IN_FLIGHT = 48;
/** The module under test. Overridable so the same child can be pointed at a
 *  pre-fix copy outside the tree as the control (must abort), without ever
 *  mutating the shared checkout. */
const SUBJECT = process.env.EMBED_RETIRE_TEST_SUBJECT ?? resolve(__dirname, 'local-embedder-worker.ts');

/** The child: warm, idle reset (control), reset with native inference running, recover. */
function childScript(modulePath: string): string {
  return `
const mod = await import(${JSON.stringify(modulePath)});
const long = Array.from({ length: 700 }, (_, i) => 'token' + (i % 97)).join(' ');
const ref = await mod.embedViaWorker(long);
console.log('CHILD_WARM dims=' + ref.length);
await mod.shutdownLocalEmbedder();
console.log('CHILD_IDLE_RESET_OK');
await mod.embedViaWorker('reload the model in a fresh worker');
// Reset the moment the FIRST of a large batch answers: inference has then
// demonstrably run, and the rest of the batch is queued on or running in
// onnxruntime's pool. (A CPU-time gate was tried first and was timing-fragile:
// on a fast run all requests finished before it tripped, a vacuous pass.)
let firstAnswered = () => {};
const first = new Promise((r) => { firstAnswered = r; });
let answeredBeforeReset = 0;
const flights = Array.from({ length: ${IN_FLIGHT} }, () =>
  mod.embedViaWorker(long).then(() => { answeredBeforeReset++; firstAnswered(); return 'answered'; },
    (e) => 'rejected: ' + e.message));
await Promise.race([first, new Promise((r) => setTimeout(r, 60000))]);
const pendingAtReset = mod.getWorkerState().pendingCount;
const answeredAtReset = answeredBeforeReset;
await mod.shutdownLocalEmbedder();
const outcomes = await Promise.all(flights);
console.log('CHILD_RESET ' + JSON.stringify({ pendingAtReset, answeredAtReset,
  rejected: outcomes.filter((o) => o.startsWith('rejected: embedder worker was shut down')).length,
  answered: outcomes.filter((o) => o === 'answered').length, other: outcomes.filter((o) => o !== 'answered' && !o.startsWith('rejected: embedder worker was shut down')) }));
const after = await mod.embedViaWorker(long);
const same = after.length === ref.length && after.every((x, i) => Math.abs(x - ref[i]) < 1e-6);
console.log('CHILD_AFTER same=' + same);
await mod.shutdownLocalEmbedder();
console.log('CHILD_DONE');
`;
}

describe('local-embedder-worker retirement with the real ONNX binding (WI-10006567)', () => {
  it.skipIf(!cachedBgeSmall())('survives a reset with native inference in flight, and recovers to the same vector', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'embedder-retire-'));
    try {
      const child = join(dir, 'retire-host.mts');
      writeFileSync(child, childScript(SUBJECT));
      const run = await new Promise<{ code: number | null; signal: NodeJS.Signals | null; stdout: string; stderr: string }>((done) => {
        execFile('npx', ['tsx', child], { cwd: resolve(__dirname, '..'), timeout: 180_000, maxBuffer: 8 << 20 },
          (err, stdout, stderr) => {
            const e = err as (Error & { code?: number | string; signal?: NodeJS.Signals }) | null;
            done({ code: e ? (typeof e.code === 'number' ? e.code : null) : 0, signal: e?.signal ?? null, stdout, stderr });
          });
      });
      const evidence = `exit=${run.code} signal=${run.signal}\nstdout:\n${run.stdout}\nstderr:\n${run.stderr.slice(-2000)}`;
      // The idle control must pass on any build: if it does not, the harness is
      // broken and the in-flight assertions below would prove nothing.
      expect(run.stdout, evidence).toContain('CHILD_IDLE_RESET_OK');
      // The defect: SIGABRT (134) with "terminate called after throwing an
      // instance of 'Napi::Error'" right after the in-flight reset.
      expect(run.stderr, evidence).not.toMatch(/Napi::Error|terminate called/);
      expect({ code: run.code, signal: run.signal }, evidence).toEqual({ code: 0, signal: null });
      const reset = JSON.parse(run.stdout.match(/CHILD_RESET (.*)/)![1]) as
        { pendingAtReset: number; answeredAtReset: number; rejected: number; answered: number; other: string[] };
      // Non-vacuity: inference had run (one answer arrived) and the reset still
      // found requests in flight. Both must hold or the reset proved nothing.
      expect(reset.answeredAtReset, evidence).toBeGreaterThan(0);
      expect(reset.pendingAtReset, evidence).toBeGreaterThan(0);
      // Every cut-off caller got the classified shutdown rejection, nothing else.
      expect(reset.other, evidence).toEqual([]);
      expect(reset.rejected + reset.answered, evidence).toBe(IN_FLIGHT);
      expect(reset.rejected, evidence).toBe(reset.pendingAtReset);
      expect(run.stdout, evidence).toContain('CHILD_AFTER same=true');
      expect(run.stdout, evidence).toContain('CHILD_DONE');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }, 240_000);
});
