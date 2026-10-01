/**
 * WI-10005090 guard: the ONNX native binding must stay loadable in this process
 * after the worker that first loaded it has terminated.
 *
 * Without the pin, a terminated embed worker (a crash respawn, or
 * `recycleEmbedWorker()` on a device change) leaves every later load in the
 * process failing with "Module did not self-register" — the respawned worker
 * AND the embedders' main-thread fallback — so local embedding is dead until a
 * restart. `pinOnnxRuntimeBinding` holds one reference in the spawning thread.
 *
 * Real binding, no model, no `Worker` mock: each "load" is a fresh worker
 * thread requiring onnxruntime-node and building a tensor. The UNPINNED arm runs
 * in a separate plain-node process, because once this process holds the pin it
 * can no longer show the hazard. That control is what makes the pinned arm
 * mean anything: if it stops reproducing (an onnxruntime-node upgrade that
 * registers correctly), the control fails and says so, and the pin can be
 * reconsidered. Skips only when onnxruntime-node is not installed.
 */
import { describe, it, expect, afterAll, vi } from 'vitest';
import { Worker } from 'node:worker_threads';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { createRequire } from 'node:module';
import {
  _resetBeforeExitHookForTest,
  _resetOnnxBindingPinForTest,
  embedViaWorker,
  getOnnxBindingPin,
  getWorkerState,
  pinOnnxRuntimeBinding,
  resolveWorkerScriptPath,
  shutdownLocalEmbedder,
} from './local-embedder-worker';

const execFileAsync = promisify(execFile);

/** Plain CJS worker body: load the binding by absolute path, touch it, report. */
const LOAD_BINDING_IN_WORKER = `
const { parentPort, workerData } = require('node:worker_threads');
try {
  const ort = require(workerData.path);
  const t = new ort.Tensor('float32', new Float32Array([1, 2, 3]), [3]);
  parentPort.postMessage({ ok: true, dims: t.dims });
} catch (err) {
  parentPort.postMessage({ ok: false, error: String((err && err.message) || err) });
}
`;

type LoadResult = { ok: true; dims: number[] } | { ok: false; error: string };

async function loadInFreshWorker(path: string): Promise<LoadResult> {
  const w = new Worker(LOAD_BINDING_IN_WORKER, { eval: true, workerData: { path } });
  try {
    return await new Promise<LoadResult>((resolve, reject) => {
      w.once('message', resolve);
      w.once('error', reject);
    });
  } finally {
    await w.terminate();
  }
}

/** onnxruntime-node as transformers would resolve it, or null when absent. */
function resolveOnnxFromTransformers(): string | null {
  try {
    const transformersEntry = createRequire(resolveWorkerScriptPath()).resolve('@huggingface/transformers');
    return createRequire(transformersEntry).resolve('onnxruntime-node');
  } catch {
    return null;
  }
}

const onnxPath = resolveOnnxFromTransformers();
const describeIfOnnx = onnxPath ? describe : describe.skip;

afterAll(async () => {
  await shutdownLocalEmbedder();
});

describeIfOnnx('ONNX binding pin (WI-10005090)', () => {
  it('unpinned control: a second worker cannot load the binding after the first terminated', async () => {
    // Separate process: this one may already hold the pin (other tests in this file).
    const script = `
      const { Worker } = require('node:worker_threads');
      const body = ${JSON.stringify(LOAD_BINDING_IN_WORKER)};
      const path = process.argv[1];
      function load() {
        const w = new Worker(body, { eval: true, workerData: { path } });
        return new Promise((res, rej) => { w.once('message', (m) => w.terminate().then(() => res(m))); w.once('error', rej); });
      }
      (async () => {
        const first = await load();
        const second = await load();
        process.stdout.write(JSON.stringify({ first, second }));
      })().catch((e) => { process.stdout.write(JSON.stringify({ crashed: String(e) })); });
    `;
    const { stdout } = await execFileAsync(process.execPath, ['-e', script, onnxPath!], { timeout: 60_000 });
    const out = JSON.parse(stdout) as { first: LoadResult; second: LoadResult };
    expect(out.first, 'the first load must work, or this control measures nothing').toMatchObject({ ok: true });
    expect(
      out.second,
      'the UNPINNED second load no longer fails: onnxruntime-node may now register correctly on reload. ' +
        'The pinned arm below no longer proves anything; re-measure and consider retiring pinOnnxRuntimeBinding.',
    ).toMatchObject({ ok: false, error: expect.stringMatching(/did not self-register/) });
  }, 90_000);

  it('pinned: the binding loads again in a fresh worker after the first worker terminated', async () => {
    const pin = pinOnnxRuntimeBinding(resolveWorkerScriptPath());
    expect(pin).toEqual({ status: 'pinned', path: onnxPath });
    // Same file transformers will load, so the pin covers the real worker.
    expect(await loadInFreshWorker(onnxPath!)).toMatchObject({ ok: true, dims: [3] });
    expect(await loadInFreshWorker(onnxPath!)).toMatchObject({ ok: true, dims: [3] });
    expect(await loadInFreshWorker(onnxPath!)).toMatchObject({ ok: true, dims: [3] });
  }, 60_000);

  it('spawning the embed worker pins the binding before the worker starts', async () => {
    _resetOnnxBindingPinForTest();
    expect(getOnnxBindingPin()).toBeNull();
    // Keep the embed request from reaching the worker: no model is loaded or
    // downloaded. The request stays pending until shutdown rejects it.
    const post = vi.spyOn(Worker.prototype, 'postMessage').mockImplementation(() => {});
    const pending = embedViaWorker('pin-order probe').catch((error: unknown) => error);
    try {
      await vi.waitFor(() => expect(getWorkerState().pendingCount).toBe(1));
      expect(getOnnxBindingPin()).toEqual({ status: 'pinned', path: onnxPath });
    } finally {
      post.mockRestore();
      await shutdownLocalEmbedder();
      await pending;
      _resetBeforeExitHookForTest();
    }
  }, 60_000);
});

describe('ONNX binding pin when transformers is not installed', () => {
  it('reports unavailable without throwing, and the first outcome stands', () => {
    _resetOnnxBindingPinForTest();
    try {
      const bogus = '/nonexistent/dir/local-embedder-worker.script.mjs';
      expect(pinOnnxRuntimeBinding(bogus)).toMatchObject({ status: 'unavailable' });
      expect(getOnnxBindingPin()).toMatchObject({ status: 'unavailable' });
    } finally {
      // The record only; a binding loaded earlier in this process stays loaded.
      _resetOnnxBindingPinForTest();
    }
  });
});
