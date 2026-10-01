/**
 * WI-10005070: the embed worker (and with it the model) is released after the
 * host has embedded nothing for `PAPERCUSP_EMBED_WORKER_IDLE_MS`, and the next
 * embed respawns it.
 *
 * The worker thread is real; its replies are not. `Worker#postMessage` is
 * replaced by a stub that answers each embed request with a tiny vector, so no
 * model is loaded or downloaded. The real-model round trip (embed, recycle,
 * embed; embed, idle-unload, embed) runs only with PAPERCUSP_EMBED_E2E=1,
 * because it needs the EmbeddingGemma model files on disk.
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import { Worker } from 'node:worker_threads';
import { createRequire } from 'node:module';
import {
  DEFAULT_EMBED_WORKER_IDLE_MS,
  EMBED_WORKER_IDLE_MS_ENV,
  _resetBeforeExitHookForTest,
  embedViaWorker,
  embedWorkerIdleMs,
  getEmbedWorkerIdleStats,
  getOnnxBindingPin,
  getWorkerState,
  recycleEmbedWorker,
  resolveWorkerScriptPath,
  shutdownLocalEmbedder,
} from './local-embedder-worker';

/** Whether onnxruntime-node is installed where the worker would load it. Without
 *  it the binding cannot be pinned, so the unload is (correctly) never armed and
 *  the timer tests have nothing to observe. */
function onnxInstalled(): boolean {
  try {
    const transformersEntry = createRequire(resolveWorkerScriptPath()).resolve('@huggingface/transformers');
    createRequire(transformersEntry).resolve('onnxruntime-node');
    return true;
  } catch {
    return false;
  }
}

const savedIdle = process.env[EMBED_WORKER_IDLE_MS_ENV];

function setIdleMs(value: string | undefined): void {
  if (value === undefined) delete process.env[EMBED_WORKER_IDLE_MS_ENV];
  else process.env[EMBED_WORKER_IDLE_MS_ENV] = value;
}

/** Answer every embed request from the worker side without touching the model. */
function stubWorkerReplies(opts: { reply: boolean }) {
  return vi.spyOn(Worker.prototype, 'postMessage').mockImplementation(function (this: Worker, msg: unknown) {
    const m = msg as { kind?: string; id?: number };
    if (!opts.reply || m.kind !== 'embed' || typeof m.id !== 'number') return;
    const id = m.id;
    setImmediate(() => this.emit('message', { kind: 'embed_ok', id, vector: [0.25, 0.5] }));
  });
}

afterEach(async () => {
  vi.restoreAllMocks();
  await shutdownLocalEmbedder();
  _resetBeforeExitHookForTest();
  setIdleMs(savedIdle);
});

describe('embedWorkerIdleMs', () => {
  it('defaults to 10 minutes and accepts a non-negative override', () => {
    expect(embedWorkerIdleMs({})).toBe(DEFAULT_EMBED_WORKER_IDLE_MS);
    expect(DEFAULT_EMBED_WORKER_IDLE_MS).toBe(600_000);
    expect(embedWorkerIdleMs({ [EMBED_WORKER_IDLE_MS_ENV]: '2500' })).toBe(2500);
    expect(embedWorkerIdleMs({ [EMBED_WORKER_IDLE_MS_ENV]: '0' })).toBe(0);
  });

  it('keeps the default for a malformed or negative value', () => {
    for (const bad of ['', '  ', 'soon', '-5', 'NaN']) {
      expect(embedWorkerIdleMs({ [EMBED_WORKER_IDLE_MS_ENV]: bad })).toBe(DEFAULT_EMBED_WORKER_IDLE_MS);
    }
  });
});

describe.skipIf(!onnxInstalled())('embed worker idle unload (WI-10005070)', () => {
  it('releases the worker after the idle window and respawns on the next embed', async () => {
    setIdleMs('150');
    stubWorkerReplies({ reply: true });
    const before = getEmbedWorkerIdleStats().idleUnloads;

    expect(await embedViaWorker('first')).toEqual([0.25, 0.5]);
    // The unload is only armed when the binding is pinned (WI-10005090).
    expect(getOnnxBindingPin()).toMatchObject({ status: 'pinned' });
    expect(getEmbedWorkerIdleStats().armed).toBe(true);
    expect(getWorkerState().alive).toBe(true);

    await vi.waitFor(() => expect(getWorkerState().alive).toBe(false), { timeout: 5_000, interval: 25 });
    expect(getEmbedWorkerIdleStats()).toMatchObject({ armed: false, idleUnloads: before + 1 });

    expect(await embedViaWorker('after unload')).toEqual([0.25, 0.5]);
    expect(getWorkerState().alive).toBe(true);
  }, 20_000);

  it('a busy host never unloads: each embed restarts the window', async () => {
    setIdleMs('400');
    stubWorkerReplies({ reply: true });
    const before = getEmbedWorkerIdleStats().idleUnloads;
    const started = Date.now();
    // Embeds every ~100 ms for ~1 s, well past one 400 ms window.
    while (Date.now() - started < 1_000) {
      await embedViaWorker('steady');
      await new Promise((r) => setTimeout(r, 100));
    }
    expect(getWorkerState().alive).toBe(true);
    expect(getEmbedWorkerIdleStats().idleUnloads).toBe(before);
  }, 20_000);

  it('does not arm while a request is in flight', async () => {
    setIdleMs('100');
    stubWorkerReplies({ reply: false });
    const pending = embedViaWorker('never answered').catch((error: unknown) => error);
    await vi.waitFor(() => expect(getWorkerState().pendingCount).toBe(1));
    await new Promise((r) => setTimeout(r, 300));
    expect(getEmbedWorkerIdleStats().armed).toBe(false);
    expect(getWorkerState().alive).toBe(true);
    await shutdownLocalEmbedder();
    expect(await pending).toBeInstanceOf(Error);
  }, 20_000);

  it('a window of 0 disables the unload', async () => {
    setIdleMs('0');
    stubWorkerReplies({ reply: true });
    await embedViaWorker('kept');
    expect(getEmbedWorkerIdleStats().armed).toBe(false);
    await new Promise((r) => setTimeout(r, 200));
    expect(getWorkerState().alive).toBe(true);
  }, 20_000);
});

const GEMMA = 'onnx-community/embeddinggemma-300m-ONNX';

describe.skipIf(process.env.PAPERCUSP_EMBED_E2E !== '1')('real model: embedding survives worker restarts', () => {
  it('embeds after a recycle and after an idle unload (WI-10005090 + WI-10005070)', async () => {
    const opts = { model: GEMMA, normalize: false };
    const first = await embedViaWorker('task: search result | query: release gate holder', opts);
    expect(first.length).toBe(768);

    await recycleEmbedWorker();
    expect(getWorkerState().alive).toBe(false);
    const afterRecycle = await embedViaWorker('task: search result | query: release gate holder', opts);
    expect(afterRecycle.length).toBe(768);

    setIdleMs('200');
    await embedViaWorker('task: search result | query: arm the idle window', opts);
    await vi.waitFor(() => expect(getWorkerState().alive).toBe(false), { timeout: 10_000, interval: 50 });
    const afterIdle = await embedViaWorker('task: search result | query: release gate holder', opts);
    expect(afterIdle).toEqual(afterRecycle);
  }, 120_000);
});
