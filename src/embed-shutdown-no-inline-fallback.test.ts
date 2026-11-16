/**
 * WI-10006602 guard: a deliberate embedder shutdown (`shutdownLocalEmbedder` /
 * `_resetWorker`) rejects every in-flight request with `EmbedWorkerShutdownError`,
 * and every builder that HAS an inline (main-thread) fallback must re-throw it
 * instead of rescuing the request by loading the model on the main thread.
 *
 * Measured 2026-10-06 (P-007 real-weight robustness, plan
 * mdenseon-adoption-measurements-2026-10-01): gemma rescued 4 torn-down requests
 * inline, blocking the sidecar's event loop ~10s (2 failed /healthz probes,
 * interrupted requests settling 8-18s late), where mdenseon — no inline path —
 * answered fast classified 500s.
 *
 * Since EI-28961860650735669 no builder has an inline path at all. Each guarded
 * case is paired with a generic worker-path error that must be warned and
 * re-thrown unchanged; an inline rescue would surface 'inline-path-entered'.
 * `buildLocalEmbedder` lives inside the worker module itself, so it is guarded
 * against a real worker in `local-embedder-worker.test.ts`.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

type Embed = (text: string) => Promise<number[]>;

const BUILDERS: Array<{ name: string; build: () => Promise<Embed> }> = [
  { name: 'gemma', build: async () => (await import('./gemma-embedder')).buildGemmaEmbedder({ kind: 'query' }) },
  { name: 'harrier', build: async () => (await import('./harrier-embedder')).buildHarrierEmbedder({ kind: 'query' }) },
  { name: 'qwen3', build: async () => (await import('./qwen3-embedder')).buildQwen3Embedder({ kind: 'query' }) },
  { name: 'granite', build: async () => (await import('./granite-embedder')).buildGraniteEmbedder({ variant: '97m' }) },
];

function mockWorkerPath() {
  const embedViaWorker = vi.fn();
  const warnEmbedFallback = vi.fn();
  vi.doMock('./local-embedder-worker', async (importOriginal) => {
    const actual = await importOriginal<typeof import('./local-embedder-worker')>();
    return {
      ...actual,
      embedViaWorker,
      getWorkerState: () => ({ alive: true, disabled: false, pendingCount: 0 }),
      warnEmbedFallback,
    };
  });
  // The first thing every inline branch does is resolve transformers through
  // dynamicImport, so a call here IS "the inline path was entered".
  const dynamicImport = vi.fn(async () => {
    throw new Error('inline-path-entered');
  });
  vi.doMock('./dynamic-import', () => ({ dynamicImport }));
  return { embedViaWorker, warnEmbedFallback, dynamicImport };
}

beforeEach(() => {
  vi.resetModules();
});

afterEach(() => {
  vi.doUnmock('./local-embedder-worker');
  vi.doUnmock('./dynamic-import');
});

describe('builders never rescue a deliberate worker shutdown inline (WI-10006602)', () => {
  for (const b of BUILDERS) {
    it(`${b.name}: re-throws EmbedWorkerShutdownError without entering the inline path`, async () => {
      const m = mockWorkerPath();
      const { EmbedWorkerShutdownError } = await import('./local-embedder-worker');
      const shutdown = new EmbedWorkerShutdownError(1);
      m.embedViaWorker.mockRejectedValue(shutdown);
      const embed = await b.build();

      await expect(embed('hello')).rejects.toBe(shutdown);
      expect(m.embedViaWorker).toHaveBeenCalledTimes(1);
      expect(m.warnEmbedFallback).not.toHaveBeenCalled();
      expect(m.dynamicImport).not.toHaveBeenCalled();
    });

    // EI-28961860650735669: there is no inline path any more. A generic worker
    // error is warned (it IS a worker failure, unlike a shutdown) and rejects
    // with that same error. Any inline rescue would reject with
    // 'inline-path-entered' instead, so `rejects.toBe(failure)` is the guard.
    it(`${b.name}: a generic worker-path error is warned and re-thrown, never retried inline`, async () => {
      const m = mockWorkerPath();
      const failure = new Error('transient worker hiccup');
      m.embedViaWorker.mockRejectedValue(failure);
      const embed = await b.build();

      await expect(embed('hello')).rejects.toBe(failure);
      expect(m.warnEmbedFallback).toHaveBeenCalledTimes(1);
      expect(m.dynamicImport).not.toHaveBeenCalled();
    });
  }
});

describe('isEmbedWorkerShutdownError', () => {
  it('matches by code (so a split module record still agrees), not by message', async () => {
    const { isEmbedWorkerShutdownError, EmbedWorkerShutdownError, EMBED_WORKER_SHUTDOWN_CODE } = await import(
      './local-embedder-worker'
    );
    expect(isEmbedWorkerShutdownError(new EmbedWorkerShutdownError(2))).toBe(true);
    expect(isEmbedWorkerShutdownError(Object.assign(new Error('x'), { code: EMBED_WORKER_SHUTDOWN_CODE }))).toBe(true);
    expect(isEmbedWorkerShutdownError(new Error('embedder worker was shut down while 1 request(s) were in flight'))).toBe(
      false,
    );
    expect(isEmbedWorkerShutdownError(null)).toBe(false);
    expect(isEmbedWorkerShutdownError(EMBED_WORKER_SHUTDOWN_CODE)).toBe(false);
  });
});
