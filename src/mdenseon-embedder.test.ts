import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
const worker = vi.hoisted(() => ({ embed: vi.fn() }));
vi.mock('./local-embedder-worker', () => ({ embedViaWorker: worker.embed }));
import { buildMdenseOnEmbedder, mdenseOnPrompt, MDENSEON_MODEL, MDENSEON_REVISION, readMdenseOnExport } from './mdenseon-embedder';

const dirs: string[] = [];
function artifact(overrides: Record<string, unknown> = {}): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'mdenseon-contract-'));
  dirs.push(dir);
  fs.writeFileSync(path.join(dir, 'export-manifest.json'), JSON.stringify({
    formatVersion: 1, model: MDENSEON_MODEL, revision: MDENSEON_REVISION, dimensions: 768,
    pooling: 'cls', normalization: 'l2', maxLength: 8192, output: 'last_hidden_state', weightDtype: 'fp32',
    prompts: { query: 'query: ', document: 'document: ' }, files: { 'onnx/model.onnx': { bytes: 1, sha256: 'pinned-graph' } },
    ...overrides,
  }));
  fs.writeFileSync(path.join(dir, 'tokenizer_config.json'), JSON.stringify({ model_max_length: 8192 }));
  return dir;
}
afterEach(() => { vi.resetAllMocks(); for (const dir of dirs.splice(0)) fs.rmSync(dir, { recursive: true, force: true }); });

describe('pinned mDenseOn worker contract', () => {
  it('honors both publisher prefixes and uses native CLS/l2 worker inference', async () => {
    const model = artifact();
    const vector = new Array(768).fill(1 / Math.sqrt(768));
    worker.embed.mockResolvedValue(vector);
    expect(await buildMdenseOnEmbedder({ model, kind: 'query' })('question')).toEqual(vector);
    await buildMdenseOnEmbedder({ model, kind: 'document' })('passage');
    expect(worker.embed.mock.calls).toEqual([
      ['query: question', { model, pooling: 'cls', normalize: true, tokenizerBackend: 'rust' }],
      ['document: passage', { model, pooling: 'cls', normalize: true, tokenizerBackend: 'rust' }],
    ]);
    expect(mdenseOnPrompt('document', '')).toBe('document: ');
  });

  it.each([{ revision: 'wrong' }, { pooling: 'mean' }, { dimensions: 384 }, { weightDtype: 'q8' }, { normalization: 'none' }])(
    'refuses a different space instead of treating matching files as the candidate: %j', (change) => {
      expect(() => readMdenseOnExport(artifact(change))).toThrow('contract mismatch');
    });

  it('refuses missing artifacts and tokenizer-limit drift', () => {
    expect(() => readMdenseOnExport('remote/model')).toThrow('absolute local');
    const model = artifact();
    fs.writeFileSync(path.join(model, 'tokenizer_config.json'), JSON.stringify({ model_max_length: 512 }));
    expect(() => readMdenseOnExport(model)).toThrow('context limit');
  });

  it.each([new Array(384).fill(1), new Array(768).fill(0), new Array(768).fill(Number.NaN), new Array(768).fill(1)].map((vector) => [vector]))(
    'rejects malformed or unnormalized vectors', async (vector) => {
      worker.embed.mockResolvedValue(vector);
      await expect(buildMdenseOnEmbedder({ model: artifact(), kind: 'query' })('q')).rejects.toThrow('invalid native-768');
    });

  it('propagates worker failure without an inline model load', async () => {
    worker.embed.mockRejectedValue(new Error('worker unavailable'));
    await expect(buildMdenseOnEmbedder({ model: artifact(), kind: 'query' })('q')).rejects.toThrow('worker unavailable');
    expect(worker.embed).toHaveBeenCalledOnce();
  });
  it('passes explicit input-evidence collection through the pinned worker route', async () => {
    const model = artifact(), onInputTrace = vi.fn(), onInferenceTrace = vi.fn(), onNativeInferenceTrace = vi.fn();
    worker.embed.mockResolvedValue(new Array(768).fill(1 / Math.sqrt(768)));
    await buildMdenseOnEmbedder({ model, kind: 'query', onInputTrace, onInferenceTrace, onNativeInferenceTrace })('q');
    expect(worker.embed).toHaveBeenCalledWith('query: q', { model, pooling: 'cls', normalize: true,
      tokenizerBackend: 'rust', onInputTrace, onInferenceTrace, onNativeInferenceTrace });
  });
});
