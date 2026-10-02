/** Pinned local ONNX candidate. Availability does not select a production model. */
import fs from 'node:fs';
import path from 'node:path';
import { embedViaWorker, type WorkerInputTrace } from './local-embedder-worker';
import { CANDIDATE_DIM_SPECS } from './embedder-dims';

export const MDENSEON_MODEL = 'lightonai/mDenseOn';
export const MDENSEON_REVISION = 'a5fdb000f7a21da96c3bddde3a782ef777316df3';
export const MDENSEON_NATIVE_DIMS = CANDIDATE_DIM_SPECS.mdenseon.nativeDims;
export type MdenseOnKind = 'query' | 'document';

export function mdenseOnPrompt(kind: MdenseOnKind, text: string): string {
  return `${kind}: ${text}`;
}

export interface MdenseOnExportManifest {
  formatVersion: number;
  model: string;
  revision: string;
  dimensions: number;
  pooling: string;
  normalization: string;
  maxLength: number;
  output: string;
  weightDtype: string;
  prompts: { query: string; document: string };
  files: Record<string, { bytes: number; sha256: string }>;
}

export function readMdenseOnExport(model: string): MdenseOnExportManifest {
  if (!path.isAbsolute(model)) throw new Error('mDenseOn requires an absolute local exported model directory');
  const manifest = JSON.parse(fs.readFileSync(path.join(model, 'export-manifest.json'), 'utf8')) as MdenseOnExportManifest;
  if (manifest.formatVersion !== 1 || manifest.model !== MDENSEON_MODEL || manifest.revision !== MDENSEON_REVISION ||
      manifest.dimensions !== MDENSEON_NATIVE_DIMS || manifest.pooling !== 'cls' || manifest.normalization !== 'l2' ||
      manifest.maxLength !== 8192 || manifest.output !== 'last_hidden_state' || manifest.weightDtype !== 'fp32' ||
      manifest.prompts?.query !== 'query: ' || manifest.prompts?.document !== 'document: ' ||
      !manifest.files?.['onnx/model.onnx']?.sha256) {
    throw new Error('mDenseOn export contract mismatch: expected pinned native-768 CLS/l2 fp32 graph and asymmetric prefixes');
  }
  const tokenizerConfig = JSON.parse(fs.readFileSync(path.join(model, 'tokenizer_config.json'), 'utf8')) as { model_max_length?: number };
  if (tokenizerConfig.model_max_length !== manifest.maxLength) {
    throw new Error('mDenseOn tokenizer context limit disagrees with the exported contract');
  }
  return manifest;
}

/** Reuses the shipped worker and its thread/device policy. No inline fallback:
 * sidecar validation must fail if inference would move onto the main loop. */
export function buildMdenseOnEmbedder(opts: { kind: MdenseOnKind; model: string; onInputTrace?: (trace: WorkerInputTrace) => void }): (text: string) => Promise<number[]> {
  readMdenseOnExport(opts.model);
  return async (text) => {
    const vector = await embedViaWorker(mdenseOnPrompt(opts.kind, text), {
      model: opts.model, pooling: 'cls', normalize: true, tokenizerBackend: 'rust',
      ...(opts.onInputTrace ? { onInputTrace: opts.onInputTrace } : {}),
    });
    const norm = Math.sqrt(vector.reduce((sum, x) => sum + x * x, 0));
    if (vector.length !== MDENSEON_NATIVE_DIMS || vector.some((x) => !Number.isFinite(x)) || Math.abs(norm - 1) > 0.001) {
      throw new Error('mDenseOn worker returned an invalid native-768 unit vector');
    }
    return vector;
  };
}
