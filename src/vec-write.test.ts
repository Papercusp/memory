/**
 * vec-write — the shared "write a row's vector without touching payload.data"
 * primitive (EI-10048 write-time augmentation + the re-embed pass).
 *
 * The PURE helpers (vecUpsertSql, toVectorLiteral) and embedAndUpsertVector's
 * BEST-EFFORT guards (disabled embedder / wrong-width vector → false, never
 * throw) are unit-testable without PG. The happy-path upsert needs a live
 * pgvector store and is exercised by the bench + live-verify, not here.
 */
import { describe, it, expect } from 'vitest';
import {
  vecUpsertSql,
  toVectorLiteral,
  embedAndUpsertVector,
  VEC_TABLE,
  MODE_DIMS,
} from './vec-write';
import { configureMemory, type MemoryHost, type ResolvedEmbedder } from './config';

function hostWith(resolved: ResolvedEmbedder): MemoryHost {
  return {
    getAdminUrl: () => 'postgres://u:p@localhost:5432/db',
    getCredentials: async () => ({}),
    resolveEmbedder: async () => resolved,
    buildEmbedderForMode: async () => async () => [],
    schema: 'harness_shared',
  };
}

describe('vec-write helpers', () => {
  it('vecUpsertSql targets the mode vec table with an in-place ON CONFLICT upsert', () => {
    const sql = vecUpsertSql('harness_shared', 'harrier');
    expect(sql).toContain('harness_shared.memory_vec_harrier');
    expect(sql).toContain('ON CONFLICT (memory_id) DO UPDATE SET vector = EXCLUDED.vector');
    expect(sql).toContain('$2::vector');
  });

  it('toVectorLiteral formats a pgvector literal', () => {
    expect(toVectorLiteral([1, 2, 3])).toBe('[1,2,3]');
  });

  it('VEC_TABLE / MODE_DIMS agree on the four shipped modes (harrier is 1024)', () => {
    expect(Object.keys(VEC_TABLE).sort()).toEqual(['gemma', 'harrier', 'local', 'openai']);
    expect(MODE_DIMS.harrier).toBe(1024);
    expect(MODE_DIMS.gemma).toBe(384);
  });
});

describe('embedAndUpsertVector — best-effort guards (never throw)', () => {
  it('returns false when the embedder is disabled (no write attempted)', async () => {
    configureMemory(hostWith({ mode: 'disabled', reason: 'off' }));
    expect(await embedAndUpsertVector('m1', 'text')).toBe(false);
  });

  it('returns false when the embedding is the wrong width (guarded before PG)', async () => {
    // vec length 3 !== 384 (gemma) → rejected before any pg connection opens.
    configureMemory(hostWith({ mode: 'gemma', dims: 384, embed: async () => [0.1, 0.2, 0.3] }));
    expect(await embedAndUpsertVector('m1', 'text')).toBe(false);
  });

  it('returns false (not throw) when the embedder itself throws', async () => {
    configureMemory(
      hostWith({
        mode: 'gemma',
        dims: 384,
        embed: async () => {
          throw new Error('embedder down');
        },
      }),
    );
    expect(await embedAndUpsertVector('m1', 'text')).toBe(false);
  });
});
