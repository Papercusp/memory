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

  it('VEC_TABLE / MODE_DIMS agree on the four shipped modes, and the widths are NOT uniform', () => {
    expect(Object.keys(VEC_TABLE).sort()).toEqual(['gemma', 'harrier', 'local', 'openai']);
    // Every mode is asserted explicitly. These are STORAGE facts — each must match
    // the width its `memory_vec_<mode>` column was created at, so changing one here
    // without its migration is meant to red THIS test (vec-write.ts L42-64).
    //
    // gemma + openai moved 384 -> 768 with migration 727 (D-005): gemma at
    // EmbeddingGemma-300m's native width, openai alongside it to stay prose-eligible.
    // local (bge-small-en-v1.5) is natively 384 with no MRL and CANNOT emit 768;
    // harrier is native 1024. The non-uniformity is the point — a blanket
    // "everything is 384" is what this test exists to catch.
    expect(MODE_DIMS.gemma).toBe(768);
    expect(MODE_DIMS.openai).toBe(768);
    expect(MODE_DIMS.local).toBe(384);
    expect(MODE_DIMS.harrier).toBe(1024);
  });
});

describe('embedAndUpsertVector — best-effort guards (never throw)', () => {
  it('returns false when the embedder is disabled (no write attempted)', async () => {
    configureMemory(hostWith({ mode: 'disabled', reason: 'off' }));
    expect(await embedAndUpsertVector('m1', 'text')).toBe(false);
  });

  it('returns false when the embedding is the wrong width (guarded before PG)', async () => {
    // vec length 3 !== 768 (gemma) → rejected before any pg connection opens.
    configureMemory(hostWith({ mode: 'gemma', dims: 768, embed: async () => [0.1, 0.2, 0.3] }));
    expect(await embedAndUpsertVector('m1', 'text')).toBe(false);
  });

  it('returns false (not throw) when the embedder itself throws', async () => {
    configureMemory(
      hostWith({
        mode: 'gemma',
        dims: 768,
        embed: async () => {
          throw new Error('embedder down');
        },
      }),
    );
    expect(await embedAndUpsertVector('m1', 'text')).toBe(false);
  });
});
