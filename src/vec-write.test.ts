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
  MEMORY_VECTOR_STORAGE_PROFILES,
  memoryStorageAcceptsProfile,
  validateMemoryStorageCompatibility,
  type MemoryVectorStorageProfile,
} from './vec-write';
import { EMBEDDER_DIM_SPECS } from './embedder-dims';
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

  it('independently declared storage bindings accept each exact current profile', () => {
    for (const mode of Object.keys(EMBEDDER_DIM_SPECS) as Array<keyof typeof EMBEDDER_DIM_SPECS>) {
      const storage = MEMORY_VECTOR_STORAGE_PROFILES[mode];
      expect(validateMemoryStorageCompatibility(EMBEDDER_DIM_SPECS[mode], storage), mode).toEqual([]);
      expect(memoryStorageAcceptsProfile(EMBEDDER_DIM_SPECS[mode], storage), mode).toBe(true);
      expect(VEC_TABLE[mode]).toBe(storage.table);
      expect(MODE_DIMS[mode]).toBe(storage.dimensions);
    }
  });

  it('rejects same-width vectors from a different profile identity', () => {
    // Gemma and OpenAI are both 768-dim cosine spaces. That does NOT make them
    // cross-compatible; exact profile identity is the first predicate.
    const problems = validateMemoryStorageCompatibility(
      EMBEDDER_DIM_SPECS.openai,
      MEMORY_VECTOR_STORAGE_PROFILES.gemma,
    );
    expect(problems).toEqual([expect.stringContaining('does not accept profile')]);
    expect(memoryStorageAcceptsProfile(EMBEDDER_DIM_SPECS.openai, MEMORY_VECTOR_STORAGE_PROFILES.gemma)).toBe(false);
  });

  it('rejects metric and index-operator-class skew independently of profile identity', () => {
    const base = MEMORY_VECTOR_STORAGE_PROFILES.local;
    const wrongMetric: MemoryVectorStorageProfile = {
      ...base,
      distanceMetric: 'l2',
    };
    expect(validateMemoryStorageCompatibility(EMBEDDER_DIM_SPECS.local, wrongMetric)).toEqual([
      expect.stringContaining('uses l2'),
      expect.stringContaining('uses vector_cosine_ops'),
    ]);

    const wrongIndex: MemoryVectorStorageProfile = {
      ...base,
      indexOperatorClass: 'vector_l2_ops',
    };
    expect(validateMemoryStorageCompatibility(EMBEDDER_DIM_SPECS.local, wrongIndex)).toEqual([
      expect.stringContaining('cosine requires vector_cosine_ops'),
    ]);
  });

  it('fails closed when a legacy row has no explicit accepted profile identity', () => {
    const unknownLegacy: MemoryVectorStorageProfile = {
      ...MEMORY_VECTOR_STORAGE_PROFILES.local,
      acceptedProfileIds: [],
    };
    expect(validateMemoryStorageCompatibility(EMBEDDER_DIM_SPECS.local, unknownLegacy)).toEqual([
      expect.stringContaining('accepted=(none)'),
    ]);
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
