/** Atomic storage half of managed writes. The fence is deliberately independent
 * of memory_canonical: hard privacy deletion must not let a delayed create revive
 * the row. It retains only scope/ids/hashes, never the deleted body or metadata.
 * Installer ownership stays in the host's existing resource journal. */
import type { Pool, PoolClient } from 'pg';
import type { ManagedMemoryResource, ManagedMemoryWrites, RememberOptions } from './backend';

interface Fence {
  write_key: string;
  scope: string;
  request_hash: string | null;
  fingerprint: string | null;
  canceled: boolean;
}

const digest = (value: string) => `encode(sha256(convert_to((${value})::text, 'UTF8')), 'hex')`;

export class CanonicalManagedWrites implements ManagedMemoryWrites {
  private readonly canonical: string;
  private readonly fences: string;
  private readonly vectors: string;

  constructor(private readonly deps: {
    pool: () => Promise<Pool>;
    schema: string;
    vecTable: string;
    dims: number;
    embed: (text: string) => Promise<number[] | null>;
  }) {
    if (![deps.schema, deps.vecTable].every((name) => /^[a-z_][a-z0-9_]*$/.test(name))) {
      throw new Error('invalid managed memory table identifier');
    }
    this.canonical = `${deps.schema}.memory_canonical`;
    this.fences = `${deps.schema}.memory_managed_writes`;
    this.vectors = `${deps.schema}.${deps.vecTable}`;
  }

  private validate(key: string, scope: string): void {
    if (!/^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i.test(key) || !scope.trim()) {
      throw new Error('managed memory requires a UUID write key and nonempty scope');
    }
  }

  private async transaction<T>(run: (tx: PoolClient) => Promise<T>): Promise<T> {
    const tx = await (await this.deps.pool()).connect();
    try {
      await tx.query('BEGIN');
      const result = await run(tx);
      await tx.query('COMMIT');
      return result;
    } catch (error) {
      await tx.query('ROLLBACK');
      throw error;
    } finally { tx.release(); }
  }

  private async read(tx: Pool | PoolClient, key: string, scope: string): Promise<ManagedMemoryResource[]> {
    const { rows } = await tx.query(`SELECT f.scope, f.fingerprint, c.id,
        (${digest('c.payload')} = f.fingerprint AND c.invalid_at IS NULL AND c.superseded_by IS NULL) AS pristine
      FROM ${this.fences} f LEFT JOIN ${this.canonical} c ON c.id = f.write_key
      WHERE f.write_key = $1`, [key]);
    const row = rows[0];
    if (!row) return [];
    if (row.scope !== scope) throw new Error('managed memory scope mismatch');
    if (!row.fingerprint) return [];
    return [{ id: key, fingerprint: row.fingerprint,
      ...(!row.id ? { disposition: 'absent' as const } : !row.pristine ? { disposition: 'changed' as const } : {}) }];
  }

  async recover(key: string, scope: string): Promise<ManagedMemoryResource[]> {
    this.validate(key, scope);
    return this.read(await this.deps.pool(), key, scope);
  }

  async create(key: string, text: string, opts: RememberOptions): Promise<ManagedMemoryResource[]> {
    this.validate(key, opts.scope);
    if (opts.verbatim !== true) throw new Error('managed memory requires verbatim content');
    if (opts.metadata?.entityType !== undefined) throw new Error('managed writes cannot create entities');
    // Reserved storage fields cannot be overridden by caller metadata.
    const payload = { ...opts.metadata, ...(opts.kind === undefined ? {} : { kind: opts.kind }),
      data: text, user_id: opts.scope, shareable: false };
    const encoded = JSON.stringify(payload);
    const pool = await this.deps.pool();
    // Persist the operation before embedding, which can outlive the caller.
    await pool.query(`INSERT INTO ${this.fences} (write_key, scope, request_hash)
      VALUES ($1, $2, ${digest('$3::jsonb')}) ON CONFLICT (write_key) DO NOTHING`, [key, opts.scope, encoded]);
    const check = async (tx: Pool | PoolClient, locked: boolean): Promise<Fence> => {
      const { rows } = await tx.query(`SELECT *, request_hash = ${digest('$2::jsonb')} AS same_request
        FROM ${this.fences} WHERE write_key = $1 ${locked ? 'FOR UPDATE' : ''}`, [key, encoded]);
      const fence = rows[0];
      if (!fence || fence.scope !== opts.scope) throw new Error('managed memory scope mismatch');
      if (fence.canceled) throw new Error('managed memory write canceled');
      if (!fence.same_request) throw new Error('managed memory write key reused with different content');
      return fence;
    };
    const prior = await check(pool, false);
    if (prior.fingerprint) return this.read(pool, key, opts.scope);
    const vector = await this.deps.embed(opts.embedText ?? text);
    if (!vector || vector.length !== this.deps.dims || !vector.every(Number.isFinite)) {
      throw new Error('managed memory embedding unavailable or incompatible');
    }
    return this.transaction(async (tx) => {
      const fence = await check(tx, true);
      if (!fence.fingerprint) {
        // No UPSERT: collisions and user deletion never authorize overwrites.
        await tx.query(`INSERT INTO ${this.canonical} (id, payload, state)
          VALUES ($1, $2::jsonb, 'archived')`, [key, encoded]);
        await tx.query(`INSERT INTO ${this.vectors} (memory_id, vector, embedded_at)
          VALUES ($1, $2::vector, now())`, [key, `[${vector.join(',')}]`]);
        await tx.query(`UPDATE ${this.fences} SET fingerprint = request_hash WHERE write_key = $1`, [key]);
      }
      return this.read(tx, key, opts.scope);
    });
  }

  async cancel(key: string, scope: string): Promise<void> {
    this.validate(key, scope);
    const { rowCount } = await (await this.deps.pool()).query(`INSERT INTO ${this.fences} (write_key, scope, canceled)
      VALUES ($1, $2, true) ON CONFLICT (write_key) DO UPDATE SET canceled = true
      WHERE ${this.fences}.scope = EXCLUDED.scope`, [key, scope]);
    if (!rowCount) throw new Error('managed memory scope mismatch');
  }

  async removeIfUnchanged(resource: ManagedMemoryResource, scope: string): Promise<'removed' | 'absent' | 'changed'> {
    this.validate(resource.id, scope);
    return this.transaction(async (tx) => {
      const { rows: fences } = await tx.query(`SELECT * FROM ${this.fences} WHERE write_key = $1 FOR UPDATE`, [resource.id]);
      const fence: Fence | undefined = fences[0];
      if (!fence || fence.scope !== scope || fence.fingerprint !== resource.fingerprint) {
        throw new Error('managed memory receipt mismatch');
      }
      // Deletion itself fences too; a caller cannot accidentally omit cancellation.
      await tx.query(`UPDATE ${this.fences} SET canceled = true WHERE write_key = $1`, [resource.id]);
      const { rows } = await tx.query(`SELECT ${digest('payload')} AS fingerprint,
          invalid_at, superseded_by FROM ${this.canonical} WHERE id = $1 FOR UPDATE`, [resource.id]);
      if (!rows.length) return 'absent';
      if (rows[0].fingerprint !== resource.fingerprint || rows[0].invalid_at || rows[0].superseded_by) return 'changed';
      await tx.query(`DELETE FROM ${this.canonical} WHERE id = $1`, [resource.id]);
      return 'removed';
    });
  }
}
