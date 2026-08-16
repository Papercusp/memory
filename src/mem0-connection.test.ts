/**
 * Unit tests for the mem0 pgvector connection resolver (mem0-connection.ts). mem0ai's PGVector
 * provider wants DISCRETE fields, so this parses the canonical admin URL into pieces — the
 * bug-prone bits being default port, URL-decoded credentials, and the empty-path dbname fallback.
 * The host `getAdminUrl` seam is faked via `configureMemory`, so no real PG / discovery is needed.
 */
import { beforeEach, describe, expect, it } from 'vitest';
import { configureMemory, type MemoryHost } from './config';
import { connectionString, pgFields, pgClientFields } from './mem0-connection';

/** Configure the memory host with just the fields mem0-connection reads (getAdminUrl + defaultDbName). */
function withUrl(url: string | (() => string | Promise<string>), defaultDbName?: string): void {
  const getAdminUrl = typeof url === 'function' ? url : () => url;
  configureMemory({ getAdminUrl, ...(defaultDbName !== undefined ? { defaultDbName } : {}) } as MemoryHost);
}

describe('mem0-connection', () => {
  beforeEach(() => withUrl('postgres://u:p@h:5432/db'));

  describe('connectionString', () => {
    it('returns the host admin URL (sync getAdminUrl)', async () => {
      withUrl('postgres://u:p@host:5432/papercusp');
      expect(await connectionString()).toBe('postgres://u:p@host:5432/papercusp');
    });

    it('awaits an async getAdminUrl', async () => {
      withUrl(async () => 'postgres://u:p@host:5432/async-db');
      expect(await connectionString()).toBe('postgres://u:p@host:5432/async-db');
    });
  });

  describe('pgFields', () => {
    it('parses host / port / user / password / dbname from a full URL', async () => {
      withUrl('postgres://alice:secret@db.host:6543/papercusp');
      expect(await pgFields()).toEqual({
        host: 'db.host',
        port: 6543,
        user: 'alice',
        password: 'secret',
        dbname: 'papercusp',
      });
    });

    it('defaults the port to 5432 when the URL omits it', async () => {
      withUrl('postgres://u:p@h/db');
      expect((await pgFields()).port).toBe(5432);
    });

    it('URL-decodes percent-encoded credentials', async () => {
      withUrl('postgres://us%40er:p%3As%2Fs@h/db');
      const f = await pgFields();
      expect(f.user).toBe('us@er');
      expect(f.password).toBe('p:s/s');
    });

    it('falls back to the host defaultDbName, then "postgres", on an empty path', async () => {
      withUrl('postgres://u:p@h:5432/', 'papercusp');
      expect((await pgFields()).dbname).toBe('papercusp');
      withUrl('postgres://u:p@h:5432/'); // no defaultDbName configured
      expect((await pgFields()).dbname).toBe('postgres');
    });
  });

  describe('pgClientFields', () => {
    it('renames dbname → database for node-postgres `new Client()`', async () => {
      withUrl('postgres://u:p@h:6543/mydb');
      expect(await pgClientFields()).toEqual({
        host: 'h',
        port: 6543,
        user: 'u',
        password: 'p',
        database: 'mydb',
      });
    });
  });
});
