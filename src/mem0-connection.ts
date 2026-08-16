/**
 * Connection-string resolver for mem0's pgvector backend.
 *
 * mem0ai/oss's PGVector provider takes DISCRETE fields (user, password,
 * host, port, dbname) — NOT a connectionString. So we parse the canonical
 * URL into pieces here.
 *
 * The URL itself comes from the injected `getAdminUrl` host seam (the
 * operator wires `@/lib/embedded-pg-discovery`, which already resolves
 * env → discovery-file → native-fallback). Part of P-021.
 */

import { memoryHost } from './config';

export interface Mem0PgConnection {
  host: string;
  port: number;
  user: string;
  password: string;
  dbname: string;
}

export async function connectionString(): Promise<string> {
  return memoryHost().getAdminUrl();
}

export async function pgFields(): Promise<Mem0PgConnection> {
  const url = await connectionString();
  const u = new URL(url);
  return {
    host: u.hostname,
    port: u.port ? Number(u.port) : 5432,
    user: decodeURIComponent(u.username),
    password: decodeURIComponent(u.password),
    dbname: u.pathname.replace(/^\//, '') || memoryHost().defaultDbName || 'postgres',
  };
}

/**
 * Same fields shaped for node-postgres' `new Client()` — which keys
 * on `database` instead of mem0's `dbname`. Use this in any route or
 * worker that opens its own pg.Client; use pgFields() only when
 * passing to mem0ai.
 */
export async function pgClientFields(): Promise<{
  host: string;
  port: number;
  user: string;
  password: string;
  database: string;
}> {
  const f = await pgFields();
  return {
    host: f.host,
    port: f.port,
    user: f.user,
    password: f.password,
    database: f.dbname,
  };
}
