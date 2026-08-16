/**
 * Deterministic synthetic-corpus generator for the scale tier
 * (memory-backend-benchmark D-004): quality + latency at 111 → 1k → 10k.
 *
 * The synthetic entries are DISTRACTORS — plausible dev-box memory facts
 * layered around the real corpus so the frozen gold set can be replayed
 * against a noisy store and the degradation curve measured. Determinism
 * matters (same seed → byte-identical corpus → comparable runs), so the
 * generator uses a seeded mulberry32 PRNG, never Math.random.
 */

import type { CorpusEntry } from './types';

/** mulberry32 — tiny deterministic PRNG (public-domain construction). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const SUBSYSTEMS = [
  'scheduler', 'ingest pipeline', 'billing worker', 'auth proxy', 'cache layer',
  'notifier', 'export service', 'sync daemon', 'media transcoder', 'search indexer',
  'queue consumer', 'webhook relay', 'session store', 'rate limiter', 'metrics shipper',
];
const COMPONENTS = [
  'config loader', 'connection pool', 'retry loop', 'migration runner', 'health probe',
  'feature gate', 'token refresher', 'snapshot writer', 'lock manager', 'event bus',
];
const FAILURES = [
  'hangs on shutdown when the drain timer races the socket close',
  'double-fires when the cron tick lands inside a leap second window',
  'silently drops rows when the batch exceeds the wire frame size',
  'leaks file descriptors under repeated TLS renegotiation',
  'returns stale reads when the replica lag passes the lease TTL',
  'corrupts the cursor file when killed between fsync and rename',
  'deadlocks when two writers rotate the log in the same tick',
  'misroutes traffic when the DNS cache outlives the pod IP',
  'truncates unicode keys at the legacy 191-byte index boundary',
  'starves the low-priority queue when backpressure never clears',
];
const FIXES = [
  'pin the worker count to the cgroup CPU quota, not os.cpus()',
  'wrap the handle in a finally-dispose block and assert at exit',
  'bump the lease renewal to half the TTL and jitter it',
  'switch the probe to the loopback admin port',
  'serialize rotation behind the advisory lock',
  'flush before rename and tolerate the duplicate on replay',
  'route through the connection-string resolver instead of env',
  'gate the path behind the feature flag until the index lands',
  'use the monotonic clock for the deadline math',
  'batch by byte budget instead of row count',
];
const TOOLS = [
  'sysctl-tuner', 'pgbouncer', 'redis-sentinel', 'haproxy', 'fluentbit',
  'vault-agent', 'cert-rotator', 'minio-gateway', 'etcd-defrag', 'kafka-mirror',
];

function pick<T>(rng: () => number, xs: readonly T[]): T {
  return xs[Math.floor(rng() * xs.length)];
}

/** Generate `count` deterministic distractor entries for a seed. */
export function generateSyntheticCorpus(count: number, seed = 1337): CorpusEntry[] {
  const rng = mulberry32(seed);
  const out: CorpusEntry[] = [];
  for (let i = 0; i < count; i++) {
    const subsystem = pick(rng, SUBSYSTEMS);
    const component = pick(rng, COMPONENTS);
    const failure = pick(rng, FAILURES);
    const fix = pick(rng, FIXES);
    const tool = pick(rng, TOOLS);
    const errCode = `E${(1000 + Math.floor(rng() * 9000)).toString()}`;
    const port = 1024 + Math.floor(rng() * 60000);
    const key = `syn_${seed}_${i}`;
    const text =
      `The ${subsystem}'s ${component} ${failure}. ` +
      `Fix: ${fix}. Diagnosed via ${tool} on :${port}; the failing call returns ${errCode}. ` +
      `(synthetic distractor ${key})`;
    out.push({
      key,
      text,
      kind: 'project',
      description: `${subsystem} ${component} — ${failure.slice(0, 60)}`,
      metadata: { synthetic: true },
    });
  }
  return out;
}
