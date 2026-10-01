/** Pin the Rust-tokenizer normalization boundary without loading model weights.
 * The patch-package postimage check also guards installed bundle drift. */
import { createRequire } from 'node:module';
import { describe, expect, it } from 'vitest';
import { Tokenizer } from '@huggingface/tokenizers';
import { PreTrainedTokenizer } from '@huggingface/transformers';

const require = createRequire(import.meta.url);
const { Tokenizer: CjsTokenizer } = require('@huggingface/tokenizers');
const { PreTrainedTokenizer: CjsSdkTokenizer } = require('@huggingface/transformers');

const json = (normalized = false) => ({
  version: '1.0', truncation: null, padding: null,
  added_tokens: [{ id: 2, content: normalized ? '  ' : '▁▁', single_word: false,
    lstrip: false, rstrip: false, normalized, special: false }],
  normalizer: { type: 'Replace', pattern: { String: ' ' }, content: '▁' },
  pre_tokenizer: { type: 'Metaspace', replacement: '▁', prepend_scheme: 'always', split: true },
  post_processor: null, decoder: null,
  model: { type: 'BPE', vocab: { '▁': 0, a: 1, '▁▁': 2, '<unk>': 3 }, merges: [], unk_token: '<unk>' },
});

const factories = {
  'tokenizers ESM': (normalized: boolean) => (text: string) => new Tokenizer(json(normalized), {}).encode(text).ids,
  'tokenizers CJS': (normalized: boolean) => (text: string) => new CjsTokenizer(json(normalized), {}).encode(text).ids,
  'SDK Node ESM': (normalized: boolean) => (text: string) => new PreTrainedTokenizer(json(normalized), {}).encode(text, { add_special_tokens: false }),
  'SDK Node CJS': (normalized: boolean) => (text: string) => new CjsSdkTokenizer(json(normalized), {}).encode(text, { add_special_tokens: false }),
};

describe.each(Object.entries(factories))('%s normalization contract', (_name, factory) => {
  it('keeps generated whitespace separate from an unnormalized added token', () => {
    expect(factory(false)('  ')).toEqual([0, 0]);
  });
  it('still recognizes the actual literal unnormalized token before normalization', () => {
    expect(factory(false)('▁▁')).toEqual([2]);
  });
  it('still recognizes an explicitly normalized added token', () => {
    expect(factory(true)('  ')).toEqual([2]);
  });
});
