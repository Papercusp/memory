/** Pin Rust token IDs, normalization and Metaspace boundaries without weights. */
import { describe, expect, it } from 'vitest';
import { Tokenizer } from 'tokenizers';

const json = (normalized = false) => ({
  version: '1.0', truncation: null, padding: null,
  added_tokens: [{ id: 2, content: '▁▁', single_word: false,
    lstrip: false, rstrip: false, normalized, special: false }],
  normalizer: { type: 'Replace', pattern: { String: ' ' }, content: '▁' },
  pre_tokenizer: { type: 'Metaspace', replacement: '▁', prepend_scheme: 'always', split: true },
  post_processor: null, decoder: null,
  model: { type: 'BPE', vocab: { '▁': 0, a: 1, '▁▁': 2, '<unk>': 3 }, merges: [['▁', '▁']], unk_token: '<unk>' },
});

const encode = async (normalized: boolean, text: string) =>
  (await Tokenizer.fromString(JSON.stringify(json(normalized))).encode(text)).getIds();

describe('candidate Rust tokenizer contract', () => {
  it('honors Metaspace split:true even with a learned whitespace BPE merge', async () => {
    expect(await encode(false, '  ')).toEqual([0, 0]);
  });
  it('still recognizes the actual literal unnormalized token before normalization', async () => {
    expect(await encode(false, '▁▁')).toEqual([2]);
  });
  it('still recognizes an explicitly normalized added token', async () => {
    expect(await encode(true, '  ')).toEqual([2]);
  });
});
