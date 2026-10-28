import test from 'node:test';
import assert from 'node:assert/strict';
import { parseGguf, GgufError, FILE_TYPES } from '../src/gguf.mjs';
import { buildGguf } from '../demo/gguf-writer.mjs';

test('parses header, metadata types and tensor descriptors', () => {
  const buf = buildGguf({
    metadata: {
      'general.architecture': { type: 'string', value: 'qwen3' },
      'general.alignment': { type: 'uint32', value: 32 },
      'qwen3.block_count': { type: 'uint32', value: 28 },
      'qwen3.context_length': { type: 'uint64', value: 131072 },
      'qwen3.rope.freq_base': { type: 'float32', value: 1000000 },
      'tokenizer.ggml.add_bos_token': { type: 'bool', value: true },
      'tokenizer.ggml.token_type': { type: 'array', elem: 'int32', value: [1, 2, 3, 4] },
      'tokenizer.ggml.tokens': { type: 'array', elem: 'string', value: ['<s>', '</s>', 'hi'] },
      'some.negative': { type: 'int32', value: -7 },
    },
    tensors: [
      { name: 'token_embd.weight', dims: [4096, 150000], type: 12 },
      { name: 'output_norm.weight', dims: [4096], type: 0 },
    ],
  });

  const model = parseGguf(buf);
  assert.equal(model.version, 3);
  assert.equal(model.tensorCount, 2);
  assert.equal(model.kvCount, 9);
  assert.equal(model.metadata.get('general.architecture').value, 'qwen3');
  assert.equal(model.metadata.get('qwen3.context_length').value, 131072);
  assert.equal(model.metadata.get('qwen3.rope.freq_base').value, 1000000);
  assert.equal(model.metadata.get('tokenizer.ggml.add_bos_token').value, true);
  assert.deepEqual(model.metadata.get('tokenizer.ggml.token_type').value, [1, 2, 3, 4]);
  assert.deepEqual(model.metadata.get('tokenizer.ggml.tokens').value, ['<s>', '</s>', 'hi']);
  assert.equal(model.metadata.get('some.negative').value, -7);
  assert.equal(model.alignment, 32);

  assert.equal(model.tensors[0].name, 'token_embd.weight');
  assert.deepEqual(model.tensors[0].dims, [4096, 150000]);
  assert.equal(model.tensors[0].typeName, 'Q4_K');
  assert.equal(model.tensors[1].typeName, 'F32');
  // descriptor bytes end before the (aligned) tensor data section
  assert.ok(model.headerBytes > 24);
  assert.ok(model.headerBytes < buf.length);
});

test('file type names are mapped, unknown ids stay honest', () => {
  assert.equal(FILE_TYPES[15], 'MOSTLY_Q4_K_M');
  assert.equal(FILE_TYPES[30], 'MOSTLY_BF16');
  assert.equal(FILE_TYPES[999], undefined);
});

test('rejects a file without the GGUF magic', () => {
  const buf = Buffer.from('PK\u0003\u0004not a model at all, just a zip', 'latin1');
  assert.throws(() => parseGguf(buf), (err) => {
    assert.ok(err instanceof GgufError);
    assert.match(err.message, /not a GGUF file/);
    return true;
  });
});

test('rejects a truncated header instead of guessing', () => {
  const buf = buildGguf({
    metadata: { 'general.architecture': { type: 'string', value: 'llama' } },
  });
  const cut = buf.subarray(0, 30);
  assert.throws(() => parseGguf(cut), GgufError);
});

test('rejects unsupported GGUF versions', () => {
  const buf = buildGguf({ version: 1, metadata: {} });
  assert.throws(() => parseGguf(buf), /unsupported GGUF version 1/);
});

test('rejects a corrupt metadata count', () => {
  const buf = buildGguf({ metadata: {} });
  // bump kv_count to an absurd value: bytes 16..24 hold tensor_count/kv_count
  buf.writeBigUInt64LE(10_000_000_000n, 16);
  assert.throws(() => parseGguf(buf), /looks corrupt/);
});

test('parses a file with zero tensors and zero metadata', () => {
  const model = parseGguf(buildGguf({}));
  assert.equal(model.tensorCount, 0);
  assert.equal(model.metadata.size, 0);
});
