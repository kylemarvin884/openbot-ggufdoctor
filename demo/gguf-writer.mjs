// Writes spec-shaped GGUF v2/v3 files for demo and tests.
// This is a fixture generator, not a model converter: it writes metadata plus
// tensor descriptors and zero-filled tensor data, which is enough for any tool
// that only reads the header.

const VALUE_IDS = {
  uint8: 0,
  int8: 1,
  uint16: 2,
  int16: 3,
  uint32: 4,
  int32: 5,
  float32: 6,
  bool: 7,
  string: 8,
  array: 9,
  uint64: 10,
  int64: 11,
  float64: 12,
};

function u32(v) {
  const b = Buffer.alloc(4);
  b.writeUInt32LE(v, 0);
  return b;
}

function u64(v) {
  const b = Buffer.alloc(8);
  b.writeBigUInt64LE(BigInt(v), 0);
  return b;
}

function encodeScalar(type, value) {
  switch (type) {
    case 'uint8': return Buffer.from([value]);
    case 'int8': {
      const b = Buffer.alloc(1);
      b.writeInt8(value, 0);
      return b;
    }
    case 'uint16': {
      const b = Buffer.alloc(2);
      b.writeUInt16LE(value, 0);
      return b;
    }
    case 'int16': {
      const b = Buffer.alloc(2);
      b.writeInt16LE(value, 0);
      return b;
    }
    case 'uint32': return u32(value);
    case 'int32': {
      const b = Buffer.alloc(4);
      b.writeInt32LE(value, 0);
      return b;
    }
    case 'float32': {
      const b = Buffer.alloc(4);
      b.writeFloatLE(value, 0);
      return b;
    }
    case 'bool': return Buffer.from([value ? 1 : 0]);
    case 'uint64': return u64(value);
    case 'int64': {
      const b = Buffer.alloc(8);
      b.writeBigInt64LE(BigInt(value), 0);
      return b;
    }
    case 'float64': {
      const b = Buffer.alloc(8);
      b.writeDoubleLE(value, 0);
      return b;
    }
    case 'string': {
      const body = Buffer.from(value, 'utf8');
      return Buffer.concat([u64(body.length), body]);
    }
    default:
      throw new Error(`cannot encode type ${type}`);
  }
}

function encodeValue(entry) {
  if (entry.type === 'array') {
    const parts = [u32(VALUE_IDS[entry.elem]), u64(entry.value.length)];
    for (const v of entry.value) {
      parts.push(encodeScalar(entry.elem, v));
    }
    return Buffer.concat(parts);
  }
  return encodeScalar(entry.type, entry.value);
}

/**
 * @param {{version?:number, metadata?:Record<string,{type:string,value:any,elem?:string}>, tensors?:Array<{name:string,dims:number[],type?:number}>, alignment?:number}} spec
 * @returns {Buffer}
 */
export function buildGguf(spec) {
  const version = spec.version ?? 3;
  const metadata = spec.metadata ?? {};
  const tensors = spec.tensors ?? [];
  const alignment = spec.alignment ?? 32;

  const head = Buffer.concat([
    Buffer.from('GGUF', 'ascii'),
    u32(version),
    u64(tensors.length),
    u64(Object.keys(metadata).length),
  ]);

  const parts = [head];
  for (const [key, entry] of Object.entries(metadata)) {
    const keyBuf = encodeScalar('string', key);
    const typeBuf = u32(VALUE_IDS[entry.type]);
    parts.push(keyBuf, typeBuf, encodeValue(entry));
  }

  let offset = 0;
  for (const tensor of tensors) {
    const nameBuf = encodeScalar('string', tensor.name);
    const dims = tensor.dims ?? [8];
    const dimBuf = Buffer.concat([u32(dims.length), ...dims.map((d) => u64(d))]);
    parts.push(nameBuf, dimBuf, u32(tensor.type ?? 0), u64(offset));
    offset += dims.reduce((a, b) => a * b, 1) * 4;
  }

  let body = Buffer.concat(parts);
  const pad = (alignment - (body.length % alignment)) % alignment;
  body = Buffer.concat([body, Buffer.alloc(pad)]);
  return Buffer.concat([body, Buffer.alloc(Math.min(offset, 64))]);
}

export const CHATML_TOOLS_TEMPLATE = [
  "{%- if tools %}",
  "    {{- '<|im_start|>system\\n' }}",
  '    {{- "You have access to these tools:\\n" }}',
  "    {%- for tool in tools %}",
  '        {{- tool | tojson }}',
  "    {%- endfor %}",
  "    {{- '<|im_end|>\\n' }}",
  '{%- endif %}',
  "{%- for message in messages %}",
  "    {%- if message.role == 'tool' %}",
  "        {{- '<|im_start|>tool\\n' + message.content + '<|im_end|>\\n' }}",
  '    {%- else %}',
  "        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}",
  '    {%- endif %}',
  '{%- endfor %}',
  "{%- if add_generation_prompt %}",
  "    {{- '<|im_start|>assistant\\n' }}",
  '{%- endif %}',
].join('\n');

export const CHATML_NO_TOOLS_TEMPLATE = [
  '{%- for message in messages %}',
  "    {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}",
  '{%- endfor %}',
  "{%- if add_generation_prompt %}",
  "    {{- '<|im_start|>assistant\\n' }}",
  '{%- endif %}',
].join('\n');

export const BASE_VOCAB = [
  '<unk>', '<s>', '</s>', '<|im_start|>', '<|im_end|>', 'hello', 'world',
];

export function baseMetadata({ template = CHATML_TOOLS_TEMPLATE, eos = 2, ctx = 40960, vocab = BASE_VOCAB } = {}) {
  return {
    'general.architecture': { type: 'string', value: 'qwen3' },
    'general.name': { type: 'string', value: 'qwen3-fixture' },
    'general.file_type': { type: 'uint32', value: 15 },
    'general.alignment': { type: 'uint32', value: 32 },
    'qwen3.block_count': { type: 'uint32', value: 32 },
    'qwen3.context_length': { type: 'uint32', value: ctx },
    'qwen3.embedding_length': { type: 'uint32', value: 4096 },
    'qwen3.attention.head_count': { type: 'uint32', value: 32 },
    'qwen3.attention.head_count_kv': { type: 'uint32', value: 8 },
    'tokenizer.ggml.model': { type: 'string', value: 'gpt2' },
    'tokenizer.ggml.tokens': { type: 'array', elem: 'string', value: vocab },
    'tokenizer.ggml.eos_token_id': { type: 'uint32', value: eos },
    'tokenizer.ggml.bos_token_id': { type: 'uint32', value: 1 },
    'tokenizer.chat_template': { type: 'string', value: template },
  };
}
