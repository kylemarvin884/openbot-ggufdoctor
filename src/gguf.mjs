// Minimal GGUF metadata reader. Reads header, KV metadata and tensor descriptors.
// Does not touch tensor data, so a multi-GB model is never loaded into memory.
// Format reference: ggml-org/ggml docs/gguf.md (GGUF v2 / v3).

export const VALUE_TYPES = {
  0: 'uint8',
  1: 'int8',
  2: 'uint16',
  3: 'int16',
  4: 'uint32',
  5: 'int32',
  6: 'float32',
  7: 'bool',
  8: 'string',
  9: 'array',
  10: 'uint64',
  11: 'int64',
  12: 'float64',
};

export class GgufError extends Error {
  constructor(message, offset) {
    super(message);
    this.name = 'GgufError';
    this.offset = offset;
  }
}

// Guard against corrupt counts (a truncated file can claim 2^63 tensors).
const MAX_COUNT = 50_000_000;
const MAX_DEPTH = 4;

class Reader {
  constructor(buf, offset = 0) {
    this.buf = buf;
    this.pos = offset;
  }

  need(n) {
    if (this.pos + n > this.buf.length) {
      throw new GgufError(
        `unexpected end of data at byte ${this.pos} (needed ${n} more bytes)`,
        this.pos,
      );
    }
  }

  u8() {
    this.need(1);
    const v = this.buf.readUInt8(this.pos);
    this.pos += 1;
    return v;
  }

  i8() {
    this.need(1);
    const v = this.buf.readInt8(this.pos);
    this.pos += 1;
    return v;
  }

  u16() {
    this.need(2);
    const v = this.buf.readUInt16LE(this.pos);
    this.pos += 2;
    return v;
  }

  i16() {
    this.need(2);
    const v = this.buf.readInt16LE(this.pos);
    this.pos += 2;
    return v;
  }

  u32() {
    this.need(4);
    const v = this.buf.readUInt32LE(this.pos);
    this.pos += 4;
    return v;
  }

  i32() {
    this.need(4);
    const v = this.buf.readInt32LE(this.pos);
    this.pos += 4;
    return v;
  }

  f32() {
    this.need(4);
    const v = this.buf.readFloatLE(this.pos);
    this.pos += 4;
    return v;
  }

  f64() {
    this.need(8);
    const v = this.buf.readDoubleLE(this.pos);
    this.pos += 8;
    return v;
  }

  u64() {
    this.need(8);
    const v = this.buf.readBigUInt64LE(this.pos);
    this.pos += 8;
    if (v > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new GgufError(`64-bit value out of safe range at byte ${this.pos - 8}`, this.pos - 8);
    }
    return Number(v);
  }

  i64() {
    this.need(8);
    const v = this.buf.readBigInt64LE(this.pos);
    this.pos += 8;
    return Number(v);
  }

  string() {
    const len = this.u64();
    if (len > this.buf.length) {
      throw new GgufError(`string length ${len} exceeds file size at byte ${this.pos - 8}`, this.pos - 8);
    }
    this.need(len);
    const s = this.buf.toString('utf8', this.pos, this.pos + len);
    this.pos += len;
    return s;
  }
}

function readValue(reader, type, depth = 0) {
  switch (type) {
    case 0: return reader.u8();
    case 1: return reader.i8();
    case 2: return reader.u16();
    case 3: return reader.i16();
    case 4: return reader.u32();
    case 5: return reader.i32();
    case 6: return reader.f32();
    case 7: return reader.u8() !== 0;
    case 8: return reader.string();
    case 9: {
      if (depth >= MAX_DEPTH) throw new GgufError('array nesting too deep', reader.pos);
      const elemType = reader.u32();
      if (!(elemType in VALUE_TYPES)) {
        throw new GgufError(`unknown array element type ${elemType} at byte ${reader.pos - 4}`, reader.pos - 4);
      }
      if (elemType === 9) throw new GgufError('nested arrays are not valid in GGUF', reader.pos - 4);
      const count = reader.u64();
      if (count > MAX_COUNT) {
        throw new GgufError(`array count ${count} looks corrupt at byte ${reader.pos - 8}`, reader.pos - 8);
      }
      const out = new Array(count);
      for (let i = 0; i < count; i += 1) out[i] = readValue(reader, elemType, depth + 1);
      return out;
    }
    case 10: return reader.u64();
    case 11: return reader.i64();
    case 12: return reader.f64();
    default:
      throw new GgufError(`unknown value type ${type} at byte ${reader.pos - 4}`, reader.pos - 4);
  }
}

/**
 * Parse a GGUF buffer. Only metadata is read.
 * @param {Buffer} buf
 * @returns {{version:number, tensorCount:number, kvCount:number, metadata:Map<string,{type:string,value:any}>, tensors:Array<{name:string,dims:number[],type:number,offset:number,typeName:string}>, alignment:number, headerBytes:number}}
 */
export function parseGguf(buf) {
  if (buf.length < 4) {
    throw new GgufError('file is too small to be a GGUF file', 0);
  }
  const magic = buf.toString('ascii', 0, 4);
  if (magic !== 'GGUF') {
    throw new GgufError(
      `not a GGUF file (magic is ${JSON.stringify(magic)}, expected "GGUF")`,
      0,
    );
  }
  if (buf.length < 24) {
    throw new GgufError('GGUF header is truncated (a v2/v3 header needs at least 24 bytes)', 0);
  }

  const reader = new Reader(buf, 4);
  const version = reader.u32();
  if (version !== 2 && version !== 3) {
    throw new GgufError(
      `unsupported GGUF version ${version} (this tool reads v2 and v3)`,
      4,
    );
  }

  const tensorCount = reader.u64();
  const kvCount = reader.u64();
  if (tensorCount > MAX_COUNT) throw new GgufError(`tensor count ${tensorCount} looks corrupt`, 8);
  if (kvCount > MAX_COUNT) throw new GgufError(`metadata count ${kvCount} looks corrupt`, 8);

  const metadata = new Map();
  for (let i = 0; i < kvCount; i += 1) {
    const key = reader.string();
    const type = reader.u32();
    if (!(type in VALUE_TYPES)) {
      throw new GgufError(`unknown metadata type ${type} for key "${key}"`, reader.pos - 4);
    }
    metadata.set(key, { type: VALUE_TYPES[type], value: readValue(reader, type) });
  }

  const tensors = [];
  for (let i = 0; i < tensorCount; i += 1) {
    const name = reader.string();
    const nDims = reader.u32();
    if (nDims > 8) throw new GgufError(`tensor "${name}" claims ${nDims} dimensions`, reader.pos - 4);
    const dims = [];
    for (let d = 0; d < nDims; d += 1) dims.push(reader.u64());
    const type = reader.u32();
    const offset = reader.u64();
    tensors.push({ name, dims, type, offset, typeName: GGML_TYPES[type] ?? `unknown(${type})` });
  }

  const alignmentEntry = metadata.get('general.alignment');
  const alignment = Number(alignmentEntry?.value ?? 32) || 32;

  return {
    version,
    tensorCount,
    kvCount,
    metadata,
    tensors,
    alignment,
    headerBytes: reader.pos,
  };
}

// ggml tensor type ids that are worth naming in a report. Unmapped ids are printed as-is.
export const GGML_TYPES = {
  0: 'F32',
  1: 'F16',
  2: 'Q4_0',
  3: 'Q4_1',
  6: 'Q5_0',
  7: 'Q5_1',
  8: 'Q8_0',
  9: 'Q8_1',
  10: 'Q2_K',
  11: 'Q3_K',
  12: 'Q4_K',
  13: 'Q5_K',
  14: 'Q6_K',
  15: 'Q8_K',
  16: 'IQ2_XXS',
  17: 'IQ2_XS',
  18: 'IQ3_XXS',
  19: 'IQ1_S',
  20: 'IQ4_NL',
  21: 'IQ3_S',
  22: 'IQ2_S',
  23: 'IQ4_XS',
  24: 'I8',
  25: 'I16',
  26: 'I32',
  27: 'I64',
  28: 'F64',
  29: 'IQ1_M',
  30: 'BF16',
  34: 'TQ1_0',
  35: 'TQ2_0',
};

// general.file_type values from llama.cpp's llama_ftype enum (partial but stable subset).
export const FILE_TYPES = {
  0: 'ALL_F32',
  1: 'MOSTLY_F16',
  2: 'MOSTLY_Q4_0',
  3: 'MOSTLY_Q4_1',
  7: 'MOSTLY_Q8_0',
  8: 'MOSTLY_Q5_0',
  9: 'MOSTLY_Q5_1',
  10: 'MOSTLY_Q2_K',
  11: 'MOSTLY_Q3_K_S',
  12: 'MOSTLY_Q3_K_M',
  13: 'MOSTLY_Q3_K_L',
  14: 'MOSTLY_Q4_K_S',
  15: 'MOSTLY_Q4_K_M',
  16: 'MOSTLY_Q5_K_S',
  17: 'MOSTLY_Q5_K_M',
  18: 'MOSTLY_Q6_K',
  19: 'MOSTLY_IQ2_XXS',
  20: 'MOSTLY_IQ2_XS',
  21: 'MOSTLY_Q2_K_S',
  22: 'MOSTLY_IQ3_XS',
  23: 'MOSTLY_IQ3_XXS',
  24: 'MOSTLY_IQ1_S',
  25: 'MOSTLY_IQ4_NL',
  26: 'MOSTLY_IQ3_S',
  27: 'MOSTLY_IQ2_S',
  28: 'MOSTLY_IQ4_XS',
  29: 'MOSTLY_IQ1_M',
  30: 'MOSTLY_BF16',
  32: 'MOSTLY_TQ1_0',
  33: 'MOSTLY_TQ2_0',
};
