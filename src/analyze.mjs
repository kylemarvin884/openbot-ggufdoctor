// Deterministic readiness checks for a GGUF file used as an agent's model.
// No model calls, no network. Every finding points at metadata that was actually read.

import { FILE_TYPES } from './gguf.mjs';

const GIB = 1024 ** 3;

function plural(n, word) {
  return `${n} ${word}${n === 1 ? '' : 's'}`;
}

function formatGiB(bytes) {
  if (typeof bytes !== 'number') return 'unknown size';
  return `${(bytes / GIB).toFixed(2)} GiB`;
}

function num(metadata, key) {
  const entry = metadata.get(key);
  if (!entry) return null;
  const v = entry.value;
  return typeof v === 'number' ? v : null;
}

function str(metadata, key) {
  const entry = metadata.get(key);
  if (!entry) return null;
  return typeof entry.value === 'string' ? entry.value : null;
}

export function extractTemplates(metadata) {
  const out = [];
  const raw = metadata.get('tokenizer.chat_template')?.value;
  if (typeof raw === 'string' && raw.trim() !== '') out.push({ name: 'default', text: raw });
  if (Array.isArray(raw)) {
    raw.forEach((entry, i) => {
      if (typeof entry === 'string' && entry.trim() !== '') {
        out.push({ name: `chat_template[${i}]`, text: entry });
      } else if (entry && typeof entry === 'object') {
        const text = entry.template ?? entry.value;
        if (typeof text === 'string' && text.trim() !== '') {
          out.push({ name: entry.name ?? `chat_template[${i}]`, text });
        }
      }
    });
  }
  const named = metadata.get('tokenizer.chat_templates')?.value;
  if (Array.isArray(named)) {
    named.forEach((entry, i) => {
      if (entry && typeof entry === 'object') {
        const text = entry.template ?? entry.value;
        if (typeof text === 'string' && text.trim() !== '') {
          out.push({ name: entry.name ?? `chat_templates[${i}]`, text });
        }
      }
    });
  }
  return out;
}

export function inspectTemplate(text) {
  const hasToolsBranch = /\{%-?\s*if\s+tools\b/.test(text) || /\{%-?\s*for\s+tool\s+in\s+tools\b/.test(text);
  const emitsToolCalls = /\btool_calls\b/.test(text);
  const handlesToolRole = /role\s*\]?\s*==\s*['"]tool['"]/.test(text) || /==\s*['"]tool['"]\s*\]?\s*%\}/.test(text);
  const hasGenerationPrompt = /\badd_generation_prompt\b/.test(text);
  const raisesException = /\braise_exception\b/.test(text);

  let family = 'unknown';
  if (/<\|im_start\|>/.test(text)) family = 'chatml';
  else if (/<\|start_header_id\|>/.test(text)) family = 'llama3';
  else if (/<start_of_turn>/.test(text)) family = 'gemma';
  else if (/\[INST\]/.test(text)) family = 'llama2';
  else if (/<\|turn\|>/.test(text)) family = 'deepseek';

  const tokens = new Set();
  for (const m of text.matchAll(/<\|[A-Za-z0-9_.\-]{1,40}\|>/g)) tokens.add(m[0]);
  for (const m of text.matchAll(/\[\/?INST\]/g)) tokens.add(m[0]);

  return {
    hasToolsBranch,
    emitsToolCalls,
    handlesToolRole,
    hasGenerationPrompt,
    raisesException,
    family,
    controlTokens: [...tokens],
    length: text.length,
  };
}

function headCounts(value, layers) {
  if (typeof value === 'number') return { perLayer: Array(layers).fill(value), max: value };
  if (Array.isArray(value) && value.every((v) => typeof v === 'number')) {
    return { perLayer: value.slice(), max: Math.max(...value) };
  }
  return null;
}

export function estimateCapacity(metadata, { ctx, vramGb, fileSizeBytes } = {}) {
  const arch = str(metadata, 'general.architecture');
  if (!arch) return null;
  const layers = num(metadata, `${arch}.block_count`);
  const headCount = num(metadata, `${arch}.attention.head_count`);
  const embed = num(metadata, `${arch}.embedding_length`);
  const kvRaw = metadata.get(`${arch}.attention.head_count_kv`)?.value ?? headCount;
  if (!layers || !headCount || !embed || layers <= 0) return null;
  const kv = headCounts(kvRaw ?? headCount, layers);
  if (!kv) return null;

  const headDim = embed / headCount;
  const keyLen = num(metadata, `${arch}.attention.key_length`) ?? headDim;
  const valueLen = num(metadata, `${arch}.attention.value_length`) ?? headDim;
  if (!Number.isFinite(keyLen) || !Number.isFinite(valueLen)) return null;

  // f16 KV cache: 2 bytes per element, K and V per layer.
  const kvHeadsTotal = kv.perLayer.reduce((a, b) => a + b, 0);
  const bytesPerToken = kvHeadsTotal * (keyLen + valueLen) * 2;
  const trainedCtx = num(metadata, `${arch}.context_length`);
  const budgetBytes = typeof vramGb === 'number' ? vramGb * GIB : null;
  const weightsBytes = typeof fileSizeBytes === 'number' ? fileSizeBytes : null;

  let maxCtx = null;
  if (budgetBytes !== null && weightsBytes !== null) {
    const free = budgetBytes - weightsBytes;
    maxCtx = free > 0 ? Math.floor(free / bytesPerToken) : 0;
  }

  return {
    layers,
    headCount,
    kvHeadsTotal,
    keyLen,
    valueLen,
    bytesPerToken,
    trainedCtx,
    budgetBytes,
    weightsBytes,
    kvBytesAtCtx: typeof ctx === 'number' ? bytesPerToken * ctx : null,
    maxCtx,
  };
}

/**
 * @param {{metadata:Map, tensors:Array, version:number, headerBytes:number}} model
 * @param {{ctx?:number, vramGb?:number, fileSizeBytes?:number, strict?:boolean}} opts
 */
export function analyze(model, opts = {}) {
  const { metadata } = model;
  const findings = [];
  const add = (id, severity, title, detail, fix) => findings.push({ id, severity, title, detail, fix });

  const arch = str(metadata, 'general.architecture');
  const name = str(metadata, 'general.name') ?? '(unnamed)';
  const fileTypeId = num(metadata, 'general.file_type');
  const quant = fileTypeId === null ? 'unknown' : FILE_TYPES[fileTypeId] ?? `file_type ${fileTypeId}`;

  const tokensRaw = metadata.get('tokenizer.ggml.tokens')?.value;
  const vocab = Array.isArray(tokensRaw) ? tokensRaw : null;
  const vocabSize = vocab ? vocab.length : null;

  // --- structure -----------------------------------------------------------
  if (!arch) {
    add('arch-missing', 'error', 'No general.architecture',
      'The file has no architecture key, so no inference engine can load it.',
      'Re-convert the model from source weights.');
  }
  if (!vocab || vocabSize === 0) {
    add('vocab-missing', 'error', 'No tokenizer vocabulary',
      'tokenizer.ggml.tokens is absent or empty. The model cannot tokenize input.',
      'Re-convert with a tokenizer attached (convert_hf_to_gguf.py needs tokenizer.json).');
  }
  if (!metadata.has('tokenizer.ggml.model')) {
    add('tokenizer-model-missing', 'warn', 'Tokenizer type missing',
      'tokenizer.ggml.model is not set; some runtimes will guess and guess wrong.',
      'Keep the converter default instead of hand-editing metadata.');
  }

  // --- tool calling --------------------------------------------------------
  const templates = extractTemplates(metadata);
  let templateInfo = null;
  if (templates.length === 0) {
    add('template-missing', 'error', 'No chat template',
      'tokenizer.chat_template is absent. llama.cpp / ollama will fall back to a format the model was not trained on, and no engine can build tool-call turns.',
      'Re-convert the model, or inject the template the model card ships with.');
  } else {
    const inspected = templates.map((t) => ({ ...t, ...inspectTemplate(t.text) }));
    const withTools = inspected.filter((t) => t.hasToolsBranch);
    templateInfo = {
      count: inspected.length,
      family: inspected[0].family,
      length: inspected[0].length,
      hasToolsBranch: withTools.length > 0,
      emitsToolCalls: inspected.some((t) => t.emitsToolCalls),
      handlesToolRole: inspected.some((t) => t.handlesToolRole),
      hasGenerationPrompt: inspected.some((t) => t.hasGenerationPrompt),
      raisesException: inspected.some((t) => t.raisesException),
      controlTokens: [...new Set(inspected.flatMap((t) => t.controlTokens))],
      names: inspected.map((t) => t.name),
    };

    if (!templateInfo.hasToolsBranch) {
      add('tools-unsupported', 'error', 'Chat template has no tools branch',
        `Template "${inspected[0].name}" (${inspected[0].family}, ${inspected[0].length} chars) never reads the "tools" variable. `
        + 'The server has nowhere to put tool definitions, so tool_calls stays empty and the model narrates its plan instead.',
        'Ship the tool-aware template from the model card, or inject one before serving.');
    } else if (!templateInfo.handlesToolRole) {
      add('tool-role-unhandled', 'warn', 'Template defines tools but not tool results',
        'A tools branch exists, but no branch keys on role == "tool". Tool output will be rendered as a user message or dropped, which usually ends in a repeated call.',
        'Check the template renders role == "tool" in the same format the model was trained on.');
    }
    if (!templateInfo.hasGenerationPrompt) {
      add('generation-prompt-missing', 'warn', 'No add_generation_prompt branch',
        'The template never keys on add_generation_prompt, so the assistant turn opener is unreliable across engines.',
        'Take the template from the model card; this branch is normally present.');
    }
    if (templateInfo.raisesException) {
      add('template-raise-exception', 'info', 'Template raises exceptions',
        'The template calls raise_exception for some inputs. llama.cpp supports this; other engines may fail to render instead of erroring clearly.',
        null);
    }
    if (templateInfo.count > 1) {
      add('multiple-templates', 'info', `${templateInfo.count} chat templates in file`,
        `Named templates: ${templateInfo.names.join(', ')}. Pick one explicitly when serving, or the engine default decides for you.`,
        null);
    }

    // Control tokens the template prints must exist in the vocabulary.
    if (vocab && templateInfo.controlTokens.length > 0) {
      const vocabSet = new Set(vocab);
      const missing = templateInfo.controlTokens.filter((t) => !vocabSet.has(t));
      if (missing.length > 0) {
        add('template-token-missing', 'warn', 'Template uses tokens the vocab does not contain',
          `Not in tokenizer.ggml.tokens: ${missing.slice(0, 5).join(', ')}${missing.length > 5 ? ` (+${missing.length - 5} more)` : ''}. `
          + 'The model sees raw text where a control token was intended, so it often fails to stop or repeats the delimiter.',
          'Fix the template or re-convert so template and vocabulary come from the same source.');
      }
    }
  }

  // --- tokenizer consistency ----------------------------------------------
  const eosRaw = metadata.get('tokenizer.ggml.eos_token_id')?.value;
  const bosRaw = metadata.get('tokenizer.ggml.bos_token_id')?.value;
  const eosIds = Array.isArray(eosRaw) ? eosRaw : eosRaw === undefined ? [] : [eosRaw];
  if (eosIds.length === 0) {
    add('eos-missing', 'warn', 'No EOS token id',
      'tokenizer.ggml.eos_token_id is absent; generation may run to the context limit on every turn.',
      'Set the EOS id from the source config (eos_token_id).');
  } else if (vocabSize !== null) {
    const bad = eosIds.filter((id) => !Number.isInteger(id) || id < 0 || id >= vocabSize);
    if (bad.length > 0) {
      add('eos-out-of-range', 'error', 'EOS id is outside the vocabulary',
        `eos_token_id ${bad.join(', ')} but the vocabulary has ${vocabSize} entries (valid 0..${vocabSize - 1}). `
        + 'Some runtimes error out; others never stop.',
        'Re-convert; this usually means metadata from one model was merged into another.');
    }
  }
  if (bosRaw !== undefined && vocabSize !== null && !Array.isArray(bosRaw)) {
    if (!Number.isInteger(bosRaw) || bosRaw < 0 || bosRaw >= vocabSize) {
      add('bos-out-of-range', 'warn', 'BOS id is outside the vocabulary',
        `bos_token_id ${bosRaw}, vocabulary size ${vocabSize}.`,
        'Check the source config before serving.');
    }
  }

  // --- capacity ------------------------------------------------------------
  const trainedCtx = arch ? num(metadata, `${arch}.context_length`) : null;
  if (trainedCtx === null) {
    add('ctx-missing', 'warn', 'No trained context length',
      'general.architecture + .context_length is missing, so engines fall back to a small default (often 2048/4096) and silently truncate.',
      'Set <arch>.context_length from the source config (max_position_embeddings).');
  } else if (trainedCtx <= 0) {
    add('ctx-invalid', 'error', 'Invalid trained context length', `${arch}.context_length is ${trainedCtx}.`, null);
  }

  const capacity = estimateCapacity(metadata, {
    ctx: opts.ctx,
    vramGb: opts.vramGb,
    fileSizeBytes: opts.fileSizeBytes,
  });

  if (typeof opts.ctx === 'number' && trainedCtx !== null && opts.ctx > trainedCtx) {
    add('ctx-exceeds-trained', 'error', 'Requested context exceeds what the model was trained for',
      `--ctx ${opts.ctx} > ${trainedCtx}. Output degrades sharply past the trained window, and many runtimes just clamp silently.`,
      `Serve with --ctx ${trainedCtx} or lower.`);
  }

  if (capacity && typeof opts.ctx === 'number' && capacity.kvBytesAtCtx) {
    const kvGb = capacity.kvBytesAtCtx / GIB;
    if (capacity.budgetBytes !== null && capacity.weightsBytes !== null) {
      const total = capacity.kvBytesAtCtx + capacity.weightsBytes;
      if (total > capacity.budgetBytes) {
        const suggested = capacity.maxCtx ?? 0;
        add('kv-over-budget', 'error', 'KV cache does not fit the VRAM budget',
          `At --ctx ${opts.ctx} the KV cache alone is ${kvGb.toFixed(2)} GiB (f16) on top of ${(capacity.weightsBytes / GIB).toFixed(2)} GiB of weights, `
          + `against a ${(capacity.budgetBytes / GIB).toFixed(2)} GiB budget.`,
          suggested > 0 ? `Max context that fits: about ${suggested}.` : 'Quantize the KV cache or use a smaller model.');
      } else {
        add('kv-fits', 'info', 'KV cache fits the VRAM budget',
          `${kvGb.toFixed(2)} GiB KV at --ctx ${opts.ctx} + ${(capacity.weightsBytes / GIB).toFixed(2)} GiB weights `
          + `<= ${(capacity.budgetBytes / GIB).toFixed(2)} GiB. Headroom allows up to about ${capacity.maxCtx} ctx.`,
          null);
      }
    } else {
      add('kv-estimate', 'info', 'KV cache estimate',
        `${kvGb.toFixed(2)} GiB at --ctx ${opts.ctx} (f16, ${capacity.layers} layers, ${capacity.kvHeadsTotal} KV heads). `
        + 'Pass --vram <GiB> to check it against a budget.',
        null);
    }
  }

  // --- informational -------------------------------------------------------
  add('quant', 'info', 'Quantization',
    `${quant} (general.file_type=${fileTypeId ?? 'unset'}), ${plural(model.tensors.length, 'tensor')}, ${formatGiB(opts.fileSizeBytes)}.`,
    null);

  const counts = { error: 0, warn: 0, info: 0 };
  for (const f of findings) counts[f.severity] += 1;
  const verdict = counts.error > 0 ? 'fail' : counts.warn > 0 ? 'warn' : 'pass';

  return {
    model: {
      name,
      arch,
      quant,
      version: model.version,
      tensorCount: model.tensors.length,
      fileSizeBytes: opts.fileSizeBytes ?? null,
      vocabSize,
      trainedCtx,
      eosIds,
    },
    template: templateInfo,
    capacity,
    findings,
    counts,
    verdict,
  };
}
