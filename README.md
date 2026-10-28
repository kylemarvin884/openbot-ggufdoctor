# ggufdoctor

**在把本地模型接进 agent 之前，先用一条命令确认它能调工具、能停下来、塞得进显存。**

读 GGUF 的 metadata，不加载权重、不跑推理、不联网，给结论和退出码。

```bash
git clone https://github.com/kylemarvin884/openbot-ggufdoctor
cd openbot-ggufdoctor
node src/cli.mjs check Qwen3-8B-Q4_K_M.gguf --ctx 32768 --vram 12
```

Node ≥22，零依赖，不需要 `npm install`，不需要 API key。

## 它回答四个问题

1. 这个文件的 `tokenizer.chat_template` 有 `tools` 分支吗？没有的话，llama.cpp / vLLM / ollama 拿不到放工具定义的地方，`tool_calls` 会一直是空的，模型改成「我要调用 xxx」的叙述。
2. 模板里引用的 `<|im_end|>` 之类的控制 token，真的在词表里吗？不在的话模型停不下来或重复分隔符。
3. `eos_token_id` 在词表范围内吗？
4. `--ctx 32768` 在给定显存上会不会爆？f16 KV cache 需要多少，最大能开多少 ctx？

## 示例输出

下面是对 demo 里故意做坏的 fixture（`demo/.run/no-tools.gguf`，metadata-only，1 KB）的真实输出：

```text
$ node src/cli.mjs check demo/.run/no-tools.gguf --ctx 32768 --vram 2

ggufdoctor 0.1.0
model   qwen3-fixture   1.0 KiB   arch=qwen3  layers=32  ctx=40960  vocab=7
file    GGUF v3, 1 tensor, quant MOSTLY_Q4_K_M, template chatml

verdict FAIL (3 errors)

  ✗ tools-unsupported  Chat template has no tools branch
      Template "default" (chatml, 209 chars) never reads the "tools" variable. The server has nowhere to put tool definitions, so tool_calls stays empty and the model narrates its plan instead.
      fix: Ship the tool-aware template from the model card, or inject one before serving.
  ✗ eos-out-of-range  EOS id is outside the vocabulary
      eos_token_id 99 but the vocabulary has 7 entries (valid 0..6). Some runtimes error out; others never stop.
      fix: Re-convert; this usually means metadata from one model was merged into another.
  ✗ kv-over-budget  KV cache does not fit the VRAM budget
      At --ctx 32768 the KV cache alone is 4.00 GiB (f16) on top of 0.00 GiB of weights, against a 2.00 GiB budget.
      fix: Max context that fits: about 16383.
  · quant  Quantization
      MOSTLY_Q4_K_M (general.file_type=15), 1 tensor, 0.00 GiB.

exit 1
```

对真实模型文件跑，会打印真实的 `general.name`、层数、词表大小和文件体积。

## 为什么需要它

这四类故障都在文件里，不在权重里，而现成工具只打印 metadata，不给结论：

- **量化版/重打包的 GGUF 会静默丢掉 tool 分支。** [ollama#8982](https://github.com/ollama/ollama/issues/8982) 里说得很直接：ollama 库里的模板和 GGUF 自带的不一致，自带那份是支持 tool 的，被换掉的那份不是。
- **转成 GGUF 之后工具调用直接失效。** [vllm#16195](https://github.com/vllm-project/vllm/issues/16195)
- **模板里的控制 token 和词表对不上，模型停不住。** 用户最后是手改 `tokenizer.chat_template` 才修好的：[unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF#1](https://huggingface.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF/discussions/1)
- **同一个症状的官方归因**：`tool_calls` 永远为空 → 模板没有 tool support（[本地模型工具调用排查](https://llmconfigurator.com/en/guides/coding-agents/tool-calling-local-models)）。

## 3 个真实用例

1. **自己发的量化版**：发布前跑一次，确认 `tools` 分支、`role == "tool"` 分支、EOS 都在，再写 release note。
2. **换了显卡 / 想开长上下文**：`--ctx 65536 --vram 16` 直接告诉你 KV cache 要多少、最大能开多少，而不是先 OOM 再猜。
3. **agent 里工具调用一直空**：先排除文件层原因（模板、词表、EOS），再去怀疑模型本身，省掉一轮「换模型试试」。

## 用法

```bash
# 结论 + 退出码
node src/cli.mjs check model.gguf

# 带上你要服务的上下文和显存预算
node src/cli.mjs check model.gguf --ctx 32768 --vram 12

# 机器可读，给 CI 用
node src/cli.mjs check model.gguf --json --strict

# 附带一份 Markdown 报告
node src/cli.mjs check model.gguf --report report.md
```

| 退出码 | 含义 |
| --- | --- |
| 0 | 通过（或只有 warning） |
| 1 | 有 error；`--strict` 下 warning 也算失败 |
| 2 | 读不出 GGUF（不是 GGUF、版本不支持、文件截断） |

CI 里最省事的一条：

```yaml
- run: node path/to/ggufdoctor/src/cli.mjs check model.gguf --ctx 8192 --strict
```

## 检查项

| id | 级别 | 含义 |
| --- | --- | --- |
| `template-missing` | error | 没有 `tokenizer.chat_template` |
| `tools-unsupported` | error | 模板没有 `tools` 分支 → `tool_calls` 会一直是空 |
| `tool-role-unhandled` | warn | 有 tools 分支，但不处理 `role == "tool"` 的返回 |
| `generation-prompt-missing` | warn | 没有 `add_generation_prompt` 分支 |
| `template-token-missing` | warn | 模板里打印的控制 token 不在词表里 |
| `eos-out-of-range` / `eos-missing` | error / warn | EOS id 越界或缺失 |
| `ctx-exceeds-trained` | error | `--ctx` 大于训练上下文 |
| `kv-over-budget` | error | KV cache + 权重大于 `--vram`，并给出能塞下的最大 ctx |
| `arch-missing` / `vocab-missing` | error | 架构或词表缺失，引擎无法加载 |
| `multiple-templates` | info | 文件里有多个模板，提醒引擎默认可能不是你想要的 |

KV cache 按 f16 计算：`层数 × KV head 总数 × (key 维度 + value 维度) × 2 字节`。`head_count_kv` 是数组时按层求和，不取平均。

## 不做什么

- 不加载权重、不跑推理、不测速度、不评量化质量。**本项目没有任何 benchmark 数字，也不会编。**
- 不下载、不修改你的文件，只读。
- 不判断模型「聪明不聪明」。它只回答文件层面的可运行性。

## 验证范围（诚实说明）

- `npm test` 覆盖 32 个用例：GGUF v2/v3 头、各类 metadata 类型、数组、tensor descriptor、截断/坏 magic/坏计数、每条规则的触发与不触发、CLI 的 0/1/2 退出码。
- fixture 由 `demo/gguf-writer.mjs` 按 [GGUF 规范](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) 生成，metadata-only，不含权重。
- **没有在本机对下载的真实大模型做过端到端验证**：构建环境访问不到 huggingface.co。第一次跑真实文件如果报错，请开 issue 附上 `node src/cli.mjs check x.gguf --json` 的输出（不含权重，只有 metadata）。

## demo

```bash
npm run demo
```

生成三个 metadata-only fixture（tool 齐全 / 无 tools 分支 + EOS 越界 / 模板 token 不在词表）并逐个诊断。

## License

MIT
