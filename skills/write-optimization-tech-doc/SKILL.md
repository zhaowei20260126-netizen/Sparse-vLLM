---
name: write-optimization-tech-doc
description: Create concise Markdown technical documents for Sparse-vLLM optimization points, especially attnpredict-offload/runtime/cache-manager optimizations. Use when Codex needs to turn an optimization idea, code change, profiler result, or benchmark result into a readable Chinese technical report section with mechanism, scope, benefits, limitations, and observed results.
---

# Write Optimization Tech Doc

Use this skill to write one focused `.md` document for one optimization point. The document should be short enough for a meeting report, but precise enough that another engineer can understand what changed and why it helped.

## Workflow

1. Identify the optimization point name, affected phase, changed code path, and evidence source.
2. Read the relevant code/report before writing. Do not invent benchmark numbers, call counts, or profiler results.
3. Create or update one Markdown file. In this repo, default to `attnpredict-offload资料/YYYY-MM-DD_<优化点名>.md` unless the user specifies another path.
4. Keep the document focused on this optimization only. Do not mix unrelated optimizations.
5. Prefer direct observed metrics over derived estimates. If data is missing, write “暂未单独测量” instead of guessing.

## Document Shape

Use this structure unless the user provides a stricter format:

```md
# 优化N

## <优化点名称>

> 核心原理：用一句话说明该优化如何把原路径换成更轻的路径，以及减少了哪类开销。

### 背景

说明原路径在做什么、为什么有额外成本。只解释理解该优化必须知道的概念。

### 优化方法

说明改了什么、在哪个函数/阶段发生、触发条件是什么。代码片段要短。

> 说明：用 Markdown 引用块解释容易误解的点，不能用 HTML 注释。

### 为什么有收益

列出有证据或机制明确的收益来源。不要写泛泛的“更符合缓存/带宽习惯”这类无证据表述。

### 实验结果

说明实验配置、该优化作用于 prefill 还是 decode、是否直接减少单步 decode 时间。

| 指标 | 优化前 | 优化后 |
|---|---:|---:|
| <面向读者的指标名> | <数值> | <数值> |

> 观测结果：用一句话解释端到端结果。
```

## Style Rules

- 默认中文写作。英文工程词第一次出现时，只在必要处解释，不要在顶部堆基础名词表。
- 不要解释过于基础的词，例如用户已经在上下文中反复使用的 `prefill`、`decode`、`bs`、`output_len`。
- 关键注释使用 Markdown 引用块 `>`，不要使用 `<!-- ... -->`，也不要写成 Python 注释风格。
- 表格里的指标名要面向读者，不要直接暴露内部 `profiler.record(...)` 名称。比如写 `CPU backing 写入平均耗时`，不要写 `attnpredict_offload_store_cpu_full_kv`。
- 实验结果不要只写“单次多少 ms”。如果该路径会重复调用，要说明调用次数、作用阶段和端到端指标；没有可靠总量时不要估算。
- 不要写“估算累计减少”这类推导指标，除非用户明确要求。优先写实际观测的 `TTFT`、`DecTP`、`PreTP`、总耗时或 profiler 总时间。
- 明确说明该优化是否影响输出质量。只改变调度、写入方式、buffer 复用等不改变 token 集合的优化，可以写“不改变稀疏选择和预测结果，因此不影响输出质量”。
- 明确写适用范围和触发条件。例：只作用于 prefill；slots 必须连续；不连续时回退原路径。
- 删除无意义收益来源。没有测量或无法从代码机制直接推出的收益不要写。

## Example Patterns

解释索引写入换连续写入时，可以这样写：

```md
> 说明：即使 `cpu_slots = [5000, 5001, ...]`，`index_copy_` 也会按“读取索引 -> 定位目标 slot -> 写入”的散写流程执行，不能直接退化成简单的连续 slice copy。
```

解释 segment 复用时，可以这样写：

```md
> 例子：一次 prefill 准备阶段得到 segment `(0, 4096, 5000, 9096)`，第 0 层到第 31 层写 CPU backing 时都可以复用这个 segment，只是写入的 `layer_idx` 不同。
```

写结果时，可以这样写：

```md
实验配置为 `128k / bs=2 / output_len=16`。该优化发生在 prefill 阶段，不直接减少每个 decode step 的计算时间。

| 指标 | 优化前 | 优化后 |
|---|---:|---:|
| CPU backing 写入平均耗时 | 47.67 ms/次 | 47.05 ms/次 |
| 单次调用平均减少 | - | 0.62 ms/次 |
| TTFT | 53.20 s | 52.56 s |
| PreTP | 4812.18 tok/s | 4870.42 tok/s |

> 观测结果：TTFT 从 `53.20 s` 降到 `52.56 s`，说明该优化对 128k / bs=2 的 prefill 阶段有小幅正收益。
```

## Final Check

Before finishing, verify:

- The file contains only this optimization point.
- The core principle is one sentence in a `>` block near the top.
- The result section uses reader-facing names and observed metrics.
- No internal profiler range name is exposed unless the user explicitly asks for raw profiler labels.
- No estimate is presented as a measured result.
- Scope, trigger condition, and quality impact are stated.
