---
name: python-code-slim
description: Keep Python inference, cache-management, offload, CUDA-kernel integration, benchmark glue, and engineering-optimization code small, correctly placed, behavior-preserving, and testable. Use when implementing, refactoring, reviewing, or shrinking Python code where AI code bloat, excess abstraction, extra config, broad try/except blocks, verbose comments, or file sprawl would make Sparse-vLLM harder to read or tune.
---

# Python Code Slim

Use this skill when changing Python code for model inference systems, especially Sparse-vLLM cache managers, decode/prefill flow, offload, prefetch, CUDA/Triton wrapper calls, score buffers, benchmarks, and profiling glue.

The goal is not the fewest possible characters. The goal is the smallest code that is in the right place, keeps behavior correct, preserves performance, and stays easy to test.

## Value Order

Apply these priorities in order.

1. Correct location beats fewer lines.
2. Behavior preservation beats deletion.
3. Performance preservation beats elegance.
4. Fewer lines beat extra abstraction layers.
5. One existing file beats a new file.
6. A local helper beats a new class hierarchy.
7. A local constant beats a new config knob unless users need to tune it.
8. Let exceptions propagate unless handling adds real recovery, context, or cleanup.
9. Self-explaining code beats comments that restate the code.
10. Short critical comments beat long narrative comments.
11. Deleting behavior-changing code is a bug; revert or redesign.
12. Testability beats raw line count.

## Placement Rules

Before editing, decide where the code belongs.

- Cache-manager state: keep method-specific persistent state in `src/sparsevllm/engine/cache_manager/`.
- Decode view logic: prefer `build_decode_view(...)` and related cache-manager hooks.
- KV-store metadata updates: prefer `on_kv_stored(...)`.
- Attention layer: keep `src/sparsevllm/layers/attention.py` generic; it should call hooks or kernels, not own method policy.
- Cross-layer orchestration: use `src/sparsevllm/engine/sparse_controller.py`.
- Config: add knobs only when a user or benchmark must change them; do not expose every internal constant.
- Utils: use `src/sparsevllm/utils/` only for helpers shared by multiple methods.
- Benchmarks: keep benchmark parsing and result writing in benchmark scripts; do not hide runtime behavior there.

When adding a first-class sparse method, use `$add-sparse-method` first, then apply this skill to keep the implementation lean.

## Edit Workflow

### New Feature or Optimization

1. State the contract in one or two sentences: affected phase (`prefill`, `decode`, or both), target files, expected behavior, and performance budget.
2. Prefer editing an existing file. Create a new file only when the code has a distinct runtime owner.
3. Start with direct Python code and local helpers. Add classes, dataclasses, registries, or policy objects only when state lifetime or multiple call sites require them.
4. Add at most the config needed to reproduce experiments. Internal constants can stay local.
5. Keep tensor shape logic close to the operation that needs it.
6. Validate with `python -m py_compile` on touched Python files, then the smallest correctness or benchmark command that exercises the path.

If the request is ambiguous enough that two implementations would have different behavior or performance, ask a short clarifying question before coding.

### Refactor or Shrink

1. Establish the current behavior: read the relevant code, check recent diffs if useful, and identify the invariants that must not change.
2. Mark each removal as behavior-preserving or behavior-changing. Only apply behavior-preserving removals.
3. Merge duplicate helpers before creating new abstractions.
4. Remove unused config, wrappers, result containers, and comments after confirming no caller depends on them.
5. Run the same focused validation before and after when possible. If behavior or performance regresses, revert the shrink.

### Review

Report concrete issues, not style impressions.

For each issue, include:

- `path:line`
- bloat type: wrong location, unnecessary abstraction, unnecessary config, duplicate logic, broad exception handling, verbose comments, or behavior-risky deletion
- exact slim action: delete, merge, move, inline, or keep with reason

Prioritize correctness and performance risks before line-count complaints.

## Python Inference Slimming Rules

- Do not split `prefill` and `decode` code unless their behavior really differs.
- Do not introduce `Policy`, `Plan`, `Result`, or `Manager` classes for values used in one function.
- Do not add async prefetch, background queues, residency caches, or CPU/GPU transfer machinery without a measured reason.
- Do not replace a vectorized tensor operation with shorter Python loops on hot paths.
- Do not catch `Exception` in inference flow just to log and continue.
- Do not create a new config option for a constant that has no benchmark-facing need.
- Do not add a new file for a helper used by one module.
- Do not preserve old compatibility branches unless a real caller still uses them.
- Do not keep dead debug/profiling code in the runtime path; gate it behind existing env flags or remove it.

## Comment Rules

Use comments only for facts the code cannot show locally.

Good short comments:

```python
# Shape: [batch, active_slots, num_kv_heads, head_size].
# Keep H2D copy before decode reads the packed view.
# Sink/recent are already included in topk.
```

Avoid comments that narrate the next line, explain obvious Python, or preserve a design discussion. Put long reasoning in docs, not in hot runtime files.

## Size Checks

Treat these as review triggers, not excuses for clever unreadable code.

- A new private helper should usually stay under 30 logical lines.
- A new public method should usually stay under 50 logical lines.
- Adding more than 100 production lines to one file requires a short reason for why the behavior cannot be expressed by editing existing code.
- Growing an already large runtime file by more than about 20 percent requires checking whether old code can be removed in the same change.
- If shrinking code makes shapes, phases, or error behavior harder to test, keep the slightly longer version.

## Final Self-Check

Before finishing, answer these internally:

- Is each new line in the right owner?
- Can any new helper, class, config, or file be deleted without changing behavior?
- Did any deletion change behavior?
- Did the hot path keep the same or better tensor/device behavior?
- Are comments short and only about invariants, shapes, ordering, or non-obvious reasons?
- Did validation cover the touched path?
