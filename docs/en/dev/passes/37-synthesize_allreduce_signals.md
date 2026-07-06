# SynthesizeAllReduceSignals Pass

## Overview

`SynthesizeAllReduceSignals` normalizes host-level
`pld.tensor.allreduce(data, op=...)` calls to the internal explicit-signal IR
form. It keeps the public host DSL ergonomic while preserving the existing
downstream contract:

```python
data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
```

The pass only rewrites host orchestrator functions. InCore allreduce continues
to use the explicit signal argument and is lowered by
[`LowerCompositeOps`](12-lower_composite_ops.md).

## Position in the pipeline

```text
... -> ExpandManualPhaseFence -> SynthesizeAllReduceSignals -> MaterializeCommDomainScopes -> LowerHostTensorCollectives -> Simplify (final)
```

The pass runs immediately before
[`MaterializeCommDomainScopes`](38-materialize_comm_domain_scopes.md), while the
host `alloc_window_buffer` / `window` / dispatch chain is still visible.
Materialization then sees the synthesized signal buffer as an ordinary window
allocation and can place it in the same communication domain as the allreduce
data buffer.

## Algorithm

For every host-orchestration function:

1. Collect existing variable names in the program.
2. Visit direct `AssignStmt`, `EvalStmt`, and `ReturnStmt` forms of
   `pld.tensor.allreduce`.
3. Preserve calls that already pass an explicit signal argument.
4. For calls that pass only the target tensor, allocate fresh generated names
   for a private signal buffer and signal view.
5. Insert statement-level bindings immediately before the allreduce call:

```python
__allreduce_signal_world_size_0 = pld.system.world_size()
__allreduce_signal_buf_0: pl.Ptr = pld.tensor.alloc_window_buffer(__allreduce_signal_world_size_0 * 4)
__allreduce_signal_0 = pld.tensor.window(
    __allreduce_signal_buf_0,
    [__allreduce_signal_world_size_0, 1],
    dtype=pl.INT32,
)
data = pld.tensor.allreduce(data, __allreduce_signal_0, op=pld.ReduceOp.Sum)
```

The generated signal shape is rank-2 `[world_size, 1]`. This matches the InCore
allreduce signal indexing model and gives host lowering a single canonical
signal representation.

## Print / Parse Round Trip

The synthesized buffer allocation is emitted as a normal assignment. The IR call
may carry the internal `name` kwarg for consumers, but the Python printer omits
that kwarg and relies on the assignment LHS. When the printed source is parsed
again, the parser derives the buffer name from the LHS exactly as it does for
user-written `pld.tensor.alloc_window_buffer` statements.

This keeps dump / reparse flows stable: the printed program contains ordinary
DSL statements, and reparsing reconstructs the same alloc / window / allreduce
chain.

## Checks

The pass raises `pypto::ValueError` when:

- an allreduce call has a positional argument count other than `target` or
  `target, signal`,
- an allreduce appears inside a `for` or `while` loop,
- an allreduce appears as a nested expression instead of a direct assignment,
  expression statement, or return value.

Loop allreduce is rejected because the current signal protocol is single-use;
the compiler must not reuse one signal buffer across dynamic invocations.

## Pass Properties

| Field | Value |
| ----- | ----- |
| `required` | `{}` |
| `produced` | `{}` |
| `invalidated` | `{}` |

## Reference

- Source: [src/ir/transforms/synthesize_allreduce_signals_pass.cpp](../../../../src/ir/transforms/synthesize_allreduce_signals_pass.cpp)
- Header: [include/pypto/ir/transforms/passes.h](../../../../include/pypto/ir/transforms/passes.h)
- Tests: [tests/ut/ir/transforms/test_materialize_comm_domain_scopes.py](../../../../tests/ut/ir/transforms/test_materialize_comm_domain_scopes.py)
