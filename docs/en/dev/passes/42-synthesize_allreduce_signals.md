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
[`MaterializeCommDomainScopes`](43-materialize_comm_domain_scopes.md), while the
host `alloc_window_buffer` / `window` / dispatch chain is still visible.
Materialization then sees the synthesized signal buffer as an ordinary window
allocation and can place it in the same communication domain as the allreduce
data buffer.

## Algorithm

For every host-orchestration function:

1. Collect existing variable names in the program.
2. Pre-scan the body: does it carry any `pld.tensor.allreduce` call, and does
   any of those calls omit the signal argument?
3. If the function has implicit-signal (single-argument) calls, group them by
   the lineage of their data buffer (traced back through `pld.tensor.window` to
   the `pld.tensor.alloc_window_buffer` LHS), and hoist one shared signal
   binding per group to the top of the function body:

    ```python
    __allreduce_signal_world_size_0 = pld.system.world_size()
    __allreduce_signal_buf_0: pl.Ptr = pld.tensor.alloc_window_buffer(__allreduce_signal_world_size_0 * core_num * pl.INT32.get_byte())
    __allreduce_signal_0 = pld.tensor.window(
        __allreduce_signal_buf_0,
        [__allreduce_signal_world_size_0, core_num],
        dtype=pl.INT32,
    )
    ```

4. Rewrite every implicit-signal `pld.tensor.allreduce` call — including calls
   inside `for` / `while` loops — to pass the shared signal:

    ```python
    data = pld.tensor.allreduce(data, __allreduce_signal_0, op=pld.ReduceOp.Sum)
    ```

5. Preserve calls that already pass an explicit signal argument unchanged;
   return-position calls are still lifted to an assignment so host lowering can
   dispatch them.

The generated signal shape is rank-2 `[world_size, core_num]`, where `core_num`
is the widest lane count requested by that lineage group's implicit-signal calls
(the default `core_num=1` preserves the previous single-lane representation);
the byte allocation uses the same lane count. One binding is shared per
data-buffer lineage (device-coverage) group and reused by every implicit-signal
call over that buffer — correct because the host builtin kernels self-clear
their barrier cells after every call, so a reused signal is safe back-to-back
and across loop iterations. Distinct data buffers get distinct signals, so
implicit allreduces over different device subsets in one function do not merge
into a single comm-domain scope.

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
- an allreduce appears as a nested expression instead of a direct assignment,
  expression statement, or return value.

Implicit-signal calls inside `for` / `while` loops are accepted: the shared
signal for the call's data-buffer lineage is reused on every iteration, which
is correct because the host builtin kernels
(`builtin.tensor.allreduce` / `builtin.tensor.allreduce_ring`, lowered by
`LowerHostTensorCollectives`) self-clear their barrier cells after
every call via a credit-barrier epilogue. InCore composites lowered by
[`LowerCompositeOps`](12-lower_composite_ops.md#barrier-signal-protocol) are
loop-safe for the same reason — that pass emits the self-clearing epilogue.

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
