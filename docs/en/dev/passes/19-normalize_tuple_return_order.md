# NormalizeTupleReturnOrder Pass

Reorders multi-value `ReturnStmt` values (and matching return types) so that each return slot aligns with **Out/InOut parameter declaration order**.

## Overview

InCore kernels may return a tuple of tensor handles whose **syntactic order** does not match the order of **Out/InOut** parameters (for example when mixing `tile.store` outputs with loop-carried `yield` outputs). Orchestration code generation previously traced each return value to its parameter; that analysis now runs in this pass so codegen can map tuple elements sequentially.

**Key responsibilities**:

- Inspect the function body using the same rules as the former codegen helper: top-level `tile.store` result variables and `ForStmt` yield chains (`iter_arg` → `return_var`)
- For functions with more than one Out/InOut parameter and a multi-value return, permute `ReturnStmt::value_` so that `return[i]` corresponds to the *i*-th Out/InOut parameter in declaration order
- Permute `Function::return_types_` (or the inner types of a single `TupleType`) consistently

**When to use**: Run after `SplitVectorKernel` and before `InitMemRef` (does not require MemRefs or allocated addresses). Required before orchestration codegen for tuple-returning callees.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::NormalizeTupleReturnOrder()` | `passes.normalize_tuple_return_order()` | Function-level |

**Factory function**:

```cpp
Pass NormalizeTupleReturnOrder();
```

**Python usage**:

```python
from pypto.pypto_core import passes

norm_pass = passes.normalize_tuple_return_order()
program = norm_pass(program)
```

## Preconditions

- Same tile pipeline stage as after `SplitVectorKernel`: `IRProperty::VectorKernelSplit` plus `SSAForm`, `SplitIncoreOrch`, `IncoreTileOps`, `TileOps2D`, and `TileMemoryInferred` (see `kNormalizeTupleReturnOrderProperties` in `pass_properties.h`).

## Failure modes

If a multi-value return cannot be traced unambiguously to Out/InOut parameters, the pass fails with an internal check (the IR cannot be normalized safely).

## Related

- Issue [#814](https://github.com/hw-native-sys/pypto/issues/814)
- [Orchestration codegen](../codegen/01-orchestration_codegen.md) — expects normalized return order for tuple outputs
