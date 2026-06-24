# FuseCreateAssembleToSlice Pass

Fuses `tensor.create` + `tensor.assemble` pairs into a single `tensor.slice` view, eliminating the intermediate buffer.

## Overview

Orchestration code commonly allocates a small staging tensor with `tensor.create`, fills it via an InCore call, and then writes the result back into a sub-region of an existing target with `tensor.assemble(target, source, offsets)`. When the staging tensor is consumed by **exactly one** assemble, the stage-then-copy pattern is equivalent to writing directly into a `tensor.slice(target, shape, offsets)` view of the target — no intermediate buffer is needed.

This pass detects that pattern and rewrites it: the `tensor.create` becomes a `tensor.slice` of the assemble target, and the assemble itself is dropped. Downstream codegen then writes through the slice view directly, avoiding both the temporary allocation and the explicit copy.

**Requirements**:

- The pass requires `IRProperty::SplitIncoreOrch` — Orchestration functions must already be split out from InCore code (`OutlineHierarchyScopes` / `OutlineIncoreScopes` / `OutlineClusterScopes` have run).
- Only Orchestration functions are scanned; InCore, AIC, AIV, and Opaque functions are returned unchanged.

**When to use**: 27th pass in the `Default` strategy, after `AllocateMemoryAddr` (so memory addresses are already assigned for any tensors that survive) and before `DeriveCallDirections` and the trailing `Simplify`. It is the last tensor-shape rewrite before call-direction inference.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::FuseCreateAssembleToSlice()` | `passes.fuse_create_assemble_to_slice()` | Program-level |

**Python usage**:

```python
from pypto.pypto_core import passes

fuse_pass = passes.fuse_create_assemble_to_slice()
program_fused = fuse_pass(program)
```

## Algorithm

For each Orchestration function (others are returned unchanged) the pass runs three phases:

1. **Buffer-root analysis** — `BufferRootCollector` walks the function body and builds a `var → root` map. Function parameters are their own roots; `tensor.create` and `tensor.slice` results define new roots; var-aliasing assignments inherit the root of the aliased value; and the result of `tensor.assemble(target, source, offsets)` inherits the root of `target` (arg0), not `source` (arg1). The collector also threads roots through `ForStmt` / `WhileStmt` iter args (linking `iter_arg`, the corresponding `return_var`, and the loop-body uses), tracks tuple roots for tuple-returning calls via `tuple_output_roots_` and resolves `TupleGetItemExpr` from those call results, and propagates roots through call output parameters whose direction is `Out`/`InOut`. The result is a single buffer identity per var across loop-carried state, supported tuple-returning call outputs, and cross-function aliasing.

2. **Pattern detection** — `AssemblePatternCollector` scans for the eligible pairs:
   - Each `tensor.create` whose root resolves to itself (i.e. the create is the buffer's origin) is recorded in `create_vars`.
   - Each `tensor.assemble(target, source, offsets)` whose `source` resolves back to a recorded create root is recorded as a candidate fuse with `FuseInfo{target_expr, offset_tuple}`.
   - If a single create root is observed in two or more assembles it is moved from `fusible_roots` to `non_fusible_roots` and excluded from rewriting.

3. **Rewrite** — `FuseCreateAssembleMutator` performs the IR mutation:
   - `tensor.create(shape, dtype)` → `tensor.slice(target, shape_tuple, offset_tuple)`. When the assemble target rank exceeds the created tile rank (e.g. a 2D tile assembled into a 3D tensor at `[b, p, q]`), the slice's shape tuple is padded with leading singleton `1` dims so that shape and offset ranks match.
   - The matched `tensor.assemble` AssignStmt is replaced with an empty `SeqStmts`, and the var it bound is remapped to the slice target so downstream uses still see the same identity.
   - When eliminating an assemble inside a `ForStmt` or `WhileStmt` body causes a yielded iter arg to become pass-through (`yield(iter_arg)` instead of `yield(new_value)`), `StripPassThroughIterArgs` / `StripPassThroughWhileIterArgs` removes that iter arg, drops the corresponding `return_var`, and substitutes the iter arg's `init_value` into the body. Other iter args carrying real loop-carried state are preserved.

| Source pattern | Action |
| -------------- | ------ |
| `create` assembled exactly once into a target | Replace `create` with `slice`; drop `assemble`; remap aliased var |
| `create` assembled ≥ 2 times | Marked non-fusible; IR unchanged |
| `create` not followed by any `assemble` | IR unchanged |
| `assemble` whose source is a `tensor.slice` (not a `create`) | IR unchanged |
| Function is InCore / AIC / AIV / Opaque | Function returned unchanged |

## Example

### Basic fusion

**Before**:

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def fill_row(
        self,
        x: pl.Tensor[[4, 8], pl.FP32],
        r: pl.Scalar[pl.INDEX],
        out: pl.Out[pl.Tensor[[1, 8], pl.FP32]],
    ) -> pl.Tensor[[1, 8], pl.FP32]:
        row_tile: pl.Tile[[1, 8], pl.FP32] = pl.load(x, [r, 0], [1, 8])
        out_1: pl.Tensor[[1, 8], pl.FP32] = pl.store(row_tile, [0, 0], out)
        return out_1

    @pl.function(type=pl.FunctionType.Orchestration)
    def orch(
        self,
        x: pl.Tensor[[4, 8], pl.FP32],
        out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
    ) -> pl.Tensor[[4, 8], pl.FP32]:
        for r in pl.range(4):
            row: pl.Tensor[[1, 8], pl.FP32] = pl.create_tensor([1, 8], dtype=pl.FP32)
            row = self.fill_row(x, r, row)
            out = pl.assemble(out, row, [r, 0])
        return out
```

**After**:

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.Orchestration)
    def orch(
        self,
        x: pl.Tensor[[4, 8], pl.FP32],
        out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
    ) -> pl.Tensor[[4, 8], pl.FP32]:
        for r in pl.range(4):
            row: pl.Tensor[[1, 8], pl.FP32] = pl.slice(out, [1, 8], [r, 0])
            row = self.fill_row(x, r, row)
        return out
```

The `pl.create_tensor` is replaced with a `pl.slice` view of `out`; `pl.assemble` is removed; the trailing `out = pl.assemble(...)` iter arg is stripped because it has become pass-through.

### Rank-padded shape (2D tile into 3D target)

When the assemble target has higher rank than the create, the slice shape is padded with leading singleton `1` dims so shape and offsets align:

**Before**:

```python
@pl.function(type=pl.FunctionType.Orchestration)
def orch(
    self,
    x: pl.Tensor[[4, 8], pl.FP32],
    out: pl.Out[pl.Tensor[[2, 4, 8], pl.FP32]],
) -> pl.Tensor[[2, 4, 8], pl.FP32]:
    for b in pl.range(2):
        for c in pl.range(2):
            col = c * 4
            chunk: pl.Tensor[[2, 4], pl.FP32] = pl.create_tensor([2, 4], dtype=pl.FP32)
            chunk = self.compute(x, chunk)
            out = pl.assemble(out, chunk, [b, 0, col])
    return out
```

**After**:

```python
@pl.function(type=pl.FunctionType.Orchestration)
def orch(
    self,
    x: pl.Tensor[[4, 8], pl.FP32],
    out: pl.Out[pl.Tensor[[2, 4, 8], pl.FP32]],
) -> pl.Tensor[[2, 4, 8], pl.FP32]:
    for b in pl.range(2):
        for c in pl.range(2):
            col = c * 4
            chunk: pl.Tensor[[1, 2, 4], pl.FP32] = pl.slice(out, [1, 2, 4], [b, 0, col])
            chunk = self.compute(x, chunk)
    return out
```

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

**Implementation**: `src/ir/transforms/fuse_create_assemble_to_slice_pass.cpp`

- `BufferRootCollector` (IRVisitor) — buffer-identity analysis through assignments, loop iter args, tuples, and call-output aliasing.
- `AssemblePatternCollector` (IRVisitor) — detects creates assembled exactly once; multi-assemble roots are excluded.
- `FuseCreateAssembleMutator` (IRMutator) — rewrites `create` → `slice`, drops the matched `assemble`, and strips pass-through iter args from `for` / `while` loops.

**Python binding**: `python/bindings/modules/passes.cpp`

**Tests**: `tests/ut/ir/transforms/test_fuse_create_assemble_to_slice.py`

- `test_basic_create_assemble_fused_to_slice` — basic fusion + iter-arg stripping
- `test_duplicate_assemble_not_fused` — multi-assemble negative case
- `test_slice_source_not_fused` — assemble whose source is a slice is not fused
- `test_multi_iter_arg_partial_fuse` — only the assembled iter arg is stripped; other state-carrying iter args survive
- `test_3d_target_2d_tile_offset_padded` — leading-singleton padding when target rank > tile rank
- `test_no_orchestration_function_noop` — pass is a no-op without an Orchestration function

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | `SplitIncoreOrch` |
| Produced | — |
| Invalidated | — |

The pass preserves all input properties: it rewrites Orchestration body statements only, does not introduce new IR forms outside `tensor.slice`, and is idempotent — re-running it on already-fused IR finds no `create + single assemble` pattern and is a no-op.

## Scope

| Function type | Action |
| ------------- | ------ |
| Orchestration | Scanned; eligible `create + single assemble` pairs fused to `slice` |
| InCore (InCore, AIC, AIV) | Returned unchanged |
| Opaque | Returned unchanged |

The pass is a no-op when no Orchestration function contains a fusable `create + assemble` pair.
