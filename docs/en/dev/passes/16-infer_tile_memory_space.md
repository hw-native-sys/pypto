# InferTileMemorySpace Pass

Infers the on-chip `MemorySpace` for every `TileType` variable inside InCore functions, inserts `tile.move` ops to legalize residual mismatches between producers and consumer constraints, and keeps provably loop-invariant Mat operands resident across sequential loops.

## Overview

After `FlattenTileNdTo2D`, every InCore tile has a static 2D shape but its `TileType::memory_space_` is still unset (or only set on a subset of producers via the `target_memory` kwarg). The PTO-ISA hardware exposes several distinct on-chip buffers — `Vec` (unified buffer / vector), `Mat` (L1), `Left` / `Right` (L0A / L0B matmul operand buffers), `Acc` (L0C accumulator), `Bias` — and most ops constrain which spaces their inputs and outputs may live in. This pass runs that constraint solver: it forwards memory spaces along data flow, honors explicit `target_memory` kwargs, propagates demand backward through view chains, and inserts `tile.move` where producer and consumer cannot agree on a single space.

After this pass every `TileType` in InCore functions carries a concrete `memory_space_`, satisfying the `TileMemoryInferred` IR property required by `ExpandMixedKernel`, `InitMemRef`, and downstream codegen.

**Requirements**:

- Input IR must be in SSA form (`SSAForm`)
- Input IR must have InCore tile ops (`IncoreTileOps`)
- InCore / Orchestration outlining must be done (`SplitIncoreOrch`)
- Statement structure must be normalized (`NormalizedStmtStructure`)

**When to use**: Run immediately after `FlattenTileNdTo2D` and before `ResolveBackendOpLayouts` / `ExpandMixedKernel`. It is the canonical point at which tile memory becomes a contract that downstream passes (especially `ExpandMixedKernel`'s mixed-kernel detection and `InitMemRef`'s buffer allocation) read.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::InferTileMemorySpace()` | `passes.infer_tile_memory_space()` | Program-level |

**Python usage**:

```python
from pypto.pypto_core import passes

infer_pass = passes.infer_tile_memory_space()
program_inferred = infer_pass(program)
```

The pass only rewrites functions whose `func_type_ == FunctionType::InCore`. Orchestration and Opaque functions pass through unchanged.

## Algorithm

Each InCore function is processed in five phases. All phases run as IR visitors / mutators with O(N log N) complexity in the size of the function body. Phase 4 builds one bottom-up loop inventory; it never rescans a complete nested body for every ancestor loop.

### Phase 0 — Backward demand collection (`DemandCollector`)

Walks the function body once and records two pieces of information:

1. For every `Call` whose op has `input_constraints` registered in `OpRegistry`, the *first* allowed memory space for each constrained input is recorded as a "demand" on the input variable. Backends list the canonical (cheapest, no-move) space first — e.g. `tile.store` lists `{Vec, Acc}` so a Vec producer needs no move and an Acc-origin tile keeps its space.
2. For every op marked `OutputMemoryInheritsInput()` (e.g. `tile.fillpad`, `tile.slice`, `tile.reshape`), an edge `dst → src` from output var to first tile-typed input is captured in program order.

Demands are then propagated *backward* through those edges by a single reverse-order sweep. Because inherit-input ops in SSA always have `dst` defined after `src`, one reverse pass reaches the fixed point in O(N). When two demands collide on the same var, a non-`Vec` demand wins (`ShouldOverrideDemand`) — `Vec` is the permissive default and a specialized demand from a compute op should override it.

This phase is what lets `slice(tensor) → fillpad → matmul` push the matmul's `Left`/`Right` demand all the way back to the `tile.slice` output, so Phase 1 can resolve that producer directly to `Left`/`Right` instead of routing through `Vec`.

### Phase 1 — Forward analysis (`TileMemorySpaceAnalyzer`)

Walks the function body and assigns a `MemorySpace` to every TileType variable, storing the result in a `var_memory_` map.

For each `AssignStmt` whose LHS has `TileType`, the analyzer dispatches by RHS shape:

- **`Call` to a `tile.*` op** → `InferFromOp` (see resolution table below).
- **`Call` to a non-`tile.*` op** producing TileType → defaults to `Vec`.
- **Plain SSA alias `y = x`** → inherit `x`'s memory space. The Python frontend emits these when eliding no-op `tensor.fillpad(pad=zero)` calls whose input already has a matching `valid_shape`; the alias is value-identical to its source and must agree on memory space.

For each `ForStmt` with `return_vars_`, after visiting the body the analyzer copies the memory space of each yielded var to the corresponding `return_var_`. Critically, the same space is also forced onto:

- The matching `iter_arg_` — covers the accumulator pattern where `tile.create` conservatively defaults to `Vec` but the loop body writes a different space (e.g. `Acc` from `matmul_acc`). Without this back-propagation the final `tile.store` reads a `Vec` tile and `ExpandMixedKernel` misclassifies the kernel as mixed, producing broken AIC/AIV IR.
- The TileType `init_var_` carrier underneath the `iter_arg_` — handles cases where an `IfStmt` `return_var` (never visited as an `AssignStmt`) is used as a loop init.

#### Per-op resolution table (Phase 1)

| Producer kind | Resolved memory space |
| ------------- | --------------------- |
| Unregistered cube ops (`tile.matmul_mx*`) | `Acc` |
| Other unregistered ops | `Vec` |
| Registered op with no `MemorySpec` | Read from `Call` return type if set & not `DDR`; else `Vec` |
| Registered op with `deduce_output_memory` returning `Some(s)` (e.g. `tile.matmul → Acc`) | `s` |
| `output_inherits_input` op (e.g. `tile.slice`, `tile.fillpad`, `tile.reshape`) and resolver returned `None` | First tile input's space; else `Vec` |
| `HasRetargetableMemoryKwarg()` op (e.g. `tile.load`, `tile.create`) and resolver returned `None` (kwarg absent) | Phase-0 demand if it is `Vec` or `Mat`; otherwise input-inherit; else `Vec` |
| `tile.*` op with `deduce_output_memory` returning `None` and not retargetable / not inherit | Input-inherit; else `Vec` |

The "clamp to `{Vec, Mat}`" step on retargetable producers is deliberate: a DDR-facing `tile.load` cannot directly produce `Left`/`Right`/`Acc`/`Bias`, so even when downstream demand is one of those, the producer must stop at `Mat` (or `Vec`) and Phase 2 inserts a `tile.move` to reach the specialized space.

The pass *never* overrides a present `target_memory` kwarg in Phase 1. If a user wrote `pl.load(..., target_memory=Mat)` and a downstream `matmul` demands `Left`, the load stays at `Mat` and a `tile.move` is inserted.

### Phase 2 — Move collection (`MoveCollector`)

Walks the function body again. For every `Call` whose op has `input_constraints`, it checks each constrained input variable's resolved `var_memory_` against the allowed list. Any mismatch is recorded as a `MoveKey = (producer_var, target_space)` in `needed_moves_`, where `target_space` is the first allowed space for that input slot. Phase 3 will materialize at most one `tile.move` per unique key per enclosing `SeqStmts` scope (i.e. per insertion-site cache scope), so the same `(producer_var, target_space)` may still appear in sibling scopes such as `then` / `else` branches.

### Phase 3 — Mutation (`TileMemorySpaceMutator`)

A full `IRMutator` rewrite that produces the new function body:

1. **Var rewrite (`VisitExpr_(Var)`)** — for every TileType var with a resolved space, build a fresh `Var` whose `TileType` carries `memory_space_` set. When the space changes, also refresh the `tile_view_` to the implicit view for the new space (e.g. `Acc` expects col_major / row_major / fractal=1024 rather than the Vec-style row_major / none_box / fractal=512). Cached in `var_cache_` so identity holds across multiple references to the same var.
2. **`tile.move` insertion (`VisitStmt_(SeqStmts)` → `InsertMovesForConsumer`)** — at every `AssignStmt` / `EvalStmt` whose RHS is a constrained `Call`, for each input that has a pending `MoveKey`, emit a fresh `tile.move` `AssignStmt` *before* the consumer. The new `Var` (`<orig>_<TargetSpace>`) is recorded in `created_moves_`, scoped to the enclosing `SeqStmts` so a move emitted inside the `then` branch of an `IfStmt` does not leak into the `else` branch (which would leave a dangling SSA reference). When the backend is configured, `BackendTileLayoutSpec::input_layouts` is consulted so the inserted `tile.move` carries the consumer-required `blayout` (and `slayout=none_box` for `Vec` targets), avoiding a later `ResolveBackendOpLayouts` repair.
3. **Argument substitution (`VisitExpr_(Call)`)** — replaces each constrained input arg with the matching `created_moves_` entry where present.
4. **Retargetable producer kwarg rewrite (`VisitStmt_(AssignStmt)`)** — for ops registered with `HasRetargetableMemoryKwarg()`, if Phase 1 resolved the output to a different space than the kwarg said (or the kwarg was absent), rewrite the `Call`'s `target_memory` kwarg and the result `TileType` to match. This keeps codegen and the assigned `Var` annotation in sync, and is necessary because Phase 1 may have resolved the producer using backward demand that the kwarg never saw.
5. **LHS / RHS type sync** — when `VisitExpr_(Call)` rebuilds a `Call` via `OpRegistry` after argument substitution, the deduced result type may differ from the LHS `Var`'s original type (the rebuilt call sees inputs with new layouts). The mutator syncs the LHS Var's `TileType` to the rebuilt call's shape / dtype / memref / view while preserving the `memory_space_` chosen by Var rewrite, so roundtrip equality is preserved.

### Phase 4 — Loop-invariant Mat residency (`LoopInvariantTileLoadHoister`)

After all spaces are explicit, the pass recognizes an invariant prefix of the form `tile.load(GM → Mat) → tile.transpose_view* → tile.move/tile.extract(Mat → Left/Right)` and moves that prefix to the loop preheader. The chain must have one terminal `Left` / `Right` value used in the matching operand slot of exactly one `tile.matmul`, `tile.matmul_bias`, or `tile.matmul_acc`. This keeps the optimization specific to stationary matmul operands rather than turning it into general tile LICM. A stationary tensor-level matmul operand is therefore loaded once while a loop-dependent peer remains inside the loop and streams normally.

Eligibility begins with private compiler provenance. `ConvertTensorToTileOps` marks every generated `GM → Mat` bridge load; the marker survives printing, flattening, and L0 auto-tiling until this phase consumes it. Phase 4 then proves that a marked load participates in the exact stationary matmul-operand chain described above. A user-authored `tile.load(..., target_memory=Mat)` is not marked and is never hoisted by this optimization. This boundary keeps explicit tile programs under user control.

The initial legality rules are intentionally strict:

- the loop is `Sequential`, has constant bounds, positive step, and at least one iteration;
- every moved assignment is an unconditional top-level loop-body statement;
- the GM source is a direct `ParamDirection::In` tensor parameter on a compiler-marked Mat bridge load;
- the InCore function has at least one analyzable in-program call site, and at every call site the candidate `Tensor In` actual root and every writable `Tensor Out` / `Tensor InOut` actual root are known;
- at every call site, the candidate root is distinct from every writable root, and the InCore function does not write that root locally; unrelated scalars and peer read-only `Tensor In` roots do not participate in this proof;
- offsets, shapes, and the complete moved dependency prefix are loop invariant;
- a direct control-flow or effect statement before the candidate closes the hoistable prefix;
- no moved result is loop-carried or yielded;
- all allocation-owning `Mat`, `Left`, and `Right` tiles in the function have static sizes, with their allocator-aligned whole-function upper bound no larger than the backend capacity; and
- the function has no explicit reserved-buffer region whose capacity contribution is not represented as a tile allocation.

`InOut` / `Out` sources, manual tile loads, direct or externally-entered InCore functions, unknown candidate or writable call-site roots, candidate/write aliases, conditional loads, dynamic or zero-trip loops, capacity-unknown cases, yielded or loop-carried results, and loop-dependent extracts decline without changing the IR. One unsafe call site invalidates the candidate even when other calls are safe. The capacity test counts allocation owners rather than zero-copy views or SSA aliases, uses the same byte sizing and address alignment as `InitMemRef` / `AllocateMemoryAddr`, and includes allocations already live outside the loop. A memory space containing allocations that later pipeline lowering may replicate is also declined unless the moved prefix stays in an unaffected space. This whole-function bound is deliberately stronger than either planner's lifetime reuse, so a residency rewrite cannot introduce a later capacity failure under the PyPTO or PTOAS planner. Nested invariant chains move one preheader at a time and can reach the outermost legal loop. This phase does not globally remap parameters and does not move K-dependent L0 extracts out of `AutoTileMatmulL0` pipeline loops.

#### Residency example

For a caller-backed tensor program, the stationary LHS bridge moves before the user loop, while the N-dependent RHS remains streamed:

```python
# Tensor source
for n, (acc,) in pl.range(0, 256, 128, init_values=(out,)):
    rhs_n = pl.slice(rhs, [128, 128], [0, n])
    c_n = pl.matmul(lhs, rhs_n, out_dtype=pl.FP32)
    result = pl.yield_(pl.assemble(acc, c_n, [0, n]))
```

```python
# After conversion, L0 auto-tiling, and InferTileMemorySpace
lhs_mat = pl.tile.load(lhs, [0, 0], [16, 128], target_memory=pl.Mem.Mat)
lhs_left = pl.tile.move(lhs_mat, target_memory=pl.Mem.Left)
for n, (acc,) in pl.range(0, 256, 128, init_values=(out,)):
    rhs_mat = pl.tile.load(rhs, [0, n], [128, 128], target_memory=pl.Mem.Mat)
    rhs_right = pl.tile.move(rhs_mat, target_memory=pl.Mem.Right)
    c_n = pl.tile.matmul(lhs_left, rhs_right)
    result = pl.yield_(pl.tile.store(c_n, [0, n], acc))
```

The internal provenance attribute is omitted above for readability. The transformation requires an orchestration caller that proves the candidate `lhs` root is distinct from the writable `out` root. The peer read-only `rhs` root is irrelevant to that proof. Without the required caller evidence, the original loop-local placement is retained.

## General memory-space inference example

Source: `tests/ut/ir/transforms/test_infer_tile_memory_space.py::test_matmul_gets_acc`.

**Before**:

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def main_incore_0(
        self,
        x: pl.Tensor[[16, 128], pl.BF16],
        y: pl.Tensor[[128, 128], pl.BF16],
        out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
    ) -> pl.Tensor[[16, 128], pl.FP32]:
        x_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(x, [0, 0], [16, 128])
        y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128])
        z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_tile, y_tile)
        out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_tile, [0, 0], out_0)
        return out_0
```

**After**:

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.InCore)
    def main_incore_0(
        self,
        x: pl.Tensor[[16, 128], pl.BF16],
        y: pl.Tensor[[128, 128], pl.BF16],
        out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
    ) -> pl.Tensor[[16, 128], pl.FP32]:
        x_tile: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Vec] = pl.load(x, [0, 0], [16, 128])
        y_tile: pl.Tile[[128, 128], pl.BF16, pl.MemorySpace.Vec] = pl.load(y, [0, 0], [128, 128])
        x_tile_L: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Left] = pl.move(
            x_tile, target_memory=pl.MemorySpace.Left
        )
        y_tile_R: pl.Tile[[128, 128], pl.BF16, pl.MemorySpace.Right] = pl.move(
            y_tile, target_memory=pl.MemorySpace.Right
        )
        z_tile: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Acc] = pl.matmul(x_tile_L, y_tile_R)
        out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_tile, [0, 0], out_0)
        return out_0
```

What changed:

- Both `tile.load` outputs got `pl.MemorySpace.Vec` (no `target_memory` kwarg, no Mat demand reachable for these particular inputs).
- `tile.matmul`'s `deduce_output_memory` resolved its output to `Acc`.
- `tile.matmul`'s input constraints (`Left`, `Right`) did not match the producer's `Vec`, so Phase 2 recorded two move keys and Phase 3 inserted `x_tile_L`, `y_tile_R` immediately before the consumer.

If the user had instead written `pl.load(..., target_memory=pl.MemorySpace.Mat)` for both inputs, Phase 1 would honor the kwarg and the `tile.load` outputs would already be `Mat`. The matmul still demands `Left`/`Right`, so the moves are inserted starting from `Mat` — which is the canonical full pipeline tested by `test_matmul_full_pipeline`.

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

**Implementation**: `src/ir/transforms/infer_tile_memory_space_pass.cpp`

**Python binding**: `python/bindings/modules/passes.cpp`

**Tests**: `tests/ut/ir/transforms/test_infer_tile_memory_space.py`

The pass also registers a `TileMemoryInferred` `PropertyVerifier` (defined in the same `.cpp`) that runs whenever the `TileMemoryInferred` IR property must be verified. It checks two invariants on every InCore function:

1. Every TileType `Var` defined by an `AssignStmt` has `memory_space_` set.
2. Every `Call` input that has registered `input_constraints` references a tile whose `memory_space_` is in the allowed set.

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | `SSAForm`, `IncoreTileOps`, `SplitIncoreOrch`, `NormalizedStmtStructure` |
| Produced | `SSAForm`, `TileMemoryInferred`, `NormalizedStmtStructure` |
| Invalidated | — |

The `TileMemoryInferred` property is the contract this pass establishes. Downstream passes (notably `ExpandMixedKernel` and `InitMemRef`) rely on it, and the matching property verifier guards regressions.

## Scope

| Function kind | Action |
| ------------- | ------ |
| `InCore` (incl. `AIC`, `AIV`) | Transformed |
| `Orchestration` | Unchanged |
| `Opaque` | Unchanged |

The pass also asserts that no InCore function parameter has `TileType` — InCore params must be `TensorType`. This is checked at the start of Phase 1 and raises a `CHECK` failure if violated.
