# InferTileMemorySpace Pass

Infers the on-chip `MemorySpace` for every `TileType` variable inside InCore functions, inserts `tile.move` ops to legalize residual mismatches between producers and consumer constraints, and keeps provably loop-invariant Mat operands resident across sequential loops.

## Overview

After `FlattenTileNdTo2D`, every device-function tile has a static 2D shape but its `TileType::memory_space_` is still unset (or only set on a subset of producers via the `target_memory` kwarg). The PTO-ISA hardware exposes several distinct on-chip buffers — `Vec` (unified buffer / vector), `Mat` (L1), `Left` / `Right` (L0A / L0B matmul operand buffers), `Acc` (L0C accumulator), `Bias` — and most ops constrain which spaces their inputs and outputs may live in. This pass runs that constraint solver: it forwards memory spaces along data flow, honors explicit `target_memory` kwargs, propagates demand backward through view chains, and inserts `tile.move` where producer and consumer cannot agree on a single space.

After this pass every `TileType` in every InCore-variant function -- `InCore`, `AIC` and `AIV`, all three of which are user-writable -- carries a concrete `memory_space_`, satisfying the `TileMemoryInferred` IR property required by `ExpandMixedKernel`, `InitMemRef`, and downstream codegen.

**Requirements**:

- Input IR must be in SSA form (`SSAForm`)
- Input IR must have InCore tile ops (`IncoreTileOps`)
- InCore / Orchestration outlining must be done (`SplitIncoreOrch`)
- Statement structure must be normalized (`NormalizedStmtStructure`)

**When to use**: Run after `FlattenTileNdTo2D` — with `LegalizeTileCast`, `AutoTileMatmulL0` and `CanonicalizeTileSlice` in between — and before `InsertMxScaleAddr` / `ResolveBackendOpLayouts` / `ExpandMixedKernel`. It is the canonical point at which tile memory becomes a contract that downstream passes (especially `InsertMxScaleAddr`'s scale binding, `ExpandMixedKernel`'s mixed-kernel detection and `InitMemRef`'s buffer allocation) read.

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

The pass rewrites every InCore-variant function -- `IsInCoreType`, i.e. `InCore`, `AIC` and `AIV`. Orchestration and Opaque functions pass through unchanged; a tile reaching `InitMemRef` from one of those is reported as an authoring error, since a non-device function has no on-chip buffer to place it in.

## Algorithm

Each InCore function is processed in five phases implemented as IR visitors / mutators. Phase 4 builds one bottom-up loop inventory and one complete syntactic-use map. It analyzes each loop's original direct body exactly once, so nested loops are rewritten independently and the phase remains O(N). A chain moves across at most one lexical loop per pass invocation rather than repeatedly bubbling through newly created preheaders.

### Phase 0 — Backward demand collection (`DemandCollector`)

Walks the function body once and records two pieces of information:

1. For every `Call` whose op has `input_constraints` registered in `OpRegistry`, the *first* allowed memory space for each constrained input is recorded as a "demand" on the input variable. Backends list the canonical (cheapest, no-move) space first — e.g. `tile.store` lists `{Vec, Acc}` so a Vec producer needs no move and an Acc-origin tile keeps its space.
2. For every op marked `OutputMemoryInheritsInput()` (e.g. `tile.fillpad`, `tile.slice`, `tile.reshape`), an edge `dst → src` from output var to first tile-typed input is captured in program order.

Demands are then propagated *backward* through those edges by a single reverse-order sweep. Because inherit-input ops in SSA always have `dst` defined after `src`, one reverse pass reaches the fixed point in O(N). When two demands collide on the same var, a non-`Vec` demand wins (`ShouldOverrideDemand`) — `Vec` is the permissive default and a specialized demand from a compute op should override it.

This phase is what lets `slice(tensor) → fillpad → matmul` push the matmul's `Left`/`Right` demand all the way back to the `tile.slice` output. Phase 1 then resolves that producer to `Mat` — the staging space a cube demand maps to, per the table below — and Phase 2 inserts the `Mat -> Left` / `Mat -> Right` move. The demand is what selects `Mat` over `Vec`; without it the operand would take the longer GM -> UB -> L1 -> L0 route.

### Phase 1 — Forward analysis (`TileMemorySpaceAnalyzer`)

Walks the function body and assigns a `MemorySpace` to every TileType variable, storing the result in a `var_memory_` map.

For each `AssignStmt` whose LHS has `TileType`, the analyzer dispatches by RHS shape:

- **`Call` to a `tile.*` op** → `InferFromOp` (see resolution table below).
- **`Call` to a non-`tile.*` op** producing TileType → defaults to `Vec`.
- **Plain SSA alias `y = x`** → inherit `x`'s memory space. The Python frontend emits these when eliding no-op `tensor.fillpad(pad=zero)` calls whose input already has a matching `valid_shape`; the alias is value-identical to its source and must agree on memory space.

For each `ForStmt` with `return_vars_`, after visiting the body the analyzer copies the memory space of each yielded var to the corresponding `return_var_`. Critically, the same space is also forced onto:

- The matching `iter_arg_` — covers the accumulator pattern where the loop body writes a space the init carrier does not yet have (e.g. `Acc` from `matmul_acc`). This used to be load-bearing because `tile.create` stamped `Vec` by default and the back-propagation had to overwrite it; now an unset `tile.create` simply resolves to `Acc` from demand, and the back-propagation only has to reach carriers no `AssignStmt` traversal visits. Without this back-propagation the final `tile.store` reads a `Vec` tile and `ExpandMixedKernel` misclassifies the kernel as mixed, producing broken AIC/AIV IR.
- The TileType `init_var_` carrier underneath the `iter_arg_` — handles cases where an `IfStmt` `return_var` (never visited as an `AssignStmt`) is used as a loop init, and cases where a nested loop's init is the enclosing loop's `IterArg`. The enclosing loop's own back-propagation cannot stand in for the latter: it runs only after its whole body has been analysed, so a mid-body consumer of the outer carry (an `output_inherits_input` op such as `tile.reshape`) resolves against whatever the carry holds at that moment.

Both the yield lookup and the init-carrier lookup use `AsVarLike`, not `As<Var>`. A carry yielded unchanged — the pass-through slot of `pl.yield_(a, b_next)` — *is* the `IterArg`, and a nested loop seeds its carries with the enclosing loop's `IterArg`; `IterArg` has its own `ObjectKind`, so `As<Var>` returns null for both and the slot is skipped silently. See `.claude/rules/ir-kind-traits.md`.

For each `IfStmt` with `return_vars_`, the analyzer likewise records each TileType phi from its branch yields (then-branch canonical, else-branch as fallback, the phi's own annotation last). This is the sibling of the `ForStmt` carry propagation above and is equally load-bearing: without it a phi never enters `var_memory_`, and every consumer that looks it up degrades silently on the miss — `InheritFromInput` falls through to a co-argument, Phase 2's `CheckInputConstraints` skips the operand entirely (queueing **no** `tile.move`, so the op's declared input space is left violated), and Phase 3 skips the retype. The shape that exposed it: `pl.cast` of an `if`/`else` accumulator phi kept its `Acc` operand although `tile.cast` requires `Vec`, leaving `ExpandMixedKernel` no boundary `tile.move` to lower into a `tpush_to_aiv` / `tpop_from_aic` pair. The spaces are derived from the yields rather than read off the phi's annotation because a branch may have been re-inferred during the same run (the accumulator pattern above), which leaves that annotation stale.

#### Per-op resolution table (Phase 1)

| Producer kind | Resolved memory space |
| ------------- | --------------------- |
| Registered cube ops (`tile.matmul`, `tile.matmul_mx`, …) | From op memory spec (`Acc`) |
| Other unregistered ops | `Vec` |
| Registered op with no `MemorySpec` | Read from `Call` return type if set & not `DDR`; else `Vec` |
| Registered op with `deduce_output_memory` returning `Some(s)` (e.g. `tile.matmul → Acc`) | `s` |
| `output_inherits_input` op (e.g. `tile.slice`, `tile.fillpad`, `tile.reshape`) and resolver returned `None` | First tile input's space; else `Vec` |
| `HasRetargetableMemoryKwarg()` op (e.g. `tile.load`, `tile.create`) and resolver returned `None` (kwarg absent) | Phase-0 demand if it is `Vec` or `Mat`; a cube-operand demand (`Left`, `Right`, `LeftScale`, `RightScale`, `Bias`) resolves to `Mat`; otherwise input-inherit; else `Vec` |
| `tile.*` op with `deduce_output_memory` returning `None` and not retargetable / not inherit | Input-inherit; else `Vec` |

`tile.load` of a `TensorLayout.MX_A_ZZ` / `MX_B_NN` source must carry an
explicit `target_memory=Mat`; type deduction rejects an omitted or different
target before this pass runs.

The clamp to `{Vec, Mat}` on retargetable producers is deliberate: a DDR-facing `tile.load` cannot directly produce `Left`/`Right`/`Acc`/`Bias`, so even when downstream demand is one of those, the producer must stop at `Mat` (or `Vec`) and Phase 2 inserts a `tile.move` to reach the specialized space.

Which of the two it stops at is decided by the demand, not by a default. A cube-operand demand resolves the producer to **`Mat`**, because L1 is the only buffer a `tload` can fill that MTE1 can then move into L0A/L0B — the `Mat -> Left` / `Mat -> Right` pairs are the only ones PTOAS implements (`TMovOp::verify`). Routing such an operand to `Vec` instead would cost a GM -> UB -> L1 -> L0 chain and, worse, put a cube-only operand on the vector core, which `ExpandMixedKernel` then reads as a mixed kernel and splits across AIC/AIV.

`Acc` is handled separately, because **no target moves anything into `Acc`** on any path — only the matrix unit writes L0C. A tile that must be an accumulator therefore has to be *created* there; Phase 2 cannot bridge to it. `OpRegistry::Create` rejects an operand whose *explicit* space cannot reach an `Acc` constraint, but an *unset* producer still arrives here carrying that demand, and the clamp above must not swallow it.

So a demand for a space with no inbound move edge (`IsTileMoveEverPossibleInto`) splits on the producer's registered execution-memory-access evidence:

| Producer | Evidence | Outcome |
| -------- | -------- | ------- |
| `tile.create` | `no_execution_memory_access()` | Honour the demand directly — the allocation is born in `Acc` |
| `tile.load` | `functional_execution_memory_access()` | User-facing error — MTE2 fills {`Vec`, `Mat`} and never L0C, so no placement satisfies it |

Keying this on the registry's evidence rather than an op-name list means a producer added later is classified by what it actually does. Falling through to the `Vec` fallback instead would let Phase 2 "repair" the mismatch with a `tile.move` into `Acc` that no target implements — invalid IR that survives to the backend and aborts there, naming neither the tile nor the line that created it. Phase 2 asserts on that case (`INTERNAL_CHECK_SPAN`) so it can never be emitted silently again.

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

### Phase 4 — Loop-invariant Mat residency (`loop_invariant_mat_residency`)

After all spaces are explicit, a focused internal transform recognizes an invariant prefix of the form `tile.load(GM → Mat) → tile.transpose_view* → tile.move/tile.extract(Mat → Left/Right)`. For an exact single-use chain, it moves the entire invariant prefix to the loop preheader. It also recognizes a compiler-generated Mat panel whose complete read-only use graph fans through `transpose_view` and one or more `move` / `extract` operations into matching matmul operand slots. In that case only the whole-panel GM→Mat load moves; K-dependent Left/Right staging remains in its original loop or pipeline. This keeps the optimization specific to stationary matmul operands rather than turning it into general tile LICM. A stationary tensor-level operand is therefore loaded from GM once while its loop-dependent peer continues to stream normally.

This is a conservative first subset of the broader residency behavior requested by issue #2077, not a general tensor-level residency contract. A direct or externally entered InCore function has no caller evidence and therefore declines. Distinct external tensor parameters also decline: PyPTO has no runtime `noalias` contract that makes their backing allocations disjoint. Positive caller provenance is currently limited to storage allocated by `tensor.create` inside root orchestration IR. Extending coverage to external operands requires an enforced no-alias contract or a checked fallback; it is not assumed by this transform. `AutoTileMatmulL0` fanout is supported without moving its K-dependent L0 extracts: panel residency and optional Mat→L0 prefix motion are analyzed independently.

Eligibility begins with private compiler provenance. `ConvertTensorToTileOps` marks every generated `GM → Mat` bridge load; the marker survives printing, flattening, and L0 auto-tiling until this phase consumes it. Phase 4 then proves that a marked load participates in either the exact stationary prefix or the read-only matmul panel fanout described above. A user-authored `tile.load(..., target_memory=Mat)` is not marked and is never hoisted by this optimization. This boundary keeps explicit tile programs under user control.

The initial legality rules are intentionally strict:

- the loop is `Sequential`, has constant bounds, positive step, and at least one iteration;
- every moved assignment is an unconditional top-level loop-body statement;
- the GM source is a direct `ParamDirection::In` tensor parameter on a compiler-marked Mat bridge load;
- the InCore function has at least one direct `Call` site in a root orchestration function (an orchestration function with no in-program caller), and every call site to that InCore function is such a direct root-orchestration `Call`; a `Submit` site always poisons the candidate because asynchronous submission is not positive alias evidence;
- at every such call site, the candidate `Tensor In` actual resolves to a compiler-owned allocation created by `tensor.create`; plain aliases and `tensor.slice` / `tensor.assemble` / `tensor.view` aliases are canonicalized to that storage root, every writable `Tensor Out` / `Tensor InOut` root is known, and none aliases the candidate root; the InCore function also does not write that root locally, while unrelated scalars and peer read-only `Tensor In` roots do not participate in this filter;
- offsets, shapes, and the complete moved dependency prefix are loop invariant;
- any function call, task submission, cross-core operation, synchronization, cache-maintenance, or unknown builtin in the loop header (bounds or loop-carried initializers) or body subtree declines residency because preheader motion could cross an unknown or hidden ordering effect between iterations; other direct control-flow or effect statements before the candidate close the hoistable prefix;
- every value in an exact movable prefix has exactly its expected single syntactic use; a retained-panel load may instead have multiple fully accounted direct read paths, but every path must consist only of Mat `transpose_view` aliases followed by Left/Right `move` / `extract`, and every resulting L0 value may be used only in its matching operand slot of a matmul-family call; plain SSA aliases, `Submit` arguments, nested expressions, loop initializers, yields, returns, and unsupported or additional consumers decline the candidate;
- no moved result is loop-carried or yielded;
- all allocation-owning `Mat`, `Left`, and `Right` tiles in the function have static sizes, with their allocator-aligned whole-function upper bound no larger than the backend capacity; and
- the function has no explicit reserved-buffer region whose capacity contribution is not represented as a tile allocation.

`InOut` / `Out` sources, external input allocations, manual tile loads, direct or externally-entered InCore functions, `Submit` sites, calls through InCore wrappers or called orchestration helpers, unknown candidate or writable call-site roots, candidate/write aliases, extra syntactic uses, conditional loads, dynamic or zero-trip loops, capacity-unknown cases, yielded or loop-carried results, and loop-dependent extracts decline without changing the IR. One unsafe or non-root call site invalidates the candidate even when other calls are safe. Wrapper evidence is deliberately not propagated in this initial implementation: syntactically distinct wrapper parameters may alias at the wrapper's own caller. The capacity test counts allocation owners rather than zero-copy views or SSA aliases, uses the same byte sizing and address alignment as `InitMemRef` / `AllocateMemoryAddr`, and includes allocations already live outside the loop. A memory space containing allocations that later pipeline lowering may replicate is also declined unless the moved prefix stays in an unaffected space. This whole-function bound is deliberately stronger than either planner's lifetime reuse, so a residency rewrite cannot introduce a later capacity failure under the PyPTO or PTOAS planner. Nested loops are processed independently; a chain moves only to its immediate lexical preheader in one pass invocation. This phase does not globally remap parameters and does not move K-dependent L0 extracts out of `AutoTileMatmulL0` pipeline loops.

#### Residency example

For a tensor program whose root orchestration function creates fresh LHS storage before calling the InCore kernel, the stationary LHS bridge moves before the user loop, while the N-dependent RHS remains streamed:

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

The internal provenance attribute and root orchestration call are omitted above for readability. The caller creates `fresh_lhs = pl.create_tensor([16, 128], dtype=pl.BF16)` and passes it as `lhs`; the compiler can therefore prove that its allocation is distinct from the external writable `out`. Merely passing distinct external `lhs` and `out` parameters is not sufficient. The peer read-only `rhs` root is irrelevant to the write-alias filter. Without the required trusted storage provenance, the original loop-local placement is retained.

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
| Produced | `SSAForm`, `TileMemoryInferred`, `NormalizedStmtStructure`, `AivSplitValid`, `AccToGmStoreValid` |
| Invalidated | `AivSplitValid` |

The `TileMemoryInferred` property is the contract this pass establishes. Downstream passes (notably `ExpandMixedKernel` and `InitMemRef`) rely on it, and the matching property verifier guards regressions.

`AccToGmStoreValid` becomes decidable only here: whether a `tile.store` narrows from Acc into GM depends on the memory space this pass resolves. `AivSplitValid` is invalidated and re-produced for the same reason — this is the last verification point at which an AIV-split boundary whose operand space was still unresolved at `ConvertTensorToTileOps` becomes observable, before `LowerAutoVectorSplit` erases the region node.

## Scope

| Function kind | Action |
| ------------- | ------ |
| `InCore` (incl. `AIC`, `AIV`) | Transformed |
| `Orchestration` | Unchanged |
| `Opaque` | Unchanged |

Tile *parameters* are handled by function type, at the start of Phase 1:

| Function type | Tile parameter | Why |
| ------------- | -------------- | --- |
| `InCore` | Rejected (`INTERNAL_CHECK`) | An InCore kernel is entered from orchestration, which speaks tensors; a tile parameter means an earlier pass produced a malformed signature |
| `AIC` / `AIV` | Accepted, and **seeds** the analysis | These are sub-workers entered from a mixed kernel, so a tile parameter is the ordinary cross-core handoff |

A parameter's space is part of the signature — the caller decides where the tile lives — so this pass never infers it. An `AIC`/`AIV` tile parameter that omits its space is a user error, reported with the annotation to add.
