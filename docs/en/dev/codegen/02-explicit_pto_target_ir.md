# Explicit PTO Target IR

> **Status:** `MaterializePTOTileHandles` and `LowerTileToPTOIR` are enabled at
> the end of the `Default` and `DebugTileOptimization` pipelines. Supported
> functions are printed by `PTOIRPrinter`. Functions that contain a Tile family
> not yet covered by the target IR are marked
> `pto.target_lowering_deferred` and remain wholly on the legacy path; a
> function is never partially bufferized. This is a diagnostic default
> activation and is not yet production complete.

## Scope

This design applies only to PTO InCore code generation. Orchestration C++
codegen has different object-identity and lifetime requirements and is not
lowered to this target IR.

The immediate goal is to move these decisions out of PTO codegen:

- which PTO tile buffer receives each logical Tile result;
- where each `pto.alloc_tile` handle is defined;
- which handles are inputs and outputs of a target operation;
- which tensor partition offsets and extents a transfer uses.

The printer should consume those decisions mechanically. It should not recover
them from `AssignStmt`, `TileType::memref_`, planner-specific alias maps, or the
currently emitted SSA name.

## Value Model

The experimental path does not replace SSA or introduce a separate non-SSA IR.
It distinguishes logical values from mutable target storage:

| Concept | IR representation | Semantics |
| ------- | ----------------- | --------- |
| Logical tile value | `TileType` | Value-oriented Tile IR used before target lowering |
| Target buffer handle | `PTOTileBufType` | SSA handle naming a mutable PTO tile buffer |
| Physical storage metadata | `pto.alloc_tile` operands | Optional address plus valid row/column extents |
| Target operation | `pto.*` call in `EvalStmt` | Reads/writes explicit handles according to its operand schema |

For example:

```text
%lhs = pto.alloc_tile(...)
%rhs = pto.alloc_tile(...)
%out = pto.alloc_tile(...)
pto.tadd(%lhs, %rhs, %out)
```

`%lhs`, `%rhs`, and `%out` each have one SSA definition. `pto.tadd` returns no
value because it writes through `%out`; its registered operand roles and memory
effects declare `%lhs/%rhs` as reads and `%out` as a write. Mutable buffer
semantics therefore do not require destroying handle SSA.

`PTOTileBufType` is internal target-lowering state. It is not a user-facing DSL
type and is not interchangeable with `TileType`.

## Staged Lowering

The implemented path has two IR passes followed by a target printer:

```text
Logical Tile SSA
  |
  | MaterializePTOTileHandles (Step 3)
  v
Logical Tile SSA + explicit PTO handle plan
  |
  | LowerTileToPTOIR (Step 4)
  v
Destination-passing PTO target IR
  |
  | PTOIRPrinter
  v
PTO-ISA MLIR
```

### Step 3: Materialize PTO Tile Handles

`MaterializePTOTileHandles` keeps the logical Tile program intact while adding
a verified logical-value-to-buffer plan:

- each supported Tile producer gets a dominating `pto.alloc_tile` definition;
- each logical Tile operand is mapped to an ordered input-handle list;
- each logical Tile result is mapped to one output handle;
- transitional call attrs carry the input/output mapping;
- `PTOHandlesMaterialized` verifies dominance, uniqueness, exact mapping, and
  structured `ForStmt`/`IfStmt` region mappings;
- `tile.slice` creates an explicit non-allocating `pto.subview` handle;
- loop-carried and branch-merged Tile values are mapped to explicit shared
  handles, with `pto.tmov` inserted only when storage is not already shared.

Allocation operands depend on the selected memory planner:

| Planner | `pto.alloc_tile` operands |
| ------- | ------------------------- |
| `PYPTO` | `(byte_offset: i64, valid_row: index, valid_col: index)` |
| `PTOAS` | `(valid_row: index, valid_col: index)` |

The attrs are temporary lowering state. They are consumed and removed by Step
4; `PTOIRPrinter` never reads them.

### Step 4: Lower Logical Tile Operations

`LowerTileToPTOIR` rewrites the supported logical operations to
destination-passing target calls and removes their `TileType` definitions.

Before:

```text
a = tile.load(input_a, [0, 0])
b = tile.load(input_b, [0, 0])
root = tile.sqrt(a)
added = tile.add(root, b)
result = tile.mul(added, b)
output_result = tile.store(result, [0, 0], output)
return output_result
```

After:

```text
a_buf = pto.alloc_tile(...)
pto.tload(input_a, [0, 0], [16, 16], a_buf)

b_buf = pto.alloc_tile(...)
pto.tload(input_b, [0, 0], [16, 16], b_buf)

root_buf = pto.alloc_tile(...)
pto.tsqrt(a_buf, root_buf)

added_buf = pto.alloc_tile(...)
pto.tadd(root_buf, b_buf, added_buf)

result_buf = pto.alloc_tile(...)
pto.tmul(added_buf, b_buf, result_buf)

pto.tstore(result_buf, [0, 0], [16, 16], output)
output_result = output
return output_result
```

The final tensor alias preserves the logical `tile.store` tensor result without
reintroducing a Tile result. Transfer offsets and valid extents are ordinary IR
operands, not printer-side reconstruction.

## Target Operation Contracts

Every target op has a registered `PTOOpSpec` describing operand roles, memory
effects, type constraints, and result placement:

| Operation | Input operands | Metadata | Output | IR result |
| --------- | -------------- | -------- | ------ | --------- |
| `pto.alloc_tile` | none | optional address, valid rows/cols | allocated handle | `PTOTileBufType` |
| `pto.tload` | tensor | offsets, extents | tile buffer | none |
| `pto.tstore` | tile buffer | offsets, extents | tensor | none |
| `pto.tsqrt` | one tile buffer | none | tile buffer | none |
| unary / binary / tile-scalar `pto.t*` | one or two tile buffers, optional scalar | none | tile buffer | none |
| `pto.trowsum` | input and temporary tile buffers | none | tile buffer | none |
| `pto.subview` | source tile buffer | shape, offset, valid shape | view handle | `PTOTileBufType` |

Allocating ops must be the value of an `AssignStmt`. Destination-passing ops
must have `UnknownType` and appear directly in an `EvalStmt`.

## Verification Contracts

Two IR properties define the stage boundaries:

| Property | Required invariant |
| -------- | ------------------ |
| `PTOHandlesMaterialized` | Logical Tile IR remains, but every supported producer/operand has an exact, dominance-valid handle mapping |
| `PTOBufferized` | InCore target IR contains no logical Tile values and all target buffer uses are explicit and dominance-valid |

The `PTOBufferized` verifier checks that:

- no logical Tile parameter, assignment, return, or non-PTO Tile call remains;
- each tile-buffer handle is defined by a registered handle op
  (`pto.alloc_tile`/`pto.subview`) or a function parameter;
- every handle use is dominated by its definition;
- target result placement and operand types match `PTOOpSpec`;
- `tload`/`tstore` offset and extent ranks are non-empty, equal, and match the
  tensor rank;
- handle dominance is checked recursively through `ForStmt` and `IfStmt`;
- PTO target calls do not escape into Orchestration functions.

### Default Pipeline Integration

The pass-property contract is planner-independent. Neither target-lowering pass
unconditionally requires `AllocatedMemoryAddr`, because the PTOAS planner skips
`AllocateMemoryAddr` and emits address-free allocations. Step 3 instead checks
for a constant MemRef byte offset only when the active planner is PYPTO.

`PTOHandlesMaterialized` and `PTOBufferized` are part of the default verified
property set, so normal pipeline execution validates both transition boundaries.
Step 4 invalidates the logical Tile/MemRef properties after their information
has been transferred to explicit target operands.

The verifier permits a `PTOTileBufType` function parameter as a dominating
handle definition, but the current experimental printer accepts only Tensor and
Scalar parameters. Step 3 never creates buffer parameters, so the implemented
slice is internally consistent; the verifier/printer contract must be aligned
before buffer handles may cross function boundaries.

## Printer Boundary

`PTOCodegen::Generate` selects the printer independently for each function.
A function containing destination-passing PTO calls is emitted by
`PTOIRPrinter`; a function carrying `pto.target_lowering_deferred` remains
wholly on the legacy logical-Tile path. A multi-function program may therefore
contain both fully lowered and deferred functions without sending the deferred
functions through the target printer.

For target IR, `PTOIRPrinter` may perform formatting and local expansion only:

| Printer responsibility | Must already be explicit in IR |
| ---------------------- | ------------------------------ |
| MLIR names and type spelling | allocation placement and optional address |
| constants and tensor-view formatting | input/output buffer identity |
| `partition_view` syntax from transfer operands | offsets and valid extents |
| `ins(...)` / `outs(...)` rendering from `PTOOpSpec` | alias and destination decisions |

It must not read `TileType`, recover a destination from an enclosing assignment,
or maintain planner-specific result-buffer inference state.

## Supported Surface

The current implementation supports:

- static-shape rank-N Tensor views and transfers with physical rank-2 Tile
  buffers;
- static or dynamic rank-2 valid shapes;
- `tile.load`/`store`, `create`/`full`, `slice`, alloc-backed `reshape`, basic
  unary/binary/tile-scalar elementwise operations, move/fillpad, and row sum;
- explicit SPMD block/subblock parameters;
- structured `ForStmt` and `IfStmt`, including scalar carries/results and Tile
  carry/phi handles;
- scalar tensor read/write inside structured regions;
- PYPTO explicit-address and PTOAS address-free allocations.

For PTOAS, branch-local producers write the shared phi/carry handle directly;
pass-through values and metadata-only views use an explicit `pto.tmov` when a
real copy is required. PYPTO retains its address-based alias plan. Static
`pto.subview` valid extents are represented independently per dimension in the
result `PTOTileBufType`.

Cross-core pipe operations, dynamic-shape Tensor parameters, general reductions
such as `tile.sum`, `WhileStmt`, and the remaining Tile families are not
target-lowered yet. Such a function is marked `pto.target_lowering_deferred`
before any handle is inserted and uses the legacy path as one indivisible unit.

## Completion Plan

### Phase A: Default Diagnostic Activation

Completed:

1. make the property contract planner-independent while retaining the PYPTO-only
   constant-address check;
2. run Step 3 and Step 4 at the end of both maintained Tile pipelines;
3. verify both transition properties during normal pipeline execution;
4. dispatch target IR to `PTOIRPrinter` and compile it with PTOAS in both PYPTO
   level-3 and PTOAS level-2 modes; and
5. mark an unsupported whole InCore function as deferred before mutation,
   without mixing target and logical Tile operations.

### Phase B: Straight-Line Operation Coverage

Add operation families only after defining their operand roles, memory effects,
result placement, lowering, verifier rules, shadow comparison, and `ptoas`
coverage.

### Phase C: Control Flow and Aliasing

`IfStmt` and `ForStmt` now carry an explicit Step-3 handle plan. Scalar values
remain SSA `scf` results/carries; Tensor values remain reference aliases; Tile
values use shared handles or explicit `pto.tmov`. `WhileStmt` and broader
region/block transport remain to be implemented and validated under both
planners.

### Phase D: Production-Readiness Gate

Keep the path default-enabled for coverage discovery, but do not treat it as
production-ready until representative real kernels and system tests pass,
shadow differences are explained, and the corresponding destination and handle
recovery code can be removed from legacy `PTOCodegen`.

## Validation Evidence

The implemented static slice has been validated with:

- ordered new-vs-legacy PTO operation comparison;
- `PTOBufferized` positive and negative verifier tests;
- binary serialization round trips;
- PYPTO/level-3 MLIR accepted by `ptoas`;
- PTOAS/level-2 MLIR accepted by `ptoas`;
- default-pipeline PYPTO and PTOAS compilation of a static elementwise kernel
  through real `ptoas`;
- focused Step-3/Step-4 and pass-manager tests; and
- a default-enabled codegen baseline with an explicit whole-function deferred
  boundary for unsupported surfaces.

At initial default activation, `tests/ut/codegen/test_pto_codegen.py` passed
61/86 tests. After rank-N transfers, dynamic valid shapes, views/slices, basic
operation families, SPMD parameters, and structured `ForStmt`/`IfStmt` support,
the same isolated worktree suite passes 86/86. This is unit-test coverage, not
a production-readiness claim. After fixing PTOAS control-flow output sharing,
per-dimension subview valid types, scalar `min`/`max`, and the private SPMD
parameter boundary, the complete `tests/ut/codegen/` suite passes 573/573 and
the focused target-op/pass/property/verifier suite passes 82/82. Cross-core
target IR, `WhileStmt`, broader op coverage, and representative system/PTOAS
validation remain open.

## Implementation Map

| Component | Location |
| --------- | -------- |
| Target buffer type and transitional attrs | `include/pypto/ir/type.h`, `include/pypto/ir/expr.h` |
| Target op schemas | `src/ir/op/pto_ops/` |
| Step 3 | `src/ir/transforms/materialize_pto_tile_handles_pass.cpp` |
| Step 4 | `src/ir/transforms/lower_tile_to_pto_ir_pass.cpp` |
| Property verifiers | `src/ir/verifier/verify_pto_ir.cpp` |
| Mechanical printer | `src/codegen/pto/pto_ir_printer.cpp` |
| Legacy/new-path dispatch | `src/codegen/pto/pto_codegen.cpp` |

See also [PTO Codegen](00-pto_codegen.md),
[MaterializePTOTileHandles](../passes/43-materialize_pto_tile_handles.md),
[LowerTileToPTOIR](../passes/44-lower_tile_to_pto_ir.md),
[#1956](https://github.com/hw-native-sys/pypto/issues/1956), and
[#2032](https://github.com/hw-native-sys/pypto/issues/2032).
