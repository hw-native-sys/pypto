# Torch Code Generation (Torch Codegen)

## Overview

`torch_codegen` lowers PyPTO IR into an executable Python/PyTorch script that can be run with `exec()` for debugging and numerical validation.

Unlike production codegen (PTO/Orchestration), `torch_codegen` is designed to:

- quickly reproduce IR semantics,
- expose intermediate behavior in Python,
- provide executable references for pass debugging and system tests.

**Source location:** `python/pypto/debug/torch_codegen.py`

**Public entry API:**

```python
torch_codegen(node: _ir.Program | _ir.Function, check_shapes: bool = False) -> str
```

## Goals and Boundaries

### Goals

- Keep IR-to-Python expression/statement mapping readable.
- Cover common tensor/tile operations.
- Simulate concurrent cross-core `tpush/tpop` behavior.
- Provide actionable diagnostics for suspicious inputs (timeouts, invalid split dimensions, etc.).

### Non-goals

- No performance optimization.
- No cycle-accurate Ascend hardware timing/memory simulation.
- Not a production backend for training/inference.

## High-Level Architecture

Generated code is assembled from three parts:

1. **Runtime preamble (`_PREAMBLE`)**
   Provides helpers, tile/tensor boundary handling, cross-core runtime, and mixed-kernel scheduler.
2. **Group metadata injection (`_GROUP_META.update(...)`)**
   Injected only when input is a `Program` and Group/AIC/AIV pairs are detected.
3. **Function bodies**
   Emitted by `TorchCodegen(IRVisitor)` function-by-function and statement-by-statement.

Data flow:

`Program/Function IR -> (optional _build_group_meta) -> TorchCodegen visitor emission -> final script string`

## Expression and Statement Emission Model

### Expression layer

- `_OP_MAP` maps `op_name` to `OpHandler(args, kwargs) -> str`.
- `_visit_expr_str()` forces nested `Call` nodes through Python-side `visit_call`, avoiding nested-call result loss in some C++ visitor paths.
- Binary/unary IR nodes are emitted through `_BINARY_OP_STR` plus generic visitor adapters.

### Statement layer

- `AssignStmt`: `var = expr`
- `EvalStmt`: emit expression as-is (for side-effect calls)
- `ReturnStmt`: supports single and tuple returns
- `ScopeStmt`: transparent passthrough
- `ForStmt`/`WhileStmt`: lower SSA `iter_args + yield` into mutable variable updates
- `IfStmt`: branch-local `yield` writes back to `return_vars`

## Naming and Function Isolation

`_unique_name()` normalizes variables by:

- replacing non-identifier characters with `_`,
- collapsing repeated underscores,
- avoiding Python keywords and digit-leading names.

`visit_function()` resets `_var_names/_name_counter/_yield_targets` per function to prevent cross-function name pollution when object IDs are reused.

## Operation Mapping System (`_OP_MAP`)

Mappings are registered by category:

- tensor/tile elementwise, broadcast, reduction, logic, bitwise
- `matmul/matmul_acc` and tile variants
- `create/full/cast/slice/read/write/assemble/fillpad`
- cross-core pipe operations
- `system.*` operations (no-op in debug codegen)

### Cross-core operation mappings

- `tile.tpush_to_aiv` -> `_cross_core_rt.push_to_aiv(tile, split)`
- `tile.tpush_to_aic` -> `_cross_core_rt.push_to_aic(tile, split)`
- `tile.tpop_from_aic` -> `_cross_core_rt.pop_from_aic(split)`
- `tile.tpop_from_aiv` -> `_cross_core_rt.pop_from_aiv(split)`
- `tile.get_subblock_idx` -> `_get_subblock_idx()`

`split` is normalized by `_split_mode_to_int()`:

- `0`: NONE
- `1`: UP_DOWN
- `2`: LEFT_RIGHT
- invalid/unparsable values: raise `ValueError` (fail fast)

## Tile/Tensor Boundary and Valid-Region Semantics

Helpers in `_PREAMBLE` define boundary behavior:

- `_tile_load`: zero-pad to requested shape on OOB and attach valid region
- `_tile_store`: store only the valid region
- `_tensor_slice`: materialize requested shape even when slicing out-of-bounds
- `_fillpad`: pad invalid area with `zero/min/max`
- `_assemble`: write source valid region into target at offsets

Valid region is propagated via dynamic attributes:

- `_pypto_valid_shape`
- `_pypto_full_shape`

Cross-core queueing/split/merge paths preserve these attributes as well, so
boundary-tile valid regions remain consistent through `tpush/tpop`.

## Cross-Core Runtime Design

### Structure

`_CrossCoreRuntime` uses `threading.Condition` to implement blocking queue semantics.
It maintains:

- normal channels:
  - to AIV: `_to_aiv` and `_to_aiv_split[split][lane]`
  - to AIC: `_to_aic` and `_to_aic_split[split][lane]`
- no-split dual-dispatch channels:
  - to AIV: `_to_aiv_dual_nosplit[lane]`
  - to AIC: `_to_aic_dual_nosplit[lane]`

It also provides:

- `reset(no_split_dual_aiv_dispatch=False)`: clear channels and configure no-split dual-dispatch mode before each mixed-group call
- `snapshot()`: queue-depth snapshot for timeout diagnostics, including no-split dual-dispatch mode and dual-nosplit queue depths

### Push/Pop semantics

### `push_to_aiv(tile, split)` / `pop_from_aic(split)`

- `split=0` and `no_split_dual_aiv_dispatch=False`: single queue
- `split=0` and `no_split_dual_aiv_dispatch=True`:
  - `push_to_aiv` broadcasts one tile copy to both lane queues
  - `pop_from_aic` consumes from the current lane queue (`lane in {0,1}`)
- `split=1/2`: push splits tile into lane0/lane1; pop consumes by current lane

### `push_to_aic(tile, split)` / `pop_from_aiv(split)`

- `split=0` and `no_split_dual_aiv_dispatch=False`: single queue
- `split=0` and `no_split_dual_aiv_dispatch=True`:
  - `push_to_aic` enqueues by current lane (`lane in {0,1}`)
  - `pop_from_aiv` waits for both lane queues, then consumes a pair and returns lane0 payload
- `split=1/2`: push enqueues by current lane; pop waits until both lanes are ready, then merges

Split dimension semantics:

- `split=1` (UP_DOWN): split/merge on `dim=0`
- `split=2` (LEFT_RIGHT): split/merge on `dim=1`

### Synchronization and timeouts

- per-pop timeout: `_PIPE_WAIT_TIMEOUT_SEC` (default 10s)
- mixed-kernel group timeout: `_MIXED_KERNEL_TIMEOUT_SEC` (default 30s)
- on mixed-kernel timeout/failure, runtime signals cancellation and notifies
  all waiters to avoid stale blocked threads polluting subsequent runs

Timeout errors include:

- operation name, split, lane
- alive thread names (group timeout)
- current pipe snapshot (queue depths)

## Group Mixed-Kernel Concurrent Scheduler

### Metadata construction (`_build_group_meta`)

For each `FunctionType.Group` function in a `Program`, codegen pairs by naming convention:

- `<group>_aic`
- `<group>_aiv`

Metadata fields:

- `aic` / `aiv`: callee names
- `split`: Group split first; fallback to AIV split when Group split is 0
- `dual_aiv_dispatch`: read from AIV attrs; for `split==0`, this forces dual AIV-lane dispatch in debug runtime

### Scheduler entry

In `visit_call()`, for `GlobalVar` calls:

- if name exists in `_group_meta`, emit `_run_group_call(group_name, *args)`
- otherwise emit a normal direct function call

`_run_group_call()` dispatches to `_run_mixed_kernels()` when metadata exists.

### Thread model

`_run_mixed_kernels(group_name, meta, *args)` behavior:

- exactly 1 AIC thread
- AIV thread count:
  - `split in (1,2)` -> 2 lanes
  - `split == 0 and dual_aiv_dispatch == True` -> 2 lanes
  - `split == 0 and dual_aiv_dispatch == False` -> 1 lane
- each thread writes thread-local `subblock_idx` (AIC uses 0, AIV uses lane id)
- runtime mode switch:
  - scheduler calls `reset(no_split_dual_aiv_dispatch=(split == 0 and dual_aiv_dispatch))`
  - no-split dual-lane pipe semantics are enabled only when required by group metadata
- return contract: only `aiv lane0` return value is propagated as Group return;
  if other lanes/roles produce non-`None` returns, runtime raises a contract violation error

## Shape/Type Checks (`check_shapes`)

When `check_shapes=True`:

- for function parameters: dtype check only, shape check disabled
  - rationale: InCore parameters may be boundary tiles with partial data
- for assignment targets: tensor type and dtype checks enabled; shape checks follow static/dynamic strategy

Dynamic-dimension strategy:

- if dimensions are not all static `ConstInt`, codegen checks `ndim` plus each static dimension index.

## Error Handling and Observability

Main error classes:

- `TypeError`: invalid entry node type
- `ValueError`: unsupported op, invalid split/lane, non-even split dimension
- `RuntimeError`: pipe wait timeout, mixed-kernel thread failure/timeout

For concurrency failures, codegen reports the first captured thread traceback to improve localization by function and lane.

## Current Limitations

- `system.*` ops are emitted as no-ops in debug mode
- Group pairing relies on `<group>_aic/_aiv` naming convention
- `split=1/2` requires the split dimension to be divisible by 2
- Group return is fixed to AIV lane0 output
- Runtime is a Python semantic simulator, not equivalent to hardware pipeline timing

## Recommended Test Coverage

Recommended coverage dimensions:

- all split modes: `NONE/UP_DOWN/LEFT_RIGHT`
- all communication directions: `V->C`, `C->V`, bidirectional `V<->C`
- lane semantics via `tile.get_subblock_idx`
- no-split + `dual_aiv_dispatch` lane behavior (`lane0` + `lane1`)
- Group call rewrite path (`_run_group_call`)
- timeout/error paths (for example, intentionally unpaired push/pop)

References:

- `tests/ut/debug/test_torch_codegen.py`
- `tests/st/codegen/test_torch_codegen_cross_core.py`
- `tests/st/codegen/test_torch_codegen_qwen3_decode_scope3_mixed.py`

## Relation to Other Codegen Docs

- This document: Python debug codegen (`torch_codegen`)
- [00-pto_codegen.md](00-pto_codegen.md): PTO kernel codegen
- [01-orchestration_codegen.md](01-orchestration_codegen.md): orchestration-side C++ codegen

They are complementary: `torch_codegen` provides executable semantic reference, while PTO/Orchestration target production generation.
