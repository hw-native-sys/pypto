# InsertSync Pass

Analyzes data dependencies and inserts synchronization operations for correct multi-pipeline execution.

## Overview

This pass is the most complex transformation pass in PyPTO. It analyzes data dependencies across hardware pipelines and inserts synchronization operations (sync_src, sync_dst, bar_v, bar_m) to ensure correct execution.

**Key responsibilities**:

- Analyze inter-pipeline data dependencies
- Insert sync_src/sync_dst for producer-consumer synchronization
- Insert barriers (bar_v, bar_m) for global synchronization
- Manage event IDs and pipeline masks

**When to use**: Run after InitMemRef and BasicMemoryReuse, before code generation. Required for correct multi-pipeline hardware execution.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::InsertSync()` | `passes.insert_sync()` | Function-level |

**Factory function**:

```cpp
Pass InsertSync();
```

**Python usage**:

```python
from pypto.pypto_core import passes

sync_pass = passes.insert_sync()
program_with_sync = sync_pass(program)
```

## Algorithm

1. **Phase 1 — Dependency Collection**: Walk the IR tree, determine pipeline assignment for each operation (using backend pipe info), and collect producer-consumer sync pairs (both cross-pipeline and same-pipeline). For loops, unroll one extra iteration to detect cross-iteration dependencies.
2. **Phase 2 — Scope Adjustment**: Adjust sync pairs that cross scope boundaries (IfStmt/ForStmt):
   - Cross-iteration (wait ≤ set in same for body, including same-pipe bars): move sync_dst/bar to end of iteration
   - When wait is in a deeper scope than set: move sync_dst to end of set's OpStmts
   - When set is in a deeper scope than wait: move sync_src to beginning of wait's OpStmts
3. **Phase 3 — Event ID Allocation**: Assign unique event IDs for sync operations, reusing IDs when possible
4. **Phase 4 — AST Construction**: Build the final IR with sync_src/sync_dst/barrier insertions

**Synchronization patterns**:

- **Producer-consumer**: sync_src (producer) → sync_dst (consumer)
- **Pipe barrier**: bar_v / bar_m
- **If branch scope**: when producer is in if branch and consumer is outside, sync_src is moved to beginning of consumer's OpStmts
- **For body to parent**: when producer is in for body and consumer is outside, sync_src is moved to beginning of consumer's OpStmts
- **Cross-iteration**: sync_dst/bar placed at end of iteration (before yield)
- **OpStmts merging**: sync operations merged into adjacent OpStmts (no standalone sync OpStmts)

## Example

### Cross-Pipeline Dependency (MTE2 → V → MTE3)

Load (MTE2) → compute (V) → store (MTE3), with sync_src/sync_dst inserted at each pipeline boundary.

**Before**:

```text
tile_a = load(input_a)              # MTE2
tile_b = load(input_b)              # MTE2
tile_c = add(tile_a, tile_b)        # V
store(tile_c, output)               # MTE3
```

**After**:

```text
tile_a = load(input_a)              # MTE2
tile_b = load(input_b)              # MTE2
sync_src(MTE2 -> V, event=0)
sync_dst(MTE2 -> V, event=0)
tile_c = add(tile_a, tile_b)        # V
sync_src(V -> MTE3, event=0)
sync_dst(V -> MTE3, event=0)
store(tile_c, output)               # MTE3
```

### Intra-Pipeline Dependency (V → V)

When consecutive V-pipe operations have a data dependency, a `bar_v` barrier is inserted instead of sync_src/sync_dst.

**Before**:

```text
t_c = add(t_a, t_b)                # V
t_d = add(t_c, t_a)                # V (depends on t_c)
```

**After**:

```text
t_c = add(t_a, t_b)                # V
bar_v                               # intra-pipe barrier
t_d = add(t_c, t_a)                # V
```

### CUBE Pipeline (MTE2 → MTE1 → M → MTE3)

Matrix multiply requires moving data through L1 (MTE1) to L0 (CUBE/M pipe), with sync at each boundary. Multiple event IDs are used when the same pipeline pair has multiple independent transfers.

**Before**:

```text
tile_a = load(input_a)              # MTE2 -> L1
tile_b = load(input_b)              # MTE2 -> L1
tile_a_cube = move(tile_a)          # MTE1 -> L0A
tile_b_cube = move(tile_b)          # MTE1 -> L0B
tile_c = matmul(tile_a_cube, tile_b_cube)  # CUBE (M pipe)
store(tile_c, output)               # MTE3
```

**After**:

```text
tile_a = load(input_a)              # MTE2
sync_src(MTE2 -> MTE1, event=0)
tile_b = load(input_b)              # MTE2
sync_src(MTE2 -> MTE1, event=1)
sync_dst(MTE2 -> MTE1, event=0)
tile_a_cube = move(tile_a)          # MTE1
sync_dst(MTE2 -> MTE1, event=1)
tile_b_cube = move(tile_b)          # MTE1
sync_src(MTE1 -> M, event=0)
sync_dst(MTE1 -> M, event=0)
tile_c = matmul(tile_a_cube, tile_b_cube)  # M
sync_src(M -> MTE3, event=0)
sync_dst(M -> MTE3, event=0)
store(tile_c, output)               # MTE3
```

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass InsertSync();
```

**Implementation**: `src/ir/transforms/insert_sync_pass.cpp`

- Uses backend pipe information (via globally configured backend)
- 4-phase pipeline: Collect → AdjustScopeCrossings → AssignEventIds → BuildAST
- Scope-aware: sync pairs never cross IfStmt/ForStmt boundaries
- Cross-iteration detection via loop unrolling

**Backend integration**:

```cpp
#include "pypto/backend/common/backend_config.h"
// Uses Backend::GetOpInfo() to determine operation pipelines
```

**Python binding**: `python/bindings/modules/passes.cpp`

```cpp
passes.def("insert_sync", &pass::InsertSync, "Insert synchronization operations");
```

**Tests**: `tests/ut/ir/transforms/test_insert_sync.py`

- Tests same-scope cross-pipeline dependency (MTE2→V→MTE3)
- Tests intra-pipe barrier insertion (V→V)
- Tests CUBE pipeline (MTE2→MTE1→M→MTE3)
- Tests IfStmt scope crossing (both branches, one branch, branch merge)
- Tests ForStmt scope crossing (load before for, compute inside)
- Tests cross-iteration dependencies (V→MTE2, MTE3→MTE2)
- Tests combined for+if patterns

## Backend Dependency

This pass requires a configured backend to obtain pipeline information:

```python
from pypto import backend

# Set backend before running InsertSync
backend.set_backend(backend.Ascend910B())
program_with_sync = passes.insert_sync()(program)
```
